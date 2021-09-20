/* NET-MC.C - Interface between neural network and Markov chain modules. */

/* Copyright (c) 1995-2021 by Radford M. Neal 
 *
 * Permission is granted for anyone to copy, use, modify, or distribute this
 * program and accompanying programs and documents for any purpose, provided 
 * this copyright notice is retained and prominently displayed, along with
 * a note saying that the original programs are available from Radford Neal's
 * web page, and note is made of any changes made to the programs.  The
 * programs and documents are distributed without any warranty, express or
 * implied.  As the programs were written for research purposes only, they have
 * not been tested to the degree that would be advisable in any important
 * application.  All use of these programs is entirely at the user's own risk.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "cuda-use.h"

#include "misc.h"
#include "rand.h"
#include "log.h"
#include "mc.h"
#include "data.h"
#include "prior.h"
#include "model.h"
#include "net.h"
#include "net-data.h"

#include "intrinsics-use.h"


/* SETTING TO DISABLE SIMULATION OF "TYPICAL" SQUARED VALUES. */

#define TYPICAL_VALUES_ALL_ONE 0  /* Set to 1 to disable simulation of values,
                                     thereby reverting to the old heuristic */


/* CUDA SETTINGS. */

#if __CUDACC__

#define MAX_BLKSIZE 128		/* Limit on # of threads in a block, to avoid
				   exceeding the per-block register use limit
				   (max 255 reg/thread, min 32K reg/block) */

#define DEFAULT_PERTHRD 2 	/* Defaults, if not set by CUDA_SIZES  */
#define DEFAULT_BLKSIZE 64	/*   environment variable, which has   */
#define DEFAULT_NUMBLKS	48	/*   the form PERTHRD:BLKSIZE:NUMBLKS  */

static int perthrd = DEFAULT_PERTHRD;	/* Number of cases for a CUDA thread */
static int blksize = DEFAULT_BLKSIZE;	/* Number of threads per block */
static int numblks = DEFAULT_NUMBLKS;	/* Number of blocks per kernel */

#endif


/* FUNCTION TO SQUARE ITS ARGUMENT. */

static inline double sq (double x) { return x*x; }


/* SHOULD A CHEAP ENERGY FUNCTION BE USED?  If set to 0, the full energy
   function is used, equal to minus the log of the probability of the 
   training data, given the current weights and noise hyperparameters.
   This is necessary if marginal likelihoods are to be found using Annealed
   Importance Sampling.  If set to 1, the energy omits constant terms.
   If set to 2, the energy omits terms involving the noise hyperparameters,
   which is OK for sampling weights with hybrid Monte Carlo, etc., but does
   not work when tempering or annealing schemes are used. */

#define Cheap_energy 0		/* Normally set to 0 */


/* NETWORK VARIABLES. */

static int initialize_done = 0;	/* Has this all been set up? */

static net_arch *arch;		/* Network architecture */
static net_flags *flgs;		/* Network flags, null if none */
static net_priors *priors;	/* Network priors */

static model_specification *model; /* Data model */
static model_survival *surv;	/* Hazard type for survival model */

static net_sigmas sigmas;	/* Hyperparameters for network, auxiliary state
				   for Monte Carlo.  Includes noise std. dev. */

static net_params params;	/* Pointers to parameters, which are position
				   coordinates for dynamical Monte Carlo */

static net_values *deriv;	/* Derivatives for training cases */

static int approx_count;	/* Number of entries in approx-file, 0 if none*/

static int *approx_case; 	/* Data on how approximations are to be done  */
static int *approx_times;	/*   as read from approx_file                 */

static double *quadratic_approx;/* Quadratic approximation to log likelihood  */

static net_params stepsizes;	/* Pointers to stepsizes */
static net_values seconds;	/* Second derivatives */
static double *train_sumsq;	/* Sums of squared training input values */
static net_values typical;	/* Typical squared values for hidden units */

static net_params grad;		/* Pointers to gradient for network parameters*/

/* Values used or computed by threads, in managed or constant memory. */

#if __CUDACC__

__constant__ int const_N_train;    /* Copy of N_train in constant memory */
__constant__ int const_N_inputs;   /* Copy of N_inputs in constant memory */
__constant__ int const_N_targets;  /* Copy of N_targets in constant memory */

__constant__ net_arch const_arch;  /* Copy of arch in constant memory */
__constant__ net_flags const_flgs; /* Copy of flgs in constant memory */
__constant__ int const_has_flgs;   /* Are flags present in const_flgs? */

__constant__ model_specification const_model;  /* Constant copy of model */
__constant__ model_survival const_surv;  /* Constant copy of surv */

__constant__ net_sigmas const_sigmas;  /* Copy of sigmas in constant memory */
__constant__ net_params const_params;  /* Copy of params in constant memory */

__constant__ net_values *const_deriv;  /* Copy of deriv ptr in constant memory*/

static double *thread_energy;	/* Energies computed by concurrent threads */
static net_params *thread_grad;	/* Gradients computed by concurrent threads */

#endif


/* PROCEDURES. */

static void gibbs_noise (int, double);

static void gibbs_unit (int, net_param *, net_sigma *, net_sigma *, 
                        int, prior_spec *);

static void gibbs_conn (int, net_param *, net_sigma *, net_sigma *, net_sigma *,
                        int, int, prior_spec *);

static void gibbs_conn_config (int, net_param *, net_sigma *, net_sigma *, 
                        int, int, prior_spec *);

static void gibbs_adjustments (net_sigma *, double, int,
                               net_param *, net_sigma *, double,
                               net_param *, net_sigma *, double, int,
                               int, int *, net_param **, net_sigma **, 
                               prior_spec *, int *);

static void rgrid_met_noise (double, mc_iter *, double);

static void rgrid_met_unit (double, mc_iter *,
                            net_param *, net_sigma *, net_sigma *, 
                            int, prior_spec *);

static void rgrid_met_conn (double, mc_iter *,
                            net_param *, net_sigma *, net_sigma *, net_sigma *,
                            int, int, prior_spec *);

static void rgrid_met_conn_config (double, mc_iter *,
                            net_param *, net_sigma *, net_sigma *,
                            int, int, prior_spec *);

static double sum_squares (net_param *, net_sigma *, int);

static double rgrid_sigma (double, mc_iter *, double, 
                           double, double, double, double, int);


/* SET UP REQUIRED RECORD SIZES PRIOR TO GOBBLING RECORDS. */

void mc_app_record_sizes
( log_gobbled *logg	/* Structure to hold gobbled data */
)
{ 
  net_record_sizes(logg);
}


/* INITIALIZE AND SET UP DYNAMIC STATE STRUCTURE.  Skips some stuff
   if it's already been done, as indicated by the initialize_done
   variable. */

void mc_app_initialize
( log_gobbled *logg,	/* Records gobbled up from head and tail of log file */
  mc_dynamic_state *ds	/* Structure holding pointers to dynamical state */
)
{ 
  net_value *value_block;
  int value_count;
  int i, j, junk;

  if (!initialize_done)
  {
#   if __CUDACC__
    { show_gpu();

      char *cuda_sizes = getenv("CUDA_SIZES");
      if (cuda_sizes)
      { int t, b, n;
        char junk;
        if (sscanf(cuda_sizes,"%d:%d:%d%c",&t,&b,&n,&junk)!=3 
             || t<1 || b<1 || n<1)
        { fprintf(stderr,
           "Bad format for CUDA_SIZES: should be perthrd:blksize:numblks\n");
          exit(1);
        }
        if (b>MAX_BLKSIZE)
        { fprintf(stderr,"CUDA_SIZES blksize must not exceed %d\n",MAX_BLKSIZE);
          exit(1);
        }
        perthrd = t;
        blksize = b;
        numblks = n;
      }

      if (ask_show_gpu())
      { printf (
  "Computing with %d cases per thread, %d threads per block, %d blocks max\n",
         perthrd, blksize, numblks);
      }
    }
#   endif

    /* Check that required specification records are present. */

    arch   = (net_arch *) logg->data['A'];
    flgs   = (net_flags *) logg->data['F'];

    model  = (model_specification *) logg->data['M'];
    surv   = (model_survival *) logg->data['V'];

    priors = (net_priors *) logg->data['P'];

    net_check_specs_present(arch,priors,0,model,surv);

    if (model!=0 && model->type=='R' && model->autocorr)
    { fprintf(stderr,"Can't handle autocorrelated noise in net-mc\n");
      exit(1);
    }

    /* Look for quadratic approximation record.  If there is one, we use it. */

    quadratic_approx = (double *) logg->data['Q'];
  
    /* Locate existing network, if one exists. */
  
    sigmas.total_sigmas = net_setup_sigma_count(arch,flgs,model);
    params.total_params = net_setup_param_count(arch,flgs);
  
    sigmas.sigma_block = (net_sigma *) 
                          make_managed (logg->data['S'],logg->actual_size['S']);
    params.param_block = (net_param *) 
                          make_managed (logg->data['W'],logg->actual_size['W']);
  
    grad.total_params = params.total_params;
  
    if (sigmas.sigma_block!=0 || params.param_block!=0)
    {
      if (sigmas.sigma_block==0 || logg->index['S']!=logg->last_index
       || params.param_block==0 || logg->index['W']!=logg->last_index)
      { fprintf(stderr,
          "Network stored in log file is apparently incomplete\n");
        exit(1);
      }
  
      if (logg->actual_size['S'] != sigmas.total_sigmas*sizeof(net_sigma)
       || logg->actual_size['W'] != params.total_params*sizeof(net_param))
      { fprintf(stderr,"Bad size for network record\n");
        exit(1);
      }
  
      net_setup_sigma_pointers (&sigmas, arch, flgs, model);
      net_setup_param_pointers (&params, arch, flgs);
    }
    else
    {
      sigmas.sigma_block = 
        (net_sigma *) managed_alloc (sigmas.total_sigmas, sizeof (net_sigma));
      params.param_block = 
        (net_param *) managed_alloc (params.total_params, sizeof (net_param));
  
      net_setup_sigma_pointers (&sigmas, arch, flgs, model);
      net_setup_param_pointers (&params, arch, flgs);
   
      net_prior_generate (&params, &sigmas, arch, flgs, model, priors, 1, 0, 0);
    }

    /* Set up stepsize structure. */
  
    stepsizes.total_params = params.total_params;
    stepsizes.param_block = 
      (net_param *) chk_alloc (params.total_params, sizeof (net_param));
  
    net_setup_param_pointers (&stepsizes, arch, flgs);

    /* Find number of network values. */

    value_count = net_setup_value_count(arch);

    /* Set up second derivative and typical value structures. */

    value_block = (net_value *) chk_alloc (value_count, sizeof *value_block);
    net_setup_value_pointers (&seconds, value_block, arch);

    value_block = (net_value *) chk_alloc (value_count, sizeof *value_block);
    net_setup_value_pointers (&typical, value_block, arch);
  
    /* Read training data, if any, and allocate space for derivatives. */
  
    data_spec = (data_specifications *)
                   make_managed(logg->data['D'],logg->actual_size['D']);

    if (data_spec!=0 && model==0)
    { fprintf(stderr,"No model specified for data\n");
      exit(1);
    }

    if (data_spec && logg->actual_size['D'] !=
                       data_spec_size(data_spec->N_inputs,data_spec->N_targets))
    { fprintf(stderr,"Data specification record is the wrong size!\n");
      exit(1);
    }

    train_sumsq = (double *) chk_alloc (arch->N_inputs, sizeof *train_sumsq);
    for (j = 0; j<arch->N_inputs; j++) train_sumsq[j] = 0;
  
    if (data_spec!=0)
    { 
      net_data_read (1, 0, arch, model, surv);
    
      deriv = (net_values *) managed_alloc (N_train, sizeof *deriv);
    
      value_block = 
        (net_value *) managed_alloc (value_count*N_train, sizeof *value_block);
    
      for (i = 0; i<N_train; i++) 
      { net_setup_value_pointers (&deriv[i], value_block+value_count*i, arch);
      }
    
      for (j = 0; j<arch->N_inputs; j++)
      { for (i = 0; i<N_train; i++)
        { train_sumsq[j] += sq (train_values[i].i[j]);
        }
      }

      if (model!=0 && model->type=='V' && surv->hazard_type!='C')
      {
        double tsq;
        int n;

        tsq = 0;

        for (n = 0; surv->time[n]!=0; n++)
        { if (n==Max_time_points) abort();
          tsq += sq (surv->log_time ? log(surv->time[n]) : surv->time[n]);
        }

        train_sumsq[0] = N_train * tsq / n;
      }
    }

    /* Look for trajectory specification, and if there is one, read the
       approximation file, if there is one. */

    approx_count = 0;

    if (logg->data['t']!=0)
    { 
      mc_traj *trj = (mc_traj *) logg->data['t'];

      if (trj->approx_file[0]!=0)
      { 
        FILE *af;

        /* Open approximation file. */

        af = fopen(trj->approx_file,"r");
        if (af==NULL)
        { fprintf(stderr,
            "Can't open approximation file (%s)\n",trj->approx_file);
          exit(1);
        }

        /* Count how many entries it has. */

        approx_count = 0;
        while (fscanf(af,"%d",&junk)==1)
        { approx_count += 1;
        }

        /* Allocate space for data from file and read it. */

        approx_case = (int *) calloc (approx_count, sizeof *approx_case);
        approx_times = (int *) calloc (N_train, sizeof *approx_times);

        for (i = 0; i<N_train; i++)
        { approx_times[i] = 0;
        }

        rewind(af);
        for (i = 0; i<approx_count; i++)
        { if (fscanf(af,"%d",&approx_case[i])!=1
           || approx_case[i]<1 || approx_case[i]>N_train)
          { fprintf (stderr, "Bad entry in approximation file (%d:%d)\n",
                     i, approx_case[i]);
            exit(1);
          }
          approx_times[approx_case[i]-1] += 1;
        }
 
        /* Close file. */

        fclose(af);

        /* Check that all training cases appear in approx-file at least once. */

        for (i = 0; i<N_train; i++)
        { if (approx_times[i]==0)
          { fprintf (stderr,
              "Training case %d does not appear in approx-file\n", i);
            exit(1);
          }
        }
      }
    }

    /* Copy some data to constant memory in the GPU, if using CUDA. */

#   if __CUDACC__
    { check_cuda_error (cudaGetLastError(), 
                        "Before copying to constants");
      cudaMemcpyToSymbol (const_N_train, &N_train, sizeof N_train);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_N_train");
      cudaMemcpyToSymbol (const_N_inputs, &N_inputs, sizeof N_inputs);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_N_inputs");
      cudaMemcpyToSymbol (const_N_targets, &N_targets, sizeof N_targets);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_N_targets");
      cudaMemcpyToSymbol (const_arch, arch, sizeof *arch);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_arch");
      int has_flgs = flgs != 0;
      cudaMemcpyToSymbol (const_has_flgs, &has_flgs, sizeof has_flgs);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_has_flgs");
      if (has_flgs)
      { cudaMemcpyToSymbol (const_flgs, flgs, sizeof *flgs);
        check_cuda_error (cudaGetLastError(), 
                          "After copying to const_flgs");
      }
      if (model)
      { cudaMemcpyToSymbol (const_model, model, sizeof *model);
        check_cuda_error (cudaGetLastError(), 
                          "After copying to const_model");
      }
      if (surv)
      { cudaMemcpyToSymbol (const_model, surv, sizeof *surv);
        check_cuda_error (cudaGetLastError(), 
                          "After copying to const_surv");
      }
      cudaMemcpyToSymbol (const_sigmas, &sigmas, sizeof sigmas);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_sigmas");
      cudaMemcpyToSymbol (const_params, &params, sizeof params);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_params");
      cudaMemcpyToSymbol (const_deriv, &deriv, sizeof deriv);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_deriv");
    }
#   endif

    /* Make sure we don't do all this again. */

    initialize_done = 1;
  }

  /* Set up Monte Carlo state structure. */

  ds->aux_dim = sigmas.total_sigmas;
  ds->aux     = sigmas.sigma_block;

  ds->dim = params.total_params;
  ds->q   = params.param_block;

  ds->temp_state = 0;
  
  ds->stepsize = stepsizes.param_block;

  if (quadratic_approx && logg->actual_size['Q'] 
       != (1+ds->dim+ds->dim*ds->dim) * sizeof *quadratic_approx)
  { fprintf(stderr,"Approximation record is the wrong size (%d!=%d)\n",
       logg->actual_size['Q'], 
       (1+ds->dim+ds->dim*ds->dim) * (int)sizeof *quadratic_approx);
    exit(1);
  }

}


/* RESET INITIALIZE_DONE IN PREPARATION FOR NEW LOG FILE. */

void net_mc_cleanup(void)
{
  initialize_done = 0;
}


/* SAVE POSITION AND AUXILIARY PART OF STATE. */

void mc_app_save
( mc_dynamic_state *ds,	/* Current dyanamical state */
  log_file *logf,	/* Log file state structure */
  int index		/* Index of iteration being saved */
)
{ 
  logf->header.type = 'S';
  logf->header.index = index;
  logf->header.size = sigmas.total_sigmas * sizeof (net_sigma);
  log_file_append (logf, sigmas.sigma_block);

  logf->header.type = 'W';
  logf->header.index = index;
  logf->header.size = params.total_params * sizeof (net_param);
  log_file_append (logf, params.param_block);
}


/* APPLICATION-SPECIFIC SAMPLING PROCEDURE.  Does gibbs sampling for 
   hyperparameters ("sample-hyper"), or for noise levels ("sample-noise"),
   or for both ("sample-sigmas"), or does things separately for "upper"
   or "lower" level hyperparameters. */

int mc_app_sample 
( mc_dynamic_state *ds,
  char *op,
  double pm,
  double pm2,
  mc_iter *it,
  mc_temp_sched *sch
)
{
  int sample_hyper, sample_noise, rgrid_hyper, rgrid_noise;
  int l, pm0, grp;

  grp = 0;
  pm0 = pm;
  if (pm==0) pm = 0.1;

  sample_hyper = sample_noise = rgrid_hyper = rgrid_noise = 0;

  if (strcmp(op,"sample-sigmas")==0)
  { sample_hyper = sample_noise = 1;
  }
  else if (strcmp(op,"sample-hyper")==0)
  { sample_hyper = 1;
    grp = pm0;
  }
  else if (strcmp(op,"sample-noise")==0)
  { sample_noise = 1;
  }
  else if (strcmp(op,"sample-lower-sigmas")==0)
  { sample_hyper = sample_noise = -1;
  }
  else if (strcmp(op,"sample-lower-hyper")==0)
  { sample_hyper = -1;
  }
  else if (strcmp(op,"sample-lower-noise")==0)
  { sample_noise = -1;
  }
  else if (strcmp(op,"rgrid-upper-sigmas")==0)
  { rgrid_hyper = rgrid_noise = 1;
  }
  else if (strcmp(op,"rgrid-upper-hyper")==0)
  { rgrid_hyper = 1;
  }
  else if (strcmp(op,"rgrid-upper-noise")==0)
  { rgrid_noise = 1;
  }
  else
  { return 0;
  }

  if (rgrid_noise && model->type=='R')
  { rgrid_met_noise (pm, it, !ds->temp_state ? 1 : ds->temp_state->inv_temp);
  }

  if (rgrid_hyper)
  {
    if (arch->has_ti) rgrid_met_unit (pm, it, params.ti, sigmas.ti_cm, 0,
                                      arch->N_inputs, &priors->ti);
  
    for (l = 0; l<arch->N_layers; l++)
    {
      if (l>0)
      { if (arch->has_hh[l-1]) 
        { if (arch->hidden_config[l])
          { rgrid_met_conn_config (pm, it,
                        params.hh[l-1], sigmas.hh_cm[l-1], sigmas.hh[l-1], 
                        arch->N_hidden[l-1], arch->hidden_config[l]->N_wts,
                        &priors->hh[l-1]);
          }
          else
          { rgrid_met_conn (pm, it,
                        params.hh[l-1], sigmas.hh_cm[l-1], sigmas.hh[l-1], 
                        sigmas.ah[l], arch->N_hidden[l-1], arch->N_hidden[l], 
                        &priors->hh[l-1]);
          }
        }
      }
    
      if (arch->has_ih[l]) 
      { if (arch->input_config[l])
        { rgrid_met_conn_config (pm, it,
                      params.ih[l], sigmas.ih_cm[l], sigmas.ih[l],
                      arch->N_inputs, arch->input_config[l]->N_wts,
                      &priors->ih[l]); 
        }
        else
        { rgrid_met_conn (pm, it,
                      params.ih[l], sigmas.ih_cm[l], sigmas.ih[l], sigmas.ah[l],
                      not_omitted(flgs?flgs->omit:0,arch->N_inputs,1<<(l+1)), 
                      arch->N_hidden[l], &priors->ih[l]); 
        }
      }

      if (arch->has_bh[l])
      { if (arch->bias_config[l])
        { rgrid_met_unit (pm, it, params.bh[l], sigmas.bh_cm[l], 
                          0, arch->bias_config[l]->N_wts, &priors->bh[l]);
        }
        else
        { rgrid_met_unit (pm, it, params.bh[l], sigmas.bh_cm[l], 
                          sigmas.ah[l], arch->N_hidden[l], &priors->bh[l]);
        }
      }
      
      if (arch->has_th[l]) rgrid_met_unit (pm, it,
                                           params.th[l], sigmas.th_cm[l], 0,
                                           arch->N_hidden[l], &priors->th[l]);
  
      if (arch->has_ho[l]) 
      { int k = 2*arch->N_layers-1-l;
        if (arch->hidden_config[k])
        { rgrid_met_conn_config (pm, it,
                       params.ho[l], sigmas.ho_cm[l], sigmas.ho[l], 
                       arch->N_hidden[l], arch->hidden_config[k]->N_wts,
                       &priors->ho[l]);
          }
        else
        { rgrid_met_conn (pm, it,
                       params.ho[l], sigmas.ho_cm[l], sigmas.ho[l], sigmas.ao,
                       arch->N_hidden[l], arch->N_outputs, &priors->ho[l]);
        }
      }
          
    }
  
    if (arch->has_io) 
    { if (arch->input_config[arch->N_layers])
      { rgrid_met_conn_config (pm, it, params.io, sigmas.io_cm, sigmas.io,
                    arch->N_inputs, arch->input_config[arch->N_layers]->N_wts,
                    &priors->io); 
      }
      else
      { rgrid_met_conn (pm, it, params.io, sigmas.io_cm, sigmas.io, sigmas.ao,
                        not_omitted(flgs?flgs->omit:0,arch->N_inputs,1), 
                        arch->N_outputs, &priors->io);
      }
    }
  
    if (arch->has_bo) 
    { if (arch->bias_config[arch->N_layers])
      { rgrid_met_unit (pm, it, params.bo, sigmas.bo_cm, 0,
                        arch->bias_config[arch->N_layers]->N_wts, &priors->bo);
      }
      else
      { rgrid_met_unit (pm, it, params.bo, sigmas.bo_cm, sigmas.ao,
                        arch->N_outputs, &priors->bo);
      }
    }
  }

  if (sample_noise && model->type=='R')
  { 
    gibbs_noise (sample_noise, !ds->temp_state ? 1 : ds->temp_state->inv_temp);
  }

  if (sample_hyper)
  {
#   define THISGRP (pm0==0 || --grp==0)

    if (arch->has_ti && THISGRP) 
    { gibbs_unit (sample_hyper, params.ti, sigmas.ti_cm, 0,
                  arch->N_inputs, &priors->ti);
    }
  
    for (l = 0; l<arch->N_layers; l++)
    {
      if (l>0)
      { if (arch->has_hh[l-1] && THISGRP) 
        { if (arch->hidden_config[l])
          { gibbs_conn_config (sample_hyper, 
                            params.hh[l-1], sigmas.hh_cm[l-1], sigmas.hh[l-1],
                            arch->N_hidden[l-1], arch->hidden_config[l]->N_wts,
                            &priors->hh[l-1]);
          }
          else
          { gibbs_conn (sample_hyper, 
                        params.hh[l-1], sigmas.hh_cm[l-1], sigmas.hh[l-1], 
                        sigmas.ah[l], arch->N_hidden[l-1], arch->N_hidden[l], 
                        &priors->hh[l-1]);
          }
        }
      }
    
      if (arch->has_ih[l] && THISGRP) 
      { if (arch->input_config[l])
        { gibbs_conn_config (sample_hyper, 
                             params.ih[l], sigmas.ih_cm[l], sigmas.ih[l],
                             arch->N_inputs, arch->input_config[l]->N_wts,
                             &priors->ih[l]); 
        }
        else
        { gibbs_conn (sample_hyper, 
                      params.ih[l], sigmas.ih_cm[l], sigmas.ih[l], sigmas.ah[l],
                      not_omitted(flgs?flgs->omit:0,arch->N_inputs,1<<(l+1)), 
                      arch->N_hidden[l], &priors->ih[l]); 
        }
      }

      if (arch->has_bh[l] && THISGRP)
      { if (arch->bias_config[l])
        { gibbs_unit (sample_hyper, params.bh[l], sigmas.bh_cm[l], 
                      0, arch->bias_config[l]->N_wts, &priors->bh[l]);
        }
        else
        { gibbs_unit (sample_hyper, params.bh[l], sigmas.bh_cm[l], 
                      sigmas.ah[l], arch->N_hidden[l], &priors->bh[l]);
        }
      }

      if (arch->has_ah[l] && THISGRP)
      { gibbs_adjustments (sigmas.ah[l], priors->ah[l], arch->N_hidden[l], 
          arch->has_bh[l] && !arch->bias_config[l] ? params.bh[l] : 0,
            sigmas.bh_cm[l], 
            priors->bh[l].alpha[1],
          arch->has_ih[l] && !arch->input_config[l] ? params.ih[l] : 0, 
            sigmas.ih[l],
            priors->ih[l].alpha[2], 
            not_omitted(flgs?flgs->omit:0,arch->N_inputs,1<<(l+1)), 
          l>0 && !arch->hidden_config[l], 
            l>0 ? &arch->has_hh[l-1] : 0, 
            l>0 ? &params.hh[l-1] : 0,
            l>0 ? &sigmas.hh[l-1] : 0, 
            l>0 ? &priors->hh[l-1] : 0,
            l>0 ? &arch->N_hidden[l-1] : 0);
      }
      
      if (arch->has_th[l] && THISGRP) 
      { gibbs_unit (sample_hyper, params.th[l], sigmas.th_cm[l], 0,
                    arch->N_hidden[l], &priors->th[l]);
      }
    }
  
    for (l = arch->N_layers-1; l>=0; l--)
    { if (arch->has_ho[l] && THISGRP)
      { int k = 2*arch->N_layers-1-l;
        if (arch->hidden_config[k])
        { gibbs_conn_config (sample_hyper, 
                             params.ho[l], sigmas.ho_cm[l], sigmas.ho[l],
                             arch->N_hidden[l], arch->hidden_config[k]->N_wts,
                             &priors->ho[l]);
          }
        else
        { gibbs_conn (sample_hyper, 
                      params.ho[l], sigmas.ho_cm[l], sigmas.ho[l], sigmas.ao,
                      arch->N_hidden[l], arch->N_outputs, &priors->ho[l]);
        }
      }
    }
  
    if (arch->has_io && THISGRP)
    { if (arch->input_config[arch->N_layers])
      { gibbs_conn_config (sample_hyper, 
                      params.io, sigmas.io_cm, sigmas.io,
                      arch->N_inputs, arch->input_config[arch->N_layers]->N_wts,
                      &priors->io); 
      }
      else
      { gibbs_conn (sample_hyper, 
                    params.io, sigmas.io_cm, sigmas.io, sigmas.ao,
                    not_omitted(flgs?flgs->omit:0,arch->N_inputs,1), 
                    arch->N_outputs, &priors->io);
      }
    }
  
    if (arch->has_bo && THISGRP)
    { gibbs_unit (sample_hyper, params.bo, sigmas.bo_cm, sigmas.ao,
                                arch->N_outputs, &priors->bo);
    }

    if (arch->has_ao && THISGRP)
    { int has_ho[Max_layers];
      memcpy (has_ho, arch->has_ho, Max_layers * sizeof *has_ho);
      if (arch->hidden_config[arch->N_layers]) 
      { has_ho[arch->N_layers-1] = 0;
      }
      gibbs_adjustments (sigmas.ao, priors->ao, arch->N_outputs, 
        arch->has_bo ? params.bo : 0, 
          sigmas.bo_cm,
          priors->bo.alpha[1],
        arch->has_io && !arch->input_config[arch->N_layers] ? params.io : 0,   
          sigmas.io, 
          priors->io.alpha[2], 
          not_omitted(flgs?flgs->omit:0,arch->N_inputs,1), 
        arch->N_layers,
          has_ho,
          params.ho,
          sigmas.ho,
          priors->ho,
          arch->N_hidden);
    }
  }
  
  ds->know_pot  = 0;
  ds->know_grad = 0;

  return 1;
}


/* DO GIBBS SAMPLING FOR NOISE SIGMAS. */

static void gibbs_noise 
( int sample_noise,	/* +1 for all, -1 for lower only */
  double inv_temp
)
{
  double nalpha, nprec, sum, d, ps;
  prior_spec *pr;
  int i, j;

  for (i = 0; i<N_train; i++) 
  { net_func (&train_values[i], 0, arch, flgs, &params);
  }

  pr = &model->noise;

  if (pr->alpha[1]!=0 && pr->alpha[2]==0)
  {
    for (j = 0; j<arch->N_outputs; j++)
    {
      sum = pr->alpha[1] * (*sigmas.noise_cm * *sigmas.noise_cm);
      for (i = 0; i<N_train; i++)
      { d = train_values[i].o[j] - train_targets[i*arch->N_outputs+j];
        sum += inv_temp * d*d;
      }

      nalpha = pr->alpha[1] + inv_temp * N_train;
      nprec = nalpha / sum;

      sigmas.noise[j] = prior_pick_sigma (1/sqrt(nprec), nalpha);
    }
  }

  if (pr->alpha[1]!=0 && pr->alpha[2]!=0 && sample_noise>0)
  {
    for (j = 0; j<arch->N_outputs; j++)
    {
      ps = pr->alpha[2] * (sigmas.noise[j] * sigmas.noise[j]);

      sum = 0;
      for (i = 0; i<N_train; i++)
      { d = train_values[i].o[j] - train_targets[i*arch->N_outputs+j];
        sum += rand_gamma((pr->alpha[2]+inv_temp)/2) / ((ps+inv_temp*d*d)/2);
      }

      sigmas.noise[j] = cond_sigma (*sigmas.noise_cm, pr->alpha[1],
                                    pr->alpha[2], sum, N_train);
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]==0 && pr->alpha[2]==0)
  {
    sum = pr->alpha[0] * (pr->width * pr->width);
    for (i = 0; i<N_train; i++)
    { for (j = 0; j<arch->N_outputs; j++)
      { d = train_values[i].o[j] - train_targets[i*arch->N_outputs+j];
        sum += inv_temp * d*d;
      }
    }

    nalpha = pr->alpha[0] + inv_temp * N_train * arch->N_outputs;
    nprec = nalpha / sum;
    *sigmas.noise_cm = prior_pick_sigma (1/sqrt(nprec), nalpha);

    for (j = 0; j<arch->N_outputs; j++)
    { sigmas.noise[j] = *sigmas.noise_cm;
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]==0 && pr->alpha[2]!=0 && sample_noise>0)
  {
    ps = pr->alpha[2] * (*sigmas.noise_cm * *sigmas.noise_cm);

    sum = 0;
    for (i = 0; i<N_train; i++)
    { for (j = 0; j<arch->N_outputs; j++) 
      { d = train_values[i].o[j] - train_targets[i*arch->N_outputs+j];
        sum += rand_gamma((pr->alpha[2]+inv_temp)/2) / ((ps+inv_temp*d*d)/2);
      }
    }

    *sigmas.noise_cm = cond_sigma (pr->width, pr->alpha[0],
                                   pr->alpha[2], sum, arch->N_outputs*N_train);

    for (j = 0; j<arch->N_outputs; j++)
    { sigmas.noise[j] = *sigmas.noise_cm;
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]!=0 && sample_noise>0)
  {
    sum = 0;
    for (j = 0; j<arch->N_outputs; j++) 
    { sum += 1 / (sigmas.noise[j] * sigmas.noise[j]);
    }

    *sigmas.noise_cm = cond_sigma (pr->width, pr->alpha[0],
                                   pr->alpha[1], sum, arch->N_outputs);
  }
}


/* DO RANDOM-GRID METROPOLIS UPDATES FOR UPPER NOISE SIGMAS. */

static void rgrid_met_noise 
( double stepsize,	/* Stepsize for update */
  mc_iter *it,
  double inv_temp
)
{
  double sum, d, ps;
  prior_spec *pr;
  int i, j;

  for (i = 0; i<N_train; i++) 
  { net_func (&train_values[i], 0, arch, flgs, &params);
  }

  pr = &model->noise;

  if (pr->alpha[1]!=0 && pr->alpha[2]!=0)
  {
    for (j = 0; j<arch->N_outputs; j++)
    {
      ps = pr->alpha[2] * (sigmas.noise[j] * sigmas.noise[j]);

      sum = 0;
      for (i = 0; i<N_train; i++)
      { d = train_values[i].o[j] - train_targets[i*arch->N_outputs+j];
        sum += rand_gamma((pr->alpha[2]+inv_temp)/2) / ((ps+inv_temp*d*d)/2);
      }

      sigmas.noise[j] = rgrid_sigma (stepsize, it, sigmas.noise[j],
                                     *sigmas.noise_cm, pr->alpha[1],
                                     pr->alpha[2], sum, N_train);
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]==0 && pr->alpha[2]!=0)
  {
    ps = pr->alpha[2] * (*sigmas.noise_cm * *sigmas.noise_cm);

    sum = 0;
    for (i = 0; i<N_train; i++)
    { for (j = 0; j<arch->N_outputs; j++) 
      { d = train_values[i].o[j] - train_targets[i*arch->N_outputs+j];
        sum += rand_gamma((pr->alpha[2]+inv_temp)/2) / ((ps+inv_temp*d*d)/2);
      }
    }

    *sigmas.noise_cm = rgrid_sigma (stepsize, it, *sigmas.noise_cm,
                                    pr->width, pr->alpha[0],
                                    pr->alpha[2], sum, arch->N_outputs*N_train);

    for (j = 0; j<arch->N_outputs; j++)
    { sigmas.noise[j] = *sigmas.noise_cm;
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]!=0)
  {
    sum = 0;
    for (j = 0; j<arch->N_outputs; j++) 
    { sum += 1 / (sigmas.noise[j] * sigmas.noise[j]);
    }

    *sigmas.noise_cm = rgrid_sigma (stepsize, it, *sigmas.noise_cm,
                                    pr->width, pr->alpha[0],
                                    pr->alpha[1], sum, arch->N_outputs);
  }
}


/* DO GIBBS SAMPLING FOR SIGMA ASSOCIATED WITH GROUP OF UNITS. */

static void gibbs_unit
( int sample_hyper,	/* +1 for all, -1 for lower only */
  net_param *wt,	/* Parameters associated with each unit */
  net_sigma *sg_cm,	/* Common sigma controlling parameter distribution */
  net_sigma *adj,	/* Adjustments for each unit, or zero */
  int n,		/* Number of units */
  prior_spec *pr		/* Prior for sigmas */
)
{ 
  double nalpha, nprec, sum, ps, d;
  int i;

  if (pr->alpha[0]!=0 && pr->alpha[1]==0)
  {
    nalpha = pr->alpha[0] + n;

    nprec= nalpha / (pr->alpha[0] * (pr->width*pr->width)
                      + sum_squares(wt,adj,n));

    *sg_cm = prior_pick_sigma (1/sqrt(nprec), nalpha);
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]!=0 && sample_hyper>0)
  { 
    ps = pr->alpha[1] * (*sg_cm * *sg_cm);

    sum = 0;
    for (i = 0; i<n; i++)
    { d = adj==0 ? wt[i] : wt[i]/adj[i];
      sum += rand_gamma((pr->alpha[1]+1)/2) / ((ps+d*d)/2);
    }

    *sg_cm = cond_sigma (pr->width, pr->alpha[0], pr->alpha[1], sum, n);
  }
  
}


/* DO RANDOM-GRID UPDATES FOR UPPER SIGMA ASSOCIATED WITH GROUP OF UNITS. */

static void rgrid_met_unit
( double stepsize,	/* Stepsize for update */
  mc_iter *it,
  net_param *wt,	/* Parameters associated with each unit */
  net_sigma *sg_cm,	/* Common sigma controlling parameter distribution */
  net_sigma *adj,	/* Adjustments for each unit, or zero */
  int n,		/* Number of units */
  prior_spec *pr		/* Prior for sigmas */
)
{ 
  double sum, ps, d;
  int i;

  if (pr->alpha[0]!=0 && pr->alpha[1]!=0)
  { 
    ps = pr->alpha[1] * (*sg_cm * *sg_cm);

    sum = 0;
    for (i = 0; i<n; i++)
    { d = adj==0 ? wt[i] : wt[i]/adj[i];
      sum += rand_gamma((pr->alpha[1]+1)/2) / ((ps+d*d)/2);
    }

    *sg_cm = rgrid_sigma (stepsize, it, *sg_cm,
                          pr->width, pr->alpha[0], pr->alpha[1], sum, n);
  }
  
}


/* DO GIBBS SAMPLING FOR SIGMAS ASSOCIATED WITH GROUP OF CONNECTIONS. */

static void gibbs_conn
( int sample_hyper,	/* +1 for all, -1 for lower only */
  net_param *wt,	/* Weights on connections */
  net_sigma *sg_cm,	/* Common sigma controlling weights */
  net_sigma *sg,	/* Individual sigmas for source units */
  net_sigma *adj,	/* Adjustments for each destination unit, or zero */
  int ns,		/* Number of source units */
  int nd,		/* Number of destination units */
  prior_spec *pr	/* Prior for sigmas */
)
{ 
  double width, nalpha, nprec, sum, ps, d;
  int i, j;

  width = prior_width_scaled(pr,ns);

  if (pr->alpha[1]!=0 && pr->alpha[2]==0)
  {
    for (i = 0; i<ns; i++)
    {
      nalpha = pr->alpha[1] + nd;
      nprec = nalpha / (pr->alpha[1] * (*sg_cm * *sg_cm)
                         + sum_squares(wt+nd*i,adj,nd));

      sg[i] = prior_pick_sigma (1/sqrt(nprec), nalpha);
    }
  }

  if (pr->alpha[1]!=0 && pr->alpha[2]!=0 && sample_hyper>0)
  { 
    for (i = 0; i<ns; i++)
    { 
      ps = pr->alpha[2] * (sg[i]*sg[i]);

      sum = 0;        
      for (j = 0; j<nd; j++)
      { d = adj==0 ? wt[nd*i+j] : wt[nd*i+j]/adj[j];
        sum += rand_gamma((pr->alpha[2]+1)/2) / ((ps+d*d)/2);
      }  

      sg[i] = cond_sigma (*sg_cm, pr->alpha[1], pr->alpha[2], sum, nd);
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]==0 && pr->alpha[2]==0)
  {
    nalpha = pr->alpha[0] + ns*nd;

    sum = pr->alpha[0] * (width * width);
    for (i = 0; i<ns; i++)
    { sum += sum_squares(wt+nd*i,adj,nd);
    }
    nprec = nalpha / sum;

    *sg_cm = prior_pick_sigma (1/sqrt(nprec), nalpha);

    for (i = 0; i<ns; i++)
    { sg[i] = *sg_cm;
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]==0 && pr->alpha[2]!=0 && sample_hyper>0)
  {
    ps = pr->alpha[2] * (*sg_cm * *sg_cm);

    sum = 0;        
    for (i = 0; i<ns; i++)
    { for (j = 0; j<nd; j++)
      { d = adj==0 ? wt[nd*i+j] : wt[nd*i+j]/adj[j];
        sum += rand_gamma((pr->alpha[2]+1)/2) / ((ps+d*d)/2);
      }     
    }

    *sg_cm = cond_sigma (width, pr->alpha[0], pr->alpha[2], sum, ns*nd);

    for (i = 0; i<ns; i++)
    { sg[i] = *sg_cm;
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]!=0 && sample_hyper>0)
  {
    sum = 0;
    for (i = 0; i<ns; i++) 
    { sum += 1 / (sg[i] * sg[i]);
    }

    *sg_cm = cond_sigma (width, pr->alpha[0], pr->alpha[1], sum, ns);
  }

}


/* DO GIBBS SAMPLING FOR SIGMA ASSOCIATED WITH CONNECTIONS WITH CONFIG. 
   Note that pr->alpha[1] must be zero, and the prior must not be scaled. */

static void gibbs_conn_config
( int sample_hyper,	/* +1 for all, -1 for lower only */
  net_param *wt,	/* Weights on connections */
  net_sigma *sg_cm,	/* Common sigma controlling weights */
  net_sigma *sg,	/* Individual sigmas for source units (all = common) */
  int ns,		/* Number of source units */
  int nw,		/* Number of weights */
  prior_spec *pr	/* Prior for sigmas */
)
{ 
  double width, nalpha, nprec, sum, ps;
  int i;

  width = pr->width;

  if (pr->alpha[0]!=0 && pr->alpha[2]==0)
  {
    nalpha = pr->alpha[0] + nw;

    sum = pr->alpha[0] * (width * width);
    sum += sum_squares(wt,0,nw);
    nprec = nalpha / sum;

    *sg_cm = prior_pick_sigma (1/sqrt(nprec), nalpha);

    for (i = 0; i<ns; i++)
    { sg[i] = *sg_cm;
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[2]!=0 && sample_hyper>0)
  {
    ps = pr->alpha[2] * (*sg_cm * *sg_cm);

    sum = 0;        
    for (i = 0; i<nw; i++)
    { double d = wt[i];
      sum += rand_gamma((pr->alpha[2]+1)/2) / ((ps+d*d)/2);
    }

    *sg_cm = cond_sigma (width, pr->alpha[0], pr->alpha[2], sum, nw);

    for (i = 0; i<ns; i++)
    { sg[i] = *sg_cm;
    }
  }
}


/* DO RANDOM-GRID UPDATES FOR UPPER SIGMAS ASSOCIATED WITH GROUP OF CONNECTIONS.
 */

static void rgrid_met_conn
( double stepsize,	/* Stepsize for update */
  mc_iter *it,
  net_param *wt,	/* Weights on connections */
  net_sigma *sg_cm,	/* Common sigma controlling weights */
  net_sigma *sg,	/* Individual sigmas for source units */
  net_sigma *adj,	/* Adjustments for each destination unit, or zero */
  int ns,		/* Number of source units */
  int nd,		/* Number of destination units */
  prior_spec *pr	/* Prior for sigmas */
)
{ 
  double width, sum, ps, d;
  int i, j;

  width = prior_width_scaled(pr,ns);

  if (pr->alpha[1]!=0 && pr->alpha[2]!=0)
  { 
    for (i = 0; i<ns; i++)
    { 
      ps = pr->alpha[2] * (sg[i]*sg[i]);

      sum = 0;        
      for (j = 0; j<nd; j++)
      { d = adj==0 ? wt[nd*i+j] : wt[nd*i+j]/adj[j];
        sum += rand_gamma((pr->alpha[2]+1)/2) / ((ps+d*d)/2);
      }  

      sg[i] = rgrid_sigma (stepsize, it, sg[i],
                           *sg_cm, pr->alpha[1], pr->alpha[2], sum, nd);
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]==0 && pr->alpha[2]!=0)
  {
    ps = pr->alpha[2] * (*sg_cm * *sg_cm);

    sum = 0;        
    for (i = 0; i<ns; i++)
    { for (j = 0; j<nd; j++)
      { d = adj==0 ? wt[nd*i+j] : wt[nd*i+j]/adj[j];
        sum += rand_gamma((pr->alpha[2]+1)/2) / ((ps+d*d)/2);
      }     
    }

    *sg_cm = rgrid_sigma (stepsize, it, *sg_cm,
                          width, pr->alpha[0], pr->alpha[2], sum, ns*nd);

    for (i = 0; i<ns; i++)
    { sg[i] = *sg_cm;
    }
  }

  if (pr->alpha[0]!=0 && pr->alpha[1]!=0)
  {
    sum = 0;
    for (i = 0; i<ns; i++) 
    { sum += 1 / (sg[i] * sg[i]);
    }

    *sg_cm = rgrid_sigma (stepsize, it, *sg_cm,
                          width, pr->alpha[0], pr->alpha[1], sum, ns);
  }
}


/* DO RANDOM-GRID UPDATES FOR UPPER SIGMAS ASSOCIATED WITH GROUP OF CONNECTIONS.
   Version for group with weight configuration file. Note that pr->alpha[1] 
   must be zero, and the prior must not be scaled. */

static void rgrid_met_conn_config
( double stepsize,	/* Stepsize for update */
  mc_iter *it,
  net_param *wt,	/* Weights on connections */
  net_sigma *sg_cm,	/* Common sigma controlling weights */
  net_sigma *sg,	/* Individual sigmas for source units */
  int ns,		/* Number of source units */
  int nw,		/* Number of weights */
  prior_spec *pr	/* Prior for sigmas */
)
{ 
  double width, sum, ps, d;
  int i;

  width = pr->width;

  if (pr->alpha[0]!=0 && pr->alpha[1]==0 && pr->alpha[2]!=0)
  {
    ps = pr->alpha[2] * (*sg_cm * *sg_cm);

    sum = 0;        
    for (i = 0; i<nw; i++)
    { d = wt[i];
      sum += rand_gamma((pr->alpha[2]+1)/2) / ((ps+d*d)/2);
    }

    *sg_cm = rgrid_sigma (stepsize, it, *sg_cm,
                          width, pr->alpha[0], pr->alpha[2], sum, nw);

    for (i = 0; i<ns; i++)
    { sg[i] = *sg_cm;
    }
  }
}


/* DO GIBBS SAMPLING FOR UNIT ADJUSTMENTS. */

static void gibbs_adjustments
( net_sigma *adj,	/* Adjustments to sample for */
  double alpha,		/* Alpha for adjustments */
  int nd,		/* Number of units with adjustments */
  net_param *b,		/* Biases for destination units, or zero */
  net_sigma *s,		/* Sigma associated with biases */
  double a,		/* Alpha for this sigma */
  net_param *w1,	/* First set of weights, or zero */
  net_sigma *s1,	/* Sigmas associated with first set */
  double a1,		/* Alpha for first set */
  int n1,		/* Number of source units for first set */
  int nrem,		/* Number of remaining weight sets */
  int *has,		/* Whether each remaining set is present */
  net_param **wr,	/* Remaining sets of weights */
  net_sigma **sr,	/* Sigmas associated with remaining sets */
  prior_spec *ar,	/* Priors for remaining sets */
  int *nr		/* Numbers of source units in remaining sets */
)
{ 
  double nalpha, nprec, sum, d, ad;
  int r, i, j;

  for (i = 0; i<nd; i++)
  {
    nalpha = alpha;
    sum = alpha;

    ad = adj[i];
  
    if (b!=0)
    { 
      nalpha += 1;

      d = b[i];
      if (a==0)
      { d /= *s;
      }
      else
      { d /= prior_pick_sigma (sqrt ((a * (*s * *s) + (d*d)/(ad*ad)) 
                                      / (a+1)), a+1);
      } 
      sum += d*d;
    }
  
    if (w1!=0)
    { 
      nalpha += n1;

      for (j = 0; j<n1; j++)
      { d = w1[nd*j+i];
        if (a1==0)
        { d /= s1[j];
        }
        else
        { d /= prior_pick_sigma (sqrt ((a1 * (s1[j] * s1[j]) + (d*d)/(ad*ad)) 
                                       / (a1+1)), a1+1);
        } 
        sum += d*d;
      }
    }

    for (r = 0; r<nrem; r++)
    {
      if (has[r])
      { 
        nalpha += nr[r];

        for (j = 0; j<nr[r]; j++)
        { d = wr[r][nd*j+i];
          if (ar[r].alpha[2]==0)
          { d /= sr[r][j];
          }
          else
          { d /= prior_pick_sigma 
                 (sqrt ((ar[r].alpha[2] * (sr[r][j] * sr[r][j]) + (d*d)/(ad*ad))
                          / (ar[r].alpha[2]+1)), ar[r].alpha[2]+1);
          } 
          sum += d*d;
        }
      }
    }
  
    nprec = nalpha / sum;
  
    adj[i] = prior_pick_sigma (1/sqrt(nprec), nalpha);
  }
}


/* EVALUATE POTENTIAL ENERGY AND ITS GRADIENT DUE TO ONE TRAINING CASE. 
   Adds the results to the accumulators passed. */

#if __CUDACC__ && __CUDA_ARCH__  /* Compiling for GPU */

#define N_targets	const_N_targets
#define arch		(&const_arch)
#define flgs		(const_has_flgs ? &const_flgs : 0)
#define model		(&const_model)
#define surv		(&const_surv)
#define sigmas		const_sigmas
#define params		const_params
#define deriv		const_deriv

#endif

HOSTDEV static void one_case  /* Energy and gradient from one training case */
( 
  double *energy,	/* Place to increment energy, null if not required */
  net_params *grd,	/* Place to increment gradient, null if not required */
  int i,		/* Case to look at */
  double en_weight,	/* Weight for this case for energy */
  double gr_weight	/* Weight for this case for gradient */
)
{
  int k;

  if (model->type=='V'          /* Handle piecewise-constant hazard    */
   && surv->hazard_type=='P')   /*   model specially                   */
  { 
    double ot, t0, t1;
    int censored;
    int w;
  
    if (train_targets[i]<0)
    { censored = 1;
      ot = -train_targets[i];
    }
    else
    { censored = 0;
      ot = train_targets[i];
    }
  
    t0 = 0;
    t1 = surv->time[0];
    train_values[i].i[0] = surv->log_time ? log(t1) : t1;
  
    w = 0;
  
    for (;;)
    {
      net_func (&train_values[i], 0, arch, flgs, &params);
      
      double fudged_target = ot>t1 ? -(t1-t0) : censored ? -(ot-t0) : (ot-t0);
      double log_prob;
  
      net_model_prob(&train_values[i], &fudged_target,
                     &log_prob, grd ? &deriv[i] : 0, arch, model, surv, 
                     &sigmas, Cheap_energy);
  
      if (energy) *energy -= en_weight * log_prob;
  
      if (grd)
      { if (gr_weight!=1)
        { for (k = 0; k<arch->N_outputs; k++)
          { deriv[i].o[k] *= gr_weight;
          }
        }
        net_back (&train_values[i], &deriv[i], arch->has_ti ? -1 : 0,
                  arch, flgs, &params);
        net_grad (grd, &params, &train_values[i], &deriv[i], arch, flgs);
      }
  
      if (ot<=t1) break;
   
      t0 = t1;
      w += 1;
      
      if (surv->time[w]==0) 
      { t1 = ot;
        train_values[i].i[0] = surv->log_time ? log(t0) : t0;
      }
      else
      { t1 = surv->time[w];
        train_values[i].i[0] = surv->log_time ? (log(t0)+log(t1))/2
                                              : (t0+t1)/2;
      }
    }
  }
  
  else /* Everything except piecewise-constant hazard model */
  { 
    net_func (&train_values[i], 0, arch, flgs, &params);

    double log_prob;
    net_model_prob(&train_values[i], train_targets+N_targets*i,
                   &log_prob, grd ? &deriv[i] : 0, arch, model, surv,
                   &sigmas, Cheap_energy);
    
    if (energy)
    { *energy -= en_weight * log_prob;
    }

    if (grd)
    { if (gr_weight!=1)
      { for (k = 0; k<arch->N_outputs; k++)
        { deriv[i].o[k] *= gr_weight;
        }
      }

      net_back (&train_values[i], &deriv[i], arch->has_ti ? -1 : 0,
                arch, flgs, &params);

      net_grad (grd, &params, &train_values[i], &deriv[i], arch, flgs);
    }
  }
}

#if __CUDACC__ && __CUDA_ARCH__  /* Compiling for GPU */

#undef N_targets
#undef arch
#undef flgs
#undef model
#undef surv
#undef sigmas
#undef params
#undef deriv

#endif


#if __CUDACC__

__global__ void many_cases 
(
  double *thread_energy,   /* Places to store energy, null if not required */
  net_params *thread_grad, /* Places to store gradient, null if not required */
  int start,		/* Start of cases to look at */
  int cases_per_thread,	/* Number of case to handle in one thread */
  double en_weight,	/* Weight for this case for energy */
  double gr_weight	/* Weight for this case for gradient */
)
{ 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = start + cases_per_thread * i;

  if (j < const_N_train) 
  { 
    double *threi;
    net_params *thrgi;
    
    if (thread_energy)
    { threi = thread_energy + i;
      *threi = 0;
    }

    if (thread_grad)
    { thrgi = thread_grad + i;
      if (0)  /* seems to be slower */
      { memset(thrgi->param_block, 0, thrgi->total_params * sizeof (net_param));
      }
      else
      { unsigned t = thrgi->total_params;
        unsigned k;
        for (k = 0; k < t; k++)
        { thrgi->param_block[k] = 0;
        }
      }
    }

    int h;
    for (h = j; h < j+cases_per_thread && h < const_N_train; h++)
    { one_case (thread_energy ? threi : 0, 
                thread_grad ? thrgi : 0, 
                h, en_weight, gr_weight);
    }

    if (0)  /* Linear-time reduction */
    { __syncthreads();
      if (threadIdx.x==0)
      { for (h = 1; 
             h<blockDim.x && start + cases_per_thread * (i+h) < const_N_train;
             h++)
        { if (thread_energy)
          { *threi += threi[h];
          }
          if (thread_grad)
          { unsigned k;
            for (k = 0; k < thrgi->total_params; k++)
            { thrgi->param_block[k] += thrgi[h].param_block[k];
            }
          }
        }
      }
    }
    else  /* Logarithmic-time reduction */
    {
      int stride;
      for (stride = 1; stride < blockDim.x; stride <<= 1)
      { __syncthreads();
        if ((i & (2*stride-1)) == 0 
              && j + cases_per_thread*stride < const_N_train)
        { if (thread_energy)
          { *threi += threi[stride];
          }
          if (thread_grad)
          { net_param *p = thrgi[stride].param_block;
            net_param *q = thrgi->param_block;
            unsigned t = thrgi->total_params;
            unsigned k;
            for (k = 0; k < t; k++)
            { q[k] += p[k];
            }
          }
        }
      }
    }
  }
}

#endif


void mc_app_energy
( mc_dynamic_state *ds,	/* Current dynamical state */
  int N_approx,		/* Number of gradient approximations in use */
  int w_approx,		/* Which approximation to use this time */
  double *energy,	/* Place to store energy, null if not required */
  mc_value *gr		/* Place to store gradient, null if not required */
)
{
  double log_prob, inv_temp;
  int i, j, low, high;

  inv_temp = !ds->temp_state ? 1 : ds->temp_state->inv_temp;

  if (gr && grad.param_block!=gr)
  { grad.param_block = gr;
    net_setup_param_pointers (&grad, arch, flgs);
  }

#   if __CUDACC__
    { if (N_train>0)
      { unsigned max_threads = blksize*numblks*perthrd > N_train 
                                ? (N_train + perthrd -1) / perthrd
                                : blksize*numblks;
        if (energy && thread_energy==0)
        { thread_energy = (double *) 
                            managed_alloc (max_threads, sizeof (double));
        }
        if (gr && thread_grad==0)
        { thread_grad = (net_params *) 
                          managed_alloc (max_threads, sizeof (net_params));
          thread_grad->total_params = grad.total_params;
          thread_grad->param_block = (net_param *) 
            managed_alloc (max_threads * grad.total_params, sizeof (net_param));
          net_setup_param_pointers (thread_grad, arch, flgs);
          net_replicate_param_pointers (thread_grad, arch, max_threads);
        }
      }
    }
#   endif

  if (inv_temp>=0)
  { net_prior_prob (&params, &sigmas, &log_prob, gr ? &grad : 0, 
                    arch, flgs, priors, 2);
  }
  else
  { log_prob = 0;
    if (gr)
    { for (i = 0; i<ds->dim; i++) 
      { grad.param_block[i] = 0;
      }
    }
    inv_temp = -inv_temp;
  }

  if (energy) *energy = -log_prob;

  if (-log_prob>=1e30)
  { if (energy) *energy = 1e30;
    if (gr)
    { for (i = 0; i<ds->dim; i++)
      { grad.param_block[i] = 0;
      }
    }
    return;
  }

  if (inv_temp!=0 && (data_spec!=0 || quadratic_approx))
  {
    net_model_check (model);    

    if (quadratic_approx)
    {
      double *b, *V;
      int i, j;

      if (energy) *energy += *quadratic_approx;

      b = quadratic_approx + 1;
      V = quadratic_approx + 1 + ds->dim;

      for (i = 0; i<ds->dim; i++)
      { for (j = 0; j<ds->dim; j++)
        { if (energy) 
          { *energy += inv_temp * (ds->q[i]-b[i]) * (ds->q[j]-b[j]) * *V / 2;
          }
          if (gr)
          { gr[i] += inv_temp * (ds->q[j]-b[j]) * *V / 2;
            gr[j] += inv_temp * (ds->q[i]-b[i]) * *V / 2;
          }
          V += 1;
        }
      }
    }

    else /* Not approximated by quadratic */
    {
      if (N_approx==1 || gr==0)
      {
#       if __CUDACC__
        { 
          int max_cases_per_launch = perthrd*blksize*numblks;

          i = 0;
          while (i < N_train)
          { 
            int c = N_train-i < max_cases_per_launch ? N_train-i
                     : max_cases_per_launch;

            int thrds = (c + perthrd - 1) / perthrd;
            int blks = (thrds + blksize - 1) / blksize;

            if (0)
            { printf("Launching with <<<%d,%d>>>, %d cases per thread\n",
                      blks,blksize,perthrd);
            }

            check_cuda_error (cudaGetLastError(), 
                              "Before launching many_cases");
            many_cases <<<blks, blksize>>> 
                       (energy ? thread_energy : 0, 
                        gr ? thread_grad : 0, 
                        i, perthrd, inv_temp, inv_temp);
            check_cuda_error (cudaDeviceSynchronize(), 
                              "Synchronizing after launching many_cases");
            check_cuda_error (cudaGetLastError(), 
                              "After synchronizing with many_cases");

            if (energy)
            { for (j = 0; j<blks; j++)
              { *energy += thread_energy[j*blksize];
              }
            }

            if (gr)
            { net_params *np = thread_grad;
#             if USE_SIMD_INTRINSICS && __AVX__
              { for (j = 0; j<blks; j++)
                { unsigned k;
                  unsigned e = grad.total_params & ~(unsigned)0x3;
                  for (k = 0; k < e; k += 4)
                  { _mm256_store_pd (grad.param_block+k, 
                                     _mm256_add_pd (
                                       _mm256_loadu_pd (grad.param_block+k),
                                       _mm256_loadu_pd (np->param_block+k)));
                  }
                  for (; k < grad.total_params; k++)
                  { grad.param_block[k] += np->param_block[k];
                  }
                  np += blksize;
                }
              }
#             else
              { for (j = 0; j<blks; j++)
                { unsigned k;
                  for (k = 0; k < grad.total_params; k++)
                  { grad.param_block[k] += np->param_block[k];
                  }
                  np += blksize;
                }
              }
#             endif
            }

            i += c;
          }
        }
#       else
        { for (i = 0; i<N_train; i++)
          { one_case (energy, gr ? &grad : 0, i, inv_temp, inv_temp);
          }
        }
#       endif
      }
      else /* We're using multiple approximations */
      {
        if (approx_count) /* There's a file saying how to do approximations */
        { 
          low  = (approx_count * (w_approx-1)) / N_approx;
          high = (approx_count * w_approx) / N_approx;

          for (j = low; j<high; j++)
          { i = approx_case[j] - 1;
            one_case(0, &grad, i, 1, (double)inv_temp*N_approx/approx_times[i]);
          }

          if (energy)
          { for (i = 0; i<N_train; i++)
            { one_case (energy, 0, i, inv_temp, 1);
            }
          }
        }
        else /* There's no file saying how to do approximations */
        {
          low  = (N_train * (w_approx-1)) / N_approx;
          high = (N_train * w_approx) / N_approx;

          if (energy)    
          { for (i = 0; i<low; i++)
            { one_case (energy, 0, i, inv_temp, 1);
            }
          }

          for (i = low; i<high; i++)
          { one_case (energy, &grad, i, inv_temp, inv_temp*N_approx);
          }

          if (energy)    
          { for (i = high; i<N_train; i++)
            { one_case (energy, 0, i, inv_temp, 1);
            }
          }
        }
      }
    }
  }
}


/* SAMPLE FROM DISTRIBUTION AT INVERSE TEMPERATURE OF ZERO.  Returns zero
   if this is not possible. */

int mc_app_zero_gen
( mc_dynamic_state *ds	/* Current dynamical state */
)
{ 
  net_prior_generate (&params, &sigmas, arch, flgs, model, priors, 0, 0, 0);

  return 1;
}


/* SET STEPSIZES FOR EACH COORDINATE. */

void mc_app_stepsizes
( mc_dynamic_state *ds	/* Current dynamical state */
)
{ 
  double inv_temp, w;
  int i, j, k, l;

  inv_temp = !ds->temp_state ? 1 : ds->temp_state->inv_temp;

  /* Find "typical" squared values for hidden units. */

  for (l = 0; l<arch->N_layers; l++)
  { net_value *typl = typical.h[l];
    double alpha, var_adj;
    if (TYPICAL_VALUES_ALL_ONE)
    { for (j = 0; j<arch->N_hidden[l]; j++)
      { typl[j] = 1;
      }
    }
    else
    { for (j = 0; j<arch->N_hidden[l]; j++)
      { typl[j] = 0;
      }
      if (arch->has_bh[l])
      { alpha = priors->bh[l].alpha[1];
        var_adj = alpha==0 ? 1 : alpha<3 ? 3 : alpha/(alpha-2);
        if (arch->bias_config[l])
        { for (k = 0; k<arch->bias_config[l]->N_conn; k++)
          { j = arch->bias_config[l]->conn[k].d;
            typl[j] += var_adj * sq(*sigmas.bh_cm[l]);
          }
        }
        else
        { for (j = 0; j<arch->N_hidden[l]; j++)
          { typl[j] += var_adj * sq(*sigmas.bh_cm[l]);
          }
        }
      }
      if (arch->has_ih[l])
      { alpha = priors->ih[l].alpha[2];
        var_adj = alpha==0 ? 1 : alpha<3 ? 3 : alpha/(alpha-2);
        if (arch->input_config[l])
        { for (k = 0; k<arch->input_config[l]->N_conn; k++)
          { i = arch->input_config[l]->conn[k].s;
            j = arch->input_config[l]->conn[k].d;
            typl[j] += var_adj * (train_sumsq[i]/N_train)*sq(*sigmas.ih_cm[l]);
          }
        }
        else if (flgs && flgs->any_omitted[l])
        { for (j = 0; j<arch->N_hidden[l]; j++)
          { for (i = 0; i<arch->N_inputs; i++)
            { if (flgs->omit[i] & (1<<(l+1)))
              { continue;
              }
              typl[j] += var_adj * (train_sumsq[i]/N_train)*sq(sigmas.ih[l][i]);
            }
          }
        }
        else
        { for (j = 0; j<arch->N_hidden[l]; j++)
          { for (i = 0; i<arch->N_inputs; i++)
            { typl[j] += var_adj * (train_sumsq[i]/N_train)*sq(sigmas.ih[l][i]);
            }
          }
        }
      }
      if (l>0 && arch->has_hh[l-1])
      { alpha = priors->hh[l-1].alpha[2];
        var_adj = alpha==0 ? 1 : alpha<3 ? 3 : alpha/(alpha-2);
        if (arch->hidden_config[l])
        { for (k = 0; k<arch->hidden_config[l]->N_conn; k++)
          { i = arch->hidden_config[l]->conn[k].s;
            j = arch->hidden_config[l]->conn[k].d;
            typl[j] += var_adj * sq (typical.h[l-1][i] * *sigmas.hh_cm[l-1]);
          }
        }
        else
        { for (j = 0; j<arch->N_hidden[l]; j++)
          { for (i = 0; i<arch->N_hidden[l-1]; i++)
            { typl[j] += var_adj * sq (typical.h[l-1][i] * sigmas.hh[l-1][i]);
            }
          }
        }
      }
      if (arch->has_ah[l])
      { for (j = 0; j<arch->N_hidden[l]; j++)
        { typl[j] *= sq (sigmas.ah[l][j]);
        }
      }
    }

    if (!TYPICAL_VALUES_ALL_ONE)
    { if (flgs==0 || flgs->layer_type[l]==Tanh_type)
      { for (j = 0; j<arch->N_hidden[l]; j++)
        { if (typl[j]>1)
          { typl[j] = 1;
          }
        }
      }
      if (flgs!=0 && flgs->layer_type[l]==Sin_type)
      { for (j = 0; j<arch->N_hidden[l]; j++)
        { if (typl[j]>2)
          { typl[j] = 2;
          }
        }
      }
    }
  }

  /* Compute estimated second derivatives of minus log likelihood for
     unit values. */

  net_model_max_second (seconds.o, arch, model, surv, &sigmas);

  if (inv_temp!=1)
  { for (i = 0; i<arch->N_outputs; i++)
    { seconds.o[i] *= inv_temp;
    }
  }

  for (l = arch->N_layers-1; l>=0; l--)
  { 
    for (i = 0; i<arch->N_hidden[l]; i++)
    { seconds.h[l][i] = 0;
    }

    for (i = 0; i<arch->N_hidden[l]; i++)
    { if (arch->has_ho[l])
      { int kk = 2*arch->N_layers-1-l;
        if (arch->hidden_config[kk])
        { w = *sigmas.ho_cm[l];
          for (k = 0; k<arch->hidden_config[kk]->N_conn; k++)
          { i = arch->hidden_config[kk]->conn[k].s;
            j = arch->hidden_config[kk]->conn[k].d;
            seconds.h[l][i] += (w*w) * seconds.o[j];
          }
        }
        else
        { for (j = 0; j<arch->N_outputs; j++)
          { w = sigmas.ho[l][i];
            if (sigmas.ao!=0) w *= sigmas.ao[j];
            seconds.h[l][i] += (w*w) * seconds.o[j];
          }
        }
      }
    }

    if (l<arch->N_layers-1 && arch->has_hh[l])
    { if (arch->hidden_config[l+1])
      { w = *sigmas.hh_cm[l];
        for (k = 0; k<arch->hidden_config[l+1]->N_conn; k++)
        { i = arch->hidden_config[l+1]->conn[k].s;
          j = arch->hidden_config[l+1]->conn[k].d;
          seconds.h[l][i] += (w*w) * seconds.s[l+1][j];
        }
      }
      else
      { for (i = 0; i<arch->N_hidden[l]; i++)
        { for (j = 0; j<arch->N_hidden[l+1]; j++)
          { w = sigmas.hh[l][i];
            if (sigmas.ah[l+1]!=0) w *= sigmas.ah[l+1][j];
            seconds.h[l][i] += (w*w) * seconds.s[l+1][j];
          }
        }
      }
    }

    for (i = 0; i<arch->N_hidden[l]; i++)
    { switch (flgs==0 ? Tanh_type : flgs->layer_type[l])
      { case Tanh_type: 
        case Identity_type:
        case Softplus_type:
        { seconds.s[l][i] = seconds.h[l][i];
          break;
        }
        case Sin_type:
        { seconds.s[l][i] = 2*seconds.h[l][i];
          break;
        }
        default: abort();
      }
    }
  }

  if (arch->has_ti)
  { 
    for (i = 0; i<arch->N_inputs; i++)
    { seconds.i[i] = 0;
    }

    for (l = 0; l<arch->N_layers; l++)
    { if (arch->has_ih[l])
      { if (arch->input_config[l])
        { w = *sigmas.ih_cm[l];
          for (k = 0; k<arch->input_config[l]->N_conn; k++)
          { i = arch->input_config[l]->conn[k].s;
            j = arch->input_config[l]->conn[k].d;
            seconds.i[i] += (w*w) * seconds.s[l][j];
          }
        }
        else
        { for (i = 0; i<arch->N_inputs; i++)
          { if  (flgs==0 || (flgs->omit[i]&(1<<(l+1)))==0)
            for (j = 0; j<arch->N_hidden[l]; j++)
            { w = sigmas.ih[l][i];
              if (sigmas.ah[l]!=0) w *= sigmas.ah[l][j];
              seconds.i[i] += (w*w) * seconds.s[l][j];
            }
          }
        }
      }
    }

    if (arch->has_io)
    { if (arch->input_config[arch->N_layers])
      { w = *sigmas.io;
        for (k = 0; k<arch->input_config[arch->N_layers]->N_conn; k++)
        { i = arch->input_config[arch->N_layers]->conn[k].s;
          j = arch->input_config[arch->N_layers]->conn[k].d;
          seconds.i[i] += (w*w) * seconds.o[j];
        }
      }
      else
      { for (i = 0; i<arch->N_inputs; i++)
        { if (flgs==0 || (flgs->omit[i]&1)==0)
          for (j = 0; j<arch->N_outputs; j++)
          { w = sigmas.io[i];
            if (sigmas.ao!=0) w *= sigmas.ao[j];
            seconds.i[i] += (w*w) * seconds.o[j];
          }
        }
      }
    }
  }

  /* Initialize stepsize variables to second derivatives of minus log prior. */

  net_prior_max_second (&stepsizes, &sigmas, arch, flgs, priors);

  /* Add second derivatives of minus log likelihood to stepsize variables. */

  if (arch->has_ti)
  { for (i = 0; i<arch->N_inputs; i++)
    { stepsizes.ti[i] += N_train * seconds.i[i];
    }
  }

  for (l = 0; l<arch->N_layers; l++)
  {
    if (arch->has_th[l])
    { for (i = 0; i<arch->N_hidden[l]; i++)
      { stepsizes.th[l][i] += N_train * seconds.h[l][i];
      }
    }

    if (arch->has_bh[l])
    { if (arch->bias_config[l])
      { for (k = 0; k<arch->bias_config[l]->N_conn; k++)
        { j = arch->bias_config[l]->conn[k].d;
          stepsizes.bh [l] [arch->bias_config[l]->conn[k].w] 
            += N_train * seconds.s[l][j];
        }
      }
      else
      { for (j = 0; j<arch->N_hidden[l]; j++)
        { stepsizes.bh[l][j] += N_train * seconds.s[l][j];
        }
      }
    }

    if (arch->has_ih[l])
    { if (arch->input_config[l])
      { for (k = 0; k<arch->input_config[l]->N_conn; k++)
        { i = arch->input_config[l]->conn[k].s;
          j = arch->input_config[l]->conn[k].d;
          stepsizes.ih [l] [arch->input_config[l]->conn[k].w]
            += train_sumsq[i] * seconds.s[l][j];
        }
      }
      else
      { k = 0;
        for (i = 0; i<arch->N_inputs; i++)
        { if (flgs==0 || (flgs->omit[i]&(1<<(l+1)))==0)
          { for (j = 0; j<arch->N_hidden[l]; j++)
            { stepsizes.ih [l] [k*arch->N_hidden[l] + j] 
                += train_sumsq[i] * seconds.s[l][j];
            }
            k += 1;
          }
        }
      }
    }
 
    if (l<arch->N_layers-1 && arch->has_hh[l])
    { if (arch->hidden_config[l+1])
      { for (k = 0; k<arch->hidden_config[l+1]->N_conn; k++)
        { i = arch->hidden_config[l+1]->conn[k].s;
          j = arch->hidden_config[l+1]->conn[k].d;
          stepsizes.hh [l] [arch->hidden_config[l+1]->conn[k].w]
            += N_train * typical.h[l][i] * seconds.s[l+1][j];
        }
      }
      else
      { for (i = 0; i<arch->N_hidden[l]; i++)
        { net_value pv = N_train * typical.h[l][i];
          for (j = 0; j<arch->N_hidden[l+1]; j++)
          { stepsizes.hh [l] [i*arch->N_hidden[l+1] + j] 
              += pv * seconds.s[l+1][j];
          }
        }
      }
    }

    if (arch->has_ho[l])
    { int kk = 2*arch->N_layers-1-l;
      if (arch->hidden_config[kk])
      { for (k = 0; k<arch->hidden_config[kk]->N_conn; k++)
        { i = arch->hidden_config[kk]->conn[k].s;
          j = arch->hidden_config[kk]->conn[k].d;
          stepsizes.ho [l] [arch->hidden_config[kk]->conn[k].w]
            += N_train * typical.h[l][i] * seconds.o[j];
        }
      }
      else
      { for (i = 0; i<arch->N_hidden[l]; i++)
        { net_value pv = N_train * typical.h[l][i];
          for (j = 0; j<arch->N_outputs; j++)
          { stepsizes.ho [l] [i*arch->N_outputs + j] += pv * seconds.o[j];
          }
        }
      }
    }
  }

  if (arch->has_io)
  { if (arch->input_config[arch->N_layers])
    { for (k = 0; k<arch->input_config[arch->N_layers]->N_conn; k++)
      { i = arch->input_config[arch->N_layers]->conn[k].s;
        j = arch->input_config[arch->N_layers]->conn[k].d;
        stepsizes.io [arch->input_config[arch->N_layers]->conn[k].w]
          += train_sumsq[i] * seconds.o[j];
      }
    }
    else
    { k = 0;
      for (i = 0; i<arch->N_inputs; i++)
      { if (flgs==0 || (flgs->omit[i]&1)==0)
        { for (j = 0; j<arch->N_outputs; j++)
          { stepsizes.io [k*arch->N_outputs + j] += train_sumsq[i]*seconds.o[j];
          }
          k += 1;
        }
      }
    }
  }

  if (arch->has_bo)
  { for (j = 0; j<arch->N_outputs; j++)
    { stepsizes.bo[j] += N_train * seconds.o[j];
    }
  }

  /* Convert from second derivatives to appropriate stepsizes. */

  for (k = 0; k<ds->dim; k++)
  { ds->stepsize[k] = 1/sqrt(ds->stepsize[k]);
  }
}


/* COMPUTE ADJUSTED SUM OF SQUARES OF A SET OF PARAMETERS. */

static double sum_squares
( net_param *wt,	/* Parameters to compute sum of squares for */
  net_sigma *adj,	/* Adjustments, or zero */
  int n			/* Number of of parameters */
)
{
  double sum_sq, d;
  int i;

  sum_sq = 0;

  if (adj==0)
  { for (i = 0; i<n; i++)
    { d = wt[i];
      sum_sq += d*d;
    }
  }
  else
  { for (i = 0; i<n; i++)
    { d = wt[i] / adj[i];
      sum_sq += d*d;
    }
  }

  return sum_sq;
}


/* RANDOM-GRID METROPOLIS UPDATE FOR AN UPPER SIGMA VALUE.  The value is
   taken in and returned in standard deviation form, but the Metropolis
   update is done in log precision form. */

static double rgrid_sigma
( double stepsize,	/* Stepsize for update */
  mc_iter *it,
  double current,	/* Current value of hyperparameter */
  double width,		/* Width parameter for top-level prior */
  double alpha0,	/* Alpha for top-level prior */
  double alpha1,	/* Alpha for lower-level prior */
  double sum,		/* Sum of lower-level precisions */
  int n			/* Number of lower-level precision values */
)
{
  double logcur, lognew, U;
  double Ecur, Enew;  
  double w, a;

  w  = 1 / (width * width);
  a  = alpha0 - n*alpha1;

  logcur = -2.0 * log(current);

  Ecur = -logcur*a/2 + exp(logcur)*alpha0/(2*w) + exp(-logcur)*alpha1*sum/2;

  U = rand_uniopen() - 0.5;
  lognew = (2*stepsize) * (U + floor (0.5 + logcur/(2*stepsize) - U));

  Enew = -lognew*a/2 + exp(lognew)*alpha0/(2*w) + exp(-lognew)*alpha1*sum/2;

  it->proposals += 1;
  it->delta = Enew - Ecur;

  U = rand_uniform(); /* Do every time to keep in sync for coupling purposes */

  if (U<exp(-it->delta))
  { 
    it->move_point = 1;
    logcur = lognew;
  }
  else
  { 
    it->rejects += 1;
    it->move_point = 0;
  }

  return exp (-0.5*logcur);
}
