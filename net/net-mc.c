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
static int numblks = DEFAULT_NUMBLKS;	/* Max number of blocks per kernel */

static int n_launches;			/* Number of launches needed to 
                                           handle all training cases */
static int max_cases_per_launch;        /* Largest number of cases handled by
					   one CUDA kernel launch */
static int max_blocks_per_launch;	/* Largest number of blocks for one
                                           CUDA kernel launch */
static int max_threads_per_launch;	/* Largest number of threads for one
                                           CUDA kernel launch */
#endif


/* FORWARD DECLARATION OF CUDA KERNEL. */

#if __CUDACC__

__global__ void many_cases 
(
  double *thread_energy,   /* Places to store energy, null if not required */
  net_params *thread_grad, /* Places to store gradient, null if not required */
  int start,		/* Start of cases to look at */
  int cases_per_thread,	/* Number of case to handle in one thread */
  double en_weight,	/* Weight for this case for energy */
  double gr_weight	/* Weight for this case for gradient */
);

#endif


/* FUNCTION TO SQUARE ITS ARGUMENT. */

static inline net_value sq (net_value x) { return x*x; }


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
static net_sigma *noise;	/* Just the noise hyperparameters in sigmas */

static net_params params;	/* Pointers to parameters, which are position
				   coordinates for dynamical Monte Carlo */

static net_values *deriv;	/* Derivatives for training cases */

static int approx_count;	/* Number of entries in approx-file, 0 if none*/

static int *approx_case; 	/* Data on how approximations are to be done  */
static int *approx_times;	/*   as read from approx_file                 */

static double *quadratic_approx;/* Quadratic approximation to log likelihood  */

static net_params stepsizes;	/* Pointers to stepsizes */
static net_values seconds;	/* Second derivatives */
static net_value *train_sumsq;	/* Sums of squared training input values */
static net_values typical;	/* Typical squared values for hidden units */

static net_params grad;		/* Pointers to gradient for network parameters*/

/* Values used or computed by threads, in managed, device, or constant memory.  
   Sometimes, pointers are in constant memory, but what they point to is not. */

#if __CUDACC__

static net_arch dev_arch;	/* Copy of arch with GPU config pointers */

static double *thread_energy;	/* Energies computed by concurrent threads,
                                   points to GPU memory */
static net_params *thread_grad;	/* Gradients computed by concurrent threads,
                                   points to GPU memory */

static double *block_energy;	/* Energies computed by thread blocks (CPU) */
static double *dev_block_energy;/*    - version in GPU */

static net_params *block_grad;	/* Gradients computed by thread blocks */
static net_params *dev_block_grad; /* - version in GPU */
static net_param *dev_block_grad_params; /* Parameter block of dev_block_grad */
static unsigned block_aligned_total;     /* Aligned size for each case */

static net_value *dev_train_targets; /* Copy of train_targets in GPU memory */
static net_values *dev_train_values; /* Value structures in GPU memory */

static net_values *dev_deriv;	/* GPU copy  of derivatives for training cases*/

__constant__ int const_N_train;    /* Copy of N_train in constant memory */
__constant__ int const_N_inputs;   /* Copy of N_inputs in constant memory */
__constant__ int const_N_targets;  /* Copy of N_targets in constant memory */

__constant__ net_arch const_arch;  /* Copy of dev_arch in constant memory */
__constant__ net_flags const_flgs; /* Copy of flgs in constant memory */
__constant__ int const_has_flgs;   /* Are flags present in const_flgs? */

__constant__ model_specification const_model;  /* Constant copy of model */
__constant__ model_survival const_surv;  /* Constant copy of surv */

static net_sigma *dev_noise;          /* GPU copy of noise sigmas */
__constant__ net_sigma *const_noise;  /* Pointer to GPU copy of noise sigmas */

__constant__ net_params const_params; /* version of params in GPU */
static net_param *dev_param_block;    /* parameter block in const_params */

__constant__ net_values *const_deriv;  /* Copy of deriv ptr in constant memory*/
__constant__ net_values *const_train_values; /* Const copy of train_values ptr*/
__constant__ net_value *const_train_targets; /* Const copy of train_targets */

__constant__ double *const_block_energy;   /* Copy of dev_block_energy ptr */
__constant__ net_params *const_block_grad; /* Copy of dev_block_grad ptr */

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
    if (FP32!=0 && FP32!=1 || FP64!=0 && FP64 !=1 || FP32!=!FP64)
    { fprintf (stderr, "Invalid FP32/FP64 setting!\n");
      exit(1);
    }
  
    if (FP64 ? sizeof(mc_value)!=8 || sizeof(net_param)!=8 
                 || sizeof(net_value)!=8 || sizeof(data_value)!=8
             : sizeof(mc_value)!=4 || sizeof(net_param)!=4 
                 || sizeof(net_value)!=4 || sizeof(data_value)!=4)
    { fprintf (stderr, "Sizes don't match FP32/FP64 setting!\n");
      exit(1);
    }

    char *e =  getenv("SHOW_INFO");
    int show_info = e!=0 && strcmp(e,"false")!=0
                         && strcmp(e,"FALSE")!=0
                         && strcmp(e,"0")!=0;

    if (show_info)
    { fprintf (stderr, "Precision: %s, SIMD features: %s\n",
               FP32 ? "FP32" : "FP64",
#              if __AVX2__ && __FMA__
                 "SSE2 SSE4.2 AVX AVX2 FMA");
#              elif __AVX2__
                 "SSE2 SSE4.2 AVX AVX2");
#              elif __AVX__
                 "SSE2 SSE4.2 AVX");
#              elif __SSE4_2__
                 "SSE2 SSE4.2");
#              elif __SSE2__
                 "SSE2");
#              elif __SSE2__
                 "SSE2");
#              else
                 "none");
#              endif
    }

#   if __CUDACC__
    {
      if (show_info)
      { struct cudaDeviceProp cuda_prop;
        check_cuda_error (cudaGetDeviceProperties(&cuda_prop,0),
                          "Get properties");
        fprintf (stderr,
          "%s, Compute Capability %d.%d, %d processors, %.1f GBytes%s\n",
          cuda_prop.name, cuda_prop.major, cuda_prop.minor,
          cuda_prop.multiProcessorCount, 
          (double)cuda_prop.totalGlobalMem/1024/1024/1024,
          cuda_prop.ECCEnabled ? " ECC" : "");
      }

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

      if (show_info)
      { printf (
         "Specified %d cases per thread, %d threads per block, %d blocks max\n",
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
    params.total_params = net_setup_param_count(arch,flgs);  /* also config */
  
    sigmas.sigma_block = (net_sigma *) logg->data['S'];
    params.param_block = (net_param *) logg->data['W'];
  
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
        (net_sigma *) chk_alloc (sigmas.total_sigmas, sizeof (net_sigma));

      params.param_block = 
        (net_param *) chk_alloc (params.total_params, sizeof (net_param));

      net_setup_sigma_pointers (&sigmas, arch, flgs, model);
      net_setup_param_pointers (&params, arch, flgs);
   
      net_prior_generate (&params, &sigmas, arch, flgs, model, priors, 1, 0, 0);
    }

    /* Make copy of architecture with config pointers going to GPU memory. */

#   if __CUDACC__
    { int l;
      dev_arch = *arch;
      for (l = 0; l<=arch->N_layers; l++)
      { if (arch->input_config[l])
        { dev_arch.input_config[l] = net_config_to_gpu (arch->input_config[l]);
        }
        if (arch->bias_config[l])
        { dev_arch.bias_config[l] = net_config_to_gpu (arch->bias_config[l]);
        }
      }
      for (l = 1; l<2*arch->N_layers; l++)
      { if (arch->hidden_config[l])
        { dev_arch.hidden_config[l] = net_config_to_gpu(arch->hidden_config[l]);
        }
      }
    }
#   endif    

    /* Set up noise sigmas in CPU and GPU memory. */

    noise = sigmas.noise;

#   if __CUDACC__
    { if (sigmas.noise != 0)
      { check_cuda_error (cudaMalloc (&dev_noise, 
                                      arch->N_outputs * sizeof *dev_noise),
                          "alloc of dev_noise");
      }
    }
#   endif    

    /* Set up 'params' structure in GPU memory. */

#   if __CUDACC__
    { net_params tmp_params;
      tmp_params.total_params = params.total_params;
      check_cuda_error (cudaMalloc (&dev_param_block,
                          params.total_params * sizeof *tmp_params.param_block),
                        "alloc of params block for GPU");
      tmp_params.param_block = dev_param_block;
      net_setup_param_pointers (&tmp_params, arch, flgs);
      check_cuda_error (cudaMemcpyToSymbol 
                          (const_params, &tmp_params, sizeof tmp_params),
                        "copy to const_params");
    }
#   endif

    /* Set up stepsize structure. */
  
    stepsizes.total_params = params.total_params;
    stepsizes.param_block = (net_param *) chk_alloc
     (params.total_params, sizeof *stepsizes.param_block);
  
    net_setup_param_pointers (&stepsizes, arch, flgs);

    /* Find number of network values. */

    value_count = net_setup_value_count(arch);

    /* Set up second derivative and typical value structures. */

    value_block = (net_value *) chk_alloc (value_count, sizeof *value_block);
    net_setup_value_pointers (&seconds, value_block, arch, 0);

    value_block = (net_value *) chk_alloc (value_count, sizeof *value_block);
    net_setup_value_pointers (&typical, value_block, arch, 0);
  
    /* Read training data, if any, and allocate space for derivatives. */
  
    data_spec = (data_specifications *) logg->data['D'];

    if (data_spec!=0 && model==0)
    { fprintf(stderr,"No model specified for data\n");
      exit(1);
    }

    if (data_spec && logg->actual_size['D'] !=
                       data_spec_size(data_spec->N_inputs,data_spec->N_targets))
    { fprintf(stderr,"Data specification record is the wrong size!\n");
      exit(1);
    }

    train_sumsq = (net_value *) chk_alloc (arch->N_inputs, sizeof *train_sumsq);
    for (j = 0; j<arch->N_inputs; j++) train_sumsq[j] = 0;
  
    if (data_spec!=0)
    { 
      net_data_read (1, 0, arch, model, surv);
    
      deriv = (net_values *) chk_alloc (N_train, sizeof *deriv);
    
      value_block = 
        (net_value *) chk_alloc (value_count*N_train, sizeof *value_block);
    
      for (i = 0; i<N_train; i++) 
      { net_setup_value_pointers(&deriv[i], value_block+value_count*i, arch, 0);
      }
    
      for (j = 0; j<arch->N_inputs; j++)
      { for (i = 0; i<N_train; i++)
        { train_sumsq[j] += sq (train_values[i].i[j]);
        }
      }

      if (model!=0 && model->type=='V' && surv->hazard_type!='C')
      {
        net_value tsq;
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

    /* Copy training inputs and training targets to GPU memory, and allocate
       space on GPU for values in all training cases, with pointers set up. */

#   if __CUDACC__
    { 
      size_t sz;
      int i;

      net_values *tmp_values;
      tmp_values = (net_values *) chk_alloc (N_train, sizeof *tmp_values);

      int value_count = net_setup_value_count(arch) - N_inputs;
      net_value *iblk, *vblk;

      check_cuda_error (cudaGetLastError(), "Before copying to data to GPU");

      sz = N_inputs * N_train * sizeof *iblk;
      check_cuda_error (cudaMalloc (&iblk, sz), "cudaMalloc of iblk");
      check_cuda_error (cudaMemcpy 
          (iblk, train_iblock, sz, cudaMemcpyHostToDevice),
        "copy to iblk");

      sz = value_count * N_train * sizeof *vblk;
      check_cuda_error (cudaMalloc (&vblk, sz), "cudaMalloc of vblk for train");

      for (i = 0; i<N_train; i++) 
      { net_setup_value_pointers (&tmp_values[i], vblk+value_count*i, arch, 
                                  iblk+N_inputs*i);
      }

      sz = N_train * sizeof *dev_train_values;
      check_cuda_error (cudaMalloc (&dev_train_values, sz), 
                        "cudaMalloc of dev_train_values");
      check_cuda_error (cudaMemcpy 
          (dev_train_values, tmp_values, sz, cudaMemcpyHostToDevice),
        "copy to dev_train_values");
      
      sz = N_targets * N_train * sizeof *dev_train_targets;
      check_cuda_error (cudaMalloc (&dev_train_targets, sz),
                        "cudaMalloc of dev_train_targets");
      check_cuda_error (cudaMemcpy
          (dev_train_targets, train_targets, sz, cudaMemcpyHostToDevice),
        "copy to dev_train_targets");

      sz = value_count * N_train * sizeof *vblk;
      check_cuda_error (cudaMalloc (&vblk, sz), "cudaMalloc of vblk for deriv");

      for (i = 0; i<N_train; i++) 
      { net_setup_value_pointers (&tmp_values[i], vblk+value_count*i, arch, 
                                  iblk+N_inputs*i);
      }

      sz = N_train * sizeof *dev_deriv;
      check_cuda_error (cudaMalloc (&dev_deriv, sz),
                        "cudaMalloc of dev_deriv");
      check_cuda_error (cudaMemcpy 
          (dev_deriv, tmp_values, sz, cudaMemcpyHostToDevice),
        "copy to dev_deriv");

      free(tmp_values);
    }
#   endif

    /* Copy some data to constant memory in the GPU, if using CUDA. 
       Constant memory is limited, so nothing that could be large. */

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
      cudaMemcpyToSymbol (const_arch, &dev_arch, sizeof dev_arch);
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
      cudaMemcpyToSymbol (const_noise, &dev_noise, sizeof dev_noise);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_noise");
      cudaMemcpyToSymbol (const_deriv, &dev_deriv, sizeof deriv);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_deriv");
      cudaMemcpyToSymbol
        (const_train_values, &dev_train_values, sizeof dev_train_values);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_train_values");
      cudaMemcpyToSymbol
        (const_train_targets, &dev_train_targets, sizeof dev_train_targets);
      check_cuda_error (cudaGetLastError(), 
                        "After copying to const_train_targets");
    }
#   endif

    /* Figure out stuff about blocksizes and numbers of blocks for
       launching of CUDA kernels. */

#   if __CUDACC__
    { n_launches 
        = (N_train + perthrd*blksize*numblks - 1) / (perthrd*blksize*numblks);
      max_cases_per_launch 
        = (N_train + n_launches - 1) / n_launches;
      max_blocks_per_launch 
        = (max_cases_per_launch + perthrd*blksize - 1) / (perthrd*blksize);
      max_threads_per_launch = max_blocks_per_launch * blksize;
      max_cases_per_launch = max_threads_per_launch * perthrd;
      if (show_info)
      { printf ("With %d cases, need %d launches, max %d blocks/launch\n",
                N_train, n_launches, max_blocks_per_launch);
      }
    }

    /* Set GPU to use memory for L1 cache rather than for shared memory when
       executing the many_cases kernel (which doesn't use shared memory). */

    check_cuda_error (
      cudaFuncSetCacheConfig (many_cases, cudaFuncCachePreferL1),
      "Set cache config");

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
#define noise		const_noise
#define params		const_params
#define deriv		const_deriv
#define train_values	const_train_values
#define train_targets	const_train_targets

#endif

#define DEBUG_ONE_CASE 0

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

  if (DEBUG_ONE_CASE)
  { printf("Starting one_case, for case %d\n",i);
  }

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
      
      net_value fudged_target 
                  = ot>t1 ? -(t1-t0) : censored ? -(ot-t0) : (ot-t0);

      double log_prob;
  
      net_model_prob(&train_values[i], &fudged_target,
                     &log_prob, grd ? &deriv[i] : 0, arch, model, surv, 
                     noise, Cheap_energy);
  
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
    net_values *train_vals_i = train_values+i;
    net_values *deriv_i = grd ? deriv+i : 0;

    if (DEBUG_ONE_CASE)
    { printf("train_values[%d]->i[0] = %f\n",i,train_vals_i->i[0]);
      // printf("train_values[%d]->i[1] = %f\n",i,train_vals_i->i[1]);
    }

    net_func (train_vals_i, 0, arch, flgs, &params);

    if (DEBUG_ONE_CASE)
    { printf("train_values[%d]->o[0] = %f\n",i,train_vals_i->o[0]);
    }

    double log_prob;
    net_model_prob (train_vals_i, train_targets+N_targets*i, &log_prob, 
                    deriv_i, arch, model, surv, noise, Cheap_energy);

    if (DEBUG_ONE_CASE)
    { printf("log_prob = %f\n",log_prob);
    }
    
    if (energy)
    { *energy -= en_weight * log_prob;
    }

    if (grd)
    { 
      if (gr_weight!=1)
      { for (k = 0; k<arch->N_outputs; k++)
        { deriv_i->o[k] *= gr_weight;
        }
      }

      net_back (train_vals_i, deriv_i, arch->has_ti ? -1 : 0,
                arch, flgs, &params);

      net_grad (grd, &params, train_vals_i, deriv_i, arch, flgs);
    }
  }
}

#if __CUDACC__ && __CUDA_ARCH__  /* Compiling for GPU */

#undef N_targets
#undef arch
#undef flgs
#undef model
#undef surv
#undef noise
#undef params
#undef deriv
#undef train_values
#undef train_targets

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
  unsigned total_params = const_params.total_params;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = start + cases_per_thread * i;

  if (j < const_N_train) 
  { 
    double *restrict threi;
    net_params *restrict thrgi;
    
    if (thread_energy)
    { 
      threi = threadIdx.x==0 ? const_block_energy + blockIdx.x 
                             : thread_energy + i;
      *threi = 0;
    }

    if (thread_grad)
    { 
      thrgi = threadIdx.x==0 ? const_block_grad + blockIdx.x 
                             : thread_grad + i;

      if (0)  /* seems to be slower */
      { memset(thrgi->param_block, 0, total_params * sizeof (net_param));
      }
      else
      { unsigned k;
        net_param *q = thrgi->param_block;
        for (k = 0; k < total_params; k++)
        { q[k] = 0;
        }
      }
    }

    for (int h = j; h < j+cases_per_thread && h < const_N_train; h++)
    { one_case (thread_energy ? threi : 0, 
                thread_grad ? thrgi : 0, 
                h, en_weight, gr_weight);
    }

    { int stride;
      net_param *restrict q;
      if (thread_grad) q = thrgi->param_block;
      for (stride = 1; stride < blockDim.x; stride <<= 1)
      { __syncthreads();
        if ((threadIdx.x & (2*stride-1)) == 0 
              && threadIdx.x + stride < blockDim.x
              && j + cases_per_thread*stride < const_N_train)
        { if (thread_energy)
          { *threi += thread_energy[i+stride];
          }
          if (thread_grad)
          { net_param *restrict p = thread_grad[i+stride].param_block;
            unsigned k;
            for (k = 0; k < total_params; k++)
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

  if (energy==0 && gr==0) return;  /* Nothing being returned -> nothing to do */

  inv_temp = !ds->temp_state ? 1 : ds->temp_state->inv_temp;

  if (gr && grad.param_block!=gr)
  { grad.param_block = gr;
    net_setup_param_pointers (&grad, arch, flgs);
  }

# if __CUDACC__
  { if (N_train>0)
    { 
      if (energy && thread_energy==0)
      { check_cuda_error (cudaMalloc (&thread_energy,
                            max_threads_per_launch * sizeof *thread_energy),
                          "alloc thread_energy");
        block_energy = (double *) 
          chk_alloc (max_blocks_per_launch, sizeof *block_energy);
        check_cuda_error (cudaMalloc (&dev_block_energy,
                             max_blocks_per_launch * sizeof *dev_block_energy),
                          "alloc of dev_block_energy");
        cudaMemcpyToSymbol
          (const_block_energy, &dev_block_energy, sizeof dev_block_energy);
        check_cuda_error (cudaGetLastError(), 
                          "After copying to const_block_energy");
      }

      if (gr && thread_grad==0)
      { 
        net_params *tmp_grad;

        /* Create thread_grad array on GPU. */

        tmp_grad = (net_params *) 
          chk_alloc (max_threads_per_launch, sizeof *tmp_grad);
        tmp_grad->total_params = grad.total_params;
        check_cuda_error (cudaMalloc 
           (&tmp_grad->param_block, max_threads_per_launch * grad.total_params
                                     * sizeof *tmp_grad->param_block),
         "alloc tmp_grad param block for thread_grad");
        net_setup_param_pointers (tmp_grad, arch, flgs);
        net_replicate_param_pointers(tmp_grad, arch, max_threads_per_launch,
                                     tmp_grad->total_params);
        check_cuda_error (cudaMalloc (&thread_grad,
                            max_threads_per_launch * sizeof *thread_grad),
                          "alloc thread_grad");
        check_cuda_error (cudaMemcpy (thread_grad, tmp_grad,
                            max_threads_per_launch * sizeof *thread_grad,
                            cudaMemcpyHostToDevice),
                          "cudaMemcpy to thread_grad");
        free(tmp_grad);

        /* Create block_grad array on CPU and on GPU. */

        block_aligned_total = (grad.total_params + 0x3f) & ~0x3f;

        block_grad = (net_params *) 
          chk_alloc (max_blocks_per_launch, sizeof *block_grad);
        block_grad->total_params = grad.total_params;
        block_grad->param_block = (net_param *) 
          chk_alloc (max_blocks_per_launch * block_aligned_total, 
                     sizeof *block_grad->param_block);
        net_setup_param_pointers (block_grad, arch, flgs);
        net_replicate_param_pointers (block_grad, arch, max_blocks_per_launch,
                                      block_aligned_total);

        tmp_grad = (net_params *) 
          chk_alloc (max_blocks_per_launch, sizeof *tmp_grad);
        tmp_grad->total_params = grad.total_params;
        check_cuda_error (cudaMalloc 
           (&dev_block_grad_params, max_blocks_per_launch * block_aligned_total
                                     * sizeof *tmp_grad->param_block),
         "alloc for dev_block_grad_params");
        tmp_grad->param_block = dev_block_grad_params;
        net_setup_param_pointers (tmp_grad, arch, flgs);
        net_replicate_param_pointers(tmp_grad, arch, max_blocks_per_launch,
                                     block_aligned_total);
        check_cuda_error (cudaMalloc (&dev_block_grad,
                            max_blocks_per_launch * sizeof *dev_block_grad),
                          "alloc dev_block_grad");
        check_cuda_error (cudaMemcpy (dev_block_grad, tmp_grad,
                            max_blocks_per_launch * sizeof *dev_block_grad,
                            cudaMemcpyHostToDevice),
                          "cudaMemcpy to dev_block_grad");
        free(tmp_grad);

        /* Put pointer to dev_block_grad array in constant GPU memory. */

        cudaMemcpyToSymbol
          (const_block_grad, &dev_block_grad, sizeof dev_block_grad);
        check_cuda_error (cudaGetLastError(), 
                          "After copying to const_block_grad");
      }
    }
  }
# endif

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
          check_cuda_error (cudaMemcpy (dev_param_block, params.param_block,
                              params.total_params * sizeof *params.param_block,
                              cudaMemcpyHostToDevice),
                            "Copying parameters to GPU");

          if (sigmas.noise != 0)
          { check_cuda_error (cudaMemcpy (dev_noise, sigmas.noise,
                                arch->N_outputs * sizeof *dev_noise,
                                cudaMemcpyHostToDevice),
                              "Copying noise sigmas to GPU");
          }

          i = 0;
          while (i < N_train)
          { 
            int c = N_train-i < max_cases_per_launch ? N_train-i
                     : max_cases_per_launch;

            int thrds = (c + perthrd - 1) / perthrd;
            int blks = (thrds + blksize - 1) / blksize;

            if (0)
            { static int first = 1;
              if (first)
              { printf("Launching with <<<%d,%d>>>, %d cases per thread\n",
                        blks,blksize,perthrd);
                first = 0;
              }
            }

            check_cuda_error (cudaGetLastError(), 
                              "Before launching many_cases");
            many_cases <<<blks, blksize>>> 
                       (energy ? thread_energy : 0, 
                        gr ? thread_grad : 0, 
                        i, perthrd, inv_temp, inv_temp);

#           if MANAGED_MEMORY_USED
            { 
              check_cuda_error (cudaDeviceSynchronize(), 
                                "Synchronizing after launching many_cases");
              check_cuda_error (cudaGetLastError(), 
                                "After synchronizing with many_cases");
            }
#           endif

            if (energy)
            { check_cuda_error (cudaMemcpy (block_energy, dev_block_energy,
                                            blks * sizeof *block_energy,
                                            cudaMemcpyDeviceToHost),
                                "copy to block_energy");
              double e = *energy;
              for (j = 0; j<blks; j++)
              { e += block_energy[j];
              }
              *energy = e;
            }

            if (gr)
            { 
              check_cuda_error (cudaMemcpy (
                 block_grad->param_block, dev_block_grad_params, 
                   blks * block_aligned_total * sizeof *block_grad->param_block,
                 cudaMemcpyDeviceToHost),
               "copy from GPU to block_grad");

              net_params *np = block_grad;

              for (j = 0; j<blks; j++)
              { net_param *restrict npb = np->param_block;
                unsigned k;
#               if FP32 && USE_SIMD_INTRINSICS && __AVX__
                { unsigned e = grad.total_params & ~(unsigned)0xf;
                  for (k = 0; k < e; k += 16)
                  { _mm256_storeu_ps (grad.param_block+k, 
                                      _mm256_add_ps (
                                        _mm256_loadu_ps (grad.param_block+k),
                                        _mm256_loadu_ps (npb+k)));
                    _mm256_storeu_ps (grad.param_block+k+8, 
                                      _mm256_add_ps (
                                        _mm256_loadu_ps (grad.param_block+k+8),
                                        _mm256_loadu_ps (npb+k+8)));
                  }
                  if (k < (grad.total_params & ~(unsigned)0x7))
                  { _mm256_storeu_ps (grad.param_block+k, 
                                      _mm256_add_ps (
                                        _mm256_loadu_ps (grad.param_block+k),
                                        _mm256_loadu_ps (npb+k)));
                    k += 8;
                  }
                  if (k < (grad.total_params & ~(unsigned)0x3))
                  { _mm_storeu_ps (grad.param_block+k, 
                                   _mm_add_ps (
                                     _mm_loadu_ps (grad.param_block+k),
                                     _mm_loadu_ps (npb+k)));
                    k += 4;
                  }
                  if (k < (grad.total_params & ~(unsigned)0x1))
                  { __m128 Z = _mm_setzero_ps();
                    _mm_storel_pi ((__m64 *)(grad.param_block+k), 
                              _mm_add_ps (
                                _mm_loadl_pi (Z, (__m64 *)(grad.param_block+k)),
                                _mm_loadl_pi (Z, (__m64 *)(npb+k))));
                    k += 2;
                  }
                  if (k < grad.total_params)
                  { grad.param_block[k] += npb[k];
                  }
                }
#               elif FP32 && USE_SIMD_INTRINSICS && __SSE2__
                { unsigned e = grad.total_params & ~(unsigned)0x7;
                  for (k = 0; k < e; k += 8)
                  { _mm_storeu_ps (grad.param_block+k, 
                                   _mm_add_ps (
                                     _mm_loadu_ps (grad.param_block+k),
                                     _mm_loadu_ps (npb+k)));
                    _mm_storeu_ps (grad.param_block+k+4, 
                                   _mm_add_ps (
                                     _mm_loadu_ps (grad.param_block+k+4),
                                     _mm_loadu_ps (npb+k+4)));
                  }
                  if (k < (grad.total_params & ~(unsigned)0x3))
                  { _mm_storeu_ps (grad.param_block+k, 
                                   _mm_add_ps (
                                     _mm_loadu_ps (grad.param_block+k),
                                     _mm_loadu_ps (npb+k)));
                    k += 4;
                  }
                  if (k < (grad.total_params & ~(unsigned)0x1))
                  { __m128 Z = _mm_setzero_ps();
                    _mm_storel_pi ((__m64 *)(grad.param_block+k), 
                              _mm_add_ps (
                                _mm_loadl_pi (Z, (__m64 *)(grad.param_block+k)),
                                _mm_loadl_pi (Z, (__m64 *)(npb+k))));
                    k += 2;
                  }
                  if (k < grad.total_params)
                  { grad.param_block[k] += npb[k];
                  }
                }
#               elif FP64 && USE_SIMD_INTRINSICS && __AVX__
                { unsigned e = grad.total_params & ~(unsigned)0x7;
                  for (k = 0; k < e; k += 8)
                  { _mm256_storeu_pd (grad.param_block+k, 
                                      _mm256_add_pd (
                                        _mm256_loadu_pd (grad.param_block+k),
                                        _mm256_loadu_pd (npb+k)));
                    _mm256_storeu_pd (grad.param_block+k+4, 
                                      _mm256_add_pd (
                                        _mm256_loadu_pd (grad.param_block+k+4),
                                        _mm256_loadu_pd (npb+k+4)));
                  }
                  if (k < (grad.total_params & ~(unsigned)0x3))
                  { _mm256_storeu_pd (grad.param_block+k, 
                                      _mm256_add_pd (
                                        _mm256_loadu_pd (grad.param_block+k),
                                        _mm256_loadu_pd (npb+k)));
                    k += 4;
                  }
                  if (k < (grad.total_params & ~(unsigned)0x1))
                  { _mm_storeu_pd (grad.param_block+k, 
                                   _mm_add_pd (
                                     _mm_loadu_pd (grad.param_block+k),
                                     _mm_loadu_pd (npb+k)));
                    k += 2;
                  }
                  if (k < grad.total_params)
                  { grad.param_block[k] += npb[k];
                  }
                }
#               elif FP64 && USE_SIMD_INTRINSICS && __SSE2__
                { unsigned e = grad.total_params & ~(unsigned)0x3;
                  for (k = 0; k < e; k += 4)
                  { _mm_storeu_pd (grad.param_block+k, 
                                   _mm_add_pd (
                                     _mm_loadu_pd (grad.param_block+k),
                                     _mm_loadu_pd (npb+k)));
                    _mm_storeu_pd (grad.param_block+k+2, 
                                   _mm_add_pd (
                                     _mm_loadu_pd (grad.param_block+k+2),
                                     _mm_loadu_pd (npb+k+2)));
                  }
                  if (k < (grad.total_params & ~(unsigned)0x1))
                  { _mm_storeu_pd (grad.param_block+k, 
                                   _mm_add_pd (
                                     _mm_loadu_pd (grad.param_block+k),
                                     _mm_loadu_pd (npb+k)));
                    k += 2;
                  }
                  if (k < grad.total_params)
                  { grad.param_block[k] += npb[k];
                  }
                }
#               else
                { for (k = 0; k < grad.total_params; k++)
                  { grad.param_block[k] += npb[k];
                  }
                }
#               endif
                np += 1;
              }
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

  net_model_max_second (seconds.o, arch, model, surv, sigmas.noise);

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
