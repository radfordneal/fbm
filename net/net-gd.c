/* NET-GD.C - Program to train a network by gradient descent in the error. */

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
 *
 * The features allowing the number of iterations to be specified in terms of
 * cpu time are adapted from modifications done by Carl Edward Rasmussen, 1995.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "misc.h"
#include "rand.h"
#include "log.h"
#include "mc.h"
#include "data.h"
#include "prior.h"
#include "model.h"
#include "net.h"
#include "net-data.h"


/* CLOCKS_PER_SEC should be defined by <time.h>, but it seems that it
   isn't always.  1000000 seems to be the best guess for Unix systems. */

#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC 1000000	/* Best guess */
#endif


/* NETWORK VARIABLES. */

static net_arch *arch;		/* Network architecture */
static net_flags *flgs;		/* Network flags, null if none */
static model_specification *model; /* Data model */
static net_priors *priors;	/* Network priors */
static model_survival *surv;	/* Hazard type for survival model */

static net_sigmas sigmas;	/* Hyperparameters for network */
static net_params params;	/* Pointers to parameters */

static net_values *deriv;	/* Derivatives for training cases */
static net_params grad;		/* Pointers to gradient for network parameters*/
static net_params gradp;	/* Gradient of the log prior */
static net_params ograd;	/* Total gradient from last pass through data */
static net_params tgrad;	/* Gradient currently being accumulated */


/* PARAMETER GROUPS. */

#define Max_groups 100		/* Maximum number of weight groups */
#define Max_subgroups 1000	/* Maximum number of subgroups in a group */

static int n_groups;		/* Number of groups */

static int start_group[Max_groups+1];  /* Index of first weight in group */
static double stepsize[Max_groups];    /* Stepsizes for each group */
static int subdivide[Max_groups];      /* Should group be subdivided? */
static int configured[Max_groups];     /* Is group configured? */

static double submag[Max_subgroups];   /* Magnitudes of gradients in subgroups*/

/* PROCEDURES. */

static void usage(void);


/* MAIN PROGRAM. */

int main
( int argc,
  char **argv
)
{
  int index, max, modulus;
  int timelimit, diff;

  static mc_iter it0, *it; /* Static so fields get initialized to zero */

  enum { Batch, Online } method;

  log_file logf;
  log_gobbled logg;

  char **ap;

  net_value *value_block;
  int value_count;

  unsigned old_clock; /* Theoretically, these should be of type clock_t, but  */
  unsigned new_clock; /* that type is inexplicably declared signed on most    */
                      /* systems, cutting the already-too-small range in half */
  int ns, np;
  int i, j, g, o, n, s, c;

  /* Look at initial program arguments. */

  timelimit = 0;
  modulus = 1;

  if (argc<3) usage();

  logf.file_name = argv[1];

  ap = argv+2;

  if (**ap == '@') timelimit = 1; 

  if ((max = atoi(**ap=='@' ? *ap+1 : *ap)) <= 0) usage();
  ap += 1;

  if (*ap!=0 && strcmp(*ap,"/")!=0 && ((modulus = atoi(*ap++))<=0)) usage();

  if (*ap==0 || strcmp(*ap++,"/")!=0) usage();

  /* Look at stepsize and method arguments. */

  method = Online;
  ns = 0;

  while (*ap!=0)
  {
    if (strcmp(*ap,"batch")==0)
    { method = Batch;
      ap += 1;
      break;
    }
    else if (strcmp(*ap,"online")==0)
    { method = Online;
      ap += 1;
      break;
    }

    if (ns>=Max_groups) 
    { fprintf(stderr,"Too many stepsizes specified\n");
      exit(1);
    }

    if ((stepsize[ns] = atof(*ap++))<=0) usage();

    ns += 1;
  }

  if (ns==0) usage();

  diff = 0;

  while (*ap!=0)
  { if ((g = atoi(*ap++))<=0 || g>Max_groups) usage();
    subdivide[g-1] = 1;
    diff = 1;
  }

  if (*ap!=0) usage();

  /* Open log file and read all records. */

  log_file_open (&logf, 1);

  log_gobble_init(&logg,0);

  logg.req_size['r'] = sizeof (rand_state);
  logg.req_size['o'] = sizeof (mc_ops);
  logg.req_size['t'] = sizeof (mc_traj);
  logg.req_size['b'] = sizeof (mc_temp_state);

  while (!logf.at_end)
  { log_gobble(&logf,&logg);
  }

  index = log_gobble_last(&logf,&logg);

  if (!timelimit && index>max)
  { fprintf(stderr,"Iterations up to %d already exist in log file\n",max);
    exit(1);
  }

  /* Used saved pseudo-random number state. */

  if (logg.data['r']!=0) 
  { rand_use_state(logg.data['r']);
  }

  /* Check that required network specification records are present. */
  
  arch   = logg.data['A'];
  flgs   = logg.data['F'];
  model  = logg.data['M'];
  priors = logg.data['P'];
  surv   = logg.data['V'];

  net_check_specs_present(arch,priors,1,model,surv);

  if (model && model->type=='V')
  { fprintf(stderr,"Can't handle survival models\n");
    exit(1);
  }
  
  /* Locate existing network, if one exists, or set one up randomly. */
  
  sigmas.total_sigmas = net_setup_sigma_count(arch,flgs,model);
  params.total_params = net_setup_param_count(arch,flgs);

  sigmas.sigma_block = logg.data['S'];
  params.param_block = logg.data['W'];

  np = params.total_params;
  
  if (sigmas.sigma_block!=0 || params.param_block!=0)
  {
    if (sigmas.sigma_block==0 || logg.index['S']!=logg.last_index
     || params.param_block==0 || logg.index['W']!=logg.last_index)
    { fprintf(stderr,
        "Network stored in log file is apparently incomplete\n");
      exit(1);
    }
  
    if (logg.actual_size['S'] != sigmas.total_sigmas*sizeof(net_sigma)
     || logg.actual_size['W'] != params.total_params*sizeof(net_param))
    { fprintf(stderr,"Bad size for network record\n");
      exit(1);
    }
  
    net_setup_sigma_pointers (&sigmas, arch, flgs, model);
    net_setup_param_pointers (&params, arch, flgs);
  }
  else
  {
    sigmas.sigma_block = chk_alloc (sigmas.total_sigmas, sizeof (net_sigma));
    params.param_block = chk_alloc (params.total_params, sizeof (net_param));
  
    net_setup_sigma_pointers (&sigmas, arch, flgs, model);
    net_setup_param_pointers (&params, arch, flgs);
   
    net_prior_generate (&params, &sigmas, arch, flgs, model, priors, 1, 0, 0);

    for (j = 0; j<np; j++) 
    { params.param_block[j] += 0.01 - 0.02*rand_uniopen();
    }
  }

  /* Read training data, and "test" data if needed for cross-validation,
     and allocate space for derivatives. */
  
  data_spec = logg.data['D'];

  if (data_spec!=0)
  { 
    net_data_read (1, 0, arch, model, surv);
    
    deriv = chk_alloc (1, sizeof *deriv);
    
    value_count = net_setup_value_count(arch);
    value_block = chk_alloc (value_count, sizeof *value_block);
    
    net_setup_value_pointers (deriv, value_block, arch);
  }

  if (N_train==0)
  { fprintf (stderr,"Can't do training when there's no data\n");
    exit(1);
  }

  /* See how many groups there are, and how divided into subgroups. */

  for (n_groups = 0; 
       net_setup_param_group (arch, flgs, n_groups+1, &o, &n, &s, &c); 
       n_groups++)
  { 
    if (n_groups>=Max_groups) 
    { fprintf(stderr,"Too many weight groups in this network\n");
      exit(1);
    }

    start_group[n_groups] = o;

    if (subdivide[n_groups]) 
    { if (c)
      { fprintf(stderr,
"Differential stepsizes with subgroups not allowed when group is configured\n");
        exit(1);
      }
      if (s>Max_subgroups)
      { fprintf(stderr,"Too many subgroups in a group\n");
        exit(1);
      }
      subdivide[n_groups] = s;
    }

    if (n_groups>0 && start_group[n_groups]<start_group[n_groups-1]) abort();
  }

  start_group[n_groups] = np;

  /* Set stepsizes for each group. */

  for (g = 0; g<ns; g++)
  { stepsize[g] /= N_train+1;
  }

  if (ns==1)
  { for (g = 1; g<n_groups; g++)
    { stepsize[g] = stepsize[0];
    }
  }
  else if (ns!=n_groups)
  { fprintf(stderr,
"Number of stepsizes given (%d) does not match number of parameter groups (%d)\n",
            ns, n_groups);
    exit(1);
  }

  /* Initialize for performing iterations. */
  
  grad.total_params = params.total_params;
  grad.param_block  = chk_alloc (params.total_params, sizeof (net_param));
  net_setup_param_pointers (&grad, arch, flgs);

  gradp.total_params = params.total_params;
  gradp.param_block  = chk_alloc (params.total_params, sizeof (net_param));
  net_setup_param_pointers (&gradp, arch, flgs);

  tgrad.total_params = params.total_params;
  tgrad.param_block  = chk_alloc (params.total_params, sizeof (net_param));
  net_setup_param_pointers (&tgrad, arch, flgs);

  ograd.total_params = params.total_params;
  ograd.param_block  = chk_alloc (params.total_params, sizeof (net_param));
  net_setup_param_pointers (&ograd, arch, flgs);

  it = logg.data['i']==0 ? &it0 : (mc_iter *) logg.data['i'];

  old_clock = clock();

  if (diff && method==Online)
  { 
    net_prior_prob (&params, &sigmas, 0, &ograd, arch, flgs, priors, 2);

    for (i = 0; i<N_train; i++)
    { 
      net_func (&train_values[i], 0, arch, flgs, &params);

      net_model_check (model);    
      net_model_prob (&train_values[i], 
                      train_targets + data_spec->N_targets*i,
                      0, deriv, arch, model, surv, &sigmas, 2);
  
      net_back (&train_values[i], deriv, arch->has_ti ? -1 : 0,
                arch, flgs, &params);

      net_grad (&ograd, &params, &train_values[i], deriv, arch, flgs);
    }
  }

  /* Perform gradient descent iterations. */

  for (; (!timelimit && index<=max) || (timelimit && it->time<60000*max); 
         index++)
  {
    /* Change parameters by the selected gradient descent method. */

    switch (method)
    {
      case Batch:
      { 
        net_prior_prob (&params, &sigmas, 0, &grad, arch, flgs, priors, 2);

        for (i = 0; i<N_train; i++)
        { 
          net_func (&train_values[i], 0, arch, flgs, &params);
  
          net_model_check (model);    
          net_model_prob (&train_values[i], 
                          train_targets + data_spec->N_targets*i,
                          0, deriv, arch, model, surv, &sigmas, 2);
  
          net_back (&train_values[i], deriv, arch->has_ti ? -1 : 0,
                    arch, flgs, &params);

          net_grad (&grad, &params, &train_values[i], deriv, arch, flgs);
        }

        for (g = 0; g<n_groups; g++)
        { 
          if (subdivide[g])
          { 
            int size, k, n, d;
            double max;

            d = subdivide[g];
            size = (start_group[g+1] - start_group[g]) / d;

            j = start_group[g];
            for (k = 0; k<d; k++) 
            { submag[k] = 0;
              for (n = 0; n<size; n++)
              { submag[k] += grad.param_block[j] * grad.param_block[j];
                j += 1;
              }
              submag[k] *= submag[k];
              if (k==0 || submag[k]>max) 
              { max = submag[k];
              }
            }
            if (max<=0) max = 1e-10;
  
            j = start_group[g];
            for (k = 0; k<d; k++)
            { double s;
              s = stepsize[g] * (submag[k]/max);
              for (n = 0; n<size; n++)
              { params.param_block[j] -= s * grad.param_block[j];
                j += 1;
              }
            }
          }

          else
          { double s;
            s = stepsize[g];
            for (j = start_group[g]; j<start_group[g+1]; j++)
            { params.param_block[j] -= s * grad.param_block[j];
            } 
          }
        }

        break;

      }

      case Online: 
      {
        if (diff)
        { for (j = 0; j<np; j++) tgrad.param_block[j] = 0;
        }

        net_prior_prob (&params, &sigmas, 0, &gradp, arch, flgs, priors, 2);

        for (j = 0; j<np; j++) gradp.param_block[j] /= N_train;

        for (i = 0; i<N_train; i++)
        { 
          for (j = 0; j<np; j++) grad.param_block[j] = gradp.param_block[j];

          net_func (&train_values[i], 0, arch, flgs, &params);
    
          net_model_check (model);    
          net_model_prob (&train_values[i], 
                         train_targets + data_spec->N_targets*i,
                         0, deriv, arch, model, surv, &sigmas, 2);
  
          net_back (&train_values[i], deriv, arch->has_ti ? -1 : 0,
                    arch, flgs, &params);

          net_grad (&grad, &params, &train_values[i], deriv, arch, flgs);

          if (diff)
          { for (j = 0; j<np; j++) tgrad.param_block[j] += grad.param_block[j];
          }

          for (g = 0; g<n_groups; g++)
          { 
            if (subdivide[g])
            { 
              int size, k, n, d;
              double max;

              d = subdivide[g];
              size = (start_group[g+1] - start_group[g]) / d;

              j = start_group[g];
              for (k = 0; k<d; k++) 
              { submag[k] = 0;
                for (n = 0; n<size; n++)
                { submag[k] += ograd.param_block[j] * ograd.param_block[j];
                  j += 1;
                }
                submag[k] *= submag[k]; 
                if (k==0 || submag[k]>max) 
                { max = submag[k];
                }
              }
              if (max<=0) max = 1e-10;

              j = start_group[g];
              for (k = 0; k<d; k++)
              { double s;
                s = stepsize[g] * (submag[k]/max);
                for (n = 0; n<size; n++)
                { params.param_block[j] -= s * grad.param_block[j];
                  j += 1;
                }
              }
            }

            else
            { double s;
              s = stepsize[g];
              for (j = start_group[g]; j<start_group[g+1]; j++)
              { params.param_block[j] -= s * grad.param_block[j];
              } 
            }
          }
        }

        if (diff)
        { for (j = 0; j<np; j++) ograd.param_block[j] = tgrad.param_block[j];
        }

        break;
      }
    }

    /* Update timing information. */

    new_clock = clock(); 
    it->time += (int) (0.5 + 
             (1000.0 * (unsigned) (new_clock - old_clock)) / CLOCKS_PER_SEC);
    old_clock = new_clock;

    /* Write to log file, if this is an iteration we should save. */

    if (index%modulus==0)
    { 
      logf.header.type = 'S';
      logf.header.index = index;
      logf.header.size = sigmas.total_sigmas * sizeof (net_sigma);
      log_file_append (&logf, sigmas.sigma_block);

      logf.header.type = 'W';
      logf.header.index = index;
      logf.header.size = params.total_params * sizeof (net_param);
      log_file_append (&logf, params.param_block);

      logf.header.type = 'i';
      logf.header.index = index;
      logf.header.size = sizeof *it;
      log_file_append (&logf, it);

      logf.header.type = 'r';
      logf.header.index = index;
      logf.header.size = sizeof (rand_state);
      log_file_append (&logf, rand_get_state());
    }
  }

  log_file_close(&logf);

  exit(0);
}


/* DISPLAY USAGE MESSAGE AND EXIT. */

static void usage(void)
{
  fprintf (stderr, 
    "Usage: net-gd log-file [\"@\"]iteration [ save-mod ] / stepsize { stepsize } \n");  
  fprintf (stderr, 
    "              [ method [ group { group } ] ]\n");
  exit(1);
}
