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

#define EXTERN extern
#include "net-mc.h"


/* CLOCKS_PER_SEC should be defined by <time.h>, but it seems that it
   isn't always.  1000000 seems to be the best guess for Unix systems. */

#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC 1000000	/* Best guess */
#endif


/* NETWORK VARIABLES.  Some also in net-mc.h. */

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

  unsigned old_clock; /* Theoretically, these should be of type clock_t, but  */
  unsigned new_clock; /* that type is inexplicably declared signed on most    */
                      /* systems, cutting the already-too-small range in half */

  int i, j, g, o, n, s, c, ns;

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

  while (!logf.at_end)
  { log_gobble(&logf,&logg);
  }

  index = log_gobble_last(&logf,&logg);

  if (!timelimit && index>max)
  { fprintf(stderr,"Iterations up to %d already exist in log file\n",max);
    exit(1);
  }

  /* Initialize using records found. */

  mc_dynamic_state ds;
  mc_app_initialize(&logg,&ds);

  int np = params.total_params;

  if (logg.data['r']!=0) 
  { rand_use_state ((rand_state *) logg.data['r']);
  }

  if (model && model->type=='V')
  { fprintf(stderr,"Can't handle survival models\n");
    exit(1);
  }
  
  if (logg.data['W'] == 0) /* no existing network, initialize randomly */
  { for (j = 0; j<np; j++) 
    { params.param_block[j] += 0.01 - 0.02*rand_uniopen();
    }
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
  grad.param_block  = (net_param *) 
                        chk_alloc (params.total_params, sizeof (net_param));
  net_setup_param_pointers (&grad, arch, flgs);

  gradp.total_params = params.total_params;
  gradp.param_block  = (net_param *)
                         chk_alloc (params.total_params, sizeof (net_param));
  net_setup_param_pointers (&gradp, arch, flgs);

  tgrad.total_params = params.total_params;
  tgrad.param_block  = (net_param *)
                         chk_alloc (params.total_params, sizeof (net_param));
  net_setup_param_pointers (&tgrad, arch, flgs);

  ograd.total_params = params.total_params;
  ograd.param_block  = (net_param *)
                         chk_alloc (params.total_params, sizeof (net_param));
  net_setup_param_pointers (&ograd, arch, flgs);

  it = logg.data['i']==0 ? &it0 : (mc_iter *) logg.data['i'];

  old_clock = clock();

  net_model_check (model);    

  if (diff && method==Online)
  { 
    net_prior_prob (&params, &sigmas, 0, &ograd, arch, flgs, priors, 2);
    net_training_cases (0, &ograd, 0, N_train, 1, 1);
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
        net_training_cases (0, &grad, 0, N_train, 1, 1);

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

          net_training_cases (0, &grad, i, 1, 1, 1);

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
