/* GP-EVAL.C - Program to evaluate functions drawn from Gaussian processes. */

/* Copyright (c) 1996 by Radford M. Neal 
 *
 * Permission is granted for anyone to copy, use, or modify this program 
 * for purposes of research or education, provided this copyright notice 
 * is retained, and note is made of any changes that have been made. 
 *
 * This program is distributed without any warranty, express or implied.
 * As this program was written for research purposes only, it has not been
 * tested to the degree that would be advisable in any important application.
 * All use of this program is entirely at the user's own risk.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "misc.h"
#include "log.h"
#include "data.h"
#include "prior.h"
#include "model.h"
#include "matrix.h"
#include "rand.h"
#include "gp.h"
#include "gp-data.h"


/* VARIOUS CONSTANTS. */

#define Max_dim 2		/* Maximum number of inputs allowed */

#define Regularization 1e-6	/* Regularization added to diagonal of 
				   covariance if no jitter or noise */

static void usage(void);


/* MAIN PROGRAM. */

main
( int argc,
  char **argv
)
{
  gp_spec *gp;
  model_specification *m;

  gp_hypers hypers, *h = &hypers;

  double grid_low[Max_dim];
  double grid_high[Max_dim];

  int grid_size[Max_dim];
  int grid_point[Max_dim];
  int grid_total;

  double *grid_inputs;

  double *latent_values;
  double *noise_variances;

  log_file logf;
  log_gobbled logg;

  int lindex, hindex, index_mod;
  int ng;

  int N, I;

  int gen_targets;

  double *train_cov, *grid_cov, *grid_cov2, *tr_gr_cov, *gr_tr_cov;
  double *prd;

  double *scr1, *scr2;
  double *output, *mean, *rnd;

  double *g;
  char **ap;
  int first;
  int i, j;

  /* Look at arguments. */

  gen_targets = 0;

  if (argc>1 && strcmp(argv[argc-1],"targets")==0)
  { gen_targets = 1;
    argc -= 1;
    argv[argc] = 0;
  }
  
  if (argc<3) usage();

  logf.file_name = argv[1];
  
  parse_range (argv[2], &lindex, &hindex, &index_mod);

  if (hindex==-2) 
  { hindex = lindex;
  }
  else if (hindex==-1)
  { hindex = 1000000000;
  }

  ap = argv+3;

  N = 1;

  if (*ap!=0 && strcmp(*ap,"/")!=0)
  { if ((N = atoi(*ap))==0) usage();
    ap += 1;
  }

  if (*ap==0 || (argv+argc-ap)%4!=0) usage();

  ng = 0;

  for ( ; *ap!=0; ap += 4)
  { if (strcmp(ap[0],"/")!=0 
     || (grid_size[ng] = atoi(ap[3]))<=0 && strcmp(ap[3],"0")!=0) usage();
    grid_low[ng] = atof(ap[1]);
    grid_high[ng] = atof(ap[2]);
    ng += 1;
  }

  if (ng>Max_dim)
  { fprintf(stderr,"Too many input dimensions (max %d)\n",Max_dim);
    exit(1);
  }

  /* Open log file and read Gaussian process and model specifications. */

  log_file_open (&logf, 0);

  log_gobble_init(&logg,0);
  gp_record_sizes(&logg);

  if (!logf.at_end && logf.header.index==-1)
  { log_gobble(&logf,&logg);
  }

  gp = logg.data['P'];
  m = logg.data['M'];

  if (gp==0)
  { fprintf(stderr,"No specification for Gaussian process in log file\n");
    exit(1);
  }

  if (gp->N_inputs!=ng)
  { fprintf(stderr,
      "Number of grid ranges doesn't match number of input dimensions\n");
    exit(1);
  }

  if (gp->N_outputs!=1)
  { fprintf(stderr,"Number of outputs must be one for gp-eval\n");
    exit(1);
  }

  if (gen_targets) 
  { 
    if (m==0)
    { fprintf(stderr,"No model specification in log file\n");
      exit(1);
    }

    if (m->type!='R')
    { fprintf(stderr,
 "Currently, targets can be generated randomly only for the 'real' data model\n"
      );
      exit(1);
    }
  }

  /* Do some initialization. */

  h->total_hypers = gp_hyper_count(gp,m);

  logg.req_size['S'] = h->total_hypers * sizeof(double);

  data_spec = logg.data['D'];

  if (data_spec!=0) 
  { gp_data_read (1, 0, gp, m, 0);
  }
  else
  { N_train = 0;
  }

  /* Compute the input points over the grid. */

  grid_total = 1;
  for (i = 0; i<gp->N_inputs; i++) grid_total *= grid_size[i]+1;

  grid_inputs = chk_alloc ((grid_total+1)*gp->N_inputs, sizeof(double));

  g = grid_inputs;

  for (i = 0; i<gp->N_inputs; i++) 
  { grid_point[i] = 0;
    g[i] = grid_low[i];
  }
    
  for (;;)
  {
    g += gp->N_inputs;

    for (i = gp->N_inputs-1; i>=0 && grid_point[i]==grid_size[i]; i--) 
    { grid_point[i] = 0;
      g[i] = grid_low[i];
    }

    if (i<0) break;

    grid_point[i] += 1;
    g[i] = grid_low[i] 
             + grid_point[i] * (grid_high[i]-grid_low[i]) / grid_size[i];

    for (i = i-1 ; i>=0; i--)
    { g[i] = g[i-gp->N_inputs];
    }
  }

  /* Allocate some space. */

  rnd    = chk_alloc (grid_total, sizeof(double));
  mean   = chk_alloc (grid_total, sizeof(double));
  output = chk_alloc (grid_total, sizeof(double));

  train_cov = chk_alloc (N_train*N_train, sizeof (double));

  scr1 = chk_alloc (N_train, sizeof(double));
  scr2 = chk_alloc (N_train, sizeof(double));

  grid_cov = chk_alloc (grid_total*grid_total, sizeof (double));
  if (N_train>0) grid_cov2 = chk_alloc (grid_total*grid_total, sizeof (double));

  tr_gr_cov = chk_alloc (N_train*grid_total, sizeof (double));
  gr_tr_cov = chk_alloc (grid_total*N_train, sizeof (double));
  prd       = chk_alloc (grid_total*N_train, sizeof (double));

  /* Draw function values for the specified iterations. */

  first = 1;

  for (;;)
  {
    /* Skip to next desired index, or to end of range. */

    while (!logf.at_end && logf.header.index<=hindex
     && (logf.header.index<lindex 
          || (logf.header.index-lindex)%index_mod!=0))
    { log_file_forward(&logf);
    }
  
    if (logf.at_end || logf.header.index>hindex)
    { break;
    }
  
    /* Gobble up records for this index. */

    log_gobble(&logf,&logg);
       
    /* Read the desired network from the log file. */
  
    if (logg.data['S']==0 || logg.index['S']!=logg.last_index)
    { fprintf(stderr, "Record missing at index %d\n",logg.last_index);
      exit(1);
    }
  
    h->hyper_block = logg.data['S'];
    gp_hyper_pointers (h, gp, m);

    latent_values = logg.data['F']!=0 && logg.index['F']==logg.last_index
                   ? (double*) logg.data['F'] : 0;

    noise_variances = logg.data['N']!=0 && logg.index['N']==logg.last_index
                       ? (double*) logg.data['N'] : 0;

    if (latent_values==0 && N_train>0
     && m!=0 && m->type!='R')
    { fprintf(stderr,"Latent values needed, but not stored in log file\n");
      exit(1);
    }

    if (noise_variances==0 && N_train>0 
     && m!=0 && m->type=='R' && m->noise.alpha[2]!=0)
    { fprintf(stderr,"Noise variances needed, but not stored in log file\n");
      exit(1);
    }

    if (latent_values!=0 && logg.actual_size['F']!=N_train*sizeof(double))
    { fprintf(stderr,"Record with latent values is wrong length\n");
      exit(1);
    }

    if (noise_variances!=0 && logg.actual_size['N']!=N_train*sizeof(double))
    { fprintf(stderr,"Record with noise variances is wrong length\n");
      exit(1);
    }

    /* Compute the covariance matrix for training cases or function values, 
       then invert it. */

    if (N_train>0)
    { 
      /* Find covariance for latent or target values at training points,
         storing in train_cov. */

      gp_train_cov (gp, m, h, 0, noise_variances, 
                    latent_values ? train_cov : 0,
                    latent_values ? 0 : train_cov,
                    0);

      /* Invert covariance matrix computed above. */

      if (!cholesky (train_cov, N_train, 0)
       || !inverse_from_cholesky (train_cov, scr1, scr2, N_train))
      { fprintf(stderr,"Couldn't invert covariance matrix of training cases\n");
        exit(1);
      }
    }

    /* Compute covariances between grid and training points, then
       multiply by inverse covariance matrix of training points found 
       above.  This gives the vector of regression coefficients for
       finding the conditional means at the training points, which is
       also used in finding the covariance. */

    if (N_train>0)
    { 
      gp_cov (gp, h, grid_inputs, grid_total, train_inputs, N_train, 
              gr_tr_cov, 0, 0);

      matrix_product (gr_tr_cov, train_cov, prd, grid_total, N_train, N_train);
    }

    /* Compute the mean of the values at the grid points. */

    if (N_train==0)
    { 
      for (i = 0; i<grid_total; i++) mean[i] = 0;
    }
    else
    { 
      matrix_product (prd, latent_values ? latent_values : train_targets,
                      mean, grid_total, 1, N_train);
    }

    /* Compute the prior covariance matrix for function values at the 
       grid points. */

    gp_cov (gp, h, grid_inputs, grid_total, grid_inputs, grid_total, 
            grid_cov, 0, 0);

    if (gp->has_jitter)
    { for (i = 0; i<grid_total; i++)
      { grid_cov[i*grid_total+i] += exp(2 * *h->jitter);
      }
    }

    /* Update this to the posterior covariance matrix for values at grid 
       points, if we have any training data. */

    if (N_train>0)
    { 
      gp_cov (gp, h, train_inputs, N_train, grid_inputs, grid_total,
              tr_gr_cov, 0, 0);
 
      matrix_product (prd, tr_gr_cov, grid_cov2, grid_total, grid_total, 
                      N_train);

      for (j = 0; j<grid_total*grid_total; j++) grid_cov[j] -= grid_cov2[j];
    }

    /* Add the noise variance if we're generating targets. */

    if (gen_targets)
    { if (m->noise.alpha[2]!=0)
      { for (i = 0; i<grid_total; i++) 
        { double n;
          n = prior_pick_sigma(exp(*h->noise[0]),m->noise.alpha[2]);
          grid_cov[i*grid_total+i] += n*n;
        }
      }
      else
      { for (i = 0; i<grid_total; i++) 
        { grid_cov[i*grid_total+i] += exp(2 * *h->noise[0]);
        }
      }
    }

    /* Add the regularization if we haven't done anything else that might
       be adequate to regularize the computation. */

    if (!gp->has_jitter && !gen_targets)
    { for (i = 0; i<grid_total; i++)
      { grid_cov[i*grid_total+i] += Regularization;
      }   
    }

    /* Find the Cholesky decomposition of the posterior covariance for the
       grid points, which is what is needed for generating them at random. */

    if (!cholesky (grid_cov, grid_total, 0))
    { fprintf(stderr,"Couldn't find Cholesky decomposition for grid points\n");
      exit(1);
    }

    /* Generate function values for each random number seed. */

    for (I = 0; I < (N>0 ? N : -N); I++)
    {
      rand_seed (N>0 ? logg.last_index*100+I : I);

      /* Generate function at random using posterior mean and Cholesky 
         decomposition of posterior covariance. */

      for (i = 0; i<grid_total; i++)
      { rnd[i] = rand_gaussian();
      }

      for (i = 0; i<grid_total; i++)
      { output[i] = 
            mean[i] + inner_product (rnd, 1, grid_cov+i*grid_total, 1, i+1);
      }

      /* Print the value of the function, or targets generated from it, 
         for the grid of points. */
  
      if (first)
      { first = 0;
      }
      else
      { printf("\n");
      }

      for (j = 0; j<grid_total; j++)
      {
        if (j>0 && gp->N_inputs>0
          && grid_inputs[gp->N_inputs*(j+1)-1]<grid_inputs[gp->N_inputs*j-1])
        { printf("\n");
        }

        for (i = 0; i<gp->N_inputs; i++)  
        { printf (" %8.5f", grid_inputs[gp->N_inputs*j+i]);
        }

        printf (" %.7e\n", output[j]);
      }
    
    }
  }
  
  exit(0);
}


/* DISPLAY USAGE MESSAGE AND EXIT. */

static void usage(void)
{
  fprintf (stderr, 
"Usage: gp-eval log-file range [ [-]N ] { / low high grid-size } [ \"targets\" ]\n"
);
  exit(1);
}
