/* NET-GEN.C - Program to generate networks (eg, from prior distribution). */

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

#include "misc.h"
#include "log.h"
#include "prior.h"
#include "data.h"
#include "model.h"
#include "net.h"
#include "rand.h"

static void usage()
{ fprintf(stderr,
"Usage: net-gen log-file [ max-index ] [ \"fix\" [ value [ out-value ] ] [ \"-\" ] ]\n");
  exit(1);
}


/* MAIN PROGRAM. */

int main
( int argc,
  char **argv
)
{
  net_arch *a;
  net_flags *flgs;
  net_priors *p;

  model_specification *m;
  model_survival *v;

  net_sigmas  sigmas, *s = &sigmas;
  net_params  params, *w = &params;

  int max_index, index, fix, has_out_value, from_stdin;
  double value, out_value;

  log_file logf;
  log_gobbled logg;
 
  /* Look at program arguments. */

  max_index = 0;
  value = out_value = 0;
  from_stdin = 0;

  if (argc<2) 
  { usage();
  }

  logf.file_name = argv[1];

  if (argc>2 && strcmp(argv[2],"fix")!=0)
  { if ((max_index = atoi(argv[2]))<=0 && strcmp(argv[2],"0")!=0)
    { usage();
    }
    argv += 1;
    argc -= 1;
  }

  fix = argc>2 && strcmp(argv[2],"fix")==0;
  if (fix)
  { if (strcmp(argv[argc-1],"-")==0)
    { from_stdin = 1; 
      argc -= 1;
    }
    if (argc>3 && (value = out_value = atof(argv[3]))<=0
     || argc>4 && (out_value = atof(argv[4]))<=0
     || argc>5)
    { usage();
    }
  }

  /* Open log file and read network architecture and priors. */

  log_file_open (&logf, 1);

  log_gobble_init(&logg,0);
  net_record_sizes(&logg);
  logg.req_size['r'] = sizeof (rand_state);

  while (!logf.at_end && logf.header.index<0)
  { log_gobble(&logf,&logg);
  }

  a = logg.data['A'];
  m = logg.data['M'];
  p = logg.data['P'];
  v = logg.data['V'];

  flgs = logg.data['F'];

  net_check_specs_present(a,p,0,m,v);

  /* Allocate space for parameters and hyperparameters. */

  s->total_sigmas = net_setup_sigma_count(a,flgs,m);
  w->total_params = net_setup_param_count(a,flgs);

  s->sigma_block = chk_alloc (s->total_sigmas, sizeof (net_sigma));
  w->param_block = chk_alloc (w->total_params, sizeof (net_param));

  net_setup_sigma_pointers (s, a, flgs, m);
  net_setup_param_pointers (w, a, flgs);

  /* Read weights from standard input, if doing that. */

  net_param *wts;
  if (from_stdin)
  { double d;
    int i;
    wts = chk_alloc (w->total_params, sizeof (net_param));
    for (i = 0; i<w->total_params; i++)
    { if (scanf("%lf",&d) != 1) 
      { fprintf(stderr,"Error reading weights: %d of %d\n",i+1,w->total_params);
        exit(2);
      }
      wts[i] = d;
    }
    char junk = 0;
    if (scanf(" %c",&junk) != 0 && junk != 0) 
    { fprintf(stderr,"Junk present after weights: '%c'\n",junk);
      exit(2);
    }
  }

  /* Read last records in log file to see where to start, and to get random
     number state left after last network was generated. */

  index = log_gobble_last(&logf,&logg);

  if (logg.last_index<0) 
  { index = 0;
  }

  if (index>max_index)
  { fprintf(stderr,"Networks up to %d already exist in log file\n",max_index);
    exit(1);
  }

  if (logg.data['r']!=0) 
  { rand_use_state(logg.data['r']);
  }

  /* Generate new networks and write them to the log file. */

  for ( ; index<=max_index; index++)
  {
    net_prior_generate (w, s, a, flgs, m, p, fix, value, out_value);

    logf.header.type = 'S';
    logf.header.index = index;
    logf.header.size = s->total_sigmas * sizeof (net_sigma);
    log_file_append (&logf, s->sigma_block);

    logf.header.type = 'W';
    logf.header.index = index;
    logf.header.size = w->total_params * sizeof (net_param);
    log_file_append (&logf, from_stdin ? wts : w->param_block);

    if (!fix)
    { logf.header.type = 'r';
      logf.header.index = index;
      logf.header.size = sizeof (rand_state);
      log_file_append (&logf, rand_get_state());
    }
  }

  log_file_close(&logf);

  exit(0);
}
