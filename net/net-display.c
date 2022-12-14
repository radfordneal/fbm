/* NET-DISPLAY.C - Program to print network parameters and other such info. */

/* Copyright (c) 1995-2022 by Radford M. Neal 
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

static void usage(void)
{ fprintf (stderr,
   "Usage: net-display [ -p | -h | -P | -H ] [ -lN ] [ -sN ] [ group ]\n");
  fprintf (stderr,
   "                   log-file [ index ]\n");
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
  model_specification *m;

  net_params  params, *w = &params;
  net_sigmas  sigmas, *s = &sigmas;

  log_file logf;
  log_gobbled logg;

  int sigmas_only;
  int params_only;
  int dump_params;
  int dump_sigmas;
  int index, group;
  int per_line, per_section;

  char junk;
  int g;

  /* Look at arguments. */

  sigmas_only = 0;
  params_only = 0;
  dump_params = 0;
  dump_sigmas = 0;
  group = 0;  /* all */
  per_line = 10;
  per_section= -1;  /* no sections */

  for (;;)
  { if (argc>1 && (strcmp(argv[1],"-p")==0 || strcmp(argv[1],"-w")==0))
    { params_only = 1;
      argv += 1;
      argc -= 1;
    }
    else if (argc>1 && (strcmp(argv[1],"-h")==0 || strcmp(argv[1],"-s")==0))
    { sigmas_only = 1;
      argv += 1;
      argc -= 1;
    }
    else if (argc>1 && strcmp(argv[1],"-P")==0)
    { dump_params = 1;
      argv += 1;
      argc -= 1;
    }
    else if (argc>1 && strcmp(argv[1],"-H")==0)
    { dump_sigmas = 1;
      argv += 1;
      argc -= 1;
    }
    else if (argc>1 && argv[1][0]=='-' && argv[1][1]=='l')
    { if (sscanf(argv[1]+2,"%d%c",&per_line,&junk)!=1) usage();
      argv += 1;
      argc -= 1;
    }
    else if (argc>1 && argv[1][0]=='-' && argv[1][1]=='s')
    { if (sscanf(argv[1]+2,"%d%c",&per_section,&junk)!=1) usage();
      argv += 1;
      argc -= 1;
    }
    else
    { break;
    }
  }

  if (argc>1 && sscanf(argv[1],"%d%c",&g,&junk) == 1 && g>0)
  { group = g;
    argv += 1;
    argc -= 1;
  }

  index = -1;

  if (argc!=2 && argc!=3 
   || argc>2 && (index = atoi(argv[2]))<=0 && strcmp(argv[2],"0")!=0) 
  { usage();
  }

  logf.file_name = argv[1];

  /* Open log file and read network architecture. */

  log_file_open (&logf, 0);

  log_gobble_init(&logg,0);
  net_record_sizes(&logg);

  while (!logf.at_end && logf.header.index<0)
  { log_gobble(&logf,&logg);
  }

  a = logg.data['A'];
  m = logg.data['M'];
  flgs = logg.data['F'];
  
  if (a==0)
  { fprintf(stderr,"No architecture specification in log file\n");
    exit(1);
  }

  s->total_sigmas = net_setup_sigma_count(a,flgs,m);
  w->total_params = net_setup_param_count(a,flgs,0);

  logg.req_size['S'] = s->total_sigmas * sizeof(net_sigma);
  logg.req_size['W'] = w->total_params * sizeof(net_param);

  /* Read the desired network from the log file. */

  if (index<0)
  { 
    log_gobble_last(&logf,&logg);

    if (logg.last_index<0)
    { fprintf(stderr,"No network in log file\n");
      exit(1);
    }

    index = logg.last_index;
  }
  else
  {
    while (!logf.at_end && logf.header.index!=index)
    { log_file_forward(&logf);
    }

    if (logf.at_end)
    { fprintf(stderr,"No network with that index is in the log file\n");
      exit(1);
    }

    log_gobble(&logf,&logg);
  }

  if (logg.index['W']!=index)
  { fprintf(stderr,"No weights stored for the network with that index\n");
    exit(1);
  }

  if (logg.index['S']!=index)
  { fprintf(stderr,"No sigmas stored for the network with that index\n");
    exit(1);
  }

  s->sigma_block = logg.data['S'];
  w->param_block = logg.data['W'];

  net_setup_sigma_pointers (s, a, flgs, m);
  net_setup_param_pointers (w, a, flgs);

  /* Do only a raw dump, if asked to. */

  if (dump_params)
  { int i;
    if (group)
    { int offset, number, source, configured;
      if (net_setup_param_group (a, flgs, group, 
                                 &offset, &number, &source, &configured))
      { for (i = offset; i<offset+number; i++)
        { printf("%.16g\n",w->param_block[i]);
        }
      }
    }
    else
    { for (i = 0; i<w->total_params; i++)
      { printf("%.16g\n",w->param_block[i]);
      }
    }
    exit(0);
  }

  if (dump_sigmas)
  { int i;
    if (group)
    { int offset, number, adj;
      if (net_setup_hyper_group (a, flgs, group, 
                                 &offset, &number, &adj))
      { for (i = offset; i<offset+number; i++)
        { printf("%.16g\n",s->sigma_block[i]);
        }
      }
      else if (m && m->type=='R') /* check if it's the group for noise sigmas */
      { offset = 0; number = 0;
        if (group==1 || net_setup_hyper_group (a, flgs, group-1, 
                                               &offset, &number, &adj))
        { for (i = offset+number; i<s->total_sigmas; i++)
          { printf("%.16g\n",s->sigma_block[i]);
          }
        }
      }
    }
    else
    { for (i = 0; i<s->total_sigmas; i++)
      { printf("%.16g\n",s->sigma_block[i]);
      }
    }
    exit(0);
  }

  /* Print values of the parameters and/or hyperparameters. */

  printf("\nNetwork in file \"%s\" with index %d", logf.file_name, index);
  if (sigmas_only) printf(", sigmas only");
  if (params_only) printf(", parameters only");
  if (group) printf(", group %d only",group);
  printf("\n");

  if (params_only)
  { net_print_params(w,0,a,flgs,m,group,per_line,per_section);
  }
  else if (sigmas_only)
  { net_print_sigmas(s,a,flgs,m,group);
  }
  else
  { net_print_params(w,s,a,flgs,m,group,per_line,per_section);
  }

  printf("\n");

  exit(0);
}
