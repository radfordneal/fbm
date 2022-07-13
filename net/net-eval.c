/* NET-EVAL.C - Program to evaluate the network function at a grid of points. */

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
#include "net-data.h"

static void usage(void);


/* MAIN PROGRAM. */

int main
( int argc,
  char **argv
)
{
  char *fname;

  net_arch *a;
  net_precomputed pre;
  net_flags *flgs;
  model_specification *m;
  model_survival *sv;

  net_sigmas sigmas, *s = &sigmas;
  net_params params, *w = &params;
  net_values values, *v = &values;

  net_value *value_block;
  int value_count;

  static net_value grid_low[Max_inputs];
  static net_value grid_high[Max_inputs];

  static int grid_size[Max_inputs];
  static int grid_point[Max_inputs];

  log_file logf;
  log_gobbled logg;

  int lindex, hindex, index_mod;
  int ng;

  net_value targets[Max_targets];
  int gen_targets;
  int inputs, hid, layer, outputs;
  int N_targets;

  char **ap;
  char junk;
  int first;
  int i, j;

  /* Look at arguments. */

  inputs = 1;  /* yes */
  hid = 0;     /* no */
  outputs = 1; /* yes */
  layer = -1;  /* all */
  gen_targets = 0;

  while (argc>1 && argv[1][0]=='-')
  { if (strcmp(argv[1],"-i")==0)
    { inputs = 0;  /* no */
    }
    else if (strcmp(argv[1],"-o")==0)
    { outputs = 0;  /* no */
    }
    else if (strcmp(argv[1],"-t")==0)
    { gen_targets = 1;
      outputs = 1;  /* yes (as targets) */
    }
    else if (strcmp(argv[1],"-h")==0)
    { hid = 1;     /* yes */
      layer = -1;  /* all */
    }
    else if (argv[1][1]=='h' && sscanf(argv[1]+2,"%d%c",&layer,&junk)==1)
    { if (layer<0) usage();
      hid = 1;  /* yes */
    }
    argc -= 1;
    argv += 1;
  }

  if (argc<4 || strcmp(argv[3],"/")!=0
             || argc!=5 && (argc-3)%4!=0 || (argc-3)/4>Max_inputs)
  { usage();
  }

  logf.file_name = argv[1];
  
  parse_range (argv[2], &lindex, &hindex, &index_mod);

  if (hindex==-2) 
  { hindex = lindex;
  }
  else if (hindex==-1)
  { hindex = 1000000000;
  }

  if (argc==5)
  { fname = argv[4];
  }
  else
  { 
    fname = NULL;
    ng = 0;

    for (ap = argv+3; *ap!=0; ap += 4)
    { if (strcmp(ap[0],"/")!=0 
       || ((grid_size[ng] = atoi(ap[3]))<=0 && strcmp(ap[3],"0")!=0)) usage();
      grid_low[ng] = atof(ap[1]);
      grid_high[ng] = atof(ap[2]);
      ng += 1;
    }
  }

  /* Open log file and read network architecture and data model. */

  log_file_open (&logf, 0);

  log_gobble_init(&logg,0);
  net_record_sizes(&logg);

  if (!logf.at_end && logf.header.index==-1)
  { log_gobble(&logf,&logg);
  }

  a = logg.data['A'];
  flgs = logg.data['F'];
  m = logg.data['M'];
  sv = logg.data['V'];

  if (a==0)
  { fprintf(stderr,"No architecture specification in log file\n");
    exit(1);
  }

  if (gen_targets) 
  { 
    if (m==0)
    { fprintf(stderr,"No model specification in log file\n");
      exit(1);
    }

    if (m->type=='V' && v==0)
    { fprintf(stderr,"No hazard specification for survival model\n");
      exit(1);
    }

    if (m->type=='V' && sv->hazard_type!='C')
    { fprintf(stderr,
 "Can't generate targets randomly for survival model with non-constant hazard\n"
      );
      exit(1);
    }

    N_targets = m->type=='C' ? 1 : a->N_outputs;
  }

  if (fname==NULL && a->N_inputs!=ng)
  { fprintf(stderr,
      "Number of grid ranges doesn't match number of input dimensions\n");
    exit(1);
  }

  if (hid && layer>=a->N_layers)
  { fprintf(stderr, "Hidden layer specified does not exist\n");
    exit(1);
  }

  s->total_sigmas = net_setup_sigma_count(a,flgs,m);
  w->total_params = net_setup_param_count(a,flgs,&pre);

  logg.req_size['S'] = s->total_sigmas * sizeof(net_sigma);
  logg.req_size['W'] = w->total_params * sizeof(net_param);

  /* Allocate space for values in network. */

  value_count = net_setup_value_count(a);
  value_block = chk_alloc (value_count, sizeof *value_block);

  net_setup_value_pointers (v, value_block, a, 0, 0);

  /* Evaluate function for the specified networks. */

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
  
    if (logg.data['W']==0 || logg.index['W']!=logg.last_index)
    { fprintf(stderr,
        "No weights stored for the network with index %d\n",logg.last_index);
      exit(1);
    }
  
    w->param_block = logg.data['W'];
    net_setup_param_pointers (w, a, flgs);
  
    if (gen_targets) 
    {
      if (logg.data['S']==0 || logg.index['S']!=logg.last_index)
      { fprintf(stderr,
          "No sigmas stored for network with index %d\n",logg.last_index);
        exit(1);
      }
  
      s->sigma_block = logg.data['S'];
      net_setup_sigma_pointers (s, a, flgs, m);
    }
  
    /* Print the value of the network function, or targets generated from it, 
       or hidden layer values, for the file or grid of points. */

    if (fname!=NULL)
    {
      static data_specifications ds0;  /* static so it's initialized to zero. */

      data_spec = logg.data['D']==0 ? &ds0 : logg.data['D'];
    
      if (logg.data['D']==0)
      { data_spec->N_inputs = a->N_inputs;
        data_spec->N_targets = m!=0 && m->type=='C' ? 1 : a->N_outputs;
        if (m!=0 && m->type=='B') 
        { data_spec->int_target = 2;
        }
        if (m!=0 && m->type=='C') 
        { data_spec->int_target = a->N_outputs;
        }
        if (m!=0 && m->type=='N')
        { data_spec->int_target = -1;
        }
        for (i = 0; i<data_spec->N_inputs+data_spec->N_targets; i++)
        { data_spec->trans[i] = data_trans_parse("I");
        }
      }

      if (strlen(fname)>=Max_data_source)
      { fprintf(stderr,"Source for inputs is too long\n");
        exit(1);
      }
      strcpy(data_spec->test_inputs,fname);

      net_data_read (0, 1, a, m, sv);

      for (i = 0; i<N_test; i++)
      {
        v = test_values+i;

        net_func (v, a, &pre, flgs, w, 1);

        if (inputs)    
        { for (j = 0; j<a->N_inputs; j++) printf(" %+.6e",v->i0[j]);
        }

        if (hid)
        { int l;
          for (l = 0; l<a->N_layers; l++)
          { if (layer<0 || layer==l)
            { for (j = 0; j<a->N_hidden[l]; j++) 
              { printf(" %+.6e",v->h0[l][j]);
              }
            }
          }
        }

        if (outputs)
        { if (gen_targets)
          { net_model_guess (v, targets, a, &pre, flgs, m, sv, w, s->noise, 1);
            for (j = 0; j<N_targets; j++) printf(" %+.6e",targets[j]);
          }
          else
          { for (j = 0; j<a->N_outputs; j++) printf(" %+.6e",v->o[j]);
          }
        }
    
        printf("\n");
      }
    }
    else  /* from grid, not file */
    {
      if (first)
      { first = 0;
      }
      else
      { printf("\n");
      }
    
      for (i = 0; i<a->N_inputs; i++) 
      { grid_point[i] = 0;
        v->i[i] = grid_low[i];
      }
    
      for (;;)
      {
        net_func (v, a, &pre, flgs, w, 1);

        if (inputs)    
        { for (j = 0; j<a->N_inputs; j++) printf(" %+.6e",v->i0[j]);
        }

        if (hid)
        { int l;
          for (l = 0; l<a->N_layers; l++)
          { if (layer<0 || layer==l)
            { for (j = 0; j<a->N_hidden[l]; j++) 
              { printf(" %+.6e",v->h0[l][j]);
              }
            }
          }
        }

        if (outputs)
        { if (gen_targets)
          { net_model_guess (v, targets, a, &pre, flgs, m, sv, w, s->noise, 1);
            for (j = 0; j<N_targets; j++) printf(" %+.6e",targets[j]);
          }
          else
          { for (j = 0; j<a->N_outputs; j++) printf(" %+.6e",v->o[j]);
          }
        }
    
        printf("\n");

        for (i = a->N_inputs-1; i>=0 && grid_point[i]==grid_size[i]; i--) 
        { grid_point[i] = 0;
          v->i[i] = grid_low[i];
        }
     
        if (i<0) break;
    
        // if (i!=a->N_inputs-1) printf("\n");
    
        grid_point[i] += 1;
        v->i[i] = grid_low[i] 
                    + grid_point[i] * (grid_high[i]-grid_low[i]) / grid_size[i];
      }
    }
  }
  
  exit(0);
}


/* DISPLAY USAGE MESSAGE AND EXIT. */

static void usage(void)
{
  fprintf (stderr, 
   "Usage: net-eval [ -i ] [ -h[#] ] [ -o | -t ] log-file range / input-file\n");
  fprintf (stderr,
   "   or: net-eval [ -i ] [ -h[#] ] [ -o | -t ] log-file range { / low high sz }\n");
  exit(1);
}
