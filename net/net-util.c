/* NET-UTIL.C - Various utility procedures for use in neural network code. */

/* Copyright (c) 1995-2004 by Radford M. Neal 
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
#include "data.h"
#include "prior.h"
#include "model.h"
#include "net.h"


/* SET UP REQUIRED RECORD SIZES.  Doesn't set the sizes of the variable-sized
   'S' and 'W' records. */

void net_record_sizes
( log_gobbled *logg	/* Structure to hold gobbled data */
)
{
  logg->req_size['A'] = sizeof (net_arch);
  logg->req_size['F'] = sizeof (net_flags);
  logg->req_size['M'] = sizeof (model_specification);
  logg->req_size['V'] = sizeof (model_survival);
  logg->req_size['P'] = sizeof (net_priors);
}


/* REPORT ERROR IF SPECS FOR NET ARCHITECTURE, DATA MODEL, ETC. ARE MISSING. */

void net_check_specs_present
( net_arch *a,			/* Architecture, or null */
  net_priors *p,		/* Priors, or null */
  int need_model,		/* Must model be present? */
  model_specification *m,	/* Model, or null */
  model_survival *v		/* Survival model parameters, or null */
)
{
  if (a==0)
  { fprintf(stderr,"No architecture specification in log file\n");
    exit(1);
  }

  if (p==0)
  { fprintf(stderr,"No prior specification in log file\n");
    exit(1);
  }

  if (m==0 && need_model)
  { fprintf(stderr,"No model specification in log file\n");
    exit(1);
  }

  if (m!=0 && m->type=='V' && v==0)
  { fprintf(stderr,"No hazard specification for survival model\n");
    exit(1);
  }
}


/* READ WEIGHT CONFIGURATION.  Passed the configuration file and numbers
   units in the source and destination layers.  Returns a newly-allocated
   structure with the configuration. */

net_config *net_config_read (char *file, int ns, int nd)
{ 
  FILE *fp = file[0]=='%' ? popen(file+1,"r") : fopen(file,"r");
  if (fp==NULL)
  { fprintf(stderr,"Can't open weight configuration file: %s\n",file);
    exit(2);
  }
  
  net_config *p;
  p = calloc (1, sizeof *p);
  if (p==NULL || (p->conn = calloc (Max_conn, sizeof *p->conn))==NULL)
  { fprintf(stderr,"Can't allocate memory for weight configuration\n");
    exit(3);
  }

  p->N_wts = 0;
  p->N_conn = 0;

  for (;;)
  { 
    int s, d, w, n;

    n = fscanf(fp,"%d%d%d\n",&s,&d,&w);
    if (n==EOF) break;

    if (n!=3 || s<1 || d<1 || w<1 || s>ns || d>nd)
    { fprintf (stderr, "Bad weight configuration file: %s, item %d\n",
               file, p->N_conn+1);
      exit(2);
    }

    if (p->N_conn==Max_conn)
    { fprintf (stderr,"Too many connections in weight configuration file: %s\n",
               file);
      exit(2);
    }

    p->conn[p->N_conn].s = s;
    p->conn[p->N_conn].d = d;
    p->conn[p->N_conn].w = w;

    if (w>p->N_wts) p->N_wts = w;

    p->N_conn += 1;
  }

  net_connection *q;
  q = calloc (p->N_conn, sizeof *q);
  if (q==NULL)
  { fprintf(stderr,"Can't allocate memory for weight configuration\n");
    exit(3);
  }
 
  memcpy (q, p->conn, p->N_conn * sizeof *q);
  free(p->conn);
  p->conn = q;

  return p;
}
