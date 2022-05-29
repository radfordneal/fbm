/* NET-CONFIG-CHECK.C - Check/display a network configuration specification. */

/* Copyright (c) 2022 by Radford M. Neal 
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


static void usage(void);
static void print_config (net_config *, int);


/* MAIN PROGRAM. */

int main
( int argc,
  char **argv
)
{
  int p_option = 0;
  int i_option = 0;
  int ns = -1;
  int nd;

  while (argc>1)
  { if (strcmp(argv[1],"-p")==0)
    { if (p_option) usage();
      p_option = 1;
      argc -= 1;
      argv += 1;
    }
    else if (strcmp(argv[1],"-i")==0)
    { if (i_option) usage();
      i_option = 1;
      argc -= 1;
      argv += 1;
    }
    else if (strcmp(argv[1],"-I")==0)
    { if (i_option) usage();
      i_option = 2;
      argc -= 1;
      argv += 1;
    }
    else
    { break;
    }
  }

  if (argc!=3 && argc!=4) usage();

  if ((nd = atoi(argv[argc-1])) <= 0) usage();
  if (argc==4 && (ns = atoi(argv[argc-2])) <= 0) usage();

  net_config *cf = net_config_read (argv[1], ns, nd);

  if (i_option)
  { 
    fprintf (stderr, "\n%d connections, %d %s\n\n",
                     cf->N_conn,cf->N_wts, ns<0 ? "biases" : "weights");

    if (cf->N_wts>0)
    { int *scnt = chk_alloc (ns<0?1:ns, sizeof *scnt);
      int *dcnt = chk_alloc (nd, sizeof *dcnt);
      int *wcnt = chk_alloc (cf->N_wts, sizeof *wcnt);
      int i, k;
      for (k = 0; k<cf->N_conn; k++)
      { if (ns>0) scnt[cf->conn[k].s] += 1;
        dcnt[cf->conn[k].d] += 1;
        wcnt[cf->conn[k].w] += 1;
      }
      int maxs=0, mins=cf->N_conn;
      int maxd=0, mind=cf->N_conn;
      int maxw=0, minw=cf->N_conn;
      for (i = 0; i<ns; i++)
      { if (scnt[i]<mins) mins = scnt[i];
        if (scnt[i]>maxs) maxs = scnt[i];
      }
      for (i = 0; i<nd; i++)
      { if (dcnt[i]<mind) mind = dcnt[i];
        if (dcnt[i]>maxd) maxd = dcnt[i];
      }
      for (i = 0; i<cf->N_wts; i++)
      { if (wcnt[i]<minw) minw = wcnt[i];
        if (wcnt[i]>maxw) maxw = wcnt[i];
      }
      if (ns>0)
      { fprintf (stderr, "Connections per source: ");
        if (mins==maxs) fprintf(stderr,"%d\n",mins);
        else fprintf(stderr,"%d to %d\n",mins,maxs);
        if (i_option>1)
        { fprintf(stderr,"\n");
          for (i = 0; i<ns; i++)
          { fprintf(stderr,"%6d %6d\n",i+1,scnt[i]);
          }
          fprintf(stderr,"\n");
        }
      }
      fprintf (stderr, "Connections per destination: ");
      if (mind==maxd) fprintf(stderr,"%d\n",mind);
      else fprintf(stderr,"%d to %d\n",mind,maxd);
      if (i_option>1)
      { fprintf(stderr,"\n");
        for (i = 0; i<nd; i++)
        { fprintf(stderr,"%6d %6d\n",i+1,dcnt[i]);
        }
        fprintf(stderr,"\n");
      }
      fprintf (stderr, "Connections per %s: ", ns<0 ? "bias" : "weight");
      if (minw==maxw) fprintf(stderr,"%d\n\n",minw);
      else fprintf(stderr,"%d to %d\n\n",minw,maxw);
      if (i_option>1)
      { for (i = 0; i<cf->N_wts; i++)
        { fprintf(stderr,"%6d %6d\n",i+1,wcnt[i]);
        }
        fprintf(stderr,"\n");
      }
    }
  }

  if (p_option)
  { print_config (cf, ns<0);
  }

  exit(0);
}


/* PRINT WEIGHT CONFIGURATION. */

static void print_config (net_config *cf, int biases)
{ int k;
  for (k = 0; k<cf->N_conn; k++)
  { if (!biases) 
    { printf ("%6d ", cf->conn[k].s+1);
    }
    printf ("%6d %6d\n", cf->conn[k].d+1, cf->conn[k].w+1);
  }
}


/* DISPLAY USAGE MESSAGE AND EXIT. */

static void usage(void)
{
  fprintf(stderr,
          "Usage: net-config-check [ -p ] [ -i | -I ] config-file [ ns ] nd\n");
  exit(1);
}

