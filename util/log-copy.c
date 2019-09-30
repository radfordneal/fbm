/* LOG-COPY.C - Program to copy part of a log file. */

/* Copyright (c) 1995 by Radford M. Neal 
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


static void usage(void);


/* MAIN PROGRAM. */

main
( int argc,
  char **argv
)
{
  log_file logf_in, logf_out;
  int low, high, modulus, neg_only;
  int olow, omod, lasti, lasto;
  void *data;
  char *p;

  if (argc!=4 && argc!=5) usage();

  logf_in.file_name = argv[1];
  log_file_open(&logf_in,0);

  neg_only = strcmp(argv[2],"-")==0;

  if (!neg_only)
  { parse_range(argv[2],&low,&high,&modulus);
    if (modulus<0)
    { fprintf(stderr,"Bad range specification: %s\n",argv[2]);
      exit(1);
    }
    if (high==-2) high = low;
  }

  olow = -1;

  if (argc==5)
  { if (*argv[4]=='%' || *argv[4]==0) usage();
    olow = atoi(argv[4]);
    for (p = argv[4]; *p!='%' && *p!=0; p++) ;
    omod = modulus;
    if (*p=='%')
    { omod = atoi(p+1);
    }
    if (olow<0 || omod<1) usage();
  }

  logf_out.file_name = argv[3];
  log_file_create(&logf_out);

  lasti = -1;
  lasto = olow - omod;

  while (!logf_in.at_end && 
   (neg_only ? logf_in.header.index<0 : (high<0 || logf_in.header.index<=high)))
  {
    data = 0;

    if (logf_in.header.index<0 
     || logf_in.header.index>=low && logf_in.header.index%modulus==0)
    { 
      data = malloc(logf_in.header.size);
      if (data==0)
      { fprintf(stderr,"Not enough memory for record of size %d - ignored\n",
          logf_in.header.size);
      }

      if (olow>=0 && logf_in.header.index>=0)
      { if (logf_in.header.index!=lasti)
        { lasto += omod;
          lasti = logf_in.header.index;
        }
      }
    }

    if (data==0)
    { log_file_forward(&logf_in);
    }
    else
    {
      logf_out.header = logf_in.header;

      if (olow>=0 && logf_in.header.index>=0)
      { logf_out.header.index = lasto;
      }

      log_file_read (&logf_in, data, logf_in.header.size);

      log_file_append (&logf_out, data);

      free(data);
    } 
  }

  log_file_close(&logf_in);
  log_file_close(&logf_out);
  
  exit(0);
}


/* PRINT USAGE MESSAGE AND EXIT. */

void usage(void)
{ 
  fprintf(stderr,
    "Usage: log-copy input-log-file range output-log-file [ low[%%mod] ]\n");
  exit(1);
}
