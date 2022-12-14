/* NET-PLT.C - Procedures used to plot data on neural networks. */

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
#include "log.h"
#include "quantities.h"
#include "mc.h"
#include "mc-quantities.h"


extern void net_arguments  (char ***);
extern void net_initialize (log_gobbled *);
extern void net_available  (struct quantdesc[Max_quantities], log_gobbled *);
extern void net_evaluate   (struct quantdesc[Max_quantities], quantities_held *,
                            log_gobbled *);
extern void net_cleanup    (void);

void (*quant_app_arguments[]) (char ***) =
{ net_arguments,
  0 
};

void (*quant_app_record_sizes[]) (log_gobbled *) =
{ mc_record_sizes,
  0 
};

void (*quant_app_initialize[]) (log_gobbled *) =
{ mc_initialize,   /* do first, so KLUDGE in net-mc.c will be seen later */
  net_initialize,
  0 
};

void (*quant_app_available[]) (struct quantdesc[Max_quantities], log_gobbled*) =
{ net_available,
  mc_available, 
  0 
};

void (*quant_app_evaluate[])(struct quantdesc[Max_quantities], quantities_held*,
                             log_gobbled *) =
{ net_evaluate,
  mc_evaluate,
  0
};

void (*quant_app_cleanup[]) (void) = 
{ net_cleanup,
  0
};


/* DISPLAY USAGE MESSAGE. */

void plt_usage
( char *str
)
{ 
  fprintf(stderr,"Usage: net-%s [ / test-inputs [ test-targets ] ]\n",str);
}
