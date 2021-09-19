/* NET-DATA.H - Interface to module for reading data for networks. */

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


#ifndef __managed__
#define __managed__  /* don't declare some variables as "managed" by CUDA */
#endif


/* VARIABLES HOLDING TRAINING AND/OR TEST DATA.  When the values or targets
   aren't known, the pointers are null. */

extern data_specifications *data_spec; /* data set specifications */

__managed__ extern int N_train;	  /* Number of training cases */
__managed__ extern int N_inputs;  /* Number of input values, as in data_spec */
__managed__ extern int N_targets; /* Number of target values, as in data_spec */

__managed__ extern net_values *train_values;  /* Values for training cases */
__managed__ extern double *train_targets;  /* True targets for training cases */

extern int N_test;		/* Number of test cases */

extern net_values *test_values;	/* Values associated with test cases */
extern double *test_targets;	/* True targets for test cases */


/* PROCEDURES. */

void net_data_read (int, int, net_arch *, 
                    model_specification *, model_survival *);

void net_data_free (void);
