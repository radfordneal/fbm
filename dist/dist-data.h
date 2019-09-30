/* DIST-DATA.H - Interface for reading data for Bayesian model. */

/* Copyright (c) 1995-2003 by Radford M. Neal 
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


/* VARIABLES HOLDING TRAINING DATA.  The inputs and targets for training
   and test cases are stored with all the inputs or targets for one case
   together - for example, input i for training case c is stored here in 
   train_inputs[N_inputs*c+i]. */

extern data_specifications *data_spec; /* Specifications of data sets */

extern int N_train;		/* Number of training cases */

extern double *train_inputs;	/* Inputs for training cases */
extern double *train_targets;	/* True targets for training cases */


/* PROCEDURES. */

void dist_data_read (void);
void dist_data_free (void);
