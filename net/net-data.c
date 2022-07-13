/* NET-DATA.C - Procedures for reading data for neural networks. */

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
#include "data.h"
#include "prior.h"
#include "model.h"
#include "net.h"
#include "net-data.h"
#include "numin.h"


/* VARIABLES HOLDING DATA.  As declared in net-data.h. */

data_specifications *data_spec;	/* Specifications of data sets */

int N_train;			/* Number of training cases */
int N_inputs;			/* Number of input values, as in data_spec */
int N_targets;			/* Number of target values, as in data_spec */

net_values *train_values;	/* Values for training cases */
net_value *train_iblock;	/* Block of input values for training cases */
net_value *train_targets;	/* True training case targets */

double train_zero_frac;		/* Fraction of input values that are zero */

int N_test;			/* Number of test cases */

net_values *test_values;	/* Values associated with test cases */
net_value *test_iblock;		/* Block of input values for test cases */
net_value *test_targets;	/* True targets for test cases */


/* PROCEDURES. */

static net_value *read_targets  (numin_source *, int,   net_arch *);
static net_values *read_inputs  (numin_source *, int *, net_arch *, 
                                 model_specification *, model_survival *,
                                 net_value **);


/* FREE SPACE OCCUPIED BY DATA.  Also useful as a way to reset when the
   architecture changes. */

void net_data_free (void)
{
  if (train_values!=0)
  { free(train_values);  /* doesn't free what's pointed to yet... */
    train_values = 0;
    N_train = 0;
  }

  if (train_targets!=0)
  { free(train_targets);
    train_targets = 0;
  }

  if (test_values!=0)
  { free(test_values);  /* doesn't free what's pointed to yet... */
    test_values = 0;
    N_test = 0;
  }

  if (test_targets!=0)
  { free(test_targets);
    test_targets = 0;
  }
}


/* READ TRAINING AND/OR TEST DATA FOR NETWORK.  Either or both of these
   data sets are read, depending on the options passed.  If a data set
   has already been read, it isn't read again.  This procedure also checks
   that the data specifications are consistent with the network architecture. 

   When training data is read, train_zero_frac is computed.

   For survival models with non-constant hazard, the first input in a case, 
   representing time, is set to zero by this procedure. */

void net_data_read
( int want_train,	/* Do we want the training data? */
  int want_test,	/* Do we want the test data? */
  net_arch *arch,	/* Network architecture */
  model_specification *model, /* Data model being used */
  model_survival *surv	/* Survival model, or zero if irrelevant */
)
{
  numin_source ns;

  if (train_values!=0) want_train = 0;
  if (test_values!=0)  want_test = 0;

  N_inputs = data_spec->N_inputs;
  N_targets = data_spec->N_targets;

  if (model!=0 && model->type=='C' && N_targets!=1)
  { fprintf(stderr,"Only one target is allowed for 'class' models\n");
    exit(1);
  }

  model_values_check(model,data_spec,arch->N_outputs,"");

  if (arch->N_inputs 
       != N_inputs + (model!=0 && model->type=='V' && surv->hazard_type!='C'))
  { fprintf(stderr,
     "Number of inputs in data specification doesn't match network inputs\n");
    exit(1);
  }

  if (want_train)
  { 
    numin_spec (&ns, "data@1,0",1);
    numin_spec (&ns, data_spec->train_inputs, N_inputs);
    train_values = read_inputs(&ns, &N_train, arch, model, surv, &train_iblock);

    int i, j;
    train_zero_frac = 0;
    for (i = 0; i<N_train; i++) 
    { for (j = 0; j<arch->N_inputs; j++)
      { if (train_values[i].i0[j]==0) train_zero_frac += 1;
      }
    }
    train_zero_frac /= N_train * arch->N_inputs;

    numin_spec (&ns, data_spec->train_targets, N_targets);
    train_targets = read_targets (&ns, N_train, arch);
  }

  if (want_test && data_spec->test_inputs[0]!=0)
  {
    numin_spec (&ns, "data@1,0",1);
    numin_spec (&ns, data_spec->test_inputs, N_inputs);
    test_values = read_inputs (&ns, &N_test, arch, model, surv, &test_iblock);

    if (data_spec->test_targets[0]!=0)
    { numin_spec (&ns, data_spec->test_targets, N_targets);
      test_targets = read_targets (&ns, N_test, arch);
    }
  }
}


/* READ INPUTS VALUES FOR A SET OF CASES. */

static net_values *read_inputs
( numin_source *ns,
  int *N_cases_ptr,
  net_arch *arch,
  model_specification *model, 
  model_survival *surv,
  net_value **input_block
)
{
  net_value *value_block;
  net_values *values;
  int value_count;
  int N_cases;
  int i, j, j0;

  N_cases = numin_start(ns);

  value_count = net_setup_value_count_aligned
                 (arch, NET_VALUE_ALIGN_ELEMENTS, 0, 1);

  value_block = 
    (net_value *) chk_alloc (value_count*N_cases, sizeof *value_block);

  *input_block = 
    (net_value *) chk_alloc (arch->N_inputs*N_cases, sizeof **input_block);

  values = 
    (net_values *) chk_alloc (N_cases, sizeof *values);

  for (i = 0; i<N_cases; i++) 
  { net_setup_value_pointers_aligned (&values[i], value_block+value_count*i, 
                                      arch, NET_VALUE_ALIGN_ELEMENTS,
                                      (*input_block)+arch->N_inputs*i, 0);
  }

  double ind[N_inputs];

  for (i = 0; i<N_cases; i++) 
  { if (model!=0 && model->type=='V' && surv->hazard_type!='C')
    { values[i].i0[0] = 0;
      j0 = 1;
    }
    else
    { j0 = 0;
    }
    numin_read(ns,ind);
    for (j = j0; j<arch->N_inputs; j++)
    { values[i].i0[j] = data_trans (ind[j-j0], data_spec->trans[j-j0]);
    }
  }

  numin_close(ns);

  *N_cases_ptr = N_cases;

  return values;
}


/* READ TARGET VALUES FOR A SET OF CASES. */

static net_value *read_targets
( numin_source *ns,
  int N_cases,
  net_arch *arch
)
{
  net_value *tg;
  int i, j;

  if (numin_start(ns)!=N_cases)
  { fprintf(stderr,
      "Number of input cases doesn't match number of target cases\n");
    exit(1);
  }

  tg = (net_value *) chk_alloc (N_targets*N_cases, sizeof *tg);

  double tgd[N_targets];

  for (i = 0; i<N_cases; i++)
  { 
    numin_read(ns,tgd);

    for (j = 0; j<N_targets; j++)
    { tg[N_targets*i+j] = data_trans (tgd[j], data_spec->trans[N_inputs+j]);
    }
  }

  numin_close(ns);

  return tg;
}
