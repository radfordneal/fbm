/* NET-SETUP.C - Procedures for setting up network data structures. */

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

#include "cuda-use.h"

#include "misc.h"
#include "log.h"
#include "prior.h"
#include "data.h"
#include "model.h"
#include "net.h"


/* This module contains routines for setting up the structures containing
   pointers to arrays that are used to describe the parameters, hyperparameters,
   and unit values for networks.  

   The scheme used allows all quantites of a given type to be stored in
   a contiguous block of storage.  The pointer structure allows the various 
   sections of this block to be easily accessed. 

   Sub-groups of parameters or hyperparameters may also be located via
   a group index, using the net_setup_hyper_group and net_setup_param_group
   procedures.
*/


#define ALIGN(o,a) (((o)+(a)-1)&~((a)-1))  /* Align o to a multiple of a, which
                                              must be a power of two */


/* RETURN NUMBER OF HYPERPARAMETERS.  Returns the number of 'sigma' values
   for a network with the given architecture. */
   
unsigned net_setup_sigma_count
( net_arch *a,		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  model_specification *m /* Data model */
)
{ 
  unsigned count, b;
  int l, ls;

  count = 0;

  if (a->has_ti) count += 1;
  
  for (l = 0; l<a->N_layers; l++)
  { 
    for (ls = 0, b = a->has_nsq[l]; b!=0; ls++, b>>=1)
    { if (b&1)
      { if (ls>=l-1) abort();
        if (flgs && flgs->nonseq_config[ls])
        { count += 1;
        }
        else
        { count += 1 + a->N_hidden[ls];
        }
      }
    }

    if (l>0 && a->has_hh[l-1])
    { if (flgs && flgs->hidden_config[l])
      { count += 1;
      }
      else
      { count += 1 + a->N_hidden[l-1];
      }
    }

    if (a->has_ih[l]) 
    { if (flgs && flgs->input_config[l])
      { count += 1;
      }
      else
      { count += 1 + not_omitted(flgs?flgs->omit:0,a->N_inputs,1<<(l+1));
      }
    }

    if (a->has_bh[l]) count += 1;
    if (a->has_ah[l]) count += a->N_hidden[l];
    if (a->has_th[l]) count += 1;

    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (flgs && flgs->hidden_config[k])
      { count += 1;
      }
      else
      { count += 1 + a->N_hidden[l];
      }
    }
  }

  if (a->has_io) 
  { if (flgs && flgs->input_config[a->N_layers])
    { count += 1;
    }
    else
    { count += 1 + not_omitted(flgs?flgs->omit:0,a->N_inputs,1);
    }
  }
  if (a->has_bo) count += 1;
  if (a->has_ao) count += a->N_outputs;

  if (m!=0 && m->type=='R') count += 1 + a->N_outputs;
  
  return count;
}


/* RETURN NUMBER OF NETWORK PARAMETERS.  Returns the number of
   weights, biases, and offsets in a network with the given
   architecture.  

   Also reads any weight/bias configuration files, and fills in the
   configuration pointers in the net_arch structure.

   Furthermore, if the 'pre' pointer is non-null, it precomputes 
   (some of) the information stored there.  */

unsigned net_setup_param_count
( net_arch *a,		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  net_precomputed *pre	/* Place to store pre-computed information, or null */
)
{
  unsigned count, b;
  int l, ls, nsqi;

  count = 0;
 
  if (a->has_ti) count += a->N_inputs;

  nsqi = 0;
  for (l = 0; l<a->N_layers; l++)
  {
    for (ls = 0, b = a->has_nsq[l]; ls<l; ls++, b>>=1)
    { if (b&1)
      { if (pre)
        { pre->nonseq[ls][l] = nsqi;
        }
        if (flgs && flgs->nonseq_config[nsqi])
        { a->nonseq_config[nsqi] = 
            net_config_read (flgs->config_files + flgs->nonseq_config[nsqi],
                             a->N_hidden[ls], a->N_hidden[l]);
          count += a->nonseq_config[nsqi]->N_wts;
        }
        else
        { count += a->N_hidden[ls]*a->N_hidden[l];
        }
        nsqi += 1;
      }
      else
      { if (pre)
        { pre->nonseq[ls][l] = -1;
        }
      }
    }

    if (l>0 && a->has_hh[l-1])
    { if (flgs && flgs->hidden_config[l])
      { a->hidden_config[l] = 
          net_config_read (flgs->config_files + flgs->hidden_config[l],
                           a->N_hidden[l-1], a->N_hidden[l]);
        count += a->hidden_config[l]->N_wts;
      }
      else
      { count += a->N_hidden[l-1]*a->N_hidden[l];
      }
    }

    if (a->has_ih[l]) 
    { if (flgs && flgs->input_config[l])
      { a->input_config[l] =
          net_config_read (flgs->config_files + flgs->input_config[l],
                           a->N_inputs, a->N_hidden[l]);
        count += a->input_config[l]->N_wts;
      }
      else
      { count += not_omitted(flgs?flgs->omit:0,a->N_inputs,1<<(l+1))
                  * a->N_hidden[l];
      }
    }

    if (a->has_bh[l]) 
    { if (flgs && flgs->bias_config[l])
      { a->bias_config[l] =
          net_config_read (flgs->config_files + flgs->bias_config[l],
                           -1, a->N_hidden[l]);
        count += a->bias_config[l]->N_wts;
      }
      else
      { count += a->N_hidden[l];
      }
    }

    if (a->has_th[l]) count += a->N_hidden[l];

    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (flgs && flgs->hidden_config[k])
      { a->hidden_config[k] = 
          net_config_read (flgs->config_files + flgs->hidden_config[k],
                           a->N_hidden[l], a->N_outputs);
        count += a->hidden_config[k]->N_wts;
      }
      else
      { count += a->N_hidden[l]*a->N_outputs;
      }
    }
  }

  if (a->has_io) 
  { if (flgs && flgs->input_config[a->N_layers])
    { a->input_config[a->N_layers] = 
        net_config_read (flgs->config_files + flgs->input_config[a->N_layers],
                         a->N_inputs, a->N_outputs);
      count += a->input_config[a->N_layers]->N_wts;
    }
    else
    { count += not_omitted(flgs?flgs->omit:0,a->N_inputs,1)
                * a->N_outputs;
    }
  }

  if (a->has_bo)
  { if (flgs && flgs->bias_config[a->N_layers])
    { a->bias_config[a->N_layers] =
        net_config_read (flgs->config_files + flgs->bias_config[a->N_layers],
                         -1, a->N_outputs);
      count += a->bias_config[l]->N_wts;
    }
    else
    { count += a->N_outputs;
    }
  }

  return count;
}


/* RETURN NUMBER OF UNIT-RELATED VALUES IN NETWORK.  Returns the number 
   of unit-related values for a network with the given architecture. 
   Includes the inputs and the outputs. */

unsigned net_setup_value_count
( net_arch *a		/* Network architecture */
)
{ 
  return net_setup_value_count_aligned (a, 1, 1, 1);
}


/* RETURN NUMBER OF UNIT-RELATED VALUES IN NETWORK, WITH ALIGNMENT.  Returns
   the number of unit-related values for a network with the given architecture,
   including padding for alignment to 'align' boundary (elements, not bytes,
   must be a power of two).  May or may not include inputs and outputs, 
   depending on the last two arguments. */

unsigned net_setup_value_count_aligned
( net_arch *a,		/* Network architecture */
  int align,		/* Alignment, must be a power of 2 */
  int include_inputs,	/* Should inputs be included in the count? */
  int include_outputs	/* Should outputs be included in the count? */
)
{ 
  unsigned count = 0;
  int l;

  if (include_inputs) 
  { count += ALIGN(a->N_inputs,align);
  }

  for (l = 0; l<a->N_layers; l++)
  { int n = a->N_hidden[l];
    if (a->layer_type[l] == Normalize_type)
    { int c = a->N_channels[l];
      n += c>0 ? c : a->N_hidden[l]/(-c);
    } 
    count += ALIGN(n,align);
  }

  if (include_outputs)
  { count += ALIGN(a->N_outputs,align);
  }

  return count;
}


/* SET UP POINTERS TO HYPERPARAMETERS.  Sets the pointers in the net_sigmas
   structure to point the appropriate places in a block of sigma values.
   Pointers associated with parts of the network that don't exist are set to 
   zero.  

   The size of the storage block, as returned by the net_setup_sigma_count 
   procedure, must be stored by the caller in the total_sigmas field, and
   the caller must also set the sigma_block field to point to a block of this
   many net_sigma values. */

void net_setup_sigma_pointers
( net_sigmas *s,	/* Structure to set up pointers in */
  net_arch *a,		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  model_specification *m /* Data model */
)
{ 
  net_sigma *b;
  unsigned bits;
  int l, ls;
  int nsqi;

  b = s->sigma_block;
  nsqi = 0;

  s->ti_cm = a->has_ti ? b++ : 0;

  for (l = 0; l<a->N_layers; l++)
  {
    for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (bits&1)
      { if (ls>=l-1) abort();
        s->nsq_cm[nsqi] = b++;
        s->nsq[nsqi] = 0;
        if (!flgs || !flgs->nonseq_config[ls])
        { s->nsq[nsqi] = b;
          b += a->N_hidden[ls];
        }
        nsqi += 1;
      }
    }

    if (l>0)
    { s->hh_cm[l-1] = a->has_hh[l-1] ? b++ : 0;
      s->hh[l-1] = 0;
      if (a->has_hh[l-1] && !(flgs && flgs->hidden_config[l])) 
      { s->hh[l-1] = b;
        b += a->N_hidden[l-1];
      }
    }

    s->ih_cm[l] = a->has_ih[l] ? b++ : 0;
    s->ih[l] = 0;
    if (a->has_ih[l] && !(flgs && flgs->input_config[l])) 
    { s->ih[l] = b;
      b += not_omitted(flgs?flgs->omit:0,a->N_inputs,1<<(l+1));
    }

    s->bh_cm[l] = a->has_bh[l] ? b++ : 0;

    s->ah[l] = 0;
    if (a->has_ah[l])
    { s->ah[l] = b;
      b += a->N_hidden[l];
    }
  
    s->th_cm[l] = a->has_th[l] ? b++ : 0;
  }

  for (l = a->N_layers-1; l>=0; l--)
  { s->ho_cm[l] = a->has_ho[l] ? b++ : 0;
    s->ho[l] = 0;
    if (a->has_ho[l] && !(flgs && flgs->hidden_config[2*a->N_layers-1-l])) 
    { s->ho[l] = b;
      b += a->N_hidden[l];
    }
  }

  s->io_cm = a->has_io ? b++ : 0;
  s->io = 0;
  if (a->has_io && !(flgs && flgs->input_config[a->N_layers]))
  { s->io = b;
    b += not_omitted(flgs?flgs->omit:0,a->N_inputs,1);
  }

  s->bo_cm = a->has_bo ? b++ : 0;

  s->ao = 0;
  if (a->has_ao)
  { s->ao = b;
    b += a->N_outputs;
  }

  s->noise_cm = m!=0 && m->type=='R' ? b++ : 0;
  s->noise = 0;
  if (m!=0 && m->type=='R') 
  { s->noise = b;
    b += a->N_outputs;
  }
}


/* SET UP POINTERS TO NETWORK PARAMETERS.  Sets the pointers in the net_params
   structure to point the appropriate places in a block of parameter values.
   Pointers associated with parts of the network that don't exist are set to 
   zero.  

   The size of the storage block, as returned by the net_setup_param_count
   procedure, must be stored by the caller in the total_params field, and
   the caller must also set the param_block field to point to a block of this 
   many net_param values. */

void net_setup_param_pointers
( net_params *w,	/* Structure to set up pointers in */
  net_arch *a,		/* Network architecture */
  net_flags *flgs	/* Network flags, null if none */
)
{
  net_param *b;
  unsigned bits;
  int l, ls;
  int nsqi;

  b = w->param_block;
  nsqi = 0;

  w->ti = 0;
  if (a->has_ti)
  { w->ti = b;
    b += a->N_inputs;
  }

  for (l = 0; l<a->N_layers; l++)
  {
    for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (bits&1)
      { if (ls>=l-1) abort();
        w->nsq[nsqi] = b;
        b += a->nonseq_config[nsqi] ? a->nonseq_config[nsqi]->N_wts
              : a->N_hidden[ls]*a->N_hidden[l];
        nsqi += 1;
      }
    }

    if (l>0)
    { w->hh[l-1] = 0;
      if (a->has_hh[l-1]) 
      { w->hh[l-1] = b;
        b += a->hidden_config[l] ? a->hidden_config[l]->N_wts
              : a->N_hidden[l-1]*a->N_hidden[l];
      }
    }

    w->ih[l] = 0;
    if (a->has_ih[l]) 
    { w->ih[l] = b;
      b += a->input_config[l] ? a->input_config[l]->N_wts
         : not_omitted(flgs?flgs->omit:0,a->N_inputs,1<<(l+1)) * a->N_hidden[l];
    }
  
    w->bh[l] = 0;
    if (a->has_bh[l])
    { w->bh[l] = b;
      b += a->bias_config[l] ? a->bias_config[l]->N_wts : a->N_hidden[l];
    }
  
    w->th[l] = 0;
    if (a->has_th[l])
    { w->th[l] = b;
      b += a->N_hidden[l];
    }
  }

  for (l = a->N_layers-1; l>=0; l--)
  { w->ho[l] = 0;
    if (a->has_ho[l]) 
    { int k = 2*a->N_layers-1-l;
      w->ho[l] = b;
      b += a->hidden_config[k] ? a->hidden_config[k]->N_wts
                               : a->N_hidden[l]*a->N_outputs;
    }
  }

  w->io = 0;
  if (a->has_io) 
  { w->io = b;
    b += a->input_config[a->N_layers] ? a->input_config[a->N_layers]->N_wts
       : not_omitted(flgs?flgs->omit:0,a->N_inputs,1) * a->N_outputs;
  }

  w->bo = 0;
  if (a->has_bo)
  { w->bo = b;
    b += a->bias_config[a->N_layers] ? a->bias_config[a->N_layers]->N_wts 
                                     : a->N_outputs;
  }
}


/* SET UP POINTERS TO NETWORK PARAMETER STRUCTURES FOR GRADIENTS.

   The 'w' argument points to 'n' structures to be set up.  The first
   must have the total_params fields set to the number of parameters,
   and the param_block field set to the storage for all 'n' sets of
   parameters.  The aligned_total argument gives the number of
   parameters after any increase for desired alignment.

   If 'v' is 1, 'n' structures in the space pointed to by 'w' are set
   up to point into separate parameter blocks.  If 'v' is greater than
   1, 'n' structures are set up that point to parameters with v-way
   interleaving, with each parameter group being associated with 'v'
   times as many net_param values.
 */

void net_setup_gradients
( net_params *w,	/* Array of structures to set up pointers in */
  int n,		/* Number of param structures to set up */
  int v,		/* Interleaving group size, 1 for no interleaving */
  net_arch *a,		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  unsigned aligned_total  /* Number of parameters with increase for alignment */
)
{
  unsigned total_params = w->total_params;
  net_param *param_block = w->param_block;

  unsigned bits;
  net_param *b;
  int l, ls;
  int nsqi;
  int i, j;

  /* Set up each structure. */

  for (i = 0; i<n; i++)
  { 
    /* Find start of parameters for this structure.  There are i/v previous
       interleave groups, each with v*aligned_total parameters.  This 
       structure within the group is offset by i%v. */

    b = param_block + (i/v)*v*aligned_total + i%v;
    w[i].total_params = total_params;
    w[i].param_block = b;

    /* Look for what is present in the architecture.  Space for parameters
       is multipled by v, to account for interleaving. */

    nsqi = 0;

    w[i].ti = 0;
    if (a->has_ti)
    { w[i].ti = b;
      b += v * a->N_inputs;
    }

    for (l = 0; l<a->N_layers; l++)
    {
      for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
      { if (bits&1)
        { if (ls>=l-1) abort();
          w[i].nsq[nsqi] = b;
          b += v * (a->nonseq_config[nsqi] ? a->nonseq_config[nsqi]->N_wts
                     : a->N_hidden[ls]*a->N_hidden[l]);
          nsqi += 1;
        }
      }

      if (l>0)
      { w[i].hh[l-1] = 0;
        if (a->has_hh[l-1]) 
        { w[i].hh[l-1] = b;
          b += v * (a->hidden_config[l] ? a->hidden_config[l]->N_wts
                     : a->N_hidden[l-1]*a->N_hidden[l]);
        }
      }

      w[i].ih[l] = 0;
      if (a->has_ih[l]) 
      { w[i].ih[l] = b;
        b += v * (a->input_config[l] ? a->input_config[l]->N_wts
                   : not_omitted(flgs?flgs->omit:0,a->N_inputs,1<<(l+1)) 
                      * a->N_hidden[l]);
      }
    
      w[i].bh[l] = 0;
      if (a->has_bh[l])
      { w[i].bh[l] = b;
        b += v * (a->bias_config[l] ? a->bias_config[l]->N_wts 
                                    : a->N_hidden[l]);
      }
    
      w[i].th[l] = 0;
      if (a->has_th[l])
      { w[i].th[l] = b;
        b += v * a->N_hidden[l];
      }
    }

    for (l = a->N_layers-1; l>=0; l--)
    { w[i].ho[l] = 0;
      if (a->has_ho[l]) 
      { int k = 2*a->N_layers-1-l;
        w[i].ho[l] = b;
        b += v * (a->hidden_config[k] ? a->hidden_config[k]->N_wts
                                      : a->N_hidden[l]*a->N_outputs);
      }
    }

    w[i].io = 0;
    if (a->has_io) 
    { w[i].io = b;
      b += v * (a->input_config[a->N_layers] 
                 ? a->input_config[a->N_layers]->N_wts
                 : not_omitted(flgs?flgs->omit:0,a->N_inputs,1) * a->N_outputs);
    }

    w[i].bo = 0;
    if (a->has_bo)
    { w[i].bo = b;
      b += v * (a->bias_config[a->N_layers] 
                 ? a->bias_config[a->N_layers]->N_wts 
                 : a->N_outputs);
    }
  }
}


/* REPLICATE POINTERS TO NETWORK PARAMETERS.  Given a pointer to a
   net_params structure, this procedure replicates that structure n-1
   times in structures that follow, offsetting the param_block pointer
   and pointers to parts by 'offset' for each replication.  The 'offset'
   parameter might be set to the w->total_params, or to some larger
   value (for alignment purposes). */

void net_replicate_param_pointers
( net_params *w,	/* The array of structures to replicate the first of */
  net_arch *a,		/* Network architecture */
  int n,		/* Number of structures to end up with */
  unsigned offset	/* Offset to get to next section of param_block */
)
{
  net_params *w1, *w2;
  int i, l, ls, nsqi;
  unsigned bits;

  for (i = 1; i<n; i++)
  { 
    w1 = w+i-1;
    w2 = w+i;

    w2->total_params = w1->total_params;
    w2->param_block = w1->param_block + offset;

    if (a->has_ti) w2->ti = w1->ti + offset;

    nsqi = 0;
    for (l = 0; l<a->N_layers; l++)
    { for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
      { if (bits&1)
        { if (ls>=l-1) abort();
          w2->nsq[nsqi] = w1->nsq[nsqi] + offset;
          nsqi += 1;
        }
      }
      if (l>0)
      { if (a->has_hh[l-1]) w2->hh[l-1] = w1->hh[l-1] + offset;
      }
      if (a->has_ih[l]) w2->ih[l] = w1->ih[l] + offset;
      if (a->has_bh[l]) w2->bh[l] = w1->bh[l] + offset;
      if (a->has_th[l]) w2->th[l] = w1->th[l] + offset;
    }

    for (l = a->N_layers-1; l>=0; l--)
    { if (a->has_ho[l]) w2->ho[l] = w1->ho[l] + offset;
    }

    if (a->has_io) w2->io = w1->io + offset;
    if (a->has_bo) w2->bo = w1->bo + offset;
  }
}


/* SET UP POINTERS TO UNIT VALUES.  Sets the pointers in the net_values
   structure to point to the appropriate places in the block of net_value
   values passed.  The size of this block must be as indicated by the
   net_setup_value_count procedure, with inputs/outputs included according
   to match whether they are provided separately here. */

void net_setup_value_pointers
( net_values *v,	/* Structure to set up pointers in */
  net_value *b,		/* Block of 'value' values */
  net_arch *a,		/* Network architecture */
  net_value *inputs,	/* Input values, 0 if part of value block */
  net_value *outputs	/* Output values, 0 if part of value block */
)
{
  net_setup_value_pointers_aligned (v, b, a, 1, inputs, outputs);
}


/* SET UP POINTERS TO UNIT VALUES WITH ALIGNMENT.  Like net_setup_value_pointers
   except that values for each layer are given offsets from 'b' that are aligned
   (by elements, not bytes) at a multiple of 'align', which must be a power of 
   two.  The size of 'b' must be sufficient when this alignment is done (as the
   net_setup_value_count_aligned function will do). */

void net_setup_value_pointers_aligned
( net_values *v,	/* Structure to set up pointers in */
  net_value *b,		/* Block of 'value' values */
  net_arch *a,		/* Network architecture */
  int align,		/* Alignment, must be power of 2 */
  net_value *inputs,	/* Input values, 0 if part of value block */
  net_value *outputs	/* Output values, 0 if part of value block */
)
{
  int l;

  if (inputs)
  { v->i = inputs;
  }
  else
  { v->i = b;
    b += ALIGN(a->N_inputs,align);
  }

  for (l = 0; l<a->N_layers; l++)
  { int n = a->N_hidden[l];
    if (a->layer_type[l] == Normalize_type)
    { int c = a->N_channels[l];
      n += c>0 ? c : a->N_hidden[l]/(-c);
    } 
    v->h[l] = b;
    b += ALIGN(n,align);
  }

  if (outputs)
  { v->o = outputs;
  }
  else
  { v->o = b;
  }
}


/* LOCATE GROUP OF HYPERPARAMETERS.  Finds the offset and number of 
   hyperparameters in a group, plus whether it's a group of adjustments.  
   Groups are identified by integers from one on up.  Zero is returned 
   if the group index is too big. 

   Noise sigmas are not considered hyperparameters for this procedure. */

int net_setup_hyper_group
( net_arch *a,		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  int grp,		/* Index of group */
  int *offset,		/* Set to offset of group within block */
  int *number,		/* Set to number of items in group */
  int *adj		/* Set to whether this is a group of adjustments */
)
{ 
  int i, l, ls, nsqi;
  unsigned bits;

  *adj = 0;

  if (grp<1) return 0;

  i = 0;
  nsqi = 0;

  if (a->has_ti) 
  { *offset = i; 
    i += 1; 
    if (--grp==0) goto done;
  }

  for (l = 0; l<a->N_layers; l++)
  {
    for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (bits&1)
      { if (ls>=l-1) abort();
        *offset = i;
        i += 1 + a->N_hidden[ls];
        nsqi += 1;
        if (--grp==0) goto done;
      }
    }

    if (l>0)
    { if (a->has_hh[l-1]) 
      { *offset = i; 
        i += 1 + a->N_hidden[l-1]; 
        if (--grp==0) goto done;
      }
    }

    if (a->has_ih[l]) 
    { *offset = i; 
      i += 1 + not_omitted(flgs?flgs->omit:0,a->N_inputs,1<<(l+1)); 
      if (--grp==0) goto done;
    }

    if (a->has_bh[l]) 
    { *offset = i; 
      i += 1; 
      if (--grp==0) goto done;
    }

    if (a->has_ah[l]) 
    { *offset = i;
      i += a->N_hidden[l]; 
      if (--grp==0) { *adj = 1; goto done; }
    }

    if (a->has_th[l]) 
    { *offset = i; 
      i += 1; 
      if (--grp==0) goto done;
    }
  }

  for (l = a->N_layers-1; l>=0; l--)
  { if (a->has_ho[l]) 
    { *offset = i; 
      i += 1 + a->N_hidden[l]; 
      if (--grp==0) goto done;   
    }
  }

  if (a->has_io) 
  { *offset = i;   
    i += 1 + not_omitted(flgs?flgs->omit:0,a->N_inputs,1); 
    if (--grp==0) goto done;
  }

  if (a->has_bo) 
  { *offset = i; 
    i += 1; 
    if (--grp==0) goto done;  
  }

  if (a->has_ao) 
  { *offset = i;
    i += a->N_outputs; 
    if (--grp==0) { *adj = 1; goto done; }
  }

  return 0;

done:
  *number = i-*offset;
  return 1;
}


/* LOCATE GROUP OF PARAMETERS.  Finds the offset, number, and dimension of 
   source for a group of parameters, plus whether the group is configured.
   Groups are identified by integers from one on up.  Zero is returned if 
   the group index is too big, or refers to an adjustment group (having no 
   parameters in it). */

int net_setup_param_group
( net_arch *a,		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  int grp,		/* Index of group */
  int *offset,		/* Set to offset of group within block */
  int *number,		/* Set to number of items in group */
  int *source,		/* Set to number of source units associated with group,
                           or to zero if it's a one-dimensional group */
  int *configured	/* Set to 1 if group has configured connections */
)
{ 
  int i, l, ls, nsqi;
  unsigned bits;

  if (grp<1) return 0;

  i = 0;
  nsqi = 0;

  if (a->has_ti) 
  { *offset = i; 
    *source = 0;
    *configured = 0;
    i += a->N_inputs; 
    if (--grp==0) goto done;
  }

  for (l = 0; l<a->N_layers; l++)
  {
    for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (bits&1)
      { if (ls>=l-1) abort();
        *offset = i;
        *source = a->N_hidden[ls];
        *configured = a->nonseq_config[nsqi]!=0;
        i += *configured ? a->nonseq_config[l]->N_wts 
                         : *source * a->N_hidden[l]; 
        nsqi += 1;
        if (--grp==0) goto done;
      }
    }

    if (l>0)
    { if (a->has_hh[l-1]) 
      { *offset = i; 
        *source = a->N_hidden[l-1];
        *configured = a->hidden_config[l]!=0;
        i += *configured ? a->hidden_config[l]->N_wts 
                         : *source * a->N_hidden[l]; 
        if (--grp==0) goto done;
      }
    }

    if (a->has_ih[l]) 
    { *offset = i; 
      *source = not_omitted(flgs?flgs->omit:0,a->N_inputs,1<<(l+1));
      *configured = a->input_config[l]!=0;
      i += *configured ? a->input_config[l]->N_wts 
                       : *source * a->N_hidden[l]; 
      if (--grp==0) goto done;
    }

    if (a->has_bh[l]) 
    { *offset = i; 
      *source = 0;
      *configured = a->bias_config[l]!=0;
      i += *configured ? a->bias_config[l]->N_wts 
                       : a->N_hidden[l]; 
      if (--grp==0) goto done;
    }

    if (a->has_ah[l])
    { if (--grp==0) return 0;
    }

    if (a->has_th[l]) 
    { *offset = i; 
      *source = 0;
      *configured = 0;
      i += a->N_hidden[l]; 
      if (--grp==0) goto done;
    }
  }

  for (l = a->N_layers-1; l>=0; l--)
  { if (a->has_ho[l]) 
    { int k = 2*a->N_layers-1-l;
      *offset = i; 
      *source = a->N_hidden[l];
      *configured = a->hidden_config[k]!=0;
      i += *configured ? a->hidden_config[k]->N_wts
                       : a->N_hidden[l]*a->N_outputs; 
      if (--grp==0) goto done;
    }
  }

  if (a->has_io) 
  { *offset = i;    
    *source = not_omitted(flgs?flgs->omit:0,a->N_inputs,1);
    *configured = a->input_config[a->N_layers]!=0;
    i += *configured ? a->input_config[a->N_layers]->N_wts
                     : *source * a->N_outputs; 
    if (--grp==0) goto done;
  }

  if (a->has_bo) 
  { *offset = i; 
    *source = 0;
    *configured = a->bias_config[a->N_layers]!=0;
    i += *configured ? a->bias_config[a->N_layers]->N_wts 
                     : a->N_outputs; 
    if (--grp==0) goto done;
  }

  if (a->has_ao)
  { if (--grp==0) return 0;
  }

  return 0;

done:
  *number = i-*offset;
  return 1;
}
