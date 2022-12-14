/* NET-PRIOR.C - Routines dealing with priors for networks. */

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
#include "rand.h"


/* CONSTANT PI.  Defined here if not in <math.h>. */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/* LOCAL PROCEDURES. */

static void pick_unit_params (net_param *, net_sigma *, int, net_sigma *,
                              prior_spec, int, double, int);

static void pick_unit_params_config (net_param *, net_sigma *, int,
                                     prior_spec, int, double, int);

static void pick_weights (net_param *, net_sigma *, net_sigma *, 
                          int, int, net_sigma *, prior_spec, int, double, int);

static void pick_weights_config (net_param *, net_sigma *, 
                                 int, int, prior_spec, int, double, int);

static void compute_prior (net_param *, int, double *, net_param *, 
                           net_sigma, double, net_sigma *, int);

static void max_second (net_param *, int, net_sigma, net_sigma *, double);


/* GENERATE VALUES FOR HYPERPARAMETERS AND PARAMETERS.  Values for the 
   parameters and hyperparameters that exist in the network (according to 
   the has_xx fields of the architecture structure) are generated from the 
   prior distributions passed.  These parameter and hyperparameter values 
   are stored in the arrays pointed to from the structures passed as the
   first two arguments, which must have been properly set up by the caller. 
   
   Further arguments parallel the options for the net-gen program. */

void net_prior_generate
( net_params *w,	/* Arrays to store weights, biases, and offsets in */
  net_sigmas *s,	/* Arrays to store hyperparameters in */
  net_arch *a,		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  model_specification *m, /* Specification for data model */
  net_priors *p,	/* Network priors */
  int fix,		/* Choose "centre" rather than random value for sigma?*/
  double value,		/* Use specific value for centre */
  double out_value,	/* Use specific value for centre for output weights */
  int param_opt,	/* Option for param gen: 0=zero, 1=rand, 2=stdin */
  int out_param_opt	/* Option for gen of params going to output units */
)
{ 
  int l, ls, nsqi, i;
  unsigned bits;

  nsqi = 0;

  if (a->has_ti) 
  { pick_unit_params (w->ti, s->ti_cm, a->N_inputs, 0, p->ti, 
                      fix, value, param_opt);
  }

  if (a->has_ao)
  { for (i = 0; i<a->N_outputs; i++)
    { s->ao[i] = fix ? 1.0 : prior_pick_sigma(1.0,p->ao);
    }
  }

  for (l = 0; l<a->N_layers; l++)
  {
    if (a->has_ah[l])
    { for (i = 0; i<a->N_hidden[l]; i++)
      { s->ah[l][i] = fix ? 1.0 : prior_pick_sigma(1.0,p->ah[l]);
      }
    }

    for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (bits&1)
      { if (ls>=l-1) abort();
        if (a->nonseq_config[nsqi])
        { pick_weights_config (w->nsq[nsqi], s->nsq_cm[nsqi], 
                               a->N_hidden[ls], a->nonseq_config[nsqi]->N_wts,
                               p->nsq[nsqi], fix, value, param_opt);
        }
        else
        { pick_weights (w->nsq[nsqi], s->nsq_cm[nsqi], s->nsq[nsqi], 
                        a->N_hidden[ls], a->N_hidden[l], 
                        s->ah[l], p->nsq[nsqi], fix, value, param_opt);
        }
        nsqi += 1;
      }
    }

    if (l>0 && a->has_hh[l-1]) 
    { if (a->hidden_config[l])
      { pick_weights_config (w->hh[l-1], s->hh_cm[l-1], 
                             a->N_hidden[l-1], a->hidden_config[l]->N_wts,
                             p->hh[l-1], fix, value, param_opt);
      }
      else
      { pick_weights (w->hh[l-1], s->hh_cm[l-1], s->hh[l-1], 
                      a->N_hidden[l-1], a->N_hidden[l], 
                      s->ah[l], p->hh[l-1], fix, value, param_opt);
      }
    }

    if (a->has_ih[l]) 
    { if (a->input_config[l])
      { pick_weights_config (w->ih[l], s->ih_cm[l],
                             a->N_inputs, a->input_config[l]->N_wts,
                             p->ih[l], fix, value, param_opt);
      }
      else
      { pick_weights (w->ih[l], s->ih_cm[l], s->ih[l], 
          not_omitted(flgs?flgs->omit:0,a->N_inputs,1<<(l+1)), a->N_hidden[l], 
          s->ah[l], p->ih[l], fix, value, param_opt);
      }
    }

    if (a->has_bh[l])
    { if (a->bias_config[l])
      { pick_unit_params_config(w->bh[l], s->bh_cm[l], a->bias_config[l]->N_wts,
                                p->bh[l], fix, value, param_opt);
      }
      else
      { pick_unit_params (w->bh[l], s->bh_cm[l], a->N_hidden[l], s->ah[l],
                          p->bh[l], fix, value, param_opt);
      }
    }
    
    if (a->has_th[l])
    { pick_unit_params (w->th[l], s->th_cm[l], a->N_hidden[l], 
                        0, p->th[l], fix, value, param_opt);
    }
  }

  for (l = a->N_layers-1; l>=0; l--)
  { if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { pick_weights_config (w->ho[l], s->ho_cm[l], 
                             a->N_hidden[l], a->hidden_config[k]->N_wts,
                             p->ho[l], fix, out_value, out_param_opt);
      }
      else
      { pick_weights (w->ho[l], s->ho_cm[l], s->ho[l], 
                      a->N_hidden[l], a->N_outputs, 
                      s->ao, p->ho[l], fix, out_value, out_param_opt);
      }
    }
  }

  if (a->has_io) 
  { if (a->input_config[a->N_layers])
    { pick_weights_config (w->io, s->io_cm,
                           a->N_inputs, a->input_config[a->N_layers]->N_wts,
                           p->io, fix, out_value, out_param_opt);
    }
    else
    { pick_weights (w->io, s->io_cm, s->io,
                    not_omitted(flgs?flgs->omit:0,a->N_inputs,1), a->N_outputs, 
                    s->ao, p->io, fix, out_value, out_param_opt);
    }
  }

  if (a->has_bo)
  { if (a->bias_config[a->N_layers])
    { pick_unit_params_config 
                       (w->bo, s->bo_cm, a->bias_config[a->N_layers]->N_wts,
                        p->bo, fix, out_value, out_param_opt);
    }
    else
    { pick_unit_params (w->bo, s->bo_cm, a->N_outputs, 
                        s->ao, p->bo, fix, out_value, out_param_opt);
    }
  }

  if (m!=0 && m->type=='R') 
  {
    *s->noise_cm = fix ? m->noise.width
                         : prior_pick_sigma (m->noise.width, m->noise.alpha[0]);

    for (i = 0; i<a->N_outputs; i++)
    { s->noise[i] = fix ? *s->noise_cm
                           : prior_pick_sigma (*s->noise_cm, m->noise.alpha[1]);
    }
  }
}


/* GENERATE VALUES AND SIGMAS FOR PARAMETERS ASSOCIATED WITH UNITS. */

static void pick_unit_params
( net_param *wt,	/* Array to store parameters */
  net_sigma *sd_cm,	/* Place to store common sigma */
  int n,		/* Number of units */
  net_sigma *adj,	/* Adjustments for destination units, or zero */
  prior_spec pr,	/* Prior to use */
  int fix,		/* Choose "centre" rather than random value? */
  double value,		/* Use specific value for sigma? */
  int param_opt		/* How to generate parameters */
)
{ 
  net_sigma unit_sigma;
  int i;

  *sd_cm = fix ? (value==0 || pr.alpha[0]==0 ? pr.width : value)
               : prior_pick_sigma (pr.width, pr.alpha[0]);

  for (i = 0; i<n; i++)
  { 
    if (param_opt==1 || pr.one_or_two_point)
    { unit_sigma = prior_pick_sigma (*sd_cm, pr.alpha[1]);
      *wt = unit_sigma * (pr.one_or_two_point==1 ? 1 
                           : pr.one_or_two_point==2 ? 2*rand_int(2)-1 
                           : pr.one_or_two_point==3 ? -1
                           : rand_gaussian());
      if (adj!=0) 
      { *wt *= adj[i];
      }
    }
    else if (param_opt==0)
    { *wt = 0;
    }
    else if (param_opt==2)
    { double d; 
      if (scanf("%lf",&d)!=1)
      { fprintf (stderr, "Error reading parameter from standard input\n");
        exit(3);
      }
      *wt = d;
    }
    wt += 1;
  }
}


/* GENERATE VALUES AND SIGMAS FOR UNIT PARAMETERS, WITH SPECIFIED CONFIG. */

static void pick_unit_params_config
( net_param *wt,	/* Array to store parameters */
  net_sigma *sd_cm,	/* Place to store common sigma */
  int n,		/* Number of weights, possibly sparse and/or shared */
  prior_spec pr,	/* Prior to use */
  int fix,		/* Choose "centre" rather than random value? */
  double value,		/* Use specific value for sigma? */
  int param_opt		/* How to generate parameters */
)
{ 
  net_sigma unit_sigma;
  int i;

  *sd_cm = fix ? (value==0 || pr.alpha[0]==0 ? pr.width : value)
               : prior_pick_sigma (pr.width, pr.alpha[0]);

  for (i = 0; i<n; i++)
  { 
    if (param_opt==1 || pr.one_or_two_point)
    { unit_sigma = prior_pick_sigma (*sd_cm, pr.alpha[1]);
      *wt = unit_sigma * (pr.one_or_two_point==1 ? 1 
                           : pr.one_or_two_point==2 ? 2*rand_int(2)-1 
                           : pr.one_or_two_point==3 ? -1
                           : rand_gaussian());
    }
    else if (param_opt==0)
    { *wt = 0;
    }
    else if (param_opt==2)
    { double d; 
      if (scanf("%lf",&d)!=1)
      { fprintf (stderr, "Error reading parameter from standard input\n");
        exit(3);
      }
      *wt = d;
    }

    wt += 1;
  }
}


/* GENERATE VALUES AND SIGMAS FOR WEIGHTS OF ONE TYPE. */

static void pick_weights
( net_param *wt,	/* Array to store weights */
  net_sigma *sd_cm,	/* Place to store common sigma */
  net_sigma *sd,	/* Array to store sigmas for each unit */
  int n,		/* Number of source units */
  int nd,		/* Number of destination units */
  net_sigma *adj,	/* Adjustments for destination units, or zero */
  prior_spec pr,	/* Prior to use */
  int fix,		/* Choose "centre" rather than random value? */
  double value,		/* Use specific value for sigma? */
  int param_opt		/* How to generate parameters */
)
{ 
  net_sigma width, weight_sigma;
  int i, j;

  width = prior_width_scaled(&pr,n);

  *sd_cm = fix ? (value==0 || pr.alpha[0]==0 ? width : value)
               : prior_pick_sigma (width, pr.alpha[0]);

  for (i = 0; i<n; i++)
  { 
    sd[i] = fix ? (value==0 || pr.alpha[1]==0 ? *sd_cm : value)
                : prior_pick_sigma (*sd_cm, pr.alpha[1]);

    for (j = 0; j<nd; j++)
    { 
      if (param_opt==1 || pr.one_or_two_point)
      { weight_sigma = prior_pick_sigma (sd[i], pr.alpha[2]);
        *wt = weight_sigma * (pr.one_or_two_point==1 ? 1 
                               : pr.one_or_two_point==2 ? 2*rand_int(2)-1 
                               : pr.one_or_two_point==3 ? -1
                               : rand_gaussian());
        if (adj!=0) 
        { *wt *= adj[j];
        }
      }
      else if (param_opt==0)
      { *wt = 0;
      }
      else if (param_opt==2)
      { double d; 
        if (scanf("%lf",&d)!=1)
        { fprintf (stderr, "Error reading parameter from standard input\n");
          exit(3);
        }
        *wt = d;
      }
      wt += 1;
    }
  }
}


/* GENERATE VALUES AND SIGMAS FOR WEIGHTS WITH A SPECIFIED CONFIGURATION. */

static void pick_weights_config
( net_param *wt,	/* Array to store weights */
  net_sigma *sd_cm,	/* Place to store common sigma */
  int n,		/* Number of source units */
  int nw,		/* Number of weights, possibly sparse and/or shared */
  prior_spec pr,	/* Prior to use */
  int fix,		/* Choose "centre" rather than random value? */
  double value,		/* Use specific value for sigma? */
  int param_opt		/* How to generate parameters */
)
{ 
  net_sigma width, weight_sigma;
  int i, j;

  width = pr.width;

  *sd_cm = fix ? (value==0 || pr.alpha[0]==0 ? width : value)
               : prior_pick_sigma (width, pr.alpha[0]);

  for (j = 0; j<nw; j++)
  { 
    if (param_opt==1 || pr.one_or_two_point)
    { weight_sigma = prior_pick_sigma (*sd_cm, pr.alpha[2]);
      *wt = weight_sigma * (pr.one_or_two_point==1 ? 1 
                             : pr.one_or_two_point==2 ? 2*rand_int(2)-1 
                             : pr.one_or_two_point==3 ? -1
                             : rand_gaussian());
    }
    else if (param_opt==0)
    { *wt = 0;
    }
    else if (param_opt==2)
    { double d; 
      if (scanf("%lf",&d)!=1)
      { fprintf (stderr, "Error reading parameter from standard input\n");
        exit(3);
      }
      *wt = d;
    }

    wt += 1;
  }
}


/* COMPUTE LOG OF PRIOR PROBABILITY AND/OR ITS DERIVATIVES.  Compute the 
   log of the prior probability density of the network parameters passed
   with respect to the sigma values passed, and/or minus the derivative of 
   this with respect to the parameters.  If 'op' is 2, the probability may be
   computed ignoring factors that depend only on the sigma values; if 'op'
   is 1, factors that are constant or that depend only on the alphas may be 
   ignored; if 'op' is 0, the exact probability will be computed. */

void net_prior_prob 
( net_params *w,	/* Weights, biases, offsets, etc. */ 
  net_sigmas *s, 	/* Hyperpameters defining prior for parameters */
  double *lp,		/* Place to store log probability, zero if not wanted */
  net_params *dp,	/* Place to store minus log prob. derivative, or zero */
  net_arch *a, 		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  net_priors *p,	/* Network prior specifications */
  int op		/* Can we ignore some factors? */
)
{
  int l, ls, i, k, nsqi;
  unsigned bits;

  if (lp!=0)  
  { *lp = 0;
  }

  if (dp!=0)
  { for (i = 0; i<dp->total_params; i++)
    { dp->param_block[i] = 0;
    }
  }

  nsqi = 0;

  if (a->has_ti) compute_prior (w->ti, a->N_inputs, lp, dp ? dp->ti : 0, 
                                *s->ti_cm, p->ti.alpha[1], 0, op);

  for (l = 0; l<a->N_layers; l++)
  {
    for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (bits&1)
      { if (ls>=l-1) abort();
        if (a->nonseq_config[nsqi])
        { compute_prior (w->nsq[nsqi], a->nonseq_config[nsqi]->N_wts, lp,
                         dp ? dp->nsq[nsqi] : 0, 
                         *s->nsq_cm[nsqi], p->nsq[nsqi].alpha[2], 0, op);
        }
        else
        { for (i = 0; i<a->N_hidden[ls]; i++)
          { compute_prior (w->nsq[nsqi] + i*a->N_hidden[l], a->N_hidden[l], lp, 
                          dp ? dp->nsq[nsqi] + i*a->N_hidden[l] : 0, 
                          s->nsq[nsqi][i], p->nsq[nsqi].alpha[2], s->ah[l], op);
          }
        }
        nsqi += 1;
      }
    }

    if (l>0 && a->has_hh[l-1]) 
    { if (a->hidden_config[l])
      { compute_prior (w->hh[l-1], a->hidden_config[l]->N_wts, lp,
                       dp ? dp->hh[l-1] : 0, 
                       *s->hh_cm[l-1], p->hh[l-1].alpha[2], 0, op);
      }
      else
      { for (i = 0; i<a->N_hidden[l-1]; i++)
        { compute_prior (w->hh[l-1] + i*a->N_hidden[l], a->N_hidden[l], lp, 
                         dp ? dp->hh[l-1] + i*a->N_hidden[l] : 0, 
                         s->hh[l-1][i], p->hh[l-1].alpha[2], s->ah[l], op);
        }
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { compute_prior (w->ih[l], a->input_config[l]->N_wts, lp,
                       dp ? dp->ih[l] : 0,
                       *s->ih_cm[l], p->ih[l].alpha[2], 0, op);
      }
      else
      { k = 0;
        for (i = 0; i<a->N_inputs; i++)
        { if (flgs==0 || (flgs->omit[i]&(1<<(l+1)))==0)
          { compute_prior (w->ih[l] + k*a->N_hidden[l], a->N_hidden[l], lp, 
                           dp ? dp->ih[l] + k*a->N_hidden[l] : 0,
                           s->ih[l][k], p->ih[l].alpha[2], s->ah[l], op);
            k += 1;
          }
        }
      }
    }

    if (a->has_bh[l])
    { if (a->bias_config[l])
      { compute_prior (w->bh[l], a->bias_config[l]->N_wts, lp, 
                                 dp ? dp->bh[l] : 0, *s->bh_cm[l], 
                                 p->bh[l].alpha[1], 0, op); 
      }
      else
      { compute_prior (w->bh[l], a->N_hidden[l], lp, 
                                 dp ? dp->bh[l] : 0, *s->bh_cm[l], 
                                 p->bh[l].alpha[1], s->ah[l], op); 
      }
    }
    
    if (a->has_th[l]) compute_prior (w->th[l], a->N_hidden[l], lp, 
                                     dp ? dp->th[l] : 0, *s->th_cm[l], 
                                     p->th[l].alpha[1], 0, op);

    if (a->has_ho[l]) 
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { compute_prior (w->ho[l], a->hidden_config[k]->N_wts, lp,
                       dp ? dp->ho[l] : 0, 
                       *s->ho_cm[l], p->ho[l].alpha[2], 0, op);
      }
      else
      { for (i = 0; i<a->N_hidden[l]; i++)
        { compute_prior (w->ho[l] + i*a->N_outputs, a->N_outputs, lp, 
                         dp ? dp->ho[l] + i*a->N_outputs : 0,
                         s->ho[l][i], p->ho[l].alpha[2], s->ao, op); 
        }
      }
    }
  }

  if (a->has_io) 
  { if (a->input_config[a->N_layers])
    { compute_prior (w->io, a->input_config[a->N_layers]->N_wts, lp,
                     dp ? dp->io : 0,
                     *s->io_cm, p->io.alpha[2], 0, op);
    }
    else
    { k = 0;
      for (i = 0; i<a->N_inputs; i++)
      { if (flgs==0 || (flgs->omit[i]&1)==0)
        { compute_prior (w->io + k*a->N_outputs, a->N_outputs, lp, 
                         dp ? dp->io + k*a->N_outputs : 0,
                         s->io[k], p->io.alpha[2], s->ao, op); 
          k += 1;
        }
      }
    }
  }

  if (a->has_bo)
  { if (a->bias_config[a->N_layers])
    { compute_prior (w->bo, a->bias_config[a->N_layers]->N_wts, lp, 
                            dp ? dp->bo : 0, *s->bo_cm, 
                            p->bo.alpha[1], 0, op); 
    }
    else
    { compute_prior (w->bo, a->N_outputs, lp, dp ? dp->bo : 0,
                            *s->bo_cm, p->bo.alpha[1], s->ao, op); 
    }
  }

  return;
}


/* COMPUTE PRIOR FOR ONE SET OF PARAMETERS.  Computes the prior probability
   for a set of parameter values that are associated with given sigma and
   alpha (and adds it to an accumulator), and the derivatives of this 
   probability with respect to the parameters.  If the alpha is infinite 
   (represented as zero), each parameter has a Gaussian distribution.  If it 
   is finite, each parameter has a t-distribution with that many degress of 
   freedom. */

static void compute_prior
( net_param *wt,	/* Set of parameters */
  int n,		/* Number of parameters in set */
  double *lp,		/* Place to add log probability, zero if not wanted */
  net_param *dp,	/* Place to store minus log prob. derivatives, or zero*/
  net_sigma sigma,	/* Sigma for this set of parameters */
  double alpha,		/* Alpha for going from common sigma to that for one */
  net_sigma *adj,	/* Adjustments to sigma for each destination, or zero */
  int op		/* Can we ignore some factors? */
)
{
  extern double lgamma(double);

  double v, t;
  int i;

  if (alpha==0) /* Gaussian distribution */
  {
    for (i = 0; i<n; i++)
    {
      v = adj!=0 ? sigma*adj[i] : sigma;
      v *= v;

      if (lp!=0)
      { *lp -= (wt[i]*wt[i]) / (2*v);
        if (op<2 && adj!=0)
        { *lp -= log(adj[i]);
        }
      }

      if (dp!=0)
      { dp[i] = wt[i] / v;
      }
    }

    if (op<2)
    { *lp -= n * log(sigma);
    }

    if (op==0) 
    { *lp -= (n/2.0) * log(2*M_PI);
    }
  }
  else /* Student t-distribution */
  {
    for (i = 0; i<n; i++)
    { 
      v = adj!=0 ? sigma*adj[i] : sigma;
      v *= v;
      t = 1 + (wt[i]*wt[i]) / (v*alpha);

      if (lp!=0) 
      { *lp -= ((alpha+1)/2) * log (t);
        if (op<2 && adj!=0) 
        { *lp -= log(adj[i]);
        }
      }

      if (dp!=0) 
      { dp[i] = wt[i] * ((alpha+1)/(alpha*v)) / t;
      }
    }

    if (lp!=0)
    {
      if (op<2)
      { *lp -= n * log(sigma);
      }

      if (op==0)
      { *lp += n * (lgamma((alpha+1)/2)-lgamma(alpha/2)-0.5*log(M_PI*alpha));
      }
    }
  }

}


/* COMPUTE MAXIMUM LOG PRIOR PROBABILITY SECOND DERIVATIVES.  Computes for
   each network parameter the maximum value of the second derivative of minus
   the log prior probability with respect to that parameter.  The maximum is
   over the possible values of other parameters, and is based on the current 
   values of the hyperparameters. */

void net_prior_max_second
( net_params *d,	/* Place to store maximum second derivatives */
  net_sigmas *s,	/* Hyperparameters */
  net_arch *a,		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  net_priors *p		/* Network priors */
)
{
  int i, k, l, ls, nsqi;
  unsigned bits;

  nsqi = 0;

  if (a->has_ti) 
  { max_second (d->ti, a->N_inputs, *s->ti_cm, 0, p->ti.alpha[1]);
  }
 
  for (l = 0; l<a->N_layers; l++)
  {
    if (a->has_bh[l])
    { if (a->bias_config[l])
      { max_second (d->bh[l], a->bias_config[l]->N_wts,
                    *s->bh_cm[l], 0, p->bh[l].alpha[1]);
      }
      else
      { max_second (d->bh[l], a->N_hidden[l], 
                    *s->bh_cm[l], s->ah[l], p->bh[l].alpha[1]);
      }
    }

    if (a->has_th[l])
    { max_second (d->th[l], a->N_hidden[l], 
                  *s->th_cm[l], 0, p->th[l].alpha[1]);
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { max_second (d->ih[l], a->input_config[l]->N_wts, *s->ih_cm[l],
                    0, p->ih[l].alpha[2]);
      }
      else
      { k = 0;
        for (i = 0; i<a->N_inputs; i++)
        { if (flgs==0 || (flgs->omit[i]&(1<<(l+1)))==0)
          { max_second (d->ih[l] + k*a->N_hidden[l], a->N_hidden[l], 
                        s->ih[l][k], s->ah[l], p->ih[l].alpha[2]);
            k += 1;
          }
        }
      }
    }

    for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (bits&1)
      { if (ls>=l-1) abort();
        if (a->nonseq_config[nsqi])
        { max_second (d->nsq[nsqi], a->nonseq_config[nsqi]->N_wts, 
                      *s->nsq_cm[nsqi], 0, p->nsq[nsqi].alpha[2]);
        }
        else
        { for (i = 0; i<a->N_hidden[ls]; i++)
          { max_second (d->nsq[nsqi] + i*a->N_hidden[l], a->N_hidden[l],
                        s->nsq[nsqi][i], s->ah[l], p->nsq[nsqi].alpha[2]);
          }
        }
        nsqi += 1;
      }
    }

    if (l>0 && a->has_hh[l-1]) 
    { if (a->hidden_config[l])
      { max_second (d->hh[l-1], a->hidden_config[l]->N_wts, *s->hh_cm[l-1],
                    0, p->hh[l-1].alpha[2]);
      }
      else
      { for (i = 0; i<a->N_hidden[l-1]; i++)
        { max_second (d->hh[l-1] + i*a->N_hidden[l], a->N_hidden[l], 
                      s->hh[l-1][i], s->ah[l], p->hh[l-1].alpha[2]);
        }
      }
    }

    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { max_second (d->ho[l], a->hidden_config[k]->N_wts, *s->ho_cm[l],
                    0, p->ho[l].alpha[2]);
      }
      else
      { for (i = 0; i<a->N_hidden[l]; i++)
        { max_second (d->ho[l] + i*a->N_outputs, a->N_outputs, 
                      s->ho[l][i], s->ao, p->ho[l].alpha[2]);
        }
      }
    }
  }

  if (a->has_io)
  { if (a->input_config[l])
    { max_second (d->io, a->input_config[a->N_layers]->N_wts, *s->io_cm,
                  0, p->io.alpha[2]);
    }
    else
    { k = 0; 
      for (i = 0; i<a->N_inputs; i++)
      { if (flgs==0 || (flgs->omit[i]&1)==0)
        { max_second (d->io + k*a->N_outputs, a->N_outputs,
                      s->io[k], s->ao, p->io.alpha[2]);
          k += 1;
        }
      }
    }
  }

  if (a->has_bo)
  { if (a->bias_config[a->N_layers])
    { max_second (d->bo, a->bias_config[a->N_layers]->N_wts, 
                  *s->bo_cm, 0, p->bo.alpha[1]);
    }
    else
    { max_second (d->bo, a->N_outputs, *s->bo_cm, s->ao, p->bo.alpha[1]);
    }
  }
}


/* COMPUTE MAXIMUM SECOND DERIVATIVE FOR GROUP OF PARAMETERS. */

static void max_second
( net_param *d,		/* Place to store maximum second derivatives */
  int n,		/* Number of parameters in group */
  net_sigma s,		/* Width of prior */
  net_sigma *adj,	/* Pointer to adjustments for width, or zero */
  double alpha		/* Alpha of prior */
)
{ 
  double v;
  int i;

  v = alpha==0 ? 1 / (s*s) : (alpha+1) / (alpha*s*s);

  if (adj==0)
  { for (i = 0; i<n; i++) 
    { d[i] = v;
    }
  }
  else 
  { for (i = 0; i<n; i++) 
    { d[i] = v / (adj[i] * adj[i]);
    }
  }
}
