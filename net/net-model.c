/* NET-MODEL.C - Module dealing with the interpretation of network outputs. */

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

#ifndef GPU_SRC_INCLUDE  /* Not included in another source file */

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

#include "intrinsics-use.h"
#define __SLEEF_REMPITAB__  /* Need to leave undefined in exactly one file */
#include "sleef-use.h"

#endif


/* CONSTANTS INVOLVING PI. */

#ifndef M_PI
#define M_PI 3.14159265358979323846	/* Define pi, if not defined already */
#endif

#define Log2pi  1.83787706640934548356	/* Log(2*M_PI) */


/* CHECK THAT A DATA MODEL IS PRESENT. */

void STATIC_IF_INCLUDED net_model_check (model_specification const*m)
{
  if (m->type==0)
  { fprintf(stderr,"Network has no data model defined\n");
    exit(1);
  }
}


/* COMPUTE LOG PROBABILITY OF TARGETS AND/OR ITS DERIVATIVES.  Computes the 
   log of the probability (or probability density) of the observed target 
   values given the observed inputs, as defined by the current network outputs 
   for those inputs, and the derivative of minus the log probability with 
   respect to the network outputs. 
 
   A data model must be specified to use this procedure.  For survival models
   with piecewise constant hazard, this procedure should be called several
   times, once for each piece, with a target value that is the survival
   time after the start of the piece (negative if censored).  The log
   probabilities from these calls should then be added together (though the 
   derivatives will generally be required in separate form).

   If the 'op' parameter is 2, the probability may be computed ignoring factors
   that depend only on the overall and per-unit noise hyperparameters, and this
   freedom will be used to ensure that values greater than zero will not be
   returned, except for survival models; if 'op' is 1, factors that are 
   constant or depend only on the alphas may be ignored; if 'op' is 0, the 
   exact probability will be computed.

   By passing zero pointers, computation of either the log probability or of
   its derivatives may be suppressed, with a possible saving in time. */

HOSTDEV STATIC_IF_INCLUDED void net_model_prob
( net_values const*v,	/* Values for units in network */
  net_value const*t,	/* Target values, fudged for piecewise const hazard */
  double *restrict pr,	/* Place to store log probability, zero if not wanted */
  net_values *restrict dp,/* Place to store neg log probability derivs, or 0 */
  net_arch const*a,	/* Network architecture */
  model_specification const*m, /* Data model */
  model_survival const*sv,/* Type of hazard function for survival model, or 0 */
  net_sigma const*noise,/* Noise sigmas, or null */
  int op		/* Can we ignore some factors? */
)
{
  int N_outputs = a->N_outputs;
  int i;

  switch (m->type)
  {
    case 'B':  /* Binary data values */
    { 
      if (pr) *pr = 0;

      for (i = 0; i<N_outputs; i++)
      { 
        if (isnan(t[i]))  /* target not observed */
        { if (dp) dp->o[i] = 0;
          continue;
        }

        net_value oi = v->o[i];
        net_value sign_oi = prec_copysign((net_value)1.0,oi);
        net_value abs_oi = prec_fabs(oi);

        /* Note: The exp computation below never overflows. */

        net_value ep1 = prec_exp (-abs_oi) + 1;

        if (pr)  /* find log probability */
        { *pr += (t[i] - 0.5*(sign_oi+1)) * oi - prec_log(ep1);;
        }

        if (dp)  /* find derivative of log probability */
        { dp->o[i] = sign_oi/ep1 + 0.5*(1.0-sign_oi) - t[i];
        }
      }

      break;
    }

    case 'C':  /* Single class with multiple possible values */
    {
      net_value m, e, s;

      if (isnan(*t))  /* target not observed */
      { if (pr) *pr = 0;
        if (dp) 
        { for (i = 0; i<N_outputs; i++)
          { dp->o[i] = 0;
          }
        }
        break;
      }

      m = v->o[0];
      for (i = 1; i<N_outputs; i++)
      { if (v->o[i]>m) m = v->o[i];
      }

      s = 0;
      for (i = 0; i<N_outputs; i++)
      { e = prec_exp (v->o[i] - m);
        if (dp) dp->o[i] = e;
        s += e;
      }

      if (pr) 
      { *pr = v->o[(int)*t] - m - prec_log(s);
      }

      if (dp)
      { s = 1/s;
        for (i = 0; i<N_outputs; i++)
        { dp->o[i] *= s;
        }
        dp->o[(int)*t] -= 1;
      }

      break;
    }
  
    case 'R':  /* Real-valued target */
    { 
      double alpha = m->noise.alpha[2];
      double nconst;

      if (alpha==0) /* Gaussian distribution for noise */
      { 
        if (pr)
        { nconst = op>0 ? 0 : - 0.5 * Log2pi;
          *pr = 0;
        }

        for (i = 0; i<N_outputs; i++)
        { if (isnan(t[i]))  /* target not observed */
          { if (dp) dp->o[i] = 0;
            continue;
          }
          net_sigma rn = 1 / noise[i];
          net_value d = (v->o[i] - t[i]) * rn;
          if (d<-1e10) d = -1e10;
          if (d>+1e10) d = +1e10;
          if (pr) 
          { double p = nconst - 0.5 * (d*d);
            if (op<2) p += prec_log(rn);
            *pr += p;
          }
          if (dp)
          { dp->o[i] = d * rn;
          }
        }
      }

      else /* Student t distribution for noise */
      {
        if (pr)
        { nconst = op>0 ? 0 : m->alpha_cnst;
          *pr = 0;
        }

        for (i = 0; i<N_outputs; i++)
        { if (isnan(t[i]))  /* target not observed */
          { if (dp) dp->o[i] = 0;
            continue;
          }
          net_sigma rn = 1 / noise[i];
          net_value d = (v->o[i] - t[i]) * rn;
          if (d<-1e10) d = -1e10;
          if (d>+1e10) d = +1e10;
          net_value x = 1 + d*d/(net_value)alpha;
          if (pr) 
          { double p = nconst - ((alpha+1)/2) * prec_log(x);
            if (op<2) p += log(rn);
            *pr += p;
          }
          if (dp)
          { dp->o[i] = (net_value)((alpha+1)/alpha) * (d*rn) / x;
          }
        }
      }

      break;
    }

    case 'V':  /* Survival model */
    { 
      int censored;
      double m, ot, ho;

      if (isnan(t[0]))  /* target not observed */
      { if (pr) *pr = 0;
        if (dp) dp->o[0] = 0;
        break;
      }

      if (t[0]<0)
      { censored = 1;
        ot = -t[0];
      }
      else
      { censored = 0;
        ot = t[0];
      }

      m = v->o[0]+log(ot);

      if (m>200.0)
      { ho = prec_exp(200.0);
        if (pr) *pr = - ho;
        if (dp) dp->o[0] = 0;
      }
      else
      { 
        ho = prec_exp(m);

        if (pr) *pr = - ho;
        if (dp) dp->o[0] = ho;

        if (!censored)
        { if (pr) *pr += v->o[0];
          if (dp) dp->o[0] -= 1;
        }
      }

      break;
    }

    default:  /* No data model, or unknown type of model */
    { abort();
    }
  }
}


/* COMPUTE MAXIMUM LOG LIKELIHOOD SECOND DERIVATIVES.  Computes the maximum
   values of the second derivatives of minus the log probability of the targets
   in a (single) training case with respect to the outputs of the net, for the 
   current values of the hyperparameters.  The maximum is with respect to 
   possible values of the outputs of the net (the real outputs must not be 
   looked at, or validity of the Markov chain methods would be undermined), 
   and of the true targets (which could be looked at, but which aren't here). 
   In some cases, one must make do with approximations.
 
   A data model must be specified to use this procedure. */

void STATIC_IF_INCLUDED net_model_max_second
( net_value *msd,	/* Place to store maximum second derivatives */
  net_arch *a,		/* Network architecture */
  model_specification *m, /* Data model */
  model_survival *sv,	/* Type of hazard function for survival model, or null*/
  net_sigma *noise	/* Noise sigmas, or null */
)
{
  int N_outputs = a->N_outputs;
  double alpha;
  int i;

  switch (m->type)
  {
    case 'B':  /* Binary data values */
    {
      for (i = 0; i<N_outputs; i++)
      { msd[i] = 0.25;
      }

      break;
    }

    case 'C':  /* Single class with multiple possible values */
    {
      for (i = 0; i<N_outputs; i++)
      { msd[i] = 0.25;
      }

      break;
    }

    case 'R':  /* Real-valued target */
    {
      alpha = m->noise.alpha[2];

      for (i = 0; i<N_outputs; i++)
      { msd[i] = alpha==0 ? 1 / (noise[i] * noise[i])
                          : (alpha+1) / (alpha * noise[i] * noise[i]);
      }

      break;
    }

    case 'V':  /* Survival data */
    {
      msd[0] = 1; /* Rather crude, but not completely arbitrary */

      break;
    }

    case 0:
    { fprintf(stderr,"Network has no data model defined\n");
      exit(1);
    }

    default:
    { fprintf(stderr,"Unknown data model: %c\n",m->type);
      exit(1);
    }
  }
}


/* MAKE GUESSES AT TARGET VALUES.  The guesses are based on the outputs
   of the network (which must already have been computed), except for
   piecewise-constant survival models (see below).  The noise sigma/prior
   may also be relevant.  Guesses can be "mean" values, "median" values,
   or values randomly drawn from the target distribution, depending on 
   the setting of the 'type' argument.  Median guesses are not allowed
   for binary and class targets.

   For binary targets, the "mean" gueses is the probability of the target 
   being one.  The random guess is a 0 or 1, with 1 having the specified 
   probability.

   For class targets, the "mean" guess is a vector of probabilities for the 
   various classes.  (Note that this requires an array for the target, not 
   a single value.)  The random guess is one of the classes (numbered from
   zero on up), chosen according to this distribution.  (In this case, only
   a single target is produced.)
 
   For real-valued data, the "mean" and "median" guesses are both just the 
   network output.  The random guess is chosen from a Gaussian distribution 
   with this mean and with a standard deviation determined from the noise 
   hyperparameters.  (When each case has a different noise level, the noise 
   is effectively from a t distribution.) 

   For a survival model, the "mean" guess is the mean of the distribution
   of survival times, the "median" is the median of this distribution, and
   the random guess is a value drawn from this distribution.  For 
   piecewise-constant hazards, the network output at the start of this
   procedure is ignored.  The output is instead evaluated several times
   by this procedure, with different settings for the first input (time).

   If no data model is specified, a real-valued data model with no noise
   is assumed. */

void STATIC_IF_INCLUDED net_model_guess
( net_values *v,	/* Values for units in network */
  net_value *t,		/* Place to store guesses at targets */
  net_arch *a,		/* Network architecture */
  net_precomputed *pre,  /* Precomputed aspects of architecture */
  net_flags *flgs,	/* Network flags, null if none */
  model_specification *m, /* Data model */
  model_survival *sv,	/* Type of hazard function for survival model, or null*/
  net_params *params,	/* Network parameters (used only for pw-const-hazard) */
  net_sigma *noise,	/* Noise sigmas, or null */
  int type		/* 0=mean, 1=random, 2=median */
)
{
  int N_outputs = a->N_outputs;
  double z, pr, r, noi, alpha;
  int i;

  switch (m->type)
  {
    case 'B':  /* Binary data values */
    { 
      if (type==2) abort();

      for (i = 0; i<N_outputs; i++)
      { pr = 1 / (1+exp(-v->o[i]));
        t[i] = type==1 ? rand_uniform()<pr : pr;
      }

      break;
    }

    case 'C':  /* Single class with multiple possible values */
    {
      if (type==2) abort();

      z = v->o[0];

      for (i = 1; i<N_outputs; i++)
      { z = addlogs(z,v->o[i]);
      }

      if (type==1)
      { r = rand_uniform();
        *t = 0; /* just in case */
      }

      for (i = 0; i<N_outputs; i++)
      { pr = exp (v->o[i] - z);
        if (type==1)
        { r -= pr;
          if (r<0) 
          { *t = i;
            break;
          }
        }
        else
        { t[i] = pr;
        }
      }

      break;
    }
  
    case 'R': case 0:  /* Real-valued target */
    { 
      for (i = 0; i<N_outputs; i++)
      { t[i] = v->o[i];
        if (type==1 && m->type!=0)
        { noi = noise[i];
          alpha = m->noise.alpha[2];
          if (alpha!=0)
          { noi /= sqrt (rand_gamma(alpha/2) / (alpha/2));
          }
          t[i] += noi * rand_gaussian();
        }
      }

      break;
    }
  
    case 'V': 
    {
      double U;

      U = type==1 ? rand_uniopen() : 0.5;

      switch (sv->hazard_type)
      {
        case 'C':  /* Constant hazard */
        {
          double h;
          h = exp(v->o[0]);
          t[0] = type==0 ? 1/h : -log(U)/h;
          break;
        }

        case 'P':  /* Piecewise constant hazard */ 
        {
          double t0, t1, h, pr, T;
          int w;

          t0 = 0;
          t1 = sv->time[0];
          v->i[0] = sv->log_time ? log(t1) : t1;
 
          t[0] = 0;
          pr = 1;
          w = 0;
 
          for (;;)
          {
            net_func (v, a, pre, flgs, params, 1);
            h = exp(v->o[0]);
            
            if (type==0)
            {
              t[0] += pr * (t0 + 1/h);
              if (t1==-1) 
              { break;
              }
              pr *= exp(-h*(t1-t0));
              t[0] -= pr * (t1 + 1/h);
            }
            else
            {
              t[0] = t0 - log(U)/h;
              if (t1==-1 || t[0]<=t1) 
              { break;       
              }
              T = exp(-h*(t1-t0));
              U /= T;
            }

            t0 = t1;
            w += 1;
          
            if (sv->time[w]==0) 
            { t1 = -1;
              v->i[0] = sv->log_time ? log(t0) : t0;
            }
            else
            { t1 = sv->time[w];
              v->i[0] = sv->log_time ? (log(t0)+log(t1))/2 : (t0+t1)/2;
            }
          }

          break;
        }

        default: 
        { fprintf(stderr,"Unknown hazard type: %c\n",sv->hazard_type);
          exit(1);
        }
      }

      break;
    }

    default:
    { fprintf(stderr,"Unknown data model: %c\n",m->type);
      exit(1);
    }
  }
}


/* VERSION OF NET_MODEL_PROB USING MULTIPLE GPU THREADS.  Must not be used
   for survival models (type 'V').  

   The probability pointed to by 'pr' is updated by the thread with
   th==0.

   The derivative at index i in 'dp' is computed by the thread with 'th'
   equal to i mod THREADS_PER_CASE.

   A no syncthreads call is made after these computations are done, so
   if threads are not synchronized later, only the threads that
   computed a value can reliably access it.

   Assumes that on entry the i'th output is accessible to thread 'th'
   if i mod THREADS_PER_CASE is 'th'. 

   The memory pointed to by const_scratch is used temporarily for the
   class model, at offset 'scroff', with space there for twice the 
   number of outputs. 
*/

#if __CUDACC__

#define A const_arch
#define M const_model
#define NOISE const_noise

__device__ STATIC_IF_INCLUDED void net_model_prob_gpu
( int th,		/* Thread index, if negative, just sync */
  net_values const*v,	/* Values for units in network */
  net_value const*t,	/* Target values */
  double *restrict pr,	/* Place to store log probability, zero if not wanted */
  net_values *restrict dp,/* Place to store neg log probability derivs, or 0 */
  int scroff,           /* Scratch memory offset, for twice number of outputs */
  int op		/* Can we ignore some factors? */
)
{
  int N_outputs = A.N_outputs;
  int i;

  switch (M.type)
  {
    case 'B':  /* Binary data values */
    { 
      if (th<0) break;

      for (i = th; i<N_outputs; i+=NTH)
      { 
        if (isnan(t[i]))  /* target not observed */
        { if (dp) dp->o[i] = 0;
          if (pr) const_scratch[scroff+i] = 0;
          continue;
        }

        net_value oi = v->o[i];
        net_value sign_oi = prec_copysign((net_value)1.0,oi);
        net_value abs_oi = prec_fabs(oi);

        /* Note: The exp computation below never overflows. */

        net_value ep1 = prec_exp (-abs_oi) + 1;

        if (dp)  /* find derivative of log probability */
        { dp->o[i] = sign_oi/ep1 + 0.5*(1.0-sign_oi) - t[i];
        }

        if (pr)  /* store log probability in scratch */
        { const_scratch[scroff+i] = 
            (t[i] - 0.5*(sign_oi+1)) * oi - prec_log(ep1);
        }
      }

      break;
    }

    case 'C':  /* Single class with multiple possible values */
    {
      if (isnan(*t))  /* target not observed */
      { if (th<0)
        { return;
        }
        if (dp) 
        { for (i = th; i<N_outputs; i+=NTH)
          { dp->o[i] = 0;
          }
        }
        if (pr && th==0) *pr = 0;
        return;
      }

      if (th<0) goto sync_c;

      /* Compute things in scratch memory.  Either value at i is
         exponential of the output, and that at i+N_outputs is zero,
         or value at i is zero and that at i+N_outputs is the
         unchanged output.  Avoids the exponential computation at this
         point if it might overflow or underflow, while doing as many
         exponentials in parallel as possible (typically all of them). */

      net_value possible_exp_overflow;
      possible_exp_overflow = 77.2;  /* Maximum value so exp of +- this won't
                                        overflow with FP32, and such values
                                        still won't overflow in a summation 
                                        of up to 100,000 terms */

      for (i = th; i<N_outputs; i+=NTH)
      { if (v->o[i]>possible_exp_overflow || v->o[i]<-possible_exp_overflow) 
        { const_scratch[scroff+i] = 0;
          const_scratch[scroff+i+N_outputs] = v->o[i];
        }
        else
        { const_scratch[scroff+i] = prec_exp (v->o[i]);
          const_scratch[scroff+i+N_outputs] = 0;
        }
      }

    sync_c:
      if (NTH>1)
      { __syncthreads();
      }

      if (th<0)
      { return;
      }

      /* Find sum of exponentials (redundantly in all threads being used). */

      net_value m;

      m = const_scratch[scroff+N_outputs];
      for (i = 1; i<N_outputs; i++)
      { if (const_scratch[scroff+i+N_outputs]>m)
        { m = const_scratch[scroff+i+N_outputs];
        }
      }

      net_value s = 0;

      if (m==0)  /* no big ones, not all small ones */
      {  for (i = 0; i<N_outputs; i++)
         { s += const_scratch[scroff+i];
         }
      }
      else  /* some big ones, or all small ones - can ignore normal ones */
      { for (i = 0; i<N_outputs; i++)
         { if (const_scratch[scroff+i+N_outputs]!=0)
           { s += prec_exp (const_scratch[scroff+i+N_outputs] - m);
           }
         }
      }

      /* Compute log probability in thread 0. */

      if (pr && th==0) 
      { *pr = v->o[(int)*t] - m - prec_log(s);
      }

      /* Compute derivatives of log probability using all threads in use. */

      if (dp)
      { s = 1/s;
        int w = *t;
        for (i = th; i<N_outputs; i+=NTH)
        { if (m==0) 
          { dp->o[i] = s * const_scratch[scroff+i];
          }
          else if (const_scratch[scroff+i+N_outputs]==0)
          { dp->o[i] = 0;
          }
          else
          { dp->o[i] = s * prec_exp (const_scratch[scroff+i+N_outputs] - m);
          }
          if (i==w) dp->o[i] -= 1;
        }
      }

      return;
    }

    case 'R':  /* Real-valued target */
    { 
      if (th<0) break;

      double alpha = M.noise.alpha[2];

      if (alpha==0) /* Gaussian distribution for noise */
      { 
        double nconst = op>0 ? 0 : - 0.5 * Log2pi;

        for (i = th; i<N_outputs; i+=NTH)
        { if (isnan(t[i]))  /* target not observed */
          { if (dp) dp->o[i] = 0;
            if (pr) const_scratch[scroff+i] = 0;
            continue;
          }
          net_sigma rn = 1 / NOISE[i];
          net_value d = (v->o[i] - t[i]) * rn;
          if (d<-1e10) d = -1e10;
          if (d>+1e10) d = +1e10;
          if (pr) 
          { const_scratch[scroff+i] = nconst - 0.5 * (d*d);
            if (op<2) const_scratch[scroff+i] += prec_log(rn);
          }
          if (dp)
          { dp->o[i] = d * rn;
          }
        }
      }

      else /* Student t distribution for noise */
      {
        double nconst = op>0 ? 0 : M.alpha_cnst;

        for (i = th; i<N_outputs; i+=NTH)
        { if (isnan(t[i]))  /* target not observed */
          { if (dp) dp->o[i] = 0;
            if (pr) const_scratch[scroff+i] = 0;
            continue;
          }
          net_sigma rn = 1 / NOISE[i];
          net_value d = (v->o[i] - t[i]) * rn;
          if (d<-1e10) d = -1e10;
          if (d>+1e10) d = +1e10;
          net_value x = 1 + d*d/(net_value)alpha;
          if (pr) 
          { const_scratch[scroff+i] = nconst - ((alpha+1)/2) * prec_log(x);
            if (op<2) const_scratch[scroff+i] += log(rn);
          }
          if (dp)
          { dp->o[i] = (net_value)((alpha+1)/alpha) * (d*rn) / x;
          }
        }
      }

      break;
    }
  }

  /* Compute total log probability from scratch values (except class models). */

  if (pr)
  { 
    if (NTH>1 && N_outputs>1) 
    { __syncthreads();
    }

    if (th==0)
    { double p = 0;
      for (i = 0; i<N_outputs; i++)
      { p += const_scratch[scroff+i];
      }
      *pr = p;
    }
  }
}

#undef A
#undef M
#undef NOISE

#endif
