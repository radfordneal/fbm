/* NET-BACK.C - Routine for backpropagating the "error" through the network. */

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
#include "prior.h"
#include "data.h"
#include "model.h"
#include "net.h"

#include "intrinsics-use.h"
#include "sleef-use.h"


/* This module finds the derivative of the "error" for a particular case
   with respect to the values of the hidden and input units, and the with
   respect to the summed input for the hidden units, given the derivative 
   with respect to the output units.  There are facilities for calculating 
   these derivatives only back to a certain layer.
*/

#define sqrt_2 1.4142135623730950488



/* ---------------------------- net_back ----------------------------------- */

HOSTDEV static void sum_derivatives (net_value const*, int, net_value *restrict,
                                     int, net_param const*, 
                                     unsigned short const*, int);
HOSTDEV static void sum_derivatives_config(net_value const*, 
                                           net_value *restrict,
                                           net_param const*, net_config const*);


/* BACKPROPAGATE ERROR DERIVATIVES.  The first argument must contain the 
   values of all the units in the network.  The second must contain the
   derivatives of the "error" with respect to the output units (in g->o).
   These derivatives are backpropagated to give the derivatives of the
   error with respect to the other units in the network, and with respect
   to the summed input into the hidden units.  This is done back to hidden 
   layer 'start', or back all the way to the inputs if 'start' is -1. */

HOSTDEV void net_back
( net_values const*v,	/* Values for units in network */
  net_values *restrict d,/* Place to get output derivatives, and store others */
  int start,		/* Earliest layer to find derivatives for */
  net_arch const*a,	/* Network architecture */
  net_flags const*flgs,	/* Network flags, null if none */
  net_params const*w	/* Network parameters */
)
{
  int l, ls, ld, i;

  /* Find locations of non-sequential configurations. */

  char ns[Max_nonseq][Max_nonseq];
  unsigned bits;
  int nsqi = 0;
  for (l = 0; l<a->N_layers; l++)
  { for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (ls>=l-1) abort();
      ns[ls][l] = nsqi;
      nsqi += 1;
    }
  }

  /* Backpropagate through hidden layers. */

  for (l = a->N_layers-1; l>=0 && l>=start; l--)
  { 
    int N_hidden = a->N_hidden[l];
    net_value *restrict dh = d->h[l];

    memset (dh, 0, N_hidden * sizeof *dh);
    
    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { sum_derivatives_config (d->o, dh, w->ho[l], a->hidden_config[k]);
      }
      else 
      { sum_derivatives (d->o, a->N_outputs, dh, N_hidden, 
                         w->ho[l], (unsigned short *) 0, 0);
      }
    }

    for (ld = l+1; ld<a->N_layers; ld++)
    { if ((a->has_nsq[ld]>>l) & 1)
      { nsqi = ns[l][ld];
        if (a->nonseq_config[nsqi])
        { sum_derivatives_config 
            (d->h[ld], dh, w->nsq[nsqi], a->nonseq_config[nsqi]);
        }
        else 
        { sum_derivatives 
            (d->h[ld], a->N_hidden[ld], dh, N_hidden, 
             w->nsq[nsqi], (unsigned short *) 0, 0);
        }
      }
    }

    if (l<a->N_layers-1 && a->has_hh[l])
    { if (a->hidden_config[l+1])
      { sum_derivatives_config (d->h[l+1], dh, w->hh[l], a->hidden_config[l+1]);
      }
      else 
      { sum_derivatives (d->h[l+1], a->N_hidden[l+1], dh, N_hidden, 
                         w->hh[l], (unsigned short *) 0, 0);
      }
    }

    net_value const* vh = v->h[l];

    if (flgs==0 || flgs->layer_type[l]==Tanh_type)
    {
#     if FP64 && USE_SIMD_INTRINSICS && __AVX__
      { __m256d ONE = _mm256_set1_pd(1.0);
        i = 3;
        while (i<N_hidden)
        { __m256d VH = _mm256_loadu_pd(vh+i-3);
          _mm256_storeu_pd (dh+i-3, _mm256_mul_pd (_mm256_loadu_pd(dh+i-3),
                                      _mm256_sub_pd(ONE,_mm256_mul_pd(VH,VH))));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128d VH = _mm_loadu_pd(vh+i-1);
          _mm_storeu_pd (dh+i-1, _mm_mul_pd (_mm_loadu_pd(dh+i-1),
           _mm_sub_pd (cast128d(ONE), _mm_mul_pd(VH,VH))));
          i += 2;
        }
        if (i<=N_hidden)
        { __m128d VH = _mm_load_sd(vh+i-1);
          _mm_store_sd (dh+i-1, _mm_mul_sd (_mm_load_sd(dh+i-1),
           _mm_sub_sd (cast128d(ONE), _mm_mul_sd(VH,VH))));
        }
      }
#     elif FP64 && USE_SIMD_INTRINSICS && __SSE2__
      { __m128d ONE = _mm_set1_pd(1.0);
        __m128d VH;
        i = 3;
        while (i<N_hidden)
        { VH = _mm_loadu_pd(vh+i-3);
          _mm_storeu_pd (dh+i-3, _mm_mul_pd(_mm_loadu_pd(dh+i-3),
                                            _mm_sub_pd(ONE,_mm_mul_pd(VH,VH))));
          VH = _mm_loadu_pd(vh+i-1);
          _mm_storeu_pd (dh+i-1, _mm_mul_pd(_mm_loadu_pd(dh+i-1),
                                            _mm_sub_pd(ONE,_mm_mul_pd(VH,VH))));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { VH = _mm_loadu_pd(vh+i-1);
          _mm_storeu_pd (dh+i-1, _mm_mul_pd(_mm_loadu_pd(dh+i-1),
                                            _mm_sub_pd(ONE,_mm_mul_pd(VH,VH))));
          i += 2;
        }
        if (i<=N_hidden)
        { VH = _mm_load_sd(vh+i-1);
          _mm_store_sd (dh+i-1, _mm_mul_sd (_mm_load_sd(dh+i-1),
                                            _mm_sub_sd(ONE,_mm_mul_sd(VH,VH))));
        }
      }
#     elif FP32 && USE_SIMD_INTRINSICS && __AVX__
      { __m256 ONE = _mm256_set1_ps(1.0f);
        i = 7;
        while (i<N_hidden)
        { __m256 VH = _mm256_loadu_ps(vh+i-7);
          _mm256_storeu_ps (dh+i-7, _mm256_mul_ps (_mm256_loadu_ps(dh+i-7),
                                      _mm256_sub_ps(ONE,_mm256_mul_ps(VH,VH))));
          i += 8;
        }
        i -= 4;
        if (i<N_hidden)
        { __m128 VH = _mm_loadu_ps(vh+i-3);
          _mm_storeu_ps (dh+i-3, _mm_mul_ps (_mm_loadu_ps(dh+i-3),
                             _mm_sub_ps (cast128f(ONE), _mm_mul_ps(VH,VH))));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128 Z = _mm_setzero_ps();
          __m128 VH = _mm_loadl_pi(Z, (__m64 *)(vh+i-1));
          _mm_storel_pi ((__m64 *)(dh+i-1), 
                         _mm_mul_ps (_mm_loadl_pi(Z, (__m64 *)(dh+i-1)),
                           _mm_sub_ps (cast128f(ONE), _mm_mul_ps(VH,VH))));
          i += 2;
        }
        if (i<=N_hidden)
        { __m128 VH = _mm_load_ss(vh+i-1);
          _mm_store_ss (dh+i-1, _mm_mul_ss (_mm_load_ss(dh+i-1),
               _mm_sub_ss (cast128f(ONE), _mm_mul_ss(VH,VH))));
        }
      }
#     elif FP32 && USE_SIMD_INTRINSICS && __SSE2__
      { __m128 ONE = _mm_set1_ps(1.0f);
        __m128 VH;
        i = 3;
        while (i<N_hidden)
        { VH = _mm_loadu_ps(vh+i-3);
          _mm_storeu_ps (dh+i-3, _mm_mul_ps (_mm_loadu_ps(dh+i-3),
                                 _mm_sub_ps (ONE, _mm_mul_ps(VH,VH))));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128 Z = _mm_setzero_ps();
          VH = _mm_loadl_pi(Z, (__m64 *)(vh+i-1));
          _mm_storel_pi ((__m64 *)(dh+i-1), 
                         _mm_mul_ps (_mm_loadl_pi(Z, (__m64 *)(dh+i-1)),
                           _mm_sub_ps (ONE, _mm_mul_ps(VH,VH))));
          i += 2;
        }
        if (i<=N_hidden)
        { VH = _mm_load_ss(vh+i-1);
          _mm_store_ss (dh+i-1, _mm_mul_ss (_mm_load_ss(dh+i-1),
               _mm_sub_ss (ONE, _mm_mul_ss(VH,VH))));
        }
      }
#     else
      { for (i = 0; i<N_hidden; i++)
        { dh[i] *= (1 - vh[i]*vh[i]);
        }
      }
#     endif
    }
    else if (flgs->layer_type[l]==Sin_type)
    { net_value const* vs = v->h[l];
      for (i = 0; i<N_hidden; i++)
      { dh[i] = 2 * prec_cos(vs[i]*sqrt_2) * dh[i];
      }
    }
    else if (flgs->layer_type[l]==Softplus_type)
    { 
#     if FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__ && 0 /* for now */
      { __m256d ONE = _mm256_set1_pd(1.0);
        __m256d ZERO = _mm256_setzero_pd();
        i = 3;
        while (i<N_hidden)
        { __m256d NVS = _mm256_sub_pd (ZERO, _mm256_loadu_pd(vs+i-3));
          _mm256_storeu_pd (dh+i-3, 
                            _mm256_div_pd (_mm256_loadu_pd(dh+i-3),
                              _mm256_add_pd (ONE, sleef_expd4(NVS))));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128d NVS = _mm_sub_pd (cast128d(ZERO), _mm_loadu_pd(vs+i-1));
          _mm_storeu_pd (dh+i-1, 
                         _mm_div_pd (_mm_loadu_pd(dh+i-1),
                         _mm_add_pd (cast128d(ONE), sleef_expd2(NVS))));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i] = dh[i] / (1+prec_exp(-vs[i]));
        }
      }
#     elif FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__ && 0 /* for now */
      { __m128d ONE = _mm_set1_pd(1.0);
        __m128d ZERO = _mm_setzero_pd();
        i = 1;
        while (i<N_hidden)
        { __m128d NVS = _mm_sub_pd (ZERO, _mm_loadu_pd(vs+i-1));
          _mm_storeu_pd (dh+i-1, 
                         _mm_div_pd (_mm_loadu_pd(dh+i-1),
                         _mm_add_pd (ONE, sleef_expd2(NVS))));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i] = dh[i] / (1+prec_exp(-vs[i]));
        }
      }
#     elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__ && 0 /* for now */
      { __m256 ONE = _mm256_set1_ps(1.0f);
        __m256 ZERO = _mm256_setzero_ps();
        i = 7;
        while (i<N_hidden)
        { __m256 NVS = _mm256_sub_ps (ZERO, _mm256_loadu_ps(vs+i-7));
          _mm256_storeu_ps (dh+i-7, 
                            _mm256_div_ps (_mm256_loadu_ps(dh+i-7),
                            _mm256_add_ps (ONE, sleef_expf8(NVS))));
          i += 8;
        }
        i -= 4;
        while (i<N_hidden)
        { __m128 NVS = _mm_sub_ps (cast128f(ZERO), _mm_loadu_ps(vs+i-3));
          _mm_storeu_ps (dh+i-3, 
                         _mm_div_ps (_mm_loadu_ps(dh+i-3),
                         _mm_add_ps (cast128f(ONE), sleef_expf4(NVS))));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128 NVS = _mm_sub_ps (cast128f(ZERO),
                               _mm_loadl_pi(cast128f(ZERO),(__m64 *)(vs+i-1)));
          _mm_storel_pi ((__m64 *)(dh+i-1), 
                     _mm_div_ps (_mm_loadl_pi(cast128f(ZERO),(__m64 *)(dh+i-1)),
                                 _mm_add_ps (cast128f(ONE), sleef_expf4(NVS))));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i] = dh[i] / (1+prec_exp(-vs[i]));
        }
      }
#     elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__ && 0 /* for now */
      { __m128 ONE = _mm_set1_ps(1.0f);
        __m128 ZERO = _mm_setzero_ps();
        i = 3;
        while (i<N_hidden)
        { __m128 NVS = _mm_sub_ps (ZERO, _mm_loadu_ps(vs+i-3));
          _mm_storeu_ps (dh+i-3, 
                         _mm_div_ps (_mm_loadu_ps(dh+i-3),
                         _mm_add_ps (ONE, sleef_expf4(NVS))));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128 NVS = _mm_sub_ps (ZERO, _mm_loadl_pi(ZERO,(__m64 *)(vs+i-1)));
          _mm_storel_pi ((__m64 *)(dh+i-1), 
                         _mm_div_ps (_mm_loadl_pi(ZERO,(__m64 *)(dh+i-1)),
                                     _mm_add_ps (ONE, sleef_expf4(NVS))));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i] = dh[i] / (1+prec_exp(-vs[i]));
        }
      }
#     else
      { for (i = 0; i<N_hidden; i++)
        { net_value e = prec_exp(vh[i]);
          dh[i] *= (e-1) / e;
        }
      }
#     endif
    }
    else if (flgs->layer_type[l]==Square_type)
    { net_value const* vs = v->h[l];
      for (i = 0; i<N_hidden; i++)
      { dh[i] *= 2*vs[i];
      }
    }
    else if (flgs->layer_type[l]==Cube_type)
    { net_value const* vs = v->h[l];
      for (i = 0; i<N_hidden; i++)
      { dh[i] = 3*vs[i]*vs[i] * dh[i];
      }
    }
    else if (flgs->layer_type[l]==Identity_type)
    { /* nothing to do */
    }
    else
    { abort();
    }
  }

  /* Backpropagate to input layer (if needed). */

  if (start<0)
  {
    memset (d->i, 0, a->N_inputs * sizeof *d->i);
 
    for (l = 0; l<a->N_layers; l++)
    { if (a->has_ih[l])
      { if (a->input_config[l])
        { sum_derivatives_config (d->h[l], d->i, w->ih[l], a->input_config[l]);
        }
        else 
        { sum_derivatives (d->h[l], a->N_hidden[l], d->i, a->N_inputs, w->ih[l],
                      flgs && flgs->any_omitted[l]? flgs->omit : 0, 1<<(l+1));
        }
      }
    }

    if (a->has_io)
    { if (a->input_config[a->N_layers])
      { sum_derivatives_config(d->o, d->i, w->io, a->input_config[a->N_layers]);
      }
      else
      { sum_derivatives (d->o, a->N_outputs, d->i, a->N_inputs, w->io,
                    flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0, 1);
      }
    }
  }
}


/* SUM UP CONTRIBUTIONS TO THE DERIVATIVES FROM ONE GROUP OF CONNECTIONS.  Adds 
   the weighted sum of derivatives due to connections from source units to 
   a given destination layer to the totals for the source layer. */

HOSTDEV static void sum_derivatives
( net_value const* dd,    /* Derivatives with respect to destination units */
  int nd,		  /* Number of destination units */
  net_value *restrict ds, /* Derivatives w.r.t. source units to add to */
  int ns,		  /* Number of source units */
  net_param const* w,     /* Connection weights */
  unsigned short const* omit,  /* Omit flags, null if not present */
  int bit		  /* Bit to look at in omit flags */
)
{
  net_value tv;
  int i, j;

  if (omit==0)
  {
#   if FP64 && USE_SIMD_INTRINSICS && __AVX__
    { 
      if (nd==1)
      { __m256d D0 = _mm256_broadcast_sd(dd);
        i = 3;
        while (i<ns)
        { _mm256_storeu_pd (ds+i-3, FMA256_pd (D0, _mm256_loadu_pd(w+i-3),
                                                   _mm256_loadu_pd(ds+i-3)));
          i += 4;
        }
        i -= 2;
        if (i<ns)
        { _mm_storeu_pd (ds+i-1, FMA_pd (cast128d(D0), _mm_loadu_pd(w+i-1),
                                                       _mm_loadu_pd(ds+i-1)));
          i += 2;
        }
        if (i<=ns)
        { _mm_store_sd (ds+i-1, FMA_sd (cast128d(D0), _mm_load_sd(w+i-1), 
                                                      _mm_load_sd(ds+i-1)));
        }
      }
      else
      { __m256d TV, TV2;
        for (i = 1; i<ns; i+=2)
        { net_param const*w2 = w+nd;
          TV = _mm256_setzero_pd();
          TV2 = _mm256_setzero_pd();
          j = 3;
          while (j<nd)
          { __m256d DD = _mm256_loadu_pd(dd+j-3);
            TV = FMA256_pd (_mm256_loadu_pd(w+j-3), DD, TV);
            TV2 = FMA256_pd (_mm256_loadu_pd(w2+j-3), DD, TV2);
            j += 4;
          }
          __m128d T, T2;
          T = _mm_add_pd (cast128d(TV),
                          _mm256_extractf128_pd(TV,1));
          T2 = _mm_add_pd (cast128d(TV2),
                           _mm256_extractf128_pd(TV2,1));
          j -= 2;
          if (j<nd)
          { __m128d DD = _mm_loadu_pd(dd+j-1);
            T = FMA_pd (_mm_loadu_pd(w+j-1), DD, T);
            T2 = FMA_pd (_mm_loadu_pd(w2+j-1), DD, T2);
            j += 2;
          }
          T = _mm_hadd_pd(T,T2);
          if (j<=nd)
          { __m128d DD = _mm_load_pd1(dd+j-1);
            __m128d WW = _mm_loadh_pd (_mm_load_sd(w+j-1), w2+j-1);
            T = FMA_pd (WW, DD, T);
          }
          _mm_storeu_pd (ds+i-1, _mm_add_pd (_mm_loadu_pd(ds+i-1), T));
          w = w2+nd;
        }
        if (i<=ns)
        { TV = _mm256_setzero_pd();
          j = 3;
          while (j<nd)
          { TV = FMA256_pd(_mm256_loadu_pd(w+j-3), _mm256_loadu_pd(dd+j-3), TV);
            j += 4;
          }
          __m128d T;
          T = _mm_add_pd (cast128d(TV),
                          _mm256_extractf128_pd(TV,1));
          j -= 2;
          if (j<nd)
          { T = FMA_pd (_mm_loadu_pd(w+j-1), _mm_loadu_pd(dd+j-1), T);
            j += 2;
          }
          T = _mm_hadd_pd(T,T);
          if (j<=nd)
          { T = FMA_sd (_mm_load_sd(w+j-1), _mm_load_sd(dd+j-1), T);
          }
          _mm_store_sd (ds+i-1, _mm_add_sd(_mm_load_sd(ds+i-1), T));
        }
      }
    }

#   elif FP64 && USE_SIMD_INTRINSICS && __SSE3__
    { 
      if (nd==1)
      { __m128d D0 = _mm_set1_pd(*dd);
        i = 3;
        while (i<ns)
        { _mm_storeu_pd (ds+i-3, FMA_pd (D0, _mm_loadu_pd(w+i-3),
                                             _mm_loadu_pd(ds+i-3)));
          _mm_storeu_pd (ds+i-1, FMA_pd (D0, _mm_loadu_pd(w+i-1),
                                             _mm_loadu_pd(ds+i-1)));
          i += 4;
        }
        i -= 2;
        if (i<ns)
        { _mm_storeu_pd (ds+i-1, FMA_pd (D0, _mm_loadu_pd(w+i-1),
                                             _mm_loadu_pd(ds+i-1)));
          i += 2;
        }
        if (i<=ns)
        { _mm_store_sd (ds+i-1, FMA_sd (D0, _mm_load_sd(w+i-1), 
                                            _mm_load_sd(ds+i-1)));
        }
      }
      else
      { __m128d TVa, TVb, TV2a, TV2b;
        for (i = 1; i<ns; i+=2)
        { net_param const*w2 = w+nd;
          TVa = TVb = _mm_setzero_pd();
          TV2a = TV2b = _mm_setzero_pd();
          j = 3;
          while (j<nd)
          { __m128d DD;
            DD = _mm_loadu_pd(dd+j-3);
            TVa = FMA_pd (_mm_loadu_pd(w+j-3), DD, TVa);
            TV2a = FMA_pd (_mm_loadu_pd(w2+j-3), DD, TV2a);
            DD = _mm_loadu_pd(dd+j-1);
            TVb = FMA_pd (_mm_loadu_pd(w+j-1), DD, TVb);
            TV2b = FMA_pd (_mm_loadu_pd(w2+j-1), DD, TV2b);
            j += 4;
          }
          __m128d T, T2;
          T = _mm_add_pd (TVa, TVb);
          T2 = _mm_add_pd (TV2a, TV2b);
          j -= 2;
          if (j<nd)
          { __m128d DD = _mm_loadu_pd(dd+j-1);
            T = FMA_pd (_mm_loadu_pd(w+j-1), DD, T);
            T2 = FMA_pd (_mm_loadu_pd(w2+j-1), DD, T2);
            j += 2;
          }
          T = _mm_hadd_pd(T,T2);
          if (j<=nd)
          { __m128d DD = _mm_load_pd1(dd+j-1);
            __m128d WW = _mm_loadh_pd (_mm_load_sd(w+j-1), w2+j-1);
            T = FMA_pd (WW, DD, T);
          }
          _mm_storeu_pd (ds+i-1, _mm_add_pd (_mm_loadu_pd(ds+i-1), T));
          w = w2+nd;
        }
        if (i<=ns)
        { TVa = TVb = _mm_setzero_pd();
          j = 3;
          while (j<nd)
          { TVa = FMA_pd(_mm_loadu_pd(w+j-3), _mm_loadu_pd(dd+j-3), TVa);
            TVb = FMA_pd(_mm_loadu_pd(w+j-1), _mm_loadu_pd(dd+j-1), TVb);
            j += 4;
          }
          __m128d T;
          T = _mm_add_pd (TVa, TVb);
          j -= 2;
          if (j<nd)
          { T = FMA_pd (_mm_loadu_pd(w+j-1), _mm_loadu_pd(dd+j-1), T);
            j += 2;
          }
          T = _mm_hadd_pd(T,T);
          if (j<=nd)
          { T = FMA_sd (_mm_load_sd(w+j-1), _mm_load_sd(dd+j-1), T);
          }
          _mm_store_sd (ds+i-1, _mm_add_sd(_mm_load_sd(ds+i-1), T));
        }
      }
    }

#   elif FP32 && USE_SIMD_INTRINSICS && __SSE3__
    { 
      __m128 Z = _mm_setzero_ps();
      if (nd==1)
      { __m128 D0 = _mm_set1_ps(*dd);
        i = 3;
        while (i<ns)
        { _mm_storeu_ps (ds+i-3, FMA_ps (D0, _mm_loadu_ps(w+i-3),
                                             _mm_loadu_ps(ds+i-3)));
          i += 4;
        }
        i -= 2;
        if (i<ns)
        { _mm_storel_pi ((__m64 *)(ds+i-1), 
                         FMA_ps (D0, _mm_loadl_pi (Z, (__m64 *)(w+i-1)),
                                     _mm_loadl_pi (Z, (__m64 *)(ds+i-1))));
          i += 2;
        }
        if (i<=ns)
        { _mm_store_ss (ds+i-1, FMA_ss (D0, _mm_load_ss(w+i-1), 
                                            _mm_load_ss(ds+i-1)));
        }
      }
      else
      { __m128 TV, TV2;
        for (i = 1; i<ns; i+=2)
        { net_param const*w2 = w+nd;
          TV = _mm_setzero_ps();
          TV2 = _mm_setzero_ps();
          j = 3;
          while (j<nd)
          { __m128 DD = _mm_loadu_ps(dd+j-3);
            TV = FMA_ps (_mm_loadu_ps(w+j-3), DD, TV);
            TV2 = FMA_ps (_mm_loadu_ps(w2+j-3), DD, TV2);
            j += 4;
          }
          __m128 T, T2;
          T = _mm_add_ps (TV, _mm_movehl_ps(Z,TV));
          T2 = _mm_add_ps (TV2, _mm_movehl_ps(Z,TV2));
          j -= 2;
          if (j<nd)
          { __m128 DD = _mm_loadl_pi (Z, (__m64 *)(dd+j-1));
            T = FMA_ps (_mm_loadl_pi (Z, (__m64 *)(w+j-1)), DD, T);
            T2 = FMA_ps (_mm_loadl_pi (Z, (__m64 *)(w2+j-1)), DD, T2);
            j += 2;
          }
          if (j<=nd)
          { __m128 DD = _mm_load_ss(dd+j-1);
            T = FMA_ps (_mm_load_ss(w+j-1), DD, T);
            T2 = FMA_ps (_mm_load_ss(w2+j-1), DD, T2);
          }
          T = _mm_shuffle_ps (_mm_hadd_ps(T,T2), Z, 8);
          _mm_storel_pi((__m64 *)(ds+i-1), _mm_add_ps (T, 
                                            _mm_loadl_pi(Z,(__m64 *)(ds+i-1))));
          w = w2+nd;
        }
        if (i<=ns)
        { TV = _mm_setzero_ps();
          j = 3;
          while (j<nd)
          { TV = FMA_ps (_mm_loadu_ps(w+j-3), _mm_loadu_ps(dd+j-3), TV);
            j += 4;
          }
          __m128 T;
          T = _mm_add_ps (TV, _mm_movehl_ps(Z,TV));
          j -= 2;
          if (j<nd)
          { T = FMA_ps (_mm_loadl_pi (Z, (__m64 *)(w+j-1)), 
                        _mm_loadl_pi (Z, (__m64 *)(dd+j-1)), T);
            j += 2;
          }
          if (j<=nd)
          { T = FMA_ps (_mm_load_ss(w+j-1), _mm_load_ss(dd+j-1), T);
          }
          T = _mm_hadd_ps(T,T);
          _mm_store_ss (ds+i-1, _mm_add_ss (_mm_load_ss(ds+i-1), T));
        }
      }
    }

#   else
    {
      if (nd==1)
      { net_value d0 = dd[0];
        i = 3;
        while (i<ns)
        { ds[i-3] += w[i-3] * d0;
          ds[i-2] += w[i-2] * d0;
          ds[i-1] += w[i-1] * d0;
          ds[i-0] += w[i-0] * d0;
          i += 4;
        }
        i -= 3;
        while (i<ns)
        { ds[i] += w[i] * d0;
          i += 1;
        }
      }
      else
      { for (i = 0; i<ns; i++)
        { tv = 0;
          j = 3;
          while (j<nd)
          { tv += w[j-3] * dd[j-3];
            tv += w[j-2] * dd[j-2];
            tv += w[j-1] * dd[j-1];
            tv += w[j-0] * dd[j-0];
            j += 4;
          }
          j -= 3;
          while (j<nd)
          { tv += w[j] * dd[j];
            j += 1;
          }
          w += nd;
          ds[i] += tv;
        }
      }
    }

#   endif

  }
  else  /* omit is not absent */
  { 
    if (nd==1)
    { net_value d0 = dd[0];
      i = 3;
      while (i<ns)
      { if (! (omit[i-3] & bit)) ds[i-3] += *w++ * d0;
        if (! (omit[i-2] & bit)) ds[i-2] += *w++ * d0;
        if (! (omit[i-1] & bit)) ds[i-1] += *w++ * d0;
        if (! (omit[i-0] & bit)) ds[i-0] += *w++ * d0;
        i += 4;
      }
      i -= 3;
      while (i<ns)
      { if (! (omit[i] & bit)) ds[i] += *w++ * d0;
        i += 1;
      }
    }
    else
    { for (i = 0; i<ns; i++)
      { if ((omit) && ((omit)[i]&(bit))) continue;
        tv = 0;
        j = 3;
        while (j<nd)
        { tv += w[j-3] * dd[j-3];
          tv += w[j-2] * dd[j-2];
          tv += w[j-1] * dd[j-1];
          tv += w[j-0] * dd[j-0];
          j += 4;
        }
        j -= 3;
        while (j<nd)
        { tv += w[j] * dd[j];
          j += 1;
        }
        w += nd;
        ds[i] += tv;
      }
    }
  }
}


/* SUM UP CONTRIBUTIONS TO THE DERIVATIVES FROM CONNECTIONS WITH CONFIGURATION.
   Adds the weighted sum of derivatives due to connections from source units to
   a given destination layer to the totals for the source layer. */

HOSTDEV static void sum_derivatives_config
( net_value const* dd,    /* Derivatives with respect to destination units */
  net_value *restrict ds, /* Derivatives w.r.t. source units to add to */
  net_param const* w,     /* Connection weights */
  net_config const* cf    /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w;
#   if FP64 && USE_SIMD_INTRINSICS && __AVX__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { i = cn[c].s; j = cn[c].d;
        __m256d P = _mm256_mul_pd (_mm256_loadu_pd(dd+j), _mm256_loadu_pd(w+k));
        __m128d S = _mm_add_pd (_mm256_extractf128_pd(P,1), cast128d(P));
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
      }
    }
#   elif FP64 && USE_SIMD_INTRINSICS && __SSE3__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { i = cn[c].s; j = cn[c].d;
        __m128d S = _mm_add_pd (
                     _mm_mul_pd (_mm_loadu_pd(dd+j), _mm_loadu_pd(w+k)),
                     _mm_mul_pd (_mm_loadu_pd(dd+j+2), _mm_loadu_pd(w+k+2)));
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __SSE3__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { i = cn[c].s; j = cn[c].d;
        __m128 P = _mm_mul_ps (_mm_loadu_ps(dd+j), _mm_loadu_ps(w+k));
        __m128 S = _mm_add_ps (_mm_movehl_ps(P,P), P);
        _mm_store_ss (ds+i, _mm_add_ss (_mm_load_ss(ds+i), _mm_hadd_ps(S,S)));
      }
    }
#   else
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { i = cn[c].s; j = cn[c].d;
        ds[i] += (dd[j+0]*w[k+0] + dd[j+2]*w[k+2])      /* same order as SIMD */
                   + (dd[j+1]*w[k+1] + dd[j+3]*w[k+3]); /* instructions above */
      }
    }
#   endif
    cn = cf->quad_s_4d_4w_2;
#   if FP64 && USE_SIMD_INTRINSICS && __AVX__
    { for (c = 0; (k = cn[c].w) >= 0; c+=2)
      { __m256d WK = _mm256_loadu_pd(w+k);
        __m256d P;
        __m128d S;
        i = cn[c].s; j = cn[c].d;
        P = _mm256_mul_pd (_mm256_loadu_pd(dd+j), WK);
        S = _mm_add_pd (_mm256_extractf128_pd(P,1), cast128d(P));
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
        i = cn[c+1].s; j = cn[c+1].d;
        P = _mm256_mul_pd (_mm256_loadu_pd(dd+j), WK);
        S = _mm_add_pd (_mm256_extractf128_pd(P,1), cast128d(P));
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
      }
    }
#   elif FP64 && USE_SIMD_INTRINSICS && __SSE3__
    { for (c = 0; (k = cn[c].w) >= 0; c+=2)
      { __m128d WK = _mm_loadu_pd(w+k), WK2 = _mm_loadu_pd(w+k+2);
        __m128d S;
        i = cn[c].s; j = cn[c].d;
        S = _mm_add_pd (
                     _mm_mul_pd (_mm_loadu_pd(dd+j), WK),
                     _mm_mul_pd (_mm_loadu_pd(dd+j+2), WK2));
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
        i = cn[c+1].s; j = cn[c+1].d;
        S = _mm_add_pd (
                     _mm_mul_pd (_mm_loadu_pd(dd+j), WK),
                     _mm_mul_pd (_mm_loadu_pd(dd+j+2), WK2));
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __SSE3__
    { for (c = 0; (k = cn[c].w) >= 0; c+=2)
      { __m128 WK = _mm_loadu_ps(w+k);
        __m128 P, S;
        i = cn[c].s; j = cn[c].d;
        P = _mm_mul_ps (_mm_loadu_ps(dd+j), WK);
        S = _mm_add_ps (_mm_movehl_ps(P,P), P);
        _mm_store_ss (ds+i, _mm_add_ss (_mm_load_ss(ds+i), _mm_hadd_ps(S,S)));
        i = cn[c+1].s; j = cn[c+1].d;
        P = _mm_mul_ps (_mm_loadu_ps(dd+j), WK);
        S = _mm_add_ps (_mm_movehl_ps(P,P), P);
        _mm_store_ss (ds+i, _mm_add_ss (_mm_load_ss(ds+i), _mm_hadd_ps(S,S)));
      }
    }
#   else
    { for (c = 0; (k = cn[c].w) >= 0; c+=2)
      { net_value w0 = w[k+0];
        net_value w1 = w[k+1];
        net_value w2 = w[k+2];
        net_value w3 = w[k+3];
        i = cn[c].s; j = cn[c].d;
        ds[i] += (dd[j+0]*w0 + dd[j+2]*w2)      /* same order as SIMD */
                   + (dd[j+1]*w1 + dd[j+3]*w3); /* instructions above */
        i = cn[c+1].s; j = cn[c+1].d;
        ds[i] += (dd[j+0]*w0 + dd[j+2]*w2)      /* same order as SIMD */
                   + (dd[j+1]*w1 + dd[j+3]*w3); /* instructions above */
      }
    }
#   endif
  }

  if (CONFIG_SINGLE4)
  { 
    cn = cf->single4_s;
    for (c = 0; (k = cn[c].w) >= 0; c+=4)
    { i = cn[c].s;
      net_value dsi = ds[i];
      j = cn[c].d;
      dsi += dd[j] * w[k];
      j = cn[c+1].d; k = cn[c+1].w; 
      dsi += dd[j] * w[k];
      j = cn[c+2].d; k = cn[c+2].w; 
      dsi += dd[j] * w[k];
      j = cn[c+3].d; k = cn[c+3].w; 
      dsi += dd[j] * w[k];
      ds[i] = dsi;
    }

    cn = cf->single4_d;
    for (c = 0; (k = cn[c].w) >= 0; c+=4)
    { net_value ddj = dd[cn[c].d];
      i = cn[c].s;
      ds[i] += ddj * w[k];
      i = cn[c+1].s; k = cn[c+1].w; 
      ds[i] += ddj * w[k];
      i = cn[c+2].s; k = cn[c+2].w; 
      ds[i] += ddj * w[k];
      i = cn[c+3].s; k = cn[c+3].w; 
      ds[i] += ddj * w[k];
    }
  }

  cn = CONFIG_ORIGINAL ? cf->conn : cf->single;
  for (c = 0; (k = cn[c].w) >= 0; c++)
  { i = cn[c].s; j = cn[c].d;
    ds[i] += dd[j] * w[k];
  }
}


/* -------------------------- net_back_gpu --------------------------------- */

__device__ static void sum_derivatives_gpu 
    (int, net_value const*, int, net_value *restrict, 
     int, net_param const*, unsigned short const*, int);

__device__ static void sum_derivatives_config_gpu
    (int, net_value const*, net_value *restrict,
     net_param const*, net_config const*);


/* BACKPROPAGATE ERROR DERIVATIVES, GPU VERSION.

   Assumes that threads have been synchronized on entry, so that
   d->o[i] is accessible to all threads for all i.

   Synchronizes threads after last layer computation if asked to;
   otherwise, thread 'th' will have computed values of d->h[i] with i
   mod THREAD_PER_CASE equal to 'th', and can use those values without
   synchronization later. */

#if __CUDACC__

__device__ void net_back_gpu
( int th,		/* Which thread, negative for surplus thread */
  net_values const*v,	/* Values for units in network */
  net_values *restrict d,/* Place to get output derivatives, and store others */
  int start,		/* Earliest layer to find derivatives for */
  net_arch const*a,	/* Network architecture */
  net_flags const*flgs,	/* Network flags, null if none */
  net_params const*w,	/* Network parameters */
  int sync		/* Sync threads after last layer computation? */
)
{
  int l, ls, ld, i;

  /* Find locations of non-sequential configurations. */

  char ns[Max_nonseq][Max_nonseq];
  unsigned bits;
  int nsqi = 0;
  for (l = 0; l<a->N_layers; l++)
  { for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (ls>=l-1) abort();
      ns[ls][l] = nsqi;
      nsqi += 1;
    }
  }

  /* Backpropagate through hidden layers. */

  for (l = a->N_layers-1; l>=0 && l>=start; l--)
  { 
    int N_hidden = a->N_hidden[l];
    net_value *restrict dh = d->h[l];

    if (th<0) goto sync_layer;

    for (i = th; i<N_hidden; i+=NTH)
    { dh[i] = 0;
    }
    
    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { sum_derivatives_config_gpu 
          (th, d->o, dh, w->ho[l], a->hidden_config[k]);
      }
      else 
      { sum_derivatives_gpu 
          (th, d->o, a->N_outputs, dh, N_hidden, 
           w->ho[l], (unsigned short *) 0, 0);
      }
    }

    for (ld = l+1; ld<a->N_layers; ld++)
    { if ((a->has_nsq[ld]>>l) & 1)
      { nsqi = ns[l][ld];
        if (a->nonseq_config[nsqi])
        { sum_derivatives_config_gpu
            (th, d->h[ld], dh, w->nsq[nsqi], a->nonseq_config[nsqi]);
        }
        else
        { sum_derivatives_gpu
            (th, d->h[ld], a->N_hidden[ld], dh, N_hidden,
             w->nsq[nsqi], (unsigned short *) 0, 0);
        }
      }
    }

    if (l<a->N_layers-1 && a->has_hh[l])
    { if (a->hidden_config[l+1])
      { sum_derivatives_config_gpu 
          (th, d->h[l+1], dh, w->hh[l], a->hidden_config[l+1]);
      }
      else 
      { sum_derivatives_gpu 
          (th, d->h[l+1], a->N_hidden[l+1], dh, N_hidden, 
           w->hh[l], (unsigned short *) 0, 0);
      }
    }

    if (flgs==0 || flgs->layer_type[l]==Tanh_type)
    { net_value const* vh = v->h[l];
      for (i = th; i<N_hidden; i+=NTH)
      { dh[i] *= (1 - vh[i]*vh[i]);
      }
    }
    else if (flgs->layer_type[l]==Sin_type)
    { net_value const* vs = v->h[l];
      for (i = th; i<N_hidden; i+=NTH)
      { dh[i] *= 2 * prec_cos(vs[i]*sqrt_2);
      }
    }
    else if (flgs->layer_type[l]==Softplus_type)
    { net_value const* vh = v->h[l];
      for (i = th; i<N_hidden; i+=NTH)
      { net_value e = prec_exp(vh[i]);
        dh[i] *= (e-1) / e;
      }
    }
    else if (flgs->layer_type[l]==Square_type)
    { net_value const* vs = v->h[l];
      for (i = th; i<N_hidden; i+=NTH)
      { dh[i] *= 2*vs[i] * dh[i];
      }
    }
    else if (flgs->layer_type[l]==Cube_type)
    { net_value const* vs = v->h[l];
      for (i = th; i<N_hidden; i+=NTH)
      { dh[i] *= 3*vs[i]*vs[i] * dh[i];
      }
    }
    else /* identity */
    { /* nothing to do */
    }

  sync_layer:
    if (NTH>1 && (l>start || sync))
    { __syncthreads();
    }
  }

  /* Backpropagate to input layer (if needed). */

  if (start<0)
  {
    if (th<0) goto sync_input;

    for (i = th; i<a->N_inputs; i+=NTH)
    { d->i[i] = 0;
    }
 
    for (l = 0; l<a->N_layers; l++)
    { if (a->has_ih[l])
      { if (a->input_config[l])
        { sum_derivatives_config_gpu 
            (th, d->h[l], d->i, w->ih[l], a->input_config[l]);
        }
        else 
        { sum_derivatives_gpu
            (th, d->h[l], a->N_hidden[l], d->i, a->N_inputs, w->ih[l],
             flgs && flgs->any_omitted[l]? flgs->omit : 0, 1<<(l+1));
        }
      }
    }

    if (a->has_io)
    { if (a->input_config[a->N_layers])
      { sum_derivatives_config_gpu
          (th, d->o, d->i, w->io, a->input_config[a->N_layers]);
      }
      else
      { sum_derivatives_gpu 
          (th, d->o, a->N_outputs, d->i, a->N_inputs, w->io,
           flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0, 1);
      }
    }

  sync_input:
    if (NTH>1 && sync)
    { __syncthreads();
    }
  }
}


/* SUM UP CONTRIBUTIONS TO THE DERIVATIVES FROM ONE GROUP OF CONNECTIONS.  Adds 
   the weighted sum of derivatives due to connections from source units to 
   a given destination layer to the totals for the source layer. */

__device__ static void sum_derivatives_gpu
( int th,		/* Which thread, negative for surplus thread */
  net_value const* dd,    /* Derivatives with respect to destination units */
  int nd,		  /* Number of destination units */
  net_value *restrict ds, /* Derivatives w.r.t. source units to add to */
  int ns,		  /* Number of source units */
  net_param const* w,     /* Connection weights */
  unsigned short const* omit,  /* Omit flags, null if not present */
  int bit		  /* Bit to look at in omit flags */
)
{
  net_value tv;
  int i, j, k;

  if (omit==0)
  { if (nd==1)
    { net_value d0 = dd[0];
      for (i = th; i<ns; i+=NTH)
      { ds[i] += w[i] * d0;
      }
    }
    else
    { for (i = th; i<ns; i+=NTH)
      { const net_param *ww = w+i*nd;
        tv = 0;
        for (j = 0; j<nd; j++)
        { tv += ww[j] * dd[j];
        }
        ds[i] += tv;
      }
    }
  }
  else  /* omit is not absent */
  { if (nd==1)
    { net_value d0 = dd[0];
      k = 0;
      for (i = th; i<ns; i+=NTH)
      { while (k < i) 
        { if (!(omit[k]&bit)) w += 1;
          k += 1;
        }
        if (omit[i]&bit) 
        { k += 1;
          continue;
        }
        ds[i] += *w * d0;
      }
    }
    else
    { k = 0;
      for (i = th; i<ns; i+=NTH)
      { while (k < i) 
        { if (!(omit[k]&bit)) w += nd;
          k += 1;
        }
        if (omit[i]&bit) 
        { k += 1;
          continue;
        }
        tv = 0;
        for (j = 0; j<nd; j++)
        { tv += w[j] * dd[j];
        }
        ds[i] += tv;
      }
    }
  }
}


/* SUM UP CONTRIBUTIONS TO THE DERIVATIVES FROM CONNECTIONS WITH CONFIGURATION.
   Adds the weighted sum of derivatives due to connections from source units to
   a given destination layer to the totals for the source layer. */

__device__ static void sum_derivatives_config_gpu
( int th,		/* Which thread, negative for surplus thread */
  net_value const* dd,    /* Derivatives with respect to destination units */
  net_value *restrict ds, /* Derivatives w.r.t. source units to add to */
  net_param const* w,     /* Connection weights */
  net_config const* cf    /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int c, i, j, k, m;

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_sgpu;
    for (m = 0; m<4; m+=NTH)
    { c = cf->start_quad_sgpu[th+m];
      for (;;)
      { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        ds[i] = ds[i] + dd[j+0]*w[k+0] + dd[j+1]*w[k+1]  /* written so it can */
                      + dd[j+2]*w[k+2] + dd[j+3]*w[k+3]; /*  use multiply-add */
      }
    }
  }

  cn = cf->other_sgpu;
  for (m = 0; m<4; m+=NTH)
  { c = cf->start_other_sgpu[th+m];
    for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      ds[i] += dd[j] * w[k];
    }
  }
}

#endif
