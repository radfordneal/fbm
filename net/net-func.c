/* NET-FUNC.C - Routine for calculating the function defined by a network. */

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

#include "intrinsics-use.h"
#define __SLEEF_REMPITAB__  /* Need to leave undefined in exactly one file */
#include "sleef-use.h"

#ifndef CHECK_NAN
#define CHECK_NAN 0                 /* Normally 0, can set to 1 for debugging */
#endif

#endif


#define USE_QUICK_AND_DIRTY_TANH 1  /* Whether to use the faster tanh below */

#if USE_QUICK_AND_DIRTY_TANH
# define TANH(x) quick_and_dirty_tanh(x)
#else
# define TANH(x) prec_tanh(x)
#endif


/* COMPUTE TANH QUICKLY, BUT NOT VERY ACCURATELY.  Loses accuracy for
   x near zero, but that shouldn't matter much for neural network use. */

#define quick_and_dirty_tanh(x) (1 - 2 / (1+prec_exp(2*(x))))


/* This module calculates the values of the output units in a network, given 
   values for the input units.  The values of hidden units are calculated
   along the way.  There are facilities for starting the calculation on the 
   assumption the values are already known up to some layer, as would be 
   the case if the weights into earlier layers have not changed since the
   last calculation. 
*/

static void bias_values (net_value *restrict, int, net_param const*);

static void bias_values_config (net_value *restrict, int, 
                                net_param const*, net_config const*);

static void add_connections (net_value *restrict, int, net_value const*,
                             int, net_param const*, net_param const*,
                             unsigned short const*, int, int);

static void add_connections_config (net_value *restrict, int,
                                    net_value const*,
                                    net_param const*, net_param const*,
                                    net_config const*);


/* ---------------------------- net_func ----------------------------------- */

/* EVALUATE NETWORK FUNCTION FOR GIVEN INPUTS.  The inputs are taken from
   the net_values structure passed. */

void STATIC_IF_INCLUDED net_func 
( net_values *restrict v, /* Place to get inputs and store outputs */
  net_arch const* a,	/* Network architecture */
  net_precomputed const* pre,  /* Precomputed aspects of architecture */
  net_flags const* flgs,/* Network flags, null if none */
  net_params const* w,	/* Network parameters */
  int sparse		/* Are input values sparse? */
)
{
  int l, ls, j;

  /* Compute values for successive hidden layers. */

  for (l = 0; l<a->N_layers; l++)
  {
    int N_hidden = a->N_hidden[l];

    net_value *restrict vh = v->h[l];

    if (a->has_bh[l])
    { if (a->bias_config[l])
      { bias_values_config (vh, N_hidden, w->bh[l], a->bias_config[l]);
      }
      else
      { bias_values (vh, N_hidden, w->bh[l]);
      }
    }
    else
    { memset (vh, 0, N_hidden * sizeof *vh);
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { add_connections_config (vh, N_hidden, v->i, w->ih[l], 
          a->has_ti ? w->ti : 0, a->input_config[l]);
      }
      else
      { add_connections (vh, N_hidden, v->i, a->N_inputs, 
          w->ih[l], a->has_ti ? w->ti : 0, 
          flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1), sparse);
      }
    }

    if (a->has_nsq[l])
    { for (ls = 0; ls<l; ls++)
      { int nsqi = pre->nonseq[ls][l];
        if (nsqi>=0)
        { if (a->nonseq_config[nsqi])
          { add_connections_config (vh, N_hidden, v->h[ls], w->nsq[nsqi], 
              a->has_th[ls] ? w->th[ls] : 0, a->nonseq_config[nsqi]);
          }
          else
          { add_connections (vh, N_hidden, v->h[ls], a->N_hidden[ls],
              w->nsq[nsqi], a->has_th[ls] ? w->th[ls] : 0, 
              (unsigned short *) 0, 0, 0);
          }
        }
      }
    }

    if (l>0 && a->has_hh[l-1])
    { if (a->hidden_config[l])
      { add_connections_config (vh, N_hidden, v->h[l-1], w->hh[l-1], 
          a->has_th[l-1] ? w->th[l-1] : 0, a->hidden_config[l]);
      }
      else
      { add_connections (vh, N_hidden, v->h[l-1], a->N_hidden[l-1],
          w->hh[l-1], a->has_th[l-1] ? w->th[l-1] : 0, (unsigned short *) 0, 0,
          0);
      }
    }

    /* Put values through hidden unit activation function. */

    if (flgs==0 || flgs->layer_type[l]==Tanh_type)
    { 
#     if USE_QUICK_AND_DIRTY_TANH
#       if FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
        { __m256d one = _mm256_set1_pd(1.0);
          __m256d two = _mm256_set1_pd(2.0);
          j = 3;
          while (j<N_hidden)
          { __m256d x = _mm256_loadu_pd(vh+j-3);
            x = _mm256_add_pd(x,x);
            x = sleef_expd4(x);
            x = _mm256_sub_pd(one, _mm256_div_pd (two, _mm256_add_pd (one, x)));
            _mm256_storeu_pd (vh+j-3, x);
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { __m128d x = _mm_loadu_pd(vh+j-1);
            x = _mm_add_pd(x,x);
            x = sleef_expd2(x);
            x = _mm_sub_pd (cast128d(one), 
                   _mm_div_pd (cast128d(two), _mm_add_pd (cast128d(one), x)));
            _mm_storeu_pd (vh+j-1, x);
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (vh[j-1]);
          }
        }
#       elif FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
        { __m128d one = _mm_set1_pd(1.0);
          __m128d two = _mm_set1_pd(2.0);
          j = 1;
          while (j<N_hidden)
          { __m128d x = _mm_loadu_pd(vh+j-1);
            x = _mm_add_pd(x,x);
            x = sleef_expd2(x);
            x = _mm_sub_pd (one, _mm_div_pd (two, _mm_add_pd (one, x)));
            _mm_storeu_pd (vh+j-1, x);
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (vh[j-1]);
          }
        }
#       elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
        { __m256 one = _mm256_set1_ps(1.0f);
          __m256 two = _mm256_set1_ps(2.0f);
          j = 7;
          while (j<N_hidden)
          { __m256 x = _mm256_loadu_ps(vh+j-7);
            x = _mm256_add_ps(x,x);
            x = sleef_expf8(x);
            x = _mm256_sub_ps(one, _mm256_div_ps (two, _mm256_add_ps (one, x)));
            _mm256_storeu_ps (vh+j-7, x);
            j += 8;
          }
          j -= 4;
          while (j<N_hidden)
          { __m128 x = _mm_loadu_ps(vh+j-3);
            x = _mm_add_ps(x,x);
            x = sleef_expf4(x);
            x = _mm_sub_ps (cast128f(one), 
                    _mm_div_ps (cast128f(two), _mm_add_ps (cast128f(one), x)));
            _mm_storeu_ps (vh+j-3, x);
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { __m128 x = _mm_loadl_pi (_mm_setzero_ps(), (__m64 *)(vh+j-1));
            x = _mm_add_ps(x,x);
            x = sleef_expf4(x);
            x = _mm_sub_ps (cast128f(one), 
                    _mm_div_ps (cast128f(two), _mm_add_ps (cast128f(one), x)));
            _mm_storel_pi ((__m64 *)(vh+j-1), x);
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (vh[j-1]);
          }
        }
#       elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
        { __m128 one = _mm_set1_ps(1.0f);
          __m128 two = _mm_set1_ps(2.0f);
          j = 3;
          while (j<N_hidden)
          { __m128 x = _mm_loadu_ps(vh+j-3);
            x = _mm_add_ps(x,x);
            x = sleef_expf4(x);
            x = _mm_sub_ps(one, _mm_div_ps (two, _mm_add_ps (one, x)));
            _mm_storeu_ps (vh+j-3, x);
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { __m128 x = _mm_loadl_pi (_mm_setzero_ps(), (__m64 *)(vh+j-1));
            x = _mm_add_ps(x,x);
            x = sleef_expf4(x);
            x = _mm_sub_ps (one, _mm_div_ps (two, _mm_add_ps (one, x)));
            _mm_storel_pi ((__m64 *)(vh+j-1), x);
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (vh[j-1]);
          }
        }
#       else
        { for (j = 0; j<N_hidden; j++)
          { vh[j] = TANH (vh[j]);
          }
        }
#       endif

#     else  /* Use actual tanh functions, not quick and dirty */

#       if FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
        { j = 3;
          while (j<N_hidden)
          { _mm256_storeu_pd (vh+j-3, sleef_tanhd4 (_mm256_loadu_pd(vh+j-3)));
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { _mm_storeu_pd (vh+j-1, sleef_tanhd2 (_mm_loadu_pd(vh+j-1)));
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (vh[j-1]);
          }
        }
#       elif FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
        { j = 1;
          while (j<N_hidden)
          { _mm_storeu_pd (vh+j-1, sleef_tanhd2 (_mm_loadu_pd(vh+j-1)));
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (vh[j-1]);
          }
        }
#       elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
        { j = 7;
          while (j<N_hidden)
          { _mm256_storeu_ps (vh+j-7, sleef_tanhf8 (_mm256_loadu_ps(vh+j-7)));
            j += 8;
          }
          j -= 4;
          while (j<N_hidden)
          { _mm_storeu_ps (vh+j-3, sleef_tanhf4 (_mm_loadu_ps(vh+j-3)));
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { _mm_storel_pi ((__m64 *)(vh+j-1), 
               sleef_tanhf4 (_mm_loadl_pi(_mm_setzero_ps(),(__m64 *)(vh+j-1))));
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (vh[j-1]);
          }
        }
#       elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
        { j = 3;
          while (j<N_hidden)
          { _mm_storeu_ps (vh+j-3, sleef_tanhf4 (_mm_loadu_ps(vh+j-3)));
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { _mm_storel_pi ((__m64 *)(vh+j-1), 
               sleef_tanhf4 (_mm_loadl_pi(_mm_setzero_ps(),(__m64 *)(vh+j-1))));
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (vh[j-1]);
          }
        }
#       else
        { for (j = 0; j<N_hidden; j++)
          { vh[j] = TANH (vh[j]);
          }
        }
#       endif
#     endif
    }
    else if (flgs->layer_type[l]==Softplus_type)
    {
#     if FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
      { __m256d zero = _mm256_setzero_pd();
        __m256d one = _mm256_set1_pd(1.0);
        __m256d mask = 
          _mm256_castsi256_pd (_mm256_set1_epi64x ((long long)1<<63));
        j = 3;
        while (j<N_hidden)
        { __m256d a = _mm256_loadu_pd(vh+j-3);
          __m256d v = _mm256_or_pd(a,mask);  /* compute -fabs(a) */
          v = sleef_expd4(v);
          v = _mm256_add_pd(one,v);
          v = sleef_logd4(v);
          v = _mm256_add_pd (v, _mm256_and_pd (a, 
                                  _mm256_cmp_pd(a,zero,_CMP_GT_OQ)));
          _mm256_storeu_pd (vh+j-3, v);
          j += 4;
        }
        j -= 2;
        if (j<N_hidden)
        { __m128d a = _mm_loadu_pd(vh+j-1);
          __m128d v = _mm_or_pd(a,cast128d(mask));
          v = sleef_expd2(v);
          v = _mm_add_pd(cast128d(one),v);
          v = sleef_logd2(v);
          v = _mm_add_pd (v, _mm_and_pd (a, 
                _mm_cmp_pd (a, cast128d(zero), _CMP_GT_OQ)));
          _mm_storeu_pd (vh+j-1, v);
          j += 2;
        }
        if (j<=N_hidden)
        { net_value a = vh[j-1];
          net_value v = 
            prec_log (1 + prec_exp(-prec_fabs(a)));  /* avoid overflow */
          if (a>0) v += a;
          vh[j-1] = v;
        }
      }
#     elif FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
      { __m128d zero = _mm_setzero_pd();
        __m128d one = _mm_set1_pd(1.0);
        __m128d mask = _mm_castsi128_pd (_mm_set1_epi64x ((long long)1<<63));
        j = 1;
        while (j<N_hidden)
        { __m128d a = _mm_loadu_pd(vh+j-1);
          __m128d v = _mm_or_pd(a,mask);  /* compute -fabs(a) */
          v = sleef_expd2(v);
          v = _mm_add_pd(one,v);
          v = sleef_logd2(v);
          v = _mm_add_pd (v, _mm_and_pd (a, _mm_cmpgt_pd (a, zero)));
          _mm_storeu_pd (vh+j-1, v);
          j += 2;
        }
        if (j<=N_hidden)
        { net_value a = vh[j-1];
          net_value v = 
            prec_log (1 + prec_exp(-prec_fabs(a)));  /* avoid overflow */
          if (a>0) v += a;
          vh[j-1] = v;
        }
      }
#     elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
      { __m256 zero = _mm256_setzero_ps();
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 mask = _mm256_castsi256_ps (_mm256_set1_epi32(1<<31));
        j = 7;
        while (j<N_hidden)
        { __m256 a = _mm256_loadu_ps(vh+j-7);
          __m256 v = _mm256_or_ps(a,mask);  /* compute -fabs(a) */
          v = sleef_expf8(v);
          v = _mm256_add_ps(one,v);
          v = sleef_logf8(v);
          v = _mm256_add_ps (v, 
                _mm256_and_ps (a, _mm256_cmp_ps (a, zero, _CMP_GT_OQ)));
          _mm256_storeu_ps (vh+j-7, v);
          j += 8;
        }
        j -= 4;
        if (j<N_hidden)
        { __m128 a = _mm_loadu_ps(vh+j-3);
          __m128 v = _mm_or_ps(a,cast128f(mask));  /* compute -fabs(a) */
          v = sleef_expf4(v);
          v = _mm_add_ps(cast128f(one),v);
          v = sleef_logf4(v);
          v = _mm_add_ps (v, _mm_and_ps (a, _mm_cmpgt_ps (a, cast128f(zero))));
          _mm_storeu_ps (vh+j-3, v);
          j += 4;
        }
        j -= 2;
        if (j<N_hidden)
        { __m128 a = _mm_loadl_pi (cast128f(zero), (__m64 *)(vh+j-1));
          __m128 v = _mm_or_ps(a,cast128f(mask));
          v = sleef_expf4(v);
          v = _mm_add_ps(cast128f(one),v);
          v = sleef_logf4(v);
          v = _mm_add_ps (v, _mm_and_ps (a, _mm_cmpgt_ps (a, cast128f(zero))));
          _mm_storel_pi ((__m64 *)(vh+j-1), v);
          j += 2;
        }
        if (j<=N_hidden)
        { net_value a = vh[j-1];
          net_value v = 
            prec_log (1 + prec_exp(-prec_fabs(a)));  /* avoid overflow */
          if (a>0) v += a;
          vh[j-1] = v;
        }
      }
#     elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
      { __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        __m128 mask = _mm_castsi128_ps (_mm_set1_epi32(1<<31));
        j = 3;
        while (j<N_hidden)
        { __m128 a = _mm_loadu_ps(vh+j-3);
          __m128 v = _mm_or_ps(a,mask);  /* compute -fabs(a) */
          v = sleef_expf4(v);
          v = _mm_add_ps(one,v);
          v = sleef_logf4(v);
          v = _mm_add_ps (v, _mm_and_ps (a, _mm_cmpgt_ps (a, zero)));
          _mm_storeu_ps (vh+j-3, v);
          j += 4;
        }
        j -= 2;
        if (j<N_hidden)
        { __m128 a = _mm_loadl_pi (zero, (__m64 *)(vh+j-1));
          __m128 v = _mm_or_ps(a,mask);
          v = sleef_expf4(v);
          v = _mm_add_ps(one,v);
          v = sleef_logf4(v);
          v = _mm_add_ps (v, _mm_and_ps (a, _mm_cmpgt_ps (a, zero)));
          _mm_storel_pi ((__m64 *)(vh+j-1), v);
          j += 2;
        }
        if (j<=N_hidden)
        { net_value a = vh[j-1];
          net_value v = 
            prec_log (1 + prec_exp(-prec_fabs(a)));  /* avoid overflow */
          if (a>0) v += a;
          vh[j-1] = v;
        }
      }
#     else
      { for (j = 0; j<N_hidden; j++)
        { net_value a = vh[j];
          net_value v = 
            prec_log (1 + prec_exp(-prec_fabs(a)));  /* avoid overflow */
          if (a>0) v += a;
          vh[j] = v;
        }
      }
#     endif
    }
    else /* identity */
    { /* nothing to do */
    }

    if (CHECK_NAN)
    { for (j = 0; j<N_hidden; j++)
      { if (isnan(vh[j])) abort();
      }
    }
  }

  /* Compute values for the outputs. */

  if (a->has_bo)
  { if (a->bias_config[a->N_layers])
    { bias_values_config (v->o, a->N_outputs, w->bo, a->bias_config[l]);
    }
    else
    { bias_values (v->o, a->N_outputs, w->bo);
    }
  }
  else
  { memset (v->o, 0, a->N_outputs * sizeof *v->o);
  }

  if (a->has_io)
  { if (a->input_config[a->N_layers])
    { add_connections_config (v->o, a->N_outputs, v->i, w->io,
        a->has_ti ? w->ti : 0, a->input_config[a->N_layers]);
    }
    else
    { add_connections (v->o, a->N_outputs, v->i, a->N_inputs,
                    w->io, a->has_ti ? w->ti : 0, 
                    flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0, 1,
                    sparse);
    }
  }

  for (l = 0; l<a->N_layers; l++)
  { if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { add_connections_config (v->o, a->N_outputs, v->h[l], w->ho[l], 
                         a->has_th[l] ? w->th[l] : 0, a->hidden_config[k]);
      }
      else
      { add_connections (v->o, a->N_outputs, v->h[l], a->N_hidden[l], w->ho[l],
                         a->has_th[l] ? w->th[l] : 0, (unsigned short *) 0, 0,
                         0);
      }
    }
  }
}


/* SET UNIT VALUES TO BIASES. */

static void bias_values
( net_value *restrict v,	/* Array of unit values to set */
  int n,			/* Number of units */
  net_param const* b		/* Biases */
)
{ 
  int j;
  for (j = 0; j<n; j++)
  { v[j] = b[j];
  }

  if (CHECK_NAN)  /* check for NaN */
  { for (j = 0; j<n; j++)
    { if (isnan(v[j])) abort();
    }
  }
}


/* SET UNIT VALUES TO BIASES WHEN THERE IS A CONFIGURATON. */

static void bias_values_config
( net_value *restrict v,	/* Array of unit values to set */
  int n,			/* Number of units */
  net_param const* b,		/* Biases */
  net_config const* cf		/* Configuration for biases */
)
{ 
  net_connection *cn;
  int c, j, k;

  memset (v, 0, n * sizeof *v);

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w;
    for (c = 0; (k = cn[c].w) >= 0; c++)
    { j = cn[c].d;
      v[j+0] += b[k+0];
      v[j+1] += b[k+1];
      v[j+2] += b[k+2];
      v[j+3] += b[k+3];
    }
    cn = cf->quad_s_4d_4w_2;
    for (c = 0; (k = cn[c].w) >= 0; c+=2)
    { net_value b0 = b[k+0];
      net_value b1 = b[k+1];
      net_value b2 = b[k+2];
      net_value b3 = b[k+3];
      j = cn[c].d;
      v[j+0] += b0;
      v[j+1] += b1;
      v[j+2] += b2;
      v[j+3] += b3;
      j = cn[c+1].d;
      v[j+0] += b0;
      v[j+1] += b1;
      v[j+2] += b2;
      v[j+3] += b3;
    }
  }

  cn = CONFIG_ORIGINAL ? cf->conn : cf->single;
  for (c = 0; (k = cn[c].w) >= 0; c++)
  { j = cn[c].d;
    v[j] += b[k];
  }

  if (CHECK_NAN)  /* check for NaN */
  { for (j = 0; j<n; j++)
    { if (isnan(v[j])) abort();
    }
  }
}


/* ADD CONTRIBUTION FROM ONE GROUP OF CONNECTIONS.  Adds the weighted input
   due to connections from one source layer to the current unit values for
   the destination layer. */

#define ADD_CONNECTIONS(offset,omit,sprs) \
do \
{ int i, j; \
  net_param o; \
  if (nd==1) \
  { net_value sv = 0; \
    for (i = 0; i<ns; i++) \
    { o = (offset); \
      if (!(omit)) \
      { sv += (v[i] + o) * *w; \
        w += 1; \
      } \
    } \
    *s += sv; \
  } \
  else if (sprs) \
  { for (i = 0; i<ns; i++) \
    { o = (offset); \
      if (omit) continue; \
      net_value tv = v[i] + o; \
      if (tv!=0)  \
      { for (j = 0; j<nd; j++) \
        { s[j] += w[j] * tv; \
        } \
      } \
      w += nd; \
    } \
  } \
  else \
  { for (i = 0; i<ns; i++) \
    { o = (offset); \
      if (omit) continue; \
      net_value tv = v[i] + o; \
      for (j = 0; j<nd; j++) \
      { s[j] += w[j] * tv; \
      } \
      w += nd; \
    } \
  } \
} while (0)

#if FP64 && USE_SIMD_INTRINSICS && __AVX__

#define ADD_CONNECTIONS00(one_more,done,sprs) \
do \
{ int i, j; \
  if (nd==1) \
  { __m256d SV = _mm256_setzero_pd(); \
    i = 3; \
    while (i<ns) \
    { SV = FMA256_pd (_mm256_loadu_pd(v+i-3), _mm256_loadu_pd(w+i-3), SV); \
      i += 4; \
    } \
    __m128d S; \
    S = _mm_add_pd (cast128d(SV), _mm256_extractf128_pd(SV,1)); \
    i -= 2; \
    if (i<ns) \
    { S = FMA_pd (_mm_loadu_pd(v+i-1), _mm_loadu_pd(w+i-1), S); \
      i += 2; \
    } \
    S = _mm_hadd_pd(S,S); \
    if (i<=ns) \
    { S = FMA_sd (_mm_load_sd(v+i-1), _mm_load_sd(w+i-1), S); \
    } \
    S = _mm_add_sd (_mm_load_sd(s), S); \
    _mm_store_sd (s, S); \
  } \
  else \
  { __m256d TV, TV2; \
    __m128d Z128d = _mm_setzero_pd(); \
    i = 0; \
    for (;;) \
    { for (;;) \
      { if (i==ns) goto done; \
        TV = _mm256_broadcast_sd (v+i); \
        if (!sprs || _mm_ucomineq_sd (cast128d(TV), Z128d)) \
        { break; \
        } \
        i += 1; \
        w += nd; \
      } \
      net_param const*w2 = w+nd; \
      i += 1; \
      for (;;) \
      { if (i==ns) goto one_more; \
        TV2 = _mm256_broadcast_sd (v+i); \
        if (!sprs || _mm_ucomineq_sd (cast128d(TV2), Z128d)) \
        { break; \
        } \
        i += 1; \
        w2 += nd; \
      } \
      j = 3; \
      while (j<nd) \
      { __m256d S = _mm256_loadu_pd(s+j-3); \
        S = FMA256_pd (TV, _mm256_loadu_pd(w+j-3), S); \
        S = FMA256_pd (TV2, _mm256_loadu_pd(w2+j-3), S); \
        _mm256_storeu_pd (s+j-3, S); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { __m128d S = _mm_loadu_pd(s+j-1); \
        S = FMA_pd (cast128d(TV), _mm_loadu_pd(w+j-1), S); \
        S = FMA_pd (cast128d(TV2), _mm_loadu_pd(w2+j-1), S); \
        _mm_storeu_pd (s+j-1, S); \
        j += 2; \
      } \
      if (j<=nd) \
      { __m128d S = _mm_load_sd(s+j-1); \
        S = FMA_sd (cast128d(TV), _mm_load_sd(w+j-1), S); \
        S = FMA_sd (cast128d(TV2), _mm_load_sd(w2+j-1), S); \
        _mm_store_sd (s+j-1, S); \
      } \
      i += 1; \
      w = w2+nd; \
    } \
  one_more: \
    j = 3; \
    while (j<nd) \
    { __m256d S = _mm256_loadu_pd(s+j-3); \
      S = FMA256_pd (TV, _mm256_loadu_pd(w+j-3), S); \
      _mm256_storeu_pd (s+j-3, S); \
      j += 4; \
    } \
    j -= 2; \
    if (j<nd) \
    { __m128d S = _mm_loadu_pd(s+j-1); \
      S = FMA_pd (cast128d(TV), _mm_loadu_pd(w+j-1), S); \
      _mm_storeu_pd (s+j-1, S); \
      j += 2; \
    } \
    if (j<=nd) \
    { __m128d S = _mm_load_sd(s+j-1); \
      S = FMA_sd (cast128d(TV), _mm_load_sd(w+j-1), S); \
      _mm_store_sd (s+j-1, S); \
    } \
  done: ; \
  } \
} while (0)

#elif FP64 && USE_SIMD_INTRINSICS && __SSE3__

#define ADD_CONNECTIONS00(one_more,done,sprs) \
do \
{ int i, j; \
  if (nd==1) \
  { __m128d S = _mm_setzero_pd(); \
    i = 1; \
    while (i<ns) \
    { S = FMA_pd (_mm_loadu_pd(v+i-1), _mm_loadu_pd(w+i-1), S); \
      i += 2; \
    } \
    S = _mm_hadd_pd(S,S); \
    if (i<=ns) \
    { S = FMA_sd (_mm_load_sd(v+i-1), _mm_load_sd(w+i-1), S); \
    } \
    S = _mm_add_sd (_mm_load_sd(s), S); \
    _mm_store_sd (s, S); \
  } \
  else \
  { __m128d TV, TV2; \
    __m128d Z128d = _mm_setzero_pd(); \
    i = 0; \
    for (;;) \
    { for (;;) \
      { if (i==ns) goto done; \
        TV = _mm_set1_pd (*(v+i)); \
        if (!sprs || _mm_ucomineq_sd (TV, Z128d)) \
        { break; \
        } \
        i += 1; \
        w += nd; \
      } \
      net_param const*w2 = w+nd; \
      i += 1; \
      for (;;) \
      { if (i==ns) goto one_more; \
        TV2 = _mm_set1_pd (*(v+i)); \
        if (!sprs || _mm_ucomineq_sd (TV2, Z128d)) \
        { break; \
        } \
        i += 1; \
        w2 += nd; \
      } \
      j = 1; \
      while (j<nd) \
      { __m128d S = _mm_loadu_pd(s+j-1); \
        S = FMA_pd (TV, _mm_loadu_pd(w+j-1), S); \
        S = FMA_pd (TV2, _mm_loadu_pd(w2+j-1), S); \
        _mm_storeu_pd (s+j-1, S); \
        j += 2; \
      } \
      if (j<=nd) \
      { __m128d S = _mm_load_sd(s+j-1); \
        S = FMA_sd (TV, _mm_load_sd(w+j-1), S); \
        S = FMA_sd (TV2, _mm_load_sd(w2+j-1), S); \
        _mm_store_sd (s+j-1, S); \
      } \
      i += 1; \
      w = w2+nd; \
    } \
  one_more: \
    j = 1; \
    while (j<nd) \
    { __m128d S = _mm_loadu_pd(s+j-1); \
      S = FMA_pd (TV, _mm_loadu_pd(w+j-1), S); \
      _mm_storeu_pd (s+j-1, S); \
      j += 2; \
    } \
    if (j<=nd) \
    { __m128d S = _mm_load_sd(s+j-1); \
      S = FMA_sd (TV, _mm_load_sd(w+j-1), S); \
      _mm_store_sd (s+j-1, S); \
    } \
  done: ; \
  } \
} while (0)

#elif FP32 && USE_SIMD_INTRINSICS && __AVX__

#define ADD_CONNECTIONS00(one_more,done,sprs) \
do \
{ int i, j; \
  if (nd==1) \
  { __m256 SV256 = _mm256_setzero_ps(); \
    i = 7; \
    while (i<ns) \
    { SV256 = FMA256_ps(_mm256_loadu_ps(v+i-7),_mm256_loadu_ps(w+i-7),SV256); \
      i += 8; \
    } \
    __m128 SV = _mm_add_ps (cast128f(SV256), _mm256_extractf128_ps(SV256,1)); \
    i -= 4; \
    if (i<ns) \
    { SV = FMA_ps (_mm_loadu_ps(v+i-3), _mm_loadu_ps(w+i-3), SV); \
      i += 4; \
    } \
    __m128 S; \
    S = _mm_add_ps (SV, _mm_movehl_ps(SV,SV)); \
    i -= 2; \
    if (i<ns) \
    { __m128 Z = _mm_setzero_ps(); \
      S = FMA_ps (_mm_loadl_pi (Z, (__m64 *)(v+i-1)), \
                  _mm_loadl_pi (Z, (__m64 *)(w+i-1)), S); \
      i += 2; \
    } \
    S = _mm_hadd_ps(S,S); \
    if (i<=ns) \
    { S = FMA_ss (_mm_load_ss(v+i-1), _mm_load_ss(w+i-1), S); \
    } \
    S = _mm_add_ss (_mm_load_ss(s), S); \
    _mm_store_ss (s, S); \
  } \
  else \
  { __m128 Z = _mm_setzero_ps(); \
    __m256 TV, TV2; \
    i = 0; \
    for (;;) \
    { for (;;) \
      { if (i==ns) goto done; \
        TV = _mm256_set1_ps (*(v+i)); \
        if (!sprs || _mm_ucomineq_ss (cast128f(TV), Z)) \
        { break; \
        } \
        i += 1; \
        w += nd; \
      } \
      net_param const*w2 = w+nd; \
      i += 1; \
      for (;;) \
      { if (i==ns) goto one_more; \
        TV2 = _mm256_set1_ps (*(v+i)); \
        if (!sprs || _mm_ucomineq_ss (cast128f(TV2), Z)) \
        { break; \
        } \
        i += 1; \
        w2 += nd; \
      } \
      j = 7; \
      while (j<nd) \
      { __m256 S = _mm256_loadu_ps(s+j-7); \
        S = FMA256_ps (TV, _mm256_loadu_ps(w+j-7), S); \
        S = FMA256_ps (TV2, _mm256_loadu_ps(w2+j-7), S); \
        _mm256_storeu_ps (s+j-7, S); \
        j += 8; \
      } \
      j -= 4; \
      if (j<nd) \
      { __m128 S = _mm_loadu_ps(s+j-3); \
        S = FMA_ps (cast128f(TV), _mm_loadu_ps(w+j-3), S); \
        S = FMA_ps (cast128f(TV2), _mm_loadu_ps(w2+j-3), S); \
        _mm_storeu_ps (s+j-3, S); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { __m128 S = _mm_loadl_pi (Z, (__m64 *)(s+j-1)); \
        S = FMA_ps (cast128f(TV), _mm_loadl_pi(Z,(__m64 *)(w+j-1)), S); \
        S = FMA_ps (cast128f(TV2), _mm_loadl_pi(Z,(__m64 *)(w2+j-1)), S); \
        _mm_storel_pi ((__m64 *)(s+j-1), S); \
        j += 2; \
      } \
      if (j<=nd) \
      { __m128 S = _mm_load_ss(s+j-1); \
        S = FMA_ss (cast128f(TV), _mm_load_ss(w+j-1), S); \
        S = FMA_ss (cast128f(TV2), _mm_load_ss(w2+j-1), S); \
        _mm_store_ss (s+j-1, S); \
      } \
      i += 1; \
      w = w2+nd; \
    } \
  one_more: \
    j = 7; \
    while (j<nd) \
    { __m256 S = _mm256_loadu_ps(s+j-7); \
      S = FMA256_ps (TV, _mm256_loadu_ps(w+j-7), S); \
      _mm256_storeu_ps (s+j-7, S); \
      j += 8; \
    } \
    j -= 4; \
    if (j<nd) \
    { __m128 S = _mm_loadu_ps(s+j-3); \
      S = FMA_ps (cast128f(TV), _mm_loadu_ps(w+j-3), S); \
      _mm_storeu_ps (s+j-3, S); \
      j += 4; \
    } \
    j -= 2; \
    if (j<nd) \
    { __m128 S = _mm_loadl_pi (Z, (__m64 *)(s+j-1)); \
      S = FMA_ps (cast128f(TV), _mm_loadl_pi(Z,(__m64 *)(w+j-1)), S); \
      _mm_storel_pi ((__m64 *)(s+j-1), S); \
      j += 2; \
    } \
    if (j<=nd) \
    { __m128 S = _mm_load_ss(s+j-1); \
      S = FMA_ss (cast128f(TV), _mm_load_ss(w+j-1), S); \
      _mm_store_ss (s+j-1, S); \
    } \
  done: ; \
  } \
} while (0)

#elif FP32 && USE_SIMD_INTRINSICS && __SSE3__

#define ADD_CONNECTIONS00(one_more,done,sprs) \
do \
{ int i, j; \
  __m128 Z = _mm_setzero_ps(); \
  if (nd==1) \
  { __m128 SV = Z; \
    i = 7; \
    while (i<ns) \
    { SV = FMA_ps (_mm_loadu_ps(v+i-7), _mm_loadu_ps(w+i-7), SV); \
      SV = FMA_ps (_mm_loadu_ps(v+i-3), _mm_loadu_ps(w+i-3), SV); \
      i += 8; \
    } \
    i -= 4; \
    if (i<ns) \
    { SV = FMA_ps (_mm_loadu_ps(v+i-3), _mm_loadu_ps(w+i-3), SV); \
      i += 4; \
    } \
    __m128 S; \
    S = _mm_add_ps (SV, _mm_movehl_ps(SV,SV)); \
    i -= 2; \
    if (i<ns) \
    { S = FMA_ps (_mm_loadl_pi (Z, (__m64 *)(v+i-1)), \
                  _mm_loadl_pi (Z, (__m64 *)(w+i-1)), S); \
      i += 2; \
    } \
    S = _mm_hadd_ps(S,S); \
    if (i<=ns) \
    { S = FMA_ss (_mm_load_ss(v+i-1), _mm_load_ss(w+i-1), S); \
    } \
    S = _mm_add_ss (_mm_load_ss(s), S); \
    _mm_store_ss (s, S); \
  } \
  else \
  { __m128 TV, TV2; \
    i = 0; \
    for (;;) \
    { for (;;) \
      { if (i==ns) goto done; \
        TV = _mm_set1_ps (*(v+i)); \
        if (!sprs || _mm_ucomineq_ss (TV, Z)) \
        { break; \
        } \
        i += 1; \
        w += nd; \
      } \
      net_param const*w2 = w+nd; \
      i += 1; \
      for (;;) \
      { if (i==ns) goto one_more; \
        TV2 = _mm_set1_ps (*(v+i)); \
        if (!sprs || _mm_ucomineq_ss (TV2, Z)) \
        { break; \
        } \
        i += 1; \
        w2 += nd; \
      } \
      j = 7; \
      while (j<nd) \
      { __m128 S = _mm_loadu_ps(s+j-7); \
        S = FMA_ps (TV, _mm_loadu_ps(w+j-7), S); \
        S = FMA_ps (TV2, _mm_loadu_ps(w2+j-7), S); \
        _mm_storeu_ps (s+j-7, S); \
        S = _mm_loadu_ps(s+j-3); \
        S = FMA_ps (TV, _mm_loadu_ps(w+j-3), S); \
        S = FMA_ps (TV2, _mm_loadu_ps(w2+j-3), S); \
        _mm_storeu_ps (s+j-3, S); \
        j += 8; \
      } \
      j -= 4; \
      if (j<nd) \
      { __m128 S = _mm_loadu_ps(s+j-3); \
        S = FMA_ps (TV, _mm_loadu_ps(w+j-3), S); \
        S = FMA_ps (TV2, _mm_loadu_ps(w2+j-3), S); \
        _mm_storeu_ps (s+j-3, S); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { __m128 S = _mm_loadl_pi (Z, (__m64 *)(s+j-1)); \
        S = FMA_ps (TV, _mm_loadl_pi(Z,(__m64 *)(w+j-1)), S); \
        S = FMA_ps (TV2,_mm_loadl_pi(Z,(__m64*)(w2+j-1)), S); \
        _mm_storel_pi ((__m64 *)(s+j-1), S); \
        j += 2; \
      } \
      if (j<=nd) \
      { __m128 S = _mm_load_ss(s+j-1); \
        S = FMA_ss (TV, _mm_load_ss(w+j-1), S); \
        S = FMA_ss (TV2, _mm_load_ss(w2+j-1), S); \
        _mm_store_ss (s+j-1, S); \
      } \
      i += 1; \
      w = w2+nd; \
    } \
  one_more: \
    j = 7; \
    while (j<nd) \
    { __m128 S = _mm_loadu_ps(s+j-7); \
      S = FMA_ps (TV, _mm_loadu_ps(w+j-7), S); \
      _mm_storeu_ps (s+j-7, S); \
      S = _mm_loadu_ps(s+j-3); \
      S = FMA_ps (TV, _mm_loadu_ps(w+j-3), S); \
      _mm_storeu_ps (s+j-3, S); \
      j += 8; \
    } \
    j -= 4; \
    if (j<nd) \
    { __m128 S = _mm_loadu_ps(s+j-3); \
      S = FMA_ps (TV, _mm_loadu_ps(w+j-3), S); \
      _mm_storeu_ps (s+j-3, S); \
      j += 4; \
    } \
    j -= 2; \
    if (j<nd) \
    { __m128 S = _mm_loadl_pi (Z, (__m64 *)(s+j-1)); \
      S = FMA_ps (TV, _mm_loadl_pi (Z, (__m64 *)(w+j-1)), S); \
      _mm_storel_pi ((__m64 *)(s+j-1), S); \
      j += 2; \
    } \
    if (j<=nd) \
    { __m128 S = _mm_load_ss(s+j-1); \
      S = FMA_ss (TV, _mm_load_ss(w+j-1), S); \
      _mm_store_ss (s+j-1, S); \
    } \
  done: ; \
  } \
} while (0)

#else

#define ADD_CONNECTIONS00(lab1,lab2,sprs) ADD_CONNECTIONS(0,0,sprs)

#endif

static void add_connections
( net_value *restrict s,  /* Summed input for destination units to add to */
  int nd,		  /* Number of destination units */
  net_value const* v,     /* Values for source units */
  int ns,		  /* Number of source units */
  net_param const* w,     /* Connection weights */
  net_param const* off,   /* Offsets to add to source unit values */
  unsigned short const* omit, /* Omit flags, null if not present/relevant */
  int ob,		  /* Bit to look at in omit flags */
  int sparse		  /* Are input values sparse? */
)
{
  if (sparse && off==0)
  { if (omit==0)
    { ADD_CONNECTIONS00(one_more1,done1,1);
    }
    else
    { ADD_CONNECTIONS(0,(*omit++)&ob,1);
    }
  }
  else
  { if (omit==0)
    { if (off==0)
      { ADD_CONNECTIONS00(one_more2,done2,0);
      }
      else
      { ADD_CONNECTIONS(*off++,0,0);
      }
    }
    else
    { if (off==0)
      { ADD_CONNECTIONS(0,(*omit++)&ob,0);
      }
      else
      { ADD_CONNECTIONS(*off++,(*omit++)&ob,0);
      }
    }
  }

  if (CHECK_NAN)  /* check for NaN */
  { int j;
    for (j = 0; j<nd; j++)
    { if (isnan(s[j])) abort();
    }
  }
}


/* ADD CONTRIBUTION FROM ONE GROUP OF CONNECTIONS WITH CONFIGURATION FROM FILE.
   Adds the weighted input due to connections from one source layer to the 
   current unit values for the destination layer. */

static void add_connections_config
( net_value *restrict s,  /* Summed input for destination units to add to */
  int nd,		  /* Number of destination units, for debug check only*/
  net_value const* v,     /* Values for source units */
  net_param const* w,     /* Connection weights */
  net_param const* off,   /* Offsets to add to source unit values */
  net_config const* cf    /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w;
#   if FP64 && USE_SIMD_INTRINSICS && __AVX__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256d VOI = _mm256_set1_pd (v[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (s+j, FMA256_pd (VOI, _mm256_loadu_pd(w+k),
                                                       _mm256_loadu_pd(s+j)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256d VOI = _mm256_set1_pd (v[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (s+j, FMA256_pd (VOI, _mm256_loadu_pd(w+k),
                                                       _mm256_loadu_pd(s+j)));
        }
      }
    }
#   elif FP64 && USE_SIMD_INTRINSICS && __SSE2__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m128d VOI = _mm_set1_pd (v[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_pd (s+j, FMA_pd(VOI, _mm_loadu_pd(w+k), 
                                          _mm_loadu_pd(s+j)));
          _mm_storeu_pd (s+j+2, FMA_pd(VOI, _mm_loadu_pd(w+k+2), 
                                            _mm_loadu_pd(s+j+2)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m128d VOI = _mm_set1_pd (v[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_pd (s+j, FMA_pd(VOI, _mm_loadu_pd(w+k), 
                                          _mm_loadu_pd(s+j)));
          _mm_storeu_pd (s+j+2, FMA_pd(VOI, _mm_loadu_pd(w+k+2),
                                            _mm_loadu_pd(s+j+2)));
        }
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __SSE2__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m128 VOI = _mm_set1_ps (v[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_ps (s+j, FMA_ps (VOI, _mm_loadu_ps(w+k),
                                           _mm_loadu_ps(s+j)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m128 VOI = _mm_set1_ps (v[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_ps (s+j, FMA_ps (VOI, _mm_loadu_ps(w+k),
                                           _mm_loadu_ps(s+j)));
        }
      }
    }
#   else
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { net_value voi = v[cn[c].s] + off[cn[c].s];
          j = cn[c].d;
          s[j+0] += voi * w[k+0];
          s[j+1] += voi * w[k+1];
          s[j+2] += voi * w[k+2];
          s[j+3] += voi * w[k+3];
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { net_value vi = v[cn[c].s];
          j = cn[c].d; 
          s[j+0] += vi * w[k+0];
          s[j+1] += vi * w[k+1];
          s[j+2] += vi * w[k+2];
          s[j+3] += vi * w[k+3];
        }
      }
    }
#   endif
    cn = cf->quad_s_4d_4w_2;
#   if FP64 && USE_SIMD_INTRINSICS && __AVX__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m256d WK = _mm256_loadu_pd(w+k);
          __m256d VOI;
          VOI = _mm256_set1_pd (v[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (s+j, FMA256_pd (VOI, WK, _mm256_loadu_pd(s+j)));
          VOI = _mm256_set1_pd (v[cn[c+1].s] + off[cn[c+1].s]);
          j = cn[c+1].d;
          _mm256_storeu_pd (s+j, FMA256_pd (VOI, WK, _mm256_loadu_pd(s+j)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m256d WK = _mm256_loadu_pd(w+k);
          __m256d VOI;
          VOI = _mm256_set1_pd (v[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (s+j, FMA256_pd (VOI, WK, _mm256_loadu_pd(s+j)));
          VOI = _mm256_set1_pd (v[cn[c+1].s]);
          j = cn[c+1].d;
          _mm256_storeu_pd (s+j, FMA256_pd (VOI, WK, _mm256_loadu_pd(s+j)));
        }
      }
    }
#   elif FP64 && USE_SIMD_INTRINSICS && __SSE2__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m128d WK = _mm_loadu_pd(w+k), WK2 = _mm_loadu_pd(w+k+2);
          __m128d VOI;
          VOI = _mm_set1_pd (v[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_pd (s+j, FMA_pd(VOI, WK, _mm_loadu_pd(s+j)));
          _mm_storeu_pd (s+j+2, FMA_pd(VOI, WK2, _mm_loadu_pd(s+j+2)));
          VOI = _mm_set1_pd (v[cn[c+1].s] + off[cn[c+1].s]);
          j = cn[c+1].d;
          _mm_storeu_pd (s+j, FMA_pd(VOI, WK, _mm_loadu_pd(s+j)));
          _mm_storeu_pd (s+j+2, FMA_pd(VOI, WK2, _mm_loadu_pd(s+j+2)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m128d WK = _mm_loadu_pd(w+k), WK2 = _mm_loadu_pd(w+k+2);
          __m128d VOI;
          VOI = _mm_set1_pd (v[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_pd (s+j, FMA_pd(VOI, WK, _mm_loadu_pd(s+j)));
          _mm_storeu_pd (s+j+2, FMA_pd(VOI, WK2, _mm_loadu_pd(s+j+2)));
          VOI = _mm_set1_pd (v[cn[c+1].s]);
          j = cn[c+1].d;
          _mm_storeu_pd (s+j, FMA_pd(VOI, WK, _mm_loadu_pd(s+j)));
          _mm_storeu_pd (s+j+2, FMA_pd(VOI, WK2, _mm_loadu_pd(s+j+2)));
        }
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __SSE2__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m128 WK = _mm_loadu_ps(w+k);
          __m128 VOI;
          VOI = _mm_set1_ps (v[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_ps (s+j, FMA_ps (VOI, WK, _mm_loadu_ps(s+j)));
          VOI = _mm_set1_ps (v[cn[c+1].s] + off[cn[c+1].s]);
          j = cn[c+1].d;
          _mm_storeu_ps (s+j, FMA_ps (VOI, WK, _mm_loadu_ps(s+j)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m128 WK = _mm_loadu_ps(w+k);
          __m128 VOI;
          VOI = _mm_set1_ps (v[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_ps (s+j, FMA_ps (VOI, WK, _mm_loadu_ps(s+j)));
          VOI = _mm_set1_ps (v[cn[c+1].s]);
          j = cn[c+1].d;
          _mm_storeu_ps (s+j, FMA_ps (VOI, WK, _mm_loadu_ps(s+j)));
        }
      }
    }
#   else
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { net_value w0 = w[k+0];
          net_value w1 = w[k+1];
          net_value w2 = w[k+2];
          net_value w3 = w[k+3];
          net_value voi = v[cn[c].s] + off[cn[c].s];
          j = cn[c].d;
          s[j+0] += voi * w0;
          s[j+1] += voi * w1;
          s[j+2] += voi * w2;
          s[j+3] += voi * w3;
          voi = v[cn[c+1].s] + off[cn[c+1].s];
          j = cn[c+1].d;
          s[j+0] += voi * w0;
          s[j+1] += voi * w1;
          s[j+2] += voi * w2;
          s[j+3] += voi * w3;
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { net_value w0 = w[k+0];
          net_value w1 = w[k+1];
          net_value w2 = w[k+2];
          net_value w3 = w[k+3];
          net_value vi = v[cn[c].s];
          j = cn[c].d; 
          s[j+0] += vi * w0;
          s[j+1] += vi * w1;
          s[j+2] += vi * w2;
          s[j+3] += vi * w3;
          vi = v[cn[c+1].s];
          j = cn[c+1].d; 
          s[j+0] += vi * w0;
          s[j+1] += vi * w1;
          s[j+2] += vi * w2;
          s[j+3] += vi * w3;
        }
      }
    }
#   endif
  }

  if (CONFIG_SINGLE4)
  { 
    cn = cf->single4_s;
    if (off)
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value voi = v[cn[c].s] + off[cn[c].s];
        j = cn[c].d;
        s[j] += voi * w[k];
        j = cn[c+1].d; k = cn[c+1].w; 
        s[j] += voi * w[k];
        j = cn[c+2].d; k = cn[c+2].w; 
        s[j] += voi * w[k];
        j = cn[c+3].d; k = cn[c+3].w;
        s[j] += voi * w[k];
      }
    }
    else
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value vi = v[cn[c].s];
        j = cn[c].d; 
        s[j] += vi * w[k];
        j = cn[c+1].d; k = cn[c+1].w; 
        s[j] += vi * w[k];
        j = cn[c+2].d; k = cn[c+2].w; 
        s[j] += vi * w[k];
        j = cn[c+3].d; k = cn[c+3].w;
        s[j] += vi * w[k];
      }
    }

    cn = cf->single4_d;
    if (off)
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { j = cn[c].d;
        net_value sj = s[j];
        i = cn[c].s; 
        sj += (v[i]+off[i]) * w[k];
        i = cn[c+1].s; k = cn[c+1].w; 
        sj += (v[i]+off[i]) * w[k];
        i = cn[c+2].s; k = cn[c+2].w; 
        sj += (v[i]+off[i]) * w[k];
        i = cn[c+3].s; k = cn[c+3].w;
        sj += (v[i]+off[i]) * w[k];
        s[j] = sj;
      }
    }
    else
    { 
#     if FP64 && USE_SIMD_INTRINSICS && __AVX2__ && 0 /* disabled: slower */
      { __m256i MASK16 = _mm256_set1_epi64x (0xffff);
        for (c = 0; ; c+=4)
        { __m256i K = _mm256_loadu_si256 ((__m256i *) &cn[c]);
          int64_t t = _mm_cvtsi128_si64 (_mm256_castsi256_si128(K));
          if (t<0)
          { break; 
          }
          __m256i I = _mm256_and_si256 (MASK16, K);
          K = _mm256_srli_epi64 (K, 32);
          __m256d VI = _mm256_i64gather_pd (v, I, 8);
          __m256d WK = _mm256_i64gather_pd (w, K, 8);
          j = (t >> 16) & 0xffff;
          __m128d SJ = _mm_load_sd(s+j);
          __m256d P = _mm256_mul_pd (VI, WK);
          __m128d S = _mm_add_pd (cast128d(P),
                                  _mm256_extractf128_pd(P,1));
          _mm_store_sd (s+j, _mm_add_sd (SJ, _mm_hadd_pd(S,S)));
        }
      }
#     else
      { for (c = 0; (k = cn[c].w) >= 0; c+=4)
        { j = cn[c].d;
          net_value sj = s[j];
          i = cn[c].s; 
          sj += v[i] * w[k];
          i = cn[c+1].s; k = cn[c+1].w; 
          sj += v[i] * w[k];
          i = cn[c+2].s; k = cn[c+2].w; 
          sj += v[i] * w[k];
          i = cn[c+3].s; k = cn[c+3].w;
          sj += v[i] * w[k];
          s[j] = sj;
        }
      }
#     endif
    }
  }

  cn = CONFIG_ORIGINAL ? cf->conn : cf->single;
  if (off)
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { i = cn[c].s; j = cn[c].d;
      s[j] += (v[i]+off[i]) * w[k];
    }
  }
  else
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { i = cn[c].s; j = cn[c].d;
      s[j] += v[i] * w[k];
    }
  }

  if (CHECK_NAN)  /* check for NaN */
  { int j;
    for (j = 0; j<nd; j++)
    { if (isnan(s[j])) abort();
    }
  }
}


/* -------------------------- net_func_gpu --------------------------------- */

#if __CUDACC__

__device__ static void bias_values_gpu (int th, net_value *restrict, int, 
                                        net_param const*, unsigned);

__device__ static void bias_values_config_gpu (int, net_value *restrict, int, 
                                        net_param const*, net_config const*,
                                        unsigned);

__device__ static void add_connections_gpu (int, net_value *restrict, int, 
                      net_value const*, int, net_param const*, net_param const*,
                      unsigned short const*, int, int, unsigned);

__device__ static void add_connections_config_gpu (int, net_value *restrict, 
                                    net_value const*,
                                    net_param const*, net_param const*,
                                    net_config const*, unsigned);


/* EVALUATE NETWORK FUNCTION FOR GIVEN INPUTS.  The inputs are taken
   from the net_values structure passed. 

   This version uses THREADS_PER_CASE (power of two) GPU threads to do
   the computation.  It should be called from each of these threads,
   with 'th' set to 0 up to THREADS_PER_CASE-1.  If called with a
   negative 'th' (done when there are spare threads at end of training
   set), it just skips to the synchronization points.
  
   Layers are computed in sequence, using all threads, with a
   __syncthreads call after each layer's computations, so that all
   threads will correctly see all values when computing the next
   layer.  Note that ALL threads must call this function so that all
   will make the __syncthreads calls here.

   Synchronization is avoided when there is only one thread - note
   that this means that outputs for different cases will not have been
   synchronized.

   Thread 'th' is used to compute the units whose index is 'th' mod
   THREADS_PER_CASE.  Consistent use of this scheme for the various
   componenets avoids any need to synchronize threads within
   computations for a single layer.

   No synchronization is done after computing the outputs.  Without
   further synchronization, thread 'th' will have computed output
   values with index mod THREADS_PER_CASE equal to 'th', and may use
   these values without synchronization.
*/

#define A const_arch
#define PRE const_pre
#define FLGS const_flgs
#define W const_params
#define HAS_FLGS const_has_flgs

__device__ STATIC_IF_INCLUDED void net_func_gpu
( int th,		/* Thread index */
  net_values *restrict v, /* Place to get inputs and store outputs */
  int sparse,		/* Are input values sparse? */
  unsigned syncmask     /* Mask of active threads */
)
{
  net_value *vhp;
  int l, ls, j;

  /* Compute values for successive hidden layers. */

  for (l = 0; l<A.N_layers; l++)
  {
    int N_hidden = A.N_hidden[l];
    net_value *restrict vh;

    if (th<0) goto sync_layer;

    vh = fw_hidden_loc(&PRE,v,l);

    /* Find summed inputs into each hidden unit in the layer. */

    if (A.has_bh[l])
    { if (A.bias_config[l])
      { bias_values_config_gpu (th, vh, N_hidden, W.bh[l], A.bias_config[l], 
                                syncmask);
      }
      else
      { bias_values_gpu (th, vh, N_hidden, W.bh[l], syncmask);
      }
    }
    else
    { for (j = th; j < N_hidden; j+=NTH)
      { vh[j] = 0;
      }
      if (SYNC_AFTER && N_hidden % NTH != 0) __syncwarp(syncmask);
    }

    if (A.has_ih[l])
    { if (A.input_config[l])
      { add_connections_config_gpu (th, vh, v->i, W.ih[l], 
          A.has_ti ? W.ti : 0, A.input_config[l], syncmask);
      }
      else
      { add_connections_gpu (th, vh, N_hidden, v->i, A.N_inputs, 
          W.ih[l], A.has_ti ? W.ti : 0, 
          HAS_FLGS && FLGS.any_omitted[l] ? FLGS.omit : 0, 1<<(l+1), 
          sparse, syncmask);
      }
    }

    if (A.has_nsq[l])
    { for (ls = 0; ls<l; ls++)
      { int nsqi = PRE.nonseq[ls][l];
        net_value *vhs = fw_hidden_loc(&PRE,v,ls);
        if (nsqi>=0)
        { if (A.nonseq_config[nsqi])
          { add_connections_config_gpu (th, vh, vhs, W.nsq[nsqi],
              A.has_th[ls] ? W.th[ls] : 0, A.nonseq_config[nsqi], syncmask);
          }
          else
          { add_connections_gpu (th, vh, N_hidden, vhs, A.N_hidden[ls],
              W.nsq[nsqi], A.has_th[ls] ? W.th[ls] : 0,
              (unsigned short *) 0, 0, 0, syncmask);
          }
        }
      }
    }

    if (l>0 && A.has_hh[l-1])
    { if (A.hidden_config[l])
      { add_connections_config_gpu (th, vh, vhp, W.hh[l-1], 
          A.has_th[l-1] ? W.th[l-1] : 0, A.hidden_config[l], syncmask);
      }
      else
      { add_connections_gpu (th, vh, N_hidden, vhp, A.N_hidden[l-1],
          W.hh[l-1], A.has_th[l-1] ? W.th[l-1] : 0, (unsigned short *) 0, 
          0, 0, syncmask);
      }
    }

    /* Put values through hidden unit activation function. */

    if (!HAS_FLGS || FLGS.layer_type[l]==Tanh_type)
    { for (j = th; j<N_hidden; j+=NTH)
      { vh[j] = TANH (vh[j]);
      }
      if (SYNC_AFTER && N_hidden % NTH != 0) __syncwarp(syncmask);
    }
    else if (FLGS.layer_type[l]==Softplus_type)
    { for (j = th; j<N_hidden; j+=NTH)
      { net_value a = vh[j];
        net_value v = 
         prec_log ((net_value)1 + prec_exp(-prec_fabs(a)));/* avoid overflow*/
        if (a>0) v += a;
        vh[j] = v;
      }
      if (SYNC_AFTER && N_hidden % NTH != 0) __syncwarp(syncmask);
    }
    else /* identity */ 
    { /* nothing to do */
    }

    /* Synchronize threads so that up-to-date values computed for this
       layer will be seen by all threads. */

  sync_layer:
    if (NTH>1) __syncthreads();

    vhp = vh;
  }

  /* Compute values for the outputs. */

  if (th<0) 
  { return;
  }

  if (A.has_bo)
  { if (A.bias_config[A.N_layers])
    { bias_values_config_gpu (th, v->o, A.N_outputs, W.bo, A.bias_config[l],
                              syncmask);
    }
    else
    { bias_values_gpu (th, v->o, A.N_outputs, W.bo, syncmask);
    }
  }
  else
  { for (j = th; j < A.N_outputs; j+=NTH)
    { v->o[j] = 0;
    }
    if (SYNC_AFTER && A.N_outputs % NTH != 0) __syncwarp(syncmask);
  }

  if (A.has_io)
  { if (A.input_config[A.N_layers])
    { add_connections_config_gpu (th, v->o, v->i, W.io,
        A.has_ti ? W.ti : 0, A.input_config[A.N_layers], syncmask);
    }
    else
    { add_connections_gpu (th, v->o, A.N_outputs, v->i, A.N_inputs,
                    W.io, A.has_ti ? W.ti : 0, 
                    HAS_FLGS && FLGS.any_omitted[A.N_layers] ? FLGS.omit : 0, 1,
                    sparse, syncmask);
    }
  }

  for (l = 0; l<A.N_layers; l++)
  { if (A.has_ho[l])
    { vhp = fw_hidden_loc(&PRE,v,l);
      int k = 2*A.N_layers-1-l;
      if (A.hidden_config[k])
      { add_connections_config_gpu (th, v->o, vhp, W.ho[l], 
                         A.has_th[l] ? W.th[l] : 0, A.hidden_config[k],
                         syncmask);
      }
      else
      { add_connections_gpu (th, v->o, A.N_outputs, vhp, A.N_hidden[l], 
                             W.ho[l], A.has_th[l] ? W.th[l] : 0, 
                             (unsigned short *) 0, 0, 0, syncmask);
      }
    }
  }
}


/* SET UNIT VALUES TO BIASES. */

__device__ static void bias_values_gpu
( int th,			/* Thread index */
  net_value *restrict v,	/* Array of unit values to set */
  int n,			/* Number of units */
  net_param const* b,		/* Biases */
  unsigned syncmask     	/* Mask of active threads */
)
{ 
  int j;
  for (j = th; j<n; j+=NTH)
  { v[j] = b[j];
  }

  if (SYNC_AFTER && n % NTH != 0) __syncwarp(syncmask);
}


/* SET UNIT VALUES TO BIASES WHEN THERE IS A CONFIGURATON. */

__device__ static void bias_values_config_gpu
( int th,			/* Thread index */
  net_value *restrict v,	/* Array of unit values to set */
  int n,			/* Number of units */
  net_param const* b,		/* Biases */
  net_config const* cf,		/* Configuration for biases */
  unsigned syncmask     	/* Mask of active threads */
)
{ 
  net_connection *cn;
  int c, j, k, m, ix;

  for (j = th; j < n; j+=NTH)
  { v[j] = 0;
  }

  if (SYNC_AFTER && n % NTH != 0) __syncwarp(syncmask);

  if (CONFIG_OCT_GPU_S_8D_8W)
  { cn = cf->oct_s_8d_8w_dgpu;
    c = 0;
    for (m = 0; m<NTH; m++)
    { if (NTH>1) ix = (th+NTH-m)&(NTH-1);
      for (;;)
      { j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        if (NTH==1)
        { v[j] += b[k];
          v[j+1] += b[k+1];
          v[j+2] += b[k+2];
          v[j+3] += b[k+3];
          v[j+4] += b[k+4];
          v[j+5] += b[k+5];
          v[j+6] += b[k+6];
          v[j+7] += b[k+7];
        }
        else if (NTH==2)
        { v[j+ix] += b[k+ix];
          v[j+ix+2] += b[k+ix+2];
          v[j+ix+4] += b[k+ix+4];
          v[j+ix+6] += b[k+ix+6];
        }
        else if (NTH==4)
        { v[j+ix] += b[k+ix];
          v[j+ix+4] += b[k+ix+4];
        }
        else if (NTH==8)
        { v[j+ix] += b[k+ix];
        }
        else if (ix<8)
        { v[j+ix] += b[k+ix];
        }
      }
    }
    if (SYNC_AFTER) __syncwarp(syncmask);
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_dgpu;
    c = 0;
    for (m = 0; m<NTH; m++)
    { if (NTH>1) ix = (th+NTH-m)&(NTH-1);
      for (;;)
      { j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        if (NTH==1)
        { v[j+0] += b[k+0];
          v[j+1] += b[k+1];
          v[j+2] += b[k+2];
          v[j+3] += b[k+3];
        }
        else if (NTH==2)
        { v[j+ix] += b[k+ix];
          v[j+ix+2] += b[k+ix+2];
        }
        else if (NTH==4)
        { v[j+ix] += b[k+ix];
        }
        else if (ix<4)
        { v[j+ix] += b[k+ix];
        }
      }
    }
    if (SYNC_AFTER) __syncwarp(syncmask);
  }

  cn = cf->other_dgpu;
  c = cf->start_other_dgpu[th];
  for (;;)
  { j = cn[c].d; k = cn[c].w; c += 1;
    if (k<0) break;
    v[j] += b[k];
  }
  if (SYNC_AFTER) __syncwarp(syncmask);
}


/* ADD CONTRIBUTION FROM ONE GROUP OF CONNECTIONS.  Adds the weighted input
   due to connections from one source layer to the current unit values for
   the destination layer. */

#define ADD_CONNECTIONS_GPU(off,sprs) \
do \
{ int i, j; \
  if (sprs && nd>NTH) \
  { for (i = 0; i<ns; i++) \
    { net_value tv = off ? v[i] + off[i] : v[i]; \
      if (tv!=0) \
      { for (j = th; j<nd; j+=NTH) \
        { s[j] += w[j] * tv; \
        } \
        if (SYNC_AFTER && nd % NTH != 0) __syncwarp(syncmask); \
      } \
      w += nd; \
    } \
  } \
  else if (0 && !off && nd>NTH) /* DISABLE FOR NOW */ \
  { i = 3; \
    while (i<ns) \
    { net_value tv0 = v[i-3]; \
      net_value tv1 = v[i-2]; \
      net_value tv2 = v[i-1]; \
      net_value tv3 = v[i]; \
      net_param const* w1 = w+nd; \
      net_param const* w2 = w1+nd; \
      net_param const* w3 = w2+nd; \
      for (j = th; j<nd; j+=NTH) \
      { s[j] = s[j] + w[j] * tv0 + w1[j] * tv1 + w2[j] * tv2 + w3[j] * tv3; \
      } \
      if (SYNC_AFTER && nd % NTH != 0) __syncwarp(syncmask); \
      w = w3 + nd; \
      i += 4; \
    } \
    i -= 2; \
    if (i<ns) \
    { net_value tv0 = v[i-1]; \
      net_value tv1 = v[i]; \
      net_param const* w1 = w+nd; \
      for (j = th; j<nd; j+=NTH) \
      { s[j] = s[j] + w[j] * tv0 + w1[j] * tv1; \
      } \
      if (SYNC_AFTER && nd % NTH != 0) __syncwarp(syncmask); \
      w = w1 + nd; \
      i += 2; \
    } \
    if (i<=ns) \
    { net_value tv = v[i-1]; \
      for (j = th; j<nd; j+=NTH) \
      { s[j] = s[j] + w[j] * tv; \
      } \
      if (SYNC_AFTER && nd % NTH != 0) __syncwarp(syncmask); \
    } \
  } \
  else if (nd==1) \
  { if (th==0) \
    { net_value sv = 0; \
      if (off) \
      { for (i = 0; i<ns; i++) \
        { sv += (v[i] + off[i]) * *w; \
          w += 1; \
        } \
      } \
      else \
      { for (i = 0; i<ns; i++) \
        { sv += v[i] * *w; \
          w += 1; \
        } \
      } \
      s[0] += sv; \
    } \
    if (SYNC_AFTER) __syncwarp(syncmask); \
  } \
  else \
  { for (j = th; j<nd; j+=NTH) \
    { net_value sv = s[j]; \
      const net_param *wj = w+j; \
      if (off) \
      { for (i = 0; i<ns; i++) \
        { sv += (v[i] + off[i]) * *wj; \
          wj += nd; \
        } \
      } \
      else \
      { for (i = 0; i<ns; i++) \
        { sv += v[i] * *wj; \
          wj += nd; \
        } \
      } \
      s[j] = sv; \
    } \
    if (SYNC_AFTER && nd % NTH != 0) __syncwarp(syncmask); \
  } \
} while (0)

#define ADD_CONNECTIONS_OMIT_GPU(off,sprs) \
do \
{ int i, j; \
  if (sprs && nd>NTH) \
  { for (i = 0; i<ns; i++) \
    { if (omit[i]&ob) continue; \
      net_value tv = off ? v[i] + off[i] : v[i]; \
      if (tv!=0)  \
      { for (j = th; j<nd; j+=NTH) \
        { s[j] += w[j] * tv; \
        } \
        if (SYNC_AFTER && nd % NTH != 0) __syncwarp(syncmask); \
      } \
      w += nd; \
    } \
  } \
  else if (nd==1) \
  { if (th==0) \
    { net_value sv = 0; \
      for (i = 0; i<ns; i++) \
      { if (omit[i]&ob) continue; \
        sv += off ? (v[i] + off[i]) * *w : v[i] * *w; \
        w += 1; \
      } \
      s[j] += sv; \
    } \
    if (SYNC_AFTER) __syncwarp(syncmask); \
  } \
  else \
  { for (j = th; j<nd; j+=NTH) \
    { net_value sv = s[j]; \
      const net_param *wj = w+j; \
      int k = 0; \
      for (i = 0; i<ns; i++) \
      { while (k < i) \
        { if (!(omit[i]&ob)) wj += nd; \
          k += 1; \
        } \
        if (!(omit[i]&ob)) \
        { sv += off ? (v[i] + off[i]) * *wj : v[i] * *wj; \
        } \
      } \
      s[j] = sv; \
    } \
    if (SYNC_AFTER && nd % NTH != 0) __syncwarp(syncmask); \
  } \
} while (0)

__device__ static void add_connections_gpu
( int th,		  /* Thread index */
  net_value *restrict s,  /* Summed input for destination units to add to */
  int nd,		  /* Number of destination units */
  net_value const* v,     /* Values for source units */
  int ns,		  /* Number of source units */
  net_param const* w,     /* Connection weights */
  net_param const* off,   /* Offsets to add to source unit values */
  unsigned short const* omit, /* Omit flags, null if not present/relevant */
  int ob,		  /* Bit to look at in omit flags */
  int sparse,             /* Might source unit values often be zero? */
  unsigned syncmask       /* Mask of active threads */
)
{
  if (sparse && off==0)
  { if (omit==0)
    { ADD_CONNECTIONS_GPU(((char*)0),1);
    }
    else
    { ADD_CONNECTIONS_OMIT_GPU(((char*)0),1);
    }
  }
  else
  { if (omit==0)
    { if (off==0)
      { ADD_CONNECTIONS_GPU(((char*)0),0);
      }
      else
      { ADD_CONNECTIONS_GPU(off,0);
      }
    }
    else
    { if (off==0)
      { ADD_CONNECTIONS_OMIT_GPU(((char*)0),0);
      }
      else
      { ADD_CONNECTIONS_OMIT_GPU(off,0);
      }
    }
  }
}


/* ADD CONTRIBUTION FROM ONE GROUP OF CONNECTIONS WITH CONFIGURATION FROM FILE.
   Adds the weighted input due to connections from one source layer to the 
   current unit values for the destination layer. */

__device__ static void add_connections_config_gpu
( int th,		  /* Thread index */
  net_value *restrict s,  /* Summed input for destination units to add to */
  net_value const* v,     /* Values for source units */
  net_param const* w,     /* Connection weights */
  net_param const* off,   /* Offsets to add to source unit values */
  net_config const* cf,    /* Configuration for connections and weights */
  unsigned syncmask       /* Mask of active threads */
)
{
  net_connection *cn;
  int c, i, j, k, m, ix;
  net_value vi;

  if (CONFIG_OCT_GPU_S_8D_8W)
  { cn = cf->oct_s_8d_8w_dgpu;
    c = 0;
    for (m = 0; m<NTH; m++)
    { if (NTH>1) ix = (th+NTH-m)&(NTH-1);
      for (;;)
      { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        vi = v[i];
        if (off) vi += off[i];
        if (NTH==1)
        { s[j] += vi * w[k];
          s[j+1] += vi * w[k+1];
          s[j+2] += vi * w[k+2];
          s[j+3] += vi * w[k+3];
          s[j+4] += vi * w[k+4];
          s[j+5] += vi * w[k+5];
          s[j+6] += vi * w[k+6];
          s[j+7] += vi * w[k+7];
        }
        else if (NTH==2)
        { s[j+ix] += vi * w[k+ix];
          s[j+ix+2] += vi * w[k+ix+2];
          s[j+ix+4] += vi * w[k+ix+4];
          s[j+ix+6] += vi * w[k+ix+6];
        }
        else if (NTH==4)
        { s[j+ix] += vi * w[k+ix];
          s[j+ix+4] += vi * w[k+ix+4];
        }
        else if (NTH==8)
        { s[j+ix] += vi * w[k+ix];
        }
        else if (ix<8)
        { s[j+ix] += vi * w[k+ix];
        }
      }
    }
    if (SYNC_AFTER) __syncwarp(syncmask);
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_dgpu;
    c = 0;
    for (m = 0; m<NTH; m++)
    { if (NTH>1) ix = (th+NTH-m)&(NTH-1);
      for (;;)
      { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        vi = v[i];
        if (off) vi += off[i];
        if (NTH==1)
        { s[j+0] += vi * w[k+0];
          s[j+1] += vi * w[k+1];
          s[j+2] += vi * w[k+2];
          s[j+3] += vi * w[k+3];
        }
        else if (NTH==2)
        { s[j+ix] += vi * w[k+ix];
          s[j+ix+2] += vi * w[k+ix+2];
        }
        else if (NTH==4)
        { s[j+ix] += vi * w[k+ix];
        }
        else if (ix<4)
        { s[j+ix] += vi * w[k+ix];
        }
      }
    }
    if (SYNC_AFTER) __syncwarp(syncmask);
  }

  cn = cf->other_dgpu;
  c = cf->start_other_dgpu[th];
  for (;;)
  { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
    if (k<0) break;
    vi = v[i];
    if (off) vi += off[i];
    s[j] += vi * w[k];
  }
  if (SYNC_AFTER) __syncwarp(syncmask);
}

#undef A
#undef PRE
#undef FLGS
#undef W
#undef HAS_FLGS

#endif
