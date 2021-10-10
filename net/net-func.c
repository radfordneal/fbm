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

#define sqrt_2 1.4142135623730950488

HOSTDEV static void bias_values (net_value *restrict, int, net_param const*);

HOSTDEV static void bias_values_config (net_value *restrict, int, 
                                        net_param const*, net_config const*);

HOSTDEV static void add_connections (net_value *restrict, int, net_value const*,
                             int, net_param const*, net_param const*,
                             unsigned short const*, int);

HOSTDEV static void add_connections_config (net_value *restrict, 
                                    net_value const*,
                                    net_param const*, net_param const*,
                                    net_config const*);


/* EVALUATE NETWORK FUNCTION FOR GIVEN INPUTS.  The inputs are taken from
   the net_values structure passed.  When 'start' is greater than zero, the
   correct unit values for that number of hidden layers are assumed to be
   already present in the net_values structure. */

HOSTDEV void net_func 
( net_values *restrict v, /* Place to get inputs and store outputs */
  int start,		/* Number of hidden layers with known values */
  net_arch const* a,	/* Network architecture */
  net_flags const* flgs,/* Network flags, null if none */
  net_params const* w	/* Network parameters */
)
{
  int l, j;

  /* Compute values for successive hidden layers. */

  for (l = start; l<a->N_layers; l++)
  {
    int N_hidden = a->N_hidden[l];

    net_value *sh = v->s[l];

    if (a->has_bh[l])
    { if (a->bias_config[l])
      { bias_values_config (sh, N_hidden, w->bh[l], a->bias_config[l]);
      }
      else
      { bias_values (sh, N_hidden, w->bh[l]);
      }
    }
    else
    { memset (sh, 0, N_hidden * sizeof *sh);
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { add_connections_config (sh, v->i, w->ih[l], 
          a->has_ti ? w->ti : 0, a->input_config[l]);
      }
      else
      { add_connections (sh, N_hidden, v->i, a->N_inputs, 
          w->ih[l], a->has_ti ? w->ti : 0, 
          flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1));
      }
    }

    if (l>0 && a->has_hh[l-1])
    { if (a->hidden_config[l])
      { add_connections_config (sh, v->h[l-1], w->hh[l-1], 
          a->has_th[l-1] ? w->th[l-1] : 0, a->hidden_config[l]);
      }
      else
      { add_connections (sh, N_hidden, v->h[l-1], a->N_hidden[l-1],
          w->hh[l-1], a->has_th[l-1] ? w->th[l-1] : 0, (unsigned short *) 0, 0);
      }
    }

    /* Put values through hidden unit activation function. */

    net_value *vh = v->h[l];

    if (flgs==0 || flgs->layer_type[l]==Tanh_type)
    { 
#     if USE_QUICK_AND_DIRTY_TANH
#       if FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
        { __m256d one = _mm256_set1_pd(1.0);
          __m256d two = _mm256_set1_pd(2.0);
          j = 3;
          while (j<N_hidden)
          { __m256d x = _mm256_loadu_pd(sh+j-3);
            x = _mm256_add_pd(x,x);
            x = sleef_expd4(x);
            x = _mm256_sub_pd(one, _mm256_div_pd (two, _mm256_add_pd (one, x)));
            _mm256_storeu_pd (vh+j-3, x);
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { __m128d x = _mm_loadu_pd(sh+j-1);
            x = _mm_add_pd(x,x);
            x = sleef_expd2(x);
            x = _mm_sub_pd (cast128d(one), 
                   _mm_div_pd (cast128d(two), _mm_add_pd (cast128d(one), x)));
            _mm_storeu_pd (vh+j-1, x);
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (sh[j-1]);
          }
        }
#       elif FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
        { __m128d one = _mm_set1_pd(1.0);
          __m128d two = _mm_set1_pd(2.0);
          j = 1;
          while (j<N_hidden)
          { __m128d x = _mm_loadu_pd(sh+j-1);
            x = _mm_add_pd(x,x);
            x = sleef_expd2(x);
            x = _mm_sub_pd (one, _mm_div_pd (two, _mm_add_pd (one, x)));
            _mm_storeu_pd (vh+j-1, x);
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (sh[j-1]);
          }
        }
#       elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
        { __m256 one = _mm256_set1_ps(1.0f);
          __m256 two = _mm256_set1_ps(2.0f);
          j = 7;
          while (j<N_hidden)
          { __m256 x = _mm256_loadu_ps(sh+j-7);
            x = _mm256_add_ps(x,x);
            x = sleef_expf8(x);
            x = _mm256_sub_ps(one, _mm256_div_ps (two, _mm256_add_ps (one, x)));
            _mm256_storeu_ps (vh+j-7, x);
            j += 8;
          }
          j -= 4;
          while (j<N_hidden)
          { __m128 x = _mm_loadu_ps(sh+j-3);
            x = _mm_add_ps(x,x);
            x = sleef_expf4(x);
            x = _mm_sub_ps (cast128f(one), 
                    _mm_div_ps (cast128f(two), _mm_add_ps (cast128f(one), x)));
            _mm_storeu_ps (vh+j-3, x);
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { __m128 x = _mm_loadl_pi (_mm_setzero_ps(), (__m64 *)(sh+j-1));
            x = _mm_add_ps(x,x);
            x = sleef_expf4(x);
            x = _mm_sub_ps (cast128f(one), 
                    _mm_div_ps (cast128f(two), _mm_add_ps (cast128f(one), x)));
            _mm_storeu_ps (vh+j-1, x);
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (sh[j-1]);
          }
        }
#       elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
        { __m128 one = _mm_set1_ps(1.0f);
          __m128 two = _mm_set1_ps(2.0f);
          j = 3;
          while (j<N_hidden)
          { __m128 x = _mm_loadu_ps(sh+j-3);
            x = _mm_add_ps(x,x);
            x = sleef_expf4(x);
            x = _mm_sub_ps(one, _mm_div_ps (two, _mm_add_ps (one, x)));
            _mm_storeu_ps (vh+j-3, x);
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { __m128 x = _mm_loadl_pi (_mm_setzero_ps(), (__m64 *)(sh+j-1));
            x = _mm_add_ps(x,x);
            x = sleef_expf4(x);
            x = _mm_sub_ps (one, _mm_div_ps (two, _mm_add_ps (one, x)));
            _mm_storeu_ps (vh+j-1, x);
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (sh[j-1]);
          }
        }
#       else
        { for (j = 0; j<N_hidden; j++)
          { vh[j] = TANH (sh[j]);
          }
        }
#       endif

#     else  /* Use actual tanh functions, not quick and dirty */

#       if FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
        { j = 3;
          while (j<N_hidden)
          { _mm256_storeu_pd (vh+j-3, sleef_tanhd4 (_mm256_loadu_pd(sh+j-3)));
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { _mm_storeu_pd (vh+j-1, sleef_tanhd2 (_mm_loadu_pd(sh+j-1)));
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (sh[j-1]);
          }
        }
#       elif FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
        { j = 1;
          while (j<N_hidden)
          { _mm_storeu_pd (vh+j-1, sleef_tanhd2 (_mm_loadu_pd(sh+j-1)));
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (sh[j-1]);
          }
        }
#       elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
        { j = 7;
          while (j<N_hidden)
          { _mm256_storeu_ps (vh+j-7, sleef_tanhf8 (_mm256_loadu_ps(sh+j-7)));
            j += 8;
          }
          j -= 4;
          while (j<N_hidden)
          { _mm_storeu_ps (vh+j-3, sleef_tanhf4 (_mm_loadu_ps(sh+j-3)));
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { _mm_storel_pi ((__m64 *)(vh+j-1), 
               sleef_tanhf4 (_mm_loadl_pi(_mm_setzero_ps(),(__m64 *)(sh+j-1))));
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (sh[j-1]);
          }
        }
#       elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
        { j = 3;
          while (j<N_hidden)
          { _mm_storeu_ps (vh+j-3, sleef_tanhf4 (_mm_loadu_ps(sh+j-3)));
            j += 4;
          }
          j -= 2;
          if (j<N_hidden)
          { _mm_storel_pi ((__m64 *)(vh+j-1), 
               sleef_tanhf4 (_mm_loadl_pi(_mm_setzero_ps(),(__m64 *)(sh+j-1))));
            j += 2;
          }
          if (j<=N_hidden)
          { vh[j-1] = TANH (sh[j-1]);
          }
        }
#       else
        { for (j = 0; j<N_hidden; j++)
          { vh[j] = TANH (sh[j]);
          }
        }
#       endif
#     endif
    }
    else if (flgs->layer_type[l]==Sin_type)
    { for (j = 0; j<N_hidden; j++)
      { vh[j] = sqrt_2 * prec_sin(sh[j]*sqrt_2);
      }
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
        { __m256d a = _mm256_loadu_pd(sh+j-3);
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
        { __m128d a = _mm_loadu_pd(sh+j-1);
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
        { net_value a = sh[j-1];
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
        { __m128d a = _mm_loadu_pd(sh+j-1);
          __m128d v = _mm_or_pd(a,mask);  /* compute -fabs(a) */
          v = sleef_expd2(v);
          v = _mm_add_pd(one,v);
          v = sleef_logd2(v);
          v = _mm_add_pd (v, _mm_and_pd (a, _mm_cmpgt_pd (a, zero)));
          _mm_storeu_pd (vh+j-1, v);
          j += 2;
        }
        if (j<=N_hidden)
        { net_value a = sh[j-1];
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
        { __m128 a = _mm_loadu_ps(sh+j-3);
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
        { __m128 a = _mm_loadl_pi (zero, (__m64 *)(sh+j-1));
          __m128 v = _mm_or_ps(a,mask);
          v = sleef_expf4(v);
          v = _mm_add_ps(one,v);
          v = sleef_logf4(v);
          v = _mm_add_ps (v, _mm_and_ps (a, _mm_cmpgt_ps (a, zero)));
          _mm_storel_pi ((__m64 *)(vh+j-1), v);
          j += 2;
        }
        if (j<=N_hidden)
        { net_value a = sh[j-1];
          net_value v = 
            prec_log (1 + prec_exp(-prec_fabs(a)));  /* avoid overflow */
          if (a>0) v += a;
          vh[j-1] = v;
        }
      }
#     else
      { for (j = 0; j<N_hidden; j++)
        { net_value a = sh[j];
          net_value v = 
            prec_log (1 + prec_exp(-prec_fabs(a)));  /* avoid overflow */
          if (a>0) v += a;
          vh[j] = v;
        }
      }
#     endif
    }
    else if (flgs->layer_type[l]==Square_type)
    { for (j = 0; j<N_hidden; j++)
      { vh[j] = sh[j]*sh[j];
      }
    }
    else if (flgs->layer_type[l]==Cube_type)
    { for (j = 0; j<N_hidden; j++)
      { vh[j] = sh[j]*sh[j]*sh[j];
      }
    }
    else if (flgs->layer_type[l]==Identity_type)
    { for (j = 0; j<N_hidden; j++)
      { vh[j] = sh[j];
      }
    }
    else
    { abort();
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
    { add_connections_config (v->o, v->i, w->io,
        a->has_ti ? w->ti : 0, a->input_config[a->N_layers]);
    }
    else
    { add_connections (v->o, a->N_outputs, v->i, a->N_inputs,
                    w->io, a->has_ti ? w->ti : 0, 
                    flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0, 1);
    }
  }

  for (l = 0; l<a->N_layers; l++)
  { if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { add_connections_config (v->o, v->h[l], w->ho[l], 
                         a->has_th[l] ? w->th[l] : 0, a->hidden_config[k]);
      }
      else
      { add_connections (v->o, a->N_outputs, v->h[l], a->N_hidden[l], w->ho[l],
                         a->has_th[l] ? w->th[l] : 0, (unsigned short *) 0, 0);
      }
    }
  }
}


/* SET UNIT VALUES TO BIASES. */

HOSTDEV static void bias_values
( net_value *restrict v,	/* Array of unit values to set */
  int n,			/* Number of units */
  net_param const* b		/* Biases */
)
{ 
  int j;
  for (j = 0; j<n; j++) v[j] = b[j];
}


/* SET UNIT VALUES TO BIASES WHEN THERE IS A CONFIGURATON.  At present,
   just goes through the original list of connections in the configuration,
   without trying to optimize. */

HOSTDEV static void bias_values_config
( net_value *restrict v,	/* Array of unit values to set */
  int n,			/* Number of units */
  net_param const* b,		/* Biases */
  net_config const* cf		/* Configuration for biases */
)
{ 
  net_connection *cn = cf->conn;
  int c, j, k;

  memset (v, 0, n * sizeof *v);
  for (c = 0; (k = cn[c].w) >= 0; c++)
  { j = cn[c].d;
    v[j] += b[k];
  }
}


/* ADD CONTRIBUTION FROM ONE GROUP OF CONNECTIONS.  Adds the weighted input
   due to connections from one source layer to the current unit values for
   the destination layer. */

#define ADD_CONNECTIONS(offset,omit) \
do \
{ int i, j; \
  net_param o; \
  if (nd==1) \
  { net_value sv[4] = { 0, 0, 0, 0 }; \
    i = 3; \
    while (i<ns) \
    { o = (offset); if (!(omit)) sv[0] += (v[i-3] + o) * *w++; \
      o = (offset); if (!(omit)) sv[1] += (v[i-2] + o) * *w++; \
      o = (offset); if (!(omit)) sv[2] += (v[i-1] + o) * *w++; \
      o = (offset); if (!(omit)) sv[3] += (v[i-0] + o) * *w++; \
      i += 4; \
    } \
    i -= 3; \
    *s += (sv[0] + sv[2]) + (sv[1] + sv[3]); \
    while (i<ns) \
    { o = (offset); if (!(omit)) *s += (v[i] + o) * *w++; \
      i += 1; \
    } \
  } \
  else \
  { for (i = 0; i<ns; i++) \
    { o = (offset); \
      if (omit) continue; \
      net_value tv = v[i] + o; \
      if (tv!=0)  \
      { j = 3; \
        while (j<nd) \
        { s[j-3] += w[j-3] * tv; \
          s[j-2] += w[j-2] * tv; \
          s[j-1] += w[j-1] * tv; \
          s[j-0] += w[j-0] * tv; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { s[j] += w[j] * tv; \
          j += 1; \
        } \
      } \
      w += nd; \
    } \
  } \
} while (0)

#if FP64 && USE_SIMD_INTRINSICS && __AVX__

#define ADD_CONNECTIONS00 \
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
        if (_mm_ucomineq_sd (cast128d(TV), Z128d)) \
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
        if (_mm_ucomineq_sd (cast128d(TV2), Z128d)) \
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

#define ADD_CONNECTIONS00 \
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
        if (_mm_ucomineq_sd (TV, Z128d)) \
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
        if (_mm_ucomineq_sd (TV2, Z128d)) \
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

#define ADD_CONNECTIONS00 \
do \
{ int i, j; \
  if (nd==1) /* this part same as SSE3 code, could be improved */ \
  { __m128 Z = _mm_setzero_ps(); \
    __m128 SV = Z; \
    i = 7; \
    while (i<ns) \
    { SV = _mm_add_ps (SV, _mm_mul_ps (_mm_loadu_ps(v+i-7), \
                                       _mm_loadu_ps(w+i-7))); \
      SV = _mm_add_ps (SV, _mm_mul_ps (_mm_loadu_ps(v+i-3), \
                                       _mm_loadu_ps(w+i-3))); \
      i += 8; \
    } \
    i -= 4; \
    if (i<ns) \
    { SV = _mm_add_ps (SV, _mm_mul_ps (_mm_loadu_ps(v+i-3), \
                                       _mm_loadu_ps(w+i-3))); \
      i += 4; \
    } \
    __m128 S; \
    S = _mm_add_ps (SV, _mm_movehl_ps(SV,SV)); \
    i -= 2; \
    if (i<ns) \
    { S = _mm_add_ps (S, _mm_mul_ps (_mm_loadl_pi (Z, (__m64 *)(v+i-1)), \
                                     _mm_loadl_pi (Z, (__m64 *)(w+i-1)))); \
      i += 2; \
    } \
    S = _mm_hadd_ps(S,S); \
    if (i<=ns) \
    { S = _mm_add_ss (S, _mm_mul_ss (_mm_load_ss(v+i-1), _mm_load_ss(w+i-1))); \
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
        if (_mm_ucomineq_ss (cast128f(TV), Z)) \
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
        if (_mm_ucomineq_ss (cast128f(TV2), Z)) \
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
        S = FMA_ps(cast128f(TV2), _mm_loadl_pi(Z,(__m64 *)(w2+j-1)), S); \
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

#define ADD_CONNECTIONS00 \
do \
{ int i, j; \
  __m128 Z = _mm_setzero_ps(); \
  if (nd==1) \
  { __m128 SV = Z; \
    i = 7; \
    while (i<ns) \
    { SV = _mm_add_ps (SV, _mm_mul_ps (_mm_loadu_ps(v+i-7), \
                                       _mm_loadu_ps(w+i-7))); \
      SV = _mm_add_ps (SV, _mm_mul_ps (_mm_loadu_ps(v+i-3), \
                                       _mm_loadu_ps(w+i-3))); \
      i += 8; \
    } \
    i -= 4; \
    if (i<ns) \
    { SV = _mm_add_ps (SV, _mm_mul_ps (_mm_loadu_ps(v+i-3), \
                                       _mm_loadu_ps(w+i-3))); \
      i += 4; \
    } \
    __m128 S; \
    S = _mm_add_ps (SV, _mm_movehl_ps(SV,SV)); \
    i -= 2; \
    if (i<ns) \
    { S = _mm_add_ps (S, _mm_mul_ps (_mm_loadl_pi (Z, (__m64 *)(v+i-1)), \
                                     _mm_loadl_pi (Z, (__m64 *)(w+i-1)))); \
      i += 2; \
    } \
    S = _mm_hadd_ps(S,S); \
    if (i<=ns) \
    { S = _mm_add_ss (S, _mm_mul_ss (_mm_load_ss(v+i-1), _mm_load_ss(w+i-1))); \
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
        if (_mm_ucomineq_ss (TV, Z)) \
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
        if (_mm_ucomineq_ss (TV2, Z)) \
        { break; \
        } \
        i += 1; \
        w2 += nd; \
      } \
      j = 7; \
      while (j<nd) \
      { __m128 S = _mm_loadu_ps(s+j-7); \
        S = _mm_add_ps (S, _mm_mul_ps (TV, _mm_loadu_ps(w+j-7))); \
        S = _mm_add_ps (S, _mm_mul_ps (TV2, _mm_loadu_ps(w2+j-7))); \
        _mm_storeu_ps (s+j-7, S); \
        S = _mm_loadu_ps(s+j-3); \
        S = _mm_add_ps (S, _mm_mul_ps (TV, _mm_loadu_ps(w+j-3))); \
        S = _mm_add_ps (S, _mm_mul_ps (TV2, _mm_loadu_ps(w2+j-3))); \
        _mm_storeu_ps (s+j-3, S); \
        j += 8; \
      } \
      j -= 4; \
      if (j<nd) \
      { __m128 S = _mm_loadu_ps(s+j-3); \
        S = _mm_add_ps (S, _mm_mul_ps (TV, _mm_loadu_ps(w+j-3))); \
        S = _mm_add_ps (S, _mm_mul_ps (TV2, _mm_loadu_ps(w2+j-3))); \
        _mm_storeu_ps (s+j-3, S); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { __m128 S = _mm_loadl_pi (Z, (__m64 *)(s+j-1)); \
        S = _mm_add_ps (S, _mm_mul_ps (TV, _mm_loadl_pi(Z,(__m64 *)(w+j-1)))); \
        S = _mm_add_ps (S, _mm_mul_ps (TV2,_mm_loadl_pi(Z,(__m64*)(w2+j-1)))); \
        _mm_storel_pi ((__m64 *)(s+j-1), S); \
        j += 2; \
      } \
      if (j<=nd) \
      { __m128 S = _mm_load_ss(s+j-1); \
        S = _mm_add_ss (S, _mm_mul_ss (TV, _mm_load_ss(w+j-1))); \
        S = _mm_add_ss (S, _mm_mul_ss (TV2, _mm_load_ss(w2+j-1))); \
        _mm_store_ss (s+j-1, S); \
      } \
      i += 1; \
      w = w2+nd; \
    } \
  one_more: \
    j = 7; \
    while (j<nd) \
    { __m128 S = _mm_loadu_ps(s+j-7); \
      S = _mm_add_ps (S, _mm_mul_ps (TV, _mm_loadu_ps(w+j-7))); \
      _mm_storeu_ps (s+j-7, S); \
      S = _mm_loadu_ps(s+j-3); \
      S = _mm_add_ps (S, _mm_mul_ps (TV, _mm_loadu_ps(w+j-3))); \
      _mm_storeu_ps (s+j-3, S); \
      j += 8; \
    } \
    j -= 4; \
    if (j<nd) \
    { __m128 S = _mm_loadu_ps(s+j-3); \
      S = _mm_add_ps (S, _mm_mul_ps (TV, _mm_loadu_ps(w+j-3))); \
      _mm_storeu_ps (s+j-3, S); \
      j += 4; \
    } \
    j -= 2; \
    if (j<nd) \
    { __m128 S = _mm_loadl_pi (Z, (__m64 *)(s+j-1)); \
      S = _mm_add_ps (S, _mm_mul_ps (TV, _mm_loadl_pi (Z, (__m64 *)(w+j-1)))); \
      _mm_storel_pi ((__m64 *)(s+j-1), S); \
      j += 2; \
    } \
    if (j<=nd) \
    { __m128 S = _mm_load_ss(s+j-1); \
      S = _mm_add_ss (S, _mm_mul_ss (TV, _mm_load_ss(w+j-1))); \
      _mm_store_ss (s+j-1, S); \
    } \
  done: ; \
  } \
} while (0)

#else

#define ADD_CONNECTIONS00 ADD_CONNECTIONS(0,0)

#endif

HOSTDEV static void add_connections
( net_value *restrict s,  /* Summed input for destination units to add to */
  int nd,		  /* Number of destination units */
  net_value const* v,     /* Values for source units */
  int ns,		  /* Number of source units */
  net_param const* w,     /* Connection weights */
  net_param const* off,   /* Offsets to add to source unit values */
  unsigned short const* omit, /* Omit flags, null if not present/relevant */
  int ob		  /* Bit to look at in omit flags */
)
{
  if (omit==0)
  { if (off==0)
    { ADD_CONNECTIONS00;
    }
    else
    { ADD_CONNECTIONS(*off++,0);
    }
  }
  else
  { if (off==0)
    { ADD_CONNECTIONS(0,(*omit++)&ob);
    }
    else
    { ADD_CONNECTIONS(*off++,(*omit++)&ob);
    }
  }
}


/* ADD CONTRIBUTION FROM ONE GROUP OF CONNECTIONS WITH CONFIGURATION FROM FILE.
   Adds the weighted input due to connections from one source layer to the 
   current unit values for the destination layer. */

HOSTDEV static void add_connections_config
( net_value *restrict s,  /* Summed input for destination units to add to */
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
}
