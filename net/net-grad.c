/* NET-GRAD.C - Routine for calculating gradient from backpropagated info. */

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


/* This module finds the derivatives of the "error" on the training set
   with respect to the network parameters, using the backpropagated 
   derivatives of the error for each training case with respect to the
   unit values. 

   A version that adds derivatives from a training case to an existing 
   derivative structure is accessible to both CPU and GPU.  Versions that 
   store a derivative from one training case or the sum of two derivatives
   from two training cases to a derivative structure are accessible to 
   the GPU.
*/


/* ---------------------------- add_grad ------------------------------------ */

HOSTDEV static void add_grad1 (net_param *restrict, net_value const*, int);
HOSTDEV static void add_grad1_config (net_param *restrict, net_value const*,
                                      net_config const*);
HOSTDEV static void add_grad2 (net_param *restrict, net_value const*, 
                               net_param const*, int, net_value const*, int,
                               unsigned short const*, int);
HOSTDEV static void add_grad2_config (net_param *restrict, net_value const*, 
                                      net_param const*, net_value const*,
                                      net_config const*);


/* ADD TO GRADIENT OF ERROR WITH RESPECT TO NETWORK PARAMETERS.  Adds
   (if 'increment is 1) to a set of derivatives with respect to
   network parameters, stored in a structure of the same form as the
   parameters.  The derivatives added are of the "error" for a
   training case, derived from unit values and derivatives previously
   computed.

   One can economize by not bothering to compute the derivatives of the 
   error with respect to the input unit values if the network does not
   have input offset parameters. */

HOSTDEV void net_add_grad
( net_params *restrict g, /* Gradient with respect to parameters to add to */
  net_params const*w,	/* Network parameters */
  net_values const*v,	/* Values for units in network for a case */
  net_values const*d,	/* Backpropagated derivatives for a case */
  net_arch const*a,	/* Network architecture */
  net_flags const*flgs	/* Network flags, null if none */
)
{ 
  int l;

  if (a->has_ti) 
  { add_grad1 (g->ti, d->i, a->N_inputs);
  }

  for (l = 0; l<a->N_layers; l++)
  { 
    int N_hidden = a->N_hidden[l];

    if (a->has_bh[l]) 
    { if (a->bias_config[l])
      { add_grad1_config (g->bh[l], d->s[l], a->bias_config[l]);
      }
      else
      { add_grad1 (g->bh[l], d->s[l], N_hidden);
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { add_grad2_config (g->ih[l], v->i, a->has_ti ? w->ti : 0, d->s[l], 
                          a->input_config[l]);
      }
      else
      { add_grad2 (g->ih[l], v->i, a->has_ti ? w->ti : 0, a->N_inputs, 
                   d->s[l], N_hidden, 
                   flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1));
      }
    }

    if (l>0 && a->has_hh[l-1])
    { if (a->hidden_config[l])
      { add_grad2_config
           (g->hh[l-1], v->h[l-1], a->has_th[l-1] ? w->th[l-1] : 0,
            d->s[l], a->hidden_config[l]);
      }
      else
      { add_grad2 (g->hh[l-1], v->h[l-1], a->has_th[l-1] ? w->th[l-1] : 0,
          a->N_hidden[l-1], d->s[l], N_hidden, (unsigned short *)0, 0);
      }
    }

    if (a->has_th[l]) 
    { add_grad1 (g->th[l], d->h[l], N_hidden);
    }

    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { add_grad2_config (g->ho[l], v->h[l], a->has_th[l] ? w->th[l] : 0,
                          d->o, a->hidden_config[k]);
      }
      else
      { add_grad2 (g->ho[l], v->h[l], a->has_th[l] ? w->th[l] : 0,
                   N_hidden, d->o, a->N_outputs, (unsigned short *) 0, 0);
      }
    }
  }

  if (a->has_io) 
  { if (a->input_config[a->N_layers])
    { add_grad2_config (g->io, v->i, a->has_ti ? w->ti : 0, d->o,
                        a->input_config[a->N_layers]);
    }
    else
    { add_grad2 (g->io, v->i, a->has_ti ? w->ti : 0, a->N_inputs, 
                 d->o, a->N_outputs, 
                 flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0, 1);
    }
  }

  if (a->has_bo) 
  { if (a->bias_config[a->N_layers])
    { add_grad1_config (g->bo, d->o, a->bias_config[a->N_layers]);
    }
    else
    { add_grad1 (g->bo, d->o, a->N_outputs);
    }
  }
}


/* ADD TO GRADIENT FROM UNIT DERIVATIVE. */

HOSTDEV static void add_grad1
( net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* v,     /* Derivatives with respect to unit values */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = 0; i<n; i++)
  { g[i] += v[i];
  }
}


/* ADD TO GRADIENT FROM UNIT DERIVATIVE, WITH CONFIGURATION.  At present,
   just goes through the original list of connections in the configuration,
   without trying to optimize. */

HOSTDEV static void add_grad1_config
( net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* v,     /* Derivatives with respect to unit values */
  net_config const* cf    /* Configuration for biases */
)
{ net_connection *cn = cf->conn;
  int c, j, k;

  for (c = 0; (k = cn[c].w) >= 0; c++)
  { j = cn[c].d;
    g[k] += v[j];
  }
}


/* ADD TO GRADIENT FROM PRODUCT OF UNIT VALUE AND UNIT DERIVATIVE. */

#define ADD_GRAD2(offset,omit) \
do \
{ net_value o; \
  int i, j; \
  if (nd==1) \
  { net_value d0 = d[0]; \
    i = 3; \
    while (i<nv) \
    { o = (offset); if (!(omit)) *g++ += (v[i-3] + o) * d0; \
      o = (offset); if (!(omit)) *g++ += (v[i-2] + o) * d0; \
      o = (offset); if (!(omit)) *g++ += (v[i-1] + o) * d0; \
      o = (offset); if (!(omit)) *g++ += (v[i-0] + o) * d0; \
      i += 4; \
    } \
    i -= 3; \
    while (i<nv) \
    { o = (offset); if (!(omit)) *g++ += (v[i] + o) * d0; \
      i += 1; \
    } \
  } \
  else \
  { for (i = 0; i<nv; i++) \
    { net_value tv; \
      o = (offset); \
      if (omit) continue; \
      tv = v[i] + o; \
      if (tv!=0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-3] += tv * d[j-3]; \
          g[j-2] += tv * d[j-2]; \
          g[j-1] += tv * d[j-1]; \
          g[j-0] += tv * d[j-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { g[j] += tv * d[j]; \
          j += 1; \
        } \
      } \
      g += nd; \
    } \
  } \
} while (0)

#if FP64 && USE_SIMD_INTRINSICS && __AVX__

#define ADD_GRAD2_00 \
do \
{ int i, j; \
  if (nd==1) \
  { __m256d D0 = _mm256_broadcast_sd(d); \
    i = 3; \
    while (i<nv) \
    { _mm256_storeu_pd (g+i-3, FMA256_pd (D0, _mm256_loadu_pd(v+i-3), \
                                              _mm256_loadu_pd(g+i-3))); \
      i += 4; \
    } \
    i -= 2; \
    if (i<nv) \
    { _mm_storeu_pd (g+i-1, FMA_pd (cast128d(D0), _mm_loadu_pd(v+i-1), \
                                                  _mm_loadu_pd(g+i-1))); \
      i += 2; \
    } \
    if (i<=nv) \
    { _mm_store_sd (g+i-1, FMA_sd (cast128d(D0), _mm_load_sd(v+i-1), \
                                                 _mm_load_sd(g+i-1))); \
    } \
  } \
  else \
  { __m256d TV, TV2; \
    i = 0; \
    for (;;) \
    { __m128d Z = _mm_setzero_pd(); \
      for (;;) \
      { if (i==nv) goto done; \
        TV = _mm256_broadcast_sd (v+i); \
        if (_mm_ucomineq_sd (cast128d(TV), Z)) \
        { break; \
        } \
        i += 1; \
        g += nd; \
      } \
      net_value *g2 = g+nd; \
      i += 1; \
      for (;;) \
      { if (i==nv) goto one_more; \
        TV2 = _mm256_broadcast_sd (v+i); \
        if (_mm_ucomineq_sd (cast128d(TV2), Z)) \
        { break; \
        } \
        i += 1; \
        g2 += nd; \
      } \
      j = 3; \
      while (j<nd) \
      { __m256d D = _mm256_loadu_pd(d+j-3); \
        _mm256_storeu_pd (g+j-3, FMA256_pd (TV, D, \
                                                  _mm256_loadu_pd(g+j-3))); \
        _mm256_storeu_pd (g2+j-3, FMA256_pd (TV2, D, \
                                                   _mm256_loadu_pd(g2+j-3))); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { __m128d D = _mm_loadu_pd(d+j-1); \
        _mm_storeu_pd (g+j-1, FMA_pd (cast128d(TV), D, _mm_loadu_pd(g+j-1))); \
        _mm_storeu_pd (g2+j-1, FMA_pd(cast128d(TV2), D,_mm_loadu_pd(g2+j-1))); \
        j += 2; \
      } \
      if (j<=nd) \
      { __m128d D = _mm_load_sd(d+j-1); \
        _mm_store_sd (g+j-1, FMA_sd (cast128d(TV), D, _mm_load_sd(g+j-1))); \
        _mm_store_sd (g2+j-1, FMA_sd (cast128d(TV2), D, _mm_load_sd(g2+j-1))); \
      } \
      i += 1; \
      g = g2+nd; \
    } \
    goto done; \
  one_more: \
    j = 3; \
    while (j<nd) \
    { __m256d D = _mm256_loadu_pd(d+j-3); \
      _mm256_storeu_pd (g+j-3, FMA256_pd (TV, D, _mm256_loadu_pd(g+j-3))); \
      j += 4; \
    } \
    j -= 2; \
    if (j<nd) \
    { __m128d D = _mm_loadu_pd(d+j-1); \
      _mm_storeu_pd (g+j-1, FMA_pd (cast128d(TV), D, _mm_loadu_pd(g+j-1))); \
      j += 2; \
    } \
    if (j<=nd) \
    { __m128d D = _mm_load_sd(d+j-1); \
      _mm_store_sd (g+j-1, FMA_sd (cast128d(TV), D, _mm_load_sd(g+j-1))); \
    } \
  done: ; \
  } \
} while (0)

#elif FP64 && USE_SIMD_INTRINSICS && __SSE2__

#define ADD_GRAD2_00 \
do \
{ int i, j; \
  if (nd==1) \
  { __m128d D0 = _mm_set1_pd(*d); \
    i = 3; \
    while (i<nv) \
    { _mm_storeu_pd (g+i-3, FMA_pd (D0, _mm_loadu_pd(v+i-3), \
                                        _mm_loadu_pd(g+i-3))); \
      _mm_storeu_pd (g+i-1, FMA_pd (D0, _mm_loadu_pd(v+i-1), \
                                        _mm_loadu_pd(g+i-1))); \
      i += 4; \
    } \
    i -= 2; \
    if (i<nv) \
    { _mm_storeu_pd (g+i-1, FMA_pd (D0, _mm_loadu_pd(v+i-1), \
                                        _mm_loadu_pd(g+i-1))); \
      i += 2; \
    } \
    if (i<=nv) \
    { _mm_store_sd (g+i-1, FMA_sd (D0, _mm_load_sd(v+i-1), \
                                       _mm_load_sd(g+i-1))); \
    } \
  } \
  else \
  { __m128d TV, TV2, D; \
    i = 0; \
    for (;;) \
    { __m128d Z = _mm_setzero_pd(); \
      for (;;) \
      { if (i==nv) goto done; \
        TV = _mm_set1_pd (*(v+i)); \
        if (_mm_ucomineq_sd (TV, Z)) \
        { break; \
        } \
        i += 1; \
        g += nd; \
      } \
      net_value *g2 = g+nd; \
      i += 1; \
      for (;;) \
      { if (i==nv) goto one_more; \
        TV2 = _mm_set1_pd (*(v+i)); \
        if (_mm_ucomineq_sd (TV2, Z)) \
        { break; \
        } \
        i += 1; \
        g2 += nd; \
      } \
      j = 3; \
      while (j<nd) \
      {  D = _mm_loadu_pd(d+j-3); \
        _mm_storeu_pd (g+j-3, FMA_pd (TV, D, _mm_loadu_pd(g+j-3))); \
        _mm_storeu_pd (g2+j-3, FMA_pd (TV2, D, _mm_loadu_pd(g2+j-3))); \
         D = _mm_loadu_pd(d+j-1); \
        _mm_storeu_pd (g+j-1, FMA_pd (TV, D, _mm_loadu_pd(g+j-1))); \
        _mm_storeu_pd (g2+j-1, FMA_pd (TV2, D, _mm_loadu_pd(g2+j-1))); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { D = _mm_loadu_pd(d+j-1); \
        _mm_storeu_pd (g+j-1, FMA_pd (TV, D, _mm_loadu_pd(g+j-1))); \
        _mm_storeu_pd (g2+j-1, FMA_pd(TV2, D,_mm_loadu_pd(g2+j-1))); \
        j += 2; \
      } \
      if (j<=nd) \
      { D = _mm_load_sd(d+j-1); \
        _mm_store_sd (g+j-1, FMA_sd (TV, D, _mm_load_sd(g+j-1))); \
        _mm_store_sd (g2+j-1, FMA_sd (TV2, D, _mm_load_sd(g2+j-1))); \
      } \
      i += 1; \
      g = g2+nd; \
    } \
    goto done; \
  one_more: \
    j = 3; \
    while (j<nd) \
    {  D = _mm_loadu_pd(d+j-3); \
      _mm_storeu_pd (g+j-3, FMA_pd (TV, D, _mm_loadu_pd(g+j-3))); \
       D = _mm_loadu_pd(d+j-1); \
      _mm_storeu_pd (g+j-1, FMA_pd (TV, D, _mm_loadu_pd(g+j-1))); \
      j += 4; \
    } \
    j -= 2; \
    if (j<nd) \
    { D = _mm_loadu_pd(d+j-1); \
      _mm_storeu_pd (g+j-1, FMA_pd (TV, D, _mm_loadu_pd(g+j-1))); \
      j += 2; \
    } \
    if (j<=nd) \
    { D = _mm_load_sd(d+j-1); \
      _mm_store_sd (g+j-1, FMA_sd (TV, D, _mm_load_sd(g+j-1))); \
    } \
  done: ; \
  } \
} while (0)

#elif FP32 && USE_SIMD_INTRINSICS && __AVX__

#define ADD_GRAD2_00 \
do \
{ int i, j; \
  if (nd==1) \
  { __m256 D0 = _mm256_set1_ps(*d); \
    i = 7; \
    while (i<nv) \
    { _mm256_storeu_ps (g+i-7, \
        FMA256_ps (D0, _mm256_loadu_ps(v+i-7), _mm256_loadu_ps(g+i-7))); \
      i += 8; \
    } \
    i -= 4; \
    if (i<nv) \
    { _mm_storeu_ps (g+i-3, \
        FMA_ps(cast128f(D0), _mm_loadu_ps(v+i-3), _mm_loadu_ps(g+i-3))); \
      i += 4; \
    } \
    i -= 2; \
    if (i<nv) \
    { __m128 Z = _mm_setzero_ps(); \
      _mm_storel_pi ((__m64 *)(g+i-1), \
        FMA_ps (cast128f(D0), _mm_loadl_pi (Z, (__m64 const*) (v+i-1)), \
                      _mm_loadl_pi(Z, (__m64 const*) (g+i-1)))); \
      i += 2; \
    } \
    if (i<=nv) \
    { _mm_store_ss (g+i-1, FMA_ss (cast128f(D0), _mm_load_ss(v+i-1), \
                                   _mm_load_ss(g+i-1))); \
    } \
  } \
  else \
  { __m256 TV, TV2; \
    __m128 Z = _mm_setzero_ps(); \
    i = 0; \
    for (;;) \
    { for (;;) \
      { if (i==nv) goto done; \
        TV = _mm256_set1_ps (*(v+i)); \
        if (_mm_ucomineq_ss (cast128f(TV), Z)) \
        { break; \
        } \
        i += 1; \
        g += nd; \
      } \
      net_value *g2 = g+nd; \
      i += 1; \
      for (;;) \
      { if (i==nv) goto one_more; \
        TV2 = _mm256_set1_ps (*(v+i)); \
        if (_mm_ucomineq_ss (cast128f(TV2), Z)) \
        { break; \
        } \
        i += 1; \
        g2 += nd; \
      } \
      j = 7; \
      while (j<nd) \
      { __m256 D = _mm256_loadu_ps(d+j-7); \
        _mm256_storeu_ps (g+j-7, FMA256_ps (TV, D, _mm256_loadu_ps(g+j-7))); \
        _mm256_storeu_ps (g2+j-7,FMA256_ps (TV2, D, _mm256_loadu_ps(g2+j-7))); \
        j += 8; \
      } \
      j -= 4; \
      if (j<nd) \
      { __m128 D = _mm_loadu_ps(d+j-3); \
        _mm_storeu_ps(g+j-3, FMA_ps (cast128f(TV), D, _mm_loadu_ps(g+j-3))); \
        _mm_storeu_ps(g2+j-3,FMA_ps (cast128f(TV2), D, _mm_loadu_ps(g2+j-3))); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { __m128 D = _mm_loadl_pi (Z, (__m64 const*) (d+j-1)); \
        _mm_storel_pi ((__m64 *)(g+j-1), \
           FMA_ps (cast128f(TV), D, _mm_loadl_pi(Z,(__m64 const*) (g+j-1)))); \
        _mm_storel_pi ((__m64 *)(g2+j-1), \
           FMA_ps (cast128f(TV2), D,_mm_loadl_pi(Z,(__m64 const*) (g2+j-1)))); \
        j += 2; \
      } \
      if (j<=nd) \
      { __m128 D = _mm_load_ss(d+j-1); \
        _mm_store_ss (g+j-1, FMA_ss (cast128f(TV), D, _mm_load_ss(g+j-1))); \
        _mm_store_ss (g2+j-1, FMA_ss (cast128f(TV2), D, _mm_load_ss(g2+j-1))); \
      } \
      i += 1; \
      g = g2+nd; \
    } \
    goto done; \
  one_more: \
    j = 7; \
    while (j<nd) \
    { __m256 D = _mm256_loadu_ps(d+j-7); \
      _mm256_storeu_ps (g+j-7, FMA256_ps (TV, D, _mm256_loadu_ps(g+j-7))); \
      j += 8; \
    } \
    j -= 4; \
    while (j<nd) \
    { __m128 D = _mm_loadu_ps(d+j-3); \
      _mm_storeu_ps (g+j-3, FMA_ps (cast128f(TV), D, _mm_loadu_ps(g+j-3))); \
      j += 4; \
    } \
    j -= 2; \
    if (j<nd) \
    { __m128 D = _mm_loadl_pi (Z, (__m64 const*) (d+j-1)); \
      _mm_storel_pi ((__m64 *)(g+j-1), \
         FMA_ps (cast128f(TV), D, _mm_loadl_pi (Z, (__m64 const*) (g+j-1)))); \
      j += 2; \
    } \
    if (j<=nd) \
    { __m128 D = _mm_load_ss(d+j-1); \
      _mm_store_ss (g+j-1, FMA_ss (cast128f(TV), D, _mm_load_ss(g+j-1))); \
    } \
  done: ; \
  } \
} while (0)

#elif FP32 && USE_SIMD_INTRINSICS && __SSE2__

#define ADD_GRAD2_00 \
do \
{ int i, j; \
  if (nd==1) \
  { __m128 D0 = _mm_set1_ps(*d); \
    i = 3; \
    while (i<nv) \
    { _mm_storeu_ps (g+i-3, \
                     FMA_ps (D0, _mm_loadu_ps(v+i-3), _mm_loadu_ps(g+i-3))); \
      i += 4; \
    } \
    i -= 2; \
    if (i<nv) \
    { __m128 Z = _mm_setzero_ps(); \
      _mm_storel_pi ((__m64 *)(g+i-1), \
        FMA_ps (D0, _mm_loadl_pi (Z, (__m64 const*) (v+i-1)), \
                    _mm_loadl_pi (Z, (__m64 const*) (g+i-1)))); \
      i += 2; \
    } \
    if (i<=nv) \
    { _mm_store_ss(g+i-1, FMA_ss(D0, _mm_load_ss(v+i-1), _mm_load_ss(g+i-1))); \
    } \
  } \
  else \
  { __m128 TV, TV2; \
    __m128 Z = _mm_setzero_ps(); \
    i = 0; \
    for (;;) \
    { for (;;) \
      { if (i==nv) goto done; \
        TV = _mm_set1_ps (*(v+i)); \
        if (_mm_ucomineq_ss (TV, Z)) \
        { break; \
        } \
        i += 1; \
        g += nd; \
      } \
      net_value *g2 = g+nd; \
      i += 1; \
      for (;;) \
      { if (i==nv) goto one_more; \
        TV2 = _mm_set1_ps (*(v+i)); \
        if (_mm_ucomineq_ss (TV2, Z)) \
        { break; \
        } \
        i += 1; \
        g2 += nd; \
      } \
      j = 3; \
      while (j<nd) \
      { __m128 D = _mm_loadu_ps(d+j-3); \
        _mm_storeu_ps (g+j-3, FMA_ps (TV, D, _mm_loadu_ps(g+j-3))); \
        _mm_storeu_ps (g2+j-3, FMA_ps (TV2, D, _mm_loadu_ps(g2+j-3))); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { __m128 D = _mm_loadl_pi (Z, (__m64 const*) (d+j-1)); \
        _mm_storel_pi ((__m64 *)(g+j-1), \
           FMA_ps (TV, D, _mm_loadl_pi (Z, (__m64 const*) (g+j-1)))); \
        _mm_storel_pi ((__m64 *)(g2+j-1), \
           FMA_ps (TV2, D, _mm_loadl_pi(Z, (__m64 const*) (g2+j-1)))); \
        j += 2; \
      } \
      if (j<=nd) \
      { __m128 D = _mm_load_ss(d+j-1); \
        _mm_store_ss (g+j-1, FMA_ss (TV, D, _mm_load_ss(g+j-1))); \
        _mm_store_ss (g2+j-1, FMA_ss (TV2, D, _mm_load_ss(g2+j-1))); \
      } \
      i += 1; \
      g = g2+nd; \
    } \
    goto done; \
  one_more: \
    j = 3; \
    while (j<nd) \
    { __m128 D = _mm_loadu_ps(d+j-3); \
      _mm_storeu_ps (g+j-3, FMA_ps (TV, D, _mm_loadu_ps(g+j-3))); \
      j += 4; \
    } \
    j -= 2; \
    if (j<nd) \
    { __m128 D = _mm_loadl_pi (Z, (__m64 const*) (d+j-1)); \
      _mm_storel_pi ((__m64 *)(g+j-1), \
         FMA_ps (TV, D, _mm_loadl_pi (Z, (__m64 const*) (g+j-1)))); \
      j += 2; \
    } \
    if (j<=nd) \
    { __m128 D = _mm_load_ss(d+j-1); \
      _mm_store_ss (g+j-1, FMA_ss (TV, D, _mm_load_ss(g+j-1))); \
    } \
  done: ; \
  } \
} while (0)

#else

#define ADD_GRAD2_00 \
do \
{ int i, j; \
  if (nd==1) \
  { net_value d0 = d[0]; \
    i = 3; \
    while (i<nv) \
    { g[i-3] += v[i-3] * d0; \
      g[i-2] += v[i-2] * d0; \
      g[i-1] += v[i-1] * d0; \
      g[i-0] += v[i-0] * d0; \
      i += 4; \
    } \
    i -= 3; \
    while (i<nv) \
    { g[i] += v[i] * d0; \
      i += 1; \
    } \
  } \
  else \
  { net_value tv; \
    for (i = 0; i<nv; i++) \
    { tv = v[i]; \
      if (tv!=0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-3] += tv * d[j-3]; \
          g[j-2] += tv * d[j-2]; \
          g[j-1] += tv * d[j-1]; \
          g[j-0] += tv * d[j-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { g[j] += tv * d[j]; \
          j += 1; \
        } \
      } \
      g += nd; \
    } \
  } \
} while (0)

#endif

HOSTDEV static void add_grad2
( net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* v,     /* Source unit values */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value const* d,     /* Derivatives with respect to destination units */
  int nd,		  /* Number of destination units */
  unsigned short const* omit, /* Omit flags, null if not present */
  int ob		  /* Bit to look at in omit flags */
)
{ 
  if (omit==0)
  { if (off==0)
    { ADD_GRAD2_00;
    }
    else
    { ADD_GRAD2(*off++,0);
    }
  }
  else
  { if (off==0)
    { ADD_GRAD2(0,(*omit++)&ob);
    }
    else
    { ADD_GRAD2(*off++,(*omit++)&ob);
    }
  }
}


/* ADD TO GRADIENT FROM PRODUCT OF UNIT VALUE AND UNIT DERIVATIVE.  For
   when the connections are specified by a configuration file. */

HOSTDEV static void add_grad2_config
( net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* s,     /* Source unit values */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  net_value const* d,     /* Derivatives with respect to destination units */
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
        { __m256d SI = _mm256_set1_pd (s[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (g+k, FMA256_pd (SI, _mm256_loadu_pd(d+j),
                                                _mm256_loadu_pd(g+k)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256d SI = _mm256_set1_pd (s[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (g+k, FMA256_pd (SI, _mm256_loadu_pd(d+j),
                                                _mm256_loadu_pd(g+k)));
        }
      }
    }
#   elif FP64 && USE_SIMD_INTRINSICS && __SSE2__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m128d SI = _mm_set1_pd (s[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_pd (g+k, FMA_pd (SI, _mm_loadu_pd(d+j),
                                          _mm_loadu_pd(g+k)));
          _mm_storeu_pd (g+k+2, FMA_pd (SI, _mm_loadu_pd(d+j+2),
                                            _mm_loadu_pd(g+k+2)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m128d SI = _mm_set1_pd (s[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_pd (g+k, FMA_pd (SI, _mm_loadu_pd(d+j),
                                          _mm_loadu_pd(g+k)));
          _mm_storeu_pd (g+k+2, FMA_pd (SI, _mm_loadu_pd(d+j+2),
                                            _mm_loadu_pd(g+k+2)));
        }
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __SSE2__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m128 SI = _mm_set1_ps (s[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_ps (g+k, FMA_ps(SI, _mm_loadu_ps(d+j), _mm_loadu_ps(g+k)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m128 SI = _mm_set1_ps (s[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_ps (g+k, FMA_ps(SI, _mm_loadu_ps(d+j), _mm_loadu_ps(g+k)));
        }
      }
    }
#   else
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { net_value soi = s[cn[c].s] + off[cn[c].s];
          j = cn[c].d;
          g[k+0] += soi * d[j+0];
          g[k+1] += soi * d[j+1];
          g[k+2] += soi * d[j+2];
          g[k+3] += soi * d[j+3];
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { net_value si = s[cn[c].s];
          j = cn[c].d;
          g[k+0] += si * d[j+0];
          g[k+1] += si * d[j+1];
          g[k+2] += si * d[j+2];
          g[k+3] += si * d[j+3];
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
      { net_value soi = s[cn[c].s] + off[cn[c].s];
        j = cn[c].d;
        g[k] += soi * d[j];
        j = cn[c+1].d; k = cn[c+1].w; 
        g[k] += soi * d[j];
        j = cn[c+2].d; k = cn[c+2].w; 
        g[k] += soi * d[j];
        j = cn[c+3].d; k = cn[c+3].w; 
        g[k] += soi * d[j];
      }
    }
    else
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value si = s[cn[c].s];
        j = cn[c].d;
        g[k] += si * d[j];
        j = cn[c+1].d; k = cn[c+1].w; 
        g[k] += si * d[j];
        j = cn[c+2].d; k = cn[c+2].w; 
        g[k] += si * d[j];
        j = cn[c+3].d; k = cn[c+3].w; 
        g[k] += si * d[j];
      }
    }

    cn = cf->single4_d;
    if (off)
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value dj = d[cn[c].d];
        i = cn[c].s;
        g[k] += (s[i]+off[i]) * dj;
        i = cn[c+1].s; k = cn[c+1].w; 
        g[k] += (s[i]+off[i]) * dj;
        i = cn[c+2].s; k = cn[c+2].w; 
        g[k] += (s[i]+off[i]) * dj;
        i = cn[c+3].s; k = cn[c+3].w; 
        g[k] += (s[i]+off[i]) * dj;
      }
    }
    else
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value dj = d[cn[c].d];
        i = cn[c].s;
        g[k] += s[i] * dj;
        i = cn[c+1].s; k = cn[c+1].w; 
        g[k] += s[i] * dj;
        i = cn[c+2].s; k = cn[c+2].w; 
        g[k] += s[i] * dj;
        i = cn[c+3].s; k = cn[c+3].w; 
        g[k] += s[i] * dj;
      }
    }
  }

  cn = CONFIG_ORIGINAL ? cf->conn : cf->single;
  if (off)
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { i = cn[c].s; j = cn[c].d;
      g[k] += (s[i]+off[i]) * d[j];
    }
  }
  else
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { i = cn[c].s; j = cn[c].d;
      g[k] += s[i] * d[j];
    }
  }
}


/* --------------------------- store_grad ----------------------------------- */

#if __CUDACC__ 

__device__ static void store_grad1 (net_param *restrict, net_value const*, int);
__device__ static void store_grad1_config (net_param *restrict, 
                                           net_value const*, net_config const*);
__device__ static void store_grad2 (net_param *restrict, net_value const*, 
                               net_param const*, int, net_value const*, int,
                               unsigned short const*, int);
__device__ static void store_grad2_config (net_param *restrict, 
                                      net_value const*, 
                                      net_param const*, net_value const*,
                                      net_config const*);


/* STORE GRADIENT OF ERROR WITH RESPECT TO NETWORK PARAMETERS.  Stores
   to a set of derivatives with respect to network parameters, stored
   in a structure of the same form as the parameters.  The derivatives
   stored are of the "error" for a training case, derived from unit
   values and derivatives previously computed.

   One can economize by not bothering to compute the derivatives of the 
   error with respect to the input unit values if the network does not
   have input offset parameters. */

__device__ void net_store_grad
( net_params *restrict g, /* Gradient with respect to parameters to add to */
  net_params const*w,	/* Network parameters */
  net_values const*v,	/* Values for units in network for a case */
  net_values const*d,	/* Backpropagated derivatives for a case */
  net_arch const*a,	/* Network architecture */
  net_flags const*flgs	/* Network flags, null if none */
)
{
  int l;

  if (a->has_ti) 
  { store_grad1 (g->ti, d->i, a->N_inputs);
  }

  for (l = 0; l<a->N_layers; l++)
  { 
    int N_hidden = a->N_hidden[l];

    if (a->has_bh[l]) 
    { if (a->bias_config[l])
      { store_grad1_config (g->bh[l], d->s[l], a->bias_config[l]);
      }
      else
      { store_grad1 (g->bh[l], d->s[l], N_hidden);
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { store_grad2_config (g->ih[l], v->i, a->has_ti ? w->ti : 0, d->s[l], 
                          a->input_config[l]);
      }
      else
      { store_grad2 (g->ih[l], v->i, a->has_ti ? w->ti : 0, a->N_inputs, 
                   d->s[l], N_hidden, 
                   flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1));
      }
    }

    if (l>0 && a->has_hh[l-1])
    { if (a->hidden_config[l])
      { store_grad2_config
           (g->hh[l-1], v->h[l-1], a->has_th[l-1] ? w->th[l-1] : 0,
            d->s[l], a->hidden_config[l]);
      }
      else
      { store_grad2 (g->hh[l-1], v->h[l-1], a->has_th[l-1] ? w->th[l-1] : 0,
          a->N_hidden[l-1], d->s[l], N_hidden, (unsigned short *)0, 0);
      }
    }

    if (a->has_th[l]) 
    { store_grad1 (g->th[l], d->h[l], N_hidden);
    }

    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { store_grad2_config (g->ho[l], v->h[l], a->has_th[l] ? w->th[l] : 0,
                          d->o, a->hidden_config[k]);
      }
      else
      { store_grad2 (g->ho[l], v->h[l], a->has_th[l] ? w->th[l] : 0,
                   N_hidden, d->o, a->N_outputs, (unsigned short *) 0, 0);
      }
    }
  }

  if (a->has_io) 
  { if (a->input_config[a->N_layers])
    { store_grad2_config (g->io, v->i, a->has_ti ? w->ti : 0, d->o,
                        a->input_config[a->N_layers]);
    }
    else
    { store_grad2 (g->io, v->i, a->has_ti ? w->ti : 0, a->N_inputs, 
                 d->o, a->N_outputs, 
                 flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0, 1);
    }
  }

  if (a->has_bo) 
  { if (a->bias_config[a->N_layers])
    { store_grad1_config (g->bo, d->o, a->bias_config[a->N_layers]);
    }
    else
    { store_grad1 (g->bo, d->o, a->N_outputs);
    }
  }
}


/* STORE GRADIENT FROM UNIT DERIVATIVE. */

__device__ static void store_grad1
( net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v,     /* Derivatives with respect to unit values */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = 0; i<n; i++)
  { g[i] = v[i];
  }
}


/* STORE GRADIENT FROM UNIT DERIVATIVE, WITH CONFIGURATION.  At present,
   just goes through the original list of connections in the configuration,
   without trying to optimize. */

__device__ static void store_grad1_config
( net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v,     /* Derivatives with respect to unit values */
  net_config const* cf    /* Configuration for biases */
)
{ int k;
  for (k = 0; k<cf->N_wts; k++)
  { g[k] = 0;
  }
  add_grad1_config (g, v, cf);
}


/* STORE GRADIENT FROM PRODUCT OF UNIT VALUE AND UNIT DERIVATIVE. */

#define STORE_GRAD2(offset,omit) \
do \
{ net_value tv, o; \
  int i, j; \
  if (nd==1) \
  { net_value d0 = d[0]; \
    i = 3; \
    while (i<nv) \
    { o = (offset); if (!(omit)) *g++ = (v[i-3] + o) * d0; \
      o = (offset); if (!(omit)) *g++ = (v[i-2] + o) * d0; \
      o = (offset); if (!(omit)) *g++ = (v[i-1] + o) * d0; \
      o = (offset); if (!(omit)) *g++ = (v[i-0] + o) * d0; \
      i += 4; \
    } \
    i -= 3; \
    while (i<nv) \
    { o = (offset); if (!(omit)) *g++ = (v[i] + o) * d0; \
      i += 1; \
    } \
  } \
  else \
  { for (i = 0; i<nv; i++) \
    { o = (offset); \
      if (omit) continue; \
      tv = v[i] + o; \
      if (tv!=0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-3] = tv * d[j-3]; \
          g[j-2] = tv * d[j-2]; \
          g[j-1] = tv * d[j-1]; \
          g[j-0] = tv * d[j-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { g[j] = tv * d[j]; \
          j += 1; \
        } \
      } \
      g += nd; \
    } \
  } \
} while (0)

#define STORE_GRAD2_00 \
do \
{ int i, j; \
  if (nd==1) \
  { net_value d0 = d[0]; \
    i = 3; \
    while (i<nv) \
    { g[i-3] = v[i-3] * d0; \
      g[i-2] = v[i-2] * d0; \
      g[i-1] = v[i-1] * d0; \
      g[i-0] = v[i-0] * d0; \
      i += 4; \
    } \
    i -= 3; \
    while (i<nv) \
    { g[i] = v[i] * d0; \
      i += 1; \
    } \
  } \
  else \
  { net_value tv; \
    for (i = 0; i<nv; i++) \
    { tv = v[i]; \
      if (tv!=0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-3] = tv * d[j-3]; \
          g[j-2] = tv * d[j-2]; \
          g[j-1] = tv * d[j-1]; \
          g[j-0] = tv * d[j-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { g[j] = tv * d[j]; \
          j += 1; \
        } \
      } \
      g += nd; \
    } \
  } \
} while (0)

__device__ static void store_grad2
( net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v,     /* Source unit values */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value const* d,     /* Derivatives with respect to destination units */
  int nd,		  /* Number of destination units */
  unsigned short const* omit, /* Omit flags, null if not present */
  int ob		  /* Bit to look at in omit flags */
)
{ 
  if (omit==0)
  { if (off==0)
    { STORE_GRAD2_00;
    }
    else
    { STORE_GRAD2(*off++,0);
    }
  }
  else
  { if (off==0)
    { STORE_GRAD2(0,(*omit++)&ob);
    }
    else
    { STORE_GRAD2(*off++,(*omit++)&ob);
    }
  }
}


/* STORE GRADIENT FROM PRODUCT OF UNIT VALUE AND UNIT DERIVATIVE.  For
   when the connections are specified by a configuration file. */

__device__ static void store_grad2_config
( net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* s,     /* Source unit values */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  net_value const* d,     /* Derivatives with respect to destination units */
  net_config const* cf    /* Configuration for connections and weights */
)
{ int k;
  for (k = 0; k<cf->N_wts; k++)
  { g[k] = 0;
  }
  add_grad2_config (g, s, off, d, cf);
}

#endif


/* --------------------------- store2_grad ---------------------------------- */

#if __CUDACC__

__device__ static void net_store2_grad1 
        (int, net_param *restrict, net_value const*, net_value const*, int);
__device__ static void net_store2_grad1_config 
        (int, net_param *restrict, net_value const*, net_value const*,
         net_config const*);
__device__ static void net_store2_grad2 
        (int, net_param *restrict, net_value const*,  net_value const*, 
         net_param const*, int, net_value const*, net_value const*, int,
         unsigned short const*, int);
__device__ static void net_store2_grad2_config 
        (int, net_param *restrict, net_value const*, net_value const*,
         net_param const*, net_value const*, net_value const*,
         net_config const*);


/* STORE SUM OF GRADIENT FROM A PAIR OF CASES, USING A PAIR OF THREADS.  
   Typically, one thread handles even index, the other odd indexes.  
   But sometimes only one thread is used. */

__device__ void net_store2_grad
( int th,		/* Which thread (0 or 1) */
  net_params *restrict g, /* Gradient with respect to parameters to store to */
  net_params const*w,	/* Network parameters (only offsets used) */
  net_values const*v0,	/* Values for units in network for case 0 */
  net_values const*v1,	/* Values for units in network for case 1 */
  net_values const*d0,	/* Backpropagated derivatives for case 0 */
  net_values const*d1,	/* Backpropagated derivatives for case 1 */
  net_arch const*a,	/* Network architecture */
  net_flags const*flgs 	/* Network flags, null if none */
)
{ 
  int l;
  if (a->has_ti) 
  { net_store2_grad1 (th, g->ti, d0->i, d1->i, a->N_inputs);
  }

  for (l = 0; l<a->N_layers; l++)
  { 
    int N_hidden = a->N_hidden[l];

    if (a->has_bh[l]) 
    { if (a->bias_config[l])
      { net_store2_grad1_config (th, g->bh[l], d0->s[l], d1->s[l], 
                                 a->bias_config[l]);
      }
      else
      { net_store2_grad1 (th, g->bh[l], d0->s[l], d1->s[l], N_hidden);
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { net_store2_grad2_config (th, g->ih[l], v0->i, v1->i, 
                                 a->has_ti ? w->ti : 0, 
                                 d0->s[l], d1->s[l], a->input_config[l]);
      }
      else
      { net_store2_grad2 (th, g->ih[l], v0->i, v1->i, a->has_ti ? w->ti : 0, 
                    a->N_inputs, d0->s[l], d1->s[l], N_hidden, 
                    flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1));
      }
    }

    if (l>0 && a->has_hh[l-1])
    { if (a->hidden_config[l])
      { net_store2_grad2_config
           (th, g->hh[l-1], v0->h[l-1], v1->h[l-1], 
            a->has_th[l-1] ? w->th[l-1] : 0,
            d0->s[l], d1->s[l], a->hidden_config[l]);
      }
      else
      { net_store2_grad2 (th, g->hh[l-1], v0->h[l-1], v1->h[l-1], 
          a->has_th[l-1] ? w->th[l-1] : 0,
          a->N_hidden[l-1], d0->s[l], d1->s[l], N_hidden,
          (unsigned short *)0, 0);
      }
    }

    if (a->has_th[l]) 
    { net_store2_grad1 (th, g->th[l], d0->h[l], d1->h[l], N_hidden);
    }

    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { net_store2_grad2_config (th, g->ho[l], v0->h[l], v1->h[l], 
                           a->has_th[l] ? w->th[l] : 0,
                           d0->o, d1->o, a->hidden_config[k]);
      }
      else
      { net_store2_grad2 (th, g->ho[l], v0->h[l], v1->h[l], 
                    a->has_th[l] ? w->th[l] : 0,
                    N_hidden, d0->o, d1->o, a->N_outputs, 
                    (unsigned short *) 0, 0);
      }
    }
  }

  if (a->has_io) 
  { if (a->input_config[a->N_layers])
    { net_store2_grad2_config (th, g->io, v0->i, v1->i, a->has_ti ? w->ti : 0, 
                         d0->o, d1->o, a->input_config[a->N_layers]);
    }
    else
    { net_store2_grad2 (th, g->io, v0->i, v1->i, a->has_ti ? w->ti : 0, 
                        a->N_inputs, d0->o, d1->o, a->N_outputs, 
                        flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0,
                        1);
    }
  }

  if (a->has_bo) 
  { if (a->bias_config[a->N_layers])
    { net_store2_grad1_config (th, g->bo, d0->o, d1->o, 
                               a->bias_config[a->N_layers]);
    }
    else
    { net_store2_grad1 (th, g->bo, d0->o, d1->o, a->N_outputs);
    }
  }
}


__device__ static void net_store2_grad1
( int th,		  /* Which thread (0 or 1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v0,    /* Derivatives with respect to unit values, case 0 */
  net_value const* v1,    /* Derivatives with respect to unit values, case 1 */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = th; i<n; i+=2)
  { g[i] = v0[i] + v1[i];
  }
}


__device__ static void net_store2_grad1_config
( int th,		  /* Which thread (0 or 1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v0,    /* Derivatives with respect to unit values, case 0 */
  net_value const* v1,    /* Derivatives with respect to unit values, case 1 */
  net_config const* cf    /* Configuration for biases */
)
{ net_connection *cn = cf->conn;
  int c, j, k;

  for (k = th; k<cf->N_wts; k+=2)
  { g[k] = 0;
  }

  if (th) return;  /* remainder done by a single thread */

  for (c = 0; (k = cn[c].w) >= 0; c++)
  { j = cn[c].d;
    g[k] += v0[j] + v1[j];
  }
}


#define NET_STORE2_GRAD2(offset,omit) \
do \
{ net_value o; \
  int i, j; \
  if (nd==1) \
  { net_value d00 = d0[0]; \
    net_value d10 = d1[0]; \
    if (th) \
    { i = 3; \
      while (i<nv) \
      { o = (offset); if (!(omit)) *g++ = (v0[i-3]+o)*d00 +  (v1[i-3]+o)*d10; \
        o = (offset); if (!(omit)) *g++ = (v0[i-2]+o)*d00 +  (v1[i-2]+o)*d10; \
        o = (offset); if (!(omit)) *g++ = (v0[i-1]+o)*d00 +  (v1[i-1]+o)*d10; \
        o = (offset); if (!(omit)) *g++ = (v0[i-0]+o)*d00 +  (v1[i-0]+o)*d10; \
        i += 4; \
      } \
      i -= 3; \
      while (i<nv) \
      { o = (offset); if (!(omit)) *g++ = (v0[i]+o)*d00 +  (v1[i]+o)*d10; \
        i += 1; \
      } \
    } \
  } \
  else \
  { for (i = 0; i<nv; i++) \
    { net_value tv0, tv1; \
      o = (offset); \
      if (omit) continue; \
      tv0 = v0[i] + o; \
      tv1 = v1[i] + o; \
      if (tv0==0 && tv1==0) \
      { continue; \
      } \
      if (tv1==0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv0 * d0[j-th-2]; \
          g[j-th-0] = tv0 * d0[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv0 * d0[j]; \
          j += 1; \
        } \
      } \
      else if (tv0==0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv1 * d1[j-th-2]; \
          g[j-th-0] = tv1 * d1[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv1 * d1[j]; \
          j += 1; \
        } \
      } \
      else \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv0 * d0[j-th-2] + tv1 * d1[j-th-2]; \
          g[j-th-0] = tv0 * d0[j-th-0] + tv1 * d1[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv0 * d0[j] + tv1 * d1[j]; \
          j += 1; \
        } \
      } \
      g += nd; \
    } \
  } \
} while (0)

#define NET_STORE2_GRAD2_00 \
do \
{ int i, j; \
  if (nd==1) \
  { net_value d00 = d0[0]; \
    net_value d10 = d1[0]; \
    i = 3; \
    while (i<nv) \
    { g[i-th-2] = v0[i-th-2] * d00 + v1[i-th-2] * d10; \
      g[i-th-0] = v0[i-th-0] * d00 + v1[i-th-0] * d10; \
      i += 4; \
    } \
    i -= 3; \
    while (i<nv) \
    { if (th) g[i] = v0[i] * d00 + v1[i] * d10; \
      i += 1; \
    } \
  } \
  else \
  { net_value tv0, tv1; \
    for (i = 0; i<nv; i++) \
    { tv0 = v0[i]; \
      tv1 = v1[i]; \
      if (tv0==0 && tv1==0) \
      { continue; \
      } \
      if (tv1==0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv0 * d0[j-th-2]; \
          g[j-th-0] = tv0 * d0[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv0 * d0[j]; \
          j += 1; \
        } \
      } \
      else if (tv0==0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv1 * d1[j-th-2]; \
          g[j-th-0] = tv1 * d1[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv1 * d1[j]; \
          j += 1; \
        } \
      } \
      else \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv0 * d0[j-th-2] + tv1 * d1[j-th-2]; \
          g[j-th-0] = tv0 * d0[j-th-0] + tv1 * d1[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv0 * d0[j] + tv1 * d1[j]; \
          j += 1; \
        } \
      } \
      g += nd; \
    } \
  } \
} while (0)

__device__ static void net_store2_grad2
( int th,		  /* Which thread (0 or 1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v0,    /* Source unit values, case 0 */
  net_value const* v1,    /* Source unit values, case 1 */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value const* d0,    /* Derivatives with respect to destination units, 0 */
  net_value const* d1,    /* Derivatives with respect to destination units, 1 */
  int nd,		  /* Number of destination units */
  unsigned short const* omit, /* Omit flags, null if not present */
  int ob		  /* Bit to look at in omit flags (mask, not number) */
)
{ 
  if (omit==0)
  { if (off==0)
    { NET_STORE2_GRAD2_00;
    }
    else
    { NET_STORE2_GRAD2(*off++,0);
    }
  }
  else
  { if (off==0)
    { NET_STORE2_GRAD2(0,(*omit++)&ob);
    }
    else
    { NET_STORE2_GRAD2(*off++,(*omit++)&ob);
    }
  }
}


__device__ static void net_store2_grad2_config
( int th,		  /* Which thread (0 or 1) */
  net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* s0,    /* Source unit values, case 0 */
  net_value const* s1,    /* Source unit values, case 1 */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  net_value const* d0,    /* Derivatives with respect to destination units, 0 */
  net_value const* d1,    /* Derivatives with respect to destination units, 1 */
  net_config const* cf    /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  for (k = th; k<cf->N_wts; k+=2)
  { g[k] = 0;
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w;
    if (off)
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { net_value soi0 = s0[cn[c].s] + off[cn[c].s];
        net_value soi1 = s1[cn[c].s] + off[cn[c].s];
        j = cn[c].d;
        g[k+th+0] += soi0 * d0[j+th+0] + soi1 * d1[j+th+0];
        g[k+th+2] += soi0 * d0[j+th+2] + soi1 * d1[j+th+2];
      }
    }
    else
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { net_value si0 = s0[cn[c].s];
        net_value si1 = s1[cn[c].s];
        j = cn[c].d;
        g[k+th+0] += si0 * d0[j+th+0] + si1 * d1[j+th+0];
        g[k+th+2] += si0 * d0[j+th+2] + si1 * d1[j+th+2];
      }
    }
  }

  if (th) return;  /* remainder done by a single thread */

  if (CONFIG_SINGLE4)
  { 
    cn = cf->single4_s;
    if (off)
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value soi0 = s0[cn[c].s] + off[cn[c].s];
        net_value soi1 = s1[cn[c].s] + off[cn[c].s];
        j = cn[c].d;
        g[k] += soi0 * d0[j] + soi1 * d1[j];
        j = cn[c+1].d; k = cn[c+1].w; 
        g[k] += soi0 * d0[j] + soi1 * d1[j];
        j = cn[c+2].d; k = cn[c+2].w; 
        g[k] += soi0 * d0[j] + soi1 * d1[j];
        j = cn[c+3].d; k = cn[c+3].w; 
        g[k] += soi0 * d0[j] + soi1 * d1[j];
      }
    }
    else
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value si0 = s0[cn[c].s];
        net_value si1 = s1[cn[c].s];
        j = cn[c].d;
        g[k] += si0 * d0[j] + si1 * d1[j];
        j = cn[c+1].d; k = cn[c+1].w; 
        g[k] += si0 * d0[j] + si1 * d1[j];
        j = cn[c+2].d; k = cn[c+2].w; 
        g[k] += si0 * d0[j] + si1 * d1[j];
        j = cn[c+3].d; k = cn[c+3].w; 
        g[k] += si0 * d0[j] + si1 * d1[j];
      }
    }

    cn = cf->single4_d;
    if (off)
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value dj0 = d0[cn[c].d];
        net_value dj1 = d1[cn[c].d];
        i = cn[c].s;
        g[k] += (s0[i]+off[i]) *dj0 + (s1[i]+off[i]) * dj1;
        i = cn[c+1].s; k = cn[c+1].w; 
        g[k] += (s0[i]+off[i]) *dj0 + (s1[i]+off[i]) * dj1;
        i = cn[c+2].s; k = cn[c+2].w; 
        g[k] += (s0[i]+off[i]) *dj0 + (s1[i]+off[i]) * dj1;
        i = cn[c+3].s; k = cn[c+3].w; 
        g[k] += (s0[i]+off[i]) *dj0 + (s1[i]+off[i]) * dj1;
      }
    }
    else
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value dj0 = d0[cn[c].d];
        net_value dj1 = d1[cn[c].d];
        i = cn[c].s;
        g[k] += s0[i] * dj0 + s1[i] * dj1;
        i = cn[c+1].s; k = cn[c+1].w; 
        g[k] += s0[i] * dj0 + s1[i] * dj1;
        i = cn[c+2].s; k = cn[c+2].w; 
        g[k] += s0[i] * dj0 + s1[i] * dj1;
        i = cn[c+3].s; k = cn[c+3].w; 
        g[k] += s0[i] * dj0 + s1[i] * dj1;
      }
    }
  }

  cn = CONFIG_ORIGINAL ? cf->conn : cf->single;
  if (off)
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { i = cn[c].s; j = cn[c].d;
      g[k] += (s0[i]+off[i]) * d0[j] + (s1[i]+off[i]) * d1[j];
    }
  }
  else
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { i = cn[c].s; j = cn[c].d;
      g[k] += s0[i] * d0[j] + s1[i] * d1[j];
    }
  }
}

#endif


/* --------------------------- store3_grad ---------------------------------- */

#if __CUDACC__

__device__ static void net_store3_grad1 (int, net_param *restrict, 
   net_value const*, net_value const*, net_value const*, int);
__device__ static void net_store3_grad1_config (int, net_param *restrict, 
   net_value const*, net_value const*, net_value const*, net_config const*);
__device__ static void net_store3_grad2 (int, net_param *restrict, 
   net_value const*, net_value const*, net_value const*, net_param const*, int,
   net_value const*, net_value const*, net_value const*, 
   int, unsigned short const*, int);
__device__ static void net_store3_grad2_config (int, net_param *restrict, 
   net_value const*, net_value const*, net_value const*, net_param const*,
   net_value const*, net_value const*, net_value const*, net_config const*);


/* STORE SUM OF GRADIENT FROM THREE CASES, USING A PAIR OF THREADS.  
   Typically, one thread handles even index, the other odd indexes.  
   But sometimes only one thread is used. */

__device__ void net_store3_grad
( int th,		/* Which thread (0 or 1) */
  net_params *restrict g, /* Gradient with respect to parameters to store to */
  net_params const*w,	/* Network parameters (only offsets used) */
  net_values const*v0,	/* Values for units in network for case 0 */
  net_values const*v1,	/* Values for units in network for case 1 */
  net_values const*v2,	/* Values for units in network for case 2 */
  net_values const*d0,	/* Backpropagated derivatives for case 0 */
  net_values const*d1,	/* Backpropagated derivatives for case 1 */
  net_values const*d2,	/* Backpropagated derivatives for case 2 */
  net_arch const*a,	/* Network architecture */
  net_flags const*flgs 	/* Network flags, null if none */
)
{ 
  int l;
  if (a->has_ti) 
  { net_store3_grad1 (th, g->ti, d0->i, d1->i, d2->i, a->N_inputs);
  }

  for (l = 0; l<a->N_layers; l++)
  { 
    int N_hidden = a->N_hidden[l];

    if (a->has_bh[l]) 
    { if (a->bias_config[l])
      { net_store3_grad1_config (th, g->bh[l], d0->s[l], d1->s[l], d2->s[l],
                                 a->bias_config[l]);
      }
      else
      { net_store3_grad1 (th, g->bh[l], d0->s[l], d1->s[l], d2->s[l], N_hidden);
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { net_store3_grad2_config (th, g->ih[l], v0->i, v1->i, v2->i,
                                 a->has_ti ? w->ti : 0, 
                                 d0->s[l], d1->s[l], d2->s[l], 
                                 a->input_config[l]);
      }
      else
      { net_store3_grad2 (th, g->ih[l], v0->i, v1->i, v2->i,
                    a->has_ti ? w->ti : 0, a->N_inputs, 
                    d0->s[l], d1->s[l], d2->s[l], N_hidden, 
                    flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1));
      }
    }

    if (l>0 && a->has_hh[l-1])
    { if (a->hidden_config[l])
      { net_store3_grad2_config
           (th, g->hh[l-1], v0->h[l-1], v1->h[l-1], v2->h[l-1],
            a->has_th[l-1] ? w->th[l-1] : 0,
            d0->s[l], d1->s[l], d2->s[l], a->hidden_config[l]);
      }
      else
      { net_store3_grad2 (th, g->hh[l-1], v0->h[l-1], v1->h[l-1], v2->h[l-1],
          a->has_th[l-1] ? w->th[l-1] : 0,
          a->N_hidden[l-1], d0->s[l], d1->s[l], d2->s[l], N_hidden,
          (unsigned short *)0, 0);
      }
    }

    if (a->has_th[l]) 
    { net_store3_grad1 (th, g->th[l], d0->h[l], d1->h[l], d2->h[l], N_hidden);
    }

    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { net_store3_grad2_config (th, g->ho[l], v0->h[l], v1->h[l], v2->h[l],
                           a->has_th[l] ? w->th[l] : 0,
                           d0->o, d1->o, d2->o, a->hidden_config[k]);
      }
      else
      { net_store3_grad2 (th, g->ho[l], v0->h[l], v1->h[l], v2->h[l],
                    a->has_th[l] ? w->th[l] : 0,
                    N_hidden, d0->o, d1->o, d2->o, a->N_outputs, 
                    (unsigned short *) 0, 0);
      }
    }
  }

  if (a->has_io) 
  { if (a->input_config[a->N_layers])
    { net_store3_grad2_config (th, g->io, v0->i, v1->i, v2->i, 
                               a->has_ti ? w->ti : 0, 
                               d0->o, d1->o, d2->o, 
                               a->input_config[a->N_layers]);
    }
    else
    { net_store3_grad2 (th, g->io, v0->i, v1->i, v2->i, a->has_ti ? w->ti : 0, 
                        a->N_inputs, d0->o, d1->o, d2->o, a->N_outputs, 
                        flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0,
                        1);
    }
  }

  if (a->has_bo) 
  { if (a->bias_config[a->N_layers])
    { net_store3_grad1_config (th, g->bo, d0->o, d1->o, d2->o,
                               a->bias_config[a->N_layers]);
    }
    else
    { net_store3_grad1 (th, g->bo, d0->o, d1->o, d2->o, a->N_outputs);
    }
  }
}


__device__ static void net_store3_grad1
( int th,		  /* Which thread (0 or 1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v0,    /* Derivatives with respect to unit values, case 0 */
  net_value const* v1,    /* Derivatives with respect to unit values, case 1 */
  net_value const* v2,    /* Derivatives with respect to unit values, case 2 */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = th; i<n; i+=2)
  { g[i] = v0[i] + v1[i] + v2[i];
  }
}


__device__ static void net_store3_grad1_config
( int th,		  /* Which thread (0 or 1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v0,    /* Derivatives with respect to unit values, case 0 */
  net_value const* v1,    /* Derivatives with respect to unit values, case 1 */
  net_value const* v2,    /* Derivatives with respect to unit values, case 2 */
  net_config const* cf    /* Configuration for biases */
)
{ net_connection *cn = cf->conn;
  int c, j, k;

  for (k = th; k<cf->N_wts; k+=2)
  { g[k] = 0;
  }

  if (th) return;  /* remainder done by a single thread */

  for (c = 0; (k = cn[c].w) >= 0; c++)
  { j = cn[c].d;
    g[k] += v0[j] + v1[j] + v2[j];
  }
}


#define NET_STORE3_GRAD2(offset,omit) \
do \
{ net_value o; \
  int i, j; \
  if (nd==1) \
  { net_value d00 = d0[0]; \
    net_value d10 = d1[0]; \
    net_value d20 = d2[0]; \
    if (th) \
    { i = 3; \
      while (i<nv) \
      { o = (offset); \
        if (!(omit)) *g++ = (v0[i-3]+o)*d00+(v1[i-3]+o)*d10+(v2[i-3]+o)*d20; \
        o = (offset); \
        if (!(omit)) *g++ = (v0[i-2]+o)*d00+(v1[i-2]+o)*d10+(v2[i-2]+o)*d20; \
        o = (offset); \
        if (!(omit)) *g++ = (v0[i-1]+o)*d00+(v1[i-1]+o)*d10+(v2[i-1]+o)*d20; \
        o = (offset); \
        if (!(omit)) *g++ = (v0[i-0]+o)*d00+(v1[i-0]+o)*d10+(v2[i-0]+o)*d20; \
        i += 4; \
      } \
      i -= 3; \
      while (i<nv) \
      { o = (offset); \
        if (!(omit)) *g++ = (v0[i]+o)*d00 + (v1[i]+o)*d10 + (v2[i]+o)*d20; \
        i += 1; \
      } \
    } \
  } \
  else \
  { for (i = 0; i<nv; i++) \
    { net_value tv0, tv1, tv2; \
      o = (offset); \
      if (omit) continue; \
      tv0 = v0[i] + o; \
      tv1 = v1[i] + o; \
      tv2 = v2[i] + o; \
      if (tv0==0 && tv1==0 && tv2==0) \
      { continue; \
      } \
      if (tv1==0 && tv2==0) \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv0 * d0[j-th-2]; \
          g[j-th-0] = tv0 * d0[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv0 * d0[j]; \
          j += 1; \
        } \
      } \
      else if (tv0==0 && tv2==0) \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv1 * d1[j-th-2]; \
          g[j-th-0] = tv1 * d1[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv1 * d1[j]; \
          j += 1; \
        } \
      } \
      else if (tv0==0 && tv1==0) \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv2 * d2[j-th-2]; \
          g[j-th-0] = tv2 * d2[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv2 * d2[j]; \
          j += 1; \
        } \
      } \
      else \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv0 * d0[j-th-2] + tv1 * d1[j-th-2] + tv2 * d2[j-th-2]; \
          g[j-th-0] = tv0 * d0[j-th-0] + tv1 * d1[j-th-0] + tv2 * d2[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv0 * d0[j] + tv1 * d1[j] + tv2 * d2[j]; \
          j += 1; \
        } \
      } \
      g += nd; \
    } \
  } \
} while (0)

#define NET_STORE3_GRAD2_00 \
do \
{ int i, j; \
  if (nd==1) \
  { net_value d00 = d0[0]; \
    net_value d10 = d1[0]; \
    net_value d20 = d2[0]; \
    i = 3; \
    while (i<nv) \
    { g[i-th-2] = v0[i-th-2] * d00 + v1[i-th-2] * d10 + v2[i-th-2] * d20; \
      g[i-th-0] = v0[i-th-0] * d00 + v1[i-th-0] * d10 + v2[i-th-0] * d20; \
      i += 4; \
    } \
    i -= 3; \
    while (i<nv) \
    { if (th) g[i] = v0[i] * d00 + v1[i] * d10 + v2[i] * d20; \
      i += 1; \
    } \
  } \
  else \
  { net_value tv0, tv1, tv2; \
    for (i = 0; i<nv; i++) \
    { tv0 = v0[i]; \
      tv1 = v1[i]; \
      tv2 = v2[i]; \
      if (tv0==0 && tv1==0 && tv2==0) \
      { continue; \
      } \
      if (tv1==0 && tv2==0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv0 * d0[j-th-2]; \
          g[j-th-0] = tv0 * d0[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv0 * d0[j]; \
          j += 1; \
        } \
      } \
      else if (tv0==0 && tv2==0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv1 * d1[j-th-2]; \
          g[j-th-0] = tv1 * d1[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv1 * d1[j]; \
          j += 1; \
        } \
      } \
      else if (tv0==0 && tv1==0)  \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv2 * d2[j-th-2]; \
          g[j-th-0] = tv2 * d2[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv2 * d2[j]; \
          j += 1; \
        } \
      } \
      else \
      { j = 3; \
        while (j<nd) \
        { g[j-th-2] = tv0 * d0[j-th-2] + tv1 * d1[j-th-2] + tv2 * d2[j-th-2]; \
          g[j-th-0] = tv0 * d0[j-th-0] + tv1 * d1[j-th-0] + tv2 * d2[j-th-0]; \
          j += 4; \
        } \
        j -= 3; \
        while (j<nd) \
        { if (th) g[j] = tv0 * d0[j] + tv1 * d1[j] + tv2 * d2[j]; \
          j += 1; \
        } \
      } \
      g += nd; \
    } \
  } \
} while (0)


__device__ static void net_store3_grad2
( int th,		  /* Which thread (0 or 1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v0,    /* Source unit values, case 0 */
  net_value const* v1,    /* Source unit values, case 1 */
  net_value const* v2,    /* Source unit values, case 2 */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value const* d0,    /* Derivatives with respect to destination units, 0 */
  net_value const* d1,    /* Derivatives with respect to destination units, 1 */
  net_value const* d2,    /* Derivatives with respect to destination units, 2 */
  int nd,		  /* Number of destination units */
  unsigned short const* omit, /* Omit flags, null if not present */
  int ob		  /* Bit to look at in omit flags (mask, not number) */
)
{ 
  if (omit==0)
  { if (off==0)
    { NET_STORE3_GRAD2_00;
    }
    else
    { NET_STORE3_GRAD2(*off++,0);
    }
  }
  else
  { if (off==0)
    { NET_STORE3_GRAD2(0,(*omit++)&ob);
    }
    else
    { NET_STORE3_GRAD2(*off++,(*omit++)&ob);
    }
  }
}


__device__ static void net_store3_grad2_config
( int th,		  /* Which thread (0 or 1) */
  net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* s0,    /* Source unit values, case 0 */
  net_value const* s1,    /* Source unit values, case 1 */
  net_value const* s2,    /* Source unit values, case 2 */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  net_value const* d0,    /* Derivatives with respect to destination units, 0 */
  net_value const* d1,    /* Derivatives with respect to destination units, 1 */
  net_value const* d2,    /* Derivatives with respect to destination units, 2 */
  net_config const* cf    /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  for (k = th; k<cf->N_wts; k+=2)
  { g[k] = 0;
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w;
    if (off)
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { net_value soi0 = s0[cn[c].s] + off[cn[c].s];
        net_value soi1 = s1[cn[c].s] + off[cn[c].s];
        net_value soi2 = s2[cn[c].s] + off[cn[c].s];
        j = cn[c].d;
        g[k+th+0] += soi0 * d0[j+th+0] + soi1 * d1[j+th+0] + soi2 * d2[j+th+0];
        g[k+th+2] += soi0 * d0[j+th+2] + soi1 * d1[j+th+2] + soi2 * d2[j+th+2];
      }
    }
    else
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { net_value si0 = s0[cn[c].s];
        net_value si1 = s1[cn[c].s];
        net_value si2 = s1[cn[c].s];
        j = cn[c].d;
        g[k+th+0] += si0 * d0[j+th+0] + si1 * d1[j+th+0] + si2 * d2[j+th+0];
        g[k+th+2] += si0 * d0[j+th+2] + si1 * d1[j+th+2] + si2 * d2[j+th+2];
      }
    }
  }

  if (th) return;  /* remainder done by a single thread */

  if (CONFIG_SINGLE4)
  { 
    cn = cf->single4_s;
    if (off)
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value soi0 = s0[cn[c].s] + off[cn[c].s];
        net_value soi1 = s1[cn[c].s] + off[cn[c].s];
        net_value soi2 = s2[cn[c].s] + off[cn[c].s];
        j = cn[c].d;
        g[k] += soi0 * d0[j] + soi1 * d1[j] + soi2 * d2[j];
        j = cn[c+1].d; k = cn[c+1].w; 
        g[k] += soi0 * d0[j] + soi1 * d1[j] + soi2 * d2[j];
        j = cn[c+2].d; k = cn[c+2].w; 
        g[k] += soi0 * d0[j] + soi1 * d1[j] + soi2 * d2[j];
        j = cn[c+3].d; k = cn[c+3].w; 
        g[k] += soi0 * d0[j] + soi1 * d1[j] + soi2 * d2[j];
      }
    }
    else
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value si0 = s0[cn[c].s];
        net_value si1 = s1[cn[c].s];
        net_value si2 = s2[cn[c].s];
        j = cn[c].d;
        g[k] += si0 * d0[j] + si1 * d1[j] + si2 * d2[j];
        j = cn[c+1].d; k = cn[c+1].w; 
        g[k] += si0 * d0[j] + si1 * d1[j] + si2 * d2[j];
        j = cn[c+2].d; k = cn[c+2].w; 
        g[k] += si0 * d0[j] + si1 * d1[j] + si2 * d2[j];
        j = cn[c+3].d; k = cn[c+3].w; 
        g[k] += si0 * d0[j] + si1 * d1[j] + si2 * d2[j];
      }
    }

    cn = cf->single4_d;
    if (off)
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value dj0 = d0[cn[c].d];
        net_value dj1 = d1[cn[c].d];
        net_value dj2 = d2[cn[c].d];
        i = cn[c].s;
        g[k] += (s0[i]+off[i])*dj0 + (s1[i]+off[i])*dj1 + (s2[i]+off[i])*dj2;
        i = cn[c+1].s; k = cn[c+1].w; 
        g[k] += (s0[i]+off[i])*dj0 + (s1[i]+off[i])*dj1 + (s2[i]+off[i])*dj2;
        i = cn[c+2].s; k = cn[c+2].w; 
        g[k] += (s0[i]+off[i])*dj0 + (s1[i]+off[i])*dj1 + (s2[i]+off[i])*dj2;
        i = cn[c+3].s; k = cn[c+3].w; 
        g[k] += (s0[i]+off[i])*dj0 + (s1[i]+off[i])*dj1 + (s2[i]+off[i])*dj2;
      }
    }
    else
    { for (c = 0; (k = cn[c].w) >= 0; c+=4)
      { net_value dj0 = d0[cn[c].d];
        net_value dj1 = d1[cn[c].d];
        net_value dj2 = d2[cn[c].d];
        i = cn[c].s;
        g[k] += s0[i] * dj0 + s1[i] * dj1 + s2[i] * dj2;
        i = cn[c+1].s; k = cn[c+1].w; 
        g[k] += s0[i] * dj0 + s1[i] * dj1 + s2[i] * dj2;
        i = cn[c+2].s; k = cn[c+2].w; 
        g[k] += s0[i] * dj0 + s1[i] * dj1 + s2[i] * dj2;
        i = cn[c+3].s; k = cn[c+3].w; 
        g[k] += s0[i] * dj0 + s1[i] * dj1 + s2[i] * dj2;
      }
    }
  }

  cn = CONFIG_ORIGINAL ? cf->conn : cf->single;
  if (off)
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { i = cn[c].s; j = cn[c].d;
      g[k] += (s0[i]+off[i])*d0[j]+(s1[i]+off[i])*d1[j]+(s2[i]+off[i])*d2[j];
    }
  }
  else
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { i = cn[c].s; j = cn[c].d;
      g[k] += s0[i] * d0[j] + s1[i] * d1[j] + s2[i] * d2[j];
    }
  }
}

#endif
