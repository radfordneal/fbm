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
   store a derivative from one training case or the sum of 2, 3, or 4 
   derivatives from 2, 3, or 4 training cases to a derivative structure
   are accessible to the GPU.
*/


/* ---------------------------- add_grad ------------------------------------ */

HOSTDEV static void add_grad1 (net_param *restrict, net_value const*, int);
HOSTDEV static void add_grad1_config (net_param *restrict, net_value const*,
                                      net_config const*);
HOSTDEV static void add_grad2 (net_param *restrict, net_value const*, 
                               net_param const*, int, net_value const*, int,
                               unsigned short const*, int, int);
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
  net_precomputed const* pre,  /* Precomputed aspects of architecture */
  net_flags const*flgs,	/* Network flags, null if none */
  int sparse            /* Might source unit values often be zero? */
)
{ 
  int l, ls, nsqi;

  if (a->has_ti) 
  { add_grad1 (g->ti, d->i, a->N_inputs);
  }

  for (l = 0; l<a->N_layers; l++)
  { 
    int N_hidden = a->N_hidden[l];

    if (a->has_bh[l]) 
    { if (a->bias_config[l])
      { add_grad1_config (g->bh[l], d->h[l], a->bias_config[l]);
      }
      else
      { add_grad1 (g->bh[l], d->h[l], N_hidden);
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { add_grad2_config (g->ih[l], v->i, a->has_ti ? w->ti : 0, d->h[l], 
                          a->input_config[l]);
      }
      else
      { add_grad2 (g->ih[l], v->i, a->has_ti ? w->ti : 0, a->N_inputs, 
                   d->h[l], N_hidden, 
                   flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1),
                   sparse);
      }
    }

    if (a->has_nsq[l])
    { for (ls = 0; ls<l; ls++)
      { nsqi = pre->nonseq[ls][l];
        if (nsqi>=0)
        { if (a->nonseq_config[nsqi])
          { add_grad2_config
                (g->nsq[nsqi], v->h[ls], a->has_th[ls] ? w->th[ls] : 0,
                d->h[l], a->nonseq_config[nsqi]);
          }
          else
          { add_grad2 (g->nsq[nsqi], v->h[ls], a->has_th[ls] ? w->th[ls] : 0,
              a->N_hidden[ls], d->h[l], N_hidden, (unsigned short *)0, 0, 0);
          }
        }
      }
    }

    if (l>0 && a->has_hh[l-1])
    { if (a->hidden_config[l])
      { add_grad2_config
           (g->hh[l-1], v->h[l-1], a->has_th[l-1] ? w->th[l-1] : 0,
            d->h[l], a->hidden_config[l]);
      }
      else
      { add_grad2 (g->hh[l-1], v->h[l-1], a->has_th[l-1] ? w->th[l-1] : 0,
          a->N_hidden[l-1], d->h[l], N_hidden, (unsigned short *)0, 0, 0);
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
                   N_hidden, d->o, a->N_outputs, (unsigned short *) 0, 0, 0);
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
                 flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0, 1,
                 sparse);
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
  net_value const* d,     /* Derivatives with respect to unit values */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = 0; i<n; i++)
  { g[i] += d[i];
  }
}


/* ADD TO GRADIENT FROM UNIT DERIVATIVE, WITH CONFIGURATION. */

HOSTDEV static void add_grad1_config
( net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* d,     /* Derivatives with respect to unit values */
  net_config const* cf    /* Configuration for biases */
)
{ net_connection *cn;
  int c, j, j2, k;

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w;
    for (c = 0; (k = cn[c].w) >= 0; c++)
    { j = cn[c].d;
      g[k+0] += d[j+0];
      g[k+1] += d[j+1];
      g[k+2] += d[j+2];
      g[k+3] += d[j+3];
    }
    cn = cf->quad_s_4d_4w_2;
    for (c = 0; (k = cn[c].w) >= 0; c+=2)
    { j = cn[c].d; j2 = cn[c+1].d;
      g[k+0] += d[j+0] + d[j2+0];
      g[k+1] += d[j+1] + d[j2+1];
      g[k+2] += d[j+2] + d[j2+2];
      g[k+3] += d[j+3] + d[j2+3];
    }
  }

  cn = CONFIG_ORIGINAL ? cf->conn : cf->single;
  for (c = 0; (k = cn[c].w) >= 0; c++)
  { j = cn[c].d;
    g[k] += d[j];
  }
}


/* ADD TO GRADIENT FROM PRODUCT OF UNIT VALUE AND UNIT DERIVATIVE. */

#define ADD_GRAD2(offset,omit,sprs) \
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
      if (!sprs || tv!=0)  \
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

#define ADD_GRAD2_00(one_more,done,sprs) \
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
        if (!sprs || _mm_ucomineq_sd (cast128d(TV), Z)) \
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
        if (!sprs || _mm_ucomineq_sd (cast128d(TV2), Z)) \
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

#define ADD_GRAD2_00(one_more,done,sprs) \
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
        if (!sprs || _mm_ucomineq_sd (TV, Z)) \
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
        if (!sprs || _mm_ucomineq_sd (TV2, Z)) \
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

#define ADD_GRAD2_00(one_more,done,sprs) \
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
        if (!sprs || _mm_ucomineq_ss (cast128f(TV), Z)) \
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
        if (!sprs || _mm_ucomineq_ss (cast128f(TV2), Z)) \
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

#define ADD_GRAD2_00(one_more,done,sprs) \
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
        if (!sprs || _mm_ucomineq_ss (TV, Z)) \
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
        if (!sprs || _mm_ucomineq_ss (TV2, Z)) \
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

#define ADD_GRAD2_00(one_more,done,sprs) \
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
      if (!sprs || tv!=0)  \
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
  int ob,		  /* Bit to look at in omit flags */
  int sparse              /* Might source unit values often be zero? */
)
{ 
  if (sparse && off==0)
  { if (omit==0)
    { ADD_GRAD2_00(one_more1,done1,1);
    }
    else
    { ADD_GRAD2(0,(*omit++)&ob,1);
    }
  }
  else 
  { if (omit==0)
    { if (off==0)
      { ADD_GRAD2_00(one_more2,done2,0);
      }
      else
      { ADD_GRAD2(*off++,0,0);
      }
    }
    else
    { if (off==0)
      { ADD_GRAD2(0,(*omit++)&ob,0);
      }
      else
      { ADD_GRAD2(*off++,(*omit++)&ob,0);
      }
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
    cn = cf->quad_s_4d_4w_2;
#   if FP64 && USE_SIMD_INTRINSICS && __AVX__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m256d GK = _mm256_loadu_pd(g+k);
          __m256d SI;
          SI = _mm256_set1_pd (s[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          GK = FMA256_pd (SI, _mm256_loadu_pd(d+j), GK);
          SI = _mm256_set1_pd (s[cn[c+1].s] + off[cn[c+1].s]);
          j = cn[c+1].d;
          GK = FMA256_pd (SI, _mm256_loadu_pd(d+j), GK);
          _mm256_storeu_pd (g+k, GK);
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m256d GK = _mm256_loadu_pd(g+k);
          __m256d SI;
          SI = _mm256_set1_pd (s[cn[c].s]);
          j = cn[c].d;
          GK = FMA256_pd (SI, _mm256_loadu_pd(d+j), GK);
          SI = _mm256_set1_pd (s[cn[c+1].s]);
          j = cn[c+1].d;
          GK = FMA256_pd (SI, _mm256_loadu_pd(d+j), GK);
          _mm256_storeu_pd (g+k, GK);
        }
      }
    }
#   elif FP64 && USE_SIMD_INTRINSICS && __SSE2__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m128d GK = _mm_loadu_pd(g+k), GK2 = _mm_loadu_pd(g+k+2);
          __m128d SI;
          SI = _mm_set1_pd (s[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          GK = FMA_pd (SI, _mm_loadu_pd(d+j), GK);
          GK2 = FMA_pd (SI, _mm_loadu_pd(d+j+2), GK2);
          SI = _mm_set1_pd (s[cn[c+1].s] + off[cn[c+1].s]);
          j = cn[c+1].d;
          GK = FMA_pd (SI, _mm_loadu_pd(d+j), GK);
          GK2 = FMA_pd (SI, _mm_loadu_pd(d+j+2), GK2);
          _mm_storeu_pd (g+k, GK);
          _mm_storeu_pd (g+k+2, GK2);
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m128d GK = _mm_loadu_pd(g+k), GK2 = _mm_loadu_pd(g+k+2);
          __m128d SI;
          SI = _mm_set1_pd (s[cn[c].s]);
          j = cn[c].d;
          GK = FMA_pd (SI, _mm_loadu_pd(d+j), GK);
          GK2 = FMA_pd (SI, _mm_loadu_pd(d+j+2), GK2);
          SI = _mm_set1_pd (s[cn[c+1].s]);
          j = cn[c+1].d;
          GK = FMA_pd (SI, _mm_loadu_pd(d+j), GK);
          GK2 = FMA_pd (SI, _mm_loadu_pd(d+j+2), GK2);
          _mm_storeu_pd (g+k, GK);
          _mm_storeu_pd (g+k+2, GK2);
        }
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __SSE2__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m128 GK = _mm_loadu_ps(g+k);
          __m128 SI;
          SI = _mm_set1_ps (s[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          GK = FMA_ps(SI, _mm_loadu_ps(d+j), GK);
          SI = _mm_set1_ps (s[cn[c+1].s] + off[cn[c+1].s]);
          j = cn[c+1].d;
          GK = FMA_ps(SI, _mm_loadu_ps(d+j), GK);
          _mm_storeu_ps (g+k, GK);
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { __m128 GK = _mm_loadu_ps(g+k);
          __m128 SI;
          SI = _mm_set1_ps (s[cn[c].s]);
          j = cn[c].d;
          GK = FMA_ps(SI, _mm_loadu_ps(d+j), GK);
          SI = _mm_set1_ps (s[cn[c+1].s]);
          j = cn[c+1].d;
          GK = FMA_ps(SI, _mm_loadu_ps(d+j), GK);
          _mm_storeu_ps (g+k, GK);
        }
      }
    }
#   else
    { int j2;
      if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { net_value soi = s[cn[c].s] + off[cn[c].s];
          j = cn[c].d;
          net_value soi2 = s[cn[c+1].s] + off[cn[c+1].s];
          j2 = cn[c+1].d;
          g[k+0] = g[k+0] + soi * d[j+0] + soi2 * d[j2+0];  /* not using +=  */
          g[k+1] = g[k+1] + soi * d[j+1] + soi2 * d[j2+1];  /* allows use of */
          g[k+2] = g[k+2] + soi * d[j+2] + soi2 * d[j2+2];  /* multiply-add, */
          g[k+3] = g[k+3] + soi * d[j+3] + soi2 * d[j2+3];  /* matching AVX  */
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c+=2)
        { net_value si = s[cn[c].s];
          j = cn[c].d;
          net_value si2 = s[cn[c+1].s];
          j2 = cn[c+1].d;
          g[k+0] = g[k+0] + si * d[j+0] + si2 * d[j2+0];  /* not using +=  */
          g[k+1] = g[k+1] + si * d[j+1] + si2 * d[j2+1];  /* allows use of */
          g[k+2] = g[k+2] + si * d[j+2] + si2 * d[j2+2];  /* multiply-add, */
          g[k+3] = g[k+3] + si * d[j+3] + si2 * d[j2+3];  /* matching AVX  */
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


/* --------------------------- store1_grad ---------------------------------- */

#if __CUDACC__ 

#define FASTMEM(o,w) \
  (sharedvalues + ((w)+(threadIdx.x/GTH)*GROUP_SIZE)*pre->memused + (o))

#define HIDLOC(pre,layer,values,w) \
  ((pre)->fwgpumem[layer]>=0 ? FASTMEM(pre->fwgpumem[layer],(w)) \
                             : ((values)+(w))->h[layer])

__device__ static void net_store1_grad1 
        (int, net_param *restrict, net_value const*, int);
__device__ static void net_store1_grad1_config 
        (int, net_param *restrict, net_value const*,
         net_config const*);
__device__ static void net_store1_grad2 
        (int, net_param *restrict,  net_value const*, 
         net_param const*, int, net_value const*, int,
         unsigned short const*, int, int);
__device__ static void net_store1_grad2_config 
        (int, net_param *restrict, net_value const*,
         net_param const*, net_value const*,
         net_config const*);


/* STORE GRADIENT FROM A CASE.  Threads handle indexes equal to their
   'th' mod GTH.  The exact meaning of this may depend on the kind of
   parameter group.  This eliminates any need for thread
   synchronization, even if the gradient is produced incrementally by
   adding several terms, since there is no overlap in the set of
   places the two threads store to.  Such a consistent may also
   improve performance.

   Assumes any thread synchronization has been done that's needed to
   make derivatives with respect to unit values accessible to all the
   threads executing here.
*/

__device__ void net_store1_grad
( int th,		/* Which thread (0 to GTH-1) */
  net_params *restrict g, /* Gradient with respect to parameters to store to */
  net_params const*w,	/* Network parameters (only offsets used) */
  net_values const*v0,	/* Values for units in network for case */
  net_values const*d0,	/* Backpropagated derivatives for case */
  net_arch const*a,	/* Network architecture */
  net_precomputed const* pre,  /* Precomputed aspects of architecture */
  net_flags const*flgs,	/* Network flags, null if none */
  int sparse            /* Might source unit values often be zero? */
)
{ 
  const net_value *u0;
  int l, ls, nsqi;

  if (a->has_ti) 
  { net_store1_grad1 (th, g->ti, d0->i, a->N_inputs);
  }

  for (l = 0; l<a->N_layers; l++)
  { 
    int N_hidden = a->N_hidden[l];

    if (a->has_bh[l]) 
    { if (a->bias_config[l])
      { net_store1_grad1_config (th, g->bh[l], d0->h[l],
                                 a->bias_config[l]);
      }
      else
      { net_store1_grad1 (th, g->bh[l], d0->h[l], N_hidden);
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { net_store1_grad2_config (th, g->ih[l], v0->i,
                                 a->has_ti ? w->ti : 0, 
                                 d0->h[l], a->input_config[l]);
      }
      else
      { net_store1_grad2 (th, g->ih[l], v0->i, a->has_ti ? w->ti : 0, 
                    a->N_inputs, d0->h[l], N_hidden, 
                    flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1),
                    sparse);
      }
    }

    if (a->has_nsq[l])
    { for (ls = 0; ls<l; ls++)
      { u0 = HIDLOC(pre,ls,v0,0);
        nsqi = pre->nonseq[ls][l];
        if (nsqi>=0)
        { if (a->nonseq_config[nsqi])
          { net_store1_grad2_config
                (th, g->nsq[nsqi], u0,
                 a->has_th[ls] ? w->th[ls] : 0,
                 d0->h[l], a->nonseq_config[nsqi]);
          }
          else
          { net_store1_grad2 (th, g->nsq[nsqi], u0,
              a->has_th[ls] ? w->th[ls] : 0,
              a->N_hidden[ls], d0->h[l], N_hidden, 
              (unsigned short *)0, 0, 0);
          }
        }
      }
    }

    if (l>0 && a->has_hh[l-1])
    { u0 = HIDLOC(pre,l-1,v0,0);
      if (a->hidden_config[l])
      { net_store1_grad2_config
           (th, g->hh[l-1], u0,
            a->has_th[l-1] ? w->th[l-1] : 0,
            d0->h[l], a->hidden_config[l]);
      }
      else
      { net_store1_grad2 (th, g->hh[l-1], u0,
          a->has_th[l-1] ? w->th[l-1] : 0,
          a->N_hidden[l-1], d0->h[l], N_hidden,
          (unsigned short *)0, 0, 0);
      }
    }

    if (a->has_th[l]) 
    { net_store1_grad1 (th, g->th[l], d0->h[l], N_hidden);
    }

    if (a->has_ho[l])
    { u0 = HIDLOC(pre,l,v0,0);
      int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { net_store1_grad2_config (th, g->ho[l], u0,
                           a->has_th[l] ? w->th[l] : 0,
                           d0->o, a->hidden_config[k]);
      }
      else
      { net_store1_grad2 (th, g->ho[l], u0,
                    a->has_th[l] ? w->th[l] : 0,
                    N_hidden, d0->o, a->N_outputs, 
                    (unsigned short *) 0, 0, 0);
      }
    }
  }

  if (a->has_io) 
  { if (a->input_config[a->N_layers])
    { net_store1_grad2_config (th, g->io, v0->i, a->has_ti ? w->ti : 0, 
                         d0->o, a->input_config[a->N_layers]);
    }
    else
    { net_store1_grad2 (th, g->io, v0->i, a->has_ti ? w->ti : 0, 
                        a->N_inputs, d0->o, a->N_outputs, 
                        flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0,
                        1, sparse);
    }
  }

  if (a->has_bo) 
  { if (a->bias_config[a->N_layers])
    { net_store1_grad1_config (th, g->bo, d0->o,
                               a->bias_config[a->N_layers]);
    }
    else
    { net_store1_grad1 (th, g->bo, d0->o, a->N_outputs);
    }
  }
}


/* STORE GRADIENT FOR BIASES FOR CASE.  The thread mod scheme is
   based on indexes for the biases/destination units. */

__device__ static void net_store1_grad1
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* d0,    /* Derivatives with respect to unit values */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = th; i<n; i+=GTH)
  { g[i] = d0[i];
  }
}


/* STORE GRADIENT FOR BIASES FOR CASE, WITH CONFIGURATION.  The
   thread mod scheme is based on indexes for the biases.  Note that
   the connections in quad_s_4d_4w_wgpu, other_wgpu, and other_2_wgpu
   come in GTH sections. */

__device__ static void net_store1_grad1_config
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* d0,    /* Derivatives with respect to unit values */
  net_config const* cf    /* Configuration for biases */
)
{ net_connection *cn;
  int c, j, j2, k, m, ix;
  int thmod4 = th&3;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[k] = 0;
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_wgpu;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m+4) & 3;
      c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
      for (;;)
      { j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        g[k+ix] += d0[j+ix];
      }
    }
    cn = cf->quad_s_4d_4w_2_wgpu;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m+4) & 3;
      c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
      for (;;)
      { j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        j2 = cn[c].d; c += 1;
        g[k+ix] += d0[j+ix] + d0[j2+ix];
      }
    }
  }

  cn = cf->other_wgpu;
  c = cf->start_other_wgpu[th];
  for (;;)
  { j = cn[c].d; k = cn[c].w; c += 1;
    if (k<0) break;
    g[k] += d0[j];
  }
  cn = cf->other_2_wgpu;
  c = cf->start_other_2_wgpu[th];
  for (;;)
  { j = cn[c].d; k = cn[c].w; c += 1;
    if (k<0) break;
    j2 = cn[c].d; c += 1;
    g[k] += d0[j] + d0[j2];
  }
}


/* STORE GRADIENT FOR WEIGHTS FOR CASE.  The thread mod scheme is
   based on the indexes for the destination units, unless there is
   only one destination unit, in which case it is based on the indexes
   for the source units. */

#define NET_STORE1_GRAD2(has_off,has_omit,sprs) \
do \
{ int i; \
  if (nd==1 && !has_omit) \
  { net_value d00 = d0[0]; \
    if (has_off) \
    { net_value o; \
      for (i = th; i<nv; i+=GTH) \
      { o = off[i]; \
        g[i] = (v0[i]+o)*d00; \
      } \
    } \
    else \
    { for (i = th; i<nv; i+=GTH) \
      { g[i] = v0[i]*d00; \
      } \
    } \
  } \
  else if (sprs) \
  { net_value tv0, o; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      tv0 = v0[i] + o; \
      if (tv0==0) \
      { for (j = th; j<nd; j+=GTH) \
        { g[j] = 0; \
        } \
      } \
      else \
      { for (j = th; j<nd; j+=GTH) \
        { g[j] = tv0*d0[j]; \
        } \
      } \
      g += nd; \
    } \
  } \
  else \
  { net_value tv0, o; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      tv0 = v0[i] + o; \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = tv0*d0[j]; \
      } \
      g += nd; \
    } \
  } \
} while (0)

__device__ static void net_store1_grad2
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v0,    /* Source unit values */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value const* d0,    /* Derivatives with respect to destination units */
  int nd,		  /* Number of destination units */
  unsigned short const* omit, /* Omit flags, null if not present */
  int ob,		  /* Bit to look at in omit flags (mask, not number) */
  int sparse              /* Might source unit values often be zero? */
)
{ 
  if (sparse && off==0)
  { if (omit==0)
    { NET_STORE1_GRAD2(0,0,1);
    }
    else
    { NET_STORE1_GRAD2(0,1,1);
    }
  }
  else
  { if (omit==0)
    { if (off==0)
      { NET_STORE1_GRAD2(0,0,0);
      }
      else
      { NET_STORE1_GRAD2(1,0,0);
      }
    }
    else
    { if (off==0)
      { NET_STORE1_GRAD2(0,1,0);
      }
      else
      { NET_STORE1_GRAD2(1,1,0);
      }
    }
  }
}


/* STORE GRADIENT FOR WEIGHTS FOR 2 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on the indexes for the weights.  Note
   that the connections in quad_s_4d_4w_wgpu, other_wgpu, and
   other_2_wgpu come in GTH sections. */

__device__ static void net_store1_grad2_config
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* s0,    /* Source unit values */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  net_value const* d0,    /* Derivatives with respect to destination units */
  net_config const* cf    /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, i2, j, j2, k, c, m, ix;
  int thmod4 = th&3;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[k] = 0;
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_wgpu;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i];
          net_param o = off[i];
          g[k+ix] = g[k+ix] + (s0i+o)*d0[j+ix];
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i];
          g[k+ix] = g[k+ix] + s0i*d0[j+ix];
        }
      }
    }
    cn = cf->quad_s_4d_4w_2_wgpu;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i];
          net_param o = off[i];
          i2 = cn[c].s; j2 = cn[c].d; c += 1;
          net_value s0i2 = s0[i2];
          net_param o2 = off[i2];
          g[k+ix] = g[k+ix] + (s0i+o)*d0[j+ix]
                            + (s0i2+o2)*d0[j2+ix];
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i];
          i2 = cn[c].s; j2 = cn[c].d; c += 1;
          net_value s0i2 = s0[i2];
          g[k+ix] = g[k+ix] + s0i*d0[j+ix]
                            + s0i2*d0[j2+ix];
        }
      }
    }
  }

  cn = cf->other_wgpu;
  c = cf->start_other_wgpu[th];
  if (off)
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      net_param o = off[i];
      g[k] = g[k] + (s0[i]+o)*d0[j];
    }
  }
  else
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      g[k] = g[k] + s0[i]*d0[j];
    }
  }
  cn = cf->other_2_wgpu;
  c = cf->start_other_2_wgpu[th];
  if (off)
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      net_param o = off[i];
      i2 = cn[c].s; j2 = cn[c].d; c += 1;
      net_param o2 = off[i2];
      g[k] = g[k] + (s0[i]+o)*d0[j]
                  + (s0[i2]+o2)*d0[j2];
    }
  }
  else
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      i2 = cn[c].s; j2 = cn[c].d; c += 1;
      g[k] = g[k] + s0[i]*d0[j]
                  + s0[i2]*d0[j2];
    }
  }
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
         unsigned short const*, int, int);
__device__ static void net_store2_grad2_config 
        (int, net_param *restrict, net_value const*, net_value const*,
         net_param const*, net_value const*, net_value const*,
         net_config const*);


/* STORE SUM OF GRADIENT FROM A PAIR OF CASES.  Threads handle indexes
   equal to their 'th' mod GTH.  The exact meaning of this may depend
   on the kind of parameter group.  This eliminates any need for
   thread synchronization, even if the gradient is produced
   incrementally by adding several terms, since there is no overlap in
   the set of places the two threads store to.  Such a consistent may
   also improve performance.

   Assumes any thread synchronization has been done that's needed to
   make derivatives with respect to unit values accessible to all the
   threads executing here.
*/

__device__ void net_store2_grad
( int th,		/* Which thread (0 to GTH-1) */
  net_params *restrict g, /* Gradient with respect to parameters to store to */
  net_params const*w,	/* Network parameters (only offsets used) */
  net_values const*v0,	/* Values for units in network for case 0, rest follow*/
  net_values const*d0,	/* Backpropagated derivatives for case 0, rest follow */
  net_arch const*a,	/* Network architecture */
  net_precomputed const* pre,  /* Precomputed aspects of architecture */
  net_flags const*flgs,	/* Network flags, null if none */
  int sparse            /* Might source unit values often be zero? */
)
{ 
  const net_values *d1 = d0+1;
  const net_values *v1 = v0+1;
  const net_value *u0, *u1;
  int l, ls, nsqi;

  if (a->has_ti) 
  { net_store2_grad1 (th, g->ti, d0->i, d1->i, a->N_inputs);
  }

  for (l = 0; l<a->N_layers; l++)
  { 
    int N_hidden = a->N_hidden[l];

    if (a->has_bh[l]) 
    { if (a->bias_config[l])
      { net_store2_grad1_config (th, g->bh[l], d0->h[l], d1->h[l], 
                                 a->bias_config[l]);
      }
      else
      { net_store2_grad1 (th, g->bh[l], d0->h[l], d1->h[l], N_hidden);
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { net_store2_grad2_config (th, g->ih[l], v0->i, v1->i, 
                                 a->has_ti ? w->ti : 0, 
                                 d0->h[l], d1->h[l], a->input_config[l]);
      }
      else
      { net_store2_grad2 (th, g->ih[l], v0->i, v1->i, a->has_ti ? w->ti : 0, 
                    a->N_inputs, d0->h[l], d1->h[l], N_hidden, 
                    flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1),
                    sparse);
      }
    }

    if (a->has_nsq[l])
    { for (ls = 0; ls<l; ls++)
      { u0 = HIDLOC(pre,ls,v0,0); u1 = HIDLOC(pre,ls,v0,1); 
        nsqi = pre->nonseq[ls][l];
        if (nsqi>=0)
        { if (a->nonseq_config[nsqi])
          { net_store2_grad2_config
                (th, g->nsq[nsqi], u0, u1,
                 a->has_th[ls] ? w->th[ls] : 0,
                 d0->h[l], d1->h[l], a->nonseq_config[nsqi]);
          }
          else
          { net_store2_grad2 (th, g->nsq[nsqi], u0, u1,
              a->has_th[ls] ? w->th[ls] : 0,
              a->N_hidden[ls], d0->h[l], d1->h[l], N_hidden, 
              (unsigned short *)0, 0, 0);
          }
        }
      }
    }

    if (l>0 && a->has_hh[l-1])
    { u0 = HIDLOC(pre,l-1,v0,0); u1 = HIDLOC(pre,l-1,v0,1); 
      if (a->hidden_config[l])
      { net_store2_grad2_config
           (th, g->hh[l-1], u0, u1, 
            a->has_th[l-1] ? w->th[l-1] : 0,
            d0->h[l], d1->h[l], a->hidden_config[l]);
      }
      else
      { net_store2_grad2 (th, g->hh[l-1], u0, u1, 
          a->has_th[l-1] ? w->th[l-1] : 0,
          a->N_hidden[l-1], d0->h[l], d1->h[l], N_hidden,
          (unsigned short *)0, 0, 0);
      }
    }

    if (a->has_th[l]) 
    { net_store2_grad1 (th, g->th[l], d0->h[l], d1->h[l], N_hidden);
    }

    if (a->has_ho[l])
    { u0 = HIDLOC(pre,l,v0,0); u1 = HIDLOC(pre,l,v0,1); 
      int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { net_store2_grad2_config (th, g->ho[l], u0, u1, 
                           a->has_th[l] ? w->th[l] : 0,
                           d0->o, d1->o, a->hidden_config[k]);
      }
      else
      { net_store2_grad2 (th, g->ho[l], u0, u1, 
                    a->has_th[l] ? w->th[l] : 0,
                    N_hidden, d0->o, d1->o, a->N_outputs, 
                    (unsigned short *) 0, 0, 0);
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
                        1, sparse);
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


/* STORE GRADIENT FOR BIASES FOR 2 CASES.  The thread mod scheme is
   based on indexes for the biases/destination units. */

__device__ static void net_store2_grad1
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* d0,    /* Derivatives with respect to unit values, case 0 */
  net_value const* d1,    /* Derivatives with respect to unit values, case 1 */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = th; i<n; i+=GTH)
  { g[i] = d0[i] + d1[i];
  }
}


/* STORE GRADIENT FOR BIASES FOR 2 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on indexes for the biases.  Note that
   the connections in quad_s_4d_4w_wgpu, other_wgpu, and other_2_wgpu
   come in GTH sections. */

__device__ static void net_store2_grad1_config
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* d0,    /* Derivatives with respect to unit values, case 0 */
  net_value const* d1,    /* Derivatives with respect to unit values, case 1 */
  net_config const* cf    /* Configuration for biases */
)
{ net_connection *cn;
  int c, j, j2, k, m, ix;
  int thmod4 = th&3;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[k] = 0;
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_wgpu;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m+4) & 3;
      c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
      for (;;)
      { j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        g[k+ix] += d0[j+ix] + d1[j+ix];
      }
    }
    cn = cf->quad_s_4d_4w_2_wgpu;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m+4) & 3;
      c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
      for (;;)
      { j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        j2 = cn[c].d; c += 1;
        g[k+ix] += (d0[j+ix] + d1[j+ix]) + (d0[j2+ix] + d1[j2+ix]);
      }
    }
  }

  cn = cf->other_wgpu;
  c = cf->start_other_wgpu[th];
  for (;;)
  { j = cn[c].d; k = cn[c].w; c += 1;
    if (k<0) break;
    g[k] += d0[j] + d1[j];
  }
  cn = cf->other_2_wgpu;
  c = cf->start_other_2_wgpu[th];
  for (;;)
  { j = cn[c].d; k = cn[c].w; c += 1;
    if (k<0) break;
    j2 = cn[c].d; c += 1;
    g[k] += (d0[j] + d1[j]) + (d0[j2] + d1[j2]);
  }
}


/* STORE GRADIENT FOR WEIGHTS FOR 2 CASES.  The thread mod scheme is
   based on the indexes for the destination units, unless there is
   only one destination unit, in which case it is based on the indexes
   for the source units. */

#define NET_STORE2_GRAD2(has_off,has_omit,alllab,onelab,sprs) \
do \
{ int i; \
  if (nd==1 && !has_omit) \
  { net_value d00 = d0[0]; \
    net_value d10 = d1[0]; \
    if (has_off) \
    { net_value o; \
      for (i = th; i<nv; i+=GTH) \
      { o = off[i]; \
        g[i] = (v0[i]+o)*d00 + (v1[i]+o)*d10; \
      } \
    } \
    else \
    { for (i = th; i<nv; i+=GTH) \
      { g[i] = v0[i]*d00 + v1[i]*d10; \
      } \
    } \
  } \
  else if (sprs) \
  { net_value tv0, tv1, tvh, o; \
    net_value const*dh; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      tv0 = v0[i] + o; \
      tv1 = v1[i] + o; \
      if (tv0!=0) \
      { if (tv1!=0) goto alllab; \
        tvh = tv0; dh = d0; \
        goto onelab; \
      } \
      else if (tv1!=0) \
      { tvh = tv1; dh = d1; \
        goto onelab; \
      } \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = 0; \
      } \
      g += nd; \
      continue; \
    onelab: \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = tvh * dh[j]; \
      } \
      g += nd; \
      continue; \
    alllab: \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = tv0*d0[j] + tv1*d1[j]; \
      } \
      g += nd; \
    } \
  } \
  else \
  { net_value tv0, tv1, o; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      tv0 = v0[i] + o; \
      tv1 = v1[i] + o; \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = tv0*d0[j] + tv1*d1[j]; \
      } \
      g += nd; \
    } \
  } \
} while (0)

__device__ static void net_store2_grad2
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v0,    /* Source unit values, case 0 */
  net_value const* v1,    /* Source unit values, case 1 */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value const* d0,    /* Derivatives with respect to destination units, 0 */
  net_value const* d1,    /* Derivatives with respect to destination units, 1 */
  int nd,		  /* Number of destination units */
  unsigned short const* omit, /* Omit flags, null if not present */
  int ob,		  /* Bit to look at in omit flags (mask, not number) */
  int sparse              /* Might source unit values often be zero? */
)
{ 
  if (sparse && off==0)
  { if (omit==0)
    { NET_STORE2_GRAD2(0,0,all1s,one1s,1);
    }
    else
    { NET_STORE2_GRAD2(0,1,all3s,one3s,1);
    }
  }
  else
  { if (omit==0)
    { if (off==0)
      { NET_STORE2_GRAD2(0,0,all1,one1,0);
      }
      else
      { NET_STORE2_GRAD2(1,0,all2,one2,0);
      }
    }
    else
    { if (off==0)
      { NET_STORE2_GRAD2(0,1,all3,one3,0);
      }
      else
      { NET_STORE2_GRAD2(1,1,all4,one4,0);
      }
    }
  }
}


/* STORE GRADIENT FOR WEIGHTS FOR 2 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on the indexes for the weights.  Note
   that the connections in quad_s_4d_4w_wgpu, other_wgpu, and
   other_2_wgpu come in GTH sections. */

__device__ static void net_store2_grad2_config
( int th,		  /* Which thread (0 to GTH-1) */
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
  int i, i2, j, j2, k, c, m, ix;
  int thmod4 = th&3;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[k] = 0;
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_wgpu;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i], s1i = s1[i];
          net_param o = off[i];
          g[k+ix] = g[k+ix] + (s0i+o)*d0[j+ix] + (s1i+o)*d1[j+ix];
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i], s1i = s1[i];
          g[k+ix] = g[k+ix] + s0i*d0[j+ix] + s1i*d1[j+ix];
        }
      }
    }
    cn = cf->quad_s_4d_4w_2_wgpu;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i], s1i = s1[i];
          net_param o = off[i];
          i2 = cn[c].s; j2 = cn[c].d; c += 1;
          net_value s0i2 = s0[i2], s1i2 = s1[i2];
          net_param o2 = off[i2];
          g[k+ix] = g[k+ix] + (s0i+o)*d0[j+ix] + (s1i+o)*d1[j+ix]
                            + (s0i2+o2)*d0[j2+ix] + (s1i2+o2)*d1[j2+ix];
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i], s1i = s1[i];
          i2 = cn[c].s; j2 = cn[c].d; c += 1;
          net_value s0i2 = s0[i2], s1i2 = s1[i2];
          g[k+ix] = g[k+ix] + s0i*d0[j+ix] + s1i*d1[j+ix]
                            + s0i2*d0[j2+ix] + s1i2*d1[j2+ix];
        }
      }
    }
  }

  cn = cf->other_wgpu;
  c = cf->start_other_wgpu[th];
  if (off)
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      net_param o = off[i];
      g[k] = g[k] + (s0[i]+o)*d0[j] + (s1[i]+o)*d1[j];
    }
  }
  else
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      g[k] = g[k] + s0[i]*d0[j] + s1[i]*d1[j];
    }
  }
  cn = cf->other_2_wgpu;
  c = cf->start_other_2_wgpu[th];
  if (off)
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      net_param o = off[i];
      i2 = cn[c].s; j2 = cn[c].d; c += 1;
      net_param o2 = off[i2];
      g[k] = g[k] + (s0[i]+o)*d0[j] + (s1[i]+o)*d1[j]
                  + (s0[i2]+o2)*d0[j2] + (s1[i2]+o2)*d1[j2];
    }
  }
  else
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      i2 = cn[c].s; j2 = cn[c].d; c += 1;
      g[k] = g[k] + s0[i]*d0[j] + s1[i]*d1[j]
                  + s0[i2]*d0[j2] + s1[i2]*d1[j2];
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
   int, unsigned short const*, int, int);
__device__ static void net_store3_grad2_config (int, net_param *restrict, 
   net_value const*, net_value const*, net_value const*, net_param const*,
   net_value const*, net_value const*, net_value const*, net_config const*);


/* STORE SUM OF GRADIENT FROM THREE CASES.  Threads handle indexes
   equal to their 'th' mod GTH.  The exact meaning of this may depend
   on the kind of parameter group.  This eliminates any need for
   thread synchronization, even if the gradient is produced
   incrementally by adding several terms, since there is no overlap in
   the set of places the two threads store to.  Such a consistent may
   also improve performance.

   Assumes any thread synchronization has been done that's needed to
   make derivatives with respect to unit values accessible to all the
   threads executing here.
*/


__device__ void net_store3_grad
( int th,		/* Which thread (0 to GTH-1) */
  net_params *restrict g, /* Gradient with respect to parameters to store to */
  net_params const*w,	/* Network parameters (only offsets used) */
  net_values const*v0,	/* Values for units in network for case 0, rest follow*/
  net_values const*d0,	/* Backpropagated derivatives for case 0, rest follow */
  net_arch const*a,	/* Network architecture */
  net_precomputed const* pre,  /* Precomputed aspects of architecture */
  net_flags const*flgs,	/* Network flags, null if none */
  int sparse            /* Might source unit values often be zero? */
)
{ 
  const net_values *d1 = d0+1, *d2 = d1+1;
  const net_values *v1 = v0+1, *v2 = v1+1;
  const net_value *u0, *u1, *u2;
  int l, ls, nsqi;

  if (a->has_ti) 
  { net_store3_grad1 (th, g->ti, d0->i, d1->i, d2->i, a->N_inputs);
  }

  for (l = 0; l<a->N_layers; l++)
  { 
    int N_hidden = a->N_hidden[l];

    if (a->has_bh[l]) 
    { if (a->bias_config[l])
      { net_store3_grad1_config (th, g->bh[l], d0->h[l], d1->h[l], d2->h[l],
                                 a->bias_config[l]);
      }
      else
      { net_store3_grad1 (th, g->bh[l], d0->h[l], d1->h[l], d2->h[l], N_hidden);
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { net_store3_grad2_config (th, g->ih[l], v0->i, v1->i, v2->i,
                                 a->has_ti ? w->ti : 0, 
                                 d0->h[l], d1->h[l], d2->h[l], 
                                 a->input_config[l]);
      }
      else
      { net_store3_grad2 (th, g->ih[l], v0->i, v1->i, v2->i,
                    a->has_ti ? w->ti : 0, a->N_inputs, 
                    d0->h[l], d1->h[l], d2->h[l], N_hidden, 
                    flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1),
                    sparse);
      }
    }

    if (a->has_nsq[l])
    { for (ls = 0; ls<l; ls++)
      { u0 = HIDLOC(pre,ls,v0,0); u1 = HIDLOC(pre,ls,v0,1); 
        u2 = HIDLOC(pre,ls,v0,2);
        nsqi = pre->nonseq[ls][l];
        if (nsqi>=0)
        { if (a->nonseq_config[nsqi])
          { net_store3_grad2_config
                (th, g->nsq[nsqi], u0, u1, u2,
                 a->has_th[ls] ? w->th[ls] : 0,
                 d0->h[l], d1->h[l], d2->h[l], a->nonseq_config[nsqi]);
          }
          else
          { net_store3_grad2 (th, g->nsq[nsqi], u0, u1, u2,
              a->has_th[ls] ? w->th[ls] : 0,
              a->N_hidden[ls], d0->h[l], d1->h[l], d2->h[l], N_hidden,
              (unsigned short *)0, 0, 0);
          }
        }
      }
    }

    if (l>0 && a->has_hh[l-1])
    { u0 = HIDLOC(pre,l-1,v0,0); u1 = HIDLOC(pre,l-1,v0,1); 
      u2 = HIDLOC(pre,l-1,v0,2);
      if (a->hidden_config[l])
      { net_store3_grad2_config
           (th, g->hh[l-1], u0, u1, u2,
            a->has_th[l-1] ? w->th[l-1] : 0,
            d0->h[l], d1->h[l], d2->h[l], a->hidden_config[l]);
      }
      else
      { net_store3_grad2 (th, g->hh[l-1], u0, u1, u2,
          a->has_th[l-1] ? w->th[l-1] : 0,
          a->N_hidden[l-1], d0->h[l], d1->h[l], d2->h[l], N_hidden,
          (unsigned short *)0, 0, 0);
      }
    }

    if (a->has_th[l]) 
    { net_store3_grad1 (th, g->th[l], d0->h[l], d1->h[l], d2->h[l], N_hidden);
    }

    if (a->has_ho[l])
    { u0 = HIDLOC(pre,l,v0,0); u1 = HIDLOC(pre,l,v0,1); 
      u2 = HIDLOC(pre,l,v0,2);
      int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { net_store3_grad2_config (th, g->ho[l], u0, u1, u2,
                           a->has_th[l] ? w->th[l] : 0,
                           d0->o, d1->o, d2->o, a->hidden_config[k]);
      }
      else
      { net_store3_grad2 (th, g->ho[l], u0, u1, u2,
                    a->has_th[l] ? w->th[l] : 0,
                    N_hidden, d0->o, d1->o, d2->o, a->N_outputs, 
                    (unsigned short *) 0, 0, 0);
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
                        1, sparse);
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

/* STORE GRADIENT FOR BIASES FOR 3 CASES.  The thread mod scheme is
   based on indexes for the biases/destination units. */

__device__ static void net_store3_grad1
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* d0,    /* Derivatives with respect to unit values, case 0 */
  net_value const* d1,    /* Derivatives with respect to unit values, case 1 */
  net_value const* d2,    /* Derivatives with respect to unit values, case 2 */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = th; i<n; i+=GTH)
  { g[i] = d0[i] + d1[i] + d2[i];
  }
}


/* STORE GRADIENT FOR BIASES FOR 3 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on indexes for the biases.  Note that
   the connections in quad_s_4d_4w_wgpu, other_wgpu, and other_2_wgpu
   come in GTH sections. */

__device__ static void net_store3_grad1_config
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* d0,    /* Derivatives with respect to unit values, case 0 */
  net_value const* d1,    /* Derivatives with respect to unit values, case 1 */
  net_value const* d2,    /* Derivatives with respect to unit values, case 2 */
  net_config const* cf    /* Configuration for biases */
)
{ net_connection *cn;
  int c, j, j2, k, m, ix;
  int thmod4 = th&3;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[k] = 0;
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_wgpu;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m+4) & 3;
      c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
      for (;;)
      { j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        g[k+ix] += d0[j+ix] + d1[j+ix] + d2[j+ix];
      }
    }
    cn = cf->quad_s_4d_4w_2_wgpu;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m+4) & 3;
      c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
      for (;;)
      { j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        j2 = cn[c].d; c += 1;
        g[k+ix] += (d0[j+ix] + d1[j+ix] + d2[j+ix])
                 + (d0[j2+ix] + d1[j2+ix] + d2[j2+ix]);
      }
    }
  }

  cn = cf->other_wgpu;
  c = cf->start_other_wgpu[th];
  for (;;)
  { j = cn[c].d; k = cn[c].w; c += 1;
    if (k<0) break;
    g[k] += d0[j] + d1[j] + d2[j];
  }
  cn = cf->other_2_wgpu;
  c = cf->start_other_2_wgpu[th];
  for (;;)
  { j = cn[c].d; k = cn[c].w; c += 1;
    if (k<0) break;
    j2 = cn[c].d; c += 1;
    g[k] += (d0[j] + d1[j] + d2[j]) + (d0[j2] + d1[j2] + d2[j2]);
  }
}


/* STORE GRADIENT FOR WEIGHTS FOR 3 CASES.  The thread mod scheme is
   based on the indexes for the destination units, unless there is
   only one destination unit, in which case it is based on the indexes
   for the source units. */

#define NET_STORE3_GRAD2(has_off,has_omit,alllab,onelab,sprs) \
do \
{ int i; \
  if (nd==1 && !has_omit) \
  { net_value d00 = d0[0]; \
    net_value d10 = d1[0]; \
    net_value d20 = d2[0]; \
    if (has_off) \
    { net_value o; \
      for (i = th; i<nv; i+=GTH) \
      { o = off[i]; \
        g[i] = (v0[i]+o)*d00 + (v1[i]+o)*d10 + (v2[i]+o)*d20; \
      } \
    } \
    else \
    { for (i = th; i<nv; i+=GTH) \
      { g[i] = v0[i]*d00 + v1[i]*d10 + v2[i]*d20; \
      } \
    } \
  } \
  else if (sprs) \
  { net_value tv0, tv1, tv2, tvh, o; \
    net_value const*dh; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      tv0 = v0[i] + o; \
      tv1 = v1[i] + o; \
      tv2 = v2[i] + o; \
      if (tv0!=0) \
      { if (tv1!=0 || tv2!=0) goto alllab; \
        tvh = tv0; dh = d0; \
        goto onelab; \
      } \
      else if (tv1!=0) \
      { if (tv2!=0) goto alllab; \
        tvh = tv1; dh = d1; \
        goto onelab; \
      } \
      else if (tv2!=0) \
      { tvh = tv2; dh = d2; \
        goto onelab; \
      } \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = 0; \
      } \
      g += nd; \
      continue; \
    onelab: \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = tvh * dh[j]; \
      } \
      g += nd; \
      continue; \
    alllab: \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = tv0*d0[j] + tv1*d1[j] + tv2*d2[j]; \
      } \
      g += nd; \
    } \
  } \
  else \
  { net_value tv0, tv1, tv2, o; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      tv0 = v0[i] + o; \
      tv1 = v1[i] + o; \
      tv2 = v2[i] + o; \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = tv0*d0[j] + tv1*d1[j] + tv2*d2[j]; \
      } \
      g += nd; \
    } \
  } \
} while (0)

__device__ static void net_store3_grad2
( int th,		  /* Which thread (0 to GTH-1) */
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
  int ob,		  /* Bit to look at in omit flags (mask, not number) */
  int sparse              /* Might source unit values often be zero? */
)
{ 
  if (sparse && off==0)
  { if (omit==0)
    { NET_STORE3_GRAD2(0,0,all1s,one1s,1);
    }
    else
    { NET_STORE3_GRAD2(0,1,all3s,one3s,1);
    }
  }
  else
  { if (omit==0)
    { if (off==0)
      { NET_STORE3_GRAD2(0,0,all1,one1,0);
      }
      else
      { NET_STORE3_GRAD2(1,0,all2,one2,0);
      }
    }
    else
    { if (off==0)
      { NET_STORE3_GRAD2(0,1,all3,one3,0);
      }
      else
      { NET_STORE3_GRAD2(1,1,all4,one4,0);
      }
    }
  }
}


/* STORE GRADIENT FOR WEIGHTS FOR 2 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on the indexes for the weights.  Note
   that the connections in quad_s_4d_4w_wgpu, other_wgpu, and
   other_2_wgpu come in GTH sections. */

__device__ static void net_store3_grad2_config
( int th,		  /* Which thread (0 to GTH-1) */
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
  int i, i2, j, j2, k, c, m, ix;
  int thmod4 = th&3;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[k] = 0;
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_wgpu;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i], s1i = s1[i], s2i = s2[i];
          net_param o = off[i];
          g[k+ix] = g[k+ix]
                  + (s0i+o)*d0[j+ix] + (s1i+o)*d1[j+ix] + (s2i+o)*d2[j+ix];
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i], s1i = s1[i], s2i = s2[i];
          g[k+ix] = g[k+ix] 
                  + s0i*d0[j+ix] + s1i*d1[j+ix] + s2i*d2[j+ix];
        }
      }
    }
    cn = cf->quad_s_4d_4w_2_wgpu;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i], s1i = s1[i], s2i = s2[i];
          net_param o = off[i];
          i2 = cn[c].s; j2 = cn[c].d; c += 1;
          net_value s0i2 = s0[i2], s1i2 = s1[i2], s2i2 = s2[i2];
          net_param o2 = off[i2];
          g[k+ix] = g[k+ix]
            + (s0i+o)*d0[j+ix] + (s1i+o)*d1[j+ix] + (s2i+o)*d2[j+ix]
            + (s0i2+o2)*d0[j2+ix] + (s1i2+o2)*d1[j2+ix] + (s2i2+o2)*d2[j2+ix];
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_value s0i = s0[i], s1i = s1[i], s2i = s2[i];
          i2 = cn[c].s; j2 = cn[c].d; c += 1;
          net_value s0i2 = s0[i2], s1i2 = s1[i2], s2i2 = s2[i2];
          g[k+ix] = g[k+ix] 
                  + s0i*d0[j+ix] + s1i*d1[j+ix] + s2i*d2[j+ix]
                  + s0i2*d0[j2+ix] + s1i2*d1[j2+ix] + s2i2*d2[j2+ix];
        }
      }
    }
  }

  cn = cf->other_wgpu;
  c = cf->start_other_wgpu[th];
  if (off)
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      net_param o = off[i];
      g[k] = g[k] + (s0[i]+o)*d0[j] + (s1[i]+o)*d1[j] + (s2[i]+o)*d2[j];
    }
  }
  else
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      g[k] = g[k] + s0[i]*d0[j] + s1[i]*d1[j] + s2[i]*d2[j];
    }
  }
  cn = cf->other_2_wgpu;
  c = cf->start_other_2_wgpu[th];
  if (off)
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      net_param o = off[i];
      i2 = cn[c].s; j2 = cn[c].d; c += 1;
      net_param o2 = off[i2];
      g[k] = g[k] + (s0[i]+o)*d0[j] + (s1[i]+o)*d1[j] + (s2[i]+o)*d2[j]
               + (s0[i2]+o2)*d0[j2] + (s1[i2]+o2)*d1[j2] + (s2[i2]+o2)*d2[j2];
    }
  }
  else
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      i2 = cn[c].s; j2 = cn[c].d; c += 1;
      g[k] = g[k] + s0[i]*d0[j] + s1[i]*d1[j] + s2[i]*d2[j]
                  + s0[i2]*d0[j2] + s1[i2]*d1[j2] + s2[i2]*d2[j2];
    }
  }
}

#endif


/* --------------------------- store4_grad ---------------------------------- */

#if __CUDACC__

__device__ static void net_store4_grad1 (int, net_param *restrict, 
   net_value const*, net_value const*, net_value const*, net_value const*,
   int);
__device__ static void net_store4_grad1_config (int, net_param *restrict, 
   net_value const*, net_value const*, net_value const*, net_value const*,
   net_config const*);
__device__ static void net_store4_grad2 (int, net_param *restrict, 
   net_value const*, net_value const*, net_value const*, net_value const*,
   net_param const*, int, net_value const*, net_value const*, net_value const*,
   net_value const*, int, unsigned short const*, int, int);
__device__ static void net_store4_grad2_config (int, net_param *restrict, 
   net_value const*, net_value const*, net_value const*, net_value const*,
   net_param const*, net_value const*, net_value const*, net_value const*, 
   net_value const*, net_config const*);


/* STORE SUM OF GRADIENT FROM FOUR CASES.  Threads handle indexes
   equal to their 'th' mod GTH.  The exact meaning of this may depend
   on the kind of parameter group.  This eliminates any need for
   thread synchronization, even if the gradient is produced
   incrementally by adding several terms, since there is no overlap in
   the set of places the two threads store to.  Such a consistent may
   also improve performance.

   Assumes any thread synchronization has been done that's needed to
   make derivatives with respect to unit values accessible to all the
   threads executing here.
*/

__device__ void net_store4_grad
( int th,		/* Which thread (0 to GTH-1) */
  net_params *restrict g, /* Gradient with respect to parameters to store to */
  net_params const*w,	/* Network parameters (only offsets used) */
  net_values const*v0,	/* Values for units in network for case 0, rest follow*/
  net_values const*d0,	/* Backpropagated derivatives for case 0, rest follow */
  net_arch const*a,	/* Network architecture */
  net_precomputed const* pre,  /* Precomputed aspects of architecture */
  net_flags const*flgs,	/* Network flags, null if none */
  int sparse            /* Might source unit values often be zero? */
)
{ 
  const net_values *d1 = d0+1, *d2 = d1+1, *d3 = d2+1;
  const net_values *v1 = v0+1, *v2 = v1+1, *v3 = v2+1;
  const net_value *u0, *u1, *u2, *u3;
  int l, ls, nsqi;

  if (a->has_ti) 
  { net_store4_grad1 (th, g->ti, d0->i, d1->i, d2->i, d3->i, a->N_inputs);
  }

  for (l = 0; l<a->N_layers; l++)
  { 
    int N_hidden = a->N_hidden[l];

    if (a->has_bh[l]) 
    { if (a->bias_config[l])
      { net_store4_grad1_config (th, g->bh[l],
                                 d0->h[l], d1->h[l], d2->h[l], d3->h[l],
                                 a->bias_config[l]);
      }
      else
      { net_store4_grad1 (th, g->bh[l], 
                          d0->h[l], d1->h[l], d2->h[l], d3->h[l], N_hidden);
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { net_store4_grad2_config (th, g->ih[l], v0->i, v1->i, v2->i, v3->i,
                                 a->has_ti ? w->ti : 0, 
                                 d0->h[l], d1->h[l], d2->h[l], d3->h[l],
                                 a->input_config[l]);
      }
      else
      { net_store4_grad2 (th, g->ih[l], v0->i, v1->i, v2->i, v3->i,
                    a->has_ti ? w->ti : 0, a->N_inputs, 
                    d0->h[l], d1->h[l], d2->h[l], d3->h[l], N_hidden, 
                    flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1), 
                    sparse);
      }
    }

    if (a->has_nsq[l])
    { for (ls = 0; ls<l; ls++)
      { u0 = HIDLOC(pre,ls,v0,0); u1 = HIDLOC(pre,ls,v0,1); 
        u2 = HIDLOC(pre,ls,v0,2); u3 = HIDLOC(pre,ls,v0,3); 
        nsqi = pre->nonseq[ls][l];
        if (nsqi>=0)
        { if (a->nonseq_config[nsqi])
          { net_store4_grad2_config
               (th, g->nsq[nsqi], u0, u1, u2, u3,
                a->has_th[ls] ? w->th[ls] : 0,
                d0->h[l], d1->h[l], d2->h[l], d3->h[l], a->nonseq_config[nsqi]);
          }
          else
          { net_store4_grad2 (th, g->nsq[nsqi], u0, u1, u2, u3,
              a->has_th[ls] ? w->th[ls] : 0,
              a->N_hidden[ls], d0->h[l], d1->h[l], d2->h[l], d3->h[l], N_hidden,
              (unsigned short *)0, 0, 0);
          }
        }
      }
    }

    if (l>0 && a->has_hh[l-1])
    { u0 = HIDLOC(pre,l-1,v0,0); u1 = HIDLOC(pre,l-1,v0,1); 
      u2 = HIDLOC(pre,l-1,v0,2); u3 = HIDLOC(pre,l-1,v0,3); 
      if (a->hidden_config[l])
      { net_store4_grad2_config
           (th, g->hh[l-1], u0, u1, u2, u3,
            a->has_th[l-1] ? w->th[l-1] : 0,
            d0->h[l], d1->h[l], d2->h[l], d3->h[l], a->hidden_config[l]);
      }
      else
      { net_store4_grad2 (th, g->hh[l-1], u0, u1, u2, u3,
          a->has_th[l-1] ? w->th[l-1] : 0,
          a->N_hidden[l-1], d0->h[l], d1->h[l], d2->h[l], d3->h[l], N_hidden,
          (unsigned short *)0, 0, 0);
      }
    }

    if (a->has_th[l]) 
    { net_store4_grad1 (th, g->th[l], 
                        d0->h[l], d1->h[l], d2->h[l], d3->h[l], N_hidden);
    }

    if (a->has_ho[l])
    { u0 = HIDLOC(pre,l,v0,0); u1 = HIDLOC(pre,l,v0,1); 
      u2 = HIDLOC(pre,l,v0,2); u3 = HIDLOC(pre,l,v0,3); 
      int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { net_store4_grad2_config (th, g->ho[l], u0, u1, u2, u3,
                           a->has_th[l] ? w->th[l] : 0,
                           d0->o, d1->o, d2->o, d3->o, a->hidden_config[k]);
      }
      else
      { net_store4_grad2 (th, g->ho[l], u0, u1, u2, u3,
                    a->has_th[l] ? w->th[l] : 0,
                    N_hidden, d0->o, d1->o, d2->o, d3->o, a->N_outputs, 
                    (unsigned short *) 0, 0, 0);
      }
    }
  }

  if (a->has_io) 
  { if (a->input_config[a->N_layers])
    { net_store4_grad2_config (th, g->io, v0->i, v1->i, v2->i, v3->i,
                               a->has_ti ? w->ti : 0, 
                               d0->o, d1->o, d2->o, d3->o,
                               a->input_config[a->N_layers]);
    }
    else
    { net_store4_grad2 (th, g->io, v0->i, v1->i, v2->i, v3->i,
                        a->has_ti ? w->ti : 0, a->N_inputs, 
                        d0->o, d1->o, d2->o, d3->o, a->N_outputs, 
                        flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0,
                        1, sparse);
    }
  }

  if (a->has_bo) 
  { if (a->bias_config[a->N_layers])
    { net_store4_grad1_config (th, g->bo, d0->o, d1->o, d2->o, d3->o,
                               a->bias_config[a->N_layers]);
    }
    else
    { net_store4_grad1 (th, g->bo, d0->o, d1->o, d2->o, d3->o, a->N_outputs);
    }
  }
}


/* STORE GRADIENT FOR BIASES FOR 4 CASES.  The thread mod scheme is
   based on indexes for the biases/destination units. */

__device__ static void net_store4_grad1
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* d0,    /* Derivatives with respect to unit values, case 0 */
  net_value const* d1,    /* Derivatives with respect to unit values, case 1 */
  net_value const* d2,    /* Derivatives with respect to unit values, case 2 */
  net_value const* d3,    /* Derivatives with respect to unit values, case 3 */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = th; i<n; i+=GTH)
  { g[i] = d0[i] + d1[i] + d2[i] + d3[i];
  }
}


/* STORE GRADIENT FOR BIASES FOR 4 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on indexes for the biases.  Note that
   the connections in quad_s_4d_4w_wgpu, other_wgpu, and other_2_wgpu
   come in GTH sections. */

__device__ static void net_store4_grad1_config
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* d0,    /* Derivatives with respect to unit values, case 0 */
  net_value const* d1,    /* Derivatives with respect to unit values, case 1 */
  net_value const* d2,    /* Derivatives with respect to unit values, case 2 */
  net_value const* d3,    /* Derivatives with respect to unit values, case 3 */
  net_config const* cf    /* Configuration for biases */
)
{ net_connection *cn;
  int c, j, j2, k, m, ix;
  int thmod4 = th&3;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[k] = 0;
  }

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_wgpu;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m+4) & 3;
      c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
      for (;;)
      { j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        g[k+ix] += (d0[j+ix] + d1[j+ix]) + (d2[j+ix] + d3[j+ix]);
      }
    }
    cn = cf->quad_s_4d_4w_2_wgpu;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m+4) & 3;
      c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
      for (;;)
      { j = cn[c].d; k = cn[c].w; c += 1;
        if (k<0) break;
        j2 = cn[c].d; c += 1;
        g[k+ix] += ((d0[j+ix] + d1[j+ix]) + (d2[j+ix] + d3[j+ix]))
                 + ((d0[j2+ix] + d1[j2+ix]) + (d2[j2+ix] + d3[j2+ix]));
      }
    }
  }

  cn = cf->other_wgpu;
  c = cf->start_other_wgpu[th];
  for (;;)
  { j = cn[c].d; k = cn[c].w; c += 1;
    if (k<0) break;
    g[k] += (d0[j] + d1[j]) + (d2[j] + d3[j]);
  }
  cn = cf->other_2_wgpu;
  c = cf->start_other_2_wgpu[th];
  for (;;)
  { j = cn[c].d; k = cn[c].w; c += 1;
    if (k<0) break;
    j2 = cn[c].d; c += 1;
    g[k] += ((d0[j] + d1[j]) + (d2[j] + d3[j]))
          + ((d0[j2] + d1[j2]) + (d2[j2] + d3[j2]));
  }
}


/* STORE GRADIENT FOR WEIGHTS FOR 4 CASES.  The thread mod scheme is
   based on the indexes for the destination units, unless there is
   only one destination unit, in which case it is based on the indexes
   for the source units. */

#define NET_STORE4_GRAD2(has_off,has_omit,alllab,onelab,sprs) \
do \
{ int i; \
  if (nd==1 && !has_omit) \
  { net_value d00 = d0[0]; \
    net_value d10 = d1[0]; \
    net_value d20 = d2[0]; \
    net_value d30 = d3[0]; \
    if (has_off) \
    { net_value o; \
      for (i = th; i<nv; i+=GTH) \
      { o = off[i]; \
        g[i] = (v0[i]+o)*d00 + (v1[i]+o)*d10 + (v2[i]+o)*d20 + (v3[i]+o)*d30; \
      } \
    } \
    else \
    { for (i = th; i<nv; i+=GTH) \
      { g[i] = v0[i]*d00 + v1[i]*d10 + v2[i]*d20 + v3[i]*d30; \
      } \
    } \
  } \
  else if (sprs) \
  { net_value tv0, tv1, tv2, tv3, tvh, o; \
    net_value const*dh; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      tv0 = v0[i] + o; \
      tv1 = v1[i] + o; \
      tv2 = v2[i] + o; \
      tv3 = v3[i] + o; \
      if (tv0!=0) \
      { if (tv1!=0 || tv2!=0 || tv3!=0) goto alllab; \
        tvh = tv0; dh = d0; \
        goto onelab; \
      } \
      else if (tv1!=0) \
      { if (tv2!=0 || tv3!=0) goto alllab; \
        tvh = tv1; dh = d1; \
        goto onelab; \
      } \
      else if (tv2!=0) \
      { if (tv3!=0) goto alllab; \
        tvh = tv2; dh = d2; \
        goto onelab; \
      } \
      else if (tv3!=0) \
      { tvh = tv3; dh = d3; \
        goto onelab; \
      } \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = 0; \
      } \
      g += nd; \
      continue; \
    onelab: \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = tvh * dh[j]; \
      } \
      g += nd; \
      continue; \
    alllab: \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = tv0*d0[j] + tv1*d1[j] + tv2*d2[j] + tv3*d3[j]; \
      } \
      g += nd; \
    } \
  } \
  else \
  { net_value tv0, tv1, tv2, tv3, o; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      tv0 = v0[i] + o; \
      tv1 = v1[i] + o; \
      tv2 = v2[i] + o; \
      tv3 = v3[i] + o; \
      for (j = th; j<nd; j+=GTH) \
      { g[j] = tv0*d0[j] + tv1*d1[j] + tv2*d2[j] + tv3*d3[j]; \
      } \
      g += nd; \
    } \
  } \
} while (0)

__device__ static void net_store4_grad2
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const* v0,    /* Source unit values, case 0 */
  net_value const* v1,    /* Source unit values, case 1 */
  net_value const* v2,    /* Source unit values, case 2 */
  net_value const* v3,    /* Source unit values, case 3 */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value const* d0,    /* Derivatives with respect to destination units, 0 */
  net_value const* d1,    /* Derivatives with respect to destination units, 1 */
  net_value const* d2,    /* Derivatives with respect to destination units, 2 */
  net_value const* d3,    /* Derivatives with respect to destination units, 3 */
  int nd,		  /* Number of destination units */
  unsigned short const* omit, /* Omit flags, null if not present */
  int ob,		  /* Bit to look at in omit flags (mask, not number) */
  int sparse		  /* Might source unit values often be zero? */
)
{ 
  if (sparse && off==0)
  { if (omit==0)
    { NET_STORE4_GRAD2(0,0,all1s,one1s,1);
    }
    else
    { NET_STORE4_GRAD2(0,1,all3s,one3s,1);
    }
  }
  else
  { if (omit==0)
    { if (off==0)
      { NET_STORE4_GRAD2(0,0,all1,one1,0);
      }
      else
      { NET_STORE4_GRAD2(1,0,all2,one2,0);
      }
    }
    else
    { if (off==0)
      { NET_STORE4_GRAD2(0,1,all3,one3,0);
      }
      else
      { NET_STORE4_GRAD2(1,1,all4,one4,0);
      }
    }
  }
}


/* STORE GRADIENT FOR WEIGHTS FOR 4 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on the indexes for the weights.  Note
   that the connections in quad_s_4d_4w_wgpu, other_wgpu, and
   other_2_wgpu come in GTH sections. */

__device__ static void net_store4_grad2_config
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* s0,    /* Source unit values, case 0 */
  net_value const* s1,    /* Source unit values, case 1 */
  net_value const* s2,    /* Source unit values, case 2 */
  net_value const* s3,    /* Source unit values, case 3 */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  net_value const* d0,    /* Derivatives with respect to destination units, 0 */
  net_value const* d1,    /* Derivatives with respect to destination units, 1 */
  net_value const* d2,    /* Derivatives with respect to destination units, 2 */
  net_value const* d3,    /* Derivatives with respect to destination units, 3 */
  net_config const* cf    /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, i2, j, j2, k, c;
  int thmod4 = th&3;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[k] = 0;
  }
 
  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w_wgpu;
    int m, ix;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_param o = off[i];
          g[k+ix] = g[k+ix] + (s0[i]+o)*d0[j+ix] + (s1[i]+o)*d1[j+ix] 
                            + (s2[i]+o)*d2[j+ix] + (s3[i]+o)*d3[j+ix];
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          g[k+ix] = g[k+ix] + s0[i]*d0[j+ix] + s1[i]*d1[j+ix] 
                            + s2[i]*d2[j+ix] + s3[i]*d3[j+ix];
        }
      }
    }
    cn = cf->quad_s_4d_4w_2_wgpu;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          net_param o = off[i];
          i2 = cn[c].s; j2 = cn[c].d; c += 1;
          net_param o2 = off[i2];
          g[k+ix] = g[k+ix]
                  + (s0[i]+o)*d0[j+ix] + (s1[i]+o)*d1[j+ix]
                  + (s2[i]+o)*d2[j+ix] + (s3[i]+o)*d3[j+ix]
                  + (s0[i2]+o2)*d0[j2+ix] + (s1[i2]+o2)*d1[j2+ix]
                  + (s2[i2]+o2)*d2[j2+ix] + (s3[i2]+o2)*d3[j2+ix];
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m+4) & 3;
        c = cf->start_quad_2_wgpu [(th-ix+GTH) & (GTH-1)];
        for (;;)
        { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          if (k<0) break;
          i2 = cn[c].s; j2 = cn[c].d; c += 1;
          g[k+ix] = g[k+ix] 
                  + s0[i]*d0[j+ix] + s1[i]*d1[j+ix]       /* avoid += so  */
                  + s2[i]*d2[j+ix] + s3[i]*d3[j+ix]       /* multiply-add */
                  + s0[i2]*d0[j2+ix] + s1[i2]*d1[j2+ix]   /* can be used  */
                  + s2[i2]*d2[j2+ix] + s3[i2]*d3[j2+ix];
        }
      }
    }
  }

  cn = cf->other_wgpu;
  c = cf->start_other_wgpu[th];
  if (off)
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      net_param o = off[i];
      g[k] = g[k] + (s0[i]+o)*d0[j] + (s1[i]+o)*d1[j] 
                  + (s2[i]+o)*d2[j] + (s3[i]+o)*d3[j];
    }
  }
  else
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      g[k]  = g[k] + s0[i]*d0[j] + s1[i]*d1[j] 
                   + s2[i]*d2[j] + s3[i]*d3[j];
    }
  }
  cn = cf->other_2_wgpu;
  c = cf->start_other_2_wgpu[th];
  if (off)
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      net_param o = off[i];
      i2 = cn[c].s; j2 = cn[c].d; c += 1;
      net_param o2 = off[i2];
      g[k] = g[k] + (s0[i]+o)*d0[j] + (s1[i]+o)*d1[j] 
                  + (s2[i]+o)*d2[j] + (s3[i]+o)*d3[j]
                  + (s0[i2]+o2)*d0[j2] + (s1[i2]+o2)*d1[j2] 
                  + (s2[i2]+o2)*d2[j2] + (s3[i2]+o2)*d3[j2];
    }
  }
  else
  { for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      i2 = cn[c].s; j2 = cn[c].d; c += 1;
      g[k] = g[k] + s0[i]*d0[j] + s1[i]*d1[j]        /* avoid += so  */
                  + s2[i]*d2[j] + s3[i]*d3[j]        /* multiply-add */
                  + s0[i2]*d0[j2] + s1[i2]*d1[j2]    /* can be used  */
                  + s2[i2]*d2[j2] + s3[i2]*d3[j2];
    }
  }
}

#endif
