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

#include "misc.h"
#include "log.h"
#include "prior.h"
#include "data.h"
#include "model.h"
#include "net.h"

#include "intrinsics-use.h"


/* This module finds the derivative of the "error" for a particular case
   with respect to the values of the hidden and input units, and the with
   respect to the summed input for the hidden units, given the derivative 
   with respect to the output units.  There are facilities for calculating 
   these derivatives only back to a certain layer.
*/

#define sqrt_2 1.4142135623730950488

static void sum_derivatives  (net_value *restrict, int, net_value *restrict,
                              int, net_param *restrict, 
                              unsigned short *restrict, int);
static void sum_derivatives_config (net_value *restrict, net_value *restrict,
                                    net_param *restrict, net_config *restrict);


/* BACKPROPAGATE ERROR DERIVATIVES.  The first argument must contain the 
   values of all the units in the network.  The second must contain the
   derivatives of the "error" with respect to the output units (in g->o).
   These derivatives are backpropagated to give the derivatives of the
   error with respect to the other units in the network, and with respect
   to the summed input into the hidden units.  This is done back to hidden 
   layer 'start', or back all the way to the inputs if 'start' is -1. */

void net_back
( net_values *v,	/* Values for units in network */
  net_values *d,	/* Place to get output derivatives, and store others */
  int start,		/* Earliest layer to find derivatives for */
  net_arch *a,		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  net_params *w		/* Network parameters */
)
{
  int l, i;

  /* Backpropagate through hidden layers. */

  for (l = a->N_layers-1; l>=0 && l>=start; l--)
  { 
    int N_hidden = a->N_hidden[l];
    net_value *restrict dh = d->h[l];

    memset (dh, 0, N_hidden * sizeof *dh);
    
    if (a->has_ho[l])
    { sum_derivatives (d->o, a->N_outputs, dh, N_hidden, 
                       w->ho[l], (unsigned short *) 0, 0);
    }

    if (l<a->N_layers-1 && a->has_hh[l])
    { if (a->hidden_config[l+1])
      { sum_derivatives_config (d->s[l+1], dh, w->hh[l], a->hidden_config[l+1]);
      }
      else 
      { sum_derivatives (d->s[l+1], a->N_hidden[l+1], dh, N_hidden, 
                         w->hh[l], (unsigned short *) 0, 0);
      }
    }

    net_value *restrict vh = v->h[l];
    net_value *restrict ds = d->s[l];

    if (flgs==0 || flgs->layer_type[l]==Tanh_type)
    {
#     if USE_SIMD_INTRINSICS && __AVX__ && 0  /* disabled, since doesn't help */
      { __m256d ONE = _mm256_set1_pd(1.0);
        i = 3;
        while (i<N_hidden)
        { __m256d VH = _mm256_loadu_pd(vh+i-3);
          _mm256_storeu_pd (ds+i-3, _mm256_mul_pd (_mm256_loadu_pd(dh+i-3),
                                      _mm256_sub_pd(ONE,_mm256_mul_pd(VH,VH))));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128d VH = _mm_loadu_pd(vh+i-1);
          _mm_storeu_pd (ds+i-1, _mm_mul_pd (_mm_loadu_pd(dh+i-1),
           _mm_sub_pd (_mm256_castpd256_pd128(ONE), _mm_mul_pd(VH,VH))));
          i += 2;
        }
        if (i<=N_hidden)
        { __m128d VH = _mm_load_sd(vh+i-1);
          _mm_store_sd (ds+i-1, _mm_mul_sd (_mm_load_sd(dh+i-1),
           _mm_sub_sd (_mm256_castpd256_pd128(ONE), _mm_mul_sd(VH,VH))));
        }
      }
#     else
      { for (i = 0; i<N_hidden; i++)
        { ds[i] = (1 - vh[i]*vh[i]) * dh[i];
        }
      }
#     endif
    }
    else if (flgs->layer_type[l]==Sin_type)
    { for (i = 0; i<N_hidden; i++)
      { ds[i] = 2 * cos(v->s[l][i]*sqrt_2) * dh[i];
      }
    }
    else if (flgs->layer_type[l]==Identity_type)
    { for (i = 0; i<N_hidden; i++)
      { ds[i] = dh[i];
      }
    }
    else
    { abort();
    }
  }

  /* Backpropagate to input layer. */

  if (start<0)
  {
    memset (d->i, 0, a->N_inputs * sizeof *d->i);
 
    for (l = 0; l<a->N_layers; l++)
    { if (a->has_ih[l])
      { if (a->input_config[l])
        { sum_derivatives_config (d->s[l], d->i, w->ih[l], a->input_config[l]);
        }
        else 
        { sum_derivatives (d->s[l], a->N_hidden[l], d->i, a->N_inputs, w->ih[l],
                           flgs ? flgs->omit : 0, 1<<(l+1));
        }
      }
    }

    if (a->has_io)
    { if (a->input_config[a->N_layers])
      { sum_derivatives_config(d->o, d->i, w->io, a->input_config[a->N_layers]);
      }
      else
      { sum_derivatives (d->o, a->N_outputs, d->i, a->N_inputs, w->io,
                         flgs ? flgs->omit : 0, 1);
      }
    }
  }
}


/* SUM UP CONTRIBUTIONS TO THE DERIVATIVES FROM ONE GROUP OF CONNECTIONS.  Adds 
   the weighted sum of derivatives due to connections from source units to 
   a given destination layer to the totals for the source layer. */

#define SUM_DERIVATIVES(omit) \
do \
{ net_value tv; \
  int i, j; \
  if (nd==1) \
  { double d0 = dd[0]; \
    i = 3; \
    while (i<ns) \
    { if (!(omit)) ds[i-3] += *w++ * d0; \
      if (!(omit)) ds[i-2] += *w++ * d0; \
      if (!(omit)) ds[i-1] += *w++ * d0; \
      if (!(omit)) ds[i-0] += *w++ * d0; \
      i += 4; \
    } \
    i -= 3; \
    while (i<ns) \
    { if (!(omit)) ds[i] += *w++ * d0; \
      i += 1; \
    } \
  } \
  else \
  { for (i = 0; i<ns; i++) \
    { if (omit) continue; \
      tv = 0; \
      j = 3; \
      while (j<nd) \
      { tv += w[j-3] * dd[j-3]; \
        tv += w[j-2] * dd[j-2]; \
        tv += w[j-1] * dd[j-1]; \
        tv += w[j-0] * dd[j-0]; \
        j += 4; \
      } \
      j -= 3; \
      while (j<nd) \
      { tv += w[j] * dd[j]; \
        j += 1; \
      } \
      w += nd; \
      ds[i] += tv; \
    } \
  } \
} while (0)

#if USE_SIMD_INTRINSICS && __AVX__ && USE_FMA &&__FMA__

#define SUM_DERIVATIVES0 \
do \
{ net_value tv; \
  int i, j; \
  if (nd==1) \
  { __m256d D0 = _mm256_broadcast_sd(dd); \
    i = 3; \
    while (i<ns) \
    { _mm256_storeu_pd (ds+i-3, _mm256_fmadd_pd (D0, \
                         _mm256_loadu_pd(w+i-3), _mm256_loadu_pd(ds+i-3))); \
      i += 4; \
    } \
    i -= 2; \
    if (i<ns) \
    { _mm_storeu_pd (ds+i-1, _mm_fmadd_pd (_mm256_castpd256_pd128(D0), \
                              _mm_loadu_pd(w+i-1), _mm_loadu_pd(ds+i-1))); \
      i += 2; \
    } \
    if (i<=ns) \
    { _mm_store_sd (ds+i-1, _mm_fmadd_sd (_mm256_castpd256_pd128(D0), \
                             _mm_load_sd(w+i-1), _mm_load_sd(ds+i-1))); \
    } \
  } \
  else \
  { __m256d TV, TV2; \
    for (i = 1; i<ns; i+=2) \
    { net_param *w2 = w+nd; \
      TV = _mm256_setzero_pd(); \
      TV2 = _mm256_setzero_pd(); \
      j = 3; \
      while (j<nd) \
      { __m256d DD = _mm256_loadu_pd(dd+j-3); \
        TV = _mm256_fmadd_pd (_mm256_loadu_pd(w+j-3), DD, TV); \
        TV2 = _mm256_fmadd_pd (_mm256_loadu_pd(w2+j-3), DD, TV2); \
        j += 4; \
      } \
      __m128d T, T2; \
      T = _mm_add_pd (_mm256_castpd256_pd128(TV), \
                      _mm256_extractf128_pd(TV,1)); \
      T2 = _mm_add_pd (_mm256_castpd256_pd128(TV2), \
                       _mm256_extractf128_pd(TV2,1)); \
      j -= 2; \
      if (j<nd) \
      { __m128d DD = _mm_loadu_pd(dd+j-1); \
        T = _mm_fmadd_pd (_mm_loadu_pd(w+j-1), DD, T); \
        T2 = _mm_fmadd_pd (_mm_loadu_pd(w2+j-1), DD, T2); \
        j += 2; \
      } \
      T = _mm_hadd_pd(T,T2); \
      if (j<=nd) \
      { __m128d DD = _mm_load_pd1(dd+j-1); \
        __m128d WW = _mm_loadh_pd (_mm_load_sd(w+j-1), w2+j-1); \
        T = _mm_fmadd_pd (WW, DD, T); \
      } \
      _mm_storeu_pd (ds+i-1, _mm_add_pd (_mm_loadu_pd(ds+i-1), T)); \
      w = w2+nd; \
    } \
    if (i<=ns) \
    { TV = _mm256_setzero_pd(); \
      j = 3; \
      while (j<nd) \
      { TV = _mm256_fmadd_pd (_mm256_loadu_pd(w+j-3), \
                              _mm256_loadu_pd(dd+j-3), TV); \
        j += 4; \
      } \
      __m128d T; \
      T = _mm_add_pd (_mm256_castpd256_pd128(TV), \
                      _mm256_extractf128_pd(TV,1)); \
      j -= 2; \
      if (j<nd) \
      { T = _mm_fmadd_pd (_mm_loadu_pd(w+j-1), _mm_loadu_pd(dd+j-1), T); \
        j += 2; \
      } \
      T = _mm_hadd_pd(T,T); \
      if (j<=nd) \
      { T = _mm_fmadd_sd (_mm_load_sd(w+j-1), _mm_load_sd(dd+j-1), T); \
      } \
      _mm_store_sd (ds+i-1, _mm_add_sd(_mm_load_sd(ds+i-1), T)); \
    } \
  } \
} while (0)

#elif USE_SIMD_INTRINSICS && __AVX__

#define SUM_DERIVATIVES0 \
do \
{ net_value tv; \
  int i, j; \
  if (nd==1) \
  { __m256d D0 = _mm256_broadcast_sd(dd); \
    i = 3; \
    while (i<ns) \
    { _mm256_storeu_pd (ds+i-3, _mm256_add_pd (_mm256_loadu_pd(ds+i-3), \
         _mm256_mul_pd (D0, _mm256_loadu_pd(w+i-3)))); \
      i += 4; \
    } \
    i -= 2; \
    if (i<ns) \
    { _mm_storeu_pd (ds+i-1, _mm_add_pd (_mm_loadu_pd(ds+i-1), \
         _mm_mul_pd (_mm256_castpd256_pd128(D0), _mm_loadu_pd(w+i-1)))); \
      i += 2; \
    } \
    if (i<=ns) \
    { _mm_store_sd (ds+i-1, _mm_add_sd (_mm_load_sd(ds+i-1), \
         _mm_mul_sd (_mm256_castpd256_pd128(D0), _mm_load_sd(w+i-1)))); \
    } \
  } \
  else \
  { __m256d TV, TV2; \
    for (i = 1; i<ns; i+=2) \
    { net_param *w2 = w+nd; \
      TV = _mm256_setzero_pd(); \
      TV2 = _mm256_setzero_pd(); \
      j = 3; \
      while (j<nd) \
      { __m256d DD = _mm256_loadu_pd(dd+j-3); \
        TV = _mm256_add_pd (TV, _mm256_mul_pd (_mm256_loadu_pd(w+j-3), DD)); \
        TV2 = _mm256_add_pd (TV2, _mm256_mul_pd (_mm256_loadu_pd(w2+j-3),DD)); \
        j += 4; \
      } \
      __m128d T, T2; \
      T = _mm_add_pd (_mm256_castpd256_pd128(TV), \
                      _mm256_extractf128_pd(TV,1)); \
      T2 = _mm_add_pd (_mm256_castpd256_pd128(TV2), \
                       _mm256_extractf128_pd(TV2,1)); \
      j -= 2; \
      if (j<nd) \
      { __m128d DD = _mm_loadu_pd(dd+j-1); \
        T = _mm_add_pd (T, _mm_mul_pd (_mm_loadu_pd(w+j-1), DD)); \
        T2 = _mm_add_pd (T2, _mm_mul_pd (_mm_loadu_pd(w2+j-1), DD)); \
        j += 2; \
      } \
      T = _mm_hadd_pd(T,T2); \
      if (j<=nd) \
      { __m128d DD = _mm_load_pd1(dd+j-1); \
        __m128d WW = _mm_loadh_pd (_mm_load_sd(w+j-1), w2+j-1); \
        T = _mm_add_pd (T, _mm_mul_pd (WW, DD)); \
      } \
      _mm_storeu_pd (ds+i-1, _mm_add_pd (_mm_loadu_pd(ds+i-1), T)); \
      w = w2+nd; \
    } \
    if (i<=ns) \
    { TV = _mm256_setzero_pd(); \
      j = 3; \
      while (j<nd) \
      { TV = _mm256_add_pd (TV, _mm256_mul_pd (_mm256_loadu_pd(w+j-3), \
                                               _mm256_loadu_pd(dd+j-3))); \
        j += 4; \
      } \
      __m128d T; \
      T = _mm_add_pd (_mm256_castpd256_pd128(TV), \
                      _mm256_extractf128_pd(TV,1)); \
      j -= 2; \
      if (j<nd) \
      { T = _mm_add_pd (T, _mm_mul_pd (_mm_loadu_pd(w+j-1), \
                                       _mm_loadu_pd(dd+j-1))); \
        j += 2; \
      } \
      T = _mm_hadd_pd(T,T); \
      if (j<=nd) \
      { T = _mm_add_sd (T, _mm_mul_sd (_mm_load_sd(w+j-1), \
                                       _mm_load_sd(dd+j-1))); \
      } \
      _mm_store_sd (ds+i-1, _mm_add_sd(_mm_load_sd(ds+i-1), T)); \
    } \
  } \
} while (0)

#else

#define SUM_DERIVATIVES0 \
do \
{ net_value tv; \
  int i, j; \
  if (nd==1) \
  { double d0 = dd[0]; \
    i = 3; \
    while (i<ns) \
    { ds[i-3] += w[i-3] * d0; \
      ds[i-2] += w[i-2] * d0; \
      ds[i-1] += w[i-1] * d0; \
      ds[i-0] += w[i-0] * d0; \
      i += 4; \
    } \
    i -= 3; \
    while (i<ns) \
    { ds[i] += w[i] * d0; \
      i += 1; \
    } \
  } \
  else \
  { for (i = 0; i<ns; i++) \
    { tv = 0; \
      j = 3; \
      while (j<nd) \
      { tv += w[j-3] * dd[j-3]; \
        tv += w[j-2] * dd[j-2]; \
        tv += w[j-1] * dd[j-1]; \
        tv += w[j-0] * dd[j-0]; \
        j += 4; \
      } \
      j -= 3; \
      while (j<nd) \
      { tv += w[j] * dd[j]; \
        j += 1; \
      } \
      w += nd; \
      ds[i] += tv; \
    } \
  } \
} while (0)

#endif

static void sum_derivatives
( net_value *restrict dd, /* Derivatives with respect to destination units */
  int nd,		  /* Number of destination units */
  net_value *restrict ds, /* Derivatives w.r.t. source units to add to */
  int ns,		  /* Number of source units */
  net_param *restrict w,  /* Connection weights */
  unsigned short *restrict omit,  /* Omit flags, null if not present */
  int b			  /* Bit to look at in omit flags */
)
{
  if (omit==0)
  { SUM_DERIVATIVES0;
  }
  else
  { SUM_DERIVATIVES((*omit++)&b);
  }
}


/* SUM UP CONTRIBUTIONS TO THE DERIVATIVES FROM CONNECTIONS WITH CONFIGURATION.
   Adds the weighted sum of derivatives due to connections from source units to
   a given destination layer to the totals for the source layer. */

static void sum_derivatives_config
( net_value *restrict dd, /* Derivatives with respect to destination units */
  net_value *restrict ds, /* Derivatives w.r.t. source units to add to */
  net_param *restrict w,  /* Connection weights */
  net_config *restrict cf /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w;
#   if USE_SIMD_INTRINSICS && __AVX__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { i = cn[c].s; j = cn[c].d;
        __m256d P = _mm256_mul_pd (_mm256_loadu_pd(dd+j),
                                   _mm256_loadu_pd(w+k));
        __m128d S = _mm_add_pd (_mm256_extractf128_pd(P,1),
                                _mm256_castpd256_pd128(P));
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
      }
    }
#   else
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { i = cn[c].s; j = cn[c].d;
        ds[i] += (dd[j+0]*w[k+0] + dd[j+2]*w[k+2])      /* same order as AVX  */
                   + (dd[j+1]*w[k+1] + dd[j+3]*w[k+3]); /* instructions above */
      }
    }
#   endif
  }

  if (CONFIG_SINGLE4)
  { 
    cn = cf->single4_s;
    for (c = 0; (k = cn[c].w) >= 0; c+=4)
    { i = cn[c].s;
      double dsi = ds[i];
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
    { double ddj = dd[cn[c].d];
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
