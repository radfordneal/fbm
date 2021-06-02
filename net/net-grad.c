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

#include "misc.h"
#include "log.h"
#include "prior.h"
#include "data.h"
#include "model.h"
#include "net.h"


#if USE_AVX_INTRINSICS && __AVX__
#include  <immintrin.h>
#endif


/* This module finds the derivatives of the "error" on the training set
   with respect to the network parameters, using the backpropagated 
   derivatives of the error for each training case with respect to the
   unit values. */


static void add_grad1 (net_param *, net_value *, int);
static void add_grad2 (net_param *, net_value *, net_param *, int, 
                       net_value *, int, char *, int);


/* ADD TO GRADIENT OF ERROR WITH RESPECT TO NETWORK PARAMETERS.  Adds to 
   a set of derivatives with respect to network parameters, stored in a
   structure of the same form as the parameters.  The derivatives added are
   of the "error" for a training case, derived from unit values and 
   derivatives previously computed. 

   One can economize by not bothering to compute the derivatives of the 
   error with respect to the input unit values if the network does not
   have input offset parameters. */

void net_grad
( net_params *g,	/* Gradient with respect to parameters to add to */
  net_params *w,	/* Network parameters */
  net_values *v,	/* Values for units in network for a case */
  net_values *d,	/* Backpropagated derivatives for a case */
  net_arch *a,		/* Network architecture */
  net_flags *flgs	/* Network flags, null if none */
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
    { add_grad1 (g->bh[l], d->s[l], N_hidden);
    }

    if (a->has_ih[l])
    { add_grad2 (g->ih[l], v->i, a->has_ti ? w->ti : 0, a->N_inputs, 
                 d->s[l], N_hidden, flgs?flgs->omit:0, 1<<(l+1));
    }

    if (l>0 && a->has_hh[l-1])
    { add_grad2 (g->hh[l-1], v->h[l-1], a->has_th[l-1] ? w->th[l-1] : 0,
                 a->N_hidden[l-1], d->s[l], N_hidden, (char *) 0, 0);
    }

    if (a->has_th[l]) 
    { add_grad1 (g->th[l], d->h[l], N_hidden);
    }

    if (a->has_ho[l])
    { add_grad2 (g->ho[l], v->h[l], a->has_th[l] ? w->th[l] : 0,
                 N_hidden, d->o, a->N_outputs, (char *) 0, 0);
    }
  }

  if (a->has_io) 
  { add_grad2 (g->io, v->i, a->has_ti ? w->ti : 0, a->N_inputs, 
               d->o, a->N_outputs, flgs?flgs->omit:0, 1);
  }

  if (a->has_bo) 
  { add_grad1 (g->bo, d->o, a->N_outputs);
  }
}


/* ADD TO GRADIENT FROM UNIT DERIVATIVE. */

static void add_grad1
( net_param *restrict g,  /* Array of derivatives to add to */
  net_value *restrict v,  /* Derivatives with respect to unit values */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = 0; i<n; i++)
  { g[i] += v[i];
  }
}


/* ADD TO GRADIENT FROM PRODUCT OF UNIT VALUE AND UNIT DERIVATIVE. */

#define ADD_GRAD2(offset,omit) \
do \
{ double tv, o; \
  int i, j; \
  if (nd==1) \
  { double d0 = d[0]; \
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
    { o = (offset); \
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

#if USE_AVX_INTRINSICS && __AVX__ && __FMA__

#define ADD_GRAD2_00 \
do \
{ int i, j; \
  if (nd==1) \
  { __m256d D0 = _mm256_set1_pd(d[0]); \
    i = 3; \
    while (i<nv) \
    { _mm256_storeu_pd (g+i-3, _mm256_fmadd_pd (D0, _mm256_loadu_pd(v+i-3), \
                                                    _mm256_loadu_pd(g+i-3))); \
      i += 4; \
    } \
    i -= 2; \
    if (i<nv) \
    { _mm_storeu_pd (g+i-1, _mm_fmadd_pd (_mm256_castpd256_pd128(D0), \
                               _mm_loadu_pd(v+i-1), _mm_loadu_pd(g+i-1))); \
      i += 2; \
    } \
    if (i<=nv) \
    { _mm_store_sd (g+i-1, _mm_fmadd_sd (_mm256_castpd256_pd128(D0), \
                              _mm_load_sd(v+i-1), _mm_load_sd(g+i-1))); \
    } \
  } \
  else \
  { for (i = 0; i<nv; i++) \
    { if (v[i]!=0)  \
      { __m256d TV = _mm256_set1_pd(v[i]); \
        j = 3; \
        while (j<nd) \
        { _mm256_storeu_pd (g+j-3, _mm256_fmadd_pd (TV, \
                _mm256_loadu_pd(d+j-3), _mm256_loadu_pd(g+j-3))); \
          j += 4; \
        } \
        j -= 2; \
        if (j<nd) \
        { _mm_storeu_pd (g+j-1, _mm_fmadd_pd (_mm256_castpd256_pd128(TV), \
                _mm_loadu_pd(d+j-1), _mm_loadu_pd(g+j-1))); \
          j += 2; \
        } \
        if (j<=nd) \
        { _mm_store_sd (g+j-1, _mm_fmadd_sd (_mm256_castpd256_pd128(TV), \
                _mm_load_sd(d+j-1), _mm_load_sd(g+j-1))); \
        } \
      } \
      g += nd; \
    } \
  } \
} while (0)

#elif USE_AVX_INTRINSICS && __AVX__

#define ADD_GRAD2_00 \
do \
{ int i, j; \
  if (nd==1) \
  { __m256d D0 = _mm256_set1_pd(d[0]); \
    i = 3; \
    while (i<nv) \
    { _mm256_storeu_pd (g+i-3, _mm256_add_pd (_mm256_loadu_pd(g+i-3), \
         _mm256_mul_pd (D0, _mm256_loadu_pd(v+i-3)))); \
      i += 4; \
    } \
    i -= 2; \
    if (i<nv) \
    { _mm_storeu_pd (g+i-1, _mm_add_pd (_mm_loadu_pd(g+i-1), \
         _mm_mul_pd (_mm256_castpd256_pd128(D0), _mm_loadu_pd(v+i-1)))); \
      i += 2; \
    } \
    if (i<=nv) \
    { _mm_store_sd (g+i-1, _mm_add_sd (_mm_load_sd(g+i-1), \
         _mm_mul_sd (_mm256_castpd256_pd128(D0), _mm_load_sd(v+i-1)))); \
    } \
  } \
  else \
  { for (i = 0; i<nv; i++) \
    { if (v[i]!=0)  \
      { __m256d TV = _mm256_set1_pd(v[i]); \
        j = 3; \
        while (j<nd) \
        { _mm256_storeu_pd (g+j-3, _mm256_add_pd (_mm256_loadu_pd(g+j-3), \
             _mm256_mul_pd (TV, _mm256_loadu_pd(d+j-3)))); \
          j += 4; \
        } \
        j -= 2; \
        if (j<nd) \
        { _mm_storeu_pd (g+j-1, _mm_add_pd (_mm_loadu_pd(g+j-1), \
             _mm_mul_pd (_mm256_castpd256_pd128(TV), _mm_loadu_pd(d+j-1)))); \
          j += 2; \
        } \
        if (j<=nd) \
        { _mm_store_sd (g+j-1, _mm_add_pd (_mm_load_sd(g+j-1), \
             _mm_mul_sd (_mm256_castpd256_pd128(TV), _mm_load_sd(d+j-1)))); \
        } \
      } \
      g += nd; \
    } \
  } \
} while (0)

#else

#define ADD_GRAD2_00 \
do \
{ double tv; \
  int i, j; \
  if (nd==1) \
  { double d0 = d[0]; \
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
  { for (i = 0; i<nv; i++) \
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

static void add_grad2
( net_param *restrict g,  /* Array of derivatives to add to */
  net_value *restrict v,  /* Source unit values */
  net_param *restrict off,/* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value *restrict d,  /* Derivatives with respect to destination units */
  int nd,		  /* Number of destination units */
  char *restrict omit,	  /* Omit flags, null if not present */
  int ob		  /* Bit to look at in omit flags */
)
{ 
  double tv;
  int i, j;

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
