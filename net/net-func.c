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

#include "misc.h"
#include "log.h"
#include "prior.h"
#include "data.h"
#include "model.h"
#include "net.h"


#if USE_SIMD_INTRINSICS && __AVX__
#include  <immintrin.h>
#endif


/* This module calculates the values of the output units in a network, given 
   values for the input units.  The values of hidden units are calculated
   along the way.  There are facilities for starting the calculation on the 
   assumption the values are already known up to some layer, as would be 
   the case if the weights into earlier layers have not changed since the
   last calculation. 
*/

#define sqrt_2 1.4142135623730950488

static void bias_values (net_value *, int, net_param *);

static void add_connections (net_value *, int, net_value *, int, 
                             net_param *, net_param *, char *, int);


/* EVALUATE NETWORK FUNCTION FOR GIVEN INPUTS.  The inputs are taken from
   the net_values structure passed.  When 'start' is greater than zero, the
   correct unit values for that number of hidden layers are assumed to be
   already present in the net_values structure. */

void net_func 
( net_values *v,	/* Place to get inputs and store outputs */
  int start,		/* Number of hidden layers with known values */
  net_arch *a,		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  net_params *w		/* Network parameters */
)
{
  int l, j;

  /* Compute values for successive hidden layers. */

  for (l = start; l<a->N_layers; l++)
  {
    int N_hidden = a->N_hidden[l];

    net_value *sh = v->s[l];

    if (a->has_bh[l])
    { bias_values (sh, N_hidden, w->bh[l]);
    }
    else
    { memset (sh, 0, N_hidden * sizeof *sh);
    }

    if (a->has_ih[l])
    { add_connections (sh, N_hidden, v->i, a->N_inputs, 
          w->ih[l], a->has_ti ? w->ti : 0, flgs ? flgs->omit : 0, 1<<(l+1));
    }

    if (l>0 && a->has_hh[l-1])
    { add_connections (sh, N_hidden, v->h[l-1], a->N_hidden[l-1],
          w->hh[l-1], a->has_th[l-1] ? w->th[l-1] : 0, (char *) 0, 0);
    }

    /* Put values through hidden unit activation function. */

    net_value *vh = v->h[l];

    if (flgs==0 || flgs->layer_type[l]==Tanh_type)
    { for (j = 0; j<N_hidden; j++)
      { vh[j] = tanh(sh[j]);
      }
    }
    else if (flgs->layer_type[l]==Sin_type)
    { for (j = 0; j<N_hidden; j++)
      { vh[j] = sqrt_2*sin(sh[j]*sqrt_2);
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
  { bias_values (v->o, a->N_outputs, w->bo);
  }
  else
  { memset (v->o, 0, a->N_outputs * sizeof *v->o);
  }

  if (a->has_io)
  { add_connections (v->o, a->N_outputs, v->i, a->N_inputs,
                     w->io, a->has_ti ? w->ti : 0, flgs ? flgs->omit : 0, 1);
  }

  for (l = 0; l<a->N_layers; l++)
  { if (a->has_ho[l])
    { add_connections (v->o, a->N_outputs, v->h[l], a->N_hidden[l], 
                       w->ho[l], a->has_th[l] ? w->th[l] : 0, (char *) 0, 0);
    }
  }
}


/* SET UNIT VALUES TO BIASES. */

static void bias_values
( net_value *restrict v,	/* Array of unit values to set */
  int n,			/* Number of units */
  net_param *restrict b		/* Biases */
)
{ 
# if USE_SIMD_INTRINSICS && __AVX__
  { int j;
    j = 3;
    while (j<n)
    { _mm256_storeu_pd (v+j-3, _mm256_loadu_pd(b+j-3));
      j += 4;
    }
    j -= 2;
    if (j<n)
    { _mm_storeu_pd (v+j-1, _mm_loadu_pd(b+j-1));
      j += 2;
    }
    if (j<=n)
    { _mm_store_sd (v+j-1, _mm_load_sd(b+j-1));
    }
  }
# else
  { int j;
    for (j = 0; j<n; j++) v[j] = *b++;
  }
# endif
}


/* ADD CONTRIBUTION FROM ONE GROUP OF CONNECTIONS.  Adds the weighted input
   due to connections from one source layer to the current unit values for
   the destination layer. */

#define ADD_CONNECTIONS(offset,omit) \
do \
{ int i, j; \
  double o; \
  if (nd==1) \
  { double sv[4] = { 0, 0, 0, 0 }; \
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

#if USE_SIMD_INTRINSICS && __AVX__ && USE_FMA && __FMA__

#define ADD_CONNECTIONS00 \
do \
{ int i, j; \
  if (nd==1) \
  { __m256d SV = _mm256_setzero_pd(); \
    i = 3; \
    while (i<ns) \
    { SV = _mm256_fmadd_pd (_mm256_loadu_pd(v+i-3), \
                            _mm256_loadu_pd(w+i-3), SV); \
      i += 4; \
    } \
    __m128d S; \
    S = _mm_add_pd (_mm256_castpd256_pd128(SV), \
                    _mm256_extractf128_pd(SV,1)); \
    i -= 2; \
    if (i<ns) \
    { S = _mm_fmadd_pd (_mm_loadu_pd(v+i-1), _mm_loadu_pd(w+i-1), S); \
      i += 2; \
    } \
    S = _mm_hadd_pd(S,S); \
    if (i<=ns) \
    { S = _mm_fmadd_sd (_mm_load_sd(v+i-1), _mm_load_sd(w+i-1), S); \
    } \
    S = _mm_add_sd (_mm_load_sd(s), S); \
    _mm_store_sd (s, S); \
  } \
  else \
  { for (i = 0; i<ns; i++, w+=nd) \
    { net_value tv = v[i]; \
      if (tv==0)  \
      { continue; \
      } \
      j = 3; \
      __m256d TV = _mm256_set1_pd(tv); \
      while (j<nd) \
      { _mm256_storeu_pd (s+j-3, _mm256_fmadd_pd (TV, \
                           _mm256_loadu_pd(w+j-3), _mm256_loadu_pd(s+j-3))); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { _mm_storeu_pd (s+j-1, _mm_fmadd_pd (_mm256_castpd256_pd128(TV), \
                        _mm_loadu_pd(w+j-1), _mm_loadu_pd(s+j-1))); \
        j += 2; \
      } \
      if (j<=nd) \
      { _mm_store_sd (s+j-1, _mm_fmadd_sd (_mm256_castpd256_pd128(TV), \
                       _mm_load_sd(w+j-1), _mm_load_sd(s+j-1))); \
      } \
    } \
  } \
} while (0)

#elif USE_SIMD_INTRINSICS && __AVX__

#define ADD_CONNECTIONS00 \
do \
{ int i, j; \
  if (nd==1) \
  { __m256d SV = _mm256_setzero_pd(); \
    i = 3; \
    while (i<ns) \
    { SV = _mm256_add_pd (SV, _mm256_mul_pd (_mm256_loadu_pd(v+i-3), \
                                             _mm256_loadu_pd(w+i-3))); \
      i += 4; \
    } \
    __m128d S; \
    S = _mm_add_pd (_mm256_castpd256_pd128(SV), \
                    _mm256_extractf128_pd(SV,1)); \
    i -= 2; \
    if (i<ns) \
    { S = _mm_add_pd (S, _mm_mul_pd(_mm_loadu_pd(v+i-1),_mm_loadu_pd(w+i-1))); \
      i += 2; \
    } \
    S = _mm_hadd_pd(S,S); \
    if (i<=ns) \
    { S = _mm_add_sd (S, _mm_mul_sd (_mm_load_sd(v+i-1), _mm_load_sd(w+i-1))); \
    } \
    S = _mm_add_sd (_mm_load_sd(s), S); \
    _mm_store_sd (s, S); \
  } \
  else \
  { for (i = 0; i<ns; i++, w+=nd) \
    { net_value tv = v[i]; \
      if (tv==0)  \
      { continue; \
      } \
      j = 3; \
      __m256d TV = _mm256_set1_pd(tv); \
      while (j<nd) \
      { _mm256_storeu_pd (s+j-3, _mm256_add_pd (_mm256_loadu_pd(s+j-3), \
                                   _mm256_mul_pd(_mm256_loadu_pd(w+j-3),TV))); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { _mm_storeu_pd (s+j-1, _mm_add_pd (_mm_loadu_pd(s+j-1), \
          _mm_mul_pd (_mm_loadu_pd(w+j-1), _mm256_castpd256_pd128(TV)))); \
        j += 2; \
      } \
      if (j<=nd) \
      { _mm_store_sd (s+j-1, _mm_add_sd (_mm_load_sd(s+j-1), \
          _mm_mul_sd (_mm_load_sd(w+j-1), _mm256_castpd256_pd128(TV)))); \
      } \
    } \
  } \
} while (0)

#else

#define ADD_CONNECTIONS00 ADD_CONNECTIONS(0,0)

#endif

static void add_connections
( net_value *restrict s,  /* Summed input for destination units to add to */
  int nd,		  /* Number of destination units */
  net_value *restrict v,  /* Values for source units */
  int ns,		  /* Number of source units */
  net_param *restrict w,  /* Connection weights */
  net_param *restrict off,/* Offsets to add to source unit values */
  char *restrict omit,	  /* Omit flags, null if not present */
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
