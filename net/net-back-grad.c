/* NET-BACK-GRAD.C - Combined backpropagation and gradient computation. */

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

#define LOG2 0.69314718055994530941  /* Set to log(2) */


/* -------------------------------------------------------------------------- */


/* SUM UP CONTRIBUTIONS TO THE DERIVATIVES FROM ONE GROUP OF CONNECTIONS.  
   Adds the weighted sum of derivatives due to connections from source
   units to a given destination layer to the totals for the source layer. */

static void sum_derivatives
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

  if (CHECK_NAN)
  { for (i = 0; i<ns; i++)
    { if (isnan(ds[i])) abort();
    }
  }
}


/* SUM UP CONTRIBUTIONS TO THE DERIVATIVES FROM CONNECTIONS WITH CONFIGURATION.
   Adds the weighted sum of derivatives due to connections from source units to
   a given destination layer to the totals for the source layer. */

static void sum_derivatives_config
( net_value const* dd,    /* Derivatives with respect to destination units */
  net_value *restrict ds, /* Derivatives w.r.t. source units to add to */
  int ns,		  /* Number of source units, for debug check only */
  net_param const* w,     /* Connection weights */
  net_config const* cf    /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  if (CONFIG_OCT_S_8D_8W && (cn = cf->oct_s_8d_8w))
  {
#   if FP64 && USE_SIMD_INTRINSICS && __AVX__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { __m256d P;
        j = cn[c].d;
        i = cn[c].s;
        P = _mm256_mul_pd (_mm256_loadu_pd(dd+j), _mm256_loadu_pd(w+k));
        P = FMA256_pd (_mm256_loadu_pd(dd+j+4), _mm256_loadu_pd(w+k+4), P);
        __m128d S = _mm_add_pd (_mm256_extractf128_pd(P,1), cast128d(P));
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
      }
    }
#   elif FP64 && USE_SIMD_INTRINSICS && __SSE3__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { __m128d S;
        j = cn[c].d;
        i = cn[c].s;
        S = _mm_add_pd (_mm_mul_pd (_mm_loadu_pd(dd+j), _mm_loadu_pd(w+k)),
                        _mm_mul_pd (_mm_loadu_pd(dd+j+2), _mm_loadu_pd(w+k+2)));
        S = _mm_add_pd (S,
             _mm_add_pd(_mm_mul_pd (_mm_loadu_pd(dd+j+4), _mm_loadu_pd(w+k+4)),
                        _mm_mul_pd (_mm_loadu_pd(dd+j+6), _mm_loadu_pd(w+k+6)))
            );
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __AVX__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { j = cn[c].d;
        i = cn[c].s;
        __m256 B = _mm256_mul_ps (_mm256_loadu_ps(dd+j), _mm256_loadu_ps(w+k));
        __m128 P = _mm_add_ps (cast128f(B), _mm256_extractf128_ps(B,1));
        __m128 S = _mm_add_ps (_mm_movehl_ps(P,P), P);
        _mm_store_ss (ds+i, _mm_add_ss (_mm_load_ss(ds+i), _mm_hadd_ps(S,S)));
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __SSE3__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { __m128 P;
        j = cn[c].d;
        i = cn[c].s;
        P = _mm_mul_ps (_mm_loadu_ps(dd+j), _mm_loadu_ps(w+k));
        P = FMA_ps (_mm_loadu_ps(dd+j+4), _mm_loadu_ps(w+k+4), P);
        __m128 S = _mm_add_ps (_mm_movehl_ps(P,P), P);
        _mm_store_ss (ds+i, _mm_add_ss (_mm_load_ss(ds+i), _mm_hadd_ps(S,S)));
      }
    }
#   else
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { j = cn[c].d;
        i = cn[c].s;
        ds[i] = ds[i] + dd[j+0]*w[k+0] + dd[j+1]*w[k+1]
                      + dd[j+2]*w[k+2] + dd[j+3]*w[k+3]
                      + dd[j+4]*w[k+4] + dd[j+5]*w[k+5]
                      + dd[j+6]*w[k+6] + dd[j+7]*w[k+7];
      }
    }
#   endif
  }

  if (CONFIG_QUAD_S_4D_4W && (cn = cf->quad_s_4d_4w))
  {
#   if FP64 && USE_SIMD_INTRINSICS && __AVX__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { j = cn[c].d;
        i = cn[c].s;
        __m256d P = _mm256_mul_pd (_mm256_loadu_pd(dd+j), _mm256_loadu_pd(w+k));
        __m128d S = _mm_add_pd (_mm256_extractf128_pd(P,1), cast128d(P));
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
      }
    }
#   elif FP64 && USE_SIMD_INTRINSICS && __SSE3__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { j = cn[c].d;
        i = cn[c].s;
        __m128d S = _mm_add_pd (
                     _mm_mul_pd (_mm_loadu_pd(dd+j), _mm_loadu_pd(w+k)),
                     _mm_mul_pd (_mm_loadu_pd(dd+j+2), _mm_loadu_pd(w+k+2)));
        _mm_store_sd (ds+i, _mm_add_sd (_mm_load_sd(ds+i), _mm_hadd_pd(S,S)));
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __SSE3__
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { j = cn[c].d;
        i = cn[c].s;
        __m128 P = _mm_mul_ps (_mm_loadu_ps(dd+j), _mm_loadu_ps(w+k));
        __m128 S = _mm_add_ps (_mm_movehl_ps(P,P), P);
        _mm_store_ss (ds+i, _mm_add_ss (_mm_load_ss(ds+i), _mm_hadd_ps(S,S)));
      }
    }
#   else
    { for (c = 0; (k = cn[c].w) >= 0; c++)
      { j = cn[c].d;
        i = cn[c].s;
        ds[i] += (dd[j+0]*w[k+0] + dd[j+2]*w[k+2])      /* same order as SIMD */
                   + (dd[j+1]*w[k+1] + dd[j+3]*w[k+3]); /* instructions above */
      }
    }
#   endif
  }

  if (CONFIG_QUAD_S_4D_4W && MAKE_QUAD_PAIRS && (cn = cf->quad_s_4d_4w_2))
  {
#   if FP64 && USE_SIMD_INTRINSICS && __AVX__
    { for (c = 0; (k = cn[c].w) >= 0; c+=2)
      { __m256d WK = _mm256_loadu_pd(w+k);
        __m256d P;
        __m128d S;
        j = cn[c].d;
        i = cn[c].s;
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
        j = cn[c].d;
        i = cn[c].s;
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
        j = cn[c].d;
        i = cn[c].s;
        P = _mm_mul_ps (_mm_loadu_ps(dd+j), WK);
        S = _mm_add_ps (_mm_movehl_ps(P,P), P);
        _mm_store_ss (ds+i, _mm_add_ss (_mm_load_ss(ds+i), _mm_hadd_ps(S,S)));
        j = cn[c+1].d;
        i = cn[c+1].s;
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
        j = cn[c].d;
        i = cn[c].s;
        ds[i] += (dd[j+0]*w0 + dd[j+2]*w2)      /* same order as SIMD */
                   + (dd[j+1]*w1 + dd[j+3]*w3); /* instructions above */
        j = cn[c+1].d;
        i = cn[c+1].s;
        ds[i] += (dd[j+0]*w0 + dd[j+2]*w2)      /* same order as SIMD */
                   + (dd[j+1]*w1 + dd[j+3]*w3); /* instructions above */
      }
    }
#   endif
  }

  if (CONFIG_SINGLE4 && (cn = cf->single4_s))
  { 
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
  }

  if (CONFIG_SINGLE4 && (cn = cf->single4_d))
  {
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

  if (cn = CONFIG_ORIGINAL ? cf->conn : cf->single)
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { i = cn[c].s; j = cn[c].d;
      ds[i] += dd[j] * w[k];
    }
  }

  if (CHECK_NAN)
  { for (i = 0; i<ns; i++)
    { if (isnan(ds[i])) abort();
    }
  }
}

#if __CUDACC__

/* SUM CONTRIBUTIONS TO THE DERIVATIVES FROM ONE GROUP OF CONNECTIONS, IN GPU.*/

__device__ static void sum_derivatives_gpu
( int th,		  /* Which thread */
  net_value const* dd,    /* Derivatives with respect to destination units */
  int nd,		  /* Number of destination units */
  net_value *restrict ds, /* Derivatives w.r.t. source units to add to */
  int ns,		  /* Number of source units */
# if STATIC_GPU_PARAMETERS
  net_param const* w0,    /* Connection weights */
# else
  net_param const* w,     /* Connection weights */
# endif
  unsigned short const* omit,  /* Omit flags, null if not present */
  int bit,		  /* Bit to look at in omit flags */
  unsigned syncmask       /* Mask of active threads for backprop */
)
{
# if STATIC_GPU_PARAMETERS
  /* Try to inform the compiler that parameters are in global memory */
  net_param const* w = &dev_param_block [w0 - dev_param_block];
# endif

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
  if (SYNC_AFTER && ns % NTH != 0) __syncwarp(syncmask);
}


/* SUM CONTRIBUTIONS TO DERIVATIVES FROM CONNECTIONS WITH CONFIG., IN GPU. */

__device__ static void sum_derivatives_config_gpu
( int th,		  /* Which thread */
  net_value const* dd,    /* Derivatives with respect to destination units */
  net_value *restrict ds, /* Derivatives w.r.t. source units to add to */
# if STATIC_GPU_PARAMETERS
  net_param const* w0,    /* Connection weights */
# else
  net_param const* w,     /* Connection weights */
# endif
  net_config const* cf,   /* Configuration for connections and weights */
  unsigned syncmask       /* Mask of active threads for backprop */
)
{
# if STATIC_GPU_PARAMETERS
  /* Try to inform the compiler that parameters are in global memory */
  net_param const* w = &dev_param_block [w0 - dev_param_block];
# endif

  net_connection *cn;
  int c, i, j, k;

  if (CONFIG_OCT_GPU_S_8D_8W_BW && (cn = cf->oct_s_8d_8w_sgpu))
  { c = cf->start_oct_sgpu[th];
    for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      ds[i] = ds[i] + (dd[j+0]*w[k+0] + dd[j+1]*w[k+1])
                    + (dd[j+2]*w[k+2] + dd[j+3]*w[k+3])
                    + (dd[j+4]*w[k+4] + dd[j+5]*w[k+5])
                    + (dd[j+6]*w[k+6] + dd[j+7]*w[k+7]);
    }
    if (SYNC_AFTER) __syncwarp(syncmask);
  }

  if (CONFIG_QUAD_GPU_S_4D_4W_BW && (cn = cf->quad_s_4d_4w_sgpu))
  { c = cf->start_quad_sgpu[th];
    for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      ds[i] = ds[i] + (dd[j+0]*w[k+0] + dd[j+1]*w[k+1])
                    + (dd[j+2]*w[k+2] + dd[j+3]*w[k+3]);
    }
    if (SYNC_AFTER) __syncwarp(syncmask);
  }

  if (cn = cf->other_sgpu)
  { c = cf->start_other_sgpu[th];
    for (;;)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      if (k<0) break;
      ds[i] += dd[j] * w[k];
    }
    if (SYNC_AFTER) __syncwarp(syncmask);
  }
}

#endif


/* -------------------------------------------------------------------------- */


/* ADD TO GRADIENT FROM UNIT DERIVATIVE. */

static void add_grad1
( net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* d,     /* Derivatives with respect to unit values */
  int n			  /* Number of units */
)
{ 
  int i;
  for (i = 0; i<n; i++)
  { g[i] += d[i];
    if (CHECK_NAN && isnan(g[i])) abort();
  }
}


/* ADD TO GRADIENT FROM UNIT DERIVATIVE, WITH CONFIGURATION. */

static void add_grad1_config
( net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* d,     /* Derivatives with respect to unit values */
  net_config const* cf    /* Configuration for biases */
)
{ net_connection *cn;
  int c, j, j2, k;

  if (CONFIG_OCT_S_8D_8W && (cn = cf->oct_s_8d_8w))
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { j = cn[c].d;
      g[k+0] += d[j+0];
      g[k+1] += d[j+1];
      g[k+2] += d[j+2];
      g[k+3] += d[j+3];
      g[k+4] += d[j+4];
      g[k+5] += d[j+5];
      g[k+6] += d[j+6];
      g[k+7] += d[j+7];
    }
  }

  if (CONFIG_QUAD_S_4D_4W && (cn = cf->quad_s_4d_4w))
  { for (c = 0; (k = cn[c].w) >= 0; c++)
    { j = cn[c].d;
      g[k+0] += d[j+0];
      g[k+1] += d[j+1];
      g[k+2] += d[j+2];
      g[k+3] += d[j+3];
    }
  }

  if (CONFIG_QUAD_S_4D_4W && MAKE_QUAD_PAIRS && (cn = cf->quad_s_4d_4w_2))
  { for (c = 0; (k = cn[c].w) >= 0; c+=2)
    { j = cn[c].d; j2 = cn[c+1].d;
      g[k+0] += d[j+0] + d[j2+0];
      g[k+1] += d[j+1] + d[j2+1];
      g[k+2] += d[j+2] + d[j2+2];
      g[k+3] += d[j+3] + d[j2+3];
    }
  }

  if (cn = CONFIG_ORIGINAL ? cf->conn : cf->single)
  { for (c = 0; (k = cn[c].w) >= 0; c++)
   { j = cn[c].d;
      g[k] += d[j];
   }
  }

  if (CHECK_NAN)
  { for (k = 0; k<cf->N_wts; k++)
    { if (isnan(g[k])) abort();
    }
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

static void add_grad2
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


/* ADD TO GRADIENT FROM PRODUCT OF UNIT VALUE AND UNIT DERIV., WITH CONFIG. */

static void add_grad2_config
( net_param *restrict g,  /* Array of derivatives to add to */
  net_value const* s,     /* Source unit values */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  net_value const* d,     /* Derivatives with respect to destination units */
  net_config const* cf    /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  if (CONFIG_OCT_S_8D_8W && (cn = cf->oct_s_8d_8w))
  {
#   if FP64 && USE_SIMD_INTRINSICS && __AVX__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256d SI = _mm256_set1_pd (s[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (g+k, FMA256_pd (SI, _mm256_loadu_pd(d+j),
                                                _mm256_loadu_pd(g+k)));
          _mm256_storeu_pd (g+k+4, FMA256_pd (SI, _mm256_loadu_pd(d+j+4),
                                                  _mm256_loadu_pd(g+k+4)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256d SI = _mm256_set1_pd (s[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (g+k, FMA256_pd (SI, _mm256_loadu_pd(d+j),
                                                _mm256_loadu_pd(g+k)));
          _mm256_storeu_pd (g+k+4, FMA256_pd (SI, _mm256_loadu_pd(d+j+4),
                                                  _mm256_loadu_pd(g+k+4)));
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
          _mm_storeu_pd (g+k+4, FMA_pd (SI, _mm_loadu_pd(d+j+4),
                                            _mm_loadu_pd(g+k+4)));
          _mm_storeu_pd (g+k+6, FMA_pd (SI, _mm_loadu_pd(d+j+6),
                                            _mm_loadu_pd(g+k+6)));
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
          _mm_storeu_pd (g+k+4, FMA_pd (SI, _mm_loadu_pd(d+j+4),
                                            _mm_loadu_pd(g+k+4)));
          _mm_storeu_pd (g+k+6, FMA_pd (SI, _mm_loadu_pd(d+j+6),
                                            _mm_loadu_pd(g+k+6)));
        }
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __AVX__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256 SI = _mm256_set1_ps (s[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_ps (g+k, FMA256_ps(SI, _mm256_loadu_ps(d+j), 
                                               _mm256_loadu_ps(g+k)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256 SI = _mm256_set1_ps (s[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_ps (g+k, FMA256_ps(SI, _mm256_loadu_ps(d+j), 
                                               _mm256_loadu_ps(g+k)));
        }
      }
    }
#   elif FP32 && USE_SIMD_INTRINSICS && __SSE2__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m128 SI = _mm_set1_ps (s[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_ps (g+k, FMA_ps(SI, _mm_loadu_ps(d+j), _mm_loadu_ps(g+k)));
          _mm_storeu_ps (g+k+4, FMA_ps(SI, _mm_loadu_ps(d+j+4), 
                                           _mm_loadu_ps(g+k+4)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m128 SI = _mm_set1_ps (s[cn[c].s]);
          j = cn[c].d;
          _mm_storeu_ps (g+k+4, FMA_ps(SI, _mm_loadu_ps(d+j+4), 
                                           _mm_loadu_ps(g+k+4)));
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
          g[k+4] += soi * d[j+4];
          g[k+5] += soi * d[j+5];
          g[k+6] += soi * d[j+6];
          g[k+7] += soi * d[j+7];
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
          g[k+4] += si * d[j+4];
          g[k+5] += si * d[j+5];
          g[k+6] += si * d[j+6];
          g[k+7] += si * d[j+7];
        }
      }
    }
#   endif
  }

  if (CONFIG_QUAD_S_4D_4W && (cn = cf->quad_s_4d_4w))
  {
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

  if (CONFIG_QUAD_S_4D_4W && MAKE_QUAD_PAIRS && (cn = cf->quad_s_4d_4w_2))
  {
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

  if (CONFIG_SINGLE4 && (cn = cf->single4_s))
  {
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
  }

  if (CONFIG_SINGLE4 && (cn = cf->single4_d))
  {
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

  if (cn = CONFIG_ORIGINAL ? cf->conn : cf->single)
  { if (off)
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

  if (CHECK_NAN)
  { int k;
    for (k = 0; k<cf->N_wts; k++)
    { if (isnan(g[k])) abort();
    }
  }
}

#if __CUDACC__ 

/* STORE GRADIENT FOR BIASES FOR 1 CASE.  The thread mod scheme is
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
  { g[ILV*i] = d0[i];
  }
}


/* STORE GRADIENT FOR BIASES FOR 1 CASE, WITH CONFIGURATION.  The
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
  int c, j, k;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[ILV*k] = 0;
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();

  if (CONFIG_OCT_GPU_S_8D_8W_GRAD && (cn = cf->oct_s_8d_8w_wgpu))
  { int thmod8 = (th&7) + 8;
    int m, ix;
    for (m = 0; m<8; m++)
    { ix = (thmod8-m) & 7;
      c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
      j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*(kk+ix)];
        do
        { s += d0[j+ix];
          j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*(kk+ix)] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (CONFIG_QUAD_GPU_S_4D_4W_GRAD && (cn = cf->quad_s_4d_4w_wgpu))
  { int thmod4 = (th&3) + 4;
    int m, ix;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m) & 3;
      c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
      j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*(kk+ix)];
        do
        { s += d0[j+ix];
          j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*(kk+ix)] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (cn = cf->other_wgpu)
  { c = cf->start_other_wgpu[th];
    j = cn[c].d; k = cn[c].w; c += 1;
    while (k>=0)
    { int kk = k;
      net_value s = g[ILV*kk];
      do
      { s += d0[j];
        j = cn[c].d; k = cn[c].w; c += 1;
      } while (k==kk);
      g[ILV*kk] = s;
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }
}


/* STORE GRADIENT FOR WEIGHTS FOR 1 CASE.  The thread mod scheme is
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
        g[ILV*i] = (v0[i]+o)*d00; \
      } \
    } \
    else \
    { for (i = th; i<nv; i+=GTH) \
      { g[ILV*i] = v0[i]*d00; \
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
        { g[ILV*j] = 0; \
        } \
      } \
      else \
      { for (j = th; j<nd; j+=GTH) \
        { g[ILV*j] = tv0*d0[j]; \
        } \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
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
      { g[ILV*j] = tv0*d0[j]; \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
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


/* STORE GRADIENT FOR WEIGHTS FOR 1 CASE, WITH CONFIGURATION.  The
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
  int i, j, k, c;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[ILV*k] = 0;
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();

  if (CONFIG_OCT_GPU_S_8D_8W_GRAD && (cn = cf->oct_s_8d_8w_wgpu))
  { int thmod8 = (th&7) + 8;
    int m, ix;
    if (off)
    { for (m = 0; m<8; m++)
      { ix = (thmod8-m) & 7;
        c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { s += (s0[i]+off[i]) * d0[j+ix];
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    else
    { for (m = 0; m<8; m++)
      { ix = (thmod8-m) & 7;
        c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { s += s0[i] * d0[j+ix];
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (CONFIG_QUAD_GPU_S_4D_4W_GRAD && (cn = cf->quad_s_4d_4w_wgpu))
  { int thmod4 = (th&3) + 4;
    int m, ix;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { s += (s0[i]+off[i]) * d0[j+ix];
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { s += s0[i] * d0[j+ix];
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (cn = cf->other_wgpu)
  { c = cf->start_other_wgpu[th];
    if (off)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*kk];
        do
        { s += (s0[i]+off[i])*d0[j];
          i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*kk] = s;
      }
    }
    else
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*kk];
        do
        { s += s0[i]*d0[j];
          i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*kk] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }
}

#endif


#if __CUDACC__

/* STORE GRADIENT FOR BIASES FOR 2 CASES.  The thread mod scheme is
   based on indexes for the biases/destination units. */

__device__ static void net_store2_grad1
( int th,                 /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivs for following cases */
  int n	                  /* Number of units */ 
)
{ 
  int i;
  for (i = th; i<n; i+=GTH)
  { net_value const*restrict d = d0+i;
    net_value s = *d;
    d += ds; s += *d;
    g[ILV*i] = s;
  }
}


/* STORE GRADIENT FOR BIASES FOR 2 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on indexes for the biases.  Note that
   the connections in quad_s_4d_4w_wgpu, other_wgpu, and other_2_wgpu
   come in GTH sections. */

__device__ static void net_store2_grad1_config
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivs for following cases */
  net_config const*restrict cf  /* Configuration for biases */
)
{ net_connection *cn;
  int c, j, k;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[ILV*k] = 0;
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();

  if (CONFIG_OCT_GPU_S_8D_8W_GRAD && (cn = cf->oct_s_8d_8w_wgpu))
  { int thmod8 = (th&7) + 8;
    int m, ix;
    for (m = 0; m<8; m++)
    { ix = (thmod8-m) & 7;
      c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
      j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*(kk+ix)];
        do
        { net_value const*restrict d = d0+j+ix;
          s += *d; d += ds;
          s += *d;
          j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*(kk+ix)] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (CONFIG_QUAD_GPU_S_4D_4W_GRAD && (cn = cf->quad_s_4d_4w_wgpu))
  { int thmod4 = (th&3) + 4;
    int m, ix;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m) & 3;
      c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
      j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*(kk+ix)];
        do
        { net_value const*restrict d = d0+j+ix;
          s += *d; d += ds;
          s += *d;
          j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*(kk+ix)] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }


  if (cn = cf->other_wgpu)
  { c = cf->start_other_wgpu[th];
    j = cn[c].d; k = cn[c].w; c += 1;
    while (k>=0)
    { int kk = k;
      net_value s = g[ILV*kk];
      do
      { net_value const*restrict d = d0+j;
        s += *d; d += ds;
        s += *d;
        j = cn[c].d; k = cn[c].w; c += 1;
      } while (k==kk);
      g[ILV*kk] = s;
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
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
  { net_value const*restrict d = d0; \
    net_value d00 = *d; d += ds; \
    net_value d10 = *d; \
    if (has_off) \
    { net_value o; \
      for (i = th; i<nv; i+=GTH) \
      { o = off[i]; \
        net_value const*restrict v = v0 + i; \
        net_value tv0 = *v; v += vs; \
        net_value tv1 = *v; \
        g[ILV*i] = (tv0+o)*d00 + (tv1+o)*d10; \
      } \
    } \
    else \
    { for (i = th; i<nv; i+=GTH) \
      { net_value const*restrict v = v0 + i; \
        net_value tv0 = *v; v += vs; \
        net_value tv1 = *v; \
        g[ILV*i] = tv0*d00 + tv1*d10; \
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
      net_value const*restrict v = v0 + i; \
      tv0 = *v + o; v += vs; \
      tv1 = *v + o; \
      if (tv0!=0) \
      { if (tv1!=0) goto alllab; \
        tvh = tv0; dh = d0; \
        goto onelab; \
      } \
      else if (tv1!=0) \
      { tvh = tv1; dh = d0+ds; \
        goto onelab; \
      } \
      for (j = th; j<nd; j+=GTH) \
      { g[ILV*j] = 0; \
      } \
      g += ILV*nd; \
      continue; \
    onelab: \
      for (j = th; j<nd; j+=GTH) \
      { g[ILV*j] = tvh * dh[j]; \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
      continue; \
    alllab: \
      for (j = th; j<nd; j+=GTH) \
      { net_value const*restrict d = d0 + j; \
        net_value td0 = *d; d += ds; \
        net_value td1 = *d; \
        g[ILV*j] = td0*tv0 + td1*tv1; \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
    } \
  } \
  else \
  { net_value tv0, tv1, o; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      net_value const*restrict v = v0 + i; \
      tv0 = *v + o; v += vs; \
      tv1 = *v + o; \
      for (j = th; j<nd; j+=GTH) \
      { net_value const*restrict d = d0 + j; \
        net_value td0 = *d; d += ds; \
        net_value td1 = *d; \
        g[ILV*j] = td0*tv0 + td1*tv1; \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
    } \
  } \
} while (0)

__device__ static void net_store2_grad2
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict v0,  /* Source unit values, case 0 */
  int vs,                 /* Stride to get to values for following cases */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivatives for following cases */
  int nd,		  /* Number of destination units */
  unsigned short const* omit, /* Omit flags, null if not present */
  int ob,		  /* Bit to look at in omit flags (mask, not number) */
  int sparse		  /* Might source unit values often be zero? */
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
  net_value const*restrict v0,  /* Source unit values, case 0 */
  int vs,                 /* Stride to get to values for following cases */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivatives for following cases */
  net_config const*restrict cf  /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[ILV*k] = 0;
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();
 
  if (CONFIG_OCT_GPU_S_8D_8W_GRAD && (cn = cf->oct_s_8d_8w_wgpu))
  { int thmod8 = (th&7) + 8;
    int m, ix;
    if (off)
    { for (m = 0; m<8; m++)
      { ix = (thmod8-m) & 7;
        c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_param o = off[i];
            net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; s += (tv+o)*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    else
    { for (m = 0; m<8; m++)
      { ix = (thmod8-m) & 7;
        c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; s += tv*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }
 
  if (CONFIG_QUAD_GPU_S_4D_4W_GRAD && (cn = cf->quad_s_4d_4w_wgpu))
  { int thmod4 = (th&3) + 4;
    int m, ix;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_param o = off[i];
            net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; s += (tv+o)*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; s += tv*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (cn = cf->other_wgpu)
  { c = cf->start_other_wgpu[th];
    if (off)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*kk];
        do
        { net_param o = off[i];
          net_value const*restrict v = v0+i;
          net_value const*restrict d = d0+j;
          net_value tv, td;
          tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
          tv = *v; td = *d; s += (tv+o)*td;
          i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*kk] = s;
      }
    }
    else
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*kk];
        do
        { net_value const*restrict v = v0+i;
          net_value const*restrict d = d0+j;
          net_value tv, td;
          tv = *v; td = *d; v += vs; d += ds; s += tv*td;
          tv = *v; td = *d; s += tv*td;
          i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*kk] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }
}

#endif


#if __CUDACC__

/* STORE GRADIENT FOR BIASES FOR 3 CASES.  The thread mod scheme is
   based on indexes for the biases/destination units. */

__device__ static void net_store3_grad1
( int th,                 /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivs for following cases */
  int n	                  /* Number of units */ 
)
{ 
  int i;
  for (i = th; i<n; i+=GTH)
  { net_value const*restrict d = d0+i;
    net_value s = *d;
    d += ds; s += *d; d += ds; s += *d;
    g[ILV*i] = s;
  }
}


/* STORE GRADIENT FOR BIASES FOR 3 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on indexes for the biases.  Note that
   the connections in quad_s_4d_4w_wgpu, other_wgpu, and other_2_wgpu
   come in GTH sections. */

__device__ static void net_store3_grad1_config
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivs for following cases */
  net_config const*restrict cf  /* Configuration for biases */
)
{ net_connection *cn;
  int c, j, k;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[ILV*k] = 0;
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();

  if (CONFIG_OCT_GPU_S_8D_8W_GRAD && (cn = cf->oct_s_8d_8w_wgpu))
  { int thmod8 = (th&7) + 8;
    int m, ix;
    for (m = 0; m<8; m++)
    { ix = (thmod8-m) & 7;
      c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
      j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*(kk+ix)];
        do
        { net_value const*restrict d = d0+j+ix;
          s += *d; d += ds;
          s += *d; d += ds;
          s += *d;
          j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*(kk+ix)] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (CONFIG_QUAD_GPU_S_4D_4W_GRAD && (cn = cf->quad_s_4d_4w_wgpu))
  { int thmod4 = (th&3) + 4;
    int m, ix;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m) & 3;
      c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
      j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*(kk+ix)];
        do
        { net_value const*restrict d = d0+j+ix;
          s += *d; d += ds;
          s += *d; d += ds;
          s += *d;
          j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*(kk+ix)] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (cn = cf->other_wgpu)
  { c = cf->start_other_wgpu[th];
    j = cn[c].d; k = cn[c].w; c += 1;
    while (k>=0)
    { int kk = k;
      net_value s = g[ILV*kk];
      do
      { net_value const*restrict d = d0+j;
        s += *d; d += ds;
        s += *d; d += ds;
        s += *d;
        j = cn[c].d; k = cn[c].w; c += 1;
      } while (k==kk);
      g[ILV*kk] = s;
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
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
  { net_value const*restrict d = d0; \
    net_value d00 = *d; d += ds; \
    net_value d10 = *d; d += ds; \
    net_value d20 = *d; \
    if (has_off) \
    { net_value o; \
      for (i = th; i<nv; i+=GTH) \
      { o = off[i]; \
        net_value const*restrict v = v0 + i; \
        net_value tv0 = *v; v += vs; \
        net_value tv1 = *v; v += vs; \
        net_value tv2 = *v; \
        g[ILV*i] = (tv0+o)*d00 + (tv1+o)*d10 + (tv2+o)*d20; \
      } \
    } \
    else \
    { for (i = th; i<nv; i+=GTH) \
      { net_value const*restrict v = v0 + i; \
        net_value tv0 = *v; v += vs; \
        net_value tv1 = *v; v += vs; \
        net_value tv2 = *v; \
        g[ILV*i] = tv0*d00 + tv1*d10 + tv2*d20; \
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
      net_value const*restrict v = v0 + i; \
      tv0 = *v + o; v += vs; \
      tv1 = *v + o; v += vs; \
      tv2 = *v + o; \
      if (tv0!=0) \
      { if (tv1!=0 || tv2!=0) goto alllab; \
        tvh = tv0; dh = d0; \
        goto onelab; \
      } \
      else if (tv1!=0) \
      { if (tv2!=0) goto alllab; \
        tvh = tv1; dh = d0+ds; \
        goto onelab; \
      } \
      else if (tv2!=0) \
      { tvh = tv2; dh = d0+ds+ds; \
        goto onelab; \
      } \
      for (j = th; j<nd; j+=GTH) \
      { g[ILV*j] = 0; \
      } \
      g += ILV*nd; \
      continue; \
    onelab: \
      for (j = th; j<nd; j+=GTH) \
      { g[ILV*j] = tvh * dh[j]; \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
      continue; \
    alllab: \
      for (j = th; j<nd; j+=GTH) \
      { net_value const*restrict d = d0 + j; \
        net_value td0 = *d; d += ds; \
        net_value td1 = *d; d += ds; \
        net_value td2 = *d; \
        g[ILV*j] = td0*tv0 + td1*tv1 + td2*tv2; \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
    } \
  } \
  else \
  { net_value tv0, tv1, tv2, o; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      net_value const*restrict v = v0 + i; \
      tv0 = *v + o; v += vs; \
      tv1 = *v + o; v += vs; \
      tv2 = *v + o; \
      for (j = th; j<nd; j+=GTH) \
      { net_value const*restrict d = d0 + j; \
        net_value td0 = *d; d += ds; \
        net_value td1 = *d; d += ds; \
        net_value td2 = *d; \
        g[ILV*j] = td0*tv0 + td1*tv1 + td2*tv2; \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
    } \
  } \
} while (0)

__device__ static void net_store3_grad2
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict v0,  /* Source unit values, case 0 */
  int vs,                 /* Stride to get to values for following cases */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivatives for following cases */
  int nd,		  /* Number of destination units */
  unsigned short const* omit, /* Omit flags, null if not present */
  int ob,		  /* Bit to look at in omit flags (mask, not number) */
  int sparse		  /* Might source unit values often be zero? */
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


/* STORE GRADIENT FOR WEIGHTS FOR 3 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on the indexes for the weights.  Note
   that the connections in quad_s_4d_4w_wgpu, other_wgpu, and
   other_2_wgpu come in GTH sections. */

__device__ static void net_store3_grad2_config
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to add to */
  net_value const*restrict v0,  /* Source unit values, case 0 */
  int vs,                 /* Stride to get to values for following cases */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivatives for following cases */
  net_config const*restrict cf  /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[ILV*k] = 0;
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();
 
  if (CONFIG_OCT_GPU_S_8D_8W_GRAD && (cn = cf->oct_s_8d_8w_wgpu))
  { int thmod8 = (th&7) + 8;
    int m, ix;
    if (off)
    { for (m = 0; m<8; m++)
      { ix = (thmod8-m) & 7;
        c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_param o = off[i];
            net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; s += (tv+o)*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    else
    { for (m = 0; m<8; m++)
      { ix = (thmod8-m) & 7;
        c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; s += tv*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }
 
  if (CONFIG_QUAD_GPU_S_4D_4W_GRAD && (cn = cf->quad_s_4d_4w_wgpu))
  { int thmod4 = (th&3) + 4;
    int m, ix;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_param o = off[i];
            net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; s += (tv+o)*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; s += tv*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (cn = cf->other_wgpu)
  { c = cf->start_other_wgpu[th];
    if (off)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*kk];
        do
        { net_param o = off[i];
          net_value const*restrict v = v0+i;
          net_value const*restrict d = d0+j;
          net_value tv, td;
          tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
          tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
          tv = *v; td = *d; s += (tv+o)*td;
          i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*kk] = s;
      }
    }
    else
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*kk];
        do
        { net_value const*restrict v = v0+i;
          net_value const*restrict d = d0+j;
          net_value tv, td;
          tv = *v; td = *d; v += vs; d += ds; s += tv*td;
          tv = *v; td = *d; v += vs; d += ds; s += tv*td;
          tv = *v; td = *d; s += tv*td;
          i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*kk] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }
}

#endif


#if __CUDACC__

/* STORE GRADIENT FOR BIASES FOR 4 CASES.  The thread mod scheme is
   based on indexes for the biases/destination units. */

__device__ static void net_store4_grad1
( int th,                 /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivs for following cases */
  int n	                  /* Number of units */ 
)
{ 
  int i;
  for (i = th; i<n; i+=GTH)
  { net_value const*restrict d = d0+i;
    net_value s = *d;
    d += ds; s += *d; d += ds; s += *d; d += ds; s += *d;
    g[ILV*i] = s;
  }
}


/* STORE GRADIENT FOR BIASES FOR 4 CASES, WITH CONFIGURATION.  The
   thread mod scheme is based on indexes for the biases.  Note that
   the connections in quad_s_4d_4w_wgpu, other_wgpu, and other_2_wgpu
   come in GTH sections. */

__device__ static void net_store4_grad1_config
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivs for following cases */
  net_config const*restrict cf  /* Configuration for biases */
)
{ net_connection *cn;
  int c, j, k;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[ILV*k] = 0;
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();

  if (CONFIG_OCT_GPU_S_8D_8W_GRAD && (cn = cf->oct_s_8d_8w_wgpu))
  { int thmod8 = (th&7) + 8;
    int m, ix;
    for (m = 0; m<8; m++)
    { ix = (thmod8-m) & 7;
      c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
      j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*(kk+ix)];
        do
        { net_value const*restrict d = d0+j+ix;
          s += *d; d += ds;
          s += *d; d += ds;
          s += *d; d += ds;
          s += *d;
          j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*(kk+ix)] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (CONFIG_QUAD_GPU_S_4D_4W_GRAD && (cn = cf->quad_s_4d_4w_wgpu))
  { int thmod4 = (th&3) + 4;
    int m, ix;
    for (m = 0; m<4; m++)
    { ix = (thmod4-m) & 3;
      c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
      j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*(kk+ix)];
        do
        { net_value const*restrict d = d0+j+ix;
          s += *d; d += ds;
          s += *d; d += ds;
          s += *d; d += ds;
          s += *d;
          j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*(kk+ix)] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (cn = cf->other_wgpu)
  { c = cf->start_other_wgpu[th];
    j = cn[c].d; k = cn[c].w; c += 1;
    while (k>=0)
    { int kk = k;
      net_value s = g[ILV*kk];
      do
      { net_value const*restrict d = d0+j;
        s += *d; d += ds;
        s += *d; d += ds;
        s += *d; d += ds;
        s += *d;
        j = cn[c].d; k = cn[c].w; c += 1;
      } while (k==kk);
      g[ILV*kk] = s;
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
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
  { net_value const*restrict d = d0; \
    net_value d00 = *d; d += ds; \
    net_value d10 = *d; d += ds; \
    net_value d20 = *d; d += ds; \
    net_value d30 = *d; \
    if (has_off) \
    { net_value o; \
      for (i = th; i<nv; i+=GTH) \
      { o = off[i]; \
        net_value const*restrict v = v0 + i; \
        net_value tv0 = *v; v += vs; \
        net_value tv1 = *v; v += vs; \
        net_value tv2 = *v; v += vs; \
        net_value tv3 = *v; \
        g[ILV*i] = (tv0+o)*d00 + (tv1+o)*d10 + (tv2+o)*d20 + (tv3+o)*d30; \
      } \
    } \
    else \
    { for (i = th; i<nv; i+=GTH) \
      { net_value const*restrict v = v0 + i; \
        net_value tv0 = *v; v += vs; \
        net_value tv1 = *v; v += vs; \
        net_value tv2 = *v; v += vs; \
        net_value tv3 = *v; \
        g[ILV*i] = tv0*d00 + tv1*d10 + tv2*d20 + tv3*d30; \
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
      net_value const*restrict v = v0 + i; \
      tv0 = *v + o; v += vs; \
      tv1 = *v + o; v += vs; \
      tv2 = *v + o; v += vs; \
      tv3 = *v + o; \
      if (tv0!=0) \
      { if (tv1!=0 || tv2!=0 || tv3!=0) goto alllab; \
        tvh = tv0; dh = d0; \
        goto onelab; \
      } \
      else if (tv1!=0) \
      { if (tv2!=0 || tv3!=0) goto alllab; \
        tvh = tv1; dh = d0+ds; \
        goto onelab; \
      } \
      else if (tv2!=0) \
      { if (tv3!=0) goto alllab; \
        tvh = tv2; dh = d0+ds+ds; \
        goto onelab; \
      } \
      else if (tv3!=0) \
      { tvh = tv3; dh = d0+ds+ds+ds; \
        goto onelab; \
      } \
      for (j = th; j<nd; j+=GTH) \
      { g[ILV*j] = 0; \
      } \
      g += ILV*nd; \
      continue; \
    onelab: \
      for (j = th; j<nd; j+=GTH) \
      { g[ILV*j] = tvh * dh[j]; \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
      continue; \
    alllab: \
      for (j = th; j<nd; j+=GTH) \
      { net_value const*restrict d = d0 + j; \
        net_value td0 = *d; d += ds; \
        net_value td1 = *d; d += ds; \
        net_value td2 = *d; d += ds; \
        net_value td3 = *d; \
        g[ILV*j] = td0*tv0 + td1*tv1 + td2*tv2 + td3*tv3; \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
    } \
  } \
  else \
  { net_value tv0, tv1, tv2, tv3, o; \
    int j; \
    for (i = 0; i<nv; i++) \
    { if (has_omit && (omit[i]&ob)) continue; \
      o = has_off ? off[i] : 0; \
      net_value const*restrict v = v0 + i; \
      tv0 = *v + o; v += vs; \
      tv1 = *v + o; v += vs; \
      tv2 = *v + o; v += vs; \
      tv3 = *v + o; \
      for (j = th; j<nd; j+=GTH) \
      { net_value const*restrict d = d0 + j; \
        net_value td0 = *d; d += ds; \
        net_value td1 = *d; d += ds; \
        net_value td2 = *d; d += ds; \
        net_value td3 = *d; \
        g[ILV*j] = td0*tv0 + td1*tv1 + td2*tv2 + td3*tv3; \
      } \
      if (SYNC_AFTER && GTH>=32 && nd % GTH != 0) __syncwarp(); \
      g += ILV*nd; \
    } \
  } \
} while (0)

__device__ static void net_store4_grad2
( int th,		  /* Which thread (0 to GTH-1) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict v0,  /* Source unit values, case 0 */
  int vs,                 /* Stride to get to values for following cases */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  int nv,		  /* Number of source units */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivatives for following cases */
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
  net_value const*restrict v0,  /* Source unit values, case 0 */
  int vs,                 /* Stride to get to values for following cases */
  net_param const* off,   /* Offsets for source units, or zero if no offsets */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivatives for following cases */
  net_config const*restrict cf  /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  for (k = th; k<cf->N_wts; k+=GTH)
  { g[ILV*k] = 0;
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();
 
  if (CONFIG_OCT_GPU_S_8D_8W_GRAD && (cn = cf->oct_s_8d_8w_wgpu))
  { int thmod8 = (th&7) + 8;
    int m, ix;
    if (off)
    { for (m = 0; m<8; m++)
      { ix = (thmod8-m) & 7;
        c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_param o = off[i];
            net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; s += (tv+o)*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    else
    { for (m = 0; m<8; m++)
      { ix = (thmod8-m) & 7;
        c = cf->start_oct_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; s += tv*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }
 
  if (CONFIG_QUAD_GPU_S_4D_4W_GRAD && (cn = cf->quad_s_4d_4w_wgpu))
  { int thmod4 = (th&3) + 4;
    int m, ix;
    if (off)
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_param o = off[i];
            net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
            tv = *v; td = *d; s += (tv+o)*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    else
    { for (m = 0; m<4; m++)
      { ix = (thmod4-m) & 3;
        c = cf->start_quad_wgpu [(th-ix+GTH) & (GTH-1)];
        i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        while (k>=0)
        { int kk = k;
          net_value s = g[ILV*(kk+ix)];
          do
          { net_value const*restrict v = v0+i;
            net_value const*restrict d = d0+j+ix;
            net_value tv, td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; v += vs; d += ds; s += tv*td;
            tv = *v; td = *d; s += tv*td;
            i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
          } while (k==kk);
          g[ILV*(kk+ix)] = s;
        }
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }

  if (cn = cf->other_wgpu)
  { c = cf->start_other_wgpu[th];
    if (off)
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*kk];
        do
        { net_param o = off[i];
          net_value const*restrict v = v0+i;
          net_value const*restrict d = d0+j;
          net_value tv, td;
          tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
          tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
          tv = *v; td = *d; v += vs; d += ds; s += (tv+o)*td;
          tv = *v; td = *d; s += (tv+o)*td;
          i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*kk] = s;
      }
    }
    else
    { i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
      while (k>=0)
      { int kk = k;
        net_value s = g[ILV*kk];
        do
        { net_value const*restrict v = v0+i;
          net_value const*restrict d = d0+j;
          net_value tv, td;
          tv = *v; td = *d; v += vs; d += ds; s += tv*td;
          tv = *v; td = *d; v += vs; d += ds; s += tv*td;
          tv = *v; td = *d; v += vs; d += ds; s += tv*td;
          tv = *v; td = *d; s += tv*td;
          i = cn[c].s; j = cn[c].d; k = cn[c].w; c += 1;
        } while (k==kk);
        g[ILV*kk] = s;
      }
    }
    if (SYNC_AFTER && GTH>=32) __syncwarp();
  }
}

#endif


/* -------------------------------------------------------------------------- */


/* FIND GRADIENT, USING BACKPROPAGATED DERIVATIVES.  Assumes that values
   of hidden units for the training case have been computed (in v->h[.]),
   and that the derivative of the energy with respect to the outputs has
   been computed (in d->o).  Uses other parts of d to store derivatives 
   of energy with respect to hidden units, and with respect to input units,
   if input offsets are present.  Takes parameters or transposed parameters
   from global 'params' or 'params_trans', and precomputed stuff from 'pre'. */

void STATIC_IF_INCLUDED net_back_add_grad
( net_params *restrict g, /* Gradient with respect to parameters to add to */
  net_values const*v,	/* Values for units in network */
  net_values *restrict d,/* Has output derivatives, storage for other derivs */
  net_arch const*a,	/* Network architecture */
  net_priors const*p,	/* Priors for parameters */
  net_flags const*flgs,	/* Network flags, null if none */
  int sparse            /* Might source unit values often be zero? */
)
{
  int l, ld, ls, i;

  /* Add parts of gradients that don't depend on computing derivatives
     with respect to hidden or input unit values - only on inputs and hidden
     unit values, and on derivatives with respect to outputs, */

  if (a->has_bo && !p->bo.one_or_two_point)
  { if (a->bias_config[a->N_layers])
    { add_grad1_config (g->bo, d->o, a->bias_config[a->N_layers]);
    }
    else
    { add_grad1 (g->bo, d->o, a->N_outputs);
    }
  }

  if (a->has_io && !p->io.one_or_two_point)
  { if (a->input_config[a->N_layers])
    { add_grad2_config (g->io, v->i, a->has_ti ? params.ti : 0, d->o,
                        a->input_config[a->N_layers]);
    }
    else
    { add_grad2 (g->io, v->i, a->has_ti ? params.ti : 0, a->N_inputs,
                 d->o, a->N_outputs,
                 a->any_omitted[a->N_layers] ? flgs->omit : 0, 1,
                 sparse);
    }
  }

  for (l = 0; l<a->N_layers; l++)
  {
    if (a->has_ho[l] && !p->ho[l].one_or_two_point)
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { add_grad2_config (g->ho[l], v->h[l], a->has_th[l] ? params.th[l] : 0,
                          d->o, a->hidden_config[k]);
      }
      else
      { add_grad2 (g->ho[l], v->h[l], a->has_th[l] ? params.th[l] : 0,
                a->N_hidden[l], d->o, a->N_outputs, (unsigned short *) 0, 0, 0);
      }
    }
  }

  /* Start computation of derivatives with respect to input values, if
     they will be needed, with possible contribution from input-output
     connections. */

  if (a->has_ti)
  { 
    memset (d->i, 0, a->N_inputs * sizeof *d->i);

    if (a->has_io)
    { if (a->input_config[a->N_layers])
      { sum_derivatives_config (d->o, d->i, a->N_inputs, params.io, 
                                a->input_config[a->N_layers]);
      }
      else
      { sum_derivatives (d->o, a->N_outputs, d->i, a->N_inputs, params.io,
                         a->any_omitted[a->N_layers] ? flgs->omit : 0, 1);
      }
    }
  }

  /* Go backwards through hidden layers, computing derivatives with respect to
     hidden unit values, and then adding to gradients that depend on these. */

  for (l = a->N_layers-1; l>=0; l--)
  {
    int N_hidden = a->N_hidden[l];

    /* Place to store derivatives computed for this hidden layer. */

    net_value *restrict dh = d->h[l];

    /* Find derivatives with respect to values of units in this hidden layer. */

    memset (dh, 0, N_hidden * sizeof *dh);

    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { sum_derivatives_config (d->o, dh, N_hidden, 
                                params.ho[l], a->hidden_config[k]);
      }
      else if (TRANS_WEIGHTS(N_hidden,a->N_outputs))
      { add_connections (dh, N_hidden, d->o, a->N_outputs, params_trans.ho[l],
                         (net_param const*) 0, (unsigned short *) 0, 0, 0);
      }
      else
      { sum_derivatives (d->o, a->N_outputs, dh, N_hidden,
                         params.ho[l], (unsigned short *) 0, 0);
      }
    }

    for (ld = l+1; ld<a->N_layers; ld++)
    { net_config *cf; net_param *wh; net_param *wht;
      if (ld==l+1)
      { if (!a->has_hh[l]) continue;
        cf = a->hidden_config[ld];
        wh = params.hh[l];
#       if USE_TRANSPOSED_WEIGHTS
        { if (TRANS_WEIGHTS(N_hidden,a->N_hidden[ld]))
          { wht = params_trans.hh[l];
          }
        }
#       endif
      }
      else
      { int nsqi = pre.nonseq[l][ld];
        if (nsqi<0) continue;
        cf = a->nonseq_config[nsqi];
        wh = params.nsq[nsqi];
#       if USE_TRANSPOSED_WEIGHTS
        { if (TRANS_WEIGHTS(N_hidden,a->N_hidden[ld]))
          { wht = params_trans.nsq[nsqi];
          }
        }
#       endif
      }
      net_value *restrict dhd = d->h[ld];
      if (cf)
      { sum_derivatives_config (dhd, dh, N_hidden, wh, cf);
      }
      else if (TRANS_WEIGHTS(N_hidden,a->N_hidden[ld]))
      { add_connections (dh, N_hidden, dhd, a->N_hidden[ld], wht,
                         (net_param const*) 0, (unsigned short *) 0, 0, 0);
      }
      else
      { sum_derivatives (dhd, a->N_hidden[ld], dh, N_hidden,
                         wh, (unsigned short *) 0, 0);
      }
    }

    /* Add to gradient with respect to hidden offsets, based on derivatives
       with respect to hidden unit values (before these are converted to
       derivatives with respect to the summed input, prior to the activation
       function). */

    if (a->has_th[l] && !p->th[l].one_or_two_point)
    { add_grad1 (g->th[l], dh, N_hidden);
    }

    /* Pass backwards through activation function to get derivatives with 
       respect to the summed inputs of units in this hidden layer. */

    net_value const* vh = v->h[l];

    if (a->layer_type[l]==Tanh_type)
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

    else if (a->layer_type[l]==Softplus_type)
    { 
#     if FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
      { __m256d ZERO = _mm256_setzero_pd();
        __m256d ONE = _mm256_set1_pd(1.0);
        i = 3;
        while (i<N_hidden)
        { __m256d E = sleef_expd4 (_mm256_sub_pd (ZERO,
                                                  _mm256_loadu_pd(vh+i-3)));
          _mm256_storeu_pd (dh+i-3, _mm256_mul_pd (_mm256_loadu_pd(dh+i-3),
                                                   _mm256_sub_pd(ONE,E)));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128d E = sleef_expd2 (_mm_sub_pd (cast128d(ZERO),
                                               _mm_loadu_pd(vh+i-1)));
          _mm_storeu_pd (dh+i-1, _mm_mul_pd (_mm_loadu_pd(dh+i-1),
                                             _mm_sub_pd(cast128d(ONE),E)));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i-1] *= 1 - prec_exp(-vh[i-1]);
        }
      }
#     elif FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
      { __m128d ZERO = _mm_setzero_pd();
        __m128d ONE = _mm_set1_pd(1.0);
        i = 1;
        while (i<N_hidden)
        { __m128d E = sleef_expd2 (_mm_sub_pd (ZERO,
                                               _mm_loadu_pd(vh+i-1)));
          _mm_storeu_pd (dh+i-1, _mm_mul_pd (_mm_loadu_pd(dh+i-1),
                                             _mm_sub_pd(ONE,E)));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i-1] *= 1 - prec_exp(-vh[i-1]);
        }
      }
#     elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
      { __m256 ZERO = _mm256_setzero_ps();
        __m256 ONE = _mm256_set1_ps(1.0f);
        i = 7;
        while (i<N_hidden)
        { __m256 E = sleef_expf8 (_mm256_sub_ps (ZERO,
                                                 _mm256_loadu_ps(vh+i-7)));
          _mm256_storeu_ps (dh+i-7, _mm256_mul_ps (_mm256_loadu_ps(dh+i-7),
                                                   _mm256_sub_ps(ONE,E)));
          i += 8;
        }
        i -= 4;
        if (i<N_hidden)
        { __m128 E = sleef_expf4 (_mm_sub_ps (cast128f(ZERO),
                                              _mm_loadu_ps(vh+i-3)));
          _mm_storeu_ps (dh+i-3, _mm_mul_ps (_mm_loadu_ps(dh+i-3),
                                             _mm_sub_ps(cast128f(ONE),E)));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128 E = sleef_expf4 (_mm_sub_ps (cast128f(ZERO),
                               _mm_loadl_pi(cast128f(ZERO),(__m64 *)(vh+i-1))));
          _mm_storel_pi ((__m64 *)(dh+i-1), _mm_mul_ps (_mm_loadu_ps(dh+i-1),
                                                  _mm_sub_ps(cast128f(ONE),E)));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i-1] *= 1 - prec_exp(-vh[i-1]);
        }
      }
#     elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
      { __m128 ZERO = _mm_setzero_ps();
        __m128 ONE = _mm_set1_ps(1.0f);
        i = 3;
        while (i<N_hidden)
        { __m128 E = sleef_expf4 (_mm_sub_ps (ZERO,
                                              _mm_loadu_ps(vh+i-3)));
          _mm_storeu_ps (dh+i-3, _mm_mul_ps (_mm_loadu_ps(dh+i-3),
                                             _mm_sub_ps(ONE,E)));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128 E = sleef_expf4 (_mm_sub_ps (ZERO,
                                         _mm_loadl_pi(ZERO,(__m64 *)(vh+i-1))));
          _mm_storel_pi ((__m64 *)(dh+i-1), _mm_mul_ps (_mm_loadu_ps(dh+i-1),
                                                        _mm_sub_ps(ONE,E)));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i-1] *= 1 - prec_exp(-vh[i-1]);
        }
      }
#     else
      { for (i = 0; i<N_hidden; i++)
        { dh[i] *= 1 - prec_exp(-vh[i]);
        }
      }
#     endif
    }

    else if (a->layer_type[l]==Softplus0_type)
    {
#     if FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
      { __m256d MINUS_LOG2 = _mm256_set1_pd(-LOG2);
        __m256d ONE = _mm256_set1_pd(1.0);
        i = 3;
        while (i<N_hidden)
        { __m256d E = sleef_expd4 (_mm256_sub_pd (MINUS_LOG2,
                                                  _mm256_loadu_pd(vh+i-3)));
          _mm256_storeu_pd (dh+i-3, _mm256_mul_pd (_mm256_loadu_pd(dh+i-3),
                                                   _mm256_sub_pd(ONE,E)));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128d E = sleef_expd2 (_mm_sub_pd (cast128d(MINUS_LOG2),
                                               _mm_loadu_pd(vh+i-1)));
          _mm_storeu_pd (dh+i-1, _mm_mul_pd (_mm_loadu_pd(dh+i-1),
                                             _mm_sub_pd(cast128d(ONE),E)));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i-1] *= 1 - prec_exp(-LOG2-vh[i-1]);
        }
      }
#     elif FP64 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
      { __m128d MINUS_LOG2 = _mm_set1_pd(-LOG2);
        __m128d ONE = _mm_set1_pd(1.0);
        i = 1;
        while (i<N_hidden)
        { __m128d E = sleef_expd2 (_mm_sub_pd (MINUS_LOG2,
                                               _mm_loadu_pd(vh+i-1)));
          _mm_storeu_pd (dh+i-1, _mm_mul_pd (_mm_loadu_pd(dh+i-1),
                                             _mm_sub_pd(ONE,E)));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i-1] *= 1 - prec_exp(-LOG2-vh[i-1]);
        }
      }
#     elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __AVX__
      { __m256 MINUS_LOG2 = _mm256_set1_ps(-LOG2);
        __m256 ONE = _mm256_set1_ps(1.0f);
        i = 7;
        while (i<N_hidden)
        { __m256 E = sleef_expf8 (_mm256_sub_ps (MINUS_LOG2,
                                                 _mm256_loadu_ps(vh+i-7)));
          _mm256_storeu_ps (dh+i-7, _mm256_mul_ps (_mm256_loadu_ps(dh+i-7),
                                                   _mm256_sub_ps(ONE,E)));
          i += 8;
        }
        i -= 4;
        if (i<N_hidden)
        { __m128 E = sleef_expf4 (_mm_sub_ps (cast128f(MINUS_LOG2),
                                              _mm_loadu_ps(vh+i-3)));
          _mm_storeu_ps (dh+i-3, _mm_mul_ps (_mm_loadu_ps(dh+i-3),
                                             _mm_sub_ps(cast128f(ONE),E)));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128 ZERO = _mm_setzero_ps();
          __m128 E = sleef_expf4 (_mm_sub_ps (cast128f(MINUS_LOG2),
                                    _mm_loadl_pi(ZERO,(__m64 *)(vh+i-1))));
          _mm_storel_pi ((__m64 *)(dh+i-1), _mm_mul_ps (_mm_loadu_ps(dh+i-1),
                                                  _mm_sub_ps(cast128f(ONE),E)));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i-1] *= 1 - prec_exp(-LOG2-vh[i-1]);
        }
      }
#     elif FP32 && USE_SIMD_INTRINSICS && USE_SLEEF && __SSE2__
      { __m128 MINUS_LOG2 = _mm_set1_ps(-LOG2);
        __m128 ONE = _mm_set1_ps(1.0f);
        i = 3;
        while (i<N_hidden)
        { __m128 E = sleef_expf4 (_mm_sub_ps (MINUS_LOG2,
                                              _mm_loadu_ps(vh+i-3)));
          _mm_storeu_ps (dh+i-3, _mm_mul_ps (_mm_loadu_ps(dh+i-3),
                                             _mm_sub_ps(ONE,E)));
          i += 4;
        }
        i -= 2;
        if (i<N_hidden)
        { __m128 ZERO = _mm_setzero_ps();
          __m128 E = sleef_expf4 (_mm_sub_ps (MINUS_LOG2,
                                         _mm_loadl_pi(ZERO,(__m64 *)(vh+i-1))));
          _mm_storel_pi ((__m64 *)(dh+i-1), _mm_mul_ps (_mm_loadu_ps(dh+i-1),
                                                        _mm_sub_ps(ONE,E)));
          i += 2;
        }
        if (i<=N_hidden)
        { dh[i-1] *= 1 - prec_exp(-LOG2-vh[i-1]);
        }
      }
#     else
      { for (i = 0; i<N_hidden; i++)
        { dh[i] *= 1 - prec_exp(-LOG2-vh[i]);
        }
      }
#     endif
    }

    else if (a->layer_type[l]==Identity_type)
    { /* nothing to do */
    }

    else  /* Normalize layer */
    { int cc = a->N_channels[l];
      int c = cc>0 ? cc : N_hidden/(-cc);   /* number of groups */
      int cn = N_hidden / c;                /* number of units in a group */
      int k;
      for (k = 0; k<c; k++)
      { net_value s = vh[N_hidden+k];
        net_value t = 0;
        if (cc>0)  /* normalize%... */
        { for (i = k; i<N_hidden; i+=c)
          { t += dh[i] * vh[i];
          }
        }
        else  /* normalize/... */
        { int kk = cn*k;
          for (i = 0; i<cn; i++)
          { t += dh[kk+i] * vh[kk+i];
          }
        }
        t /= cn;
        if (cc>0)  /* normalize%... */
        { for (i = k; i<N_hidden; i+=c)
          { dh[i] = s * (dh[i] - vh[i]*t);
          }
        }
        else  /* normalize/... */
        { int kk = cn*k;
          for (i = 0; i<cn; i++)
          { dh[kk+i] = s * (dh[kk+i] - vh[kk+i]*t);
          }
        }
      }
    }

    if (CHECK_NAN)
    { for (i = 0; i<N_hidden; i++)
      { if (isnan(dh[i])) abort();
      }
    }

    /* Add to gradients that depend on the derivatives with respect to
       the inputs of units in this hidden layer. */

    if (a->has_bh[l] && !p->bh[l].one_or_two_point)
    { if (a->bias_config[l])
      { add_grad1_config (g->bh[l], dh, a->bias_config[l]);
      }
      else
      { add_grad1 (g->bh[l], dh, N_hidden);
      }
    }

    if (a->has_ih[l] && !p->ih[l].one_or_two_point)
    { if (a->input_config[l])
      { add_grad2_config (g->ih[l], v->i, a->has_ti ? params.ti : 0, dh,
                          a->input_config[l]);
      }
      else
      { add_grad2 (g->ih[l], v->i, a->has_ti ? params.ti : 0, a->N_inputs,
                   dh, N_hidden,
                   a->any_omitted[l] ? flgs->omit : 0, 1<<(l+1),
                   sparse);
      }
    }

    for (ls = 0; ls<l; ls++)
    { net_config *cf; net_param *gh;
      if (ls==l-1)
      { if (!a->has_hh[ls] || p->hh[ls].one_or_two_point) break;
        cf = a->hidden_config[l];
        gh = g->hh[ls];
      }
      else
      { if (!a->has_nsq[l]) continue;
        int nsqi = pre.nonseq[ls][l];
        if (nsqi<0 || p->nsq[nsqi].one_or_two_point) continue;
        cf = a->nonseq_config[nsqi];
        gh = g->nsq[nsqi];
      }
      if (cf)
      { add_grad2_config (gh, v->h[ls], a->has_th[ls] ? params.th[ls] : 0, 
                          dh, cf);
      }
      else
      { add_grad2 (gh, v->h[ls], a->has_th[ls] ? params.th[ls] : 0,
          a->N_hidden[ls], dh, N_hidden, (unsigned short *)0, 0, 0);
      }
    }

    /* Add contribution from this hidden layer's derivatives to the derivatives
       with respect to inputs, if that will be needed. */

    if (a->has_ti)
    { if (a->has_ih[l])
      { if (a->input_config[l])
        { sum_derivatives_config (dh, d->i, a->N_inputs, 
                                  params.ih[l], a->input_config[l]);
        }
        else
        { sum_derivatives (dh, a->N_hidden[l], d->i, a->N_inputs, params.ih[l],
                           a->any_omitted[l]? flgs->omit : 0, 1<<(l+1));
        }
      }
    }
  }

  /* Add to gradients for input offsets, now that derivatives with respect
     to inputs have been computed. */

  if (a->has_ti && !p->ti.one_or_two_point)
  { add_grad1 (g->ti, d->i, a->N_inputs);
  }
}

#if __CUDACC__

/* FIND GRADIENT, USING BACKPROPAGATED DERIVATIVES, GPU VERSION.
   
   Called by each thread in a block, with threads divided according to
   which group of cases they handle.  The 'thrg' argument (0 to GTH-1)
   identifies the thread within such a group, while 'gsz' gives the
   size of the group - it will be GROUP_SIZE except when there are
   extra cases at the end of the training set, or it may be zero or
   negative for threads that are not used, but must be synchronized.

   The v and d arguments point to the places where hidden unit values
   are found, where derivatives with respect to the outputs are found,
   and where derivatives with respect to hidden and/or input units may
   be stored - v and d are for the first case in the group, with other
   cases following.  If USE_FAST_SHARED_MEM is 1, fast shared memory
   may be used instead to hold values for some or all hidden units,
   and/or derivatives with respect to hidden units.

   The gradient due to the cases in the group is stored in 'g', except
   that components of the gradient for parameters with one or two point
   priors are not stored. */

/* Dispatch functions.  Comparisons with GROUP_SIZE are redundant, but may 
   let the compiler eliminate some comparisons at compile time.  Note that
   no function is called when gsz is less than 1 (unused thread). */

__device__ static void store_grad1
( int th,                 /* Which thread (0 to GTH-1) */
  int gsz,                /* Number of cases in this group (<= GROUP_SIZE) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivs for following cases */
  int n                   /* Number of units */
)
{
  if (GROUP_SIZE>=4 && gsz==4)
  { net_store4_grad1 (th, g, d0, ds, n);
  }
  else if (GROUP_SIZE>=3 && gsz==3)
  { net_store3_grad1 (th, g, d0, ds, n);
  }
  else if (GROUP_SIZE>=2 && gsz==2)
  { net_store2_grad1 (th, g, d0, ds, n);
  }
  else if (gsz==1)
  { net_store1_grad1 (th, g, d0, n);
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();
}

__device__ static void store_grad1_config
( int th,                 /* Which thread (0 to GTH-1) */
  int gsz,                /* Number of cases in this group (<= GROUP_SIZE) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict d0,  /* Derivs with respect to unit values, case 0 */
  int ds,                 /* Stride to get to derivs for following cases */
  net_config const*restrict cf  /* Configuration for biases */
)
{
  if (GROUP_SIZE>=4 && gsz==4)
  { net_store4_grad1_config (th, g, d0, ds, cf);
  }
  else if (GROUP_SIZE>=3 && gsz==3)
  { net_store3_grad1_config (th, g, d0, ds, cf);
  }
  else if (GROUP_SIZE>=2 && gsz==2)
  { net_store2_grad1_config (th, g, d0, ds, cf);
  }
  else if (gsz==1)
  { net_store1_grad1_config (th, g, d0, cf);
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();
}

__device__ static void store_grad2
( int th,                 /* Which thread (0 to GTH-1) */
  int gsz,                /* Number of cases in this group (<= GROUP_SIZE) */
  net_param *restrict g,  /* Array of derivatives to store to */
  net_value const*restrict v0,  /* Source unit values, case 0 */
  int vs,                 /* Stride to get to values for following cases */
  net_param const*restrict off, /* Offsets for source units, 0 if no offsets */
  int nv,                       /* Number of source units */
  net_value const*restrict d0,  /* Derivs with respect to dest units, case 0 */
  int ds,                 /* Stride to get to derivatives for following cases */
  int nd,                 /* Number of destination units */
  unsigned short const*restrict omit, /* Omit flags, null if not present */
  int ob,                 /* Bit to look at in omit flags (mask, not number) */
  int sparse              /* Might source unit values often be zero? */
)
{
  if (GROUP_SIZE>=4 && gsz==4)
  { net_store4_grad2 (th, g, v0, vs, off, nv, d0, ds, nd, omit, ob, sparse);
  }
  else if (GROUP_SIZE>=3 && gsz==3)
  { net_store3_grad2 (th, g, v0, vs, off, nv, d0, ds, nd, omit, ob, sparse);
  }
  else if (GROUP_SIZE>=2 && gsz==2)
  { net_store2_grad2 (th, g, v0, vs, off, nv, d0, ds, nd, omit, ob, sparse);
  }
  else if (gsz==1)
  { net_store1_grad2 (th, g, v0, off, nv, d0, nd, omit, ob, sparse);
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();
}

__device__ static void store_grad2_config
( int th,                 /* Which thread (0 to GTH-1) */
  int gsz,                /* Number of cases in this group (<= GROUP_SIZE) */
  net_param *restrict g,  /* Array of derivatives to add to */
  net_value const*restrict v0,  /* Source unit values, case 0 */
  int vs,                 /* Stride to get to values for following cases */
  net_param const*restrict off, /* Offsets for source units, 0 if no offsets */
  net_value const*restrict d0,  /* Derivs with respect to dest units, case 0 */
  int ds,                 /* Stride to get to derivatives for following cases */
  net_config const*restrict cf  /* Configuration for connections and weights */
)
{
  if (GROUP_SIZE>=4 && gsz==4)
  { net_store4_grad2_config (th, g, v0, vs, off, d0, ds, cf);
  }
  else if (GROUP_SIZE>=3 && gsz==3)
  { net_store3_grad2_config (th, g, v0, vs, off, d0, ds, cf);
  }
  else if (GROUP_SIZE>=2 && gsz==2)
  { net_store2_grad2_config (th, g, v0, vs, off, d0, ds, cf);
  }
  else if (gsz==1)
  { net_store1_grad2_config (th, g, v0, off, d0, cf);
  }

  if (SYNC_AFTER && GTH>=32) __syncwarp();
}

#define A const_arch
#define PRE const_pre
#define PRIORS const_priors
#define FLGS const_flgs
#define W const_params
#define WT const_params_trans

__device__ __forceinline__ static void net_back_grad_gpu
( int thrg,		/* Which thread, from 0 to GTH-1 */
  int gsz,		/* Size of gradient group (maybe < GROUP_SIZE at end) */
  net_params *restrict g,   /* Where to store gradient with respect to params */
  net_values const*restrict v, /* Values for units in network, 1st train case */
  net_values *restrict d,      /* Place for derivatives, first training case */
  int sparse,           /* Might source unit values often be zero? */
  unsigned syncmask     /* Mask of active threads for backprop */
)
{
  int l, ld, ls, i;

  /* Compute parts of gradients that don't depend on computing derivatives
     with respect to hidden or input unit values - only on inputs and hidden
     unit values, and on derivatives with respect to outputs, */

  if (A.has_bo && !PRIORS.bo.one_or_two_point)
  { if (A.bias_config[A.N_layers])
    { store_grad1_config (thrg, gsz, g->bo, d[0].o, d[1].o-d[0].o,
                          A.bias_config[A.N_layers]);
    }
    else
    { store_grad1 (thrg, gsz, g->bo, d[0].o, d[1].o-d[0].o, A.N_outputs);
    }
  }

  if (A.has_io && !PRIORS.io.one_or_two_point)
  { if (A.input_config[A.N_layers])
    { store_grad2_config (thrg, gsz, g->io, 
                          v[0].i, v[1].i-v[0].i,
                          A.has_ti ? W.ti : 0, 
                          d[0].o, d[1].o-d[0].o,
                          A.input_config[A.N_layers]);
    }
    else
    { store_grad2 (thrg, gsz, g->io, 
                   v[0].i, v[1].i-v[0].i,
                   A.has_ti ? W.ti : 0, A.N_inputs,
                   d[0].o, d[1].o-d[0].o, A.N_outputs,
                   A.any_omitted[A.N_layers] ? FLGS.omit : 0, 1,
                   sparse);
    }
  }

  for (l = 0; l<A.N_layers; l++)
  {
    if (A.has_ho[l] && !PRIORS.ho[l].one_or_two_point)
    { 
      net_value *restrict u0 = fw_hidden_loc_grad(&PRE,v,l,0);
      int us = fw_hidden_stride(&PRE,l);

      int k = 2*A.N_layers-1-l;
      if (A.hidden_config[k])
      { store_grad2_config (thrg, gsz, g->ho[l], u0, us, 
                            A.has_th[l] ? W.th[l] : 0,
                            d[0].o, A.N_outputs,
                            A.hidden_config[k]);
      }
      else
      { store_grad2 (thrg, gsz, g->ho[l], u0, us,
                     A.has_th[l] ? W.th[l] : 0, A.N_hidden[l], 
                     d[0].o, A.N_outputs, A.N_outputs, 
                     (unsigned short *) 0, 0, 0);
      }
    }
  }

  /* Find case handled by this thread for backpropagation, and index, thrb, of
     this thread within that case (0 to NTH-1, except -1 for unused thread). */

  const net_values *dth, *vth;
  int thrb;
  if (thrg>=gsz*NTH)
  { thrb = -1;
  }
  else
  { thrb = thrg & (NTH-1);
    dth = d + thrg / NTH;
    vth = v + thrg / NTH;
  }

  /* Start computation of derivatives with respect to input values, if
     they will be needed, with possible contribution from input-output
     connections. */

  if (A.has_ti && thrb>=0)
  { 
    for (i = thrb; i<A.N_inputs; i+=NTH)
    { dth->i[i] = 0;
    }

    if (SYNC_AFTER && A.N_inputs % NTH != 0) __syncwarp(syncmask);

    if (A.has_io)
    { if (A.input_config[A.N_layers])
      { sum_derivatives_config_gpu 
         (thrb, dth->o, dth->i, W.io, A.input_config[A.N_layers], syncmask);
      }
      else
      { sum_derivatives_gpu 
         (thrb, dth->o, A.N_outputs, dth->i, A.N_inputs, W.io,
          A.any_omitted[A.N_layers] ? FLGS.omit : 0, 1,
          syncmask);
      }
    }
  }

  /* Go backwards through hidden layers, computing derivatives with respect to
     hidden unit values, and then adding to gradients that depend on these. */

  for (l = A.N_layers-1; l>=0; l--)
  {
    int N_hidden = A.N_hidden[l];

    /* Place to store derivatives computed for this hidden layer. */

    net_value *restrict dh;

    if (thrb<0) goto sync_layer;

    dh = bw_hidden_loc(&PRE,dth,l);

    /* Find derivatives with respect to values of units in this hidden layer. */

    for (i = thrb; i<N_hidden; i+=NTH)
    { dh[i] = 0;
    }

    if (SYNC_AFTER && N_hidden % NTH != 0) __syncwarp(syncmask);

    if (A.has_ho[l])
    { int k = 2*A.N_layers-1-l;
      if (A.hidden_config[k])
      { sum_derivatives_config_gpu 
         (thrb, dth->o, dh, W.ho[l], A.hidden_config[k], syncmask);
      }
      else if (TRANS_WEIGHTS(N_hidden,A.N_outputs))
      { add_connections_gpu (thrb, dh, N_hidden, dth->o, A.N_outputs, WT.ho[l],
                             (net_param const*) 0, (unsigned short *) 0, 0, 0,
                             syncmask);
      }
      else
      { sum_derivatives_gpu 
          (thrb, dth->o, A.N_outputs, dh, N_hidden,
           W.ho[l], (unsigned short *) 0, 0, syncmask);
      }
    }

    for (ld = l+1; ld<A.N_layers; ld++)
    { net_config *cf; net_param *wh; net_param *wht;
      if (ld==l+1)
      { if (!A.has_hh[l]) continue;
        cf = A.hidden_config[ld];
        wh = W.hh[l];
#       if USE_TRANSPOSED_WEIGHTS
        { if (TRANS_WEIGHTS(N_hidden,a->N_hidden[ld]))
          { wht = WT.hh[l];
          }
        }
#       endif
      }
      else
      { int nsqi = PRE.nonseq[l][ld];
        if (nsqi<0) continue;
        cf = A.nonseq_config[nsqi];
        wh = W.nsq[nsqi];
#       if USE_TRANSPOSED_WEIGHTS
        { if (TRANS_WEIGHTS(N_hidden,a->N_hidden[ld]))
          { wht = WT.nsq[nsqi];
          }
        }
#       endif
      }
      net_value *restrict dhd = bw_hidden_loc(&PRE,dth,ld);
      if (cf)
      { sum_derivatives_config_gpu (thrb, dhd, dh, wh, cf, syncmask);
      }
      else if (TRANS_WEIGHTS(N_hidden,A.N_hidden[ld]))
      { add_connections_gpu (thrb, dh, N_hidden, dhd, A.N_hidden[ld], wht,
                             (net_param const*) 0, (unsigned short *) 0, 0, 0,
                             syncmask);
      }
      else
      { sum_derivatives_gpu (thrb, dhd, A.N_hidden[ld], dh, N_hidden,
                             wh, (unsigned short *) 0, 0, syncmask);
      }
    }

  sync_layer:

    /* Add to gradient with respect to hidden offsets, based on derivatives
       with respect to hidden unit values (before these are converted to
       derivatives with respect to the summed input, prior to the activation
       function). */

    if (A.has_th[l] && !PRIORS.th[l].one_or_two_point)
    { 
      __syncthreads();

      net_value *restrict c0 = bw_hidden_loc_grad(&PRE,d,l,0);
      int cs = bw_hidden_stride(&PRE,l);

      store_grad1 (thrg, gsz, g->th[l], c0, cs, N_hidden);
    }

    /* Pass backwards through activation function to get derivatives with 
       respect to the summed inputs of units in this hidden layer. */

    net_value const*restrict vh = thrb<0 ? 0 : fw_hidden_loc(&PRE,vth,l);

    if (A.layer_type[l]==Normalize_type)
    { 
      if (NTH>1) __syncthreads();

      if (thrb>=0)
      { 
        int cc = A.N_channels[l];
        int c = cc>0 ? cc : N_hidden/(-cc);   /* number of groups */
        int cn = N_hidden / c;                /* number of units in a group */
        int k;

        for (k = thrb; k<c; k+=NTH)
        { net_value s = vth->h[l][N_hidden+k];
          net_value t = 0;
          if (cc>0)  /* normalize%... */
          { for (i = k; i<N_hidden; i+=c)
            { t += dh[i] * vh[i];
            }
          }
          else  /* normalize/... */
          { int kk = cn*k;
            for (i = 0; i<cn; i++)
            { t += dh[kk+i] * vh[kk+i];
            }
          }
          t /= cn;
          if (cc>0)  /* normalize%... */
          { for (i = k; i<N_hidden; i+=c)
            { dh[i] = s * (dh[i] - vh[i]*t);
            }
          }
          else  /* normalize/... */
          { int kk = cn*k;
            for (i = 0; i<cn; i++)
            { dh[kk+i] = s * (dh[kk+i] - vh[kk+i]*t);
            }
          }
        }
      }
    }
    else if (thrb<0)
    { /* nothing, surplus thread */
    }
    else if (A.layer_type[l]==Tanh_type)
    { for (i = thrb; i<N_hidden; i+=NTH)
      { dh[i] *= (1 - vh[i]*vh[i]);
      }
    }
    else if (A.layer_type[l]==Softplus_type)
    { for (i = thrb; i<N_hidden; i+=NTH)
      { dh[i] *= 1 - prec_exp(-vh[i]);
      }
    }
    else if (A.layer_type[l]==Softplus0_type)
    { for (i = thrb; i<N_hidden; i+=NTH)
      { dh[i] *= 1 - prec_exp(-LOG2-vh[i]);
      }
    }
    else if (A.layer_type[l]==Identity_type)
    { /* nothing to do */
    }
    else
    { abort();
    }

    if (CHECK_NAN)
    { if (NTH>1) __syncthreads();
      if (thrb>=0)
      { for (i = thrb; i<N_hidden; i+=NTH)
        { if (isnan(dh[i])) abort();
        }
      }
    }

    __syncthreads();

    /* Add contribution from this hidden layer's derivatives to the derivatives
       with respect to inputs, if they will be needed. */

    if (A.has_ti && thrb>=0)
    { if (A.has_ih[l])
      { if (A.input_config[l])
        { sum_derivatives_config_gpu (thrb, dh, dth->i, W.ih[l], 
                                      A.input_config[l], syncmask);
        }
        else
        { sum_derivatives_gpu (thrb, dh, A.N_hidden[l], dth->i, A.N_inputs, 
            W.ih[l], A.any_omitted[l] ? FLGS.omit : 0, 1<<(l+1), syncmask);
        }
      }
    }

    /* Add to gradients that depend on the derivatives of energy with respect 
       to the inputs of units in this hidden layer. */

    net_value *restrict c0 = bw_hidden_loc_grad(&PRE,d,l,0);
    int cs = bw_hidden_stride(&PRE,l);

    if (A.has_bh[l] && !PRIORS.bh[l].one_or_two_point)
    { if (A.bias_config[l])
      { store_grad1_config (thrg, gsz, g->bh[l], c0, cs, A.bias_config[l]);
      }
      else
      { store_grad1 (thrg, gsz, g->bh[l], c0, cs, N_hidden);
      }
    }

    if (A.has_ih[l] && !PRIORS.ih[l].one_or_two_point)
    { if (A.input_config[l])
      { store_grad2_config (thrg, gsz, g->ih[l], 
                            v[0].i, A.N_inputs,
                            A.has_ti ? W.ti : 0, 
                            c0, cs, A.input_config[l]);
      }
      else
      { store_grad2 (thrg, gsz, g->ih[l], 
                     v[0].i, v[1].i-v[0].i,
                     A.has_ti ? W.ti : 0, A.N_inputs,
                     c0, cs, N_hidden,
                     A.any_omitted[l] ? FLGS.omit : 0, 1<<(l+1), sparse);
      }
    }

    for (ls = 0; ls<l; ls++)
    { net_config *cf; net_param *gh;
      if (ls==l-1)
      { if (!A.has_hh[ls] || PRIORS.hh[ls].one_or_two_point) break;
        cf = A.hidden_config[l];
        gh = g->hh[ls];
      }
      else
      { if (!A.has_nsq[l]) continue;
        int nsqi = PRE.nonseq[ls][l];
        if (nsqi<0 || PRIORS.nsq[nsqi].one_or_two_point) continue;
        cf = A.nonseq_config[nsqi];
        gh = g->nsq[nsqi];
      }
      net_value *restrict u0 = fw_hidden_loc_grad(&PRE,v,ls,0);
      int us = fw_hidden_stride(&PRE,ls);
      if (cf)
      { store_grad2_config (thrg, gsz, gh, u0, us, A.has_th[ls] ? W.th[ls] : 0,
                            c0, cs, cf);
      }
      else
      { store_grad2 (thrg, gsz, gh, u0, us,
                     A.has_th[ls] ? W.th[ls] : 0, A.N_hidden[ls], 
                     c0, cs, N_hidden, (unsigned short *)0, 0, 0);
      }
    }
  }

  /* Add to gradients for input offsets, now that derivatives with respect
     to inputs have been computed. */

  if (A.has_ti && !PRIORS.ti.one_or_two_point)
  { __syncthreads();
    store_grad1 (thrg, gsz, g->ti, d[0].i, d[1].i-d[0].i, A.N_inputs);
  }

  if (CHECK_NAN)
  { if (blockIdx.x==0)
    { __syncthreads();
      if (thrg==0)
      { int k;
        for (k = 0; k<g->total_params; k++)
        { if (isnan(g->param_block[k]))
          { if (k==265646)
            { printf("NaN in gradient component %d, block %d\n",k,blockIdx.x);
            }
          }
        }
      }
    }
  }
}

#undef A
#undef PRE
#undef FLGS
#undef W

#endif
