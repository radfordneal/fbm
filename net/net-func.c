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

#include "intrinsics-use.h"
#include "sleef-use.h"


/* This module calculates the values of the output units in a network, given 
   values for the input units.  The values of hidden units are calculated
   along the way.  There are facilities for starting the calculation on the 
   assumption the values are already known up to some layer, as would be 
   the case if the weights into earlier layers have not changed since the
   last calculation. 
*/

#define sqrt_2 1.4142135623730950488

static void bias_values (net_value *restrict, int, net_param *restrict);

static void bias_values_config (net_value *restrict, int, net_param *restrict,
                                net_config *restrict);

static void add_connections (net_value *restrict, int, net_value *restrict, int,
                             net_param *restrict, net_param *restrict,
                             unsigned short *restrict, int);

static void add_connections_config (net_value *restrict, net_value *restrict,
                                    net_param *restrict, net_param *restrict,
                                    net_config *restrict);


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
          w->ih[l], a->has_ti ? w->ti : 0, flgs ? flgs->omit : 0, 1<<(l+1));
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
#     if USE_SLEEF && __AVX__ && __FMA__
      { j = 3;
        while (j<N_hidden)
        { _mm256_storeu_pd (vh+j-3,  
                            Sleef_tanhd4_u35avx2 (_mm256_loadu_pd(sh+j-3)));
          j += 4;
        }
        j -= 2;
        if (j<N_hidden)
        { _mm_storeu_pd (vh+j-1,  
                         Sleef_tanhd2_u35avx2128 (_mm_loadu_pd(sh+j-1)));
          j += 2;
        }
        if (j<=N_hidden)
        { vh[j-1] = Sleef_tanhd1_u35purecfma (sh[j-1]);
        }
      }
#     elif USE_SLEEF && __AVX__
      { j = 3;
        while (j<N_hidden)
        { _mm256_storeu_pd (vh+j-3,  
                            Sleef_tanhd4_u35avx (_mm256_loadu_pd(sh+j-3)));
          j += 4;
        }
        j -= 2;
        if (j<N_hidden)
        { _mm_storeu_pd (vh+j-1,  
                         Sleef_tanhd2_u35sse4 (_mm_loadu_pd(sh+j-1)));
          j += 2;
        }
        if (j<=N_hidden)
        { vh[j-1] = Sleef_tanhd1_u35purec (sh[j-1]);
        }
      }
#     elif USE_SLEEF
      { for (j = 0; j<N_hidden; j++)
        { vh[j] = Sleef_tanhd1_u35purec (sh[j]);
        }
      }
#     else
      { for (j = 0; j<N_hidden; j++)
        { vh[j] = tanh (sh[j]);
        }
      }
#     endif
    }
    else if (flgs->layer_type[l]==Sin_type)
    { for (j = 0; j<N_hidden; j++)
      { vh[j] = sqrt_2*sin(sh[j]*sqrt_2);
      }
    }
    else if (flgs->layer_type[l]==Softplus_type)
    { for (j = 0; j<N_hidden; j++)
      { vh[j] = sh[j]>0 ? sh[j] + log(1+exp(-sh[j])) /* avoid overflow */
                        : log(1+exp(sh[j]));
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
                       w->io, a->has_ti ? w->ti : 0, flgs ? flgs->omit : 0, 1);
    }
  }

  for (l = 0; l<a->N_layers; l++)
  { if (a->has_ho[l])
    { if (l==a->N_layers-1 && a->hidden_config[l+1])  /* only last for now... */
      { add_connections_config (v->o, v->h[l], w->ho[l], 
                         a->has_th[l] ? w->th[l] : 0, a->hidden_config[l+1]);
      }
      else
      { add_connections (v->o, a->N_outputs, v->h[l], a->N_hidden[l], w->ho[l],
                         a->has_th[l] ? w->th[l] : 0, (unsigned short *) 0, 0);
      }
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
  int j;
  for (j = 0; j<n; j++) v[j] = b[j];
}


/* SET UNIT VALUES TO BIASES WHEN THERE IS A CONFIGURATON.  At present,
   just goes through the original list of connections in the configuration,
   without trying to optimize. */

static void bias_values_config
( net_value *restrict v,	/* Array of unit values to set */
  int n,			/* Number of units */
  net_param *restrict b,	/* Biases */
  net_config *restrict cf	/* Configuration for biases */
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
  { __m256d TV, TV2; \
    i = 0; \
    for (;;) \
    { for (;;) \
      { if (i==ns) goto done; \
        TV = _mm256_broadcast_sd (v+i); \
        if (_mm_ucomineq_sd (_mm256_castpd256_pd128(TV), _mm_setzero_pd())) \
        { break; \
        } \
        i += 1; \
        w += nd; \
      } \
      net_param *w2 = w+nd; \
      i += 1; \
      for (;;) \
      { if (i==ns) goto one_more; \
        TV2 = _mm256_broadcast_sd (v+i); \
        if (_mm_ucomineq_sd (_mm256_castpd256_pd128(TV2), _mm_setzero_pd())) \
        { break; \
        } \
        i += 1; \
        w2 += nd; \
      } \
      j = 3; \
      while (j<nd) \
      { __m256d S = _mm256_loadu_pd(s+j-3); \
        S = _mm256_fmadd_pd (TV, _mm256_loadu_pd(w+j-3), S); \
        S = _mm256_fmadd_pd (TV2, _mm256_loadu_pd(w2+j-3), S); \
        _mm256_storeu_pd (s+j-3, S); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { __m128d S = _mm_loadu_pd(s+j-1); \
        S = _mm_fmadd_pd (_mm256_castpd256_pd128(TV), _mm_loadu_pd(w+j-1), S); \
        S = _mm_fmadd_pd (_mm256_castpd256_pd128(TV2), _mm_loadu_pd(w2+j-1), \
                          S); \
        _mm_storeu_pd (s+j-1, S); \
        j += 2; \
      } \
      if (j<=nd) \
      { __m128d S = _mm_load_sd(s+j-1); \
        S = _mm_fmadd_sd (_mm256_castpd256_pd128(TV), _mm_load_sd(w+j-1), S); \
        S = _mm_fmadd_sd (_mm256_castpd256_pd128(TV2), _mm_load_sd(w2+j-1), \
                          S); \
        _mm_store_sd (s+j-1, S); \
      } \
      i += 1; \
      w = w2+nd; \
    } \
  one_more: \
    j = 3; \
    while (j<nd) \
    { __m256d S = _mm256_loadu_pd(s+j-3); \
      S = _mm256_fmadd_pd (TV, _mm256_loadu_pd(w+j-3), S); \
      _mm256_storeu_pd (s+j-3, S); \
      j += 4; \
    } \
    j -= 2; \
    if (j<nd) \
    { __m128d S = _mm_loadu_pd(s+j-1); \
      S = _mm_fmadd_pd (_mm256_castpd256_pd128(TV), _mm_loadu_pd(w+j-1), S); \
      _mm_storeu_pd (s+j-1, S); \
      j += 2; \
    } \
    if (j<=nd) \
    { __m128d S = _mm_load_sd(s+j-1); \
      S = _mm_fmadd_sd (_mm256_castpd256_pd128(TV), _mm_load_sd(w+j-1), S); \
      _mm_store_sd (s+j-1, S); \
    } \
  done: ; \
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
  { __m256d TV, TV2; \
    i = 0; \
    for (;;) \
    { for (;;) \
      { if (i==ns) goto done; \
        TV = _mm256_broadcast_sd (v+i); \
        if (_mm_ucomineq_sd (_mm256_castpd256_pd128(TV), _mm_setzero_pd())) \
        { break; \
        } \
        i += 1; \
        w += nd; \
      } \
      net_param *w2 = w+nd; \
      i += 1; \
      for (;;) \
      { if (i==ns) goto one_more; \
        TV2 = _mm256_broadcast_sd (v+i); \
        if (_mm_ucomineq_sd (_mm256_castpd256_pd128(TV2), _mm_setzero_pd())) \
        { break; \
        } \
        i += 1; \
        w2 += nd; \
      } \
      j = 3; \
      while (j<nd) \
      { __m256d S = _mm256_loadu_pd(s+j-3); \
        S = _mm256_add_pd (S, _mm256_mul_pd (TV, _mm256_loadu_pd(w+j-3))); \
        S = _mm256_add_pd (S, _mm256_mul_pd (TV2, _mm256_loadu_pd(w2+j-3))); \
        _mm256_storeu_pd (s+j-3, S); \
        j += 4; \
      } \
      j -= 2; \
      if (j<nd) \
      { __m128d S = _mm_loadu_pd(s+j-1); \
        S = _mm_add_pd (S, _mm_mul_pd (_mm256_castpd256_pd128(TV), \
                                       _mm_loadu_pd(w+j-1))); \
        S = _mm_add_pd (S, _mm_mul_pd (_mm256_castpd256_pd128(TV2), \
                                       _mm_loadu_pd(w2+j-1))); \
        _mm_storeu_pd (s+j-1, S); \
        j += 2; \
      } \
      if (j<=nd) \
      { __m128d S = _mm_load_sd(s+j-1); \
        S = _mm_add_sd (S, _mm_mul_sd (_mm256_castpd256_pd128(TV), \
                                       _mm_load_sd(w+j-1))); \
        S = _mm_add_sd (S, _mm_mul_sd (_mm256_castpd256_pd128(TV2), \
                                       _mm_load_sd(w2+j-1))); \
        _mm_store_sd (s+j-1, S); \
      } \
      i += 1; \
      w = w2+nd; \
    } \
  one_more: \
    j = 3; \
    while (j<nd) \
    { __m256d S = _mm256_loadu_pd(s+j-3); \
      S = _mm256_add_pd (S, _mm256_mul_pd (TV, _mm256_loadu_pd(w+j-3))); \
      _mm256_storeu_pd (s+j-3, S); \
      j += 4; \
    } \
    j -= 2; \
    if (j<nd) \
    { __m128d S = _mm_loadu_pd(s+j-1); \
      S = _mm_add_pd (S, _mm_mul_pd (_mm256_castpd256_pd128(TV), \
                                     _mm_loadu_pd(w+j-1))); \
      _mm_storeu_pd (s+j-1, S); \
      j += 2; \
    } \
    if (j<=nd) \
    { __m128d S = _mm_load_sd(s+j-1); \
      S = _mm_add_sd (S, _mm_mul_sd (_mm256_castpd256_pd128(TV), \
                                     _mm_load_sd(w+j-1))); \
      _mm_store_sd (s+j-1, S); \
    } \
  done: ; \
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
  unsigned short *restrict omit,  /* Omit flags, null if not present */
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

static void add_connections_config
( net_value *restrict s,  /* Summed input for destination units to add to */
  net_value *restrict v,  /* Values for source units */
  net_param *restrict w,  /* Connection weights */
  net_param *restrict off,/* Offsets to add to source unit values */
  net_config *restrict cf /* Configuration for connections and weights */
)
{
  net_connection *cn;
  int i, j, k, c;

  if (CONFIG_QUAD_S_4D_4W)
  { cn = cf->quad_s_4d_4w;
#   if USE_SIMD_INTRINSICS && __AVX__ && USE_FMA && __FMA__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256d VOI = _mm256_set1_pd (v[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (s+j, _mm256_fmadd_pd (VOI, _mm256_loadu_pd(w+k),
                                                       _mm256_loadu_pd(s+j)));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256d VOI = _mm256_set1_pd (v[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (s+j, _mm256_fmadd_pd (VOI, _mm256_loadu_pd(w+k),
                                                       _mm256_loadu_pd(s+j)));
        }
      }
    }
#   elif USE_SIMD_INTRINSICS && __AVX__
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256d VOI = _mm256_set1_pd (v[cn[c].s] + off[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (s+j, _mm256_add_pd (_mm256_loadu_pd(s+j),
                                   _mm256_mul_pd (VOI, _mm256_loadu_pd(w+k))));
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { __m256d VOI = _mm256_set1_pd (v[cn[c].s]);
          j = cn[c].d;
          _mm256_storeu_pd (s+j, _mm256_add_pd (_mm256_loadu_pd(s+j),
                                   _mm256_mul_pd (VOI, _mm256_loadu_pd(w+k))));
        }
      }
    }
#   else
    { if (off)
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { double voi = v[cn[c].s] + off[cn[c].s];
          j = cn[c].d;
          s[j+0] += voi * w[k+0];
          s[j+1] += voi * w[k+1];
          s[j+2] += voi * w[k+2];
          s[j+3] += voi * w[k+3];
        }
      }
      else
      { for (c = 0; (k = cn[c].w) >= 0; c++)
        { double vi = v[cn[c].s];
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
      { double voi = v[cn[c].s] + off[cn[c].s];
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
      { double vi = v[cn[c].s];
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
        double sj = s[j];
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
#     if USE_SIMD_INTRINSICS && __AVX2__ && 0  /* disabled since it's slower */
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
          __m128d S = _mm_add_pd (_mm256_castpd256_pd128(P),
                                  _mm256_extractf128_pd(P,1));
          _mm_store_sd (s+j, _mm_add_sd (SJ, _mm_hadd_pd(S,S)));
        }
      }
#     else
      { for (c = 0; (k = cn[c].w) >= 0; c+=4)
        { j = cn[c].d;
          double sj = s[j];
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
