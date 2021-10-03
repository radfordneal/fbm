/* SLEEF-USE.H - Possibly include header files for SLEEF SIMD functions. */

/* Copyright (c) 2021 by Radford M. Neal 
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


/* DEFINE SHORT FORMS FOR SLEEF FUNCTIONS.  For sin, cos, and tanh, one could
   decide to use either the u10 or u35 forms (with different precisions). */

#if USE_SLEEF
# include <stdint.h>
# define __SLEEF_REMPITAB__
# if USE_SIMD_INTRINSICS && __AVX2__ && USE_FMA && __FMA__
#   include "../sleef-include/sleefinline_avx2128.h"
#   include "../sleef-include/sleefinline_avx2.h"
#   define sleef_expf8 Sleef_expf8_u10avx2
#   define sleef_logf8 Sleef_logf8_u10avx2
#   define sleef_sinf8 Sleef_sinf8_u35avx2
#   define sleef_cosf8 Sleef_cosf8_u35avx2
#   define sleef_tanhf8 Sleef_tanhf8_u35avx2
#   define sleef_fabsf8 Sleef_fabsf8_avx2
#   define sleef_copysignf8 Sleef_copysignf8_avx2
#   define sleef_expd4 Sleef_expd4_u10avx2
#   define sleef_logd4 Sleef_logd4_u10avx2
#   define sleef_sind4 Sleef_sind4_u35avx2
#   define sleef_cosd4 Sleef_cosd4_u35avx2
#   define sleef_tanhd4 Sleef_tanhd4_u35avx2
#   define sleef_fabsd4 Sleef_fabsd4_avx2
#   define sleef_copysignd4 Sleef_copysignd4_avx2
#   define sleef_expf4 Sleef_expf4_u10avx2128
#   define sleef_logf4 Sleef_logf4_u10avx2128
#   define sleef_sinf4 Sleef_sinf4_u35avx2128
#   define sleef_cosf4 Sleef_cosf4_u35avx2128
#   define sleef_tanhf4 Sleef_tanhf4_u35avx2128
#   define sleef_fabsf4 Sleef_fabsf4_avx2128
#   define sleef_copysignf4 Sleef_copysignf4_avx2128
#   define sleef_expd2 Sleef_expd2_u10avx2128
#   define sleef_logd2 Sleef_logd2_u10avx2128
#   define sleef_sind2 Sleef_sind2_u35avx2128
#   define sleef_cosd2 Sleef_cosd2_u35avx2128
#   define sleef_tanhd2 Sleef_tanhd2_u35avx2128
#   define sleef_fabsd2 Sleef_fabsd2_avx2128
#   define sleef_copysignd2 Sleef_copysignd2_avx2128
# elif USE_SIMD_INTRINSICS && __AVX__
#   include "../sleef-include/sleefinline_sse4.h"
#   include "../sleef-include/sleefinline_avx.h"
#   define sleef_expf8 Sleef_expf8_u10avx
#   define sleef_logf8 Sleef_logf8_u10avx
#   define sleef_sinf8 Sleef_sinf8_u35avx
#   define sleef_cosf8 Sleef_cosf8_u35avx
#   define sleef_tanhf8 Sleef_tanhf8_u35avx
#   define sleef_fabsf8 Sleef_fabsf8_avx
#   define sleef_copysignf8 Sleef_copysignf8_avx
#   define sleef_expd4 Sleef_expd4_u10avx
#   define sleef_logd4 Sleef_logd4_u10avx
#   define sleef_sind4 Sleef_sind4_u35avx
#   define sleef_cosd4 Sleef_cosd4_u35avx
#   define sleef_tanhd4 Sleef_tanhd4_u35avx
#   define sleef_fabsd4 Sleef_fabsd4_avx
#   define sleef_copysignd4 Sleef_copysignd4_avx
#   define sleef_expf4 Sleef_expf4_u10sse4
#   define sleef_logf4 Sleef_logf4_u10sse4
#   define sleef_sinf4 Sleef_sinf4_u35sse4
#   define sleef_cosf4 Sleef_cosf4_u35sse4
#   define sleef_tanhf4 Sleef_tanhf4_u35sse4
#   define sleef_fabsf4 Sleef_fabsf4_sse4
#   define sleef_copysignf4 Sleef_copysignf4_sse4
#   define sleef_expd2 Sleef_expd2_u10sse4
#   define sleef_logd2 Sleef_logd2_u10sse4
#   define sleef_sind2 Sleef_sind2_u35sse4
#   define sleef_cosd2 Sleef_cosd2_u35sse4
#   define sleef_tanhd2 Sleef_tanhd2_u35sse4
#   define sleef_fabsd2 Sleef_fabsd2_sse4
#   define sleef_copysignd2 Sleef_copysignd2_sse4
# elif USE_SIMD_INTRINSICS && __SSE4_2__
#   include "../sleef-include/sleefinline_sse4.h"
#   define sleef_expf4 Sleef_expf4_u10sse4
#   define sleef_logf4 Sleef_logf4_u10sse4
#   define sleef_sinf4 Sleef_sinf4_u35sse4
#   define sleef_cosf4 Sleef_cosf4_u35sse4
#   define sleef_tanhf4 Sleef_tanhf4_u35sse4
#   define sleef_fabsf4 Sleef_fabsf4_sse4
#   define sleef_copysignf4 Sleef_copysignf4_sse4
#   define sleef_expd2 Sleef_expd2_u10sse4
#   define sleef_logd2 Sleef_logd2_u10sse4
#   define sleef_sind2 Sleef_sind2_u35sse4
#   define sleef_cosd2 Sleef_cosd2_u35sse4
#   define sleef_tanhd2 Sleef_tanhd2_u35sse4
#   define sleef_fabsd2 Sleef_fabsd2_sse4
#   define sleef_copysignd2 Sleef_copysignd2_sse4
# elif USE_SIMD_INTRINSICS && __SSE2__
#   include "../sleef-include/sleefinline_sse2.h"
#   define sleef_expf4 Sleef_expf4_u10sse2
#   define sleef_logf4 Sleef_logf4_u10sse2
#   define sleef_sinf4 Sleef_sinf4_u35sse2
#   define sleef_cosf4 Sleef_cosf4_u35sse2
#   define sleef_tanhf4 Sleef_tanhf4_u35sse2
#   define sleef_fabsf4 Sleef_fabsf4_sse2
#   define sleef_copysignf4 Sleef_copysignf4_sse2
#   define sleef_expd2 Sleef_expd2_u10sse2
#   define sleef_logd2 Sleef_logd2_u10sse2
#   define sleef_sind2 Sleef_sind2_u35sse2
#   define sleef_cosd2 Sleef_cosd2_u35sse2
#   define sleef_tanhd2 Sleef_tanhd2_u35sse2
#   define sleef_fabsd2 Sleef_fabsd2_sse2
#   define sleef_copysignd2 Sleef_copysignd2_sse2
# endif
#endif
