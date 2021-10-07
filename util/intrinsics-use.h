/* INTRINSICS-USE.H - Possibly include header files for Intel intrinsics. */

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

#if (USE_SIMD_INTRINSICS || USE_SLEEF) && __SSE2__
# include  <immintrin.h>
#endif

#define cast128d(x) (_mm256_castpd256_pd128(x))
#define cast128f(x) (_mm256_castps256_ps128(x))

#if USE_SIMD_INTRINSICS && USE_FMA && __AVX2__ && __FMA__
# define FMA256_pd(a,b,c) _mm256_fmadd_pd(a,b,c)
# define FMA256_ps(a,b,c) _mm256_fmadd_ps(a,b,c)
# define FMA_pd(a,b,c)    _mm_fmadd_pd(a,b,c)
# define FMA_ps(a,b,c)    _mm_fmadd_ps(a,b,c)
# define FMA_sd(a,b,c)    _mm_fmadd_sd(a,b,c)
# define FMA_ss(a,b,c)    _mm_fmadd_ss(a,b,c)
#elif USE_SIMD_INTRINSICS && __AVX__
# define FMA256_pd(a,b,c) _mm256_add_pd(_mm256_mul_pd(a,b),c)
# define FMA256_ps(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
# define FMA_pd(a,b,c)    _mm_add_pd(_mm_mul_pd(a,b),c)
# define FMA_ps(a,b,c)    _mm_add_ps(_mm_mul_ps(a,b),c)
# define FMA_sd(a,b,c)    _mm_add_sd(_mm_mul_sd(a,b),c)
# define FMA_ss(a,b,c)    _mm_add_ss(_mm_mul_ss(a,b),c)
#elif USE_SIMD_INTRINSICS && __SSE2__
# define FMA_pd(a,b,c)    _mm_add_pd(_mm_mul_pd(a,b),c)
# define FMA_ps(a,b,c)    _mm_add_ps(_mm_mul_ps(a,b),c)
# define FMA_sd(a,b,c)    _mm_add_sd(_mm_mul_sd(a,b),c)
# define FMA_ss(a,b,c)    _mm_add_ss(_mm_mul_ss(a,b),c)
#endif
