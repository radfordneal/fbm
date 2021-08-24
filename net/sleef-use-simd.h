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

#if USE_SLEEF
# include <stdint.h>
# if __AVX__ && __FMA__
#   include "../sleef-include/sleefinline_avx2128.h"
#   include "../sleef-include/sleefinline_avx2.h"
# elif __AVX__
#   include "../sleef-include/sleefinline_sse4.h"
#   include "../sleef-include/sleefinline_avx.h"
# endif
#endif
