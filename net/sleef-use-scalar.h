/* SLEEF-USE.H - Possibly include header files for SLEEF scalar functions. */

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


/* DEFINE FAST VERSIONS OF EXP AND LOG.  Just the ordinary versions if SLEEF 
   isn't being used.  Note that since the fast versions may be inlined, they
   should be avoided in code that isn't time critical. 

   May be tweaked according to how good the standard library functions are. */

#if USE_SLEEF
# include <stdint.h>
# define __SLEEF_REMPITAB__
# if USE_SIMD_INTRINSICS && __AVX2__ && USE_FMA && __FMA__
#   include "../sleef-include/sleefinline_purecfma_scalar.h"
#   define fast_exp exp // Sleef_expd1_u10purecfma is slower on ubuntu 20.04
#   define fast_log log // Sleef_logd1_u10purecfma is slower on ubuntu 20.04
#   define fast_tanh Sleef_tanhd1_u10purecfma
# else
#   include "../sleef-include/sleefinline_purec_scalar.h"
#   define fast_exp Sleef_expd1_u10purec
#   define fast_log Sleef_logd1_u10purec
#   define fast_tanh Sleef_tanhd1_u10purec
# endif
#else
# define fast_exp exp
# define fast_log log
# define fast_tanh tanh
#endif
