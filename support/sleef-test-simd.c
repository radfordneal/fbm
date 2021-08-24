/* Program for performance testing of SLEEF SIMD functions. 

   Compile with 

     gcc -O3 -march=native sleef-test-simd.c sleef-test-fns-simd.c \
         -o sleef-test-simd

   with or without adding -DUSE_SLEEF, -DUSE_SIMD_INTRINSICS, and -DUSE_FMA

   Run with 

     sleef-test-simd <fn> <n>

   where <fn> is tanh, tanhx, exp, or log, and <n> is the number of calls to do.
   The tanhx option computes tanh as 2/(1+exp(-2*a)) - 1.  One can also pass
   zero to get the function that returns the constant zero (to check overhead).
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern void do_tanh(double*restrict,double*restrict), 
            do_tanhx(double*restrict,double*restrict), 
            do_exp(double*restrict,double*restrict), 
            do_log(double*restrict,double*restrict), 
            do_zero(double*restrict,double*restrict);

double arg[4] = { 2.1, 1.4, 3.7, 0.2 };

int main (int argc, char **argv)
{
  void (*p)(double*restrict,double*restrict);
  double b[4];
  char *f;
  int n;

  if (argc != 3) abort();
  f = argv[1];
  n = atoi(argv[2]);
  
  p = strcmp(f,"tanh")==0 ? do_tanh
    : strcmp(f,"tanhx")==0 ? do_tanhx
    : strcmp(f,"exp")==0 ? do_exp
    : strcmp(f,"log")==0 ? do_log
    : strcmp(f,"zero")==0 ? do_zero
    : (abort(), (void(*)(double*restrict,double*restrict))0);

  while (n>0) { p(arg,b); n -= 1; }

  printf("%.6f %.6f %.6f %.6f\n",b[0],b[1],b[2],b[3]);
  return 0;
}
