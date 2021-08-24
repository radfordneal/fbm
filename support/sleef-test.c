/* Program for performance testing of SLEEF scalar functions. 

   Compile with 

       gcc -O3 -march=native sleef-test.c sleef-test-fns.c -o sleef-test

   with or without adding -DUSE_SLEEF, -DUSE_SIMD_INTRINSICS, and -DUSE_FMA

   Run with 

       sleef-test <fn> <n>

   where <fn> is tanh, tanhx, exp, or log, and <n> is the number of calls to do.
   The tanhx option computes tanh as 2/(1+exp(-2*a)) - 1.  One can also pass
   zero to get the function that returns the constant zero (to check overhead).
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern double do_tanh(double), do_tanhx(double), do_exp(double), 
              do_log(double), do_zero(double);

double arg = 2.1;

int main (int argc, char **argv)
{
  double (*p)(double);
  double b;
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
    : (abort(), (double(*)(double))0);

  while (n>0) { b = p(arg); n -= 1; }

  printf("%.6f\n",b);
  return 0;
}
