/* XGEN.C - Generate cases for binary response mixture of experts example. */

#include <stdio.h>
#include <math.h>

#include "rand.h"

#define N_cases 10000

int main(void)
{
  double x1, x2, x1n, x2n, v;
  int i, b, w;

  for (i = 0; i<N_cases; i++)
  {
    x1 = 1.3*rand_gaussian();
    x2 = 1.3*rand_gaussian();

    x1n = x1 + 0.03*rand_gaussian();
    x2n = x2 + 0.03*rand_gaussian();

    if (x1n+x2n>1.5)
    { v = -3+x1n+1.3*x2n;
    }
    else if (x2n>3*sin(0.1+x1n*0.9))
    { v = 2*sin(3.05+9*x1n)-0.5;
    }
    else
    { v = 3*sin(0.95+8*x2n)+1;
    }

    b = rand_uniform() < 1/(1+exp(-v));

    printf(" %+8.5f %+8.5f %d\n",x1,x2,b);
  }

  return 0;
}
