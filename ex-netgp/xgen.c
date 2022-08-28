/* XGEN.C - Generate cases for binary response with regions example. */

#include <stdio.h>
#include <math.h>

#include "rand.h"

#define N_cases 13000   /* Total number of cases */
#define N_train 3000    /* Cases potentially used for training */

int main(void)
{
  double x1, x2, v, p;
  int i, b, w;
  double lp, sqe;
  int e;

  lp = 0;
  sqe = 0;
  e = 0;

  for (i = 0; i<N_cases; i++)
  {
    x1 = 1.7*rand_gaussian();
    x2 = 1.5*rand_gaussian();

    if (x1+x2>1.5)
    { v = -3+x1+1.3*x2;
    }
    else if (x2>3*sin(0.1+x1*0.9))
    { v = 2*sin(3.05+9*x1+0.04*x1*x1)-0.5;
    }
    else
    { v = 3*sin(0.95+8*x2+0.06*x2*x2)+1;
    }

    p = 1/(1+exp(-v));
    b = rand_uniform() < p;

    printf(" %+8.5f %+8.5f %d\n",x1,x2,b);

    if (i>=N_train)
    { lp += b ? log(p) : log(1-p);
      sqe += (b-p)*(b-p);
      e += b != (p>0.5);
    }
  }

  fprintf(stderr,"Average log probability of test cases: %.5lf\n",
                  lp/(N_cases-N_train));

  fprintf(stderr,"Average squared error on test cases: %.5lf\n",
                  sqe/(N_cases-N_train));

  fprintf(stderr,"Error rate on test cases: %.5lf\n",
                  (double)e/(N_cases-N_train));

  return 0;
}
