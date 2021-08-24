#include <math.h>
#include <string.h>

#include "../net/intrinsics-use.h"
#include "../net/sleef-use-simd.h"

#if USE_SLEEF && USE_SIMD_INTRINSICS && __AVX__ && 0

#else

void do_tanh (double *restrict a, double *restrict b)
{ int i;
  for (i = 0; i<4; i++)
  { b[i] = tanh(a[i]);
  }
}

void do_tanhx (double *restrict a, double *restrict b)
{ int i;
  for (i = 0; i<4; i++)
  { b[i] = 2/(1+exp(-2*a[i])) - 1;
  }
}

void do_exp (double *restrict a, double *restrict b)
{ int i;
  for (i = 0; i<4; i++)
  { b[i] = exp(a[i]);
  }
}

void do_log (double *restrict a, double *restrict b)
{ int i;
  for (i = 0; i<4; i++)
  { b[i] = log(a[i]);
  }
}

void do_zero(double *restrict a, double *restrict b)
{ int i;
  for (i = 0; i<4; i++)
  { b[i] = 0.0;
  }
}

#endif
