#include <math.h>
#include <string.h>

#include "../net/intrinsics-use.h"
#include "../net/sleef-use-simd.h"

#if USE_SLEEF && USE_SIMD_INTRINSICS && __AVX2__ && USE_FMA && __FMA__

void do_tanh (double *restrict a, double *restrict b)
{ _mm256_storeu_pd (b, Sleef_tanhd4_u35avx2 (_mm256_loadu_pd(a)));
}

void do_tanhx (double *restrict a, double *restrict b)
{ __m256d x;
  x = _mm256_loadu_pd(a);
  x = _mm256_add_pd(x,x);
  x = Sleef_expd4_u10avx2(x);
  __m256d one = _mm256_set1_pd(1.0);
  __m256d two = _mm256_add_pd(one,one);
  x = _mm256_sub_pd (one, _mm256_div_pd (two, _mm256_add_pd(one,x)));
  _mm256_storeu_pd(b,x);
}

void do_exp (double *restrict a, double *restrict b)
{ _mm256_storeu_pd (b, Sleef_expd4_u10avx2 (_mm256_loadu_pd(a)));
}

void do_log (double *restrict a, double *restrict b)
{ _mm256_storeu_pd (b, Sleef_logd4_u10avx2 (_mm256_loadu_pd(a)));
}

void do_zero(double *restrict a, double *restrict b)
{ _mm256_storeu_pd (b, _mm256_setzero_pd());
}

#elif USE_SLEEF && USE_SIMD_INTRINSICS && __AVX__

void do_tanh (double *restrict a, double *restrict b)
{ _mm256_storeu_pd (b, Sleef_tanhd4_u35avx (_mm256_loadu_pd(a)));
}

void do_tanhx (double *restrict a, double *restrict b)
{ __m256d x;
  x = _mm256_loadu_pd(a);
  x = _mm256_add_pd(x,x);
  x = Sleef_expd4_u10avx(x);
  __m256d one = _mm256_set1_pd(1.0);
  __m256d two = _mm256_add_pd(one,one);
  x = _mm256_sub_pd (one, _mm256_div_pd (two, _mm256_add_pd(one,x)));
  _mm256_storeu_pd(b,x);
}

void do_exp (double *restrict a, double *restrict b)
{ _mm256_storeu_pd (b, Sleef_expd4_u10avx (_mm256_loadu_pd(a)));
}

void do_log (double *restrict a, double *restrict b)
{ _mm256_storeu_pd (b, Sleef_logd4_u10avx (_mm256_loadu_pd(a)));
}

void do_zero(double *restrict a, double *restrict b)
{ _mm256_storeu_pd (b, _mm256_setzero_pd());
}

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
  { b[i] = 1 - 2/(1+exp(2*a[i]));
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
