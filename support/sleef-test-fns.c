#include <math.h>
#include <string.h>

#include "../net/intrinsics-use.h"
#include "../net/sleef-use-scalar.h"

double do_tanh  (double a) { return fast_tanh(a); }
double do_tanhx (double a) { return 2/(1+fast_exp(-2*a)) - 1; }
double do_exp   (double a) { return fast_exp(a); }
double do_log   (double a) { return fast_log(a); }
double do_zero  (double a) { return 0.0; }
