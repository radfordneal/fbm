/* IGEN.C - Generate cases for image classification problem. */

#include <stdio.h>
#include <math.h>

#include "rand.h"

#define N_cases 4000
#define N_train 1536

#define D 6
#define noise 0.5

double image[D][D];

int main (int argc, char **argv)
{
  int i, j, k, cc, ci, cj, ck, pi, pc, pj, pk;
  double pp[4];

  fprintf (stderr, "N_cases: %d, N_train: %d, noise: %f\n\n",
                    N_cases, N_train, noise);

  double logp = 0;
  double sqerr = 0;
  int errors = 0;

  for (i = 0; i<N_cases; i++)
  {
    /* Generate random background.. */

    for (j = 0; j<D; j++)
    { for (k = 0; k<D; k++)
      { image[j][k] = rand_gaussian();
      }
    }

    /* Generate random location and class of object. */

    cj = rand_int(D-3+1);
    ck = rand_int(D-3+1);

    ci = rand_int(4);
    cc = "+XOH"[ci];

    /* Replace background with object at its location. */

    switch (cc)
    { case '+':
      { image[cj+0][ck+0] = -1; image[cj+0][ck+1] = +1; image[cj+0][ck+2] = -1;
        image[cj+1][ck+0] = +1; image[cj+1][ck+1] = +1; image[cj+1][ck+2] = +1;
        image[cj+2][ck+0] = -1; image[cj+2][ck+1] = +1; image[cj+2][ck+2] = -1;
        break;
      }
      case 'X':
      { image[cj+0][ck+0] = +1; image[cj+0][ck+1] = -1; image[cj+0][ck+2] = +1;
        image[cj+1][ck+0] = -1; image[cj+1][ck+1] = +1; image[cj+1][ck+2] = -1;
        image[cj+2][ck+0] = +1; image[cj+2][ck+1] = -1; image[cj+2][ck+2] = +1;
        break;
      }
      case 'O':
      { image[cj+0][ck+0] = +1; image[cj+0][ck+1] = +1; image[cj+0][ck+2] = +1;
        image[cj+1][ck+0] = +1; image[cj+1][ck+1] = -1; image[cj+1][ck+2] = +1;
        image[cj+2][ck+0] = +1; image[cj+2][ck+1] = +1; image[cj+2][ck+2] = +1;
        break;
      }
      case 'H':
      { image[cj+0][ck+0] = +1; image[cj+0][ck+1] = -1; image[cj+0][ck+2] = +1;
        image[cj+1][ck+0] = +1; image[cj+1][ck+1] = +1; image[cj+1][ck+2] = +1;
        image[cj+2][ck+0] = +1; image[cj+2][ck+1] = -1; image[cj+2][ck+2] = +1;
        break;
      }
    }

    /* Add random noise, and round to three decimal places. */

    for (j = 0; j<D; j++)
    { for (k = 0; k<D; k++)
      { image[j][k] += round (1000 * noise * rand_gaussian()) / 1000;
      }
    }

    /* Send image data and class to standard output, as training/test case. */

    for (j = 0; j<D; j++)
    { for (k = 0; k<D; k++)
      { printf("%.3f ",image[j][k]);
      }
    }
    printf("%d\n",ci);

    /* Compute optimal posterior probabilities with true model known. */

    for (pi = 0; pi<4; pi++)
    { pc = "+XOH"[pi];
      pp[pi] = 0;
      for (pj = 0; pj<=D-3; pj++)
      { for (pk = 0; pk<=D-3; pk++)
        { double p = 1;
          for (j = 0; j<D; j++)
          { for (k = 0; k<D; k++)
            { double im = image[j][k];
              double nv = noise*noise;
              if (j<pj || j>pj+2 || k<pk || k>pk+2)
              { p *= exp(-0.5*im*im/(1+nv));
              }
              else
              { switch (pc)
                { case '+':
                  { if (j==pj+0 && k==pk+0) p *= exp(-0.5*(im+1)*(im+1)/nv);
                    if (j==pj+0 && k==pk+1) p *= exp(-0.5*(im-1)*(im-1)/nv); 
                    if (j==pj+0 && k==pk+2) p *= exp(-0.5*(im+1)*(im+1)/nv);
                    if (j==pj+1 && k==pk+0) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+1 && k==pk+1) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+1 && k==pk+2) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+2 && k==pk+0) p *= exp(-0.5*(im+1)*(im+1)/nv);
                    if (j==pj+2 && k==pk+1) p *= exp(-0.5*(im-1)*(im-1)/nv); 
                    if (j==pj+2 && k==pk+2) p *= exp(-0.5*(im+1)*(im+1)/nv);
                    break;
                  }
                  case 'X':
                  { if (j==pj+0 && k==pk+0) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+0 && k==pk+1) p *= exp(-0.5*(im+1)*(im+1)/nv); 
                    if (j==pj+0 && k==pk+2) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+1 && k==pk+0) p *= exp(-0.5*(im+1)*(im+1)/nv);
                    if (j==pj+1 && k==pk+1) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+1 && k==pk+2) p *= exp(-0.5*(im+1)*(im+1)/nv);
                    if (j==pj+2 && k==pk+0) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+2 && k==pk+1) p *= exp(-0.5*(im+1)*(im+1)/nv); 
                    if (j==pj+2 && k==pk+2) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    break;
                  }
                  case 'O':
                  { if (j==pj+0 && k==pk+0) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+0 && k==pk+1) p *= exp(-0.5*(im-1)*(im-1)/nv); 
                    if (j==pj+0 && k==pk+2) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+1 && k==pk+0) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+1 && k==pk+1) p *= exp(-0.5*(im+1)*(im+1)/nv);
                    if (j==pj+1 && k==pk+2) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+2 && k==pk+0) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+2 && k==pk+1) p *= exp(-0.5*(im-1)*(im-1)/nv); 
                    if (j==pj+2 && k==pk+2) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    break;
                  }
                  case 'H':
                  { if (j==pj+0 && k==pk+0) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+0 && k==pk+1) p *= exp(-0.5*(im+1)*(im+1)/nv); 
                    if (j==pj+0 && k==pk+2) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+1 && k==pk+0) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+1 && k==pk+1) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+1 && k==pk+2) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+2 && k==pk+0) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    if (j==pj+2 && k==pk+1) p *= exp(-0.5*(im+1)*(im+1)/nv); 
                    if (j==pj+2 && k==pk+2) p *= exp(-0.5*(im-1)*(im-1)/nv);
                    break;
                  }
                }
              }
            }
          }
          pp[pi] += p;
        }
      }
    }
    double spp = pp[0]+pp[1]+pp[2]+pp[3];
    for (pi = 0; pi<4; pi++)
    { pp[pi] /= spp;
    }

    /* Compute the error rate, average log probability, and average squared
       error (for 0-1 representation) based on the true model for test cases. */

    if (i>=N_train)
    { logp += log(pp[ci]);
      for (pi = 0; pi<4; pi++)
      { sqerr += pi==ci ? (1-pp[pi])*(1-pp[pi]) : pp[pi]*pp[pi];
      }
      for (pi = 0; pi<4; pi++)
      { if (pp[pi]>pp[ci])
        { errors += 1;
          break;
        }
      }
    }

    /* Display image and other data, if there's a program argument. */

    if (argc>1)
    { 
      fprintf(stderr,
 "Case %d, Class %c(%d), Centred at %d,%d, PP: %c%.3f %c%.3f %c%.3f %c%.3f\n\n",
       i+1, cc, ci, cj+1, ck+1, '+', pp[0], 'X', pp[1], 'O', pp[2], 'H', pp[3]);
      fprintf(stderr,"   ");
      for (k = 0; k<D; k++)
      { fprintf(stderr,"   %d   ",k);
      }
      fprintf(stderr,"   ");
      for (k = 0; k<D; k++)
      { fprintf(stderr," %d",k);
      }
      fprintf(stderr,"\n");
      for (j = 0; j<D; j++)
      { fprintf(stderr,"%d  ",j);
        for (k = 0; k<D; k++)
        { fprintf(stderr," %+.3f",image[j][k]);
        }
        fprintf(stderr,"   ");
        for (k = 0; k<D; k++)
        { fprintf(stderr," %c",
            image[j][k]<-0.5 ? 'O' : image[j][k]>0.5 ? '#' : ' ');
        }
        fprintf(stderr,"\n");
      }
      fprintf(stderr,"\n\n");
    }
  }

  /* Display error and negative log probability summary. */

  fprintf (stderr,
   "Error rate on test cases with true model: %.3f\n",
   (double)errors/(N_cases-N_train));
  fprintf (stderr,
   "Average squared error for test cases with true model: %.3f\n",
   sqerr/(N_cases-N_train));
  fprintf (stderr,
   "Average log probability for test cases with true model: %.3f\n",
   logp/(N_cases-N_train));

  return 0;
}
