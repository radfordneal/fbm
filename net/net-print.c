/* NET-PRINT.C - Procedures to print network parameters and hyperprameters. */

/* Copyright (c) 1995-2021 by Radford M. Neal 
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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "cuda-use.h"

#include "misc.h"
#include "log.h"
#include "prior.h"
#include "data.h"
#include "model.h"
#include "net.h"


/* This module is used to print the parameters and hyperparameters in a
   network.  See the documentation in net-display.c for the formats.
*/


static void print_param_array (net_param *, int, int);
static void print_adjustment_array (net_sigma *, int);
static void print_sigma_array (net_sigma *, int);


/* PRINT PARAMETERS, OPTIONALLY ACCOMPANIED BY HYPERPARAMETERS. */

void net_print_params
( net_params *w,	/* Network parameters */
  net_sigmas *s,	/* Network sigmas, null if none to display */
  net_arch *a, 		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  model_specification *m, /* Data model, may be null */
  int group             /* Group to print, 0 for all */
)
{
  int i, j, l, ls, nsqi, g;
  unsigned bits;
  char ps[1000];

  nsqi = 0;
  g = 0; 

  if (a->has_ti && (++g==group || group==0))
  { printf("\nInput Offsets [%d]\n\n",g);
    if (s!=0) printf("%10.2f:",*s->ti_cm);
    print_param_array (w->ti, a->N_inputs, s!=0);
  }

  for (l = 0; l<a->N_layers; l++)
  {
    for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (bits&1)
      { if (ls>=l-1) abort();
        if (++g==group || group==0)
        { printf("\nHidden Layer %d to Hidden Layer %d Weights [%d]\n\n",
                 ls,l,g);
          if (s!=0) printf("%5.2f",*s->nsq_cm[nsqi]);
          if (a->nonseq_config[nsqi])
          { if (s!=0) printf(":     ");
            print_param_array(w->nsq[nsqi],a->nonseq_config[nsqi]->N_wts,s!=0);
          }
          else
          { for (i = 0; i<a->N_hidden[ls]; i++)
            { if (i>0) printf("\n");
              if (s!=0 && i>0) printf("     ");
              if (s!=0) printf(" %4.2f:",s->nsq[nsqi][i]);
              print_param_array
                (w->nsq[nsqi]+a->N_hidden[l]*i, a->N_hidden[l], s!=0);
            }
          }
        }
        nsqi += 1;
      }
    }

    if (l>0 && a->has_hh[l-1] && (++g==group || group==0))
    { printf("\nHidden Layer %d to Hidden Layer %d Weights [%d]\n\n",l-1,l,g);
      if (s!=0) printf("%5.2f",*s->hh_cm[l-1]);
      if (a->hidden_config[l])
      { if (s!=0) printf(":     ");
        print_param_array(w->hh[l-1], a->hidden_config[l]->N_wts, s!=0);
      }
      else
      { for (i = 0; i<a->N_hidden[l-1]; i++)
        { if (i>0) printf("\n");
          if (s!=0 && i>0) printf("     ");
          if (s!=0) printf(" %4.2f:",s->hh[l-1][i]);
          print_param_array(w->hh[l-1]+a->N_hidden[l]*i, a->N_hidden[l], s!=0);
        }
      }
    }
  
    if (a->has_ih[l] && (++g==group || group==0))
    { printf("\nInput to Hidden Layer %d Weights [%d]",l,g);
      if (flgs && list_flags(flgs->omit,a->N_inputs,1<<(l+1),ps)!=0)
      { printf(" (omit%s)",ps);
      }
      printf("\n\n");
      if (s!=0) printf("%5.2f",*s->ih_cm[l]);
      if (a->input_config[l])
      { if (s!=0) printf(":     ");
        print_param_array(w->ih[l], a->input_config[l]->N_wts, s!=0);
      }
      else
      { j = 0;
        for (i = 0; i<a->N_inputs; i++)
        { if (flgs==0 || (flgs->omit[i]&(1<<(l+1)))==0)
          { if (j>0) printf("\n");
            if (s!=0 && j>0) printf("     ");
            if (s!=0) printf(" %4.2f:",s->ih[l][j]);
            print_param_array(w->ih[l]+a->N_hidden[l]*j, a->N_hidden[l], s!=0);
            j += 1;
          }
        }
        if (j==0) printf("\n");
      }
    }

    if (a->has_bh[l] && (++g==group || group==0))
    { printf("\nHidden Layer %d Biases [%d]\n\n",l,g);
      if (s!=0) printf("%10.2f:",*s->bh_cm[l]);
      if (a->bias_config[l])
      { print_param_array (w->bh[l], a->bias_config[l]->N_wts, s!=0);
      }
      else
      { print_param_array (w->bh[l], a->N_hidden[l], s!=0);
      }
    }

    if (a->has_ah[l] && (++g==group || group==0))
    { if (s!=0)
      { printf("\nHidden Layer %d Adjustments [%d]\n\n",l,g);
        printf("           ");
        print_adjustment_array(s->ah[l],a->N_hidden[l]);
      }
    }
  
    if (a->has_th[l] && (++g==group || group==0))
    { printf("\nHidden Layer %d Offsets [%d]\n\n",l,g);
      if (s!=0) printf("%10.2f:",*s->th_cm[l]);
      print_param_array (w->th[l], a->N_hidden[l], s!=0);
    }
  }

  for (l = a->N_layers-1; l>=0; l--)
  { if (a->has_ho[l] && (++g==group || group==0))
    { int k = 2*a->N_layers-1-l;
      printf("\nHidden Layer %d to Output Weights [%d]\n\n",l,g);
      if (s!=0) printf("%5.2f",*s->ho_cm[l]);
      if (a->hidden_config[k])
      { if (s!=0) printf(":     ");
        print_param_array (w->ho[l], a->hidden_config[k]->N_wts, s!=0);
      }
      else
      { for (i = 0; i<a->N_hidden[l]; i++)
        { if (i>0) printf("\n");
          if (s!=0 && i>0) printf("     ");
          if (s!=0) printf(" %4.2f:",s->ho[l][i]);
          print_param_array (w->ho[l]+a->N_outputs*i, a->N_outputs, s!=0);
        }
      }
    }
  }

  if (a->has_io && (++g==group || group==0))
  { printf("\nInput to Output Weights [%d]",g);
    if (flgs && list_flags(flgs->omit,a->N_inputs,1,ps)!=0)
    { printf(" (omit%s)",ps);
    }
    printf("\n\n");
    if (s!=0) printf("%5.2f",*s->io_cm);
    if (a->input_config[a->N_layers])
    { if (s!=0) printf(":     ");
      print_param_array(w->io, a->input_config[a->N_layers]->N_wts, s!=0);
    }
    else
    { j = 0;
      for (i = 0; i<a->N_inputs; i++)
      { if (flgs==0 || (flgs->omit[i]&1)==0)
        { if (j>0) printf("\n");
          if (s!=0 && j>0) printf("     ");
          if (s!=0) printf(" %4.2f:",s->io[j]);
          print_param_array (w->io+a->N_outputs*j, a->N_outputs, s!=0);
          j += 1;
        }
      }
      if (j==0) printf("\n");
    }
  }

  if (a->has_bo && (++g==group || group==0))
  { printf("\nOutput Biases [%d]\n\n",g);
    if (s!=0) printf("%10.2f:",*s->bo_cm);
    if (a->bias_config[a->N_layers])
    { print_param_array (w->bo, a->bias_config[a->N_layers]->N_wts, s!=0);
    }
    else   
    { print_param_array (w->bo, a->N_outputs, s!=0);
    }
  }

  if (a->has_ao && (++g==group || group==0))
  { if (s!=0)
    { printf("\nOutput Adjustments [%d]\n\n",g);
      printf("           ");
      print_adjustment_array(s->ao,a->N_outputs);
    }
  }

  if (m!=0 && m->type=='R' && s!=0 && (++g==group || group==0))
  { printf("\nNoise levels [%d]\n\n",g);
    printf("%7.2f - ",*s->noise_cm);
    print_sigma_array(s->noise,a->N_outputs);
  }

  if (group && g<group)
  { printf("\nThere is no group %d\n",group);
  }
}


/* PRINT HYPERPARAMETERS. */

void net_print_sigmas
( net_sigmas *s,	/* Network sigmas, null if none to display */
  net_arch *a, 		/* Network architecture */
  net_flags *flgs,	/* Network flags, null if none */
  model_specification *m, /* Data model, may be null */
  int group             /* Group to print, 0 for all */
)
{
  int l, ls, nsqi, g;
  unsigned bits;

  nsqi = 0;
  g = 0;

  if (a->has_ti && (++g==group || group==0))
  { printf("\nInput Offsets [%d]\n\n",g);
    printf("%7.2f\n",*s->ti_cm);
  }

  for (l = 0; l<a->N_layers; l++)
  {
    for (ls = 0, bits = a->has_nsq[l]; bits!=0; ls++, bits>>=1)
    { if (bits&1)
      { if (ls>=l-1) abort();
        if (++g==group || group==0)
        { printf("\nHidden Layer %d to Hidden Layer %d Weights [%d]\n\n",
                  ls,l,g);
          printf("%7.2f - ",*s->nsq_cm[nsqi]);
          print_sigma_array(s->nsq[nsqi],a->N_hidden[ls]);
        }
        nsqi += 1;
      }
    }

    if (l>0 && a->has_hh[l-1] && (++g==group || group==0))
    { printf("\nHidden Layer %d to Hidden Layer %d Weights [%d]\n\n",l-1,l,g);
      printf("%7.2f - ",*s->hh_cm[l-1]);
      print_sigma_array(s->hh[l-1],a->N_hidden[l-1]);
    }
  
    if (a->has_ih[l] && (++g==group || group==0))
    { printf("\nInput to Hidden Layer %d Weights [%d]\n\n",l,g);
      printf("%7.2f - ",*s->ih_cm[l]);
      print_sigma_array(s->ih[l],
        not_omitted(flgs?flgs->omit:0,a->N_inputs,1<<(l+1)));
    }

    if (a->has_bh[l] && (++g==group || group==0))
    { printf("\nHidden Layer %d Biases [%d]\n\n",l,g);
      printf("%7.2f\n",*s->bh_cm[l]);
    }

    if (a->has_ah[l] && (++g==group || group==0))
    { printf("\nHidden Layer %d Adjustments [%d]\n\n",l,g);
      printf("          ");
      print_sigma_array(s->ah[l],a->N_hidden[l]);
    }
  
    if (a->has_th[l] && (++g==group || group==0))
    { printf("\nHidden Layer %d Offsets [%d]\n\n",l,g);
      printf("%7.2f\n",*s->th_cm[l]);
    }
  }

  for (l = a->N_layers-1; l>=0; l--)
  { if (a->has_ho[l] && (++g==group || group==0))
    { printf("\nHidden Layer %d to Output Weights [%d]\n\n",l,g);
      printf("%7.2f - ",*s->ho_cm[l]);
      print_sigma_array(s->ho[l],a->N_hidden[l]);
    }
  }

  if (a->has_io && (++g==group || group==0))
  { printf("\nInput to Output Weights [%d]\n\n",g);
    printf("%7.2f - ",*s->io_cm);
    print_sigma_array(s->io,
        not_omitted(flgs?flgs->omit:0,a->N_inputs,1));
  }

  if (a->has_bo && (++g==group || group==0))
  { printf("\nOutput Biases [%d]\n\n",g);
    printf("%7.2f\n",*s->bo_cm);
  }

  if (a->has_ao && (++g==group || group==0))
  { printf("\nOutput Adjustments [%d]\n\n",g);
    printf("          ");
    print_sigma_array(s->ao,a->N_outputs);
  }

  if (m!=0 && m->type=='R' && (++g==group || group==0))
  { printf("\nNoise levels [%d]\n\n",g);
    printf("%7.2f - ",*s->noise_cm);
    print_sigma_array(s->noise,a->N_outputs);
  }

  if (group && g<group)
  { printf("\nThere is no group %d\n",group);
  }
}


/* PRINT ARRAY OF PARAMETERS.  The array may have to extend over several
   lines.  If op is non-zero, each new line (but not the first) is preceded 
   by eleven spaces. */

static void print_param_array
( net_param *p,
  int n,
  int op
)
{ 
  int i;

  for (i = 0; i<n; i++)
  { if (i!=0)
    { if (i%10==0) 
      { printf("\n");
        if (op) printf("           ");
      }
      else printf(" ");
    }
    if (op) printf("%+6.2f",p[i]);
    else    printf("%+7.3f",p[i]);
  }

  printf("\n");
}


/* PRINT ARRAY OF SIGMA VALUES.  The array may have to extend over several
   lines.  Each new line (but not the first) is preceded by ten spaces. */

static void print_sigma_array
( net_sigma *s,
  int n
)
{ 
  int i;

  for (i = 0; i<n; i++)
  { if (i!=0)
    { if (i%10==0) printf("\n          ");
      else printf(" ");
    }
    printf("%5.2f",s[i]);
  }

  printf("\n");
}


/* PRINT ARRAY OF ADJUSTMENT VALUES.  The array may have to extend over several
   lines.  Each new line (but not the first) is preceded by eleven spaces. */

static void print_adjustment_array
( net_sigma *s,
  int n
)
{ 
  int i;

  for (i = 0; i<n; i++)
  { if (i!=0)
    { if (i%10==0) printf("\n           ");
      else printf(" ");
    }
    printf("%6.2f",s[i]);
  }

  printf("\n");
}
