/* MC-TRAJ.C - Procedures for computing dynamical trajectories. */

/* Copyright (c) 1995-2019 by Radford M. Neal 
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

#include "misc.h"
#include "rand.h"
#include "log.h"
#include "mc.h"


/* HOW TO COMPUTE THE CURRENT TRAJECTORY. */

int approx_order[Max_approx]; /* Order in which approximations are used */

static mc_traj *tj;	/* Description of how to compute trajectory */
static mc_iter *it;	/* Description of iteration (contains approx order) */

static int firsti;	/* First coordinate to update (from 0) */
static int lasti;	/* Last coordinate to update */

static int na;		/* Number of distinct approximations used */
static int ta;		/* Total approximations, counting repeats for symmetry*/

float traj_frac = 1.0;	/* Global variable giving fraction of terms (eg, case)
			   to use when computing gradient.  Used only by 
			   modules that know about it. */

int new_frac = 0;	/* Set to 1 for a new trajectory requiring a new 
			   random fraction. */


/* SET UP FOR COMPUTING A TRAJECTORY.  Passed the description of how to
   compute the trajectory and the stepsizes to use.  Saves these, and
   sets other variables describing how the trajectory will be computed
   this time. */

void mc_traj_init
( mc_traj *tj0,		/* Description of how to compute trajectory */
  mc_iter *it0,		/* Description of this iteration */
  int firsti0,		/* First and last coordinates to update */
  int lasti0
)
{ 
  tj = tj0;
  it = it0;

  firsti = firsti0;
  lasti = lasti0;

  if (tj->type!='L')
  { na = 1;
    ta = 1;
  }
  else if (tj->N_approx>0)
  { na = tj->N_approx;
    ta = na;
  }
  else
  { na = -tj->N_approx;
    ta = 2*na;
  }

  if (na>Max_approx)
  { fprintf (stderr, "Using too many energy approximations (max %d)\n",
             Max_approx);
    exit(1);
  }

  traj_frac = 1-tj->frac;
  new_frac = 1;
}


/* PERMUTE ORDER OF APPROXIMATIONS. */

void mc_traj_permute (void)
{
  int i, j, t;

  for (i = 0; i<na; i++) 
  { approx_order[i] = i+1;
  }

  for (i = 0; i<na; i++)
  { t = approx_order[i];
    j = i + rand_int(na-i);
    approx_order[i] = approx_order[j];
    approx_order[j] = t;
  }
}


/* COMPUTE SOME NUMBER OF STEPS OF THE TRAJECTORY.  Updates the dynamical
   state by following the trajectory for some number of steps, computed
   as previously specified using mc_traj_init.  The number of steps may
   be negative, in which case the state is taken backwards along the
   trajectory that number of steps.  If the last parameter is non-zero,
   the potential energy will be evaluated at the last state, if it is
   convenient to do so. */

void mc_trajectory
( mc_dynamic_state *ds,	/* Dynamical state to update */
  int n,		/* Number of steps to compute */
  int need_pot		/* Need potential energy for last state? */
)
{
  double sf, sfh;
  int k, a, x, o;

  if (n==0) return;

  /* Set up stepsize, negating if a backward trajectory is desired.  Also
     divide stepsize by the number of approximations used, and multiply
     number of steps by the same number, to give a constant total time. */

  sf = it->stepsize_factor / ta;

  x = +1; 

  if (n<0)
  { n = -n;
    sf = -sf;
    x = -1;
  }

  n *= ta;

  sfh = sf / 2;

  /* Compute the trajectory, using as many approximations as asked for. */

  mc_value *restrict ds_q = ds->q;
  mc_value *restrict ds_p = ds->p;
  mc_value *restrict ds_grad = ds->grad;
  mc_value *restrict ds_stepsize = ds->stepsize;

  switch (tj->type)
  { 
    case '2': case 'G':
    { 
      double a1, a2, b1, b2;

      if (tj->rev_sym==1) sf /= 2;

      a2 = tj->param;
      a1 = 1-a2;
      b2 = 0.5/a1;
      b1 = 1-b2;

      a1 *= sf; a2 *= sf; b1 *= sf; b2 *= sf;

      if (tj->halfp)
      {
        /* Compute trajectory with initial step for momentum (in non-reversed
           form). */

        while (n>0)
        {
          if (tj->rev_sym!=-1)
          { 
            if (ds->know_grad!=1)
            { mc_app_energy(ds,1,1,0,ds_grad);
            }
  
            for (k = firsti; k<=lasti; k++)
            { ds_p[k] -= b1 * ds_stepsize[k] * ds_grad[k];
              ds_q[k] += a1 * ds_stepsize[k] * ds_p[k];
            }
  
            mc_app_energy(ds,1,1,0,ds_grad);
  
            for (k = firsti; k<=lasti; k++)
            { ds_p[k] -= b2 * ds_stepsize[k] * ds_grad[k];
              ds_q[k] += a2 * ds_stepsize[k] * ds_p[k];
            }

            ds->know_grad = 0;
            ds->know_pot = 0;
          }

          if (tj->rev_sym!=0)
          {
            for (k = firsti; k<=lasti; k++)
            { ds_q[k] += a2 * ds_stepsize[k] * ds_p[k];
            }

            mc_app_energy(ds,1,1,0,ds_grad);

            for (k = firsti; k<=lasti; k++)
            { ds_p[k] -= b2 * ds_stepsize[k] * ds_grad[k];
              ds_q[k] += a1 * ds_stepsize[k] * ds_p[k];
            }

            mc_app_energy (ds, 1, 1, 
                           need_pot && n==1 ? &ds->pot_energy : 0,
                           ds_grad);

            for (k = firsti; k<=lasti; k++)
            { ds_p[k] -= b1 * ds_stepsize[k] * ds_grad[k];
            }

            ds->know_grad = 1;
            ds->know_pot = need_pot && n==1;
          }
  
          n -= 1;
        }
      }
      else
      {
        /* Compute trajectory with initial step for position (in non-reversed
           form). */

        while (n>0)
        {
          if (tj->rev_sym!=-1)
          {
            for (k = firsti; k<=lasti; k++)
            { ds_q[k] += b1 * ds_stepsize[k] * ds_p[k];
            }
    
            mc_app_energy(ds,1,1,0,ds_grad);
  
            for (k = firsti; k<=lasti; k++)
            { ds_p[k] -= a1 * ds_stepsize[k] * ds_grad[k];
              ds_q[k] += b2 * ds_stepsize[k] * ds_p[k];
            }
    
            mc_app_energy (ds, 1, 1, 
                       need_pot && n==1 && tj->rev_sym==0 ? &ds->pot_energy : 0,
                       ds_grad);
  
            for (k = firsti; k<=lasti; k++)
            { ds_p[k] -= a2 * ds_stepsize[k] * ds_grad[k];
            }
    
            ds->know_grad = 1;
            ds->know_pot  = need_pot && n==1 && tj->rev_sym==0;
          }

          if (tj->rev_sym!=0)
          {
            if (ds->know_grad!=1)
            { mc_app_energy(ds,1,1,0,ds_grad);
            }

            for (k = firsti; k<=lasti; k++)
            { ds_p[k] -= a2 * ds_stepsize[k] * ds_grad[k];
              ds_q[k] += b2 * ds_stepsize[k] * ds_p[k];
            }

            mc_app_energy(ds,1,1,0,ds_grad);
  
            for (k = firsti; k<=lasti; k++)
            { ds_p[k] -= a1 * ds_stepsize[k] * ds_grad[k];
              ds_q[k] += b1 * ds_stepsize[k] * ds_p[k];
            }

            ds->know_grad = 0;
            ds->know_pot = 0;
          }

          n -= 1;
        }
      }

      break;
    }

    case '4':
    {
      static double a[4] =
      {  0.5153528374311229364,
        -0.085782019412973646,
         0.4415830236164665242,
         0.1288461583653841854
      };

      static double b[4] =
      {  0.1344961992774310892,
        -0.2248198030794208058,
         0.7563200005156682911,
         0.3340036032863214255
      };

      if (tj->rev_sym==1) sf /= 2;

      while (n>0)
      {
        if (tj->rev_sym!=-1)
        {
          if (ds->know_grad!=1)
          { mc_app_energy(ds,1,1,0,ds_grad);
          }

          for (k = firsti; k<=lasti; k++)
          { ds_p[k] -= sf * b[0] * ds_stepsize[k] * ds_grad[k];
            ds_q[k] += sf * a[0] * ds_stepsize[k] * ds_p[k];
          }
  
          mc_app_energy(ds,1,1,0,ds_grad);
  
          for (k = firsti; k<=lasti; k++)
          { ds_p[k] -= sf * b[1] * ds_stepsize[k] * ds_grad[k];
            ds_q[k] += sf * a[1] * ds_stepsize[k] * ds_p[k];
          }
  
          mc_app_energy(ds,1,1,0,ds_grad);
  
          for (k = firsti; k<=lasti; k++)
          { ds_p[k] -= sf * b[2] * ds_stepsize[k] * ds_grad[k];
            ds_q[k] += sf * a[2] * ds_stepsize[k] * ds_p[k];
          }
  
          mc_app_energy(ds,1,1,0,ds_grad);
  
          for (k = firsti; k<=lasti; k++)
          { ds_p[k] -= sf * b[3] * ds_stepsize[k] * ds_grad[k];
            ds_q[k] += sf * a[3] * ds_stepsize[k] * ds_p[k];
          }

          ds->know_grad = 0;
          ds->know_pot  = 0;
        }

        if (tj->rev_sym!=0)
        {
          for (k = firsti; k<=lasti; k++)
          { ds_q[k] += sf * a[3] * ds_stepsize[k] * ds_p[k]; 
          }

          mc_app_energy(ds,1,1,0,ds_grad);

          for (k = firsti; k<=lasti; k++)
          { ds_p[k] -= sf * b[3] * ds_stepsize[k] * ds_grad[k];
            ds_q[k] += sf * a[2] * ds_stepsize[k] * ds_p[k];
          }

          mc_app_energy(ds,1,1,0,ds_grad);

          for (k = firsti; k<=lasti; k++)
          { ds_p[k] -= sf * b[2] * ds_stepsize[k] * ds_grad[k];
            ds_q[k] += sf * a[1] * ds_stepsize[k] * ds_p[k];
          }

          mc_app_energy(ds,1,1,0,ds_grad);

          for (k = firsti; k<=lasti; k++)
          { ds_p[k] -= sf * b[1] * ds_stepsize[k] * ds_grad[k];
            ds_q[k] += sf * a[0] * ds_stepsize[k] * ds_p[k];
          }

          mc_app_energy (ds, 1, 1, 
                         need_pot && n==1 ? &ds->pot_energy : 0,
                         ds_grad);

          for (k = firsti; k<=lasti; k++)
          { ds_p[k] -= sf * b[0] * ds_stepsize[k] * ds_grad[k];
          }

          ds->know_grad = 1;
          ds->know_pot = need_pot && n==1;
        }

        n -= 1;
      }

      break;
    }

    case 'L':
    { 
      if (tj->halfp)
      {
        /* Compute trajectory with initial and final half-steps for momentum .*/

        if (ds->know_grad!=approx_order[0])
        { mc_app_energy (ds, na, approx_order[0], 0, ds_grad);
        }
  
        for (k = firsti; k<=lasti; k++)
        { ds_p[k] -= sfh * ds_stepsize[k] * ds_grad[k];
        }

        a = 0;
  
        for (;;)
        {
          for (k = firsti; k<=lasti; k++)
          { ds_q[k] += sf * ds_stepsize[k] * ds_p[k];
          }
  
          a = (a+x+ta) % ta;
          o = a<na ? a : a==na ? 0 : 2*na-a;

          mc_app_energy (ds, na, approx_order[o], 
                         need_pot && n==1 ? &ds->pot_energy : 0, 
                         ds_grad);
  
          n -= 1;
          if (n==0) break;
  
          for (k = firsti; k<=lasti; k++)
          { ds_p[k] -= sf * ds_stepsize[k] * ds_grad[k];
          }
        }

        if (a!=0) abort();
  
        for (k = firsti; k<=lasti; k++)
        { ds_p[k] -= sfh * ds_stepsize[k] * ds_grad[k];
        }

        ds->know_grad = approx_order[0];
        ds->know_pot  = need_pot;
      }
      else 
      {
        /* Compute trajectory with initial and final half-steps for position. */

        for (k = firsti; k<=lasti; k++)
        { ds_q[k] += sfh * ds_stepsize[k] * ds_p[k];
        }

        a = x==-1 ? 0 : ta-1;
  
        for (;;)
        {
          a = (a+x+ta) % ta;
          o = a<na ? a : 2*na-a-1;

          mc_app_energy (ds, na, approx_order[o], 0, ds_grad);

          for (k = firsti; k<=lasti; k++)
          { ds_p[k] -= sf * ds_stepsize[k] * ds_grad[k];
          } 
  
          n -= 1;
          if (n==0) break;

          for (k = firsti; k<=lasti; k++)
          { ds_q[k] += sf * ds_stepsize[k] * ds_p[k];
          }
        }

        if (a != (x==-1 ? 0 : ta-1)) abort();
  
        for (k = firsti; k<=lasti; k++)
        { ds_q[k] += sfh * ds_stepsize[k] * ds_p[k];
        }

        ds->know_grad = 0;
        ds->know_pot  = 0;
      }
  
      break;
    }
  }

  /* Since momentum has changed, we don't know kinetic energy anymore. */

  ds->know_kinetic = 0;
}


/* TERMOSTATED DYNAMICS.  Currently disabled. 

   therm-dynamic steps [ stepsize-adjust[:stepsize-alpha] ] 

       Follow a thermostated dynamical trajectory for the given number 
       of steps, always accepting the result (hence does not leave 
       distribution exactly invariant).  The extra thermostat variables 
       included in the dynamics cause the dynamics to explore the 
       canonical distribution for the ordinary variables, rather than a 
       microcanonical distribution as for the plain "dynamic" operation
       (when used alone).
*/

#if 0

#define S 0.5
#define K 100

#if 0
#define f(x) (S*(x))
#define Df(x) (S)
#endif

#if 1
#define f(x) (S * sin((x)+1))
#define Df(x) (S * cos((x)+1))
#endif

#define sqrt2pi 2.5066282746310005024

static void therm_step (double *q, double *p, double dt)
{
  extern double phi(double), Phi(double), Phi_inverse(double);
  double q0, t;

  if (*q<0) 
  { dt = -dt;
    *q = - *q;
  }
  
  q0 = *q * exp (*p * *p / 2);
  t = q0 * sqrt2pi * (0.5 - Phi(*p));
  
  t += dt;

  if (t>q0*sqrt2pi/2)
  { t = q0*sqrt2pi - t;
    dt = -dt;
  }
  else if (t<-q0*sqrt2pi/2)
  { t = -q0*sqrt2pi - t;
    dt = -dt;
  }

  *p = Phi_inverse (0.5 - t / (sqrt2pi * q0));
  *q = q0 * exp ( - *p * *p / 2);

  if (dt<0)
  { *q = -*q;
  }

}

void mc_therm_trajectory
( mc_dynamic_state *ds,	/* Dynamical state to update */
  int n,		/* Number of steps to compute */
  int need_pot		/* Need potential energy for last state? */
)
{
  double sf, sfh, d, dd;
  int k, a, x, o, i;

  if (n==0) return;

  sf = it->stepsize_factor;

  x = +1; 

  if (n<0)
  { n = -n;
    sf = -sf;
    x = -1;
  }

  sfh = sf / 2;

  switch (tj->type)
  { 
    case 'L':
    { 
      if (tj->halfp)
      {
        if (ds->know_grad!=1)
        { mc_app_energy (ds, 1, 1, 0, ds->grad);
        }

        for ( ; n>0; n--)
        {  
          for (k = firsti; k<=lasti; k++)
          { ds->p[k] -= sfh * ds->stepsize[k] * ds->grad[k];
          }
  
          for (k = firsti; k<=lasti; k++)
          { ds->q[k] += sf * ds->stepsize[k] * ds->p[k];
          }

          mc_app_energy (ds, 1, 1, 0, ds->grad);

          for (i = 0; i<K; i++)
          {
            dd = f(ds->therm_state->tp);
            therm_step (&ds->therm_state->tq, &ds->therm_state->tp, sfh/K);
            dd -= f(ds->therm_state->tp);
            for (k = firsti; k<=lasti; k++) ds->p[k] += dd;

            dd = Df(ds->therm_state->tp);
            for (k = firsti; k<=lasti; k++)
            { ds->therm_state->tq -= sf * ds->p[k] * dd / K; 
            }

            dd = f(ds->therm_state->tp);
            therm_step (&ds->therm_state->tq, &ds->therm_state->tp, sfh/K);
            dd -= f(ds->therm_state->tp);
            for (k = firsti; k<=lasti; k++) ds->p[k] += dd;
          }

          for (k = firsti; k<=lasti; k++)
          { ds->p[k] -= sfh * ds->stepsize[k] * ds->grad[k];
          }
        }

        ds->know_grad = 0;
        ds->know_pot  = 0;
      }
      else  
      { abort();
      }
  
      break;
    }

    default:
    { abort();
    }
  }

  /* Since momentum has changed, we don't know kinetic energy anymore. */

  ds->know_kinetic = 0;
}

#endif
