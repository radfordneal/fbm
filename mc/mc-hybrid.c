/* MC-HYBRID.C - Procedure for performing Hybrid Monte Carlo updates. */

/* Copyright (c) 1995, 1996 by Radford M. Neal 
 *
 * Permission is granted for anyone to copy, use, or modify this program 
 * for purposes of research or education, provided this copyright notice 
 * is retained, and note is made of any changes that have been made. 
 *
 * This program is distributed without any warranty, express or implied.
 * As this program was written for research purposes only, it has not been
 * tested to the degree that would be advisable in any important application.
 * All use of this program is entirely at the user's own risk.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "misc.h"
#include "rand.h"
#include "log.h"
#include "mc.h"
  

/* PERFORM HYBRID MONTE CARLO OPERATION. */

void mc_hybrid
( mc_dynamic_state *ds,	/* State to update */
  mc_iter *it,		/* Description of this iteration */
  mc_traj *tj,		/* Trajectory specification */
  int steps,		/* Total number of steps to do */
  int window,		/* Window size to use */
  int jump,		/* Number of steps in each jump */
  double temper_factor,	/* Temper factor, zero if not tempering */
  int N_quantities,	/* Number of quantities to plot, -1 for tt plot */
  mc_value *q_save,	/* Place to save original q */
  mc_value *p_save,	/* Place to save original p */
  mc_value *q_asv,	/* Place to save q values for accept state */
  mc_value *p_asv,	/* Place to save p values for accept state */
  mc_value *q_rsv,	/* Place to save q values for reject state */
  mc_value *p_rsv	/* Place to save p values for reject state */
)
{ 
  double rej_pot_energy, rej_kinetic_energy, rej_free_energy;
  double acc_pot_energy, acc_kinetic_energy, acc_free_energy;
  double old_pot_energy, old_kinetic_energy;

  int have_rej, rej_point;
  int have_acc, acc_point;

  int n, k, dir, jmps;
  double H;

  if (steps%jump!=0) abort();

  jmps = steps/jump;

  /* Decide on window offset, and save start state if we'll have to restore. */

  it->window_offset = rand_int(window);

  if (it->window_offset>0)  
  {
    mc_value_copy (q_save, ds->q, ds->dim);
    mc_value_copy (p_save, ds->p, ds->dim);

    if (!ds->know_pot)
    { mc_app_energy (ds, 1, 1, &ds->pot_energy, 0);
      ds->know_pot = 1;
    }
  
    if (!ds->know_kinetic)
    { ds->kinetic_energy = mc_kinetic_energy(ds);
      ds->know_kinetic = 1;
    }

    old_pot_energy = ds->pot_energy;
    old_kinetic_energy = ds->kinetic_energy;
  }

  /* Set up for travelling about trajectory. */

  mc_traj_init(tj,it);
  mc_traj_permute();

  have_rej = have_acc = 0;
  n = it->window_offset;
  dir = -1;

  /* Loop that looks at one new state each iteration, which may possibly
     be in the accept and/or reject windows. */

  if (temper_factor!=0 && N_quantities<0)
  { printf("\n");
  }

  while (dir==-1 || n!=jmps)
  {
    /* Restore if next state should be original start state. */

    if (dir==-1 && n==0)
    { 
      if (it->window_offset>0)
      { 
        mc_value_copy (ds->q, q_save, ds->dim);
        mc_value_copy (ds->p, p_save, ds->dim);
 
        ds->pot_energy = old_pot_energy;
        ds->kinetic_energy = old_kinetic_energy;

        ds->know_pot     = 1;
        ds->know_kinetic = 1;
        ds->know_grad    = 0;
      }

      n = it->window_offset;
      dir = 1;
    }

    else 
    { 
      /* Do tempering in second half if appropriate. */

      if (temper_factor!=0 && n<=jmps-window && 2*n>jmps)
      { for (k = 0; k<ds->dim; k++)
        { ds->p[k] /= temper_factor;
        } 
        if (N_quantities<0)
        { printf ("%6d %20.10lf\n", jmps-n+1, 
                   mc_kinetic_energy(ds) * (temper_factor*temper_factor - 1));
          printf ("%6d %20.10lf\n", jmps-n, 
                   mc_kinetic_energy(ds) * (temper_factor*temper_factor - 1));
        }
      }

      /* Next state is found by following trajectory for 'jump' steps. */

      mc_trajectory (ds, dir * jump);
      n += dir;

      /* Do tempering in first half if appropriate. */

      if (temper_factor!=0 && n>=window && 2*n<jmps)
      { 
        if (N_quantities<0)
        { printf ("%6d %20.10lf\n", n, 
                   mc_kinetic_energy(ds) * (temper_factor*temper_factor - 1));
          printf ("%6d %20.10lf\n", n+1, 
                   mc_kinetic_energy(ds) * (temper_factor*temper_factor - 1));
          if (2*(n+1)>=jmps) printf("\n");
        }
        for (k = 0; k<ds->dim; k++)
        { ds->p[k] *= temper_factor;
        } 
      }
    }

    /* Make sure we know energy, if needed. */

    if (n<window || n>jmps-window)
    {
      if (!ds->know_pot)
      { mc_app_energy (ds, 1, 1, &ds->pot_energy, 0);
        ds->know_pot = 1;
      }
   
      if (!ds->know_kinetic)
      { ds->kinetic_energy = mc_kinetic_energy(ds);
        ds->know_kinetic = 1;
      }

      H = ds->pot_energy + ds->kinetic_energy;
    }

    /* Account for state in reject window.  Reject window can be ignored if
       windows consist of the entire trajectory. */
  
    if (window!=jmps+1 && n<window)
    { 
      rej_free_energy = !have_rej ? H : - addlogs (-rej_free_energy, -H);

      if (!have_rej || rand_uniform() < exp(rej_free_energy-H))
      { 
        mc_value_copy (q_rsv, ds->q, ds->dim);
        mc_value_copy (p_rsv, ds->p, ds->dim);

        rej_pot_energy = ds->pot_energy;
        rej_kinetic_energy = ds->kinetic_energy;

        rej_point = n;
        have_rej = 1;
      }
    }

    /* Account for state in the accept window. */

    if (n>jmps-window)
    {
      acc_free_energy = !have_acc ? H : - addlogs (-acc_free_energy, -H);

      if (!have_acc || rand_uniform() < exp(acc_free_energy-H))
      { 
        if (n!=jmps)
        { mc_value_copy (q_asv, ds->q, ds->dim);
          mc_value_copy (p_asv, ds->p, ds->dim);
        }
  
        acc_pot_energy = ds->pot_energy;
        acc_kinetic_energy = ds->kinetic_energy;
  
        acc_point = n;
        have_acc = 1;
      }
    }

  }

  /* Take new state from the appropriate window. */

  if (!have_acc || !have_rej) abort();

  it->proposals += 1;
  it->delta = acc_free_energy - rej_free_energy;

  if (it->delta<=0 || rand_uniform() < exp(-it->delta/it->temperature))
  { 
    it->move_point = jump * (acc_point - it->window_offset);

    if (acc_point!=jmps)
    { mc_value_copy (ds->q, q_asv, ds->dim);
      mc_value_copy (ds->p, p_asv, ds->dim);
      ds->pot_energy     = acc_pot_energy;
      ds->kinetic_energy = acc_kinetic_energy;
      ds->know_pot     = 1;
      ds->know_kinetic = 1;
      ds->know_grad    = 0;
    }

    for (k = 0; k<ds->dim; k++)
    { ds->p[k] = -ds->p[k];
    }
  }
  else
  { 
    it->rejects += 1;
    it->move_point = jump * (rej_point - it->window_offset);

    mc_value_copy (ds->q, q_rsv, ds->dim);
    mc_value_copy (ds->p, p_rsv, ds->dim);

    ds->pot_energy     = rej_pot_energy;
    ds->kinetic_energy = rej_kinetic_energy;
    ds->know_pot     = 1;
    ds->know_kinetic = 1;
    ds->know_grad    = 0;
  }
}
