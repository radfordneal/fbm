/* MC-METROPOLIS.C - Procedures for performing Metropolis-style updates. */

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


/* PERFORM METROPOLIS UPDATE ON ALL COMPONENTS AT ONCE. */

void mc_metropolis
( mc_dynamic_state *ds,	/* State to update */
  mc_iter *it,		/* Description of this iteration */
  mc_value *q_save	/* Place to save old q values */
)
{
  double old_energy;
  double sf;
  int k;

  if (!ds->know_pot)
  { mc_app_energy (ds, 1, 1, &ds->pot_energy, 0);
    ds->know_pot = 1;
  }

  old_energy = ds->pot_energy;

  sf = it->stepsize_factor;

  mc_value_copy (q_save, ds->q, ds->dim);

  for (k = 0; k<ds->dim; k++) 
  { ds->q[k] += sf * ds->stepsize[k] * rand_gaussian();
  }
  
  mc_app_energy (ds, 1, 1, &ds->pot_energy, 0);

  it->proposals += 1;
  it->delta = ds->pot_energy - old_energy;

  if (it->delta<=0 || rand_uniform() < exp(-it->delta/it->temperature))
  { 
    it->move_point = 1;
    ds->know_grad = 0;
  }
  else
  { 
    it->rejects += 1;
    it->move_point = 0;

    ds->pot_energy = old_energy;

    mc_value_copy (ds->q, q_save, ds->dim);
  }
}


/* PERFORM METROPOLIS UPDATE ON ONE COMPONENT AT A TIME. */

void mc_met_1
( mc_dynamic_state *ds,	/* State to update */
  mc_iter *it,		/* Description of this iteration */
  int firsti,		/* Index of first component to update (-1 for all) */
  int lasti		/* Index of last component to update */
)
{
  double old_energy, qsave;
  double sf;
  int k;

  if (firsti==-1) 
  { firsti = 0;
    lasti = ds->dim-1;
  }

  if (lasti>=ds->dim-1) lasti = ds->dim-1;
  if (firsti>lasti) firsti = lasti;

  sf = it->stepsize_factor;

  for (k = firsti; k<=lasti; k++)
  {
    if (!ds->know_pot)
    { mc_app_energy (ds, 1, 1, &ds->pot_energy, 0);
      ds->know_pot = 1;
    }
  
    old_energy = ds->pot_energy;
  
    qsave = ds->q[k];
  
    ds->q[k] += sf * ds->stepsize[k] * rand_gaussian();
    
    mc_app_energy (ds, 1, 1, &ds->pot_energy, 0);
  
    it->proposals += 1;
    it->delta = ds->pot_energy - old_energy;
  
    if (it->delta<=0 || rand_uniform() < exp(-it->delta/it->temperature))
    { 
      it->move_point = 1;
      ds->know_grad = 0;
    }
    else
    { 
      it->rejects += 1;
      it->move_point = 0;
  
      ds->pot_energy = old_energy;
  
      ds->q[k] = qsave;
    }
  }
}
