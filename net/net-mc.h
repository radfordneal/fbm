/* NET-MC.H - Variables or neural netowrk Monte Carlo and gradient descent. */

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


/* VARIABLES MEANT TO BE USED ONLY INTERNALLY IN NET-MC.C AND NET-GD.C. */

EXTERN int initialize_done;	/* Has all been set up?  (Will be 0 at start) */

EXTERN net_arch *arch;		/* Network architecture */
EXTERN net_precomputed pre;	/* Precomputed stuff about the architecture */
EXTERN net_flags *flgs;		/* Network flags, null if none */
EXTERN net_priors *priors;	/* Network priors */

EXTERN model_specification *model; /* Data model */
EXTERN model_survival *surv;	/* Hazard type for survival model */

EXTERN net_sigmas sigmas;	/* Hyperparameters for network, auxiliary state
				   for Monte Carlo.  Includes noise std. dev. */
EXTERN net_sigma *noise;	/* Just the noise hyperparameters in sigmas */

EXTERN net_params params;	/* Pointers to parameters, which are position
				   coordinates for dynamical Monte Carlo */

EXTERN net_values *deriv;	/* Derivatives for training cases */

EXTERN net_flags zero_flags;	/* Flags structure that's all zeros */
