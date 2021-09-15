/* MC-QUANTITIES.H - Declarations of procedures for Monte Carlo quantities. */

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

extern void mc_initialize  (log_gobbled *);
extern void mc_available   (struct quantdesc[Max_quantities], log_gobbled *);
extern void mc_evaluate    (struct quantdesc[Max_quantities], quantities_held *,
                            log_gobbled *);
