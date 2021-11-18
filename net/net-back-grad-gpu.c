/* NET-BACK-GRAD-GPU.C - Combined backprop / gradient computation for GPU. */

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


/* This file is included multiple times in net-back-grad.c, with CASES defined
   1, 2, 3, or 4, in order to define four versions of the combined backprop
   and gradient routine that handle from 1 to 4 training cases at once. */

#undef NET_BACK_GRAD_GPU
#undef STORE_GRAD1
#undef STORE_GRAD1_CONFIG
#undef STORE_GRAD2
#undef STORE_GRAD2_CONFIG

#if CASES==1
#  define NET_BACK_GRAD_GPU net_back_grad_gpu_1
#  define STORE_GRAD1(th,g,d0,d1,d2,d3,n) \
          net_store1_grad1(th,g,d0,n)
#  define STORE_GRAD1_CONFIG(th,g,d0,d1,d2,d3,cf) \
          net_store1_grad1_config(th,g,d0,cf)
#  define STORE_GRAD2(th,g,v0,v1,v2,v3,off,nv,d0,d1,d2,d3,nd,omit,ob,sparse) \
          net_store1_grad2(th,g,v0,off,nv,d0,nd,omit,ob,sparse)
#  define STORE_GRAD2_CONFIG(th,g,s0,s1,s2,s3,off,d0,d1,d2,d3,cf) \
          net_store1_grad2_config(th,g,s0,off,d0,cf)
#elif CASES==2
#  define NET_BACK_GRAD_GPU net_back_grad_gpu_2
#  define STORE_GRAD1(th,g,d0,d1,d2,d3,n) \
          net_store2_grad1(th,g,d0,d1,n)
#  define STORE_GRAD1_CONFIG(th,g,d0,d1,d2,d3,cf) \
          net_store2_grad1_config(th,g,d0,d1,cf)
#  define STORE_GRAD2(th,g,v0,v1,v2,v3,off,nv,d0,d1,d2,d3,nd,omit,ob,sparse) \
          net_store2_grad2(th,g,v0,v1,off,nv,d0,d1,nd,omit,ob,sparse)
#  define STORE_GRAD2_CONFIG(th,g,s0,s1,s2,s3,off,d0,d1,d2,d3,cf) \
          net_store2_grad2_config(th,g,s0,s1,off,d0,d1,cf)
#elif CASES==3
#  define NET_BACK_GRAD_GPU net_back_grad_gpu_3
#  define STORE_GRAD1(th,g,d0,d1,d2,d3,n) \
          net_store3_grad1(th,g,d0,d1,d2,n)
#  define STORE_GRAD1_CONFIG(th,g,d0,d1,d2,d3,cf) \
          net_store3_grad1_config(th,g,d0,d1,d2,cf)
#  define STORE_GRAD2(th,g,v0,v1,v2,v3,off,nv,d0,d1,d2,d3,nd,omit,ob,sparse) \
          net_store3_grad2(th,g,v0,v1,v2,off,nv,d0,d1,d2,nd,omit,ob,sparse)
#  define STORE_GRAD2_CONFIG(th,g,s0,s1,s2,s3,off,d0,d1,d2,d3,cf) \
          net_store3_grad2_config(th,g,s0,s1,s2,off,d0,d1,d2,cf)
#else /* CASES==4 */
#  define NET_BACK_GRAD_GPU net_back_grad_gpu_4
#  define STORE_GRAD1(th,g,d0,d1,d2,d3,n) \
          net_store4_grad1(th,g,d0,d1,d2,d3,n)
#  define STORE_GRAD1_CONFIG(th,g,d0,d1,d2,d3,cf) \
          net_store4_grad1_config(th,g,d0,d1,d2,d3,cf)
#  define STORE_GRAD2(th,g,v0,v1,v2,v3,off,nv,d0,d1,d2,d3,nd,omit,ob,sparse) \
         net_store4_grad2(th,g,v0,v1,v2,v3,off,nv,d0,d1,d2,d3,nd,omit,ob,sparse)
#  define STORE_GRAD2_CONFIG(th,g,s0,s1,s2,s3,off,d0,d1,d2,d3,cf) \
         net_store4_grad2_config(th,g,s0,s1,s2,s3,off,d0,d1,d2,d3,cf)
#endif


/* FIND GRADIENT, USING BACKPROPAGATED DERIVATIVES.  Assumes that values
   of hidden units for the training case have been computed (in v->h[.]),
   and that the derivative of the energy with respect to the outputs has
   been computed (in d->o).  Uses other parts of d to store derivatives 
   of energy with respect to hidden units, and with respect to input units,
   if input offsets are present. */

__device__ void NET_BACK_GRAD_GPU
( int thrg,		/* Which thread, from 0 to GTH-1 */
  net_params *restrict g, /* Gradient with respect to parameters to add to */
  net_values const*v0,	/* Values for units in network, first training case */
  net_values *restrict d0, /* Place for derivatives, first training case */
  net_arch const*a,	/* Network architecture */
  net_precomputed const* pre,  /* Precomputed aspects of architecture */
  net_flags const*flgs,	/* Network flags, null if none */
  net_params const*w,	/* Network parameters */
  int sparse            /* Might source unit values often be zero? */
)
{
  int l, ld, ls, nsqi, i;

# if CASES==1
# elif CASES==2
  const net_values *d1 = d0+1;
  const net_values *v1 = v0+1;
# elif CASES==3
  const net_values *d1 = d0+1, *d2 = d1+1;
  const net_values *v1 = v0+1, *v2 = v1+1;
# else /* CASES==4 */
  const net_values *d1 = d0+1, *d2 = d1+1, *d3 = d2+1;
  const net_values *v1 = v0+1, *v2 = v1+1, *v3 = v2+1;
# endif

  /* Add parts of gradients that don't depend on computing derivatives
     with respect to hidden or input unit values - only on inputs and hidden
     unit values, and on derivatives with respect to outputs, */

  if (a->has_bo)
  { if (a->bias_config[a->N_layers])
    { STORE_GRAD1_CONFIG (thrg, g->bo, d0->o, d1->o, d2->o, d3->o,
                          a->bias_config[a->N_layers]);
    }
    else
    { STORE_GRAD1 (thrg, g->bo, d0->o, d1->o, d2->o, d3->o, a->N_outputs);
    }
  }

  if (a->has_io)
  { if (a->input_config[a->N_layers])
    { STORE_GRAD2_CONFIG (thrg, g->io, 
                          v0->i, v1->i, v2->i, v3->i, 
                          a->has_ti ? w->ti : 0, 
                          d0->o, d1->o, d2->o, d3->o,
                          a->input_config[a->N_layers]);
    }
    else
    { STORE_GRAD2 (thrg, g->io, 
                   v0->i, v1->i, v2->i, v3->i, 
                   a->has_ti ? w->ti : 0, a->N_inputs,
                   d0->o, d1->o, d2->o, d3->o, a->N_outputs,
                   flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0, 1,
                   sparse);
    }
  }

  for (l = 0; l<a->N_layers; l++)
  {
    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { STORE_GRAD2_CONFIG (thrg, g->ho[l], 
                            v0->h[l], v1->h[l], v2->h[l], v3->h[l], 
                            a->has_th[l] ? w->th[l] : 0,
                            d0->o, d1->o, d2->o, d3->o, 
                            a->hidden_config[k]);
      }
      else
      { STORE_GRAD2 (thrg, g->ho[l], 
                     v0->h[l], v1->h[l], v2->h[l], v3->h[l], 
                     a->has_th[l] ? w->th[l] : 0, a->N_hidden[l], 
                     d0->o, d1->o, d2->o, d3->o, a->N_outputs, 
                     (unsigned short *) 0, 0, 0);
      }
    }
  }

  /* Find case handled by this thread for backpropagation, and index of
     this thread within that case (0 to NTH-1). */

  const net_values *d, *v;
  int thrb;
  if (thrg > CASES*NTH)
  { thrb = -1;
  }
  else
  { thrb = thrg & (NTH-1);
    d = d0 + thrg / NTH;
    v = v0 + thrg / NTH;
  }

  /* Start computation of derivatives with respect to input values, if
     they will be needed, with possible contribution from input-output
     connections. */

  if (a->has_ti && thrb>=0)
  { 
    for (i = thrb; i<a->N_inputs; i+=NTH)
    { d->i[i] = 0;
    }

    if (a->has_io)
    { if (a->input_config[a->N_layers])
      { sum_derivatives_config_gpu 
         (thrb, d->o, d->i, w->io, a->input_config[a->N_layers]);
      }
      else
      { sum_derivatives_gpu 
         (thrb, d->o, a->N_outputs, d->i, a->N_inputs, w->io,
          flgs && flgs->any_omitted[a->N_layers] ? flgs->omit : 0, 1);
      }
    }
  }

  /* Go backwards through hidden layers, computing derivatives with respect to
     hidden unit values, and then adding to gradients that depend on these. */

  for (l = a->N_layers-1; l>=0; l--)
  {
    int N_hidden = a->N_hidden[l];

    /* Place to store derivatives computed for this hidden layer. */

    net_value *restrict dh = d->h[l];

    /* Find derivatives with respect to values of units in this hidden layer. */

    for (i = thrb; i<N_hidden; i+=NTH)
    { dh[i] = 0;
    }

    if (a->has_ho[l])
    { int k = 2*a->N_layers-1-l;
      if (a->hidden_config[k])
      { sum_derivatives_config_gpu 
         (thrb, d->o, dh, w->ho[l], a->hidden_config[k]);
      }
      else
      { sum_derivatives_gpu 
          (thrb, d->o, a->N_outputs, dh, N_hidden,
           w->ho[l], (unsigned short *) 0, 0);
      }
    }

    for (ld = l+1; ld<a->N_layers; ld++)
    { int nsqi = pre->nonseq[l][ld];
      if (nsqi>=0)
      { if (a->nonseq_config[nsqi])
        { sum_derivatives_config_gpu
            (thrb, d->h[ld], dh, w->nsq[nsqi], a->nonseq_config[nsqi]);
        }
        else
        { sum_derivatives_gpu
            (thrb, d->h[ld], a->N_hidden[ld], dh, N_hidden,
             w->nsq[nsqi], (unsigned short *) 0, 0);
        }
      }
    }

    if (l<a->N_layers-1 && a->has_hh[l])
    { if (a->hidden_config[l+1])
      { sum_derivatives_config_gpu 
          (thrb, d->h[l+1], dh, w->hh[l], a->hidden_config[l+1]);
      }
      else
      { sum_derivatives_gpu 
          (thrb, d->h[l+1], a->N_hidden[l+1], dh, N_hidden,
           w->hh[l], (unsigned short *) 0, 0);
      }
    }

    /* Add to gradient with respect to hidden offsets, based on derivatives
       with respect to hidden unit values (before these are converted to
       derivatives with respect to the summed input, prior to the activation
       function). */

    if (a->has_th[l])
    { 
      __syncthreads();

      STORE_GRAD1 (thrg, g->th[l], d0->i, d1->i, d2->i, d3->i, N_hidden);
    }

    /* Pass backwards through activation function to get derivatives with 
       respect to the summed inputs of units in this hidden layer. */

    net_value const* vh = v->h[l];

    if (flgs==0 || flgs->layer_type[l]==Tanh_type)
    { for (i = thrb; i<N_hidden; i+=NTH)
      { dh[i] *= (1 - vh[i]*vh[i]);
      }
    }
    else if (flgs->layer_type[l]==Softplus_type)
    { for (i = thrb; i<N_hidden; i+=NTH)
      { net_value e = prec_exp(vh[i]);
        dh[i] *= (e-1) / e;
      }
    }
    else /* identity */
    { /* nothing to do */
    }

    __syncthreads();

    /* Add contribution from this hidden layer's derivatives to the derivatives
       with respect to inputs, if they will be needed. */

    if (a->has_ti && thrb>=0)
    { 
      if (a->has_ih[l])
      { if (a->input_config[l])
        { sum_derivatives_config_gpu (thrb, dh, d->i, w->ih[l], 
                                      a->input_config[l]);
        }
        else
        { sum_derivatives_gpu (thrb, dh, a->N_hidden[l], d->i, a->N_inputs, 
             w->ih[l], flgs && flgs->any_omitted[l]? flgs->omit : 0, 1<<(l+1));
        }
      }
    }

    /* Add to gradients that depend on the derivatives of energy with respect 
       to the inputs of units in this hidden layer. */

#   if CASES==1
    const net_value *c0 = bw_hidden_loc_grad(pre,d0,l,0);
#   elif CASES==2
    const net_value *c0 = bw_hidden_loc_grad(pre,d0,l,0);
    const net_value *c1 = bw_hidden_loc_grad(pre,d0,l,1);
#   elif CASES==3
    const net_value *c0 = bw_hidden_loc_grad(pre,d0,l,0);
    const net_value *c1 = bw_hidden_loc_grad(pre,d0,l,1);
    const net_value *c2 = bw_hidden_loc_grad(pre,d0,l,2);
#   else
    const net_value *c0 = bw_hidden_loc_grad(pre,d0,l,0);
    const net_value *c1 = bw_hidden_loc_grad(pre,d0,l,1);
    const net_value *c2 = bw_hidden_loc_grad(pre,d0,l,2);
    const net_value *c3 = bw_hidden_loc_grad(pre,d0,l,3);
#   endif

    if (a->has_bh[l])
    { if (a->bias_config[l])
      { STORE_GRAD1_CONFIG (thrg, g->bh[l], c0, c1, c2, c3, a->bias_config[l]);
      }
      else
      { STORE_GRAD1 (thrg, g->bh[l], c0, c1, c2, c3, N_hidden);
      }
    }

    if (a->has_ih[l])
    { if (a->input_config[l])
      { STORE_GRAD2_CONFIG (thrg, g->ih[l], 
                            v0->i, v1->i, v2->i, v3->i, 
                            a->has_ti ? w->ti : 0, 
                            c0, c1, c2, c3,
                            a->input_config[l]);
      }
      else
      { STORE_GRAD2 (thrg, g->ih[l], 
                     v0->i, v1->i, v2->i, v3->i,
                     a->has_ti ? w->ti : 0, a->N_inputs,
                     c0, c1, c2, c3, N_hidden,
                     flgs && flgs->any_omitted[l] ? flgs->omit : 0, 1<<(l+1),
                     sparse);
      }
    }

    if (a->has_nsq[l])
    { for (ls = 0; ls<l; ls++)
      { nsqi = pre->nonseq[ls][l];
        if (nsqi>=0)
        { 
#         if CASES==1
          net_value *u0 = fw_hidden_loc_grad(pre,v0,ls,0);
#         elif CASES==2
          net_value *u0 = fw_hidden_loc_grad(pre,v0,ls,0);
          net_value *u1 = fw_hidden_loc_grad(pre,v0,ls,1);
#         elif CASES==3
          net_value *u0 = fw_hidden_loc_grad(pre,v0,ls,0);
          net_value *u1 = fw_hidden_loc_grad(pre,v0,ls,1);
          net_value *u2 = fw_hidden_loc_grad(pre,v0,ls,2);
#         else
          net_value *u0 = fw_hidden_loc_grad(pre,v0,ls,0);
          net_value *u1 = fw_hidden_loc_grad(pre,v0,ls,1);
          net_value *u2 = fw_hidden_loc_grad(pre,v0,ls,2);
          net_value *u3 = fw_hidden_loc_grad(pre,v0,ls,3);
#         endif
          if (a->nonseq_config[nsqi])
          { STORE_GRAD2_CONFIG (thrg, g->nsq[nsqi], 
                                u0, u1, u2, u3,
                                a->has_th[ls] ? w->th[ls] : 0,
                                c0, c1, c2, c3, a->nonseq_config[nsqi]);
          }
          else
          { STORE_GRAD2 (thrg, g->nsq[nsqi], 
                         u0, u1, u2, u3,
                         a->has_th[ls] ? w->th[ls] : 0, a->N_hidden[ls], 
                         c0, c1, c2, c3, N_hidden, 
                         (unsigned short *)0, 0, 0);
          }
        }
      }
    }

    if (l>0 && a->has_hh[l-1])
    { 
#     if CASES==1
      net_value *u0 = fw_hidden_loc_grad(pre,v0,l-1,0);
#     elif CASES==2
      net_value *u0 = fw_hidden_loc_grad(pre,v0,l-1,0);
      net_value *u1 = fw_hidden_loc_grad(pre,v0,l-1,1);
#     elif CASES==3
      net_value *u0 = fw_hidden_loc_grad(pre,v0,l-1,0);
      net_value *u1 = fw_hidden_loc_grad(pre,v0,l-1,1);
      net_value *u2 = fw_hidden_loc_grad(pre,v0,l-1,2);
#     else
      net_value *u0 = fw_hidden_loc_grad(pre,v0,l-1,0);
      net_value *u1 = fw_hidden_loc_grad(pre,v0,l-1,1);
      net_value *u2 = fw_hidden_loc_grad(pre,v0,l-1,2);
      net_value *u3 = fw_hidden_loc_grad(pre,v0,l-1,3);
#     endif
      if (a->hidden_config[l])
      { STORE_GRAD2_CONFIG (thrg, g->hh[l-1], 
                            u0, u1, u2, u3,
                            a->has_th[l-1] ? w->th[l-1] : 0,
                            c0, c1, c2, c3,
                            a->hidden_config[l]);
      }
      else
      { STORE_GRAD2 (thrg, g->hh[l-1], 
                     u0, u1, u2, u3,
                     a->has_th[l-1] ? w->th[l-1] : 0, a->N_hidden[l-1], 
                     c0, c1, c2, c3, N_hidden, 
                     (unsigned short *)0, 0, 0);
      }
    }
  }

  /* Add to gradients for input offsets, now that derivatives with respect
     to inputs have been computed. */

  if (a->has_ti)
  { 
    if (a->N_layers==0)  /* otherwise done at end of loop above */
    {__syncthreads();
    }

    STORE_GRAD1 (thrg, g->ti, d0->i, d1->i, d2->i, d3->i, a->N_inputs);
  }
}
