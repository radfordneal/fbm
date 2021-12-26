/* NET.H - Interface to neural network modules. */

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


/* TRICKS FOR GPU COMPILATION. */

#ifndef HOSTDEV
#define HOSTDEV  /* don't declare some procedures as __host__ or __device__ */
#define __device__
#endif


/* SETUP TO ALLOW SOURCE INCLUSION.  Used to include net-func.c, net-model.c,
   and net-back-grad.c in net-mc.c, with functions in those files declared 
   static.  (They may still be compiled on their own for other uses.)  

   This may allow the compilers to generate better code. */

#ifdef GPU_SRC_INCLUDE
#define STATIC_IF_INCLUDED static
#else
#define STATIC_IF_INCLUDED
#endif


/* VARIOUS ADJUSTABLE CONSTANTS.  Note that FP32 and FP64, specifying precision
   must be set by the build process. */

#define SPARSE_THRESHOLD 0.3 /* Do sparse handling for training inputs if 
                                fraction of zero inputs is greater than this */

#define TYPICAL_VALUES_ALL_ONE 0   /* Set to 1 to disable simulation of squared
                                      hidden unit values, thereby reverting to 
                                      the old stepsize heuristic */

#define USE_TRANSPOSED_WEIGHTS 1   /* Should transposed weight matrices be
                                      computed and used in the backward pass? */

/* Should these weights have a transposed form?  Adjustable criterion based on 
   the numbers of sourse and destination units. */

#define TRANS_WEIGHTS(ns,nd) (USE_TRANSPOSED_WEIGHTS && 1) /* always, for now */

/* Alignment for value structures. */

#define NET_VALUE_ALIGN_BYTES 16   /* Alignment for value arrays, in bytes
                                       - must be power of two, minimum of 8 */

#define NET_VALUE_ALIGN_ELEMENTS (NET_VALUE_ALIGN_BYTES / 4 / (1+FP64))


/* HOW GPU COMPUTATIONS ARE SPLIT INTO KERNELS.  The computation
   consists of four parts: forward pass, model/energy evaluation,
   combined backward pass & gradient computation, and gradient
   reduction (last two not needed if gradient not needed).  The
   setting of SPLIT_KERNELS controls whether they are all done as
   separate kernels (useful for profiling how long they take), or all
   done as one kernel (minimizing launch overhead). */

#define SPLIT_KERNELS 0   /* 0 = one kernel for all four parts
                             1 = four kernels for the four parts */


/* CONSTANTS RELATING TO GPU COMPUTATIONS: */

#define THREADS_PER_CASE 8   /* Number of GPU threads used per training case,
                                must be a power of two, max 32 (warpsize) */

#define GROUP_SHIFT 2        /* Log2 of number of training cases in a group for
                                computing gradients, must be 0, 1, or 2 */

#define GROUP_SIZE (1<<GROUP_SHIFT)  /* Number of cases in a gradient group */
#define GROUP_MASK (GROUP_SIZE-1)

#define BLKCASES 32          /* Number of training cases per thread block,
                                must be a multiple of GROUP_SIZE */

#define THREADS_PER_BLOCK (THREADS_PER_CASE*BLKCASES)
#define THREADS_PER_GROUP (THREADS_PER_CASE*GROUP_SIZE)

#define NTH THREADS_PER_CASE /* abbreviations */
#define GTH THREADS_PER_GROUP
#define BTH THREADS_PER_BLOCK

#define MIN_WARPS_PER_SM 16  /* Desired minimum number of warps per SM */
#define MIN_BLOCKS_PER_SM 2  /* Desired minimum number of blocks per SM */

#define GRAD_ALIGN_BYTES 64  /* Alignment for gradient blocks in GPU, bytes
                                  - must be a power of two, minimum of 8 */

#define GRAD_ALIGN_ELEMENTS (GRAD_ALIGN_BYTES / 4 / (1+FP64))

#define PIN_MEMORY 2         /* 0 = no host memory is pinned, 
                                1 = parameters going to gpu only,
                                2 = parameters + energy & deriv from gpu */

#define SYNC_AFTER (NTH>1 && 1) /* Should __syncwarp be called after a code
                                   section that could lead to threads diverging?
                                   Note: for performance, not correctness. */

#define DEFAULT_MAXBLKS	500  /* Default, if not set by MAXBLKS env var */

#define USE_FAST_SHARED_MEM 1  /* Use fast shared GPU memory for unit values
                                  and derivatives? */

#define AVOID_BANK_CONFLICTS 1 /* Increase size of shared memory per case if
                                  necessary to avoid bank conflicts? */

#define GPU_CACHE_PREFERENCE \
 (!USE_FAST_SHARED_MEM || SPLIT_KERNELS \
   ? cudaFuncCachePreferL1      /* L1 is better if shared memory isn't used */ \
   : cudaFuncCachePreferShared) /* Might be better as Shared, or Equal, or L1 */


/* FUNCTIONS WITH SPECIFIED PRECISION. */

#if FP32
# define prec_exp expf
# define prec_log logf
# define prec_sin sinf
# define prec_cos cosf
# define prec_tanh tanhf
# define prec_fabs fabsf
# define prec_copysign copysignf
#else
# define prec_exp exp
# define prec_log log
# define prec_sin sin
# define prec_cos cos
# define prec_tanh tanh
# define prec_fabs fabs
# define prec_copysign copysign
#endif


/* WEIGHT CONFIGURATION.  Records network weight configuration set up for a 
   group of connections with a cfg-b, cfg-i, or cfg-h flag.  Note that
   the stored indexes start at zero, though they are 1-based in the file. 

   There are also derived arrays of connections for faster computation. 

   Note that net_config_to_gpu may need to be updated when net_config 
   changes.
 */

#define Max_conn 1000000	/* Maximum number of connections in a group */

/* A single connection, or a sequence of connections, depending on context.
   For biases, s is always 0. */

typedef struct
{ unsigned short s;		/* Index of source unit(s), from 0, 0 for bias*/
  unsigned short d; 		/* Index of destination unit(s), from 0 */
  int w; 			/* Index of weight(s), from 0, negative used */
} net_connection;		/*   to indicate end of array */

/* Options for how connections are sorted and grouped, for performance. */

#define CONFIG_ORIGINAL 0	/* Use original weight configuration array  */
				/*  --- meant only for testing, not for GPU */

#define CONFIG_QUAD_S_4D_4W	(!CONFIG_ORIGINAL && 1)
#define CONFIG_SINGLE4		(!CONFIG_ORIGINAL && 1)

#define MAKE_QUAD_PAIRS 1	/* Make quad_s_4d_4w_2 versions with pairs? */

#define CONFIG_OCT_GPU_S_8D_8W 0   /* Make oct_s_8d_8w_dgpu groups? */
#define CONFIG_QUAD_GPU_S_4D_4W	1  /* Make quad_s_4d_4w_?gpu groups? */
#define MAKE_QUAD_GPU_PAIRS 1	/* Make quad_s_4d_4w_2_?gpu ver with pairs? */
#define MAKE_OTHER_GPU_PAIRS 1	/* Make other_2_?gpu versions with pairs? */

/* Set of connections between layers. */

typedef struct
{ int N_wts;			/* Number of weights */
  int N_conn;			/* Number of connections */

  net_connection *conn;		/* Array of connections, in original order */

  /* For CPU computions */
  net_connection *single;	/* Single connections, taken one-at-a-time */
  net_connection *single4_s;	/* Single connections, in groups of 4, same s */
  net_connection *single4_d;	/* Single connections, in groups of 4, same d */
  net_connection *quad_s_4d_4w;	/* Four connections, same s, sequential d & w */
  net_connection *quad_s_4d_4w_2; /* Pairs of connections, with same s, 
                                     sequential d & w, w same for pair */
  net_connection *all;		/* Pointer to block with items above */
  int all_length;		/* Length of 'all' block in use */

  /* For GPU forward computations: */

  net_connection *oct_s_8d_8w_dgpu;  /* Eight connections, same s, sequential
                                        d & w, sorted by d, grouped d mod NTH*/
  net_connection *quad_s_4d_4w_dgpu;  /* Four connections, same s, sequential
                                         d & w, sorted by d, grouped d mod NTH*/
  net_connection *other_dgpu;	/* Other connections for dest, has NTH -1s */
  int start_other_dgpu[NTH];	/* Start indexes for sections in other_dgpu */

  /* For GPU backward computations: */

  net_connection *quad_s_4d_4w_sgpu;  /* Four connections, same s, sequential
                                         d & w, sorted by s, grouped s mod NTH*/
  int start_quad_sgpu[NTH];	/* Start indexes for sections in quad...sgpu */
  net_connection *other_sgpu;	/* Other connections for dource, has NTH -1s */
  int start_other_sgpu[NTH];	/* Start indexes for sections in other_sgpu */

  /* For GPU gradient computations: */

  net_connection *quad_s_4d_4w_wgpu;  /* GPU grad version, GTH -1 terminators*/
  int start_quad_wgpu[GTH];	/* Starts for sections in quad_s_4d_4w_wgpu */
  net_connection *quad_s_4d_4w_2_wgpu;/* GPU grad version, GTH -1 terminators*/
  int start_quad_2_wgpu[GTH];	/* Starts for sections in quad_s_4d_4w_2_wgpu */
  net_connection *other_wgpu;	/* Other connections for grad, has GTH -1s */
  int start_other_wgpu[GTH];	/* Start indexes for sections in other_wgpu */
  net_connection *other_2_wgpu;	/* Pairs of other connections, same w for pair*/
  int start_other_2_wgpu[GTH];	/* Start indexes for sections in other_2_wgpu */

  net_connection *all_gpu;	/* Pointer to block with items above */
  int all_gpu_length;		/* Length of 'all_gpu' block in use */

} net_config;


/* NETWORK ARCHITECTURE.  Defines the dimensions of the input and output, the
   number of hidden layers, and the number of units in each hidden layer. 
   Also indicates which groups of network parameters the network contains,
   and the data model used (if any). 

   Stored in log files under type 'A'.  Changes may invalidate old log files. */

#define Max_layers 15  /* Maximum number of hidden layers in a network - no more
                          than 15, due to keeping flags in an unsigned short */

#define Max_nonseq 16  /* Maximum number of non-sequential connections between
                          hidden layers */

typedef struct
{ 
  int N_inputs;			/* Number of input units */
  int N_layers;			/* Number of layers of hidden units */
  int N_hidden[Max_layers];	/* Number of hidden units in each layer */
  int N_outputs;		/* Number of output units */

  int has_ti;			/* Does net contain offsets for input units? */
  int has_hh[Max_layers-1];	/* ... hidden to hidden weights? */
  int has_ih[Max_layers];	/* ... input to hidden weights? */
  int has_bh[Max_layers];	/* ... biases for hidden units? */
  int has_th[Max_layers];	/* ... offsets for hidden units? */
  int has_ho[Max_layers];	/* ... hidden to output weights? */
  int has_io;			/* ... input to output weights? */
  int has_bo;			/* ... biases for output units? */

  int has_ah[Max_layers];	/* Do hidden layers have adjustments? */
  int has_ao;			/* Does output layer have adjustments? */

  unsigned short has_nsq[Max_layers]; /* Bit vectors specifying which earlier
                                         hidden layers have a non-sequential 
                                         connection to this one.  (Note that 
                                         nsq[0] nd nsq[1] must be all 0.) */

  net_config *input_config[Max_layers+1]; /* Pointers used during program,    */
  net_config *bias_config[Max_layers+1];  /*  but set to zero in log file.    */
  net_config *hidden_config[2*Max_layers];/*  In hidden_config, 0 is unused,  */
                                          /*  has hh first, then ho (reversed)*/

  net_config *nonseq_config[Max_nonseq];  /* Pointers used in program, in 
                                             same order as net_prior nsq */

} net_arch;


/* FLAGS MODIFYING ARCHITECTURE.  This record records extra flags modifying
   the architecture, which are recorded here to avoid taking up space
   when the flags aren't used. 

   The omit flags are 1 when an input is omitted for a layer.  The low-order
   bit pertains to the output, with bits above that pertaining to successive
   hidden layers.  The any_omitted array indicates in element a->N_layers 
   whether any inputs are omitted for the output, and in element l whether 
   any inputs are omitted for hidden layer l.

   Stored in log files under type 'F', but may be omitted if all the flags
   are zero.  Changes may invalidate old log files. */

#define Tanh_type 0		/* Tanh units */
#define Identity_type 1		/* Identity units */
#define Softplus_type 3		/* Softplus units */

typedef struct
{
  unsigned short omit[Max_inputs]; /* Whether inputs omitted, for each layer */
  char any_omitted[Max_layers+1];  /* Whether any inputs omitted for layer */

  char layer_type[Max_layers];     /* Type of hidden units in layer */

                                   /* Below with +1 for output layer, at end */
  short input_config[Max_layers+1]; /* Index of input config file, 0 if none */
  short bias_config[Max_layers+1];  /* Index of bias config file, 0 if none */

  short hidden_config[2*Max_layers];/* Index of hidden config file, 0 if none */
                                    /*   has hh first, then ho (reversed) */

  short nonseq_config[Max_nonseq]; /* Index of non-sequential config file name,
                                      0 if none, order as in net_priors nsq */

  char config_files[2000];         /* Names of files for input/hidden configs */

} net_flags;


/* NETWORK PRIORS.  Defines the priors to be used for various groups of 
   network parameters, and, in the case of a regression model, for the
   noise levels.  The general hierarchical scheme is used (see prior.h)
   for priors on weights, except that the priors for the "adjustments" of 
   the distributions of weights and biases going into a unit are specified 
   by giving single alpha values. 

   A record of type net_priors is stored in log files under type 'P'.  
   Changes may invalidate old log files. */

typedef struct
{ 
  prior_spec ti;		/* Prior for offsets of input units */
  prior_spec hh[Max_layers-1];	/* Priors for hidden to (next) hidden weights */
  prior_spec ih[Max_layers];	/* Priors for input to hidden weights */
  prior_spec bh[Max_layers];	/* Priors for biases of hidden units */
  prior_spec th[Max_layers];	/* Priors for offsets of hidden units */
  prior_spec ho[Max_layers];	/* Priors for hidden to output weights */
  prior_spec io;		/* Prior for input to output weights */
  prior_spec bo;		/* Prior for biases of output units */

  prior_spec nsq[Max_nonseq];	/* Priors for weights for non-sequential 
				   hidden-to-hidden connections, in order by
				   destination layer and then source layer,
				   by what exists according to has_nsq */

  double ah[Max_layers];	/* Alphas for adjustments of hidden units */
  double ao;			/* Alpha for adjustments of output units */

} net_priors;


/* NETWORK SIGMAS.  This structure stores pointers to the standard deviations 
   (sigmas) associated with the various types of network parameters, and, in 
   the case of weights (but not biases and offsets), the sigmas one level down, 
   associated with particular units.  The sigmas associated with particular 
   weights are not stored.  The array pointers are null when the corresponding 
   parameters do not exist in the network.  All the 'xx_cm' fields point to
   single values; they are referenced indirectly to allow allow all the sigma 
   values to be stored in a contiguous block. 

   Stored in log files under type 'S'.  Changes may invalidate old log files. */

#ifndef net_sigma         /* May be defined by a compiler option */
typedef double net_sigma;   /* Precision of sigma values */
#endif

typedef struct
{ 
  unsigned total_sigmas;	/* Total number of sigma values */
  net_sigma *sigma_block;	/* Block of all sigma values */

  net_sigma *ti_cm;		/* Pointer to common sigma for input offsets */
  net_sigma *hh_cm[Max_layers-1];/*... for hidden to hidden weights */
  net_sigma *ih_cm[Max_layers];	/* ... for input to hidden weights */
  net_sigma *bh_cm[Max_layers];	/* ... for biases of hidden units */
  net_sigma *th_cm[Max_layers];	/* ... for offsets of hidden units */
  net_sigma *ho_cm[Max_layers];	/* ... for hidden to output weights */
  net_sigma *io_cm;		/* ... for input to output weights */
  net_sigma *bo_cm;		/* ... for biases of output units */

  net_sigma *hh[Max_layers-1];	/* Points to sigmas for hidden-hidden weights */
  net_sigma *ih[Max_layers];	/* ... for input-hidden weights*/
  net_sigma *ho[Max_layers];	/* ... for hidden to output weights */
  net_sigma *io;		/* ... for input to output weights */

  net_sigma *nsq_cm[Max_nonseq];/* Points to common sigmas nonseq weights*/
  net_sigma *nsq[Max_nonseq];	/* Points to sigmas for non-sequential weights*/

  net_sigma *ah[Max_layers];	/* Pointers to adjustments for hidden units */
  net_sigma *ao;		/* ... for output units */

  net_sigma *noise_cm;		/* Pointer to common sigma for all outputs */
  net_sigma *noise;		/* Pointer to sigmas for each output */

} net_sigmas;


/* NETWORK PARAMETERS.  Network weights, biases, and offsets are stored 
   in arrays pointed to by the following structure, arranged first by source
   unit, then by destination unit. For example, the weight from input unit 
   i to unit j of hidden layer l is ih[l][N_hidden[l]*i + j], if no weight
   configuration file is used.  The array pointers are null when the 
   corresponding parameters do not exist in the network.  Structures of 
   the same type are also used for other data associated with parameters, 
   such as components of the "error" gradient.

   Stored in log files under type 'W'.  Changes may invalidate old log files. */

#ifndef net_param       /* May be defined by a compiler option */
typedef double net_param;  /* Precision of weights, baises, and offsets */
#endif

typedef struct
{ 
  unsigned total_params;	/* Total number of parameter values */
  net_param *param_block;	/* Block of all parameters values */

  net_param *ti;		/* Offsets of input units */
  net_param *hh[Max_layers-1];	/* Hidden to hidden weights */
  net_param *ih[Max_layers];	/* Input to hidden weights */
  net_param *bh[Max_layers];	/* Biases of hidden units */
  net_param *th[Max_layers];	/* Offsets of hidden units */
  net_param *ho[Max_layers];	/* Hidden to output weights */
  net_param *io;		/* Input to output weights */
  net_param *bo;		/* Biases of output units */

  net_param *nsq[Max_nonseq];	/* Weights for non-sequential connections,
				   ordered by destination, then by source */
} net_params;


/* NETWORK VALUES.  Structures of the following type contain pointers to
   arrays of values for units in the network.  Structures of the same type
   are also used for other data associated with units, such as derivatives 
   of the "error" for a case with respect to unit values.  The value of an
   input or hidden unit does not include the offset; instead this is added 
   in whenever the value is used. */

#ifndef net_value       /* May be defined by a compiler option */
typedef double net_value;  /* Precision of unit values */
#endif

typedef struct
{ 
  net_value *i;			/* Values of input units */
  net_value *h[Max_layers];	/* Values of hidden units */ 
  net_value *o;			/* Values of output units */

} net_values;


/* STUFF ABOUT THE NETWORK ARCHITECTURE THAT'S BEEN PRE-COMPUTED. */

typedef struct
{ 
  short memused;                    /* Amount of fast GPU shared memory used
                                       (in net_value units) */
  short fwgpumem[Max_layers];       /* Offset to forward hidden layer in fast
                                       GPU shared mem, or -1 if in global mem */
  short bwgpumem[Max_layers];       /* Offset to backward hidden layer in fast
                                       GPU shared mem, or -1 if in global mem */

  signed char nonseq [Max_layers]   /* nonseq[from][to] indexes non-sequential*/
                     [Max_layers+1];/*  connection in nsq, or is -1 if none.  */
                                    /*  (The +1 makes power of two for speed) */
} net_precomputed;


/* LOCATE UNIT VALUES / DERIVATIVES, IN FAST SHARED OR IN REGULAR GPU MEMORY.

   The fw_hidden_loc and bw_hidden_loc function give locations of
   hidden unit values or derivatives in layer 'l' for the case handled
   by this thread, given a pointer to the value structure for this case.

   The fw_hidden_loc_grad and bs_hidden_loc functions give locations
   of hidden unit values or derivatives in layer 'l' for case 'w' in
   the group handled by this thread, given a pointer to the value
   structure for the first case in this group. */

#if __CUDACC__

extern __shared__ net_value sharedvalues[];

__device__ static inline net_value *fw_hidden_loc 
  (net_precomputed const*pre, net_values const*v, int l)
{ int t;
  return !USE_FAST_SHARED_MEM || SPLIT_KERNELS || (t = pre->fwgpumem[l]) < 0 
           ? v->h[l]
           : sharedvalues + (threadIdx.x/NTH) * pre->memused + t;
}

__device__ static inline net_value *bw_hidden_loc 
  (net_precomputed const*pre, net_values const*v, int l)
{ int t;
  return !USE_FAST_SHARED_MEM || SPLIT_KERNELS || (t = pre->bwgpumem[l]) < 0
           ? v->h[l]
           : sharedvalues + (threadIdx.x/NTH) * pre->memused + t;
}

__device__ static inline net_value *fw_hidden_loc_grad 
  (net_precomputed const*pre, net_values const*v, int l, int w)
{ int t;
  return !USE_FAST_SHARED_MEM || SPLIT_KERNELS || (t = pre->fwgpumem[l]) < 0 
           ? (v+w)->h[l] 
           : sharedvalues + (w+(threadIdx.x/GTH)*GROUP_SIZE)*pre->memused + t;
}

__device__ static inline net_value *bw_hidden_loc_grad 
  (net_precomputed const*pre, net_values const*v, int l, int w)
{ int t;
  return !USE_FAST_SHARED_MEM || SPLIT_KERNELS || (t = pre->bwgpumem[l]) < 0 
           ? (v+w)->h[l] 
           : sharedvalues + (w+(threadIdx.x/GTH)*GROUP_SIZE)*pre->memused + t;
}

#endif


/* PROCEDURES. */

void net_training_cases (double *, net_params *, int, int, double, double);

#ifdef __cplusplus
extern "C" {
#endif

unsigned net_setup_sigma_count (net_arch *, net_flags *, model_specification *);
unsigned net_setup_param_count (net_arch *, net_flags *, net_precomputed *);

void net_setup_sigma_pointers (net_sigmas *, net_arch *, net_flags *, 
                               model_specification *);
void net_setup_param_pointers (net_params *, net_arch *, net_flags *);
void net_replicate_param_pointers (net_params *, net_arch *, int, unsigned);

int net_setup_hyper_group (net_arch *, net_flags *, int, int *, int *, int *);
int net_setup_param_group 
  (net_arch *, net_flags *, int, int *, int *, int *, int*);

unsigned net_setup_value_count (net_arch *);
unsigned net_setup_value_count_aligned (net_arch *, int, int, int);
void net_setup_value_pointers 
  (net_values *, net_value *, net_arch *, net_value *, net_value *);
void net_setup_value_pointers_aligned
  (net_values *, net_value *, net_arch *, int, net_value *, net_value *);

void net_prior_generate (net_params *, net_sigmas *, net_arch *, net_flags *,
                         model_specification *m, net_priors *, int, 
                         double, double);

void net_prior_prob (net_params *, net_sigmas *, double *, net_params *,
                     net_arch *, net_flags *, net_priors *, int);

void net_prior_max_second (net_params *, net_sigmas *, net_arch *, net_flags *,
                           net_priors *);

void net_print_params (net_params *, net_sigmas *, net_arch *, net_flags *,
                       model_specification *);
void net_print_sigmas (net_sigmas *, net_arch *, net_flags *,
                       model_specification *);

void net_record_sizes        (log_gobbled *);
void net_check_specs_present (net_arch *, net_priors *, int,
                              model_specification *, model_survival *);

net_config *net_config_read (char *, int, int);

STATIC_IF_INCLUDED
void net_func (net_values *restrict, net_arch const*, 
               net_precomputed const*, net_flags const*, 
               net_params const*, int);

STATIC_IF_INCLUDED
void net_model_prob (net_values const*, net_value const*, 
                     double *restrict, net_values *restrict, 
                     net_arch const*, model_specification const*,
                     model_survival const*, net_sigma const*, int);

STATIC_IF_INCLUDED 
void net_model_check (model_specification const*);

STATIC_IF_INCLUDED
void net_model_max_second (net_value *, net_arch *, model_specification *,
                           model_survival *, net_sigma *);

STATIC_IF_INCLUDED
void net_model_guess (net_values *, net_value *, net_arch *, 
                      net_precomputed *, net_flags *,
                      model_specification *, model_survival *, net_params *,
                      net_sigma *, int);

#ifdef __cplusplus
}
#endif
