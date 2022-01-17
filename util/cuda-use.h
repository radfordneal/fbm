/* CUDA-USE.H - Possibly set up for use of CUDA. */

/* Copyright (c) 2021 by Radford M. Neal 
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

#define MANAGED_MEMORY_USED 0    /* Set to 1 if 'managed' CPU/GPU memory used */

#if __CUDACC__  /* USING CUDA */

# define restrict __restrict__

  static void check_cuda_error (cudaError_t err, const char *where)
  { if (err != cudaSuccess)
    { printf ("%s: CUDA error: %s\n", where, cudaGetErrorString(err));
      fflush(stdout);
      abort();
    }
  }

# if MANAGED_MEMORY_USED

    static void *managed_alloc (unsigned n, unsigned size)
    { size_t sz = (size_t)n*size;
      void *p;
      if (0) printf("Allocating %.0f bytes of managed memory\n",(double)sz);
      check_cuda_error (cudaMallocManaged (&p, sz), "Alloc failed");
      return p;
    }

#   define managed_free cudaFree

    static void *make_managed (void *p, unsigned sz)
    { if (p==0) return p;
      void *q = managed_alloc (1, sz);
      memcpy (q, p, sz);
      return q;
    }

# else  /* not using managed memory */

#   undef __managed__
#   define __managed__   __this_is_an_error__
#   define managed_alloc __this_is_an_error__

# endif

#else  /* NOT USING CUDA */

# define __host__
# define __device__
# define __managed__

# define __restrict__ restrict

# define managed_alloc chk_alloc
# define managed_free free

# define make_managed(p,sz) (p)

#endif


#define HOSTDEV __host__ __device__  /* For convenience */
#define STAMAN static __managed__
#define EXTMAN extern __managed__


#if __CUDACC__ && __CUDA_ARCH__  /* COMPILING FOR GPU */

# define abort abort_in_GPU

  __device__ static void abort(void)
  { printf("Call of abort() in GPU code - but continuing anyway\n");
  }

# undef USE_SIMD_INTRINSICS
# undef USE_FMA

#endif
