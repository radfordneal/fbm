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

#if __CUDACC__  /* USING CUDA */

# define restrict __restrict__

  static void check_cuda_error (cudaError_t err, const char *where)
  { if (err != cudaSuccess)
    { printf ("%s: CUDA error: %s\n", where, cudaGetErrorString(err));
      abort();
    }
 }

  static void *managed_alloc (unsigned n, unsigned size)
  { void *p;
    check_cuda_error (cudaMallocManaged (&p, (size_t)n*size), "Alloc failed");
    return p;
  }

# define managed_free cudaFree

  static void *make_managed (void *p, unsigned sz)
  { if (p==0) return p;
    void *q = managed_alloc (1, sz);
    memcpy (q, p, sz);
    return q;
  }

  static void show_gpu (void)
  { struct cudaDeviceProp prop;
    char *e = getenv("SHOW_GPU");
    if (e && strcmp(e,"false")!=0 && strcmp(e,"FALSE")!=0 && strcmp(e,"0")!=0)
    { check_cuda_error (cudaGetDeviceProperties(&prop,0), "Get properties");
      printf("%s, Compute Capability %d.%d\n",  
              prop.name, prop.major, prop.minor);
    }
  }

# define BLKSIZE 64	/* Block size to use when launching CUDA kernels */
# define MAXBLKS 32	/* Maximum number of blocks when launching */

#else  /* NOT USING CUDA */

# define __host__
# define __device__
# define __managed__

# define __restrict__ restrict

# define managed_alloc chk_alloc
# define managed_free free

# define make_managed(p,sz) (p)

# define show_gpu() do {} while(0)

#endif


#define HOSTDEV __host__ __device__  /* For convenience */
#define STAMAN static __managed__
#define EXTMAN extern __managed__


#if __CUDA_ARCH__  /* COMPILING FOR GPU */

# define abort abort_in_GPU

  HOSTDEV static void abort(void)
  { printf("Call of abort() in GPU code - but continuing anyway\n");
  }

# undef USE_SIMD_INTRINSICS
# undef USE_FMA
# undef USE_SLEEF

#endif
