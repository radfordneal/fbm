/* NET-CONFIG.C - Procedures relating to weight configuration files. */

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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "cuda-use.h"

#include "misc.h"
#include "log.h"
#include "data.h"
#include "prior.h"
#include "model.h"
#include "net.h"


/* STATE DURING ITEM PROCESSING. */

#define letters "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

static int varval[2*26];          /* Values of variables */
static int lasts, lastd, lastw;   /* Previous source, dest., weight indexes */
static int at_index;              /* Index for output from "@" item */


static void net_config_sort (net_config *cf, int);


/* READ ITEMS, AS CHARACTER STRINGS.  Returns a pointer to an array of
   pointers to null-terminated strings, with NULL after the last. */

#define Max_items 1000000

static char **read_items (char *file)
{ FILE *fp = file[0]=='%' ? popen(file+1,"r") : fopen(file,"r");
  if (fp==NULL)
  { fprintf(stderr,"Can't open weight configuration file: %s\n",file);
    exit(2);
  }
  char **item = (char **) chk_alloc (Max_items+1, sizeof *item);
  int n = 0;
  char s[1002]; 
  s[1001] = 0;
  while (fscanf(fp,"%1001s",s)==1)
  { if (strlen(s)>1000)
    { fprintf (stderr, "Line in configuration file is too long (max 1000)\n");
      exit(2);
    }
    char *h = strchr(s,'#');
    if (h)
    { char c;
      *h = 0;
      do { c = fgetc(fp); } while (c!=EOF && c!='\n');
      if (h==s) continue;
    }
    if (n==Max_items)
    { fprintf (stderr, 
               "Too many items in configuration file: %s\n",
               file);
      exit(2);
    }
    item[n] = (char *) chk_alloc(strlen(s)+1,1);
    strcpy(item[n],s);
    n += 1;
  }
  item[n] = NULL;
  fclose(fp);
  (void) fopen("/dev/null","r");  /* Kludge to bypass macOS bug */
  return item;
}


/* CONVERT AN ITEM TO A NUMBER.  The value may be relative to "last". 
   The number may contain variable references.  The paren argument
   is the trailing paren for repeat factors, or zero for indexes. */

static int convert_item (char *s, int last, char paren)
{ 
  if (strcmp(s,"=")==0)
  { return last;
  }
  else if (strcmp(s,"+")==0)
  { return last+1;
  }
  else if (strcmp(s,"-")==0)
  { return last-1;
  }

  int v, t, n;
  char next, follow;
  char *p;

  v = *s=='+' || *s=='-' ? last : 0;
  p = s;

  for (;;)
  { int has_sign = p[0]=='+' || p[0]=='-';
    if (strchr(letters,p[has_sign]))
    { t = varval [strchr(letters,p[has_sign]) - letters];
      if (p[0]=='-') t = -t;
      next = p[has_sign+1];
      n = 1 + (next!=0);
    }
    else
    { n = sscanf(p,"%d%c%c",&t,&next,&follow);
    }
    if (n<1 || n>1 && next!='+' && next!='-' && next!=paren
            || n>2 && next==paren) 
    { fprintf (stderr, "Bad item in weight configuration file: %s\n", s);
      exit(2);
    }
    v += t;
    if (n==1 || next==paren)
    { break;
    }
    for (p++; *p!='+' && *p!='-'; p++) ;
  }
  
  return v;
}


/* PROCESS ITEMS, RETURNING POINTER TO ITEM AFTER LAST PROCESSED.  Stores
   results of processing in the configuration structure pointed to by p.
   The paren argument controls the actions taken for items - '(' or '{'
   creates connections, '[' only updates "last" values. */

static char **do_items 
( char *file,	/* File name, for error messages */
  char **item,	/* Array of items to process */
  net_config *p,/* Place to store connections produced */
  int ns0,	/* Number of source units (for error checking), -1 if biases */
  int nd,	/* Number of destination units, for error checking */
  char paren	/* Operation to perform */
)
{
  int biases = 0;  /* 1 if doing biases, with doublets, not triplets */
  int ns = ns0;    /* number of source units, 1 for biases */

  if (ns0==-1)
  { ns = 1;
    biases = 1;
  }

  for (;;)
  { 
    int s, d, w, r, l;
    char nparen;
    char *it = item[0];

    if (it==NULL || strcmp(it,")")==0 || strcmp(it,"]")==0 || strcmp(it,"}")==0)
    { return item;
    }

    l = strlen(it);
    if (l>0 && (it[l-1]=='(' || it[l-1]=='[' || it[l-1]=='{'))
    { 
      nparen = it[l-1];
      if (l==1)
      { r = 1;  /* default repetition factor is 1 */
      }
      else
      { r = convert_item(it,0,nparen);
      }

      if (r<1)
      { fprintf (stderr,
        "Non-positive repeat factor in weight configuration file: %s, %s, %d\n",
         file, it, r);
        exit(2);
      }

      item += 1;
      char **start_item = item;
      int sv_lasts = lasts, sv_lastd = lastd, sv_lastw = lastw;
        
      while (r>0)
      { item = do_items (file, start_item, p, ns0, nd, nparen);
        r -= 1;
      }

      if (nparen=='{')
      { lasts = sv_lasts; lastd = sv_lastd; lastw = sv_lastw;
      }

      if (*item!=NULL) 
      { if (nparen=='(' && **item!=')' 
         || nparen=='[' && **item!=']'
         || nparen=='{' && **item!='}')
        { fprintf (stderr,
                  "Mis-matched bracket type in weigth configuration file: %s\n",
                   file);
          exit(2);
        }
        item += 1;
      }

      continue;
    }

    if (l>2 && it[1]=='=' && strchr(letters,it[0]))
    { 
      varval [strchr(letters,it[0]) - letters] = convert_item(it+2,0,0);
      item += 1;
      continue;
    }

    if (strcmp(it,"@")==0)
    { fprintf(stderr,"%d @ %d %d %d\n",at_index,lasts,lastd,lastw);
      at_index += 1;
      item += 1;
      continue;
    }

    if (biases)
    { if (item[1]==NULL || strcmp(item[1],")")==0 || strcmp(item[1],"]")==0 
                        || strcmp(item[1],"}")==0)
      { fprintf (stderr, 
                 "Incomplete doublet in bias configuration file: %s\n",
                 file);
        exit(2);
      }
    }
    else
    { if (item[1]==NULL || strcmp(item[1],")")==0 || strcmp(item[1],"]")==0 
                        || strcmp(item[1],"}")==0 
       || item[2]==NULL || strcmp(item[2],")")==0 || strcmp(item[2],"]")==0
                        || strcmp(item[2],"}")==0)
      { fprintf (stderr, 
                 "Incomplete triplet in weight configuration file: %s\n",
                 file);
        exit(2);
      }
    }

    int j = 0;

    s = biases ? 1 : convert_item(item[j++],lasts,0);
    d = convert_item(item[j++],lastd,0);
    w = convert_item(item[j++],lastw,0);

    lasts = s; lastd = d; lastw = w;

    if (paren=='(' || paren=='{')
    {
      if (s<1 || d<1 || w<1 || s>ns || d>nd || w<1)
      { fprintf (stderr, 
         "Out of range index in weight configuration: %s, %d, %d %d %d\n",
          file, p->N_conn+1, s, d, w);
        exit(2);
      }

      if (p->N_conn==Max_conn)
      { fprintf (stderr,
                 "Too many connections in weight configuration: %s\n",
                 file);
        exit(2);
      }

      p->conn[p->N_conn].s = s-1;  /* stored indexes are 0-based */
      p->conn[p->N_conn].d = d-1;
      p->conn[p->N_conn].w = w-1;

      if (w>p->N_wts) p->N_wts = w;

      p->N_conn += 1;

      if (0) fprintf(stderr,"Conn %d: %d %d %d\n",p->N_conn,s,d,w);
    }

    item += j;
  }
}


/* READ WEIGHT CONFIGURATION.  Passed the configuration file, the
   number of units in the source layer (-1 if for biases), and the
   number of units in the destination layer.  Returns a newly-allocated 
   structure with the configuration.  Also produces the sorted
   versions of the configuration, used for more efficient processing. */

net_config *net_config_read (char *file, int ns, int nd)
{ 
  char **item = read_items (file);
  int i;
  
  net_config *p;
  p = (net_config *) chk_alloc (1, sizeof *p);
  p->conn = (net_connection *) chk_alloc (Max_conn+1, sizeof *p->conn);

  p->N_wts = 0;
  p->N_conn = 0;

  lasts = 0;
  lastd = 0;
  lastw = 0;

  for (i = 0; i<2*26; i++) varval[i] = 0;

  at_index = 1;

  char **ir = do_items (file, item, p, ns, nd, '(');
  if (*ir!=NULL)
  { fprintf (stderr, 
             "Excess items in weight configuration file: %s, have %s\n",
             file, *ir);
    exit(2);
  }

  for (i = 0; item[i]!=NULL; i++) free(item[i]);
  free (item);

  net_connection *q = (net_connection *) chk_alloc (p->N_conn+1, sizeof *q);
  memcpy (q, p->conn, p->N_conn * sizeof *q);
  q[p->N_conn].w = -1;
  free(p->conn);
  p->conn = q;

  net_config_sort (p, ns == -1);

  return p;
}


/* PRODUCE SORTED / GROUPED VERSIONS OF THE CONFIGURATION.  Sets up the
   'single', 'single4_s', etc. fields of the net_config structure, based
   on the connections in 'conn'.  Only the quad versions are set up for
   biases. */

/* Return a measure of nonadjacency in a sequence of connections.  May also
   print this out (with 'prefix') if enabled below, for performance assessment
   purposes. */

static int non_adjacency (net_connection *a, const char *prefix)
{ 
  int n = 0, c0 = 0, c1 = 0, c2 = 0, cx = 0;
  while (a->w >= 0)
  { net_connection *b = a+1;
    if (b->w >= 0)
    { if (b->w < a->w || b->w > a->w+2) cx += 1;
      else if (b->w == a->w+2) c2 += 1;
      else if (b->w == a->w+1) c1 += 1;
      else c0 += 1;
      if (b->s < a->s || b->s > a->s+2) cx += 1;
      else if (b->s == a->s+2) c2 += 1;
      else if (b->s == a->s+1) c1 += 1;
      else c0 += 1;
      if (b->d < a->d || b->d > a->d+2) cx += 1;
      else if (b->d == a->d+2) c2 += 1;
      else if (b->d == a->d+1) c1 += 1;
      else c0 += 1;
    }
    n += 1;
    a += 1;
  }

  int s = c1 + 2*c2 + 5*cx;

  if (0)   /* may be enabled for debugging or performance evaluation */
  { printf ("%s: n=%d s=%d c0=%d c1=%d c2=%d cx=%d\n",
            prefix ? prefix : "", n, s, c0, c1, c2, cx);
  }
  
  return s;
}

/* Comparison functions for sorting. */

static int cmp_s_d_w (const void *a0, const void *b0)
{ net_connection *a = (net_connection *) a0, *b = (net_connection *) b0;
  int r;
  r = (int)a->s - (int)b->s;
  if (r!=0) return r;
  r = (int)a->d - (int)b->d;
  if (r!=0) return r;
  r = (int)a->w - (int)b->w;
  return r;
}

static int cmp_d_s_w (const void *a0, const void *b0)
{ net_connection *a = (net_connection *) a0, *b = (net_connection *) b0;
  int r;
  r = (int)a->d - (int)b->d;
  if (r!=0) return r;
  r = (int)a->s - (int)b->s;
  if (r!=0) return r;
  r = (int)a->w - (int)b->w;
  return r;
}

static int cmp_wmod4_w_d_s (const void *a0, const void *b0)
{ net_connection *a = (net_connection *) a0, *b = (net_connection *) b0;
  int r;
  r = (int)(a->w&3) - (int)(b->w&3);
  if (r!=0) return r;
  r = (int)a->w - (int)b->w;
  if (r!=0) return r;
  r = (int)a->d - (int)b->d;
  if (r!=0) return r;
  r = (int)a->s - (int)b->s;
  return r;
}

static int cmp_wmodGTH_w_d_s (const void *a0, const void *b0)
{ net_connection *a = (net_connection *) a0, *b = (net_connection *) b0;
  int r;
  r = (int)(a->w&(GTH-1)) - (int)(b->w&(GTH-1));
  if (r!=0) return r;
  r = (int)a->w - (int)b->w;
  if (r!=0) return r;
  r = (int)a->d - (int)b->d;
  if (r!=0) return r;
  r = (int)a->s - (int)b->s;
  return r;
}

static int cmp_dmod4_d_w_s (const void *a0, const void *b0)
{ net_connection *a = (net_connection *) a0, *b = (net_connection *) b0;
  int r;
  r = (int)(a->d&3) - (int)(b->d&3);
  if (r!=0) return r;
  r = (int)a->d - (int)b->d;
  if (r!=0) return r;
  r = (int)a->w - (int)b->w;
  if (r!=0) return r;
  r = (int)a->s - (int)b->s;
  return r;
}

static int cmp_smod4_s_w_d (const void *a0, const void *b0)
{ net_connection *a = (net_connection *) a0, *b = (net_connection *) b0;
  int r;
  r = (int)(a->s&3) - (int)(b->s&3);
  if (r!=0) return r;
  r = (int)a->s - (int)b->s;
  if (r!=0) return r;
  r = (int)a->w - (int)b->w;
  if (r!=0) return r;
  r = (int)a->d - (int)b->d;
  return r;
}

static int cmp_s_wmd_d (const void *a0, const void *b0)
{ net_connection *a = (net_connection *) a0, *b = (net_connection *) b0;
  int r;
  r = (int)a->s - (int)b->s;
  if (r!=0) return r;
  r = (a->w-a->d) - (b->w-b->d);
  if (r!=0) return r;
  r = (int)a->d - (int)b->d;
  return r;
}

/* Copy connections already sorted by w mod M (terminated by w of -1), 
   inserting  w of -1 after the end of each section.  Returns the number 
   of connections, including the M with -1s.  Stores starting indexes
   for section for each mod value in start, if it is not null.  M must
   be a power of two. */

static int copy_wmod 
  (net_connection *dst, net_connection *src, int *start, int M)
{
  int i, j, w, m;

  if (start) start[0] = 0;

  i = 0;
  j = 0;
  m = 0;

  while ((w = src[i].w) >= 0)
  { while (m != (w & (M-1))) 
    { dst[j++].w = -1;
      m += 1;
      if (start && m<M) start[m] = j;
    }
    dst[j++] = src[i++];
  }

  while (m < M)
  { dst[j++].w = -1;
    m += 1;
    if (start && m<M) start[m] = j;
  }

  return j;
}

/* Copy connections already sorted by d mod M (terminated by w of -1), 
   inserting  w of -1 after the end of each section.  Returns the number 
   of connections, including the M with -1s.  Stores starting indexes
   for section for each mod value in start, if it is not null.  M must
   be a power of two. */

static int copy_dmod 
  (net_connection *dst, net_connection *src, int *start, int M)
{
  int i, j, d, m;

  if (start) start[0] = 0;

  i = 0;
  j = 0;
  m = 0;

  while (src[i].w >= 0)
  { d = src[i].d;
    while (m != (d & (M-1))) 
    { dst[j++].w = -1;
      m += 1;
      if (start && m<M) start[m] = j;
    }
    dst[j++] = src[i++];
  }

  while (m < M)
  { dst[j++].w = -1;
    m += 1;
    if (start && m<M) start[m] = j;
  }

  return j;
}

/* Copy connections already sorted by s mod M (terminated by w of -1), 
   inserting  w of -1 after the end of each section.  Returns the number 
   of connections, including the M with -1s.  Stores starting indexes
   for section for each mod value in start, if it is not null.  M must
   be a power of two. */

static int copy_smod 
  (net_connection *dst, net_connection *src, int *start, int M)
{
  int i, j, s, m;

  if (start) start[0] = 0;

  i = 0;
  j = 0;
  m = 0;

  while (src[i].w >= 0)
  { s = src[i].s;
    while (m != (s & (M-1))) 
    { dst[j++].w = -1;
      m += 1;
      if (start && m<M) start[m] = j;
    }
    dst[j++] = src[i++];
  }

  while (m < M)
  { dst[j++].w = -1;
    m += 1;
    if (start && m<M) start[m] = j;
  }

  return j;
}

/* Copy out pairs of connections with same w.  Updates 'a' to hold the
   remaining connections, terminated with -1.  Puts the pairs in 'b',
   terminated with -1. */

static void copy_pairs
( net_connection *a,	/* -1 terminated array of all connections, updated */
  net_connection *b,	/* Place to store pairs, with -1 terminator */
  int *an,		/* Set to number of connections left in a, if not 0 */
  int *bn		/* Set to number of conn. (2xpairs) put in b, if not 0*/
)
{ int i, j, k;
  i = j = k = 0;
  while (a[i].w != -1)
  { if (a[i].w==a[i+1].w)
    { b[k] = a[i];
      b[k+1] = a[i+1];
      i += 2;
      k += 2;
    }
    else
    { a[j] = a[i];
      i += 1;
      j += 1;
    }
  }
  a[j].w = -1;
  b[k].w = -1;
  if (an) *an = j;
  if (bn) *bn = k;
}

/* The actual net_config_sort function, called from elsewhere. */

static void net_config_sort (net_config *cf, int biases)
{ 
  int minus_ones = 2*GTH + 8 + 3;  /* should be more than enough for each set,
                                      with +3 to ensure AVX loads are OK */
  int n = cf->N_conn;
  int i;

  /* Temporary storage. */

  net_connection *tmp = 
    (net_connection *) chk_alloc (n+1, sizeof *tmp);  /* one -1 at end */
  net_connection *tmp2 = 
    (net_connection *) chk_alloc (n+1, sizeof *tmp2); /* one -1 at end */

  non_adjacency (cf->conn,  "original");

  /* Find groups of four connections with same value for s, and
     sequential values for d and w.  Condense these to single entries,
     stored in 'quad', with number in 'q'.  Put remaining connections
     in 'rem', with number in 'r'. */

  int q;  /* number of items in quad, not including -1 at end */
  int r;  /* number of items not in quad, not including -1 at end */

  net_connection *quad = 
    (net_connection *) chk_alloc (n+1, sizeof *quad); /* one -1 at end */
  net_connection *rem = 
    (net_connection *) chk_alloc (n+1, sizeof *rem);  /* one -1 at end */

  memcpy (rem, cf->conn, (n+1) * sizeof *rem);
  qsort (rem, n, sizeof *rem, cmp_s_wmd_d);

  i = 0;
  r = 0;
  q = 0;
  while (i < n)
  { int s = rem[i].s;
    int wmd = rem[i].w-rem[i].d;
    if (n-i >= 4 && rem[i+1].s==s  && rem[i+2].s==s && rem[i+3].s==s
         && rem[i+1].w-rem[i+1].d==wmd  && rem[i+2].w-rem[i+2].d==wmd 
         && rem[i+3].w-rem[i+3].d==wmd)
    { quad[q] = rem[i];
      q += 1;
      i += 4;
    }
    else
    { rem[r] = rem[i];
      r += 1;
      i += 1;
    }
  }

  rem[r].w = -1;
  quad[q].w = -1;

  /* Start work on connections for use in CPU.  We will put all
     connections used on CPU, as sorted and grouped, in successive
     parts of 'all', setting pointers to parts of it in cf.  The next
     unused entry in 'all' is stored in 'a'. */

  net_connection *all = (net_connection *) chk_alloc(n+minus_ones, sizeof *all);
  int a = 0;

  net_connection *left = (net_connection *) chk_alloc(n+1, sizeof *left);
  int l;  /* number of items in 'left' */

  /* If CONFIG_QUAD_S_4D_4W enabled, create quad entries for use in CPU. 
     Otherwise, create null entries for this.  Put whatever isn't used
     for quads in 'left', with number 'l'. */

  if (!CONFIG_QUAD_S_4D_4W)
  { cf->quad_s_4d_4w = all+a;
    all[a++].w = -1;
    cf->quad_s_4d_4w_2 = all+a;
    all[a++].w = -1;
    memcpy (left, cf->conn, (n+1) * sizeof *left);
    l = n;
  }
  else
  {
    memcpy (tmp, quad, (q+1) * sizeof *tmp);
    qsort (tmp, q, sizeof *quad, cmp_wmod4_w_d_s);

    non_adjacency (tmp, "quads4d4w");  /* only useful for info, if enabled */

    int jj = q;

    if (!MAKE_QUAD_PAIRS)
    { cf->quad_s_4d_4w_2 = all+a;
      all[a++].w = -1;
    }
    else
    { int m;
      copy_pairs (tmp, all+a, &jj, &m);
      cf->quad_s_4d_4w_2 = all+a;
      a += m+1;
    }

    cf->quad_s_4d_4w = all+a;
    memcpy (all+a, tmp, jj * sizeof *all);
    all[a+jj].w = -1;
    a += jj+1;

    memcpy (left, rem, (r+1) * sizeof *left);
    l = r;
  }

  /* Find groups of four single connections with the same value for d, if
     this is enabled, for use in CPU computations.  Not done for biases. */

  if (!CONFIG_SINGLE4 || biases)
  { cf->single4_d = all+a;
    all[a++].w = -1;
  }
  else
  { 
    qsort (left, l, sizeof *left, cmp_d_s_w);

    int i, j, k;
    i = j = k = 0;
    while (i < l)
    { int d = left[i].d;
      if (l-i >= 4 && left[i+1].d==d  && left[i+2].d==d && left[i+3].d==d)
      { tmp[j+0] = left[i+0];
        tmp[j+1] = left[i+1];
        tmp[j+2] = left[i+2];
        tmp[j+3] = left[i+3];
        j += 4;
        i += 4;
      }
      else
      { left[k] = left[i];
        k += 1;
        i += 1;
      }
    }

    tmp[j].w = -1;
    non_adjacency (tmp, "single4d");  /* only useful for info, if enabled */

    l = k;
    left[k].w = -1;

    cf->single4_d = all+a;
    memcpy (all+a, tmp, j * sizeof *all);
    all[a+j].w = -1;
    a += j+1;
  }

  /* Find groups of four single connections with the same value for s, if
     this is enabled, for use in CPU computations.  Not done for biases. */

  if (!CONFIG_SINGLE4 || biases)
  { cf->single4_s = all+a;
    all[a++].w = -1;
  }
  else
  { 
    qsort (left, l, sizeof *left, cmp_s_d_w);

    int i, j, k;
    i = j = k = 0;
    while (i < l)
    { int s = left[i].s;
      if (l-i >= 4 && left[i+1].s==s  && left[i+2].s==s && left[i+3].s==s)
      { tmp[j+0] = left[i+0];
        tmp[j+1] = left[i+1];
        tmp[j+2] = left[i+2];
        tmp[j+3] = left[i+3];
        j += 4;
        i += 4;
      }
      else
      { left[k] = left[i];
        k += 1;
        i += 1;
      }
    }

    tmp[j].w = -1;
    non_adjacency (tmp, "single4s");  /* only useful for info, if enabled */

    l = k;
    left[k].w = -1;

    cf->single4_s = all+a;
    memcpy (all+a, tmp, j * sizeof *all);
    all[a+j].w = -1;
    a += j+1;
  }

  /* Copy remaining connections from 'left' to end of 'all', sorting them
     by s or d, whichever seems better. */

  qsort (left, l, sizeof *left, cmp_s_d_w);
  int nadj_s = non_adjacency (left, "left-by-s");
  memcpy (all+a, left, l * sizeof *all);
  qsort (left, l, sizeof *left, cmp_d_s_w);
  int nadj_d = non_adjacency (left, "left-by-d");
  if (nadj_d < nadj_s)
  { memcpy (all+a, left, l * sizeof *all);
  }
  cf->single = all+a;
  a += l;
  all[a++].w = -1;

  /* Record the block all the CPU versions came from, in config structure. */

  cf->all = all;
  cf->all_length = a;

  /* Start work on connections for For use in GPU computations.  There
     are three sets, sorted by weight, destination, and source, used
     for gradient, forward, and backwards computations, respectively.
     They come in multiple sections according to mod values. */

  net_connection *all_gpu = (net_connection *) 
                               chk_alloc(3*(n+minus_ones), sizeof *all_gpu);
  int a_gpu = 0;

  /* If enabled, set up quad_s_4d_4w_wgpu connections for use in GPU
     gradient computations, sections marked by extra -1 indicators for
     when the first weight mod GTH changes.  Optionally, sets are taken
     out for pairs with the same weight, put in quad_s_4d_4w_2_wgpu. */

  net_connection *p;  /* pointer to what's not put in the quad sets here */
  int pl;
  int e;
    
  if (!CONFIG_QUAD_GPU_S_4D_4W)
  { cf->quad_s_4d_4w_wgpu = all_gpu+a_gpu;
    for (e = 0; e<GTH; e++)
    { all_gpu[a_gpu++].w = -1;
      cf->start_quad_wgpu[e] = e;
    }
    cf->quad_s_4d_4w_2_wgpu = all_gpu+a_gpu;
    for (e = 0; e<GTH; e++)
    { all_gpu[a_gpu++].w = -1;
      cf->start_quad_2_wgpu[e] = e;
    }
    p = cf->conn;
    pl = n;
  }
  else
  { int jj = q;
    memcpy (tmp, quad, (q+1) * sizeof *tmp);
    qsort (tmp, q, sizeof *tmp, cmp_wmodGTH_w_d_s);
    if (!MAKE_QUAD_GPU_PAIRS)
    { cf->quad_s_4d_4w_2_wgpu = all_gpu+a_gpu;
      for (e = 0; e<GTH; e++)
      { all_gpu[a_gpu++].w = -1;
        cf->start_quad_2_wgpu[e] = e;
      }
    }
    else
    { int n;
      copy_pairs (tmp, tmp2, &jj, &n);
      cf->quad_s_4d_4w_2_wgpu = all_gpu+a_gpu;
      a_gpu += 
        copy_wmod (cf->quad_s_4d_4w_2_wgpu, tmp2, cf->start_quad_2_wgpu, GTH);
    }
    cf->quad_s_4d_4w_wgpu = all_gpu+a_gpu;
    a_gpu += copy_wmod (cf->quad_s_4d_4w_wgpu, tmp, cf->start_quad_wgpu, GTH);
    p = rem;
    pl = r;
  }

  /* Set up other connections (not in quad wgpu sets) for use in GPU
     gradient computations, in other_wgpu.  In sections by w mod GTH.
     May make paired version in other_2_wgpu. */

  memcpy (tmp, p, pl * sizeof *tmp);  
  qsort (tmp, pl, sizeof *tmp, cmp_wmodGTH_w_d_s);
  tmp[pl].w = -1;

  if (!MAKE_OTHER_GPU_PAIRS)
  { int e;
    cf->other_2_wgpu = all_gpu+a_gpu;
    for (e = 0; e<GTH; e++)
    { all_gpu[a_gpu++].w = -1;
      cf->start_other_2_wgpu[e] = e;
   }
  }
  else
  { copy_pairs (tmp, tmp2, 0, 0);
    cf->other_2_wgpu = all_gpu+a_gpu;
    a_gpu += copy_wmod (cf->other_2_wgpu, tmp2, cf->start_other_2_wgpu, GTH);
  }

  cf->other_wgpu = all_gpu+a_gpu;
  a_gpu += copy_wmod (cf->other_wgpu, tmp, cf->start_other_wgpu, GTH);

  /* Similarly, the quad_s_4d_4w_dgpu connections are used for gpu
     function computations, sorted by d, but a paired version is not
     set up. In sections by d mod 4. */

  if (!CONFIG_QUAD_GPU_S_4D_4W)
  { int e;
    cf->quad_s_4d_4w_dgpu = all_gpu+a_gpu;
    for (e = 0; e<4; e++) all_gpu[a_gpu++].w = -1;
    p = cf->conn;
    pl = n;
  }
  else
  { memcpy (tmp, quad, (q+1) * sizeof *tmp);
    qsort (tmp, q, sizeof *tmp, cmp_dmod4_d_w_s);
    cf->quad_s_4d_4w_dgpu = all_gpu+a_gpu;
    a_gpu += copy_dmod (cf->quad_s_4d_4w_dgpu, quad, 0, 4);
    p = rem;
    pl = r;
  }

  /* Set up other connections (not in quad_s_4d_4w_dgpu) for use in gpu
     forward pass computations. */

  memcpy (tmp, p, pl * sizeof *tmp);  
  qsort (tmp, pl, sizeof *tmp, cmp_dmod4_d_w_s);
  tmp[pl].w = -1;
  cf->other_dgpu = all_gpu+a_gpu;
  a_gpu += copy_dmod (cf->other_dgpu, tmp, cf->start_other_dgpu, 4);

  /* And, the quad_s_4d_4w_sgpu connections are used for gpu backward pass
     computations, sorted by s. In sections by s mod 4. */

  if (!CONFIG_QUAD_GPU_S_4D_4W)
  { int e;
    cf->quad_s_4d_4w_sgpu = all_gpu+a_gpu;
    for (e = 0; e<4; e++) all_gpu[a_gpu++].w = -1;
    p = cf->conn;
    pl = n;
  }
  else
  { memcpy (tmp, quad, q * sizeof *tmp);
    qsort (tmp, q, sizeof *tmp, cmp_smod4_s_w_d);
    cf->quad_s_4d_4w_sgpu = all_gpu+a_gpu;
    a_gpu += copy_smod (cf->quad_s_4d_4w_sgpu, quad, cf->start_quad_sgpu, 4);
    p = rem;
    pl = r;
  }

  /* Set up other connections (not in quad_s_4d_4w_sgpu) for use in gpu
     backward pass computations. */

  memcpy (tmp, p, pl * sizeof *tmp);  
  qsort (tmp, pl, sizeof *tmp, cmp_smod4_s_w_d);
  tmp[pl].w = -1;
  cf->other_sgpu = all_gpu+a_gpu;
  a_gpu += copy_smod (cf->other_sgpu, tmp, cf->start_other_sgpu, 4);

  /* Record the block all the GPU versions came from, in config structure. */

  cf->all_gpu = all_gpu;
  cf->all_gpu_length = a_gpu;

  free(tmp);
  free(tmp2);
  free(rem);
  free(quad);
}


/* MAKE COPY OF CONFIG IN GPU MEMORY.  Returns a pointer to a config 
   structure in GPU memory with pointers to GPU memory set up, copied 
   from the config structure passed. */

#if __CUDACC__

net_config *net_config_to_gpu (net_config *cf)
{ 
  net_config dcf;

  dcf = *cf;

  check_cuda_error (cudaMalloc (&dcf.conn, (dcf.N_conn+1) * sizeof *dcf.conn),
                    "alloc of dev config conn");
  check_cuda_error (cudaMemcpy (dcf.conn, cf->conn, 
                               (dcf.N_conn+1) * sizeof *dcf.conn,
                               cudaMemcpyHostToDevice),
                    "copy to dev config conn");

  check_cuda_error (cudaMalloc (&dcf.all, dcf.all_length * sizeof *dcf.all),
                    "alloc of dev config all");
  check_cuda_error (cudaMemcpy (dcf.all, cf->all, 
                                dcf.all_length * sizeof *dcf.all,
                                cudaMemcpyHostToDevice),
                    "copy to dev config all");

  dcf.single = dcf.all + (cf->single - cf->all);
  dcf.single4_s = dcf.all + (cf->single4_s - cf->all);
  dcf.single4_d = dcf.all + (cf->single4_d - cf->all);
  dcf.quad_s_4d_4w = dcf.all + (cf->quad_s_4d_4w - cf->all);
  dcf.quad_s_4d_4w_2 = dcf.all + (cf->quad_s_4d_4w_2 - cf->all);

  check_cuda_error (cudaMalloc (&dcf.all_gpu, 
                                dcf.all_gpu_length * sizeof *dcf.all_gpu),
                    "alloc of dev config all_gpu");
  check_cuda_error (cudaMemcpy (dcf.all_gpu, cf->all_gpu, 
                                dcf.all_gpu_length * sizeof *dcf.all_gpu,
                                cudaMemcpyHostToDevice),
                    "copy to dev config all_gpu");

  dcf.quad_s_4d_4w_wgpu = dcf.all_gpu + (cf->quad_s_4d_4w_wgpu - cf->all_gpu);
  dcf.quad_s_4d_4w_2_wgpu = dcf.all_gpu 
                             + (cf->quad_s_4d_4w_2_wgpu - cf->all_gpu);
  dcf.other_wgpu = dcf.all_gpu + (cf->other_wgpu - cf->all_gpu);
  dcf.other_2_wgpu = dcf.all_gpu + (cf->other_2_wgpu - cf->all_gpu);

  dcf.quad_s_4d_4w_dgpu = dcf.all_gpu + (cf->quad_s_4d_4w_dgpu - cf->all_gpu);
  dcf.other_dgpu = dcf.all_gpu + (cf->other_dgpu - cf->all_gpu);

  dcf.quad_s_4d_4w_sgpu = dcf.all_gpu + (cf->quad_s_4d_4w_sgpu - cf->all_gpu);
  dcf.other_sgpu = dcf.all_gpu + (cf->other_sgpu - cf->all_gpu);
  

  net_config *dev_dcf;
  check_cuda_error (cudaMalloc (&dev_dcf, sizeof *dev_dcf),
                    "alloc of dev config struct");
  check_cuda_error (cudaMemcpy (dev_dcf, &dcf, sizeof dcf,
                                cudaMemcpyHostToDevice),
                    "copy to dev config struct");
  
  return dev_dcf;
}

#endif
