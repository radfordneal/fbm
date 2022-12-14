/* NET-CONFIG.C - Procedures relating to weight configuration files. */

/* Copyright (c) 2021-2022 by Radford M. Neal 
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

static char **read_items (char *file0)
{ 
  char *file =   /* Copy so we don't modify the argument, and can prepend dir */
          chk_alloc (strlen(file0)+1, 1);
  strcpy (file, file0);

  char **item = (char **) chk_alloc (Max_items+1, sizeof *item);
  FILE *fp;
  int n = 0;
  char s[1002]; 
  s[1001] = 0;

  while (*file)
  {
    char *comma = strchr(file,',');
    if (comma) *comma = 0;

    int literal = *file!='%' && 
                    (strchr(file,' ')!=NULL || strchr(file,'=')!=NULL);

    if (literal)
    { while (file[0]==' ') file += 1;
      char *p = file+strlen(file);
      while (p>file && *(p-1)==' ') *--p = 0;
    }
    else
    { if (file[0]=='%')
      { fp = popen(file+1,"r");
      }
      else
      { fp = fopen(file,"r");
        if (fp==NULL)
        { char *fn = chk_alloc (strlen(file)+strlen(CONFIG_DIR)+1, 1);
          strcpy(fn,CONFIG_DIR);
          strcat(fn,file);
          fp = fopen(fn,"r");
        }
      }
      if (fp==NULL)
      { fprintf(stderr,"Can't open weight configuration file: %s\n",file);
        exit(2);
      }
    }

    for (;;)
    { if (literal)
      { if (*file==0) 
        { break;
        }
        char *sp = strchr(file,' ');
        if (sp) *sp = 0;
        strncpy(s,file,1001);
        file = sp ? sp+1 : ""; 
        while (file[0]==' ') file += 1;
      }
      else
      { if (fscanf(fp,"%1001s",s)!=1)
        { break;
        }
      }
      if (strlen(s)>1000)
      { fprintf (stderr, "Line in configuration file is too long (max 1000)\n");
        exit(2);
      }
      char *h = strchr(s,'#');
      if (h)
      { char c;
        if (!literal) do { c = fgetc(fp); } while (c!=EOF && c!='\n');
        *h = 0;
        if (h==s)
        { continue;
        }
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

    if (!literal)
    { fclose(fp);
      (void) fopen("/dev/null","r");  /* Kludge to bypass macOS bug */
    }

    file = comma ? comma+1 : "";
  }

  item[n] = NULL;
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
                  "Mis-matched bracket type in weight configuration file: %s\n",
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

    if (l>3 
        && (it[1]=='+' || it[1]=='-' || it[1]=='/' || it[1]=='*' || it[1]=='?')
        && it[2]=='=' && strchr(letters,it[0]))
    { 
      int op = convert_item(it+3,0,0);
      if (it[1]=='/' && op<=0)
      { fprintf (stderr, 
                 "Non-positive divisor in file: %s, %s, %d\n", file, it, op);
        exit(2);
      }
      if (it[1]=='+')
      { varval [strchr(letters,it[0]) - letters] += op;
      }
      else if (it[1]=='-')
      { varval [strchr(letters,it[0]) - letters] -= op;
      }
      else if (it[1]=='/')
      { varval [strchr(letters,it[0]) - letters] /= op;
      }
      else if (it[1]=='*')
      { varval [strchr(letters,it[0]) - letters] *= op;
      }
      else
      { if (varval [strchr(letters,it[0]) - letters] == 0)
        { varval [strchr(letters,it[0]) - letters] = op;
        }
      }
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
         "Out of range %s index in weight configuration: %s\n", 
          s<1 || s>ns ? "source" : d<1 || d>nd ? "destination" : "weight", 
          file);
        fprintf (stderr, 
         "conn idx %d, %d %d %d\n", p->N_conn+1, s, d, w);
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

static int cmp_w_d_s (const void *a0, const void *b0)
{ net_connection *a = (net_connection *) a0, *b = (net_connection *) b0;
  int r;
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

static int cmp_dmodNTH_d_w_s (const void *a0, const void *b0)
{ net_connection *a = (net_connection *) a0, *b = (net_connection *) b0;
  int r;
  r = (int)(a->d&(NTH-1)) - (int)(b->d&(NTH-1));
  if (r!=0) return r;
  r = (int)a->d - (int)b->d;
  if (r!=0) return r;
  r = (int)a->w - (int)b->w;
  if (r!=0) return r;
  r = (int)a->s - (int)b->s;
  return r;
}

static int cmp_smodNTH_s_w_d (const void *a0, const void *b0)
{ net_connection *a = (net_connection *) a0, *b = (net_connection *) b0;
  int r;
  r = (int)(a->s&(NTH-1)) - (int)(b->s&(NTH-1));
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
   inserting a w of -1 after the end of each section.  Returns the number 
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
   inserting a w of -1 after the end of each section.  Returns the number 
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
   inserting a w of -1 after the end of each section.  Returns the number 
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

/* Find groups of four connections in 'src' with same value for s,
   and sequential values for d and w.  Condense these to single
   entries, stored in 'quad', with number in *quadn.  Put remaining
   connections in 'rem8', with number in *remn.  Both are terminated
   with -1 (not counted in number). */

static void find_quad
( net_connection *src,	/* Original set of connections */
  int n,		/* Number of connections in src */
  net_connection *quad,	/* Place to store condensed sets of four connections */
  int *quadn,		/* Number of sets in quad */
  net_connection *rem,	/* Place to store remaining connections */
  int *remn		/* Number of connections in rem */
)
{
  memcpy (rem, src, (n+1) * sizeof *rem);
  qsort (rem, n, sizeof *rem, cmp_s_wmd_d);

  int i = 0;
  *remn = 0;
  *quadn = 0;
  while (i < n)
  { int s = rem[i].s;
    int wmd = rem[i].w-rem[i].d;
    if (n-i >= 4 
      && rem[i+1].s==s  && rem[i+2].s==s && rem[i+3].s==s
      && rem[i+1].w-rem[i+1].d==wmd && rem[i+2].w-rem[i+2].d==wmd 
      && rem[i+3].w-rem[i+3].d==wmd)
    { quad[*quadn] = rem[i];
      *quadn += 1;
      i += 4;
    }
    else
    { rem[*remn] = rem[i];
      *remn += 1;
      i += 1;
    }
  }

  rem[*remn].w = -1;
  quad[*quadn].w = -1;
}

/* Find groups of eight connections in 'src' with same value for s,
   and sequential values for d and w.  Condense these to single
   entries, stored in 'oct', with number in *octn.  Put remaining
   connections in 'rem8', with number in *remn.  Both are terminated
   with -1 (not counted in number). */

static void find_oct
( net_connection *src,	/* Original set of connections */
  int n,		/* Number of connections in src */
  net_connection *oct,	/* Place to store condensed sets of eight connections */
  int *octn,		/* Number of sets in oct */
  net_connection *rem,	/* Place to store remaining connections */
  int *remn		/* Number of connections in remn */
)
{
  memcpy (rem, src, (n+1) * sizeof *rem);
  qsort (rem, n, sizeof *rem, cmp_s_wmd_d);

  int i = 0;
  *remn = 0;
  *octn = 0;
  while (i < n)
  { int s = rem[i].s;
    int wmd = rem[i].w-rem[i].d;
    if (n-i >= 8 
      && rem[i+1].s==s  && rem[i+2].s==s && rem[i+3].s==s
      && rem[i+4].s==s  && rem[i+5].s==s && rem[i+6].s==s && rem[i+7].s==s
      && rem[i+1].w-rem[i+1].d==wmd && rem[i+2].w-rem[i+2].d==wmd 
      && rem[i+3].w-rem[i+3].d==wmd && rem[i+4].w-rem[i+4].d==wmd
      && rem[i+5].w-rem[i+5].d==wmd && rem[i+6].w-rem[i+6].d==wmd
      && rem[i+7].w-rem[i+7].d==wmd)
    { oct[*octn] = rem[i];
      *octn += 1;
      i += 8;
    }
    else
    { rem[*remn] = rem[i];
      *remn += 1;
      i += 1;
    }
  }

  rem[*remn].w = -1;
  oct[*octn].w = -1;
}

/* The actual net_config_sort function, called from elsewhere. */

static void net_config_sort (net_config *cf, int biases)
{ 
  non_adjacency (cf->conn,  "original");  /* For information, if enabled */

  int n = cf->N_conn;

  int minus_ones = 10*GTH + 13;   /* should be more than enough for all sets */
  int i, c;

  /* Use 'left' to hold connections not yet put in other sets, with
     the number in 'leftn'. */

  net_connection *left = (net_connection *) chk_alloc(n+1, sizeof *left);
  int leftn;

  /* Temporary storage.  Enough space for all connections plus one -1. */

  net_connection *tmp = 
    (net_connection *) chk_alloc (n+1, sizeof *tmp);  /* one -1 at end */
  net_connection *tmp2 = 
    (net_connection *) chk_alloc (n+1, sizeof *tmp2); /* one -1 at end */

  /* ----------  Start work on connections for use in CPU  ----------

     We will put all connections used on CPU, as sorted and grouped,
     in successive parts of 'all', setting pointers to parts of it in
     cf.  The next unused entry in 'all' is stored in 'a'. */

  net_connection *all = (net_connection *) chk_alloc(n+minus_ones, sizeof *all);
  int a = 0;

  memcpy (left, cf->conn, (n+1) * sizeof *left);
  leftn = n;

  /* Find groups of eight, if enabled. */

  cf->oct_s_8d_8w = 0;
  if (CONFIG_OCT_S_8D_8W)
  { memcpy (tmp, left, (leftn+1) * sizeof *tmp);
    find_oct (tmp, leftn, all+a, &c, left, &leftn);
    if (c>0)
    { cf->oct_s_8d_8w = all+a;
      a += c+1;
    }
  }

  /* Find groups of four, if enabled. */

  cf->quad_s_4d_4w = 0;
  cf->quad_s_4d_4w_2 = 0;
  if (CONFIG_QUAD_S_4D_4W)
  { memcpy (tmp, left, (leftn+1) * sizeof *tmp);
    find_quad (tmp, leftn, all+a, &c, left, &leftn);
    if (c>0)
    { if (!MAKE_QUAD_PAIRS)
      { cf->quad_s_4d_4w = all+a;
        a += c+1;
      }
      else
      { int m;
        copy_pairs (all+a, tmp, &c, &m);
        if (cf->quad_s_4d_4w = c==0 ? 0 : all+a) a += c+1;
        memcpy (all+a, tmp, (m+1) * sizeof *tmp);
        if (cf->quad_s_4d_4w_2 = m==0 ? 0 : all+a) a += m+1;
      }
    }
  }

  /* Find groups of four single connections with the same value for d, if
     this is enabled, for use in CPU computations.  Not done for biases. */

  cf->single4_d = 0;
  if (CONFIG_SINGLE4 && !biases)
  { 
    qsort (left, leftn, sizeof *left, cmp_d_s_w);

    int i, j, k;
    i = j = k = 0;
    while (i < leftn)
    { int d = left[i].d;
      if (leftn-i >= 4 && left[i+1].d==d  && left[i+2].d==d && left[i+3].d==d)
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

    leftn = k;
    left[k].w = -1;

    if (j>0)
    { memcpy (all+a, tmp, (j+1) * sizeof *all);
      cf->single4_d = all+a;
      a += j+1;
    }
  }

  /* Find groups of four single connections with the same value for s, if
     this is enabled, for use in CPU computations.  Not done for biases. */

  cf->single4_s = 0;
  if (CONFIG_SINGLE4 && !biases)
  { 
    qsort (left, leftn, sizeof *left, cmp_s_d_w);

    int i, j, k;
    i = j = k = 0;
    while (i < leftn)
    { int s = left[i].s;
      if (leftn-i >= 4 && left[i+1].s==s  && left[i+2].s==s && left[i+3].s==s)
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

    leftn = k;
    left[k].w = -1;

    if (j>0)
    { memcpy (all+a, tmp, (j+1) * sizeof *all);
      cf->single4_s = all+a;
      a += j+1;
    }
  }

  /* Copy remaining connections from 'left' to end of 'all', sorting them
     by s or d, whichever seems better. */

  cf->single = 0;
  if (leftn>0)
  { qsort (left, leftn, sizeof *left, cmp_s_d_w);
    int nadj_s = non_adjacency (left, "left-by-s");
    memcpy (all+a, left, (leftn+1) * sizeof *all);
    qsort (left, leftn, sizeof *left, cmp_d_s_w);
    int nadj_d = non_adjacency (left, "left-by-d");
    if (nadj_d < nadj_s)
    { memcpy (all+a, left, (leftn+1) * sizeof *all);
    }
    all[a+leftn].w = -1;
    cf->single = all+a;
    a += leftn+1;
  }

  /* Record the block all the CPU versions came from, in config structure. */

  cf->all = all;
  cf->all_length = a;

  /* --------  Start work on connections for use in GPU computations  -------- 

     There are three sets, sorted by weight, destination, and source,
     used for gradient, forward, and backwards computations,
     respectively.  They come in multiple sections according to mod
     values.  All are stored in all_gpu. */

  net_connection *all_gpu = (net_connection *) 
                               chk_alloc(3*(n+minus_ones), sizeof *all_gpu);
  int a_gpu = 0;

  /* -- Connections for gradient computation (..._wgpu) -- */

  memcpy (left, cf->conn, (n+1) * sizeof *left);
  leftn = n;

  /* The oct_s_8d_8w_wgpu connections are used for gpu gradient computations. */

  cf->oct_s_8d_8w_wgpu = 0;
  if (CONFIG_OCT_GPU_S_8D_8W_GRAD)
  { memcpy (tmp, left, (leftn+1) * sizeof *tmp);
    find_oct (tmp, leftn, tmp2, &c, left, &leftn);
    if (c>0)
    { qsort (tmp2, c, sizeof *tmp2, cmp_wmodGTH_w_d_s);
      cf->oct_s_8d_8w_wgpu = all_gpu+a_gpu;
      a_gpu += copy_wmod (cf->oct_s_8d_8w_wgpu, tmp2, cf->start_oct_wgpu, GTH);
    }
  }

  /* If enabled, set up quad_s_4d_4w_wgpu connections for use in GPU
     gradient computations, sections marked by extra -1 indicators for
     when the first weight mod GTH changes. */

  cf->quad_s_4d_4w_wgpu = 0;
  if (CONFIG_QUAD_GPU_S_4D_4W_GRAD)
  { memcpy (tmp, left, (leftn+1) * sizeof *tmp);
    find_quad (tmp, leftn, tmp2, &c, left, &leftn);
    if (c>0)
    { qsort (tmp2, c, sizeof *tmp2, cmp_wmodGTH_w_d_s);
      c = copy_wmod (all_gpu+a_gpu, tmp2, cf->start_quad_wgpu, GTH);
      cf->quad_s_4d_4w_wgpu = all_gpu+a_gpu;
      a_gpu += c+1;
    }
  }

  /* Set up other connections (not in oct or quad wgpu sets) for use in
     GPU gradient computations, in other_wgpu.  In sections by w mod GTH. */

  cf->other_wgpu = 0;
  if (leftn>0)
  { memcpy (tmp, left, (leftn+1) * sizeof *tmp);  
    qsort (tmp, leftn, sizeof *tmp, cmp_wmodGTH_w_d_s);
    cf->other_wgpu = all_gpu+a_gpu;
    a_gpu += copy_wmod (cf->other_wgpu, tmp, cf->start_other_wgpu, GTH);
  }

  /* -- Connections for forward computation (..._dgpu) -- */

  memcpy (left, cf->conn, (n+1) * sizeof *left);
  leftn = n;

  /* The oct_s_8d_8w_dgpu connections are used for gpu forward
     function computations, sorted by d.  In sections by d mod NTH. */

  cf->oct_s_8d_8w_dgpu = 0;
  if (CONFIG_OCT_GPU_S_8D_8W_FW)
  { memcpy (tmp, left, (leftn+1) * sizeof *tmp);
    find_oct (tmp, leftn, tmp2, &c, left, &leftn);
    if (c>0)
    { qsort (tmp2, c, sizeof *tmp2, cmp_dmodNTH_d_w_s);
      cf->oct_s_8d_8w_dgpu = all_gpu+a_gpu;
      a_gpu += copy_dmod (cf->oct_s_8d_8w_dgpu, tmp2, cf->start_oct_dgpu, NTH);
    }
  }

  /* Similarly for quad_s_4d_4w_dgpu connections. */

  cf->quad_s_4d_4w_dgpu = 0;
  if (CONFIG_QUAD_GPU_S_4D_4W_FW)
  { memcpy (tmp, left, (leftn+1) * sizeof *tmp);
    find_quad (tmp, leftn, tmp2, &c, left, &leftn);
    if (c>0)
    { qsort (tmp2, c, sizeof *tmp2, cmp_dmodNTH_d_w_s);
      cf->quad_s_4d_4w_dgpu = all_gpu+a_gpu;
      a_gpu += copy_dmod(cf->quad_s_4d_4w_dgpu, tmp2, cf->start_quad_dgpu, NTH);
    }
  }

  /* Set up other connections (not in oct_s_8d_8w_dgpu and quad_s_4d_4w_dgpu)
     for use in gpu forward pass computations. */

  cf->other_dgpu = 0;
  if (leftn>0)
  { qsort (left, leftn, sizeof *left, cmp_dmodNTH_d_w_s);
    cf->other_dgpu = all_gpu+a_gpu;
    a_gpu += copy_dmod (cf->other_dgpu, left, cf->start_other_dgpu, NTH);
  }

  /* -- Connections for backward computation (..._sgpu) -- */

  memcpy (left, cf->conn, (n+1) * sizeof *left);
  leftn = n;

  /* The oct_s_8d_8w_sgpu connections are used for gpu backward pass
     computations, sorted by s.  In sections by s mod NTH. */

  cf->oct_s_8d_8w_sgpu = 0;
  if (CONFIG_OCT_GPU_S_8D_8W_BW)
  { memcpy (tmp, left, (leftn+1) * sizeof *tmp);
    find_oct (tmp, leftn, tmp2, &c, left, &leftn);
    if (c>0)
    { qsort (tmp2, c, sizeof *tmp2, cmp_smodNTH_s_w_d);
      cf->oct_s_8d_8w_sgpu = all_gpu+a_gpu;
      a_gpu += copy_smod (cf->oct_s_8d_8w_sgpu, tmp2, cf->start_oct_sgpu, NTH);
    }
  }

  /* Similarly for quad_s_4d_4w_sgpu connections. */

  cf->quad_s_4d_4w_sgpu = 0;
  if (CONFIG_QUAD_GPU_S_4D_4W_BW)
  { memcpy (tmp, left, (leftn+1) * sizeof *tmp);
    find_quad (tmp, leftn, tmp2, &c, left, &leftn);
    if (c>0)
    { qsort (tmp2, c, sizeof *tmp2, cmp_smodNTH_s_w_d);
      cf->quad_s_4d_4w_sgpu = all_gpu+a_gpu;
      a_gpu += copy_smod(cf->quad_s_4d_4w_sgpu, tmp2, cf->start_quad_sgpu, NTH);
    }
  }

  /* Set up other connections (not in oct_s_8d_8w_sgpu and quad_s_4d_4w_sgpu)
     for use in gpu backward pass computations. */

  cf->other_sgpu = 0;
  if (leftn>0)
  { qsort (left, leftn, sizeof *left, cmp_smodNTH_s_w_d);
    cf->other_sgpu = all_gpu+a_gpu;
    a_gpu += copy_smod (cf->other_sgpu, left, cf->start_other_sgpu, NTH);
  }

  /* Record the block all the GPU versions came from, in config structure. */

  cf->all_gpu = all_gpu;
  cf->all_gpu_length = a_gpu;

  free(tmp2);
  free(tmp);
  free(left);
}
