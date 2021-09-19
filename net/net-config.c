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
  p = (net_config *) managed_alloc (1, sizeof *p);
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

  net_connection *q = (net_connection *) managed_alloc (p->N_conn+1, sizeof *q);
  memcpy (q, p->conn, p->N_conn * sizeof *q);
  q[p->N_conn].w = -1;
  free(p->conn);
  p->conn = q;

  net_config_sort(p);

  return p;
}


/* PRODUCE SORTED / GROUPED VERSIONS OF THE CONFIGURATION. */

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

void net_config_sort (net_config *cf)
{ 
  int n = cf->N_conn;

  /* We will keep remaining connections, not otherwise handled, in 'rem'. */

  net_connection *rem = 
    (net_connection *) chk_alloc (n+1, sizeof *rem);  /* one -1 at end */
  memcpy (rem, cf->conn, (n+1) * sizeof *rem);
  int r = n;

  /* We will put all connections, as sorted and grouped, in successive parts 
     of 'all', setting pointers to parts of it in cf. */

  net_connection *all = (net_connection *) managed_alloc (n+9+3, sizeof *all);  
              /* Allow up to nine -1's, then +3 to ensure AVX loads are OK */
  int a = 0;

  /* Temporary storage. */

  net_connection *tmp = 
    (net_connection *) chk_alloc (n+1, sizeof *rem);  /* one -1 at end */

  non_adjacency (cf->conn,  "original");

  /* Find groups of four connections with same value for s, and sequential
     values for d and w.  Condense these to single entries. */

  if (!CONFIG_QUAD_S_4D_4W)
  { cf->quad_s_4d_4w = all+a;
    all[a++].w = -1;
  }
  else
  {
    qsort (rem, r, sizeof *rem, cmp_s_wmd_d);

    int i, j, k;
    i = j = k = 0;
    while (i < r)
    { int s = rem[i].s;
      int wmd = rem[i].w-rem[i].d;
      if (r-i >= 4 && rem[i+1].s==s  && rem[i+2].s==s && rem[i+3].s==s
           && rem[i+1].w-rem[i+1].d==wmd  && rem[i+2].w-rem[i+2].d==wmd 
           && rem[i+3].w-rem[i+3].d==wmd)
      { tmp[j] = rem[i];
        j += 1;
        i += 4;
      }
      else
      { rem[k] = rem[i];
        k += 1;
        i += 1;
      }
    }

    tmp[j].w = -1;
    non_adjacency (tmp, "quads4d4w");

    r = k;
    rem[k].w = -1;

    cf->quad_s_4d_4w = all+a;
    memcpy (all+a, tmp, j * sizeof *all);
    all[a+j].w = -1;
    a += j+1;
  }

  /* Find groups of four single connections with the same value for d, if
     this is enabled. */

  if (!CONFIG_SINGLE4)
  { cf->single4_d = all+a;
    all[a++].w = -1;
  }
  else
  { 
    qsort (rem, r, sizeof *rem, cmp_d_s_w);

    int i, j, k;
    i = j = k = 0;
    while (i < r)
    { int d = rem[i].d;
      if (r-i >= 4 && rem[i+1].d==d  && rem[i+2].d==d && rem[i+3].d==d)
      { tmp[j+0] = rem[i+0];
        tmp[j+1] = rem[i+1];
        tmp[j+2] = rem[i+2];
        tmp[j+3] = rem[i+3];
        j += 4;
        i += 4;
      }
      else
      { rem[k] = rem[i];
        k += 1;
        i += 1;
      }
    }

    tmp[j].w = -1;
    non_adjacency (tmp, "single4d");

    r = k;
    rem[k].w = -1;

    cf->single4_d = all+a;
    memcpy (all+a, tmp, j * sizeof *all);
    all[a+j].w = -1;
    a += j+1;
  }

  /* Find groups of four single connections with the same value for s, if
     this is enabled. */

  if (!CONFIG_SINGLE4)
  { cf->single4_s = all+a;
    all[a++].w = -1;
  }
  else
  { 
    qsort (rem, r, sizeof *rem, cmp_s_d_w);

    int i, j, k;
    i = j = k = 0;
    while (i < r)
    { int s = rem[i].s;
      if (r-i >= 4 && rem[i+1].s==s  && rem[i+2].s==s && rem[i+3].s==s)
      { tmp[j+0] = rem[i+0];
        tmp[j+1] = rem[i+1];
        tmp[j+2] = rem[i+2];
        tmp[j+3] = rem[i+3];
        j += 4;
        i += 4;
      }
      else
      { rem[k] = rem[i];
        k += 1;
        i += 1;
      }
    }

    tmp[j].w = -1;
    non_adjacency (tmp, "single4s");

    r = k;
    rem[k].w = -1;

    cf->single4_s = all+a;
    memcpy (all+a, tmp, j * sizeof *all);
    all[a+j].w = -1;
    a += j+1;
  }

  /* Copy remaining connections from 'rem' to end of 'all', sorting them
     by s or d, whichever seems better. */

  qsort (rem, r, sizeof *rem, cmp_s_d_w);
  int nadj_s = non_adjacency (rem, "rem-by-s");
  memcpy (all+a, rem, r * sizeof *all);
  qsort (rem, r, sizeof *rem, cmp_d_s_w);
  int nadj_d = non_adjacency (rem, "rem-by-d");
  if (nadj_d < nadj_s)
  { memcpy (all+a, rem, r * sizeof *all);
  }
  cf->single = all+a;
  all[a+r].w = -1;

  free(tmp);
  free(rem);

  if (0)  /* can enable for debugging */
  {
    if (CONFIG_SINGLE4)
    { printf("single4_s:\n");
      for (int i = 0; cf->single4_s[i].w >= 0; i++)
      { printf("%d %d %d\n",
                cf->single4_s[i].s, cf->single4_s[i].d, cf->single4_s[i].w);
      }
      printf("single4_d:\n");
      for (int i = 0; cf->single4_d[i].w >= 0; i++)
      { printf("%d %d %d\n",
                cf->single4_d[i].s, cf->single4_d[i].d, cf->single4_d[i].w);
      }
    }
    printf("single:\n");
    for (int i = 0; cf->single[i].w >= 0; i++)
    { printf("%d %d %d\n",
              cf->single[i].s, cf->single[i].d, cf->single[i].w);
    }
  }
}
