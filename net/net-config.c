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
  char s[101]; 
  s[100] = 0;
  while (fscanf(fp,"%100s",s)==1)
  { char *h = strchr(s,'#');
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
  int ns,	/* Number of source units, for error checking */
  int nd,	/* Number of destination units, for error checking */
  char paren	/* Operation to perform */
)
{
  for (;;)
  { 
    int s, d, w, r, l;
    char junk, nparen;
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
      { item = do_items (file, start_item, p, ns, nd, nparen);
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

    if (item[1]==NULL || strcmp(item[1],")")==0 || strcmp(item[1],"]")==0 
                      || strcmp(item[1],"}")==0 
     || item[2]==NULL || strcmp(item[2],")")==0 || strcmp(item[2],"]")==0
                      || strcmp(item[2],"}")==0)
    { fprintf (stderr, 
               "Incomplete triple in weight configuration file: %s\n",
               file);
      exit(2);
    }

    s = convert_item(item[0],lasts,0);
    d = convert_item(item[1],lastd,0);
    w = convert_item(item[2],lastw,0);

    lasts = s; lastd = d; lastw = w;

    if (paren=='(' || paren=='{')
    {
      if (s<1 || d<1 || w<1 || s>ns || d>nd || w<1)
      { fprintf (stderr, 
                 "Out of range index in weight configuration: %s\n",
                 file);
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
    }

    item += 3;
  }
}


/* READ WEIGHT CONFIGURATION.  Passed the configuration file and numbers of
   units in the source and destination layers.  Returns a newly-allocated
   structure with the configuration. */

net_config *net_config_read (char *file, int ns, int nd)
{ 
  char **item = read_items (file);
  int i;
  
  net_config *p;
  p = (net_config *) chk_alloc (1, sizeof *p);
  p->conn = (net_connection *) chk_alloc (Max_conn, sizeof *p->conn);

  p->N_wts = 0;
  p->N_conn = 0;

  lasts = 0;
  lastd = 0;
  lastw = 0;

  for (i = 0; i<2*26; i++) varval[i] = 0;

  char **ir = do_items (file, item, p, ns, nd, '(');
  if (*ir!=NULL)
  { fprintf (stderr, 
             "Excess items in weight configuration file: %s, have %s\n",
             file, *ir);
    exit(2);
  }

  for (i = 0; item[i]!=NULL; i++) free(item[i]);
  free (item);

  net_connection *q = (net_connection *) chk_alloc (p->N_conn, sizeof *q);
  memcpy (q, p->conn, p->N_conn * sizeof *q);
  free(p->conn);
  p->conn = q;

  return p;
}
