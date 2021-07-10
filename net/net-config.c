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


/* READ WEIGHT CONFIGURATION.  Passed the configuration file and numbers of
   units in the source and destination layers.  Returns a newly-allocated
   structure with the configuration. */

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

static int convert_item (char *s, int last)
{ if (strcmp(s,"=")==0) return last;
  else if (strcmp(s,"+")==0) return last+1;
  else if (strcmp(s,"-")==0) return last-1;
  int v; char junk;
  if (sscanf(s,"%d%c",&v,&junk)!=1) 
  { fprintf (stderr, "Bad item in weight configuration file: %s\n", s);
    exit(2);
  }
  return *s=='+' || *s=='-' ? last+v : v;
}

static int lasts, lastd, lastw;

static char **do_items 
        (char *file, char **item, net_config *p, int ns, int nd, char paren)
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
      if (l==1) /* default repetition factor is 1 */
      { r = 1;
        nparen = it[0];
      }
      else if (sscanf(it,"%d%c%c",&r,&nparen,&junk)!=2 || r<1)
      { fprintf (stderr,
                 "Bad repeat start in weight configuration file: %s, %s\n",
                 file, it);
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

    if (item[1]==NULL || strcmp(item[1],")")==0 || strcmp(item[1],"]")==0 
                      || strcmp(item[1],"}")==0 
     || item[2]==NULL || strcmp(item[2],")")==0 || strcmp(item[2],"]")==0
                      || strcmp(item[2],"}")==0)
    { fprintf (stderr, 
               "Incomplete triple in weight configuration file: %s\n",
               file);
      exit(2);
    }

    s = convert_item(item[0],lasts);
    d = convert_item(item[1],lastd);
    w = convert_item(item[2],lastw);

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

  char **ir = do_items (file, item, p, ns, nd, '(');
  if (*ir!=NULL)
  { fprintf (stderr, 
             "Not all items read from weight configuration file: %s, have %s\n",
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
