/* MISC.C - Miscellaneous utility procedures. */

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
 *
 * Some of the "range" facilities are adapted from modifications by
 * Carl Edward Rasmussen, 1995.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "misc.h"


/* ADD NUMBERS REPRESENTED BY THEIR LOGARITHMS.  Computes log(exp(a)+exp(b))
   in such a fashion that it works even when a and b have large magnitude. */

double addlogs
( double a,
  double b
)
{ 
  return a>b ? a + log(1+exp(b-a)) : b + log(1+exp(a-b));
}


/* SUBTRACT NUMBERS REPRESENTED BY THEIR LOGARITHMS.  If a>=b, computes 
   log(exp(a)-exp(b)) in such a fashion that it works even when a has
   large magnitude.  It's an error if a<b.  */

double sublogs
( double a,
  double b
)
{ 
  if (a<b) abort();
  return a + log(1-exp(b-a));
}


/* ALLOCATE SPACE AND CHECK FOR ERROR.  Calls 'calloc' to allocate space,
   and then displays an error message and exits if the space couldn't be
   found. */

void *chk_alloc
( unsigned n,		/* Number of elements */
  unsigned size		/* Size of each element */
)
{ 
  void *p;

  p = calloc(n,size);

  if (p==0)
  { fprintf(stderr,"Ran out of memory (while trying to allocate %d bytes)\n",
      n*size);
    exit(1);
  }

  return p;
}


/* PARSE RANGE SPECIFICATION.  Parses a string of the following form:

       [low][:[high]][%modulus|+number]

   'low' and 'high' must be a non-negative integers.  If present, 'modulus' 
   or 'number' must be a positive integer (in fact, 'number' must be greater
   than one).  If 'low' is omitted, it defaults to one (not zero, even though      zero is a legal value).  If neither 'modulus' nor 'number' are specified, 
   a modulus of one is assumed.  If 'high' is omitted but the colon or a 
   modulus or a number is present, 'high' is set to -1 as a signal to the 
   caller.  If just a single number appears, it is interpreted as a value 
   for 'low', and 'high' is set to -2.  Something must appear (i.e. the empty 
   string is not legal).

   The caller provides pointers to places to store 'low', 'high', and 
   'modulus' or 'number'.  The pointer for modulus/number may be null, in 
   which case a modulus/number specification is illegal.  If a 'number'
   is specified, its negation is stored, to distinguish it from a
   'modulus'.

   Incorrect specifications lead to an error message being displayed, and
   the program being terminated.  The checks include verifying that 
   low<=high when 'high' is specified, but the caller may need to check
   this also after in cases where 'high' is left to default. */

void parse_range
( char *str,		/* String to parse */
  int *low,		/* Place to store 'low' */
  int *high,		/* Place to store 'high', or -1 or -2 if absent */
  int *modulus		/* Place to store 'modulus' or 'number' (negated),
                           or a null pointer if these are illegal */
)
{
  char *s;
  char t;

  s = str;

  if (*s==0) goto error;

  /* Look for value for 'low', or let it default. */

  *low = 1;
  
  if (*s>='0' && *s<='9')
  { 
    *low = 0;

    while (*s>='0' && *s<='9') 
    { *low = 10*(*low) + (*s-'0');
      s += 1;
    }
  }

  /* Look for value for 'high', or signal its absence. */

  *high = *s==0 ? -2 : -1;

  if (*s==':')
  { 
    s += 1;

    if (*s!=0 && *s>='0' && *s<='9')
    {
      *high = 0;

      while (*s>='0' && *s<='9') 
      { *high = 10*(*high) + (*s-'0');
        s += 1;
      }

      if (*high<*low)
      { fprintf(stderr,"High end of range is less than low end: %s\n",str);
        exit(1);
      }
    }
  }

  /* Look for value for 'modulus' or 'number' if such is legal, or let 
     modulus default to one. */

  if (modulus!=0)
  { *modulus = 1;
  }

  t = *s;

  if (*s=='%' || *s=='+')
  { 
    if (modulus==0) goto error;

    s += 1;

    *modulus = 0;

    while (*s>='0' && *s<='9') 
    { *modulus = 10*(*modulus) + (*s-'0');
      s += 1;
    }
    
    if (*modulus==0) goto error;

    if (t=='+') 
    { if (*modulus==1) goto error;
      *modulus = -*modulus;
    }
  }

  /* Check for garbage at end. */

  if (*s!=0) goto error;

  return;

  /* Report error. */

error:

  fprintf(stderr,"Bad range specification: %s\n",str);
  exit(1);
}


/* PARSE RANGE GIVEN IN TERMS OF TIMES.  Like the parse_range procedure
   above, except that 'low' and 'high' are floating-point numbers (typically
   representing quantities of compute time), with the default for 'low'
   being zero. */

void parse_time_range
( char *str,		/* String to parse */
  double *low,		/* Place to store 'low' */
  double *high,		/* Place to store 'high', or -1 or -2 if absent */
  int *modulus		/* Place to store 'modulus', or null if illegal */
)
{
  int point;
  char *s;
  char t;

  s = str;

  if (*s==0) goto error;

  /* Look for value for 'low', or let it default. */

  *low = 0.0; 
  point = 0;
  
  while (*s>='0' && *s<='9' || *s=='.') 
  {
    if (*s=='.') 
    { if (point) goto error; 
      point = 1; 
    }
    else 
    { *low = 10*(*low) + (*s-'0');
      point *= 10;
    }

    s += 1;
  }

  if (point) *low /= point;

  /* Look for value for 'high', or signal its absence. */

  *high = *s==0 ? -2.0 : -1.0;

  if (*s==':')
  { 
    s += 1;

    if (*s!=0 && (*s>='0' && *s<='9' || *s=='.'))
    {
      *high = 0.0;
      point = 0;

      while (*s>='0' && *s<='9' || *s=='.' && !point) 
      {
        if (*s=='.')
        { if (point) goto error; 
          point = 1; 
        }
        else 
        { *high = 10*(*high) + (*s-'0');
          point *= 10;
        }

        s += 1;
      }

      if (point) *high /= point;

      if (*high<*low)
      { fprintf(stderr,"High end of time range is less than low end: %s\n",str);
        exit(1);
      }
    }
  }

  /* Look for value for 'modulus' or 'number' if such is legal, or let 
     modulus default to one. */

  if (modulus!=0)
  { *modulus = 1;
  }

  t = *s;

  if (t=='%' || t=='+')
  { 
    if (modulus==0) goto error;

    s++;
    *modulus = 0;

    while (*s>='0' && *s<='9') 
    { *modulus = 10*(*modulus) + (*s-'0');
      s += 1;
    }
    
    if (*modulus==0) goto error;

    if (t=='+') 
    { if (*modulus==1) goto error;
      *modulus = -*modulus;
    }
  }

  /* Check for garbage at end. */

  if (*s!=0) goto error;

  return;

  /* Report error. */

error:

  fprintf(stderr,"Bad time range specification: %s\n",str);
  exit(1);
}


/* PARSE RANGE OF NON-NEGATIVE INTEGERS.  Parses a string of the following form:

       low[:high]]

   'low' and 'high' must be non-negative integers.  If the colon and 'high' 
   are omitted, high is set to low.

   The caller provides pointers to places to store 'low' and 'high'.

   If a specification is syntactically incorrect, or if high<low, the
   value 0 is returned.  The value 1 is returned if everything is OK. */

int parse_non_neg_range
( char *str,		/* String to parse */
  int *low,		/* Place to store 'low' */
  int *high		/* Place to store 'high' */
)
{
  char junk;

  if (strchr(str,':')==0)
  { 
    if (sscanf(str,"%d%c",low,&junk)!=1) return 0;
    if (*low<0) return 0;
    *high = *low;
  }
  else
  {
    if (sscanf(str,"%d:%d%c",low,high,&junk)!=2) return 0;
    if (*low<0 || *high<*low) return 0;
  }
  
  return 1;
}


/* PARSE RANGE OF REALS.  Parses a string of the following form:

       low[:[high][:pow]]

   'low' and 'high' must be real numbers.  If 'high' is omitted but the colon 
   is present, 'high' is set to infinity.  If just a single number appears, it 
   is interpreted as the value for both 'low' and 'high'.  The 'pow' part
   may be present if the 'pow' argument is non-zero.  It may be any real value,
   and defaults to 1.

   The caller provides pointers to places to store 'low' and 'high', and
   perhaps 'pow'.

   If a specification is syntactically incorrect, or if high<low, the
   value 0 is returned.  The value 1 is returned if everything is OK. */

int parse_real_range
( char *str,		/* String to parse */
  double *low,		/* Place to store 'low' */
  double *high,		/* Place to store 'high' */
  double *pow		/* Place to store 'pow', or zero */
)
{
  char *p, *q;
  char junk;

  p = strchr(str,':');
  if (p!=0) 
  { q = strchr(p+1,':');
  }

  if (p==0) /* no colons */
  { 
    if (sscanf(str,"%lf%c",low,&junk)!=1) return 0;
    *high = *low;
    if (pow) *pow = 1;
  }

  else if (q==0) /* one colon */
  { if (*(p+1)==0)
    { if (sscanf(str,"%lf:%c",low,&junk)!=1) return 0;
      *high = 1.0/0.0;
    }
    else
    { if (sscanf(str,"%lf:%lf%c",low,high,&junk)!=2) return 0;
      if (*high<*low) return 0;
    }
    if (pow) *pow = 1;
  }

  else /* two colons */
  { if (pow==0) return 0;
    if (*(p+1)==':')
    { if (sscanf(str,"%lf::%lf%c",low,pow,&junk)!=2) return 0;
      *high = 1.0/0.0;
    }
    else
    { if (sscanf(str,"%lf:%lf:%lf%c",low,high,pow,&junk)!=3) return 0;
      if (*high<*low) return 0;
    }
  }
  
  return 1;
}


/* PARSE A LIST OF FLAG ITEMS.  The list has the form:

        [:[-]<item>{,<item>}]

   A null list means that all items are included.  The "-" indicates that
   all EXCEPT the listed items are included.

   Parse_flags sets bits in a char array to indicate which items the 
   flag applies to.  It exits with an error message if an item in
   the string is out of range, or there are other problems. */

void parse_flags
( char *s,		/* String to parse */
  unsigned short *a,	/* Array in which to set bits */
  int n,		/* Length of a, maximum value of an item */
  int flag		/* Bit to set in a to indicate flag applies */
)
{ 
  int i, r;

  if (*s==0)
  { for (i = 0; i<n; i++)
    { a[i] |= flag;
    }
    return;
  }

  if (*s++!=':')
  { fprintf(stderr,"Bad format for flag argument\n");
    exit(1);
  }

  r = *s=='-';
  if (r) 
  { s += 1;
  }

  while (*s!=0)
  {
    i = atoi(s);
    if (i<1 || i>n)
    { fprintf(stderr,"Flag item out of range: %d\n",i);
      exit(1);
    }

    while (*s>='0' && *s<='9') 
    { s += 1;
    }

    a[i-1] |= flag;

    if (*s!=0 && *s++!=',')
    { fprintf(stderr,"Bad format for flag argument\n");
      exit(1);
    }
  }

  if (r)
  { for (i = 0; i<n; i++)
    { a[i] ^= flag;
    }
  }
}


/* CREATE A LIST OF FLAG ITEMS.  Creates a string listing all the items
   to which a flag applies.  The string has the format parsed by parse_flags.
   A null string is generated if the flag applies to all items, and the
   "-" option is used if that is a shorter way of expressing the set.  

   Returns the number of items that the flag applies to. */

int list_flags
( unsigned short *a,	/* Array holding flag bits */
  int  n,		/* Length of a */
  int  flag,		/* Bit set in a to indicate flag applies */
  char *s		/* Place to store list - must be big enough! */
)
{ 
  int i, c, r, f;

  c = 0;
  for (i = 0; i<n; i++)
  { if (a[i]&flag) 
    { c += 1;
    }
  }

  if (c==n)
  { *s = 0;
    return c;
  }

  *s++ = ':';

  r = c>n/2;
  if (r)
  { *s++ = '-';
  }

  f = 1;

  for (i = 0; i<n; i++)
  {
    if (r ? (a[i]&flag)==0 : (a[i]&flag)!=0)
    { if (f) 
      { f = 0;
      }
      else
      { *s++ = ','; 
      }
      sprintf(s,"%d",i+1);
      while (*s!=0)
      { s += 1;
      }
    }
  }

  *s = 0;

  return c;
}


/* COUNT INPUTS NOT OMITTED. */

int not_omitted
( unsigned short *a,	/* Array holding omit flag bits, null if none */
  int  n,		/* Length of a (ie, total number of inputs) */
  int  flag		/* Bit set in a to indicate input omitted */
)
{
  int c, i;

  if (a==0) return n;

  c = n;
  for (i = 0; i<n; i++)
  { if (a[i]&flag) 
    { c -= 1;
    }
  }

  return c;
}
