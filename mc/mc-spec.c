/* MC-SPEC.C - Specify parameters of Markov chain Monte Carlo simulation. */

/* Copyright (c) 1995-2022 by Radford M. Neal 
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
#include "mc.h"


static void usage (void);
static void display_specs (log_gobbled *);


/* MAIN PROGRAM. */

int main
( int argc,
  char **argv
)
{
  static mc_ops ops0, *ops = &ops0; /* Static so types get set to zero (empty)*/
  static mc_traj traj0, *traj = &traj0;

  int have_traj, any_specs;
  char junk;
  int o;

  log_file logf;
  log_gobbled logg;
  int index;
  int depth;

  char **ap;
  char *s;

  /* Look for log file name. */

  if (argc<2) usage();

  logf.file_name = argv[1];

  /* See if we are to display existing specifications. */

  if (argc==2 
   || argc==3 && (*argv[2]>='0' && *argv[2]<='9' || strcmp(argv[2],"all")==0))
  {
    index = argc==2 ? -1 : strcmp(argv[2],"all")==0 ? -2 : atoi(argv[2]);

    log_file_open(&logf,0);

    log_gobble_init(&logg,0);
    logg.req_size['o'] = sizeof *ops;
    logg.req_size['t'] = sizeof *traj;

    any_specs = 0;

    if (index==-1)
    {
      while (!logf.at_end)
      { log_gobble(&logf,&logg);
      }
  
      if (logg.data['o']!=0 || logg.data['t']!=0)
      { printf("\nLAST SPECIFICATIONS IN LOG FILE:\n");
        display_specs(&logg);
        any_specs = 1;
      }
  
    }
    else if (index==-2)
    {
      while (!logf.at_end)
      { 
        log_gobble(&logf,&logg);
  
        if (logg.data['o']!=0 && logg.index['o']==logg.last_index
         || logg.data['t']!=0 && logg.index['t']==logg.last_index)
        { printf("\nSPECIFICATIONS AT INDEX %d IN LOG FILE:\n",logg.last_index);
          display_specs(&logg);
          any_specs = 1;
        }
      }
    }
    else
    {
      while (!logf.at_end && logf.header.index<=index)
      { log_gobble(&logf,&logg);
      }

      if (logg.index['o']!=index) logg.data['o'] = 0;
      if (logg.index['t']!=index) logg.data['t'] = 0;
  
      if (logg.data['o']!=0 || logg.data['t']!=0)
      { printf("\nSPECIFICATIONS AT INDEX %d IN LOG FILE:\n",index);
        display_specs(&logg);
        any_specs = 1;
      }
    }

    if (!any_specs)
    { printf("\nNo Monte Carlo specification found\n\n");
    }

    exit(0);
  }

  /* We're adding a specification, not displaying... */

  /* Open log file and read all records (so we see all 'o' and 't' records). */

  log_file_open (&logf, 1);

  log_gobble_init(&logg,0);
  logg.req_size['o'] = sizeof *ops;
  logg.req_size['t'] = sizeof *traj;

  while (!logf.at_end)
  { log_gobble(&logf,&logg);
  }

  index = logf.header.index<0 ? 0 : logf.header.index;

  /* Look at remaining arguments with operation / trajectory specifications. */

  ap = argv+2; 

  o = 0;
  depth = 0;

  while (*ap && strcmp(*ap,"/")!=0)
  {
    if (o==Max_mc_ops)
    { fprintf(stderr,"Too many operations (max %d)\n",Max_mc_ops);
      exit(1);
    }

    if (strcmp(*ap,"end")==0)
    { 
      ops->op[o].type = 'E';
      
      ap += 1;

      depth -= 1;
      if (depth<0) 
      { fprintf(stderr,"Too many 'end' operations\n");
        exit(1);
      }
    }

    else if (strcmp(*ap,"heatbath")==0)
    { 
      ops->op[o].type = 'B';
      ops->op[o].heatbath_decay = 0;

      ap += 1;

      if (*ap && strchr("0123456789+-.",**ap))
      { if ((ops->op[o].heatbath_decay = atof(*ap++))<0) usage();
      }
    }

    else if (strcmp(*ap,"radial-heatbath")==0)
    { 
      ops->op[o].type = 'r';
      ap += 1;
    }

    else if (strcmp(*ap,"mix-momentum")==0)
    { 
      ops->op[o].type = 'X';

      ap += 1;

      if (!*ap || (ops->op[o].heatbath_decay = atof(*ap++))<0) usage();
    }

    else if (strcmp(*ap,"negate")==0)
    { 
      ops->op[o].type = 'N';
      ap += 1;
    }

    else if (strcmp(*ap,"metropolis")==0 || strcmp(*ap,"met")==0
          || strcmp(*ap,"rgrid-met")==0)
    {
      ops->op[o].type = 
         strcmp(*ap,"metropolis")==0 || strcmp(*ap,"met")==0 ? 'M' : 'G';
      ops->op[o].stepsize_adjust = 1;
      ops->op[o].stepsize_alpha = 0;
      ops->op[o].b_accept = 0;
      ops->op[o].r_update = 0;

      ap += 1;

      while (*ap && (*ap)[0]=='-' 
          && strchr("abcdefghijklmnopqrstuvwxyz",(*ap)[1]))
      { if (strcmp("-b",*ap)==0)
        { ops->op[o].b_accept = 1;
        }
        else
        { usage();
        }
        ap += 1;
      }

      if (*ap && strchr("0123456789+-.",**ap))
      { if ((ops->op[o].stepsize_adjust = atof(*ap))==0) usage();
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].stepsize_alpha = atof(strchr(*ap,':')+1))==0) usage();
        }
        ap += 1;
      }
    }

    else if (strcmp(*ap,"slice")==0)
    {
      ops->op[o].type = 'l';
      ops->op[o].stepsize_adjust = 1;
      ops->op[o].stepsize_alpha = 0;
      ops->op[o].g_shrink = 0;

      ap += 1;

      while (*ap && (*ap)[0]=='-' 
     && strchr("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",(*ap)[1]))
      { if (strcmp("-g",*ap)==0)
        { ops->op[o].g_shrink = 1;
        }
        else if (strcmp("-G",*ap)==0)
        { ops->op[o].g_shrink = 2;
        }
        else
        { usage();
        }
        ap += 1;
      }

      if (*ap && strchr("0123456789+-.",**ap))
      { if ((ops->op[o].stepsize_adjust = atof(*ap))==0) usage();
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].stepsize_alpha = atof(strchr(*ap,':')+1))==0) usage();
        }
        ap += 1;
      }
    }

    else if (strcmp(*ap,"slice-gaussian")==0)
    {
      ops->op[o].type = 'u';
      ops->op[o].stepsize_adjust = 1;
      ops->op[o].stepsize_alpha = 0;
      ops->op[o].e_shrink = 0;

      ap += 1;

      while (*ap && (*ap)[0]=='-' 
     && strchr("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",(*ap)[1]))
      { if (strcmp("-e",*ap)==0)
        { ops->op[o].e_shrink = 1;
        }
        else
        { usage();
        }
        ap += 1;
      }

      if (*ap && strchr("0123456789+-.",**ap))
      { if ((ops->op[o].stepsize_adjust = atof(*ap))==0) usage();
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].stepsize_alpha = atof(strchr(*ap,':')+1))==0) usage();
        }
        ap += 1;
      }
    }

    else if (strcmp(*ap,"met-1")==0 
          || strcmp(*ap,"rgrid-met-1")==0
          || strcmp(*ap,"slice-1")==0
          || strcmp(*ap,"slice-over")==0)
    {
      ops->op[o].type = strcmp(*ap,"met-1")==0 ? 'm'
                      : strcmp(*ap,"rgrid-met-1")==0 ? 'g'
                      : strcmp(*ap,"slice-1")==0 ? 'S' : 'O';
                      
      ops->op[o].refinements = 0;
      ops->op[o].b_accept = 0;
      ops->op[o].r_update = 0;
      ops->op[o].refresh_prob = 0.0;
      ops->op[o].stepsize_adjust = 1;
      ops->op[o].stepsize_alpha = 0;
      ops->op[o].steps = 0;
      ops->op[o].firsti = -1;
      ops->op[o].s_factor = 0;
      ops->op[o].s_threshold = 0;

      ap += 1;

      while (*ap && (*ap)[0]=='-' 
          && strchr("abcdefghijklmnopqrstuvwxyz",(*ap)[1]))
      { if (strcmp("-b",*ap)==0 
             && (ops->op[o].type=='m' || ops->op[o].type=='g'))
        { ops->op[o].b_accept = 1;
        }
        else if (strcmp("-r",*ap)==0)
        { ops->op[o].r_update = 1;
        }
        else if (ops->op[o].type=='S' && strcmp("-s",*ap)==0)
        { ap += 1;
          if (*ap==0) usage();
          if (strchr(*ap,'/')!=0)
          { if (sscanf(*ap,"%d/%f%c",&ops->op[o].s_factor,
                       &ops->op[o].s_threshold, &junk)!=2 
             || ops->op[o].s_factor==0)
            { usage();
            }
          }
          else
          { if (sscanf(*ap,"%d%c",&ops->op[o].s_factor,&junk)!=1
             || ops->op[o].s_factor==0)
            { usage();
            }
          }
        }
        else
        { usage();
        }
        ap += 1;
      }

      if (ops->op[o].type=='O')
      { 
        if (*ap && strchr("0123456789",**ap))
        { if ((ops->op[o].refinements = atoi(*ap)) < 0) usage();
          ap += 1;
        }

        if (*ap && strchr("0123456789.",**ap))
        { if ((ops->op[o].refresh_prob = atof(*ap)) < 0) usage();
          ap += 1;
        }
      }

      if (*ap && strchr("0123456789+-.",**ap))
      { if ((ops->op[o].stepsize_adjust = atof(*ap))==0) usage();
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].stepsize_alpha = atof(strchr(*ap,':')+1))==0) usage();
        }
        ap += 1;
      }

      if ((ops->op[o].type=='S' || ops->op[o].type=='O')
           && *ap && strchr("-0123456789",**ap))
      { ops->op[o].steps = atoi(*ap);
        ap += 1;
      }

      if (*ap && strchr("0123456789",**ap))
      { ops->op[o].firsti = atoi(*ap);
        ops->op[o].lasti = ops->op[o].firsti;
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].lasti = atoi(strchr(*ap,':')+1)) < ops->op[o].firsti)
          { usage();
          }
        }
        ap += 1;
      }
      else if (*ap && **ap>='A' && **ap<='Z' && *(*ap+1)==0)
      { ops->op[o].firsti = -(**ap-'A'+2);
        ops->op[o].lasti = 0;
        ap += 1;
      }
    }

    else if (strcmp(*ap,"gaussian-gibbs")==0 || strcmp(*ap,"binary-gibbs")==0)
    {
      ops->op[o].type = strcmp(*ap,"binary-gibbs")==0 ? '1' : 'U';

      ops->op[o].r_update = 0;
      ops->op[o].firsti = -1;

      ap += 1;

      while (*ap && (*ap)[0]=='-' 
          && strchr("abcdefghijklmnopqrstuvwxyz",(*ap)[1]))
      { if (strcmp("-r",*ap)==0)
        { ops->op[o].r_update = 1;
        }
        else
        { usage();
        }
        ap += 1;
      }

      if (*ap && strchr("0123456789",**ap))
      { ops->op[o].firsti = atoi(*ap);
        ops->op[o].lasti = ops->op[o].firsti;
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].lasti = atoi(strchr(*ap,':')+1)) < ops->op[o].firsti)
          { usage();
          }
        }
        ap += 1;
      }
      else if (*ap && **ap>='A' && **ap<='Z' && *(*ap+1)==0)
      { ops->op[o].firsti = -(**ap-'A'+2);
        ops->op[o].lasti = 0;
        ap += 1;
      }
    }

    else if (strcmp(*ap,"dynamic")==0 || strcmp(*ap,"permuted-dynamic")==0
          || strcmp(*ap,"slice-inside")==0 || strcmp(*ap,"slice-outside")==0)
    {
      ops->op[o].type = strcmp(*ap,"dynamic")==0 ? 'D' 
                      : strcmp(*ap,"permuted-dynamic")==0 ? 'P'
                      : strcmp(*ap,"slice-inside")==0 ? 'i' 
                      : strcmp(*ap,"slice-outside")==0 ? 'o' : 0;

      ops->op[o].stepsize_adjust = 1;
      ops->op[o].stepsize_alpha = 0;
      ops->op[o].options = 0;

      ap += 1;

      if (ops->op[o].type=='D' || ops->op[o].type=='P')
      { ops->op[o].adapt_target = ops->op[o].adapt_rate = 0;
        if (*ap && strcmp(*ap,"-D")==0)
        { ops->op[o].options |= 1;
          ap += 1;
        }
        else if (*ap && (*ap)[0]=='-' && (*ap)[1]=='a')
        { float a = 0.25, b = 0.01;
          char junk;
          if ((*ap)[2]!=0 && sscanf(*ap+2,"%f%c",&a,&junk)!=1
                          && sscanf(*ap+2,"%f/%f%c",&a,&b,&junk)!=2
                || a<=0 || a>=0.5 || b<=0)
          { usage();
          }
          ops->op[o].adapt_target = a;
          ops->op[o].adapt_rate = b;
          ops->op[o].options |= 2;  /* implies ^ for stepsize */
          ap += 1;
        }
      }

      if (!*ap || !strchr("0123456789+-.",**ap)) usage();

      if ((ops->op[o].steps = atoi(*ap))<=0) usage();

      if (ops->op[o].type=='o')
      { ops->op[o].in_steps = ops->op[o].steps;
        if (strchr(*ap,'/')!=0)
        { if ((ops->op[o].in_steps = atoi(strchr(*ap,'/')+1))<=0
           || ops->op[o].in_steps>ops->op[o].steps) usage();
        }
      }
      ap += 1;

      if (*ap && strchr("0123456789+-.^",**ap))
      { if (**ap=='^' && (ops->op[o].type=='D' || ops->op[o].type=='P'))
        { ops->op[o].options |= 2;
          if ((*ap)[1]==0 || (*ap)[1]==':')
          { ops->op[o].stepsize_adjust = 1;
          }
          else
          { if ((ops->op[o].stepsize_adjust = atof(*ap+1))==0) usage();
          }
        }
        else
        { if ((ops->op[o].stepsize_adjust = atof(*ap))==0) usage();
        }
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].stepsize_alpha = atof(strchr(*ap,':')+1))==0) usage();
        }
        ap += 1;
      }
    }

    else if (strcmp(*ap,"hybrid")==0 || strcmp(*ap,"tempered-hybrid")==0
          || strcmp(*ap,"spiral")==0 || strcmp(*ap,"double-spiral")==0)
    {
      if (strcmp(*ap,"hybrid")==0)
      {
        ops->op[o].type = 'H';
      }
      else
      {
        ops->op[o].type = strcmp(*ap,"tempered-hybrid")==0 ? 'T'
                        : strcmp(*ap,"spiral")==0 ? '@' : '^';

        ap += 1;
        if (!*ap || !strchr("0123456789+-.",**ap)) usage();
        
        if ((ops->op[o].temper_factor = atof(*ap))<=0) usage();
      }

      ops->op[o].stepsize_adjust = 1;
      ops->op[o].stepsize_alpha = 0;
      ops->op[o].in_steps = 0;
      ops->op[o].window = 1;
      ops->op[o].jump = 1;
      ops->op[o].r_update = 0;
      ops->op[o].firsti = -1;

      ap += 1;

      if (!*ap || !strchr("0123456789+-.",**ap)) usage();

      if ((ops->op[o].steps = atoi(*ap))<=0) usage();
      if (strchr(*ap,'/')!=0)
      { s = strchr(*ap,'/')+1;
        if ((ops->op[o].in_steps = atoi(s))<=0) usage();
        if (strchr(s,':')!=0)
        { if ((ops->op[o].jump = atoi(strchr(s,':')+1))<=0) usage();
          if (ops->op[o].steps%ops->op[o].jump!=0)
          { fprintf (stderr,
             "Maximum number of steps must be multiple of jump\n");
            exit(1);
          }
        }
      } 
      else if (strchr(*ap,':')!=0)
      { s = strchr(*ap,':')+1;
        if ((ops->op[o].window = atoi(s))<=0) usage();
        if (strchr(s,':')!=0)
        { if ((ops->op[o].jump = atoi(strchr(s,':')+1))<=0) usage();
          if (ops->op[o].steps%ops->op[o].jump!=0)
          { fprintf(stderr,"Total number of steps must be multiple of jump\n");
            exit(1);
          }
        }
        if (ops->op[o].window > ops->op[o].steps/ops->op[o].jump + 1)
        { fprintf(stderr,"Window can't be bigger than whole trajectory\n");
          exit(1);
        }
      } 
      ap += 1;

      if (*ap && strchr("0123456789+-.^",**ap))
      { if (**ap=='^' && ops->op[o].type=='H')
        { ops->op[o].options |= 2;
          if ((*ap)[1]==0 || (*ap)[1]==':')
          { ops->op[o].stepsize_adjust = 1;
          }
          else
          { if ((ops->op[o].stepsize_adjust = atof(*ap+1))==0) usage();
          }
        }
        else
        { ops->op[o].options &= ~2;
          if ((ops->op[o].stepsize_adjust = atof(*ap))==0) usage();
        }
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].stepsize_alpha = atof(strchr(*ap,':')+1))==0) usage();
        }
        ap += 1;
      }

      if (*ap && strchr("0123456789",**ap))
      { ops->op[o].firsti = atoi(*ap);
        ops->op[o].lasti = ops->op[o].firsti;
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].lasti = atoi(strchr(*ap,':')+1)) < ops->op[o].firsti)
          { usage();
          }
        }
        ap += 1;
      }
      else if (*ap && **ap>='A' && **ap<='Z' && *(*ap+1)==0)
      { ops->op[o].firsti = -(**ap-'A'+2);
        ops->op[o].lasti = 0;
        ap += 1;
      }

      if (ops->op[o].type!='H')
      { if (ops->op[o].in_steps!=0)
        { fprintf(stderr,
 "The max-steps/max-ok form is allowed only for plain hybrid Monte Carlo\n");
          exit(1);
        }
      }      

      if (ops->op[o].type=='@' || ops->op[o].type=='^')
      { if (ops->op[o].firsti!=-1)
        { fprintf(stderr,
         "The first:last option is not allowed for spiral and double-spiral\n");
          exit(1);
        }
        if (ops->op[o].jump!=1)
        { fprintf(stderr,
           "Jump must be one for spiral and double-spiral operations\n");
          exit(1);
        }
        if (ops->op[o].window!=1)
        { fprintf(stderr,
           "Window size must be one for spiral and double-spiral operations\n");
          exit(1);
        }
      }
    }

    else if (strcmp(*ap,"repeat")==0)
    {
      ops->op[o].type = 'R';

      ap += 1;

      if (!*ap || !strchr("0123456789",**ap)) usage();

      if ((ops->op[o].repeat_count = atoi(*ap++))<=0) usage();

      depth += 1;
    }

    else if (strcmp(*ap,"set-adaptive-factor")==0)
    {
      ops->op[o].type = '&';

      ap += 1;

      if (*ap==0 || (ops->op[o].adapt_set_value = atof(*ap)) <= 0) usage();
      ap += 1;
    }

    else if (strcmp(*ap,"multiply-stepsizes")==0)
    {
      ops->op[o].type = 'x';

      ap += 1;

      if (*ap==0 || (ops->op[o].stepsize_adjust = atof(*ap)) <= 0) usage();
      ap += 1;

      ops->op[o].firsti = -1;

      if (*ap && strchr("0123456789",**ap))
      { ops->op[o].firsti = atoi(*ap);
        ops->op[o].lasti = ops->op[o].firsti;
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].lasti = atoi(strchr(*ap,':')+1)) < ops->op[o].firsti)
          { usage();
          }
        }
        ap += 1;
      }
      else if (*ap && **ap>='A' && **ap<='Z' && *(*ap+1)==0)
      { ops->op[o].firsti = -(**ap-'A'+2);
        ops->op[o].lasti = 0;
        ap += 1;
      }
    }

    else if (strcmp(*ap,"slevel")==0)
    {
      ops->op[o].type = 'v';
      ops->op[o].op_slevel_move = 0;
      ops->op[o].op_slevel_random = 2;

      ap += 1;
      if (*ap && strchr("0123456789.-",**ap))
      { ops->op[o].op_slevel_move = atof(*ap);
        if (ops->op[o].op_slevel_move < -1
         || ops->op[o].op_slevel_move > 1) usage();
        ap += 1;
        ops->op[o].op_slevel_random = 0;
        if (*ap && strchr("0123456789.",**ap))
        { ops->op[o].op_slevel_random = atof(*ap);
          if (ops->op[o].op_slevel_random < 0
           || ops->op[o].op_slevel_random > 1) usage();
           ap += 1;
        }
      }
    }

    else if (strcmp(*ap,"temp-trans")==0)
    {
      ops->op[o].type = 't';

      ap += 1;

      depth += 1;
    }

    else if (strcmp(*ap,"sim-temp")==0)
    { ops->op[o].type = 's';
      ap += 1;
    }

    else if (strcmp(*ap,"set-temp")==0)
    { ops->op[o].type = '~';
      ap += 1;

      if (!*ap || !strchr("0123456789",**ap)) usage();
      ops->op[o].op_set_temp = atoi(*ap++);
    }

    else if (strcmp(*ap,"rand-dir")==0)
    { 
      ops->op[o].type = 'b';
      ap += 1;
    }

    else if (strcmp(*ap,"neg-dir")==0)
    { 
      ops->op[o].type = 'n';
      ap += 1;
    }

    else if (strcmp(*ap,"AIS")==0 || strcmp(*ap,"ais")==0)
    { ops->op[o].type = 'a';
      ap += 1;
    }

    else if (strcmp(*ap,"plot")==0)
    { 
      ops->op[o].type = 'p';
      ap += 1;
    }

    else if (strcmp(*ap,"multiply-momentum")==0)
    { 
      double d;

      ops->op[o].type = '*';
      ap += 1;

      if (!*ap || !strchr("0123456789.",**ap)) usage();
      if ((d = atof(*ap++))<=0) usage();
      ops->op[o].heatbath_decay = d-1;
    }

    else if (strcmp(*ap,"set-momentum")==0)
    { 
      double d;

      ops->op[o].type = '=';
      ap += 1;

      if (!*ap || !strchr("+-0123456789.",**ap)) usage();
      ops->op[o].heatbath_decay = atof(*ap++);
    }

    else if (strcmp(*ap,"set-value")==0)
    { 
      double d;

      ops->op[o].type = ':';
      ap += 1;

      if (!*ap || !strchr("+-0123456789.",**ap)) usage();
      ops->op[o].heatbath_decay = atof(*ap++);

      if (*ap && strchr("0123456789",**ap))
      { ops->op[o].firsti = atoi(*ap);
        ops->op[o].lasti = ops->op[o].firsti;
        if (strchr(*ap,':')!=0)
        { if ((ops->op[o].lasti = atoi(strchr(*ap,':')+1)) < ops->op[o].firsti)
          { usage();
          }
        }
        ap += 1;
      }
      else if (*ap && **ap>='A' && **ap<='Z' && *(*ap+1)==0)
      { ops->op[o].firsti = -(**ap-'A'+2);
        ops->op[o].lasti = 0;
        ap += 1;
      }
    }

    else if (**ap>='a' && **ap<='z' || **ap>='A' && **ap<='Z') 
    {                                       /* Application-specific operation */
      ops->op[o].type = 'A';
      ops->op[o].app_param = 0;
      ops->op[o].app_param2 = 0;

      strcpy(ops->op[o].appl,*ap);

      ap += 1;

      if (*ap && strchr("0123456789+-.",**ap))
      { ops->op[o].app_param = atof(*ap);
        ap += 1;
      }

      if (*ap && strchr("0123456789+-.",**ap))
      { ops->op[o].app_param2 = atof(*ap);
        ap += 1;
      }
    }

    else
    { usage();
    }

    o += 1;
  }

  have_traj = 0;

  if (*ap)
  {
    have_traj = 1;
    traj->frac = 0;
    ap += 1;
 
    if (*ap==0) usage();
  
    if (strcmp(*ap,"leapfrog")==0)
    {
      traj->type = 'L';
      traj->halfp = 1;
      traj->N_approx = 1;

      ap += 1;

      if (*ap && (strcmp(*ap,"halfp")==0 || strcmp(*ap,"firstp")==0))
      { traj->halfp = 1;
        ap += 1;
      }
      else if (*ap && (strcmp(*ap,"halfq")==0 || strcmp(*ap,"firstq")==0))
      { traj->halfp = 0;
        ap += 1;
      }

      if (*ap && strchr(*ap,'.')==0)
      { if ((traj->N_approx = atoi(*ap++))==0) 
        { usage();
        }
        if (*ap)
        { if (strlen(*ap)>200) 
          { fprintf(stderr,"Approximation file name is too long\n");
            exit(1);
          }
          strcpy(traj->approx_file,*ap);
          ap += 1;
        }
        else
        { traj->approx_file[0] = 0;
        }
      }
    }

    else if (strcmp(*ap,"opt2")==0)
    { 
      traj->type = '2';
      traj->rev_sym = 0;
      traj->halfp = 1;

      ap += 1;

      if (*ap && strcmp(*ap,"rev")==0)
      { traj->rev_sym = -1;
        ap += 1;
      }
      else if (*ap && strcmp(*ap,"sym")==0)
      { traj->rev_sym = 1;
        ap += 1;
      }

      if (*ap && strcmp(*ap,"firstp")==0)
      { traj->halfp = 1;
        ap += 1;
      }
      else if (*ap && strcmp(*ap,"firstq")==0)
      { traj->halfp = 0;
        ap += 1;
      }

      traj->param = 1 - sqrt(2.0)/2;
    }

    else if (strcmp(*ap,"gen2")==0)
    { 
      traj->type = 'G';
      traj->rev_sym = 0;
      traj->halfp = 1;

      ap += 1;

      if (*ap && strcmp(*ap,"rev")==0)
      { traj->rev_sym = -1;
        ap += 1;
      }
      else if (*ap && strcmp(*ap,"sym")==0)
      { traj->rev_sym = 1;
        ap += 1;
      }

      if (*ap && strcmp(*ap,"firstp")==0)
      { traj->halfp = 1;
        ap += 1;
      }
      else if (*ap && strcmp(*ap,"firstq")==0)
      { traj->halfp = 0;
        ap += 1;
      }

      if (!*ap || !strchr("0123456789.+-",**ap)) usage();

      traj->param = atof(*ap);

      ap += 1;
    }

    else if (strcmp(*ap,"opt4")==0)
    { 
      traj->type = '4';
      traj->rev_sym = 0;
      traj->halfp = 0; /* Not used, but set here to something definite to allow
                          possible later use without invalidating log files. */
      ap += 1;

      if (*ap && strcmp(*ap,"rev")==0)
      { traj->rev_sym = -1;
        ap += 1;
      }
      else if (*ap && strcmp(*ap,"sym")==0)
      { traj->rev_sym = 1;
        ap += 1;
      }
    }

    else 
    { fprintf(stderr,"Unknown trajectory type: %s\n",*ap);
      exit(1);
    }

    if (*ap)
    { traj->frac = atof(*ap);
      if (traj->frac==0 && strcmp(*ap,"0")!=0 || traj->frac<0 || traj->frac>1)
      { usage();
      }
      traj->frac = 1-traj->frac;
      ap += 1;
    }

    if (*ap) usage();
  }

  /* Write specifications to log file. */

  if (o>0)
  { 
    logf.header.type = 'o';
    logf.header.index = index;
    logf.header.size = sizeof *ops;
    log_file_append (&logf, ops);
  }

  if (have_traj || logg.data['t'])
  { 
    if (traj->type==0)
    { traj->halfp = 1;
      traj->N_approx = 1;
    }
    logf.header.type = 't';
    logf.header.index = index;
    logf.header.size = sizeof *traj;
    log_file_append (&logf, traj);
  }

  log_file_close (&logf);

  exit(0);
}


/* DISPLAY SPECIFICATIONS. */

static void display_specs
( log_gobbled *logg
)
{
  mc_ops *ops;
  mc_traj *traj;
  int depth;
  int o;

  if (logg->data['o']!=0)
  {
    printf("\nOperations making up each iteration:\n\n");

    ops = logg->data['o'];
    depth = 0;
  
    for (o = 0; o<Max_mc_ops && ops->op[o].type!=0; o++)
    {
      printf(" %*s", 2 * (depth - (ops->op[o].type=='E')), "");

      switch (ops->op[o].type)
      {
        case 'E':
        { printf(" end\n");
          depth -= 1;
          break;
        }

        case 'B':
        { printf(" heatbath");
          if (ops->op[o].heatbath_decay!=0)
          { printf(" %.6f",ops->op[o].heatbath_decay);
          }
          printf("\n");
          break;
        }

        case 'r':
        { printf(" radial-heatbath\n");
          break;
        }

        case 'X':
        { printf(" mix-momentum %.6f\n",ops->op[o].heatbath_decay);
          break;
        }

        case 'N':
        { printf(" negate\n");
          break;
        }
  
        case 'M': case 'G':
        { printf (ops->op[o].type=='M' ? " metropolis" : " rgrid-met");
          if (ops->op[o].b_accept)
          { printf(" -b");
          }
          if (ops->op[o].stepsize_alpha!=0)
          { printf(" %.4f:%.4f", ops->op[o].stepsize_adjust,
                                 ops->op[o].stepsize_alpha);
          }
          else if (ops->op[o].stepsize_adjust!=1)
          { printf(" %.4f", ops->op[o].stepsize_adjust);
          }
          printf("\n");
          break;
        }

        case 'l':
        { printf (" slice");
          if (ops->op[o].g_shrink==1)
          { printf(" -g");
          }
          else if (ops->op[o].g_shrink==2)
          { printf(" -G");
          }
          else if (ops->op[o].g_shrink!=0)
          { printf(" -???");
          }
          if (ops->op[o].stepsize_alpha!=0)
          { printf(" %.4f:%.4f", ops->op[o].stepsize_adjust,
                                 ops->op[o].stepsize_alpha);
          }
          else if (ops->op[o].stepsize_adjust!=1)
          { printf(" %.4f", ops->op[o].stepsize_adjust);
          }
          printf("\n");
          break;
        }

        case 'u':
        { printf (" slice-gaussian");
          if (ops->op[o].e_shrink==1)
          { printf(" -e");
          }
          else if (ops->op[o].g_shrink!=0)
          { printf(" -???");
          }
          if (ops->op[o].stepsize_alpha!=0)
          { printf(" %.4f:%.4f", ops->op[o].stepsize_adjust,
                                 ops->op[o].stepsize_alpha);
          }
          else if (ops->op[o].stepsize_adjust!=1)
          { printf(" %.4f", ops->op[o].stepsize_adjust);
          }
          printf("\n");
          break;
        }

        case 'm': case 'g':
        { printf(ops->op[o].type=='m' ? " met-1" : " rgrid-met-1");
          if (ops->op[o].b_accept)
          { printf(" -b");
          }
          if (ops->op[o].r_update)
          { printf(" -r");
          }
          if (ops->op[o].stepsize_alpha!=0)
          { printf(" %.4f:%.4f", ops->op[o].stepsize_adjust,
                                 ops->op[o].stepsize_alpha);
          }
          else if (ops->op[o].stepsize_adjust!=1 || ops->op[o].firsti!=-1)
          { printf(" %.4f", ops->op[o].stepsize_adjust);
          }
          if (ops->op[o].firsti>=0)
          { printf(" %d",ops->op[o].firsti);
            if (ops->op[o].lasti!=ops->op[o].firsti)
            { printf(":%d",ops->op[o].lasti);
            }
          }
          else if (ops->op[o].firsti<=-2)
          { printf(" %c",'A'+(-2-ops->op[o].firsti));
          }
          printf("\n");
          break;
        }

        case 'U': case '1':
        { printf(ops->op[o].type=='U' ? " gaussian-gibbs" : " binary-gibbs");
          if (ops->op[o].r_update)
          { printf(" -r");
          }
          if (ops->op[o].firsti>=0)
          { printf(" %d",ops->op[o].firsti);
            if (ops->op[o].lasti!=ops->op[o].firsti)
            { printf(":%d",ops->op[o].lasti);
            }
          }
          else if (ops->op[o].firsti<=-2)
          { printf(" %c",'A'+(-2-ops->op[o].firsti));
          }
          printf("\n");
          break;
        }
  
        case 'D': case 'P': case 'i': case 'o':
        { printf (" %s", ops->op[o].type=='D' ? "dynamic" 
                       : ops->op[o].type=='P' ? "permuted-dynamic"
                       : ops->op[o].type=='i' ? "slice-inside" 
                       : ops->op[o].type=='o' ? "slice-outside" : NULL);
          if (ops->op[o].options & 1)
          { printf(" -D");
          }
          if (ops->op[o].adapt_target!=0)
          { printf(" -a%.2f/%.3f", ops->op[o].adapt_target, 
                                   ops->op[o].adapt_rate);
          }
          printf (" %d", ops->op[o].steps);
          if (ops->op[o].type=='o' && ops->op[o].in_steps!=ops->op[o].steps)
          { printf ("/%d", ops->op[o].in_steps);
          }

          printf(" %s", ops->op[o].options&2 ? "^" : "");
          if (ops->op[o].stepsize_alpha!=0)
          { printf("%.4f:%.4f", ops->op[o].stepsize_adjust,
                                ops->op[o].stepsize_alpha);
          }
          else if (ops->op[o].stepsize_adjust!=1)
          { printf("%.4f", ops->op[o].stepsize_adjust);
          }
          printf("\n");
          break;
        }
  
        case 'H': 
        { printf(" hybrid %d",ops->op[o].steps);
          if (ops->op[o].in_steps!=0)
          { printf("/%d",ops->op[o].in_steps);
          }
          else
          { if (ops->op[o].window!=1 || ops->op[o].jump!=1)
            { printf(":%d",ops->op[o].window);
            }
          }
          if (ops->op[o].jump!=1)
          { printf(":%d",ops->op[o].jump);
          }
          printf(" %s", ops->op[o].options&2 ? "^" : "");
          if (ops->op[o].stepsize_alpha!=0)
          { printf("%.4f:%.4f", ops->op[o].stepsize_adjust,
                                ops->op[o].stepsize_alpha);
          }
          else if (ops->op[o].stepsize_adjust!=1 || ops->op[o].firsti!=-1)
          { printf("%.4f",ops->op[o].stepsize_adjust);
          }
          if (ops->op[o].firsti>=0)
          { printf(" %d",ops->op[o].firsti);
            if (ops->op[o].lasti!=ops->op[o].firsti)
            { printf(":%d",ops->op[o].lasti);
            }
          }
          else if (ops->op[o].firsti<=-2)
          { printf(" %c",'A'+(-2-ops->op[o].firsti));
          }
          printf("\n");
          break;
        }

        case 'S':
        { printf(" slice-1");
          if (ops->op[o].r_update)
          { printf(" -r");
          }
          if (ops->op[o].s_factor!=0)
          { if (ops->op[o].s_threshold==0)
            { printf(" -s %d",ops->op[o].s_factor);
            }
            else
            { printf(" -s %d/%f",ops->op[o].s_factor,ops->op[o].s_threshold);
            }
          }
          if (ops->op[o].stepsize_alpha!=0)
          { printf(" %.4f:%.4f",ops->op[o].stepsize_adjust,
                                ops->op[o].stepsize_alpha);
          }
          else if (ops->op[o].stepsize_adjust!=1 
                || ops->op[o].steps!=0
                || ops->op[o].firsti!=-1)
          { printf(" %.4f",ops->op[o].stepsize_adjust);
          }
          if (ops->op[o].steps!=0 || ops->op[o].firsti!=-1)
          { printf(" %d",ops->op[o].steps);
          }
          if (ops->op[o].firsti>=0)
          { printf(" %d",ops->op[o].firsti);
            if (ops->op[o].lasti!=ops->op[o].firsti)
            { printf(":%d",ops->op[o].lasti);
            }
          }
          else if (ops->op[o].firsti<=-2)
          { printf(" %c",'A'+(-2-ops->op[o].firsti));
          }
          printf("\n");
          break;
        }
    
        case 'O':
        { printf(" slice-over");
          if (ops->op[o].r_update)
          { printf(" -r");
          }
          if (ops->op[o].refinements!=0
           || ops->op[o].refresh_prob!=0 
           || ops->op[o].stepsize_alpha!=0
           || ops->op[o].stepsize_adjust!=1 
           || ops->op[o].firsti!=-1)
          { printf(" %d",ops->op[o].refinements);
          }
          if (ops->op[o].refresh_prob!=0 
           || ops->op[o].stepsize_alpha!=0
           || ops->op[o].stepsize_adjust!=1 
           || ops->op[o].firsti!=-1)
          { printf(" %.5f",ops->op[o].refresh_prob);
          }
          if (ops->op[o].stepsize_alpha!=0)
          { printf(" %.4f:%.4f",ops->op[o].stepsize_adjust,
                                ops->op[o].stepsize_alpha);
          }
          else if (ops->op[o].stepsize_adjust!=1 
                || ops->op[o].steps!=0
                || ops->op[o].firsti!=-1)
          { printf(" %.4f",ops->op[o].stepsize_adjust);
          }
          if (ops->op[o].steps!=0 || ops->op[o].firsti!=-1)
          { printf(" %d",ops->op[o].steps);
          }
          if (ops->op[o].firsti>=0)
          { printf(" %d",ops->op[o].firsti);
            if (ops->op[o].lasti!=ops->op[o].firsti)
            { printf(":%d",ops->op[o].lasti);
            }
          }
          else if (ops->op[o].firsti<=-2)
          { printf(" %c",'A'+(-2-ops->op[o].firsti));
          }
          printf("\n");
          break;
        }
  
        case 'T': case '@': case '^':
        { printf (" %s %.6f %d",
                  ops->op[o].type=='T' ? "tempered-hybrid" 
                   : ops->op[o].type=='@' ? "spiral" : "double-spiral",
                  ops->op[o].temper_factor, ops->op[o].steps);
          if (ops->op[o].window!=1)
          { printf(":%d",ops->op[o].window);
            if (ops->op[o].jump!=0)
            { printf(":%d",ops->op[o].jump);
            }
          }
          if (ops->op[o].stepsize_alpha!=0)
          { printf(" %.4f:%.4f",ops->op[o].stepsize_adjust,
                                ops->op[o].stepsize_alpha);
          }
          else if (ops->op[o].stepsize_adjust!=1)
          { printf(" %.4f",ops->op[o].stepsize_adjust);
          }
          printf("\n");
          break;
        }

        case 's':
        { printf(" sim-temp\n");
          break;
        }

        case '~':
        { printf(" set-temp %d\n",ops->op[o].op_set_temp);
          break;
        }

        case 'b':
        { printf(" rand-dir\n");
          break;
        }

        case 'n':
        { printf(" neg-dir\n");
          break;
        }
  
        case 'A':
        { printf(" %s",ops->op[o].appl);
          if (ops->op[o].app_param!=0) printf(" %.4f",ops->op[o].app_param);
          if (ops->op[o].app_param2!=0) printf(" %.4f",ops->op[o].app_param2);
          printf("\n");
          break;
        }
  
        case 'R':
        { printf(" repeat %d",ops->op[o].repeat_count);
          printf("\n");
          depth += 1;
          break;
        }

        case '&':
        { printf(" set-adaptive-factor %f",ops->op[o].adapt_set_value);
          printf("\n");
          break;
        }

        case 'x':
        { printf(" multiply-stepsizes %f",ops->op[o].stepsize_adjust);
          if (ops->op[o].firsti>=0)
          { printf(" %d",ops->op[o].firsti);
            if (ops->op[o].lasti!=ops->op[o].firsti)
            { printf(":%d",ops->op[o].lasti);
            }
          }
          else if (ops->op[o].firsti<=-2)
          { printf(" %c",'A'+(-2-ops->op[o].firsti));
          }
          printf("\n");
          break;
        }

        case 'v':
        { printf(" slevel");
          if (ops->op[o].op_slevel_random!=2)
          { printf(" %f",ops->op[o].op_slevel_move);
            if (ops->op[o].op_slevel_random!=0)
            { printf(" %f",ops->op[o].op_slevel_random);
            }
          }
          printf("\n");
          break;
        }
  
        case 't':
        { printf(" temp-trans\n");
          depth += 1;
          break;
        }

        case 'a':
        { printf(" AIS\n");
          break;
        }

        case '*':
        { printf(" multiply-momentum %.8f\n",
             (double)ops->op[o].heatbath_decay+1);
          break;
        }

        case '=':
        { printf(" set-momentum %.4f\n",
             (double)ops->op[o].heatbath_decay);
          break;
        }

        case ':':
        { printf (" set-value %.4f", (double)ops->op[o].heatbath_decay);
          if (ops->op[o].firsti>=0)
          { printf(" %d",ops->op[o].firsti);
            if (ops->op[o].lasti!=ops->op[o].firsti)
            { printf(":%d",ops->op[o].lasti);
            }
          }
          else if (ops->op[o].firsti<=-2)
          { printf(" %c",'A'+(-2-ops->op[o].firsti));
          }
          printf("\n");
          break;
        }

        case 'p':
        { printf(" plot\n");
          break;
        }
  
        default:
        { printf(" unknown operation: %c\n",ops->op[o].type);
          break;
        }
      }
    }
  }

  if (logg->data['t']!=0 && (traj = logg->data['t'])->type!=0)
  {
    printf("\nMethod for computing trajectories:\n\n");
    
    switch (traj->type)
    {
      case 'L': 
      { printf ("  leapfrog %s", traj->halfp ? "halfp" : "halfq");
        if (traj->N_approx!=1)
        { printf (" %d %s", traj->N_approx, traj->approx_file);
        }
        break;
      }

      case '2':
      { printf ("  opt2 %s%s", 
          traj->rev_sym==-1 ? "rev " : traj->rev_sym==1 ? "sym " : "",
          traj->halfp ? "firstp" : "firstq");
        break;
      }

      case 'G':
      { printf ("  gen2 %s%s %.15e", 
          traj->rev_sym==-1 ? "rev " : traj->rev_sym==1 ? "sym " : "",
          traj->halfp ? "firstp" : "firstq", traj->param);
        break;
      }

      case '4':
      { printf ("  opt4 %s",
          traj->rev_sym==-1 ? "rev " : traj->rev_sym==1 ? "sym " : "");
        break;
      }

      default:
      { printf("  unknown method: %c",traj->type);
        goto out;
      }
    }

    if (traj->frac!=0)
    { printf(" %.3f",1-traj->frac);
    }

  out:
    printf("\n");
  }
 
  printf("\n");
}


/* DISPLAY USAGE MESSAGE AND EXIT. */

static void usage(void)
{
  fprintf(stderr, 
"Usage: mc-spec log-file { operation-spec } [ / trajectory-spec ]\n");

  fprintf(stderr,
"   or: mc-spec log-file [ index | \"all\" ] (to display stored specifications)\n");

  exit(1);
}
