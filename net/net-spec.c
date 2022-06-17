/* NET-SPEC.C - Program to specify a new network (and create log file). */

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
#include "prior.h"
#include "data.h"
#include "model.h"
#include "net.h"


static void usage(void);
static void print_config (net_config *, int);

static int show_config_details;


/* TAKE A CONFIGURATION FILE SPEC FROM CONFIG ARGUMENT. */

static int fileix = 1;  /* don't start at 0, since 0 is used for "none" */

static int take_config (char *p, net_flags *flgs)
{ int len = strlen(p+6);
  int strt = fileix;
  if (strt+len+1>Config_file_space)
  { fprintf(stderr,"Config specs occupy too much space\n");
    exit(1);
  }
  strcpy(flgs->config_files+strt,p+7);
  fileix += len+1;
  return strt;
}


/* MAIN PROGRAM. */

int main
( int argc,
  char **argv
)
{
  static net_arch    arch,   *a = &arch;   /* Static so irrelevant fields are */
  static net_priors  priors, *p = &priors; /*   set to zero (just in case)    */
  static net_flags   flags,  *flgs = &flags;
  int any_flags;

  log_file logf;
  log_gobbled logg;

  char ps[Config_file_space];
  char **ap;
  int i, j, l;
  int nsqi;

  /* Look for log file name. */

  if (argc<2) usage();

  logf.file_name = argv[1];

  /* See if we are to display specifications for existing network. */

  int show_sizes = 0;
  int show_configs = 0;

  while (argc>2)
  { if (strcmp(argv[argc-1],"config")==0 || strcmp(argv[argc-1],"config+")==0)
    { if (show_configs) usage();
      show_configs = 1;
      show_config_details = strcmp(argv[argc-1],"config+")==0;
      argc -= 1;
    }
    else if (strcmp(argv[argc-1],"sizes")==0)
    { if (show_sizes) usage();
      show_sizes = 1;
      argc -= 1;
    }
    else
    { break;
    }
  }

  if (argc==2 || show_configs || show_sizes)
  {
    if (argc>2) usage();

    /* Open log file and gobble up initial records. */
  
    log_file_open(&logf,0);

    log_gobble_init(&logg,0);
    net_record_sizes(&logg);

    if (!logf.at_end && logf.header.index==-1)
    { log_gobble(&logf,&logg);
    }
  
    /* Display architecture record. */
  
    printf("\n");
  
    if ((a = logg.data['A'])==0)
    { printf ("No architecture specification found\n\n");
      exit(0);
    }

    flgs = logg.data['F'];

    printf("Network Architecture:\n\n");

    printf ("  Input layer:     size %d\n", a->N_inputs);  

    for (l = 0; l<a->N_layers; l++) 
    { printf("  Hidden layer %d:  size %-4d",l,a->N_hidden[l]);
      if (a->layer_type[l]==Tanh_type)            printf("  tanh");
      else if (a->layer_type[l]==Softplus_type)   printf("  softplus");
      else if (a->layer_type[l]==Softplus0_type)  printf("  softplus0");
      else if (a->layer_type[l]==Identity_type)   printf("  identity");
      else if (a->layer_type[l]>Normalize_base)
      { int c = a->layer_type[l]-Normalize_base;
        if (c==1)
        { printf("  normalize");
        }
        else
        { printf("  normalize/%d",c);
        }
      }
      else                                        printf("  UNKNOWN TYPE!");
      printf("\n");
    }

    printf ("  Output layer:    size %d\n", a->N_outputs);
  
    printf("\n");
    
    /* Display priors record. */
  
    printf("\n");
  
    if ((p = logg.data['P'])==0)
    { printf("No prior specifications found\n\n");
      exit(0);
    }
  
    printf("Prior Specifications:\n");

    int g = 1;
  
    if (a->has_ti) 
    { printf("\n  %2d. Input Offsets:          %s\n",g++,prior_show(ps,p->ti));
    }
  
    nsqi = 0;
    for (l = 0; l<a->N_layers; l++)
    { printf("\n         Hidden Layer %d\n\n",l);
      int lp;
      for (lp = 0; lp<l; lp++)
      { if ((a->has_nsq[l]>>lp) & 1)
        { printf("  %2d. Hidden%d-Hidden Weights: %s", g++,
                 lp, prior_show(ps,p->nsq[nsqi]));
          if (flgs && flgs->nonseq_config[nsqi])
          { printf("  config:%s", flgs->config_files+flgs->nonseq_config[nsqi]);
          }
          nsqi += 1;
          printf("\n");
        }
      }
      if (l>0 && a->has_hh[l-1])
      { printf("  %2d. Hidden-Hidden Weights:  %s", g++,
               prior_show(ps,p->hh[l-1]));
        if (flgs && flgs->hidden_config[l])
        { printf("  config:%s", flgs->config_files+flgs->hidden_config[l]);
        }
        printf("\n");
      }
      if (a->has_ih[l]) 
      { printf("  %2d. Input-Hidden Weights:   %s", g++,
               prior_show(ps,p->ih[l]));
        if (flgs && list_flags (flgs->omit, a->N_inputs, 1<<(l+1), ps) > 0)
        { printf("  omit%s",ps);
        }
        if (flgs && flgs->input_config[l])
        { printf("  config:%s", flgs->config_files+flgs->input_config[l]);
        }
        printf("\n");
      }
      if (a->has_bh[l]) 
      { printf("  %2d. Hidden Biases:          %s", g++,
               prior_show(ps,p->bh[l]));
        if (flgs && flgs->bias_config[l])
        { printf("  config:%s", flgs->config_files+flgs->bias_config[l]);
        }
        printf("\n");
      }
      if (a->has_ah[l])
      { printf("  %2d. Hidden Adjustments:         %.2f\n", g++, p->ah[l]);
      }
      if (a->has_th[l]) 
      { printf("  %2d. Hidden Offsets:         %s\n", g++,
               prior_show(ps,p->th[l]));
      }
    }

    printf("\n         Output Layer\n\n");

    for (l = a->N_layers-1; l>=0; l--)
    { if (a->has_ho[l]) 
      { printf("  %2d. Hidden%d-Output Weights: %s", g++, l,
               prior_show(ps,p->ho[l]));
        int k = 2*a->N_layers-1-l;
        if (flgs && flgs->hidden_config[k])
        { printf("  config:%s", flgs->config_files+flgs->hidden_config[k]);
        }
        printf("\n");
      }
    }

    if (a->has_io) 
    { printf("  %2d. Input-Output Weights:   %s", g++, prior_show(ps,p->io));
      if (flgs && list_flags (flgs->omit, a->N_inputs, 1, ps) > 0)
      { printf("  omit%s",ps);
      }
      if (flgs && flgs->input_config[a->N_layers])
      { printf("  config:%s",
               flgs->config_files+flgs->input_config[a->N_layers]);
      }
      printf("\n");
    }

    if (a->has_bo) 
    { printf("  %2d. Output Biases:          %s", g++, prior_show(ps,p->bo));
      if (flgs && flgs->bias_config[a->N_layers])
      { printf("  config:%s",
               flgs->config_files+flgs->bias_config[a->N_layers]);
      }
      printf("\n");
    }

    if (a->has_ao)
    { printf("  %2d. Output Adjustments:         %.2f\n", g++, p->ao);
    }
  
    printf("\n");

    net_precomputed pre;
    unsigned total;

    if (show_sizes || show_configs)
    { total = net_setup_param_count (a, flgs, &pre);
    }

    if (show_sizes)
    { 
      printf("\nNumbers of parameters and connections in each group:\n\n");
      printf(" %-27s  offset  parameters  connections\n\n","");

      int pa, cn, offset=0, conns = 0;  
      char t[100];
        
      if (a->has_ti) 
      { pa = a->N_inputs, cn = pa;
        printf("  %-27s%7d  %7d    %9d\n","Input Offsets:", offset, pa, cn);
        offset += pa; conns += cn;
      }
    
      nsqi = 0;
      for (l = 0; l<a->N_layers; l++)
      { int lp;
        for (lp = 0; lp<l; lp++)
        { if ((a->has_nsq[l]>>lp) & 1)
          { sprintf(t, "Hidden%d-Hidden%d Weights:", lp, l);
            if (a->nonseq_config[nsqi])
            { pa = a->hidden_config[nsqi]->N_wts;
              cn = a->hidden_config[nsqi]->N_conn;
            }
            else
            { pa = a->N_hidden[lp]*a->N_hidden[lp];
              cn = pa;
            }
            printf("  %-27s%7d  %7d    %9d\n", t, offset, pa, cn);
            offset += pa; conns += cn;
            nsqi += 1;
          }
        }
        if (l>0 && a->has_hh[l-1])
        { sprintf(t, "Hidden%d-Hidden%d Weights:", l-1, l);
          if (a->hidden_config[l])
          { pa = a->hidden_config[l]->N_wts;
            cn = a->hidden_config[l]->N_conn;
          }
          else
          { pa = a->N_hidden[l-1]*a->N_hidden[l];
            cn = pa;
          }
          printf("  %-27s%7d  %7d    %9d\n", t, offset, pa, cn);
          offset += pa; conns += cn;
        }
        if (a->has_ih[l]) 
        { sprintf(t, "Input-Hidden%d Weights:", l);
          if (a->input_config[l])
          { pa = a->input_config[l]->N_wts;
            cn = a->input_config[l]->N_conn;
          }
          else
          { pa = 
             not_omitted(flgs?flgs->omit:0,a->N_inputs,1<<(l+1))*a->N_hidden[l];
            cn = pa;
          }
          printf("  %-27s%7d  %7d    %9d\n", t, offset, pa, cn);
          offset += pa; conns += cn;
        }
        if (a->has_bh[l]) 
        { sprintf(t, "Hidden%d Biases:", l);
          if (a->bias_config[l])
          { pa = a->bias_config[l]->N_wts;
            cn = a->bias_config[l]->N_conn;
          }
          else
          { pa = a->N_hidden[l];
            cn = pa;
          }
          printf("  %-27s%7d  %7d    %9d\n", t, offset, pa, cn);
          offset += pa; conns += cn;
        }
        if (a->has_th[l]) 
        { sprintf(t, "Hidden%d Offsets:", l);
          pa = a->N_hidden[l]; cn = pa;
          printf("  %-27s%7d  %7d    %9d\n", t, offset, pa, cn);
          offset += pa; conns += cn;
        }
      }

      for (l = a->N_layers-1; l>=0; l--)
      { if (a->has_ho[l]) 
        { sprintf(t, "Hidden%d-Output Weights:", l);
          if (a->hidden_config[2*a->N_layers-l-1])
          { pa = a->hidden_config[2*a->N_layers-l-1]->N_wts;
            cn = a->hidden_config[2*a->N_layers-l-1]->N_conn;
          }
          else
          { pa = a->N_hidden[l]*a->N_outputs;
            cn = pa;
          }
          printf("  %-27s%7d  %7d    %9d\n", t, offset, pa, cn);
          offset += pa; conns += cn;
        }
      }

      if (a->has_io) 
      { if (a->input_config[a->N_layers])
        { pa = a->input_config[a->N_layers]->N_wts;
          cn = a->input_config[a->N_layers]->N_conn;
        }
        else
        { pa = not_omitted(flgs?flgs->omit:0,a->N_inputs,1) * a->N_outputs;
          cn = pa;
        }
        printf("  %-27s%7d  %7d    %9d\n", "Input-Output Weights:",
               offset, pa, cn);
        offset += pa; conns += cn;
      }

      if (a->has_bo) 
      { if (a->bias_config[a->N_layers])
        { pa = a->bias_config[a->N_layers]->N_wts;
          cn = a->bias_config[a->N_layers]->N_conn;
        }
        else
        { pa = a->N_outputs;
          cn = pa;
        }
        printf("  %-27s%7d  %7d    %9d\n", "Output Biases:", offset, pa, cn);
        offset += pa; conns += cn;
      }

      printf("\n  %-27s%7s  %7d    %9d\n", "Total:", "", total, conns);
    }

    nsqi = 0;
    if (show_configs && flgs)
    { for (l = 0; l<a->N_layers; l++)
      { if (flgs->input_config[l])
        { printf("Hidden layer %d input weight configuration\n",l);
          print_config (net_config_read 
                         (flgs->config_files+flgs->input_config[l],
                          a->N_inputs, a->N_hidden[l]), 0);
        }
        int lp;
        for (lp = 0; lp<l; lp++)
        { if ((a->has_nsq[l]>>lp) & 1)
          { int c = flgs->nonseq_config[nsqi++];
            if (c)
            { printf("Hidden layer %d hidden%d weight configuration\n",l,lp);
              print_config (net_config_read 
                             (flgs->config_files+c,
                              a->N_hidden[lp], a->N_hidden[l]), 0);
            }
          }
        }
        if (flgs->hidden_config[l])
        { printf("Hidden layer %d hidden weight configuration\n",l);
          print_config (net_config_read 
                         (flgs->config_files+flgs->hidden_config[l],
                          a->N_hidden[l-1], a->N_hidden[l]), 0);
        }
        if (flgs->bias_config[l])
        { printf("Hidden layer %d bias configuration\n",l);
          print_config (net_config_read 
                         (flgs->config_files+flgs->bias_config[l],
                          -1, a->N_hidden[l]), 1);
        }
      }
      if (flgs->input_config[a->N_layers])
      { printf("Output layer input weight configuration\n");
        print_config (net_config_read 
                       (flgs->config_files+flgs->input_config[a->N_layers],
                        a->N_inputs, a->N_outputs), 0);
      }
      for (l = a->N_layers-1; l>=0; l--)
      { int k = 2*a->N_layers-1-l;
        if (flgs->hidden_config[k])
        { if (a->N_layers==1)
          { printf("Output layer hidden weight configuration\n");
          }
          else
          { printf("Output layer hidden layer %d weight configuration\n",l);
          }
          print_config (net_config_read 
                         (flgs->config_files+flgs->hidden_config[k],
                          a->N_hidden[l], a->N_outputs), 0);
        }
      }
      if (flgs->bias_config[a->N_layers])
      { printf("Output layer bias configuration\n");
        print_config (net_config_read 
                       (flgs->config_files+flgs->bias_config[a->N_layers],
                        -1, a->N_outputs), 1);
      }
    }
  
    log_file_close(&logf);
  
    exit(0);
  }

  /* Otherwise, figure out architecture and priors from program arguments. */

  any_flags = 0;

  a->N_layers = 0;
  
  ap = argv+2;

  if (*ap==0 || (a->N_inputs = atoi(*ap++))<=0) usage();

  while (*ap!=0 && strcmp(*ap,"/")!=0)
  { 
    int size, type;
    int i;

    if ((size = atoi(*ap++))<=0) usage();

    if (*ap==0) usage();

    type = -1;

    while (*ap!=0 && (*ap)[0]>='a' && (*ap)[0]<='z')
    { if (strcmp(*ap,"tanh")==0)
      { if (type>=0) usage();
        type = Tanh_type;
      }
      else if (strcmp(*ap,"identity")==0)
      { if (type>=0) usage();
        type = Identity_type;
      }
      else if (strcmp(*ap,"softplus")==0)
      { if (type>=0) usage();
        type = Softplus_type;
      }
      else if (strcmp(*ap,"softplus0")==0)
      { if (type>=0) usage();
        type = Softplus0_type;
      }
      else if (strncmp(*ap,"normalize",9)==0)
      { if (strcmp(*ap,"normalize")==0)
        { type = Normalize_base + 1;
        }
        else
        { int c; char junk;
          if (sscanf(*ap,"normalize/%d%c",&c,&junk)!=1) usage();
          if (c<=0 || size%c!=0)
          { fprintf (stderr,
                     "Invalid number of channels for 'normalize' layer\n");
            exit(1);
          }
          if (c>100)
          { fprintf (stderr,
                     "Too many channels in 'normalize' layer (max 100)\n");
            exit(1);
          }
          type = Normalize_base + c;
        }
      }
      else
      { usage();
      }
      ap += 1;
    }

    if (*ap!=0 && strcmp(*ap,"/")!=0)  /* more to come, so a hidden layer */
    { 
      if (a->N_layers == Max_layers)
      { fprintf(stderr,"Too many layers specified (maximum is %d)\n",
                        Max_layers);
        exit(1);
      }
      a->N_hidden[a->N_layers] = size;
      a->layer_type[a->N_layers] = type==-1 ? Tanh_type : type;
      a->N_layers += 1;
    }

    else  /* last layer size, so this is the output layer */
    { 
      a->N_outputs = size;
      if (type!=-1) usage();
    }
  }

  if (*ap==0 || strcmp(*ap++,"/")!=0) usage();

  nsqi = 0;

  while (*ap!=0)
  {
    char *pr;
    char eq;
    int ls;

    pr = strchr(*ap,'=');
    if (pr==0) usage();
    pr += 1;
 
    l = -1;
    ls = -1;

    if (sscanf(*ap,"ti%c",&eq)==1 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { a->has_ti = 1;
        if (!prior_parse(&p->ti,pr)) usage();
      }
    }
    else if (sscanf(*ap,"ih%c",&eq)==1 && eq=='='
          || sscanf(*ap,"ih%d%c",&l,&eq)==2 && l>=0 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { if (l==-1) l = 0;
        if (l>=a->N_layers) 
        { fprintf(stderr,"Invalid layer number: %d\n",l); 
          exit(1); 
        }
        a->has_ih[l] = 1;
        if (!prior_parse(&p->ih[l],pr)) usage();
        if (*(ap+1)!=0 && strncmp(*(ap+1),"omit:",5)==0)
        { ap += 1;
          parse_flags (*ap+4, flgs->omit, a->N_inputs, 1<<(l+1));
          a->any_omitted[l] = 1;
          any_flags = 1;
        }
        else if (*(ap+1)!=0 && strncmp(*(ap+1),"config:",7)==0)
        { ap += 1;
          flgs->input_config[l] = take_config(*ap,flgs);
          any_flags = 1;
        }
      }
    }
    else if (sscanf(*ap,"bh%c",&eq)==1 && eq=='='
          || sscanf(*ap,"bh%d%c",&l,&eq)==2 && l>=0 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { if (l==-1) l = 0;
        if (l>=a->N_layers) 
        { fprintf(stderr,"Invalid layer number: %d\n",l); 
          exit(1); 
        }
        a->has_bh[l] = 1;
        if (!prior_parse(&p->bh[l],pr)) usage();
        if (*(ap+1)!=0 && strncmp(*(ap+1),"config:",7)==0)
        { ap += 1;
          flgs->bias_config[l] = take_config(*ap,flgs);
          any_flags = 1;
        }
      }
    }
    else if (sscanf(*ap,"th%c",&eq)==1 && eq=='='
          || sscanf(*ap,"th%d%c",&l,&eq)==2 && l>=0 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { if (l==-1) l = 0;
        if (l>=a->N_layers) 
        { fprintf(stderr,"Invalid layer number: %d\n",l); 
          exit(1); 
        }
        a->has_th[l] = 1;
        if (!prior_parse(&p->th[l],pr)) usage();
      }
    }
    else if (sscanf(*ap,"hh%c",&eq)==1 && eq=='='
          || sscanf(*ap,"hh%d%c",&l,&eq)==2 && l>=0 && eq=='='
          || sscanf(*ap,"h%dh%d%c",&ls,&l,&eq)==3 && ls>=0 && l>=0 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { if (l==-1) l = 1;
        if (ls==-1) ls = l-1;
        if (l==0 || l>=a->N_layers) 
        { fprintf(stderr,"Invalid layer number: %d\n",l); 
          exit(1); 
        }
        if (ls==l-1)
        { a->has_hh[l-1] = 1;
          if (!prior_parse(&p->hh[l-1],pr)) usage();
          if (*(ap+1)!=0 && strncmp(*(ap+1),"config:",7)==0)
          { ap += 1;
            flgs->hidden_config[l] = take_config(*ap,flgs);
            any_flags = 1;
          }
        }
        else
        { a->has_nsq[l] |= 1<<ls;
          if (!prior_parse(&p->nsq[nsqi],pr)) usage();
          if (*(ap+1)!=0 && strncmp(*(ap+1),"config:",7)==0)
          { ap += 1;
            flgs->nonseq_config[nsqi] = take_config(*ap,flgs);
            any_flags = 1;
          }
          nsqi += 1;
        }
      }
    }
    else if (sscanf(*ap,"ho%c",&eq)==1 && eq=='='
          || sscanf(*ap,"h%do%c",&l,&eq)==2 && l>=0 && eq=='='
          || sscanf(*ap,"ho%d%c",&l,&eq)==2 && l>=0 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { if (l==-1) l = a->N_layers-1;
        if (l<0 || l>=a->N_layers) 
        { fprintf(stderr,"Invalid layer number: %d\n",l); 
          exit(1); 
        }
        a->has_ho[l] = 1;
        if (!prior_parse(&p->ho[l],pr)) usage();
        if (*(ap+1)!=0 && strncmp(*(ap+1),"config:",7)==0)
        { ap += 1;
          flgs->hidden_config[2*a->N_layers-l-1] = take_config(*ap,flgs);
          any_flags = 1;
        }
      }
    }
    else if (sscanf(*ap,"io%c",&eq)==1 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { a->has_io = 1;
        if (!prior_parse(&p->io,pr)) usage();
        if (*(ap+1)!=0 && strncmp(*(ap+1),"omit:",5)==0)
        { ap += 1;
          parse_flags (*ap+4, flgs->omit, a->N_inputs, 1);
          a->any_omitted[a->N_layers] = 1;
          any_flags = 1;
        }
        else if (*(ap+1)!=0 && strncmp(*(ap+1),"config:",7)==0)
        { ap += 1;
          flgs->input_config[a->N_layers] = take_config(*ap,flgs);
          any_flags = 1;
        }
      }
    }
    else if (sscanf(*ap,"bo%c",&eq)==1 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { a->has_bo = 1;
        if (!prior_parse(&p->bo,pr)) usage();
        if (*(ap+1)!=0 && strncmp(*(ap+1),"config:",7)==0)
        { ap += 1;
          flgs->bias_config[a->N_layers] = take_config(*ap,flgs);
          any_flags = 1;
        }
      }
    }
    else if (sscanf(*ap,"ah%c",&eq)==1 && eq=='='
          || sscanf(*ap,"ah%d%c",&l,&eq)==2 && l>=0 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { if (l==-1) l = 0;
        if (l>=a->N_layers) 
        { fprintf(stderr,"Invalid layer number: %d\n",l); 
          exit(1); 
        }
        if (flgs->input_config[l]
         || flgs->hidden_config[l]
         || flgs->bias_config[l])
        { fprintf(stderr,
            "Adjustments not allowed for layer with configured weights\n");
          exit(1);
        }
        a->has_ah[l] = 1;
        if ((p->ah[l] = atof(pr))<=0) usage();
      }
    }
    else if (sscanf(*ap,"ao%c",&eq)==1 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { if (flgs->input_config[a->N_layers] 
         || flgs->bias_config[a->N_layers])
        { fprintf(stderr,
            "Adjustments not allowed for layer with configured weights\n");
          exit(1);
        }
        for (l = 0; l<a->N_layers; l++)
        { if (flgs->hidden_config[2*a->N_layers-1-l])
          { fprintf(stderr,
              "Adjustments not allowed for layer with configured weights\n");
            exit(1);
          }
        }
        a->has_ao = 1;
        if ((p->ao = atof(pr))<=0) usage();
      }
    }
    else
    { usage();
    }

    ap += 1;
  }

  if (*ap!=0) usage();

  if (p->ti.scale || p->ti.alpha[2]!=0)
  { fprintf(stderr,"Illegal prior for input offsets\n");
    exit(1); 
  }

  for (l = 0; l<a->N_layers; l++)
  { if (p->bh[l].scale || p->bh[l].alpha[2]!=0
     || p->th[l].scale || p->th[l].alpha[2]!=0)
    { fprintf(stderr,
       "Illegal prior for hidden biases or offsets (bh%d or th%d\n",l,l);
      exit(1); 
    }
  }

  if (p->bo.scale || p->bo.alpha[2]!=0)
  { fprintf(stderr,"Illegal prior for output biases\n");
    exit(1); 
  }

  nsqi = 0;
  for (l = 0; l<a->N_layers; l++)
  { if (flgs->input_config[l] && (p->ih[l].scale || p->ih[l].alpha[1]!=0))
    { fprintf (stderr, 
        "Illegal prior for weights with input configuration file (ih%d)\n",l);
      exit(1); 
    }
    if (l>0 && flgs->hidden_config[l] 
            && (p->hh[l-1].scale || p->hh[l-1].alpha[1]!=0))
    { fprintf (stderr,
        "Illegal prior for weights with hidden configuration file (hh%d)\n",l);
      exit(1); 
    }
    int lp;
    for (lp = 0; lp<l; lp++)
    { if ((a->has_nsq[l]>>lp) & 1)
      { if (flgs->nonseq_config[nsqi] 
             && (p->nsq[nsqi].scale || p->nsq[nsqi].alpha[1]!=0))
        { fprintf (stderr,
          "Illegal prior for weights with hidden configuration file (h%dh%d)\n",
           lp, l);
          exit(1); 
        }
        nsqi += 1;
      }
    }
  }

  if (flgs->input_config[a->N_layers] && (p->io.scale || p->io.alpha[1]!=0))
  { fprintf (stderr, 
      "Illegal prior for weights with input configuration file (io)\n");
    exit(1); 
  }
  for (l = 0; l<a->N_layers; l++)
  { if (flgs->hidden_config[2*a->N_layers-l-1] 
         && (p->ho[l].scale || p->ho[l].alpha[1]!=0))
    { fprintf (stderr,
        "Illegal prior for weights with hidden configuration file (h%do)\n",l);
      exit(1); 
    }
  }

  /* Create log file and write records. */

  log_file_create(&logf);

  logf.header.type = 'A';
  logf.header.index = -1;
  logf.header.size = sizeof *a;
  log_file_append(&logf,a);

  if (any_flags)
  { logf.header.type = 'F';
    logf.header.index = -1;
    logf.header.size = sizeof *flgs;
    log_file_append(&logf,flgs);
  }

  logf.header.type = 'P';
  logf.header.index = -1;
  logf.header.size = sizeof *p;
  log_file_append(&logf,p);

  log_file_close(&logf);

  exit(0);
}


/* PRINT WEIGHT CONFIGURATION. */

static void print_config (net_config *cf, int biases)
{ int k;
  printf("\n%d connections, %d %s\n\n",cf->N_conn,cf->N_wts,
                                       biases ? "biases" : "weights");
  for (k = 0; k<cf->N_conn; k++)
  { if (!biases) 
    { printf ("%6d ", cf->conn[k].s+1);
    }
    printf ("%6d %6d\n", cf->conn[k].d+1, cf->conn[k].w+1);
  }
  printf("\n");

  if (show_config_details)
  { 
    int i, r;

    /* For CPU: */

    if (CONFIG_OCT_S_8D_8W && cf->oct_s_8d_8w)
    { printf("oct_s_8d_8w:\n");
      for (i = 0; cf->oct_s_8d_8w[i].w >= 0; i++)
      { printf("%3d %3d-%-3d %3d-%-3d\n", cf->oct_s_8d_8w[i].s, 
                               cf->oct_s_8d_8w[i].d, cf->oct_s_8d_8w[i].d+7,
                               cf->oct_s_8d_8w[i].w, cf->oct_s_8d_8w[i].w+7);
      }
      printf("\n");
    }

    if (CONFIG_QUAD_S_4D_4W && cf->quad_s_4d_4w)
    { printf("quad_s_4d_4w:\n");
      for (i = 0; cf->quad_s_4d_4w[i].w >= 0; i++)
      { printf("%3d %3d-%-3d %3d-%-3d\n", cf->quad_s_4d_4w[i].s, 
                               cf->quad_s_4d_4w[i].d, cf->quad_s_4d_4w[i].d+3,
                               cf->quad_s_4d_4w[i].w, cf->quad_s_4d_4w[i].w+3);
      }
      printf("\n");
    }

    if (CONFIG_QUAD_S_4D_4W && cf->quad_s_4d_4w_2)
    { printf("quad_s_4d_4w_2 (pairs with same w):\n");
      for (i = 0; cf->quad_s_4d_4w_2[i].w >= 0; i++)
      { printf("%3d %3d-%-3d %3d-%-3d\n", cf->quad_s_4d_4w_2[i].s, 
                            cf->quad_s_4d_4w_2[i].d, cf->quad_s_4d_4w_2[i].d+3,
                            cf->quad_s_4d_4w_2[i].w, cf->quad_s_4d_4w_2[i].w+3);
      }
      printf("\n");
    }

    if (CONFIG_SINGLE4 && cf->single4_s)
    { printf("single4_s:\n");
      for (i = 0; cf->single4_s[i].w >= 0; i++)
      { printf("%3d %3d %3d\n",
                cf->single4_s[i].s, cf->single4_s[i].d, cf->single4_s[i].w);
      }
      printf("\n");
    }

    if (CONFIG_SINGLE4 && cf->single4_d)
    { printf("single4_d:\n");
      for (i = 0; cf->single4_d[i].w >= 0; i++)
      { printf("%3d %3d %3d\n",
                cf->single4_d[i].s, cf->single4_d[i].d, cf->single4_d[i].w);
      }
      printf("\n");
    }

    if (cf->single)
    { printf("single:\n");
      for (i = 0; cf->single[i].w >= 0; i++)
      { printf("%3d %3d %3d\n",
                cf->single[i].s, cf->single[i].d, cf->single[i].w);
      }
      printf("\n");
    }

    /* For GPU: */

    if (CONFIG_OCT_GPU_S_8D_8W_GRAD && cf->oct_s_8d_8w_wgpu)
    { printf("oct_s_8d_8w_wgpu:\n");
      printf("start indexes:");
      for (r = 0; r<NTH; r++)
      { printf(" %d",cf->start_oct_wgpu[r]);
      }
      printf("\n");
      i = 0;
      for (r = 0; r<NTH; r++)
      { printf("First weight %d mod %d:\n",r,NTH);
        while (cf->oct_s_8d_8w_wgpu[i].w >= 0)
        { printf("%3d %3d-%-3d %3d-%-3d\n", cf->oct_s_8d_8w_wgpu[i].s, 
                      cf->oct_s_8d_8w_wgpu[i].d, cf->oct_s_8d_8w_wgpu[i].d+7,
                      cf->oct_s_8d_8w_wgpu[i].w, cf->oct_s_8d_8w_wgpu[i].w+7);
          i += 1;
        }
        i += 1;
        printf("\n");
      }
      printf("\n");
    }

    if (CONFIG_QUAD_GPU_S_4D_4W_GRAD && cf->quad_s_4d_4w_wgpu)
    { printf("quad_s_4d_4w_wgpu:\n");
      printf("start indexes:");
      for (r = 0; r<GTH; r++)
      { printf(" %d",cf->start_quad_wgpu[r]);
      }
      printf("\n");
      i = 0;
      for (r = 0; r<GTH; r++)
      { printf("First weight %d mod %d:\n",r,GTH);
        while (cf->quad_s_4d_4w_wgpu[i].w >= 0)
        { printf("%3d %3d-%-3d %3d-%-3d\n", cf->quad_s_4d_4w_wgpu[i].s, 
                      cf->quad_s_4d_4w_wgpu[i].d, cf->quad_s_4d_4w_wgpu[i].d+3,
                      cf->quad_s_4d_4w_wgpu[i].w, cf->quad_s_4d_4w_wgpu[i].w+3);
          i += 1;
        }
        i += 1;
        printf("\n");
      }
      printf("\n");
    }

    if (cf->other_wgpu)
    { printf("other for wgpu:\n");
      printf("start indexes:");
      for (r = 0; r<GTH; r++)
      { printf(" %d",cf->start_other_wgpu[r]);
      }
      printf("\n");
      i = 0;
      for (r = 0; r<GTH; r++)
      { printf("Weight %d mod %d:\n",r,GTH);
        while (cf->other_wgpu[i].w >= 0)
        { printf("%3d %3d %3d\n", 
           cf->other_wgpu[i].s, cf->other_wgpu[i].d, cf->other_wgpu[i].w);
          i += 1;
        }
        i += 1;
        printf("\n");
      }
      printf("\n");
    }

    if (CONFIG_OCT_GPU_S_8D_8W_FW && cf->oct_s_8d_8w_dgpu)
    { printf("oct_s_8d_8w_dgpu:\n");
      printf("start indexes:");
      for (r = 0; r<NTH; r++)
      { printf(" %d",cf->start_oct_dgpu[r]);
      }
      printf("\n");
      i = 0;
      for (r = 0; r<NTH; r++)
      { printf("First destination unit %d mod %d:\n",r,NTH);
        while (cf->oct_s_8d_8w_dgpu[i].w >= 0)
        { printf("%3d %3d-%-3d %3d-%-3d\n", cf->oct_s_8d_8w_dgpu[i].s, 
                      cf->oct_s_8d_8w_dgpu[i].d, cf->oct_s_8d_8w_dgpu[i].d+7,
                      cf->oct_s_8d_8w_dgpu[i].w, cf->oct_s_8d_8w_dgpu[i].w+7);
          i += 1;
        }
        i += 1;
        printf("\n");
      }
      printf("\n");
    }

    if (CONFIG_QUAD_GPU_S_4D_4W_FW && cf->quad_s_4d_4w_dgpu)
    { printf("quad_s_4d_4w_dgpu:\n");
      printf("start indexes:");
      for (r = 0; r<NTH; r++)
      { printf(" %d",cf->start_quad_dgpu[r]);
      }
      printf("\n");
      i = 0;
      for (r = 0; r<NTH; r++)
      { printf("First destination unit %d mod %d:\n",r,NTH);
        while (cf->quad_s_4d_4w_dgpu[i].w >= 0)
        { printf("%3d %3d-%-3d %3d-%-3d\n", cf->quad_s_4d_4w_dgpu[i].s, 
                      cf->quad_s_4d_4w_dgpu[i].d, cf->quad_s_4d_4w_dgpu[i].d+3,
                      cf->quad_s_4d_4w_dgpu[i].w, cf->quad_s_4d_4w_dgpu[i].w+3);
          i += 1;
        }
        i += 1;
        printf("\n");
      }
      printf("\n");
    }

    if (cf->other_dgpu)
    { printf("other for dgpu:\n");
      printf("start indexes:");
      for (r = 0; r<NTH; r++)
      { printf(" %d",cf->start_other_dgpu[r]);
      }
      printf("\n");
      i = 0;
      for (r = 0; r<NTH; r++)
      { printf("Destination unit %d mod %d:\n",r,NTH);
        while (cf->other_dgpu[i].w >= 0)
        { printf("%3d %3d %3d\n", 
           cf->other_dgpu[i].s, cf->other_dgpu[i].d, cf->other_dgpu[i].w);
          i += 1;
        }
        i += 1;
        printf("\n");
      }
      printf("\n");
    }

    if (CONFIG_OCT_GPU_S_8D_8W_BW && cf->oct_s_8d_8w_sgpu)
    { printf("oct_s_8d_8w_sgpu:\n");
      printf("start indexes:");
      for (r = 0; r<NTH; r++)
      { printf(" %d",cf->start_oct_sgpu[r]);
      }
      printf("\n");
      i = 0;
      for (r = 0; r<NTH; r++)
      { printf("First destination unit %d mod %d:\n",r,NTH);
        while (cf->oct_s_8d_8w_sgpu[i].w >= 0)
        { printf("%3d %3d-%-3d %3d-%-3d\n", cf->oct_s_8d_8w_sgpu[i].s, 
                      cf->oct_s_8d_8w_sgpu[i].d, cf->oct_s_8d_8w_sgpu[i].d+7,
                      cf->oct_s_8d_8w_sgpu[i].w, cf->oct_s_8d_8w_sgpu[i].w+7);
          i += 1;
        }
        i += 1;
        printf("\n");
      }
      printf("\n");
    }

    if (CONFIG_QUAD_GPU_S_4D_4W_BW && cf->quad_s_4d_4w_sgpu)
    { printf("quad_s_4d_4w_sgpu:\n");
      printf("start indexes:");
      for (r = 0; r<NTH; r++)
      { printf(" %d",cf->start_quad_sgpu[r]);
      }
      printf("\n");
      i = 0;
      for (r = 0; r<NTH; r++)
      { printf("Source unit %d mod %d:\n",r,NTH);
        while (cf->quad_s_4d_4w_sgpu[i].w >= 0)
        { printf("%3d %3d-%-3d %3d-%-3d\n", cf->quad_s_4d_4w_sgpu[i].s, 
                      cf->quad_s_4d_4w_sgpu[i].d, cf->quad_s_4d_4w_sgpu[i].d+3,
                      cf->quad_s_4d_4w_sgpu[i].w, cf->quad_s_4d_4w_sgpu[i].w+3);
          i += 1;
        }
        i += 1;
        printf("\n");
      }
    }

    if (cf->other_sgpu)
    { printf("other for sgpu:\n");
      printf("start indexes:");
      for (r = 0; r<NTH; r++)
      { printf(" %d",cf->start_other_sgpu[r]);
      }
      printf("\n");
      i = 0;
      for (r = 0; r<NTH; r++)
      { printf("Source unit %d mod %d:\n",r,NTH);
        while (cf->other_sgpu[i].w >= 0)
        { printf("%3d %3d %3d\n", 
           cf->other_sgpu[i].s, cf->other_sgpu[i].d, cf->other_sgpu[i].w);
          i += 1;
        }
        i += 1;
        printf("\n");
      }
    }
  }
}


/* DISPLAY USAGE MESSAGE AND EXIT. */

static void usage(void)
{
  fprintf(stderr,
   "Usage: net-spec log-file N-inputs { N-hidden [ act-func ] } N-outputs\n");

  fprintf(stderr,
   "                / { group=prior [ config-spec | omit-spec ] }\n");

  fprintf(stderr,
   "   or: net-spec log-file [ \"sizes\" ] [ \"config\" ]  (display specifications)\n");

  fprintf(stderr,
   " config-spec: config:<file>{,<file>} omit-spec: omit:[-]<input>{,<input>}\n");

  fprintf(stderr,
   " act-fun: tanh softplus[0] identity  group: ti ih# bh# th# h#h# h#o io bo ah# ao\n");

  fprintf(stderr,
   " prior: [x]Width[:[Alpha-type][:[Alpha-unit][:[Alpha-weight]]]][!|!!]\n");

  exit(1);
}

