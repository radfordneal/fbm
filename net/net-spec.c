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

static int hconfig[Max_layers+1][Max_layers]; /* Indexes of cfg-h, or 0 if
                                                 not appeared yet */

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

  char ps[2000];
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

    nsqi = 0;
    for (l = 0; l<a->N_layers; l++) 
    { printf("  Hidden layer %d:  size %d",l,a->N_hidden[l]);
      if (a->layer_type[l]==Tanh_type)            printf("  tanh");
      else if (a->layer_type[l]==Softplus_type)   printf("  softplus");
      else if (a->layer_type[l]==Identity_type)   printf("  identity");
      else                                        printf("  UNKNOWN TYPE!");
      if (flgs && list_flags (flgs->omit, a->N_inputs, 1<<(l+1), ps) > 0)
      { printf("  omit%s",ps);
      }
      if (flgs && flgs->input_config[l])
      { printf("  cfg-i:%s",flgs->config_files+flgs->input_config[l]);
      }
      if (flgs)
      { unsigned b;
        int ls;
        for (ls = 0, b = a->has_nsq[l]; b!=0; ls++, b>>=1)
        { if (b&1)
          { if (ls>=l-1) abort();
            int c = flgs->nonseq_config[nsqi++];
            if (c)
            { printf("  cfg-h%d:%s", ls, flgs->config_files+c);
            }
          }
        }
      }
      if (flgs && flgs->hidden_config[l])
      { printf("  cfg-h:%s",flgs->config_files+flgs->hidden_config[l]);
      }
      if (flgs && flgs->bias_config[l])
      { printf("  cfg-b:%s",flgs->config_files+flgs->bias_config[l]);
      }
      printf("\n");
    }

    printf ("  Output layer:    size %d", a->N_outputs);
    if (flgs && list_flags (flgs->omit, a->N_inputs, 1, ps) > 0)
    { printf("  omit%s",ps);
    }
    if (flgs && flgs->input_config[a->N_layers])
    { printf("  cfg-i:%s",
             flgs->config_files+flgs->input_config[a->N_layers]);
    }
    for (l = a->N_layers-1; l>=0; l--)
    { int k = 2*a->N_layers-1-l;
      if (flgs && flgs->hidden_config[k])
      { if (a->N_layers==1)
        { printf("  cfg-h:%s",
                 flgs->config_files+flgs->hidden_config[k]);
        }
        else
        { printf("  cfg-h%d:%s", l,
                 flgs->config_files+flgs->hidden_config[k]);
        }
      }
    }
    if (flgs && flgs->bias_config[a->N_layers])
    { printf("  cfg-b:%s",
             flgs->config_files+flgs->bias_config[a->N_layers]);
    }
    printf("\n");
  
    printf("\n");
    
    /* Display priors record. */
  
    printf("\n");
  
    if ((p = logg.data['P'])==0)
    { printf("No prior specifications found\n\n");
      exit(0);
    }
  
    printf("Prior Specifications:\n");
  
    if (a->has_ti) 
    { printf("\n  Input Offsets:          %s\n",prior_show(ps,p->ti));
    }
  
    nsqi = 0;
    for (l = 0; l<a->N_layers; l++)
    { printf("\n         Hidden Layer %d\n\n",l);
      int lp;
      for (lp = 0; lp<l; lp++)
      { if ((a->has_nsq[l]>>lp) & 1)
        { printf("  Hidden%d-Hidden Weights: %s\n", 
                  lp, prior_show(ps,p->nsq[nsqi++]));
        }
      }
      if (l>0 && a->has_hh[l-1])
      { printf("  Hidden-Hidden Weights:  %s\n", prior_show(ps,p->hh[l-1]));
      }
      if (a->has_ih[l]) 
      { printf("  Input-Hidden Weights:   %s\n", prior_show(ps,p->ih[l]));
      }
      if (a->has_bh[l]) 
      { printf("  Hidden Biases:          %s\n", prior_show(ps,p->bh[l]));
      }
      if (a->has_th[l]) 
      { printf("  Hidden Offsets:         %s\n", prior_show(ps,p->th[l]));
      }
    }

    printf("\n         Output Layer\n\n");

    for (l = a->N_layers-1; l>=0; l--)
    { if (a->has_ho[l]) 
      { printf("  Hidden%d-Output Weights: %s\n",l,prior_show(ps,p->ho[l]));
      }
    }

    if (a->has_io) 
    { printf("  Input-Output Weights:   %s\n",prior_show(ps,p->io));
    }

    if (a->has_bo) 
    { printf("  Output Biases:          %s\n",prior_show(ps,p->bo));
    }

    for (l = 0; l<a->N_layers && !a->has_ah[l]; l++) ;

    if (l<a->N_layers || a->has_ao)
    {
      if (a->N_layers>0 && l<a->N_layers)
      { printf("\n  Hidden adjustments: ");
        for (l = 0; l<a->N_layers; l++)
        { if (p->ah[l]==0) printf(" -");
          else             printf(" %.2f",p->ah[l]);
        }
      }

      if (a->has_ao)
      { printf("\n  Output adjustments: ");
        if (p->ao==0) printf(" -");
        else          printf(" %.2f",p->ao);
      }
      printf("\n");
    }
  
    printf("\n");

    net_precomputed pre;
    unsigned total;

    if (show_sizes || show_configs)
    { total = net_setup_param_count (a, flgs, &pre);
    }

    if (show_sizes)
    { 
      printf("\nNumbers of parameters in groups (total %u):\n\n",total);
  
      char t[100];
        
      if (a->has_ti) 
      { printf("  %-27s%6d\n","Input Offsets:", a->N_inputs);
      }
    
      nsqi = 0;
      for (l = 0; l<a->N_layers; l++)
      { int lp;
        for (lp = 0; lp<l; lp++)
        { if ((a->has_nsq[l]>>lp) & 1)
          { sprintf(t, "Hidden%d-Hidden%d Weights:", lp, l);
            printf("  %-27s%6d\n", t, a->nonseq_config[nsqi] ?
              a->nonseq_config[nsqi]->N_wts : a->N_hidden[lp]*a->N_hidden[l]);
            nsqi += 1;
          }
        }
        if (l>0 && a->has_hh[l-1])
        { sprintf(t, "Hidden%d-Hidden%d Weights:", l-1, l);
          printf("  %-27s%6d\n", t, a->hidden_config[l] ?
             a->hidden_config[l]->N_wts : a->N_hidden[l-1]*a->N_hidden[l]);
        }
        if (a->has_ih[l]) 
        { sprintf(t, "Input-Hidden%d Weights:", l);
          printf("  %-27s%6d\n", t, a->input_config[l] ?
             a->input_config[l]->N_wts : a->N_inputs*a->N_hidden[l]);
        }
        if (a->has_bh[l]) 
        { sprintf(t, "Hidden%d Biases:", l);
          printf("  %-27s%6d\n", t, a->bias_config[l] ?
             a->bias_config[l]->N_wts : a->N_hidden[l]);
        }
        if (a->has_th[l]) 
        { sprintf(t, "Hidden%d Offsets:", l);
          printf("  %-27s%6d\n", t, a->N_hidden[l]);
        }
      }

      for (l = a->N_layers-1; l>=0; l--)
      { if (a->has_ho[l]) 
        { sprintf(t, "Hidden%d-Output Weights:", l);
          printf("  %-27s%6d\n", t, a->hidden_config[2*a->N_layers-l-1] ?
              a->hidden_config[2*a->N_layers-l-1]->N_wts : 
              a->N_hidden[l]*a->N_outputs);
        }
      }

      if (a->has_io) 
      { printf("  %-27s%6d\n", "Input-Output Weights:", 
                a->N_inputs*a->N_outputs);
      }

      if (a->has_bo) 
      { printf("  %-27s%6d\n", "Output Biases:", a->N_outputs);
      }
    
      printf("\n");
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

  int fileix = 1;  /* don't start at 0 since 0 is used for "none" */

  while (*ap!=0 && strcmp(*ap,"/")!=0)
  { 
    double size;
    int omit, iconfig, bconfig, type;
    int i;

    if ((size = atoi(*ap++))<=0) usage();

    if (*ap==0) usage();

    omit = 0;
    iconfig = 0;
    bconfig = 0;
    type = -1;

    while (*ap!=0 && (*ap)[0]>='a' && (*ap)[0]<='z')
    { if (strncmp(*ap,"omit:",5)==0)
      { if (omit) usage();
        omit = 1;
        parse_flags (*ap+4, flgs->omit, a->N_inputs, 1);
        any_flags = 1;
      }
      else if (strncmp(*ap,"cfg-i:",6)==0)
      { if (iconfig) usage();
        iconfig = 1;
        strcpy(flgs->config_files+fileix,*ap+6);
        if (a->N_layers<=Max_layers) flgs->input_config[a->N_layers] = fileix;
        fileix += strlen(flgs->config_files+fileix) + 1;
        any_flags = 1;
      }
      else if (strncmp(*ap,"cfg-h",5)==0)
      { char *q;
        int k;
        if (ap[0][5]==':')
        { q = *ap+6;
          k = a->N_layers-1;
        }
        else
        { q = strchr(*ap,':');
          if (q==0) usage();
          q += 1;
          k = atoi(*ap+5);
          if (k<0) usage();
        }
        if (a->N_layers<1)
        { fprintf(stderr,
                 "Can't have cfg-h flag when no previous hidden layer\n");
          exit(2);
        }
        if (hconfig[a->N_layers][k])
        { fprintf(stderr,"Duplicate cfg-h%d flag\n",k);
          exit(2);
        }
        if (k>a->N_layers-1)
        { fprintf(stderr,"Connections must be from earlier hidden layer\n");
          exit(2);
        }
        strcpy(flgs->config_files+fileix,q);
        hconfig[a->N_layers][k] = fileix;
        fileix += strlen(flgs->config_files+fileix) + 1;
        any_flags = 1;
      }
      else if (strncmp(*ap,"cfg-b:",6)==0)
      { if (bconfig) usage();
        bconfig = 1;
        strcpy(flgs->config_files+fileix,*ap+6);
        if (a->N_layers<=Max_layers) flgs->bias_config[a->N_layers] = fileix;
        fileix += strlen(flgs->config_files+fileix) + 1;
        any_flags = 1;
      }
      else if (strcmp(*ap,"tanh")==0)
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
      else
      { usage();
      }
      ap += 1;
    }

    if (omit && iconfig)
    { fprintf(stderr, "omit flag may not be combined with cfg-i\n");
      exit(2);
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
      for (i = 0; i<a->N_inputs; i++) 
      { flgs->omit[i] = 
          (flgs->omit[i] | ((flgs->omit[i]&1)<<(a->N_layers+1))) & ~1;
      }
      a->any_omitted[a->N_layers] = omit;

      if (a->N_layers>0 && hconfig[a->N_layers][a->N_layers-1])
      { flgs->hidden_config[a->N_layers] = hconfig[a->N_layers][a->N_layers-1];
      }

      a->N_layers += 1;
    }

    else  /* last layer size, so this is the output layer */
    { 
      a->N_outputs = size;
      if (type!=-1) usage();
      a->any_omitted[a->N_layers] = omit;

      for (l = 0; l<a->N_layers; l++)
      { if (hconfig[a->N_layers][l])
        { flgs->hidden_config[2*a->N_layers-1-l] = hconfig[a->N_layers][l];
        }
      }
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
        }
        else
        { a->has_nsq[l] |= 1<<ls;
          if (!prior_parse(&p->nsq[nsqi++],pr)) usage();
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
      }
    }
    else if (sscanf(*ap,"io%c",&eq)==1 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { a->has_io = 1;
        if (!prior_parse(&p->io,pr)) usage();
      }
    }
    else if (sscanf(*ap,"bo%c",&eq)==1 && eq=='=')
    { if (strcmp(pr,"-")!=0)
      { a->has_bo = 1;
        if (!prior_parse(&p->bo,pr)) usage();
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
    { fprintf(stderr,"Illegal prior for hidden biases or offsets\n");
      exit(1); 
    }
  }

  if (p->bo.scale || p->bo.alpha[2]!=0)
  { fprintf(stderr,"Illegal prior for output biases\n");
    exit(1); 
  }

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

  /* Set up any configurations for non-sequential connections, now that
     both configs and priors are available. */

  nsqi = 0;
  for (l = 0; l<a->N_layers; l++)
  { unsigned b;
    int ls;
    for (ls = 0, b = a->has_nsq[l]; b!=0; ls++, b>>=1)
    { if (b&1)
      { if (ls>=l-1) abort();
        flgs->nonseq_config[nsqi++] = hconfig[l][ls];
      }
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

    if (CONFIG_OCT_GPU_S_8D_8W_GRAD)
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

    if (CONFIG_QUAD_GPU_S_4D_4W_GRAD)
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
      printf("quad_s_4d_4w_2_wgpu:\n");
      printf("start indexes:");
      for (r = 0; r<GTH; r++)
      { printf(" %d",cf->start_quad_2_wgpu[r]);
      }
      printf("\n");
      i = 0;
      for (r = 0; r<GTH; r++)
      { printf("First weight %d mod %d:\n",r,GTH);
        while (cf->quad_s_4d_4w_2_wgpu[i].w >= 0)
        { printf("%3d %3d-%-3d %3d-%-3d\n", cf->quad_s_4d_4w_2_wgpu[i].s, 
                  cf->quad_s_4d_4w_2_wgpu[i].d, cf->quad_s_4d_4w_2_wgpu[i].d+3,
                  cf->quad_s_4d_4w_2_wgpu[i].w, cf->quad_s_4d_4w_2_wgpu[i].w+3);
          i += 1;
        }
        i += 1;
        printf("\n");
      }
      printf("\n");
    }

    printf("other for wgpu:\n");
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
    printf("other_2 for wgpu (pairs with same w):\n");
    printf("start indexes:");
    for (r = 0; r<GTH; r++)
    { printf(" %d",cf->start_other_2_wgpu[r]);
    }
    printf("\n");
    i = 0;
    for (r = 0; r<GTH; r++)
    { printf("Weight %d mod %d:\n",r,GTH);
      while (cf->other_2_wgpu[i].w >= 0)
      { printf("%3d %3d %3d\n", 
           cf->other_2_wgpu[i].s, cf->other_2_wgpu[i].d, cf->other_2_wgpu[i].w);
        i += 1;
      }
      i += 1;
      printf("\n");
    }
    printf("\n");

    if (CONFIG_OCT_GPU_S_8D_8W_FW)
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

    if (CONFIG_QUAD_GPU_S_4D_4W_FW)
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

    printf("other for dgpu:\n");
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

    if (CONFIG_OCT_GPU_S_8D_8W_BW)
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

    if (CONFIG_QUAD_GPU_S_4D_4W_BW)
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

    printf("other for sgpu:\n");
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


/* DISPLAY USAGE MESSAGE AND EXIT. */

static void usage(void)
{
  fprintf(stderr,
   "Usage: net-spec log-file N-inputs { N-hidden { flag } } N-outputs { flag }\n");

  fprintf(stderr,
   "                / { group=prior }\n");

  fprintf(stderr,
   "   or: net-spec log-file [ \"sizes\" ] [ \"config\" ]  (display specifications)\n");

  fprintf(stderr,
   "Group: ti ih# bh# th# h#h# h#o io bo ah# ao\n");

  fprintf(stderr,
   "Prior: [x]Width[:[Alpha-type][:[Alpha-unit][:[Alpha-weight]]]]\n");

  fprintf(stderr,
   "Flags: cfg-i:<file> cfg-h:<file> cfg-b:<file>\n");
  fprintf(stderr,
   "       omit:[-]<input>{,<input>} tanh softplus identity\n");

  exit(1);
}

