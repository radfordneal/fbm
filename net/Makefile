# MAKEFILE FOR NEURAL NET PROGRAMS. 

# Copyright (c) 1995-2021 by Radford M. Neal 
#
# Permission is granted for anyone to copy, use, modify, or distribute this
# program and accompanying programs and documents for any purpose, provided 
# this copyright notice is retained and prominently displayed, along with
# a note saying that the original programs are available from Radford Neal's
# web page, and note is made of any changes made to the programs.  The
# programs and documents are distributed without any warranty, express or
# implied.  As the programs were written for research purposes only, they have
# not been tested to the degree that would be advisable in any important
# application.  All use of these programs is entirely at the user's own risk.

include ../make.include


# Set mc_value, net_param, and net_value according to the definition of
# NET_PRECISION ('double' or 'float'), with 'double' the default.

NET_PRECISION ?= double
PREC_FLAGS = -Ddata_value=$(NET_PRECISION) \
             -Dmc_value=$(NET_PRECISION) \
             -Dnet_param=$(NET_PRECISION) \
             -Dnet_sigma=$(NET_PRECISION) \
             -Dnet_value=$(NET_PRECISION) \
             -DFP64=`echo $(NET_PRECISION) | sed -e s/double/1/ -e s/float/0/` \
             -DFP32=`echo $(NET_PRECISION) | sed -e s/float/1/ -e s/double/0/`


CFLAGS += $(PREC_FLAGS)
NVCC_OPTIONS += $(PREC_FLAGS)


programs:	net-spec net-gen net-rej net-eval net-display net-pred net-mc \
		net-gd net-plt net-tbl net-hist net-dvar \
		net-grad-test net-stepsizes net-genp net-approx net-his

# GPU versions of some programs.  Note that the GPU version of net-eval 
# is primarily for testing.  It is probably slower than the CPU version.

gpu:		net-mc-gpu net-gd-gpu net-grad-test-gpu

clean:
	rm -f *.o *.exe \
		  net-spec net-gen net-rej net-eval net-display net-pred net-mc\
	          net-gd net-plt net-tbl net-hist net-dvar \
	          net-grad-test net-stepsizes net-genp net-approx net-his \
	          net-mc-gpu net-gd-gpu net-grad-test-gpu


include ../util/util.make
include ../mc/mc.make


net-func.o:	net-func.c	misc.h log.h prior.h model.h net.h data.h \
				intrinsics-use.h sleef-use.h cuda-use.h
net-back.o:	net-back.c	misc.h log.h prior.h model.h net.h data.h \
				intrinsics-use.h sleef-use.h cuda-use.h
net-grad.o:	net-grad.c	misc.h log.h prior.h model.h net.h data.h \
				intrinsics-use.h cuda-use.h
net-util.o:	net-util.c	misc.h log.h prior.h model.h net.h data.h \
				cuda-use.h
net-config.o:	net-config.c	misc.h log.h prior.h model.h net.h data.h \
				cuda-use.h
net-setup.o:	net-setup.c	misc.h log.h prior.h model.h net.h data.h \
				cuda-use.h
net-prior.o:	net-prior.c	misc.h log.h prior.h model.h net.h data.h \
				rand.h cuda-use.h
net-model.o:	net-model.c	misc.h log.h prior.h model.h net.h data.h \
				rand.h cuda-use.h
net-data.o:	net-data.c	misc.h log.h prior.h model.h net.h data.h \
				net-data.h numin.h cuda-use.h
net-print.o:	net-print.c	misc.h log.h prior.h model.h net.h data.h \
				cuda-use.h
net-quantities.o: net-quantities.c misc.h log.h prior.h model.h net.h data.h \
				net-data.h quantities.h cuda-use.h


net-spec:	net-spec.o	net-util.o net-config.o prior.o \
				net-prior.o ars.o misc.o log.o rand.o
		$(CC) $(LFLAGS) net-spec.o net-util.o net-config.o prior.o \
		  net-prior.o ars.o misc.o log.o rand.o -lm -o net-spec

net-spec.o:	net-spec.c	misc.h prior.h model.h net.h log.h data.h

net-gen:	net-gen.o	net-util.o net-config.o prior.o \
				net-prior.o net-setup.o \
				misc.o log.o rand.o ars.o
		$(CC) $(LFLAGS) net-gen.o net-util.o net-config.o prior.o \
		  net-prior.o net-setup.o misc.o log.o rand.o ars.o \
		  -lm -o net-gen

net-gen.o:	net-gen.c	misc.h prior.h model.h net.h data.h log.h rand.h

net-rej:	net-rej.o	net-util.o net-config.o prior.o \
				net-prior.o net-setup.o \
				net-data.o model.o net-func.o net-model.o \
				misc.o log.o numin.o data-trans.o rand.o ars.o
		$(CC) $(LFLAGS) net-rej.o net-util.o net-config.o prior.o \
		  net-prior.o \
		  net-setup.o net-data.o model.o net-func.o net-model.o misc.o \
		  log.o numin.o data-trans.o rand.o ars.o -lm -o net-rej

net-rej.o:	net-rej.c	misc.h prior.h model.h net.h data.h log.h rand.h

net-eval:	net-eval.o	net-util.o net-config.o prior.o net-prior.o \
				net-setup.o \
				net-func.o net-model.o model.o misc.o \
				log.o rand.o ars.o
		$(CC) $(LFLAGS) net-eval.o net-util.o net-config.o prior.o \
		  net-prior.o \
		  net-setup.o net-func.o net-model.o model.o misc.o \
		  log.o rand.o ars.o -lm -o net-eval

net-eval.o:	net-eval.c	misc.h prior.h model.h net.h data.h log.h \
				cuda-use.h

net-values:	net-values.o	net-util.o net-config.o prior.o net-prior.o \
				net-setup.o \
				net-func.o net-model.o misc.o log.o rand.o ars.o
		$(CC) $(LFLAGS) net-values.o net-util.o net-config.o prior.o \
		  net-prior.o \
		  net-setup.o net-func.o net-model.o \
		  misc.o log.o rand.o ars.o -lm -o net-values

net-values.o:	net-values.c	misc.h prior.h model.h net.h data.h log.h 

net-display:	net-display.o	net-util.o net-config.o prior.o net-prior.o \
				net-setup.o \
				net-print.o misc.o log.o rand.o ars.o
		$(CC) $(LFLAGS) net-display.o net-util.o net-config.o prior.o \
		  net-prior.o \
		  net-setup.o net-print.o misc.o log.o rand.o ars.o \
		  -lm -o net-display

net-display.o:	net-display.c	misc.h prior.h model.h net.h data.h log.h

net-pred:	pred.o		net-pred.o net-setup.o net-func.o net-model.o \
				net-data.o model.o net-util.o net-config.o \
				misc.o log.o rand.o numin.o data-trans.o 
		$(CC) $(LFLAGS) pred.o net-pred.o net-setup.o net-func.o \
		  net-model.o net-data.o \
		  model.o net-util.o net-config.o misc.o log.o rand.o numin.o \
		  data-trans.o -lm -o net-pred

net-pred.o:	net-pred.c	misc.h prior.h model.h net.h net-data.h log.h \
				data.h numin.h rand.h mc.h pred.h

net-dvar:	net-dvar.o	net-util.o net-config.o net-setup.o net-func.o \
				net-model.o misc.o log.o rand.o 
		$(CC) $(LFLAGS) net-dvar.o net-setup.o net-func.o net-model.o \
		  net-util.o net-config.o misc.o log.o rand.o -lm -o net-dvar

net-dvar.o:	net-dvar.c	misc.h prior.h model.h net.h data.h log.h 

net-grad-test:	mc-grad-test.o	net-mc.o ars.o net-plt.o misc.o log.o rand.o \
				numin.o data-trans.o mc-iter.o mc-traj.o \
				mc-util.o mc-metropolis.o mc-hybrid.o \
				mc-slice.o net-setup.o prior.o net-prior.o \
				net-model.o net-func.o net-back.o net-grad.o \
				net-data.o model.o net-util.o net-config.o \
				net-quantities.o \
				mc-quantities.o quantities.o mc-heatbath.o 
		$(CC) $(LFLAGS) mc-grad-test.o net-mc.o ars.o misc.o log.o \
		  rand.o numin.o data-trans.o mc-iter.o mc-traj.o mc-util.o \
		  mc-metropolis.o mc-hybrid.o mc-slice.o mc-heatbath.o \
		  net-setup.o prior.o net-prior.o net-model.o net-func.o \
		  net-back.o net-grad.o net-data.o model.o net-util.o \
		  net-config.o \
		  net-plt.o net-quantities.o mc-quantities.o quantities.o \
		  -lm -o net-grad-test

net-stepsizes:	mc-stepsizes.o	net-mc.o ars.o net-plt.o misc.o log.o rand.o \
				numin.o data-trans.o mc-iter.o mc-traj.o \
				mc-util.o mc-heatbath.o \
				mc-metropolis.o mc-hybrid.o mc-slice.o \
				net-setup.o prior.o net-prior.o net-model.o \
				net-func.o net-back.o net-grad.o net-data.o \
				model.o net-util.o net-config.o \
				net-quantities.o mc-quantities.o quantities.o 
		$(CC) $(LFLAGS) mc-stepsizes.o net-mc.o ars.o misc.o log.o \
		  rand.o numin.o data-trans.o mc-iter.o mc-traj.o mc-util.o \
		  mc-metropolis.o mc-hybrid.o mc-slice.o mc-heatbath.o \
		  net-setup.o prior.o net-prior.o net-model.o net-func.o \
		  net-back.o net-grad.o net-data.o model.o net-util.o \
		  net-config.o \
		  net-plt.o net-quantities.o mc-quantities.o quantities.o \
		  -lm -o net-stepsizes

net-genp:	mc-genp.o	net-mc.o ars.o net-plt.o misc.o log.o rand.o \
				numin.o data-trans.o mc-iter.o mc-traj.o \
				mc-util.o mc-heatbath.o \
				mc-metropolis.o mc-hybrid.o mc-slice.o \
				net-setup.o prior.o net-prior.o net-model.o \
				net-func.o net-back.o net-grad.o net-data.o \
				model.o net-util.o net-config.o \
				net-quantities.o \
				mc-quantities.o quantities.o 
		$(CC) $(LFLAGS) mc-genp.o net-mc.o ars.o misc.o log.o \
		  rand.o numin.o data-trans.o mc-iter.o mc-traj.o mc-util.o \
		  mc-metropolis.o mc-hybrid.o mc-slice.o mc-heatbath.o \
		  net-setup.o prior.o net-prior.o net-model.o net-func.o \
		  net-back.o net-grad.o net-data.o model.o net-util.o \
		  net-config.o \
		  net-plt.o net-quantities.o mc-quantities.o quantities.o \
		  -lm -o net-genp

net-approx:	net-approx.o	net-util.o net-config.o net-setup.o \
				misc.o log.o rand.o
		$(CC) $(LFLAGS) net-approx.o net-util.o net-config.o \
		  net-setup.o misc.o log.o rand.o -lm -o net-approx

net-approx.o:	net-approx.c	misc.h prior.h model.h net.h log.h data.h

net-mc:		mc.o net-mc.o	ars.o misc.o log.o rand.o numin.o data-trans.o \
				net-plt.o mc-iter.o mc-traj.o mc-util.o \
				mc-metropolis.o mc-hybrid.o mc-slice.o \
				net-setup.o prior.o net-prior.o net-model.o \
				net-func.o net-back.o net-grad.o net-data.o \
				model.o net-util.o net-config.o \
				net-quantities.o \
				mc-quantities.o quantities.o mc-heatbath.o 
		$(CC) $(LFLAGS) mc.o net-mc.o ars.o misc.o log.o rand.o numin.o\
		  data-trans.o mc-iter.o mc-traj.o mc-util.o \
		  mc-metropolis.o mc-hybrid.o mc-slice.o mc-heatbath.o \
		  net-setup.o prior.o net-prior.o net-model.o net-func.o \
		  net-plt.o net-quantities.o mc-quantities.o quantities.o \
		  net-back.o net-grad.o net-data.o model.o net-util.o \
		  net-config.o -lm -o net-mc

net-his:	mc-his.o net-mc.o ars.o misc.o log.o rand.o data-trans.o \
				numin.o \
				net-plt.o mc-traj.o mc-util.o \
				net-setup.o prior.o net-prior.o net-model.o \
				net-func.o net-back.o net-grad.o net-data.o \
				model.o net-util.o net-config.o \
				net-quantities.o \
				mc-quantities.o quantities.o mc-heatbath.o 
		$(CC) $(LFLAGS) mc-his.o net-mc.o ars.o misc.o log.o rand.o \
		  data-trans.o mc-traj.o mc-util.o numin.o \
		  mc-heatbath.o \
		  net-setup.o prior.o net-prior.o net-model.o net-func.o \
		  net-plt.o net-quantities.o mc-quantities.o quantities.o \
		  net-back.o net-grad.o net-data.o model.o net-util.o \
		  net-config.o -lm -o net-his

net-mc.o:	net-mc.c	misc.h rand.h log.h mc.h data.h prior.h \
				model.h net.h net-data.h net-mc.h \
				cuda-use.h

net-gd:		net-gd.o	net-setup.o net-func.o net-model.o net-data.o \
				net-back.o net-grad.o net-prior.o net-util.o \
				net-config.o net-mc.o \
				model.o misc.o log.o rand.o \
				numin.o data-trans.o prior.o ars.o
		$(CC) $(LFLAGS) net-gd.o net-setup.o net-func.o net-model.o \
		  net-back.o net-grad.o net-data.o net-prior.o net-util.o \
		  net-config.o net-mc.o \
		  model.o misc.o log.o rand.o numin.o data-trans.o prior.o \
		  ars.o -lm -o net-gd

net-gd.o:	net-gd.c	misc.h rand.h log.h mc.h data.h prior.h \
				model.h	net.h net-data.h net-mc.h

net-plt:	net-plt.o	net-mc.o ars.o net-setup.o prior.o net-prior.o \
				net-func.o net-data.o model.o net-quantities.o \
				net-back.o net-grad.o net-util.o net-config.o \
				net-model.o \
				mc-quantities.o mc-util.o plt.o quantities.o \
				misc.o log.o rand.o numin.o data-trans.o 
		$(CC) $(LFLAGS) net-plt.o net-mc.o ars.o net-setup.o prior.o \
		  net-model.o net-func.o net-data.o model.o net-quantities.o \
		  net-back.o net-grad.o net-util.o net-config.o net-prior.o \
		  mc-quantities.o mc-util.o plt.o quantities.o \
		  misc.o log.o rand.o numin.o data-trans.o \
		  -lm -o net-plt

net-tbl:	net-plt.o	net-mc.o ars.o net-setup.o prior.o net-prior.o \
				net-func.o net-data.o model.o net-quantities.o \
				net-back.o net-grad.o net-util.o net-config.o \
				 net-model.o \
				mc-quantities.o mc-util.o tbl.o quantities.o \
				misc.o log.o rand.o numin.o data-trans.o 
		$(CC) $(LFLAGS) net-plt.o net-mc.o ars.o net-setup.o prior.o \
		  net-model.o net-func.o net-data.o model.o net-quantities.o \
		  net-back.o net-grad.o net-util.o net-config.o net-prior.o \
		  mc-quantities.o mc-util.o tbl.o quantities.o \
		  misc.o log.o rand.o numin.o data-trans.o \
		  -lm -o net-tbl

net-hist:	net-plt.o	net-mc.o ars.o net-setup.o prior.o net-prior.o \
				net-func.o net-data.o model.o net-quantities.o \
				net-back.o net-grad.o net-util.o net-config.o \
				net-model.o \
				mc-quantities.o mc-util.o hist.o quantities.o \
				misc.o log.o rand.o numin.o data-trans.o
		$(CC) $(LFLAGS) net-plt.o net-mc.o ars.o net-setup.o prior.o \
		  net-prior.o net-model.o net-func.o net-data.o model.o \
		  net-quantities.o net-back.o net-grad.o net-util.o \
		  net-config.o mc-quantities.o mc-util.o hist.o quantities.o \
		  misc.o log.o rand.o numin.o data-trans.o \
		  -lm -o net-hist

net-plt.o:	net-plt.c	misc.h log.h quantities.h mc.h \
				cuda-use.h


# GPU versions.

net-mc-gpu:	mc.o net-mc-gpu.o \
		  ars.o misc.o log.o rand.o numin.o prior.o \
		  data-trans.o mc-iter.o mc-traj.o mc-util.o \
		  mc-metropolis.o mc-hybrid.o mc-slice.o mc-heatbath.o \
		  net-setup-gpu.o net-prior-gpu.o net-model-gpu.o \
		  mc-quantities.o quantities.o model.o \
		  net-plt-gpu.o net-quantities-gpu.o \
                  net-data-gpu.o net-util-gpu.o net-config-gpu.o \
		  net-back-gpu.o net-grad-gpu.o net-func-gpu.o
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets \
		  mc.o net-mc-gpu.o \
		  ars.o misc.o log.o rand.o numin.o prior.o \
		  data-trans.o mc-iter.o mc-traj.o mc-util.o \
		  mc-metropolis.o mc-hybrid.o mc-slice.o mc-heatbath.o \
		  net-setup-gpu.o net-prior-gpu.o net-model-gpu.o \
		  mc-quantities.o quantities.o model.o \
		  net-plt-gpu.o net-quantities-gpu.o \
		  net-data-gpu.o net-util-gpu.o net-config-gpu.o \
		  net-back-gpu.o net-grad-gpu.o net-func-gpu.o \
		  -lm -o net-mc-gpu

net-mc-gpu.o:	net-mc.c misc.h rand.h log.h mc.h data.h prior.h \
		  model.h net.h net-data.h \
		  cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-mc.c -o net-mc-gpu.o

net-gd-gpu:	net-gd-gpu.o	net-setup-gpu.o net-func-gpu.o net-model-gpu.o \
				net-data-gpu.o net-back-gpu.o net-grad-gpu.o \
				net-prior-gpu.o net-util-gpu.o \
				net-config-gpu.o net-mc-gpu.o \
				model.o misc.o log.o rand.o \
				numin.o data-trans.o prior.o ars.o
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets \
		  net-gd-gpu.o net-setup-gpu.o net-func-gpu.o net-model-gpu.o \
		  net-back-gpu.o net-grad-gpu.o net-data-gpu.o net-prior-gpu.o \
		  net-util-gpu.o net-config-gpu.o net-mc-gpu.o \
		  model.o misc.o log.o rand.o numin.o data-trans.o prior.o \
		  ars.o -lm -o net-gd-gpu

net-gd-gpu.o:	net-gd.c	misc.h rand.h log.h mc.h data.h prior.h \
				model.h	net.h net-data.h net-mc.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-gd.c -o net-gd-gpu.o

net-grad-test-gpu:	mc-grad-test.o \
				net-mc-gpu.o ars.o misc.o log.o rand.o \
				numin.o data-trans.o mc-iter.o mc-traj.o \
				mc-util.o mc-metropolis.o mc-hybrid.o \
				mc-quantities.o quantities.o mc-heatbath.o \
				mc-slice.o prior.o model.o \
				net-model-gpu.o net-func-gpu.o net-back-gpu.o \
				net-grad-gpu.o net-data-gpu.o net-util-gpu.o \
				net-config-gpu.o net-quantities-gpu.o \
				net-plt-gpu.o net-setup-gpu.o net-prior-gpu.o
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets \
			mc-grad-test.o \
				net-mc-gpu.o ars.o misc.o log.o rand.o \
				numin.o data-trans.o mc-iter.o mc-traj.o \
				mc-util.o mc-metropolis.o mc-hybrid.o \
				mc-quantities.o quantities.o mc-heatbath.o \
				mc-slice.o prior.o model.o \
				net-model-gpu.o net-func-gpu.o net-back-gpu.o \
				net-grad-gpu.o net-data-gpu.o net-util-gpu.o \
				net-config-gpu.o net-quantities-gpu.o \
				net-plt-gpu.o net-setup-gpu.o net-prior-gpu.o \
		  -lm -o net-grad-test-gpu

net-func-gpu.o:	net-func.c misc.h prior.h model.h net.h data.h \
			   intrinsics-use.h sleef-use.h cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-func.c -o net-func-gpu.o

net-back-gpu.o:	net-back.c misc.h prior.h model.h net.h data.h \
			   intrinsics-use.h sleef-use.h cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-back.c -o net-back-gpu.o

net-grad-gpu.o:	net-grad.c misc.h prior.h model.h net.h data.h \
			   intrinsics-use.h cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-grad.c -o net-grad-gpu.o

net-util-gpu.o:	net-util.c misc.h prior.h model.h net.h data.h \
			   cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-util.c -o net-util-gpu.o

net-config-gpu.o:	net-config.c misc.h prior.h model.h net.h data.h \
			   cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-config.c -o net-config-gpu.o

net-setup-gpu.o:	net-setup.c misc.h prior.h model.h net.h data.h \
			   cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-setup.c -o net-setup-gpu.o

net-prior-gpu.o:	net-prior.c misc.h prior.h model.h net.h data.h rand.h \
			   cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-prior.c -o net-prior-gpu.o

net-model-gpu.o:	net-model.c misc.h prior.h model.h net.h data.h rand.h \
			   cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-model.c -o net-model-gpu.o

net-data-gpu.o:	net-data.c misc.h prior.h model.h net.h data.h net-data.h \
			   numin.h \
			   cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-data.c -o net-data-gpu.o

net-print-gpu.o:	net-print.c misc.h prior.h model.h net.h data.h \
			   cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-print.c -o net-print-gpu.o

net-quantities-gpu.o: net-quantities.c misc.h prior.h model.h net.h data.h \
			   net-data.h quantities.h log.h \
			   cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-quantities.c -o net-quantities-gpu.o

net-plt-gpu.o:	net-plt.c misc.h log.h quantities.h mc.h \
			   cuda-use.h
		nvcc $(NVCC_OPTIONS) -Wno-deprecated-gpu-targets -x cu -dc \
		  net-plt.c -o net-plt-gpu.o
