#Makefile for LASSEN.LLNL.GOV

COMP = xlc

CUDA_INCLUDES = -I${CUDA_HOME}/include

CUDA_LIBS = -L${CUDA_HOME}/lib64

MPI_LIBS = -L${MPI_ROOT}/lib

MPI_INCLUDES = -I${MPI_ROOT}/include

FFTW_INCLUDES = -I${FFTW_ROOT}/include

FFTW_LIBS = -L${FFTW_ROOT}/lib

LAPACK_LIBS=-L${LAPACK_DIR}  -llapack -Wl,-rpath,${LAPACK_DIR}

OBJS1 = deform.o densities.o external_pot.o operators.o create_destroy_groups.o fposio-wf.o ctdslda.o nuclear-gpu.o getnwfip.o potentials.o system_energy.o wf.o rotation.o  

OBJS2 = deform.o densities.o external_pot.o operators.o create_destroy_groups.o fposio-wf.o ctdslda_rst.o nuclear-gpu.o getnwfip.o potentials.o system_energy.o wf.o rotation.o  

OBJS3 = postprocess.o deform.o operators.o potentials.o

LINKOPT = ${CUDA_LIBS} -lcudart -lcufft  ${FFTW_LIBS}  -lfftw3 ${LAPACK_LIBS} -llapack ${MPI_LIBS} -lmpi_ibm -lm

EXE1 = lise-tdslda-gpu

EXE2 = lise-tdslda-rst-gpu

EXE3 = lise-tdslda-postprocess

all: $(EXE1) $(EXE2) $(EXE3)

nuclear-gpu.o: nuclear-gpu.c vars.h nuclear-gpu.h
	cp nuclear-gpu.c nuclear-gpu.cu; nvcc --gpu-architecture=sm_60 -c -O3 ${FFTW_INCLUDES} ${MPI_INCLUDES} -D__INTEL_COMPILER nuclear-gpu.cu;
ctdslda.o: ctdslda.c vars.h
	$(COMP) -c -O3 -DEXTBOOST ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} -DCP ctdslda.c;
ctdslda_rst.o: ctdslda.c vars.h
	$(COMP) -o ctdslda_rst.o -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} -DCP -DRSTRT ctdslda.c;
create_destroy_groups.o: create_destroy_groups.c vars.h
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} create_destroy_groups.c;
densities.o: densities.c vars.h
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} densities.c;
deform.o: deform.c vars.h
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} deform.c;
external_pot.o: external_pot.c
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} external_pot.c;
fposio-wf.o: fposio-wf.c
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} fposio-wf.c;
getnwfip.o: getnwfip.c
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} getnwfip.c;
operators.o: operators.c
	$(COMP) -c -O3 ${FFTW_INCLUDES} ${CUDA_INCLUDES} operators.c;
potentials.o: potentials.c
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} potentials.c;
system_energy.o: system_energy.c vars.h
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} system_energy.c;
wf.o: wf.c 
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} wf.c;
rotation.o: rotation.c 
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} rotation.c;
fluctuations.o: fluctuations.c vars.h
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} fluctuations.c;
postprocess.o: postprocess.c vars.h
	$(COMP) -c -O3 ${MPI_INCLUDES} ${FFTW_INCLUDES} ${CUDA_INCLUDES} postprocess.c;

clean:
	rm -f $(OBJS1) $(OBJS2) $(OBJS3) $(EXE1) $(EXE2) $(EXE3)

$(EXE1): $(OBJS1) 
	$(COMP) -o $(EXE1) $(OBJS1) $(LINKOPT) 

$(EXE2): $(OBJS2) 
	$(COMP) -o $(EXE2) $(OBJS2) $(LINKOPT) 

$(EXE3): $(OBJS3)
	$(COMP) -o $(EXE3) $(OBJS3) $(LINKOPT)



