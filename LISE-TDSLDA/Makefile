#
# Parameters for compiling LISE time-dependent:
# CC                    -the C compiler command
# LISE_FFTW_INC         -the path to resolve header file for 'fftw.h'
# LISE_FFTW_LNK         -the link line for FFTW
# LISE_MPI_INC          -the path to resolve 'mpi.h' header file
# LISE_MPI_LNK          -the link line for MPI
# NCC                   -the NVIDIA CUDA compiler driver, 'nvcc'
# LISE_CUFFT_INC        -the path to resolve CUDA FFT header file 'cufft.h'
# LISE_CUFFT_LIB        -the link line for CUDA FFT
# LISE_CUDA_INC         -the path to the CUDA Toolkit header files
# LISE_CUDA_LIB         -the line for the CUDA runtime library
#
# summit.olcf.ornl.gov: IBM AC922 - IBM Power9 + NVIDIA VT100
#
# compiler: IBM XL (xl-16.1.1-5)
# cuda compiler driver: NVIDIA CUDA (cuda/10.1.243)
# operating system: linux-rhel7-ppc64le
# MPI: IBM Spectrum MPI (spectrum-mpi/10.3.1.2-20200121)
# BLAS: IBM ESSL (essl/6.1.0-2)
# FFTW: FFTW 3.3.8 (fftw/3.3.8) 
#
# NOTE: 
#       It is recommended that Darshan, a default, automated I/O profiler, be excluded from building and running LISE codes:
#       module unload darshan-runtime
#
# Modules:
# xl/16.1.1-5
# spectrum-mpi/10.3.1.2-20200121
# hsi/5.0.2.p5
# xalt/1.2.1
# lsf-tools/2.0
# DefApps
# fftw/3.3.8
# cuda/10.1.243
# essl/6.1.0-2
#
# Environment variables:
#
# OLCF_ESSL_ROOT
# CUDA_TOOLKIT_ROOT_DIR
# OLCF_FFTW_ROOT
# MPI_ROOT
#
# NB:
# 	CUDA_TOOLKIT_ROOT_DIR=/sw/summit/cuda/10.1.243
# 	CUDA_DIR=/sw/summit/cuda/10.1.243
# 	OLCF_CUDA_ROOT=/sw/summit/cuda/10.1.243
# 	CUDAPATH=/sw/summit/cuda/10.1.243
#
# compile commands:
# cp nuclear-gpu.c nuclear-gpu.cu;
# nvcc -c --gpu-architecture=sm_70 nuclear-gpu.cu 
# xlc -c ctdslda.c 
# xlc -c -DEXTBOOST -DCP ctdslda.c 
# xlc -c create_destroy_groups.c
# xlc -c densities.c
# xlc -c deform.c 
# xlc -c external_pot.c
# xlc -c fposio-wf.c
# xlc -c getnwfip.c
# xlc -c operators.c
# xlc -c potentials.c
# xlc -c system_energy.c
# xlc -c wf.c
# xlc -c rotation.c
# xlc -c postprocess.c
# xlc -o xlise-tdslda-gpu deform.o densities.o external_pot.o operators.o create_destroy_groups.o fposio-wf.o ctdslda.o nuclear-gpu.o getnwfip.o potentials.o system_energy.o wf.o rotation.o -L${OLCF_FFTW_ROOT}/lib -lfftw3 -L${OLCF_ESSL_ROOT}/lib64 -lessl -L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -lcufft -lcudart -L${MPI_ROOT}/lib -lmpi_ibm 
#
# environment varialbles:
# OLCF_ESSL_ROOT=/sw/summit/essl/6.1.0-2/essl/6.1
#
# modules:
# xl/16.1.1-5
# spectrum-mpi/10.3.1.2-20200121
# hsi/5.0.2.p5
# xalt/1.2.1
# lsf-tools/2.0
# DefApps
# fftw/3.3.8
# cuda/10.1.243
# essl/6.1.0-2
#
# xlc -o xlise-tdslda-gpu deform.o densities.o external_pot.o operators.o create_destroy_groups.o fposio-wf.o ctdslda.o nuclear-gpu.o getnwfip.o potentials.o system_energy.o wf.o rotation.o -L${OLCF_FFTW_ROOT}/lib -lfftw3 -L${OLCF_ESSL_ROOT}/lib64 -lessl -L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -lcufft -lcudart -L${MPI_ROOT}/lib -lmpi_ibm 
#
#The following parameters are inherited recursively from the top-level Makefile definitions.
#reference: 'make -e' 
#
#CC = xlc
#NCC= ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc
#NCC_SM=--gpu-architecture=sm_70
#LISE_CUDA_INC=-I${CUDA_TOOLKIT_ROOT_DIR}/include
#LISE_CUDA_LNK=-L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -lcufft -lcudart
#LISE_LAS_INC=-I${OLCF_ESSL_ROOT}/include
#LISE_LAS_LNK=-L${OLCF_ESSL_ROOT}/lib64 -lessl
#LISE_FFTW_INC=-I${OLCF_FFTW_ROOT}/include
#LISE_FFTW_LNK=-L${OLCF_FFTW_ROOT}/lib -lfftw3
#LISE_MPI_INC=-I${MPI_ROOT}/include
#LISE_MPI_LNK=-L${MPI_ROOT}/lib -lmpi_ibm
#
##LISE_LA_ESSL, targets IBM ESSL LAPACK and BLAS
##LISE_LA_MKL, targets Intel MKL LAPACK and BLAS
##LISE_LA_REF, targets Netlib reference LAPACK and BLAS:  LAPACKE, CBLAS
#LA=LISE_LA_ESSL

LINK= $(LISE_CUDA_LNK) $(LISE_FFTW_LNK) $(LISE_LAS_LNK) $(LISE_MPI_LNK)

OBJS1 = deform.o densities.o external_pot.o operators.o create_destroy_groups.o fposio-wf.o ctdslda.o nuclear-gpu.o getnwfip.o potentials.o system_energy.o wf.o rotation.o  

OBJS2 = deform.o densities.o external_pot.o operators.o create_destroy_groups.o fposio-wf.o ctdslda_rst.o nuclear-gpu.o getnwfip.o potentials.o system_energy.o wf.o rotation.o  

OBJS3 = postprocess.o deform.o operators.o potentials.o nuclear-gpu.o

EXE1 = xlise-tdslda-gpu
EXE2 = xlise-tdslda-gpu-rst
EXE3 = xlise-tdslda-postprocess

all: $(EXE1) $(EXE2) $(EXE3)
	cp $(EXE1) $(EXE2) $(EXE3) $(LISE_BIN_DIR)

$(EXE1): $(OBJS1) 
	$(CC) -o $(EXE1) $(OBJS1) $(LINK)

$(EXE2): $(OBJS2) 
	$(CC) -o $(EXE2) $(OBJS2) $(LINK)

$(EXE3): $(OBJS3)
	$(CC) -o $(EXE3) $(OBJS3) $(LINK)


nuclear-gpu.o: nuclear-gpu.c vars.h nuclear-gpu.h
	cp nuclear-gpu.c nuclear-gpu.cu; 
	$(NCC) $(NCC_SM) -c -O2 $(LISE_FFTW_INC) $(LISE_CUDA_INC) $(LISE_MPI_INC) nuclear-gpu.cu

ctdslda.o: ctdslda.c vars.h tdslda_func.h nuclear-gpu.h
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_CUDA_INC) $(LISE_MPI_INC) -DEXTBOOST -DCP ctdslda.c

ctdslda_rst.o: ctdslda.c vars.h tdslda_func.h nuclear-gpu.h
	$(CC) -o ctdslda_rst.o -c -O2 $(LISE_FFTW_INC) $(LISE_CUDA_INC) $(LISE_MPI_INC) -DCP -DRSTRT ctdslda.c

create_destroy_groups.o: create_destroy_groups.c vars.h
	$(CC) -c -O2 $(LISE_MPI_INC) create_destroy_groups.c

densities.o: densities.c vars.h tdslda_func.h
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_CUDA_INC) $(LISE_MPI_INC) densities.c

deform.o: deform.c vars.h tdslda_func.h
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_MPI_INC) deform.c

external_pot.o: external_pot.c vars.h tdslda_func.h
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_MPI_INC) external_pot.c

fposio-wf.o: fposio-wf.c
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_CUDA_INC) $(LISE_MPI_INC) fposio-wf.c

getnwfip.o: getnwfip.c
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_CUDA_INC) $(LISE_MPI_INC) getnwfip.c

operators.o: operators.c vars.h
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_CUDA_INC) $(LISE_MPI_INC) operators.c 

potentials.o: potentials.c
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_LAS_INC) $(LISE_MPI_INC) potentials.c

system_energy.o: system_energy.c vars.h tdslda_func.h
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_MPI_INC) system_energy.c

wf.o: wf.c 
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_MPI_INC) wf.c

rotation.o: rotation.c vars.h tdslda_func.h 
	$(CC) -c -O2 $(LISE_FFTW_INC) $(LISE_LAS_INC) $(LISE_MPI_INC) -D$(LA) rotation.c

postprocess.o: postprocess.c vars.h
	$(CC) -c -O2 $(LISE_FFTW_INC) postprocess.c


.PHONY : clean
clean:
	rm -f $(OBJS1) $(OBJS2) $(OBJS3) $(EXE1) $(EXE2) $(EXE3)

