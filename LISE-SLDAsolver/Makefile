#
# Parameters for compiling LISE solver:
#
# CC                    -the C compiler command
# LISE_LA_INC   	-the path to resolve header files for 'blas.h' (or equivalent)
# LISE_LA_LNK   	-the link line for ScaLAPACK and dependencies
# LISE_FFTW_INC         -the path to resolve header files for 'fftw.h' 
# LISE_FFTW_LNK         -the link line for FFTW 
# LISE_MPI_INC          -the path to resolve 'mpi.h' header file
# LISE_MPI_LNK          -the link line for MPI 
#
# summit.olcf.ornl.gov: IBM AC922 - IBM Power9 + NVIDIA VT100
#
# compiler: IBM XL (xl-16.1.1-5)
# operating system: linux-rhel7-ppc64le
# MPI: IBM Spectrum MPI (spectrum-mpi/10.3.1.2-20200121)
# ScaLAPACK: Netlib ScaLAPACK 2.0.2 
# BLAS: IBM ESSL (essl/6.1.0-2)
# FFTW: FFTW 3.3.8 (fftw/3.3.8) 
#
# NOTE: 
# 	It is recommended that Darshan, a default, automated I/O profiler, be excluded from building and running LISE codes:
# 	module unload darshan-runtime
#
# Modules:
# hsi/5.0.2.p5
# xalt/1.2.1
# lsf-tools/2.0
# DefApps
# essl/6.1.0-2
# xl/16.1.1-5
# spectrum-mpi/10.3.1.2-20200121
# fftw/3.3.8
# netlib-scalapack/2.0.2
#
# Environment variables:
#
# OLCF_ESSL_ROOT 		= /sw/summit/essl/6.1.0-2/essl/6.1
# OLCF_NETLIB_SCALAPACK_ROOT 	= /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/netlib-scalapack-2.0.2-xo4jhy4ydnx3qqmnegbzfrvz4am6q2v6
# OLCF_FFTW_ROOT 		= /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/fftw-3.3.8-azzdjlzx2j6dpqvzdir2nwvxypohyfq4
# MPI_ROOT 			= /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2
#
#

LINK= $(LISE_FFTW_LNK) $(LISE_LA_LNK) -lm $(LISE_MPI_LNK)

EXE = xlise-static

OBJS = broyden_min.o print_wf.o deform.o \
	get-blcs-dscr.o create_destroy_groups.o system_energy.o external_pot.o \
	pots_io.o cnuclear-slda-solver_pn.o dens_io.o dens_start.o make_potentials.o \
	get-mem-req-blk-cyc.o ham_matrix.o operators.o constr_dens.o axial_symmetry.o \
	2dbc-slda-mpi-wr.o

all: $(EXE)
$(EXE): $(OBJS) 
	$(CC) -o $(EXE) $(OBJS) $(LINK) 
	cp $(EXE) $(LISE_BIN_DIR)

2dbc-slda-mpi-wr.o: 2dbc-slda-mpi-wr.c
	$(CC) -c $(LISE_MPI_INC) 2dbc-slda-mpi-wr.c
get-blcs-dscr.o: get-blcs-dscr.c
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) get-blcs-dscr.c 
get-mem-req-blk-cyc.o: get-mem-req-blk-cyc.c
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) get-mem-req-blk-cyc.c
ham_matrix.o: ham_matrix.c vars_nuclear.h 
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) ham_matrix.c 
pots_io.o: pots_io.c vars_nuclear.h
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) pots_io.c 
constr_dens.o: constr_dens.c vars_nuclear.h 
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) constr_dens.c
operators.o: operators.c vars_nuclear.h
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) operators.c 
cnuclear-slda-solver_pn.o: cnuclear-slda-solver_pn.c vars_nuclear.h 
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) cnuclear-slda-solver_pn.c 
external_pot.o: external_pot.c vars_nuclear.h 
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) external_pot.c 
dens_start.o: dens_start.c vars_nuclear.h
	$(CC) -c $(LISE_LA_INC) dens_start.c 
make_potentials.o: make_potentials.c vars_nuclear.h 
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) make_potentials.c 
dens_io.o: dens_io.c vars_nuclear.h
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) dens_io.c 
system_energy.o: system_energy.c vars_nuclear.h
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) system_energy.c 
deform.o: deform.c vars_nuclear.h
	$(CC) -c $(LISE_LA_INC) deform.c 
broyden_min.o: broyden_min.c
	$(CC) -c $(LISE_LA_INC) -D$(LA) broyden_min.c 
print_wf.o: print_wf.c vars_nuclear.h
	$(CC) -c $(LISE_MPI_INC) print_wf.c 
create_destroy_groups.o: create_destroy_groups.c
	$(CC) -c $(LISE_MPI_INC) create_destroy_groups.c 
axial_symmetry.o: axial_symmetry.c vars_nuclear.h
	$(CC) -c $(LISE_LA_INC) $(LISE_MPI_INC) axial_symmetry.c 

.PHONY : clean
clean:
	rm -f $(OBJS) $(EXE) $(COBSJ)

