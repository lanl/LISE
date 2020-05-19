# Makefile tested on LASSEN.LLNL.GOV
#environment module load essl lapack
# this version requires the clang compitler
# module swap xl clang

COMP=clang 
FFTW_INCLUDE_OPTS=-I${ESSLHEADERDIR}
LAPACK_LIBS=-L${ESSLLIBDIR64} -Wl,-rpath,${ESSLLIBDIR64} -lessl -L${LAPACK_DIR}  -lscalapack -Wl,-rpath,${LAPACK_DIR} 

MPI_LIBS = -L${MPI_ROOT}/lib
MPI_INCLUDE = -I${MPI_ROOT}/include/

EXE = lise-static

OBJS = broyden_min.o print_wf.o deform.o \
	get-blcs-dscr.o create_destroy_groups.o system_energy.o external_pot.o \
	pots_io.o cnuclear-slda-solver_pn.o dens_io.o dens_start.o make_potentials.o \
	get-mem-req-blk-cyc.o ham_matrix.o operators.o constr_dens.o axial_symmetry.o \
	2dbc-slda-mpi-wr.o



all: $(EXE)

2dbc-slda-mpi-wr.o: 2dbc-slda-mpi-wr.c
	$(COMP)  -c ${MPI_INCLUDE} 2dbc-slda-mpi-wr.c
get-blcs-dscr.o: get-blcs-dscr.c
	$(COMP)  -c ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} get-blcs-dscr.c
get-mem-req-blk-cyc.o: get-mem-req-blk-cyc.c
	$(COMP)  -c ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} get-mem-req-blk-cyc.c
ham_matrix.o: ham_matrix.c vars_nuclear.h 
	$(COMP)  -c ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} ham_matrix.c
pots_io.o: pots_io.c vars_nuclear.h
	$(COMP)  -c ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} pots_io.c
constr_dens.o: constr_dens.c vars_nuclear.h 
	$(COMP)  -c $(OPT_OMP) ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} constr_dens.c 
operators.o: operators.c vars_nuclear.h
	$(COMP)  -c $(OPT_OMP) ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} operators.c
cnuclear-slda-solver_pn.o: cnuclear-slda-solver_pn.c vars_nuclear.h 
	$(COMP)  -c  ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} cnuclear-slda-solver_pn.c
external_pot.o: external_pot.c vars_nuclear.h 
	$(COMP)  -c ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} external_pot.c 
dens_start.o: dens_start.c vars_nuclear.h
	$(COMP)  -c ${FFTW_INCLUDE_OPTS} dens_start.c
make_potentials.o: make_potentials.c vars_nuclear.h 
	$(COMP)  -c ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} make_potentials.c 
dens_io.o: dens_io.c vars_nuclear.h
	$(COMP)  -c ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} dens_io.c 
system_energy.o: system_energy.c vars_nuclear.h
	$(COMP)  -c ${FFTW_INCLUDE_OPTS} ${MPI_INCLUDE} system_energy.c 
deform.o: deform.c vars_nuclear.h
	$(COMP)  -c ${FFTW_INCLUDE_OPTS} deform.c 
broyden_min.o: broyden_min.c
	$(COMP)  -c broyden_min.c 
print_wf.o: print_wf.c vars_nuclear.h
	$(COMP)  ${MPI_INCLUDE} -c print_wf.c
create_destroy_groups.o: create_destroy_groups.c
	$(COMP)  -c ${MPI_INCLUDE} create_destroy_groups.c 
axial_symmetry.o: axial_symmetry.c vars_nuclear.h
	$(COMP)  -c ${MPI_INCLUDE} axial_symmetry.c
clean:
	rm -f $(OBJS) $(EXE) $(COBSJ)

$(EXE): $(OBJS) 
	$(COMP) $(OPT_OMP) -o $(EXE) $(OBJS) ${FFTW_DIR} -lfftw3 -lfftw3_threads ${SCALAPACK_LIBS} ${LAPACK_LIBS} -llapack -lscalapack -lm ${MPI_LIBS} -lmpi_ibm
