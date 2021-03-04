#!/bin/bash
#COBALT -t 59 
#COBALT -n 128
#COBALT -q default 
#COBALT --attrs mcdram=cache:numa=quad 
#COBALT -A CSC249ADCD502

module unload darshan
module unload cray-libsci
module load cray-fftw
export LD_LIBRARY_PATH=${MKLROOT}/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2020.0.166/linux/compiler/lib/intel64_lin/:$LD_LIBRARY_PATH

time aprun -n 42 -N1 ./xlise-static input_file.txt >& O20.out

