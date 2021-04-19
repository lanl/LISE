# SLDA-Solver & TDSLDA
Contains static (LANL C19040) and time-dependent (LANL C19048) codes 

******************************************************************

README file for building and executing the LISE software:

0. Obtaining the codes
1. LISE directory structure
2. Software dependencies 
3. Target computer architectures 
4. Compiling and linking the codes 
5. Running the programs
 
******************************************************************

## 0. Obtaining the codes

git clone https://github.com/lanl/LISE

******************************************************************

## 1. LISE directory structure

The LISE directory structure will be referenced in these instructions.

```
LISE
|-- LICENSE
|-- LISE-SLDAsolver
|-- LISE-TDSLDA
|-- LISE-TESTS
|-- LISE.defs
|-- Makefile
|-- README.md
```

'./LISE-SLDAsolver': directory that contains the source codes and a default makefile for the LISE solver program; './LISE-SLDAsolver/builds' contains successful 'LISE.defs' and makefiles for a couple of target platforms

'./LISE-TDSLDA': directory that contains the source codes and a default makefile for the LISE time-dependent programs; './LISE-TDSLDA/builds' contains successful 'LISE.defs' and makefiles for a couple of target platforms

'./LISE-TESTS': directory that contains the LISE build acceptance tests ('./LISE-TESTS/O20'), prototype problem examples ('./LISE-TESTS/Collisions_238U', './LISE-TESTS/Fission_240Pu'), and 'README.txt' for clarifications

'./LISE.defs': users should only need to edit this file to set the variables and software paths for including and linking the LISE software dependencies 

'./Makefile': generic makefile structure that should not need to be edited; default case imports 'LISE.defs' specified for 'summit.olcf.ornl.gov' target

'./README.md': this file

******************************************************************

## 2. Software dependencies 

This version of the LISE solver code is written in C, and uses MPI to employ a distributed memory parallel execution model. It requires a C language compiler, a Linux operating environment, and depends on the MPI, ScaLAPACK, LAPACK, BLAS, and FFTW software libraries.

Note that ScaLAPACK depends on PBLAS, BLACS, MPI, and BLAS. PBLAS depends on BLACS, MPI, and BLAS. BLACS depends on MPI. Since Netlib ScaLAPACK 2.0.0, PBLAS and BLACS are bundled into the ScaLAPACK build. This leads to the simple dependency structure. 

```
LISE-SLDAsolver
|-- FFTW
|-- BLAS
|-- LAPACK
|   `-- BLAS
|-- MPI
|-- ScaLAPACK
    |-- BLAS
    |-- MPI
```

This version of LISE time-dependent code is written in C, uses a combination of MPI for distributed memory and CUDA for accelerated shared memory parallel execution models. It requires a C language compiler, the NVIDIA CUDA compiler driver, a Linux operating environment, and depends on the FFTW, LAPACK, CUFFT, MPI, and CUDA software libraries.

```
LISE-TDSLDA
|-- FFTW
|-- LAPACK
|   `-- BLAS
|-- MPI
|-- CUDA
|-- CUFFT
|   `-- CUDA
```
The LISE software library dependencies are widely supported on HPC systems worldwide, and there exist open-source versions of each, except for NVIDIA's 'mvcc' which is proprietary, that can be built on Linux server clusters. 

No reference versions of these libraries are packaged with the LISE software. References for obtaining the required software are provided for convenience. Most vendors provide a highly tuned variant of each library. However, installation-specific modifications to header files, routine names, apis, variable types, and routine parameters are not uncommon -see the build examples for more details. 

Reference versions of the LISE software dependencies: 

- BLAS: Netlib, http://netlib.org/blas/blas.tgz
- LAPACK: Netlib, http://netlib.org/lapack/lapack.tgz
- ScaLAPACK: Netlib, http://netlib.org/scalapack/scalapack-2.0.2.tgz
- FFTW: http://www.fftw.org/fftw-3.3.9.tar.gz 
- MPI: MPICH, http://www.mpich.org/static/downloads/3.4.1/mpich-3.4.1.tar.gz
- MPI: Open MPI, https://www.open-mpi.org/software/ompi/v4.1
- Linux: Ubuntu, https://ubuntu.com/download/server
- Linux: Debian, https://www.debian.org/distrib 
- Linux: Centos, https://www.centos.org/download 
- Linux: Fedora, https://getfedora.org/en/server/download
- Linux: RHEL, https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux
- Linux: FreeBSD,	https://www.freebsd.org/where
- CUDA: NVIDIA, https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
- CUFFT: NVIDIA, http://developer.nvidia.com/cufft
- C compiler: GNU GCC, https://gcc.gnu.org/install/download.html 
- C compiler: IBM XL, https://www.ibm.com/products/xl-cpp-linux-compiler-power
- C compiler: Intel ICC, https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top.html
- CUDA Compiler Driver: NVIDIA, https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
		  
******************************************************************

## 3. Target computer architectures  

The solver code is parallelized using MPI for a distributed memory network of Linux servers. 

The time-dependent code is parallelized using a hybrid execution model that combines MPI distributed memory and CUDA programming models (sm_35, or newer) for a network of Linux servers each endowed with NVIDIA GPUs as coprocessors for accelerating single program, multiple data parallel tasks.

******************************************************************

## 4. Compiling and linking the codes  

The GNU Make utility is used for simplicity. 

Edit file './LISE.defs' to the specifics of the target platform. The LISE parameters are set for 'summit.olcf.ornl.gov' by default, and assumes the environment (i.e. modules) is appropriately set. See comments in './LISE.defs'. 

Invocation(s) from './':

```
 	make clean
 	make -e 
```

Successful builds will copy program executables to './bin'. A copy of the programs is left in the respective directories './LISE-SLDAsolver', './LISE-TDSLDA'. 

'clean' removes the objects and programs from these directories only, not from './bin'. The programs in './bin' are overwritten on subsequent builds. 

NOTES:

--Implicit function resolution: 
A function that is used somewhere in a code but is not prototyped or declared is known as an implicit function. Such functions are implicitly declared on first use, must be resolved during linking, and the argument types must match those linked into the program text. 

On most high-capability supercomputers deployed in the US DOE system, the vendors or specific labs provide a packaged configuration software ecosystem that includes a proprietary branch of an open-source Linux server operating system, language compilers, network semantics, and a software stack that normally includes prebuilt versions of all LISE dependencies.

LISE routines 'broyden_min.c' and 'rotation.c' call BLAS and LAPACK routines. To appreciate the difficulties in developing a robust build system, consider the APIs and argument types of routines ddot(), dgemm(), dgesdd() as implemented in the Netlib reference, in IBM's ESSL, and in Intel's MKL - because LISE software uses these (and other) routines.

Netlib CBLAS / LAPACK / LAPACKE:
double cblas_ddot(const int, const double *, const int, const double *, const int);

void cblas_dgemm(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, const int, const int, const int, const double, const double *, const int, const double *, const int, const double, double *, const int);

lapack_int LAPACKE_dgesdd(int, char, lapack_int, lapack_int, double *, lapack_int, double *, double *, lapack_int, double *, lapack_int);

IBM's ESSL:
double ddot(int, const double *, int, const double *, int);

void dgemm(const char *, const char *, int, int, int, double, const void *, int, const void *, int, double, void *, int);

void dgesdd(const char *, int, int, void *, int, double *, void *, int, void *, int, double *, int, int *, int *);

(another example from <essl.h>)
#define dsyev      esvdsyev
#define _ESVINT int
#define _ESVI    _ESVINT *
void esvdsyev(const char *, const char *, _ESVINT, void *, _ESVINT, double *, double *, _ESVINT, _ESVI);

Intel's MKL:
double ddot(const int *, const double *, const int *, const double *, const int *);

void dgemm(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);

void dgesdd(const char *, const int *, const int *, double *, const int *, double *, double *, const int *, double *, const int *, double *, const int *, int *, int *);

--Underscores and (Sca)LAPACK installations:
Most compilers require distinct Fortran and C routine namespaces. It is common practice for an underscore '\_' to be appended to C routine names which will be called from Fortran. As an example, f2c has added this underscore to all the names in CLAPACK. Thus, Fortran uses 'call dgetrf()' while C uses 'dgetrf_();'. In such builds, the user must pass ALL arguments by reference, i.e. as pointers, since this is how Fortran works. This includes all scalar arguments. This restriction means that you cannot make a call with numbers directly in the parameter sequence. The vendors have supported this name mangling to some extent by providing wrappers to most routines they support.

See, 'https://www.netlib.org/clapack/readme', for more details.

******************************************************************

## 5. Running the programs

Successful builds will place a copy of the program executables in './bin':
xlise-static  
xlise-tdslda-gpu  
xlise-tdslda-gpu-rst  
xlise-tdslda-postprocess

Copy './bin/xlise-static' to:
'./LISE-TESTS/O20/Static/'
'./LISE-TESTS/Collisions_238U/Static/'
'./LISE-TESTS/Fission_240Pu/Static/'

Copy './bin/xlise-tdslda*' to: 
'./LISE-TESTS/O20/Time_Dependent/'
'./LISE-TESTS/Collisions_238U/Time_Dependent/'
'./LISE-TESTS/Fission_240Pu/Time_Dependent/'

Example job submission scripts and 'README.txt' files are provided in the respective test directories. The 'O20' test cases are designed to quickly accept both the solver and time-dependent builds. 

Note that './LISE-SLDAsolver/builds' is a directory that contains sample 'LISE.defs', makefiles, job submission scripts, and a log of a successful build of the solver on both Intel and IBM fabric:

'./LISE-SLDAsolver/builds/summit-ibmxl-essl'
LISE.defs  Makefile.summit-ibmxl-essl  lise-solver-build.log  myjob.lsf

'./LISE-SLDAsolver/builds/theta-intel-mkl'
LISE.defs  Makefile.theta-intel-mkl  lise-solver-build.log  myjob.cblt

NOTES:

-Job launchers:
Different HPC installations use distinct job launching semantics, and often substitute 'mpirun' with a specific launch command with the same intent. For example, 'theta.alcf.anl.gov', a Cray XC40, uses the Cobalt batch scheduler. 'aprun' is the equivalent of the 'mpirun' command used by ALPS in the Cray Linux Environment (CLE). 'summit.olcf.ornl.gov', a IBM AC922, uses IBM Spectrum Load Sharing Facility (LSF) as the batch scheduling system, and 'jsrun' is the equivalent to the 'mpirun' command. Etc.

-Tested architecture:
The target system has approximately 4600 nodes connected by a Mellanox InfiniBand network cluster. Each node is a IBM Power System AC922 architecture and is comprised of two IBM POWER9 processors and six NVIDIA Tesla V100 accelerators.

The user guide for the tested system ( https://docs.olcf.ornl.gov/systems/summit_user_guide.html ) provides an overview of the architecture of the target computer for this software, and covers the relevant software topics required to build software and run programs on the machine. 

