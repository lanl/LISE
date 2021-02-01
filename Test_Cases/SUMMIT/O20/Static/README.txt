The file explains how to generate coordinate space solutions for an O20 system from scratch using the LISE static solver.

*** Setup ***
In order to generate HFB 20O wfs from scratch the following three files are needed in the working directory: input_file.txt, lise-static, myjob.lsf.

*** Inputs ***
The lise-static is the executable, generated from compiling the code in the LISE-SLDAsolver folder (the makefile should be modified depending on the architecture of the machine the simulations are run on). 

The input_file.txt defines the parameters of the simulation:
	nx 8 -- the number of lattice sites along the x direction.
	ny 8 -- the number of lattice sites along the y direction.
	nz 8 -- the number of lattice sites along the z direction.
	dx 1.25 -- the lattice spacing along the x direction.
	dy 1.25 -- the lattice spacing along the y direction.
	dz 1.25 -- the lattice spacing along the z direction.
	broyden 0 -- if 0 linear mixing of potentials is used, if 1 broyden mixing of potentials is used.
	coulomb 1 -- determines if we include the coulomb interaction (0 for not include, 1 for include).
	niter 100 -- number of self consistent iterations.
	N 12 -- number of neutrons.
	Z 8 -- number of protons.
	iext 0 -- if different from 0 we include a external potential.
	force 7 -- determines the density functional (0 for none, 1 for SLy4, 7 for SeaLL1,...).
	pairing 1 -- determines if we include the pairing interaction (0 for not include, 1 for include).
	alpha_mix 0.25 -- determines coefficient for linear mixing of potentials (0.00 corresponds to using the old potentials, 1.00 corresponds to new ones).
	resc_dens 0 -- if 1 will scale initial densities to correct number of particles (if 0 no change).
	ecut 100.0 -- energy cutoff in MeV used for the pairing renormalization scheme.
	irun 0 -- different run modes for solver (0 for scratch, 1 for reading from densities, 2 for reading from potentials).
	icub 1 -- if 1 we use a cubic momentum cutoff, if 0 we use a spherical momentum cutoff.
	imass 0 -- if 1 we use different proton and neutron masses, if 0 we use the same mass.
	icm 0 -- if 0 we don't include the center of mass correction, if 1 we do.
	isymm 0 -- if 0 full coordinate space solver, if 1 cylindrical solver.
	print_wf 0 -- print mode for wfs (-1 no printing, 0 all wfs printed, 1 only proton wfs, 2 only neutron wfs).
	p 7 -- dimension of group processor grid (note number of cpus = 2*p*q due to isospin).	
	q 3 -- dimension of group processor grid.
	mb 40 -- dimension of block for block cyclic decomposition used in diagonalizing H.
	nb 40 -- dimention of block for block cyclic decomposition use in diagonalizing H.

The myjob.lsf file is a script used to run the job (will change depending on system). In the jsrun command line:

	jsrun -n 42 -a 1 -g 0 ./lise-static input_file.txt > O20.out

	-n -- refers to number of resource sets.
	-a -- number of cpus per resource set.
	-g -- number of gpus per resrouce set.

*** Running ***
With all inputs set correctly, use the command "bsub myjob.lsf" to run the program.

*** Outputs ***
When the code is finished running (for this working example it should be under 10 mins) the following outputs will be printed:

	dens_0.dat
	dens_1.dat
	dens_p_0.dat
	dens_p_1.dat
	dens_n_0.cwr
	dens_n_1.cwr
	dens_p_0.cwr
	dens_p_1.cwr

	pots_0.dat
	pots_1.dat
	pots_p_0.dat
	pots_p_1.dat
	pots_n_0.cwr
	pots_n_1.cwr
	pots_p_0.cwr
	pots_p_1.cwr

	dens_n_start_info.txt
	dens_p_start_info.txt
	pots_n_start.txt
	pots_p_start.txt

	These files contain the densities and potentials.  Every iteration we save both the old and the new densities/potentials (denoted with _0/1) in both text format (.dat) and binary format (.cwr).  The files with _start_ are the initial guess for the densities and potentials.  

	The density files contain (in the following order) the number density, kinetic density, divergence of the current density, and anomalous (pairing) density.  Additionally the chemical potential and other constraints (currently unused).  In the text files the order is the number density, anomalous density, kinetic density, and divergence of the current density with no chemical potential or other constraints included.

	The potential files contain (in the following order) nx, ny, nz, dx, dy, dz, U (the central potential), the effective mass, the pairing field, the spin orbit potential (a 3-d vector field), and 4 constraints including the chemical potential.  The text files do no include the contraints or the lattice parameters.

	All the densities and potentials are real valued except for the pairing field and anomalous density which are complex.  All of them contain nx*ny*nz elements order with ix being the outer most index followed by iy followed by iz.  For example the first elements of the number density would be n(ix=0,iy=0,iz=0), n(ix=0,iy=0,iz=1), n(ix=0,iy=1,iz=0), n(ix=0,iy=1,iz=1), n(ix=1,iy=0,iz=0),... etc.  This convention is the same for all spatial dependent quantities in both the time dependent and static codes (such as the wfs).

	out_n.txt
	out_p.txt

	These text files contain the iteration number, number of neutrons/protons, and number of wfs.

	qpe_p.cwr
	qpe_n.cwr

	These binary files contain the quasiparticle energies.

	wf_n.cwr
	wf_p.cwr
	info.slda_solver

	These files are needed to run the time dependent code. wf_n.cwr, wf_p.cwr contain the quasi-particle wfs.  We should have 2*nx*ny*nz wfs each containing 4*nx*ny*nz components for both neutrons and protons seperately.  The info.slda_solver contains the following information (in this order): the number of proton wfs, the number of neutron wfs, the proton chemical potential, the neutron chemical potential, the lattice spacing the x,y,z directions, the number of lattice points in the x,y,z directions, and the cutoff for the energy.

	O20.out

	This is the standard output file, which will contain information at each iteration, such as the particle numbers, chemical potentials, total energy, individual components of the energy, errors, and so forth.

*** Expected Results ***
At the end of the O20.out file expect the following results (right before the line "entering MPI only write"):

	neutron pairing: -5.06125
	Delta_n =  1.53531 MeV

	proton  pairing: -0.00288
	Delta_p =  0.04030 MeV


	Iteration number 100, mu_n=   -5.873919
	N_n =  11.999930019
	Nwf_n = 1024
	e_kin=    331.0460 e_rho=   -518.2871 e_rhotau=      0.0000 e_laprho=     40.5162 e_gamma=      0.0000 e_so=     -9.5894 e_coul=     13.5963
	field energy:  -477.770943
	total energy:  -147.782149

	Iteration number 100, mu_p=  -12.933009
	N_p =   8.000735777
	Nwf_p = 1024
	err_n =   0.000804888

	err_p =   0.018926447

*** Tree Before ***
Static
|-- README.txt
|-- input_file.txt
|-- lise-static
`-- myjob.lsf

*** Tree After ***
Static
|-- O20.out
|-- README.txt
|-- dens_0.dat
|-- dens_1.dat
|-- dens_n_0.cwr
|-- dens_n_1.cwr
|-- dens_n_start_info.txt
|-- dens_p_0.cwr
|-- dens_p_0.dat
|-- dens_p_1.cwr
|-- dens_p_1.dat
|-- dens_p_start_info.txt
|-- info.slda_solver
|-- input_file.txt
|-- lise-static
|-- myjob.lsf
|-- out_n.txt
|-- out_p.txt
|-- pots_n_0.cwr
|-- pots_n_1.cwr
|-- pots_n_start.txt
|-- pots_p_0.cwr
|-- pots_p_1.cwr
|-- pots_p_start.txt
|-- qpe_n.cwr
|-- qpe_p.cwr
|-- wf_n.cwr
`-- wf_p.cwr

*** Size of Files in Bytes ***
28740 		-- pots_p_0.cwr
28740 		-- pots_p_1.cwr
28740 		-- pots_n_0.cwr
28740 		-- pots_n_1.cwr
20556 		-- dens_p_0.cwr
20556 		-- dens_p_1.cwr
20556 		-- dens_n_0.cwr
20556 		-- dens_n_1.cwr
33554432 	-- wf_n.cwr
33554432 	-- wf_p.cwr
68 		-- info.slda_solver
