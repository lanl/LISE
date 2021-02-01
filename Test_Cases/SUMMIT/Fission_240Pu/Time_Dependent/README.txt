The file explains how to time evolve hfb wfs for fission using the LISE time dependent code. In this specific example we are evolving 240Pu wfs until the center of masses of the fission fragments are seperated by a distance of 30fm. 

*** Setup ***
In order to time evolve the 240Pu wfs the following five files are needed in the working directory: lise-tdslda-gpu, myjob_TD.lsf, wf_p.cwr, wf_n.cwr, info.slda_solver.

*** Inputs ***
The lise-tdslda-gpu is the executable, generated from compiling the code in the LISE-TDSLDA folder (the makefile should be modified depending on the architecture of the machine the simulations are run on). 

wf_p.cwr, wf_n.cwr are the quasiparticle wf files generated from the static solver.
	
The info.slda_solver file is also generated from the static solver and contains the following information about the wfs (in this order): the number of proton wfs, the number of neutron wfs, the proton chemical potential, the neutron chemical potential, the lattice spacing the x,y,z directions, the number of lattice points in the x,y,z directions, and the cutoff for the energy.

The files info.slda_solver, wf_p.cwr, wf_n.cwr should be copied from the static solver into the run directory.

The myjob_TD.lsf file is a script used to run the job (will change depending on system). In the jsrun command line:

	jsrun -n 900 -a 1 -g 1 ./lise-tdslda-gpu -g 1 -f 7 -i 0 -s 200000 -t 43200 -e 0 -p 5 -m 5 >& 240Pu_TD.out

	-n -- refers to number of resource sets.
	-a -- number of cpus per resource set.
	-g -- number of gpus per resrouce set.
	-f -- the energy density functional (7 is SeaLL1). This should be consistent with static solver.
	-i -- the run type (0 is fission or standard, 1 is collisions, 2 is a test with plane waves).
	-s -- total number of time steps.
	-t -- total requested run time in seconds.
	-e -- type of external potential (0 for none).
	-p -- number of bootstraps steps before ABM method.
	-m -- order of exponential expansion for Hamiltonian.

*** Running ***
With all inputs set correctly, use the command "bsub myjob_TD.lsf" to run the program.

*** Outputs ***
When the code is finished running (for this working example it should be under 10 mins) the following outputs will be printed:

	dens_all_n.dat.X
	dens_all_p.dat.X

	These files contain the time dependent densities every 10 times steps (each file contains 10 sets of densities).  The densities are recorded in the following order: the number density, kinetic density, spin density, divergence of the current density, spin current density, curl of the current density, and anomalous density (only double complex density).  
	
	occ_p.bn
	occ_n.bn
	eqp_p.bn
	eqp_n.bn
	lz_p.bn
	lz_n.bn

	These files contain the occupation numbers, quasiparticle energies, and single particle orbital angular momentum for protons and neutrons respectively.  In this case expect the files to be empty.  To get results define SAVEOCC in the "vars.h" file when compiling the code.

	wf_n.cwr
	wf_p.cwr

	The wfs are overwritten at the final time.

	results_td.dat
	240Pu_TD.out

	240Pu_TD.out is the standard output file, which will contain information at each iteration, such as the particle numbers, total energy, individual components of the energy, and so forth. results_td.dat contains similar information, without explanatory text.  The first 4 columns are time, total energy, number of protons, and number of neutrons. 

*** Expected Results ***
At the end of the 240Pu_TD.out file expect the following results (before the line "program terminated"):

	TIME[108000] Ctime=0.000000
	time_current = 908.902135 time_step = 108000 w0 = 0.000000 distance = 30.030177
	pairing energy: protons = -2.205592, neutrons = -3.469282
	pairing gap:  protons = 0.353556, neutrons = 0.387273
	#  protons =    94.005036
	# neutrons =   145.999987
	e_kin= 4541.957112 e_rho=  -7226.193541 e_rhotau=    0.000000 e_laprho=  276.831146 e_so= -111.271508 e_coul=  719.883636 e_j=   -0.089064 e_flow =    68.488263 e_ext =     0.000000 e_cm =     0.003340
	field energy: -6949.451459
	total energy: -1804.557093

	(nx,ny,nz)[ 24 24 48 ] nwf[ 110592 ] PEs[ 900 ] timesteps[ 107995 ] Wtime[ 1590798069.452138 ] Ctime[ 6583.650000 ]
	program terminated after CP
	program terminated after CP
	Step completed, time used 14598.629031 s
	root closing FILE 1317734400
	 program terminated

*** Checkpoint Restart ***
To restart the code we need an additional two files: restart.info and lise-tdslda-rst-gpu.

The restart.info file will be updated when we reach a number of time steps equal to loop_cp (defined in ctdslda.c in the LISE_TDSLDA folder).  This is currently set to loop_cp = (int) floor(500.0/dt_step), where dt_step is our time spacing.  This file records the last time step.

The lise-tdslda-rst-gpu file is an additional executionable that can be copied from the LISE_TDSLDA folder after the code has been compiled.

The last step is to modify the run line in myjob_TD.lsf to include the new executable (shown in the following line):

	jsrun -n 900 -a 1 -g 1 ./lise-tdslda-rst-gpu -g 1 -f 7 -i 0 -s 200000 -t 43200 -e 0 -p 5 -m 5 >& 240Pu_TD-rst.out

*** Tree Before ***
Time_Dependent/
|-- README.txt
|-- info.slda_solver
|-- lise-tdslda-gpu
|-- myjob_TD.lsf
|-- wf_n.cwr
`-- wf_p.cwr

*** Tree After ***
Time_Dependent/
|-- 240Pu_TD.out
|-- README.txt
|-- dens_all_n.dat.X
|-- dens_all_p.dat.X
|-- eqp_n.bn
|-- eqp_p.bn
|-- info.slda_solver
|-- lise-tdslda-gpu
|-- lz_n.bn
|-- lz_p.bn
|-- myjob_TD.lsf
|-- occ_n.bn
|-- occ_p.bn
|-- results_td.dat
|-- wf_n.cwr
`-- wf_p.cwr

*** Size of Files in Bytes ***
35389440 	-- dens_all_p.dat.X
35389440 	-- dens_all_n.dat.X
97844723712	-- wf_n.cwr
97844723712	-- wf_p.cwr
68 		-- info.slda_solver
