The file explains how to run a time dependent nuclear collisions calculation using the LISE time dependent code for a U238 + U238 system. In this test case the impact parameter is set to 0 fm and the initial collision energy in the center of mass frame is set to 800 MeV (the total TKE of the incident system = KE_L + KE_R + Coul).  Collisions are programmed to end when center of masses of the two nuclei are seperated by 50fm.

*** Setup ***
In order to time evolve the wfs the following five files are needed in the working directory: lise-tdslda-gpu, myjob_TD.lsf, wf_p.cwr, wf_n.cwr, info.slda_solver.

*** Inputs ***
The lise-tdslda-gpu is the executable, generated from compiling the code in the LISE-TDSLDA folder (the makefile should be modified depending on the architecture of the machine the simulations are run on). NOTE: for collisions the ifdef EXTBOOST must be defined in the header file vars.h before compiling the code.

wf_p.cwr, wf_n.cwr are the quasiparticle wf files generated from the static solver.
	
The info.slda_solver file is also generated from the static solver and contains the following information about the wfs (in this order): the number of proton wfs, the number of neutron wfs, the proton chemical potential, the neutron chemical potential, the lattice spacing the x,y,z directions, the number of lattice points in the x,y,z directions, and the cutoff for the energy.

The files info.slda_solver, wf_p.cwr, and wf_n.cwr should be copied from the static solver to the run directory.

The myjob_TD.lsf file is a script used to run the job (will change depending on system). In the jsrun command line:

	jsrun -n 900 -a 1 -g 1 ./lise-tdslda-gpu -g 1 -f 7 -i 1 -s 300000 -t 43200 -e 0 -p 5 -m 5 -v 800.0 -b 0.0 >& 238U_TD.out

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
	-v -- the collision energy of each nucleus on the center of mass frame (KE - coul).
	-b -- the impact parameter.

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
	238U_TD.out

	238U_TD.out is the standard output file, which will contain information at each iteration, such as the particle numbers, total energy, individual components of the energy, and so forth. results_td.dat contains similar information, without explanatory text.  The first 4 columns are time, total energy, number of protons, and number of neutrons. 

*** Expected Results ***
At the end of the U238_TD.out file expect the following results (before the line "program terminated"):

	TIME[97800] Ctime=0.000000
	time_current = 823.061378 time_step = 97800 w0 = 0.000000 distance = 50.023054
	pairing energy: protons = -0.276103, neutrons = -0.474205
	pairing gap:  protons = 0.077633, neutrons = 0.106308
	#  protons =   184.000001
	# neutrons =   292.000012
	e_kin= 9238.288314 e_rho= -14334.470737 e_rhotau=    0.000000 e_laprho=  414.865647 e_so= -168.022037 e_coul= 2086.031604 e_j=   -0.714451 e_flow =   329.959520 e_ext =     0.000000 e_cm =     0.000010
	field energy: -13920.319542
	total energy: -2764.771970

	(nx,ny,nz)[ 24 24 64 ] nwf[ 147456 ] PEs[ 900 ] timesteps[ 97795 ] Wtime[ 1590799028.637689 ] Ctime[ 8323.530000 ]
	program terminated after CP
	Step completed, time used 21511.986388 s
	root closing FILE 1361319488
	program terminated after CP
	 program terminated

*** Checkpoint Restart ***
To restart the code we need an additional two files: restart.info and lise-tdslda-rst-gpu.

The restart.info file will be updated when we reach a number of time steps equal to loop_cp (defined in ctdslda.c in the LISE_TDSLDA folder).  This is currently set to loop_cp = (int) floor(500.0/dt_step), where dt_step is our time spacing.  This file records the last time step.

The lise-tdslda-rst-gpu file is an additional executionable that can be copied from the LISE_TDSLDA folder after the code has been compiled.

The last step is to modify the run line in myjob_TD.lsf to include the new executable (shown in the following line):

	jsrun -n 900 -a 1 -g 1 ./lise-tdslda-rst-gpu -g 1 -f 7 -i 1 -s 300000 -t 43200 -e 0 -p 5 -m 5 -v 800.0 -b 0.0 >& 238U_TD.out

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
|-- 238U_TD.out
|-- README.txt
|-- dens_all_n.dat.0
|-- dens_all_n.dat.100
|-- dens_all_n.dat.200
|-- dens_all_n.dat.300
|-- dens_all_n.dat.400
|-- dens_all_n.dat.500
|-- dens_all_n.dat.600
|-- dens_all_n.dat.700
|-- dens_all_n.dat.800
|-- dens_all_n.dat.900
|-- dens_all_p.dat.0
|-- dens_all_p.dat.100
|-- dens_all_p.dat.200
|-- dens_all_p.dat.300
|-- dens_all_p.dat.400
|-- dens_all_p.dat.500
|-- dens_all_p.dat.600
|-- dens_all_p.dat.700
|-- dens_all_p.dat.800
|-- dens_all_p.dat.900
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
47185920 	-- dens_all_p.dat.X
47185920 	-- dens_all_n.dat.X
173946175488	-- wf_n.cwr
173946175488	-- wf_p.cwr
68 		-- info.slda_solver
