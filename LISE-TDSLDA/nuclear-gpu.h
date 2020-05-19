//#define LZCALC
//#define SAVEOCC 

// small number
__constant__ cufftDoubleReal d_eps;

//Minumum momenta on the lattice
__constant__ cufftDoubleReal d_kx_min;
__constant__ cufftDoubleReal d_ky_min;
__constant__ cufftDoubleReal d_kz_min;

/*   Isoscalar and isovector coupling constants */
__constant__ cufftDoubleReal d_gamma ;
__constant__ cufftDoubleReal d_gg ;
__constant__ cufftDoubleReal d_rhoc ;

__constant__ cufftDoubleReal d_c_rho_a0 ;
__constant__ cufftDoubleReal d_c_rho_b0 ;
__constant__ cufftDoubleReal d_c_rho_c0 ;
__constant__ cufftDoubleReal d_c_rho_a1 ;
__constant__ cufftDoubleReal d_c_rho_b1 ;
__constant__ cufftDoubleReal d_c_rho_c1 ;
__constant__ cufftDoubleReal d_c_rho_a2 ;
__constant__ cufftDoubleReal d_c_rho_b2 ;
__constant__ cufftDoubleReal d_c_rho_c2 ;
__constant__ cufftDoubleReal d_c_isospin;

__constant__ cufftDoubleReal d_c_rho_0 ;
__constant__ cufftDoubleReal d_c_rho_1;
__constant__ cufftDoubleReal d_c_gamma_0;
__constant__ cufftDoubleReal d_c_gamma_1;
__constant__ cufftDoubleReal d_c_laprho_0;
__constant__ cufftDoubleReal d_c_laprho_1;
__constant__ cufftDoubleReal d_c_tau_0  ;
__constant__ cufftDoubleReal d_c_tau_1 ;
__constant__ cufftDoubleReal d_c_divjj_0;
__constant__ cufftDoubleReal d_c_divjj_1;
__constant__ cufftDoubleReal d_c_j_0   ;
__constant__ cufftDoubleReal d_c_j_1  ;
__constant__ cufftDoubleReal d_c_divj_0;
__constant__ cufftDoubleReal d_c_divj_1;

__constant__ int d_c_Skyrme ;
__constant__ int d_c_iexcoul;

/*   Proton and neutron coupling constants */
__constant__ cufftDoubleReal d_c_rho_p;
__constant__ cufftDoubleReal d_c_rho_n ;
__constant__ cufftDoubleReal d_c_laprho_p;
__constant__ cufftDoubleReal d_c_laprho_n;
__constant__ cufftDoubleReal d_c_tau_p  ;
__constant__ cufftDoubleReal d_c_tau_n ;
__constant__ cufftDoubleReal d_c_divjj_p;
__constant__ cufftDoubleReal d_c_divjj_n;
__constant__ cufftDoubleReal d_c_j_p   ;
__constant__ cufftDoubleReal d_c_j_n  ;
__constant__ cufftDoubleReal d_c_divj_p ;
__constant__ cufftDoubleReal d_c_divj_n;

/* Pairing field couplings */
__constant__ cufftDoubleReal d_gg_p;
__constant__ cufftDoubleReal d_gg_n;

/* the following will be constant in global memory */
__constant__ int d_nx;
__constant__ int d_ny;
__constant__ int d_nz;
__constant__ int d_nxyz;
__constant__ int d_nwfip;
__constant__ cufftDoubleReal d_dxyz ;
__constant__ cufftDoubleReal d_dx ;
__constant__ cufftDoubleReal d_amu;
__constant__ cufftDoubleReal d_alpha;
__constant__ cufftDoubleReal d_beta;
__constant__ cufftDoubleReal d_e_cut ;
__constant__ cufftDoubleReal d_PI ; // pi ~ 3.1415926535897932 38462643383279502884197 
__constant__ cufftDoubleReal d_dtstep;
__constant__ cufftDoubleReal d_hbarc;
__constant__ cufftDoubleReal d_hbar2m;
__constant__ cufftDoubleReal d_xpow;
__constant__ cufftDoubleReal d_e2;

