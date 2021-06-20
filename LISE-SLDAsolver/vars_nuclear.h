/* 

Variable definitions

*/

//#define CONSTRCALC
//#undef CONSTR_Q0

#define PI 3.1415926535897932 /* pi ~ 3.1415926535897932 38462643383279502884197 */

#include <complex.h>

#include <fftw3.h>

typedef struct 

{

  int nxyz ;

  double * amu ;

  double * u_re ;

  double * mass_eff ;

  double * wx ;

  double * wy ;

  double * wz ;

  double complex * delta ;

  double complex * delta_ext ;

  double * v_ext ;

  double * v_constraint ;

  double * lam , * lam2; // extra constraints, at the moment only q2is used, two other possible

  double amu2;

} Potentials ;

typedef struct 

{

  int nstart ;

  int nstop ;

  double * rho ;

  double * tau ;

  double * divjj ;

  double complex * nu ;

  double complex * phases;  // e^(i\phi)

  double * jjx, * jjy, * jjz;

  double * jx , * jy , * jz ;

} Densities ;

typedef struct 

{
  double c_rho_a0 ;

  double c_rho_b0 ;

  double c_rho_c0 ;

  double c_rho_a1 ;

  double c_rho_b1 ;

  double c_rho_c1 ;

  double c_rho_a2 ;

  double c_rho_b2 ;

  double c_rho_c2 ;
  

  double c_rho_0 ;

  double c_rho_1 ;

  double c_laprho_0 ;

  double c_laprho_1 ;

  double c_tau_0 ;

  double c_tau_1 ;

  double c_divjj_0 ;

  double c_divjj_1 ;

  double c_gamma_0 ;

  double c_gamma_1 ;

  double c_j_0 ;

  double c_j_1 ;

  double c_divj_0 ;

  double c_divj_1 ;

  double gamma ;

  double c_rho_p ;

  double c_rho_n ;

  double c_laprho_p ;

  double c_laprho_n ;

  double c_tau_p ;

  double c_tau_n ;

  double c_divjj_p ;

  double c_divjj_n ;

  double c_j_p ;

  double c_j_n ;

  double c_divj_p ;

  double c_divj_n ;

  double gg ;   /* pairing strength */

  double gg_n ; /* pairing strength for neutrons */

  double gg_p ; /* pairing strength for protons */

  double rhoc; /// for surface pairing

  int iexcoul; // exchange couloumb

  int Skyrme; // 1 if skyrme EDF, 0 if SeaLL1

} Couplings ;

typedef struct

{

  double * xa ;

  double * ya ;

  double * za ;

  double * kx ;

  double * ky ;

  double * kz ;

  double * kin ;

  double lx , ly , lz ; 

  int * wx, * wy , *wz;

} Lattice_arrays ;

typedef struct

{

  int nxyz3 ;

  double complex * buff , * buff3 ;

  fftw_plan plan_f , plan_b , plan_f3 , plan_b3 ;

  int * i_s2l , * i_l2s ;

  double * fc ;

  double * filter ;

} FFtransf_vars ;


#define MAX_REC_LEN 1024

typedef struct
{
  int * npts_xy;  // number of points need to be calculated in xy plane

  int * npts_xyz; // number of points need to be calculated in 3d system; npxyz = npxy * nz

  int * ind_xyz; // indices of points in xyz need to be calculated, length: npxyz 

  int * car2cyl; // indices of points in rz for points in xyz, length: nxyz
  
} Axial_symmetry;


typedef struct
{
  int nx;               // lattice number nx

  int ny;               // lattice number ny

  int nz;               // lattice number nz

  double dx;            // lattice constant dx

  double dy;            // lattice constant dy

  double dz;            // lattice constant dz

  int broyden;          // if broyden = 1, then broyden minimization will be turned on, if broyden = 0; linear mixing is performed

  int coulomb;          // if coulomb = 1, coulomb potential will be added, if coulomb = 0, no coulomb potential will be added

  int niter;            // largest iteration number for self-consistent iterations

  double nprot;            // proton number

  double nneut;            // neutron number

  int iext;             // external potential choice

  int force;            // type of functional be chosen, force=1 -> Sly4 

  int pairing;          // pairing = 0, then there will be no pairing

  double alpha_mix;     // mixing factor in linear  mixing

  double ecut;          // energy cut-off

//  int icub; // 0 for spherical cutoff, 1 for cubic cutoff

  int imass; // 0 for same proton neutron mass, 1 for different masses

  int icm; // 0 for no correction, 1 for correction

  int irun;      // irun=0, start from a gaussian-like density

  int isymm;    // isymm = 0, full lattice calculated; isymm = 1, axial symmetry performed

  int resc_dens;  // whether rescale the potential each time. default = 1 

  int iprint_wf;  // choice of printing wave-function

  int deformation;  // deformation = 0 -> spherical

  double q0;  // expected value of quadrupole moment(default value 0)

  double v0;  // strength of external field

  double z0;  // reference point in fission process

  double wneck;  // parameter of twisted potential

  double rneck; // parameter of twisted potential

  int p;   // pproc

  int q;    // qproc

  int mb;   // grid size

  int nb;   // grid size

  double ggn; // couplings constant for neutrons

  double ggp;  // coupling constant for protons
 
  double alpha_pairing; // pairing mixing parameter: 0 for volume, 0.5 for mixed, 1.0 for surface (default is volume).
 
}metadata_t;
