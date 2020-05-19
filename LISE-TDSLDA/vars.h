/* 

Variable definitions

*/

#include <complex.h>

#include <fftw3.h>

#define CUBIC_CUTOFF
//#define SAVEOCC
//#define RSTRT
//#define LZCALC
//#define EXTBOOST
//#define BENCHMARK

typedef struct 

{

  int nxyz ;

  double * amu ;

  double * u_re ;

  double * u_im ;

  double * mass_eff ;

  double * wx ;

  double * wy ;

  double * wz ;

  double * ugrad_x , * ugrad_y , * ugrad_z ;

  double * mgrad_x , * mgrad_y , * mgrad_z ;

  double * u1_x , * u1_y , * u1_z ;

  double * w_abs ;

  double complex * delta ;

  double complex * delta_ext ;

  double * v_ext ;

  double * rv_cm ;

  double m_tot ;

  double complex w0 ;

  double * a_vf[ 4 ] ;

  double ctime;

} Potentials ;

typedef struct 

{

  double * rho ;

  double * tau ;

  double * divjj ;

  double * jx , * jy , * jz ;

  double * cjx , * cjy , * cjz ;

  double * sx , * sy , * sz ;

  double complex * nu ;

  double lx , ly , lz ;

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

  double c_isospin;
  
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

// For skyrme forces specifically.
  double gg_p ;

  double gg_n ;
//

  double rhoc;

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

  double k1x , k1y , k1z ;

} Lattice_arrays ;

typedef struct

{

  int nxyz3 ;

  double complex * buff , * buff3 ;

  fftw_plan plan_f , plan_b , plan_f3 , plan_b3 ;

  int * i_l2s , * i_s2l ;  /* mapping of indexes from the large to small lattices (i_l2s) and reverse */

  double * fc ;

} FFtransf_vars ;

typedef struct

{

  double complex ** wavf[ 2 ] ;

  double complex ** wavf_t_der[ 4 ] ;

  double complex ** wavf_predictor[ 2 ] , ** wavf_corrector[ 2 ] , ** wavf_modifier ;

  double complex ** deriv_x , ** deriv_y , ** deriv_z ;

} Wfs ;


typedef struct
{
  double * thetaL;
  
  double * thetaR;
  
  double * densf_n;
  
  double * densf_p;
  
  
} Fragments ;
