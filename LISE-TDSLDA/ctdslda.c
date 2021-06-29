// for license information, see the accompanying LICENSE file

//main driver 

#include <mpi.h>
#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <assert.h>
#include <fcntl.h>
#include <time.h> 
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include "vars.h"
#include "tdslda_func.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "nuclear-gpu.h"

#define rank 3
const double PI=3.141592653589793238462643383279502884197; 

static double t_gettimeofday ;
static clock_t t_clock, t_clock1 ;
static struct timeval s, is ;

double com_tim = 0.;

void b_t( void ) 
{ /* hack together a clock w/ microsecond resolution */
  gettimeofday( &s , NULL ) ;
  t_clock = clock() ;
  t_gettimeofday = s.tv_sec + 1e-6 * s.tv_usec ;
}

double e_t( int type ) 
{
  switch ( type ) 
    {
    case 0 :
      t_clock1 = clock() ;
      gettimeofday( &s , NULL ) ;
      t_clock = t_clock1 - t_clock ;
      t_gettimeofday = s.tv_sec + 1e-6 * s.tv_usec - t_gettimeofday ;
      return t_gettimeofday ;
    case 1 :
      return t_gettimeofday ;
    case 2 :
      return t_clock / ( double ) CLOCKS_PER_SEC ;
    }
  return t_gettimeofday ;
}

//basically the same ideas as the global timer but now internal scope of variables 
void b_it( double *itgtod, clock_t *it_clock )
{
  gettimeofday( &is , NULL ) ;
  *it_clock = clock() ;
  *itgtod = is.tv_sec + 1e-6 * is.tv_usec ;
}

double e_it( int type, double *itgtod, clock_t *it_clock )
{
  clock_t itclock1;
  switch ( type ) 
    {
    case 0 :
      itclock1 = clock() ;
      gettimeofday( &is , NULL ) ;
      *it_clock = itclock1 - *it_clock;
      *itgtod = is.tv_sec + 1e-6 * is.tv_usec - *itgtod ;
      return *itgtod ;
    case 2 :
      return *it_clock / ( double ) CLOCKS_PER_SEC ;
    }
  return *itgtod ;
}

void check_densities (double * copy_from, Densities * dens , const int nxyz , char * iso_label )
{

  int k, ktot=0;
  int i , j ;

  k = 0; //local count of problems 
  double xpart_g=0., xpart_c=0.;
  for ( i = 0 ; i < nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->rho[i] ) > 0.0000001 ) 
	{
	  printf( "rho_gpu: %g rho_cpu %g \n" , copy_from[i] , dens->rho[i] ) ;
	  k++;
	}
      xpart_g += copy_from[i];
      xpart_c += dens->rho[i];
    }
  printf( "N%s: GPU: %g CPU: %g\n" , iso_label , xpart_g*1.25*1.25*1.25 , xpart_c*1.25*1.25*1.25 );
  if ( k != 0 ) printf( "\t... RHO%s ERROR [%d]\nCPUvalue[%f]\n" , iso_label , k, dens->rho[7] ) ;
  ktot+=k;

  k = 0; 
  j = 0;
  for ( i = nxyz ; i < 2*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->tau[j] ) > 0.0000001 ) k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... TAU%s ERROR [%d]\n" , iso_label , k ) ;
  if ( k == 0 ) printf( "\t... TAU%s CORRECT\n" , iso_label ) ;
  ktot+=k;

  k = 0; 
  j = 0;
  for ( i = 2*nxyz ; i < 3*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->sx[j] ) > 0.0000001 ) //k++;
	k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... SX%s ERROR [%d]\n" , iso_label , k ) ;
  ktot+=k;

  k = 0; 
  j = 0;
  for ( i = 3*nxyz ; i < 4*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->sy[j] ) > 0.0000001 ) k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... SY%s ERROR [%d]\n" , iso_label , k ) ;
  ktot+=k;

  k = 0; 
  j = 0;
  for ( i = 4*nxyz ; i < 5*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->sz[j] ) > 0.0000001 ) k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... SZ%s ERROR [%d]\n" , iso_label , k ) ;
  ktot+=k;

  k = 0; 
  j = 0;
  for ( i = 5*nxyz ; i < 6*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->divjj[j] ) > 0.0000001 ) k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... DIVJJ%s ERROR [%d]\n" , iso_label, k ) ;
  ktot+=k;

  k = 0; //local count of problems 
  j = 0;
  for ( i = 6*nxyz ; i < 7*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->jx[j] ) > 0.0000001 ) k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... JX%s ERROR [%d]\n" , iso_label , k ) ;
  ktot+=k;

  k = 0; //local count of problems 
  j = 0;
  for ( i = 7*nxyz ; i < 8*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->jy[j] ) > 0.0000001 ) k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... JY%s ERROR [%d]\n" , iso_label , k ) ;
  ktot+=k;

  k = 0; //local count of problems 
  j = 0;
  for ( i = 8*nxyz ; i < 9*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->jz[j] ) > 0.0000001 ) k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... JZ%s ERROR [%d]\n" , iso_label , k ) ;
  ktot+=k;

  k = 0; //local count of problems 
  j = 0;
  for ( i = 9*nxyz ; i < 10*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->cjx[j] ) > 0.0000001 ) //k++;
	k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... CJX%s ERROR [%d]\n" , iso_label, k ) ;
  ktot+=k;

  k = 0; //local count of problems 
  j = 0;
  for ( i = 10*nxyz ; i < 11*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->cjy[j] ) > 0.0000001 ) k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... CJY%s ERROR [%d]\n" , iso_label , k ) ;
  ktot+=k;

  k = 0; //local count of problems 
  j = 0;
  for ( i = 11*nxyz ; i < 12*nxyz ; i++ )
    {
      if ( fabs(copy_from[i]-dens->cjz[j] ) > 0.0000001 ) k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... CJZ%s ERROR [%d]\n" , iso_label , k ) ;
  ktot+=k;

  k = 0; //local count of problems 
  j = 0;
  double dtmp_r,dtmp_z;
  for ( i = 12*nxyz ; i < 14*nxyz ; i+=2 )
    {
      dtmp_r=creal(dens->nu[j]);
      dtmp_z = cimag(dens->nu[j]);
      if ( fabs(copy_from[i]-dtmp_r )  > 0.0000001 || fabs(copy_from[i+1]-dtmp_z) > 0.0000001 )
	k++;
      j++ ;
    }
  if ( k != 0 ) printf( "\t... NU%s ERROR [%d]\n" , iso_label , k ) ;
  ktot+=k;

  if(ktot==0)
    printf( "dens%s CORRECT\n" , iso_label );
  else
    printf( "dens%s ERROR\n" , iso_label );

}

void mix_densities(cufftDoubleReal *copy_from, cufftDoubleReal *copy_from_new, cufftDoubleReal * densities, int dim, double *err)
{
  int i;

  *err = 0.0;
  for(i=0; i<dim; i++) *err+= fabs( (copy_from[i] - copy_from_new[i]) / 2.0)  ;

  for (i=0; i<dim; i++) copy_from[i] = (copy_from[i] + copy_from_new[i]) / 2.0;
  
  cudaMemcpy(densities, copy_from, dim*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
      
}

// functions as factor in time for fluctuations

double fact_f(double x,double xi,double xx0){ //switch on rotation, friction
  double pi = acos(-1.);
  if(x<xi)
    return 0.;
  if(x<xi+xx0)
    return pow(sin(.5*pi*(x-xi)/xx0),2);
  else
    return 1.;

}

double fact_rot(double ratio){

  double ratio1 = 8;
  double dratio = 0.2;


  return 1./(1.+exp( (ratio-ratio1)/dratio));

}



int set_gpu(int device);
void make_wallpot_gpu( cufftDoubleReal * qzz , cufftDoubleReal * cc_constr , cufftDoubleReal *xyz , cufftDoubleReal z0, int nxyz , cufftDoubleReal rneck , cufftDoubleReal wneck );

// MPI-only version for writing wavefunctions
int write_wf(char *fn, int nwf, int nxyz, const int ip , const int np , const MPI_Comm comm , MPI_Status * status, double complex **wf, int * wf_tbl);
int write_wf_MPI(char *fn, int nwf, int nxyz, const int ip , const int np , const MPI_Comm comm , MPI_Status * status, double complex **wf, int * wf_tbl);

void check_time_reversal_symmetry(int knxyz, double complex **wavef, int nwfip, int *wavef_index, double * norms_tr, int *labels_tr, double dxyz);

void do_copy_cp_gpu( const int nn_time, const int n0_time, cufftDoubleComplex * wavf, cufftDoubleComplex *wavf_c, cufftDoubleComplex * wavf_p, int nxyz, int nwfip);

void check_ortho_gpu( int nwfip, int nxyz, cufftDoubleComplex * wavf, cufftDoubleReal * norms_ortho);

void mix_potentials(cufftDoubleReal * potentials, cufftDoubleReal * potentials_new, int dim, cufftDoubleReal c0)  ;

int make_boost_quadrupole(double complex * boost_field, const int n, Lattice_arrays * lattice_coords, double alpha);

int make_boost_twobody(double complex *boost_field, const int n, Lattice_arrays * lattice_coords, double ecm, double A_L, double A_R, double Z_L, double Z_R, double *rcm_L, double *rcm_R, int ip, double b, double ec);

int make_boost_onebody(double complex *boost_field, const int n, Lattice_arrays * lattice_coords, double ecm, double A);

int make_phase_diff(double complex *boostfield, const int n, Lattice_arrays * lattice_coords, double phi);

void make_localization (double complex *boost_field, const int n, Lattice_arrays * lattice_coords);

void do_boost_wf(cufftDoubleComplex * boost_field, cufftDoubleComplex * wavf ,int dim);

int read_qpe(char *fn, cufftDoubleReal * e_qp, const MPI_Comm comm, const int iam, const int nwf);
  
void tstep_gpu(const int nxyz , const int nwfip , const int mxp, const int n0_time, const int nn_time, cufftDoubleComplex * wavf, cufftDoubleComplex * wavf_psi, cufftDoubleComplex * wavf_hpsi, cufftDoubleComplex * wavf_mxp, cufftDoubleReal * coeffs, cufftDoubleComplex * grad, cufftDoubleComplex * lapl, cufftDoubleComplex * wfft, cufftDoubleComplex * fft3 , cufftHandle plan_b , cufftHandle plan_r , cufftDoubleReal * potentials, cufftDoubleReal * avf, const int batch, cufftDoubleReal * kxyz, cufftHandle plan_ev, cufftDoubleComplex * fft3_ev, cufftDoubleReal *xyz, cufftDoubleReal *rcm );

// from nuclear-kerns.cu
void chk_constants( int * nwfip , int * nxyz );
void chk_constants_two( double *hbar2m, double *hbarc, double *amu, double *e_cut, double *PI, double *dt_step );
void kenny_copy( int nx , int ny , int nz , int nwfip, double dxyz, double dx );
void kenny_copy_( int nx , int ny , int nz , int nwfip );
void kenny_copy_two( double hbar2m, double hbarc, double amu, double e_cut, double PI, double dt_step , double eps );

void shi_copy(double xpow, double e2);

void copy_nuc_params_to_gpu(Couplings * cc_edf);
void l_gpu_norm( int n , int nwfip , cufftDoubleComplex * wavf , cufftDoubleReal * gpu_norm , cufftDoubleReal * part_sum );
void compute_densities_gpu( const int nxyz , const int nwfip , cufftDoubleComplex * wavf , cufftDoubleComplex * grad, cufftDoubleComplex * lapl, cufftDoubleComplex * wfft , cufftDoubleComplex * fft3 , cufftDoubleReal * kxyz , cufftHandle plan_b , cufftHandle plan_r , cufftHandle plan_d, int batch , cufftDoubleReal * density, cufftDoubleReal * avf, const MPI_Comm comm , cufftDoubleReal * copy_to , cufftDoubleReal * copy_from , int isoff , cufftDoubleReal Ctime , double * divj);

void get_u_re_gpu( cufftDoubleReal * density, cufftDoubleReal * potentials, int nxyz, int isospin, cufftDoubleReal * kxyz, cufftDoubleComplex * fft3 , cufftHandle plan_b, cufftDoubleComplex * fft3_c,cufftHandle plan_c, cufftDoubleReal * fc, int * map_ls, int * map_sl , cufftDoubleReal *avf , cufftDoubleReal Ctime , cufftDoubleReal Ctime_qzz,int nxyz3, MPI_Comm gr_comm) ;

void update_potentials_gpu( cufftDoubleReal * potentials , int nxyz , cufftDoubleReal * density , cufftDoubleReal * fc, int * map_ls, int * map_sl, cufftHandle plan_c, cufftDoubleComplex * fft3_c , cufftHandle plan_b , cufftDoubleComplex * fft3 , cufftDoubleReal * kxyz , int isospin , cufftDoubleReal * avf , cufftDoubleReal ct, cufftDoubleReal ct_constr,MPI_Comm comm,cufftDoubleComplex *wavf,int batch,cufftDoubleReal *cc_pair_qf, cufftDoubleReal *copy_from,cufftDoubleReal *copy_to,int nwfip,int gr_ip,int nxyz3,int icub);

// from evolution-gpu.cu
void test_norm_gpu( int nxyz , int nwfip , cufftDoubleComplex * wavf , int batch, double * cpu_norm );
void adams_bashforth_pm_gpu( const int nn_time , const int n0_time , const int n1_time , const int m0_time , const int m1_time , const int m2_time , const int m3_time , const int nxyz , const int nwfip , cufftDoubleComplex * wavf , cufftDoubleComplex * wavf_td , cufftDoubleComplex * wavf_p , cufftDoubleComplex * wavf_c , cufftDoubleComplex * wavf_m );
void adams_bashforth_cy_gpu( const int nn_time , const int n0_time , const int n1_time , const int m0_time , const int m1_time , const int m2_time, const int nxyz , const int nwfip , cufftDoubleComplex * wavf , cufftDoubleComplex * wavf_td , cufftDoubleComplex * wavf_p , cufftDoubleComplex * wavf_c , cufftDoubleComplex * wavf_m , cufftDoubleComplex * grad, cufftDoubleComplex * fft3 , cufftHandle plan_b , cufftHandle plan_r , cufftDoubleComplex * df_dt , cufftDoubleReal * potentials,const int batch,cufftDoubleReal * kxyz,cufftHandle plan_ev,cufftDoubleComplex *fft3_ev, cufftDoubleReal *xyz, cufftDoubleReal *rcm );
void adams_bashforth_dfdt_gpu(const int nxyz , const int nwfip , cufftDoubleComplex * wavf , cufftDoubleComplex * wavf_td , cufftDoubleComplex * grad, cufftDoubleComplex * fft3 , cufftHandle plan_b , cufftHandle plan_r , cufftDoubleReal * potentials, const int batch, cufftDoubleReal * kxyz,int i_saveocc,int norm_switch,double * e_qp, double * e_sp, double * n_qp,cufftHandle plan_ev,cufftDoubleComplex *fft3_ev,cufftDoubleReal *xyz,cufftDoubleReal *rcm,double * lz_qp);
void check_densities(double * copy_from,Densities * dens , const int nxyz , char * iso_label );
void define_VF_gpu(cufftDoubleReal *avf,cufftDoubleReal *xyz, int nxyz,int isospin);
cufftDoubleReal gaussian_(cufftDoubleReal t,cufftDoubleReal t0,cufftDoubleReal sigma,cufftDoubleReal c0) ;
cufftDoubleReal theta(cufftDoubleReal t,cufftDoubleReal t0,cufftDoubleReal t1);
void external_potential_gpu(cufftDoubleReal *vext,cufftDoubleReal * xyz,int isospin,int iext,int nxyz,cufftDoubleReal str_p,cufftDoubleReal str_n);



void get_phase_imprint_potential(double *ext_phase, const int nxyz, Lattice_arrays * latt_coords, double time_interval, double phase_diff);
  
cufftDoubleReal non_sym_f_cpu(cufftDoubleReal t, cufftDoubleReal Ti,cufftDoubleReal T0,cufftDoubleReal T1,cufftDoubleReal TT,cufftDoubleReal c0);

cufftDoubleReal non_sym_f_cpu_collision_time(cufftDoubleReal t, cufftDoubleReal Ti,cufftDoubleReal T0,cufftDoubleReal T1,cufftDoubleReal TT,cufftDoubleReal c0, cufftDoubleReal c1);

cufftDoubleReal non_sym_f_cpu_collision_ecm(cufftDoubleReal t, cufftDoubleReal Ti,cufftDoubleReal T0, cufftDoubleReal c0, cufftDoubleReal c1);

cufftDoubleReal non_sym_f_cpu_random(cufftDoubleReal t, cufftDoubleReal t0, cufftDoubleReal dt);

void copy_gg(double gg);
void make_qzz_gpu( cufftDoubleReal * qzz , cufftDoubleReal * cc_constr , cufftDoubleReal *xyz , int nxyz , int iext , cufftDoubleReal v0 , cufftDoubleReal rneck , cufftDoubleReal wneck , cufftDoubleReal z0 );

#ifdef LZCALC
void calc_lz_dens( cufftDoubleReal * lz_dens , cufftDoubleReal * dens_lz , cufftDoubleReal * copy_to , cufftDoubleComplex * grad , cufftDoubleReal * xyz , cufftDoubleReal * rcm , int nxyz , int nwfip , MPI_Comm comm );

void calc_jz_dens( cufftDoubleReal * jz_dens , cufftDoubleReal * dens_jz , cufftDoubleReal * copy_to , cufftDoubleComplex * grad , cufftDoubleComplex * wavf, cufftDoubleReal * xyz , cufftDoubleReal * rcm , int nxyz , int nwfip , MPI_Comm comm );

void calc_jz2_dens( cufftDoubleReal * jz_dens , cufftDoubleReal * dens_jz , cufftDoubleReal * copy_to , cufftDoubleComplex * grad , cufftDoubleComplex * wavf, cufftDoubleReal * xyz , cufftDoubleReal * rcm , int nxyz , int nwfip , MPI_Comm comm );

double center_dist_pn( double * rho_p , double * rho_n , const int n , Lattice_arrays * latt_coords , double * xc , double * yc , double * zc );
#endif


// Get distance of fragments (in fission).
double get_distance(Densities * dens_p , Densities * dens_n, Fragments * frag, const int nxyz , const double dxyz , Lattice_arrays * latt , double *rcm_L, double *rcm_R, double *A_L, double *Z_L, double *A_R, double *Z_R);

double coul_frag( double * rho , double * xa , double * ya , double * za , int nxyz , double dxyz,double z0 );
// Functions tracks the rotation angle and velocity.
// Noted: the evolution functions are also updated, an extra omega . J term is added to reduce the rotation of the system.

///////////////////////////////////////////////

double system_energy( Couplings * cc_edf , Densities * dens , Densities * dens_p , Densities * dens_n , const int isospin , const int nxyz , double complex * delta , double * chi, const int ip , const int root_p , const int root_n , const MPI_Comm comm , const double hbar2m , const double dxyz , Lattice_arrays * latt_coords , FFtransf_vars * fftransf_vars , MPI_Status * status , const double time , FILE * fd );

double rotation(double *theta, double *rho_n, double *rho_p, int nxyz, Lattice_arrays * latt_coords, double dxyz );

void get_omega(double * phi_xs, double * phi_ys, double * phi_zs, int l0, int l1, int l2, int l3, int l4, int l5, double * omega, double dt_step);


int read_boost_operator(double complex *boost,char *filename,MPI_Comm commw, int ip, double alpha, int n);

/////////

// For testing only, generates a set of orthogonal plane wave wfs (self starting)
void generate_plane_wave_wfs( double complex ** wf , const int nxyz , const int nwf , const int ip , const int np , const MPI_Comm comm , char * fn , double dxyz , int *wavef_index , Lattice_arrays * latt) ;

int parse_input_file( char * dir_in , int * nx , int * ny , int * nz , int * nwf_p , int * nwf_n , double * amu_p , double * amu_n , double * dx , double * dy , double * dz , double * e_cut , double * cc_constr , const int ip , const MPI_Comm comm );
////////////////////////////////

int main( int argc , char ** argv )
{
  double req_tim , run_tim = 0. , est_wr_tim , est_bw ;

  //new internal timers
  double itgtod ;
  clock_t it_clk ;
  double iwt, ict;

  //Defining cubic or spherical cutoff here.
  int icub;
  icub = 1; // icub = 1 is cubic cutoff, icub = 0 is spherical cutoff. 
  
  MPI_Comm commw , gr_comm ;
  MPI_Group group_comm ;
  MPI_Status status ;
  int np , gr_np , gr_ip , ip , np_c ;
  int root_p , root_n ;
  int nwf_p , nwf_n , * nwf , nwf_tot ;
  int isospin, isoshift ;
  int nwfip ;
  int nx , ny , nz , nxyz , nx3 , ny3 , nz3 , nxyz3 , knxyz ;
  double dx , dy , dz , dxyz ;
  double Lc ;

  int option_index = 0 ;
	
  Wfs * wavef ;
  Densities dens_p , dens_n , * dens , densG_p , densG_n , * densG;
  Lattice_arrays latt , latt3 ;
  FFtransf_vars fftrans ;
  Potentials pots ;
  Couplings cc_edf ;

  Fragments frag;
	
  char iso_label[] = "_p" , * file_wf ,  file_tr[120], file_res[120] , file_random[120], * file_pot , file_dens_out[120],file_dens_all[120], file_dens_td[120];
  int iforce = 1 , ihfb = 1 ;
  int iabort = 0 , iflag ;
  int i , k ;
  int time_start = 0 , timeloop , total_time_steps = 25000000 , t_pp, loop_cp , loop_io, loop_rot;
  int time_bootstrap = 20;  // the bootstrap time step limit
  
  int itime;
  // Added by Shi Jin at 08/28/18:
  // time_kick: a timer to track the time between two kicks, every time it reach time_to_kick, it will be reset to 0.
  // kick_mean, kick_var:
  // time_to_kick = kick_mean +- kick_var
  // alpha_kick, a random number [0, 1] that control the strength of kick
  // R_kick, the random rotation matrix (3*3) that generate the boost field
  // a counter of number of expansion evolver after random kick
  //double kick_var, kick_mean, dt_random;  // time to kick the wavefunctions
  //double alpha_kick, R_kick[9];
  //double *time_kicks, *chis;//, *delta_v ;
  //int nkicks; // # of kicks to be implemented
  int nsteps_bootstrap;

  //////////////////////////////////
  setbuf(stdout, NULL);
  
#ifdef EXTBOOST
  double alpha = 1.0e-3; // strength of boost
#endif
  
  int nn_time , n0_time , n1_time , mm_time , m0_time , m1_time , m2_time , m3_time ;

  int l0_time , l1_time , l2_time , l3_time , l4_time , l5_time;
  
  double e_cut , emax , tcur, t0g = 10. ;
  char dir_in[120], file_extern_op[120] ;
  double amu_p , amu_n ;
  double * cc_qzz;
  double hbar2m , * qp_energy , * e_qp , * e_sp, * n_qp , *lz_qp ;
  double tolerance = 1.e-7 ;
  double *_e_qp_total, * _n_qp_total, * _e_sp_total, *e_qp_total, * n_qp_total, * e_sp_total;

  double mass_p = 938.272013 ;
  double mass_n = 939.565346 ;
  double mass = .5 * ( mass_p + mass_n ) ;
  double hbarc = 197.3269631 ;

  double xpow=1./3.;
  double e2 = -197.3269631*pow(3./acos(-1.),xpow) / 137.035999679 ;
	
  double dt_step ;
  double xpart , ratio ;
  double * buff ;
  double err;
  const int isave = 10 , iopen = 100 ;
  FILE * fd , * fd_rs_info , * fd_vel, *fd_tr , *fd_eqp_txt;
  FILE * fd_random;
  int ifd_dens,ifd_all, ifd_td , ifd_random ;
  double forces[ 6 ] , vel[ 3 ] ;
  const double v_proj = 0.7 ;
  double mtot ;
  const int Z_proj = 92 ;
  int irun = 0;  // nuclear fission
  int op_read_flag=-1;

  double alpha_pairing = 0.0; // pairing mixing parameter: 0 is volume, 0.5 is mixed, 1.0 is surface.
  double ggn = 1e10, ggp = 1e10; // pairing coupling constants.

  /*
     distance: distance between two fragments (in fm).
     ec: Coulomb energy between two fragments (in MeV).
     A_L, A_R: mass numbers of left (L) and right (R) fragments.
     Z_L, Z_R: proton numbers of left (L) and right (R) fragments.
     rcm_L, rcm_R: centos of mass coordinates of left (L) and right (R) fragments.
   */
  double distance;  // in fm
  double ec, A_L, A_R, Z_L, Z_R;
  double rcm_L[3], rcm_R[3];

  double d0 = 30.;  // in fm, if distance > d0, stop the evolution  

  double d1 = 15.; // in fm, if distance > d1, stop the random kicks

  int icp = 0; // check whether checkpoint (save wf) is done.

  // Shi Jin
  int mxp = 5; // expand the series up to 4th order in the evolution
  /////

  // Shi Jin
  double phi[3], omega[3]; // rotation angles and angular velocities
  double phi_xs[6], phi_ys[6], phi_zs[6];
  ////////
  
  MPI_Init( &argc , &argv ) ;
  MPI_Comm_rank( MPI_COMM_WORLD, &ip ); 
  MPI_Comm_size( MPI_COMM_WORLD, &np );
  commw = MPI_COMM_WORLD ;
  int gpupernode=1; // For titan, piz-daint
  //  int gpupernode=4; // For hyak, tsubame
  //int gpupernode=2; // For moonlight

  cufftDoubleReal c0=(cufftDoubleReal)(0.0);
  cufftDoubleReal c1=(cufftDoubleReal)(0.0);
 
  double e_tot, e_gs; // current total energy and initial total energy (gs)
  
  MPI_Barrier(commw);

  dir_in[0] = '.';
  dir_in[1] ='\0';

  int iext=0;
  double z0=0.,wneck=1000.,rneck=1000.;
  int p;

  // collisions
  // ecm: center of mass energy of two fragments (in MeV).
  // b: impact parameter.
  double ecm=0.,b=0.;

  // New variables related to normalization.
  int norm_switch = 0; // If this variable is one then we normalize the wfs.
  int i_norm = 10000000; // Number of time steps between normalizations.

  static struct option long_options[] = {
    {"gpupernode", required_argument, 0,  'g' },
    {"tsteps", required_argument, 0,  's' },
    {"directory", required_argument, 0,  'd' },
    {"c0", required_argument, 0,  'c' },
    {"iext", required_argument, 0,  'e' },
    {"iforce", required_argument, 0,  'f' },
    {"irun", required_argument, 0,  'i' },
    {"reqtim", required_argument, 0,  't' }, /* has to be in seconds */
    {"fileext", required_argument, 0, 'x'},
#ifdef EXTBOOST
    {"alpha", required_argument, 0,  'a' },
    {"ecm", required_argument, 0,  'v' },
    {"b", required_argument, 0,  'b' },
#endif
    {"mxp", required_argument, 0, 'm' },
    {"inorm", required_argument , 0,  'n' },
    {"tbstrap", required_argument, 0, 'p' },
    {"apair" , required_argument , 0 , 0 },
    {"ggp" , required_argument , 0 , 0 },
    {"ggn" , required_argument , 0 , 0 },
    {0,         0,                0,  0 }
  };

  while ( ( p = getopt_long(argc, argv, "g:s:d:a:c:t:e:i:f:v:b:x:m:n:p:", long_options, &option_index) ) != -1 )
    switch ( p )
    {
      case 0:
        if ( long_options[option_index].name == "gpupernode" )
          gpupernode=atoi(optarg);
        if ( long_options[option_index].name == "tsteps" )
          total_time_steps=atoi(optarg);
        if ( long_options[option_index].name == "directory" )
          strcpy(dir_in,optarg);
        if ( long_options[option_index].name == "c0" )
	  c0=(cufftDoubleReal) atof(optarg);
        if ( long_options[option_index].name == "iext" )
          iext=atoi(optarg);
        if ( long_options[option_index].name == "iforce" )
	  iforce=atoi(optarg);
        if ( long_options[option_index].name == "irun" )
	  irun = atoi(optarg);
        if ( long_options[option_index].name == "reqtim" )
	  req_tim=atof(optarg);
#ifdef EXTBOOST
        if ( long_options[option_index].name == "alpha" )
	  alpha=atof(optarg);
        if ( long_options[option_index].name == "ecm" )
	  ecm=atof(optarg);
        if ( long_options[option_index].name == "b" )
	  b=atof(optarg);
#endif
        if ( long_options[option_index].name == "mxp" )
	  mxp=atoi(optarg);
        if ( long_options[option_index].name == "inorm" )
	  i_norm=atoi(optarg);
        if ( long_options[option_index].name == "tbstrap" )
	  time_bootstrap=atoi(optarg);
        if ( long_options[option_index].name == "apair" )
          alpha_pairing = atof( optarg ) ;
        if ( long_options[option_index].name == "ggp" )
          ggp = atof( optarg ) ;
        if ( long_options[option_index].name == "ggn" )
          ggn = atof( optarg ) ;
      break;
      case 'g': gpupernode=atoi(optarg);break;
      case 's': total_time_steps=atoi(optarg);break;
      case 'd': strcpy(dir_in,optarg);break;
      case 'c': c0=(cufftDoubleReal) atof(optarg);break;
      case 'e': iext=atoi(optarg); break;
      case 'f': iforce=atoi(optarg); break;
      case 'i': irun = atoi(optarg);break;
      case 't': req_tim=atof(optarg);break; /* has to be in seconds */
      case 'x': strcpy(file_extern_op,optarg); op_read_flag=0;break;
#ifdef EXTBOOST
      case 'a': alpha=atof(optarg); break;
      case 'v': ecm=atof(optarg);break;
      case 'b': b=atof(optarg);break;
#endif
      case 'm': mxp=atoi(optarg);break;
      case 'n': i_norm=atoi(optarg);break;
      case 'p': time_bootstrap=atoi(optarg);break;
    }

  if(iext == 70)
    c1 = 0.2; // potential strength to push the nuclei to approach

  if(ip == 0)
    {
      if (irun == 0)
	printf("******* TDSLDA calculation for fission *********\n");
      
      if(irun == 1)
	printf("******* TDSLDA calculation for two nuclei collision *********\n");

      if (irun == 2){
	printf("******* TDSLDA calculation from plane waves input (test only) *********\n");
	d0 = 1.0e10;  // a very large number	
      }
    }
  

  int ierr=set_gpu(ip % gpupernode);
  if(ierr!=0) { 
    printf("GPU incorrectly set; program terminated");
    MPI_Abort( commw, ierr);
    return(EXIT_FAILURE);
  }

  if(ip==0)printf("test 2\n" );
  MPI_Barrier(commw);
  
  b_t() ; /* start running internal clock for check point */

  cc_qzz=malloc(4*sizeof(double));
  if(irun != 2)
    read_input_solver( dir_in , &nx , &ny , &nz , &nwf_p , &nwf_n , &amu_p , &amu_n , &dx , &dy , &dz , &e_cut , cc_qzz , ip , commw ) ;
  else
    parse_input_file( dir_in , &nx , &ny , &nz , &nwf_p , &nwf_n , &amu_p , &amu_n , &dx , &dy , &dz , &e_cut , cc_qzz , ip , commw ) ;

  if( ip == 0 ) printf(" %d %d \n %d %d %d \n %f %f %f\n" , nwf_p , nwf_n , nx , ny , nz , dx , dy , dz ) ; 
  dxyz = dx * dy * dz ;
  nxyz = nx * ny * nz ;

  double dx_=dx;
  int n3=nx;

  if( n3 < ny ){
    n3=ny;
    dx_=dy;
  }

  if( n3 < nz ){
    n3=nz;
    dx_=dz;
  }

  nx3 = 3 * n3 ;
  ny3 = 3 * n3 ;
  nz3 = 3 * n3 ;

  fftrans.nxyz3 = nx3 * ny3 *nz3 ;

  nxyz3 = fftrans.nxyz3 ;

  //CPU
  make_coordinates( fftrans.nxyz3 , nx3 , ny3 , nz3 , dx , dy , dz , &latt3 ) ;
  make_coordinates( nxyz , nx , ny , nz , dx , dy , dz , &latt ) ;
  dens_p.lx = nx * dx ;
  dens_p.ly = ny * dy ;
  dens_p.lz = nz * dz ;
  dens_n.lx = nx * dx ;
  dens_n.ly = ny * dy ;
  dens_n.lz = nz * dz ;
  	
  // GPU
  cufftDoubleReal * d_kxyz; /* gpu k-space lattice */
  cufftDoubleReal * d_latt_xyz;
  cudaError_t icudaError = cudaMalloc( (void **)&d_kxyz , 4*nxyz*sizeof( cufftDoubleReal ) ) ;
  if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_latt_xyz)\n");
  cudaMemcpy(d_kxyz          , latt.kx , nxyz*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ; 
  cudaMemcpy(d_kxyz +   nxyz , latt.ky , nxyz*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ; 
  cudaMemcpy(d_kxyz + 2*nxyz , latt.kz , nxyz*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ; 
  cudaMemcpy(d_kxyz + 3*nxyz , latt.kin, nxyz*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ; 
	
  //CPU
  Lc = n3*dx_*sqrt(3.);
  match_lattices( &latt , &latt3 , nx , ny , nz , nx3 , ny3 , nz3 , &fftrans , Lc ) ;
  assert( fftrans.buff = malloc( nxyz * sizeof( double complex ) ) ) ;
  assert( fftrans.buff3 = malloc( nxyz3 * sizeof( double complex ) ) ) ;
  fftrans.plan_f = fftw_plan_dft_3d( nx , ny , nz , fftrans.buff , fftrans.buff , FFTW_FORWARD , FFTW_ESTIMATE ) ;
  fftrans.plan_b = fftw_plan_dft_3d( nx , ny , nz , fftrans.buff , fftrans.buff , FFTW_BACKWARD , FFTW_ESTIMATE ) ;
  fftrans.plan_f3 = fftw_plan_dft_3d( nx3 , ny3 , nz3 , fftrans.buff3 , fftrans.buff3 , FFTW_FORWARD , FFTW_ESTIMATE ) ;
  fftrans.plan_b3 = fftw_plan_dft_3d( nx3 , ny3 , nz3 , fftrans.buff3 , fftrans.buff3 , FFTW_BACKWARD , FFTW_ESTIMATE ) ;
	
  //CPU
  if ( ip == 0 ) printf( " nwf_p = %d nwf_n = %d \n" , nwf_p , nwf_n ) ;
  isospin = create_mpi_groups( commw , &gr_comm , np , &gr_np , &gr_ip , &group_comm , nwf_p , nwf_n , &root_p , &root_n ) ;
  isoshift=7*nxyz*(1-isospin);

  //GPU	
  cufftDoubleReal * d_avf;
  icudaError = cudaMalloc( (void **)&d_avf, 4 * nxyz * sizeof(cufftDoubleReal) ) ;
  if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_avf)\n");
  cufftDoubleReal * d_xyz;
  icudaError = cudaMalloc( (void **)&d_xyz, 3 * nxyz * sizeof(cufftDoubleReal) ) ;
  if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_xyz)\n");
  cudaMemcpy(d_xyz,latt.xa,nxyz*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ; 
  cudaMemcpy(d_xyz+nxyz,latt.ya,nxyz*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ; 
  cudaMemcpy(d_xyz+2*nxyz,latt.za, nxyz*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ; 
  define_VF_gpu(d_avf,d_xyz,nxyz,isospin);

  // fluctuations
  double *chi;

  assert( chi = malloc(4*nxyz * sizeof( double ) ) ) ;

  for(i=0; i<4*nxyz;i++) chi[i] = 0.; // initialize

  double fact_f1 = 0.;
  ///////////////////////////
	
  for ( i = 0 ; i < 4 ; i++ )
    {
      assert( pots.a_vf[ i ] = malloc( nxyz * sizeof( double ) ) ) ;
      for( k = 0 ; k < nxyz ; k++ ) pots.a_vf[ i ][ k ] = 0. ;
    }
  np_c = gr_np - 1 ;
  if ( isospin == 0 )
    {
      if ( ip == 0 ) printf( "\n **** \n Error: the number of processors is larger than the number of wave functions, do not know how to handle\n Aborting execution \n **** \n\n" ) ;
      ierr = -1;
      MPI_Abort(commw, ierr) ;
      return( EXIT_FAILURE ) ;
    }
  hbar2m = pow( hbarc , 2.0 ) / ( mass_p + mass_n ) ;
  emax = 0.5 * pow( acos( -1. ) , 2. ) * hbar2m / pow( dx , 2. ) ;

if(icub==1)
  emax = 4*emax; // larger energy cutoff

  
  dt_step = pow( tolerance , 0.2 )  * hbarc / emax ;
  //IS: changed the time step to accommodate the finer lattice
  dt_step = .25*pow(10.,-5./3.)*dx*dx;

  loop_cp = (int) floor(500.0/dt_step); // number of steps for check point

  loop_io = 100; // do io for every 100 steps
  loop_rot = 100;

  if ( isospin == 1 )
    nwf = &nwf_p ;
  else
    {
      nwf = &nwf_n ;
      sprintf( iso_label , "_n" ) ;
    }

  // Shi Jin:
  assert(e_qp_total = (double *)malloc(*nwf * sizeof(double)));
  assert(e_sp_total = (double *)malloc(*nwf * sizeof(double)));
  assert(n_qp_total = (double *)malloc(*nwf * sizeof(double)));
  assert(_e_qp_total = (double *)malloc(*nwf * sizeof(double)));
  assert(_e_sp_total = (double *)malloc(*nwf * sizeof(double)));
  assert(_n_qp_total = (double *)malloc(*nwf * sizeof(double)));

  for(i=0; i<*nwf; i++)
    {
      _e_qp_total[i] = 0.0;
      _e_sp_total[i] = 0.0;
      _n_qp_total[i] = 0.0;
      e_qp_total[i] = 0.0;
      e_sp_total[i] = 0.0;
      n_qp_total[i] = 0.0;
    }
  /************/
  
  file_wf = malloc( 140 * sizeof( char ) ) ;
  file_pot = malloc( 140 * sizeof( char ) ) ;
  sprintf( file_wf , "%s/wf%s.cwr" , dir_in , iso_label ) ;
  if ( gr_ip == 0 ) printf(" isospin=%d gr_np = %d \n root_p = %d root_n = %d \n" , isospin , gr_np , root_p ,root_n ) ;
  
  /* calculate now the number of wavefunctions per process */
  if ( gr_ip == 0 ) /* exclude root */
    nwfip = 0 ;
  else
    {
      nwfip = *nwf / np_c ;
      if ( gr_ip - 1 < ( *nwf ) - nwfip * np_c ) nwfip++ ;
      assert( qp_energy = malloc( nwfip * sizeof(double) )) ;
      assert( e_qp = malloc( nwfip * sizeof( double ) ) )  ;
      assert( e_sp = malloc( nwfip * sizeof( double ) ) )  ;
      assert( n_qp = malloc( nwfip * sizeof( double ) ) ) ;
      assert( lz_qp = malloc( nwfip * sizeof( double ) ) ) ;
    }
  MPI_Barrier( commw );
  MPI_Allreduce( &nwfip , &nwf_tot , 1 , MPI_INT , MPI_SUM , gr_comm ) ;
  if ( nwf_tot != * nwf )
    {
      printf( " The wave functions are not distributed correctly \n %d %d \nProgram terminated \n\n" , nwf_tot , * nwf ) ;
      ierr = -1;
      MPI_Abort(commw, ierr) ;
      return( EXIT_FAILURE ) ;
    } 
  wavef = allocate_phys_mem_wf( nwfip , nxyz ) ;

  /* Shi Jin*/
  int *wavef_index;  // store the index of the wavefunction (ascending order of qp energy)
  assert(wavef_index = (int *) malloc(nwfip * sizeof(int)));

  double *norms_tr, *rnorms_tr;  // store the inner products of the doublets
  assert(norms_tr = (double *) malloc(*nwf * sizeof(double)));
  assert(rnorms_tr = (double *) malloc(*nwf * sizeof(double)));

  int *labels_tr;
  assert(labels_tr = (int *) malloc(nwfip/2 * sizeof(int)));
  /*********/
  
  kenny_copy( nx , ny , nz , nwfip, dxyz, dx );  
  kenny_copy_( nx , ny , nz , nwfip );

  shi_copy(xpow, e2);
  
  int gpu_nwfip , gpu_nxyz ;
  chk_constants( &gpu_nwfip , &gpu_nxyz ) ;

  MPI_Status mpi_st ; 
  int itag = 11 ;
  int ioff = 0 ; 
  double * iobf ; 
  int fd_n_qp, fd_e_qp, fd_lz_qp;
  mode_t fd_mode = S_IRUSR | S_IWUSR ; 
  int * wf_tbl;

  //should be done only one time somewhere near the top of ctdslda
  assert( wf_tbl = malloc( np_c * sizeof( int ) ) ) ; 
  if ( gr_ip == 0 ) 
    {
      char fn[120];
      assert( iobf = malloc( (*nwf) * sizeof( double ) ) ) ; 
      sprintf( fn, "occ%s.bn" , iso_label );
      if ( ( fd_n_qp = open( fn , O_RDWR | O_CREAT , fd_mode ) ) == -1 ) //need to set char * fn here to whatever you want
	printf( "[%d]error: cannot open() FILE %s for WRITE\n" , gr_ip , fn ) ; 
      sprintf( fn, "eqp%s.bn" , iso_label );
      if ( ( fd_e_qp = open( fn , O_RDWR | O_CREAT , fd_mode ) ) == -1 ) //need to set char * fn here to whatever you want
	printf( "[%d]error: cannot open() FILE %s for WRITE\n" , gr_ip , fn ) ; 
      sprintf( fn, "lz%s.bn" , iso_label );
      if ( ( fd_lz_qp = open( fn , O_RDWR | O_CREAT , fd_mode ) ) == -1 ) //need to set char * fn here to whatever you want
	printf( "[%d]error: cannot open() FILE %s for WRITE\n" , gr_ip , fn ) ; 


      /* io buffer and local formation of wave function table */
      for ( i = 0 ; i < np_c ; i++ )
	{
	  int j  = * nwf / np_c ;
	  if ( i < ( (*nwf) % np_c ) ) j++ ;
	  wf_tbl[ i ] = j ;
	}
    }
  MPI_Bcast( wf_tbl , np_c , MPI_INT , 0 , gr_comm );
  if( gr_ip != 0 && wf_tbl[gr_ip-1]!=nwfip )
    printf("nwfip for gr_ip=%d does not match: %d vs. %d \n" , gr_ip,nwfip,wf_tbl[gr_ip-1] );



  // GPU
  /* set up the cufft configuration */
  int batch = 8 ;
  cufftHandle d_plan_b , d_plan_r ; /* bath and remainder plans */
  /* create 3D FFT plans on the gpu for batch and remainder offunctions to evolve */
  int ncufft[rank] = {nx,ny,nz} ;
  int ncufft3[rank] = {nx3,ny3,nz3} ;
  int chunks = (4*nwfip) / batch ;
  int chunk_diff = 4*nwfip - chunks * batch  ; //int chunk_diff  = (4*nwfip) % batch ;
  int istride, ostride;
  istride = ostride = 1; /* array is contiguous in memory */
  int *inembed = ncufft, *onembed = ncufft;
  if ( cufftPlanMany( &d_plan_b , rank , ncufft , inembed , istride , nxyz , onembed , ostride , nxyz , CUFFT_Z2Z, batch ) != CUFFT_SUCCESS )
    { fprintf(stderr, "error: batch plan creation failed (CUFFT)\n"); /* need a fallout plan here ... */ }
  if ( chunk_diff != 0 ) /* in case we need it */
    if ( cufftPlanMany( &d_plan_r , rank , ncufft , inembed , istride , nxyz , onembed , ostride , nxyz , CUFFT_Z2Z, chunk_diff ) != CUFFT_SUCCESS )
      { fprintf(stderr, "error: remainder plan creation failed (CUFFT)\n"); /* need a fallout plan here ... */ }
  cufftDoubleComplex *d_fft3 , * d_wfft;
  icudaError = cudaMalloc( (void **)&d_fft3, sizeof(cufftDoubleComplex) * nxyz * batch ) ;
  if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_fft3)\n");
  icudaError = cudaMalloc( (void **)&d_wfft, sizeof(cufftDoubleComplex) * nxyz * batch ) ; /* workarray */
  if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_wfft)\n");

  cufftHandle d_plan_d ; /* Handle for laplacean of density. */
  if ( cufftPlan3d(&d_plan_d , nx , ny , nz , CUFFT_Z2Z ) != CUFFT_SUCCESS )
    { fprintf(stderr, "error: laplacean of density plan creation failed (CUFFT)\n"); /* need a fallout plan here ... */ }	
 

  cufftHandle d_plan_ev ;  // this plan will be used in evolution, for the symmetrized application of the Hamiltonian, maybe later expanded to some calculations of densities
  cufftDoubleComplex *d_fft3_ev;
  int batch_ev=8;
  icudaError = cudaMalloc((void **)&d_fft3_ev, sizeof(cufftDoubleComplex) * nxyz * batch_ev ) ;
  if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_fft3_ev)\n");
  if ( cufftPlanMany( &d_plan_ev , rank , ncufft , inembed , istride , nxyz , onembed , ostride , nxyz , CUFFT_Z2Z, batch_ev ) != CUFFT_SUCCESS )
    { fprintf(stderr, "error: batch plan creation failed (CUFFT)\n"); /* need a fallout plan here ... */ }

  // cufft plan for Coulomb interaction
  cufftHandle d_plan_c ;
  cufftDoubleComplex *d_fft3_c ;
  icudaError = cudaMalloc((void **)&d_fft3_c, sizeof(cufftDoubleComplex) * nxyz3 ) ;
  if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_fft3_c)\n");
  if ( cufftPlan3d(&d_plan_c , nx3 , ny3 , nz3 , CUFFT_Z2Z ) != CUFFT_SUCCESS )
    { fprintf(stderr, "error: coulomb plan creation failed (CUFFT)\n"); /* need a fallout plan here ... */ }	
  cufftDoubleReal *d_fc ; //re coulomb 
  int * d_map_sl , * d_map_ls ;
  icudaError = cudaMalloc((void **)&d_fc, sizeof(cufftDoubleReal) * nxyz3 ) ;
  if (icudaError != cudaSuccess) printf("error in cudaMalloc (d_fc)\n");
  icudaError = cudaMalloc((void **)&d_map_ls, sizeof(int) * nxyz3 ) ; // large to small 
  if (icudaError != cudaSuccess) printf("error in cudaMalloc (d_map_ls)\n");
  icudaError = cudaMalloc((void **)&d_map_sl, sizeof(int) * nxyz ) ; // small to large 
  if (icudaError != cudaSuccess) printf("error in cudaMalloc (d_map_sl)\n");

  cufftDoubleComplex * d_wavf, * d_wavf_td, *d_wavf_p, *d_wavf_c, *d_wavf_m ;

  cufftDoubleComplex * d_wavf_psi, * d_wavf_hpsi, * d_wavf_mxp;

  cufftDoubleReal* d_coeffs;

  cufftDoubleReal* d_ft, * d_wavf_td_buffer;

  double * ft;


  // the coefficients in the series expansion: (-idt/hbar)^m/m!
  double * coeffs;

  assert(coeffs = (double *) malloc( mxp * sizeof(double)));

  double coeff = 1.0;

  coeffs[0] = coeff;

  int m;
  
  for(m=1; m<mxp;m++)
    {
      coeff*=(dt_step)/ (double) m;
      coeffs[m] = coeff;
    }

  /////////

  int j, iwf, ixyz ;
  double complex * copy_buf ;
  if(gr_ip!=0)
    assert(copy_buf = (double complex *) malloc(4*nwfip*nxyz*sizeof(double complex))) ;

  if (gr_ip != 0)
    {
      /* gpu wavefunction opertions on the cpu */
      icudaError = cudaMalloc( (void **)&d_wavf   , 8*nwfip*nxyz*sizeof( cufftDoubleComplex ) ) ;
      if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_wavf)\n");      
      
      icudaError = cudaMalloc( (void **)&d_wavf_td, 16*nwfip*nxyz*sizeof( cufftDoubleComplex ) );	
      
      if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_wavf_td)\n");
      
      icudaError = cudaMalloc( (void **)&d_wavf_mxp , 4*mxp*nwfip*nxyz*sizeof( cufftDoubleComplex ) );	
      if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_wavf_mxp)\n");

      icudaError = cudaMalloc( (void **)&d_wavf_psi , 4*nwfip*nxyz*sizeof( cufftDoubleComplex ) );	
      if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_wavf_psi)\n");

      icudaError = cudaMalloc( (void **)&d_wavf_hpsi , 4*nwfip*nxyz*sizeof( cufftDoubleComplex ) );	
      if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_wavf_hpsi)\n");

      icudaError = cudaMalloc( (void **)&d_coeffs , mxp*sizeof( cufftDoubleReal ) );	
      if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_coeffs)\n");

      icudaError = cudaMalloc( (void **)&d_ft   , nwfip*sizeof( cufftDoubleReal ) ) ;
      if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_ft)\n");

      assert(ft = malloc(nwfip*sizeof(double)));      
      
    }

  if(gr_ip !=0)
    {
      cudaMemcpy(d_coeffs, coeffs, mxp*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
    }
  
  free(coeffs);
	
  //CPU
#ifndef RSTRT
  /* read in the initial wave functions */
  knxyz = 4 * nxyz ;
  if(irun != 2){
     read_wf_MPI( wavef->wavf[ 0 ] , nxyz , *nwf , gr_ip , gr_np , gr_comm , file_wf , dxyz , wavef_index, wf_tbl);
     if( ip == 0)
	printf("Read successful MPI \n");
  }
  else
    generate_plane_wave_wfs( wavef->wavf[ 0 ] , nxyz , *nwf , gr_ip , gr_np , gr_comm , file_wf , dxyz , wavef_index, &latt) ;
  
  MPI_Barrier( gr_comm ) ;
	
  if ( gr_ip == 0 ) 
    printf("%d wave functions successfully read from %s\n" , * nwf , file_wf ) ;
  else
    initialize_wfs( wavef , nwfip , nxyz ) ;

#endif


  
#ifdef RSTRT
  knxyz = 4 * nxyz ;
  if(irun != 2){
  read_wf_MPI( wavef->wavf[ 0 ] , nxyz , *nwf , gr_ip , gr_np , gr_comm , file_wf , dxyz , wavef_index, wf_tbl);
  }
  else
    generate_plane_wave_wfs( wavef->wavf[ 0 ] , nxyz , *nwf , gr_ip , gr_np , gr_comm , file_wf , dxyz , wavef_index, &latt) ;
  
  //if( gr_ip == 0 ) printf( " wfs read \n"  ) ; 
  MPI_Barrier( commw ) ;
  	
  if ( gr_ip == 0 ) 
    printf("RSTRT: %d wave functions successfully read from %s\n" , * nwf , file_wf ) ;
  else
    initialize_wfs( wavef , nwfip , nxyz ) ;

  
  if( ip == 0 ){
    fd_rs_info = fopen( "restart.info" , "rb" ) ;
    fread( &time_start , sizeof( int ) , 1 , fd_rs_info ); 
    fclose( fd_rs_info ) ;
    printf( "restart time:%d \n" , time_start ) ;
  }
  
  
  MPI_Barrier( commw ) ;
  MPI_Bcast( &time_start , 1 , MPI_INT , 0 , commw ) ;
#endif
 
  //GPU - copy wavefunctions from CPU to GPU 
  // wavf
  if (gr_ip != 0)
    {
      for ( i = 0 ; i < 2 ; i++ )
	{
	  j = 0 ;
	  for ( iwf = 0 ; iwf < nwfip ; iwf++ )
	    for ( ixyz = 0 ; ixyz < 4*nxyz ; ixyz++ )
	      {
		copy_buf[ j ] = wavef->wavf[i][iwf][ixyz] ;
		j++ ;
	      }
	  /* copy ~ (to , from, size_t , direction) */
	  cudaMemcpy(d_wavf + i*nwfip*4*nxyz , copy_buf , 4*nwfip*nxyz*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice) ; 
	}

      for ( i = 0 ; i < 4 ; i++ )
	{
	  j = 0 ;
	  for ( iwf = 0 ; iwf < nwfip ; iwf++ )
	    for ( ixyz = 0 ; ixyz < 4*nxyz ; ixyz++ )
	      {
		copy_buf[ j ] = wavef->wavf_t_der[i][iwf][ixyz] ;
		j++ ;
	      }
	  /* copy ~ (to , from, size_t , direction) */
	  cudaMemcpy(d_wavf_td + i*nwfip*4*nxyz , copy_buf , 4*nwfip*nxyz*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice) ; 
	}
    }

  // GPU --densities 
  cufftDoubleReal * d_densities ; 
  icudaError = cudaMalloc( (void **)&d_densities, 28*nxyz*sizeof( cufftDoubleReal ) ) ; 
  if (icudaError != cudaSuccess) printf("error in cudaMalloc (d_densities)\n");
  /* setup the copy_to and copy_from buffers for density computations */
  cufftDoubleReal * copy_to ;  /* the gpu will copy from -consider write-combining page-locked memory here */
  cufftDoubleReal * copy_from ;

  
  
  
  cudaHostAlloc( (void **)&copy_to , 28 * nxyz * sizeof(double) , cudaHostAllocDefault );
  cudaHostAlloc( (void **)&copy_from , 28 * nxyz * sizeof(double) , cudaHostAllocDefault );

  // divergence of current 
  double * divj;
  assert(divj = (double *) malloc(2 * nxyz*sizeof(double)));
  ////////
  
  cufftDoubleReal * norms_ortho;
  cudaHostAlloc( (void **)&norms_ortho , nwfip*(nwfip-1) * sizeof(double) , cudaHostAllocDefault );
  
  cufftDoubleReal * copy_from_new;
  assert(copy_from_new = malloc( 28*nxyz*sizeof(cufftDoubleReal)));
  
  cufftDoubleReal *d_rcm;
  double rcm[3];
  icudaError = cudaMalloc( (void **)&d_rcm, 3*sizeof( cufftDoubleReal ) ) ; 
  if (icudaError != cudaSuccess) printf("error in cudaMalloc (d_rcm)\n");
  // initialize the c.m. coordinates
  for(i=0;i<3;i++) rcm[i] = 0.;
  cudaMemcpy( (void *)d_rcm , (const void *)rcm , 3*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice ) ;
  

#ifdef LZCALC

  int ifd_jz;
  char file_dens_jz[120];

  int ifd_jz2;
  char file_dens_jz2[120];

  // buffers for jz densities
  cufftDoubleReal * d_jz_dens, * dens_jz ;
  assert(dens_jz = malloc( nxyz * sizeof( cufftDoubleReal ) ) );

  icudaError = cudaMalloc( (void **)&d_jz_dens, nxyz*sizeof( cufftDoubleReal ) ) ; 
  if (icudaError != cudaSuccess) printf("error in cudaMalloc (d_jz_dens)\n");

  // buffers for jz2 densities
  cufftDoubleReal * d_jz2_dens, * dens_jz2 ;
  assert(dens_jz2 = malloc( nxyz * sizeof( cufftDoubleReal ) ) );

  icudaError = cudaMalloc( (void **)&d_jz2_dens, nxyz*sizeof( cufftDoubleReal ) ) ; 
  if (icudaError != cudaSuccess) printf("error in cudaMalloc (d_jz2_dens)\n");

#endif

  cufftDoubleReal * copy_to_,*copy_from_;
  cudaHostAlloc( (void **)&copy_to_ , nxyz * sizeof(double) , cudaHostAllocDefault );
  cudaHostAlloc( (void **)&copy_from_ , nxyz * sizeof(double) , cudaHostAllocDefault );

  
  // CPU
  /* allocate memory for densities */
  if ( isospin == 1 ) {
    dens = &dens_p ;
    
    densG= &densG_p;
  }
  else{
    dens = &dens_n ;
    
    densG= &densG_n;
  }

  if( gr_ip == 0 ) /* only on root processes one needs the densities */

    {
      allocate_phys_mem_dens( &dens_p , nxyz ) ;
      allocate_phys_mem_dens( &dens_n , nxyz ) ;
    }                                                                                                                               

  // filter for left and right fragments
  assert(frag.thetaL= (double*) malloc(nxyz*sizeof(double)));
  assert(frag.thetaR= (double*) malloc(nxyz*sizeof(double)));

  for(i=0;i<nxyz;i++){
    if( latt.za[i] > 0.){
      frag.thetaR[i]=1.;
      frag.thetaL[i]=0.;
    }
    if( latt.za[i] < 0.){
      frag.thetaR[i]=0.;
      frag.thetaL[i]=1.;
    }
    if(latt.za[i] == 0.){
      frag.thetaR[i]=.5;
      frag.thetaL[i]=.5;
    }
  }
  
  // density buffers for fragments
  assert(frag.densf_p = (double*) malloc(14*nxyz*sizeof(double)));
  assert(frag.densf_n = (double*) malloc(14*nxyz*sizeof(double)));
  /******* ********************************/
  


  /******      prepare the random kick potentials and time of moments where the peaks of kicks are located               ******/

  // Prepare the random potentials and time of kicks ....
  b_it(&itgtod,&it_clk);

  /*******************************************/
  dens_p.rho = copy_from ;
  dens_p.tau = copy_from + nxyz ;
  dens_p.sx = copy_from + 2 * nxyz ;
  dens_p.sy = copy_from + 3 * nxyz ;
  dens_p.sz = copy_from + 4 * nxyz ;
  dens_p.divjj = copy_from + 5 * nxyz ;
  dens_p.jx = copy_from + 6 * nxyz ;
  dens_p.jy = copy_from + 7 * nxyz ;
  dens_p.jz = copy_from + 8 * nxyz ;
  dens_p.cjx = copy_from + 9 * nxyz ;
  dens_p.cjy = copy_from + 10 * nxyz ;
  dens_p.cjz = copy_from + 11 * nxyz ;
  dens_p.nu = (double complex *) ( copy_from + 12 * nxyz ) ;

  dens_n.rho = copy_from + 14 * nxyz;
  dens_n.tau = copy_from + 15 * nxyz ;
  dens_n.sx = copy_from + 16 * nxyz ;
  dens_n.sy = copy_from + 17 * nxyz ;
  dens_n.sz = copy_from + 18 * nxyz ;
  dens_n.divjj = copy_from + 19 * nxyz ;
  dens_n.jx = copy_from + 20 * nxyz ;
  dens_n.jy = copy_from + 21 * nxyz ;
  dens_n.jz = copy_from + 22 * nxyz ;
  dens_n.cjx = copy_from + 23 * nxyz ;
  dens_n.cjy = copy_from + 24 * nxyz ;
  dens_n.cjz = copy_from + 25 * nxyz ;
  dens_n.nu = (double complex *)( copy_from + 26 * nxyz ) ;

  // CPU --potentials
  /* finally, allocate potentials on all nodes allocate_phys_mem_pots( &pots , nxyz ) ; */
  assert( pots.u_re = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.u_im = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.mass_eff = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.wx = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.wy = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.wz = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.u1_x = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.u1_y = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.u1_z = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.ugrad_x = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.ugrad_y = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.ugrad_z = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.mgrad_x = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.mgrad_y = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.mgrad_z = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.w_abs = malloc( nxyz * sizeof( double ) ) ) ;
  assert( pots.delta = malloc( nxyz * sizeof( double complex ) ) ) ;
  assert( pots.rv_cm = malloc( 6 * sizeof( double ) ) ) ;
  absorbing_potential( dens_p.lx , dens_p.ly , dens_p.lz , &latt , pots.w_abs , nxyz );
  
  double complex * delta_gpu ;  // will hold the pairing 
  assert( delta_gpu = malloc(nxyz * sizeof(double complex))) ;
	
  // GPU
  // memory for potentials
  cufftDoubleReal * d_potentials , *d_cc_pair_qf, * d_potentials_new ;
  
  icudaError = cudaMalloc( (void **)&d_potentials, ( 16 * nxyz + 6 ) * sizeof(cufftDoubleReal) ) ;
  //icudaError = cudaMalloc( (void **)&d_potentials_new, ( 16 * nxyz + 3 ) * sizeof(cufftDoubleReal) ) ;
  
  icudaError = cudaMalloc( (void **)&d_cc_pair_qf, nxyz * sizeof(cufftDoubleReal) ) ;
  if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_potentials)\n");

  // gradients (0,1,2) and laplaceans (3) of wfs 
  cufftDoubleComplex * d_grad , * d_lapl;  
  if ( gr_ip != 0 )
    {
      icudaError = cudaMalloc( (void **)&d_grad, 12 * nwfip * nxyz * sizeof(cufftDoubleComplex) ) ;
      if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_grad)\n");

      icudaError = cudaMalloc( (void **)&d_lapl, 4 * nwfip * nxyz * sizeof(cufftDoubleComplex) ) ;
      if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_lapl)\n");
      
      
    }

  if(ip==0) printf("Quadrupole constraint: %f \n" , cc_qzz[0] );

  cufftDoubleReal * d_constr ;
  icudaError = cudaMalloc( (void **)&d_constr, 4 * sizeof(cufftDoubleReal) ) ;
  cudaMemcpy(d_constr, cc_qzz , 4*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ; 
  free(cc_qzz);

  make_qzz_gpu( d_potentials+15*nxyz , d_constr , d_xyz , nxyz , iext , c0, (cufftDoubleReal) rneck, (cufftDoubleReal) wneck, (cufftDoubleReal) z0 );

  cudaFree(d_constr);

  //CPU
  if( isospin == 1 )
    pots.amu = &amu_p ;
  else
    pots.amu = &amu_n ;
  dens_func_params( iforce , ihfb , isospin , &cc_edf , ip, icub,alpha_pairing) ;

  if(ggp<1e9){
    cc_edf.gg_p=ggp;
    if(isospin==1) cc_edf.gg=ggp;
  }
  if(ggn<1e9){
    cc_edf.gg_n=ggn;
    if(isospin==-1) cc_edf.gg=ggn;
  }

  if(ip == 0)
    fprintf( stdout, " ** Pairing parameters ** \n proton strength = %f neutron strength = %f \n" , cc_edf.gg_p, cc_edf.gg_n ) ;

  double gg=cc_edf.gg;

  // GPU
  double epsilon =1.e-12; 
  kenny_copy_two( hbar2m, hbarc, *(pots.amu), e_cut, PI, dt_step , epsilon );
  copy_nuc_params_to_gpu( &cc_edf ) ;
  int isoff = 7 * nxyz * (1-isospin) ;
  int isoff1 = nxyz * (1-isospin)/2 ;
  cufftDoubleReal t0,sig;
  t0=(cufftDoubleReal)(1000.);
  sig=(cufftDoubleReal)(300.);

  
  cufftDoubleReal T0=(cufftDoubleReal)(500.);
  cufftDoubleReal T1=(cufftDoubleReal)(500.);
  cufftDoubleReal TT=(cufftDoubleReal)(100.);
  cufftDoubleReal Ti=(cufftDoubleReal) (10.);
  cufftDoubleReal t0q=(cufftDoubleReal)(20.);
  cufftDoubleReal t1q=(cufftDoubleReal)(120.);

  cufftDoubleReal Ctime,t;
  cufftDoubleReal Ctime_qzz=(cufftDoubleReal)(1.);

  // time factor of random kicks

  tcur=time_start*dt_step;	
  t=(cufftDoubleReal)tcur;
  Ctime=non_sym_f_cpu(t,Ti,T0,T1,TT,c0);

  if(iext == 70) // for collision
    {
      Ctime = non_sym_f_cpu_collision_time(t,Ti,T0,T1,TT, c0, c1);
    }
  Ctime_qzz=(cufftDoubleReal)(1.)-theta(t,t0q,t1q);

  pots.ctime=(double)Ctime;

  compute_densities_gpu( nxyz , nwfip , d_wavf , d_grad, d_lapl, d_wfft , d_fft3 , d_kxyz , d_plan_b , d_plan_r, d_plan_d , batch , d_densities , d_avf, commw , copy_to , copy_from , isoff , Ctime, divj );

#ifdef EXTBOOST

#ifndef RSTRT

  double complex * boost_field;

  assert(boost_field = (double complex *) malloc(nxyz*sizeof(double complex)));

  for(i = 0; i<nxyz; i++) boost_field[ i ] = 1.0 + 0.0*I; // default value

  cufftDoubleComplex * d_boost_field;
  
  icudaError = cudaMalloc( (void **)&d_boost_field , nxyz*sizeof( cufftDoubleComplex ) ) ;
  if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_boost_field)\n");  
  
    
  if(irun == 0) // quadrupole boost
    {
      if(ip == 0)
	printf("external quadrupole boost is performed on wfs, boost strength: alpha = %f\n", alpha);
      if(op_read_flag<0)
	make_boost_quadrupole(boost_field, nxyz, &latt, alpha);
      else
	read_boost_operator(boost_field,file_extern_op,commw,ip,alpha,nxyz);
    }

  if(irun == 1)  // two nuclei collision
    {
      // distance between fragments.
      distance = get_distance( &dens_p, &dens_n, &frag, nxyz, dxyz, &latt, rcm_L, rcm_R, &A_L, &Z_L, &A_R,&Z_R);

      // calculates the coloumb energy between the two fragments.
      ec = coul_frag( dens_p.rho , latt.xa , latt.ya , latt.za , nxyz , dxyz, 0. );
      
      if(ip == 0)
	{
	  printf("N_L = %.1f, N_R = %.1f, Z_L = %.1f, Z_R = %.1f\n", A_L-Z_L, A_R-Z_R, Z_L, Z_R);

          printf("xmc_L = %f, ycm_L = %f, zcm_L = %f, xcm_R = %f, ycm_R = %f, zcm_R = %f\n",rcm_L[0],rcm_L[1],rcm_L[2],rcm_R[0],rcm_R[1],rcm_R[2]);

          printf(" Colomb energy between two fragments: %f\n", ec);
	  
	  printf("c.m. energy (including the initial inter Coulomb energy): %.2f\n", ecm);
	}

      if(iext == 0)  // boost two nuclei directly, no external field is added
	{

	  if (ip == 0) printf("Immediate boost is performed on wavefunctions. \n");
	  
	  if ((ierr = make_boost_twobody(boost_field, nxyz, &latt, ecm, A_L, A_R, Z_L, Z_R, rcm_L, rcm_R, ip, b, ec)) !=0)
	  {
	    printf("error in make_boost_twobody! exiting ... \n");
	    MPI_Abort(commw, ierr);
	  }
	}

      else if (iext == 70)
	{
	  if(ip == 0) printf("External field is added to accelerate the nuclei. \n");
	}

      else
	{
	  if(ip == 0) printf("Not Implemented Error: illegal value for iext: %d", iext);

	  MPI_Abort(commw, -1);
	  

	  
	}
      
    }


  cudaMemcpy(d_boost_field, boost_field, nxyz*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice) ;
  
  if(gr_ip !=0)
    do_boost_wf(d_boost_field, d_wavf, 8*nxyz*nwfip);

  cudaFree(d_boost_field);

  

#endif  
  
#endif


  for(i=0;i<nwfip;i++)  ft[i] = 1.0 ; // T = 0 case
      
  cudaMemcpy(d_ft, ft, nwfip*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice); 
 
  compute_densities_gpu( nxyz , nwfip , d_wavf , d_grad, d_lapl, d_wfft , d_fft3 , d_kxyz , d_plan_b , d_plan_r, d_plan_d , batch , d_densities , d_avf, commw , copy_to , copy_from , isoff , Ctime, divj );
 
  assert( pots.v_ext = malloc( nxyz * sizeof( double ) ) ) ;

  external_pot( iext , nxyz , 20 , 2. , pots.v_ext , pots.delta_ext , hbar2m , &latt , z0);



  if(irun == 1)
    {
      double phase = 0.0;

      get_phase_imprint_potential(pots.v_ext, nxyz, &latt, T0+T1, phase);

      if (ip == 0)
	printf("The phase difference between two nuclei is %.6f\n", phase);
    }


  if( iext == 21 || iext == 22 || iext == 23 ){

    double str_dip;

    if( isospin == 1 )
      str_dip=-12./20.;
    else
      str_dip=8./20.;
    
    for(i=0;i<nxyz;++i)
      pots.v_ext[i]*=str_dip;

  }
  
  cudaMemcpy(d_potentials+14*nxyz, pots.v_ext , nxyz*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ;

  assert( buff = malloc( nxyz * sizeof( double ) ) ) ;
  if ( gr_ip == 0 ) 
    {
      if ( ip == root_p )
	{
	  sprintf( file_res , "results_td.dat" ) ;
#ifdef RSTRT
	  fd = fopen( file_res , "a+" ) ;
#else
	  fd = fopen( file_res , "w" ) ;
          fprintf(fd,"Time E_tot Z N xcm ycm zcm xcm_p ycm_p zcm_p xcm_n ycm_n zcm_n Beta E_flow_tot egs Q20 Q30 Q40 Pair_gap_p Pair_gap_n E_ext E_cm \n");
#endif
	}
      
      get_u_re( &dens_p , &dens_n , &pots , &cc_edf , hbar2m , nxyz , isospin , &latt , &fftrans , dxyz ) ;
      
      update_potentials( isospin , &pots , &dens_p , &dens_n , dens , &cc_edf , e_cut , nxyz , hbar2m , &latt , &fftrans , dxyz , buff ,icub) ;
    }

  // GPU
  cudaMemcpy(d_fc, fftrans.fc , nxyz3*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ; 
  cudaMemcpy(d_map_ls, fftrans.i_l2s , nxyz3*sizeof(int),cudaMemcpyHostToDevice) ; 
  cudaMemcpy(d_map_sl, fftrans.i_s2l , nxyz*sizeof(int),cudaMemcpyHostToDevice) ; 

  // do it 
  t=(cufftDoubleReal)tcur;
  Ctime=non_sym_f_cpu(t,Ti,T0,T1,TT,(cufftDoubleReal)0.);

  if(iext == 70) // for collision
    {
      Ctime=non_sym_f_cpu_collision_time(t,Ti,T0,T1,TT, c0, c1);
    }
  Ctime_qzz=(cufftDoubleReal)(1.)-theta(t,t0q,t1q);
  pots.ctime=(double)Ctime;
 
  get_u_re_gpu(d_densities,d_potentials,nxyz,isospin,d_kxyz,d_fft3,d_plan_b,d_fft3_c,d_plan_c,d_fc,d_map_ls,d_map_sl,d_avf,Ctime,Ctime_qzz,nxyz3, gr_comm);
  
#ifdef RSTRT
  // GPU
 
  compute_densities_gpu( nxyz , nwfip , d_wavf + ( ( time_start + 1 ) %2 ) * nwfip * 4 * nxyz , d_grad, d_lapl, d_wfft , d_fft3 , d_kxyz , d_plan_b , d_plan_r,d_plan_d , batch , d_densities, d_avf, commw , copy_to , copy_from , isoff , Ctime , divj) ;

  // GPU
  update_potentials_gpu(d_potentials,nxyz,d_densities,d_fc,d_map_ls,d_map_sl,d_plan_c,d_fft3_c,d_plan_b,d_fft3,d_kxyz,isospin,d_avf,Ctime,Ctime_qzz,gr_comm,d_wavf + ( ( time_start + 1 ) %2 ) * nwfip * 4 * nxyz,batch,d_cc_pair_qf,copy_from_,copy_to_,nwfip,gr_ip,nxyz3,icub);

#else

  update_potentials_gpu(d_potentials,nxyz,d_densities,d_fc,d_map_ls,d_map_sl,d_plan_c,d_fft3_c,d_plan_b,d_fft3,d_kxyz,isospin,d_avf,Ctime,Ctime_qzz,gr_comm,d_wavf,batch,d_cc_pair_qf,copy_from_,copy_to_,nwfip,gr_ip,nxyz3,icub); 

#endif
   
  // CPU
  if ( gr_ip == 0 ) 
    {
#ifdef RSTRT
      if ( ( time_start ) % iopen != 0 )
	{
	  sprintf( file_dens_all , "dens_all%s.dat.%d" , iso_label , ( time_start / iopen ) * iopen ) ;
	  ifd_all = f_copn( file_dens_all ) ;
#ifdef LZCALC
	  if( ip == 0 ){
	    sprintf( file_dens_jz , "jz.dat.%d", ( time_start / iopen ) * iopen );
	    ifd_jz = f_copn( file_dens_jz );

	    sprintf( file_dens_jz2 , "jz2.dat.%d", ( time_start / iopen ) * iopen );
	    ifd_jz2 = f_copn( file_dens_jz2 );
	  }
#endif

	}
      printf("restart_energy: \n" );
      
      cudaMemcpy( delta_gpu , (cufftDoubleComplex *) ( d_potentials + 12 * nxyz ) , nxyz*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost) ; 
      e_gs = system_energy( &cc_edf , dens , &dens_p , &dens_n , isospin , nxyz , delta_gpu , chi, ip , root_p , root_n , commw , hbar2m , dxyz , &latt , &fftrans , &status , tcur , stdout ) ;

#else
      cudaMemcpy( delta_gpu , (cufftDoubleComplex *) ( d_potentials + 12 * nxyz ) , nxyz*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost) ; 


      e_gs = system_energy( &cc_edf , dens , &dens_p , &dens_n , isospin , nxyz , delta_gpu , chi, ip , root_p , root_n , commw , hbar2m , dxyz , &latt , &fftrans , &status , tcur , stdout ) ;

#endif
    }

  MPI_Bcast( &e_gs , 1 , MPI_DOUBLE , 0 , commw );
  e_tot = e_gs;
 
  MPI_Barrier( commw ) ;
  
  int n_saveocc = -1 ;
  int i_saveocc;

  // initialize the angular velocity
  for(i=0; i<3; i++) omega[i] = 0.;
  cudaMemcpy(d_potentials+16*nxyz+3, omega , 3*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ;
  ///////////////////////
  
  time_start++;

  itime = 0;
  if( ip == 0 ) printf( " ******* BOOTSTRAP THE EVOLUTION USING SERIES EXPANSION ******\n");
  if( ip == 0 ) printf( " ******* ORDER OF EXPANSION: %d ***********\n", mxp-1); 
  
  for ( timeloop = time_start ; timeloop < time_start + time_bootstrap ; timeloop++)
    {
      if (ip==0) printf("TIME[%d] Ctime=%f \n",timeloop, (double)Ctime);
      MPI_Barrier(commw);
      
      nn_time = ( timeloop + 1 ) % 2 ;
      n0_time = timeloop % 2 ;
      n1_time = ( timeloop - 1 + 2 ) % 2 ;
      mm_time = ( timeloop + 1 ) % 4 ;
      m0_time = timeloop % 4 ;
      m1_time = ( timeloop - 1 + 4 ) % 4 ;
      m2_time = ( timeloop - 2 + 4 ) % 4 ;
      m3_time = ( timeloop - 3 + 4 ) % 4 ;


      tcur = timeloop * dt_step ;

      itime++;

#ifdef SAVEOCC
      i_saveocc = 0; 
#else
      i_saveocc = 1;
#endif
      
      
      /** one time step **/
      if(gr_ip != 0)
	{
	  tstep_gpu(nxyz, nwfip, mxp, n0_time, nn_time, d_wavf, d_wavf_psi, d_wavf_hpsi, d_wavf_mxp, d_coeffs, d_grad, d_lapl, d_wfft, d_fft3, d_plan_b, d_plan_r, d_potentials, d_avf, batch, d_kxyz, d_plan_ev, d_fft3_ev, d_xyz, d_rcm);
	}
      else
	{
	  if ( ( timeloop - 1 ) % iopen == 0 ) 
	    {
	      sprintf( file_dens_all , "dens_all%s.dat.%d" , iso_label , timeloop - 1 ) ;
	      f_crm( file_dens_all ) ;
	      ifd_all = f_copn( file_dens_all ) ;
	    }
	  if ( ( timeloop - 1 ) % isave == 0 )
	    {
	      f_cwr( ifd_all , copy_from+isoshift, 8 , 14*nxyz );
	      f_cwr( ifd_all , delta_gpu, 16 , nxyz );
	    }
	  if ( ( timeloop - 1 ) % iopen == iopen - isave ){
	    f_ccls( &ifd_all ) ;
	  }

	  
	  
	}

      // get \tilde{\rho}
      compute_densities_gpu( nxyz , nwfip , d_wavf+nn_time*nwfip*4*nxyz , d_grad, d_lapl, d_wfft , d_fft3 , d_kxyz , d_plan_b , d_plan_r,d_plan_d , batch , d_densities, d_avf, commw , copy_to , copy_from_new , isoff , Ctime, divj ) ;
     
      // mix \rho and \tilde{\rho}
      
      mix_densities(copy_from, copy_from_new, d_densities, 28*nxyz, &err);

      t=(cufftDoubleReal)tcur + dt_step/2.0;  // middle time (t + dt/2)
      Ctime=non_sym_f_cpu(t,Ti,T0,T1,TT,c0);
      if(iext == 70) // for collision
	{
	  
	  Ctime=non_sym_f_cpu_collision_time(t,Ti,T0,T1,TT, c0, c1);
	}
      Ctime_qzz=(cufftDoubleReal)(1.)-theta(t,t0q,t1q);

      pots.ctime=(double)non_sym_f_cpu(t,(cufftDoubleReal) (4.),(cufftDoubleReal)120.,(cufftDoubleReal)(200.),(cufftDoubleReal)(150.),(cufftDoubleReal) 1.);

      // update potentials
      update_potentials_gpu(d_potentials,nxyz,d_densities,d_fc,d_map_ls,d_map_sl,d_plan_c,d_fft3_c,d_plan_b,d_fft3,d_kxyz,isospin,d_avf,Ctime,Ctime_qzz,gr_comm,d_wavf + nn_time * nwfip * 4 * nxyz,batch,d_cc_pair_qf,copy_from_,copy_to_,nwfip,gr_ip,nxyz3,icub);
           
      // second half:
      if(gr_ip != 0)
	{
	  tstep_gpu(nxyz, nwfip, mxp, n0_time, nn_time, d_wavf, d_wavf_psi, d_wavf_hpsi, d_wavf_mxp, d_coeffs, d_grad, d_lapl, d_wfft, d_fft3, d_plan_b, d_plan_r, d_potentials, d_avf, batch, d_kxyz, d_plan_ev, d_fft3_ev,d_xyz,d_rcm);
	}

      // now we have \psi(t+dt) : nn_time
      // get new densities


      compute_densities_gpu( nxyz , nwfip , d_wavf+nn_time*nwfip*4*nxyz , d_grad, d_lapl, d_wfft , d_fft3 , d_kxyz , d_plan_b , d_plan_r,d_plan_d , batch , d_densities, d_avf, commw , copy_to , copy_from , isoff , Ctime , divj) ;

      // update the new potential
      t=(cufftDoubleReal)tcur + dt_step;  // new time (t + dt)
     
      Ctime=non_sym_f_cpu(t,Ti,T0,T1,TT,c0);
      if(iext == 70) // for collision
	Ctime=non_sym_f_cpu_collision_time(t,Ti,T0,T1,TT, c0, c1);
      Ctime_qzz=(cufftDoubleReal)(1.)-theta(t,t0q,t1q);

      pots.ctime=(double)non_sym_f_cpu(t,(cufftDoubleReal) (4.),(cufftDoubleReal)120.,(cufftDoubleReal)(200.),(cufftDoubleReal)(150.),(cufftDoubleReal) 1.);
     
      // update potentials      
      update_potentials_gpu(d_potentials,nxyz,d_densities,d_fc,d_map_ls,d_map_sl,d_plan_c,d_fft3_c,d_plan_b,d_fft3,d_kxyz,isospin,d_avf,Ctime,Ctime_qzz,gr_comm,d_wavf + nn_time * nwfip * 4 * nxyz,batch,d_cc_pair_qf,copy_from_,copy_to_,nwfip,gr_ip,nxyz3,icub);


      if(timeloop % loop_io == 0){
	if ( gr_ip == 0 ) 
	  {
	    if ( isospin == 1 )
	      printf( "time_current = %f time_step = %d w0 = %f \n" , tcur , timeloop , cimag(pots.w0) ) ;
	    cudaMemcpy(delta_gpu , (cufftDoubleComplex *) ( d_potentials + 12 * nxyz ) , nxyz*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost) ; 
	    e_tot = system_energy( &cc_edf , dens , &dens_p , &dens_n , isospin , nxyz , delta_gpu , chi,ip , root_p , root_n , commw , hbar2m , dxyz , &latt , &fftrans , &status , tcur,  fd ) ;

	    if( ip == root_p && timeloop % (loop_cp * 10) == 0 )
	      {
		fclose( fd ) ;
		fd = fopen( file_res , "a+" ) ;		
	      }
	  }
	
	MPI_Bcast( &e_tot , 1 , MPI_DOUBLE , 0 , commw );

	if(isfinite(e_tot)==0){
	  if(ip==0)
	    printf("An error detected, the energy is not finite.\nProgram terminated.\n");
	  MPI_Finalize();
	  return 211;

	}
      }
      
#ifdef SAVEOCC
      if(i_saveocc==0){
	center_dist_pn( dens_p.rho , dens_n.rho , nxyz , &latt , rcm , rcm+1 , rcm+2 ) ;
	cudaMemcpy( (void *)d_rcm , (const void *)rcm , 3*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice ) ;
      }
#endif
      
      // now need to calculate the time derivatives of \psi(t+dt)
      if ( gr_ip != 0 )
	{
          if(timeloop % i_norm == 0)
          {
            norm_switch = 1;
          }
          else
          {
            norm_switch = 0;
          } 
	  //GPU
	  adams_bashforth_dfdt_gpu(nxyz,nwfip,d_wavf+nn_time*nwfip*4*nxyz,d_wavf_td+mm_time*nwfip*4*nxyz,d_grad,d_lapl,d_plan_b,d_plan_r,d_potentials,batch,d_kxyz,i_saveocc,norm_switch,e_qp,e_sp,n_qp,d_plan_ev,d_fft3_ev,d_xyz,d_rcm,lz_qp);
	}

#ifdef LZCALC
      if((timeloop-1) % iopen == 0 && ip == 0 ){
	sprintf( file_dens_jz , "jz.dat.%d", timeloop -1 );
	f_crm( file_dens_jz );
	ifd_jz = f_copn( file_dens_jz );


	sprintf( file_dens_jz2 , "jz2.dat.%d", timeloop -1 );
	f_crm( file_dens_jz2 );
	ifd_jz2 = f_copn( file_dens_jz2 );
      }

      if( (timeloop - 1) % isave == 0 ){
	center_dist_pn( dens_p.rho , dens_n.rho , nxyz , &latt , rcm , rcm+1 , rcm+2 ) ;
	if(  ip == 0 ) printf( "xcm=%f ycm=%f zcm=%f \n" , rcm[0], rcm[1], rcm[2] );
	MPI_Barrier(commw);
	cudaMemcpy( (void *)d_rcm , (const void *)rcm , 3*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice ) ;

	// calculate total angular momentum on z direction: jz
	calc_jz_dens( d_jz_dens , dens_jz , copy_to_ , d_grad , d_wavf+nn_time*nwfip*4*nxyz, d_xyz , d_rcm , nxyz , nwfip , commw );
	
	if( ip == 0 ) f_cwr( ifd_jz , dens_jz , 8 , nxyz ) ;

	MPI_Barrier(commw);
	
	// calculate total angular momentum on z direction: jz^2
	calc_jz2_dens( d_jz2_dens , dens_jz2 , copy_to_ , d_grad , d_wavf+nn_time*nwfip*4*nxyz, d_xyz , d_rcm , nxyz , nwfip , commw );

	if( ip == 0 ) f_cwr( ifd_jz2 , dens_jz2 , 8 , nxyz ) ;
      }
      

      if ( ( timeloop - 1 ) % iopen == iopen - isave && ip == 0 ){
	f_ccls( &ifd_jz ) ;

	f_ccls( &ifd_jz2 ) ;
      }
#endif


      
#ifdef SAVEOCC
      if(i_saveocc==0){
	if (gr_ip != 0 ){

	  for(i=0;i<nwfip;++i)
	    {
	      n_qp[i]*=dxyz;
	    }

	  for(i=0; i< nwfip; i++) _n_qp_total[wavef_index[i]] = n_qp[i];

	  MPI_Send( n_qp , nwfip , MPI_DOUBLE , 0 , itag , gr_comm ) ; //0 in gr_comm.
	  
	}
	if (gr_ip == 0 )
	  {

	    ioff=0;
	    for ( i = 0 ; i < np_c ; i++ )
	      {
		MPI_Recv( iobf + ioff , wf_tbl[ i ] , MPI_DOUBLE , i+1 , itag , gr_comm , &mpi_st ) ; 
		ioff += wf_tbl[ i ] ;
	      }

	    if ( ( i = write( fd_n_qp , ( const void * ) iobf , (*nwf) * sizeof( double ) ) ) != (*nwf) * sizeof( double ) )
	      printf("warning, could not copy the occupation numbers for time step %d.\n",timeloop);

	    for(i=0;i<20;++i){
	      if(isospin==-1) printf(" %12.6f" ,iobf[i]);
	      if(i%10==9)printf("\n");
	    }

	  }

	MPI_Barrier( gr_comm );

	if (gr_ip != 0 )
	  {
	    for(i=0; i< nwfip; i++) _e_qp_total[wavef_index[i]] = e_qp[i];
	    
	    //extra minus sign in spe: <v|h*-mu|v>/<v|v>
	    for(i=0; i< nwfip; i++) _e_sp_total[wavef_index[i]] = -1.0*e_sp[i];
	    
	    MPI_Send( e_qp , nwfip , MPI_DOUBLE , 0 , itag , gr_comm ) ; //0 in gr_comm.

	  }
	if (gr_ip == 0 )
	  {
	    ioff=0;
	    for ( i = 0 ; i < np_c ; i++ )
	      {
		MPI_Recv( iobf + ioff , wf_tbl[ i ] , MPI_DOUBLE , i+1 , itag , gr_comm , &mpi_st ) ; 
		ioff += wf_tbl[ i ] ;
	      }

	    if ( ( i = write( fd_e_qp , ( const void * ) iobf , (*nwf) * sizeof( double ) ) ) != (*nwf)* sizeof( double ) )
	      printf("warning, could not copy the occupation numbers for time step %d.\n",timeloop);
	  }

	MPI_Barrier( gr_comm );
	if (gr_ip != 0 )
	  MPI_Send( lz_qp , nwfip , MPI_DOUBLE , 0 , itag , gr_comm ) ; //0 in gr_comm.
	if (gr_ip == 0 )
	  {
	    ioff=0;
	    for ( i = 0 ; i < np_c ; i++ )
	      {
		MPI_Recv( iobf + ioff , wf_tbl[ i ] , MPI_DOUBLE , i+1 , itag , gr_comm , &mpi_st ) ; 
		ioff += wf_tbl[ i ] ;
	      }
	    
	    if ( ( i = write( fd_lz_qp , ( const void * ) iobf , (*nwf) * sizeof( double ) ) ) != (*nwf)* sizeof( double ) )
	      printf("warning, could not copy the occupation numbers for time step %d.\n",timeloop);
	  }

	// write the E_qp and Occ into txt file for checking

	if (( timeloop - 1 ) % 100 == 0)
	  {

	    MPI_Barrier(gr_comm);

	    MPI_Allreduce(_e_qp_total, e_qp_total, *nwf, MPI_DOUBLE, MPI_SUM, gr_comm);

	    MPI_Allreduce(_e_sp_total, e_sp_total, *nwf, MPI_DOUBLE, MPI_SUM, gr_comm);

	    MPI_Allreduce(_n_qp_total, n_qp_total, *nwf, MPI_DOUBLE, MPI_SUM, gr_comm);
	    
	    if (gr_ip == 0)
	      {
		char fn[120];
			
		sprintf(fn, "E_qp%s.dat.%d", iso_label, ( timeloop - 1 ));
		
		fd_eqp_txt =fopen(fn, "w");
	
		for(i=0;i<*nwf;i++)
		  fprintf(fd_eqp_txt, "%12.8f      %12.8f      %8.6f\n", e_qp_total[i], e_sp_total[i], n_qp_total[i]);
		
		fclose(fd_eqp_txt);
	      }
	  }

	//#endif
      }
#endif
    }

  time_start = timeloop;

  // Copy the newest wavefunction into predictor and corrector

  if(gr_ip != 0){

    // free buffers in bootstrap
    cudaFree(d_wavf_psi);
    cudaFree(d_wavf_hpsi);
    cudaFree(d_wavf_mxp);
    cudaFree(d_coeffs);
      
       
    icudaError = cudaMalloc( (void **)&d_wavf_p , 8*nwfip*nxyz*sizeof( cufftDoubleComplex ) );	
    if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_wavf_p)\n");
    icudaError = cudaMalloc( (void **)&d_wavf_c , 8*nwfip*nxyz*sizeof( cufftDoubleComplex ) );	
    if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_wavf_c)\n");
    
    icudaError = cudaMalloc( (void **)&d_wavf_m , 4*nwfip*nxyz*sizeof( cufftDoubleComplex ) );	
    if(icudaError != cudaSuccess) printf("error in cudaMalloc (d_wavf_m)\n");

    do_copy_cp_gpu(nn_time, n0_time, d_wavf, d_wavf_c, d_wavf_p, nxyz, nwfip);
  }

  MPI_Barrier(commw);
#ifdef BENCHMARK
  //kr -for benchmarking 
  MPI_Barrier(commw);
  double init_tim;
  init_tim=e_t(0); ict=e_t(2);
  
  if ( ip == 0 ) fprintf( stdout, "KRTDSLDA: INIT::(nx,ny,nz)[ %d %d %d ] nwf[ %d ] PEs[ %d ] Wtime[ %f ] Ctime[ %f ]\n",nx,ny,nz,nwf_p+nwf_n,np, init_tim, ict);

  //kr --jan2017, 
  double abm_tim, dens_comp_tim, dens_io_tim, iter_tim_tot, pot_tim, sys_tim;
  abm_tim =0.; dens_comp_tim=0.; dens_io_tim=0.; iter_tim_tot=0.; pot_tim=0.; sys_tim = 0.;
  com_tim = 0.;
  b_t(); //restart a clock around the time loop for the benchmark
 
#endif

  if( ip == 0 ) printf( " ******* START THE TIME EVOLUTION LOOP WITH ABM ******\n");
  
  //////////////////////////////////////

  for ( timeloop = time_start ; timeloop < total_time_steps ; timeloop++ )
    {
      
      MPI_Barrier(commw);

      nn_time = ( timeloop + 1 ) % 2 ;
      n0_time = timeloop % 2 ;
      n1_time = ( timeloop - 1 + 2 ) % 2 ;
      mm_time = ( timeloop + 1 ) % 4 ;
      m0_time = timeloop % 4 ;
      m1_time = ( timeloop - 1 + 4 ) % 4 ;
      m2_time = ( timeloop - 2 + 4 ) % 4 ;
      m3_time = ( timeloop - 3 + 4 ) % 4 ;

            

      tcur = timeloop * dt_step ;

      itime++;
     
      /******  START ABM *******/
      
      // ABM - PM step 
      if ( gr_ip > 0 )
	{
	  //GPU
#ifdef BENCHMARK
	  b_it(&itgtod,&it_clk);
#endif
	  adams_bashforth_pm_gpu(nn_time,n0_time,n1_time,m0_time,m1_time,m2_time,m3_time,nxyz,nwfip,d_wavf,d_wavf_td,d_wavf_p,d_wavf_c,d_wavf_m);

#ifdef BENCHMARK
	  abm_tim+=e_it(0,&itgtod,&it_clk);
#endif
	}
      else
	{
#ifdef BENCHMARK	  
	  //kr
	  b_it(&itgtod,&it_clk);
#endif
	  if ( ( timeloop - 1 ) % iopen == 0 ) 
	    {
	      sprintf( file_dens_all , "dens_all%s.dat.%d" , iso_label , timeloop - 1 ) ;
	      f_crm( file_dens_all ) ;
	      ifd_all = f_copn( file_dens_all ) ;
	    }
	  if ( ( timeloop - 1 ) % isave == 0 )
	    {
	      f_cwr( ifd_all , copy_from+isoshift, 8 , 14*nxyz );
	      f_cwr( ifd_all , delta_gpu, 16 , nxyz );
	    }
	  if ( ( timeloop - 1 ) % iopen == iopen - isave ){
	    f_ccls( &ifd_all ) ;
	  }	  

#ifdef BENCHMARK
	  dens_io_tim+=e_it(0,&itgtod,&it_clk); //this is gr_ip == 0
#endif
	}
      
      t=(cufftDoubleReal)tcur;
      Ctime=non_sym_f_cpu(t,Ti,T0,T1,TT,c0);
      if(iext == 70) // for collision
	Ctime=non_sym_f_cpu_collision_time(t,Ti,T0,T1,TT, c0, c1);
      Ctime_qzz=(cufftDoubleReal)(1.)-theta(t,t0q,t1q);
      
      pots.ctime=(double)non_sym_f_cpu(t,(cufftDoubleReal) (4.),(cufftDoubleReal)120.,(cufftDoubleReal)(200.),(cufftDoubleReal)(150.),(cufftDoubleReal) 1.);

                
      // GPU
#ifdef BENCHMARK
      //kr
      b_it(&itgtod,&it_clk);
#endif
      compute_densities_gpu( nxyz , nwfip , d_wavf_m , d_grad, d_lapl, d_wfft , d_fft3 , d_kxyz , d_plan_b , d_plan_r,d_plan_d , batch , d_densities, d_avf, commw , copy_to , copy_from , isoff , Ctime , divj) ;
      
#ifdef BENCHMARK
      dens_comp_tim += e_it(0,&itgtod,&it_clk);
#endif

     
#ifdef BENCHMARK                  
      b_it(&itgtod,&it_clk);
#endif
      update_potentials_gpu(d_potentials,nxyz,d_densities,d_fc,d_map_ls,d_map_sl,d_plan_c,d_fft3_c,d_plan_b,d_fft3,d_kxyz,isospin,d_avf,Ctime,Ctime_qzz,gr_comm,d_wavf_m,batch,d_cc_pair_qf,copy_from_,copy_to_,nwfip,gr_ip,nxyz3,icub);

#ifdef BENCHMARK                        
      pot_tim += e_it(0,&itgtod,&it_clk);
#endif
     	
      if ( gr_ip != 0 )
	{
	  // GPU
#ifdef BENCHMARK                  	  
	  b_it(&itgtod,&it_clk);
#endif
	  adams_bashforth_cy_gpu(nn_time,n0_time,n1_time,m0_time,m1_time,m2_time,nxyz,nwfip,d_wavf,d_wavf_td,d_wavf_p,d_wavf_c,d_wavf_m,d_grad,d_lapl,d_plan_b,d_plan_r,d_wfft,d_potentials,batch,d_kxyz,d_plan_ev,d_fft3_ev,d_xyz,d_rcm) ;

#ifdef BENCHMARK                  	  
	  abm_tim += e_it(0,&itgtod,&it_clk);
#endif
	}
      // GPU
      //kr .
#ifdef BENCHMARK                        
      b_it(&itgtod,&it_clk);
#endif
      compute_densities_gpu( nxyz , nwfip , d_wavf + nn_time * nwfip * 4 * nxyz , d_grad, d_lapl, d_wfft , d_fft3 , d_kxyz , d_plan_b , d_plan_r , d_plan_d,batch , d_densities, d_avf, commw , copy_to , copy_from , isoff , Ctime , divj) ;
#ifdef BENCHMARK                        
      dens_comp_tim += e_it(0,&itgtod,&it_clk);
#endif
      
#ifdef BENCHMARK                        
      b_it(&itgtod,&it_clk);
#endif
      update_potentials_gpu(d_potentials,nxyz,d_densities,d_fc,d_map_ls,d_map_sl,d_plan_c,d_fft3_c,d_plan_b,d_fft3,d_kxyz,isospin,d_avf,Ctime,Ctime_qzz,gr_comm,d_wavf + nn_time * nwfip * 4 * nxyz,batch,d_cc_pair_qf,copy_from_,copy_to_,nwfip,gr_ip,nxyz3,icub);
#ifdef BENCHMARK                  
      pot_tim += e_it(0,&itgtod,&it_clk);
#endif      

     
      if(timeloop % loop_io == 0){
#ifdef BENCHMARK                        
	b_it(&itgtod,&it_clk);
#endif	          
	distance = get_distance( &dens_p, &dens_n, &frag, nxyz, dxyz, &latt, rcm_L, rcm_R, &A_L, &Z_L, &A_R,&Z_R);
#ifdef BENCHMARK                  
	sys_tim += e_it(0,&itgtod,&it_clk);
#endif      	    
	if (ip==0) printf("TIME[%d] Ctime=%f\n",timeloop,(double)Ctime);
	
	if ( gr_ip == 0 ) 
	  {
	    if ( isospin == 1 ){
	      printf( "time_current = %f time_step = %d w0 = %f distance = %f\n" , tcur , timeloop , cimag(pots.w0) , distance ) ;

#ifdef ROTATION
	      
#endif
	    }
	    
#ifdef BENCHMARK                        
	    b_it(&itgtod,&it_clk);
#endif	    
	    cudaMemcpy(delta_gpu , (cufftDoubleComplex *) ( d_potentials + 12 * nxyz ) , nxyz*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost) ;
	    
#ifdef BENCHMARK                  
	    com_tim += e_it(0,&itgtod,&it_clk);
#endif
	    
#ifdef BENCHMARK                        
	    b_it(&itgtod,&it_clk);
#endif	          
	    e_tot = system_energy( &cc_edf , dens , &dens_p , &dens_n , isospin , nxyz , delta_gpu , chi, ip , root_p , root_n , commw , hbar2m , dxyz , &latt , &fftrans , &status , tcur , fd ) ;

      
#ifdef BENCHMARK                  
	    sys_tim += e_it(0,&itgtod,&it_clk);
#endif      	    
	    if( ip == root_p && timeloop % (loop_io * 10) == 0 )
	      {
		fclose( fd ) ;
		fd = fopen( file_res , "a+" ) ;						
	      }
	  }
    
	MPI_Bcast( &e_tot , 1 , MPI_DOUBLE , 0 , commw );

#ifdef ROTATION

	    l0_time = (timeloop/loop_rot) % 6 ;
	    l1_time = ( (timeloop/loop_rot) - 1 + 6 ) % 6 ;
	    l2_time = ( (timeloop/loop_rot) - 2 + 6 ) % 6 ;
	    l3_time = ( (timeloop/loop_rot) - 3 + 6 ) % 6 ;
	    l4_time = ( (timeloop/loop_rot) - 4 + 6 ) % 6 ;
	    l5_time = ( (timeloop/loop_rot) - 5 + 6 ) % 6 ;
	    
	    
	    ratio = rotation(phi,  dens_n.rho, dens_p.rho, nxyz, &latt, dxyz);
	    
	    phi_xs[l0_time] = phi[0];	  
	    phi_ys[l0_time] = phi[1];	  
	    phi_zs[l0_time] = phi[2];
	    
	    fact_f1 = fact_f(tcur,10.,10.)*fact_rot(ratio);
	    
	    if( (timeloop/loop_rot) >= 6) get_omega(phi_xs, phi_ys, phi_zs, l0_time, l1_time, l2_time, l3_time, l4_time, l5_time, omega, dt_step*loop_rot);
	    for(i=0; i<3; i++) omega[i] = (fabs(omega[i]) < 1e-2)? omega[i] : 0.;//sqrt(ecm*2./L);//phi[i] / 200.*fact_f(ratio,1.,0.5);;

	    if(ip == 0){
	      for(i=0;i<3;i++) printf("phi[%d]  = %f,  ", i, phi[i]);
	      printf("\n");
	      
	      for(i=0;i<3;i++) printf("omega[%d]  = %f,  ", i, omega[i]);
	      printf("\n");
	    }
	    
	    
	    cudaMemcpy(d_potentials+16*nxyz+3, omega , 3*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ;
      
      
#endif
	    
      }
	
	
#ifdef SAVEOCC
	if(i_saveocc==0){
	  center_dist_pn( dens_p.rho , dens_n.rho , nxyz , &latt , rcm , rcm+1 , rcm+2 ) ;
	  cudaMemcpy( (void *)d_rcm , (const void *)rcm , 3*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice ) ;
	}
#endif
	
	if ( gr_ip != 0 )
	  { 
	    //GPU
#ifdef BENCHMARK                  	    
	    b_it(&itgtod,&it_clk);
#endif
          if(timeloop % i_norm == 0)
          {
            norm_switch = 1;
          }
          else
          {
            norm_switch = 0;
          }
	    adams_bashforth_dfdt_gpu(nxyz,nwfip,d_wavf+nn_time*nwfip*4*nxyz,d_wavf_td+mm_time*nwfip*4*nxyz,d_grad,d_lapl,d_plan_b,d_plan_r,d_potentials,batch,d_kxyz,i_saveocc,norm_switch,e_qp,e_sp,n_qp,d_plan_ev,d_fft3_ev,d_xyz,d_rcm,lz_qp);
#ifdef BENCHMARK                  	    
	    abm_tim+=e_it(0,&itgtod,&it_clk);
#endif
	  }
	
	
#ifdef LZCALC
	if((timeloop-1) % iopen == 0 && ip == 0 ){
	  sprintf( file_dens_jz , "jz.dat.%d", timeloop -1 );
	  f_crm( file_dens_jz );
	  ifd_jz = f_copn( file_dens_jz );
	  
	  
	  sprintf( file_dens_jz2 , "jz2.dat.%d", timeloop -1 );
	  f_crm( file_dens_jz2 );
	  ifd_jz2 = f_copn( file_dens_jz2 );
	}
	
	if( (timeloop - 1) % isave == 0 ){
	  center_dist_pn( dens_p.rho , dens_n.rho , nxyz , &latt , rcm , rcm+1 , rcm+2 ) ;
	  if(  ip == 0 ) printf( "xcm=%f ycm=%f zcm=%f \n" , rcm[0], rcm[1], rcm[2] );
	  MPI_Barrier(commw);
	  cudaMemcpy( (void *)d_rcm , (const void *)rcm , 3*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice ) ;
	  calc_jz_dens( d_jz_dens , dens_jz , copy_to_ , d_grad , d_wavf+nn_time*nwfip*4*nxyz, d_xyz , d_rcm , nxyz , nwfip , commw );
	  
	  if( ip == 0 ) f_cwr( ifd_jz , dens_jz , 8 , nxyz ) ;
	  
	  MPI_Barrier(commw);
	  
	  calc_jz2_dens( d_jz2_dens , dens_jz2 , copy_to_ , d_grad , d_wavf+nn_time*nwfip*4*nxyz, d_xyz , d_rcm , nxyz , nwfip , commw );
	  
	  if( ip == 0 ) f_cwr( ifd_jz2 , dens_jz2 , 8 , nxyz ) ;
	}
	
	
	if ( ( timeloop - 1 ) % iopen == iopen - isave && ip == 0 ){
	  f_ccls( &ifd_jz ) ;
	  
	  f_ccls( &ifd_jz2 ) ;
	}
#endif
	

#ifdef SAVEOCC
	if(i_saveocc==0){
	  if (gr_ip != 0 ){
	    for(i=0;i<nwfip;++i)
	      {
		n_qp[i]*=dxyz;
	      }
	    for(i=0; i< nwfip; i++) _n_qp_total[wavef_index[i]] = n_qp[i];
	    MPI_Send( n_qp , nwfip , MPI_DOUBLE , 0 , itag , gr_comm ) ; //0 in gr_comm .
	    
	    
	  }
	  if (gr_ip == 0 )
	    {
	      ioff=0;
	      for ( i = 0 ; i < np_c ; i++ )
		{
		  MPI_Recv( iobf + ioff , wf_tbl[ i ] , MPI_DOUBLE , i+1 , itag , gr_comm , &mpi_st ) ; 
		  ioff += wf_tbl[ i ] ;
		}
	      
	      if ( ( i = write( fd_n_qp , ( const void * ) iobf , (*nwf) * sizeof( double ) ) ) != (*nwf) * sizeof( double ) )
		printf("warning, could not copy the occupation numbers for time step %d.\n",timeloop);
	    for(i=0;i<20;++i){
	      if(isospin==-1) printf(" %12.6f" ,iobf[i]);
	      if(i%10==9)printf("\n");
	    }
	    }
	  MPI_Barrier( gr_comm );
	  if (gr_ip != 0 )
	    {
	      for(i=0; i< nwfip; i++) _e_qp_total[wavef_index[i]] = e_qp[i];
	      
	      //extra minus sign in spe: <v|h*-mu|v>/<v|v>
	      for(i=0; i< nwfip; i++) _e_sp_total[wavef_index[i]] = -1.0*e_sp[i];
	      MPI_Send( e_qp , nwfip , MPI_DOUBLE , 0 , itag , gr_comm ) ; //0 in gr_comm .
	      
	    }
	  if (gr_ip == 0 )
	    {
	      ioff=0;
	      for ( i = 0 ; i < np_c ; i++ )
		{
		  MPI_Recv( iobf + ioff , wf_tbl[ i ] , MPI_DOUBLE , i+1 , itag , gr_comm , &mpi_st ) ; 
		  ioff += wf_tbl[ i ] ;
		}
	      
	      if ( ( i = write( fd_e_qp , ( const void * ) iobf , (*nwf) * sizeof( double ) ) ) != (*nwf)* sizeof( double ) )
		printf("warning, could not copy the occupation numbers for time step %d.\n",timeloop);
	    }
	  
	  MPI_Barrier( gr_comm );
	  if (gr_ip != 0 )
	    MPI_Send( lz_qp , nwfip , MPI_DOUBLE , 0 , itag , gr_comm ) ; //0 in gr_comm .
	  if (gr_ip == 0 )
	    {
	      ioff=0;
	      for ( i = 0 ; i < np_c ; i++ )
		{
		  MPI_Recv( iobf + ioff , wf_tbl[ i ] , MPI_DOUBLE , i+1 , itag , gr_comm , &mpi_st ) ; 
		  ioff += wf_tbl[ i ] ;
		}
	      
	      if ( ( i = write( fd_lz_qp , ( const void * ) iobf , (*nwf) * sizeof( double ) ) ) != (*nwf)* sizeof( double ) )
		printf("warning, could not copy the occupation numbers for time step %d.\n",timeloop);
	    }
	  
	  // write the E_qp and Occ into txt file for checking
	  if (( timeloop - 1 ) % 100 == 0)
	    {
	      
	      MPI_Barrier(gr_comm);
	      
	      MPI_Allreduce(_e_qp_total, e_qp_total, *nwf, MPI_DOUBLE, MPI_SUM, gr_comm);
	      
	      MPI_Allreduce(_e_sp_total, e_sp_total, *nwf, MPI_DOUBLE, MPI_SUM, gr_comm);
	      
	      MPI_Allreduce(_n_qp_total, n_qp_total, *nwf, MPI_DOUBLE, MPI_SUM, gr_comm);
	      
	      if (gr_ip == 0)
		{
		  char fn[120];

		  sprintf(fn, "E_qp%s.dat.%d", iso_label, ( timeloop - 1 ));
		  
		  fd_eqp_txt =fopen(fn, "w");
	  
		  for(i=0;i<*nwf;i++)
		    fprintf(fd_eqp_txt, "%12.8f      %12.8f      %8.6f\n", e_qp_total[i], e_sp_total[i], n_qp_total[i]);
		  
		  fclose(fd_eqp_txt);
		}
	    }
	  /******/
	  
	}
#endif
	
	
	if( (timeloop % loop_cp) == 0 )
	  {
	    iwt = e_it(0,&itgtod,&it_clk);
	    ict = e_it(2,&itgtod,&it_clk);
	    if (ip==0) printf("(nx,ny,nz)[ %d %d %d ] nwf[ %d ] PEs[ %d ] timesteps[ %d ] Wtime[ %f ] Ctime[ %f ]\n",nx,ny,nz, nwf_p + nwf_n,np,timeloop-time_start+1, iwt, ict);
	    if (gr_ip != 0)
	      {
		for ( i = 0 ; i < 2 ; i++ )
		  {
		    cudaMemcpy(copy_buf, d_wavf + i*nwfip*4*nxyz , 4*nwfip*nxyz*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost) ; 
		    j = 0 ;
		    for ( iwf = 0 ; iwf < nwfip ; iwf++ )
		      for ( ixyz = 0 ; ixyz < 4*nxyz ; ixyz++ )
			{
			  wavef->wavf[i][iwf][ixyz] = copy_buf[ j ];
			  j++ ;
			}
		  }
		
	      }
	    
	    MPI_Barrier( commw ) ;
	    
	    knxyz = 4 * nxyz ;
	    sprintf( file_wf , "wf%s.cwr" , iso_label ) ;
	    write_wf_MPI(  file_wf, *nwf, nxyz, gr_ip, gr_np, gr_comm, &status, wavef->wavf[nn_time], wf_tbl);
    
	    if (ip == 0)
	      {
		fd_rs_info = fopen( "restart.info" , "wb" ) ;
		fwrite( (void *) &timeloop , sizeof(int) , 1 , fd_rs_info ) ;
		fclose( fd_rs_info ) ;
	      }
	    
	  }
	

	
	


#ifndef BENCHMARK
	run_tim += e_t( 0 ) ;
	MPI_Bcast( &run_tim , 1 , MPI_DOUBLE , 0 , commw ) ; /* forces a global synchronization here for flow control */
	b_t() ; /* restart the internal clock */
#endif
	/* check to see if we need to checkpoint */
	est_wr_tim = 4. * ( double ) nwf_n * ( double ) nx * ( double ) ny * ( double ) nz * 16. / ( 220. * 1048576. ) ; /* enforce POSIX I/O rule of thumb and assume more neutrons than protons for now */
	est_wr_tim *= 1.5 ; /* hopefully not wrong but skew it just in case */
	// terminate the loop when two fragments are well separated.
	if (  (( run_tim + est_wr_tim ) > req_tim ) || distance > d0 )
	  { /* conduct the checkpoint here */
    
	    iwt = e_it(0,&itgtod,&it_clk);
	    ict = e_it(2,&itgtod,&it_clk);
	    if (ip==0) printf("(nx,ny,nz)[ %d %d %d ] nwf[ %d ] PEs[ %d ] timesteps[ %d ] Wtime[ %f ] Ctime[ %f ]\n",nx,ny,nz, nwf_p + nwf_n,np,timeloop-time_start+1, iwt, ict);
	    if (gr_ip != 0)
	      {
		/* copy ~ (to , from, size_t , direction) */
		for ( i = 0 ; i < 2 ; i++ )
		  {
		    cudaMemcpy(copy_buf, d_wavf + i*nwfip*4*nxyz , 4*nwfip*nxyz*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost) ; 
		    j = 0 ;
		    for ( iwf = 0 ; iwf < nwfip ; iwf++ )
		      for ( ixyz = 0 ; ixyz < 4*nxyz ; ixyz++ )
			{
			  wavef->wavf[i][iwf][ixyz] = copy_buf[ j ];
			  j++ ;
			}
		  }
		free(copy_buf); // don't need it any longer
	      }
	    MPI_Barrier( commw ) ;
	        
	    //add gpu copy to cpu prior to CP 
	    knxyz = 4 * nxyz ;
	    sprintf( file_wf , "wf%s.cwr" , iso_label ) ;
	    write_wf_MPI(  file_wf, *nwf, nxyz, gr_ip, gr_np, gr_comm, &status, wavef->wavf[nn_time], wf_tbl);
	    
	    if (ip == 0)
	      {
		fd_rs_info = fopen( "restart.info" , "wb" ) ;
		fwrite( (void *) &timeloop , sizeof(int) , 1 , fd_rs_info ) ;
		fclose( fd_rs_info ) ;
	      }
    
#ifdef LZCALC
	    if ( ( ( timeloop - 1 ) % iopen ) != iopen - isave && ip == 0 ) f_ccls( &ifd_jz ) ;
	    
	    if ( ( ( timeloop - 1 ) % iopen ) != iopen - isave && ip == 0 ) f_ccls( &ifd_jz2 ) ;
#endif
	    
	    if ( gr_ip == 0 ) printf( "program terminated after CP\n") ;
	    
	    icp = 1;
	  break ;
	  }

      

      
	  }
  
#ifdef BENCHMARK
  run_tim = e_t(0);
#endif  
  if(ip == 0)
    printf("Step completed, time used %f s\n", run_tim);


#ifdef BENCHMARK
  if (ip==0)
    {
      fprintf(stdout, "process 0 is member of gr_ip 0...\n");
      fprintf(stdout, "KRTDSLDA: INIT TIME:: %f\n",init_tim);
      fprintf(stdout, "KRTDSLDA: TOTAL RUN TIME:: %f\n",run_tim);
      fprintf(stdout, "KRTDSLDA:\tABM time:: %f\n",abm_tim);      
      fprintf(stdout, "KRTDSLDA:\tDENS time:: %f\n",dens_comp_tim);
      fprintf(stdout, "KRTDSLDA:\tPotential time:: %f\n", pot_tim);
      fprintf(stdout, "KRTDSLDA:\tData I/o time:: %f\n",dens_io_tim);
      fprintf(stdout, "KRTDSLDA:\tEnergy(CPU) time:: %f\n",sys_tim);
      fprintf(stdout, "KRTDSLDA:\tCP time:: (see above)\n");
      fprintf(stdout, "data exchange time: %f \n", com_tim);
    }
  if (ip==2)
    {
      fprintf(stdout, "process 2 is member of gr_ip 2...\n");
      fprintf(stdout, "KRTDSLDA: INIT TIME:: %f\n",init_tim);
      fprintf(stdout, "KRTDSLDA: TOTAL RUN TIME:: %f\n",run_tim);
      fprintf(stdout, "KRTDSLDA:\tABM time:: %f\n",abm_tim);
      fprintf(stdout, "KRTDSLDA:\tDENS time:: %f\n",dens_comp_tim);
      fprintf(stdout, "KRTDSLDA:\tPotential time:: %f\n", pot_tim);
      fprintf(stdout, "KRTDSLDA:\tData I/o time:: %f\n",dens_io_tim);
      fprintf(stdout, "KRTDSLDA:\tEnergy(CPU) time:: %f\n",sys_tim);
      fprintf(stdout, "KRTDSLDA:\tCP time:: (see above)\n");
      fprintf(stdout, "data exchange time: %f \n", com_tim);
    }

  if(ip == 0){
    
  }


#endif
  if(gr_ip !=0)
    {
      
      cudaFree(d_wavf);
      cudaFree(d_wavf_td);
      cudaFree(d_wavf_c);
      cudaFree(d_wavf_p);
      cudaFree(d_wavf_m);
            
    }
  
  free(_e_qp_total);free(e_qp_total); 
  free(_n_qp_total);free(n_qp_total);
  free(_e_sp_total);free(e_sp_total);
  free(wavef_index); free(norms_tr); free(rnorms_tr); free(labels_tr);
      
  if ( ip == root_p )
    {
      printf("root closing FILE %d\n", (int) fd);
      fclose( fd ) ;
    }

  free( pots.u_re ) ; free( pots.mass_eff ) ; free( pots.wx ) ; free( pots.wy ) ; free( pots.wz ) ;
  free( pots.u1_x ) ; free( pots.u1_y ) ; free( pots.u1_z ) ; free( pots.ugrad_x ) ; free( pots.ugrad_y ) ; free( pots.ugrad_z ) ; free( pots.delta ) ;


  cudaFreeHost(copy_from); cudaFreeHost(copy_to);
  free(copy_from_new);



  
  // Free gpu buffers .
  cudaFree(d_xyz);
  cudaFree(d_kxyz);
  cudaFree(d_avf);
  cudaFree(d_fft3);
  cudaFree(d_wfft);
  cudaFree(d_fft3_ev);
  cudaFree(d_fft3_c);
  cudaFree(d_fc);
  cudaFree(d_map_ls);
  cudaFree(d_map_sl);
  cudaFree(d_densities);
  cudaFree(d_rcm);
#ifdef LZCALC
  cudaFree(d_jz_dens);
  cudaFree(d_jz2_dens);
#endif
  cudaFree(d_potentials);
  cudaFree(d_cc_pair_qf);
 
  if(gr_ip != 0){
    
    cudaFree(d_wavf);
    cudaFree(d_wavf_td);
    cudaFree(d_wavf_c);
    cudaFree(d_wavf_p);
    cudaFree(d_wavf_m);
    cudaFree(d_grad);
    cudaFree(d_lapl);
    
  }
  
  fftw_destroy_plan( fftrans.plan_f );
  fftw_destroy_plan( fftrans.plan_b );
  fftw_destroy_plan( fftrans.plan_f3 );
  fftw_destroy_plan( fftrans.plan_b3 );
  free( fftrans.buff ) ; free( fftrans.buff3 ) ;
  destroy_mpi_groups( &group_comm , &gr_comm ) ; 
  free( buff ) ; 


  
  free(frag.thetaL); free(frag.thetaR);
  free(frag.densf_n); free(frag.densf_p);
  free(divj);
  ///////

  
#ifdef SAVEOCC
  //need to write here, for each time step, e_qp, n_qp 
  //note that in this version they only exist on processors with gr_ip>0
  if(gr_ip==0){
    close(fd_n_qp);
    close(fd_e_qp);
    close(fd_lz_qp);
  }
#endif

  MPI_Barrier( commw ) ;
  if ( ip == 0 ) printf( " program terminated\n") ; 
  MPI_Finalize() ;
	
  return( EXIT_SUCCESS ) ;
}

