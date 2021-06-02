// for license information, see the accompanying LICENSE file

#include <stdlib.h>

#include <stdio.h>

#include <string.h>

#include "vars_nuclear.h"

#include <math.h>

#include <mpi.h> 

#include <assert.h>

#include <complex.h>

#include <fftw3.h>

extern double pi ;

void gradient_real( double * , const int , double * , double * , double * , const int , const int , const MPI_Comm , double complex * , double complex * , double complex * , const int , const int , const int , const int ) ;

void laplacean( double * , const int , double * , const int , const int , const MPI_Comm , double * , double * , double * , const int , const int , const int , const int ) ;

void get_u_re( const MPI_Comm , Densities * , Densities * , Potentials * , Couplings * , const double , const int , const int , const int , const int , const int , double * , double * , double * , const int , const int , const int , Lattice_arrays * , FFtransf_vars * , const double , const double ) ;

void coul_pot( double * , double * , double * , double * , Lattice_arrays * , const int , const int , const int , const double , FFtransf_vars * , const double ) ;

void coul_pot3( double * , double * , double * , double * , Lattice_arrays * , const int , const int , const int , const double , FFtransf_vars * , const double ) ;

void filter_hm_c( double complex * , const int , FFtransf_vars * ) ;

void filter_hm_r( double * , const int , FFtransf_vars * ) ;

double center_dist( double * , const int , Lattice_arrays * , double * , double * , double * ) ;

int dens_func_params( const int iforce , const int ihfb , const int isospin , Couplings * cc_edf , const int ip ,int icub)

{

  double t0 , t1 , t2 , t3 , x0 , x1 , x2 , x3 , w0 ;

  double a0, b0, c0, a1, b1, c1, a2, b2, c2, eta_s, a_n, b_n, c_n;

  a_n= -32.588; b_n= -115.44; c_n= 109.12;

  char edf_name[100] ;

  double mass_p = 938.272013 ;
      
  double mass_n = 939.565346 ;
  
  double hbarc = 197.3269631 ;
  
  double hbar2m = pow( hbarc , 2.0 ) / ( mass_p + mass_n ) ;        
  

  // default values
  t0 = 0.0 ;
  
  t1 = 0.0 ;
  
  t2 = 0.0 ;
  
  t3 = 0.0 ;
  
  x0 = 0.0 ;
  
  x1 = 0.0 ;
  
  x2 = 0.0 ;
  
  x3 = 0.0 ;
  
  cc_edf->gamma = 0.0 ;
  
  w0 = 0.0 ;
  
  a0 = 0.0 ;
  
  b0 = 0.0 ;
  
  c0 = 0.0 ;
  
  a1 = 0.0 ;
  
  b1 = 0.0 ;
  
  c1 = 0.0 ;
  
  a2 = 0.0 ;
  
  b2 = 0.0 ;
  
  c2 = 0.0 ;
  
  eta_s = 0.0 ;
  
  
  cc_edf->gg = 0.0 ;

  cc_edf->Skyrme = 1;

  cc_edf->iexcoul = 1;
  
  /* no force */
  if( iforce == 0 )	
    {
      sprintf( edf_name , "no intertion" ) ;
    }
  
  
  /*   SLy4 force */
  
  if ( iforce == 1 )    
    {
      if(icub==0) 
        sprintf( edf_name , "SLy4" ) ;
      else if(icub==1)
        sprintf( edf_name , "SLy4 with cubic cutoff" ) ;
      
      t0 = -2488.913 ;

      t1 =   486.818 ;

      t2 =  -546.395 ;

      t3 = 13777.000 ;

      x0 =      .834 ;

      x1 =     -.344 ;

      x2 =    -1.000 ;

      x3 =     1.354 ;

      cc_edf->gamma =  1.0 / 6.0 ;

      w0    =  123.0 ;

    if(icub==0)
      cc_edf->gg = -233.0 ; // original
    else if(icub==1)
      cc_edf->gg = -262.0 ; 

      cc_edf->rhoc = 0.;

    }
  /*   SLy4 force */

  if ( iforce == 11 )

    {

      sprintf( edf_name , "SLy4 with mixed pairing" ) ;

      t0 = -2488.913 ;

      t1 =   486.818 ;

      t2 =  -546.395 ;

      t3 = 13777.000 ;

      x0 =      .834 ;

      x1 =     -.344 ;

      x2 =    -1.000 ;

      x3 =     1.354 ;

      cc_edf->gamma =  1.0 / 6.0 ;

      w0    =  123.0 ;

      cc_edf->gg = -370. ; 

      cc_edf->rhoc=1./.32;

    }
  /*   SLy4 force */

  if ( iforce == 12 )

    {

      sprintf( edf_name , "SLy4 with surface pairing" ) ;

      t0 = -2488.913 ;

      t1 =   486.818 ;

      t2 =  -546.395 ;

      t3 = 13777.000 ;

      x0 =      .834 ;

      x1 =     -.344 ;

      x2 =    -1.000 ;

      x3 =     1.354 ;

      cc_edf->gamma =  1.0 / 6.0 ;

      w0    =  123.0 ;

      cc_edf->gg = -690. ; 

      cc_edf->rhoc=1./.16;

    }


  /*   SLy4 force */

  if ( iforce == 13 )

    {

      sprintf( edf_name , "SLy4 w/ .25 Vol + .75 Surface " ) ;

      t0 = -2488.913 ;

      t1 =   486.818 ;

      t2 =  -546.395 ;

      t3 = 13777.000 ;

      x0 =      .834 ;

      x1 =     -.344 ;

      x2 =    -1.000 ;

      x3 =     1.354 ;

      cc_edf->gamma =  1.0 / 6.0 ;

      w0    =  123.0 ;

      cc_edf->gg = -480. ; 

      cc_edf->rhoc=.75/.16;

    }

  /*   SkP force */

  if ( iforce == 2)

    { 

      sprintf( edf_name , "SkP" ) ;

      t0 = -2931.6960 ;

      t1 =   320.6182 ;

      t2 =  -337.4091 ;

      t3 = 18708.9600 ;

      x0 =     0.2921515 ;

      x1 =     0.6531765 ;

      x2 =    -0.5373230 ;

      x3 =     0.1810269 ;

      cc_edf->gamma =  1.0/6.0 ;

      w0    =  100.0 ;

      cc_edf->gg = -196.6 ;

      cc_edf->rhoc = 0.;

    }

  /*   SkM* force */

  if ( iforce == 3 || iforce == 4 )

    { 

      sprintf( edf_name , "SKM*" ) ;

      t0 = -2645.0 ;

      t1 =   410.0 ;

      t2 =  -135.0 ;

      t3 = 15595.0 ;

      x0 =     0.09 ;

      x1 =     0.0 ;

      x2 =     0.0 ;

      x3 =     0.0 ;

      cc_edf->gamma =  1.0/6.0 ;

      w0    =  130.0 ;

      cc_edf->gg = -184.7 ;

      if( iforce == 4 )

	{
	  
	  if( isospin == 1 )
	    {
	      if(icub==0)
	        cc_edf->gg = -340.0625 ;
	      else if(icub==1)
         	cc_edf->gg = -292.5417;
	    }
	  
	  else
	    {
	      if(icub==0)
	      cc_edf->gg = -265.2500 ; 
	      else if(icub==1)
	      cc_edf->gg = -225.3672;
	    }
	}

      cc_edf->rhoc = 0.5/.16;

    }

  /*   SLy6 force */

  if ( iforce == 6 )

    {

      sprintf( edf_name , "SLy6" ) ;

      t0 = -2479.500 ;

      t1 =   462.18 ;

      t2 =  -448.61 ;

      t3 = 13673.000 ;

      x0 =      .825 ;

      x1 =     -.465 ;

      x2 =    -1.000 ;

      x3 =     1.355 ;

      cc_edf->gamma =  1.0 / 6.0 ;

      w0    =  122.0 ;

      cc_edf->gg = -233.0 ;

      cc_edf->rhoc = 0.;

    }


  if ( iforce == 7 )

    {

      // Fix rho_c = 0.154 fm^-3
      // W0 is included in the fit
      // The best NEDF until now
      sprintf( edf_name , "SeaLL1" ) ;

      a0 = 0.0;
      
      b0 = -684.524043779;

      c0 = 827.26287841;

      a1 = 64.2474102072;

      b1 = 119.862146959;

      c1 = -256.492703921;

      
      a2 = -96.8354102072; 

      b2 = 449.22189682; 
	
      c2 = -461.650174489 ;
	
      eta_s = 81.3917529003 ;

      w0 = 73.5210618422 ;

      if(icub==0)
      cc_edf->gg = -200.0 ; // original
      else if(icub==1)
      cc_edf->gg = -230.0; // tuned to reproduce the same pairing gap with spherical cutoff.

      cc_edf->rhoc = 0.;

      cc_edf->Skyrme = 0;
                  
    }

  

  if( ip == 0 ){

    fprintf( stdout, " Density functional parameters \n" ) ;

    fprintf( stdout, " Density functional name: %s \n" , edf_name ) ;

    if(icub==1)
    printf("cubic-cutoff is used ");

    if( iforce != 7){

         fprintf( stdout, " ** NEDF parameters ** \n x0 = %f x1 = %f x2 = %f x3 = %f \n" , x0 , x1 , x2 , x3 );

	 fprintf( stdout, " t0 = %f t1 = %f t2 = %f t3 = %f gamma = %f w0 = %f \n" , t0 , t1 , t2 , t3 , cc_edf->gamma , w0 ) ;
    }
    else{
    
    fprintf( stdout, " ** NEDF parameters ** \n a0 = %.12lf b0 = %.12lf c0 = %.12lf \n a1 = %.12lf b1 = %.12lf c1 = %.12lf \n a2 = %.12lf b2 = %.12lf c2 = %.12lf eta_s = %.12lf \n" , a0, b0 , c0, a1, b1 , c1, a2, b2, c2, eta_s );

    fprintf( stdout, " spin-orbit strength w0 = %f \n" , w0 ) ;

  }

    fprintf( stdout, " ** Pairing parameters ** \n strength = %f \n" , cc_edf->gg ) ;

  }

  /*   Isoscalar and isovector coupling constants */

  cc_edf->c_rho_0      =  3.0 * t0 / 8. ;

  cc_edf->c_rho_1      = -.25 * t0 * ( x0 + 0.5 ) ;

  cc_edf->c_gamma_0    = t3 / 16. ;

  cc_edf->c_gamma_1    = - ( x3 + 0.5 ) * t3 / 24. ;


  // E_0 term 
  cc_edf->c_rho_a0 = a0;
  
  cc_edf->c_rho_b0 = b0;
  
  cc_edf->c_rho_c0 = c0;
  
  
  // E_2 term
  cc_edf->c_rho_a1 = a1;
  
  cc_edf->c_rho_b1 = b1;
  
  cc_edf->c_rho_c1 = c1;
      
  // E_4 term
  cc_edf->c_rho_a2 = a2;  
  
  cc_edf->c_rho_b2 = b2;  
  
  cc_edf->c_rho_c2 = c2;
  


      
  if(iforce != 7){

    cc_edf->c_laprho_0   = - ( 9.0 / 4.0 * t1 - t2 * ( x2 + 1.25) ) / 16. ;
    
    cc_edf->c_laprho_1   = - ( -3.0 * t1 * ( x1 + .5 ) - t2 *( x2 + .5 ) ) / 32. ;
    
    cc_edf->c_tau_0      = ( 3.0/16.0 * t1 + 0.25 * t2 * ( x2 + 1.25 ) ) ;
  
    cc_edf->c_tau_1      = - ( t1 * ( x1 + .5 ) - t2 * ( x2 + 0.5 ) ) / 8.0 ;
    
    cc_edf->c_divjj_0    = -0.750 * w0 ;
    
    cc_edf->c_divjj_1    = -0.250 * w0 ;
  }
  else{
    
    // (\rho \lap \rho) term 
    cc_edf->c_laprho_0   = - eta_s/2.;   
    
    // no isovector contribution
    cc_edf->c_laprho_1   = - eta_s/2.; // - ( -3.0 * t1 * ( x1 + .5 ) - t2 *( x2 + .5 ) ) / 32. ;
    
    // effective mass to be m* = m
    cc_edf->c_tau_0      = 0.; // ( 3.0/16.0 * t1 + 0.25 * t2 * ( x2 + 1.25 ) ) ;
    
    // no isovector contribution
    cc_edf->c_tau_1      = 0.; //- ( t1 * ( x1 + .5 ) - t2 * ( x2 + 0.5 ) ) / 8.0 ;
    
    cc_edf->c_divjj_0    =  - w0 ; //- C0 * r0*r0 * kappa * 0.8; // isoscalar part 
    
    cc_edf->c_divjj_1    =  0.0 ; // no isovector part
    
  }

  cc_edf->c_j_0        = - cc_edf->c_tau_0 ;

  cc_edf->c_j_1        = - cc_edf->c_tau_1 ;

  cc_edf->c_divj_0     =   cc_edf->c_divjj_0 ;

  cc_edf->c_divj_1     =   cc_edf->c_divjj_1 ;


    

  
  
  /*   Proton and neutron coupling constants */

  cc_edf->c_rho_p      =  cc_edf->c_rho_0    + isospin * cc_edf->c_rho_1 ;

  cc_edf->c_rho_n      =  cc_edf->c_rho_0    - isospin * cc_edf->c_rho_1 ;

  cc_edf->c_laprho_p   =  cc_edf->c_laprho_0 + isospin * cc_edf->c_laprho_1 ;

  cc_edf->c_laprho_n   =  cc_edf->c_laprho_0 - isospin * cc_edf->c_laprho_1 ;

  cc_edf->c_tau_p      =  cc_edf->c_tau_0    + isospin * cc_edf->c_tau_1 ;

  cc_edf->c_tau_n      =  cc_edf->c_tau_0    - isospin * cc_edf->c_tau_1 ;

  cc_edf->c_divjj_p    =  cc_edf->c_divjj_0  + isospin * cc_edf->c_divjj_1 ;

  cc_edf->c_divjj_n    =  cc_edf->c_divjj_0  - isospin * cc_edf->c_divjj_1 ;

  cc_edf->c_j_p        =  cc_edf->c_j_0      + isospin * cc_edf->c_j_1 ;

  cc_edf->c_j_n        =  cc_edf->c_j_0      - isospin * cc_edf->c_j_1 ;

  cc_edf->c_divj_p     =  cc_edf->c_divj_0   + isospin * cc_edf->c_divj_1 ;

  cc_edf->c_divj_n     =  cc_edf->c_divj_0   - isospin * cc_edf->c_divj_1 ;

  if ( ihfb == 0 ) 

    cc_edf->gg = 0.0 ;

  return ( EXIT_SUCCESS ) ;

}

void make_constraint( double * v , double * xa , double * ya , double * za , int nxyz , double y0 , double z0 , double asym , int * wx , int *wy , int *wz , double v0 ){

  int i;

  for(i=0;i<nxyz;i++){
    v[i]=2.*za[i]*za[i]-xa[i]*xa[i]-ya[i]*ya[i];  // shifted quadrupole
    v[i+nxyz]=wx[i]*xa[i];
    v[i+2*nxyz]=wy[i]*ya[i];
    v[i+3*nxyz]=wz[i]*za[i];
  }

}

void allocate_pots( Potentials * pots , const double hbar2m , double * pot_array , const int ishift , const int n )

{

  int i , nxyz ;

  nxyz = pots -> nxyz ;

  assert( pots -> u_re = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots -> mass_eff = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots -> wx = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots -> wy = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots -> wz = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots -> delta = malloc( nxyz * sizeof( double complex ) ) ) ;

  assert( pots -> delta_ext = malloc( nxyz * sizeof( double complex ) ) ) ;

  assert( pots -> v_ext = malloc( nxyz * sizeof( double ) ) ) ;

#ifdef CONSTRCALC

  assert( pots -> v_constraint = malloc( 4 * nxyz * sizeof( double ) ) ) ;

  assert( pots -> lam = malloc( 4 * sizeof( double ) ) ) ;

  assert( pots -> lam2 = malloc( 4 * sizeof( double ) ) ) ;

#endif

  for( i = 0 ; i < n ; i++ )

    *( pot_array + i ) = 0. ;

  pots->u_re = pot_array + ishift ;

  pots->mass_eff = pot_array + ishift + 4 * nxyz ;

  pots->wx = pot_array + ishift + nxyz ;

  pots->wy = pot_array + ishift + 2 * nxyz ;

  pots->wz = pot_array + ishift + 3 * nxyz ;

  pots->delta = ( double complex * ) ( pot_array + ishift + 5 * nxyz ) ; 

  pots->amu = pot_array + ishift + 7 * nxyz ;

  pots->lam = pot_array + ishift + 7 * nxyz + 1 ;

  for ( i = 0 ; i < nxyz ; i++ )

    {

      *( pots -> mass_eff + i ) = hbar2m ;

      *( pots -> v_ext + i ) = 0. ;

      *( pots -> delta_ext + i ) = 0. + 0. * I ;

#ifdef CONSTRCALC

      pots->v_constraint[ i ] = 0.;

      pots->v_constraint[ i + nxyz ] = 0.;

      pots->v_constraint[ i + 2 * nxyz ] = 0.;

      pots->v_constraint[ i + 3 * nxyz ] = 0.;

#endif

    }

}

void update_potentials( const int icoul , const int isospin , Potentials * pots , Densities * dens_p , Densities * dens_n , Densities * dens , Couplings * cc_edf , const double e_cut , const int nstart , const int nstop , const MPI_Comm comm , const int nx , const int ny , const int nz , const double hbar2m , double complex * d1_x , double complex * d1_y , double complex * d1_z , double * k1d_x , double * k1d_y , double * k1d_z , Lattice_arrays * latt_coords , FFtransf_vars * fftransf_vars , const double nprot, const double dxyz, int icub ) 

{

  int i , nxyz , j ;

  double ureal , pc , lc , p0 ;

  double * wso ;

  double complex * delta ;

  double pi2 , xp , xn ;

#ifdef CONSTR_Q0
  int j0=0;
#else
  int j0=1;
#endif

  pi2 = 4. * pow( pi , 2. ) ;

  nxyz = nx * ny * nz ;

  assert( delta = malloc( nxyz * sizeof( double complex ) ) ) ;

  for( i = 0 ; i < nxyz ; i++)

    delta[ i ] = 0. + 0. * I ;

  for( i = nstart ; i < nstop ; i++ )

    {

      ureal = pots->u_re[ i ] - *( pots->amu ) ;

#ifdef CONSTRCALC

      for( j = j0 ; j < 4 ; j++ ){

	ureal += pots->lam2[j]*pots->v_constraint[i+j*nxyz];

      }

#endif

      p0 = sqrt( fabs( ureal ) / pots->mass_eff[ i ] ) ;

      if( e_cut > ureal )

	{

	  pc = sqrt( ( e_cut - ureal ) / pots->mass_eff[ i ] ) ;

	  if( ureal < 0 )

	    pc = pc - .5 * p0 * log( ( pc + p0 ) / ( pc - p0 ) ) ;

	  else

	    pc = pc + p0 * atan( p0 / pc ) ;

	}

      else

	pc = 0. ;

      double gg=cc_edf->gg * ( 1. - cc_edf->rhoc* (dens_p->rho[i]+dens_n->rho[i]));      

      delta[ i ] = -  gg*dens->nu[ i - nstart ] / ( 1.0 - gg * pc / ( pi2 * pots->mass_eff[ i ] ) ) ;

     // coupling constant g independent of coordinates
      if (icub == 1)
	{
	  double kk = 2.442749;
	  
	  double dx = latt_coords->za[1] - latt_coords->za[0];
	  
	  delta[ i ] = - gg * dens->nu[ i - nstart ] / ( 1. - gg * kk / pots->mass_eff[ i] / 8.0 / (double) PI / dx) ;
	}
      
    }

  MPI_Allreduce( delta , pots->delta , nxyz , MPI_DOUBLE_COMPLEX , MPI_SUM , comm ) ;

  free( delta ) ;

  assert( wso = malloc( nxyz * sizeof( double ) ) ) ;

  for( i = 0 ; i < nxyz ; i++ )

    wso[ i ] = .5 * ( cc_edf->c_divjj_p * dens_p->rho[ i ] + cc_edf->c_divjj_n * dens_n->rho[ i ] ) ;

  gradient_real( wso , nxyz , pots->wx , pots->wy , pots->wz , nstart , nstop , comm , d1_x , d1_y , d1_z , nx , ny , nz , 1 ) ;

  free( wso ) ;

  get_u_re( comm , dens_p , dens_n , pots , cc_edf , hbar2m , nstart , nstop , nxyz , icoul , isospin , k1d_x , k1d_y , k1d_z , nx , ny , nz , latt_coords , fftransf_vars , nprot , dxyz ) ;

}

void get_u_re( const MPI_Comm comm , Densities * dens_p , Densities * dens_n , Potentials * pots , Couplings * cc_edf , const double hbar2m , const int nstart , const int nstop , const int nxyz , const int icoul , const int isospin , double * k1d_x , double * k1d_y , double * k1d_z , const int nx , const int ny , const int nz , Lattice_arrays * latt_coords , FFtransf_vars * fftransf_vars , const double nprot , const double dxyz ) 

{

  double * work1 , * work2 , * work3 ;

  int i ;

  double xpow=1./3.;

  double e2 = -197.3269631*pow(3./acos(-1.),xpow) / 137.035999679 ;

  assert( work1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( work2 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( work3 = malloc( nxyz * sizeof( double ) ) ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      *( work1 + i ) = cc_edf->c_laprho_p * dens_p->rho[ i ] + cc_edf->c_laprho_n * dens_n->rho[ i ] ;

      *( work3 + i ) = 0. ;

      *( pots->mass_eff + i ) = 0. ;

    }


  laplacean( work1 , nxyz , work2 , nstart , nstop , comm , k1d_x , k1d_y , k1d_z , nx , ny , nz , 0 ) ;

  for( i = nstart ; i < nstop ; i++ )

    {

      double rho_0 = dens_n->rho[i]  + dens_p->rho[i];

      double rho_1 = dens_p->rho[i]  - dens_n->rho[i];
      
      pots->mass_eff[ i ] = hbar2m + cc_edf->c_tau_p * dens_p->rho[ i ] + cc_edf->c_tau_n * dens_n->rho[ i ] ;

	double mass_eff_p = hbar2m + (cc_edf->c_tau_0 + cc_edf->c_tau_1)*dens_p->rho[i] + (cc_edf->c_tau_0 - cc_edf->c_tau_1)*dens_n->rho[i];
	double mass_eff_n = hbar2m + (cc_edf->c_tau_0 - cc_edf->c_tau_1)*dens_p->rho[i] + (cc_edf->c_tau_0 + cc_edf->c_tau_1)*dens_n->rho[i];

      if( cc_edf->Skyrme){
	*( work3 + i)  =	  
	  (  2. * ( cc_edf->c_rho_p * dens_p->rho[ i ] + cc_edf->c_rho_n * dens_n->rho[ i ] )
	     + cc_edf->c_gamma_0 * ( cc_edf->gamma + 2. ) * pow( dens_p->rho[ i ] + dens_n->rho[ i ] , cc_edf->gamma + 1. ) 	   
	     + cc_edf->c_gamma_1 * ( cc_edf->gamma * pow( dens_p->rho[ i ] + dens_n->rho[ i ] + 1.e-15, cc_edf->gamma - 1. ) * pow( dens_p->rho[ i ] - dens_n->rho[ i ] , 2. ) + 2. * ( double ) isospin * pow( dens_p->rho[ i ] + dens_n->rho[ i ] , cc_edf->gamma ) * ( dens_p->rho[ i ] - dens_n->rho[ i ] ) ) ) ;
      }
      else{
	// SeaLL1
	*( work3 + i) =
	  
	  (
	   // isoscalar part  
	   2. * cc_edf->c_rho_b0 * rho_0 + 5./3. * cc_edf->c_rho_a0 * pow(rho_0, 2./3.)
	   
	   + 7./3.*cc_edf->c_rho_c0 * pow(rho_0, 4./3.)
	   
	   - 1./3.*cc_edf->c_rho_a1 * pow(rho_1, 2.0) / (pow(rho_0, 4./3.) + 1e-14)
	   
	   + 1./3.*cc_edf->c_rho_c1 * pow(rho_1, 2.0) / (pow(rho_0, 2./3.) + 1e-14)
	   
	   - 7./3.*cc_edf->c_rho_a2 * pow(rho_1, 4.0) / (pow(rho_0, 10./3.) + 1e-14)
	   
	   - 2.0*cc_edf->c_rho_b2 * pow(rho_1, 4.0) / (pow(rho_0, 3.) + 1e-14)
	   
	   - 5./3.*cc_edf->c_rho_c2 *pow(rho_1, 4.0) / (pow(rho_0, 8./3.) + 1e-14)

	   // isovector part

	   + isospin * ( 2 * cc_edf->c_rho_a1 * rho_1 / (pow(rho_0, 1./3.) + 1e-14)
			 
			 + 2 * cc_edf->c_rho_b1 * rho_1
			 
			 + 2 * cc_edf->c_rho_c1 * rho_1 * pow(rho_0, 1./3.)
			 
			 + 4 * cc_edf->c_rho_a2 * pow(rho_1, 3.0) / (pow(rho_0, 7./3.) + 1e-14)
			 
			 + 4 * cc_edf->c_rho_b2 * pow(rho_1, 3.0) / (pow(rho_0, 2.0) + 1e-14)
			 
			 + 4 * cc_edf->c_rho_c2 * pow(rho_1, 3.0) / (pow(rho_0, 5./3.) + 1e-14)
			 
			 )
	   );
	   
      }
	  
	  
	  
      * (work3 + i) +=  (
			 // laprho part 
			 2. * work2[ i ] 
			 
			 + cc_edf->c_tau_p * dens_p->tau[ i - nstart ] + cc_edf->c_tau_n * dens_n->tau[ i - nstart ] 
			 
			 // local spin-orbit term
			 + cc_edf->c_divjj_p * dens_p->divjj[ i - nstart ] + cc_edf->c_divjj_n * dens_n->divjj[ i -nstart ] 
		 // external field
			 + pots->v_ext[i]
			 ) ;// + ( dens_p->rho[i]+dens_n->rho[i] ) * pots->v_ext[i] ) ;
   				   
    }

MPI_Allreduce( work3 , pots->u_re , nxyz , MPI_DOUBLE , MPI_SUM , comm ) ;

MPI_Allreduce( pots->mass_eff , work3 , nxyz , MPI_DOUBLE , MPI_SUM , comm ) ;
 
  for( i = 0 ; i < nxyz ; i++ )

    *( pots->mass_eff + i ) = *( work3 + i ) ;

  if( icoul == 1 )

    {

      coul_pot3( work3 , dens_p->rho , work1 , work2 , latt_coords , 1 , nxyz , nxyz , nprot , fftransf_vars , dxyz ) ; /* Vcoul in work3 now */

      for( i = 0 ; i < nxyz ; i++ ){

	pots->u_re[ i ] += work3[ i ] ;

	pots->u_re[i]+=(e2*pow(dens_p->rho[i],xpow))*cc_edf->iexcoul;

      }

    }

  free( work1 ) ;

  free( work3 ) ;

  free( work2 ) ;

}

void mix_potentials( double * pot_array , double * pot_array_old , const double alpha , const int ishift , const int n )

{

  int i ;

  double beta ;

  beta = 1. - alpha ;

  for( i = ishift ; i < n ; i++ )

    {

      *( pot_array + i ) =  alpha * *( pot_array + i ) + beta * *( pot_array_old + i ) ;

      *( pot_array_old + i ) = *( pot_array + i ) ;

    }

}

void mix_potentials1( double * pot_array , double * pot_array_old , const double alpha , const int ishift , const int n )

{

  int i ;

  double beta ;

  beta = 1. - alpha ;

  for( i = ishift ; i < n ; i++ )

    {

      *( pot_array + i ) =  alpha * *( pot_array + i ) + beta * *( pot_array_old + i ) ;

    }

}

void coul_pot( double * vcoul , double * rho , double * work1 , double * work2 , Lattice_arrays * latt_coords , const int nstart , const int nstop , const int nxyz , const double npart , FFtransf_vars * fftransf_vars , const double dxyz ) 

{

  double a2 , a_gauss , a_gauss_2 , a_gauss_sq , r2 , qzz ;

  double e2 ;
 
  int i ;

  double prot_number , z_sep , cnst , rp , rm ;

  double x_ch , y_ch , z_ch ;

  e2 = 197.3269631 / 137.035999679 ;

  prot_number = center_dist( rho , nxyz , latt_coords , &x_ch , &y_ch , &z_ch ) ;

  /*
    Calculate the charge density produced by a Gaussian
    It also takes into account the proton form factor  NOT YET IMPLEMENTED 

  */

  a2 = 0. ;

  qzz = 0. ;

  r2 = 0. ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      *( vcoul + i ) = 0. ;

      *( work1 + i ) = pow( *( latt_coords->xa + i ) - x_ch , 2. ) + pow( *( latt_coords->ya + i ) - y_ch , 2. ) ;

      *( work2 + i ) = *( work1 + i ) ;

      a2 += ( *( work1 + i ) * *( rho + i ) ) ;

      qzz += ( ( 2. * pow( *( latt_coords->za + i ) - z_ch , 2. ) - *( work1 + i ) ) * *( rho + i ) ) ;

      r2 += ( ( * ( work1 + i ) + pow( *( latt_coords->za + i ) - z_ch , 2. ) ) * *( rho + i ) ) ;

    } 

  a2 = a2 / prot_number ;

  a_gauss = sqrt( a2 ) ;

  qzz = fabs( qzz ) / r2 ;

  z_sep = a_gauss * sqrt( 1.5 * qzz / ( 2. - qzz ) ) ;

  cnst = pow( a2 * pi , 1.5 ) ; 

  prot_number = prot_number * dxyz ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      *( work1 + i ) += pow( *( latt_coords->za + i ) - z_ch - z_sep , 2. ) ;

      *( work2 + i ) += pow( *( latt_coords->za + i ) - z_ch + z_sep , 2. ) ;

      *( fftransf_vars->buff + i ) = ( double complex ) ( *( rho + i ) / prot_number - .5 * ( exp( - *( work1 + i ) / a2 ) + exp( - *( work2 + i ) / a2 ) ) / cnst ) ;

    }

  fftw_execute( fftransf_vars->plan_f ) ;

  * fftransf_vars->buff = 0. + 0. * I ;

  for( i = 1 ; i < nxyz ; i++ )

    *( fftransf_vars->buff + i ) = *( fftransf_vars->buff + i ) / ( pow( *( latt_coords->kx + i ) , 2. ) + pow( *( latt_coords->ky + i ) , 2. ) + pow( *( latt_coords->kz + i ) , 2. ) ) ;

  fftw_execute( fftransf_vars->plan_b ) ;

  cnst = 4. * pi / ( double ) nxyz ;

  prot_number = npart * e2 ;

  for( i = nstart ; i < nstop ; i++ )

    {

      rm = sqrt( *( work1 + i ) ) + 1.e-16 ;

      rp = sqrt( *( work2 + i ) ) + 1.e-16 ;

      *( vcoul + i ) = prot_number * ( cnst * creal( *( fftransf_vars->buff + i ) ) + 0.5 * ( erf( rp / a_gauss ) / rp + erf( rm / a_gauss ) / rm ) ) ; 

    }

}

double center_dist( double * rho , const int n , Lattice_arrays * latt_coords , double * xc , double * yc , double * zc )

{

  /* warning: the number of particles returned is missing a volume element dxyz */

  double part ;

  int i ;

  part = 0. ;

  * xc = 0. ;

  * yc = 0. ;

  * zc = 0. ;

  double xp=0.,xm=0.,yp=0.,ym=0.,zp=0.,zm=0.;

  for( i = 0 ; i < n ; i++ )

    {

      part += *( rho + i ) ;

      if( latt_coords->xa[i] > 0. )
	xp+=rho[i]*latt_coords->xa[i]*latt_coords->wx[i];
      else
	xm+=rho[i]*latt_coords->xa[i]*latt_coords->wx[i];

      if( latt_coords->ya[i] > 0. )
	yp+=rho[i]*latt_coords->ya[i]*latt_coords->wy[i];
      else
	ym+=rho[i]*latt_coords->ya[i]*latt_coords->wy[i];

      if( latt_coords->za[i] > 0. )
	zp+=rho[i]*latt_coords->za[i]*latt_coords->wz[i];
      else
	zm+=rho[i]*latt_coords->za[i]*latt_coords->wz[i];


    }

  *xc = (xp+xm) / part ;

  *yc = (yp+ym) / part ;

  *zc = (zp+zm) / part ;

  return( part ) ;

}

double center_dist_pn( double * rho_p , double * rho_n, const int n , Lattice_arrays * latt_coords , double * xc , double * yc , double * zc )

{

  /* warning: the number of particles returned is missing a volume element dxyz */

  double part , rho;

  int i ;

  part = 0. ;

  * xc = 0. ;

  * yc = 0. ;

  * zc = 0. ;

  double xp=0.,xm=0.,yp=0.,ym=0.,zp=0.,zm=0.;

  for( i = 0 ; i < n ; i++ )

    {

      rho=rho_p[i]+rho_n[i];

      part += rho ;

      if( latt_coords->xa[i] > 0. )
	xp+=rho*latt_coords->xa[i]*latt_coords->wx[i];
      else
	xm+=rho*latt_coords->xa[i]*latt_coords->wx[i];

      if( latt_coords->ya[i] > 0. )
	yp+=rho*latt_coords->ya[i]*latt_coords->wy[i];
      else
	ym+=rho*latt_coords->ya[i]*latt_coords->wy[i];

      if( latt_coords->za[i] > 0. )
	zp+=rho*latt_coords->za[i]*latt_coords->wz[i];
      else
	zm+=rho*latt_coords->za[i]*latt_coords->wz[i];

    }

  *xc = (xp+xm) / part ;

  *yc = (yp+ym) / part ;

  *zc = (zp+zm) / part ;

  return( part ) ;

}

void filter_hm_c( double complex * vec , const int n , FFtransf_vars * fftransf_vars )

{

  int i ;

  double xn ;

  xn = 1. / ( ( double ) n ) ;

  for( i = 0 ; i < n ; i++ )

    fftransf_vars->buff[ i ] = vec[ i ] ;

  fftw_execute( fftransf_vars->plan_f ) ;

  for( i = 0 ; i < n ; i++ )

    {

      fftransf_vars->buff[ i ] *= fftransf_vars->filter[ i ] ;

    }

  fftw_execute( fftransf_vars->plan_b ) ;

  for( i = 0 ; i < n ; i++ )

    vec[ i ] = fftransf_vars->buff[ i ] * xn ;

}

void filter_hm_r( double * vec , const int n , FFtransf_vars * fftransf_vars )

{

  int i ;

  double xn ;

  xn = 1. / ( ( double ) n ) ;

  for( i = 0 ; i < n ; i++ )

    fftransf_vars->buff[ i ] = vec[ i ] + I * 0. ;

  fftw_execute( fftransf_vars->plan_f ) ;

  for( i = 0 ; i < n ; i++ )

    {

      *( fftransf_vars->buff + i ) *= fftransf_vars->filter[ i ] ;

    }

  fftw_execute( fftransf_vars->plan_b ) ;

  for( i = 0 ; i < n ; i++ )

    vec[ i ] = creal( fftransf_vars->buff[ i ] ) * xn ;

}

void coul_pot3( double * vcoul , double * rho , double * work1 , double * work2 , Lattice_arrays * latt_coords , const int nstart , const int nstop , const int nxyz , const double npart , FFtransf_vars * fftransf_vars , const double dxyz )

{

  /* newest version, calculates Coulomb using a bigger box */

  int i , ii ;

  double xnxyz , xp ;

  xp = 0.;

  for( i = 0 ; i < nxyz ; i++ )

    xp += rho[ i ] ;   // renormalize to the correct number of particles for Coulomb

  xp *= dxyz ;

  xp = npart / xp ;

  xp = 1.;

  xnxyz = 1. / ( ( double ) fftransf_vars->nxyz3 ) ;

  for( i = 0 ; i < fftransf_vars->nxyz3 ; i++ )

    fftransf_vars->buff3[ i ] = 0. + 0. * I ;

  for( i = 0 ; i < nxyz ; i++ )

    fftransf_vars->buff3[ fftransf_vars->i_s2l[ i ] ] = ( rho[ i ] * xp + 0. * I ) ;

  fftw_execute( fftransf_vars->plan_f3 ) ;

  for( i = 0 ; i < fftransf_vars->nxyz3 ; i++ )

    fftransf_vars->buff3[ i ] *= fftransf_vars->fc[ i ] ;

  fftw_execute( fftransf_vars->plan_b3 ) ;

  for( i = 0 ; i < nxyz ; i++ )

    vcoul[ i ] = 0. ;

  for( i = 0 ; i < fftransf_vars->nxyz3 ; i++ )

    {

      ii = fftransf_vars->i_l2s[ i ] ;

      if( ii < nstart || ii > nstop - 1 ) continue ;

      vcoul[ ii ] = creal( fftransf_vars->buff3[ i ] ) * xnxyz ;
      
    }

}
