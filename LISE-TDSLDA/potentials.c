// for license information, see the accompanying LICENSE file

#include <assert.h>

#include <mpi.h>

#include <stdlib.h>

#include <stdio.h>

#include "vars.h"

#include "tdslda_func.h"

#include <math.h>

extern double pi; 

void diverg_real( double * f , double * fx , double * fy , double * fz , FFtransf_vars * fftrans , Lattice_arrays * latt , const int nxyz );

void allocate_phys_mem_pots( Potentials * pots , const int nxyz )

{

  assert( pots->u_re = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->mass_eff = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->wx = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->wy = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->wz = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->u1_x = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->u1_y = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->u1_z = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->ugrad_x = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->ugrad_y = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->ugrad_z = malloc( nxyz * sizeof( double ) ) ) ;

  assert( pots->delta = malloc( nxyz * sizeof( double complex ) ) ) ;

  assert( pots->v_ext = malloc( nxyz * sizeof( double ) ) ) ;

}

void free_phys_mem_pots( Potentials * pots )

{

  free( pots->u_re ) ;

  free( pots->mass_eff ) ;

  free( pots->wx ) ;

  free( pots->wy ) ;

  free( pots->wz ) ;

  free( pots->u1_x ) ;

  free( pots->u1_y ) ;

  free( pots->u1_z ) ;

  free( pots->ugrad_x ) ;

  free( pots->ugrad_y ) ;

  free( pots->ugrad_z ) ;

  free( pots->delta ) ;

}

void ext_field( Potentials * pots , int ichoice , Lattice_arrays * latt , const int nxyz )

{

  int i ;

  switch( ichoice )

    {

    case 11: /* sin( kx * x ) */

      for( i = 0 ; i < nxyz ; i++ )

	pots->v_ext[ i ] = sin( latt->k1x * latt->xa[ i ] ) ;

      break ;

    case 12: /* sin( ky * y ) */

      for( i = 0 ; i < nxyz ; i++ )

	pots->v_ext[ i ] = sin( latt->k1y * latt->ya[ i ] ) ;

      break ;

    case 13: /* sin( kz * z ) */

      for( i = 0 ; i < nxyz ; i++ )

	pots->v_ext[ i ] = sin( latt->k1z * latt->za[ i ] ) ;

      break ;

    case 21: /* cos( kx * x ) */

      for( i = 0 ; i < nxyz ; i++ )

	pots->v_ext[ i ] = cos( latt->k1x * latt->xa[ i ] ) ;

      break ;

    case 22: /* cos( ky * y ) */

      for( i = 0 ; i < nxyz ; i++ )

	pots->v_ext[ i ] = cos( latt->k1y * latt->ya[ i ] ) ;

      break;

    case 23: /* cos( kz * z ) */

      for( i = 0 ; i < nxyz ; i++ )

	pots->v_ext[ i ] = cos( latt->k1z * latt->za[ i ] ) ;

      break ;

    default:

      for( i = 0 ; i < nxyz ; i++ )

	pots->v_ext[ i ] = 0. ;

      break ;

    }

}

int dens_func_params( const int iforce , const int ihfb , const int isospin , Couplings * cc_edf , const int ip, int icub)

{

  double a_n, b_n ,c_n;
  
  a_n= -32.588; b_n= -115.44; c_n= 109.12;
    
  double a0, b0, c0, a1, b1, c1, a2, b2, c2, eta_s, w0 ;

  double t0 , t1 , t2 , t3 , x0 , x1 , x2 , x3 ;

  char edf_name[100] ;

  double mass_p = 938.272013 ;
      
  double mass_n = 939.565346 ;
  
  double hbarc = 197.3269631 ;
  
  double hbar2m = pow( hbarc , 2.0 ) / ( mass_p + mass_n ) ;        

  /* no force */

  cc_edf->rhoc=0.;

  t0 = 0.0 ;

  t1 = 0.0 ;
  
  t2 = 0.0 ;
  
  t3 = 0.0 ;
  
  x0 = 0.0 ;
  
  x1 = 0.0 ;
  
  x2 = 0.0 ;

  x3 = 0.0 ;
  
  cc_edf->gamma = 0.0 ;
  
  w0    = 0.0 ;
  
  a0 = 0.0;
  
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

  cc_edf->gg_p = cc_edf->gg;

  cc_edf->gg_n = cc_edf->gg;

  cc_edf->Skyrme = 1;

  cc_edf->iexcoul = 1;
     
  /* no force */
  if( iforce == 0 )
    {
      sprintf( edf_name , "no interaction" ) ;
    }
 
  /*   SLy4 force */

  if ( iforce == 1 ) {

      sprintf( edf_name , "SLy4" ) ;

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

      cc_edf->gg = -233.0 ;

if(icub==1)
      cc_edf->gg = -262.0 ;

      cc_edf->gg_p = cc_edf->gg;

      cc_edf->gg_n = cc_edf->gg;

      cc_edf->rhoc=0.;

    }

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

      cc_edf->gg_p = cc_edf->gg;

      cc_edf->gg_n = cc_edf->gg;

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

      cc_edf->gg_p = cc_edf->gg;

      cc_edf->gg_n = cc_edf->gg;

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

      cc_edf->gg_p = cc_edf->gg;

      cc_edf->gg_n = cc_edf->gg;

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

      cc_edf->gg = -250.0 ;

      cc_edf->gg_p = cc_edf->gg;

      cc_edf->gg_n = cc_edf->gg;

    }

  /*   SkM* force */

  if ( iforce == 3 || iforce == 4 || iforce == 5)

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

      cc_edf->gg_p = cc_edf->gg;

      cc_edf->gg_n = cc_edf->gg;

      if( iforce == 4 || iforce==5)

	{
	  
	  if( isospin == 1 )
	    {
	      cc_edf->gg = -340.0625 ;
	      
if(icub==1)
	     cc_edf->gg = -292.5417;
	    }
	  
	  else
	    {
	      cc_edf->gg = -265.2500 ;
	    
if(icub==1)
	      cc_edf->gg = -225.3672;
	    }
	}

      if(iforce==4 || iforce==5){
        if(icub==0){
  	        cc_edf->gg_p = -292.5417; 
	        cc_edf->gg_n = -225.3672; 
        }
        else if(icub==1){
          cc_edf->gg_p = -325.90 ;
          cc_edf->gg_n = -240.99 ;
        }
      }

      cc_edf->rhoc = 0.5/.16;

      if(iforce==5){
        cc_edf->rhoc = 0.0;
        sprintf( edf_name , "SKM* Volume Pairing" ) ;
      }
    }


  /**** SeaLL1 NEDF ****/
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

      cc_edf->gg = -200.0 ; // original

if(icub==1)
      cc_edf->gg = -230.0; // tuned to reproduce the same pairing gap with spherical cutoff.

      cc_edf->gg_p = cc_edf->gg;

      cc_edf->gg_n = cc_edf->gg;

      cc_edf->rhoc = 0.;

      cc_edf->Skyrme = 0;

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


  if( cc_edf->Skyrme ){

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

  cc_edf->c_isospin    =  (double) isospin ;

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
  }

  if ( ihfb == 0 ) 

    cc_edf->gg = 0.0 ;

  return ( EXIT_SUCCESS ) ;

}

void update_potentials( const int isospin , Potentials * pots , Densities * dens_p , Densities * dens_n , Densities * dens , Couplings * cc_edf , const double e_cut , const int nxyz , const double hbar2m , Lattice_arrays * latt , FFtransf_vars * fftrans , const double dxyz , double * sx ,int icub) 

{

  int i ;

  double pc , lc , p0 ;

  double pi2 , pi;

  double * sy , * sz ;

  pi2 = 4. * pow( acos( -1.0 ) , 2. ) ;

  pi = acos(-1.0);

  assert( sy = malloc( nxyz * sizeof( double ) ) ) ;

  assert( sz = malloc( nxyz * sizeof( double ) ) ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      p0 = sqrt( fabs( pots->u_re[ i ] ) / pots->mass_eff[ i ] ) ;

      if( e_cut > pots->u_re[ i ] )

	{

	  pc = sqrt( ( e_cut - pots->u_re[ i ] ) / pots->mass_eff[ i ] ) ;

	  if( pots->u_re[ i ] < 0 )

	    pc = pc - .5 * p0 * log( ( pc + p0 ) / ( pc - p0 ) ) ;

	  else

	    pc = pc + p0 * atan( p0 / pc ) ;

	}

      else

	pc = 0. ;

      pots->delta[ i ] = - cc_edf->gg * dens->nu[ i ] / ( 1.0 - cc_edf->gg * pc / ( pi2 * pots->mass_eff[ i ] ) ) ;


if(icub==1){
      // coupling constant g independent of coordinates
      
      double kk = 2.442749;
      
      double dx = latt->za[1] - latt->za[0];
      
      double gg = -230.0;
      
      pots->delta[ i ] = - gg * dens->nu[ i ] / ( 1. - gg * kk / pots->mass_eff[ i] / 8.0 / pi / dx) ;
	}

      //if( isospin == 1 ) pots->delta[ i ] = 0. ;

      sx[ i ] = cc_edf->c_divj_p * dens_p->sx[ i ] + cc_edf->c_divj_n * dens_n->sx[ i ] ; 

      sy[ i ] = cc_edf->c_divj_p * dens_p->sy[ i ] + cc_edf->c_divj_n * dens_n->sy[ i ] ;

      sz[ i ] = cc_edf->c_divj_p * dens_p->sz[ i ] + cc_edf->c_divj_n * dens_n->sz[ i ] ;

    }

  curl( sx , sy , sz , pots->ugrad_x , pots->ugrad_y , pots->ugrad_z , nxyz , fftrans , latt ) ;

  for( i = 0 ; i < nxyz ; i++ ){

    sx[ i ] = cc_edf->c_j_p * dens_p->jx[ i ] + cc_edf->c_j_n * dens_n->jx[ i ] ;

    sy[ i ] = cc_edf->c_j_p * dens_p->jy[ i ] + cc_edf->c_j_n * dens_n->jy[ i ] ;

    sz[ i ] = cc_edf->c_j_p * dens_p->jz[ i ] + cc_edf->c_j_n * dens_n->jz[ i ] ; 

  }

  diverg_real( pots->u_im , sx , sy , sz , fftrans , latt , nxyz ) ;
  
  get_u_re( dens_p , dens_n , pots , cc_edf , hbar2m , nxyz , isospin , latt , fftrans , dxyz ) ;

  free( sy ) ; free( sz ) ;

  diverg_real( sx , pots->a_vf[ 1 ] , pots->a_vf[ 2 ] , pots->a_vf[ 3 ] , fftrans , latt , nxyz ) ;

  for( i = 0 ; i < nxyz ; i++ ){

    pots->u_im[ i ] -= pots->mass_eff[ i ] * sx[ i ] ;

    sx[ i ] = cc_edf->c_divjj_p * dens_p->rho[ i ] + cc_edf->c_divjj_n * dens_n->rho[ i ] ;

  }

  gradient_real( sx , pots->wx , pots->wy , pots->wz , fftrans , latt , nxyz ) ;

  gradient_real( pots->mass_eff , pots->mgrad_x , pots->mgrad_y , pots->mgrad_z , fftrans , latt , nxyz ) ;

  for( i = 0 ; i < nxyz ; i++ ) {

    pots->ugrad_x[ i ] += ( 2. * ( cc_edf->c_j_p * dens_p->jx[ i ] + cc_edf->c_j_n * dens_n->jx[ i ] ) );

    pots->ugrad_y[ i ] += ( 2. * ( cc_edf->c_j_p * dens_p->jy[ i ] + cc_edf->c_j_n * dens_n->jy[ i ] ) );

    pots->ugrad_z[ i ] += ( 2. * ( cc_edf->c_j_p * dens_p->jz[ i ] + cc_edf->c_j_n * dens_n->jz[ i ] ) );

    pots->u1_x[ i ] = ( cc_edf->c_divj_p * dens_p->cjx[ i ] + cc_edf->c_divj_n * dens_n->cjx[ i ] );

    pots->u1_y[ i ] = ( cc_edf->c_divj_p * dens_p->cjy[ i ] + cc_edf->c_divj_n * dens_n->cjy[ i ] );

    pots->u1_z[ i ] = ( cc_edf->c_divj_p * dens_p->cjz[ i ] + cc_edf->c_divj_n * dens_n->cjz[ i ] );

  }

  for( i = 0 ; i < nxyz ; i++ ) {

    pots->ugrad_x[ i ]  -= ( 2. * pots->mass_eff[ i ] * pots->a_vf[ 1 ][ i ] ) ;

    pots->ugrad_y[ i ]  -= ( 2. * pots->mass_eff[ i ] * pots->a_vf[ 2 ][ i ] ) ;

    pots->ugrad_z[ i ]  -= ( 2. * pots->mass_eff[ i ] * pots->a_vf[ 3 ][ i ] ) ;

  }

}

void get_u_re( Densities * dens_p , Densities * dens_n , Potentials * pots , Couplings * cc_edf , const double hbar2m , const int nxyz , const int isospin , Lattice_arrays * latt , FFtransf_vars * fftrans , const double dxyz ) 

{

  double * work1 , * work2 ;

  int i ;

  // for exchange potential
  double xpow=1./3.;

  double e2 = -197.3269631*pow(3./acos(-1.),xpow) / 137.035999679 ;

  double pi;
  pi = acos(-1.0);
  double kk = 2.442749;
  double dx = latt->za[1] - latt->za[0];
  double hbarc = 197.3269631 ;
  double ggp = -292.5417;
  double ggn = -225.3672;
  double prefactor;
  prefactor  =kk/8.0/ pi /dx; 

  double numerator_p, numerator_n;
  numerator_p = -1.0*prefactor * ggp*ggp;
  numerator_n = -1.0*prefactor * ggn*ggn;

  assert( work1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( work2 = malloc( nxyz * sizeof( double ) ) ) ;

  if( isospin == -1 ) /* neutrons */

    for( i = 0 ; i < nxyz ; i++ )

      {

	work1[ i ] = cc_edf->c_laprho_p * dens_p->rho[ i ] + cc_edf->c_laprho_n * dens_n->rho[ i ] ;

	pots->u_re[ i ] = 0. ;

      }

  else

    {

      coul_pot3( pots->u_re , dens_p->rho , work1 , work2 , latt , nxyz , fftrans , dxyz ) ; /* Vcoul */

      for( i = 0 ; i < nxyz ; i++ )

	{

	  work1[ i ] = cc_edf->c_laprho_p * dens_p->rho[ i ] + cc_edf->c_laprho_n * dens_n->rho[ i ] ;

	}

    }

  laplacean( work1 , work2 , nxyz , fftrans , latt ) ;

  free( work1 ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {
      double rho_0 = dens_n->rho[i]  + dens_p->rho[i];

      double rho_1 = dens_p->rho[i]  - dens_n->rho[i];
      
      
      pots->mass_eff[ i ] = ( hbar2m + cc_edf->c_tau_p * dens_p->rho[ i ] + cc_edf->c_tau_n * dens_n->rho[ i ] ) ;
      
      double mass_eff_p = hbar2m + (cc_edf->c_tau_0 + cc_edf->c_tau_1)*dens_p->rho[i] + (cc_edf->c_tau_0 - cc_edf->c_tau_1)*dens_n->rho[i];
      double mass_eff_n = hbar2m + (cc_edf->c_tau_0 - cc_edf->c_tau_1)*dens_p->rho[i] + (cc_edf->c_tau_0 + cc_edf->c_tau_1)*dens_n->rho[i];


 if( cc_edf->Skyrme){
        pots->u_re[i] +=          
          (  2. * ( cc_edf->c_rho_p * dens_p->rho[ i ] + cc_edf->c_rho_n * dens_n->rho[ i ] )
             + cc_edf->c_gamma_0 * ( cc_edf->gamma + 2. ) * pow( dens_p->rho[ i ] + dens_n->rho[ i ] , cc_edf->gamma + 1. )        
             + cc_edf->c_gamma_1 * ( cc_edf->gamma * pow( dens_p->rho[ i ] + dens_n->rho[ i ] + 1.e-15, cc_edf->gamma - 1. ) * pow( dens_p->rho[ i ] - dens_n->rho[ i ] , 2. ) + 2. * ( double ) isospin * pow( dens_p->rho[ i ] + dens_n->rho[ i ] , cc_edf->gamma ) * ( dens_p->rho[ i ] - dens_n->rho[ i ] ) ) ) ;
	}
      
      else{

      pots->u_re[ i ]  +=

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

	 + cc_edf->c_isospin * ( 2 * cc_edf->c_rho_a1 * rho_1 / (pow(rho_0, 1./3.) + 1e-14)

		       + 2 * cc_edf->c_rho_b1 * rho_1

		       + 2 * cc_edf->c_rho_c1 * rho_1 * pow(rho_0, 1./3.)

		       + 4 * cc_edf->c_rho_a2 * pow(rho_1, 3.0) / (pow(rho_0, 7./3.) + 1e-14)

		       + 4 * cc_edf->c_rho_b2 * pow(rho_1, 3.0) / (pow(rho_0, 2.0) + 1e-14)

		       + 4 * cc_edf->c_rho_c2 * pow(rho_1, 3.0) / (pow(rho_0, 5./3.) + 1e-14)
		      
		      ) ) ;

	}
	 // laprho part (eta_s part)
	 + ( 2 * work2[ i ] 

	 + cc_edf->c_tau_p * dens_p->tau[ i ] + cc_edf->c_tau_n * dens_n->tau[ i ] 

	 // local spin-orbit term
	 + cc_edf->c_divjj_p * dens_p->divjj[ i ] + cc_edf->c_divjj_n * dens_n->divjj[ i ] 
	 
	 - * ( pots->amu ) + pots->a_vf[ 0 ][ i ] - pots->mass_eff[ i ] * ( pots->a_vf[ 1 ][ i ] * pots->a_vf[ 1 ][ i ] + pots->a_vf[ 2 ][ i ] * pots->a_vf[ 2 ][ i ] + pots->a_vf[ 3 ][ i ] * pots->a_vf[ 3 ][ i ] ) + pots->ctime * pots->v_ext[i] ) ;

      if(isospin == 1)
	pots->u_re[ i ] += e2*pow(dens_p->rho[i],xpow);

   double denominator_p, denominator_n;
   denominator_p = pow((mass_eff_p - prefactor*ggp),2.0);
   denominator_n = pow((mass_eff_n - prefactor*ggn),2.0);

 if( cc_edf->Skyrme){

    pots->u_re[i] += numerator_p/denominator_p*cc_edf->c_tau_p *  creal(dens_p->nu[i]*conj(dens_p->nu[i]))
             + numerator_n/denominator_n*cc_edf->c_tau_n * creal(dens_n->nu[i]*conj(dens_n->nu[i]));  // Careful diving by zero here.
//  Only programmed from cubic case so far. 
     }


    }

  free( work2 ) ;

}


void coul_pot( double * vcoul , double * rho , double * work1 , double * work2 , Lattice_arrays * latt_coords , const int nxyz , FFtransf_vars * fftransf_vars , const double dxyz ) 

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

  cnst = pow( a2 * acos( -1. ) , 1.5 ) ; 

  prot_number *= dxyz ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      *( work1 + i ) += pow( *( latt_coords->za + i ) - z_ch - z_sep , 2. ) ;

      *( work2 + i ) += pow( *( latt_coords->za + i ) - z_ch + z_sep , 2. ) ;

      *( fftransf_vars->buff + i ) = rho[ i ] / prot_number - .5 * ( exp( - *( work1 + i ) / a2 ) + exp( - *( work2 + i ) / a2 ) ) / cnst + I * 0. ;

    }

  fftw_execute( fftransf_vars->plan_f ) ;

  * fftransf_vars->buff = 0. + 0. * I ;

  for( i = 1 ; i < nxyz ; i++ )

    *( fftransf_vars->buff + i ) = *( fftransf_vars->buff + i ) / ( latt_coords->kin[ i ] ) ;

  fftw_execute( fftransf_vars->plan_b ) ;

  cnst = 4. * acos( -1. ) / ( ( double ) nxyz ) ;

  prot_number *= e2 ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      rm = sqrt( *( work1 + i ) ) + 1.e-18 ;

      rp = sqrt( *( work2 + i ) ) + 1.e-18 ;

      *( vcoul + i ) = prot_number * ( cnst * creal( *( fftransf_vars->buff + i ) ) + 0.5 * ( erf( rp / a_gauss ) / rp + erf( rm / a_gauss ) / rm ) ) ; 

    }

}

void coul_pot3( double * vcoul , double * rho , double * work1 , double * work2 , Lattice_arrays * latt_coords , const int nxyz , FFtransf_vars * fftransf_vars , const double dxyz )

{

  /* newest version, calculates Coulomb using a bigger box */

  int i , ii ;

  double xnxyz ;

  xnxyz = 1. / ( ( double ) fftransf_vars->nxyz3 ) ;

  for( i = 0 ; i < fftransf_vars->nxyz3 ; i++ )

    fftransf_vars->buff3[ i ] = 0. + 0. * I ;

  for( i = 0 ; i < nxyz ; i++ )

    fftransf_vars->buff3[ fftransf_vars->i_s2l[ i ] ] = ( rho[ i ] + 0. * I ) ;

  fftw_execute( fftransf_vars->plan_f3 ) ;

  for( i = 0 ; i < fftransf_vars->nxyz3 ; i++ )

    fftransf_vars->buff3[ i ] *= fftransf_vars->fc[ i ] ;

  fftw_execute( fftransf_vars->plan_b3 ) ;

  for( i = 0 ; i < nxyz ; i++ )

    vcoul[ i ] = 0. ;

  for( i = 0 ; i < fftransf_vars->nxyz3 ; i++ ){

      ii = fftransf_vars->i_l2s[ i ] ;

      if( ii < 0 ) continue ;

      vcoul[ ii ] = creal( fftransf_vars->buff3[ i ] ) * xnxyz ;
      
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

  for( i = 0 ; i < n ; i++ )

    {

      part += *( rho + i ) ;

      * xc += *( latt_coords->xa + i ) * *( rho + i ) ;

      * yc += *( latt_coords->ya + i ) * *( rho + i ) ;

      * zc += *( latt_coords->za + i ) * *( rho + i ) ;

    }

  *xc = *xc / part ;

  *yc = *yc / part ;

  *zc = *zc / part ;

  return( part ) ;

}


double center_dist_pn( double * rho_p , double * rho_n , const int n , Lattice_arrays * latt_coords , double * xc , double * yc , double * zc )

{

  /* warning: the number of particles returned is missing a volume element dxyz */

  double part ;

  int i ;

  part = 0. ;

  * xc = 0. ;

  * yc = 0. ;

  * zc = 0. ;

  double rho;

  for( i = 0 ; i < n ; i++ )

    {

      rho=rho_p[i]+rho_n[i];

      part +=  rho ;

      * xc += *( latt_coords->xa + i ) * rho ;

      * yc += *( latt_coords->ya + i ) * rho ;

      * zc += *( latt_coords->za + i ) * rho ;

    }

  *xc = *xc / part ;

  *yc = *yc / part ;

  *zc = *zc / part ;

  return( part ) ;

}

void bcast_potentials( Potentials * pots , const int nxyz , const MPI_Comm comm )

{

  MPI_Bcast( pots->u_re , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->u_im , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->mass_eff , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->wx , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->wy , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->wz , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->u1_x , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->u1_y , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->u1_z , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->ugrad_x , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->ugrad_y , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->ugrad_z , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->mgrad_x , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->mgrad_y , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->mgrad_z , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( pots->delta , nxyz , MPI_DOUBLE_COMPLEX , 0 , comm ) ;

}

void absorbing_potential( const double lx , const double ly , const double lz , Lattice_arrays * latt , double * w_abs , const int nxyz ){

  int i ;

  double x_wall , y_wall , z_wall ;

  double w0_strength = -0.1 ;

  x_wall = .35 * lx ;

  y_wall = .35 * ly ;

  z_wall = .35 * lz ;

  for( i = 0 ; i < nxyz ; i++ ){

    w_abs[ i ] = 0. ;

    if( fabs( latt->xa[ i ] ) > x_wall ) w_abs[ i ]  = ( fabs( latt->xa[ i ] ) - x_wall ) ;

    if( fabs( latt->ya[ i ] ) > y_wall ) w_abs[ i ] += ( fabs( latt->ya[ i ] ) - y_wall ) ;

    if( fabs( latt->za[ i ] ) > z_wall ) w_abs[ i ] += ( fabs( latt->za[ i ] ) - z_wall ) ;

    w_abs[ i ] *= w0_strength ;

  }

  return ;

}

void strength_abs( const double time , double complex * w0 ){

  const double tau = 30. ; double time0 = 360. ;

  * w0 = 0. * I * .5 * ( erf( ( time - time0 ) / tau ) + 1. ) ;

}
