
#include <stdlib.h>

#include <math.h>

#include <complex.h>

#include "vars_nuclear.h"

#include <mpi.h>


void gradient_real( double * , const int , double * , double * , double * , const int , const int , const MPI_Comm , double complex * , double complex * , double complex * , const int , const int , const int , const int ) ;

void external_pot( const int iext , const int n , const int n_part , const double hbo , double * v_ext , double complex * delta_ext , const double hbar2m , Lattice_arrays * lattice_coords , const double rr , const double rneck, const double wneck , const double z0 , const double v0 )

{
  /* iext: in, choice for external potentials */
  /* n : in , Nxyz */
  /* N_particles: in , total number of particles, needed for some options */
  /* hbo: in , real parameter, hbar omega for the HO choice */
  /* v_ext: out external potential */
  /* delta_ext: out , external pairing field */

  int i ;

  double hw , r1 , r2 ;

  /* iext = 0 : no external field */

  if( iext == 1 )  /* HO */

    {

      hw = 0.25 * hbo * hbo / hbar2m ;
      
      for( i = 0 ; i < n ; i++ )

	*( v_ext + i ) = hw * ( pow( * ( lattice_coords->xa + i ) , 2. ) + pow( * ( lattice_coords->ya + i ) , 2. ) + pow( * ( lattice_coords->za + i ) , 2. ) ) ;

    }

  if( iext == 2 ) /* External Woods-Saxon potential, proportional external pairing */

    {

      hw = ( double ) n_part ;

      hw = 1.2 * pow( hw , 1./3. ) ;

      for( i = 0 ; i < n ; i++ )

	{

	  *( v_ext + i ) = -50.0 / ( 1. + exp( sqrt( pow( *( lattice_coords->xa + i ) , 2 ) +pow( *( lattice_coords->ya + i ) , 2. ) + pow( *( lattice_coords->za + i ) , 2.0 ) ) ) ) ;

	  *( delta_ext + i ) = - hbo * *( v_ext + i ) ;

	}

	

    }

  if( iext == 3 )  /* Constant pairing */
      
    for( i = 0 ; i < n ; i++ )

      *( delta_ext + i ) = hbo + 0.0 * I ;

  if( iext == 4 || iext == 5 )

    for( i = 0 ; i < n ; i++)

      {

	r1 = sqrt( pow( *( lattice_coords->xa + i ) , 2 ) + pow( *( lattice_coords->ya + i ) , 2. ) + pow( *( lattice_coords->za + i ) + 7.5 , 2.0 ) ) ;

	r2 = sqrt( pow( *( lattice_coords->xa + i ) , 2 ) + pow( *( lattice_coords->ya + i ) , 2. ) + pow( *( lattice_coords->za + i ) - 7.5 , 2.0 ) ) ;
      
	*( v_ext + i ) = -50.0 / ( 1.0 + exp( -2.0 ) * cosh( r1 ) ) - 50.0 / ( 1.0 + exp( -2.0 ) * cosh( r2 ) ) ;

	if( iext == 5 )

	  *( delta_ext + i ) = 0.05 * *( v_ext + i ) + 0. * I ;

      }

  if( iext == 6 )

    {

      hw = 0.25 * hbo * hbo / hbar2m ;
      
      for( i = 0 ; i < n ; i++ )

	{

	  *( v_ext + i ) = hw * ( pow( * ( lattice_coords->xa + i ) , 2. ) + pow( * ( lattice_coords->ya + i ) , 2. ) + pow( * ( lattice_coords->za + i ) , 2. ) ) ;

	  *( delta_ext + i ) = 0.1 * hbo ;

	}

    }

  if( iext == 7 )  /* HO */

    {

      hw = 0.25 * hbo * hbo / hbar2m ;
      
      for( i = 0 ; i < n ; i++ )

	*( v_ext + i ) = hw * ( pow( * ( lattice_coords->xa + i ) , 2. ) + pow( * ( lattice_coords->ya + i ) , 2. ) + .5 * pow( * ( lattice_coords->za + i ) , 2. ) ) ;

    }

  if( iext == 8 )  /* HO */

    {

      hw = 0.25 * hbo * hbo / hbar2m ;
      
      for( i = 0 ; i < n ; i++ )

	{

	  if( fabs( lattice_coords->xa[ i ] ) > rr )

	    v_ext[ i ] = hw * pow( fabs( lattice_coords->xa[ i ] ) - rr , 2. ) ;

	  if( fabs( lattice_coords->ya[ i ] ) > rr )

	    v_ext[ i ] += hw * pow( fabs( lattice_coords->ya[ i ] ) - rr , 2. ) ;

          if( fabs( lattice_coords->za[ i ] ) > rr )

	    v_ext[ i ] += .5 * hw * pow( fabs( lattice_coords->za[ i ] ) - rr , 2. ) ;

	}

    }

  if( iext == 9 ){
    
    hw = ( double ) n_part ;

    hw = 1.2 * pow( hw , 1./3. ) ;

    for( i = 0 ; i < n ; i++){

      *( delta_ext + i ) = hbo / ( 1. + exp( ( sqrt( pow( *( lattice_coords->xa + i ) , 2 ) +pow( *( lattice_coords->ya + i ) , 2. ) + pow( *( lattice_coords->za + i ) , 2.0 ) ) - hw ) / 2. ) )  ;

    }

  }

  if( iext == 60 ){
    double aa=0.65;
    double w=wneck;
    double amp=(1.+exp((z0-w)/aa))*(1.+exp(-(z0+w)/aa));
    double kk=1.+exp(-rneck/aa);
    for(i=0; i<n;i++){
      double rho=sqrt(lattice_coords->xa[i]*lattice_coords->xa[i]+lattice_coords->ya[i]*lattice_coords->ya[i]);
      double z=lattice_coords->za[i]-z0;
      v_ext[i]=amp*v0*(1.-kk/((1.+exp((rho-rneck)/aa))))/((1.+exp(-(z+w)/aa))*(1+exp((z-w)/aa)));
    }
  }

  if( iext == 61 ){
    for(i=0; i<n;i++){
      double z=lattice_coords->za[i];

      v_ext[i]=v0*z*z*z;
    }
  }


}

void external_so_m( double * v_ext , double * mass_eff , double * wx , double * wy , double * wz , const double hbar2m , Lattice_arrays * lattice_coords , const MPI_Comm gr_comm , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz )

{

  int i ;

  int nxyz ;

  double mass = 939.565346 ;

  double hbarc = 197.3269631 ;

  double lambda ;

  nxyz = nx * ny * nz ;

  lambda = 2.5 * pow( .5 * hbarc / mass , 2. ) ; /* half the strength */

  gradient_real( v_ext , nxyz , wx , wy , wz , 0 , nxyz , gr_comm , d1_x , d1_y , d1_z , nx , ny , nz , 0 ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      *( wx + i ) = lambda * *( wx + i ) ;

      *( wy + i ) = lambda * *( wy + i ) ;

      *( wz + i ) = lambda * *( wz + i ) ;

    }

}

// make filter for constraining field
void make_filter(double *filter, Lattice_arrays *latt_coords, int nxyz)
{
  int i;
  double xa, ya, za, rr;
  double R0, aa, beta;

  beta = 1.5; aa = 1.; R0 = 10.;
  for(i=0;i<nxyz;i++)
    {
      xa = latt_coords->xa[i];
      ya = latt_coords->ya[i];
      za = latt_coords->za[i];
      rr = sqrt(pow(xa,2.0)+pow(ya,2.0)+pow(za/beta,2.0));
      
      filter[i] = 1./(1. + exp( (rr - R0)/aa));
    }

}
