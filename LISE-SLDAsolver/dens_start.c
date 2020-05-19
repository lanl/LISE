// for license information, see the accompanying LICENSE file

#include <stdlib.h>

#include <stdio.h>

#include "vars_nuclear.h"

#include <math.h>

void dens_startTheta( const double A_mass , const double npart , const int nxyz , Densities * dens ,  Lattice_arrays * lattice_coords , const double dxyz , const double lx , const double ly , const double lz , const int ideform )

{
  /* one option to start the code with a normal/anomalous spherical density */

  int i ;

  double r0 , a = 1.8;

  r0 = 1.12*pow( A_mass , 0.33333334 );

  double rz = r0;

  double rx , ry ;



  switch ( ideform ) {

  case 0: 

    rx = r0;

    ry = r0 ;

    break;

  case 2:

    rx = 0.8 * r0;

    ry = rx ;

    break ;

  case -2:

    rx = 1.2 * r0;

    ry = rx ;

    break;

  case 3:

    rz = 1.2 * r0;

    ry = r0;

    rx = 0.8 * r0 ;

    break;

  default:

    printf ( "this option does not exist, switching to spherical densities\n" );

    rx = rz;

    ry = rz ;

    break;

  }

  for( i = 0 ; i < nxyz ; i++ )

    {

      double xa = ( fabs(lattice_coords->xa[ i ]) - rx ) / a ;

      double ya = ( fabs(lattice_coords->ya[ i ]) - ry ) / a ;

      double za = ( fabs(lattice_coords->za[ i ]) - rz ) / a ;

      dens->rho[ i ] = 0.08 * .125 * ( 1. - erf( xa ) ) * ( 1. - erf( ya ) ) * ( 1. -erf( za ) ) ;

    }

  for( i = dens->nstart ; i < dens->nstop ; i++ ){
    dens->nu[ i - dens->nstart ] = 0.02 * dens->rho[ i ] + 0.*I ;

  }

}





void dens_gauss( const int A_mass , const int npart , const int nxyz , Densities * dens ,  Lattice_arrays * lattice_coords , const double dxyz , const double lx , const double ly , const double lz , const int ideform )

{
  /* one option to start the code with a normal/anomalous spherical density */

  int i ;

  double r , r0 , r01 , r02 , a , apart , xa , ya , za ;

  r0 = lx / 5. /* 1. * pow( ( double ) A_mass , 1./3. ) */ ;

  if( ideform == 2 || ideform == 3 )

    {

      r01 = 1.4 * r0 ;

      if( ideform == 3 ) 

	r02 = 0.8 * r0 ;

      else

	r02 = r0 ;

    }

  else

    {

      r01 = r0 ;

      r02 = r0 ;

    }

  apart = 0. ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      xa = lattice_coords->xa[ i ] ;

      ya = lattice_coords->ya[ i ] ;

      za = lattice_coords->za[ i ] ;

      dens->rho[ i ] = 0.08 * ( exp( - pow(  xa / r0 , 2. ) ) + exp( - pow( ( xa - lx ) / r0 , 2. ) ) + exp( - pow( ( xa + lx ) / r0 , 2. ) ) + exp( - pow( ( xa - 2. * lx ) / r0 , 2. ) ) + exp( - pow( ( xa + 2. * lx ) / r0 , 2. ) ) + exp( - pow( ( xa - 3. * lx ) / r0 , 2. ) ) + exp( - pow( ( xa + 3. * lx ) / r0 , 2. ) ) ) 

	* ( exp( - pow(  ya / r02 , 2. ) ) + exp( - pow( ( ya - ly ) / r02 , 2. ) ) + exp( - pow( ( ya + ly ) / r02 , 2. ) ) + exp( - pow( ( ya - 2. * ly ) / r02 , 2. ) ) + exp( - pow( ( ya + 2. * ly ) / r02 , 2. ) ) + exp( - pow( ( ya - 3. * ly ) / r02 , 2. ) ) + exp( - pow( ( ya + 3. * ly ) / r02 , 2. ) ) ) 

	* ( exp( - pow(  za / r01 , 2. ) ) + exp( - pow( ( za - lz ) / r01 , 2. ) ) + exp( - pow( ( za + lz ) / r01 , 2. ) ) + exp( - pow( ( za - 2. * lz ) / r01 , 2. ) ) + exp( - pow( ( za + 2. * lz ) / r01 , 2. ) )  + exp( - pow( ( za - 3. * lz ) / r01 , 2. ) ) + exp( - pow( ( za + 3. * lz ) / r01 , 2. ) ) ) ; 

      apart += dens->rho[ i ] ;

    }

  apart = apart * dxyz / ( (double) npart ) ;

  for( i=0; i < nxyz ; ++i){
    dens->rho[ i ] *= 0.08 ;
  }

  for( i = dens->nstart ; i < dens->nstop ; i++ ){
    dens->nu[ i - dens->nstart ] = 0.1 * dens->rho[ i ] + 0.*I ;
  }

}
