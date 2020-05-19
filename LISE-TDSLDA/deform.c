// for license information, see the accompanying LICENSE file

/* 

compute the deformation parameters beta and gamma

gamma not computed yet

*/

#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include "vars.h"

#include <mpi.h>

#include "tdslda_func.h"

#define PI 3.141592653589793238462643383279502884197

double deform( double * rho , int nxyz , Lattice_arrays * latt_coords  , double dxyz )

{
  double q[ 3 ][ 3 ] , beta , r2 , xcm , ycm , zcm , num ;

  double xa , ya , za ;

  int i , j ;

  for( i = 0 ; i < 3 ; ++i )

    for( j = i ; j < 3 ; ++j )

      q[ i ][ j ] = 0. ;

  num = center_dist( rho , nxyz , latt_coords , &xcm , &ycm , &zcm ) ;

  r2 = 0. ;

  for( i = 0 ; i < nxyz ; ++i )

    {

      xa = latt_coords->xa[ i ] - xcm ;

      ya = latt_coords->ya[ i ] - ycm ;

      za = latt_coords->za[ i ] - zcm ;

      r2 += ( xa * xa + ya * ya + za * za ) * rho[ i ] ;

      q[ 2 ][ 2 ] += ( ( - xa * xa - ya * ya + 2. * za * za ) * rho[ i ] ) ;

    }


  beta = sqrt( PI / 5. ) * q[ 2 ][ 2 ] / r2 ;

  return beta
 ;

}
