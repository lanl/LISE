// for license information, see the accompanying LICENSE file

/* 

compute the deformation parameters beta and gamma

gamma not computed yet

*/

#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include "vars_nuclear.h"

extern double pi ;

double center_dist( double * , const int , Lattice_arrays * , double * , double * , double * ) ;

void deform( double * rho_p , double * rho_n , int nxyz , Lattice_arrays * latt_coords  , double dxyz , FILE * fout )

{
  double q[ 3 ][ 3 ] , * rho , beta , r2 , xcm , ycm , zcm , num ;

  double xa , ya , za ;

  double q30;

  int i , j ;

  rho = malloc( nxyz * sizeof( double ) ) ;

  for( i = 0 ; i < nxyz ; ++i )

    rho[ i ] = rho_p[ i ] + rho_n[ i ] ;

  for( i = 0 ; i < 3 ; ++i )

    for( j = i ; j < 3 ; ++j )

      q[ i ][ j ] = 0. ;

  num = center_dist( rho , nxyz , latt_coords , &xcm , &ycm , &zcm ) ;

  r2 = 0. ; pi = acos(-1.0);

  for( i = 0 ; i < nxyz ; ++i )

    {

      xa = latt_coords->xa[ i ] ;
      ya = latt_coords->ya[ i ] ;
      za = latt_coords->za[ i ] ;

      double _q30 = za*(2*za*za-3*xa*xa-3*ya*ya)*sqrt(7/pi)/4.0;

      xa = xa*xa - 2.*xa*latt_coords->wx[i]*xcm + xcm*xcm;
      ya = ya*ya - 2.*ya*latt_coords->wy[i]*ycm + ycm*ycm;
      za = za*za - 2.*za*latt_coords->wz[i]*zcm + zcm*zcm;

      double xa1 = latt_coords->wx[i]*latt_coords->xa[ i ] - xcm ;
      double ya1 = latt_coords->wy[i]*latt_coords->ya[ i ] - ycm ;
      double za1 = latt_coords->wz[i]*latt_coords->za[ i ] - zcm ;

      r2 += ( xa + ya + za ) * rho[ i ] ;

      q[ 0 ][ 0 ] += ( 2. * xa - ya - za ) * rho[ i ] ;

      q[ 1 ][ 1 ] += ( - xa + 2. * ya - za ) * rho[ i ] ;

      q[ 2 ][ 2 ] += ( - xa - ya + 2. * za ) * rho[ i ] ;

      q[ 0 ][ 1 ] += ( 3. * xa1 * ya1 * rho[ i ] ) ;

      q[ 0 ][ 2 ] += ( 3. * xa1 * za1 * rho[ i ] ) ;

      q[ 1 ][ 2 ] += ( 3. * ya1 * za1 * rho[ i ] ) ;

      q30 += _q30*rho[i];

      

    }

  free( rho ) ;

  q[ 1 ][ 0 ] = q[ 0 ][ 1 ] ;

  q[ 2 ][ 0 ] = q[ 0 ][ 2 ] ;

  q[ 2 ][ 1 ] = q[ 1 ][ 2 ] ;

  printf("qxx = %g fm^2  qyy = %g fm^2   qzz = %g fm^2\n" , q[ 0 ][ 0 ] * dxyz , q[ 1 ][ 1 ] * dxyz , q[ 2 ][ 2 ] * dxyz ) ;

  printf("xcm = %g fm  ycm = %g fm   zcm = %g fm \n\n" , xcm , ycm , zcm ) ;

  printf("q30 = %g fm^3\n" , q30*dxyz);

  fprintf( fout , "qxx = %g   qyy = %g    qzz = %g \n" , q[ 0 ][ 0 ] * dxyz , q[ 1 ][ 1 ] * dxyz , q[ 2 ][ 2 ] * dxyz ) ;

  fprintf( fout , "xcm = %g fm  ycm = %g fm   zcm = %g fm \n\n" , xcm , ycm , zcm ) ;

  fprintf(fout, "q30 = %g b^3\n" , q30*dxyz/1000.0);

  beta = 0. ;

  for( i = 0 ; i < 3 ; ++i )

    for( j = 0 ; j < 3 ; ++j )

      beta += q[ i ][ j ] * q[ j ][ i ] ;

  beta = 5.0 * sqrt( beta / 216.0 ) / r2 ;

  printf( " beta= %g \n" , beta ) ;

  fprintf( fout , " beta= %g \n" , beta ) ;

  beta = sqrt( pi / 5. ) * q[ 2 ][ 2 ] / r2 ;

  printf( " beta_2 = %9.6f \n" , beta ) ;

  fprintf( fout , " beta_2 = %9.6f \n" , beta ) ;

}

void i2xyz( const int i , int * ix , int * iy , int * iz , const int ny , const int nz );

double q2av( double * rho_p , double * rho_n , double * qzz , const int nxyz, const double np, const double nn ){

  double sum=0.;

  int i;

  double sum_p=0.,sum_n=0.;

  for(i=0;i<nxyz;i++){

    sum_p += rho_p[i];
    sum_n += rho_n[i];

  }  

  sum = 0.;

  for(i=0;i<nxyz;i++)

    sum += (rho_p[i]+rho_n[i])*qzz[i];

  return( sum );

}


double cm_coord( double * rho_p , double * rho_n , double * qzz , const int nxyz, const double np, const double nn ){

  double sum=0.;

  int i;

  double sum_p=0.,sum_n=0.;

  for(i=0;i<nxyz;i++){

    sum_p += rho_p[i];
    sum_n += rho_n[i];

  }  

  sum_p /= np ;
  sum_n /= nn;

  sum = 0.;

  for(i=0;i<nxyz;i++)

    sum += (rho_p[i]/sum_p+rho_n[i]/sum_n)*qzz[i];

  return( sum  / ( nn+np) );

}
