// for license information, see the accompanying LICENSE file

#include "vars.h"

#include <assert.h>

#include <math.h>

#include <stdlib.h>

#include <stdio.h>

void momenta_copy(double kx_min,double ky_min,double kz_min);

void grid3( double * xx , double * yy , double * zz , const int nx , const int ny , const int nz , double * X , double * Y , double * Z )

{

  int ix , iy , iz , ixyz ;

  for( ix = 0 ; ix < nx ; ix++)

    for( iy = 0 ; iy < ny ; iy++ )

      for ( iz = 0 ; iz < nz ; iz++ )

	{

	  ixyz = iz + nz * ( iy + ny * ix ) ;

	  X[ ixyz ] = xx[ ix ] ;

	  Y[ ixyz ] = yy[ iy ] ;

	  Z[ ixyz ] = zz[ iz ] ;

	}

}

void make_coordinates( const int nxyz , const int nx , const int ny , const int nz , const double dx , const double dy , const double dz , Lattice_arrays * lattice_vars )

{

  double * xx , * yy , * zz ;

  double xn ;

  int i ;

  assert( lattice_vars-> xa = malloc( nxyz * sizeof( double ) ) ) ; 

  assert( lattice_vars-> ya = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> za = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> kx = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> ky = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> kz = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> kin = malloc( nxyz * sizeof( double ) ) ) ;

  assert( xx = malloc( nx * sizeof( double ) ) ) ;

  assert( yy = malloc( ny * sizeof( double ) ) ) ;

  assert( zz = malloc( nz * sizeof( double ) ) ) ;

  xn = ( ( double ) nx ) / 2.0 ;

  for( i = 0 ; i < nx ; i++ )

    xx[ i ] = ( ( double)  i - xn ) * dx ;

  xn = ( ( double ) ny ) / 2.0 ;

  for( i = 0 ; i < ny ; i++ )

    yy[ i ] = ( ( double ) i - xn ) * dy ;

  xn = ( ( double ) nz ) / 2.0 ;

  for( i = 0 ; i < nz ; i++ )

    zz[ i ] = ( ( double ) i - xn ) * dz ;

  grid3( xx , yy , zz , nx , ny , nz , lattice_vars->xa , lattice_vars->ya , lattice_vars->za ) ;

  /* set also the momentum space */

  xn = acos( -1. ) / ( ( double ) nx * dx ) ;

  for( i = 0 ; i < nx / 2 ; i++ )

    {

      *( xx + i ) = ( double) ( 2 * i ) * xn ;

      *( xx + i + nx / 2 ) = ( double ) ( 2 * i - nx ) * xn ;

    }

  lattice_vars->k1x = xx[ 1 ] ;

  xn = acos( -1. ) / ( ( double ) ny * dy ) ;

  for( i = 0 ; i < ny / 2 ; i++ )

    {

      *( yy + i ) = ( double) ( 2 * i ) * xn ;

      *( yy + i + ny / 2 ) = ( double ) ( 2 * i - ny ) * xn ;

    }

  lattice_vars->k1y = yy[ 1 ] ;

  xn = acos( -1. ) / ( ( double ) nz * dz ) ;

  for( i = 0 ; i < nz / 2 ; i++ )

    {

      *( zz + i ) = ( double) ( 2 * i ) * xn ;

      *( zz + i + nz / 2 ) = ( double ) ( 2 * i - nz ) * xn ;

    }

  lattice_vars->k1z = zz[ 1 ] ;

  momenta_copy(xx[1],yy[1],zz[1]) ;

  grid3( xx , yy , zz , nx , ny , nz , lattice_vars->kx , lattice_vars->ky , lattice_vars->kz ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      lattice_vars->kin[ i ] = pow( lattice_vars->kx[ i ] , 2. ) + pow( lattice_vars->ky[ i ] , 2. ) + pow( lattice_vars->kz[ i ] , 2. ) ;

    }


  xx[ nx / 2 ] = 0. ;

  yy[ ny / 2 ] = 0. ;

  zz[ nz / 2 ] = 0. ;

  grid3( xx , yy , zz , nx , ny , nz , lattice_vars->kx , lattice_vars->ky , lattice_vars->kz ) ;

  free( xx ) ;

  free( yy ) ;

  free( zz ) ;

}

void gradient( double complex * f , double complex * fx , double complex * fy , double complex * fz , FFtransf_vars * fftrans , Lattice_arrays * latt , const int nxyz , double complex * buff )

{

  int i ;

  double complex aix = I / ( ( double ) nxyz ) ;

  for( i = 0 ; i < nxyz ; i++ )

    fftrans->buff[ i ] = f[ i ] ;

  fftw_execute( fftrans->plan_f ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      buff[ i ] = fftrans->buff[ i ] ;

      fftrans->buff[ i ] *= latt->kx[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      fx[ i ] = aix * fftrans->buff[ i ] ;

      fftrans->buff[ i ] = latt->ky[ i ] * buff[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      fy[ i ] = aix * fftrans->buff[ i ] ;

      fftrans->buff[ i ] = latt->kz[ i ] * buff[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      fz[ i ] = aix * fftrans->buff[ i ] ;

    }

}

void gradient_real( double * f , double * fx , double * fy , double * fz , FFtransf_vars * fftrans , Lattice_arrays * latt , const int nxyz )

{

  int i ;

  double aix = 1. / ( ( double ) nxyz ) ;

  double complex * buff ;

  assert( buff = malloc( nxyz * sizeof( double complex ) ) ) ;

  for( i = 0 ; i < nxyz ; i++ )

    fftrans->buff[ i ] = f[ i ] + I * 0. ;

  fftw_execute( fftrans->plan_f ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      buff[ i ] = - aix * fftrans->buff[ i ] ;

      fftrans->buff[ i ] = buff[ i ] * latt->kx[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      fx[ i ] = cimag ( fftrans->buff[ i ] ) ;

      fftrans->buff[ i ] = latt->ky[ i ] * buff[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      fy[ i ] = cimag( fftrans->buff[ i ] ) ;

      fftrans->buff[ i ] = latt->kz[ i ] * buff[ i ] ;

    }

  free( buff ) ;

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      fz[ i ] = cimag ( fftrans->buff[ i ] ) ;

    }

}

void diverg_real( double * f , double * fx , double * fy , double * fz , FFtransf_vars * fftrans , Lattice_arrays * latt , const int nxyz )

{

  int i ;

  double aix = - 1. / ( ( double ) nxyz ) ;

  for( i = 0 ; i < nxyz ; i++ )

    fftrans->buff[ i ] = fx[ i ] + I * 0. ;

  fftw_execute( fftrans->plan_f ) ;

  for( i = 0 ; i < nxyz ; i++ )

      fftrans->buff[ i ] = fftrans->buff[ i ] * latt->kx[ i ] ;

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      f[ i ] = cimag ( fftrans->buff[ i ] ) ;

      fftrans->buff[ i ] = fy[ i ] + 0. * I ;

    }

  fftw_execute( fftrans->plan_f ) ;

  for( i = 0 ; i < nxyz ; i++ )

      fftrans->buff[ i ] = latt->ky[ i ] * fftrans->buff[ i ] ;

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++){

    f[ i ] += cimag( fftrans->buff[ i ] ) ;

    fftrans->buff[ i ] = fz[ i ] + 0. * I ;

  }

  fftw_execute( fftrans->plan_f ) ;

  for( i = 0 ; i < nxyz ; i++ )

      fftrans->buff[ i ] = latt->kz[ i ] * fftrans->buff[ i ] ;

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ ){

      f[ i ] += cimag ( fftrans->buff[ i ] ) ;

      f[ i ] *= aix ;

    }

}

void laplacean( double * f , double * lap_f , const int nxyz , FFtransf_vars * fftrans , Lattice_arrays * latt )

{

  int i ;

  double xn ;

  xn = 1. / ( ( double ) nxyz ) ;

  for( i = 0 ; i < nxyz ; i++ )

    fftrans->buff[ i ] = f[ i ] + I * 0. ;

  fftw_execute( fftrans->plan_f ) ;

  for( i = 0 ; i < nxyz ; i++ )

    fftrans->buff[ i ] *= latt->kin[ i ] ;

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    lap_f[ i ] = - creal ( xn * fftrans->buff[ i ] ) ;

}  

void laplacean_complex( double complex * f , double complex * lap_f , const int nxyz , FFtransf_vars * fftrans , Lattice_arrays * latt )

{

  int i ;

  double xn ;

  xn = 1. / ( ( double ) nxyz ) ;

  for( i = 0 ; i < nxyz ; i++ )

    fftrans->buff[ i ] = f[ i ] ;

  fftw_execute( fftrans->plan_f ) ;

  for( i = 0 ; i < nxyz ; i++ )

    fftrans->buff[ i ] *= latt->kin[ i ] ;

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    lap_f[ i ] = - xn * fftrans->buff[ i ] ;

}  

void curl( double * vx , double * vy , double * vz , double * cvx , double * cvy , double * cvz , const int nxyz , FFtransf_vars * fftrans , Lattice_arrays * latt )

{

  double complex * buff ;

  int i ;

  double xn ;

  xn = 1. / ( ( double ) nxyz ) ;

  assert( buff = malloc( nxyz * sizeof( double complex ) ) ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      fftrans->buff[ i ] = - xn * vx[ i ] + 0. * I ;

    }

  fftw_execute( fftrans->plan_f ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      buff[ i ] = fftrans->buff[ i ] ;

      fftrans->buff[ i ] *= latt->ky[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      cvz[ i ] = - cimag( fftrans->buff[ i ] ) ;

      fftrans->buff[ i ] = buff[ i ] * latt->kz[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      cvy[ i ] = cimag( fftrans->buff[ i ] ) ;

      fftrans->buff[ i ] = - xn * vy[ i ] + 0. * I ;

    }

  fftw_execute( fftrans->plan_f ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      buff[ i ] = fftrans->buff[ i ] ;

      fftrans->buff[ i ] *= latt->kx[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      cvz[ i ] += cimag( fftrans->buff[ i ] ) ;

      fftrans->buff[ i ] = buff[ i ] * latt->kz[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      cvx[ i ] = - cimag( fftrans->buff[ i ] ) ;

      fftrans->buff[ i ] = - xn * vz[ i ] + 0. * I ;

    }

  fftw_execute( fftrans->plan_f ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      buff[ i ] = fftrans->buff[ i ] ;

      fftrans->buff[ i ] *= latt->kx[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      cvy[ i ] -= cimag( fftrans->buff[ i ] ) ;

      fftrans->buff[ i ] = buff[ i ] * latt->ky[ i ] ;

    }

  fftw_execute( fftrans->plan_b ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      cvx[ i ] += cimag( fftrans->buff[ i ] ) ;

    }

  free( buff ) ;

}


void  match_lattices( Lattice_arrays *latt , Lattice_arrays * latt3 , const int nx , const int ny , const int nz , const int nx3 , const int ny3 , const int nz3 , FFtransf_vars * fftrans , const double Lc ) {

  int i , ix , iy , iz , ix3 , iy3 , iz3 ;

  int nxyz = nx * ny * nz , nxyz3 = fftrans->nxyz3 ;

  double fpi , sqrt3 ,xx ;

  sqrt3 = sqrt( 3. ) ;

  fpi = 4. * acos( -1. ) * 197.3269631 / 137.035999679 ;  /* 4 * pi * e2 */

  assert( fftrans->i_l2s = malloc( nxyz3 * sizeof( int ) ) ) ;

  assert( fftrans->fc = malloc( nxyz3 * sizeof( double ) ) ) ;

  assert( fftrans->i_s2l = malloc( nxyz * sizeof( int ) ) ) ;

  for( i = 0 ; i < nxyz3 ; i++ )

    fftrans->i_l2s[ i ] = -1 ;

  for( ix3 = 0 ; ix3 < nx ; ix3++ ){

    ix = ix3 ;

    for( iy3 = 0 ; iy3 < ny ; iy3++ ){

      iy = iy3 ;

      for( iz3 = 0 ; iz3 < nz ; iz3++ ){

	iz = iz3 ;

	fftrans->i_s2l[ iz + nz * ( iy + ny * ix ) ] = iz3 + nz3 * ( iy3 + ny3 * ix3 ) ;

	fftrans->i_l2s[ iz3 + nz3 * ( iy3 + ny3 * ix3 ) ] = iz + nz * ( iy + ny * ix ) ;

      }

    }

  }

  fftrans->fc[ 0 ] = fpi * .5 * Lc * Lc ;

  for( i = 1 ; i < nxyz3 ; i++ )

    fftrans->fc[ i ] = fpi * ( 1. - cos( sqrt( latt3->kin[ i ] ) * Lc ) ) / latt3->kin[ i ] ;

  free( latt3->kx ) ; free( latt3->ky ) ; free( latt3->kz ) ; free( latt3->xa ) ; free( latt3->ya ) ; free( latt3->za ) ; free( latt3->kin ) ;

}

