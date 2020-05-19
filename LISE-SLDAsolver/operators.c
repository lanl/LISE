// for license information, see the accompanying LICENSE file

#include "vars_nuclear.h"

#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <complex.h>

#include <assert.h>

#include <mpi.h>

void i2xyz( const int i , int * ix , int * iy , int * iz , const int ny , const int nz );

void gradient_real( double * f , const int n , double * g_x , double * g_y , double * g_z , const int nstart , const int nstop , const MPI_Comm comm , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz , const int ired )

{

  /* 

  ired = 1: reduction

       = anything else : no reduction

  */

  int i , j ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  double * gr_x , * gr_y , * gr_z , sum ;

  int nx1 , ny1 , nz1 ;

  assert( gr_x = malloc( n * sizeof( double ) ) ) ;

  assert( gr_y = malloc( n * sizeof( double ) ) ) ;

  assert( gr_z = malloc( n * sizeof( double ) ) ) ;

  nx1 = nx - 1 ;

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

  for( i = 0 ; i < n ; i++ )

    {

      *( gr_x + i ) = 0.0 ;

      *( gr_y + i ) = 0.0 ;

      *( gr_z + i ) = 0.0 ;

    }

  for( i = nstart ; i < nstop ; i++ )

    {

      i2xyz( i , &ix1 , &iy1 , &iz1 , ny , nz ) ;

      sum = 0. ; 

#pragma omp parallel for default(shared) private(ix2) \
  reduction(+:sum) 

      for( ix2 = 0 ; ix2 < nx ; ix2++ )

	sum += creal( d1_x[ ix1 - ix2 + nx1 ] * f[ iz1 + nz * ( iy1 + ny * ix2 ) ] ) ;

      gr_x[ i ] += sum ;

      sum = 0. ;

#pragma omp parallel for default(shared) private(iz2) \
  reduction(+:sum) 

      for( iz2 = 0 ; iz2 < nz ; iz2++ )

	sum += creal( d1_z[ iz1 - iz2 + nz1 ] * f[ iz2 + nz * ( iy1 + ny * ix1 ) ] ) ;

      gr_z[ i ] += sum ;

      sum = 0. ;

#pragma omp parallel for default(shared) private(iy2) \
  reduction(+:sum) 

      for( iy2 = 0 ; iy2 < ny ; iy2++ )

	sum += creal( d1_y[ iy1 - iy2 + ny1 ] * f[ iz1 + nz * ( iy2 + ny * ix1 ) ] ) ;

      gr_y[ i ] += sum ;

    }

  if( ired == 1 ) /* all reduce */

    {
      
      MPI_Allreduce( gr_x , g_x , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      MPI_Allreduce( gr_y , g_y , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      MPI_Allreduce( gr_z , g_z , n , MPI_DOUBLE , MPI_SUM , comm ) ;

    }

  else {

#pragma omp parallel for default(shared) private(i)

    for( i = 0 ; i < n ; i++ )

	{

	  *( g_x + i ) =  *( gr_x + i ) ; 

	  *( g_y + i ) =  *( gr_y + i ) ; 

	  *( g_z + i ) =  *( gr_z + i ) ; 

	}

  }

  free( gr_x ) ; free( gr_y ) ; free( gr_z ) ;

}

void gradient_real_orig( double * f , const int n , double * g_x , double * g_y , double * g_z , const int nstart , const int nstop , const MPI_Comm comm , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz , const int ired )

{

  /* 

  ired = 1: reduction

       = anything else : no reduction

  */

  int i , j ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  double * gr_x , * gr_y , * gr_z ;

  int nx1 , ny1 , nz1 ;

  assert( gr_x = malloc( n * sizeof( double ) ) ) ;

  assert( gr_y = malloc( n * sizeof( double ) ) ) ;

  assert( gr_z = malloc( n * sizeof( double ) ) ) ;

  nx1 = nx - 1 ;

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

  for( i = 0 ; i < n ; i++ )

    {

      *( gr_x + i ) = 0.0 ;

      *( gr_y + i ) = 0.0 ;

      *( gr_z + i ) = 0.0 ;

    }

  for( i = nstart ; i < nstop ; i++ )

    {

      i2xyz( i , &ix1 , &iy1 , &iz1 , ny , nz ) ;

      for( j = 0 ; j < n ; j++ )

	{

	  i2xyz( j , &ix2 , &iy2 , &iz2 , ny , nz ) ;

	  if( iy1 == iy2 && iz1 == iz2 )

	    gr_x[ i ] += creal( d1_x[ ix1 - ix2 + nx1 ] * f[ j ] ) ;

	  if( iy1 == iy2 && ix1 == ix2 )

	    gr_z[ i ] += creal( d1_z[ iz1 - iz2 + nz1 ] * f[ j ] ) ;

	  if( ix1 == ix2 && iz1 == iz2 )

	    gr_y[ i ] += creal( d1_y[ iy1 - iy2 + ny1 ] * f[ j ] ) ;

	}

    }

  if( ired == 1 ) /* all reduce */

    {
      
      MPI_Allreduce( gr_x , g_x , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      MPI_Allreduce( gr_y , g_y , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      MPI_Allreduce( gr_z , g_z , n , MPI_DOUBLE , MPI_SUM , comm ) ;

    }

  else

    for( i = 0 ; i < n ; i++ )

	{

	  *( g_x + i ) =  *( gr_x + i ) ; 

	  *( g_y + i ) =  *( gr_y + i ) ; 

	  *( g_z + i ) =  *( gr_z + i ) ; 

	}

  free( gr_x ) ; free( gr_y ) ; free( gr_z ) ;

}

void gradient( double complex * f , const int n , double complex * g_x , double complex * g_y , double complex * g_z , const int nstart , const int nstop , const MPI_Comm comm , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz , const int ired )

{

  int i , j ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  int nx1 , ny1 , nz1 ;

  double complex * gx , * gy , * gz , sum ;

  nx1 = nx - 1 ;

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

#pragma omp parallel for default(shared) private(i)

  for( i = 0 ; i < n ; i++ )

    {

      *( g_x + i ) = 0.0 + 0. * I ;

      *( g_y + i ) = 0.0 + 0. * I ;

      *( g_z + i ) = 0.0 + 0. * I ;

    }

  for( i = nstart ; i < nstop ; i++ )

    {

      i2xyz( i , &ix1 , &iy1 , &iz1 , ny , nz ) ;

      sum = 0. + I * 0. ;

#pragma omp parallel for default(shared) private(ix2) \
  reduction(+:sum) 

      for( ix2 = 0 ; ix2 < nx ; ix2++ )

	sum += d1_x[ ix1 - ix2 + nx1 ] * f[ iz1 + nz * ( iy1 + ny * ix2 ) ] ;

      g_x[ i ] += sum ;

      sum = 0. + 0. * I ;

#pragma omp parallel for default(shared) private(iz2) \
  reduction(+:sum) 

      for( iz2 = 0 ; iz2 < nz ; iz2++ )

	sum += d1_z[ iz1 - iz2 + nz1 ] * f[ iz2 + nz * ( iy1 + ny * ix1 ) ] ;

      g_z[ i ] += sum ;

      sum = 0. + 0. * I ;

#pragma omp parallel for default(shared) private(iy2) \
  reduction(+:sum) 

      for( iy2 = 0 ; iy2 < ny ; iy2++ )

	sum += d1_y[ iy1 - iy2 + ny1 ] * f[ iz1 + nz * ( iy2 + ny * ix1 ) ] ;

      g_y[ i ] += sum ;

    }

  if( ired == 1 ) 

    {

      assert( gx = malloc( n * sizeof( double ) ) ) ;

      assert( gy = malloc( n * sizeof( double ) ) ) ;

      assert( gz = malloc( n * sizeof( double ) ) ) ;

      for( i = 0 ; i < n ; i++ )

	{

	  gx[ i ] = g_x[ i ] ;

	  gy[ i ] = g_y[ i ] ;

	  gz[ i ] = g_z[ i ] ;

	}

      MPI_Allreduce( gx , g_x , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      MPI_Allreduce( gy , g_y , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      MPI_Allreduce( gz , g_z , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      free( gx ) ; free( gy ) ; free( gz ) ;

    }

}

void gradient_ud( double complex * f , const int n , double complex * g_x , double complex * g_y , double complex * g_z , double complex * g_xd , double complex * g_yd , double complex * g_zd , const int nstart , const int nstop , const MPI_Comm comm , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz )

{

  int i , j ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  int nx1 , ny1 , nz1 ;

  double complex sum1 , sum2 ;

  nx1 = nx - 1 ;

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

  for( i = 0 ; i < nstop - nstart ; i++ )

    {

      g_x[ i ] = 0.0 + 0. * I ;

      g_y[ i ] = 0.0 + 0. * I ;

      g_z[ i ] = 0.0 + 0. * I ;

      g_xd[ i ] = 0.0 + 0. * I ;

      g_yd[ i ] = 0.0 + 0. * I ;

      g_zd[ i ] = 0.0 + 0. * I ;

    }

  for( i = 0 ; i < nstop - nstart ; i++ )

    {

      i2xyz( i + nstart , &ix1 , &iy1 , &iz1 , ny , nz ) ;

      sum1 = 0. + I * 0. ;

      sum2 = 0. + I * 0. ;

#pragma omp parallel for default(shared) private(ix2) \
  reduction(+:sum1,sum2) 

      for( ix2 = 0 ; ix2 < nx ; ix2++ )

	{

	  sum1 += d1_x[ ix1 - ix2 + nx1 ] * f[ iz1 + nz * ( iy1 + ny * ix2 ) ] ;

	  sum2 += d1_x[ ix1 - ix2 + nx1 ] * f[ iz1 + nz * ( iy1 + ny * ix2 ) + n ] ;

	}

      g_x[ i ] += sum1 ;

      g_xd[ i ] += sum2 ;

      sum1 = 0. + I * 0. ;

      sum2 = 0. + I * 0. ;

#pragma omp parallel for default(shared) private(iz2) \
  reduction(+:sum1,sum2) 

      for( iz2 = 0 ; iz2 < nz ; iz2++ )

	{

	  sum1 += d1_z[ iz1 - iz2 + nz1 ] * f[ iz2 + nz * ( iy1 + ny * ix1 ) ] ;
	  
	  sum2 += d1_z[ iz1 - iz2 + nz1 ] * f[ iz2 + nz * ( iy1 + ny * ix1 ) + n ] ;

	}

      g_z[ i ] += sum1 ;

      g_zd[ i ] += sum2 ;

      sum1 = 0. + I * 0. ;

      sum2 = 0. + I * 0. ;

#pragma omp parallel for default(shared) private(iy2) \
  reduction(+:sum1,sum2) 

      for( iy2 = 0 ; iy2 < ny ; iy2++ )

	{

	  sum1 += d1_y[ iy1 - iy2 + ny1 ] * f[ iz1 + nz * ( iy2 + ny * ix1 ) ] ;

	  sum2 += d1_y[ iy1 - iy2 + ny1 ] * f[ iz1 + nz * ( iy2 + ny * ix1 ) + n ] ;

	}

      g_y[ i ] += sum1 ;

      g_yd[ i ] += sum2 ;

    }

}

void gradient_orig( double complex * f , const int n , double complex * g_x , double complex * g_y , double complex * g_z , const int nstart , const int nstop , const MPI_Comm comm , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz , const int ired )

{

  int i , j ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  int nx1 , ny1 , nz1 ;

  double complex * gx , * gy , * gz ;

  nx1 = nx - 1 ;

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

  for( i = 0 ; i < n ; i++ )

    {

      *( g_x + i ) = 0.0 + 0. * I ;

      *( g_y + i ) = 0.0 + 0. * I ;

      *( g_z + i ) = 0.0 + 0. * I ;

    }

  for( i = nstart ; i < nstop ; i++ )

    {

      i2xyz( i , &ix1 , &iy1 , &iz1 , ny , nz ) ;

      for( j = 0 ; j < n ; j++ )

	{

	  i2xyz( j , &ix2 , &iy2 , &iz2 , ny , nz ) ;

	  if( iy1 == iy2 && iz1 == iz2 )

	    g_x[ i ] += d1_x[ ix1 - ix2 + nx1 ] * f[ j ] ;

	  if( iy1 == iy2 && ix1 == ix2 )

	    g_z[ i ] += d1_z[ iz1 - iz2 + nz1 ] * f[ j ] ;

	  if( ix1 == ix2 && iz1 == iz2 )

	    g_y[ i ] += d1_y[ iy1 - iy2 + ny1 ] * f[ j ] ;

	}

    }

  if( ired == 1 ) 

    {

      assert( gx = malloc( n * sizeof( double ) ) ) ;

      assert( gy = malloc( n * sizeof( double ) ) ) ;

      assert( gz = malloc( n * sizeof( double ) ) ) ;

      for( i = 0 ; i < n ; i++ )

	{

	  gx[ i ] = g_x[ i ] ;

	  gy[ i ] = g_y[ i ] ;

	  gz[ i ] = g_z[ i ] ;

	}

      MPI_Allreduce( gx , g_x , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      MPI_Allreduce( gy , g_y , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      MPI_Allreduce( gz , g_z , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      free( gx ) ; free( gy ) ; free( gz ) ;

    }

}

void laplacean_complex( double complex * f , const int n , double complex * lapf , const int nstart , const int nstop , const MPI_Comm comm , double * k1d_x , double * k1d_y , double * k1d_z , const int nx , const int ny , const int nz , const int ired )

{

  int i , j ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  double  complex * tmp; 

  for( i = 0 ; i < n ; i++ )

      *( lapf + i ) = 0.0 ;

  for( i = nstart ; i < nstop ; i++ )

    {

      i2xyz( i , &ix1 , &iy1 , &iz1 , ny , nz ) ;

      for( ix2 = 0 ; ix2 < nx ; ix2++)

	lapf[ i ] -= ( k1d_x[ abs( ix1 - ix2 ) ] * f[ iz1 + nz * ( iy1 + ny * ix2 ) ] ) ;

      for( iz2 = 0 ; iz2 < nz ; iz2++)

	lapf[ i ] -= ( k1d_z[ abs( iz1 - iz2 ) ] * f[ iz2 + nz * ( iy1 + ny * ix1 ) ] ) ;

      for( iy2 = 0 ; iy2 < ny ; iy2++)

	lapf[ i ] -= ( k1d_y[ abs( iy1 - iy2 ) ] * f[ iz1 + nz * ( iy2 + ny * ix1 ) ] ) ;

    }

  if( ired == 1 )

    {

      assert( tmp = malloc( n * sizeof( double complex) ) ) ;
 
      for( i = 0 ; i < n ; i++ )

	*( tmp + i ) = *( lapf + i ) ;

      MPI_Allreduce( tmp , lapf , n , MPI_DOUBLE_COMPLEX , MPI_SUM , comm ) ; 

      free( tmp ) ;

    }

}




void laplacean( double * f , const int n , double * lapf , const int nstart , const int nstop , const MPI_Comm comm , double * k1d_x , double * k1d_y , double * k1d_z , const int nx , const int ny , const int nz , const int ired )

{

  int i , j ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  double * tmp , sum ;

  for( i = 0 ; i < n ; i++ )

      *( lapf + i ) = 0.0 ;

  for( i = nstart ; i < nstop ; i++ )

    {

      i2xyz( i , &ix1 , &iy1 , &iz1 , ny , nz ) ;

      for( ix2 = 0 ; ix2 < nx ; ix2++)

	lapf[ i ] -= ( k1d_x[ abs( ix1 - ix2 ) ] * f[ iz1 + nz * ( iy1 + ny * ix2 ) ] ) ;

      for( iz2 = 0 ; iz2 < nz ; iz2++)

	lapf[ i ] -= ( k1d_z[ abs( iz1 - iz2 ) ] * f[ iz2 + nz * ( iy1 + ny * ix1 ) ] ) ;

      for( iy2 = 0 ; iy2 < ny ; iy2++)

	lapf[ i ] -= ( k1d_y[ abs( iy1 - iy2 ) ] * f[ iz1 + nz * ( iy2 + ny * ix1 ) ] ) ;

    }

  if( ired == 1 )

    {

      assert( tmp = malloc( n * sizeof( double ) ) ) ;

      for( i = 0 ; i < n ; i++ )

	*( tmp + i ) = *( lapf + i ) ;

      MPI_Allreduce( tmp , lapf , n , MPI_DOUBLE , MPI_SUM , comm ) ;

      free( tmp ) ;

    }

}

void diverg( double * fx , double * fy , double * fz , double * divf , const int n , const int nstart , const int nstop , const MPI_Comm comm , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz )

{

  int i , j ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  int nx1 , ny1 , nz1 ;

  double * divf_r ;

  nx1 = nx - 1 ;

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

  assert( divf_r = malloc( n * sizeof( double ) ) ) ;

  for( i = 0 ; i < n ; i++ )

      *( divf_r + i ) = 0.0 ;

  for( i = 0 ; i < n ; i++ )

    {

      i2xyz( i , &ix1 , &iy1 , &iz1 , ny , nz ) ;

      for( j = nstart ; j < nstop ; j++ )

	{

	  i2xyz( j , &ix2 , &iy2 , &iz2 , ny , nz ) ;

	  if( iy1 == iy2 && iz1 == iz2 )
	    
	    divf_r[ i ] += creal( d1_x[ ix1 - ix2 + nx1 ] * fx[ j ] ) ;

	  if( iy1 == iy2 && ix1 == ix2 )

	    divf_r[ i ] += creal( d1_z[ iz1 - iz2 + nz1 ] * fz[ j ] ) ;

	  if( ix1 == ix2 && iz1 == iz2 )
	    
	    divf_r[ i ] += creal( d1_y[ iy1 - iy2 + ny1 ] * fy[ j ] ) ;

	}

    }

  MPI_Allreduce( divf_r , divf , n , MPI_DOUBLE , MPI_SUM , comm ) ;

  free( divf_r ) ;

}

void  match_lattices( Lattice_arrays *latt , Lattice_arrays * latt3 , const int nx , const int ny , const int nz , const int nx3 , const int ny3 , const int nz3 , FFtransf_vars * fftrans , const double Lc ) {

  int i , ix , iy , iz , ix3 , iy3 , iz3 ;

  int nxyz = nx * ny * nz , nxyz3 = fftrans->nxyz3 ;

  double fpi ,xx ;

  fpi = 4. * acos( -1. ) * 197.3269631 / 137.035999679 ;  /* 4 * pi * e2 */

  assert( fftrans->i_l2s = malloc( nxyz3 * sizeof( int ) ) ) ;

  assert( fftrans->fc = malloc( nxyz3 * sizeof( double ) ) ) ;

  assert( fftrans->i_s2l = malloc( nxyz * sizeof( int ) ) ) ;

  for( i = 0 ; i < nxyz3 ; i++ )

    fftrans->i_l2s[ i ] = -1 ;

  for( ix3 = 0 ; ix3 < nx ; ix3++ ){

    ix=ix3;

    for( iy3 = 0 ; iy3 < ny ; iy3++ ){

      iy=iy3;

      for( iz3 = 0 ; iz3 < nz ; iz3++ ){

	iz=iz3;

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

void  match_lattices_orig( Lattice_arrays *latt , Lattice_arrays * latt3 , const int nx , const int ny , const int nz , const int nx3 , const int ny3 , const int nz3 , FFtransf_vars * fftrans , const double Lc ) {

  int i , ix , iy , iz , ix3 , iy3 , iz3 ;

  int nxyz = nx * ny * nz , nxyz3 = fftrans->nxyz3 ;

  double fpi , sqrt3 ;

  sqrt3 = sqrt( 3. ) ;

  fpi = 4. * acos( -1. ) * 197.3269631 / 137.035999679 ;  /* 4 * pi * e2 */

  assert( fftrans->i_l2s = malloc( nxyz3 * sizeof( int ) ) ) ;

  assert( fftrans->fc = malloc( nxyz3 * sizeof( double ) ) ) ;

  assert( fftrans->i_s2l = malloc( nxyz * sizeof( int ) ) ) ;

  for( i = 0 ; i < nxyz3 ; i++ )

    fftrans->i_l2s[ i ] = -1 ;

  for( ix3 = nx ; ix3 < 2 * nx ; ix3++ ){

    ix = ix3 - nx ;

    for( iy3 = ny ; iy3 < 2 * ny ; iy3++ ){

      iy = iy3 - ny ;

      for( iz3 = nz ; iz3 < 2 * nz ; iz3++ ){

	iz = iz3 - nz ;

	fftrans->i_s2l[ iz + nz * ( iy + ny * ix ) ] = iz3 + nz3 * ( iy3 + ny3 * ix3 ) ;

	fftrans->i_l2s[ iz3 + nz3 * ( iy3 + ny3 * ix3 ) ] = iz + nz * ( iy + ny * ix ) ;

      }

    }

  }

  fftrans->fc[ 0 ] = fpi * 1.5 * Lc * Lc ;

  for( i = 1 ; i < nxyz3 ; i++ )

    fftrans->fc[ i ] = fpi * ( 1. - cos( sqrt( latt3->kin[ i ] ) * sqrt3 * Lc ) ) / latt3->kin[ i ] ;

  free( latt3->kx ) ; free( latt3->ky ) ; free( latt3->kz ) ; free( latt3->xa ) ; free( latt3->ya ) ; free( latt3->za ) ; free( latt3->kin ) ;

}

