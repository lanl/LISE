// for license information, see the accompanying LICENSE file

#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <assert.h>

double ddot_( int * , double * , int * , double * , int * ) ;

void daxpy_( int * , double * , double * , int * , double * , int * ) ;

double ddotII( int * , double * , int * , double * , int * ) ;

void daxpyII( int * , double * , double * , int * , double * , int * ) ;

int broyden_minB( double * v_in , double * v_in_old , double * f_old , double * b , double * v_out , const int n , int * it_out , const double alpha ) ;

int broyden_min( double * v_in , double * v_in_old , double * f_old , double * b_bra , double * b_ket , double * d , double * v_out , const int n , int * it_out , const double alpha ) ;

int broydenMod_min( double * v_in , double * v_in_old , double * f_old , double * b_bra , double * b_ket , double * d , double * v_out , const int n , int * it_out , const double alpha ) ;

void broyden_min_( double * v_in , double * v_in_old , double * f_old , double * b_bra , double * b_ket , double * d , double * v_out , int * n , int * it_out , double * alpha ) 

{
  broydenMod_min( v_in , v_in_old , f_old , b_bra , b_ket , d , v_out , * n , it_out , * alpha ) ;
}

void broyden_minb_( double * v_in , double * v_in_old , double * f_old , double * b_bra , double * v_out , int * n , int * it_out , double * alpha ) 

{

  broyden_minB( v_in , v_in_old , f_old , b_bra , v_out , * n , it_out , * alpha ) ;

}

int broyden_min( double * v_in , double * v_in_old , double * f_old , double * b_bra , double * b_ket , double * d , double * v_out , const int n , int * it_out , const double alpha )

{

  double * f , * dv , * df ;

  int i , ii , it1 ;

  int ione = 1 ;

  double norm , norm1 , norm2 ;

  int n1 , ishift ;

  int it ;

  it = * it_out ;

  n1 = n ;

  if( it == 0 )

    {

      * it_out = 1 ;

      for( i = 0 ; i < n ; ++i ) {

	v_in_old[ i ] = v_in[ i ] ;

	f_old[ i ] = alpha * ( v_in[ i ] - v_out[ i ] ) ;

	v_in[ i ]  -= f_old[ i ] ;

	v_out[ i ] = v_in[ i ] ;

      }

      return( 0 ) ;

    }

  assert( f = malloc( n * sizeof( double ) ) ) ;

  assert( df = malloc( n * sizeof( double ) ) ) ;

  assert( dv = malloc( n * sizeof( double ) ) ) ;

  it1 = it - 1 ;

  ishift = it1 * n ;

  norm = 0. ; /* || F || */

  norm1 = 0. ; /* || F_old || */

  for( i = 0 ; i < n ; i++ )

    {

      f[ i ] =  alpha * ( v_in[ i ] - v_out[ i ] ) ;

      norm += f[ i ] * f[ i ] ;

      norm1 += f_old[ i ] * f_old[ i ] ;

      dv[ i ] = v_in[ i ] - v_in_old[ i ] ;

      df[ i ] = f[ i ] - f_old[ i ] ;

      b_ket[ ishift + i ] = df[ i ] ;

      b_bra[ ishift + i ] = dv[ i ] ;

    }

  for( ii = 0 ; ii < it1 ; ++ii )

    {

      norm =  d[ ii ] * ddot_( & n1 , df , &ione , b_bra + ii * n , &ione ) ;

      daxpy_( & n1 , & norm , b_ket + ii * n , & ione , b_ket + ishift , &ione ) ;

      norm = d[ ii ] * ddot_( &n1 , dv , &ione , b_ket + ii * n , &ione ) ;

      daxpy_( & n1 , & norm , b_bra + ii * n , & ione , b_bra + ishift , &ione ) ;

    }

  norm = ddot_( & n1 , dv , &ione , b_ket + ishift , &ione ) ;

  printf( "<dv|B|df> =%12.6e\n", norm ) ;

  if( fabs( norm ) < 1.e-15 ) return( 0 ) ;

  for( i = 0 ; i < n ; ++i )

    b_ket[ ishift + i ] = ( dv[ i ] - b_ket[ ishift + i ] ) / norm ;

  norm1 = sqrt( ddot_( & n1 , b_ket + ishift , &ione , b_ket + ishift , &ione ) ) ;

  norm2 = sqrt( ddot_( & n1 , b_bra + ishift , &ione , b_bra + ishift , &ione ) ) ;

  for( i = 0 ; i < n ; i++ )

    {

      b_ket[ ishift + i ] = b_ket[ ishift + i ] / norm1 ;

      b_bra[ ishift + i ] = b_bra[ ishift + i ] / norm2 ;

    }

  d[ it1 ] = norm1 * norm2 ;

  for( i = 0 ; i < n ; i++)
    {

      v_in_old[ i ] = v_in[ i ] ;

      v_in[ i ] -= f[ i ] ;

    }

  for( ii = 0 ; ii < it ; ii++ )

    {

      norm = - d[ ii ] * ddot_( & n1 , b_bra + ii * n , &ione , f , &ione ) ;

      daxpy_( & n1 , & norm , b_ket + ii * n , & ione , v_in , &ione ) ;

    }

  for( i = 0 ; i < n ; i++ )

    {

      f_old[ i ] = f[ i ] ;

      v_out[ i ] = v_in[ i ] ;

    }

  ( * it_out ) ++ ;

  free( f ) ; free( dv ) ; free( df ) ;

  return( 0 ) ;

}

int broyden_minB( double * v_in , double * v_in_old , double * f_old , double * b , double * v_out , const int n , int * it_out , const double alpha )

{

  double * f , * dv , * df , * bdf , * bdv ;

  int i , ii , it1 ;

  int ione = 1 ;

  double norm , norm1 , norm2 ;

  int n1 , ishift ;

  int it , j ;

  double * u , * vt , * s , * work , wk_[ 2 ] ;

  int lwork = -1 , info , * iwork ;

  it = * it_out ;

  printf( "Starting Broyden %d \n\n" , it ) ;

  n1 = n ;

  if( it == 0 )

    {

      * it_out = 1 ;

      for( i = 0 ; i < n ; ++i ) {

	v_in_old[ i ] = v_in[ i ] ;

	f_old[ i ] = alpha * ( v_in[ i ] - v_out[ i ] ) ;

	v_in[ i ]  -= f_old[ i ] ;

	v_out[ i ] = v_in[ i ] ;

	for( j = 0 ; j < n ; j++ )

	  b[ i + n * j ] = 0. ;

	b[ i + n * i ] = 1. ;

      }

      return( 0 ) ;

    }

  assert( f = malloc( n * sizeof( double ) ) ) ;

  assert( df = malloc( n * sizeof( double ) ) ) ;

  assert( dv = malloc( n * sizeof( double ) ) ) ;

  assert( bdv = malloc( n * sizeof( double ) ) ) ;

  assert( bdf = malloc( n * sizeof( double ) ) ) ;

  it1 = it - 1 ;

  ishift = it1 * n ;

  for( i = 0 ; i < n ; i++ )

    {

      f[ i ] =  alpha * ( v_in[ i ] - v_out[ i ] ) ;

      dv[ i ] = v_in[ i ] - v_in_old[ i ] ;

      df[ i ] = f[ i ] - f_old[ i ] ;

    }

  norm = 0. ;

  for( i = 0 ; i < n ; i ++ ) {

    bdv[ i ] = 0. ;

    bdf[ i ] = dv[ i ] ;

    for( j = 0 ; j < n ; j++ ){

      bdv[ i ] += b[ j + n * i ] * dv[ j ] ;

      bdf[ i ] -= b[ i + n * j ] * df[ j ] ;

    }

    norm += df[ i ] * bdv[ i ] ;

  }

  printf( "<dv|B|df> =%12.6e\n", norm ) ;

  norm = 1./norm ;

  dger_( &n1 , &n1 , &norm , bdf , &ione , bdv , &ione , b , &n1 ) ;

  if( it > 5 ){

  /* add a SVD decomposition of matrix B */

    assert( s = malloc( n * sizeof( double ) ) ) ;

    assert( u = malloc( n * n  * sizeof( double ) ) ) ;

    assert( vt = malloc( n * n * sizeof( double ) ) ) ;

    assert( iwork = malloc( 8 * n * sizeof( double ) ) ) ;

    dgesdd_( "A", &n1 , &n1 , b , &n1 , s , u , &n1 , vt , &n1 , wk_ , &lwork , iwork , &info ) ;

    lwork = ( int ) wk_[ 0 ] ;

    assert( work = malloc( lwork * sizeof( double ) ) ) ;

    dgesdd_( "A" , &n1 , &n1 , b , &n1 , s , u , &n1 , vt , &n1 , work , &lwork , iwork , &info ) ;

    free( work ) ; free( iwork ) ;

    for( i = 0 ; i < n ; i++ ){

      if( s[ i ] < 1.e-6 ) {

	printf( " s[ %d ] = %g \n" , i , s[ i ] ) ;

	s[ i ] = 0. ;

      }

      for( j = 0 ; j < n ; j++ )

	vt[ i + j * n ] *= s[ i ] ;

    }

    free( s ) ;

    norm = 0. ;

    norm1 = 1. ;
    
    dgemm_( "N" , "N" , &n1 , &n1 , &n1 , &norm1 , u , &n1 , vt , &n1 , &norm , b , &n1 ) ;

    free( u ) ; free( vt ) ;

  }

  dcopy_( &n1 , v_in , &ione , v_in_old , &ione ) ;

  norm1 = -1. ;

  norm = 1. ;

  dgemv_( "N" , &n1 , &n1 , &norm1 , b , &n1 , f , &ione , &norm , v_in , &ione ) ;

  for( i = 0 ; i < n ; i++ )

    {

      f_old[ i ] = f[ i ] ;

      v_out[ i ] = v_in[ i ] ;

    }

  ( * it_out ) ++ ;

  free( f ) ; free( dv ) ; free( df ) ; free( bdf ) ; free( bdv ) ;

  printf( "Done with Broyden \n\n" ) ;

  return( 0 ) ;

}

int broydenMod_min( double * v_in , double * v_in_old , double * f_old , double * b_bra , double * b_ket , double * d , double * v_out , const int n , int * it_out , const double alpha )
/*

Change in the formula. Use Eq. (10) in J.Chem.Phys. 134 (2011) 134109

 */

{

  double * f , * dv , * df ;

  int i , ii , it1 ;

  int ione = 1 ;

  double norm , norm1 , norm2 ;

  int n1 , ishift ;

  int it ;

  it = * it_out ;

  printf( "Starting Broyden %d \n\n" , it ) ;

  n1 = n ;

  if( it == 0 )

    {

      * it_out = 1 ;

      for( i = 0 ; i < n ; ++i ) {

	v_in_old[ i ] = v_in[ i ] ;

	f_old[ i ] = alpha * ( v_in[ i ] - v_out[ i ] ) ;

	v_in[ i ]  -= f_old[ i ] ;

	v_out[ i ] = v_in[ i ] ;

      }

      return( 0 ) ;

    }

  assert( f = malloc( n * sizeof( double ) ) ) ;

  assert( df = malloc( n * sizeof( double ) ) ) ;

  assert( dv = malloc( n * sizeof( double ) ) ) ;

  it1 = it - 1 ;

  ishift = it1 * n ;

  norm = 0. ; /* || F || */

  norm1 = 0. ; /* || F_old || */

  for( i = 0 ; i < n ; i++ )

    {

      f[ i ] =  alpha * ( v_in[ i ] - v_out[ i ] ) ;

      norm += f[ i ] * f[ i ] ;

      norm1 += f_old[ i ] * f_old[ i ] ;

      dv[ i ] = v_in[ i ] - v_in_old[ i ] ;

      df[ i ] = f[ i ] - f_old[ i ] ;

      b_ket[ ishift + i ] = df[ i ] ;

      b_bra[ ishift + i ] = df[ i ] ;

    }

  for( ii = 0 ; ii < it1 ; ++ii )

    {

      norm =  d[ ii ] * ddot_( & n1 , df , &ione , b_bra + ii * n , &ione ) ;

      daxpy_( & n1 , & norm , b_ket + ii * n , & ione , b_ket + ishift , &ione ) ;

    }

  norm = ddot_( & n1 , df , &ione , df , &ione ) ;

  printf( "<df|df> =%12.6e\n", norm ) ;

  for( i = 0 ; i < n ; ++i )

    b_ket[ ishift + i ] = ( dv[ i ] - b_ket[ ishift + i ] ) / norm ;

  norm1 = sqrt( ddot_( & n1 , b_ket + ishift , &ione , b_ket + ishift , &ione ) ) ;

  norm2 = sqrt( ddot_( & n1 , b_bra + ishift , &ione , b_bra + ishift , &ione ) ) ;

  for( i = 0 ; i < n ; i++ )

    {

      b_ket[ ishift + i ] = b_ket[ ishift + i ] / norm1 ;

      b_bra[ ishift + i ] = b_bra[ ishift + i ] / norm2 ;

    }

  d[ it1 ] = norm1 * norm2 ;

  for( i = 0 ; i < n ; i++)
    {

      v_in_old[ i ] = v_in[ i ] ;

      v_in[ i ] -= f[ i ] ;

    }

  for( ii = 0 ; ii < it ; ii++ )

    {

      norm = - d[ ii ] * ddot_( & n1 , b_bra + ii * n , &ione , f , &ione ) ;

      daxpy_( & n1 , & norm , b_ket + ii * n , & ione , v_in , &ione ) ;

    }

  for( i = 0 ; i < n ; i++ )

    {

      f_old[ i ] = f[ i ] ;

      v_out[ i ] = v_in[ i ] ;

    }

  ( * it_out ) ++ ;

  free( f ) ; free( dv ) ; free( df ) ;

  printf( "Done with Broyden \n\n" ) ;

  return( 0 ) ;

}

double ddotII( int * n , double *v1 , int * i1 , double * v2 , int * i2 )

{

  double sum = 0. ;

  int i ;

  for( i = 0 ; i < * n ; i++ )

    sum += v1[ i ] * v2[ i ] ;

  return( sum ) ;

}

void daxpyII( int * n , double * alpha , double * x , int * i1 , double * y , int * i2 ) 

{

  int i ;

  for( i = 0 ; i < * n ; i++ )

    y[ i ] += *alpha * x[ i ] ;

}
