// for license information, see the accompanying LICENSE file

#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <assert.h>

// CBLAS, LAPACKE are f2c translated Netlib F77 reference routines.
// 'cblas.h' is widely supported, and is the f2c translated Netlib F77 refernce version. 
// There are known issues with name mangling schemes regarding wrapped code and inter-language operability between Fortran and C. 
// 
// Netlib CBLAS / LAPACKE target
// #define LISE_LA_REF
// #include <lapacke.h>
// #include <cblas.h>
//
// IBM ESSL target
// #define LISE_LA_ESSL
// #include <essl.h>
//
// Intel MKL target
// #define LISE_LA_MKL
// #include <mkl.h>

#ifdef LISE_LA_MKL
#include <mkl.h>
double ddot(const int *, const double *, const int *, const double *, const int *);
void daxpy(const int *, const double *, const double *, const int *, double *, const int *);
void dgemv(const char *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);
void dcopy(const int *, const double *, const int *, double *, const int *);
void dgemm(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);
void dgesdd(const char *, const int *, const int *, double *, const int *, double *, double *, const int *, double *, const int *, double *, const int *, int *, int *);
void dger(const int *, const int *, const double *, const double *, const int *, const double *, const int *, double *, const int *);
#endif

#ifdef LISE_LA_ESSL
#include <essl.h>
double ddot(int, const double *, int, const double *,int);
void daxpy(int,  double,  double *, int,  double *,  int);
void dgemv(const char *, int, int,  double, const void *, int, const  double *, int,  double,  double *, int);
void dcopy(int,  double *, int,  double *, int);
void dgemm(const char *, const char *, int, int, int,  double, const void *, int, const void *, int,  double, void *, int);
void dgesdd(const char *, int, int, void *, int, double *, void *, int, void *, int, double *, int, int *, int *);
void dger(int, int,  double, const  double *, int, const  double *, int, void *, int);
// void dger1(int, int,  double, const  double *, int, const  double *, int, void *, int);
#endif

#ifdef LISE_LA_REF
#include <lapacke.h>
#include <cblas.h>
double cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
void cblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY);
void cblas_dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const int M, const int N, const double alpha, const double *A, const int lda, const double *X, const int incX, const double beta, double *Y, const int incY);
void cblas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY);
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
void cblas_dger(CBLAS_LAYOUT layout, const int M, const int N, const double alpha, const double *X, const int incX, const double *Y, const int incY, double *A, const int lda);
lapack_int LAPACKE_dgesdd( int matrix_layout, char jobz, lapack_int m, lapack_int n, double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt );
#endif

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

#ifdef LISE_LA_MKL
      norm =  d[ ii ] * ddot( &n1 , df , &ione , b_bra + ii * n , &ione ) ;
      daxpy( &n1 , &norm , b_ket + ii * n , &ione , b_ket + ishift , &ione ) ;
      norm = d[ ii ] * ddot( &n1 , dv , &ione , b_ket + ii * n , &ione ) ;
      daxpy( &n1 , &norm , b_bra + ii * n , &ione , b_bra + ishift ,&ione ) ;
#endif
#ifdef LISE_LA_ESSL
      norm =  d[ ii ] * ddot( n1 , df , ione , b_bra + ii * n , ione ) ;
      daxpy( n1 , norm , b_ket + ii * n , ione , b_ket + ishift , ione ) ;
      norm = d[ ii ] * ddot( n1 , dv , ione , b_ket + ii * n , ione ) ;
      daxpy( n1 , norm , b_bra + ii * n , ione , b_bra + ishift ,ione ) ;
#endif
#ifdef LISE_LA_REF
//double cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
//void cblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY);
      norm =  d[ ii ] * cblas_ddot( n1 , df , ione , b_bra + ii * n , ione ) ;
      cblas_daxpy( n1 , norm , b_ket + ii * n , ione , b_ket + ishift , ione ) ;
      norm = d[ ii ] * cblas_ddot( n1 , dv , ione , b_ket + ii * n , ione ) ;
      cblas_daxpy( n1 , norm , b_bra + ii * n , ione , b_bra + ishift ,ione ) ;
#endif

    }

#ifdef LISE_LA_MKL
  norm = ddot(  &n1 , dv , &ione , b_ket + ishift , &ione ) ;
#endif
#ifdef LISE_LA_ESSL
  norm = ddot(  n1 , dv , ione , b_ket + ishift , ione ) ;
#endif
#ifdef LISE_LA_REF
  norm = cblas_ddot(  n1 , dv , ione , b_ket + ishift , ione ) ;
#endif

  printf( "<dv|B|df> =%12.6e\n", norm ) ;

  if( fabs( norm ) < 1.e-15 ) return( 0 ) ;

  for( i = 0 ; i < n ; ++i )

    b_ket[ ishift + i ] = ( dv[ i ] - b_ket[ ishift + i ] ) / norm ;

#ifdef LISE_LA_MKL
  norm1 = sqrt( ddot( &n1 , b_ket + ishift , &ione , b_ket + ishift , &ione ) ) ;
  norm2 = sqrt( ddot( &n1 , b_bra + ishift , &ione , b_bra + ishift , &ione ) ) ;
#endif
#ifdef LISE_LA_ESSL
  norm1 = sqrt( ddot( n1 , b_ket + ishift , ione , b_ket + ishift , ione ) ) ;
  norm2 = sqrt( ddot( n1 , b_bra + ishift , ione , b_bra + ishift , ione ) ) ;
#endif
#ifdef LISE_LA_REF
  norm1 = sqrt( cblas_ddot( n1 , b_ket + ishift , ione , b_ket + ishift , ione ) ) ;
  norm2 = sqrt( cblas_ddot( n1 , b_bra + ishift , ione , b_bra + ishift , ione ) ) ;
#endif

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

#ifdef LISE_LA_MKL
      norm = - d[ ii ] * ddot( &n1 , b_bra + ii * n , &ione , f , &ione ) ;
      daxpy( &n1 , &norm , b_ket + ii * n , &ione , v_in , &ione ) ;
#endif
#ifdef LISE_LA_ESSL
      norm = - d[ ii ] * ddot( n1 , b_bra + ii * n , ione , f , ione ) ;
      daxpy( n1 , norm , b_ket + ii * n , ione , v_in , ione ) ;
#endif
#ifdef LISE_LA_REF
      norm = - d[ ii ] * cblas_ddot( n1 , b_bra + ii * n , ione , f , ione ) ;
      cblas_daxpy( n1 , norm , b_ket + ii * n , ione , v_in , ione ) ;
#endif

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

#ifdef LISE_LA_REF
  cblas_dger( CBLASColMajor, n1 , n1 , norm , bdf , ione , bdv , ione , b , n1 ) ;
#endif
#ifdef LISE_LA_ESSL
  dger( n1 , n1 , norm , bdf , ione , bdv , ione , b , n1 ) ;
#endif
#ifdef LISE_LA_MKL
// void dger(const int *, const int *, const double *, const double *, const int *, const double *, const int *, void *, const int *);
  dger( &n1 , &n1 , &norm , bdf , &ione , bdv , &ione , b , &n1 ) ;
#endif

  if( it > 5 ){

  /* add a SVD decomposition of matrix B */

    assert( s = malloc( n * sizeof( double ) ) ) ;

    assert( u = malloc( n * n  * sizeof( double ) ) ) ;

    assert( vt = malloc( n * n * sizeof( double ) ) ) ;

    assert( iwork = malloc( 8 * n * sizeof( double ) ) ) ;

#ifdef LISE_LA_MKL
    dgesdd( "A", &n1 , &n1 , b , &n1 , s , u , &n1 , vt , &n1 , wk_ , &lwork , iwork , &info ) ;
#endif
#ifdef LISE_LA_ESSL
    dgesdd( "A", n1 , n1 , b , n1 , s , u , n1 , vt , n1 , wk_ , lwork , iwork , &info ) ;
#endif
#ifdef LISE_LA_REF
    info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', n1 , n1 , b , n1 , s , u , n1 , vt , n1 ) ; 
#endif

    lwork = ( int ) wk_[ 0 ] ;

    assert( work = malloc( lwork * sizeof( double ) ) ) ;

#ifdef LISE_LA_ESSL
    dgesdd( "A" , n1 , n1 , b , n1 , s , u , n1 , vt , n1 , work , lwork , iwork , &info ) ;
#endif
#ifdef LISE_LA_REF
    info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', n1 , n1 , b , n1 , s , u , n1 , vt , n1 ) ; 
#endif
#ifdef LISE_LA_MKL
    dgesdd( "A", &n1 , &n1 , b , &n1 , s , u , &n1 , vt , &n1 , work , &lwork , iwork , &info ) ;
#endif

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
    
#ifdef LISE_LA_ESSL
    dgemm( "N" , "N" , n1 , n1 , n1 , norm1 , u , n1 , vt , n1 , norm , b , n1 ) ;
#endif
#ifdef LISE_LA_MKL
// void dgemm(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);
    dgemm( "N" , "N" , &n1 , &n1 , &n1 , &norm1 , u , &n1 , vt , &n1 , &norm , b , &n1 ) ;
#endif
#ifdef LISE_LA_REF
    cblas_dgemm(CBLAS_COL_MAJOR,"N" , "N" , n1 , n1 , n1 , norm1 , u , n1 , vt , n1 , norm , b , n1);
#endif

    free( u ) ; free( vt ) ;

  }

#ifdef LISE_LA_ESSL
  dcopy( n1 , v_in , ione , v_in_old , ione ) ;
#endif
#ifdef LISE_LA_MKL
//void dcopy(const int *, const double *, const int *, double *, const int *);
  dcopy( &n1 , v_in , &ione , v_in_old , &ione ) ;
#endif
#ifdef LISE_LA_REF
  cblas_dcopy( n1 , v_in , ione , v_in_old , ione ) ;
#endif

  norm1 = -1. ;

  norm = 1. ;

#ifdef LISE_LA_MKL
//void dgemv(const char *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);
  dgemv( "N" , &n1 , &n1 , &norm1 , b , &n1 , f , &ione , &norm , v_in , &ione ) ;
#endif
#ifdef LISE_LA_ESSL
  dgemv( "N" , n1 , n1 , norm1 , b , n1 , f , ione , norm , v_in , ione ) ;
#endif
#ifdef LISE_LA_REF
  cblas_dgemv(CBLAS_COL_MAJOR, "N" , n1 , n1 , norm1 , b , n1 , f , ione , norm , v_in , ione ) ; 
#endif

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

#ifdef LISE_LA_MKL
      norm =  d[ ii ] * ddot( &n1 , df , &ione , b_bra + ii * n , &ione ) ;
      daxpy( &n1 , &norm , b_ket + ii * n , &ione , b_ket + ishift , &ione ) ;
#endif
#ifdef LISE_LA_ESSL 
      norm =  d[ ii ] * ddot( n1 , df , ione , b_bra + ii * n , ione ) ;
      daxpy( n1 , norm , b_ket + ii * n , ione , b_ket + ishift , ione ) ;
#endif
#ifdef LISE_LA_REF
      norm =  d[ ii ] * cblas_ddot( n1 , df , ione , b_bra + ii * n , ione ) ;
      cblas_daxpy( n1 , norm , b_ket + ii * n , ione , b_ket + ishift , ione ) ;
#endif

    }

#ifdef LISE_LA_MKL
  norm = ddot( &n1 , df , &ione , df , &ione ) ;
#endif
#ifdef LISE_LA_ESSL 
  norm = ddot( n1 , df , ione , df , ione ) ;
#endif
#ifdef LISE_LA_REF
  norm = cblas_ddot( n1 , df , ione , df , ione ) ;
#endif

  printf( "<df|df> =%12.6e\n", norm ) ;

  for( i = 0 ; i < n ; ++i )

    b_ket[ ishift + i ] = ( dv[ i ] - b_ket[ ishift + i ] ) / norm ;

#ifdef LISE_LA_MKL
  norm1 = sqrt( ddot( &n1 , b_ket + ishift , &ione , b_ket + ishift , &ione ) ) ;
  norm2 = sqrt( ddot( &n1 , b_bra + ishift , &ione , b_bra + ishift ,&ione ) ) ;
#endif
#ifdef LISE_LA_ESSL
  norm1 = sqrt( ddot( n1 , b_ket + ishift , ione , b_ket + ishift , ione ) ) ;
  norm2 = sqrt( ddot( n1 , b_bra + ishift , ione , b_bra + ishift ,ione ) ) ;
#endif
#ifdef LISE_LA_REF
  norm1 = sqrt( cblas_ddot( n1 , b_ket + ishift , ione , b_ket + ishift , ione ) ) ;
  norm2 = sqrt( cblas_ddot( n1 , b_bra + ishift , ione , b_bra + ishift ,ione ) ) ;
#endif


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

#ifdef LISE_LA_MKL
      norm = - d[ ii ] * ddot( &n1 , b_bra + ii * n , &ione , f , &ione ) ;
      daxpy( &n1 , &norm , b_ket + ii * n , &ione , v_in , &ione ) ;
#endif
#ifdef LISE_LA_ESSL
      norm = - d[ ii ] * ddot( n1 , b_bra + ii * n , ione , f , ione ) ;
      daxpy( n1 , norm , b_ket + ii * n , ione , v_in , ione ) ;
#endif
#ifdef LISE_LA_REF
      norm = - d[ ii ] * cblas_ddot( n1 , b_bra + ii * n , ione , f , ione ) ;
      cblas_daxpy( n1 , norm , b_ket + ii * n , ione , v_in , ione ) ;
#endif

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
