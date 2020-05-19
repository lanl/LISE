// for license information, see the accompanying LICENSE file

/* used to print the wave functions; will be replaced in the near future */

#include <stdlib.h>

#include <stdio.h>

#include <mpi.h>

#include <math.h>

#include <complex.h>

#include <assert.h>

#include <sys/types.h>

#include <sys/stat.h>

#include <fcntl.h>

#include <unistd.h>

#include <string.h>

double factor_ec( const double , const double , int icub) ;

int print_wf( char * fn , MPI_Comm comm , double * lam , double complex * z , const int ip , const int nwf , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc , const int nx , const int ny , const int nz , const double e_cut , double * occ ,int icub)

{

  double complex * vec , * vec1 ;

  double f1 ;

  int n , nhalf , ntot , nxyz ;

  int i , j , jj , li , lj ;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  int fd , iflag ;

  long int bytes_written ; long int bytestowrite ;

  nxyz = nx * ny * nz ;

  n = 4 * nxyz ;

  nhalf = n / 2 ;

  ntot = nhalf + nwf ;

  MPI_Barrier( comm ) ;

  assert( vec = malloc( n * sizeof( double complex ) ) ) ;

  assert( vec1 = malloc( n * sizeof( double complex ) ) ) ;

  bytestowrite = ( long ) n * sizeof( double complex ) ;

  iflag = EXIT_SUCCESS ;

  if( ip == 0 )

    {

      if ( ( i = unlink( ( const char * ) fn ) ) != 0 ) fprintf( stderr , "Cannot unlink() FILE %s\n" , fn ) ;
 
      if ( ( fd = open( fn , O_CREAT | O_WRONLY , fd_mode ) ) == -1 ) 
	
	{

	  fprintf( stderr , "error: cannot open FILE %s for WRITE\n" , fn ) ;
   
	  iflag = EXIT_FAILURE ; 

	}
  
    }

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  if( iflag == EXIT_FAILURE ) 

    return( EXIT_FAILURE ) ;

  for( jj = nhalf ; jj < ntot ; jj++ )

    /* construct one vector at a time for positive eigenvalues */

    {

      f1 = sqrt( factor_ec( *( lam + jj ) , e_cut, icub ) ) ;

      //      printf( "posix: gj[ %d ] f1[ %f ]\n" , jj , f1 ) ;

      for( i = 0 ; i < n ; i++ )

	*( vec1 + i ) = 0. + I * 0. ;

      for( lj = 0 ; lj < n_iq ; lj++ )

	{

	  j =  i_q * nb + ( int ) ( floor( ( double ) lj / nb ) ) * q_proc * nb + lj % nb ;

	  if( j == jj )

	    for( li = 0 ; li < m_ip ; li++ ) 

	      vec1[ i_p * mb + ( int ) ( floor( ( double ) li / mb ) ) * p_proc * mb + li % mb ] = z[ lj * m_ip + li ] * f1 ;

	}

      MPI_Reduce( vec1 , vec , n , MPI_DOUBLE_COMPLEX , MPI_SUM , 0 , comm ) ;

      if( ip == 0 )

	{
	  
	  if ( ( bytes_written = ( long ) write( fd , ( const void * ) vec , ( size_t ) bytestowrite ) ) != bytestowrite )

	    {

	      fprintf( stderr , "err in print_wf: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	      iflag = EXIT_FAILURE ;

	      close( fd ) ;

	    }

	}

      MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

      if( iflag == EXIT_FAILURE )

	return( EXIT_FAILURE );

    }

  if( ip == 0 )

    {

      close( fd ) ;

      printf( "%d wave functions successfully written in %s\n" , nwf , fn ) ;

    }

  free( vec ) ; free( vec1 ) ;

  return( EXIT_SUCCESS ) ;

}


int print_wf2( char * fn , MPI_Comm comm , double * lam , double complex * z , const int ip , const int nwf , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc , const int nx , const int ny , const int nz , const double e_cut , double * occ , int icub)

{

  double complex * vec , * vec1 ;

  double f1 ;

  int n , nhalf , nxyz ;

  int i , j , jj , li , lj ;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  int fd , iflag , nwf1 = 0 ;

  long int bytes_written ; long int bytestowrite ;

  nxyz = nx * ny * nz ;

  n = 4 * nxyz ;

  nhalf = n / 2 ;

  assert( vec = malloc( n * sizeof( double complex ) ) ) ;

  assert( vec1 = malloc( n * sizeof( double complex ) ) ) ;

  bytestowrite = ( long ) n * sizeof( double complex ) ;

  iflag = EXIT_SUCCESS ;

  if( ip == 0 )

    {

      if ( ( i = unlink( ( const char * ) fn ) ) != 0 )
	
	fprintf( stderr , "Cannot unlink() FILE %s\n" , fn ) ;
 
      if ( ( fd = open( fn , O_CREAT | O_WRONLY , fd_mode ) ) == -1 ) 
	
	{

	  fprintf( stderr , "error: cannot open FILE %s for WRITE\n" , fn ) ;
   
	  iflag = EXIT_FAILURE ; 

	}
  
    }

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  if( iflag == EXIT_FAILURE ) 

    return( EXIT_FAILURE ) ;

  for( jj = nhalf ; jj < n ; jj++ )

    /* construct one vector at a time for positive eigenvalues , time reversal states constructed by hand */

    {

      if( occ[ jj - nhalf ] < .9 )

	continue ;

      nwf1++ ;

      f1 = sqrt( factor_ec( *( lam + jj ) , e_cut, icub ) ) ;

      for( i = 0 ; i < n ; i++ )

	*( vec1 + i ) = 0. + 0. * I ;

      for( lj = 0 ; lj < n_iq ; lj++ )

	{

	  j =  i_q * nb + ( int ) ( floor( ( double ) lj / nb ) ) * q_proc * nb + lj % nb ;

	  if( j == jj )

	    for( li = 0 ; li < m_ip ; li++ ) 

	      vec1[ i_p * mb + ( int ) ( floor( ( double ) li / mb ) ) * p_proc * mb + li % mb ] = z[ lj * m_ip + li ] * f1 ;

	}

      MPI_Reduce( vec1 , vec , n , MPI_DOUBLE_COMPLEX , MPI_SUM , 0 , comm ) ;

      if( ip == 0 )

	{
	  
	  if ( ( bytes_written = ( long ) write( fd , ( const void * ) vec , ( size_t ) bytestowrite ) ) != bytestowrite )

	    {

	      fprintf( stderr , "err in print_wf2: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	      iflag = EXIT_FAILURE ;

	      close( fd ) ;

	    }

	}

      MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

      if( iflag == EXIT_FAILURE )

	return( EXIT_FAILURE );

    }

  if( ip == 0 )

    {

      close( fd ) ;

      printf( "%d wave functions successfully written in %s\n" , nwf1 , fn ) ;

    }

  free( vec ) ; free( vec1 ) ;

  return( EXIT_SUCCESS ) ;

}

