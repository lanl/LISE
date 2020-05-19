// for license information, see the accompanying LICENSE file

/* 

   Used to save and read back potentials

*/

#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <complex.h>

#include <assert.h>

#include "vars_nuclear.h"

#include <sys/types.h>

#include <sys/stat.h>

#include <fcntl.h>

#include <mpi.h>

#include <unistd.h>

#include <string.h>

int copy_lattice_arrays( void * , void * , size_t , const int , const int , const int , const int , const int , const int ) ;

int write_pots( char * fn , double * pot_arrays , const int nx , const int ny , const int nz , const double dx , const double dy , const double dz , const int ishift )

{

  int fd ;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  long int bytes_written ; long int bytestowrite ;

  int nxyz ;

  int i ;

  nxyz = nx * ny * nz ;

  if ( ( i = unlink( ( const char * ) fn ) ) != 0 )
	
    fprintf( stderr , "Cannot unlink() FILE %s\n" , fn ) ;
 
  if ( ( fd = open( fn , O_CREAT | O_WRONLY , fd_mode ) ) == -1 ) 
	
    {

      fprintf( stderr , "error: cannot open FILE %s for WRITE\n" , fn ) ;
   
      return( EXIT_FAILURE ) ; 

    }

  bytestowrite = sizeof( int ) ;
  
  if ( ( bytes_written = ( long ) write( fd , ( const void * ) &nx , ( size_t ) bytestowrite ) ) != bytestowrite )

    {

      fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

      close( fd ) ;

      return( EXIT_FAILURE ) ;

    }

  if ( ( bytes_written = ( long ) write( fd , ( const void * ) &ny , ( size_t ) bytestowrite ) ) != bytestowrite )

    {

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  close( fd ) ;

	  return( EXIT_FAILURE ) ;

    }

  if ( ( bytes_written = ( long ) write( fd , ( const void * ) &nz , ( size_t ) bytestowrite ) ) != bytestowrite )

    {

      fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

      close( fd ) ;

      return( EXIT_FAILURE ) ;

    }

  bytestowrite = sizeof( double ) ;
  
  if ( ( bytes_written = ( long ) write( fd , ( const void * ) &dx , ( size_t ) bytestowrite ) ) != bytestowrite )

    {

      fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

      close( fd ) ;

      return( EXIT_FAILURE ) ;

    }

  if ( ( bytes_written = ( long ) write( fd , ( const void * ) &dy , ( size_t ) bytestowrite ) ) != bytestowrite )

    {

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  close( fd ) ;

	  return( EXIT_FAILURE ) ;

    }

  if ( ( bytes_written = ( long ) write( fd , ( const void * ) &dz , ( size_t ) bytestowrite ) ) != bytestowrite )

    {

      fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

      close( fd ) ;

      return( EXIT_FAILURE ) ;

    }

  bytestowrite =  ( long ) ( 7 * nxyz + 4 ) * sizeof( double ) ;

  if ( ( bytes_written = ( long ) write( fd , ( const void * ) ( pot_arrays + ishift ) , ( size_t ) bytestowrite ) ) != bytestowrite )

    {

      fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

      close( fd ) ;

      return( EXIT_FAILURE ) ;

    }

  close( fd ) ;

  return( EXIT_SUCCESS ) ;

}

int read_pots( char * fn , double * pot_arrays , const int nx , const int ny , const int nz , const double dx , const double dy , const double dz , const int ishift )

{

  int fd , iflag ;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  int nx1 , ny1 , nz1 ;

  int nxyz ;

  int i ;

  double complex * delta , * delta1 ;

  double * tmp ;

  double dx1 , dy1 , dz1 ;

  long int bytes_read ; long int bytestoread ;

  int nxyz1 ;

  nxyz = nx * ny * nz ;

  nxyz1 = nxyz ;
      
  /* Check to see if the file already exists, if so exit */
      
  if ( ( fd = open( fn , O_RDONLY , fd_mode ) ) == -1 ) 
	
    {

      fprintf( stderr , "error: cannot open FILE %s for READ\n" , fn ) ;
   
      return( EXIT_FAILURE ); 

    }
      
  bytestoread = ( long ) sizeof( int ) ;

  if ( ( bytes_read = ( long ) read( fd , &nx1 , ( size_t ) bytestoread ) ) != bytestoread )
	
    {

      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

      close( fd ) ;

      return( EXIT_FAILURE ) ;

    }

  if ( ( bytes_read = ( long ) read( fd , &ny1 , ( size_t ) bytestoread ) ) != bytestoread )

    {

      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

      close( fd ) ;

      return( EXIT_FAILURE );

    }

  if ( ( bytes_read = ( long ) read( fd , &nz1 , ( size_t ) bytestoread ) ) != bytestoread )

    {

      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

      close( fd ) ;

      return( EXIT_FAILURE );

    }

  printf( "potentials: nx1=%d ny1=%d nz1=%d\n" , nx1 , ny1 , nz1 ) ;
      
  bytestoread = ( long ) sizeof( double ) ;

  if ( ( bytes_read = ( long ) read( fd , &dx1 , ( size_t ) bytestoread ) ) != bytestoread )
	
    {

      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

      close( fd ) ;

      return( EXIT_FAILURE ) ;

    }

  if ( ( bytes_read = ( long ) read( fd , &dy1 , ( size_t ) bytestoread ) ) != bytestoread )

    {

      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

      close( fd ) ;

      return( EXIT_FAILURE );

    }

  if ( ( bytes_read = ( long ) read( fd , &dz1 , ( size_t ) bytestoread ) ) != bytestoread )

    {

      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

      close( fd ) ;

      return( EXIT_FAILURE );

    }

  if( dx1 != dx || dy1 != dy || dz1 != dz )

    {

      printf( "error: the lattice constants do not coincide \n new: %6.4f %6.4f %6.4f \n old: %6.4f %6.4f %6.4f \n" , dx , dy , dz , dx1 , dy1 , dz1 );

      close( fd ) ;

      return( EXIT_FAILURE ) ;

    }

  if( nx != nx1 || ny != ny1 || nz != nz1 )

    {

      nxyz1 = nx1 * ny1 * nz1 ;

      assert( tmp = malloc( ( 7 * nxyz1 + 4 ) * sizeof( double ) ) ) ;

      bytestoread = ( long ) ( 7 * nxyz1 + 4 ) * sizeof( double ) ;

      if ( ( bytes_read = ( long ) read( fd , ( void * ) tmp , ( size_t ) bytestoread ) ) != bytestoread )

	{

	  fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	  close( fd ) ;

	  return( EXIT_FAILURE ) ;

	}

      /* Coulomb should be added here */

      copy_lattice_arrays( tmp , pot_arrays + ishift , sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;

      copy_lattice_arrays( tmp + nxyz1 , pot_arrays + ishift + nxyz , sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;

      copy_lattice_arrays( tmp + 2 * nxyz1 , pot_arrays + ishift + 2 * nxyz , sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;

      copy_lattice_arrays( tmp + 3 * nxyz1 , pot_arrays + ishift + 3 * nxyz , sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;

      copy_lattice_arrays( tmp + 4 * nxyz1 , pot_arrays + ishift + 4 * nxyz , sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;

      copy_lattice_arrays( tmp + 5 * nxyz1 , pot_arrays + ishift + 5 * nxyz , sizeof( double complex ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;

      *( pot_arrays + 7 * nxyz + ishift ) = * ( tmp + 7 * nxyz ) ;

      pot_arrays[ 7 * nxyz1 + ishift + 1 ] = tmp[ 7 * nxyz + 1 ] ;

      pot_arrays[ 7 * nxyz1 + ishift + 2 ] = tmp[ 7 * nxyz + 2 ] ;

      pot_arrays[ 7 * nxyz1 + ishift + 3 ] = tmp[ 7 * nxyz + 3 ] ;

      free( tmp ) ;

    }

  else

    {
     
      bytestoread = ( long ) ( 7 * nxyz + 4 ) * sizeof( double ) ;

      if ( ( bytes_read = ( long ) read( fd , ( void * ) ( pot_arrays + ishift ) , ( size_t ) bytestoread ) ) != bytestoread )

	{

	  fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	  close( fd ) ;

	  return( EXIT_FAILURE ) ;

	}

    }
  
  close( fd ) ;

  return( EXIT_SUCCESS ) ;

}

