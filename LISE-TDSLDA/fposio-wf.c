// for license information, see the accompanying LICENSE file

/* kenneth.roche@pnl.gov ; k8r@uw.edu */

/* modified by I. Stetcu to be used more easily in c codes */

#include <stdlib.h>

#include <stdio.h>

#include <string.h>

#include <sys/types.h>

#include <sys/stat.h>

#include <fcntl.h>

#include <time.h> 

#include <unistd.h> 

#include <math.h>

#include <complex.h>

int f_copn( char * fn )

{

  int fd ; /* file descriptor for handling the file or device */
  
  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  if ( ( fd = open( fn , O_CREAT | O_RDWR | O_APPEND , fd_mode ) ) == -1 ) 
    
    fprintf( stderr , "error: cannot create FILE %s for WRITE\n" , fn ) ;

  return( fd ) ; 

} 

void f_ccls( int * fd )

{

  if( close( * fd ) == -1 ) printf("Error: file not open.\n") ;

}

void f_crm( char * fn )

{

  int i ;

  /* remove the file in path fname */
  
  if ( ( i = unlink( ( const char * ) fn ) ) != 0 ) 
    
    fprintf( stderr , "f_crm(): cannot unlink() FILE %s\n" , fn ) ;

}

void f_cwr( const int fd , void * fbf , const int fsz , const int nobj ) 

{

  /* will write nobj * fsz elements from fbf */

  long int bytes_written ; long int bytestowrite ;
  
  bytestowrite =  ( long ) fsz * nobj ; 
  
  if ( ( long ) ( bytes_written = write( fd , ( const void * ) fbf , ( size_t ) bytestowrite ) ) != bytestowrite ) 

    {
      
      fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;
      
    }
  
}

void f_crd( int fd , void * fbf , const int fsz , const int nobj ) 

{

  /* will write nobj * fsz elements from fbf */

  long int bytes_read ; long int bytestoread ;
  
  bytestoread =  ( long ) fsz * nobj ; 
  
  if ( ( long ) ( bytes_read = read( fd , ( void * ) fbf , ( size_t ) bytestoread ) ) != bytestoread ) 

    {
      
      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;
      
    }
  
}

