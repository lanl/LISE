// for license information, see the accompanying LICENSE file

/* 

   Used to save and read back densities

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

void i2xyz( const int i , int * ix , int * iy , int * iz , const int ny , const int nz );

void interpolate_lattice( double *x , double *y , double *z , const int nx , const int ny , const int nz , double *x1 , double *y1, double *z1 , const int nx1 , const double ny1 , const double nz1 , double * dens , double * dens1 ){

  // Trlinear interpolation: http://en.wikipedia.org/wiki/Trilinear_interpolation

  int i,i1;
  int nxyz = nx * ny * nz;
  int ix,iy,iz;
  int ix1,iy1,iz1;
  int n;
  int ix10,iy10,iz10;
  double xx1,yy1,zz1;

  for(i=0;i<nxyz;i++){
    i2xyz( i , &ix , &iy , &iz , ny , nz );
    ix1=-1;
    for(n=1;n<nx1;n++){
      if(x[ix]<=x1[n]){
	ix1=n;
	break;
      }
    }
    if(ix1<0){
      ix1=0;
      ix10=nx1-1;
      xx1=fabs(x1[0]);
    }else{
      ix10=ix1-1;
      xx1=x1[ix1];
    }
    iy1=-1;
    for(n=1;n<ny1;n++){
      if(y[iy]<=y1[n]){
	iy1=n;
	break;
      }
    }
    if(iy1<0){
      iy1=0;
      iy10=ny1-1;
      yy1=fabs(y1[0]);
    }else{
      iy10=iy1-1;
      yy1=y1[iy1];
    }

    iz1=-1;
    for(n=1;n<nz1;n++){
      if(z[iz]<=z1[n]){
	iz1=n;
	break;
      }
    }
    if(iz1<0){
      iz1=0;
      iz10=nz1-1;
      zz1=fabs(z1[0]);
    }else{
      iz10=iz1-1;
      zz1=z1[iz1];
    }

    //determined now where I am 
    int i000= iz10+nz1*(iy10+ny1*ix10); 
    int i100= iz10+nz1*(iy10+ny1*ix1); 
    int i010= iz10+nz1*(iy1+ny1*ix10); 
    int i001= iz1+nz1*(iy10+ny1*ix10); 
    int i110= iz10+nz1*(iy1+ny1*ix1); 
    int i101= iz1+nz1*(iy10+ny1*ix1); 
    int i011= iz1+nz1*(iy1+ny1*ix10); 
    int i111= iz1+nz1*(iy1+ny1*ix1);

    double xd=(x[ix]-x1[ix10])/(xx1-x1[ix10]);
    double yd=(y[iy]-y1[iy10])/(yy1-y1[iy10]);
    double zd=(z[iz]-z1[iz10])/(zz1-z1[iz10]);

    double c00=dens1[i000]*(1.-xd)+dens1[i100]*xd;
    double c10=dens1[i010]*(1.-xd)+dens1[i110]*xd;
    double c01=dens1[i001]*(1.-xd)+dens1[i101]*xd;
    double c11=dens1[i011]*(1.-xd)+dens1[i111]*xd;

    double c0=c00*(1.-yd)+c10*yd;
    double c1=c01*(1.-yd)+c11*yd;

    dens[i]=c0*(1.-zd)+c1*zd;

  }
}


void interpolate_lattice_complex( double *x , double *y , double *z , const int nx , const int ny , const int nz , double *x1 , double *y1, double *z1 , const int nx1 , const double ny1 , const double nz1 , double complex * dens , double complex * dens1 ){

  // Trlinear interpolation: http://en.wikipedia.org/wiki/Trilinear_interpolation

  int i,i1;
  int nxyz = nx * ny * nz;
  int ix,iy,iz;
  int ix1,iy1,iz1;
  int n;
  int ix10,iy10,iz10;
  double xx1,yy1,zz1;

  for(i=0;i<nxyz;i++){
    i2xyz( i , &ix , &iy , &iz , ny , nz );
    ix1=-1;
    for(n=1;n<nx1;n++){
      if(x[ix]<=x1[n]){
	ix1=n;
	break;
      }
    }
    if(ix1<0){
      ix1=0;
      ix10=nx1-1;
      xx1=fabs(x1[0]);
    }else{
      ix10=ix1-1;
      xx1=x1[ix1];
    }
    iy1=-1;
    for(n=1;n<ny1;n++){
      if(y[iy]<=y1[n]){
	iy1=n;
	break;
      }
    }
    if(iy1<0){
      iy1=0;
      iy10=ny1-1;
      yy1=fabs(y1[0]);
    }else{
      iy10=iy1-1;
      yy1=y1[iy1];
    }

    iz1=-1;
    for(n=1;n<nz1;n++){
      if(z[iz]<=z1[n]){
	iz1=n;
	break;
      }
    }
    if(iz1<0){
      iz1=0;
      iz10=nz1-1;
      zz1=fabs(z1[0]);
    }else{
      iz10=iz1-1;
      zz1=z1[iz1];
    }

    //determined now where I am 
    int i000= iz10+nz1*(iy10+ny1*ix10); 
    int i100= iz10+nz1*(iy10+ny1*ix1); 
    int i010= iz10+nz1*(iy1+ny1*ix10); 
    int i001= iz1+nz1*(iy10+ny1*ix10); 
    int i110= iz10+nz1*(iy1+ny1*ix1); 
    int i101= iz1+nz1*(iy10+ny1*ix1); 
    int i011= iz1+nz1*(iy1+ny1*ix10); 
    int i111= iz1+nz1*(iy1+ny1*ix1);

    double xd=(x[ix]-x1[ix10])/(xx1-x1[ix10]);
    double yd=(y[iy]-y1[iy10])/(yy1-y1[iy10]);
    double zd=(z[iz]-z1[iz10])/(zz1-z1[iz10]);

    double complex c00=dens1[i000]*(1.-xd)+dens1[i100]*xd;
    double complex c10=dens1[i010]*(1.-xd)+dens1[i110]*xd;
    double complex c01=dens1[i001]*(1.-xd)+dens1[i101]*xd;
    double complex c11=dens1[i011]*(1.-xd)+dens1[i111]*xd;

    double complex c0=c00*(1.-yd)+c10*yd;
    double complex c1=c01*(1.-yd)+c11*yd;

    dens[i]=c0*(1.-zd)+c1*zd;

  }
}


int write_dens( char * fn , Densities * dens , const MPI_Comm comm , const int iam , const int nx , const int ny , const int nz , double * amu , const double dx , const double dy , const double dz )

{

  int fd , iflag ;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  double * tau , * tau1 , * divjj , * divjj1 ;

  double complex * nu , * nu1 ;

  long int bytes_written ; long int bytestowrite ;

  int nxyz ;

  int i ;

  double dx_dy_dz[3];

  dx_dy_dz[0]=dx;
  dx_dy_dz[1]=dy;
  dx_dy_dz[2]=dz;

  nxyz = nx * ny * nz ;

  iflag = EXIT_SUCCESS ;

  if ( iam == 0 )
    
    {

      //if ( ( i = unlink( ( const char * ) fn ) ) != 0 ) fprintf( stderr , "Cannot unlink() FILE %s\n" , fn ) ;
 
      if ( ( fd = open( fn , O_CREAT | O_WRONLY , fd_mode ) ) == -1 ) 
	
	{

	  fprintf( stderr , "error: cannot open FILE %s for WRITE\n" , fn ) ;
   
	  iflag = EXIT_FAILURE ; 

	}
  
    }

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  MPI_Barrier( comm ) ;

  if( iflag == EXIT_FAILURE ) 

    return( iflag ) ;

  assert( tau1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( divjj1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( nu1 = malloc( nxyz * sizeof( double complex ) ) ) ;

  assert( tau = malloc( nxyz * sizeof( double ) ) ) ;

  assert( divjj = malloc( nxyz * sizeof( double ) ) ) ;

  assert( nu = malloc( nxyz * sizeof( double complex ) ) ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      *( tau1 + i ) = 0. ;

      *( divjj1 + i ) = 0. ;

      *( nu1 + i ) = 0. + I * 0. ;

    }

  for( i = dens->nstart ; i < dens->nstop ; i++ )

    {

      *( tau1 + i ) = *( dens->tau + i - dens->nstart ) ;

      *( divjj1 + i ) = *( dens->divjj + i - dens->nstart ) ;

      *( nu1 + i ) = *( dens->nu + i - dens->nstart ) ;

    }

  MPI_Reduce( tau1 , tau , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( divjj1 , divjj , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( nu1 , nu , nxyz , MPI_DOUBLE_COMPLEX , MPI_SUM , 0 , comm ) ;

  free( nu1 ) ; free( tau1 ) ; free( divjj1 ) ;

  if( iam == 0 )

    {

      bytestowrite = sizeof( int ) ;

      if ( ( bytes_written = ( long ) write( fd , ( const void * ) &nx , ( size_t ) bytestowrite ) ) != bytestowrite )

	{

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      if ( ( bytes_written = ( long ) write( fd , ( const void * ) &ny , ( size_t ) bytestowrite ) ) != bytestowrite )

	{

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      if ( ( bytes_written = ( long ) write( fd , ( const void * ) &nz , ( size_t ) bytestowrite ) ) != bytestowrite )

	{

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      bytestowrite =  ( long ) ( 3 * sizeof( double ) ) ;

      if ( ( bytes_written = ( long ) write( fd , ( const void * ) dx_dy_dz , ( size_t ) bytestowrite ) ) != bytestowrite )

	{

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  iflag = EXIT_FAILURE ;

	}

      bytestowrite =  ( long ) nxyz * sizeof( double ) ;

      if ( ( bytes_written = ( long ) write( fd , ( const void * ) dens->rho , ( size_t ) bytestowrite ) ) != bytestowrite )

	{

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      if ( ( bytes_written = ( long ) write( fd , ( const void * ) tau , ( size_t ) bytestowrite ) ) != bytestowrite )

	{

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      if ( ( bytes_written = ( long ) write( fd , ( const void * ) divjj , ( size_t ) bytestowrite ) ) != bytestowrite )

	{

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      bytestowrite =  ( long ) nxyz * sizeof( double complex ) ;

      if ( ( bytes_written = ( long ) write( fd , ( const void * ) nu , ( size_t ) bytestowrite ) ) != bytestowrite )

	{

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      bytestowrite =  ( long ) ( 5 * sizeof( double ) ) ;

      if ( ( bytes_written = ( long ) write( fd , ( const void * ) amu , ( size_t ) bytestowrite ) ) != bytestowrite )

	{

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  iflag = EXIT_FAILURE ;

	}

      close( fd ) ;

    }

  free( tau ) ; free( nu ) ; free( divjj ) ;

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  return( iflag ) ;

}


int read_constr( char * fn , int nxyz, double * cc_lambda, const int iam , MPI_Comm comm)
{
   int fd , iflag ;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  int i ;


  long int bytes_read ; long int bytestoread ;

  iflag = EXIT_SUCCESS ;

  if ( iam == 0 )
    
    {
      
      /* Check to see if the file already exists, if so exit */
      
      if ( ( fd = open( fn , O_RDONLY , fd_mode ) ) == -1 ) 
	
	{

	  fprintf( stderr , "error: cannot open FILE %s for READ\n" , fn ) ;
   
	  iflag = EXIT_FAILURE ; 

	}
  
    }

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  if( iflag == EXIT_FAILURE ) 

    return( iflag ) ;

  
  


  if(iam == 0)
    {

      bytestoread = ( long ) 3 * sizeof( double ) ;
	
      if ( ( bytes_read = ( long ) read( fd , cc_lambda , ( size_t ) bytestoread ) ) != bytestoread )
	
	{
	  
	  fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;
	  
	  iflag = EXIT_FAILURE ;
	  
	  close( fd ) ;
      
	}

    }

  MPI_Bcast( cc_lambda , 3 , MPI_DOUBLE , 0 , comm ) ;

  return ( iflag );
}

int copy_lattice_arrays_l2s( void * bf1 , void * bf , size_t siz , const int nx1 , const int ny1 , const int nz1 , const int nx , const int ny , const int nz );

int read_dens( char * fn , Densities * dens , const MPI_Comm comm , const int iam , const int nx , const int ny , const int nz , double * amu , const double dx , const double dy , const double dz , char * filename )

{

  int fd , iflag ;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  double * tau , * divjj ;

  double complex * nu ;

  int nx1 , ny1 , nz1 ;

  int nxyz ;

  int i ;

  double dx_dy_dz[3];

  double * buff ; double complex * buffc ;

  long int bytes_read ; long int bytestoread ;

  int nxyz1 ;

  iflag = EXIT_SUCCESS ;

  nxyz = nx * ny * nz ;

  if ( iam == 0 )
    
    {
      
      /* Check to see if the file already exists, if so exit */
      
      if ( ( fd = open( fn , O_RDONLY , fd_mode ) ) == -1 ) 
	
	{

	  fprintf( stderr , "error: cannot open FILE %s for READ\n" , fn ) ;
   
	  iflag = EXIT_FAILURE ; 

	}
  
    }

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  if( iflag == EXIT_FAILURE ) 

    return( iflag ) ;

  assert( tau = malloc( nxyz * sizeof( double ) ) ) ;

  assert( divjj = malloc( nxyz * sizeof( double ) ) ) ;

  assert( nu = malloc( nxyz * sizeof( double complex ) ) ) ;

  if( iam == 0 )

    {

      
      bytestoread = ( long ) sizeof( int ) ;

      if ( ( bytes_read = ( long ) read( fd , &nx1 , ( size_t ) bytestoread ) ) != bytestoread )
	
	{

	  fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      if ( ( bytes_read = ( long ) read( fd , &ny1 , ( size_t ) bytestoread ) ) != bytestoread )

	{

	  fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      if ( ( bytes_read = ( long ) read( fd , &nz1 , ( size_t ) bytestoread ) ) != bytestoread )

	{

	  fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      printf( "nx1=%d ny1=%d nz1=%d\n" , nx1 , ny1 , nz1 ) ;

      bytestoread = (long ) ( 3 * sizeof(double) );

      if ( ( bytes_read = ( long ) read( fd , dx_dy_dz , ( size_t ) bytestoread ) ) != bytestoread )

	{

	  fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      if( nx != nx1 || ny != ny1 || nz != nz1 )

	{

	  for( i = 0 ; i < nxyz ; i++ )

	    {

	      *( dens->rho + i ) = 0. ;

	      *( tau + i ) = 0. ;

	      *( divjj + i ) = 0. ;

	      *( nu + i ) = 0. + 0. * I ;

	    }

	  nxyz1 = nx1 * ny1 * nz1 ;

	  assert( buff = malloc( nxyz1 * sizeof( double ) ) ) ;

	  bytestoread = nxyz1 * sizeof( double ) ;

	  if ( ( bytes_read = ( long ) read( fd , buff , ( size_t ) bytestoread ) ) != bytestoread )

	    {

	      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	      iflag = EXIT_FAILURE ;

	      close( fd ) ;

	    }

	  if (nx >= nx1 && ny>=ny1 && nz >= nz1)
	    copy_lattice_arrays( buff , dens->rho , sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;
	  else if (nx <= nx1 && ny<=ny1 && nz <= nz1)
	    copy_lattice_arrays_l2s( buff , dens->rho , sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;
	  else{

	    fprintf( stderr , "err: nx1 ny1 nz1 should be either larger or smaller than nx ny nz uniformly\n" ) ;
	    
	    iflag = EXIT_FAILURE ;
	    
	    close( fd ) ;

	    }

	  if ( ( bytes_read = ( long ) read( fd , buff , ( size_t ) bytestoread ) ) != bytestoread )

	    {

	      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	      iflag = EXIT_FAILURE ;

	      close( fd ) ;

	    }

	  if (nx >= nx1 && ny>=ny1 && nz >= nz1)
	    copy_lattice_arrays( buff , tau, sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;
	  else if (nx <= nx1 && ny<=ny1 && nz <= nz1)
	    copy_lattice_arrays_l2s( buff , tau , sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;
	  else{

	    fprintf( stderr , "err: nx1 ny1 nz1 should be either larger or smaller than nx ny nz uniformly\n" ) ;
	    
	    iflag = EXIT_FAILURE ;
	    
	    close( fd ) ;

	    }

	  if ( ( bytes_read = ( long ) read( fd , buff , ( size_t ) bytestoread ) ) != bytestoread )

	    {

	      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	      iflag = EXIT_FAILURE ;

	      close( fd ) ;

	    }

	  if (nx >= nx1 && ny>=ny1 && nz >= nz1)
	    copy_lattice_arrays( buff , divjj , sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;
	  else if (nx <= nx1 && ny<=ny1 && nz <= nz1)
	    copy_lattice_arrays_l2s( buff , divjj , sizeof( double ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;
	  else{

	    fprintf( stderr , "err: nx1 ny1 nz1 should be either larger or smaller than nx ny nz uniformly\n" ) ;
	    
	    iflag = EXIT_FAILURE ;
	    
	    close( fd ) ;

	    }

	  free( buff ) ;

	  assert( buffc = malloc( nxyz1 * sizeof( double complex ) ) ) ;
	  
	  bytestoread = nxyz1 * sizeof( double complex ) ;

	  if ( ( bytes_read = ( long ) read( fd , buffc , ( size_t ) bytestoread ) ) != bytestoread )

	    {

	      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	      iflag = EXIT_FAILURE ;

	      close( fd ) ;

	    }

	  if (nx >= nx1 && ny>=ny1 && nz >= nz1)
	    copy_lattice_arrays( buffc , nu , sizeof( double complex) , nx1 , ny1 , nz1 , nx , ny , nz ) ;
	  else if (nx <= nx1 && ny<=ny1 && nz <= nz1)
	    copy_lattice_arrays_l2s( buffc , nu , sizeof( double complex ) , nx1 , ny1 , nz1 , nx , ny , nz ) ;
	  else{

	    fprintf( stderr , "err: nx1 ny1 nz1 should be either larger or smaller than nx ny nz uniformly\n" ) ;
	    
	    iflag = EXIT_FAILURE ;
	    
	    close( fd ) ;

	    }
	  

	  free( buffc ) ;

	}

      else

	{
 
	  bytestoread = nxyz * sizeof( double ) ;

	  if ( ( bytes_read = ( long ) read( fd , dens->rho , ( size_t ) bytestoread ) ) != bytestoread )

	    {

	      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	      iflag = EXIT_FAILURE ;

	      close( fd ) ;

	    }

	  if ( ( bytes_read = ( long ) read( fd , tau , ( size_t ) bytestoread ) ) != bytestoread )

	    {

	      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	      iflag = EXIT_FAILURE ;

	      close( fd ) ;

	    }

	  if ( ( bytes_read = ( long ) read( fd , divjj , ( size_t ) bytestoread ) ) != bytestoread )

	    {

	      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	      iflag = EXIT_FAILURE ;

	      close( fd ) ;

	    }

	  bytestoread = nxyz * sizeof( double complex ) ;

	  if ( ( bytes_read = ( long ) read( fd , nu , ( size_t ) bytestoread ) ) != bytestoread )

	    {

	      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	      iflag = EXIT_FAILURE ;

	      close( fd ) ;

	    }

	}

      bytestoread = 5 * sizeof( double ) ;

      if ( ( bytes_read = ( long ) read( fd , amu , ( size_t ) bytestoread ) ) != bytestoread )

	{

	  fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	  iflag = EXIT_FAILURE ;

	}

      close( fd ) ;

    }

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  if( iflag == EXIT_FAILURE )

    {

      free( tau ) ; free( nu ) ;

      return( EXIT_FAILURE ) ;

    }

  MPI_Bcast( dens->rho , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( tau , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( divjj , nxyz , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( nu , nxyz , MPI_DOUBLE_COMPLEX , 0 , comm ) ;

  MPI_Bcast( amu , 5 , MPI_DOUBLE , 0 , comm ) ;

  for( i = dens->nstart ; i < dens->nstop ; i++ )

    {

      *( dens->tau + i - dens->nstart ) = *( tau + i ) ;

      *( dens->divjj + i - dens->nstart ) = *( divjj + i ) ;

      *( dens->nu + i - dens->nstart ) = *( nu + i ) ;

    }

  free( tau ) ; free( nu ) ; free( divjj ) ;

  return( EXIT_SUCCESS ) ;

}

int copy_lattice_arrays( void * bf1 , void * bf , size_t siz , const int nx1 , const int ny1 , const int nz1 , const int nx , const int ny , const int nz )

{

  int ixyz1 = 0 , ixyz = 0 , ix , iy , iz , nx_start , nx_stop , ny_start , ny_stop , nz_start , nz_stop ;

  nx_start = ( nx - nx1 ) / 2 ;

  nx_stop  = ( nx + nx1 ) / 2 ;

  ny_start = ( ny - ny1 ) / 2 ;

  ny_stop  = ( ny + ny1 ) / 2 ;

  nz_start = ( nz - nz1 ) / 2 ;

  nz_stop  = ( nz + nz1 ) / 2 ;

  for( ix = 0 ; ix < nx1 ; ix++)

    for( iy = 0 ; iy < ny1 ; iy++ )

      for( iz = 0 ; iz < nz1 ; iz++ )

	{

	  ixyz = iz + nz_start + nz * ( iy + ny_start + ny * ( ix + nx_start ) ) ;

	  ixyz1 = iz + nz1 * ( iy + ny1 * ix ) ;

	  memcpy( bf + ( size_t ) ixyz * siz , bf1 + ( size_t ) ixyz1 * siz , siz ) ;

	}

  return ( EXIT_SUCCESS ) ;
  
}

// used for problems that cut densities in larger lattice into smaller
int copy_lattice_arrays_l2s( void * bf1 , void * bf , size_t siz , const int nx1 , const int ny1 , const int nz1 , const int nx , const int ny , const int nz )

{

  int ixyz1 = 0 , ixyz = 0 , ix , iy , iz , nx_start , nx_stop , ny_start , ny_stop , nz_start , nz_stop ;

  nx_start = ( nx1 - nx ) / 2 ;

  nx_stop  = ( nx1 + nx ) / 2 ;

  ny_start = ( ny1 - ny ) / 2 ;

  ny_stop  = ( ny1 + ny ) / 2 ;

  nz_start = ( nz1 - nz ) / 2 ;

  nz_stop  = ( nz1 + nz ) / 2 ;

  for( ix = 0 ; ix < nx ; ix++)

    for( iy = 0 ; iy < ny ; iy++ )

      for( iz = 0 ; iz < nz ; iz++ )

	{

	  ixyz1 = iz + nz_start + nz1 * ( iy + ny_start + ny1 * ( ix + nx_start ) ) ;

	  ixyz = iz + nz * ( iy + ny * ix ) ;

	  memcpy( bf + ( size_t ) ixyz * siz , bf1 + ( size_t ) ixyz1 * siz , siz ) ;

	}

  return ( EXIT_SUCCESS ) ;
  
}



int write_dens_txt(FILE *fd, Densities * dens, const MPI_Comm comm, const int iam, const int nx, const int ny, const int nz, double * amu)
{
  double * tau, * tau1, * divjj, *divjj1;

  double complex * nu, * nu1;

  int iflag = EXIT_SUCCESS;

  int i;

  int nxyz = nx*ny*nz;

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  MPI_Barrier( comm ) ;

  if( iflag == EXIT_FAILURE ) 

    return( iflag ) ;

  assert( tau1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( divjj1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( nu1 = malloc( nxyz * sizeof( double complex ) ) ) ;

  assert( tau = malloc( nxyz * sizeof( double ) ) ) ;

  assert( divjj = malloc( nxyz * sizeof( double ) ) ) ;

  assert( nu = malloc( nxyz * sizeof( double complex ) ) ) ;
  
  for( i = 0 ; i < nxyz ; i++ )

    {

      *( tau1 + i ) = 0. ;

      *( divjj1 + i ) = 0. ;

      *( nu1 + i ) = 0. + I * 0. ;

    }

  for( i = dens->nstart ; i < dens->nstop ; i++ )

    {

      *( tau1 + i ) = *( dens->tau + i - dens->nstart ) ;

      *( divjj1 + i ) = *( dens->divjj + i - dens->nstart ) ;

      *( nu1 + i ) = *( dens->nu + i - dens->nstart ) ;

    }

  MPI_Reduce( tau1 , tau , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( divjj1 , divjj , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( nu1 , nu , nxyz , MPI_DOUBLE_COMPLEX , MPI_SUM , 0 , comm ) ;

  free( nu1 ) ; free( tau1 ) ; free( divjj1 ) ;

  if(iam == 0)
    {
      for(i=0;i<nxyz;i++)
	fprintf(fd, "rho[%d] = %.12le\n", i, dens->rho[i]);
      
      for(i=0;i<nxyz;i++)
	fprintf(fd, "nu[%d] = %.12le %12leI\n", i, creal(nu[i]), cimag(nu[i]));
      
      for(i=0;i<nxyz;i++)
	fprintf(fd, "tau[%d] = %.12le\n", i, tau[i]);
      
      for(i=0;i<nxyz;i++)
	fprintf(fd, "divjj[%d] = %.12le\n", i, divjj[i]);
    }

  return(iflag);
  
}


int write_qpe( char * fn, double * lam, const MPI_Comm comm, const int iam, const int nwf)

{
  int fd , iflag ;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  long int bytes_written ; long int bytestowrite ;

  iflag = EXIT_SUCCESS ;

  

  if ( iam == 0 )
    
    {

      if ( ( fd = open( fn , O_CREAT | O_WRONLY , fd_mode ) ) == -1 ) 
	
	{

	  fprintf( stderr , "error: cannot open FILE %s for WRITE\n" , fn ) ;
   
	  iflag = EXIT_FAILURE ; 

	}
  
    }

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  MPI_Barrier( comm ) ;

  if( iflag == EXIT_FAILURE ) 

    return( iflag ) ;


  if( iam == 0 )

    {      

      bytestowrite = nwf * sizeof( double ) ;

      if ( ( bytes_written = ( long ) write( fd , ( const void * ) lam , ( size_t ) bytestowrite ) ) != bytestowrite )

	{

	  fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}

      
      close( fd ) ;

    }

  return(iflag);
  
}
