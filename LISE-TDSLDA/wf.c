// for license information, see the accompanying LICENSE file

#include <stdlib.h>

#include <getopt.h>

#include <stdio.h>

#include <math.h>

#include <assert.h>

#include <string.h>

#include <sys/types.h>

#include <sys/stat.h>

#include <fcntl.h>

#include <time.h> 

#include <unistd.h> 

#include <string.h>

#include <mpi.h>

#include "vars.h"

#define MAX_REC_LEN 1024

Wfs * allocate_phys_mem_wf( const int nwfip , const int nxyz )

{

  int n , i , nxyz_4 = 4 * nxyz ;

  Wfs * wavef ;

  assert( wavef = ( Wfs * ) malloc( sizeof( Wfs ) ) ) ;

  if( nwfip == 0 ) 

    return( wavef ) ;

  assert( wavef->wavf_modifier = malloc( nwfip * sizeof( double complex * ) ) ) ;

  assert( wavef->deriv_x = malloc( nwfip * sizeof( double complex * ) ) ) ;

  assert( wavef->deriv_y = malloc( nwfip * sizeof( double complex * ) ) ) ;

  assert( wavef->deriv_z = malloc( nwfip * sizeof( double complex * ) ) ) ;

  for( i = 0 ; i < 2 ; i++ )

    {

      assert( wavef->wavf[ i ] = ( double complex ** ) malloc( nwfip * sizeof( double complex * ) ) ) ;

      assert( wavef->wavf_predictor[ i ] = ( double complex ** ) malloc( nwfip * sizeof( double complex * ) ) ) ;

      assert( wavef->wavf_corrector[ i ] = ( double complex ** ) malloc( nwfip * sizeof( double complex * ) ) ) ;

      for( n = 0 ; n < nwfip ; n++ )

	{

	  assert( wavef->wavf[ i ][ n ] = ( double complex * ) malloc( nxyz_4 * sizeof( double complex ) ) ) ;

	  assert( wavef->wavf_predictor[ i ][ n ] = malloc( nxyz_4 * sizeof( double complex ) ) ) ;

	  assert( wavef->wavf_corrector[ i ][ n ] = malloc( nxyz_4 * sizeof( double complex ) ) ) ;

	}

    }

  for( i = 0 ; i < 4 ; i++ )

    {

      assert( wavef->wavf_t_der[ i ] = malloc( nwfip * sizeof( double complex * ) ) ) ;

      for( n = 0 ; n < nwfip ; n++ )

	assert( wavef->wavf_t_der[ i ][ n ] = malloc( nxyz_4 * sizeof( double complex ) ) ) ;

    }

  for( n = 0 ; n < nwfip ; n++ )

    {

      assert( wavef->wavf_modifier[ n ] = malloc( nxyz_4 * sizeof( double complex ) ) ) ;

      assert( wavef->deriv_x[ n ] = malloc( nxyz_4 * sizeof( double complex ) ) ) ;

      assert( wavef->deriv_y[ n ] = malloc( nxyz_4 * sizeof( double complex ) ) ) ;

      assert( wavef->deriv_z[ n ] = malloc( nxyz_4 * sizeof( double complex ) ) ) ;

    }

  return ( wavef ) ;

}

void free_phys_mem_wf( const int nwfip , Wfs * wavef )

{

  int n , i ;

  if( nwfip == 0 ) 

    {

      free( wavef->wavf ) ;

      return ;

    }

  for( i = 0 ; i < 2 ; i++ )

    {

      for( n = 0 ; n < nwfip ; n++ )

	{

	  free( wavef->wavf[ i ][ n ] ) ;

	  free( wavef->wavf_predictor[ i ][ n ] ) ;

	  free( wavef->wavf_corrector[ i ][ n ] ) ;

	}

      free( wavef->wavf[ i ] ) ;

      free( wavef->wavf_predictor[ i ] ) ;

      free( wavef->wavf_corrector[ i ] ) ;

    }

  for( i = 0 ; i < 4 ; i++ )

    {

      for( n = 0 ; n < nwfip ; n++ )

	free( wavef->wavf_t_der[ i ][ n ] ) ;

      free( wavef->wavf_t_der[ i ] ) ;

    }

  for( n = 0 ; n < nwfip ; n++ )

    {

      free( wavef->wavf_modifier[ n ] ) ;

      free( wavef->deriv_x[ n ] ) ;

      free( wavef->deriv_y[ n ] ) ;

      free( wavef->deriv_z[ n ] ) ;

    }

  free( wavef->wavf_modifier ) ;

  free( wavef->deriv_x ) ;

  free( wavef->deriv_y ) ;

  free( wavef->deriv_z ) ;

}

int read_wf( double complex ** wf , const int nxyz , const int nwf , const int ip , const int np , const MPI_Comm comm , char * fn , double dxyz , int *wavef_index) 

{

  int ip_c , np_c , nwfip ;

  int n , i , ii , il , iu , iflag ;

  int nxyz_4 = 4 * nxyz ;

  double complex * wf_read ;

  long int bytes_read ;

  long int bytestoread ;

  int fd ;

  double nrm , norm ;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  ip_c = ip - 1 ;

  np_c = np - 1 ;

  nwfip = nwf / np_c ;

  iflag = 0 ;

  if( ip > 0 )

    {

      iu = 0 ;

      for( i = 0 ; i < ip ; i++ )

	{

	  il = iu ;

	  if( i < nwf % np_c )

	    iu = il + nwfip + 1 ;

	  else

	    iu = il + nwfip ;

	}

    }

  else

    {

      iu = -1 ;

      il = -1 ;

      if ( ( fd = open( fn , O_RDONLY , fd_mode ) ) == -1 ) 
	
	{

	  fprintf( stderr , "error: cannot open FILE %s for READ\n" , fn ) ;
   
	  iflag = 1 ; 

	}
  
    }

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  if( iflag == 1 ) 

    return( 1 ) ;

  assert( wf_read = malloc( nxyz_4 * sizeof( double complex ) ) ) ;

  for( n = 0 ; n < nwf ; n++ )

    {

      if( ip == 0 ) 

	{

	  bytestoread = ( long ) nxyz_4 * sizeof( double complex ) ; 
  
	  if ( ( long ) ( bytes_read = read( fd , ( void * ) wf_read , ( size_t ) bytestoread ) ) != bytestoread ) 

	    {
      
	      fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;
      
	      iflag = 1 ;
      
	    }
  
	}

      MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

      if( iflag == 1 ) 

	break ;

      MPI_Bcast( wf_read , nxyz_4 , MPI_DOUBLE_COMPLEX , 0 , comm ) ;

      if( il <= n && n < iu )

	{ 

	  ii = n - il ;

	  norm = 0. ;

	  wavef_index[ii] = n;

	  for( i = 0 ; i < nxyz_4 ; i++ )

	    {

	      wf[ ii ][ i ] = wf_read[ i ] ;

	      norm += creal( conj( wf[ ii ][ i ] ) * wf[ ii ][ i ] )  ;

	    }

	}

    }

  free( wf_read ) ;

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  return( iflag ) ;

}

int write_wf(char *fn, int nwf, int nxyz, const int ip , const int np , const MPI_Comm comm , MPI_Status * status, double complex **wf, int * wf_tbl)
{
  int i, iwf, n;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  int fd , ii, il, iu, iflag ;

  int nwfip;

  long int bytes_written ; long int bytestowrite ;

  double complex *wf_write;

  int nxyz_4 = 4*nxyz;

  int tag = 10;

  int ip_c, np_c;
  
  ip_c = ip - 1 ;

  np_c = np - 1 ;

  nwfip = nwf / np_c ;

  iflag = 0 ;

  assert(wf_write = (double complex *) malloc(nxyz_4*sizeof(double complex)));

  int *wf_tbl_lb;
  assert(wf_tbl_lb =  (int *) malloc(np_c * sizeof(int)));
  
  int nwf_max = 0;
  
  wf_tbl_lb[0] = 0;
  
  for(i=1; i<np_c; i++)
    wf_tbl_lb[i] = wf_tbl_lb[i-1]+wf_tbl[i-1];


   

  if( ip > 0 )

    {

      iu = 0 ;

      for( i = 0 ; i < ip ; i++ )

	{

	  il = iu ;
	  
	  if( i < nwf % np_c )

	    iu = il + nwfip + 1 ;

	  else

	    iu = il + nwfip ;

	}

    }

  else

    {

      iu = -1 ;

      il = -1 ;

      if ( ( fd = open( fn , O_CREAT|O_WRONLY , fd_mode ) ) == -1 ) 
	
	{

	  fprintf( stderr , "error: cannot open FILE %s for READ\n" , fn ) ;
   
	  iflag = 1 ; 

	}
  
    }

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;



  if( iflag == 1 ) 

    return( 1 ) ;


  for( n = 0 ; n < nwf ; n++ )

    {


      if( il <= n && n < iu )
	
	{ 

	  ii = n - il ;

	  
	  MPI_Send( wf[ii] , nxyz_4 , MPI_DOUBLE_COMPLEX , 0, tag, comm ) ;
	}
      
    


      if( ip == 0 ) 
    
	{
	  // find the source
	  
	  for(i=0; i<np_c; i++){
	    
	    if(wf_tbl_lb[i] > n) break;
	    
	  }
	  
	  MPI_Recv(wf_write, nxyz_4, MPI_DOUBLE_COMPLEX, i, tag, comm, status);
	  
	  bytestowrite = ( long ) nxyz_4 * sizeof( double complex ) ; 
	  
	  if ( ( long ) ( bytes_written = write( fd , ( void * ) wf_write , ( size_t ) bytestowrite ) ) != bytestowrite ) 
	    
	    {
	      
	      fprintf( stderr , "err: failed to WRITE %ld bytes\n" , bytestowrite ) ;
	      
	      iflag = 1 ;
	  
	    }
	  
	}
      
    }
  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;
  
    
  

  close(fd);
  
  free( wf_write ) ;

  free(wf_tbl_lb);

  return( iflag ) ;
}


void initialize_wfs( Wfs * wavef , const int nwfip , const int nxyz ) 

{

  int n , i , j ;

  int nxyz_4 = 4 * nxyz ;
  
  double norm;

  for( n = 0 ; n < nwfip ; n++ )

    {

      norm = 0.;

      for( i = 0 ; i < nxyz_4 ; i++ )

	{

	  wavef->wavf[ 1 ][ n ][ i ] = wavef->wavf[ 0 ][ n ][ i ] ;

	  norm += pow( cabs( wavef->wavf[0][ n ][ i ] ) , 2 ) ;

	  for( j = 0 ; j < 2 ; j++ )

	    {

	      wavef->wavf_predictor[ j ][ n ][ i ] = wavef->wavf[ 0 ][ n ][ i ] ;

	      wavef->wavf_corrector[ j ][ n ][ i ] = wavef->wavf[ 0 ][ n ][ i ] ;

	    }

	  for( j = 0 ; j < 4 ; j++ )

	    wavef->wavf_t_der[ j ][ n ][ i ] = 0. + 0. * I ;

	}

      //      printf( "init: iwf=%d norm=%g\n" , n , norm ) ;

    }

}

void initialize_wfs_swap( Wfs * wavef , const int nwfip , const int nxyz ) 

{

  int n , i , j ;

  int nxyz_4 = 4 * nxyz ;

  int nxyz2 = 2 * nxyz ;

  double sign ;

  for( n = 0 ; n < nwfip ; n++ )

    {

      sign = - 1 ;

      for( i = 0 ; i < nxyz_4 ; i++ )

	{

	  wavef->wavf[ 1 ][ n ][ ( i + nxyz2 ) % nxyz_4 ] = sign * conj( wavef->wavf[ 0 ][ n ][ i ] ) ;

	  if( i == nxyz2 - 1 ) 

	    sign = 1. ;

	}

      for( i = 0 ; i < nxyz_4 ; i++ )

	{

	  wavef->wavf[ 0 ][ n ][ i ] = wavef->wavf[ 1 ][ n ][ i ] ;

	  for( j = 0 ; j < 2 ; j++ )

	    {

	      wavef->wavf_predictor[ j ][ n ][ i ] = wavef->wavf[ 1 ][ n ][ i ] ;

	      wavef->wavf_corrector[ j ][ n ][ i ] = wavef->wavf[ 1 ][ n ][ i ] ;

	    }

	  for( j = 0 ; j < 4 ; j++ )

	    wavef->wavf_t_der[ j ][ n ][ i ] = 0. + 0. * I ;

	}

    }

}

void read_input_solver( char * dir_in , int * nx , int * ny , int * nz , int * nwf_p , int * nwf_n , double * amu_p , double * amu_n , double * dx , double * dy , double * dz , double * e_cut , double * cc_constr , const int ip , const MPI_Comm comm )

{

  char * fn ;

  FILE * fd ;
  int i;

  if( ip == 0 )

    {
      
      fn = malloc( 130 * sizeof( char ) ) ;

      sprintf( fn , "%s/info.slda_solver" , dir_in ) ;

      fd = fopen( fn , "rb" ) ;

      fread( nwf_p , sizeof( int ) , 1 , fd ) ;

      fread( nwf_n , sizeof( int ) , 1 , fd ) ;

      fread( amu_p , sizeof( double ) , 1 , fd ) ;

      fread( amu_n , sizeof( double ) , 1 , fd ) ;

      fread( dx , sizeof( double ) , 1 , fd ) ;

      fread( dy , sizeof( double ) , 1 , fd ) ;

      fread( dz , sizeof( double ) , 1 , fd ) ;

      fread( nx , sizeof( int ) , 1 , fd ) ;

      fread( ny , sizeof( int ) , 1 , fd ) ;

      fread( nz , sizeof( int ) , 1 , fd ) ;

      fread( e_cut , sizeof( double ) , 1 , fd ) ;

      fread( cc_constr , sizeof( double ) , 4 , fd );

      fclose( fd ) ;

      free( fn ) ;

    }

  MPI_Bcast( nwf_p , 1 , MPI_INT , 0 , comm ) ;

  MPI_Bcast( nwf_n , 1 , MPI_INT , 0 , comm ) ;

  MPI_Bcast( nx , 1 , MPI_INT , 0 , comm ) ;

  MPI_Bcast( ny , 1 , MPI_INT , 0 , comm ) ;

  MPI_Bcast( nz , 1 , MPI_INT , 0 , comm ) ;

  MPI_Bcast( amu_p , 1 , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( amu_n , 1 , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( dx , 1 , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( dy , 1 , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( dz , 1 , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( e_cut , 1 , MPI_DOUBLE , 0 , comm ) ;

  MPI_Bcast( cc_constr , 4 , MPI_DOUBLE , 0 , comm ) ;

}

void check_time_reversal_symmetry(int knxyz, double complex **wavef, int nwfip, int *wavef_index, double * norms_tr, int * labels_tr, double dxyz)
/*
  check the time-reversal(TR) symmetry in the wfs
  The idea is: the spectrum is always double degenerate from the very beginning.
  Then for two wavefunctions with index 2i-1 and 2i, i=1,....., they are two-degenerate 
  states, \phi1, \phi2, then |<\phi1 | -i\sigma_y \phi2^*>| should be 1
 */
{
  int i=0, j;
  int ind1, ind2;
  int nxyz = knxyz / 4;
  ind1 = wavef_index[0];
  int imod = ind1 % 2;

  double complex *buf_nxyz4;
  
  assert(buf_nxyz4 = (double complex *)malloc(knxyz*sizeof(double complex)));

  double complex tmp, tmp1;
  // start index is even, search its next neighborhood to see if adjacent in index
  int ipair = 0;
  
  
  for (i=0; i<nwfip-1;i++)
    {
      ind1 = wavef_index[i];
      ind2 = wavef_index[i+1];
          
      if( ((ind1 % 2) == 0) && ((ind2 % 2) == 1) && (ind2 == ind1+1))
	{
	  // check TR symmetry
	  for(j=0; j<nxyz; j++)
	    {
	      buf_nxyz4[j] = -conj(wavef[i+1][j+nxyz]);
	      buf_nxyz4[j+nxyz] = conj(wavef[i+1][j]);
	      buf_nxyz4[j+2*nxyz] = -conj(wavef[i+1][j+3*nxyz]);
	      buf_nxyz4[j+3*nxyz] = conj(wavef[i+1][j+2*nxyz]); 
	    }

	  tmp = 0.0 + I*0.0; tmp1 = 0.0 + 0.0*I;
	  for(j=0; j<knxyz; j++) tmp += conj(wavef[i][j])*buf_nxyz4[j]*dxyz;
	  for(j=0; j<knxyz; j++) tmp1 += conj(wavef[i][j])*wavef[i][j]*dxyz;
	  
	  norms_tr[ind1] = sqrt(creal(tmp)*creal(tmp) + cimag(tmp)*cimag(tmp))/
          sqrt(creal(tmp1)*creal(tmp1) + cimag(tmp1)*cimag(tmp1));
	  norms_tr[ind2] = norms_tr[ind1];
	  ipair+=1;
	}
       
    }
  
  free(buf_nxyz4);
}


int parse_input_file( char * dir_in , int * nx , int * ny , int * nz , int * nwf_p , int * nwf_n , double * amu_p , double * amu_n , double * dx , double * dy , double * dz , double * e_cut , double * cc_constr , const int ip , const MPI_Comm comm )
{
    FILE *fp;

    char *fn = malloc( 130 * sizeof( char ) ) ;

    sprintf( fn , "%s/input.test.txt" , dir_in ) ;
      


    if (ip == 0){
      fp=fopen(fn, "r");


      if(fp==NULL)
        return 0;
      
      int i;
        
      char s[MAX_REC_LEN];
      char tag[MAX_REC_LEN];
      char ptag[MAX_REC_LEN];
    
      while(fgets(s, MAX_REC_LEN, fp) != NULL)
	{
	  // Read first element of line
	  tag[0]='#'; tag[1]='\0';
	  sscanf (s,"%s %*s",tag);
	  
	  // Loop over known tags;
	  if(strcmp (tag,"#") == 0)
            continue;
	  else if (strcmp (tag,"nx") == 0)
            sscanf (s,"%s %d %*s",tag,nx);
	  
	  else if (strcmp (tag,"ny") == 0)
            sscanf (s,"%s %d %*s",tag,ny);
	  
	  else if (strcmp (tag,"nz") == 0)
            sscanf (s,"%s %d %*s",tag,nz);
	  
	  else if (strcmp (tag,"dx") == 0)
            sscanf (s,"%s %lf %*s",tag,dx);
	  
	  else if (strcmp (tag,"dy") == 0)
            sscanf (s,"%s %lf %*s",tag,dy);
	  
	  else if (strcmp (tag,"dz") == 0)
            sscanf (s,"%s %lf %*s",tag,dz);
	}

      *amu_p = 0.;
      *amu_n = 0.;

      *nwf_p = 2*(*nx)*(*ny)*(*nz);
      *nwf_n = *nwf_p;
      *e_cut = 0.;

      for(i=0; i<4; i++) cc_constr[i] = 0.;
    
      fclose(fp);
    }

    MPI_Bcast( nwf_p , 1 , MPI_INT , 0 , comm ) ;

    MPI_Bcast( nwf_n , 1 , MPI_INT , 0 , comm ) ;
    
    MPI_Bcast( nx , 1 , MPI_INT , 0 , comm ) ;
    
    MPI_Bcast( ny , 1 , MPI_INT , 0 , comm ) ;
    
    MPI_Bcast( nz , 1 , MPI_INT , 0 , comm ) ;
    
    MPI_Bcast( amu_p , 1 , MPI_DOUBLE , 0 , comm ) ;
    
    MPI_Bcast( amu_n , 1 , MPI_DOUBLE , 0 , comm ) ;
    
    MPI_Bcast( dx , 1 , MPI_DOUBLE , 0 , comm ) ;
    
    MPI_Bcast( dy , 1 , MPI_DOUBLE , 0 , comm ) ;
    
    MPI_Bcast( dz , 1 , MPI_DOUBLE , 0 , comm ) ;
    
    MPI_Bcast( e_cut , 1 , MPI_DOUBLE , 0 , comm ) ;
    
    MPI_Bcast( cc_constr , 4 , MPI_DOUBLE , 0 , comm ) ;

  
    return 1;
}



void generate_plane_wave_wfs( double complex ** wf , const int nxyz , const int nwf , const int ip , const int np , const MPI_Comm comm , char * fn , double dxyz , int *wavef_index , Lattice_arrays * latt) 

{

  int ip_c , np_c , nwfip ;

  int n , i , ii , il , iu , iflag ;

  double kx, ky, kz;

  int ixyz;

  int nxyz_4 = 4 * nxyz ;

  double complex * wf_read ;

  int fd ;

  double nrm , norm ;

  ip_c = ip - 1 ;

  np_c = np - 1 ;

  nwfip = nwf / np_c ;

  iflag = 0 ;

  if( ip > 0 )

    {

      iu = 0 ;

      for( i = 0 ; i < ip ; i++ )

	{

	  il = iu ;

	  if( i < nwf % np_c )

	    iu = il + nwfip + 1 ;

	  else

	    iu = il + nwfip ;

	}

    }

  else

    {

      iu = -1 ;

      il = -1 ;
      
    }

  MPI_Bcast( &iflag , 1 , MPI_INT , 0 , comm ) ;

  assert( wf_read = malloc( nxyz_4 * sizeof( double complex ) ) ) ;

  for( n = 0 ; n < nwf ; n++ )

    
    {

      if( ip == 0 ) 

	{
	  
	  kx = latt->kx[n%nxyz];
	  ky = latt->ky[n%nxyz];
	  kz = latt->kz[n%nxyz];
	  for(ixyz = 0; ixyz< nxyz; ixyz++){
	    wf_read[ixyz] = cexp( (kx*latt->xa[ixyz]+ky*latt->ya[ixyz]+kz*latt->za[ixyz] )*I )/sqrt((double)nxyz*4.0); // u_up
	    wf_read[ixyz+nxyz] = cexp( (kx*latt->xa[ixyz]+ky*latt->ya[ixyz]+kz*latt->za[ixyz] )*I )/sqrt((double)nxyz*4.0); // u_down
	    wf_read[ixyz+2*nxyz] = cexp( -(kx*latt->xa[ixyz]+ky*latt->ya[ixyz]+kz*latt->za[ixyz] )*I)/sqrt((double)nxyz*4.0); // v_up
	    wf_read[ixyz+3*nxyz] = cexp( -(kx*latt->xa[ixyz]+ky*latt->ya[ixyz]+kz*latt->za[ixyz] )*I)/sqrt((double)nxyz*4.0); // v_down	    
	  }
  
	}

      MPI_Bcast( wf_read , nxyz_4 , MPI_DOUBLE_COMPLEX , 0 , comm ) ;

      if( il <= n && n < iu )

	{ 

	  ii = n - il ;

	  norm = 0. ;

	  wavef_index[ii] = n;

	  for( i = 0 ; i < nxyz_4 ; i++ )

	    {

	      wf[ ii ][ i ] = wf_read[ i ] ;

	      norm += creal( conj( wf[ ii ][ i ] ) * wf[ ii ][ i ] )  ;

	    }


	}

    }

  free( wf_read ) ;

}



int read_wf_MPI( double complex ** wf , const int nxyz , const int nwf , const int ip , const int np , const MPI_Comm comm , char * fn , double dxyz , int *wavef_index, int * wf_tbl)

{
  int np_c ;

  int n , i , j, k, iflag ;

  int nxyz_4 = 4 * nxyz ;

  double complex * wf_read ;

  long int bytes_read ;

  long int bytestoread ;

  MPI_File fh ;

  double nrm , norm ;

  MPI_Status mpi_st ;
  int itag = 11 ;

  int itgt;

  np_c = np - 1 ;

  iflag = 0;

  iflag = MPI_File_open(comm,fn,MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  if( iflag == 1 )

    return( 1 ) ;

  MPI_Offset offset = 0 ;
  int offset_int = 0;

  for (i = 1; i < ip; i++){
      offset = offset + wf_tbl[i-1];
      offset_int = offset_int + wf_tbl[i-1];
  }

  offset = offset * ( nxyz_4*sizeof(double complex) );

  if( ip>0){

   wf_read = malloc( wf_tbl[ip-1] * nxyz_4 * sizeof( double complex ) ) ;

   MPI_File_read_at(fh, offset, wf_read, wf_tbl[ip-1]*nxyz_4, MPI_DOUBLE_COMPLEX,&mpi_st);

   k = 0;

    for (j=0 ; j< wf_tbl[ip - 1]; j++){
      for(i=0; i<nxyz_4; i++){
        wf[j][i] = wf_read[k];
	wavef_index[j]=j+offset_int; // IBRAHIM.  Needed correction. 
        k++;
      }
    }
 }

  MPI_Bcast(&iflag,1,MPI_INT,np-1,comm);

  MPI_File_close( &fh ) ;

  if (ip>0) free (wf_read);

  return( iflag ) ;

}

int write_wf_MPI(char *fn, int nwf, int nxyz, const int ip , const int np , const MPI_Comm comm , MPI_Status * status, double complex **wf, int * wf_tbl)
{
  int i, j;

  double complex *wf_write;

  int nxyz_4 = 4*nxyz;

  MPI_File fh;

  MPI_Offset offset = 0 ;

  for (i = 1; i < ip; i++){
      offset = offset + wf_tbl[i-1];
  }
  offset = offset * ( nxyz_4*sizeof(double complex) );

  int iflag = MPI_File_open( comm, fn, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, & fh );

  if(ip==0 && iflag!=0)
    printf( "Error: File %s could not be open \n", fn);

  if( iflag == 1 )

    return( 1 ) ;

  if(ip>0){
    wf_write = (double complex *) malloc(wf_tbl[ip-1]*nxyz_4*sizeof(double complex));
    int k=0;
    for (j=0 ; j< wf_tbl[ip - 1]; j++){
      for(i=0; i<nxyz_4; i++){
        wf_write[k]=wf[j][i] ;
        k++;
      }
    }
  }

  if(ip>0)
    iflag=MPI_File_write_at(fh, offset, wf_write, wf_tbl[ip - 1]*nxyz_4, MPI_DOUBLE_COMPLEX,status);
  MPI_Bcast(&iflag,1,MPI_INT,np-1,comm);

  MPI_File_close( &fh ) ;

  if(ip>0)
    free( wf_write ) ;

  return( iflag ) ;
}
