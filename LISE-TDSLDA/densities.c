// for license information, see the accompanying LICENSE file

#include <assert.h>

#include <math.h>

#include <mpi.h>

#include "vars.h"

#include <stdlib.h>

#include <stdio.h>

#include "tdslda_func.h"

#include <sys/types.h>

#include <sys/stat.h>

#include <fcntl.h>

#include <unistd.h>

#include <cufft.h>

void diverg_real( double * f , double * fx , double * fy , double * fz , FFtransf_vars * fftrans , Lattice_arrays * latt , const int nxyz );

int read_dens(char *fn, cufftDoubleReal * copy_from, const MPI_Comm comm, const int iam, const int nxyz)
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

  if( iam == 0 )

    {

      
      bytestoread = ( long ) sizeof( cufftDoubleReal )*14*nxyz ;

      if ( ( bytes_read = ( long ) read( fd , copy_from , ( size_t ) bytestoread ) ) != bytestoread )
	
	{

	  fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}
    }

  return(iflag);
}


int read_qpe(char *fn, cufftDoubleReal * e_qp, const MPI_Comm comm, const int iam, const int nwf)
{
  int fd , iflag ;

  mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */

  int i ;

  long int bytes_read ; long int bytestoread ;

  iflag = EXIT_SUCCESS ;

  //nxyz = nx * ny * nz ;

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
  
  if( iam == 0 )

    {
      
      bytestoread = ( long ) sizeof( double )*nwf ;

      if ( ( bytes_read = ( long ) read( fd , e_qp , ( size_t ) bytestoread ) ) != bytestoread )
	
	{

	  fprintf( stderr , "err: failed to READ %ld bytes\n" , bytestoread ) ;

	  iflag = EXIT_FAILURE ;

	  close( fd ) ;

	}
    }

  MPI_Barrier(comm);

  MPI_Bcast(e_qp, nwf, MPI_DOUBLE, 0, comm);

  return(iflag);
  
}

void allocate_phys_mem_dens( Densities * dens , const int nxyz )

{

  int i ;

  assert( dens->rho = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->tau = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->divjj = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->jx = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->jy = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->jz = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->cjx = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->cjy = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->cjz = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->sx = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->sy = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->sz = malloc( nxyz * sizeof( double ) ) ) ;

  assert( dens->nu = malloc( nxyz * sizeof( double complex ) ) ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      dens->rho[ i ] = 0. ;

      dens->tau[ i ] = 0. ;

      dens->divjj[ i ] = 0. ;

      dens->jx[ i ] = 0. ;

      dens->jy[ i ] = 0. ;

      dens->jz[ i ] = 0. ;

      dens->cjx[ i ] = 0. ;

      dens->cjy[ i ] = 0. ;

      dens->cjz[ i ] = 0. ;

      dens->sx[ i ] = 0. ;

      dens->sy[ i ] = 0. ;

      dens->sz[ i ] = 0. ;

      dens->nu[ i ] = 0. + 0. * I ;

    }

}

void compute_densities( const int nxyz , const int nwfip , double complex ** wavf_in , double complex ** deriv_x , double complex ** deriv_y , double complex ** deriv_z , FFtransf_vars * fftrans , Densities * dens , Lattice_arrays * latt , const MPI_Comm comm , double ** a_vf , const int ip )

{

  int i , n , k , kshift ;

  double complex * buff ;

  double complex * nu ;

  double * buff2 , * arho;

  int nxyz2 = 2 * nxyz , nxyz3 = 3 * nxyz , nxyz4 = 4 * nxyz ;

  assert( nu = malloc( nxyz * sizeof( double complex ) ) ) ;

  for( i = 0 ; i < nxyz ; i++ )

    nu[ i ] = 0. + 0. * I ;

  if( nwfip > 0 )

    assert( buff = malloc( nxyz * sizeof( double complex ) ) ) ;

  for( n = 0 ; n < nwfip ; n++ )  { /* calculate derivatives first */

    for( k = 0 ; k < 4 ; k++){

      kshift = k * nxyz ;

      gradient( wavf_in[ n ] + kshift , deriv_x[ n ] + kshift , deriv_y[ n ] + kshift , deriv_z[ n ] + kshift , fftrans , latt , nxyz , buff ) ;

    } 

    /* take complex conjugate of v */

    for( i = nxyz2 ; i < nxyz4 ; i++ ) {

      wavf_in[ n ][ i ] = conj( wavf_in[ n ][ i ] ) ;

    }

    for( i = 0 ; i < nxyz ; i++ )

      nu[ i ] += ( wavf_in[ n ][ i ] * wavf_in[ n ][ i + nxyz3 ] ) ;

  }

  if( nwfip > 0 )

    free( buff ) ;

  MPI_Reduce( nu , dens->nu , nxyz , MPI_DOUBLE_COMPLEX , MPI_SUM , 0 , comm ) ;

  free( nu ) ;

  assert( buff2 = malloc( nxyz4 * sizeof( double ) ) ) ;

  compute_diag_dens( wavf_in , dens->rho , dens->sz , nwfip , nxyz , comm , buff2 , buff2 + nxyz ) ;

  compute_nondiag_dens( wavf_in , dens->sx , dens->sy , nwfip , nxyz , comm , buff2 , buff2 + nxyz ) ;

  compute_so_dens( deriv_x , deriv_y , deriv_z , dens->divjj , nwfip , nxyz , comm , buff2 ) ;

  compute_ctau_dens( wavf_in , deriv_x , deriv_y , deriv_z , dens->tau , dens->jx , dens->jy , dens->jz , nwfip , nxyz , comm , buff2 , buff2 + nxyz , buff2 +nxyz2 , buff2 + nxyz3 ) ;

  for( n = 0 ; n < nwfip ; n++ ) { 

    for( i = nxyz2 ; i < nxyz4 ; i++ )

      wavf_in[ n ][ i ] = conj( wavf_in[ n ][ i ] ) ; /* take the complex conjugate of v again */


    for( k = 0 ; k < 4 ; k++) { /* add the vector potential term to the derivatives */

      kshift = k * nxyz ;

      if( k < 2 ){

	for( i = kshift ; i < kshift + nxyz ; i++ ){

	  deriv_x[ n ][ i ] -= ( I * a_vf[ 1 ][ i - kshift ] * wavf_in[ n ][ i ] ) ;

	  deriv_y[ n ][ i ] -= ( I * a_vf[ 2 ][ i - kshift ] * wavf_in[ n ][ i ] ) ;

	  deriv_z[ n ][ i ] -= ( I * a_vf[ 3 ][ i - kshift ] * wavf_in[ n ][ i ] ) ;

	}

      } else{

	for( i = kshift ; i < kshift + nxyz ; i++ ){

	  deriv_x[ n ][ i ] += ( I * a_vf[ 1 ][ i - kshift ] * wavf_in[ n ][ i ] );

	  deriv_y[ n ][ i ] += ( I * a_vf[ 2 ][ i - kshift ] * wavf_in[ n ][ i ] );

	  deriv_z[ n ][ i ] += ( I * a_vf[ 3 ][ i - kshift ] * wavf_in[ n ][ i ] );

	}

      }

    }

  }

  if( ip == 0 ) {

      for( i = 0 ; i < nxyz ; i++ ) {

	dens->tau[ i ] -= ( 2. * ( dens->jx[ i ] * a_vf[ 1 ][ i ] + dens->jy[ i ] * a_vf[ 2 ][ i ] + dens->jz[ i ] * a_vf[ 3 ][ i ] ) - ( a_vf[ 1 ][ i ] * a_vf[ 1 ][ i ] + a_vf[ 2 ][ i ] * a_vf[ 2 ][ i ] + a_vf[ 3 ][ i ] * a_vf[ 3 ][ i ] ) * dens->rho[ i ] ) ;

	dens->jx[ i ] -= ( a_vf[ 1 ][ i ] * dens->rho[ i ] ) ;

	dens->jy[ i ] -= ( a_vf[ 2 ][ i ] * dens->rho[ i ] ) ;

	dens->jz[ i ] -= ( a_vf[ 3 ][ i ] * dens->rho[ i ] ) ;

      }

      assert( arho = malloc( nxyz3 * sizeof( double ) ) ) ;

      /* first compute diverg(Axs) */

      for( i = 0 ; i < nxyz ; i++ ){

	arho[ i ]         = a_vf[ 2 ][ i ] * dens->sz[ i ] - a_vf[ 3 ][ i ] * dens->sy[ i ] ;

	arho[ i + nxyz ]  = a_vf[ 3 ][ i ] * dens->sx[ i ] - a_vf[ 1 ][ i ] * dens->sz[ i ] ;

	arho[ i + nxyz2 ] = a_vf[ 1 ][ i ] * dens->sy[ i ] - a_vf[ 2 ][ i ] * dens->sx[ i ] ;

      }

      diverg_real( buff2 , arho , arho + nxyz , arho + nxyz2 , fftrans , latt , nxyz ) ;

      free( arho ) ;

      for( i = 0 ; i < nxyz ; i++ ){

	dens->divjj[ i ] -= buff2[ i ] ;

      }

    }

  free( buff2 ) ;

  if( nwfip == 0 ) // on proton/neutron root process
    curl( dens->jx , dens->jy , dens->jz , dens->cjx , dens->cjy , dens->cjz , nxyz , fftrans , latt ) ;

}



void compute_diag_dens( double complex ** wavf_in , double * rho , double * sz , const int nwfip , const int nxyz , const MPI_Comm comm , double * buff_rho , double * buff_sz )

{

  int n , i , ii ;

  int nxyz2 = nxyz + nxyz , nxyz3 = 3 * nxyz ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      buff_rho[ i ] = 0. ;

      buff_sz[ i ] = 0. ;

    }

  for( n = 0 ; n < nwfip ; n++ )

    {

      for( i = nxyz2 ; i < nxyz3 ; i++ )

	{

	  ii = i - nxyz2 ;

	  buff_rho[ ii ] += creal( conj( wavf_in[ n ][ i ] ) * wavf_in[ n ][ i ] + conj( wavf_in[ n ][ i + nxyz ] ) * wavf_in[ n ][ i + nxyz ] ) ;

	  buff_sz[ ii ]  += creal( conj( wavf_in[ n ][ i ] ) * wavf_in[ n ][ i ] - conj( wavf_in[ n ][ i + nxyz ] ) * wavf_in[ n ][ i + nxyz ] ) ;

	}

    }

  MPI_Reduce( buff_rho , rho , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( buff_sz , sz , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

}


void compute_nondiag_dens( double complex ** wavf_in , double * sx , double * sy , const int nwfip , const int nxyz , const MPI_Comm comm , double * buff_sx , double * buff_sy )

{

  int n , i , ii ;

  int nxyz2 = nxyz + nxyz , nxyz3 = 3 * nxyz ;

  double complex tmp ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      buff_sx[ i ] = 0. ;

      buff_sy[ i ] = 0. ;

    }

  for( n = 0 ; n < nwfip ; n++ )

    {

      for( i = nxyz2 ; i < nxyz3 ; i++ )

	{

	  ii = i - nxyz2 ;

	  tmp = wavf_in[ n ][ i ] * conj( wavf_in[ n ][ i + nxyz ] ) ;

	  buff_sx[ ii ] += creal( tmp ) ;

	  buff_sy[ ii ] -= cimag( tmp ) ;

	}

    }

  MPI_Reduce( buff_sx , sx , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( buff_sy , sy , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  if( nwfip == 0 )

    {

      for( i = 0 ; i < nxyz ; i++ )

	{

	  sx[ i ] *= 2. ;

	  sy[ i ] *= 2. ;

	}

    }

}

void compute_ctau_dens( double complex ** wavf_in , double complex ** deriv_x , double complex ** deriv_y , double complex ** deriv_z , double * tau , double * jx , double * jy , double * jz , const int nwfip , const int nxyz , const MPI_Comm comm , double * buff_jx , double * buff_jy , double * buff_jz , double * buff_tau )

{

  int n , i , ii , i1 ;

  int nxyz2 = nxyz + nxyz , nxyz3 = 3 * nxyz ;

  double complex tmp ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      buff_jx[ i ] = 0. ;

      buff_jy[ i ] = 0. ;

      buff_jz[ i ] = 0. ;

      buff_tau[ i ] = 0. ;

    }

  for( n = 0 ; n < nwfip ; n++ ) {

    for( i = nxyz2 ; i < nxyz3 ; i++ ){

      ii = i - nxyz2 ;

      i1 = i + nxyz ;

      buff_tau[ ii ] += creal( conj( deriv_x[ n ][ i ] ) * deriv_x[ n ][ i ] + conj( deriv_y[ n ][ i ] ) * deriv_y[ n ][ i ] + conj( deriv_z[ n ][ i ] ) * deriv_z[ n ][ i ] + conj( deriv_x[ n ][ i1 ] ) * deriv_x[ n ][ i1 ] + conj( deriv_y[ n ][ i1 ] ) * deriv_y[ n ][ i1 ] + conj( deriv_z[ n ][ i1 ] ) * deriv_z[ n ][ i1 ] )  ;

      buff_jx[ ii ] -= cimag( deriv_x[ n ][ i ] * wavf_in[ n ][ i ] + deriv_x[ n ][ i1 ] * wavf_in[ n ][ i1 ] ) ;

      buff_jy[ ii ] -= cimag( deriv_y[ n ][ i ] * wavf_in[ n ][ i ] + deriv_y[ n ][ i1 ] * wavf_in[ n ][ i1 ] ) ;

      buff_jz[ ii ] -= cimag( deriv_z[ n ][ i ] * wavf_in[ n ][ i ] + deriv_z[ n ][ i1 ] * wavf_in[ n ][ i1 ] ) ;

    }

  }

  MPI_Reduce( buff_tau , tau , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( buff_jx , jx , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( buff_jy , jy , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( buff_jz , jz , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

}

void compute_curlj_dens( double complex ** deriv_x , double complex ** deriv_y , double complex ** deriv_z , double * cjx , double * cjy , double * cjz , const int nwfip , const int nxyz , const MPI_Comm comm , double * buff_x , double * buff_y , double * buff_z )

{

  int n , iu , id , ii ;

  for( ii = 0 ; ii < nxyz ; ii++ )

    {

      buff_x[ ii ] = 0. ;

      buff_y[ ii ] = 0. ;

      buff_z[ ii ] = 0. ;

    }

  for( n = 0 ; n < nwfip ; n++ ) {

    for( ii = 0 ; ii < nxyz ; ii++ )

      {

	iu = ii + 2 * nxyz ;

	id = iu + nxyz ;

	buff_x[ ii ] -= cimag( conj( deriv_y[ n ][ iu ] ) * deriv_z[ n ][ iu ] - conj( deriv_z[ n ][ iu ] ) * deriv_y[ n ][ iu ] + conj( deriv_y[ n ][ id ] ) * deriv_z[ n ][ id ] - conj( deriv_z[ n ][ id ] ) * deriv_y[ n ][ id ] ) ;

	buff_y[ ii ] -= cimag( conj( deriv_z[ n ][ iu ] ) * deriv_x[ n ][ iu ] - conj( deriv_x[ n ][ iu ] ) * deriv_z[ n ][ iu ] + conj( deriv_z[ n ] [ id ] ) * deriv_x[ n ][ id ] - conj( deriv_x[ n ][ id ] ) * deriv_z[ n ][ id ] ) ;

	buff_z[ ii ] -= cimag( conj( deriv_x[ n ][ iu ] ) * deriv_y[ n ][ iu ] - conj( deriv_y[ n ][ iu ] ) * deriv_x[ n ][ iu ] + conj( deriv_x[ n ][ id ] ) * deriv_y[ n ][ id ] - conj( deriv_y[ n ][ id ] ) * deriv_x[ n ][ id ]  ) ;

      }

  }

  MPI_Reduce( buff_x , cjx , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( buff_y , cjy , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  MPI_Reduce( buff_z , cjz , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

}

void compute_so_dens( double complex ** deriv_x , double complex ** deriv_y , double complex ** deriv_z , double * divjj , const int nwfip , const int nxyz , const MPI_Comm comm , double * buff )

{

  int n , iu , id , ii ;

  for( ii = 0 ; ii < nxyz ; ii++ )

    {

      buff[ ii ] = 0. ;

    }

  for( n = 0 ; n < nwfip ; n++ )

    {

      for( ii = 0 ; ii < nxyz ; ii++ )

	{

	  iu = ii + 2 * nxyz ;
	  id = iu + nxyz ;
	  buff[ ii ] -= ( cimag( deriv_y[ n ][ iu ] * conj( deriv_x[ n ][ iu ] ) - deriv_y[ n ][ id ] * conj( deriv_x[ n ][ id ] ) + deriv_z[ n ][ iu ] * conj( deriv_y[ n ][ id ] ) - deriv_y[ n ][ iu ] * conj( deriv_z[ n ][ id ] ) ) + creal( deriv_x[ n ][ id ] * conj( deriv_z[ n ][ iu ] ) - deriv_x[ n ][ iu ] * conj( deriv_z[ n ][ id ] ) ) ) ;

	}

    }

  for( ii = 0 ; ii < nxyz ; ii++ )

    buff[ ii ] *= 2. ;

  MPI_Reduce( buff , divjj , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

}

void exchange_buffers( double * buff1 , double * buff2 , const int n , const int ip , const int ip1 , const int ip2 , const MPI_Comm commw , MPI_Status * status ) 

{

  int tag1 = 10 , tag2 = 100 ;

  if( ip == ip1 )

    MPI_Send( buff1 , n , MPI_DOUBLE , ip2 , tag1 , commw ) ;

  if( ip == ip2 )

    MPI_Recv( buff1 , n , MPI_DOUBLE , ip1 , tag1 , commw , status ) ;

  if( ip == ip2 )

    MPI_Send( buff2 , n , MPI_DOUBLE , ip1 , tag2 , commw ) ;

  if( ip == ip1 )

    MPI_Recv( buff2 , n , MPI_DOUBLE , ip2 , tag2 , commw , status ) ;

}

void exchange_nuclear_densities( Densities * dens_p , Densities * dens_n , const int nxyz , const int ip , const int root_p , const int root_n , const MPI_Comm commw , MPI_Status * status ) 

{

  exchange_buffers( dens_p->rho , dens_n->rho , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->tau , dens_n->tau , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->sx , dens_n->sx , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->sy , dens_n->sy , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->sz , dens_n->sz , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->jx , dens_n->jx , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->jy , dens_n->jy , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->jz , dens_n->jz , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->divjj , dens_n->divjj , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->cjx , dens_n->cjx , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->cjy , dens_n->cjy , nxyz , ip , root_p , root_n , commw , status ) ;

  exchange_buffers( dens_p->cjz , dens_n->cjz , nxyz , ip , root_p , root_n , commw , status ) ;

}

void free_phys_mem_dens( Densities * dens )

{

  free( dens->rho ) ;

  free( dens->tau ) ;

  free( dens->divjj ) ;

  free( dens->jx ) ;

  free( dens->jy ) ;

  free( dens->jz ) ;

  free( dens->sx ) ;

  free( dens->sy ) ;

  free( dens->sz ) ;

  free( dens->cjx ) ;

  free( dens->cjy ) ;

  free( dens->cjz ) ;

}

