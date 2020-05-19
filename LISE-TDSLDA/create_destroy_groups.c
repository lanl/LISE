// for license information, see the accompanying LICENSE file

/*

Translation of the fortran subroutine to create two groups
for protons and neutrons, respectively

*/

#include <mpi.h>

#include <stdlib.h>

#include <stdio.h>

#include <assert.h>

#include <math.h>

int create_mpi_groups( const MPI_Comm commw , MPI_Comm * gr_comm , const int np , int * gr_np , int * gr_ip , MPI_Group * group_comm , const int nwf_p , const int nwf_n , int * root_p , int * root_n )

{

  int isospin ;

  int i ;

  int * rankbuf ;

  MPI_Group world_group , group_p , group_n ;

  MPI_Comm gr_p , gr_n ;

  int ip , ip_p , ip_n , nwf ;

  int np_p , np_n ;

  double ratio ;

  MPI_Comm_rank( commw , &ip ); 

  nwf = nwf_p + nwf_n ;

  ratio = ( ( double ) np - 2. ) / ( ( double ) nwf ) ;

  if( ratio > 1.0 ) 

    return( 0 ) ; /* too many processors, abort */

  np_p = floor( ratio * nwf_p + 0.0001 ) + 1 ;

  if( np_p < 2 ) 

    np_p = 2 ;

  if( ip == 0 )

    printf( "the number of proton processors is %d of %d \n " , np_p , np ) ;

  np_n = np - np_p ;

  MPI_Comm_group( commw , &world_group ) ;

  rankbuf = malloc( np_p * sizeof( int ) ) ;

  for( i = 0 ; i < np_p ; i++ )

    rankbuf[ i ] = i ;

  /* form group of processes for protons now */

  MPI_Group_incl( world_group , np_p , rankbuf , &group_p ) ;

  MPI_Group_size( group_p , &i ) ;

  if( i != np_p )

    MPI_Abort( commw , 1 ) ;

  MPI_Comm_create( commw , group_p , &gr_p ) ;

  MPI_Group_rank( group_p , &ip_p ) ;

  free( rankbuf ) ;

  /* form the neutron group */

  MPI_Group_difference( world_group , group_p , &group_n ) ;

  MPI_Group_size( group_n , &i ) ;

  if( i != np_n )

    MPI_Abort( commw , -1 ) ;

  MPI_Comm_create( commw , group_n , &gr_n ) ;

  MPI_Group_rank( group_n , &ip_n ) ;

  * root_p = 0 ;

  * root_n = 0 ;

  if( ip_p != MPI_UNDEFINED )

    {

      isospin = 1 ;

      * gr_comm = gr_p ;

      * gr_ip = ip_p ;

      * gr_np = np_p ;

      * group_comm = group_p ;

      if( * gr_ip == 0 )

	* root_p = ip ;

    }

  if( ip_n != MPI_UNDEFINED )

    {

      isospin = -1 ;

      * gr_comm = gr_n ;

      * gr_ip = ip_n ;

      * gr_np = np_n ;

      * group_comm = group_n ;

      if( * gr_ip == 0 )

	* root_n = ip ;

    }

  MPI_Allreduce( root_p , &i , 1 , MPI_INT , MPI_SUM , commw ) ;

  * root_p = i ;

  MPI_Allreduce( root_n , &i , 1 , MPI_INT , MPI_SUM , commw ) ;

  * root_n = i ;

  return( isospin ) ;

}

void destroy_mpi_groups( MPI_Group * group_comm , MPI_Comm * gr_comm )

{

  MPI_Barrier( * gr_comm ) ;

  MPI_Group_free( group_comm ) ;

  MPI_Comm_free( gr_comm ) ;

}
