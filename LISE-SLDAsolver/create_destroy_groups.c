// for license information, see the accompanying LICENSE file

/*

Translation of the fortran subroutine to create two groups
for protons and neutrons, respectively

*/

#include <mpi.h>

#include <stdlib.h>

#include <stdio.h>

#include <assert.h>

int create_mpi_groups( const MPI_Comm commw , MPI_Comm * gr_comm , const int np , int * gr_np , int * gr_ip , MPI_Group * group_comm )

{

  int isospin ;

  int i ;

  int * rankbuf ;

  MPI_Group world_group , group_p , group_n ;

  MPI_Comm gr_p , gr_n ;

  int ip_p , ip_n ;

  * gr_np = np / 2 ;

  MPI_Comm_group( commw , &world_group ) ;

  assert( rankbuf = malloc( * gr_np * sizeof( int ) ) ) ;

  for( i = 0 ; i < * gr_np ; i++ )

    *( rankbuf + i ) = i ;

  /* form group of processes for protons now */

  MPI_Group_incl( world_group , * gr_np , rankbuf , &group_p ) ;

  MPI_Group_size( group_p , &i ) ;

  if( i != * gr_np )

    MPI_Abort( commw , 1 ) ;

  MPI_Comm_create( commw , group_p , &gr_p ) ;

  MPI_Group_rank( group_p , &ip_p ) ;

  /* form the neutron group */

  for( i = 0 ; i < * gr_np ; i++ )

    *( rankbuf + i ) += * gr_np ;

  /*  MPI_Group_difference( world_group , group_p , &group_n ) ; */

  MPI_Group_incl( world_group , * gr_np , rankbuf , &group_n ) ;

  MPI_Group_size( group_n , &i ) ;

  if( i != * gr_np )

    MPI_Abort( commw , -1 ) ;

  MPI_Comm_create( commw , group_n , &gr_n ) ;

  MPI_Group_rank( group_n , &ip_n ) ;

  MPI_Comm_rank( commw , &i ); 

  if( ip_p != MPI_UNDEFINED )

    {

      isospin = 1 ;

      * gr_comm = gr_p ;

      * gr_ip = ip_p ;

      if( ip_p != i )

	printf( " the process will fail, the proton process does not have the expected group ip: %d != %d\n" , i , * gr_ip ) ;

      * group_comm = group_p ;

    }

  if( ip_n != MPI_UNDEFINED )

    {

      isospin = -1 ;

      * gr_comm = gr_n ;

      * gr_ip = ip_n ;

      if( ip_n + * gr_np != i )

	printf( " the process will fail, the proton process does not have the expected group ip: %d != %d\n" , i , * gr_ip + * gr_np ) ;

      * group_comm = group_n ;

    }

  free( rankbuf ) ;

  return( isospin ) ;

}

void destroy_mpi_groups( MPI_Group * group_comm , MPI_Comm * gr_comm )

{

  MPI_Barrier( * gr_comm ) ;

  MPI_Group_free( group_comm ) ;

  MPI_Comm_free( gr_comm ) ;

}
