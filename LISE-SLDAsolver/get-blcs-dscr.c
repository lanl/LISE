// for license information, see the accompanying LICENSE file

/* kenneth.roche@pnl.gov */

#include <assert.h>

#include <math.h>

#include <stddef.h>

#include <stdlib.h>

#include <mpi.h>

#include <stdio.h>

void Cblacs_pinfo( int * , int * ) ;

void Cblacs_setup( int * , int * ) ;

void Cblacs_get( int , int , int * ) ;

void Cblacs_gridinit( int * , char * , int , int ) ;

void Cblacs_gridinfo( int , int * , int * , int * , int * ) ;

void Cblacs_exit( int ) ;

void Cfree_blacs_system_handle( int blacshandle ) ;

int  Csys2blacs_handle( MPI_Comm comm ) ;

void get_mem_req_blk_cyc ( int ip , int iq , int np , int nq , int ma , int na , int mblk , int nblk , int * nip , int * niq ) ;

void get_blcs_dscr( MPI_Comm commc , int m , int n , int mb , int nb , int p , int q , int * ip , int * iq , int * blcs_dscr , int * nip , int * niq )

{

  int i , MONE = -1 , ZERO = 0 ;

  char * b_order, * scope ;

  int blacshandle ; /* translate MPI communicator to BLACS integer value */

  int iam , np ; /* mpi process ids and sizeof(MPI_COMM_WORLD) */

  int info , iam_blacs , ictxt , nprocs_blacs ;

  MPI_Comm_size( commc , &np ) ;

  MPI_Comm_rank( commc , &iam ) ;

  if ( np != p * q ) 

    if ( iam == 0 ) 

      printf( "[%d]error in grid parameters: np != p.q\n" , iam ) ; /* recall that np_ is the number of processes in commw */
 
  if ( p * q == 1 ) /* small problem case :: calls LAPACK directly */
    
    if ( iam == 0 ) 
      
      printf( "[%d]Single processor case\n\tcall to zgeev_() here\n...exiting\n" , iam ) ;

  for ( i = 0 ; i < 9 ; i++ )

    *( blcs_dscr + i ) = -1 ;

  /* initialize the BLACS grid - a virtual rectangular grid */
  
  b_order = "R" ;
  
  scope = "All" ;
  
  Cblacs_pinfo( &iam_blacs , &nprocs_blacs ) ;
  
  if ( nprocs_blacs < 1 ) 
    
    Cblacs_setup( &iam_blacs , &nprocs_blacs ) ;
  
  blacshandle =  Csys2blacs_handle( commc ) ;
  
  ictxt = blacshandle ;
  
  Cblacs_gridinit( &ictxt , b_order , p , q ) ;  /* 'Row-Major' */
  
  Cblacs_gridinfo( ictxt , &p , &q , ip , iq ) ; /* get (ip,iq) , the process (row,column) id */
  
  get_mem_req_blk_cyc ( *ip , *iq , p , q , m , n , mb , nb , nip , niq ) ;

  *( blcs_dscr + 0 ) = 1 ;

  *( blcs_dscr + 1 ) = ictxt ;

  *( blcs_dscr + 2 ) = m ;

  *( blcs_dscr + 3 ) = n ;

  *( blcs_dscr + 4 ) = mb ;

  *( blcs_dscr + 5 ) = nb ;

  *( blcs_dscr + 6 ) = 0 ; /* C vs F conventions */

  *( blcs_dscr + 7 ) = 0 ; /* C vs F conventions */

  *( blcs_dscr + 8 ) = * nip ;
  
  MPI_Barrier( commc ) ;

}
