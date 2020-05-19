// for license information, see the accompanying LICENSE file
//
// kenneth.roche@pnl.gov ; k8r@uw.edu
// initial version ... rochekj@ornl.gov 
// writes 2dbc decomposed matrix to global column vectors in a FILE
// ... using > MPI2 ROMIO semantics for portability
//

//grab a bag of routines ... not all needed here
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#include <mpi.h>

//for allocating arrays in the 2d-block cyclic space 
void get_blk_cyc( int ip , int npdim , int ma , int mblk , int * nele ) 
{ //virtual process(0,0) owns the first element(s), rows are traversed first in the loop of this map - this is known as a column major ordering 
  int my_ele, nele_, np;
  int srcproc, extra;

  srcproc = 0;
  //virtual process(0,0) owns the first element(s) 
  my_ele = ( npdim + ip - srcproc ) % npdim;
  nele_ = ma / mblk;
  np = ( nele_ / npdim ) * mblk;
  extra = nele_ % npdim;
  if ( my_ele < extra ) np += mblk; 
  else if ( my_ele == extra ) np += ma % mblk;
  *nele = np;
}

//for determining how many elements / data portion of the symbolic matrix the process at (ip,iq) of the pXq virtual rectangular process grid will manage in memory
void get_blk_cyc( int ip , int npdim , int ma , int mblk , int * nele ) ;
void get_mem_req_blk_cyc_new ( int ip , int iq , int np , int nq , int ma , int na , int mblk , int nblk , int * nip , int * niq ) 
{
  if ( ip >= 0 && iq >= 0 )
    {
      get_blk_cyc( ip , np , ma , mblk , nip ); //get rows
      get_blk_cyc( iq , nq , na , nblk , niq ); //get columns ...
#ifdef VERBOSE
      printf( "ip,iq = %d,%d\t nip, niq = %d %d\n" , ip , iq , *nip , *niq );
#endif
    }
  else 
    {
      *nip = -1;
      *niq = -1;
    }
}

void get_mem_req_blk_cyc_new ( int ip , int iq , int np , int nq , int ma , int na , int mblk , int nblk , int * nip , int * niq ) ;
void bc_wr_mpi( char * fn , MPI_Comm com , int p , int q , int ip , int iq , int blk , int jstrt , int jstp , int jstrd , int nxyz , double complex * z )
{ // i.e. ...  bc_wr_mpi( fnP, MPI_COMM_WORLD, p , q , ip , iq , mb , 0 , na , 1 , na , a );
  int i , j , k ;
  int ixyz , ierr ;
  int fd , niogrp ;
  int igrp , * nq_grp , * nv_grp , * vgrd , * itmp , * ranks , rcnt , lcnt , indx ;
  int irow , icol , gi , gj , li , lj , iqtmp ; 
  int nip , niq ;
  double complex * rz , * sz ;
  MPI_Comm * comg ;
  MPI_Group gw , * g ;
  int * iamg , mygrp , gsz ;
  MPI_Status mpi_st ; 
  MPI_File mpifd;
  MPI_Offset loffset;
  int iam , np , iflg ; 
  long int loff ;

  // working communicator and communicator internals 
  MPI_Comm_size( com , &np ) ;
  MPI_Comm_rank( com , &iam ) ;

  if ( jstrt < 0 || jstrt >= nxyz || jstp < 0 || jstp > nxyz ) 
    { // error 
      if ( iam == 0 )
	printf("error: invalid index range[b,e,s:%d,%d,%d]\n\t....exiting\n" , jstrt , jstp , jstrd ) ; 
      i = -1 ;
      MPI_Abort( com , i ) ;
    }

  if ( np != p * q )
    { // diagnostic to be safe 
      if ( iam == 0 )
	printf("error: grid dimensions do not match process count[p,q:%d,%d][np:%d]\n\t....exiting\n" , p , q , np ) ; 
      i = -1 ;
      MPI_Abort( com , i ) ;
    }

  niogrp = q ; // set the number of writers to be the number of processes in the column dimension of the virtual grid
  
  // going to get a communicator for each process column in the virtual process grid
  // the size of each group will be the number of process rows in the virtual process grid
  if ( ( nq_grp = malloc( sizeof( int ) * niogrp ) ) == NULL )
    {
      i = -1 ;
      MPI_Abort( com , i ) ;
    }
  
  if ( ( nv_grp = malloc( sizeof( int ) * niogrp ) ) == NULL ) 
    {
      i = -1 ;
      MPI_Abort( com , i ) ;
    } /* the number of global columns from z in [jstrt,jstp;jstrd] to be written by a particular io group that belong to a particular io group */

  for ( igrp = 0 ; igrp < niogrp ; igrp++ )
    {
      nq_grp[ igrp ] = q / niogrp ;
      if ( igrp < q % niogrp ) nq_grp[ igrp ]++ ;
    }
  
  /* create mpi to blacs id map - will need this to construct communicators */
  if ( iam == 0 ) 
    if ( ( itmp = malloc( sizeof( int ) * np ) ) == NULL ) 
      {
	i = -1 ;
	MPI_Abort( com , i ) ;
      }
  
  if ( ( vgrd = malloc( sizeof( int ) * np ) ) == NULL ) 
    {
      i = -1 ;
      MPI_Abort( com , i ) ;
    }

  // NOTE - this for testing purposes! single column major index from ip, iq, p, q
  i = ip + p * iq ;
  MPI_Gather( &i , 1 , MPI_INT , itmp , 1 , MPI_INT , 0 , com ) ;
  
  if ( iam == 0 ) 
    {
      for ( i = 0 ; i < np ; i++ ) vgrd[ itmp[ i ] ] = i ;
      free ( itmp ) ;
    }
  
  //note this is within the calling communicator
  MPI_Bcast( vgrd , np , MPI_INT , 0 , com ) ;

  // form unique io group communicators 
  if ( ( comg = malloc( sizeof( MPI_Comm ) * niogrp ) ) == NULL ) 
    {
      i = -1 ;
      MPI_Abort( com , i ) ;
    }
  if ( ( g = malloc( sizeof( MPI_Group ) * niogrp ) ) == NULL ) 
    {
      i = -1 ;
      MPI_Abort( com , i ) ;
    }

  // mpi id in io group communicator
  if ( ( iamg = malloc( niogrp * sizeof( int ) ) ) == NULL ) 
    {
      i = -1 ;
      MPI_Abort( com , i ) ;
    }

  // form a mpi group for com
  MPI_Comm_group( com , &gw ) ;
  
  i = 0 ; // used here to find max( nq_grp() ) 
  j = 0 ; // ... will hold the group index of the io group containing largest number of virtual process columns 

  for ( igrp = 0 ; igrp < niogrp ; igrp++ )
    if ( nq_grp[ igrp ] > i ) 
      {
	i = nq_grp[ igrp ] ;
	j = igrp ;
      }

  // process ranks from gw that belong to a particular io group 
  if ( ( ranks = malloc( sizeof( int ) * nq_grp[ j ] * p ) ) == NULL ) 
    {
      i = -1 ;
      MPI_Abort( com , i ) ;
    }

  lcnt = rcnt = 0 ;

  for ( igrp = 0 ; igrp < niogrp ; igrp++ )
    { 
      nv_grp[ igrp ] = 0 ;
      rcnt += nq_grp[ igrp ] ;
      indx = 0 ;
      for ( icol = lcnt ; icol < rcnt ; icol++ )
	{ // substitute loop over iq in virtual grid constrained to the iogrp space 

	  if ( iq == icol ) mygrp = igrp ;

	  for ( gj = jstrt ; gj < jstp ; gj += jstrd )
	    { // update the number of vectors to be written from igrp 
	      iqtmp = ( int ) floor( ( double ) ( gj / blk ) ) % q ; 
	      if ( icol == iqtmp ) nv_grp[ igrp ]++ ;
	    }

	  for ( irow = 0 ; irow < p ; irow++ ) 
	    { // assign PEs by virtual grid id to list of processes to be included in io group igrp 
	      ranks[ indx ] = vgrd[ irow + p * icol ] ; 
	      indx++ ;	    
	    }
	}

      lcnt = rcnt ;

      // form communicator for specific igrp */
      MPI_Group_incl( gw , p * nq_grp[ igrp ] , ranks , &g[ igrp ] ) ;
      MPI_Group_size( g[ igrp ] , &gsz ) ;
      if ( gsz != p * nq_grp[ igrp ] ) 
	{
	  i = -1 ;
	  MPI_Abort( com , i ) ;
	}
      MPI_Comm_create( com , g[ igrp ] , &comg[ igrp ] ) ; 
      MPI_Group_rank( g[ igrp ] , &iamg[ igrp ] ) ;
    }
  free( vgrd ) ;

  // at this point each of the io groups has been formed using mpi semantics ...
  // now, determine offsets into FILE -each group will be unique 
  loff = 0L ;
  for ( igrp = 0 ; igrp < mygrp ; igrp++ )
    loff += ( long ) nv_grp[ igrp ] * nxyz * sizeof( double complex ) ; // type should be made general but fine for now

  // int MPI_File_open( MPI_Comm comm, char *filename, int amode, MPI_Info info, MPI_File *mpi_fh );
  MPI_File_open(com, fn, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpifd);
 
  if ( iamg[ mygrp ] != MPI_UNDEFINED ) 
    {
      // send / receive buffers - length of column vector of global array -NOTE: this should be bsiz for optimal performance
      assert( sz = malloc( nxyz * sizeof( double complex ) ) ) ;
      assert( rz = malloc( nxyz * sizeof( double complex ) ) ) ;

      indx = 0 ; // now used to bump offset index into the lustre file 
      iflg = -1 ;
      lcnt = rcnt = 0 ;
      for ( igrp = 0 ; igrp < mygrp ; igrp++ ) lcnt += nq_grp[ igrp ] ;
      rcnt = lcnt + nq_grp[ mygrp ] ;
      get_mem_req_blk_cyc_new ( ip , iq , p , q , nxyz , nxyz , blk , blk , &nip , &niq ) ;
      for ( gj = jstrt ; gj < jstp ; gj += jstrd )
	{ 
	  iqtmp = ( int ) floor( ( double ) ( gj / blk ) ) % q ; 
	  for ( icol = lcnt ; icol < rcnt ; icol++ )
	    if ( icol == iqtmp ) 
	      iflg = 1 ;

	  if ( iflg > 0 )
	    { // ... the iogrp owns the column index gj 
	      for ( i = 0 ; i < nxyz ; i++ ) rz[ i ] = sz[ i ] = 0. + I * 0. ;
	      
	      if ( iq == iqtmp ) // ... iq owns gj and a copy needs to occur -all the other PEs contribute 0.,0. to the sum 
		{
		  lj = ( int ) floor( floor( ( double ) ( gj / blk ) ) / (double) q ) * blk + gj % blk ;
		  for ( li = 0 ; li < nip ; li++ )
		    { // form contribution to single global vector 
		      gi = ip * blk + ( int ) floor( ( double ) ( li / blk ) ) * p * blk + li % blk ; 
		      sz[ gi ] = z[ li + lj * nip ] ; 
		    }
		}
	      
	      MPI_Reduce( sz , rz , nxyz , MPI_DOUBLE_COMPLEX , MPI_SUM , 0 , comg[ mygrp ] ) ;
	      
	      if ( iamg[ mygrp ] == 0 )
		{
		  loffset = ( MPI_Offset ) ( loff + indx * nxyz * sizeof( double complex ) );
		  MPI_File_write_at(mpifd, loffset, ( const void * ) rz , 2*nxyz, MPI_DOUBLE, MPI_STATUS_IGNORE);
		  printf("");
		  indx++ ;
		}
	    }
	  iflg = -1 ;
	}
           
      free( sz ) ; free( rz ) ; 
      
    }

  MPI_File_close(&mpifd);
  MPI_Barrier(com);

  free( comg ) ; free( g ) ; free( iamg ) ; free( nq_grp ) ; free( nv_grp ) ; free( ranks ) ;
}
