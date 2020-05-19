// for license information, see the accompanying LICENSE file

/* rochekj@ornl.gov */

/* 

process (ip,iq) in virtual process grid of dimension np * nq 

will own *nip * *niq * sizeof (data_type) BYTES of

matrix elements of matrix A : ma x na decomposed block cyclically

over blocks of dimension mblk x nblk

*/

#include <stdlib.h>

#include <math.h>

void get_num_rows( int iamprow , int nprows , int ma , int mblk , int * nrow ) {
  
  int mydist , nrows , np ;
  
  int srcproc , extrarows ;
  
  srcproc = 0 ; /* assume that the process(0,0) owns the first element(s) */
  
  mydist  = ( nprows + iamprow - srcproc ) % nprows ;
  
  nrows = ma / mblk ;

  np = ( nrows / nprows ) * mblk ;

  extrarows = nrows % nprows ;

  if ( mydist < extrarows ) np += mblk ; 

  else if ( mydist == extrarows ) np += ma % mblk ;
  
  *nrow = np ;

}

void get_num_columns( int iampcol , int npcols , int na , int nblk , int * ncol ) 
  
{
  
  int mydist , ncols , np ;
  
  int srcproc , extracols ;
  
  srcproc = 0 ; /* assume that the process(0,0) owns the first element(s) */
  
  mydist  = ( npcols + iampcol - srcproc ) % npcols ;
  
  ncols = na / nblk ;
  
  np = ( ncols / npcols ) * nblk ;
  
  extracols = ncols % npcols ;
  
  if ( mydist < extracols ) np += nblk ; 
  
  else if ( mydist == extracols ) np += na % nblk ;
  
  *ncol = np ;
  
}

void get_num_rows( int , int , int , int , int * ) ;

void get_num_columns( int , int , int , int , int * ) ;

void get_mem_req_blk_cyc ( int ip , int iq , int np , int nq , int ma , int na , int mblk , int nblk , int * nip , int * niq ) 

{
  
  if ( ip >= 0 && iq >= 0 )

    {

      get_num_rows( ip , np , ma , mblk , nip ) ;
      
      get_num_columns( iq , nq , na , nblk , niq ) ;
      
    }

  else 

    {

      *nip = -1 ; *niq = -1 ;
      
    }
  
}
