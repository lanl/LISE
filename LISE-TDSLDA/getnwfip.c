// for license information, see the accompanying LICENSE file

/* kenneth.roche@pnl.gov ; k8r@uw.edu */

#include <stdlib.h>
#include <math.h>

void getnwfip( int ip , int np , int nwf , int * nwfip ) 
{
  *nwfip  = nwf / np ;
  if ( ip < ( nwf % np ) ) 
    (*nwfip)++ ;
}
