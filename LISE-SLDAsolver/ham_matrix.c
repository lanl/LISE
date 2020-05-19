// for license information, see the accompanying LICENSE file

/* includes all the routines necessary to construct the Hamiltonian */

#include <stdlib.h>

#include <stdio.h>

#include <math.h>

#include <complex.h>

#include <inttypes.h>

#include <assert.h>

#include "vars_nuclear.h"

#include <mpi.h>

void i2xyz( const int , int * , int * , int * , const int , const int ) ;

void make_p_term( double complex * ham , double complex * d1_x , double complex * d1_y , double complex * d1_z , const double Vx , const double Vy , const double Vz , const int nx , const int ny , const int nz , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc ) ;

void make_ke_loc( double complex * , double * , double * , double * , double * , double * , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int ) ;

void make_pairing( double complex * , const int , double complex * , const int , const int , const int , const int , const int , const int , const int , const int ) ;

void so_contributions_1( double complex * , double * , double * , double * , double complex * , double complex * , double complex * , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int ) ;

void so_contributions_2( double complex * , double * , double * , double * , double complex * , double complex * , double complex * , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int ) ;

void so_contributions_3( double complex * , double * , double * , double * , double complex * , double complex * , double complex * , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int ) ;

void so_contributions_4( double complex * , double * , double * , double * , double complex * , double complex * , double complex * , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int ) ;

void laplacean( double * , const int , double * , const int , const int , const MPI_Comm , double * , double * , double * , const int , const int , const int , const int ) ;

void make_ham( double complex * ham , double * k1d_x , double * k1d_y , double * k1d_z , double complex * d1_x , double complex * d1_y , double complex * d1_z , Potentials * pots , Densities * dens,const int nx , const int ny , const int nz , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc , const int nstart , const int nstop , const MPI_Comm comm , Lattice_arrays * latt )

{

  /* constructs the full Hamiltonian matrix */

  int i , ii ;

  double * u_loc ;

#ifdef CONSTR_Q0
  int ii0=0;
#else
  int ii0=1;
#endif


  double complex * delta2 ;

  assert( u_loc = malloc( pots->nxyz * sizeof( double ) ) ) ;

  laplacean( pots->mass_eff , pots->nxyz , u_loc , nstart , nstop , comm , k1d_x , k1d_y , k1d_z , nx , ny , nz , 1 ) ;

  for( i = 0 ; i < pots -> nxyz ; i++ ){

    u_loc[ i ] = 0.5 * u_loc[ i ] + pots->u_re[ i ] - *( pots->amu ) ;

#ifdef CONSTRCALC
    for( ii = ii0; ii < 4; ii++)
      u_loc[ i ] += pots->lam2[ii]*pots->v_constraint[i+ii*pots->nxyz];
#endif
  }

  make_ke_loc( ham , k1d_x , k1d_y , k1d_z , pots->mass_eff , u_loc , nx , ny , nz , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc ) ;

  free( u_loc ) ;

  assert( delta2 = malloc( pots->nxyz * sizeof( double complex ) ) ) ;

  for( i = 0 ; i < pots -> nxyz ; i++ )

    *( delta2 + i ) = *( pots->delta + i ) + *( pots->delta_ext + i ) ;

  make_pairing( ham , pots->nxyz , delta2 , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc ) ;

  free( delta2 ) ;

  so_contributions_1( ham , pots->wx , pots->wy , pots->wz , d1_x , d1_y , d1_z , nx , ny , nz , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc ) ;

  so_contributions_2( ham , pots->wx , pots->wy , pots->wz , d1_x , d1_y , d1_z , nx , ny , nz , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc ) ;

  so_contributions_3( ham , pots->wx , pots->wy , pots->wz , d1_x , d1_y , d1_z , nx , ny , nz , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc ) ;

  so_contributions_4( ham , pots->wx , pots->wy , pots->wz , d1_x , d1_y , d1_z , nx , ny , nz , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc ) ;

}

void make_ke_loc( double complex * ham , double * k1d_x , double * k1d_y , double * k1d_z , double * mass_eff , double * u_loc , const int nx , const int ny , const int nz , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc )
{

  int li , lj , gi , gj ;

  int i ;

  int n , nxyz ;

  int ix_i , ix_j , iy_i , iy_j , iz_i , iz_j ;

  int i_xyz , j_xyz ;

  double dsign ;

  n = m_ip * n_iq ;

  nxyz = nx * ny * nz ;

  for ( i = 0 ; i < n ; i++ )  /* the hamiltonian is zeroed here */

    * ( ham + i ) = 0. + 0. * I ;

  for ( lj = 0 ; lj < n_iq ; lj++ )

    {

      gj = i_q * nb + ( lj / nb ) * q_proc * nb + lj % nb ;

      if( gj < 2 * nxyz ) 

	dsign = 0.5 ;

      else

	dsign = - 0.5 ;
      
      j_xyz = gj % nxyz ;

      i2xyz( j_xyz , &ix_j , &iy_j , &iz_j , ny , nz ) ;
      
      for( li = 0 ; li < m_ip ; li++ )

	{

	  gi = i_p * mb + ( li / mb ) * p_proc * mb + li % mb ;

	  if( gi / nxyz == gj / nxyz ) 

	    {

	      i_xyz = gi % nxyz ;

	      i2xyz( i_xyz , &ix_i , &iy_i , &iz_i , ny , nz ) ;

	      i = lj * m_ip + li ;

	      if( gi == gj )

		ham[ i ] += 2.0 * dsign * u_loc[ i_xyz ] ;

	      if( iy_i == iy_j && iz_i == iz_j )

		ham[ i ] += dsign * k1d_x[ abs( ix_i - ix_j ) ] * ( mass_eff[ i_xyz ] + mass_eff[ j_xyz ] ) ; 

	      if( ix_i == ix_j && iz_i == iz_j )

		ham[ i ] += dsign * k1d_y[ abs( iy_i - iy_j ) ] * ( mass_eff[ i_xyz ] + mass_eff[ j_xyz ] ) ; 

	      if( iy_i == iy_j && ix_i == ix_j )

		ham[ i ] += dsign * k1d_z[ abs( iz_i - iz_j ) ] * ( mass_eff[ i_xyz ] + mass_eff[ j_xyz ] ) ; 

	    }

	}

    }
 
}

void make_pairing( double complex * ham , const int nxyz , double complex * delta , const int m_ip , const int n_iq , const int i_p , const int i_q ,const int mb , const int nb , const int p_proc , const int q_proc )
{

  int li , lj , gi , gj ;

  int i ;

  int n ;

  int i_xyz , j_xyz ;

  double dsign ;

  n = m_ip * n_iq ;

  for ( lj = 0 ; lj < n_iq ; lj++ )

    {

      gj = i_q * nb + ( lj / nb ) * q_proc * nb + lj % nb ;

      j_xyz = gj % nxyz ;
      
      for( li = 0 ; li < m_ip ; li++ )

	{

	  gi = i_p * mb + ( li / mb ) * p_proc * mb + li % mb ;

	  if( j_xyz != gi % nxyz )

	    continue ;

	  i = lj * m_ip + li ;

	  if( gi / nxyz == 0 && gj / nxyz == 3 ) 

	    ham[ i ] = delta[ j_xyz ] ;

	  if( gi / nxyz == 1 && gj / nxyz == 2 ) 

	    ham[ i ] = - delta[ j_xyz ] ; 

	  if( gi / nxyz == 2 && gj / nxyz == 1 ) 

	    ham[ i ] = - conj( delta[ j_xyz ] ) ; /* conjugate here */

	  if( gi / nxyz == 3 && gj / nxyz == 0 ) 

	    ham[ i ] = conj( delta[ j_xyz ] ) ; /* conjugate here */

	}

    }
 
}

void so_contributions_1( double complex * ham , double * wx , double * wy , double * wz , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc )

     /* Calculates the SO contribution between u_up and u_up and u_up and u_dn */

{

  int li , lj , gi , gj ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  int i , ii , jj ;

  int nxyz , n ;

  int nx1 , ny1 , nz1 ;

  nx1 = nx - 1 ;

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

  n = m_ip * n_iq ;

  nxyz = nx * ny * nz ;

  for( lj = 0 ; lj < n_iq ; lj++ )

    {

      gj = i_q * nb + ( lj / nb ) * q_proc * nb + lj % nb ;

      if( gj > 2 * nxyz - 1 )

	continue ;

      jj = gj % nxyz ;

      i2xyz( jj , &ix2 , &iy2 , &iz2 , ny , nz ) ;

      for( li = 0 ; li < m_ip ; li++ )

	{
	  
	  gi = i_p * mb + ( li / mb ) * p_proc * mb + li % mb ;

	  if( gi / nxyz != 0 )

	    continue ;

	  ii = gi ;

	  i2xyz( ii , &ix1 , &iy1 , &iz1 , ny , nz ) ;

	  i = lj * m_ip + li ;

	  if( gj / nxyz == 0 )

	    {

	      if( iy1 == iy2 && iz1 == iz2 )

		ham[ i ] -= I * ( wy[ ii ] + wy[ jj ] ) * d1_x[ ix1 - ix2 + nx1 ] ;

	      if( ix1 == ix2 && iz1 == iz2 )

		ham[ i ] += I * ( wx[ ii ] + wx[ jj ] ) * d1_y[ iy1 - iy2 + ny1 ] ;

	    }

	  else

	    {

              if( iy1 == iy2 && iz1 == iz2 )

		ham[ i ] += ( wz[ ii ] + wz[ jj ] ) * d1_x[ ix1 - ix2 + nx1 ] ;

              if( ix1 == ix2 && iz1 == iz2 )

		ham[ i ] -= I * ( wz[ ii ] + wz[ jj ] ) * d1_y[ iy1 - iy2 + ny1 ] ;

              if( ix1 == ix2 && iy1 == iy2 )

		ham[ i ] -= ( wx[ ii ] + wx[ jj ] - I * ( wy[ ii ] + wy[ jj ] ) ) * d1_z[ iz1 - iz2 + nz1 ] ;

	    }

	}

    }

}

void so_contributions_2( double complex * ham , double * wx , double * wy , double * wz , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc )

     /* Calculates the SO contribution between u_dn and u_up and u_dn and u_dn */

{

  int li , lj , gi , gj ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  int i , ii , jj ;

  int nxyz , n ;

  int nx1 , ny1 , nz1 ;

  nx1 = nx - 1 ;

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

  n = m_ip * n_iq ;

  nxyz = nx * ny * nz ;

  for( lj = 0 ; lj < n_iq ; lj++ )

    {

      gj = i_q * nb + ( lj / nb ) * q_proc * nb + lj % nb ;

      if( gj > 2 * nxyz - 1 )

	continue ;

      jj = gj % nxyz ;

      i2xyz( jj , &ix2 , &iy2 , &iz2 , ny , nz ) ;

      for( li = 0 ; li < m_ip ; li++ )

	{
	  
	  gi = i_p * mb + ( li / mb ) * p_proc * mb + li % mb ;

	  if( gi / nxyz != 1 )

	    continue ;

	  ii = gi % nxyz ;

	  i2xyz( ii , &ix1 , &iy1 , &iz1 , ny , nz ) ;

	  i = lj * m_ip + li ;

	  if( gj / nxyz == 1 )

	    {

              if( iy1 == iy2 && iz1 == iz2 )

		ham[ i ] += I * ( wy[ ii ] + wy[ jj ] ) * d1_x[ ix1 - ix2 + nx1 ] ;

	      if( ix1 == ix2 && iz1 == iz2 )

		ham[ i ] -= I * ( wx[ ii ] + wx[ jj ] ) * d1_y[ iy1 - iy2 + ny1 ] ;

	    }

	  else

	    {

	      if( iy1 == iy2 && iz1 == iz2 )

		ham[ i ] -= ( wz[ ii ] + wz[ jj ] ) * d1_x[ ix1 - ix2 + nx1 ] ;

	      if( ix1 == ix2 && iz1 == iz2 )

		ham[ i ] -= I * ( wz[ ii ] + wz[ jj ] ) * d1_y[ iy1 - iy2 + ny1 ] ;

	      if( ix1 == ix2 && iy1 == iy2 )

		ham[ i ] += ( wx[ ii ] + wx[ jj ] + I * ( wy[ ii ] + wy[ jj ] ) ) * d1_z[ iz1 - iz2 + nz1 ] ;

	    }

	}

    }

}

void so_contributions_3( double complex * ham , double * wx , double * wy , double * wz , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc )

     /* Calculates the SO contribution between v_up and v_up and v_up and v_dn */

{

  int li , lj , gi , gj ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  int i , ii , jj ;

  int nxyz , n ;

  int nx1 , ny1 , nz1 ;

  nx1 = nx - 1 ;

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

  n = m_ip * n_iq ;

  nxyz = nx * ny * nz ;

  for( lj = 0 ; lj < n_iq ; lj++ )

    {

      gj = i_q * nb + ( lj / nb ) * q_proc * nb + lj % nb ;

      if( gj < 2 * nxyz )

	continue ;

      jj = gj % nxyz ;

      i2xyz( jj , &ix2 , &iy2 , &iz2 , ny , nz ) ;

      for( li = 0 ; li < m_ip ; li++ )

	{
	  
	  gi = i_p * mb + ( li / mb ) * p_proc * mb + li % mb ;

	  if( gi / nxyz != 2 )

	    continue ;

	  ii = gi % nxyz ;

	  i2xyz( ii , &ix1 , &iy1 , &iz1 , ny , nz ) ;

	  i = lj * m_ip + li ;

	  if( gj / nxyz == 2 )

	    {

	    if( iy1 == iy2 && iz1 == iz2 )

	      ham[ i ] -= I * ( wy[ ii ] + wy[ jj ] ) * d1_x[ ix1 - ix2 + nx1 ] ;

	    if( ix1 == ix2 && iz1 == iz2 )

	      ham[ i ] += I * ( wx[ ii ] + wx[ jj ] ) * d1_y[ iy1 - iy2 + ny1 ] ;

	    }

	  else

	    {

	      if( iy1 == iy2 && iz1 == iz2 )

		ham[ i ] -= ( wz[ ii ] + wz[ jj ] ) * d1_x[ ix1 - ix2 + nx1 ] ;

	      if( ix1 == ix2 && iz1 == iz2 )

		ham[ i ] -= I * ( wz[ ii ] + wz[ jj ] ) * d1_y[ iy1 - iy2 + ny1 ] ;

              if( ix1 == ix2 && iy1 == iy2 )

		ham[ i ] += ( wx[ ii ] + wx[ jj ] + I * ( wy[ ii ] + wy[ jj ] ) ) * d1_z[ iz1 - iz2 + nz1 ] ;

	    }

	}

    }

}

void so_contributions_4( double complex * ham , double * wx , double * wy , double * wz , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc )

     /* Calculates the SO contribution between v_up and v_dn and v_dn and v_dn */

{

  int li , lj , gi , gj ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  int i , ii , jj ;

  int nxyz , n ;

  int nx1 , ny1 , nz1 ;

  nx1 = nx - 1 ;

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

  n = m_ip * n_iq ;

  nxyz = nx * ny * nz ;

  for( lj = 0 ; lj < n_iq ; lj++ )

    {

      gj = i_q * nb + ( lj / nb ) * q_proc * nb + lj % nb ;

      if( gj < 2 * nxyz )

	continue ;

      jj = gj % nxyz ;

      i2xyz( jj , &ix2 , &iy2 , &iz2 , ny , nz ) ;

      for( li = 0 ; li < m_ip ; li++ )

	{
	  
	  gi = i_p * mb + ( li / mb ) * p_proc * mb + li % mb ;

	  if( gi / nxyz != 3 )

	    continue ;

	  ii = gi % nxyz ;

	  i2xyz( ii , &ix1 , &iy1 , &iz1 , ny , nz ) ;

	  i = lj * m_ip + li ;

	  if( gj / nxyz == 3 )

	    {

              if( iy1 == iy2 && iz1 == iz2 )

		ham[ i ] += I * ( wy[ ii ] + wy[ jj ] ) * d1_x[ ix1 - ix2 + nx1 ] ;

              if( ix1 == ix2 && iz1 == iz2 )

		ham[ i ] -= I * ( wx[ ii ] + wx[ jj ] ) * d1_y[ iy1 - iy2 + ny1 ] ;

	    }

	  else

	    {
              if( iy1 == iy2 && iz1 == iz2 )

		ham[ i ] += ( wz[ ii ] + wz[ jj ] ) * d1_x[ ix1 - ix2 + nx1 ] ;

              if( ix1 == ix2 && iz1 == iz2 )

		ham[ i ] -= I * ( wz[ ii ] + wz[ jj ] ) * d1_y[ iy1 - iy2 + ny1 ] ;

              if( ix1 == ix2 && iy1 == iy2 )

		ham[ i ] -= ( wx[ ii ] + wx[ jj ] - I * ( wy[ ii ] + wy[ jj ] ) ) * d1_z[ iz1 - iz2 + nz1 ] ;

	    }

	}

    }

}


void occ_numbers( double complex * z , double * occ , const int nxyz , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc, const MPI_Comm gr_comm , const int gr_ip )

{

  /* computes the occupation numbers = sum over the v's for each eigenvector */

  int li , lj , gi , gj ;

  int ix1 , ix2 , iy1 , iy2 , iz1 , iz2 ;

  int i ;

  double * occbuff ;

  assert( occbuff = malloc( 2 * nxyz * sizeof( double ) ) ) ;

  for( i = 0 ; i < 2 * nxyz ; i++ )

    *( occbuff + i ) = 0.0 ;

  for( lj = 0 ; lj < n_iq ; lj++ )

    {

      gj = i_q * nb + ( lj / nb ) * q_proc * nb + lj % nb - 2 * nxyz ;

      if( gj < 0 )

	continue ;

      for( li = 0 ; li < m_ip ; li++ )

	{
	  
	  gi = i_p * mb + ( li / mb ) * p_proc * mb + li % mb ;

	  if( gi / nxyz < 2 )

	    continue ;

	  i = lj * m_ip + li ;

	  occbuff[ gj ] += pow( cabs( z[ i ] ) , 2.0 ) ;

	}

    }

  MPI_Allreduce( occbuff , occ , 2 * nxyz , MPI_DOUBLE , MPI_SUM , gr_comm ) ;

  free( occbuff ) ;

}

void i2xyz( const int i , int * ix , int * iy , int * iz , const int ny , const int nz )

{

  *iz = i % nz ;

  *iy = ( ( i - * iz ) / nz ) % ny ;

  *ix = ( i - *iz - nz * *iy ) / ( nz * ny ) ;
	  
  if( *iz + nz * ( *iy + ny * *ix ) != i )

    printf( " wrong mapping %d != %d " , *iz + nz * ( *iy + ny * *ix ) , i ) ;

}


void make_p_term( double complex * ham , double complex * d1_x , double complex * d1_y , double complex * d1_z , const double Vx , const double Vy , const double Vz , const int nx , const int ny , const int nz , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc ) 

{

  int li , lj , gi , gj ;

  int i ;

  int n , nxyz ;

  int ix_i , ix_j , iy_i , iy_j , iz_i , iz_j ;

  int i_xyz , j_xyz , nx1 , ny1 , nz1 ;

  double dsign ;

  n = m_ip * n_iq ;

  nx1 = nx - 1 ; 

  ny1 = ny - 1 ;

  nz1 = nz - 1 ;

  nxyz = nx * ny * nz ;

  for ( lj = 0 ; lj < n_iq ; lj++ )

    {

      gj = i_q * nb + ( lj / nb ) * q_proc * nb + lj % nb ;

      j_xyz = gj % nxyz ;

      i2xyz( j_xyz , &ix_j , &iy_j , &iz_j , ny , nz ) ;
      
      for( li = 0 ; li < m_ip ; li++ )

	{

	  gi = i_p * mb + ( li / mb ) * p_proc * mb + li % mb ;

	  if( gi / nxyz == gj / nxyz ) 

	    {

	      i_xyz = gi % nxyz ;

	      i2xyz( i_xyz , &ix_i , &iy_i , &iz_i , ny , nz ) ;

	      i = lj * m_ip + li ;

	      if( iy_i == iy_j && iz_i == iz_j )

		ham[ i ] += I * d1_x[ ix_i - ix_j + nx1 ] * Vx ; 

	      if( ix_i == ix_j && iz_i == iz_j )

		ham[ i ] += I * d1_y[ iy_i - iy_j + ny1 ] * Vy ; 

	      if( iy_i == iy_j && ix_i == ix_j )

		ham[ i ] += I * d1_z[ iz_i - iz_j + nz1 ] * Vz ;

	    }

	}

    }
 
}
