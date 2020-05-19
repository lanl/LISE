// for license information, see the accompanying LICENSE file

/*

Functions to handle the computation of the energy 

*/

#include <stdio.h>

#include <stdlib.h>

#include <complex.h>

#include <assert.h>

#include <math.h>

#include <mpi.h>

#include "vars_nuclear.h"

void laplacean( double * , const int , double * , const int , const int , const MPI_Comm , double * , double * , double * , const int , const int , const int , const int ) ;

void coul_pot3( double * , double * , double * , double * , Lattice_arrays * , const int , const int , const int , const double, FFtransf_vars * , const double dxyz ) ;

void system_energy( Couplings * cc_edf , const int icoul , Densities * dens , Densities * dens_p , Densities * dens_n , const int isospin , double complex * delta , const int nstart , const int nstop , const int ip , const int gr_ip , const MPI_Comm comm , const MPI_Comm gr_comm , double * k1d_x , double * k1d_y , double * k1d_z , const int nx , const int ny , const int nz , const double hbar2m , const double dxyz , Lattice_arrays * latt_coords , FFtransf_vars * fftransf_vars , const double nprot , const double nneut, FILE * out )

{

  double e_tot , e_pair , e_rho , e_rhotau , e_gamma , e_so , e_laprho , e_kin ;

  int  ixyz , ix , iy , iz , nxyz ;

  double e_pair_n , e_j , tmp , e_coul , n_part ;

  double complex pair_gap , pair_gap_tot ;

  double * rho_0 , * rho_1 , * lap_rho_0 , * lap_rho_1 , * vcoul ;

  double xpow=1./3.;

  double e2 = -197.3269631*pow(3./acos(-1.),xpow) / 137.035999679 ;

  xpow*=4.;

  e2*=(3./2.);

  nxyz = nx * ny * nz ;

  assert( rho_0 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( rho_1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lap_rho_0 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lap_rho_1 = malloc( nxyz * sizeof( double ) ) ) ;

  if( icoul == 1 ) 

    {

      assert( vcoul = malloc( nxyz * sizeof( double ) ) ) ;

      coul_pot3( vcoul , dens_p->rho , rho_0 , rho_1 , latt_coords , nstart , nstop , nxyz , nprot , fftransf_vars , dxyz ) ;

    }

  for( ixyz = 0 ; ixyz < nxyz ; ixyz++ )

    {

      *( rho_0 + ixyz ) = *( dens_p->rho + ixyz ) + *( dens_n->rho + ixyz ) ;

      *( rho_1 + ixyz ) = *( dens_n->rho + ixyz ) - *( dens_p->rho + ixyz ) ;

    }

  laplacean( rho_0 , nxyz , lap_rho_0 , nstart , nstop , gr_comm , k1d_x , k1d_y , k1d_z , nx , ny , nz , 0 ) ;

  laplacean( rho_1 , nxyz , lap_rho_1 , nstart , nstop , gr_comm , k1d_x , k1d_y , k1d_z , nx , ny , nz , 0 ) ;

  e_kin    = 0. ;

  e_rho    = 0. ;

  e_rhotau = 0. ;

  e_laprho = 0. ;

  e_gamma  = 0. ;

  e_so     = 0. ;

  e_j      = 0. ;

  e_pair   = 0. ;

  e_coul   = 0. ;

  n_part   = 0. ;

  pair_gap = 0. + 0. * I ;

  for( ixyz = nstart ; ixyz < nstop ; ixyz++ )

    {

      n_part += dens->rho[ ixyz ] ;

      e_kin    += *( dens->tau + ixyz - nstart ) ;

      if(cc_edf->Skyrme){
	// Skyrme EDF
	e_rho    += ( cc_edf->c_rho_0 * pow( *( rho_0 + ixyz ) , 2. ) + cc_edf->c_rho_1 * pow( *( rho_1 + ixyz ) , 2. ) ) ;
	
	e_gamma  += ( cc_edf->c_gamma_0 * pow( *( rho_0 + ixyz ) , cc_edf->gamma + 2. ) + cc_edf->c_gamma_1 * pow( *( rho_0 + ixyz ) , cc_edf->gamma ) * pow( *( rho_1 + ixyz ) , 2. ) ) ;
      }
      else{
	// SeaLL1 EDF
	e_rho    += cc_edf->c_rho_a0 * pow( *(rho_0 + ixyz), 5./3. )
	+ cc_edf->c_rho_b0 * pow( *(rho_0 + ixyz), 2. )
	+ cc_edf->c_rho_c0 * pow( *(rho_0 + ixyz), 7./3. )
	+ cc_edf->c_rho_a1 * pow( *(rho_1 + ixyz), 2.) / (pow( *(rho_0 + ixyz), 1./3. ) + 1e-14)
	+ cc_edf->c_rho_b1 * pow( *(rho_1 + ixyz), 2.) 
	+ cc_edf->c_rho_c1 * pow( *(rho_1 + ixyz), 2.) * pow( *(rho_0 + ixyz), 1./3. )
	+ cc_edf->c_rho_a2 * pow( *(rho_1 + ixyz), 4.) / (pow( *(rho_0 + ixyz), 7./3. ) + 1e-14)
	+ cc_edf->c_rho_b2 * pow( *(rho_1 + ixyz), 4.) / (pow( *(rho_0 + ixyz), 2. ) + 1e-14)
	+ cc_edf->c_rho_c2 * pow( *(rho_1 + ixyz), 4.) / (pow( *(rho_0 + ixyz), 5./3. ) + 1e-14);
      }

      e_rhotau += ( cc_edf->c_tau_0 * ( *( dens_p->tau + ixyz - nstart ) + *( dens_n->tau + ixyz - nstart ) ) * *( rho_0 + ixyz ) + cc_edf->c_tau_1 * ( *( dens_n->tau + ixyz - nstart ) - *( dens_p->tau + ixyz - nstart ) ) * *( rho_1 + ixyz ) ) ;

      e_laprho += ( cc_edf->c_laprho_0 * *( lap_rho_0 + ixyz ) * ( *( rho_0 + ixyz ) ) + cc_edf->c_laprho_1 * *( lap_rho_1 + ixyz ) * *( rho_1 + ixyz ) ) ;

      

      e_so     += ( cc_edf->c_divjj_0 * *(rho_0 + ixyz ) * ( *( dens_n->divjj + ixyz - nstart ) + *( dens_p->divjj + ixyz - nstart ) ) + cc_edf->c_divjj_1 * *( rho_1 + ixyz ) * ( *( dens_n->divjj + ixyz - nstart ) - *( dens_p->divjj + ixyz - nstart ) ) ) ;

      e_pair -= creal( *( delta + ixyz ) * conj( *( dens->nu + ixyz - nstart ) ) ) ;

      pair_gap += delta[ ixyz ] * dens->rho[ ixyz ] ;

      if( icoul == 1 ) {
	e_coul += *( dens_p->rho + ixyz ) * *( vcoul + ixyz ) ;
	e_coul += (e2*pow(dens_p->rho[ixyz],xpow))*cc_edf->iexcoul;
      }

    }

  free( rho_0 ) ; free( rho_1 ) ; free( lap_rho_0 ) ; free( lap_rho_1 ) ;

  tmp = e_pair * dxyz ;

  MPI_Reduce( &tmp , &e_pair_n , 1 , MPI_DOUBLE , MPI_SUM , 0 , gr_comm ) ;

  tmp = n_part * dxyz ;

  MPI_Reduce( &tmp , &n_part , 1 , MPI_DOUBLE , MPI_SUM , 0 , gr_comm ) ;

  pair_gap = pair_gap * dxyz ;

  MPI_Reduce( &pair_gap , &pair_gap_tot , 1 , MPI_DOUBLE_COMPLEX , MPI_SUM , 0 , gr_comm ) ;

  if( gr_ip == 0 )

    if( isospin == 1 )

      {

	printf( " \n proton  pairing: %8.5f\n" , e_pair_n ) ;

	fprintf( out , "\n proton  pairing: %8.5f\n" , e_pair_n ) ;

	fprintf( out , "Delta_p = %8.5f MeV \n\n" , cabs( pair_gap_tot ) / n_part ) ;

	printf( "Delta_p = %8.5f MeV \n\n" , cabs( pair_gap_tot ) / n_part ) ;

      }

    else

      {

	printf( "\n neutron pairing: %8.5f\n" , e_pair_n ) ;

	fprintf( out , "\n neutron pairing: %8.5f\n" , e_pair_n ) ;

	fprintf( out , "Delta_n = %8.5f MeV \n\n" , cabs( pair_gap_tot ) / n_part ) ;

	printf( "Delta_n = %8.5f MeV \n \n" , cabs( pair_gap_tot ) / n_part ) ;

      }

  tmp = e_kin * hbar2m * dxyz ;

  MPI_Reduce( &tmp , &e_kin , 1 , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  tmp = e_pair * dxyz ;

  MPI_Reduce( &tmp , &e_pair , 1 , MPI_DOUBLE , MPI_SUM , 0 , comm ) ;

  if( isospin == -1 )

    return ;

  tmp = e_rho * dxyz ;

  MPI_Reduce( &tmp , &e_rho , 1 , MPI_DOUBLE , MPI_SUM , 0 , gr_comm ) ;

  tmp = e_rhotau * dxyz ;

  MPI_Reduce( &tmp , &e_rhotau , 1 , MPI_DOUBLE , MPI_SUM , 0 , gr_comm ) ;

  tmp = e_so * dxyz ;

  MPI_Reduce( &tmp , &e_so , 1 , MPI_DOUBLE , MPI_SUM , 0 , gr_comm ) ;

  tmp = e_laprho  * dxyz ;

  MPI_Reduce( &tmp , &e_laprho , 1 , MPI_DOUBLE , MPI_SUM , 0 , gr_comm ) ;

  tmp = e_j * dxyz ;

  MPI_Reduce( &tmp , &e_j , 1 , MPI_DOUBLE , MPI_SUM , 0 , gr_comm ) ;

  tmp = e_gamma * dxyz ;

  MPI_Reduce( &tmp , &e_gamma , 1 , MPI_DOUBLE , MPI_SUM , 0 , gr_comm ) ;

  if( icoul == 1 )

    {

      free( vcoul ) ;

      tmp = e_coul * .5 * dxyz ;

      MPI_Reduce( &tmp , &e_coul , 1 , MPI_DOUBLE , MPI_SUM , 0 , gr_comm ) ;

    }

  if( ip == 0 )

    {

      e_tot = e_kin + e_pair + e_rho + e_rhotau + e_laprho + e_so + e_gamma + e_coul + e_j ;

      printf("e_kin=%12.4f e_rho=%12.4f e_rhotau=%12.4f e_laprho=%12.4f e_gamma=%12.4f e_so=%12.4f e_coul=%12.4f \n" , e_kin , e_rho , e_rhotau , e_laprho , e_gamma , e_so , e_coul ) ;

      printf("field energy: %12.6f \n" , e_rho + e_rhotau + e_laprho + e_gamma ) ;

      printf("total energy: %12.6f \n" , e_tot ) ;

      fprintf( out , "e_kin=%12.4f e_rho=%12.4f e_rhotau=%12.4f e_laprho=%12.4f e_gamma=%12.4f e_so=%12.4f e_coul=%12.4f \n" , e_kin , e_rho , e_rhotau , e_laprho , e_gamma , e_so , e_coul ) ;

      fprintf( out , "field energy: %12.6f \n" , e_rho + e_rhotau + e_laprho + e_gamma ) ;

      fprintf( out , "total energy: %12.6f \n\n" , e_tot ) ;

    }

}
