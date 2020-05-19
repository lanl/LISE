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

#include "vars.h"

#include "tdslda_func.h"

double deform( double * rho , int nxyz , Lattice_arrays * latt_coords  , double dxyz );



// filter the density into left and right fragments
void makeFragment(double * dens, double *densf,double *theta,int n){
    
    int i;
    
    for(i=0;i<n;i++){
        densf[i]=dens[i]*theta[i%n];
    }
    return;
    
}

// calculate the coulomb energy between two fragments
/*
double coul_frag( double * rho , double * xa , double * ya , double * za , int nxyz , double dxyz,double z0 ){
    
    int i,j;
    double r;
    double sum=0.;
#pragma omp parallel for private(i,j) reduction(sum)
    for(i=0;i<nxyz;i++){
        if(za[i]>=z0)continue;
        for(j=0;j<nxyz;j++){
            if(za[j]<=z0)continue;
            sum+=rho[i]*rho[j]/sqrt((xa[i]-xa[j])*(xa[i]-xa[j])+(ya[i]-ya[j])*(ya[i]-ya[j])+(za[i]-za[j])*(za[i]-za[j]));
        }
    }
    double e2 = 197.3269631 / 137.035999679 ;
    return( sum*e2*dxyz*dxyz );
}
*/


double system_energy( Couplings * cc_edf , Densities * dens , Densities * dens_p , Densities * dens_n , const int isospin , const int nxyz , double complex * delta , double * chi, const int ip , const int root_p , const int root_n , const MPI_Comm comm , const double hbar2m , const double dxyz , Lattice_arrays * latt_coords , FFtransf_vars * fftransf_vars , MPI_Status * status , const double time , FILE * fd )

{

  const double mass_p = 938.272013 ;

  const double mass_n = 939.565346 ;

  double mass=.5*(mass_p+mass_n);

  const double hbarc = 197.3269631 ;

  // EXCOUL
  double xpow=1./3.;
  double e2 = -197.3269631*pow(3./acos(-1.),xpow) / 137.035999679 ;
  xpow*=4.;
  e2*=(3./2.);
  /////

  static double egs;

  static int compute_gs =0 ;

  double e_tot , e_pair , e_rho, e_rhotau, e_so , e_laprho , e_kin, e_cm ;

  double e_flow_n, e_flow_p , e_ext;

  int  ixyz , ix , iy , iz , i;

  double e_pair_n , e_kin_n , e_j , tmp , e_coul , n_part ;

  double pair_gap, pair_gap_n;

  double * rho_0 , * rho_1 , * lap_rho_0 , * lap_rho_1 , * vcoul ;

  double xcm , ycm , zcm , xcm_p , ycm_p , zcm_p , xcm_n , ycm_n , zcm_n ;

  double vx, vy, vz;

  int tag1 = 10 , tag2 = 100 , tag3 = 1000, tag4 = 10000 , tag5 = 100000, tag6 = 100001 ;

  double num_part , num_n , q20, q30, q40, q22;

  double xa, ya, za, ra;

  double beta;

  assert( rho_0 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( rho_1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lap_rho_0 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lap_rho_1 = malloc( nxyz * sizeof( double ) ) ) ;

  if( isospin == 1 ) 

    {

      assert( vcoul = malloc( nxyz * sizeof( double ) ) ) ;

      coul_pot3( vcoul , dens->rho , rho_0 , rho_1 , latt_coords , nxyz , fftransf_vars , dxyz ) ;

    }

  for( ixyz = 0 ; ixyz < nxyz ; ixyz++ )

    {

      rho_0[ ixyz ] = dens_p->rho[ixyz] + dens_n->rho[ixyz] ;

      rho_1[ixyz] = dens_n->rho[ixyz] - dens_p->rho[ixyz] ;

    }

  center_dist( rho_0 , nxyz , latt_coords , &xcm , &ycm , &zcm ) ;

  laplacean( rho_0 , lap_rho_0 , nxyz , fftransf_vars , latt_coords ) ;

  laplacean( rho_1 , lap_rho_1 , nxyz , fftransf_vars , latt_coords ) ;

  e_kin    = 0. ;

  e_rho    = 0. ;

  e_rhotau = 0. ;

  e_laprho = 0. ;

  e_so     = 0. ;

  e_j      = 0. ;

  e_pair   = 0. ;

  e_coul   = 0. ;

  num_part = 0. ;

  num_n = 0. ;

  //e_flow = 0.;
  e_flow_p = 0.;
  e_flow_n = 0.;


  e_ext = 0.;
  
  pair_gap = 0. ;

  q20 = 0. ;

  q22 = 0. ;

  q30 = 0. ;

  q40 = 0. ;


  vx = 0.; vy = 0.; vz = 0.;
  

  for( ixyz = 0 ; ixyz < nxyz ; ixyz++ )

    {

      num_part += dens->rho[ ixyz ] ;

      num_n += dens_n->rho[ ixyz ] ;

      xa = latt_coords->xa[ ixyz ]-xcm ;

      ya = latt_coords->ya[ ixyz ]-ycm ;

      za = latt_coords->za[ ixyz ]-zcm ;

      ra = sqrt(xa*xa + ya*ya + za*za);

      q20 += rho_0[ ixyz ] * (3*za*za - ra*ra);

      q30 += rho_0[ ixyz ] * za * ( 5*za*za - 3*ra*ra);;

      q40 += rho_0[ ixyz ] * (35.*pow(za, 4.0) - 30.*pow(za,2.0)*pow(ra,2.0) + 3.*pow(ra, 4.0));

      //      if( fabs( latt->xa[ i ] ) > .

      e_kin    += *( dens->tau + ixyz ) ;
	       
      if(cc_edf->Skyrme){
	e_rho += ( cc_edf->c_rho_0 * pow( *( rho_0 + ixyz ) , 2. ) ) + ( cc_edf->c_rho_1 * pow( *( rho_1 + ixyz ) , 2. ) ) + cc_edf->c_gamma_0 * pow( *( rho_0 + ixyz ) , cc_edf->gamma + 2. ) + cc_edf->c_gamma_1 * pow( *( rho_0 + ixyz ) , cc_edf->gamma ) * pow( *( rho_1 + ixyz ) , 2. );
      }
      else{
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
      e_rhotau += ( cc_edf->c_tau_0 * ( *( dens_p->tau + ixyz ) + *( dens_n->tau + ixyz ) ) * *( rho_0 + ixyz ) + cc_edf->c_tau_1 * ( *( dens_n->tau + ixyz ) - *( dens_p->tau + ixyz ) ) * *( rho_1 + ixyz ) ) ;

      e_laprho += cc_edf->c_laprho_0 * lap_rho_0[ixyz] * rho_0[ixyz] + cc_edf->c_laprho_1 * lap_rho_1[ixyz] * rho_1[ixyz] ;

      e_so     += ( cc_edf->c_divjj_0 * *(rho_0 + ixyz ) * ( *( dens_n->divjj + ixyz ) + *( dens_p->divjj + ixyz ) ) + cc_edf->c_divjj_1 * *( rho_1 + ixyz ) * ( *( dens_n->divjj + ixyz ) - *( dens_p->divjj + ixyz ) ) ) ;

      e_pair -= creal( *( delta + ixyz ) * conj( *( dens->nu + ixyz ) ) ) ;

      pair_gap +=  cabs(*( delta + ixyz)) * dens->rho[ixyz];

      e_j +=  ( cc_edf->c_j_0 * ( pow( dens_n->jx[ ixyz ] + dens_p->jx[ ixyz ] , 2 ) + pow( dens_n->jy[ ixyz ] + dens_p->jy[ ixyz ] , 2 ) + pow( dens_n->jz[ ixyz ] + dens_p->jz[ ixyz ] , 2 ) ) 

	      + cc_edf->c_j_1 * ( pow( dens_n->jx[ ixyz ] - dens_p->jx[ ixyz ] , 2 ) + pow( dens_n->jy[ ixyz ] - dens_p->jy[ ixyz ] , 2 ) + pow( dens_n->jz[ ixyz ] - dens_p->jz[ ixyz ] , 2 ) ) 

		+ cc_edf->c_divj_0 * ( ( dens_n->sx[ ixyz ] + dens_p->sx[ ixyz ] ) * ( dens_n->cjx[ ixyz ] + dens_p->cjx[ ixyz ] ) + ( dens_n->sy[ ixyz ] + dens_p->sy[ ixyz ] ) * ( dens_n->cjy[ ixyz ] + dens_p->cjy[ ixyz ] ) + ( dens_n->sz[ ixyz ] + dens_p->sz[ ixyz ] ) * ( dens_n->cjz[ ixyz ] + dens_p->cjz[ ixyz ] ) ) 

		+ cc_edf->c_divj_1 * ( ( dens_n->sx[ ixyz ] - dens_p->sx[ ixyz ] ) * ( dens_n->cjx[ ixyz ] - dens_p->cjx[ ixyz ] ) + ( dens_n->sy[ ixyz ] - dens_p->sy[ ixyz ] ) * ( dens_n->cjy[ ixyz ] - dens_p->cjy[ ixyz ] ) + ( dens_n->sz[ ixyz ] - dens_p->sz[ ixyz ] ) * ( dens_n->cjz[ ixyz ] - dens_p->cjz[ ixyz ] ) ) ) ;
      /*
      if( rho_0[ixyz] > 1-7 ){
	double j2=pow(dens_p->jx[ixyz]+dens_n->jx[ixyz],2.);
	j2+=pow(dens_p->jy[ixyz]+dens_n->jy[ixyz],2.);
	j2+=pow(dens_p->jz[ixyz]+dens_n->jz[ixyz],2.);
	e_flow += j2/rho_0[ixyz];
      }
      */
      if( dens_p->rho[ixyz] > 1e-7 ){
	e_flow_p += (pow(dens_p->jx[ixyz], 2.)+pow(dens_p->jy[ixyz], 2.)+pow(dens_p->jz[ixyz], 2.)) / dens_p->rho[ixyz];
      }
      if( dens_n->rho[ixyz] > 1e-7 ){
	e_flow_n += (pow(dens_n->jx[ixyz], 2.)+pow(dens_n->jy[ixyz], 2.)+pow(dens_n->jz[ixyz], 2.)) / dens_n->rho[ixyz];
      }
                  
      if( isospin == 1 ) 
	{
	e_coul += dens_p->rho[ ixyz ] * vcoul[ ixyz ] ;
	e_coul += (e2*pow(dens_p->rho[ixyz],xpow)) * (double) cc_edf->iexcoul;
	
	}
      

      e_ext += chi[ixyz] * rho_0[ixyz]; 

      vx += (dens_n->jx[ixyz] + dens_p->jx[ixyz]) ;
      
      vy += (dens_n->jy[ixyz] + dens_p->jy[ixyz]) ; 
      
      vz += (dens_n->jz[ixyz] + dens_p->jz[ixyz]) ; 
    }

  beta = deform( rho_0 , nxyz , latt_coords  , dxyz ) ;

  free( rho_0 );

  free( rho_1 ) ; free( lap_rho_0 ) ; free( lap_rho_1 ) ;

  e_pair *= dxyz ;

  pair_gap *= dxyz;

  e_kin *= ( hbar2m * dxyz ) ;

  num_n *= dxyz ;

  num_part *= dxyz;    

  pair_gap /= num_part;

  e_ext *= dxyz;


  if( ip == root_p )

    {

      MPI_Recv( &e_pair_n , 1 , MPI_DOUBLE , root_n , tag1 , comm , status ) ;

      MPI_Recv( &e_kin_n , 1 , MPI_DOUBLE , root_n , tag2 , comm , status ) ;

      MPI_Recv( &num_n , 1 , MPI_DOUBLE , root_n , tag3 , comm , status ) ;

      MPI_Recv( &pair_gap_n , 1 , MPI_DOUBLE , root_n , tag4 , comm , status ) ;

      e_kin += e_kin_n ;

      printf( "pairing energy: protons = %f, neutrons = %f \n" , e_pair , e_pair_n ) ;

      printf( "pairing gap:  protons = %f, neutrons = %f \n" , pair_gap , pair_gap_n ) ;

      printf( "#  protons = %12.6f \n# neutrons = %12.6f \n" , num_part , num_n ) ;

      e_pair += e_pair_n ;

      double mtot = mass * (num_n + num_part) ;

      vx *= hbarc*dxyz/mtot;
      vy *= hbarc*dxyz/mtot;
      vz *= hbarc*dxyz/mtot;

      e_cm = 0.5 * mtot * ( vx*vx + vy*vy + vz*vz );
    }

  if( ip == root_n )

    {

      MPI_Send( &e_pair , 1 , MPI_DOUBLE , root_p , tag1 , comm ) ;

      MPI_Send( &e_kin , 1 , MPI_DOUBLE , root_p , tag2 , comm ) ;

      MPI_Send( &num_part , 1 , MPI_DOUBLE , root_p , tag3 , comm ) ;

      MPI_Send( &pair_gap , 1 , MPI_DOUBLE , root_p , tag4 , comm ) ;

      return( e_kin ) ;

    }

  //e_rho_0 *= dxyz ;

  //e_rho_1 *= dxyz ;

  e_rho *= dxyz;

  e_rhotau *= dxyz ;

  e_so *= dxyz ;

  e_laprho  *= dxyz ;

  e_j *= dxyz ;

  //e_gamma_0 *= dxyz ;

  //e_gamma_1 *= dxyz ;

  e_flow_n *= ( hbar2m * dxyz );

  e_flow_p *= ( hbar2m * dxyz );

  if(isospin == 1) free( vcoul ) ;

  e_coul *= ( .5 * dxyz ) ;

  e_tot = e_kin + e_pair + e_rho + e_rhotau + e_laprho + e_so + e_coul + e_j ;

  center_dist( dens_p->rho , nxyz , latt_coords , &xcm_p , &ycm_p , &zcm_p ) ;

  center_dist( dens_n->rho , nxyz , latt_coords , &xcm_n , &ycm_n , &zcm_n ) ;

  if( compute_gs == 0 ){

    compute_gs = 1;

    egs = e_tot;

  }

  //  printf("C_lap_rho0=%f C_lap_rho1=%f\n",cc_edf->c_laprho_0,cc_edf->c_laprho_1);
  printf("e_kin=%12.6f e_rho=%14.6f e_rhotau=%12.6f e_laprho=%12.6f e_so=%12.6f e_coul=%12.6f e_j=%12.6f e_flow = %12.6f e_ext = %12.6f e_cm = %12.6f\n" , e_kin , e_rho , e_rhotau , e_laprho, e_so , e_coul , e_j , e_flow_n+e_flow_p, e_ext, e_cm) ;

  printf("field energy: %12.6f \n" , e_rho + e_rhotau + e_laprho + e_j ) ;

  printf("total energy: %12.6f \n\n" , e_tot ) ;

  fprintf( fd , "%12.6f    %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f  %6.3f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f  %12.6f  %12.6f \n" , time , e_tot , num_part , num_n , xcm , ycm , zcm , xcm_p , ycm_p , zcm_p , xcm_n , ycm_n , zcm_n , beta , e_flow_n+e_flow_p , egs , q20*dxyz, q30*dxyz, q40*dxyz , pair_gap, pair_gap_n, e_ext, e_cm ) ;

  return( e_tot ) ;

}



// calculate the distance between two fragments

void get_distance(Densities * dens_p , Densities * dens_n, Fragments * frag, const int nxyz , const double dxyz , Lattice_arrays * latt , double* distance)
{
  
  int ixyz, i;
  //double vel_L[3], vel_R[3];
  double xcm_L, ycm_L, zcm_L, xcm_R, ycm_R, zcm_R; 
  double num_L, num_R;
  // density of fragments
  double *rhof;
  assert(rhof  = (double *) malloc(nxyz*sizeof(double)));
  
  makeFragment(dens_n->rho, frag->densf_n,frag->thetaL,nxyz);
  makeFragment(dens_p->rho, frag->densf_p,frag->thetaL,nxyz);

  for(i=0;i<nxyz;i++) rhof[i] = frag->densf_n[i] + frag->densf_p[i];
  
  num_L = center_dist( rhof, nxyz , latt , &xcm_L, &ycm_L , &zcm_L ) ;


  makeFragment(dens_n->rho, frag->densf_n,frag->thetaR,nxyz);
  makeFragment(dens_p->rho, frag->densf_p,frag->thetaR,nxyz);

  for(i=0;i<nxyz;i++) rhof[i] = frag->densf_n[i] + frag->densf_p[i];
  
  num_R = center_dist( rhof, nxyz , latt , &xcm_R, &ycm_R , &zcm_R ) ;

  *distance = sqrt( (xcm_R-xcm_L)*(xcm_R-xcm_L) + (ycm_R-ycm_L)*(ycm_R-ycm_L) + (zcm_R-zcm_L)*(zcm_R-zcm_L));

  //free(rho_nf); free(rho_pf); free(rho_f);

  //free(thetaL); free(thetaR);
  free(rhof);

}

