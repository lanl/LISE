// for license information, see the accompanying LICENSE file


#include "vars_nuclear.h"

#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <complex.h>

#include <assert.h>

#include <mpi.h>

extern double pi ;

double minr( double , double ) ;

void grid3( double * , double * , double * , const int , const int , const int , double * , double * , double * ) ;

void i2xyz( const int , int * , int * , int * , const int , const int ) ;

void divide_work( const int , const int , const int , int * , int * ) ;

void gradient( double complex * , const int , double complex * , double complex * , double complex * , const int , const int , const MPI_Comm , double complex * , double complex * , double complex * , const int , const int , const int , const int ) ;

void gradient_ud( double complex * f , const int n , double complex * g_x , double complex * g_y , double complex * g_z , double complex * g_xd , double complex * g_yd , double complex * g_zd , const int nstart , const int nstop , const MPI_Comm comm , double complex * d1_x , double complex * d1_y , double complex * d1_z , const int nx , const int ny , const int nz ) ;

void diverg( double * , double * , double * , double * , const int , const int , const int , const MPI_Comm , double complex * , double complex * , double complex * , const int , const int , const int ) ;

double factor_ec( const double , const double, int icub) ;

void shift_coords( double complex * vec , const int n , FFtransf_vars * fftransf_vars , Lattice_arrays * latt_coords , const double xcm , const double ycm , const double zcm ) ;

void laplacean( double * f , const int n , double * lapf , const int nstart , const int nstop , const MPI_Comm comm , double * k1d_x , double * k1d_y , double * k1d_z , const int nx , const int ny , const int nz , const int ired );

void laplacean_complex( double complex * f , const int n , double complex * lapf , const int nstart , const int nstop , const MPI_Comm comm , double * k1d_x , double * k1d_y , double * k1d_z , const int nx , const int ny , const int nz , const int ired ) ;


void allocate_dens( Densities * dens , const int ip , const int np , const int nxyz )

{

  int iwork ;

  int i ;
   
  divide_work( nxyz , ip , np , &( dens->nstart ) , &( dens->nstop) ) ;

  iwork = dens->nstop - dens->nstart ;

  if( iwork > 0 )

    {

      assert( dens->tau = malloc( iwork * sizeof( double ) ) ) ;

      assert( dens->nu = malloc( iwork * sizeof( double complex ) ) ) ;

      assert( dens->divjj = malloc( iwork * sizeof( double ) ) ) ;

      assert( dens->jjx = malloc( iwork * sizeof( double ) ) ) ;

      assert( dens->jjy = malloc( iwork * sizeof( double ) ) ) ;

      assert( dens->jjz = malloc( iwork * sizeof( double ) ) ) ;

      for( i = 0 ; i < iwork ; i++ )
	
	{

	  dens->tau[ i ] = 0. ;

	  dens->divjj[ i ] = 0. ;

	  dens->nu[ i ] = 0. + 0. * I ;

	}

    }

  assert( dens->rho = malloc( nxyz * sizeof( double ) ) ) ;

#pragma omp parallel for default(shared) private(i) 

  for( i = 0 ; i < nxyz ; i++ )

    dens->rho[ i ] = 0. ;

}

void mem_share( Densities * dens , double * ex_array , const int nxyz , const int idim )

{

  int i ;

  for( i = 0 ; i < idim ; i++ )

    *( ex_array + i ) = 0. ;
  
  dens->rho = ex_array ;

  if( idim > nxyz ) {

    dens->divjj = ex_array + nxyz ;

    dens->tau = ex_array + ( nxyz + dens->nstop - dens->nstart ) ;

  }

}



/* function to generate the 1D kinetic energy */

void generate_ke_1d( double * ke , const int n , const double a , const int iopt_der ) 
{

  int i , j ;

  double pi2 , angle , xn2 , constant , isign , xn ;

  pi2 = pow( pi / a , 2 ) ;

  xn = ( double ) n ;

  xn2 = xn * xn ;

  constant = 2.0 * pi2 / xn2 ;

  if( iopt_der == 0 )

    * ke = pi2 * ( 1.0 - 1.0 / xn2 ) / 3.0 ;   /* Baye */

  else

    * ke = pi2 * ( 1.0 + 2.0 / xn2 ) / 3.0 ;

  isign = - 1.0 ;

  for ( i = 1 ; i < n ; i++ ) 

    {

      angle = ( double ) i * pi / xn ;

      ke [ i ] = isign * constant / ( pow( sin( angle ) , 2.0 ) ) ;

      if( iopt_der == 0 ) 

        ke [ i ] = ke[ i ] * cos( angle ) ; /* Baye */

      isign = - isign ;

    }

}


/* Generate the 1D derivative */

void generate_der_1d( double complex * der , const int n , const double a , const int iopt_der ) 

{

  int i , n1 ;

  double angle , xn , c , isign ;

  double c1 = 0. ;  /* remove the highest component for c1=0 */

  xn = (double) n ;

  n1 = n - 1 ;

  c = pi / xn ;

  /* Note: the order in the derivative array is: -( N - 1 ) cooresponding to der[0],
     -(N-1) + 1 corresponds to der[1] , and so forth */

  if ( iopt_der == 0 ) 

    * ( der + n1 ) = 0.0 ;

  else

    * ( der + n1 ) = - c * c1 * I / a ;

  isign = -1.0 / a ;

  for ( i = 1 ; i < n ; i++ )

    {

      angle = i * c ;

      if ( iopt_der == 0 ) 

	*( der + n1 + i ) = isign * c / sin( angle ) ;

      else

	*( der + n1 + i ) = isign * c * ( 1.0 / tan( angle ) - c1 * I ) ;

      isign = - isign ;

      *( der + n1 - i ) = - conj( *( der + n1 + i ) ) ;

    }

}

void make_coordinates( const int nxyz , const int nx , const int ny , const int nz , const double dx , const double dy , const double dz , Lattice_arrays * lattice_vars )

{

  double * xx , * yy , * zz ;

  double xn ;

  int i ;

  assert( lattice_vars-> wx = malloc( nxyz * sizeof( int ) ) ) ; 

  assert( lattice_vars-> wy = malloc( nxyz * sizeof( int ) ) ) ;

  assert( lattice_vars-> wz = malloc( nxyz * sizeof( int ) ) ) ;

  assert( lattice_vars-> xa = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> ya = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> za = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> kx = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> ky = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> kz = malloc( nxyz * sizeof( double ) ) ) ;

  assert( lattice_vars-> kin = malloc( nxyz * sizeof( double ) ) ) ;

  assert( xx = malloc( nx * sizeof( double ) ) ) ;

  assert( yy = malloc( ny * sizeof( double ) ) ) ;

  assert( zz = malloc( nz * sizeof( double ) ) ) ;


  for(i=0;i<nx;i++)
    xx[i]=1.;
  xx[nx/2]=0.;

  for(i=0;i<ny;i++)
    yy[i]=1.;
  yy[ny/2]=0.;

  for(i=0;i<nz;i++)
    zz[i]=1.;
  zz[nz/2]=0.;

  grid3( xx , yy , zz , nx , ny , nz , lattice_vars->xa , lattice_vars->ya , lattice_vars->za ) ;

  for(i=0;i<nxyz;i++){
    lattice_vars->wx[i]=(int) lattice_vars->xa[i];
    lattice_vars->wy[i]=(int) lattice_vars->ya[i];
    lattice_vars->wz[i]=(int) lattice_vars->za[i];
  }

  xn = ( ( double ) nx ) / 2.0 ;

  for( i = 0 ; i < nx ; i++ )

    xx[ i ] = ( ( double)  i - xn ) * dx ;

  xn = ( ( double ) ny ) / 2.0 ;

  for( i = 0 ; i < ny ; i++ )

    yy[ i ] = ( ( double ) i - xn ) * dy ;

  xn = ( ( double ) nz ) / 2.0 ;

  for( i = 0 ; i < nz ; i++ )

    zz[ i ] = ( ( double ) i - xn ) * dz ;

  grid3( xx , yy , zz , nx , ny , nz , lattice_vars->xa , lattice_vars->ya , lattice_vars->za ) ;

  /* set also the momentum space */

  xn = pi / ( ( double ) nx * dx ) ;

  for( i = 0 ; i < nx / 2 ; i++ )

    {

      *( xx + i ) = ( double) ( 2 * i ) * xn ;

      *( xx + i + nx / 2 ) = ( double ) ( 2 * i - nx ) * xn ;

    }

  xn = pi / ( ( double ) ny * dy ) ;

  for( i = 0 ; i < ny / 2 ; i++ )

    {

      *( yy + i ) = ( double) ( 2 * i ) * xn ;

      *( yy + i + ny / 2 ) = ( double ) ( 2 * i - ny ) * xn ;

    }

  xn = pi / ( ( double ) nz * dz ) ;

  for( i = 0 ; i < nz / 2 ; i++ )

    {

      *( zz + i ) = ( double) ( 2 * i ) * xn ;

      *( zz + i + nz / 2 ) = ( double ) ( 2 * i - nz ) * xn ;

    }

  grid3( xx , yy , zz , nx , ny , nz , lattice_vars->kx , lattice_vars->ky , lattice_vars->kz ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      lattice_vars->kin[ i ] = pow( lattice_vars->kx[ i ] , 2. ) + pow( lattice_vars->ky[ i ] , 2. ) + pow( lattice_vars->kz[ i ] , 2. ) ;

    }

  free( xx ) ;

  free( yy ) ;

  free( zz ) ;

}

void compute_densities( double * lam , double complex * z , const int nxyz , const int ip , const MPI_Comm comm , Densities * dens , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc , const int nx , const int ny , const int nz , double complex * d1_x , double complex * d1_y , double complex * d1_z , double * k1d_x, double * k1d_y, double * k1d_z, const double e_max , int * nwf  , double * occ , FFtransf_vars * fftransf_vars , Lattice_arrays * latt_coords , int icub)

{

  /*

  Computes the particle, anomalous and current densities from the eigenvectors z

  */

  int n , nhalf , iwork ;

  int i , j , li , lj , ii , jj , iu , id , kk1 , nn ;

  int nstart , nstop ;

  double complex * vec , * vec1 , * vec2 , * vec3 ;

  double complex * dx_d , * dy_d , * dz_d , * dx_u , * dy_u , * dz_u ;

  double num_part , num_part_sum , jxy , jxz , jyz , jzy , jyx , jzx , f1 ;

  double complex * vvx, * vvy, * vvz;

  double complex * lapl_d, * lapl_u; // laplacean of up and down components 

  double * rho , * occ1 ;

  int * indx , i_tot ;

  nstart = dens->nstart ;

  nstop = dens->nstop ;

  iwork = nstop - nstart ;

  n = 4 * nxyz ;

  nn = 3 * nxyz ;

  nhalf = 2 * nxyz ;

  assert( lapl_u = malloc( nxyz * sizeof( double complex ) ) ) ; // Asserting the laplacean elements.
  assert( lapl_d = malloc( nxyz * sizeof( double complex ) ) ) ;

  if( iwork > 0 ) {

    assert( dx_d = malloc( iwork * sizeof( double complex ) ) ) ;

    assert( dy_d = malloc( iwork * sizeof( double complex ) ) ) ;

    assert( dz_d = malloc( iwork * sizeof( double complex ) ) ) ;

    assert( dx_u = malloc( iwork * sizeof( double complex ) ) ) ;

    assert( dy_u = malloc( iwork * sizeof( double complex ) ) ) ;

    assert( dz_u = malloc( iwork * sizeof( double complex ) ) ) ;

  }

  assert( occ1 = malloc( nhalf * sizeof( double ) ) ) ;

  assert( vec = malloc( nn * sizeof( double complex ) ) ) ;

  assert( vec1 = malloc( n * sizeof( double complex ) ) ) ;

  assert( rho = malloc( nxyz * sizeof( double ) ) ) ;     
  

  for( i = 0 ; i < nxyz ; i++ ) 

    *( rho + i ) = 0. ;

  for( i = 0 ; i < iwork ; i++ )

    {

      *( dens->tau + i ) = 0. ;

      *( dens->divjj + i ) = 0. ;

      *( dens->nu + i ) = 0. + I * 0. ;

    }

  * nwf = 0 ;

  for( jj = 0 ; jj < nhalf ; jj++ ) {

    occ1[ jj ] = 0. ;

  }

  f1 = 1.0 ;

  for( jj = nhalf ; jj < n ; jj++ )

    /* construct one vector at a time for positive eigenvalues */

    {

      f1 = sqrt( factor_ec( *( lam + jj ) , e_max ,icub) ) ;

      if( f1 < 1.e-6 ) 

	break ;

      for( i = 0 ; i < n ; i++ )

	*( vec1 + i ) = 0. + 0. * I ;

#pragma omp parallel for default(shared) private(li,ii,lj,j) 

      for( lj = 0 ; lj < n_iq ; lj++ )

	{

	  j =  i_q * nb + ( lj / nb ) * q_proc * nb + lj % nb ;

	  if( j == jj )

	    for( li = 0 ; li < m_ip ; li++ ) 

	      {

		ii = i_p * mb + ( li / mb ) * p_proc * mb + li % mb ;

		vec1[ ii ] = z[ lj * m_ip + li ] * f1 ;

	      }

	}

      MPI_Allreduce( vec1 + nxyz , vec , nn , MPI_DOUBLE_COMPLEX , MPI_SUM , comm ) ;

      gradient_ud( vec + nxyz , nxyz , dx_u , dy_u , dz_u , dx_d , dy_d , dz_d , nstart , nstop , comm , d1_x , d1_y , d1_z , nx , ny , nz ) ;

// Calculating Laplacians.
      laplacean_complex( vec + nxyz, nxyz, lapl_u, nstart, nstop, comm, k1d_x, k1d_y, k1d_z, nx, ny, nz, 0) ;
      laplacean_complex( vec + 2*nxyz, nxyz, lapl_d, nstart, nstop, comm, k1d_x, k1d_y, k1d_z, nx, ny, nz, 0) ;

#pragma omp parallel for default(shared) private(i,iu,id,ii) 
      for( i = nstart ; i < nstop ; i++ )

	{

	  ii = i - nstart ;

	  iu = i + nxyz ;

	  id = iu + nxyz ;

	  occ1[ *nwf ] += ( pow( cabs( vec[ iu ] ) , 2. ) + pow( cabs( vec[ id ] ) , 2. ) ) ;

	  *( rho + i ) += ( pow( cabs( vec[ iu ] ) , 2. ) + pow( cabs( vec[ id ] ) , 2. ) ) ;
	  
	  *( dens->tau + ii ) -= creal(conj(vec[iu])*lapl_u[i] + conj(vec[id])*lapl_d[i]); 

	  *( dens->nu + ii ) -= ( conj( vec[ i ] ) * vec[ iu ] ) ; 

	  *( dens->divjj + ii ) -= ( cimag( dy_u[ ii ] * conj( dx_u[ ii ] ) - dy_d[ ii ] * conj( dx_d[ ii ] ) + dz_u[ ii ] * conj( dy_d[ ii ] ) - dy_u[ ii ] * conj( dz_d[ ii ] ) ) + creal( dx_d[ ii ] * conj( dz_u[ ii ] ) - dx_u[ ii ] * conj( dz_d[ ii ] ) ) ) ;

	}

      (*nwf)++ ;

    }

  free( vec ) ; free( vec1 ) ; 

  if( iwork > 0 ) { free( dx_d ) ; free( dy_d ) ; free( dz_d ) ; free( dx_u ) ; free( dy_u ) ; free( dz_u ) ; }

  MPI_Allreduce( occ1 , occ , nhalf , MPI_DOUBLE , MPI_SUM , comm ) ; 

  MPI_Allreduce( rho , dens->rho , nxyz , MPI_DOUBLE , MPI_SUM , comm ) ; 

  free( lapl_u ) ; free( lapl_d );
  free( occ1 ) ;

  laplacean( dens->rho, nxyz, rho, 0, nxyz, comm, k1d_x, k1d_y, k1d_z, nx, ny, nz, 0 ); // Now buffer rho contains lap rho.

  for( i = 0 ; i < nstop - nstart ; i++)
  {
    dens->tau[i]+=.5*rho[i+nstart] ; 
    dens->divjj[ i ] *= 2. ;

  }

  free( rho ) ; 

}

void compute_densities_finitetemp( double * lam , double complex * z , const int nxyz , const int ip , const MPI_Comm comm , Densities * dens , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc , const int nx , const int ny , const int nz , double complex * d1_x , double complex * d1_y , double complex * d1_z , const double e_max , const double temp, int * nwf  , double * occ , FFtransf_vars * fftransf_vars , Lattice_arrays * latt_coords ,int icub)

{

  /*

  Computes the particle, anomalous and current densities from the eigenvectors z in 
  
  finite temperature case

  */

  int n , nhalf , iwork ;

  int i , j , li , lj , ii , jj , iu , id , kk1 , nn ;

  int nstart , nstop ;

  double complex * vec , * vec1 , * vec2 , * vec3 ;

  double complex * dx_d , * dy_d , * dz_d , * dx_u , * dy_u , * dz_u ;

  double num_part , num_part_sum , jxy , jxz , jyz , jzy , jyx , jzx , f1 , ft;

  double complex * vvx, * vvy, * vvz;

  double * rho , * occ1 ;

  int * indx , i_tot ;

  nstart = dens->nstart ;

  nstop = dens->nstop ;

  iwork = nstop - nstart ;

  n = 4 * nxyz ;

  nn = 3 * nxyz ;

  nhalf = 2 * nxyz ;

  if( iwork > 0 ) {

    assert( dx_d = malloc( iwork * sizeof( double complex ) ) ) ;

    assert( dy_d = malloc( iwork * sizeof( double complex ) ) ) ;

    assert( dz_d = malloc( iwork * sizeof( double complex ) ) ) ;

    assert( dx_u = malloc( iwork * sizeof( double complex ) ) ) ;

    assert( dy_u = malloc( iwork * sizeof( double complex ) ) ) ;

    assert( dz_u = malloc( iwork * sizeof( double complex ) ) ) ;

  }

  assert( occ1 = malloc( nhalf * sizeof( double ) ) ) ;

  assert( vec = malloc( nn * sizeof( double complex ) ) ) ;

  assert( vec1 = malloc( n * sizeof( double complex ) ) ) ;

  assert( rho = malloc( nxyz * sizeof( double ) ) ) ;     
  

  for( i = 0 ; i < nxyz ; i++ ) 

    *( rho + i ) = 0. ;

  for( i = 0 ; i < iwork ; i++ )

    {

      *( dens->tau + i ) = 0. ;

      *( dens->divjj + i ) = 0. ;

      *( dens->nu + i ) = 0. + I * 0. ;

    }

  * nwf = 0 ;

  for( jj = 0 ; jj < nhalf ; jj++ ) {

    occ1[ jj ] = 0. ;

  }

  f1 = 1.0 ;

  for( jj = nhalf ; jj < n ; jj++ )

    /* construct one vector at a time for positive eigenvalues */

    {

      f1 = sqrt( factor_ec( *( lam + jj ) , e_max ,icub) ) ;

      if( f1 < 1.e-6 ) 

	break ;
      
      ft = 1.0 / (1.0 + exp(-1.0 * *(lam + jj) / temp) );
      
      for( i = 0 ; i < n ; i++ )

	*( vec1 + i ) = 0. + 0. * I ;

#pragma omp parallel for default(shared) private(li,ii,lj,j) 

      for( lj = 0 ; lj < n_iq ; lj++ )

	{

	  j =  i_q * nb + ( lj / nb ) * q_proc * nb + lj % nb ;

	  if( j == jj )

	    for( li = 0 ; li < m_ip ; li++ ) 

	      {

		ii = i_p * mb + ( li / mb ) * p_proc * mb + li % mb ;

		vec1[ ii ] = z[ lj * m_ip + li ] * f1 ;

	      }

	}

      // finite temperature part. I

      MPI_Allreduce( vec1 + nxyz , vec , nn , MPI_DOUBLE_COMPLEX , MPI_SUM , comm ) ;

      gradient_ud( vec + nxyz , nxyz , dx_u , dy_u , dz_u , dx_d , dy_d , dz_d , nstart , nstop , comm , d1_x , d1_y , d1_z , nx , ny , nz ) ;

#pragma omp parallel for default(shared) private(i,iu,id,ii) 
      for( i = nstart ; i < nstop ; i++ )

	{

	  ii = i - nstart ;  

	  iu = i +  nxyz ;  // v - up component

	  id = iu + nxyz ;   // v - down component 

	  occ1[ *nwf ] += ( pow( cabs( vec[ iu ] ) , 2. ) + pow( cabs( vec[ id ] ) , 2. ) )*ft ;

	  *( rho + i ) += ( pow( cabs( vec[ iu ] ) , 2. ) + pow( cabs( vec[ id ] ) , 2. ) )*ft ;
	  
	  *( dens->tau + ii ) += ( pow( cabs( dx_d[ ii ] ) , 2. ) + pow ( cabs( dy_d[ ii ] ) , 2. ) + pow( cabs( dz_d[ ii ] ) , 2. ) + pow( cabs( dx_u[ ii ] ) , 2. ) + pow ( cabs( dy_u[ ii ] ) , 2. ) + pow( cabs( dz_u[ ii ] ) , 2. ) )*ft ;

	  *( dens->nu + ii ) -= ( conj( vec[ i ] ) * vec[ iu ] ) * (2*ft-1) ; 

	  *( dens->divjj + ii ) -= ( cimag( dy_u[ ii ] * conj( dx_u[ ii ] ) - dy_d[ ii ] * conj( dx_d[ ii ] ) + dz_u[ ii ] * conj( dy_d[ ii ] ) - dy_u[ ii ] * conj( dz_d[ ii ] ) ) + creal( dx_d[ ii ] * conj( dz_u[ ii ] ) - dx_u[ ii ] * conj( dz_d[ ii ] ) ) )*ft ;

	}



      // finite temperature part. II
      MPI_Allreduce( vec1, vec , nn , MPI_DOUBLE_COMPLEX , MPI_SUM , comm ) ;

      gradient_ud( vec , nxyz , dx_u , dy_u , dz_u , dx_d , dy_d , dz_d , nstart , nstop , comm , d1_x , d1_y , d1_z , nx , ny , nz ) ;

#pragma omp parallel for default(shared) private(i,iu,id,ii) 
      for( i = nstart ; i < nstop ; i++ )

	{

	  ii = i - nstart ;  

	  iu = i;  // u - up component

	  id = iu + nxyz ;   // u - down component 

	  occ1[ *nwf ] += ( pow( cabs( vec[ iu ] ) , 2. ) + pow( cabs( vec[ id ] ) , 2. ) )*(1-ft) ;

	  *( rho + i ) += ( pow( cabs( vec[ iu ] ) , 2. ) + pow( cabs( vec[ id ] ) , 2. ) )*(1-ft) ;
	  
	  *( dens->tau + ii ) += ( pow( cabs( dx_d[ ii ] ) , 2. ) + pow ( cabs( dy_d[ ii ] ) , 2. ) + pow( cabs( dz_d[ ii ] ) , 2. ) + pow( cabs( dx_u[ ii ] ) , 2. ) + pow ( cabs( dy_u[ ii ] ) , 2. ) + pow( cabs( dz_u[ ii ] ) , 2. ) )*(1-ft) ;

	  *( dens->divjj + ii ) -= ( cimag( dy_u[ ii ] * conj( dx_u[ ii ] ) - dy_d[ ii ] * conj( dx_d[ ii ] ) + dz_u[ ii ] * conj( dy_d[ ii ] ) - dy_u[ ii ] * conj( dz_d[ ii ] ) ) + creal( dx_d[ ii ] * conj( dz_u[ ii ] ) - dx_u[ ii ] * conj( dz_d[ ii ] ) ) )*(1-ft) ;

	}
      

      (*nwf)++ ;

    }

  free( vec ) ; free( vec1 ) ; 

  if( iwork > 0 ) { free( dx_d ) ; free( dy_d ) ; free( dz_d ) ; free( dx_u ) ; free( dy_u ) ; free( dz_u ) ; }

  MPI_Allreduce( occ1 , occ , nhalf , MPI_DOUBLE , MPI_SUM , comm ) ; 

  MPI_Allreduce( rho , dens->rho , nxyz , MPI_DOUBLE , MPI_SUM , comm ) ; 

  free( rho ) ; free( occ1 ) ;

  for( i = 0 ; i < nstop - nstart ; i++)

    dens->divjj[ i ] *= 2. ;

}



double factor_ec( const double e , const double e_cut, int icub)

{

  double e1 ;

  double fact ;

  e1 = ( e - e_cut )/ 0.25 ;

  if( e1 > 40.0 )

    fact = 0. ;

  if( e1 < -40.0 )

    fact = 1. ;

  else

    fact = 1. / ( 1. + exp( e1 ) ) ;

  if(icub == 0) // Spherical cutoff.

    if(e > e_cut)
	fact = 0.;
    else
	fact = 1.;

  else if (icub == 1)
    fact = 1.0; // for cubic-cutoff

  return( fact ) ;

}

void shift_coords( double complex * vec , const int n , FFtransf_vars * fftransf_vars , Lattice_arrays * latt_coords , const double xcm , const double ycm , const double zcm )

{

  int i ;

  double xarg ;

  for( i = 0 ; i < n ; i++ )

    * ( fftransf_vars->buff + i ) = vec[ i ] ;

  fftw_execute( fftransf_vars->plan_f ) ;

  for( i = 0 ; i < n ; i++ )

    {

      xarg = latt_coords->kx[ i ] * xcm + latt_coords->ky[ i ] * ycm + latt_coords->kz[ i ] * zcm ;

      *( fftransf_vars->buff + i ) *= ( cos( xarg ) + I * sin( xarg ) ) ;

    }

  fftw_execute( fftransf_vars->plan_b ) ;

  xarg = 1. / ( ( double ) n ) ;

  for( i = 0 ; i < n ; i++ )

    vec[ i ] = fftransf_vars->buff[ i ] * xarg ;

}

double rescale_dens( Densities * dens , const int nxyz , const double npart , const double dxyz , const int iscale )

{

  int i ;

  double xpart = 0. ;

  for( i = 0 ; i < nxyz ; i++ )

    xpart += *( dens->rho + i ) ;

  xpart = xpart * dxyz / npart ;

  if( iscale == 0 )

    return( xpart ) ;

  for( i = 0 ; i < nxyz ; i++ )

    {

      *( dens->rho + i ) = *( dens->rho + i ) / xpart ;

    }

  return( xpart ) ;

}

void exch_nucl_dens( const MPI_Comm commw , const int ip , const int gr_ip , const int gr_np , const int idim , double * ex_array_p , double * ex_array_n )

{

  MPI_Status istats ;

  int tag1 = 100 , tag2 = 300 ;

  if( ip == gr_ip )

    MPI_Send( ex_array_p , idim , MPI_DOUBLE , gr_ip + gr_np , tag1 , commw ) ;

  if( ip == gr_ip + gr_np ) 

    MPI_Recv( ex_array_p , idim , MPI_DOUBLE , gr_ip , tag1 , commw , &istats ) ; 

  if( ip == gr_ip + gr_np )

    MPI_Send( ex_array_n , idim , MPI_DOUBLE , gr_ip , tag2 , commw ) ;

  if( ip == gr_ip ) 

    MPI_Recv( ex_array_n , idim , MPI_DOUBLE , gr_ip + gr_np , tag2 , commw , &istats ) ; 

}


void grid3( double * xx , double * yy , double * zz , const int nx , const int ny , const int nz , double * X , double * Y , double * Z )

{

  int ix , iy , iz , ixyz ;

  for( ix = 0 ; ix < nx ; ix++)

    for( iy = 0 ; iy < ny ; iy++ )

      for ( iz = 0 ; iz < nz ; iz++ )

	{

	  ixyz = iz + nz * ( iy + ny * ix ) ;

	  X[ ixyz ] = xx[ ix ] ;

	  Y[ ixyz ] = yy[ iy ] ;

	  Z[ ixyz ] = zz[ iz ] ;

	}

}

void divide_work( const int n , const int ip , const int np , int * nstart , int * nstop )
{

  int nav , nspill , i , n0 ;

  nav = n / np ;

  n0 = 0 ;

  nspill = n - np * nav ;

  * nstop = 0 ;

  for( i = 0 ; i <= ip ; i++ )

    {

      * nstart = * nstop ;

      * nstop = * nstart + nav ;

      if( i < nspill ) 

	* nstop = * nstop + 1 ;

    }

}

void array_rescale( double * a , const int n , const double alpha )

{

  int i ;

  for( i = 0 ; i < n ; i++ )

    a[ i ] = alpha * a[ i ] ;

}

void cm_initial( double * lam , double complex * z , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc , const int nx , const int ny , const int nz , const double e_max , Lattice_arrays * latt_coords , double * xcm , double * ycm , double * zcm , const MPI_Comm comm , int icub)

{

  int n ;

  int nxyz = nx * ny * nz ;

  int nhalf , jj , j , i ;

  int li , lj ;

  double f1 , xpart , xcm1 , ycm1 , zcm1 ;

  double wf ;

  xcm1 = 0. ;

  ycm1 = 0. ;

  zcm1 = 0. ;

  xpart = 0. ;

  n = 4 * nxyz ;

  nhalf = 2 * nxyz ;

  for( jj = nhalf ; jj < n ; jj++ )

    /* construgct one vector at a time for positive eigenvalues */

    {

      f1 = sqrt( factor_ec( *( lam + jj ) , e_max ,icub) ) ;

      if( f1 < 1.e-6 ) 

	break ;

      for( lj = 0 ; lj < n_iq ; lj++ )

	{

	  j =  i_q * nb +  ( lj / nb ) * q_proc * nb + lj % nb ;

	  if( j == jj )

	    for( li = 0 ; li < m_ip ; li++ ) 

	      {

		wf = pow( cabs( z[ lj * m_ip + li ] ) * f1 , 2. ) ;

		i = ( i_p * mb + ( li / mb ) * p_proc * mb + li % mb ) ;

		if( i > nhalf - 1 )

		  {

		    i = i % nxyz ;

		    xcm1 += ( latt_coords->xa[ i ] * wf ) ;

		    ycm1 += ( latt_coords->ya[ i ] * wf ) ;

		    zcm1 += ( latt_coords->za[ i ] * wf ) ;

		    xpart += wf ;

		  }

	      }

	}

    }

  MPI_Allreduce( &xpart , xcm , 1 , MPI_DOUBLE , MPI_SUM , comm ) ;

  xpart = * xcm ;

  MPI_Allreduce( &xcm1 , xcm , 1 , MPI_DOUBLE , MPI_SUM , comm ) ;

  MPI_Allreduce( &ycm1 , ycm , 1 , MPI_DOUBLE , MPI_SUM , comm ) ;

  MPI_Allreduce( &zcm1 , zcm , 1 , MPI_DOUBLE , MPI_SUM , comm ) ;

  * xcm /= xpart ;

  * ycm /= xpart ;

  * zcm /= xpart ;

}


void change_mu_eq_sp( double * amu , double * lam , double * occ , const int nwf , const double npart )

{

  double * e_eq , * del2 , e_qp ;

  int i , k ;

  double xpart , dndmu ;

  assert( e_eq = malloc( nwf * sizeof( double ) ) ) ;

  assert( del2 = malloc( nwf * sizeof( double ) ) ) ;

  for( i = 0 ; i < nwf ; ++i )

    {

      e_eq[ i ] = lam[ i ] * ( 1. - 2. * occ[ i ] ) + * amu ;

      del2[ i ] = pow( lam[ i ] , 2. ) - pow( e_eq[ i ] - * amu , 2. ) ;

    }

  for( k = 0 ; k < 51 ; ++k )

    {

      xpart = ( double ) nwf ;

      dndmu = 0. ;

      for( i = 0 ; i < nwf ; ++i )

	{

	  e_qp = sqrt( pow( e_eq[ i ] - * amu , 2. ) + del2[ i ] ) ;

	  xpart -= ( e_eq[ i ] - * amu ) / e_qp ;

	  dndmu += del2[ i ] / pow( e_qp , 3. ) ;

	}

      xpart = xpart / 2. ;

      if( fabs( xpart - npart ) < 1.e-12 )

	{

	  *amu -= 6. * ( xpart - npart ) / npart ;

	  break ;

	}

      dndmu = 1.0 / dndmu ;

      dndmu = minr( 0.8 * dndmu , 3. ) ;

      *amu += dndmu * ( npart - xpart ) ;

    }

  free( del2 ) ; free( e_eq ) ;

}

double minr( double x , double y )

{

  if( x < y )

    return( x ) ;

  else

    return( y ) ;

}


double distMass( double * rho_p, double * rho_n, double n, double z0 , double * za,int *wz ,double dxyz){

  double sum1=0., sum2=0., sum1_=0., sum2_=0.;
  double rho;
  int i;

  for(i=0;i<n;i++){
    rho=rho_p[i]+rho_n[i];
    if( za[i] < z0 ){
      sum1_ += rho;
      sum1 += (z0-za[i]*wz[i])*rho;
    }
    if( za[i] > z0 ){
      sum2_ += rho;
      sum2 += (za[i]*wz[i]-z0)*rho;
    }
    if( za[i]==z0 ){
      sum1_ += .5*rho;
      sum2_ += .5*rho;
    }
  }
  printf("Mass1=%f Mass2=%f\n",sum1_*dxyz,sum2_*dxyz);
  return( sum2/sum2_+sum1/sum1_ );

}
