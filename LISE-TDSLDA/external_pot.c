// for license information, see the accompanying LICENSE file

#include <stdlib.h>

#include <stdio.h>

#include <math.h>

#include <mpi.h>

#include "vars.h"

#include "tdslda_func.h"

void external_pot( const int iext , const int n , const int n_part , const double hbo , double * v_ext , double complex * delta_ext , const double hbar2m , Lattice_arrays * lattice_coords , const double z0 )

{
  /* iext: in, choice for external potentials */
  /* n : in , Nxyz */
  /* N_particles: in , total number of particles, needed for some options */
  /* hbo: in , real parameter, hbar omega for the HO choice */
  /* v_ext: out external potential */
  /* delta_ext: out , external pairing field */

  int i ;

  double hw , r1 , r2 ;

  /* iext = 0 : no external field */

  switch (iext) 

    {

    case 1:   /* HO */

      hw = 0.25 * hbo * hbo / hbar2m ;      
      for( i = 0 ; i < n ; i++ )
	*( v_ext + i ) = hw * ( pow( * ( lattice_coords->xa + i ) , 2. ) + pow( * ( lattice_coords->ya + i ) , 2. ) + pow( * ( lattice_coords->za + i ) , 2. ) ) ;
      break;

    case 2: /* External Woods-Saxon potential, proportional external pairing */

      hw = ( double ) n_part ;
      hw = 1.2 * pow( hw , 1./3. ) ;
      for( i = 0 ; i < n ; i++ )
	{
	  *( v_ext + i ) = -50.0 / ( 1. + exp( sqrt( pow( *( lattice_coords->xa + i ) , 2 ) +pow( *( lattice_coords->ya + i ) , 2. ) + pow( *( lattice_coords->za + i ) , 2.0 ) ) ) ) ;
	  *( delta_ext + i ) = - hbo * *( v_ext + i ) ;
	}
      break;

    case 3:  /* Constant pairing */
      
      for( i = 0 ; i < n ; i++ )
	*( delta_ext + i ) = hbo + 0.0 * I ;
      break;

    case 4:

      for( i = 0 ; i < n ; i++)
	{
	  r1 = sqrt( pow( *( lattice_coords->xa + i ) , 2 ) + pow( *( lattice_coords->ya + i ) , 2. ) + pow( *( lattice_coords->za + i ) + 7.5 , 2.0 ) ) ;
	  r2 = sqrt( pow( *( lattice_coords->xa + i ) , 2 ) + pow( *( lattice_coords->ya + i ) , 2. ) + pow( *( lattice_coords->za + i ) - 7.5 , 2.0 ) ) ;
	  *( v_ext + i ) = -50.0 / ( 1.0 + exp( -2.0 ) * cosh( r1 ) ) - 50.0 / ( 1.0 + exp( -2.0 ) * cosh( r2 ) ) ;
	}
      break;

    case 5:

      for( i = 0 ; i < n ; i++)
	{
	  r1 = sqrt( pow( *( lattice_coords->xa + i ) , 2 ) + pow( *( lattice_coords->ya + i ) , 2. ) + pow( *( lattice_coords->za + i ) + 7.5 , 2.0 ) ) ;
	  r2 = sqrt( pow( *( lattice_coords->xa + i ) , 2 ) + pow( *( lattice_coords->ya + i ) , 2. ) + pow( *( lattice_coords->za + i ) - 7.5 , 2.0 ) ) ;
	  *( v_ext + i ) = -50.0 / ( 1.0 + exp( -2.0 ) * cosh( r1 ) ) - 50.0 / ( 1.0 + exp( -2.0 ) * cosh( r2 ) ) ;
	  *( delta_ext + i ) = 0.05 * *( v_ext + i ) + 0. * I ;
	}
      break;      

    case 6:

      hw = 0.25 * hbo * hbo / hbar2m ;
      for( i = 0 ; i < n ; i++ )
	{
	  *( v_ext + i ) = hw * ( pow( * ( lattice_coords->xa + i ) , 2. ) + pow( * ( lattice_coords->ya + i ) , 2. ) + pow( * ( lattice_coords->za + i ) , 2. ) ) ;
	  *( delta_ext + i ) = 0.1 * hbo ;
	}
      break;

    case 21: 
      for( i=0; i<n ; ++i)
	v_ext[i]=sin(lattice_coords->k1x*lattice_coords->xa[i])/lattice_coords->k1x;
      break;

    case 22:
      for( i=0; i<n ; ++i)
	v_ext[i]=sin(lattice_coords->k1y*lattice_coords->ya[i])/lattice_coords->k1y;
      break;

    case 23:
      for( i=0; i<n ; ++i)
	v_ext[i]=sin(lattice_coords->k1z*lattice_coords->za[i])/lattice_coords->k1z;
      break;

    case 20:
      for( i=0; i<n ; ++i){
	v_ext[i]=2.*pow(sin(lattice_coords->k1z*lattice_coords->za[i])/lattice_coords->k1z,2.)-pow(sin(lattice_coords->k1x*lattice_coords->xa[i])/lattice_coords->k1x,2.)
	  - pow(sin(lattice_coords->k1y*lattice_coords->ya[i])/lattice_coords->k1y,2.);
      }
      break;

    case 30:
      for( i=0; i<n ; ++i){
	double zz=sin(lattice_coords->k1z*lattice_coords->za[i])/lattice_coords->k1z;
	double xx=sin(lattice_coords->k1x*lattice_coords->xa[i])/lattice_coords->k1x;
	double yy=sin(lattice_coords->k1y*lattice_coords->ya[i])/lattice_coords->k1y;
	v_ext[i]=lattice_coords->za[i]*(2.*lattice_coords->za[i]*lattice_coords->za[i]-3.*lattice_coords->xa[i]*lattice_coords->xa[i]-3.*lattice_coords->ya[i]*lattice_coords->ya[i])+.05*sin((lattice_coords->xa[i]*lattice_coords->xa[i]-lattice_coords->ya[i]*lattice_coords->ya[i])*lattice_coords->k1z);
      }
      break;

    case 50: // Gaussian on z direction, cenetered in z0

      for( i=0;i<n;++i){
	double zz=sin(lattice_coords->k1z*lattice_coords->za[i])/lattice_coords->k1z;
	double xx=sin(lattice_coords->k1x*lattice_coords->xa[i])/lattice_coords->k1x;
	double yy=sin(lattice_coords->k1y*lattice_coords->ya[i])/lattice_coords->k1y;
	v_ext[i]= exp(-pow((lattice_coords->za[i]-z0)/3.5,2.0))+.05*(xx*xx-yy*yy)/(xx*xx+yy*yy+1.e-12);
      }
      break;

    case 51: // Gaussian on z direction, cenetered in z0

      for( i=0;i<n;++i){
	double zz=sin(lattice_coords->k1z*lattice_coords->za[i])/lattice_coords->k1z;
	double xx=sin(lattice_coords->k1x*lattice_coords->xa[i])/lattice_coords->k1x;
	double yy=sin(lattice_coords->k1y*lattice_coords->ya[i])/lattice_coords->k1y;
	v_ext[i]=2.*pow(sin(lattice_coords->k1z*lattice_coords->za[i])/lattice_coords->k1z,2.)-pow(sin(lattice_coords->k1x*lattice_coords->xa[i])/lattice_coords->k1x,2.)
	  - pow(sin(lattice_coords->k1y*lattice_coords->ya[i])/lattice_coords->k1y,2.);
	v_ext[i]-= 0.1*exp(-pow((lattice_coords->za[i]-z0)/3.5,2.0))+.05*(xx*xx-yy*yy)/(xx*xx+yy*yy+1.e-12);
      }
      break;


    case 70: // linear potential to compensate the coulomb repulsion (used only in collision)

    for(i=0; i<n; i++)
      {
	double z = lattice_coords->za[i];

	if(z<-25.0)
	  v_ext[i] = -1.0/10.0*pow((z+30.0), 2.0) + 20;

	else if (z<-15.0)
	  v_ext[i] = -z-7.5;

	else if (z<15.0)
	  v_ext[i] = z*z/2.0/15.0;
	
	else if (z<25.0)
	  v_ext[i] = z-7.5;

	else
	  v_ext[i] = -1.0/10.0*pow((z-30.0),2.0) + 20;
      }
    break;

    default:
      for( i=0; i<n ; ++i)
	v_ext[ i ] = 0.;
      break;

    }

}

void external_so_m( double * v_ext , double * mass_eff , double * wx , double * wy , double * wz , const double hbar2m , Lattice_arrays * latt , FFtransf_vars * fftrans , const int nxyz )

{

  int i ;

  double mass = 939.565346 ;

  double hbarc = 197.3269631 ;

  double lambda ;

  gradient_real( v_ext , wx , wy , wz , fftrans , latt , nxyz ) ;

  lambda = 2.5 * pow( .5 * hbarc / mass , 2. ) ; /* half the strength */

  for( i = 0 ; i < nxyz ; i++ )

    {

      mass_eff[ i ] = .5 * hbar2m ;

      *( wx + i ) = lambda * *( wx + i ) ;

      *( wy + i ) = lambda * *( wy + i ) ;

      *( wz + i ) = lambda * *( wz + i ) ;

    }

}

double gaussian(double t,double t0,double sigma,double c0)
{
  double tau=(t-t0)/sigma; //,cc0=c0/(sqrt(2*PI_) * sigma) ;                                                                                                          
  return ( c0*exp(-tau*tau/2.) );
}


int make_boost_quadrupole(double complex *boost_field, const int n, Lattice_arrays * lattice_coords, double alpha)
{
  
  int i;
  // periodic quadrupole field boost

  double Aa = 240.;
  double r0 = 2.*1.12*pow(Aa,1./3.);
  double aa = 0.05;

  for( i=0; i<n ; ++i){
    boost_field[i]= cexp( I*alpha * (2.*pow(sin(lattice_coords->k1z*lattice_coords->za[i])/lattice_coords->k1z,2.)-pow(sin(lattice_coords->k1x*lattice_coords->xa[i])/lattice_coords->k1x,2.) - pow(sin(lattice_coords->k1y*lattice_coords->ya[i])/lattice_coords->k1y,2.)));
  }

  return 0;

}


double transit(double x, double w, double alpha)
{
  double pi = acos(-1.0);
  double zz = alpha*tan(pi*x/w-pi/(double)2.0);
  return (0.5 + 0.5*tanh(zz));

}

int make_boost_twobody(double complex *boost_field, const int n, Lattice_arrays * lattice_coords, double ecm, double A_L, double A_R, double Z_L, double Z_R, double *rcm_L, double *rcm_R, int ip, double b, double ec)
{


  // generate the velocity boost for two colliding nuclei
  // params@boost_field: the generated boost field
  // params@n: nxyz
  // params@lattice_coords: struct of lattice coordinates
  // params@ecm: c.m. energy
  // params@A: mass number of nucleus
  // params@b: impact parameter

  
  int i;

  double hbarc = 197.3269631 ;

  double mass_p = 938.272013 ;
  double mass_n = 939.565346 ;
  double mass = .5 * ( mass_p + mass_n ) ;
  

  // distance between two nuclei
  double distance = sqrt(pow(rcm_L[0]-rcm_R[0],2.0)   
			 + pow(rcm_L[1]-rcm_R[1],2.0)
			 + pow(rcm_L[2]-rcm_R[2],2.0));

  

  if (ec > ecm)
    {
      if (ip == 0)
	printf("Ecm must be higher than the current Coulomb energy ! \n");

      return -1;
      
    }
  double xmu = A_L*A_R/(A_L+A_R);   // reduced mass

  double vel = sqrt(2*(ecm - ec)/xmu/mass);  // relative velocity, in unit of c 

  double v1, v2;   // velocity of 1,2 in cm frame

  v1 = vel * A_R/(A_L+A_R);

  v2 = -1.0 * vel * A_L/(A_L+A_R);  // opposite direction

  double ak1, ak2;

  ak1 = mass*v1 / hbarc;

  ak2 = mass*v2 / hbarc;

  double theta = asin(b/distance); // angle for initial velocity

  double z1 = lattice_coords->za[0];

  double z2 = lattice_coords->za[n-1];

  double alpha = 2.0, ww = 3.75; // width of transition area in the middle
  // for collision: velocity boost: k1z, k2z
  for(i=0;i<n; ++i)
    {
      double za = lattice_coords->za[i];      
      double xa = lattice_coords->xa[i];

      if (za < (-1.0*ww)){	
	boost_field[i] *= cexp(I*ak1*(cos(theta)*za+sin(theta)*xa));
      }

      else if (za < ww)
	{
	  double tmp = (ak2-ak1)*transit(za+ww, 2*ww, alpha) + ak1;
	  boost_field[i] *= cexp(I*tmp*(cos(theta)*za+sin(theta)*xa));
	}
      else 
	boost_field[i] *= cexp(I*ak2*(cos(theta)*za+sin(theta)*xa));

	
    }

  return 0;
}


int make_phase_diff(double complex *boost_field, const int n, Lattice_arrays * lattice_coords, double phi)
{
  // Change the uniform phase of pairing field on the right fragment ...
  double phi_ = phi / 2.0;  // the phase difference on pairing field is 2*phi_
  int i;

  double alpha = 2.0, ww = 3.75; // width of transition area in the middle
  
  for(i=0;i<n;i++)
    {
      double za = lattice_coords->za[i];
      double z1 = lattice_coords->za[0];

      if(za<z1+ww) // left boundary (for transition)
	{
	  double tmp = phi_*(1.0-transit(za-z1, ww, alpha) );
	  boost_field[i] = cexp(I*tmp);
	}
      else if(za < (-1.0*ww))  // left space
	boost_field[i] = 1.0 + 0.0*I;
      else if (za < ww)  // transition space (middle)
	{
	  double tmp = phi_*transit(za+ww, 2*ww, alpha) ;
	  boost_field[i] = cexp(I*tmp);
	}
      else // right space
	{
	  boost_field[i] = cexp(I*phi_);
	}
    }
  
  return 0;
}


int make_boost_onebody(double complex *boost_field, const int n, Lattice_arrays * lattice_coords, double ecm, double A)
{

  // for testing only ...
  // generate the velocity boost for 1 nucleus
  // params@boost_field: the generated boost field
  // params@n: nxyz
  // params@lattice_coords: struct of lattice coordinates
  // params@ecm: c.m. energy
  // params@A: mass number of nucleus
  
  int i;

  double hbarc = 197.3269631 ;

  double mass_p = 938.272013 ;
  double mass_n = 939.565346 ;
  double mass = .5 * ( mass_p + mass_n ) ;
  
  double e2 = 197.3269631 / 137.035999679 ;

  double v = sqrt(2*ecm/A/mass);  // relative velocity, in unit of c 


  double ak = mass*v/ hbarc;

  // for collision: velocity boost: k1z, k2z
  for(i=0;i<n; ++i)
    {
      double za = lattice_coords->za[i];
      
      boost_field[i] = cexp(I*ak*za);
    }


  return 0;
}

void make_localization (double complex *boost_field, const int n, Lattice_arrays * lattice_coords)
{
  double z1 = lattice_coords->za[0];

  double z2 = lattice_coords->za[n-1];

  double ww = 3.75;

  int i;

  for(i=0;i<n;i++)
    {
      double za = lattice_coords->za[i];

      boost_field[i] *= (1.0-2.0/(1.0+exp(pow((za-z1)/ww, 2.0))))  \
	* (1.0-2.0/(1.0+exp(pow((za-z2)/ww, 2.0))));
		
    }
  
}



void get_phase_imprint_potential(double *ext_phase, const int nxyz, Lattice_arrays * latt_coords, double time_interval, double phase_diff)
{
  /* this function prepare a phase imprinting potential, used to introduce a phase difference on 
   * pairing field between two fragments.
   * Because the phase difference in pairing field is 2 U t / hbar, the strength of U is determined by 
   * U = phase_diff * hbar / 2 / t
   */

  const double hbarc = 197.3269631 ;
  int i;

  double z1 = latt_coords->za[0];

  double z2 = latt_coords->za[nxyz-1];

  double ww = 2.5;

  double alpha = 2.0;
  double za;  
  
  double u0 = phase_diff / 2.0 / time_interval * hbarc;
  
  for(i=0;i<nxyz;i++)
    {
      za = latt_coords->za[i];
      if(za <= z1+ww)
	ext_phase[i] += u0*transit(za-z1, ww, alpha);
      else if (za <= 0.0)
	ext_phase[i] += u0;
      else if (za <= ww)
	ext_phase[i] += u0*(1.0-transit(za, ww, alpha));
      else
	ext_phase[i] += 0.0;      
    }
  
}

int read_boost_operator(double complex *boost,char *filename,MPI_Comm commw, int ip, double alpha, int n){

  double *op;
  op=malloc(n*sizeof(double));
  if(ip==0){
    int ifd = f_copn( filename ) ;
    f_crd( ifd , op , sizeof(double) , n);
    f_ccls( &ifd);
  }
  MPI_Bcast(op,n,MPI_DOUBLE,0,commw);
  for(int i=0;i<n;i++){
    boost[i]=cexp( I*alpha*op[i]);
  }
  free(op);

  return 0;

}
