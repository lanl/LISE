// for license information, see the accompanying LICENSE file

/* symmetrized version */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <mpi.h>

#include "nuclear-gpu.h"
#include "vars.h"

#define Tfull 5000.0
#define Tfull2 6500.0
#define Tstop 7000.
#define ALPHA 2.0
#define PI_ 3.141592653589793238462643383279502884197

extern double com_tim;
/*

densities:
0: rho
1: tau
2,3,4: sx,sy,sz
5: divjj
6,7,8: jx,jy,jz
9,10,11: curl(j)_x,curl(j)_y,curl(j)_z
12,13:nu

potentials:
0: u_re
1: mass_eff
2: lapl(mass)
3,4,5: wx,wy,wz
6,7,8: u1_x,u1_y,u1_z
9,10,11: ugrad_x,ugrad_y,ugrad_z
12,13: delta
14:vext
15:vconstr (Qzz in this case)
16: Vcm_x, Vcm_y, Vcm_z, Omega_x, Omega_y, Omega_z  (6 double)
 */

static __device__ __host__ inline cufftDoubleReal gaussian(cufftDoubleReal t, cufftDoubleReal t0,cufftDoubleReal sigma,cufftDoubleReal c0)
{
  cufftDoubleReal tau=(t-t0)/sigma; //,cc0=c0/(sqrt(2*PI_) * sigma) ;
  return ( c0*exp(-tau*tau/(cufftDoubleReal)(2.)) );
}

static __device__ __host__ inline cufftDoubleReal thetaf(cufftDoubleReal t, cufftDoubleReal t0,cufftDoubleReal t1)
{
  if(t<t0)
    return( (cufftDoubleReal)(0.));
  if(t>t1)
    return( (cufftDoubleReal)(1.));
  cufftDoubleReal two=(cufftDoubleReal)(2.);
  cufftDoubleReal pi=(cufftDoubleReal)(acos(-1.));
  return ( pow(sin(pi*(t-t0)/two/(t1-t0)),two));
}

static __device__ __host__ inline cufftDoubleReal non_sym_f(cufftDoubleReal t, cufftDoubleReal Ti,cufftDoubleReal T0,cufftDoubleReal T1,cufftDoubleReal c0)
{
  cufftDoubleReal two=(cufftDoubleReal)(2.);
  cufftDoubleReal pi=(cufftDoubleReal)(acos(-1.));

  if(t<Ti || t > Ti+T0+T1 )
    return ((cufftDoubleReal)(0.));

  if(t<T0+Ti){
    cufftDoubleReal tau=(t-Ti)/T0;
    return( c0*pow(sin(pi*tau/two ),two) );
  }

  cufftDoubleReal tau=(t-Ti-T0)/T1;
  return( c0*pow( cos(pi*tau/two ),two) );

}

static __device__ __host__ inline cufftDoubleReal non_sym_f2(cufftDoubleReal t, cufftDoubleReal Ti,cufftDoubleReal T0,cufftDoubleReal T1,cufftDoubleReal TT,cufftDoubleReal c0)
{
  cufftDoubleReal two=(cufftDoubleReal)(2.);
  cufftDoubleReal pi=(cufftDoubleReal)(acos(-1.));

  if(t<Ti || t > Ti+T0+T1+TT )
    return ((cufftDoubleReal)(0.));

  if(t<T0+Ti){
    cufftDoubleReal tau=(t-Ti)/T0;
    return( c0*pow(sin(pi*tau/two ),two) );
  }else if( t < T0+Ti+TT ){
    return( c0 );
  }
  cufftDoubleReal tau=(t-Ti-T0-TT)/T1;
  return( c0*pow( cos(pi*tau/two ),two) );

}


static __device__ __host__ inline cufftDoubleReal non_sym_f3(cufftDoubleReal t, cufftDoubleReal Ti,cufftDoubleReal T0,cufftDoubleReal T1,cufftDoubleReal TT,cufftDoubleReal c0, cufftDoubleReal c1)
{
  cufftDoubleReal two=(cufftDoubleReal)(2.);
  cufftDoubleReal pi=(cufftDoubleReal)(acos(-1.));

  if(t > Ti+T0+T1+TT )
    return ((cufftDoubleReal)(0.));
  if(t<Ti) return(c0);
  

  if(t<T0+Ti){
    cufftDoubleReal tau=(t-Ti)/T0;
    return( (c1-c0)*pow(sin(pi*tau/two ),two) +c0);
  }else if( t < T0+Ti+TT ){
    return( c1 );
  }
  cufftDoubleReal tau=(t-Ti-T0-TT)/T1;
  return( c1*pow( cos(pi*tau/two ),two) );

}


static __device__ __host__ inline cufftDoubleReal non_sym_f4(cufftDoubleReal t, cufftDoubleReal Ti,cufftDoubleReal T0,cufftDoubleReal c0, cufftDoubleReal c1)
{
  cufftDoubleReal two=(cufftDoubleReal)(2.);
  cufftDoubleReal pi=(cufftDoubleReal)(acos(-1.));

  if(t > Ti+T0 )
    return (c1);
  if(t<Ti) return(c0);
  

  
  cufftDoubleReal tau=(t-Ti)/T0;
  return( (c1-c0)*pow(sin(pi*tau/two ),two) +c0);
  
}


static __device__ __host__ inline cufftDoubleReal non_sym_f_random(cufftDoubleReal t, cufftDoubleReal t0, cufftDoubleReal dt)
{

  cufftDoubleReal two=(cufftDoubleReal)(2.);
  cufftDoubleReal pi=(cufftDoubleReal)(acos(-1.));
  cufftDoubleReal norm = (cufftDoubleReal) 1.0 / (cufftDoubleReal ) sqrt(two*pi) / dt ;
			  
  return( norm * exp((cufftDoubleReal) (-1.0)*pow((t-t0)/dt,two)/two));

}


static __device__ __host__ inline cufftDoubleComplex cplxAdd(cufftDoubleComplex a, cufftDoubleComplex b)
{
  cufftDoubleComplex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

static __device__ __host__ inline cufftDoubleComplex cplxSub(cufftDoubleComplex a, cufftDoubleComplex b)
{
  cufftDoubleComplex c;
  c.x = a.x - b.x;
  c.y = a.y - b.y;
  return c;
}

static __device__ __host__ inline cufftDoubleComplex cplxScale(cufftDoubleComplex a, cufftDoubleReal s)
{
  cufftDoubleComplex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

static __device__ __host__ inline cufftDoubleComplex cplxExpi(double x)
{
    cufftDoubleComplex r;
    r.x=cos(x);
    r.y=sin(x);    
    return r;
}

static __device__ __host__ inline cufftDoubleComplex cplxConj(cufftDoubleComplex a)
{
  a.y=(cufftDoubleReal)(-1.0*a.y);
  return a;
}

static __device__ __host__ inline cufftDoubleReal cplxNorm2(cufftDoubleComplex a)
{ // norm2=|a|^2
  return (cufftDoubleReal)(a.x*a.x + a.y*a.y);
}

static __device__ __host__ inline cufftDoubleReal cplxArg(cufftDoubleComplex a)
{ // norm2=|a|^2

  return (cufftDoubleReal) atan2(a.y, a.x);
}

static __device__ __host__ inline cufftDoubleComplex cplxMul(cufftDoubleComplex a, cufftDoubleComplex b)
{
  cufftDoubleComplex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

__global__ void scale_expansion(cufftDoubleReal * coeff, cufftDoubleComplex * buf, int dim)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex z;
  
  if(idx < dim)
    {            
      z = cplxScale(buf[idx], coeff[0]);
      buf[idx] = z;
    }
}

__global__ void combine_expansion(int mxp, cufftDoubleComplex * wavf_mxp, cufftDoubleComplex * wavf_new, int dim)
/*
 * coeff: coefficient of series expansion
 * buf: the buffer to be scaled
 * dim: the dimension of buffer
 */
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int m;

  if(idx < dim)
    {
      cufftDoubleComplex zz;
      zz.x = 0.0; zz.y = 0;
      cufftDoubleComplex w;
      
      // sum in the inverse sequence: avoid round-off error
      for(m=0; m<mxp; m++)
	{
	  w = cplxAdd(zz, wavf_mxp[idx+(mxp-1-m)*dim]);
	  zz = w;
	}
      
      wavf_new[idx] = zz;
    }
 
}

__global__ void do_mix_potentials( cufftDoubleReal * potentials , cufftDoubleReal * potentials0 , int n , cufftDoubleReal c0 )
{

  int i=blockIdx.x * blockDim.x + threadIdx.x ;
  cufftDoubleReal c1,r;
  c1=(cufftDoubleReal)(1.)-c0;

  if(i<n)
    {
      r=potentials[i];
      potentials[i]=c0*r+c1*potentials0[i];
    }

}

// Pre laplacean for rho contribution to tau.
__global__ void do_pre_laplacean_rho(cufftDoubleComplex * fft3, cufftDoubleReal * rho)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x ;
  cufftDoubleReal zero=(cufftDoubleReal)(0.);
  if ( idx < d_nxyz )
    {
	fft3[idx].x=rho[idx];
	fft3[idx].y=zero;
    }
}

__global__  void do_pre_laplacean_eff_mass( cufftDoubleComplex * fft3 , cufftDoubleReal * density , cufftDoubleReal * j, cufftDoubleReal * potentials) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x ; 
  cufftDoubleReal zero=(cufftDoubleReal)(0.);
  if ( idx < d_nxyz )
    {
      potentials[ idx ] = zero ;
      potentials[ idx + d_nxyz ] =  d_hbar2m + d_c_tau_p * density[ idx ] + d_c_tau_n * density[ idx + 14*d_nxyz];
      fft3[idx].x = d_c_laprho_p * density[ idx ] + d_c_laprho_n * density[ 14*d_nxyz + idx ];
      fft3[idx].y = zero;
      fft3[idx+d_nxyz].x = potentials[ idx + d_nxyz ] ; // this is to compute the laplacean of the mass
      fft3[idx+d_nxyz].y = zero;

      fft3[idx+2*d_nxyz].x = potentials[idx+d_nxyz]*j[idx] ;
      fft3[idx+3*d_nxyz].x = potentials[idx+d_nxyz]*j[idx+d_nxyz] ;
      fft3[idx+4*d_nxyz].x = potentials[idx+d_nxyz]*j[idx+2*d_nxyz] ;
      fft3[idx+2*d_nxyz].y = zero ;
      fft3[idx+3*d_nxyz].y = zero ;
      fft3[idx+4*d_nxyz].y = zero ;
    }
}

__global__ void make_fourierSpace_potentials(cufftDoubleReal *density, cufftDoubleComplex * fft3)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x ; 
  if ( idx < d_nxyz )
    {
      fft3[idx].x =  (d_c_divjj_p * density[ idx ] + d_c_divjj_n * density[ 14*d_nxyz + idx ]) ;
      fft3[idx+d_nxyz].x =  ( d_c_j_p * density[idx+6*d_nxyz] + d_c_j_n * density[idx+20*d_nxyz] ) ;
      fft3[idx+2*d_nxyz].x =  ( d_c_j_p * density[idx+7*d_nxyz] + d_c_j_n * density[idx+21*d_nxyz] ) ;
      fft3[idx+3*d_nxyz].x =  ( d_c_j_p * density[idx+8*d_nxyz] + d_c_j_n * density[idx+22*d_nxyz] ) ;
      fft3[idx+4*d_nxyz].x =  ( d_c_divj_p * density[idx+2*d_nxyz] + d_c_divj_n * density[idx+16*d_nxyz] ) ; //Sx
      fft3[idx+5*d_nxyz].x =  ( d_c_divj_p * density[idx+3*d_nxyz] + d_c_divj_n * density[idx+17*d_nxyz] ) ;
      fft3[idx+6*d_nxyz].x =  ( d_c_divj_p * density[idx+4*d_nxyz] + d_c_divj_n * density[idx+18*d_nxyz] );
      fft3[idx].y =  (cufftDoubleReal)(0.);
      fft3[idx+d_nxyz].y =  (cufftDoubleReal)(0.);
      fft3[idx+2*d_nxyz].y =  (cufftDoubleReal)(0.);
      fft3[idx+3*d_nxyz].y =  (cufftDoubleReal)(0.);
      fft3[idx+4*d_nxyz].y =  (cufftDoubleReal)(0.);
      fft3[idx+5*d_nxyz].y =  (cufftDoubleReal)(0.);
      fft3[idx+6*d_nxyz].y =  (cufftDoubleReal)(0.);
    }
}

__global__ void pre_operators(cufftDoubleComplex * fft3, cufftDoubleReal * kxyz )
{
  //will prepare the arrays to obtain grad(rho) --0,1,2--, div(j) --3--, curl(s) --4,5,6--
  int idx = blockIdx.x * blockDim.x + threadIdx.x ; 
  cufftDoubleComplex rx,ry,rz, divj, csx, csy, csz;
  cufftDoubleComplex zi;

  zi.x=(cufftDoubleReal)(0.);
  zi.y=(cufftDoubleReal)(1.);
  if ( idx < d_nxyz )
    {
      rx=cplxScale(cplxMul(zi,fft3[idx]),kxyz[idx]);            // grad(rho)_z
      ry=cplxScale(cplxMul(zi,fft3[idx]),kxyz[idx+d_nxyz]);     // grad(rho)_y
      rz=cplxScale(cplxMul(zi,fft3[idx]),kxyz[idx+2*d_nxyz]);   // grad(rho)_z

      csx=cplxAdd(cplxScale(fft3[idx+d_nxyz],kxyz[idx]),cplxScale(fft3[idx+2*d_nxyz],kxyz[idx+d_nxyz]));
      divj=cplxAdd(csx,cplxScale(fft3[idx+3*d_nxyz],kxyz[idx+2*d_nxyz]));

      csx = cplxSub(cplxScale(fft3[idx+6*d_nxyz],kxyz[idx+d_nxyz]),cplxScale(fft3[idx+5*d_nxyz],kxyz[idx+2*d_nxyz])); // curl(S)_x
      csy = cplxSub(cplxScale(fft3[idx+4*d_nxyz],kxyz[idx+2*d_nxyz]),cplxScale(fft3[idx+6*d_nxyz],kxyz[idx]));        // curl(S)_y
      csz = cplxSub(cplxScale(fft3[idx+5*d_nxyz],kxyz[idx]),cplxScale(fft3[idx+4*d_nxyz],kxyz[idx+d_nxyz]));          // curl(S)_z

      fft3[idx]=rx;
      fft3[idx+d_nxyz]=ry;
      fft3[idx+2*d_nxyz]=rz;

      fft3[idx+3*d_nxyz]=cplxMul(zi,divj);

      fft3[idx+4*d_nxyz]=cplxMul(zi,csx);      
      fft3[idx+5*d_nxyz]=cplxMul(zi,csy);
      fft3[idx+6*d_nxyz]=cplxMul(zi,csz);
            
    }

}

__global__ void compute_laplMass( cufftDoubleComplex * fft3, cufftDoubleReal * u_im )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
	
  cufftDoubleReal resc = ( ( cufftDoubleReal ) (.5) ) / ( ( cufftDoubleReal )(d_nxyz )) ; 

  if ( i < d_nxyz ) u_im[i] = resc*fft3[i].x;

}

__global__ void compute_w( cufftDoubleComplex * fft3, cufftDoubleReal * w )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
	
  cufftDoubleReal resc = ( ( cufftDoubleReal ) (.5) ) / ( ( cufftDoubleReal )(d_nxyz )) ; 

  if ( i < d_nxyz )
    {
      w[i] = resc*fft3[i].x;
      w[i+d_nxyz] = resc*fft3[i+d_nxyz].x;
      w[i+2*d_nxyz] = resc*fft3[i+2*d_nxyz].x;
    }
}

__global__ void compute_u1( cufftDoubleReal * density, cufftDoubleReal * u1 )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x ;

  if ( i<d_nxyz ) 
    {
      u1[i]= d_c_divj_p * density[i+9*d_nxyz] + d_c_divj_n * density[i+23*d_nxyz] ;
      u1[i+d_nxyz]= d_c_divj_p * density[i+10*d_nxyz] + d_c_divj_n * density[i+24*d_nxyz] ;
      u1[i+2*d_nxyz]= d_c_divj_p * density[i+11*d_nxyz] + d_c_divj_n * density[i+25*d_nxyz] ;
    }
  
}

__global__ void compute_ugrad( cufftDoubleComplex * fft3, cufftDoubleReal * ugrad , cufftDoubleReal * density )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
	
  cufftDoubleReal resc = ( ( cufftDoubleReal ) (.5) ) / ( ( cufftDoubleReal )(d_nxyz )) ; 

  if ( i < d_nxyz )
    {
      ugrad[i] =  fft3[i].x * resc + d_c_j_p * density[i+6*d_nxyz] + d_c_j_n * density[i+20*d_nxyz];
      ugrad[i+d_nxyz] = fft3[i+d_nxyz].x * resc + d_c_j_p * density[i+7*d_nxyz] + d_c_j_n * density[i+21*d_nxyz] ;
      ugrad[i+2*d_nxyz] = fft3[i+2*d_nxyz].x * resc + d_c_j_p * density[i+8*d_nxyz] + d_c_j_n * density[i+22*d_nxyz] ;
    }
  
}

__global__ void add_to_masseff(cufftDoubleReal * mass_eff , cufftDoubleReal alpha)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
	
  cufftDoubleReal r =  alpha;

  if ( i < d_nxyz ) mass_eff[i] += r*d_hbar2m; 
      
}

// add vector coupling fluctuations to ugrad
__global__ void add_to_ugrad(cufftDoubleReal * ugrad , cufftDoubleReal * avf , double factor)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
	
  cufftDoubleReal resc =  factor;

  if ( i < d_nxyz*3 ) ugrad[i] += avf[i] * resc * ( ( cufftDoubleReal ) (.5) ) * d_hbarc;
      
}

// add scalar coupling fluctuations to u
__global__ void add_to_u(cufftDoubleReal * u, cufftDoubleReal *chi , cufftDoubleReal * vfric, double factor)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  cufftDoubleReal r = factor ;
  
  if ( i < d_nxyz ) u[i] += r* d_hbarc*(chi[i] - vfric[i]);//*d_hbarc;
    
  
}

// add fluctuations to pairing field
__global__ void add_to_delta(cufftDoubleComplex * delta, cufftDoubleReal *chi , double factor)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  cufftDoubleReal r = factor ;
  cufftDoubleComplex z;
  if ( i < d_nxyz ) {
    z = cplxScale(delta[i], (cufftDoubleReal) 1.+r*d_hbarc*chi[i]);
    delta[i] = z;
  }
    
  
}

// compute \sum_k Im v_k^*(r) | T H v_k(r) /hbar
// Tv_k = - hbar2m * laplacean v_k
// Hv_k = dv_dt_k * i * hbar
__global__ void sum_TH_local(cufftDoubleComplex *lapl, cufftDoubleComplex * df_dt, cufftDoubleReal * buf, int nxyz){

  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;

  int iwf;

  cufftDoubleReal alpha_;

  cufftDoubleComplex tmp;

  if (ixyz < 4*d_nxyz){
    alpha_ = (cufftDoubleReal) 0.;
    for(iwf=0; iwf<d_nwfip; iwf++){
      tmp = cplxMul( cplxConj(lapl[iwf*nxyz*4+ixyz]), df_dt[iwf*nxyz*4+ixyz]);
      alpha_ -= tmp.x*d_hbar2m*d_dxyz;
    }
    buf[ixyz] = alpha_;
  }
}

__global__ void get_nu_rho_tau_j(cufftDoubleReal * rho , cufftDoubleComplex * nu , cufftDoubleReal * tau, cufftDoubleReal * jx , cufftDoubleReal * jy , cufftDoubleReal * jz , cufftDoubleComplex * grad , cufftDoubleComplex * lapl , cufftDoubleComplex * wavf, cufftDoubleReal * ksq) 
{ 
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  int iwf;
  cufftDoubleReal trho , _tau , _jx , _jy , _jz ;
  cufftDoubleComplex tnu ;
  cufftDoubleComplex z , up , vp , vm , w , wclap;  // wclap will store the conjugate of the wavefunction times its laplacian.
   
  trho = (cufftDoubleReal)(0.) ; 
  _jx = (cufftDoubleReal)(0.) ; 
  _jy = (cufftDoubleReal)(0.) ; 
  _jz = (cufftDoubleReal)(0.) ; 
  _tau = (cufftDoubleReal)(0.) ; 
  tnu.x = (cufftDoubleReal)(0.) ; 
  tnu.y = (cufftDoubleReal)(0.) ;
   
  if ( ixyz < d_nxyz ) 
    {
      for ( iwf = 0 ; iwf < d_nwfip ; iwf++ )
	{
	  up = wavf[ ixyz + iwf * 4 * d_nxyz ] ;
	  vp = wavf[ ixyz + iwf * 4 * d_nxyz + 2 * d_nxyz ] ;
	  vm = wavf[ ixyz + iwf * 4 * d_nxyz + 3 * d_nxyz ] ;
	  trho  += vp.x*vp.x + vp.y*vp.y + vm.x*vm.x + vm.y*vm.y ;
	  tnu.x += vm.x * up.x + vm.y * up.y ;
	  tnu.y += (vm.x * up.y - vm.y * up.x) ;
          //New tau.
          wclap = cplxAdd(cplxMul(cplxConj(vp),lapl[ixyz+iwf*4*d_nxyz + 2*d_nxyz]),cplxMul(cplxConj(vm),lapl[ixyz+iwf*4*d_nxyz+3*d_nxyz]));
          _tau -= wclap.x ;

	  //dX v+
	  w = grad[ ixyz + iwf * 12 * d_nxyz + 6 * d_nxyz ] ;
	  z = cplxMul(w,cplxConj(vp)) ;
	  _jx -= z.y ;
	  //dX v-
	  w = grad[ ixyz + iwf * 12 * d_nxyz + 9 * d_nxyz ] ;
	  z = cplxMul(w,cplxConj(vm)) ;
	  _jx -= z.y ;
	  //dY v+
	  w = grad[ ixyz + iwf * 12 * d_nxyz + 7 * d_nxyz ] ;
	  z = cplxMul(w,cplxConj(vp)) ;
	  _jy -= z.y ;
	  //dY v-
	  w = grad[ ixyz + iwf * 12 * d_nxyz + 10 * d_nxyz ] ;
	  z = cplxMul(w,cplxConj(vm)) ;
	  _jy -= z.y ;
	  //dZ v+
	  w = grad[ ixyz + iwf * 12 * d_nxyz + 8 * d_nxyz ] ;
	  z = cplxMul(w,cplxConj(vp)) ;
	  _jz -= z.y ;
	  //dZ v-
	  w = grad[ ixyz + iwf * 12 * d_nxyz + 11 * d_nxyz ] ;
	  z = cplxMul(w,cplxConj(vm)) ;
	  _jz -= z.y ;
	}
      rho[ ixyz ] = trho ;
      nu[ ixyz ] = tnu ; 
      tau[ ixyz ] = _tau ; 
      jx[ixyz] = _jx ;
      jy[ixyz] = _jy ;
      jz[ixyz] = _jz ;
    }
}


__global__ void get_spin_dens(cufftDoubleComplex * wavf, cufftDoubleReal * sx , cufftDoubleReal * sy , cufftDoubleReal * sz )
{ 
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  int iwf;
  cufftDoubleComplex vp , vm ;
  cufftDoubleReal _sx , _sy , _sz ; 

  _sx = (cufftDoubleReal)(0.);
  _sy = (cufftDoubleReal)(0.);
  _sz = (cufftDoubleReal)(0.);
   
  if ( ixyz < d_nxyz ) 
    {
      for ( iwf = 0 ; iwf < d_nwfip ; iwf++ )
	{
	  vp = cplxMul(cplxConj(wavf[ ixyz + iwf * 4 * d_nxyz + 2 * d_nxyz ]), wavf[ ixyz + iwf * 4 * d_nxyz + 3 * d_nxyz ]) ;
	  _sx += (cufftDoubleReal)(2.0)*vp.x ;
	  _sy -= (cufftDoubleReal)(2.0)*vp.y ;
	  vp = wavf[ ixyz + iwf * 4 * d_nxyz + 2 * d_nxyz ] ;
	  vm = wavf[ ixyz + iwf * 4 * d_nxyz + 3 * d_nxyz ] ;
	  _sz += (vp.x*vp.x + vp.y*vp.y - vm.x*vm.x - vm.y*vm.y) ;
	}
      sx[ixyz] = _sx ;
      sy[ixyz] = _sy ;
      sz[ixyz] = _sz ;
    }
}


__global__ void get_divjj_curlj( cufftDoubleComplex * grad , cufftDoubleReal * divjj , cufftDoubleReal * cjx , cufftDoubleReal * cjy , cufftDoubleReal * cjz ) 
{ 
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  int iwf;
  cufftDoubleReal _divjj , _cjx , _cjy , _cjz ;
  cufftDoubleComplex z , dxvu , dxvd , dyvu , dyvd , dzvu , dzvd ;
   
  _divjj = (cufftDoubleReal)(0.) ; 
  _cjx = (cufftDoubleReal)(0.) ; 
  _cjy = (cufftDoubleReal)(0.) ; 
  _cjz = (cufftDoubleReal)(0.) ; 

  if ( ixyz < d_nxyz ) 
    {
      for ( iwf = 0 ; iwf < d_nwfip ; iwf++ )
	{
	  // v-part
	  dxvu = grad[ ixyz + iwf * 12 * d_nxyz + 6 * d_nxyz ] ;
	  dxvd = grad[ ixyz + iwf * 12 * d_nxyz + 9 * d_nxyz ] ;
	  dyvu = grad[ ixyz + iwf * 12 * d_nxyz + 7 * d_nxyz ] ;
	  dyvd = grad[ ixyz + iwf * 12 * d_nxyz + 10 * d_nxyz ] ;
	  dzvu = grad[ ixyz + iwf * 12 * d_nxyz + 8 * d_nxyz ] ;
	  dzvd = grad[ ixyz + iwf * 12 * d_nxyz + 11 * d_nxyz ] ;
	  z = cplxAdd(cplxSub(cplxMul(dyvu,cplxConj(dxvu)),cplxMul(dyvd,cplxConj(dxvd))), cplxSub(cplxMul(dzvu,cplxConj(dyvd)),cplxMul(dyvu,cplxConj(dzvd)))); 
	  _divjj -= (cufftDoubleReal)(2.0)*z.y ;
	  z = cplxSub(cplxMul(dxvd,cplxConj(dzvu)), cplxMul(dxvu,cplxConj(dzvd)));
	  _divjj -= (cufftDoubleReal)(2.0)*z.x ;
	  // CURL j 
	  z = cplxAdd(cplxSub(cplxMul(cplxConj(dyvu),dzvu),cplxMul(cplxConj(dzvu),dyvu)), cplxSub(cplxMul(cplxConj(dyvd),dzvd), cplxMul(cplxConj(dzvd),dyvd)));
	  _cjx -= z.y ;
	  z = cplxAdd(cplxSub(cplxMul(cplxConj(dzvu),dxvu), cplxMul(cplxConj(dxvu),dzvu)), cplxSub(cplxMul(cplxConj(dzvd),dxvd), cplxMul(cplxConj(dxvd),dzvd)));
	  _cjy -= z.y ;
	  z = cplxAdd(cplxSub(cplxMul(cplxConj(dxvu),dyvu),cplxMul(cplxConj(dyvu),dxvu)), cplxSub(cplxMul(cplxConj(dxvd),dyvd), cplxMul(cplxConj(dyvd),dxvd)));
	  _cjz -= z.y ;

	}
      divjj[ ixyz ] = _divjj ;
      cjx[ ixyz ] = _cjx ;
      cjy[ ixyz ] = _cjy ;
      cjz[ ixyz ] = _cjz ;
    }
}

/* Kernel function of do_get_divj */
__global__ void get_divj( cufftDoubleComplex * lapl , cufftDoubleComplex * wavf, cufftDoubleReal * divj) 
{ 
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  int iwf;
  cufftDoubleReal _divj ;
  cufftDoubleComplex z , vu, vd, d2vu , d2vd;
   
  _divj = (cufftDoubleReal)(0.) ; 
   if ( ixyz < d_nxyz ) 
    {
      for ( iwf = 0 ; iwf < d_nwfip ; iwf++ )
	{
	  // v-part
	  vu = wavf[ ixyz + iwf * 4 * d_nxyz + 2 * d_nxyz ] ;
	  vd = wavf[ ixyz + iwf * 4 * d_nxyz + 3 * d_nxyz ] ;
	  // lapl-v part
	  d2vu = lapl[ ixyz + iwf * 4 * d_nxyz + 2 * d_nxyz ] ;
	  d2vd = lapl[ ixyz + iwf * 4 * d_nxyz + 3 * d_nxyz ] ;

	  z = cplxAdd(cplxMul(d2vu, cplxConj(vu)), cplxMul(d2vd, cplxConj(vd)));
	  _divj -= z.y ;
	  
	}
      divj[ ixyz ] = _divj ;

    }
}
//////////

__global__ void	calculate_qpe(cufftDoubleReal * qpe,cufftDoubleReal * norm,int n)
{

  int i=threadIdx.x + blockIdx.x * blockDim.x;
  if(i<n)
    qpe[i]/=norm[i];

}

__global__ void	calculate_CMvel(cufftDoubleReal * vcm,cufftDoubleReal * part)
{

  int i=threadIdx.x + blockIdx.x * blockDim.x;
  if(i<3){
    vcm[i]*=(((cufftDoubleReal)(2.))*d_hbar2m/part[0]);
  }

}


__global__ void make_dzero( cufftDoubleReal * darray )
{
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  if ( ixyz < d_nxyz ) 
    darray[ ixyz ] = (cufftDoubleReal)(0.) ;
}

__global__ void make_dzero_ln( cufftDoubleReal * darray , int len )
{
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  if ( ixyz < len ) 
    darray[ ixyz ] = (cufftDoubleReal)(0.) ;
}

__global__ void copy_carray( cufftDoubleComplex * zin, cufftDoubleComplex * zout , int nelm )
{ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x ; 
  if ( idx < nelm )
    zout[idx] = zin[idx] ; 
}

__global__ void grad_stepone (cufftDoubleComplex * wfft , cufftDoubleComplex * fft3 , cufftDoubleReal * kxyz , int batch ) 
{
  cufftDoubleComplex w ;
  cufftDoubleReal r ;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int ixyz ;

  if ( i < batch*d_nxyz ) // < 4nxyz by construction
    {
      ixyz = i % d_nxyz;
      r = kxyz[ ixyz ] ;

      w = wfft[i];// ixyz + i * d_nxyz ] ;
      fft3[ i ].x = (cufftDoubleReal)(-1.) * w.y * r ;
      fft3[ i ].y = w.x * r ;
      
    }
} 

__global__ void grad_steptwo( cufftDoubleComplex * grad , cufftDoubleComplex * fft3 , int batch ) 
{
  int ii = threadIdx.x + blockIdx.x * blockDim.x;
  int ixyz ;
  int i;
  /* spatial component -step two of two of gradient computation */
  if ( ii < batch* d_nxyz )
    {
      ixyz = ii % d_nxyz;
      i = ii / d_nxyz;
      grad[ ixyz + i * 3 * d_nxyz ] = cplxScale(fft3[ ii ], (cufftDoubleReal)(1.)/ (cufftDoubleReal)d_nxyz) ;
    }
}

// times -k2
__global__ void lapl_step_one (cufftDoubleComplex * wfft, cufftDoubleComplex * fft3 , cufftDoubleReal * k2 , int batch ) 
{
  cufftDoubleComplex w ;
  cufftDoubleReal r ;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int ixyz ;

  if ( i < d_nxyz * batch) // < 4nxyz by construction
    {
      ixyz = i % d_nxyz;
      r = k2[ ixyz ] ;
      w = wfft[ i ] ;
      fft3[ i ].x = (cufftDoubleReal)(-1.) * w.x * r ;
      fft3[ i ].y = (cufftDoubleReal)(-1.) * w.y * r ;
    }
}

// times 1/nxyz
__global__ void lapl_step_two( cufftDoubleComplex * lapl , cufftDoubleComplex * fft3 , int batch ) 
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  /* spatial component -step two of two of gradient computation */
  if ( i < d_nxyz*batch )
    {
      lapl[ i] = cplxScale(fft3[ i ], (cufftDoubleReal)(1.)/ (cufftDoubleReal)d_nxyz) ;
    }
}
/////////////////////////////////

template <unsigned int blockSize>
__device__ void eReduce(volatile cufftDoubleReal * sdata , unsigned int tid) 
{// extended to block size 64 for warp case --factors of two here 
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; //from 64 to 32 
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
  if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
  if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
  if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

template <unsigned int blockSize>
__global__ void get_energy(cufftDoubleReal * obuf , cufftDoubleReal * rho , cufftDoubleReal * tau , cufftDoubleComplex * nu, cufftDoubleComplex * delta , int nxyz )
{
  unsigned int ixyz = blockIdx.x * (blockDim.x*2) + threadIdx.x ; // for reduce4 and beyond
  int tid = threadIdx.x ; 
  extern __shared__ cufftDoubleReal sbuf[] ;
  cufftDoubleReal q , r , _tau , _rho53;
  cufftDoubleComplex z , c ;
  unsigned int loff=ixyz+blockDim.x;

  if ( loff < nxyz ) 
    {
      _tau = (cufftDoubleReal)(.5) * d_alpha * ( tau[ ixyz ] + tau[ loff ] ) ; // .5 * d_alpha * tau[ ixyz ] 
      _rho53 = (cufftDoubleReal) pow(rho[ ixyz ],(cufftDoubleReal)(1.66666666666667)) + (cufftDoubleReal)pow(rho[ loff ],(cufftDoubleReal)(5./3.)) ; 
      z = cplxMul(delta[ixyz], cplxConj(nu[ixyz]));
      c = cplxMul(delta[loff], cplxConj(nu[loff]));
      r = z.x + c.x ; 
      q = (cufftDoubleReal)(.3) * d_beta * (cufftDoubleReal)pow((cufftDoubleReal)(29.6088132032681), (cufftDoubleReal)(2./3.)) ;
      sbuf[ tid ] = _tau + q * _rho53 - r ; 
    }
  else if ( ixyz < nxyz ) 
    {
      _tau = (cufftDoubleReal)(.5) * d_alpha * tau[ ixyz ] ; // .5 * d_alpha * tau[ ixyz ] 
      _rho53 = (cufftDoubleReal) pow( rho[ ixyz ] , (cufftDoubleReal) (1.66666666666667) ) ; 
      z = cplxMul(delta[ixyz], cplxConj(nu[ixyz ]));
      q = (cufftDoubleReal)(.3) * d_beta * (cufftDoubleReal)pow((cufftDoubleReal)(29.6088132032681), (cufftDoubleReal)(2./3.)) ;
      sbuf[ tid ] = _tau + q * _rho53 - z.x ; 
    }
  else
    {
      sbuf[ tid ] = (cufftDoubleReal)(0.0) ;
    }
  __syncthreads();

  if (blockSize >= 1024) { if (tid < 512) { sbuf[tid] += sbuf[tid + 512]; } __syncthreads(); }
  if (blockSize >=  512) { if (tid < 256) { sbuf[tid] += sbuf[tid + 256]; } __syncthreads(); }
  if (blockSize >=  256) { if (tid < 128) { sbuf[tid] += sbuf[tid + 128]; } __syncthreads(); }
  if (blockSize >=  128) { if (tid <  64) { sbuf[tid] += sbuf[tid +  64]; } __syncthreads(); }
  // at this point only the elements in first 64 indices of sbuf matter
  if (tid < 32) eReduce<blockSize> (sbuf, tid) ;
  if (tid < 16) eReduce<blockSize/2> (sbuf, tid) ;
  if (tid < 8) eReduce<blockSize/4> (sbuf, tid) ;
  if (tid < 4) eReduce<blockSize/8> (sbuf, tid) ;
  if (tid < 2) eReduce<blockSize/16> (sbuf, tid) ;

  if (tid == 0) obuf[blockIdx.x] = sbuf[0];
}

__global__ void dzero_buf( cufftDoubleReal * darray , int nele )
{
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  if ( ixyz < nele ) 
    darray[ ixyz ] = (cufftDoubleReal)(0.) ;
}

__global__ void make_czero( cufftDoubleComplex * array , int nele )
{
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  if ( ixyz < nele ) 
    {
      array[ ixyz ].x = (cufftDoubleReal)(0.) ;
      array[ ixyz ].y = (cufftDoubleReal)(0.) ;
    }
}

__global__ void map_density_to_latt3(cufftDoubleReal * density, cufftDoubleComplex * fft3_c, int * map_sl )
{
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  if ( ixyz < d_nxyz ) 
    {
      fft3_c[ map_sl[ixyz] ].x = density[ixyz];
      fft3_c[ map_sl[ixyz] ].y = (cufftDoubleReal)(0.0);
    }
}

__global__ void fc_step(cufftDoubleComplex * fft3, cufftDoubleReal * fc,int nxyz3)
{
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex z;
  
  if ( ixyz < nxyz3 ) 
    {
      z = fft3[ixyz] ;
      fft3[ ixyz ] = cplxScale(z,fc[ixyz]);
    }
} 

__global__ void map_latt_to_coul(cufftDoubleReal * coulomb, cufftDoubleComplex * fft3_c, int * map_ls , int nxyz3 )
{
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;

  cufftDoubleReal r=(cufftDoubleReal)(1.)/ (cufftDoubleReal)(nxyz3);

  if ( ixyz < nxyz3 ) 
    if ( map_ls[ixyz] >= 0 )
      coulomb[map_ls[ixyz]] = fft3_c[ixyz].x * r ;
}

__global__ void lapl_stepone(cufftDoubleComplex * fft3, cufftDoubleReal * kxyz )
{
  cufftDoubleComplex w ;
  cufftDoubleReal r ;
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal mone=(cufftDoubleReal)(-1.);
  
  if ( ixyz < d_nxyz ) 
    {
      r = mone*kxyz[ixyz + 3 * d_nxyz ];
      w = fft3[ ixyz ] ;
      fft3[ixyz] = cplxScale(w,r);

      w = fft3[ ixyz + d_nxyz ] ;
      fft3[ixyz+d_nxyz]=cplxScale(w,r);

      w = cplxScale(fft3[ ixyz + 2 * d_nxyz ],kxyz[ixyz]) ;
      fft3[ixyz+2*d_nxyz] = cplxAdd(cplxAdd(w,cplxScale(fft3[ixyz+3*d_nxyz] ,kxyz[ixyz+d_nxyz])),cplxScale(fft3[ixyz+4*d_nxyz],kxyz[ixyz+2*d_nxyz]));
    }
}

__global__ void lapl_steptwo(cufftDoubleComplex * fft3, int nele )
{
  cufftDoubleReal r ;
  cufftDoubleComplex w ;
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  
  if ( ixyz < nele ) 
    {
      r = (cufftDoubleReal)(1.) / (cufftDoubleReal)(d_nxyz) ; 
      w = cplxScale(fft3[ixyz],r);
      fft3[ ixyz ] = w;
    }
}

// Computes last correction for tau.
__global__ void get_tau_laplrho(cufftDoubleComplex * fft3, cufftDoubleReal * tau)
{
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal r=(cufftDoubleReal)(1.)/(cufftDoubleReal)(d_nxyz);
  cufftDoubleReal r2=((cufftDoubleReal)(0.5))*r;
  if ( ixyz < d_nxyz ) 
  {
    tau[ixyz] += fft3[ixyz].x * r2 ;
  }
}

__global__ void get_ure_lapl_mass(cufftDoubleComplex * fft3, cufftDoubleReal * potentials, cufftDoubleReal * density, int isospin)
{
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal rho_p, rho_n, rho_0, rho_1 ;
  cufftDoubleReal r=(cufftDoubleReal)(1.)/(cufftDoubleReal)(d_nxyz);
  cufftDoubleReal r2=((cufftDoubleReal)(0.5))*r;

  cufftDoubleReal kk = 2.442749;
  cufftDoubleReal ggp = -292.5417;
  cufftDoubleReal ggn = -225.3672;
  cufftDoubleReal prefactor = kk * (cufftDoubleReal) (8.0) /d_PI/d_dx;

  cufftDoubleReal numerator_p = (cufftDoubleReal) (-1.0)*prefactor*ggp*ggp;
  cufftDoubleReal numerator_n = (cufftDoubleReal) (-1.0)*prefactor*ggn*ggn;
 
  if ( ixyz < d_nxyz ) 
    {

      rho_p = density[ixyz] ;
      rho_n = density[ixyz + 14*d_nxyz];
      rho_0 = rho_n + rho_p;
      rho_1 = rho_p - rho_n;

  cufftDoubleReal mass_eff_p = d_hbar2m + (d_c_tau_0 + d_c_tau_1)*rho_p + (d_c_tau_0 - d_c_tau_1)*rho_n;
  cufftDoubleReal mass_eff_n = d_hbar2m + (d_c_tau_0 - d_c_tau_1)*rho_p + (d_c_tau_0 + d_c_tau_1)*rho_n;
  cufftDoubleComplex nu_p;
  nu_p.x = density[2*ixyz+12*d_nxyz];
  nu_p.y = density[2*ixyz+1+12*d_nxyz];
  cufftDoubleComplex nu_n;
  nu_n.x = density[2*ixyz+26*d_nxyz];
  nu_n.y = density[2*ixyz+1+26*d_nxyz];

      if(d_c_Skyrme){
	potentials[ ixyz ]  += (  (cufftDoubleReal)(2.) * ( d_c_rho_p * rho_p + d_c_rho_n * rho_n + fft3[ixyz].x*r ) 
				+ d_c_tau_p * density[ ixyz + d_nxyz ] + d_c_tau_n * density[ ixyz + 15*d_nxyz ] 
				+ d_c_divjj_p * density[ ixyz + 5*d_nxyz ] + d_c_divjj_n * density[ ixyz + 19*d_nxyz ] 
				+ d_c_gamma_0 * ( d_gamma + (cufftDoubleReal)(2.) ) * pow( rho_p + rho_n , d_gamma + (cufftDoubleReal)(1.) ) 
				+ d_c_gamma_1 * ( d_gamma * pow( rho_p + rho_n + d_eps, d_gamma - (cufftDoubleReal)(1.) ) * pow( rho_p - rho_n , (cufftDoubleReal)(2.) ) 
						  + (cufftDoubleReal)(2.) * ( cufftDoubleReal )(isospin) * pow( rho_p + rho_n, d_gamma ) * ( rho_p - rho_n ) ) - d_amu) ; 
      }
      else{
	potentials[ ixyz ]  += (   
			      (cufftDoubleReal)(2.) * d_c_rho_b0 * rho_0
			      + (cufftDoubleReal)(5./3.)*d_c_rho_a0 * pow(rho_0, (cufftDoubleReal)(2./3.))
			      + (cufftDoubleReal)(7./3.)*d_c_rho_c0 * pow(rho_0, (cufftDoubleReal)(4./3.))			      
			      - (cufftDoubleReal)(1./3.)*d_c_rho_a1 * pow(rho_1, (cufftDoubleReal)(2.0)) / (pow(rho_0, (cufftDoubleReal)(4./3.)) + d_eps)
			      
			      + (cufftDoubleReal)(1./3.)*d_c_rho_c1 * pow(rho_1, (cufftDoubleReal)(2.0)) / (pow(rho_0, (cufftDoubleReal)(2./3.)) + d_eps)
			      
			      - (cufftDoubleReal)(7./3.)*d_c_rho_a2 * pow(rho_1, (cufftDoubleReal)(4.0)) / (pow(rho_0, (cufftDoubleReal)(10./3.)) + d_eps)
			      
			      - (cufftDoubleReal)(2.0)*d_c_rho_b2 * pow(rho_1, (cufftDoubleReal)(4.0)) / (pow(rho_0, (cufftDoubleReal)(3.)) + d_eps)
			      
			      - (cufftDoubleReal)(5./3.)*d_c_rho_c2 *pow(rho_1, (cufftDoubleReal)(4.0)) / (pow(rho_0, (cufftDoubleReal)(8./3.)) + d_eps)
			      
			      // isovector part
			      
			      + d_c_isospin * ( (cufftDoubleReal)(2.) * d_c_rho_a1 * rho_1 / (pow(rho_0, (cufftDoubleReal)(1./3.)) + d_eps)
					    
						+ (cufftDoubleReal)(2.) * d_c_rho_b1 * rho_1
						
						+ (cufftDoubleReal)(2.) * d_c_rho_c1 * rho_1 * pow(rho_0, (cufftDoubleReal)(1./3.))
						
						+ (cufftDoubleReal)(4.) * d_c_rho_a2 * pow(rho_1, (cufftDoubleReal)(3.0)) / (pow(rho_0, (cufftDoubleReal)(7./3.)) + d_eps)
						
						+ (cufftDoubleReal)(4.) * d_c_rho_b2 * pow(rho_1, (cufftDoubleReal)(3.0)) / (pow(rho_0, (cufftDoubleReal)(2.0)) + d_eps)
						
						+ (cufftDoubleReal)(4.) * d_c_rho_c2 * pow(rho_1, (cufftDoubleReal)(3.0)) / (pow(rho_0, (cufftDoubleReal)(5./3.)) + d_eps)
					    
						)
			      + (cufftDoubleReal)(2.) * fft3[ixyz].x*r  
			      + d_c_tau_p * density[ ixyz + d_nxyz ] + d_c_tau_n * density[ ixyz + 15*d_nxyz ] 
			      + d_c_divjj_p * density[ ixyz + 5*d_nxyz ] + d_c_divjj_n * density[ ixyz + 19*d_nxyz ] 
			      - d_amu );
      }
      
      //exchange Coulomb potential
      potentials[ ixyz ] += ( (cufftDoubleReal) (0.5) * (d_c_isospin + (cufftDoubleReal)(1.)) ) * (d_e2*pow(density[ixyz],d_xpow)) * (cufftDoubleReal) d_c_iexcoul;

     cufftDoubleReal denominator_p = pow((mass_eff_p - prefactor*ggp),(cufftDoubleReal)(2.0));
     cufftDoubleReal denominator_n = pow((mass_eff_n - prefactor*ggn),(cufftDoubleReal)(2.0));

     // only cubic case programmed so far.
      if(d_c_Skyrme){
     potentials[ixyz] += numerator_p/denominator_p*d_c_tau_p*pow(cuCabs(nu_p),(cufftDoubleReal)(2.0))
     +numerator_n/denominator_n*d_c_tau_n*pow(cuCabs(nu_n),(cufftDoubleReal)(2.0));
	}

      // laplacean of effective mass
      potentials[ ixyz + 2*d_nxyz ] = fft3[ixyz + d_nxyz ].x * r2 ;
      
    }
}

__global__ void addVF_ure( cufftDoubleReal * potentials , cufftDoubleReal ct , cufftDoubleReal * avf )
{
	
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  // U^A = U - A \cdot U_grad
  if(i<d_nxyz)
    {
      potentials[i]-= ct*(avf[i]*potentials[i+9*d_nxyz]+avf[i+d_nxyz]*potentials[i+10*d_nxyz]+avf[i+2*d_nxyz]*potentials[i+11*d_nxyz]);
    }

}

__global__ void addExtField( cufftDoubleReal * potentials , cufftDoubleReal ct , cufftDoubleReal ct_constr , cufftDoubleReal * rho , cufftDoubleComplex * fft3 )
{
	
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if(i<d_nxyz)
    {
      cufftDoubleReal fact = ( (cufftDoubleReal) 0. ) ; // (25./d_nxyz) ) ; // * potentials[i+d_nxyz]; // / ( rho[i]+1.e-8) ;

      potentials[i]+= ct*( potentials[i+14*d_nxyz] + fft3[i+2*d_nxyz].y * fact ) + ct_constr*potentials[i+15*d_nxyz] ; //*(rho[i]+rho[i+14*d_nxyz]); // fft3 term is cooling
    }

}

__global__ void addVF_uim_ugrad( cufftDoubleReal * potentials , cufftDoubleReal ct , cufftDoubleReal * avf )
{
	
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if(i<d_nxyz)
    {
      cufftDoubleReal ct2=(cufftDoubleReal)(2.)*ct;

      cufftDoubleReal mass_eff=potentials[i+d_nxyz];

      // add vector field to u1 (spin-dependent potential)
      potentials[i+6*d_nxyz] += ct*(potentials[i+4*d_nxyz] * avf[i+2*d_nxyz] - potentials[i+5*d_nxyz] * avf[i+d_nxyz] ) ;
      potentials[i+7*d_nxyz] += ct*(potentials[i+5*d_nxyz] * avf[i] - potentials[i+3*d_nxyz] * avf[i+2*d_nxyz] ) ;
      potentials[i+8*d_nxyz] += ct*(potentials[i+3*d_nxyz] * avf[i+d_nxyz] - potentials[i+4*d_nxyz] * avf[i] ) ;
      
      // add vector field to u_grad
      potentials[i+9*d_nxyz]-= ct*mass_eff*avf[i];
      potentials[i+10*d_nxyz]-= ct*mass_eff*avf[i+d_nxyz];
      potentials[i+11*d_nxyz]-= ct*mass_eff*avf[i+2*d_nxyz];
    }

}


__global__ void make_delta( cufftDoubleReal *rho_p, cufftDoubleReal *rho_n, cufftDoubleReal * potentials , cufftDoubleComplex * nu , cufftDoubleComplex * delta , cufftDoubleReal * cc_pair_qf , cufftDoubleReal ct ,int icub)
{
	
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal p0 , pc ;
  cufftDoubleReal four=(cufftDoubleReal)(4.);
  cufftDoubleReal gg;
	
  if ( i < d_nxyz )
    {
      p0 = sqrt( fabs( potentials[ i ] ) / potentials[ i + d_nxyz ] ) ;
      if ( d_e_cut > potentials[ i ] )			
	{
	  pc = sqrt( ( d_e_cut - potentials[ i ] ) / potentials[ i + d_nxyz ] ) ;
	  if ( potentials[ i ] < (cufftDoubleReal)(0.) )
	    pc -= (cufftDoubleReal)(.5) * p0 * log( ( pc + p0 ) / ( pc - p0 ) ) ;
	  else
	    pc += p0 * atan( p0 / pc ) ;
	}
      else
	pc = (cufftDoubleReal)(0.) ;
      gg=d_gg*((cufftDoubleReal)(1.)-d_rhoc*(rho_p[i]+rho_n[i]));
      cufftDoubleReal xfact=( (cufftDoubleReal)(1.0) - gg * pc / ( four*d_PI * d_PI * potentials[ i + d_nxyz ] ) ) ;
      delta[ i ].x = - nu[ i ].x * gg / xfact ;
      delta[ i ].y = - nu[ i ].y * gg / xfact ;

if(icub==1) {
      cufftDoubleReal kk = 2.442749;

      cufftDoubleReal gg0 = d_gg; 
 
      gg = gg0 / (1.0 - gg0*kk/potentials[i+d_nxyz] /8.0/d_PI/d_dx);

      delta[ i ].x = - nu[ i ].x * gg  ;
      delta[ i ].y = - nu[ i ].y * gg  ;      
    }
    }
}	




__global__ void mult_darray( cufftDoubleReal dnum , cufftDoubleReal * darray ) 
{
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  if ( ixyz < d_nxyz ) 
    darray[ ixyz ] *= dnum ;
}

__global__ void scale_carray( cufftDoubleReal dnum , cufftDoubleComplex * carray , int batch ) 
{ //should pass length but being lazy (is batch*nxyz here ) 
  int ixyz = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex z ;
  if ( ixyz < d_nxyz * batch ) 
    {
      z = carray[ ixyz ] ;
      carray[ ixyz ] = cplxScale(z,dnum);
    }
}

__global__ void copy_current_to_fft3(cufftDoubleReal * j_p,cufftDoubleReal * j_n,cufftDoubleComplex * fft3)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal rzero=(cufftDoubleReal)(0.) ;

  if(i<d_nxyz)
    {
      fft3[i].x=j_p[i];
      fft3[i].y=rzero;
      fft3[i+d_nxyz].x=j_p[i+d_nxyz];
      fft3[i+d_nxyz].y=rzero;
      fft3[i+2*d_nxyz].x=j_p[i+2*d_nxyz];
      fft3[i+2*d_nxyz].y=rzero;
      fft3[i+3*d_nxyz].x=j_n[i];
      fft3[i+3*d_nxyz].y=rzero;
      fft3[i+4*d_nxyz].x=j_n[i+d_nxyz];
      fft3[i+4*d_nxyz].y=rzero;
      fft3[i+5*d_nxyz].x=j_n[i+2*d_nxyz];
      fft3[i+5*d_nxyz].y=rzero;
    }
  
}

__global__ void pre_curl(cufftDoubleComplex * fft3,cufftDoubleReal * kxyz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex jp_x,jp_y,jp_z,jn_x,jn_y,jn_z;
  cufftDoubleReal kx,ky,kz;
  if(i<d_nxyz)
    {
      jp_x=fft3[i];
      jp_y=fft3[i+d_nxyz];
      jp_z=fft3[i+2*d_nxyz];
      jn_x=fft3[i+3*d_nxyz];
      jn_y=fft3[i+4*d_nxyz];
      jn_z=fft3[i+5*d_nxyz];
      kx=kxyz[i];
      ky=kxyz[i+d_nxyz];
      kz=kxyz[i+2*d_nxyz];

      fft3[i]=cplxSub(cplxScale(jp_z,ky),cplxScale(jp_y,kz));
      fft3[i+d_nxyz]=cplxSub(cplxScale(jp_x,kz),cplxScale(jp_z,kx));
      fft3[i+2*d_nxyz]=cplxSub(cplxScale(jp_y,kx),cplxScale(jp_x,ky));

      fft3[i+3*d_nxyz]=cplxSub(cplxScale(jn_z,ky),cplxScale(jn_y,kz));
      fft3[i+4*d_nxyz]=cplxSub(cplxScale(jn_x,kz),cplxScale(jn_z,kx));
      fft3[i+5*d_nxyz]=cplxSub(cplxScale(jn_y,kx),cplxScale(jn_x,ky));

    }
}

__global__ void make_curl(cufftDoubleComplex * fft3,cufftDoubleReal *cj_p,cufftDoubleReal *cj_n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal r=(cufftDoubleReal)(-1.)/(cufftDoubleReal)(d_nxyz);
  if(i<d_nxyz)
    {
      
      cj_p[i]=fft3[i].y*r;
      cj_p[i+d_nxyz]=fft3[i+d_nxyz].y*r;
      cj_p[i+2*d_nxyz]=fft3[i+2*d_nxyz].y*r;

      cj_n[i]=fft3[i+3*d_nxyz].y*r;
      cj_n[i+d_nxyz]=fft3[i+4*d_nxyz].y*r;
      cj_n[i+2*d_nxyz]=fft3[i+5*d_nxyz].y*r;

    }

}

__global__ void get_pred_mod( int nn_time , int n0_time , int n1_time , int m0_time , int m1_time , int m2_time , int m3_time , cufftDoubleComplex * wavf , cufftDoubleComplex * wavf_td , cufftDoubleComplex * wavf_p , cufftDoubleComplex * wavf_c , cufftDoubleComplex * wavf_m )
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int dim = (int)(4 * d_nxyz * d_nwfip) ;

  int nn_offset = nn_time * dim ;
  int n0_offset = n0_time * dim ;
  int n1_offset = n1_time * dim ;
  int m0_offset = m0_time * dim ;
  int m1_offset = m1_time * dim ;
  int m2_offset = m2_time * dim ;
  int m3_offset = m3_time * dim ;

  if ( i < dim )
    {			
      wavf_p[nn_offset+i].x = (cufftDoubleReal) (.5) * ( wavf[n0_offset+i].x + wavf[n1_offset+i].x ) + d_dtstep * ( (cufftDoubleReal) (119.0) * wavf_td[m0_offset+i].x - (cufftDoubleReal) (99.0) * wavf_td[m1_offset+i].x + (cufftDoubleReal) (69.) * wavf_td[m2_offset+i].x - (cufftDoubleReal) (17.) * wavf_td[m3_offset+i].x ) /(cufftDoubleReal) (48.)  ;
      wavf_p[nn_offset+i].y = (cufftDoubleReal) (.5) * ( wavf[n0_offset+i].y + wavf[n1_offset+i].y ) + d_dtstep * ( (cufftDoubleReal) (119.0) * wavf_td[m0_offset+i].y - (cufftDoubleReal) (99.0) * wavf_td[m1_offset+i].y + (cufftDoubleReal) (69.) * wavf_td[m2_offset+i].y - (cufftDoubleReal) (17.) * wavf_td[m3_offset+i].y ) /(cufftDoubleReal) (48.)  ;
      
      wavf_m[i].x = wavf_p[ nn_offset+i].x - (cufftDoubleReal) (161.) * ( wavf_p[n0_offset+i].x - wavf_c[ n0_offset+i].x ) / (cufftDoubleReal) (170.);
      wavf_m[i].y = wavf_p[ nn_offset+i].y - (cufftDoubleReal) (161.) * ( wavf_p[n0_offset+i].y - wavf_c[ n0_offset+i].y ) / (cufftDoubleReal) (170.);
    }
}

__global__ void laplacean_step1( cufftDoubleComplex * fft3,cufftDoubleReal * kxyz,int dim)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i < dim )
    {
      int idx = i - (i/d_nxyz)*d_nxyz ;
      cufftDoubleComplex z=fft3[i];
      fft3[ i ] = cplxScale(z,kxyz[idx]);
    }
}

__global__ void laplacean_step2( cufftDoubleComplex * fft3,int dim)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal resc = ( ( cufftDoubleReal ) (-1.) ) / ( ( cufftDoubleReal )(d_nxyz )) ; 
  if(i<dim)
    {
      cufftDoubleComplex z=fft3[i];
      fft3[ i ] = cplxScale(z,resc);
    }
}

__global__ void get_cy(int nn_time, int n0_time, int n1_time, int m0_time,int m1_time, int m2_time,cufftDoubleComplex * wavf_td,cufftDoubleComplex * wavf,cufftDoubleComplex * wavf_p,cufftDoubleComplex * wavf_c,cufftDoubleComplex * df_dt,int dim,int offset)
{
  int dim2=d_nwfip*4*d_nxyz;
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  int nn_offset = nn_time * dim2 + offset ;
  int n0_offset = n0_time * dim2 + offset ;
  int n1_offset = n1_time * dim2 + offset ;
  int m0_offset = m0_time * dim2 + offset ;
  int m1_offset = m1_time * dim2 + offset ;
  int m2_offset = m2_time * dim2 + offset ;


  if(i<dim)
    {

      wavf_c[nn_offset+i].x = (cufftDoubleReal) (.5) * ( wavf[n0_offset+i].x + wavf[n1_offset+i].x ) + d_dtstep * ( (cufftDoubleReal)(17.0) * df_dt[ i ].x + (cufftDoubleReal) (51.0) * wavf_td[m0_offset+i].x + (cufftDoubleReal) (3.0) * wavf_td[m1_offset+i].x + wavf_td[m2_offset+i].x ) / (cufftDoubleReal) (48.) ;
      wavf_c[nn_offset+i].y = (cufftDoubleReal) (.5) * ( wavf[n0_offset+i].y + wavf[n1_offset+i].y ) + d_dtstep * ( (cufftDoubleReal)(17.0) * df_dt[ i ].y + (cufftDoubleReal) (51.0) * wavf_td[m0_offset+i].y + (cufftDoubleReal) (3.0) * wavf_td[m1_offset+i].y + wavf_td[m2_offset+i].y ) / (cufftDoubleReal) (48.) ;
      
      wavf[nn_offset+i].x = wavf_c[nn_offset+i].x + (cufftDoubleReal)(9.) * ( wavf_p[nn_offset+i].x - wavf_c[nn_offset+i].x ) / (cufftDoubleReal) (170.) ;
      wavf[nn_offset+i].y = wavf_c[nn_offset+i].y + (cufftDoubleReal)(9.) * ( wavf_p[nn_offset+i].y - wavf_c[nn_offset+i].y ) / (cufftDoubleReal) (170.) ;

    }

}


// Function can be used to shift the center of mass of the system.  
__global__ void shiftCM( cufftDoubleComplex * df_dt , cufftDoubleComplex * wf , cufftDoubleComplex * grad , cufftDoubleReal * vcm ){

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j;

  if(i<2*d_nxyz){
    int i1=i;
    if(i>=d_nxyz)
      i1=2*d_nxyz+i; // 3*d_nxyz+(i-d_nxyz)
    for(j=0;j<3;++j){
      df_dt[i].x -= vcm[j]*grad[i1+j*d_nxyz].y;
      df_dt[i].y += vcm[j]*grad[i1+j*d_nxyz].x;
    }
  }
}

// code to reduce the rotation of the global system, author: Shi Jin 08/01/18
/*
 * parameters: df_dt: derivatives of wavefunction (has a factor of i hbar)
 * wf: wavefunction
 * grad: gradient of wf (in x y z)
 * omega: angular velocity
 * xyz: spatial coordinates
 * rcm: c.m. coordinates
 */

__global__ void fix_rotation( cufftDoubleComplex * df_dt , cufftDoubleComplex * wf , cufftDoubleComplex * grad , cufftDoubleReal * omega , cufftDoubleReal *xyz , cufftDoubleReal * rcm ){

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal half = 0.5; // spin
  cufftDoubleComplex lzpsi, lypsi, lxpsi, num;

  cufftDoubleComplex sypsi;
  cufftDoubleComplex zmi,zi;
  zi.x = (cufftDoubleReal) 0.;
  zi.y = (cufftDoubleReal) 1.;
  zmi.x = (cufftDoubleReal) 0.;
  zmi.y = (cufftDoubleReal) -1.;
 
  if(i<2*d_nxyz){
    int i1=i;
    int idx = i % d_nxyz;
    if(i>=d_nxyz)
      i1=2*d_nxyz+i; // 3*d_nxyz+(i-d_nxyz)

    // lzpsi
    lzpsi = cplxSub(cplxScale(grad[i1+d_nxyz],xyz[idx]-rcm[0]),cplxScale(grad[i1],xyz[idx+d_nxyz]-rcm[1]));
    // lypsi
    lypsi = cplxSub(cplxScale(grad[i1],xyz[idx+2*d_nxyz]-rcm[2]),cplxScale(grad[i1+2*d_nxyz],xyz[idx]-rcm[0]));
    // lxpsi
    lxpsi = cplxSub(cplxScale(grad[i1+2*d_nxyz],xyz[idx+d_nxyz]-rcm[1]),cplxScale(grad[i1+d_nxyz],xyz[idx+2*d_nxyz]-rcm[2]));
    
    num = cplxScale( cplxScale(lypsi, omega[1]), d_hbarc);

    df_dt[i].x -= num.y;
    df_dt[i].y += num.x;

    if(i<d_nxyz)
      sypsi = cplxMul(wf[i+d_nxyz], zmi);
    else
      sypsi = cplxMul(wf[i-d_nxyz], zi);

    num = cplxScale( sypsi, omega[1]*d_hbarc*(cufftDoubleReal) 0.5);
    df_dt[i].x -= num.x;
    df_dt[i].y -= num.y;
    
  }
  

}

////////////////////////////////


__global__ void compute_du_dt_step1( cufftDoubleComplex * df_dt , cufftDoubleComplex * u , cufftDoubleComplex * grad , cufftDoubleComplex * lapl , cufftDoubleReal * potentials )
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int ipot = i - (i/d_nxyz)*d_nxyz ; // modulo .

  cufftDoubleComplex zi, zzero ;
  cufftDoubleComplex grad_x,grad_y,grad_z,dudt1,dudt2,dudt_x,dudt_y,dudt_z,z,w;
  cufftDoubleComplex wx,wy,wz;

  zi.x   = (cufftDoubleReal)(0.);
  zi.y   = (cufftDoubleReal)(1.);
  zzero.x= (cufftDoubleReal)(0.);
  zzero.y= (cufftDoubleReal)(0.);

  int k = i / d_nxyz; // 0 if u_up, 1 if u_down

  if ( i < 2*d_nxyz )
    {
      grad_x.x= (cufftDoubleReal)(0.);
      grad_x.y= potentials[ipot+9*d_nxyz];
      grad_y.x= (cufftDoubleReal)(0.);
      grad_y.y= potentials[ipot+10*d_nxyz];
      grad_z.x= (cufftDoubleReal)(0.);
      grad_z.y= potentials[ipot+11*d_nxyz];

      cufftDoubleReal mass_eff=((cufftDoubleReal)(.5))*potentials[ipot+d_nxyz];
      
      dudt1=cplxSub(cplxScale(u[i],potentials[ipot]+potentials[ipot+2*d_nxyz]),cplxScale(lapl[i],mass_eff));
      dudt2=cplxAdd(cplxAdd(cplxMul(grad_x,grad[ipot+3*k*d_nxyz]),cplxMul(grad_y,grad[ipot+d_nxyz+3*k*d_nxyz])),cplxMul(grad_z,grad[ipot+2*d_nxyz+3*k*d_nxyz]));
      df_dt[i]=cplxSub(dudt1,dudt2);

      //spin-orbit and spin-dependent potentials:
      wx.x=(cufftDoubleReal)(0.);
      wx.y=(cufftDoubleReal)(-1.)*potentials[ipot+3*d_nxyz];
      wy.x=(cufftDoubleReal)(0.);
      wy.y=(cufftDoubleReal)(-1.)*potentials[ipot+4*d_nxyz];
      wz.x=(cufftDoubleReal)(0.);
      wz.y=(cufftDoubleReal)(-1.)*potentials[ipot+5*d_nxyz];

      w.x =potentials[ipot+6*d_nxyz];
      w.y =(cufftDoubleReal)(-1.0)*potentials[ipot+7*d_nxyz];

      if ( k==0 )
	{
	  dudt1=cplxAdd(grad[i+d_nxyz],cplxMul(zi,grad[i+5*d_nxyz]));
	  dudt_x=cplxSub(zzero,dudt1);
	  dudt_y=cplxSub(grad[i],grad[i+5*d_nxyz]);
	  dudt_z=cplxAdd(cplxMul(zi,grad[i+3*d_nxyz]),grad[i+4*d_nxyz]);
	  dudt1=cplxAdd(cplxMul(wx,dudt_x),cplxMul(wy,dudt_y));
	  dudt2=cplxAdd(dudt1,cplxMul(wz,dudt_z));  //S-O contribution

	  dudt1=cplxAdd(cplxScale(u[i],potentials[ipot+8*d_nxyz]),cplxMul(w,u[i+d_nxyz]));

	  z=df_dt[i];
	  df_dt[i]=cplxAdd(z,cplxAdd(dudt1,dudt2));
	}
      
      w.y = potentials[ipot+7*d_nxyz];

      if ( k==1 )
	{
	  dudt_x=cplxAdd(cplxMul(zi,grad[ipot+2*d_nxyz]),grad[ipot+4*d_nxyz]);
	  dudt1=cplxAdd(grad[ipot+2*d_nxyz],grad[ipot+3*d_nxyz]);
	  dudt_y=cplxSub(zzero,dudt1);
	  dudt_z=cplxAdd(cplxMul(cplxConj(zi),grad[ipot]),grad[ipot+d_nxyz]);
	  dudt1=cplxAdd(cplxMul(wx,dudt_x),cplxMul(wy,dudt_y));
	  dudt2=cplxAdd(dudt1,cplxMul(wz,dudt_z));  //S-O contribution

	  dudt1=cplxAdd(cplxScale(u[i],(cufftDoubleReal)(-1.0)*potentials[ipot+8*d_nxyz]),cplxMul(w,u[ipot])); 

	  z=df_dt[i];
	  df_dt[i]=cplxAdd(z,cplxAdd(dudt1,dudt2));
	}
    }

}

__global__ void  compute_du_dt_step11(cufftDoubleComplex *fft3_ev,cufftDoubleComplex *u,cufftDoubleReal *potentials)
{

  // completes the application of the symmetrized Hamiltonian -- step1

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int k=i/d_nxyz;

  if(k > 1 )
    return;
  int i1=i;
  if(k==1)
    i1-=d_nxyz;
  cufftDoubleReal mone=(cufftDoubleReal)(-1.);

  cufftDoubleReal wx=potentials[i1+3*d_nxyz];
  cufftDoubleReal wy=potentials[i1+4*d_nxyz];
  cufftDoubleReal wz=potentials[i1+5*d_nxyz];
  cufftDoubleComplex zi;
  zi.x=(cufftDoubleReal)(0.);
  zi.y=(cufftDoubleReal)(1.);


  // prepare the fourier space for the application of derivative terms with u_up, u_down, and laplacean(mass*psi)


  fft3_ev[i+6*d_nxyz]=cplxScale(u[i],((cufftDoubleReal)(0.5))*potentials[i1+d_nxyz]);  // laplacean term
  cufftDoubleComplex z1,z2,z3;

  if(k==0)  // u_up
    {
      z1=cplxScale(u[i1],wy);
      z2=cplxScale(cplxMul(u[i1+d_nxyz],zi),wz);
      z3=cplxScale(u[i],potentials[i1+9*d_nxyz]);
      fft3_ev[i1]=cplxAdd(z1,cplxAdd(z2,z3));

      z1=cplxScale(u[i],mone*wx);
      z2=cplxScale(u[i+d_nxyz],wz);
      z3=cplxScale(u[i],potentials[i1+10*d_nxyz]);
      fft3_ev[i1+d_nxyz]=cplxAdd(z1,cplxAdd(z2,z3));

      cufftDoubleComplex wxy;
      wxy.x=mone*wy;
      wxy.y=mone*wx;
      z1=cplxMul(wxy,u[i+d_nxyz]);
      z3=cplxScale(u[i],potentials[i1+11*d_nxyz]);
      fft3_ev[i1+2*d_nxyz]=cplxAdd(z1,z3);
    }
  else // u_down
    {
      z1=cplxScale(u[i],mone*wy);
      z2=cplxScale(cplxMul(u[i1],zi),mone*wz);
      z3=cplxScale(u[i],potentials[i1+9*d_nxyz]);
      fft3_ev[i1+3*d_nxyz]=cplxAdd(z1,cplxAdd(z2,z3));

      z1=cplxScale(u[i],wx);
      z2=cplxScale(u[i1],wz);
      z3=cplxScale(u[i],potentials[i1+10*d_nxyz]);
      fft3_ev[i1+4*d_nxyz]=cplxAdd(z1,cplxAdd(z2,z3));

      cufftDoubleComplex wxy;
      wxy.x=mone*wy;
      wxy.y=wx;
      z1=cplxMul(wxy,u[i1]);
      z3=cplxScale(u[i],potentials[i1+11*d_nxyz]);
      fft3_ev[i1+5*d_nxyz]=cplxAdd(z1,z3);

    }
}

__global__ void  compute_du_dt_step12(cufftDoubleComplex *fft3_ev,cufftDoubleReal * kxyz)
{

  // completes the application of the symmetrized Hamiltonian -- step2: after the fourier transform, multiply by the appropriate k

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int k=i/d_nxyz;
  if( k < 2 ){
    int i1,i2;
    if(k==0)
      {
	i2=i;
      }
    else
      {
	i2=i-d_nxyz;
      }
    i1=i2+3*k*d_nxyz;

    cufftDoubleComplex z1,z2,z3;
    z1=cplxScale(fft3_ev[i1],kxyz[i2]);
    z2=cplxScale(fft3_ev[i1+d_nxyz],kxyz[i2+d_nxyz]);
    z3=cplxScale(fft3_ev[i1+2*d_nxyz],kxyz[i2+2*d_nxyz]);
    fft3_ev[i1]=cplxAdd(cplxAdd(z1,cplxAdd(z2,z3)),cplxScale(fft3_ev[i+6*d_nxyz],kxyz[i2+3*d_nxyz]));  // this is -laplacean, which is needed in the evolution

  }

}

__global__ void compute_du_dt_step13( cufftDoubleComplex * df_dt , cufftDoubleComplex * fft3_ev )
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int k=i/d_nxyz;
  if(k<2)
    {
      cufftDoubleReal r=((cufftDoubleReal)(1.))/((cufftDoubleReal)(d_nxyz));
      int i1=i;
      if(k==1)
	i1=i-d_nxyz;
      cufftDoubleComplex z1;
      z1=cplxAdd(df_dt[i],cplxScale(fft3_ev[i1+3*k*d_nxyz],r));
      df_dt[i]=z1;
    }

}
  


__global__ void compute_dv_dt_step1( cufftDoubleComplex * df_dt , cufftDoubleComplex * v , cufftDoubleComplex * grad , cufftDoubleComplex * lapl , cufftDoubleReal * potentials )
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int ipot=i-(i/d_nxyz)*d_nxyz;
  cufftDoubleComplex zi ;
  cufftDoubleComplex grad_x,grad_y,grad_z,dvdt1,dvdt2,dvdt_x,dvdt_y,dvdt_z,z,w;
  cufftDoubleComplex wx,wy,wz;

  zi.x=(cufftDoubleReal)(0.);
  zi.y=(cufftDoubleReal)(1.);

  int k=i/d_nxyz; // 0 if u_up, 1 if u_down

  if( i < 2*d_nxyz )
    {
      grad_x.x=(cufftDoubleReal)(0.);
      grad_x.y=(cufftDoubleReal)(-1.0)*potentials[ipot+9*d_nxyz];
      grad_y.x=(cufftDoubleReal)(0.);
      grad_y.y=(cufftDoubleReal)(-1.0)*potentials[ipot+10*d_nxyz];
      grad_z.x=(cufftDoubleReal)(0.);
      grad_z.y=(cufftDoubleReal)(-1.0)*potentials[ipot+11*d_nxyz];
      cufftDoubleReal mass_eff=((cufftDoubleReal)(0.5))*potentials[ipot+d_nxyz];

      dvdt1=cplxSub(cplxScale(lapl[i],mass_eff),cplxScale(v[i],potentials[ipot]+potentials[ipot+2*d_nxyz]));
      dvdt2=cplxAdd(cplxAdd(cplxMul(grad_x,grad[ipot+3*k*d_nxyz]),cplxMul(grad_y,grad[ipot+d_nxyz+3*k*d_nxyz])),cplxMul(grad_z,grad[ipot+2*d_nxyz+3*k*d_nxyz]));
      df_dt[i]=cplxAdd(dvdt2,dvdt1);

      //spin-orbit and spin-dependent potentials:
      wx.x=(cufftDoubleReal)(0.);
      wx.y=(cufftDoubleReal)(-1.0)*potentials[ipot+3*d_nxyz];
      wy.x=(cufftDoubleReal)(0.);
      wy.y=(cufftDoubleReal)(-1.0)*potentials[ipot+4*d_nxyz];
      wz.x=(cufftDoubleReal)(0.);
      wz.y=(cufftDoubleReal)(-1.0)*potentials[ipot+5*d_nxyz];

      w.x=potentials[ipot+6*d_nxyz];
      w.y=potentials[ipot+7*d_nxyz];

      
      if ( k==0 )
	{
	  dvdt_x=cplxSub(cplxMul(zi,grad[i+5*d_nxyz]),grad[i+d_nxyz]);
	  dvdt_y=cplxSub(grad[i],grad[i+5*d_nxyz]);
	  dvdt_z=cplxSub(grad[i+4*d_nxyz],cplxMul(zi,grad[i+3*d_nxyz]));
	  dvdt1=cplxAdd(cplxMul(wx,dvdt_x),cplxMul(wy,dvdt_y));
	  dvdt2=cplxAdd(dvdt1,cplxMul(wz,dvdt_z));  //S-O contribution
	  dvdt1=cplxAdd(cplxScale(v[i],potentials[ipot+8*d_nxyz]),cplxMul(w,v[i+d_nxyz]));

	  z=df_dt[i];
	  df_dt[i]=cplxAdd(z,cplxSub(dvdt2,dvdt1));
	}

      w.y=(cufftDoubleReal)(-1.0)*potentials[ipot+7*d_nxyz];

      if( k==1 )
	{
	  dvdt_x=cplxAdd(cplxMul(cplxConj(zi),grad[ipot+2*d_nxyz]),grad[ipot+4*d_nxyz]);
	  dvdt_y=cplxAdd(grad[ipot+2*d_nxyz],grad[ipot+3*d_nxyz]);
	  dvdt_z=cplxAdd(cplxMul(zi,grad[ipot]),grad[ipot+d_nxyz]);
	  dvdt1=cplxSub(cplxMul(wx,dvdt_x),cplxMul(wy,dvdt_y));
	  dvdt2=cplxAdd(dvdt1,cplxMul(wz,dvdt_z));  //S-O contribution

	  dvdt1=cplxSub(cplxScale(v[i],potentials[ipot+8*d_nxyz]),cplxMul(w,v[ipot]));

	  z=df_dt[i];
	  df_dt[i]=cplxAdd(z,cplxAdd(dvdt2,dvdt1));
	}

    }
}


__global__ void  compute_dv_dt_step11(cufftDoubleComplex *fft3_ev,cufftDoubleComplex *u,cufftDoubleReal *potentials)
{

  // completes the application of the symmetrized Hamiltonian -- step1

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int k=i/d_nxyz;

  if(k > 1 )
    return;
  int i1=i;
  if(k==1)
    i1-=d_nxyz;
  cufftDoubleReal mone=(cufftDoubleReal)(-1.);

  cufftDoubleReal wx=potentials[i1+3*d_nxyz];
  cufftDoubleReal wy=potentials[i1+4*d_nxyz];
  cufftDoubleReal wz=potentials[i1+5*d_nxyz];
  cufftDoubleComplex zi;
  zi.x=(cufftDoubleReal)(0.);
  zi.y=(cufftDoubleReal)(1.);


  // prepare the fourier space for the application of derivative terms with v_up, v_down, and laplacean(mass*psi)


  fft3_ev[i+6*d_nxyz]=cplxScale(u[i],((cufftDoubleReal)(-0.5))*potentials[i1+d_nxyz]);  // laplacean term
  cufftDoubleComplex z1,z2,z3;


  if(k==0)  // v_up
    {
      z1=cplxScale(u[i],wy);
      z2=cplxScale(cplxMul(u[i1+d_nxyz],zi),mone*wz);
      z3=cplxScale(u[i],potentials[i1+9*d_nxyz]);
      fft3_ev[i1]=cplxAdd(z1,cplxAdd(z2,z3));

      z1=cplxScale(u[i1],mone*wx);
      z2=cplxScale(u[i1+d_nxyz],wz);
      z3=cplxScale(u[i],potentials[i1+10*d_nxyz]);
      fft3_ev[i1+d_nxyz]=cplxAdd(z1,cplxAdd(z2,z3));

      cufftDoubleComplex wxy;
      wxy.x=mone*wy;
      wxy.y=wx;
      z1=cplxMul(wxy,u[i1+d_nxyz]);
      z3=cplxScale(u[i],potentials[i1+11*d_nxyz]);
      fft3_ev[i1+2*d_nxyz]=cplxAdd(z1,z3);
    }
  else // v_down
    {
      z1=cplxScale(u[i],mone*wy);
      z2=cplxScale(cplxMul(u[i1],zi),wz);
      z3=cplxScale(u[i],potentials[i1+9*d_nxyz]);
      fft3_ev[i1+3*d_nxyz]=cplxAdd(z1,cplxAdd(z2,z3));

      z1=cplxScale(u[i],wx);
      z2=cplxScale(u[i1],wz);
      z3=cplxScale(u[i],potentials[i1+10*d_nxyz]);
      fft3_ev[i1+4*d_nxyz]=cplxAdd(z1,cplxAdd(z2,z3));

      cufftDoubleComplex wxy;
      wxy.x=mone*wy;
      wxy.y=mone*wx;
      z1=cplxMul(wxy,u[i1]);
      z3=cplxScale(u[i],potentials[i1+11*d_nxyz]);
      fft3_ev[i1+5*d_nxyz]=cplxAdd(z1,z3);

    }
}  


__global__ void compute_du_dt_step2(cufftDoubleComplex * df_dt,cufftDoubleComplex *wavf,cufftDoubleComplex * delta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex z;

  if (i<2*d_nxyz)
    {
      z=df_dt[i];
      if(i<d_nxyz)
	df_dt[i]=cplxAdd(z,cplxMul(delta[i],wavf[i+3*d_nxyz]));
      else
	df_dt[i]=cplxSub(z,cplxMul(delta[i-d_nxyz],wavf[i+d_nxyz]));
    }

}

__global__ void compute_dv_dt_step2(cufftDoubleComplex * df_dt,cufftDoubleComplex *wavf,cufftDoubleComplex * delta)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex z;

  if (idx<2*d_nxyz)
    {
      z=df_dt[idx + 2 * d_nxyz];

      if ( idx < d_nxyz )
	df_dt[ idx + 2 * d_nxyz ] = cplxSub(z,cplxMul(cplxConj(delta[idx]),wavf[idx+d_nxyz]));
      else
	df_dt[ idx + 2 * d_nxyz ] = cplxAdd(z,cplxMul(cplxConj(delta[idx-d_nxyz]),wavf[idx-d_nxyz]));
      
    }

}

__global__ void copy_AxS_to_fft3(cufftDoubleReal * sx,cufftDoubleReal *sy,cufftDoubleReal *sz,cufftDoubleReal *avf,cufftDoubleComplex * fft3)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i<d_nxyz)
    {
      fft3[i].x= avf[i+d_nxyz]*sz[i]-avf[i+2*d_nxyz]*sy[i];
      fft3[i].y=(cufftDoubleReal)(0.);
      fft3[i+d_nxyz].x=avf[i+2*d_nxyz]*sx[i]-avf[i]*sz[i];
      fft3[i+d_nxyz].y=(cufftDoubleReal)(0.);
      fft3[i+2*d_nxyz].x=avf[i]*sy[i]-avf[i+d_nxyz]*sx[i];
      fft3[i+2*d_nxyz].y=(cufftDoubleReal)(0.);
    }
}

__global__ void take_div_step1(cufftDoubleComplex *fft3,cufftDoubleReal * kxyz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex z1,z2;
  if(i<d_nxyz)
    {
      z1=cplxAdd(cplxAdd(cplxScale(fft3[i],kxyz[i]),cplxScale(fft3[i+d_nxyz],kxyz[i+d_nxyz])),cplxScale(fft3[i+2*d_nxyz],kxyz[i+2*d_nxyz]));
      z2=cplxAdd(cplxAdd(cplxScale(fft3[i+3*d_nxyz],kxyz[i]),cplxScale(fft3[i+4*d_nxyz],kxyz[i+d_nxyz])),cplxScale(fft3[i+5*d_nxyz],kxyz[i+2*d_nxyz]));
      fft3[i]=z1;           // extra i from ik
      fft3[i+d_nxyz]=z2;    // extra i from ik
    }  
}

__global__ void update_divjj(cufftDoubleReal *divjj,cufftDoubleReal ct,cufftDoubleComplex * fft3)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal r;
  if(i<d_nxyz)
    {
      r=ct/((cufftDoubleReal)(d_nxyz));
      divjj[i]+=fft3[i].y*r ;   // this already has a minus in it, because of i^2=-1. (i(Re+iIm))
    }
}

__global__ void do_get_VF(cufftDoubleReal *avf,cufftDoubleReal *xyz,int nxyz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal two;
  two=(cufftDoubleReal)(-2.);
  if(i<nxyz){
    avf[i]=two*sin(d_kx_min*xyz[i])/d_kx_min;
    avf[i+nxyz]=two*sin(d_ky_min*xyz[i+nxyz])/d_ky_min;
    avf[i+2*nxyz]=two*two*sin(d_kz_min*xyz[i+2*nxyz])/d_kz_min;
    avf[i+3*nxyz]=two*(cos(d_kx_min*xyz[i])+cos(d_ky_min*xyz[i+nxyz])+two*cos(d_kz_min*xyz[i+2*nxyz]));  // div(A)
  }
}

__global__ void do_get_VF_zero(cufftDoubleReal *avf,cufftDoubleReal *xyz,int nxyz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal zero;
  zero=(cufftDoubleReal)(0.);
  if(i<nxyz){
    avf[i]=zero;
    avf[i+nxyz]=zero;
    avf[i+2*nxyz]=zero;
    avf[i+3*nxyz]=zero;
  }
}

__global__ void dipole_field(cufftDoubleReal *vext,cufftDoubleReal * coord,cufftDoubleReal str,int idir,int nxyz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if(i<nxyz)
    {

      cufftDoubleReal xk;
      switch (idir)
	{
	case (1): xk=d_kx_min; break;
	case (2): xk=d_ky_min; break;
	default:  xk=d_kz_min; break;
	}
      vext[i]=str*sin(xk*coord[i])/xk;
    }  
}


__global__ void quadrupole_field(cufftDoubleReal *vext,cufftDoubleReal *xyz,int nxyz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal one,two;
  one=(cufftDoubleReal)(1.);
  two=(cufftDoubleReal)(2.);
  cufftDoubleReal kx2=d_kx_min*d_kx_min;
  cufftDoubleReal ky2=d_ky_min*d_ky_min; 
  cufftDoubleReal kz2=d_kz_min*d_kz_min;
  if(i<nxyz){
    vext[i]=two*(two*(one-cos(d_kz_min*xyz[i+2*d_nxyz]))/kz2-(one-cos(d_kx_min*xyz[i]))/kx2-(one-cos(d_ky_min*xyz[i+d_nxyz]))/ky2);
  }
}

__global__ void scale_v_by_mass(cufftDoubleComplex *wavf,cufftDoubleComplex *fft3,cufftDoubleReal *mass){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx<d_nxyz){
    fft3[idx].x=mass[idx]*wavf[idx].x;
    fft3[idx].y=mass[idx]*wavf[idx].y;
  }

}

__global__ void rescale_k2(cufftDoubleComplex *fft3,cufftDoubleReal *kxyz,int n){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx<n){
    int i=idx%d_nxyz;
    cufftDoubleComplex z=cplxScale(fft3[idx],kxyz[i]);  // extra minus sign in the routine that puts it together
    fft3[idx]=z;
  }

}

__global__ void save_half_setup_lapl_mv(cufftDoubleComplex *v_down,cufftDoubleComplex * u_up,cufftDoubleComplex *fft3,cufftDoubleComplex *nu,cufftDoubleReal *density_out){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx<d_nxyz){
    cufftDoubleReal r=((cufftDoubleReal) 0.5)/( (cufftDoubleReal) d_nxyz);
    cufftDoubleReal d=density_out[idx];
    cufftDoubleComplex z;
    z.x=u_up[idx].x*fft3[idx].x+u_up[idx].y*fft3[idx].y;
    z.y=u_up[idx].x*fft3[idx].y-u_up[idx].y*fft3[idx].x;
    density_out[idx]=d-r*(z.x*nu[idx].y+z.y*nu[idx].x);
    fft3[idx]=v_down[idx];
  }

}

__global__ void save_m_lapl_v(cufftDoubleComplex *v_down,cufftDoubleComplex * u_up,cufftDoubleComplex *fft3,cufftDoubleReal *density_out,cufftDoubleComplex *nu,cufftDoubleReal *mass){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx<d_nxyz){
    cufftDoubleReal r=((cufftDoubleReal) 0.5 )*mass[idx] /( (cufftDoubleReal) d_nxyz);
    cufftDoubleReal d=density_out[idx];
    cufftDoubleComplex z;
    z.x=u_up[idx].x*fft3[idx].x+u_up[idx].y*fft3[idx].y;
    z.y=u_up[idx].x*fft3[idx].y-u_up[idx].y*fft3[idx].x;
    density_out[idx]=d-(z.x*nu[idx].y+z.y*nu[idx].x)*r;
  }

}

__global__ void make_qzz( cufftDoubleReal * qzz , cufftDoubleReal *cc , cufftDoubleReal * xa , cufftDoubleReal * ya , cufftDoubleReal * za ){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal y0=(cufftDoubleReal) 0.005;

  if( idx < d_nxyz )
    qzz[idx]=cc[0]*(2.*za[idx]*za[idx]-xa[idx]*xa[idx]-ya[idx]*ya[idx])
      +cc[1]*xa[idx]+cc[2]*ya[idx]+cc[3]*za[idx];

}

#ifdef LZCALC
__global__ void compute_lz_dens( cufftDoubleReal * lz_dens , cufftDoubleComplex * grad , cufftDoubleReal * xyz , cufftDoubleReal * rcm ){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if( idx < d_nxyz ){
    cufftDoubleComplex f;
    f=cplxSub(cplxScale(grad[idx+d_nxyz],xyz[idx]-rcm[0]),cplxScale(grad[idx],xyz[idx+d_nxyz]-rcm[1]));
    lz_dens[idx]+=(f.x*f.x+f.y*f.y);
    f=cplxSub(cplxScale(grad[idx+4*d_nxyz],xyz[idx]-rcm[0]),cplxScale(grad[idx+3*d_nxyz],xyz[idx+d_nxyz]-rcm[1]));
    lz_dens[idx]+=(f.x*f.x+f.y*f.y);
  }  

}


__global__ void compute_jz2_dens( cufftDoubleReal * jz_dens , cufftDoubleComplex * grad , cufftDoubleComplex * wavf, cufftDoubleReal * xyz , cufftDoubleReal * rcm ){

  // calculate the jz2 density:
  // \sum_k <v_k| jz^2 |v_k>
  
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex mi;
  mi.x = 0.0; mi.y = -1.0;
  cufftDoubleReal half = 0.5;
  
  if( idx < d_nxyz ){
    cufftDoubleComplex f;
    f=cplxAdd(cplxMul(cplxSub(cplxScale(grad[idx+d_nxyz],xyz[idx]-rcm[0]),cplxScale(grad[idx],xyz[idx+d_nxyz]-rcm[1])), mi), cplxScale(wavf[idx], half));
    jz_dens[idx]+=(f.x*f.x+f.y*f.y);
    f=cplxSub(cplxMul(cplxSub(cplxScale(grad[idx+4*d_nxyz],xyz[idx]-rcm[0]),cplxScale(grad[idx+3*d_nxyz],xyz[idx+d_nxyz]-rcm[1])), mi), cplxScale(wavf[idx+d_nxyz], half));
    jz_dens[idx]+=(f.x*f.x+f.y*f.y);
  }  

}



__global__ void compute_jz_dens( cufftDoubleReal * jz_dens , cufftDoubleComplex * grad , cufftDoubleComplex * wavf, cufftDoubleReal * xyz , cufftDoubleReal * rcm ){

  // calculate the jz density:
  // \sum_k <v_k|jz|v_k>
  
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex mi;
  mi.x = 0.0; mi.y = -1.0;
  cufftDoubleReal half = 0.5;
  
  if( idx < d_nxyz ){
    cufftDoubleComplex f, g;
    // v-up
    f=cplxAdd(cplxMul(cplxSub(cplxScale(grad[idx+d_nxyz],xyz[idx]-rcm[0]),cplxScale(grad[idx],xyz[idx+d_nxyz]-rcm[1])), mi), cplxScale(wavf[idx], half));
    g = wavf[idx];
    jz_dens[idx]+=(f.x*g.x+f.y*g.y);
    // v-down
    f=cplxSub(cplxMul(cplxSub(cplxScale(grad[idx+4*d_nxyz],xyz[idx]-rcm[0]),cplxScale(grad[idx+3*d_nxyz],xyz[idx+d_nxyz]-rcm[1])), mi), cplxScale(wavf[idx+d_nxyz], half));
    g = wavf[idx+d_nxyz];
    jz_dens[idx]+=(f.x*g.x+f.y*g.y);
  }  

}

#endif

#ifdef SAVEOCC
__global__ void compute_lz_wf( cufftDoubleComplex * lzwf  , cufftDoubleComplex * grad , cufftDoubleReal * xyz , cufftDoubleReal * rcm ){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if( idx < d_nxyz ){
    lzwf[idx]=cplxSub(cplxScale(grad[idx+d_nxyz],xyz[idx]-rcm[0]),cplxScale(grad[idx],xyz[idx+d_nxyz]-rcm[1]));
    lzwf[idx+d_nxyz]=cplxSub(cplxScale(grad[idx+4*d_nxyz],xyz[idx]-rcm[0]),cplxScale(grad[idx+3*d_nxyz],xyz[idx+d_nxyz]-rcm[1]));
  }  

}
#endif

__global__ void addWall( cufftDoubleReal * vpot , cufftDoubleReal v0 , cufftDoubleReal rneck, cufftDoubleReal w , cufftDoubleReal z0, cufftDoubleReal *xa, cufftDoubleReal *ya, cufftDoubleReal *za ){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < d_nxyz){

    cufftDoubleReal aa=(cufftDoubleReal)0.65;
    cufftDoubleReal one=(cufftDoubleReal)1.;
    cufftDoubleReal amp=(one+exp((z0-w)/aa))*(one+exp(-(z0+w)/aa));
    cufftDoubleReal kk=one+exp(-rneck/aa);
    cufftDoubleReal rho=sqrt(xa[idx]*xa[idx]+ya[idx]*ya[idx]);
    cufftDoubleReal z=za[idx]-z0;
    cufftDoubleReal vv=vpot[idx];

    vpot[idx]=vv+amp*v0*(1.-kk/((1.+exp((rho-rneck)/aa))))/((1.+exp(-(z+w)/aa))*(1+exp((z-w)/aa)));

  }
}


__global__ void wall_pot( cufftDoubleReal * wall , cufftDoubleReal *cc , cufftDoubleReal * xa , cufftDoubleReal * ya , cufftDoubleReal * za , cufftDoubleReal z0 , cufftDoubleReal rneck, cufftDoubleReal w ){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleReal y0=(cufftDoubleReal) 0.005;
  cufftDoubleReal aa=(cufftDoubleReal) 0.65;

  if( idx < d_nxyz ){
    cufftDoubleReal amp=((cufftDoubleReal)(1.)+exp((z0-w)/aa))*((cufftDoubleReal)(1.)+exp(-(z0+w)/aa));
    cufftDoubleReal kk=pow((cufftDoubleReal)(1.)+exp(-rneck/aa),(cufftDoubleReal)(2.));
    cufftDoubleReal rho=sqrt(xa[idx]*xa[idx]+ya[idx]*ya[idx]);
    cufftDoubleReal z=za[idx]-z0;
    wall[idx]=((cufftDoubleReal)(8.))*amp*((cufftDoubleReal)(1.)-kk/(((cufftDoubleReal)(1.)+exp((rho-rneck)/aa))
	     *((cufftDoubleReal)(1.)-exp(-(rho+rneck)/aa))))/(((cufftDoubleReal)(1.)+exp(-(z+w)/aa))*((cufftDoubleReal)(1.)+exp((z-w)/aa)))
             +cc[1]*xa[idx]+cc[2]*ya[idx]+cc[3]*za[idx];
  }

}

__global__ void remove_qpe(cufftDoubleReal * qpe , cufftDoubleReal * norm , cufftDoubleComplex * df_dt , cufftDoubleComplex * wavf , int dim )
{

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex hi;

  hi.x=(cufftDoubleReal)(0.);
  hi.y=(cufftDoubleReal)(-1.)/d_hbarc;

  if ( idx < dim )
    {
      int i = idx / (4*d_nxyz);
      cufftDoubleComplex z=df_dt[idx];

      df_dt[idx]=cplxMul(cplxSub(z,cplxScale(wavf[idx],qpe[i]/norm[i])),hi);
    }

}


__global__ void boost_wf(cufftDoubleComplex * boost_field ,  cufftDoubleComplex * wavf , int dim )
{

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex w;
  int iwf, ixyz;

  
  if ( idx < dim )
    {
      int i = idx % d_nxyz;   // i < nxyz

      iwf = idx / (4*d_nxyz);   // iwf < nwfip
      
      ixyz = idx - iwf*(4*d_nxyz);  // ixyz < 4*nxyz

      if (ixyz < 2*d_nxyz) // u - component
	w = cplxMul(wavf[idx],boost_field[i]);
      else  // v - component
	w = cplxMul(wavf[idx],cplxConj(boost_field[i]));
      
      wavf[idx] = w;

    }

}

__global__ void boost_wf2(cufftDoubleComplex * boost_field ,  cufftDoubleComplex * wavf , int dim )
{

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  cufftDoubleComplex w;
  cufftDoubleReal tmp, phase;
  int iwf, ixyz;
  if ( idx < dim )
    {
      int i = idx % d_nxyz;   // i < nxyz

      iwf = idx / (4*d_nxyz);   // iwf < nwfip
      
      ixyz = idx - iwf*(4*d_nxyz);  // ixyz < 4*nxyz

      if (ixyz < 2*d_nxyz){ // u - component
	tmp = sqrt(cplxNorm2(wavf[idx]));
	w = cplxScale(cplxExpi(phase* (cufftDoubleReal) 0.8), tmp);
      }
      else{  // v - component
	tmp = sqrt(cplxNorm2(wavf[idx]));
	phase = cplxArg(wavf[idx]);
	w = cplxScale(cplxExpi(phase* (cufftDoubleReal) 0.8), tmp);
      }
      wavf[idx] = w;

    }

}


template <unsigned int blockSize>
__device__ void warpReduce(volatile cufftDoubleReal * sdata, unsigned int tid) 
{// extended to block size 64 for warp case --factors of two here 

    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

template <unsigned int blockSize>
__global__ void __reduce_kernel__(cufftDoubleReal *g_idata, cufftDoubleReal *g_odata, int n)
{
  extern __shared__ cufftDoubleReal sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int loff=i+blockDim.x;
    
  if ( loff < n ) sdata[tid] = g_idata[i] + g_idata[loff]; //let these threads load more than a single element
  else if ( i < n && loff >= n ) sdata[tid] = g_idata[i];
  else sdata[tid] = (cufftDoubleReal)(0.0);
  __syncthreads();
  if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
  if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
  if (tid < 32) warpReduce<blockSize> (sdata, tid) ;

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void qpe_energy_nuc(cufftDoubleReal * obuf , cufftDoubleComplex * wavf, cufftDoubleComplex * dt , int n )
{
  unsigned int ixyz = blockIdx.x * (2*blockDim.x) + threadIdx.x ; // for reduce4 and higher
  unsigned int tid = threadIdx.x;
  unsigned int loff = ixyz + blockDim.x;
  cufftDoubleComplex z ;
  extern __shared__ cufftDoubleReal sbuf[] ;

  if      ( loff < n ) 
    {
      z = cplxAdd(cplxMul(cplxConj( wavf[ ixyz ] ), dt[ixyz] ), cplxMul(cplxConj( wavf[ loff ] ), dt[loff] ));
      sbuf[tid] = z.x;
    }
  else if ( ixyz < n && loff >= n ) 
    {
      z = cplxMul(cplxConj( wavf[ ixyz ] ), dt[ixyz] );
      sbuf[tid] = z.x;
    }
  else sbuf[ tid ] = (cufftDoubleReal)(0.) ;

  __syncthreads();

  if (blockSize >= 1024) { if (tid < 512) { sbuf[tid] += sbuf[tid + 512]; } __syncthreads(); }
  if (blockSize >=  512) { if (tid < 256) { sbuf[tid] += sbuf[tid + 256]; } __syncthreads(); }
  if (blockSize >=  256) { if (tid < 128) { sbuf[tid] += sbuf[tid + 128]; } __syncthreads(); }
  if (blockSize >=  128) { if (tid <  64) { sbuf[tid] += sbuf[tid +  64]; } __syncthreads(); }
  
  if (tid < 32) warpReduce<blockSize> (sbuf, tid) ;
  
  if (tid == 0) obuf[blockIdx.x] = sbuf[0];
}

template <unsigned int blockSize>
__global__ void do_norm(cufftDoubleComplex * wf , cufftDoubleReal * obuf , int n )
{
  extern __shared__ cufftDoubleReal sbuf[] ;
  unsigned int ixyz = blockIdx.x * (blockDim.x*2) + threadIdx.x ; 
  unsigned int tid = threadIdx.x;
  unsigned int loff = ixyz + blockDim.x;

  if      ( loff < n ) sbuf[tid] = cplxNorm2(wf[ixyz]) + cplxNorm2(wf[loff]) ;
  else if ( ixyz < n && loff >= n ) sbuf[tid] = cplxNorm2(wf[ixyz]) ;
  else                 sbuf[tid] = (cufftDoubleReal)(0.) ;
  __syncthreads();

  if (blockSize >= 1024) { if (tid < 512) { sbuf[tid] += sbuf[tid + 512]; } __syncthreads(); }
  if (blockSize >=  512) { if (tid < 256) { sbuf[tid] += sbuf[tid + 256]; } __syncthreads(); }
  if (blockSize >=  256) { if (tid < 128) { sbuf[tid] += sbuf[tid + 128]; } __syncthreads(); }
  if (blockSize >=  128) { if (tid <  64) { sbuf[tid] += sbuf[tid +  64]; } __syncthreads(); }
  if (tid < 32) warpReduce<blockSize> (sbuf, tid) ;

  if (tid == 0) obuf[blockIdx.x] = sbuf[0];
}

template <unsigned int blockSize>
__global__ void do_mom(cufftDoubleReal * j1 , cufftDoubleReal * j2 , cufftDoubleReal * obuf , int n )
{
  extern __shared__ cufftDoubleReal sbuf[] ;
  unsigned int ixyz = blockIdx.x * (blockDim.x*2) + threadIdx.x ; 
  unsigned int tid = threadIdx.x;
  unsigned int loff = ixyz + blockDim.x;

  if      ( loff < n ) sbuf[tid] = j1[ixyz] + j2[ ixyz ] + j1[loff] + j2[loff] ;
  else if ( ixyz < n && loff >= n ) sbuf[tid] = j1[ixyz] + j2[ ixyz ] ; 
  else                 sbuf[tid] = (cufftDoubleReal)(0.) ;
  __syncthreads();

  if (blockSize >= 1024) { if (tid < 512) { sbuf[tid] += sbuf[tid + 512]; } __syncthreads(); }
  if (blockSize >=  512) { if (tid < 256) { sbuf[tid] += sbuf[tid + 256]; } __syncthreads(); }
  if (blockSize >=  256) { if (tid < 128) { sbuf[tid] += sbuf[tid + 128]; } __syncthreads(); }
  if (blockSize >=  128) { if (tid <  64) { sbuf[tid] += sbuf[tid +  64]; } __syncthreads(); }
  if (tid < 32) warpReduce<blockSize> (sbuf, tid) ;

  if (tid == 0) obuf[blockIdx.x] = sbuf[0];
}

// EXTERN C ROUTINES


extern "C" int set_gpu(int device)
{
  cudaError err=cudaSetDevice( device );
  return (int)(err);   
}

void do_make_dens_zero( cufftDoubleReal * darray , int len , int blocks , int threads_per_block ) ;

#ifdef LZCALC

extern "C" void calc_lz_dens( cufftDoubleReal * lz_dens , cufftDoubleReal * dens_lz , cufftDoubleReal * copy_to , cufftDoubleComplex * grad , cufftDoubleReal * xyz , cufftDoubleReal * rcm , int nxyz , int nwfip , MPI_Comm comm ){

  int threads_per_block=512;
  int blocks=(int)ceil((float)nxyz/(float)threads_per_block) ;
  int iwf;

  make_dzero<<<blocks,threads_per_block>>>(lz_dens) ;

  for(iwf=0;iwf<nwfip;iwf++){
    compute_lz_dens<<<blocks,threads_per_block>>>( lz_dens , grad + 6*nxyz*(1+2*iwf) , xyz , rcm );
  }
  cudaMemcpy( (void *)copy_to , (const void *)lz_dens , nxyz*sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost ) ;
  MPI_Reduce( copy_to , dens_lz , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm );

}


extern "C" void calc_jz2_dens( cufftDoubleReal * jz2_dens , cufftDoubleReal * dens_jz2 , cufftDoubleReal * copy_to , cufftDoubleComplex * grad , cufftDoubleComplex * wavf, cufftDoubleReal * xyz , cufftDoubleReal * rcm , int nxyz , int nwfip , MPI_Comm comm ){

  int threads_per_block=512;
  int blocks=(int)ceil((float)nxyz/(float)threads_per_block) ;
  int iwf;

  make_dzero<<<blocks,threads_per_block>>>(jz2_dens) ;

  for(iwf=0;iwf<nwfip;iwf++){
    compute_jz2_dens<<<blocks,threads_per_block>>>( jz2_dens , grad + 6*nxyz*(1+2*iwf), wavf + 2*nxyz*(1+2*iwf) , xyz , rcm );
  }
  cudaMemcpy( (void *)copy_to , (const void *)jz2_dens , nxyz*sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost ) ;
  MPI_Reduce( copy_to , dens_jz2 , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm );

}

extern "C" void calc_jz_dens( cufftDoubleReal * jz_dens , cufftDoubleReal * dens_jz , cufftDoubleReal * copy_to , cufftDoubleComplex * grad , cufftDoubleComplex * wavf, cufftDoubleReal * xyz , cufftDoubleReal * rcm , int nxyz , int nwfip , MPI_Comm comm ){

  int threads_per_block=512;
  int blocks=(int)ceil((float)nxyz/(float)threads_per_block) ;
  int iwf;

  make_dzero<<<blocks,threads_per_block>>>(jz_dens) ;

  for(iwf=0;iwf<nwfip;iwf++){
    compute_jz_dens<<<blocks,threads_per_block>>>( jz_dens , grad + 6*nxyz*(1+2*iwf), wavf + 2*nxyz*(1+2*iwf) , xyz , rcm );
  }
  cudaMemcpy( (void *)copy_to , (const void *)jz_dens , nxyz*sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost ) ;
  MPI_Reduce( copy_to , dens_jz , nxyz , MPI_DOUBLE , MPI_SUM , 0 , comm );

}

#endif

extern "C" void make_qzz_gpu( cufftDoubleReal * qzz , cufftDoubleReal * cc_constr , cufftDoubleReal *xyz , int nxyz , int iext , cufftDoubleReal v0 , cufftDoubleReal rneck , cufftDoubleReal wneck , cufftDoubleReal z0 ){

  int threads_per_block=512;
  int blocks=ceil((float)(nxyz)/(float)threads_per_block);

  make_qzz<<<blocks,threads_per_block>>>(qzz,cc_constr,xyz,xyz+nxyz,xyz+2*nxyz);

  if(iext==60)
    addWall<<<blocks,threads_per_block>>>(qzz,v0,rneck,wneck,z0,xyz,xyz+nxyz,xyz+2*nxyz);

  return;

}

extern "C" void make_wallpot_gpu( cufftDoubleReal * qzz , cufftDoubleReal * cc_constr , cufftDoubleReal *xyz , cufftDoubleReal z0, int nxyz , cufftDoubleReal rneck , cufftDoubleReal wneck ){

  int threads_per_block=512;
  int blocks=ceil((float)(nxyz)/(float)threads_per_block);

  wall_pot<<<blocks,threads_per_block>>>(qzz,cc_constr,xyz,xyz+nxyz,xyz+2*nxyz,z0,rneck,wneck);

  return;

}

void cooling_pairing_cc(int nwfip,cufftDoubleReal *potentials,int nxyz,cufftDoubleComplex * nu,cufftDoubleComplex * fft3,cufftHandle plan_b,cufftDoubleComplex * wavf,cufftDoubleReal * kxyz,int isospin,int batch,cufftDoubleReal *cc_pair_qf,cufftDoubleReal *copy_from, cufftDoubleReal *copy_to,MPI_Comm comm){

  int chunks=(nwfip)/batch; // warning - chunks and batch explicitly related !!!                            
  int ispill=(nwfip)%batch;
  int threads_per_block=512;

  int blocks=(int) ceil((float)(nxyz)/(float)threads_per_block);
  int blocks2=(int) ceil((float)(batch*nxyz)/(float)threads_per_block);

  int i,iwf,ioffset;

  for( i=0;i<chunks;i++){

    for(iwf=0;iwf<batch;iwf++){
      scale_v_by_mass<<<blocks,threads_per_block>>>(wavf+(i*batch+iwf)*4*nxyz+3*nxyz,fft3+iwf*nxyz,potentials+nxyz);
    }
    if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error in add_cooling_pairing_cc(): cufftExecZ2Z FORWARD failed\n") ;

    rescale_k2<<<blocks2,threads_per_block>>>(fft3,kxyz,batch*nxyz);

    if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;

    for(iwf=0;iwf<batch;iwf++){
      ioffset=(i*batch+iwf)*4*nxyz;
      save_half_setup_lapl_mv<<<blocks,threads_per_block>>>(wavf+ioffset+3*nxyz,wavf+ioffset,fft3+iwf*nxyz,nu,cc_pair_qf);
    }

    if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error in add_cooling_pairing_cc(): cufftExecZ2Z FORWARD failed\n") ;

    rescale_k2<<<blocks2,threads_per_block>>>( fft3,kxyz,batch*nxyz);

    if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;

    for(iwf=0;iwf<batch;iwf++){
      ioffset=(i*batch+iwf)*4*nxyz;
      save_m_lapl_v<<<blocks,threads_per_block>>>(wavf+ioffset+3*nxyz,wavf+ioffset,fft3+iwf*nxyz,cc_pair_qf,nu,potentials+nxyz);

    }

  }
  
  if(ispill!=0){

    int ioffset1= chunks*batch*4*nxyz;

    for(iwf=0;iwf<ispill;iwf++){
      ioffset = ioffset1+ iwf*4*nxyz+3*nxyz;
      scale_v_by_mass<<<blocks,threads_per_block>>>(wavf+ioffset,fft3+iwf*nxyz,potentials+nxyz);
    }
    if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error in add_cooling_pairing_cc(): cufftExecZ2Z FORWARD failed\n") ;

    blocks2=(int) ceil((float)(ispill*nxyz)/(float)threads_per_block);
    rescale_k2<<<blocks2,threads_per_block>>>(fft3,kxyz,ispill*nxyz);

    if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;

    for(iwf=0;iwf<ispill;iwf++){
      ioffset = ioffset1+iwf*4*nxyz;
      save_half_setup_lapl_mv<<<blocks,threads_per_block>>>(wavf+ioffset+3*nxyz,wavf+ioffset,fft3+iwf*nxyz,nu,cc_pair_qf);
    }

    if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error in add_cooling_pairing_cc(): cufftExecZ2Z FORWARD failed\n") ;

    rescale_k2<<<blocks2,threads_per_block>>>( fft3,kxyz,ispill*nxyz);

    if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;

    for(iwf=0;iwf<ispill;iwf++){
      ioffset= ioffset1+iwf*4*nxyz;
      save_m_lapl_v<<<blocks,threads_per_block>>>(wavf+ioffset+3*nxyz,wavf+ioffset,fft3+iwf*nxyz,cc_pair_qf,nu,potentials+nxyz);
    }  
  }

  cudaMemcpy( (void *)copy_to , (const void *)cc_pair_qf , nxyz*sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost ) ;

  MPI_Barrier(comm);

  MPI_Allreduce( copy_to , copy_from , nxyz , MPI_DOUBLE , MPI_SUM , comm ) ;
  cudaMemcpy( cc_pair_qf , copy_from , nxyz*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ;

}

//basically the same ideas as the global timer but now internal scope of variables 
extern "C" void b_it( double *itgtod, clock_t *it_clock );

extern "C" double e_it( int type, double *itgtod, clock_t *it_clock ) ; 

extern "C" cufftDoubleReal non_sym_f_cpu(cufftDoubleReal t, cufftDoubleReal Ti,cufftDoubleReal T0,cufftDoubleReal T1,cufftDoubleReal TT,cufftDoubleReal c0)
{
  cufftDoubleReal Ctime=non_sym_f2(t,Ti,T0,T1,TT,c0);
  return Ctime;
}

// determine the strength of random kicks
extern "C" cufftDoubleReal non_sym_f_cpu_random(cufftDoubleReal t, cufftDoubleReal t0, cufftDoubleReal dt)
{  
  cufftDoubleReal Ctime=non_sym_f_random(t, t0, dt);
  return Ctime;
}


// determine the strength of external field by time moment
extern "C" cufftDoubleReal non_sym_f_cpu_collision_time(cufftDoubleReal t, cufftDoubleReal Ti,cufftDoubleReal T0,cufftDoubleReal T1,cufftDoubleReal TT,cufftDoubleReal c0, cufftDoubleReal c1)
{  
  cufftDoubleReal Ctime=non_sym_f3(t,Ti,T0,T1,TT,c0,c1);
  return Ctime;
}

// determine the strength of external field by c.m. energy
extern "C" cufftDoubleReal non_sym_f_cpu_collision_ecm(cufftDoubleReal t, cufftDoubleReal Ti,cufftDoubleReal T0, cufftDoubleReal c0, cufftDoubleReal c1)
{
  cufftDoubleReal Ctime=non_sym_f4(t,Ti,T0,c0,c1);
  return Ctime;
}


extern "C" cufftDoubleReal gaussian_(cufftDoubleReal t,cufftDoubleReal t0,cufftDoubleReal sigma,cufftDoubleReal c0)
{
  cufftDoubleReal Ctime=gaussian(t,t0,sigma,c0);
  return Ctime;
}

extern "C" cufftDoubleReal theta(cufftDoubleReal t,cufftDoubleReal t0,cufftDoubleReal t1)
{
  cufftDoubleReal Ctime=thetaf(t,t0,t1);
  return Ctime;
}

extern "C" void chk_constants( int * nwfip , int * nxyz )
{   if( cudaMemcpyFromSymbol( nxyz , d_nxyz , sizeof(int), (size_t)0 , cudaMemcpyDeviceToHost )!= cudaSuccess ) printf("error: failed MemcpyFromSymbol\n") ; 
  if( cudaMemcpyFromSymbol( nwfip , d_nwfip , sizeof(int), (size_t)0 , cudaMemcpyDeviceToHost )!= cudaSuccess ) printf("error: failed MemcpyFromSymbol\n") ; 
}

extern "C" void shi_copy( double xpow, double e2)
{
  cudaMemcpyToSymbol(d_xpow, &xpow, sizeof(xpow));
  cudaMemcpyToSymbol(d_e2, &e2, sizeof(e2));
}

extern "C" void kenny_copy( int nx , int ny , int nz , int nwfip, double dxyz, double dx )
{
  int nxyz = nx * ny * nz ;      

  cudaMemcpyToSymbol(d_nx, &nx, sizeof(nx));
  cudaMemcpyToSymbol(d_ny, &ny, sizeof(ny));
  cudaMemcpyToSymbol(d_nz, &nz, sizeof(nz));
  cudaMemcpyToSymbol(d_nxyz, &nxyz, sizeof(nxyz));
  cudaMemcpyToSymbol(d_dxyz, &dxyz, sizeof(dxyz));
  cudaMemcpyToSymbol(d_dx, &dx, sizeof(dx));
  cudaMemcpyToSymbol(d_nwfip, &nwfip, sizeof(nwfip));
}
extern "C" void momenta_copy(double kx_min,double ky_min,double kz_min)
{
  cudaMemcpyToSymbol(d_kx_min, &kx_min, sizeof(kx_min));
  cudaMemcpyToSymbol(d_ky_min, &ky_min, sizeof(ky_min));
  cudaMemcpyToSymbol(d_kz_min, &kz_min, sizeof(kz_min));

}

extern "C" void chk_constants_two(double *hbar2m,double *hbarc,double *amu,double *e_cut,double * PI,double *dt_step)
{ 
  if( cudaMemcpyFromSymbol( hbar2m , d_hbar2m , sizeof(double), (size_t)0 , cudaMemcpyDeviceToHost )!= cudaSuccess ) printf("error: failed MemcpyFromSymbol\n") ; 
  if( cudaMemcpyFromSymbol( hbarc , d_hbarc , sizeof(double), (size_t)0 , cudaMemcpyDeviceToHost )!= cudaSuccess ) printf("error: failed MemcpyFromSymbol\n") ; 
  if( cudaMemcpyFromSymbol( amu , d_amu , sizeof(double), (size_t)0 , cudaMemcpyDeviceToHost )!= cudaSuccess ) printf("error: failed MemcpyFromSymbol\n") ; 
  if( cudaMemcpyFromSymbol( e_cut , d_e_cut , sizeof(double), (size_t)0 , cudaMemcpyDeviceToHost )!= cudaSuccess ) printf("error: failed MemcpyFromSymbol\n") ; 
  if( cudaMemcpyFromSymbol( PI , d_PI , sizeof(double), (size_t)0 , cudaMemcpyDeviceToHost )!= cudaSuccess ) printf("error: failed MemcpyFromSymbol\n") ; 
  if( cudaMemcpyFromSymbol( dt_step , d_dtstep , sizeof(double), (size_t)0 , cudaMemcpyDeviceToHost )!= cudaSuccess ) printf("error: failed MemcpyFromSymbol\n") ; 
}
extern "C" void kenny_copy_two( double hbar2m, double hbarc, double amu, double e_cut, double PI, double dt_step , double eps )
 {
  cudaMemcpyToSymbol(d_hbar2m, &hbar2m, (size_t)sizeof(hbar2m));
  cudaMemcpyToSymbol(d_eps, &eps, (size_t)sizeof(eps));
  cudaMemcpyToSymbol(d_hbarc,&hbarc, sizeof(hbarc));
  cudaMemcpyToSymbol(d_amu,&amu, sizeof(amu));
  cudaMemcpyToSymbol(d_e_cut,&e_cut, sizeof(e_cut));
  cudaMemcpyToSymbol(d_PI,&PI, sizeof(PI));  
  cudaMemcpyToSymbol(d_dtstep,&dt_step, sizeof(dt_step));
 }

extern "C" void copy_gg(double gg){
  cufftDoubleReal tmp=(cufftDoubleReal) gg;
  cudaMemcpyToSymbol(d_gg, &tmp, sizeof(tmp));
  return;
}

extern "C" void copy_nuc_params_to_gpu(Couplings * cc_edf)
{
  cufftDoubleReal tmp ; 
  int itmp;

  itmp = cc_edf->iexcoul ;
  cudaMemcpyToSymbol(d_c_iexcoul, &itmp, sizeof(itmp));

  itmp = cc_edf->Skyrme ;
  cudaMemcpyToSymbol(d_c_Skyrme, &itmp, sizeof(itmp));

  tmp = cc_edf->gamma ;
  cudaMemcpyToSymbol(d_gamma, &tmp, sizeof(tmp));

  
  tmp = cc_edf->gamma ;
  cudaMemcpyToSymbol(d_gamma, &tmp, sizeof(tmp));
  tmp = cc_edf->gg;
  cudaMemcpyToSymbol(d_gg, &tmp, sizeof(tmp));
  tmp = cc_edf->rhoc;
  cudaMemcpyToSymbol(d_rhoc, &tmp, sizeof(tmp));

  /* For skyrme forces we need both gg_p and gg_n. */
  tmp = cc_edf->gg_p;
  cudaMemcpyToSymbol(d_gg_p, &tmp, sizeof(tmp));
  tmp = cc_edf->gg_n;
  cudaMemcpyToSymbol(d_gg_n, &tmp, sizeof(tmp));


  /*   Isoscalar and isovector coupling constants */
  tmp = cc_edf->c_rho_a0 ;
  cudaMemcpyToSymbol(d_c_rho_a0, &tmp, sizeof(tmp));
  tmp = cc_edf->c_rho_b0 ;
  cudaMemcpyToSymbol(d_c_rho_b0, &tmp, sizeof(tmp));
  tmp = cc_edf->c_rho_c0 ;
  cudaMemcpyToSymbol(d_c_rho_c0, &tmp, sizeof(tmp));

  tmp = cc_edf->c_rho_a1 ;
  cudaMemcpyToSymbol(d_c_rho_a1, &tmp, sizeof(tmp));
  tmp = cc_edf->c_rho_b1 ;
  cudaMemcpyToSymbol(d_c_rho_b1, &tmp, sizeof(tmp));
  tmp = cc_edf->c_rho_c1 ;
  cudaMemcpyToSymbol(d_c_rho_c1, &tmp, sizeof(tmp));

  tmp = cc_edf->c_rho_a2 ;
  cudaMemcpyToSymbol(d_c_rho_a2, &tmp, sizeof(tmp));
  tmp = cc_edf->c_rho_b2 ;
  cudaMemcpyToSymbol(d_c_rho_b2, &tmp, sizeof(tmp));
  tmp = cc_edf->c_rho_c2 ;
  cudaMemcpyToSymbol(d_c_rho_c2, &tmp, sizeof(tmp));
  
  tmp = cc_edf->c_isospin;
  cudaMemcpyToSymbol(d_c_isospin, &tmp, sizeof(tmp));
  
  tmp = cc_edf->c_rho_0 ;
  cudaMemcpyToSymbol(d_c_rho_0, &tmp, sizeof(tmp));
  tmp = cc_edf->c_rho_1;
  cudaMemcpyToSymbol(d_c_rho_1, &tmp, sizeof(tmp));
  tmp = cc_edf->c_gamma_0;
  cudaMemcpyToSymbol(d_c_gamma_0, &tmp, sizeof(tmp));
  tmp = cc_edf->c_gamma_1;
  cudaMemcpyToSymbol(d_c_gamma_1, &tmp, sizeof(tmp));

  
  tmp = cc_edf->c_laprho_0;
  cudaMemcpyToSymbol(d_c_laprho_0, &tmp, sizeof(tmp));
  tmp = cc_edf->c_laprho_1;
  cudaMemcpyToSymbol(d_c_laprho_1, &tmp, sizeof(tmp));
  tmp = cc_edf->c_tau_0  ;
  cudaMemcpyToSymbol(d_c_tau_0, &tmp, sizeof(tmp));
  tmp = cc_edf->c_tau_1 ;
  cudaMemcpyToSymbol(d_c_tau_1, &tmp, sizeof(tmp));
  tmp = cc_edf->c_divjj_0;
  cudaMemcpyToSymbol(d_c_divjj_0, &tmp, sizeof(tmp));
  tmp = cc_edf->c_divjj_1;
  cudaMemcpyToSymbol(d_c_divjj_1, &tmp, sizeof(tmp));
  tmp = cc_edf->c_j_0   ;
  cudaMemcpyToSymbol(d_c_j_0, &tmp, sizeof(tmp));
  tmp = cc_edf->c_j_1  ;
  cudaMemcpyToSymbol(d_c_j_1, &tmp, sizeof(tmp));
  tmp = cc_edf->c_divj_0;
  cudaMemcpyToSymbol(d_c_divj_0, &tmp, sizeof(tmp));
  tmp = cc_edf->c_divj_1;
  cudaMemcpyToSymbol(d_c_divj_1, &tmp, sizeof(tmp));
  
  /*   Proton and neutron coupling constants */
  tmp = cc_edf->c_rho_p;
  cudaMemcpyToSymbol(d_c_rho_p, &tmp, sizeof(tmp));
  tmp = cc_edf->c_rho_n ;
  cudaMemcpyToSymbol(d_c_rho_n, &tmp, sizeof(tmp));
  tmp = cc_edf->c_laprho_p;
  cudaMemcpyToSymbol(d_c_laprho_p, &tmp, sizeof(tmp));
  tmp = cc_edf->c_laprho_n;
  cudaMemcpyToSymbol(d_c_laprho_n, &tmp, sizeof(tmp));
  tmp = cc_edf->c_tau_p  ;
  cudaMemcpyToSymbol(d_c_tau_p, &tmp, sizeof(tmp));
  tmp = cc_edf->c_tau_n ;
  cudaMemcpyToSymbol(d_c_tau_n, &tmp, sizeof(tmp));
  tmp = cc_edf->c_divjj_p;
  cudaMemcpyToSymbol(d_c_divjj_p, &tmp, sizeof(tmp));
  tmp = cc_edf->c_divjj_n;
  cudaMemcpyToSymbol(d_c_divjj_n, &tmp, sizeof(tmp));
  tmp = cc_edf->c_j_p   ;
  cudaMemcpyToSymbol(d_c_j_p, &tmp, sizeof(tmp));
  tmp = cc_edf->c_j_n  ;
  cudaMemcpyToSymbol(d_c_j_n, &tmp, sizeof(tmp));
  tmp = cc_edf->c_divj_p ;
  cudaMemcpyToSymbol(d_c_divj_p, &tmp, sizeof(tmp));
  tmp = cc_edf->c_divj_n;
  cudaMemcpyToSymbol(d_c_divj_n, &tmp, sizeof(tmp));

}



void do_get_gradient_laplacean ( cufftDoubleComplex * grad , cufftDoubleComplex * lapl , cufftDoubleComplex * wavf , cufftDoubleComplex * wfft , cufftDoubleComplex * fft3 , cufftDoubleReal * kxyz , cufftHandle plan_b , cufftHandle plan_r , int batch  , int nwfip , int nxyz )
{ 
  int i ;

  int tpb = 512;
  int chunks = (4*nwfip) / batch ;
  int ispill  = (4*nwfip) % batch ; //4*nwfip - chunks * batch 

  // note that here blocks := threads_per_block
  int blocks = (int) ceil( ((float)batch*(float)nxyz)/(float)tpb);

  for ( i = 0 ; i < chunks ; i++)
    {
      copy_carray<<<blocks, tpb >>> (wavf+i*batch*nxyz, fft3 , batch * nxyz ) ; // in, out, nele 
      if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error in do_get_tau(): cufftExecZ2Z FORWARD failed\n") ;
      copy_carray<<<blocks , tpb >>> (fft3 , wfft , batch * nxyz ) ; 
      
      /* x component */
      grad_stepone <<<blocks , tpb >>> ( wfft , fft3 , kxyz , batch ) ; // STRICT
      if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
      grad_steptwo <<<blocks , tpb >>> (grad + i*batch*3*nxyz, fft3 , batch ) ;
      
      /* y component */
      grad_stepone <<<blocks , tpb >>> ( wfft , fft3 , kxyz + nxyz , batch ) ;
      if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
      grad_steptwo <<<blocks , tpb >>> (grad + nxyz + i*batch*3*nxyz, fft3 , batch ) ;
      
      /* z component */
      grad_stepone <<<blocks , tpb >>> ( wfft , fft3 , kxyz + 2 * nxyz , batch );
      if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
      grad_steptwo <<<blocks , tpb >>> (grad + 2 * nxyz + i*batch*3*nxyz , fft3 , batch ) ;


      /* laplacean */
      
      lapl_step_one <<<blocks , tpb >>> ( wfft, fft3 , kxyz + 3*nxyz , batch ) ; // STRICT
      
      if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
      lapl_step_two <<<blocks , tpb >>> (lapl + i*batch*nxyz , fft3 , batch ) ;
      
    }
  if ( ispill != 0 ) 
    {
      blocks = (int) ceil( ((float)ispill*(float)nxyz)/(float)tpb);
      copy_carray<<<blocks , tpb >>> (wavf+chunks*batch*nxyz, fft3 , ispill * nxyz ) ;
      if( cufftExecZ2Z( plan_r , fft3 , fft3 , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z FORWARD failed\n") ;
      copy_carray<<<blocks , tpb >>> ( fft3 , wfft , ispill * nxyz ) ;
      
      /* x component */
      grad_stepone <<<blocks , tpb >>> ( wfft , fft3 , kxyz , ispill ) ;
      if ( cufftExecZ2Z( plan_r , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
      grad_steptwo <<<blocks , tpb >>> (grad + chunks*batch*3*nxyz, fft3 , ispill ) ;
      
      /* y component */
      grad_stepone <<<blocks , tpb >>> ( wfft , fft3 , kxyz + nxyz , ispill ) ;
      if ( cufftExecZ2Z( plan_r , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
      grad_steptwo <<<blocks , tpb >>> (grad + nxyz + chunks*batch*3*nxyz, fft3 , ispill ) ;
      
      /* z component */
      grad_stepone <<<blocks , tpb >>> ( wfft , fft3 , kxyz + 2 * nxyz , ispill ) ;
      if ( cufftExecZ2Z( plan_r , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
      grad_steptwo <<<blocks , tpb >>> (grad + 2 * nxyz + chunks*batch*3*nxyz, fft3 , ispill ) ;

      /* laplacean */
      
      lapl_step_one <<<blocks , tpb >>> ( wfft, fft3 , kxyz + 3*nxyz , ispill ) ; // STRICT
      
      if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
      lapl_step_two <<<blocks , tpb >>> (lapl + chunks*batch*nxyz , fft3 , ispill ) ;
      
    } /* end loop over remainder of wavefunctions */
}

void do_complete_divjj(cufftDoubleReal * density,cufftDoubleReal * avf,cufftDoubleReal ct,cufftDoubleComplex *fft3,cufftHandle plan_b,cufftDoubleReal *kxyz,int nxyz)
{
  int threads_per_block=512;
  int blocks=ceil((double)(nxyz)/(double)threads_per_block);

  copy_AxS_to_fft3<<<blocks,threads_per_block>>>(density+2*nxyz,density+3*nxyz,density+4*nxyz,avf,fft3);
  copy_AxS_to_fft3<<<blocks,threads_per_block>>>(density+16*nxyz,density+17*nxyz,density+18*nxyz,avf,fft3+3*nxyz);
  if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z FORWARD failed\n") ;
  take_div_step1<<<blocks,threads_per_block>>>(fft3,kxyz);
  if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
  update_divjj<<<blocks,threads_per_block>>>(density+5*nxyz,ct,fft3);
  update_divjj<<<blocks,threads_per_block>>>(density+19*nxyz,ct,fft3+nxyz);

}

void do_get_spin_dens( cufftDoubleComplex * wavf , cufftDoubleReal * spindens , int blocks , int threads_per_block , int nxyz )
{
  get_spin_dens<<<blocks, threads_per_block>>>( wavf , spindens , spindens + nxyz , spindens + 2 * nxyz ) ; //wavf , sd_x , sd_y , sd_z
}

void do_get_divjj_curlj( cufftDoubleComplex * grad , cufftDoubleReal * divjj , cufftDoubleReal * j , int blocks , int threads_per_block , int nxyz ) 
{
  get_divjj_curlj<<<blocks, threads_per_block>>>( grad , divjj , j , j+nxyz , j+2*nxyz) ; 
}

// calculate the divergence (div) of current (j)
void do_get_divj( cufftDoubleComplex * lapl , cufftDoubleComplex * wavf, cufftDoubleReal * divj , int blocks , int threads_per_block , int nxyz ) 
{
  get_divj<<<blocks, threads_per_block>>>( lapl, wavf, divj ) ; 
}
/////////////////////////////

void do_get_nu_rho_tau_j( cufftDoubleReal * rho , cufftDoubleComplex * nu , cufftDoubleReal * tau , cufftDoubleReal * j , cufftDoubleComplex * grad , cufftDoubleComplex * lapl, cufftDoubleComplex * wavf , cufftDoubleReal * ksq ,int blocks , int threads_per_block , int nxyz )
{
  get_nu_rho_tau_j <<<blocks, threads_per_block>>> ( rho , nu , tau, j , j + nxyz , j + 2*nxyz , grad , lapl, wavf, ksq ) ;
}


////////////////////////////////////////

void do_make_dens_zero( cufftDoubleReal * darray , int len , int blocks , int threads_per_block ) 
{
  make_dzero_ln<<<blocks,threads_per_block>>>( darray , len );
}

void do_get_divjj_curlj_fft(cufftDoubleReal * cj_p , cufftDoubleReal * cj_n , cufftDoubleReal * kxyz , cufftDoubleReal * j_p , cufftDoubleReal * j_n , cufftDoubleComplex * fft3 , cufftHandle plan_b , int nxyz )
{

  int threads_per_block=512;
  int blocks;
  blocks = nxyz / threads_per_block ;
  if ( nxyz % threads_per_block != 0 ) blocks++ ;

  copy_current_to_fft3<<<blocks,threads_per_block>>>(j_p,j_n,fft3);
  if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z FORWARD failed\n") ;
  pre_curl<<<blocks,threads_per_block>>>(fft3,kxyz);
  if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
  make_curl<<<blocks,threads_per_block>>>(fft3,cj_p,cj_n);
}

void do_dipole(cufftDoubleReal *vext,cufftDoubleReal * coord,cufftDoubleReal str,int nxyz,int idir)
{
  int threads_per_block=512;
  int blocks;
  blocks = nxyz / threads_per_block ;
  if ( nxyz % threads_per_block != 0 ) blocks++ ;

  dipole_field<<<blocks,threads_per_block>>>(vext,coord,str,idir,nxyz);

}


void do_quadrupole_field(cufftDoubleReal * vext,double *xyz,int nxyz)
{

  int threads_per_block=512;
  int blocks;
  blocks = nxyz / threads_per_block ;
  if ( nxyz % threads_per_block != 0 ) blocks++ ;

  quadrupole_field<<<blocks,threads_per_block>>>(vext,xyz,nxyz);
}

void do_zero_ext_field(cufftDoubleReal * vext,int nxyz)
{

  int threads_per_block=512;
  int blocks;
  blocks = nxyz / threads_per_block ;
  if ( nxyz % threads_per_block != 0 ) blocks++ ;
  make_dzero_ln<<<blocks,threads_per_block>>>( vext , nxyz );

}



void do_get_nu_rho_tau_j( cufftDoubleReal * rho , cufftDoubleComplex * nu , cufftDoubleReal * tau , cufftDoubleReal * j , cufftDoubleComplex * grad , cufftDoubleComplex * lapl , cufftDoubleComplex * wavf , cufftDoubleReal * kxyz, int blocks , int threads_per_block , int nxyz );
void do_get_spin_dens( cufftDoubleComplex * wavf , cufftDoubleReal * spindens , int blocks , int threads_per_block, int nxyz );
void do_get_divjj_curlj( cufftDoubleComplex * grad , cufftDoubleReal * divjj , cufftDoubleReal * j , int blocks , int threads_per_block , int nxyz ) ;

extern "C" void external_potential_gpu(cufftDoubleReal *vext,cufftDoubleReal * xyz,int isospin,int iext,int nxyz,cufftDoubleReal str_p,cufftDoubleReal str_n)
{

  cufftDoubleReal str=str_p ;

  if( isospin == -1 )
    str=str_n;

  switch(iext)
    {

    case 1:  // quadrupole field
      do_quadrupole_field(vext,xyz,nxyz);
      break;
    case 21: // dipole excitation
      do_dipole(vext,xyz,str,nxyz,1) ; break;
    case 22:
      do_dipole(vext,xyz+nxyz,str,nxyz,2) ; break;
    case 23:
      do_dipole(vext,xyz+2*nxyz,str,nxyz,3) ; break;

    default:
      do_zero_ext_field(vext,nxyz); break;

    }

}

void call_cmmoment(int blocks, int threads, int size, cufftDoubleReal * d_idata1 , cufftDoubleReal * d_idata2 , cufftDoubleReal * d_odata) ;
int opt_threads(int new_blocks,int threads, int current_size);

void call_qpe_kernel_(int blocks, int threads, int size, cufftDoubleReal * d_idata , cufftDoubleReal * d_odata);

extern "C" void compute_friction_gpu(const int nxyz, cufftDoubleComplex *df_dt, cufftDoubleComplex *lapl, cufftDoubleReal * mass_eff, const MPI_Comm comm, double b_strength, double * alpha_fric){

  // up to you to zero ALL the density array prior to use  
  int blocks , threads_per_block ;
  threads_per_block = 512 ;
  blocks = (int)ceil((float)(4*nxyz)/(float)threads_per_block);

  double alpha_fric_;

  
  cufftDoubleReal * th;
  int icudaError=cudaMalloc( ( void ** ) &th , 4*nxyz*sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- compute_friction_gpu()\n");
  
    
  int blocks2=(int)ceil((float)(4*nxyz)/(float)threads_per_block);
  int blocks2_=(int)ceil((float)(blocks2)/(float)threads_per_block);
  
  unsigned int new_blocks, current_size ;
  unsigned int lthreads ;
  int offset ;

  cufftDoubleReal * buf ;
  icudaError=cudaMalloc( ( void ** ) &buf , blocks2*sizeof(cufftDoubleReal) );
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- compute_friction_gpu()\n");  
  
  // \sum_k Im (v_k^*(r) T H v_k(r) /hbar)
  sum_TH_local<<<blocks,threads_per_block>>>(lapl, df_dt, th, nxyz);

  dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
  lthreads = threads_per_block / 2 ; // threads should be power of 2
  if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
  call_qpe_kernel_( blocks2 , lthreads , 4*nxyz , th , buf ) ;
  current_size = blocks2;
  while ( current_size > 1 )
    {
      new_blocks = (int)ceil((float)current_size/(float)threads_per_block);
      lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
      if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
      call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
      current_size = new_blocks ;
    }
  cudaMemcpy( &alpha_fric_, buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToHost) ;

  MPI_Allreduce(&alpha_fric_, alpha_fric, 1, MPI_DOUBLE, MPI_SUM, comm);

  alpha_fric[0] *= (-1.)*b_strength; // alpha ~ -int_d^3r Im ( \sum_k v_k^*(r) T H v_k(r) /hbar )
 
  cudaFree(buf);
  cudaFree(th);
  
}


void do_get_laplacean( cufftDoubleReal * kxyz, cufftDoubleComplex * fft3, cufftHandle plan_b, int nxyz ) ;


extern "C" void compute_densities_gpu( const int nxyz , const int nwfip , cufftDoubleComplex * wavf , cufftDoubleComplex * grad, cufftDoubleComplex * lapl, cufftDoubleComplex * wfft , cufftDoubleComplex * fft3 , cufftDoubleReal * kxyz , cufftHandle plan_b , cufftHandle plan_r , cufftHandle plan_d, int batch , cufftDoubleReal * density, cufftDoubleReal * avf, const MPI_Comm comm , cufftDoubleReal * copy_to , cufftDoubleReal * copy_from , int isoff , cufftDoubleReal Ctime, double * divj )
{ // up to you to zero ALL the density array prior to use 
  int blocks , threads_per_block ;
  threads_per_block = 512 ; 
  blocks = nxyz / threads_per_block ;
  if ( nxyz % threads_per_block != 0 ) blocks++ ;
 
  int len = 28 * nxyz ;
  do_make_dens_zero( density , len , (int)ceil((float)len/(float)threads_per_block),threads_per_block) ; // not done yet!

  do_get_gradient_laplacean( grad, lapl, wavf , wfft , fft3 , kxyz , plan_b, plan_r, batch , nwfip , nxyz ) ;

  do_get_nu_rho_tau_j(density + isoff ,(cufftDoubleComplex *)(density + isoff + 12 * nxyz) , density + isoff  + nxyz , density  + isoff + 6 * nxyz , grad , lapl, wavf , kxyz + 3*nxyz ,blocks , threads_per_block , nxyz );

  do_pre_laplacean_rho<<<blocks,threads_per_block>>>(fft3, density + isoff);  // Computing the laplacean for rho.
  do_get_laplacean( kxyz, fft3, plan_d, nxyz ) ;  // Computing the laplacean for rho.
  get_tau_laplrho<<<blocks,threads_per_block>>>(fft3, density + isoff + nxyz ) ; //Adding the laplacean term of rho to tau.

  do_get_spin_dens( wavf , density  + isoff + 2 * nxyz , blocks , threads_per_block , nxyz ) ;
  do_get_divjj_curlj( grad , density  + isoff + 5 * nxyz , density  + isoff + 9 * nxyz , blocks , threads_per_block , nxyz ) ;

#ifdef BENCHMARK
  double itgtod ;
  clock_t it_clk ;
  b_it(&itgtod,&it_clk);
#endif
  
  cudaMemcpy( (void *)(copy_to + isoff) , (const void *)(density + isoff) , (len/2)*sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost ) ;

  MPI_Barrier( comm );

  MPI_Allreduce( copy_to , copy_from , len , MPI_DOUBLE , MPI_SUM , comm ) ;
  cudaMemcpy( density , copy_from , len*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice) ;
  
#ifdef BENCHMARK
  com_tim+=e_it(0,&itgtod,&it_clk);
#endif
}




void do_get_coulomb( cufftDoubleReal * coulomb, cufftDoubleReal * density, cufftDoubleReal * fc, int * map_ls, int * map_sl, cufftHandle plan_c , cufftDoubleComplex * fft3_c, int nxyz , int nxyz3 )
{
  int blocks , threads_per_block ;
  
  threads_per_block = 512 ; 
  
  // zero big lattice
  blocks = (int)ceil((float)(nxyz3)/(float)threads_per_block);
  make_czero<<<blocks,threads_per_block>>>( fft3_c , nxyz3 );

  //remap density : small to big 
  blocks = (int)ceil((float)nxyz/(float)threads_per_block);
  map_density_to_latt3<<<blocks,threads_per_block>>> (density, fft3_c , map_sl ) ; 

  if ( cufftExecZ2Z( plan_c , fft3_c , fft3_c , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error in do_get_coulomb(): cufftExecZ2Z FORWARD failed\n") ;
  blocks = (int)ceil((float)(nxyz3)/(float)threads_per_block);
  fc_step<<<blocks,threads_per_block >>> (fft3_c , fc , nxyz3 ) ; 

  if ( cufftExecZ2Z( plan_c , fft3_c , fft3_c , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error in do_get_coulomboostomb(): cufftExecZ2Z INVERSE failed\n") ;
  map_latt_to_coul<<<blocks,threads_per_block>>> (coulomb, fft3_c , map_ls , nxyz3) ; 
}

void do_get_laplacean( cufftDoubleReal * kxyz, cufftDoubleComplex * fft3, cufftHandle plan_b, int nxyz ) 
{
  int blocks , threads_per_block ;
  
  threads_per_block = 512 ; 
  blocks = (int)ceil((float)nxyz/(float)threads_per_block);
  if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error in do_get_laplacean: cufftExecZ2Z FORWARD failed\n") ;
  lapl_stepone<<<blocks,threads_per_block>>>(fft3,kxyz);
  if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error in do_get_laplacean: cufftExecZ2Z INVERSE failed\n") ;
}

void compute_cmshift( cufftDoubleReal * rho_p , cufftDoubleReal * rho_n , cufftDoubleReal * j_p , cufftDoubleReal * j_n , cufftDoubleReal * vcm , int nxyz ){

  cufftDoubleReal * buf;
  int threads_per_block = 512 ;
  int i;

  int blocks2=(int)ceil((float)(nxyz)/(float)threads_per_block);
  int icudaError = cudaMalloc( ( void ** ) &buf, blocks2*sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- compute_cmshift()\n");
  int blocks2_=(int)ceil((float)(blocks2)/(float)threads_per_block);

  for(i=0;i<3;++i){
    dzero_buf<<<blocks2_,threads_per_block>>>(buf,blocks2) ;
    int lthreads = threads_per_block / 2 ; 
    if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
    call_cmmoment(blocks2, lthreads, nxyz,j_p+i*nxyz,j_n+i*nxyz,buf);
    int current_size = blocks2;
    while ( current_size > 1 )
      {
	int new_blocks = (int)ceil((float)current_size/threads_per_block);
	lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size                                                                                               
	call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
	current_size = new_blocks ;
      }
    cudaMemcpy( vcm + i , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
  }

  dzero_buf<<<blocks2_,threads_per_block>>>(buf,blocks2) ;
  int lthreads = threads_per_block / 2 ; 
  if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
  call_cmmoment(blocks2, lthreads, nxyz,rho_p,rho_n,buf);
  int current_size = blocks2;
  while ( current_size > 1 )
    {
      int new_blocks = (int)ceil((float)current_size/threads_per_block);
      lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
      if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size                                                                                               
      call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
      current_size = new_blocks ;
    }
  calculate_CMvel<<<1,3>>>(vcm,buf);  // contains an extra hbar, multiply directy the derivatives in the evolution
  cudaFree(buf);
}

extern "C" void define_VF_gpu(cufftDoubleReal *avf,cufftDoubleReal *xyz, int nxyz,int isospin)
{
  int threads_per_block=512;
  int blocks = (int)ceil((float)nxyz/(float)threads_per_block);
  
  do_get_VF_zero<<<blocks,threads_per_block>>>(avf,xyz,nxyz);

}

extern "C" void get_u_re_gpu( cufftDoubleReal * density, cufftDoubleReal * potentials, int nxyz, int isospin, cufftDoubleReal * kxyz, cufftDoubleComplex * fft3 , cufftHandle plan_b, cufftDoubleComplex * fft3_c,cufftHandle plan_c, cufftDoubleReal * fc, int * map_ls, int * map_sl , cufftDoubleReal * avf , cufftDoubleReal ct , cufftDoubleReal ct_constr , int nxyz3, MPI_Comm comm ) 
{ 
  int threads_per_block = 512 ; 
  int blocks = (int)ceil((float)nxyz/(float)threads_per_block);
  int isoff = 7 * nxyz * (1-isospin) ;
  int ierr;

  do_pre_laplacean_eff_mass<<<blocks,threads_per_block>>>( fft3 , density , density + isoff + 6 * nxyz , potentials ) ; // computes the 
  do_get_laplacean( kxyz, fft3, plan_b, nxyz ) ;

  
 if ( isospin == 1 ) do_get_coulomb(potentials,density,fc,map_ls,map_sl,plan_c,fft3_c,nxyz,nxyz3);
 
 get_ure_lapl_mass<<<blocks,threads_per_block>>>(fft3,potentials,density,isospin);

  addExtField<<<blocks,threads_per_block>>>(potentials,ct,ct_constr,density+isoff,fft3);
}

extern "C" void update_potentials_gpu( cufftDoubleReal * potentials , int nxyz , cufftDoubleReal * density , cufftDoubleReal * fc, int * map_ls, int * map_sl, cufftHandle plan_c, cufftDoubleComplex * fft3_c , cufftHandle plan_b , cufftDoubleComplex * fft3 , cufftDoubleReal * kxyz , int isospin , cufftDoubleReal * avf , cufftDoubleReal ct, cufftDoubleReal ct_constr,MPI_Comm comm,cufftDoubleComplex *wavf,int batch,cufftDoubleReal *cc_pair_qf,cufftDoubleReal *copy_from,cufftDoubleReal *copy_to,int nwfip,int gr_ip,int nxyz3,int icub)
{ 
  int threads_per_block = 512 ;
  int blocks = (int)ceil((float)nxyz/(float)threads_per_block);
  int isoff = 7 * nxyz * (1-isospin) ; 

  make_dzero_ln<<<blocks,threads_per_block>>>(cc_pair_qf,nxyz);

  // first, compute Delta
  make_delta<<<blocks,threads_per_block>>>( density,density+14*nxyz,potentials , (cufftDoubleComplex * ) ( density + 12 * nxyz + isoff ), (cufftDoubleComplex *) (potentials + 12 * nxyz ) , cc_pair_qf , ct ,icub) ;


  get_u_re_gpu(density,potentials, nxyz, isospin, kxyz, fft3 , plan_b, fft3_c, plan_c, fc, map_ls, map_sl , avf , ct , ct_constr ,nxyz3, comm);

  //prepare the fourier transform space for two grad(rho), combinations of j, combinations of S
  make_fourierSpace_potentials<<<blocks,threads_per_block>>>(density,fft3);
  if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error in update_potentials_gpu(): cufftExecZ2Z FORWARD failed\n") ;
  pre_operators<<<blocks,threads_per_block>>>(fft3,kxyz);
  if ( cufftExecZ2Z( plan_b , fft3 , fft3 , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error in update_potentials_gpu(): cufftExecZ2Z INVERSE failed\n") ;

  //compute W (spin-orbit)
  compute_w<<<blocks,threads_per_block>>>(fft3,potentials+3*nxyz);

  //compute U1 (spin-dependent potential)
  compute_u1<<<blocks,threads_per_block>>>(density,potentials+6*nxyz);

  //compute U_grad
  compute_ugrad<<<blocks,threads_per_block>>>(fft3+4*nxyz,potentials+9*nxyz,density);

  compute_cmshift(density,density+14*nxyz,density+6*nxyz,density+20*nxyz,potentials+16*nxyz,nxyz);
}

extern "C" int do_copy_carray( cufftDoubleComplex * zin, cufftDoubleComplex * zout , int nelm )
{
  int blocks, tpb;

  tpb = 512;
  blocks = (nelm / tpb); if ((nelm %tpb) !=0) blocks++;
  copy_carray<<<blocks,tpb>>>( zin , zout , nelm ) ;
  return (EXIT_SUCCESS) ;
}

extern "C" void do_add_to_ugrad( cufftDoubleReal * ugrad, cufftDoubleReal * avf, int nxyz, double factor)
{
  int blocks, tpb;

  tpb = 512;

  blocks = (int)ceil((float)nxyz*3/(float)tpb);

  add_to_ugrad<<<blocks,tpb>>>(ugrad,avf,factor);  

}



extern "C" void do_add_to_delta( cufftDoubleComplex * delta, cufftDoubleReal * chi, int nxyz, double factor)
{
  int blocks, tpb;

  tpb = 512;

  blocks = (int)ceil((float)nxyz/(float)tpb);

  add_to_delta<<<blocks,tpb>>>(delta,chi,factor);
  

}


extern "C" void do_add_to_u( cufftDoubleReal * u, cufftDoubleReal * chi, cufftDoubleReal * vfric, int nxyz, double factor)
{
  int blocks, tpb;

  tpb = 512;

  blocks = (int)ceil((float)nxyz/(float)tpb);

  add_to_u<<<blocks,tpb>>>(u,chi,vfric,factor);
  

}


extern "C" void do_add_to_masseff( cufftDoubleReal * masseff, double alpha, int nxyz, double factor)
{
  int blocks, tpb;

  tpb = 512;

  blocks = (int)ceil((float)nxyz/(float)tpb);

  add_to_masseff<<<blocks,tpb>>>(masseff,alpha*factor);
  

}

// START OF SEPARATE FILE 


// EXTERN ROUTINES

void df_over_dt_u( cufftDoubleComplex * df_dt , cufftDoubleComplex * u , cufftDoubleComplex * grad , cufftDoubleComplex * lapl , cufftDoubleReal * potentials , int nxyz , cufftDoubleComplex * fft3_ev , cufftHandle plan_ev , cufftDoubleReal * kxyz, cufftDoubleReal *xyz, cufftDoubleReal *rcm)
{
  int threads_per_block=512;
  int blocks=(int)ceil((float)(2.*nxyz)/(float)threads_per_block);

  compute_du_dt_step1<<<blocks,threads_per_block>>>(df_dt,u,grad,lapl,potentials);
  compute_du_dt_step11<<<blocks,threads_per_block>>>(fft3_ev,u,potentials);
  if ( cufftExecZ2Z( plan_ev , fft3_ev , fft3_ev , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error in df_over_dt_u(): cufftExecZ2Z FORWARD failed\n") ;
  compute_du_dt_step12<<<blocks,threads_per_block>>>(fft3_ev,kxyz);
  if ( cufftExecZ2Z( plan_ev , fft3_ev , fft3_ev , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
  compute_du_dt_step13<<<blocks,threads_per_block>>>( df_dt , fft3_ev );


  shiftCM<<<blocks,threads_per_block>>>(df_dt,u,grad,potentials+16*nxyz);
  
  // reduce the rotation of the system, author: Shi Jin, 08/01/18
#ifdef ROTATION
  fix_rotation<<<blocks,threads_per_block>>>(df_dt,u,grad,potentials+16*nxyz+3,xyz,rcm );
#endif
  /////
} 

void df_over_dt_v( cufftDoubleComplex * df_dt , cufftDoubleComplex * v , cufftDoubleComplex * grad , cufftDoubleComplex * lapl , cufftDoubleReal * potentials , int nxyz , cufftDoubleComplex * fft3_ev , cufftHandle plan_ev , cufftDoubleReal * kxyz, cufftDoubleReal *xyz, cufftDoubleReal *rcm )
{
  int threads_per_block=512;
  int blocks= (int) ceil((float)(2*nxyz)/(float)threads_per_block);

  compute_dv_dt_step1<<<blocks,threads_per_block>>>(df_dt,v,grad,lapl,potentials);
  compute_dv_dt_step11<<<blocks,threads_per_block>>>(fft3_ev,v,potentials);
  if ( cufftExecZ2Z( plan_ev , fft3_ev , fft3_ev , CUFFT_FORWARD ) != CUFFT_SUCCESS ) printf("error in df_over_dt_u(): cufftExecZ2Z FORWARD failed\n") ;
  compute_du_dt_step12<<<blocks,threads_per_block>>>(fft3_ev,kxyz); // same routine like for u
  if ( cufftExecZ2Z( plan_ev , fft3_ev , fft3_ev , CUFFT_INVERSE ) != CUFFT_SUCCESS ) printf("error: cufftExecZ2Z INVERSE failed\n") ;
  compute_du_dt_step13<<<blocks,threads_per_block>>>( df_dt , fft3_ev );

 shiftCM<<<blocks,threads_per_block>>>(df_dt,v,grad,potentials+16*nxyz);  
#ifdef ROTATION
  fix_rotation<<<blocks,threads_per_block>>>(df_dt,v,grad,potentials+16*nxyz+3,xyz,rcm);
#endif
  //////////////
} 

void df_over_dt_pairing( cufftDoubleComplex * df_dt , cufftDoubleComplex * wavf , cufftDoubleComplex * delta , int nxyz )
{

  int threads_per_block=512;
  int blocks=(int) ceil((float)(2*nxyz)/(float)threads_per_block);

  compute_du_dt_step2<<<blocks,threads_per_block>>>(df_dt,wavf,delta);
  compute_dv_dt_step2<<<blocks,threads_per_block>>>(df_dt,wavf,delta);

} 

void do_remove_qpe(cufftDoubleReal * qpe,cufftDoubleReal * norm, cufftDoubleComplex * df_dt, cufftDoubleComplex * wavf ,int dim)
{

  int threads_per_block=512;
  int blocks= (int) ceil((float)dim/(float)threads_per_block);

  remove_qpe<<<blocks,threads_per_block>>>(qpe,norm,df_dt,wavf,dim);

}


extern "C" void do_boost_wf(cufftDoubleComplex * boost_field, cufftDoubleComplex * wavf ,int dim)
{

  int threads_per_block=512;
  int blocks= (int) ceil((float)dim/(float)threads_per_block);

  boost_wf<<<blocks,threads_per_block>>>(boost_field, wavf, dim); 

}

extern "C" void do_boost_wf2(cufftDoubleComplex * boost_field, cufftDoubleComplex * wavf ,int dim)
{

  int threads_per_block=512;
  int blocks= (int) ceil((float)dim/(float)threads_per_block);

  boost_wf2<<<blocks,threads_per_block>>>(boost_field, wavf, dim); 

}

void do_adams_bashforth_cy(const int nn_time,const int n0_time,const int n1_time,const int m0_time,const int m1_time,const int m2_time,const int dim,cufftDoubleComplex * wavf_td,cufftDoubleComplex * wavf,cufftDoubleComplex * wavf_p,cufftDoubleComplex * wavf_c,cufftDoubleComplex * df_dt,int offset)
{
  int threads_per_block=512;
  int blocks = (int) ceil((float)dim/(float)threads_per_block);

  get_cy<<<blocks,threads_per_block>>>(nn_time,n0_time,n1_time,m0_time,m1_time,m2_time,wavf_td,wavf,wavf_p,wavf_c,df_dt,dim,offset);

}

extern "C" void do_copy_cp_gpu( const int nn_time, const int n0_time, cufftDoubleComplex * wavf, cufftDoubleComplex *wavf_c, cufftDoubleComplex * wavf_p, int nxyz, int nwfip)
{
  int tpb = 512;
  int dim = 4*nxyz*nwfip;
  int blocks = dim / tpb; if( (dim%tpb) != 0) blocks++;

  copy_carray<<<blocks, tpb>>>(wavf+nn_time*dim, wavf_c+n0_time*dim, dim); cudaDeviceSynchronize();

  copy_carray<<<blocks, tpb>>>(wavf+nn_time*dim, wavf_c+nn_time*dim, dim); cudaDeviceSynchronize();

  copy_carray<<<blocks, tpb>>>(wavf+nn_time*dim, wavf_p+n0_time*dim, dim); cudaDeviceSynchronize();

  copy_carray<<<blocks, tpb>>>(wavf+nn_time*dim, wavf_p+nn_time*dim, dim); cudaDeviceSynchronize();
  
}



extern "C" void adams_bashforth_pm_gpu( const int nn_time , const int n0_time , const int n1_time , const int m0_time , const int m1_time , const int m2_time , const int m3_time , const int nxyz , const int nwfip , cufftDoubleComplex * wavf , cufftDoubleComplex * wavf_td , cufftDoubleComplex * wavf_p , cufftDoubleComplex * wavf_c , cufftDoubleComplex * wavf_m )
{ 
  int threads_per_block = 512 ;
  int dim = 4 * nxyz * nwfip ;
  int blocks = (int) ceil((float)( dim ) / (float)threads_per_block) ;

  get_pred_mod<<<blocks,threads_per_block>>>(nn_time,n0_time,n1_time,m0_time,m1_time,m2_time,m3_time,wavf,wavf_td,wavf_p,wavf_c,wavf_m);

}

void call_norm(int blocks, int threads, int size, cufftDoubleComplex * d_idata , cufftDoubleReal * d_odata);
void call_qpe_kernel(int blocks, int threads, int size, cufftDoubleReal * obuf , cufftDoubleComplex * wavf, cufftDoubleComplex * dt);
extern "C" void adams_bashforth_cy_gpu( int nn_time , int n0_time , int n1_time , int m0_time , int m1_time , int m2_time, int nxyz , int nwfip , cufftDoubleComplex * wavf , cufftDoubleComplex * wavf_td , cufftDoubleComplex * wavf_p , cufftDoubleComplex * wavf_c , cufftDoubleComplex * wavf_m , cufftDoubleComplex * grad, cufftDoubleComplex * lapl , cufftHandle plan_b , cufftHandle plan_r , cufftDoubleComplex * df_dt , cufftDoubleReal * potentials, int batch,cufftDoubleReal * kxyz,cufftHandle plan_ev,cufftDoubleComplex * fft3_ev, cufftDoubleReal *xyz, cufftDoubleReal *rcm )
{  int i;
  int chunks=(4*nwfip)/batch; 
  int ispill=(4*nwfip)%batch;
  int threads_per_block=512;
  int blocks=(int) ceil((float)(batch*nxyz)/(float)threads_per_block);
  int iwf;
  int k = batch/4; 
  int blocks2=(int)ceil((float)(4*nxyz)/(float)threads_per_block);
  
  cufftDoubleReal * qpe , * norm , * buf ;
  unsigned int new_blocks, current_size ;
  unsigned int lthreads ;
  int offset ; 
  int icudaError=cudaMalloc( ( void ** ) &buf, blocks2*sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- adams_bashforth_cy()\n");
  icudaError = cudaMalloc( ( void ** ) &qpe, k  * sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- adams_bashforth_cy()\n");
  icudaError=cudaMalloc( ( void ** ) &norm, k  * sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- adams_bashforth_cy()\n");
  int blocks2_=(int)ceil((float)(blocks2)/(float)threads_per_block);
  for (i=0;i<chunks;i++)
    {
      for ( iwf=0 ; iwf< k ; iwf++ ) 
	{

	  df_over_dt_u(df_dt+4*iwf*nxyz       ,wavf_m+i*batch*nxyz+iwf*4*nxyz       ,grad+i*batch*3*nxyz+iwf*12*nxyz       ,lapl+i*batch*nxyz+iwf*4*nxyz       ,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
	  df_over_dt_v(df_dt+4*iwf*nxyz+2*nxyz,wavf_m+i*batch*nxyz+iwf*4*nxyz+2*nxyz,grad+i*batch*3*nxyz+iwf*12*nxyz+6*nxyz,lapl+i*batch*nxyz+iwf*4*nxyz+2*nxyz,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
	  df_over_dt_pairing(df_dt+4*iwf*nxyz,wavf_m+i*batch*nxyz+iwf*4*nxyz,(cufftDoubleComplex *) (potentials+12*nxyz) ,nxyz) ;

	  dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	  lthreads = threads_per_block / 2 ; // threads should be power of 2
	  if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	  call_qpe_kernel(blocks2, lthreads, 4*nxyz , buf , wavf_m+i*batch*nxyz+iwf*4*nxyz, df_dt+4*iwf*nxyz );
	  current_size = blocks2;
	  while ( current_size > 1 )
	    {
	      new_blocks = (int)ceil((float)current_size/(float)threads_per_block);
	      lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	      if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
	      call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
	      current_size = new_blocks ;
	    }    
	  cudaMemcpy( qpe + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;

	  //norm
	  dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	  lthreads = threads_per_block / 2 ; 
	  if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	  call_norm(blocks2, lthreads, 4*nxyz, wavf_m+i*batch*nxyz+iwf*4*nxyz,buf);
	  current_size = blocks2;
	  while ( current_size > 1 )
	    {
	      new_blocks = (int)ceil((float)current_size/threads_per_block);
	      lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	      if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
	      call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
	      current_size = new_blocks ;
	    } 
	  cudaMemcpy( norm + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
	}
      do_remove_qpe(qpe,norm,df_dt,wavf_m+i*batch*nxyz,batch*nxyz);
      do_adams_bashforth_cy(nn_time,n0_time,n1_time,m0_time,m1_time,m2_time,batch*nxyz,wavf_td,wavf,wavf_p,wavf_c,df_dt,i*batch*nxyz);
    }

  if (ispill!=0)
    {
      
      blocks=(int) ceil((float)(ispill*nxyz)/(float)threads_per_block);
      for ( iwf=0 ; iwf < ispill/4 ;++iwf )
	{
	  offset=(chunks*batch+4*iwf)*nxyz;
	  df_over_dt_u(df_dt+iwf*4*nxyz       ,wavf_m+offset       ,grad+offset*3       ,lapl+offset       ,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
	  df_over_dt_v(df_dt+iwf*4*nxyz+2*nxyz,wavf_m+offset+2*nxyz,grad+3*offset+6*nxyz,lapl+offset+2*nxyz,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
	  df_over_dt_pairing(df_dt+iwf*4*nxyz,wavf_m+offset,(cufftDoubleComplex *) (potentials+12*nxyz) ,nxyz) ;

	  lthreads = threads_per_block / 2 ; // threads should be power of 2
	  if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	  dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	  //	  call_qpe_kernel(blocks2, lthreads, 4*nxyz, buf , wavf_m+offset+iwf*4*nxyz, df_dt+4*iwf*nxyz );
	  call_qpe_kernel(blocks2, lthreads, 4*nxyz, buf , wavf_m+offset, df_dt+4*iwf*nxyz );
	  current_size = blocks2;
	  while ( current_size > 1 )
	    {
	      new_blocks = (int)ceil((float)current_size/(float)threads_per_block);
	      lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	      if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
	      call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
	      current_size = new_blocks ;
	    }    
	  cudaMemcpy( qpe + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
	  
	  // do norm
	  dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	  lthreads = threads_per_block / 2 ; 
	  if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	  call_norm(blocks2, lthreads, 4*nxyz, wavf_m+offset,buf);
	  current_size = blocks2;
	  while ( current_size > 1 )
	    {
	      new_blocks = (int)ceil((float)current_size/threads_per_block);
	      lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	      if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
	      call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
	      current_size = new_blocks ;
	    } 
	  cudaMemcpy( norm + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
	}
    
      do_remove_qpe(qpe,norm,df_dt,wavf_m+chunks*batch*nxyz,ispill*nxyz);
      do_adams_bashforth_cy(nn_time,n0_time,n1_time,m0_time,m1_time,m2_time,ispill*nxyz,wavf_td,wavf,wavf_p,wavf_c,df_dt,chunks*batch*nxyz);
    }
  cudaFree(buf);
  cudaFree(qpe);
  cudaFree(norm);
}


extern "C" void test_norm_gpu( int nxyz , int nwfip , cufftDoubleComplex * wavf , int batch, double * cpu_norm )
{
  int threads_per_block=512;
  int blocks=(int) ceil((float)(batch*nxyz)/(float)threads_per_block);
  int iwf;
  int k = batch/4; //compare k to nwfip 
  int blocks2=(int)ceil((double)(2*nxyz)/(double)threads_per_block);
  cufftDoubleReal * qpe , * norm , * buf ;
  unsigned int new_blocks, current_size ;
  unsigned int lthreads ;
  int icudaError=cudaMalloc( ( void ** ) &buf , blocks2  * sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- test_norm_gpu()\n");
  cudaMalloc( ( void ** ) &qpe , nwfip  * sizeof(cufftDoubleReal) ); 
  cudaMalloc( ( void ** ) &norm , nwfip  * sizeof(cufftDoubleReal) ); 

  double * gpu_norm;
  cudaHostAlloc( (void **)&gpu_norm , nwfip * sizeof(double) , cudaHostAllocDefault );
  for ( iwf=0 ; iwf<k ; ++iwf ) 
    {
      dzero_buf<<<1,blocks2>>>( buf , blocks2 ) ;
      lthreads = threads_per_block / 2 ; 
      if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
      current_size = blocks2;
      while ( current_size > 1 )
	{
	  new_blocks = (int)ceil((float)current_size/(float)threads_per_block);
	  lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	  if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
	  call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
	  current_size = new_blocks ;
	}    
      cudaMemcpy( norm + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
    }
  cudaMemcpy(gpu_norm , norm , k*sizeof(cufftDoubleReal),cudaMemcpyDeviceToHost) ; 
  for (iwf=0;iwf<k;iwf++)
    printf("gpu_norm[%d]=%f\tcpu_norm[%d]=%f\n", iwf, (double)(gpu_norm[iwf]*1.25*1.25*1.25),iwf, cpu_norm[iwf]);
  cudaFree(buf); cudaFree(qpe); cudaFree(norm); cudaFree(gpu_norm);
  printf("\t...exit tst_norm\n");
}

void call_norm(int blocks, int threads, int size, cufftDoubleComplex * d_idata , cufftDoubleReal * d_odata);
extern "C" void l_gpu_norm( int n , int nwfip , cufftDoubleComplex * wavf , cufftDoubleReal * gpu_norm , cufftDoubleReal * part_sum )
{ int iwf;
  int threads_per_block = 512;
  int blocks=(int) ceil((float)n/threads_per_block);
  unsigned int new_blocks, current_size, lthreads ;

  for ( iwf=0 ; iwf<nwfip ; iwf++ ) 
    {
      dzero_buf<<<blocks,threads_per_block>>>( part_sum , n );
      lthreads = threads_per_block / 2 ; 
      if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
      call_norm(blocks, lthreads, n, (wavf+iwf*n),part_sum);
      current_size = blocks;
      while ( current_size > 1 )
	{
	  new_blocks = (int)ceil((float)current_size/threads_per_block);
	  lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	  if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
	  call_qpe_kernel_( new_blocks , lthreads , current_size , part_sum , part_sum ) ;
	  current_size = new_blocks ;
	} 
      cudaMemcpy( gpu_norm + iwf , part_sum , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToHost ) ;
    }

  printf("SIZEOF(cufftDoubleReal)%d\n",(int)sizeof(cufftDoubleReal));
  printf("\t...exit tst_norm\n");
}


/* calculate (H - mu)psi 
 * utility function in tstep_gpu
 */



void get_hpsi_gpu(const int nxyz , const int nwfip , cufftDoubleComplex * wavf1 , cufftDoubleComplex * wavf2, cufftDoubleComplex * grad, cufftDoubleComplex * lapl , cufftHandle plan_b , cufftHandle plan_r , cufftDoubleReal * potentials, const int batch, cufftDoubleReal * kxyz, cufftHandle plan_ev, cufftDoubleComplex * fft3_ev, cufftDoubleReal * qpe, cufftDoubleReal * norm, int m, cufftDoubleReal *xyz, cufftDoubleReal *rcm )
{
  /* calculate hpsi once
   * wavf1: the function to be left multiplied by hamiltonian h
   * wavf2: the result after the multiplication
   * qpe : quasi-particle energy (only calculated in 1st order expansion)
   * norm: norm of wfs (only calculated in 1st order expansion)
   */
  int i;
  int chunks=(4*nwfip)/batch; // warning - chunks and batch explicitly related !!!
  int ispill=(4*nwfip)%batch;
  int threads_per_block=512;

  int blocks=(int) ceil((float)(batch*nxyz)/(float)threads_per_block);

  int iwf;
  int k=batch/4;

  cufftDoubleReal * buf ;
  int blocks2=(int)ceil((float)(4*nxyz)/(float)threads_per_block);

  unsigned int new_blocks, current_size ;
  unsigned int lthreads ;
  int offset ;

  int icudaError=cudaMalloc( ( void ** ) &buf , blocks2*sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- get_hpsi_gpu()\n");
  
  
  int blocks2_=(int)ceil((float)(blocks2)/(float)threads_per_block);

  for (i=0;i<chunks;i++)
    {

      for ( iwf=0 ; iwf<k ; iwf++ ) // warning - now batch is k * 4 as well 
	{
	  df_over_dt_u(wavf2+i*batch*nxyz+iwf*4*nxyz       ,wavf1+i*batch*nxyz+iwf*4*nxyz       ,grad+i*batch*3*nxyz+iwf*12*nxyz       ,lapl+i*batch*nxyz+iwf*4*nxyz       ,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
	  df_over_dt_v(wavf2+i*batch*nxyz+iwf*4*nxyz+2*nxyz,wavf1+i*batch*nxyz+iwf*4*nxyz+2*nxyz,grad+i*batch*3*nxyz+iwf*12*nxyz+6*nxyz,lapl+i*batch*nxyz+iwf*4*nxyz+2*nxyz,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
	  df_over_dt_pairing(wavf2+i*batch*nxyz+iwf*4*nxyz ,wavf1+i*batch*nxyz+iwf*4*nxyz, (cufftDoubleComplex *)(potentials+12*nxyz),nxyz) ;
	

	  if(m==1)  // for 1st order expansion, calculate the qpe and norm of wfs and save, for usage in  higher order expansion
	    {
	      dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	      lthreads = threads_per_block / 2 ; // threads should be power of 2
	      if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	      call_qpe_kernel(blocks2, lthreads, 4*nxyz, buf , wavf1+i*batch*nxyz+iwf*4*nxyz, wavf2+i*batch*nxyz+iwf*4*nxyz);
	      current_size = blocks2;
	      while ( current_size > 1 )
		{
		  new_blocks = (int)ceil((float)current_size/(float)threads_per_block);
		  lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
		  if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
		  call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
		  current_size = new_blocks ;
		}
	      // be careful of the offset 
	      cudaMemcpy( qpe + i*k + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
	      
	      //norm
	      dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	      lthreads = threads_per_block / 2 ; 
	      if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	      call_norm(blocks2, lthreads, 4*nxyz, wavf1+i*batch*nxyz+iwf*4*nxyz,buf);
	      current_size = blocks2;
	      while ( current_size > 1 )
		{
		  new_blocks = (int)ceil((float)current_size/threads_per_block);
		  lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
		  if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
		  call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
		  current_size = new_blocks ;
		} 
	      cudaMemcpy( norm + i*k + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
	    }
	  

	}
      do_remove_qpe(qpe+i*k,norm+i*k,wavf2+i*batch*nxyz,wavf1+i*batch*nxyz,batch*nxyz);


	  
    }

  if (ispill!=0)
    {
      blocks=(int)ceil((float)(ispill*nxyz)/(float)threads_per_block);

      for(iwf=0;iwf<ispill/4;++iwf)
	{
	  offset=(chunks*batch+4*iwf)*nxyz;
	  df_over_dt_u(wavf2+offset       ,wavf1+offset       ,grad+offset*3       ,lapl+offset       ,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
	  df_over_dt_v(wavf2+offset+2*nxyz,wavf1+offset+2*nxyz,grad+3*offset+6*nxyz,lapl+offset+2*nxyz,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
	  df_over_dt_pairing(wavf2+offset,wavf1+offset,(cufftDoubleComplex *) (potentials+12*nxyz) ,nxyz) ;

	  if(m==1)
	    {
	      lthreads = threads_per_block / 2 ; // threads should be power of 2
	      if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	      dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	      //	  call_qpe_kernel(blocks2, lthreads, 4*nxyz, buf , wavf_m+offset+iwf*4*nxyz, df_dt+4*iwf*nxyz );
	      call_qpe_kernel(blocks2, lthreads, 4*nxyz, buf , wavf1+offset,wavf2+offset );
	      current_size = blocks2;
	      while ( current_size > 1 )
		{
		  new_blocks = (int)ceil((float)current_size/(float)threads_per_block);
		  lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
		  if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
		  call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
		  current_size = new_blocks ;
		}    
	      cudaMemcpy( qpe + chunks*k + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
	      
	      // do norm
	      dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	      lthreads = threads_per_block / 2 ; 
	      if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	      call_norm(blocks2, lthreads, 4*nxyz, wavf1+offset, buf);
	      //call_norm(blocks2, lthreads, 4*nxyz, wavf_m+offset,buf);
	      current_size = blocks2;
	      while ( current_size > 1 )
		{
		  new_blocks = (int)ceil((float)current_size/threads_per_block);
		  lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
		  if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
		  call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
		  current_size = new_blocks ;
		} 
	      cudaMemcpy( norm + chunks*k + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
	    }
	  
	}
      do_remove_qpe(qpe+chunks*k,norm+chunks*k,wavf2+chunks*batch*nxyz,wavf1+chunks*batch*nxyz,ispill*nxyz);
    }

  cudaFree(buf);
}



extern "C" void adams_bashforth_dfdt_gpu(const int nxyz , const int nwfip , cufftDoubleComplex * wavf , cufftDoubleComplex * wavf_td , cufftDoubleComplex * grad, cufftDoubleComplex * lapl , cufftHandle plan_b , cufftHandle plan_r , cufftDoubleReal * potentials, const int batch, cufftDoubleReal * kxyz, int i_saveocc,double *e_qp, double * e_sp, double *n_qp,cufftHandle plan_ev,cufftDoubleComplex * fft3_ev, cufftDoubleReal *xyz,cufftDoubleReal *rcm,double * lz_qp)
{

  int i;
  int chunks=(4*nwfip)/batch; // warning - chunks and batch explicitly related !!!
  int ispill=(4*nwfip)%batch;
  int threads_per_block=512;

  int blocks=(int) ceil((float)(batch*nxyz)/(float)threads_per_block);

  int iwf;
  int k=batch/4;

  cufftDoubleReal * buf ;
  int blocks2=(int)ceil((float)(4*nxyz)/(float)threads_per_block);
  cufftDoubleReal * qpe , * norm;
  
  unsigned int new_blocks, current_size ;
  unsigned int lthreads ;
  int offset ;

#ifdef SAVEOCC

  cufftDoubleReal * d_occ;
  cudaMalloc( (void ** ) &d_occ , k*sizeof(cufftDoubleReal) );
  cufftDoubleReal * d_lzav;
  cudaMalloc( (void ** ) &d_lzav , k*sizeof(cufftDoubleReal) );
  cufftDoubleComplex * lzwf;
  cudaMalloc( (void ** ) &lzwf , 2*nxyz*sizeof(cufftDoubleComplex) );
  
#endif

  int icudaError=cudaMalloc( ( void ** ) &buf , blocks2*sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- adams_bashforth_dfdt_gpu()\n");
  icudaError=cudaMalloc( ( void ** ) &qpe , k  * sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- adams_bashforth_dfdt_gpu()\n");
 
  
  icudaError=cudaMalloc( ( void ** ) &norm , k  * sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- adams_bashforth_dfdt_gpu()\n");
  int blocks2_=(int)ceil((float)(blocks2)/(float)threads_per_block);

  for (i=0;i<chunks;i++)
    {
      //calculate first df_over_dt

      for ( iwf=0 ; iwf<k ; iwf++ ) // warning - now batch is k * 4 as well 
	{
	  df_over_dt_u(wavf_td+i*batch*nxyz+iwf*4*nxyz       ,wavf+i*batch*nxyz+iwf*4*nxyz       ,grad+i*batch*3*nxyz+iwf*12*nxyz       ,lapl+i*batch*nxyz+iwf*4*nxyz       ,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
	  df_over_dt_v(wavf_td+i*batch*nxyz+iwf*4*nxyz+2*nxyz,wavf+i*batch*nxyz+iwf*4*nxyz+2*nxyz,grad+i*batch*3*nxyz+iwf*12*nxyz+6*nxyz,lapl+i*batch*nxyz+iwf*4*nxyz+2*nxyz,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
  
	  df_over_dt_pairing(wavf_td+i*batch*nxyz+iwf*4*nxyz ,wavf+i*batch*nxyz+iwf*4*nxyz, (cufftDoubleComplex *)(potentials+12*nxyz),nxyz) ;

	  dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	  lthreads = threads_per_block / 2 ; // threads should be power of 2
	  if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	  call_qpe_kernel(blocks2, lthreads, 4*nxyz, buf , wavf+i*batch*nxyz+iwf*4*nxyz, wavf_td+i*batch*nxyz+iwf*4*nxyz);
	  current_size = blocks2;
	  while ( current_size > 1 )
	    {
	      new_blocks = (int)ceil((float)current_size/(float)threads_per_block);
	      lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	      if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
	      call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
	      current_size = new_blocks ;
	    }    
	  cudaMemcpy( qpe + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;

	  //norm
	  dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	  lthreads = threads_per_block / 2 ; 
	  if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	  call_norm(blocks2, lthreads, 4*nxyz, wavf+i*batch*nxyz+iwf*4*nxyz,buf);
	  current_size = blocks2;
	  while ( current_size > 1 )
	    {
	      new_blocks = (int)ceil((float)current_size/threads_per_block);
	      lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	      if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
	      call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
	      current_size = new_blocks ;
	    } 
	  cudaMemcpy( norm + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
#ifdef SAVEOCC
	  if(i_saveocc==0)
	    {
	      dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	      lthreads = threads_per_block / 2 ; 
	      if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	      call_norm(blocks2, lthreads, 2*nxyz, wavf+i*batch*nxyz+iwf*4*nxyz+2*nxyz,buf);
	      current_size = blocks2;
	      while ( current_size > 1 )
		{
		  new_blocks = (int)ceil((float)current_size/threads_per_block);
		  lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
		  if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size                                                                                               
		  call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
		  current_size = new_blocks ;
		}
	      cudaMemcpy( d_occ + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;

	      dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	      lthreads = threads_per_block / 2 ; 
	      if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	      
	      compute_lz_wf<<<blocks,threads_per_block>>>(lzwf,grad+i*batch*3*nxyz+iwf*12*nxyz+6*nxyz,xyz,rcm);
	      call_qpe_kernel(blocks2, lthreads, 2*nxyz, buf , wavf+i*batch*nxyz+iwf*4*nxyz+2*nxyz,lzwf);

	      current_size = blocks2;
	      while ( current_size > 1 )
		{
		  new_blocks = (int)ceil((float)current_size/threads_per_block);
		  lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
		  if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size                                                                                               
		  call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
		  current_size = new_blocks ;
		}
	      cudaMemcpy( d_lzav+iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;

	    }
#endif
	}
      do_remove_qpe(qpe,norm,wavf_td+i*batch*nxyz,wavf+i*batch*nxyz,batch*nxyz);

      
#ifdef SAVEOCC 
      if( i_saveocc == 0 ){
	// normalize the quasi-particle energy (qpe)
	calculate_qpe<<<1,k>>>(qpe,norm,k);
	cudaMemcpy( e_qp+i*k, qpe , k*sizeof(cufftDoubleReal) , cudaMemcpyDeviceToHost ) ;
	cudaMemcpy( n_qp+i*k, d_occ , k*sizeof(cufftDoubleReal) , cudaMemcpyDeviceToHost ) ;
	cudaMemcpy( lz_qp+i*k, d_lzav , k*sizeof(cufftDoubleReal) , cudaMemcpyDeviceToHost ) ;
      }
#endif
    }

  if (ispill!=0)
    {
      blocks=(int)ceil((float)(ispill*nxyz)/(float)threads_per_block);

      for(iwf=0;iwf<ispill/4;++iwf)
	{
	  offset=(chunks*batch+4*iwf)*nxyz;
	  df_over_dt_u(wavf_td+offset       ,wavf+offset,       grad+offset*3       ,lapl+offset       ,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;
	  df_over_dt_v(wavf_td+offset+2*nxyz,wavf+offset+2*nxyz,grad+3*offset+6*nxyz,lapl+offset+2*nxyz,potentials,nxyz,fft3_ev,plan_ev,kxyz,xyz,rcm) ;  
	  
	  df_over_dt_pairing(wavf_td+offset,wavf+offset,(cufftDoubleComplex *) (potentials+12*nxyz) ,nxyz) ;

	  lthreads = threads_per_block / 2 ; // threads should be power of 2
	  if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	  dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	  call_qpe_kernel(blocks2, lthreads, 4*nxyz, buf , wavf+offset,wavf_td+offset );
	  current_size = blocks2;
	  while ( current_size > 1 )
	    {
	      new_blocks = (int)ceil((float)current_size/(float)threads_per_block);
	      lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	      if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
	      call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
	      current_size = new_blocks ;
	    }    
	  cudaMemcpy( qpe + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
	  
	  // do norm
	  dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	  lthreads = threads_per_block / 2 ; 
	  if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	  call_norm(blocks2, lthreads, 4*nxyz, wavf+offset, buf);
	  current_size = blocks2;
	  while ( current_size > 1 )
	    {
	      new_blocks = (int)ceil((float)current_size/threads_per_block);
	      lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
	      if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
	      call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
	      current_size = new_blocks ;
	    } 
	  cudaMemcpy( norm + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
#ifdef SAVEOCC
	  if(i_saveocc==0)
	    {
	      dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	      lthreads = threads_per_block / 2 ; 
	      if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	      call_norm(blocks2, lthreads, 2*nxyz, wavf+offset+2*nxyz,buf);
	      current_size = blocks2;
	      while ( current_size > 1 )
		{
		  new_blocks = (int)ceil((float)current_size/threads_per_block);
		  lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
		  if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
		  call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
		  current_size = new_blocks ;
		}
	      cudaMemcpy( d_occ + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
	    }

	      dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	      lthreads = threads_per_block / 2 ; 
	      if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	      
	      compute_lz_wf<<<blocks,threads_per_block>>>(lzwf,grad+i*batch*3*nxyz+iwf*12*nxyz+6*nxyz,xyz,rcm);
	      call_qpe_kernel(blocks2, lthreads, 2*nxyz, buf , wavf+i*batch*nxyz+iwf*4*nxyz+2*nxyz,lzwf);

	      current_size = blocks2;
	      while ( current_size > 1 )
		{
		  new_blocks = (int)ceil((float)current_size/threads_per_block);
		  lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
		  if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size                                                                                               
		  call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
		  current_size = new_blocks ;
		}
	      cudaMemcpy( d_lzav + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
#endif
	}
      do_remove_qpe(qpe,norm,wavf_td+chunks*batch*nxyz,wavf+chunks*batch*nxyz,ispill*nxyz);
      
#ifdef SAVEOCC 
      if( i_saveocc == 0 ){


	calculate_qpe<<<1,ispill/4>>>(qpe,norm,ispill/4);
	cudaMemcpy( e_qp+chunks*k, qpe , (ispill/4)*sizeof(cufftDoubleReal) , cudaMemcpyDeviceToHost ) ;
	
	cudaMemcpy( n_qp+chunks*k, d_occ , (ispill/4)*sizeof(cufftDoubleReal) , cudaMemcpyDeviceToHost ) ;
	cudaMemcpy( lz_qp+chunks*k, d_lzav , (ispill/4)*sizeof(cufftDoubleReal) , cudaMemcpyDeviceToHost ) ;
      }
#endif
    }
  cudaFree(norm);
  cudaFree(qpe);
  cudaFree(buf);

#ifdef SAVEOCC
  cudaFree(d_occ);
  cudaFree(d_lzav);
  cudaFree(lzwf);
#endif

}

extern "C" void mix_potentials(cufftDoubleReal * potentials, cufftDoubleReal * potentials_new, int dim, cufftDoubleReal c0)  
{
  int tpb = 512;

  int blocks = (dim/tpb); if( (dim%tpb) != 0) blocks++;

  do_mix_potentials<<<blocks, tpb>>>(potentials, potentials_new, dim, c0);
  
}

extern "C" void check_ortho_gpu( int nwfip, int nxyz, cufftDoubleComplex * wavf, cufftDoubleReal * norms_ortho)
{
  int threads_per_block=512;
  int iwf;
 
  cufftDoubleReal * buf , * norm ;

  int blocks2=(int)ceil((float)(4*nxyz)/(float)threads_per_block);

  unsigned int new_blocks, current_size ;
  unsigned int lthreads ;

  int icudaError=cudaMalloc( ( void ** ) &buf , blocks2*sizeof(cufftDoubleReal) );
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- check_ortho_gpu()\n");

  icudaError=cudaMalloc( ( void ** ) &norm , nwfip*(nwfip-1)*sizeof(cufftDoubleReal) );
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- check_ortho_gpu()\n");

  int blocks2_=(int)ceil((float)(blocks2)/(float)threads_per_block);
  int i, j;

  iwf = 0;
  
  for(i=0;i<nwfip; i++)
    for(j=0; j<nwfip; j++)
      {

	if(j != i)
	  {
	    //norm
	    dzero_buf<<<blocks2_,threads_per_block>>>( buf , blocks2 ) ;
	    lthreads = threads_per_block / 2 ; 
	    if ( lthreads < 64 ) lthreads = 64 ; //at least 2*warp_size
	    call_qpe_kernel(blocks2, lthreads, 4*nxyz, buf , wavf+i*4*nxyz, wavf+j*4*nxyz);
	    current_size = blocks2;
	    while ( current_size > 1 )
	      {
		new_blocks = (int)ceil((float)current_size/threads_per_block);
		lthreads = opt_threads( new_blocks , threads_per_block , current_size ) / 2 ;
		if ( lthreads < 64 ) lthreads = 64; // at least 2*warp_size
		call_qpe_kernel_( new_blocks , lthreads , current_size , buf , buf ) ;
		current_size = new_blocks ;
	      } 
	    cudaMemcpy( norm + iwf , buf , sizeof(cufftDoubleReal) , cudaMemcpyDeviceToDevice ) ;
	    iwf++;
	  }
      }

  cudaMemcpy( norms_ortho, norm , nwfip*(nwfip-1)*sizeof(cufftDoubleReal) , cudaMemcpyDeviceToHost ) ;

  cudaFree(norm);
  cudaFree(buf);
  
  
}
extern "C" void tstep_gpu(const int nxyz , const int nwfip , const int mxp, const int n0_time, const int nn_time, cufftDoubleComplex * wavf, cufftDoubleComplex * wavf_psi, cufftDoubleComplex * wavf_hpsi, cufftDoubleComplex * wavf_mxp, cufftDoubleReal * coeffs,cufftDoubleComplex * grad, cufftDoubleComplex * lapl, cufftDoubleComplex * wfft, cufftDoubleComplex * fft3 , cufftHandle plan_b , cufftHandle plan_r , cufftDoubleReal * potentials, cufftDoubleReal *avf, const int batch, cufftDoubleReal * kxyz, cufftHandle plan_ev, cufftDoubleComplex * fft3_ev, cufftDoubleReal *xyz, cufftDoubleReal *rcm )
{
  /*
   * calculate the series expansion:
   * \sum_n=0^{mxp} \hat{h}^n/(n! hbar^n) (idt)^n \psi
   * wavf: wavefunction \psi(t)
   * wavf_psi: the function to be performed hamiltonian
   * wavf_hpsi: the result of hpsi product
   * wavf_mxp: store m terms in the expansion of \psi(t+dt) (or \tilde{\psi} )
   * mxp: the maximum order of the series expansion
   */
  int tpb = 512;

  int blocks = (4*nxyz*nwfip) / tpb; if ( (4*nwfip*nxyz)%tpb != 0 ) blocks++;

  int blocks2 = (nxyz) / tpb; if ( (nxyz)%tpb != 0 ) blocks2++;
    // buffers store the qpe and norm of wavfs
  cufftDoubleReal * qpe , * norm, ct = (cufftDoubleReal) 1.0;
  
  int icudaError=cudaMalloc( ( void ** ) &qpe , nwfip  * sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- tstep_gpu()\n");
  icudaError=cudaMalloc( ( void ** ) &norm , nwfip  * sizeof(cufftDoubleReal) ); 
  if(icudaError != cudaSuccess ) printf( "error in cudaMalloc  -- tstep_gpu()\n");
  
  copy_carray<<<blocks, tpb>>>(wavf+n0_time*4*nxyz*nwfip, wavf_psi, 4*nxyz*nwfip); //cudaDeviceSynchronize();

  // calculate gradient and laplacean of wfs
  do_get_gradient_laplacean( grad, lapl, wavf_psi , wfft , fft3 , kxyz , plan_b, plan_r, batch ,  nwfip , nxyz ) ;

  int m;

  // copy the wfs (0th order) to wavf_mxp firstly; 
  
  copy_carray<<<blocks,tpb>>>(wavf_psi, wavf_mxp, 4*nxyz*nwfip); //cudaDeviceSynchronize();

  // perform hpsi for (mxp-1) times
  for(m=1; m<mxp; m++)
    {
      // perform hpsi operation 
      get_hpsi_gpu(nxyz , nwfip , wavf_psi , wavf_hpsi, grad, lapl , plan_b , plan_r , potentials, batch, kxyz, plan_ev, fft3_ev, qpe, norm, m, xyz, rcm);
      
      // copy back to wavf_psi for next hpsi operation 
      copy_carray<<<blocks, tpb>>>(wavf_hpsi, wavf_psi, 4*nxyz*nwfip); //cudaDeviceSynchronize();

      // recalculate the gradient and laplacean of wfs
      do_get_gradient_laplacean( grad, lapl, wavf_psi , wfft , fft3 , kxyz , plan_b, plan_r, batch , nwfip , nxyz ) ;
     
      // multiplies the coefficients
      scale_expansion<<<blocks,tpb>>>(&coeffs[m], wavf_hpsi, 4*nxyz*nwfip); //cudaDeviceSynchronize();
      // copy to wavf_mxp;
      copy_carray<<<blocks,tpb>>>(wavf_hpsi, wavf_mxp+m*4*nwfip*nxyz, 4*nxyz*nwfip); //cudaDeviceSynchronize();
    }

  // Add the expansion terms together
  combine_expansion<<<blocks,tpb>>>(mxp, wavf_mxp, wavf+nn_time*4*nxyz*nwfip, 4*nxyz*nwfip); //cudaDeviceSynchronize();

  
  cudaFree(qpe);
  cudaFree(norm);
}


void call_qpe_kernel_(int blocks, int threads, int size, cufftDoubleReal * d_idata , cufftDoubleReal * d_odata)
{// call_qpe_kernel_( new_blocks , lthreads , current_size , part_sum , part_sum ) ;
  int smemSize = threads * sizeof(cufftDoubleReal);
  switch ( threads )
    {
    case 1024:
      __reduce_kernel__<1024><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    case 512:
      __reduce_kernel__< 512><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    case 256:
      __reduce_kernel__< 256><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    case 128:
      __reduce_kernel__< 128><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    case 64:
      __reduce_kernel__<  64><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    }   
}

void call_norm(int blocks, int threads, int size, cufftDoubleComplex * d_idata , cufftDoubleReal * d_odata)
{// call_norm(blocks, lthreads, n, wavf+iwf*n,part_sum);
  int smemSize = threads * sizeof(cufftDoubleReal);
  switch ( threads )
    {
    case 1024:
      do_norm<1024><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    case 512:
      do_norm< 512><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    case 256:
      do_norm< 256><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    case 128:
      do_norm< 128><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    case 64:
      do_norm<  64><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    }   
}

void call_cmmoment(int blocks, int threads, int size, cufftDoubleReal * d_idata1 , cufftDoubleReal * d_idata2 , cufftDoubleReal * d_odata)
{// call_norm(blocks, lthreads, n, wavf+iwf*n,part_sum);
  int smemSize = threads * sizeof(cufftDoubleReal);
  switch ( threads )
    {
    case 1024:
      do_mom<1024><<< blocks, threads, smemSize >>>(d_idata1, d_idata2 , d_odata, size); break;
    case 512:
      do_mom< 512><<< blocks, threads, smemSize >>>(d_idata1, d_idata2 , d_odata, size); break;
    case 256:
      do_mom< 256><<< blocks, threads, smemSize >>>(d_idata1, d_idata2 , d_odata, size); break;
    case 128:
      do_mom< 128><<< blocks, threads, smemSize >>>(d_idata1, d_idata2 , d_odata, size); break;
    case 64:
      do_mom<  64><<< blocks, threads, smemSize >>>(d_idata1, d_idata2 , d_odata, size); break;
    }   
}

void call_qpe_kernel(int blocks, int threads, int size, cufftDoubleReal * obuf , cufftDoubleComplex * wavf, cufftDoubleComplex * dt )
{
  int smemSize = threads * sizeof(cufftDoubleReal);
  switch ( threads )
    {
    case 1024:
      qpe_energy_nuc<1024><<< blocks, threads, smemSize >>> ( obuf , wavf, dt, size); break;
    case 512:
      qpe_energy_nuc<512><<< blocks, threads, smemSize >>> ( obuf , wavf, dt, size); break;
    case 256:
      qpe_energy_nuc<256><<< blocks, threads, smemSize >>> ( obuf , wavf, dt, size); break;
    case 128:
      qpe_energy_nuc<128><<< blocks, threads, smemSize >>> ( obuf , wavf, dt, size); break;
    case 64:
      qpe_energy_nuc<64><<< blocks, threads, smemSize >>> ( obuf , wavf, dt, size); break;
    }   
}

int opt_threads(int new_blocks,int threads, int current_size)
{
  int new_threads;

  if ( new_blocks == 1 ) 
    {
      new_threads = 2; 
      while ( new_threads < threads ) 
	{ 
	  if ( new_threads >= current_size ) 
	    break;
	  new_threads *= 2 ;
	}
    }
  else 
    new_threads = threads ;

  return new_threads;
}

extern "C" void kenny_copy_( int nx , int ny , int nz , int nwfip )
{
  int nxyz = nx * ny * nz ;      

  cudaMemcpyToSymbol(d_nx, &nx, sizeof(nx));
  cudaMemcpyToSymbol(d_ny, &ny, sizeof(ny));
  cudaMemcpyToSymbol(d_nz, &nz, sizeof(nz));
  cudaMemcpyToSymbol(d_nxyz, &nxyz, sizeof(nxyz));
  cudaMemcpyToSymbol(d_nwfip, &nwfip, sizeof(nwfip));
}


double switchingFunction( const double time ) {

  double f1 ;

  if( time < Tfull ){

    f1 = .5 * ( 1. + tanh ( ALPHA * tan( PI_ * ( 2. * time / Tfull - 1. ) / 2. ) ) ) ;

  }else{

    if( time < Tfull2 )

      f1 = 1;

    else

      if( time < Tstop ){

	f1 = .5 * ( 1. - tanh( ALPHA * tan( PI_ * ( 2. * ( time - Tfull2 ) / ( Tstop - Tfull2 ) - 1. ) / 2. ) ) ) ;

      }else{

        f1 = 0. ;

      }

  }

  return f1 ;


}
