#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <assert.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "vars.h"

double deform( double * rho , int nxyz , Lattice_arrays * latt_coords  , double dxyz );

double center_dist( double * rho , const int n , Lattice_arrays * latt_coords , double * xc , double * yc , double * zc );

void laplacean( double * f , double * lap_f , const int nxyz , FFtransf_vars * fftrans , Lattice_arrays * latt );

void make_coordinates( const int nxyz , const int nx , const int ny , const int nz , const double dx , const double dy , const double dz , Lattice_arrays * lattice_vars );

void match_lattices( Lattice_arrays *latt , Lattice_arrays * latt3 , const int nx , const int ny , const int nz , const int nx3 , const int ny3 , const int nz3 , FFtransf_vars * fftrans , const double Lc );

void coul_pot3( double * vcoul , double * rho , double * work1 , double * work2 , Lattice_arrays * latt_coords , const int nxyz , FFtransf_vars * fftransf_vars , const double dxyz );

int dens_func_params( const int iforce , const int ihfb , const int isospin , Couplings * cc_edf , const int ip, int icub, double alpha_pairing);

void read_input_solver( int * nx , int * ny , int * nz , int * nwf_p , int * nwf_n , double * amu_p , double * amu_n , double * dx , double * dy , double * dz , double * e_cut ){
    
    char * fn ;
    
    FILE * fd ;
    
    fn = malloc( 130 * sizeof( char ) ) ;
    
    sprintf( fn , "info.slda_solver" ) ;
    
    fd = fopen( fn , "rb" ) ;
    
    fread( nwf_p , sizeof( int ) , 1 , fd ) ;
    
    fread( nwf_n , sizeof( int ) , 1 , fd ) ;
    
    fread( amu_p , sizeof( double ) , 1 , fd ) ;
    
    fread( amu_n , sizeof( double ) , 1 , fd ) ;
    
    fread( dx , sizeof( double ) , 1 , fd ) ;
    
    fread( dy , sizeof( double ) , 1 , fd ) ;
    
    fread( dz , sizeof( double ) , 1 , fd ) ;
    
    fread( nx , sizeof( int ) , 1 , fd ) ;
    
    fread( ny , sizeof( int ) , 1 , fd ) ;
    
    fread( nz , sizeof( int ) , 1 , fd ) ;
    
    fread( e_cut , sizeof( double ) , 1 , fd ) ;
    
    printf("nx=%d ny=%d nz=%d\n",*nx,*ny,*nz);
    printf("dx=%f dy=%f dz=%f\n",*dx,*dy,*dz);
    
    fclose( fd ) ;
    
    free( fn ) ;
    
}

void pairingfluct(FILE * fd, double complex * delta, double * rho, int nxyz,double dxyz){
    
    int i;
    double complex delta0=0.+I*0.;
    double delta2=0.,delta0r=0.;
    int ivol=0;

    
    for(i=0;i<nxyz;i++){
        if( rho[i]>=0.02){
            ivol++;
            delta0+=delta[i];
            delta0r+=cabs(delta[i]);
        }
    }
    delta0/=ivol;
    delta0r/=ivol;
    
    for (i=0; i<nxyz; i++) {
        if( rho[i]>=0.02)
            delta2+=pow(cabs((delta[i]-delta0)),2.);
    }
    delta2/=ivol;
    fprintf(fd, " %12.6f %12.6f %12.6f",cabs(delta0),sqrt(delta2),delta0r);
    
}

double coul_frag( double * rho , double * xa , double * ya , double * za , int nxyz , double dxyz,double z0 ){
    
    int i,j;
    double r;
    double sum=0.;
#pragma omp parallel for private(i,j) reduction(+:sum)
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

void makeFragment(double * dens, double *densf,double *theta,int n){
    
    int i;
    
    for(i=0;i<14*n;i++){
        densf[i]=dens[i]*theta[i%n];
    }
    return;
    
}

double system_energy( Couplings * cc_edf , double * dens_p , double * dens_n , const int nxyz , double complex * delta_p , double complex * delta_n , double complex * nu_p, double complex * nu_n , const double hbar2m , const double dxyz , Lattice_arrays * latt , FFtransf_vars * fftransf_vars , const double time , FILE * fd ,double * buff, FILE * fd_kin ){
    
    // buff = double size 5*nxyz
    
  const double mass_p = 938.272013 ;
  const double mass_n = 939.565346 ;
    
    double mass=.5*(mass_p+mass_n);

    double xpow=1./3.;
    double e2 = -197.3269631*pow(3./acos(-1.),xpow) / 137.035999679 ;
    xpow*=4.;
    e2*=(3./2.);
    
    static double egs;
    
    static int compute_gs =0 ;
    
    double e_tot , e_pair_p , e_rho , e_rhotau, e_so , e_laprho , e_kin ;
    
    double e_flow_p , e_flow_n ;

    double e_coll;
    
    int  ixyz , ix , iy , iz ;
    
    double e_pair_n , e_kin_n , e_j , tmp , e_coul , n_part ;
    
    double * rho_0 , * rho_1 , * lap_rho_0 , * lap_rho_1 , * vcoul ;
    
    double xcm , ycm , zcm , xcm_p , ycm_p , zcm_p , xcm_n , ycm_n , zcm_n ;
    
    double num_p , num_n , q30=0., q40=0.;

    double qxx, qyy, qzz, qxy, qyz, qzx;
    
    double beta;
    
    double vx, vy, vz;

    double v2;
    
    coul_pot3( buff+4*nxyz , dens_p , buff , buff+nxyz , latt , nxyz , fftransf_vars , dxyz ) ;
    
    for( ixyz = 0 ; ixyz < nxyz ; ixyz++ ) {
        buff[ ixyz ] = dens_p[ ixyz ] + dens_n[ixyz ] ;
        buff[ ixyz + nxyz ] = dens_n[ ixyz ] - dens_p[ixyz ] ;
    }

    rho_0 = buff;
    rho_1 = buff + nxyz;
    
    center_dist( buff , nxyz , latt , &xcm , &ycm , &zcm ) ;
    
    laplacean( buff , buff+2*nxyz , nxyz , fftransf_vars , latt ) ;
    
    laplacean( buff+nxyz , buff+3*nxyz , nxyz , fftransf_vars , latt ) ;
    
    e_kin    = 0. ;
    e_rho    = 0. ;
    e_rhotau = 0. ;
    e_laprho = 0. ;
    e_so     = 0. ;
    e_j      = 0. ;
    e_pair_p   = 0. ;
    e_pair_n   = 0. ;

    e_coll = 0.;
    e_coul   = 0. ;
    e_flow_p = 0.;
    e_flow_n = 0.;

    q30 = 0.;
    q40 = 0.;
    vx=0.;
    vy=0.;
    vz=0.;
    v2=0.;
    qxx = 0.; qyy = 0.; qzz = 0.;
    qxy = 0.; qyz = 0.; qzx = 0.;

    num_n = 0; num_p = 0.;
    
#pragma omp parallel for reduction(+: qxx,qyy,qzz,qxy,qyz,qzx,q30,q40,e_kin,e_rho,e_rhotau,e_laprho,e_so,e_pair_p,e_pair_n,e_j,e_flow_p,e_flow_n,e_coul,vx,vy,vz,v2)
    for( ixyz = 0 ; ixyz < nxyz ; ixyz++ ) {
        
        double x2=pow(latt->xa[ ixyz ] -xcm,2.);
        double y2=pow(latt->ya[ ixyz ] -ycm,2.);
        double z2=pow(latt->za[ ixyz ] -zcm,2.);
        double r2=x2+y2+z2;

	qxx += buff[ ixyz ] * x2;
	qyy += buff[ ixyz ] * y2;
	qzz += buff[ ixyz ] * z2;
	
	qxy += buff[ ixyz ] * (latt->xa[ ixyz ] -xcm)*(latt->ya[ ixyz ] -ycm);
	qyz += buff[ ixyz ] * (latt->ya[ ixyz ] -ycm)*(latt->za[ ixyz ] -zcm);
	qzx += buff[ ixyz ] * (latt->za[ ixyz ] -zcm)*(latt->xa[ ixyz ] -xcm);
	
        
        q30 += buff[ ixyz ]*(latt->za[ ixyz ] -zcm ) *( 2.*z2-3.*x2-3.*y2);
        q40 += buff[ ixyz ]*(35.*z2*z2-30.*z2*r2+3.*r2*r2);
	

	num_n += dens_n[ixyz] *dxyz;

	num_p += dens_p[ixyz] *dxyz;
	
        e_kin    += dens_p[ixyz+nxyz]+dens_n[ixyz+nxyz] ;
	

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
	
        e_rhotau += ( cc_edf->c_tau_0 * ( dens_p[ixyz+nxyz] + dens_n[ ixyz+nxyz] ) * buff[ixyz] + cc_edf->c_tau_1 * ( dens_n[ixyz+nxyz] - dens_p[ixyz+nxyz] ) * buff[ixyz+nxyz] ) ;
        e_laprho += ( cc_edf->c_laprho_0 * buff[ixyz+2*nxyz] * buff[ ixyz ] + cc_edf->c_laprho_1 * buff[ixyz+3*nxyz] * buff[ixyz+nxyz] ) ;

        e_so     += ( cc_edf->c_divjj_0 * buff[ixyz] * ( dens_n[ixyz+5*nxyz] + dens_p[ixyz+5*nxyz] ) + cc_edf->c_divjj_1 * buff[ixyz+nxyz] * (  dens_n[ixyz+5*nxyz] - dens_p[ixyz+5*nxyz] ) ) ;
        e_pair_p -= creal( delta_p[ixyz] * conj( nu_p[ixyz] ) ) ;
        e_pair_n -= creal( delta_n[ixyz] * conj( nu_n[ixyz] ) ) ;
        e_j +=  ( cc_edf->c_j_0 * ( pow( dens_n[ ixyz+6*nxyz ] + dens_p[ ixyz+6*nxyz ] , 2 ) + pow( dens_n[ ixyz+7*nxyz ] + dens_p[ ixyz+7*nxyz ] , 2 ) + pow( dens_n[ ixyz+8*nxyz ] + dens_p[ ixyz+8*nxyz ] , 2 ) )
                 + cc_edf->c_j_1 * ( pow( dens_n[ ixyz+6*nxyz ] - dens_p[ ixyz+6*nxyz ] , 2 ) + pow( dens_n[ ixyz+7*nxyz ] - dens_p[ ixyz+7*nxyz ] , 2 ) + pow( dens_n[ ixyz+8*nxyz ] - dens_p[ ixyz+8*nxyz ] , 2 ) )
                 + cc_edf->c_divj_0 * ( ( dens_n[ ixyz+2*nxyz ] + dens_p[ ixyz+2*nxyz ] ) * ( dens_n[ ixyz+9*nxyz ] + dens_p[ ixyz+9*nxyz ] ) + ( dens_n[ ixyz+3*nxyz ] + dens_p[ ixyz+3*nxyz ] ) * ( dens_n[ ixyz+10*nxyz ] + dens_p[ ixyz+10*nxyz ] ) + ( dens_n[ ixyz+4*nxyz ] + dens_p[ ixyz+4*nxyz ] ) * ( dens_n[ ixyz+11*nxyz ] + dens_p[ ixyz+11*nxyz ] ) )
                 + cc_edf->c_divj_1 * ( ( dens_n[ ixyz+2*nxyz ] - dens_p[ ixyz+2*nxyz ] ) * ( dens_n[ ixyz+9*nxyz ] - dens_p[ ixyz+9*nxyz ] ) + ( dens_n[ ixyz+3*nxyz ] - dens_p[ ixyz+3*nxyz ] ) * ( dens_n[ ixyz+10*nxyz ] - dens_p[ ixyz+10*nxyz ] ) + ( dens_n[ ixyz+4*nxyz ] - dens_p[ ixyz+4*nxyz ] ) * ( dens_n[ ixyz+11*nxyz ] - dens_p[ ixyz+11*nxyz ] ) ) ) ;
        
        if( dens_p[ixyz]>1.e-7)
            e_flow_p += ( dens_p[ ixyz+6*nxyz ] * dens_p[ ixyz+6*nxyz ] + dens_p[ ixyz+7*nxyz ] * dens_p[ ixyz+7*nxyz ] + dens_p[ ixyz+8*nxyz ] * dens_p[ ixyz+8*nxyz ] )/dens_p[ixyz];
        if( dens_n[ixyz]>1.e-7)
            e_flow_n += (dens_n[ ixyz+6*nxyz ] * dens_n[ ixyz+6*nxyz ] + dens_n[ ixyz+7*nxyz ] * dens_n[ ixyz+7*nxyz ] + dens_n[ ixyz+8*nxyz ] * dens_n[ ixyz+8*nxyz ] )/dens_n[ixyz];
        
        e_coul += dens_p[ ixyz ] * buff[ ixyz+4*nxyz ] ;
	e_coul += (e2*pow(dens_p[ ixyz ],xpow));
        
        vx += dens_p[ixyz+6*nxyz]+dens_n[ixyz+6*nxyz];
        vy += dens_p[ixyz+7*nxyz]+dens_n[ixyz+7*nxyz];
        vz += dens_p[ixyz+8*nxyz]+dens_n[ixyz+8*nxyz];
	v2 += pow(dens_p[ixyz+6*nxyz] + dens_n[ixyz+6*nxyz],2.0) \
	  +pow(dens_p[ixyz+7*nxyz] + dens_n[ixyz+7*nxyz],2.0) \
	  +pow(dens_p[ixyz+8*nxyz] + dens_n[ixyz+8*nxyz],2.0);
        
    }
    
    double hbarc = 197.3269631 ;
    
    beta = deform( buff , nxyz , latt  , dxyz ) ;
    
    e_pair_p *= dxyz ;
    e_pair_n *= dxyz ;
    
    e_kin *= ( hbar2m * dxyz ) ;
    
    center_dist( dens_p , nxyz , latt , &xcm_p , &ycm_p , &zcm_p )*dxyz ;
    center_dist( dens_n , nxyz , latt , &xcm_n , &ycm_n , &zcm_n )*dxyz ;
    
    double mtot=mass*(num_p+num_n);
    
    vx *= hbarc*dxyz/mtot;
    vy *= hbarc*dxyz/mtot;
    vz *= hbarc*dxyz/mtot;

    e_coll = v2*hbarc*hbarc*dxyz/2./mtot;
    
    e_rho*= dxyz ;
        
    e_rhotau *= dxyz ;
    
    e_so *= dxyz ;
    
    e_laprho  *= dxyz ;
    
    e_j *= dxyz ;
    
    e_flow_p *= ( hbar2m * dxyz ) ;
    
    e_flow_n *= ( hbar2m * dxyz );
    
    e_coul *= ( .5 * dxyz ) ;
    
    e_tot = e_kin + e_pair_p + e_pair_n + e_rho + e_rhotau + e_laprho + e_so + e_coul + e_j ;
    
    if( compute_gs == 0 ){
        
        compute_gs = 1;
        
        egs = e_tot;
        
    }
    
    printf("e_pair_p=%12.6f e_pair_n=%12.6f\n",e_pair_p,e_pair_n);
    printf("e_kin=%12.6f e_rho=%14.6f e_rhotau=%12.6f e_laprho=%12.6f e_so=%12.6f e_coul=%12.6f e_j=%12.6f\n" , e_kin , e_rho , e_rhotau , e_laprho , e_so , e_coul , e_j ) ;
    
    printf("field energy: %12.6f \n" , e_rho + e_rhotau + e_laprho + e_j ) ;
    
    printf("total energy: %12.6f \n\n" , e_tot ) ;
    
    fprintf( fd_kin," %12.6f",.5*mtot*vz*vz);
    
    fprintf( fd , "%12.6f    %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f  %6.3f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f  %12.6f  %12.6f  %12.6f" , time , e_tot , num_p , num_n , xcm , ycm , zcm , xcm_p , ycm_p , zcm_p , xcm_n , ycm_n , zcm_n , beta , e_flow_n+e_flow_p , (2.*qzz-qxx-qyy)*dxyz , vx , vy , vz, q30*dxyz , q40*dxyz, 2*qzz/(qxx+qyy), (qxx-qyy)*dxyz, qxy*dxyz, qyz*dxyz, qzx*dxyz) ;
    
    return( e_tot ) ;
    
}



int main( int argc , char ** argv ){
    
    double *dens_p, * dens_n;
    double complex *delta_p,*delta_n;
    double * buff ;
    double e_cut;
    int nx,ny,nz,nwf_p,nwf_n;
    double cc_qzz[4];
    double dx,dy,dz,dxyz,amu_n,amu_p;
    int isospin;
    int ip;
    Couplings cc_edf;
    int iforce=1,ihfb=1;
    //Defining cubic or spherical cutoff here.
    int icub;
    icub = 1; // icub = 1 is cubic cutoff, icub = 0 is spherical cutoff.
    double alpha_pairing=0.0; // pairing mixing parameter: 0 is volume, 0.5 is mixed, and 1.0 is surface.
    double ggn = 1e10, ggp = 1e10; // pairing coupling constants.
    int ifile,ik,i;
    int ibrk=0;
    FILE *fd_out,*fd_out_L,*fd_out_R,*fd_kin ;
    FILE *fd_pf;
    double tolerance = 1.e-7 ;
    double mass_p = 938.272013 ;
    double mass_n = 939.565346 ;
    Lattice_arrays latt , latt3 ;
    FFtransf_vars fftrans ;
    int fd_p,fd_n;
    
    mode_t fd_mode = S_IRUSR | S_IWUSR ; /* S_IRWXU ; S_IRGRP, S_IRWXG ; S_IROTH , S_IRWXO; etc. */
    char fn_p[ 50 ] , fn_n[ 50 ] ;
    
    printf("reading input solver \n" );
    
    read_input_solver( &nx , &ny , &nz , &nwf_p , &nwf_n , &amu_p , &amu_n , &dx , &dy , &dz , &e_cut) ;
    printf("Done. \n" );
    
    double z0; // Boundary for left/right nuclear fragments.

    // User inputs.
    int p;
    while ((p=getopt(argc,argv,"f:z:s:"))!=-1) {
      switch(p){
        case 'f': iforce=atoi(optarg);break; 
        case 'z': z0=atof(optarg);break;
	case 'a': alpha_pairing=atof(optarg);break; 
	case 'p': ggp=atof(optarg);break; 
	case 'n': ggn=atof(optarg);break; 
      }
    }
    
    dens_func_params( iforce , ihfb , 1 , &cc_edf , 0, icub, alpha_pairing);

    if(ggp<1e9){
      cc_edf.gg_p=ggp;
      if(isospin==1) cc_edf.gg=ggp;
    }
    if(ggn<1e9){
      cc_edf.gg_n=ggn;
      if(isospin==-1) cc_edf.gg=ggn;
    }   
 
    int nxyz=nx*ny*nz;
    dens_p=malloc(14*nxyz*sizeof(double));
    dens_n=malloc(14*nxyz*sizeof(double));
    buff=malloc(5*nxyz*sizeof(double));
    delta_p=malloc(nxyz*sizeof(double complex));
    delta_n=malloc(nxyz*sizeof(double complex));
    
    double *thetaL,*thetaR;
    thetaL=malloc(nxyz*sizeof(double));
    thetaR=malloc(nxyz*sizeof(double));
    
    double * densf_p, *densf_n;
    densf_p=malloc(14*nxyz*sizeof(double));
    densf_n=malloc(14*nxyz*sizeof(double));
    
    fd_out = fopen("out.dat","w");
    fd_out_L = fopen("outL.dat","w");
    fd_out_R = fopen("outR.dat","w");
    fd_kin = fopen("out_kin.dat","w");
    fd_pf = fopen("out_pairFluct.dat","w");
    
    dxyz=dx*dy*dz;
    double hbarc = 197.3269631 ;
    double hbar2m = pow( hbarc , 2.0 ) / ( mass_p + mass_n ) ;
    double emax = 0.5 * pow( acos( -1. ) , 2. ) * hbar2m / pow( dx , 2. ) ;

if(icub==1)
    emax *= 4.0;

#ifdef RANDOM
    emax *= 2.0;
#endif
    
    double dt_step = pow( tolerance , 0.2 )  * hbarc / emax ;
    //IS: changed the time step to accommodate the finer lattice
    dt_step = .25*pow(10.,-5./3.)*dx*dx;
    
    int n3=nx;
    int nx3, ny3, nz3;
    if( n3 < ny ){
      n3=ny;
    }
    
    if( n3 < nz ){
      n3=nz;
    }
    
    nx3 = 3 * n3 ;
    ny3 = 3 * n3 ;
    nz3 = 3 * n3 ;
    int nxyz3=nx3*ny3*nz3;
  
    fftrans.nxyz3=nxyz3;
    make_coordinates( nxyz3 , nx3 , ny3 , nz3 , dx , dy , dz , &latt3 ) ;
    make_coordinates( nxyz , nx , ny , nz , dx , dy , dz , &latt ) ;
    
    double lx=dx*nx;
    double ly=dy*ny;
    double lz=dz*nz;
        
    for(i=0;i<nxyz;i++){
      if( latt.za[i]>z0){
            thetaR[i]=1.;
            thetaL[i]=0.;
        }
        if( latt.za[i]<z0){
            thetaR[i]=0.;
            thetaL[i]=1.;
        }
        if(latt.za[i]==z0){
            thetaR[i]=.5;
            thetaL[i]=.5;
        }
    }
    
    double Lc=sqrt(lx*lx+ly*ly+lz*lz);
    match_lattices( &latt , &latt3 , nx , ny , nz , nx3 , ny3 , nz3 , &fftrans , Lc ) ;
    assert( fftrans.buff = malloc( nxyz * sizeof( double complex ) ) ) ;
    assert( fftrans.buff3 = malloc( nxyz3 * sizeof( double complex ) ) ) ;
    fftrans.plan_f = fftw_plan_dft_3d( nx , ny , nz , fftrans.buff , fftrans.buff , FFTW_FORWARD , FFTW_ESTIMATE ) ;
    fftrans.plan_b = fftw_plan_dft_3d( nx , ny , nz , fftrans.buff , fftrans.buff , FFTW_BACKWARD , FFTW_ESTIMATE ) ;
    fftrans.plan_f3 = fftw_plan_dft_3d( nx3 , ny3 , nz3 , fftrans.buff3 , fftrans.buff3 , FFTW_FORWARD , FFTW_ESTIMATE ) ;
    fftrans.plan_b3 = fftw_plan_dft_3d( nx3 , ny3 , nz3 , fftrans.buff3 , fftrans.buff3 , FFTW_BACKWARD , FFTW_ESTIMATE ) ;
    
    int itime=0;
    printf("starting loop...\n" );
    for(ifile=itime;ifile<100000000;ifile+=100){
        sprintf( fn_n, "dens_all_n.dat.%d" , ifile );
        sprintf( fn_p, "dens_all_p.dat.%d" , ifile );
        if ( ( fd_p = open( fn_p , O_RDONLY , fd_mode ) ) == -1 ){
            printf( "File %s was not found " , fn_p );
            break;
        }
        if ( ( fd_n = open( fn_n , O_RDONLY , fd_mode ) ) == -1 ){
            printf( "File %s was not found " , fn_n );
            break;
        }
        for( ik=0; ik<10;ik++){
            if ( ( long ) ( i = read( fd_n , ( void * ) dens_n , 14*nxyz * sizeof( double ) ) ) != ( long ) 14*nxyz * sizeof( double ) ){
                fprintf( stderr , "err: failed to READ %ld bytes from FILE %s (dens_n)\n" , ( long ) 14*nxyz * sizeof( double ) , fn_n ) ;
                ibrk = -1 ;
                break ;
            }
            if ( ( long ) ( i = read( fd_n , ( void * ) delta_n , nxyz * sizeof( double complex ) ) ) != ( long ) nxyz * sizeof( double complex ) ){
                fprintf( stderr , "err: failed to READ %ld bytes from FILE %s (delta_n) \n" , ( long ) nxyz * sizeof( double complex) , fn_n ) ;
                ibrk = -1 ;
                break ;
            }
            if ( ( long ) ( i = read( fd_p , ( void * ) dens_p , 14*nxyz * sizeof( double ) ) ) != ( long ) 14*nxyz * sizeof( double ) ){
                fprintf( stderr , "err: failed to READ %ld bytes from FILE %s (dens_p) \n" , ( long ) 14*nxyz * sizeof( double ) , fn_p ) ;
                ibrk = -1 ;
                break ;
            }
            if ( ( long ) ( i = read( fd_p , ( void * ) delta_p , nxyz * sizeof( double complex ) ) ) != ( long ) nxyz * sizeof( double complex ) ){
                fprintf( stderr , "err: failed to READ %ld bytes from FILE %s (dens_n) \n" , ( long ) nxyz * sizeof( double complex ) , fn_p ) ;
                ibrk = -1 ;
                break ;
            }
            printf("time=%f [%d]\n",itime*dt_step,itime);
            fprintf(fd_kin,"%12.6f",itime*dt_step);
            // the densities are read if you got here
            system_energy( &cc_edf , dens_p , dens_n , nxyz , delta_p , delta_n , (double complex *)(dens_p+12*nxyz) , (double complex *) (dens_n+12*nxyz) , hbar2m , dxyz , &latt , &fftrans , itime*dt_step , fd_out , buff , fd_kin);
            double cf=coul_frag( dens_p , latt.xa , latt.ya , latt.za , nxyz , dxyz,0. );
            fprintf( fd_out," %12.6f \n ", cf );
            
            makeFragment(dens_p, densf_p,thetaL,nxyz);
            makeFragment(dens_n, densf_n,thetaL,nxyz);
            system_energy( &cc_edf , densf_p , densf_n , nxyz , delta_p , delta_n , (double complex *)(densf_p+12*nxyz) , (double complex *) (densf_n+12*nxyz) , hbar2m , dxyz , &latt , &fftrans , itime*dt_step , fd_out_L , buff , fd_kin );
            makeFragment(dens_p, densf_p,thetaR,nxyz);
            makeFragment(dens_n, densf_n,thetaR,nxyz);
            system_energy( &cc_edf , densf_p , densf_n , nxyz , delta_p , delta_n , (double complex *)(densf_p+12*nxyz) , (double complex *) (densf_n+12*nxyz) , hbar2m , dxyz , &latt , &fftrans , itime*dt_step , fd_out_R , buff , fd_kin );
            fprintf( fd_out_L, "\n" );
            fprintf( fd_out_R, "\n" );
            
            fprintf( fd_kin," %12.6f \n", cf);
            
            fprintf(fd_pf,"%12.6f",itime*dt_step);
            
            pairingfluct(fd_pf,delta_p,dens_p,nxyz,dxyz);
            pairingfluct(fd_pf,delta_n,dens_n,nxyz,dxyz);
            fprintf(fd_pf,"\n");
            
            itime+=10;
        }
        if( ibrk == -1 )
            break;
        close(fd_n);
        close(fd_p);
        if(ifile%1000==0){
            fclose(fd_out);
            fclose(fd_out_L);
            fclose(fd_out_R);
            fclose(fd_kin);
            fclose(fd_pf);
            fd_out = fopen("out.dat","a+");
            fd_out_L = fopen("outL.dat","a+");
            fd_out_R = fopen("outR.dat","a+");
            fd_kin = fopen("out_kin.dat","a+");
            fd_pf = fopen("out_pairFluct.dat","a+");
        }
    }
    
    
    free(dens_p); free(dens_n);free(buff);free(delta_p);free(delta_n);
    free(thetaL);free(thetaR);free(densf_p);free(densf_n);
    return 0;
    
}
