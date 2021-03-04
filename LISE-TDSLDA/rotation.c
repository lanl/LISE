// for license information, see the accompanying LICENSE file

/*
  Compute the rotation angle and axis of the nucleus
  Lapack library is needed to diagonalize the mass quadrupole matrix
  
  Author: Shi Jin <js1421@uw.edu>
  Date: 07/20/2017
 */


#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include "vars.h"

#include <mpi.h>

#include <assert.h>

#include "tdslda_func.h"

#define PI 3.141592653589793238462643383279502884197

// <DO NOT CHANGE> lapack routine: diagonalize a symmetric matrix 
//extern void dsyev(char *jobz, char *uplo, int *n, double *A, int *lda, double *w, double *work, int *lwork, int *info);
//
// kr: implicitly declared use of LAPACK is not advised. 
// here's a dirty fix ... one of these declarations is usually supported.
//
// Netlib CBLAS / LAPACKE target
// #define LISE_LA_REF
// #include <lapacke.h>
// #include <cblas.h>
// lapack_int LAPACKE_dsyev( int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w );
//
// (from <essl.h>)
// #define dsyev      esvdsyev
// #define _ESVINT int
// #define _ESVI    _ESVINT *
// void esvdsyev(const char *, const char *, _ESVINT, void *, _ESVINT, double *, double *, _ESVINT, _ESVI);
//
// IBM ESSL target
// #define LISE_LA_ESSL
// #include <essl.h>
// void    dsyev(const char *, const char *, int    , void *, int    , double *, double *, int    , int *);
//
// Intel MKL target
// #define LISE_LA_MKL
// #include <mkl.h>
// void dsyev( const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork, MKL_INT* info );
//

#ifdef LISE_LA_REF
#include <lapacke.h>
#include <cblas.h>
lapack_int LAPACKE_dsyev( int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w );
#endif

#ifdef LISE_LA_ESSL
#include <essl.h>
//void esvdsyev(const char *, const char *, _ESVINT, void *, _ESVINT, double *, double *, _ESVINT, _ESVI);
void dsyev(const char *, const char *, int    , void *, int    , double *, double *, int    , int *);
#endif

#ifdef LISE_LA_MKL
#include <mkl.h>
void dsyev( const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork, MKL_INT* info );
#endif

double rotation(double *theta, double *rho_n, double *rho_p, int nxyz, Lattice_arrays * latt_coords, double dxyz )
{
  double qxx=0.,qxy=0.,qxz=0.,qyy=0.,qyz=0.,qzz=0., xcm, ycm, zcm, num ;
  
  double xa, ya, za, theta0;

  int i , j ;

  // variables in dsyev
  double qq[9], w[3], work[200];

  int lwork=200, info, dim=3;

  char uplo = 'L';
  char jobz = 'V';
  /////

  double rz[3], ax[3] , norm, ratio;

  double * rho;

  assert( rho = malloc( nxyz * sizeof( double  ) ) ) ;
  
  for( i = 0 ; i < nxyz ; i++ ) rho[ i ] = rho_n[i] + rho_p[i] ;
  
  num = center_dist( rho , nxyz , latt_coords , &xcm , &ycm , &zcm ) ;

  // constructing the quadrupole matrix


  for ( i = 0 ; i < nxyz ; ++i )
    {
      xa = latt_coords->xa[ i ] - xcm ;

      ya = latt_coords->ya[ i ] - ycm ;

      za = latt_coords->za[ i ] - zcm ;
      
      qxx += xa*xa*rho[ i ]; // Qxx

      qyy += ya*ya*rho[ i ]; // Qyy

      qzz += za*za*rho[ i ]; // Qzz

      qxy += xa*ya*rho[ i ]; // Qxy

      qxz += xa*za*rho[ i ]; // Qxz

      qyz += ya*za*rho[ i ]; // Qyz
      
    }
    
  qq[0] = qxx*dxyz;
  qq[1] = qxy*dxyz;
  qq[2] = qxz*dxyz;
  qq[3] = qq[1];
  qq[4] = qyy*dxyz;
  qq[5] = qyz*dxyz;
  qq[6] = qq[2];
  qq[7] = qq[5];
  qq[8] = qzz*dxyz;

  
  /* diagonalize the inertial tensor */
  
  // Now we have eigenvectors in qq.
  
#ifdef LISE_LA_ESSL
// IBM ESSL 
// void esvdsyev(const char *, const char *, _ESVINT, void *, _ESVINT, double *, double *, _ESVINT, _ESVI);
//void     dsyev(const char *, const char *, int    , void *, int    , double *, double *, int    , int *);
  dsyev(&jobz, &uplo, dim, qq, dim, w, work, lwork, &info);
#endif

#ifdef LISE_LA_REF
  info = LAPACKE_dsyev(LAPACK_COL_MAJOR,jobz, uplo, dim, qq, dim, w);
#endif

#ifdef LISE_LA_MKL
  dsyev(&jobz, &uplo, &dim, qq, &dim, w, work, &lwork, &info);
#endif

  ratio = w[2]/ (w[0]+w[1]);

  // choose the princile axis with largest eigenvalue

  /*  
  rz = (double *) malloc(dim*sizeof(double));

  ax = (double *) malloc(dim*sizeof(double));
  */

  double rsign=1.;
  if(qq[8]<0.)
    rsign=-1.;
  for (i=0 ;i<3; i++) rz[i] = rsign*qq[6+i];
  
  
  // get the rotation angle magnitude (angle between rz and z axis)
  if(rz[2] > 1) rz[2] = 1.;
  theta0 = acos(rz[2]);

  // get the rotation axis (- rz x z )
  ax[0] = -rz[1]; ax[1] = rz[0]; ax[2] = 0.;

  norm = 0.;

  for(i=0; i<3; i++) norm += ax[i]*ax[i];
  
  for(i=0; i<3; i++) theta[i] = theta0 * ax[i] / sqrt(norm);

  free(rho);

  return ratio;
  
}


void get_omega(double * phi_xs, double * phi_ys, double * phi_zs, int l0, int l1, int l2, int l3, int l4, int l5, double * omega, double dt_step){
  
  omega[0] = (phi_xs[l0] - phi_xs[l1])/dt_step;//(274.*phi_xs[l0] - 600.*phi_xs[l1]+600.*phi_xs[l2]-400.*phi_xs[l3]+150.*phi_xs[l4]-24.*phi_xs[l5]) / 120./ dt_step;
  omega[1] = (phi_ys[l0] - phi_ys[l1])/dt_step;//(274.*phi_ys[l0] - 600.*phi_ys[l1]+600.*phi_ys[l2]-400.*phi_ys[l3]+150.*phi_ys[l4]-24.*phi_ys[l5]) / 120./ dt_step;
  omega[2] = (phi_zs[l0] - phi_zs[l1])/dt_step;//(274.*phi_zs[l0] - 600.*phi_zs[l1]+600.*phi_zs[l2]-400.*phi_zs[l3]+150.*phi_zs[l4]-24.*phi_zs[l5]) / 120./ dt_step;
}
