// for license information, see the accompanying LICENSE file
/*
Axial_symmetry.c 
This file contains util functions for axial-symmetry stuff. 

author: Shi Jin <js1421@uw.edu>
Date: 12/31/15
 */
#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <assert.h>

#include <mpi.h>

#include "vars_nuclear.h"

#define EPS 1e-14

typedef struct
{
  int ix;
  int iy;
  double norm1;  // |xx| + |yy|
  double norm2;  // sqrt(xx**2 + yy**2)
} Points;

int get_pts_tbc(int nx, int ny, int nz, double dx, double dy, Axial_symmetry * ax)
/*
  Using axial symmetry, get the index of pts with different distances to symmetry axis (z)
 */
{
  int ix,iy,iz, ixy, ixyz;
  
  double *xx, *yy;  // the array that stores the square (ix-nx/2)^2+(iy-ny/2)^2

  Points * points_tbc;

  double norm1, norm2;
  
  int npxy;

  int flag; /* flag: 1, the equivalent point is already in the points_tbc array
	              0, the equivalent point is not in the array */
  int * indxy; // the array that stores the index of the points with different distances to the symmetry axis z

  int i,j,k;
  
  assert(xx = (double *)malloc(nx*sizeof(double)));

  assert(yy = (double *)malloc(ny*sizeof(double)));

  // malloc maximum possible space for buffers in the xy-plane

  assert(points_tbc = (Points *)malloc(nx*ny*sizeof(Points)));

  assert(indxy = (int *)malloc(nx*ny*sizeof(int)));   


  for ( ix = 0 ; ix < nx ; ix++ ) xx[ ix ] = dx * (( double ) ix - (double)nx/2.);

  for ( iy = 0 ; iy < ny ; iy++ ) yy[ iy ] = dy * (( double ) iy - (double)ny/2.);

  
  // Search for points on xy-plane with different distances to the z-axis
  
  npxy = 0;
  
  for(ix=0;ix<nx;ix++)
    for(iy=0;iy<ny;iy++)
      {
	flag = 0;

	norm1 = fabs(xx[ix]) + fabs(yy[iy]);
	
        norm2 = sqrt(pow(xx[ix],2.0) + pow(yy[iy],2.0)); // the square of distance to the symmetry axis

	for(i=0;i<npxy;i++)
	  if ((fabs(points_tbc[i].norm1-norm1) < EPS) && (fabs(points_tbc[i].norm2-norm2) < EPS)) 
	    {
	      flag = 1;
	      break;
	    }

	if (flag == 0)  // new distance, add into sq array
	  {
	    points_tbc[npxy].ix = ix;

	    points_tbc[npxy].iy = iy;

	    points_tbc[npxy].norm1 = norm1;

	    points_tbc[npxy].norm2 = norm2;

	    ixy = ix*ny+iy;

	    indxy[npxy] = ixy;

	    npxy += 1;
	  }
		
      }

  * (ax->npts_xy) = npxy;  // number of points need to be calculated in xy plane
  
  // now cast it into 3d system with different iz
  j = 0;

  
  for(i=0;i<npxy;i++)
    for(iz=0;iz<nz;iz++)
      {
	ixyz = indxy[i]*nz+iz;

	* (ax->ind_xyz + j) = ixyz;

	j+=1;
      }

  * (ax->npts_xyz) = nz * npxy; // number of points need to be calculated in xy plane


  // now relates each pt in 3d with a pt in cylindrical coordinates
  
  for(ix=0;ix<nx;ix++)
    for(iy=0;iy<ny;iy++)
      for(iz=0;iz<nz;iz++)
	{
	  ixyz = ix*ny*nz + iy*nz + iz;

	  norm1 = fabs(xx[ix]) + fabs(yy[iy]);
	  
	  norm2 = sqrt(pow(xx[ix],2.0) + pow(yy[iy],2.0));

	  flag = 0;	  
	  
	  for(i=0;i<npxy;i++)
	    {
	      if ((fabs(points_tbc[i].norm1-norm1) < EPS) && (fabs(points_tbc[i].norm2-norm2) < EPS)) 
		{
		  k = nz*i + iz;
		  
		  * (ax->car2cyl + ixyz ) = * (ax->ind_xyz + k);

		  flag = 1;

		  break;
		}
	    }

	  if(flag == 0)
	    {
	      fprintf(stderr, "ERROR: Fail to find pts in cylindrical coordinates");
	      	      	      
	      return -1;
	    }
	  
	}

  free(indxy);

  free(points_tbc);

  free(xx); free(yy);

  return 0;
}



void axial_symmetry_densities(Densities * dens, Axial_symmetry * ax, int nxyz, Lattice_arrays * lattice_coords, FFtransf_vars * fftransf_vars, const MPI_Comm comm, const int iam)
/*
routine that refill the densities in lattice with axial symmetry
 */
{
  int i, j;

  double * tau, * tau1;

  double complex * nu, * nu1;

  double *jjx, * jjy, * jjz, * jjx1, * jjy1, * jjz1;

  assert( tau1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( nu1 = malloc( nxyz * sizeof( double complex ) ) ) ;

  assert( tau = malloc( nxyz * sizeof( double ) ) ) ;

  assert( nu = malloc( nxyz * sizeof( double complex ) ) ) ;

  assert( jjx = malloc( nxyz * sizeof( double ) ) ) ;

  assert( jjy = malloc( nxyz * sizeof( double ) ) ) ;

  assert( jjz = malloc( nxyz * sizeof( double ) ) ) ;

  assert( jjx1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( jjy1 = malloc( nxyz * sizeof( double ) ) ) ;

  assert( jjz1 = malloc( nxyz * sizeof( double ) ) ) ;

  // refill the density: rho
  for(i=0; i<nxyz; i++) *(dens->rho + i) = *(dens->rho + ax->car2cyl[i]);

  for( i = 0 ; i < nxyz ; i++ )

    {

      *( tau1 + i ) = 0. ;

      *( nu1 + i ) = 0. + I * 0. ;

    }

  for( i = dens->nstart ; i < dens->nstop ; i++ )

    {

      *( tau1 + i ) = *( dens->tau + i - dens->nstart ) ;

      *( nu1 + i ) = *( dens->nu + i - dens->nstart ) ;

    }

  MPI_Allreduce( tau1 , tau , nxyz , MPI_DOUBLE , MPI_SUM  , comm ) ;

  MPI_Allreduce( nu1 , nu , nxyz , MPI_DOUBLE_COMPLEX , MPI_SUM  , comm ) ;

  free( nu1 ) ; free( tau1 ) ; 

  // refile tau, nu

  for(i=0; i<nxyz; i++)
    {
      *(tau + i) = *(tau + ax->car2cyl[i]);

      *(nu + i) = *(nu + ax->car2cyl[i]);
    }

 
  // refill jjx, jjy, jjz:

  double * ff;

  assert( ff = malloc( nxyz * sizeof( double ) ) ) ;

  double * xa = lattice_coords->xa;

  double * ya = lattice_coords->ya;

  for( i = 0 ; i < nxyz ; i++ )

    {

      *( jjx1 + i ) = 0. ;

      *( jjy1 + i ) = 0. ;

      *( jjz1 + i ) = 0. ;

    }

  for( i = dens->nstart ; i < dens->nstop ; i++ )

    {

      *( jjx1 + i ) = *( dens->jjx + i - dens->nstart ) ;

      *( jjy1 + i ) = *( dens->jjy + i - dens->nstart ) ;

      *( jjz1 + i ) = *( dens->jjz + i - dens->nstart ) ;

    }

  MPI_Allreduce( jjx1 , jjx , nxyz , MPI_DOUBLE , MPI_SUM  , comm ) ;

  MPI_Allreduce( jjy1 , jjy , nxyz , MPI_DOUBLE , MPI_SUM  , comm ) ;

  MPI_Allreduce( jjz1 , jjz , nxyz , MPI_DOUBLE , MPI_SUM  , comm ) ;

  for(i=0; i<nxyz; i++)
      {
	double tmp = sqrt(pow(xa[i],2.0) + pow(ya[i],2.0));
	
	if (tmp>EPS)
	  {
	    ff[i] = jjx[i] * xa[i] / tmp + jjy[i] * ya[i] / tmp ;
	}
	else
	  {
	    ff[i] = jjx[i] ;
	  }
      }

  // fill in lattices
  for(i=0; i<nxyz; i++)
    {
      ff[i] = ff[ax->car2cyl[i]];
      
      jjz[i] = jjz[ax->car2cyl[i]];
    }

  // re-expand into cartesian

  for(i=0; i<nxyz; i++)
    {
      double tmp = sqrt(pow(xa[i],2.0) + pow(ya[i],2.0));
      
      if (tmp>EPS)
	{
	  jjx[i] = ff[i] * xa[i] / tmp ;
	  
	  jjy[i] = ff[i] * ya[i] / tmp ;	  
	}
      else
	{
	  jjx[i] = ff[i];
	  
	  jjy[i] = 0.0+0.0*I;
	}
    }

  int iwork = dens->nstop - dens->nstart;

  int nstart = dens->nstart;

  int nstop = dens->nstop;

  for(i=0; i<iwork; i++)
    {
      *(dens->tau+i) = *(tau + i + dens->nstart);

      *(dens->nu+i) = *(nu + i + dens->nstart);

      *(dens->jjx+i) = *(jjx + i + dens->nstart);

      *(dens->jjy+i) = *(jjy + i + dens->nstart);

      *(dens->jjz+i) = *(jjz + i + dens->nstart);

    }
  // calculate divjj

  // jjx ***********
  for(i=0; i<nxyz; i++)
    *(fftransf_vars->buff + i) =  *(jjx+i);
  
  fftw_execute(fftransf_vars->plan_f);
  
  for(i=0; i<nxyz; i++)
    *(fftransf_vars->buff + i) *= *(lattice_coords->kx+i) * I / (double) nxyz;
  
  fftw_execute(fftransf_vars->plan_b);
  
  for(i=0;i<iwork;i++)
    *( dens->divjj+i ) = creal( * (fftransf_vars->buff + i + nstart ));
  
  // jjy *************
  
  for(i=0; i<nxyz; i++)
    *(fftransf_vars->buff + i) =  *(jjy+i);
  
  fftw_execute(fftransf_vars->plan_f);
  
  for(i=0; i<nxyz; i++)
    *(fftransf_vars->buff + i) *= *(lattice_coords->ky + i) * I / (double) nxyz;
  
  fftw_execute(fftransf_vars->plan_b);
  
  for(i=0;i<iwork;i++)
    *( dens->divjj+i ) += creal( * (fftransf_vars->buff + i + nstart ));
  
  // jjz **************
  
  for(i=0; i<nxyz; i++)
    *(fftransf_vars->buff + i) =  *(jjz+i);
  
  fftw_execute(fftransf_vars->plan_f);
  
  for(i=0; i<nxyz; i++)
    *(fftransf_vars->buff + i) *= *(lattice_coords->kz + i) * I / (double) nxyz;
  
  fftw_execute(fftransf_vars->plan_b);
  
  for(i=0;i<iwork;i++)
    *( dens->divjj+i ) += creal( * (fftransf_vars->buff + i + nstart ));

  free(jjx); free(jjy); free(jjz);

  free(jjx1); free(jjy1); free(jjz1);

  free(nu); free(tau);
}



