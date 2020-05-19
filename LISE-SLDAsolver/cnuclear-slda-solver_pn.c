// for license information, see the accompanying LICENSE file


/* Main program */

#include <stdlib.h>

#include <getopt.h>

#include <stdio.h>

#include <string.h>

#include <math.h>

#include <mpi.h>

#include "vars_nuclear.h"

#include <complex.h>

#include <assert.h>

int parse_input_file(char * file_name);

int readcmd(int argc, char *argv[], int ip);

double center_dist( double * rho , const int n , Lattice_arrays * latt_coords , double * xc , double * yc , double * zc );

double center_dist_pn( double * rho_p , double * rho_n, const int n , Lattice_arrays * latt_coords , double * xc , double * yc , double * zc );

void print_help( const int ip , char ** argv ) ;

void make_constraint( double * v , double * xa , double * ya , double * za , int nxyz , double y0 , double z0 , double asym , int * wx, int * wy, int * wz,double v0);

void make_ham( double complex * , double * , double * , double * , double complex * , double complex * , double complex * , Potentials * , Densities * , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const MPI_Comm , Lattice_arrays * ) ;

void get_blcs_dscr( MPI_Comm , int , int , int , int , int , int , int * , int * , int * , int * , int * ) ;

int dens_func_params( const int , const int , const int , Couplings * , const int , int icub) ;

void generate_ke_1d( double * , const int , const double , const int ) ;

void generate_der_1d( double complex * , const int , const double , const int ) ;

void external_pot( const int , const int , const int , const double , double * , double complex * , const double , Lattice_arrays * , const double , const double , const double , const double , const double ) ;

void allocate_pots( Potentials * , const double , double * , const int , const int ) ;

void allocate_dens( Densities * , const int , const int , const int ) ;

void make_coordinates( const int , const int , const int , const int , const double , const double , const double , Lattice_arrays * ) ;

void external_so_m( double * , double * , double * , double * , double * , const double , Lattice_arrays * , const MPI_Comm , double complex * , double complex * , double complex * , const int , const int , const int ) ;

void occ_numbers( double complex * , double * , const int , const int , const int , const int , const int , const int , const int , const int , const int , const MPI_Comm , const int ) ;

void compute_densities( double * lam , double complex * z , const int nxyz , const int ip , const MPI_Comm comm , Densities * dens , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc , const int nx , const int ny , const int nz , double complex * d1_x , double complex * d1_y , double complex * d1_z , double * k1d_x, double * k1d_y, double * k1d_z, const double e_max , int * nwf  , double * occ , FFtransf_vars * fftransf_vars , Lattice_arrays * latt_coords ,int icub) ;

void compute_densities_finitetemp( double * lam , double complex * z , const int nxyz , const int ip , const MPI_Comm comm , Densities * dens , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc , const int nx , const int ny , const int nz , double complex * d1_x , double complex * d1_y , double complex * d1_z , const double e_max , const double temp, int * nwf  , double * occ , FFtransf_vars * fftransf_vars , Lattice_arrays * latt_coords ,int icub);

void get_u_re( const MPI_Comm , Densities * , Densities * , Potentials * , Couplings * , const double , const int , const int , const int , const int , const int , double * , double * , double * , const int , const int , const int , Lattice_arrays * , FFtransf_vars * , const double , const double ) ;

void update_potentials( const int , const int , Potentials * , Densities * , Densities * , Densities * , Couplings * , const double , const int , const int , const MPI_Comm , const int , const int , const int , const double , double complex * , double complex * , double complex * , double * , double * , double * , Lattice_arrays * , FFtransf_vars * , const double , const double , int) ;

void mix_potentials( double * , double * , const double , const int , const int ) ;

void dens_gauss( const int , const int , const int , Densities * ,  Lattice_arrays * , const double , const double , const double , const double , const int ) ;

void dens_startTheta( const double A_mass , const double npart , const int nxyz , Densities * dens ,  Lattice_arrays * lattice_coords , const double dxyz , const double lx , const double ly , const double lz , const int ideform );

int minim_int( const int , const int ) ;

int write_dens( char * , Densities * , const MPI_Comm , const int , const int , const int , const int , double * , const double , const double , const double ) ;

int write_dens_td( char * fn , Densities * dens , const MPI_Comm comm , const int iam , const int nx , const int ny , const int nz );

int write_qpe( char * fn, double * lam, const MPI_Comm comm, const int iam, const int nwf);

int read_constr( char * fn , int nxyz, double * cc_lambda, const int iam , MPI_Comm comm);

int read_dens( char * , Densities * , const MPI_Comm , const int , const int , const int , const int , double * , const double , const double , const double , char * filename ) ;

int read_pots( char * , double * , const int , const int , const int , const double , const double , const double , const int ) ;

int write_pots( char * , double * , const int , const int , const int , const double , const double , const double , const int ) ;

int write_dens_txt(FILE *fd, Densities * dens, const MPI_Comm comm, const int iam, const int nx, const int ny, const int nz, double * amu);

void system_energy( Couplings * cc_edf , const int icoul , Densities * dens , Densities * dens_p , Densities * dens_n , const int isospin , double complex * delta , const int nstart , const int nstop , const int ip , const int gr_ip , const MPI_Comm comm , const MPI_Comm gr_comm , double * k1d_x , double * k1d_y , double * k1d_z , const int nx , const int ny , const int nz , const double hbar2m , const double dxyz , Lattice_arrays * latt_coords , FFtransf_vars * fftransf_vars , const double nprot , const double nneut, FILE * out );

void mem_share( Densities * , double * , const int , const int ) ;

double rescale_dens( Densities * , const int , const double , const double , const int ) ;

int create_mpi_groups( const MPI_Comm , MPI_Comm * , const int , int * , int * , MPI_Group * ) ;

void destroy_mpi_groups( MPI_Group * , MPI_Comm * ) ;

void exch_nucl_dens( const MPI_Comm , const int , const int , const int , const int , double * , double * ) ;

int broyden_min( double * , double * , double * , double * , double * , double * , double * , const int , int * , const double ) ;

int broydenMod_min( double * , double * , double * , double * , double * , double * , double * , const int , int * , const double ) ;

extern void broyden_minf_( int *, double * , double * , double * , double * , int * , int * ) ;

int print_wf( char * , MPI_Comm , double * , double complex * , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const double , double * ,int icub) ;

void bc_wr_mpi( char * fn , MPI_Comm com , int p , int q , int ip , int iq , int blk , int jstrt , int jstp , int jstrd , int nxyz , double complex * z ) ;

int print_wf2( char * , MPI_Comm , double * , double complex * , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const int , const double , double * , int icub) ;

void array_rescale( double * , const int , const double ) ;

void change_mu_eq_sp( double * , double * , double * , const int , const int ) ;

double factor_ec( const double , const double ) ;

void deform( double * , double * , int , Lattice_arrays * , double , FILE * ) ;

void cm_initial( double * lam , double complex * z , const int m_ip , const int n_iq , const int i_p , const int i_q , const int mb , const int nb , const int p_proc , const int q_proc , const int nx , const int ny , const int nz , const double e_max , Lattice_arrays * latt_coords , double * xcm , double * ycm , double * zcm , const MPI_Comm comm , int icub) ;

double distMass( double * rho_p, double * rho_n, double n, double z0 , double * za,int *,double dxyz);

double pi ;

double q2av( double * rho_p , double * rho_n , double * qzz , const int nxyz , const double , const double );

double cm_coord( double * rho_p , double * rho_n , double * qzz , const int nxyz, const double np, const double nn );

void axial_symmetry_densities(Densities *, Axial_symmetry *, int, Lattice_arrays *, FFtransf_vars *, const MPI_Comm, const int);

int get_pts_tbc(int, int, int, double, double, Axial_symmetry * );

void make_filter(double *filter, Lattice_arrays *latt_coords, int nxyz);

metadata_t md =
  {
    8, // nx
    8, // ny
    8, // nz
    1.0, // dx
    1.0, // dy
    1.0, // dz
    0, // broyden
    1, // coulomb
    100, // niter
    8, // nprot
    8, // nneut
    0, // iext
    1, // force
    1, // pairing
    0.25, // alpha_mixing
    200.0, // ecut
    1, // icub
    0, // imass
    0, // icm
    0, // irun
    1, // isymm
    0, // resc_dens
    0, // iprint_wf
    0, // deformation
    0.0, // q0 (quadrupole constraints)
    0.0, // v0 (strength of external field)
    0.0, // z0: reference point in fission process
    0.0, // wneck: parameter of twisted potential
    0.0, // rneck: parameter of twisted potential
    8, // p
    8, // q
    40, //mb
    40 // nb
  };


int main( int argc , char ** argv )

{

  double complex * z_eig ;

  double * lam , * lam_old ;

  double complex * ham ; /* the Hamiltonian to be diagonalized, cyclicly decomposed */

  int gr_ip , ip , np , gr_np ;

  int root_n , root_p ;

  int it , niter = 2 , it_broy = 0 ;

  int isospin ;

  double e_cut = 75. ;

  MPI_Comm commw , gr_comm ; /* World and group communicators */

  MPI_Group group_comm ;

  MPI_Status istats ;

  int tag1 = 120 , tag2 = 200 ;

  int i_p , i_q , p_proc = -1 , q_proc = -1 , mb = 40 , nb = 40 , m_ip , n_iq ;  /* used for decomposition */

  Potentials pots , * pots_ptr , pots_old ;

  double v[ 3 ] ;

  double * pot_array , * pot_array_old , * v_in_old ;

  double * f_old , * b_bra , * b_ket , * diag ;

  int i_b ;

  int hw ;

  long long int rcy , rus , ucy , uus ;

  Densities dens_p , dens_n , * dens ;

  Couplings cc_edf ;

  Lattice_arrays lattice_coords , lattice_coords3 ;

  FFtransf_vars fftransf_vars ;

  Axial_symmetry ax;

  double hbar2m ;

  double dx = 1.25 , dy = 1.25 , dz = 1.25 ;

  double Lc ;

  double xcm , ycm , zcm ;

  double rcm[3];

  int descr_h[ 9 ] , descr_z[ 9 ] ;

  int icoul = 1 , ihfb = 1 , imin = 0 , iforce = 1 , ider = 1 ; /* options for 

                                            icoul = 0 no Coulomb, 1 Coulomb = default

				            ihfb = 0 no pairing , 1 pairing

				            imin = 0 simple mixing, 1 Broyden

					    iforce = 0 no interaction, 1 = SLy4 ( default ) , 2 = SkM* , 11 = Sly4 + surface pairing

					    */

  int ideform = 0 ;  /* for initial densities, the deformation ideform = 0 sherical, ideform = 2 axially symmetric , ideform = 3 triaxial */ 

  int iext = 0 ; /* different external potential choices */

  int irun = 0 ; /* 0 = run with spherical density, 1 = read densities */

  int icub = 1; /* 0 = use spherical cutoff, 1 = use cubic cutoff*/

  int imass = 0; /* 0 = same masses for protons and neutrons, 1 = different masses */

  int icm = 0; /* 0 = no center of mass correction, 1 = center of mass correction */

  int irsc = 0 ;

  int iprint_wf = -1 ; /* -1 no wfs saved , 0 all wfs saved , 1 Z proton wfs saved, 2 N neutron wfs saved */

  int m_broy , ishift , m_keep = 7 , it1 ;

  double si ;

  int nx = -1 , ny = -1 , nz = -1 , nxyz ;

  int nx3 , ny3 , nz3 , nxyz3 ;

  double nprot = -1. , nneut = -1. ; /* number of protons, number of neutrons */

  double * k1d_x , * k1d_y , * k1d_z ;

  double * ex_array_n , * ex_array_p ;

  double complex * d1_x , * d1_y , * d1_z ;

  double xpart , sdxyz , dxyz ;

  double * npart ;

  int i , ii , j , na , lwork , lrwork , liwork , info, Ione = 1 , itb ;

  int * iwork ;

  double complex s1 ;

  double complex * work , tw[ 2 ] ;

  double complex * rwork , tw_[2] ;

  double * occ , amu_n , * amu , const_amu=4.e-2,c_q2=1.e-5 ;

  double mass_p = 938.272013 ;

  double mass_n = 939.565346 ;

  double hbarc = 197.3269631 ;

  double hbo = 2.0 , alpha_mix = 0.25 , rone = 1.0 ;

  double err ;

  FILE * file_out ;

  int idim ;

  int nwf , nwf_n ;

  int ix , iy , iz ;

  char fn[ FILENAME_MAX ] = "densities_p_0.cwr" , iso_label[] ="_p";

  char fn_[ FILENAME_MAX ] ; /* for testing the different versions */

  FILE *fd_dns;

  int option_index = 0 ;

  double xconstr[4],z0=0.,y0=0.,asym=0.,v0=0.,wneck=100.,rneck=100.;

  double xlagmul[4];

  double Lx=-1.,Ly=-1.,Lz=-1;

  int ierr;

  setbuf(stdout, NULL);

  static struct option long_options[] = {
    {"nx", required_argument, 0,  0 },
    {"ny", required_argument, 0,  0 },
    {"nz", required_argument, 0,  0 },
    {"Lx", required_argument, 0,  0 },
    {"Ly", required_argument, 0,  0 },
    {"Lz", required_argument, 0,  0 },
    {"dx", required_argument, 0,  0 },
    {"dy", required_argument, 0,  0 },
    {"dz", required_argument, 0,  0 },
    {"broyden", no_argument, 0, 'b' },
    {"nocoulomb", no_argument , 0 , 0 },
    {"niter" , required_argument , 0 , 'n' },
    {"nopairing" , no_argument , 0 , 0 },
    {"iext" , required_argument , 0 , 'e'},
    {"alpha_mix" , required_argument , 0 , 'a'},
    {"force", required_argument , 0 , 'f'},
    {"nprot", required_argument , 0 , 'Z'},
    {"nneut", required_argument , 0 , 'N'},
    {"irun", required_argument , 0 , 'r'},
    {"resc_dens", no_argument , 0 , 0 },
    {"iprint_wf" , required_argument , 0 , 'p' },
    {"ecut" , required_argument , 0 , 0 },
    {"pproc" , required_argument , 0 , 0 },
    {"qproc" , required_argument , 0 , 0 },
    {"nb" , required_argument , 0 , 0 },
    {"mb" , required_argument , 0 , 0 },
    {"deformation" , required_argument , 0 , 0 },
    {"hbo" , required_argument , 0 , 0 },
    {"q0" , required_argument , 0 , 'q' },
    {"v0" , required_argument , 0 , 'v' },
    {"y3" , required_argument , 0 , 0 },
    {"asym" , required_argument , 0 , 0 },
    {"z0" , required_argument , 0 , 'z' },
    {"y0" , required_argument , 0 , 'y' },
    {"wneck" , required_argument , 0 , 0 },
    {"rneck" , required_argument , 0 , 0 },
    {0,         0,                 0,  0 }

  };

  MPI_Init( &argc , &argv ) ;

  MPI_Comm_rank( MPI_COMM_WORLD, &ip ); 

  MPI_Comm_size( MPI_COMM_WORLD, &np );

  commw = MPI_COMM_WORLD ;

  setbuf(stdout, NULL);

  isospin = create_mpi_groups( commw , &gr_comm , np , &gr_np , &gr_ip , &group_comm ) ;

  hbar2m = pow( hbarc , 2.0 ) / ( mass_p + mass_n ) ;


  if( ip == 0 )
    {
      i = readcmd( argc , argv, ip) ;
        if( i == -1 )
        {
	  
	  printf( "TERMINATING! NO INPUT FILE.\n" ) ;
            ierr = -1 ;
            MPI_Abort( MPI_COMM_WORLD , ierr ) ;
            return( EXIT_FAILURE ) ;
        }
        
        // Read input file
        // Info from file is loaded into metadata structure
        printf("READING INPUT: `%s`\n", argv[ i ]);
        
        j = parse_input_file(argv[i]);
        if ( j == 0 )
        {
            ierr = -1 ;
            printf("PROBLEM WITH INPUT FILE: `%s`.\n" , argv[ i ] ) ;
            MPI_Abort( MPI_COMM_WORLD , ierr ) ;
            return( EXIT_FAILURE ) ;      
        }   
    }
    
    // Broadcast input parameters
    MPI_Bcast( &md , sizeof(md) , MPI_BYTE , 0 , MPI_COMM_WORLD ) ;


    nx = md.nx; ny = md.ny; nz = md.nz;
    
    dx = md.dx; dy = md.dy; dz = md.dz;
    
    if (md.broyden == 1) imin = 1;  // broyden minimization will be turned on
    
    if (md.coulomb == 0) icoul = 0; // no coulomb potential
    
    nprot = md.nprot;
    
    nneut = md.nneut;
    
    niter = md.niter;
   
    icm = md.icm; 

   if(icm == 1)

    hbar2m = hbar2m * (1.0 - 1.0/(nneut + nprot));

   imass = md.imass;

   if(imass == 1) {

    hbar2m = .5*pow( hbarc , 2.0 ) ;

    if(isospin==1)

      hbar2m /= mass_p ;
 
    else

      hbar2m /= mass_n;

   }

    
    iext = md.iext;
    
    iforce = md.force;
  
  if (md.pairing == 0) ihfb = 0;
  
  alpha_mix = md.alpha_mix;   

  e_cut = md.ecut;   // energy cutoff

  double e_cut_spherical = hbar2m * PI * PI / dx / dx;

  if(icub == 0)
    e_cut = e_cut_spherical;
 
  irun = md.irun;   // choice of start of density
 
  if (md.resc_dens == 1) irsc = 1;
  
  iprint_wf = md.iprint_wf;   // choice of printing wfs
  
  ideform = md.deformation;

  xconstr[0] = md.q0; // quadruple constraint

  z0 = md.z0;  // reference point of distribution
  
  wneck = md.wneck;
  
  rneck = md.rneck;
  
  v0 = md.v0;
 
  // cyclic distribution in scalapack
  p_proc = md.p;   
  
  q_proc = md.q;
  
  mb = md.mb;
  
  nb = md.nb;
  

  if( nx < 0 || ny < 0 || nz < 0 || nprot < 0. || nneut < 0. )

    {

      if( ip == 0 )

	{

	  fprintf(stdout,"nx=%d ny=%d nz=%d \n" , nx , ny , nz ) ;

	  fprintf(stdout,"Lx=%f Ly=%f Lz=%f \n" , Lx , Ly , Lz ) ;

	  fprintf(stdout,"nprot=%f nneut=%f\n" , nprot, nneut ) ;

	  fprintf(stdout, "required parameters not provided \n\n " ) ;

	}

      print_help( ip , argv ) ;

      MPI_Finalize() ;

      return( EXIT_FAILURE ) ;

    }

  if( p_proc < 0 && q_proc < 0 )

    {

      p_proc = ( int ) sqrt( gr_np ) ;

      q_proc = p_proc ; /* square grids, although not required */

    }

  else

    {

      if( p_proc < 0 )

	p_proc = gr_np / q_proc ;

      if( q_proc < 0 )

	q_proc = gr_np / p_proc ;

      if( gr_ip == 0 )

	fprintf(stdout, "%d x %d grid \n" , p_proc , q_proc ) ;

    }

  pi = acos( -1.0 ) ; 

  nxyz = nx * ny * nz ;

  Lx = nx * dx;

  Ly = ny * dy;

  Lz = nz * dz;

  dxyz = dx * dy * dz ;

  sdxyz = 1. / sqrt( dxyz ) ;

  pots.nxyz = nxyz ;

  pots_old.nxyz = nxyz ;

  int isymm = md.isymm;

  if( ip == 0 )

    {

      fprintf( stdout, " *********************************** \n" ) ;
      
      fprintf( stdout, " * Welcome to the world of wonders *\n * brought to you by the magic of SLDA *\n " ) ;

      fprintf( stdout, " *********************************** \n\n\n" ) ;

      fprintf( stdout, " You have requested a calculation with the following parameters: \n" ) ;

      fprintf( stdout, " nx=%d ny=%d nz=%d dx=%g dy=%g dz=%g\n" , nx , ny , nz , dx , dy , dz ) ;

      fprintf( stdout, " Lx=%g fm Ly=%g fm Lz=%g fm\n" , Lx , Ly , Lz ) ;

      fprintf( stdout, " Z=%f N=%f\n\n" , nprot , nneut ) ;

    }

  assert(ax.npts_xy = (int *) malloc(sizeof(int)));

  assert(ax.npts_xyz = (int *) malloc(sizeof(int)));

  assert(ax.ind_xyz = (int *) malloc(nxyz*sizeof(int)));

  assert(ax.car2cyl = (int *) malloc(nxyz*sizeof(int)));

  
  if (isymm == 1)
    {
      if ((ierr = get_pts_tbc(nx, ny, nz, dx, dy, &ax)) != 0)
	{
	  fprintf(stdout, "ip[%d], ERROR IN CONSTRUCTING CYLINDRICAL COORDINATES, EXITTING ....\n", ip);
	  MPI_Abort(MPI_COMM_WORLD, ierr);
	  return(EXIT_FAILURE);
        }
      else
	{
          if(ip==0)
	    {
              fprintf(stdout,"%d pts need to be calculated in xy-plane, %d totally in full space\n", *(ax.npts_xy), *(ax.npts_xyz));
            }
	  
        }
    }
 else
  // calculate full lattice 
   {
      ax.npts_xy[0] = nx*ny;
      
      ax.npts_xyz[0] = nxyz;
      
      for(i=0;i<nxyz;i++)
	{
          ax.ind_xyz[i] = i;
          ax.car2cyl[i] = i;
        }

      if(ip==0)
	{
	  fprintf( stdout, "%d pts need to be calculated in xy-plane, %d totally in full space\n", *(ax.npts_xy), *(ax.npts_xyz));
	}
  
    }


  if( imin == 0 )

    {

      m_broy = 7 * nxyz + 5 ;

      ishift = 0 ;

      if( ip == 0 )

	fprintf(stdout,  "Linear mixing will be performed \n" ) ;

    }

  else

    {

      m_broy =  2 * ( 7 * nxyz + 5 ) ;

      if( ip == 0 )

	{

	  fprintf(stdout,  "Broyden mixing will be performed\n" ) ;

	  assert( f_old = malloc( m_broy * sizeof( double ) ) ) ;

	  assert( b_bra = malloc( ( niter - 1 ) * m_broy * sizeof( double ) ) ) ;

	  assert( b_ket = malloc( ( niter - 1 ) * m_broy * sizeof( double ) ) ) ;

	  assert( diag = malloc( ( niter - 1 ) * sizeof( double ) ) ) ;

	  assert( v_in_old = malloc( m_broy * sizeof( double ) ) ) ;

	}

      ishift = ( 7 * nxyz + 5 ) * ( 1 - isospin ) / 2 ;

    }

  if( ip == 0 )

    fprintf(stdout, "alpha_mix = %f \n" , alpha_mix ) ;

  root_p = 0 ;

  root_n = gr_np ;

  assert( pot_array = malloc( m_broy * sizeof( double ) ) ) ;

  assert( pot_array_old = malloc( m_broy * sizeof( double ) ) ) ;

  allocate_pots( &pots , hbar2m , pot_array , ishift , m_broy ) ;

  allocate_pots( &pots_old , hbar2m , pot_array_old , ishift , m_broy ) ;

  if( isospin == 1 ) 

    {

      dens = & dens_p ;

    }

  else

    {

      dens = & dens_n ;

    }

  *( pots.amu ) =-12.5 ;

  *( pots_old.amu ) = -12.5 ;

  for(i=0;i<4;i++){
    pots.lam[i]=0.;
  }

  for(i=1;i<4;i++){
    xconstr[i]=0.;
  }

  allocate_dens( &dens_p , gr_ip , gr_np , nxyz ) ;

  allocate_dens( &dens_n , gr_ip , gr_np , nxyz ) ;

  

  if( dens->nstop - dens->nstart > 0 )

    idim = 2. * ( dens->nstop - dens->nstart ) + nxyz ;

  else

    idim = nxyz ;

  assert( ex_array_p = malloc( idim * sizeof( double ) ) ) ;

  assert( ex_array_n = malloc( idim * sizeof( double ) ) ) ;

  mem_share( &dens_p , ex_array_p , nxyz , idim ) ;

  mem_share( &dens_n , ex_array_n , nxyz , idim ) ;

  if( isospin == 1 )

    {
    
      dens->nu = dens_p.nu ;

      npart = &nprot ;

      amu = pots.amu ;

      if( dens->nstart - dens->nstop > 0 ) 

	free( dens_n.nu ) ;

    }

  if( isospin == -1 )

    {
   
      dens->nu = dens_n.nu ;

      icoul = 0 ;

      npart = &nneut ;

      if( dens->nstart - dens->nstop > 0 ) 

	free( dens_p.nu ) ;

      amu = pots.amu ;

      sprintf( iso_label , "_n" ) ;

    }

    
  double dx_=dx;
  int n3=nx;

  if( n3 < ny ){
    n3=ny;
    dx_=dy;
  }

  if( n3 < nz ){
    n3=nz;
    dx_=dz;
  }

  nx3 = 3 * n3 ;

  ny3 = 3 * n3 ;

  nz3 = 3 * n3 ;

  fftransf_vars.nxyz3 = nx3 * ny3 *nz3 ;

  nxyz3 = fftransf_vars.nxyz3 ;

  make_coordinates( nxyz , nx , ny , nz , dx , dy , dz , &lattice_coords ) ;

  make_coordinates( nxyz3 , nx3 , ny3 , nz3 , dx , dy , dz , &lattice_coords3 ) ;

  Lc = sqrt( pow( n3 * dx_ , 2. ) + pow( n3 * dx_ , 2. ) + pow( n3 * dx_ , 2. ) ) ;

  match_lattices( &lattice_coords , &lattice_coords3 , nx , ny , nz , nx3 , ny3 , nz3 , & fftransf_vars , Lc ) ;

  assert( fftransf_vars.buff = malloc( nxyz * sizeof( double complex ) ) ) ;

  assert( fftransf_vars.buff3 = malloc( nxyz3 * sizeof( double complex ) ) ) ;

  fftransf_vars.plan_f = fftw_plan_dft_3d( nx , ny , nz , fftransf_vars.buff , fftransf_vars.buff , FFTW_FORWARD , FFTW_ESTIMATE ) ;

  fftransf_vars.plan_b = fftw_plan_dft_3d( nx , ny , nz , fftransf_vars.buff , fftransf_vars.buff , FFTW_BACKWARD , FFTW_ESTIMATE ) ;

  fftransf_vars.plan_f3 = fftw_plan_dft_3d( nx3 , ny3 , nz3 , fftransf_vars.buff3 , fftransf_vars.buff3 , FFTW_FORWARD , FFTW_ESTIMATE ) ;

  fftransf_vars.plan_b3 = fftw_plan_dft_3d( nx3 , ny3 , nz3 , fftransf_vars.buff3 , fftransf_vars.buff3 , FFTW_BACKWARD , FFTW_ESTIMATE ) ;

  assert( fftransf_vars.filter = malloc( nxyz * sizeof( double ) ) ) ;

  for( ix = 0 ; ix < nx ; ix++ )

    for( iy = 0 ; iy < ny ; iy++ )

      for( iz = 0 ; iz < nz ; iz++ )

	if( ix == nx / 2 || iy == ny / 2 || iz == nz / 2 )

	  fftransf_vars.filter[ iz + nz * ( iy + ny * ix ) ] = 0. ;

	else

	  fftransf_vars.filter[ iz + nz * ( iy + ny * ix ) ] = 1. ;


  /* constructing the phase factor in densities */
  
  assert(dens->phases = (double complex *) malloc(nxyz * sizeof(double complex)));

  #define EPS 1e-14
  
  for(i = 0; i<nxyz; i++)
    {
      double xa = lattice_coords.xa[i];

      double ya = lattice_coords.ya[i];
      
      double rr = sqrt(pow(xa, 2.0) + pow(ya, 2.0));

      if(rr < (double) EPS)  // at center of vortex (vortex line) the pairing field should vanish
	dens->phases[i] = 0.0 + 0.0*I;
      else
	{
	  double phi = asin(ya / rr);

	  if(phi > (double)EPS)
	    {
	      if (xa < 0.)
		phi = (double) PI - phi;
	    }
	  else if (phi < - (double) EPS)
	    {
	      if (xa < 0.)
		phi = - (double) PI - phi;
	    }
	  else
	    {
	      if (xa < 0.)
		phi = - (double) PI;
	    }
	  dens->phases[i] = cexp(I*phi);
	}
    }
  /*************************************************************/  

  if( icoul == 1 ) /* declare variables for fourier transforms needed for Coulomb */

    {

      if( gr_ip == 0 )

	fprintf(stdout, "Coulomb interaction between protons included \n" ) ;

    }

  switch( irun )

    {

    case 0:

      dens_startTheta( nprot + nneut , * npart , nxyz , dens , &lattice_coords , dxyz , nx * dx , ny * dy , nz * dz , ideform ) ;

      exch_nucl_dens( commw , ip , gr_ip , gr_np , idim , ex_array_p , ex_array_n ) ;

      if( ip == 0 ){

	fprintf( stdout, "initial densities set\n" ) ;
	
	double * rho_t;

	rho_t=malloc(nxyz*sizeof(double));

	for(i=0;i<nxyz;i++)
	  rho_t[i]=dens_p.rho[i]+dens_n.rho[i];
	center_dist( rho_t , nxyz , &lattice_coords , &xcm , &ycm , &zcm );

	fprintf(stdout, "Initial: xcm=%f ycm=%f zcm=%f\n",xcm,ycm,zcm);

	free(rho_t);

      }
      break ;

    case 1:

      sprintf( fn , "dens%s.cwr" , iso_label ) ;
      sprintf( fn_ , "interpDens%s.cwr" , iso_label ) ;

      if( gr_ip == 0 )

	fprintf(stdout,  "Reading densities from %s\n" , fn ) ;

      if( read_dens( fn , dens , gr_comm , gr_ip , nx , ny , nz , amu , dx , dy , dz , fn_ ) == EXIT_FAILURE )

	{

	  if( gr_ip == 0 )

	    fprintf( stdout , " \n *** \n Could not read the densities \n Exiting \n" ) ;

	  MPI_Finalize() ;

	  return( EXIT_FAILURE ) ;

	}

      else

	{

	  MPI_Barrier( gr_comm ) ;

	  if( gr_ip == 0 )

	    fprintf(stdout, "The densities were successfully read by proc %d\n" , ip ) ;
	
	}

      exch_nucl_dens( commw , ip , gr_ip , gr_np , idim , ex_array_p , ex_array_n ) ;

      break ; 

    case 2:

      if( gr_ip == 0 )

	{ 

	  sprintf( fn , "pots%s.cwr" , iso_label ) ;

	  i = read_pots( fn , pot_array , nx , ny , nz , dx , dy , dz , ishift ) ;

	  ii = i ;

	}

      MPI_Bcast( &i , 1 , MPI_INT , root_p , commw ) ;

      if( i == EXIT_FAILURE )

	{

	  MPI_Finalize() ;

	  return( EXIT_FAILURE ) ;

	}

      MPI_Bcast( &ii , 1 , MPI_INT , root_n , commw ) ;

      if( ii == EXIT_FAILURE )

	{

	  MPI_Finalize() ;

	  return( EXIT_FAILURE ) ;

	}

      if( imin == 0 )

	MPI_Bcast( pot_array , 7 * nxyz + 5 , MPI_DOUBLE , 0 , gr_comm ) ;

      else

	{

	  MPI_Bcast( pot_array + 7 * nxyz + 5 , 7 * nxyz + 5 , MPI_DOUBLE , root_n , commw ) ;

	  MPI_Bcast( pot_array , 7 * nxyz + 5 , MPI_DOUBLE , root_p , commw ) ;

	}

      break ;

    }

  external_pot( iext , nxyz , (int) ( nprot + nneut ) , hbo , pots.v_ext , pots.delta_ext , hbar2m , &lattice_coords , .35 * nx * dx , rneck , wneck , z0 , v0 ) ;


  // using HFBTHO constraint field
  if(iext == 70)
    {

      double cc_lambda[3];

      double * constr;

      double *filter;

      assert(constr = (double *) malloc(3*nxyz*sizeof(double)));

      assert(filter = (double *) malloc(3*nxyz*sizeof(double)));
  
      read_constr( "constr.cwr" , nxyz, cc_lambda, gr_ip , gr_comm);

      for(i=0;i<nxyz;i++) filter[i] = 1.;

      make_filter(filter, &lattice_coords, nxyz);

      for(i=0;i<nxyz;i++)
	  {
	    double xa =lattice_coords.xa[i];
	    double ya =lattice_coords.ya[i];
	    double za =lattice_coords.za[i];

	    	    
	    constr[i] = za/10;  // dipole

	    constr[i+nxyz] = (2*za*za-xa*xa-ya*ya)/100.0;  // quadrupole

	    constr[i+2*nxyz] = za*(2*za*za-3*xa*xa-3*ya*ya)*sqrt(7/pi)/4/1000.0; // oxatupole

	    
	  }

      
      for(i=0;i<nxyz;i++) pots.v_ext[i] = 0.0;

      for(j=0;j<3;j++)
	for(i=0;i<nxyz;i++)
	  pots.v_ext[i] += -1.*(constr[i+j*nxyz]*cc_lambda[j])*filter[i];

      // add a Wood-Saxon filter
      
      free(constr); free(filter);
    }


#ifdef CONSTRCALC

  make_constraint( pots.v_constraint , lattice_coords.xa , lattice_coords.ya , lattice_coords.za , nxyz , y0 , z0 , asym ,lattice_coords.wx, lattice_coords.wy,lattice_coords.wz,v0);

#ifdef CONSTR_Q0
  c_q2=0.05/xconstr[0];
  pots.lam2[ 0 ] = c_q2* ( q2av( dens_p.rho , dens_n.rho , pots.v_constraint , nxyz , nprot , nneut )*dxyz - xconstr[0] ) + pots.lam[ 0 ];  
#endif
  center_dist_pn( dens_p.rho , dens_n.rho, nxyz , &lattice_coords , rcm , rcm+1 , rcm+2 );
  for(j=1;j<4;j++){
    pots.lam2[ j ] = 3.e-1* ( rcm[j-1] - xconstr[j] ) + pots.lam[ j ];
  }

#endif

  dens_func_params( iforce , ihfb , isospin , &cc_edf , ip ,icub) ; 

  /* need to set the grid here */

  na = 4 * nxyz ;

  get_blcs_dscr( gr_comm , na , na , mb , nb , p_proc , q_proc , &i_p , &i_q , descr_h , &m_ip , &n_iq ) ; 

  assert( ham = malloc( m_ip * n_iq * sizeof( double complex ) ) ) ;

  assert( z_eig = malloc( m_ip * n_iq * sizeof( double complex ) ) ) ;

  assert( lam = malloc( na * sizeof( double ) ) ) ;

  assert( lam_old = malloc( 2 * nxyz * sizeof( double ) ) ) ;

  for(i=0;i<2*nxyz;i++) lam_old[i] = 0.0;

  /* allocate some extra arrays to store the 1D KE and Derivatives */

  assert( k1d_x = malloc( nx * sizeof( double ) ) ) ;

  assert( k1d_y = malloc( ny * sizeof( double ) ) ) ;

  assert( k1d_z = malloc( nz * sizeof( double ) ) ) ;

  generate_ke_1d( k1d_x , nx , dx , ider ) ;

  generate_ke_1d( k1d_y , ny , dy , ider ) ;

  generate_ke_1d( k1d_z , nz , dz , ider ) ;

  assert( d1_x = malloc( ( 2 * nx - 1 ) * sizeof( double complex ) ) ) ;

  assert( d1_y = malloc( ( 2 * ny - 1 ) * sizeof( double complex ) ) ) ;

  assert( d1_z = malloc( ( 2 * nz - 1 ) * sizeof( double complex ) ) ) ;

  generate_der_1d( d1_x , nx , dx , ider ) ;

  generate_der_1d( d1_y , ny , dy , ider ) ;

  generate_der_1d( d1_z , nz , dz , ider ) ;

  if( iext == 5 && irun < 2 )

    external_so_m( pots.v_ext , pots.mass_eff , pots.wx , pots.wy , pots.wz , hbar2m , &lattice_coords , gr_comm , d1_x , d1_y , d1_z , nx , ny , nz ) ;

  assert( occ = malloc( 2 * nxyz * sizeof( double ) ) ) ;

  if( iforce != 0 && irun < 2 )

    {

      get_u_re( gr_comm , &dens_p , &dens_n , &pots , &cc_edf , hbar2m , dens->nstart , dens->nstop , nxyz , icoul , isospin , k1d_x , k1d_y , k1d_z , nx , ny , nz , & lattice_coords , & fftransf_vars , nprot , dxyz ) ;

      update_potentials( icoul , isospin , &pots , &dens_p , &dens_n , dens , &cc_edf , e_cut , dens->nstart , dens->nstop , gr_comm , nx , ny , nz , hbar2m , d1_x , d1_y , d1_z , k1d_x , k1d_y , k1d_z , & lattice_coords , & fftransf_vars , nprot , dxyz, icub ) ;

    }

  #ifdef USEPOTENT
  if( gr_ip == 0 )

	{ 

	  sprintf( fn , "pots%s.cwr" , iso_label ) ;

	  i = read_pots( fn , pot_array , nx , ny , nz , dx , dy , dz , ishift ) ;

	  ii = i ;

	}

      MPI_Bcast( &i , 1 , MPI_INT , root_p , commw ) ;

      if( i == EXIT_FAILURE )

	{

	  MPI_Finalize() ;

	  return( EXIT_FAILURE ) ;

	}

      MPI_Bcast( &ii , 1 , MPI_INT , root_n , commw ) ;

      if( ii == EXIT_FAILURE )

	{

	  MPI_Finalize() ;

	  return( EXIT_FAILURE ) ;

	}

      if( imin == 0 )

	MPI_Bcast( pot_array , 7 * nxyz + 5 , MPI_DOUBLE , 0 , gr_comm ) ;

      else

	{

	  MPI_Bcast( pot_array + 7 * nxyz + 5 , 7 * nxyz + 5 , MPI_DOUBLE , root_n , commw ) ;

	  MPI_Bcast( pot_array , 7 * nxyz + 5 , MPI_DOUBLE , root_p , commw ) ;

	}


  #endif

  if( imin != 0 )

    {

      if( ip == root_n )
    
	MPI_Send( pot_array + m_broy/2 , m_broy/2 , MPI_DOUBLE , root_p , tag1 , commw ) ;

      if( ip == root_p )

	MPI_Recv( pot_array + m_broy/2 , m_broy/2 , MPI_DOUBLE , root_n , tag1 , commw , &istats ) ;

    }

  mix_potentials( pot_array ,  pot_array_old , rone , 0 , m_broy ) ; 
  
  xpart = rescale_dens( dens , nxyz , * npart , dxyz , irsc ) ;

  double xxpart = *npart * xpart;

  const_amu = 25. / *npart;

  if(gr_ip == 0)
    {
      sprintf( fn , "dens%s_start_info.txt" , iso_label ) ;
      
      file_out = fopen(fn,"w");
      
      if( isospin == 1 )
	
	deform( dens_p.rho , dens_n.rho , nxyz , &lattice_coords , dxyz , file_out) ;
      
      fprintf( stdout , "N%s = %13.9f\n" , iso_label , xxpart ) ;

      fprintf( stdout , "mu%s = %f\n", iso_label, *amu);
    }
  
  if(irun == 1)
    system_energy( &cc_edf , icoul , dens , &dens_p , &dens_n , isospin , pots.delta , dens->nstart , dens->nstop , ip , gr_ip , commw , gr_comm , k1d_x , k1d_y , k1d_z , nx , ny , nz , hbar2m , dxyz , & lattice_coords , & fftransf_vars , nprot , nneut, file_out ) ;
     
  double complex *delta_old;

  assert(delta_old = (double complex *) malloc(nxyz*sizeof (double complex)));

  for(i=0;i<nxyz;i++) delta_old[i] = pots.delta[i];
  
  if( write_dens_txt( file_out , dens , gr_comm , gr_ip , nx , ny , nz , amu ) != EXIT_SUCCESS )
    
  {
      
       fprintf(stdout, "Error, could not write densities from input file\n");
      
      MPI_Finalize() ;
      
      return( EXIT_FAILURE ) ;
      
    }
  
  
  if(gr_ip == 0)
    fclose(file_out);

  if(gr_ip == 0)
    {
      sprintf( fn , "pots%s_start.txt" , iso_label ) ;
      
      file_out = fopen(fn,"w");      
      
      for(i=0;i<nxyz;i++)
	fprintf(file_out, "delta[%d] = %.12le %12leI\n", i, creal(pots.delta[i]), cimag(pots.delta[i]));
      fclose(file_out);
    }
  
  for( it = 0 ; it < niter ; it++ )

    {

      make_ham( ham , k1d_x , k1d_y , k1d_z , d1_x , d1_y , d1_z , &pots , dens, nx , ny , nz , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc , dens->nstart , dens->nstop , gr_comm , & lattice_coords ) ;

      /* scalapack diag */

      lwork = -1 ; /* probe the system for information */

      lrwork = -1 ;

      liwork = 7 * na + 8 * n_iq + 2 ;

      assert( iwork = malloc( liwork * sizeof( int ) ) ) ;

      pzheevd( "V" , "L" , &na , ham , &Ione , &Ione , descr_h , lam , z_eig , &Ione , &Ione , descr_h , tw , &lwork , tw_ , &lrwork , iwork , &liwork , &info ) ;

      liwork = iwork[ 0 ] ;

      free( iwork ) ;

      assert( iwork = malloc( liwork * sizeof( int ) ) ) ;

      lwork = ( int ) creal( tw[ 0 ] ) ;

      assert( work = malloc( lwork * sizeof( double complex ) ) ) ;

      lrwork = ( int ) creal( tw_[ 0 ] ) ;

      assert( rwork = malloc( lrwork * sizeof( double complex ) ) ) ;

      pzheevd( "V" , "L" , &na , ham , &Ione , &Ione , descr_h , lam , z_eig , &Ione , &Ione , descr_h , work , &lwork , rwork , &lrwork , iwork , &liwork , &info ) ;

      free( iwork ) ; free( work ) ; free( rwork ) ;

      for( ii = 0 ; ii < m_ip * n_iq ; ii++ )

	*( z_eig + ii ) = sdxyz * *( z_eig + ii ) ;

      compute_densities( lam , z_eig , nxyz , gr_ip , gr_comm , dens , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc , nx , ny , nz , d1_x , d1_y , d1_z , k1d_x, k1d_y, k1d_z, e_cut , &nwf , occ , &fftransf_vars , &lattice_coords ,icub) ;

      if(isymm == 1)
	axial_symmetry_densities(dens, &ax, nxyz, &lattice_coords, &fftransf_vars, gr_comm, gr_ip);

      for( i = 0 ; i < nwf ; i++ )

	occ[ i ] = occ[ i ] * dxyz ;

      xpart = rescale_dens( dens , nxyz , * npart , dxyz , irsc ) ;

      exch_nucl_dens( commw , ip , gr_ip , gr_np , idim , ex_array_p , ex_array_n ) ;

      sprintf( fn , "dens%s_%1d.cwr" , iso_label , it % 2 ) ;

      if( write_dens( fn , dens , gr_comm , gr_ip , nx , ny , nz , amu , dx , dy , dz ) != EXIT_SUCCESS )

	{

	  fprintf(stdout, "Error, could not save densities in iteration %d of %d", it + 1 , niter ) ;

	  MPI_Finalize() ;

	  return( EXIT_FAILURE ) ;

	}

      update_potentials( icoul , isospin , &pots , &dens_p , &dens_n , dens , &cc_edf , e_cut , dens->nstart , dens->nstop , gr_comm , nx , ny , nz , hbar2m , d1_x , d1_y , d1_z , k1d_x , k1d_y , k1d_z , & lattice_coords , & fftransf_vars , nprot , dxyz , icub) ;

      if(gr_ip == 0)
	{
	  err = 0.0;
	  
	  for(i=0;i<nxyz;i++)
	    err += cabs(pots.delta[i] - delta_old[i]);
	  
	  fprintf(stdout, "err of pairing_gap%s: %.12le\n", iso_label, err);
	}   

      xxpart = *npart * xpart;

      *amu -= const_amu*(xxpart-*npart);

#ifdef CONSTRCALC

#ifdef CONSTR_Q0
      pots.lam[ 0 ] += c_q2*( q2av( dens_p.rho , dens_n.rho , pots.v_constraint , nxyz , nprot , nneut )*dxyz - xconstr[0] ) ;
#endif

      center_dist_pn( dens_p.rho , dens_n.rho, nxyz , &lattice_coords , rcm , rcm+1 , rcm+2 );

      for( j=1; j< 4;j++){

	pots.lam[ j ] += 3.e-1*( rcm[j-1] - xconstr[j] ) ;
      }

      if( ip == 0 ){
	fprintf(stdout, "D12=%f fm\n" ,  distMass( dens_p.rho , dens_n.rho, nxyz , 0. , lattice_coords.za , lattice_coords.wz , dxyz ) );
	j=0;
	fprintf(stdout, "lam[%d]=%f <O[%d]>=%e %f \n" , j, pots.lam[j] , j , dxyz*q2av( dens_p.rho , dens_n.rho , pots.v_constraint + j*nxyz, nxyz , nprot , nneut ) , xconstr[j] );
	for(j=1; j<4;j++)
	  fprintf(stdout, "lam[%d]=%f <O[%d]>=%e %f \n" , j, pots.lam[j] , j , rcm[j-1] , xconstr[j] );

      }
#endif

      if( ip == 0 ){
	sprintf( fn , "dens%s_%1d.dat" , iso_label, it % 2 ) ;
	fd_dns=fopen( fn , "w" );
	for( i=0;i<nxyz;i++ ){
	  if(lattice_coords.xa[i]==0. && lattice_coords.ya[i]==0.)
	    fprintf( fd_dns , "%f %e\n" , lattice_coords.za[i] , dens_n.rho[i]+dens_p.rho[i]);
	}
	fclose(fd_dns);
	sprintf( fn , "dens_%1d.dat" , it % 2 ) ;
	fd_dns=fopen( fn , "w" );
	for( i=0;i<nxyz;i++ ){
	  if(lattice_coords.ya[i]==0.)
	    fprintf( fd_dns , "%f %f %e\n" , lattice_coords.xa[i],lattice_coords.za[i] , dens_n.rho[i]+dens_p.rho[i]);
	}

	fclose(fd_dns);
      }


      MPI_Barrier(commw);

      if( imin == 0 )

	mix_potentials( pot_array ,  pot_array_old , alpha_mix , ishift , m_broy ) ; 

      else

	{

	  if( ip == root_n ){

	    MPI_Send( pot_array + m_broy / 2 , m_broy / 2 , MPI_DOUBLE , root_p , tag1 , commw ) ;

	  }

	  if( ip == root_p )

	    {

	      MPI_Recv( pot_array + m_broy / 2 , m_broy / 2 , MPI_DOUBLE , root_n , tag1 , commw , &istats ) ;

	      i_b = broydenMod_min( pot_array_old , v_in_old , f_old , b_bra , b_ket , diag , pot_array , m_broy , &it_broy , 1. - alpha_mix ) ;

	    }

	  MPI_Bcast( &i_b , 1 , MPI_INT , root_p , commw ) ;
 
	  MPI_Bcast( pot_array , m_broy , MPI_DOUBLE , root_p , commw ) ;

	}

#ifdef CONSTRCALC
#ifdef CONSTR_Q0
      pots.lam2[ 0 ] = c_q2* ( dxyz*q2av( dens_p.rho , dens_n.rho , pots.v_constraint , nxyz , nprot , nneut ) - xconstr[0] ) + pots.lam[ 0 ];
#endif
      for( j=1; j< 4;j++){
	pots.lam2[ j ] = 3.e-1* ( rcm[j-1] - xconstr[j] ) + pots.lam[ j ];
      }
#endif

      if( gr_ip == 0 )

	{

	  sprintf( fn , "out%s.txt" , iso_label ) ;

	  file_out = fopen( fn , "w" ) ;

	  fprintf( file_out , "Iteration number %d, mu%s=%12.6f\n" , it + 1 , iso_label , * amu ) ;

	  fprintf( file_out , "N%s = %13.9f\n" , iso_label , xpart * ( double ) * npart ) ;

	  fprintf( file_out , "Nwf%s = %d\n" , iso_label , nwf ) ;

	  if( isospin == 1 )

	    deform( dens_p.rho , dens_n.rho , nxyz , &lattice_coords , dxyz , file_out ) ;

	}
     
      system_energy( &cc_edf , icoul , dens , &dens_p , &dens_n , isospin , pots.delta , dens->nstart , dens->nstop , ip , gr_ip , commw , gr_comm , k1d_x , k1d_y , k1d_z , nx , ny , nz , hbar2m , dxyz , & lattice_coords , & fftransf_vars , nprot , nneut, file_out ) ;

      
      if( gr_ip == 0 )

	{

	  fprintf(stdout,"\n Iteration number %d, mu%s=%12.6f\n" , it + 1 , iso_label , * amu ) ;

	  fprintf(stdout,"N%s = %13.9f\n" , iso_label , xpart * ( double ) * npart ) ;

	  fprintf(stdout,"Nwf%s = %d\n" , iso_label , nwf ) ;

	  fprintf( file_out , "      E_qp        Occ       log( | Eqp-Eqp_old | ) \n" ) ;

	  err = 0. ;

	  for( i = 2 * nxyz ; i < 2 * nxyz + nwf ; ++i )

	    {

	      fprintf( file_out , " %12.8f    %8.6f    %10.6f\n" , lam[ i ] , occ[ i - 2 * nxyz ] , log10( fabs( lam[ i ] - lam_old[ i - 2 * nxyz ] ) ) ) ;

	      err += pow( lam[ i ] - lam_old[ i - 2 * nxyz ] , 2. ) ;

	      *( lam_old + i - 2 * nxyz ) = *( lam + i ) ;

	    }

	  err = sqrt( err ) ;

	  fprintf(stdout,"err%s = %13.9f\n" , iso_label , err ) ;

	  fprintf( file_out , "err%s = %13.9f\n" , iso_label , err ) ;

	  printf(" \n" ) ;

	  fclose( file_out ) ;

	  sprintf( fn , "pots%s_%1d.cwr" , iso_label , it % 2 ) ;

	  i = write_pots( fn , pot_array , nx , ny , nz , dx , dy , dz , ishift ) ;

	}

    }

  if( icoul == 1 ) 

    {

      fftw_destroy_plan( fftransf_vars.plan_f ) ;

      fftw_destroy_plan( fftransf_vars.plan_b ) ;

      free( fftransf_vars.buff ) ;

    }
  
  free( ham ) ; free( lam_old ) ;

  free( k1d_x ) ; free( k1d_y ) ; free( k1d_z ) ;

  free( d1_x ) ; free( d1_y ) ; free( d1_z ) ;

  free( ex_array_n ) ; free( ex_array_p ) ;

  /* write the info for the TD code */

  if( ip == root_n )

    {

      nwf_n = nwf ;

      amu_n = * amu ;

    }

  MPI_Bcast( &nwf_n , 1 , MPI_INT , root_n , commw ) ;

  MPI_Bcast( &amu_n , 1 , MPI_DOUBLE , root_n , commw ) ;

  if( ip == 0 )

    {

      if( imin == 1 )

	{

	  free( f_old ) ; free( b_bra ) ; free( b_ket ) ; free( diag ) ; free( v_in_old ) ;

	}

      if( iprint_wf == 1 )

	nwf = (int) nprot ;

      else if( iprint_wf == 2 )

	nwf_n = nneut ;

      else if (iprint_wf == 3 ){

	nwf = (int) nprot ;

	nwf_n = (int) nneut;
      }

      sprintf( fn , "info.slda_solver" ) ;

      file_out = fopen( fn , "wb" ) ;

      fwrite( &nwf , sizeof( int ) , 1 , file_out );

      fwrite( &nwf_n , sizeof( int ) , 1 , file_out );

      fwrite( amu , sizeof( double ) , 1 , file_out );

      fwrite( &amu_n , sizeof( double ) , 1 , file_out );

      fwrite( &dx , sizeof( double ) , 1 , file_out );

      fwrite( &dy , sizeof( double ) , 1 , file_out );

      fwrite( &dz , sizeof( double ) , 1 , file_out );

      fwrite( &nx , sizeof( int ) , 1 , file_out );

      fwrite( &ny , sizeof( int ) , 1 , file_out );

      fwrite( &nz , sizeof( int ) , 1 , file_out );

      fwrite( &e_cut , sizeof( double ) , 1 , file_out );

#ifdef CONSTRCALC

      fwrite( pots.lam2 , sizeof( double ) , 4 , file_out );

#endif

      fclose( file_out ) ;

    }

  free( pot_array ) ; free( pot_array_old ) ; 

  double * qpe;

  assert(qpe = (double *)malloc(nwf * sizeof(double)));
 
  if ( iprint_wf > -1 )

    {
      sprintf( fn , "wf%s.cwr" , iso_label ) ;
      sprintf( fn_ , "%s.lstr" , fn ) ;
      //      printf( "iam[ %d ] fn[ %s ] fn_[ %s ]\n" , gr_ip , fn , fn_ ) ;
     
      if ( iprint_wf == 0 )

	{
	  if ( gr_ip == 0 ) printf( "entering MPI only write\n" ) ;
	  bc_wr_mpi( fn , gr_comm , p_proc , q_proc , i_p , i_q , nb , 2*nxyz , 4*nxyz , 1 ,  4*nxyz , z_eig );

	}

      else

	if( iprint_wf == 1 ){

	  if( isospin == 1 ) 

	    print_wf2( fn , gr_comm , lam , z_eig , gr_ip , nwf , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc , nx , ny , nz , e_cut , occ ,icub) ;

	  else{
	    if ( gr_ip == 0 ) printf( "entering MPI-only write\n" ) ;
	    bc_wr_mpi( fn , gr_comm , p_proc , q_proc , i_p , i_q , nb , 2*nxyz , 4*nxyz , 1 ,  4*nxyz , z_eig );
	  }

	}
      
	else if (iprint_wf == 2){

	  if( isospin == -1 ) 

	    print_wf2( fn , gr_comm , lam , z_eig , gr_ip , nwf , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc , nx , ny , nz , e_cut , occ ,icub) ;

	  else{
	    if ( gr_ip == 0 ) printf( "entering MPI-only write\n" ) ;
	    bc_wr_mpi( fn , gr_comm , p_proc , q_proc , i_p , i_q , nb , 2*nxyz , 4*nxyz , 1 ,  4*nxyz , z_eig );
	  }

	}

	else if (iprint_wf == 3){

	  print_wf2( fn , gr_comm , lam , z_eig , gr_ip , nwf , m_ip , n_iq , i_p , i_q , mb , nb , p_proc , q_proc , nx , ny , nz , e_cut , occ ,icub) ;
	  
	}

    }

      else

	if( ip == 0 )

	  printf( "wave functions not saved, iprint=%d\n" , iprint_wf ) ;




  // save positive qpe
  sprintf( fn , "qpe%s.cwr" , iso_label ) ;
  if( write_qpe( fn , qpe , gr_comm , gr_ip , nwf) != EXIT_SUCCESS )
    
    {
      
      fprintf(stdout, "Error, could not save qpe at the end of iterations. \n") ;
      
      MPI_Finalize() ;
      
      return( EXIT_FAILURE ) ;
      
    }
  
  free( z_eig ) ;

  free( lam ) ; free( occ ) ;

  destroy_mpi_groups( &group_comm , &gr_comm ) ;

  MPI_Finalize() ;

  return( EXIT_SUCCESS ) ;

 }

int minim_int( const int i1 , const int i2 )

{

  if( i1 < i2 )

    return( i1 ) ;

  else

    return( i2 ) ;

}

void print_help( const int ip , char ** argv )

{

  if( ip == 0 )

    {

      printf(" usage: %s --options \n" , argv[ 0 ] ) ;

      printf("       --nx Nx                  Nx **required** the number of points on the x direction\n" ) ;

      printf("       --ny Ny                  Ny **required** the number of points on the y direction\n" ) ;

      printf("       --nz Nz                  Nz **required** the number of points on the z direction\n" ) ;

      printf("       --nprot np or -Z np       number of protons **required** \n" ) ;

      printf("       --nneut nn or -N nn       number of neutrons **required** \n") ;

      printf("       --Lx Lx                  Lx: x-lattice length **required** \n" ) ;

      printf("       --Ly Ly                  Ly: y-lattice length **required** \n" ) ;

      printf("       --Lz Lz                  Lz: z-lattice length **required** \n" ) ; 

      printf("       --nopairing              pairing coupling set to 0; default with pairing; note that the wfs will still have u and v components \n" ) ;

      printf("       --broyden or -b          Broyden mixing will be performed\n" ) ;

      printf("       --alpha_mix alpha or -a alpha \n" ) ;

      printf("                                the mixing parameter; default alpha=0.25 \n" ) ;

      printf("       --nocoulomb              Coulomb interaction between protons set to zero; by default, Coulomb is included \n" ) ; 

      printf("       --niter n                 number of iterations\n") ;

      printf("       --irun n                  restart option: n = 0 start with guess for densities (default), n = 1 restart reading previously computed densities , n=2 read previously computed potentials \n" ) ;

      printf("       --iprint_wf n             printing options for the wave functions: n=-1 (default) no priting, n=0 all wfs printed, n=1 only Z proton wave functions saved, n=2 only N neutron wave functions saved \n") ;

      printf("       --hw hbaromega            option for a HO potential hbaromega=2 MeV by default\n" ) ;

      printf("       --ecut e_cut              the energy cut in MeV, default e_cut = 75 MeV  \n" ) ;

      printf("       --pproc p_proc            number of processors on the grid, default sqrt( total number of processors / 2 )  \n" ) ;

      printf("       --qproc q_proc            number of processors on the grid, default sqrt( total number of processors / 2 )  \n" ) ;

      printf("       --mb mb                   size of the block (mb=nb), default mb=40 \n" );

      printf("       --nb nb                   size of the block (mb=nb), default nb=40 \n" ) ;

      printf("       --hbo hw                  hw , default hw=2 MeV\n" ) ;

      printf("       --resc_dens               if set, the densities will be scalled to the correct number of particles; nonscalling is the default \n" ) ;

      printf("       --deformation n           if n = 0 (default) spherical densities , n = 2 axially symmetrix densities , n = triaxial densities \n" ) ;

      printf("                                 this is meaningful only if irun = 0, otherwize will be ignored \n" ) ;

    }

}

int parse_input_file(char * file_name)
{
    FILE *fp;
    fp=fopen(file_name, "r");
    if(fp==NULL)
        return 0;
    
    int i;
        
    char s[MAX_REC_LEN];
    char tag[MAX_REC_LEN];
    char ptag[MAX_REC_LEN];
    
    while(fgets(s, MAX_REC_LEN, fp) != NULL)
    {
        // Read first element of line
        tag[0]='#'; tag[1]='\0';
        sscanf (s,"%s %*s",tag);
        
        // Loop over known tags;
        if(strcmp (tag,"#") == 0)
            continue;
        else if (strcmp (tag,"nx") == 0)
            sscanf (s,"%s %d %*s",tag,&md.nx);
	
        else if (strcmp (tag,"ny") == 0)
            sscanf (s,"%s %d %*s",tag,&md.ny);
	
        else if (strcmp (tag,"nz") == 0)
            sscanf (s,"%s %d %*s",tag,&md.nz);
	
        else if (strcmp (tag,"dx") == 0)
            sscanf (s,"%s %lf %*s",tag,&md.dx);
	
        else if (strcmp (tag,"dy") == 0)
            sscanf (s,"%s %lf %*s",tag,&md.dy);
	
        else if (strcmp (tag,"dz") == 0)
            sscanf (s,"%s %lf %*s",tag,&md.dz);

	else if (strcmp (tag,"broyden") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.broyden);
	
	else if (strcmp (tag,"Z") == 0)
	  sscanf (s,"%s %lf %*s",tag,&md.nprot);
	
	else if (strcmp (tag,"N") == 0)
	  sscanf (s,"%s %lf %*s",tag,&md.nneut);

	else if (strcmp (tag,"niter") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.niter);
	
        else if (strcmp (tag,"iext") == 0)
            sscanf (s,"%s %d %*s",tag,&md.iext);
	
	else if (strcmp (tag,"force") == 0)
            sscanf (s,"%s %d %*s",tag,&md.force);
	
	else if (strcmp (tag,"pairing") == 0)
            sscanf (s,"%s %d %*s",tag,&md.pairing);

	else if (strcmp (tag,"print_wf") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.iprint_wf);

	else if (strcmp (tag,"alpha_mix") == 0)
            sscanf (s,"%s %lf %*s",tag,&md.alpha_mix);
	
	else if (strcmp (tag,"ecut") == 0)
	  sscanf (s,"%s %lf %*s",tag,&md.ecut);

	else if (strcmp (tag,"icub") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.icub);

	else if (strcmp (tag,"imass") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.imass);
	
	else if (strcmp (tag,"icm") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.icm);
	
	else if (strcmp (tag,"irun") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.irun);
	
	else if (strcmp (tag,"isymm") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.isymm);

	else if (strcmp (tag,"deform") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.deformation);

	else if (strcmp (tag,"q0") == 0)
	  sscanf (s,"%s %lf %*s",tag,&md.q0);

	else if (strcmp (tag,"v0") == 0)
	  sscanf (s,"%s %lf %*s",tag,&md.v0);

	else if (strcmp (tag,"z0") == 0)
	  sscanf (s,"%s %lf %*s",tag,&md.z0);

	else if (strcmp (tag,"wneck") == 0)
	  sscanf (s,"%s %lf %*s",tag,&md.wneck);

	else if (strcmp (tag,"rneck") == 0)
	  sscanf (s,"%s %lf %*s",tag,&md.rneck);
        
	else if (strcmp (tag,"resc_dens") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.resc_dens);
	
	else if (strcmp (tag,"p") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.p);
	
        else if (strcmp (tag,"q") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.q);
	
        else if (strcmp (tag,"mb") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.mb);
	
        else if (strcmp (tag,"nb") == 0)
	  sscanf (s,"%s %d %*s",tag,&md.nb);
	
    }
    
    fclose(fp);
    return 1;
}



int readcmd(int argc, char *argv[], int ip)
{
    static const char *optString = "h";
    int opt = 0;
    do
    {
        opt = getopt( argc, argv, optString );
        switch( opt )
        {
	case 'h':
	  print_help(ip, argv);
	  break;
          
	default:
	  break;
        }
    }
    while(opt != -1);
        
    if (optind < argc)
        return optind;
    else
        return -1;
}
