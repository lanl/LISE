
int create_mpi_groups( const MPI_Comm , MPI_Comm * , const int , int * , int * , MPI_Group * , const int , const int , int * , int * ) ;

void destroy_mpi_groups( MPI_Group * , MPI_Comm * ) ;

Wfs * allocate_phys_mem_wf( const int nwfip , const int nxyz ) ;

void allocate_phys_mem_dens( Densities * dens , const int nxyz ) ;

void allocate_phys_mem_pots( Potentials * pots , const int nxyz ) ;

void compute_densities( const int nxyz , const int nwfip , double complex ** wavf_in , double complex ** deriv_x , double complex ** deriv_y , double complex ** deriv_z , FFtransf_vars * fftrans , Densities * dens , Lattice_arrays * latt , const MPI_Comm comm , double ** , const int ip ) ;

void compute_diag_dens( double complex ** wavf_in , double * rho , double * sz , const int nwfip , const int nxyz , const MPI_Comm comm , double * , double * ) ;

void compute_nondiag_dens( double complex ** wavf_in , double * sx , double * sy , const int nwfip , const int nxyz , const MPI_Comm comm , double * , double * ) ;

void compute_ctau_dens( double complex ** wavf_in , double complex ** deriv_x , double complex ** deriv_y , double complex ** deriv_z , double * tau , double * jx , double * jy , double * jz , const int nwfip , const int nxyz , const MPI_Comm comm , double * , double * , double * , double * ) ;

void make_coordinates( const int nxyz , const int nx , const int ny , const int nz , const double dx , const double dy , const double dz , Lattice_arrays * lattice_vars ) ;

void compute_curlj_dens( double complex ** deriv_x , double complex ** deriv_y , double complex ** deriv_z , double * cjx , double * cjy , double * cjz , const int nwfip , const int nxyz , const MPI_Comm comm , double * , double * , double * ) ;

void compute_so_dens( double complex ** deriv_x , double complex ** deriv_y , double complex ** deriv_z , double * divjj , const int nwfip , const int nxyz , const MPI_Comm comm , double * ) ;

int dens_func_params( const int iforce , const int ihfb , const int isospin , Couplings * cc_edf , const int ip, int icub) ;

void gradient( double complex * f , double complex * fx , double complex * fy , double complex * fz , FFtransf_vars * fftrans , Lattice_arrays * latt , const int nxyz , double complex * buff ) ;

void gradient_real( double * f , double * fx , double * fy , double * fz , FFtransf_vars * fftrans , Lattice_arrays * latt , const int nxyz ) ;

void laplacean( double * f , double * lap_f , const int nxyz , FFtransf_vars * fftrans , Lattice_arrays * latt ) ;

void coul_pot( double * vcoul , double * rho , double * work1 , double * work2 , Lattice_arrays * latt_coords , const int nxyz , FFtransf_vars * fftransf_vars , const double dxyz ) ;

double center_dist( double * rho , const int n , Lattice_arrays * latt_coords , double * xc , double * yc , double * zc ) ;

void get_u_re( Densities * dens_p , Densities * dens_n , Potentials * pots , Couplings * cc_edf , const double hbar2m , const int nxyz , const int isospin , Lattice_arrays * latt , FFtransf_vars * fftrans , const double dxyz ) ;

void update_potentials( const int isospin , Potentials * pots , Densities * dens_p , Densities * dens_n , Densities * dens , Couplings * cc_edf , const double e_cut , const int nxyz , const double hbar2m , Lattice_arrays * latt , FFtransf_vars * fftrans , const double dxyz , double * buff ,int icub) ;

int read_wf( double complex ** wf , const int nxyz , const int nwf , const int ip , const int np , const MPI_Comm comm , char * fn , double dxyz , int *wavef_index) ;

void read_input_solver( char * dir_in , int * nx , int * ny , int * nz , int * nwf_p , int * nwf_n , double * amu_p , double * amu_n , double * dx , double * dy , double * dz , double * e_cut , double * cc_qzz , const int ip , const MPI_Comm comm ) ;

void curl( double * vx , double * vy , double * vz , double * cvx , double * cvy , double * cvz , const int nxyz , FFtransf_vars * fftrans , Lattice_arrays * latt ) ;

void initialize_wfs( Wfs * wavef , const int nwfip , const int nxyz ) ;

void initialize_wfs_swap( Wfs * wavef , const int nwfip , const int nxyz ) ;

void adam_bashforth_pm( const int mm_time , const int n0_time , const int n1_time , const int m0_time , const int m1_time , const int m2_time , const int m3_time , const int nxyz , const int nwfip , Wfs * wavef , double dt_step  ) ;

void adam_bashfort_cy( const int nn_time , const int n0_time , const int n1_time , const int m0_time , const int m1_time , const int m2_time , const int m3_time , const int nxyz , const int nwfip , Wfs * wavef , double dt_step , Potentials * pots , FFtransf_vars * fftrans , Lattice_arrays * latt , const double dxyz , const double hbarc ) ;

void adams_bashforth_dfdt( const int mm_time , const int nn_time , const int nxyz , const int nwfip , Wfs * wavef , double dt_step , Potentials * pots , FFtransf_vars * fftrans , Lattice_arrays * latt , const double dxyz , const double hbarc , double * qp_energy ) ;

void df_over_dt( double complex * wavf_in , double complex * df_dt , double complex * deriv_x , double complex * deriv_y , double complex * deriv_z , const int nxyz , Potentials * pots , FFtransf_vars * fftrans , Lattice_arrays * latt , double complex * buff , const double dxyz , const double hbarc , double * qp_energy ) ;

void df_over_dt_u( double complex * wavf_in , double complex * df_dt , double complex * deriv_x , double complex * deriv_y , double complex * deriv_z , const int nxyz , Potentials * pots , FFtransf_vars * fftrans , Lattice_arrays * latt , double complex * buff ) ;

void df_over_dt_v( double complex * wavf_in , double complex * df_dt , double complex * deriv_x , double complex * deriv_y , double complex * deriv_z , const int nxyz , Potentials * pots , FFtransf_vars * fftrans , Lattice_arrays * latt , double complex * buff ) ;

void external_pot( const int iext , const int n , const int n_part , const double hbo , double * v_ext , double complex * delta_ext , const double hbar2m , Lattice_arrays * lattice_coords , const double z0) ;

void external_so_m( double * v_ext , double * mass_eff , double * wx , double * wy , double * wz , const double hbar2m , Lattice_arrays * latt , FFtransf_vars * fftrans , const int nxyz ) ;

void free_phys_mem_dens( Densities * dens ) ;

void free_phys_mem_pots( Potentials * pots ) ;

void free_phys_mem_wf( const int nwfip , Wfs * wavef ) ;

void bcast_potentials( Potentials * pots , const int nxyz , const MPI_Comm comm ) ;

void exchange_nuclear_densities( Densities * dens_p , Densities * dens_n , const int nxyz , const int ip , const int root_p , const int root_n , const MPI_Comm commw , MPI_Status * status ) ;

void laplacean_complex( double complex * f , double complex * lap_f , const int nxyz , FFtransf_vars * fftrans , Lattice_arrays * latt ) ;

void laplacean( double * f , double * lap_f , const int nxyz , FFtransf_vars * fftrans , Lattice_arrays * latt ) ;

void coul_pot( double * vcoul , double * rho , double * work1 , double * work2 , Lattice_arrays * latt_coords , const int nxyz , FFtransf_vars * fftransf_vars , const double dxyz ) ;

Wfs * allocate_phys_mem_wf2( const int nwfip , const int nxyz ) ;

int read_potentials( Potentials * pots , const int nx , const int ny , const int nz , const double dx , const double dy , const double dz , char * fn ) ;

//void mix_potentials( Potentials * pots0 , Potentials * pots , const int nxyz , const double alpha ) ;

void ext_field( Potentials * pots , int ichoice , Lattice_arrays * latt , const int nxyz ) ;

void  match_lattices( Lattice_arrays *latt , Lattice_arrays * latt3 , const int nx , const int ny , const int nz , const int nx3 , const int ny3 , const int nz3 , FFtransf_vars * fftrans , const double Lc ) ;

void coul_pot3( double * vcoul , double * rho , double * work1 , double * work2 , Lattice_arrays * latt_coords , const int nxyz , FFtransf_vars * fftransf_vars , const double dxyz ) ;

void init_cm( double ** y , double ** y_predictor , double * y_modifier , double ** y_corrector , double ** y_t_der , double * forces ) ;

void adam_bashforth_pm_coord( const int nn_time , const int n0_time , const int n1_time , const int m0_time , const int m1_time , const int m2_time , const int m3_time , double ** y , double ** y_predictor , double ** y_corrector , double * y_modifier , double ** y_t_der , double * forces , double dt_step  ) ;

void adam_bashfort_cy_coord( const int nn_time , const int n0_time , const int n1_time , const int m0_time , const int m1_time , const int m2_time , const int m3_time , double ** y , double ** y_predictor , double ** y_corrector , double ** y_t_der , double * forces , double dt_step  ) ;

void adams_bashforth_dfdt_coord( const int mm_time , const int nn_time , double ** y_t_der , double * forces ) ;

void absorbing_potential( const double lx , const double ly , const double lz , Lattice_arrays * latt , double * w_abs , const int nxyz );

void strength_abs( const double time , double complex * w0 );

void computeForces(const int Z_proj , const double time , const double v , double * cm , double * rho , double * rho_n , Lattice_arrays * latt , const int nxyz , double ** ephi , FFtransf_vars * fftrans , double * forces , const double m_tot , const double dxyz ) ;

int f_copn( char * fn );

void f_ccls( int * fd );

void f_crm( char * fn );

void f_cwr( const int fd , void * fbf , const int fsz , const int nobj ) ;

void f_crd( int fd , void * fbf , const int fsz , const int nobj ) ;

int read_wf_MPI( double complex ** wf , const int nxyz , const int nwf , const int ip , const int np , const MPI_Comm comm , char * fn , double dxyz , int *wavef_index, int * wf_tbl);
