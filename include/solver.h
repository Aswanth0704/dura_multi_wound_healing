/*
	Pre and Post processing functions
	Solver functions
	Struct and classes for the problem definition
*/

#ifndef solver_h
#define solver_h

#include <omp.h>
#include <vector>
#include <map>
#include <Eigen/Dense> // most of the vector functions I will need inside of an element
using namespace Eigen;

// Structure for the problem
struct tissue{

	// connectivity (topology)
	int n_node;
	int n_vol_elem;
    int n_surf_elem;
	int n_IP;
	std::vector<std::vector<int> > vol_elem_connectivity;
    std::vector<std::vector<int> > surf_elem_connectivity;
	std::vector<int> boundaryNodes;
    std::vector<int> boundary_flag;
    std::vector<int> surface_boundary_flag;

	// reference geometry
	//
	// nodal values
	std::vector<Vector3d> node_X;
	std::vector<double> node_rho_0;
	std::vector<double> node_c_0;
	//
	// integration point values
	// order by element then by integration point of the element
	std::vector<double> ip_phif_0;
	std::vector<Vector3d> ip_a0_0;
    std::vector<Vector3d> ip_s0_0;
    std::vector<Vector3d> ip_n0_0;
	std::vector<double> ip_kappa_0;
	std::vector<Vector3d> ip_lamdaP_0;
	
	// deformed geometry
	//
	// nodal values
	std::vector<Vector3d> node_x;
	std::vector<double> node_rho;
	std::vector<double> node_c;
	//
	// integration point values
	std::vector<double> ip_phif;
	std::vector<Vector3d> ip_a0;
    std::vector<Vector3d> ip_s0;
    std::vector<Vector3d> ip_n0;
	std::vector<double> ip_kappa;
	std::vector<Vector3d> ip_lamdaP;
	std::vector<Vector3d> ip_lamdaE;
    std::vector<Matrix3d> ip_strain;
    std::vector<Matrix3d> ip_stress;

	// boundary conditions
	//
	// essential boundary conditions for displacement
	std::map<int,double>  eBC_x;
	// essential boundary conditions for concentrations
	std::map<int,double>  eBC_rho;
	std::map<int,double>  eBC_c;
	//
	// traction boundary conditions for displacements
	std::map<int,double> nBC_x;
	// traction boundary conditions for concentrations
	std::map<int,double> nBC_rho;
	std::map<int,double> nBC_c;

	// degree of freedom maps
	//
	// displacements
	std::vector< int > dof_fwd_map_x;
	//
	// concentrations
	std::vector< int > dof_fwd_map_rho;
	std::vector< int > dof_fwd_map_c;
	
	// all dof inverse map
	std::vector< std::vector<int> > dof_inv_map;
	
	// material parameters
	std::vector<double> global_parameters;
	std::vector<double> local_parameters;
	
	// internal element constant (jacobians at IP)
	std::vector<std::vector<Matrix3d> > elem_jac_IP;
    std::vector<std::vector<double> > elem_jac_IP_surface;
	
	// parameters for the simulation
	int n_dof;
	double time_final;
	double time_step;
	double time;
	double tol;
	int max_iter;
	
};


//-------------------------------------------------//
// PRE PROCESS
//-------------------------------------------------//

//----------------------------//
// FILL DOF
//----------------------------//
//
// now I have the mesh and 
// =>> Somehow filled in the essential boundary conditions
// so I create the dof maps.  
void fillDOFmap(tissue &myTissue);
void fillDOFmapSurface(tissue &myTissue);

//----------------------------//
// EVAL JAC
//----------------------------//
//
// eval the jacobians that I use later on in the element subroutines
void evalElemJacobians(tissue &myTissue);
void evalElemJacobiansSurface(tissue &myTissue);

//-------------------------------------------------//
// SOLVER
//-------------------------------------------------//

//----------------------------//
// SPARSE SOLVER
//----------------------------//
//
void sparseWoundSolver(tissue &myTissue, const std::string& filename, int save_freq,const std::vector<int> &save_node,const std::vector<int> &save_ip);
void sparseLoadSolver(tissue &myTissue, const std::string& filename, int save_freq,const std::vector<int> &save_node,const std::vector<int> &save_ip);
// dense solver in previous version of wound.cpp
// void denseWoundSolver(tissue &myTissue, std::string filename, int save_freq);

#endif