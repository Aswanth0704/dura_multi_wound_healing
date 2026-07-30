// last date modified: Sept, 17, 2025

/*
Used for finding the forces corresponding to given z-displacements.
Read myTissue from file.
Then apply the displacements at selected nodes in a small cicular region.
solve the mechanics --> Residuals --> Forces
Apply some parameter finding algorithm to find the mechanical properties.
*/
#include <limits> 
#include <omp.h>
#include "file_io.h"
#include "file_io_abaqus.h"
#include "wound.h"
#include "solver.h"
#include "myMeshGenerator.h"
#include "element_functions.h"
#include "local_solver.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <ctime>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
using namespace Eigen;
// MKL is included through the CMake file
// #define EIGEN_USE_MKL_ALL

double frand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

// sum z-reactions on a displacement controlled-patch.
// re-assemble internal element residuals at the converged state and 
// accumulate the z entries for those constrained patch nodes.

static double computePatchReactionZ(
	tissue &myTissue,
	const std::vector<int> &patchNodeIds){

	// lookup for the members under load
	std::vector<char> is_in_patch(myTissue.n_node, 0);
	for (int nid: patchNodeIds) {
		if (nid>=0 && nid<myTissue.n_node) {

			is_in_patch[nid] = 1;

		}
	}

	const int n_dof = myTissue.n_dof;
    const int elem_size = myTissue.vol_elem_connectivity[0].size();
    const int surf_elem_size = myTissue.surf_elem_connectivity[0].size();
    const int n_coord = myTissue.node_X[0].size();

	// IP rule:  
	std::vector<Vector4d> IP;
    if(elem_size == 8 || elem_size == 20){
        // linear hexahedron
        IP = LineQuadriIP();
    } else if(elem_size == 27){
        // quadratic hexahedron
        IP = LineQuadriIPQuadratic();
    } else if(elem_size==4){
        // linear tetrahedron
        IP = LineQuadriIPTet();
    } else if(elem_size==10){
        // quadratic tetrahedron
        IP = LineQuadriIPTetQuadratic();
    }

	const int IP_size = (int)IP.size();
	double sum_reaction_z = 0.0;

	// loop over elements: form internal Re_x (mechanics-only) and accumulate.
	#pragma omp parallel for reduction(+:sum_reaction_z)
	for (int ei= 0; ei< myTissue.n_vol_elem; ei++){

		std::vector<int> elem_ei = myTissue.vol_elem_connectivity[ei];

		// kinematics  (current & reference)
		std::vector<Eigen::Vector3d> node_x_ni; node_x_ni.clear();
		std::vector<Eigen::Vector3d> node_X_ni; node_X_ni.clear();

		// concentration and cells for previous and current time
		std::vector<double> node_rho_0_ni; node_rho_0_ni.clear();
		std::vector<double> node_c_0_ni; node_c_0_ni.clear();
		std::vector<double> node_rho_ni; node_rho_ni.clear();
		std::vector<double> node_c_ni; node_c_ni.clear();


		// values of the structural variables at the IP
		std::vector<double> ip_phif_0_pi; ip_phif_0_pi.clear();
		std::vector<Vector3d> ip_a0_0_pi; ip_a0_0_pi.clear();
		std::vector<Vector3d> ip_s0_0_pi; ip_s0_0_pi.clear();
		std::vector<Vector3d> ip_n0_0_pi; ip_n0_0_pi.clear();
		std::vector<double> ip_kappa_0_pi; ip_kappa_0_pi.clear();
		std::vector<Vector3d> ip_lamdaP_0_pi; ip_lamdaP_0_pi.clear();
		std::vector<double> ip_phif_pi; ip_phif_pi.clear();
		std::vector<Vector3d> ip_a0_pi; ip_a0_pi.clear();
		std::vector<Vector3d> ip_s0_pi; ip_s0_pi.clear();
		std::vector<Vector3d> ip_n0_pi; ip_n0_pi.clear();
		std::vector<double> ip_kappa_pi; ip_kappa_pi.clear();
		std::vector<Vector3d> ip_lamdaP_pi; ip_lamdaP_pi.clear();
		std::vector<Matrix3d> ip_strain(IP_size,Matrix3d::Identity(3,3));
		std::vector<Matrix3d> ip_stress(IP_size,Matrix3d::Zero(3,3));

		for(int ni=0;ni<elem_size;ni++) {
			// nodal reference and deformed positions
			node_X_ni.push_back(myTissue.node_X[elem_ei[ni]]);
			node_x_ni.push_back(myTissue.node_x[elem_ei[ni]]);
			// cells and chemical
			node_rho_0_ni.push_back(myTissue.node_rho_0[elem_ei[ni]]);
			node_c_0_ni.push_back(myTissue.node_c_0[elem_ei[ni]]);
			node_rho_ni.push_back(myTissue.node_rho[elem_ei[ni]]);
			node_c_ni.push_back(myTissue.node_c[elem_ei[ni]]);
		}

		for(int ipi=0;ipi<IP_size;ipi++){
			// structural variables
			ip_phif_0_pi.push_back(myTissue.ip_phif_0[ei*IP_size+ipi]);
			ip_phif_pi.push_back(myTissue.ip_phif[ei*IP_size+ipi]);
			ip_a0_0_pi.push_back(myTissue.ip_a0_0[ei*IP_size+ipi]);
			ip_a0_pi.push_back(myTissue.ip_a0[ei*IP_size+ipi]);
			ip_s0_0_pi.push_back(myTissue.ip_s0_0[ei*IP_size+ipi]);
			ip_s0_pi.push_back(myTissue.ip_s0[ei*IP_size+ipi]);
			ip_n0_0_pi.push_back(myTissue.ip_n0_0[ei*IP_size+ipi]);
			ip_n0_pi.push_back(myTissue.ip_n0[ei*IP_size+ipi]);
			ip_kappa_0_pi.push_back(myTissue.ip_kappa_0[ei*IP_size+ipi]);
			ip_kappa_pi.push_back(myTissue.ip_kappa[ei*IP_size+ipi]);
			ip_lamdaP_0_pi.push_back(myTissue.ip_lamdaP_0[ei*IP_size+ipi]);
			ip_lamdaP_pi.push_back(myTissue.ip_lamdaP[ei*IP_size+ipi]);
		}

		// element residual/tangent (mechanics only)
		Eigen::VectorXd Re_x(n_coord*elem_size); Re_x.setZero();
		Eigen::MatrixXd Ke_x_x(n_coord*elem_size, n_coord*elem_size); Ke_x_x.setZero();

		// subroutine to evaluate the element:
		evalWoundMechanics(
			/* dt, time, time_final (not used here): */ 0.0, 0.0, 0.0,
			myTissue.elem_jac_IP[ei], myTissue.global_parameters,
			ip_strain, ip_stress, node_rho_0_ni, node_c_0_ni,
			ip_phif_0_pi, ip_a0_0_pi, ip_s0_0_pi, ip_n0_0_pi, ip_kappa_0_pi, ip_lamdaP_0_pi,
			node_rho_ni, node_c_ni,
			ip_phif_pi, ip_a0_pi, ip_s0_pi, ip_n0_pi, ip_kappa_pi, ip_lamdaP_pi,
			node_x_ni, Re_x, Ke_x_x);

		// sum z-components for nodes in the patch that are displacement-controlled in z.

		for (int ni = 0; ni< elem_size; ni++) {

			int nid = elem_ei[ni];
			if (is_in_patch[nid]){
				// z-index in the Re_x 
				int iz = ni*n_coord + 2; 
				sum_reaction_z += Re_x(iz);
			}

		}

	}

	return sum_reaction_z;

}

int main(int argc, char *argv[])
{
    Eigen::initParallel();
	std::cout<<"\nRunning full domain simulations with " << Eigen::nbThreads( ) << " threads.\n";
	srand (time(NULL));
	
	//---------------------------------//
	// GLOBAL PARAMETERS
	//
	// for normalization
	double rho_phys = 1000*55.05126; // [cells/mm^3]
	double c_max = 1.0e-4; // [g/mm3] from tgf beta review, 5e-5g/mm3 was good for tissues
	double k0 = 0.00202026; //0.00202026; neo hookean for skin, used previously, in MPa
	double kf = 0.1833; // 0.1833 // stiffness of collagen in MPa, from previous paper
	double k2 = 1.54; //1.54; // nonlinear exponential coefficient, non-dimensional
	double t_rho = 1e-1*(1.28571E-5/55.05125); // 0.0045 force of fibroblasts in MPa, this is per cell. so, in an average sense this is the production by the natural density
	double t_rho_c = 1e-1*(1.28571E-5*3.28571/55.05125); // 0.045 force of myofibroblasts enhanced by chemical, I'm assuming normalized chemical, otherwise I'd have to add a normalizing constant
	double K_t = 0.2; // Saturation of mechanical force by collagen
	double K_t_c = c_max/10.; // saturation of chemical on force. this can be calculated from steady state
	double D_rhorho = 0.0833; // 0.0833 diffusion of cells in [mm^2/hour], This is calculated later as a function of phi. so  
	double D_rhoc = 0.0; // diffusion of chemotactic gradient, an order of magnitude greater than random walk [mm^2/hour], not normalized
	double D_cc = 0.01208; // 0.15 diffusion of chemical TGF, not normalized.
	double p_rho = 0.04958333/55.05126; // in 1/hour production of fibroblasts naturally, proliferation rate, not normalized, based on data of doubling rate from commercial use
	double p_rho_c = 0.015314; // production enhanced by the chem, if the chemical is normalized, then suggest two fold,
	double p_rho_theta = p_rho/2; // enhanced production by theta
	double K_rho_c = c_max/10.; // saturation of cell proliferation by chemical, this one is definitely not crucial, just has to be small enough <cmax
    double K_rho_rho = 10000*55.05126; // saturation of cell by cell, from steady state
    double d_rho = p_rho*(1-rho_phys/K_rho_rho); // percent of cells die per day, 0.1*p_rho 10% in the original, now much less, determined to keep cells in dermis constant
	double vartheta_e = 2.; // physiological state of area stretch
	double gamma_theta = 5.; // sensitivity of heaviside function
	double p_c_rho = 90.0e-16/rho_phys*10;// production of c by cells in g/cells/h
	double p_c_thetaE = 300.0e-16/rho_phys*10; // coupling of elastic and chemical, three fold
	double K_c_c = 1.;// saturation of chem by chem, from steady state
	double d_c = 0.01/2; // 0.01 decay of chemical in 1/hours
	double bx = 0; // body force
    double by = 0; //-0.001; // body force
    double bz = 0; // body force
	//---------------------------------//
	std::vector<double> global_parameters = {k0,kf,k2,t_rho,t_rho_c,K_t,K_t_c,D_rhorho,D_rhoc,D_cc,p_rho,p_rho_c,p_rho_theta,K_rho_c,K_rho_rho,d_rho,vartheta_e,gamma_theta,p_c_rho,p_c_thetaE,K_c_c,d_c,bx,by,bz};

	//---------------------------------//
	// LOCAL PARAMETERS
	//
	// collagen fraction
	double p_phi = 0.002/rho_phys; // production by fibroblasts, natural rate in percent/hour, 5% per day
	double p_phi_c = p_phi; // production up-regulation, weighted by C and rho
	double p_phi_theta = p_phi; // mechanosensing upregulation. no need to normalize by Hmax since Hmax = 1
	double K_phi_c = 0.0001; // saturation of C effect on deposition.
	double d_phi = 0.000970; // rate of degradation, in the order of the wound process, 100 percent in one year for wound, means 0.000116 effective per hour means degradation = 0.002 - 0.000116
	double d_phi_rho_c = 0.5*0.000970/rho_phys/c_max/10; //0.000194; // degradation coupled to chemical and cell density to maintain phi equilibrium
	double K_phi_rho = rho_phys*p_phi/d_phi - 1; // >0 saturation of collagen fraction itself, from steady state
	//
	//
	// fiber alignment
	double tau_omega = 10./(K_phi_rho+1); // time constant for angular reorientation, think 100 percent in one year
	//
	// dispersion parameter
	double tau_kappa = 1./(K_phi_rho+1); // time constant, on the order of a year
	double gamma_kappa = 5.; // exponent of the principal stretch ratio
	// 
	// permanent contracture/growth
	double tau_lamdaP_a = 0.05; // 1.0 time constant for direction a, on the order of a year
	double tau_lamdaP_s = 0.05; // 1.0 time constant for direction s, on the order of a year
    double tau_lamdaP_n = 0.05; // 1.0 time constant for direction s, on the order of a year

    // solution parameters
    double tol_local = 1e-8; // local tolerance (also try 1e-5)
    double time_step_ratio = 100; // time step ratio between local and global (explicit)
    double max_iter = 100; // max local iter (implicit)
    //---------------------------------//
	std::vector<double> local_parameters = {p_phi,p_phi_c,p_phi_theta,K_phi_c,K_phi_rho,d_phi,d_phi_rho_c,tau_omega,tau_kappa,gamma_kappa,tau_lamdaP_a,tau_lamdaP_s,tau_lamdaP_n,vartheta_e,gamma_theta,tol_local,time_step_ratio,max_iter};

	
	
	//---------------------------------//
	// values for the wound
	double rho_wound = 0.0; // originally 1000 // [cells/mm^3]
	double c_wound = c_max;
	double phif0_wound = 1e-2; //0.01;
	double kappa0_wound = 1./3;
    double a0x = frand(-1,1.);
    double a0y = frand(-1,1.);
    double a0z = 0.;
    Vector3d a0_wound; a0_wound << a0x, a0y, a0z;
    a0_wound = a0_wound/sqrt(a0_wound.dot(a0_wound));
	Vector3d lamda0_wound;lamda0_wound << 1.,1.,1.;
	//---------------------------------//
	
	
	//---------------------------------//
	// values for the healthy
	double rho_healthy = rho_phys; // [cells/mm^3]
	double c_healthy = 1e-4;
	double phif0_healthy = 1.;
	double kappa0_healthy = 1./3;
	Vector3d a0_healthy;a0_healthy<<1.,0.,0.;
	Vector3d lamda0_healthy;lamda0_healthy<<1.,1.,1.;
	//---------------------------------//


    //---------------------------------//
    Matrix3d Rot90;Rot90 << 0.,-1.,0., 1.,0.,0., 0.,0.,1.;
    Vector3d s0_wound = Rot90*a0_wound;
    Vector3d s0_healthy= Rot90*a0_healthy;
    Vector3d n0_wound = s0_wound.cross(a0_wound);
    if(n0_wound(2)<0){
        n0_wound = a0_wound.cross(s0_wound);
    }
    Vector3d n0_healthy= s0_healthy.cross(a0_healthy);
    if(n0_healthy(2)<0){
        n0_healthy = a0_healthy.cross(s0_healthy);
    }
    //---------------------------------//

	
	//---------------------------------//
	// create mesh (only nodes and elements)
	std::cout<<"Going to create the mesh\n";
	// read the abaqus input file and return the Hexmesh
	std::string mesh_filename = "square_mechanics_patchSize_01_C3D8_meshSize_003_patch_global_02_full.inp";
	HexMesh myMesh;
	readAbaqusInputLikeCOMSOL(mesh_filename, myMesh);

	// ---- Bounding box ----
	double xmin =  std::numeric_limits<double>::infinity();
	double ymin =  std::numeric_limits<double>::infinity();
	double zmin =  std::numeric_limits<double>::infinity();
	double xmax = -std::numeric_limits<double>::infinity();
	double ymax = -std::numeric_limits<double>::infinity();
	double zmax = -std::numeric_limits<double>::infinity();

	for (const auto& p : myMesh.nodes) {
		xmin = std::min(xmin, p(0)); xmax = std::max(xmax, p(0));
		ymin = std::min(ymin, p(1)); ymax = std::max(ymax, p(1));
		zmin = std::min(zmin, p(2)); zmax = std::max(zmax, p(2));
	}

    std::cout<<"Created the mesh with "<<myMesh.n_nodes<<" nodes and "<<myMesh.boundary_flag.size()<<" boundaries and "<<myMesh.n_elements<<" elements\n";
    std::cout<<"Created the surface mesh with "<<myMesh.n_nodes<<" nodes and "<<myMesh.surface_boundary_flag.size()<<" surface boundaries and "<<myMesh.n_surf_elements<<" surface elements\n";
	
	// std::cout<<"nodes\n";
	// for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
	// 	std::cout<<myMesh.nodes[nodei](0)<<","<<myMesh.nodes[nodei](1)<<","<<myMesh.nodes[nodei](2)<<"\n";
	// }
	// // prints nodes associated with each element
	// std::cout<<"elements\n";
    // for(int elemi=0;elemi<myMesh.n_elements;elemi++){
    //     for(int nodei=0;nodei<myMesh.elements[elemi].size();nodei++){
    //         std::cout<<myMesh.elements[elemi][nodei]<<" ";
    //     }
    //     std::cout<<"\n";
    // }
	// prints boundary
	std::cout<<"boundary\n";
	std::cout<<myMesh.boundary_flag.size()<<"\n";
	

	// create the other fields needed in the tissue struct.
	int elem_size = myMesh.elements[0].size();

	// integration points
    std::vector<Vector4d> IP;
    if(elem_size == 8 || elem_size == 20){
        // linear hexahedron
        IP = LineQuadriIP();
    }
    else if(elem_size == 27){
        // quadratic hexahedron
        IP = LineQuadriIPQuadratic();
    }
    else if(elem_size==4){
        // linear tetrahedron
        IP = LineQuadriIPTet();
    }
    else if(elem_size==10){
        // quadratic tetrahedron
        IP = LineQuadriIPTetQuadratic();
    }
    int IP_size = IP.size();
	//
	// global fields rho and c initial conditions 
	std::vector<double> node_rho0(myMesh.n_nodes,rho_healthy);
	std::vector<double> node_c0 (myMesh.n_nodes,c_healthy);
	//
	// values at the (8) integration points
	std::vector<double> ip_phi0(myMesh.n_elements*IP_size,phif0_healthy);
	std::vector<Vector3d> ip_a00(myMesh.n_elements*IP_size,a0_healthy);
    std::vector<Vector3d> ip_s00(myMesh.n_elements*IP_size,s0_healthy);
    std::vector<Vector3d> ip_n00(myMesh.n_elements*IP_size,n0_healthy);
	std::vector<double> ip_kappa0(myMesh.n_elements*IP_size,kappa0_healthy);
	std::vector<Vector3d> ip_lamda0(myMesh.n_elements*IP_size,lamda0_healthy);
	//
    // double tol_boundary = 1e-5; // I need to adjust this value based on the actual dimension of the raius of point load

	// -- Problem geometry --
	const double top_z = zmax; // should be ~0.1
	const double bottom_z = zmin; // should be ~ 0.0
	std::cout<<"zmin"<<zmin<<std::endl;
	// define load domain
	double x_center = 0.0;
	double y_center = 0.0;
	double rad_load = 0.05;

	const double tol_face = 1e-6; //* std::max({1.0, std::abs(xmax-xmin), std::abs(ymax-ymin), std::abs(top_z-bottom_z)});
	const double tol_patch = 1e-6; // for radius check safety
	
	// boundary conditions and definition of the wound
	std::map<int,double> eBC_x;
	std::map<int,double> eBC_rho;
	std::map<int,double> eBC_c;
	
	// neumann boundary conditions. // here I am not using any Neumann boundary conditions
	std::map<int,double> nBC_x; /// This is a map from the node to the condition (three times as long for x)
	std::map<int,double> nBC_rho; /// Could also use the map from face numbering
	std::map<int,double> nBC_c; /// Make some kind of flag so that if we are running BC we don't update cells, etc

	// initialize my tissue. // I need to update these values with the values of the healed wound.
	tissue myTissue;

	// connectivity
	myTissue.vol_elem_connectivity = myMesh.elements;
    myTissue.surf_elem_connectivity = myMesh.surface_elements;


	// parameters
	myTissue.global_parameters = global_parameters;
	myTissue.local_parameters = local_parameters;
	myTissue.boundary_flag = myMesh.boundary_flag;
    myTissue.surface_boundary_flag = myMesh.surface_boundary_flag;

    myTissue.n_node = myMesh.n_nodes;
    myTissue.n_vol_elem = myMesh.n_elements;
    myTissue.n_surf_elem = myMesh.n_surf_elements;
	myTissue.n_IP = IP_size*myMesh.n_elements;

	myTissue.node_X = myMesh.nodes;
    myTissue.node_x = myMesh.nodes;
	myTissue.node_rho_0 = node_rho0;
	myTissue.node_rho = node_rho0;
	myTissue.node_c_0 = node_c0;
	myTissue.node_c = node_c0;
	myTissue.ip_phif_0 = ip_phi0;	
	myTissue.ip_phif = ip_phi0;	
	myTissue.ip_a0_0 = ip_a00;
	myTissue.ip_a0 = ip_a00;
    myTissue.ip_s0_0 = ip_s00;
    myTissue.ip_s0 = ip_s00;
    myTissue.ip_n0_0 = ip_n00;
    myTissue.ip_n0 = ip_n00;
    myTissue.ip_kappa_0 = ip_kappa0;
	myTissue.ip_kappa = ip_kappa0;	
	myTissue.ip_lamdaP_0 = ip_lamda0;
	myTissue.ip_lamdaP = ip_lamda0;
    myTissue.ip_lamdaE = ip_lamda0;
    std::vector<Matrix3d> ip_strain(myMesh.n_elements*IP_size,Matrix3d::Identity(3,3));
    std::vector<Matrix3d> ip_stress(myMesh.n_elements*IP_size,Matrix3d::Zero(3,3));
	myTissue.ip_strain = ip_strain;
    myTissue.ip_stress = ip_stress; 

	// actually I might need elements. but save this anyway
    std::vector<int> elemsZPush;
    std::vector<double> elemsZPushArea; // area of the elements that are pushed

	// let me first fix the boundary conditions for rho and c.
	// Here all the nodes will be fixed with rho and c values

	double z_displ = 0.005;              // positive magnitude (we apply in -z)
	std::vector<int> nodesZPush;         // nodes on top circular patch

	for (int nodei = 0; nodei < myTissue.n_node; nodei++) {

		
		// insert the boundary condition for rho and c
		eBC_rho.insert( std::pair<int,double>(nodei, myTissue.node_rho[nodei])); // if the compiler complains, need to adjust few values.
		eBC_c.insert( std::pair<int,double>(nodei,myTissue.node_c[nodei])); 

		// --- Coordinates ---
		const double x_coord = myTissue.node_x[nodei](0);
		const double y_coord = myTissue.node_x[nodei](1);
		const double z_coord = myTissue.node_x[nodei](2);

		// ---------------------------------------------------------
		// 1) Fix sides + bottom surface:
		//    - sides: x = xmin or xmax, OR y = ymin or ymax
		//    - bottom: z = bottom_z
		// ---------------------------------------------------------
		const bool on_xmin = std::abs(x_coord - xmin) <= tol_face;
		const bool on_xmax = std::abs(x_coord - xmax) <= tol_face;
		const bool on_ymin = std::abs(y_coord - ymin) <= tol_face;
		const bool on_ymax = std::abs(y_coord - ymax) <= tol_face;
		const bool on_bottom = std::abs(z_coord - bottom_z) <= tol_face;
		
		if (on_xmin || on_xmax || on_ymin || on_ymax || on_bottom) { // || on_bottom: for now let's see
			// fully fixed displacement at these nodes
			// insert the boundary condition for displacement
			std::cout<<"fixing node "<<nodei<<"\n";
			eBC_x.insert ( std::pair<int,double>(nodei*3+0,myTissue.node_x[nodei](0)) ); // fix x coordinate with the initial value
			eBC_x.insert ( std::pair<int,double>(nodei*3+1,myTissue.node_x[nodei](1)) ); // fix y coordinate with the initial value
			eBC_x.insert ( std::pair<int,double>(nodei*3+2,myTissue.node_x[nodei](2)) ); // fix z coordinate with the initial value
		}

		// ---------------------------------------------------------
		// 2) Apply displacement ONLY on TOP surface patch:
		//    - top: z == top_z
		//    - circle: (x-xc)^2 + (y-yc)^2 <= rad^2
		//    - impose z displacement (Dirichlet in z only)
		// ---------------------------------------------------------
		const bool on_top = std::abs(z_coord - top_z) <= tol_face;
		const double r2 = (x_coord - x_center)*(x_coord - x_center)
						+ (y_coord - y_center)*(y_coord - y_center);

		if ((on_top) &&(r2 <= rad_load*rad_load + tol_patch)) {  
			// apply -z displacement (push downward)
			std::cout<<"loaded node"<<nodei<<"\n";
			myTissue.node_x[nodei](2) = myTissue.node_x[nodei](2) - z_displ; // apply the displacement
			nodesZPush.push_back(nodei);
			// enforce only z DOF
			eBC_x.insert(std::pair<int,double>(nodei*3+2, myTissue.node_x[nodei](2)) ); // fix z coordinate to given value
		}
	}

	std::cout << "eBC_x size = " << eBC_x.size() << "\n";
	std::cout << "loaded nodes = " << nodesZPush.size() << "\n";

	//
	myTissue.eBC_x = eBC_x;
	myTissue.eBC_rho = eBC_rho;
	myTissue.eBC_c = eBC_c;
	myTissue.nBC_x = nBC_x;
	myTissue.nBC_rho = nBC_rho;
	myTissue.nBC_c = nBC_c;

	// 
	myTissue.time_final = (7*24)+1; // in hours
	myTissue.time_step = 0.2;
	myTissue.tol = 1e-8;
	myTissue.max_iter = 25;
	
	//
	std::cout<<"filling dofs...\n";
	fillDOFmap(myTissue);
	std::cout<<"going to eval jacobians...\n";
	evalElemJacobians(myTissue);
    std::cout<<"going to eval surface jacobians...\n";
    //evalElemJacobiansSurface(myTissue);
	//
	//print out the Jacobians
	std::cout<<"element jacobians\nJacobians= ";
	std::cout<<myTissue.elem_jac_IP.size()<<"\n";
	for(int i=0;i<10;i++){
		std::cout<<"element: "<<i<<"\n";
		for(int j=0;j<IP_size;j++){
			std::cout<<"ip; "<<j<<"\n"<<myTissue.elem_jac_IP[i][j]<<"\n";
		}
	}

	// ---- Check element Jacobian determinants ----
	double det_min =  std::numeric_limits<double>::infinity();
	double det_max = -std::numeric_limits<double>::infinity();
	int n_neg = 0, n_zeroish = 0;

	const double det_eps = 1e-14;   // adjust depending on your mesh scale

	for (int ei = 0; ei < myTissue.n_vol_elem; ++ei) {
		for (int ip = 0; ip < (int)myTissue.elem_jac_IP[ei].size(); ++ip) {

			// assuming elem_jac_IP[ei][ip] is an Eigen::Matrix3d (or 3x3)
			const double detJ = myTissue.elem_jac_IP[ei][ip].determinant();

			det_min = std::min(det_min, detJ);
			det_max = std::max(det_max, detJ);

			if (detJ < 0.0) n_neg++;
			if (std::abs(detJ) < det_eps) n_zeroish++;

			// print only suspicious ones (recommended)
			if (detJ <= 0.0 || std::abs(detJ) < det_eps) {
				std::cout << "[BAD J] elem " << ei << " ip " << ip
						<< " detJ = " << detJ << "\n"
						<< myTissue.elem_jac_IP[ei][ip] << "\n";
			}
		}
	}

	std::cout << "Jacobian det summary:\n";
	std::cout << "  detJ_min   = " << det_min << "\n";
	std::cout << "  detJ_max   = " << det_max << "\n";
	std::cout << "  n_neg      = " << n_neg << "\n";
	std::cout << "  n_zeroish  = " << n_zeroish << " (|detJ| < " << det_eps << ")\n";
	// // print out the forward dof map
	// std::cout<<"Total :"<<myTissue.n_dof<<" dof\n";
	// for(int i=0;i<myTissue.dof_fwd_map_x.size();i++){
	// 	std::cout<<"x node*3+coord: "<<i<<", dof: "<<myTissue.dof_fwd_map_x[i]<<"\n";
	// }
	// for(int i=0;i<myTissue.dof_fwd_map_rho.size();i++){
	// 	std::cout<<"rho node: "<<i<<", dof: "<<myTissue.dof_fwd_map_rho[i]<<"\n";
	// }
	// for(int i=0;i<myTissue.dof_fwd_map_c.size();i++){
	// 	std::cout<<"c node: "<<i<<", dof: "<<myTissue.dof_fwd_map_c[i]<<"\n";
	// }
	//
	std::cout<<"going to start solver\n";
	// save a node and an integration point to a file
	std::vector<int> save_node;save_node.clear();
	std::vector<int> save_ip;save_ip.clear();

	// Save an original configuration file
    std::stringstream ss;
	std::string outPrefix = "paraviewoutput_mech"+ss.str()+"_";

    //----------------------------------------------------------//
	// SOLVE
 	sparseLoadSolver(myTissue, outPrefix, 1,save_node,save_ip);

	double Fz = computePatchReactionZ(myTissue, nodesZPush);

   	std::cout<< "uz = " << z_displ << ", Fz = " << Fz << std::endl;
   return 0;	
}
