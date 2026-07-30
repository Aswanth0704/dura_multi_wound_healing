/*

RESULTS for circular wound problem.

Read a mesh defined by myself or imported from another file,
Then apply boundary conditions.
Solve.
*/

#include <omp.h>
#include "file_io.h"
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

inline void build_cylinder_frame(
    const Eigen::Vector3d& X,              // point location
    const Eigen::Vector3d& a0_in,           // preferred fiber direction (axial)
    double xc, double yc,                   // cylinder center in x-y
    Eigen::Vector3d& a0,
    Eigen::Vector3d& s0,
    Eigen::Vector3d& n0
){
    const double eps = 1e-12;

    // 1) normalize a0
    a0 = a0_in;
    if(a0.norm() < eps) a0 = Eigen::Vector3d(0,0,1);
    a0.normalize();

    // 2) geometric normal = radial direction (project to x-y plane)
    Eigen::Vector3d r(X(0) - xc, X(1) - yc, 0.0);
    if(r.norm() < eps){
        // pathological: point on axis; choose any normal perpendicular to a0
        Eigen::Vector3d tmp = (std::abs(a0.dot(Eigen::Vector3d::UnitX())) < 0.9)
                              ? Eigen::Vector3d::UnitX()
                              : Eigen::Vector3d::UnitY();
        n0 = a0.cross(tmp);
    } else {
        n0 = r;
    }
    n0.normalize();

    // 3) circumferential/tangential direction
    s0 = n0.cross(a0);
    if(s0.norm() < eps){
        // a0 parallel to n0 (shouldn't happen for axial a0, but be robust)
        Eigen::Vector3d tmp = (std::abs(n0.dot(Eigen::Vector3d::UnitZ())) < 0.9)
                              ? Eigen::Vector3d::UnitZ()
                              : Eigen::Vector3d::UnitX();
        s0 = n0.cross(tmp);
    }
    s0.normalize();

    // 4) re-orthogonalize n0 to guarantee right-handed orthonormal frame
    n0 = a0.cross(s0);
    n0.normalize();

    // Optional: enforce n0 outward (for a cylinder centered at xc,yc outward already)
    // Optional: enforce a consistent sign (e.g., n0(2) >= 0) if you need continuity
}


int main(int argc, char *argv[])
{
    Eigen::initParallel();
	std::cout<<"\nRunning full domain simulations with " << Eigen::nbThreads( ) << " threads.\n";
	srand (time(NULL));

	// for normalization
	double rho_phys = 1000*55.05126; // [cells/mm^3]
	double c_max = 1.0e-4; // [g/mm3] from tgf beta review, 5e-5g/mm3 was good for tissues

	// values for the wound
	double rho_wound = 1000; // [cells/mm^3]
	double c_wound = c_max;
	double phif0_wound = 0.01;
	double kappa0_wound = 1./3; // fine. using the uniform dispersion

	//---------------------------------//
	// values for the healthy
	double rho_healthy = rho_phys; // [cells/mm^3]
	double c_healthy = 0.0; // 0.0;
	double phif0_healthy = 1.0;
	double kappa0_healthy = 0.024; // fiber dispersion from my own experiments. Zach : 1./3;
    
	//---------------------------------//
	// GLOBAL PARAMETERS
	double k0 = 0.02;  //  Breast:0.00667792 -->  neo hookean for skin, used previously, in MPa. Value is modified for each patient based on breast density calculation. (0.00667792 represents a patient with 33% fibroglandular tissue).
	double kf = 40; // Breast: 0.015 --> stiffness of collagen in MPa, from previous paper
	double k2 = 0.048; // used the same value as Zach. nonlinear exponential coefficient, non-dimensional
	double t_rho = (1.28571E-6/55.05126); // Zach: (1.28571E-5/55.05126); 0.0045 force of fibroblasts in MPa, this is per cell. so, in an average sense this is the production by the natural density
	double t_rho_c = (1.28571E-6*3.28571)/55.05126; // Zach: (1.28571E-5*3.28571)/55.05126 0.045 force of myofibroblasts enhanced by chemical, I'm assuming normalized chemical, otherwise I'd have to add a normalizing constant
	double K_t = 0.2; // Saturation of mechanical force by collagen
	double K_t_c = c_max/10.; // saturation of chemical on force. this can be calculated from steady state
	double D_rhorho = 0.0833; // diffusion of cells in [mm^2/hour]; 0.0833 was used in Buganza et al. (2017) and Sohutskay et al. (2021), but the value is a placeholder and not implemented — actual value is defined in wound.cpp
	double D_rhoc = 0; // diffusion of chemotactic gradient, an order of magnitude greater than random walk [mm^2/hour], not normalized
	double D_cc = 0.01208; // 0.15 diffusion of chemical TGF, not normalized.
	double p_rho = 0.04958333/55.05126; // in 1/hour production of fibroblasts naturally, proliferation rate, not normalized, based on data of doubling rate from commercial use
	double p_rho_c = 0.015314; // production enhanced by the chem, if the chemical is normalized, then suggest two fold,
	double p_rho_theta = p_rho/2; // enhanced production by theta
	double K_rho_c = c_max/10.; // saturation of cell proliferation by chemical, this one is definitely not crucial, just has to be small enough <cmax
    double K_rho_rho = 10000*55.05126; // saturation of cell by cell, from steady state
	double vartheta_e = 2.; // physiological state of area stretch
	double gamma_theta = 5.; // sensitivity of heaviside function
	// const double H0 = 1.0 / (1.0 + std::exp(-gamma_theta * (vartheta_e - vartheta_e))); // 0.5??
    // const double A         = p_rho
	// 						+ p_rho_c * (c_max / (K_rho_c + c_max))
	// 						+ p_rho_theta * H0;
	// const double A         = p_rho
							// + p_rho_theta * H0;
	// double d_rho           = A * (1.0 - rho_phys / K_rho_rho);

	
	
    double d_rho = p_rho*(1-rho_phys/K_rho_rho); // percent of cells die per day, 0.1*p_rho 10% in the original, now much less, determined to keep cells in dermis constant
	double p_c_rho = 90.0e-16/rho_phys*10;// production of c by cells in g/cells/h
	double d_c = 0.01/2; // 0.01 decay of chemical in 1/hours
	double K_c_c = 1.;// saturation of chem by chem, from steady state
	double p_c_thetaE = 300.0e-16/rho_phys*10; // coupling of elastic and chemical, three fold
	// double p_c_thetaE      = (d_c * c_max * c_max
	// 							+ (d_c * K_c_c - p_c_rho * rho_phys) * c_max)
	// 							/ (H0 * rho_phys);
	double bx = 0; // body force
    double by = 0; //-0.001; // body force
    double bz = 0; // body force
	//---------------------------------//
	std::vector<double> global_parameters = {k0,kf,k2,t_rho,t_rho_c,K_t,K_t_c,D_rhorho,D_rhoc,D_cc,p_rho,p_rho_c,p_rho_theta,K_rho_c,K_rho_rho,d_rho,vartheta_e,gamma_theta,p_c_rho,p_c_thetaE,K_c_c,d_c,bx,by,bz};

	//---------------------------------//
	// LOCAL PARAMETERS
	//
	// collagen fraction
	double p_phi = 1.4E-8; // production by fibroblasts, natural rate in percent/hour, 5% per day
	double p_phi_c = 7E-8; // production up-regulation, weighted by C and rho
	double p_phi_theta = p_phi; // mechanosensing upregulation. no need to normalize by Hmax since Hmax = 1
	double K_phi_c = 0.0001; // saturation of C effect on deposition.
	double d_phi = 3.7413E-4; // rate of degradation, in the order of the wound process, 100 percent in one year for wound, means 0.000116 effective per hour means degradation = 0.002 - 0.000116
	double d_phi_rho_c = 0.5*0.000970/rho_phys/c_max/10.0; //0.000194; // degradation coupled to chemical and cell density to maintain phi equilibrium
	// const double G         = p_phi
	// 						+ p_phi_c * (c_max / (K_phi_c + c_max))
	// 						+ p_phi_theta * H0;
	// const double C         = d_phi + d_phi_rho_c * c_max * rho_phys;
	// double K_phi_rho       = (G * rho_phys) / (C * phif0_healthy) - phif0_healthy;

	// const double G         = p_phi
	// 						+ p_phi_theta * H0;
	// const double C         = d_phi;
	// double K_phi_rho       = (G * rho_phys) / (C * phif0_healthy) - phif0_healthy;
	double K_phi_rho = rho_phys*p_phi/d_phi - 1; // saturation of collagen fraction itself, from steady state
	
	// fiber alignment
	double tau_omega = 10./(K_phi_rho+1); // time constant for angular reorientation, think 100 percent in one year
	//
	// dispersion parameter
	double tau_kappa = 1./(K_phi_rho+1); // time constant, on the order of a year
	double gamma_kappa = 5.; // exponent of the principal stretch ratio
	// 
	// permanent contracture/growth
	double tau_lamdaP_a =  0.05;// 5.0; // 1.0 time constant for direction a, on the order of a year
	double tau_lamdaP_s =  0.05;// 5.0; // 1.0 time constant for direction s, on the order of a year
    double tau_lamdaP_n =  0.05;// 5.0; // 1.0 time constant for direction s, on the order of a year

    // solution parameters
    double tol_local = 1e-8; // local tolerance (also try 1e-5)
    double time_step_ratio = 100; // time step ratio between local and global (explicit)
    double max_iter = 100; // max local iter (implicit)
    //---------------------------------//
	std::vector<double> local_parameters = {p_phi,p_phi_c,p_phi_theta,K_phi_c,K_phi_rho,d_phi,d_phi_rho_c,tau_omega,tau_kappa,gamma_kappa,tau_lamdaP_a,tau_lamdaP_s,tau_lamdaP_n,vartheta_e,gamma_theta,tol_local,time_step_ratio,max_iter};

	
	
	//---------------------------------//
	// double a0x = frand(-1,1.);
    // double a0y = frand(-1,1.);
    // double a0z = 0.;
    Vector3d a0_wound; a0_wound << 0.0, 0.0, 1.0; // start with the same axial orientation.
    // a0_wound = a0_wound/sqrt(a0_wound.dot(a0_wound));
	Vector3d lamda0_wound;lamda0_wound << 1.,1.,1.;
	//---------------------------------//
	Vector3d a0_healthy;a0_healthy<<0.,0.,1.; // assuming the collagen is oriented along the longitudinal direction
 	Vector3d lamda0_healthy;lamda0_healthy<<1.,1.,1.;
	//---------------------------------//


    // --------- Invalid for dura mater with cylindrical geometry --------------//
    // Matrix3d Rot90;Rot90 << 0.,-1.,0., 1.,0.,0., 0.,0.,1.;
    // Vector3d s0_wound = Rot90*a0_wound;
    // Vector3d s0_healthy= Rot90*a0_healthy;
    // Vector3d n0_wound = s0_wound.cross(a0_wound);
    // if(n0_wound(2)<0){
    //     n0_wound = a0_wound.cross(s0_wound);
    // }
    // Vector3d n0_healthy= s0_healthy.cross(a0_healthy);
    // if(n0_healthy(2)<0){
    //     n0_healthy = a0_healthy.cross(s0_healthy);
    // }
    //---------------------------------//

	Vector3d s0_healthy(1.,0.,0.);
	Vector3d n0_healthy(0.,1.,0.);
	Vector3d s0_wound = s0_healthy;
	Vector3d n0_wound = n0_healthy;


	
	//---------------------------------//
	// create mesh (only nodes and elements)
	std::cout<<"Going to create the mesh\n";
    // The hex dimensions should be specified here even if you are importing a file!
    // This will allow correct specification of the boundary values

    // geometry dimensions:
    double r_cord = 5.0; // [mm] radius of the spinal cord
    double t_dura = 0.4; // [mm] thickness of the dura mater.
    double r_wound = 0.25; // [mm] 25 gauge needle
    double h_lumbar = 10.0; // [mm] approximate length of the lumbar dura
    double Xmin = -(r_cord + t_dura); //
    double Xmax = (r_cord + t_dura); //
    double Ymin = Xmin;
    double Ymax = Xmax;
    double Zmin = 0.0;
    double Zmax = 10.0;
	std::vector<double> hexDimensions = {Xmin, Xmax, Ymin, Ymax, Zmin, Zmax};
	std::vector<int> meshResolution =  {16,16,6};
    std::string mesh_filename = "dura_cyl_repeated_wound_v62_20t_finer.mphtxt"; // CHANGE
    HexMesh myMesh = readCOMSOLInput(mesh_filename, hexDimensions, meshResolution);

    // Other possibles meshes:
	//HexMesh myLinearMesh = myHexMesh(hexDimensions, meshResolution);
    //HexMesh myMesh = myQuadraticHexMesh(hexDimensions, meshResolution);
    //HexMesh myMesh = myQuadraticLagrangianHexMesh(hexDimensions, meshResolution);
	//HexMesh myMesh = myMultiBlockMesh(hexDimensions, meshResolution);
    //std::string mesh_filename = "COMSOL_3D_hex_linear_100x30.vtk";
    //HexMesh myMesh = readParaviewInput(mesh_filename, hexDimensions, meshResolution);
    //HexMesh myMesh = SerendipityQuadraticHexMeshfromLinear(myLinearMesh, hexDimensions, meshResolution);

    std::cout<<"Created the mesh with "<<myMesh.n_nodes<<" nodes and "<<myMesh.boundary_flag.size()<<" boundaries and "<<myMesh.n_elements<<" elements\n";
    std::cout<<"Created the surface mesh with "<<myMesh.n_nodes<<" nodes and "<<myMesh.surface_boundary_flag.size()<<" boundaries and "<<myMesh.n_surf_elements<<" elements\n";
	// print the mesh
	// prints x, y, z coordinates
	std::cout<<"nodes\n";
	for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
		std::cout<<myMesh.nodes[nodei](0)<<","<<myMesh.nodes[nodei](1)<<","<<myMesh.nodes[nodei](2)<<"\n";
	}
	// prints nodes associated with each element
	std::cout<<"elements\n";
    for(int elemi=0;elemi<myMesh.n_elements;elemi++){
        for(int nodei=0;nodei<myMesh.elements[elemi].size();nodei++){
            std::cout<<myMesh.elements[elemi][nodei]<<" ";
        }
        std::cout<<"\n";
    }
	// prints boundary
	std::cout<<"boundary\n";
	for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
		std::cout<<myMesh.boundary_flag[nodei]<<"\n";
	}
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
    std::vector<Vector3d> ip_s00(myMesh.n_elements*IP_size,s0_healthy); // will update it later inside
    std::vector<Vector3d> ip_n00(myMesh.n_elements*IP_size,n0_healthy); // will update it later inside
	std::vector<double> ip_kappa0(myMesh.n_elements*IP_size,kappa0_healthy);
	std::vector<Vector3d> ip_lamda0(myMesh.n_elements*IP_size,lamda0_healthy);
	//
    double tol_boundary = 1e-5;
	// define wound domain. Note: These values are unique for each patient.
	double y_center = 0.0; // assuming the wound is a cylinder along the -ve x axis
	double z_center = 2*t_dura; // brought wound_1 down
    const double Xmin_wound = Xmin-tol_boundary;
    const double Xmax_wound = -r_cord+ 0.01;
	
	// boundary conditions and definition of the wound
	std::map<int,double> eBC_x;
	std::map<int,double> eBC_rho;
	std::map<int,double> eBC_c;
	for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
		double x_coord = myMesh.nodes[nodei](0);
		double y_coord = myMesh.nodes[nodei](1);
		double z_coord = myMesh.nodes[nodei](2);
		// check if node is fixed. Since tumor was in left breast, all nodes on the right breast are fixed (x_coord <= 0). All nodes past the chest wall are also fixed (y_coord >= 50)
		
		if(myMesh.boundary_flag[nodei] == 1){ 

			// insert the boundary condition for displacement
			std::cout<<"fixing node "<<nodei<<"\n";
			eBC_x.insert ( std::pair<int,double>(nodei*3+0,myMesh.nodes[nodei](0)) ); // x coordinate
			eBC_x.insert ( std::pair<int,double>(nodei*3+1,myMesh.nodes[nodei](1)) ); // y coordinate
			eBC_x.insert ( std::pair<int,double>(nodei*3+2,myMesh.nodes[nodei](2)) ); // z coordinate
			// insert the boundary condition for rho
			eBC_rho.insert ( std::pair<int,double>(nodei,rho_healthy) );
			// insert the boundary condition for c
			eBC_c.insert   ( std::pair<int,double>(nodei, c_healthy)); //c_healthy // CHANGE
		}
		//-----------------//
		
        // test inside cylinder segment:
        double r2 = (y_coord - y_center)*(y_coord - y_center) + (z_coord - z_center)*(z_coord - z_center);

        bool inside_cyl = (x_coord >= Xmin_wound && x_coord <= Xmax_wound) && (r2 <= (r_wound+ tol_boundary)*(r_wound+ tol_boundary));

        if(inside_cyl){
            std::cout << "wound node " << nodei << "\n";
            node_rho0[nodei] = rho_wound;
            node_c0[nodei]   = c_wound;
        }

	}
	for(int elemi=0;elemi<myMesh.n_elements;elemi++){
		for(int ip=0;ip<IP_size;ip++)
		{
			double xi = IP[ip](0);
			double eta = IP[ip](1);
			double zeta = IP[ip](2);
			// weight of the integration point
			double wip = IP[ip](3);
            std::vector<double> R;
            if(elem_size == 8){
                R = evalShapeFunctionsR(xi,eta,zeta);
            }
            else if(elem_size == 20){
                R = evalShapeFunctionsQuadraticR(xi,eta,zeta);
            }
            else if(elem_size == 27){
                R = evalShapeFunctionsQuadraticLagrangeR(xi,eta,zeta);
            }
            else if(elem_size == 4){
                R = evalShapeFunctionsTetR(xi,eta,zeta);
            }
            else if(elem_size == 10){
                R = evalShapeFunctionsTetQuadraticR(xi,eta,zeta);
            }
            else{
                throw std::runtime_error("Wrong number of nodes in element!");
            }
			Vector3d X_IP; X_IP.setZero();
			for(int nodej=0;nodej<elem_size;nodej++){
			    //std::cout<<R[nodej]<<"\n";
                //std::cout<<myMesh.nodes[myMesh.elements[elemi][nodej]]<<"\n";
				X_IP += R[nodej]*myMesh.nodes[myMesh.elements[elemi][nodej]];
			}

            // test inside cylinder segment:
            double r2 = (X_IP(1) - y_center)*(X_IP(1) - y_center) + (X_IP(2)- z_center)*(X_IP(2) - z_center);

            bool inside_cyl = (X_IP(0) >= Xmin_wound && X_IP(0) <= Xmax_wound) && (r2 <= (r_wound+ tol_boundary)*(r_wound+ tol_boundary));

            if(inside_cyl){
                std::cout<<"IP node: "<<IP_size*elemi+ip<<"\n";
				// update the wound geometry with orientations:
				Eigen::Vector3d a0,s0,n0;
				build_cylinder_frame(X_IP, a0_wound , 0.0, 0.0, a0, s0, n0);
				ip_a00[elemi*IP_size+ip] = a0;
				ip_s00[elemi*IP_size+ip] = s0;
				ip_n00[elemi*IP_size+ip] = n0;		
                ip_phi0[elemi*IP_size+ip] = phif0_wound;
                ip_kappa0[elemi*IP_size+ip] = kappa0_wound;
                ip_lamda0[elemi*IP_size+ip] = lamda0_wound;
                
            }
			else{
				Eigen::Vector3d a0,s0,n0;
				build_cylinder_frame(X_IP, a0_healthy , 0.0, 0.0, a0, s0, n0);
				ip_a00[elemi*IP_size+ip] = a0;
				ip_s00[elemi*IP_size+ip] = s0;
				ip_n00[elemi*IP_size+ip] = n0;		
			}
            
            //-----------------//
		}
	}
	// neumann boundary conditions.
	std::map<int,double> nBC_x; /// This is a map from the node to the condition (three times as long for x)
	std::map<int,double> nBC_rho; /// Could also use the map from face numbering
	std::map<int,double> nBC_c; /// Make some kind of flag so that if we are running BC we don't update cells, etc

	// initialize my tissue
	tissue myTissue;
	// connectivity
	myTissue.vol_elem_connectivity = myMesh.elements;
    myTissue.surf_elem_connectivity = myMesh.surface_elements;
	// parameters
	myTissue.global_parameters = global_parameters;
	myTissue.local_parameters = local_parameters;
	myTissue.boundary_flag = myMesh.boundary_flag;
    myTissue.surface_boundary_flag = myMesh.surface_boundary_flag;
	//
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
    //
	myTissue.eBC_x = eBC_x;
	myTissue.eBC_rho = eBC_rho;
	myTissue.eBC_c = eBC_c;
	myTissue.nBC_x = nBC_x;
	myTissue.nBC_rho = nBC_rho;
	myTissue.nBC_c = nBC_c;
	myTissue.time_final = (7*24*4)+1; // CHANGE
	myTissue.time_step = 0.2;
	myTissue.tol = 1e-8;
	myTissue.max_iter = 25;
	myTissue.n_node = myMesh.n_nodes;
	myTissue.n_vol_elem = myMesh.n_elements;
    myTissue.n_surf_elem = myMesh.n_surf_elements;
	myTissue.n_IP = IP_size*myMesh.n_elements;
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
	for(int i=0;i<myTissue.elem_jac_IP.size();i++){
		std::cout<<"element: "<<i<<"\n";
		for(int j=0;j<IP_size;j++){
			std::cout<<"ip; "<<j<<"\n"<<myTissue.elem_jac_IP[i][j]<<"\n";
		}
	}
	// print out the forward dof map
	std::cout<<"Total :"<<myTissue.n_dof<<" dof\n";
	for(int i=0;i<myTissue.dof_fwd_map_x.size();i++){
		std::cout<<"x node*3+coord: "<<i<<", dof: "<<myTissue.dof_fwd_map_x[i]<<"\n";
	}
	for(int i=0;i<myTissue.dof_fwd_map_rho.size();i++){
		std::cout<<"rho node: "<<i<<", dof: "<<myTissue.dof_fwd_map_rho[i]<<"\n";
	}
	for(int i=0;i<myTissue.dof_fwd_map_c.size();i++){
		std::cout<<"c node: "<<i<<", dof: "<<myTissue.dof_fwd_map_c[i]<<"\n";
	}
	//
	// 
	std::cout<<"going to start solver\n";
	// save a node and an integration point to a file
	std::vector<int> save_node;save_node.clear();
	std::vector<int> save_ip;save_ip.clear();

	std::stringstream ss;
	std::string filename = "wound_1_20t_4w_output"+ss.str()+"_"; // CHANGE

    // check the bc and wound initial conditions
    // std::string tag = "REF_BC";

    // std::string filename_step  = "wound_1_12t_output_" + tag + ".vtk"; // CHANGE
    // std::string filename_step2 = "wound_1_20t_output_second_" + tag + ".vtk"; // CHANGE

    // writeParaview(myTissue,
    //           filename_step.c_str(),
    //           filename_step2.c_str());

	//----------------------------------------------------------//
	// SOLVE
//    sparseLoadSolver(myTissue, filename, 1,save_node,save_ip);
    sparseWoundSolver(myTissue, filename, 5,save_node,save_ip);
	//----------------------------------------------------------//

	// Here I want to inflict another wound.
	// wound is along the same -ve x axis with same Xlimits
	// with y_center = 0.0 and h_center = h_lumbar/2.0 + 2*t_dura
	z_center = z_center + 20*t_dura; // CHANGE 
	for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
		double x_coord = myTissue.node_x[nodei](0);
		double y_coord = myTissue.node_x[nodei](1);
		double z_coord = myTissue.node_x[nodei](2);
		
        // test inside cylinder segment:
        double r2 = (y_coord - y_center)*(y_coord - y_center) + (z_coord - z_center)*(z_coord - z_center);

        bool inside_cyl = (x_coord >= Xmin_wound && x_coord <= Xmax_wound) && (r2 <= (r_wound+ tol_boundary)*(r_wound+ tol_boundary));

        if(inside_cyl){
            std::cout << "wound node " << nodei << "\n";
			myTissue.node_rho_0[nodei] = rho_wound;
			myTissue.node_rho[nodei] = rho_wound;
			myTissue.node_c_0[nodei] = c_wound;
			myTissue.node_c[nodei] = c_wound;
        }

	}
	for(int elemi=0;elemi<myMesh.n_elements;elemi++){
		for(int ip=0;ip<IP_size;ip++)
		{
			double xi = IP[ip](0);
			double eta = IP[ip](1);
			double zeta = IP[ip](2);
			// weight of the integration point
			double wip = IP[ip](3);
            std::vector<double> R;
            if(elem_size == 8){
                R = evalShapeFunctionsR(xi,eta,zeta);
            }
            else if(elem_size == 20){
                R = evalShapeFunctionsQuadraticR(xi,eta,zeta);
            }
            else if(elem_size == 27){
                R = evalShapeFunctionsQuadraticLagrangeR(xi,eta,zeta);
            }
            else if(elem_size == 4){
                R = evalShapeFunctionsTetR(xi,eta,zeta);
            }
            else if(elem_size == 10){
                R = evalShapeFunctionsTetQuadraticR(xi,eta,zeta);
            }
            else{
                throw std::runtime_error("Wrong number of nodes in element!");
            }
			Vector3d X_IP; X_IP.setZero();
			for(int nodej=0;nodej<elem_size;nodej++){
			    //std::cout<<R[nodej]<<"\n";
                //std::cout<<myMesh.nodes[myMesh.elements[elemi][nodej]]<<"\n";
				X_IP += R[nodej]*myTissue.node_x[myMesh.elements[elemi][nodej]];
			}

            // test inside cylinder segment:
            double r2 = (X_IP(1) - y_center)*(X_IP(1) - y_center) + (X_IP(2)- z_center)*(X_IP(2) - z_center);

            bool inside_cyl = (X_IP(0) >= Xmin_wound && X_IP(0) <= Xmax_wound) && (r2 <= (r_wound+ tol_boundary)*(r_wound+ tol_boundary));

            if(inside_cyl){
                std::cout<<"IP node: "<<IP_size*elemi+ip<<"\n";
				Eigen::Vector3d a0,s0,n0;
				build_cylinder_frame(X_IP, a0_wound , 0.0, 0.0, a0, s0, n0);
				myTissue.ip_phif_0[elemi*IP_size+ip] = phif0_wound;	
				myTissue.ip_phif[elemi*IP_size+ip] = phif0_wound;	
				myTissue.ip_a0_0[elemi*IP_size+ip] = a0;
				myTissue.ip_a0[elemi*IP_size+ip] = a0;
				myTissue.ip_s0_0[elemi*IP_size+ip] = s0;
				myTissue.ip_s0[elemi*IP_size+ip] = s0;
				myTissue.ip_n0_0[elemi*IP_size+ip] = n0;
				myTissue.ip_n0[elemi*IP_size+ip] = n0;
				myTissue.ip_kappa_0[elemi*IP_size+ip] = kappa0_wound;
				myTissue.ip_kappa[elemi*IP_size+ip] = kappa0_wound;	
				myTissue.ip_lamdaP_0[elemi*IP_size+ip] = lamda0_wound;
				myTissue.ip_lamdaP[elemi*IP_size+ip] = lamda0_wound;
				myTissue.ip_lamdaE[elemi*IP_size+ip] = lamda0_wound;
				// std::vector<Matrix3d> ip_strain(myMesh.n_elements*IP_size,Matrix3d::Identity(3,3));
				// std::vector<Matrix3d> ip_stress(myMesh.n_elements*IP_size,Matrix3d::Zero(3,3));
				// myTissue.ip_strain = ; // what should be this
				// myTissue.ip_stress = ; // what should be this

                // ip_phi0[elemi*IP_size+ip] = phif0_wound;
                // ip_a00[elemi*IP_size+ip] = a0_wound;
                // ip_s00[elemi*IP_size+ip] = s0_wound;
                // ip_n00[elemi*IP_size+ip] = n0_wound;
                // ip_kappa0[elemi*IP_size+ip] = kappa0_wound;
                // ip_lamda0[elemi*IP_size+ip] = lamda0_wound;
                
            }
            
            //-----------------//

		}
	}
	// for now we will only solve for single wound for testing.
	// // // check the bc and wound initial conditions
    // // tag = "REF_BC";

    // // filename_step  = "wound_2_20t_output_" + tag + ".vtk"; // CHANGE
    // // filename_step2 = "wound_2_20t_output_second_" + tag + ".vtk"; // CHANGE

    // // writeParaview(myTissue,
    // //           filename_step.c_str(),
    // //           filename_step2.c_str());
	// myTissue.time_final = (7*24*4)+1; // in hours 
	// save_node.clear();
	// save_ip.clear();
	// std::stringstream ss2;
	// std::string filename2 = "wound_2_20t_4w_output"+ss2.str()+"_"; // CHANGE
	// sparseWoundSolver(myTissue, filename2, 5,save_node,save_ip);
	return 0;	
}