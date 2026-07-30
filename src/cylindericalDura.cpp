// last date modified: Sept, 16, 2025

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
	double c0 = 1.0e-4;
	double k0 = 0.02; // 0.0511; neo hookean for skin, used previously, in MPa
	double kf = 40.; // stiffness of collagen in MPa, from previous paper
	double k2 = 0.048; // nonlinear exponential coefficient, non-dimensional
	double t_rho = 1e-1*(1.28571E-5); // 0.0045 force of fibroblasts in MPa, this is per cell. so, in an average sense this is the production by the natural density
	double t_rho_c = 1e-1*(1.28571E-5*3.28571); // 0.045 force of myofibroblasts enhanced by chemical, I'm assuming normalized chemical, otherwise I'd have to add a normalizing constant
	double K_t = 0.2; // Saturation of mechanical force by collagen
	double K_t_c = c_max/10.; // saturation of chemical on force. this can be calculated from steady state
	double D_rhorho = 0.0833; // 0.0833 diffusion of cells in [mm^2/hour], not normalized
	double D_rhoc = 0.0; // diffusion of chemotactic gradient, an order of magnitude greater than random walk [mm^2/hour], not normalized
	double D_cc = 0.01208; // 0.15 diffusion of chemical TGF, not normalized.
	double p_rho = 0.04958333; // in 1/hour production of fibroblasts naturally, proliferation rate, not normalized, based on data of doubling rate from commercial use
	double p_rho_c = 0.015314; // production enhanced by the chem, if the chemical is normalized, then suggest two fold,
	double p_rho_theta = p_rho/2; // enhanced production by theta
	double K_rho_c = c_max/10.; // saturation of cell proliferation by chemical, this one is definitely not crucial, just has to be small enough <cmax
    double K_rho_rho = 10000; // saturation of cell by cell, from steady state
    // double d_rho = p_rho*(1-rho_phys/K_rho_rho); // percent of cells die per day, 0.1*p_rho 10% in the original, now much less, determined to keep cells in dermis constant
	double vartheta_e = 2.; // physiological state of area stretch
	double gamma_theta = 5.; // sensitivity of heaviside function
	const double H0 = 1.0 / (1.0 + std::exp(-gamma_theta * (1.0 - vartheta_e)));
    const double A = p_rho
                 + p_rho_c * (c0 / (K_rho_c + c0))
                 + p_rho_theta * H0;
    double d_rho = A * (1.0 - rho_phys / K_rho_rho);
	double p_c_rho = 90.0e-16/rho_phys*10;// production of c by cells in g/cells/h
	double K_c_c = c0;// saturation of chem by chem, from steady state
	double d_c = 0.01; // 0.01 decay of chemical in 1/hours
	// double p_c_thetaE = 300.0e-16/rho_phys*10; // coupling of elastic and chemical, three fold
	double p_c_thetaE = (d_c * c0 * c0
               + (d_c * K_c_c - p_c_rho * rho_phys) * c0)
               / (H0 * rho_phys);
	
	double bx = 0; // body force
    double by = 0; //-0.001; // body force
    double bz = 0; // body force
	//---------------------------------//
	std::vector<double> global_parameters = {k0,kf,k2,t_rho,t_rho_c,K_t,K_t_c,D_rhorho,D_rhoc,D_cc,p_rho,p_rho_c,p_rho_theta,K_rho_c,K_rho_rho,d_rho,vartheta_e,gamma_theta,p_c_rho,p_c_thetaE,K_c_c,d_c,bx,by,bz};

	//---------------------------------//
	// LOCAL PARAMETERS
	//
	// collagen fraction
	double phif0 = 1.0;
	double p_phi = 0.002/rho_phys; // production by fibroblasts, natural rate in percent/hour, 5% per day
	double p_phi_c = p_phi; // production up-regulation, weighted by C and rho
	double p_phi_theta = p_phi; // mechanosensing upregulation. no need to normalize by Hmax since Hmax = 1
	double K_phi_c = c_max; // saturation of C effect on deposition.
	double d_phi = 0.000970; // rate of degradation, in the order of the wound process, 100 percent in one year for wound, means 0.000116 effective per hour means degradation = 0.002 - 0.000116
	double d_phi_rho_c = 0.5*0.000970/rho_phys/c_max; //0.000194; // degradation coupled to chemical and cell density to maintain phi equilibrium
	// double K_phi_rho = rho_phys*p_phi/d_phi - 1; // >0 saturation of collagen fraction itself, from steady state
	const double G = p_phi
                 + p_phi_c * (c0 / (K_phi_c + c0))
                 + p_phi_theta * H0;
    const double C = d_phi + d_phi_rho_c * c0 * rho_phys;

    double K_phi_rho = (G * rho_phys) / (C * phif0) - phif0; 
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
    // The hex dimensions should be specified here even if you are importing a file!
    // This will allow correct specification of the boundary values
	// double rad = 0.282;
	// double h = 0.1;
	// double g = 0.1;
	// double Xmin = -rad -g;
	// double Xmax = rad + g;
	// double Ymin = Xmin;
	// double Ymax = Xmax;
	// double Zmin = -0.05;
	// double Zmax = 0.05;

	double r_cord   = 6.5;   // [mm] inner cord radius (cord is rigid & not meshed)
    double t_dura   = 0.4;   // [mm] dura thickness
    double t_fat    = 0.8;   // [mm] fat thickness (outer radius = r_cord + t_dura + t_fat)
    double r_needle = 0.5;  // [mm] needle radius; considered 25 gauge needle, cylinder around x-axis
    double z_min    = 0.0;  // [mm] this has to be the lower z dimension. We can change it to different axis in original dura geom
    double z_max    = 8.0;  // [mm]

    // =========================================================================
    // USER SETTINGS (units should match your mesh!)
    // =========================================================================
    // Spatial box (used for boundary tagging in reader and for the concentric assignment)
    double Xmin = -r_cord - t_dura- t_fat, Xmax = r_cord + t_dura + t_fat;
    double Ymin = Xmin,     Ymax = Xmax;
    double Zmin = 0.0,    Zmax =  8.0;

	
	std::vector<double> hexDimensions = {Xmin, Xmax, Ymin, Ymax, Zmin, Zmax};
	std::vector<int> meshResolution =  {16,16,6};
    std::string mesh_filename = "cylindericalDura.mphtxt";
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
    std::vector<Vector3d> ip_s00(myMesh.n_elements*IP_size,s0_healthy);
    std::vector<Vector3d> ip_n00(myMesh.n_elements*IP_size,n0_healthy);
	std::vector<double> ip_kappa0(myMesh.n_elements*IP_size,kappa0_healthy);
	std::vector<Vector3d> ip_lamda0(myMesh.n_elements*IP_size,lamda0_healthy);
	//
    double tol_boundary = 1e-5;
	// // define wound domain
	// double x_center = 0;
	// double y_center = 0;
	// double x_axis =  rad + tol_boundary;
	// double y_axis = rad + tol_boundary;
	// // double z_axis = 30.21096;
	// double alpha_ellipse = 0.;
	// boundary conditions and definition of the wound
	std::map<int,double> eBC_x;
	std::map<int,double> eBC_rho;
	std::map<int,double> eBC_c;

	// for mechanics only
	for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
		double x_coord = myMesh.nodes[nodei](0);
		double y_coord = myMesh.nodes[nodei](1);
		double z_coord = myMesh.nodes[nodei](2);

		// insert the boundary condition for rho
		eBC_rho.insert ( std::pair<int,double>(nodei,rho_healthy) );
		// insert the boundary condition for c
		eBC_c.insert   ( std::pair<int,double>(nodei,c_healthy) );
		// check if node is fixed
		if(myMesh.boundary_flag[nodei]==1){ //   myMesh.boundary_flag[nodei]>1 && myMesh.boundary_flag[nodei]<6
			// insert the boundary condition for displacement
			std::cout<<"fixing node "<<nodei<<"\n";
			eBC_x.insert ( std::pair<int,double>(nodei*3+0,myMesh.nodes[nodei](0)) ); // x coordinate
			eBC_x.insert ( std::pair<int,double>(nodei*3+1,myMesh.nodes[nodei](1)) ); // y coordinate
			eBC_x.insert ( std::pair<int,double>(nodei*3+2,myMesh.nodes[nodei](2)) ); // z coordinate
		}
		// check if it is in the center of the wound
//		double check_ellipse = pow((x_coord-x_center)*cos(alpha_ellipse)+(y_coord-y_center)*sin(alpha_ellipse),2)/(x_axis*x_axis) +\
//						pow((x_coord-x_center)*sin(alpha_ellipse)+(y_coord-y_center)*cos(alpha_ellipse),2)/(y_axis*y_axis);
//		if(check_ellipse<=1){
//			// inside ellipse
//			if(z_coord>=z_axis){
//				// inside cylinder
//				std::cout<<"wound node "<<nodei<<"\n";
//				node_rho0[nodei] = rho_wound;
//				node_c0[nodei] = c_wound;
//			}
//		}
		//-----------------//
		// // Alternative version with an ellipse instead of a cylinder
		// double check_ellipsoid = pow((x_coord-x_center),2)/(x_axis*x_axis) + pow((y_coord-y_center),2)/(y_axis*y_axis);
        // if(check_ellipsoid<=1){
        //     // inside ellipsoid
        //     std::cout<<"wound node "<<nodei<<"\n";
        //     node_rho0[nodei] = rho_wound;
        //     node_c0[nodei] = c_wound;
        // }
        //-----------------//
	}
// 	for(int elemi=0;elemi<myMesh.n_elements;elemi++){
// 		for(int ip=0;ip<IP_size;ip++)
// 		{
// 			double xi = IP[ip](0);
// 			double eta = IP[ip](1);
// 			double zeta = IP[ip](2);
// 			// weight of the integration point
// 			double wip = IP[ip](3);
//             std::vector<double> R;
//             if(elem_size == 8){
//                 R = evalShapeFunctionsR(xi,eta,zeta);
//             }
//             else if(elem_size == 20){
//                 R = evalShapeFunctionsQuadraticR(xi,eta,zeta);
//             }
//             else if(elem_size == 27){
//                 R = evalShapeFunctionsQuadraticLagrangeR(xi,eta,zeta);
//             }
//             else if(elem_size == 4){
//                 R = evalShapeFunctionsTetR(xi,eta,zeta);
//             }
//             else if(elem_size == 10){
//                 R = evalShapeFunctionsTetQuadraticR(xi,eta,zeta);
//             }
//             else{
//                 throw std::runtime_error("Wrong number of nodes in element!");
//             }
// 			Vector3d X_IP; X_IP.setZero();
// 			for(int nodej=0;nodej<elem_size;nodej++){
// 			    //std::cout<<R[nodej]<<"\n";
//                 //std::cout<<myMesh.nodes[myMesh.elements[elemi][nodej]]<<"\n";
// 				X_IP += R[nodej]*myMesh.nodes[myMesh.elements[elemi][nodej]];
// 			}
// 			//std::cout<<" IP node " << ip << " reference coordinates: " <<xi<< " " <<eta << " " <<zeta<< " "<<"\n";
//             //std::cout<<"Element " << elemi <<" IP node " << ip << " coordinates: " <<X_IP(0)<< " " <<X_IP(1) << " " <<X_IP(2)<< " "<<"\n";
// //            double check_ellipse_ip = pow((X_IP(0)-x_center)*cos(alpha_ellipse)+(X_IP(1)-y_center)*sin(alpha_ellipse),2)/(x_axis*x_axis) +\
// //						pow((X_IP(0)-x_center)*sin(alpha_ellipse)+(X_IP(1)-y_center)*cos(alpha_ellipse),2)/(y_axis*y_axis);
// //            //std::cout<<"Check ellipse: " << check_ellipse_ip <<"\n";
// //			if(check_ellipse_ip<=1.){
// //				if(X_IP(2)>=z_axis){
// //					// inside cylinder
// //                    //a0x = frand(-1,1.);
// //                    //a0y = frand(-1,1.);
// //                    //a0z = 0.;
// //                    //a0_wound << a0x, a0y, a0z;
// //                    //a0_wound = a0_wound/sqrt(a0_wound.dot(a0_wound));
// //                    std::cout<<"IP node: "<<IP_size*elemi+ip<<"\n";
// //                    ip_phi0[elemi*IP_size+ip] = phif0_wound;
// //					//ip_phi0[elemi*IP_size+ip] = 1. - phif0_wound*(X_IP(2)-z_axis)/(20-z_axis);
// //                    //ip_phi0[elemi*IP_size+ip] = phif0_wound*(X_IP(2)-z_axis)/(20-z_axis);
// //                    ip_a00[elemi*IP_size+ip] = a0_wound;
// //                    ip_s00[elemi*IP_size+ip] = s0_wound;
// //                    ip_n00[elemi*IP_size+ip] = n0_wound;
// //					ip_kappa0[elemi*IP_size+ip] = kappa0_wound;
// //					ip_lamda0[elemi*IP_size+ip] = lamda0_wound;
// //				}
// //			}
//             //-----------------//
//             // Alternative version with an ellipse instead of a cylinder
//             double check_ellipsoid_ip = pow((X_IP(0)-x_center),2)/(x_axis*x_axis) + pow((X_IP(1)-y_center),2)/(y_axis*y_axis);
//             if(check_ellipsoid_ip<=1.){
//                 // inside ellipsoid
//                 //a0x = frand(-1,1.);
//                 //a0y = frand(-1,1.);
//                 //a0z = 0.;
//                 //a0_wound << a0x, a0y, a0z;
//                 //a0_wound = a0_wound/sqrt(a0_wound.dot(a0_wound));
//                 std::cout<<"IP node: "<<IP_size*elemi+ip<<"\n";
//                 ip_phi0[elemi*IP_size+ip] = phif0_wound;
//                 //ip_phi0[elemi*IP_size+ip] = 1. - phif0_wound*(X_IP(2)-z_axis)/(20-z_axis);
//                 //ip_phi0[elemi*IP_size+ip] = phif0_wound*(X_IP(2)-z_axis)/(20-z_axis);
//                 ip_a00[elemi*IP_size+ip] = a0_wound;
//                 ip_s00[elemi*IP_size+ip] = s0_wound;
//                 ip_n00[elemi*IP_size+ip] = n0_wound;
//                 ip_kappa0[elemi*IP_size+ip] = kappa0_wound;
//                 ip_lamda0[elemi*IP_size+ip] = lamda0_wound;
//             }
//             //-----------------//
// 		}
// 	}
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
	myTissue.time_final = 7*24; // in hours
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
	std::string filename = "cylindDura"+ss.str()+"_";

	//----------------------------------------------------------//
	// SOLVE
//    sparseLoadSolver(myTissue, filename, 1,save_node,save_ip);
//
//    std::cout<<"filling dofs...\n";
//    fillDOFmap(myTissue);
//    std::cout<<"going to eval jacobians...\n";
//    evalElemJacobians(myTissue);
//    std::cout<<"going to eval surface jacobians...\n";
//    //evalElemJacobiansSurface(myTissue);
//    //
//    //print out the Jacobians
//    std::cout<<"element jacobians\nJacobians= ";
//    std::cout<<myTissue.elem_jac_IP.size()<<"\n";
//    for(int i=0;i<myTissue.elem_jac_IP.size();i++){
//        std::cout<<"element: "<<i<<"\n";
//        for(int j=0;j<IP_size;j++){
//            std::cout<<"ip; "<<j<<"\n"<<myTissue.elem_jac_IP[i][j]<<"\n";
//        }
//    }
//    // print out the forward dof map
//    std::cout<<"Total :"<<myTissue.n_dof<<" dof\n";
//    for(int i=0;i<myTissue.dof_fwd_map_x.size();i++){
//        std::cout<<"x node*3+coord: "<<i<<", dof: "<<myTissue.dof_fwd_map_x[i]<<"\n";
//    }
//    for(int i=0;i<myTissue.dof_fwd_map_rho.size();i++){
//        std::cout<<"rho node: "<<i<<", dof: "<<myTissue.dof_fwd_map_rho[i]<<"\n";
//    }
//    for(int i=0;i<myTissue.dof_fwd_map_c.size();i++){
//        std::cout<<"c node: "<<i<<", dof: "<<myTissue.dof_fwd_map_c[i]<<"\n";
//    }

    sparseWoundSolver(myTissue, filename, 5,save_node,save_ip);
	

	return 0;	
}
