// last date modified: Sept, 16, 2025
/*
	Pre and Post processing functions
	Solver functions
	Struct and classes for the problem definition
*/

#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <stdexcept> 
#include <cmath>
#include "file_io.h"
#include "element_functions.h"
#include "solver.h"
#include "myMeshGenerator.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <Eigen/OrderingMethods>



//
// read in the Abaqus file and generate the mesh and fill in
void readAbaqusInput(const char* filename,tissue &myTissue)
{
	// READ NODES
	std::vector<Vector3d> node_X; node_X.clear();
	std::ifstream myfile(filename);
	std::string line;
	std::string keyword_node = "*Node";
	if (myfile.is_open())
	{
		// read in until you find the keyword *NODE
		while ( getline (myfile,line) )
    	{
      		// check for the keyword
      		std::size_t found = line.find(keyword_node);
			if (found!=std::string::npos)
      		{
      			// found the beginning of the nodes, so keep looping until you get '*'
      			while ( getline (myfile,line) )
      			{
      				if(line[0]=='*'){break;}
      				std::vector<std::string> strs;
					boost::split(strs,line,boost::is_any_of(","));
					node_X.push_back(Vector3d(std::stod(strs[1]),std::stod(strs[2]),std::stod(strs[3])));
      			}
      		}
    	}
    }
    myfile.close();
    myTissue.node_X = node_X;

	// READ ELEMENTS
	std::vector<std::vector<int> > LineQuadri; LineQuadri.clear();
	myfile.open(filename);
	std::string keyword_element = "*Element";
	if (myfile.is_open())
	{
		// read in until you find the keyword *NODE
		while ( getline (myfile,line) )
    	{
      		// check for the keyword
      		std::size_t found = line.find(keyword_element);
			if (found!=std::string::npos)
      		{
      			// found the beginning of the nodes, so keep looping until you get '*'
      			while ( getline (myfile,line) )
      			{
      				if(line[0]=='*'){break;}
      				// the nodes for the C3D8 element
      				// also remember that abaqus has node numbering starting in 1
      				std::string line2;
      				getline (myfile,line2);
      				std::vector<std::string> strs1;
					boost::split(strs1,line,boost::is_any_of(","));
					std::vector<int> elemi;elemi.clear();
					//std::cout<<"line1: ";
					// CHECK //
					for(int nodei=1;nodei<strs1.size()-1;nodei++)
					{
						//std::cout<<strs1[nodei]<<",";
						elemi.push_back(std::stoi(strs1[nodei])-1);
					}
					//std::cout<<"\n";
					LineQuadri.push_back(elemi);
      			}
      		}
    	}
    }
    myfile.close();
    myTissue.vol_elem_connectivity = LineQuadri;

	// in addition to the connectivity and the nodes, some other things
	myTissue.n_node = node_X.size();
	myTissue.n_vol_elem = LineQuadri.size();

}

void readAbaqusInputLikeCOMSOL(const std::string& filename, HexMesh &myMesh)
{
    std::vector<Eigen::Vector3d> nodes;
    std::vector<std::vector<int>> elements;

    std::ifstream myfile(filename);
    std::string line;

    // ---------------------------
    // READ NODES
    // ---------------------------
    while (std::getline(myfile, line)) {
        if (line.find("*Node") != std::string::npos) {
            while (std::getline(myfile, line)) {
                if (!line.empty() && line[0] == '*') break;
                if (line.empty()) continue;

                std::vector<std::string> strs;
                boost::split(strs, line, boost::is_any_of(","));

                // Abaqus: id, x, y, z
                double x = std::stod(strs[1]);
                double y = std::stod(strs[2]);
                double z = std::stod(strs[3]);
                nodes.push_back(Eigen::Vector3d(x,y,z));
            }
        }
    }
    myfile.close();

    // ---------------------------
    // READ ELEMENTS
    // ---------------------------
    myfile.open(filename);
    while (std::getline(myfile, line)) {
        if (line.find("*Element") != std::string::npos) {

            while (std::getline(myfile, line)) {
                if (!line.empty() && line[0] == '*') break;
                if (line.empty()) continue;

                std::vector<std::string> strs;
                boost::split(strs, line, boost::is_any_of(","));

                // Expect: elemId, n1,n2,n3,n4,n5,n6,n7,n8
                // (sometimes there are spaces; stod/stoi handle it)

                std::vector<int> n(8);
                for (int i=0;i<8;i++) n[i] = std::stoi(strs[i+1]) - 1; // 0-based

                // ---- IMPORTANT ----
                // Reorder to match your COMSOL reader convention:
                // COMSOL hex mapping: [0,1,3,2,4,5,7,6]
                // std::vector<int> ele(8);
                // ele[0] = n[0];
                // ele[1] = n[1];
                // ele[2] = n[3];
                // ele[3] = n[2];
                // ele[4] = n[4];
                // ele[5] = n[5];
                // ele[6] = n[7];
                // ele[7] = n[6];

                elements.push_back(n);
            }
        }
    }
    myfile.close();

    // ---------------------------
    // Fill HexMesh
    // ---------------------------
    myMesh.nodes = nodes;
    myMesh.elements = elements;
    // myMesh.boundary_flag = boundary_flag;

    myMesh.n_nodes = (int)nodes.size();
    myMesh.n_elements = (int)elements.size();

    // If you need these later, set empty for now:
    myMesh.surface_elements.clear();
    myMesh.surface_boundary_flag.clear();
    myMesh.n_surf_elements = 0;
}


//---------------------------------------//
// READ MY OWN FILE
//---------------------------------------//
tissue readTissue(const char* filename)
{
	// initialize the structure
	tissue myTissue;
	
	std::ifstream myfile(filename);
	std::string line;
	if (myfile.is_open())
	{
		
		// time
		getline (myfile,line);
		std::stringstream ss0(line);
		ss0>>myTissue.time;
		
		// time final
		getline (myfile,line);
		std::stringstream ss1(line);
		ss1>>myTissue.time_final;
		
		// time step
		getline (myfile,line);
		std::stringstream ss2(line);
		ss2>>myTissue.time_step;
		
		// tol
		getline (myfile,line);
		std::stringstream ss3(line);
		ss3>>myTissue.tol;
		
		// max iter
		getline (myfile,line);
		std::stringstream ss4(line);
		ss4>>myTissue.max_iter;
		
		// global parameters
		int n_global_parameters;
		getline (myfile,line);
		std::stringstream ss5(line);
		ss5>>n_global_parameters;
		std::vector<double> global_parameters(n_global_parameters,0.);
		getline (myfile,line);
		std::stringstream ss6(line);
		for(int i=0;i<n_global_parameters;i++){
			ss6>>global_parameters[i];
		}
		myTissue.global_parameters = global_parameters;
		
		// local parameters
		int n_local_parameters;
		getline (myfile,line);
		std::stringstream ss7(line);
		ss7>>n_local_parameters;
		std::vector<double> local_parameters(n_local_parameters,0.);
		getline (myfile,line);
		std::stringstream ss8(line);
		for(int i=0;i<n_local_parameters;i++){
			ss8>>local_parameters[i];
		}
		myTissue.local_parameters = local_parameters;
		
		// n_node
		getline (myfile,line);
		std::stringstream ss9(line);
		ss9>>myTissue.n_node;
		
		// n_quadri
		getline (myfile,line);
		std::stringstream ss10(line);
		ss10>>myTissue.n_vol_elem;
		
		// n_IP
		getline (myfile,line);
		std::stringstream ss11(line);
		ss11>>myTissue.n_IP;
		if(myTissue.n_IP>8*myTissue.n_vol_elem || myTissue.n_IP<8*myTissue.n_vol_elem )
		{std::cout<<"number of integration points and elements don't match\n";myTissue.n_IP = 8*myTissue.n_vol_elem;}
		
		// n_dof
		getline (myfile,line);
		std::stringstream ss12(line);
		ss12>>myTissue.n_dof;
		
		// LineQuadri
		std::vector<int> temp_elem(8,0);
		std::vector<std::vector<int > > LineQuadri(myTissue.n_vol_elem,temp_elem);
		myTissue.vol_elem_connectivity = LineQuadri;
		for(int i=0;i<myTissue.vol_elem_connectivity.size();i++){
			getline (myfile,line);
			std::stringstream ss13(line);
			ss13>>myTissue.vol_elem_connectivity[i][0]; ss13>>myTissue.vol_elem_connectivity[i][1]; ss13>>myTissue.vol_elem_connectivity[i][2]; ss13>>myTissue.vol_elem_connectivity[i][3]; ss13>>myTissue.vol_elem_connectivity[i][4]; ss13>>myTissue.vol_elem_connectivity[i][5]; ss13>>myTissue.vol_elem_connectivity[i][6]; ss13>>myTissue.vol_elem_connectivity[i][7];
		}
		
		// boundaryNodes
		std::vector<int> boundaryNodes(myTissue.n_node,0);
		myTissue.boundaryNodes = boundaryNodes;
		for(int i=0;i<myTissue.boundaryNodes.size();i++){
			getline (myfile,line);
			std::stringstream ss14(line);
			ss14>>myTissue.boundaryNodes[i];
		}
		
		// node_X 
		std::vector<Vector3d> node_X(myTissue.n_node,Vector3d(0,0,0));
		myTissue.node_X = node_X;
		for(int i=0;i<myTissue.node_X.size();i++){
			getline (myfile,line);
			std::stringstream ss15(line);
			ss15>>myTissue.node_X[i](0);ss15>>myTissue.node_X[i](1);ss15 >> myTissue.node_X[i](2);
		}
		
		// node_rho_0
		std::vector<double> node_rho_0(myTissue.n_node,0.0);
		myTissue.node_rho_0 = node_rho_0;
		for(int i=0;i<myTissue.node_rho_0.size();i++){
			getline (myfile,line);
			std::stringstream ss16(line);
			ss16>>myTissue.node_rho_0[i];
		}
		
		// node_c_0
		std::vector<double> node_c_0(myTissue.n_node,0.0);
		myTissue.node_c_0 = node_c_0;
		for(int i=0;i<myTissue.node_c_0.size();i++){
			getline (myfile,line);
			std::stringstream ss17(line);
			ss17>>myTissue.node_c_0[i];
		}
		
		// ip_phif_0
		std::vector<double> ip_phif_0(myTissue.n_IP,0.0);
		myTissue.ip_phif_0 = ip_phif_0;
		for(int i=0;i<myTissue.ip_phif_0.size();i++){
			getline (myfile,line);
			std::stringstream ss18(line);
			ss18>>myTissue.ip_phif_0[i];
		}
		
		// ip_a0_0
		std::vector<Vector3d> ip_a0_0(myTissue.n_IP,Vector3d(0,0,0));
		myTissue.ip_a0_0 = ip_a0_0;
		for(int i=0;i<myTissue.ip_a0_0.size();i++){
			getline (myfile,line);
			std::stringstream ss19(line);
			ss19>>myTissue.ip_a0_0[i](0);ss19>>myTissue.ip_a0_0[i](1);ss19>>myTissue.ip_a0_0[i](2);
		}
				
		// ip_kappa_0
		std::vector<double> ip_kappa_0(myTissue.n_IP,0.0);
		myTissue.ip_kappa_0 = ip_kappa_0;
		for(int i=0;i<myTissue.ip_kappa_0.size();i++){
			getline (myfile,line);
			std::stringstream ss20(line);
			ss20>>myTissue.ip_kappa_0[i];
		}
		
		// ip_lamdaP_0
		std::vector<Vector3d> ip_lamdaP_0(myTissue.n_IP,Vector3d(0,0));
		myTissue.ip_lamdaP_0 = ip_lamdaP_0;
		for(int i=0;i<myTissue.ip_lamdaP_0.size();i++){
			getline (myfile,line);
			std::stringstream ss21(line);
			ss21>>myTissue.ip_lamdaP_0[i](0);ss21>>myTissue.ip_lamdaP_0[i](1);
		}
		
		// node_x 
		std::vector<Vector3d> node_x(myTissue.n_node,Vector3d(0,0,0));
		myTissue.node_x = node_x;
		for(int i=0;i<myTissue.node_x.size();i++){
			getline (myfile,line);
			std::stringstream ss22(line);
			ss22>>myTissue.node_x[i](0);ss22>>myTissue.node_x[i](1);ss22>>myTissue.node_x[i](1);
		}
		
		// node_rho
		myTissue.node_rho = myTissue.node_rho_0;
		
		// node_c
		myTissue.node_c = myTissue.node_c_0;
		
		// ip_phif
		myTissue.ip_phif = myTissue.ip_phif_0;
		
		// ip_a0
		myTissue.ip_a0 = myTissue.ip_a0_0;
		
		// ip_kappa
		myTissue.ip_kappa = myTissue.ip_kappa_0;
		
		// ip_lamdaP
		myTissue.ip_lamdaP = myTissue.ip_lamdaP_0;
		
		// eBC_x
		int n_eBC_x;
		int dofx;
		double dofx_value;
		getline (myfile,line);
		std::stringstream ss23(line);
		ss23>>n_eBC_x;
		myTissue.eBC_x.clear();
		for(int i=0;i<n_eBC_x;i++){
			getline (myfile,line);
			std::stringstream ss24(line);
			ss24>>dofx;ss24>>dofx_value;
			myTissue.eBC_x.insert ( std::pair<int,double>(dofx,dofx_value) ); 
		}

		// eBC_rho
		int n_eBC_rho;
		int dofrho;
		double dofrho_value;
		getline (myfile,line);
		std::stringstream ss25(line);
		ss25>>n_eBC_rho;
		myTissue.eBC_rho.clear();
		for(int i=0;i<n_eBC_rho;i++){
			getline (myfile,line);
			std::stringstream ss26(line);
			ss26>>dofrho;ss26>>dofrho_value;
			myTissue.eBC_rho.insert ( std::pair<int,double>(dofrho,dofrho_value) ); 
		}

		// eBC_c
		int n_eBC_c;
		int dofc;
		double dofc_value;
		getline (myfile,line);
		std::stringstream ss27(line);
		ss27>>n_eBC_c;
		myTissue.eBC_c.clear();
		for(int i=0;i<n_eBC_c;i++){
			getline (myfile,line);
			std::stringstream ss28(line);
			ss28>>dofc;ss28>>dofc_value;
			myTissue.eBC_c.insert ( std::pair<int,double>(dofc,dofc_value) ); 
		}

		// nBC_x

		int n_nBC_x;
		double forcex_value;
		getline (myfile,line);
		std::stringstream ss29(line);
		ss29>>n_nBC_x;
		myTissue.nBC_x.clear();
		for(int i=0;i<n_nBC_x;i++){
			getline (myfile,line);
			std::stringstream ss30(line);
			ss30>>dofx;ss30>>forcex_value;
			myTissue.nBC_x.insert ( std::pair<int,double>(dofx,forcex_value) ); 
		}

		// nBC_rho
		int n_nBC_rho;
		double forcerho_value;
		getline (myfile,line);
		std::stringstream ss31(line);
		ss31>>n_nBC_rho;
		myTissue.nBC_rho.clear();
		for(int i=0;i<n_nBC_rho;i++){
			getline (myfile,line);
			std::stringstream ss32(line);
			ss32>>dofrho;ss32>>forcerho_value;
			myTissue.nBC_rho.insert ( std::pair<int,double>(dofrho,forcerho_value) ); 
		}

		// nBC_c
		int n_nBC_c;
		double forcec_value;
		getline (myfile,line);
		std::stringstream ss33(line);
		ss33>>n_nBC_c;
		myTissue.nBC_c.clear();
		for(int i=0;i<n_nBC_c;i++){
			getline (myfile,line);
			std::stringstream ss34(line);
			ss34>>dofc;ss34>>forcec_value;
			myTissue.nBC_c.insert ( std::pair<int,double>(dofc,forcec_value) ); 
		}

		// dof_fwd_map_x
		int n_dof_fwd_map_x;
		getline (myfile,line);
		std::stringstream ss35(line);
		ss35>>n_dof_fwd_map_x;
		std::vector<int> dof_fwd_map_x(n_dof_fwd_map_x,-1);
		myTissue.dof_fwd_map_x = dof_fwd_map_x;
		for(int i=0;myTissue.dof_fwd_map_x.size();i++){
			getline (myfile,line);
			std::stringstream ss36(line);
			ss36>>myTissue.dof_fwd_map_x[i];
		}
	
		// dof_fwd_map_rho
		int n_dof_fwd_map_rho;
		getline (myfile,line);
		std::stringstream ss37(line);
		ss37>>n_dof_fwd_map_rho;
		std::vector<int> dof_fwd_map_rho(n_dof_fwd_map_rho,-1);
		myTissue.dof_fwd_map_rho = dof_fwd_map_rho;
		for(int i=0;myTissue.dof_fwd_map_rho.size();i++){
			getline (myfile,line);
			std::stringstream ss38(line);
			ss38>>myTissue.dof_fwd_map_rho[i];
		}
		
		// dof_fwd_map_c
		int n_dof_fwd_map_c;
		getline (myfile,line);
		std::stringstream ss39(line);
		ss39>>n_dof_fwd_map_c;
		std::vector<int> dof_fwd_map_c(n_dof_fwd_map_c,-1);
		myTissue.dof_fwd_map_c = dof_fwd_map_c;
		for(int i=0;myTissue.dof_fwd_map_c.size();i++){
			getline (myfile,line);
			std::stringstream ss40(line);
			ss40>>myTissue.dof_fwd_map_c[i];
		}
	
		// dof_inv_map
		int n_dof_inv_map;
		getline (myfile,line);
		std::stringstream ss41(line);
		ss41>>n_dof_inv_map;
		std::vector<int> temp_inv_dof(2,0);
		std::vector<std::vector<int> > dof_inv_map(n_dof_inv_map,temp_inv_dof);
		myTissue.dof_inv_map = dof_inv_map;
		for(int i=0;myTissue.dof_inv_map.size();i++){
			getline (myfile,line);
			std::stringstream ss42(line);
			ss42>>myTissue.dof_inv_map[i][0];ss42>>myTissue.dof_inv_map[i][1];
		}		
	}
	myfile.close();
	evalElemJacobians(myTissue);
	return myTissue;
}


//---------------------------------------//
// WRITE OUT MY OWN FILE
//---------------------------------------//
//
void writeTissue(tissue &myTissue, const char* filename,double time)
{
	std::ofstream savefile(filename);
	if (!savefile) {
		throw std::runtime_error("Unable to open output file.");
	}
	savefile<<time<<"\n";
	savefile<<myTissue.time_final<<"\n";
	savefile<<myTissue.time_step<<"\n";
	savefile<<myTissue.tol<<"\n";
	savefile<<myTissue.max_iter<<"\n";	
	savefile<<myTissue.global_parameters.size()<<"\n";
	for(int i=0;i<myTissue.global_parameters.size();i++){
		savefile<<myTissue.global_parameters[i]<<" ";
	}
	savefile<<"\n";
	savefile<<myTissue.local_parameters.size()<<"\n";
	for(int i=0;i<myTissue.local_parameters.size();i++){
		savefile<<myTissue.local_parameters[i]<<" ";
	}
	savefile<<"\n";
	savefile<<myTissue.n_node<<"\n";
	savefile<<myTissue.n_vol_elem<<"\n";
	savefile<<myTissue.n_IP<<"\n";
	savefile<<myTissue.n_dof<<"\n";
	// Write however many nodes the element actually has. This used to be
	// hard-coded to 8, which read past the end of 4-node tet connectivity and
	// emitted uninitialized memory (differing between runs of the same binary).
	for(int i=0;i<myTissue.vol_elem_connectivity.size();i++){
		for(int j=0;j<myTissue.vol_elem_connectivity[i].size();j++){
			savefile<<myTissue.vol_elem_connectivity[i][j];
			if(j+1<myTissue.vol_elem_connectivity[i].size()) savefile<<" ";
		}
		savefile<<"\n";
	}
	for(int i=0;i<myTissue.boundaryNodes.size();i++){
		savefile<<myTissue.boundaryNodes[i]<<"\n";
	}
	for(int i=0;i<myTissue.node_X.size();i++){
		savefile<<myTissue.node_X[i](0)<<" "<<myTissue.node_X[i](1)<<" "<<myTissue.node_X[i](2)<<"\n";
	}
	for(int i=0;i<myTissue.node_rho_0.size();i++){
		savefile<<myTissue.node_rho_0[i]<<"\n";
	}
	for(int i=0;i<myTissue.node_c_0.size();i++){
		savefile<<myTissue.node_c_0[i]<<"\n";
	}
	for(int i=0;i<myTissue.node_alpha_0.size();i++){
		savefile<<myTissue.node_alpha_0[i]<<"\n";
	}
	for(int i=0;i<myTissue.ip_phif_0.size();i++){
		savefile<<myTissue.ip_phif_0[i]<<"\n";
	}
	for(int i=0;i<myTissue.ip_a0_0.size();i++){
		savefile<<myTissue.ip_a0_0[i](0)<<" "<<myTissue.ip_a0_0[i](1)<<" "<<myTissue.ip_a0_0[i](2)<<"\n";
	}
	for(int i=0;i<myTissue.ip_kappa_0.size();i++){
		savefile<<myTissue.ip_kappa_0[i]<<"\n";
	}	
	// All three components (this used to drop the through-thickness one).
	for(int i=0;i<myTissue.ip_lamdaP_0.size();i++){
		savefile<<myTissue.ip_lamdaP_0[i](0)<<" "<<myTissue.ip_lamdaP_0[i](1)<<" "<<myTissue.ip_lamdaP_0[i](2)<<"\n";
	}
	for(int i=0;i<myTissue.node_x.size();i++){
		savefile<<myTissue.node_x[i](0)<<" "<<myTissue.node_x[i](1)<<" "<<myTissue.node_x[i](2)<<"\n";
	}
	std::map<int,double>::iterator it_map_BC;
	savefile<<myTissue.eBC_x.size()<<"\n";
	for(it_map_BC = myTissue.eBC_x.begin(); it_map_BC != myTissue.eBC_x.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.eBC_rho.size()<<"\n";
	for(it_map_BC = myTissue.eBC_rho.begin(); it_map_BC != myTissue.eBC_rho.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.eBC_c.size()<<"\n";
	for(it_map_BC = myTissue.eBC_c.begin(); it_map_BC != myTissue.eBC_c.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.nBC_x.size()<<"\n";
	for(it_map_BC = myTissue.nBC_x.begin(); it_map_BC != myTissue.nBC_x.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.nBC_rho.size()<<"\n";
	for(it_map_BC = myTissue.nBC_rho.begin(); it_map_BC != myTissue.nBC_rho.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.nBC_c.size()<<"\n";
	for(it_map_BC = myTissue.nBC_c.begin(); it_map_BC != myTissue.nBC_c.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.dof_fwd_map_x.size()<<"\n";
	for(int i=0;i<myTissue.dof_fwd_map_x.size();i++){
		savefile<<myTissue.dof_fwd_map_x[i]<<"\n";
	}
	savefile<<myTissue.dof_fwd_map_rho.size()<<"\n";
	for(int i=0;i<myTissue.dof_fwd_map_rho.size();i++){
		savefile<<myTissue.dof_fwd_map_rho[i]<<"\n";
	}
	savefile<<myTissue.dof_fwd_map_c.size()<<"\n";
	for(int i=0;i<myTissue.dof_fwd_map_c.size();i++){
		savefile<<myTissue.dof_fwd_map_c[i]<<"\n";
	}
	savefile<<myTissue.dof_inv_map.size()<<"\n";
	for(int i=0;i<myTissue.dof_inv_map.size();i++){
		savefile<<myTissue.dof_inv_map[i][0]<<" "<<myTissue.dof_inv_map[i][1]<<"\n";
	}
	savefile.close();
}


//---------------------------------------//
// READ COMSOL
//---------------------------------------//
//
// read in the COMSOL file and generate the mesh and fill in
HexMesh readCOMSOLInput(const std::string& filename, const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution)
{
    //std::cout << filename;
    std::cout << "Importing a mesh of from COMSOL \n";
    std::vector<Vector3d> NODES;
    std::vector<std::vector<int> > ELEMENTS;
    std::vector<int> SURFACE_ZERO(0,0);
    std::vector<std::vector<int> > SURFACE_ELEMENTS; //(0,SURFACE_ZERO);
    std::vector<int> BOUNDARIES;
    std::vector<int> SURFACE_BOUNDARIES(0,0);
    // READ NODES
    std::ifstream myfile;
    myfile.open(filename.c_str());
    std::string keyword_node = "Mesh vertex coordinates";
    //std::cout << myfile.is_open();
    if (myfile.is_open()){
        // read in until you find the keyword *NODE
        for (std::string line; std::getline(myfile, line); ){
            //std::cout << line << std::endl;
            // check for the keyword
            std::size_t found = line.find(keyword_node);
            if (found!=std::string::npos){
                // found the beginning of the nodes, so keep looping until you get '*'
                for ( std::string node_line; std::getline(myfile, node_line); ){
                    //std::cout << node_line << std::endl;
                    // std::cout << node_line.size() << std::endl;
                    if(node_line.size() == 1 || node_line.empty()){
                        break;
                    }
                    std::vector<std::string> strs;
                    boost::split(strs,node_line,boost::is_any_of(" "));
                    Vector3d nodei = Vector3d(std::stod(strs[0]),std::stod(strs[1]),std::stod(strs[2]));
                    NODES.push_back(nodei);
					double tol_b = 1e-4;

                    //-------------//
                    // left, right, north, south are fixed and top and bottom faces are free
                    // The displacement along the z-direction in the point load case is fixed to given value.
                    // hence, this also comes under the Dirichlet BCn
                    // And we need to fix rho and c values for each node. So I need to include those in the Ebc_rho and Ebc_c
                    //-------------//
                    // Be careful. Make sure you remember which nodes are part of which face
                    // Apply Dirichlet BCs first, so the corner nodes become part of those faces

                    // for cylinderical dura wound mechanics
                    if(nodei(2)<=(hexDimensions[4]+tol_b)) { // x bottom
                        BOUNDARIES.push_back(1);
						std::cout<<"bottom"<<std::endl;
                    }
                    else if(nodei(2)>=(hexDimensions[5]-tol_b)) { // x top
                        BOUNDARIES.push_back(1);
						std::cout<<"top"<<std::endl;
                    }
                    else{
                        BOUNDARIES.push_back(0);
						std::cout<<"notBC"<<std::endl;
                    }
                }
            }
        }
    }
    else{
        std::cout<<"\nFailed to open file.\n";
    }
    myfile.close();

    // std::cout << "read the nodes"<< std::endl;


    // READ VOLUME ELEMENTS
    myfile.open(filename);
    std::string keyword_element = "4 # number of vertices per element";
    if (myfile.is_open()){
        // read in until you find the keyword ELEMENT
        for (std::string line; std::getline(myfile, line); ){
            // check for the keyword
            std::size_t found = line.find(keyword_element);
            if (found!=std::string::npos){
                // Skip two lines to get to element connectivity
                getline (myfile,line); getline (myfile,line);
                // found the beginning of the nodes, so keep looping until you get '*'
                for ( std::string element_line; std::getline(myfile, element_line); ){
                    if(element_line.size() == 1 || element_line.empty()){
                        break;
                    }
                    //std::cout << element_line << std::endl;
                    std::vector<std::string> strs1;
                    boost::split(strs1,element_line,boost::is_any_of(" "));
                    std::vector<int> elemi; elemi.clear();
                    if(strs1.size() == 9){
                        // Push nodes. COMSOL has a weird format, so this makes it correct for Paraview
                        // the nodes for the C3D8 hex element
                        elemi.push_back(std::stoi(strs1[0]));
                        elemi.push_back(std::stoi(strs1[1]));
                        elemi.push_back(std::stoi(strs1[3]));
                        elemi.push_back(std::stoi(strs1[2]));
                        elemi.push_back(std::stoi(strs1[4]));
                        elemi.push_back(std::stoi(strs1[5]));
                        elemi.push_back(std::stoi(strs1[7]));
                        elemi.push_back(std::stoi(strs1[6]));
                    }
                    else if (strs1.size() == 5){
                        // the nodes for the C3D4 tet element
                        elemi.push_back(std::stoi(strs1[0]));
                        elemi.push_back(std::stoi(strs1[1]));
                        elemi.push_back(std::stoi(strs1[2]));
                        elemi.push_back(std::stoi(strs1[3]));
                    }
                    //std::cout<<"\n";
                    ELEMENTS.push_back(elemi);
                }
            }
        }
    }
    myfile.close();

    // std::cout << "read the volume elements"<< std::endl;

	// READ SURFACE ELEMENTS
    myfile.open(filename);
    keyword_element = "3 # number of vertices per element";
    if (myfile.is_open()){
        // read in until you find the keyword ELEMENT
        for (std::string line; std::getline(myfile, line); ){
            // check for the keyword
            std::size_t found = line.find(keyword_element);
            if (found!=std::string::npos){
                // Skip two lines to get to element connectivity
                getline (myfile,line); getline (myfile,line);
                
				// 
                for ( std::string element_line; std::getline(myfile, element_line); ){
                    if(element_line.size() == 1 || element_line.empty()){
                        break;
                    }
                    //std::cout << element_line << std::endl;
                    std::vector<std::string> strs1;
                    boost::split(strs1,element_line,boost::is_any_of(" "));
                    std::vector<int> elemi; elemi.clear();
                    if(strs1.size() == 4){
                        
                        // It has 3 nodes per element on the surface.
                        elemi.push_back(std::stoi(strs1[0]));
                        elemi.push_back(std::stoi(strs1[1]));
                        elemi.push_back(std::stoi(strs1[2]));
                    }
                    else if (strs1.size() == 5){
                        // FOR QUAD
                        elemi.push_back(std::stoi(strs1[0]));
                        elemi.push_back(std::stoi(strs1[1]));
                        elemi.push_back(std::stoi(strs1[2]));
                        elemi.push_back(std::stoi(strs1[3]));
                    }
                    //std::cout<<"\n";
                    SURFACE_ELEMENTS.push_back(elemi);
                }
            }
        }
    }
    myfile.close();
    std::cout << "read the surface elements"<< std::endl;

    //std::vector<int> BOUNDARIES(NODES.size(),0);
    // READ SURFACE ELEMENTS
    /*myfile.open(filename);
    std::string keyword_surface_element = "3 # number of nodes per element";
    if (myfile.is_open()){
        // read in until you find the keyword ELEMENT
        for (std::string line; std::getline(myfile, line); ){
            // check for the keyword
            std::size_t found = line.find(keyword_surface_element);
            if (found!=std::string::npos){
                // Skip two lines to get to element connectivity
                getline (myfile,line); getline (myfile,line);
                // found the beginning of the nodes, so keep looping until you get '*'
                for ( std::string element_line; std::getline(myfile, element_line); ){
                    if(element_line.size() == 1 || element_line.empty()){
                        break;
                    }
                    // std::cout << element_line << std::endl;
                    std::vector<std::string> strs1;
                    boost::split(strs1,element_line,boost::is_any_of(" "));
                    std::vector<int> elemi; elemi.clear();
                    if (strs1.size() == 5){
                        // the nodes for the C2D4 surface element
                        elemi.push_back(std::stoi(strs1[0]));
                        elemi.push_back(std::stoi(strs1[1]));
                        elemi.push_back(std::stoi(strs1[2]));
                        elemi.push_back(std::stoi(strs1[3]));
                    }
                    //std::cout<<"\n";
                    //double distance1 = sqrt(pow((NODES[elemi[0]](0) - hexDimensions[1]/2),2) + pow((NODES[elemi[0]](1) - hexDimensions[3]/2),2));
                    //double distance2 = sqrt(pow((NODES[elemi[1]](0) - hexDimensions[1]/2),2) + pow((NODES[elemi[1]](1) - hexDimensions[3]/2),2));
                    //double distance3 = sqrt(pow((NODES[elemi[2]](0) - hexDimensions[1]/2),2) + pow((NODES[elemi[2]](1) - hexDimensions[3]/2),2));
                    //double distance4 = sqrt(pow((NODES[elemi[3]](0) - hexDimensions[1]/2),2) + pow((NODES[elemi[3]](1) - hexDimensions[3]/2),2));
                    //double tolerance = 1;
                    // Keep track of which boundary we are on
                    if(BOUNDARIES[elemi[0]] == 1 && BOUNDARIES[elemi[1]] == 1 && BOUNDARIES[elemi[2]] == 1 && BOUNDARIES[elemi[3]] == 1){
                    //if(distance1 >= hexDimensions[1]/2 - tolerance && distance2 >= hexDimensions[1]/2 - tolerance && distance3 >= hexDimensions[1]/2 - tolerance && distance4 >= hexDimensions[1]/2 - tolerance){
                        SURFACE_BOUNDARIES.push_back(1);
                        SURFACE_ELEMENTS.push_back(elemi);
//                        for(int i : elemi){
//                            BOUNDARIES[i] = 1;
//                        }
                    }
                    else if(BOUNDARIES[elemi[0]] == 2 && BOUNDARIES[elemi[1]] == 2 && BOUNDARIES[elemi[2]] == 2 && BOUNDARIES[elemi[3]] == 2){
                    //else if(round(NODES[elemi[0]](2)) < hexDimensions[4]+1 && round(NODES[elemi[1]](2)) < hexDimensions[4]+1 && round(NODES[elemi[2]](2)) < hexDimensions[4]+1 && round(NODES[elemi[3]](2)) < hexDimensions[4]+1){
                        SURFACE_BOUNDARIES.push_back(2);
                        SURFACE_ELEMENTS.push_back(elemi);
//                        for(int i : elemi){
//                            BOUNDARIES[i] = 2;
//                        }
                    }
                    else if(BOUNDARIES[elemi[0]] == 6 && BOUNDARIES[elemi[1]] == 6 && BOUNDARIES[elemi[2]] == 6 && BOUNDARIES[elemi[3]] == 6){
                    //else if(round(NODES[elemi[0]](2)) == hexDimensions[5] && round(NODES[elemi[1]](2)) == hexDimensions[5] && round(NODES[elemi[2]](2)) == hexDimensions[5] && round(NODES[elemi[3]](2)) == hexDimensions[5]){
                        SURFACE_BOUNDARIES.push_back(6);
                        SURFACE_ELEMENTS.push_back(elemi);
//                        for(int i : elemi){
//                            BOUNDARIES[i] = 3;
//                        }
                    }
                    else{
                        // This is not a boundary element!
                        //SURFACE_BOUNDARIES.push_back(0);
//                        for(int i : elemi){
//                            BOUNDARIES[i] = 0;
//                        }
                    }
                }
            }
        }
    }
    myfile.close();*/

    HexMesh myMesh;
    myMesh.nodes = NODES;
    myMesh.elements = ELEMENTS;
    myMesh.surface_elements = SURFACE_ELEMENTS;
    myMesh.boundary_flag = BOUNDARIES;
    myMesh.surface_boundary_flag = SURFACE_BOUNDARIES;
    myMesh.n_nodes = NODES.size();
    myMesh.n_elements = ELEMENTS.size();
    myMesh.n_surf_elements = SURFACE_ELEMENTS.size();
    //std::cout << "\n NODES \n" << myMesh.n_nodes << "\n ELEMENTS \n" << myMesh.n_elements << "\n BOUNDARIES \n" << BOUNDARIES.size() << "\n";
    return myMesh;
}

//---------------------------------------//
// READ PARAVIEW
//---------------------------------------//
//
// read in the Paraview file and generate the mesh and fill in
HexMesh readParaviewInput(const std::string& filename, const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution)
{
    //std::cout << filename;
    std::cout << "Importing a mesh of from Paraview \n";
    std::vector<Vector3d> NODES;
    std::vector<std::vector<int> > ELEMENTS;
    std::vector<int> BOUNDARIES;
    // READ NODES
    std::ifstream myfile;
    myfile.open(filename.c_str());
    std::string keyword_node = "float";
    //std::cout << myfile.is_open();
    if (myfile.is_open()){
        // read in until you find the keyword *NODE
        for (std::string line; std::getline(myfile, line); ){
            //std::cout << line << std::endl;
            // check for the keyword
            std::size_t found = line.find(keyword_node);
            if (found!=std::string::npos){
                // found the beginning of the nodes, so keep looping until you get '*'
                for ( std::string node_line; std::getline(myfile, node_line); ){
                    //std::cout << node_line << std::endl;
                    //std::cout << node_line.size() << std::endl;
                    std::size_t foundend = node_line.find("METADATA");
                    if(node_line.size() == 1 || node_line.empty() || foundend!=std::string::npos){
                        break;
                    }
                    std::vector<std::string> strs;
                    boost::split(strs,node_line,boost::is_any_of(" "));
                    Vector3d nodei;
                    for(int i = 0; i < strs.size()/3; i++){
                        nodei = Vector3d(std::stod(strs[0+3*i]),std::stod(strs[1+3*i]),std::stod(strs[2+3*i]));
                        NODES.push_back(nodei);

                        //-------------//
                        // BOUNDARIES
                        //-------------//
                        // Be careful. Make sure you remember which nodes are part of which face
                        // Apply Dirichlet BCs first, so the corner nodes become part of those faces
                        //
                        if(round(nodei(0))==hexDimensions[0]){ // x = 0
                            BOUNDARIES.push_back(1);
                        }
                        else if (round(nodei(0))==hexDimensions[1]){ // x = end
                            BOUNDARIES.push_back(2);
                        }
                        else if(round(nodei(1))==hexDimensions[2]){ // y = 0
                            BOUNDARIES.push_back(3);
                        }
                        else if (round(nodei(1))==hexDimensions[3]){ // y = end
                            BOUNDARIES.push_back(4);
                        }
                        else if(round(nodei(2))==hexDimensions[4]){ // z = 0
                            BOUNDARIES.push_back(5);
                        }
                        else if (round(nodei(2))==hexDimensions[5]){ // z = end
                            BOUNDARIES.push_back(6);
                        }
                        else{
                            BOUNDARIES.push_back(0);
                        }
                    }
                }
            }
        }
    }
    else{
        std::cout<<"\nFailed to open file.\n";
    }
    myfile.close();


    // READ ELEMENTS
    myfile.open(filename);
    std::string keyword_element = "CELLS";
    if (myfile.is_open()){
        // read in until you find the keyword ELEMENT
        for (std::string line; std::getline(myfile, line); ){
            // check for the keyword
            std::size_t found = line.find(keyword_element);
            if (found!=std::string::npos){
                // found the beginning of the nodes, so keep looping until you get '*'
                for ( std::string element_line; std::getline(myfile, element_line); ){
                    if(element_line.size() == 1 || element_line.empty()){
                        break;
                    }
                    // std::cout << element_line << std::endl;
                    // the nodes for the C3D8 element
                    // also remember that abaqus has node numbering starting in 1
                    std::vector<std::string> strs1;
                    boost::split(strs1,element_line,boost::is_any_of(" "));
                    std::vector<int> elemi; elemi.clear();

                    // Push nodes. PARAVIEW starts at the second number
                    if(std::stoi(strs1[0]) == 8){
                        elemi.push_back(std::stoi(strs1[1]));
                        elemi.push_back(std::stoi(strs1[2]));
                        elemi.push_back(std::stoi(strs1[3]));
                        elemi.push_back(std::stoi(strs1[4]));
                        elemi.push_back(std::stoi(strs1[5]));
                        elemi.push_back(std::stoi(strs1[6]));
                        elemi.push_back(std::stoi(strs1[7]));
                        elemi.push_back(std::stoi(strs1[8]));
                    }
                    else if(std::stoi(strs1[0]) == 20){
                        elemi.push_back(std::stoi(strs1[1]));
                        elemi.push_back(std::stoi(strs1[2]));
                        elemi.push_back(std::stoi(strs1[3]));
                        elemi.push_back(std::stoi(strs1[4]));
                        elemi.push_back(std::stoi(strs1[5]));
                        elemi.push_back(std::stoi(strs1[6]));
                        elemi.push_back(std::stoi(strs1[7]));
                        elemi.push_back(std::stoi(strs1[8]));
                        elemi.push_back(std::stoi(strs1[9]));
                        elemi.push_back(std::stoi(strs1[10]));
                        elemi.push_back(std::stoi(strs1[11]));
                        elemi.push_back(std::stoi(strs1[12]));
                        elemi.push_back(std::stoi(strs1[13]));
                        elemi.push_back(std::stoi(strs1[14]));
                        elemi.push_back(std::stoi(strs1[15]));
                        elemi.push_back(std::stoi(strs1[16]));
                        elemi.push_back(std::stoi(strs1[17]));
                        elemi.push_back(std::stoi(strs1[18]));
                        elemi.push_back(std::stoi(strs1[19]));
                        elemi.push_back(std::stoi(strs1[20]));
                    }

                    else if(std::stoi(strs1[0]) == 4){
                        // linear tetrahedra
                        elemi.push_back(std::stoi(strs1[1]));
                        elemi.push_back(std::stoi(strs1[2]));
                        elemi.push_back(std::stoi(strs1[3]));
                        elemi.push_back(std::stoi(strs1[4]));
                    }
                    else if(std::stoi(strs1[0]) == 10){
                        // quadratic tetrahedra
                        elemi.push_back(std::stoi(strs1[1]));
                        elemi.push_back(std::stoi(strs1[2]));
                        elemi.push_back(std::stoi(strs1[3]));
                        elemi.push_back(std::stoi(strs1[4]));
                        elemi.push_back(std::stoi(strs1[5]));
                        elemi.push_back(std::stoi(strs1[6]));
                        elemi.push_back(std::stoi(strs1[7]));
                        elemi.push_back(std::stoi(strs1[8]));
                        elemi.push_back(std::stoi(strs1[9]));
                        elemi.push_back(std::stoi(strs1[10]));
                    }
                    //std::cout<<"\n";
                    ELEMENTS.push_back(elemi);
                }
            }
        }
    }
    myfile.close();

    HexMesh myMesh;
    myMesh.nodes = NODES;
    myMesh.elements = ELEMENTS;
    myMesh.boundary_flag = BOUNDARIES;
    myMesh.n_nodes = NODES.size();
    myMesh.n_elements = ELEMENTS.size();
    //std::cout << "\n NODES \n" << myMesh.n_nodes << "\n ELEMENTS \n" << myMesh.n_elements << "\n BOUNDARIES \n" << BOUNDARIES.size() << "\n";
    return myMesh;
}

//---------------------------------------//
// WRITE OUT A PARAVIEW FILE
//---------------------------------------//
void writeParaview(tissue &myTissue, const char* filename, const char* filename2)
{
    int elem_size = myTissue.vol_elem_connectivity[0].size();
	std::ofstream savefile(filename, std::ios::out);
    std::ofstream savefile2(filename2, std::ios::out);
	if (!savefile || !savefile2) {
		throw std::runtime_error("Unable to open output file.");
	}
	savefile<<"# vtk DataFile Version 2.0\nWound Healing Simulation\nASCII\nDATASET UNSTRUCTURED_GRID\n";
	savefile<<"POINTS "<<myTissue.node_x.size()<<" float\n";
    savefile2<<"# vtk DataFile Version 2.0\nWound Healing Simulation\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    savefile2<<"POINTS "<<myTissue.node_x.size()<<" float\n";
	for(int i=0;i<myTissue.node_x.size();i++)
	{
		savefile<<myTissue.node_x[i](0)<<" "<<myTissue.node_x[i](1)<<" "<<myTissue.node_x[i](2)<<"\n";
        savefile2<<myTissue.node_x[i](0)<<" "<<myTissue.node_x[i](1)<<" "<<myTissue.node_x[i](2)<<"\n";
	}
    savefile<<"\n";
    if(elem_size==27){
        savefile<<"CELLS "<<myTissue.vol_elem_connectivity.size()<<" "<<myTissue.vol_elem_connectivity.size()*(20+1)<<"\n";
        savefile2<<"CELLS "<<myTissue.vol_elem_connectivity.size()<<" "<<myTissue.vol_elem_connectivity.size()*(20+1)<<"\n";
    }
	else{
	    savefile<<"CELLS "<<myTissue.vol_elem_connectivity.size()<<" "<<myTissue.vol_elem_connectivity.size()*(elem_size+1)<<"\n";
        savefile2<<"CELLS "<<myTissue.vol_elem_connectivity.size()<<" "<<myTissue.vol_elem_connectivity.size()*(elem_size+1)<<"\n";
	}
	for(int i=0;i<myTissue.vol_elem_connectivity.size();i++)
	{
        if(elem_size==20){
            savefile<<elem_size;
            savefile<<" "<<myTissue.vol_elem_connectivity[i][0]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][0];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][1]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][1];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][2]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][2];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][3]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][3];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][4]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][4];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][5]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][5];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][6]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][6];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][7]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][7];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][8]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][8];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][9]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][9];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][10]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][10];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][11]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][11];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][12]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][12];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][13]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][13];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][14]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][14];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][15]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][15];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][16]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][16];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][17]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][17];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][18]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][18];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][19]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][19];
        }
        else if(elem_size==27){
            savefile<<20;
            savefile<<" "<<myTissue.vol_elem_connectivity[i][0]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][0];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][1]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][1];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][2]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][2];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][3]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][3];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][4]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][4];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][5]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][5];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][6]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][6];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][7]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][7];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][8]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][8];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][9]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][9];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][10]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][10];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][11]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][11];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][16]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][16];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][17]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][17];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][18]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][18];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][19]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][19];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][12]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][12];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][13]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][13];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][14]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][14];
            savefile<<" "<<myTissue.vol_elem_connectivity[i][15]; savefile2<<" "<<myTissue.vol_elem_connectivity[i][15];
        }
        else if(elem_size ==4 || elem_size==8 || elem_size == 10){
            savefile<<elem_size;
            savefile2<<elem_size;
            for (int j = 0; j < elem_size; j++) {
                savefile << " " << myTissue.vol_elem_connectivity[i][j];
                savefile2 << " " << myTissue.vol_elem_connectivity[i][j];
            }
        }
        else{
            throw std::runtime_error("Wrong number of nodes!");
        }
		savefile<<"\n";
        savefile2<<"\n";
	}
	savefile<<"\nCELL_TYPES "<<myTissue.vol_elem_connectivity.size()<<"\n";
    savefile2<<"\nCELL_TYPES "<<myTissue.vol_elem_connectivity.size()<<"\n";
	for(int i=0;i<myTissue.vol_elem_connectivity.size();i++)
	{
        if(elem_size==8){
            savefile<<"12\n";
            savefile2<<"12\n";
        }
        else if(elem_size==20 || elem_size==27){
            savefile<<"25\n";
            savefile2<<"25\n";
        }
        else if(elem_size==4){
            savefile<<"10\n";
            savefile2<<"10\n";
        }
        else if(elem_size==10){
            savefile<<"24\n";
            savefile2<<"24\n";
        }
        else{
            throw std::runtime_error("Wrong number of nodes!");
        }
	}
	
	// SAVE ATTRIBUTESa
	// up to four scalars I can plot...
	// first bring back from the integration points to the nodes
	// SCALARS
	std::vector<double> node_phi(myTissue.n_node,0);
	std::vector<double> node_kappa(myTissue.n_node,0);
	// VECTORS
    std::vector<Vector3d> node_a0(myTissue.n_node,Vector3d(0,0,0));
    std::vector<Vector3d> node_s0(myTissue.n_node,Vector3d(0,0,0));
	std::vector<Vector3d> node_lamdaP(myTissue.n_node,Vector3d(0,0,0));
    std::vector<Vector3d> node_lamdaE(myTissue.n_node,Vector3d(0,0,0));
	// TENSORS
    std::vector<Matrix3d> node_strain(myTissue.n_node,Matrix3d::Zero(3,3));
    std::vector<Matrix3d> node_stress(myTissue.n_node,Matrix3d::Zero(3,3));
	// count to divide by
	std::vector<int> node_ip_count(myTissue.n_node,0);
	//std::cout<<"saving attributes in paraview file\n";
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
	for(int elemi=0;elemi<myTissue.n_vol_elem;elemi++){
		for(int ip=0;ip<IP_size;ip++){
            if(elem_size == 8){
                // Push closest integration point to the node
                node_phi[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_phif[elemi*IP_size+ip];
                node_a0[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_a0[elemi*IP_size+ip];
                node_s0[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_s0[elemi*IP_size+ip];
                node_kappa[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_kappa[elemi*IP_size+ip];
                node_lamdaP[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_lamdaP[elemi*IP_size+ip];
                node_strain[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_strain[elemi*IP_size+ip];
                node_stress[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_stress[elemi*IP_size+ip];
                node_ip_count[myTissue.vol_elem_connectivity[elemi][ip]] += 1;
            }
            else if(elem_size == 20){
                // For now just average the element...
                for(int nodei=0;nodei<elem_size;nodei++){
                    node_phi[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_phif[elemi*IP_size+ip];
                    node_a0[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_a0[elemi*IP_size+ip];
                    node_s0[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_s0[elemi*IP_size+ip];
                    node_kappa[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_kappa[elemi*IP_size+ip];
                    node_lamdaP[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_lamdaP[elemi*IP_size+ip];
                    node_strain[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_strain[elemi*IP_size+ip];
                    node_stress[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_stress[elemi*IP_size+ip];
                    node_ip_count[myTissue.vol_elem_connectivity[elemi][nodei]] += 1;
                }
            }
            else if(elem_size == 27){
                node_phi[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_phif[elemi*IP_size+ip];
                node_a0[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_a0[elemi*IP_size+ip];
                node_s0[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_s0[elemi*IP_size+ip];
                node_kappa[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_kappa[elemi*IP_size+ip];
                node_lamdaP[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_lamdaP[elemi*IP_size+ip];
                node_strain[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_strain[elemi*IP_size+ip];
                node_stress[myTissue.vol_elem_connectivity[elemi][ip]]+=myTissue.ip_stress[elemi*IP_size+ip];
                node_ip_count[myTissue.vol_elem_connectivity[elemi][ip]] += 1;
            }
            else if(elem_size == 4){
                // Average all IPs of the element onto each of its nodes
                // (node_ip_count is incremented to match).
                for(int nodei=0;nodei<elem_size;nodei++){
                    node_phi[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_phif[elemi*IP_size+ip];
                    node_a0[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_a0[elemi*IP_size+ip];
                    node_s0[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_s0[elemi*IP_size+ip];
                    node_kappa[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_kappa[elemi*IP_size+ip];
                    node_lamdaP[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_lamdaP[elemi*IP_size+ip];
                    node_lamdaE[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_lamdaE[elemi*IP_size+ip];
                    node_strain[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_strain[elemi*IP_size+ip];
                    node_stress[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_stress[elemi*IP_size+ip];
                    node_ip_count[myTissue.vol_elem_connectivity[elemi][nodei]] += 1;
                }
            }
            else if(elem_size==10){
                // For now just average the element...
                // Update to push to the closest IP
                for(int nodei=0;nodei<elem_size;nodei++){
                    node_phi[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_phif[elemi*IP_size+ip];
                    node_a0[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_a0[elemi*IP_size+ip];
                    node_s0[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_s0[elemi*IP_size+ip];
                    node_kappa[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_kappa[elemi*IP_size+ip];
                    node_lamdaP[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_lamdaP[elemi*IP_size+ip];
                    node_strain[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_strain[elemi*IP_size+ip];
                    node_stress[myTissue.vol_elem_connectivity[elemi][nodei]]+=myTissue.ip_stress[elemi*IP_size+ip];
                    node_ip_count[myTissue.vol_elem_connectivity[elemi][nodei]] += 1;
                }
            }
		}
	}
	for(int nodei=0;nodei<myTissue.n_node;nodei++){
		node_phi[nodei] = node_phi[nodei]/node_ip_count[nodei];
		node_a0[nodei] = node_a0[nodei]/node_ip_count[nodei];
        node_s0[nodei] = node_s0[nodei]/node_ip_count[nodei];
		node_kappa[nodei] = node_kappa[nodei]/node_ip_count[nodei];
		node_lamdaP[nodei] = node_lamdaP[nodei]/node_ip_count[nodei];
        node_lamdaE[nodei] = node_lamdaE[nodei]/node_ip_count[nodei];
        node_strain[nodei] = node_strain[nodei]/node_ip_count[nodei];
        node_stress[nodei] = node_stress[nodei]/node_ip_count[nodei];
	}
	// rho, c, phi, theta
	savefile<<"\nPOINT_DATA "<<myTissue.n_node<<"\nSCALARS rho_c_phif_kappa float "<<4<<"\nLOOKUP_TABLE default\n";
    savefile2<<"\nPOINT_DATA "<<myTissue.n_node<<"\nSCALARS Jp_lamdaP float "<<4<<"\nLOOKUP_TABLE default\n";
	for(int i=0;i<myTissue.n_node;i++){
		savefile<< myTissue.node_rho[i] <<" "<<myTissue.node_c[i]<<" "<<node_phi[i]<<" "<<node_kappa[i]<<"\n"; //   myTissue.boundary_flag[i]
        savefile2<<  node_lamdaP[i](0)*node_lamdaP[i](1)*node_lamdaP[i](2) << " " << node_lamdaP[i](0)<<" "<<node_lamdaP[i](1)<<" "<<node_lamdaP[i](2) << "\n";
	}
	// write out the fiber direction
	// Append alpha as its OWN attribute block. The packed SCALARS above is
	// already at 4 components, which is the legacy-VTK maximum, so it cannot be
	// widened; legacy VTK does allow several attribute blocks inside one
	// POINT_DATA section, and a second POINT_DATA line must NOT be emitted.
	if(myTissue.node_alpha.size() == (size_t)myTissue.n_node){
		savefile<<"\nSCALARS alpha float 1\nLOOKUP_TABLE default\n";
		for(int i=0;i<myTissue.n_node;i++){
			savefile<<myTissue.node_alpha[i]<<"\n";
		}
	}

	savefile<<"\nVECTORS a0 float\n";
    savefile2<<"\nVECTORS lamdaE float\n";
	for(int i=0;i<myTissue.n_node;i++){
		savefile<<node_a0[i](0)<<" "<<node_a0[i](1)<<" "<<node_a0[i](2)<<"\n";
        savefile2<<node_lamdaE[i](0)<<" "<<node_lamdaE[i](1)<<" "<<node_lamdaE[i](2)<<"\n";
	}
    // write out the stress or strain tensors
    savefile<<"\nTENSORS strain float\n";
    savefile2<<"\nTENSORS stress float\n";
    for(int i=0;i<myTissue.n_node;i++){
        savefile<<node_strain[i](0,0)<<" "<<node_strain[i](0,1)<<" "<<node_strain[i](0,2)<<"\n";
        savefile<<node_strain[i](1,0)<<" "<<node_strain[i](1,1)<<" "<<node_strain[i](1,2)<<"\n";
        savefile<<node_strain[i](2,0)<<" "<<node_strain[i](2,1)<<" "<<node_strain[i](2,2)<<"\n";
        savefile2<<node_stress[i](0,0)<<" "<<node_stress[i](0,1)<<" "<<node_stress[i](0,2)<<"\n";
        savefile2<<node_stress[i](1,0)<<" "<<node_stress[i](1,1)<<" "<<node_stress[i](1,2)<<"\n";
        savefile2<<node_stress[i](2,0)<<" "<<node_stress[i](2,1)<<" "<<node_stress[i](2,2)<<"\n";
    }
	savefile.close();
    savefile2.close();
}

//---------------------------------------//
// write NODE data to a file
//---------------------------------------//
void writeNode(tissue &myTissue,const char* filename,int nodei,double time)
{
	// write node i to a file
	std::ofstream savefile;
	savefile.open(filename, std::ios_base::app);
	savefile<< time<<","<<myTissue.node_x[nodei](0)<<","<<myTissue.node_x[nodei](1)<<","<<myTissue.node_x[nodei](2)<<","<<myTissue.node_rho[nodei]<<","<<myTissue.node_c[nodei]<<"\n";
	savefile.close();
}

//---------------------------------------//
// write Integration Point data to a file
//---------------------------------------//
void writeIP(tissue &myTissue,const char* filename,int ipi,double time)
{
	// write integration point i to a file
	std::ofstream savefile;
	savefile.open(filename, std::ios_base::app);
	savefile<< time<<","<<myTissue.ip_phif[ipi]<<","<<myTissue.ip_a0[ipi](0)<<","<<myTissue.ip_a0[ipi](1)<<","<<myTissue.ip_a0[ipi](2)<<","<<myTissue.ip_kappa[ipi]<<","<<myTissue.ip_lamdaP[ipi](0)<<","<<myTissue.ip_lamdaP[ipi](1)<<","<<myTissue.ip_lamdaP[ipi](2)<<"\n";
	savefile.close();
}

//---------------------------------------//
// write Element data to a file
//---------------------------------------//
void writeElement(tissue &myTissue,const char* filename,int elemi,double time)
{
	// write element i to a file
	std::ofstream savefile;
	savefile.open(filename, std::ios_base::app);
	// average the nodes and the integration points of this element
	std::vector<int> element = myTissue.vol_elem_connectivity[elemi];
	double rho=0;
	double c = 0;
	Vector3d x;x.setZero();
	double phif = 0;
	Vector3d a0;a0.setZero();
	double kappa = 0;
	Vector3d lamdaP;lamdaP.setZero();
	for(int i=0;i<8;i++){
		x += myTissue.node_x[element[i]];
		rho += myTissue.node_rho[element[i]];
		c += myTissue.node_c[element[i]];
		phif += myTissue.ip_phif[elemi*8+i];
		a0 += myTissue.ip_a0[elemi*8+i];
		kappa += myTissue.ip_kappa[elemi*8+i];
		lamdaP += myTissue.ip_lamdaP[elemi*8+i];
	}
	x = x/8.; rho = rho/8.; c = c/8.;
	phif = phif/8.; a0 = a0/8.; kappa = kappa/8.; lamdaP = lamdaP/8.;
	savefile<<time<<","<<phif<<","<<a0(0)<<","<<a0(1)<<","<<a0(2)<<","<<kappa<<","<<lamdaP(0)<<","<<lamdaP(1)<<","<<rho<<","<<c<<"\n";
}