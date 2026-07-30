// mesh generator for a simple hexahedral domain

// no need to have a header for this. I will just have a function 

//#define EIGEN_USE_MKL_ALL
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <math.h>
#include "myMeshGenerator.h"
#include <element_functions.h>

//---------------------------------------//
// SIMPLE HEX MESH
//---------------------------------------//
//
HexMesh myHexMesh(const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution)
{
	// number of points in the x and y directions
	int n_x_points = meshResolution[0];
	int n_y_points = meshResolution[1];
	int n_z_points = meshResolution[2];

	// dimensions of the mesh in the x, y, and z direction
	double x_init = hexDimensions[0];
	double x_final = hexDimensions[1];
	double y_init = hexDimensions[2];
	double y_final = hexDimensions[3];
	double z_init = hexDimensions[4];
	double z_final = hexDimensions[5];
	int n_nodes = n_x_points*n_y_points*n_z_points;;
	int n_elems = (n_x_points-1)*(n_y_points-1)*(n_z_points-1);
	std::cout<<"Going to create a mesh of "<<n_nodes<<" nodes and "<<n_elems<<" elements\n";
	std::cout<<"X0 = "<<x_init<<", XF = "<<x_final<<", Y0 = "<<y_init<<", YF = "<<y_final<<", Z0 = " << z_init << ", ZF = " << z_final << "\n";
	std::vector<Vector3d> NODES(n_nodes,Vector3d(0.,0.,0.));
	std::vector<int> elem0 = {0,0,0,0,0,0,0,0};
	std::vector<std::vector<int> > ELEMENTS(n_elems,elem0);
	std::vector<int> BOUNDARIES(n_x_points*n_y_points*n_z_points,0);
	// create the nodes row by row
	for(int i=0;i<n_z_points;i++){
		for(int j=0;j<n_y_points;j++){
			for(int k=0;k<n_x_points;k++){
				// std::cout << "Node Iter" << i*n_x_points*n_y_points+j*n_x_points+k << "\n";
				double x_coord, y_coord, z_coord;
                x_coord = x_init + k*(x_final-x_init)/(n_x_points-1);
                y_coord = y_init + j*(y_final-y_init)/(n_y_points-1);
                z_coord = z_init + i*(z_final-z_init)/(n_z_points-1);
				// std::cout << "X = " << x_coord << ", " << "Y = " << y_coord << ", Z = " << z_coord << " \n";
				NODES[i*n_x_points*n_y_points+j*n_x_points+k](0) = x_coord;
				NODES[i*n_x_points*n_y_points+j*n_x_points+k](1) = y_coord;
				NODES[i*n_x_points*n_y_points+j*n_x_points+k](2) = z_coord;

                //-------------//
                // ALTERNATIVE BOUNDARIES VERSION
                //-------------//
                // Be careful. Make sure you remember which nodes are part of which face
                // Apply Dirichlet BCs first, so the corner nodes become part of those faces
                //
                if(k==0){ // x = 0
					BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=1;
				}
                else if (k==n_x_points-1){ // x = end
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=2;
                }
                else if(j==0){ // y = 0
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=3;
                }
                else if (j==n_y_points-1){ // y = end
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=4;
                }
                else if(i==0){ // z = 0
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=5;
                }
                else if (i==n_z_points-1){ // z = end
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=6;
                }
			}
		}
	}
	std::cout<<"... filled nodes...\n";		
	// create the 3D element connectivity
	for(int i=0;i<n_z_points-1;i++){
		for(int j=0;j<n_y_points-1;j++){
			for (int k=0;k<n_x_points-1;k++){
				std::cout << "Element Iter" << i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k << "\n";
				ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][0] = (i*n_x_points*n_y_points)+(j*n_x_points)+k;
				ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][1] = (i*n_x_points*n_y_points)+(j*n_x_points)+(k+1);
				ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][2] = (i*n_x_points*n_y_points)+((j+1)*n_x_points)+(k+1);
				ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][3] = (i*n_x_points*n_y_points)+((j+1)*n_x_points)+k;
				ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][4] = ((i+1)*n_x_points*n_y_points)+(j*n_x_points)+k;
				ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][5] = ((i+1)*n_x_points*n_y_points)+(j*n_x_points)+(k+1);
				ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][6] = ((i+1)*n_x_points*n_y_points)+((j+1)*n_x_points)+(k+1);
				ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][7] = ((i+1)*n_x_points*n_y_points)+((j+1)*n_x_points)+k;
			}
		}
	}
	// create the 2D boundary connectivity
	/// loop over boundaries

	HexMesh myMesh;
	myMesh.nodes = NODES;
	myMesh.elements = ELEMENTS;
	myMesh.boundary_flag = BOUNDARIES;
	myMesh.n_nodes = n_x_points*n_y_points*n_z_points;
	myMesh.n_elements = (n_x_points-1)*(n_y_points-1)*(n_z_points-1);
	return myMesh;
}


// a simple serendipity quadratic hex mesh generation, really stupid one
// NOT YET FULLY FUNCTIONAL
HexMesh myQuadraticHexMesh(const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution)
{
    throw std::runtime_error("This is currently not functional, need to fix the element definition. Just import a mesh instead.");
    // number of points in the x and y directions
    int n_x_points = meshResolution[0]*2+1;
    int n_y_points = meshResolution[1]*2+1;
    int n_z_points = meshResolution[2]*2+1;

    // dimensions of the mesh in the x and y direction
    double x_init = hexDimensions[0];
    double x_final = hexDimensions[1];
    double y_init = hexDimensions[2];
    double y_final = hexDimensions[3];
    double z_init = hexDimensions[4];
    double z_final = hexDimensions[5];
    std::cout<<"X0 = "<<x_init<<", XF = "<<x_final<<", Y0 = "<<y_init<<", YF = "<<y_final<<", Z0 = " << z_init << ", ZF = " << z_final << "\n";
    std::vector<Vector3d> NODES;
    std::vector<std::vector<int> > ELEMENTS;
    std::vector<int> BOUNDARIES;
    // create the nodes row by row
    for(int i=0;i<n_z_points;i++){
        for(int j=0;j<n_y_points;j++){
            for(int k=0;k<n_x_points;k++){
                if(!(i%2==1 && j%2==1) && !(i%2==1 && k%2==1)&& !(j%2==1 && k%2==1)) { // Do not count nodes that are two odd one even, or three odd
                    double x_coord = x_init + k * (x_final - x_init) / (n_x_points - 1);
                    double y_coord = y_init + j * (y_final - y_init) / (n_y_points - 1);
                    double z_coord = z_init + i * (z_final - z_init) / (n_z_points - 1);
                    Vector3d nodei = Vector3d(x_coord, y_coord, z_coord);
                    NODES.push_back(nodei);
                    if (round(x_coord) == x_init) {
                        BOUNDARIES.push_back(1);
                    } else if (round(x_coord) == x_final) {
                        BOUNDARIES.push_back(2);
                    } else if (round(y_coord) == y_init) {
                        BOUNDARIES.push_back(3);
                    } else if (round(y_coord) == y_final) {
                        BOUNDARIES.push_back(4);
                    } else if (round(z_coord) == z_init) {
                        BOUNDARIES.push_back(5);
                    } else if (round(z_coord) == z_final) {
                        BOUNDARIES.push_back(6);
                    } else {
                        BOUNDARIES.push_back(0);
                    }
                }
            }
        }
    }
    std::cout<<"... filled nodes...\n";
    // create the connectivity
    for(int k=0;k<n_z_points-2;k+=2){
        for(int j=0;j<n_y_points-2;j+=2){
            for(int i=0;i<n_x_points-2;i+=2) {
                // Push nodes.
                std::vector<int> elemi;
                elemi.clear();
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                elemi.push_back((k/2)* + ((j/2)*n_x_points) + (j/2)*((n_x_points + 1)/2) + (i));
                //std::cout<<"\n";
                ELEMENTS.push_back(elemi);
            }
        }
    }
    HexMesh myMesh;
    myMesh.nodes = NODES;
    myMesh.elements = ELEMENTS;
    myMesh.boundary_flag = BOUNDARIES;
    myMesh.n_nodes = NODES.size();
    myMesh.n_elements = ELEMENTS.size();
    return myMesh;
}

// a simple Lagrangian quadratic hex mesh generation, really stupid one
HexMesh myQuadraticLagrangianHexMesh(const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution)
{
    Eigen::initParallel();
    // number of points in the x and y directions
    int n_x_points = meshResolution[0]*2+1;
    int n_y_points = meshResolution[1]*2+1;
    int n_z_points = meshResolution[2]*2+1;

    // dimensions of the mesh in the x and y direction
    double x_init = hexDimensions[0];
    double x_final = hexDimensions[1];
    double y_init = hexDimensions[2];
    double y_final = hexDimensions[3];
    double z_init = hexDimensions[4];
    double z_final = hexDimensions[5];
    std::cout<<"X0 = "<<x_init<<", XF = "<<x_final<<", Y0 = "<<y_init<<", YF = "<<y_final<<", Z0 = " << z_init << ", ZF = " << z_final << "\n";
    std::vector<Vector3d> NODES;
    std::vector<std::vector<int> > ELEMENTS;
    std::vector<int> BOUNDARIES;
    // create the nodes row by row
    for(int i=0;i<n_z_points;i++){
        for(int j=0;j<n_y_points;j++){
            for(int k=0;k<n_x_points;k++){
                double x_coord = x_init + k * (x_final - x_init) / (n_x_points - 1);
                double y_coord = y_init + j * (y_final - y_init) / (n_y_points - 1);
                double z_coord = z_init + i * (z_final - z_init) / (n_z_points - 1);
                Vector3d nodei = Vector3d(x_coord, y_coord, z_coord);
                NODES.push_back(nodei);
                if (round(x_coord) == x_init) {
                    BOUNDARIES.push_back(1);
                } else if (round(x_coord) == x_final) {
                    BOUNDARIES.push_back(2);
                } else if (round(y_coord) == y_init) {
                    BOUNDARIES.push_back(3);
                } else if (round(y_coord) == y_final) {
                    BOUNDARIES.push_back(4);
                } else if (round(z_coord) == z_init) {
                    BOUNDARIES.push_back(5);
                } else if (round(z_coord) == z_final) {
                    BOUNDARIES.push_back(6);
                } else {
                    BOUNDARIES.push_back(0);
                }
            }
        }
    }
    std::cout<<"... filled nodes...\n";
    // create the connectivity
    for(int k=0;k<n_z_points-2;k+=2){
        for(int j=0;j<n_y_points-2;j+=2){
            for(int i=0;i<n_x_points-2;i+=2) {
                // Push nodes.
                std::vector<int> elemi;
                elemi.clear();
                // Corner nodes
                elemi.push_back((k)*(n_x_points*n_y_points) + (j)*(n_x_points) + (i));
                elemi.push_back((k)*(n_x_points*n_y_points) + (j)*(n_x_points) + (i+2));
                elemi.push_back((k)*(n_x_points*n_y_points) + (j+2)*(n_x_points) + (i+2));
                elemi.push_back((k)*(n_x_points*n_y_points) + (j+2)*(n_x_points) + (i));
                elemi.push_back((k+2)*(n_x_points*n_y_points) + (j)*(n_x_points) + (i));
                elemi.push_back((k+2)*(n_x_points*n_y_points) + (j)*(n_x_points) + (i+2));
                elemi.push_back((k+2)*(n_x_points*n_y_points) + (j+2)*(n_x_points) + (i+2));
                elemi.push_back((k+2)*(n_x_points*n_y_points) + (j+2)*(n_x_points) + (i));
                // Bottom midedge
                elemi.push_back((k)*(n_x_points*n_y_points) + (j)*(n_x_points) + (i+1));
                elemi.push_back((k)*(n_x_points*n_y_points) + (j+1)*(n_x_points) + (i+2));
                elemi.push_back((k)*(n_x_points*n_y_points) + (j+2)*(n_x_points) + (i+1));
                elemi.push_back((k)*(n_x_points*n_y_points) + (j+1)*(n_x_points) + (i));
                // Middle midedge
                elemi.push_back((k+1)*(n_x_points*n_y_points) + (j)*(n_x_points) + (i));
                elemi.push_back((k+1)*(n_x_points*n_y_points) + (j)*(n_x_points) + (i+2));
                elemi.push_back((k+1)*(n_x_points*n_y_points) + (j+2)*(n_x_points) + (i+2));
                elemi.push_back((k+1)*(n_x_points*n_y_points) + (j+2)*(n_x_points) + (i));
                // Top midedge
                elemi.push_back((k+2)*(n_x_points*n_y_points) + (j)*(n_x_points) + (i+1));
                elemi.push_back((k+2)*(n_x_points*n_y_points) + (j+1)*(n_x_points) + (i+2));
                elemi.push_back((k+2)*(n_x_points*n_y_points) + (j+2)*(n_x_points) + (i+1));
                elemi.push_back((k+2)*(n_x_points*n_y_points) + (j+1)*(n_x_points) + (i));
                // Midface
                elemi.push_back((k)*(n_x_points*n_y_points) + (j+1)*(n_x_points) + (i+1));
                elemi.push_back((k+1)*(n_x_points*n_y_points) + (j)*(n_x_points) + (i+1));
                elemi.push_back((k+1)*(n_x_points*n_y_points) + (j+1)*(n_x_points) + (i+2));
                elemi.push_back((k+1)*(n_x_points*n_y_points) + (j+2)*(n_x_points) + (i+1));
                elemi.push_back((k+1)*(n_x_points*n_y_points) + (j+1)*(n_x_points) + (i));
                elemi.push_back((k+2)*(n_x_points*n_y_points) + (j+1)*(n_x_points) + (i+1));
                // Center node
                elemi.push_back((k+1)*(n_x_points*n_y_points) + (j+1)*(n_x_points) + (i+1));
                //std::cout<<"\n";
                ELEMENTS.push_back(elemi);
            }
        }
    }
    HexMesh myMesh;
    myMesh.nodes = NODES;
    myMesh.elements = ELEMENTS;
    myMesh.boundary_flag = BOUNDARIES;
    myMesh.n_nodes = NODES.size();
    myMesh.n_elements = ELEMENTS.size();
    return myMesh;
}

//---------------------------------------//
// MULTIBLOCK MESH
//---------------------------------------//
//
HexMesh myMultiBlockMesh(const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution)
{
    // number of points in the x and y directions
    int n_x_points = meshResolution[0];
    int n_y_points = meshResolution[1];
    int n_z_points = meshResolution[2];

    // dimensions of the mesh in the x, y, and z direction
    double x_init = hexDimensions[0];
    double x_final = hexDimensions[1];
    double y_init = hexDimensions[2];
    double y_final = hexDimensions[3];
    double z_init = hexDimensions[4];
    double z_final = hexDimensions[5];
    int n_nodes = n_x_points*n_y_points*n_z_points;;
    int n_elems = (n_x_points-1)*(n_y_points-1)*(n_z_points-1);
    std::cout<<"Going to create a mesh of "<<n_nodes<<" nodes and "<<n_elems<<" elements\n";
    std::cout<<"X0 = "<<x_init<<", XF = "<<x_final<<", Y0 = "<<y_init<<", YF = "<<y_final<<", Z0 = " << z_init << ", ZF = " << z_final << "\n";
    std::vector<Vector3d> NODES(n_nodes,Vector3d(0.,0.,0.));
    std::vector<int> elem0 = {0,0,0,0,0,0,0,0};
    std::vector<std::vector<int> > ELEMENTS(n_elems,elem0);
    std::vector<int> BOUNDARIES(n_x_points*n_y_points*n_z_points,0);
    // create the nodes row by row
    for(int i=0;i<n_z_points;i++){
        for(int j=0;j<n_y_points;j++){
            for(int k=0;k<n_x_points;k++){
                // std::cout << "Node Iter" << i*n_x_points*n_y_points+j*n_x_points+k << "\n";
                double x_coord, y_coord, z_coord;
                double inner = 1./3;
                double outer = (1-inner)/2.;
                double zinner = 1./2;
                double zouter = 1-zinner;
                // x coord
                if(k<=(n_x_points-1)/4){
                    x_coord = x_init + k*(outer)*(x_final-x_init)/((n_x_points-1)/4.);
                }
                else if(k<=3*(n_x_points-1)/4){
                    x_coord = x_init + (outer)*(x_final-x_init) + (k - (n_x_points-1)/4.)*(inner)*(x_final-x_init)/((n_x_points-1)/2.);
                }
                else{
                    x_coord = x_init + (outer+inner)*(x_final-x_init) + (k - 3*(n_x_points-1)/4.)*(outer)*(x_final-x_init)/((n_x_points-1)/4.);
                }
                // y coord
                if(j<=(n_y_points-1)/4){
                    y_coord = y_init + j*(outer)*(y_final-y_init)/((n_y_points-1)/4.);
                }
                else if(j<=3*(n_y_points-1)/4){
                    y_coord = y_init + (outer)*(y_final-y_init) + (j - (n_y_points-1)/4.)*(inner)*(y_final-y_init)/((n_y_points-1)/2.);
                }
                else{
                    y_coord = y_init + (outer+inner)*(y_final-y_init) + (j - 3*(n_y_points-1)/4.)*(outer)*(y_final-y_init)/((n_y_points-1)/4.);
                }
                // z coord
                if(i<=(n_z_points-1)/2){
                    z_coord = z_init + i*(zouter)*(z_final-z_init)/((n_z_points-1)/3.);
                }
                else{
                    z_coord = z_init + (zouter)*(z_final-z_init) + (i - (n_z_points-1)/3.)*(zinner)*(z_final-z_init)/(2*(n_z_points-1)/3.);
                }

                // std::cout << "X = " << x_coord << ", " << "Y = " << y_coord << ", Z = " << z_coord << " \n";
                NODES[i*n_x_points*n_y_points+j*n_x_points+k](0) = x_coord;
                NODES[i*n_x_points*n_y_points+j*n_x_points+k](1) = y_coord;
                NODES[i*n_x_points*n_y_points+j*n_x_points+k](2) = z_coord;

                //-------------//
                // ALTERNATIVE BOUNDARIES VERSION
                //-------------//
                // Be careful. Make sure you remember which nodes are part of which face
                // Apply Dirichlet BCs first, so the corner nodes become part of those faces
                //
                if(k==0){ // x = 0
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=1;
                }
                else if (k==n_x_points-1){ // x = end
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=2;
                }
                else if(j==0){ // y = 0
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=3;
                }
                else if (j==n_y_points-1){ // y = end
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=4;
                }
                else if(i==0){ // z = 0
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=5;
                }
                else if (i==n_z_points-1){ // z = end
                    BOUNDARIES[i*n_x_points*n_y_points+j*n_x_points+k]=6;
                }
            }
        }
    }
    std::cout<<"... filled nodes...\n";
    // create the 3D element connectivity
    for(int i=0;i<n_z_points-1;i++){
        for(int j=0;j<n_y_points-1;j++){
            for (int k=0;k<n_x_points-1;k++){
                std::cout << "Element Iter" << i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k << "\n";
                ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][0] = (i*n_x_points*n_y_points)+(j*n_x_points)+k;
                ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][1] = (i*n_x_points*n_y_points)+(j*n_x_points)+(k+1);
                ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][2] = (i*n_x_points*n_y_points)+((j+1)*n_x_points)+(k+1);
                ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][3] = (i*n_x_points*n_y_points)+((j+1)*n_x_points)+k;
                ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][4] = ((i+1)*n_x_points*n_y_points)+(j*n_x_points)+k;
                ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][5] = ((i+1)*n_x_points*n_y_points)+(j*n_x_points)+(k+1);
                ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][6] = ((i+1)*n_x_points*n_y_points)+((j+1)*n_x_points)+(k+1);
                ELEMENTS[i*(n_x_points-1)*(n_y_points-1)+j*(n_x_points-1)+k][7] = ((i+1)*n_x_points*n_y_points)+((j+1)*n_x_points)+k;
            }
        }
    }
    // create the 2D boundary connectivity
    /// loop over boundaries

    HexMesh myMesh;
    myMesh.nodes = NODES;
    myMesh.elements = ELEMENTS;
    myMesh.boundary_flag = BOUNDARIES;
    myMesh.n_nodes = n_x_points*n_y_points*n_z_points;
    myMesh.n_elements = (n_x_points-1)*(n_y_points-1)*(n_z_points-1);
    return myMesh;
}

HexMesh SerendipityQuadraticHexMeshfromLinear(HexMesh myMesh, const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution)
{
    // Generate Lagrangian quadratic mesh from a linear one
    std::vector<Vector3d> NODES = myMesh.nodes;
    std::vector<std::vector<int>> ELEMENTS = myMesh.elements;
    std::vector<int> BOUNDARIES = myMesh.boundary_flag;

    // Make a vector of xi, eta, zeta for the new points in reference coordinates
    std::vector<Vector3d> new_nodes_xietazeta(12, Vector3d(0.,0.,0.));
    // Edge nodes
    new_nodes_xietazeta[0] = Vector3d(0.,-1.,-1.);
    new_nodes_xietazeta[1] = Vector3d(1.,0.,-1.);
    new_nodes_xietazeta[2] = Vector3d(0.,1.,-1.);
    new_nodes_xietazeta[3] = Vector3d(-1.,0.,-1.);
    new_nodes_xietazeta[4] = Vector3d(-1.,-1.,0.);
    new_nodes_xietazeta[5] = Vector3d(1.,-1.,0.);
    new_nodes_xietazeta[6] = Vector3d(1.,1.,0.);
    new_nodes_xietazeta[7] = Vector3d(-1.,1.,0.);
    new_nodes_xietazeta[8] = Vector3d(0.,-1.,1.);
    new_nodes_xietazeta[9] = Vector3d(1.,0.,1.);
    new_nodes_xietazeta[10] = Vector3d(0.,1.,1.);
    new_nodes_xietazeta[11] = Vector3d(-1.,0.,1.);

    int n_elem = myMesh.n_elements;
    int elem_size = myMesh.elements[0].size();
    for(int elemi=0;elemi<n_elem;elemi++) {
        // Transform to the real coordinates
        std::vector<Vector3d> new_nodes_xyz(12, Vector3d(0,0,0));
        int new_nodes_size = new_nodes_xietazeta.size();
        for (int new_node = 0; new_node < new_nodes_size; new_node++) {
            double xi = new_nodes_xietazeta[new_node](0);
            double eta = new_nodes_xietazeta[new_node](1);
            double zeta = new_nodes_xietazeta[new_node](2);
            std::vector<double> R = evalShapeFunctionsR(xi, eta, zeta);
            for (int nodej = 0; nodej < elem_size; nodej++) {
                //std::cout<<R[nodej]<<"\n";
                //std::cout<<myMesh.nodes[myMesh.elements[elemi][nodej]]<<"\n";
                new_nodes_xyz[new_node] += R[nodej] * myMesh.nodes[myMesh.elements[elemi][nodej]];
            }
            //std::cout<<new_nodes_xyz[new_node]<<"\n";
            // Now check if they already exist,
            std::vector<Vector3d>::iterator it = std::find(NODES.begin(), NODES.end(), new_nodes_xyz[new_node]);
            int index = std::distance(NODES.begin(), it);
            if (it == NODES.end()){
                // If not, add to nodes and check boundaries
                NODES.push_back(new_nodes_xyz[new_node]);

                // Check which face it belongs to
                if(round(new_nodes_xyz[new_node](0))==hexDimensions[0]){ // x = 0
                    BOUNDARIES.push_back(1);
                }
                else if (round(new_nodes_xyz[new_node](0))==hexDimensions[1]){ // x = end
                    BOUNDARIES.push_back(2);
                }
                else if(round(new_nodes_xyz[new_node](1))==hexDimensions[2]){ // y = 0
                    BOUNDARIES.push_back(3);
                }
                else if (round(new_nodes_xyz[new_node](1))==hexDimensions[3]){ // y = end
                    BOUNDARIES.push_back(4);
                }
                else if(round(new_nodes_xyz[new_node](2))==hexDimensions[4]){ // z = 0
                    BOUNDARIES.push_back(5);
                }
                else if (round(new_nodes_xyz[new_node](2))==hexDimensions[5]){ // z = end
                    BOUNDARIES.push_back(6);
                }
                else{
                    // Else it is an inner node
                    BOUNDARIES.push_back(0);
                }
                // Then add to the element
                ELEMENTS[elemi].push_back(NODES.size()-1);
            }
            else{
                // Then add to the element
                ELEMENTS[elemi].push_back(index);
            }
        }
    }

    HexMesh myQuadraticMesh;
    myQuadraticMesh.nodes = NODES;
    myQuadraticMesh.elements = ELEMENTS;
    myQuadraticMesh.boundary_flag = BOUNDARIES;
    myQuadraticMesh.n_nodes = NODES.size();
    myQuadraticMesh.n_elements = ELEMENTS.size();
    return myQuadraticMesh;
}

HexMesh QuadraticHexMeshfromLinear(HexMesh myMesh, const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution)
{
    // Generate Lagrangian quadratic mesh from a linear one
    std::vector<Vector3d> NODES = myMesh.nodes;
    std::vector<std::vector<int>> ELEMENTS = myMesh.elements;
    std::vector<int> BOUNDARIES = myMesh.boundary_flag;

    // Make a vector of xi, eta, zeta for the new points in reference coordinates
    std::vector<Vector3d> new_nodes_xietazeta(19, Vector3d(0.,0.,0.));
    // Edge nodes
    new_nodes_xietazeta[0] = Vector3d(0.,-1.,-1.);
    new_nodes_xietazeta[1] = Vector3d(1.,0.,-1.);
    new_nodes_xietazeta[2] = Vector3d(0.,1.,-1.);
    new_nodes_xietazeta[3] = Vector3d(-1.,0.,-1.);
    new_nodes_xietazeta[4] = Vector3d(-1.,-1.,0.);
    new_nodes_xietazeta[5] = Vector3d(1.,-1.,0.);
    new_nodes_xietazeta[6] = Vector3d(1.,1.,0.);
    new_nodes_xietazeta[7] = Vector3d(-1.,1.,0.);
    new_nodes_xietazeta[8] = Vector3d(0.,-1.,1.);
    new_nodes_xietazeta[9] = Vector3d(1.,0.,1.);
    new_nodes_xietazeta[10] = Vector3d(0.,1.,1.);
    new_nodes_xietazeta[11] = Vector3d(-1.,0.,1.);
    // Face nodes
    new_nodes_xietazeta[12] = Vector3d(0.,0.,-1.);
    new_nodes_xietazeta[13] = Vector3d(0.,-1.,0.);
    new_nodes_xietazeta[14] = Vector3d(1.,0.,0.);
    new_nodes_xietazeta[15] = Vector3d(0.,1.,0.);
    new_nodes_xietazeta[16] = Vector3d(-1.,0.,0.);
    new_nodes_xietazeta[17] = Vector3d(0.,0.,1.);
    // Center node
    new_nodes_xietazeta[18] = Vector3d(0.,0.,0.);

    int n_elem = myMesh.n_elements;
    int elem_size = myMesh.elements[0].size();
    for(int elemi=0;elemi<n_elem;elemi++) {
        // Transform to the real coordinates
        std::vector<Vector3d> new_nodes_xyz(19, Vector3d(0,0,0));
        int new_nodes_size = new_nodes_xietazeta.size();
        for (int new_node = 0; new_node < new_nodes_size; new_node++) {
            double xi = new_nodes_xietazeta[new_node](0);
            double eta = new_nodes_xietazeta[new_node](1);
            double zeta = new_nodes_xietazeta[new_node](2);
            std::vector<double> R = evalShapeFunctionsR(xi, eta, zeta);
            for (int nodej = 0; nodej < elem_size; nodej++) {
                //std::cout<<R[nodej]<<"\n";
                //std::cout<<myMesh.nodes[myMesh.elements[elemi][nodej]]<<"\n";
                new_nodes_xyz[new_node] += R[nodej] * myMesh.nodes[myMesh.elements[elemi][nodej]];
            }
            //std::cout<<new_nodes_xyz[new_node]<<"\n";
            // Now check if they already exist,
            std::vector<Vector3d>::iterator it = std::find(NODES.begin(), NODES.end(), new_nodes_xyz[new_node]);
            int index = std::distance(NODES.begin(), it);
            if (it == NODES.end()){
                // If not, add to nodes and check boundaries
                NODES.push_back(new_nodes_xyz[new_node]);

                // Check which face it belongs to
                if(round(new_nodes_xyz[new_node](0))==hexDimensions[0]){ // x = 0
                    BOUNDARIES.push_back(1);
                }
                else if (round(new_nodes_xyz[new_node](0))==hexDimensions[1]){ // x = end
                    BOUNDARIES.push_back(2);
                }
                else if(round(new_nodes_xyz[new_node](1))==hexDimensions[2]){ // y = 0
                    BOUNDARIES.push_back(3);
                }
                else if (round(new_nodes_xyz[new_node](1))==hexDimensions[3]){ // y = end
                    BOUNDARIES.push_back(4);
                }
                else if(round(new_nodes_xyz[new_node](2))==hexDimensions[4]){ // z = 0
                    BOUNDARIES.push_back(5);
                }
                else if (round(new_nodes_xyz[new_node](2))==hexDimensions[5]){ // z = end
                    BOUNDARIES.push_back(6);
                }
                else{
                    // Else it is an inner node
                    BOUNDARIES.push_back(0);
                }
                // Then add to the element
                ELEMENTS[elemi].push_back(NODES.size()-1);
            }
            else{
                // Then add to the element
                ELEMENTS[elemi].push_back(index);
            }
        }
    }

    HexMesh myQuadraticMesh;
    myQuadraticMesh.nodes = NODES;
    myQuadraticMesh.elements = ELEMENTS;
    myQuadraticMesh.boundary_flag = BOUNDARIES;
    myQuadraticMesh.n_nodes = NODES.size();
    myQuadraticMesh.n_elements = ELEMENTS.size();
    return myQuadraticMesh;
}

/*
double distanceX2E(std::vector<double> &ellipse, double x_coord, double y_coord, double z_coord, double mesh_size)
{
	// given a point and the geometry of an ellipse, give me the
	// distance along the x axis towards the ellipse
	double x_center = ellipse[0];
	double y_center = ellipse[1];
	double x_axis = ellipse[2];
	double y_axis = ellipse[3];
	double alpha = ellipse[4];
	
	// equation of the ellipse 
	double x_ellipse_1 = (pow(x_axis,2)*x_center*pow(sin(alpha),2) + pow(x_axis,2)*y_center*sin(2*alpha)/2 - pow(x_axis,2)*y_coord*sin(2*alpha)/2 \
						+ x_center*pow(y_axis,2)*pow(cos(alpha),2) + pow(y_axis,2)*y_center*sin(2*alpha)/2 - pow(y_axis,2)*y_coord*sin(2*alpha)/2 -\
						sqrt(pow(x_axis,2)*pow(y_axis,2)*(pow(x_axis,2)*pow(sin(alpha),2) - pow(y_axis,2)*pow(sin(alpha),2) + pow(y_axis,2) \
				- 4*pow(y_center,2)*pow(sin(alpha),4) + 4*pow(y_center,2)*pow(sin(alpha),2) - pow(y_center,2) + 8*y_center*y_coord*pow(sin(alpha),4)\
				 - 8*y_center*y_coord*pow(sin(alpha),2) + 2*y_center*y_coord - 4*pow(y_coord,2)*pow(sin(alpha),4) + 4*pow(y_coord,2)*pow(sin(alpha),2)\
				  - pow(y_coord,2))))/(pow(x_axis,2)*pow(sin(alpha),2) + pow(y_axis,2)*pow(cos(alpha),2));
	double x_ellipse_2 = (pow(x_axis,2)*x_center*pow(sin(alpha),2) + pow(x_axis,2)*y_center*sin(2*alpha)/2 - pow(x_axis,2)*y_coord*sin(2*alpha)/2 \
						+ x_center*pow(y_axis,2)*pow(cos(alpha),2) + pow(y_axis,2)*y_center*sin(2*alpha)/2 - pow(y_axis,2)*y_coord*sin(2*alpha)/2 +\
						sqrt(pow(x_axis,2)*pow(y_axis,2)*(pow(x_axis,2)*pow(sin(alpha),2) - pow(y_axis,2)*pow(sin(alpha),2) + pow(y_axis,2) \
				- 4*pow(y_center,2)*pow(sin(alpha),4) + 4*pow(y_center,2)*pow(sin(alpha),2) - pow(y_center,2) + 8*y_center*y_coord*pow(sin(alpha),4)\
				 - 8*y_center*y_coord*pow(sin(alpha),2) + 2*y_center*y_coord - 4*pow(y_coord,2)*pow(sin(alpha),4) + 4*pow(y_coord,2)*pow(sin(alpha),2)\
				  - pow(y_coord,2))))/(pow(x_axis,2)*pow(sin(alpha),2) + pow(y_axis,2)*pow(cos(alpha),2));
	// which is closer?
	double distance1 = fabs(x_ellipse_1 - x_coord);
	double distance2 = fabs(x_ellipse_2 - x_coord);
	if(distance1<distance2 && distance1<mesh_size){
		return x_ellipse_1-x_coord;
	}else if(distance2<mesh_size){
		return x_ellipse_2-x_coord;
	}
	return 0;
}

// conform the mesh to a given ellipse
void conformMesh2Ellipse(HexMesh &myMesh, std::vector<double> &ellipse)
{
	// the ellipse is defined by center, axis, and angle
	double x_center = ellipse[0];
	double y_center = ellipse[1];
	double z_center = ellipse[2];
	double x_axis = ellipse[3];
	double y_axis = ellipse[4];
	double z_axis = ellipse[5];
	double alpha_ellipse = ellipse[6];
	
	// loop over the mesh nodes 
	double x_coord,y_coord,z_coord,check,d_x2e,mesh_size;
	mesh_size = (myMesh.nodes[1](0)-myMesh.nodes[0](0))/1.1;
	for(int i=0;i<myMesh.n_nodes;i++){
		// if the point is inside check if it is close, if it is in a certain 
		// range smaller than mesh size then move it the ellipse along x. ta ta
		x_coord = myMesh.nodes[i](0);
		y_coord = myMesh.nodes[i](1);
		check = pow((x_coord-x_center)*cos(alpha_ellipse)+(y_coord-y_center)*sin(alpha_ellipse),2)/(x_axis*x_axis) +\
				pow((x_coord-x_center)*sin(alpha_ellipse)+(y_coord-y_center)*cos(alpha_ellipse),2)/(y_axis*y_axis);
		if(check>1){
			// calculate the distance to the ellipse along x axis
			d_x2e = distanceX2E(ellipse,x_coord,y_coord,z_coord,mesh_size);
			myMesh.nodes[i](0) += d_x2e;
		}
	}					
}
*/