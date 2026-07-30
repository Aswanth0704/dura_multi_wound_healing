/*
	Mesh generator
*/

#ifndef myMeshGenerator_h
#define myMeshGenerator_h

#include <omp.h>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

using namespace Eigen;

// a simple structure to keep nodes and elements
struct HexMesh
{
	int n_nodes;
	int n_elements;
    int n_surf_elements;
	std::vector<Vector3d> nodes;
	std::vector<std::vector<int> > elements;
    std::vector<std::vector<int> > surface_elements;
	std::vector<int> boundary_flag; // for the nodes
    std::vector<int> surface_boundary_flag; // for the surface elements
	std::vector<int> element_domain_id;  // for domain ids if there are any.
};

// a simple quad mesh generation, really stupid one
HexMesh myHexMesh(const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution);
HexMesh myQuadraticHexMesh(const std::vector<double> &rectangleDimensions, const std::vector<int> &meshResolution);
HexMesh myQuadraticLagrangianHexMesh(const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution);

// a simple quad mesh generation, really stupid one
HexMesh myMultiBlockMesh(const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution);

//----------------------------//
// CONVERT MESH TYPE
//----------------------------//
//
HexMesh SerendipityQuadraticHexMeshfromLinear(HexMesh myMesh, const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution);
HexMesh QuadraticHexMeshfromLinear(HexMesh myMesh, const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution);

//-------------------------------------------------------------------------------------//
// DEPRECATED FUNCTIONS NOT USED
//-------------------------------------------------------------------------------------//

/*
// conform the mesh
void conformMesh2Ellipse(HexMesh &myMesh, std::vector<double> &ellipse);
double distanceX2E(std::vector<double> &ellipse, double x_coord, double y_coord,double mesh_size);
*/

#endif