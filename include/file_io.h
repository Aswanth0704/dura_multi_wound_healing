/*
	Pre and Post processing functions
	Solver functions
	Struct and classes for the problem definition
*/

#ifndef file_io_h
#define file_io_h
#include <string> 
#include "solver.h"
#include "myMeshGenerator.h"
#include <omp.h>
#include <vector>
#include <map>
#include <Eigen/Dense> // most of the vector functions I will need inside of an element
using namespace Eigen;

//-------------------------------------------------//
// IO
//-------------------------------------------------//

//----------------------------//
// REAd OWN FILE
//----------------------------//
//
tissue readTissue(const char* filename);

//----------------------------//
// READ MESH FILES
//----------------------------//
//
// read input file with mesh and fill in structures
void readAbaqusInput(const char* filename, tissue &myTissue);
void readAbaqusInputLikeCOMSOL(const std::string& filename, HexMesh &myMesh);
HexMesh readCOMSOLInput(const std::string& filename, const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution);
HexMesh readParaviewInput(const std::string& filename, const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution);


//----------------------------//
// WRITE PARAVIEW
//----------------------------//
//
// write the paraview file with 
//	RHO, C, PHI, THETA_B at the nodes
void writeParaview(tissue &myTissue, const char* filename, const char* filename2);

//----------------------------//
// WRITE OWN FILE
//----------------------------//
//
void writeTissue(tissue &myTissue, const char* filename,double time);

//----------------------------//
// WRITE NODE
//----------------------------//
//
// write the node information
// DEFORMED x, RHO, C
void writeNode(tissue &myTissue,const char* filename,int nodei,double time);

//----------------------------//
// WRITE IP
//----------------------------//
//
// write integration point information
// just ip variables
// PHI A0 KAPPA LAMBDA 
void writeIP(tissue &myTissue,const char* filename,int ipi,double time);

//----------------------------//
// WRITE ELEMENT
//----------------------------//
//
// write an element information to a text file. write the average
// of variables at the center in the following order
//	PHI A0 KAPPA LAMDA_B RHO C
void writeElement(tissue &myTissue,const char* filename,int elemi,double time);

#endif