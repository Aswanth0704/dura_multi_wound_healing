/*

WOUND

This code is the implementation of the wound model
Particularly, the Dalai Lama Wound Healing, or DaLaWoHe

*/

#ifndef element_functions_h
#define element_functions_h

#include <omp.h>
#include <vector>
#include <map>
#include <Eigen/Dense> // most of the vector functions I will need inside of an element
using namespace Eigen;


/////////////////////////////////////////////////////////////////////////////////////////
// GEOMETRY and ELEMENT ROUTINES
/////////////////////////////////////////////////////////////////////////////////////////

//-----------------------------//
// Jacobians, at all ip and xi,eta
//
std::vector<Matrix3d> evalJacobian(const std::vector<Vector3d> node_X);
std::vector<double> evalJacobianSurface(const std::vector<Vector3d> node_X); //std::vector<MatrixXd>
Matrix3d evalJacobian(const std::vector<Vector3d> node_X, double xi, double eta, double zeta);
//
//-----------------------------//

//-----------------------------//
// Integration points
//
std::vector<Vector4d> LineQuadriIP();
std::vector<Vector4d> LineQuadriIPQuadratic();
std::vector<Vector4d> LineQuadriIPTet();
std::vector<Vector4d> LineQuadriIPTetQuadratic();
std::vector<Vector3d> LineQuadriIPSurfaceQuad();
//
//-----------------------------//

//-----------------------------//
// Basis functions
//
std::vector<double> evalShapeFunctionsR(double xi,double eta, double zeta);
std::vector<double> evalShapeFunctionsRxi(double xi,double eta, double zeta);
std::vector<double> evalShapeFunctionsReta(double xi,double eta, double zeta);
std::vector<double> evalShapeFunctionsRzeta(double xi, double eta, double zeta);
std::vector<double> evalShapeFunctionsQuadraticR(double xi,double eta, double zeta);
std::vector<double> evalShapeFunctionsQuadraticRxi(double xi,double eta, double zeta);
std::vector<double> evalShapeFunctionsQuadraticReta(double xi,double eta, double zeta);
std::vector<double> evalShapeFunctionsQuadraticRzeta(double xi, double eta, double zeta);
std::vector<double> evalShapeFunctionsQuadraticLagrangeR(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsQuadraticLagrangeRxi(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsQuadraticLagrangeReta(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsQuadraticLagrangeRzeta(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsTetR(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsTetRxi(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsTetReta(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsTetRzeta(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsTetQuadraticR(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsTetQuadraticRxi(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsTetQuadraticReta(double xi,double eta,double zeta);
std::vector<double> evalShapeFunctionsTetQuadraticRzeta(double xi,double eta,double zeta);

std::vector<double> evalShapeFunctionsSurfaceQuadR(double xi,double eta);
std::vector<double> evalShapeFunctionsSurfaceQuadRxi(double xi,double eta);
std::vector<double> evalShapeFunctionsSurfaceQuadReta(double xi,double eta);
//
//-----------------------------//




#endif