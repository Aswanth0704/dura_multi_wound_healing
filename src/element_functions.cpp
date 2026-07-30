// last date modified: Sept, 16, 2025

/*

WOUND

This code is the implementation of the DaLaWoHe

*/

//#define EIGEN_USE_MKL_ALL
#include <omp.h>
#include "element_functions.h"
#include <iostream>
#include <cmath>
#include <map>
#include <Eigen/OrderingMethods>
#include <fstream>
using namespace Eigen;



//--------------------------------------------------------//
// GEOMETRY and ELEMENT ROUTINES
//--------------------------------------------------------//

//-----------------------------//
// Jacobians
//-----------------------------//

std::vector<Matrix3d> evalJacobian(const std::vector<Vector3d> node_X)
{
    // The gradient of the shape functions with respect to the reference coordinates

    // Vector with 4 elements, each element is the inverse transpose of the
    // Jacobian at the corresponding integration point of the linear hexahedron
    std::vector<Matrix3d> ip_Jac;
    int elem_size = node_X.size();

    // LOOP OVER THE INTEGRATION POINTS
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
    for(int ip=0;ip<IP_size;ip++){

        // evaluate basis functions derivatives
        // coordinates of the integration point in parent domain
        double xi = IP[ip](0);
        double eta = IP[ip](1);
        double zeta = IP[ip](2);

        // eval shape functions
        std::vector<double> R;
        // eval derivatives
        std::vector<double> Rxi;
        std::vector<double> Reta;
        std::vector<double> Rzeta;
        if(elem_size == 8){
            R = evalShapeFunctionsR(xi,eta,zeta);
            Rxi = evalShapeFunctionsRxi(xi,eta,zeta);
            Reta = evalShapeFunctionsReta(xi,eta,zeta);
            Rzeta = evalShapeFunctionsRzeta(xi,eta,zeta);
        }
        else if(elem_size == 20){
            R = evalShapeFunctionsQuadraticR(xi,eta,zeta);
            Rxi = evalShapeFunctionsQuadraticRxi(xi,eta,zeta);
            Reta = evalShapeFunctionsQuadraticReta(xi,eta,zeta);
            Rzeta = evalShapeFunctionsQuadraticRzeta(xi,eta,zeta);
        }
        else if(elem_size == 27){
            R = evalShapeFunctionsQuadraticLagrangeR(xi,eta,zeta);
            Rxi = evalShapeFunctionsQuadraticLagrangeRxi(xi,eta,zeta);
            Reta = evalShapeFunctionsQuadraticLagrangeReta(xi,eta,zeta);
            Rzeta = evalShapeFunctionsQuadraticLagrangeRzeta(xi,eta,zeta);
        }
        else if(elem_size == 4){
            R = evalShapeFunctionsTetR(xi,eta,zeta);
            Rxi = evalShapeFunctionsTetRxi(xi,eta,zeta);
            Reta = evalShapeFunctionsTetReta(xi,eta,zeta);
            Rzeta = evalShapeFunctionsTetRzeta(xi,eta,zeta);
        }
        else if(elem_size == 10){
            R = evalShapeFunctionsTetQuadraticR(xi,eta,zeta);
            Rxi = evalShapeFunctionsTetQuadraticRxi(xi,eta,zeta);
            Reta = evalShapeFunctionsTetQuadraticReta(xi,eta,zeta);
            Rzeta = evalShapeFunctionsTetQuadraticRzeta(xi,eta,zeta);
        }
        else{
            throw std::runtime_error("Wrong number of nodes in element!");
        }
        // sum over the 8 nodes
        Vector3d dXdxi;dXdxi.setZero();
        Vector3d dXdeta;dXdeta.setZero();
        Vector3d dXdzeta;dXdzeta.setZero();
        for(int ni=0;ni<elem_size;ni++)
        {
            dXdxi += Rxi[ni]*node_X[ni];
            dXdeta += Reta[ni]*node_X[ni];
            dXdzeta += Rzeta[ni]*node_X[ni];
        }
        // put them in one column
        Matrix3d Jac; Jac<<dXdxi(0),dXdeta(0),dXdzeta(0),dXdxi(1),dXdeta(1),dXdzeta(1),dXdxi(2),dXdeta(2),dXdzeta(2);
        // invert and transpose it
        Matrix3d Jac_iT = (Jac.inverse()).transpose();
        // save this for the vector
        ip_Jac.push_back(Jac_iT);
    }

    // return the vector with all the inverse jacobians
    return ip_Jac;
}

std::vector<double> evalJacobianSurface(const std::vector<Vector3d> node_X)
{
    // The gradient of the shape functions with respect to the reference coordinates

    // Vector with 4 elements, each element is the inverse transpose of the
    // Jacobian at the corresponding integration point of the linear quadrilateral

    std::vector<double> ip_Jac;
    int elem_size = node_X.size();
    // LOOP OVER THE INTEGRATION POINTS
    std::vector<Vector3d> IP = LineQuadriIPSurfaceQuad();
    int n_IP = IP.size();
    for(int ip=0;ip<n_IP;ip++){

        // evaluate basis functions derivatives
        // coordinates of the integration point in parent domain
        double xi = IP[ip](0);
        double eta = IP[ip](1);

        // eval shape functions
        std::vector<double> R;
        // eval derivatives
        std::vector<double> Rxi;
        std::vector<double> Reta;
        if(elem_size == 4){
            R = evalShapeFunctionsSurfaceQuadR(xi,eta);
            Rxi = evalShapeFunctionsSurfaceQuadRxi(xi,eta);
            Reta = evalShapeFunctionsSurfaceQuadReta(xi,eta);
        }
        else{
            throw std::runtime_error("Wrong number of nodes in element!");
        }

        // sum over the 4 nodes
        Vector3d dXdxi;dXdxi.setZero();
        Vector3d dXdeta;dXdeta.setZero();
        for(int ni=0;ni<elem_size;ni++)
        {
            // R is a vector of double with 4 entries, node_X is a Vector3d
            dXdxi += Rxi[ni]*node_X[ni];
            dXdeta += Reta[ni]*node_X[ni];
        }
        // put them in one column
        MatrixXd Jac(3,2);
        Jac(0,0) = dXdxi(0);
        Jac(1,0) = dXdeta(0);
        Jac(1,0) = dXdxi(1);
        Jac(1,1) = dXdeta(1);
        Jac(2,0) = dXdxi(2);
        Jac(2,1) = dXdeta(2);
        // invert and transpose it
        //Matrix2d Jac_iT = (Jac.inverse()).transpose();
        Vector3d dGamma = dXdxi.cross(dXdeta);
        double Jac_det = sqrt(dGamma.dot(dGamma));
        // save this for the vector
        ip_Jac.push_back(Jac_det);
    }
    // return the vector with all the inverse jacobians
    return ip_Jac;
}

Matrix3d evalJacobian(const std::vector<Vector3d> node_X, double xi, double eta, double zeta)
{
    // eval the inverse Jacobian at given xi, eta, and zeta coordinates
    int elem_size = node_X.size();
    //
    // eval shape functions
    std::vector<double> R;
    // eval derivatives
    std::vector<double> Rxi;
    std::vector<double> Reta;
    std::vector<double> Rzeta;
    if(elem_size == 8){
        R = evalShapeFunctionsR(xi,eta,zeta);
        Rxi = evalShapeFunctionsRxi(xi,eta,zeta);
        Reta = evalShapeFunctionsReta(xi,eta,zeta);
        Rzeta = evalShapeFunctionsRzeta(xi,eta,zeta);
    }
    else if(elem_size == 20){
        R = evalShapeFunctionsQuadraticR(xi,eta,zeta);
        Rxi = evalShapeFunctionsQuadraticRxi(xi,eta,zeta);
        Reta = evalShapeFunctionsQuadraticReta(xi,eta,zeta);
        Rzeta = evalShapeFunctionsQuadraticRzeta(xi,eta,zeta);
    }
    else if(elem_size == 27){
        R = evalShapeFunctionsQuadraticLagrangeR(xi,eta,zeta);
        Rxi = evalShapeFunctionsQuadraticLagrangeRxi(xi,eta,zeta);
        Reta = evalShapeFunctionsQuadraticLagrangeReta(xi,eta,zeta);
        Rzeta = evalShapeFunctionsQuadraticLagrangeRzeta(xi,eta,zeta);
    }
    else if(elem_size == 4){
        R = evalShapeFunctionsTetR(xi,eta,zeta);
        Rxi = evalShapeFunctionsTetRxi(xi,eta,zeta);
        Reta = evalShapeFunctionsTetReta(xi,eta,zeta);
        Rzeta = evalShapeFunctionsTetRzeta(xi,eta,zeta);
    }
    else if(elem_size == 10){
        R = evalShapeFunctionsTetQuadraticR(xi,eta,zeta);
        Rxi = evalShapeFunctionsTetQuadraticRxi(xi,eta,zeta);
        Reta = evalShapeFunctionsTetQuadraticReta(xi,eta,zeta);
        Rzeta = evalShapeFunctionsTetQuadraticRzeta(xi,eta,zeta);
    }
    else{
        throw std::runtime_error("Wrong number of nodes in element!");
    }

    // sum over the 8 nodes

    Vector3d dXdxi;dXdxi.setZero();
    Vector3d dXdeta;dXdeta.setZero();
    Vector3d dXdzeta;dXdzeta.setZero();
    for(int ni=0;ni<elem_size;ni++)
    {
        dXdxi += Rxi[ni]*node_X[ni];
        dXdeta += Reta[ni]*node_X[ni];
        dXdzeta += Rzeta[ni]*node_X[ni];
    }
    // put them in one column
    Matrix3d Jac; Jac<<dXdxi(0),dXdeta(0),dXdzeta(0),dXdxi(1),dXdeta(1),dXdzeta(1),dXdxi(2),dXdeta(2),dXdzeta(2);
    // invert and transpose it
    Matrix3d Jac_iT = (Jac.inverse()).transpose();
    return Jac_iT;
}

//-----------------------------//
// Integration points
//-----------------------------//

std::vector<Vector4d> LineQuadriIP()
{
    // return the integration points of the hex element
    std::vector<Vector4d> IP;
    std::vector<double> pIP = {-1./sqrt(3.),1./sqrt(3.)};
    std::vector<double> wIP = {1.,1.,1.};
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            for(int k=0;k<2;k++){
                IP.push_back(Vector4d(pIP[i],pIP[j],pIP[k],wIP[i]*wIP[j]*wIP[k]));
            }
        }
    }
    return IP;
}

std::vector<Vector4d> LineQuadriIPQuadratic()
{
    // return the integration points of the hex element
    std::vector<Vector4d> IP;
    IP.push_back(Vector4d(-sqrt(3./5),-sqrt(3./5),-sqrt(3./5),pow((5./9),3))); // 1
    IP.push_back(Vector4d(-sqrt(3./5),-sqrt(3./5),0.,pow((5./9),2)*(8./9)));
    IP.push_back(Vector4d(-sqrt(3./5),-sqrt(3./5),+sqrt(3./5),pow((5./9),3)));
    IP.push_back(Vector4d(-sqrt(3./5),0.,-sqrt(3./5),pow((5./9),2)*(8./9)));
    IP.push_back(Vector4d(-sqrt(3./5),0.,0.,pow((8./9),2)*(5./9))); // 5
    IP.push_back(Vector4d(-sqrt(3./5),0.,+sqrt(3./5),pow((5./9),2)*(8./9)));
    IP.push_back(Vector4d(-sqrt(3./5),+sqrt(3./5),-sqrt(3./5),pow((5./9),3)));
    IP.push_back(Vector4d(-sqrt(3./5),+sqrt(3./5),0.,pow((5./9),2)*(8./9)));
    IP.push_back(Vector4d(-sqrt(3./5),+sqrt(3./5),+sqrt(3./5),pow((5./9),3)));
    IP.push_back(Vector4d(0.,-sqrt(3./5),-sqrt(3./5),pow((5./9),2)*(8./9))); // 10
    IP.push_back(Vector4d(0.,-sqrt(3./5),0.,pow((8./9),2)*(5./9)));
    IP.push_back(Vector4d(0.,-sqrt(3./5),+sqrt(3./5),pow((5./9),2)*(8./9)));
    IP.push_back(Vector4d(0.,0.,-sqrt(3./5),pow((8./9),2)*(5./9)));
    IP.push_back(Vector4d(0.,0.,0.,pow((8./9),3)));
    IP.push_back(Vector4d(0.,0.,sqrt(3./5),pow((8./9),2)*(5./9))); // 15
    IP.push_back(Vector4d(0.,+sqrt(3./5),-sqrt(3./5),pow((5./9),2)*(8./9)));
    IP.push_back(Vector4d(0.,+sqrt(3./5),0.,pow((8./9),2)*(5./9)));
    IP.push_back(Vector4d(0.,+sqrt(3./5),+sqrt(3./5),pow((5./9),2)*(8./9)));
    IP.push_back(Vector4d(+sqrt(3./5),-sqrt(3./5),-sqrt(3./5),pow((5./9),3)));
    IP.push_back(Vector4d(+sqrt(3./5),-sqrt(3./5),0.,pow((5./9),2)*(8./9))); // 20
    IP.push_back(Vector4d(+sqrt(3./5),-sqrt(3./5),+sqrt(3./5),pow((5./9),3)));
    IP.push_back(Vector4d(+sqrt(3./5),0.,-sqrt(3./5),pow((5./9),2)*(8./9)));
    IP.push_back(Vector4d(+sqrt(3./5),0.,0.,pow((8./9),2)*(5./9)));
    IP.push_back(Vector4d(+sqrt(3./5),0.,+sqrt(3./5),pow((5./9),2)*(8./9)));
    IP.push_back(Vector4d(+sqrt(3./5),+sqrt(3./5),-sqrt(3./5),pow((5./9),3))); // 25
    IP.push_back(Vector4d(+sqrt(3./5),+sqrt(3./5),0.,pow((5./9),2)*(8./9)));
    IP.push_back(Vector4d(+sqrt(3./5),+sqrt(3./5),+sqrt(3./5),pow((5./9),3)));
    return IP;
}

std::vector<Vector4d> LineQuadriIPTet()
{
    // return the integration points of the hex element
    std::vector<Vector4d> IP;
    IP.push_back(Vector4d(1./4,1./4,1./4,1./6));
    return IP;
}

std::vector<Vector4d> LineQuadriIPTetQuadratic()
{
    // return the integration points of the hex element
    std::vector<Vector4d> IP;
    std::vector<double> pIP = {(5.-sqrt(5))/20.,(5.+3.*sqrt(5))/20.};
    double wIP = 1./24;
    IP.push_back(Vector4d(pIP[0],pIP[0],pIP[0],wIP));
    IP.push_back(Vector4d(pIP[0],pIP[0],pIP[1],wIP));
    IP.push_back(Vector4d(pIP[0],pIP[1],pIP[0],wIP));
    IP.push_back(Vector4d(pIP[1],pIP[0],pIP[0],wIP));
    return IP;
}

std::vector<Vector3d> LineQuadriIPSurfaceQuad()
{
    // return the integration points of the quadratic hex element
    std::vector<Vector3d> IP;
    std::vector<double> pIP = {-sqrt(3.)/3.,sqrt(3.)/3.};
    std::vector<double> wIP = {1.,1.};
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            IP.push_back(Vector3d(pIP[i],pIP[j],wIP[i]*wIP[j]));
        }
    }
    return IP;
}

//-----------------------------//
// Basis functions
//-----------------------------//

std::vector<double> evalShapeFunctionsR(double xi,double eta,double zeta)
{
    // Evaluate shape functions with xi, eta, and zeta
    std::vector<Vector3d> node_Xi = {Vector3d(-1.,-1.,-1.),Vector3d(+1.,-1.,-1.),Vector3d(+1.,+1.,-1.),Vector3d(-1.,+1.,-1.),Vector3d(-1.,-1.,+1.),Vector3d(+1.,-1.,+1.),Vector3d(+1.,+1.,+1.),Vector3d(-1.,+1.,+1.)};
    std::vector<double> R;
    for(int i=0;i<node_Xi.size();i++)
    {
        R.push_back((1./8.)*(1+xi*node_Xi[i](0))*(1+eta*node_Xi[i](1))*(1+zeta*node_Xi[i](2))); //
    }
    return R;
}
std::vector<double> evalShapeFunctionsRxi(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<Vector3d> node_Xi = { Vector3d(-1.,-1.,-1.),Vector3d(+1.,-1.,-1.),Vector3d(+1.,+1.,-1.),Vector3d(-1.,+1.,-1.),Vector3d(-1.,-1.,+1.),Vector3d(+1.,-1.,+1.),Vector3d(+1.,+1.,+1.),Vector3d(-1.,+1.,+1.) };
    std::vector<double> Rxi;
    for(int i=0;i<node_Xi.size();i++)
    {
        Rxi.push_back((1./8.)*(node_Xi[i](0))*(1+eta*node_Xi[i](1))*(1+zeta*node_Xi[i](2)));
    }
    return Rxi;
}
std::vector<double> evalShapeFunctionsReta(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<Vector3d> node_Xi = { Vector3d(-1.,-1.,-1.),Vector3d(+1.,-1.,-1.),Vector3d(+1.,+1.,-1.),Vector3d(-1.,+1.,-1.),Vector3d(-1.,-1.,+1.),Vector3d(+1.,-1.,+1.),Vector3d(+1.,+1.,+1.),Vector3d(-1.,+1.,+1.) };
    std::vector<double> Reta;
    for(int i=0;i<node_Xi.size();i++)
    {
        Reta.push_back((1./8.)*(1+xi*node_Xi[i](0))*(node_Xi[i](1))*(1+zeta*node_Xi[i](2)));
    }
    return Reta;
}
std::vector<double> evalShapeFunctionsRzeta(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<Vector3d> node_Xi = { Vector3d(-1.,-1.,-1.),Vector3d(+1.,-1.,-1.),Vector3d(+1.,+1.,-1.),Vector3d(-1.,+1.,-1.),Vector3d(-1.,-1.,+1.),Vector3d(+1.,-1.,+1.),Vector3d(+1.,+1.,+1.),Vector3d(-1.,+1.,+1.) };
    std::vector<double> Rzeta;
    for(int i=0;i<node_Xi.size();i++)
    {
        Rzeta.push_back((1./8.)*(1+xi*node_Xi[i](0))*(1+eta*node_Xi[i](1))*(node_Xi[i](2)));
    }
    return Rzeta;
}

//-----------------------------//
// Quadratic basis functions
//-----------------------------//

std::vector<double> evalShapeFunctionsQuadraticR(double xi,double eta,double zeta)
{
    // Evaluate shape functions with xi, eta, and zeta
    std::vector<double> R(20,0.);
    R[ 8] = 0.25*(1 - xi*xi)*(1 - eta)*(1 - zeta);
    R[ 9] = 0.25*(1 - eta*eta)*(1 + xi)*(1 - zeta);
    R[10] = 0.25*(1 - xi*xi)*(1 + eta)*(1 - zeta);
    R[11] = 0.25*(1 - eta*eta)*(1 - xi)*(1 - zeta);
    R[12] = 0.25*(1 - xi*xi)*(1 - eta)*(1 + zeta);
    R[13] = 0.25*(1 - eta*eta)*(1 + xi)*(1 + zeta);
    R[14] = 0.25*(1 - xi*xi)*(1 + eta)*(1 + zeta);
    R[15] = 0.25*(1 - eta*eta)*(1 - xi)*(1 + zeta);
    R[16] = 0.25*(1 - zeta*zeta)*(1 - xi)*(1 - eta);
    R[17] = 0.25*(1 - zeta*zeta)*(1 + xi)*(1 - eta);
    R[18] = 0.25*(1 - zeta*zeta)*(1 + xi)*(1 + eta);
    R[19] = 0.25*(1 - zeta*zeta)*(1 - xi)*(1 + eta);
    R[0] = 0.125*(1 - xi)*(1 - eta)*(1 - zeta) - 0.5*(R[ 8] + R[11] + R[16]);
    R[1] = 0.125*(1 + xi)*(1 - eta)*(1 - zeta) - 0.5*(R[ 8] + R[ 9] + R[17]);
    R[2] = 0.125*(1 + xi)*(1 + eta)*(1 - zeta) - 0.5*(R[ 9] + R[10] + R[18]);
    R[3] = 0.125*(1 - xi)*(1 + eta)*(1 - zeta) - 0.5*(R[10] + R[11] + R[19]);
    R[4] = 0.125*(1 - xi)*(1 - eta)*(1 + zeta) - 0.5*(R[12] + R[15] + R[16]);
    R[5] = 0.125*(1 + xi)*(1 - eta)*(1 + zeta) - 0.5*(R[12] + R[13] + R[17]);
    R[6] = 0.125*(1 + xi)*(1 + eta)*(1 + zeta) - 0.5*(R[13] + R[14] + R[18]);
    R[7] = 0.125*(1 - xi)*(1 + eta)*(1 + zeta) - 0.5*(R[14] + R[15] + R[19]);
    return R;
}

std::vector<double> evalShapeFunctionsQuadraticRxi(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Rxi(20,0.);
    Rxi[ 8] = -0.5*xi*(1 - eta)*(1 - zeta);
    Rxi[ 9] =  0.25*(1 - eta*eta)*(1 - zeta);
    Rxi[10] = -0.5*xi*(1 + eta)*(1 - zeta);
    Rxi[11] = -0.25*(1 - eta*eta)*(1 - zeta);
    Rxi[12] = -0.5*xi*(1 - eta)*(1 + zeta);
    Rxi[13] =  0.25*(1 - eta*eta)*(1 + zeta);
    Rxi[14] = -0.5*xi*(1 + eta)*(1 + zeta);
    Rxi[15] = -0.25*(1 - eta*eta)*(1 + zeta);
    Rxi[16] = -0.25*(1 - zeta*zeta)*(1 - eta);
    Rxi[17] =  0.25*(1 - zeta*zeta)*(1 - eta);
    Rxi[18] =  0.25*(1 - zeta*zeta)*(1 + eta);
    Rxi[19] = -0.25*(1 - zeta*zeta)*(1 + eta);
    Rxi[0] = -0.125*(1 - eta)*(1 - zeta) - 0.5*(Rxi[ 8] + Rxi[11] + Rxi[16]);
    Rxi[1] =  0.125*(1 - eta)*(1 - zeta) - 0.5*(Rxi[ 8] + Rxi[ 9] + Rxi[17]);
    Rxi[2] =  0.125*(1 + eta)*(1 - zeta) - 0.5*(Rxi[ 9] + Rxi[10] + Rxi[18]);
    Rxi[3] = -0.125*(1 + eta)*(1 - zeta) - 0.5*(Rxi[10] + Rxi[11] + Rxi[19]);
    Rxi[4] = -0.125*(1 - eta)*(1 + zeta) - 0.5*(Rxi[12] + Rxi[15] + Rxi[16]);
    Rxi[5] =  0.125*(1 - eta)*(1 + zeta) - 0.5*(Rxi[12] + Rxi[13] + Rxi[17]);
    Rxi[6] =  0.125*(1 + eta)*(1 + zeta) - 0.5*(Rxi[13] + Rxi[14] + Rxi[18]);
    Rxi[7] = -0.125*(1 + eta)*(1 + zeta) - 0.5*(Rxi[14] + Rxi[15] + Rxi[19]);

    return Rxi;
}
std::vector<double> evalShapeFunctionsQuadraticReta(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Reta(20,0.);
    Reta[ 8] = -0.25*(1 - xi*xi)*(1 - zeta);
    Reta[ 9] = -0.5*eta*(1 + xi)*(1 - zeta);
    Reta[10] = 0.25*(1 - xi*xi)*(1 - zeta);
    Reta[11] = -0.5*eta*(1 - xi)*(1 - zeta);
    Reta[12] = -0.25*(1 - xi*xi)*(1 + zeta);
    Reta[13] = -0.5*eta*(1 + xi)*(1 + zeta);
    Reta[14] = 0.25*(1 - xi*xi)*(1 + zeta);
    Reta[15] = -0.5*eta*(1 - xi)*(1 + zeta);
    Reta[16] = -0.25*(1 - zeta*zeta)*(1 - xi);
    Reta[17] = -0.25*(1 - zeta*zeta)*(1 + xi);
    Reta[18] =  0.25*(1 - zeta*zeta)*(1 + xi);
    Reta[19] =  0.25*(1 - zeta*zeta)*(1 - xi);
    Reta[0] = -0.125*(1 - xi)*(1 - zeta) - 0.5*(Reta[ 8] + Reta[11] + Reta[16]);
    Reta[1] = -0.125*(1 + xi)*(1 - zeta) - 0.5*(Reta[ 8] + Reta[ 9] + Reta[17]);
    Reta[2] =  0.125*(1 + xi)*(1 - zeta) - 0.5*(Reta[ 9] + Reta[10] + Reta[18]);
    Reta[3] =  0.125*(1 - xi)*(1 - zeta) - 0.5*(Reta[10] + Reta[11] + Reta[19]);
    Reta[4] = -0.125*(1 - xi)*(1 + zeta) - 0.5*(Reta[12] + Reta[15] + Reta[16]);
    Reta[5] = -0.125*(1 + xi)*(1 + zeta) - 0.5*(Reta[12] + Reta[13] + Reta[17]);
    Reta[6] =  0.125*(1 + xi)*(1 + zeta) - 0.5*(Reta[13] + Reta[14] + Reta[18]);
    Reta[7] =  0.125*(1 - xi)*(1 + zeta) - 0.5*(Reta[14] + Reta[15] + Reta[19]);
    return Reta;
}
std::vector<double> evalShapeFunctionsQuadraticRzeta(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Rzeta(20,0.);
    Rzeta[ 8] = -0.25*(1 - xi*xi)*(1 - eta);
    Rzeta[ 9] = -0.25*(1 - eta*eta)*(1 + xi);
    Rzeta[10] = -0.25*(1 - xi*xi)*(1 + eta);
    Rzeta[11] = -0.25*(1 - eta*eta)*(1 - xi);
    Rzeta[12] =  0.25*(1 - xi*xi)*(1 - eta);
    Rzeta[13] =  0.25*(1 - eta*eta)*(1 + xi);
    Rzeta[14] =  0.25*(1 - xi*xi)*(1 + eta);
    Rzeta[15] =  0.25*(1 - eta*eta)*(1 - xi);
    Rzeta[16] = -0.5*zeta*(1 - xi)*(1 - eta);
    Rzeta[17] = -0.5*zeta*(1 + xi)*(1 - eta);
    Rzeta[18] = -0.5*zeta*(1 + xi)*(1 + eta);
    Rzeta[19] = -0.5*zeta*(1 - xi)*(1 + eta);
    Rzeta[0] = -0.125*(1 - xi)*(1 - eta) - 0.5*(Rzeta[ 8] + Rzeta[11] + Rzeta[16]);
    Rzeta[1] = -0.125*(1 + xi)*(1 - eta) - 0.5*(Rzeta[ 8] + Rzeta[ 9] + Rzeta[17]);
    Rzeta[2] = -0.125*(1 + xi)*(1 + eta) - 0.5*(Rzeta[ 9] + Rzeta[10] + Rzeta[18]);
    Rzeta[3] = -0.125*(1 - xi)*(1 + eta) - 0.5*(Rzeta[10] + Rzeta[11] + Rzeta[19]);
    Rzeta[4] =  0.125*(1 - xi)*(1 - eta) - 0.5*(Rzeta[12] + Rzeta[15] + Rzeta[16]);
    Rzeta[5] =  0.125*(1 + xi)*(1 - eta) - 0.5*(Rzeta[12] + Rzeta[13] + Rzeta[17]);
    Rzeta[6] =  0.125*(1 + xi)*(1 + eta) - 0.5*(Rzeta[13] + Rzeta[14] + Rzeta[18]);
    Rzeta[7] =  0.125*(1 - xi)*(1 + eta) - 0.5*(Rzeta[14] + Rzeta[15] + Rzeta[19]);
    return Rzeta;
}

//-----------------------------//
// Quadratic Lagrange basis functions
//-----------------------------//

std::vector<double> evalShapeFunctionsQuadraticLagrangeR(double xi,double eta,double zeta)
{
    // Evaluate shape functions with xi, eta, and zeta
    std::vector<double> R;
    // Vertex nodes
    R.push_back((1./8.)*xi*(xi-1)*eta*(eta-1)*zeta*(zeta-1)); // Node 1
    R.push_back((1./8.)*xi*(xi+1)*eta*(eta-1)*zeta*(zeta-1)); // Node 2
    R.push_back((1./8.)*xi*(xi+1)*eta*(eta+1)*zeta*(zeta-1)); // Node 3
    R.push_back((1./8.)*xi*(xi-1)*eta*(eta+1)*zeta*(zeta-1)); // Node 4
    R.push_back((1./8.)*xi*(xi-1)*eta*(eta-1)*zeta*(zeta+1)); // Node 5
    R.push_back((1./8.)*xi*(xi+1)*eta*(eta-1)*zeta*(zeta+1)); // Node 6
    R.push_back((1./8.)*xi*(xi+1)*eta*(eta+1)*zeta*(zeta+1)); // Node 7
    R.push_back((1./8.)*xi*(xi-1)*eta*(eta+1)*zeta*(zeta+1)); // Node 8
    // Bottom midedge
    R.push_back((1./4.)*(1-xi*xi)*eta*(eta-1)*zeta*(zeta-1)); // Node 9
    R.push_back((1./4.)*xi*(xi+1)*(1-eta*eta)*zeta*(zeta-1)); // Node 10
    R.push_back((1./4.)*(1-xi*xi)*eta*(eta+1)*zeta*(zeta-1)); // Node 11
    R.push_back((1./4.)*xi*(xi-1)*(1-eta*eta)*zeta*(zeta-1)); // Node 12
    // Middle midedge
    R.push_back((1./4.)*xi*(xi-1)*eta*(eta-1)*(1-zeta*zeta)); // Node 13
    R.push_back((1./4.)*xi*(xi+1)*eta*(eta-1)*(1-zeta*zeta)); // Node 14
    R.push_back((1./4.)*xi*(xi+1)*eta*(eta+1)*(1-zeta*zeta)); // Node 15
    R.push_back((1./4.)*xi*(xi-1)*eta*(eta+1)*(1-zeta*zeta)); // Node 16
    // Top midedge
    R.push_back((1./4.)*(1-xi*xi)*eta*(eta-1)*zeta*(zeta+1)); // Node 17
    R.push_back((1./4.)*xi*(xi+1)*(1-eta*eta)*zeta*(zeta+1)); // Node 18
    R.push_back((1./4.)*(1-xi*xi)*eta*(eta+1)*zeta*(zeta+1)); // Node 19
    R.push_back((1./4.)*xi*(xi-1)*(1-eta*eta)*zeta*(zeta+1)); // Node 20
    // Midface
    R.push_back((1./2.)*(1-xi*xi)*(1-eta*eta)*zeta*(zeta-1)); // Node 21
    R.push_back((1./2.)*(1-xi*xi)*eta*(eta-1)*(1-zeta*zeta)); // Node 22
    R.push_back((1./2.)*xi*(xi+1)*(1-eta*eta)*(1-zeta*zeta)); // Node 23
    R.push_back((1./2.)*(1-xi*xi)*eta*(eta+1)*(1-zeta*zeta)); // Node 24
    R.push_back((1./2.)*xi*(xi-1)*(1-eta*eta)*(1-zeta*zeta)); // Node 25
    R.push_back((1./2.)*(1-xi*xi)*(1-eta*eta)*zeta*(zeta+1)); // Node 26
    // Center node
    R.push_back((1-xi*xi)*(1-eta*eta)*(1-zeta*zeta)); // Node 27
    return R;
}

std::vector<double> evalShapeFunctionsQuadraticLagrangeRxi(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Rxi;
    // Vertex nodes
    Rxi.push_back((1./8.)*(2*xi-1)*eta*(eta-1)*zeta*(zeta-1)); // Node 1
    Rxi.push_back((1./8.)*(2*xi+1)*eta*(eta-1)*zeta*(zeta-1)); // Node 2
    Rxi.push_back((1./8.)*(2*xi+1)*eta*(eta+1)*zeta*(zeta-1)); // Node 3
    Rxi.push_back((1./8.)*(2*xi-1)*eta*(eta+1)*zeta*(zeta-1)); // Node 4
    Rxi.push_back((1./8.)*(2*xi-1)*eta*(eta-1)*zeta*(zeta+1)); // Node 5
    Rxi.push_back((1./8.)*(2*xi+1)*eta*(eta-1)*zeta*(zeta+1)); // Node 6
    Rxi.push_back((1./8.)*(2*xi+1)*eta*(eta+1)*zeta*(zeta+1)); // Node 7
    Rxi.push_back((1./8.)*(2*xi-1)*eta*(eta+1)*zeta*(zeta+1)); // Node 8
    // Bottom midedge
    Rxi.push_back((1./4.)*(-2*xi)*eta*(eta-1)*zeta*(zeta-1)); // Node 9
    Rxi.push_back((1./4.)*(2*xi+1)*(1-eta*eta)*zeta*(zeta-1)); // Node 10
    Rxi.push_back((1./4.)*(-2*xi)*eta*(eta+1)*zeta*(zeta-1)); // Node 11
    Rxi.push_back((1./4.)*(2*xi-1)*(1-eta*eta)*zeta*(zeta-1)); // Node 12
    // Middle midedge
    Rxi.push_back((1./4.)*(2*xi-1)*eta*(eta-1)*(1-zeta*zeta)); // Node 13
    Rxi.push_back((1./4.)*(2*xi+1)*eta*(eta-1)*(1-zeta*zeta)); // Node 14
    Rxi.push_back((1./4.)*(2*xi+1)*eta*(eta+1)*(1-zeta*zeta)); // Node 15
    Rxi.push_back((1./4.)*(2*xi-1)*eta*(eta+1)*(1-zeta*zeta)); // Node 16
    // Top midedge
    Rxi.push_back((1./4.)*(-2*xi)*eta*(eta-1)*zeta*(zeta+1)); // Node 17
    Rxi.push_back((1./4.)*(2*xi+1)*(1-eta*eta)*zeta*(zeta+1)); // Node 18
    Rxi.push_back((1./4.)*(-2*xi)*eta*(eta+1)*zeta*(zeta+1)); // Node 19
    Rxi.push_back((1./4.)*(2*xi-1)*(1-eta*eta)*zeta*(zeta+1)); // Node 20
    // Midface
    Rxi.push_back((1./2.)*(-2*xi)*(1-eta*eta)*zeta*(zeta-1)); // Node 21
    Rxi.push_back((1./2.)*(-2*xi)*eta*(eta-1)*(1-zeta*zeta)); // Node 22
    Rxi.push_back((1./2.)*(2*xi+1)*(1-eta*eta)*(1-zeta*zeta)); // Node 23
    Rxi.push_back((1./2.)*(-2*xi)*eta*(eta+1)*(1-zeta*zeta)); // Node 24
    Rxi.push_back((1./2.)*(2*xi-1)*(1-eta*eta)*(1-zeta*zeta)); // Node 25
    Rxi.push_back((1./2.)*(-2*xi)*(1-eta*eta)*zeta*(zeta+1)); // Node 26
    // Center node
    Rxi.push_back((-2*xi)*(1-eta*eta)*(1-zeta*zeta)); // Node 27
    return Rxi;
}
std::vector<double> evalShapeFunctionsQuadraticLagrangeReta(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Reta;
    // Vertex nodes
    Reta.push_back((1./8.)*xi*(xi-1)*(2*eta-1)*zeta*(zeta-1)); // Node 1
    Reta.push_back((1./8.)*xi*(xi+1)*(2*eta-1)*zeta*(zeta-1)); // Node 2
    Reta.push_back((1./8.)*xi*(xi+1)*(2*eta+1)*zeta*(zeta-1)); // Node 3
    Reta.push_back((1./8.)*xi*(xi-1)*(2*eta+1)*zeta*(zeta-1)); // Node 4
    Reta.push_back((1./8.)*xi*(xi-1)*(2*eta-1)*zeta*(zeta+1)); // Node 5
    Reta.push_back((1./8.)*xi*(xi+1)*(2*eta-1)*zeta*(zeta+1)); // Node 6
    Reta.push_back((1./8.)*xi*(xi+1)*(2*eta+1)*zeta*(zeta+1)); // Node 7
    Reta.push_back((1./8.)*xi*(xi-1)*(2*eta+1)*zeta*(zeta+1)); // Node 8
    // Bottom midedge
    Reta.push_back((1./4.)*(1-xi*xi)*(2*eta-1)*zeta*(zeta-1)); // Node 9
    Reta.push_back((1./4.)*xi*(xi+1)*(-2*eta)*zeta*(zeta-1)); // Node 10
    Reta.push_back((1./4.)*(1-xi*xi)*(2*eta+1)*zeta*(zeta-1)); // Node 11
    Reta.push_back((1./4.)*xi*(xi-1)*(-2*eta)*zeta*(zeta-1)); // Node 12
    // Middle midedge
    Reta.push_back((1./4.)*xi*(xi-1)*(2*eta-1)*(1-zeta*zeta)); // Node 13
    Reta.push_back((1./4.)*xi*(xi+1)*(2*eta-1)*(1-zeta*zeta)); // Node 14
    Reta.push_back((1./4.)*xi*(xi+1)*(2*eta+1)*(1-zeta*zeta)); // Node 15
    Reta.push_back((1./4.)*xi*(xi-1)*(2*eta+1)*(1-zeta*zeta)); // Node 16
    // Top midedge
    Reta.push_back((1./4.)*(1-xi*xi)*(2*eta-1)*zeta*(zeta+1)); // Node 17
    Reta.push_back((1./4.)*xi*(xi+1)*(-2*eta)*zeta*(zeta+1)); // Node 18
    Reta.push_back((1./4.)*(1-xi*xi)*(2*eta+1)*zeta*(zeta+1)); // Node 19
    Reta.push_back((1./4.)*xi*(xi-1)*(-2*eta)*zeta*(zeta+1)); // Node 20
    // Midface
    Reta.push_back((1./2.)*(1-xi*xi)*(-2*eta)*zeta*(zeta-1)); // Node 21
    Reta.push_back((1./2.)*(1-xi*xi)*(2*eta-1)*(1-zeta*zeta)); // Node 22
    Reta.push_back((1./2.)*xi*(xi+1)*(-2*eta)*(1-zeta*zeta)); // Node 23
    Reta.push_back((1./2.)*(1-xi*xi)*(2*eta+1)*(1-zeta*zeta)); // Node 24
    Reta.push_back((1./2.)*xi*(xi-1)*(-2*eta)*(1-zeta*zeta)); // Node 25
    Reta.push_back((1./2.)*(1-xi*xi)*(-2*eta)*zeta*(zeta+1)); // Node 26
    // Center node
    Reta.push_back((1-xi*xi)*(-2*eta)*(1-zeta*zeta)); // Node 27
    return Reta;
}
std::vector<double> evalShapeFunctionsQuadraticLagrangeRzeta(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Rzeta;
    // Vertex nodes
    Rzeta.push_back((1./8.)*xi*(xi-1)*eta*(eta-1)*(2*zeta-1)); // Node 1
    Rzeta.push_back((1./8.)*xi*(xi+1)*eta*(eta-1)*(2*zeta-1)); // Node 2
    Rzeta.push_back((1./8.)*xi*(xi+1)*eta*(eta+1)*(2*zeta-1)); // Node 3
    Rzeta.push_back((1./8.)*xi*(xi-1)*eta*(eta+1)*(2*zeta-1)); // Node 4
    Rzeta.push_back((1./8.)*xi*(xi-1)*eta*(eta-1)*(2*zeta+1)); // Node 5
    Rzeta.push_back((1./8.)*xi*(xi+1)*eta*(eta-1)*(2*zeta+1)); // Node 6
    Rzeta.push_back((1./8.)*xi*(xi+1)*eta*(eta+1)*(2*zeta+1)); // Node 7
    Rzeta.push_back((1./8.)*xi*(xi-1)*eta*(eta+1)*(2*zeta+1)); // Node 8
    // Bottom midedge
    Rzeta.push_back((1./4.)*(1-xi*xi)*eta*(eta-1)*(2*zeta-1)); // Node 9
    Rzeta.push_back((1./4.)*xi*(xi+1)*(1-eta*eta)*(2*zeta-1)); // Node 10
    Rzeta.push_back((1./4.)*(1-xi*xi)*eta*(eta+1)*(2*zeta-1)); // Node 11
    Rzeta.push_back((1./4.)*xi*(xi-1)*(1-eta*eta)*(2*zeta-1)); // Node 12
    // Middle midedge
    Rzeta.push_back((1./4.)*xi*(xi-1)*eta*(eta-1)*(-2*zeta)); // Node 13
    Rzeta.push_back((1./4.)*xi*(xi+1)*eta*(eta-1)*(-2*zeta)); // Node 14
    Rzeta.push_back((1./4.)*xi*(xi+1)*eta*(eta+1)*(-2*zeta)); // Node 15
    Rzeta.push_back((1./4.)*xi*(xi-1)*eta*(eta+1)*(-2*zeta)); // Node 16
    // Top midedge
    Rzeta.push_back((1./4.)*(1-xi*xi)*eta*(eta-1)*(2*zeta+1)); // Node 17
    Rzeta.push_back((1./4.)*xi*(xi+1)*(1-eta*eta)*(2*zeta+1)); // Node 18
    Rzeta.push_back((1./4.)*(1-xi*xi)*eta*(eta+1)*(2*zeta+1)); // Node 19
    Rzeta.push_back((1./4.)*xi*(xi-1)*(1-eta*eta)*(2*zeta+1)); // Node 20
    // Midface
    Rzeta.push_back((1./2.)*(1-xi*xi)*(1-eta*eta)*(2*zeta-1)); // Node 21
    Rzeta.push_back((1./2.)*(1-xi*xi)*eta*(eta-1)*(-2*zeta)); // Node 22
    Rzeta.push_back((1./2.)*xi*(xi+1)*(1-eta*eta)*(-2*zeta)); // Node 23
    Rzeta.push_back((1./2.)*(1-xi*xi)*eta*(eta+1)*(-2*zeta)); // Node 24
    Rzeta.push_back((1./2.)*xi*(xi-1)*(1-eta*eta)*(-2*zeta)); // Node 25
    Rzeta.push_back((1./2.)*(1-xi*xi)*(1-eta*eta)*(2*zeta+1)); // Node 26
    // Center node
    Rzeta.push_back((1-xi*xi)*(1-eta*eta)*(-2*zeta)); // Node 27
    return Rzeta;
}

//-----------------------------//
// Tetrahedral basis functions
//-----------------------------//

std::vector<double> evalShapeFunctionsTetR(double xi,double eta,double zeta)
{
    // Evaluate shape functions with xi, eta, and zeta
    std::vector<double> R;
    R.push_back(eta);
    R.push_back(zeta);
    R.push_back(1.-xi-eta-zeta);
    R.push_back(xi);
    return R;
}
std::vector<double> evalShapeFunctionsTetRxi(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Rxi;
    Rxi.push_back(0.);
    Rxi.push_back(0.);
    Rxi.push_back(-1.);
    Rxi.push_back(1.);
    return Rxi;
}
std::vector<double> evalShapeFunctionsTetReta(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Reta;
    Reta.push_back(1.);
    Reta.push_back(0.);
    Reta.push_back(-1.);
    Reta.push_back(0.);
    return Reta;
}
std::vector<double> evalShapeFunctionsTetRzeta(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Rzeta;
    Rzeta.push_back(0.);
    Rzeta.push_back(1.);
    Rzeta.push_back(-1.);
    Rzeta.push_back(0.);
    return Rzeta;
}

//-----------------------------//
// Tetrahedral quadratic basis functions
//-----------------------------//

std::vector<double> evalShapeFunctionsTetQuadraticR(double xi,double eta,double zeta)
{
    // Evaluate shape functions with xi, eta, and zeta
    std::vector<double> R(10,0.);
    double r1 = 1.0 - xi - eta - zeta;
    double r2 = xi;
    double r3 = eta;
    double r4 = zeta;
    R[0] = r1*(2.0*r1 - 1.0);
    R[1] = r2*(2.0*r2 - 1.0);
    R[2] = r3*(2.0*r3 - 1.0);
    R[3] = r4*(2.0*r4 - 1.0);
    R[4] = 4.0*r1*r2;
    R[5] = 4.0*r2*r3;
    R[6] = 4.0*r3*r1;
    R[7] = 4.0*r1*r4;
    R[8] = 4.0*r2*r4;
    R[9] = 4.0*r3*r4;
    return R;
}
std::vector<double> evalShapeFunctionsTetQuadraticRxi(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Rxi(10,0.);
    Rxi[0] = -3.0 + 4.0*xi + 4.0*(eta + zeta);
    Rxi[1] =  4.0*xi - 1.0;
    Rxi[2] =  0.0;
    Rxi[3] =  0.0;
    Rxi[4] =  4.0 - 8.0*xi - 4.0*(eta + zeta);
    Rxi[5] =  4.0*eta;
    Rxi[6] = -4.0*eta;
    Rxi[7] = -4.0*zeta;
    Rxi[8] =  4.0*zeta;
    Rxi[9] =  0.0;
    return Rxi;
}
std::vector<double> evalShapeFunctionsTetQuadraticReta(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Reta(10,0.);
    Reta[0] = -3.0 + 4.0*eta + 4.0*(xi + zeta);
    Reta[1] =  0.0;
    Reta[2] =  4.0*eta - 1.0;
    Reta[3] =  0.0;
    Reta[4] = -4.0*xi;
    Reta[5] =  4.0*xi;
    Reta[6] =  4.0 - 8.0*eta - 4.0*(xi + zeta);
    Reta[7] = -4.0*zeta;
    Reta[8] =  0.0;
    Reta[9] =  4.0*zeta;
    return Reta;
}
std::vector<double> evalShapeFunctionsTetQuadraticRzeta(double xi,double eta,double zeta)
{
    // Evaluate the shape function derivative
    std::vector<double> Rzeta(10,0.);
    Rzeta[0] = -3.0 + 4.0*zeta + 4.0*(xi + eta);
    Rzeta[1] =  0.0;
    Rzeta[2] =  0.0;
    Rzeta[3] =  4.0*zeta - 1.0;
    Rzeta[4] = -4.0*xi;
    Rzeta[5] =  0.0;
    Rzeta[6] = -4.0*eta;
    Rzeta[7] =  4.0 - 8.0*zeta - 4.0*(xi + eta);
    Rzeta[8] =  4.0*xi;
    Rzeta[9] =  4.0*eta;
    return Rzeta;
}

std::vector<double> evalShapeFunctionsSurfaceQuadR(double xi,double eta)
{
    std::vector<Vector2d> node_Xi = {Vector2d(-1.,-1.),Vector2d(+1.,-1.),Vector2d(+1.,+1.),Vector2d(-1.,+1.)};
    std::vector<double> R;
    for(int i=0;i<node_Xi.size();i++)
    {
        R.push_back((1./4.)*(1+xi*node_Xi[i](0))*(1+eta*node_Xi[i](1)));
    }
    return R;
}
std::vector<double> evalShapeFunctionsSurfaceQuadRxi(double xi,double eta)
{
    std::vector<Vector2d> node_Xi = {Vector2d(-1.,-1.),Vector2d(+1.,-1.),Vector2d(+1.,+1.),Vector2d(-1.,+1.)};
    std::vector<double> Rxi;
    for(int i=0;i<node_Xi.size();i++)
    {
        Rxi.push_back((1./4.)*(node_Xi[i](0))*(1+eta*node_Xi[i](1)));
    }
    return Rxi;
}
std::vector<double> evalShapeFunctionsSurfaceQuadReta(double xi,double eta)
{
    std::vector<Vector2d> node_Xi = {Vector2d(-1.,-1.),Vector2d(+1.,-1.),Vector2d(+1.,+1.),Vector2d(-1.,+1.)};
    std::vector<double> Reta;
    for(int i=0;i<node_Xi.size();i++)
    {
        Reta.push_back((1./4.)*(1+xi*node_Xi[i](0))*(node_Xi[i](1)));
    }
    return Reta;
}
