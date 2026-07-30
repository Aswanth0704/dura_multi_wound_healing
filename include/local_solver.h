/*

WOUND

This code is the implementation of the wound model
Particularly, the Dalai Lama Wound Healing, or DaLaWoHe

*/

#ifndef local_solver_h
#define local_solver_h

#include <omp.h>
#include <vector>
#include <map>
#include <Eigen/Dense> // most of the vector functions I will need inside of an element

using namespace Eigen;


//========================================================//
// LOCAL PROBLEM: structural update
//
void localWoundProblemExplicit(
        double dt, const std::vector<double> &local_parameters,const Vector3d &X,
        double C,double rho,const Matrix3d &FF,
        const double &phif_0, const Vector3d &a0_0, const Vector3d &s0_0, const Vector3d &n0_0, const double &kappa_0, const Vector3d &lamdaP_0,
        double &phif, Vector3d &a0, Vector3d &s0, Vector3d &n0, double &kappa, Vector3d &lamdaP,
        VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadC);

void evalForwardEulerUpdate(double local_dt, const std::vector<double> &local_parameters, double c,double rho,const Matrix3d &FF, const Matrix3d &CC,
                            const double &phif, const Vector3d &a0, const Vector3d &s0, const double &kappa, const Vector3d &lamdaP,
                            double &phif_dot, Vector3d &a0_dot, double &kappa_dot, Vector3d &lamdaP_dot);
//
//========================================================//


//========================================================//
// LOCAL PROBLEM: structural update
//
void localWoundProblemExplicit2d(
        double dt, const std::vector<double> &local_parameters,double C,double rho,const Matrix3d &FF,
        const double &phif_0, const Vector3d &a0_0, const Vector3d &s0_0, const double &kappa_0, const Vector3d &lamdaP_0,
        double &phif, Vector3d &a0, Vector3d &s0, double &kappa, Vector3d &lamdaP,
        VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadC);

void evalForwardEulerUpdate2d(double local_dt, const std::vector<double> &local_parameters, double c,double rho,const Matrix3d &FF, const Matrix3d &CC,
                              const double &phif, const Vector3d &a0, const Vector3d &s0, const double &kappa, const Vector3d &lamdaP,
                              double &phif_dot, Vector3d &a0_dot, double &kappa_dot, Vector3d &lamdaP_dot);
//
//========================================================//



//========================================================//
// LOCAL PROBLEM: structural update
//
void localWoundProblemImplicit(
        double dt, const std::vector<double> &local_parameters,
        const double &c,const double &rho,const Matrix3d &FF,
        const double &phif_0, const Vector3d &a0_0, const Vector3d &s0_0, const Vector3d &n0_0,
        const double &kappa_0, const Vector3d &lamdaP_0,
        double &phif, Vector3d &a0, Vector3d &s0, Vector3d &n0,
        double &kappa, Vector3d &lamdaP,
        VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadc);

//
//========================================================//



//========================================================//
// LOCAL PROBLEM: structural update
//
void localWoundProblemImplicit2d(
        double dt, const std::vector<double> &local_parameters,
        double C,double rho,const Matrix3d &FF,
        double phif_0, const Vector3d &a0_0, const Vector3d &s0_0, double kappa_0, const Vector3d &lamdaP_0,
        double &phif, Vector3d &a0, Vector3d &s0, double &kappa, Vector3d &lamdaP,
        VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadC);
//
//========================================================//



#endif
