/*

WOUND

This code is the implementation of the wound model
Particularly, the Dalai Lama Wound Healing, or DaLaWoHe

*/

#ifndef wound_h
#define wound_h

#include <omp.h>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense> // most of the vector functions I will need inside of an element

using namespace Eigen;


//========================================================//
// RESIDUAL AND TANGENT
//
void evalWound(
double dt, double time, double time_final,
const std::vector<Matrix3d> &ip_Jac,
const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
std::vector<Matrix3d> &ip_strain,std::vector<Matrix3d> &ip_stress, const std::vector<double> &node_rho_0, const std::vector<double> &node_c_0, //
const std::vector<double> &ip_phif_0,const std::vector<Vector3d> &ip_a0_0,const std::vector<Vector3d> &ip_s0_0,const std::vector<Vector3d> &ip_n0_0,const std::vector<double> &ip_kappa_0, const std::vector<Vector3d> &ip_lamdaP_0, //
const std::vector<double> &node_rho, const std::vector<double> &node_c,
std::vector<double> &ip_phif, std::vector<Vector3d> &ip_a0, std::vector<Vector3d> &ip_s0, std::vector<Vector3d> &ip_n0, std::vector<double> &ip_kappa, std::vector<Vector3d> &ip_lamdaP, //
std::vector<Vector3d> &ip_lamdaE,
const std::vector<Vector3d> &node_x, const std::vector<Vector3d> &node_X,
std::vector<Vector3d> &ip_dphifdu, std::vector<double> &ip_dphifdrho, std::vector<double> &ip_dphifdc,
VectorXd &Re_x,MatrixXd &Ke_x_x,MatrixXd &Ke_x_rho,MatrixXd &Ke_x_c,
VectorXd &Re_rho,MatrixXd &Ke_rho_x, MatrixXd &Ke_rho_rho,MatrixXd &Ke_rho_c,
VectorXd &Re_c,MatrixXd &Ke_c_x,MatrixXd &Ke_c_rho,MatrixXd &Ke_c_c);
//
//========================================================//



//========================================================//
// EVAL SOURCE AND FLUX 
//
void evalFluxesSources(const std::vector<double> &global_parameters, const double& phif,Vector3d a0, const Vector3d& a0_0, const Vector3d& s0_0, const Vector3d& n0_0,double kappa, const Vector3d& lamdaP,
                       const Matrix3d& FF, const double& rho, const double& c, const Vector3d& Grad_rho, const Vector3d& Grad_c,
                       Matrix3d & SS,Vector3d &Q_rho,double &S_rho, Vector3d &Q_c,double &S_c);

//
//========================================================//


//========================================================//
// OUTPUT FUNCTION: eval FF at coordinates xi, eta
//
/*
Matrix3d evalWoundFF(
double dt,
const std::vector<Matrix3d> &ip_Jac,
const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
const std::vector<double> &node_X,
std::vector<double> &ip_phif, std::vector<Vector3d> &ip_a0, std::vector<double> &ip_kappa, std::vector<Vector3d> &ip_lamdaP,
const std::vector<Vector3d> &node_x,const std::vector<double> &node_rho, const std::vector<double> &node_c,
double xi, double eta, double zeta);
*/
 //
//========================================================//


/////////////////////////////////////////////////////////////////////////////////////////
// NUMERICAL CHECKS
/////////////////////////////////////////////////////////////////////////////////////////

//-----------------------------//
// Eval strain energy 
//
void evalPsif(const std::vector<double> &global_parameters,double kappa, double I1e,double I4e,double &Psif,double &Psif1,double &Psif4);
//
//-----------------------------//

//-----------------------------//
// Eval passive reference stress
//
void evalSS(const std::vector<double> &global_parameters, double phif, Vector3d a0, Vector3d s0, Vector3d n0, double kappa, double lamdaP_a,double lamdaP_s,
        double lamdaP_N,const Matrix3d &CC,double rho, double c, Matrix3d &SS_pas,Matrix3d &SS_act, Matrix3d&SS_vol);
void evalQ(const std::vector<double> &global_parameters, const double& phif,Vector3d a0,Vector3d s0,Vector3d n0, double kappa, const Vector3d& lamdaP,
           const Matrix3d& CC, const double& rho, const double& c, const Vector3d& Grad_rho, const Vector3d& Grad_c, Vector3d &Q_rho, Vector3d &Q_c);
void evalS(const std::vector<double> &global_parameters, const double& phif,Vector3d a0,Vector3d s0,Vector3d n0, double kappa, const Vector3d& lamdaP,
           const Matrix3d& CC, const double& rho, const double& c, double &S_rho, double &S_c);
//
//-----------------------------//

/*
//-----------------------------//
// Eval residuals only
//
void evalWoundRes(
double dt,
const std::vector<Matrix3d> &ip_Jac,
const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
const std::vector<double> &node_rho_0, const std::vector<double> &node_c_0,
const std::vector<double> &ip_phif_0,const std::vector<Vector3d> &ip_a0_0,const std::vector<double> &ip_kappa_0, const std::vector<Vector3d> &ip_lamdaP_0,
const std::vector<double> &node_rho, const std::vector<double> &node_c,
std::vector<double> &ip_phif, std::vector<Vector3d> &ip_a0, std::vector<double> &ip_kappa, std::vector<Vector3d> &ip_lamdaP,
const std::vector<Vector3d> &node_x,
VectorXd &Re_x,VectorXd &Re_rho,VectorXd &Re_c);
//
//-----------------------------//
*/

void evalBC(int surface_boundary_flag, const std::vector<double> &elem_jac_IP, const std::vector<double> &global_parameters,
            const std::vector<double> &node_rho, const std::vector<double> &node_c, const std::vector<double> &ip_phif,
            const std::vector<Vector3d> &node_x, const std::vector<Vector3d> &node_X,
            const std::vector<Vector3d> &node_dphifdu, const std::vector<double> &node_dphifdrho, const std::vector<double> &node_dphifdc,
            VectorXd &Re_x_surf,MatrixXd &Ke_x_x_surf,MatrixXd &Ke_x_rho_surf,MatrixXd &Ke_x_c_surf,
            VectorXd &Re_rho_surf,MatrixXd &Ke_rho_x_surf, MatrixXd &Ke_rho_rho_surf,MatrixXd &Ke_rho_c_surf,
            VectorXd &Re_c_surf,MatrixXd &Ke_c_x_surf,MatrixXd &Ke_c_rho_surf,MatrixXd &Ke_c_c_surf);

void evalWoundMechanics(double dt, double time, double time_final,
                        const std::vector<Matrix3d> &ip_Jac, const std::vector<double> &global_parameters,
                        std::vector<Matrix3d> &ip_strain,std::vector<Matrix3d> &ip_stress, const std::vector<double> &node_rho_0, const std::vector<double> &node_c_0,
                        const std::vector<double> &ip_phif_0,const std::vector<Vector3d> &ip_a0_0,const std::vector<Vector3d> &ip_s0_0,const std::vector<Vector3d> &ip_n0_0,const std::vector<double> &ip_kappa_0, const std::vector<Vector3d> &ip_lamdaP_0, //
                        const std::vector<double> &node_rho, const std::vector<double> &node_c,
                        std::vector<double> &ip_phif, std::vector<Vector3d> &ip_a0, std::vector<Vector3d> &ip_s0, std::vector<Vector3d> &ip_n0, std::vector<double> &ip_kappa, std::vector<Vector3d> &ip_lamdaP,
                        const std::vector<Vector3d> &node_x, VectorXd &Re_x,MatrixXd &Ke_x_x);

#endif
