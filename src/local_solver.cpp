// last date modified: Sept, 16, 2025

/*

WOUND

This code is the implementation of the DaLaWoHe

*/

#include <omp.h>
#include <iomanip>
#include "wound.h"
#include "local_solver.h"
#include "mechanosensing.h"
#include "element_functions.h"
#include <iostream>
#include <cmath>
#include <map>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <myMeshGenerator.h>
#include "file_io.h"
#include "solver.h"
#include <string>
#include <ctime>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;


//========================================================//
// EXPLICIT LOCAL PROBLEM: structural update
//========================================================//
void localWoundProblemExplicit(
        double dt, const std::vector<double> &local_parameters, const Vector3d &X,
        double c,double rho,const Matrix3d &FF,
        const double &phif_0, const Vector3d &a0_0, const Vector3d &s0_0, const Vector3d &n0_0, const double &kappa_0, const Vector3d &lamdaP_0,
        double &phif, Vector3d &a0, Vector3d &s0, Vector3d &n0, double &kappa, Vector3d &lamdaP,
        VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadc)
{
    //---------------------------------//
    //
    // INPUT
    // 	matParam: material parameters
    //	rho: value of the cell density at the point
    //	c: concentration at the point
    //	CC: deformation at the point
    //	Theta_t: value of the parameters at previous time step
    //
    // OUTPUT
    //	Theta: value of the parameters at the current time step
    //	dThetadCC: derivative of Theta wrt global mechanics (CC)
    // 	dThetadrho: derivative of Theta wrt global rho
    // 	dThetadc: derivative of Theta wrt global C
    //
    //---------------------------------//
   
    //---------------------------------//
    // Parameters
    //
    // collagen fraction
    double p_phi = local_parameters[0]; // production by fibroblasts, natural rate
    double p_phi_c = local_parameters[1]; // production up-regulation, weighted by C and rho
    double p_phi_theta = local_parameters[2]; //production regulated by stretch
    double K_phi_c = local_parameters[3]; // saturation of C effect on deposition
    double K_phi_rho = local_parameters[4]; // saturation of collagen fraction itself
    double d_phi = local_parameters[5]; // rate of degradation
    double d_phi_rho_c = local_parameters[6]; // rate of degradation
    //
    // fiber alignment
    double tau_omega = local_parameters[7]; // time constant for angular reorientation
    //
    // dispersion parameter
    double tau_kappa = local_parameters[8]; // time constant
    double gamma_kappa = local_parameters[9]; // exponent of the principal stretch ratio
    //
    // permanent contracture/growth
    double tau_lamdaP_a = local_parameters[10]; // time constant for direction a
    double tau_lamdaP_s = local_parameters[11]; // time constant for direction s
    double tau_lamdaP_n = local_parameters[12]; // time constant for direction s
    //
    double vartheta_e = local_parameters[13]; // mechanosensing response
    double gamma_theta = local_parameters[14]; // exponent of the Heaviside function
    // solution parameters
    double tol_local = local_parameters[15]; // local tolerance
    double time_step_ratio = local_parameters[16]; // time step ratio
    double max_iter = local_parameters[17]; // max local iter
    //
    // other local stuff
    double local_dt = dt/time_step_ratio;
    //
    double PIE = 3.14159;

    double lamdaE_a, lamdaE_s, lamdaE_n;

    // Use these to get at the six elements of CC that we need
    VectorXd voigt_table_I(6);
    voigt_table_I(0) = 0; voigt_table_I(1) = 1; voigt_table_I(2) = 2;
    voigt_table_I(3) = 1; voigt_table_I(4) = 0; voigt_table_I(5) = 0;
    VectorXd voigt_table_J(6);
    voigt_table_J(0) = 0; voigt_table_J(1) = 1; voigt_table_J(2) = 2;
    voigt_table_J(3) = 2; voigt_table_J(4) = 2; voigt_table_J(5) = 1;

    //---------------------------------//
    // KINEMATICS
    //---------------------------------//
    Matrix3d CC = FF.transpose()*FF;
    Matrix3d CCinv = CC.inverse();
    // re-compute basis a0, s0, n0. (If not always vertical, could find n0 with cross product or rotations. Be careful of sign.)
    // fiber tensor in the reference
    Matrix3d a0a0, s0s0, n0n0;
    // recompute split
    Matrix3d FFg, FFginv, FFe;
    // elastic strain
    Matrix3d CCe, CCeinv;
    //
    //---------------------------------//

    //---------------------------------//
    // LOCAL EXPLICIT FORWARD-EULER ITERATION
    //---------------------------------//
    //
    // Declare Variables
    // New phi, a0x, a0y, kappa, lamdaPa, lamdaPs
    phif = phif_0;
    // make sure it is unit length
    a0 = a0_0/(sqrt(a0_0.dot(a0_0)));
    s0 = s0_0/(sqrt(s0_0.dot(s0_0)));
    n0 = n0_0/(sqrt(n0_0.dot(n0_0)));
    kappa = kappa_0;
    lamdaP = lamdaP_0;
    //
    double phif_dot, kappa_dot;
    Vector3d lamdaP_dot, a0_dot;
    //
    VectorXd dThetadCC_num(48); dThetadCC_num.setZero();
    VectorXd dThetadrho_num(8); dThetadrho_num.setZero();
    VectorXd dThetadc_num(8); dThetadc_num.setZero();
    
    // Deadband for permanent (plastic) growth: no remodelling while the elastic
    // stretch stays inside [lowlim, uplim].
    //
    // These used to be 0.95 / 1.05, which is INCOMPATIBLE with the physiological
    // prestretch. Healthy dura sits at lamdaE = (1.098, 1.035, 0.880), so the
    // axial and through-thickness stretches both fall outside [0.95, 1.05] and
    // plastic growth fires continuously in perfectly healthy tissue, slowly
    // eating the prestretch (theta_e drifts down and H away from 1/2). The band
    // must contain the homeostatic elastic stretches, so it is now supplied by
    // the driver via local_parameters[18]/[19] instead of being hard-coded.
    double lowlim = (local_parameters.size() > 18) ? local_parameters[18] : 0.85;
    double uplim  = (local_parameters.size() > 19) ? local_parameters[19] : 1.15;

    //std::ofstream myfile;
    //myfile.open("FE_results.csv");

     // skip remodeling for chest wall
    // if(X(2)>0){ // I need to modify this for my case.

    for(int step=0;step<time_step_ratio;step++){
        // Save results
        //myfile << std::fixed << std::setprecision(10) << local_dt*step << "," << phif << "," << kappa << "," << a0(0) << "," << a0(1) << "," << lamdaP(0) << "," << lamdaP(1) << "\n";
        //std::cout << "\n a0.dot(s0) " << a0.dot(s0) << "\n  a0.dot(n0)" << a0.dot(n0) << "\n s0.dot(n0)" << s0.dot(n0) << "\n";

        // fiber tensor in the reference
        a0a0 = a0*a0.transpose();
        s0s0 = s0*s0.transpose();
        n0n0 = n0*n0.transpose();
        // recompute split
        FFg = lamdaP(0)*(a0a0) + lamdaP(1)*(s0s0) + lamdaP(2)*n0n0;
        FFginv = (1./lamdaP(0))*(a0a0) + (1./lamdaP(1))*(s0s0) + (1./lamdaP(2))*(n0n0);
        //Matrix3d FFe = FF*FFginv;
        // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
        // elastic strain
        CCe = FFginv*CC*FFginv;
        CCeinv = CCe.inverse();
        // Jacobian of the deformations
        double Jp = lamdaP(0)*lamdaP(1)*lamdaP(2);
        double Je = sqrt(CCe.determinant());
        double J = Je*Jp;

        // Eigenvalues
        // Use .compute() for the QR algorithm, .computeDirect() for an explicit trig
        // QR may be more accurate, but explicit is faster
        // Eigenvalues
        SelfAdjointEigenSolver<Matrix3d> eigensolver;
        eigensolver.compute(CCe);
        Vector3d lamda = eigensolver.eigenvalues();
        Matrix3d vectors = eigensolver.eigenvectors();
        double lamdamax = lamda(2);
        double lamdamed = lamda(1);
        double lamdamin = lamda(0);
        Vector3d vectormax = vectors.col(2);
        Vector3d vectormed = vectors.col(1);
        Vector3d vectormin = vectors.col(0);
        if (a0.dot(vectormax) < 0) {
            vectormax = -vectormax;
        }
        // If CC is the identity matrix, the eigenvectors are arbitrary which is problematic.
        // Beware the matrix becoming singular. Need to perturb.
        double epsilon = 1e-7;
        double delta = 1e-7;
        if(abs(lamdamin-lamdamed) < epsilon || abs(lamdamin-lamdamax) < epsilon || abs(lamdamed-lamdamax) < epsilon){
            lamdamax = lamdamax*(1+delta);
            lamdamin = lamdamin*(1-delta);
            lamdamed = lamdamed/((1+delta)*(1-delta));
        }
        //std::cout << "\n vectormax" << vectormax << "\n lamdaMax" << lamdamax << "\n";

        // Mechanosensing
        // Mechanosensing stimulus: in-plane AREAL elastic stretch of the dural
        // mid-surface, theta_e = ||cof(F^e).n0||, NOT det(F^e). The tissue is
        // treated as incompressible so det(F^e)==1 and could never respond to
        // membrane stretch. See include/mechanosensing.h.
        double theta_e = evalThetaE(J, CCinv, n0, lamdaP(0), lamdaP(1));
        double He = evalHe(theta_e, vartheta_e, gamma_theta);
        //if(He<0.002){He=0;}

        //----------------------------//
        // 2D FORWARD-EULER EQUATIONS
        //----------------------------//
        // Collagen density PHI
        double phif_dot_plus = (p_phi + (p_phi_c*c)/(K_phi_c+c) + p_phi_theta*He)*(rho/(K_phi_rho+phif));
        //std::cout<<"phidotplus: "<<phif_dot_plus<<"\n";
        phif_dot = phif_dot_plus - (d_phi + c*rho*d_phi_rho_c)*phif;

        // Principal direction A0
        // Alternatively, see Menzel (NOTE THAT THE THIRD COMPONENT IS THE LARGEST ONE)
        // a0 = a0 + local_dt*(((2.*PIE*phif_dot_plus)/(tau_omega))*lamda(2)*(Identity-a0a0)*(vectors.col(2)));
        a0_dot = ((2.*PIE*phif_dot_plus)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*vectormax;

        // Dispersion KAPPA
        // kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamda(2)/lamda(1),gamma_kappa)/2. - kappa);
        kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamdamed/lamdamax,gamma_kappa)/3. - kappa);

        // elastic stretches of the directions a and s
        double Ce_aa = a0.transpose()*CCe*a0;
        double Ce_ss = s0.transpose()*CCe*s0;
        double Ce_nn = n0.transpose()*CCe*n0;
        lamdaE_a = sqrt(Ce_aa);
        lamdaE_s = sqrt(Ce_ss);
        lamdaE_n = sqrt(Ce_nn);

	

//        // Permanent deformation LAMDAP
//        // No threshold
//        lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-1)/tau_lamdaP_a;
//        lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-1)/tau_lamdaP_s;
//        lamdaP_dot(2) = phif_dot_plus*(lamdaE_n-1)/tau_lamdaP_n;
//
//        // Threshold
//        // lamdaP_a
//        if((lamdaE_a < 0.995 || lamdaE_a > 1.005)){ // && (lamdaE_a < 1.05 && lamdaE_a > 0.95)
//            lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-1)/tau_lamdaP_a;
//        } else{
//            lamdaP_dot(0) = 0;
//        }
//        // lamdaP_s
//        if((lamdaE_s < 0.995 || lamdaE_s > 1.005)){ //  && (lamdaE_s < 1.05 && lamdaE_s > 0.95)
//            lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-1)/tau_lamdaP_s;
//        } else{
//            lamdaP_dot(1) = 0;
//        }
//        // lamdaP_n
//        if((lamdaE_n < 0.995 || lamdaE_n > 1.005)){ //  && (lamdaE_n < 1.05 && lamdaE_n > 0.95)
//            lamdaP_dot(2) = phif_dot_plus*(lamdaE_n-1)/tau_lamdaP_n;
//        } else{
//            lamdaP_dot(2) = 0;
//        }

        // Threshold
        // lamdaP_a
        if(lamdaE_a < lowlim){
            lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-lowlim)/tau_lamdaP_a;
        } else if(lamdaE_a > uplim){
            lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-uplim)/tau_lamdaP_a;
        }
        else{
            lamdaP_dot(0) = 0;
        }
        // lamdaP_s
        if(lamdaE_s < lowlim){
            lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-lowlim)/tau_lamdaP_s;
        } else if(lamdaE_s > uplim){
            lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-uplim)/tau_lamdaP_s;
        }else{
            lamdaP_dot(1) = 0;
        }
        // lamdaP_n
        if(lamdaE_n < lowlim){
            lamdaP_dot(2) = phif_dot_plus*(lamdaE_n-lowlim)/tau_lamdaP_n;
        } else if(lamdaE_n > uplim){
            lamdaP_dot(2) = phif_dot_plus*(lamdaE_n-uplim)/tau_lamdaP_n;
        }else{
            lamdaP_dot(2) = 0;
        }
            

        //----------------------------------------//
        // CALCULATE GLOBAL CHAIN RULE DERIVATIVES
        //----------------------------------------//

        // Calculate derivatives of eigenvalues and eigenvectors
        Matrix3d dCCdCC; dCCdCC.setZero();
        Matrix4d LHS; LHS.setZero();
        Vector4d RHS,SOL; RHS.setZero(), SOL.setZero();
        //std::vector<Vector3d> dvectordCCe(9,Vector3d::Zero());
        //std::vector<Vector3d> dvectordCC(9,Vector3d::Zero());
        std::vector<Matrix3d> dvectormaxdCCe(3,Matrix3d::Zero());
        std::vector<Matrix3d> dvectormaxdCC(3,Matrix3d::Zero());

        // We actually only  need one eigenvector so an outer loop is not needed, but if we want more just change to 3.
        // Create matrix for calculation of derivatives of eigenvalues and vectors.
        LHS << CCe(0,0) - lamdamax, CCe(0,1), CCe(0,2), -vectormax(0),
                CCe(1,0), CCe(1,1) - lamdamax, CCe(1,2), -vectormax(1),
                CCe(2,0), CCe(2,1), CCe(2,2) - lamdamax, -vectormax(2),
                vectormax(0), vectormax(1), vectormax(2), 0;
        // CC is symmetric so we actually only need 6 components.
        //std::cout<<"\n"<<MM<<"\n"<<MM.determinant()<<"\n";
        for (int ii=0; ii<3; ii++){
            for (int jj=0; jj<3; jj++) {
                // Create vector for right hand side. It is the product of an elementary matrix with the eigenvector.
                RHS.setZero();
                RHS(ii) = -vectormax(jj);
                // Solve
                SOL = LHS.lu().solve(RHS);
                dvectormaxdCCe[0](ii,jj) = SOL(0); // II counts the Voigt components of CCe, index has the eigenvector components
                dvectormaxdCCe[1](ii,jj) = SOL(1);
                dvectormaxdCCe[2](ii,jj) = SOL(2);
                //dlamdamaxdCCe[II] = SOL(3);
            }
        }

        for (int ii=0; ii<3; ii++){
            for (int jj=0; jj<3; jj++) {
                for (int kk=0; kk<3; kk++){
                    for (int ll=0; ll<3; ll++) {
                        for (int mm=0; mm<3; mm++) {
                            dvectormaxdCC[mm](kk,ll) = dvectormaxdCCe[mm](ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                        }
                    }
                }
            }
        }

        // Alternatively for the eigenvalue we can use the rule from Holzapfel
        // But we still need dCCedCC for the chain rule
        Matrix3d dlamdamaxdCCe = vectormax*vectormax.transpose();
        Matrix3d dlamdameddCCe = vectormed*vectormed.transpose();
        Matrix3d dlamdamindCCe = vectormin*vectormin.transpose();

        // Multiply by dCCdCCe to get dlamdadCC
        Matrix3d dlamdamaxdCC; Matrix3d dlamdameddCC; Matrix3d dlamdamindCC;
        dlamdamaxdCC.setZero(); dlamdameddCC.setZero(); dlamdamindCC.setZero();
        for (int ii=0; ii<3; ii++){
            for (int jj=0; jj<3; jj++) {
                for (int kk=0; kk<3; kk++){
                    for (int ll=0; ll<3; ll++) {
                        dlamdamaxdCC(kk,ll) = dlamdamaxdCCe(ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                        dlamdameddCC(kk,ll) = dlamdameddCCe(ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                        dlamdamindCC(kk,ll) = dlamdamindCCe(ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                    }
                }
            }
        }

        // Calculate derivative of lamdaE wrt CC. This will involve an elementary matrix.
        Matrix3d dlamdaE_a_dCC; dlamdaE_a_dCC.setZero();
        Matrix3d dlamdaE_s_dCC; dlamdaE_s_dCC.setZero();
        Matrix3d dlamdaE_n_dCC; dlamdaE_n_dCC.setZero();
        // Matrix multiplication is associative, so d(a0*FFginv)*CC*(FFginv*a0)/dCC
        // is the outer product of the two vectors we get from a0*FFginv
        // and the symmetry makes the calculation easier
        for (int ii=0; ii<3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                for (int kk = 0; kk < 3; kk++) {
                    for (int ll = 0; ll < 3; ll++) {
                        dlamdaE_a_dCC(jj,kk) = (a0(ii) * FFginv(ii,jj)) * (FFginv(kk,ll) * a0(ll));
                        dlamdaE_s_dCC(jj,kk) = (s0(ii) * FFginv(ii,jj)) * (FFginv(kk,ll) * s0(ll));
                        dlamdaE_n_dCC(jj,kk) = (n0(ii) * FFginv(ii,jj)) * (FFginv(kk,ll) * n0(ll));
                    }
                }
            }
        }
        // Calculate derivative of He wrt to CC. If this is the same H, this is the same as in the main code.
        Matrix3d dHedCC_explicit, dphifdotplusdCC; dHedCC_explicit.setZero(); dphifdotplusdCC.setZero();
        phif_dot_plus = (p_phi + (p_phi_c*c)/(K_phi_c+c) + p_phi_theta*He)*(rho/(K_phi_rho+phif));
        // dH/dCC = gamma_e H (1-H) dtheta_e/dCC  (was dJe/dCC = J CCinv/(2 Jp))
        dHedCC_explicit = evalDHedCC(theta_e, He, gamma_theta, CCinv, n0);
        dphifdotplusdCC = p_phi_theta*dHedCC_explicit*(rho/(K_phi_rho+phif));
        //std::cout<<"RHO " << rho << " p_phi_theta " << p_phi_theta << " dHedCC_explicit " << dHedCC_explicit << " CCinv " << CCinv;

        //----------//
        // X
        //----------//
        // Explicit derivatives of the local variables with respect to CC (phi, a0, kappa, lamdap)
        // Really, there are 8*9 = 72 components.
        // Then, remember that CC is symmetric so we actually only need 8*6 = 48 unique values.
        std::vector<Matrix3d> da0dCC(3,Matrix3d::Zero());
        for (int ii=0; ii<3; ii++){
            for (int jj=0; jj<3; jj++) {
                for (int kk=0; kk<3; kk++){
                    for (int ll=0; ll<3; ll++) {
                        da0dCC[kk](ii,jj) += (2.*PIE/tau_omega)*((dphifdotplusdCC(ii,jj)*lamdamax*(Matrix3d::Identity()(kk,ll)-a0a0(kk,ll))*(vectormax(ll)))
                                                                 + (phif_dot_plus*dlamdamaxdCC(ii,jj)*(Matrix3d::Identity()(kk,ll)-a0a0(kk,ll))*(vectormax(ll)))
                                                                 + (phif_dot_plus*lamdamax*((Matrix3d::Identity()(kk,ll)-a0a0(kk,ll))*dvectormaxdCC[ll](ii,jj))));
                    }
                }
            }
        }


        for (int II=0; II<6; II++){
            int ii = voigt_table_I(II);
            int jj = voigt_table_J(II);
            // phif
            dThetadCC(0+II) += local_dt*dphifdotplusdCC(ii,jj);
            // a0x, a0y, a0z
            dThetadCC(6+II) += local_dt*da0dCC[0](ii,jj);
            dThetadCC(12+II) += local_dt*da0dCC[1](ii,jj);
            dThetadCC(18+II) += local_dt*da0dCC[2](ii,jj);
            // kappa
            dThetadCC(24+II) += (local_dt/(tau_kappa))*((dphifdotplusdCC(ii,jj)*(pow(lamdamed/lamdamax,gamma_kappa)/3. - kappa))
                                                        + ((phif_dot_plus/3.)*(pow(dlamdameddCC(ii,jj)/lamdamax,gamma_kappa) - pow(lamdamed*dlamdamaxdCC(ii,jj)/(lamdamax*lamdamax),gamma_kappa))));
//            // No threshold
//            // lamdaPa, lamdaPs, lamdaPn
//            dThetadCC(30+II) += (local_dt/tau_lamdaP_a)*((dphifdotplusdCC(ii,jj)*(lamdaE_a-1)) + (phif_dot_plus*(dlamdaE_a_dCC(ii,jj))));
//            dThetadCC(36+II) += (local_dt/tau_lamdaP_s)*((dphifdotplusdCC(ii,jj)*(lamdaE_s-1)) + (phif_dot_plus*(dlamdaE_s_dCC(ii,jj))));
//            dThetadCC(42+II) += (local_dt/tau_lamdaP_n)*((dphifdotplusdCC(ii,jj)*(lamdaE_n-1)) + (phif_dot_plus*(dlamdaE_n_dCC(ii,jj))));
//
//            // Threshold
//            // lamdaP_a
//            if((lamdaE_a < lowlim || lamdaE_a > uplim)){ // && (lamdaE_a < uplim && lamdaE_a > lowlim)
//                dThetadCC(30+II) += (local_dt/tau_lamdaP_a)*((dphifdotplusdCC(ii,jj)*(lamdaE_a-1)) + (phif_dot_plus*(dlamdaE_a_dCC(ii,jj))));
//            } else{
//                dThetadCC(30+II) = 0;
//            }
//            // lamdaP_s
//            if((lamdaE_s < 0.995 || lamdaE_s > 1.005)){ //  && (lamdaE_s < 1.05 && lamdaE_s > 0.95)
//                dThetadCC(36+II) += (local_dt/tau_lamdaP_s)*((dphifdotplusdCC(ii,jj)*(lamdaE_s-1)) + (phif_dot_plus*(dlamdaE_s_dCC(ii,jj))));
//            } else{
//                dThetadCC(36+II) = 0;
//            }
//            // lamdaP_n
//            if((lamdaE_n < 0.995 || lamdaE_n > 1.005)){ // && (lamdaE_n < 1.05 && lamdaE_n > 0.95)
//                dThetadCC(42+II) += (local_dt/tau_lamdaP_n)*((dphifdotplusdCC(ii,jj)*(lamdaE_n-1)) + (phif_dot_plus*(dlamdaE_n_dCC(ii,jj))));
//            } else{
//                dThetadCC(42+II) = 0;
//            }

            // Threshold
            // lamdaP_a
            if(lamdaE_a < lowlim){ // && (lamdaE_a < uplim && lamdaE_a > lowlim)
                dThetadCC(30+II) += (local_dt/tau_lamdaP_a)*((dphifdotplusdCC(ii,jj)*(lamdaE_a-lowlim)) + (phif_dot_plus*(dlamdaE_a_dCC(ii,jj))));
            } else if(lamdaE_a > uplim){ // && (lamdaE_a < uplim && lamdaE_a > lowlim)
                dThetadCC(30+II) += (local_dt/tau_lamdaP_a)*((dphifdotplusdCC(ii,jj)*(lamdaE_a-uplim)) + (phif_dot_plus*(dlamdaE_a_dCC(ii,jj))));
            } else{
                dThetadCC(30+II) = 0;
            }
            // lamdaP_s
            if(lamdaE_s < lowlim){ //  && (lamdaE_s < uplim && lamdaE_s > lowlim)
                dThetadCC(36+II) += (local_dt/tau_lamdaP_s)*((dphifdotplusdCC(ii,jj)*(lamdaE_s-lowlim)) + (phif_dot_plus*(dlamdaE_s_dCC(ii,jj))));
            } else if(lamdaE_s > uplim){ //  && (lamdaE_s < uplim && lamdaE_s > lowlim)
                dThetadCC(36+II) += (local_dt/tau_lamdaP_s)*((dphifdotplusdCC(ii,jj)*(lamdaE_s-uplim)) + (phif_dot_plus*(dlamdaE_s_dCC(ii,jj))));
            } else{
                dThetadCC(36+II) = 0;
            }
            // lamdaP_n
            if(lamdaE_n < lowlim){ // && (lamdaE_n < uplim && lamdaE_n > lowlim)
                dThetadCC(42+II) += (local_dt/tau_lamdaP_n)*((dphifdotplusdCC(ii,jj)*(lamdaE_n-lowlim)) + (phif_dot_plus*(dlamdaE_n_dCC(ii,jj))));
            } else if(lamdaE_n > uplim){ // && (lamdaE_n < uplim && lamdaE_n > lowlim)
                dThetadCC(42+II) += (local_dt/tau_lamdaP_n)*((dphifdotplusdCC(ii,jj)*(lamdaE_n-uplim)) + (phif_dot_plus*(dlamdaE_n_dCC(ii,jj))));
            } else{
                dThetadCC(42+II) = 0;
            }
	    //for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
              //  double z_coord = myMesh.nodes[nodei](2);
                //if(z_coord<1e-30){
                  //  dThetadCC(30+II) = 0;
                    //dThetadCC(36+II) = 0;
                    //dThetadCC(42+II) = 0;
                //}
	    //}

        }





        //----------//
        // RHO
        //----------//

        // Explicit derivatives of the local variables with respect to rho
        // Assemble in one vector (phi, a0x, a0y, kappa, lamdap1, lamdap2)
        double dphifdotplusdrho = ((p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*He)*(1/(K_phi_rho+phif)));
        dThetadrho(0) += local_dt*(dphifdotplusdrho - (c*d_phi_rho_c)*phif);
        dThetadrho(1) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*(vectormax))(0)*dphifdotplusdrho;
        dThetadrho(2) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*(vectormax))(1)*dphifdotplusdrho;
        dThetadrho(3) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*(vectormax))(2)*dphifdotplusdrho;
        dThetadrho(4) += local_dt*(1/tau_kappa)*( pow(lamdamed/lamdamax,gamma_kappa)/3. - kappa)*dphifdotplusdrho;
        // No threshold
        dThetadrho(5) += local_dt*((lamdaE_a-1)/tau_lamdaP_a)*dphifdotplusdrho;
        dThetadrho(6) += local_dt*((lamdaE_s-1)/tau_lamdaP_s)*dphifdotplusdrho;
        dThetadrho(7) += local_dt*((lamdaE_n-1)/tau_lamdaP_n)*dphifdotplusdrho;

//        // Threshold
//        // lamdaP_a
//        if((lamdaE_a < 0.995 || lamdaE_a > 1.005)){ // && (lamdaE_a < 1.05 && lamdaE_a > 0.95)
//            dThetadrho(5) += local_dt*((lamdaE_a-1)/tau_lamdaP_a)*dphifdotplusdrho;
//        } else{
//            dThetadrho(5) = 0;
//        }
//        // lamdaP_s
//        if((lamdaE_s < 0.995 || lamdaE_s > 1.005)){ //  && (lamdaE_s < 1.05 && lamdaE_s > 0.95)
//            dThetadrho(6) += local_dt*((lamdaE_s-1)/tau_lamdaP_s)*dphifdotplusdrho;
//        } else{
//            dThetadrho(6) = 0;
//        }
//        // lamdaP_n
//        if((lamdaE_n < 0.995 || lamdaE_n > 1.005)){ // && (lamdaE_n < 1.05 && lamdaE_n > 0.95)
//            dThetadrho(7) += local_dt*((lamdaE_n-1)/tau_lamdaP_n)*dphifdotplusdrho;
//        } else{
//            dThetadrho(7) = 0;
//        }

        // Threshold
        // lamdaP_a
        if(lamdaE_a < lowlim){ // && (lamdaE_a < uplim && lamdaE_a > lowlim)
            dThetadrho(5) += local_dt*((lamdaE_a-lowlim)/tau_lamdaP_a)*dphifdotplusdrho;
        } else if(lamdaE_a > uplim){ // && (lamdaE_a < uplim && lamdaE_a > lowlim)
            dThetadrho(5) += local_dt*((lamdaE_a-uplim)/tau_lamdaP_a)*dphifdotplusdrho;
        }else{
            dThetadrho(5) = 0;
        }
        // lamdaP_s
        if(lamdaE_s < lowlim){ //  && (lamdaE_s < uplim && lamdaE_s > lowlim)
            dThetadrho(6) += local_dt*((lamdaE_s-lowlim)/tau_lamdaP_s)*dphifdotplusdrho;
        } else if(lamdaE_s > uplim){ //  && (lamdaE_s < uplim && lamdaE_s > lowlim)
            dThetadrho(6) += local_dt*((lamdaE_s-uplim)/tau_lamdaP_s)*dphifdotplusdrho;
        } else{
            dThetadrho(6) = 0;
        }
        // lamdaP_n
        if(lamdaE_n < lowlim){ // && (lamdaE_n < uplim && lamdaE_n > lowlim)
            dThetadrho(7) += local_dt*((lamdaE_n-lowlim)/tau_lamdaP_n)*dphifdotplusdrho;
        } else if(lamdaE_n > uplim){ // && (lamdaE_n < uplim && lamdaE_n > lowlim)
            dThetadrho(7) += local_dt*((lamdaE_n-uplim)/tau_lamdaP_n)*dphifdotplusdrho;
        } else{
            dThetadrho(7) = 0;
        }


	//for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
          //  double z_coord = myMesh.nodes[nodei](2);
            //if(z_coord<1e-30){
              //  dThetadrho(5) = 0;
                //dThetadrho(6) = 0;
                //dThetadrho(7) = 0;
            //}
	//}



        //----------//
        // c
        //----------//

        // Explicit derivatives of the local variables with respect to c
        // Assemble in one vector (phi, a0x, a0y, kappa, lamdap1, lamdap2)
        //double dphifdotplusdc = ((p_phi_c*K_phi_c)/(pow(K_phi_c+c,2)))*(rho/(K_phi_rho+phif_0));
        double dphifdotplusdc = (rho/(K_phi_rho+phif))*((p_phi_c)/(K_phi_c+c) - (p_phi_c*c)/((K_phi_c+c)*(K_phi_c+c)));
        dThetadc(0) += local_dt*(dphifdotplusdc - (rho*d_phi_rho_c)*phif);
        dThetadc(1) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*(vectormax))(0)*dphifdotplusdc;
        dThetadc(2) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*(vectormax))(1)*dphifdotplusdc;
        dThetadc(3) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*(vectormax))(2)*dphifdotplusdc;
        dThetadc(4) += local_dt*(1/tau_kappa)*( pow(lamdamed/lamdamax,gamma_kappa)/3. - kappa)*dphifdotplusdc;
        // No threshold
        dThetadc(5) += local_dt*((lamdaE_a-1)/tau_lamdaP_a)*dphifdotplusdc;
        dThetadc(6) += local_dt*((lamdaE_s-1)/tau_lamdaP_s)*dphifdotplusdc;
        dThetadc(7) += local_dt*((lamdaE_n-1)/tau_lamdaP_n)*dphifdotplusdc;

//        // Threshold
//        // lamdaP_a
//        if((lamdaE_a < 0.995 || lamdaE_a > 1.005)){ // && (lamdaE_a < 1.05 && lamdaE_a > 0.95)
//            dThetadc(5) += local_dt*((lamdaE_a-1)/tau_lamdaP_a)*dphifdotplusdc;
//        } else{
//            dThetadc(5) = 0;
//        }
//        // lamdaP_s
//        if((lamdaE_s < 0.995 || lamdaE_s > 1.005)){ // && (lamdaE_s < 1.05 && lamdaE_s > 0.95)
//            dThetadc(6) += local_dt*((lamdaE_s-1)/tau_lamdaP_s)*dphifdotplusdc;
//        } else{
//            dThetadc(6) = 0;
//        }
//        // lamdaP_n
//        if((lamdaE_n < 0.995 || lamdaE_n > 1.005)){ // && (lamdaE_n < 1.05 && lamdaE_n > 0.95)
//            dThetadc(7) += local_dt*((lamdaE_n-1)/tau_lamdaP_n)*dphifdotplusdc;
//        } else{
//            dThetadc(7) = 0;
//        }

        // Threshold
        // lamdaP_a
        if(lamdaE_a < lowlim){ // && (lamdaE_a < uplim && lamdaE_a > lowlim)
            dThetadc(5) += local_dt*((lamdaE_a-lowlim)/tau_lamdaP_a)*dphifdotplusdc;
        } else if(lamdaE_a > uplim){ // && (lamdaE_a < uplim && lamdaE_a > lowlim)
            dThetadc(5) += local_dt*((lamdaE_a-uplim)/tau_lamdaP_a)*dphifdotplusdc;
        } else{
            dThetadc(5) = 0;
        }
        // lamdaP_s
        if(lamdaE_s < lowlim){ // && (lamdaE_s < uplim && lamdaE_s > lowlim)
            dThetadc(6) += local_dt*((lamdaE_s-lowlim)/tau_lamdaP_s)*dphifdotplusdc;
        } else if(lamdaE_s > uplim){ // && (lamdaE_s < uplim && lamdaE_s > lowlim)
            dThetadc(6) += local_dt*((lamdaE_s-uplim)/tau_lamdaP_s)*dphifdotplusdc;
        } else{
            dThetadc(6) = 0;
        }
        // lamdaP_n
        if(lamdaE_n < lowlim){ // && (lamdaE_n < uplim && lamdaE_n > lowlim)
            dThetadc(7) += local_dt*((lamdaE_n-lowlim)/tau_lamdaP_n)*dphifdotplusdc;
        } else if(lamdaE_n > uplim){ // && (lamdaE_n < uplim && lamdaE_n > lowlim)
            dThetadc(7) += local_dt*((lamdaE_n-uplim)/tau_lamdaP_n)*dphifdotplusdc;
        } else{
            dThetadc(7) = 0;
        }


	/*for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
            double z_coord = myMesh.nodes[nodei](2);
            if(z_coord<1e-30){
                dThetadc(5) = 0;
                dThetadc(6) = 0;
                dThetadc(7) = 0;
            }
	}*/



        //---------------------//
        // COMPARE WITH NUMERICAL
        //---------------------//
        /*
        // Calculate numerical derivatives
        //std::cout << "Last iteration" << "\n";
        double phif_dot_plus_num; double phif_dot_minus;
        double kappa_dot_plus; double kappa_dot_minus;
        Vector3d a0_dot_plus; Vector3d a0_dot_minus;
        Vector3d lamdaP_dot_plus; Vector3d lamdaP_dot_minus;
        epsilon = 1e-7;
        double rho_plus = rho + epsilon;
        double rho_minus = rho - epsilon;
        double c_plus = c + epsilon;
        double c_minus = c - epsilon;
        Matrix3d CC_plus, CC_minus;

        // Call update function with plus and minus to get numerical derivatives
        evalForwardEulerUpdate(local_dt, local_parameters, c, rho_plus, FF, CC, phif, a0, s0, kappa, lamdaP, phif_dot_plus_num, a0_dot_plus, kappa_dot_plus, lamdaP_dot_plus);
        evalForwardEulerUpdate(local_dt, local_parameters, c, rho_minus, FF, CC,phif, a0, s0, kappa, lamdaP, phif_dot_minus, a0_dot_minus, kappa_dot_minus, lamdaP_dot_minus);
        dThetadrho_num(0) += local_dt*(1./(2.*epsilon))*(phif_dot_plus_num-phif_dot_minus);
        dThetadrho_num(1) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(0)-a0_dot_minus(0));
        dThetadrho_num(2) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(1)-a0_dot_minus(1));
        dThetadrho_num(3) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(2)-a0_dot_minus(2));
        dThetadrho_num(4) += local_dt*(1./(2.*epsilon))*(kappa_dot_plus-kappa_dot_minus);
        dThetadrho_num(5) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(0)-lamdaP_dot_minus(0));
        dThetadrho_num(6) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(1)-lamdaP_dot_minus(1));
        dThetadrho_num(7) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(2)-lamdaP_dot_minus(2));

        evalForwardEulerUpdate(local_dt, local_parameters, c_plus, rho, FF, CC, phif, a0, s0, kappa, lamdaP, phif_dot_plus_num, a0_dot_plus, kappa_dot_plus, lamdaP_dot_plus);
        evalForwardEulerUpdate(local_dt, local_parameters, c_minus, rho, FF, CC, phif, a0, s0, kappa, lamdaP, phif_dot_minus, a0_dot_minus, kappa_dot_minus, lamdaP_dot_minus);
        dThetadc_num(0) += local_dt*(1./(2.*epsilon))*(phif_dot_plus_num-phif_dot_minus);
        dThetadc_num(1) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(0)-a0_dot_minus(0));
        dThetadc_num(2) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(1)-a0_dot_minus(1));
        dThetadc_num(3) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(2)-a0_dot_minus(2));
        dThetadc_num(4) += local_dt*(1./(2.*epsilon))*(kappa_dot_plus-kappa_dot_minus);
        dThetadc_num(5) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(0)-lamdaP_dot_minus(0));
        dThetadc_num(6) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(1)-lamdaP_dot_minus(1));
        dThetadc_num(7) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(2)-lamdaP_dot_minus(2));

        for (int II=0; II<6; II++){
            int ii = voigt_table_I(II);
            int jj = voigt_table_J(II);

            CC_plus = CC;
            CC_minus = CC;
            CC_plus(ii,jj) += epsilon;
            CC_minus(ii,jj) -= epsilon;
            evalForwardEulerUpdate(local_dt, local_parameters, c, rho, FF, CC_plus, phif, a0, s0, kappa, lamdaP, phif_dot_plus_num, a0_dot_plus, kappa_dot_plus, lamdaP_dot_plus);
            evalForwardEulerUpdate(local_dt, local_parameters, c, rho, FF, CC_minus, phif, a0, s0, kappa, lamdaP, phif_dot_minus, a0_dot_minus, kappa_dot_minus, lamdaP_dot_minus);

            // phif
            dThetadCC_num(0+II) += local_dt*(1./(2.*epsilon))*(phif_dot_plus_num-phif_dot_minus);
            // a0x, a0y, a0z
            dThetadCC_num(6+II) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(0)-a0_dot_minus(0));
            dThetadCC_num(12+II) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(1)-a0_dot_minus(1));
            dThetadCC_num(18+II) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(2)-a0_dot_minus(2));
            // kappa
            dThetadCC_num(24+II) += local_dt*(1./(2.*epsilon))*(kappa_dot_plus-kappa_dot_minus);
            // lamdaPa, lamdaPs, lamdaPn
            dThetadCC_num(30+II) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(0)-lamdaP_dot_minus(0));
            dThetadCC_num(36+II) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(1)-lamdaP_dot_minus(1));
            dThetadCC_num(42+II) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(2)-lamdaP_dot_minus(2));
        }*/

        //---------------------//
        // UPDATE VARIABLES
        //---------------------//
        // Collagen density PHI
        phif = phif + local_dt*(phif_dot);

        // Principal direction A0
        Vector3d a0_00 = a0;
        a0 = a0 + local_dt*a0_dot;
        // normalize a0
        a0 = a0/sqrt(a0.dot(a0));
        //std::cout << "\n a0_00: " << a0_00 << "\n a0: " << a0 << "\n";

        // Construct a rotation using Rodriguez formula to get the new s0 and n0
        Matrix3d Rot = Matrix3d::Identity();
        if(a0 != a0_00){
            Vector3d across = a0_00.cross(a0);
            double sinacross = sqrt(across.dot(across));
            double cosacross = a0_00.dot(a0);
            Vector3d acrossunit = across/sqrt(across.dot(across));
            Matrix3d askew; askew << 0, -across(2), across(1), across(2), 0, -across(0), -across(1), across(0), 0;

            // All of the below should be equivalent forms of Rodrigues formula
            Rot = Matrix3d::Identity() + askew + (askew*askew)*(1/(1+cosacross));
            //Rot = Matrix3d::Identity() + sinacross*askew + (1-cosacross)*(askew*askew);
            //Rot = (1/(sqrt(a0.dot(a0))*sqrt(a0_0.dot(a0_0))))*((a0.dot(a0_0))*Identity + askew + ((sqrt(a0.dot(a0))*sqrt(a0_0.dot(a0_0))-(a0.dot(a0_0)))/(sqrt(across.dot(across))))*across * across.transpose());
            //Rot = Matrix3d::Identity() + askew + (askew*askew)*(1-cosacross)/(sinacross*sinacross);
            //std::cout << "\n" << Rot*Rot.transpose() << "\n";
        }

        // These should retain their normality, but we can renormalize
        s0 = Rot*s0;
        s0 = s0/sqrt(s0.dot(s0));
        n0 = Rot*n0;
        n0 = n0/sqrt(n0.dot(n0));

        // These should retain their orthogonality, but we could refacor with modified Gram-Schmidt if they become non-orthogonal
//        n = size(V,1);
//        k = size(V,2);
//        U = zeros(n,k);
//        U(:,1) = V(:,1)/sqrt(V(:,1)'*V(:,1));
//        for i = 2:k
//        U(:,i) = V(:,i);
//        for j = 1:i-1
//        U(:,i) = U(:,i) - ( U(:,j)'*U(:,i) )/( U(:,j)'*U(:,j) )*U(:,j);
//        end
//        U(:,i) = U(:,i)/sqrt(U(:,i)'*U(:,i));
//        end

        // Dispersion KAPPA
        kappa = kappa + local_dt*(kappa_dot);

        // Permanent deformation LAMDAP
        lamdaP = lamdaP + local_dt*(lamdaP_dot);

        //std::cout << "\nphif: " << phif << ", kappa: " << kappa << ", lamdaP:" << lamdaP(0) << "," << lamdaP(1) << "," << lamdaP(2)
        //          << ",a0:" << a0(0) << "," << a0(1) << "," << a0(2) << ",s0:" << s0(0) << "," << s0(1) << "," << s0(2) << ",n0:" << n0(0) << "," << n0(1) << "," << n0(2) << "\n";
    }
// }//closing the if statement for the chest wall
    //myfile.close();

    //std::cout << "lamdaE " << lamdaE_a << " " << lamdaE_s << " " << lamdaE_n;

    //std::cout<<"lamda1: "<<lamda2d(1)<<", lamda0: "<<lamda2d(0)<<", lamdaP:"<<lamdaP(0)<<","<<lamdaP(1)<<",a0:"<<a0(0)<<","<<a0(1)<<"Ce_aa: "<<Ce_aa<<","<<Ce_ss<<"\n";
    //std::cout<<"\n CC \n"<<CCproj;
    //std::cout << "\n" << dThetadCC_num << "\n";
    //std::cout << "\n" << dThetadrho_num << "\n";
    //std::cout << "\n" << dThetadc_num << "\n";
}

void evalForwardEulerUpdate(double local_dt, const std::vector<double> &local_parameters, double c,double rho,const Matrix3d &FF, const Matrix3d &CC,
                            const double &phif, const Vector3d &a0, const Vector3d &s0, const double &kappa, const Vector3d &lamdaP,
                            double &phif_dot, Vector3d &a0_dot, double &kappa_dot, Vector3d &lamdaP_dot)
{
    //---------------------------------//
    // Parameters
    //
    // collagen fraction
    double p_phi = local_parameters[0]; // production by fibroblasts, natural rate
    double p_phi_c = local_parameters[1]; // production up-regulation, weighted by C and rho
    double p_phi_theta = local_parameters[2]; //production regulated by stretch
    double K_phi_c = local_parameters[3]; // saturation of C effect on deposition
    double K_phi_rho = local_parameters[4]; // saturation of collagen fraction itself
    double d_phi = local_parameters[5]; // rate of degradation
    double d_phi_rho_c = local_parameters[6]; // rate of degradation
    //
    // fiber alignment
    double tau_omega = local_parameters[7]; // time constant for angular reorientation
    //
    // dispersion parameter
    double tau_kappa = local_parameters[8]; // time constant
    double gamma_kappa = local_parameters[9]; // exponent of the principal stretch ratio
    //
    // permanent contracture/growth
    double tau_lamdaP_a = local_parameters[10]; // time constant for direction a
    double tau_lamdaP_s = local_parameters[11]; // time constant for direction s
    double tau_lamdaP_n = local_parameters[12]; // time constant for direction s
    //
    double vartheta_e = local_parameters[13]; // mechanosensing response
    double gamma_theta = local_parameters[14]; // exponent of the Heaviside function
    //

    double PIE = 3.14159;

    Vector3d n0 = s0.cross(a0);
    if(n0(2)<0){
        n0 = a0.cross(s0);
    }
    //std::cout<<"a0"<<"\n"<<a0<<"\n"<<"s0"<<"\n"<<s0<<"\n"<<"n0"<<"\n"<<n0<<'\n';
    // fiber tensor in the reference
    Matrix3d a0a0 = a0*a0.transpose();
    Matrix3d s0s0 = s0*s0.transpose();
    Matrix3d n0n0 = n0*n0.transpose();
    // recompute split
    Matrix3d FFg = lamdaP(0)*(a0a0) + lamdaP(1)*(s0s0) + 1.*n0n0;
    Matrix3d FFginv = (1./lamdaP(0))*(a0a0) + (1./lamdaP(1))*(s0s0) + 1.*n0n0;
    //Matrix3d FFe = FF*FFginv;
    // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
    // elastic strain
    Matrix3d CCe = FFginv*CC*FFginv;
    Matrix3d CCeinv = CCe.inverse();
    // Jacobian of the deformations
    double Jp = lamdaP(0)*lamdaP(1)*1.;
    double Je = sqrt(CCe.determinant());
    double J = Je*Jp;

    // Eigenvalues
    // Use .compute() for the QR algorithm, .computeDirect() for an explicit trig
    // QR may be more accurate, but explicit is faster
    // Eigenvalues
    SelfAdjointEigenSolver<Matrix3d> eigensolver;
    eigensolver.compute(CCe);
    Vector3d lamda = eigensolver.eigenvalues();
    Matrix3d vectors = eigensolver.eigenvectors();
    double lamdamax = lamda(2);
    double lamdamed = lamda(1);
    double lamdamin = lamda(0);
    Vector3d vectormax = vectors.col(2);
    Vector3d vectormed = vectors.col(1);
    Vector3d vectormin = vectors.col(0);
    if (a0.dot(vectormax) < 0) {
        vectormax = -vectormax;
    }

    double He = 1./(1.+exp(-gamma_theta*(Je - vartheta_e)));
    //if(He<0.002){He=0;}

    //----------------//
    // 2D FORWARD-EULER EQUATIONS
    //----------------//
    // Collagen density PHI
    double phif_dot_plus = (p_phi + (p_phi_c*c)/(K_phi_c+c) + p_phi_theta*He)*(rho/(K_phi_rho+phif));
    //std::cout<<"phidotplus: "<<phif_dot_plus<<"\n";
    phif_dot = phif_dot_plus - (d_phi + c*rho*d_phi_rho_c)*phif;

    // Principal direction A0
    // Alternatively, see Menzel (NOTE THAT THE THIRD COMPONENT IS THE LARGEST ONE)
    a0_dot = ((2.*PIE*phif_dot_plus)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*vectormax;

    // Dispersion KAPPA
    // kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamda(2)/lamda(1),gamma_kappa)/2. - kappa);
    kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamdamed/lamdamax,gamma_kappa)/3. - kappa);

    // elastic stretches of the directions a and s
    double Ce_aa = a0.transpose()*CCe*a0;
    double Ce_ss = s0.transpose()*CCe*s0;
    double Ce_nn = n0.transpose()*CCe*n0;
    double lamdaE_a = sqrt(Ce_aa);
    double lamdaE_s = sqrt(Ce_ss);
    double lamdaE_n = sqrt(Ce_nn);

    // Permanent deformation LAMDAP
    lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-1)/tau_lamdaP_a;
    lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-1)/tau_lamdaP_s;
    lamdaP_dot(2) = phif_dot_plus*(lamdaE_n-1)/tau_lamdaP_n;
}

//========================================================//
// EXPLICIT LOCAL PROBLEM: structural update
//========================================================//
void localWoundProblemExplicit2d(
        double dt, const std::vector<double> &local_parameters,double c,double rho,const Matrix3d &FF,
        const double &phif_0, const Vector3d &a0_0, const Vector3d &s0_0, const double &kappa_0, const Vector3d &lamdaP_0,
        double &phif, Vector3d &a0, Vector3d &s0, double &kappa, Vector3d &lamdaP,
        VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadc)
{
    //---------------------------------//
    //
    // INPUT
    // 	matParam: material parameters
    //	rho: value of the cell density at the point
    //	c: concentration at the point
    //	CC: deformation at the point
    //	Theta_t: value of the parameters at previous time step
    //
    // OUTPUT
    //	Theta: value of the parameters at the current time step
    //	dThetadCC: derivative of Theta wrt global mechanics (CC)
    // 	dThetadrho: derivative of Theta wrt global rho
    // 	dThetadc: derivative of Theta wrt global C
    //
    //---------------------------------//

    //---------------------------------//
    // Parameters
    //
    // collagen fraction
    double p_phi = local_parameters[0]; // production by fibroblasts, natural rate
    double p_phi_c = local_parameters[1]; // production up-regulation, weighted by C and rho
    double p_phi_theta = local_parameters[2]; //production regulated by stretch
    double K_phi_c = local_parameters[3]; // saturation of C effect on deposition
    double K_phi_rho = local_parameters[4]; // saturation of collagen fraction itself
    double d_phi = local_parameters[5]; // rate of degradation
    double d_phi_rho_c = local_parameters[6]; // rate of degradation
    //
    // fiber alignment
    double tau_omega = local_parameters[7]; // time constant for angular reorientation
    //
    // dispersion parameter
    double tau_kappa = local_parameters[8]; // time constant
    double gamma_kappa = local_parameters[9]; // exponent of the principal stretch ratio
    //
    // permanent contracture/growth
    double tau_lamdaP_a = local_parameters[10]; // time constant for direction a
    double tau_lamdaP_s = local_parameters[11]; // time constant for direction s
    double tau_lamdaP_n = local_parameters[12]; // time constant for direction s
    //
    double vartheta_e = local_parameters[13]; // mechanosensing response
    double gamma_theta = local_parameters[14]; // exponent of the Heaviside function
    // solution parameters
    double tol_local = local_parameters[15]; // local tolerance
    double time_step_ratio = local_parameters[16]; // time step ratio
    double max_iter = local_parameters[17]; // max local iter
    //
    // other local stuff
    double local_dt = dt/time_step_ratio;
    //
    double PIE = 3.14159;
    Matrix3d Identity;Identity<<1.,0.,0., 0.,1.,0., 0.,0.,1.;
    Matrix2d Identity2d; Identity2d << 1.,0., 0.,1.;
    Matrix3d Rot90;Rot90 << 0.,-1.,0., 1.,0.,0., 0.,0.,1.;
    Matrix2d Rot902d; Rot902d << 0.,-1., 1.,0.;
    Vector3d n0 = s0.cross(a0);
    if(n0(2)<0){
        n0 = a0.cross(s0);
    }
    //

    //---------------------------------//
    // KINEMATICS
    //---------------------------------//
    Matrix3d CC = FF.transpose()*FF;
    // Update kinematics
    Matrix3d CCinv = CC.inverse();
    // re-compute basis a0, s0, n0. (If not always vertical, could find n0 with cross product or rotations. Be careful of sign.)
    // fiber tensor in the reference
    Matrix3d a0a0, s0s0, n0n0;
    // recompute split
    Matrix3d FFg, FFginv, FFe;
    // elastic strain
    Matrix3d CCe, CCeinv;
    //
    //---------------------------------//

    //---------------------------------//
    // LOCAL EXPLICIT FORWARD-EULER ITERATION
    //---------------------------------//
    //
    // Declare Variables
    // New phi, a0x, a0y, kappa, lamdaPa, lamdaPs
    phif = phif_0;
    // make sure it is unit length
    a0 = a0_0/(sqrt(a0_0.dot(a0_0)));
    kappa = kappa_0;
    lamdaP = lamdaP_0;
    //
    double phif_dot, kappa_dot;
    Vector3d lamdaP_dot, a0_dot, lamdaP_final, a0_final;
    //
    VectorXd dThetadCC_num(24); dThetadCC_num.setZero();
    VectorXd dThetadrho_num(6); dThetadrho_num.setZero();
    VectorXd dThetadc_num(6); dThetadc_num.setZero();

    //std::ofstream myfile;
    //myfile.open("FE_results.csv");
    for(int step=0;step<time_step_ratio;step++){
        // Save results
        //myfile << std::fixed << std::setprecision(10) << local_dt*step << "," << phif << "," << kappa << "," << a0(0) << "," << a0(1) << "," << lamdaP(0) << "," << lamdaP(1) << "\n";

        // fiber tensor in the reference
        a0a0 = a0*a0.transpose();
        s0s0 = s0*s0.transpose();
        n0n0 = n0*n0.transpose();
        // recompute split
        FFg = lamdaP(0)*(a0a0) + lamdaP(1)*(s0s0) + lamdaP(2)*n0n0;
        FFginv = (1./lamdaP(0))*(a0a0) + (1./lamdaP(1))*(s0s0) + (1./lamdaP(2))*(n0n0);
        //Matrix3d FFe = FF*FFginv;
        // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
        // elastic strain
        CCe = FFginv*CC*FFginv;
        CCeinv = CCe.inverse();
        // Jacobian of the deformations
        double Jp = lamdaP(0)*lamdaP(1)*lamdaP(2);
        double Je = sqrt(CCe.determinant());
        double J = Je*Jp;

        //-----------------------------//
        // UPDATE VARIABLES 2D VERSION
        //-----------------------------//
        // elastic strain
        Matrix2d CCeproj; CCeproj << CCe(0,0), CCe(0,1), CCe(1,0), CCe(1,1);
        //std::cout<<"\n"<<CCeproj<<"\n";

        // Eigenvalues
        // Use .compute() for the QR algorithm, .computeDirect() for an explicit trig
        // QR may be more accurate, but explicit is faster
        // Eigenvalues
        SelfAdjointEigenSolver<Matrix2d> eigensolver;
        eigensolver.compute(CCeproj);
        Vector2d lamda = eigensolver.eigenvalues();
        Matrix2d vectors = eigensolver.eigenvectors();
        double lamdamax = lamda(1);
        double lamdamin = lamda(0);
        Vector2d vectormax2d = vectors.col(1);
        Vector2d vectormin2d = vectors.col(0);
        Vector3d vectormax; vectormax << vectormax2d(0), vectormax2d(1), 0.;
        Vector3d vectormin; vectormin << vectormin2d(0), vectormin2d(1), 0.;
        if (a0.dot(vectormax) < 0) {
            vectormax = -vectormax;
        }
        //std::cout << "\n vectormax" << vectormax << "\n lamdaMax" << lamdamax << "\n";

        // Mechanosensing
        double He = 1./(1.+exp(-gamma_theta*(Je - vartheta_e)));
        //if(He<0.002){He=0;}

        //----------------------------//
        // 2D FORWARD-EULER EQUATIONS
        //----------------------------//
        // Collagen density PHI
        double phif_dot_plus = (p_phi + (p_phi_c*c)/(K_phi_c+c) + p_phi_theta*He)*(rho/(K_phi_rho+phif));
        //std::cout<<"phidotplus: "<<phif_dot_plus<<"\n";
        phif_dot = phif_dot_plus - (d_phi + c*rho*d_phi_rho_c)*phif;

        // Principal direction A0
        // Alternatively, see Menzel (NOTE THAT THE THIRD COMPONENT IS THE LARGEST ONE)
        // a0 = a0 + local_dt*(((2.*PIE*phif_dot_plus)/(tau_omega))*lamda(2)*(Identity-a0a0)*(vectors.col(2)));
        a0_dot = ((2.*PIE*phif_dot_plus)/(tau_omega))*lamdamax*(Identity-a0a0)*vectormax;

        // Dispersion KAPPA
        // kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamda(2)/lamda(1),gamma_kappa)/2. - kappa);
        kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamdamin/lamdamax,gamma_kappa)/3. - kappa);

        // elastic stretches of the directions a and s
        double Ce_aa = a0.transpose()*CCe*a0;
        double Ce_ss = s0.transpose()*CCe*s0;
        double lamdaE_a = sqrt(Ce_aa);
        double lamdaE_s = sqrt(Ce_ss);

        // Permanent deformation LAMDAP
        lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-1)/tau_lamdaP_a;
        lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-1)/tau_lamdaP_s;

        //----------------------------------------//
        // CALCULATE GLOBAL CHAIN RULE DERIVATIVES
        //----------------------------------------//
        // For these, we use the initial values for the parameters to appropriately couple with the global derivatives

        // Calculate derivatives of eigenvalues and eigenvectors
        Matrix3d dCCdCC; dCCdCC.setZero();
        Matrix3d LHS; LHS.setZero();
        Vector3d RHS,SOL; RHS.setZero(), SOL.setZero();
        std::vector<Matrix3d> dvectormaxdCCe(3,Matrix3d::Zero());
        std::vector<Matrix3d> dvectormaxdCC(3,Matrix3d::Zero());

        // If CC is the identity matrix, the eigenvectors are arbitrary which is problematic.
        // Beware the matrix becoming singular. Need to perturb.
        double epsilon = 1e-7;
        double delta = 1e-7;
        if(abs(lamdamin-lamdamax) < epsilon){
            lamdamax = lamdamax*(1+delta);
            lamdamin = lamdamin*(1-delta);
        }
        // We actually only  need one eigenvector so an outer loop is not needed, but if we want more just change to 3.
        // Create matrix for calculation of derivatives of eigenvalues and vectors.
        LHS << CCe(0,0) - lamdamax, CCe(0,1), -vectormax(0),
                CCe(1,0), CCe(1,1) - lamdamax, vectormax(1),
                vectormax(0), vectormax(1), 0;
        // CC is symmetric so we actually only need 6 components.
        //std::cout<<"\n"<<MM<<"\n"<<MM.determinant()<<"\n";
        for (int ii=0; ii<2; ii++){
            for (int jj=0; jj<2; jj++) {
                // Create vector for right hand side. It is the product of an elementary matrix with the eigenvector.
                RHS.setZero();
                RHS(ii) = -vectormax(jj);
                // Solve
                SOL = LHS.lu().solve(RHS);
                dvectormaxdCCe[0](ii,jj) = SOL(0); // II counts the Voigt components of CCe, index has the eigenvector components
                dvectormaxdCCe[1](ii,jj) = SOL(1);
                //dlamdamaxdCCe[II] = SOL(2);
            }
        }

        for (int ii=0; ii<2; ii++){
            for (int jj=0; jj<2; jj++) {
                for (int kk=0; kk<2; kk++){
                    for (int ll=0; ll<2; ll++) {
                        for (int mm=0; mm<2; mm++) {
                            dvectormaxdCC[mm](kk,ll) = dvectormaxdCCe[mm](ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                            //std::cout << "\n dvectormaxdCC" << dvectormaxdCC[mm] << "\n";
                        }
                    }
                }
            }
            //std::cout << "\n dvectormaxdCCe" << dvectormaxdCCe[ii] << "\n";
        }

        // Alternatively for the eigenvalue we can use the rule from Holzapfel
        // But we still need dCCedCC for the chain rule
        Matrix3d dlamdamaxdCCe = vectormax*vectormax.transpose();
        Matrix3d dlamdamindCCe = vectormin*vectormin.transpose();

        // Multiply by dCCdCCe to get dlamdadCC
        Matrix3d dlamdamaxdCC; Matrix3d dlamdamindCC;
        dlamdamaxdCC.setZero(); dlamdamindCC.setZero();
        for (int ii=0; ii<3; ii++){
            for (int jj=0; jj<3; jj++) {
                for (int kk=0; kk<3; kk++){
                    for (int ll=0; ll<3; ll++) {
                        dlamdamaxdCC(kk,ll) = dlamdamaxdCCe(ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                        dlamdamindCC(kk,ll) = dlamdamindCCe(ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                    }
                }
            }
        }

        // Calculate derivative of lamdaE wrt CC. This will involve an elementary matrix.
        Matrix3d dlamdaE_a_dCC; dlamdaE_a_dCC.setZero();
        Matrix3d dlamdaE_s_dCC; dlamdaE_s_dCC.setZero();
        // Matrix multiplication is associative, so d(a0*FFginv)*CC*(FFginv*a0)/dCC
        // is the outer product of the two vectors we get from a0*FFginv
        // and the symmetry makes the calculation easier
        for (int ii=0; ii<3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                for (int kk = 0; kk < 3; kk++) {
                    for (int ll = 0; ll < 3; ll++) {
                        dlamdaE_a_dCC(jj,kk) = (a0(ii) * FFginv(ii,jj)) * (FFginv(kk,ll) * a0(ll));
                        dlamdaE_s_dCC(jj,kk) = (s0(ii) * FFginv(ii,jj)) * (FFginv(kk,ll) * s0(ll));
                    }
                }
            }
        }
        // Calculate derivative of He wrt to CC. If this is the same H, this is the same as in the main code.
        Matrix3d dHedCC_explicit, dphifdotplusdCC; dHedCC_explicit.setZero(); dphifdotplusdCC.setZero();
        dHedCC_explicit = (-1./pow((1.+exp(-gamma_theta*(Je - vartheta_e))),2))*(exp(-gamma_theta*(Je - vartheta_e)))*(-gamma_theta)*(J*CCinv/(2*Jp));
        dphifdotplusdCC = p_phi_theta*dHedCC_explicit*(rho/(K_phi_rho+phif));
        //std::cout<<"RHO " << rho << " p_phi_theta " << p_phi_theta << " dHedCC_explicit " << dHedCC_explicit << " dphifdotplusdCC " << dphifdotplusdCC;

        //----------//
        // X
        //----------//
        // Explicit derivatives of the local variables with respect to CC (phi, a0, kappa, lamdap)
        // Really, there are 8*9 = 72 components.
        // Then, remember that CC is symmetric so we actually only need 8*6 = 48 unique values.
        std::vector<Matrix3d> da0dCC(3,Matrix3d::Zero());
        for (int ii=0; ii<3; ii++){
            for (int jj=0; jj<3; jj++) {
                for (int kk=0; kk<3; kk++){
                    for (int ll=0; ll<3; ll++) {
                        da0dCC[kk](ii,jj) += (local_dt*2.*PIE/tau_omega)*((dphifdotplusdCC(ii,jj)*lamdamax*(Matrix3d::Identity()(kk,ll)-a0a0(kk,ll))*(vectormax(ll)))
                                                                          + (phif_dot_plus*dlamdamaxdCC(ii,jj)*(Matrix3d::Identity()(kk,ll)-a0a0(kk,ll))*(vectormax(ll)))
                                                                          + (phif_dot_plus*lamdamax*((Matrix3d::Identity()(kk,ll)-a0a0(kk,ll))*dvectormaxdCC[ll](ii,jj))));
                    }
                }
            }
        }

        int counter = 0;
        for(int ii = 0; ii < 2; ii++){
            for(int jj = 0; jj < 2; jj++){
                dThetadCC(0 + counter) += local_dt*dphifdotplusdCC(ii,jj);
                dThetadCC(4 + counter) += da0dCC[0](ii,jj);
                dThetadCC(8 + counter) += da0dCC[1](ii,jj);
                dThetadCC(12 + counter) += (local_dt/(tau_kappa))*((dphifdotplusdCC(ii,jj)*(pow(lamdamin/lamdamax,gamma_kappa)/3. - kappa))
                                                                   + ((phif_dot_plus/3.)*(pow(dlamdamindCC(ii,jj)/lamdamax,gamma_kappa) - pow(lamdamin*dlamdamaxdCC(ii,jj)/(lamdamax*lamdamax),gamma_kappa))));
                dThetadCC(16 + counter) += (local_dt/tau_lamdaP_a)*((dphifdotplusdCC(ii,jj)*(lamdaE_a-1)) + (phif_dot_plus*(dlamdaE_a_dCC(ii,jj))));
                dThetadCC(20 + counter) += (local_dt/tau_lamdaP_s)*((dphifdotplusdCC(ii,jj)*(lamdaE_s-1)) + (phif_dot_plus*(dlamdaE_s_dCC(ii,jj))));
                counter++;
            }
        }

        //----------//
        // RHO
        //----------//

        // Explicit derivatives of the local variables with respect to rho
        // Assemble in one vector (phi, a0x, a0y, kappa, lamdap1, lamdap2)
        double dphifdotplusdrho = ((p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*He)*(1/(K_phi_rho+phif)));
        dThetadrho(0) += local_dt*(dphifdotplusdrho - (c*d_phi_rho_c)*phif);
        dThetadrho(1) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*(vectormax))(0)*dphifdotplusdrho;
        dThetadrho(2) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*(vectormax))(1)*dphifdotplusdrho;
        dThetadrho(3) += local_dt*(1/tau_kappa)*( pow(lamdamin/lamdamax,gamma_kappa)/3. - kappa)*dphifdotplusdrho;
        dThetadrho(4) += local_dt*((lamdaE_a-1)/tau_lamdaP_a)*dphifdotplusdrho;
        dThetadrho(5) += local_dt*((lamdaE_s-1)/tau_lamdaP_s)*dphifdotplusdrho;

        //----------//
        // c
        //----------//

        // Explicit derivatives of the local variables with respect to c
        // Assemble in one vector (phi, a0x, a0y, kappa, lamdap1, lamdap2)
        //double dphifdotplusdc = ((p_phi_c*K_phi_c)/(pow(K_phi_c+c,2)))*(rho/(K_phi_rho+phif_0));
        double dphifdotplusdc = (rho/(K_phi_rho+phif))*((p_phi_c)/(K_phi_c+c) - (p_phi_c*c)/((K_phi_c+c)*(K_phi_c+c)));
        dThetadc(0) += local_dt*(dphifdotplusdc - (rho*d_phi_rho_c)*phif);
        dThetadc(1) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*(vectormax))(0)*dphifdotplusdc;
        dThetadc(2) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix3d::Identity()-a0a0)*(vectormax))(1)*dphifdotplusdc;
        dThetadc(3) += local_dt*(1/tau_kappa)*( pow(lamdamin/lamdamax,gamma_kappa)/3. - kappa)*dphifdotplusdc;
        dThetadc(4) += local_dt*((lamdaE_a-1)/tau_lamdaP_a)*dphifdotplusdc;
        dThetadc(5) += local_dt*((lamdaE_s-1)/tau_lamdaP_s)*dphifdotplusdc;

        //---------------------//
        // COMPARE WITH NUMERICAL
        //---------------------//
        /*
        // Calculate numerical derivatives
        //std::cout << "Last iteration" << "\n";
        double phif_dot_plus_num; double phif_dot_minus;
        double kappa_dot_plus; double kappa_dot_minus;
        Vector3d a0_dot_plus; Vector3d a0_dot_minus;
        Vector3d lamdaP_dot_plus; Vector3d lamdaP_dot_minus;
        epsilon = 1e-7;
        double rho_plus = rho + epsilon;
        double rho_minus = rho - epsilon;
        double c_plus = c + epsilon;
        double c_minus = c - epsilon;
        Matrix3d CC_plus, CC_minus;

        // Call update function with plus and minus to get numerical derivatives
        evalForwardEulerUpdate2d(local_dt, local_parameters, c, rho_plus, FF, CC, phif, a0, s0, kappa, lamdaP, phif_dot_plus_num, a0_dot_plus, kappa_dot_plus, lamdaP_dot_plus);
        evalForwardEulerUpdate2d(local_dt, local_parameters, c, rho_minus, FF, CC,phif, a0, s0, kappa, lamdaP, phif_dot_minus, a0_dot_minus, kappa_dot_minus, lamdaP_dot_minus);
        dThetadrho_num(0) += local_dt*(1./(2.*epsilon))*(phif_dot_plus_num-phif_dot_minus);
        dThetadrho_num(1) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(0)-a0_dot_minus(0));
        dThetadrho_num(2) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(1)-a0_dot_minus(1));
        dThetadrho_num(3) += local_dt*(1./(2.*epsilon))*(kappa_dot_plus-kappa_dot_minus);
        dThetadrho_num(4) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(0)-lamdaP_dot_minus(0));
        dThetadrho_num(5) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(1)-lamdaP_dot_minus(1));

        evalForwardEulerUpdate2d(local_dt, local_parameters, c_plus, rho, FF, CC, phif, a0, s0, kappa, lamdaP, phif_dot_plus_num, a0_dot_plus, kappa_dot_plus, lamdaP_dot_plus);
        evalForwardEulerUpdate2d(local_dt, local_parameters, c_minus, rho, FF, CC, phif, a0, s0, kappa, lamdaP, phif_dot_minus, a0_dot_minus, kappa_dot_minus, lamdaP_dot_minus);
        dThetadc_num(0) += local_dt*(1./(2.*epsilon))*(phif_dot_plus_num-phif_dot_minus);
        dThetadc_num(1) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(0)-a0_dot_minus(0));
        dThetadc_num(2) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(1)-a0_dot_minus(1));
        dThetadc_num(3) += local_dt*(1./(2.*epsilon))*(kappa_dot_plus-kappa_dot_minus);
        dThetadc_num(4) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(0)-lamdaP_dot_minus(0));
        dThetadc_num(5) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(1)-lamdaP_dot_minus(1));

        counter = 0;
        for(int ii = 0; ii < 2; ii++){
            for(int jj = 0; jj < 2; jj++){
                CC_plus = CC;
                CC_minus = CC;
                CC_plus(ii,jj) += epsilon;
                CC_minus(ii,jj) -= epsilon;
                evalForwardEulerUpdate2d(local_dt, local_parameters, c, rho, FF, CC_plus, phif, a0, s0, kappa, lamdaP, phif_dot_plus_num, a0_dot_plus, kappa_dot_plus, lamdaP_dot_plus);
                evalForwardEulerUpdate2d(local_dt, local_parameters, c, rho, FF, CC_minus, phif, a0, s0, kappa, lamdaP, phif_dot_minus, a0_dot_minus, kappa_dot_minus, lamdaP_dot_minus);
                dThetadCC_num(0 + counter) += local_dt*(1./(2.*epsilon))*(phif_dot_plus_num-phif_dot_minus);
                dThetadCC_num(4 + counter) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(0)-a0_dot_minus(0));
                dThetadCC_num(8 + counter) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(1)-a0_dot_minus(1));
                dThetadCC_num(12 + counter) += local_dt*(1./(2.*epsilon))*(kappa_dot_plus-kappa_dot_minus);
                dThetadCC_num(16 + counter) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(0)-lamdaP_dot_minus(0));
                dThetadCC_num(20 + counter) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(1)-lamdaP_dot_minus(1));
                counter++;
            }
        }
        */
        // UPDATE VARIABLES
        //---------------------//
        // Collagen density PHI
        phif = phif + local_dt*(phif_dot);

        // Principal direction A0
        a0 = a0 + local_dt*a0_dot;
        // normalize a0
        a0 = a0/sqrt(a0.dot(a0));
        s0 = Rot90*a0;

        // Dispersion KAPPA
        kappa = kappa + local_dt*(kappa_dot);

        // Permanent deformation LAMDAP
        lamdaP = lamdaP + local_dt*(lamdaP_dot);
    }
    //myfile.close();

    //std::cout<<"lamda1: "<<lamda2d(1)<<", lamda0: "<<lamda2d(0)<<", lamdaP:"<<lamdaP(0)<<","<<lamdaP(1)<<",a0:"<<a0(0)<<","<<a0(1)<<"Ce_aa: "<<Ce_aa<<","<<Ce_ss<<"\n";
    //std::cout<<"\n CC \n"<<CCproj;
    //std::cout << "\n" << dThetadCC_num << "\n";
    //std::cout << "\n" << dThetadrho_num << "\n";
    //std::cout << "\n" << dThetadc_num << "\n";
}

void evalForwardEulerUpdate2d(double dt, const std::vector<double> &local_parameters, double c,double rho,const Matrix3d &FF, const Matrix3d &CC,
                              const double &phif, const Vector3d &a0, const Vector3d &s0, const double &kappa, const Vector3d &lamdaP,
                              double &phif_dot, Vector3d &a0_dot, double &kappa_dot, Vector3d &lamdaP_dot)
{
    //---------------------------------//
    // Parameters
    //
    // collagen fraction
    double p_phi = local_parameters[0]; // production by fibroblasts, natural rate
    double p_phi_c = local_parameters[1]; // production up-regulation, weighted by C and rho
    double p_phi_theta = local_parameters[2]; //production regulated by stretch
    double K_phi_c = local_parameters[3]; // saturation of C effect on deposition
    double K_phi_rho = local_parameters[4]; // saturation of collagen fraction itself
    double d_phi = local_parameters[5]; // rate of degradation
    double d_phi_rho_c = local_parameters[6]; // rate of degradation
    //
    // fiber alignment
    double tau_omega = local_parameters[7]; // time constant for angular reorientation
    //
    // dispersion parameter
    double tau_kappa = local_parameters[8]; // time constant
    double gamma_kappa = local_parameters[9]; // exponent of the principal stretch ratio
    //
    // permanent contracture/growth
    double tau_lamdaP_a = local_parameters[10]; // time constant for direction a
    double tau_lamdaP_s = local_parameters[11]; // time constant for direction s
    double tau_lamdaP_n = local_parameters[12]; // time constant for direction s
    //
    double vartheta_e = local_parameters[13]; // mechanosensing response
    double gamma_theta = local_parameters[14]; // exponent of the Heaviside function
    //

    double PIE = 3.14159;
    Matrix3d Identity;Identity<<1.,0.,0., 0.,1.,0., 0.,0.,1.;
    Matrix2d Identity2d; Identity2d << 1.,0., 0.,1.;
    Matrix3d Rot90;Rot90 << 0.,-1.,0., 1.,0.,0., 0.,0.,1.;
    Matrix2d Rot902d; Rot902d << 0.,-1., 1.,0.;

    Vector3d n0 = s0.cross(a0);
    if(n0(2)<0){
        n0 = a0.cross(s0);
    }
    //std::cout<<"a0"<<"\n"<<a0<<"\n"<<"s0"<<"\n"<<s0<<"\n"<<"n0"<<"\n"<<n0<<'\n';
    // fiber tensor in the reference
    Matrix3d a0a0 = a0*a0.transpose();
    Matrix3d s0s0 = s0*s0.transpose();
    Matrix3d n0n0 = n0*n0.transpose();
    // recompute split
    Matrix3d FFg = lamdaP(0)*(a0a0) + lamdaP(1)*(s0s0) + 1.*n0n0;
    Matrix3d FFginv = (1./lamdaP(0))*(a0a0) + (1./lamdaP(1))*(s0s0) + 1.*n0n0;
    //Matrix3d FFe = FF*FFginv;
    // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
    // elastic strain
    Matrix3d CCe = FFginv*CC*FFginv;
    Matrix3d CCeinv = CCe.inverse();
    // Jacobian of the deformations
    double Jp = lamdaP(0)*lamdaP(1)*1.;
    double Je = sqrt(CCe.determinant());
    double J = Je*Jp;

    //----------------//
    // UPDATE VARIABLES 2D VERSION
    //----------------//
    // re-compute basis a0, s0, n0. (If not always vertical, could find n0 with cross product or rotations. Be careful of sign.
    Vector2d a0proj = Vector2d(a0(0), a0(1));
    Vector2d s0proj = Rot902d*a0proj;
    // std::cout<<"a0"<<"\n"<<a0<<"\n"<<"s0"<<"\n"<<a0proj<<"\n";
    // fiber tensor in the reference
    Matrix2d a0a0proj = a0proj*a0proj.transpose();
    Matrix2d s0s0proj = s0proj*s0proj.transpose();
    // elastic strain
    Matrix2d CCeproj; CCeproj << CCe(0,0), CCe(0,1), CCe(1,0), CCe(1,1);
    //std::cout<<"\n"<<CCeproj<<"\n";

    // Eigenvalues
    // Use .compute() for the QR algorithm, .computeDirect() for an explicit trig
    // QR may be more accurate, but explicit is faster
    // Eigenvalues
    SelfAdjointEigenSolver<Matrix2d> es2d;
    Vector2d lamda2d;
    Matrix2d vectors2d;
    es2d.compute(CCeproj);
    lamda2d = es2d.eigenvalues();
    vectors2d = es2d.eigenvectors();
    if(a0proj.dot(vectors2d.col(1)) < 0){
        vectors2d = -vectors2d;
    }
    double He = 1./(1.+exp(-gamma_theta*(Je - vartheta_e)));
    //if(He<0.002){He=0;}

    //----------------//
    // 2D FORWARD-EULER EQUATIONS
    //----------------//
    // Collagen density PHI
    double phif_dot_plus = (p_phi + (p_phi_c*c)/(K_phi_c+c) + p_phi_theta*He)*(rho/(K_phi_rho+phif));
    //std::cout<<"phidotplus: "<<phif_dot_plus<<"\n";
    phif_dot = phif_dot_plus - (d_phi + c*rho*d_phi_rho_c)*phif;

    // Principal direction A0
    // Alternatively, see Menzel (NOTE THAT THE THIRD COMPONENT IS THE LARGEST ONE)
    // a0 = a0 + local_dt*(((2.*PIE*phif_dot_plus)/(tau_omega))*lamda(2)*(Identity-a0a0)*(vectors.col(2)));
    Vector2d a02d_dot = ((2.*PIE*phif_dot_plus)/(tau_omega))*lamda2d(1)*(Identity2d-a0a0proj)*vectors2d.col(1);
    a0_dot(0) = a02d_dot(0);
    a0_dot(1) = a02d_dot(1);

    // Dispersion KAPPA
    // kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamda(2)/lamda(1),gamma_kappa)/2. - kappa);
    kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamda2d(0)/lamda2d(1),gamma_kappa)/3. - kappa);

    // elastic stretches of the directions a and s
    double Ce_aaproj = a0proj.transpose()*CCeproj*a0proj;
    double Ce_ssproj = s0proj.transpose()*CCeproj*s0proj;
    double lamdaE_a = sqrt(Ce_aaproj);
    double lamdaE_s = sqrt(Ce_ssproj);

    // Permanent deformation LAMDAP
    lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-1)/tau_lamdaP_a;
    lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-1)/tau_lamdaP_s;
}


//========================================================//
// IMPLICIT LOCAL PROBLEM: structural update
//========================================================//
// THIS METHOD IS NOT FUNCTIONAL
// NEEDS A LOT OF WORK
void localWoundProblemImplicit(
        double dt, const std::vector<double> &local_parameters,
        const double &c, const double &rho,const Matrix3d &FF,
        const double &phif_0, const Vector3d &a0_0, const Vector3d &s0_0, const Vector3d &n0_0,
        const double &kappa_0, const Vector3d &lamdaP_0,
        double &phif, Vector3d &a0, Vector3d &s0, Vector3d &n0,
        double &kappa, Vector3d &lamdaP,
        VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadc)
{/*
    //---------------------------------//
    //
    // INPUT
    // 	matParam: material parameters
    //	rho: value of the cell density at the point
    //	c: concentration at the point
    //	CC: deformation at the point
    //	Theta_t: value of the parameters at previous time step
    //
    // OUTPUT
    //	Theta: value of the parameters at the current time step
    //	dThetadCC: derivative of Theta wrt global mechanics (CC)
    // 	dThetadrho: derivative of Theta wrt global rho
    // 	dThetadc: derivative of Theta wrt global C
    //
    //---------------------------------//

    CHECK ALL THE ROTATIONS TO MAKE SURE THERE ARE NO DELTA T TERMS NEEDED
    CHECK ALL NUMERICAL DERIVATIVES
    CHECK A0 AND A0_0 FOR THE ROTATIONS, MAKE SURE ORDER IS CORRECT

    //---------------------------------//
    // Parameters
    //
    // collagen fraction
    double p_phi = local_parameters[0]; // production by fibroblasts, natural rate
    double p_phi_c = local_parameters[1]; // production up-regulation, weighted by C and rho
    double p_phi_theta = local_parameters[2]; //production regulated by stretch
    double K_phi_c = local_parameters[3]; // saturation of C effect on deposition
    double K_phi_rho = local_parameters[4]; // saturation of collagen fraction itself
    double d_phi = local_parameters[5]; // rate of degradation
    double d_phi_rho_c = local_parameters[6]; // rate of degradation
    //
    // fiber alignment
    double tau_omega = local_parameters[7]; // time constant for angular reorientation
    //
    // dispersion parameter
    double tau_kappa = local_parameters[8]; // time constant
    double gamma_kappa = local_parameters[9]; // exponent of the principal stretch ratio
    //
    // permanent contracture/growth
    double tau_lamdaP_a = local_parameters[10]; // time constant for direction a
    double tau_lamdaP_s = local_parameters[11]; // time constant for direction s
    double tau_lamdaP_n = local_parameters[12]; // time constant for direction n
    //
    double vartheta_e = local_parameters[13]; // exponent of the Heaviside function
    double gamma_theta = local_parameters[14]; // mechanosensing response
    //
    // solution parameters
    double tol_local = local_parameters[15]; // local tolerance
    double time_step_ratio = local_parameters[16]; // time step ratio
    double max_local_iter = local_parameters[17]; // max local iter
    //
    // other local stuff
    double PIE = 3.14159;
    Matrix3d Identity;Identity<<1.,0.,0., 0.,1.,0., 0.,0.,1.;
    std::vector<Matrix3d> levicivita(3,Matrix3d::Zero());
    for(int k=0;k<3;k++){
        for(int j=0;j<3;j++){
            for(int i=0;i<3;i++){
                if(i==j || j==k || i==k){
                    levicivita[i](j,k) = +0.;
                }
                else if((i == 0 && j == 1 && k == 2) || (i == 1 && j == 2 && k == 0) || (i == 2 && j == 0 && k == 1)){
                    levicivita[i](j,k) = +1.;
                }
                else{
                    levicivita[i](j,k) = -1.;
                }
            }
        }
    }
    // some global
    Matrix3d Rot90;Rot90 << 0.,-1.,0., 1.,0.,0., 0.,0.,1.;
    //
    VectorXd voigt_table_I(6);
    voigt_table_I(0) = 0; voigt_table_I(1) = 1; voigt_table_I(2) = 2;
    voigt_table_I(3) = 1; voigt_table_I(4) = 0; voigt_table_I(5) = 0;
    VectorXd voigt_table_J(6);
    voigt_table_J(0) = 0; voigt_table_J(1) = 1; voigt_table_J(2) = 2;
    voigt_table_J(3) = 2; voigt_table_J(4) = 2; voigt_table_J(5) = 1;
    //
    //---------------------------------//



    //---------------------------------//
    // Preprocess the Newton
    //
    // initial guess for local newton
    phif = phif_0;
    // make sure it is unit length
    a0 = a0_0/(sqrt(a0_0.dot(a0_0)));
    kappa = kappa_0;
    lamdaP = lamdaP_0;

    //
    // initialize the residual and iterations
    int iter = 0;
    double residuum = 1.;
    double local_dt = dt; // /time_step_ratio
    //
    // Update kinematics
    Matrix3d CC = FF.transpose()*FF;
    Matrix3d CCinv = CC.inverse();
    // fiber tensor in the reference
    Matrix3d a0a0, s0s0, n0n0;
    // recompute split
    Matrix3d FFe, FFg, FFginv, CCe, CCeinv;

    // Jacobian of the deformations
    double Jp, Je, J, He;
    // calculate the normal stretch
    // double thetaP, theta_e, theta;
    // Eigen
    SelfAdjointEigenSolver<Matrix3d> eigensolver;
    Matrix3d vectors;
    Vector3d lamda, vectormax, vectormed, vectormin;
    double lamdamin, lamdamax, lamdamed;
    double lamdaE_a,lamdaE_s,lamdaE_n, phif_dot_plus, phif_dot, kappa_dot;
    Vector3d dlamdamaxda0, dlamdamedda0, dlamdaminda0, dlamdamaxdlamdaP, dlamdameddlamdaP, dlamdamindlamdaP;
    Matrix3d dvectormaxda0, dvectormaxdlamdaP;
    //
    // residual (phi, a0x, a0y, a0y, kappa, lamdaPa, lamdaPs, lamdaPn)
    VectorXd RR_local(8);
    double R_phif,R_kappa;
    Vector3d R_lamdaP,R_a0;
    double omeganorm;
    Vector3d omega, omegaunit, Rot_a0_0, domegadphif;
    Matrix3d Romega, omegaskew; // Be careful with dimensions
    Vector3d lamdaP_dot;
    //
    // tangent
    MatrixXd KK_local(8,8);
    double dJedlamdaP_a,dJedlamdaP_s,dJedlamdaP_n,dHedlamdaP_a,dHedlamdaP_s,dHedlamdaP_n;
    double dRphifdphif;
    Vector3d dRphifdlamdaP;
    Vector3d dphifplusdlamdaP;
    Matrix3d dRa0da0;
    Vector3d dRa0dphif, dRa0dlamdaPa, dRa0dlamdaPs, dRa0dlamdaPn;
    double dRkappadphif,dRkappadkappa;
    Vector3d dRkappada0, dRkappadlamdaP,dRlamdaPdphif;
    Vector3d dRlamdaPada0, dRlamdaPsda0, dRlamdaPnda0, dlamda_Eada0, dlamda_Esda0,dlamda_Enda0;
    double dRlamdaPadlamdaPa,dRlamdaPadlamdaPs,dRlamdaPadlamdaPn,dRlamdaPsdlamdaPa,dRlamdaPsdlamdaPs,dRlamdaPsdlamdaPn,dRlamdaPndlamdaPa,dRlamdaPndlamdaPs,dRlamdaPndlamdaPn;
    double dphifplusdphif;
    std::vector<Matrix3d> dRomegadomega(3,Matrix3d::Zero());
    std::vector<Matrix3d> dvector0da0(3,Matrix3d::Zero());
    std::vector<Matrix3d> dlamdaEdCC(3,Matrix3d::Zero());
    std::vector<Matrix3d> dCCeda0(3,Matrix3d::Zero());
    std::vector<Matrix3d> dCCeds0(3,Matrix3d::Zero());
    std::vector<Matrix3d> dCCedn0(3,Matrix3d::Zero());
    std::vector<Matrix3d> dCCedlamdaP(3,Matrix3d::Zero());
    Matrix3d domegada0, a0_skew;
    // derivatives
    double dlamda_EadlamdaPa,dlamda_EadlamdaPs,dlamda_EsdlamdaPa,dlamda_EsdlamdaPs;
    Vector3d dvectormaxda0x, dvectormaxda0y, dvectormaxda0z, dvectormaxdlamdaPa, dvectorMaxdlamdaPs, dvectorMaxdlamdaPn;
    Matrix3d dCCedCC, dlamdaEda0, dlamdaEds0, dlamdaEdn0, dlamdaEdlamdaP;
    //
    VectorXd SOL_local(8);
    //
    //---------------------------------//

    //---------------------------------//
    // NEWTON LOOP
    //---------------------------------//
    std::ofstream myfile;
    myfile.open("BE_3D_results.csv");
    while (residuum > tol_local && iter < max_local_iter) {
        myfile << std::fixed << std::setprecision(10) << local_dt*iter << "," << phif << "," << kappa << "," << a0(0) << "," << a0(1) << "," << a0(2) << "," << lamdaP(0) << "," << lamdaP(1) << "," << lamdaP(2) << "\n";
        //std::cout<<"iter : "<<iter<<"\n";

        //----------------//
        // UPDATE VARIABLES 3D VERSION
        //----------------//
        // Do not re-compute basis a0, s0, n0, since we are now updating all with Rodriguez
        // fiber tensor in the reference
        a0a0 = a0 * a0.transpose();
        s0s0 = s0 * s0.transpose();
        n0n0 = n0 * n0.transpose();
        // recompute split
        FFg = lamdaP(0) * (a0a0) + lamdaP(1) * (s0s0) + lamdaP(2) * n0n0;
        FFginv = (1. / lamdaP(0)) * (a0a0) + (1. / lamdaP(1)) * (s0s0) + (1. / lamdaP(2)) * (n0n0);
        FFe = FF * FFginv;
        // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
        // elastic strain
        CCe = FFe.transpose() * FFe;
        CCeinv = CCe.inverse();
        // Jacobian of the deformations
        Jp = lamdaP(0) * lamdaP(1) * lamdaP(2);
        Je = sqrt(CCe.determinant());
        J = Je * Jp;

        // Eigenvalues
        // Use .compute() for the QR algorithm, .computeDirect() for an explicit trig
        // QR may be more accurate, but explicit is faster (NOTE THAT THE LAST COMPONENT IS THE LARGEST ONE)
        eigensolver.compute(CCe);
        lamda = eigensolver.eigenvalues();
        lamdamax = lamda(2);
        lamdamed = lamda(1);
        lamdamin = lamda(0);
        vectors = eigensolver.eigenvectors();
        vectormax = vectors.col(2);
        vectormed = vectors.col(1);
        vectormin = vectors.col(0);
        if (a0.dot(vectormax) < 0) {
            vectormax = -vectormax;
        }
        //std::cout<<"\n"<<vector0<<"\n"<<lamda0<<"\n"<<lamda1<<"\n";

        //----------//
        // BACKWARD EULER RESIDUALS
        //----------//

        RR_local.setZero();

        //--------------------//
        // phif residual
        //--------------------//

        // heaviside functions
        He = 1. / (1 + exp(-gamma_theta * (Je - vartheta_e)));
        //if (He < 0.002) { He = 0; }
        //std::cout<<"He: "<<He<<"\n";
        phif_dot_plus = (p_phi + (p_phi_c * c) / (K_phi_c + c) + p_phi_theta * He) * (rho / (K_phi_rho + phif));
        //std::cout<<"phidotplus: "<<phif_dot_plus<<"\n";
        R_phif = (-phif + phif_0) / local_dt + phif_dot_plus - (d_phi + c * rho * d_phi_rho_c) * phif;
        //std::cout<<"Collagen fraction residual.\nphif_0= "<<phif_0<<", phif = "<<phif<<"\n";
        //std::cout<<"phif_dot_plus = "<<phif_dot_plus<<", phif_dot = "<<phif_dot<<"\n";
        //std::cout<<"R_phif = "<<R_phif<<"\n";

        //--------------------//
        // a0 residual
        //--------------------//
        omega = (2*phif_dot_plus*PIE/(tau_omega))*lamdamax*(a0.cross(vectormax));
        omeganorm = sqrt(omega.dot(omega));
        if(fabs(omeganorm)>1e-8){
            omegaunit = omega/omeganorm;
            omegaskew << 0, -omegaunit(2), omegaunit(1), omegaunit(2), 0, -omegaunit(0), -omegaunit(1), omegaunit(0), 0;
            Romega = cos(omeganorm*local_dt)*Identity + (1-cos(omeganorm*local_dt))*(omegaunit*omegaunit.transpose()) + sin(omeganorm*local_dt)*omegaskew;
            Rot_a0_0 = Romega*a0_0;
            if(fabs(omeganorm)<1e-8){
                a0 = Rot_a0_0;
            }
        }
        else{
            Rot_a0_0 = Identity*a0;
        }
        R_a0 = -a0 + Rot_a0_0;
        //std::cout<<"Fiber direction residual.\na0_0 = ["<<a0_0(0)<<","<<a0_0(1)<<"], a0 = ["<<a0(0)<<","<<a0(1)<<"]\n";
        //std::cout<<"R_a0 = "<<R_a0<<"\n";

        //--------------------//
        // kappa residual
        //--------------------//
        kappa_dot = (phif_dot_plus / tau_kappa) * (pow(lamdamin / lamdamax, gamma_kappa) / 3. - kappa);
        R_kappa = (-kappa + kappa_0) / local_dt + kappa_dot;

        //--------------------//
        // lamdaP residual
        //--------------------//
        // elastic stretches of the directions a, s, n
        lamdaE_a = sqrt(a0.transpose() * CCe * a0);
        lamdaE_s = sqrt(s0.transpose() * CCe * s0);
        lamdaE_n = sqrt(n0.transpose() * CCe * n0);
        lamdaP_dot(0) = phif_dot_plus * (lamdaE_a - 1) / tau_lamdaP_a;
        lamdaP_dot(1) = phif_dot_plus * (lamdaE_s - 1) / tau_lamdaP_s;
        lamdaP_dot(2) = phif_dot_plus * (lamdaE_n - 1) / tau_lamdaP_n;
        R_lamdaP = (1. / local_dt) * (-lamdaP + lamdaP_0) + lamdaP_dot;

        // Assemble into the residual vector
        RR_local(0) = R_phif;
        RR_local(1) = R_a0(0);
        RR_local(2) = R_a0(1);
        RR_local(3) = R_a0(2);
        RR_local(4) = R_kappa;
        RR_local(5) = R_lamdaP(0);
        RR_local(6) = R_lamdaP(1);
        RR_local(7) = R_lamdaP(2);

        //---------------------------//
        // BACKWARD EULER TANGENT
        //---------------------------//

        KK_local.setZero();

        //----------------------------------------------//
        // Differentiation of CCe, lamdaE wrt a0 and lamdaP
        //----------------------------------------------//

        // Analytical using Kronecker product definition
        for(int ii=0; ii<3; ii++){
            for(int jj=0; jj<3; jj++){
                for(int kk=0; kk<3; kk++){
                    for(int ll=0; ll<3; ll++) {
                        for (int mm=0; mm<3; mm++) {
                            //dCCeda0[mm](ii,jj) = 1/(lamdaP(0)*lamdaP(0))*CC + 1/(lamdaP(0)*lamdaP(0))*CC + 1/(lamdaP(0)*lamdaP(0))*CC;
                            //dCCeds0[mm](ii,jj) = ;
                            //dCCedn0[mm](ii,jj) = ;

                            //dCCedlamdaP[mm](ii,jj) = ;
                        }
                    }
                }
            }
        }

        // Compare with numerical
        std::vector<Matrix3d> dCCeda0num(3,Matrix3d::Zero());
        std::vector<Matrix3d> dCCeds0num(3,Matrix3d::Zero());
        std::vector<Matrix3d> dCCedn0num(3,Matrix3d::Zero());
        std::vector<Matrix3d> dCCedlamdaPnum(3,Matrix3d::Zero());
        for(int ii=0; ii<3; ii++){
            double epsilon = 1e-7;
            Vector3d a0_plus = a0; a0_plus(ii) += epsilon;
            Vector3d a0_minus = a0; a0_minus(ii) -= epsilon;
            Matrix3d a0a0_plus = a0_plus*a0_plus.transpose();
            Matrix3d a0a0_minus = a0_minus*a0_minus.transpose();
            Matrix3d FFginv_plus = (1. / lamdaP(0)) * (a0a0_plus) + (1. / lamdaP(1)) * (s0s0) + (1. / lamdaP(2)) * (n0n0);
            Matrix3d FFginv_minus = (1. / lamdaP(0)) * (a0a0_minus) + (1. / lamdaP(1)) * (s0s0) + (1. / lamdaP(2)) * (n0n0);
            Matrix3d CCe_plus = FFginv_plus*CC*FFginv_plus;
            Matrix3d CCe_minus = FFginv_minus*CC*FFginv_minus;
            dCCeda0num[ii] = (1./(2.*epsilon))*(CCe_plus-CCe_minus);

            Vector3d s0_plus = s0; s0_plus(ii) += epsilon;
            Vector3d s0_minus = s0; s0_minus(ii) -= epsilon;
            Matrix3d s0s0_plus = s0_plus*s0_plus.transpose();
            Matrix3d s0s0_minus = s0_minus*s0_minus.transpose();
            FFginv_plus = (1. / lamdaP(0)) * (a0a0) + (1. / lamdaP(1)) * (s0s0_plus) + (1. / lamdaP(2)) * (n0n0);
            FFginv_minus = (1. / lamdaP(0)) * (a0a0) + (1. / lamdaP(1)) * (s0s0_minus) + (1. / lamdaP(2)) * (n0n0);
            CCe_plus = FFginv_plus*CC*FFginv_plus;
            CCe_minus = FFginv_minus*CC*FFginv_minus;
            dCCeds0num[ii] = (1./(2.*epsilon))*(CCe_plus-CCe_minus);

            Vector3d n0_plus = n0; n0_plus(ii) += epsilon;
            Vector3d n0_minus = n0; n0_minus(ii) -= epsilon;
            Matrix3d n0n0_plus = n0_plus*n0_plus.transpose();
            Matrix3d n0n0_minus = n0_minus*n0_minus.transpose();
            FFginv_plus = (1. / lamdaP(0)) * (a0a0) + (1. / lamdaP(1)) * (s0s0) + (1. / lamdaP(2)) * (n0n0_plus);
            FFginv_minus = (1. / lamdaP(0)) * (a0a0) + (1. / lamdaP(1)) * (s0s0) + (1. / lamdaP(2)) * (n0n0_minus);
            CCe_plus = FFginv_plus*CC*FFginv_plus;
            CCe_minus = FFginv_minus*CC*FFginv_minus;
            dCCedn0num[ii] = (1./(2.*epsilon))*(CCe_plus-CCe_minus);

            Vector3d lamdaP_plus = lamdaP; lamdaP_plus(ii) += epsilon;
            Vector3d lamdaP_minus = lamdaP; lamdaP_minus(ii) -= epsilon;
            FFginv_plus = (1. / lamdaP_plus(0)) * (a0a0) + (1. / lamdaP_plus(1)) * (s0s0) + (1. / lamdaP_plus(2)) * (n0n0);
            FFginv_minus = (1. / lamdaP_minus(0)) * (a0a0) + (1. / lamdaP_minus(1)) * (s0s0) + (1. / lamdaP_minus(2)) * (n0n0);
            CCe_plus = FFginv_plus*CC*FFginv_plus;
            CCe_minus = FFginv_minus*CC*FFginv_minus;
            dCCedlamdaPnum[ii] = (1./(2.*epsilon))*(CCe_plus-CCe_minus);
        }



        Matrix3d ds0da0; ds0da0.setZero();
        Matrix3d dn0da0; dn0da0.setZero();

        // Construct a rotation using Rodriguez formula to get the new s0 and n0
        Vector3d across = a0.cross(a0_0);
        double across_norm = sqrt(across.dot(across));
        Vector3d across_unit = across/across_norm;
        //Matrix3d a0skew; a0skew << 0, -a0cross(2), a0cross(1), a0cross(2), 0, -a0cross(0), -a0cross(1), a0cross(0), 0;
        Matrix3d across_skew; across_skew << 0, -across(2), across(1), across(2), 0, -across(0), -across(1), across(0), 0;
        //double theta = acos(a0.dot(a0_00)/(sqrt(a0.dot(a0))*sqrt(a0_00.dot(a0_00))));
        std::vector<Matrix3d> dRota0da0(3,Matrix3d::Identity());
        for(int ii=0; ii<3; ii++){
            for(int jj=0; jj<3; jj++){
                for(int kk=0; kk<3; kk++){
                    if(!a0.isApprox(a0_0)){
                        //Rot = Identity + sin(theta)*a0skew + (1-cos(theta))*(a0skew*a0skew);
                        Matrix3d Rot = (1/(sqrt(a0.dot(a0))*sqrt(a0_0.dot(a0_0))))*((a0.dot(a0_0))*Matrix3d::Identity() + across_skew + ((sqrt(a0.dot(a0))*sqrt(a0_0.dot(a0_0))-(a0.dot(a0_0)))/(sqrt(across.dot(across))))*across * across.transpose());
                    }
                }
            }
        }

        // s0 is Rot(a0_0 to a0)*s0_0 so ds0da0 is dRotda0*s0_0
        std::vector<Matrix3d> dRacrossda0(3,Matrix3d::Zero());
        for(int ii = 0; ii<3; ii++){
            for(int jj = 0; jj<3; jj++){
                for(int kk = 0; kk<3; kk++){
                    for(int ll = 0; ll<3; ll++){
                        dRacrossda0[ii](jj,kk) = -sin(across_norm)*Identity(ii,jj)*across_unit(kk) +
                                ((1-cos(across_norm))/(across_norm))*(Identity(ii,kk)*across_unit(jj) + Identity(jj,kk)*across_unit(ii)
                                - (sin(across_norm)/(across_norm))*levicivita[ii](jj,kk)
                                + (sin(across_norm) - (2*(1-cos(across_norm))/(across_norm)))*across_unit(ii)*across_unit(jj)*across_unit(kk)
                                + (sin(across_norm)/(across_norm) - cos(across_norm))*levicivita[ii](jj,ll)*across_unit(ll)*across_unit(kk));
                    }
                }
            }
        }

        for(int ii = 0; ii<3; ii++){
            for(int jj = 0; jj<3; jj++){
                for(int kk = 0; kk<3; kk++){
                    if(fabs(across_norm) > 1e-8){
                        ds0da0(ii,kk) = dRacrossda0[ii](jj,kk)*s0_0(jj);
                        dn0da0(ii,kk) = dRacrossda0[ii](jj,kk)*n0_0(jj);
                    }
                    else{
                        // If a0_0 ~ a_0, the derivative is just given by the Levi-Civita symbol
                        ds0da0(ii,kk) = levicivita[ii](jj,kk)*s0_0(jj);
                        dn0da0(ii,kk) = levicivita[ii](jj,kk)*n0_0(jj);
                    }
                }
            }
        }

        // dCCeda0 with implicit derivatives
        for(int ii=0; ii<3; ii++){
            for(int jj=0; jj<3; jj++){
                for(int kk=0; kk<3; kk++){
                    for(int ll=0; ll<3; ll++) {
                        dCCeda0[ii](jj,kk) = dCCeda0num[ii](jj,kk) + dCCeds0num[ll](jj,kk)*ds0da0(ll,ii) + dCCedn0num[ll](jj,kk)*dn0da0(ll,ii);
                        dCCedlamdaP[ii](jj,kk) = dCCedlamdaPnum[ii](jj,kk);
                    }
                }
            }
        }

        // dCCedlamdaP is fine
        for(int ii=0; ii<3; ii++){
            for(int jj=0; jj<3; jj++){
                for(int kk=0; kk<3; kk++){
                    dlamdaEdlamdaP(0,kk) = a0(ii)*dCCedlamdaP[kk](ii,jj)*a0(jj);
                    dlamdaEdlamdaP(1,kk) = s0(ii)*dCCedlamdaP[kk](ii,jj)*s0(jj);
                    dlamdaEdlamdaP(2,kk) = n0(ii)*dCCedlamdaP[kk](ii,jj)*n0(jj);
                }
            }
        }

        //----------------------------------------------//
        // Numerical differentiation of lamda, vector
        //----------------------------------------------//

        // Use numerical approximation to eigenvalue derivatives
        for(int ii=0; ii<3; ii++){
            for(int jj=0; jj<3; jj++){
                for(int kk=0; kk<3; kk++){
                    dlamdamaxda0(kk) = vectormax(ii)*dCCeda0[kk](ii,jj)*vectormax(jj);
                    dlamdamedda0(kk) = vectormed(ii)*dCCeda0[kk](ii,jj)*vectormed(jj);
                    dlamdaminda0(kk) = vectormin(ii)*dCCeda0[kk](ii,jj)*vectormin(jj);
                    dlamdamaxdlamdaP(kk) = vectormax(ii)*dCCedlamdaP[kk](ii,jj)*vectormax(jj);
                    dlamdameddlamdaP(kk) = vectormed(ii)*dCCedlamdaP[kk](ii,jj)*vectormed(jj);
                    dlamdamindlamdaP(kk) = vectormin(ii)*dCCedlamdaP[kk](ii,jj)*vectormin(jj);
                }
            }
        }
        // Calculate derivatives of eigenvectors
        Matrix3d LHS; LHS.setZero();
        MatrixXd SOL(3,3); SOL.setZero();
        MatrixXd RHS_a0(3,3); RHS_a0.setZero();
        MatrixXd RHS_lamdaP(3,3); RHS_lamdaP.setZero();
        // If CC is the identity matrix, the eigenvectors are arbitrary which is problematic.
        // Beware the matrix becoming singular. Need to perturb.
        double epsilon = 1e-8;
        double delta = 1e-8;
        if(abs(lamdamin-lamdamed) < epsilon || abs(lamdamin-lamdamax) < epsilon || abs(lamdamed-lamdamax) < epsilon){
            lamdamax = lamdamax*(1+delta);
            lamdamin = lamdamin*(1-delta);
            lamdamed = lamdamed/((1+delta)*(1-delta));
        }
        // Create matrix for calculation of derivatives of eigenvectors.
        LHS << CCe(0,0) - lamdamax, CCe(0,1), CCe(0,2),
                CCe(1,0), CCe(1,1) - lamdamax, CCe(1,2),
                CCe(2,0), CCe(2,1), CCe(2,2) - lamdamax;
        for (int ii=0; ii<3; ii++){
            for (int jj=0; jj<3; jj++) {
                for (int kk=0; kk<3; kk++) {
                    // Create vector for right hand side. It is the product of the derivative and eigenvector
                    RHS_a0(ii,kk) = dCCeda0[kk](ii,jj)*vectormax(jj);
                    RHS_lamdaP(ii,kk) = dCCedlamdaP[kk](ii,jj)*vectormax(jj);
                }
            }
        }
        // This is an eigenvector system, so it is singular and we need a Moore-Penrose Pseudoinverse
        // See The Matrix Cookbook or Abou-Moustafa 2009
        //JacobiSVD<Matrix3d> SVD(LHS,ComputeThinU|ComputeThinV);
        //SOL = SVD.solve(RHS_a0);
        SOL = LHS.colPivHouseholderQr().solve(RHS_a0);
        dvectormaxda0 = SOL;
        //SOL = SVD.solve(RHS_lamdaP);
        SOL = LHS.colPivHouseholderQr().solve(RHS_lamdaP);
        dvectormaxdlamdaP = SOL;

        //------------------//
        // Tangent of phif
        //------------------//
        // derivative of the phifdotplus (phif, lamdaP) (not kappa, a0)
        dphifplusdphif = (p_phi + (p_phi_c*c)/(K_phi_c+c) + p_phi_theta*He)*(-rho/((K_phi_rho+phif)*(K_phi_rho+phif)));
        dRphifdphif = -1. / local_dt + dphifplusdphif - (d_phi + c * rho * d_phi_rho_c);
        dJedlamdaP_a = -J/(lamdaP(0)*lamdaP(0)*lamdaP(1)*lamdaP(2));
        dJedlamdaP_s = -J/(lamdaP(0)*lamdaP(1)*lamdaP(1)*lamdaP(2));
        dJedlamdaP_n = -J/(lamdaP(0)*lamdaP(1)*lamdaP(2)*lamdaP(2));
        dHedlamdaP_a = -1.0*He*He*exp(-gamma_theta*(Je-vartheta_e))*(-gamma_theta*(dJedlamdaP_a));
        dHedlamdaP_s = -1.0*He*He*exp(-gamma_theta*(Je-vartheta_e))*(-gamma_theta*(dJedlamdaP_s));
        dHedlamdaP_n = -1.0*He*He*exp(-gamma_theta*(Je-vartheta_e))*(-gamma_theta*(dJedlamdaP_n));
        dRphifdlamdaP(0) = p_phi_theta * dHedlamdaP_a * (rho / (K_phi_rho + phif));
        dRphifdlamdaP(1) = p_phi_theta * dHedlamdaP_s * (rho / (K_phi_rho + phif));
        dRphifdlamdaP(2) = p_phi_theta * dHedlamdaP_n * (rho / (K_phi_rho + phif));
        dphifplusdlamdaP = dRphifdlamdaP;
        //std::cout<<"Collagen fraction tangent.\n";
        //std::cout<<"dphifplusdphif = "<<dphifplusdphif<<"\n";
        //std::cout<<"dRphifdphif = "<<dRphifdphif<<"\n";

        // Numerical check

        //------------------//
        // Tangent of a0
        //------------------//
        // derivative of Romega wrt omega
        for(int ii = 0; ii<3; ii++){
            for(int jj = 0; jj<3; jj++){
                for(int kk = 0; kk<3; kk++){
                    if(fabs(omeganorm) > 1e-8){
                        for(int ll = 0; ll<3; ll++){
                            dRomegadomega[ii](jj,kk) = -sin(omeganorm*local_dt)*Identity(ii,jj)*omegaunit(kk) +
                                    ((1-cos(omeganorm*local_dt))/(omeganorm*local_dt))*(Identity(ii,kk)*omegaunit(jj) + Identity(jj,kk)*omegaunit(ii)
                                    - (sin(omeganorm*local_dt)/(omeganorm*local_dt))*levicivita[ii](jj,kk)
                                    + (sin(omeganorm*local_dt) - (2*(1-cos(omeganorm*local_dt))/(omeganorm*local_dt)))*omegaunit(ii)*omegaunit(jj)*omegaunit(kk)
                                    + (sin(omeganorm*local_dt)/(omeganorm*local_dt) - cos(omeganorm*local_dt))*levicivita[ii](jj,ll)*omegaunit(ll)*omegaunit(kk));
                        }
                    }
                    else{
                        dRomegadomega[ii](jj,kk) = levicivita[ii](jj,kk);
                    }
                }
            }
        }

        a0_skew.setZero(); a0_skew << 0, -a0(2), a0(1), a0(2), 0, -a0(0), -a0(1), a0(0), 0;
        Matrix3d skewvectormax; skewvectormax << 0, -vectormax(2), vectormax(1), vectormax(2), 0, -vectormax(0), -vectormax(1), vectormax(0), 0;
        domegada0 = -(2*PIE*dphifplusdphif/(tau_omega))*skewvectormax + (2*PIE*dphifplusdphif/(tau_omega))*a0_skew*dvectormaxda0;

        std::vector<Matrix3d> dRomegada0(3,Matrix3d::Zero());
        for(int ii = 0; ii<3; ii++){
            for(int jj = 0; jj<3; jj++){
                for(int kk = 0; kk<3; kk++){
                    for(int ll = 0; ll<3; ll++){
                        dRomegada0[kk](ii,jj) = dRomegadomega[ll](ii,jj)*domegada0(ll,kk);
                    }
                }
            }
        }

        // derivative of the omega angular velocity wrt phif
        domegadphif = (2*PIE*dphifplusdphif/(tau_omega))*lamdamax*(a0.cross(vectormax));
        // chain rule for derivative of residual wrt phif
        for(int ii=0;ii<3;ii++){
            for(int jj=0;jj<3;jj++){
                for(int kk=0;kk<3;kk++){
                    dRa0dphif(ii) = -dRomegadomega[kk](ii,jj)*a0_0(jj)*domegadphif(kk);
                }
            }
        }

        // derivative of omega wrt a0
        for(int ii=0;ii<3;ii++){
            for(int jj=0;jj<3;jj++){
                for(int kk=0;kk<3;kk++){
                    dRa0da0(ii,kk) = - Identity(ii,kk) + (dRomegada0[kk](ii,jj)*a0_0(jj));
                }
            }
        }

        // derivative of Ra0 wrt the dispersion is zero.
        // derivative of Ra0 wrt lamdaP requires some calculations
        Matrix3d domegadlamdaP; domegadlamdaP.setZero();
        for(int ii = 0; ii<3; ii++){
            for(int jj = 0; jj<3; jj++){
                for(int kk = 0; kk<3; kk++){
                    for(int ll = 0; ll<3; ll++){
                        domegadlamdaP(ii,jj) = (2*PIE*phif_dot_plus/tau_omega)*(lamdamax*a0_skew(ii,kk)*dvectormaxdlamdaP(kk,jj)
                                + (a0.cross(vectormax)(ii))*dlamdamaxdlamdaP(jj)) + ((2.*PIE*dphifplusdlamdaP(jj))/(tau_omega))*lamdamax*(a0.cross(vectormax)(ii));
                    }
                }
            }
        }

        Matrix3d dRa0dlamdaP; dRa0dlamdaP.setZero();
        for(int ii=0;ii<3;ii++){
            for(int jj=0;jj<3;jj++){
                for(int kk=0;kk<3;kk++){
                    for(int ll=0;ll<3;ll++){
                        dRa0dlamdaP(ii,jj) = (dRomegadomega[kk](ii,ll)*a0_0(ll))*domegadlamdaP(kk,jj);
                    }
                }
            }
        }

        //std::cout<<"Fiber direction KK.\nRa0dphif = ["<<dRa0dphif<<"], dRa0da0 = ["<<dRa0da0x<<","<<dRa0da0y<<"], dRa0dlamdaP = ["<<dRa0dlamdaPa<<","<<dRa0dlamdaPs<<"]\n";

        //------------------//
        // Tangent of kappa
        //------------------//
        dRkappadphif = (dphifplusdphif/tau_kappa)*(pow(lamdamin/lamdamax, gamma_kappa)/3.-kappa);
        dRkappada0 = (phif_dot_plus*gamma_kappa/(3.*tau_kappa))*((pow(lamdamin / lamdamax, gamma_kappa-1))*dlamdaminda0 -
                      (pow(lamdamin/lamdamax,gamma_kappa-1)*(1/(lamdamax * lamdamax))*dlamdamaxda0));
        dRkappadkappa = -1/local_dt - (phif_dot_plus/tau_kappa);
        dRkappadlamdaP = (dphifplusdlamdaP/tau_kappa)*(pow(lamdamin/lamdamax, gamma_kappa)/3. - kappa) +
                (phif_dot_plus*gamma_kappa/(3.*tau_kappa))*((pow(lamdamin/lamdamax,gamma_kappa-1))*dlamdamindlamdaP -
                (pow(lamdamin/lamdamax,gamma_kappa-1)*(1/(lamdamax*lamdamax))*dlamdamaxdlamdaP));

        //------------------//
        // Tangent of lamdaP
        //------------------//
        // derivative wrt phif
        dRlamdaPdphif(0) = dphifplusdphif*((lamdaE_a - 1.)/tau_lamdaP_a);
        dRlamdaPdphif(1) = dphifplusdphif*((lamdaE_s - 1.)/tau_lamdaP_s);
        dRlamdaPdphif(2) = dphifplusdphif*((lamdaE_n - 1.)/tau_lamdaP_n);
        // derivative wrt fiber direction
        dRlamdaPada0 = phif_dot_plus * (dlamda_Eada0) / tau_lamdaP_a;
        dRlamdaPsda0 = phif_dot_plus * (dlamda_Esda0) / tau_lamdaP_s;
        dRlamdaPnda0 = phif_dot_plus * (dlamda_Enda0) / tau_lamdaP_n;
        // no dependence on the fiber dispersion
        // derivative wrt the lamdaP
        dRlamdaPadlamdaPa = -1./local_dt + (phif_dot_plus/tau_lamdaP_a)*dlamdaEdlamdaP(0,0) + (dphifplusdlamdaP(0)/tau_lamdaP_a)*(lamdaE_a-1.);
        dRlamdaPadlamdaPs = (phif_dot_plus / tau_lamdaP_a)*dlamdaEdlamdaP(0,1) + (dphifplusdlamdaP(1)/tau_lamdaP_a)*(lamdaE_a - 1.);
        dRlamdaPadlamdaPn = (phif_dot_plus / tau_lamdaP_a)*dlamdaEdlamdaP(0,2) + (dphifplusdlamdaP(2)/tau_lamdaP_a)*(lamdaE_a - 1.);

        dRlamdaPsdlamdaPa = (phif_dot_plus / tau_lamdaP_s)*dlamdaEdlamdaP(1,0) + (dphifplusdlamdaP(0)/tau_lamdaP_s)*(lamdaE_s - 1.);
        dRlamdaPsdlamdaPs = -1./local_dt + (phif_dot_plus/tau_lamdaP_s)*dlamdaEdlamdaP(1,1) + (dphifplusdlamdaP(1)/tau_lamdaP_s)*(lamdaE_s - 1.);
        dRlamdaPsdlamdaPn = (phif_dot_plus/tau_lamdaP_s)*dlamdaEdlamdaP(1,2) + (dphifplusdlamdaP(2)/tau_lamdaP_s)*(lamdaE_s - 1.);

        dRlamdaPndlamdaPa = (phif_dot_plus/tau_lamdaP_n)*dlamdaEdlamdaP(2,0) + (dphifplusdlamdaP(0)/tau_lamdaP_n)*(lamdaE_n - 1.);
        dRlamdaPndlamdaPs = (phif_dot_plus/tau_lamdaP_n)*dlamdaEdlamdaP(2,1) + (dphifplusdlamdaP(1)/tau_lamdaP_n)*(lamdaE_n - 1.);
        dRlamdaPndlamdaPn = -1./local_dt + (phif_dot_plus/tau_lamdaP_s)*dlamdaEdlamdaP(2,2) + (dphifplusdlamdaP(2)/tau_lamdaP_n)*(lamdaE_n - 1.);

        //----------------------------------------------//
        // ASSEMBLY INTO TANGENT MATRIX
        //----------------------------------------------//
        // phif
        KK_local(0, 0) = dRphifdphif;
        KK_local(0, 5) = dRphifdlamdaP(0);
        KK_local(0, 6) = dRphifdlamdaP(1);
        KK_local(0, 7) = dRphifdlamdaP(2);
        // a0x
        KK_local(1, 0) = dRa0dphif(0);
        KK_local(1, 1) = dRa0da0(0, 0);
        KK_local(1, 2) = dRa0da0(0, 1);
        KK_local(1, 3) = dRa0da0(0, 2);
        KK_local(1, 5) = dRa0dlamdaP(0,0);
        KK_local(1, 6) = dRa0dlamdaP(0,1);
        KK_local(1, 7) = dRa0dlamdaP(0,2);
        // a0y
        KK_local(2, 0) = dRa0dphif(1);
        KK_local(2, 1) = dRa0da0(1, 0);
        KK_local(2, 2) = dRa0da0(1, 1);
        KK_local(2, 3) = dRa0da0(1, 2);
        KK_local(2, 5) = dRa0dlamdaP(1,0);
        KK_local(2, 6) = dRa0dlamdaP(1,1);
        KK_local(2, 7) = dRa0dlamdaP(1,2);
        // a0z
        KK_local(3, 0) = dRa0dphif(2);
        KK_local(3, 1) = dRa0da0(2, 0);
        KK_local(3, 2) = dRa0da0(2, 1);
        KK_local(3, 3) = dRa0da0(2, 2);
        KK_local(3, 5) = dRa0dlamdaP(2,0);
        KK_local(3, 6) = dRa0dlamdaP(2,1);
        KK_local(3, 7) = dRa0dlamdaP(2,2);
        // kappa
        KK_local(4, 0) = dRkappadphif;
        KK_local(4, 1) = dRkappada0(0);
        KK_local(4, 2) = dRkappada0(1);
        KK_local(4, 3) = dRkappada0(2);
        KK_local(4, 4) = dRkappadkappa;
        KK_local(4, 5) = dRkappadlamdaP(0);
        KK_local(4, 6) = dRkappadlamdaP(1);
        KK_local(4, 7) = dRkappadlamdaP(2);
        // lamdaPa
        KK_local(5, 0) = dRlamdaPdphif(0);
        KK_local(5, 1) = dRlamdaPada0(0);
        KK_local(5, 2) = dRlamdaPada0(1);
        KK_local(5, 3) = dRlamdaPada0(2);
        KK_local(5, 5) = dRlamdaPadlamdaPa;
        KK_local(5, 6) = dRlamdaPadlamdaPs;
        KK_local(5, 7) = dRlamdaPadlamdaPn;
        // lamdaPs
        KK_local(6, 0) = dRlamdaPdphif(1);
        KK_local(6, 1) = dRlamdaPsda0(0);
        KK_local(6, 2) = dRlamdaPsda0(1);
        KK_local(6, 3) = dRlamdaPsda0(2);
        KK_local(6, 5) = dRlamdaPsdlamdaPa;
        KK_local(6, 6) = dRlamdaPsdlamdaPs;
        KK_local(6, 7) = dRlamdaPsdlamdaPn;
        // lamdaPn
        KK_local(7, 0) = dRlamdaPdphif(2);
        KK_local(7, 1) = dRlamdaPnda0(0);
        KK_local(7, 2) = dRlamdaPnda0(1);
        KK_local(7, 3) = dRlamdaPnda0(2);
        KK_local(7, 5) = dRlamdaPndlamdaPa;
        KK_local(7, 6) = dRlamdaPndlamdaPs;
        KK_local(7, 7) = dRlamdaPndlamdaPn;


        //----------//
        // SOLVE
        //----------//

        //std::cout<<"SOLVE.\nRR_local\n"<<RR_local<<"\nKK_local\n"<<KK_local<<"\n";

        double normRR = sqrt(RR_local.dot(RR_local));
        residuum = normRR;
        // solve
        SOL_local = KK_local.lu().solve(-RR_local);

        //std::cout<<"SOL_local\n"<<SOL_local<<"\n";
        // update the solution
        double normSOL = sqrt(SOL_local.dot(SOL_local));
        phif += SOL_local(0);
        // Solve for delta_a0, then get delta_omega, then rotate a0, do not add
        Vector3d delta_a0; delta_a0.setZero();
        delta_a0(0) = SOL_local(1);
        delta_a0(1) = SOL_local(2);
        delta_a0(2) = SOL_local(3);
        Vector3d delta_omega; delta_omega.setZero();
        delta_omega = a0_0.cross(delta_a0);
        double delta_omega_norm = sqrt(delta_omega.dot(delta_omega));
        if(fabs(delta_omega_norm)>1e-8){
            Vector3d delta_omega_unit = delta_omega/delta_omega_norm;
            Matrix3d delta_omega_skew; delta_omega_skew << 0, -delta_omega_unit(2), delta_omega_unit(1), delta_omega_unit(2), 0, -delta_omega_unit(0), -delta_omega_unit(1), delta_omega_unit(0), 0;
            Romega = cos(delta_omega_norm)*Identity + (1-cos(delta_omega_norm))*(delta_omega_unit*delta_omega_unit.transpose()) + sin(delta_omega_norm)*delta_omega_skew;
        }
        else{
            Romega = Identity;
        }
        a0 = Romega*a0_0;
        kappa += SOL_local(4);
        lamdaP(0) += SOL_local(5);
        lamdaP(1) += SOL_local(6);
        lamdaP(2) += SOL_local(7);

        // normalize a0 and update other vectors
        a0 = a0 / sqrt(a0.dot(a0));
        // Rotate others with the same incremental rotation
        s0 = Romega * s0_0;
        s0 = s0 / sqrt(s0.dot(s0));
        n0 = Romega * n0_0;
        n0 = n0 / sqrt(n0.dot(n0));
        // Or rotate the others from the initial using a0 and a0_0
//        Matrix3d Rot = Matrix3d::Identity();
//        if(!a0.isApprox(a0_0)){
//            //Vector3d a0cross = a0.cross(a0_00)/(sqrt((a0.cross(a0_00)).dot(a0.cross(a0_00))));
//            Vector3d across = a0.cross(a0_0);
//            //Matrix3d a0skew; a0skew << 0, -a0cross(2), a0cross(1), a0cross(2), 0, -a0cross(0), -a0cross(1), a0cross(0), 0;
//            Matrix3d askew; askew << 0, -across(2), across(1), across(2), 0, -across(0), -across(1), across(0), 0;
//            //double theta = acos(a0.dot(a0_00)/(sqrt(a0.dot(a0))*sqrt(a0_00.dot(a0_00))));
//            //Rot = Identity + sin(theta)*a0skew + (1-cos(theta))*(a0skew*a0skew);
//            Matrix3d Ra0 = (1/(sqrt(a0.dot(a0))*sqrt(a0_0.dot(a0_0))))*((a0.dot(a0_0))*Matrix3d::Identity() + askew + ((sqrt(a0.dot(a0))*sqrt(a0_0.dot(a0_0))-(a0.dot(a0_0)))/(sqrt(across.dot(across))))*across * across.transpose());
//        }
//        s0 = Rot*s0;
//        n0 = Rot*n0;

        //std::cout<<"a0"<<"\n"<<a0<<"\n"<<"s0"<<"\n"<<s0<<"\n"<<"n0"<<"\n"<<n0<<'\n';
        //std::cout<<"norm(RR): "<<residuum<<"\n";
        //std::cout<<"norm(SOL): "<<normSOL<<"\n";
        iter += 1;
        std::cout<<"Finish local Newton.\niter = "<<iter<<", residuum = "<<residuum<<"\n";
        std::cout << "\nphif: " << phif << ", kappa: " << kappa << ", lamdaP:" << lamdaP(0) << "," << lamdaP(1) << "," << lamdaP(2)
                  << ",a0:" << a0(0) << "," << a0(1) << "," << a0(2) << "\n";
        if (iter == max_local_iter && normRR > tol_local) { //
            std::cout << "\nno local convergence\n";
            //std::cout << "\nphif: " << phif << ", kappa: " << kappa << ", lamdaP:" << lamdaP(0) << "," << lamdaP(1) << "," << lamdaP(2)
            //          << ",a0:" << a0(0) << "," << a0(1) << "," << a0(2) << "\n";"
            //std::cout << "Res\n" << RR_local << "\nSOL_local\n" << SOL_local << "\n";
            //throw std::runtime_error("sorry pal ");
        }
    } // END OF WHILE LOOP OF LOCAL NEWTON
    myfile.close();

    //-----------------------------------//
    // WOUND TANGENTS FOR GLOBAL PROBLEM
    //-----------------------------------//

    //----------------//
    // FINAL UPDATE VARIABLES 3D VERSION
    //----------------//
    // fiber tensor in the reference
    a0a0 = a0 * a0.transpose();
    s0s0 = s0 * s0.transpose();
    n0n0 = n0 * n0.transpose();
    // recompute split
    FFg = lamdaP(0) * (a0a0) + lamdaP(1) * (s0s0) + lamdaP(2) * n0n0;
    FFginv = (1. / lamdaP(0)) * (a0a0) + (1. / lamdaP(1)) * (s0s0) + (1. / lamdaP(2)) * (n0n0);
    FFe = FF * FFginv;
    // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
    // elastic strain
    CCe = FFe.transpose() * FFe;
    CCeinv = CCe.inverse();
    // Jacobian of the deformations
    Jp = lamdaP(0) * lamdaP(1) * lamdaP(2);
    Je = sqrt(CCe.determinant());
    J = Je * Jp;

    // Eigenvalues
    // Use .compute() for the QR algorithm, .computeDirect() for an explicit trig
    // QR may be more accurate, but explicit is faster (NOTE THAT THE LAST COMPONENT IS THE LARGEST ONE)
    eigensolver.compute(CCe);
    lamda = eigensolver.eigenvalues();
    lamdamax = lamda(2);
    lamdamed = lamda(1);
    lamdamin = lamda(0);
    vectors = eigensolver.eigenvectors();
    vectormax = vectors.col(2);
    vectormed = vectors.col(1);
    vectormin = vectors.col(0);
    if (a0.dot(vectormax) < 0) {
        vectormax = -vectormax;
    }

    // heaviside functions
    He = 1. / (1 + exp(-gamma_theta * (Je - vartheta_e)));
    //if (He < 0.002) { He = 0; }

    // elastic stretches of the directions a, s, n
    lamdaE_a = sqrt(a0.transpose() * CCe * a0);
    lamdaE_s = sqrt(s0.transpose() * CCe * s0);
    lamdaE_n = sqrt(n0.transpose() * CCe * n0);

    //----------------------------------------//
    // CALCULATE GLOBAL CHAIN RULE DERIVATIVES 3D
    //----------------------------------------//
    // For these, we use the initial values for the parameters to appropriately couple with the global derivatives

    // Calculate derivatives of eigenvalues and eigenvectors
    Matrix3d dCCdCC; dCCdCC.setZero();
    Matrix4d LHS; LHS.setZero();
    Vector4d RHS,SOL; RHS.setZero(), SOL.setZero();
    //std::vector<Vector3d> dvectordCCe(6,Vector3d::Zero());
    //std::vector<Vector3d> dvectordCC(6,Vector3d::Zero());
    std::vector<Matrix3d> dvectormaxdCCe(3,Matrix3d::Zero());
    std::vector<Matrix3d> dvectormaxdCC(3,Matrix3d::Zero());

    // If CC is the identity matrix, the eigenvectors are arbitrary which is problematic.
    // Beware the matrix becoming singular. Need to perturb.
    double epsilon = 1e-8;
    double delta = 1e-8;
    if(abs(lamdamin-lamdamed) < epsilon || abs(lamdamin-lamdamax) < epsilon || abs(lamdamed-lamdamax) < epsilon){
        lamdamax = lamdamax*(1+delta);
        lamdamin = lamdamin*(1-delta);
        lamdamed = lamdamed/((1+delta)*(1-delta));
    }
    // We actually only  need one eigenvector so an outer loop is not needed, but if we want more just change to 3.
    // Create matrix for calculation of derivatives of eigenvalues and vectors.
    LHS << CCe(0,0) - lamdamax, CCe(0,1), CCe(0,2), -vectormax(0),
            CCe(1,0), CCe(1,1) - lamdamax, CCe(1,2), -vectormax(1),
            CCe(2,0), CCe(2,1), CCe(2,2) - lamdamax, -vectormax(2),
            vectormax(0), vectormax(1), vectormax(2), 0;
    // CC is symmetric so we actually only need 6 components.
    //std::cout<<"\n"<<MM<<"\n"<<MM.determinant()<<"\n";
    for (int ii=0; ii<3; ii++){
        for (int jj=0; jj<3; jj++) {
            for (int kk=0; kk<3; kk++) {
                // Create vector for right hand side. It is the product of an elementary matrix with the eigenvector.
                RHS.setZero();
                RHS(ii) = -vectormax(jj);
                // Solve
                SOL = LHS.lu().solve(RHS);
                dvectormaxdCCe[kk](ii,jj) = SOL(kk); // ii,jj counts the components of CCe, kk index has the eigenvector components
                //dlamdamaxdCCe = SOL(3);
            }
        }
    }

    for (int ii=0; ii<3; ii++){
        for (int jj=0; jj<3; jj++) {
            for (int kk=0; kk<3; kk++){
                for (int ll=0; ll<3; ll++) {
                    for (int mm=0; mm<3; mm++) {
                        dvectormaxdCC[mm](kk,ll) = dvectormaxdCCe[0](ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                    }
                }
            }
        }
    }

    // Alternatively for the eigenvalue we can use the rule from Holzapfel
    // But we still need dCCedCC for the chain rule
    Matrix3d dlamdamaxdCCe = vectormax*vectormax.transpose();
    Matrix3d dlamdameddCCe = vectormed*vectormed.transpose();
    Matrix3d dlamdamindCCe = vectormin*vectormin.transpose();

    // Multiply by dCCdCCe to get dlamdadCC
    Matrix3d dlamdamaxdCC; Matrix3d dlamdameddCC; Matrix3d dlamdamindCC;
    dlamdamaxdCC.setZero(); dlamdameddCC.setZero(); dlamdamindCC.setZero();
    for (int ii=0; ii<3; ii++){
        for (int jj=0; jj<3; jj++) {
            for (int kk=0; kk<3; kk++){
                for (int ll=0; ll<3; ll++) {
                    dlamdamaxdCC(kk,ll) = dlamdamaxdCCe(ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                    dlamdameddCC(kk,ll) = dlamdameddCCe(ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                    dlamdamindCC(kk,ll) = dlamdamindCCe(ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                }
            }
        }
    }

    // Calculate derivative of lamdaE wrt CC. This will involve an elementary matrix.
    Matrix3d dlamdaE_a_dCC; dlamdaE_a_dCC.setZero();
    Matrix3d dlamdaE_s_dCC; dlamdaE_s_dCC.setZero();
    Matrix3d dlamdaE_n_dCC; dlamdaE_n_dCC.setZero();
    // Matrix multiplication is associative, so d(a0*FFginv)*CC*(FFginv*a0)/dCC
    // is the outer product of the two vectors we get from a0*FFginv
    // and the symmetry makes the calculation easier
    for (int ii=0; ii<3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            for (int kk = 0; kk < 3; kk++) {
                for (int ll = 0; ll < 3; ll++) {
                    dlamdaE_a_dCC(jj,kk) = (a0(ii) * FFginv(ii,jj)) * (FFginv(kk,ll) * a0(ll));
                    dlamdaE_s_dCC(jj,kk) = (s0(ii) * FFginv(ii,jj)) * (FFginv(kk,ll) * s0(ll));
                    dlamdaE_n_dCC(jj,kk) = (n0(ii) * FFginv(ii,jj)) * (FFginv(kk,ll) * n0(ll));
                }
            }
        }
    }

    // Explicit derivatives of residuals wrt CC

    // phif
    phif_dot_plus = (p_phi + (p_phi_c * c) / (K_phi_c + c) + p_phi_theta * He) * (rho / (K_phi_rho + phif));
    Matrix3d dRphifdCC;dRphifdCC.setZero();
    Matrix3d dHedCC_explicit = -1./pow((1.+exp(-gamma_theta*(Je - vartheta_e))),2)*(exp(-gamma_theta*(Je - vartheta_e)))*(-gamma_theta)*(J*CCinv/(2*Jp));
    dRphifdCC = p_phi_theta*dHedCC_explicit*(rho/(K_phi_rho+phif));
    Matrix3d dphifdotplusdCC = p_phi_theta*dHedCC_explicit*(rho/(K_phi_rho+phif));

    std::vector<Matrix3d> domegadCC;
    for(int ii=0;ii<3;ii++){
        for(int jj=0;jj<3;jj++){
            for(int kk=0;kk<3;kk++){
                for(int ll=0;ll<3;ll++){
                    domegadCC[kk](ii,jj) = (2*PIE/tau_omega)*a0_skew(kk,ll)*dvectormaxdCC[ll](ii,jj);
                }
            }
        }
    }

    // a0
    std::vector<Matrix3d> dRa0dCC(3, Matrix3d::Zero());
    for(int ii=0;ii<3;ii++){
        for(int jj=0;jj<3;jj++){
            for(int kk=0;kk<3;kk++){
                for(int ll=0;ll<3;ll++){
                    for(int mm=0;mm<3;mm++){
                        dRa0dCC[kk](ii,jj) = (-dRomegadomega[mm](kk,ll)*a0_0(ll))*domegadCC[mm](ii,jj);
                    }
                }
            }
        }
    }

    // kappa
    Matrix3d dRkappadCC;
    dRkappadCC = (dphifdotplusdCC/tau_kappa)*(pow(lamdamin/lamdamax,gamma_kappa)/3. - kappa) +
            (phif_dot_plus/tau_kappa)*((gamma_kappa/3.)*pow(lamdamin/lamdamax,gamma_kappa-1))*((1./lamdamax)*dlamdamindCC
                                                - (lamdamin/(lamdamax*lamdamax))*dlamdamaxdCC);

    // lamdaP
    Matrix3d dRlamdaP_adCC;
    Matrix3d dRlamdaP_sdCC;
    Matrix3d dRlamdaP_ndCC;
    dRlamdaP_adCC = (phif_dot_plus/tau_lamdaP_a)*dlamdaE_a_dCC  + (dphifdotplusdCC/tau_lamdaP_a)*(lamdaE_a-1.);
    dRlamdaP_sdCC = (phif_dot_plus/tau_lamdaP_s)*dlamdaE_s_dCC + (dphifdotplusdCC/tau_lamdaP_s)*(lamdaE_s-1.);
    dRlamdaP_ndCC = (phif_dot_plus/tau_lamdaP_n)*dlamdaE_n_dCC + (dphifdotplusdCC/tau_lamdaP_n)*(lamdaE_s-1.);

    // Assemble dRThetadCC, count is 9*8=72. It is symmetric, so could be 48, but I will solve full for now.
    VectorXd dRThetadCC(72);dRThetadCC.setZero();
    for(int kk=0;kk<6;kk++){
        int ii = voigt_table_I(kk);
        int jj = voigt_table_J(kk);
        dRThetadCC(0+kk) = dRphifdCC(ii,jj);
        dRThetadCC(9+kk) = dRa0dCC[0](ii,jj);
        dRThetadCC(18+kk) = dRa0dCC[1](ii,jj);
        dRThetadCC(27+kk) = dRa0dCC[2](ii,jj);
        dRThetadCC(36+kk) = dRkappadCC(ii,jj);
        dRThetadCC(45+kk) = dRlamdaP_adCC(ii,jj);
        dRThetadCC(54+kk) = dRlamdaP_sdCC(ii,jj);
        dRThetadCC(63+kk) = dRlamdaP_ndCC(ii,jj);
    }
    // Assemble dRThetadTheta/KK_local_extended
    MatrixXd dRThetadTheta_ext(72,72);dRThetadTheta_ext.setZero();
    for(int ii=0;ii<8;ii++){ // ii are the components of KK
        for(int jj=0;jj<8;jj++){ // jj are the components of CC
            for(int kk=0;kk<6;kk++){ // kk are the Voigt components
                dRThetadTheta_ext(6*ii+kk,6*jj+kk) = KK_local(ii,jj);
            }
        }
    }

    // SOLVE for the dThetadCC
    dThetadCC = dRThetadTheta_ext.lu().solve(-dRThetadCC);



    //----------//
    // RHO
    //----------//

    // Explicit derivatives of the residuals with respect to rho

    // phi
    double dphif_dot_plusdrho = (p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*He)*(1./(K_phi_rho+phif));

    // a0
    Vector3d domegadrho = ((2.*PIE*dphif_dot_plusdrho)/(tau_omega))*lamdamax*(a0.cross(vectormax));
    Vector3d dRa0drho; dRa0drho.setZero();
    for(int ii=0;ii<3;ii++){
        for(int jj=0;jj<3;jj++){
            for(int kk=0;kk<3;kk++){
                dRa0drho(ii) = -dRomegadomega[kk](ii,jj)*a0_0(jj)*domegadrho(kk);
            }
        }
    }

    // Aseemble in one vector
    VectorXd dRThetadrho(8);
    // phi
    dRThetadrho(0) = dphif_dot_plusdrho - d_phi_rho_c*c*phif;
    // a0
    dRThetadrho(1) = dRa0drho(0);
    dRThetadrho(2) = dRa0drho(1);
    dRThetadrho(3) = dRa0drho(2);
    // kappa
    dRThetadrho(4) = (dphif_dot_plusdrho/tau_kappa)*( pow(lamdamin/lamdamax,gamma_kappa)/3.  - kappa);
    // lamdaP
    dRThetadrho(5) = dphif_dot_plusdrho*(lamdaE_a-1)/tau_lamdaP_a;
    dRThetadrho(6) = dphif_dot_plusdrho*(lamdaE_s-1)/tau_lamdaP_s;
    dRThetadrho(7) = dphif_dot_plusdrho*(lamdaE_n-1)/tau_lamdaP_n;

    // the tangent matrix in this case remains the same as KK_local
    dThetadrho = KK_local.lu().solve(-dRThetadrho);



    //----------//
    // c
    //----------//

    // Explicit derivatives of the residuals with respect to c

    // phi
    double dphif_dot_plusdc = (rho/(K_phi_rho+phif))*((p_phi_c)/(K_phi_c+c) - (p_phi_c*c)/((K_phi_c+c)*(K_phi_c+c)));

    // a0
    Vector3d domegadc = ((2.*PIE*dphif_dot_plusdc)/(tau_omega))*lamdamax*(a0.cross(vectormax));
    Vector3d dRa0dc; dRa0dc.setZero();
    for(int ii=0;ii<3;ii++){
        for(int jj=0;jj<3;jj++){
            for(int kk=0;kk<3;kk++){
                dRa0dc(ii) = -dRomegadomega[kk](ii,jj)*a0_0(jj)*domegadc(kk);
            }
        }
    }

    // Aseemble in one vector
    VectorXd dRThetadc(8);
    //phif
    dRThetadc(0) = dphif_dot_plusdc - d_phi_rho_c*rho*phif;
    // a0
    dRThetadc(1) = dRa0dc(0);
    dRThetadc(2) = dRa0dc(1);
    dRThetadc(3) = dRa0dc(2);
    // kappa
    dRThetadc(4) = (dphif_dot_plusdc/tau_kappa)*( pow(lamdamin/lamdamax,gamma_kappa)/3.  - kappa);
    // lamdaP
    dRThetadc(5) = dphif_dot_plusdc*(lamdaE_a-1)/tau_lamdaP_a;
    dRThetadc(6) = dphif_dot_plusdc*(lamdaE_s-1)/tau_lamdaP_s;
    dRThetadc(7) = dphif_dot_plusdc*(lamdaE_n-1)/tau_lamdaP_n;

    // the tangent matrix in this case remains the same as KK_local
    dThetadc = KK_local.lu().solve(-dRThetadc); */
}


//========================================================//
// LOCAL PROBLEM: structural update
//========================================================//

void localWoundProblemImplicit2d(
        double dt, const std::vector<double> &local_parameters,
        double c,double rho,const Matrix3d &FF3d,
        double phif_0, const Vector3d &a0_03d, const Vector3d &s0_03d, double kappa_0, const Vector3d &lamdaP_03d,
        double &phif, Vector3d &a03d, Vector3d &s03d, double &kappa, Vector3d &lamdaP3d,
        VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadc)
{

    //---------------------------------//
    //
    // INPUT
    // 	matParam: material parameters
    //	rho: value of the cell density at the point
    //	c: concentration at the point
    //	CC: deformation at the point
    //	Theta_t: value of the parameters at previous time step
    //
    // OUTPUT
    //	Theta: value of the parameters at the current time step
    //	dThetadCC: derivative of Theta wrt global mechanics (CC)
    // 	dThetadrho: derivative of Theta wrt global rho
    // 	dThetadc: derivative of Theta wrt global C
    //
    //---------------------------------//

    Vector2d a0; a0 << a03d(0), a03d(1);
    Vector2d lamdaP; lamdaP << lamdaP3d(0), lamdaP3d(1);
    Vector2d a0_0; a0_0 << a0_03d(0), a0_03d(1);
    Vector2d lamdaP_0; lamdaP_0 << lamdaP_03d(0), lamdaP_03d(1);
    Matrix3d CC3d = FF3d.transpose()*FF3d;
    Matrix2d FF; FF << FF3d(0,0), FF3d(0,1), FF3d(1,0), FF3d(1,1);
    Matrix2d CC = FF.transpose()*FF;
    double Je, Jp; double J = sqrt(CC3d.determinant());

    //---------------------------------//
    // Parameters
    //
    // collagen fraction
    double p_phi = local_parameters[0]; // production by fibroblasts, natural rate
    double p_phi_c = local_parameters[1]; // production up-regulation, weighted by C and rho
    double p_phi_theta = local_parameters[2]; //production regulated by stretch
    double K_phi_c = local_parameters[3]; // saturation of C effect on deposition
    double K_phi_rho = local_parameters[4]; // saturation of collagen fraction itself
    double d_phi = local_parameters[5]; // rate of degradation
    double d_phi_rho_c = local_parameters[6]; // rate of degradation
    //
    // fiber alignment
    double tau_omega = local_parameters[7]; // time constant for angular reorientation
    //
    // dispersion parameter
    double tau_kappa = local_parameters[8]; // time constant
    double gamma_kappa = local_parameters[9]; // exponent of the principal stretch ratio
    //
    // permanent contracture/growth
    double tau_lamdaP_a = local_parameters[10]; // time constant for direction a
    double tau_lamdaP_s = local_parameters[11]; // time constant for direction s
    double tau_lamdaP_n = local_parameters[12]; // time constant for direction n
    //
    double vartheta_e = local_parameters[13]; // exponent of the Heaviside function
    double gamma_theta = local_parameters[14]; // mechanosensing response
    //
    // solution parameters
    double tol_local = local_parameters[15]; // local tolerance
    double time_step_ratio = local_parameters[16]; // time step ratio
    double max_local_iter = local_parameters[17]; // max local iter
    //
    // other local stuff
    double local_dt = dt/time_step_ratio;
    Matrix2d Identity;Identity<<1.,0.,0.,1.;
    // Might as well just pass this guy directly
    //Matrix2d CC = FF.transpose()*FF; // right Cauchy-Green deformation tensor
    double theta = sqrt(CC.determinant());
    double theta_e;
    //double H_theta;
    double He;
    Matrix2d CCinv = CC.inverse();
    double PIE = 3.14159;
    //
    //---------------------------------//



    //---------------------------------//
    // Preprocess the Newton
    //
    // initial guess for local newton
    phif = phif_0;
    double phif_00 = phif;
    // make sure it is unit length
    a0 = a0_0/(sqrt(a0_0.dot(a0_0)));
    Vector2d a0_00 = a0;
    kappa = kappa_0;
    double kappa_00 = kappa;
    lamdaP = lamdaP_0;
    Vector2d lamdaP_00 = lamdaP;

    //
    // initialize the residual and iterations
    int iter = 0;
    double residuum0 = 1.;
    double residuum = 1.;
    //
    // Declare Variables
    //
    // some global
    std::vector<Vector2d> Ebasis; Ebasis.clear();
    Matrix2d Rot90;Rot90<<0.,-1.,1.,0.;
    Matrix2d Rot180;Rot180<<-1.,0.,0.,-1.;
    Ebasis.push_back(Vector2d(1,0)); Ebasis.push_back(Vector2d(0,1));
    //
    // residual
    VectorXd RR_local(6);
    double R_phif,R_kappa; Vector2d R_lamdaP,R_a0;
    double phif_dot_plus,phif_dot;
    Vector2d s0;
    double Ce_aa,Ce_ss,Ce_as,lamda0, lamda1, sinVartheta,omega;
    Matrix2d Romega; Vector2d Rot_a0_0;
    double kappa_dot;
    double lamdaE_a,lamdaE_s;
    Vector2d lamdaP_dot;
    //
    // tangent
    MatrixXd KK_local(6,6);
    double dtheta_edlamdaP_a,dtheta_edlamdaP_s;
    //double dH_thetadlamdaP_a,dH_thetadlamdaP_s;
    double dJedlamdaP_a, dJedlamdaP_s;
    double dHedlamdaP_a,dHedlamdaP_s;
    double dRphifdphif,dRphifdlamdaP_a,dRphifdlamdaP_s;
    Vector2d dphifplusdlamdaP;
    Vector2d dRa0dphif; Matrix2d dRa0da0, dRa0dlamdaP;
    double dRkappadphif,dRkappadkappa;
    Vector2d dRkappada0, dRkappadlamdaP,dRlamdaPdphif;
    Vector2d dRlamdaPada0, dRlamdaPsda0;
    double dRlamdaPadlamdaPa,dRlamdaPadlamdaPs,dRlamdaPsdlamdaPa,dRlamdaPsdlamdaPs;
    double dphifplusdphif;
    Matrix2d dRomegadomega ;
    double domegadphif;
    Vector2d dCe_aada0,dCe_ssda0,dCe_asda0;
    double dCe_aada0x,dCe_aada0y,dCe_ssda0x,dCe_ssda0y,dCe_asda0x,dCe_asda0y;
    Vector2d dlamda1da0, dlamda0da0,dsinVarthetada0,domegada0;
    double aux00,aux01;
    double dlamda1dCe_aa,dlamda1dCe_ss,dlamda1dCe_as;
    double dlamda0dCe_aa,dlamda0dCe_ss,dlamda0dCe_as;
    double dsinVarthetadlamda1,dsinVarthetadCe_aa,dsinVarthetadCe_ss,dsinVarthetadCe_as;
    Vector2d dCe_aadlamdaP,dCe_ssdlamdaP,dCe_asdlamdaP;
    Vector2d dlamda1dlamdaP,dlamda0dlamdaP;
    Vector2d dsinVarthetadlamdaP, domegadlamdaP;
    //
    VectorXd SOL_local(6);
    //
    //---------------------------------//

    //---------------------------------//
    // NEWTON LOOP
    //---------------------------------//
    //std::ofstream myfile;
    //myfile.open("BE_results.csv");
    for(int step = 0; step < time_step_ratio; step++){
        //myfile << std::fixed << std::setprecision(10) << local_dt*step << "," << phif << "," << kappa << "," << a0(0) << "," << a0(1) << "," << lamdaP(0) << "," << lamdaP(1) << "\n";
        // RESET LOCAL NEWTON-RAPHSON
        iter = 0;
        residuum = 1;
        while(residuum>tol_local && iter<max_local_iter){

            //std::cout<<"iter : "<<iter<<"\n";

            //----------//
            // RESIDUALS
            //----------//

            RR_local.setZero();

            // collagen fraction residual
            // heaviside functions
            // theta_e = theta/theta_p
            //theta = sqrt(CC.determinant());
            Jp = (lamdaP(0)*lamdaP(1));
            Je = J/Jp;
            //theta_e = theta/(lamdaP(0)*lamdaP(1));
            //H_theta = 1./(1+exp(-gamma_theta*(theta_e-vartheta_e)));
            //if(H_theta<0.002){H_theta=0;}
            He = 1./(1+exp(-gamma_theta*(Je - vartheta_e)));
            //if(He<0.002){He=0;}
            //std::cout<<"H_theta: "<<H_theta<<"\n";
            phif_dot_plus = (p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*He)*(rho/(K_phi_rho+phif));
            //std::cout<<"phidotplus: "<<phif_dot_plus<<"\n";
            phif_dot = phif_dot_plus - (d_phi + c*rho*d_phi_rho_c)*phif;
            R_phif = (-phif + phif_0)/local_dt + phif_dot;

            //std::cout<<"Collagen fraction residual.\nphif_0= "<<phif_0<<", phif = "<<phif<<"\n";
            //std::cout<<"phif_dot_plus = "<<phif_dot_plus<<", phif_dot = "<<phif_dot<<"\n";
            //std::cout<<"R_phif = "<<R_phif<<"\n";

            // fiber orientation residual
            // given the current guess of the direction make the choice of s
            s0 = Rot90*a0;
            // compute the principal eigenvalues and eigenvectors of Ce as a function of
            // the structural variables, namely lamdaP and a0
            Ce_aa = (1./(lamdaP(0)*lamdaP(0)))*(CC(0,0)*a0(0)*a0(0)+2*CC(0,1)*a0(0)*a0(1)+CC(1,1)*a0(1)*a0(1));
            Ce_as = (1./(lamdaP(0)*lamdaP(1)))*(CC(0,0)*a0(0)*s0(0)+CC(0,1)*a0(0)*s0(1)+CC(1,0)*s0(0)*a0(1)+CC(1,1)*a0(1)*s0(1));
            Ce_ss = (1./(lamdaP(1)*lamdaP(1)))*(CC(0,0)*s0(0)*s0(0)+2*CC(0,1)*s0(0)*s0(1)+CC(1,1)*s0(1)*s0(1));
            lamda1 = ((Ce_aa + Ce_ss) + sqrt( (Ce_aa-Ce_ss)*(Ce_aa-Ce_ss) + 4*Ce_as*Ce_as))/2.; // the eigenvalue is a squared number by notation
            lamda0 = ((Ce_aa + Ce_ss) - sqrt( (Ce_aa-Ce_ss)*(Ce_aa-Ce_ss) + 4*Ce_as*Ce_as))/2.; // the eigenvalue is a squared number by notation
            if(fabs(lamda1-lamda0)<1e-7 || fabs(lamda1-Ce_aa)<1e-7){
                // equal eigenvalues means multiple of identity -> you can't possibly reorient.
                // or, eigenvector in the direction of a0 already -> no need to reorient since you are already there
                sinVartheta = 0.;
            }else{
                // if eigenvalues are not the same and the principal eigenvalue is not already in the direction of a0
                sinVartheta = (lamda1-Ce_aa)/sqrt(Ce_as*Ce_as + (lamda1-Ce_aa)*(lamda1-Ce_aa));
            }

            // Alternative is to do in the original coordinates
            double lamdaP_a = lamdaP(0);
            double lamdaP_s = lamdaP(1);
            double a0x = a0(0);
            double a0y = a0(1);
            double s0x = s0(0);
            double s0y = s0(1);
            double C00 = CC(0,0);
            double C01 = CC(0,1);
            double C11 = CC(1,1);
            double Ce00 = (-a0x*a0y*(lamdaP_a - lamdaP_s)*(C01*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a) - C11*a0x*a0y*(lamdaP_a - lamdaP_s)) + (C00*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a) - C01*a0x*a0y*(lamdaP_a - lamdaP_s))*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
            double Ce01 = (a0x*a0y*(lamdaP_a - lamdaP_s)*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - (a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a)*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
            double Ce11 = (a0x*a0y*(lamdaP_a - lamdaP_s)*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - (a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
            double Tr = Ce00+Ce11;
            double Det = Ce00*Ce11 - Ce01*Ce01;
            //lamda1 = Tr/2. + sqrt(Tr*Tr/4. - Det);
            //lamda0 = Tr/2. - sqrt(Tr*Tr/4. - Det);
            /*
            if(fabs(lamda1-lamda0)<1e-7){
                // equal eigenvalues, no way to reorient
                sinVartheta = 0;
            }else if(fabs(Ce01)<1e-7 && fabs(Ce00-lamda1)>1e-7){
                sinVartheta = (-Ce01*a0y + a0x*(-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/2)/(Ce00-lamda1);
            }else{
                sinVartheta = (Ce01*a0x - a0y*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/2)/(sqrt((Ce01*Ce01)+(Ce11-lamda1)*(Ce11-lamda1)));
            } */
            //std::cout<<"iter = "<<iter<<", lamda1 = "<<lamda1<<", lamda0 = "<<lamda0<<", sinVartheta = "<<sinVartheta<<"\n";
            // Compute the angular velocity
            omega = ((2.*PIE*phif_dot_plus)/(tau_omega))*lamda1*sinVartheta; // lamda1 is already squared by notation
            //omega = -((2.*PIE*phif_dot_plus)/(tau_omega))*lamda1*sinVartheta*sinVartheta; // always move opposite to sinvartheta<pi/2
            // compute the rotation tensor
            Romega(0,0) = cos(omega*local_dt); Romega(0,1) = -sin(omega*local_dt);
            Romega(1,0) = sin(omega*local_dt); Romega(1,1) = cos(omega*local_dt);
            // rotate the previous fiber
            Rot_a0_0 = Romega*a0_0;
            // residual
            if(fabs(omega)<1e-8){
                a0 = Rot_a0_0;
            }
            R_a0 = a0 - Rot_a0_0;

            //std::cout<<"Fiber direction residual.\na0_0 = ["<<a0_0(0)<<","<<a0_0(1)<<"], a0 = ["<<a0(0)<<","<<a0(1)<<"]\n";
            //std::cout<<"CCe (in a,s system) = ["<<Ce_aa<<","<<Ce_as<<","<<Ce_ss<<"]\n";
            //std::cout<<"sinVartheta = "<<sinVartheta<<", lamda1 = "<<lamda1<<", lamda0 = "<<lamda0<<"\n";
            //std::cout<<"Romega\n"<<Romega<<"\n";

            // dispersion residual
            kappa_dot = (phif_dot_plus/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/3.  -kappa);
            R_kappa = (-kappa+kappa_0)/local_dt + kappa_dot;

            // permanent deformation residual
            // elastic stretches of the directions a and s
            lamdaE_a = sqrt(Ce_aa);
            lamdaE_s = sqrt(Ce_ss);
            lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-1)/tau_lamdaP_a;
            lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-1)/tau_lamdaP_s;
            R_lamdaP= (1./local_dt)*(-lamdaP +lamdaP_0) + lamdaP_dot;

            // Assemble into the residual vector
            RR_local(0) = R_phif;
            RR_local(1) = R_a0(0);
            RR_local(2) = R_a0(1);
            RR_local(3) = R_kappa;
            RR_local(4) = R_lamdaP(0);
            RR_local(5) = R_lamdaP(1);

            //----------//
            // TANGENT
            //----------//

            KK_local.setZero();

            // Tangent of phif
            // derivative of the phifdotplus
            dphifplusdphif = (p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*He)*(-rho/((K_phi_rho+phif)*(K_phi_rho+phif)));
            dRphifdphif = -1./local_dt + dphifplusdphif - d_phi;
            //dtheta_edlamdaP_a = -theta/(lamdaP(0)*lamdaP(0)*lamdaP(1));
            //dtheta_edlamdaP_s = -theta/(lamdaP(0)*lamdaP(1)*lamdaP(1));
            //dH_thetadlamdaP_a = -1.0*H_theta*H_theta*exp(-gamma_theta*(theta_e-vartheta_e))*(-gamma_theta*(dtheta_edlamdaP_a));
            //dH_thetadlamdaP_s= -1.0*H_theta*H_theta*exp(-gamma_theta*(theta_e-vartheta_e))*(-gamma_theta*(dtheta_edlamdaP_s));
            dJedlamdaP_a = -J/(lamdaP(0)*lamdaP(0)*lamdaP(1)*1.);
            dJedlamdaP_s = -J/(lamdaP(0)*lamdaP(1)*lamdaP(1)*1.);
            dHedlamdaP_a = -1.0*He*He*exp(-gamma_theta*(Je - vartheta_e))*(-gamma_theta*(dJedlamdaP_a));
            dHedlamdaP_s = -1.0*He*He*exp(-gamma_theta*(Je - vartheta_e))*(-gamma_theta*(dJedlamdaP_s));
            dRphifdlamdaP_a = p_phi_theta*dHedlamdaP_a*(rho/(K_phi_rho+phif));
            dRphifdlamdaP_s = p_phi_theta*dHedlamdaP_s*(rho/(K_phi_rho+phif));
            dphifplusdlamdaP =Vector2d(p_phi_theta*dHedlamdaP_a*(rho/(K_phi_rho+phif)),p_phi_theta*dHedlamdaP_s*(rho/(K_phi_rho+phif)));

            //std::cout<<"Collagen fraction tangent.\n";
            //std::cout<<"dphifplusdphif = "<<dphifplusdphif<<"\n";
            //std::cout<<"dRphifdphif = "<<dRphifdphif<<"\n";

            // Tangent of a0
            // derivative of the rotation matrix wrt omega
            dRomegadomega(0,0) = -sin(omega*local_dt)*local_dt; dRomegadomega(0,1) = -cos(omega*local_dt)*local_dt;
            dRomegadomega(1,0) = cos(omega*local_dt)*local_dt;  dRomegadomega(1,1)= -sin(omega*local_dt)*local_dt;
            // derivative of the omega angular velocity wrt phif
            domegadphif = (2.*PIE*dphifplusdphif)*lamda1*sinVartheta/(tau_omega);
            //domegadphif = -(2.*PIE*dphifplusdphif)*lamda1*sinVartheta*sinVartheta/(tau_omega);
            // chain rule for derivative of residual wrt phif
            dRa0dphif = (-dRomegadomega*a0_0)*domegadphif;

            // derivative of R_a0 wrt to a0 needs some pre-calculations
            // derivatives of Ce wrt a0
            dCe_aada0.setZero(); dCe_asda0.setZero(); dCe_ssda0.setZero();
            for(int alpha=0; alpha<2; alpha++){
                for(int beta=0; beta<2; beta++){
                    dCe_aada0 += (CC(alpha,beta)/(lamdaP(0)*lamdaP(0)))*(a0(alpha)*Ebasis[beta] + a0(beta)*Ebasis[alpha]);
                    dCe_asda0 += (CC(alpha,beta)/(lamdaP(0)*lamdaP(1)))*(s0(beta)*Ebasis[alpha]+ a0(alpha)*Rot90*Ebasis[beta]);
                    dCe_ssda0 += (CC(alpha,beta)/(lamdaP(1)*lamdaP(1)))*(s0(alpha)*Rot90*Ebasis[beta] + s0(beta)*Rot90*Ebasis[alpha]);
                }
            }
            // close form
            dCe_aada0x = (1./(lamdaP(0)*lamdaP(0)))*(2*CC(0,0)*a0(0)+2*CC(0,1)*a0(1));
            dCe_aada0y = (1./(lamdaP(0)*lamdaP(0)))*(2*CC(0,1)*a0(0)+2*CC(1,1)*a0(1));
            dCe_asda0x = (1./(lamdaP(0)*lamdaP(1)))*(-1.*CC(0,0)*a0(1)+2*CC(0,1)*a0(0)+CC(1,1)*a0(1));
            dCe_asda0y = (1./(lamdaP(0)*lamdaP(1)))*(-1.*CC(0,0)*a0(0)-2*CC(0,1)*a0(1)+CC(1,1)*a0(0));
            dCe_ssda0x = (1./(lamdaP(1)*lamdaP(1)))*(-2.*CC(0,1)*a0(1)+2*CC(1,1)*a0(0));
            dCe_ssda0y = (1./(lamdaP(1)*lamdaP(1)))*(2*CC(0,0)*a0(1)-2*CC(0,1)*a0(0));
            dCe_aada0(0) = dCe_aada0x;
            dCe_aada0(1) = dCe_aada0y;
            dCe_asda0(0) = dCe_asda0x;
            dCe_asda0(1) = dCe_asda0y;
            dCe_ssda0(0) = dCe_ssda0x;
            dCe_ssda0(1) = dCe_ssda0y;
            // derivatives of the principal stretches wrt a0
            aux00 = sqrt( (Ce_aa-Ce_ss)*(Ce_aa-Ce_ss) + 4*Ce_as*Ce_as);
            if(aux00>1e-7){
                dlamda1da0 = (1./2.)*(dCe_aada0+dCe_ssda0) + (Ce_aa - Ce_ss)*(dCe_aada0 - dCe_ssda0)/aux00/2. + 2*Ce_as*dCe_asda0/aux00;
                dlamda0da0 = (1./2.)*(dCe_aada0+dCe_ssda0) - (Ce_aa - Ce_ss)*(dCe_aada0 - dCe_ssda0)/aux00/2. - 2*Ce_as*dCe_asda0/aux00;
                // Aternatively, do all the chain rule
                dlamda1dCe_aa = 0.5 + 0.5*(Ce_aa - Ce_ss)/aux00;
                dlamda1dCe_ss = 0.5 - 0.5*(Ce_aa - Ce_ss)/aux00;
                dlamda1dCe_as = 2*Ce_as/aux00;
                dlamda0dCe_aa = 0.5 - 0.5*(Ce_aa - Ce_ss)/aux00;
                dlamda0dCe_ss = 0.5 + 0.5*(Ce_aa - Ce_ss)/aux00;
                dlamda0dCe_as = -2.*Ce_as/aux00;
                //
            }else{
                dlamda1da0 = (1./2.)*(dCe_aada0+dCe_ssda0);
                dlamda0da0 = (1./2.)*(dCe_aada0+dCe_ssda0);
                // Aternatively, do all the chain rule
                dlamda1dCe_aa = 0.5 ;
                dlamda1dCe_ss = 0.5 ;
                dlamda1dCe_as = 0;
                dlamda0dCe_aa = 0.5;
                dlamda0dCe_ss = 0.5;
                dlamda0dCe_as = 0;
            }
            dlamda1da0 = dlamda1dCe_aa*dCe_aada0 + dlamda1dCe_as*dCe_asda0 + dlamda1dCe_ss*dCe_ssda0;
            dlamda0da0 = dlamda0dCe_aa*dCe_aada0 + dlamda0dCe_as*dCe_asda0 + dlamda0dCe_ss*dCe_ssda0;
            // derivative of sinVartheta
            if(fabs(lamda1-Ce_aa)<1e-7 || fabs(lamda1-lamda0)<1e-7){
                // derivative is zero
                dsinVarthetada0 = Vector2d(0.,0.);
            }else{
                aux01 = sqrt(Ce_as*Ce_as + (lamda1-Ce_aa)*(lamda1-Ce_aa));
                dsinVarthetada0 = (1./aux01)*(dlamda1da0 - dCe_aada0) - (lamda1-Ce_aa)/(2*aux01*aux01*aux01)*(2*Ce_as*dCe_asda0+2*(lamda1-Ce_aa)*(dlamda1da0-dCe_aada0));
                dsinVarthetadlamda1 = 1./aux01 - (lamda1-Ce_aa)/(2*aux01*aux01*aux01)*(2*(lamda1-Ce_aa));
                // total derivative, ignore the existence of lamda 1 here, think sinVartheta(Ce_aa,Ce_as,Ce_ss), meaning chain rule dude
                dsinVarthetadCe_aa = -1./aux01 + (lamda1-Ce_aa)/(aux01*aux01*aux01)*((lamda1-Ce_aa))   + dsinVarthetadlamda1*dlamda1dCe_aa;
                dsinVarthetadCe_ss = dsinVarthetadlamda1*dlamda1dCe_ss;
                dsinVarthetadCe_as = -(lamda1-Ce_aa)/(aux01*aux01*aux01)*Ce_as + dsinVarthetadlamda1*dlamda1dCe_as;
                //
                dsinVarthetada0 = ((1./aux01) - ((lamda1 - Ce_aa)*(lamda1-Ce_aa)/(aux01*aux01*aux01)))*dlamda1da0 + ((-1./aux01) + ((lamda1 - Ce_aa)*(lamda1 - Ce_aa)/(aux01*aux01*aux01)))*dCe_aada0 - ((lamda1-Ce_aa)/(aux01*aux01*aux01))*Ce_as*dCe_asda0;
            }
            // residual of the Ra0 wrt a0, it is a 2x2 matrix
            domegada0 = ((2.*PIE*phif_dot_plus)/(tau_omega))*(sinVartheta*dlamda1da0+lamda1*dsinVarthetada0);

            // Alternative, do everything with respect to cartesian coordinates for Ra0
            double dlamda1dCe00, dlamda1dCe01, dlamda1dCe11;
            double dlamda0dCe00, dlamda0dCe01, dlamda0dCe11;
            if(fabs(Tr*Tr/4. - Det)<1e-7){
                dlamda1dCe00 = 0.5;
                dlamda1dCe01 = 0.0;
                dlamda1dCe11 = 0.5;
                dlamda0dCe00 = 0.5;
                dlamda0dCe01 = 0.0;
                dlamda0dCe11 = 0.5;
            }else{
                dlamda1dCe00 = (Ce00/4 - Ce11/4)/sqrt(-Ce00*Ce11 + Ce01*Ce01 + Tr*Tr/4) + 0.5;
                dlamda1dCe01 = 2*Ce01/sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr);
                dlamda1dCe11 = -(Ce00/4 - Ce11/4)/sqrt(-Ce00*Ce11 + Ce01*Ce01 + Tr*Tr/4) + 0.5;
                dlamda0dCe00 = -(Ce00/4 - Ce11/4)/sqrt(-Ce00*Ce11 + Ce01*Ce01 + Tr*Tr/4) + 0.5;
                dlamda0dCe01 = -2*Ce01/sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr);
                dlamda0dCe11 = (Ce00/4 - Ce11/4)/sqrt(-Ce00*Ce11 + Ce01*Ce01 + Tr*Tr/4) + 0.5;
            }
            double dCe00da0x = 4.0*C00*a0x*a0x*a0x/(lamdaP_a*lamdaP_a) + 4.0*C00*a0x*a0y*a0y/(lamdaP_a*lamdaP_s) - 6.0*C01*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) + 6.0*C01*a0x*a0x*a0y/(lamdaP_a*lamdaP_a) - 2.0*C01*a0y*a0y*a0y/(lamdaP_s*lamdaP_s) + 2.0*C01*a0y*a0y*a0y/(lamdaP_a*lamdaP_s) + 2.0*C11*a0x*a0y*a0y/(lamdaP_s*lamdaP_s) - 4.0*C11*a0x*a0y*a0y/(lamdaP_a*lamdaP_s) + 2.0*C11*a0x*a0y*a0y/(lamdaP_a*lamdaP_a);
            double dCe00da0y = 4.0*C00*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) + 4.0*C00*a0y*a0y*a0y/(lamdaP_s*lamdaP_s) - 2.0*C01*a0x*a0x*a0x/(lamdaP_a*lamdaP_s) + 2.0*C01*a0x*a0x*a0x/(lamdaP_a*lamdaP_a) - 6.0*C01*a0x*a0y*a0y/(lamdaP_s*lamdaP_s) + 6.0*C01*a0x*a0y*a0y/(lamdaP_a*lamdaP_s) + 2.0*C11*a0x*a0x*a0y/(lamdaP_s*lamdaP_s) - 4.0*C11*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) + 2.0*C11*a0x*a0x*a0y/(lamdaP_a*lamdaP_a);
            double dCe01da0x = (1.0*a0x*a0y*(lamdaP_a - lamdaP_s)*(1.0*C01*a0y*(lamdaP_a - lamdaP_s) - 2.0*C11*a0x*lamdaP_a) - 2.0*a0x*lamdaP_s*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) + 1.0*a0y*(lamdaP_a - lamdaP_s)*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - 1.0*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a)*(1.0*C00*a0y*(lamdaP_a - lamdaP_s) - 2.0*C01*a0x*lamdaP_a))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
            double dCe01da0y = (1.0*a0x*a0y*(lamdaP_a - lamdaP_s)*(1.0*C01*a0x*(lamdaP_a - lamdaP_s) - 2.0*C11*a0y*lamdaP_s) + 1.0*a0x*(lamdaP_a - lamdaP_s)*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - 2.0*a0y*lamdaP_a*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - 1.0*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a)*(1.0*C00*a0x*(lamdaP_a - lamdaP_s) - 2.0*C01*a0y*lamdaP_s))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
            double dCe11da0x = 2.0*C00*a0x*a0y*a0y/(lamdaP_s*lamdaP_s) - 4.0*C00*a0x*a0y*a0y/(lamdaP_a*lamdaP_s) + 2.0*C00*a0x*a0y*a0y/(lamdaP_a*lamdaP_a) - 6.0*C01*a0x*a0x*a0y/(lamdaP_s*lamdaP_s) + 6.0*C01*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) - 2.0*C01*a0y*a0y*a0y/(lamdaP_a*lamdaP_s) + 2.0*C01*a0y*a0y*a0y/(lamdaP_a*lamdaP_a) + 4.0*C11*a0x*a0x*a0x/(lamdaP_s*lamdaP_s) + 4.0*C11*a0x*a0y*a0y/(lamdaP_a*lamdaP_s);
            double dCe11da0y = 2.0*C00*a0x*a0x*a0y/(lamdaP_s*lamdaP_s) - 4.0*C00*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) + 2.0*C00*a0x*a0x*a0y/(lamdaP_a*lamdaP_a) - 2.0*C01*a0x*a0x*a0x/(lamdaP_s*lamdaP_s) + 2.0*C01*a0x*a0x*a0x/(lamdaP_a*lamdaP_s) - 6.0*C01*a0x*a0y*a0y/(lamdaP_a*lamdaP_s) + 6.0*C01*a0x*a0y*a0y/(lamdaP_a*lamdaP_a) + 4.0*C11*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) + 4.0*C11*a0y*a0y*a0y/(lamdaP_a*lamdaP_a);
            //std::cout<<"derivatives dlamda1dCe\n"<<dlamda1dCe00<<", "<<dlamda1dCe01<<", "<<dlamda1dCe11<<"\n";
            //std::cout<<"derivatives dlamda0dCe\n"<<dlamda0dCe00<<", "<<dlamda0dCe01<<", "<<dlamda0dCe11<<"\n";
            //std::cout<<"derivatives dCeda0x\n"<<dCe00da0x<<", "<<dCe01da0x<<", "<<dCe11da0x<<"\n";
            //std::cout<<"derivatives dCeda0y\n"<<dCe00da0y<<", "<<dCe01da0y<<", "<<dCe11da0y<<"\n";
            //double dDetda0x = 1.0*a0x*(8.0*C00*C11*pow(a0x,6) + 24.0*C00*C11*pow(a0x,4)*a0y*a0y + 24.0*C00*C11*a0x*a0x*pow(a0y,4) + 8.0*C00*C11*pow(a0y,6) - 8.0*C01*C01*pow(a0x,6) - 24.0*C01*C01*pow(a0x,4)*a0y*a0y - 24.0*C01*C01*a0x*a0x*pow(a0y,4) - 8.0*C01*C01*pow(a0y,6))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
            //double dDetda0y = 1.0*a0y*(8.0*C00*C11*pow(a0x,6) + 24.0*C00*C11*pow(a0x,4)*a0y*a0y + 24.0*C00*C11*a0x*a0x*pow(a0y,4) + 8.0*C00*C11*pow(a0y,6) - 8.0*C01*C01*pow(a0x,6) - 24.0*C01*C01*pow(a0x,4)*a0y*a0y - 24.0*C01*C01*a0x*a0x*pow(a0y,4) - 8.0*C01*C01*pow(a0y,6))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);

            //dlamda1da0(0) = dlamda1dCe00*dCe00da0x + dlamda1dCe01*dCe01da0x + dlamda1dCe11*dCe11da0x;
            //dlamda1da0(1) = dlamda1dCe00*dCe00da0y + dlamda1dCe01*dCe01da0y + dlamda1dCe11*dCe11da0y;
            //dlamda0da0(0) = dlamda0dCe00*dCe00da0x + dlamda0dCe01*dCe01da0x + dlamda0dCe11*dCe11da0x;
            //dlamda0da0(1) = dlamda0dCe00*dCe00da0y + dlamda0dCe01*dCe01da0y + dlamda0dCe11*dCe11da0y;

            //std::cout<<"dlamda1da0 =\n"<<dlamda1da0<<"\n";
            //std::cout<<"dlamda0da0 =\n"<<dlamda0da0<<"\n";
            //
            double dsinVarthetada0x_exp, dsinVarthetada0y_exp, dsinVarthetadCe00_exp, dsinVarthetadCe01_exp, dsinVarthetadCe11_exp;
            if(fabs(lamda1-lamda0)<1e-7){
                // equal eigenvalues, no way to reorient
                dsinVarthetada0x_exp =0.0;
                dsinVarthetada0y_exp =0.0;
                dsinVarthetadCe00_exp =0.0;
                dsinVarthetadCe01_exp =0.0;
                dsinVarthetadCe11_exp =0.0;
            }else if(fabs(Ce01)<1e-7 && fabs(Ce00-lamda1)>1e-7){
                dsinVarthetada0x_exp = -1;
                dsinVarthetada0y_exp = 2*Ce01/(-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
                dsinVarthetadCe00_exp = Ce01*a0y*(-2*Ce00 + 2*Ce11 + 2.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/(pow((-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2)*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
                dsinVarthetadCe01_exp = 2*a0y*(-4*Ce01*Ce01 + (-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/(pow((-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2)*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
                dsinVarthetadCe11_exp = Ce01*a0y*(2*Ce00 - 2*Ce11 - 2.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/(pow((-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2)*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
            }else{
                dsinVarthetada0x_exp = 2*Ce01/sqrt(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) );
                dsinVarthetada0y_exp = (-Ce00 + Ce11 - sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/sqrt(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) );
                dsinVarthetadCe00_exp = (-2*a0y*(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) )*(2*Ce00 - 2*Ce11 + 2.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)) + (-2*Ce01*a0x + a0y*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)))*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))*(4*Ce00 - 4*Ce11 + 4.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)))/(4*pow((4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) ),(3/2) )*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
                dsinVarthetadCe01_exp = 2*(-2*Ce01*a0y*(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) )*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr) + 2*Ce01*(-2*Ce01*a0x + a0y*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)))*(Ce00 - 1.0*Ce11 + 2*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr) + a0x*(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) )*(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/(pow((4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) ),(3/2) )*(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
                dsinVarthetadCe11_exp = (2*a0y*(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) )*(2*Ce00 - 2*Ce11 + 2.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)) + (2*Ce01*a0x - a0y*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)))*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))*(4*Ce00 - 4*Ce11 + 4.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)))/(4*pow((4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) ),(3/2))*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
            }
            //
            //dsinVarthetada0(0) = dsinVarthetada0x_exp + dsinVarthetadCe00_exp*dCe00da0x + dsinVarthetadCe01_exp*dCe01da0x + dsinVarthetadCe11_exp*dCe11da0x;
            //dsinVarthetada0(1) = dsinVarthetada0y_exp + dsinVarthetadCe00_exp*dCe00da0y + dsinVarthetadCe01_exp*dCe01da0y + dsinVarthetadCe11_exp*dCe11da0y;
            //std::cout<<"dsinVarthetada0 = \n"<<dsinVarthetada0<<"\n";
            //
            //domegada0 = ((2.*PIE*phif_dot_plus)/(tau_omega))*(sinVartheta*dlamda1da0+lamda1*dsinVarthetada0);
            //domegada0 = -((2.*PIE*phif_dot_plus)/(tau_omega))*(sinVartheta*sinVartheta*dlamda1da0+2*lamda1*sinVartheta*dsinVarthetada0);
            //std::cout<<"domegada0 = \n"<<domegada0<<"\n";
            //
            dRa0da0 = 	Identity + (-dRomegadomega*a0_0)*domegada0.transpose();


            // derivative of Ra0 wrt the dispersion is zero.
            // derivative of Ra0 wrt lamdaP requires some calculations
            dCe_aadlamdaP.setZero();
            dCe_asdlamdaP.setZero();
            dCe_ssdlamdaP.setZero();
            for(int alpha=0; alpha<2; alpha++){
                for(int beta=0; beta<2; beta++){
                    dCe_aadlamdaP(0) += -2*CC(alpha,beta)*(a0(alpha)*a0(beta))/(lamdaP(0)*lamdaP(0)*lamdaP(0));
                    dCe_asdlamdaP(0) += -CC(alpha,beta)*(s0(beta)*a0(alpha))/(lamdaP(1)*lamdaP(0)*lamdaP(0));
                    dCe_asdlamdaP(1) += -CC(alpha,beta)*(s0(beta)*a0(alpha))/(lamdaP(0)*lamdaP(1)*lamdaP(1));
                    dCe_ssdlamdaP(1) += -2*CC(alpha,beta)*(s0(alpha)*s0(beta))/(lamdaP(1)*lamdaP(1)*lamdaP(1));
                }
            }
            // closed form
            dCe_aadlamdaP(0) = -(2.0*CC(0,0)*a0x*a0x + 4.0*CC(0,1)*a0x*a0y + 2.0*CC(1,1)*a0y*a0y)/pow(lamdaP_a,3);
            dCe_aadlamdaP(1) = 0.;
            dCe_asdlamdaP(0) = (CC(0,0)*a0x*a0y - CC(0,1)*a0x*a0x + CC(0,1)*a0y*a0y - CC(1,1)*a0x*a0y)/(lamdaP_a*lamdaP_a*lamdaP_s);
            dCe_asdlamdaP(1) = (CC(0,0)*a0x*a0y - CC(0,1)*a0x*a0x + CC(0,1)*a0y*a0y - CC(1,1)*a0x*a0y)/(lamdaP_a*lamdaP_s*lamdaP_s);
            dCe_ssdlamdaP(0) = 0.;
            dCe_ssdlamdaP(1) = (-2.0*CC(0,0)*a0y*a0y + 4.0*CC(0,1)*a0x*a0y - 2.0*CC(1,1)*a0x*a0x)/pow(lamdaP_s,3);
            // derivative of principal stretches wrt lampdaP
            if(aux00>1e-7){
                dlamda1dlamdaP = (1./2.)*(dCe_aadlamdaP + dCe_ssdlamdaP)+(1./(4*aux00))*(2.*(Ce_aa-Ce_ss)*(dCe_aadlamdaP - dCe_ssdlamdaP)+8.*Ce_as*dCe_asdlamdaP);
                dlamda0dlamdaP = (1./2.)*(dCe_aadlamdaP + dCe_ssdlamdaP)-(1./(4*aux00))*(2.*(Ce_aa-Ce_ss)*(dCe_aadlamdaP - dCe_ssdlamdaP)+8.*Ce_as*dCe_asdlamdaP);
            }else{
                dlamda1dlamdaP = (1./2.)*(dCe_aadlamdaP + dCe_ssdlamdaP);
                dlamda0dlamdaP = (1./2.)*(dCe_aadlamdaP + dCe_ssdlamdaP);
            }
            dlamda1dlamdaP = dlamda1dCe_aa*dCe_aadlamdaP + dlamda1dCe_as*dCe_asdlamdaP + dlamda1dCe_ss*dCe_ssdlamdaP;
            dlamda0dlamdaP = dlamda0dCe_aa*dCe_aadlamdaP + dlamda0dCe_as*dCe_asdlamdaP + dlamda0dCe_ss*dCe_ssdlamdaP;
            // derivative of sinvartheta wrt lamdaP
            //dsinVarthetadlamdaP = dsinVarthetadCe_aa*dCe_aadlamdaP + dsinVarthetadCe_as*dCe_asdlamdaP + dsinVarthetadCe_ss*dCe_ssdlamdaP;
            if(fabs(lamda1-Ce_aa)<1e-7 || fabs(lamda1-lamda0)<1e-7){
                // derivative is zero
                dsinVarthetadlamdaP = Vector2d(0.,0.);
            }else{
                dsinVarthetadlamdaP = ((1./aux01) - ((lamda1 - Ce_aa)/(aux01*aux01*aux01)))*dlamda1dlamdaP + ((-1./aux01) + ((lamda1 - Ce_aa)/(aux01*aux01*aux01)))*dCe_aadlamdaP - (1./(aux01*aux01*aux01))*Ce_as*dCe_asdlamdaP;
            }
            // Alternative, do everything with respect to original cartesian coordinates
            double dCe00dlamdaPa = -1.0*a0x*(2.0*C00*pow(a0x,3)*lamdaP_s + 2.0*C00*a0x*a0y*a0y*lamdaP_a - 2.0*C01*a0x*a0x*a0y*lamdaP_a + 4.0*C01*a0x*a0x*a0y*lamdaP_s + 2.0*C01*pow(a0y,3)*lamdaP_a - 2.0*C11*a0x*a0y*a0y*lamdaP_a + 2.0*C11*a0x*a0y*a0y*lamdaP_s)/(pow(lamdaP_a,3)*lamdaP_s);
            double dCe00dlamdaPs = -1.0*a0y*(2.0*C00*a0x*a0x*a0y*lamdaP_s + 2.0*C00*pow(a0y,3)*lamdaP_a - 2.0*C01*pow(a0x,3)*lamdaP_s - 4.0*C01*a0x*a0y*a0y*lamdaP_a + 2.0*C01*a0x*a0y*a0y*lamdaP_s + 2.0*C11*a0x*a0x*a0y*lamdaP_a - 2.0*C11*a0x*a0x*a0y*lamdaP_s)/(lamdaP_a*pow(lamdaP_s,3));
            double dCe01dlamdaPa = 1.0*(a0x*a0x*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) + a0x*a0y*a0y*(lamdaP_a - lamdaP_s)*(C01*a0x + C11*a0y) + a0x*a0y*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - a0y*(C00*a0x + C01*a0y)*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a))/(pow(lamdaP_a,3)*lamdaP_s);
            double dCe01dlamdaPs = 1.0*(-a0x*a0x*a0y*(lamdaP_a - lamdaP_s)*(C01*a0y - C11*a0x) - a0x*a0y*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) + a0x*(C00*a0y - C01*a0x)*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a) + a0y*a0y*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)))/(lamdaP_a*pow(lamdaP_s,3));
            double dCe11dlamdaPa = 1.0*a0y*(2.0*C00*a0x*a0x*a0y*lamdaP_a - 2.0*C00*a0x*a0x*a0y*lamdaP_s - 2.0*C01*pow(a0x,3)*lamdaP_a + 2.0*C01*a0x*a0y*a0y*lamdaP_a - 4.0*C01*a0x*a0y*a0y*lamdaP_s - 2.0*C11*a0x*a0x*a0y*lamdaP_a - 2.0*C11*pow(a0y,3)*lamdaP_s)/(pow(lamdaP_a,3)*lamdaP_s);
            double dCe11dlamdaPs = -1.0*a0x*(2.0*C00*a0x*a0y*a0y*lamdaP_a - 2.0*C00*a0x*a0y*a0y*lamdaP_s - 4.0*C01*a0x*a0x*a0y*lamdaP_a + 2.0*C01*a0x*a0x*a0y*lamdaP_s - 2.0*C01*pow(a0y,3)*lamdaP_s + 2.0*C11*pow(a0x,3)*lamdaP_a + 2.0*C11*a0x*a0y*a0y*lamdaP_s)/(lamdaP_a*pow(lamdaP_s,3));
            //double dDetdlamdaPa = (-2.0*C00*C11*pow(a0x,8) - 8.0*C00*C11*pow(a0x,6)*a0y*a0y - 12.0*C00*C11*pow(a0x,4)*pow(a0y,4) - 8.0*C00*C11*a0x*a0x*pow(a0y,6) - 2.0*C00*C11*pow(a0y,8) + 2.0*C01*C01*pow(a0x,8) + 8.0*C01*C01*pow(a0x,6)*a0y*a0y + 12.0*C01*C01*pow(a0x,4)*pow(a0y,4) + 8.0*C01*C01*a0x*a0x*pow(a0y,6) + 2.0*C01*C01*pow(a0y,8) )/(pow(lamdaP_a,3)*lamdaP_s*lamdaP_s);
            //double dDetdlamdaPs = (-2.0*C00*C11*pow(a0x,8) - 8.0*C00*C11*pow(a0x,6)*a0y*a0y - 12.0*C00*C11*pow(a0x,4)*pow(a0y,4) - 8.0*C00*C11*a0x*a0x*pow(a0y,6) - 2.0*C00*C11*pow(a0y,8) + 2.0*C01*C01*pow(a0x,8) + 8.0*C01*C01*pow(a0x,6)*a0y*a0y + 12.0*C01*C01*pow(a0x,4)*pow(a0y,4) + 8.0*C01*C01*a0x*a0x*pow(a0y,6) + 2.0*C01*C01*pow(a0y,8) )/(lamdaP_a*lamdaP_a*pow(lamdaP_s,3));
            //

            //dlamda1dlamdaP(0) = dlamda1dCe00*dCe00dlamdaPa + dlamda1dCe01*dCe01dlamdaPa +  dlamda1dCe11*dCe11dlamdaPa;
            //dlamda1dlamdaP(1) = dlamda1dCe00*dCe00dlamdaPs + dlamda1dCe01*dCe01dlamdaPs +  dlamda1dCe11*dCe11dlamdaPs;
            //dlamda0dlamdaP(0) = dlamda0dCe00*dCe00dlamdaPa + dlamda0dCe01*dCe01dlamdaPa +  dlamda0dCe11*dCe11dlamdaPa;
            //dlamda0dlamdaP(1) = dlamda0dCe00*dCe00dlamdaPs + dlamda0dCe01*dCe01dlamdaPs +  dlamda0dCe11*dCe11dlamdaPs;

            //std::cout<<"dlamda1dlamdaP =\n"<<dlamda1dlamdaP<<"\n";
            //std::cout<<"dlamda0dlamdaP =\n"<<dlamda0dlamdaP<<"\n";

            //dsinVarthetadlamdaP(0) =  dsinVarthetadCe00_exp*dCe00dlamdaPa + dsinVarthetadCe01_exp*dCe01dlamdaPa + dsinVarthetadCe11_exp*dCe11dlamdaPa;
            //dsinVarthetadlamdaP(1) =  dsinVarthetadCe00_exp*dCe00dlamdaPs + dsinVarthetadCe01_exp*dCe01dlamdaPs + dsinVarthetadCe11_exp*dCe11dlamdaPs;

            //std::cout<<"dsinVarthetadlamdaP = \n"<<dsinVarthetadlamdaP<<"\n";
            // derivative of omega wrt lamdaP
            domegadlamdaP = (2*PIE*phif_dot_plus/tau_omega)*(lamda1*dsinVarthetadlamdaP + sinVartheta*dlamda1dlamdaP)+((2.*PIE*dphifplusdlamdaP)/(tau_omega))*lamda1*sinVartheta;
            //domegadlamdaP = -(2*PIE*phif_dot_plus/tau_omega)*(2*lamda1*sinVartheta*dsinVarthetadlamdaP + sinVartheta*sinVartheta*dlamda1dlamdaP)+((-2.*PIE*dphifplusdlamdaP)/(tau_omega))*lamda1*sinVartheta*sinVartheta;
            //std::cout<<"domegadlamdaP = \n"<<domegadlamdaP<<"\n";
            // and finally, derivative of Ra0 wrt lamdaP
            dRa0dlamdaP = (dRomegadomega*a0_0)*domegadlamdaP.transpose();

            // Tangent of dispersion
            dRkappadphif = (1./tau_kappa)*(pow(lamda0/lamda1,gamma_kappa)/3.-kappa)*dphifplusdphif;
            dRkappada0 =(phif_dot_plus/tau_kappa)*((gamma_kappa/3.)*pow(lamda0/lamda1,gamma_kappa-1))*((1./lamda1)*dlamda0da0-(lamda0/(lamda1*lamda1))*dlamda1da0);
            dRkappadkappa = -1/local_dt - (phif_dot_plus/tau_kappa);
            dRkappadlamdaP = (phif_dot_plus/tau_kappa)*((gamma_kappa/3.)*pow(lamda0/lamda1,gamma_kappa-1))*((1./lamda1)*dlamda0dlamdaP-(lamda0/(lamda1*lamda1))*dlamda1dlamdaP)+(dphifplusdlamdaP/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/3.  -kappa);

            // Tangent of lamdaP
            // derivative wrt phif
            dRlamdaPdphif.setZero();
            dRlamdaPdphif(0) = ((lamdaE_a-1.)/tau_lamdaP_a)*dphifplusdphif;
            dRlamdaPdphif(1) = ((lamdaE_s-1.)/tau_lamdaP_s)*dphifplusdphif;
            // derivative wrt fiber direction
            dRlamdaPada0 = (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a*lamdaE_a))*dCe_aada0;
            dRlamdaPsda0 = (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s*lamdaE_s))*dCe_ssda0;
            // no dependence on the fiber dispersion
            // derivative wrt the lamdaP
            dRlamdaPadlamdaPa = -1./local_dt + (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a*lamdaE_a))*dCe_aadlamdaP(0)+dphifplusdlamdaP(0)*(lamdaE_a-1)/tau_lamdaP_a;
            dRlamdaPadlamdaPs = 		    (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a*lamdaE_a))*dCe_aadlamdaP(1)+dphifplusdlamdaP(1)*(lamdaE_a-1)/tau_lamdaP_a;
            dRlamdaPsdlamdaPa = 		    (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s*lamdaE_s))*dCe_ssdlamdaP(0)+dphifplusdlamdaP(0)*(lamdaE_s-1)/tau_lamdaP_s;
            dRlamdaPsdlamdaPs = -1./local_dt + (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s*lamdaE_s))*dCe_ssdlamdaP(1)+dphifplusdlamdaP(1)*(lamdaE_s-1)/tau_lamdaP_s;

            // Assemble into the tangent matrix.
            // phif
            KK_local(0,0) = dRphifdphif;  KK_local(0,4) = dRphifdlamdaP_a;  KK_local(0,5) = dRphifdlamdaP_s;
            // a0
            KK_local(1,0) = dRa0dphif(0); KK_local(1,1) = dRa0da0(0,0);KK_local(1,2) = dRa0da0(0,1); KK_local(1,4) = dRa0dlamdaP(0,0);KK_local(1,5) = dRa0dlamdaP(0,1);
            KK_local(2,0) = dRa0dphif(1); KK_local(2,1) = dRa0da0(1,0);KK_local(2,2) = dRa0da0(1,1); KK_local(2,4) = dRa0dlamdaP(1,0);KK_local(2,5) = dRa0dlamdaP(1,1);
            // kappa
            KK_local(3,0) = dRkappadphif; KK_local(3,1) = dRkappada0(0);KK_local(3,2) = dRkappada0(1);KK_local(3,3) = dRkappadkappa; KK_local(3,4) = dRkappadlamdaP(0);KK_local(3,5) = dRkappadlamdaP(1);
            // lamdaP
            KK_local(4,0) = dRlamdaPdphif(0);KK_local(4,1) = dRlamdaPada0(0);KK_local(4,2) = dRlamdaPada0(1); KK_local(4,4) =dRlamdaPadlamdaPa; KK_local(4,5) =dRlamdaPadlamdaPs;
            KK_local(5,0) = dRlamdaPdphif(1);KK_local(5,1) = dRlamdaPsda0(0);KK_local(5,2) = dRlamdaPsda0(1); KK_local(5,4) =dRlamdaPsdlamdaPa; KK_local(5,5) =dRlamdaPsdlamdaPs;

            //----------//
            // SOLVE
            //----------//

            //std::cout<<"SOLVE.\nRR_local\n"<<RR_local<<"\nKK_local\n"<<KK_local<<"\n";

            double normRR = sqrt(RR_local.dot(RR_local));
            residuum = normRR;
            // solve
            SOL_local = KK_local.lu().solve(-RR_local);

            //std::cout<<"SOL_local\n"<<SOL_local<<"\n";
            // update the solution
            double normSOL = sqrt(SOL_local.dot(SOL_local));
            phif += SOL_local(0);
            a0(0) += SOL_local(1);
            a0(1) += SOL_local(2);
            kappa += SOL_local(3);
            lamdaP(0) += SOL_local(4);
            lamdaP(1) += SOL_local(5);
            // normalize a0
            a0 = a0/sqrt(a0.dot(a0));
            //std::cout<<"norm(RR): "<<residuum<<"\n";
            //std::cout<<"norm(SOL): "<<normSOL<<"\n";
            iter += 1;
            //std::cout<<"Finish local Newton.\niter = "<<iter<<", residuum = "<<residuum<<"\n";
            //std::cout << "\nphif: " << phif << ", kappa: " << kappa << ", lamdaP:" << lamdaP(0) << "," << lamdaP(1)
            //          << ",a0:" << a0(0) << "," << a0(1) << "\n";
            if(normRR > tol_local && iter == max_local_iter){
                //std::cout<<"no local convergence\n";
                //std::cout<<"\nlamda1: "<<lamda1<<", lamda0: "<<lamda0<<", lamdaP:"<<lamdaP(0)<<","<<lamdaP(1)<<",a0:"<<a0(0)<<","<<a0(1)<<",Ce: "<<Ce_aa<<","<<Ce_as<<","<<Ce_ss<<"\n";
                //std::cout<<"Ce-lamda:"<<fabs(lamda1-Ce_aa)<<"\n";
                //std::cout<<"aux"<<aux00<<"\n";
                //std::cout<<"sinVartheta: "<<sinVartheta<<"\n";
                //std::cout<<"Res\n"<<RR_local<<"\nSOL_local\n"<<SOL_local<<"\n";
                //throw std::runtime_error("sorry pal ");
            }

            if (step == time_step_ratio-1){
                // RECALCULATE KK FOR COMPLETE GLOBAL TIME STEP
                //KK_local.setZero();

                dRphifdphif = -1./dt + dphifplusdphif - d_phi;
                //std::cout<<"Collagen fraction tangent.\n";
                //std::cout<<"dphifplusdphif = "<<dphifplusdphif<<"\n";
                //std::cout<<"dRphifdphif = "<<dRphifdphif<<"\n";

                // Tangent of a0
                // derivative of the rotation matrix wrt omega
                dRomegadomega(0,0) = -sin(omega*dt)*dt; dRomegadomega(0,1) = -cos(omega*dt)*dt;
                dRomegadomega(1,0) = cos(omega*dt)*dt;  dRomegadomega(1,1)= -sin(omega*dt)*dt;
                // derivative of the omega angular velocity wrt phif
                domegadphif = (2.*PIE*dphifplusdphif)*lamda1*sinVartheta/(tau_omega);
                //domegadphif = -(2.*PIE*dphifplusdphif)*lamda1*sinVartheta*sinVartheta/(tau_omega);
                // chain rule for derivative of residual wrt phif
                dRa0dphif = (-dRomegadomega*a0_00)*domegadphif;

                // derivative of R_a0 wrt to a0 needs some pre-calculations
                // residual of the Ra0 wrt a0, it is a 2x2 matrix
                domegada0 = ((2.*PIE*phif_dot_plus)/(tau_omega))*(sinVartheta*dlamda1da0+lamda1*dsinVarthetada0);
                dRa0da0 = 	Identity + (-dRomegadomega*a0_00)*domegada0.transpose();

                // derivative of Ra0 wrt the dispersion is zero.
                // derivative of Ra0 wrt lamdaP requires some calculations

                // derivative of omega wrt lamdaP
                domegadlamdaP = (2*PIE*phif_dot_plus/tau_omega)*(lamda1*dsinVarthetadlamdaP + sinVartheta*dlamda1dlamdaP)+((2.*PIE*dphifplusdlamdaP)/(tau_omega))*lamda1*sinVartheta;
                //domegadlamdaP = -(2*PIE*phif_dot_plus/tau_omega)*(2*lamda1*sinVartheta*dsinVarthetadlamdaP + sinVartheta*sinVartheta*dlamda1dlamdaP)+((-2.*PIE*dphifplusdlamdaP)/(tau_omega))*lamda1*sinVartheta*sinVartheta;
                //std::cout<<"domegadlamdaP = \n"<<domegadlamdaP<<"\n";
                // and finally, derivative of Ra0 wrt lamdaP
                dRa0dlamdaP = (dRomegadomega*a0_00)*domegadlamdaP.transpose();

                // Tangent of dispersion
                dRkappadphif = (1./tau_kappa)*(pow(lamda0/lamda1,gamma_kappa)/3.-kappa)*dphifplusdphif;
                dRkappada0 =(phif_dot_plus/tau_kappa)*((gamma_kappa/3.)*pow(lamda0/lamda1,gamma_kappa-1))*((1./lamda1)*dlamda0da0-(lamda0/(lamda1*lamda1))*dlamda1da0);
                dRkappadkappa = -1/dt - (phif_dot_plus/tau_kappa);
                dRkappadlamdaP = (phif_dot_plus/tau_kappa)*((gamma_kappa/3.)*pow(lamda0/lamda1,gamma_kappa-1))*((1./lamda1)*dlamda0dlamdaP-(lamda0/(lamda1*lamda1))*dlamda1dlamdaP)+(dphifplusdlamdaP/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/3.  -kappa);

                // Tangent of lamdaP
                // derivative wrt phif
                dRlamdaPdphif.setZero();
                dRlamdaPdphif(0) = ((lamdaE_a-1.)/tau_lamdaP_a)*dphifplusdphif;
                dRlamdaPdphif(1) = ((lamdaE_s-1.)/tau_lamdaP_s)*dphifplusdphif;
                // derivative wrt fiber direction
                dRlamdaPada0 = (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a*lamdaE_a))*dCe_aada0;
                dRlamdaPsda0 = (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s*lamdaE_s))*dCe_ssda0;
                // no dependence on the fiber dispersion
                // derivative wrt the lamdaP
                dRlamdaPadlamdaPa = -1./dt + (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a*lamdaE_a))*dCe_aadlamdaP(0)+dphifplusdlamdaP(0)*(lamdaE_a-1)/tau_lamdaP_a;
                dRlamdaPadlamdaPs = 		    (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a*lamdaE_a))*dCe_aadlamdaP(1)+dphifplusdlamdaP(1)*(lamdaE_a-1)/tau_lamdaP_a;
                dRlamdaPsdlamdaPa = 		    (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s*lamdaE_s))*dCe_ssdlamdaP(0)+dphifplusdlamdaP(0)*(lamdaE_s-1)/tau_lamdaP_s;
                dRlamdaPsdlamdaPs = -1./dt + (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s*lamdaE_s))*dCe_ssdlamdaP(1)+dphifplusdlamdaP(1)*(lamdaE_s-1)/tau_lamdaP_s;

                // Assemble into the tangent matrix.
                // phif
                KK_local(0,0) = dRphifdphif;  KK_local(0,4) = dRphifdlamdaP_a;  KK_local(0,5) = dRphifdlamdaP_s;
                // a0
                KK_local(1,0) = dRa0dphif(0); KK_local(1,1) = dRa0da0(0,0);KK_local(1,2) = dRa0da0(0,1); KK_local(1,4) = dRa0dlamdaP(0,0);KK_local(1,5) = dRa0dlamdaP(0,1);
                KK_local(2,0) = dRa0dphif(1); KK_local(2,1) = dRa0da0(1,0);KK_local(2,2) = dRa0da0(1,1); KK_local(2,4) = dRa0dlamdaP(1,0);KK_local(2,5) = dRa0dlamdaP(1,1);
                // kappa
                KK_local(3,0) = dRkappadphif; KK_local(3,1) = dRkappada0(0);KK_local(3,2) = dRkappada0(1);KK_local(3,3) = dRkappadkappa; KK_local(3,4) = dRkappadlamdaP(0);KK_local(3,5) = dRkappadlamdaP(1);
                // lamdaP
                KK_local(4,0) = dRlamdaPdphif(0);KK_local(4,1) = dRlamdaPada0(0);KK_local(4,2) = dRlamdaPada0(1); KK_local(4,4) =dRlamdaPadlamdaPa; KK_local(4,5) =dRlamdaPadlamdaPs;
                KK_local(5,0) = dRlamdaPdphif(1);KK_local(5,1) = dRlamdaPsda0(0);KK_local(5,2) = dRlamdaPsda0(1); KK_local(5,4) =dRlamdaPsdlamdaPa; KK_local(5,5) =dRlamdaPsdlamdaPs;
            }

        } // END OF WHILE LOOP OF LOCAL NEWTON

        // UPDATE INITIAL GUESS IF DOING MULTISTAGE
        phif_0 = phif;
        // make sure it is unit length
        a0_0 = a0/(sqrt(a0.dot(a0)));
        kappa_0 = kappa;
        lamdaP_0 = lamdaP;
    }
    //myfile.close();
    //std::cout<<"Finish local Newton.\niter = "<<iter<<", residuum = "<<residuum<<"\n";
    //std::cout<<"lamda1: "<<lamda1<<", lamda0: "<<lamda0<<", lamdaP:"<<lamdaP(0)<<","<<lamdaP(1)<<",a0:"<<a0(0)<<","<<a0(1)<<"Ce_aa: "<<Ce_aa<<","<<Ce_as<<","<<Ce_ss<<"\n";
    //std::cout<<"Ce-lamda:"<<fabs(lamda1-Ce_aa)<<"\n";
    //std::cout<<"sinVartheta: "<<sinVartheta<<"\n";
    a03d(0) = a0(0);
    a03d(1) = a0(1);
    s0 = Rot90*a0;
    s03d(0) = s0(0);
    s03d(1) = s0(1);
    lamdaP3d(0) = lamdaP(0);
    lamdaP3d(1) = lamdaP(1);

    //-----------------------------------//
    // WOUND TANGENTS FOR GLOBAL PROBLEM
    //-----------------------------------//

    //----------//
    // MECHANICs
    //----------//

    // explicit derivatives of Ce wrt CC
    Matrix2d dCe_aadCC;dCe_aadCC.setZero();
    Matrix2d dCe_ssdCC;dCe_ssdCC.setZero();
    Matrix2d dCe_asdCC;dCe_asdCC.setZero();
    for(int coordi=0;coordi<2;coordi++){
        for(int coordj=0;coordj<2;coordj++){
            dCe_aadCC(coordi,coordj) += a0(coordi)*a0(coordj)/(lamdaP(0)*lamdaP(0));
            dCe_ssdCC(coordi,coordj) += s0(coordi)*s0(coordj)/(lamdaP(1)*lamdaP(1));
            dCe_asdCC(coordi,coordj) += a0(coordi)*s0(coordj)/(lamdaP(0)*lamdaP(1));
        }
    }

    // Explicit derivatives of residuals wrt CC

    // phif
    Matrix2d dRphifdCC;dRphifdCC.setZero();
    //double dH_thetadtheta = -1.*H_theta*H_theta*exp(-gamma_theta*(theta_e-vartheta_e))*(-gamma_theta/(lamdaP(0)*lamdaP(1)));
    //Matrix2d dthetadCC = (1./2)*theta*CCinv;
    Matrix2d dHedCC_explicit = -1./pow((1.+exp(-gamma_theta*(Je - vartheta_e))),2)*(exp(-gamma_theta*(Je - vartheta_e)))*(-gamma_theta)*(J*CCinv/(2*Jp));
    dRphifdCC = p_phi_theta*dHedCC_explicit*(rho/(K_phi_rho+phif));
    Matrix2d dphifdotplusdCC = p_phi_theta*dHedCC_explicit*(rho/(K_phi_rho+phif));

    // a0
    // preprocessing
    Matrix2d dlamda1dCC;
    Matrix2d dlamda0dCC;
    dlamda1dCC = dlamda1dCe_aa*dCe_aadCC + dlamda1dCe_ss*dCe_ssdCC + dlamda1dCe_as*dCe_asdCC;
    dlamda0dCC = dlamda0dCe_aa*dCe_aadCC + dlamda0dCe_ss*dCe_ssdCC + dlamda0dCe_as*dCe_asdCC;
    Matrix2d dsinVarthetadCC;
    dsinVarthetadCC = dsinVarthetadCe_aa*dCe_aadCC+dsinVarthetadCe_ss*dCe_ssdCC+dsinVarthetadCe_as*dCe_asdCC;
    Matrix2d domegadCC;
    domegadCC = (2*PIE*phif_dot_plus/tau_omega)*(lamda1*dsinVarthetadCC + sinVartheta*dlamda1dCC)+ ((2.*PIE*dphifdotplusdCC)/(tau_omega))*lamda1*sinVartheta;
    // a0x
    Matrix2d dRa0xdCC;dRa0xdCC.setZero();
    dRa0xdCC = (-dRomegadomega *a0_00)(0)*domegadCC;
    // a0y
    Matrix2d dRa0ydCC;dRa0ydCC.setZero();
    dRa0ydCC = (-dRomegadomega *a0_00)(1)*domegadCC;

    // kappa
    Matrix2d dRkappadCC;
    dRkappadCC = (dphifdotplusdCC/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/3.  -kappa)+(phif_dot_plus/tau_kappa)*((gamma_kappa/3.)*pow(lamda0/lamda1,gamma_kappa-1))*((1./lamda1)*dlamda0dCC-(lamda0/(lamda1*lamda1))*dlamda1dCC);

    // lamdaP
    Matrix2d dRlamdaP_adCC;
    Matrix2d dRlamdaP_sdCC;
    dRlamdaP_adCC = (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a))*dCe_aadCC+dphifdotplusdCC*(lamdaE_a-1.)/tau_lamdaP_a;
    dRlamdaP_sdCC = (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s))*dCe_ssdCC+dphifdotplusdCC*(lamdaE_s-1.)/tau_lamdaP_s;

    // Assemble dRThetadCC
    // count is phi=4, a0x = 4, a0y = 4, kappa = 4, lamdaP_a =4, lamdaP_s = 4
    VectorXd dRThetadCC(24);dRThetadCC.setZero();
    // phi
    dRThetadCC(0) = dRphifdCC(0,0); dRThetadCC(1) = dRphifdCC(0,1); dRThetadCC(2) = dRphifdCC(1,0); dRThetadCC(3) = dRphifdCC(1,1);
    // a0x
    dRThetadCC(4) = dRa0xdCC(0,0);  dRThetadCC(5) = dRa0xdCC(0,1);  dRThetadCC(6)  = dRa0xdCC(1,0);   dRThetadCC(7) = dRa0xdCC(1,1);
    // a0y
    dRThetadCC(8) = dRa0ydCC(0,0);  dRThetadCC(9) = dRa0ydCC(0,1);  dRThetadCC(10) = dRa0ydCC(1,0);  dRThetadCC(11) = dRa0ydCC(1,1);
    // kappa
    dRThetadCC(12) = dRkappadCC(0,0);  dRThetadCC(13) = dRkappadCC(0,1);  dRThetadCC(14) = dRkappadCC(1,0);  dRThetadCC(15) = dRkappadCC(1,1);
    // lamdaP_a
    dRThetadCC(16) = dRlamdaP_adCC(0,0);  dRThetadCC(17) = dRlamdaP_adCC(0,1);  dRThetadCC(18) = dRlamdaP_adCC(1,0);  dRThetadCC(19) = dRlamdaP_adCC(1,1);
    // lamdaP_s
    dRThetadCC(20) = dRlamdaP_sdCC(0,0);  dRThetadCC(21) = dRlamdaP_sdCC(0,1);  dRThetadCC(22) = dRlamdaP_sdCC(1,0);  dRThetadCC(23) = dRlamdaP_sdCC(1,1);

    // Assemble KK_local_extended

    MatrixXd dRThetadTheta_ext(24,24);dRThetadTheta_ext.setZero();
    for(int kkj=0;kkj<6;kkj++){
        // phi
        dRThetadTheta_ext(0,kkj*4+0) = KK_local(0,kkj);
        dRThetadTheta_ext(1,kkj*4+1) = KK_local(0,kkj);
        dRThetadTheta_ext(2,kkj*4+2) = KK_local(0,kkj);
        dRThetadTheta_ext(3,kkj*4+3) = KK_local(0,kkj);
        // a0x
        dRThetadTheta_ext(4,kkj*4+0) = KK_local(1,kkj);
        dRThetadTheta_ext(5,kkj*4+1) = KK_local(1,kkj);
        dRThetadTheta_ext(6,kkj*4+2) = KK_local(1,kkj);
        dRThetadTheta_ext(7,kkj*4+3) = KK_local(1,kkj);
        // a0y
        dRThetadTheta_ext(8,kkj*4+0) =  KK_local(2,kkj);
        dRThetadTheta_ext(9,kkj*4+1) =  KK_local(2,kkj);
        dRThetadTheta_ext(10,kkj*4+2) = KK_local(2,kkj);
        dRThetadTheta_ext(11,kkj*4+3) = KK_local(2,kkj);
        // kappa
        dRThetadTheta_ext(12,kkj*4+0) = KK_local(3,kkj);
        dRThetadTheta_ext(13,kkj*4+1) = KK_local(3,kkj);
        dRThetadTheta_ext(14,kkj*4+2) = KK_local(3,kkj);
        dRThetadTheta_ext(15,kkj*4+3) = KK_local(3,kkj);
        // lamdaP_a
        dRThetadTheta_ext(16,kkj*4+0) = KK_local(4,kkj);
        dRThetadTheta_ext(17,kkj*4+1) = KK_local(4,kkj);
        dRThetadTheta_ext(18,kkj*4+2) = KK_local(4,kkj);
        dRThetadTheta_ext(19,kkj*4+3) = KK_local(4,kkj);
        // lamdaP_s
        dRThetadTheta_ext(20,kkj*4+0) = KK_local(5,kkj);
        dRThetadTheta_ext(21,kkj*4+1) = KK_local(5,kkj);
        dRThetadTheta_ext(22,kkj*4+2) = KK_local(5,kkj);
        dRThetadTheta_ext(23,kkj*4+3) = KK_local(5,kkj);
    }

    // SOLVE for the dThetadCC
    dThetadCC = dRThetadTheta_ext.lu().solve(-dRThetadCC);



    //----------//
    // RHO
    //----------//

    // Explicit derivatives of the residuals with respect to rho

    // phi
    double dphif_dot_plusdrho = (p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*He)*(1./(K_phi_rho+phif));
    double dRphidrho = dphif_dot_plusdrho - d_phi_rho_c*c*phif;


    // a0
    double domegadrho = ((2.*PIE*dphif_dot_plusdrho)/(tau_omega))*lamda1*sinVartheta;
    Vector2d dRa0drho = (-dRomegadomega*a0_00)*domegadrho;

    // kappa
    double dRkappadrho =(dphif_dot_plusdrho/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/3.  -kappa);

    // lamdaP
    Vector2d dRlamdaPdrho;
    dRlamdaPdrho(0) =  dphif_dot_plusdrho*(lamdaE_a-1)/tau_lamdaP_a;
    dRlamdaPdrho(1) =  dphif_dot_plusdrho*(lamdaE_s-1)/tau_lamdaP_s;

    // Aseemble in one vector
    VectorXd dRThetadrho(6);
    dRThetadrho(0) = dRphidrho;
    dRThetadrho(1) = dRa0drho(0);
    dRThetadrho(2) = dRa0drho(1);
    dRThetadrho(3) = dRkappadrho;
    dRThetadrho(4) = dRlamdaPdrho(0);
    dRThetadrho(5) = dRlamdaPdrho(1);

    // the tangent matrix in this case remains the same as KK_local
    dThetadrho = KK_local.lu().solve(-dRThetadrho);



    //----------//
    // c
    //----------//

    // Explicit derivatives of the residuals with respect to c

    // phi
    double dphif_dot_plusdc = (rho/(K_phi_rho+phif))*((p_phi_c)/(K_phi_c+c) - (p_phi_c*c)/((K_phi_c+c)*(K_phi_c+c)));
    double dRphidc = dphif_dot_plusdc - d_phi_rho_c*rho*phif;

    // a0
    double domegadc = ((2.*PIE*dphif_dot_plusdc)/(tau_omega))*lamda1*sinVartheta;
    Vector2d dRa0dc = (-dRomegadomega*a0_00)*domegadc;

    // kappa
    double dRkappadc =(dphif_dot_plusdc/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/3.  -kappa);

    // lamdaP
    Vector2d dRlamdaPdc;
    dRlamdaPdc(0) =  dphif_dot_plusdc*(lamdaE_a-1)/tau_lamdaP_a;
    dRlamdaPdc(1) =  dphif_dot_plusdc*(lamdaE_s-1)/tau_lamdaP_s;

    // Aseemble in one vector
    VectorXd dRThetadc(6);
    dRThetadc(0) = dRphidc;
    dRThetadc(1) = dRa0dc(0);
    dRThetadc(2) = dRa0dc(1);
    dRThetadc(3) = dRkappadc;
    dRThetadc(4) = dRlamdaPdc(0);
    dRThetadc(5) = dRlamdaPdc(1);

    // the tangent matrix in this case remains the same as KK_local
    dThetadc = KK_local.lu().solve(-dRThetadc);
}


//------------------------------//
// ODE VERIFICATION
//------------------------------//
// MOVE THIS TO A DIFFERENT SECTION
//
// Print results in the method as csv
/*double time = 100;
double dt = 0.1;
double c = c_wound;
double rho = rho_healthy;
double ip_phif = 1;
Vector3d ip_a0 = a0_healthy;
Vector3d ip_s0 = s0_healthy;
double ip_kappa = 0.25;
Vector3d ip_lamdaP = lamda0_healthy;
Matrix3d FF; FF << 1,0,0, 0,2,0, 0,0,1;
VectorXd dThetadCC(48);dThetadCC.setZero();
VectorXd dThetadCC2(24);dThetadCC2.setZero();
VectorXd dThetadrho(8);dThetadrho.setZero();
VectorXd dThetadc(8);dThetadc.setZero();
//localWoundProblemExplicit(time,local_parameters,c,rho,FF,ip_phif,ip_a0,ip_kappa,ip_lamdaP,ip_phif,ip_a0,ip_kappa,ip_lamdaP,dThetadCC,dThetadrho,dThetadc);
//ip_phif = 1;
//ip_a0 = a0_healthy;
//ip_s0 = s0_healthy;
//ip_kappa = 0.25;
//ip_lamdaP = lamda0_healthy;
std::ofstream myfile;
myfile.open("BE_results.csv");
for (int i=0;i<100;i++){
    //myfile << dt*i << "," << ip_phif << "," << ip_kappa << "," << ip_a0(0) << "," << ip_a0(1) << "," << ip_lamdaP(0) << "," << ip_lamdaP(1) << "\n";
    //localWoundProblemImplicit2d(time,local_parameters,c,rho,FF,ip_phif,ip_a0,ip_kappa,ip_lamdaP,ip_phif,ip_a0,ip_kappa,ip_lamdaP,dThetadCC,dThetadrho,dThetadc);
    myfile << dt*i << "," << ip_phif << "," << ip_kappa << "," << ip_a0(0) << "," << ip_a0(1) << "," << ip_a0(2) << "," << ip_lamdaP(0) << "," << ip_lamdaP(1) << "," << ip_lamdaP(2) << "\n";
    localWoundProblemImplicitAutodiff(dt,local_parameters,c,rho,FF,ip_phif,ip_a0,ip_s0,ip_kappa,ip_lamdaP,ip_phif,ip_a0,ip_s0,ip_kappa,ip_lamdaP,dThetadCC,dThetadrho,dThetadc);
}
myfile.close();
return 0;*/
//-----------------------------//
