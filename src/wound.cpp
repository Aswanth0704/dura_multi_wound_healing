// last date modified: Sept, 16, 2025
/*

WOUND

This code is the implementation of the DaLaWoHe

*/

#include <omp.h>
#include "wound.h"
#include "local_solver.h"
#include "element_functions.h"
#include "mechanosensing.h"
#include "diffusivity.h"
#include <iostream>
#include <cmath>
#include <map>
#include <Eigen/OrderingMethods>
#include <Eigen/Eigenvalues>
#include <fstream>

using namespace Eigen;

//--------------------------------------------------------//
// RESIDUAL AND TANGENT
//--------------------------------------------------------//

// ELEMENT RESIDUAL AND TANGENT
void evalWound(
        double dt, double time, double time_final,
        const std::vector<Matrix3d> &ip_Jac,
        const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
        std::vector<Matrix3d> &ip_strain,std::vector<Matrix3d> &ip_stress, const std::vector<double> &node_rho_0, const std::vector<double> &node_c_0, const std::vector<double> &node_alpha_0, //
        const std::vector<double> &ip_phif_0,const std::vector<Vector3d> &ip_a0_0,const std::vector<Vector3d> &ip_s0_0,const std::vector<Vector3d> &ip_n0_0,const std::vector<double> &ip_kappa_0, const std::vector<Vector3d> &ip_lamdaP_0, //
        const std::vector<double> &node_rho, const std::vector<double> &node_c, const std::vector<double> &node_alpha,
        std::vector<double> &ip_phif, std::vector<Vector3d> &ip_a0, std::vector<Vector3d> &ip_s0, std::vector<Vector3d> &ip_n0, std::vector<double> &ip_kappa, std::vector<Vector3d> &ip_lamdaP, //
        std::vector<Vector3d> &ip_lamdaE,
        const std::vector<Vector3d> &node_x,
        const std::vector<Vector3d> &node_X,
        std::vector<Vector3d> &ip_dphifdu, std::vector<double> &ip_dphifdrho, std::vector<double> &ip_dphifdc,
        VectorXd &Re_x,MatrixXd &Ke_x_x,MatrixXd &Ke_x_rho,MatrixXd &Ke_x_c,MatrixXd &Ke_x_alpha,
        VectorXd &Re_rho,MatrixXd &Ke_rho_x, MatrixXd &Ke_rho_rho,MatrixXd &Ke_rho_c,MatrixXd &Ke_rho_alpha,
        VectorXd &Re_c,MatrixXd &Ke_c_x,MatrixXd &Ke_c_rho,MatrixXd &Ke_c_c,MatrixXd &Ke_c_alpha,
        VectorXd &Re_alpha,MatrixXd &Ke_alpha_x,MatrixXd &Ke_alpha_rho,MatrixXd &Ke_alpha_c,MatrixXd &Ke_alpha_alpha)
{

    //std::cout<<"element routine\n";
    //---------------------------------//
    // INPUT
    //  dt: time step
    //	elem_jac_IP: jacobians at the integration points, needed for the deformation grad
    //  matParam: material parameters
    //  Xi_t: global fields at previous time step
    //  Theta_t: structural fields at previous time steps
    //  Xi: current guess of the global fields
    //  Theta: current guess of the structural fields
    //	node_x: deformed positions
    //
    // OUTPUT
    //  Re: all residuals
    //  Ke: all tangents
    //
    // Algorithm
    //  0. Loop over integration points
    //	1. F,rho,c,nabla_rho,nabla_c: deformation at IP
    //  2. LOCAL NEWTON -> update the current guess of the structural parameters
    //  3. Fe,Fp
    //	4. Se_pas,Se_act,S
    //	5. Qrho,Srho,Qc,Sc
    //  6. Residuals
    //  7. Tangents
    //---------------------------------//

    //---------------------------------//
    // PARAMETERS
    //
    double k0 = global_parameters[0]; // neo hookean
    double kf = global_parameters[1]; // stiffness of collagen
    double k2 = global_parameters[2]; // nonlinear exponential
    double t_rho = global_parameters[3]; // force of fibroblasts
    double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
    double K_t = global_parameters[5]; // saturation of collagen on force
    double K_t_c = global_parameters[6]; // saturation of chemical on force
    double D_rhoc = global_parameters[8]; // diffusion of chemotactic gradient
    double D_cc = global_parameters[9]; // diffusion of chemical
    double p_rho =global_parameters[10]; // production of fibroblasts naturally
    double p_rho_c = global_parameters[11]; // production enhanced by the chem
    double p_rho_theta = global_parameters[12]; // mechanosensing
    double K_rho_c= global_parameters[13]; // saturation of cell production by chemical
    double K_rho_rho = global_parameters[14]; // saturation of cell by cell
    double d_rho = global_parameters[15] ;// decay of cells
    double vartheta_e = global_parameters[16]; // physiological state of area stretch
    double gamma_theta = global_parameters[17]; // sensitivity of heviside function
    double p_c_rho = global_parameters[18];// production of C by cells
    double p_c_thetaE = global_parameters[19]; // coupling of elastic and chemical
    double K_c_c = global_parameters[20];// saturation of chem by chem
    double d_c = global_parameters[21]; // decay of chemical
    double bx = global_parameters[22]; // body force
    double by = global_parameters[23]; // body force
    double bz = global_parameters[24]; // body force
    // alpha (pro-inflammatory signal). APPENDED at the end so that every
    // existing literal index into global_parameters stays valid.
    double D_alpha   = (global_parameters.size()>25) ? global_parameters[25] : 0.0;
    double d_alpha   = (global_parameters.size()>26) ? global_parameters[26] : 0.0;
    double p_c_alpha = (global_parameters.size()>27) ? global_parameters[27] : 0.0;
    //std::cout<<"read all global parameters\n";
    //
    //---------------------------------//



    //---------------------------------//
    // GLOBAL VARIABLES
    // Initialize the residuals to zero and declare some global stuff
    Re_x.setZero(); Ke_x_x.setZero(); Ke_x_rho.setZero(); Ke_x_c.setZero();
    Re_rho.setZero(); Ke_rho_x.setZero(); Ke_rho_rho.setZero(); Ke_rho_c.setZero();
    Re_c.setZero(); Ke_c_x.setZero(); Ke_c_rho.setZero(); Ke_c_c.setZero();
    Ke_x_alpha.setZero(); Ke_rho_alpha.setZero(); Ke_c_alpha.setZero();
    Re_alpha.setZero(); Ke_alpha_x.setZero(); Ke_alpha_rho.setZero();
    Ke_alpha_c.setZero(); Ke_alpha_alpha.setZero();
    int elem_size = node_x.size();
    std::vector<Vector3d> Ebasis; Ebasis.clear();
    Ebasis.push_back(Vector3d(1.,0.,0.)); Ebasis.push_back(Vector3d(0.,1.,0.)); Ebasis.push_back(Vector3d(0.,0.,1.));
    //---------------------------------//



    //---------------------------------//
    // LOOP OVER INTEGRATION POINTS
    //---------------------------------//
    //---------------------------------//

    // array with integration points
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
    // we are going to interpolate the position
    Vector3d X; X.setZero();
    //std::cout<<"loop over integration points\n";
    for(int ip=0;ip<IP_size;ip++)
    {
        //---------------------------------//
        // EVALUATE FUNCTIONS
        //
        // coordinates of the integration point in parent domain
        double xi = IP[ip](0);
        double eta = IP[ip](1);
        double zeta = IP[ip](2);
        // weight of the integration point
        double wip = IP[ip](3);
        double Jac = 1./ip_Jac[ip].determinant(); // instead of Jacobian, J^(-T) is stored
        //std::cout<<"integration point: "<<xi<<", "<<eta<<", "<<zeta<<"; "<<wip<<"; "<<Jac<<"\n";
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
        //
        // declare variables and gradients at IP
        std::vector<Vector3d> dRdXi;dRdXi.clear();
        Vector3d dxdxi,dxdeta,dxdzeta; dxdxi.setZero();dxdeta.setZero();dxdzeta.setZero();
        double rho_0=0.; Vector3d drho0dXi; drho0dXi.setZero();
        double rho=0.; Vector3d drhodXi; drhodXi.setZero();
        double c_0=0.; Vector3d dc0dXi; dc0dXi.setZero();
        double c=0.; Vector3d dcdXi; dcdXi.setZero();
        double alpha_0=0.; Vector3d dalpha0dXi; dalpha0dXi.setZero();
        double alpha=0.;   Vector3d dalphadXi;  dalphadXi.setZero();
        //
        for(int ni=0;ni<elem_size;ni++)
        {
            dRdXi.push_back(Vector3d(Rxi[ni],Reta[ni],Rzeta[ni]));

            // eval the isoparametric map
	    X += node_X[ni]*R[ni];
	    // Jacobian 
            dxdxi += node_x[ni]*Rxi[ni];
            dxdeta += node_x[ni]*Reta[ni];
            dxdzeta += node_x[ni]*Rzeta[ni];

            rho_0 += node_rho_0[ni]*R[ni];
            drho0dXi(0) += node_rho_0[ni]*Rxi[ni];
            drho0dXi(1) += node_rho_0[ni]*Reta[ni];
            drho0dXi(2) += node_rho_0[ni]*Rzeta[ni];

            rho += node_rho[ni]*R[ni];
            drhodXi(0) += node_rho[ni]*Rxi[ni];
            drhodXi(1) += node_rho[ni]*Reta[ni];
            drhodXi(2) += node_rho[ni]*Rzeta[ni];

            c_0 += node_c_0[ni]*R[ni];
            dc0dXi(0) += node_c_0[ni]*Rxi[ni];
            dc0dXi(1) += node_c_0[ni]*Reta[ni];
            dc0dXi(2) += node_c_0[ni]*Rzeta[ni];

            c += node_c[ni]*R[ni];
            dcdXi(0) += node_c[ni]*Rxi[ni];
            dcdXi(1) += node_c[ni]*Reta[ni];
            dcdXi(2) += node_c[ni]*Rzeta[ni];

            alpha_0 += node_alpha_0[ni]*R[ni];
            dalpha0dXi(0) += node_alpha_0[ni]*Rxi[ni];
            dalpha0dXi(1) += node_alpha_0[ni]*Reta[ni];
            dalpha0dXi(2) += node_alpha_0[ni]*Rzeta[ni];

            alpha += node_alpha[ni]*R[ni];
            dalphadXi(0) += node_alpha[ni]*Rxi[ni];
            dalphadXi(1) += node_alpha[ni]*Reta[ni];
            dalphadXi(2) += node_alpha[ni]*Rzeta[ni];
        }
        //
        //---------------------------------//



        //---------------------------------//
        // EVAL GRADIENTS
        //
        // Deformation gradient and strain
        // assemble the columns
        Matrix3d dxdXi;dxdXi<< dxdxi(0), dxdeta(0), dxdzeta(0), dxdxi(1), dxdeta(1), dxdzeta(1), dxdxi(2), dxdeta(2), dxdzeta(2);
        // F = dxdX
        Matrix3d FF = dxdXi*ip_Jac[ip].transpose();

        // the strain
        Matrix3d Identity;Identity << 1.,0.,0., 0.,1.,0., 0.,0.,1.;
        //Matrix3d EE = 0.5*(FF.transpose()*FF - Identity);
        Matrix3d CC = FF.transpose()*FF;
        Matrix3d CCinv = CC.inverse();
        //
        // Gradient of concentrations in current configuration
        Matrix3d dXidx = dxdXi.inverse();
        Vector3d grad_rho0 = dXidx.transpose()*drho0dXi;
        Vector3d grad_rho  = dXidx.transpose()*drhodXi;
        Vector3d grad_c0   = dXidx.transpose()*dc0dXi;
        Vector3d grad_c    = dXidx.transpose()*dcdXi;
        //
        // Gradient of concentrations in reference configuration
        Vector3d Grad_rho0 = ip_Jac[ip]*drho0dXi;
        Vector3d Grad_rho = ip_Jac[ip]*drhodXi;
        Vector3d Grad_c0 = ip_Jac[ip]*dc0dXi;
        Vector3d Grad_c = ip_Jac[ip]*dcdXi;
        Vector3d Grad_alpha0 = ip_Jac[ip]*dalpha0dXi;
        Vector3d Grad_alpha  = ip_Jac[ip]*dalphadXi;
        //
        // Gradient of basis functions for the nodes in reference
        std::vector<Vector3d> Grad_R;Grad_R.clear();
        // Gradient of basis functions in deformed configuration
        std::vector<Vector3d> grad_R;grad_R.clear();
        for(int ni=0;ni<elem_size;ni++)
        {
            Grad_R.push_back(ip_Jac[ip]*dRdXi[ni]);
            grad_R.push_back(dXidx.transpose()*dRdXi[ni]);
        }
        //
        //---------------------------------//

        //std::cout<<"deformation gradient\n"<<FF<<"\n";
        //std::cout<<"rho0: "<<rho_0<<", rho: "<<rho<<"\n";
        //std::cout<<"c0: "<<c_0<<", c: "<<c<<"\n";
        //std::cout<<"gradient of rho: "<<Grad_rho<<"\n";
        //std::cout<<"gradient of c: "<<Grad_c<<"\n";



        //---------------------------------//
        // LOCAL NEWTON: structural problem
        //
        //VectorXd dThetadCC(24);dThetadCC.setZero();
        //VectorXd dThetadrho(6);dThetadrho.setZero();
        //VectorXd dThetadc(6);dThetadc.setZero();
        VectorXd dThetadCC(48);dThetadCC.setZero();
        VectorXd dThetadrho(8);dThetadrho.setZero();
        VectorXd dThetadc(8);dThetadc.setZero();
        //std::cout<<"Local variables before update:\nphif0 = "<<ip_phif_0[ip]<<"\nkappa_0 = "<<ip_kappa_0[ip]<<"\na0_0 = ["<<ip_a0_0[ip](0)<<","<<ip_a0_0[ip](1)<<"]\nlamdaP_0 = ["<<ip_lamdaP_0[ip](0)<<","<<ip_lamdaP_0[ip](1)<<"]\n";

        //localWoundProblemImplicit2d(dt,local_parameters,c,rho,FF,ip_phif_0[ip],ip_a0_0[ip],ip_s0_0[ip],ip_kappa_0[ip],ip_lamdaP_0[ip],ip_phif[ip],ip_a0[ip],ip_s0[ip],ip_kappa[ip],ip_lamdaP[ip],dThetadCC,dThetadrho,dThetadc);
        //localWoundProblemExplicit2d(dt,local_parameters,c,rho,FF,ip_phif_0[ip],ip_a0_0[ip],ip_s0_0[ip],ip_kappa_0[ip],ip_lamdaP_0[ip],ip_phif[ip],ip_a0[ip],ip_s0[ip],ip_kappa[ip],ip_lamdaP[ip],dThetadCC,dThetadrho,dThetadc);
        localWoundProblemExplicit(dt,local_parameters,X,c,rho,FF,ip_phif_0[ip],ip_a0_0[ip],ip_s0_0[ip],ip_n0_0[ip],ip_kappa_0[ip],ip_lamdaP_0[ip],ip_phif[ip],ip_a0[ip],ip_s0[ip],ip_n0[ip],ip_kappa[ip],ip_lamdaP[ip],dThetadCC,dThetadrho,dThetadc);

        // OUTPUT: Theta, dThetadCC, dThetadrho, dThetadc

        // make sure the update preserved length
        double norma0 = sqrt(ip_a0[ip].dot(ip_a0[ip]));
        if(fabs(norma0-1.)>0.001){std::cout<<"update did not preserve unit length of a0\n";}
        ip_a0[ip] = ip_a0[ip]/(sqrt(ip_a0[ip].dot(ip_a0[ip])));
        double norms0 = sqrt(ip_s0[ip].dot(ip_s0[ip]));
        if(fabs(norms0-1.)>0.001){std::cout<<"update did not preserve unit length of s0\n";}
        ip_s0[ip] = ip_s0[ip]/(sqrt(ip_s0[ip].dot(ip_s0[ip])));
        double normn0 = sqrt(ip_n0[ip].dot(ip_n0[ip]));
        if(fabs(norms0-1.)>0.001){std::cout<<"update did not preserve unit length of n0\n";}
        ip_n0[ip] = ip_n0[ip]/(sqrt(ip_n0[ip].dot(ip_n0[ip])));
        // rename variables to make it easier to track
        double phif_0 = ip_phif_0[ip];
        Vector3d a0_0 = ip_a0_0[ip];
        Vector3d s0_0 = ip_s0_0[ip];
        Vector3d n0_0 = ip_n0_0[ip];
        double kappa_0 = ip_kappa_0[ip];
        Vector3d lamdaP_0 = ip_lamdaP_0[ip];
        double lamdaP_a_0 = lamdaP_0(0);
        double lamdaP_s_0 = lamdaP_0(1);
        double lamdaP_N_0 = lamdaP_0(2);

        double phif = ip_phif[ip];
        Vector3d a0 = ip_a0[ip];
        Vector3d s0 = ip_s0[ip];
        Vector3d n0 = ip_n0[ip];
        double kappa = ip_kappa[ip];
        Vector3d lamdaP = ip_lamdaP[ip];
        double lamdaP_a = lamdaP(0);
        double lamdaP_s = lamdaP(1);
        double lamdaP_N = lamdaP(2);
        //std::cout<<"Local variables after update:\nphif0 = "<<phif_0<<",	phif = "<<phif<<"\nkappa_0 = "<<kappa_0<<",	kappa = "<<kappa<<"\ns0_0 = ["<<s0_0(0)<<","<<s0_0(1)<<","<<s0_0(2)<<"],	s0 = ["<<s0(0)<<","<<s0(1)<<","<<s0(2)<<"\na0_0 = ["<<a0_0(0)<<","<<a0_0(1)<<","<<a0_0(2)<<"],	a0 = ["<<a0(0)<<","<<a0(1)<<","<<a0(2)<<"]\nlamdaP_0 = ["<<lamdaP_0(0)<<","<<lamdaP_0(1)<<","<<lamdaP_0(2)<<"],	lamdaP = ["<<lamdaP(0)<<","<<lamdaP(1)<<","<<lamdaP(2)<<"]\n";
        //
        //--------------------------------------//
        // unpack the derivatives wrt CC
        //--------------------------------------//
        //
        // remember dThetatCC: phi, a0x, a0y, kappa, lamdaP
        Matrix3d dphifdCC; dphifdCC.setZero();
        Matrix3d da0xdCC;da0xdCC.setZero();
        Matrix3d da0ydCC;da0ydCC.setZero();
        Matrix3d da0zdCC;da0zdCC.setZero();
        Matrix3d dkappadCC; dkappadCC.setZero();
        Matrix3d dlamdaP_adCC; dlamdaP_adCC.setZero();
        Matrix3d dlamdaP_sdCC; dlamdaP_sdCC.setZero();
        Matrix3d dlamdaP_ndCC; dlamdaP_ndCC.setZero();
        //-------------------//2D VERSION------------------------//
        // ALSO REMEMBER TO COMMENT THE EXTRA ELEMENTS FOR RHO AND C
        /*dphifdCC(0,0) = dThetadCC(0);
        dphifdCC(0,1) = dThetadCC(1);
        dphifdCC(1,0) = dThetadCC(2);
        dphifdCC(1,1) = dThetadCC(3);
        da0xdCC(0,0) = dThetadCC(4);
        da0xdCC(0,1) = dThetadCC(5);
        da0xdCC(1,0) = dThetadCC(6);
        da0xdCC(1,1) = dThetadCC(7);
        da0ydCC(0,0) = dThetadCC(8);
        da0ydCC(0,1) = dThetadCC(9);
        da0ydCC(1,0) = dThetadCC(10);
        da0ydCC(1,1) = dThetadCC(11);
        dkappadCC(0,0) = dThetadCC(12);
        dkappadCC(0,1) = dThetadCC(13);
        dkappadCC(1,0) = dThetadCC(14);
        dkappadCC(1,1) = dThetadCC(15);
        dlamdaP_adCC(0,0) = dThetadCC(16);
        dlamdaP_adCC(0,1) = dThetadCC(17);
        dlamdaP_adCC(1,0) = dThetadCC(18);
        dlamdaP_adCC(1,1) = dThetadCC(19);
        dlamdaP_sdCC(0,0) = dThetadCC(20);
        dlamdaP_sdCC(0,1) = dThetadCC(21);
        dlamdaP_sdCC(1,0) = dThetadCC(22);
        dlamdaP_sdCC(1,1) = dThetadCC(23);
        // unpack the derivatives wrt rho
        double dphifdrho = dThetadrho(0);
        double da0xdrho  = dThetadrho(1);
        double da0ydrho  = dThetadrho(2);
        double da0zdrho = 0;
        double dkappadrho  = dThetadrho(3);
        double dlamdaP_adrho  = dThetadrho(4);
        double dlamdaP_sdrho  = dThetadrho(5);
        double dlamdaP_ndrho = 0;
        // unpack the derivatives wrt c
        double dphifdc = dThetadc(0);
        double da0xdc  = dThetadc(1);
        double da0ydc  = dThetadc(2);
        double da0zdc = 0;
        double dkappadc  = dThetadc(3);
        double dlamdaP_adc  = dThetadc(4);
        double dlamdaP_sdc  = dThetadc(5);
        double dlamdaP_ndc = 0;*/
        //-------------------//3D VERSION------------------------//
        // ALSO REMEMBER TO UNCOMMENT THE EXTRA ELEMENTS FOR RHO AND C
        //
        // Use these to get at the six elements of CC that we need
        VectorXd voigt_table_I(6);
        voigt_table_I(0) = 0; voigt_table_I(1) = 1; voigt_table_I(2) = 2;
        voigt_table_I(3) = 1; voigt_table_I(4) = 0; voigt_table_I(5) = 0;
        VectorXd voigt_table_J(6);
        voigt_table_J(0) = 0; voigt_table_J(1) = 1; voigt_table_J(2) = 2;
        voigt_table_J(3) = 2; voigt_table_J(4) = 2; voigt_table_J(5) = 1;
        for (int II=0; II<6; II++){
            int ii = voigt_table_I(II);
            int jj = voigt_table_J(II);
            dphifdCC(ii,jj) = dThetadCC(0+II);
            da0xdCC(ii,jj) = dThetadCC(6+II);
            da0ydCC(ii,jj) = dThetadCC(12+II);
            da0zdCC(ii,jj) = dThetadCC(18+II);
            dkappadCC(ii,jj) = dThetadCC(24+II);
            dlamdaP_adCC(ii,jj) = dThetadCC(30+II);
            dlamdaP_sdCC(ii,jj) = dThetadCC(36+II);
            dlamdaP_ndCC(ii,jj) = dThetadCC(42+II);
            if(ii!=jj){
                dphifdCC(jj,ii) = dThetadCC(0+II);
                da0xdCC(jj,ii) = dThetadCC(6+II);
                da0ydCC(jj,ii) = dThetadCC(12+II);
                da0zdCC(jj,ii) = dThetadCC(18+II);
                dkappadCC(jj,ii) = dThetadCC(24+II);
                dlamdaP_adCC(jj,ii) = dThetadCC(30+II);
                dlamdaP_sdCC(jj,ii) = dThetadCC(36+II);
                dlamdaP_ndCC(jj,ii) = dThetadCC(42+II);
            }
        }
        // unpack the derivatives wrt rho
        double dphifdrho = dThetadrho(0);
        double da0xdrho  = dThetadrho(1);
        double da0ydrho  = dThetadrho(2);
        double da0zdrho = dThetadrho(3);
        double dkappadrho  = dThetadrho(4);
        double dlamdaP_adrho  = dThetadrho(5);
        double dlamdaP_sdrho  = dThetadrho(6);
        double dlamdaP_ndrho  = dThetadrho(7);
        // unpack the derivatives wrt c
        double dphifdc = dThetadc(0);
        double da0xdc  = dThetadc(1);
        double da0ydc  = dThetadc(2);
        double da0zdc  = dThetadc(3);
        double dkappadc  = dThetadc(4);
        double dlamdaP_adc  = dThetadc(5);
        double dlamdaP_sdc  = dThetadc(6);
        double dlamdaP_ndc  = dThetadc(7);   // was dThetadrho(7): copy-paste slip
        //---------------------------------//
        //std::cout<<"SOLVE.\ndThetadCC\n"<<dThetadCC<<"\ndThetadrho\n"<<dThetadrho<<"\ndThetadc\n"<<dThetadc<<"\n";


        //---------------------------------//
        // CALCULATE SOURCE AND FLUX
        //
        // Update kinematics
        CCinv = CC.inverse();
        // fiber tensor in the reference
        Matrix3d a0a0 = a0*a0.transpose();
        Matrix3d s0s0 = s0*s0.transpose();
        Matrix3d n0n0 = n0*n0.transpose();
        // recompute split
        Matrix3d FFg = lamdaP_a*(a0a0) + lamdaP_s*(s0s0) + lamdaP_N*(n0n0);
        Matrix3d FFginv = (1./lamdaP_a)*(a0a0) + (1./lamdaP_s)*(s0s0) + (1./lamdaP_N)*(n0n0);
        Matrix3d FFe = FF*FFginv;
        // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
        // elastic strain
        Matrix3d CCe = FFe.transpose()*FFe;
        Matrix3d CCeinv = CCe.inverse();

        // Things to pass back
        // Eulerian strain
        ip_strain[ip] = 0.5*(CCe - Identity);
        // Elastic stretches of the directions a and s
        double Ce_aa = a0.transpose()*CCe*a0;
        double Ce_ss = s0.transpose()*CCe*s0;
        double Ce_nn = n0.transpose()*CCe*n0;
        double lamdaE_a = sqrt(Ce_aa);
        double lamdaE_s = sqrt(Ce_ss);
        double lamdaE_n = sqrt(Ce_nn);
        ip_lamdaE[ip] = Vector3d(lamdaE_a, lamdaE_s, lamdaE_n);

        // invariant of the elastic strain
        double I1e = CCe(0,0) + CCe(1,1) + CCe(2,2);
        double I4e = a0.dot(CCe*a0);
        double I4tot = a0.dot(CC*a0);

        // Jacobian of the deformations
        double Jp = lamdaP_a*lamdaP_s*lamdaP_N;
        double Je = sqrt(CCe.determinant());
        double J = Je*Jp;
        // calculate the normal stretch
        //double thetaP = lamdaP_a*lamdaP_s;
        // Nanson's formula for area change from intermediate to deformed (thetaE) and reference to deformed (theta)
        //double thetaE = sqrt((Je*FFe.inverse().transpose()*n0).dot(Je*FFe.inverse().transpose()*n0));
        //double theta = sqrt((J*FF.inverse().transpose()*n0).dot(J*FF.inverse().transpose()*n0));
        // if the FFg was set up correctly, theta = thetae*thetap should be satisfied
        // double theta = thetaE*thetaP;
        // std::cout<<"split of the determinants. theta = thetaE*thetaB = "<<theta<<" = "<<thetaE<<"*"<<thetaP<<"\n";
        // This should come from CC directly
        //double lamda_N = sqrt(n0.dot(CCe*n0));

        Matrix3d A0 = kappa*Identity + (1-3.*kappa)*a0a0;
        Vector3d a = FF*a0;
        Matrix3d A = kappa*FF*FF.transpose() + (1.-3.0*kappa)*a*a.transpose();
        double trA = A(0,0) + A(1,1) + A(2,2);
        Matrix3d hat_A = A/trA;

        // Second Piola Kirchhoff stress tensor
        //------------------//
        // PASSIVE STRESS
        //------------------//
        double Psif = (kf/(2.*k2))*(exp(k2*pow((kappa*I1e + (1-3*kappa)*I4e - 1),2)));
        double Psif1 = 2*k2*kappa*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif;
        double Psif4 = 2*k2*(1-3*kappa)*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif;
        //Matrix3d SSe_pas = k0*Identity + phif*(Psif1*Identity + Psif4*a0a0);
        Matrix3d SSe_pas = phif*(k0*Identity + Psif1*Identity + Psif4*a0a0);
        // pull back to the reference,
        Matrix3d SS_pas = Jp*FFginv*SSe_pas*FFginv;

        //------------------//
        // ACTIVE STRESS
        //------------------//
        double traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho;
        Matrix3d SS_act = (Jp*traction_act*phif/(trA*(K_t*K_t+phif*phif)))*A0;
        // Matrix3d SS_act = (Jp*traction_act*phif/trA)*A0;
        //double traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho;
        //Matrix3d SS_act = (Jp*traction_act*phif/(trA*(K_t*K_t+phif*phif)))*A0;

        //------------------//
        // VOLUME STRESS
        //------------------//
        // Instead of (double pressure = -k0*lamda_N*lamda_N;) directly, add volumetric part of stress SSvol
        // SSvol = 2dPsivol/dCC = 2dPsivol/dJe * dJe/dCC
        double penalty = 0.3166667;
        double Psivol = 0.5*phif*pow(penalty*(Je-1.),2) - 2*phif*k0*log(Je); //*phif
        double dPsivoldJe = phif*penalty*(Je-1.) - 2*phif*k0/Je;
        double dPsivoldJedJe = phif*penalty + 2*phif*k0/(Je*Je);
    double D_rhorho = evalDrho(phif, c);
        Matrix3d SSe_vol = dPsivoldJe*Je*CCeinv/2; // = phif*penalty*Je*(Je-1.)*CCeinv/2 - phif*k0*CCeinv;
        Matrix3d SS_vol = Jp*FFginv*SSe_vol*FFginv;

        //------------------//
        // TOTAL STRESS
        //------------------//
        // std::cout<<"stresses.\nSSpas\n"<<SS_pas<<"\nSS_act\n"<<SS_act<<"\nSS_vol"<<SS_vol<<"\n";
        Matrix3d SS = SS_pas + SS_vol + SS_act;
        ip_stress[ip] = SS;
        //double SSvm = sqrt(0.5*(pow((SS(0,0)-SS(1,1)),2)+pow((SS(1,1)-SS(2,2)),2)+pow((SS(2,2)-SS(0,0)),2))+3*(pow(SS(0,1),2)+pow(SS(1,2),2)+pow(SS(2,0),2)));
        VectorXd SS_voigt(6);
        SS_voigt(0) = SS(0,0); SS_voigt(1) = SS(1,1); SS_voigt(2) = SS(2,2);
        SS_voigt(3) = SS(1,2); SS_voigt(4) = SS(0,2); SS_voigt(5) = SS(0,1);

        //------------------//
        // FLUX
        //------------------//
        // Flux and Source terms for the rho and the C
        Vector3d Q_rho = -D_rhorho*CCinv*Grad_rho - D_rhoc*CCinv*Grad_c;
        //Vector3d Q_rho = -3.0*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0*Grad_rho/trA - 3.0*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0*Grad_c/trA;
        Vector3d Q_c = -D_cc*CCinv*Grad_c;
        //Vector3d Q_c = -3*(D_cc-phif*(D_cc-D_cc/10))*A0*Grad_c/trA;
        //Vector3d Q_c = -1.0*(D_cc-phif*(D_cc-D_cc/10))*CCinv*Grad_c;


        //------------------//
        // SOURCE
        //------------------//
        // Mechanosensing stimulus: in-plane AREAL elastic stretch of the dural
        // mid-surface, theta_e = ||cof(F^e).n0||, NOT det(F^e). The tissue is
        // treated as incompressible so det(F^e)==1 and could never respond to
        // membrane stretch. See include/mechanosensing.h.
        double theta_e = evalThetaE(J, CCinv, n0, lamdaP_a, lamdaP_s);
        double He = evalHe(theta_e, vartheta_e, gamma_theta);

        // function for elastic response of the cells
        double S_rho = (p_rho + p_rho_c*c/(K_rho_c+c) + p_rho_theta*He)*(1-rho/K_rho_rho)*rho - d_rho*rho;
        // function for elastic response of the chemical
        // Cytokine source gains the pro-inflammatory drive p_c_alpha*alpha.
        // At homeostasis alpha_h = 0 so this term vanishes and the derived
        // p_c_rho still makes (0,1,1,1) an exact fixed point.
        double S_c = (p_c_rho*c+ p_c_thetaE*He)*(rho/(K_c_c+c)) - d_c*c + p_c_alpha*alpha;

        //------------------//
        // ALPHA: pro-inflammatory signal
        //   alpha_dot + div(Q_alpha) = s_alpha,  Q_alpha = -D_alpha grad(alpha),
        //   s_alpha = -d_alpha alpha
        // No dependence on phif, a0, kappa or lamdaP, so this needs no new
        // structural sensitivities.
        //------------------//
        Vector3d Q_alpha = -D_alpha*CCinv*Grad_alpha;
        double S_alpha = -d_alpha*alpha;
        //std::cout<<"SS_voigt\n"<<SS_voigt<<"\n";
        //std::cout<<"flux of cells, Q _rho\n"<<Q_rho<<"\n";
        //std::cout<<"source of cells, S_rho: "<<S_rho<<"\n";
        //std::cout<<"flux of chemical, Q _c\n"<<Q_c<<"\n";
        //std::cout<<"source of chemical, S_c: "<<S_c<<"\n";
        //---------------------------------//



        //---------------------------------//
        // ADD TO THE RESIDUAL
        //---------------------------------//

        // Stabilization parameter for GGLS
        //double tau = 1/((4k/(h*h)) + (2*norm(a)/h) + s*(1-max(u)));
        // Element residual for x
        Matrix3d deltaFF,deltaCC;
        VectorXd deltaCC_voigt(6);
        for(int nodei=0;nodei<elem_size;nodei++){
            Vector3d b = Vector3d(bx,by,bz);
            for(int coordi=0;coordi<3;coordi++){
                // Internal force, alternatively, define the deltaCC
                deltaFF = Ebasis[coordi]*Grad_R[nodei].transpose();
                deltaCC = deltaFF.transpose()*FF + FF.transpose()*deltaFF;
                deltaCC_voigt(0) = deltaCC(0,0); deltaCC_voigt(1) = deltaCC(1,1); deltaCC_voigt(2) = deltaCC(2,2);
                deltaCC_voigt(3) = 2.*deltaCC(1,2); deltaCC_voigt(4) = 2.*deltaCC(0,2); deltaCC_voigt(5) = 2.*deltaCC(0,1);
                Re_x(nodei*3+coordi) += Jac*SS_voigt.dot(deltaCC_voigt)*wip;
                // External force/load vector, comment if not needed
                // This is not really the right way to do this, but I will fix it later
                if(time < 1.){
                    Re_x(nodei*3+coordi) -= Jac*b(coordi)*R[nodei]*wip*time/1.;
                } else{
                    Re_x(nodei*3+coordi) -= Jac*b(coordi)*R[nodei]*wip;
                }
            }
            // Element residuals for rho and c
            Re_rho(nodei) += Jac*(((rho-rho_0)/dt - S_rho)*R[nodei] - Grad_R[nodei].dot(Q_rho))*wip;
            Re_c(nodei) += Jac*(((c-c_0)/dt - S_c)*R[nodei] - Grad_R[nodei].dot(Q_c))*wip;
            Re_alpha(nodei) += Jac*(((alpha-alpha_0)/dt - S_alpha)*R[nodei] - Grad_R[nodei].dot(Q_alpha))*wip;
            // GGLS stabilization
            //Re_rho(nodei) += Jac*(((Grad_rho-Grad_rho_0)/dt - Grad_S_rho)*tau*(Grad_S_N))*wip;
        }
        //

//        //-------------------------------------------//
//        // EXTRA DERIVATIVE FOR THE BOUNDARY ELEMENTS
//        //-------------------------------------------//
//        // This is probably not that accurate
//        // It converges without this, but I will pass it for now
//        // I need linCC_Voigt to convert dphifdCC to dphifdu
//        for(int nodei=0;nodei<elem_size;nodei++){
//            for(int jj; jj<3; jj++){
//                for(int kk; kk<3; kk++){
//                    for(int ll; ll<3; ll++){
//                        for(int mm; mm<3; mm++){
//                            ip_dphifdu[ip](kk) += dphifdCC(ll,mm)*(FF(jj,ll)*Identity(kk,mm) + FF(jj,mm)*Identity(kk,ll))*Grad_R[nodei](kk);
//                        }
//                    }
//                }
//            }
//        }
//        ip_dphifdrho.push_back(dphifdrho);
//        ip_dphifdc.push_back(dphifdc);
        //
        //---------------------------------//



        //---------------------------------//
        // TANGENTS
        //---------------------------------//



        //---------------------------------//
        // NUMERICAL DERIVATIVES
        //
        // NOTE:
        // the chain rule for structural parameters is a mess, might as well use
        // numerical tangent, partially.
        // proposed: numerical tangent wrt structural parameters
        // then use the derivatives dThetadCC, dThetadrho, dThetadc
        // derivative wrt to CC is done analytically
        //
        double epsilon = 1e-7;
        //
        // structural parameters
        double phif_plus = phif + epsilon;
        double phif_minus = phif - epsilon;
        Vector3d a0_plus_x = a0 + epsilon*Ebasis[0];
        Vector3d a0_minus_x = a0 - epsilon*Ebasis[0];
        Vector3d a0_plus_y = a0 + epsilon*Ebasis[1];
        Vector3d a0_minus_y = a0 - epsilon*Ebasis[1];
        Vector3d a0_plus_z = a0 + epsilon*Ebasis[2];
        Vector3d a0_minus_z = a0 - epsilon*Ebasis[2];
        double kappa_plus = kappa + epsilon;
        double kappa_minus = kappa - epsilon;
        Vector3d lamdaP_plus_a = lamdaP + epsilon*Ebasis[0];
        Vector3d lamdaP_minus_a = lamdaP - epsilon*Ebasis[0];
        Vector3d lamdaP_plus_s = lamdaP + epsilon*Ebasis[1];
        Vector3d lamdaP_minus_s = lamdaP - epsilon*Ebasis[1];
        Vector3d lamdaP_plus_n = lamdaP + epsilon*Ebasis[2];
        Vector3d lamdaP_minus_n = lamdaP - epsilon*Ebasis[2];
        //
        // fluxes and sources (sources used for numerical derivatives)
        Matrix3d SS_plus,SS_minus;
        Vector3d Q_rho_plus,Q_rho_minus;
        Vector3d Q_c_plus,Q_c_minus;
        double S_rho_plus,S_rho_minus;
        double S_c_plus,S_c_minus;
        //
        // phif
        evalFluxesSources(global_parameters,phif_plus,a0,a0_0,s0_0,n0_0,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
        evalFluxesSources(global_parameters,phif_minus,a0,a0_0,s0_0,n0_0,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
        Matrix3d dSSdphif = (1./(2.*epsilon))*(SS_plus-SS_minus);
        Vector3d dQ_rhodphif = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
        Vector3d dQ_cdphif = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
        double dS_rhodphif = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
        double dS_cdphif = (1./(2.*epsilon))*(S_c_plus-S_c_minus);
        //std::cout << "\n dSSdphif: \n" << dSSdphif << "\n dQ_rhodphif: \n" << dQ_rhodphif << "\n dQ_cdphif: \n" << dQ_cdphif << "\n dS_rhodphif: \n" << dS_rhodphif << "\n dS_cdphif: \n" << dS_cdphif << "\n";
        //
        // a0x
        evalFluxesSources(global_parameters,phif,a0_plus_x,a0_0,s0_0,n0_0,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
        evalFluxesSources(global_parameters,phif,a0_minus_x,a0_0,s0_0,n0_0,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
        Matrix3d dSSda0x = (1./(2.*epsilon))*(SS_plus-SS_minus);
        Vector3d dQ_rhoda0x = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
        Vector3d dQ_cda0x = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
        double dS_rhoda0x = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
        double dS_cda0x = (1./(2.*epsilon))*(S_c_plus-S_c_minus);
        //std::cout << "\n dSSda0x: \n" << dSSda0x << "\n dQ_rhoda0x: \n" << dQ_rhoda0x << "\n dQ_cda0x: \n" << dQ_cda0x << "\n dS_rhoda0x: \n" << dS_rhoda0x << "\n dS_cda0x: \n" << dS_cda0x << "\n";
        //
        // a0y
        evalFluxesSources(global_parameters,phif,a0_plus_y,a0_0,s0_0,n0_0,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
        evalFluxesSources(global_parameters,phif,a0_minus_y,a0_0,s0_0,n0_0,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
        Matrix3d dSSda0y = (1./(2.*epsilon))*(SS_plus-SS_minus);
        Vector3d dQ_rhoda0y = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
        Vector3d dQ_cda0y = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
        double dS_rhoda0y = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
        double dS_cda0y = (1./(2.*epsilon))*(S_c_plus-S_c_minus);
        //std::cout << "\n dSSda0y: \n" << dSSda0y << "\n dQ_rhoda0y: \n" << dQ_rhoda0y << "\n dQ_cda0y: \n" << dQ_cda0y << "\n dS_rhoda0y: \n" << dS_rhoda0y << "\n dS_cda0y: \n" << dS_cda0y << "\n";
        //
        // a0z (assume not all principal fiber directions are tangent to surface)
        evalFluxesSources(global_parameters,phif,a0_plus_z,a0_0,s0_0,n0_0,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
        evalFluxesSources(global_parameters,phif,a0_minus_z,a0_0,s0_0,n0_0,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
        Matrix3d dSSda0z = (1./(2.*epsilon))*(SS_plus-SS_minus);
        Vector3d dQ_rhoda0z = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
        Vector3d dQ_cda0z = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
        double dS_rhoda0z = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
        double dS_cda0z = (1./(2.*epsilon))*(S_c_plus-S_c_minus);
        //std::cout << "\n dSSda0z: \n" << dSSda0z << "\n dQ_rhoda0z: \n" << dQ_rhoda0z << "\n dQ_cda0z: \n" << dQ_cda0z << "\n dS_rhoda0z: \n" << dS_rhoda0z << "\n dS_cda0z: \n" << dS_cda0z << "\n";
        //
        // kappa
        evalFluxesSources(global_parameters,phif,a0,a0_0,s0_0,n0_0,kappa_plus,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
        evalFluxesSources(global_parameters,phif,a0,a0_0,s0_0,n0_0,kappa_minus,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
        Matrix3d dSSdkappa = (1./(2.*epsilon))*(SS_plus-SS_minus);
        Vector3d dQ_rhodkappa = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
        Vector3d dQ_cdkappa = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
        double dS_rhodkappa = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
        double dS_cdkappa = (1./(2.*epsilon))*(S_c_plus-S_c_minus);
        //std::cout << "\n dSSdkappa: \n" << dSSdkappa << "\n dQ_rhodkappa: \n" << dQ_rhodkappa << "\n dQ_cdkappa: \n" << dQ_cdkappa << "\n dS_rhodkappa: \n" << dS_rhodkappa << "\n dS_cdkappa: \n" << dS_cdkappa << "\n";
        //
        // lamdaP_a
        evalFluxesSources(global_parameters,phif,a0,a0_0,s0_0,n0_0,kappa,lamdaP_plus_a,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
        evalFluxesSources(global_parameters,phif,a0,a0_0,s0_0,n0_0,kappa,lamdaP_minus_a,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
        Matrix3d dSSdlamdaPa = (1./(2.*epsilon))*(SS_plus-SS_minus);
        Vector3d dQ_rhodlamdaPa = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
        Vector3d dQ_cdlamdaPa = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
        double dS_rhodlamdaPa = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
        double dS_cdlamdaPa = (1./(2.*epsilon))*(S_c_plus-S_c_minus);
        //std::cout << "\n dSSdlamdaPa: \n" << dSSdlamdaPa << "\n dQ_rhodlamdaPa: \n" << dQ_rhodlamdaPa << "\n dQ_cdlamdaPa: \n" << dQ_cdlamdaPa << "\n dS_rhodlamdaPa: \n" << dS_rhodlamdaPa << "\n dS_cdlamdaPa: \n" << dS_cdlamdaPa << "\n";
        //
        // lamdaP_s
        evalFluxesSources(global_parameters,phif,a0,a0_0,s0_0,n0_0,kappa,lamdaP_plus_s,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
        evalFluxesSources(global_parameters,phif,a0,a0_0,s0_0,n0_0,kappa,lamdaP_minus_s,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
        Matrix3d dSSdlamdaPs = (1./(2.*epsilon))*(SS_plus-SS_minus);
        Vector3d dQ_rhodlamdaPs = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
        Vector3d dQ_cdlamdaPs = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
        double dS_rhodlamdaPs = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
        double dS_cdlamdaPs = (1./(2.*epsilon))*(S_c_plus-S_c_minus);
        //std::cout << "\n dSSdlamdaPs: \n" << dSSdlamdaPs << "\n dQ_rhodlamdaPs: \n" << dQ_rhodlamdaPs << "\n dQ_cdlamdaPs: \n" << dQ_cdlamdaPs << "\n dS_rhodlamdaPs: \n" << dS_rhodlamdaPs << "\n dS_cdlamdaPs: \n" << dS_cdlamdaPs << "\n";
        //
        // lamdaP_N (assume change in height of wound)
        evalFluxesSources(global_parameters,phif,a0,a0_0,s0_0,n0_0,kappa,lamdaP_plus_n,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
        evalFluxesSources(global_parameters,phif,a0,a0_0,s0_0,n0_0,kappa,lamdaP_minus_n,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
        Matrix3d dSSdlamdaPn = (1./(2.*epsilon))*(SS_plus-SS_minus);
        Vector3d dQ_rhodlamdaPn = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
        Vector3d dQ_cdlamdaPn = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
        double dS_rhodlamdaPn = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
        double dS_cdlamdaPn = (1./(2.*epsilon))*(S_c_plus-S_c_minus);
        //std::cout << "\n dSSdlamdaPn: \n" << dSSdlamdaPn << "\n dQ_rhodlamdaPn: \n" << dQ_rhodlamdaPn << "\n dQ_cdlamdaPn: \n" << dQ_cdlamdaPn << "\n dS_rhodlamdaPn: \n" << dS_rhodlamdaPn << "\n dS_cdlamdaPn: \n" << dS_cdlamdaPn << "\n";
        //---------------------------------//


        //---------------------------------//
        // MECHANICS TANGENT
        //
        double Psif11 = 2*k2*kappa*kappa*Psif+2*k2*kappa*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif1 ;
        // NOTE: no I4e factor here. Psif14 and Psif41 are both d2Psi/dI1e dI4e
        // and must be equal by symmetry of second derivatives; writing E for
        // (kappa*I1e + (1-3kappa)*I4e - 1), both evaluate to
        //     2*k2*kappa*(1-3kappa)*Psif*(1 + 2*k2*E*E).
        // Psif14 carried a spurious I4e on its first term (about 1.2 in the
        // healthy state), breaking that symmetry and leaving Ke_x_x
        // inconsistent with the residual. Caught by tests/test_tangent: the
        // Ke_x_x error sat flat at 1.3e-2 across an 80-fold sweep of the inner
        // substep size, so it could not be a discretisation artifact.
        double Psif14 = 2*k2*kappa*(1-3*kappa)*Psif + 2*k2*kappa*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif4;
        double Psif41 = 2*k2*(1-3*kappa)*kappa*Psif + 2*k2*(1-3*kappa)*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif1;
        double Psif44 = 2*k2*(1-3*kappa)*(1-3*kappa)*Psif + 2*k2*(1-3*kappa)*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif4;
        std::vector<double> dSSpasdCC_explicit(81,0.);
        std::vector<double> dSSvoldCC_explicit(81,0.);
        for(int ii=0;ii<3;ii++){
            for(int jj=0;jj<3;jj++){
                for(int kk=0;kk<3;kk++){
                    for(int ll=0;ll<3;ll++){
                        for(int pp=0;pp<3;pp++){
                            for(int rr=0;rr<3;rr++){
                                for(int ss=0;ss<3;ss++){
                                    for(int tt=0;tt<3;tt++){
                                        // Explicit passive mechanics tangent
                                        dSSpasdCC_explicit[ii*27+jj*9+kk*3+ll] += Jp*(phif*(Psif11*Identity(pp,rr)*Identity(ss,tt) +
                                                Psif14*Identity(pp,rr)*a0a0(ss,tt) + Psif41*a0a0(pp,rr)*Identity(ss,tt) +
                                                Psif44*a0a0(pp,rr)*a0a0(ss,tt)))*FFginv(ii,pp)*FFginv(jj,rr)*FFginv(kk,ss)*FFginv(ll,tt);

                                        // Explicit volumetric mechanics tangent
                                        dSSvoldCC_explicit[ii*27+jj*9+kk*3+ll] += Jp/4*(-2*Je*dPsivoldJe*(0.5*(CCeinv(pp,ss)*CCeinv(rr,tt)+CCeinv(pp,tt)*CCeinv(rr,ss)))
                                                +Je*(dPsivoldJe+Je*dPsivoldJedJe)*CCeinv(pp,rr)*CCeinv(ss,tt))*FFginv(ii,pp)*FFginv(jj,rr)*FFginv(kk,ss)*FFginv(ll,tt);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        //--------------------------------------------------//
        // build DD, the voigt version of CCCC = dSS_dCC
        // voigt table
        VectorXd voigt_table_I_i(6);
        voigt_table_I_i(0) = 0; voigt_table_I_i(1) = 1; voigt_table_I_i(2) = 2;
        voigt_table_I_i(3) = 1; voigt_table_I_i(4) = 2; voigt_table_I_i(5) = 0;
        VectorXd voigt_table_I_j(6);
        voigt_table_I_j(0) = 0; voigt_table_I_j(1) = 1; voigt_table_I_j(2) = 2;
        voigt_table_I_j(3) = 2; voigt_table_I_j(4) = 0; voigt_table_I_j(5) = 1;
        //VectorXd voigt_table_J_k(6);
        //voigt_table_J_k(0) = 0; voigt_table_J_k(1) = 1; voigt_table_J_k(2) = 2; /// These are identical to above
        //voigt_table_J_k(3) = 1; voigt_table_J_k(4) = 2; voigt_table_J_k(5) = 0;
        //VectorXd voigt_table_J_l(6);
        //voigt_table_J_l(0) = 0; voigt_table_J_l(1) = 1; voigt_table_J_l(2) = 2;
        //voigt_table_J_l(3) = 2; voigt_table_J_l(4) = 0; voigt_table_J_l(5) = 1;
        // some things needed first
        // only derivatives with respect to CC explicitly
        Matrix3d dtrAdCC = kappa*Identity + (1-3*kappa)*a0a0;
        MatrixXd DDvol(6,6),DDpas(6,6),DDact(6,6),DDstruct(6,6),DDtot(6,6);
        DDvol.setZero(); DDpas.setZero(); DDact.setZero(); DDstruct.setZero(); DDtot.setZero();

        //--------------------------------------------------//
        // CHECKING
//        Matrix3d CC_p,CC_m,SSpas_p,SSpas_m,SSvol_p,SSvol_m,SSact_p,SSact_m;
//        MatrixXd DDvol_num(6,6),DDpas_num(6,6),DDact_num(6,6),DDtot_num(6,6);
//        DDvol_num.setZero(); DDpas_num.setZero(); DDact_num.setZero(); DDtot_num.setZero();
        //--------------------------------------------------//

        for(int II=0;II<6;II++){
            for(int JJ=0;JJ<6;JJ++){
                int ii = voigt_table_I_i(II);
                int jj = voigt_table_I_j(II);
                int kk = voigt_table_I_i(JJ);
                int ll = voigt_table_I_j(JJ);

                // pressure, explicit  only
                DDvol(II,JJ) = dSSvoldCC_explicit[ii*27+jj*9+kk*3+ll];

                // passive, explicit only
                DDpas(II,JJ) = dSSpasdCC_explicit[ii*27+jj*9+kk*3+ll];

                // active , explicit only
                DDact(II,JJ) = -1.0*phif*traction_act*Jp/(trA*trA*(K_t*K_t+phif*phif))*dtrAdCC(kk,ll)*A0(ii,jj);
                //DDact(II,JJ) = -1.0*phif*traction_act*Jp/(trA*trA)*dtrAdCC(kk,ll)*A0(ii,jj);

                // structural
                DDstruct(II,JJ) = dSSdphif(ii,jj)*dphifdCC(kk,ll) + dSSda0x(ii,jj)*da0xdCC(kk,ll)+ dSSda0y(ii,jj)*da0ydCC(kk,ll) + dSSda0z(ii,jj)*da0zdCC(kk,ll)
                                  +dSSdkappa(ii,jj)*dkappadCC(kk,ll)+dSSdlamdaPa(ii,jj)*dlamdaP_adCC(kk,ll) +dSSdlamdaPs(ii,jj)*dlamdaP_sdCC(kk,ll)+dSSdlamdaPn(ii,jj)*dlamdaP_ndCC(kk,ll);


                //--------------------------------------------------//
                // CHECKING
                // Numerical Solutions
//                CC_p = CC + 0.5*epsilon*Ebasis[kk]*Ebasis[ll].transpose() + 0.5*epsilon*Ebasis[ll]*Ebasis[kk].transpose();
//                CC_m = CC - 0.5*epsilon*Ebasis[kk]*Ebasis[ll].transpose() - 0.5*epsilon*Ebasis[ll]*Ebasis[kk].transpose();
//                evalSS(global_parameters,phif,a0,s0,n0,kappa,lamdaP_a,lamdaP_s,lamdaP_N,CC_p,rho,c,SSpas_p,SSact_p,SSvol_p);
//                evalSS(global_parameters,phif,a0,s0,n0,kappa,lamdaP_a,lamdaP_s,lamdaP_N,CC_m,rho,c,SSpas_m,SSact_m,SSvol_m);
//                DDpas_num(II,JJ) = (1./(2.0*epsilon))*(SSpas_p(ii,jj)-SSpas_m(ii,jj));
//                DDvol_num(II,JJ) = (1./(2.0*epsilon))*(SSvol_p(ii,jj)-SSvol_m(ii,jj));
//                DDact_num(II,JJ) = (1./(2.0*epsilon))*(SSact_p(ii,jj)-SSact_m(ii,jj));
//                DDtot_num(II,JJ) =  DDvol_num(II,JJ) + DDpas_num(II,JJ) + DDact_num(II,JJ);
                //--------------------------------------------------//

                // TOTAL. now include the structural parameters
                DDtot(II,JJ) =  DDvol(II,JJ) + DDpas(II,JJ) + DDact(II,JJ) + DDstruct(II,JJ);
            }
        }

        //--------------------------------------------------//
        // CHECKING
        // PRINT ALL
        //std::cout<<"constitutive: \nDDpas\n"<<DDpas<<"\nDD_act\n"<<DDact<<"\nDD_struct"<<DDstruct<<"\n";
        // PRINT COMPARISON WITH NUMERICAL
//        if(fabs(Jp-1) > 0){
//            std::cout<<"comparing\nDD_pas\n";
//            std::cout<<DDpas<<"\nDD_pas_num\n"<<DDpas_num<<"\n";
//            std::cout<<"comparing\nDD_vol\n";
//            std::cout<<DDvol<<"\nDD_vol_num\n"<<DDvol_num<<"\n";
//            std::cout<<"comparing\nDD_act\n";
//            std::cout<<DDact<<"\nDD_act_num\n"<<DDact_num<<"\n";
//            std::cout<<"comparing\nDD_tot\n";
//            std::cout<<DDtot<<"\nDD_tot_num\n"<<DDtot_num<<"\n";
//        }
        //
        //--------------------------------------------------//

        // Derivatives for rho and c
        //
        double dtractiondrho = (t_rho + t_rho_c*c/(K_t_c + c));
        //Matrix3d dSSdrho_explicit = (Jp*dtractiondrho*phif/trA)*(kappa*Identity+(1-3*kappa)*a0a0);
        Matrix3d dSSdrho_explicit = (Jp*dtractiondrho*phif/(trA*(K_t*K_t+phif*phif)))*(kappa*Identity+(1-3*kappa)*a0a0);
        Matrix3d dSSdrho = dSSdrho_explicit + dSSdphif*dphifdrho + dSSda0x*da0xdrho + dSSda0y*da0ydrho + dSSda0z*da0zdrho
                + dSSdkappa*dkappadrho + dSSdlamdaPa*dlamdaP_adrho + dSSdlamdaPs*dlamdaP_sdrho + dSSdlamdaPn*dlamdaP_ndrho;
        VectorXd dSSdrho_voigt(6);
        dSSdrho_voigt(0) = dSSdrho(0,0); dSSdrho_voigt(1) = dSSdrho(1,1); dSSdrho_voigt(2) = dSSdrho(2,2);
        dSSdrho_voigt(3) = dSSdrho(1,2); dSSdrho_voigt(4) = dSSdrho(0,2); dSSdrho_voigt(5) = dSSdrho(0,1);
        //Matrix3d dsigma_actdc = phif*(t_rho_c/(K_t_c + c)-t_rho_c*c/pow((K_t_c + c),2))*rho*hat_A;
        double dtractiondc = (t_rho_c/(K_t_c + c)-t_rho_c*c/pow((K_t_c + c),2))*rho;
        //Matrix3d dSSdc_explicit = (Jp*dtractiondc*phif/trA)*(kappa*Identity+(1-3*kappa)*a0a0);
        Matrix3d dSSdc_explicit = (Jp*dtractiondc*phif/(trA*(K_t*K_t+phif*phif)))*(kappa*Identity+(1-3*kappa)*a0a0);
        Matrix3d dSSdc = dSSdc_explicit + dSSdphif*dphifdc + dSSda0x*da0xdc + dSSda0y*da0ydc + dSSda0z*da0zdc
                         + dSSdkappa*dkappadc + dSSdlamdaPa*dlamdaP_adc + dSSdlamdaPs*dlamdaP_sdc + dSSdlamdaPn*dlamdaP_ndc;
        VectorXd dSSdc_voigt(6);
        dSSdc_voigt(0) = dSSdc(0,0); dSSdc_voigt(1) = dSSdc(1,1); dSSdc_voigt(2) = dSSdc(2,2);
        dSSdc_voigt(3) = dSSdc(1,2); dSSdc_voigt(4) = dSSdc(0,2); dSSdc_voigt(5) = dSSdc(0,1);
        //
        // some other declared variables
        Matrix3d linFF,linCC,lindeltaCC;
        VectorXd linCC_voigt(6),lindeltaCC_voigt(6);
        //
        // Loop over nodes and coordinates twice and assemble the corresponding entry
        for(int nodei=0;nodei<elem_size;nodei++){
            for(int coordi=0;coordi<3;coordi++){
                deltaFF = Ebasis[coordi]*Grad_R[nodei].transpose();
                deltaCC = deltaFF.transpose()*FF + FF.transpose()*deltaFF;
                //VectorXd deltaCC_voigt(6);
                deltaCC_voigt(0) = deltaCC(0,0); deltaCC_voigt(1) = deltaCC(1,1); deltaCC_voigt(2) = deltaCC(2,2);
                deltaCC_voigt(3) = 2.*deltaCC(1,2); deltaCC_voigt(4) = 2.*deltaCC(0,2); deltaCC_voigt(5) = 2.*deltaCC(0,1);
                for(int nodej=0;nodej<elem_size;nodej++){
                    for(int coordj=0;coordj<3;coordj++){

                        //-----------//
                        // Ke_X_X
                        //-----------//

                        // material part of the tangent
                        linFF =  Ebasis[coordj]*Grad_R[nodej].transpose();
                        linCC = linFF.transpose()*FF + FF.transpose()*linFF;
                        linCC_voigt(0) = linCC(0,0); linCC_voigt(1) = linCC(1,1); linCC_voigt(2) = linCC(2,2);
                        linCC_voigt(3) = 2.*linCC(1,2); linCC_voigt(4) = 2.*linCC(0,2); linCC_voigt(5) = 2.*linCC(0,1);
                        //
                        Ke_x_x(nodei*3+coordi,nodej*3+coordj) += Jac*deltaCC_voigt.dot(DDtot*linCC_voigt)*wip;
                        //
                        // geometric part of the tangent
                        lindeltaCC = deltaFF.transpose()*linFF + linFF.transpose()*deltaFF;
                        lindeltaCC_voigt(0) = lindeltaCC(0,0); lindeltaCC_voigt(1) = lindeltaCC(1,1); lindeltaCC_voigt(2) = lindeltaCC(2,2);
                        lindeltaCC_voigt(3) = 2.*lindeltaCC(1,2); lindeltaCC_voigt(4) = 2.*lindeltaCC(0,2); lindeltaCC_voigt(5) = 2.*lindeltaCC(0,1);
                        //
                        Ke_x_x(nodei*3+coordi,nodej*3+coordj) += Jac*SS_voigt.dot(lindeltaCC_voigt)*wip;

                    }

                    //-----------//
                    // Ke_x_rho
                    //-----------//

                    Ke_x_rho(nodei*3+coordi,nodej) += Jac*dSSdrho_voigt.dot(deltaCC_voigt)*R[nodej]*wip;

                    //-----------//
                    // Ke_x_c
                    //-----------//

                    Ke_x_c(nodei*3+coordi,nodej) += Jac*dSSdc_voigt.dot(deltaCC_voigt)*R[nodej]*wip;

                }
            }
        }

        //-----------------//
        // RHO and C
        //-----------------//

        // Derivatives of flux and source terms wrt rho and C
        // explicit linearizations. In this case no dependence on structural parameters.
        //Matrix3d linQ_rhodGradrho = -3.0*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0/trA;
        //Vector3d linQ_rhodrho = -3.0*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*A0*Grad_c/trA;
        //Matrix3d linQ_rhodGradc = -3.0*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0/trA;
        //Matrix3d linQ_cdGradc = -1.0*(D_cc-phif*(D_cc-D_cc/10))*CCinv;
        //Matrix3d linQ_cdGradc = -3*(D_cc-phif*(D_cc-D_cc/10))*A0/trA;
        Matrix3d linQ_rhodGradrho = -D_rhorho*CCinv;
        Vector3d linQ_rhodrho = -D_rhoc*CCinv*Grad_c;
        Matrix3d linQ_rhodGradc = -D_rhoc*rho*CCinv;
        Matrix3d linQ_cdGradc = -D_cc*CCinv;
        // alpha: Q_alpha = -D_alpha CCinv Grad_alpha with D_alpha constant, so
        // the only field linearization is with respect to Grad_alpha.
        Matrix3d linQ_alphadGradalpha = -D_alpha*CCinv;
        double   dS_alphadalpha = -d_alpha;
        double   dS_cdalpha     = p_c_alpha;
        //
        // explicit derivatives of source terms
        double dS_rhodrho_explicit = (p_rho + p_rho_c*c/(K_rho_c+c)+p_rho_theta*He)*(1-rho/K_rho_rho) - d_rho + rho*(p_rho + p_rho_c*c/(K_rho_c+c)+p_rho_theta*He)*(-1./K_rho_rho);
        double dS_rhodc_explicit = (1-rho/K_rho_rho)*rho*(p_rho_c/(K_rho_c+c) - p_rho_c*c/((K_rho_c+c)*(K_rho_c+c)));
        double dS_cdrho_explicit = (p_c_rho*c + p_c_thetaE*He)*(1./(K_c_c+c));
        double dS_cdc_explicit = -d_c + (p_c_rho*c + p_c_thetaE*He)*(-rho/((K_c_c+c)*(K_c_c+c))) + (rho/(K_c_c+c))*p_c_rho;
        // total derivatives
        double dS_rhodrho = dS_rhodrho_explicit + dS_rhodphif*dphifdrho + dS_rhoda0x*da0xdrho + dS_rhoda0y*da0ydrho + dS_rhoda0z*da0zdrho
                            +dS_rhodkappa*dkappadrho+dS_rhodlamdaPa*dlamdaP_adrho+dS_rhodlamdaPs*dlamdaP_sdrho+dS_rhodlamdaPn*dlamdaP_ndrho;
        double dS_rhodc = dS_rhodc_explicit + dS_rhodphif*dphifdc + dS_rhoda0x*da0xdc + dS_rhoda0y*da0ydc + dS_rhoda0z*da0zdc
                          +dS_rhodkappa*dkappadc+dS_rhodlamdaPa*dlamdaP_adc+dS_rhodlamdaPs*dlamdaP_sdc+dS_rhodlamdaPn*dlamdaP_ndc;
        double dS_cdrho = dS_cdrho_explicit + dS_cdphif*dphifdrho + dS_cda0x*da0xdrho + dS_cda0y*da0ydrho + dS_cda0z*da0zdrho
                          +dS_cdkappa*dkappadrho+dS_cdlamdaPa*dlamdaP_adrho+dS_cdlamdaPs*dlamdaP_sdrho+dS_cdlamdaPn*dlamdaP_ndrho;
        double dS_cdc = dS_cdc_explicit + dS_cdphif*dphifdc + dS_cda0x*da0xdc + dS_cda0y*da0ydc + dS_cda0z*da0zdc
                        +dS_cdkappa*dkappadc+dS_cdlamdaPa*dlamdaP_adc+dS_cdlamdaPs*dlamdaP_sdc+dS_cdlamdaPn*dlamdaP_ndc;
        // wrt Mechanics
        // SOURCE TERMS
        Matrix3d dHedCC_explicit; dHedCC_explicit.setZero();
        //
        // dH/dCC = gamma_e H (1-H) dtheta_e/dCC  (was dJe/dCC = J CCinv/(2 Jp))
        dHedCC_explicit = evalDHedCC(theta_e, He, gamma_theta, CCinv, n0);
        Matrix3d dS_rhodCC_explicit = (1-rho/K_rho_rho)*rho*p_rho_theta*dHedCC_explicit;
        VectorXd dS_rhodCC_voigt(6);
        Matrix3d dS_cdCC_explicit = (rho / (K_c_c + c)) * (p_c_thetaE * dHedCC_explicit);
        VectorXd dS_cdCC_voigt(6);
        // FLUX TERMS
        std::vector<double> dQ_rhodCC_explicit(27,0.);
        std::vector<double> dQ_cdCC_explicit(27,0.);
        std::vector<double> dQ_alphadCC_explicit(27,0.);
        for(int ii=0;ii<3;ii++) {
            for(int jj=0;jj<3;jj++) {
                for(int kk=0;kk<3;kk++) {
                    for(int ll=0;ll<3;ll++) {
                        // These are third order tensors, but there are two contractions from matrix multiplication

                        dQ_rhodCC_explicit[ii*9+kk*3+ll] += -1.0*(-0.5)*D_rhorho*(CCinv(ii,kk)*CCinv(jj,ll)+CCinv(jj,kk)*CCinv(ii,ll))*Grad_rho(jj)
                                                            -1.0*(-0.5)*D_rhoc*rho*(CCinv(ii,kk)*CCinv(jj,ll)+CCinv(jj,kk)*CCinv(ii,ll))*Grad_c(jj);

                        //dQ_rhodCC_explicit[ii*9+kk*3+ll] += -1.0*(-3*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0(ii,jj)*Grad_rho(jj)
                        //        - 3*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0(ii,jj)*Grad_c(jj))*dtrAdCC(kk,ll) / (trA*trA);

                        // D_cc, NOT the collagen-dependent (D_cc - phif*(D_cc - D_cc/10)).
                        // The residual uses a constant cytokine diffusivity -
                        //     Q_c = -D_cc*CCinv*Grad_c
                        // - and the collagen-dependent form sits commented out
                        // directly beneath it. The tangent was never updated to
                        // match, so at healthy collagen (phif = 1) it used
                        // D_cc/10: ten times too small. Measured by
                        // tests/test_tangent as Ke_c_x carrying an error an
                        // order of magnitude LARGER than the block itself,
                        // because the flux contribution was almost entirely
                        // absent while the source contribution was correct.
                        dQ_cdCC_explicit[ii*9+kk*3+ll] += -0.5*(-1.0*D_cc)*(CCinv(ii,kk)*CCinv(jj,ll)+CCinv(jj,kk)*CCinv(ii,ll))*Grad_c(jj);

                        // alpha: Q_alpha = -D_alpha CCinv Grad_alpha, same form
                        dQ_alphadCC_explicit[ii*9+kk*3+ll] += -0.5*(-1.0*D_alpha)*(CCinv(ii,kk)*CCinv(jj,ll)+CCinv(jj,kk)*CCinv(ii,ll))*Grad_alpha(jj);

                        //dQ_cdCC_explicit[ii*9+kk*3+ll] += -1.0*(-3*(D_cc-phif*(D_cc-D_cc/10))*A0(ii,jj)*Grad_c(jj))
                        //       *dtrAdCC(kk,ll)/(trA*trA);

                        //dQ_cdCC_explicit[ii*9+kk*3+ll] += -1.0*(-0.5)*D_cc*(CCinv(ii,kk)*CCinv(jj,ll)+CCinv(jj,kk)*CCinv(ii,ll))*Grad_c(jj);
                    }
                }
            }
        }
        // Put into a Voigt form (3x6)
        MatrixXd dQ_rhodCC_voigt(3,6); dQ_rhodCC_voigt.setZero();
        MatrixXd dQ_rhodCC_explicit_voigt(3,6); dQ_rhodCC_explicit_voigt.setZero();
        MatrixXd dQ_rhodCC_structural_voigt(3,6); dQ_rhodCC_structural_voigt.setZero();
        MatrixXd dQ_cdCC_voigt(3,6); dQ_cdCC_voigt.setZero();
        MatrixXd dQ_alphadCC_voigt(3,6); dQ_alphadCC_voigt.setZero();
        MatrixXd dQ_cdCC_explicit_voigt(3,6); dQ_cdCC_explicit_voigt.setZero();
        MatrixXd dQ_cdCC_structural_voigt(3,6); dQ_cdCC_structural_voigt.setZero();
        VectorXd dS_rhodCC_explicit_voigt(6); dS_rhodCC_explicit_voigt.setZero();
        VectorXd dS_rhodCC_structural_voigt(6); dS_rhodCC_structural_voigt.setZero();
        VectorXd dS_cdCC_explicit_voigt(6); dS_cdCC_explicit_voigt.setZero();
        VectorXd dS_cdCC_structural_voigt(6); dS_cdCC_structural_voigt.setZero();
        //--------------------------------------------------//
        // CHECKING
//        MatrixXd dQ_rhodCC_voigt_num(3,6); dQ_rhodCC_voigt_num.setZero();
//        MatrixXd dQ_cdCC_voigt_num(3,6); dQ_cdCC_voigt_num.setZero();
//        Vector3d Q_rho_p; Q_rho_p.setZero(); Vector3d Q_rho_m; Q_rho_m.setZero();
//        Vector3d Q_c_p; Q_c_p.setZero(); Vector3d Q_c_m; Q_c_m.setZero();
//        VectorXd dS_rhodCC_voigt_num(6); dS_rhodCC_voigt_num.setZero();
//        VectorXd dS_cdCC_voigt_num(6); dS_cdCC_voigt_num.setZero();
//        double S_rho_p, S_rho_m, S_c_p, S_c_m;
        //--------------------------------------------------//
        for(int II=0;II<3;II++){
            for(int JJ=0;JJ<6;JJ++) {
                // We can use the same Voigt tables, but only need three of them,
                // since we have contracted on jj, we will only take the first three entries of II
                int ii = voigt_table_I_i(II);
                int kk = voigt_table_I_i(JJ);
                int ll = voigt_table_I_j(JJ);

                dQ_rhodCC_explicit_voigt(II,JJ) = dQ_rhodCC_explicit[ii*9+kk*3+ll];

                dQ_rhodCC_structural_voigt(II,JJ) = dQ_rhodphif(ii)*dphifdCC(kk,ll)
                        + dQ_rhoda0x(ii)*da0xdCC(kk,ll) + dQ_rhoda0y(ii)*da0ydCC(kk,ll) + dQ_rhoda0z(ii)*da0zdCC(kk,ll)
                            +dQ_rhodkappa(ii)*dkappadCC(kk,ll) + dQ_rhodlamdaPa(ii)*dlamdaP_adCC(kk,ll)
                            + dQ_rhodlamdaPs(ii)*dlamdaP_sdCC(kk,ll) + dQ_rhodlamdaPn(ii)*dlamdaP_ndCC(kk,ll);

                dQ_rhodCC_voigt(II,JJ) = dQ_rhodCC_explicit_voigt(II,JJ) + dQ_rhodCC_structural_voigt(II,JJ);

                dQ_cdCC_explicit_voigt(II,JJ) = dQ_cdCC_explicit[ii*9+kk*3+ll];

                dQ_cdCC_structural_voigt(II,JJ) = dQ_cdphif(ii)*dphifdCC(kk,ll)
                        + dQ_cda0x(ii)*da0xdCC(kk,ll) + dQ_cda0y(ii)*da0ydCC(kk,ll) + dQ_cda0z(ii)*da0zdCC(kk,ll)
                          +dQ_cdkappa(ii)*dkappadCC(kk,ll) + dQ_cdlamdaPa(ii)*dlamdaP_adCC(kk,ll)
                          + dQ_cdlamdaPs(ii)*dlamdaP_sdCC(kk,ll) + dQ_cdlamdaPn(ii)*dlamdaP_ndCC(kk,ll);

                dQ_cdCC_voigt(II,JJ) = dQ_cdCC_explicit_voigt(II,JJ) + dQ_cdCC_structural_voigt(II,JJ);
                // alpha has no phif/a0/kappa/lamdaP dependence, so there is no
                // structural contribution here - only the explicit CC one.
                dQ_alphadCC_voigt(II,JJ) = dQ_alphadCC_explicit[ii*9+kk*3+ll];

                dS_rhodCC_explicit_voigt(JJ) = dS_rhodCC_explicit(kk,ll);

                dS_rhodCC_structural_voigt(JJ) = dS_rhodphif*dphifdCC(kk,ll) + dS_rhoda0x*da0xdCC(kk,ll) + dS_rhoda0y*da0ydCC(kk,ll) + dS_rhoda0z*da0zdCC(kk,ll)
                                  + dS_rhodkappa*dkappadCC(kk,ll) + dS_rhodlamdaPa*dlamdaP_adCC(kk,ll) + dS_rhodlamdaPs*dlamdaP_sdCC(kk,ll) + dS_rhodlamdaPn*dlamdaP_ndCC(kk,ll);

                dS_rhodCC_voigt(JJ) = dS_rhodCC_explicit_voigt(JJ) + dS_rhodCC_structural_voigt(JJ);

                dS_cdCC_explicit_voigt(JJ) = dS_cdCC_explicit(kk,ll);

                dS_cdCC_structural_voigt(JJ) = dS_cdphif*dphifdCC(kk,ll) + dS_cda0x*da0xdCC(kk,ll) + dS_cda0y*da0ydCC(kk,ll) + dS_cda0z*da0zdCC(kk,ll)
                                + dS_cdkappa*dkappadCC(kk,ll) + dS_cdlamdaPa*dlamdaP_adCC(kk,ll) + dS_cdlamdaPs*dlamdaP_sdCC(kk,ll) + dS_cdlamdaPn*dlamdaP_ndCC(kk,ll);

                dS_cdCC_voigt(JJ) = dS_cdCC_explicit_voigt(JJ) + dS_cdCC_structural_voigt(JJ);

                //--------------------------------------------------//
                // CHECKING
                // Numerical Solutions
//                CC_p = CC + 0.5*epsilon*Ebasis[kk]*Ebasis[ll].transpose() + 0.5*epsilon*Ebasis[ll]*Ebasis[kk].transpose();
//                CC_m = CC - 0.5*epsilon*Ebasis[kk]*Ebasis[ll].transpose() - 0.5*epsilon*Ebasis[ll]*Ebasis[kk].transpose();
//                evalQ(global_parameters,phif,a0,s0,n0,kappa,lamdaP,CC_p,rho,c,Grad_rho,Grad_c,Q_rho_p,Q_c_p);
//                evalQ(global_parameters,phif,a0,s0,n0,kappa,lamdaP,CC_m,rho,c,Grad_rho,Grad_c,Q_rho_m,Q_c_m);
//                evalS(global_parameters,phif,a0,s0,n0,kappa,lamdaP,CC_p,rho,c,S_rho_p,S_c_p);
//                evalS(global_parameters,phif,a0,s0,n0,kappa,lamdaP,CC_m,rho,c,S_rho_m,S_c_m);
//                dQ_rhodCC_voigt_num(II,JJ) = (1./(2.0*epsilon))*(Q_rho_p(ii)-Q_rho_m(ii));
//                dQ_cdCC_voigt_num(II,JJ) = (1./(2.0*epsilon))*(Q_c_p(ii)-Q_c_m(ii));
//                dS_rhodCC_voigt_num(JJ) = (1./(2.0*epsilon))*(S_rho_p-S_rho_m);
//                dS_cdCC_voigt_num(JJ) = (1./(2.0*epsilon))*(S_c_p-S_c_m);
                //--------------------------------------------------//
            }
        }

        //--------------------------------------------------//
        // CHECKING
        // PRINT COMPARISON WITH NUMERICAL
//        if(fabs(Jp-1) > 0){
//            std::cout<<"\ncomparing\ndQ_rhodCC_voigt\n";
//            std::cout<<dQ_rhodCC_explicit_voigt<<"\ndQ_rhodCC_voigt_num\n"<<dQ_rhodCC_voigt_num<<"\n";
//            std::cout<<"comparing\ndQ_cdCC_voigt\n";
//            std::cout<<dQ_cdCC_explicit_voigt<<"\ndQ_cdCC_voigt_num\n"<<dQ_cdCC_voigt_num<<"\n";
//            std::cout<<"comparing\ndS_rhodCC_voigt\n";
//            std::cout<<dS_rhodCC_explicit_voigt<<"\ndS_rhodCC_voigt_num\n"<<dS_rhodCC_voigt_num<<"\n";
//            std::cout<<"comparing\ndS_cdCC_voigt\n";
//            std::cout<<dS_cdCC_explicit_voigt<<"\ndS_cdCC_voigt_num\n"<<dS_cdCC_voigt_num<<"\n";
//        }
        //
        //--------------------------------------------------//

        //std::cout<<"\ndQ_rhodCC_voigt\n"<<dQ_rhodCC_voigt<<"\ndQ_cdCC_voigt\n"<<dQ_cdCC_voigt<<"\n";
        //
        for(int nodei=0;nodei<elem_size;nodei++){
            for(int nodej=0;nodej<elem_size;nodej++){
                for(int coordj=0;coordj<3;coordj++){

                    linFF =  Ebasis[coordj]*Grad_R[nodej].transpose();
                    linCC = linFF.transpose()*FF + FF.transpose()*linFF;
                    VectorXd linCC_voigt(6);
                    linCC_voigt(0) = linCC(0,0); linCC_voigt(1) = linCC(1,1); linCC_voigt(2) = linCC(2,2);
                    linCC_voigt(3) = 2.*linCC(1,2); linCC_voigt(4) = 2.*linCC(0,2); linCC_voigt(5) = 2.*linCC(0,1);
                    //-----------//
                    // Ke_rho_X
                    //-----------//

                    Ke_rho_x(nodei,nodej*3+coordj) += -(R[nodei]*dS_rhodCC_voigt.dot(linCC_voigt) + Grad_R[nodei].dot(dQ_rhodCC_voigt*linCC_voigt))*Jac*wip;

                    //-----------//
                    // Ke_c_X
                    //-----------//

                    Ke_c_x(nodei,nodej*3+coordj) += -(R[nodei]*dS_cdCC_voigt.dot(linCC_voigt) + Grad_R[nodei].dot(dQ_cdCC_voigt*linCC_voigt))*Jac*wip;

                    // Ke_alpha_x: only the flux couples to the deformation,
                    // because s_alpha = -d_alpha alpha has no CC dependence.
                    Ke_alpha_x(nodei,nodej*3+coordj) += -(Grad_R[nodei].dot(dQ_alphadCC_voigt*linCC_voigt))*Jac*wip;

                }

                //-----------//
                // Ke_rho_rho
                //-----------//

                // The flux terms carry a chain rule through the structural
                // update that used to be dropped: D_rho depends on phif, and
                // phif depends on rho and c via the local solver, so
                //   dQ_rho/drho_j += (dQ_rho/dphif)(dphif/drho) R_j
                // The source terms were already fully chain-ruled (see
                // dS_rhodrho above); the fluxes were chain-ruled only through
                // CC, i.e. D was frozen with respect to rho and c. That was
                // survivable with a mild D(phi) but not with the tanh gate in
                // evalDrho(), whose slope reaches ~150 at phi = 0.01.
                Ke_rho_rho(nodei,nodej) += Jac*(R[nodei]*R[nodej]/dt -1.* R[nodei]*dS_rhodrho*R[nodej] -1.* Grad_R[nodei].dot(linQ_rhodGradrho*Grad_R[nodej] + linQ_rhodrho*R[nodej] + dQ_rhodphif*dphifdrho*R[nodej]))*wip;

                //-----------//
                // Ke_rho_c
                //-----------//

                Ke_rho_c(nodei,nodej) += Jac*(-1.*R[nodei]*dS_rhodc*R[nodej] -1.* Grad_R[nodei].dot(linQ_rhodGradc*Grad_R[nodej] + dQ_rhodphif*dphifdc*R[nodej]))*wip;

                //-----------//
                // Ke_c_rho
                //-----------//
                // dQ_cdphif is identically zero while D_cc is constant, but
                // keep the term so the tangent stays consistent if D_c ever
                // becomes collagen-dependent.
                Ke_c_rho(nodei,nodej) += Jac*(-1.*R[nodei]*dS_cdrho*R[nodej] -1.* Grad_R[nodei].dot(dQ_cdphif*dphifdrho*R[nodej]))*wip;

                //-----------//
                // Ke_c_c
                //-----------//

                Ke_c_c(nodei,nodej) += Jac*(R[nodei]*R[nodej]/dt -1.* R[nodei]*dS_cdc*R[nodej] -1.* Grad_R[nodei].dot(linQ_cdGradc*Grad_R[nodej] + dQ_cdphif*dphifdc*R[nodej]))*wip;

                //-----------//
                // Ke_c_alpha : cytokine production driven by alpha
                //-----------//

                Ke_c_alpha(nodei,nodej) += Jac*(-1.*R[nodei]*dS_cdalpha*R[nodej])*wip;

                //-----------//
                // Ke_alpha_alpha
                //-----------//

                Ke_alpha_alpha(nodei,nodej) += Jac*(R[nodei]*R[nodej]/dt -1.*R[nodei]*dS_alphadalpha*R[nodej] -1.*Grad_R[nodei].dot(linQ_alphadGradalpha*Grad_R[nodej]))*wip;

                //-----------//
                // Ke_alpha_rho, Ke_alpha_c : alpha has no rho or c dependence
                // (s_alpha = -d_alpha alpha, Q_alpha = -D_alpha grad alpha), so
                // these blocks stay zero. Ke_x_alpha and Ke_rho_alpha likewise:
                // alpha enters neither the stress nor the fibroblast source.
                //-----------//
            }
        }
    } // END INTEGRATION loop
}


//========================================================//
// EVAL SOURCE AND FLUX
//========================================================//

// Sources and Fluxes are :stress, biological fluxes and sources

void evalFluxesSources(const std::vector<double> &global_parameters, const double& phif,Vector3d a0, const Vector3d& a0_0, const Vector3d& s0_0, const Vector3d& n0_0,double kappa, const Vector3d& lamdaP,
                       const Matrix3d& FF, const double& rho, const double& c, const Vector3d& Grad_rho, const Vector3d& Grad_c,
                       Matrix3d & SS,Vector3d &Q_rho,double &S_rho, Vector3d &Q_c,double &S_c)
{
    double k0 = global_parameters[0]; // neo hookean
    double kf = global_parameters[1]; // stiffness of collagen
    double k2 = global_parameters[2]; // nonlinear exponential
    double t_rho = global_parameters[3]; // force of fibroblasts
    double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
    double K_t = global_parameters[5]; // saturation of collagen on force
    double K_t_c = global_parameters[6]; // saturation of chemical on force
    double D_rhorho = evalDrho(phif, c);
    double D_rhoc = global_parameters[8]; // diffusion of chemotactic gradient
    double D_cc = global_parameters[9]; // diffusion of chemical
    double p_rho =global_parameters[10]; // production of fibroblasts naturally
    double p_rho_c = global_parameters[11]; // production enhanced by the chem
    double p_rho_theta = global_parameters[12]; // mechanosensing
    double K_rho_c= global_parameters[13]; // saturation of cell production by chemical
    double K_rho_rho = global_parameters[14]; // saturation of cell by cell
    double d_rho = global_parameters[15] ;// decay of cells
    double vartheta_e = global_parameters[16]; // physiological state of area stretch
    double gamma_theta = global_parameters[17]; // sensitivity of heviside function
    double p_c_rho = global_parameters[18];// production of C by cells
    double p_c_thetaE = global_parameters[19]; // coupling of elastic and chemical
    double K_c_c = global_parameters[20];// saturation of chem by chem
    double d_c = global_parameters[21]; // decay of chemical

    Matrix3d CC = FF.transpose()*FF;
    // Update kinematics
    Matrix3d CCinv = CC.inverse();

    // Construct a rotation using Rodriguez formula to get the new s0 and n0
    Matrix3d Rot = Matrix3d::Identity();
    if(a0 != a0_0){
        Vector3d across = a0_0.cross(a0);
        double sinacross = sqrt(across.dot(across));
        double cosacross = a0_0.dot(a0);
        Vector3d acrossunit = across/sqrt(across.dot(across));
        Matrix3d askew; askew << 0, -across(2), across(1), across(2), 0, -across(0), -across(1), across(0), 0;
        Rot = Matrix3d::Identity() + askew + (askew*askew)*(1/(1+cosacross));
        //std::cout << "\n" << Rot*Rot.transpose() << "\n";
    }
    Vector3d s0 = Rot*s0_0;
    s0 = s0/sqrt(s0.dot(s0));
    Vector3d n0 = Rot*n0_0;
    n0 = n0/sqrt(n0.dot(n0));

    // 2D Rotation
    //Matrix3d Rot90;Rot90 << 0.,-1.,0., 1.,0.,0., 0.,0.,1.;
    //Vector3d s0 = Rot90*a0;
    //Vector3d n0 = a0.cross(s0);
    //if(n0(2) < 0){
    //    n0 = -n0;
    //}

    // fiber tensor in the reference
    Matrix3d a0a0 = a0*a0.transpose();
    Matrix3d s0s0 = s0*s0.transpose();
    Matrix3d n0n0 = n0*n0.transpose();
    // recompute split
    Matrix3d FFg = lamdaP(0)*(a0a0) + lamdaP(1)*(s0s0) + lamdaP(2)*n0n0;
    Matrix3d FFginv = (1./lamdaP(0))*(a0a0) + (1./lamdaP(1))*(s0s0) + (1./lamdaP(2))*n0n0;
    Matrix3d FFe = FF*FFginv;
    // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
    // elastic strain
    Matrix3d CCe = FFe.transpose()*FFe;
    Matrix3d CCeinv = CCe.inverse();

    // invariant of the elastic strain
    double I1e = CCe(0,0) + CCe(1,1) + CCe(2,2);
    double I4e = a0.dot(CCe*a0);
    double I4tot = a0.dot(CC * a0);

    // Jacobian of the deformations
    double Jp = lamdaP(0)*lamdaP(1)*lamdaP(2);
    double Je = sqrt(CCe.determinant());
    double J = Je*Jp;
    // calculate the normal stretch
    //double thetaP = lamdaP_a*lamdaP_s;
    // Nanson's formula for area change from intermediate to deformed (thetaE) and reference to deformed (theta)
    //double thetaE = sqrt((Je*FFe.inverse().transpose()*n0).dot(Je*FFe.inverse().transpose()*n0));
    //double theta = sqrt((J*FF.inverse().transpose()*n0).dot(J*FF.inverse().transpose()*n0));
    // if the FFg was set up correctly, theta = thetae*thetap should be satisfied
    // double theta = thetaE*thetaP;
    // std::cout<<"split of the determinants. theta = thetaE*thetaB = "<<theta<<" = "<<thetaE<<"*"<<thetaP<<"\n";
    // This should come from CC directly
    //double lamda_N = sqrt(n0.dot(CCe*n0));

    Matrix3d A0 = kappa*Matrix3d::Identity() + (1-3.*kappa)*a0a0;
    Vector3d a = FF*a0;
    Matrix3d A = kappa*FF*FF.transpose() + (1.-3.0*kappa)*a*a.transpose();
    double trA = A(0,0) + A(1,1) + A(2,2);
    Matrix3d hat_A = A/trA;

    //------------------//
    // PASSIVE STRESS
    //------------------//
    // Second Piola Kirchhoff stress tensor
    // passive elastic
    // NOTE: no "-1" here. Psif is not the strain energy itself but the factor
    // (kf/2k2) exp(k2 E^2) whose product with 2 k2 kappa E reproduces
    // dPsi/dI1e = kf kappa E exp(k2 E^2) in Psif1/Psif4 below. Subtracting 1
    // (as this line used to) makes those derivatives inconsistent with the
    // analytic stress in evalWound, which corrupts every finite-difference
    // structural sensitivity computed from this routine and degrades the
    // Newton convergence rate.
    double Psif = (kf/(2.*k2))*(exp( k2*pow((kappa*I1e + (1-3*kappa)*I4e -1),2)));
    double Psif1 = 2*k2*kappa*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif;
    double Psif4 = 2*k2*(1-3*kappa)*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif;
    //Matrix3d SSe_pas = k0*Identity + phif*(Psif1*Identity + Psif4*a0a0);
    Matrix3d SSe_pas = phif*(k0*Matrix3d::Identity() + Psif1*Matrix3d::Identity() + Psif4*a0a0);
    // pull back to the reference,
    Matrix3d SS_pas = Jp*FFginv*SSe_pas*FFginv;
    //------------------//
    // ACTIVE STRESS
    //------------------//
    double traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho;
    Matrix3d SS_act = (Jp*traction_act*phif/(trA*(K_t*K_t+phif*phif)))*A0;
    //Matrix3d SS_act = (Jp*traction_act*phif/trA)*A0;
    //------------------//
    // VOLUME STRESS
    //------------------//
    // Instead of (double pressure = -k0*lamda_N*lamda_N;) directly, add volumetric part of stress SSvol
    double penalty = 0.3166667;
    double Psivol = 0.5*phif*pow(penalty*(Je-1.),2) - 2*phif*k0*log(Je);
    double dPsivoldJe = phif*penalty*(Je-1.) - 2*phif*k0/Je;
    Matrix3d SSe_vol = dPsivoldJe*Je*CCeinv/2;
    Matrix3d SS_vol = Jp*FFginv*SSe_vol*FFginv;
    //------------------//
    // TOTAL STRESS
    //------------------//
    // std::cout<<"stresses.\nSSpas\n"<<SS_pas<<"\nSS_act\n"<<SS_act<<"\nSS_vol"<<SS_vol<<"\n";
    SS = SS_pas + SS_vol + SS_act;
    //SS = (SS_pas + SS_vol) + SS_act;
    VectorXd SS_voigt(6);
    SS_voigt(0) = SS(0,0); SS_voigt(1) = SS(1,1); SS_voigt(2) = SS(2,2);
    SS_voigt(3) = SS(1,2); SS_voigt(4) = SS(0,2); SS_voigt(5) = SS(0,1);

    //------------------//
    // FLUX
    //------------------//
    // Flux and Source terms for the rho and the C
    Q_rho = -D_rhorho*CCinv*Grad_rho - D_rhoc*CCinv*Grad_c;
    //Q_rho = -3*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0*Grad_rho/trA - 3*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0*Grad_c/trA;
    //Q_c = -3*(D_cc-phif*(D_cc-D_cc/10))*A0*Grad_c/trA;
    //Q_c = -(D_cc-phif*(D_cc-D_cc/10))*CCinv*Grad_c;
    Q_c = -D_cc*CCinv*Grad_c;

    //------------------//
    // SOURCE
    //------------------//
    // Mechanosensing stimulus: in-plane AREAL elastic stretch of the dural
    // mid-surface, theta_e = ||cof(F^e).n0||, NOT det(F^e). The tissue is
    // treated as incompressible so det(F^e)==1 and could never respond to
    // membrane stretch. See include/mechanosensing.h.
    double theta_e = evalThetaE(J, CCinv, n0, lamdaP(0), lamdaP(1));
    double He = evalHe(theta_e, vartheta_e, gamma_theta);

    // function for elastic response of the cells
    S_rho = (p_rho + p_rho_c*c/(K_rho_c+c)+p_rho_theta*He)*(1-rho/K_rho_rho)*rho - d_rho*rho;
    // function for elastic response of the chemical
    S_c = (p_c_rho*c+ p_c_thetaE*He)*(rho/(K_c_c+c)) - d_c*c;
    //std::cout<<"flux of celss, Q _rho\n"<<Q_rho<<"\n";
    //std::cout<<"source of cells, S_rho: "<<S_rho<<"\n";
    //std::cout<<"flux of chemical, Q _c\n"<<Q_c<<"\n";
    //std::cout<<"source of chemical, S_c: "<<S_c<<"\n";
    //---------------------------------//
}



//--------------------------------------------------------//
// PRINTING ROUTINES
//--------------------------------------------------------//
//
// The point of these functions is to print stuff. For instance for an element I can print the
// average at the center of the stress or other fields
// ELEMENT RESIDUAL AND TANGENT
Matrix3d evalWoundFF(
        double dt,
        const std::vector<Matrix3d> &ip_Jac,
        const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
        const std::vector<double> &node_X,
        std::vector<double> &ip_phif, std::vector<Vector3d> &ip_a0, std::vector<double> &ip_kappa, std::vector<Vector3d> &ip_lamdaP,
        const std::vector<Vector3d> &node_x,const std::vector<double> &node_rho, const std::vector<double> &node_c,
        double xi, double eta, double zeta)
{
    // return the deformation gradient
    //---------------------------------//
    // PARAMETERS
    //
    double k0 = global_parameters[0]; // neo hookean
    double kf = global_parameters[1]; // stiffness of collagen
    double k2 = global_parameters[2]; // nonlinear exponential
    double t_rho = global_parameters[3]; // force of fibroblasts
    double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
    double K_t = global_parameters[5]; // saturation of collagen on force
    double K_t_c = global_parameters[6]; // saturation of chemical on force
    double D_rhoc = global_parameters[8]; // diffusion of chemotactic gradient
    double D_cc = global_parameters[9]; // diffusion of chemical
    double p_rho =global_parameters[10]; // production of fibroblasts naturally
    double p_rho_c = global_parameters[11]; // production enhanced by the chem
    double p_rho_theta = global_parameters[12]; // mechanosensing
    double K_rho_c= global_parameters[13]; // saturation of cell production by chemical
    double K_rho_rho = global_parameters[14]; // saturation of cell by cell
    double d_rho = global_parameters[15] ;// decay of cells
    double vartheta_e = global_parameters[16]; // physiological state of area stretch
    double gamma_theta = global_parameters[17]; // sensitivity of heviside function
    double p_c_rho = global_parameters[18];// production of C by cells
    double p_c_thetaE = global_parameters[19]; // coupling of elastic and chemical
    double K_c_c = global_parameters[20];// saturation of chem by chem
    double d_c = global_parameters[21]; // decay of chemical
    //
    //---------------------------------//

    int elem_size = node_X.size();
    std::vector<Vector3d> Ebasis; Ebasis.clear();
    Matrix3d Rot90;Rot90 << 0.,-1.,0., 1.,0.,0., 0.,0.,1.;
    Ebasis.push_back(Vector3d(1.,0.,0.)); Ebasis.push_back(Vector3d(0.,1.,0.)); Ebasis.push_back(Vector3d(0.,0.,1.));
    //---------------------------------//


    //---------------------------------//
    // EVALUATE FUNCTIONS
    //
    // evaluate jacobian [actually J^(-T) ]
    Matrix3d Jac_iT = evalJacobian(node_x,xi,eta,zeta);
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
    else if(elem_size == 10){
        R = evalShapeFunctionsTetQuadraticR(xi,eta,zeta);
        Rxi = evalShapeFunctionsTetQuadraticRxi(xi,eta,zeta);
        Reta = evalShapeFunctionsTetQuadraticReta(xi,eta,zeta);
        Rzeta = evalShapeFunctionsTetQuadraticRzeta(xi,eta,zeta);
    }
    else{
        throw std::runtime_error("Wrong number of nodes in element!");
    }
    //
    // declare variables and gradients at IP
    std::vector<Vector3d> dRdXi;dRdXi.clear();
    Vector3d dxdxi,dxdeta,dxdzeta;
    dxdxi.setZero();dxdeta.setZero();dxdzeta.setZero();
    double rho=0.; Vector3d drhodXi; drhodXi.setZero();
    double c=0.; Vector3d dcdXi; dcdXi.setZero();
    //
    for(int ni=0;ni<elem_size;ni++)
    {
        dRdXi.push_back(Vector3d(Rxi[ni],Reta[ni],Rzeta[ni]));

        dxdxi += node_x[ni]*Rxi[ni];
        dxdeta += node_x[ni]*Reta[ni];
        dxdzeta += node_x[ni]*Rzeta[ni];

        rho += node_rho[ni]*R[ni];
        drhodXi(0) += node_rho[ni]*Rxi[ni];
        drhodXi(1) += node_rho[ni]*Reta[ni];
        drhodXi(2) += node_rho[ni]*Rzeta[ni];

        c += node_c[ni]*R[ni];
        dcdXi(0) += node_c[ni]*Rxi[ni];
        dcdXi(1) += node_c[ni]*Reta[ni];
        dcdXi(2) += node_c[ni]*Rzeta[ni];
    }
    //
    //---------------------------------//


    //---------------------------------//
    // EVAL GRADIENTS
    //
    // Deformation gradient and strain
    // assemble the columns
    Matrix3d dxdXi; dxdXi<<dxdxi(0),dxdeta(0),dxdzeta(0),dxdxi(1),dxdeta(1),dxdzeta(1),dxdxi(2),dxdeta(2),dxdzeta(2);
    // F = dxdX
    Matrix3d FF = dxdXi*Jac_iT.transpose();
    //
    // Gradient of concentrations in current configuration
    Matrix3d dXidx = dxdXi.inverse();
    Vector3d grad_rho  = dXidx.transpose()*drhodXi;
    Vector3d grad_c    = dXidx.transpose()*dcdXi;
    //
    // Gradient of concentrations in reference
    Vector3d Grad_rho = Jac_iT*drhodXi;
    Vector3d Grad_c = Jac_iT*dcdXi;

    return FF;
}

//-------------------------------//
// Functions for numerical tests
//-------------------------------//

/*void evalPsif(const std::vector<double> &global_parameters,double kappa, double I1e,double I4e,double &Psif,double &Psif1,double &Psif4)
{
    // unpack material constants
    //---------------------------------//
    // PARAMETERS
    //
    double k0 = global_parameters[0]; // neo hookean
    double kf = global_parameters[1]; // stiffness of collagen
    double k2 = global_parameters[2]; // nonlinear exponential
    Psif = (kf/(2.*k2))*(exp(k2*pow((kappa*I1e + (1-3*kappa)*I4e - 1),2)));
    Psif1 = 2*k2*kappa*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif;
    Psif4 = 2*k2*(1-3*kappa)*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif;
}*/

void evalSS(const std::vector<double> &global_parameters, double phif, Vector3d a0, Vector3d s0, Vector3d n0, double kappa, double lamdaP_a,double lamdaP_s,double lamdaP_N,const Matrix3d &CC,double rho, double c, Matrix3d &SS_pas,Matrix3d &SS_act, Matrix3d&SS_vol)
{
    Matrix3d Identity;Identity<<1,0,0, 0,1,0, 0,0,1;
    // fiber tensor in the reference
    Matrix3d a0a0 = a0*a0.transpose();
    Matrix3d s0s0 = s0*s0.transpose();
    Matrix3d n0n0 = n0*n0.transpose();
    // recompute split
    Matrix3d FFg = lamdaP_a*(a0a0) + lamdaP_s*(s0s0) + lamdaP_N*(n0n0);
    Matrix3d FFginv = (1./lamdaP_a)*(a0a0) + (1./lamdaP_s)*(s0s0) + (1./lamdaP_N)*n0n0;
    // Update kinematics
    Matrix3d CCinv = CC.inverse();
    // elastic strain
    Matrix3d CCe = FFginv*CC*FFginv;
    Matrix3d CCeinv = CCe.inverse();
    // invariant of the elastic strain
    double I1e = CCe(0,0) + CCe(1,1) + CCe(2,2);
    double I4e = a0.dot(CCe*a0);
    double I4tot = a0.dot(CC*a0);
    double trA = kappa*(CC(0,0)+CC(1,1)+CC(2,2)) + (1-3*kappa)*I4tot;
    // Jacobian of the deformations
    double Jp = lamdaP_a*lamdaP_s*lamdaP_N;
    double Je = sqrt(CCe.determinant());
    double J = Je*Jp;

    double k0 = global_parameters[0]; // neo hookean
    double kf = global_parameters[1]; // stiffness of collagen
    double k2 = global_parameters[2]; // nonlinear exponential
    double t_rho = global_parameters[3]; // force of fibroblasts
    double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
    double K_t = global_parameters[5]; // saturation of collagen on force
    double K_t_c = global_parameters[6]; // saturation of chemical on force

    double Psif,Psif1,Psif4;
    // passive elastic
    Psif = (kf/(2.*k2))*(exp(k2*pow((kappa*I1e + (1-3*kappa)*I4e - 1),2)));
    Psif1 = 2*k2*kappa*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif;
    Psif4 = 2*k2*(1-3*kappa)*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif;
    Matrix3d SSe_pas = phif*(k0*Identity + Psif1*Identity + Psif4*a0a0);
    // pull back to the reference,
    SS_pas = Jp*FFginv*SSe_pas*FFginv;
    // magnitude from systems bio
    double traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho;
    SS_act = (Jp*traction_act*phif/(trA*(K_t*K_t+phif*phif)))*(kappa*Identity+(1-3*kappa)*a0a0);
    // total stress, don't forget the pressure
    // Instead of (double pressure = -k0*lamda_N*lamda_N;) directly, add volumetric part of stress SSvol
    double penalty = 0.3166667;
    double Psivol = 0.5*phif*pow(penalty*(Je-1.),2) - 2*phif*k0*log(Je);
    double dPsivoldJe = phif*penalty*(Je-1.) - 2*phif*k0/Je;
    double dPsivoldJedJe = phif*penalty + 2*k0/(Je*Je);
    Matrix3d SSe_vol = dPsivoldJe*Je*CCeinv/2;
    SS_vol = Jp*FFginv*SSe_vol*FFginv;
}

void evalQ(const std::vector<double> &global_parameters, const double& phif,Vector3d a0,Vector3d s0,Vector3d n0, double kappa, const Vector3d& lamdaP,
                       const Matrix3d& CC, const double& rho, const double& c, const Vector3d& Grad_rho, const Vector3d& Grad_c, Vector3d &Q_rho, Vector3d &Q_c)
{
    double D_rhoc = global_parameters[8]; // diffusion of chemotactic gradient
    double D_cc = global_parameters[9]; // diffusion of chemical
    double lamdaP_a = lamdaP(0);
    double lamdaP_s = lamdaP(1);
    double lamdaP_n = lamdaP(2);
    Matrix3d Identity; Identity << 1., 0., 0., 0., 1., 0., 0., 0., 1.;
    // Update kinematics
    Matrix3d CCinv = CC.inverse();
    // fiber tensor in the reference
    Matrix3d a0a0 = a0*a0.transpose();
    Matrix3d s0s0 = s0*s0.transpose();
    Matrix3d n0n0 = n0*n0.transpose();
    // invariant of the elastic strain
    double I4tot = a0.dot(CC*a0);
    // Jacobian of the deformations
    double Jp = lamdaP_a*lamdaP_s*lamdaP_n;
    // calculate the structure tensor
    Matrix3d A0 = kappa*Identity + (1-3.*kappa)*a0a0;
    double trA = kappa*(CC(0,0)+CC(1,1)+CC(2,2)) + (1-3*kappa)*I4tot;
    double D_rhorho = evalDrho(phif, c);
    // Flux and Source terms for the rho and the C
    Q_rho = -D_rhorho*CCinv*Grad_rho - D_rhoc*CCinv*Grad_c;
    //Q_rho = -3*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0*Grad_rho/trA - 3*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0*Grad_c/trA;
    //Q_c = -3*(D_cc-phif*(D_cc-D_cc/10))*A0*Grad_c/trA;
    //Q_c = -1.0*(D_cc-phif*(D_cc-D_cc/10))*CCinv*Grad_c;
    Q_c = -D_cc*CCinv*Grad_c;
}

void evalS(const std::vector<double> &global_parameters, const double& phif,Vector3d a0,Vector3d s0,Vector3d n0, double kappa, const Vector3d& lamdaP,
           const Matrix3d& CC, const double& rho, const double& c, double &S_rho, double &S_c)
{
    double p_rho =global_parameters[10]; // production of fibroblasts naturally
    double p_rho_c = global_parameters[11]; // production enhanced by the chem
    double p_rho_theta = global_parameters[12]; // mechanosensing
    double K_rho_c= global_parameters[13]; // saturation of cell production by chemical
    double K_rho_rho = global_parameters[14]; // saturation of cell by cell
    double d_rho = global_parameters[15] ;// decay of cells
    double vartheta_e = global_parameters[16]; // physiological state of area stretch
    double gamma_theta = global_parameters[17]; // sensitivity of heviside function
    double p_c_rho = global_parameters[18];// production of C by cells
    double p_c_thetaE = global_parameters[19]; // coupling of elastic and chemical
    double K_c_c = global_parameters[20];// saturation of chem by chem
    double d_c = global_parameters[21]; // decay of chemical
    double lamdaP_a = lamdaP(0);
    double lamdaP_s = lamdaP(1);
    double lamdaP_n = lamdaP(2);
    Matrix3d Identity; Identity << 1., 0., 0., 0., 1., 0., 0., 0., 1.;
    // Update kinematics
    Matrix3d CCinv = CC.inverse();
    // fiber tensor in the reference
    Matrix3d a0a0 = a0*a0.transpose();
    Matrix3d s0s0 = s0*s0.transpose();
    Matrix3d n0n0 = n0*n0.transpose();
    // invariant of the elastic strain
    double I4tot = a0.dot(CC*a0);
    // Jacobian of the deformations
    double Jp = lamdaP_a*lamdaP_s*lamdaP_n;
    Matrix3d FFginv = (1./lamdaP_a)*(a0a0) + (1./lamdaP_s)*(s0s0) + (1./lamdaP_n)*n0n0;
    Matrix3d CCe = FFginv*CC*FFginv;
    double Je = sqrt(CCe.determinant());
    double J = Je*Jp;
    // Flux and Source terms for the rho and the C
    // Mechanosensing stimulus: in-plane AREAL elastic stretch of the dural
    // mid-surface, theta_e = ||cof(F^e).n0||, NOT det(F^e). The tissue is
    // treated as incompressible so det(F^e)==1 and could never respond to
    // membrane stretch. See include/mechanosensing.h.
    double theta_e = evalThetaE(J, CCinv, n0, lamdaP_a, lamdaP_s);
    double He = evalHe(theta_e, vartheta_e, gamma_theta);
    // function for elastic response of the cells
    S_rho = (p_rho + p_rho_c*c/(K_rho_c+c)+p_rho_theta*He)*(1-rho/K_rho_rho)*rho - d_rho*rho;
    // function for elastic response of the chemical
    S_c = (p_c_rho*c+ p_c_thetaE*He)*(rho/(K_c_c+c)) - d_c*c;
}

// ELEMENT RESIDUAL ONLY
/*void evalWoundRes(
        double dt,
        const std::vector<Matrix3d> &ip_Jac,
        const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
        const std::vector<double> &node_rho_0, const std::vector<double> &node_c_0,
        const std::vector<double> &ip_phif_0,const std::vector<Vector3d> &ip_a0_0,const std::vector<double> &ip_kappa_0, const std::vector<Vector3d> &ip_lamdaP_0,
        const std::vector<double> &node_rho, const std::vector<double> &node_c,
        std::vector<double> &ip_phif, std::vector<Vector3d> &ip_a0, std::vector<double> &ip_kappa, std::vector<Vector3d> &ip_lamdaP,
        const std::vector<Vector3d> &node_x,
        VectorXd &Re_x,VectorXd &Re_rho,VectorXd &Re_c) {


    //---------------------------------//
    // INPUT
    //  dt: time step
    //	elem_jac_IP: jacobians at the integration points, needed for the deformation grad
    //  matParam: material parameters
    //  Xi_t: global fields at previous time step
    //  Theta_t: structural fields at previous time steps
    //  Xi: current guess of the global fields
    //  Theta: current guess of the structural fields
    //	node_x: deformed positions
    //
    // OUTPUT
    //  Re: all residuals
    //
    // Algorithm
    //  0. Loop over integration points
    //	1. F,rho,c,nabla_rho,nabla_c: deformation at IP
    //  2. LOCAL NEWTON -> update the current guess of the structural parameters
    //  3. Fe,Fp
    //	4. Se_pas,Se_act,S
    //	5. Qrho,Srho,Qc,Sc
    //  6. Residuals
    //---------------------------------//

    //---------------------------------//
    // PARAMETERS
    //
    double k0 = global_parameters[0]; // neo hookean
    double kf = global_parameters[1]; // stiffness of collagen
    double k2 = global_parameters[2]; // nonlinear exponential
    double t_rho = global_parameters[3]; // force of fibroblasts
    double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
    double K_t = global_parameters[5]; // saturation of collagen on force
    double K_t_c = global_parameters[6]; // saturation of chemical on force
    double D_rhoc = global_parameters[8]; // diffusion of chemotactic gradient
    double D_cc = global_parameters[9]; // diffusion of chemical
    double p_rho =global_parameters[10]; // production of fibroblasts naturally
    double p_rho_c = global_parameters[11]; // production enhanced by the chem
    double p_rho_theta = global_parameters[12]; // mechanosensing
    double K_rho_c= global_parameters[13]; // saturation of cell production by chemical
    double K_rho_rho = global_parameters[14]; // saturation of cell by cell
    double d_rho = global_parameters[15] ;// decay of cells
    double vartheta_e = global_parameters[16]; // physiological state of area stretch
    double gamma_theta = global_parameters[17]; // sensitivity of heviside function
    double p_c_rho = global_parameters[18];// production of C by cells
    double p_c_thetaE = global_parameters[19]; // coupling of elastic and chemical
    double K_c_c = global_parameters[20];// saturation of chem by chem
    double d_c = global_parameters[21]; // decay of chemical
    //std::cout<<"read all global parameters\n";
    //
    //---------------------------------//



    //---------------------------------//
    // GLOBAL VARIABLES
    // Initialize the residuals to zero and declare some global stuff
    Re_x.setZero();
    Re_rho.setZero();
    Re_c.setZero();
    int elem_size = node_x.size();
    std::vector<Vector3d> Ebasis;
    Ebasis.clear();
    Matrix3d Rot90;Rot90 << 0., -1., 0., 1., 0., 0., 0., 0., 0.;
    Ebasis.push_back(Vector3d(1., 0., 0.)); Ebasis.push_back(Vector3d(0., 1., 0.)); Ebasis.push_back(Vector3d(0., 0., 1.));
    //---------------------------------//



    //---------------------------------//
    // LOOP OVER INTEGRATION POINTS
    //---------------------------------//

    // array with integration points
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
    //std::cout<<"loop over integration points\n";
    for (int ip = 0; ip < IP.size(); ip++) {

        //---------------------------------//
        // EVALUATE FUNCTIONS
        //
        // coordinates of the integration point in parent domain
        double xi = IP[ip](0);
        double eta = IP[ip](1);
        double zeta = IP[ip](2);
        // weight of the integration point
        double wip = IP[ip](3);
        double Jac = 1. / ip_Jac[ip].determinant();
        //std::cout<<"integration point: "<<xi<<", "<<eta<<"; "<<zeta<<"; "<<wip<<"; "<<Jac<<"\n";
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
        //
        // declare variables and gradients at IP
        std::vector<Vector3d> dRdXi;
        dRdXi.clear();
        Vector3d dxdxi, dxdeta, dxdzeta;
        dxdxi.setZero();
        dxdeta.setZero();
        dxdzeta.setZero();
        double rho_0 = 0.;
        Vector3d drho0dXi;
        drho0dXi.setZero();
        double rho = 0.;
        Vector3d drhodXi;
        drhodXi.setZero();
        double c_0 = 0.;
        Vector3d dc0dXi;
        dc0dXi.setZero();
        double c = 0.;
        Vector3d dcdXi;
        dcdXi.setZero();
        //
        for (int ni = 0; ni < elem_size; ni++) {
            dRdXi.push_back(Vector3d(Rxi[ni], Reta[ni], Rzeta[ni]));

            dxdxi += node_x[ni] * Rxi[ni];
            dxdeta += node_x[ni] * Reta[ni];
            dxdzeta += node_x[ni] * Rzeta[ni];

            rho_0 += node_rho_0[ni] * R[ni];
            drho0dXi(0) += node_rho_0[ni] * Rxi[ni];
            drho0dXi(1) += node_rho_0[ni] * Reta[ni];
            drho0dXi(2) += node_rho_0[ni] * Rzeta[ni];

            rho += node_rho[ni] * R[ni];
            drhodXi(0) += node_rho[ni] * Rxi[ni];
            drhodXi(1) += node_rho[ni] * Reta[ni];
            drhodXi(2) += node_rho[ni] * Rzeta[ni];

            c_0 += node_c_0[ni] * R[ni];
            dc0dXi(0) += node_c_0[ni] * Rxi[ni];
            dc0dXi(1) += node_c_0[ni] * Reta[ni];
            dc0dXi(2) += node_c_0[ni] * Rzeta[ni];

            c += node_c[ni] * R[ni];
            dcdXi(0) += node_c[ni] * Rxi[ni];
            dcdXi(1) += node_c[ni] * Reta[ni];
            dcdXi(2) += node_c[ni] * Rzeta[ni];
        }
        //
        //---------------------------------//



        //---------------------------------//
        // EVAL GRADIENTS
        //
        // Deformation gradient and strain
        // assemble the columns
        Matrix3d dxdXi;
        dxdXi << dxdxi(0), dxdeta(0), dxdzeta(0), dxdxi(1), dxdeta(1), dxdzeta(1), dxdxi(2), dxdeta(2), dxdzeta(2);
        // F = dxdX
        Matrix3d FF = ip_Jac[ip] * dxdXi;
        // the strain
        Matrix3d Identity;
        Identity << 1, 0, 0, 0, 1, 0, 0, 0, 1;
        Matrix3d EE = 0.5 * (FF.transpose() * FF - Identity);
        Matrix3d CC = FF.transpose() * FF;
        Matrix3d CCinv = CC.inverse();
        //
        // Gradient of concentrations in current configuration
        Matrix3d dXidx = dxdXi.inverse();
        Vector3d grad_rho0 = dXidx.transpose() * drho0dXi;
        Vector3d grad_rho = dXidx.transpose() * drhodXi;
        Vector3d grad_c0 = dXidx.transpose() * dc0dXi;
        Vector3d grad_c = dXidx.transpose() * dcdXi;
        //
        // Gradient of concentrations in reference
        Vector3d Grad_rho0 = ip_Jac[ip] * drho0dXi;
        Vector3d Grad_rho = ip_Jac[ip] * drhodXi;
        Vector3d Grad_c0 = ip_Jac[ip] * dc0dXi;
        Vector3d Grad_c = ip_Jac[ip] * dcdXi;
        //
        // Gradient of basis functions for the nodes in reference
        std::vector<Vector3d> Grad_R;Grad_R.clear();
        // Gradient of basis functions in deformed configuration
        std::vector<Vector3d> grad_R;grad_R.clear();
        for(int ni=0;ni<elem_size;ni++)
        {
            Grad_R.push_back(ip_Jac[ip]*dRdXi[ni]);
            grad_R.push_back(dXidx.transpose()*dRdXi[ni]);
        }
        //
        //---------------------------------//
        //std::cout<<"deformation gradient\n"<<FF<<"\n";
        //std::cout<<"rho0: "<<rho_0<<", rho: "<<rho<<"\n";
        //std::cout<<"c0: "<<c_0<<", c: "<<c<<"\n";
        //std::cout<<"gradient of rho: "<<Grad_rho<<"\n";
        //std::cout<<"gradient of c: "<<Grad_c<<"\n";

        //---------------------------------//
        // LOCAL NEWTON: structural problem
        //
        VectorXd dThetadCC(54);
        dThetadCC.setZero();
        VectorXd dThetadrho(6);
        dThetadrho.setZero();
        VectorXd dThetadc(6);
        dThetadc.setZero();
        //std::cout<<"Local variables before update:\nphif0 = "<<ip_phif_0[ip]<<"\nkappa_0 = "<<ip_kappa_0[ip]<<"\na0_0 = ["<<ip_a0_0[ip](0)<<","<<ip_a0_0[ip](1)<<"]\nlamdaP_0 = ["<<ip_lamdaP_0[ip](0)<<","<<ip_lamdaP_0[ip](1)<<"]\n";
        localWoundProblem(dt, local_parameters, c, rho, FF, ip_phif_0[ip], ip_a0_0[ip], ip_kappa_0[ip], ip_lamdaP_0[ip],
                          ip_phif[ip], ip_a0[ip], ip_kappa[ip], ip_lamdaP[ip], dThetadCC, dThetadrho, dThetadc);
        //
        // rename variables to make it easier
        double phif_0 = ip_phif_0[ip];
        Vector3d a0_0 = ip_a0_0[ip]; /// Copy from above again
        double kappa_0 = ip_kappa_0[ip];
        Vector3d lamdaP_0 = ip_lamdaP_0[ip];
        double phif = ip_phif[ip];
        Vector3d a0 = ip_a0[ip];
        double kappa = ip_kappa[ip];
        Vector3d lamdaP = ip_lamdaP[ip];
        double lamdaP_a_0 = lamdaP_0(0);
        double lamdaP_s_0 = lamdaP_0(1);
        double lamdaP_a = lamdaP(0);
        double lamdaP_s = lamdaP(1);
        //std::cout<<"Local variables after update:\nphif0 = "<<phif_0<<",	phif = "<<phif<<"\nkappa_0 = "<<kappa_0<<",	kappa = "<<kappa<<"\na0_0 = ["<<a0_0(0)<<","<<a0_0(1)<<"],	a0 = ["<<a0(0)<<","<<a0(1)<<"]\nlamdaP_0 = ["<<lamdaP_0(0)<<","<<lamdaP_0(1)<<"],	lamdaP = ["<<lamdaP(0)<<","<<lamdaP(1)<<"]\n";
        // make sure the update preserved length
        double norma0 = sqrt(a0.dot(a0));
        if (fabs(norma0 - 1.) > 0.001) { std::cout << "update did not preserve unit length of a0\n"; }
        ip_a0[ip] = a0 / (sqrt(a0.dot(a0)));
        a0 = a0 / (sqrt(a0.dot(a0)));
        //
        // unpack the derivatives wrt CC
        // remember dThetatCC: 9 phi, 9 a0x, 9 a0y, 9 a0z, 9 kappa, 9 lamdaPa, 9 lamdaPs, 9 lamdaPN
        Matrix3d dphifdCC; dphifdCC.setZero();
        dphifdCC(0,0) = dThetadCC(0); dphifdCC(0,1) = dThetadCC(1); dphifdCC(0,2) = dThetadCC(2);
        dphifdCC(1,0) = dThetadCC(3); dphifdCC(1,1) = dThetadCC(4); dphifdCC(1,2) = dThetadCC(5);
        dphifdCC(2,0) = dThetadCC(6); dphifdCC(2,1) = dThetadCC(7); dphifdCC(2,2) = dThetadCC(8);
        Matrix3d da0xdCC;da0xdCC.setZero();
        da0xdCC(0,0) = dThetadCC(9); da0xdCC(0,1) = dThetadCC(10); da0xdCC(0,2) = dThetadCC(11);
        da0xdCC(1,0) = dThetadCC(12); da0xdCC(1,1) = dThetadCC(13); da0xdCC(1,2) = dThetadCC(14);
        da0xdCC(2,0) = dThetadCC(15); da0xdCC(2,1) = dThetadCC(16); da0xdCC(2,2) = dThetadCC(17);
        Matrix3d da0ydCC;da0ydCC.setZero();
        da0ydCC(0,0) = dThetadCC(18); da0ydCC(0,1) = dThetadCC(19); da0ydCC(0,2) = dThetadCC(20);
        da0ydCC(1,0) = dThetadCC(21); da0ydCC(1,1) = dThetadCC(22); da0ydCC(1,2) = dThetadCC(23);
        da0ydCC(2,0) = dThetadCC(24); da0ydCC(2,1) = dThetadCC(25); da0ydCC(2,2) = dThetadCC(26);
        Matrix3d dkappadCC; dkappadCC.setZero();
        dkappadCC(0,0) = dThetadCC(27); dkappadCC(0,1) = dThetadCC(28); dkappadCC(0,2) = dThetadCC(29);
        dkappadCC(1,0) = dThetadCC(30); dkappadCC(1,1) = dThetadCC(31); dkappadCC(1,2) = dThetadCC(32);
        dkappadCC(2,0) = dThetadCC(33); dkappadCC(2,1) = dThetadCC(34); dkappadCC(2,2) = dThetadCC(35);
        Matrix3d dlamdaP_adCC; dlamdaP_adCC.setZero();
        dlamdaP_adCC(0,0) = dThetadCC(36); dlamdaP_adCC(0,1) = dThetadCC(37); dlamdaP_adCC(0,2) = dThetadCC(38);
        dlamdaP_adCC(1,0) = dThetadCC(39); dlamdaP_adCC(1,1) = dThetadCC(40); dlamdaP_adCC(1,2) = dThetadCC(41);
        dlamdaP_adCC(2,0) = dThetadCC(42); dlamdaP_adCC(2,1) = dThetadCC(43); dlamdaP_adCC(2,2) = dThetadCC(44);
        Matrix3d dlamdaP_sdCC; dlamdaP_sdCC.setZero();
        dlamdaP_sdCC(0,0) = dThetadCC(45); dlamdaP_sdCC(0,1) = dThetadCC(46); dlamdaP_sdCC(0,2) = dThetadCC(47);
        dlamdaP_sdCC(1,0) = dThetadCC(48); dlamdaP_sdCC(1,1) = dThetadCC(49); dlamdaP_sdCC(1,2) = dThetadCC(50);
        dlamdaP_sdCC(2,0) = dThetadCC(51); dlamdaP_sdCC(2,1) = dThetadCC(52); dlamdaP_sdCC(2,2) = dThetadCC(53);
        // unpack the derivatives wrt rho
        double dphifdrho = dThetadrho(0);
        double da0xdrho = dThetadrho(1);
        double da0ydrho = dThetadrho(2);
        double dkappadrho = dThetadrho(3);
        double dlamdaP_adrho = dThetadrho(4);
        double dlamdaP_sdrho = dThetadrho(5);
        // unpack the derivatives wrt c
        double dphifdc = dThetadc(0);
        double da0xdc = dThetadc(1);
        double da0ydc = dThetadc(2);
        double dkappadc = dThetadc(3);
        double dlamdaP_adc = dThetadc(4);
        double dlamdaP_sdc = dThetadc(5);
        //
        //---------------------------------//



        //---------------------------------//
        // CALCULATE SOURCE AND FLUX
        //
        // Update kinematics
        CCinv = CC.inverse(); /// Copy this from above
        // re-compute basis a0, s0
        Matrix3d Rot90;
        Rot90 << 0., -1., 0., 1., 0., 0., 0., 0., 1.;
        Vector3d s0 = Rot90 * a0;
        // fiber tensor in the reference
        Matrix3d a0a0 = a0 * a0.transpose();
        Matrix3d s0s0 = s0 * s0.transpose();
        Matrix3d A0 = kappa * Identity + (1 - 2. * kappa) * a0a0;
        Vector3d a = FF * a0;
        Matrix3d A = kappa * FF * FF.transpose() + (1. - 2.0 * kappa) * a * a.transpose();
        double trA = A(0, 0) + A(1, 1) + A(2, 2);
        Matrix3d hat_A = A / trA;
        // recompute split
        Matrix3d FFg = lamdaP_a * (a0a0) + lamdaP_s * (s0s0); /// This will need to be updated with a new rule
        double thetaP = lamdaP_a * lamdaP_s;
        Matrix3d FFginv = (1. / lamdaP_a) * (a0a0) + (1. / lamdaP_s) * (s0s0); /// Same here
        Matrix3d FFe = FF * FFginv;
        //std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
        // elastic strain
        Matrix3d CCe = FFe.transpose() * FFe;
        // invariant of the elastic strain
        double I1e = CCe(0, 0) + CCe(1, 1) + CCe(2, 2);
        double I4e = a0.dot(CCe * a0);
        // calculate the normal stretch
        double thetaE = sqrt(CCe.determinant());
        double theta = thetaE * thetaP;
        //std::cout<<"split of the determinants. theta = thetaE*thetaB = "<<theta<<" = "<<thetaE<<"*"<<thetaP<<"\n";
        double lamda_N = 1. / thetaE;
        double I4tot = a0.dot(CC * a0);
        // Second Piola Kirchhoff stress tensor
        // passive elastic
        double Psif = (kf / (2. * k2)) * exp(k2 * pow((kappa * I1e + (1 - 3 * kappa) * I4e - 1), 2));
        double Psif1 = 2 * k2 * kappa * (kappa * I1e + (1 - 2 * kappa) * I4e - 1) * Psif;
        double Psif4 = 2 * k2 * (1 - 3 * kappa) * (kappa * I1e + (1 - 3 * kappa) * I4e - 1) * Psif;
        Matrix3d SSe_pas = k0 * Identity + phif * (Psif1 * Identity + Psif4 * a0a0);
        // pull back to the reference
        Matrix3d SS_pas = thetaP * FFginv * SSe_pas * FFginv;
        // magnitude from systems bio
        double traction_act = (t_rho + t_rho_c * c / (K_t_c + c)) * rho;
        Matrix3d SS_act = (thetaP * traction_act * phif / trA) * A0;
        // total stress, don't forget the pressure
        double pressure = -k0 * lamda_N * lamda_N;
        Matrix3d SS_pres = pressure * thetaP * CCinv;
        //std::cout<<"stresses.\nSSpas\n"<<SS_pas<<"\nSS_act\n"<<SS_act<<"\nSS_pres"<<SS_pres<<"\n";
        Matrix3d SS = SS_pas + SS_act + SS_pres;
        Vector3d SS_voigt;
        SS_voigt(0) = SS(0, 0); SS_voigt(1) = SS(1, 1); SS_voigt(2) = SS(2, 2);
        SS_voigt(3) = SS(1, 2); SS_voigt(4) = SS(0, 2); SS_voigt(5) = SS(0, 1);
        // Flux and Source terms for the rho and the C
        Vector3d Q_rho = -D_rhorho*CCinv*Grad_rho/(1+phif) - D_rhoc*rho*CCinv*Grad_c/(1+phif);
        Vector3d Q_c = -D_cc*CCinv*Grad_c;
        // mechanosensing
        double He = 1. / (1. + exp(-gamma_c_thetaE * (thetaE + theta_phy)));
        double S_rho =
                (p_rho + p_rho_c * c / (K_rho_c + c) + p_rho_theta * He) * (1 - rho / K_rho_rho) * rho - d_rho * rho;
        // heviside function for elastic response of the chemical
        double S_c = (p_c_rho * c + p_c_thetaE * He) * (rho / (K_c_c + c)) - d_c * c;
        //std::cout<<"flux of celss, Q _rho\n"<<Q_rho<<"\n";
        //std::cout<<"source of cells, S_rho: "<<S_rho<<"\n";
        //std::cout<<"flux of chemical, Q _c\n"<<Q_c<<"\n";
        //std::cout<<"source of chemical, S_c: "<<S_c<<"\n";
        //---------------------------------//



        //---------------------------------//
        // ADD TO THE RESIDUAL
        //
        Matrix3d deltaFF, deltaCC;
        VectorXd deltaCC_voigt;
        for (int nodei = 0; nodei < elem_size; nodei++) {
            for (int coordi = 0; coordi < 3; coordi++) {
                // alternatively, define the deltaCC
                deltaFF = Ebasis[coordi] * Grad_R[nodei].transpose();
                deltaCC = deltaFF.transpose() * FF + FF.transpose() * deltaFF;
                // deltaCC_voigt = Vector3d(deltaCC(0,0),deltaCC(1,1),2.*deltaCC(1,0));
                deltaCC_voigt(0) = deltaCC(0, 0); deltaCC_voigt(1) = deltaCC(1, 1); deltaCC_voigt(2) = deltaCC(2, 2);
                deltaCC_voigt(3) = 2 * deltaCC(1, 2); deltaCC_voigt(4) = 2 * deltaCC(0, 2); deltaCC_voigt(5) = deltaCC(0, 1);
                Re_x(nodei * 3 + coordi) += Jac * SS_voigt.dot(deltaCC_voigt);
            }
            Re_rho(nodei) += Jac * (((rho - rho_0) / dt - S_rho) * R[nodei] - Grad_R[nodei].dot(Q_rho));
            Re_c(nodei) += Jac * (((c - c_0) / dt - S_c) * R[nodei] - Grad_R[nodei].dot(Q_c));
        }
        //
        //---------------------------------//

    }
}*/

//--------------------------------------------------------//
// RESIDUAL AND TANGENT
//--------------------------------------------------------//

// ELEMENT RESIDUAL AND TANGENT
void evalBC(int surface_boundary_flag, const std::vector<double> &ip_Jac, const std::vector<double> &global_parameters,
            const std::vector<double> &node_rho, const std::vector<double> &node_c, const std::vector<double> &node_phif,
            const std::vector<Vector3d> &node_x, const std::vector<Vector3d> &node_X,
            const std::vector<Vector3d> &node_dphifdu, const std::vector<double> &node_dphifdrho, const std::vector<double> &node_dphifdc,
            VectorXd &Re_x_surf,MatrixXd &Ke_x_x_surf,MatrixXd &Ke_x_rho_surf,MatrixXd &Ke_x_c_surf,
            VectorXd &Re_rho_surf,MatrixXd &Ke_rho_x_surf, MatrixXd &Ke_rho_rho_surf,MatrixXd &Ke_rho_c_surf,
            VectorXd &Re_c_surf,MatrixXd &Ke_c_x_surf,MatrixXd &Ke_c_rho_surf,MatrixXd &Ke_c_c_surf){

    //std::cout<<"element routine\n";
    //---------------------------------//
    // INPUT
    //	elem_jac_IP: jacobians at the integration points, needed for the deformation grad
    //	node_x: deformed positions
    //
    // OUTPUT
    //  Re: all residuals
    //  Ke: all tangents
    //
    // Algorithm
    //  0. Loop over integration points
    //  6. Residuals
    //  7. Tangents
    //---------------------------------//

    //---------------------------------//
    // PARAMETERS
    //
    double D_rhoc = global_parameters[8]; // diffusion of chemotactic gradient
    double D_cc = global_parameters[9]; // diffusion of chemical
    //std::cout<<"read only diffusion coefficients if needed\n";
    //
    //---------------------------------//

    //---------------------------------//
    // GLOBAL VARIABLES
    // Declare some global stuff
    int surf_elem_size = node_x.size();
    //---------------------------------//

    //---------------------------------//
    // LOOP OVER INTEGRATION POINTS
    //---------------------------------//
    //---------------------------------//

    // array with integration points
    std::vector<Vector3d> IP;
    if (surf_elem_size == 4) {
        // linear hexahedron
        IP = LineQuadriIPSurfaceQuad();
    } else{
        throw std::runtime_error("Wrong number of nodes in element!");
    }
    int IP_size = IP.size();
    //std::cout<<"loop over integration points\n";
    for (int ip = 0; ip < IP_size; ip++) {
        //---------------------------------//
        // EVALUATE FUNCTIONS
        //
        // coordinates of the integration point in parent domain
        double xi = IP[ip](0);
        double eta = IP[ip](1);
        // weight of the integration point
        double wip = IP[ip](2);
        // we also need the scaling factor
        // not a determinant because this is a rectangular matrix
        double Jac = ip_Jac[ip];
        //std::cout<<"integration point: "<<xi<<", "<<eta<<", "<<zeta<<"; "<<wip<<"; "<<Jac<<"\n";
        //
        // eval shape functions
        std::vector<double> R;
        // eval derivatives
        std::vector<double> Rxi;
        std::vector<double> Reta;
        if(surf_elem_size == 4){
            R = evalShapeFunctionsSurfaceQuadR(xi,eta);
            Rxi = evalShapeFunctionsSurfaceQuadRxi(xi,eta);
            Reta = evalShapeFunctionsSurfaceQuadReta(xi,eta);
        }
        else{
            throw std::runtime_error("Wrong number of nodes in element!");
        }

        //
        // declare variables and gradients at IP
        std::vector<Vector2d> dRdXi;dRdXi.clear();
        Vector3d dxdxi,dxdeta;
        dxdxi.setZero();dxdeta.setZero();
        double rho=0.; Vector2d drhodXi; drhodXi.setZero();
        double c=0.; Vector2d dcdXi; dcdXi.setZero();
        double phif=0; Vector3d dphifdu; dphifdu.setZero();
        double dphifdrho=0;
        double dphifdc=0;
        //
        for(int ni=0;ni<surf_elem_size;ni++)
        {
            dRdXi.push_back(Vector2d(Rxi[ni],Reta[ni]));

            dxdxi += node_x[ni]*Rxi[ni];
            dxdeta += node_x[ni]*Reta[ni];

            rho += node_rho[ni]*R[ni];
            drhodXi(0) += node_rho[ni]*Rxi[ni];
            drhodXi(1) += node_rho[ni]*Reta[ni];

            c += node_c[ni]*R[ni];
            dcdXi(0) += node_c[ni]*Rxi[ni];
            dcdXi(1) += node_c[ni]*Reta[ni];

            phif += node_phif[ni]*R[ni];

            dphifdu += node_dphifdu[ni]*R[ni];
            dphifdrho += node_dphifdrho[ni]*R[ni];
            dphifdc += node_dphifdc[ni]*R[ni];
        }
        //
        //---------------------------------//



        //---------------------------------//
        // EVAL GRADIENTS
        //
        // Deformation gradient and strain
        // assemble the columns
        //Matrix2d dxdXi; dxdXi<<dxdxi(0),dxdeta(0),dxdxi(1),dxdeta(1);
        //
        // Gradient of concentrations in current configuration
        //Matrix2d dXidx = dxdXi.inverse();
        //
        // Gradient of basis functions for the nodes in reference
        //std::vector<Vector2d> Grad_R;Grad_R.clear();
        // Gradient of basis functions in deformed configuration
        //std::vector<Vector2d> grad_R;grad_R.clear();
        //for(int ni=0;ni<surf_elem_size;ni++)
        //{
         //   Grad_R.push_back(ip_Jac[ip]*dRdXi[ni]);
         //   grad_R.push_back(dXidx.transpose()*dRdXi[ni]);
        //}
        //
        //---------------------------------//

        Vector3d traction = Vector3d(0,0,0);
        double spring = -1e-4;
    double D_rhorho = evalDrho(phif, c);
        double k_rho_0 = -D_rhorho/10;
        double k_rho = (k_rho_0-phif*(k_rho_0-k_rho_0/10)); // -3.0*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0*Grad_rho/trA - 3*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0*Grad_c/trA;
        double d_k_rho_dphif = (k_rho_0-k_rho_0/10);
        double k_c_0 = -D_cc/10;
        double k_c = (k_c_0-phif*(k_c_0-k_c_0/10)); // -3.0*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0*Grad_rho/trA - 3*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0*Grad_c/trA;
        double d_k_c_dphif = (k_rho_0-k_rho_0/10);
        double rho_inf = 1000;
        double rho_flux = 0;
        double c_inf = 0;
        double c_flux = 0;

        //---------------------------------//
        // ADD TO THE RESIDUAL
        //---------------------------------//
        // Residual = P(u) - F
        for (int nodei = 0; nodei < surf_elem_size; nodei++) {
            Vector3d displacement = node_x[nodei] - node_X[nodei];
            // Element residual for x
            for (int coordi = 0; coordi < 3; coordi++) {
                // This represents the traction force for a Neumann condition
                //Re_x_surf(nodei*3+coordi) -= Jac*traction(coordi)*R[nodei]*wip;
                // This represents the spring for a Robin condition
                //Re_x_surf(nodei*3+coordi) += Jac*-1.0*spring*phif*displacement(coordi)*R[nodei]*wip; //
            }
            // This represents the flux for a Neumann condition
            Re_rho_surf(nodei) -= Jac*(rho_flux)*R[nodei]*wip;
            Re_c_surf(nodei) -= Jac*(c_flux)*R[nodei]*wip;
            // This represents the flux for a Robin condition
            Re_rho_surf(nodei) += Jac*k_rho*(rho_inf - rho)*R[nodei]*wip;
            Re_c_surf(nodei) += Jac*k_c*(c_inf - c)*R[nodei]*wip;
        }

        //
        //---------------------------------//

        //---------------------------------//
        // ADD TO THE TANGENT
        //---------------------------------//

        // Loop over nodes and coordinates twice and assemble the corresponding entry
        for(int nodei=0;nodei<surf_elem_size;nodei++){
            Vector3d displacement = node_x[nodei] - node_X[nodei];
            for(int coordi=0;coordi<3;coordi++){
                for(int nodej=0;nodej<surf_elem_size;nodej++){
                    for(int coordj=0;coordj<3;coordj++){
                        // Robin traction boundary condition only includes displacement
                        //Ke_x_x_surf(nodei*3+coordi,nodej*3+coordj) += Jac*(-1.0)*(phif*Matrix3d::Identity()(coordi,coordj)+ dphifdu(coordj)*displacement(coordi))*spring*R[nodei]*R[nodej]*wip; //
                        // Robin flux boundary has no dependence on x (besides phi, which we will skip for now)
                        Ke_rho_x_surf(nodei,nodej*3+coordj) += Jac*(d_k_rho_dphif*dphifdu(coordj))*(rho_inf - rho)*R[nodei]*wip; //
                        Ke_c_x_surf(nodei,nodej*3+coordj) += Jac*(d_k_c_dphif*dphifdu(coordj))*(c_inf - c)*R[nodei]*wip; //
                    }
                    // But also has phif which has dependence on rho and c
                    //Ke_x_rho_surf(nodei*3+coordi,nodej) += Jac*(-1.0)*spring*dphifdrho*R[nodei]*displacement(coordi)*wip; //
                    //Ke_x_c_surf(nodei*3+coordi,nodej) += Jac*(-1.0)*spring*dphifdc*R[nodei]*displacement(coordi)*wip; //
                    // Flux tangents
                    Ke_rho_rho_surf(nodei,nodej) += Jac*(-1.0*k_rho*R[nodej] + d_k_rho_dphif*dphifdrho*(rho_inf - rho))*R[nodei]*wip;
                    Ke_rho_c_surf(nodei,nodej) += Jac*(d_k_rho_dphif*dphifdc*(rho_inf - rho))*R[nodei]*wip;
                    Ke_c_c_surf(nodei,nodej) += Jac*(-1.0*k_c*R[nodej] + d_k_c_dphif*dphifdc*(c_inf - c))*R[nodei]*wip;
                    Ke_c_rho_surf(nodei,nodej) += Jac*(d_k_c_dphif*dphifdrho*(c_inf - c))*R[nodei]*wip;
                }
            }
        }
        //---------------------------------//
    }
}


// ELEMENT RESIDUAL AND TANGENT
void evalWoundMechanics(double dt, double time, double time_final,
        const std::vector<Matrix3d> &ip_Jac, const std::vector<double> &global_parameters,
        std::vector<Matrix3d> &ip_strain,std::vector<Matrix3d> &ip_stress, const std::vector<double> &node_rho_0, const std::vector<double> &node_c_0,
        const std::vector<double> &ip_phif_0,const std::vector<Vector3d> &ip_a0_0,const std::vector<Vector3d> &ip_s0_0,const std::vector<Vector3d> &ip_n0_0,const std::vector<double> &ip_kappa_0, const std::vector<Vector3d> &ip_lamdaP_0, //
        const std::vector<double> &node_rho, const std::vector<double> &node_c,
        std::vector<double> &ip_phif, std::vector<Vector3d> &ip_a0, std::vector<Vector3d> &ip_s0, std::vector<Vector3d> &ip_n0, std::vector<double> &ip_kappa, std::vector<Vector3d> &ip_lamdaP,
        const std::vector<Vector3d> &node_x, VectorXd &Re_x,MatrixXd &Ke_x_x)
{

    //std::cout<<"element routine\n";
    //---------------------------------//
    // INPUT
    //  dt: time step
    //	elem_jac_IP: jacobians at the integration points, needed for the deformation grad
    //  matParam: material parameters
    //  Xi_t: global fields at previous time step
    //  Theta_t: structural fields at previous time steps
    //  Xi: current guess of the global fields
    //  Theta: current guess of the structural fields
    //	node_x: deformed positions
    //
    // OUTPUT
    //  Re: all residuals
    //  Ke: all tangents
    //
    // Algorithm
    //  0. Loop over integration points
    //	1. F,rho,c,nabla_rho,nabla_c: deformation at IP
    //  2. LOCAL NEWTON -> update the current guess of the structural parameters
    //  3. Fe,Fp
    //	4. Se_pas,Se_act,S
    //	5. Qrho,Srho,Qc,Sc
    //  6. Residuals
    //  7. Tangents
    //---------------------------------//

    //---------------------------------//
    // PARAMETERS
    //
    double k0 = global_parameters[0]; // neo hookean
    double kf = global_parameters[1]; // stiffness of collagen
    double k2 = global_parameters[2]; // nonlinear exponential
    double t_rho = global_parameters[3]; // force of fibroblasts
    double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
    double K_t = global_parameters[5]; // saturation of collagen on force
    double K_t_c = global_parameters[6]; // saturation of chemical on force
    double bx = global_parameters[22]; // body force
    double by = global_parameters[23]; // body force
    double bz = global_parameters[24]; // body force
    //std::cout<<"read all global parameters\n";
    //
    //---------------------------------//



    //---------------------------------//
    // GLOBAL VARIABLES
    // Initialize the residuals to zero and declare some global stuff
    Re_x.setZero(); Ke_x_x.setZero();
    int elem_size = node_x.size();
    std::vector<Vector3d> Ebasis; Ebasis.clear();
    Ebasis.push_back(Vector3d(1.,0.,0.)); Ebasis.push_back(Vector3d(0.,1.,0.)); Ebasis.push_back(Vector3d(0.,0.,1.));
    //---------------------------------//



    //---------------------------------//
    // LOOP OVER INTEGRATION POINTS
    //---------------------------------//
    //---------------------------------//

    // array with integration points
    std::vector<Vector4d> IP;
    if(elem_size == 8 || elem_size == 20){
        // linear hexahedron
        IP = LineQuadriIP();
    } else if(elem_size == 27){
        // quadratic hexahedron
        IP = LineQuadriIPQuadratic();
    } else if(elem_size==4){
        // linear tetrahedron
        IP = LineQuadriIPTet();
    } else if(elem_size==10){
        // quadratic tetrahedron
        IP = LineQuadriIPTetQuadratic();
    }
    int IP_size = IP.size();
    //std::cout<<"loop over integration points\n";
    for(int ip=0;ip<IP_size;ip++)
    {
        //---------------------------------//
        // EVALUATE FUNCTIONS
        //
        // coordinates of the integration point in parent domain
        double xi = IP[ip](0);
        double eta = IP[ip](1);
        double zeta = IP[ip](2);
        // weight of the integration point
        double wip = IP[ip](3);
        double Jac = 1./ip_Jac[ip].determinant(); // instead of Jacobian, J^(-T) is stored
        //std::cout<<"integration point: "<<xi<<", "<<eta<<", "<<zeta<<"; "<<wip<<"; "<<Jac<<"\n";
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
        } else if(elem_size == 20){
            R = evalShapeFunctionsQuadraticR(xi,eta,zeta);
            Rxi = evalShapeFunctionsQuadraticRxi(xi,eta,zeta);
            Reta = evalShapeFunctionsQuadraticReta(xi,eta,zeta);
            Rzeta = evalShapeFunctionsQuadraticRzeta(xi,eta,zeta);
        } else if(elem_size == 27){
            R = evalShapeFunctionsQuadraticLagrangeR(xi,eta,zeta);
            Rxi = evalShapeFunctionsQuadraticLagrangeRxi(xi,eta,zeta);
            Reta = evalShapeFunctionsQuadraticLagrangeReta(xi,eta,zeta);
            Rzeta = evalShapeFunctionsQuadraticLagrangeRzeta(xi,eta,zeta);
        } else if(elem_size == 4){
            R = evalShapeFunctionsTetR(xi,eta,zeta);
            Rxi = evalShapeFunctionsTetRxi(xi,eta,zeta);
            Reta = evalShapeFunctionsTetReta(xi,eta,zeta);
            Rzeta = evalShapeFunctionsTetRzeta(xi,eta,zeta);
        } else if(elem_size == 10){
            R = evalShapeFunctionsTetQuadraticR(xi,eta,zeta);
            Rxi = evalShapeFunctionsTetQuadraticRxi(xi,eta,zeta);
            Reta = evalShapeFunctionsTetQuadraticReta(xi,eta,zeta);
            Rzeta = evalShapeFunctionsTetQuadraticRzeta(xi,eta,zeta);
        } else{
            throw std::runtime_error("Wrong number of nodes in element!");
        }
        //
        // declare variables and gradients at IP
        std::vector<Vector3d> dRdXi;dRdXi.clear();
        Vector3d dxdxi,dxdeta,dxdzeta; dxdxi.setZero();dxdeta.setZero();dxdzeta.setZero();
        double rho_0=0.; Vector3d drho0dXi; drho0dXi.setZero();
        double rho=0.; Vector3d drhodXi; drhodXi.setZero();
        double c_0=0.; Vector3d dc0dXi; dc0dXi.setZero();
        double c=0.; Vector3d dcdXi; dcdXi.setZero();
        //
        for(int ni=0;ni<elem_size;ni++)
        {
            dRdXi.push_back(Vector3d(Rxi[ni],Reta[ni],Rzeta[ni]));

            dxdxi += node_x[ni]*Rxi[ni];
            dxdeta += node_x[ni]*Reta[ni];
            dxdzeta += node_x[ni]*Rzeta[ni];

            rho_0 += node_rho_0[ni]*R[ni];
            drho0dXi(0) += node_rho_0[ni]*Rxi[ni];
            drho0dXi(1) += node_rho_0[ni]*Reta[ni];
            drho0dXi(2) += node_rho_0[ni]*Rzeta[ni];

            rho += node_rho[ni]*R[ni];
            drhodXi(0) += node_rho[ni]*Rxi[ni];
            drhodXi(1) += node_rho[ni]*Reta[ni];
            drhodXi(2) += node_rho[ni]*Rzeta[ni];

            c_0 += node_c_0[ni]*R[ni];
            dc0dXi(0) += node_c_0[ni]*Rxi[ni];
            dc0dXi(1) += node_c_0[ni]*Reta[ni];
            dc0dXi(2) += node_c_0[ni]*Rzeta[ni];

            c += node_c[ni]*R[ni];
            dcdXi(0) += node_c[ni]*Rxi[ni];
            dcdXi(1) += node_c[ni]*Reta[ni];
            dcdXi(2) += node_c[ni]*Rzeta[ni];
        }
        //
        //---------------------------------//



        //---------------------------------//
        // EVAL GRADIENTS
        //
        // Deformation gradient and strain
        // assemble the columns
        Matrix3d dxdXi;dxdXi<< dxdxi(0), dxdeta(0), dxdzeta(0), dxdxi(1), dxdeta(1), dxdzeta(1), dxdxi(2), dxdeta(2), dxdzeta(2);
        // F = dxdX
        Matrix3d FF = dxdXi*ip_Jac[ip].transpose();

        // the strain
        Matrix3d Identity;Identity << 1.,0.,0., 0.,1.,0., 0.,0.,1.;
        //Matrix3d EE = 0.5*(FF.transpose()*FF - Identity);
        Matrix3d CC = FF.transpose()*FF;
        Matrix3d CCinv = CC.inverse();
        //
        // Gradient of concentrations in current configuration
        Matrix3d dXidx = dxdXi.inverse();
        //
        // Gradient of basis functions for the nodes in reference
        std::vector<Vector3d> Grad_R;Grad_R.clear();
        // Gradient of basis functions in deformed configuration
        std::vector<Vector3d> grad_R;grad_R.clear();
        for(int ni=0;ni<elem_size;ni++)
        {
            Grad_R.push_back(ip_Jac[ip]*dRdXi[ni]);
            grad_R.push_back(dXidx.transpose()*dRdXi[ni]);
        }
        //
        //---------------------------------//

        //std::cout<<"deformation gradient\n"<<FF<<"\n";
        //std::cout<<"rho0: "<<rho_0<<", rho: "<<rho<<"\n";
        //std::cout<<"c0: "<<c_0<<", c: "<<c<<"\n";
        //std::cout<<"gradient of rho: "<<Grad_rho<<"\n";
        //std::cout<<"gradient of c: "<<Grad_c<<"\n";



        //---------------------------------//
        // LOCAL VARIABLES
        //
        // rename variables to make it easier to track
        double phif_0 = ip_phif_0[ip];
        Vector3d a0_0 = ip_a0_0[ip];
        Vector3d s0_0 = ip_s0_0[ip];
        Vector3d n0_0 = ip_n0_0[ip];
        double kappa_0 = ip_kappa_0[ip];
        Vector3d lamdaP_0 = ip_lamdaP_0[ip];
        double lamdaP_a_0 = lamdaP_0(0);
        double lamdaP_s_0 = lamdaP_0(1);
        double lamdaP_N_0 = lamdaP_0(2);

        double phif = ip_phif[ip];
        Vector3d a0 = ip_a0[ip];
        Vector3d s0 = ip_s0[ip];
        Vector3d n0 = ip_n0[ip];
        double kappa = ip_kappa[ip];
        Vector3d lamdaP = ip_lamdaP[ip];
        double lamdaP_a = lamdaP(0);
        double lamdaP_s = lamdaP(1);
        double lamdaP_N = lamdaP(2);
        //std::cout<<"Local variables after update:\nphif0 = "<<phif_0<<",	phif = "<<phif<<"\nkappa_0 = "<<kappa_0<<",	kappa = "<<kappa<<"\ns0_0 = ["<<s0_0(0)<<","<<s0_0(1)<<","<<s0_0(2)<<"],	s0 = ["<<s0(0)<<","<<s0(1)<<","<<s0(2)<<"\na0_0 = ["<<a0_0(0)<<","<<a0_0(1)<<","<<a0_0(2)<<"],	a0 = ["<<a0(0)<<","<<a0(1)<<","<<a0(2)<<"]\nlamdaP_0 = ["<<lamdaP_0(0)<<","<<lamdaP_0(1)<<","<<lamdaP_0(2)<<"],	lamdaP = ["<<lamdaP(0)<<","<<lamdaP(1)<<","<<lamdaP(2)<<"]\n";

        //---------------------------------//
        // CALCULATE SOURCE AND FLUX
        //
        // Update kinematics
        CCinv = CC.inverse();
        // fiber tensor in the reference
        Matrix3d a0a0 = a0*a0.transpose();
        Matrix3d s0s0 = s0*s0.transpose();
        Matrix3d n0n0 = n0*n0.transpose();
        // recompute split
        Matrix3d FFg = lamdaP_a*(a0a0) + lamdaP_s*(s0s0) + lamdaP_N*(n0n0);
        Matrix3d FFginv = (1./lamdaP_a)*(a0a0) + (1./lamdaP_s)*(s0s0) + (1./lamdaP_N)*(n0n0);
        Matrix3d FFe = FF*FFginv;
        // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
        // elastic strain
        Matrix3d CCe = FFe.transpose()*FFe;
        Matrix3d CCeinv = CCe.inverse();
        ip_strain[ip] = 0.5*(CCe - Identity);

        // invariant of the elastic strain
        double I1e = CCe(0,0) + CCe(1,1) + CCe(2,2);
        double I4e = a0.dot(CCe*a0);
        double I4tot = a0.dot(CC*a0);

        // Jacobian of the deformations
        double Jp = lamdaP_a*lamdaP_s*lamdaP_N;
        double Je = sqrt(CCe.determinant());
        double J = Je*Jp;

        // Structure tensor
        Matrix3d A0 = kappa*Identity + (1-3.*kappa)*a0a0;
        Vector3d a = FF*a0;
        Matrix3d A = kappa*FF*FF.transpose() + (1.-3.0*kappa)*a*a.transpose();
        double trA = A(0,0) + A(1,1) + A(2,2);
        Matrix3d hat_A = A/trA;

        //------------------//
        // PASSIVE STRESS
        //------------------//
        double Psif = (kf/(2.*k2))*(exp(k2*pow((kappa*I1e + (1-3*kappa)*I4e - 1),2)));
        double Psif1 = 2*k2*kappa*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif;
        double Psif4 = 2*k2*(1-3*kappa)*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif;
        //Matrix3d SSe_pas = k0*Identity + phif*(Psif1*Identity + Psif4*a0a0);
        Matrix3d SSe_pas = phif*(k0*Identity + Psif1*Identity + Psif4*a0a0);
        // pull back to the reference,
        Matrix3d SS_pas = Jp*FFginv*SSe_pas*FFginv;

        //------------------//
        // ACTIVE STRESS
        //------------------//
        double traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho;
        Matrix3d SS_act = (Jp*traction_act*phif/(trA*(K_t*K_t+phif*phif)))*A0;
        // Matrix3d SS_act = (Jp*traction_act*phif/trA)*A0;
        //double traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho;
        //Matrix3d SS_act = (Jp*traction_act*phif/(trA*(K_t*K_t+phif*phif)))*A0;

        //------------------//
        // VOLUME STRESS
        //------------------//
        // Instead of (double pressure = -k0*lamda_N*lamda_N;) directly, add volumetric part of stress SSvol
        // SSvol = 2dPsivol/dCC = 2dPsivol/dJe * dJe/dCC
        double penalty = 0.3166667;
        double Psivol = 0.5*phif*pow(penalty*(Je-1.),2) - 2*phif*k0*log(Je); //*phif
        double dPsivoldJe = phif*penalty*(Je-1.) - 2*phif*k0/Je;
        double dPsivoldJedJe = phif*penalty + 2*phif*k0/(Je*Je);
        Matrix3d SSe_vol = dPsivoldJe*Je*CCeinv/2; // = phif*penalty*Je*(Je-1.)*CCeinv/2 - phif*k0*CCeinv;
        Matrix3d SS_vol = Jp*FFginv*SSe_vol*FFginv;

        //------------------//
        // TOTAL STRESS
        //------------------//
        // std::cout<<"stresses.\nSSpas\n"<<SS_pas<<"\nSS_act\n"<<SS_act<<"\nSS_vol"<<SS_vol<<"\n";
        Matrix3d SS = SS_pas + SS_vol + SS_act;
        ip_stress[ip] = SS;
        //double SSvm = sqrt(0.5*(pow((SS(0,0)-SS(1,1)),2)+pow((SS(1,1)-SS(2,2)),2)+pow((SS(2,2)-SS(0,0)),2))+3*(pow(SS(0,1),2)+pow(SS(1,2),2)+pow(SS(2,0),2)));
        VectorXd SS_voigt(6);
        SS_voigt(0) = SS(0,0); SS_voigt(1) = SS(1,1); SS_voigt(2) = SS(2,2);
        SS_voigt(3) = SS(1,2); SS_voigt(4) = SS(0,2); SS_voigt(5) = SS(0,1);

        //---------------------------------//
        // ADD TO THE RESIDUAL
        //---------------------------------//

        // Element residual for x
        Matrix3d deltaFF,deltaCC;
        VectorXd deltaCC_voigt(6);
        for(int nodei=0;nodei<elem_size;nodei++){
            Vector3d b = Vector3d(bx,by,bz);
            for(int coordi=0;coordi<3;coordi++){
                // Internal force, alternatively, define the deltaCC
                deltaFF = Ebasis[coordi]*Grad_R[nodei].transpose();
                deltaCC = deltaFF.transpose()*FF + FF.transpose()*deltaFF;
                deltaCC_voigt(0) = deltaCC(0,0); deltaCC_voigt(1) = deltaCC(1,1); deltaCC_voigt(2) = deltaCC(2,2);
                deltaCC_voigt(3) = 2.*deltaCC(1,2); deltaCC_voigt(4) = 2.*deltaCC(0,2); deltaCC_voigt(5) = 2.*deltaCC(0,1);
                Re_x(nodei*3+coordi) += Jac*SS_voigt.dot(deltaCC_voigt)*wip;
                // External force/load vector, comment if not needed
                if(time < 1.){
                    Re_x(nodei*3+coordi) -= Jac*b(coordi)*R[nodei]*wip*time/1.;
                } else{
                    Re_x(nodei*3+coordi) -= Jac*b(coordi)*R[nodei]*wip;
                }
            }
        }
        //

        //---------------------------------//
        // TANGENTS
        //---------------------------------//


        //---------------------------------//
        // MECHANICS TANGENT
        //
        double Psif11 = 2*k2*kappa*kappa*Psif+2*k2*kappa*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif1 ;
        double Psif14 = 2*k2*kappa*(1-3*kappa)*I4e*Psif + 2*k2*kappa*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif4;
        double Psif41 = 2*k2*(1-3*kappa)*kappa*Psif + 2*k2*(1-3*kappa)*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif1;
        double Psif44 = 2*k2*(1-3*kappa)*(1-3*kappa)*Psif + 2*k2*(1-3*kappa)*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif4;
        std::vector<double> dSSpasdCC_explicit(81,0.);
        std::vector<double> dSSvoldCC_explicit(81,0.);
        for(int ii=0;ii<3;ii++){
            for(int jj=0;jj<3;jj++){
                for(int kk=0;kk<3;kk++){
                    for(int ll=0;ll<3;ll++){
                        for(int pp=0;pp<3;pp++){
                            for(int rr=0;rr<3;rr++){
                                for(int ss=0;ss<3;ss++){
                                    for(int tt=0;tt<3;tt++){
                                        // Explicit passive mechanics tangent
                                        dSSpasdCC_explicit[ii*27+jj*9+kk*3+ll] += Jp*(phif*(Psif11*Identity(pp,rr)*Identity(ss,tt) +
                                                                                            Psif14*Identity(pp,rr)*a0a0(ss,tt) + Psif41*a0a0(pp,rr)*Identity(ss,tt) +
                                                                                            Psif44*a0a0(pp,rr)*a0a0(ss,tt)))*FFginv(ii,pp)*FFginv(jj,rr)*FFginv(kk,ss)*FFginv(ll,tt);

                                        // Explicit volumetric mechanics tangent
                                        dSSvoldCC_explicit[ii*27+jj*9+kk*3+ll] += Jp/4*(-2*Je*dPsivoldJe*(0.5*(CCeinv(pp,ss)*CCeinv(rr,tt)+CCeinv(pp,tt)*CCeinv(rr,ss)))
                                                                                        +Je*(dPsivoldJe+Je*dPsivoldJedJe)*CCeinv(pp,rr)*CCeinv(ss,tt))*FFginv(ii,pp)*FFginv(jj,rr)*FFginv(kk,ss)*FFginv(ll,tt);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        //--------------------------------------------------//
        // build DD, the voigt version of CCCC = dSS_dCC
        // voigt table
        VectorXd voigt_table_I_i(6);
        voigt_table_I_i(0) = 0; voigt_table_I_i(1) = 1; voigt_table_I_i(2) = 2;
        voigt_table_I_i(3) = 1; voigt_table_I_i(4) = 2; voigt_table_I_i(5) = 0;
        VectorXd voigt_table_I_j(6);
        voigt_table_I_j(0) = 0; voigt_table_I_j(1) = 1; voigt_table_I_j(2) = 2;
        voigt_table_I_j(3) = 2; voigt_table_I_j(4) = 0; voigt_table_I_j(5) = 1;

        // some things needed first
        // only derivatives with respect to CC explicitly
        Matrix3d dtrAdCC = kappa*Identity + (1-3*kappa)*a0a0;
        MatrixXd DDvol(6,6),DDpas(6,6),DDact(6,6),DDstruct(6,6),DDtot(6,6);
        DDvol.setZero(); DDpas.setZero(); DDact.setZero(); DDstruct.setZero(); DDtot.setZero();

        //--------------------------------------------------//
        // CHECKING
//        Matrix3d CC_p,CC_m,SSpas_p,SSpas_m,SSvol_p,SSvol_m,SSact_p,SSact_m;
//        MatrixXd DDvol_num(6,6),DDpas_num(6,6),DDact_num(6,6),DDtot_num(6,6);
//        DDvol_num.setZero(); DDpas_num.setZero(); DDact_num.setZero(); DDtot_num.setZero();
        //--------------------------------------------------//

        for(int II=0;II<6;II++){
            for(int JJ=0;JJ<6;JJ++){
                int ii = voigt_table_I_i(II);
                int jj = voigt_table_I_j(II);
                int kk = voigt_table_I_i(JJ);
                int ll = voigt_table_I_j(JJ);

                // pressure, explicit  only
                DDvol(II,JJ) = dSSvoldCC_explicit[ii*27+jj*9+kk*3+ll];

                // passive, explicit only
                DDpas(II,JJ) = dSSpasdCC_explicit[ii*27+jj*9+kk*3+ll];

                // active , explicit only
                DDact(II,JJ) = -1.0*phif*traction_act*Jp/(trA*trA*(K_t*K_t+phif*phif))*dtrAdCC(kk,ll)*A0(ii,jj);
                //DDact(II,JJ) = -1.0*phif*traction_act*Jp/(trA*trA)*dtrAdCC(kk,ll)*A0(ii,jj);

                //--------------------------------------------------//
                // CHECKING
                // Numerical Solutions
//                CC_p = CC + 0.5*epsilon*Ebasis[kk]*Ebasis[ll].transpose() + 0.5*epsilon*Ebasis[ll]*Ebasis[kk].transpose();
//                CC_m = CC - 0.5*epsilon*Ebasis[kk]*Ebasis[ll].transpose() - 0.5*epsilon*Ebasis[ll]*Ebasis[kk].transpose();
//                evalSS(global_parameters,phif,a0,s0,n0,kappa,lamdaP_a,lamdaP_s,lamdaP_N,CC_p,rho,c,SSpas_p,SSact_p,SSvol_p);
//                evalSS(global_parameters,phif,a0,s0,n0,kappa,lamdaP_a,lamdaP_s,lamdaP_N,CC_m,rho,c,SSpas_m,SSact_m,SSvol_m);
//                DDpas_num(II,JJ) = (1./(2.0*epsilon))*(SSpas_p(ii,jj)-SSpas_m(ii,jj));
//                DDvol_num(II,JJ) = (1./(2.0*epsilon))*(SSvol_p(ii,jj)-SSvol_m(ii,jj));
//                DDact_num(II,JJ) = (1./(2.0*epsilon))*(SSact_p(ii,jj)-SSact_m(ii,jj));
//                DDtot_num(II,JJ) =  DDvol_num(II,JJ) + DDpas_num(II,JJ) + DDact_num(II,JJ);
                //--------------------------------------------------//

                // TOTAL. now include the structural parameters
                DDtot(II,JJ) =  DDvol(II,JJ) + DDpas(II,JJ) + DDact(II,JJ);
            }
        }

        //--------------------------------------------------//
        // CHECKING
        // PRINT ALL
        //std::cout<<"constitutive: \nDDpas\n"<<DDpas<<"\nDD_act\n"<<DDact<<"\nDD_struct"<<DDstruct<<"\n";
        // PRINT COMPARISON WITH NUMERICAL
//        if(fabs(Jp-1) > 0){
//            std::cout<<"comparing\nDD_pas\n";
//            std::cout<<DDpas<<"\nDD_pas_num\n"<<DDpas_num<<"\n";
//            std::cout<<"comparing\nDD_vol\n";
//            std::cout<<DDvol<<"\nDD_vol_num\n"<<DDvol_num<<"\n";
//            std::cout<<"comparing\nDD_act\n";
//            std::cout<<DDact<<"\nDD_act_num\n"<<DDact_num<<"\n";
//            std::cout<<"comparing\nDD_tot\n";
//            std::cout<<DDtot<<"\nDD_tot_num\n"<<DDtot_num<<"\n";
//        }
        //
        //--------------------------------------------------//
        //
        // some other declared variables
        Matrix3d linFF,linCC,lindeltaCC;
        VectorXd linCC_voigt(6),lindeltaCC_voigt(6);
        //
        // Loop over nodes and coordinates twice and assemble the corresponding entry
        for(int nodei=0;nodei<elem_size;nodei++){
            for(int coordi=0;coordi<3;coordi++){
                deltaFF = Ebasis[coordi]*Grad_R[nodei].transpose();
                deltaCC = deltaFF.transpose()*FF + FF.transpose()*deltaFF;
                //VectorXd deltaCC_voigt(6);
                deltaCC_voigt(0) = deltaCC(0,0); deltaCC_voigt(1) = deltaCC(1,1); deltaCC_voigt(2) = deltaCC(2,2);
                deltaCC_voigt(3) = 2.*deltaCC(1,2); deltaCC_voigt(4) = 2.*deltaCC(0,2); deltaCC_voigt(5) = 2.*deltaCC(0,1);
                for(int nodej=0;nodej<elem_size;nodej++){
                    for(int coordj=0;coordj<3;coordj++){

                        //-----------//
                        // Ke_X_X
                        //-----------//

                        // material part of the tangent
                        linFF =  Ebasis[coordj]*Grad_R[nodej].transpose();
                        linCC = linFF.transpose()*FF + FF.transpose()*linFF;
                        linCC_voigt(0) = linCC(0,0); linCC_voigt(1) = linCC(1,1); linCC_voigt(2) = linCC(2,2);
                        linCC_voigt(3) = 2.*linCC(1,2); linCC_voigt(4) = 2.*linCC(0,2); linCC_voigt(5) = 2.*linCC(0,1);
                        //
                        Ke_x_x(nodei*3+coordi,nodej*3+coordj) += Jac*deltaCC_voigt.dot(DDtot*linCC_voigt)*wip;
                        //
                        // geometric part of the tangent
                        lindeltaCC = deltaFF.transpose()*linFF + linFF.transpose()*deltaFF;
                        lindeltaCC_voigt(0) = lindeltaCC(0,0); lindeltaCC_voigt(1) = lindeltaCC(1,1); lindeltaCC_voigt(2) = lindeltaCC(2,2);
                        lindeltaCC_voigt(3) = 2.*lindeltaCC(1,2); lindeltaCC_voigt(4) = 2.*lindeltaCC(0,2); lindeltaCC_voigt(5) = 2.*lindeltaCC(0,1);
                        //
                        Ke_x_x(nodei*3+coordi,nodej*3+coordj) += Jac*SS_voigt.dot(lindeltaCC_voigt)*wip;

                    }
                }
            }
        }
    } // END INTEGRATION loop
}
