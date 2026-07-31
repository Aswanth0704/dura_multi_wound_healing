/*

  Cylindrical dura mater, single needle-puncture wound.

  Two-phase driver:

    PHASE 1 - PRESTRETCH / SETTLING (no wound)
      Living dura sits taut. plan.md's Consolini measurement says excised dura
      shrinks by 1.098 axially and 1.035 circumferentially, so the mesh is the
      UNLOADED geometry and the in-vivo state is that mesh stretched. We
      prescribe that prestretch on every boundary node and let the tissue
      equilibrate while the biology sits at homeostasis. This is the gate: with
      theta_e = 1.136 the mechanosensing response is H = 1/2, which is the value
      every derived parameter assumes, so (rho,c,phi) = (1,1,1) must not drift.

    PHASE 2 - HEALING (wound seeded into the prestretched tissue)
      The prestretch is held on the boundary EXCEPT inside a free patch around
      the puncture, so the far field keeps theta_e = 1.136 / H = 1/2 while the
      wound is free to contract.

  All fields are NORMALIZED: rho_h = c_h = phi_h = 1 at homeostasis.

  Runtime overrides (no rebuild needed):
    WOUND_MESH      mesh file name              (default 20t_finer)
    WOUND_TSETTLE   phase-1 duration [h]        (default 100)
    WOUND_TFINAL    phase-2 duration [h]        (default 673 = 4 weeks)
    WOUND_OUT       output prefix               (default wound_1)
    WOUND_PATCH     free-patch radius / r_wound (default 4)
    WOUND_NOWOUND   set to skip phase 2 (homeostasis test only)
    WOUND_VERBOSE   full node/element/dof dumps
    WOUND_ALPHA_D     D_alpha   [mm^2/h]  (0 decouples alpha spatially)
    WOUND_ALPHA_DECAY d_alpha   [1/h]
    WOUND_ALPHA_PC    p_c_alpha [1/h]     (0 decouples alpha from cytokine)
    WOUND_DTRAMP    first (smallest) time step of the transient ladder [h]
    WOUND_RUNGS     steps per rung of that ladder (0 disables the ladder)
    WOUND_WSMOOTH   width of the smooth wound edge [mm]
    WOUND_TOL       Newton tolerance on the relative residual
    WOUND_TOL_INC   Newton tolerance on the increment norm (limit-cycle escape)
    WOUND_TRHO      active traction t_rho [MPa]; t_rho_c follows at 3.28571x
    WOUND_BANDW     smoothing width of the plastic-growth deadband edges
    WOUND_KCUT      steepness of the low-collagen gate in D_rho (default 300)
*/

#include <omp.h>
#include "file_io.h"
#include "wound.h"
#include "solver.h"
#include "myMeshGenerator.h"
#include "element_functions.h"
#include "local_solver.h"
#include "mechanosensing.h"
#include "homeostasis.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <ctime>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
using namespace Eigen;
// MKL is included through the CMake file

double frand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

//---------------------------------------------------------------------------//
// Local orthonormal frame for a cylinder about the z axis:
//   a0 axial, n0 radial (through-thickness), s0 circumferential.
//---------------------------------------------------------------------------//
inline void build_cylinder_frame(
    const Eigen::Vector3d& X,
    const Eigen::Vector3d& a0_in,
    double xc, double yc,
    Eigen::Vector3d& a0,
    Eigen::Vector3d& s0,
    Eigen::Vector3d& n0
){
    const double eps = 1e-12;

    a0 = a0_in;
    if(a0.norm() < eps) a0 = Eigen::Vector3d(0,0,1);
    a0.normalize();

    Eigen::Vector3d r(X(0) - xc, X(1) - yc, 0.0);
    if(r.norm() < eps){
        Eigen::Vector3d tmp = (std::abs(a0.dot(Eigen::Vector3d::UnitX())) < 0.9)
                              ? Eigen::Vector3d::UnitX()
                              : Eigen::Vector3d::UnitY();
        n0 = a0.cross(tmp);
    } else {
        n0 = r;
    }
    n0.normalize();

    s0 = n0.cross(a0);
    if(s0.norm() < eps){
        Eigen::Vector3d tmp = (std::abs(n0.dot(Eigen::Vector3d::UnitZ())) < 0.9)
                              ? Eigen::Vector3d::UnitZ()
                              : Eigen::Vector3d::UnitX();
        s0 = n0.cross(tmp);
    }
    s0.normalize();

    n0 = a0.cross(s0);
    n0.normalize();
}

//---------------------------------------------------------------------------//
// PHYSIOLOGICAL PRESTRETCH of a cylindrical shell.
//
// A plain diag(lam_th, lam_th, lam_z) is WRONG: its determinant is
// lam_th^2 lam_z = 1.176, so it inflates volume and THICKENS the shell, when
// incompressibility requires it to thin. The three stretches are distinct:
//   axial            lam_z  = 1.098
//   circumferential  lam_th = 1.035   -> mid-surface radius scales by lam_th
//   through-thickness lam_r = 1/(lam_z lam_th) -> offset from mid-surface scales
// which gives det F = 1 exactly and theta_e = ||cof F.n0|| = lam_th lam_z.
//---------------------------------------------------------------------------//
inline Eigen::Vector3d prestretch_target(const Eigen::Vector3d& X,
                                         double r_mid, double lam_th, double lam_z)
{
    const double lam_r = 1.0/(lam_z*lam_th);
    const double r = std::sqrt(X(0)*X(0) + X(1)*X(1));
    if(r < 1e-12) return Eigen::Vector3d(X(0), X(1), X(2)*lam_z);
    const double s  = r - r_mid;                    // signed through-thickness offset
    const double r2 = r_mid*lam_th + s*lam_r;       // new radius
    return Eigen::Vector3d(X(0)*r2/r, X(1)*r2/r, X(2)*lam_z);
}

//---------------------------------------------------------------------------//
// Diagnostic: recompute theta_e and H at every integration point directly from
// the tissue state, and report the spread of the nodal fields. Deliberately
// independent of the solver internals so it acts as a check on them.
//---------------------------------------------------------------------------//
static void reportState(const tissue& myTissue, const std::vector<Vector4d>& IP,
                        const char* label)
{
    const int elem_size = (int)myTissue.vol_elem_connectivity[0].size();
    const int IP_size   = (int)IP.size();

    double th_min= 1e30, th_max=-1e30, th_sum=0.0;
    double H_min = 1e30, H_max =-1e30, H_sum =0.0;
    double Je_min= 1e30, Je_max=-1e30;
    long   n=0;

    const double vartheta_e  = myTissue.global_parameters[16];
    const double gamma_theta = myTissue.global_parameters[17];

    for(int ei=0; ei<myTissue.n_vol_elem; ei++){
        const std::vector<int>& e = myTissue.vol_elem_connectivity[ei];
        for(int ip=0; ip<IP_size; ip++){
            const double xi=IP[ip](0), eta=IP[ip](1), zeta=IP[ip](2);
            std::vector<double> Rxi, Reta, Rzeta;
            if(elem_size==4){
                Rxi  = evalShapeFunctionsTetRxi(xi,eta,zeta);
                Reta = evalShapeFunctionsTetReta(xi,eta,zeta);
                Rzeta= evalShapeFunctionsTetRzeta(xi,eta,zeta);
            } else if(elem_size==10){
                Rxi  = evalShapeFunctionsTetQuadraticRxi(xi,eta,zeta);
                Reta = evalShapeFunctionsTetQuadraticReta(xi,eta,zeta);
                Rzeta= evalShapeFunctionsTetQuadraticRzeta(xi,eta,zeta);
            } else if(elem_size==8){
                Rxi  = evalShapeFunctionsRxi(xi,eta,zeta);
                Reta = evalShapeFunctionsReta(xi,eta,zeta);
                Rzeta= evalShapeFunctionsRzeta(xi,eta,zeta);
            } else if(elem_size==20){
                Rxi  = evalShapeFunctionsQuadraticRxi(xi,eta,zeta);
                Reta = evalShapeFunctionsQuadraticReta(xi,eta,zeta);
                Rzeta= evalShapeFunctionsQuadraticRzeta(xi,eta,zeta);
            } else {
                Rxi  = evalShapeFunctionsQuadraticLagrangeRxi(xi,eta,zeta);
                Reta = evalShapeFunctionsQuadraticLagrangeReta(xi,eta,zeta);
                Rzeta= evalShapeFunctionsQuadraticLagrangeRzeta(xi,eta,zeta);
            }
            Vector3d dxdxi=Vector3d::Zero(), dxdeta=Vector3d::Zero(), dxdzeta=Vector3d::Zero();
            for(int ni=0; ni<elem_size; ni++){
                dxdxi   += myTissue.node_x[e[ni]]*Rxi[ni];
                dxdeta  += myTissue.node_x[e[ni]]*Reta[ni];
                dxdzeta += myTissue.node_x[e[ni]]*Rzeta[ni];
            }
            Matrix3d dxdXi;
            dxdXi << dxdxi(0), dxdeta(0), dxdzeta(0),
                     dxdxi(1), dxdeta(1), dxdzeta(1),
                     dxdxi(2), dxdeta(2), dxdzeta(2);
            const Matrix3d FF = dxdXi*myTissue.elem_jac_IP[ei][ip].transpose();
            const Matrix3d CC = FF.transpose()*FF;
            const double J = FF.determinant();

            const int g = ei*IP_size+ip;
            const Vector3d& n0     = myTissue.ip_n0[g];
            const Vector3d& lamdaP = myTissue.ip_lamdaP[g];

            const double th = evalThetaE(J, CC.inverse(), n0, lamdaP(0), lamdaP(1));
            const double H  = evalHe(th, vartheta_e, gamma_theta);
            const double Je = J/(lamdaP(0)*lamdaP(1)*lamdaP(2));

            th_min=std::min(th_min,th); th_max=std::max(th_max,th); th_sum+=th;
            H_min =std::min(H_min ,H ); H_max =std::max(H_max ,H ); H_sum +=H;
            Je_min=std::min(Je_min,Je); Je_max=std::max(Je_max,Je);
            n++;
        }
    }

    auto span = [](const std::vector<double>& v, double& lo, double& hi, double& mean){
        lo=1e30; hi=-1e30; mean=0.0;
        for(double x : v){ lo=std::min(lo,x); hi=std::max(hi,x); mean+=x; }
        if(!v.empty()) mean/=(double)v.size();
    };
    double rlo,rhi,rmean, clo,chi,cmean, plo,phi_hi,pmean, alo,ahi,amean;
    span(myTissue.node_rho,   rlo,rhi,rmean);
    span(myTissue.node_c,     clo,chi,cmean);
    span(myTissue.ip_phif,    plo,phi_hi,pmean);
    span(myTissue.node_alpha, alo,ahi,amean);

    std::cout<<"\n================ STATE REPORT: "<<label<<" ================\n";
    std::cout<<std::fixed<<std::setprecision(6);
    std::cout<<"  theta_e  min/mean/max : "<<th_min<<" / "<<th_sum/n<<" / "<<th_max<<"\n";
    std::cout<<"  H        min/mean/max : "<<H_min <<" / "<<H_sum /n<<" / "<<H_max <<"\n";
    std::cout<<"  det(F^e) min/max      : "<<Je_min<<" / "<<Je_max<<"   (1 if incompressible)\n";
    std::cout<<"  rho      min/mean/max : "<<rlo<<" / "<<rmean<<" / "<<rhi<<"\n";
    std::cout<<"  c        min/mean/max : "<<clo<<" / "<<cmean<<" / "<<chi<<"\n";
    std::cout<<"  phi      min/mean/max : "<<plo<<" / "<<pmean<<" / "<<phi_hi<<"\n";
    std::cout<<"  alpha    min/mean/max : "<<alo<<" / "<<amean<<" / "<<ahi<<"\n";
    if(rlo<0.0 || clo<0.0 || plo<0.0 || alo<0.0)
        std::cout<<"  *** WARNING: negative concentration - solver has no clamping ***\n";
    std::cout<<"===========================================================\n\n";
    std::cout.unsetf(std::ios::fixed);
}

int main(int argc, char *argv[])
{
    Eigen::initParallel();
    std::cout<<"\nRunning full domain simulations with " << Eigen::nbThreads( ) << " threads.\n";
    srand (time(NULL));

    const bool verbose = (std::getenv("WOUND_VERBOSE") != nullptr);
    auto env_str = [](const char* key, const std::string& fallback){
        const char* v = std::getenv(key);
        return (v && *v) ? std::string(v) : fallback;
    };
    auto env_dbl = [](const char* key, double fallback){
        const char* v = std::getenv(key);
        return (v && *v) ? std::atof(v) : fallback;
    };

    //=======================================================================//
    // NORMALIZED HOMEOSTATIC STATE
    //=======================================================================//
    // Everything is normalized by its physiological value, so homeostasis is
    // (alpha, rho, c, phi) = (0, 1, 1, 1). This replaces the old dimensional
    // set (rho_phys = 1000*55.05126 cells/mm^3, c_max = 1e-4 g/mm^3).
    const double rho_h = 1.0;
    const double c_h   = 1.0;
    const double phi_h = 1.0;

    // values for the wound (plan.md "We are using only the normalized values")
    double rho_wound   = 1.0e-4;
    double c_wound     = 1.0e-4;
    double phif0_wound = 1.0e-2;
    // Injury releases the pro-inflammatory signal, so alpha starts HIGH in the
    // wound and decays (half-life ln2/d_alpha ~ 54 h), driving cytokine
    // production through p_c_alpha.
    double alpha_wound = 1.0;
    double kappa0_wound = 1./3;      // uniform dispersion

    // values for the healthy tissue
    double rho_healthy   = rho_h;
    double c_healthy     = c_h;
    // alpha_h = 0 exactly. Nothing divides by alpha, and zero is the exact fixed
    // point of s_alpha = -d_alpha alpha, so (0,1,1,1) is preserved and the
    // p_c_alpha term drops out of the derived p_c_rho.
    double alpha_healthy = 0.0;
    double phif0_healthy = phi_h;
    double kappa0_healthy = 0.024;   // fiber dispersion, from our own experiments

    //=======================================================================//
    // MECHANOSENSING
    //=======================================================================//
    // theta_e is the in-plane AREAL elastic stretch ||cof(F^e).n0||.
    // vartheta_e is the measured adult dural areal prestretch 1.098*1.035, so
    // H(vartheta_e) = 1/2 exactly in the healthy prestretched state. Every
    // derived parameter below assumes H_h = 1/2.
    double vartheta_e  = 1.136;
    double gamma_theta = 10.0;
    const double H_h   = 0.5;

    //=======================================================================//
    // GLOBAL PARAMETERS
    //=======================================================================//
    // --- mechanics (unchanged from the previous dimensional set) ---
    double k0 = 0.02;      // neo-Hookean ground substance [MPa]
    double kf = 40.0;      // collagen fiber stiffness [MPa]
    double k2 = 0.048;     // fiber exponential coefficient [-]

    // Active (cell-generated) traction. In wound.cpp this enters as
    //   traction_act = (t_rho + t_rho_c c/(K_t_c+c)) * rho
    // so with rho normalized to 1 these must absorb the old rho_phys factor:
    // the previous value 1.28571e-6/55.05126 multiplied by rho_phys = 55051.26
    // is 1.28571e-3, i.e. this conversion is traction-PRESERVING.
    //
    // Overridable so a traction sweep needs no rebuild. Active traction is the
    // stiffest coupling in the system - it feeds the mechanics block through
    // rho and c, so it is the first suspect whenever Newton struggles. Scale
    // both together with WOUND_TRHO; t_rho_c keeps its 3.28571 ratio to t_rho.
    double t_rho   = env_dbl("WOUND_TRHO", 1.28571e-3);  // [MPa] at rho = rho_h
    double t_rho_c = t_rho*3.28571;                      // enhancement by cytokine
    std::cout<<"active traction: t_rho="<<t_rho<<" t_rho_c="<<t_rho_c<<" MPa\n";
    double K_t     = 0.2;                     // saturation of traction by collagen
    double K_t_c   = c_h/10.0;                // saturation of traction by cytokine

    // --- transport ---
    double D_rhorho = 0.0;      // UNUSED: global_parameters[7] is never read.
                                // D_rho(phi) is defined by evalDrho() in wound.cpp.
    double D_rhoc   = 0.0;      // fibroblast chemotaxis - neglected in this model
    double D_cc     = 0.00930;  // cytokine diffusion [mm^2/h]

    // --- fibroblast kinetics [1/h] ---
    double p_rho       = 0.0154;            // baseline proliferation
    double p_rho_c     = 1.48*p_rho;        // cytokine-driven (ratio 1.48)
    double p_rho_theta = 0.109*p_rho;       // mechano-driven  (ratio 0.109)
    double K_rho_c     = 1.31;              // saturation of proliferation by c
    double d_rho       = 0.00369;           // apoptosis

    // --- cytokine kinetics [1/h] ---
    double d_c     = 0.00386;               // decay
    double K_c_c   = 1.20;                  // saturation of production by c
    double r_c_e   = 0.447;                 // p_c_e / p_c_rho (fixed ratio)

    double bx = 0.0, by = 0.0, bz = 0.0;    // body force

    // --- pro-inflammatory signal alpha ---
    // APPENDED at indices 25/26/27: wound.cpp unpacks global_parameters by
    // literal index at six separate sites, so inserting mid-vector would
    // silently corrupt every downstream read.
    // Overridable so the alpha equation can be validated in isolation: with
    // WOUND_ALPHA_D=0 and WOUND_ALPHA_PC=0 the field decouples completely and
    // must follow alpha(t) = alpha_0 exp(-d_alpha t) exactly at every node,
    // which tests the alpha residual and tangent against a closed form.
    double D_alpha   = env_dbl("WOUND_ALPHA_D",     0.00930); // [mm^2/h] = D_c
    double d_alpha   = env_dbl("WOUND_ALPHA_DECAY", 0.0128);  // [1/h]
    double p_c_alpha = env_dbl("WOUND_ALPHA_PC",    0.208);   // [1/h]

    //=======================================================================//
    // LOCAL PARAMETERS
    //=======================================================================//
    double p_phi       = 9.34e-4;           // baseline collagen production [1/h]
    double p_phi_c     = 1.41e-3;           // cytokine-driven
    double p_phi_theta = 4.96*p_phi;        // mechano-driven (ratio 4.96)
    double K_phi_c     = 1.08;              // saturation of deposition by c
                                            // NOT 1e-4: that was a c-scale artifact
    double d_phi       = 2.02e-3;           // baseline degradation [1/h]
    double d_phi_rho_c = 2.87e-4;           // degradation coupled to c and rho

    //=======================================================================//
    // DERIVED PARAMETERS - enforce (0,1,1,1) as an exact fixed point
    //=======================================================================//
    // Computed from the closed forms rather than hard-coded, so that any later
    // change to a sampled parameter automatically keeps homeostasis exact.

    // Closed forms live in include/homeostasis.h so the unit test and the
    // driver share one definition.
    double K_rho_rho = deriveKrhorho(p_rho,p_rho_c,p_rho_theta,K_rho_c,d_rho,
                                     rho_h,c_h,H_h);
    double K_phi_rho = deriveKphirho(p_phi,p_phi_c,p_phi_theta,K_phi_c,
                                     d_phi,d_phi_rho_c,rho_h,c_h,phi_h,H_h);
    double p_c_rho    = derivePcrho(d_c,K_c_c,r_c_e,H_h);
    double p_c_thetaE = r_c_e*p_c_rho;

    // Admissibility. plan.md warns that a GP fit once returned K_phi_rho =
    // -0.205 from this same constraint, so fail loudly rather than silently.
    if(!(K_phi_rho > 0.0))
        throw std::runtime_error("K_phi_rho <= 0: collagen production too weak "
                                 "relative to degradation for a physiological fixed point.");
    if(!(K_rho_rho > rho_h))
        throw std::runtime_error("K_rho_rho <= rho_h: apoptosis exceeds the "
                                 "homeostatic proliferation budget.");

    //---------------------------------//
    std::vector<double> global_parameters = {k0,kf,k2,t_rho,t_rho_c,K_t,K_t_c,
        D_rhorho,D_rhoc,D_cc,p_rho,p_rho_c,p_rho_theta,K_rho_c,K_rho_rho,d_rho,
        vartheta_e,gamma_theta,p_c_rho,p_c_thetaE,K_c_c,d_c,bx,by,bz,
        D_alpha,d_alpha,p_c_alpha};

    // fiber reorientation / dispersion time constants, co-scaled with K_phi_rho
    double tau_omega = 10./(K_phi_rho+1);
    double tau_kappa = 1./(K_phi_rho+1);
    double gamma_kappa = 5.;
    // permanent contracture/growth. NOTE: unlike tau_omega/tau_kappa these are
    // NOT co-scaled with K_phi_rho, so the plastic growth rate moves with the
    // renormalization. Revisit if lamdaP misbehaves.
    double tau_lamdaP_a = 0.05;
    double tau_lamdaP_s = 0.05;
    double tau_lamdaP_n = 0.05;

    double tol_local = 1e-8;        // inert in the explicit local solver
    // Local substeps per global step. The binding local timescale is
    // tau_lamdaP = 0.05 h, so local_dt must stay well under it; at 100 substeps
    // local_dt = 0.002 h is 25x finer than needed, and this is the dominant
    // runtime cost (100 x 79k IPs x ~6 Newton iterations per step).
    double time_step_ratio = env_dbl("WOUND_LOCALSUB", 25);
    double max_iter = 100;          // inert in the explicit local solver

    // Deadband for plastic growth: no remodelling while the elastic stretch is
    // inside [lamdaE_lo, lamdaE_hi]. This MUST contain the homeostatic elastic
    // stretches, which for the prestretched dura are
    //   (lam_z, lam_th, lam_r) = (1.098, 1.035, 0.880).
    // The old hard-coded 0.95/1.05 excluded both the axial and the
    // through-thickness value, so healthy tissue remodelled continuously and
    // slowly relaxed its own prestretch.
    double lamdaE_lo = 0.85;
    double lamdaE_hi = 1.15;
    // Width of the SMOOTH transition at each edge of that band. The band used to
    // be a hard if/else, which left lamdaP_dot continuous but its slope
    // discontinuous - a C0-but-not-C1 residual. That produced Newton limit
    // cycles AND broke the ILU preconditioner, so BiCGSTAB returned garbage.
    //
    // The width is bounded ABOVE by homeostasis, not by taste. A softplus edge
    // leaks a little growth into the band interior, and the healthy
    // through-thickness stretch lamdaE_n = 0.880 sits only 0.030 above
    // lamdaE_lo = 0.85 - the tightest approach of the three (axial has 0.052,
    // circumferential 0.115). The leak at that point is w*log1p(exp(-0.030/w)):
    //
    //     w = 0.010  ->  -4.9e-04   would remodel healthy tissue continuously,
    //                               which is exactly the bug the widened
    //                               deadband was introduced to fix
    //     w = 0.005  ->  -1.2e-05
    //     w = 0.002  ->  -6.1e-10   negligible
    //
    // so 0.002 it is: 15x narrower than the closest approach, yet still ~200
    // Newton increments wide (those run ~1e-5 in lamdaE near convergence), so
    // the transition reads as perfectly smooth to the solver. Set 0 to recover
    // the original hard threshold exactly.
    double lamdaE_bandw = env_dbl("WOUND_BANDW", 0.002);
    {
        const double lam_r_h = 1.0/(1.098*1.035);
        if(lamdaE_lo > lam_r_h || lamdaE_hi < 1.098)
            throw std::runtime_error("plastic-growth deadband excludes the "
                "physiological prestretch: healthy tissue would remodel.");
    }

    std::vector<double> local_parameters = {p_phi,p_phi_c,p_phi_theta,K_phi_c,
        K_phi_rho,d_phi,d_phi_rho_c,tau_omega,tau_kappa,gamma_kappa,
        tau_lamdaP_a,tau_lamdaP_s,tau_lamdaP_n,vartheta_e,gamma_theta,
        tol_local,time_step_ratio,max_iter,lamdaE_lo,lamdaE_hi,lamdaE_bandw};

    //---------------------------------//
    // Echo the parameter set and check the fixed point numerically.
    std::cout<<"\n===================== PARAMETERS (normalized) =====================\n";
    std::cout<<std::scientific<<std::setprecision(6);
    const char* gnames[] = {"k0","kf","k2","t_rho","t_rho_c","K_t","K_t_c",
        "D_rhorho(unused)","D_rhoc","D_cc","p_rho","p_rho_c","p_rho_theta",
        "K_rho_c","K_rho_rho*","d_rho","vartheta_e","gamma_theta","p_c_rho*",
        "p_c_thetaE*","K_c_c","d_c","bx","by","bz",
        "D_alpha","d_alpha","p_c_alpha"};
    for(size_t i=0;i<global_parameters.size();i++)
        std::cout<<"  global["<<std::setw(2)<<i<<"] "<<std::setw(18)<<gnames[i]
                 <<" = "<<global_parameters[i]<<"\n";
    const char* lnames[] = {"p_phi","p_phi_c","p_phi_theta","K_phi_c","K_phi_rho*",
        "d_phi","d_phi_rho_c","tau_omega*","tau_kappa*","gamma_kappa",
        "tau_lamdaP_a","tau_lamdaP_s","tau_lamdaP_n","vartheta_e","gamma_theta",
        "tol_local","time_step_ratio","max_iter","lamdaE_lo","lamdaE_hi","lamdaE_bandw"};
    for(size_t i=0;i<local_parameters.size();i++)
        std::cout<<"  local ["<<std::setw(2)<<i<<"] "<<std::setw(18)<<lnames[i]
                 <<" = "<<local_parameters[i]<<"\n";
    std::cout<<"  (* = derived to enforce homeostasis)\n";

    {
        // Residuals of all four source terms at (alpha,rho,c,phi)=(0,1,1,1),H=1/2.
        double S_rho   = evalSrho(p_rho,p_rho_c,p_rho_theta,K_rho_c,K_rho_rho,d_rho,
                                  rho_h,c_h,H_h);
        double S_c     = evalSc(p_c_rho,p_c_thetaE,K_c_c,d_c,p_c_alpha,
                                rho_h,c_h,alpha_healthy,H_h);
        double S_alpha = evalSalpha(d_alpha,alpha_healthy);
        double S_phi   = evalPhidot(p_phi,p_phi_c,p_phi_theta,K_phi_c,K_phi_rho,
                                    d_phi,d_phi_rho_c,rho_h,c_h,phi_h,H_h);
        std::cout<<"\n  Fixed-point residuals at (alpha,rho,c,phi)=(0,1,1,1), H=1/2:\n";
        std::cout<<"    s_rho   = "<<S_rho<<"\n    s_c     = "<<S_c
                 <<"\n    s_alpha = "<<S_alpha
                 <<"\n    s_phi   = "<<S_phi<<"\n";
        const double worst = std::max(std::max(std::fabs(S_rho),std::fabs(S_c)),
                                      std::max(std::fabs(S_alpha),std::fabs(S_phi)));
        if(worst > 1e-12)
            std::cout<<"  *** WARNING: homeostasis not exact (worst "<<worst<<") ***\n";
        else
            std::cout<<"  homeostasis exact to machine precision.\n";
    }
    std::cout<<"===================================================================\n\n";
    std::cout.unsetf(std::ios::scientific);

    //=======================================================================//
    // GEOMETRY AND MESH
    //=======================================================================//
    std::cout<<"Going to create the mesh\n";
    double r_cord  = 5.0;    // [mm] radius of the spinal cord
    double t_dura  = 0.4;    // [mm] dura thickness
    double r_wound = 0.25;   // [mm] 25-gauge needle
    double Xmin = -(r_cord + t_dura);
    double Xmax =  (r_cord + t_dura);
    double Ymin = Xmin, Ymax = Xmax;
    double Zmin = 0.0,  Zmax = 10.0;
    std::vector<double> hexDimensions = {Xmin, Xmax, Ymin, Ymax, Zmin, Zmax};
    std::vector<int> meshResolution =  {16,16,6};

    std::string mesh_filename = env_str("WOUND_MESH",
                                        "dura_cyl_repeated_wound_v62_2t_finer.mphtxt");
    std::cout<<"mesh file: "<<mesh_filename<<"\n";
    HexMesh myMesh = readCOMSOLInput(mesh_filename, hexDimensions, meshResolution);

    std::cout<<"Created the mesh with "<<myMesh.n_nodes<<" nodes and "
             <<myMesh.boundary_flag.size()<<" boundaries and "
             <<myMesh.n_elements<<" elements\n";
    if(verbose){
        std::cout<<"nodes\n";
        for(int nodei=0;nodei<myMesh.n_nodes;nodei++)
            std::cout<<myMesh.nodes[nodei](0)<<","<<myMesh.nodes[nodei](1)<<","
                     <<myMesh.nodes[nodei](2)<<"\n";
        std::cout<<"elements\n";
        for(int elemi=0;elemi<myMesh.n_elements;elemi++){
            for(size_t nodei=0;nodei<myMesh.elements[elemi].size();nodei++)
                std::cout<<myMesh.elements[elemi][nodei]<<" ";
            std::cout<<"\n";
        }
        std::cout<<"boundary\n";
        for(int nodei=0;nodei<myMesh.n_nodes;nodei++)
            std::cout<<myMesh.boundary_flag[nodei]<<"\n";
    }

    int elem_size = myMesh.elements[0].size();
    std::vector<Vector4d> IP;
    if(elem_size == 8 || elem_size == 20)      IP = LineQuadriIP();
    else if(elem_size == 27)                   IP = LineQuadriIPQuadratic();
    else if(elem_size == 4)                    IP = LineQuadriIPTet();
    else if(elem_size == 10)                   IP = LineQuadriIPTetQuadratic();
    else throw std::runtime_error("Wrong number of nodes in element!");
    int IP_size = IP.size();

    //=======================================================================//
    // WOUND LOCATION
    //=======================================================================//
    // A radial needle track: a cylinder whose axis is along -x, piercing the
    // dura wall at (y_center, z_center).
    // z_center is mid-length so the wound is far from both fixed end rings.
    // This matches Z1_CENTER = 5.0 in scripts/analyze_multiwound_centers.py;
    // the previous 2*t_dura = 0.8 sat almost on top of the clamped bottom ring.
    double tol_boundary = 1e-5;
    double y_center = 0.0;
    double z_center = 0.5*(Zmin + Zmax);
    // NOTE: the deformed-frame versions of these bounds are defined just after
    // the prestretch constants below, because seeding runs on node_x (deformed)
    // and reference-frame limits do not describe the settled geometry.

    //=======================================================================//
    // PRESTRETCH
    //=======================================================================//
    const double r_mid  = r_cord + 0.5*t_dura;   // mid-surface radius, 5.2 mm
    const double lam_z  = 1.098;                 // axial
    const double lam_th = 1.035;                 // circumferential
    std::cout<<"prestretch: lam_axial="<<lam_z<<" lam_circ="<<lam_th
             <<" lam_thick="<<1.0/(lam_z*lam_th)
             <<"  -> theta_e="<<lam_z*lam_th<<"\n";

    // Wound bounds in the DEFORMED frame.
    //
    // The wound is seeded after settling and its membership test runs on
    // myTissue.node_x - deliberately, because the puncture is made in vivo.
    // The bounds must therefore live in the same frame. Using the reference
    // limits here (as this used to) produced three coupled errors, all
    // measured from w_WOUNDCHECK.vtk:
    //
    //   1. z_center = 5.0 is the REFERENCE mid-length, but the settled mesh
    //      spans z in [0, 10.98], so its mid-length is 5.49. Testing deformed
    //      z against 5.0 put the wound at reference z = 4.554 - off the
    //      refined patch this mesh carries at z = 5.0 (549 nodes within
    //      +-0.25 mm instead of 1536).
    //   2. Xmin_wound = -(r_cord+t_dura) = -5.4 is the REFERENCE outer radius,
    //      while the settled outer surface sits at -5.55799. Every node beyond
    //      r = 5.4 failed the test, so the needle track pierced only the inner
    //      55% of the wall - a partial-thickness lesion, not a puncture.
    //   3. The free patch below tests myMesh.nodes (reference) against the same
    //      z_center, so patch and wound ended up 0.45 mm apart and the
    //      contraction boundary was asymmetric about the track.
    //
    // Mapping the reference limits through the same prestretch fixes all three.
    const double lam_r_pre  = 1.0/(lam_z*lam_th);
    const double r_in_def   = r_mid*lam_th + (r_cord          - r_mid)*lam_r_pre;
    const double r_out_def  = r_mid*lam_th + (r_cord + t_dura - r_mid)*lam_r_pre;
    const double z_center_def = z_center*lam_z;
    const double Xmin_wound = -r_out_def - tol_boundary;
    const double Xmax_wound = -r_in_def  + 0.01;
    std::cout<<"wound bounds (deformed frame): x in ["<<Xmin_wound<<", "
             <<Xmax_wound<<"], z_center "<<z_center_def
             <<" (reference "<<z_center<<")\n";

    // Identify the outer boundary: the two end rings (already flagged by
    // readCOMSOLInput via z) plus the inner and outer lateral surfaces, which
    // we find geometrically.
    const double r_in  = r_cord;
    const double r_out = r_cord + t_dura;
    const double tol_r = 1e-3;
    std::vector<char> on_boundary(myMesh.n_nodes, 0);
    int n_ring=0, n_lat=0;
    for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
        const Vector3d& X = myMesh.nodes[nodei];
        const double r = std::sqrt(X(0)*X(0)+X(1)*X(1));
        bool ring = (myMesh.boundary_flag[nodei] == 1);
        bool lat  = (std::fabs(r-r_in) < tol_r) || (std::fabs(r-r_out) < tol_r);
        if(ring) n_ring++;
        if(lat)  n_lat++;
        on_boundary[nodei] = (ring || lat) ? 1 : 0;
    }
    std::cout<<"boundary nodes: "<<n_ring<<" on end rings, "<<n_lat
             <<" on lateral surfaces\n";

    // Prestretched target position for every node, and the initial guess.
    std::vector<Vector3d> node_target(myMesh.n_nodes);
    for(int nodei=0;nodei<myMesh.n_nodes;nodei++)
        node_target[nodei] = prestretch_target(myMesh.nodes[nodei], r_mid, lam_th, lam_z);

    //=======================================================================//
    // INITIAL CONDITIONS - healthy everywhere (the wound comes in phase 2)
    //=======================================================================//
    std::vector<double> node_rho0(myMesh.n_nodes, rho_healthy);
    std::vector<double> node_c0  (myMesh.n_nodes, c_healthy);
    std::vector<double> node_alpha0(myMesh.n_nodes, alpha_healthy);
    std::vector<double> ip_phi0  (myMesh.n_elements*IP_size, phif0_healthy);
    std::vector<double> ip_kappa0(myMesh.n_elements*IP_size, kappa0_healthy);
    Vector3d a0_healthy(0.,0.,1.);            // collagen along the axis
    Vector3d lamda0_healthy(1.,1.,1.);
    std::vector<Vector3d> ip_a00(myMesh.n_elements*IP_size, a0_healthy);
    std::vector<Vector3d> ip_s00(myMesh.n_elements*IP_size, Vector3d(1.,0.,0.));
    std::vector<Vector3d> ip_n00(myMesh.n_elements*IP_size, Vector3d(0.,1.,0.));
    std::vector<Vector3d> ip_lamda0(myMesh.n_elements*IP_size, lamda0_healthy);

    // Cylindrical fiber frame at every integration point.
    for(int elemi=0;elemi<myMesh.n_elements;elemi++){
        for(int ip=0;ip<IP_size;ip++){
            std::vector<double> R;
            const double xi=IP[ip](0), eta=IP[ip](1), zeta=IP[ip](2);
            if(elem_size == 8)       R = evalShapeFunctionsR(xi,eta,zeta);
            else if(elem_size == 20) R = evalShapeFunctionsQuadraticR(xi,eta,zeta);
            else if(elem_size == 27) R = evalShapeFunctionsQuadraticLagrangeR(xi,eta,zeta);
            else if(elem_size == 4)  R = evalShapeFunctionsTetR(xi,eta,zeta);
            else                     R = evalShapeFunctionsTetQuadraticR(xi,eta,zeta);
            Vector3d X_IP = Vector3d::Zero();
            for(int nodej=0;nodej<elem_size;nodej++)
                X_IP += R[nodej]*myMesh.nodes[myMesh.elements[elemi][nodej]];
            Vector3d a0,s0,n0;
            build_cylinder_frame(X_IP, a0_healthy, 0.0, 0.0, a0, s0, n0);
            ip_a00[elemi*IP_size+ip] = a0;
            ip_s00[elemi*IP_size+ip] = s0;
            ip_n00[elemi*IP_size+ip] = n0;
        }
    }

    //=======================================================================//
    // BOUNDARY CONDITIONS - phase 1: prestretch held on the whole boundary
    //=======================================================================//
    std::map<int,double> eBC_x, eBC_rho, eBC_c, eBC_alpha;
    for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
        if(!on_boundary[nodei]) continue;
        if(verbose) std::cout<<"fixing node "<<nodei<<"\n";
        eBC_x.insert(std::pair<int,double>(nodei*3+0, node_target[nodei](0)));
        eBC_x.insert(std::pair<int,double>(nodei*3+1, node_target[nodei](1)));
        eBC_x.insert(std::pair<int,double>(nodei*3+2, node_target[nodei](2)));
        // hold the species at their healthy values on the end rings only;
        // the lateral surfaces are free surfaces for transport
        if(myMesh.boundary_flag[nodei] == 1){
            eBC_rho.insert(std::pair<int,double>(nodei, rho_healthy));
            eBC_c.insert  (std::pair<int,double>(nodei, c_healthy));
            eBC_alpha.insert(std::pair<int,double>(nodei, alpha_healthy));
        }
    }
    std::map<int,double> nBC_x, nBC_rho, nBC_c, nBC_alpha; // unused: never read by the solver

    //=======================================================================//
    // ASSEMBLE THE TISSUE
    //=======================================================================//
    tissue myTissue;
    myTissue.vol_elem_connectivity  = myMesh.elements;
    myTissue.surf_elem_connectivity = myMesh.surface_elements;
    myTissue.global_parameters = global_parameters;
    myTissue.local_parameters  = local_parameters;
    myTissue.boundary_flag = myMesh.boundary_flag;
    myTissue.surface_boundary_flag = myMesh.surface_boundary_flag;
    myTissue.node_X = myMesh.nodes;          // UNLOADED reference - never re-referenced
    myTissue.node_x = node_target;           // start at the prestretched guess
    myTissue.node_rho_0 = node_rho0;  myTissue.node_rho = node_rho0;
    myTissue.node_c_0   = node_c0;    myTissue.node_c   = node_c0;
    myTissue.node_alpha_0 = node_alpha0; myTissue.node_alpha = node_alpha0;
    myTissue.ip_phif_0  = ip_phi0;    myTissue.ip_phif  = ip_phi0;
    myTissue.ip_a0_0    = ip_a00;     myTissue.ip_a0    = ip_a00;
    myTissue.ip_s0_0    = ip_s00;     myTissue.ip_s0    = ip_s00;
    myTissue.ip_n0_0    = ip_n00;     myTissue.ip_n0    = ip_n00;
    myTissue.ip_kappa_0 = ip_kappa0;  myTissue.ip_kappa = ip_kappa0;
    myTissue.ip_lamdaP_0= ip_lamda0;  myTissue.ip_lamdaP= ip_lamda0;
    myTissue.ip_lamdaE  = ip_lamda0;
    myTissue.ip_strain = std::vector<Matrix3d>(myMesh.n_elements*IP_size, Matrix3d::Identity());
    myTissue.ip_stress = std::vector<Matrix3d>(myMesh.n_elements*IP_size, Matrix3d::Zero());
    myTissue.eBC_x = eBC_x;  myTissue.eBC_rho = eBC_rho;  myTissue.eBC_c = eBC_c;
    myTissue.eBC_alpha = eBC_alpha;
    myTissue.nBC_x = nBC_x;  myTissue.nBC_rho = nBC_rho;  myTissue.nBC_c = nBC_c;
    myTissue.nBC_alpha = nBC_alpha;
    myTissue.time       = 0.0;   // was never initialized: the solver reads it
    myTissue.time_step  = 0.2;
    // Newton tolerance on the RELATIVE residual normRR/(1+residuum0).
    //
    // 1e-8 turned out to sit at or below the accuracy floor of the linear
    // solve: rejected steps were reaching 3.6e-8 and stalling, i.e. Newton had
    // converged as far as the linear algebra allowed and the step was rejected
    // anyway, which sent dt into a halving spiral it could not escape (the
    // floor does not scale with dt).
    //
    // On accumulation: backward Euler is unconditionally stable here and the
    // homeostatic fixed point is ATTRACTING - tests/test_homeostasis.cpp checks
    // that s_rho, s_c and phi_dot all change sign in the restoring direction
    // either side of (1,1,1) - so a small per-step residual is damped rather
    // than amplified. That argument is verified empirically by rerunning the
    // 100 h no-wound gate at this tolerance and comparing against the completed
    // 1e-8 run; see ToDo.md.
    myTissue.tol        = env_dbl("WOUND_TOL", 1e-6);
    // Second convergence test, on the Newton increment, so a limit cycle at the
    // plastic-growth deadband kink does not stall the run. Default 1e-4 is the
    // L2 norm of the whole increment vector over every dof: at the observed
    // stall it is 9.3e-6 (a per-dof RMS of ~5e-8 on fields of order 1), while a
    // genuinely diverged step carries ~1e2. Four orders of headroom either way.
    myTissue.tol_inc    = env_dbl("WOUND_TOL_INC", 1e-4);
    // Newton converges LINEARLY (not quadratically) on the stiff
    // alpha-driven cytokine surge, reaching ~1e-6 by iteration 60. Rejecting
    // there costs a 5x dt cut; giving it more iterations is much cheaper, so
    // this is generous. An unconverged step is still rejected rather than
    // accepted.
    myTissue.max_iter   = (int)env_dbl("WOUND_MAXITER", 200);
    myTissue.n_node     = myMesh.n_nodes;
    myTissue.n_vol_elem = myMesh.n_elements;
    myTissue.n_surf_elem= myMesh.n_surf_elements;
    myTissue.n_IP       = IP_size*myMesh.n_elements;

    std::cout<<"filling dofs...\n";
    fillDOFmap(myTissue);
    std::cout<<"going to eval jacobians...\n";
    evalElemJacobians(myTissue);

    std::cout<<"element jacobians: "<<myTissue.elem_jac_IP.size()<<"\n";
    std::cout<<"Total :"<<myTissue.n_dof<<" dof\n";
    if(verbose){
        for(size_t i=0;i<myTissue.elem_jac_IP.size();i++){
            std::cout<<"element: "<<i<<"\n";
            for(int j=0;j<IP_size;j++)
                std::cout<<"ip; "<<j<<"\n"<<myTissue.elem_jac_IP[i][j]<<"\n";
        }
        for(size_t i=0;i<myTissue.dof_fwd_map_x.size();i++)
            std::cout<<"x node*3+coord: "<<i<<", dof: "<<myTissue.dof_fwd_map_x[i]<<"\n";
        for(size_t i=0;i<myTissue.dof_fwd_map_rho.size();i++)
            std::cout<<"rho node: "<<i<<", dof: "<<myTissue.dof_fwd_map_rho[i]<<"\n";
        for(size_t i=0;i<myTissue.dof_fwd_map_c.size();i++)
            std::cout<<"c node: "<<i<<", dof: "<<myTissue.dof_fwd_map_c[i]<<"\n";
    }

    std::vector<int> save_node; save_node.clear();
    std::vector<int> save_ip;   save_ip.clear();
    std::string out_prefix = env_str("WOUND_OUT", "wound_1");

    // Boundary-condition / initial-condition check before solving anything.
    {
        std::string f1 = out_prefix + "_BCCHECK.vtk";
        std::string f2 = out_prefix + "_second_BCCHECK.vtk";
        writeParaview(myTissue, f1.c_str(), f2.c_str());
        std::cout<<"wrote BC check: "<<f1<<"\n";
    }
    reportState(myTissue, IP, "initial guess (prestretch applied, before solve)");

    //=======================================================================//
    // PHASE 1: SETTLE THE PRESTRETCH (no wound)
    //=======================================================================//
    double t_settle = env_dbl("WOUND_TSETTLE", 100.0);
    if(t_settle > 0.0){
        myTissue.time = 0.0;
        myTissue.time_final = t_settle;
        std::string f = out_prefix + "_settle_";
        std::cout<<"\n#### PHASE 1: prestretch settling for "<<t_settle
                 <<" h (no wound) ####\n";
        sparseWoundSolver(myTissue, f, 5, save_node, save_ip);
        reportState(myTissue, IP, "after prestretch settling (HOMEOSTASIS GATE)");
    }

    if(std::getenv("WOUND_NOWOUND") != nullptr){
        std::cout<<"WOUND_NOWOUND set - stopping after the homeostasis gate.\n";
        return 0;
    }

    //=======================================================================//
    // PHASE 2: SEED THE WOUND AND HEAL
    //=======================================================================//
    // Release the boundary inside a patch around the puncture so the wound can
    // contract; the far field keeps holding theta_e = 1.136 / H = 1/2.
    const double patch_mult   = env_dbl("WOUND_PATCH", 4.0);
    const double patch_radius = patch_mult*r_wound;
    {
        std::map<int,double> eBC_x2, eBC_rho2, eBC_c2, eBC_alpha2;
        int n_freed = 0;
        for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
            if(!on_boundary[nodei]) continue;
            // Distance from the needle track, measured in the plane normal to
            // the track axis (which runs along x). Measured on the SETTLED
            // geometry against the same deformed-frame centre the wound
            // seeding uses, so the patch is concentric with the puncture -
            // testing reference coordinates here while seeding on deformed
            // ones left the two 0.45 mm apart.
            const Vector3d& X = myTissue.node_x[nodei];
            const double d2 = (X(1)-y_center)*(X(1)-y_center)
                            + (X(2)-z_center_def)*(X(2)-z_center_def);
            const bool near_wound = (d2 < patch_radius*patch_radius) && (X(0) < 0.0);
            if(near_wound){ n_freed++; continue; }
            eBC_x2.insert(std::pair<int,double>(nodei*3+0, node_target[nodei](0)));
            eBC_x2.insert(std::pair<int,double>(nodei*3+1, node_target[nodei](1)));
            eBC_x2.insert(std::pair<int,double>(nodei*3+2, node_target[nodei](2)));
            if(myMesh.boundary_flag[nodei] == 1){
                eBC_rho2.insert(std::pair<int,double>(nodei, rho_healthy));
                eBC_c2.insert  (std::pair<int,double>(nodei, c_healthy));
                eBC_alpha2.insert(std::pair<int,double>(nodei, alpha_healthy));
            }
        }
        std::cout<<"\nfree patch radius "<<patch_radius<<" mm ("<<patch_mult
                 <<"x r_wound): freed "<<n_freed<<" boundary nodes so the wound can contract\n";
        myTissue.eBC_x = eBC_x2;
        myTissue.eBC_rho = eBC_rho2;
        myTissue.eBC_c = eBC_c2;
        myTissue.eBC_alpha = eBC_alpha2;
        // Rebuild the dof maps for the new constraint set. fillDOFmap rebuilds
        // dof_fwd_map_*, dof_inv_map and n_dof from scratch, so this is safe.
        fillDOFmap(myTissue);
        std::cout<<"Total :"<<myTissue.n_dof<<" dof after releasing the patch\n";
    }

    // Seed the wound into the already-deformed tissue. Membership is tested on
    // the DEFORMED coordinates node_x, because the puncture is made in the
    // prestretched in-vivo configuration, not in the unloaded reference.
    // Seed the wound with a SMOOTH radial severity profile rather than a perfect
    // step. Seeding a discontinuity across one element makes the converged
    // Galerkin solution undershoot at the interface: with a sharp step the
    // solver converged to residual 2e-10 yet produced alpha as low as -0.072 and
    // c as low as -0.052, i.e. ~5-7% of each field's range. That is the classic
    // oscillation at a discontinuous initial condition, not a solver defect, and
    // the fix belongs in the initial condition. A damage gradient is also the
    // more faithful description of a needle track than a step.
    //
    //   sev(r) = 1/2 (1 - tanh((r - r_wound)/w))
    //
    // so sev ~ 1 on the axis, 1/2 at r_wound, ~0 beyond, over a transition of
    // roughly two elements.
    // The smoothing width must be RESOLVED by the mesh, otherwise the smooth
    // profile is itself a step as far as the discretization is concerned. A
    // first attempt used a fixed 0.06 mm, which on this mesh is only 0.67 of an
    // element edge (0.089 mm) and still produced undershoot. Scale it from the
    // actual element size instead, so it travels across meshes.
    double mean_edge = 0.0;
    {
        long ne = 0;
        const int nsample = std::min(myMesh.n_elements, 4000);
        for(int e=0;e<nsample;e++){
            const std::vector<int>& el = myMesh.elements[e];
            for(size_t a=0;a<el.size();a++)
                for(size_t b=a+1;b<el.size();b++){
                    mean_edge += (myMesh.nodes[el[a]] - myMesh.nodes[el[b]]).norm();
                    ne++;
                }
        }
        if(ne) mean_edge /= (double)ne;
    }
    const double w_smooth = env_dbl("WOUND_WSMOOTH", 1.7*mean_edge);   // [mm]
    std::cout<<"mean element edge "<<mean_edge<<" mm; wound radius "<<r_wound
             <<" mm ("<<r_wound/mean_edge<<" elements); smoothing width "
             <<w_smooth<<" mm ("<<w_smooth/mean_edge<<" elements)\n";
    if(w_smooth < 1.2*mean_edge)
        std::cout<<"  *** WARNING: smoothing width under-resolved; expect "
                   "Galerkin undershoot at the wound edge ***\n";
    // x is a DEFORMED position, so every bound here is a deformed-frame one.
    auto severity = [&](const Vector3d& x){
        if(x(0) < Xmin_wound || x(0) > Xmax_wound) return 0.0;
        const double r = std::sqrt((x(1)-y_center)*(x(1)-y_center)
                                 + (x(2)-z_center_def)*(x(2)-z_center_def));
        return 0.5*(1.0 - std::tanh((r - r_wound)/w_smooth));
    };
    auto blend = [](double healthy, double wound, double sev){
        return healthy + sev*(wound - healthy);
    };

    int n_wound_nodes = 0, n_wound_ip = 0;
    for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
        const double sev = severity(myTissue.node_x[nodei]);
        if(sev > 1e-3){
            if(verbose) std::cout << "wound node " << nodei << " sev " << sev << "\n";
            const double r = blend(rho_healthy,   rho_wound,   sev);
            const double c = blend(c_healthy,     c_wound,     sev);
            const double a = blend(alpha_healthy, alpha_wound, sev);
            myTissue.node_rho_0[nodei]   = r;  myTissue.node_rho[nodei]   = r;
            myTissue.node_c_0[nodei]     = c;  myTissue.node_c[nodei]     = c;
            myTissue.node_alpha_0[nodei] = a;  myTissue.node_alpha[nodei] = a;
            if(sev > 0.5) n_wound_nodes++;
        }
    }
    for(int elemi=0;elemi<myMesh.n_elements;elemi++){
        for(int ip=0;ip<IP_size;ip++){
            std::vector<double> R;
            const double xi=IP[ip](0), eta=IP[ip](1), zeta=IP[ip](2);
            if(elem_size == 8)       R = evalShapeFunctionsR(xi,eta,zeta);
            else if(elem_size == 20) R = evalShapeFunctionsQuadraticR(xi,eta,zeta);
            else if(elem_size == 27) R = evalShapeFunctionsQuadraticLagrangeR(xi,eta,zeta);
            else if(elem_size == 4)  R = evalShapeFunctionsTetR(xi,eta,zeta);
            else                     R = evalShapeFunctionsTetQuadraticR(xi,eta,zeta);
            Vector3d x_IP = Vector3d::Zero();
            for(int nodej=0;nodej<elem_size;nodej++)
                x_IP += R[nodej]*myTissue.node_x[myMesh.elements[elemi][nodej]];
            const double sev = severity(x_IP);
            if(sev > 1e-3){
                const int g = elemi*IP_size+ip;
                if(verbose) std::cout<<"IP node: "<<g<<" sev "<<sev<<"\n";
                Vector3d a0,s0,n0;
                build_cylinder_frame(x_IP, a0_healthy, 0.0, 0.0, a0, s0, n0);
                const double ph = blend(phif0_healthy,  phif0_wound,  sev);
                const double kp = blend(kappa0_healthy, kappa0_wound, sev);
                myTissue.ip_phif_0[g] = ph;  myTissue.ip_phif[g] = ph;
                myTissue.ip_a0_0[g]   = a0;  myTissue.ip_a0[g]   = a0;
                myTissue.ip_s0_0[g]   = s0;  myTissue.ip_s0[g]   = s0;
                myTissue.ip_n0_0[g]   = n0;  myTissue.ip_n0[g]   = n0;
                myTissue.ip_kappa_0[g]= kp;  myTissue.ip_kappa[g]= kp;
                if(sev > 0.5) n_wound_ip++;
            }
        }
    }
    std::cout<<"seeded wound: "<<n_wound_nodes<<" nodes, "<<n_wound_ip
             <<" integration points  (centre y="<<y_center<<" z="<<z_center_def
             <<" deformed, radius "<<r_wound<<" mm)\n";
    if(n_wound_nodes == 0 || n_wound_ip == 0)
        std::cout<<"  *** WARNING: wound region is empty - check the mesh and centre ***\n";

    reportState(myTissue, IP, "wound seeded, before healing solve");
    {
        std::string f1 = out_prefix + "_WOUNDCHECK.vtk";
        std::string f2 = out_prefix + "_second_WOUNDCHECK.vtk";
        writeParaview(myTissue, f1.c_str(), f2.c_str());
    }

    double t_heal = env_dbl("WOUND_TFINAL", (7*24*4)+1);

    //-------------------------------------------------------------------//
    // PHASE 2a: absorb the puncture transient with a small time step.
    //
    // Seeding the wound collapses the passive stress in those elements
    // (SSe_pas scales with phif, 1 -> 0.01), so the prestretched shell snaps
    // open. The physically required change in rho, c and phi over one 0.2 h
    // step is tiny - diffusion moves rho by ~5e-3 - but the concentration
    // block is only diagonally weighted by 1/dt = 5, and with the mechanics
    // simultaneously travelling a long way the Newton direction becomes
    // unreliable: the residual was observed jumping from 0.17 to 10.5 with a
    // requested concentration increment of ~190.
    //
    // Running the first fraction of an hour at dt_ramp makes 1/dt ~100x
    // larger, so the transport diagonal dominates and the transient is
    // resolved rather than fought. The cost is a few hundred cheap steps.
    //-------------------------------------------------------------------//
    // A GEOMETRIC dt LADDER, not a long run at one small step. The transient is
    // a one-off mechanical equilibration, so once it is absorbed the step can
    // grow straight back. 200 steps at a fixed dt = 0.002 h was ~4.75 h of wall
    // clock for 0.4 h of simulated time; the ladder below covers the same
    // transient in ~40 cheap steps.
    const double dt_normal = myTissue.time_step;
    // Steps per rung. The ladder must span the steep part of the
    // alpha-driven cytokine surge, not merely the mechanical snap-open:
    // p_c_alpha*alpha is ~48x the cytokine decay term, driving c from 1 toward
    // ~48 over a few hours, which is what stalled a 10-step ladder.
    const int    n_rung    = (int)env_dbl("WOUND_RUNGS",   40);  // steps per rung
    const double dt_start  = env_dbl("WOUND_DTRAMP", 0.002);
    double t_ramp_total = 0.0;
    if(dt_start > 0.0 && dt_start < dt_normal && n_rung > 0){
        std::cout<<"\n#### PHASE 2a: puncture transient, geometric dt ladder ####\n";
        int rung = 0;
        for(double dt = dt_start; dt < dt_normal; dt *= 5.0, ++rung){
            const double dt_use = std::min(dt, dt_normal);
            myTissue.time = 0.0;
            myTissue.time_step  = dt_use;
            myTissue.time_final = n_rung*dt_use;
            std::ostringstream fr;
            fr << out_prefix << "_ramp" << rung << "_";
            std::cout<<"  rung "<<rung<<": "<<n_rung<<" steps at dt = "<<dt_use
                     <<"  ("<<n_rung*dt_use<<" h)\n";
            sparseWoundSolver(myTissue, fr.str(), std::max(1, n_rung/2),
                              save_node, save_ip);
            t_ramp_total += n_rung*dt_use;
        }
        myTissue.time_step = dt_normal;
        reportState(myTissue, IP, "after the puncture transient");
        std::cout<<"  transient absorbed over "<<t_ramp_total<<" h\n";
    }

    myTissue.time = 0.0;
    myTissue.time_final = std::max(0.0, t_heal - t_ramp_total);
    std::string f = out_prefix + "_heal_";
    std::cout<<"\n#### PHASE 2b: healing for "<<myTissue.time_final
             <<" h at dt = "<<dt_normal<<" ####\n";
    sparseWoundSolver(myTissue, f, 5, save_node, save_ip);
    reportState(myTissue, IP, "after healing");

    return 0;
}
