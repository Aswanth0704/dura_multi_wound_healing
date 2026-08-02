// Verification of include/mechanosensing.h
//   1. theta_e == 1 in the undeformed reference state
//   2. theta_e reproduces a known cylindrical prestretch (lam_th * lam_z)
//   3. theta_e == ||cof(F^e).n0|| computed directly from F^e
//   4. d theta_e / dCC and dH/dCC match central finite differences
//   5. H(vartheta_e) == 1/2 exactly
#include <cstdio>
#include <cmath>
#include <Eigen/Dense>
#include "mechanosensing.h"

using namespace Eigen;

static int failures = 0;
static void check(const char* what, double got, double want, double tol)
{
    const double err = std::fabs(got - want);
    const bool ok = err <= tol;
    if(!ok) ++failures;
    std::printf("  %-56s got % .10f  want % .10f  %s\n",
                what, got, want, ok ? "ok" : "FAIL");
}

// Reference implementation straight from the definition, for cross-checking.
static double thetaE_bruteforce(const Matrix3d& FF, const Vector3d& a0,
                                const Vector3d& s0, const Vector3d& n0,
                                const Vector3d& lamdaP)
{
    Matrix3d Fg = lamdaP(0)*a0*a0.transpose()
                + lamdaP(1)*s0*s0.transpose()
                + lamdaP(2)*n0*n0.transpose();
    Matrix3d Fe = FF*Fg.inverse();
    // cof(A) = det(A) A^-T
    Matrix3d cofFe = Fe.determinant()*Fe.inverse().transpose();
    return (cofFe*n0).norm();
}

int main()
{
    const Vector3d a0(0,0,1);     // axial
    const Vector3d s0(1,0,0);     // circumferential
    const Vector3d n0(0,1,0);     // through-thickness
    const double vartheta_e = 1.136, gamma_theta = 10.0;

    std::printf("\n[1] undeformed reference state: F = I, lamdaP = (1,1,1)\n");
    {
        Matrix3d FF = Matrix3d::Identity();
        Matrix3d CC = FF.transpose()*FF;
        double J = FF.determinant();
        double th = evalThetaE(J, CC.inverse(), n0, 1.0, 1.0);
        check("theta_e", th, 1.0, 1e-14);
        check("H(theta_e=1) with vartheta=1.136,gamma=10",
              evalHe(th, vartheta_e, gamma_theta),
              1.0/(1.0+std::exp(-gamma_theta*(1.0-vartheta_e))), 1e-14);
    }

    std::printf("\n[2] H at the midpoint must be exactly 1/2\n");
    check("H(vartheta_e)", evalHe(vartheta_e, vartheta_e, gamma_theta), 0.5, 1e-15);

    std::printf("\n[3] cylindrical prestretch: lam_th=1.035, lam_z=1.098, lam_r=1/(lam_th lam_z)\n");
    {
        const double lam_z = 1.098, lam_th = 1.035;
        const double lam_r = 1.0/(lam_z*lam_th);
        // F in the (a0=axial=z, s0=circumf=x, n0=thickness=y) basis
        Matrix3d FF = Matrix3d::Zero();
        FF(2,2) = lam_z;   // along a0
        FF(0,0) = lam_th;  // along s0
        FF(1,1) = lam_r;   // along n0
        Matrix3d CC = FF.transpose()*FF;
        double J = FF.determinant();
        check("det F (incompressible)", J, 1.0, 1e-14);
        double th = evalThetaE(J, CC.inverse(), n0, 1.0, 1.0);
        check("theta_e == lam_th*lam_z", th, lam_th*lam_z, 1e-13);
        check("theta_e == 1.136 (target)", th, 1.136, 1e-3);
        check("H(theta_e) == 1/2", evalHe(th, vartheta_e, gamma_theta), 0.5, 2e-3);
    }

    std::printf("\n[4] agreement with ||cof(F^e).n0|| for a general state\n");
    {
        Matrix3d FF;
        FF <<  1.07, 0.03, -0.02,
              -0.01, 0.94,  0.05,
               0.04, 0.02,  1.11;
        Vector3d lamdaP(1.03, 0.97, 1.01);
        Matrix3d CC = FF.transpose()*FF;
        double J = FF.determinant();
        double th_fast  = evalThetaE(J, CC.inverse(), n0, lamdaP(0), lamdaP(1));
        double th_brute = thetaE_bruteforce(FF, a0, s0, n0, lamdaP);
        check("evalThetaE vs ||cof(Fe).n0||", th_fast, th_brute, 1e-12);
    }

    std::printf("\n[5] d theta_e/dCC and dH/dCC vs central finite differences\n");
    {
        Matrix3d FF;
        FF <<  1.07, 0.03, -0.02,
              -0.01, 0.94,  0.05,
               0.04, 0.02,  1.11;
        Vector3d lamdaP(1.03, 0.97, 1.01);
        Matrix3d CC = FF.transpose()*FF;

        auto theta_of_CC = [&](const Matrix3d& C){
            return evalThetaE(std::sqrt(C.determinant()), C.inverse(), n0,
                              lamdaP(0), lamdaP(1));
        };
        double J  = std::sqrt(CC.determinant());
        double th = evalThetaE(J, CC.inverse(), n0, lamdaP(0), lamdaP(1));
        double He = evalHe(th, vartheta_e, gamma_theta);
        Matrix3d dth = evalDThetaEdCC(th, CC.inverse(), n0);
        Matrix3d dHe = evalDHedCC(th, He, gamma_theta, CC.inverse(), n0);

        // Convention-free check: for a symmetric direction W the directional
        // derivative is  d/dt theta(CC + tW)|0 = sum_ij (dtheta/dCC)_ij W_ij.
        // This avoids any ambiguity about how a symmetric CC is perturbed
        // component-wise (which double-counts the diagonal).
        const double eps = 1e-7;
        double worst_th = 0.0, worst_He = 0.0;
        Matrix3d dirs[4];
        dirs[0] = Matrix3d::Identity();
        dirs[1] << 1.0, 0.3, -0.2,  0.3, -0.7, 0.15, -0.2, 0.15, 0.45;
        dirs[2] << 0.0, 1.0,  0.0,  1.0,  0.0, 0.0,   0.0, 0.0,  0.0;
        dirs[3] << 0.2,-0.4,  0.6, -0.4,  1.1,-0.3,   0.6,-0.3, -0.9;
        for(int d=0;d<4;d++){
            Matrix3d W = 0.5*(dirs[d] + dirs[d].transpose()); // enforce symmetry
            Matrix3d Cp = CC + eps*W;
            Matrix3d Cm = CC - eps*W;
            double num_th = (theta_of_CC(Cp) - theta_of_CC(Cm))/(2.0*eps);
            double num_He = (evalHe(theta_of_CC(Cp), vartheta_e, gamma_theta)
                           - evalHe(theta_of_CC(Cm), vartheta_e, gamma_theta))/(2.0*eps);
            double ana_th = dth.cwiseProduct(W).sum();
            double ana_He = dHe.cwiseProduct(W).sum();
            std::printf("      dir %d: theta' num=% .8f ana=% .8f | H' num=% .8f ana=% .8f\n",
                        d, num_th, ana_th, num_He, ana_He);
            worst_th = std::max(worst_th, std::fabs(num_th - ana_th));
            worst_He = std::max(worst_He, std::fabs(num_He - ana_He));
        }
        check("max |analytic - FD| for d theta_e/dCC", worst_th, 0.0, 1e-6);
        check("max |analytic - FD| for dH/dCC",        worst_He, 0.0, 1e-6);
    }

    std::printf("\n[6] monotonicity: H must rise with membrane stretch\n");
    {
        double prev = -1.0; bool mono = true;
        for(double lam=0.95; lam<=1.35001; lam+=0.05){
            Matrix3d FF = Matrix3d::Zero();
            FF(2,2)=lam; FF(0,0)=lam; FF(1,1)=1.0/(lam*lam);
            Matrix3d CC = FF.transpose()*FF;
            double th = evalThetaE(FF.determinant(), CC.inverse(), n0, 1.0, 1.0);
            double H  = evalHe(th, vartheta_e, gamma_theta);
            std::printf("      lam=%.2f  theta_e=%.4f  H=%.4f\n", lam, th, H);
            if(H < prev) mono = false;
            prev = H;
        }
        check("H monotonically increasing in theta_e", mono ? 1.0 : 0.0, 1.0, 0.0);
    }

    std::printf("\n%s (%d failure%s)\n\n",
                failures ? "TESTS FAILED" : "ALL TESTS PASSED",
                failures, failures==1?"":"s");
    return failures ? 1 : 0;
}
