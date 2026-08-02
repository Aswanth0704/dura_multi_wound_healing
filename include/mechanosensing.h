/*
    MECHANOSENSING

    The mechanosensing stimulus is the in-plane AREAL stretch of the dural
    mid-surface, not the volumetric Jacobian:

        theta_e = lambda_1 lambda_2 = || cof(F^e) . n0 ||

    with n0 the through-thickness (surface normal) direction of the fiber frame.
    The tissue is treated as incompressible, so the thickness stretch is
    lambda_N = 1/theta_e and det(F^e) = 1 identically -- which is exactly why
    det(F^e) cannot be used as the stimulus: it does not respond to membrane
    stretch at all.

    Closed form.  Since Fg . n0 = lamdaP_N n0, we have
        n0 . CCe^-1 . n0 = lamdaP_N^2 (n0 . CC^-1 . n0)
    and with Je = J / (lamdaP_a lamdaP_s lamdaP_N) the lamdaP_N cancels:

        theta_e = J sqrt(n0 . CC^-1 . n0) / (lamdaP_a lamdaP_s)

    so the stimulus needs only CC, J, n0 and the two in-plane plastic stretches.

    Derivative (used for the consistent tangent), with g = n0 . CC^-1 . n0:

        d theta_e / dCC = (theta_e/2) [ CC^-1 - (CC^-1 n0)(CC^-1 n0)^T / g ]

    Mechanosensing response is the logistic

        H(theta_e) = 1 / (1 + exp(-gamma_e (theta_e - vartheta_e)))

    so H(vartheta_e) = 1/2 exactly.  With vartheta_e set to the measured dural
    areal prestretch (1.098 x 1.035 = 1.136) the healthy state sits at H = 1/2,
    which is the value every derived parameter in plan.md assumes.

        dH/dCC = gamma_e H (1-H) d theta_e / dCC

    This header is shared by wound.cpp and local_solver.cpp so the global
    residual/tangent and the local structural update cannot drift apart.
*/

#ifndef mechanosensing_h
#define mechanosensing_h

#include <cmath>
#include <Eigen/Dense>

//---------------------------------------------------------------//
// In-plane areal elastic stretch, theta_e = ||cof(F^e).n0||
//---------------------------------------------------------------//
inline double evalThetaE(double J, const Eigen::Matrix3d &CCinv,
                         const Eigen::Vector3d &n0,
                         double lamdaP_a, double lamdaP_s)
{
    const double g = n0.dot(CCinv * n0);
    // g > 0 for any admissible CC; guard only against round-off at g ~ 0.
    return J * std::sqrt(g > 0.0 ? g : 0.0) / (lamdaP_a * lamdaP_s);
}

//---------------------------------------------------------------//
// Mechanosensing response H(theta_e)
//---------------------------------------------------------------//
inline double evalHe(double theta_e, double vartheta_e, double gamma_theta)
{
    return 1.0 / (1.0 + std::exp(-gamma_theta * (theta_e - vartheta_e)));
}

//---------------------------------------------------------------//
// d theta_e / dCC
//---------------------------------------------------------------//
inline Eigen::Matrix3d evalDThetaEdCC(double theta_e, const Eigen::Matrix3d &CCinv,
                                      const Eigen::Vector3d &n0)
{
    const double g = n0.dot(CCinv * n0);
    if(g <= 0.0) return Eigen::Matrix3d::Zero();
    const Eigen::Vector3d Cin = CCinv * n0;
    return 0.5 * theta_e * (CCinv - (Cin * Cin.transpose()) / g);
}

//---------------------------------------------------------------//
// dH/dCC = gamma_e H (1-H) d theta_e / dCC
//---------------------------------------------------------------//
inline Eigen::Matrix3d evalDHedCC(double theta_e, double He, double gamma_theta,
                                  const Eigen::Matrix3d &CCinv,
                                  const Eigen::Vector3d &n0)
{
    return gamma_theta * He * (1.0 - He) * evalDThetaEdCC(theta_e, CCinv, n0);
}

#endif
