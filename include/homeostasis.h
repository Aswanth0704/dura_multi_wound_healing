/*
    HOMEOSTASIS

    Closed forms for the parameters that are DERIVED rather than sampled, so
    that the normalized state

        (alpha, rho, c, phi) = (0, 1, 1, 1)   with   H = H_h = 1/2

    is an exact fixed point of all four species equations.

    Three of the model's constants are not free: each is pinned by requiring a
    source term to vanish at homeostasis.

      K_rho_rho  from  s_rho = 0
      K_phi_rho  from  phi_dot = 0
      p_c_rho    from  s_c = 0        (the alpha term drops out since alpha_h = 0)

    Keeping them here, rather than as literals in the driver, means any later
    change to a sampled parameter automatically keeps the fixed point exact.
    plan.md records a GP fit that returned K_phi_rho = -0.205 from this same
    constraint, so the admissibility checks matter.

    The source-term evaluators mirror the expressions in wound.cpp (s_rho, s_c)
    and local_solver.cpp (phi_dot). The driver prints their residuals at startup,
    which is what catches a drift between this header and those files.
*/

#ifndef homeostasis_h
#define homeostasis_h

//---------------------------------------------------------------//
// Derived parameters
//---------------------------------------------------------------//

// Fibroblast carrying capacity, from s_rho = 0 at (rho_h, c_h, H_h).
inline double deriveKrhorho(double p_rho, double p_rho_c, double p_rho_theta,
                            double K_rho_c, double d_rho,
                            double rho_h, double c_h, double H_h)
{
    const double prolif_h = p_rho + p_rho_c*c_h/(K_rho_c + c_h) + p_rho_theta*H_h;
    return rho_h/(1.0 - d_rho/prolif_h);
}

// Collagen saturation by rho, from phi_dot = 0 at (rho_h, c_h, phi_h, H_h).
inline double deriveKphirho(double p_phi, double p_phi_c, double p_phi_theta,
                            double K_phi_c, double d_phi, double d_phi_rho_c,
                            double rho_h, double c_h, double phi_h, double H_h)
{
    const double prod = p_phi + p_phi_c*c_h/(K_phi_c + c_h) + p_phi_theta*H_h;
    const double deg  = d_phi + c_h*rho_h*d_phi_rho_c;
    return prod*rho_h/(deg*phi_h) - phi_h;
}

// Cytokine baseline production, from s_c = 0. One equation in p_c_rho and
// K_c_c; we sample K_c_c and back-solve p_c_rho, with the mechano term written
// as the ratio r_c_e = p_c_e / p_c_rho.
inline double derivePcrho(double d_c, double K_c_c, double r_c_e, double H_h)
{
    return d_c*(K_c_c + 1.0)/(1.0 + H_h*r_c_e);
}

//---------------------------------------------------------------//
// Source terms (mirroring wound.cpp / local_solver.cpp)
//---------------------------------------------------------------//

inline double evalSrho(double p_rho, double p_rho_c, double p_rho_theta,
                       double K_rho_c, double K_rho_rho, double d_rho,
                       double rho, double c, double He)
{
    return (p_rho + p_rho_c*c/(K_rho_c + c) + p_rho_theta*He)
           *(1.0 - rho/K_rho_rho)*rho - d_rho*rho;
}

inline double evalSc(double p_c_rho, double p_c_thetaE, double K_c_c, double d_c,
                     double p_c_alpha, double rho, double c, double alpha, double He)
{
    return (p_c_rho*c + p_c_thetaE*He)*(rho/(K_c_c + c)) - d_c*c + p_c_alpha*alpha;
}

inline double evalSalpha(double d_alpha, double alpha)
{
    return -d_alpha*alpha;
}

inline double evalPhidot(double p_phi, double p_phi_c, double p_phi_theta,
                         double K_phi_c, double K_phi_rho,
                         double d_phi, double d_phi_rho_c,
                         double rho, double c, double phi, double He)
{
    return (p_phi + p_phi_c*c/(K_phi_c + c) + p_phi_theta*He)*(rho/(K_phi_rho + phi))
           - (d_phi + c*rho*d_phi_rho_c)*phi;
}

#endif
