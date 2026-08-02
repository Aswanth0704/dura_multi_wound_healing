// Verification of include/homeostasis.h
//
// The single most important invariant of the model: with the derived parameters
// in place, (alpha, rho, c, phi) = (0, 1, 1, 1) at H = 1/2 must be an EXACT
// fixed point of all four species equations. If it is not, healthy tissue
// drifts and nothing downstream is interpretable.
//
//   1. the derived values reproduce plan.md's median set (sweep #4 case 0)
//   2. all four source terms vanish to machine precision at homeostasis
//   3. the derived values stay admissible (K_phi_rho > 0, K_rho_rho > rho_h)
//   4. homeostasis survives a perturbation of any sampled parameter, i.e. the
//      derivation really is self-consistent rather than tuned
//   5. the fixed point is attracting in rho, c and phi (correct sign of the
//      restoring term on either side)
//   6. alpha decays with the expected half-life
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <initializer_list>
#include "homeostasis.h"

static int failures = 0;
static void check(const char* what, double got, double want, double tol)
{
    const bool ok = std::fabs(got - want) <= tol;
    if(!ok) ++failures;
    std::printf("  %-50s got % .6e  want % .6e  %s\n",
                what, got, want, ok ? "ok" : "FAIL");
}
static void checkTrue(const char* what, bool ok)
{
    if(!ok) ++failures;
    std::printf("  %-50s %s\n", what, ok ? "ok" : "FAIL");
}

// plan.md median set (sweep #4 case 0), normalized
struct P {
    double rho_h = 1.0, c_h = 1.0, phi_h = 1.0, alpha_h = 0.0, H_h = 0.5;
    double p_rho = 0.0154;
    double p_rho_c = 1.48*0.0154;
    double p_rho_theta = 0.109*0.0154;
    double K_rho_c = 1.31;
    double d_rho = 0.00369;
    double d_c = 0.00386;
    double K_c_c = 1.20;
    double r_c_e = 0.447;
    double p_phi = 9.34e-4;
    double p_phi_c = 1.41e-3;
    double p_phi_theta = 4.96*9.34e-4;
    double K_phi_c = 1.08;
    double d_phi = 2.02e-3;
    double d_phi_rho_c = 2.87e-4;
    double d_alpha = 0.0128;
    double p_c_alpha = 0.208;
};

struct Derived { double K_rho_rho, K_phi_rho, p_c_rho, p_c_thetaE; };
static Derived derive(const P& p)
{
    Derived d;
    d.K_rho_rho = deriveKrhorho(p.p_rho,p.p_rho_c,p.p_rho_theta,p.K_rho_c,p.d_rho,
                                p.rho_h,p.c_h,p.H_h);
    d.K_phi_rho = deriveKphirho(p.p_phi,p.p_phi_c,p.p_phi_theta,p.K_phi_c,
                                p.d_phi,p.d_phi_rho_c,p.rho_h,p.c_h,p.phi_h,p.H_h);
    d.p_c_rho    = derivePcrho(p.d_c,p.K_c_c,p.r_c_e,p.H_h);
    d.p_c_thetaE = p.r_c_e*d.p_c_rho;
    return d;
}

// worst |source term| at homeostasis
static double worstResidual(const P& p, const Derived& d)
{
    const double s_rho = evalSrho(p.p_rho,p.p_rho_c,p.p_rho_theta,p.K_rho_c,
                                  d.K_rho_rho,p.d_rho,p.rho_h,p.c_h,p.H_h);
    const double s_c   = evalSc(d.p_c_rho,d.p_c_thetaE,p.K_c_c,p.d_c,p.p_c_alpha,
                                p.rho_h,p.c_h,p.alpha_h,p.H_h);
    const double s_a   = evalSalpha(p.d_alpha,p.alpha_h);
    const double s_phi = evalPhidot(p.p_phi,p.p_phi_c,p.p_phi_theta,p.K_phi_c,
                                    d.K_phi_rho,p.d_phi,p.d_phi_rho_c,
                                    p.rho_h,p.c_h,p.phi_h,p.H_h);
    double w = std::fabs(s_rho);
    w = std::max(w, std::fabs(s_c));
    w = std::max(w, std::fabs(s_a));
    w = std::max(w, std::fabs(s_phi));
    return w;
}

int main()
{
    P p; Derived d = derive(p);

    std::printf("\n[1] derived values vs plan.md\n");
    check("K_rho_rho  (plan 1.16)",    d.K_rho_rho,  1.16,    5e-3);
    check("K_phi_rho  (plan 0.705)",   d.K_phi_rho,  0.705,   5e-3);
    check("p_c_rho    (plan 0.00695)", d.p_c_rho,    0.00695, 1e-5);
    check("p_c_thetaE (plan 0.00311)", d.p_c_thetaE, 0.00311, 1e-5);

    std::printf("\n[2] all four source terms vanish at (0,1,1,1), H=1/2\n");
    const double s_rho = evalSrho(p.p_rho,p.p_rho_c,p.p_rho_theta,p.K_rho_c,
                                  d.K_rho_rho,p.d_rho,p.rho_h,p.c_h,p.H_h);
    const double s_c   = evalSc(d.p_c_rho,d.p_c_thetaE,p.K_c_c,p.d_c,p.p_c_alpha,
                                p.rho_h,p.c_h,p.alpha_h,p.H_h);
    const double s_a   = evalSalpha(p.d_alpha,p.alpha_h);
    const double s_phi = evalPhidot(p.p_phi,p.p_phi_c,p.p_phi_theta,p.K_phi_c,
                                    d.K_phi_rho,p.d_phi,p.d_phi_rho_c,
                                    p.rho_h,p.c_h,p.phi_h,p.H_h);
    check("s_rho",   s_rho,   0.0, 1e-15);
    check("s_c",     s_c,     0.0, 1e-15);
    check("s_alpha", s_a,     0.0, 1e-18);
    check("s_phi",   s_phi,   0.0, 1e-15);

    std::printf("\n[3] admissibility (plan.md: a GP fit once returned -0.205)\n");
    checkTrue("K_phi_rho > 0",       d.K_phi_rho > 0.0);
    checkTrue("K_rho_rho > rho_h",   d.K_rho_rho > p.rho_h);

    std::printf("\n[4] homeostasis survives perturbing each sampled parameter\n");
    {
        const char* names[] = {"p_rho","p_rho_c","p_rho_theta","K_rho_c","d_rho",
                               "d_c","K_c_c","r_c_e","p_phi","p_phi_c",
                               "p_phi_theta","K_phi_c","d_phi","d_phi_rho_c"};
        double* slots[] = {&p.p_rho,&p.p_rho_c,&p.p_rho_theta,&p.K_rho_c,&p.d_rho,
                           &p.d_c,&p.K_c_c,&p.r_c_e,&p.p_phi,&p.p_phi_c,
                           &p.p_phi_theta,&p.K_phi_c,&p.d_phi,&p.d_phi_rho_c};
        double worst_all = 0.0;
        for(int i=0;i<14;i++){
            const double save = *slots[i];
            for(double f : {0.5, 1.7}){
                *slots[i] = save*f;
                Derived dd = derive(p);
                // skip combinations that are inadmissible by construction
                if(dd.K_phi_rho <= 0.0 || dd.K_rho_rho <= p.rho_h) continue;
                const double w = worstResidual(p, dd);
                if(w > worst_all) worst_all = w;
            }
            *slots[i] = save;
        }
        std::printf("      worst residual over 28 perturbations = %.3e\n", worst_all);
        check("still an exact fixed point", worst_all, 0.0, 1e-15);
    }

    std::printf("\n[5] the fixed point is attracting\n");
    {
        // rho: s_rho must push back toward 1
        const double lo = evalSrho(p.p_rho,p.p_rho_c,p.p_rho_theta,p.K_rho_c,
                                   d.K_rho_rho,p.d_rho,0.8,p.c_h,p.H_h);
        const double hi = evalSrho(p.p_rho,p.p_rho_c,p.p_rho_theta,p.K_rho_c,
                                   d.K_rho_rho,p.d_rho,1.1,p.c_h,p.H_h);
        std::printf("      s_rho(0.8) = %+.3e   s_rho(1.1) = %+.3e\n", lo, hi);
        checkTrue("s_rho > 0 below rho_h and < 0 above", lo > 0.0 && hi < 0.0);

        const double plo = evalPhidot(p.p_phi,p.p_phi_c,p.p_phi_theta,p.K_phi_c,
                                      d.K_phi_rho,p.d_phi,p.d_phi_rho_c,
                                      p.rho_h,p.c_h,0.5,p.H_h);
        const double phi2 = evalPhidot(p.p_phi,p.p_phi_c,p.p_phi_theta,p.K_phi_c,
                                       d.K_phi_rho,p.d_phi,p.d_phi_rho_c,
                                       p.rho_h,p.c_h,1.3,p.H_h);
        std::printf("      phi_dot(0.5) = %+.3e   phi_dot(1.3) = %+.3e\n", plo, phi2);
        checkTrue("phi_dot > 0 below phi_h and < 0 above", plo > 0.0 && phi2 < 0.0);

        const double clo = evalSc(d.p_c_rho,d.p_c_thetaE,p.K_c_c,p.d_c,p.p_c_alpha,
                                  p.rho_h,0.5,p.alpha_h,p.H_h);
        const double chi = evalSc(d.p_c_rho,d.p_c_thetaE,p.K_c_c,p.d_c,p.p_c_alpha,
                                  p.rho_h,1.5,p.alpha_h,p.H_h);
        std::printf("      s_c(0.5) = %+.3e   s_c(1.5) = %+.3e\n", clo, chi);
        checkTrue("s_c > 0 below c_h and < 0 above", clo > 0.0 && chi < 0.0);
    }

    std::printf("\n[6] alpha decay: s_alpha = -d_alpha alpha\n");
    {
        const double half_life = std::log(2.0)/p.d_alpha;
        std::printf("      half-life = %.1f h\n", half_life);
        check("half-life (h)", half_life, 54.15, 0.5);
        // integrate alpha' = -d_alpha alpha from 1 over one half-life
        double a = 1.0; const double dt = 1e-4;
        for(double t=0.0; t<half_life-0.5*dt; t+=dt) a += dt*evalSalpha(p.d_alpha,a);
        check("alpha after one half-life", a, 0.5, 1e-3);
        checkTrue("alpha stays non-negative", a > 0.0);
        check("s_alpha at alpha=0 (fixed point)", evalSalpha(p.d_alpha,0.0), 0.0, 0.0);
    }

    std::printf("\n%s (%d failure%s)\n\n",
                failures ? "TESTS FAILED" : "ALL TESTS PASSED",
                failures, failures==1?"":"s");
    return failures ? 1 : 0;
}
