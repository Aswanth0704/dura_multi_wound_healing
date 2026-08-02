// Verification of include/diffusivity.h  (D_rho(phi) with the C_low gate)
//   1. the floor is reached at phi = 0 and at healthy phi = 1
//   2. the plan.md reference values are reproduced
//   3. the interior peak sits near phi ~ 0.19 and is within the motility range
//   4. C_up shuts the polynomial term off above phi = 1
//   5. D_rho is strictly positive everywhere (a zero would kill the rho equation)
#include <cstdio>
#include <cmath>
#include <initializer_list>
#include "diffusivity.h"

static int failures = 0;
static void check(const char* what, double got, double want, double rtol)
{
    const double err = std::fabs(got - want);
    const double tol = rtol*std::fabs(want) + 1e-12;
    const bool ok = err <= tol;
    if(!ok) ++failures;
    std::printf("  %-52s got %.6e  want %.6e  %s\n",
                what, got, want, ok ? "ok" : "FAIL");
}

int main()
{
    const double D_floor = 6.12e-5;

    std::printf("\n[1] floor at the ends (Ppoly(0) = Ppoly(1) = 0 exactly)\n");
    check("D_rho(0.0)", evalDrho(0.0, 0.0), D_floor, 1e-9);
    check("D_rho(1.0)", evalDrho(1.0, 0.0), D_floor, 1e-3);

    std::printf("\n[2] plan.md reference values\n");
    check("D_rho(0.01)  wound collagen", evalDrho(0.01, 0.0), 2.32e-4, 2e-2);
    check("D_rho(0.02)",                 evalDrho(0.02, 0.0), 1.31e-3, 2e-2);
    check("D_rho(0.05)",                 evalDrho(0.05, 0.0), 6.00e-3, 2e-2);
    check("D_rho(1.00)  healthy floor",  evalDrho(1.00, 0.0), 6.12e-5, 2e-2);

    std::printf("\n[3] interior peak\n");
    double best = -1.0, best_phi = -1.0;
    for(int i=0;i<=1200;i++){
        const double phi = i/1000.0;
        const double D = evalDrho(phi, 0.0);
        if(D > best){ best = D; best_phi = phi; }
    }
    std::printf("      peak D_rho = %.6e at phi = %.3f\n", best, best_phi);
    check("peak location", best_phi, 0.19, 0.15);
    check("peak magnitude", best, 2.12e-2, 5e-2);
    // plan.md: in vitro 0.001-0.009, scaffold <= 0.018, in vivo <= 0.17 mm^2/h
    if(!(best < 0.17)){ ++failures; std::printf("  peak exceeds the in vivo bound  FAIL\n"); }
    else std::printf("  peak within the in vivo bound (<= 0.17)             ok\n");

    std::printf("\n[4] C_up shutoff above healthy collagen\n");
    check("D_rho(1.1)", evalDrho(1.1, 0.0), D_floor, 1e-6);
    check("D_rho(1.5)", evalDrho(1.5, 0.0), D_floor, 1e-6);

    std::printf("\n[5] strict positivity over phi in [0, 1.5]\n");
    bool positive = true; double lo = 1e30;
    for(int i=0;i<=1500;i++){
        const double D = evalDrho(i/1000.0, 0.0);
        if(!(D > 0.0)) positive = false;
        if(D < lo) lo = D;
    }
    std::printf("      min D_rho over the range = %.6e\n", lo);
    check("all values strictly positive", positive?1.0:0.0, 1.0, 0.0);

    std::printf("\n[6] shape: monotone rise then fall around the peak\n");
    {
        bool rise = true, fall = true;
        double prev = evalDrho(0.011, 0.0);
        for(double phi=0.012; phi<0.185; phi+=0.001){
            double D = evalDrho(phi,0.0); if(D < prev) rise = false; prev = D;
        }
        prev = evalDrho(0.20, 0.0);
        for(double phi=0.21; phi<0.99; phi+=0.01){
            double D = evalDrho(phi,0.0); if(D > prev) fall = false; prev = D;
        }
        check("increasing on (0.011, 0.185)", rise?1.0:0.0, 1.0, 0.0);
        check("decreasing on (0.20, 0.99)",   fall?1.0:0.0, 1.0, 0.0);
    }

    std::printf("\n[7] table over the healing range\n");
    for(double phi : {0.0,0.005,0.01,0.02,0.05,0.1,0.19,0.3,0.5,0.7,0.9,1.0,1.1})
        std::printf("      phi=%.3f  D_rho=%.6e\n", phi, evalDrho(phi,0.0));

    std::printf("\n%s (%d failure%s)\n\n",
                failures ? "TESTS FAILED" : "ALL TESTS PASSED",
                failures, failures==1?"":"s");
    return failures ? 1 : 0;
}
