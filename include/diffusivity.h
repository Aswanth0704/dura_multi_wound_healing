/*
    FIBROBLAST DIFFUSIVITY

    D_rho depends on the local collagen volume fraction. Kept in a header so
    wound.cpp and the unit tests share one definition.
*/

#ifndef diffusivity_h
#define diffusivity_h

#include <cmath>
#include <cstdlib>

//--------------------------------------------------------//
// FIBROBLAST DIFFUSIVITY  D_rho(phif)
//--------------------------------------------------------//
//
// Single definition, used by evalWound, evalFluxesSources, evalQ and evalBC.
// NOTE: global_parameters[7] is the legacy D_rhorho slot and is NOT read -
// the diffusivity is defined entirely here.
//
//   Ppoly(phi) = A phi^5 + B phi^4 + C phi^3 + D phi^2 + E phi
//   C_up       = 1 - 1/(1+exp(-500 (phi-1)))       upper shutoff at healthy phi
//   C_low      = 0.5 (1 + tanh(k_cut (phi - 0.01)))  low-collagen gate
//   D_rho      = KD (1e-3 Ppoly(phi))^2 / 6 * C_up * C_low + D_floor
//
// Note Ppoly(0) = Ppoly(1) = 0 exactly, so without the floor the diffusivity
// would collapse to ~7e-7 mm^2/h in healthy tissue, i.e. fibroblasts would
// barely migrate there. The 6.12e-5 floor lifts that ~90x into the range
// plan.md cites for measured fibroblast motility.
//
// C_low REPLACES the older low-phi cutoff, which was obtained by evaluating the
// polynomial at (phi - phif00) instead. Stacking the two would suppress D_rho
// by a further ~3.3x at phi = 0.02, squarely inside the granulation-tissue
// range that governs the healing rate.
//
// Reference values (k_cut = 300):
//   phi   0.01      0.02      0.05      0.19(peak)  1.00
//   D_rho 2.32e-4   1.31e-3   6.00e-3   2.12e-2     6.12e-5
//
// Steepness of the low-collagen gate, overridable via WOUND_KCUT.
//
// At the default 300 this gate is very nearly a step: C_low runs
// 0.0025 -> 0.5 -> 0.9975 across phi = 0 -> 0.01 -> 0.02, with dC_low/dphi
// peaking at 150. A near-step in a coefficient of the transport operator is a
// candidate for the residual floor the healing runs stall against - Newton sees
// an almost-discontinuous system matrix wherever collagen sits near phi_cut.
// Lowering k_cut widens the transition without moving its midpoint, so a sweep
// says whether this gate is what limits convergence.
inline double drhoKcut()
{
    static const double v = [](){
        const char* e = std::getenv("WOUND_KCUT");
        return e ? std::atof(e) : 300.0;
    }();
    return v;
}

inline double evalDrho(double phif, double c)
{
    (void)c; // no chemotactic dependence in this form

    const double KD  =  1582.3;
    const double A   =   182.01;
    const double B   =  -655.0;
    const double C   =   875.66;
    const double D   =  -521.57;
    const double E   =   118.9;
    const double k_cut   = drhoKcut();
    const double phi_cut = 1e-2;
    const double D_floor = 6.12e-5;

    const double Ppoly = A*pow(phif,5) + B*pow(phif,4) + C*pow(phif,3)
                       + D*pow(phif,2) + E*phif;
    const double C_up  = 1.0 - (1.0/(1.0 + exp(-500.0*(phif - 1.0))));
    const double C_low = 0.5*(1.0 + tanh(k_cut*(phif - phi_cut)));

    return KD*(pow(Ppoly*0.001, 2)/6.0)*C_up*C_low + D_floor;
}


#endif
