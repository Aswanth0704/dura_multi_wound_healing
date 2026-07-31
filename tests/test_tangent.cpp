// Finite-difference verification of the element tangent produced by evalWound.
//
// Why this exists
// ---------------
// Three separate defects were each diagnosed as "the" reason the healing run
// stalls near t = 14 h - the plastic-growth threshold kink, the reference/
// deformed frame mismatch in wound seeding, and BiCGSTAB silently returning a
// near-zero increment. Each was real and each was fixed, and the run still
// decays: the time step falls from 0.2 to 0.008 and the linear solver starts
// failing. Every one of those diagnoses was inferred from symptoms.
//
// The tangent is not something to infer. Newton's method solves K dU = -R with
// K = dR/dU, and if K disagrees with R the iteration loses quadratic
// convergence and stalls at a floor - exactly the observed signature (a
// residual alternating between two values for hundreds of iterations). So
// measure it: perturb each nodal unknown, re-evaluate the residual, and compare
// against the corresponding column of K.
//
// The comparison is reported PER BLOCK (Ke_x_x, Ke_x_rho, ... 16 of them) so
// the output says WHICH coupling is wrong, not merely that something is.
//
// Two things this test has to get right
// -------------------------------------
//  1. evalWound MUTATES its ip_* arguments - the local structural solver writes
//     the updated phif/a0/s0/n0/kappa/lamdaP back through them. Every residual
//     evaluation therefore has to start from a pristine copy of the baseline
//     state, or each finite difference would be taken about a different point.
//  2. Central differences, not forward: the local solver runs a fixed number of
//     explicit substeps, so the residual carries O(eps) noise from substep
//     boundaries and a one-sided difference would report that noise as tangent
//     error.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "wound.h"
#include "element_functions.h"
#include "homeostasis.h"

using namespace Eigen;

static const int NNODE = 4;      // linear tetrahedron
static const int NDIM  = 3;

// ---------------------------------------------------------------------------
// Parameter block, mirroring the driver so the test exercises the real regime.
// ---------------------------------------------------------------------------
// Substep count of the explicit inner structural solver, overridable so the
// tangent error can be tested for discretisation dependence: the analytic
// tangent describes the CONTINUOUS remodelling law while the residual evaluates
// the DISCRETISED one, so if that gap is the cause the error should shrink as
// the substeps are refined, and sit still if a term is genuinely missing.
static double substeps()
{
    const char* e = std::getenv("TANGENT_SUBSTEPS");
    return e ? std::atof(e) : 25.0;
}

static void buildParameters(std::vector<double>& gp, std::vector<double>& lp)
{
    // Values copied EXACTLY from the driver. An earlier version of this test
    // guessed them, and several were an order of magnitude off (gamma_kappa
    // 0.05 against 5, p_phi 0.0135 against 9.34e-4), which drove the structural
    // update far outside its valid range and produced a non-finite tangent that
    // looked like a code defect. A verification test is only as good as the
    // regime it reproduces.
    const double rho_h = 1.0, c_h = 1.0, phi_h = 1.0, H_h = 0.5;

    const double k0 = 0.02, kf = 40.0, k2 = 0.048;
    const double t_rho = 1.28571e-3, t_rho_c = t_rho*3.28571;
    const double K_t = 0.2, K_t_c = c_h/10.0;
    const double D_rhorho = 0.0, D_rhoc = 0.0, D_cc = 0.00930;
    const double p_rho = 0.0154, p_rho_c = 1.48*p_rho, p_rho_theta = 0.109*p_rho;
    const double K_rho_c = 1.31, d_rho = 0.00369;
    const double vartheta_e = 1.136, gamma_theta = 10.0;
    const double d_c = 0.00386, K_c_c = 1.20, r_c_e = 0.447;
    const double D_alpha = 0.00930, d_alpha = 0.0128, p_c_alpha = 0.208;

    const double K_rho_rho  = deriveKrhorho(p_rho, p_rho_c, p_rho_theta,
                                            K_rho_c, d_rho, c_h, rho_h, H_h);
    const double p_c_rho    = derivePcrho(d_c, K_c_c, r_c_e, H_h);
    const double p_c_thetaE = r_c_e*p_c_rho;

    gp = { k0, kf, k2, t_rho, t_rho_c, K_t, K_t_c, D_rhorho, D_rhoc, D_cc,
           p_rho, p_rho_c, p_rho_theta, K_rho_c, K_rho_rho, d_rho,
           vartheta_e, gamma_theta, p_c_rho, p_c_thetaE, K_c_c, d_c,
           0.0, 0.0, 0.0, D_alpha, d_alpha, p_c_alpha };

    const double p_phi = 9.34e-4, p_phi_c = 1.41e-3, p_phi_theta = 4.96*p_phi;
    const double K_phi_c = 1.08, d_phi = 2.02e-3, d_phi_rho_c = 2.87e-4;
    const double K_phi_rho = deriveKphirho(p_phi, p_phi_c, p_phi_theta, K_phi_c,
                                           d_phi, d_phi_rho_c, c_h, rho_h, phi_h, H_h);
    const double tau_omega = 10.0/(K_phi_rho+1.0);
    const double tau_kappa =  1.0/(K_phi_rho+1.0);
    const double gamma_kappa = 5.0;
    lp = { p_phi, p_phi_c, p_phi_theta, K_phi_c, K_phi_rho, d_phi, d_phi_rho_c,
           tau_omega, tau_kappa, gamma_kappa,
           0.05, 0.05, 0.05,           // tau_lamdaP_a/s/n
           vartheta_e, gamma_theta,
           1e-8, substeps(), 100.0,    // tol_local, time_step_ratio, max_iter
           0.85, 1.15,                 // lamdaE_lo, lamdaE_hi
           0.002 };                    // lamdaE_bandw
}

// ---------------------------------------------------------------------------
// One element's state. Copied fresh for every residual evaluation because
// evalWound writes through its ip_* arguments.
// ---------------------------------------------------------------------------
struct State {
    std::vector<Vector3d> node_x, node_X;
    std::vector<double>   node_rho, node_c, node_alpha;
    std::vector<double>   ip_phif, ip_kappa;
    std::vector<Vector3d> ip_a0, ip_s0, ip_n0, ip_lamdaP;
};

static State baselineState(int nip, double phif0, double kappa0,
                           const Vector3d& lamdaP0)
{
    State S;
    // A tetrahedron of roughly the mesh's element size (~0.09 mm edges).
    S.node_X = { Vector3d(0.00, 0.00, 0.00), Vector3d(0.09, 0.00, 0.00),
                 Vector3d(0.00, 0.09, 0.00), Vector3d(0.00, 0.00, 0.09) };

    // Deformed by the physiological prestretch, in the local frame where
    // a0 = z (axial), s0 = x (circumferential), n0 = y (through-thickness),
    // plus a small shear so the state is not artificially symmetric.
    const double lam_z = 1.098, lam_th = 1.035, lam_r = 1.0/(lam_z*lam_th);
    Matrix3d F;
    F << lam_th, 0.012, 0.0,
         0.0,    lam_r, 0.008,
         0.006,  0.0,   lam_z;
    S.node_x.resize(NNODE);
    for(int i=0;i<NNODE;i++) S.node_x[i] = F*S.node_X[i];

    // Partly-healed wound state: depleted but recovering, so the sources and
    // the structural update are both genuinely active.
    S.node_rho   = { 0.62, 0.66, 0.71, 0.58 };
    S.node_c     = { 1.14, 1.09, 1.05, 1.18 };
    S.node_alpha = { 0.21, 0.17, 0.12, 0.26 };

    S.ip_phif.assign(nip, phif0);
    S.ip_kappa.assign(nip, kappa0);
    S.ip_a0.assign(nip, Vector3d(0,0,1));
    S.ip_s0.assign(nip, Vector3d(1,0,0));
    S.ip_n0.assign(nip, Vector3d(0,1,0));
    S.ip_lamdaP.assign(nip, lamdaP0);
    return S;
}

// Concatenated residual [Re_x (12) | Re_rho (4) | Re_c (4) | Re_alpha (4)].
// S0 supplies the PREVIOUS-step state and the integration-point baseline; S
// supplies the current unknowns being differentiated. Keeping them separate is
// essential: the residual is ((u - u_0)/dt - S)*R, so the tangent is dR/du at
// FIXED u_0. Perturbing both together cancels the R_i R_j/dt mass term exactly
// - which is how the first version of this test reported a spurious 6.075e-05
// error (= V/10/dt) in all three diagonal transport blocks.
static VectorXd residual(const State& S0, const State& S,
                         const std::vector<Matrix3d>& ip_Jac,
                         const std::vector<double>& gp, const std::vector<double>& lp,
                         double dt, int nip)
{
    // Fresh copies: evalWound updates these in place.
    std::vector<double>   phif = S0.ip_phif, kappa = S0.ip_kappa;
    std::vector<Vector3d> a0 = S0.ip_a0, s0 = S0.ip_s0, n0 = S0.ip_n0, lamdaP = S0.ip_lamdaP;
    std::vector<Vector3d> lamdaE(nip, Vector3d::Ones());
    std::vector<Matrix3d> strain(nip, Matrix3d::Zero()), stress(nip, Matrix3d::Zero());
    std::vector<Vector3d> dphifdu(nip, Vector3d::Zero());
    std::vector<double>   dphifdrho(nip,0.0), dphifdc(nip,0.0);

    VectorXd Re_x(12), Re_rho(4), Re_c(4), Re_alpha(4);
    MatrixXd Kxx(12,12), Kxr(12,4), Kxc(12,4), Kxa(12,4);
    MatrixXd Krx(4,12), Krr(4,4), Krc(4,4), Kra(4,4);
    MatrixXd Kcx(4,12), Kcr(4,4), Kcc(4,4), Kca(4,4);
    MatrixXd Kax(4,12), Kar(4,4), Kac(4,4), Kaa(4,4);

    evalWound(dt, 0.0, 1.0, ip_Jac, gp, lp, strain, stress,
              S0.node_rho, S0.node_c, S0.node_alpha,          // previous step
              S0.ip_phif, S0.ip_a0, S0.ip_s0, S0.ip_n0, S0.ip_kappa, S0.ip_lamdaP,
              S.node_rho, S.node_c, S.node_alpha,             // current unknowns
              phif, a0, s0, n0, kappa, lamdaP, lamdaE,
              S.node_x, S0.node_X, dphifdu, dphifdrho, dphifdc,
              Re_x, Kxx, Kxr, Kxc, Kxa,
              Re_rho, Krx, Krr, Krc, Kra,
              Re_c, Kcx, Kcr, Kcc, Kca,
              Re_alpha, Kax, Kar, Kac, Kaa);

    VectorXd R(24);
    R << Re_x, Re_rho, Re_c, Re_alpha;
    return R;
}

int main()
{
    std::vector<Vector4d> IP = LineQuadriIPTet();
    const int nip = (int)IP.size();
    const double dt = 0.2;

    std::vector<double> gp, lp;
    buildParameters(gp, lp);

    // Probe several states: a defect in the code shows up everywhere, while a
    // pathological probe state shows up only in the state that provokes it.
    const char* which = std::getenv("TANGENT_STATE");
    std::string w = which ? which : "wound";
    double phif0 = 0.47, kappa0 = 0.11; Vector3d lamdaP0(1.04, 1.02, 0.94);
    if(w == "healthy"){ phif0 = 1.0;  kappa0 = 0.024; lamdaP0 = Vector3d(1.0,1.0,1.0); }
    if(w == "isotropic"){ phif0 = 0.47; kappa0 = 1.0/3.0; lamdaP0 = Vector3d(1.0,1.0,1.0); }
    std::printf("  state = %s   phif=%g kappa=%g lamdaP=(%g,%g,%g)\n",
                w.c_str(), phif0, kappa0, lamdaP0(0), lamdaP0(1), lamdaP0(2));
    State S0 = baselineState(nip, phif0, kappa0, lamdaP0);
    std::vector<Matrix3d> ip_Jac = evalJacobian(S0.node_X);

    std::printf("\nFinite-difference check of the element tangent\n");
    std::printf("  tetrahedron, %d integration points, dt = %.3g\n", nip, dt);
    std::printf("  local substeps = %g, deadband = [%g, %g] smoothed over %g\n\n",
                lp[16], lp[18], lp[19], lp[20]);

    // Analytic tangent at the baseline state.
    std::vector<double>   phif = S0.ip_phif, kappa = S0.ip_kappa;
    std::vector<Vector3d> a0 = S0.ip_a0, s0 = S0.ip_s0, n0 = S0.ip_n0, lamdaP = S0.ip_lamdaP;
    std::vector<Vector3d> lamdaE(nip, Vector3d::Ones());
    std::vector<Matrix3d> strain(nip, Matrix3d::Zero()), stress(nip, Matrix3d::Zero());
    std::vector<Vector3d> dphifdu(nip, Vector3d::Zero());
    std::vector<double>   dphifdrho(nip,0.0), dphifdc(nip,0.0);
    VectorXd Re_x(12), Re_rho(4), Re_c(4), Re_alpha(4);
    MatrixXd Kxx(12,12), Kxr(12,4), Kxc(12,4), Kxa(12,4);
    MatrixXd Krx(4,12), Krr(4,4), Krc(4,4), Kra(4,4);
    MatrixXd Kcx(4,12), Kcr(4,4), Kcc(4,4), Kca(4,4);
    MatrixXd Kax(4,12), Kar(4,4), Kac(4,4), Kaa(4,4);
    evalWound(dt, 0.0, 1.0, ip_Jac, gp, lp, strain, stress,
              S0.node_rho, S0.node_c, S0.node_alpha,
              S0.ip_phif, S0.ip_a0, S0.ip_s0, S0.ip_n0, S0.ip_kappa, S0.ip_lamdaP,
              S0.node_rho, S0.node_c, S0.node_alpha,
              phif, a0, s0, n0, kappa, lamdaP, lamdaE,
              S0.node_x, S0.node_X, dphifdu, dphifdrho, dphifdc,
              Re_x, Kxx, Kxr, Kxc, Kxa,
              Re_rho, Krx, Krr, Krc, Kra,
              Re_c, Kcx, Kcr, Kcc, Kca,
              Re_alpha, Kax, Kar, Kac, Kaa);

    {
        VectorXd R0(24); R0 << Re_x, Re_rho, Re_c, Re_alpha;
        std::printf("  baseline residual norm = %.6e  (finite: %s)\n\n",
                    R0.norm(), R0.allFinite() ? "yes" : "NO - state is unusable");
    }

    // Assemble the 24x24 analytic tangent in the same [x|rho|c|alpha] order.
    MatrixXd K = MatrixXd::Zero(24,24);
    K.block(0,0,12,12)  = Kxx; K.block(0,12,12,4) = Kxr;
    K.block(0,16,12,4)  = Kxc; K.block(0,20,12,4) = Kxa;
    K.block(12,0,4,12)  = Krx; K.block(12,12,4,4) = Krr;
    K.block(12,16,4,4)  = Krc; K.block(12,20,4,4) = Kra;
    K.block(16,0,4,12)  = Kcx; K.block(16,12,4,4) = Kcr;
    K.block(16,16,4,4)  = Kcc; K.block(16,20,4,4) = Kca;
    K.block(20,0,4,12)  = Kax; K.block(20,12,4,4) = Kar;
    K.block(20,16,4,4)  = Kac; K.block(20,20,4,4) = Kaa;

    // Numerical tangent by central differences on every unknown.
    MatrixXd Kfd = MatrixXd::Zero(24,24);
    const double eps_x = 1e-6;    // [mm]
    const double eps_f = 1e-6;    // normalized field
    for(int j=0;j<24;j++){
        State Sp = S0, Sm = S0;
        double eps;
        if(j < 12){
            eps = eps_x;
            Sp.node_x[j/3](j%3) += eps;  Sm.node_x[j/3](j%3) -= eps;
        }else if(j < 16){
            eps = eps_f;
            Sp.node_rho[j-12]   += eps;  Sm.node_rho[j-12]   -= eps;
        }else if(j < 20){
            eps = eps_f;
            Sp.node_c[j-16]     += eps;  Sm.node_c[j-16]     -= eps;
        }else{
            eps = eps_f;
            Sp.node_alpha[j-20] += eps;  Sm.node_alpha[j-20] -= eps;
        }
        Kfd.col(j) = (residual(S0, Sp, ip_Jac, gp, lp, dt, nip)
                    - residual(S0, Sm, ip_Jac, gp, lp, dt, nip)) / (2.0*eps);
    }

    // Where exactly are the non-finite entries?
    {
        int n_nan = 0; int first_r=-1, first_c=-1;
        for(int r=0;r<24;r++) for(int c=0;c<24;c++)
            if(!std::isfinite(K(r,c))){ ++n_nan; if(first_r<0){first_r=r; first_c=c;} }
        if(n_nan){
            std::printf("  NON-FINITE analytic entries: %d of 576, first at (row %d, col %d)\n",
                        n_nan, first_r, first_c);
            std::printf("  rows with any non-finite: ");
            for(int r=0;r<24;r++){ bool b=false; for(int c=0;c<24;c++) if(!std::isfinite(K(r,c))) b=true;
                                   if(b) std::printf("%d ", r); }
            std::printf("\n  cols with any non-finite: ");
            for(int c=0;c<24;c++){ bool b=false; for(int r=0;r<24;r++) if(!std::isfinite(K(r,c))) b=true;
                                   if(b) std::printf("%d ", c); }
            std::printf("\n\n");
        }
    }

    // Report per block, so the failure names the coupling.
    struct Blk { const char* name; int r0,c0,nr,nc; };
    const Blk blocks[16] = {
        {"Ke_x_x",      0, 0,12,12},{"Ke_x_rho",     0,12,12,4},
        {"Ke_x_c",      0,16,12, 4},{"Ke_x_alpha",   0,20,12,4},
        {"Ke_rho_x",   12, 0, 4,12},{"Ke_rho_rho",  12,12, 4,4},
        {"Ke_rho_c",   12,16, 4, 4},{"Ke_rho_alpha",12,20, 4,4},
        {"Ke_c_x",     16, 0, 4,12},{"Ke_c_rho",    16,12, 4,4},
        {"Ke_c_c",     16,16, 4, 4},{"Ke_c_alpha",  16,20, 4,4},
        {"Ke_alpha_x", 20, 0, 4,12},{"Ke_alpha_rho",20,12, 4,4},
        {"Ke_alpha_c", 20,16, 4, 4},{"Ke_alpha_alpha",20,20,4,4},
    };

    std::printf("  %-16s %12s %12s %10s   %s\n",
                "block", "max |K|", "max |err|", "rel err", "verdict");
    int bad = 0;
    for(int b=0;b<16;b++){
        const Blk& B = blocks[b];
        MatrixXd Ka = K.block(B.r0,B.c0,B.nr,B.nc);
        MatrixXd Kn = Kfd.block(B.r0,B.c0,B.nr,B.nc);
        const double scale = Ka.cwiseAbs().maxCoeff();
        const double err   = (Ka-Kn).cwiseAbs().maxCoeff();
        // Scale by the block itself, falling back to the whole tangent so a
        // block that is legitimately near zero is not judged against noise.
        const double denom = std::max(scale, 1e-3*K.cwiseAbs().maxCoeff());
        const double rel   = (denom > 0.0) ? err/denom : err;
        const bool ok = rel < 1e-4;
        if(!ok) ++bad;
        std::printf("  %-16s %12.4e %12.4e %10.2e   %s\n",
                    B.name, scale, err, rel, ok ? "ok" : "MISMATCH");
    }

    std::printf("\n  %d of 16 blocks disagree with the numerical derivative\n", bad);
    if(bad){
        std::printf("  -> Newton cannot converge quadratically with these; the\n"
                    "     mismatching blocks name the terms to fix.\n");
    }
    std::printf("\n");
    // Reported, not enforced: this test exists to MEASURE the tangent, and the
    // known defects are expected to trip it until they are fixed. Flipping it
    // to a hard failure is the right move once the blocks come back clean.
    return 0;
}
