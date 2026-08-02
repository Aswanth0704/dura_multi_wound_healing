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
// Switches to isolate which stress contribution carries a tangent error, by
// parameter rather than by editing the kernel.
static bool envOn(const char* k){ const char* e = std::getenv(k); return e && *e && *e != '0'; }

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

    const double k0 = envOn("TANGENT_NOVOL") ? 0.0 : 0.02;
    const double kf = envOn("TANGENT_NOFIB") ? 0.0 : 40.0;
    const double k2 = 0.048;
    const double t_rho = envOn("TANGENT_NOACT") ? 0.0 : 1.28571e-3;
    const double t_rho_c = t_rho*3.28571;
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

    // NOSTRUCT freezes the structural update, so dphif/da0/dkappa/dlamdaP with
    // respect to CC all vanish and DDstruct drops out of Ke_x_x.
    // NOSTRUCT freezes the structural response so DDstruct drops out of Ke_x_x:
    // it kills the mechano-driven collagen production (the only route by which
    // phif depends on CC) and stretches every structural timescale to 1e6 h so
    // a0, kappa and lamdaP cannot move within a step. p_phi itself is NOT
    // zeroed - K_phi_rho is derived from it and would blow up, which is what
    // made the first attempt at this probe return NaN.
    const bool nostruct = envOn("TANGENT_NOSTRUCT");
    // Individual freezes, to say WHICH structural variable carries the error.
    const bool no_phi    = nostruct || envOn("TANGENT_NOPHI");
    const bool no_a0     = nostruct || envOn("TANGENT_NOA0");
    const bool no_kappa  = nostruct || envOn("TANGENT_NOKAPPA");
    const bool no_lamdaP = nostruct || envOn("TANGENT_NOLAMDAP");
    const double p_phi = 9.34e-4, p_phi_c = 1.41e-3;
    const double p_phi_theta = no_phi ? 0.0 : 4.96*p_phi;
    const double K_phi_c = 1.08, d_phi = 2.02e-3, d_phi_rho_c = 2.87e-4;
    const double K_phi_rho = deriveKphirho(p_phi, p_phi_c, p_phi_theta, K_phi_c,
                                           d_phi, d_phi_rho_c, c_h, rho_h, phi_h, H_h);
    const double tau_omega = no_a0    ? 1e6 : 10.0/(K_phi_rho+1.0);
    const double tau_kappa = no_kappa  ? 1e6 :  1.0/(K_phi_rho+1.0);
    const double gamma_kappa = 5.0;
    lp = { p_phi, p_phi_c, p_phi_theta, K_phi_c, K_phi_rho, d_phi, d_phi_rho_c,
           tau_omega, tau_kappa, gamma_kappa,
           no_lamdaP?1e6:0.05, no_lamdaP?1e6:0.05, no_lamdaP?1e6:0.05,  // tau_lamdaP_a/s/n
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


// ---------------------------------------------------------------------------
// PROBE: SSe_pas vs CCe, one level below the element tangent.
//
// The element test says Ke_x_x is 5.5e-4 wrong and every sweep says the error
// is purely analytic. Ke_x_x contains the passive fiber tangent DDe pulled back
// through Fg, so this checks DDe on its own, BEFORE the pull-back - if DDe is
// right here, the error is in the pull-back or elsewhere; if it is wrong here,
// the algebra is wrong and the pull-back is innocent.
//
// Uses DIRECTIONAL derivatives along a symmetric direction M rather than
// perturbing individual components. Perturbing CCe(c,d) alone breaks the
// symmetry of CCe, and symmetrising the perturbation reintroduces the
// factor-of-2 convention trap that already bit the mechanosensing test in this
// project. A directional derivative has no convention to get wrong:
//     d/dh SSe(CCe + h M) |_0   ==   DDe : M
// ---------------------------------------------------------------------------
static void SSe_pas_of(const Matrix3d& CCe, const Vector3d& a0,
                       double kappa, double phif, double k0, double kf, double k2,
                       Matrix3d& SSe)
{
    const Matrix3d a0a0 = a0*a0.transpose();
    const double I1e = CCe.trace();
    const double I4e = a0.dot(CCe*a0);
    const double E   = kappa*I1e + (1-3*kappa)*I4e - 1.0;
    const double Psif  = (kf/(2.*k2))*std::exp(k2*E*E);
    const double Psif1 = 2*k2*kappa*E*Psif;
    const double Psif4 = 2*k2*(1-3*kappa)*E*Psif;
    SSe = phif*(k0*Matrix3d::Identity() + Psif1*Matrix3d::Identity() + Psif4*a0a0);
}

static int probeDDe()
{
    // Wound-state values, matching the element test.
    const double kappa = 0.11, phif = 0.47;
    const double k0 = 0.02, kf = 40.0, k2 = 0.048;
    const Vector3d a0(0,0,1);
    const Matrix3d a0a0 = a0*a0.transpose();

    // A representative elastic right Cauchy-Green tensor: prestretched, sheared.
    Matrix3d CCe;
    CCe << 1.0298, 0.0121, 0.0043,
           0.0121, 0.8771, 0.0072,
           0.0043, 0.0072, 1.1153;

    const double I1e = CCe.trace();
    const double I4e = a0.dot(CCe*a0);
    const double E   = kappa*I1e + (1-3*kappa)*I4e - 1.0;
    const double Psif  = (kf/(2.*k2))*std::exp(k2*E*E);
    const double Psif1 = 2*k2*kappa*E*Psif;
    const double Psif4 = 2*k2*(1-3*kappa)*E*Psif;

    // The four coefficients exactly as src/wound.cpp writes them.
    const double Psif11 = 2*k2*kappa*kappa*Psif + 2*k2*kappa*E*Psif1;
    const double Psif14 = 2*k2*kappa*(1-3*kappa)*Psif + 2*k2*kappa*E*Psif4;
    const double Psif41 = 2*k2*(1-3*kappa)*kappa*Psif + 2*k2*(1-3*kappa)*E*Psif1;
    const double Psif44 = 2*k2*(1-3*kappa)*(1-3*kappa)*Psif + 2*k2*(1-3*kappa)*E*Psif4;

    std::printf("\n[probe] passive fiber tangent DDe, before the Fg pull-back\n");
    std::printf("  I1e=%.5f I4e=%.5f E=%.5f\n", I1e, I4e, E);
    std::printf("  Psif14=%.6e  Psif41=%.6e   (must be equal)  rel diff %.2e\n",
                Psif14, Psif41, std::fabs(Psif14-Psif41)/std::fabs(Psif41));

    // Three independent symmetric directions, including a pure shear.
    Matrix3d dirs[3];
    dirs[0] = Matrix3d::Identity();
    dirs[1] = a0a0;
    dirs[2] << 0, 1, 0,
               1, 0, 0,
               0, 0, 0;

    int bad = 0;
    for(int d=0; d<3; d++){
        const Matrix3d& M = dirs[d];
        // analytic: DDe : M
        const double trM   = M.trace();
        const double a0Ma0 = a0.dot(M*a0);
        Matrix3d ana = phif*( (Psif11*trM + Psif14*a0Ma0)*Matrix3d::Identity()
                            + (Psif41*trM + Psif44*a0Ma0)*a0a0 );
        // numerical: central difference of SSe_pas along M
        const double h = 1e-6;
        Matrix3d Sp, Sm;
        SSe_pas_of(CCe + h*M, a0, kappa, phif, k0, kf, k2, Sp);
        SSe_pas_of(CCe - h*M, a0, kappa, phif, k0, kf, k2, Sm);
        Matrix3d num = (Sp - Sm)/(2.0*h);

        const double err = (ana-num).cwiseAbs().maxCoeff();
        const double rel = err/std::max(1e-30, ana.cwiseAbs().maxCoeff());
        const bool ok = rel < 1e-6;
        if(!ok) ++bad;
        std::printf("  direction %d: max|ana| %.4e  max|err| %.4e  rel %.2e  %s\n",
                    d, ana.cwiseAbs().maxCoeff(), err, rel, ok ? "ok" : "MISMATCH");
    }
    std::printf("  -> DDe itself is %s\n\n", bad ? "WRONG" : "correct");
    return bad;
}


// ---------------------------------------------------------------------------
// PROBE 2: the Fg pull-back, dSS_pas/dCC.
//
// DDe (probe 1) is correct, so if the element-level Ke_x_x error is in the
// passive path it must live in the pull-back
//     SS_pas = Jp * Fginv * SSe_pas * Fginv,     CCe = Fginv * CC * Fginv
// This replicates src/wound.cpp's dSSpasdCC_explicit expression EXACTLY and
// finite-differences SS_pas along symmetric directions in CC, holding the
// structure (phif, a0, kappa, lamdaP) fixed - which is what "explicit" means
// there.
// ---------------------------------------------------------------------------
static int probePullback()
{
    const double kappa = 0.11, phif = 0.47;
    const double k0 = 0.02, kf = 40.0, k2 = 0.048;
    const Vector3d a0(0,0,1), s0(1,0,0), n0(0,1,0);
    const Vector3d lamdaP(1.04, 1.02, 0.94);   // along (a0, s0, n0)

    Matrix3d Fg = lamdaP(0)*a0*a0.transpose()
                + lamdaP(1)*s0*s0.transpose()
                + lamdaP(2)*n0*n0.transpose();
    const Matrix3d FFginv = Fg.inverse();
    const double Jp = Fg.determinant();

    Matrix3d CC;
    CC << 1.0714, 0.0128, 0.0047,
          0.0128, 0.9126, 0.0079,
          0.0047, 0.0079, 1.2062;

    auto SS_pas_of = [&](const Matrix3d& CCin){
        Matrix3d CCe = FFginv*CCin*FFginv;
        Matrix3d SSe; SSe_pas_of(CCe, a0, kappa, phif, k0, kf, k2, SSe);
        return Matrix3d(Jp*FFginv*SSe*FFginv);
    };

    // Rebuild the coefficients at the base point, as wound.cpp does.
    Matrix3d CCe0 = FFginv*CC*FFginv;
    const Matrix3d a0a0 = a0*a0.transpose();
    const double I1e = CCe0.trace(), I4e = a0.dot(CCe0*a0);
    const double E = kappa*I1e + (1-3*kappa)*I4e - 1.0;
    const double Psif  = (kf/(2.*k2))*std::exp(k2*E*E);
    const double Psif1 = 2*k2*kappa*E*Psif;
    const double Psif4 = 2*k2*(1-3*kappa)*E*Psif;
    const double Psif11 = 2*k2*kappa*kappa*Psif + 2*k2*kappa*E*Psif1;
    const double Psif14 = 2*k2*kappa*(1-3*kappa)*Psif + 2*k2*kappa*E*Psif4;
    const double Psif41 = 2*k2*(1-3*kappa)*kappa*Psif + 2*k2*(1-3*kappa)*E*Psif1;
    const double Psif44 = 2*k2*(1-3*kappa)*(1-3*kappa)*Psif + 2*k2*(1-3*kappa)*E*Psif4;

    // dSSpasdCC exactly as coded in src/wound.cpp.
    std::vector<double> dSS(81, 0.0);
    const Matrix3d Id = Matrix3d::Identity();
    for(int ii=0;ii<3;ii++)for(int jj=0;jj<3;jj++)for(int kk=0;kk<3;kk++)for(int ll=0;ll<3;ll++)
      for(int pp=0;pp<3;pp++)for(int rr=0;rr<3;rr++)for(int ss=0;ss<3;ss++)for(int tt=0;tt<3;tt++)
        dSS[ii*27+jj*9+kk*3+ll] += Jp*(phif*(Psif11*Id(pp,rr)*Id(ss,tt)
                                   + Psif14*Id(pp,rr)*a0a0(ss,tt)
                                   + Psif41*a0a0(pp,rr)*Id(ss,tt)
                                   + Psif44*a0a0(pp,rr)*a0a0(ss,tt)))
                                   *FFginv(ii,pp)*FFginv(jj,rr)*FFginv(kk,ss)*FFginv(ll,tt);

    std::printf("[probe] Fg pull-back, dSS_pas/dCC\n");
    Matrix3d dirs[4];
    dirs[0] = Id;
    dirs[1] = a0*a0.transpose();
    dirs[2] << 0,0,1, 0,0,0, 1,0,0;      // shear in the a0-s0 plane
    dirs[3] << 0,0,0, 0,0,1, 0,1,0;      // shear in the a0-n0 plane

    int bad = 0;
    for(int d=0; d<4; d++){
        const Matrix3d& M = dirs[d];
        Matrix3d ana = Matrix3d::Zero();
        for(int ii=0;ii<3;ii++)for(int jj=0;jj<3;jj++)for(int kk=0;kk<3;kk++)for(int ll=0;ll<3;ll++)
            ana(ii,jj) += dSS[ii*27+jj*9+kk*3+ll]*M(kk,ll);
        const double h = 1e-6;
        Matrix3d num = (SS_pas_of(CC + h*M) - SS_pas_of(CC - h*M))/(2.0*h);
        const double err = (ana-num).cwiseAbs().maxCoeff();
        const double scl = std::max(ana.cwiseAbs().maxCoeff(), num.cwiseAbs().maxCoeff());
        const double rel = (scl>0) ? err/scl : err;
        const bool ok = rel < 1e-6;
        if(!ok) ++bad;
        std::printf("  direction %d: max|ana| %.4e  max|num| %.4e  rel %.2e  %s\n",
                    d, ana.cwiseAbs().maxCoeff(), num.cwiseAbs().maxCoeff(), rel,
                    ok ? "ok" : "MISMATCH");
    }
    std::printf("  -> the pull-back expression is %s\n\n", bad ? "WRONG" : "correct");
    return bad;
}


// ---------------------------------------------------------------------------
// PROBE 3: the eigenvector derivative in local_solver.cpp.
//
// Freezing the structural response makes Ke_x_x exact, and within that the
// error tracks the fiber DIRECTION and DISPERSION - both of which are driven by
// the principal eigenpair of CCe. local_solver.cpp gets dv/dCCe from a bordered
// linear system
//     [ CCe - lamda I   -v ] [ dv     ]   [ -(dCCe) v ]
//     [ v^T              0 ] [ dlamda ] = [     0     ]
// solved once per elementary perturbation E_ij. This checks that against a
// finite difference of the actual eigen-decomposition.
//
// Eigenvector SIGN is the trap here: SelfAdjointEigenSolver picks a sign
// arbitrarily and it can flip between the + and - evaluations, which would show
// up as a huge spurious error. Both perturbed eigenvectors are therefore
// re-signed to agree with the base one before differencing.
// ---------------------------------------------------------------------------
static void topEigen(const Matrix3d& A, const Vector3d& ref, double& lam, Vector3d& v)
{
    SelfAdjointEigenSolver<Matrix3d> es; es.compute(A);
    lam = es.eigenvalues()(2);
    v   = es.eigenvectors().col(2);
    if(ref.dot(v) < 0) v = -v;      // fix the arbitrary sign against a reference
}

static int probeEigenDeriv()
{
    Matrix3d CCe;
    CCe << 1.0298, 0.0121, 0.0043,
           0.0121, 0.8771, 0.0072,
           0.0043, 0.0072, 1.1153;

    SelfAdjointEigenSolver<Matrix3d> es; es.compute(CCe);
    const Vector3d lam3 = es.eigenvalues();
    Vector3d v = es.eigenvectors().col(2);
    const Vector3d a0(0,0,1);
    if(a0.dot(v) < 0) v = -v;
    const double lamdamax = lam3(2);

    std::printf("[probe] eigenvector derivative (bordered system)\n");
    std::printf("  eigenvalues %.6f %.6f %.6f   gaps %.4e %.4e\n",
                lam3(0), lam3(1), lam3(2), lam3(2)-lam3(1), lam3(1)-lam3(0));

    // Replicate the bordered solve exactly as local_solver.cpp does it.
    Matrix4d LHS; Vector4d RHS, SOL;
    LHS << CCe(0,0)-lamdamax, CCe(0,1), CCe(0,2), -v(0),
           CCe(1,0), CCe(1,1)-lamdamax, CCe(1,2), -v(1),
           CCe(2,0), CCe(2,1), CCe(2,2)-lamdamax, -v(2),
           v(0), v(1), v(2), 0;
    std::vector<Matrix3d> dvdCCe(3, Matrix3d::Zero());
    Matrix3d dlamdCCe = Matrix3d::Zero();
    for(int ii=0;ii<3;ii++) for(int jj=0;jj<3;jj++){
        RHS.setZero(); RHS(ii) = -v(jj);
        SOL = LHS.lu().solve(RHS);
        dvdCCe[0](ii,jj) = SOL(0);
        dvdCCe[1](ii,jj) = SOL(1);
        dvdCCe[2](ii,jj) = SOL(2);
        dlamdCCe(ii,jj)  = SOL(3);
    }

    // Symmetric probe directions.
    Matrix3d dirs[4];
    dirs[0] = Matrix3d::Identity();
    dirs[1] << 1,0,0, 0,0,0, 0,0,0;
    dirs[2] << 0,1,0, 1,0,0, 0,0,0;      // pure shear
    dirs[3] << 0,0,0, 0,0,1, 0,1,0;      // pure shear

    int bad = 0;
    for(int d=0; d<4; d++){
        const Matrix3d& M = dirs[d];
        // analytic, contracted over ALL nine components as the caller does
        Vector3d ana = Vector3d::Zero();
        double anaLam = 0.0;
        for(int ii=0;ii<3;ii++) for(int jj=0;jj<3;jj++){
            for(int m=0;m<3;m++) ana(m) += dvdCCe[m](ii,jj)*M(ii,jj);
            anaLam += dlamdCCe(ii,jj)*M(ii,jj);
        }
        const double h = 1e-6;
        double lp, lm; Vector3d vp, vm;
        topEigen(CCe + h*M, v, lp, vp);
        topEigen(CCe - h*M, v, lm, vm);
        const Vector3d num = (vp - vm)/(2.0*h);
        const double numLam = (lp - lm)/(2.0*h);

        const double errV = (ana-num).cwiseAbs().maxCoeff();
        const double sclV = std::max(ana.cwiseAbs().maxCoeff(), num.cwiseAbs().maxCoeff());
        const double relV = (sclV>1e-12) ? errV/sclV : errV;
        const double relL = std::fabs(anaLam-numLam)/std::max(1e-12, std::fabs(numLam));
        const bool ok = (relV < 1e-5) && (relL < 1e-5);
        if(!ok) ++bad;
        std::printf("  dir %d: dv rel %.2e (|ana| %.3e |num| %.3e)   dlamda rel %.2e  %s\n",
                    d, relV, ana.norm(), num.norm(), relL, ok ? "ok" : "MISMATCH");
    }
    std::printf("  -> the eigen-derivative is %s\n\n", bad ? "WRONG" : "correct");
    return bad;
}

int main()
{
    probeDDe();
    probePullback();
    probeEigenDeriv();

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

    // Where in Ke_x_x does the error live, and is it symmetric? Ke_x_x is not
    // required to be symmetric overall (the active traction is not
    // hyperelastic), but the passive part comes from a stored energy, so a
    // clearly ANTI-symmetric error pattern points at a transposed index pair.
    {
        MatrixXd E = K.block(0,0,12,12) - Kfd.block(0,0,12,12);
        const double sym  = 0.5*(E + E.transpose()).cwiseAbs().maxCoeff();
        const double anti = 0.5*(E - E.transpose()).cwiseAbs().maxCoeff();
        std::printf("\n  Ke_x_x error: symmetric part %.3e, antisymmetric part %.3e\n",
                    sym, anti);
        // top offenders
        for(int rank=0; rank<4; rank++){
            int br=-1, bc=-1; double best=-1;
            for(int r=0;r<12;r++) for(int c=0;c<12;c++){
                double v = std::fabs(E(r,c));
                bool taken=false;
                for(int q=0;q<rank;q++) ; // simple: just report the top few by scan
                if(v>best){ best=v; br=r; bc=c; }
            }
            if(br<0) break;
            std::printf("    largest |err| %.3e at (node %d dir %d, node %d dir %d)  K=%.4e  Kfd=%.4e\n",
                        best, br/3, br%3, bc/3, bc%3, K(br,bc), Kfd(br,bc));
            E(br,bc) = 0.0;
        }
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
