# ToDo — normalize parameters, fix mechanosensing, add pro-inflammatory species α

**Status: all 14 steps implemented.** Branch `normalize-and-alpha`, pushed to
`github.com/Aswanth0704/dura_multi_wound_healing` (private).

Companion to `plan.md` (model equations and calibrated parameter table) and
`CLAUDE.md` (architecture guidance).

---

## Completed

| Step | What | Verification |
|---|---|---|
| 0 | git init, allow-list `.gitignore`, private repo, branch `normalize-and-alpha` | 34 files / 904 KB tracked, zero data files |
| 1 | CMake repointed at the multiwound driver; Eigen/Boost located by probing; diagnostics gated behind `WOUND_VERBOSE`; env overrides added | clean build on macOS and Negishi |
| 2 | `D_rhorho` (4 copy-pasted sites) → one `evalDrho()` helper | VTK output **byte-identical** to the pre-refactor baseline |
| 3 | Areal-stretch mechanosensing `θ^e = ‖cof(F^e)·n0‖` | `tests/test_mechanosensing.cpp`: analytic dθ/dCC and dH/dCC match finite differences to ~1e-9 |
| 4 | Normalized parameter block; derived params from closed forms | matches `plan.md` median set; four source-term residuals ~1e-19 |
| 5 | Cylindrical prestretch preload (det F = 1 exactly) + settling phase | **homeostasis gate**: θ^e mean 1.13636, H mean 0.50090, ρ/c/φ within 1e-4 of 1.0 |
| 6 | Wound seeded after preload, free patch so it can contract | wound region populated; contraction visible |
| 7 | New `D_rho` with `C_low` gate (`k_cut = 300`), floor 6.12e-5 | `tests/test_diffusivity.cpp`: reproduces every `plan.md` reference value |
| 8 | `Psif` inconsistency fixed; dropped `dQ_rho/dφ` chain rule restored | settling went from hitting the 25-iteration cap to converging in 1–4 |
| 9–14 | α as a 4th monolithic field, 4×4 block system, VTK output | dof count grows by exactly one per unconstrained node; gate still holds with α in the system |
| — | Homeostasis derivation extracted to a shared header | `tests/test_homeostasis.cpp`: exact fixed point under 28 parameter perturbations |

`ctest` → **3/3 suites passing** on both macOS and Negishi.

### Decisions taken along the way
Normalized units · derived parameters from closed forms · prestretch via a preload
stage · `D_α = D_c = 0.00930` · multiwound driver only · `k_cut = 300` · `C_low`
**replaces** the old `phif - phif00` shift · free patch around the wound · git
branch before any edits.

---

## Seven defects found and fixed that were not in the original plan

These were blocking the healing phase and are worth knowing about.

1. **One-point tet quadrature was rank deficient.** `LineQuadriIPTet` returned a
   single centroid point where all four linear shape functions equal 1/4, so the
   element mass/reaction matrix `R_i R_j` is `(1/16)·ones(4,4)` — **rank 1**,
   leaving three near-null modes per element. Uniform fields never excite them,
   which is why settling always looked fine, but the wound's sharp gradient does:
   Newton asked for concentration increments of ~124 on a field whose
   physiological value is 1, while the residual stayed small. Replaced with the
   4-point Keast rule (degree 2, exact for the linear-tet mass matrix). **Peak
   increment fell from 124 to 0.74 — a factor of ~170.**

2. **Plastic-growth deadband excluded the physiological prestretch.**
   `lowlim/uplim` were hard-coded `0.95/1.05`, but healthy prestretched dura sits
   at `lamdaE = (1.098, 1.035, 0.880)` — both the axial and through-thickness
   stretches lie outside that band, so plastic growth fired continuously in
   perfectly healthy tissue and slowly ate the prestretch (θ^e drifted
   1.1365 → 1.1316 in 2 h). Now `local_parameters[18]/[19]`, default
   `[0.85, 1.15]`, with a guard that throws if the band excludes the prestretch.

3. **Rollback did not restore the geometry.** The reset path restored `node_rho`,
   `node_c` and every IP variable from their `_0` copies, but `node_x` has no
   `_0` counterpart and was left at the diverged value — so every retry after a
   failed step restarted from garbage and the adaptive time step could never
   recover. Now snapshotted per step and restored on reject.

4. **Badly scaled linear system.** The mechanics block reaches several hundred
   (kf = 40 MPa through the fiber tangent) against `1/dt ≈ 5` in transport.
   Unpreconditioned BiCGSTAB returned errors up to **1e+74 while reporting
   success**. Added symmetric diagonal equilibration, made a direct factorization
   primary, and added a NaN/Inf guard that names the offending dof and field.

5. **`writeTissue` wrote uninitialized memory.** Element connectivity was
   hard-coded to 8 nodes but the meshes are 4-node tets, so it read past the end
   — the dump differed between two runs of the same binary. Also `ip_lamdaP_0`
   dropped its third component.

6. **`myTissue.time` was never initialized** by the driver although both solvers
   read it, making `total_steps = (time_final − time)/time_step` undefined
   behaviour that happened to work.

7. **Unconverged Newton steps were accepted.** When the loop exhausted
   `max_iter` the solver printed a warning and moved on, keeping a state that was
   not a solution. Because the transport equations have no positivity limiter,
   that surfaced as **concentrations down to -1.66** during the puncture
   snap-open. Found by `scripts/verify_run.py`, not by watching the log. Such a
   step is now rejected and `dt` halved, which is what the adaptive time stepping
   existed for — and which only became usable once defect 3 made the rollback
   restore `node_x`.

Plus **damped-Newton step limiting**: puncturing prestretched dura collapses the
passive stress in the wound elements (`SSe_pas ∝ phif`, 1 → 0.01) so the hole
snaps open, and an undamped step overshot badly. The increment is scaled as a
whole, preserving the Newton direction.

And a **CMake trap worth remembering**: the `boost/` tree next to the project on
Negishi is a *partial* copy — it has `boost/algorithm/string.hpp` but is missing
`boost/preprocessor/list/detail/fold_left.hpp`, which that header pulls in. Since
the Eigen include directory is the project *parent*, `-I ..` put the broken tree
ahead of the real one; the first build only succeeded because the module had also
set `CPATH`. The probe now requires a deep header and is added with `BEFORE`.

---

## Build and run

```bash
mkdir build && cd build && cmake .. && make -j12
make tests && ctest          # 3/3
```

Negishi:
```bash
module load gcc/12.2.0 boost/1.80.0
cd build && cmake -DUSE_MKL=OFF .. && make -j12
mkdir ../run_case && cd ../run_case
cp ../files/woundcpp3D.sub ../files/<mesh>.mphtxt ../build/woundcpp3D .
sbatch woundcpp3D.sub
```

Runtime overrides: `WOUND_MESH`, `WOUND_TSETTLE`, `WOUND_TFINAL`, `WOUND_OUT`,
`WOUND_PATCH`, `WOUND_NOWOUND`, `WOUND_VERBOSE`.

Keep `--ntasks` at ~12: the element loop is parallel but the triplet assembly is
inside an `omp critical`, so scaling saturates (observed 350–530% of 1200%).

---

## Open items

- **`k_cut = 300` makes `C_low` a near-step** (0.0025 → 0.5 → 0.9975 across
  φ = 0 → 0.01 → 0.02, slope ~150 at φ = 0.01). It works, but if convergence
  degrades on finer meshes this is the first thing to soften.
- **Healthy-tissue `D_rho` rose ~90×**, from ~6.8e-7 to the 6.12e-5 floor,
  because `Ppoly(1) = 0` exactly. Intended per `plan.md` but it does change how
  far fibroblasts spread from the wound.
- **`n0` initialization on exact (non-cylindrical) dura geometry** has no general
  recipe yet. `n0` rotates correctly once set; only initialization is the gap.
  Recommend a Laplace problem at mesh-read time (inner surface 0, outer 1),
  `n0 = ∇u/‖∇u‖`. Avoid the eigenvalue-based alternative — non-smooth at
  crossings, which would damage the tangent.
- **A small residual φ drift is expected, and is discretization, not a bug.**
  The derived parameters make (0,1,1,1) an exact fixed point only where
  H = 1/2, i.e. where θ^e = ϑ^e = 1.136 exactly. Discretizing the curved shell
  gives θ^e ∈ [1.1254, 1.1462], so H ∈ [0.4736, 0.5256], and the fixed point is
  displaced in proportion to |H − 1/2|. The leading term is
  `dφ/dt ≈ p_phi_theta·(H−1/2)·ρ/(K_φρ+φ)`, which for |H−1/2| ~ 0.008 predicts
  ~6e-4 over 28 h — matching the observed local drift, and ~1e-2 over 4 weeks.
  Far-field drift is ~10x smaller. If this matters for a long run, refine the
  mesh through the thickness (it shrinks the θ^e spread) rather than retuning
  parameters.
- **`tau_lamdaP_*` are not co-scaled** with `K_phi_rho` (unlike
  `tau_omega`/`tau_kappa`), so plastic growth rate moves with any
  renormalization.
- **Four tangent-consistency defects remain** (listed in `CLAUDE.md`), notably
  the double-counted `dThetadrho(5..7)`/`dThetadc(5..7)` in `local_solver.cpp`.
  They affect convergence rate, not the residual.
- **`plan.md` line 55 sign**: written as `ρ̇ = ∇·Q_ρ + s_ρ` with `Q_ρ = −D_ρ∇ρ`,
  which is anti-diffusion. The code implements `ρ̇ + ∇·Q_ρ = s_ρ`, consistent
  with the c and α equations. Worth correcting in `plan.md`.
- **`src/cydindrical_dura_single_tissue_multiwound_original.cpp`** has now
  diverged from the driver. Safe to delete.
- **Second wound** is not re-enabled; the driver solves a single wound, per
  `plan.md`.
