# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 3D finite-element multiphysics solver for wound healing in spinal **dura mater** ("DaLaWoHe" — Dalai Lama Wound Healing). It couples large-deformation mechanics with reaction–diffusion transport of four species, plus evolving microstructure (plastic growth, collagen fiber orientation and dispersion).

**All fields are normalized**: homeostasis is `(alpha, rho, c, phi) = (0, 1, 1, 1)`.

| Field | Meaning | dof |
|---|---|---|
| `x` | deformed nodal position (mechanics) | 3/node |
| `rho` | fibroblast density | 1/node |
| `c` | cytokine / TGF-β1 | 1/node |
| `alpha` | pro-inflammatory signal | 1/node |
| `phif`, `a0/s0/n0`, `kappa`, `lamdaP` | collagen fraction, fiber frame, dispersion, plastic stretch | **integration-point**, updated by a local solver |

`plan.md` is the authoritative model spec: the governing equations and the calibrated parameter set (sweep #4 case 0 medians). `ToDo.md` tracks implementation work.

C++17 with Eigen (dense + sparse), Boost (header-only string algorithms, used by the mesh readers), and OpenMP.

## Build and run

```bash
mkdir build && cd build
cmake ..            # probes for Eigen and Boost; -DUSE_MKL=OFF to skip MKL
make -j12
make tests && ctest # 4 suites, all should pass
```

`cmake` locates Eigen and Boost by probing, so no paths need editing. Two things to know:

- **The `boost/` directory next to the project on Negishi is an incomplete copy** — it has `boost/algorithm/string.hpp` but is missing `boost/preprocessor/list/detail/fold_left.hpp`, which that header pulls in. The probe therefore requires a deep header before accepting a candidate and prefers the system module. It is also added with `BEFORE`, because the Eigen include directory is the project *parent*, which contains that broken tree.
- `MAIN_SRC` selects the driver: `cmake -DMAIN_SRC=src/<other_main>.cpp ..`. Default is `src/cydindrical_dura_single_tissue_multiwound.cpp`. Never add `src/file_io_gmsh_double.cpp` to the target — it redefines two symbols from `src/file_io_gmsh.cpp`.

### Negishi

```bash
module load gcc/12.2.0 boost/1.80.0
cd build && cmake -DUSE_MKL=OFF .. && make -j12
# then, per case:
mkdir run_<case> && cd run_<case>
cp ../files/woundcpp3D.sub ../files/<mesh>.mphtxt ../build/woundcpp3D .
sbatch woundcpp3D.sub
```

The binary reads **bare relative** mesh paths and writes all output to the cwd, so every case needs its own run directory. `files/woundcpp3D.sub` is the job template.

**Do not raise `--ntasks` much past 12.** The element loop is `#pragma omp parallel for` (`solver.cpp`) but contains two `omp critical` sections — one for the IP store-back and one wrapping the *entire* triplet assembly — so the speedup saturates and extra ranks only contend. Fixing that means per-thread triplet buffers merged after the loop.

### Runtime overrides (no rebuild)

| Variable | Effect |
|---|---|
| `WOUND_MESH` | mesh file name |
| `WOUND_TSETTLE` | phase-1 prestretch settling duration [h] |
| `WOUND_TFINAL` | phase-2 healing duration [h] |
| `WOUND_OUT` | output prefix |
| `WOUND_PATCH` | free-patch radius as a multiple of `r_wound` |
| `WOUND_NOWOUND` | stop after the homeostasis gate |
| `WOUND_VERBOSE` | full node/element/jacobian/dof dumps (gigabytes on a fine mesh) |

Model and wound-seeding knobs:

| Variable | Effect | Default |
|---|---|---|
| `WOUND_TRHO` | active traction `t_rho` — the sweep used ×1 … ×1000; `t_rho_c` follows at a fixed 3.28571 ratio | 1.28571e-3 MPa |
| `WOUND_WSMOOTH` | width of the `tanh` wound seed profile | `1.7 * mean_edge_local` (mesh-dependent) |
| `WOUND_KCUT` | steepness of the low-collagen gate `C_low` | 300 |
| `WOUND_ALPHA_D` / `WOUND_ALPHA_DECAY` / `WOUND_ALPHA_PC` | `D_alpha`, `d_alpha`, `p_c_alpha` | per `plan.md` |
| `WOUND_NOPRESTRETCH` | skip phase 1 entirely | off |
| `WOUND_FREEZEX` | fix every displacement dof (transport-only experiment) | off |
| `WOUND_SEVRAMP` / `WOUND_SEVSTEPS` | continuation: seed the wound in severity increments | off |

Solver and smoothness knobs (all default to the production behaviour):

| Variable | Effect | Default |
|---|---|---|
| `WOUND_SIGNEPS` | width of the smooth eigenvector sign; **0 restores the old hard flip and reproduces the limit cycle** | 0.05 |
| `WOUND_LAMP_LO` / `WOUND_LAMP_HI` | plastic-growth bounds | 0.5 / 2.0 |
| `WOUND_BANDCAP` | saturation cap on the deadband term | 1.0 |
| `WOUND_BANDW` | smoothing width of the deadband threshold; **0 restores the original hard threshold** | 0.002 |
| `WOUND_TAULAMP` | scales `tau_lamdaP_*` | 1 |
| `WOUND_LOCALSUB` | local forward-Euler substeps (`time_step_ratio`) | 25 |
| `WOUND_TOL` / `WOUND_TOL_INC` / `WOUND_MAXITER` | Newton tolerances and iteration cap | — |
| `WOUND_DTRAMP` / `WOUND_RUNGS` | the dt ladder that absorbs the puncture transient | — |
| `WOUND_LINESEARCH` | decoupled damping branch | off |
| `WOUND_RESLOC` | iteration at which to print the 20 largest-residual dofs | 20 |
| `WOUND_FDEPS` | finite-difference step for the tangent check | — |

Use `files/dura_cyl_repeated_wound_v62_2t_finer.mphtxt` (≈5k nodes) for verification and `..._20t_finer` for production.

## Architecture

### Layers

| Layer | Files | Role |
|---|---|---|
| Case driver (`main`) | `src/*.cpp` with `int main` | geometry, parameters, prestretch, wound seeding, BCs; calls the solver |
| Global solver | `src/solver.cpp`, `include/solver.h` | `tissue` struct, dof maps, element jacobians, sparse Newton time loop |
| Element residual/tangent | `src/wound.cpp`, `include/wound.h` | `evalWound` — assembles 4 residuals and the 4×4 block tangent per element |
| Local (IP) solver | `src/local_solver.cpp` | `localWoundProblemExplicit` — evolves `phif, a0, s0, n0, kappa, lamdaP` and returns `dTheta/d{CC,rho,c}` |
| Shape functions | `src/element_functions.cpp` | hex (8/20/27) and tet (4/10) shape functions, jacobians, quadrature |
| Mesh I/O | `src/file_io.cpp`, `src/file_io_gmsh*.cpp`, `meshing/myMeshGenerator.cpp` | COMSOL `.mphtxt`, Gmsh `.msh`, Abaqus `.inp` readers → `HexMesh`; VTK writers |
| Shared physics headers | `include/mechanosensing.h`, `include/diffusivity.h`, `include/homeostasis.h` | single definitions shared by the solver and the tests |

Element type is inferred from `elements[0].size()` (8/20/27 → hex, 4/10 → tet), which selects both quadrature and shape functions. A new element type must be added consistently in `main`, `sparseWoundSolver` and `evalWound`.

### The two-phase driver

The driver runs **prestretch settling** then **healing**, because living dura is taut: `plan.md`'s Consolini measurement says excised dura shrinks by 1.098 axially and 1.035 circumferentially. So the mesh is the *unloaded* geometry and the in-vivo state is that mesh stretched.

1. **Phase 1 — settling.** `prestretch_target()` applies three distinct cylindrical stretches (axial 1.098, circumferential 1.035 on the mid-surface radius, through-thickness `1/(lam_z·lam_th)`), giving `det F = 1` exactly and `theta_e = 1.136`. That is prescribed as `eBC_x` on every boundary node and settled with no wound. `node_X` stays the unloaded reference and is **never re-referenced**.
2. **Phase 2 — healing.** The prestretch is held except inside a free patch (default 4×`r_wound`) around the puncture, so the far field keeps `theta_e = 1.136 / H = 1/2` while the wound can contract. `eBC_x` is rebuilt and `fillDOFmap` re-run. The wound is then seeded into the already-deformed tissue, testing membership on *deformed* coordinates since the puncture is made in vivo.

`reportState()` recomputes `theta_e`, `H` and `det(F^e)` at every IP straight from the tissue state — deliberately independent of the solver, so it acts as a check on it — and reports the range of every field.

### Homeostasis is the gate

Three parameters are **not free**; each is pinned by requiring a source term to vanish at homeostasis (`include/homeostasis.h`):

| Derived | From | Value |
|---|---|---|
| `K_rho_rho` | `s_rho = 0` | 1.1646 |
| `K_phi_rho` | `phi_dot = 0` | 0.7027 |
| `p_c_rho` | `s_c = 0` | 0.006941 |

They are computed from closed forms rather than hard-coded, so changing a sampled parameter keeps the fixed point exact. The driver prints all four source-term residuals at startup (~1e-19) and throws if `K_phi_rho <= 0` or `K_rho_rho <= rho_h` — `plan.md` records a GP fit that returned `K_phi_rho = -0.205` from this same constraint.

**If a no-wound run drifts from (0,1,1,1), stop and fix that first** — nothing downstream is interpretable otherwise.

### Mechanosensing

`H(theta_e) = 1/(1+exp(-gamma_e(theta_e - vartheta_e)))` with `vartheta_e = 1.136`, `gamma_e = 10`, so `H = 1/2` exactly in the healthy prestretched state.

The stimulus is the **in-plane areal stretch** of the dural mid-surface, `theta_e = ||cof(F^e)·n0||` — *not* `det(F^e)`. This matters: the tissue is treated as incompressible, so `det(F^e) = 1` identically and could never respond to membrane stretch. Since `Fg·n0 = lamdaP_N·n0`, the closed form collapses to

```
theta_e      = J·sqrt(n0ᵀCC⁻¹n0) / (lamdaP_a·lamdaP_s)
dtheta_e/dCC = (theta_e/2)[CC⁻¹ − (CC⁻¹n0)(CC⁻¹n0)ᵀ / (n0ᵀCC⁻¹n0)]
dH/dCC       = gamma_e·H(1−H)·dtheta_e/dCC
```

`n0` is the through-thickness normal (`build_cylinder_frame` sets `n0` radial, `a0` axial, `s0` circumferential). `Je` is still used for the volumetric stress, which is a different quantity — don't conflate them.

**For non-cylindrical (exact) dura geometry this needs work**: `n0` is a state variable that rotates correctly, but *initialization* has no general recipe. The robust approach is to solve a Laplace problem once at mesh-read time (inner surface 0, outer 1) and take `n0 = ∇u/‖∇u‖`.

### Parameters are positional `std::vector<double>`

The single biggest hazard when editing. Both vectors are built by index order in `main` and unpacked by **literal index** in the kernels:

- `global_parameters` (28) — `{k0, kf, k2, t_rho, t_rho_c, K_t, K_t_c, D_rhorho(unused), D_rhoc, D_cc, p_rho, p_rho_c, p_rho_theta, K_rho_c, K_rho_rho, d_rho, vartheta_e, gamma_theta, p_c_rho, p_c_thetaE, K_c_c, d_c, bx, by, bz, D_alpha, d_alpha, p_c_alpha}`
- `local_parameters` (20) — `{p_phi, p_phi_c, p_phi_theta, K_phi_c, K_phi_rho, d_phi, d_phi_rho_c, tau_omega, tau_kappa, gamma_kappa, tau_lamdaP_a, tau_lamdaP_s, tau_lamdaP_n, vartheta_e, gamma_theta, tol_local, time_step_ratio, max_iter, lamdaE_lo, lamdaE_hi}`

`wound.cpp` unpacks `global_parameters[0..24]` at **six separate sites**, so **always append**; inserting mid-vector silently corrupts every downstream read. `global_parameters[7]` is dead — fibroblast diffusivity comes from `evalDrho()` in `include/diffusivity.h`.

### Time integration and linear solve

Monolithic Newton on all four field blocks. The residual is `((u − u_0)/dt − S)·R − Grad_R·Q`, i.e. backward Euler in **rate** form — which is why the tangent carries `+R_i R_j/dt`, not `+R_i R_j`.

The linear solve does three things that are load-bearing:

- **Diagonal equilibration.** The mechanics block reaches several hundred (kf = 40 MPa through the fiber tangent) against `1/dt ≈ 5` in transport. Unpreconditioned BiCGSTAB returned errors up to 1e+74 *while reporting success*. It now solves `(S K S)y = S b` with `S = diag(1/sqrt|K_ii|)`.
- **Direct factorization primary**, equilibrated iterative as fallback. The direct path does not silently return a garbage increment.
- **Damped Newton.** Puncturing prestretched dura collapses the passive stress in the wound elements (`SSe_pas ∝ phif`, 1 → 0.01) so the hole snaps open; an undamped step overshot badly. The increment is scaled as a whole, preserving the Newton direction.

- **Unconverged steps are rejected, not accepted.** A step that exhausts
  `max_iter` with `residuum > tol` sets `reset`, so it is thrown away and `dt` is
  halved. The old code merely printed "Check, make sure residual is small enough"
  and kept going — which is what produced concentrations as low as **-1.66**
  during the puncture snap-open, since the transport equations have no positivity
  limiter and an unconverged increment surfaces directly as negative species.

Plus a NaN/Inf guard on the assembled system that names the offending dof and field, and a per-step negative-species report. On divergence the step is rejected and `time_step` halved; the rollback restores `node_x` from a per-step snapshot as well as the `_0` fields. There is deliberately **no clamping** of concentrations — clamping would hide a convergence failure rather than fix one.

**Tet quadrature is 4-point (Keast), not 1-point.** One centroid point makes all four linear shape functions equal 1/4, so the element mass/reaction matrix is rank 1, leaving three near-null modes per element. Uniform fields never excite them — which is why settling always looked fine — but a wound gradient does: Newton asked for concentration increments of ~124 while the residual stayed small. The 4-point rule is degree 2, exact for the linear-tet mass matrix.

**`LineQuadriIPTetQuadratic` is still broken — do not use tet10.** It returns the same rule as the linear case, which cannot integrate quadratic shape functions, leaving six spurious modes per element. The quadratic tet needs a degree-4 rule (11- or 15-point Keast). The element will run and produce quietly wrong answers, exactly as the 1-point linear rule did.

**Eigenvector sign convention must stay smooth.** Fiber reorientation uses `v_max` from `C^e v = lambda v`, which is defined only up to sign. A hard `sign(a0.dot(v_max))` flip makes the residual discontinuous — the jump is `2k(I - a0 (x) a0) v_max`, of magnitude `2k|sin(theta)|`, which is *largest exactly at the flip* (theta = pi/2). That put Newton in a period-2 limit cycle and stalled every healing run. It is now `tanh(s/eps)` with `eps = WOUND_SIGNEPS` (default 0.05). Setting `WOUND_SIGNEPS=0` restores the old hard flip — useful only for reproducing the failure. Full derivation in `docs/convergence_failure.md`.

### Output

`writeParaview` emits **two** VTK files per save. The primary carries packed `SCALARS rho_c_phif_kappa` (4 components — the legacy-VTK maximum, so it cannot be widened), a separate `SCALARS alpha` block, `VECTORS a0` and `TENSORS strain`. The `_second_` file carries `Jp_lamdaP`, `VECTORS lamdaE` and `TENSORS stress`. Naming is `<prefix><step+1>.vtk`, plus `REF`, `BCCHECK` and `WOUNDCHECK` sets. `scripts/analyze_multiwound_centers.py` depends on the packed block and skips `_second_` files, so leave it alone.

## Conventions and gotchas

- **The plastic-growth deadband must contain the physiological prestretch.** Healthy dura sits at `lamdaE = (1.098, 1.035, 0.880)`; the old hard-coded `[0.95, 1.05]` excluded the axial and through-thickness values, so healthy tissue remodelled continuously and slowly ate its own prestretch. Now `local_parameters[18]/[19]`, default `[0.85, 1.15]`, with a guard in the driver.
- **`tau_omega`/`tau_kappa` are co-scaled with `K_phi_rho`; `tau_lamdaP_*` are not.** Plastic growth rate therefore moves with any renormalization.
- **`t_rho`/`t_rho_c` are per-cell tractions** multiplied by `rho` in `wound.cpp`, so under normalization they absorb the old `rho_phys` factor (1.28571e-3 MPa). That conversion is traction-preserving.
- The wound is a radial needle track at mid-length (`z = 5.0`), matching `Z1_CENTER` in `scripts/analyze_multiwound_centers.py`. It used to sit at `2*t_dura = 0.8`, almost on top of the clamped bottom ring.
- Concentrations are **normalized** (`c_h = 1`), so literature saturation constants quoted against a `c ~ 1e-4` scale (`K_phi_c`, `K_rho_c`) are artifacts — `plan.md` flags this. `K_phi_c` is 1.08, not 1e-4.
- `readCOMSOLInput` flags `boundary_flag == 1` purely by z-coordinate (Zmin/Zmax ± 1e-4); only values 0 and 1 exist. The lateral surfaces are found geometrically in `main`.
- Result folders, `.zip` archives and `files/*.mphtxt` are data, not code — `.gitignore` admits only `src/ include/ scripts/ meshing/ tests/` plus build and docs.

## Known-dead code

Do not assume these work:

- **Surface BC path is fully dead.** The block is commented out in `solver.cpp`, `evalElemJacobiansSurface` is commented out in the driver, and `readCOMSOLInput` leaves `surface_boundary_flag` empty while still reporting 8449 surface elements. Uncommenting any one alone causes an out-of-bounds read. Relevant only if internal-pressure loading is wanted.
- **`nBC_x`/`nBC_rho`/`nBC_c`/`nBC_alpha` are never read** by the solver. There is no traction BC path.
- **`readTissue` is broken and unused** — four loops use `size()` instead of `i < size()`, and it hard-codes 8-node/8-IP elements. Restart-from-file does not work.
- **`sparseLoadSolver` has no load driver** and assembles only the x-block, leaving singular rows for the other fields. Not used by the current driver.
- `tol_local` and `max_iter` are read but unused in `localWoundProblemExplicit` (fixed 100-substep forward Euler).
- `evalForwardEulerUpdate` has drifted from the live loop (hard-codes `lamdaP(2) → 1.`, 2D `Jp`, un-thresholded `lamdaP_dot`). Only reachable from a commented FD block.
- The 2D twins in `local_solver.cpp` and `evalWoundRes` in `wound.cpp` are dead and deliberately keep the old `det(F^e)` mechanosensing form.

## Remaining known defects

Not yet fixed; they affect the tangent's consistency, not the residual:

1. `local_solver.cpp` — `dThetadrho(5..7)` and `dThetadc(5..7)` receive both an un-thresholded and a thresholded contribution in the same slot (double-counted when `lamdaE` leaves the deadband), and the `else` branches assign `= 0`, wiping accumulation from prior subcycles.
2. `wound.cpp` — `dlamdaP_ndc = dThetadrho(7)` should be `dThetadc(7)`.
3. `wound.cpp` — `dSSdlamdaPs*dlamdaP_ndCC` appears twice in `DDstruct`; `dSSdlamdaPs*dlamdaP_sdCC` and `dSSdlamdaPn*dlamdaP_ndCC` are both missing.
4. `wound.cpp` — the chemotaxis flux lacks the `rho` factor its tangent carries. Inert while `D_rhoc = 0`.
5. `solver.cpp` — the `node_phi`/`node_dphif*` vectors are sized `n_node` then `.clear()`ed, yet indexed. Only reachable via the dead surface block.
