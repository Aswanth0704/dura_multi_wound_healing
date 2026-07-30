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

## Next up — parallel assembly (the throughput bottleneck)

**Measured cost:** ~2 min per global Newton step on the 2t_finer mesh (≈5k nodes,
12 threads), at 4 Newton iterations per step. A 169 h heal at `dt = 0.2` is
~845 steps ≈ 28 h; at `dt = 0.05` it is ~110 h. The 20t production mesh is ~4×
larger and will be proportionally worse. Observed thread utilisation is
350–530% of a possible 1200%.

**Cause.** The element loop (`solver.cpp:394`) is `#pragma omp parallel for`, but
it contains two `omp critical` regions that serialise most of the body:

| Site | What it guards | Verdict |
|---|---|---|
| `solver.cpp:521` | IP store-back **and** the `node_phi`/`node_dphif*`/`node_ip_count` scatter | mostly unnecessary |
| `solver.cpp:545` | the **entire** triplet assembly + `RR` scatter — ~100 lines, 16 tangent blocks | the real serialisation |

### Step A — drop the first critical section
Everything it writes to `myTissue.ip_*` is indexed `ei*IP_size + ipi`, which is
**unique per element**, so threads never touch the same slot — no race exists.
The only genuine race is the `node_*` accumulation, which is a nodal scatter over
shared nodes. Those four vectors are the ones from known defect 5: sized
`n_node`, then `.clear()`ed, then indexed, and only reachable through the dead
surface-BC block. So either delete them outright or guard just those four lines
with `#pragma omp atomic`, and remove the `critical` from the IP store-back.

### Step B — per-thread triplet buffers
Replace the single shared `KK_triplets` with one buffer per thread, merged after
the loop; likewise a per-thread `RR` reduced at the end.

```cpp
const int nthreads = omp_get_max_threads();
std::vector<std::vector<T>> KK_tl(nthreads);
std::vector<VectorXd>       RR_tl(nthreads, VectorXd::Zero(n_dof));
// reserve() per thread from an element-count estimate — the reallocation
// storm is itself a measurable cost at ~1e6 triplets.
#pragma omp parallel for
for (int ei = 0; ei < n_elem; ei++) {
    const int tid = omp_get_thread_num();
    ... KK_tl[tid].push_back(...); RR_tl[tid](dof) += ...;   // no critical
}
for (int t = 0; t < nthreads; t++) {
    KK_triplets.insert(KK_triplets.end(), KK_tl[t].begin(), KK_tl[t].end());
    RR += RR_tl[t];
}
```

`setFromTriplets` sums duplicates, so concatenation order does not change the
assembled matrix. `RR` is a floating-point reduction, so the summation order
*does* change the last bits — reduce in a fixed thread order (the loop above) to
keep it run-to-run deterministic.

*Accept:* on the 2t mesh, a settle run must be **byte-identical** to the current
output (this is a pure refactor — treat any diff as a bug), and the wall clock
per step should drop toward the serial-fraction limit. Then re-measure the
`--ntasks` scaling curve and update the "keep it at ~12" advice in `CLAUDE.md`
and `files/woundcpp3D.sub`, which only exists *because* of this bottleneck.

### Step C — is assembly even the dominant cost? Measure first.
Before doing B, put timers around the three candidates for one step: the element
loop, `localWoundProblemExplicit` inside it, and the linear solve. The local
solver does `time_step_ratio (25) × IP count × Newton iterations` forward-Euler
updates per step, and the direct `SparseLU` factorisation is refactorised every
iteration — either could dominate, in which case fixing the critical sections
buys little. **Do C first**; it is 20 lines and decides whether B is worth it.

### Multi-node is a separate, much larger job
Steps A–B are shared-memory only: they make one node's 12–16 cores actually
work, and cannot go past a single node. Spanning **multiple nodes** needs MPI —
mesh partitioning (METIS), ghost-layer exchange, a distributed sparse matrix and
a distributed solve (PETSc/Trilinos, since `SparseLU` and Eigen's `BiCGSTAB` are
both serial-in-process). That is a rewrite of `sparseWoundSolver`, not a patch,
and it is only worth starting if A–C leave the 20t production mesh too slow.
Note the current `.sub` requests `--ntasks=12` on `--nodes=1` and runs pure
OpenMP, so those "tasks" are already threads, not ranks.

---

## Open items

- **The seeded wound never reaches the nominal ICs, and the reported wound
  values are means — do not read them as the wound core.** Decided to leave as
  is; recorded here so the analysis does not get redone.

  Three separate effects stack, and only the third is a modelling choice:

  1. `verify_run.py` prints the wound **average**, not the minimum, and the mask
     is `rho < 0.5` — which spans the fully-wounded axis out to the
     half-severity radius, so its mean necessarily lands mid-range. At seeding,
     ρ reads 0.3366 as a mean against a global min of 0.0350.
  2. The first row of the healing table is **not** t = 0. The series starts
     after the three dt-ladder stages that absorb the puncture transient
     (~0.7 h), by which point diffusion has already acted (ρ mean 0.337 at
     seeding → 0.451 at the first `w_heal` save). Use `w_WOUNDCHECK.vtk` for the
     true initial condition; it is written before any solve.
  3. The seed is a tanh, not a step:
     `sev(r) = ½[1 − tanh((r − r_wound)/w_smooth)]`, and with
     `w_smooth = 0.1506` mm (1.7 element edges) the ratio `r_wound/w_smooth`
     is only 1.66, so `tanh` = 0.930 and **peak severity at the axis is 0.9651,
     not 1**:

     | field | nominal wound IC | actual at the axis |
     |---|---|---|
     | ρ, c | 1e-4 | 0.0350 |
     | φ | 1e-2 | 0.0446 at the IP, 0.0801 at the node |
     | α | 1 | 0.9651 |

     φ carries an extra step because it is an integration-point variable
     averaged onto nodes for output, so the VTK shows a smoothed field and its
     nodal minimum overstates the core value.

  The smoothing is deliberate — a sharp step gave Galerkin oscillations and
  Newton increments of ~124 — but the cost is that the wound centre keeps 3.5%
  of healthy ρ instead of 0.01%, and 4.5x the intended residual collagen. The φ
  one has physical consequence, since `SSe_pas ∝ phif` sets how far the
  punctured hole snaps open.

  If it ever needs to attain the nominal values, normalise the profile by its
  own peak — one line, keeps the tanh shoulder that buys the convergence:
  ```cpp
  const double sev0 = 0.5*(1.0 - std::tanh(-r_wound/w_smooth));   // 0.9651
  return 0.5*(1.0 - std::tanh((r - r_wound)/w_smooth)) / sev0;
  ```
  Side effect: the half-severity radius moves 0.250 → 0.258 mm, so the wound is
  ~3% wider. The alternatives are a narrower `w_smooth` (under 1.2 elements the
  undershoot returns) or a finer mesh at the wound so 1.7 elements is physically
  smaller — the latter is the right answer for production, at a runtime cost.

- **α retains a small (~1.7%) transient undershoot at the wound front**, while
  ρ, c and φ are now strictly positive. The reason is specific: ρ/c/φ have
  healthy value 1, so a ~2% Galerkin oscillation at the front stays positive,
  whereas **α_h is exactly 0** and has no headroom beneath it. Confirmed by the
  `D_alpha = 0` run, where α's undershoot is 3e-17 — i.e. it is the diffusion
  operator on the front, nothing else.
  `plan.md` line 303 hedges toward α_h = 1e-4 "in case the solver throws
  errors"; that instinct was right, though for a different reason. It is **not**
  adopted here because α_h = 0 is what makes the fixed point exact — α_h = 1e-4
  would inject `p_c_alpha·α_h` = 2.1e-5 against `d_c·c_h` = 3.9e-3, a permanent
  0.54% perturbation to the cytokine balance. A transient artifact that damps is
  preferable to a standing bias.
  If it ever needs removing: give α alone a wider initial profile (~2.5 element
  edges rather than 1.7). That is physically defensible, since the inflammatory
  signal begins diffusing from the injury immediately and its initial footprint
  is broader than the mechanical damage.
- **`time_step_ratio = 100` is ~4x more local substeps than needed** and is the
  dominant runtime cost, especially now that tets carry 4 integration points
  instead of 1. Per global time step the solver does
  `100 substeps x 79,364 IPs x ~6 Newton iterations ~ 48 million` local
  forward-Euler updates. The binding local timescale is `tau_lamdaP` = 0.05 h;
  at `ratio = 100`, `local_dt` = 0.002 h is 25x finer than that. `ratio = 25`
  gives `local_dt` = 0.008 h (0.16 of the binding timescale — still comfortable
  for forward Euler) and would cut wall time ~4x. Worth changing before any
  production run on the 20t mesh. Note the Newton solve itself is NOT the
  bottleneck: healing steps converge in 6–7 iterations at residual ~1e-9.
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
