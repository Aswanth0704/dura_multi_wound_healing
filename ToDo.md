# ToDo — normalize parameters, fix mechanosensing, add pro-inflammatory species α

Working document. Steps are ordered so each one builds and can be verified on its own.
**Implement strictly one at a time — do not start the next until the current one passes its check.**

Companion to `plan.md` (which holds the model equations and the calibrated parameter table).

---

## Context — why this ordering

`src/cydindrical_dura_single_tissue_multiwound.cpp` currently runs an **unnormalized** parameter set
(ρ_phys = 55051.26 cells/mm³, c_max = 1e-4 g/mm³) inherited from earlier skin/breast work, with a
mechanosensing term that is physically inert. `plan.md` now specifies a **normalized** set
(ρ_h = c_h = φ_h = 1) from sweep #4 case 0, an areal-stretch stimulus, a revised fibroblast
diffusivity, and a fourth species α that does not exist in the code at all.

Three findings from reading the code drive the ordering:

1. **The areal-stretch stimulus is not implemented.** It exists only as commented sketch lines
   (`src/wound.cpp:493`, `:1269`). Every live `H` uses `Je = sqrt(CCe.determinant())` = det(F^e) at
   `wound.cpp:574`, `:997`, `:1333` and `local_solver.cpp:202`, `:375`. Since the tissue is treated
   as incompressible (λ_N = (θ^e)⁻¹ ⟹ det F^e ≡ 1), the current stimulus is **frozen** — H never
   responds to membrane stretch.

2. **H_h = 1/2 requires θ^e = 1.136 in the healthy state**, but the code starts at
   `node_x = node_X`, `lamdaP = (1,1,1)` ⟹ F^e = I ⟹ θ^e = 1 ⟹ H = 1/(1+e^1.36) = 0.204.
   Every derived parameter (K_ρρ, K_φρ, p_c,ρ) assumes 0.5, so without a prestretch stage healthy
   tissue drifts and nothing downstream is interpretable.

3. **No load path exists.** `nBC_x` / `nBC_rho` / `nBC_c` are declared (`include/solver.h:73-76`)
   and filled by main but **never read** by `solver.cpp` or `wound.cpp`. `sparseLoadSolver` has no
   Dirichlet ramp, no traction, and zero body force — it converges at zero displacement, i.e. it
   does nothing today. Its closing block (`solver.cpp:1211-1220`) copies `node_x → node_X`, which
   would *erase* any prestretch. So the preload stage has to be built, not just enabled.

**Decisions already taken:** normalized units · derived parameters computed from closed forms in
`main` · prestretch via a preload stage (not `lamdaP` init) · D_α = D_c = 0.00930 mm²/h · only the
multiwound main is updated and CMake is repointed at it.

**Still open — needed before Step 6:** a value for `k_cut` in the `C_low` gate, and whether `C_low`
replaces or stacks with the existing `phif - phif00` polynomial shift (both gate at φ = 0.01).
User comment: For now use $a = 182.01$, $b = -655$, $c = 875.66$, $d = -521.57$, $e = 118.9$. $\text{const} = 1582.3$, $k_{cut} = 300$ when defining D_rhorho

---

## Build and run (Negishi)

Do not run here — too slow.

```bash
module load boost cmake intel/19.1.3.304
mkdir run_<case> && cd run_<case>
cmake .. && make                  # ../eigen and ../boost_1_71_0 resolve from the project root
cp ../files/<mesh>.mphtxt ../files/circularwoundcpp3D.sub .
sbatch circularwoundcpp3D.sub     # 12 tasks, OMP_NUM_THREADS=12 User comment: You can add more if needed.
```

The binary reads mesh paths as bare relative strings and writes all `.vtk`/`.txt` to the cwd, so each case needs its own run directory. Use `dura_cyl_repeated_wound_v62_2t_finer.mphtxt` (≈5k nodes) for every verification step; save `..._20t_finer` for production.

---
User comment: first push the code to github to a branch and then start implementing on new branch. 

## Step 0 — Repoint the build

- [ ] `CMakeLists.txt:53`: swap `src/exact_dura_only_double_wound.cpp` →
      `src/cydindrical_dura_single_tissue_multiwound.cpp`.

Leave `src/file_io_gmsh.cpp` in the target (harmless). Do **not** add `file_io_gmsh_double.cpp` — it redefines `readGmshMsh22doubleWound` and `loadNodeSet0Based`, so the link fails.

**Check:** `cmake .. && make` clean; binary runs a few steps on the 2t mesh and writes `...REF.vtk`.
Keep that VTK as the byte-comparison baseline for Step 1. # User comment: this is helpul to visually check if the boundary conditions are correct. Its important before we actually run (User comment)

---

## Step 1 — Factor `D_rhorho` into one helper

The expression is copy-pasted verbatim at `src/wound.cpp:544`, `:1199`, `:1590`, `:2225`, each
preceded by its own copy of `eq_const/eq_a..eq_e/phif00` (`:537-543` etc.).

- [ ] Replace all four with calls to one file-local helper, e.g.
      `static double evalDrho(double phif, double c)`.

This step exists purely so Step 6 is a one-line edit instead of four.

**Check:** no behavior change — output VTK byte-identical to the Step 0 baseline.

---

## Step 2 — Areal-stretch mechanosensing

- [ ] Replace the stimulus at the three live `H` sites: `wound.cpp:574`, `:1333`,
      `local_solver.cpp:202`.
- [ ] Replace both analytic derivatives: `wound.cpp:997`, `local_solver.cpp:375`.
- [ ] Update the dead-but-declared `evalS` (`wound.cpp:1633`) to keep it consistent.
- [ ] Leave `Je` alone where it feeds the **volumetric stress** (`wound.cpp:527-534`) — genuinely a
      different quantity.

Because `Fg·n0 = λ_N·n0`, the closed form collapses to something cheap:

```cpp
double g       = n0.dot(CCinv*n0);
double theta_e = J*sqrt(g)/(lamdaP_a*lamdaP_s);        // = ||cof(F^e).n0||
double He      = 1./(1.+exp(-gamma_theta*(theta_e - vartheta_e)));

// dtheta_e/dCC = (theta_e/2)*[ CCinv - (CCinv n0)(CCinv n0)^T / g ]
Vector3d Cin         = CCinv*n0;
Matrix3d dthetadCC   = 0.5*theta_e*(CCinv - (Cin*Cin.transpose())/g);
Matrix3d dHedCC_expl = gamma_theta*He*(1.-He)*dthetadCC;
```

`n0` is already the through-thickness normal — `build_cylinder_frame`
(`...multiwound.cpp:35-80`) sets `n0` radial, `a0` axial, `s0` circumferential — so this is exactly
the dural mid-surface areal stretch. `evalFluxesSources` recomputes `H` from its arguments, so the
existing central-difference machinery at `wound.cpp:691-767` picks up dθ^e/dφ_f and dθ^e/dλ^P for
free; no new FD pairs needed.

**Check:** print θ^e and H at a handful of IPs. With `lamdaP = I` and `F = I`, θ^e must be exactly
1.0. Verify the identity the old comment at `wound.cpp:495-497` asserts: θ = θ^e·θ^p where
θ^p = λ_a λ_s.

User comment: For cylindrical shape, this is fine, but for exact dura geometry, I dont know what is the thickness orientation. And we don't know how it changes. So this can be challenging. 

---

## Step 3 — Normalized parameter block with derived closed forms

Rewrite the parameter section of `src/cydindrical_dura_single_tissue_multiwound.cpp` (lines ~89-188).

- [ ] Drop `rho_phys` and `c_max` entirely; set `rho_h = c_h = phi_h = 1.0`.
- [ ] Load the sampled/fixed values from the `plan.md` median set (sweep #4 case 0).
- [ ] Compute the derived block in code:

```cpp
double H_h = 0.5;                       // by construction, since vartheta_e == theta_e_h
double prolif_h  = p_rho + p_rho_c*c_h/(K_rho_c+c_h) + p_rho_theta*H_h;
double K_rho_rho = rho_h/(1.0 - d_rho/prolif_h);                      // -> 1.164
double K_phi_rho = (p_phi + p_phi_c*c_h/(K_phi_c+c_h) + p_phi_theta*H_h)*rho_h
                 / ((d_phi + c_h*rho_h*d_phi_rho_c)*phi_h) - phi_h;   // -> 0.704
double p_c_rho    = d_c*(K_c_c+1.0)/(1.0 + H_h*r_c_e);                // -> 0.00694
double p_c_thetaE = r_c_e*p_c_rho;                                    // -> 0.00310
```

All four reproduce the `plan.md` table values to three digits (verified by hand).

- [ ] Add the two admissibility checks `plan.md` warns about (Harbin's fit returned K_φρ = −0.205),
      aborting with a clear message: `K_phi_rho > 0` and `K_rho_rho > rho_h`.
- [ ] Mechanosensing: `vartheta_e = 1.136`, `gamma_theta = 10.` — these feed **both**
      `global_parameters[16]/[17]` and `local_parameters[13]/[14]` from the same two variables, so
      they cannot drift.
- [ ] Initial conditions per `plan.md` lines 302-305: healthy (ρ,c,φ) = (1,1,1);
      wound (ρ,c,φ) = (1e-4, 1e-4, 1e-2).
- [ ] Set `global_parameters[7]` (the dead `D_rhorho` slot) to a clearly-marked unused value.

### Three traps in this block

| Trap | Where | Fix |
|---|---|---|
| **`K_phi_c = 0.0001` is a hard-coded `c_max`** written as a bare literal, so it will not follow the normalization | main `:156` | becomes `1.08` |
| **`t_rho`, `t_rho_c` are per-cell tractions** — `traction_act = (t_rho + t_rho_c*c/(K_t_c+c))*rho` | `wound.cpp:521` | with ρ_h = 1 they must absorb the old ρ_phys factor: `t_rho = 1.28571e-6*1000 = 1.28571e-3` MPa, `t_rho_c = t_rho*3.28571` |
| **`tau_lamdaP_{a,s,n} = 0.05` are not co-scaled** with K_φρ, unlike `tau_omega`/`tau_kappa` which are formulas in `(K_phi_rho+1)` | main `:172,175,179-181` | plastic growth rate shifts with renormalization — flag for retuning if λ^P misbehaves in Step 5 |

**Check:** print all 25 global and 18 local parameters plus the derived values; compare against the
`plan.md` table by eye. No run needed yet — the homeostasis test is Step 4.

User's comment: high t_rho, t_rho_c values can cause the convergence issue based on my experience. 
---

## Step 4 — Prestretch preload stage  ← the substantive new piece

Goal: reach θ^e = 1.136 uniformly so H = 0.5 exactly, and **hold it** through the healing run.

Design reuses existing machinery — no surface-integral work, no `nBC` revival:

- [ ] In `main`, identify all outer-boundary nodes geometrically: top/bottom rings (z ≈ Zmin, Zmax,
      which `readCOMSOLInput:701-712` already flags as `boundary_flag == 1`) **plus** the inner and
      outer lateral surfaces (r ≈ r_cord and r ≈ r_cord + t_dura). No reader change needed — `main`
      already loops every node.
- [ ] Prescribe `eBC_x` on those nodes at the affine prestretched positions
      `x = diag(1.035, 1.035, 1.098)·X` (circumferential × circumferential × axial, from `plan.md`'s
      1.098 × 1.035 Consolini measurement). User Comment: What is this. I did not understand this.
- [ ] Seed `node_x` with the same affine field as the initial guess so Newton starts close.
- [ ] Keep `node_X` as the **unloaded** reference and **do not re-reference** — i.e. do not use
      `sparseLoadSolver`'s closing block at `solver.cpp:1211-1220`. Then `lamdaP = I`, `F^e = F`, and
      θ^e = 1.035 × 1.098 = 1.136.
- [ ] Run a short settling solve with **no wound** so mechanics equilibrates while biology sits at
      (1,1,1).

Prefer driving the settling phase with `sparseWoundSolver` rather than repairing `sparseLoadSolver`:
the load solver assembles only the x-block (`solver.cpp:1124-1141`), leaving all-zero rows and
columns for the ρ/c dof in `KK2` — a singular system that BiCGSTAB is merely tolerating. If a
mechanics-only preload is wanted later, that singularity has to be fixed first.

**Check — this is the key gate for Steps 2-4 together.** Run ~100 h with no wound:
- θ^e must sit at 1.136 ± 1e-3 and H at 0.5 ± 1e-3 away from the boundaries
- ρ, c, φ must not drift from 1.0 by more than ~1e-3

Any systematic drift means a derived parameter or the H_h assumption is inconsistent. **Stop and fix
here** — every later step builds on this.

---

## Step 5 — Seed the wound after the preload

- [ ] Move wound seeding to after the settling solve, reusing the re-seed pattern already written
      for wound 2 (`...multiwound.cpp:529-622`) — it re-seeds `node_rho/node_c` and the IP fields
      inside a cylinder on an already-deformed tissue, which is exactly what is needed.

Note that block tests membership against `myTissue.node_x` (deformed) while the original wound-1
seeding at `:343-352` uses `myMesh.nodes` (reference). Pick deformed coordinates deliberately and
comment why.

**Check:** single-wound healing run on the 2t mesh, ~1 week simulated. ρ and c rise in the wound then
relax back toward 1; φ recovers from 0.01 upward; no negative concentrations. There is **no** clamping
or NaN guard anywhere in the solver (verified: zero hits for `isnan|max|min|clamp` in `solver.cpp`
and `wound.cpp`), so negative ρ or c is a silent failure — inspect the VTK output.

---

## Step 6 — New fibroblast diffusivity

- [ ] In the Step 1 helper, add the `C_low` gate and floor from `plan.md`:
      `D_ρ = const·(1e-3·P(φ))²/6·C_up·C_low + 6.12e-5` with `C_low = 0.5(1+tanh(k_cut(φ − 0.01)))`.

**Open questions before starting:**
- `k_cut` has no value in `plan.md`.
- The current code achieves its low-φ cutoff by *shifting the polynomial argument*
  (`phif - phif00`, `phif00 = 1e-2`) instead. Does `C_low` **replace** that shift or **stack** with
  it? Stacking gates twice at the same φ = 0.01.

**Check:** tabulate D_ρ over φ ∈ [0, 1.2] and confirm the floor (≈6.12e-5 at φ = 0.01), the interior
peak, and the C_up shutoff above φ = 1.

---

## Step 7 — Missing flux terms in the tangent

`D_ρ` depends on φ_f, and φ_f depends on ρ and c through the local solver, but that chain rule is
applied only in `Ke_rho_x`. `Ke_rho_rho` (`wound.cpp:1150`) and `Ke_rho_c` (`:1156`) use only
`linQ_rhodGradrho`/`linQ_rhodGradc` (`:974-977`), with `D` frozen. Contrast the source terms, which
*are* fully chain-ruled (`:985-992`).

- [ ] `dQ_rhodphif` is already computed at `:694` — multiply it by `dphifdrho`/`dphifdc` and contract
      with `Grad_R[nodei]` into `Ke_rho_rho` and `Ke_rho_c`.

This was survivable with a mild D(φ); it matters once Step 6 installs a sharp `tanh` gate.

**Check:** Newton iterations per step should not increase versus Step 6, and ideally drop. Compare the
printed residual histories. (Fix defect #1 below first — Step 7 is judged on convergence, and that
defect corrupts convergence.)

---

## Steps 8-13 — Add the α species

α is a 4th nodal field: `α̇ + ∇·Q_α = s_α`, `Q_α = −D_α∇α`, `s_α = −d_α α`, feeding cytokine
production via `s_c += p_c,α·α`.

Its own equation has **no** dependence on φ_f, a0, κ or λ^P, so **no new local-solver sensitivities
are needed** — `dThetadalpha` is not required and the 14 existing finite-difference calls in
`evalWound` stay untouched.

DOF layout is **interleaved per node** (one `dof_count` in a single node loop at
`solver.cpp:54-93`), so α becomes a 6th slot inside each node's block with `dof_inv_map` tag `3`.
Essential BCs are enforced purely by **dof elimination** (`dof_fwd_map_* == -1`, value written once
into the nodal field at `:70/:81/:91`) — no penalty, no row zeroing — so α follows the same pattern.

New global parameters **appended** at indices 25, 26, 27: `D_alpha = 0.00930`, `d_alpha = 0.0128`,
`p_c_alpha = 0.208`. Appending is mandatory — `global_parameters[0..24]` are unpacked by literal index
at six separate sites in `wound.cpp` (`:75-98`, `:1185-1214`, `:1368-1388`, `:1603-1614`,
`:1695-1698`, `:2334-2343`), so inserting mid-vector silently corrupts every downstream read.

### Step 8 — `include/solver.h`
- [ ] `tissue` gains `node_alpha_0`, `node_alpha` (near `:34-35`, `:49-50`), `eBC_alpha`,
      `nBC_alpha` (near `:68-76`), `dof_fwd_map_alpha` (near `:82-83`).

**Check:** compiles; nothing reads the new members yet.

### Step 9 — `solver.cpp: fillDOFmap`
- [ ] Add the α block mirroring the c block at `:83-92`, tag `3`.

**Check:** printed `n_dof` grows by exactly the number of unconstrained nodes.

### Step 10 — `include/wound.h` + `wound.cpp: evalWound`
- [ ] Signature grows from 12 to 20 residual/tangent arguments (4 residuals + 16 blocks; 7 are new).
- [ ] Add α to the field declarations (`:202-205`), the nodal interpolation loop (`:207-237`), and
      the reference gradients (`:266-269`).
- [ ] Add `Q_alpha` / `S_alpha`, the residual at `:616-617`, and the tangent loop at `:1123-1170`.
- [ ] Add `p_c_alpha*alpha` to `S_c` (`:579`, `:1338`, `:1633`) and the new `dS_cdalpha` for
      `Ke_c_alpha`.

The residual convention is `((u−u_0)/dt − S)*R − Grad_R·Q`, i.e. backward Euler in **rate** form —
which is why the tangent carries `+R_i R_j/dt` and not `+R_i R_j`.

**Check:** compiles; with α ≡ 0 everywhere, results match Step 7 exactly.

### Step 11 — `solver.cpp: sparseWoundSolver`
- [ ] Element declarations (`:434-447`) → 4 residuals + 16 blocks.
- [ ] The `evalWound` call (`:452-465`).
- [ ] Triplet assembly (`:501-571`) → a 4th top-level row block plus one extra column branch in each
      of the three existing blocks.
- [ ] Solution update `dof_inv_map` dispatch (`:782-798`), add `== 3`.
- [ ] Reset/rollback restore (`:850-854`).
- [ ] `_0` copy-forward (`:891-896`).

**Check:** α ≡ 0 still reproduces Step 7 bit-for-bit.

### Step 12 — Main file
- [ ] α initial conditions: healthy 0, wound 1.0.
- [ ] `eBC_alpha` on the flagged boundary nodes alongside `eBC_rho`/`eBC_c` at `:337-339`.
- [ ] The three appended global parameters.

`plan.md` line 303 hedges toward α_h = 1e-4 for stability. **α_h = 0 is safe** — nothing divides by
α, and with `s_α = −d_α α` zero is the exact fixed point, so 1e-4 would decay to it anyway
(half-life ln2/0.0128 ≈ 54 h). Use 0 and keep 1e-4 as a fallback.

**Check:** α decays from 1.0 in the wound with the right half-life; c shows the extra α-driven
production.

### Step 13 — `src/file_io.cpp: writeParaview`
Both `SCALARS` blocks are already at 4 components, the legacy-VTK maximum (`:1301`, `:1302`).

- [ ] **Append a new attribute block** inside the existing `POINT_DATA` section rather than bumping
      the count. Do **not** emit a second `POINT_DATA` line.
- [ ] Add `node_alpha_0` to `writeTissue` (after `:578`) — it is called at `solver.cpp:920`.

**Check:** α renders in ParaView; the existing `rho_c_phif_kappa` field is unchanged, so
`scripts/analyze_multiwound_centers.py` (which reads that block and skips `_second_` files) keeps
working.

---

## Verification summary

The single most important gate is **Step 4**: a no-wound run holding (ρ,c,φ) = (1,1,1) at
θ^e = 1.136 and H = 0.5. If that drifts, the parameter derivation or the H_h assumption is wrong and no later result means anything.

After each step, run on Negishi with the 2t mesh in a fresh run directory and diff the printed
parameter dump and residual history against the previous step. **Steps 1, 10 and 11 are pure
refactors with byte-identical expected output — treat any change as a bug.**

Note the main file prints every node, element, boundary flag, element jacobian and dof map to stdout
before solving (`...multiwound.cpp:261-499`). On a fine mesh that is gigabytes — redirect it, or gate
those loops behind a verbosity flag before production runs.

---

## Pre-existing defects found while planning

Not part of this work, but they affect results now and will confuse debugging. Ranked by impact.

1. **`wound.cpp:510` vs `:1288` — `Psif` inconsistency.** `(kf/2k2)*exp(...)` in `evalWound` but
   `(kf/2k2)*(exp(...)−1)` in `evalFluxesSources`. Since `Psif1`/`Psif4` multiply `Psif`, the
   finite-difference structural sensitivities are inconsistent with the analytic stress. Degrades
   Newton convergence, not the residual. **Worth fixing before Step 7**, which is judged on
   convergence.
2. **`local_solver.cpp:490-492` / `:566-568` vs `:514-538` / `:590-614`.** `dThetadrho(5..7)` and
   `dThetadc(5..7)` receive both an un-thresholded and a thresholded contribution in the same slot
   (double-counted when λ^E leaves [0.95, 1.05]), and the `else` branches assign `= 0`, wiping
   accumulation from prior subcycles. Tangent-consistency bug, independent of units.
3. **`wound.cpp:446`** — `double dlamdaP_ndc = dThetadrho(7);` should be `dThetadc(7)`. One-line
   copy-paste error feeding `Ke_x_c`.
4. **`wound.cpp:852`** — `dSSdlamdaPs*dlamdaP_ndCC` appears twice; `dSSdlamdaPs*dlamdaP_sdCC` and
   `dSSdlamdaPn*dlamdaP_ndCC` are both missing from `DDstruct`.
5. **`wound.cpp:564`** — the chemotaxis flux is `−D_rhoc*CCinv*Grad_c` (no `rho` factor) while its
   tangent at `:976` and `:1012` both carry `rho`. Currently inert because `D_rhoc = 0`.
6. **`solver.cpp:356-360`** — vectors are sized `n_node` then immediately `.clear()`ed, yet indexed
   at `:491-495`. Out-of-bounds writes that happen to land in still-allocated capacity.
7. **`readTissue` (`file_io.cpp:196-532`) is broken and unused.** Four loops at
   `:483/:496/:509/:523` use `size()` as the condition instead of `i < size()`; `:367` reads
   `node_x[i](1)` twice, never `(2)`; hard-codes 8-node elements and 8 IP/element (wrong for the
   4-node tets in use). Restart-from-file does not work.
8. **`local_solver.cpp:87/:89`** — `tol_local` and `max_iter` are read but never used in
   `localWoundProblemExplicit` (it is fixed-count forward-Euler subcycling, 100 substeps). Tuning
   them does nothing.
9. **`evalForwardEulerUpdate` (`local_solver.cpp:763-868`) has drifted** from the live loop
   (hard-codes `lamdaP(2) → 1.`, 2D `Jp`, un-thresholded `lamdaP_dot`). Only reachable from the
   commented FD block at `:631-691` — repair before re-enabling.
10. **Surface BC path is fully dead.** The block is commented out (`solver.cpp:576-741`),
    `evalElemJacobiansSurface` is commented out in main (`:478`), and `readCOMSOLInput` leaves
    `surface_boundary_flag` empty (`file_io.cpp:665`; push_backs exist only inside the comment at
    `:818-888`) while still reporting 8449 surface elements. Uncommenting any one of these alone
    causes an out-of-bounds read. Relevant only if a pressure load is wanted later.

---

## Two notes on `plan.md` itself

- **Line 55 sign.** The fibroblast equation is written `ρ̇ = ∇·Q_ρ + s_ρ` with `Q_ρ = −D_ρ∇ρ`, which
  gives anti-diffusion. The code implements `ρ̇ + ∇·Q_ρ = s_ρ` (`wound.cpp:616`), consistent with the
  c and α equations. Almost certainly a typo worth correcting. User comment: You are right.
- **`src/cydindrical_dura_single_tissue_multiwound_original.cpp` is byte-identical to the main file
  except for the mesh filename on line 245.** It will silently diverge once Step 3 lands. Kept for
  now by choice — consider deleting it later, since the mesh filename is the only thing it varies. User comment: Yeah, I will delete it. But I kept it here just in case.
