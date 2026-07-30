# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 3D finite-element multiphysics solver for wound healing in spinal **dura mater** ("DaLaWoHe" — Dalai Lama Wound Healing). It couples large-deformation mechanics with reaction–diffusion transport of biological species, plus evolving microstructure (plastic growth, collagen fiber orientation and dispersion).

Fields solved:
- `x` — deformed nodal position (mechanics, 3 dof/node)
- `rho` — fibroblast density (1 dof/node)
- `c` — cytokine / TGF-β1 (1 dof/node)
- `phif`, `a0/s0/n0`, `kappa`, `lamdaP` — collagen fraction, fiber frame, dispersion, plastic stretch — all **integration-point** variables updated by a local (element-level) solver, not global dof.

`plan.md` is the authoritative spec for the model equations, the calibrated parameter set (sweep #4 case 0 medians), and the derived-parameter formulas that enforce homeostasis. Read it before touching any constitutive/source term. It also lists the two open work items: (1) add the pro-inflammatory species `alpha` as a fourth field, (2) update the `D_rho(phi, c)` diffusivity.

C++ with Eigen (dense + sparse), Boost (string algorithms, used by the mesh readers), and OpenMP. Runs on Purdue's Negishi cluster.

## Build and run

There is **no build system checked into this folder** — no Makefile or CMakeLists.txt. The sources reference "MKL is included through the CMake file", so a CMake project exists on the cluster side but is not synced here. The compiled binary is named `woundcpp3D` (see `files/exact_dura/woundcpp3D`, an x86-64 Linux ELF).

Equivalent direct compile (one main + the shared library sources):

```bash
g++ -O3 -fopenmp -std=c++17 \
    -I include -I $EIGEN_DIR -I $BOOST_DIR \
    src/cydindrical_dura_single_tissue_multiwound.cpp \
    src/file_io.cpp src/wound.cpp src/solver.cpp src/local_solver.cpp \
    src/element_functions.cpp meshing/myMeshGenerator.cpp \
    -o woundcpp3D
```

Add `src/file_io_gmsh.cpp` only for the `exact_dura_*` mains. **Do not link `file_io_gmsh.cpp` and `file_io_gmsh_double.cpp` together** — both define `readGmshMsh22doubleWound` and `loadNodeSet0Based`, so they collide.

Eigen and Boost are not installed locally on this Mac (Boost is at `/opt/homebrew/include`, Eigen is absent), so builds and runs happen on Negishi at `/scratch/negishi/athani/athani_wound_healing_simulations/dura_multi_wound_healing/`, with the libraries in the parent `athani_wound_healing_simulations/`.

**Mesh paths are relative and bare** (e.g. `"dura_cyl_repeated_wound_v62_20t_finer.mphtxt"`), so the binary must be launched from a directory containing the mesh file and any node-list `.txt` files. Output `.vtk`/`.txt` files are written to the same cwd — hence the flat result folders like `cyl_dura_only_double_wound_2t_1_week/`.

SLURM submission (`files/circularwoundcpp3D.sub`):
```bash
sbatch circularwoundcpp3D.sub     # 1 node, 12 tasks, OMP_NUM_THREADS=12, 336h wall
```

There is no test suite. `src/test_read_abaqus.cpp` is a standalone mesh-reader smoke check, but its dependency `readAbaqusInputO` (declared in `include/file_io_abaqus.h`) has **no implementation in this tree** — `file_io_abaqus.cpp` is missing.

Post-processing (Python, needs matplotlib/numpy/pandas):
```bash
python scripts/analyze_multiwound_centers.py --root <dir-of-case-folders> --outdir analysis_outputs
python scripts/plot_multiwound_matplotlib.py --outdir analysis_outputs
```
`analyze_multiwound_centers.py` discovers case folders by regex (default matches `cyl_dura_only_double_wound_<N>t_<M>_week`) and hardcodes geometry constants (`T_DURA`, `R_CORD`, `Z1_CENTER`, `DT`) that must be kept in sync with the main file.

## Architecture

### Layers

| Layer | Files | Role |
|---|---|---|
| Case driver (`main`) | `src/*.cpp` with `int main` | Geometry, parameters, initial wound region, BCs, then calls the solver |
| Global solver | `src/solver.cpp`, `include/solver.h` | `tissue` struct, dof maps, element jacobians, sparse Newton–Raphson time loop |
| Element residual/tangent | `src/wound.cpp`, `include/wound.h` | `evalWound` — assembles `Re_x/Re_rho/Re_c` and the 3×3 block of `Ke_*` per element; also fluxes/sources |
| Local (IP) solver | `src/local_solver.cpp` | `localWoundProblemExplicit` — evolves `phif, a0, s0, n0, kappa, lamdaP` at each IP and returns `dTheta/d{CC,rho,c}` sensitivities back into the global tangent |
| Shape functions | `src/element_functions.cpp` | Hex (8/20/27) and tet (4/10) shape functions, jacobians, quadrature |
| Mesh I/O | `src/file_io.cpp`, `src/file_io_gmsh*.cpp`, `meshing/myMeshGenerator.cpp` | COMSOL `.mphtxt`, Gmsh v2.2 `.msh`, Abaqus `.inp`, Paraview readers → `HexMesh`; ParaView/VTK writers |

Element type is inferred everywhere from `elements[0].size()` (8/20/27 → hex, 4/10 → tet), which selects both the quadrature rule and the shape functions. Any new element type must be added in `main`, `sparseWoundSolver`, and `evalWound` consistently.

### The `tissue` struct (`include/solver.h`)

Single mutable state object carrying everything: connectivity, reference (`_0` suffix) and current values for nodal and IP fields, essential/natural BC maps, forward/inverse dof maps, both parameter vectors, cached element jacobians, and time-stepping settings. The solver mutates it in place; after each converged step the current values are copied into the `_0` fields.

### Parameters are positional `std::vector<double>`

This is the single biggest hazard when editing. Both `global_parameters` and `local_parameters` are plain vectors built by index order in `main` and unpacked by literal index in the compute kernels:

- `global_parameters` (25 entries) → `{k0, kf, k2, t_rho, t_rho_c, K_t, K_t_c, D_rhorho, D_rhoc, D_cc, p_rho, p_rho_c, p_rho_theta, K_rho_c, K_rho_rho, d_rho, vartheta_e, gamma_theta, p_c_rho, p_c_thetaE, K_c_c, d_c, bx, by, bz}`
- `local_parameters` (18 entries) → `{p_phi, p_phi_c, p_phi_theta, K_phi_c, K_phi_rho, d_phi, d_phi_rho_c, tau_omega, tau_kappa, gamma_kappa, tau_lamdaP_a, tau_lamdaP_s, tau_lamdaP_n, vartheta_e, gamma_theta, tol_local, time_step_ratio, max_iter}`

The unpacking blocks are **copy-pasted into many functions** — `global_parameters[i]` appears at roughly lines 75, 1185, 1368, 1564, 1687, 2102 of `wound.cpp`, and `local_parameters[i]` at ~64, 771, 900 of `local_solver.cpp`. Inserting a parameter mid-vector silently corrupts every downstream read. When adding the `alpha` species, append at the end and update *every* unpack site.

Note `global_parameters[7]` (`D_rhorho`) is **dead** — nothing reads index 7. The actual fibroblast diffusivity is a hardcoded polynomial-in-`phif` expression duplicated inside `wound.cpp` (search `eq_const`, ~lines 544, 1199, 1590, 2225). Changing `D_rho` per `plan.md` ToDo 2 means editing those expressions, not the parameter in `main`.

### Time loop and convergence (`sparseWoundSolver`)

Monolithic Newton–Raphson on all three field blocks, assembled into triplets and solved with Eigen `BiCGSTAB` on a row-major sparse matrix. Element loop is `#pragma omp parallel for`.

On divergence the solver **restarts the step with `time_step/5`** and simultaneously rescales `save_freq`, `step`, and `total_steps` by the same factor, resetting all `_0` state. After `slowdown` successful steps it restores the original values. Consequence: output step indices in filenames are not a fixed multiple of wall-clock time when slowdowns occur, and more than `slowdown^3` total slowdown throws `"Solver failed too many times!"`.

### Output format

`writeParaview` emits **two** VTK files per save: the primary carries `SCALARS rho_c_phif_kappa` (4 components) and `VECTORS a0`; the `_second_` file carries `SCALARS Jp_lamdaP` and `VECTORS lamdaE`. `writeTissue` writes a matching `.txt` restart-style dump. Naming is `<filename><step+1>.vtk`, `<filename>second_<step+1>.vtk`, `<filename><step+1>.txt`, plus a `REF` set written before the loop. The Python analysis scripts rely on this exact convention and skip `_second_` files.

### Multi-wound sequencing

`cydindrical_dura_single_tissue_multiwound.cpp` is the designated main file. It models repeated needle punctures by calling `sparseWoundSolver` once, then **re-seeding wound state into the already-deformed `myTissue`** (shifting `z_center`, resetting `node_rho/node_c` and the IP fields inside the new cylinder), then calling the solver again with a new output prefix. The second-wound block is currently commented out — only wound 1 runs, per `plan.md`. Note the second seeding tests membership against `myTissue.node_x` (deformed coords) while the first uses `myMesh.nodes` (reference coords).

Fiber frames are built by `build_cylinder_frame` in the main file: `a0` axial, `n0` radial from the cylinder axis at `(0,0)`, `s0` circumferential, re-orthogonalized. The old planar `Rot90` construction is left commented and explicitly marked invalid for cylindrical dura.

### Case-driver variants in `src/`

Only one is the active main; the rest are historical cases kept for reference. `cydindrical_dura_single_tissue_multiwound_original.cpp` differs from the active main by a single line (the mesh filename). The `exact_dura_*` mains use the Gmsh path with explicit `bottom_end_nodes.txt`/`top_end_nodes.txt` node lists and domain tags (wound1=101, healthy=102, wound2=104); the `cyl*`/`circular*`/`square*` mains use COMSOL or Abaqus meshes.

## Conventions and gotchas

- Lines marked `// CHANGE` in the main file are the per-run knobs: mesh filename, output prefix, `time_final`, the `z_center` offset for wound 2, and the `eBC_c` value. Sweep these when generating a case series.
- `boundary_flag[nodei] == 1` means fully fixed: displacement pinned to reference position, `rho` and `c` clamped to healthy values.
- The main file dumps every node, element, boundary flag, element jacobian, and dof map to `stdout` before solving. On a fine mesh this is gigabytes of console output — redirect it or strip the loops when running production cases.
- Concentrations are **not** normalized in the code: `c_max = 1e-4 g/mm^3`, `rho_phys = 1000*55.05126 cells/mm^3`. `plan.md` states the calibrated parameter table in **normalized** units (`c_h = 1`, `rho_h = 1`), and warns that several literature saturation constants (`K_phi_c`, `K_rho_c`) are c-scale artifacts. Convert deliberately when transferring values from the table into `main`.
- `d_rho`, `K_phi_rho`, and (per `plan.md`) `p_c_rho` are *derived* in `main` from steady-state homeostasis, not sampled. Keep the derivation and the sampled values consistent — `plan.md` notes the cytokine constraint was inverted on 2026-07-29 to sample `K_c_c` and derive `p_c_rho`.
- Result folders (`cyl_dura_only_double_wound_*/`), `.zip` archives, and `files/*.mphtxt` meshes are data, not code. Per `plan.md`, only `src/`, `include/`, `scripts/`, and `meshing/` are meant to be version-controlled; this directory is not currently a git repository.
- Files must stay in sync between this Box folder, the Negishi scratch directory, and (once created) GitHub.
