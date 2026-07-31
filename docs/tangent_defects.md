# Solver tangent defects: diagnosis, derivations, and fixes

Status as of the `normalize-and-alpha` branch, commit `b852900`.

This document records why the wound-healing run kept stalling, how the cause was
finally isolated, and the four defects that were found and fixed. Each fix
carries its derivation and the file and line where it lives.

---

## 1. The symptom

Every healing run decayed and died at roughly the same simulated time:

| run | traction `t_rho` | outcome |
|---|---|---|
| `run_w8` | 1.28571e-3 | aborted at t = 14.24 h, "Solver failed too many times" |
| `run_w10` | 1.28571e-3 | stalled at t = 14.35 h, dt 0.2 → 0.008, 83 bad linear solves |
| `run_t4` | 1.28571e-4 | stalled at t = 14.30 h, dt → 0.0016, 265 bad linear solves |
| `run_t5` | 1.28571e-5 | stalled at t = 14.27 h, dt → 0.0001 |
| `run_w11` | 1.28571e-3 | stalled at t = 14.36 h, 451 bad linear solves |

Three decades of active traction made no difference to *where* it failed, which
ruled traction out as the cause.

The Newton signature was distinctive. A typical stalling step:

```
iteration  6   residual 2.98861e-05   increment 9.27092e-06
iteration  7   residual 2.98838e-05   increment 9.27092e-06
iteration  8   residual 2.98861e-05   increment 9.27092e-06
...
iteration 200  residual 2.98861e-05   increment 9.27092e-06
```

The iterates alternate between exactly two states and repeat **bit for bit** for
hundreds of iterations. That rules out round-off — a floating-point floor would
not reproduce to the last digit — and points at either a non-smooth residual or
a tangent that disagrees with it.

Reducing `dt` did not help. The time step fell by a factor of 125 and the stall
was unchanged, which is expected if the problem belongs to the *state* rather
than to the step size.

---

## 2. Why the first three diagnoses were wrong

Three explanations were proposed and fixed, in order. Each was a genuine defect;
none was the blocker.

1. **Reference/deformed frame mismatch in wound seeding.** The wound was placed
   0.45 mm off the mesh-refined patch and pierced only 55% of the wall
   thickness. Real bug, fixed, run still failed.
2. **The plastic-growth threshold kink.** `lamdaP_dot` was continuous at the
   band edge but its slope jumped, making the residual C⁰ and not C¹. Real,
   fixed by smoothing, bought a ~5× improvement, run still failed.
3. **BiCGSTAB returning a near-zero increment while reporting success.** Real,
   fixed by checking the true linear residual, run still failed.

Each diagnosis was *inferred from symptoms*. The lesson recorded here: the
tangent is not something to infer. It is directly measurable.

---

## 3. The measurement

`tests/test_tangent.cpp` builds one linear tetrahedron in a realistic
prestretched, partly-healed state, calls `evalWound` to obtain the residual
`R` and all sixteen tangent blocks `K`, then perturbs every nodal unknown and
re-evaluates the residual by central differences:

```
K_fd[:,j] = ( R(u + eps e_j) - R(u - eps e_j) ) / (2 eps)
```

and compares block by block, so the output names the offending coupling rather
than merely reporting that something disagrees.

Two things the test must get right, both learned the hard way:

- **`evalWound` mutates its `ip_*` arguments.** The local structural solver
  writes updated `phif / a0 / s0 / n0 / kappa / lamdaP` back through them, so
  every residual evaluation must start from a pristine copy of the baseline
  state or each difference is taken about a different point.
- **The previous-step state must be held fixed.** The residual is
  `((u - u_0)/dt - S)·R`, so the tangent is `dR/du` at fixed `u_0`. The first
  version of the test perturbed both together, which cancels the `R_i R_j/dt`
  mass term *exactly* and showed up as a spurious identical error of
  `6.075e-05` in all three diagonal transport blocks. That number is
  `V/10/dt` for this element — matching to five digits is what gave it away.

A third trap, avoided: the test's parameters must be **copied** from the driver,
not guessed. An early version had `gamma_kappa = 0.05` against the real 5 and
`p_phi` an order of magnitude high, which drove the structural update outside
its valid range and produced a non-finite tangent that read as a serious code
defect.

### Discriminating probes

Beyond the block comparison, the test carries switches that isolate causes:

| switch | effect |
|---|---|
| `TANGENT_SUBSTEPS` | inner explicit-solver substep count |
| `WOUND_FDEPS` | step for the structural finite-difference sensitivities |
| `TANGENT_NOACT / NOFIB / NOVOL` | zero the active / fiber / ground-substance stress |
| `TANGENT_NOSTRUCT` | freeze the structural response entirely |
| `TANGENT_NOPHI / NOA0 / NOKAPPA / NOLAMDAP` | freeze one structural variable |
| `TANGENT_STATE` | `healthy`, `wound`, or `isotropic` base state |

Sweeping a parameter and watching whether the error *moves* distinguishes a
numerical artifact from a wrong term. Both the substep sweep (80-fold) and the
finite-difference sweep (10⁴-fold) left the error bit-identical, which proved
the remaining error was purely analytic.

---

## 4. Defect 1 — duplicate plastic-growth derivative

**File:** `src/local_solver.cpp`, around lines 583 and 646
**Commit:** `6554761`

`dThetadrho(5..7)` and `dThetadc(5..7)` were incremented **twice per substep**:

```cpp
// un-thresholded, WRONG - deleted
dThetadrho(5) += local_dt*((lamdaE_a-1)/tau_lamdaP_a)*dphifdotplusdrho;
...
// thresholded, correct - kept (now lines 615-619, 677-681)
dThetadrho(5) += local_dt*(band_a/tau_lamdaP_a)*dphifdotplusdrho;
```

### Why it is wrong

The residual uses

```
lamdaP_dot = phif_dot_plus * band(lamdaE) / tau
```

where `band()` is the soft threshold, **zero inside the tolerance range**
[0.85, 1.15]. The deleted term used `(lamdaE − 1)`, which is a different
function entirely and is *not* zero there. Healthy dura sits at
`lamdaE = (1.098, 1.035, 0.880)`, so the stray term asserted a sensitivity of up
to 0.12 to remodelling that was not occurring at all.

### Effect

| block | before | after |
|---|---|---|
| `Ke_x_rho` | 2.4e-02 | 1.5e-07 |
| `Ke_x_c` | 1.3e-03 | 2.9e-08 |

---

## 5. Defect 2 — cytokine diffusivity mismatch between residual and tangent

**File:** `src/wound.cpp`, line 1085 (residual at line 583)
**Commit:** `b1d33b9`

The residual uses a **constant** cytokine diffusivity:

```cpp
// src/wound.cpp:583
Vector3d Q_c = -D_cc*CCinv*Grad_c;
// src/wound.cpp:585  (commented out, directly beneath)
//Vector3d Q_c = -1.0*(D_cc-phif*(D_cc-D_cc/10))*CCinv*Grad_c;
```

but the tangent still used the collagen-dependent form:

```cpp
// was
dQ_cdCC_explicit[...] += -0.5*(-1.0*(D_cc - phif*(D_cc - D_cc/10)))*(...)*Grad_c(jj);
// now (line 1085)
dQ_cdCC_explicit[...] += -0.5*(-1.0*D_cc)*(...)*Grad_c(jj);
```

### Why it is wrong

At healthy collagen, `phif = 1`:

```
D_cc - phif*(D_cc - D_cc/10)  =  D_cc - (D_cc - D_cc/10)  =  D_cc/10
```

so the flux contribution to `Ke_c_x` was **ten times too small**. Someone
simplified the residual and never updated the tangent.

### How it was caught

The `Ke_c_x` error (6.2e-04) was an order of magnitude **larger than the
analytic block itself** (6.9e-05) — the signature of a missing term rather than
a wrong coefficient. After the fix the block magnitude grew to 6.9e-04, exactly
the predicted factor of ten.

### Effect

`Ke_c_x`: **2.2e-01 → 1.9e-09**.

---

## 6. Defect 3 — broken symmetry of the fiber second derivatives

**File:** `src/wound.cpp`, line 826 (comment from 817)
**Commit:** `6ee91b0`

### Derivation

The fiber strain energy uses

```
E     = kappa*I1e + (1 - 3 kappa)*I4e - 1
Psif  = (kf / 2 k2) * exp(k2 E^2)
```

with `I1e = tr(CCe)` and `I4e = a0 . CCe . a0`, so `dE/dI1e = kappa` and
`dE/dI4e = (1 - 3 kappa)`. The first derivatives are

```
Psif1 = dPsif/dI1e = 2 k2 kappa       E Psif
Psif4 = dPsif/dI4e = 2 k2 (1-3 kappa) E Psif
```

The mixed second derivative, taken either way round:

```
Psif14 = d(Psif1)/dI4e = 2 k2 kappa       [ (1-3 kappa) Psif + E Psif4 ]
Psif41 = d(Psif4)/dI1e = 2 k2 (1-3 kappa) [ kappa       Psif + E Psif1 ]
```

Substituting `Psif1` and `Psif4` back in, **both** reduce to

```
2 k2 kappa (1 - 3 kappa) Psif ( 1 + 2 k2 E^2 )
```

so they must be equal — as required by symmetry of second derivatives.

### The bug

```cpp
// was — spurious I4e on the first term
double Psif14 = 2*k2*kappa*(1-3*kappa)*I4e*Psif + 2*k2*kappa*E*Psif4;
// now (line 826)
double Psif14 = 2*k2*kappa*(1-3*kappa)*Psif     + 2*k2*kappa*E*Psif4;
```

`I4e` is about 1.2 in the healthy state, so `Psif14 != Psif41` and the passive
tangent was inconsistent with the residual.

### How it was caught

The `Ke_x_x` error sat **flat at 1.30e-02 across an 80-fold sweep of the inner
substep size**. A discretisation gap (the analytic tangent describes the
continuous remodelling law, the residual the discretised one) would shrink as
the substeps refine. It did not move at all, which left a wrong term as the only
explanation.

### Effect

`Ke_x_x`: **1.30e-02 → 5.48e-04** (wound state).

The dead 2D twin at `src/wound.cpp:2748` deliberately still carries the old
form; it is unreachable.

---

## 7. Defect 4 — chain rule for the fiber-dispersion target

**File:** `src/local_solver.cpp`, lines 489-511
**Commit:** `b852900`

### Derivation

Fiber dispersion evolves toward a target set by the eigenvalue ratio:

```
kappa_dot = (1/tau_kappa) * ( r^gamma / 3 - kappa ) * phif_dot_plus
r         = lamdamed / lamdamax
```

Differentiating the target with respect to `CC`:

```
d(r^gamma)/dCC = gamma * r^(gamma-1) * dr/dCC

dr/dCC = (dlamdamed/dCC) / lamdamax
       - lamdamed * (dlamdamax/dCC) / lamdamax^2
```

The exponent becomes a **multiplying factor**. It does not apply to the
derivative, and the difference of the two ratio pieces must be taken *before*
raising to the power, not after.

### The bug

```cpp
// was — raises the derivative itself to the fifth power
pow(dlamdameddCC(ii,jj)/lamdamax, gamma_kappa)
  - pow(lamdamed*dlamdamaxdCC(ii,jj)/(lamdamax*lamdamax), gamma_kappa)

// now (lines 502-511)
const double r       = lamdamed/lamdamax;
const double drdCC   = dlamdameddCC(ii,jj)/lamdamax
                     - lamdamed*dlamdamaxdCC(ii,jj)/(lamdamax*lamdamax);
const double dtarget = gamma_kappa*std::pow(r, gamma_kappa-1.0)*drdCC;
dThetadCC(24+II) += (local_dt/(tau_kappa))*(
                      (dphifdotplusdCC(ii,jj)*(std::pow(r,gamma_kappa)/3. - kappa))
                    + ((phif_dot_plus/3.)*dtarget));
```

With `gamma_kappa = 5` this is not a small discrepancy.

### How it was caught

By elimination, with three probes added to the test:

| probe | result |
|---|---|
| `DDe` (fiber tangent, before the Fg pull-back) | correct to 1.9e-10 |
| Fg pull-back `dSS_pas/dCC` | correct to 1.5e-10 |
| bordered-system eigenvector derivative | correct to 1e-10 |
| **freeze structural response** (`TANGENT_NOSTRUCT`) | **`Ke_x_x` exact, 6.2e-09** |

The last result placed the entire remaining error in `DDstruct`. Freezing each
structural variable individually then showed `lamdaP` was not involved
(unchanged at 5.48e-04) while `kappa` and `a0` both mattered — and the `kappa`
expression had a visibly wrong chain rule.

Note the individual freezes also shift the base state, so only their qualitative
ordering is meaningful.

### Effect

| state | before | after |
|---|---|---|
| wound | 5.48e-04 | 3.45e-04 |
| healthy | 5.73e-04 | 3.79e-04 |
| isotropic | 1.08e-06 | 1.08e-06 (exact) |

Corroboration: the new wound value **equals what freezing `kappa` entirely used
to give**, which is what a correct `kappa` path should look like — its
contribution now matches its absence.

### A misleading intermediate result

An earlier bisection zeroed `kf` and saw the error drop 65%, which appeared to
implicate the passive fiber tangent. That was wrong: zeroing `kf` also removes
the fiber stress from `DDstruct`, whose derivatives are taken by finite
differences of the same stress. The passive path was later shown correct end to
end by probes 1 and 2.

---

## 8. Supporting fixes made along the way

These were needed to reach the point where the tangent could be measured at all.

| fix | file | commit |
|---|---|---|
| Wound bounds mapped into the deformed frame | driver, lines 570-599 | `498500d` |
| Increment-based Newton acceptance (limit-cycle escape) | `solver.cpp:1039-1073` | `498500d`, gated in `c970eec` |
| Verified linear solve instead of trusting `info()` | `solver.cpp:917-975` | `03fd99c` |
| Smoothed plastic-growth threshold | `local_solver.cpp:193-216`, 347 | `2d68bf1` |
| `tests/` admitted to the `.gitignore` allow-list | `.gitignore` | `b865453` |

### Wound placement

`severity()` tested **deformed** coordinates against **reference**-frame bounds.
Three consequences, all measured from `w_WOUNDCHECK.vtk`:

1. `z_center = 5.0` is the reference mid-length, but the settled mesh spans
   `z ∈ [0, 10.98]`, so its mid-length is 5.49. The wound landed at reference
   z = 4.554, off the refined patch (549 nodes within ±0.25 mm instead of 1536).
2. `Xmin_wound = -5.4` is the reference outer radius while the settled outer
   surface is at -5.55799, so the track pierced only the inner **55%** of the
   wall.
3. The free patch tested reference coordinates against the same centre, so patch
   and wound were 0.45 mm apart.

After the fix: 100% of wall thickness, wound centre 5.4870 against a deformed
mid-length of 5.4900 — 3 µm off centre. Wound node count went 101 → 553.

### Smoothed threshold width

The smoothing width is bounded **above by homeostasis**, not chosen for
convenience. A softplus edge leaks `w·log1p(exp(-d/w))` of growth into the band
interior, and the healthy through-thickness stretch `lamdaE_n = 0.880` sits only
0.030 above `lowlim = 0.85` — the tightest of the three (axial has 0.052,
circumferential 0.115):

| w | leak at lamdaE_n = 0.880 |
|---|---|
| 0.010 | -4.9e-04 — would remodel healthy tissue continuously |
| 0.005 | -1.2e-05 |
| **0.002** | **-6.1e-10** |

0.002 is 15× narrower than the closest approach yet still ~200 Newton increments
wide. Override with `WOUND_BANDW`; 0 recovers the original hard threshold.

**Verification:** the 100 h no-wound gate on the smoothed build reproduced the
pre-smoothing reference exactly — rho 1.000008, c 1.000093, phi 1.000164,
theta_e 1.136272, H 0.500680 — with zero rejections.

---

## 9. Current state of the tangent

Measured on the wound state, `tests/test_tangent`:

| block | start of work | now |
|---|---|---|
| `Ke_c_x` | 2.2e-01 | 1.9e-09 |
| `Ke_x_rho` | 2.4e-02 | 1.5e-07 |
| `Ke_x_x` | 1.3e-02 | 3.45e-04 |
| `Ke_x_c` | 1.3e-03 | 2.9e-08 |
| other twelve blocks | exact | exact |

**15 of 16 blocks agree with the numerical derivative to 8 digits or better.**

---

## 10. Outstanding

`Ke_x_x` remains at 3.45e-04 (0.03%). What is known about it:

- It is **not** discretisation — flat across an 80-fold substep sweep.
- It is **not** the finite-difference step — flat across a 10⁴-fold sweep.
- It is **not** the active traction, the Voigt convention, the `FFginv`
  pull-back pairing, the `DDe` coefficients, the geometric stiffness, or the
  eigenvector derivative. All checked, by derivation and by direct probe.
- It **is** in `DDstruct`, and within that in the fiber-direction path, since
  freezing the structural response makes the block exact.
- The error is mostly **symmetric** (3.0e-04 against 1.2e-04 antisymmetric) and
  spread thinly across many entries at ~0.4% each, rather than concentrated —
  which argues against a transposed index pair.

The likeliest remaining explanation is that `da0dCC` (`local_solver.cpp:466-476`)
treats `a0` as fixed within a substep when forming the projector
`(delta_kl - a0_k a0_l)`. That is the explicit-update approximation rather than a
typo, and removing it would mean differentiating the discrete update exactly.

A tangent error of this size costs Newton *iterations*, not correctness: the
residual is untouched, so any converged answer remains the right answer.

---

## 11. Reproducing

```bash
cd build && make test_tangent && ctest          # 4 suites

# the block comparison, on three base states
TANGENT_STATE=healthy   ./test_tangent
TANGENT_STATE=wound     ./test_tangent
TANGENT_STATE=isotropic ./test_tangent

# is an error discretisation or a wrong term?
for n in 5 25 100 400; do TANGENT_SUBSTEPS=$n ./test_tangent | grep Ke_x_x; done

# which stress contribution carries it?
TANGENT_NOSTRUCT=1 ./test_tangent | grep Ke_x_x
TANGENT_NOKAPPA=1  ./test_tangent | grep Ke_x_x
```

Run single-threaded (`OMP_NUM_THREADS=1`) for reproducible output.
