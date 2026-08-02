# Why the wound-healing simulation would not run, and how it was fixed

**Date:** 1 August 2026
**Scope:** the failure of the coupled solver to advance past roughly 17–21 hours of
simulated time, its root cause, the fix, and every secondary defect and physical
finding uncovered along the way.

This document is a companion to `docs/tangent_defects.md`, which covers an earlier
campaign on the consistency of the element tangent matrix. That work was necessary
but it was not what stopped the simulation; this document explains what did.

---

## Contents

1. [The symptom](#1-the-symptom)
2. [What was ruled out, and by what measurement](#2-what-was-ruled-out-and-by-what-measurement)
3. [The measurement that identified the cause](#3-the-measurement-that-identified-the-cause)
4. [Root cause: a discontinuous eigenvector sign convention](#4-root-cause-a-discontinuous-eigenvector-sign-convention)
5. [The fix, and why it is safe](#5-the-fix-and-why-it-is-safe)
6. [Verification by controlled comparison](#6-verification-by-controlled-comparison)
7. [Secondary defect: wound placement in the wrong configuration](#7-secondary-defect-wound-placement-in-the-wrong-configuration)
8. [Secondary defect: the resistance to volume change vanished inside the wound](#8-secondary-defect-the-resistance-to-volume-change-vanished-inside-the-wound)
9. [Secondary defect: unbounded growth](#9-secondary-defect-unbounded-growth)
10. [Smaller defects](#10-smaller-defects)
11. [Physical findings that are not defects](#11-physical-findings-that-are-not-defects)
12. [Diagnostics added](#12-diagnostics-added)
13. [State of the simulation at the end of this work](#13-state-of-the-simulation-at-the-end-of-this-work)
14. [Files changed](#14-files-changed)

---

## 1. The symptom

The simulation solves four coupled fields on the deformed configuration: the nodal
position $\mathbf{x}$, the fibroblast density $\rho$, the cytokine concentration
$c$, and the pro-inflammatory signal $\alpha$. All four are advanced together by a
Newton–Raphson iteration at each time step.

Every run of the punctured, pre-stretched cylinder stalled between **16.8 and 21.6
hours of simulated time**. The time step would be rejected, halved, rejected again,
and the run would grind to a standstill. The target was **672 hours (four weeks)**,
so the simulation was reaching at most 3 % of what was required.

The behaviour was insensitive to every parameter tried:

| Quantity varied | Range | Effect on the stall point |
|---|---|---|
| Active traction $t_\rho$ | three orders of magnitude | none |
| Fibroblast diffusivity cut-off $k_{\text{cut}}$ | 20 to 300 | residual statistics identical to the last digit |
| Local sub-steps | 12.5 to 1000 | none |
| Finite-difference step for the tangent | $10^{-9}$ to $10^{-5}$ | none |

---

## 2. What was ruled out, and by what measurement

Each of the following was proposed, tested, and eliminated. They are recorded
because several were reported as likely causes before the evidence contradicted
them, and the record should show that.

### 2.1 The transport and biology are correct

A control run pinned every nodal displacement after the puncture, so only the three
transport fields were solved on a fixed deformed mesh. It **completed 27.6 hours in
52 minutes with zero rejected steps**, passing 13 of 14 healing checks. The
inflammation signal cleared monotonically, the cytokine peaked and resolved, the
fibroblasts repopulated, and the surrounding healthy tissue held its homeostatic
state to four decimal places.

Conclusion: the chemistry, the collagen model, the inflammation model and the time
integration are sound. The failure is in the mechanical response.

### 2.2 The failure is entirely in the mechanics

An instrument was added that prints, at a chosen Newton iteration, the twenty
degrees of freedom carrying the largest residual, with their field type and
coordinates. Across four independent runs, **every one of the top twenty was a
displacement degree of freedom.** Not a single transport degree of freedom appeared.

### 2.3 It is not the collapse of element volume

The elastic volume ratio $J^e = \det(\mathbf{F}^e)$ was measured exactly by
reconstructing the elastic right Cauchy–Green tensor from the saved strain field,
using $\mathbf{C}^e = 2\mathbf{E} + \mathbf{I}$. It falls from 0.218 to 0.154 over
sixteen hours in a ring of elements at the wound rim. This is a real pathology
(section 8) but it is **not** where the solver fails: the residual concentrates at
0.44–1.36 mm from the needle axis, while the collapsing ring sits at 0.23–0.31 mm.
Different locations.

### 2.4 It is not element inversion or a singular barrier

$\det(\mathbf{C}^e)$ was checked directly for non-positive values. The minimum was
$4.7\times10^{-2}$, with **zero** elements at or below zero and none within three
orders of magnitude of zero. The logarithmic barrier that prevents inversion is
therefore never evaluated near its singularity.

### 2.5 It is not mesh resolution

The mesh was measured rather than assumed. It is strongly graded: the mean element
edge is 0.0587 mm within 0.5 mm of the needle track and 0.556 mm in the far field.
Element quality is good — median quality measure 0.764 on a scale where 1 is a
regular tetrahedron, no slivers, aspect ratio median 1.91. The wound radius spans
4.3 local elements.

A run on a mesh with 169 727 elements and 167 166 unknowns (against 19 961 and
21 225) made no difference.

### 2.6 It is not the plastic-growth threshold

This one was proposed **with supporting numbers that turned out to be
coincidental**, and is recorded as a caution. The argument was that the healthy
through-thickness elastic stretch, $\lambda^e_n = 0.880$, sits only 0.030 above the
lower threshold of 0.85, while the smoothing width is 0.002 — fifteen smoothing
widths of margin, against seventy-five without pre-stretch. Measurements confirmed
that puncturing drives 2 155 of 5 054 integration points across the threshold and
leaves roughly thirty parked within 0.01 of it.

All of that is true, and none of it is the cause. The direct measurement in
section 3 shows the threshold counts do not move while the solver cycles.

---

## 3. The measurement that identified the cause

Instead of examining aggregate residual norms, the **sequence** of residuals inside
a single failing time step was extracted:

```
9.34885e-05
9.35687e-05
9.34885e-05
9.35687e-05
...  (one hundred repetitions)
max_iter reached with residual 9.35687e-05 > tol 1e-06 - rejecting the step
```

The residual alternates between two values, **identical to every digit**, for two
hundred iterations. This is not slow convergence and it is not a singular matrix.
It is a **period-two limit cycle**, and a period-two limit cycle in Newton's method
is the signature of a residual that is continuous but whose slope is discontinuous.

Writing the iteration as

$$\mathbf{u}_{k+1} = \mathbf{u}_k - \mathbf{K}(\mathbf{u}_k)^{-1}\mathbf{R}(\mathbf{u}_k)$$

the tangent matrix $\mathbf{K}$ is the derivative of the residual $\mathbf{R}$. If
$\mathbf{R}$ has a kink — a point where the left and right derivatives differ — then
the tangent computed on one side does not describe the residual on the other. A step
taken from state $A$ crosses the kink and lands at $B$; the step from $B$ crosses
back to $A$; and the iteration alternates forever without the residual decreasing.

Three further comparisons narrowed it:

| Configuration | Residual behaviour |
|---|---|
| All fixes active | two-cycle |
| Collagen floor removed | two-cycle |
| Growth bounds **disabled** | two-cycle |
| **Pre-stretch removed** | wanders, converges — **no cycle** |

So the cycle was not caused by any recent change, and it requires the pre-stretched
state to appear.

The **relative** amplitude of the cycle was nearly constant across three runs whose
residuals spanned two orders of magnitude:

$$\frac{|R_A - R_B|}{R} = 8.6\times10^{-4},\quad 4.9\times10^{-4},\quad 4.1\times10^{-4}$$

A proportional signature, which argues against any fixed absolute perturbation.

### Instrumenting the discontinuities

Three constructs in the local solver are not differentiable. Rather than choose
between them by argument, the code was instrumented to count, at every Newton
iteration, how many integration points sit in each branch and how close each is to
switching. A period-two cycle driven by one of them will show **that count
alternating** while the others hold steady.

The result was unambiguous:

```
iter 13   R=8.98273e-05   signflip=38465   degen=0   bandgap=3.48e-06
iter 14   R=8.98608e-05   signflip=38461   degen=0   bandgap=4.13e-06
iter 15   R=8.98273e-05   signflip=38465   degen=0   bandgap=3.48e-06
iter 16   R=8.98608e-05   signflip=38461   degen=0   bandgap=4.13e-06
```

The eigenvector sign count alternates between 38 465 and 38 461 — **exactly four
integration points switching** — in perfect lockstep with the residual. The
eigenvalue-degeneracy branch never fires at all. The threshold counts do not move.

---

## 4. Root cause: a discontinuous eigenvector sign convention

### 4.1 The physical model

Collagen fibres reorient toward the direction of greatest stretch. The fibre
direction $\mathbf{a}_0$ obeys

$$\dot{\mathbf{a}}_0 = \frac{2\pi\,\dot{\phi}_+}{\tau_\omega}\,
\lambda_{\max}\,\bigl(\mathbf{I} - \mathbf{a}_0\otimes\mathbf{a}_0\bigr)\,
\mathbf{v}_{\max}$$

where $\dot{\phi}_+$ is the positive part of the collagen production rate,
$\tau_\omega$ a time constant, and $\mathbf{v}_{\max}$ the eigenvector of the elastic
right Cauchy–Green tensor $\mathbf{C}^e$ belonging to its largest eigenvalue
$\lambda_{\max}$:

$$\mathbf{C}^e\,\mathbf{v}_{\max} = \lambda_{\max}\,\mathbf{v}_{\max}$$

The projector $(\mathbf{I} - \mathbf{a}_0\otimes\mathbf{a}_0)$ removes any component
along $\mathbf{a}_0$, so the fibre rotates without changing length.

### 4.2 Where the discontinuity enters

An eigenvector is determined only up to sign: if $\mathbf{v}$ satisfies the
eigenvalue problem then so does $-\mathbf{v}$. The numerical library returns one of
the two arbitrarily. The code resolved this by choosing the sign that points the
same way as the current fibre:

```cpp
if (a0.dot(vectormax) < 0) { vectormax = -vectormax; }
```

which is
$$\mathbf{v}_{\max} \leftarrow \operatorname{sign}(s)\,\mathbf{v}_{\max},
\qquad s \equiv \mathbf{a}_0\cdot\mathbf{v}_{\max}$$

Now consider what happens as $s$ passes through zero. For $s>0$,

$$\dot{\mathbf{a}}_0 = k\,(\mathbf{I}-\mathbf{a}_0\otimes\mathbf{a}_0)\,\mathbf{v}_{\max}$$

and for $s<0$,

$$\dot{\mathbf{a}}_0 = -k\,(\mathbf{I}-\mathbf{a}_0\otimes\mathbf{a}_0)\,\mathbf{v}_{\max}$$

so the jump across $s=0$ is

$$\Delta\dot{\mathbf{a}}_0 = 2k\,(\mathbf{I}-\mathbf{a}_0\otimes\mathbf{a}_0)\,\mathbf{v}_{\max}$$

Writing $\theta$ for the angle between $\mathbf{a}_0$ and $\mathbf{v}_{\max}$, the
magnitude of the projected vector is

$$\bigl\|(\mathbf{I}-\mathbf{a}_0\otimes\mathbf{a}_0)\,\mathbf{v}_{\max}\bigr\| = |\sin\theta|$$

and $s = \cos\theta$. The discontinuity therefore occurs at $\theta = \pi/2$, where
$|\sin\theta| = 1$ — its **maximum**.

> **The jump is largest exactly where it occurs.** The sign convention places its
> worst discontinuity precisely where the driving term is strongest.

The ambiguity is genuine, not numerical. At $\theta = \pi/2$ the fibre is
perpendicular to the principal stretch axis, both ends of that axis are equidistant,
and there is no basis for preferring either.

### 4.3 Consequence for the residual

The fibre direction enters the stress through the structure tensor

$$\mathbf{A}_0 = \kappa\,\mathbf{I} + (1-3\kappa)\,\mathbf{a}_0\otimes\mathbf{a}_0$$

and through the fibre strain invariant $I_4^e = \mathbf{a}_0\cdot\mathbf{C}^e\mathbf{a}_0$.
A jump in $\dot{\mathbf{a}}_0$ therefore produces a jump in the derivative of the
stress with respect to the displacement — that is, the residual is continuous but
its slope is not, which is exactly the condition that makes Newton's method cycle.

Measured: **four integration points** straddle $s=0$, with the closest at
$|s| = 1.37\times10^{-3}$.

---

## 5. The fix, and why it is safe

### 5.1 The replacement

Replace the discontinuous sign function by a smooth approximation:

$$\mathbf{v}_{\max} \leftarrow \tanh\!\left(\frac{s}{\varepsilon}\right)\mathbf{v}_{\max},
\qquad \varepsilon = 0.05$$

Three properties make this the right choice:

1. **It is unchanged away from the ambiguity.** For $|s|\gg\varepsilon$,
   $\tanh(s/\varepsilon)\to\pm1$, so the behaviour is identical to the sign
   function. With $\varepsilon = 0.05$ and $|s| = 0.15$, $\tanh(3) = 0.995$ — a
   deviation of five parts in a thousand.

2. **It is infinitely differentiable.** The residual becomes smooth in this term, so
   the tangent describes it on both sides and the cycle cannot form.

3. **It vanishes at the ambiguous orientation.** $\tanh(0)=0$, so the rotation rate
   goes to zero exactly where there is no preferred direction to rotate toward. This
   is the physically correct limit rather than an arbitrary regularisation.

The sign is preserved away from $s=0$, so the perpendicular orientation remains an
**unstable** equilibrium: fibres still rotate away from it, merely more slowly as
they approach it.

Exposed as `WOUND_SIGNEPS`; setting it to zero restores the original hard switch,
which is what made the controlled comparison in section 6 possible.

### 5.2 Measured impact on the physics

Before trusting the change, the distribution of $|s| = |\mathbf{a}_0\cdot\mathbf{v}_{\max}|$
was measured, since the smoothing perturbs any point with $|s| \lesssim 3\varepsilon$.

| State | minimum $|s|$ | median | fraction damped below 0.95 | below 0.50 |
|---|---|---|---|---|
| Healthy, pre-stretched | **0.99992** | 1.0000 | **0.00 %** | **0.00 %** |
| Wounded, $t=0$ | 0.00004 | 1.0000 | 1.84 % | 0.77 % |
| Wounded, $t=8$ h | 0.00017 | 1.0000 | 2.28 % | 1.19 % |
| Wounded, $t=16$ h | 0.00005 | 1.0000 | **2.65 %** | 1.48 % |

In healthy pre-stretched tissue the minimum is 0.99992 — every fibre is essentially
perfectly aligned with the direction of greatest stretch, because the fibres are
axial and the largest pre-stretch (1.098) is axial. **The change is exactly zero
there**, so the homeostatic state cannot be disturbed.

In the wound it affects 2–3 % of points, all of them genuinely near the ambiguous
orientation, and the affected fraction grows with time as the wound deforms — which
is precisely the population that should be damped.

An earlier estimate of 15 % assumed randomly oriented vectors; the measurement shows
the distribution is strongly bimodal, with a median of exactly 1.0 and a thin tail.

---

## 6. Verification by controlled comparison

Two runs were launched from the same executable, the same mesh, the same settling
duration and the same wound. The **only** difference was `WOUND_SIGNEPS`.

| | Smooth sign ($\varepsilon = 0.05$) | Original hard switch |
|---|---|---|
| Simulated hours after 1 h 14 min | **38.6** | **0.7** |
| Rejected steps | **0** | 0 |
| Phase | healing | still in the start-up ramp |

Residual sequences from the same wall-clock moment:

```
smooth:   1 -> 2.0358e-05 -> 1.29319e-06 -> 3.04371e-07     converged below tolerance
original: 1 -> 1.89884e-05 -> 1.88267e-05 -> 1.88316e-05
                           -> 1.88278e-05 -> 1.88316e-05     two-cycle, stuck
```

The smooth version drives the residual to $3.0\times10^{-7}$, through the tolerance
of $10^{-6}$. The original locks into the cycle at $1.88\times10^{-5}$.

**A 55-fold difference in progress from one change of a sign convention.**

---

## 7. Secondary defect: wound placement in the wrong configuration

### 7.1 The defect

The simulation runs in two phases. First the mesh is stretched into the living,
taut configuration; then the puncture is made. Wound membership was tested on the
**deformed** coordinates, against limits obtained by mapping the reference limits
through the pre-stretch.

This produced four separate errors over the course of the work:

1. The axial centre $z = 5.0$ is the *reference* mid-length, but the settled mesh
   spans $z\in[0, 10.98]$, so testing deformed coordinates against 5.0 placed the
   wound at reference $z = 4.554$ — off the refined region entirely (549 nodes
   instead of 1 536).
2. The outer radius limit was the *reference* value 5.4 while the settled outer
   surface sits at 5.558, so every node beyond $r = 5.4$ failed the test and the
   track pierced only the inner 55 % of the wall.
3. The released patch of boundary tested *reference* coordinates against the same
   centre, leaving patch and wound 0.45 mm apart.
4. When the pre-stretch was disabled for a diagnostic, the limits were still mapped
   through it, placing the wound centre at 5.49 while the tissue sat at 5.0 — an
   offset of 0.49 mm against a wound radius of 0.25 mm. Those runs seeded almost no
   wound (inflammation 0.061 instead of 0.330) and appeared to succeed for that
   reason, which invalidated a conclusion drawn from them.

### 7.2 The fix

Wound membership is now decided **once, on the undeformed mesh**, exactly the way
the boundary nodes are identified. Node and integration-point identifiers are
tagged, a severity is stored per identifier, and every downstream use — seeding, the
released patch, the gradual-injury ramp — reads those stored values. No downstream
code recomputes geometry.

The severity of the needle track at a reference position $\mathbf{X}$ is

$$
\text{sev}(\mathbf{X}) =
\begin{cases}
\dfrac{1}{2}\left[1 - \tanh\!\left(\dfrac{d - r_w}{w}\right)\right]
& \text{if } X_1 < 0 \text{ and } r_{\text{in}} \le r \le r_{\text{out}}\\[2ex]
0 & \text{otherwise}
\end{cases}
$$

with

$$r = \sqrt{X_1^2 + X_2^2},\qquad
d = \sqrt{(X_2 - y_c)^2 + (X_3 - z_c)^2}$$

Here $d$ is the distance from the track axis, measured in the plane normal to it,
and $r$ is the radius used to test membership in the wall. **No pre-stretch factor
appears anywhere.** Identifiers are therefore invariant under any change of
pre-stretch magnitude, settling duration, or boundary condition.

### 7.3 Result

| | Deformed-coordinate test | **Reference tagging** |
|---|---|---|
| Wound nodes | 553 | **701** |
| Wound integration points | 12 382 | **15 684** |
| Radial span | 5.301 – 5.448 mm | **5.000 – 5.400 mm** |
| **Thickness pierced** | **41.8 %** | **99.97 %** |

The wound is now a genuine full-thickness puncture. A coverage report is printed at
start-up with an explicit warning below 90 % — the check that would have caught this
at the outset.

---

## 8. Secondary defect: the resistance to volume change vanished inside the wound

### 8.1 The model

The volumetric part of the strain energy is

$$\Psi_{\text{vol}}(J^e) = \frac{\phi\,p}{2}\,(J^e - 1)^2 \;-\; 2\,\phi\,k_0\ln J^e$$

where $J^e = \det(\mathbf{F}^e)$ is the elastic volume ratio, $\phi$ the collagen
fraction, $p = 0.3167$ MPa the penalty stiffness and $k_0 = 0.02$ MPa the
ground-substance stiffness. Differentiating,

$$\frac{\partial\Psi_{\text{vol}}}{\partial J^e} = \phi\,p\,(J^e-1) - \frac{2\,\phi\,k_0}{J^e}$$

The logarithm is a **barrier**: as $J^e \to 0^+$, $-\ln J^e \to +\infty$, so
crushing an element to zero volume costs infinite energy. It is the only thing in
the model preventing element inversion.

### 8.2 The defect

Every term carries $\phi$, including the barrier. Balancing the barrier against a
confining pressure $P$, in the strongly compressed regime $J^e \ll 1$ the barrier
dominates and

$$-\frac{2\phi k_0}{J^e} \approx -P
\qquad\Longrightarrow\qquad
\boxed{\;J^e \approx \frac{2\,\phi\,k_0}{P}\;}$$

**The equilibrium elastic volume is proportional to the collagen fraction.** The
barrier that prevents inversion is weakest exactly where it is most needed — in the
wound, where collagen has dropped by two orders of magnitude.

### 8.3 Why this removes elements from the system

Every stress term is pulled back to the reference configuration with a factor of the
plastic volume ratio $J^p$:

$$\mathbf{S} = J^p\,\mathbf{F}_g^{-1}\,\mathbf{S}^e\,\mathbf{F}_g^{-\mathsf{T}}$$

so as $J^p\to0$ an element contributes **no stress and no stiffness at all**. It
drops out of the assembled system, leaving a near-null space.

Measured at the worst point of a failing run:

| Time [h] | $\lambda^e_n$ | $\lambda^P_n$ | $\lambda_n = \lambda^e_n\lambda^P_n$ | $J^p$ |
|---|---|---|---|---|
| 0 | 0.175 | 0.959 | 0.168 | 0.969 |
| 16 | 0.167 | 0.650 | **0.109** | 0.694 |

The minimum of $J^p$ over the mesh fell from 0.938 to **0.133**. The elastic stretch
is almost unchanged while both the geometry and the growth collapse together — the
growth law chasing a geometric collapse it cannot catch.

### 8.4 The fix

The pure penalty term is made independent of collagen, because a tissue resists
compression through its **water content**, not its collagen. A wound fills with
blood, fibrin and granulation tissue, all of which are roughly 90 % water, so the
resistance to volume change barely changes while the shear and fibre stiffness
genuinely do collapse:

$$\frac{\partial\Psi_{\text{vol}}}{\partial J^e} = p\,(J^e-1) - \frac{2\,\phi\,k_0}{J^e}$$

The factor $\phi$ is retained on the $k_0$ term because it is paired with
$\phi k_0\mathbf{I}$ in the passive stress, and the two must cancel at
$\mathbf{C}^e = \mathbf{I}$ for the reference state to be stress-free. Checking that:
at $\mathbf{C}^e = \mathbf{I}$ the fibre strain measure is

$$E = \kappa I_1^e + (1-3\kappa)I_4^e - 1 = 3\kappa + (1-3\kappa) - 1 = 0$$

so the fibre terms vanish; the ground-substance term contributes $+\phi k_0\mathbf{I}$
and the volumetric term $\tfrac12(0 - 2\phi k_0)\mathbf{I} = -\phi k_0\mathbf{I}$.
They cancel for **any** $\phi$.

Exposed as `WOUND_VOLPHI`. Measured effect at the puncture: minimum elastic volume
ratio improved from 0.2231 to 0.3038, and the count of severely collapsed points
went from 15 to **zero**.

---

## 9. Secondary defect: unbounded growth

The plastic stretch obeys

$$\dot{\lambda}^P_i = \frac{\dot{\phi}_+}{\tau_{\lambda P}}\;\beta(\lambda^e_i),
\qquad
\beta(\lambda) = \text{softplus}(\lambda - \lambda_{\text{hi}}) - \text{softplus}(\lambda_{\text{lo}} - \lambda)$$

with a tolerance band $[\lambda_{\text{lo}},\lambda_{\text{hi}}] = [0.85, 1.15]$. Two
problems:

1. $\beta$ is **linear far from the band**, so the driving grows without limit. With
   $\lambda^e_n = 0.17$, five times below the lower limit, $\beta = -0.68$ and
   growing.
2. **$\lambda^P$ had no bound anywhere in the code.**

Two remedies were added, both folded into $\beta$ at the single place it is computed
so that the rate and all its derivatives stay mutually consistent:

**Saturation of the driving** — bounds the rate:
$$\beta \leftarrow \beta_{\max}\tanh\!\left(\frac{\beta}{\beta_{\max}}\right),
\qquad \beta_{\max} = \lambda_{\text{hi}} - \lambda_{\text{lo}}$$

**A one-sided gate on the state** — bounds $\lambda^P$ itself:
$$g(\lambda^P,\beta) =
\begin{cases}
\tfrac12\left[1+\tanh\!\left(\dfrac{\lambda^P - \lambda^P_{\text{lo}}}{w}\right)\right] & \beta<0\\[2ex]
\tfrac12\left[1+\tanh\!\left(\dfrac{\lambda^P_{\text{hi}} - \lambda^P}{w}\right)\right] & \beta>0
\end{cases}$$

The gate is one-sided so it damps only motion heading *into* a bound; $\lambda^P$ can
always recover toward the interior. Defaults $[0.5, 2.0]$, exposed as
`WOUND_LAMP_LO` / `WOUND_LAMP_HI` / `WOUND_BANDCAP`.

Measured effect: the minimum $\lambda^P_n$ at the first healing save was held at
0.957 with the bounds active against 0.911 without.

**This removed the catastrophic divergences** — residuals fell from $1.2\times10^{12}$
to $10^{-4}$ — but did **not** remove the grinding, because it addressed the wound
rim while the residual lives further out. It is a real fix for a real pathology, but
it was not the cause of the stall.

---

## 10. Smaller defects

### 10.1 Integration-point position accumulated across an element

In the element routine, the interpolated reference position was declared and zeroed
**outside** the loop over integration points while being accumulated inside it:

```cpp
Vector3d X; X.setZero();          // outside the loop
for (int ip = 0; ip < IP_size; ip++) {
    ...
    X += node_X[ni]*R[ni];        // inside
```

By the fourth point of a tetrahedron it held the sum of four positions, roughly four
times a real coordinate. Nothing consumed it — the local solver accepts it but never
uses it — so there was no physical effect, but it made any per-point position
reporting meaningless. Fixed by zeroing per point.

### 10.2 Element size measured over an unrepresentative sample

The mean element edge was averaged over the **first 4 000 elements**, reporting
0.0886 mm against a true global mean of 0.2655 mm — a factor of three. It landed
near the correct local value (0.0587 mm) only by an accident of element ordering,
and the under-resolution warning that compared against it could never fire
correctly.

Now both a global mean and a mean local to the wound are computed and printed, and
the warning compares against the local value.

### 10.3 Quadrature rule for ten-node tetrahedra is wrong

`LineQuadriIPTetQuadratic` is **character-for-character identical** to the linear
four-point rule. For a quadratic tetrahedron this is degree-two integration applied
to degree-four mass terms: the $10\times10$ element matrix $R_iR_j$ is assembled from
four rank-one contributions, so it has rank at most four and carries **six spurious
zero-energy modes per element**.

This is the same defect that was previously found and fixed for linear tetrahedra —
the comment there records a single centroid point giving rank one, three zero-energy
modes, and concentration increments of order 100 on a field whose physical value is 1.

**Ten-node tetrahedra are therefore unusable until this is replaced** with a
degree-four rule. Not fixed here; recorded as a live trap.

---

## 11. Physical findings that are not defects

### 11.1 Healthy tissue drifts away from its homeostatic state during settling

The collagen source term is

$$\dot\phi = \frac{\left(p_\phi + p_{\phi c}\dfrac{c}{K_{\phi c}+c} + p_{\phi\theta}H\right)\rho}{K_{\phi\rho}+\phi}
\;-\;\bigl(d_\phi + c\,\rho\,d_{\phi\rho c}\bigr)\phi$$

The parameter $K_{\phi\rho}$ is chosen so that $\dot\phi = 0$ exactly at the
homeostatic state $(\alpha,\rho,c,\phi) = (0,1,1,1)$ **and** $H = 1/2$.

But the mechanical sensing function

$$H(\theta^e) = \frac{1}{1+\exp\!\left[-\gamma(\theta^e - \vartheta)\right]},
\qquad \vartheta = 1.136,\; \gamma = 10$$

only equals $1/2$ where the areal stretch $\theta^e$ is exactly $\vartheta$ — and the
discrete solution of the pre-stretch problem does not reproduce the prescribed
affine map exactly. Measured: $\theta^e \in [1.1254, 1.1462]$, a spread of
$\delta\theta = \pm0.010$ about the target.

The sensitivity is

$$\frac{\mathrm{d}H}{\mathrm{d}\theta^e} = \gamma H(1-H) = 10\times\tfrac12\times\tfrac12 = 2.5$$

giving $\delta H = 2.5\times0.010 = \pm0.025$. Measured range of $H$:
$[0.4736, 0.5256]$, i.e. $\pm0.026$ — agreement to within 4 %.

The collagen rate then acquires

$$\frac{\partial\dot\phi}{\partial H} = \frac{p_{\phi\theta}\,\rho}{K_{\phi\rho}+\phi}
= \frac{4.6326\times10^{-3}}{1.7027} = 2.72\times10^{-3}$$

so

$$\dot\phi \approx 2.72\times10^{-3}\times0.026 = 7.1\times10^{-5}\ \text{per hour}$$

Over 100 hours of settling this predicts $\Delta\phi \approx 7.1\times10^{-3}$.
**Measured spread after 100 hours: $\pm5.9\times10^{-3}$** — the prediction lands
within 20 %.

The drift is therefore driven by discretisation error in the areal stretch, and its
rate is proportional to that error. Practical consequence: **use short settling.**
Over the full 672-hour target it would extrapolate to roughly 5 % collagen drift in
healthy tissue, independent of the wound. Reducing it requires reducing the
discretisation error in $\theta^e$.

### 11.2 The wound cannot be closed by contraction at physiological force

The active stress is

$$\mathbf{S}_{\text{act}} = \frac{J^p\,t_{\text{act}}\,\phi}{\operatorname{tr}\mathbf{A}\,\bigl(K_t^2+\phi^2\bigr)}\,\mathbf{A}_0,
\qquad
t_{\text{act}} = \left(t_\rho + t_{\rho c}\frac{c}{K_{tc}+c}\right)\rho$$

The factor $\phi/(K_t^2+\phi^2)$ is maximised at $\phi = K_t = 0.2$, where it equals
$1/(2K_t) = 2.5$. With $c = 1.2$, $\rho = 0.9$, $t_\rho = 1.28571\times10^{-3}$,
$t_{\rho c} = 4.22447\times10^{-3}$ and $K_{tc} = 0.1$:

$$t_{\text{act}} = \left(1.28571\times10^{-3} + 4.22447\times10^{-3}\times\frac{1.2}{1.3}\right)\times0.9
= 4.667\times10^{-3}\ \text{MPa}$$

$$\|\mathbf{S}_{\text{act}}\|_{\text{peak}} = 4.667\times10^{-3}\times2.5
= \mathbf{1.167\times10^{-2}\ \text{MPa}}$$

Against this, the fibre stress in the surrounding pre-stretched tissue. With
$\Psi = \dfrac{k_f}{2k_2}\exp\!\left(k_2E^2\right)$ and
$E = \kappa I_1^e + (1-3\kappa)I_4^e - 1$, at
$\boldsymbol{\lambda} = (1.098, 1.035, 0.880)$ we have $I_1^e = 3.0512$,
$I_4^e = 1.2056$, and with $\kappa = 0.1$:

$$E = 0.30512 + 0.84392 - 1 = 0.14904$$
$$\Psi = \frac{40}{0.096}\exp\!\left(0.048\times0.14904^2\right) = 417.1$$
$$\Psi_4 = 2k_2(1-3\kappa)E\,\Psi = 2\times0.048\times0.7\times0.14904\times417.1
= \mathbf{4.18\ \text{MPa}}$$

$$\boxed{\;\frac{\text{fibre stress}}{\text{peak active traction}}
= \frac{4.18}{1.167\times10^{-2}} \approx 360\;}$$

A sweep confirmed this directly. Comparing at **identical simulated times** (the
first comparison attempted was confounded by unequal elapsed time and had to be
discarded):

| $t_\rho$ multiplier | change in cross-section radius at $t=4$ h |
|---|---|
| $\times1$ | $+0.28\%$ |
| $\times10$ | $+0.22\%$ |
| $\times100$ | $-0.20\%$ |
| $\times1000$ | $-2.37\%$ |

Monotone, with the sign changing between $\times10$ and $\times100$, and only
$\times1000$ contracting persistently.

For scale: fibroblast traction at tissue level is of order 1–10 kPa. The calibrated
value gives 11.7 kPa — physiologically correct. The $\times1000$ case gives 11.7 MPa,
comparable to the collagen stiffness itself.

**Interpretation.** Closure in this model means the defect *fills* with fibroblasts
and collagen, not that its edges are pulled together. Myofibroblast traction
contracts the wound somewhat but never apposes the edges. Measured against filling,
the model works (section 13); the cross-section radius is secondary information.

### 11.3 Working envelope of the solver in active traction

Deliberate limit-finding, all with the sign fix and the full-thickness wound.
The table below reports each run over its **full** duration, not over the first few
hours of simulated time:

| $t_\rho$ | peak traction | simulated time reached | rejected steps | rate at 18.5 wall-hours |
|---|---|---|---|---|
| $\times1$ | 11.7 kPa | 669.6 h (complete, 15:34) | 0 | — |
| $\times10$ | 117 kPa | 669.6 h (complete, 19:20) | 0 | 26 simulated hours per wall-hour |
| $\times100$ | 1.17 MPa | 118.6 h (stopped at 20:12) | 51 | 1 simulated hour per wall-hour |
| $\times1000$ | 11.7 MPa | 12 h (stopped) | 5 | ~20× slower than $\times1$ |

**The clean envelope is $\times1$ to $\times10$, not through $\times100$.** An earlier
version of this section reported $\times100$ as clean; that judgement was made after
only a few hours of simulated time, before the run had reached the phase where
collagen deposition and plastic growth interact strongly. Over the full run the
$\times100$ case degrades badly: the time step falls from $\Delta t = 0.2$ h to
$\Delta t = 0.008$ h — a factor of 25 — and 51 steps were rejected and retried. At
that rate it advanced one hour of simulated time per hour of wall-clock time, against
26 for the $\times10$ case. It was stopped after 20 hours of wall-clock time having
covered 118.6 h of the 672 h target; at the observed rate four weeks would have taken
roughly 550 further wall-clock hours.

Neither extreme case fails outright. Both keep converging, only expensively — a
softer failure than the pre-fix behaviour, where runs stopped altogether.

Its limit cycle is a **different mechanism** from the sign convention: relative
amplitude 1.5 % against 0.04–0.09 %, and several branch margins move together rather
than one isolating itself. Should calibrated contracture data ever land above
$\times10$, that requires a fresh diagnosis.

For reference, fibroblast traction at the tissue level is of order 1–10 kPa, so the
calibrated $\times1$ value is the physically correct one. The $\times100$ and
$\times1000$ cases are well past physiological plausibility — $\times1000$ is
comparable to the collagen stiffness $k_f = 40$ MPa itself.

---

## 12. Diagnostics added

These are the tools that made the diagnosis possible, and they remain available.

**Residual localisation** (`WOUND_RESLOC`, default: iteration 20). Prints the twenty
degrees of freedom carrying the largest residual, with field type, node number,
coordinates and distance from the needle axis. This established that the failure is
entirely mechanical and identified where it sits.

**Non-differentiable branch counters.** At every Newton iteration, counts how many
integration points sit in each non-smooth branch and how close each is to switching:
the smallest $|\mathbf{a}_0\cdot\mathbf{v}_{\max}|$, the smallest eigenvalue
separation, and the smallest distance to a growth threshold. **A period-two cycle
shows one count alternating while the others hold steady** — this is what named the
cause.

**Elastic volume ratio reporting** (`scripts/check_detFe.py`). Reconstructs
$\mathbf{C}^e = 2\mathbf{E}+\mathbf{I}$ exactly from the saved strain field and
reports the minimum and median of $J^e$, the count of collapsed points, and the
plastic volume ratio $J^p$ with the plastic stretches. Watching $J^e$ alone had
concealed the collapse: it fell only 0.218 → 0.154 while $J^p$ fell 0.938 → 0.133.

**Wound filling report** (`scripts/wound_fill.py`). Mean and minimum of each field
over the wound core against time. This is the measure that matches the definition of
closure.

**Wound coverage report at start-up.** Prints the number of tagged nodes and
integration points, the radial span, and the percentage of wall thickness pierced,
with an explicit warning below 90 %.

---

## 13. State of the simulation at the end of this work

A run with the pre-stretch applied, full deformable mechanics, all four fields and a
correctly seeded full-thickness wound:

| Simulated time [h] | $\rho$ (fibroblasts) | $\phi$ (collagen) | $c$ (cytokine) | $\alpha$ (inflammation) |
|---|---|---|---|---|
| 0 | 0.455 | 0.412 | 0.840 | 0.362 |
| 74 | 0.931 | 0.579 | 1.276 | 0.013 |
| 148 | 0.995 | 0.681 | 1.187 | 0.003 |
| **222** | **1.003** | **0.744** | 1.125 | **0.0007** |

This is the expected sequence, in order: inflammation rises and clears; the cytokine
rises, peaks and resolves once the inflammatory signal is gone; fibroblasts
repopulate the defect; collagen is deposited behind them. The minimum of $\rho$ over
the wound core is 0.9998 against a mean of 1.003, so the filling is uniform rather
than an average concealing an empty centre.

Fourteen of fifteen healing checks pass. The one failure is a small negative
excursion of the inflammatory field, $-1.5\times10^{-3}$, which has been present
throughout and diminishes with time. It is a positivity property of the transport
discretisation, and the inflammatory equation is linear and uncoupled from the other
species — so it can be split off and solved separately, where a positivity-preserving
scheme can be applied to it alone.

**Throughput before and after:**

| | Before | After |
|---|---|---|
| Furthest simulated time | 16.8–21.6 h, then stall | **222 h and continuing** |
| Rejected steps | 5–23 per run | **0–1 per run** |
| Rate | 0.2–5.9 simulated h per wall h | **27–59 simulated h per wall h** |
| Projected time for 672 h | not reachable | **roughly one day** |

---

## 14. Files changed

| File | Change |
|---|---|
| `src/local_solver.cpp` | smooth eigenvector sign convention (`WOUND_SIGNEPS`); saturation and one-sided gate on plastic growth (`WOUND_BANDCAP`, `WOUND_LAMP_LO/HI`); branch-counting instrumentation |
| `src/wound.cpp` | collagen-independent volumetric penalty (`WOUND_VOLPHI`); optional floor on the ground-substance stiffness (`WOUND_PHIFLOOR`); fix for the accumulating integration-point position |
| `src/solver.cpp` | residual localisation (`WOUND_RESLOC`); optional decoupling of the displacement and transport step limiters (`WOUND_LINESEARCH`); branch report hooks |
| `src/cydindrical_dura_single_tissue_multiwound.cpp` | wound tagging on the reference mesh with coverage report; free patch moved to reference coordinates; corrected element-size measurement; gradual-injury ramp (`WOUND_SEVRAMP`); switches for freezing displacements and disabling pre-stretch |
| `include/local_solver.h` | declarations for the branch instrumentation |
| `scripts/check_detFe.py` | elastic and plastic volume ratio reporting |
| `scripts/wound_fill.py` | wound filling report (new) |

### Environment switches

| Switch | Default | Meaning |
|---|---|---|
| `WOUND_SIGNEPS` | 0.05 | width of the smooth eigenvector sign; 0 restores the hard switch |
| `WOUND_VOLPHI` | 1 | 0 makes the volumetric penalty independent of collagen |
| `WOUND_PHIFLOOR` | 0 | floor on the collagen fraction seen by the ground-substance stiffness |
| `WOUND_BANDCAP` | 1.0 | saturation of the growth driving term, in band widths |
| `WOUND_LAMP_LO` / `HI` | 0.5 / 2.0 | bounds on the plastic stretch |
| `WOUND_SEVRAMP` | 0 | number of stages over which the injury is applied |
| `WOUND_RESLOC` | 20 | iteration at which to report the largest residuals |
| `WOUND_FREEZEX` | unset | pin every nodal displacement (transport only) |
| `WOUND_NOPRESTRETCH` | unset | skip the pre-stretch (diagnostic only — the homeostatic state is then inconsistent) |

---

## 15. What remains

1. **The negative excursion of the inflammatory field**, $-1.5\times10^{-3}$. Best
   addressed by solving that equation separately, since it is linear and receives no
   coupling from the other species.
2. **Quadrature for ten-node tetrahedra** (section 10.3) — must be replaced before
   quadratic elements can be used at all.
3. **Homeostatic drift** (section 11.1) — bounded by using short settling, but the
   underlying discretisation error remains.
4. **The limit cycle above $\times100$ traction** (section 11.3) — a different
   mechanism, undiagnosed, only relevant if calibrated contracture forces prove that
   large.
5. **Assembly parallelism.** The element loop is parallel but the assembly sits
   inside a critical section, so throughput saturates near twelve threads. Per-thread
   buffers merged afterwards would be worth a factor of two to four.
