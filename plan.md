This is multi-physics FEA problem to solve for wound healing mechanics.
Following the species of interest
alpha: pro-inflammatory cytokine signals. This code did not yet include them. Setting them up and running the entire code and verifying is one the goals.
c: cytokines. TGF Beta
rho: fibroblast
phi: collagen. 
mechanics is also included. We solve for plastic (growth), collagen dispersion and orientation as a function of time as well.


Equations of biological species:
## Pro-inflammatory signal (ToDo 1: Need to implement this in the FEA code)
$$
\dot{\alpha}
+
\nabla\cdot Q_\alpha = s_\alpha
$$
$$
Q_\alpha=-D_\alpha\nabla\alpha
$$

$$
s_\alpha=-d_\alpha\alpha
$$

## Cytokine / TGF-$\beta 1$ (Most likely nothing to change in the code)

$$
\dot{c} +
\nabla\cdot Q_c = s_c
$$

$$
Q_c=-D_c\nabla c
$$
$$
s_c
=
p_{c,\alpha}\alpha
+
\left(
p_{c,\rho}c+p_{c,e}H(J^e)
\right)
\left(
\frac{\rho}{K_{c,c}+c}
\right)
-
d_c c
$$

## Fibroblast density (ToDO 2: I have modified the equation for diffusion but have not changed the code. )

$$
\dot{\rho}
=
\nabla\cdot Q_\rho+s_\rho
$$

$$
Q_\rho=-D_\rho(\phi,c)\nabla\rho
$$
$$P_{phi} = a \phi^5 + b \phi^4 + c \phi^3 + d \phi^2 + e \phi$$
$$C_{low} = 0.5\left(1 + \tanh\left(k_{cut}(\phi - 0.01)\right)\right)$$
$$C_{up} = 1 - \frac{1}{1 + e^{-500(\phi - 1)}}$$
$$D_{\rho} = \text{const} \cdot \frac{(1 \times 10^{-3} P_{phi})^2}{6} \cdot C_{up} \cdot C_{low} + 6.12 \times 10^{-5}$$

## Collagen density (Most likely unchanged from the code)
$$
\dot{\phi}
=
\left(
p_\phi
+
p_{\phi,c}\frac{c}{K_{\phi,c}+c}
+
p_{\phi,e}H(J^e)
\right)
\left(
\frac{\rho}{K_{\phi,\rho}+\phi}
\right)
-
\left(
d_\phi+c\rho d_{\phi,c}
\right)\phi
$$

Equations of Mechanics and micro-structures like plastic growth, collagen remodeling terms like dispersion and orientation are mostly unchanged and hence no need for any modifications to the code.

Implementation details:
0. Actually we have to push the code files to gihub. Then when editing the code, we can work on a new branch and do the editing there so that these files are saved. Github is already linked to this laptop. So, we just need to create a new repository and keep updating them. No need to push unnecessary files or results. just @src/, @include/, @scripts/, @meshing/ to the github.
1. We will always first modify the code here and then we can just push the code and necessary files to Negishi /scratch/negishi/athani/athani_wound_healing_simulations/dura_multi_wound_healing/ boost and Eigen libraries are available in scratch/negishi/athani/athani_wound_healing_simulations/ for testing if the code is actually working.  Files have to be synced between this folder and negishi folder and from this folder we need to sync them to github. No need to push results to the github. only code files. We can automatically ssh to negishi without any need for duo mobile security or microsoft authenticator. Github, current folder and the negishi must be synced after every major update. Once we are confident of the edits, I will compare them against the comsol results (just pdes for species with fixed mechanosensing term (H = 0.5) in comsol)

2. Make sure that the main file uses this parameter combination (symbols might be different) Use src/cydindrical_dura_single_tissue_multiwound.cpp as the main file for now (We only need to solve for the single wound now). The corresponding mesh file is located in @files/ folder. @cyl_dura_only_double_wound_2t_1_week is a sample results folder run using the src/cydindrical_dura_single_tissue_multiwound.cpp (may be different mesh file or the distance between two wounds is wrong).

| Parameter | Symbol | Units | | **GP prior** | **Posterior** median [5–95%] | **Median set** (sweep #4 case 0) | Data status | Dura data / provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pro-inflammatory decay | $d_\alpha$ | 1/h | **S** | `U[0.005, 0.02]` | 0.0128 [0.00607, 0.0191] | 0.0128 | No dura data | Pensalfini 0.01 (assumed $=d_c$) |
| Cytokine diffusion | $D_c$ | mm²/h | **S** | `logU[0.001, 0.10]` | **0.00930** [0.00151, 0.0351] | 0.00930 | Indirect constraint only | **Now a 0-D GP dim** — it enters the log-leak ($k_u\propto D$). LAB Sohutskay/Harbin 0.01208 – Tepole 0.10; floor to 0.001 (lab's COMSOL setting; TGF-β1 binds ECM ⇒ effective $D\ll$ free). *(base: PDE-only `U[0.04,0.10]`)* |
| Cytokine decay | $d_c$ | 1/h | **S** | `logU[0.001, 0.015]` | 0.00386 [0.00120, 0.0122] | 0.00386 | Indirect constraint only | LAB Sohutskay 0.001 – Tepole/Pensalfini 0.01, to the dura ceiling 0.015 (central 0.006-0.010). *(base: `U[0.004,0.015]`)* |
| Cytokine production from $\alpha$ | $p_{c,\alpha}$ | 1/h | **S** | `U[0.05, 0.5]` | **0.208** [0.119, 0.349] | 0.208 | No data | Cannot be derived from equilibrium ($\alpha_h=0$, so the term vanishes). Pensalfini 0.23 "selected to match peak in c". *Sweep: timeline wants 0.12-0.21.* |
| Cytokine production (baseline) | $p_{c,\rho}$ | 1/h | **D** | $= d_c(K_{c,c}{+}1)/(1{+}\tfrac12 r_{c,e})$ | *derived* | 0.00695 | Indirect only | Back-solved from the sampled $K_{c,c}$ — same constraint, better conditioned, and keeps $K_{c,c}>0$ automatic. |
| Cytokine production (stretch) | $p_{c,e}$ | 1/h | **F** | `logU[0.01, 20]` × $p_{c,\rho}$ | **FIXED** ratio 0.447 (screened out) | 0.00311  *(ratio 0.447)* | **No usable data** | Fong's 3.4× excluded with Fong. Group's 3.33 doesn't transfer (that ratio carries units of $c$, and their $c$ is wound-peak-scaled). Timeline silent. See [[jacho-mechanoresponsive-2022]]. |
| Cytokine saturation | $K_{c,c}$ | — | **S** | `logU[0.05, 50]` | 1.20 [0.0735, 29.9] | 1.20 | No data | Sampled in place of $p_{c,\rho}$; Pensalfini's 0.5 for scale. NOT the group's c-scale artefact; widened for headroom. *(base: `logU[0.1,10]`)* |
| Fibroblast diffusivity | $D_\rho$ | mm²/h | **—** | *PDE stage:* $D_\rho(\phi)$, Harbin | — | — *(PDE stage)* | Direct (in vitro) | In vitro 0.001-0.009 mm²/h (rabbit, minipig, human on collagen); scaffold ≤0.018; in vivo ≤0.17. ⚠️ $P_\phi$ coefficients not in [[harbin-computational-2023]] — provenance unverified. |
| Fibroblast chemotaxis | $D_{\rho,c}$ | mm⁵/(g·h) | **—** | neglected | — | — *(neglected)* | No data | Not required in this model. |
| Proliferation (baseline) | $p_\rho$ | 1/h | **S** | `logU[9e-4, 0.034]` | **0.0154** [0.00325, 0.0312] | 0.0154 | **Direct** (dura) | LAB min-max Harbin 9e-4 – Tepole/Sohutskay/Pensalfini 0.034; the dura growth curves ([[goldschmidt-effect-2016]], [[goldschmidt-new-2013]]) gave 0.014-0.021. *(base: `U[0.014,0.021]`)* |
| Proliferation (cytokine) | $p_{\rho,c}$ | 1/h | **S** | `logU[0.03, 17]` × $p_\rho$ | ratio 1.48 [0.0602, 12.4] | 0.0229  *(ratio 1.48)* | No data | LAB min-max Tepole 0.03 – Harbin 17 × $p_\rho$ (Pensalfini's $\Omega^b_\rho=5$; notes had assumed 2.5). *(base: `U[1,6]`)* |
| **Proliferation (mechano)** | $p_{\rho,e}$ | 1/h | **S** | **`logU[0.01, 1]` × $p_\rho$** | ratio 0.109 [0.0134, 0.785] | 0.00168  *(ratio 0.109)* | **Fong EXCLUDED** | Spans Pensalfini 0.01, Sohutskay 0.25, Harbin 0.5 — and the timeline-compatible band (p5-p95 = 0.011-0.949, n=25/20000). **Excludes** the Fong-implied 5-57: 0 of 6015 sets with ratio >2 matched the timings. ⚠️ **A modelling choice, not a measurement** — it asserts [[fong-mechanical-2003]] (neonatal rat *cranial*, calvarial *osteogenesis*, in-vivo strain 9.7%→0.1% by d60) does not transfer to adult spinal dural repair. |
| Fibroblast apoptosis | $d_\rho$ | 1/h | **S** | `logU[8.1e-4, 0.048]` | 0.00369 [0.000964, 0.0213] | 0.00369 | Indirect only | LAB min-max Harbin 8.1e-4 – Pensalfini 0.048 (growth-curve decomposition gave 0.004-0.007; no TUNEL/caspase-3). ⚠️ In vitro at $H{=}0.044$, not $H_h{=}1/2$. *(base: `U[0.004,0.007]`)* |
| Carrying capacity | $K_{\rho,\rho}$ | — | **D** | $\rho_h/(1-d_\rho/\text{prolif}_h)$ | *derived* | 1.16 | — | From $s_\rho=0$. Admissibility: must exceed $\rho_h$. *Sweep: 1.08-1.22 → still no overshoot capacity.* |
| Collagen production (baseline) | $p_\phi$ | 1/h | **S** | `logU[2e-4, 5e-3]` | 9.34e-4 [2.74e-4, 3.54e-3] | 9.34e-4 | No dura data | Group normalized: Sohutskay 2e-4 → Pensalfini 2e-3 (×$\rho_h$ rule, [[normalization-conventions]]), extended to 5e-3 so $K_{\phi,\rho}>0$ at the higher $d_\phi$. *(base: `logU[2e-4,2e-3]`)* |
| Collagen production (cytokine) | $p_{\phi,c}$ | 1/h | **F** | `logU[2e-4, 1e-2]` | **FIXED** (screened out) | 1.41e-3 | No dura data | Group normalized (Sohutskay 2e-4 – Pensalfini 5e-3), extended likewise. *(base: `logU[2e-4,5e-3]`)* |
| Collagen production (stretch) | $p_{\phi,e}$ | 1/h | **S** | `U[0, 10]` × $p_\phi$ | ratio 4.96 [0.662, 9.37] | 0.00464  *(ratio 4.96)* | No dura data | ⚠️ Do **not** fix from Col I: [[jacho-mechanoresponsive-2022]] states only ~13× at 12%, is cyclic **uniaxial** on **dermal** cells in a 3D gel ($\theta^e\approx1.06$, not 1.25), and is a 7-day **cumulative** response, not a rate. |
| Collagen degradation (baseline) | $d_\phi$ | 1/h | **S** | `logU[2.4e-4, 2.92e-3]` | **2.02e-3** [1.16e-3, 2.80e-3] ⬆ | 2.02e-3 | Indirect | LAB max Pensalfini 2.92e-3 (3.3× the dura ceiling) — **this is what makes `t_phi_90` reachable**. Dura basis: turnover 80-120 d ([[singh-regulation-2023]]), DuraGen resorption d33 ([[maeda-histopathological-2025]]), connective-tissue 1-2%/day ([[holwerda-impact-2022]]). *(base: `U[2.4e-4,8.8e-4]`)* |
| Collagen degradation (cytokine) | $d_{\phi,c}$ | 1/h | **F** | `U[8.8e-5, 4.85e-4]` | **FIXED** (screened out) | 2.87e-4 | No direct data | Harbin 8.81e-5; Tepole/Sohutskay/Pensalfini 4.85e-4 |
| Collagen saturation (by $\rho$) | $K_{\phi,\rho}$ | — | **D** | see derivation below | *derived* | 0.705 | — | From $\dot\phi=0$. **Must stay >0** — Harbin's GP returned −0.205 from his own constraint. |
| Collagen saturation (by $c$) | $K_{\phi,c}$ | — | **S** | **`logU[0.05, 50]`** | 1.08 [0.0728, 28.9] | 1.08 | No data | ⚠️ **NOT the group's 1e-4** — that is a c-scale artifact (their $c$ peaks at 1e-4; ours is $c_h=1$). At 1e-4 the switch is pinned open, so cytokine only *degrades* collagen. See [[normalization-conventions]]. *(base: `logU[0.1,10]`)* |
| Proliferation saturation (by $c$) | $K_{\rho,c}$ | — | **S** | `logU[0.05, 50]` | 1.31 [0.0751, 28.9] | 1.31 | No data | Pensalfini 10 (legitimate, $c_h{=}1$). Tepole/Harbin 1e-5 **excluded** — same c-scale artifact. *(base: `logU[0.1,10]`)* |
| **Mechanosensing stimulus** | $\theta^e$ | — | **—** | *field, from mechanics* | — | — *(field)* | — | **In-plane AREAL stretch** $\lambda_1\lambda_2=\|\mathrm{cof}F^e{\cdot}N\|$, **not** $\det F^e$. No definition in the 0-D model. Reference: 1.0 in-vitro control, 1.136 healthy, 1.21 = Fong's 10% equibiaxial. |
| **Mechanosensing midpoint** | $\vartheta^e$ | — | **F** | **1.136** | *fixed* | 1.136 | **Direct measurement** | Adult dura **areal** prestretch $1.098{\times}1.035$ ([[consolini-investigation-2024]]); neonatal 1.110. Sets $H_h=1/2$. ⚠️ Mouse **cranial**, ours is spinal. Was 2 — inherited from skin tissue expansion, wrong regime. |
| **Mechanosensing shape** | $\gamma_e$ | — | **F** | **10** (nominal) | *fixed* | 10 | **Unidentifiable** | One dura point, two unknowns → not identifiable; the timeline is also silent (at $p_{\rho,e}\lesssim1$ the shape barely matters). **And inert in the 0-D model** — $H$ is a constant there, so $\gamma_e$ only feeds the Fong diagnostic. Fixed to a moderate nominal; revisit with a sensitivity scan (≈[8,20]) at the PDE stage. $H(\vartheta^e)=1/2$ for every $\gamma_e$, so this touches no derived $K$. |
| **Mechanosensing at homeostasis** | $H(\theta^e_h)$ | — | **F** | **1/2** | *fixed* | 1/2 | By construction | Since $\vartheta^e\equiv\theta^e_h$. Was assumed ≈0. Used in every derivation below. |
| Mechanosensing in wound | $H_{wound}$ | — | **F** (prod.) / S (explore) | **0.5** for production & COMSOL; `U[0,1]` as a diagnostic | *fixed* | 0.5 | 0-D only | **Production/COMSOL runs use $H = 1/2$ everywhere** (no mechanics ⇒ H sits at its physiological midpoint in the wound too). Then $(0,1,1,1)$ is the exact fixed point and the wound returns to it — no frozen-H mismatch, physiological values held automatically. The `U[0,1]` sweep stays as an *exploration* (it showed $H_{wound}$ is not data-constrained: timeline wants 0.68–0.99 / stretched, vs Pensalfini's contracting skin — unresolved without mechanics). Replaced by $H(\theta^e)$ once mechanics is coupled. |
2. go through the scripts and check where should we modify the files in the @src/ folder and @include/ folder for Todos 1 and 2. Mostly we have to modify the files @src/wound.cpp,  @local_solver.cpp @src/solver.cpp for adding the new species pro-inflammatory cytokine $\alpha$ to this FEA solver. There are so many main files which are just for different cases in @src/ folder. we will only consider @src/cydindrical_dura_single_tissue_multiwound.cpp as the main file. we need to update the parameters of this file from the Median set (sweep #4 case 0) in the table above. some of the parameters are derived so that we maintain equilibrium at steady state. 

## Derived parameters to maintain physiological values at steady state

> **All derivations below now use $H_h = H(\theta^e_h) = 1/2$**, not the earlier $H_h \approx 0$.
> The formulas themselves are unchanged in structure — $H_h$ was always carried symbolically —
> but every *value* they produce changes, and the cytokine constraint changes algebraically
> because it had $H = 0$ substituted in.
>
> ⚠️ **The cytokine constraint was also inverted (2026-07-29).** It is one equation in
> $p_{c,\rho}$ and $K_{c,c}$; the original text sampled $p_{c,\rho}$ and derived $K_{c,c}$, while
> `params.derive()` does the opposite — **samples $K_{c,c}$, derives $p_{c,\rho}$**. The section
> below now matches the code. This is why $K_{c,c}$ carries **S** in the GP table and $p_{c,\rho}$
> carries **D**.

### Fibroblast carrying capacity
At homeostasis,

$$
s_\rho=0
$$

Using

$$
s_\rho
=
\left(
p_\rho+
p_{\rho,c}\frac{c}{K_{\rho,c}+c}
+
p_{\rho,e}H(\theta^e)
\right)
\left(
1-\frac{\rho}{K_{\rho,\rho}}
\right)\rho
-
d_\rho\rho
$$

Solving for $K_{\rho,\rho}$ gives:

$$
K_{\rho,\rho}
=
\frac{\rho_h}{
1-
\dfrac{
d_\rho
}{
p_\rho+
p_{\rho,c}\dfrac{c_h}{K_{\rho,c}+c_h}
+
p_{\rho,e}H_h
}
}
,\qquad H_h=\tfrac{1}{2}
$$

With $p_{\rho,e} = 6.80\,p_\rho$ and $H_h = 1/2$, the mechano-term alone contributes
$3.40\,p_\rho$ — more than three times the baseline mitotic rate.

#### ⚠️ Consequence: $K_{\rho,\rho}$ collapses toward $\rho_h$

Evaluating with $p_\rho = 0.014$, $d_\rho = 0.004$, $p_{\rho,c} = 2.5p_\rho$, $K_{\rho,c} = 10$:

| | proliferation at homeostasis | $K_{\rho,\rho}$ |
|---|---|---|
| old ($H_h\approx0$, $p_{\rho,e}=5p_\rho$) | 0.0172 /h | 1.303 |
| **new ($H_h=1/2$, $p_{\rho,e}=6.80p_\rho$)** | **0.0648 /h** | **1.066** |

Sensitivity to $p_{\rho,e}$ at $H_h = 1/2$:

| $p_{\rho,e}/p_\rho$ | 0 | 0.5 | 1 | 2 | 5 | 6.8 |
|---|---|---|---|---|---|---|
| $K_{\rho,\rho}$ | 1.303 | 1.240 | 1.198 | 1.147 | 1.083 | **1.066** |

Because mechanics now dominates the homeostatic proliferation budget, the logistic must nearly
shut off to balance $d_\rho$ — so $K_{\rho,\rho} \to \rho_h = 1$. **This means the model cannot
produce any fibroblast overshoot in the wound.** That is a real problem:
[[harbin-computational-2023]] measured ~7× overshoot (377 504 vs 55 051 cells/mm³ at 4 weeks),
and [[pensalfini-mechano-biological-2023]] uses $K_{\rho,\rho} = 30$ while deliberately keeping
$\Omega^m_\rho = 0.01$ *tiny* — precisely to stop this collapse (their S3 Appendix shows large
$\Omega^m$ causes "sustained overexpression of ρ").

**This is an unresolved tension, not a settled result.** Three ways out, none free:
1. $d_\rho$ in vivo is larger than the in-vitro growth-curve value (0.004–0.007 from
   [[goldschmidt-effect-2016]]). Note $p_\rho$ and $d_\rho$ were both measured *in vitro* at
   $\theta^e = 1$ where $H = 0.044$, **not** at the in vivo $H_h = 1/2$ — so the in-vitro
   balance is not the in-vivo balance, and reusing both unchanged is inconsistent.
2. $p_{\rho,e}$ is genuinely smaller in adult spinal dura than [[fong-mechanical-2003]]'s
   neonatal cranial value.
3. Dura genuinely sits at carrying capacity with no overshoot capacity — plausible for a dense,
   quiescent connective tissue, but it should be a *conclusion*, not an accident.

Recommend carrying $K_{\rho,\rho}$ and $d_\rho$ jointly through the GP rather than fixing either.


### Collagen Saturation Parameter
At homeostasis,

$$
\dot{\phi}=0
$$

Solving for $K_{\phi,\rho}$:

$$
K_{\phi,\rho}
=
\frac{
\left(
p_\phi+
p_{\phi,c}\dfrac{c_h}{K_{\phi,c}+c_h}
+
p_{\phi,e}H_h
\right)\rho_h
}{
\left(
d_\phi+c_h\rho_h d_{\phi,c}
\right)\phi_h
}
-
\phi_h
,\qquad H_h=\tfrac{1}{2}
$$

Structure unchanged; only $H_h$ moves from ≈0 to 1/2, which raises the numerator by
$\tfrac{1}{2}p_{\phi,e}\rho_h$ and therefore raises $K_{\phi,\rho}$.

⚠️ Keep $K_{\phi,\rho} > 0$: this requires
$p_\phi + p_{\phi,c}\frac{c_h}{K_{\phi,c}+c_h} + \tfrac{1}{2}p_{\phi,e} > (d_\phi + c_h\rho_h d_{\phi,c})\phi_h^2/\rho_h$.
[[harbin-computational-2023]]'s GP violated exactly this constraint and returned
$K_{\phi,\rho} = -0.205$ — **bound the GP search space so the derived value stays admissible.**

### Cytokine baseline production $p_{c,\rho}$ — derived by *sampling* $K_{c,c}$

**Two changes from the original derivation:** the old one substituted $H(J^e) = 0$, and it solved
the constraint the *other way round* (sample $p_{c,\rho}$, derive $K_{c,c}$). The operative
direction is now **sample $K_{c,c}$, back-solve $p_{c,\rho}$** — matching
`params.derive()`. Section retitled accordingly.

Homeostasis of the cytokine equation:

$$p_{c,\alpha}\alpha_{h}
+
\left(
p_{c,\rho}c_{h}+p_{c,e}H(\theta^e_h)
\right)
\left(
\frac{\rho_h}{K_{c,c}+c_{h}}
\right)
-
d_c c_{h} = 0$$

$$\alpha_{h} = 0, \quad H_h = H(\theta^e_h) = \tfrac{1}{2}, \quad c_{h} = 1, \quad \rho_{h} = 1.0$$

The $\alpha$ term still vanishes ($\alpha_h = 0$), but the mechano-term no longer does:

$$
\left(p_{c,\rho} + \tfrac{1}{2}p_{c,e}\right)\frac{1}{K_{c,c}+1} = d_c
$$

This is **one equation in two unknowns** ($p_{c,\rho}$ and $K_{c,c}$) — either may be sampled and
the other derived. Writing the mechano term as a ratio $r_{c,e} \equiv p_{c,e}/p_{c,\rho}$ so that
$p_{c,e} = r_{c,e}\,p_{c,\rho}$:

$$
p_{c,\rho}\left(1 + H_h\,r_{c,e}\right) = d_c\left(K_{c,c}+1\right)
$$

**We sample $K_{c,c}$ and derive $p_{c,\rho}$:**

$$
\boxed{\;p_{c,\rho} = \frac{d_c\left(K_{c,c}+1\right)}{1 + H_h\,r_{c,e}}\;,
\qquad p_{c,e} = r_{c,e}\,p_{c,\rho}\;,\qquad H_h=\tfrac12\;}
$$

## We are using only the normalized values.
For healthy tissue:
rho = 1.0, c = 1.0, phi = 1.0, alpha = 1e-4 (actually should be zero, but I am not sure if the solver with throw erros)
for wound (intially):
rho = 1e-4, c = 1e-4, phi = 1e-2, alpha = 1.0 (I want to avoid using 0 for numerical stability)
