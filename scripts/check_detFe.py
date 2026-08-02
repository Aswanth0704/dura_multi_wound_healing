#!/usr/bin/env python3
"""
Elastic-Jacobian health check for a wound-healing run.

The tissue is modelled as incompressible, so det(F^e) should stay at 1. In
run_w15 a one-element ring on the wound rim instead collapsed from 0.218 to
0.154 over 16 h while the far field held 0.99739 exactly, and the run died at
16.8 h. This script is the readout that detects it.

The measurement is EXACT, not inferred. src/wound.cpp writes

    ip_strain[ip] = 0.5*(CCe - Identity)

into the primary VTK as `TENSORS strain`, so

    CCe        = 2*strain + I
    det(F^e)   = sqrt(det(CCe))

with no modelling assumption. Cross-checks that validated it on run_w15:
  * the product lamdaE_a*lamdaE_s*lamdaE_n from the `_second_` files is an
    upper bound by Hadamard (det A <= prod A_ii for SPD) - it gave 0.1574
    against the exact 0.15417, i.e. lower as required;
  * reportState() computes det(F^e) at integration points on a completely
    different code path and printed 0.199942 against 0.21845 here;
  * on the settling files (no wound) this returns [0.99443, 1.00586] with zero
    nodes below 0.5, so it reports ~1 when it should.

Usage:
    check_detFe.py RUNDIR [RUNDIR ...] [--prefix w_heal] [--dt 0.2]
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

# wound axis in the deformed frame, from the driver's own log line
# "seeded wound: ... centre y=0 z=5.49 deformed, radius 0.25 mm"
Y_CENTRE, Z_CENTRE, R_WOUND = 0.0, 5.49, 0.25


def _block(lines, start, ncomp_stop):
    """Collect float rows following index `start` until a non-numeric line."""
    out = []
    for s in lines[start:]:
        if not s.strip():
            break
        if s[0].isalpha():          # next attribute header
            break
        out.append([float(x) for x in s.split()])
        if ncomp_stop and len(out) >= ncomp_stop:
            break
    return out


def read_vtk(path):
    """Return (points, CCe, phif) — phif is None if the packed block is absent."""
    lines = open(path).read().split("\n")

    pts = None
    for k, s in enumerate(lines):
        if s.startswith("POINTS"):
            n = int(s.split()[1])
            pts = np.array(_block(lines, k + 1, n))[:n]
            break

    CCe = None
    for k, s in enumerate(lines):
        if s.startswith("TENSORS strain"):
            E = np.array(_block(lines, k + 1, None)).reshape(-1, 3, 3)
            CCe = 2.0 * E + np.eye(3)
            break

    phif = None
    for k, s in enumerate(lines):
        if s.startswith("SCALARS rho_c_phif_kappa"):
            # header line then LOOKUP_TABLE line
            F = np.array(_block(lines, k + 2, None))
            if F.ndim == 2 and F.shape[1] >= 3:
                phif = F[:, 2]
            break

    return pts, CCe, phif


def read_Jp(path):
    """Jp and lamdaP from the companion `_second_` file, or None.

    Jp is THE quantity to watch: every stress term in wound.cpp carries it as a
    prefactor (SS_pas, SS_act, SS_vol are all Jp*(...)), so Jp -> 0 removes the
    element from the stiffness matrix entirely. In run_w15 the mesh minimum fell
    0.938 -> 0.133 while det(F^e) fell only 0.218 -> 0.154, which is why watching
    det(F^e) alone hid the real failure.
    """
    import os
    d, base = os.path.dirname(path), os.path.basename(path)
    # w_heal_12.vtk -> w_heal_second_12.vtk
    m = re.match(r"(.*?)(\d+)\.vtk$", base)
    if not m:
        return None
    cand = os.path.join(d, f"{m.group(1)}second_{m.group(2)}.vtk")
    if not os.path.exists(cand):
        return None
    lines = open(cand).read().split("\n")
    for k, s in enumerate(lines):
        if s.startswith("SCALARS Jp_lamdaP"):
            P = np.array(_block(lines, k + 2, None))   # Jp, lamdaP_a/s/n
            return P if P.ndim == 2 and P.shape[1] >= 4 else None
    return None


def analyse(rundir, prefix, dt):
    files = glob.glob(os.path.join(rundir, f"{prefix}_[0-9]*.vtk"))
    files = [f for f in files if "_second_" not in os.path.basename(f)]
    if not files:
        print(f"{os.path.basename(rundir):24s}  no {prefix}_*.vtk yet")
        return
    files.sort(key=lambda f: int(re.findall(r"(\d+)\.vtk$", f)[0]))

    print(f"\n=== {os.path.basename(rundir)} ===")
    print("  t[h]   min detFe   median   n<0.50  n<0.25   min@r_ax  phif "
          "|   Jp_min   lamP_n_min   J_min")
    prev_bad = None
    for f in files:
        n = int(re.findall(r"(\d+)\.vtk$", f)[0])
        try:
            pts, CCe, phif = read_vtk(f)
        except Exception as exc:                     # partial write while running
            print(f"  {(n-1)*dt:6.1f}   <unreadable: {exc}>")
            continue
        if CCe is None:
            continue
        J = np.sqrt(np.clip(np.linalg.det(CCe), 0.0, None))
        worst = int(np.argmin(J))
        r = np.hypot(pts[worst, 1] - Y_CENTRE, pts[worst, 2] - Z_CENTRE) if pts is not None else float("nan")
        pf = phif[worst] if phif is not None and len(phif) == len(J) else float("nan")
        nbad = int((J < 0.25).sum())
        P = read_Jp(f)
        if P is not None:
            Jp = P[:, 0]
            extra = (f"| {Jp.min():8.4f}   {P[:, 3].min():8.4f}   "
                     f"{(J * Jp).min():7.4f}")
        else:
            extra = "|   (no _second_ file)"
        print(f"  {(n-1)*dt:6.1f}   {J.min():9.5f}  {np.median(J):8.5f}   "
              f"{int((J < 0.5).sum()):5d}   {nbad:5d}   {r:8.3f} {pf:6.3f}  {extra}")
        prev_bad = nbad

    # verdict
    Jl = np.sqrt(np.clip(np.linalg.det(read_vtk(files[-1])[1]), 0.0, None))
    Jf = np.sqrt(np.clip(np.linalg.det(read_vtk(files[0])[1]), 0.0, None))
    n_first, n_last = int((Jf < 0.25).sum()), int((Jl < 0.25).sum())
    t_end = (int(re.findall(r"(\d+)\.vtk$", files[-1])[0]) - 1) * dt
    grew = n_last > n_first
    print(f"  -> reached t = {t_end:.1f} h; collapsed-node count "
          f"{'GREW' if grew else 'stable/shrank'} ({n_first} -> {n_last}); "
          f"min detFe {Jl.min():.5f}")
    print("  -> PASS" if (not grew and Jl.min() > 0.8) else "  -> collapse still present")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundirs", nargs="+")
    ap.add_argument("--prefix", default="w_heal")
    ap.add_argument("--dt", type=float, default=0.2)
    a = ap.parse_args()
    for d in a.rundirs:
        analyse(d, a.prefix, a.dt)


if __name__ == "__main__":
    sys.exit(main())
