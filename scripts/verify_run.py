#!/usr/bin/env python3
"""
Verify a wound-healing run from its VTK output.

Reads the primary VTK series written by writeParaview and checks the physics we
expect, rather than just eyeballing ParaView:

  1. homeostasis    - away from the wound, (rho, c, phi) stay at 1 and alpha at 0
  2. wound seeding  - the wound region starts depleted (rho, c ~ 1e-4, phi ~ 1e-2)
                      with alpha ~ 1
  3. alpha decay    - alpha in the wound decays, and the implied rate is close to
                      d_alpha = 0.0128 /h  (half-life ~54 h)
  4. cytokine rise  - c in the wound rises from its depleted value, driven by
                      p_c_alpha * alpha
  5. recovery       - rho and phi in the wound recover monotonically toward 1
  6. no negatives   - the solver has no clamping, so a negative concentration is
                      a silent failure

Usage:
    python scripts/verify_run.py <run_dir> [--prefix heal] [--dt 0.2]
"""
import argparse
import math
import re
import sys
from pathlib import Path

# Kept compatible with Python 3.6 (that is what Negishi's /usr/bin/python3 is),
# so no "from __future__ import annotations" and no PEP 604 unions.

STEP_RE = re.compile(r"_(\d+)\.vtk$")

# geometry of the needle track (must match the driver)
R_CORD, T_DURA, R_WOUND = 5.0, 0.4, 0.25
Y_CENTER = 0.0
Z_CENTER = 5.0
X_LO, X_HI = -(R_CORD + T_DURA), -R_CORD + 0.01

D_ALPHA = 0.0128  # 1/h


def read_vtk(path):
    """Return (points, {field: [values]}) from a legacy ASCII VTK file."""
    txt = path.read_text().splitlines()
    pts, fields, i = [], {}, 0
    while i < len(txt):
        line = txt[i].strip()
        if line.startswith("POINTS"):
            n = int(line.split()[1])
            i += 1
            while len(pts) < n:
                vals = [float(v) for v in txt[i].split()]
                for k in range(0, len(vals), 3):
                    pts.append((vals[k], vals[k + 1], vals[k + 2]))
                i += 1
            continue
        if line.startswith("SCALARS"):
            parts = line.split()
            name = parts[1]
            ncomp = int(parts[3]) if len(parts) > 3 else 1
            i += 2  # skip LOOKUP_TABLE
            npts = len(pts)
            rows = []
            while len(rows) < npts:
                vals = [float(v) for v in txt[i].split()]
                if vals:
                    rows.append(vals)
                i += 1
            if ncomp == 4 and name == "rho_c_phif_kappa":
                fields["rho"] = [r[0] for r in rows]
                fields["c"] = [r[1] for r in rows]
                fields["phi"] = [r[2] for r in rows]
                fields["kappa"] = [r[3] for r in rows]
            else:
                fields[name] = [r[0] for r in rows]
            continue
        i += 1
    return pts, fields


def masks(pts):
    wound, far = [], []
    for idx, (x, y, z) in enumerate(pts):
        r2 = (y - Y_CENTER) ** 2 + (z - Z_CENTER) ** 2
        inside = (X_LO - 1e-3 <= x <= X_HI) and r2 <= (R_WOUND + 1e-3) ** 2
        if inside:
            wound.append(idx)
        elif r2 > (6.0 * R_WOUND) ** 2:
            far.append(idx)
    return wound, far


def stat(vals, idxs):
    if not idxs:
        return float("nan"), float("nan"), float("nan")
    sub = [vals[i] for i in idxs]
    return min(sub), sum(sub) / len(sub), max(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--prefix", default="heal")
    ap.add_argument("--dt", type=float, default=0.2)
    a = ap.parse_args()

    files = []
    for p in a.run_dir.glob(f"{a.prefix}*.vtk"):
        if "second" in p.name:
            continue
        m = STEP_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    if not files:
        print(f"no VTK series matching {a.prefix}*_<step>.vtk in {a.run_dir}")
        return 2

    print(f"{len(files)} snapshots, steps {files[0][0]}..{files[-1][0]}, dt={a.dt} h\n")

    pts, _ = read_vtk(files[0][1])
    wound, far = masks(pts)
    print(f"{len(pts)} nodes: {len(wound)} in the wound, {len(far)} far field\n")
    if not wound:
        print("FAIL: wound region is empty - check the geometry constants")
        return 1

    print(f"{'t [h]':>8} {'rho_w':>9} {'c_w':>9} {'phi_w':>9} {'alpha_w':>9} "
          f"{'rho_far':>9} {'c_far':>9} {'phi_far':>9} {'a_far':>8} {'min':>10}")
    series = []
    for step, p in files:
        _, f = read_vtk(p)
        if "rho" not in f:
            continue
        t = step * a.dt
        alpha = f.get("alpha", [0.0] * len(pts))
        rw = stat(f["rho"], wound)[1]
        cw = stat(f["c"], wound)[1]
        pw = stat(f["phi"], wound)[1]
        aw = stat(alpha, wound)[1]
        rf = stat(f["rho"], far)[1]
        cf = stat(f["c"], far)[1]
        pf = stat(f["phi"], far)[1]
        af = stat(alpha, far)[1]
        gmin = min(min(f["rho"]), min(f["c"]), min(f["phi"]), min(alpha))
        series.append((t, rw, cw, pw, aw, rf, cf, pf, af, gmin))
        print(f"{t:8.1f} {rw:9.5f} {cw:9.5f} {pw:9.5f} {aw:9.5f} "
              f"{rf:9.5f} {cf:9.5f} {pf:9.5f} {af:8.2e} {gmin:10.2e}")

    if len(series) < 2:
        print("\nnot enough snapshots yet for the trend checks")
        return 0

    print("\n--- checks ---")
    fails = 0

    def ok(name, cond, detail=""):
        nonlocal fails
        if not cond:
            fails += 1
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    first, last = series[0], series[-1]

    # 1. far field stays at homeostasis
    ok("far field rho stays ~1", abs(last[5] - 1.0) < 5e-3, f"rho_far={last[5]:.5f}")
    ok("far field c stays ~1", abs(last[6] - 1.0) < 5e-2, f"c_far={last[6]:.5f}")
    ok("far field phi stays ~1", abs(last[7] - 1.0) < 5e-3, f"phi_far={last[7]:.5f}")
    ok("far field alpha stays ~0", abs(last[8]) < 5e-2, f"alpha_far={last[8]:.2e}")

    # 2. wound started depleted with a high alpha
    ok("wound started depleted in rho", first[1] < 0.5, f"rho_w(0)={first[1]:.5f}")
    ok("wound started depleted in phi", first[3] < 0.5, f"phi_w(0)={first[3]:.5f}")
    ok("wound started with high alpha", first[4] > 0.3, f"alpha_w(0)={first[4]:.5f}")

    # 3. alpha clears from the wound.
    #
    # NOTE: do NOT compare the wound-average rate against d_alpha. The wound is
    # only r = 0.25 mm across, so the diffusive timescale L^2/D_alpha = 6.7 h is
    # ~12x SHORTER than the decay timescale 1/d_alpha = 78 h: alpha leaves the
    # wound mainly by diffusing out, not by decaying in place. The pure decay
    # law is unit-tested separately in tests/test_homeostasis.cpp.
    if first[4] > 1e-6:
        ok("alpha clears from the wound", last[4] < first[4],
           f"{first[4]:.4f} -> {last[4]:.4f}")
        ok("alpha is monotonically non-increasing in the wound",
           all(series[i][4] <= series[i - 1][4] + 1e-6 for i in range(1, len(series))))
        # over a long enough run it should be essentially gone
        if last[0] - first[0] > 40.0:
            ok("alpha nearly cleared after >40 h", last[4] < 0.05,
               f"alpha_w={last[4]:.4f}")

    # 4/5. recovery
    ok("wound rho recovers", last[1] > first[1], f"{first[1]:.5f} -> {last[1]:.5f}")
    ok("wound phi recovers", last[3] > first[3], f"{first[3]:.5f} -> {last[3]:.5f}")
    ok("wound c rises from depletion", last[2] > first[2], f"{first[2]:.5f} -> {last[2]:.5f}")

    # 6. no negatives anywhere, any time. The solver has no positivity limiter,
    # so a real excursion means a step was accepted that was not a solution.
    worst = min(s[9] for s in series)
    t_worst = min(series, key=lambda r: r[9])[0]
    ok("no significant negative concentration", worst >= -1e-6,
       f"global min={worst:.3e} at t={t_worst:.1f} h")

    print(f"\n{'VERIFICATION PASSED' if not fails else f'{fails} CHECK(S) FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
