#!/usr/bin/env python3
"""Wound FILLING progress: fibroblast and collagen recovery inside the defect.

Closure in this model means the wound FILLS with fibroblasts and collagen, the
way an ordinary wound does - not that the edges are pulled together. Myofibroblast
traction contracts the defect somewhat, but it does not join the edges. So the
success criterion is rho and phif returning toward 1 inside the wound, and the
cross-section radius is secondary.

Reports mean and minimum of each field over the wound core (within r_wound of the
needle axis), against the healthy value of 1.

Usage:
    wound_fill.py RUNDIR [RUNDIR ...] [--prefix wound_1_heal] [--dt 0.2]
                  [--rwound 0.25] [--zc 5.49] [--rows 10]
"""

import argparse
import glob
import os
import re

import numpy as np


def _block(lines, start):
    out = []
    for s in lines[start:]:
        if not s.strip() or s[0].isalpha():
            break
        out.append([float(x) for x in s.split()])
    return np.array(out)


def read(path):
    lines = open(path).read().split("\n")
    i = next(k for k, s in enumerate(lines) if s.startswith("POINTS"))
    n = int(lines[i].split()[1])
    pts = _block(lines, i + 1)[:n]
    j = next(k for k, s in enumerate(lines) if s.startswith("SCALARS rho_c_phif_kappa"))
    F = _block(lines, j + 2)          # rho, c, phif, kappa
    a = None
    for k, s in enumerate(lines):
        if s.startswith("SCALARS alpha"):
            a = _block(lines, k + 2).ravel()
            break
    return pts, F, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundirs", nargs="+")
    ap.add_argument("--prefix", default="wound_1_heal")
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--rwound", type=float, default=0.25)
    ap.add_argument("--zc", type=float, default=5.49)
    ap.add_argument("--rows", type=int, default=10)
    A = ap.parse_args()

    for run in A.rundirs:
        files = [f for f in glob.glob(os.path.join(run, f"{A.prefix}_[0-9]*.vtk"))
                 if "_second_" not in os.path.basename(f)]
        if not files:
            print(f"{run}: no saves")
            continue
        files.sort(key=lambda f: int(re.findall(r"(\d+)\.vtk$", f)[0]))

        pts0, _, _ = read(files[0])
        core = np.hypot(pts0[:, 1], pts0[:, 2] - A.zc) < A.rwound

        print(f"\n=== {os.path.basename(run)} — {len(files)} saves, "
              f"{int(core.sum())} core nodes (healthy = 1) ===")
        print("   t[h]    rho_mean  rho_min | phi_mean  phi_min | c_mean  | alpha_mean")
        step = max(1, len(files) // A.rows)
        first = last = None
        for f in files[::step] + [files[-1]]:
            n = int(re.findall(r"(\d+)\.vtk$", f)[0])
            _, F, al = read(f)
            row = (F[core, 0].mean(), F[core, 0].min(),
                   F[core, 2].mean(), F[core, 2].min(),
                   F[core, 1].mean(),
                   al[core].mean() if al is not None and len(al) == len(F) else float("nan"))
            print(f"{(n-1)*A.dt:8.1f}   {row[0]:8.4f} {row[1]:8.4f} | "
                  f"{row[2]:8.4f} {row[3]:8.4f} | {row[4]:7.4f} | {row[5]:9.4f}")
            if first is None:
                first = row
            last = row

        # Fill rate, and where a straight line would put it at four weeks.
        # Deliberately crude: recovery is not linear, so treat this as an order
        # of magnitude on whether the run is even heading toward 1, not a forecast.
        t_end = (int(re.findall(r"(\d+)\.vtk$", files[-1])[0]) - 1) * A.dt
        if t_end > 0 and first is not None:
            drho = (last[0] - first[0]) / t_end
            dphi = (last[2] - first[2]) / t_end
            print(f"  fill rate over {t_end:.1f} h: rho {drho:+.5f}/h, phi {dphi:+.5f}/h")
            print(f"  straight-line to 672 h: rho -> {first[0] + drho*672:.3f}, "
                  f"phi -> {first[2] + dphi*672:.3f}   (healthy = 1)")


if __name__ == "__main__":
    main()
