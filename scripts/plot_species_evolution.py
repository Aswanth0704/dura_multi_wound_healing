#!/usr/bin/env python3
"""Species evolution at the wound centre, the wound edge, and the wound average.

Produces a 2x2 figure - fibroblast density, cytokine, collagen fraction and
inflammatory signal - each with three curves showing how that field evolves at
three positions relative to the needle track.

Geometry. The track runs radially through the wall at reference (y = 0, z = 5.0).
The pre-stretch maps that to DEFORMED z = 5.49 via the axial stretch 1.098, and the
saved meshes are deformed, so the distance from the track axis is

    d = sqrt( y^2 + (z - 5.49)^2 )

measured in the plane normal to the track. Using the reference z = 5.0 here puts the
axis 0.49 mm off the real one - against a wound radius of 0.25 mm - so the centre
would sample tissue outside the wound entirely.

Regions (node counts are for the 2t mesh):

    centre   d <= 0.5*r_wound                 ~37 nodes
    edge     |d - r_wound| <= 0.05            ~570 nodes
    average  d <= r_wound                     ~295 nodes

The very core holds only 5 nodes, hence the half-radius window for "centre".

The node sets are chosen ONCE from the first healing frame and then held fixed, so
every curve follows the same material points for the whole four weeks. Re-selecting
by position at each frame would move different tissue in and out of a curve as the
wound deforms, which is not what "at the centre of the wound" means.

Usage:
    plot_species_evolution.py RUNDIR [--prefix wound_1_heal] [--dt 0.2]
                              [--rwound 0.25] [--zc 5.49] [--out DIR]
"""

import argparse
import glob
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _block(lines, start):
    """Collect float rows from `start` until a blank or non-numeric line."""
    out = []
    for s in lines[start:]:
        if not s.strip() or s[0].isalpha():
            break
        out.append([float(x) for x in s.split()])
    return np.array(out)


def read(path):
    """Return (points, packed[rho,c,phif,kappa], alpha)."""
    lines = open(path).read().split("\n")

    i = next(k for k, s in enumerate(lines) if s.startswith("POINTS"))
    n = int(lines[i].split()[1])
    pts = _block(lines, i + 1)[:n]

    j = next(k for k, s in enumerate(lines) if s.startswith("SCALARS rho_c_phif_kappa"))
    packed = _block(lines, j + 2)          # header line, then LOOKUP_TABLE

    alpha = None
    for k, s in enumerate(lines):
        if s.startswith("SCALARS alpha"):
            alpha = _block(lines, k + 2).ravel()
            break

    return pts, packed, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--prefix", default="wound_1_heal")
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--rwound", type=float, default=0.25)
    ap.add_argument("--zc", type=float, default=5.49,
                    help="deformed z of the track axis (reference 5.0 x lam_z 1.098)")
    ap.add_argument("--yc", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(A.rundir, f"{A.prefix}_[0-9]*.vtk"))
             if "_second_" not in os.path.basename(f)]
    if not files:
        raise SystemExit(f"no {A.prefix}_*.vtk in {A.rundir}")
    files.sort(key=lambda f: int(re.findall(r"(\d+)\.vtk$", f)[0]))

    # --- fix the three node sets once, on the first frame -------------------
    pts0, _, _ = read(files[0])
    d = np.hypot(pts0[:, 1] - A.yc, pts0[:, 2] - A.zc)
    regions = {
        "centre":  d <= 0.5 * A.rwound,
        "edge":    np.abs(d - A.rwound) <= 0.05,
        "average": d <= A.rwound,
    }
    print(f"{os.path.basename(A.rundir)}: {len(files)} frames")
    for name, m in regions.items():
        print(f"  {name:8s} {int(m.sum()):5d} nodes")

    # --- sweep the series ---------------------------------------------------
    t = []
    series = {name: {k: [] for k in ("rho", "c", "phif", "alpha")} for name in regions}
    for f in files:
        n = int(re.findall(r"(\d+)\.vtk$", f)[0])
        _, packed, alpha = read(f)
        t.append((n - 1) * A.dt)
        for name, m in regions.items():
            series[name]["rho"].append(packed[m, 0].mean())
            series[name]["c"].append(packed[m, 1].mean())
            series[name]["phif"].append(packed[m, 2].mean())
            series[name]["alpha"].append(
                alpha[m].mean() if alpha is not None and len(alpha) == len(packed)
                else np.nan)
    t = np.asarray(t)

    # --- figure -------------------------------------------------------------
    panels = [
        ("rho",   "fibroblast density", 1.0),
        ("c",     "cytokine concentration", 1.0),
        ("phif",  "collagen fraction", 1.0),
        ("alpha", "inflammatory signal", 0.0),
    ]
    style = {
        "centre":  dict(color="#c0392b", lw=2.0, label="wound centre  (d $\\leq$ 0.125 mm)"),
        "edge":    dict(color="#2874a6", lw=2.0, label="wound edge  (d $\\approx$ 0.25 mm)"),
        "average": dict(color="#1e8449", lw=2.0, ls="--",
                        label="wound average  (d $\\leq$ 0.25 mm)"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    for ax, (key, label, healthy) in zip(axes.ravel(), panels):
        for name in ("centre", "edge", "average"):
            ax.plot(t, series[name][key], **style[name])
        ax.axhline(healthy, color="0.35", ls=":", lw=1.2, zorder=0)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, lw=0.6)
        ax.margins(x=0.01)

    for ax in axes[1]:
        ax.set_xlabel("time after puncture [hours]")

    # a days axis across the top of the figure
    top = axes[0, 0].secondary_xaxis(
        "top", functions=(lambda h: h / 24.0, lambda dd: dd * 24.0))
    top.set_xlabel("time [days]")
    top2 = axes[0, 1].secondary_xaxis(
        "top", functions=(lambda h: h / 24.0, lambda dd: dd * 24.0))
    top2.set_xlabel("time [days]")

    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(f"Species evolution in the wound — {os.path.basename(A.rundir)}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = A.out or A.rundir
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, "species_evolution")
    fig.savefig(stem + ".png", dpi=200)
    fig.savefig(stem + ".pdf")
    print(f"  wrote {stem}.png and .pdf")

    # --- values for cross-checking against wound_fill.py --------------------
    print("           t=0                          t=end")
    print("         rho    phi    alpha        rho    phi    alpha")
    for name in ("centre", "edge", "average"):
        s = series[name]
        print(f"  {name:8s} {s['rho'][0]:.4f} {s['phif'][0]:.4f} {s['alpha'][0]:.4f}"
              f"     {s['rho'][-1]:.4f} {s['phif'][-1]:.4f} {s['alpha'][-1]:.4f}")


if __name__ == "__main__":
    main()
