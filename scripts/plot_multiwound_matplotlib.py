#!/usr/bin/env python3
"""
Matplotlib plotting utilities for multi-wound center analysis.

Reads precomputed CSVs from analysis_outputs/ and reproduces:
1) Final-value heatmaps (wound 1 and wound 2; rho and phi)
2) Wound 2 center trajectories vs time after second wound (faceted by Y, colored by X)
3) Wound 1 center trajectories during wound 2 healing (faceted by Y, colored by X)

Run:
    conda activate jaxenv
    python scripts/plot_multiwound_matplotlib.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PlotStyle:
    font_family: str = "DejaVu Sans"
    figure_dpi: int = 300
    save_format: str = "png"  # png, pdf, svg
    cmap: str = "viridis"
    linewidth: float = 2.3
    grid_alpha: float = 0.25
    title_size: int = 15
    label_size: int = 13
    tick_size: int = 11
    legend_size: int = 11
    annotation_size: int = 10

    def apply(self) -> None:
        plt.rcParams.update(
            {
                "font.family": self.font_family,
                "figure.dpi": self.figure_dpi,
                "savefig.dpi": self.figure_dpi,
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "legend.fontsize": self.legend_size,
            }
        )


def load_data(outdir: Path) -> Dict[str, pd.DataFrame]:
    data = {
        "final": pd.read_csv(outdir / "center_final_values_by_case.csv"),
        "samples": pd.read_csv(outdir / "center_samples_all_cases.csv"),
        "w1_during_w2": pd.read_csv(outdir / "wound1_center_during_wound2_samples.csv"),
    }
    # Ensure numeric dtypes
    for key, df in data.items():
        for col in ["X_t", "Y_week", "wound", "step_index"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in [
            "time_hours",
            "time_weeks_after_this_wound",
            "time_weeks_after_second_wound",
            "rho_fibroblast",
            "c_chemical",
            "phi_collagen",
            "kappa",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return data


def _filename(stem: str, style: PlotStyle) -> str:
    return f"{stem}.{style.save_format}"


def plot_final_heatmap(
    final_df: pd.DataFrame,
    field: str,
    wound: int,
    outdir: Path,
    style: PlotStyle,
    title: str,
    cbar_label: str,
    annotate: bool = True,
) -> Path:
    df = final_df[final_df["wound"] == wound].copy()
    pivot = (
        df.pivot_table(index="Y_week", columns="X_t", values=field, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(pivot.values, cmap=style.cmap, origin="lower", aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([int(x) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([int(y) for y in pivot.index])
    ax.set_xlabel("X (wound spacing in t)")
    ax.set_ylabel("Y (weeks healed before 2nd wound)")
    ax.set_title(title)

    if annotate:
        for iy in range(pivot.shape[0]):
            for ix in range(pivot.shape[1]):
                v = pivot.values[iy, ix]
                if np.isfinite(v):
                    ax.text(
                        ix,
                        iy,
                        f"{v:.3g}",
                        ha="center",
                        va="center",
                        fontsize=style.annotation_size,
                        color="white" if np.nanmean(pivot.values) < v else "black",
                    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    outpath = outdir / _filename(f"mpl_heatmap_final_{field}_wound{wound}", style)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _facets_layout(n_panels: int) -> Tuple[int, int]:
    if n_panels <= 1:
        return 1, 1
    if n_panels <= 2:
        return 1, 2
    if n_panels <= 4:
        return 2, 2
    return 2, 3


def plot_faceted_timeseries(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    outdir: Path,
    style: PlotStyle,
    title: str,
    ylabel: str,
    file_stem: str,
) -> Path:
    ys = sorted(df["Y_week"].dropna().unique())
    xs = sorted(df["X_t"].dropna().unique())
    nrows, ncols = _facets_layout(len(ys))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.6 * nrows),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_1d(axes).flatten()

    # stable color mapping by X
    cmap = plt.get_cmap("tab10")
    color_by_x = {x: cmap(i % 10) for i, x in enumerate(xs)}

    for i, y in enumerate(ys):
        ax = axes[i]
        ydf = df[df["Y_week"] == y]
        for x in xs:
            sdf = ydf[ydf["X_t"] == x].sort_values("step_index")
            if sdf.empty:
                continue
            ax.plot(
                sdf[time_col],
                sdf[value_col],
                label=f"{int(x)}t",
                linewidth=style.linewidth,
                color=color_by_x[x],
            )
        ax.set_title(f"Y = {int(y)} week")
        ax.set_xlabel("Weeks after second wound")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=style.grid_alpha)

    # Hide unused panels
    for j in range(len(ys), len(axes)):
        axes[j].axis("off")

    # single external legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, min(len(xs), 5)),
        frameon=False,
        title="X spacing",
    )
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    outpath = outdir / _filename(file_stem, style)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_all_from_csv(
    outdir: Path = Path("analysis_outputs"),
    style: PlotStyle = PlotStyle(),
) -> List[Path]:
    style.apply()
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_data(outdir)

    final_df = data["final"]
    samples_df = data["samples"]
    w1w2_df = data["w1_during_w2"]

    outputs: List[Path] = []

    # Heatmaps
    outputs.append(
        plot_final_heatmap(
            final_df,
            field="phi_collagen",
            wound=1,
            outdir=outdir,
            style=style,
            title="Final Collagen at Wound 1 Center",
            cbar_label="phi (collagen)",
        )
    )
    outputs.append(
        plot_final_heatmap(
            final_df,
            field="rho_fibroblast",
            wound=1,
            outdir=outdir,
            style=style,
            title="Final Fibroblast at Wound 1 Center",
            cbar_label="rho (fibroblast)",
        )
    )
    outputs.append(
        plot_final_heatmap(
            final_df,
            field="phi_collagen",
            wound=2,
            outdir=outdir,
            style=style,
            title="Final Collagen at Wound 2 Center",
            cbar_label="phi (collagen)",
        )
    )
    outputs.append(
        plot_final_heatmap(
            final_df,
            field="rho_fibroblast",
            wound=2,
            outdir=outdir,
            style=style,
            title="Final Fibroblast at Wound 2 Center",
            cbar_label="rho (fibroblast)",
        )
    )

    # Wound 2 center trajectories
    w2_df = samples_df[samples_df["wound"] == 2].copy()
    outputs.append(
        plot_faceted_timeseries(
            w2_df,
            time_col="time_weeks_after_this_wound",
            value_col="phi_collagen",
            outdir=outdir,
            style=style,
            title="Second Wound Center: Collagen vs Time",
            ylabel="phi (collagen)",
            file_stem="mpl_wound2_center_phi_vs_time_by_XY",
        )
    )
    outputs.append(
        plot_faceted_timeseries(
            w2_df,
            time_col="time_weeks_after_this_wound",
            value_col="rho_fibroblast",
            outdir=outdir,
            style=style,
            title="Second Wound Center: Fibroblast vs Time",
            ylabel="rho (fibroblast)",
            file_stem="mpl_wound2_center_rho_vs_time_by_XY",
        )
    )

    # Wound 1 center during wound 2 trajectories
    outputs.append(
        plot_faceted_timeseries(
            w1w2_df,
            time_col="time_weeks_after_second_wound",
            value_col="phi_collagen",
            outdir=outdir,
            style=style,
            title="First Wound Center During Second Wound Healing: Collagen vs Time",
            ylabel="phi (collagen)",
            file_stem="mpl_wound1_center_during_w2_phi_vs_time_by_XY",
        )
    )
    outputs.append(
        plot_faceted_timeseries(
            w1w2_df,
            time_col="time_weeks_after_second_wound",
            value_col="rho_fibroblast",
            outdir=outdir,
            style=style,
            title="First Wound Center During Second Wound Healing: Fibroblast vs Time",
            ylabel="rho (fibroblast)",
            file_stem="mpl_wound1_center_during_w2_rho_vs_time_by_XY",
        )
    )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Matplotlib figures from multi-wound center CSV outputs."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory containing analysis CSVs and destination for figures.",
    )
    parser.add_argument(
        "--format",
        dest="save_format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Figure file format (default: png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    style = PlotStyle(
        font_family="DejaVu Sans",
        figure_dpi=350,
        save_format=args.save_format,
        cmap="viridis",
        linewidth=2.4,
        title_size=15,
        label_size=13,
        tick_size=11,
        legend_size=11,
        annotation_size=10,
    )
    out_paths = plot_all_from_csv(args.outdir, style=style)
    print("Generated Matplotlib figures:")
    for p in out_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
