#!/usr/bin/env python3
"""Generate HE final-report figures from the validated final campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT
    / "artifacts"
    / "reproduction"
    / "2026-03-15_refactor_stage2b_final"
    / "full"
    / "he_final_suite_best"
    / "summary.json"
)
DEFAULT_ASSET_DIR = REPO_ROOT / "docs" / "assets" / "hyperelasticity"

SOLVER_STYLE = {
    "fenics_custom": {
        "label": "FEniCS custom",
        "color": "#1f4e79",
        "marker": "o",
    },
    "jax_petsc_element": {
        "label": "JAX + PETSc element",
        "color": "#b85c38",
        "marker": "s",
    },
}

LEVEL_DOFS = {
    1: 2187,
    2: 12075,
    3: 78003,
    4: 555747,
}


def _load_rows(summary_json: Path) -> list[dict]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    return payload["rows"]


def _pick_row(rows: list[dict], solver: str, steps: int, level: int, nprocs: int) -> dict | None:
    for row in rows:
        if (
            row["solver"] == solver
            and int(row["total_steps"]) == int(steps)
            and int(row["level"]) == int(level)
            and int(row["nprocs"]) == int(nprocs)
        ):
            return row
    return None


def _time_value(row: dict | None) -> float:
    if row is None:
        return math.nan
    value = row.get("total_time_s")
    if value is None:
        return math.nan
    return float(value)


def _add_scaling_triangle(ax, label: str = "ideal\nslope -1") -> None:
    _x_min, x_max = ax.get_xlim()
    _y_min, y_max = ax.get_ylim()
    x_right = x_max / 1.15
    x_left = x_right / 2.0
    y_top = y_max / 1.5
    y_bottom = y_top / (x_right / x_left)
    tri = Polygon(
        [(x_left, y_top), (x_right, y_top), (x_right, y_bottom)],
        closed=True,
        facecolor="none",
        edgecolor="#444444",
        linewidth=1.2,
        linestyle="-",
        alpha=0.9,
    )
    ax.add_patch(tri)
    ax.text(
        (x_left * x_right) ** 0.5,
        (y_top * y_bottom) ** 0.5,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
    )


def _add_linear_dof_triangle(ax, label: str = "linear\nslope +1") -> None:
    _x_min, x_max = ax.get_xlim()
    y_min, _y_max = ax.get_ylim()
    x_right = x_max / 1.15
    x_left = x_right / 2.0
    y_bottom = y_min * 1.6
    y_top = y_bottom * (x_right / x_left)
    tri = Polygon(
        [(x_left, y_bottom), (x_right, y_bottom), (x_right, y_top)],
        closed=True,
        facecolor="none",
        edgecolor="#444444",
        linewidth=1.2,
        linestyle="-",
        alpha=0.9,
    )
    ax.add_patch(tri)
    ax.text(
        (x_left * x_right) ** 0.5,
        (y_bottom * y_top) ** 0.5,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
    )


def _plot_scaling(rows: list[dict], asset_dir: Path, steps: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    level_axes = {1: axes[0, 0], 2: axes[0, 1], 3: axes[1, 0], 4: axes[1, 1]}

    for level, ax in level_axes.items():
        nprocs = [1, 2, 4, 8, 16, 32] if level < 4 else [8, 16, 32]
        for solver, style in SOLVER_STYLE.items():
            ys = np.asarray(
                [_time_value(_pick_row(rows, solver, steps, level, np_)) for np_ in nprocs],
                dtype=np.float64,
            )
            ax.loglog(
                nprocs,
                ys,
                marker=style["marker"],
                color=style["color"],
                linewidth=2,
                markersize=6,
                label=style["label"],
            )
        ax.set_title(f"Level {level}")
        ax.set_xlabel("MPI processes")
        ax.set_ylabel("Runtime [s]")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        _add_scaling_triangle(ax)

    handles, labels = level_axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False)
    fig.suptitle(f"HE runtime scaling, {steps} steps", fontsize=14)
    out_path = asset_dir / f"he_scaling_{steps}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_dof_runtime(rows: list[dict], asset_dir: Path, steps: int, nprocs: int = 8) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    xs = [LEVEL_DOFS[level] for level in (1, 2, 3, 4)]
    for solver, style in SOLVER_STYLE.items():
        ys = np.asarray(
            [_time_value(_pick_row(rows, solver, steps, level, nprocs)) for level in (1, 2, 3, 4)],
            dtype=np.float64,
        )
        ax.loglog(
            xs,
            ys,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            markersize=6,
            label=style["label"],
        )
    ax.set_title(f"HE runtime vs DOFs at {nprocs} MPI ranks, {steps} steps")
    ax.set_xlabel("Total DOFs")
    ax.set_ylabel("Runtime [s]")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    _add_linear_dof_triangle(ax)
    ax.legend(frameon=False)
    out_path = asset_dir / f"he_dof_runtime_np{nprocs}_{steps}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    rows = _load_rows(args.summary_json.resolve())
    asset_dir = args.asset_dir.resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for steps in (24, 96):
        generated.append(_plot_scaling(rows, asset_dir, steps))
        generated.append(_plot_dof_runtime(rows, asset_dir, steps, nprocs=8))
    for path in generated:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
