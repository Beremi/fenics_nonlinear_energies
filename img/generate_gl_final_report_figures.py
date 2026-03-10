#!/usr/bin/env python3
"""Generate GL final report figures from the completed suite."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_JSON = REPO_ROOT / "experiment_results_cache/gl_final_suite/summary.json"
OUT_DIR = Path(__file__).resolve().parent

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
    "jax_petsc_local_sfd": {
        "label": "JAX + PETSc local SFD",
        "color": "#4f772d",
        "marker": "^",
    },
}

LEVEL_DOFS = {
    5: 4225,
    6: 16641,
    7: 66049,
    8: 263169,
    9: 1050625,
}


def _load_rows() -> list[dict]:
    payload = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    return payload["rows"]


def _pick_row(rows: list[dict], solver: str, level: int, nprocs: int) -> dict:
    for row in rows:
        if (
            row["solver"] == solver
            and int(row["level"]) == int(level)
            and int(row["nprocs"]) == int(nprocs)
        ):
            return row
    raise KeyError((solver, level, nprocs))


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


def _plot_scaling(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    level_axes = {
        5: axes[0, 0],
        6: axes[0, 1],
        7: axes[0, 2],
        8: axes[1, 0],
        9: axes[1, 1],
    }
    axes[1, 2].axis("off")

    for level, ax in level_axes.items():
        nprocs = [1, 2, 4, 8, 16, 32]
        for solver, style in SOLVER_STYLE.items():
            ys = [_pick_row(rows, solver, level, np_)["total_time_s"] for np_ in nprocs]
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

    handles, labels = level_axes[5].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False)
    fig.suptitle("GL runtime scaling", fontsize=14)
    out_path = OUT_DIR / "gl_scaling.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_dof_runtime(rows: list[dict], nprocs: int = 8) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    xs = [LEVEL_DOFS[level] for level in (5, 6, 7, 8, 9)]
    for solver, style in SOLVER_STYLE.items():
        ys = [_pick_row(rows, solver, level, nprocs)["total_time_s"] for level in (5, 6, 7, 8, 9)]
        ax.loglog(
            xs,
            ys,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            markersize=6,
            label=style["label"],
        )
    ax.set_title(f"GL runtime vs DOFs at {nprocs} MPI ranks")
    ax.set_xlabel("Total DOFs")
    ax.set_ylabel("Runtime [s]")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    _add_linear_dof_triangle(ax)
    ax.legend(frameon=False)
    out_path = OUT_DIR / f"gl_dof_runtime_np{nprocs}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_fine_convergence(rows: list[dict], level: int = 9, nprocs: int = 32) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for solver, style in SOLVER_STYLE.items():
        row = _pick_row(rows, solver, level, nprocs)
        payload = json.loads(Path(row["json_path"]).read_text(encoding="utf-8"))
        history = payload["result"]["steps"][0]["history"]
        its = [int(rec["it"]) for rec in history]
        vals = [max(float(rec.get("grad_norm", 0.0)), 1e-16) for rec in history]
        ax.semilogy(
            its,
            vals,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            markersize=6,
            label=style["label"],
        )
    ax.set_title(f"Fine-mesh convergence profile, level {level}, {nprocs} MPI ranks")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Gradient norm")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    out_path = OUT_DIR / f"gl_convergence_l{level}_np{nprocs}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    rows = _load_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = [
        _plot_scaling(rows),
        _plot_dof_runtime(rows, nprocs=8),
        _plot_fine_convergence(rows, level=9, nprocs=32),
    ]
    for path in generated:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
