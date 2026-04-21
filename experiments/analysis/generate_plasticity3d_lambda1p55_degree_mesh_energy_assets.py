#!/usr/bin/env python3
"""Generate docs-facing assets for the Plasticity3D lambda=1.55 degree-energy study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments.analysis.docs_assets import common as docs_common


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_lambda1p55_degree_mesh_energy_study" / "comparison_summary.json"
)
DEFAULT_STUDY_DIR = DEFAULT_SUMMARY.parent
DEFAULT_DOCS_DIR = REPO_ROOT / "docs" / "assets" / "plasticity3d"

FIGURE_NAME = "plasticity3d_lambda1p55_degree_energy_study"
SUMMARY_NAME = "plasticity3d_lambda1p55_degree_energy_assets_summary.json"
MODEL_CARD_TABLE_NAME = "plasticity3d_lambda1p55_degree_energy_table.md"
LATEX_TABLE_NAME = "plasticity3d_lambda1p55_degree_energy_table.tex"

DEGREE_STYLE = {
    "P1": {"label": "P1", "color": "#1f77b4", "marker": "o"},
    "P2": {"label": "P2", "color": "#ff7f0e", "marker": "s"},
    "P4": {"label": "P4", "color": "#2ca02c", "marker": "^"},
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _sorted_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        (dict(row) for row in rows if isinstance(row, dict)),
        key=lambda row: (int(str(row.get("degree_line", "P0")).replace("P", "")), int(row.get("free_dofs", 0))),
    )


def _plot_panel(
    ax,
    rows: list[dict[str, object]],
    *,
    x_key: str,
    xlabel: str,
    degree_lines: tuple[str, ...] = ("P1", "P2", "P4"),
) -> None:
    for degree_line in degree_lines:
        style = DEGREE_STYLE[degree_line]
        selected = [row for row in rows if str(row.get("degree_line", "")) == degree_line]
        converged = [row for row in selected if str(row.get("status", "")) == "completed"]
        capped = [row for row in selected if str(row.get("status", "")) != "completed"]
        converged.sort(key=lambda row: _safe_float(row.get(x_key)))
        capped.sort(key=lambda row: _safe_float(row.get(x_key)))
        if converged:
            x = np.asarray([_safe_float(row.get(x_key)) for row in converged], dtype=np.float64)
            y = np.asarray([_safe_float(row.get("energy")) for row in converged], dtype=np.float64)
            ax.plot(
                x,
                y,
                marker=style["marker"],
                color=style["color"],
                linewidth=2.0,
                markersize=6.5,
                label=style["label"],
            )
        if capped:
            x = np.asarray([_safe_float(row.get(x_key)) for row in capped], dtype=np.float64)
            y = np.asarray([_safe_float(row.get("energy")) for row in capped], dtype=np.float64)
            ax.plot(
                x,
                y,
                linestyle="none",
                marker=style["marker"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.5,
                markersize=7.0,
            )
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final energy")


def _write_markdown_table(path: Path, rows: list[dict[str, object]]) -> None:
    headers = ["Degree", "Mesh", "Free DOFs", "Energy", "Total [s]", "Status", "Reused", "Artifact"]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row["degree_line"]),
                f"`{row['mesh_alias']}`",
                f"`{int(row['free_dofs'])}`",
                f"`{_safe_float(row['energy']):.6f}`",
                f"`{_safe_float(row['total_time_s']):.3f}`",
                f"`{row['status']}`",
                f"`{bool(row['reused'])}`",
                f"[artifact](/home/michal/repos/fenics_nonlinear_energies/{row['artifact_dir']})",
            ]
        )
    path.write_text(docs_common.markdown_table(headers, table_rows) + "\n", encoding="utf-8")


def _write_latex_table(path: Path, rows: list[dict[str, object]]) -> None:
    def _latex(text: object) -> str:
        return str(text).replace("_", r"\_")

    lines = [
        r"\begin{tabularx}{\textwidth}{@{}l l r r r l@{}}",
        r"\toprule",
        r"Degree & Mesh & Free DOFs & Energy & Total [s] & Status \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['degree_line']} & "
            f"{_latex(row['mesh_alias'])} & "
            f"{int(row['free_dofs'])} & "
            f"{_safe_float(row['energy']):.6f} & "
            f"{_safe_float(row['total_time_s']):.3f} & "
            f"{_latex(row['status'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--docs-out-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--degree-lines",
        nargs="+",
        choices=tuple(DEGREE_STYLE.keys()),
        default=list(DEGREE_STYLE.keys()),
        help="Subset of degree lines to include in the figure and exported tables.",
    )
    parser.add_argument(
        "--figure-stem",
        type=str,
        default=FIGURE_NAME,
        help="Stem for the emitted docs-side PDF/PNG pair.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default=SUMMARY_NAME,
        help="Filename for the emitted summary JSON inside the docs asset folder.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip writing the study markdown/LaTeX tables.",
    )
    parser.add_argument(
        "--zoom-degree-lines",
        nargs="+",
        choices=tuple(DEGREE_STYLE.keys()),
        default=["P2", "P4"],
        help="Degree lines to show in the zoomed second row; use an empty list to disable the zoom row.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json).resolve()
    study_dir = Path(args.study_dir).resolve()
    docs_out_dir = Path(args.docs_out_dir).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)
    docs_out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(summary_path)
    selected_degree_lines = set(args.degree_lines)
    rows = _sorted_rows(
        [
            row
            for row in list(summary.get("rows", []))
            if isinstance(row, dict) and str(row.get("degree_line", "")) in selected_degree_lines
        ]
    )
    if not rows:
        raise RuntimeError(f"No rows found in {summary_path}")

    plt = docs_common.configure_matplotlib()
    zoom_degree_lines = tuple(args.zoom_degree_lines)
    if zoom_degree_lines:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(2.05 * docs_common.figure_width_in(), 1.05 * docs_common.figure_width_in()),
        )
        _plot_panel(axes[0, 0], rows, x_key="free_dofs", xlabel="Free DOFs")
        _plot_panel(axes[0, 1], rows, x_key="total_time_s", xlabel="Total wall time [s]")
        _plot_panel(
            axes[1, 0],
            rows,
            x_key="free_dofs",
            xlabel="Free DOFs",
            degree_lines=zoom_degree_lines,
        )
        _plot_panel(
            axes[1, 1],
            rows,
            x_key="total_time_s",
            xlabel="Total wall time [s]",
            degree_lines=zoom_degree_lines,
        )
        axes[0, 0].set_title("All degrees: energy vs free DOFs")
        axes[0, 1].set_title("All degrees: energy vs total wall time")
        zoom_label = "/".join(zoom_degree_lines)
        axes[1, 0].set_title(f"{zoom_label} zoom: energy vs free DOFs")
        axes[1, 1].set_title(f"{zoom_label} zoom: energy vs total wall time")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 1].legend(handles, labels, loc="best")
        fig.tight_layout(h_pad=1.0, w_pad=1.2)
    else:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(2.05 * docs_common.figure_width_in(), 0.78 * docs_common.figure_width_in()),
        )
        _plot_panel(axes[0], rows, x_key="free_dofs", xlabel="Free DOFs")
        _plot_panel(axes[1], rows, x_key="total_time_s", xlabel="Total wall time [s]")
        axes[0].set_title("Energy vs free DOFs")
        axes[1].set_title("Energy vs total wall time")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[1].legend(handles, labels, loc="best")
        fig.tight_layout()

    docs_pdf = docs_out_dir / f"{args.figure_stem}.pdf"
    docs_png = docs_out_dir / f"{args.figure_stem}.png"
    fig.savefig(docs_pdf, format="pdf", dpi=600)
    fig.savefig(docs_png, format="png", dpi=220)
    plt.close(fig)

    markdown_table_path = study_dir / MODEL_CARD_TABLE_NAME
    latex_table_path = study_dir / LATEX_TABLE_NAME
    if not args.skip_tables:
        _write_markdown_table(markdown_table_path, rows)
        _write_latex_table(latex_table_path, rows)

    payload = {
        "summary_json": docs_common.repo_rel(summary_path),
        "constraint_variant": str(summary.get("constraint_variant", "")),
        "degree_lines": sorted(selected_degree_lines, key=lambda name: int(name.replace("P", ""))),
        "zoom_degree_lines": list(zoom_degree_lines),
        "rows": rows,
        "assets": {
            "docs_png": docs_common.repo_rel(docs_png),
            "docs_pdf": docs_common.repo_rel(docs_pdf),
        },
    }
    if not args.skip_tables:
        payload["assets"]["markdown_table"] = docs_common.repo_rel(markdown_table_path)
        payload["assets"]["latex_table"] = docs_common.repo_rel(latex_table_path)
    (docs_out_dir / args.summary_name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
