from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_backend_mix"
    / "comparison_summary.json"
)

ASSEMBLY_ORDER = ("local", "local_constitutiveAD", "source")
SOLVER_ORDER = ("local", "source")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _combo_label(row: dict) -> str:
    return f"{row['assembly_backend']} asm\n{row['solver_backend']} solve"


def _pair_label(row: dict) -> str:
    return f"{row['assembly_backend']} assembly"


def _row_map(summary: dict) -> dict[tuple[str, str], dict]:
    rows = [dict(row) for row in summary.get("rows", [])]
    return {
        (str(row.get("solver_backend")), str(row.get("assembly_backend"))): row
        for row in rows
    }


def _successful_rows(summary: dict) -> list[dict]:
    return [
        dict(row)
        for row in summary.get("rows", [])
        if str(row.get("status", "")).startswith("completed")
    ]


def _fmt_float(value: float, digits: int = 3) -> str:
    if not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _fmt_sci(value: float) -> str:
    if not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6e}"


def plot_bar_metrics(summary: dict, out_path: Path) -> None:
    rows = [dict(row) for row in summary.get("rows", [])]
    rows.sort(
        key=lambda row: (
            SOLVER_ORDER.index(str(row.get("solver_backend"))),
            ASSEMBLY_ORDER.index(str(row.get("assembly_backend"))),
        )
    )
    labels = [_combo_label(row) for row in rows]
    x = np.arange(len(rows), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0))
    metrics = (
        ("wall_time_s", "Wall Time [s]"),
        ("solve_time_s", "Solve Time [s]"),
        ("nit", "Newton Iterations"),
        ("linear_iterations_total", "Total Linear Iterations"),
    )
    colors = {
        ("local", "local"): "#4e79a7",
        ("local_constitutiveAD", "local"): "#59a14f",
        ("local", "source"): "#f28e2b",
        ("local_constitutiveAD", "source"): "#edc948",
        ("source", "local"): "#76b7b2",
        ("source", "source"): "#e15759",
    }
    for ax, (key, title) in zip(axes.flat, metrics, strict=True):
        values = [float(row.get(key, np.nan)) for row in rows]
        ax.bar(
            x,
            values,
            color=[
                colors.get(
                    (str(row["assembly_backend"]), str(row["solver_backend"])),
                    "#9c755f",
                )
                for row in rows
            ],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    _save(fig, out_path)


def plot_solver_pair_times(summary: dict, out_path: Path) -> None:
    row_map = _row_map(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    for ax, solver_backend in zip(axes, SOLVER_ORDER, strict=True):
        rows = [
            row_map.get((solver_backend, assembly_backend))
            for assembly_backend in ASSEMBLY_ORDER
        ]
        labels = [f"{assembly_backend} assembly" for assembly_backend in ASSEMBLY_ORDER]
        wall = [
            float(row.get("wall_time_s", np.nan)) if row is not None else np.nan
            for row in rows
        ]
        solve = [
            float(row.get("solve_time_s", np.nan)) if row is not None else np.nan
            for row in rows
        ]
        x = np.arange(len(labels), dtype=float)
        width = 0.34
        ax.bar(x - width / 2.0, wall, width=width, label="wall")
        ax.bar(x + width / 2.0, solve, width=width, label="solve")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{solver_backend.title()} Solver Backend")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Time [s]")
    _save(fig, out_path)


def plot_solver_pair_convergence(summary: dict, out_path: Path) -> None:
    row_map = _row_map(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    for ax, solver_backend in zip(axes, SOLVER_ORDER, strict=True):
        plotted = False
        for assembly_backend in ASSEMBLY_ORDER:
            row = row_map.get((solver_backend, assembly_backend))
            if row is None:
                continue
            it = np.array([int(v) for v in row.get("history_iterations", [])], dtype=int)
            metric = np.array([float(v) for v in row.get("history_metric", [])], dtype=float)
            if it.size == 0 or metric.size == 0:
                continue
            plotted = True
            ax.semilogy(
                it,
                metric,
                marker="o",
                linewidth=2.0,
                label=_pair_label(row),
            )
        if not plotted:
            ax.text(0.5, 0.5, "No history", ha="center", va="center")
        ax.set_title(f"{solver_backend.title()} Solver Backend")
        ax.set_xlabel("Newton iteration")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Relative correction")
    _save(fig, out_path)


def build_report(*, summary: dict, out_dir: Path) -> Path:
    bar_plot = out_dir / "backend_mix_bar_metrics.png"
    pair_plot = out_dir / "backend_mix_solver_pair_times.png"
    conv_plot = out_dir / "backend_mix_solver_pair_convergence.png"
    plot_bar_metrics(summary, bar_plot)
    plot_solver_pair_times(summary, pair_plot)
    plot_solver_pair_convergence(summary, conv_plot)

    row_map = _row_map(summary)
    successful_rows = _successful_rows(summary)

    best_row = None
    if successful_rows:
        best_row = min(successful_rows, key=lambda row: float(row.get("wall_time_s", np.inf)))

    lines: list[str] = []
    lines.append("# Plasticity3D Backend-Mix Comparison")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        f"- MPI ranks: `{int(summary.get('ranks', 0))}` with `relative_correction < {float(summary.get('stop_tol', 0.0))}` and `maxit = {int(summary.get('maxit', 0))}`."
    )
    lines.append(
        "- Six combinations: `local`, `local_constitutiveAD`, and `source` assembly crossed with local/source solvers."
    )
    lines.append(
        "- `local_constitutiveAD` keeps the maintained local energy and gradient path, but assembles tangents from constitutive-level autodiff in strain space."
    )
    if best_row is not None:
        lines.append(
            f"- Fastest completed wall time: `{best_row['combo_label']}` at `{_fmt_float(float(best_row['wall_time_s']))} s`."
        )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| combo | status | wall time [s] | solve time [s] | Newton iters | linear iters | final metric | energy | omega | u_max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for solver_backend in SOLVER_ORDER:
        for assembly_backend in ASSEMBLY_ORDER:
            row = row_map.get((solver_backend, assembly_backend))
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["combo_label"]),
                        f"`{row['status']}`",
                        _fmt_float(float(row.get("wall_time_s", np.nan))),
                        _fmt_float(float(row.get("solve_time_s", np.nan))),
                        str(int(row.get("nit", 0))),
                        str(int(row.get("linear_iterations_total", 0))),
                        _fmt_sci(float(row.get("final_metric", np.nan))),
                        _fmt_sci(float(row.get("energy", np.nan))),
                        _fmt_sci(float(row.get("omega", np.nan))),
                        _fmt_sci(float(row.get("u_max", np.nan))),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Same-Solver Assembly Comparison")
    lines.append("")
    for solver_backend in SOLVER_ORDER:
        solver_rows = [
            row_map.get((solver_backend, assembly_backend))
            for assembly_backend in ASSEMBLY_ORDER
        ]
        solver_rows = [row for row in solver_rows if row is not None]
        if not solver_rows:
            continue
        best_row = min(
            solver_rows,
            key=lambda row: float(row.get("wall_time_s", np.inf)),
        )
        lines.append(
            f"- `{solver_backend}` solver: fastest assembly = `{best_row['assembly_backend']}` at `{_fmt_float(float(best_row.get('wall_time_s', np.nan)))}` s."
        )
        local_row = row_map.get((solver_backend, "local"))
        if local_row is not None:
            local_wall = float(local_row.get("wall_time_s", np.nan))
            local_lin = float(local_row.get("linear_iterations_total", np.nan))
            for assembly_backend in ASSEMBLY_ORDER:
                if assembly_backend == "local":
                    continue
                other_row = row_map.get((solver_backend, assembly_backend))
                if other_row is None:
                    continue
                other_wall = float(other_row.get("wall_time_s", np.nan))
                other_lin = float(other_row.get("linear_iterations_total", np.nan))
                wall_ratio = (
                    float(other_wall / local_wall)
                    if np.isfinite(local_wall)
                    and np.isfinite(other_wall)
                    and local_wall > 0.0
                    else float("nan")
                )
                lin_ratio = (
                    float(other_lin / local_lin)
                    if np.isfinite(local_lin)
                    and np.isfinite(other_lin)
                    and local_lin > 0.0
                    else float("nan")
                )
                lines.append(
                    f"- `{solver_backend}` solver: `{assembly_backend}/local` wall-time ratio = `{_fmt_float(wall_ratio)}` and linear-iteration ratio = `{_fmt_float(lin_ratio)}`."
                )
    lines.append("")
    lines.append("## Assets")
    lines.append("")
    lines.append(f"- Bar metrics: `{_repo_rel(bar_plot)}`")
    lines.append(f"- Same-solver time comparison: `{_repo_rel(pair_plot)}`")
    lines.append(f"- Same-solver convergence comparison: `{_repo_rel(conv_plot)}`")
    lines.append(f"- Summary JSON: `{_repo_rel(out_dir / 'comparison_summary.json')}`")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots and a markdown report for the backend-mix comparison."
    )
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary_json).resolve()
    summary = _load_json(summary_path)
    out_dir = summary_path.parent if args.out_dir is None else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_path != out_dir / "comparison_summary.json":
        (out_dir / "comparison_summary.json").write_text(
            summary_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    report_path = build_report(summary=summary, out_dir=out_dir)
    print(report_path)


if __name__ == "__main__":
    main()
