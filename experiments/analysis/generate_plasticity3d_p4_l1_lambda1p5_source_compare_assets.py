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
    / "plasticity3d_p4_l1_lambda1p5"
    / "comparison_summary.json"
)


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


def _implementation_label(name: str) -> str:
    if str(name) == "source_petsc4py":
        return "Source PETSc4py"
    return "Maintained local"


def _mode_rows(summary: dict, mode: str) -> list[dict]:
    return [dict(row) for row in summary.get("rows", []) if str(row.get("mode")) == str(mode)]


def _successful_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("status", "")).startswith("completed")]


def _group_by_impl(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("implementation")), []).append(dict(row))
    for impl_rows in grouped.values():
        impl_rows.sort(key=lambda row: int(row.get("ranks", 0)))
    return grouped


def plot_fixed_work_times(rows: list[dict], out_path: Path) -> None:
    grouped = _group_by_impl(_successful_rows(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    metrics = (
        ("wall_time_s", "Wall Time [s]"),
        ("solve_time_s", "Solve Time [s]"),
    )
    for ax, (key, ylabel) in zip(axes, metrics, strict=True):
        for implementation, impl_rows in grouped.items():
            x = np.array([int(row["ranks"]) for row in impl_rows], dtype=int)
            y = np.array([float(row.get(key, np.nan)) for row in impl_rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, label=_implementation_label(implementation))
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({int(row["ranks"]) for row in rows}))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_title("Fixed-Work Strong Scaling")
    axes[1].set_title("Fixed-Work Strong Scaling")
    _save(fig, out_path)


def plot_fixed_work_iterations(rows: list[dict], out_path: Path) -> None:
    grouped = _group_by_impl(_successful_rows(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    metrics = (
        ("nit", "Newton iterations"),
        ("linear_iterations_total", "Total linear iterations"),
    )
    for ax, (key, ylabel) in zip(axes, metrics, strict=True):
        for implementation, impl_rows in grouped.items():
            x = np.array([int(row["ranks"]) for row in impl_rows], dtype=int)
            y = np.array([float(row.get(key, np.nan)) for row in impl_rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, label=_implementation_label(implementation))
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({int(row["ranks"]) for row in rows}))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_title("Fixed-Work Nonlinear Work")
    axes[1].set_title("Fixed-Work Linear Work")
    _save(fig, out_path)


def plot_reference_convergence(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    plotted = False
    for row in _successful_rows(rows):
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
            label=_implementation_label(str(row.get("implementation"))),
        )
    if not plotted:
        ax.text(0.5, 0.5, "No reference convergence history available", ha="center", va="center")
    ax.set_title("Reference-Rank Nonlinear Convergence")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Normalized metric")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out_path)


def _fmt_float(value: float, digits: int = 3) -> str:
    if not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _fmt_sci(value: float) -> str:
    if not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6e}"


def build_report(
    *,
    summary: dict,
    per_run_payloads: dict[str, dict],
    out_dir: Path,
) -> Path:
    fixed_rows = _mode_rows(summary, "fixed_work")
    reference_rows = _mode_rows(summary, "reference")

    fixed_time_plot = out_dir / "fixed_work_times.png"
    fixed_iter_plot = out_dir / "fixed_work_iterations.png"
    ref_plot = out_dir / "reference_rank_convergence.png"

    plot_fixed_work_times(fixed_rows, fixed_time_plot)
    plot_fixed_work_iterations(fixed_rows, fixed_iter_plot)
    plot_reference_convergence(reference_rows, ref_plot)

    lines: list[str] = []
    lines.append("# Plasticity3D `P4(L1), lambda = 1.5` Source-vs-Maintained Comparison")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        f"- Fixed-work sweep on MPI ranks `{', '.join(str(v) for v in summary.get('ranks', []))}` with `maxit = {int(summary.get('fixed_maxit', 0))}`."
    )
    lines.append(
        f"- Reference comparison at `{int(summary.get('reference_rank', 0))}` ranks with a nonlinear relative target."
    )
    lines.append(f"- Source environment mode: `{summary.get('source_env_mode', '')}`.")
    lines.append("")
    lines.append("## Fixed-Work Rows")
    lines.append("")
    lines.append("| implementation | ranks | status | wall time [s] | solve time [s] | Newton iters | linear iters | final metric |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in _successful_rows(fixed_rows):
        lines.append(
            "| "
            + " | ".join(
                [
                    _implementation_label(str(row["implementation"])),
                    str(int(row["ranks"])),
                    f"`{row['status']}`",
                    _fmt_float(float(row.get("wall_time_s", np.nan))),
                    _fmt_float(float(row.get("solve_time_s", np.nan))),
                    str(int(row.get("nit", 0))),
                    str(int(row.get("linear_iterations_total", 0))),
                    _fmt_sci(float(row.get("final_metric", np.nan))),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Reference Rank")
    lines.append("")
    lines.append("| implementation | status | wall time [s] | solve time [s] | Newton iters | linear iters | final metric | energy | omega | u_max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in _successful_rows(reference_rows):
        payload = per_run_payloads.get(str(row.get("result_json", "")), {})
        message = str(payload.get("message", row.get("message", "")))
        lines.append(
            "| "
            + " | ".join(
                [
                    _implementation_label(str(row["implementation"])),
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
        lines.append(f"- {_implementation_label(str(row['implementation']))}: `{message}`")
    lines.append("")
    lines.append("## Assets")
    lines.append("")
    lines.append(f"- Fixed-work timing plot: `{_repo_rel(fixed_time_plot)}`")
    lines.append(f"- Fixed-work iteration plot: `{_repo_rel(fixed_iter_plot)}`")
    lines.append(f"- Reference convergence overlay: `{_repo_rel(ref_plot)}`")
    lines.append(f"- Summary JSON: `{_repo_rel(out_dir / 'comparison_summary.json')}`")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison assets for the Plasticity3D source-vs-maintained campaign."
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

    per_run_payloads: dict[str, dict] = {}
    for row in summary.get("rows", []):
        result_json = str(row.get("result_json", "")).strip()
        if not result_json:
            continue
        path = REPO_ROOT / result_json if not Path(result_json).is_absolute() else Path(result_json)
        if path.exists():
            per_run_payloads[result_json] = _load_json(path)

    report_path = build_report(summary=summary, per_run_payloads=per_run_payloads, out_dir=out_dir)
    print(json.dumps({"report": _repo_rel(report_path)}, indent=2))


if __name__ == "__main__":
    main()
