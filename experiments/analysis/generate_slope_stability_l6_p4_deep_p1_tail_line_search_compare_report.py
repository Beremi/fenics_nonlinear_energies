#!/usr/bin/env python3
"""Generate a golden-vs-Armijo comparison report for the L6 P4 deep-P1-tail PMG run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l6_p4_deep_p1_tail_line_search_compare_lambda1_np8_maxit20/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l6_p4_deep_p1_tail_line_search_compare_lambda1_np8_maxit20"
)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _load_histories(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        payload = json.loads(Path(str(row["result_json"])).read_text(encoding="utf-8"))
        steps = list(dict(payload.get("result", {})).get("steps", []))
        history = list(dict(steps[-1]).get("history", [])) if steps else []
        out[str(row["variant"])] = history
    return out


def _summary_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| method | success | ksp max it | accept cap | steady-state [s] | solve [s] | one-time setup [s] | Newton | linear | ls evals | ls time [s] | energy | omega | u_max | worst true rel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['label']}` | {row['solver_success']} | {int(row.get('ksp_max_it', 100))} | {bool(row.get('accept_ksp_maxit_direction', False))} | {_fmt(float(row['steady_state_total_time_sec']))} | "
            f"{_fmt(float(row['solve_time_sec']))} | {_fmt(float(row['one_time_setup_time_sec']))} | "
            f"{row['newton_iterations']} | {row['linear_iterations']} | {row['line_search_evals']} | "
            f"{_fmt(float(row['line_search_time_sec']))} | {_fmt(float(row['energy']), 9)} | "
            f"{_fmt(float(row['omega']), 6)} | {_fmt(float(row['u_max']), 6)} | "
            f"{_fmt(float(row['worst_true_relative_residual']), 6)} |"
        )
    return "\n".join(lines)


def _timing_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| method | energy [s] | gradient [s] | hessian [s] | line search [s] | linear assemble [s] | PC setup [s] | KSP solve [s] | iteration total [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['label']}` | {_fmt(float(row['energy_total_time_sec']))} | "
            f"{_fmt(float(row['gradient_total_time_sec']))} | {_fmt(float(row['hessian_total_time_sec']))} | "
            f"{_fmt(float(row['line_search_time_sec']))} | {_fmt(float(row['linear_assemble_time_sec']))} | "
            f"{_fmt(float(row['linear_pc_setup_time_sec']))} | {_fmt(float(row['linear_ksp_solve_time_sec']))} | "
            f"{_fmt(float(row['iteration_time_sec']))} |"
        )
    return "\n".join(lines)


def _comparison_table(rows: list[dict[str, object]]) -> str:
    by_variant = {str(row["variant"]): row for row in rows}
    if "golden_fixed" not in by_variant:
        return "_`golden_fixed` is required for the delta table._"
    golden = by_variant["golden_fixed"]
    lines = [
        "| method | metric | golden_fixed | variant | delta variant-golden |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    metric_specs = [
        ("steady-state [s]", "steady_state_total_time_sec"),
        ("solve [s]", "solve_time_sec"),
        ("Newton iters", "newton_iterations"),
        ("linear iters", "linear_iterations"),
        ("ls evals", "line_search_evals"),
        ("ls time [s]", "line_search_time_sec"),
        ("energy total [s]", "energy_total_time_sec"),
        ("gradient total [s]", "gradient_total_time_sec"),
        ("hessian total [s]", "hessian_total_time_sec"),
        ("KSP solve [s]", "linear_ksp_solve_time_sec"),
    ]
    for variant_name, variant_row in by_variant.items():
        if variant_name == "golden_fixed":
            continue
        for label, key in metric_specs:
            g = float(golden[key])
            v = float(variant_row[key])
            lines.append(
                f"| `{variant_row['label']}` | {label} | {_fmt(g)} | {_fmt(v)} | {_fmt(v - g)} |"
            )
    return "\n".join(lines)


def _plot_energy(histories: dict[str, list[dict[str, object]]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    for variant, history in histories.items():
        its = [int(item.get("it", idx + 1)) for idx, item in enumerate(history)]
        energies = [float(item.get("energy", np.nan)) for item in history]
        ax.plot(its, energies, marker="o", linewidth=1.8, label=variant)
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("energy")
    ax.set_title("Energy vs Newton iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_gradnorm(histories: dict[str, list[dict[str, object]]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    for variant, history in histories.items():
        its = [int(item.get("it", idx + 1)) for idx, item in enumerate(history)]
        grad = np.array([float(item.get("grad_norm", np.nan)) for item in history], dtype=np.float64)
        mask = np.isfinite(grad) & (grad > 0.0)
        if np.any(mask):
            ax.plot(np.array(its)[mask], grad[mask], marker="o", linewidth=1.8, label=variant)
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("gradient norm")
    ax.set_yscale("log")
    ax.set_title("Gradient norm vs Newton iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_linear_iters(histories: dict[str, list[dict[str, object]]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    for variant, history in histories.items():
        its = [int(item.get("it", idx + 1)) for idx, item in enumerate(history)]
        linear = [int(item.get("ksp_its", 0)) for item in history]
        ax.plot(its, linear, marker="o", linewidth=1.8, label=variant)
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("outer KSP iterations")
    ax.set_title("Linear iterations vs Newton iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_ls_evals(histories: dict[str, list[dict[str, object]]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    for variant, history in histories.items():
        its = [int(item.get("it", idx + 1)) for idx, item in enumerate(history)]
        vals = [int(item.get("ls_evals", 0)) for item in history]
        ax.plot(its, vals, marker="o", linewidth=1.8, label=variant)
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("line-search evaluations")
    ax.set_title("Line-search evaluations vs Newton iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = sorted(json.loads(args.input.read_text(encoding="utf-8")), key=lambda row: str(row["variant"]))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    histories = _load_histories(rows)

    _plot_energy(histories, out_dir / "energy_vs_newton.png")
    _plot_gradnorm(histories, out_dir / "gradnorm_vs_newton.png")
    _plot_linear_iters(histories, out_dir / "linear_iters_vs_newton.png")
    _plot_ls_evals(histories, out_dir / "ls_evals_vs_newton.png")

    report = f"""# `L6` `P4` Deep-`P1`-Tail PMG: line-search comparison on `8` ranks

Comparison setting:

- level: `6`
- discretisation: same-mesh `P4`
- hierarchy: `1:1,2:1,3:1,4:1,5:1,6:1,6:2,6:4`
- ranks: `8`
- nonlinear cap: `20` Newton iterations
- nonlinear setting: `--no-use_trust_region`
- linear setting: `fgmres`, `rtol=1e-2`, `max_it=100`
- smoothers:
  - `P4`: `richardson + sor`, `3` steps
  - `P2`: `richardson + sor`, `3` steps
  - `P1`: `richardson + sor`, `3` steps
- coarse solve:
  - `cg + hypre(boomeramg)`
  - `nodal_coarsen = 6`
  - `vec_interp_variant = 3`
  - `strong_threshold = 0.5`
  - `coarsen_type = HMIS`
  - `max_iter = 4`
  - `tol = 0.0`
  - `relax_type_all = symmetric-SOR/Jacobi`

## Summary

{_summary_table(rows)}

## Timing Breakdown

{_timing_table(rows)}

## Delta

{_comparison_table(rows)}

## Convergence Plots

![Energy vs Newton iteration](energy_vs_newton.png)

![Gradient norm vs Newton iteration](gradnorm_vs_newton.png)

![Linear iterations vs Newton iteration](linear_iters_vs_newton.png)

![Line-search evaluations vs Newton iteration](ls_evals_vs_newton.png)
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
