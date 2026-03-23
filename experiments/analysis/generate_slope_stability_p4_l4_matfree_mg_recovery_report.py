#!/usr/bin/env python3
"""Report the matrix-free P4 MG recovery sweep and final L4 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_p4_l4_matfree_mg_recovery_lambda1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_p4_l4_matfree_mg_recovery_lambda1"
)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _rows(rows: list[dict[str, object]], *, stage: str) -> list[dict[str, object]]:
    return [row for row in rows if str(row.get("stage")) == stage]


def _plot_scaling(rows: list[dict[str, object]], output_path: Path) -> None:
    plt.figure(figsize=(8.2, 4.8))
    for label, prefix in (
        ("assembled baseline", "assembled_baseline_np"),
        ("matfree candidate", "matfree_candidate_np"),
    ):
        subset = sorted(
            [row for row in rows if str(row["name"]).startswith(prefix)],
            key=lambda row: int(row["ranks"]),
        )
        if not subset:
            continue
        plt.plot(
            [int(row["ranks"]) for row in subset],
            [float(row["solve_time_sec"]) for row in subset],
            marker="o",
            linewidth=2.0,
            label=label,
        )
    plt.xscale("log", base=2)
    plt.xticks([1, 8, 16, 32], ["1", "8", "16", "32"])
    plt.xlabel("MPI ranks")
    plt.ylabel("Solve time [s]")
    plt.title("L4 lambda=1.0 P4 assembled vs matrix-free MG recovery")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _stage0_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| case | success | message | first solve reason | worst true rel residual | solve [s] |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        reason = (row.get("reason_names") or [""])[0]
        message = str(row.get("message", "")).replace("|", "/")
        lines.append(
            f"| {row['name']} | {row.get('solver_success')} | {message} | {reason} | "
            f"{_fmt(float(row.get('worst_true_relative_residual', 0.0)), 6)} | "
            f"{_fmt(float(row.get('solve_time_sec', 0.0)))} |"
        )
    return "\n".join(lines)


def _stage1_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| case | success | Newton | linear | all KSP converged | worst true rel residual | solve [s] |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row.get('solver_success')} | {row.get('newton_iterations', 0)} | "
            f"{row.get('linear_iterations', 0)} | {row.get('all_ksp_converged')} | "
            f"{_fmt(float(row.get('worst_true_relative_residual', 0.0)), 6)} | "
            f"{_fmt(float(row.get('solve_time_sec', 0.0)))} |"
        )
    return "\n".join(lines)


def _stage2_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| case | success | Newton | linear | all KSP converged | worst true rel residual | solve [s] | message |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row.get('solver_success')} | {row.get('newton_iterations', 0)} | "
            f"{row.get('linear_iterations', 0)} | {row.get('all_ksp_converged')} | "
            f"{_fmt(float(row.get('worst_true_relative_residual', 0.0)), 6)} | "
            f"{_fmt(float(row.get('solve_time_sec', 0.0)))} | "
            f"{str(row.get('message', '')).replace('|', '/')} |"
        )
    return "\n".join(lines)


def _stage3_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| case | ranks | success | Newton | linear | solve [s] | prep [s] | lower asm [s] | KSP [s] | worst true rel residual |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["name"]), int(item["ranks"]))):
        lines.append(
            f"| {row['name']} | {row['ranks']} | {row.get('solver_success')} | "
            f"{row.get('newton_iterations', 0)} | {row.get('linear_iterations', 0)} | "
            f"{_fmt(float(row.get('solve_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('operator_prepare_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('lower_level_assembly_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('ksp_solve_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('worst_true_relative_residual', 0.0)), 6)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stage0 = _rows(rows, stage="stage0")
    stage1 = _rows(rows, stage="stage1")
    stage2 = _rows(rows, stage="stage2")
    stage3 = _rows(rows, stage="stage3")

    if stage3:
        _plot_scaling(stage3, output_dir / "l4_scaling_solve_time.png")

    report = f"""# Matrix-Free `P4` MG Recovery (`L4`, `lambda=1.0`)

This report tracks the PETSc-only recovery work for the no-fine-assembly `P4` multigrid path:

- Stage 0: compare the assembled control against the new explicit hierarchy
- Stage 1: sweep PETSc-built-in matrix-free fine-level choices on `L2`
- Stage 3: benchmark the selected `L4` candidate against the assembled baseline

The current promoted candidate is `explicit_pmg_fgmres_fixed_p2sor`, meaning:

- finest `P4` operator is matrix-free (`matfree_overlap`)
- lower `P2/P1` operators are assembled once at setup (`fixed_setup`)
- fine-level smoother is `FGMRES`
- assembled `P2` level uses `SOR`

## Stage 0 Controls

{_stage0_table(stage0)}

## Stage 1 PETSc-Only Sweep

{_stage1_table(stage1)}

## Stage 2 `L4` Promotion Attempts

{_stage2_table(stage2)}

## Stage 3 `L4` Scaling

{_stage3_table(stage3)}

## Notes

- The direct refresh-each-Newton explicit hierarchy stayed too weak in this sweep and either diverged in `KSP` or hit `PCSETUP` failures.
- The best PETSc-only path so far is the fixed-lower-operator variant, which keeps the finest `P4` level matrix-free and never assembles the global `P4` sparse Hessian inside the Newton loop.

{"![L4 solve time vs ranks](l4_scaling_solve_time.png)" if stage3 else "_No stage-3 scaling rows were available when this report was generated._"}
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
