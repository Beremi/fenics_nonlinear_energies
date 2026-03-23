#!/usr/bin/env python3
"""Report the L2 P4 same-mesh matrix-free search against assembled controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l2_p4_matfree_search_lambda1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l2_p4_matfree_search_lambda1"
)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _rows(rows: list[dict[str, object]], stage: str) -> list[dict[str, object]]:
    return [row for row in rows if str(row.get("stage")) == stage]


def _table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        vals = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(_fmt(value, 6 if "residual" in key else 3))
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stage0 = _rows(rows, "stage0")
    baseline = _rows(rows, "baseline")
    stage1_screen = _rows(rows, "stage1_screen")
    stage1_full = _rows(rows, "stage1_full")
    stage2 = _rows(rows, "stage2")
    stage3 = _rows(rows, "stage3")
    sanity = _rows(rows, "sanity")

    winner = next((row for row in rows if bool(row.get("passes_final_threshold"))), None)

    report = f"""# `L2` Same-Mesh `P4 -> P2 -> P1` Matrix-Free Search

This report compares matrix-free fine-`P4` PETSc options against the assembled same-problem controls on the single-mesh `L2`, `lambda=1.0` case.

## Stage 0 Controls

{_table(stage0, [
    ("case", "name"),
    ("success", "solver_success"),
    ("first KSP", "first_ksp_reason_name"),
    ("worst true rel", "worst_true_relative_residual"),
    ("energy", "energy"),
    ("fine P4 asm zero", "fine_p4_assembly_zero"),
])}

## Full Baselines

{_table(baseline, [
    ("case", "name"),
    ("success", "solver_success"),
    ("Newton", "newton_iterations"),
    ("linear", "linear_iterations"),
    ("worst true rel", "worst_true_relative_residual"),
    ("solve [s]", "solve_time_sec"),
    ("message", "message"),
])}

## Stage 1 Built-In Screen

{_table(stage1_screen[:16], [
    ("case", "name"),
    ("pass gate", "passes_stage0_gate"),
    ("first KSP", "first_ksp_reason_name"),
    ("worst true rel", "worst_true_relative_residual"),
    ("energy", "energy"),
    ("fine KSP", "mg_fine_down_ksp_type"),
    ("fine PC", "mg_fine_down_pc_type"),
])}

_Only the first 16 screen rows are shown above; the full machine-readable list is in `summary.json`._

## Stage 1 Full Candidates

{_table(stage1_full, [
    ("case", "name"),
    ("success", "solver_success"),
    ("pass target", "passes_final_threshold"),
    ("Newton", "newton_iterations"),
    ("linear", "linear_iterations"),
    ("worst true rel", "worst_true_relative_residual"),
    ("solve [s]", "solve_time_sec"),
])}

## Stage 2 Outer PCKSP Candidates

{_table(stage2, [
    ("case", "name"),
    ("success", "solver_success"),
    ("pass target", "passes_final_threshold"),
    ("linear", "linear_iterations"),
    ("worst true rel", "worst_true_relative_residual"),
    ("solve [s]", "solve_time_sec"),
]) if stage2 else "_No stage-2 rows were needed or available._"}

## Stage 3 Custom Python-PC Candidates

{_table(stage3, [
    ("case", "name"),
    ("success", "solver_success"),
    ("pass target", "passes_final_threshold"),
    ("linear", "linear_iterations"),
    ("worst true rel", "worst_true_relative_residual"),
    ("solve [s]", "solve_time_sec"),
    ("python PC", "python_pc_variant"),
    ("fine python PC", "mg_fine_python_pc_variant"),
]) if stage3 else "_No stage-3 rows were needed or available._"}

## Winner

{f"""
Winner: `{winner['name']}`

- linear iterations: `{winner['linear_iterations']}`
- worst true relative residual: `{winner['worst_true_relative_residual']:.6e}`
- solve time: `{winner['solve_time_sec']:.3f} s`
- fine `P4` assembly stayed zero: `{winner['fine_p4_assembly_zero']}`
""" if winner is not None else "No matrix-free candidate met the final threshold. The best achieved rows remain in the tables above and in `summary.json`."}

## Sanity Reruns

{_table(sanity, [
    ("case", "name"),
    ("success", "solver_success"),
    ("Newton", "newton_iterations"),
    ("linear", "linear_iterations"),
    ("worst true rel", "worst_true_relative_residual"),
    ("solve [s]", "solve_time_sec"),
]) if sanity else "_No sanity reruns were recorded._"}
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
