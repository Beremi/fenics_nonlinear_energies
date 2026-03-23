#!/usr/bin/env python3
"""Report L5 P4 PCMG preconditioner reuse benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l5_p4_pmg_reuse_bench_lambda1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l5_p4_pmg_reuse_bench_lambda1"
)


def _fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


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
                vals.append(_fmt(value, 6 if "omega" in key or "u_max" in key or "residual" in key else 3))
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _pairwise(rows: list[dict[str, object]], hierarchy: str) -> dict[str, object] | None:
    current = next(
        (row for row in rows if row["hierarchy"] == hierarchy and row["pc_reuse_preconditioner"] is False and row["pc_type"] == "mg"),
        None,
    )
    reused = next(
        (row for row in rows if row["hierarchy"] == hierarchy and row["pc_reuse_preconditioner"] is True and row["pc_type"] == "mg"),
        None,
    )
    if current is None or reused is None:
        return None
    return {
        "hierarchy": hierarchy,
        "current_success": bool(current["solver_success"]),
        "reused_success": bool(reused["solver_success"]),
        "current_newton": int(current.get("newton_iterations", 0)),
        "reused_newton": int(reused.get("newton_iterations", 0)),
        "current_linear": int(current.get("linear_iterations", 0)),
        "reused_linear": int(reused.get("linear_iterations", 0)),
        "delta_omega": float(reused.get("omega", math.nan)) - float(current.get("omega", math.nan)),
        "delta_u_max": float(reused.get("u_max", math.nan)) - float(current.get("u_max", math.nan)),
        "solve_ratio_reused_over_current": (
            float(reused.get("solve_time_sec", math.nan)) / float(current.get("solve_time_sec", math.nan))
            if float(current.get("solve_time_sec", 0.0)) > 0.0
            else math.nan
        ),
        "pc_setup_ratio_reused_over_current": (
            float(reused.get("pc_setup_time_sec", math.nan)) / float(current.get("pc_setup_time_sec", math.nan))
            if float(current.get("pc_setup_time_sec", 0.0)) > 0.0
            else math.nan
        ),
        "ksp_ratio_reused_over_current": (
            float(reused.get("ksp_solve_time_sec", math.nan)) / float(current.get("ksp_solve_time_sec", math.nan))
            if float(current.get("ksp_solve_time_sec", 0.0)) > 0.0
            else math.nan
        ),
    }


def _share(value: float, total: float) -> float:
    if total <= 0.0:
        return math.nan
    return value / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = json.loads(args.input.read_text(encoding="utf-8"))
    rows = list(summary["rows"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = [
        row
        for row in (
            _pairwise(rows, "same_mesh_p4_p2_p1"),
            _pairwise(rows, "same_mesh_p4_p2_p1_lminus1_p1"),
        )
        if row is not None
    ]
    share_rows = []
    for row in rows:
        linear_total = float(row.get("linear_total_time_sec", math.nan))
        share_rows.append(
            {
                "case": row["name"],
                "assembly_share": _share(float(row.get("fine_operator_assembly_time_sec", math.nan)), linear_total),
                "pc_setup_share": _share(float(row.get("pc_setup_time_sec", math.nan)), linear_total),
                "ksp_share": _share(float(row.get("ksp_solve_time_sec", math.nan)), linear_total),
            }
        )

    best_current = min(
        (
            row
            for row in rows
            if row["pc_type"] == "mg" and row["pc_reuse_preconditioner"] is False and bool(row["solver_success"])
        ),
        key=lambda row: float(row["solve_time_sec"]),
    )
    same_mesh_pair = next((row for row in pair_rows if row["hierarchy"] == "same_mesh_p4_p2_p1"), None)
    l4tail_pair = next((row for row in pair_rows if row["hierarchy"] == "same_mesh_p4_p2_p1_lminus1_p1"), None)
    hypre_row = next((row for row in rows if row["pc_type"] == "hypre"), None)

    report = f"""# `L5` `P4` PCMG Reuse Benchmark

This benchmark compares:

- current legacy `PCMG`
- the same `PCMG` with `PETSc PC.setReusePreconditioner(True)`
- `Hypre/BoomerAMG`

All runs use `L5`, `lambda=1.0`, `--no-use_trust_region`, `fgmres`, and `ksp_max_it=100`.

The two multigrid hierarchies compared are:

- `same_mesh_p4_p2_p1`
- `same_mesh_p4_p2_p1_lminus1_p1` (adds an `L4 P1` tail level)

## Findings

- The best `PCMG` variant on `L5` is the current `same_mesh_p4_p2_p1_lminus1_p1` hierarchy: `{_fmt(float(best_current["solve_time_sec"]))} s`, `{best_current["newton_iterations"]}` Newton steps, `{best_current["linear_iterations"]}` total outer linear iterations.
- `PETSc PC.setReusePreconditioner(True)` does not preserve preconditioning quality here. It makes both hierarchies worse in Newton steps, linear iterations, and solve time.
- On `same_mesh_p4_p2_p1`, reuse raises total linear iterations from `{same_mesh_pair["current_linear"]}` to `{same_mesh_pair["reused_linear"]}` and solve time by `{_fmt((float(same_mesh_pair["solve_ratio_reused_over_current"]) - 1.0) * 100.0)}%`.
- On `same_mesh_p4_p2_p1_lminus1_p1`, reuse raises total linear iterations from `{l4tail_pair["current_linear"]}` to `{l4tail_pair["reused_linear"]}` and solve time by `{_fmt((float(l4tail_pair["solve_ratio_reused_over_current"]) - 1.0) * 100.0)}%`.
- `Hypre/BoomerAMG` is not competitive on this `L5 P4` case under the same `ksp_max_it=100` cap: it stops on the first Newton solve with `DIVERGED_MAX_IT` and worst true relative residual `{_fmt(float(hypre_row["worst_true_relative_residual"]), 6)}`.

## Full Comparison

{_table(rows, [
    ("case", "name"),
    ("hierarchy", "hierarchy"),
    ("pc", "pc_type"),
    ("reuse PC", "pc_reuse_preconditioner"),
    ("success", "solver_success"),
    ("Newton", "newton_iterations"),
    ("linear", "linear_iterations"),
    ("omega", "omega"),
    ("u_max", "u_max"),
    ("worst true rel", "worst_true_relative_residual"),
    ("setup [s]", "setup_time_sec"),
    ("solve [s]", "solve_time_sec"),
    ("total [s]", "total_time_sec"),
])}

## Time Breakdown

{_table(rows, [
    ("case", "name"),
    ("operator prep [s]", "operator_prepare_time_sec"),
    ("fine asm [s]", "fine_operator_assembly_time_sec"),
    ("fine Pmat asm [s]", "fine_pmat_step_assembly_time_sec"),
    ("PC setup [s]", "pc_setup_time_sec"),
    ("KSP solve [s]", "ksp_solve_time_sec"),
    ("linear total [s]", "linear_total_time_sec"),
])}

## Time Shares Within Linear Phase

{_table(share_rows, [
    ("case", "case"),
    ("assembly / linear", "assembly_share"),
    ("PC setup / linear", "pc_setup_share"),
    ("KSP solve / linear", "ksp_share"),
])}

## Reuse vs Current

{_table(pair_rows, [
    ("hierarchy", "hierarchy"),
    ("current ok", "current_success"),
    ("reused ok", "reused_success"),
    ("current Newton", "current_newton"),
    ("reused Newton", "reused_newton"),
    ("current linear", "current_linear"),
    ("reused linear", "reused_linear"),
    ("delta omega", "delta_omega"),
    ("delta u_max", "delta_u_max"),
    ("solve ratio", "solve_ratio_reused_over_current"),
    ("PC setup ratio", "pc_setup_ratio_reused_over_current"),
    ("KSP ratio", "ksp_ratio_reused_over_current"),
])}
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
