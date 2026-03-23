#!/usr/bin/env python3
"""Generate an L5 P4 report with safe reuse analysis and tuned HYPRE results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/reports/slope_stability_l5_p4_safe_reuse_tuned_hypre_lambda1"
)

DEFAULT_PATHS = {
    "pmg_same_mesh_current": Path("/tmp/l5_p4_mg_same_mesh_current_timed.json"),
    "pmg_same_mesh_reused": Path("/tmp/l5_p4_mg_same_mesh_reuse.json"),
    "pmg_l4tail_current": Path("/tmp/l5_p4_mg_l4tail_current.json"),
    "pmg_l4tail_reused": Path("/tmp/l5_p4_mg_l4tail_reuse.json"),
    "hypre_plain": Path("/tmp/l5_p4_hypre_timed.json"),
    "hypre_tuned_probe": Path("/tmp/l5_p4_hypre_tuned_300_maxit1.json"),
    "hypre_tuned_full": Path("/tmp/l5_p4_hypre_tuned_full.json"),
}


def _fmt(value: float, digits: int = 3) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.{digits}f}"


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
                digits = 6 if ("omega" in key or "u_max" in key or "residual" in key or "ratio" in key) else 3
                vals.append(_fmt(value, digits))
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _step(payload: dict[str, object]) -> dict[str, object]:
    return payload["result"]["steps"][0]


def _linear_timing(payload: dict[str, object]) -> list[dict[str, object]]:
    return list(_step(payload).get("linear_timing", []))


def _reuse_savings_row(name: str, payload: dict[str, object]) -> dict[str, object]:
    records = _linear_timing(payload)
    first = records[0]
    later = records[1:]
    later_linear_avg = (
        sum(float(r.get("linear_total_time", 0.0)) for r in later) / len(later)
        if later
        else math.nan
    )
    later_pc_setup_avg = (
        sum(float(r.get("pc_setup_time", 0.0)) for r in later) / len(later)
        if later
        else math.nan
    )
    later_assembly_avg = (
        sum(float(r.get("assemble_total_time", 0.0)) for r in later) / len(later)
        if later
        else math.nan
    )
    return {
        "variant": name,
        "first_linear_total_s": float(first.get("linear_total_time", math.nan)),
        "later_linear_avg_s": float(later_linear_avg),
        "linear_delta_s": (
            float(first.get("linear_total_time", math.nan)) - float(later_linear_avg)
            if later
            else math.nan
        ),
        "first_pc_setup_s": float(first.get("pc_setup_time", math.nan)),
        "later_pc_setup_avg_s": float(later_pc_setup_avg),
        "pc_setup_delta_s": (
            float(first.get("pc_setup_time", math.nan)) - float(later_pc_setup_avg)
            if later
            else math.nan
        ),
        "first_assembly_s": float(first.get("assemble_total_time", math.nan)),
        "later_assembly_avg_s": float(later_assembly_avg),
        "assembly_delta_s": (
            float(first.get("assemble_total_time", math.nan)) - float(later_assembly_avg)
            if later
            else math.nan
        ),
    }


def _full_row(name: str, hierarchy: str, payload: dict[str, object]) -> dict[str, object]:
    step = _step(payload)
    summary = step.get("linear_summary", {})
    return {
        "case": name,
        "hierarchy": hierarchy,
        "success": bool(payload["result"]["solver_success"]),
        "Newton": int(step.get("nit", 0)),
        "linear": int(step.get("linear_iters", 0)),
        "omega": float(step.get("omega", math.nan)),
        "u_max": float(step.get("u_max", math.nan)),
        "worst_true_rel": float(summary.get("worst_true_relative_residual", math.nan)),
        "setup_s": float(payload["timings"]["setup_time"]),
        "solve_s": float(payload["timings"]["solve_time"]),
        "total_s": float(payload["timings"]["total_time"]),
    }


def _rejected_reuse_row(name: str, current_payload: dict[str, object], reused_payload: dict[str, object]) -> dict[str, object]:
    current_step = _step(current_payload)
    reused_step = _step(reused_payload)
    current_linear = int(current_step.get("linear_iters", 0))
    reused_linear = int(reused_step.get("linear_iters", 0))
    increase = (
        100.0 * (reused_linear - current_linear) / current_linear
        if current_linear > 0
        else math.nan
    )
    return {
        "hierarchy": name,
        "current_linear": current_linear,
        "reused_linear": reused_linear,
        "increase_pct": float(increase),
        "current_solve_s": float(current_payload["timings"]["solve_time"]),
        "reused_solve_s": float(reused_payload["timings"]["solve_time"]),
        "reason": "Rejected: stale preconditioner reuse exceeds +10% linear iterations",
    }


def _hypre_row(name: str, payload: dict[str, object]) -> dict[str, object]:
    step = _step(payload)
    records = _linear_timing(payload)
    md = payload["metadata"]["linear_solver"]
    linear_summary = step.get("linear_summary", {})
    return {
        "case": name,
        "probe": "first Newton step" if int(payload["metadata"]["newton"]["maxit"]) == 1 else "full solve",
        "success": bool(payload["result"]["solver_success"]),
        "message": str(step.get("message", "")),
        "ksp_max_it": int(md["ksp_max_it"]),
        "nodal": int(md.get("hypre_nodal_coarsen", -1)),
        "vec_interp": int(md.get("hypre_vec_interp_variant", -1)),
        "Newton": int(step.get("nit", 0)),
        "linear": int(step.get("linear_iters", 0)),
        "omega": float(step.get("omega", math.nan)),
        "u_max": float(step.get("u_max", math.nan)),
        "ksp_reason": str(linear_summary.get("last_reason_name", "")),
        "true_rel": float(linear_summary.get("worst_true_relative_residual", math.nan)),
        "pc_setup_s": float(sum(float(rec.get("pc_setup_time", 0.0)) for rec in records)),
        "solve_s": float(sum(float(rec.get("solve_time", 0.0)) for rec in records)),
        "linear_total_s": float(sum(float(rec.get("linear_total_time", 0.0)) for rec in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payloads = {name: _load(path) for name, path in DEFAULT_PATHS.items()}

    full_rows = [
        _full_row("pmg_same_mesh_current", "same_mesh_p4_p2_p1", payloads["pmg_same_mesh_current"]),
        _full_row("pmg_l4tail_current", "same_mesh_p4_p2_p1_lminus1_p1", payloads["pmg_l4tail_current"]),
        _full_row("hypre_tuned_full", "boomeramg_tuned", payloads["hypre_tuned_full"]),
    ]
    rejected_rows = [
        _rejected_reuse_row(
            "same_mesh_p4_p2_p1",
            payloads["pmg_same_mesh_current"],
            payloads["pmg_same_mesh_reused"],
        ),
        _rejected_reuse_row(
            "same_mesh_p4_p2_p1_lminus1_p1",
            payloads["pmg_l4tail_current"],
            payloads["pmg_l4tail_reused"],
        ),
    ]
    savings_rows = [
        _reuse_savings_row("same_mesh_p4_p2_p1", payloads["pmg_same_mesh_current"]),
        _reuse_savings_row("same_mesh_p4_p2_p1_lminus1_p1", payloads["pmg_l4tail_current"]),
    ]
    hypre_rows = [
        _hypre_row("hypre_plain", payloads["hypre_plain"]),
        _hypre_row("hypre_tuned_probe", payloads["hypre_tuned_probe"]),
        _hypre_row("hypre_tuned_full", payloads["hypre_tuned_full"]),
    ]
    reusable_rows = [
        {
            "component": "KSP/PC objects",
            "status": "already reused",
            "depends_on_tangent": "no",
            "note": "Created once outside Newton; safe to keep alive",
        },
        {
            "component": "PCMG hierarchy, transfers, permutations",
            "status": "already reused",
            "depends_on_tangent": "no",
            "note": "Mesh/space topology only; safe to reuse",
        },
        {
            "component": "Near-nullspace vectors",
            "status": "already reused",
            "depends_on_tangent": "no",
            "note": "Built once from coordinates and freedofs",
        },
        {
            "component": "Matrix containers / sparsity / preallocation",
            "status": "already reused",
            "depends_on_tangent": "values only",
            "note": "Containers persist; values are refreshed each Newton step",
        },
        {
            "component": "Built preconditioner hierarchy/factors",
            "status": "rejected",
            "depends_on_tangent": "yes",
            "note": "Stale PC reuse raised linear iterations by more than 10%",
        },
    ]

    summary = {
        "lambda_target": 1.0,
        "full_rows": full_rows,
        "rejected_rows": rejected_rows,
        "savings_rows": savings_rows,
        "hypre_rows": hypre_rows,
        "reusable_rows": reusable_rows,
        "inputs": {key: str(path) for key, path in DEFAULT_PATHS.items()},
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# `L5` `P4` Safe Reuse And Tuned `HYPRE` Report

This report keeps only reuse choices that do not degrade total linear iterations by more than `10%`.

Interpretation:

- The normal solver already reuses the safe infrastructure: `KSP` objects, `PCMG` hierarchy, transfers, near-nullspace vectors, layouts, and matrix containers.
- The stale-preconditioner experiment from the earlier report is **not** included as a valid reuse mode, because it degraded iteration counts far beyond the `10%` cap.
- The `HYPRE` comparison now includes the completed tuned BoomerAMG full solve with `nodal_coarsen=6` and `vec_interp_variant=3`.

## Valid Full-Solve Comparison

{_table(full_rows, [
    ("case", "case"),
    ("hierarchy", "hierarchy"),
    ("success", "success"),
    ("Newton", "Newton"),
    ("linear", "linear"),
    ("omega", "omega"),
    ("u_max", "u_max"),
    ("worst true rel", "worst_true_rel"),
    ("setup [s]", "setup_s"),
    ("solve [s]", "solve_s"),
    ("total [s]", "total_s"),
])}

## What Can Be Reused Safely

{_table(reusable_rows, [
    ("component", "component"),
    ("status", "status"),
    ("depends on tangent", "depends_on_tangent"),
    ("note", "note"),
])}

## Rejected Stale-PC Reuse Modes

{_table(rejected_rows, [
    ("hierarchy", "hierarchy"),
    ("current linear", "current_linear"),
    ("reused linear", "reused_linear"),
    ("increase [%]", "increase_pct"),
    ("current solve [s]", "current_solve_s"),
    ("reused solve [s]", "reused_solve_s"),
    ("status", "reason"),
])}

## Time Saved By Safe Reuse Proxy

This is the requested `first iteration cost vs others` view on the valid current solvers. It is a proxy for how much setup amortization the already-reused solver infrastructure is giving us; it is not a pure nonlinear-apples-to-apples quantity because later Newton systems also change.

{_table(savings_rows, [
    ("variant", "variant"),
    ("first linear [s]", "first_linear_total_s"),
    ("later avg linear [s]", "later_linear_avg_s"),
    ("delta [s]", "linear_delta_s"),
    ("first PC setup [s]", "first_pc_setup_s"),
    ("later avg PC setup [s]", "later_pc_setup_avg_s"),
    ("PC delta [s]", "pc_setup_delta_s"),
    ("first asm [s]", "first_assembly_s"),
    ("later avg asm [s]", "later_assembly_avg_s"),
    ("asm delta [s]", "assembly_delta_s"),
])}

## Corrected `HYPRE` Comparison

{_table(hypre_rows, [
    ("case", "case"),
    ("probe", "probe"),
    ("success", "success"),
    ("Newton", "Newton"),
    ("ksp_max_it", "ksp_max_it"),
    ("nodal", "nodal"),
    ("vec interp", "vec_interp"),
    ("linear", "linear"),
    ("omega", "omega"),
    ("u_max", "u_max"),
    ("KSP reason", "ksp_reason"),
    ("true rel", "true_rel"),
    ("PC setup [s]", "pc_setup_s"),
    ("KSP solve [s]", "solve_s"),
    ("linear total [s]", "linear_total_s"),
    ("message", "message"),
])}
"""

    (output_dir / "README.md").write_text(report, encoding="utf-8")
    (output_dir / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
