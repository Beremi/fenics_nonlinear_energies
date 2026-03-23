#!/usr/bin/env python3
"""Generate an extended L5 P4 backend comparison report."""

from __future__ import annotations

import json
import math
from pathlib import Path


OUTPUT_DIR = Path("artifacts/reports/slope_stability_l5_p4_backend_comparison_lambda1")
RAW_DIR = Path("artifacts/raw_results/slope_stability_l5_p4_backend_comparison_lambda1")

PATHS = {
    "pmg_same_mesh_current": Path("/tmp/l5_p4_mg_same_mesh_current_timed.json"),
    "pmg_l4tail_current": Path("/tmp/l5_p4_mg_l4tail_current.json"),
    "hypre_plain_full": Path("/tmp/l5_p4_hypre_timed.json"),
    "hypre_tuned_step1": Path("/tmp/l5_p4_hypre_tuned_300_maxit1.json"),
    "hypre_tuned_step2": Path("/tmp/l5_p4_hypre_tuned_maxit2.json"),
    "hypre_tuned_full": Path("/tmp/l5_p4_hypre_tuned_full.json"),
    "gamg_tuned_full": Path("/tmp/l5_p4_gamg_tuned_full.json"),
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
                digits = 6 if ("omega" in key or "u_max" in key or "rel" in key) else 3
                vals.append(_fmt(value, digits))
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _step(payload: dict[str, object]) -> dict[str, object]:
    return payload["result"]["steps"][0]


def _linear_records(payload: dict[str, object]) -> list[dict[str, object]]:
    return list(_step(payload).get("linear_timing", []))


def _full_row(label: str, hierarchy: str, payload: dict[str, object]) -> dict[str, object]:
    step = _step(payload)
    summary = step["linear_summary"]
    return {
        "case": label,
        "kind": "full",
        "success": bool(payload["result"]["solver_success"]),
        "hierarchy": hierarchy,
        "newton": int(step["nit"]),
        "linear": int(step["linear_iters"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "worst_true_rel": float(summary["worst_true_relative_residual"]),
        "setup_s": float(payload["timings"]["setup_time"]),
        "solve_s": float(payload["timings"]["solve_time"]),
        "status": str(step["message"]),
    }


def _probe_row(label: str, payload: dict[str, object], *, note: str) -> dict[str, object]:
    step = _step(payload)
    summary = step["linear_summary"]
    recs = _linear_records(payload)
    return {
        "case": label,
        "kind": "partial",
        "success": bool(payload["result"]["solver_success"]),
        "newton": int(step["nit"]),
        "linear": int(step["linear_iters"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "worst_true_rel": float(summary["worst_true_relative_residual"]),
        "setup_s": float(payload["timings"]["setup_time"]),
        "solve_s": float(payload["timings"]["solve_time"]),
        "avg_linear_s": (
            sum(float(r["linear_total_time"]) for r in recs) / len(recs) if recs else math.nan
        ),
        "first_linear_s": float(recs[0]["linear_total_time"]) if recs else math.nan,
        "last_linear_s": float(recs[-1]["linear_total_time"]) if recs else math.nan,
        "note": note,
    }


def main() -> None:
    payloads = {name: _load(path) for name, path in PATHS.items()}

    comparison_rows = [
        _full_row("PCMG same-mesh", "same_mesh_p4_p2_p1", payloads["pmg_same_mesh_current"]),
        _full_row("PCMG with L4 tail", "same_mesh_p4_p2_p1_lminus1_p1", payloads["pmg_l4tail_current"]),
        _full_row("Hypre plain", "boomeramg", payloads["hypre_plain_full"]),
        _full_row("Hypre tuned", "boomeramg(nodal=6,vec=3)", payloads["hypre_tuned_full"]),
        _full_row("GAMG tuned", "gamg(th=0.02,ns=1)", payloads["gamg_tuned_full"]),
    ]

    hypre_progress_rows = [
        _probe_row(
            "Hypre tuned step 1",
            payloads["hypre_tuned_step1"],
            note="BoomerAMG nodal_coarsen=6, vec_interp_variant=3; first Newton solve converged",
        ),
        _probe_row(
            "Hypre tuned steps 1-2",
            payloads["hypre_tuned_step2"],
            note="Same tuned BoomerAMG; first two Newton solves converged, full L5 run remains expensive",
        ),
    ]

    hypre_linear_rows = []
    for payload, label in (
        (payloads["hypre_tuned_step1"], "tuned step 1"),
        (payloads["hypre_tuned_step2"], "tuned steps 1-2"),
    ):
        for idx, record in enumerate(_linear_records(payload), start=1):
            hypre_linear_rows.append(
                {
                    "run": label,
                    "Newton step": idx,
                    "KSP iters": int(record["ksp_its"]),
                    "KSP reason": str(record["ksp_reason_name"]),
                    "true rel": float(record["true_relative_residual"]),
                    "PC setup [s]": float(record["pc_setup_time"]),
                    "linear total [s]": float(record["linear_total_time"]),
                }
            )

    summary = {
        "inputs": {name: str(path) for name, path in PATHS.items()},
        "comparison_rows": comparison_rows,
        "hypre_progress_rows": hypre_progress_rows,
        "hypre_linear_rows": hypre_linear_rows,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# `L5` `P4` Backend Comparison

This report updates the `L5` backend comparison with:

- the best full `PCMG` runs
- a corrected tuned `Hypre/BoomerAMG` full run
- a tuned `GAMG` full attempt

## Full Backends

{_table(comparison_rows, [
    ("case", "case"),
    ("kind", "kind"),
    ("success", "success"),
    ("hierarchy", "hierarchy"),
    ("Newton", "newton"),
    ("linear", "linear"),
    ("omega", "omega"),
    ("u_max", "u_max"),
    ("worst true rel", "worst_true_rel"),
    ("setup [s]", "setup_s"),
    ("solve [s]", "solve_s"),
    ("status", "status"),
])}

## Tuned `Hypre` Progression

{_table(hypre_progress_rows, [
    ("case", "case"),
    ("kind", "kind"),
    ("success", "success"),
    ("Newton reached", "newton"),
    ("linear", "linear"),
    ("omega", "omega"),
    ("u_max", "u_max"),
    ("worst true rel", "worst_true_rel"),
    ("setup [s]", "setup_s"),
    ("solve [s]", "solve_s"),
    ("first linear [s]", "first_linear_s"),
    ("last linear [s]", "last_linear_s"),
    ("avg linear [s]", "avg_linear_s"),
    ("note", "note"),
])}

## Tuned `Hypre` Linear Solves

{_table(hypre_linear_rows, [
    ("run", "run"),
    ("Newton step", "Newton step"),
    ("KSP iters", "KSP iters"),
    ("KSP reason", "KSP reason"),
    ("true rel", "true rel"),
    ("PC setup [s]", "PC setup [s]"),
    ("linear total [s]", "linear total [s]"),
])}

## Takeaways

- Best full solve remains `PCMG` with the `L4` tail hierarchy: `21` Newton steps, `185` linear iterations, `98.800 s`.
- Tuned `GAMG` still fails immediately on the first Newton solve, even after increasing `ksp_max_it` to `300` and setting `gamg_threshold=0.02`.
- Plain `Hypre` was under-configured. The tuned full `BoomerAMG` run completes successfully:
  - `23` Newton steps
  - `445` total linear iterations
  - `931.384 s` solve time
  - worst true relative residual across all linear solves `9.998e-03`
- The step-1 and step-2 tuned `Hypre` rows are retained because they explain where the long full solve comes from.
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "README.md").write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
