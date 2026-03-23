#!/usr/bin/env python3
"""Generate the staged optimization report for the L6 P4 deep-tail benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l6_p4_deep_p1_tail_optimization_lambda1_np8_maxit20/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l6_p4_deep_p1_tail_optimization_lambda1_np8_maxit20"
)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_bool(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _rows_by_variant(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["variant"]): row for row in rows}


def _guard_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| variant | stage | Newton reached | steady-state [s] | solve [s] | linear | final energy | final grad | worst true rel | accepted capped | guard | buffers |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['label']}` | {row['stage']} | {int(row['newton_iterations'])} | {_fmt(float(row['steady_state_total_time_sec']))} | "
            f"{_fmt(float(row['solve_time_sec']))} | {int(row['linear_iterations'])} | "
            f"{_fmt(float(row['energy']), 9)} | {_fmt(float(row['final_grad_norm']), 6)} | "
            f"{_fmt(float(row['worst_true_relative_residual']), 6)} | "
            f"{int(row['accepted_capped_step_count'])} | {_fmt_bool(bool(row['guard_enabled']))} | "
            f"{_fmt_bool(bool(row['reuse_buffers']))} |"
        )
    return "\n".join(lines)


def _selection_table(selection: dict[str, object], rows: dict[str, dict[str, object]]) -> str:
    final_stack = dict(selection.get("final_stack", {}))
    ordered = [
        ("Initial baseline", str(final_stack.get("baseline", ""))),
        ("Step 1 winner", str(final_stack.get("step1_winner", ""))),
        ("Step 2 winner", str(final_stack.get("step2_winner", ""))),
        ("Step 3 winner", str(final_stack.get("step3_winner", ""))),
        ("Final stack", str(final_stack.get("final", ""))),
    ]
    lines = [
        "| milestone | variant | Newton reached | steady-state [s] | linear | final energy | final grad |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, variant in ordered:
        row = rows.get(variant)
        if row is None:
            continue
        lines.append(
            f"| {label} | `{row['label']}` | {int(row['newton_iterations'])} | {_fmt(float(row['steady_state_total_time_sec']))} | "
            f"{int(row['linear_iterations'])} | {_fmt(float(row['energy']), 9)} | "
            f"{_fmt(float(row['final_grad_norm']), 6)} |"
        )
    return "\n".join(lines)


def _step2_table(
    selection: dict[str, object], rows: dict[str, dict[str, object]]
) -> str:
    step2 = dict(selection.get("step2", {}))
    evals = list(step2.get("evaluations", []))
    lines = [
        "| variant | smoother | Newton reached | steady-state [s] | linear | final energy | final grad | guard ok | improves time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in evals:
        row = rows.get(str(item.get("variant", "")))
        if row is None:
            continue
        lines.append(
            "| "
            f"`{row['label']}` | {row['smoother_desc']} | {int(row['newton_iterations'])} | {_fmt(float(row['steady_state_total_time_sec']))} | "
            f"{int(row['linear_iterations'])} | {_fmt(float(row['energy']), 9)} | "
            f"{_fmt(float(row['final_grad_norm']), 6)} | {_fmt_bool(bool(item.get('guard_ok', False)))} | "
            f"{_fmt_bool(bool(item.get('improves_time', False)))} |"
        )
    return "\n".join(lines)


def _step1_notes(selection: dict[str, object]) -> str:
    step1 = dict(selection.get("step1", {}))
    reasons = list(step1.get("candidate_guard_reasons", []))
    if reasons:
        return "; ".join(str(reason) for reason in reasons)
    return "all end-state guards passed"


def _step3_notes(selection: dict[str, object]) -> str:
    step3 = dict(selection.get("step3", {}))
    reasons = list(step3.get("candidate_guard_reasons", []))
    parts = []
    if reasons:
        parts.append("; ".join(str(reason) for reason in reasons))
    parts.append(
        f"improves stage = {_fmt_bool(bool(step3.get('candidate_improves_stage', False)))}"
    )
    parts.append(
        f"improves time = {_fmt_bool(bool(step3.get('candidate_improves_time', False)))}"
    )
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = dict(json.loads(args.input.read_text(encoding="utf-8")))
    rows = list(payload.get("rows", []))
    selection = dict(payload.get("selection", {}))
    rows_map = _rows_by_variant(rows)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = f"""# `L6` `P4` deep-tail optimization on `8` ranks (`20` Newton iterations)

Fixed benchmark setting:

- problem: `L6`, `P4`, `lambda=1.0`
- hierarchy: `1:1,2:1,3:1,4:1,5:1,6:1,6:2,6:4`
- ranks: `8`
- nonlinear: `armijo`, `maxit=20`, `--no-use_trust_region`
- linear: `fgmres`, `ksp_max_it=15`
- coarse solve: `one-rank LU + broadcast`
- `P2/P1` smoothers fixed at `richardson + sor`, `3` steps

## Accepted Stack

{_selection_table(selection, rows_map)}

## Step 1: Guarded capped-direction acceptance

- winner: `{selection.get('step1', {}).get('winner', '')}`
- notes: {_step1_notes(selection)}

## Step 2: top `P4` smoother sweep

- winner: `{selection.get('step2', {}).get('winner', '')}`

{_step2_table(selection, rows_map)}

## Step 3: Hessian buffer reuse

- winner: `{selection.get('step3', {}).get('winner', '')}`
- notes: {_step3_notes(selection)}

## All Rows

{_guard_table(rows)}
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
