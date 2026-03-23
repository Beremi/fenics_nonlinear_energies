#!/usr/bin/env python3
"""Generate the L6 P4 skipped-tail comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l6_p4_skipped_tail_compare_lambda1_np8_maxit20/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l6_p4_skipped_tail_compare_lambda1_np8_maxit20"
)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _classify(candidate: dict[str, object], baseline: dict[str, object]) -> str:
    energy_ok = abs(float(candidate["energy"]) - float(baseline["energy"])) <= 1.0e-6
    grad_ok = float(candidate["final_grad_norm"]) <= 1.2 * float(baseline["final_grad_norm"])
    linear_ok = float(candidate["linear_iterations"]) <= 1.1 * float(baseline["linear_iterations"])
    comparable = energy_ok and grad_ok and linear_ok
    faster = float(candidate["steady_state_total_time_sec"]) < float(
        baseline["steady_state_total_time_sec"]
    )
    if faster and comparable:
        return "faster and comparable"
    if faster and not comparable:
        return "faster but degraded"
    if (not faster) and comparable:
        return "slower but comparable"
    return "slower and degraded"


def _smoke_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| smoke case | hierarchy | status | message |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['label']}` | `{row['mg_custom_hierarchy']}` | {row['status']} | {row['message']} |"
        )
    return "\n".join(lines)


def _main_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| case | hierarchy | Newton reached | steady-state [s] | solve [s] | ls time [s] | ls evals | linear | accepted capped | energy | final grad | worst true rel | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['label']}` | `{row['mg_custom_hierarchy']}` | {int(row['newton_iterations'])} | "
            f"{_fmt(float(row['steady_state_total_time_sec']))} | {_fmt(float(row['solve_time_sec']))} | "
            f"{_fmt(float(row['line_search_time_sec']))} | {int(row['line_search_evals'])} | "
            f"{int(row['linear_iterations'])} | {int(row['accepted_capped_step_count'])} | "
            f"{_fmt(float(row['energy']), 9)} | {_fmt(float(row['final_grad_norm']), 6)} | "
            f"{_fmt(float(row['worst_true_relative_residual']), 6)} | {row['message']} |"
        )
    return "\n".join(lines)


def _delta_table(candidate: dict[str, object], baseline: dict[str, object]) -> str:
    specs = [
        ("steady-state [s]", "steady_state_total_time_sec"),
        ("solve [s]", "solve_time_sec"),
        ("ls time [s]", "line_search_time_sec"),
        ("ls evals", "line_search_evals"),
        ("linear iters", "linear_iterations"),
        ("accepted capped", "accepted_capped_step_count"),
        ("energy", "energy"),
        ("final grad", "final_grad_norm"),
        ("worst true rel", "worst_true_relative_residual"),
    ]
    lines = [
        "| metric | baseline | candidate | delta candidate-baseline |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in specs:
        b = float(baseline[key])
        c = float(candidate[key])
        digits = 9 if key == "energy" else 6 if "grad" in key or "rel" in key else 3
        lines.append(
            f"| {label} | {_fmt(b, digits)} | {_fmt(c, digits)} | {_fmt(c - b, digits)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = dict(json.loads(args.input.read_text(encoding="utf-8")))
    smoke_rows = list(payload.get("smoke", []))
    rows = list(payload.get("rows", []))
    by_name = {str(row["name"]): row for row in rows}
    baseline = by_name["baseline_full_p1_tail"]
    candidate = by_name["candidate_skip_intermediate_p1"]
    classification = _classify(candidate, baseline)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = f"""# `L6/P4` skipped-tail hierarchy comparison on `8` ranks

Fixed benchmark setting:

- problem: `L6`, `P4`, `lambda=1.0`
- ranks: `8`
- nonlinear: `armijo`, `maxit=20`, `--no-use_trust_region`
- linear: `fgmres`, `ksp_max_it=15`
- capped linear solve handling: guarded `accept_ksp_maxit_direction`, true-rel cap `6e-2`
- coarse solve: `rank0_lu_broadcast`
- smoothers: `P4/P2/P1 = richardson + sor`, `3` steps
- benchmark mode: `warmup_once_then_solve`
- Hessian buffer mode: `--no-reuse_hessian_value_buffers`

## Smoke Check

{_smoke_table(smoke_rows)}

## Comparison

{_main_table(rows)}

## Delta

{_delta_table(candidate, baseline)}

## Conclusion

- Classification: **{classification}**
- Baseline hierarchy: `{baseline['mg_custom_hierarchy']}`
- Candidate hierarchy: `{candidate['mg_custom_hierarchy']}`
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
