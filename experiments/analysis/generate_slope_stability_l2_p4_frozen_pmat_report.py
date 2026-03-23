#!/usr/bin/env python3
"""Report the L2 P4 frozen-fine-Pmat legacy-PCMG benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l2_p4_frozen_pmat_bench_lambda1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l2_p4_frozen_pmat_bench_lambda1"
)


def _fmt(value: float, digits: int = 3) -> str:
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
                vals.append(_fmt(value, 6 if "residual" in key else 3))
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _load_linear_timing(row: dict[str, object]) -> list[dict[str, object]]:
    result_json = row.get("result_json")
    if not result_json:
        return []
    payload = json.loads(Path(str(result_json)).read_text(encoding="utf-8"))
    steps = list(payload.get("result", {}).get("steps", []))
    if not steps:
        return []
    return list(steps[0].get("linear_timing", []))


def _newton_tables(rows: list[dict[str, object]]) -> str:
    sections: list[str] = []
    for row in rows:
        linear_timing = _load_linear_timing(row)
        if not linear_timing:
            sections.append(f"### `{row['name']}`\n\n_No per-Newton-step timing was recorded._")
            continue
        iter_rows = [
            {
                "newton_it": idx + 1,
                "ksp_its": int(record.get("ksp_its", 0)),
                "ksp_reason_name": str(record.get("ksp_reason_name", "")),
                "true_relative_residual": float(record.get("true_relative_residual", 0.0)),
                "linear_total_time": float(record.get("linear_total_time", 0.0)),
            }
            for idx, record in enumerate(linear_timing)
        ]
        sections.append(
            "\n".join(
                [
                    f"### `{row['name']}`",
                    "",
                    _table(
                        iter_rows,
                        [
                            ("Newton it", "newton_it"),
                            ("KSP its", "ksp_its"),
                            ("KSP reason", "ksp_reason_name"),
                            ("true rel", "true_relative_residual"),
                            ("linear total [s]", "linear_total_time"),
                        ],
                    ),
                ]
            )
        )
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = json.loads(args.input.read_text(encoding="utf-8"))
    rows = list(summary["rows"])
    matrix_comparison = dict(summary["matrix_comparison"])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = f"""# `L2` Hybrid Matrix-Free `P4` With Frozen Fine `Pmat`

This report compares the assembled legacy `PCMG` baseline against two split-operator variants on the same `L2`, `lambda=1.0`, `P4 -> P2 -> P1` hierarchy:

- `baseline_assembled_legacy_full`: assembled `P4` operator and legacy `PCMG`
- `matfree_legacy_pmg_elastic_frozen`: matrix-free `P4` operator with a one-time frozen elastic fine `Pmat`
- `matfree_legacy_pmg_initial_tangent_frozen`: matrix-free `P4` operator with a one-time frozen initial-tangent fine `Pmat`

## Benchmark Summary

{_table(rows, [
    ("case", "name"),
    ("success", "solver_success"),
    ("Newton", "newton_iterations"),
    ("linear", "linear_iterations"),
    ("worst true rel", "worst_true_relative_residual"),
    ("solve [s]", "solve_time_sec"),
    ("setup [s]", "setup_time_sec"),
    ("fine Pmat setup [s]", "fine_pmat_setup_assembly_time_sec"),
    ("fine Pmat step [s]", "fine_pmat_step_assembly_time_sec"),
])}

## Timing And Operator Notes

{_table(rows, [
    ("case", "name"),
    ("operator", "operator_mode"),
    ("fine Pmat policy", "fine_pmat_policy"),
    ("fine Pmat source", "fine_pmat_source"),
    ("operator prep [s]", "operator_prepare_time_sec"),
    ("operator apply [s]", "operator_apply_time_sec"),
    ("KSP solve [s]", "ksp_solve_time_sec"),
    ("fine op asm zero", "fine_p4_operator_assembly_zero"),
    ("fine Pmat asm zero", "fine_pmat_step_assembly_zero"),
])}

## Per-Newton-Step Linear Iterations

{_newton_tables(rows)}

## Elastic vs Initial-Tangent Frozen Matrix Check

- same sparsity pattern: `{matrix_comparison["same_pattern"]}`
- relative matrix difference: `{matrix_comparison["relative_difference"]:.6e}`

{"The elastic and initial-tangent frozen `P4` matrices are effectively identical on this case." if bool(matrix_comparison["same_pattern"]) and float(matrix_comparison["relative_difference"]) <= 1.0e-12 else "The elastic and initial-tangent frozen `P4` matrices differ measurably on this case."}
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
