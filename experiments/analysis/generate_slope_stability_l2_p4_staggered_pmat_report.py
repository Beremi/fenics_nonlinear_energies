#!/usr/bin/env python3
"""Report the L2 P4 staggered-fine-Pmat benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l2_p4_staggered_pmat_bench_lambda1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l2_p4_staggered_pmat_bench_lambda1"
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
    payload = json.loads(Path(str(row["result_json"])).read_text(encoding="utf-8"))
    return list(payload["result"]["steps"][0].get("linear_timing", []))


def _newton_tables(rows: list[dict[str, object]]) -> str:
    sections: list[str] = []
    for row in rows:
        linear_timing = _load_linear_timing(row)
        iter_rows = [
            {
                "newton_it": idx + 1,
                "ksp_its": int(record.get("ksp_its", 0)),
                "fine_pmat_updated": bool(record.get("fine_pmat_updated_this_step", False)),
                "fine_pmat_step_assembly_time": float(record.get("fine_pmat_step_assembly_time", 0.0)),
                "pc_operator_assemble_total_time": float(
                    sum(
                        float(level.get("assembly_total_time", 0.0))
                        for level in record.get("pc_operator_mg_level_records", [])
                    )
                ),
                "true_relative_residual": float(record.get("true_relative_residual", 0.0)),
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
                            ("fine Pmat update", "fine_pmat_updated"),
                            ("fine Pmat asm [s]", "fine_pmat_step_assembly_time"),
                            ("rest PC asm [s]", "pc_operator_assemble_total_time"),
                            ("true rel", "true_relative_residual"),
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

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = f"""# `L2` Staggered Fine-`Pmat` Matrix-Free `P4` Benchmark

This report compares the assembled legacy baseline against staggered every-other-Newton-step schemes on `L2`, `lambda=1.0`, same-mesh `P4 -> P2 -> P1`:

- `matfree_legacy_pmg_staggered_whole`: whole fine-preconditioner path staggered, current matrix-free `P4` matvec every step
- `matfree_explicit_pmg_staggered_smoother_only_fixed`: only the fine smoother matrix is staggered; lower `P2/P1` levels stay fixed
- `matfree_explicit_pmg_staggered_smoother_only_refresh_attempt`: attempted fine-smoother stagger with refreshed lower `P2/P1` levels; this is kept as a documented failed PETSc setup path

## Summary

{_table(rows, [
    ("case", "name"),
    ("success", "solver_success"),
    ("Newton", "newton_iterations"),
    ("linear", "linear_iterations"),
    ("worst true rel", "worst_true_relative_residual"),
    ("solve [s]", "solve_time_sec"),
    ("operator prep [s]", "operator_prepare_time_sec"),
    ("fine Pmat asm [s]", "fine_pmat_step_assembly_time_sec"),
    ("rest PC asm [s]", "lower_level_assembly_time_sec"),
])}

## Stagger Pattern

{_table(rows, [
    ("case", "name"),
    ("policy", "fine_pmat_policy"),
    ("cadence", "fine_pmat_stagger_period"),
    ("update steps", "fine_pmat_update_steps"),
    ("reuse steps", "fine_pmat_reuse_steps"),
    ("fine op asm zero", "fine_p4_operator_assembly_zero"),
])}

## Per-Newton-Step Tables

{_newton_tables(rows)}
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
