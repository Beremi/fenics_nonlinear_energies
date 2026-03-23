#!/usr/bin/env python3
"""Generate a report for the unguarded max-it smoother sweep on L6 P4 deep-tail PMG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l6_p4_deep_p1_tail_unguarded_smoother_sweep_lambda1_np8_maxit20/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l6_p4_deep_p1_tail_unguarded_smoother_sweep_lambda1_np8_maxit20"
)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| variant | Newton reached | steady-state [s] | solve [s] | linear | ls evals | accepted capped | final energy | final grad | worst true rel | message |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['label']}` | {int(row.get('newton_iterations', 0))} | "
            f"{_fmt(float(row.get('steady_state_total_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('solve_time_sec', 0.0)))} | "
            f"{int(row.get('linear_iterations', 0))} | "
            f"{int(row.get('line_search_evals', 0))} | "
            f"{int(row.get('accepted_capped_step_count', 0))} | "
            f"{_fmt(float(row.get('energy', 0.0)), 9)} | "
            f"{_fmt(float(row.get('final_grad_norm', 0.0)), 6)} | "
            f"{_fmt(float(row.get('worst_true_relative_residual', 0.0)), 6)} | "
            f"{str(row.get('message', ''))} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = sorted(json.loads(args.input.read_text(encoding="utf-8")), key=lambda row: str(row["variant"]))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = f"""# `L6` `P4` deep-tail PMG top-smoother sweep with unguarded capped linear steps

Setting:

- problem: `L6`, `P4`, `lambda=1.0`
- hierarchy: `1:1,2:1,3:1,4:1,5:1,6:1,6:2,6:4`
- ranks: `8`
- nonlinear: `armijo`, `maxit=20`, `--no-use_trust_region`
- linear: `fgmres`, `ksp_max_it=15`
- capped linear step policy: `--accept_ksp_maxit_direction --no-guard_ksp_maxit_direction`
- coarse solve: `one-rank LU + broadcast`
- `P2/P1` smoothers fixed at `richardson + sor`, `3` steps

## Results

{_table(rows)}
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
