#!/usr/bin/env python3
"""Generate a report for mixed-order slope-stability PCMG strategy benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("artifacts/raw_results/slope_stability_hp_strategy_bench_level4_lambda1_v2/summary.json")
DEFAULT_OUTPUT = Path("artifacts/reports/slope_stability_hp_strategy_bench_level4_lambda1_v2")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _successful(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if bool(row.get("solver_success"))]


def _rows_for_degree(rows: list[dict[str, object]], degree: int) -> list[dict[str, object]]:
    return sorted(
        [row for row in rows if int(row["elem_degree"]) == int(degree)],
        key=lambda row: (str(row["strategy"]), int(row["ranks"])),
    )


def _hypre_strategy_name(degree: int) -> str:
    return f"p{int(degree)}_hypre_boomeramg"


def _plot_degree(rows: list[dict[str, object]], *, degree: int, output_dir: Path) -> None:
    plt.figure(figsize=(8.0, 4.8))
    strategies = sorted({str(row["strategy"]) for row in rows})
    for strategy in strategies:
        subset = sorted(
            [row for row in rows if str(row["strategy"]) == strategy and bool(row["solver_success"])],
            key=lambda row: int(row["ranks"]),
        )
        if not subset:
            continue
        plt.plot(
            [row["ranks"] for row in subset],
            [row["solve_time_sec"] for row in subset],
            marker="o",
            linewidth=2.0,
            label=str(strategy),
        )
    plt.xlabel("MPI ranks")
    plt.ylabel("Solve time [s]")
    plt.title(f"L4 lambda=1.0 P{degree} mixed-order PCMG strategies")
    plt.xscale("log", base=2)
    plt.xticks([1, 8, 16, 32], ["1", "8", "16", "32"])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"p{degree}_solve_time_vs_ranks.png", dpi=180)
    plt.close()


def _best_table(rows: list[dict[str, object]], *, degree: int) -> str:
    subset = _rows_for_degree(rows, degree)
    ranks = sorted({int(row["ranks"]) for row in subset})
    lines = [
        "| ranks | best PCMG strategy | PCMG solve [s] | Hypre solve [s] | PCMG/Hypre | Newton | linear |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ranks_i in ranks:
        candidates = [
            row
            for row in subset
            if (
                int(row["ranks"]) == ranks_i
                and bool(row["solver_success"])
                and str(row["strategy"]) != _hypre_strategy_name(degree)
            )
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: float(row["solve_time_sec"]))
        hypre_rows = [
            row
            for row in subset
            if int(row["ranks"]) == ranks_i and str(row["strategy"]) == _hypre_strategy_name(degree)
        ]
        hypre_solve = float(hypre_rows[0]["solve_time_sec"]) if hypre_rows else float("nan")
        ratio = float(best["solve_time_sec"]) / hypre_solve if hypre_rows and hypre_solve > 0.0 else float("nan")
        lines.append(
            "| "
            f"{ranks_i} | {best['strategy']} | {_fmt(float(best['solve_time_sec']))} | "
            f"{_fmt(hypre_solve)} | {_fmt(ratio)} | {best['newton_iterations']} | {best['linear_iterations']} |"
        )
    return "\n".join(lines)


def _full_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| degree | strategy | pc | ranks | success | solve [s] | total [s] | Newton | linear | omega | u_max |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: (int(r["elem_degree"]), str(r["strategy"]), int(r["ranks"]))):
        lines.append(
            "| "
            f"{row['elem_degree']} | {row['strategy']} | {row.get('pc_type', 'mg')} | {row['ranks']} | {row['solver_success']} | "
            f"{_fmt(float(row.get('solve_time_sec', float('nan'))))} | "
            f"{_fmt(float(row.get('total_time_sec', float('nan'))))} | "
            f"{row.get('newton_iterations', '-')} | {row.get('linear_iterations', '-')} | "
            f"{_fmt(float(row.get('omega', float('nan'))), 6)} | {_fmt(float(row.get('u_max', float('nan'))), 6)} |"
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

    for degree in (2, 4):
        _plot_degree(_rows_for_degree(rows, degree), degree=degree, output_dir=output_dir)

    report = f"""# L4 lambda=1.0 mixed-order PCMG strategy benchmark

This benchmark compares same-mesh `P1/P2/P4` multigrid ladders on the slope-stability endpoint:

- finest geometric level: `L4`
- lambda: `1.0`
- nonlinear solve: `--no-use_trust_region`
- linear solve: `fgmres + pc_type=mg`
- the deepest geometric coarsening allowed is one extra `P1` level at `L3`

The benchmark was intentionally moved to `L4` for the `P4` comparisons. A pilot `L5 P4` run reached roughly `3.7 GB` RSS per rank at `8` ranks, so it was too expensive for a strategy sweep.

## Best P2 Strategy By Rank

{_best_table(rows, degree=2)}

## Best P4 Strategy By Rank

{_best_table(rows, degree=4)}

## Full Table

{_full_table(rows)}

![P2 solve time vs ranks](p2_solve_time_vs_ranks.png)
![P4 solve time vs ranks](p4_solve_time_vs_ranks.png)
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
