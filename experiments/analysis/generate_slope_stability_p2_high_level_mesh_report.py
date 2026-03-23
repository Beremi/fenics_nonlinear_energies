#!/usr/bin/env python3
"""Generate a report for the higher-level P2 mesh sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("artifacts/raw_results/slope_stability_p2_high_level_mesh_sweep_lambda1_np16/summary.json")
DEFAULT_OUTPUT = Path("artifacts/reports/slope_stability_p2_high_level_mesh_sweep_lambda1_np16")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _rows(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    return sorted([row for row in rows if str(row["method"]) == method], key=lambda row: int(row["level"]))


def _plot(rows: list[dict[str, object]], output_dir: Path) -> None:
    plt.figure(figsize=(8.0, 4.8))
    for method in sorted({str(row["method"]) for row in rows}):
        subset = _rows(rows, method)
        plt.plot(
            [row["level"] for row in subset],
            [row["solve_time_sec"] for row in subset],
            marker="o",
            linewidth=2.0,
            label=str(method),
        )
    plt.xlabel("Mesh level")
    plt.ylabel("Solve time [s]")
    plt.title("P2 higher-level mesh sweep at 16 MPI ranks")
    plt.xticks([4, 5, 6], ["L4", "L5", "L6"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "solve_time_vs_level.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.0, 4.8))
    for method in sorted({str(row["method"]) for row in rows}):
        subset = _rows(rows, method)
        plt.plot(
            [row["level"] for row in subset],
            [row["linear_iterations"] for row in subset],
            marker="o",
            linewidth=2.0,
            label=str(method),
        )
    plt.xlabel("Mesh level")
    plt.ylabel("Linear iterations")
    plt.title("P2 higher-level mesh sweep iterations at 16 MPI ranks")
    plt.xticks([4, 5, 6], ["L4", "L5", "L6"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "linear_iterations_vs_level.png", dpi=180)
    plt.close()


def _best_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| level | free DOFs | best MG method | MG solve [s] | Hypre solve [s] | MG/Hypre |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for level in sorted({int(row["level"]) for row in rows}):
        level_rows = [row for row in rows if int(row["level"]) == level]
        mg_rows = [row for row in level_rows if str(row["pc_type"]) == "mg" and bool(row["solver_success"])]
        hypre_rows = [row for row in level_rows if str(row["pc_type"]) == "hypre" and bool(row["solver_success"])]
        if not mg_rows or not hypre_rows:
            continue
        best = min(mg_rows, key=lambda row: float(row["solve_time_sec"]))
        hypre = hypre_rows[0]
        ratio = float(best["solve_time_sec"]) / float(hypre["solve_time_sec"])
        lines.append(
            "| "
            f"L{level} | {best['free_dofs']} | {best['method']} | {_fmt(float(best['solve_time_sec']))} | "
            f"{_fmt(float(hypre['solve_time_sec']))} | {_fmt(ratio)} |"
        )
    return "\n".join(lines)


def _full_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| level | method | pc | solve [s] | total [s] | Newton | linear | free DOFs | omega | u_max |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: (int(r["level"]), str(r["method"]))):
        lines.append(
            "| "
            f"L{row['level']} | {row['method']} | {row['pc_type']} | {_fmt(float(row['solve_time_sec']))} | "
            f"{_fmt(float(row['total_time_sec']))} | {row['newton_iterations']} | {row['linear_iterations']} | "
            f"{row['free_dofs']} | {_fmt(float(row['omega']), 6)} | {_fmt(float(row['u_max']), 6)} |"
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

    _plot(rows, output_dir)

    report = f"""# Higher-level P2 mesh sweep at 16 MPI ranks

This sweep compares the stronger `P2` PCMG candidates against `Hypre/BoomerAMG`
on the finer slope-stability meshes:

- element degree: `P2`
- levels: `L4`, `L5`, `L6`
- ranks: `16`
- lambda: `1.0`
- nonlinear solve: `--no-use_trust_region`

## Best MG vs Hypre

{_best_table(rows)}

## Full Table

{_full_table(rows)}

![Solve time vs level](solve_time_vs_level.png)
![Linear iterations vs level](linear_iterations_vs_level.png)
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
