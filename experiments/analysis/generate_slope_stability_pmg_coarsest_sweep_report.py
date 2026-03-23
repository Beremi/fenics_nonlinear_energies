"""Generate a report for the L5 PETSc multigrid coarsest-level sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("artifacts/raw_results/slope_stability_pmg_coarsest_sweep_lambda1/summary.json")
DEFAULT_OUTPUT = Path("artifacts/reports/slope_stability_pmg_coarsest_sweep_lambda1")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _rows_by_coarsest(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for row in rows:
        out.setdefault(int(row["coarsest_level"]), []).append(row)
    for key in out:
        out[key].sort(key=lambda row: int(row["ranks"]))
    return dict(sorted(out.items()))


def _plot(rows_by_coarsest: dict[int, list[dict]], output_dir: Path) -> None:
    plt.figure(figsize=(8.0, 4.8))
    for c, rows in rows_by_coarsest.items():
        plt.plot(
            [row["ranks"] for row in rows],
            [row["solve_time_sec"] for row in rows],
            marker="o",
            linewidth=2.0,
            label=f"coarsest L{c}",
        )
    plt.xlabel("MPI ranks")
    plt.ylabel("Solve time [s]")
    plt.title("L5 lambda=1.0 PCMG solve time vs coarsest level")
    plt.xscale("log", base=2)
    plt.xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "solve_time_vs_ranks.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.0, 4.8))
    for c, rows in rows_by_coarsest.items():
        plt.plot(
            [row["ranks"] for row in rows],
            [row["linear_iterations"] for row in rows],
            marker="o",
            linewidth=2.0,
            label=f"coarsest L{c}",
        )
    plt.xlabel("MPI ranks")
    plt.ylabel("Linear iterations")
    plt.title("L5 lambda=1.0 PCMG linear iterations")
    plt.xscale("log", base=2)
    plt.xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "linear_iterations_vs_ranks.png", dpi=180)
    plt.close()


def _summary_table(rows_by_coarsest: dict[int, list[dict]]) -> str:
    lines = [
        "| coarsest | ranks | success | total [s] | solve [s] | Newton | linear | energy | omega | u_max |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c, rows in rows_by_coarsest.items():
        for row in rows:
            lines.append(
                "| "
                f"L{c} | {row['ranks']} | {row['solver_success']} | {_fmt(row['total_time_sec'])} | "
                f"{_fmt(row['solve_time_sec'])} | {row['newton_iterations']} | {row['linear_iterations']} | "
                f"{_fmt(row['energy'], 6)} | {_fmt(row['omega'], 6)} | {_fmt(row['u_max'], 6)} |"
            )
    return "\n".join(lines)


def _best_table(rows_by_coarsest: dict[int, list[dict]]) -> str:
    lines = [
        "| ranks | best coarsest by solve time | solve time [s] | linear iters |",
        "| ---: | --- | ---: | ---: |",
    ]
    by_rank: dict[int, list[dict]] = {}
    for rows in rows_by_coarsest.values():
        for row in rows:
            by_rank.setdefault(int(row["ranks"]), []).append(row)
    for rank in sorted(by_rank):
        best = min(by_rank[rank], key=lambda row: float(row["solve_time_sec"]))
        lines.append(
            "| "
            f"{rank} | L{best['coarsest_level']} | {_fmt(best['solve_time_sec'])} | {best['linear_iterations']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text())
    rows_by_coarsest = _rows_by_coarsest(rows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot(rows_by_coarsest, output_dir)

    report = f"""# L5 lambda=1.0 PCMG coarsest-level sweep

This sweep keeps the tuned PETSc multigrid configuration fixed and varies only the coarsest geometric level:

- finest level: `L5`
- coarsest level: `L1`, `L2`, `L3`, `L4`
- ranks: `1, 2, 4, 8, 16, 32`
- outer Krylov: `fgmres`
- `pc_type = mg`
- nonlinear globalization: `--no-use_trust_region`

## Best Per Rank

{_best_table(rows_by_coarsest)}

## Full Table

{_summary_table(rows_by_coarsest)}

![Solve time vs ranks](solve_time_vs_ranks.png)
![Linear iterations vs ranks](linear_iterations_vs_ranks.png)
"""
    (output_dir / "report.md").write_text(report)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
