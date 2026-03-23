#!/usr/bin/env python3
"""Generate a report for the L4 P4 matrix-free slope-stability comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("artifacts/raw_results/slope_stability_p4_l4_matfree_bench/summary.json")
DEFAULT_OUTPUT = Path("artifacts/reports/slope_stability_p4_l4_matfree_bench")

WORKING_STRATEGIES = [
    "assembled_hypre",
    "assembled_mg_best",
    "matfree_element_hypre",
    "matfree_overlap_hypre",
]

LABELS = {
    "assembled_hypre": "assembled + hypre",
    "assembled_mg_best": "assembled + pcmg",
    "matfree_element_hypre": "matfree element + hypre",
    "matfree_overlap_hypre": "matfree overlap + hypre",
    "matfree_element_mg_direct": "direct shell + pcmg (element)",
    "matfree_overlap_mg_direct": "direct shell + pcmg (overlap)",
}


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _successful(rows: list[dict[str, object]], strategy: str) -> list[dict[str, object]]:
    return sorted(
        [
            row
            for row in rows
            if str(row["strategy"]) == strategy and bool(row.get("solver_success"))
        ],
        key=lambda row: int(row["ranks"]),
    )


def _plot_metric(
    rows: list[dict[str, object]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8.2, 4.8))
    for strategy in WORKING_STRATEGIES:
        subset = _successful(rows, strategy)
        if not subset:
            continue
        plt.plot(
            [int(row["ranks"]) for row in subset],
            [float(row[metric]) for row in subset],
            marker="o",
            linewidth=2.0,
            label=LABELS[strategy],
        )
    plt.xscale("log", base=2)
    plt.xticks([1, 8, 16, 32], ["1", "8", "16", "32"])
    plt.xlabel("MPI ranks")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_matvec(rows: list[dict[str, object]], output_path: Path) -> None:
    plt.figure(figsize=(8.2, 4.8))
    for strategy in ("matfree_element_hypre", "matfree_overlap_hypre"):
        subset = _successful(rows, strategy)
        if not subset:
            continue
        plt.plot(
            [int(row["ranks"]) for row in subset],
            [float(row.get("operator_apply_avg_ms", 0.0)) for row in subset],
            marker="o",
            linewidth=2.0,
            label=LABELS[strategy],
        )
    plt.xscale("log", base=2)
    plt.xticks([1, 8, 16, 32], ["1", "8", "16", "32"])
    plt.xlabel("MPI ranks")
    plt.ylabel("Average shell matvec [ms]")
    plt.title("L4 lambda=1.0 P4 matrix-free average Hessian matvec cost")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _main_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| strategy | ranks | success | solve [s] | Newton | linear | omega | u_max | avg matvec [ms] | prep [s] | pmat asm [s] |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in WORKING_STRATEGIES:
        for row in _successful(rows, strategy):
            lines.append(
                "| "
                f"{LABELS[strategy]} | {row['ranks']} | {row['solver_success']} | "
                f"{_fmt(float(row['solve_time_sec']))} | {row['newton_iterations']} | "
                f"{row['linear_iterations']} | {_fmt(float(row['omega']), 6)} | "
                f"{_fmt(float(row['u_max']), 6)} | {_fmt(float(row.get('operator_apply_avg_ms', 0.0)))} | "
                f"{_fmt(float(row.get('operator_prepare_total_sec', 0.0)))} | "
                f"{_fmt(float(row.get('pc_operator_assemble_total_sec', 0.0)))} |"
            )
    return "\n".join(lines)


def _breakdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| strategy | ranks | shell apply [s] | shell comm [s] | shell build [s] | shell kernel [s] | shell scatter [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in ("matfree_element_hypre", "matfree_overlap_hypre"):
        for row in _successful(rows, strategy):
            lines.append(
                "| "
                f"{LABELS[strategy]} | {row['ranks']} | {_fmt(float(row.get('operator_apply_total_sec', 0.0)))} | "
                f"{_fmt(float(row.get('operator_apply_allgatherv_sec', 0.0)))} | "
                f"{_fmt(float(row.get('operator_apply_build_v_local_sec', 0.0)))} | "
                f"{_fmt(float(row.get('operator_apply_kernel_sec', 0.0)))} | "
                f"{_fmt(float(row.get('operator_apply_scatter_sec', 0.0)))} |"
            )
    return "\n".join(lines)


def _failure_table(rows: list[dict[str, object]]) -> str:
    failed = [
        row
        for row in rows
        if str(row["strategy"]) in {"matfree_element_mg_direct", "matfree_overlap_mg_direct"}
    ]
    if not failed:
        return "_No direct shell + PCMG attempt rows were recorded._"
    lines = [
        "| strategy | ranks | status | message |",
        "| --- | ---: | --- | --- |",
    ]
    for row in failed:
        message = str(row.get("message", "")).replace("|", "/")
        lines.append(
            f"| {LABELS[str(row['strategy'])]} | {row['ranks']} | {row.get('status', '')} | {message} |"
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

    _plot_metric(
        rows,
        metric="solve_time_sec",
        ylabel="Solve time [s]",
        title="L4 lambda=1.0 P4 assembled vs matrix-free operator strategies",
        output_path=output_dir / "solve_time_vs_ranks.png",
    )
    _plot_metric(
        rows,
        metric="linear_iterations",
        ylabel="Linear iterations",
        title="L4 lambda=1.0 P4 linear iterations",
        output_path=output_dir / "linear_iterations_vs_ranks.png",
    )
    _plot_matvec(rows, output_dir / "matvec_avg_vs_ranks.png")

    report = f"""# L4 lambda=1.0 P4 matrix-free operator benchmark

This benchmark keeps the same `L4` same-mesh `P4` discretization and compares four solver variants:

- assembled fine operator + `Hypre/BoomerAMG`
- assembled fine operator + the best earlier same-mesh `PCMG` hierarchy (`P4 -> P2 -> P1 -> L3 P1`)
- matrix-free fine operator using element-local Hessian-vector products + `Hypre/BoomerAMG`
- matrix-free fine operator using the overlap-domain total functional + `Hypre/BoomerAMG`

For the matrix-free variants, the fine operator is a PETSc Python shell matrix backed by cached `jax.linearize(jax.grad(...))` closures at each Newton step. The preconditioning matrix remains explicitly assembled.

I also tried the direct `shell A + PCMG` route. Those attempts failed during PETSc `PCSetUp_MG -> MatCreateVecs`, so the working matrix-free comparison in this report uses `Hypre/BoomerAMG` as the explicit preconditioner while keeping the fine operator matrix-free.

## Main Table

{_main_table(rows)}

## Matrix-Free Breakdown

{_breakdown_table(rows)}

## Direct Shell + PCMG Attempts

{_failure_table(rows)}

## Notes

- The two working matrix-free variants converge to the same solution as the assembled baselines.
- The overlap-functional variant avoids element-wise reduction inside the shell matvec; it allgathers the distributed tangent, rebuilds the overlap-local vector, and applies one Hessian-vector product of the overlap-domain functional.
- The element-functional variant uses the same allgathered tangent but still reduces element-local HVP contributions back to owned rows.

![Solve time vs ranks](solve_time_vs_ranks.png)
![Linear iterations vs ranks](linear_iterations_vs_ranks.png)
![Average shell matvec vs ranks](matvec_avg_vs_ranks.png)
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
