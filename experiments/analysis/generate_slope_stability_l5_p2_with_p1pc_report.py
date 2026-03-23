"""Generate an L5 comparison report for P2 solves with standard vs refined-P1 preconditioning."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.problems.slope_stability.support import build_case_data, build_refined_p1_case_data


DEFAULT_BASELINE_DIR = Path("artifacts/tmp_l5_p1_vs_p2")
DEFAULT_MIXED_DIR = Path("artifacts/tmp_l5_p2_with_refined_p1_pc")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/reports/slope_stability_level5_lambda1_p2_with_refined_p1_pc_boomeramg"
)


@dataclass(frozen=True)
class GraphStats:
    nodes: int
    elements: int
    free_dofs: int
    adjacency_nnz: int
    avg_row_nnz: float
    max_row_nnz: int


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sum_linear_timing(entries: list[dict], key: str) -> float:
    return float(sum(float(entry.get(key, 0.0)) for entry in entries))


def _extract_result(path: Path) -> dict:
    payload = _load_json(path)
    step = payload["result"]["steps"][0]
    linear_timing = step.get("linear_timing", [])
    solve_time = float(payload["timings"]["solve_time"])
    return {
        "file": str(path),
        "label": str(payload["case"].get("preconditioner_operator", "same_operator")),
        "ranks": int(payload["metadata"]["nprocs"]),
        "nodes": int(payload["mesh"]["nodes"]),
        "elements": int(payload["mesh"]["elements"]),
        "free_dofs": int(payload["mesh"]["free_dofs"]),
        "total_time_sec": float(payload["timings"]["total_time"]),
        "setup_time_sec": float(payload["timings"]["setup_time"]),
        "solve_time_sec": solve_time,
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "avg_linear_iterations_per_newton": float(step["linear_iters"]) / float(step["nit"]),
        "energy": float(step["energy"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "assembly_time_sec": _sum_linear_timing(linear_timing, "assemble_total_time"),
        "pc_operator_assembly_time_sec": _sum_linear_timing(
            linear_timing, "pc_operator_assemble_total_time"
        ),
        "pc_setup_time_sec": _sum_linear_timing(linear_timing, "pc_setup_time"),
        "ksp_time_sec": _sum_linear_timing(linear_timing, "solve_time"),
        "linear_phase_time_sec": _sum_linear_timing(linear_timing, "linear_total_time"),
        "other_solve_time_sec": max(
            0.0, solve_time - _sum_linear_timing(linear_timing, "linear_total_time")
        ),
        "preconditioner_elem_type": payload["metadata"]["linear_solver"].get(
            "preconditioner_elem_type"
        ),
        "preconditioner_elements": payload["metadata"]["linear_solver"].get(
            "preconditioner_elements"
        ),
    }


def _graph_stats(refined_p1: bool) -> GraphStats:
    case_data = (
        build_refined_p1_case_data("ssr_homo_capture_p2_level5")
        if refined_p1
        else build_case_data("ssr_homo_capture_p2_level5")
    )
    adjacency = case_data.adjacency.tocsr()
    row_nnz = adjacency.getnnz(axis=1)
    return GraphStats(
        nodes=int(case_data.nodes.shape[0]),
        elements=int(case_data.elems_scalar.shape[0]),
        free_dofs=int(case_data.freedofs.size),
        adjacency_nnz=int(adjacency.nnz),
        avg_row_nnz=float(np.mean(row_nnz)),
        max_row_nnz=int(np.max(row_nnz)),
    )


def _plot_runtime(results_by_kind: dict[str, list[dict]], output_dir: Path) -> None:
    plt.figure(figsize=(7.0, 4.5))
    for label, rows in results_by_kind.items():
        ranks = [row["ranks"] for row in rows]
        total = [row["total_time_sec"] for row in rows]
        solve = [row["solve_time_sec"] for row in rows]
        plt.plot(ranks, total, marker="o", linewidth=2.0, label=f"{label} total")
        plt.plot(ranks, solve, marker="s", linewidth=1.5, linestyle="--", label=f"{label} solve")
    plt.xlabel("MPI ranks")
    plt.ylabel("Time [s]")
    plt.title("P2 solve: standard vs refined-P1 preconditioner")
    plt.grid(True, alpha=0.3)
    plt.xticks([8, 16, 32])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "runtime_vs_ranks.png", dpi=180)
    plt.close()


def _plot_iterations(results_by_kind: dict[str, list[dict]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for label, rows in results_by_kind.items():
        ranks = [row["ranks"] for row in rows]
        axes[0].plot(
            ranks, [row["newton_iterations"] for row in rows], marker="o", linewidth=2.0, label=label
        )
        axes[1].plot(
            ranks, [row["linear_iterations"] for row in rows], marker="o", linewidth=2.0, label=label
        )
    axes[0].set_title("Newton iterations")
    axes[1].set_title("Linear iterations")
    for ax in axes:
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("Count")
        ax.set_xticks([8, 16, 32])
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "iterations_vs_ranks.png", dpi=180)
    plt.close(fig)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _graph_table(p2_stats: GraphStats, p1_stats: GraphStats) -> str:
    lines = [
        "| graph role | elements | adjacency nnz | avg row nnz | max row nnz |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| P2 operator / standard PC | {p2_stats.elements} | {p2_stats.adjacency_nnz} | {_fmt(p2_stats.avg_row_nnz, 2)} | {p2_stats.max_row_nnz} |",
        f"| refined P1 surrogate PC | {p1_stats.elements} | {p1_stats.adjacency_nnz} | {_fmt(p1_stats.avg_row_nnz, 2)} | {p1_stats.max_row_nnz} |",
    ]
    return "\n".join(lines)


def _runtime_table(results_by_kind: dict[str, list[dict]]) -> str:
    lines = [
        "| preconditioner | ranks | total time [s] | solve time [s] | setup [s] | Newton | linear | avg linear / Newton | energy | omega | u_max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, rows in results_by_kind.items():
        for row in rows:
            lines.append(
                "| "
                f"{label} | {row['ranks']} | {_fmt(row['total_time_sec'])} | {_fmt(row['solve_time_sec'])} | "
                f"{_fmt(row['setup_time_sec'])} | {row['newton_iterations']} | {row['linear_iterations']} | "
                f"{_fmt(row['avg_linear_iterations_per_newton'], 2)} | {_fmt(row['energy'], 6)} | "
                f"{_fmt(row['omega'], 6)} | {_fmt(row['u_max'], 6)} |"
            )
    return "\n".join(lines)


def _time_breakdown_table(results_by_kind: dict[str, list[dict]]) -> str:
    lines = [
        "| preconditioner | ranks | operator assembly [s] | surrogate PC assembly [s] | PC setup [s] | KSP [s] | other solve [s] | linear phase [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, rows in results_by_kind.items():
        for row in rows:
            lines.append(
                "| "
                f"{label} | {row['ranks']} | {_fmt(row['assembly_time_sec'])} | "
                f"{_fmt(row['pc_operator_assembly_time_sec'])} | {_fmt(row['pc_setup_time_sec'])} | "
                f"{_fmt(row['ksp_time_sec'])} | {_fmt(row['other_solve_time_sec'])} | {_fmt(row['linear_phase_time_sec'])} |"
            )
    return "\n".join(lines)


def _ratio_table(baseline_rows: list[dict], mixed_rows: list[dict]) -> str:
    base_by_rank = {row["ranks"]: row for row in baseline_rows}
    mixed_by_rank = {row["ranks"]: row for row in mixed_rows}
    lines = [
        "| ranks | mixed / baseline total time | mixed / baseline solve time | mixed / baseline Newton | mixed / baseline linear |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for ranks in sorted(base_by_rank):
        base = base_by_rank[ranks]
        mixed = mixed_by_rank[ranks]
        lines.append(
            "| "
            f"{ranks} | {_fmt(mixed['total_time_sec'] / base['total_time_sec'], 3)} | "
            f"{_fmt(mixed['solve_time_sec'] / base['solve_time_sec'], 3)} | "
            f"{_fmt(mixed['newton_iterations'] / base['newton_iterations'], 3)} | "
            f"{_fmt(mixed['linear_iterations'] / base['linear_iterations'], 3)} |"
        )
    return "\n".join(lines)


def _write_report(output_dir: Path, results_by_kind: dict[str, list[dict]], p2_stats: GraphStats, p1_stats: GraphStats) -> None:
    baseline = results_by_kind["Standard P2 preconditioner"]
    mixed = results_by_kind["Refined-P1 surrogate preconditioner"]
    report = f"""# L5 lambda=1.0: P2 solve with standard vs refined-P1 preconditioning

This report keeps the nonlinear operator fixed as the `P2` slope-stability Hessian and changes only the preconditioning matrix:

- `Standard P2 preconditioner`: `KSP` sees the same `P2` matrix for both operator and preconditioning.
- `Refined-P1 surrogate preconditioner`: `KSP` uses the `P2` operator matrix but the preconditioning matrix is assembled from the refined `P1` surrogate on the same nodes.

All runs use:

- `lambda = 1.0`
- `pc_type = hypre`, `pc_hypre_type = boomeramg`
- `ksp_type = cg`
- `ksp_rtol = 1e-2`
- `ksp_max_it = 50`
- trust region enabled

## Graphs

{_graph_table(p2_stats, p1_stats)}

The refined `P1` surrogate matrix is much sparser, but that does not automatically make it a better preconditioner for the `P2` operator.

## Results

{_runtime_table(results_by_kind)}

{_ratio_table(baseline, mixed)}

![Runtime vs ranks](runtime_vs_ranks.png)
![Iterations vs ranks](iterations_vs_ranks.png)

## Time breakdown

{_time_breakdown_table(results_by_kind)}

Interpretation:

- The mixed `P2`/`P1` strategy converged to essentially the same `P2` solution, so the surrogate preconditioner is usable.
- It was clearly worse in iteration counts at every rank, and clearly worse in wall time at `16` and `32` ranks.
- The only apparent wall-time win is `8` ranks, where the standard `P2` baseline is unusually expensive per iteration; even there, the mixed preconditioner still needs far more Newton and Krylov work.
- The main reason is effectiveness, not assembly cost: the mixed runs drove Newton from `{baseline[0]['newton_iterations']}-{baseline[-1]['newton_iterations']}` up to `{mixed[0]['newton_iterations']}-{mixed[-1]['newton_iterations']}` steps.
- The linear solver hit the `50`-iteration cap on almost every mixed-PC Newton step, which is why total linear iterations jumped from `{baseline[0]['linear_iterations']}/{baseline[1]['linear_iterations']}/{baseline[2]['linear_iterations']}` to `{mixed[0]['linear_iterations']}/{mixed[1]['linear_iterations']}/{mixed[2]['linear_iterations']}`.
- The surrogate PC matrix is cheaper to assemble per step, but the worse spectral match to the `P2` operator more than cancels that gain.
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--mixed-dir", type=Path, default=DEFAULT_MIXED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_kind = {
        "Standard P2 preconditioner": [
            _extract_result(args.baseline_dir / "l5_p2_np8.json"),
            _extract_result(args.baseline_dir / "l5_p2_np16.json"),
            _extract_result(args.baseline_dir / "l5_p2_np32.json"),
        ],
        "Refined-P1 surrogate preconditioner": [
            _extract_result(args.mixed_dir / "l5_p2_with_refined_p1_pc_np8.json"),
            _extract_result(args.mixed_dir / "l5_p2_with_refined_p1_pc_np16.json"),
            _extract_result(args.mixed_dir / "l5_p2_with_refined_p1_pc_np32.json"),
        ],
    }
    p2_stats = _graph_stats(refined_p1=False)
    p1_stats = _graph_stats(refined_p1=True)

    _plot_runtime(results_by_kind, output_dir)
    _plot_iterations(results_by_kind, output_dir)
    _write_report(output_dir, results_by_kind, p2_stats, p1_stats)

    summary = {
        "results_by_kind": results_by_kind,
        "p2_graph": asdict(p2_stats),
        "refined_p1_graph": asdict(p1_stats),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
