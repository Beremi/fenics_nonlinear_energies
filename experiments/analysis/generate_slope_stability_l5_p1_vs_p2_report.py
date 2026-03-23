"""Generate a markdown comparison for L5 P2 vs refined-P1-on-same-nodes solves."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.problems.slope_stability.support import (
    build_case_data,
    build_refined_p1_case_data,
)


DEFAULT_INPUT_DIR = Path("artifacts/tmp_l5_p1_vs_p2")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/slope_stability_level5_lambda1_p2_vs_refined_p1_boomeramg")


@dataclass(frozen=True)
class GraphStats:
    nodes: int
    elements: int
    free_dofs: int
    adjacency_nnz: int
    avg_row_nnz: float
    max_row_nnz: int
    min_row_nnz: int


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sum_linear_timing(entries: list[dict], key: str) -> float:
    return float(sum(float(entry.get(key, 0.0)) for entry in entries))


def _extract_result(path: Path) -> dict:
    payload = _load_json(path)
    step = payload["result"]["steps"][0]
    linear_timing = step.get("linear_timing", [])
    solve_time = float(payload["timings"]["solve_time"])
    linear_total = _sum_linear_timing(linear_timing, "linear_total_time")
    return {
        "file": str(path),
        "elem_type": str(payload["case"]["elem_type"]),
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
        "pc_setup_time_sec": _sum_linear_timing(linear_timing, "pc_setup_time"),
        "ksp_time_sec": _sum_linear_timing(linear_timing, "solve_time"),
        "linear_phase_time_sec": linear_total,
        "other_solve_time_sec": max(0.0, solve_time - linear_total),
    }


def _graph_stats(case_name: str, refined_p1: bool) -> GraphStats:
    case_data = (
        build_refined_p1_case_data(case_name=case_name)
        if refined_p1
        else build_case_data(case_name=case_name)
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
        min_row_nnz=int(np.min(row_nnz)),
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
    plt.title("L5 lambda=1.0 BoomerAMG runtime")
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
        newton = [row["newton_iterations"] for row in rows]
        linear = [row["linear_iterations"] for row in rows]
        axes[0].plot(ranks, newton, marker="o", linewidth=2.0, label=label)
        axes[1].plot(ranks, linear, marker="o", linewidth=2.0, label=label)
    axes[0].set_title("Newton iterations")
    axes[1].set_title("Linear iterations")
    for ax in axes:
        ax.set_xlabel("MPI ranks")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([8, 16, 32])
        ax.legend()
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "iterations_vs_ranks.png", dpi=180)
    plt.close(fig)


def _plot_graph_stats(graph_stats: dict[str, GraphStats], output_dir: Path) -> None:
    labels = list(graph_stats)
    avg_row_nnz = [graph_stats[label].avg_row_nnz for label in labels]
    total_nnz = [graph_stats[label].adjacency_nnz for label in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    axes[0].bar(x, avg_row_nnz, color=["#365c8d", "#4c9f70"])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Average row nnz")
    axes[0].set_title("Graph density")
    axes[1].bar(x, total_nnz, color=["#365c8d", "#4c9f70"])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Adjacency nnz")
    axes[1].set_title("Total graph nnz")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "graph_complexity.png", dpi=180)
    plt.close(fig)


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _table_graph_stats(graph_stats: dict[str, GraphStats]) -> str:
    lines = [
        "| discretization | nodes | elements | free dofs | adjacency nnz | avg row nnz | max row nnz |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, stats in graph_stats.items():
        lines.append(
            "| "
            f"{label} | {stats.nodes} | {stats.elements} | {stats.free_dofs} | "
            f"{stats.adjacency_nnz} | {_format_float(stats.avg_row_nnz, 2)} | {stats.max_row_nnz} |"
        )
    return "\n".join(lines)


def _table_runtime(results_by_kind: dict[str, list[dict]]) -> str:
    lines = [
        "| discretization | ranks | total time [s] | solve time [s] | setup [s] | Newton | linear | avg linear / Newton | energy | omega | u_max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, rows in results_by_kind.items():
        for row in rows:
            lines.append(
                "| "
                f"{label} | {row['ranks']} | {_format_float(row['total_time_sec'])} | "
                f"{_format_float(row['solve_time_sec'])} | {_format_float(row['setup_time_sec'])} | "
                f"{row['newton_iterations']} | {row['linear_iterations']} | "
                f"{_format_float(row['avg_linear_iterations_per_newton'], 2)} | "
                f"{_format_float(row['energy'], 6)} | {_format_float(row['omega'], 6)} | {_format_float(row['u_max'], 6)} |"
            )
    return "\n".join(lines)


def _table_breakdown(results_by_kind: dict[str, list[dict]]) -> str:
    lines = [
        "| discretization | ranks | assembly [s] | PC setup [s] | KSP [s] | other solve [s] | linear phase [s] | KSP share of solve [%] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, rows in results_by_kind.items():
        for row in rows:
            ksp_share = 100.0 * row["ksp_time_sec"] / max(row["solve_time_sec"], 1.0e-12)
            lines.append(
                "| "
                f"{label} | {row['ranks']} | {_format_float(row['assembly_time_sec'])} | "
                f"{_format_float(row['pc_setup_time_sec'])} | {_format_float(row['ksp_time_sec'])} | "
                f"{_format_float(row['other_solve_time_sec'])} | {_format_float(row['linear_phase_time_sec'])} | "
                f"{_format_float(ksp_share, 1)} |"
            )
    return "\n".join(lines)


def _table_speedup(results_by_kind: dict[str, list[dict]]) -> str:
    p2_by_ranks = {row["ranks"]: row for row in results_by_kind["P2"]}
    p1_by_ranks = {row["ranks"]: row for row in results_by_kind["P1 on same nodes"]}
    lines = [
        "| ranks | refined P1 / P2 total time | refined P1 / P2 solve time | refined P1 / P2 linear iterations |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for ranks in sorted(p2_by_ranks):
        p2 = p2_by_ranks[ranks]
        p1 = p1_by_ranks[ranks]
        lines.append(
            "| "
            f"{ranks} | {_format_float(p1['total_time_sec'] / p2['total_time_sec'], 3)} | "
            f"{_format_float(p1['solve_time_sec'] / p2['solve_time_sec'], 3)} | "
            f"{_format_float(p1['linear_iterations'] / p2['linear_iterations'], 3)} |"
        )
    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    results_by_kind: dict[str, list[dict]],
    graph_stats: dict[str, GraphStats],
) -> None:
    p2 = graph_stats["P2"]
    p1 = graph_stats["P1 on same nodes"]
    nnz_ratio = p1.adjacency_nnz / p2.adjacency_nnz
    row_ratio = p1.avg_row_nnz / p2.avg_row_nnz
    report = f"""# L5 lambda=1.0: P2 vs refined P1 on the same nodes

This comparison keeps the node set fixed at the `level 5` P2 mesh and changes only the element topology:

- `P2`: the original quadratic triangles with `20800` elements
- `P1 on same nodes`: each P2 triangle split into four linear triangles, giving `83200` elements on the same `42081` nodes

All runs use the same nonlinear and linear settings:

- `lambda = 1.0`
- `pc_type = hypre` with `pc_hypre_type = boomeramg`
- `ksp_type = cg`
- trust region enabled
- `ksp_rtol = 1e-2`
- `ksp_max_it = 50`

## Graph complexity

{_table_graph_stats(graph_stats)}

The refined `P1` graph is noticeably sparser even though it has four times as many elements:

- total adjacency nnz drops from `{p2.adjacency_nnz}` to `{p1.adjacency_nnz}` (`{_format_float(nnz_ratio, 3)}x` of the P2 graph)
- average row nnz drops from `{_format_float(p2.avg_row_nnz, 2)}` to `{_format_float(p1.avg_row_nnz, 2)}` (`{_format_float(row_ratio, 3)}x`)

![Graph complexity](graph_complexity.png)

## Solver comparison

{_table_runtime(results_by_kind)}

The two discretizations reach very similar responses, but BoomerAMG consistently solves the refined `P1` system faster on this mesh:

{_table_speedup(results_by_kind)}

![Runtime vs ranks](runtime_vs_ranks.png)
![Iterations vs ranks](iterations_vs_ranks.png)

## Time breakdown

{_table_breakdown(results_by_kind)}

Interpretation:

- On the same node set, the refined `P1` graph is much narrower, which reduces BoomerAMG work per Newton step.
- At `8` ranks the refined `P1` solve time is about `{_format_float(results_by_kind['P1 on same nodes'][0]['solve_time_sec'] / results_by_kind['P2'][0]['solve_time_sec'], 3)}` of the P2 solve time.
- At `16` ranks the refined `P1` solve time is about `{_format_float(results_by_kind['P1 on same nodes'][1]['solve_time_sec'] / results_by_kind['P2'][1]['solve_time_sec'], 3)}` of the P2 solve time.
- At `32` ranks the refined `P1` solve time is about `{_format_float(results_by_kind['P1 on same nodes'][2]['solve_time_sec'] / results_by_kind['P2'][2]['solve_time_sec'], 3)}` of the P2 solve time.
- The biggest runtime bucket is still KSP time in every case, but the refined `P1` runs also need fewer Newton steps and fewer total linear iterations.
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_kind = {
        "P2": [
            _extract_result(args.input_dir / "l5_p2_np8.json"),
            _extract_result(args.input_dir / "l5_p2_np16.json"),
            _extract_result(args.input_dir / "l5_p2_np32.json"),
        ],
        "P1 on same nodes": [
            _extract_result(args.input_dir / "l5_refined_p1_np8.json"),
            _extract_result(args.input_dir / "l5_refined_p1_np16.json"),
            _extract_result(args.input_dir / "l5_refined_p1_np32.json"),
        ],
    }
    graph_stats = {
        "P2": _graph_stats("ssr_homo_capture_p2_level5", refined_p1=False),
        "P1 on same nodes": _graph_stats("ssr_homo_capture_p2_level5", refined_p1=True),
    }

    _plot_runtime(results_by_kind, output_dir)
    _plot_iterations(results_by_kind, output_dir)
    _plot_graph_stats(graph_stats, output_dir)
    _write_report(output_dir, results_by_kind, graph_stats)

    summary = {
        "results_by_kind": results_by_kind,
        "graph_stats": {key: asdict(value) for key, value in graph_stats.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
