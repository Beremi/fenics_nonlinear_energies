#!/usr/bin/env python3
"""Generate the L7 P4 deep-P1-tail scaling report with component breakdowns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l7_p4_deep_p1_tail_scaling_lambda1_maxit20/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l7_p4_deep_p1_tail_scaling_lambda1_maxit20"
)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _peak_memory(path_str: str) -> float:
    path = Path(path_str)
    if not path.exists():
        return float("nan")
    peak = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        peak = max(peak, float(row.get("rss_gb_total", 0.0)))
    return peak


def _load_details(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    details: dict[int, dict[str, object]] = {}
    for row in rows:
        rank = int(row["ranks"])
        result_path = Path(str(row.get("result_json", "")))
        payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        result = dict(payload.get("result", {}))
        steps = list(result.get("steps", []))
        last_step = steps[-1] if steps else {}
        linear_timing = list(last_step.get("linear_timing", []))
        timings = dict(payload.get("timings", {}))
        details[rank] = {
            "payload": payload,
            "timings": timings,
            "step": last_step,
            "linear_timing": linear_timing,
            "callback_summary": dict(timings.get("callback_summary", {})),
            "peak_rss_gib": _peak_memory(str(row.get("memory_watch_path", ""))),
        }
    return details


def _sum_linear(rows: list[dict[str, object]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0)) for item in rows))


def _sum_mg_runtime(rows: list[dict[str, object]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in rows:
        for diag in list(item.get("mg_runtime_diagnostics", [])):
            label = str(diag.get("label", ""))
            out[label] = out.get(label, 0.0) + float(diag.get("observed_time_sec", 0.0))
    return out


def _detail_table(
    ranks: list[int],
    parts: list[tuple[str, str]],
    values_by_rank: dict[int, dict[str, float]],
) -> str:
    lines = [
        "| part | " + " | ".join(str(rank) for rank in ranks) + " |",
        "| --- | " + " | ".join("---:" for _ in ranks) + " |",
    ]
    for label, key in parts:
        row = [f"`{label}`"]
        for rank in ranks:
            value = float(values_by_rank.get(rank, {}).get(key, 0.0))
            row.append("" if np.isnan(value) else _fmt(value))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _plot_group(
    ranks: list[int],
    parts: list[tuple[str, str]],
    values_by_rank: dict[int, dict[str, float]],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    rank_array = np.array(ranks, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(10.8, 6.0), dpi=180)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0
    for label, key in parts:
        values = np.array(
            [float(values_by_rank.get(rank, {}).get(key, np.nan)) for rank in ranks],
            dtype=np.float64,
        )
        mask = np.isfinite(values) & (values > 0.0)
        if not np.any(mask):
            continue
        color = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1
        ax.plot(
            rank_array[mask],
            values[mask],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
        first_idx = int(np.where(mask)[0][0])
        ideal = values[first_idx] * (rank_array[first_idx] / rank_array.astype(np.float64))
        ax.plot(
            rank_array,
            ideal,
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.55,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(rank_array)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_overall(rows: list[dict[str, object]], output_path: Path) -> None:
    ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
    data = [
        ("steady-state total", np.array([float(row.get("steady_state_total_time_sec", np.nan)) for row in rows])),
        ("solve", np.array([float(row.get("solve_time_sec", np.nan)) for row in rows])),
        ("one-time setup", np.array([float(row.get("one_time_setup_time_sec", np.nan)) for row in rows])),
        ("linear KSP", np.array([float(row.get("linear_ksp_solve_time_sec", np.nan)) for row in rows])),
        ("linear assemble", np.array([float(row.get("linear_assemble_time_sec", np.nan)) for row in rows])),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (label, values) in enumerate(data):
        mask = np.isfinite(values) & (values > 0.0)
        if not np.any(mask):
            continue
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(ranks[mask], values[mask], marker="o", linewidth=2.0, color=color, label=label)
        first_idx = int(np.where(mask)[0][0])
        ideal = values[first_idx] * (ranks[first_idx] / ranks.astype(np.float64))
        ax.plot(ranks, ideal, linestyle="--", linewidth=1.0, color=color, alpha=0.55)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title("L7/P4 deep-P1-tail scaling (log-log)")
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_overall_per_linear_iteration(rows: list[dict[str, object]], output_path: Path) -> None:
    ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
    linear_iters = np.array(
        [max(float(row.get("linear_iterations", 0)), 1.0) for row in rows],
        dtype=np.float64,
    )
    data = [
        (
            "steady-state total / linear it",
            np.array([float(row.get("steady_state_total_time_sec", np.nan)) for row in rows])
            / linear_iters,
        ),
        (
            "solve / linear it",
            np.array([float(row.get("solve_time_sec", np.nan)) for row in rows]) / linear_iters,
        ),
        (
            "one-time setup / linear it",
            np.array([float(row.get("one_time_setup_time_sec", np.nan)) for row in rows])
            / linear_iters,
        ),
        (
            "linear KSP / linear it",
            np.array([float(row.get("linear_ksp_solve_time_sec", np.nan)) for row in rows])
            / linear_iters,
        ),
        (
            "linear assemble / linear it",
            np.array([float(row.get("linear_assemble_time_sec", np.nan)) for row in rows])
            / linear_iters,
        ),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (label, values) in enumerate(data):
        mask = np.isfinite(values) & (values > 0.0)
        if not np.any(mask):
            continue
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(ranks[mask], values[mask], marker="o", linewidth=2.0, color=color, label=label)
        first_idx = int(np.where(mask)[0][0])
        ideal = values[first_idx] * (ranks[first_idx] / ranks.astype(np.float64))
        ax.plot(ranks, ideal, linestyle="--", linewidth=1.0, color=color, alpha=0.55)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time per total linear iteration [s]")
    ax.set_title("L7/P4 overall scaling normalized by linear iterations (log-log)")
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _summary_table(rows: list[dict[str, object]], details: dict[int, dict[str, object]]) -> str:
    lines = [
        "| ranks | status | steady-state [s] | solve [s] | setup [s] | Newton reached | linear | accepted capped | final energy | final grad | worst true rel | peak RSS [GiB] |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        rank = int(row["ranks"])
        peak = float(details.get(rank, {}).get("peak_rss_gib", float("nan")))
        lines.append(
            "| "
            f"{rank} | {row.get('status', '')} | {_fmt(float(row.get('steady_state_total_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('solve_time_sec', 0.0)))} | {_fmt(float(row.get('one_time_setup_time_sec', 0.0)))} | "
            f"{int(row.get('newton_iterations', 0))} | {int(row.get('linear_iterations', 0))} | "
            f"{int(row.get('accepted_capped_step_count', 0))} | {_fmt(float(row.get('energy', float('nan'))), 9)} | "
            f"{_fmt(float(row.get('final_grad_norm', float('nan'))), 6)} | "
            f"{_fmt(float(row.get('worst_true_relative_residual', float('nan'))), 6)} | "
            f"{'' if np.isnan(peak) else _fmt(peak)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = sorted(json.loads(args.input.read_text(encoding="utf-8")), key=lambda row: int(row["ranks"]))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    details = _load_details(rows)
    ranks = [int(row["ranks"]) for row in rows]

    setup_parts = [
        ("problem_build_time", "problem_build_time"),
        ("assembler: permutation", "assembler:permutation"),
        ("assembler: global_layout", "assembler:global_layout"),
        ("assembler: local_overlap", "assembler:local_overlap"),
        ("assembler: distribution_setup", "assembler:distribution_setup"),
        ("assembler: kernel_build", "assembler:kernel_build"),
        ("assembler: matrix_create", "assembler:matrix_create"),
        ("assembler: nullspace_build", "assembler:nullspace_build"),
        ("assembler: warmup", "assembler:warmup"),
        ("bootstrap: mg_hierarchy_build", "bootstrap:mg_hierarchy_build"),
        ("bootstrap: mg_level_build", "bootstrap:mg_level_build"),
        ("bootstrap: mg_transfer_build", "bootstrap:mg_transfer_build"),
        ("bootstrap: mg_transfer_mapping", "bootstrap:mg_transfer_mapping"),
        ("bootstrap: mg_transfer_matrix_build", "bootstrap:mg_transfer_matrix_build"),
        ("bootstrap: mg_transfer_cache_io", "bootstrap:mg_transfer_cache_io"),
        ("bootstrap: mg_configure", "bootstrap:mg_configure"),
    ]
    setup_values = {
        rank: {
            "problem_build_time": float(details[rank]["timings"].get("problem_build_time", 0.0)),
            "assembler:permutation": float(
                details[rank]["timings"].get("assembler_setup_breakdown", {}).get("permutation", 0.0)
            ),
            "assembler:global_layout": float(
                details[rank]["timings"].get("assembler_setup_breakdown", {}).get("global_layout", 0.0)
            ),
            "assembler:local_overlap": float(
                details[rank]["timings"].get("assembler_setup_breakdown", {}).get("local_overlap", 0.0)
            ),
            "assembler:distribution_setup": float(
                details[rank]["timings"].get("assembler_setup_breakdown", {}).get("distribution_setup", 0.0)
            ),
            "assembler:kernel_build": float(
                details[rank]["timings"].get("assembler_setup_breakdown", {}).get("kernel_build", 0.0)
            ),
            "assembler:matrix_create": float(
                details[rank]["timings"].get("assembler_setup_breakdown", {}).get("matrix_create", 0.0)
            ),
            "assembler:nullspace_build": float(
                details[rank]["timings"].get("assembler_setup_breakdown", {}).get("nullspace_build", 0.0)
            ),
            "assembler:warmup": float(
                details[rank]["timings"].get("assembler_setup_breakdown", {}).get("warmup", 0.0)
            ),
            "bootstrap:mg_hierarchy_build": float(
                details[rank]["timings"].get("solver_bootstrap_breakdown", {}).get("mg_hierarchy_build_time", 0.0)
            ),
            "bootstrap:mg_level_build": float(
                details[rank]["timings"].get("solver_bootstrap_breakdown", {}).get("mg_level_build_time", 0.0)
            ),
            "bootstrap:mg_transfer_build": float(
                details[rank]["timings"].get("solver_bootstrap_breakdown", {}).get("mg_transfer_build_time", 0.0)
            ),
            "bootstrap:mg_transfer_mapping": float(
                details[rank]["timings"].get("solver_bootstrap_breakdown", {}).get("mg_transfer_mapping_time", 0.0)
            ),
            "bootstrap:mg_transfer_matrix_build": float(
                details[rank]["timings"].get("solver_bootstrap_breakdown", {}).get("mg_transfer_matrix_build_time", 0.0)
            ),
            "bootstrap:mg_transfer_cache_io": float(
                details[rank]["timings"].get("solver_bootstrap_breakdown", {}).get("mg_transfer_cache_io_time", 0.0)
            ),
            "bootstrap:mg_configure": float(
                details[rank]["timings"].get("solver_bootstrap_breakdown", {}).get("mg_configure_time", 0.0)
            ),
        }
        for rank in ranks
    }

    callback_parts = [
        ("energy: total", "energy:total"),
        ("energy: kernel", "energy:kernel"),
        ("energy: ghost_exchange", "energy:ghost_exchange"),
        ("energy: allreduce", "energy:allreduce"),
        ("gradient: total", "gradient:total"),
        ("gradient: kernel", "gradient:kernel"),
        ("gradient: ghost_exchange", "gradient:ghost_exchange"),
        ("hessian: total", "hessian:total"),
        ("hessian: hvp_compute", "hessian:hvp_compute"),
        ("hessian: extraction", "hessian:extraction"),
        ("hessian: coo_assembly", "hessian:coo_assembly"),
        ("hessian: ghost_exchange", "hessian:ghost_exchange"),
    ]
    callback_values = {
        rank: {
            "energy:total": float(details[rank]["callback_summary"].get("energy", {}).get("total", 0.0)),
            "energy:kernel": float(details[rank]["callback_summary"].get("energy", {}).get("kernel", 0.0)),
            "energy:ghost_exchange": float(
                details[rank]["callback_summary"].get("energy", {}).get("ghost_exchange", 0.0)
            ),
            "energy:allreduce": float(
                details[rank]["callback_summary"].get("energy", {}).get("allreduce", 0.0)
            ),
            "gradient:total": float(
                details[rank]["callback_summary"].get("gradient", {}).get("total", 0.0)
            ),
            "gradient:kernel": float(
                details[rank]["callback_summary"].get("gradient", {}).get("kernel", 0.0)
            ),
            "gradient:ghost_exchange": float(
                details[rank]["callback_summary"].get("gradient", {}).get("ghost_exchange", 0.0)
            ),
            "hessian:total": float(
                details[rank]["callback_summary"].get("hessian", {}).get("total", 0.0)
            ),
            "hessian:hvp_compute": float(
                details[rank]["callback_summary"].get("hessian", {}).get("hvp_compute", 0.0)
            ),
            "hessian:extraction": float(
                details[rank]["callback_summary"].get("hessian", {}).get("extraction", 0.0)
            ),
            "hessian:coo_assembly": float(
                details[rank]["callback_summary"].get("hessian", {}).get("coo_assembly", 0.0)
            ),
            "hessian:ghost_exchange": float(
                details[rank]["callback_summary"].get("hessian", {}).get("ghost_exchange", 0.0)
            ),
        }
        for rank in ranks
    }

    linear_parts = [
        ("linear total", "linear_total"),
        ("KSP solve", "solve_time"),
        ("PC setup", "pc_setup_time"),
        ("assemble total", "assemble_total_time"),
        ("assemble hvp", "assemble_hvp_compute"),
        ("assemble extraction", "assemble_extraction"),
        ("assemble COO", "assemble_coo_assembly"),
    ]
    linear_values = {
        rank: {
            "linear_total": _sum_linear(details[rank]["linear_timing"], "linear_total_time"),
            "solve_time": _sum_linear(details[rank]["linear_timing"], "solve_time"),
            "pc_setup_time": _sum_linear(details[rank]["linear_timing"], "pc_setup_time"),
            "assemble_total_time": _sum_linear(details[rank]["linear_timing"], "assemble_total_time"),
            "assemble_hvp_compute": _sum_linear(details[rank]["linear_timing"], "assemble_hvp_compute"),
            "assemble_extraction": _sum_linear(details[rank]["linear_timing"], "assemble_extraction"),
            "assemble_coo_assembly": _sum_linear(details[rank]["linear_timing"], "assemble_coo_assembly"),
        }
        for rank in ranks
    }

    mg_labels: list[str] = []
    seen: set[str] = set()
    mg_values: dict[int, dict[str, float]] = {}
    for rank in ranks:
        summed = _sum_mg_runtime(details[rank]["linear_timing"])
        mg_values[rank] = summed
        for label in summed:
            if label not in seen:
                seen.add(label)
                mg_labels.append(label)
    mg_parts = [(label, label) for label in mg_labels]

    _plot_overall(rows, out_dir / "overall_scaling_loglog.png")
    _plot_overall_per_linear_iteration(
        rows, out_dir / "overall_per_linear_iteration_loglog.png"
    )
    _plot_group(
        ranks,
        setup_parts,
        setup_values,
        title="Setup subparts (log-log, dashed = ideal 1/p)",
        ylabel="time [s]",
        output_path=out_dir / "setup_subparts_loglog.png",
    )
    _plot_group(
        ranks,
        callback_parts,
        callback_values,
        title="Callback breakdown (log-log, dashed = ideal 1/p)",
        ylabel="time [s]",
        output_path=out_dir / "callback_breakdown_loglog.png",
    )
    _plot_group(
        ranks,
        linear_parts,
        linear_values,
        title="Linear breakdown (log-log, dashed = ideal 1/p)",
        ylabel="time [s]",
        output_path=out_dir / "linear_breakdown_loglog.png",
    )
    if mg_parts:
        _plot_group(
            ranks,
            mg_parts,
            mg_values,
            title="PMG internal runtime (log-log, dashed = ideal 1/p)",
            ylabel="time [s]",
            output_path=out_dir / "pmg_internal_loglog.png",
        )

    summary_table = _summary_table(rows, details)
    setup_table = _detail_table(ranks, setup_parts, setup_values)
    callback_table = _detail_table(ranks, callback_parts, callback_values)
    linear_table = _detail_table(ranks, linear_parts, linear_values)
    mg_table = _detail_table(ranks, mg_parts, mg_values) if mg_parts else "_No MG runtime diagnostics found._"

    report = f"""# `L7/P4` deep-`P1`-tail scaling on `1/2/4/8/16` ranks

Setting:

- problem: `L7`, `P4`, `lambda=1.0`
- hierarchy: `1:1,2:1,3:1,4:1,5:1,6:1,7:1,7:2,7:4`
- nonlinear: `armijo`, `maxit=20`, `--no-use_trust_region`
- linear: `fgmres`, `ksp_max_it=15`
- capped linear solve handling: `accept_ksp_maxit_direction`, `--no-guard_ksp_maxit_direction`
- coarse solve: `rank0_lu_broadcast`
- smoothers: `P4/P2/P1 = richardson + sor`, `3` steps
- problem build mode: `rank_local`
- MG level build mode: `rank_local`
- transfer build mode: `owned_rows`
- hot-path distribution: `overlap_p2p`
- benchmark mode: `warmup_once_then_solve`
- Hessian buffer mode: default reuse enabled
- memory guard: terminate a rank case if watched RSS exceeds `{_fmt(210.0)}` GiB

## Summary

{summary_table}

## Plots

![Overall scaling](overall_scaling_loglog.png)

![Overall scaling per linear iteration](overall_per_linear_iteration_loglog.png)

![Setup subparts](setup_subparts_loglog.png)

![Callback breakdown](callback_breakdown_loglog.png)

![Linear breakdown](linear_breakdown_loglog.png)
"""
    if mg_parts:
        report += "\n![PMG internal runtime](pmg_internal_loglog.png)\n"
    report += f"""

## Setup Breakdown

{setup_table}

## Callback Breakdown

{callback_table}

## Linear Breakdown

{linear_table}

## PMG Internal Breakdown

{mg_table}
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
