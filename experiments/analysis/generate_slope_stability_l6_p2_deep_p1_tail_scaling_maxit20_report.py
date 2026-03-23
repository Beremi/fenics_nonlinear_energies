#!/usr/bin/env python3
"""Generate a scaling report for the L6 P2 deep-P1-tail PMG solver with Newton maxit=20."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("artifacts/raw_results/slope_stability_l6_p2_deep_p1_tail_scaling_lambda1_maxit20/summary.json")
DEFAULT_OUTPUT = Path("artifacts/reports/slope_stability_l6_p2_deep_p1_tail_scaling_lambda1_maxit20")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _load_details(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    details: dict[int, dict[str, object]] = {}
    for row in rows:
        payload = json.loads(Path(str(row["result_json"])).read_text(encoding="utf-8"))
        result = dict(payload.get("result", {}))
        steps = list(result.get("steps", []))
        last_step = steps[-1] if steps else {}
        linear_timing = (last_step.get("linear_timing") or [{}])[-1]
        details[int(row["ranks"])] = {
            "timings": dict(payload.get("timings", {})),
            "step": last_step,
            "linear_timing": linear_timing,
        }
    return details


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
            row.append(_fmt(float(values_by_rank.get(rank, {}).get(key, 0.0))))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _mg_detail_table(ranks: list[int], details: dict[int, dict[str, object]]) -> str:
    ordered_labels: list[str] = []
    seen: set[str] = set()
    by_rank: dict[int, dict[str, float]] = {}
    for rank in ranks:
        diag = list(details[rank]["linear_timing"].get("mg_runtime_diagnostics", []))
        per_rank: dict[str, float] = {}
        for item in diag:
            label = str(item.get("label", ""))
            if label not in seen:
                seen.add(label)
                ordered_labels.append(label)
            per_rank[label] = float(item.get("observed_time_sec", 0.0))
        by_rank[rank] = per_rank
    parts = [(label, label) for label in ordered_labels]
    return _detail_table(ranks, parts, by_rank)


def _plot_detail_group(
    ranks: list[int],
    parts: list[tuple[str, str]],
    values_by_rank: dict[int, dict[str, float]],
    *,
    title: str,
    output_path: Path,
) -> None:
    rank_array = np.array(ranks, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    for label, key in parts:
        values = np.array(
            [float(values_by_rank.get(rank, {}).get(key, 0.0)) for rank in ranks],
            dtype=np.float64,
        )
        mask = values > 0.0
        if not np.any(mask):
            continue
        ax.plot(
            rank_array[mask],
            values[mask],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title(title)
    ax.set_xticks(rank_array)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| ranks | success | steady-state [s] | solve [s] | one-time setup [s] | total [s] | steady speedup | efficiency | Newton | linear | omega | u_max | worst true rel |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    success_rows = [row for row in rows if bool(row["solver_success"])]
    base_row = success_rows[0] if success_rows else rows[0]
    t0 = float(base_row["steady_state_total_time_sec"])
    r0 = int(base_row["ranks"])
    for row in rows:
        speedup = t0 / float(row["steady_state_total_time_sec"])
        efficiency = speedup / (int(row["ranks"]) / r0)
        lines.append(
            "| "
            f"{row['ranks']} | {row['solver_success']} | {_fmt(float(row['steady_state_total_time_sec']))} | "
            f"{_fmt(float(row['solve_time_sec']))} | {_fmt(float(row['setup_time_sec']))} | "
            f"{_fmt(float(row['total_time_sec']))} | {_fmt(speedup)} | {_fmt(efficiency)} | "
            f"{row['newton_iterations']} | {row['linear_iterations']} | {_fmt(float(row['omega']), 6)} | "
            f"{_fmt(float(row['u_max']), 6)} | {_fmt(float(row['worst_true_relative_residual']), 6)} |"
        )
    return "\n".join(lines)


def _breakdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| ranks | problem build [s] | assembler init [s] | PMG bootstrap [s] | solve [s] | finalize [s] | outside solve [s] | outside setup+solve [s] |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['ranks']} | {_fmt(float(row.get('problem_build_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('assembler_setup_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('solver_bootstrap_time_sec', 0.0)))} | "
            f"{_fmt(float(row['solve_time_sec']))} | {_fmt(float(row.get('finalize_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('outside_solve_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('outside_setup_solve_time_sec', 0.0)))} |"
        )
    return "\n".join(lines)


def _plot(rows: list[dict[str, object]], out_dir: Path) -> None:
    ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
    steady = np.array([float(row["steady_state_total_time_sec"]) for row in rows], dtype=np.float64)
    solve = np.array([float(row["solve_time_sec"]) for row in rows], dtype=np.float64)
    setup = np.array([float(row["setup_time_sec"]) for row in rows], dtype=np.float64)
    total = np.array([float(row["total_time_sec"]) for row in rows], dtype=np.float64)
    success_mask = np.array([bool(row["solver_success"]) for row in rows], dtype=bool)
    success_rows = [row for row in rows if bool(row["solver_success"])]
    base_row = success_rows[0] if success_rows else rows[0]
    base_ranks = int(base_row["ranks"])
    base_steady = float(base_row["steady_state_total_time_sec"])
    speedup = np.array(
        [base_steady / float(row["steady_state_total_time_sec"]) if bool(row["solver_success"]) else np.nan for row in rows],
        dtype=np.float64,
    )
    ideal = ranks / base_ranks

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)

    ax = axes[0]
    ax.plot(ranks, steady, marker="o", linewidth=2.0, label="steady-state total")
    ax.plot(ranks, solve, marker="s", linewidth=2.0, label="solve")
    ax.plot(ranks, setup, marker="d", linewidth=2.0, label="one-time setup")
    ax.plot(ranks, total, marker="^", linewidth=2.0, label="end-to-end total")
    if np.any(~success_mask):
        ax.scatter(
            ranks[~success_mask],
            total[~success_mask],
            marker="x",
            s=80,
            color="crimson",
            label="nonconverged",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title("L6 P2 deep-P1-tail PMG timing (log-log)")
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(ranks[success_mask], speedup[success_mask], marker="o", linewidth=2.0, label="measured")
    ax.plot(ranks, ideal, linestyle="--", linewidth=1.8, label="ideal")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel(f"speedup vs {base_ranks} rank")
    ax.set_title("L6 P2 deep-P1-tail strong scaling")
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "scaling.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = sorted(json.loads(args.input.read_text(encoding="utf-8")), key=lambda row: int(row["ranks"]))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot(rows, out_dir)
    details = _load_details(rows)
    ranks = [int(row["ranks"]) for row in rows]

    setup_parts = [
        ("problem_build_time", "problem_build_time"),
        ("assembler: permutation", "assembler:permutation"),
        ("assembler: global_layout", "assembler:global_layout"),
        ("assembler: local_overlap", "assembler:local_overlap"),
        ("assembler: distribution_setup", "assembler:distribution_setup"),
        ("assembler: kernel_build", "assembler:kernel_build"),
        ("assembler: scatter_build", "assembler:scatter_build"),
        ("assembler: rhs_build", "assembler:rhs_build"),
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
            "assembler:scatter_build": float(
                details[rank]["timings"].get("assembler_setup_breakdown", {}).get("scatter_build", 0.0)
            ),
            "assembler:rhs_build": float(
                details[rank]["timings"].get("assembler_setup_breakdown", {}).get("rhs_build", 0.0)
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
    setup_breakdown = _detail_table(ranks, setup_parts, setup_values)

    callback_parts = [
        ("energy: total", "energy:total"),
        ("energy: kernel", "energy:kernel"),
        ("energy: ghost_exchange", "energy:ghost_exchange"),
        ("energy: allreduce", "energy:allreduce"),
        ("energy: load", "energy:load"),
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
            "energy:total": float(details[rank]["timings"].get("callback_summary", {}).get("energy", {}).get("total", 0.0)),
            "energy:kernel": float(details[rank]["timings"].get("callback_summary", {}).get("energy", {}).get("kernel", 0.0)),
            "energy:ghost_exchange": float(details[rank]["timings"].get("callback_summary", {}).get("energy", {}).get("ghost_exchange", 0.0)),
            "energy:allreduce": float(details[rank]["timings"].get("callback_summary", {}).get("energy", {}).get("allreduce", 0.0)),
            "energy:load": float(details[rank]["timings"].get("callback_summary", {}).get("energy", {}).get("load", 0.0)),
            "gradient:total": float(details[rank]["timings"].get("callback_summary", {}).get("gradient", {}).get("total", 0.0)),
            "gradient:kernel": float(details[rank]["timings"].get("callback_summary", {}).get("gradient", {}).get("kernel", 0.0)),
            "gradient:ghost_exchange": float(details[rank]["timings"].get("callback_summary", {}).get("gradient", {}).get("ghost_exchange", 0.0)),
            "hessian:total": float(details[rank]["timings"].get("callback_summary", {}).get("hessian", {}).get("total", 0.0)),
            "hessian:hvp_compute": float(details[rank]["timings"].get("callback_summary", {}).get("hessian", {}).get("hvp_compute", 0.0)),
            "hessian:extraction": float(details[rank]["timings"].get("callback_summary", {}).get("hessian", {}).get("extraction", 0.0)),
            "hessian:coo_assembly": float(details[rank]["timings"].get("callback_summary", {}).get("hessian", {}).get("coo_assembly", 0.0)),
            "hessian:ghost_exchange": float(details[rank]["timings"].get("callback_summary", {}).get("hessian", {}).get("ghost_exchange", 0.0)),
        }
        for rank in ranks
    }
    callback_breakdown = _detail_table(ranks, callback_parts, callback_values)

    linear_parts = [
        ("linear_total_time", "linear_total_time"),
        ("assemble_total_time", "assemble_total_time"),
        ("assemble_p2p_exchange", "assemble_p2p_exchange"),
        ("assemble_hvp_compute", "assemble_hvp_compute"),
        ("assemble_extraction", "assemble_extraction"),
        ("assemble_coo_assembly", "assemble_coo_assembly"),
        ("pc_setup_time", "pc_setup_time"),
        ("ksp_solve_time", "solve_time"),
    ]
    linear_values = {
        rank: {
            key: float(details[rank]["linear_timing"].get(key, 0.0))
            for key in [
                "linear_total_time",
                "assemble_total_time",
                "assemble_p2p_exchange",
                "assemble_hvp_compute",
                "assemble_extraction",
                "assemble_coo_assembly",
                "pc_setup_time",
                "solve_time",
            ]
        }
        for rank in ranks
    }
    linear_breakdown = _detail_table(ranks, linear_parts, linear_values)

    mg_breakdown = _mg_detail_table(ranks, details)

    _plot_detail_group(
        ranks,
        setup_parts,
        setup_values,
        title="L6 P2 deep-P1-tail setup subparts (log-log)",
        output_path=out_dir / "setup_subparts_loglog.png",
    )
    _plot_detail_group(
        ranks,
        callback_parts,
        callback_values,
        title="L6 P2 deep-P1-tail callback totals (log-log)",
        output_path=out_dir / "callback_breakdown_loglog.png",
    )
    _plot_detail_group(
        ranks,
        linear_parts,
        linear_values,
        title="L6 P2 deep-P1-tail final linear solve (log-log)",
        output_path=out_dir / "linear_breakdown_loglog.png",
    )
    mg_values = {}
    mg_labels: list[str] = []
    seen_mg_labels: set[str] = set()
    for rank in ranks:
        per_rank: dict[str, float] = {}
        for item in details[rank]["linear_timing"].get("mg_runtime_diagnostics", []):
            label = str(item.get("label", ""))
            if label and label not in seen_mg_labels:
                seen_mg_labels.add(label)
                mg_labels.append(label)
            per_rank[label] = float(item.get("observed_time_sec", 0.0))
        mg_values[rank] = per_rank
    mg_parts = [(label, label) for label in mg_labels]
    _plot_detail_group(
        ranks,
        mg_parts,
        mg_values,
        title="L6 P2 deep-P1-tail PMG internals (log-log)",
        output_path=out_dir / "pmg_internal_loglog.png",
    )

    hierarchy = rows[0].get("mg_custom_hierarchy", "1:1,2:1,3:1,4:1,5:1,6:1,6:2") if rows else ""
    report = f"""# `L6` `P2` Deep-`P1`-Tail PMG Scaling With Newton `maxit=20`

Featured solver configuration:

- level: `6`
- discretisation: same-mesh `P2`
- preconditioner: `PCMG`
- hierarchy: `{hierarchy}`
- nonlinear setting: `--no-use_trust_region`
- nonlinear cap: `--maxit 20`
- linear setting: `fgmres`, `rtol=1e-2`, `max_it=100`
- smoothers:
  - `P2`: `richardson + sor`, `3` steps
  - `P1`: `richardson + sor`, `3` steps
- coarse solve:
  - `cg + hypre(boomeramg)`
  - `nodal_coarsen = 6`
  - `vec_interp_variant = 3`
  - `strong_threshold = 0.5`
  - `coarsen_type = HMIS`
  - `max_iter = 4`
  - `tol = 0.0`
  - `relax_type_all = symmetric-SOR/Jacobi`

This is the deep-`P1`-tail analogue of the `L6` `P4` scaling run, benchmarked
on `1/2/4/8` MPI ranks for `20` Newton iterations.

Every row below is intentionally capped at `20` Newton iterations. The
`success = False` entries therefore mean "reached the requested cap before the
nonlinear stop test", not linear-solver failure. The final energies and state
metrics are still directly comparable across ranks.

## Scaling Table

{_table(rows)}

## Timing Breakdown

{_breakdown_table(rows)}

## Setup Subparts

Times below are one-time setup components for the full run.

{setup_breakdown}

![L6 P2 deep-P1-tail setup subparts](setup_subparts_loglog.png)

## Callback Totals And Internals

Times below are totals accumulated over the whole capped run.

{callback_breakdown}

![L6 P2 deep-P1-tail callback breakdown](callback_breakdown_loglog.png)

## Final Linear Solve Breakdown

Times below come from the final linear solve at the capped last Newton step.

{linear_breakdown}

![L6 P2 deep-P1-tail final linear solve breakdown](linear_breakdown_loglog.png)

## PMG Internal Breakdown

These are the measured PMG internal components from the final linear solve at
the capped last Newton step.

{mg_breakdown}

![L6 P2 deep-P1-tail PMG internal breakdown](pmg_internal_loglog.png)

## Graph

![L6 P2 deep-P1-tail PMG scaling, maxit=20](scaling.png)
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
