from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/raw_results/scaling_probe")
FIXED_BASE = ROOT / "p4_cheby_jacobi5_maxit1_v3_threads1"
FULL16 = ROOT / "p4_l1_lambda1p5_np16_cheby_full_v2" / "output.json"
OUTDIR = ROOT / "p4_lambda1p5_parallel_assets"
PER_PHASE_DIR = OUTDIR / "per_phase"
REPORT = ROOT / "P4_lambda1p5_parallel_scaling.md"
GLOBAL_ELEMS = 18419.0


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    label: str


PHASE_SPECS = (
    PhaseSpec("total_time", "Total end-to-end"),
    PhaseSpec("solve_time", "Nonlinear solve"),
    PhaseSpec("assembler_create", "Assembler create"),
    PhaseSpec("linear1_t_solve", "First linear KSP solve"),
    PhaseSpec("hessian_total", "Hessian callbacks total"),
    PhaseSpec("linear1_t_assemble", "First linear assemble"),
    PhaseSpec("initial_guess_total", "Elastic initial guess"),
    PhaseSpec("mg_hierarchy_build", "MG hierarchy build"),
    PhaseSpec("problem_load", "Problem load"),
    PhaseSpec("linear1_t_setup", "First linear KSP setup"),
    PhaseSpec("energy_total", "Energy callbacks total"),
    PhaseSpec("hessian_hvp", "Hessian HVP compute"),
    PhaseSpec("gradient_total", "Gradient callbacks total"),
    PhaseSpec("hessian_coo", "Hessian COO assembly"),
    PhaseSpec("hessian_extraction", "Hessian extraction"),
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _first_linear_records(parallel_diag: list[dict]) -> list[dict]:
    return [rank["linear_history"][0] for rank in parallel_diag]


def _phase_values(obj: dict) -> dict[str, float]:
    pd = obj["parallel_diagnostics"]
    first_linear = _first_linear_records(pd)
    return {
        "total_time": float(obj["total_time"]),
        "solve_time": float(obj["solve_time"]),
        "problem_load": max(float(rank["stage_timings"].get("problem_load", 0.0)) for rank in pd),
        "assembler_create": max(
            float(rank["stage_timings"].get("assembler_create", 0.0)) for rank in pd
        ),
        "mg_hierarchy_build": max(
            float(rank["stage_timings"].get("mg_hierarchy_build", 0.0)) for rank in pd
        ),
        "initial_guess_total": max(
            float(rank["stage_timings"].get("initial_guess_total", 0.0)) for rank in pd
        ),
        "energy_total": max(
            float(rank["assembly_callbacks"]["energy"]["total"]) for rank in pd
        ),
        "gradient_total": max(
            float(rank["assembly_callbacks"]["gradient"]["total"]) for rank in pd
        ),
        "hessian_total": max(
            float(rank["assembly_callbacks"]["hessian"]["total"]) for rank in pd
        ),
        "hessian_hvp": max(
            float(rank["assembly_callbacks"]["hessian"]["hvp_compute"]) for rank in pd
        ),
        "hessian_coo": max(
            float(rank["assembly_callbacks"]["hessian"]["coo_assembly"]) for rank in pd
        ),
        "hessian_extraction": max(
            float(rank["assembly_callbacks"]["hessian"]["extraction"]) for rank in pd
        ),
        "linear1_t_assemble": max(float(rec["t_assemble"]) for rec in first_linear),
        "linear1_t_setup": max(float(rec["t_setup"]) for rec in first_linear),
        "linear1_t_solve": max(float(rec["t_solve"]) for rec in first_linear),
    }


def _collect_rows() -> list[dict]:
    rows: list[dict] = []
    for np_ranks in (1, 2, 4, 8, 16, 32):
        obj = _load_json(FIXED_BASE / f"np{np_ranks}" / "output.json")
        pd = obj["parallel_diagnostics"]
        phase_values = _phase_values(obj)
        row = {
            "np": float(np_ranks),
            "total_time": float(obj["total_time"]),
            "solve_time": float(obj["solve_time"]),
            "dup_factor": sum(int(rank["local_problem"]["local_elements"]) for rank in pd)
            / GLOBAL_ELEMS,
            "overlap_dof_factor": sum(
                int(rank["local_problem"]["overlap_total_dofs"]) for rank in pd
            )
            / sum(int(rank["local_problem"]["owned_free_dofs"]) for rank in pd),
            "local_elem_min": min(int(rank["local_problem"]["local_elements"]) for rank in pd),
            "local_elem_max": max(int(rank["local_problem"]["local_elements"]) for rank in pd),
            "output_path": str(FIXED_BASE / f"np{np_ranks}" / "output.json"),
        }
        row.update(phase_values)
        rows.append(row)

    base_total = rows[0]["total_time"]
    for row in rows:
        row["speedup"] = base_total / row["total_time"]
        row["efficiency"] = row["speedup"] / row["np"]
    return rows


def _phase_records(rows: list[dict]) -> list[dict]:
    ranks = np.array([row["np"] for row in rows], dtype=float)
    total_rank1 = rows[0]["total_time"]
    records: list[dict] = []
    for spec in PHASE_SPECS:
        values = np.array([float(row[spec.key]) for row in rows], dtype=float)
        base = float(values[0])
        last = float(values[-1])
        speedup = base / last if last > 0.0 else math.nan
        eff = speedup / ranks[-1] if last > 0.0 else math.nan
        records.append(
            {
                "key": spec.key,
                "label": spec.label,
                "times": values.tolist(),
                "rank1": base,
                "rank32": last,
                "speedup32": speedup,
                "eff32": eff,
                "share_rank1": base / total_rank1 if total_rank1 > 0.0 else math.nan,
            }
        )
    records.sort(key=lambda rec: rec["rank1"], reverse=True)
    return records


def _save(fig: plt.Figure, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _phase_plot_name(index: int, key: str) -> str:
    return f"{index:02d}_{key}.png"


def plot_scaling_overview(rows: list[dict]) -> None:
    ranks = np.array([row["np"] for row in rows], dtype=float)
    total = np.array([row["total_time"] for row in rows], dtype=float)
    solve = np.array([row["solve_time"] for row in rows], dtype=float)
    ideal_total = total[0] / ranks
    ideal_solve = solve[0] / ranks
    speedup = np.array([row["speedup"] for row in rows], dtype=float)
    efficiency = np.array([row["efficiency"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.loglog(ranks, total, marker="o", linewidth=2.0, label="Total time")
    ax.loglog(ranks, ideal_total, linestyle="--", color="0.35", label="Ideal total")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    ax.set_xticks(ranks, [str(int(v)) for v in ranks])
    ax.set_title("End-to-End Strong Scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.loglog(ranks, solve, marker="s", linewidth=2.0, label="Solve phase")
    ax.loglog(ranks, ideal_solve, linestyle="--", color="0.35", label="Ideal solve")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    ax.set_xticks(ranks, [str(int(v)) for v in ranks])
    ax.set_title("Solve-Phase Strong Scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[2]
    ax.loglog(ranks, speedup, marker="o", linewidth=2.0, label="Measured speedup")
    ax.loglog(ranks, ranks, linestyle="--", color="0.35", label="Ideal speedup")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Speedup")
    ax.set_xticks(ranks, [str(int(v)) for v in ranks])
    ax.set_title("Speedup")
    ax.grid(True, which="both", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(
        ranks,
        efficiency,
        marker="d",
        linewidth=1.7,
        color="tab:red",
        label="Efficiency",
    )
    ax2.axhline(0.7, linestyle=":", color="tab:red", alpha=0.8, label="70% target")
    ax2.set_ylabel("Efficiency")
    ax2.set_ylim(0.0, 1.05)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")

    _save(fig, "scaling_overview_loglog.png")


def plot_overlap_and_efficiency(rows: list[dict]) -> None:
    ranks = np.array([row["np"] for row in rows], dtype=float)
    labels = [str(int(v)) for v in ranks]
    eff = np.array([row["efficiency"] for row in rows], dtype=float)
    dup = np.array([row["dup_factor"] for row in rows], dtype=float)
    overlap = np.array([row["overlap_dof_factor"] for row in rows], dtype=float)
    elem_min = np.array([row["local_elem_min"] for row in rows], dtype=float)
    elem_max = np.array([row["local_elem_max"] for row in rows], dtype=float)
    elem_mean = 0.5 * (elem_min + elem_max)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(ranks, eff, marker="o", linewidth=2.0, color="tab:red", label="Efficiency")
    ax.axhline(0.7, linestyle=":", color="tab:red", alpha=0.8, label="70% target")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, labels)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="both", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(ranks, dup, marker="s", linewidth=2.0, color="tab:blue", label="Element duplication")
    ax2.plot(
        ranks,
        overlap,
        marker="d",
        linewidth=2.0,
        color="tab:green",
        label="Overlap DOF factor",
    )
    ax2.set_ylabel("Overlap factor")
    lines, labels_a = ax.get_legend_handles_labels()
    lines2, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels_a + labels_b, frameon=False, loc="upper left")
    ax.set_title("Efficiency vs Overlap Growth")

    ax = axes[1]
    x = np.arange(len(rows))
    ax.bar(x, elem_max, color="#88CCEE", label="max local elements")
    ax.bar(x, elem_min, color="#4477AA", label="min local elements")
    ax.plot(x, elem_mean, color="black", marker="o", linewidth=1.5, label="mean")
    ax.set_xticks(x, labels)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Local elements per rank")
    ax.set_title("Rank-Local Element Spread")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    _save(fig, "overlap_and_efficiency.png")


def plot_phase_breakdown(rows: list[dict], phase_records: list[dict]) -> None:
    ranks = np.array([row["np"] for row in rows], dtype=float)
    labels = [str(int(v)) for v in ranks]
    top = phase_records[:8]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))

    ax = axes[0]
    bottom = np.zeros(len(rows), dtype=float)
    cmap = plt.get_cmap("tab20")
    for idx, rec in enumerate(top):
        values = np.array(rec["times"], dtype=float)
        ax.bar(
            labels,
            values,
            bottom=bottom,
            color=cmap(idx % 20),
            label=rec["label"],
        )
        bottom += values
    total = np.array([row["total_time"] for row in rows], dtype=float)
    residual = np.maximum(total - bottom, 0.0)
    ax.bar(labels, residual, bottom=bottom, color="#999999", label="Other measured / overlap")
    ax.set_ylabel("Time [s]")
    ax.set_title("Fixed-Work Impact Breakdown")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    for rec in top:
        values = np.array(rec["times"], dtype=float)
        ideal = values[0] / ranks
        ax.loglog(ranks, values, marker="o", linewidth=1.8, label=rec["label"])
        ax.loglog(ranks, ideal, linestyle="--", alpha=0.25, color=ax.lines[-1].get_color())
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    ax.set_xticks(ranks, labels)
    ax.set_title("Top Phase Scaling With Ideal References")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    _save(fig, "phase_breakdown.png")


def plot_phase_scaling_grid(rows: list[dict], phase_records: list[dict]) -> None:
    ranks = np.array([row["np"] for row in rows], dtype=float)
    labels = [str(int(v)) for v in ranks]
    cols = 3
    n = len(phase_records)
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(16, 4.0 * rows_n), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, rec in zip(axes.flat, phase_records, strict=False):
        ax.set_visible(True)
        values = np.array(rec["times"], dtype=float)
        ideal = values[0] / ranks
        ax.loglog(ranks, values, marker="o", linewidth=2.0, label="Measured")
        ax.loglog(ranks, ideal, linestyle="--", color="0.35", label="Ideal")
        ax.set_xticks(ranks, labels)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(
            f"{rec['label']}\n1r={rec['rank1']:.2f}s, 32r eff={100.0 * rec['eff32']:.1f}%",
            fontsize=10,
        )
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("Time [s]")
        ax.legend(frameon=False, fontsize=8)

    _save(fig, "phase_scaling_grid.png")


def plot_individual_phase_curves(rows: list[dict], phase_records: list[dict]) -> None:
    PER_PHASE_DIR.mkdir(parents=True, exist_ok=True)
    ranks = np.array([row["np"] for row in rows], dtype=float)
    labels = [str(int(v)) for v in ranks]
    for index, rec in enumerate(phase_records, start=1):
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.8))
        values = np.array(rec["times"], dtype=float)
        ideal = values[0] / ranks
        ax.loglog(ranks, values, marker="o", linewidth=2.2, label="Measured")
        ax.loglog(ranks, ideal, linestyle="--", color="0.35", label="Ideal")
        ax.set_xticks(ranks, labels)
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("Time [s]")
        ax.set_title(
            f"{rec['label']} | 1r={rec['rank1']:.3f}s, 32r={rec['rank32']:.3f}s, "
            f"eff32={100.0 * rec['eff32']:.1f}%"
        )
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(PER_PHASE_DIR / _phase_plot_name(index, rec["key"]), dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_full16_newton(full16: dict) -> None:
    hist = full16.get("history", [])
    if not hist:
        return

    iters = np.arange(1, len(hist) + 1, dtype=int)
    grad_start = np.array([float(row["grad_norm"]) for row in hist], dtype=float)
    grad_end = np.array(
        [
            float(row.get("grad_norm_post"))
            if row.get("grad_norm_post") is not None and np.isfinite(row.get("grad_norm_post"))
            else np.nan
            for row in hist
        ],
        dtype=float,
    )
    ksp_its = np.array([float(row["ksp_its"]) for row in hist], dtype=float)
    t_hess = np.array([float(row["t_hess"]) for row in hist], dtype=float)
    t_iter = np.array([float(row["t_iter"]) for row in hist], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.semilogy(iters, grad_start, marker="o", linewidth=1.8, label="Gradient at iteration start")
    ax.semilogy(iters, grad_end, marker="s", linewidth=1.8, label="Gradient after step")
    ax.axhline(1.0e-2, linestyle=":", color="tab:red", label="Stopping target")
    ax.set_ylabel(r"$||g||$")
    ax.set_title("16-Rank Full-Solve Newton Convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(iters, ksp_its, marker="o", linewidth=1.8, label="KSP iterations")
    ax.plot(iters, t_hess, marker="s", linewidth=1.8, label="Hessian phase [s]")
    ax.plot(iters, t_iter, marker="^", linewidth=1.8, label="Newton iteration [s]")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Count / time")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    _save(fig, "full16_newton_convergence.png")


def plot_full16_ranklocal(full16: dict) -> None:
    pd = full16.get("parallel_diagnostics", [])
    if not pd:
        return

    ranks = np.array([int(row["rank"]) for row in pd], dtype=int)
    local_elements = np.array(
        [int(row["local_problem"]["local_elements"]) for row in pd], dtype=float
    )
    overlap_dofs = np.array(
        [int(row["local_problem"]["overlap_total_dofs"]) for row in pd], dtype=float
    )
    owned_dofs = np.array(
        [int(row["local_problem"]["owned_free_dofs"]) for row in pd], dtype=float
    )
    hessian_total = np.array(
        [float(row["assembly_callbacks"]["hessian"]["total"]) for row in pd], dtype=float
    )
    linear_solve_total = np.array(
        [sum(float(rec["t_solve"]) for rec in row["linear_history"]) for row in pd],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.bar(ranks - 0.2, local_elements, width=0.4, label="Local elements", color="#5DA5DA")
    ax2 = ax.twinx()
    ax2.bar(
        ranks + 0.2,
        overlap_dofs / owned_dofs,
        width=0.4,
        label="Overlap/owned DOFs",
        color="#F17CB0",
    )
    ax.set_ylabel("Elements")
    ax2.set_ylabel("Overlap factor")
    ax.set_title("16-Rank Local Work Distribution")
    ax.grid(True, axis="y", alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")

    ax = axes[1]
    ax.bar(ranks - 0.2, hessian_total, width=0.4, label="Hessian total", color="#60BD68")
    ax.bar(ranks + 0.2, linear_solve_total, width=0.4, label="Linear solve total", color="#B2912F")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Time [s]")
    ax.set_title("16-Rank Rank-Local Hot-Path Time")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    _save(fig, "full16_ranklocal_distribution.png")


def _report_text(rows: list[dict], phase_records: list[dict], full16: dict) -> str:
    sweep_lines = "\n".join(
        f"- [`np{int(row['np'])}/output.json`](./p4_cheby_jacobi5_maxit1_v3_threads1/np{int(row['np'])}/output.json)"
        for row in rows
    )
    total_table = "\n".join(
        f"| {int(row['np'])} | {row['total_time']:.3f} | {row['solve_time']:.3f} | {row['speedup']:.3f} | {row['efficiency']:.3f} |"
        for row in rows
    )
    overlap_table = "\n".join(
        f"| {int(row['np'])} | {row['local_elem_min']:.0f} | {row['local_elem_max']:.0f} | "
        f"{row['dup_factor']:.3f} | {row['overlap_dof_factor']:.3f} |"
        for row in rows
    )
    phase_table = "\n".join(
        f"| {idx} | {rec['label']} | {rec['rank1']:.3f} | {rec['rank32']:.3f} | "
        f"{rec['speedup32']:.3f} | {rec['eff32']:.3f} | {100.0 * rec['share_rank1']:.1f}% |"
        for idx, rec in enumerate(phase_records, start=1)
    )
    per_phase_blocks = "\n\n".join(
        (
            f"### {idx}. {rec['label']}\n\n"
            f"![{rec['label']}](./p4_lambda1p5_parallel_assets/per_phase/{_phase_plot_name(idx, rec['key'])})"
        )
        for idx, rec in enumerate(phase_records, start=1)
    )

    full16_status = full16.get("status", "unknown")
    full16_nit = int(full16.get("nit", 0))
    full16_grad = float(full16.get("final_grad_norm", math.nan))
    full16_total = float(full16.get("total_time", math.nan))
    full16_solve = float(full16.get("solve_time", math.nan))

    return f"""# `P4(L1), lambda = 1.5` Parallel Scaling Report

## Setup

- Problem: `hetero_ssr_L1`
- Discretization: same-mesh `P4 -> P2 -> P1`
- Nonlinear stack: Newton + Armijo line search, elastic initial guess, pure plastic tangent
- Linear stack: `fgmres + PMG`, Hypre coarse level with near-nullspace
- Distribution: `overlap_p2p`
- Problem build mode: `rank_local`
- Transfer build mode: `owned_rows`
- Reorder mode: `block_xyz`
- Fixed-work benchmark definition: `maxit = 1`
- Thread caps: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`
- JAX CPU cap: `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count=1"`
- CLI thread setting: `--nproc 1`

This report reruns the fixed-work strong-scaling sweep on `1/2/4/8/16/32` MPI ranks with one CPU thread per MPI rank. The older `v2` sweep is intentionally not mixed into the timing tables below.

Important timing note:
- `problem_load` includes the cold-start HDF5 read plus rank-local partition/data slicing for the `P4` snapshot. It is a real startup cost, but it is more cache-sensitive than the warmed nonlinear kernel phases.

Outputs:
{sweep_lines}

## Overview

![scaling overview](./p4_lambda1p5_parallel_assets/scaling_overview_loglog.png)

| MPI ranks | total time [s] | solve time [s] | speedup | efficiency |
| ---: | ---: | ---: | ---: | ---: |
{total_table}

Key takeaways:
- Scaling is good through `4` ranks and then bends sharply.
- `16 -> 32` gives almost no end-to-end improvement: `44.794 s -> 43.296 s`.
- The `32`-rank point is already overlap-dominated, so the ideal line is no longer a useful operational target for the current overlap-local algorithm.

## Impact-Sorted Phase Scaling

![impact-sorted phase scaling grid](./p4_lambda1p5_parallel_assets/phase_scaling_grid.png)

![phase breakdown](./p4_lambda1p5_parallel_assets/phase_breakdown.png)

All phase timings below use the maximum rank-local time for that phase. The list is sorted by the `1`-rank cost so the highest-impact parts appear first.

| Rank | Phase | 1 rank [s] | 32 ranks [s] | speedup | efficiency | 1-rank share of total |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
{phase_table}

### Per-Phase Log-Log Curves

Each plot below shows the measured scaling curve and the ideal `1 / np` reference for that same phase.

{per_phase_blocks}

## Overlap / Duplication

![overlap and efficiency](./p4_lambda1p5_parallel_assets/overlap_and_efficiency.png)

| MPI ranks | local elements min | local elements max | element duplication factor | overlap DOF factor |
| ---: | ---: | ---: | ---: | ---: |
{overlap_table}

The main structural limiter is still overlap growth:
- by `16` ranks the code does about `1.616x` the global element work
- by `32` ranks that rises to about `2.246x`
- the overlap DOF footprint grows from `1.015x` at `1` rank to `2.588x` at `32` ranks

This is why several hot phases flatten or even regress between `16` and `32` ranks.

## Full 16-Rank Reference Solve

The strong-scaling sweep above is fixed-work. For a real nonlinear reference point, the currently best converged full run is still the patched `16`-rank solve:

- status: `{full16_status}`
- Newton iterations: `{full16_nit}`
- final gradient norm: `{full16_grad:.6e}`
- solve time: `{full16_solve:.3f} s`
- total time: `{full16_total:.3f} s`
- output: [`output.json`](./p4_l1_lambda1p5_np16_cheby_full_v2/output.json)

![16-rank Newton convergence](./p4_lambda1p5_parallel_assets/full16_newton_convergence.png)

![16-rank rank-local work distribution](./p4_lambda1p5_parallel_assets/full16_ranklocal_distribution.png)

## Conclusions

- The new one-thread-per-rank sweep is now consistent across `1/2/4/8/16/32` MPI ranks.
- The main cost centers are `assembler_create`, `first linear KSP solve`, `hessian total`, and `first linear assemble`.
- The worst `32`-rank scaling regressions are in phases that are sensitive to overlap duplication or communication-heavy sparse assembly.
- The strongest `32`-rank regressions are visible in:
  - `first linear KSP solve`
  - `energy callbacks total`
  - `gradient callbacks total`
  - `hessian COO assembly`
- The practical strong-scaling knee for this current implementation is around `8-16` MPI ranks, not `32`.
"""


def write_summary_json(rows: list[dict], phase_records: list[dict]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "phase_records": phase_records}
    (OUTDIR / "phase_scaling_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    rows = _collect_rows()
    phase_records = _phase_records(rows)
    full16 = _load_json(FULL16)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    plot_scaling_overview(rows)
    plot_phase_breakdown(rows, phase_records)
    plot_phase_scaling_grid(rows, phase_records)
    plot_individual_phase_curves(rows, phase_records)
    plot_overlap_and_efficiency(rows)
    plot_full16_newton(full16)
    plot_full16_ranklocal(full16)
    write_summary_json(rows, phase_records)
    REPORT.write_text(_report_text(rows, phase_records, full16), encoding="utf-8")


if __name__ == "__main__":
    main()
