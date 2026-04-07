from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


FIXED_BASE = Path(os.environ.get("P4_L1_2_SCALING_ROOT", "artifacts/raw_results/scaling_probe/p4_l1_2_uniform_tail_maxit1_threads1"))
ROOT = FIXED_BASE.parent
OUTDIR = Path(os.environ.get("P4_L1_2_SCALING_OUTDIR", str(ROOT / "p4_l1_2_uniform_tail_assets")))
PER_PHASE_DIR = OUTDIR / "per_phase"
REPORT = Path(os.environ.get("P4_L1_2_SCALING_REPORT", str(ROOT / "P4_L1_2_uniform_tail_parallel_scaling.md")))
GLOBAL_ELEMS = 147352.0
FIXED_BASE_NAME = FIXED_BASE.name
OUTDIR_NAME = OUTDIR.name


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    label: str


PHASE_SPECS = (
    PhaseSpec("total_time", "Total end-to-end"),
    PhaseSpec("solve_time", "Nonlinear solve"),
    PhaseSpec("problem_load", "Problem load"),
    PhaseSpec("assembler_create", "Assembler create"),
    PhaseSpec("mg_hierarchy_build", "MG hierarchy build"),
    PhaseSpec("initial_guess_total", "Elastic initial guess"),
    PhaseSpec("hessian_total", "Hessian callbacks total"),
    PhaseSpec("hessian_hvp", "Hessian HVP compute"),
    PhaseSpec("hessian_extraction", "Hessian extraction"),
    PhaseSpec("hessian_coo", "Hessian COO assembly"),
    PhaseSpec("energy_total", "Energy callbacks total"),
    PhaseSpec("gradient_total", "Gradient callbacks total"),
    PhaseSpec("linear1_t_assemble", "First linear assemble"),
    PhaseSpec("linear1_t_setup", "First linear KSP setup"),
    PhaseSpec("linear1_t_solve", "First linear KSP solve"),
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
        "assembler_create": max(float(rank["stage_timings"].get("assembler_create", 0.0)) for rank in pd),
        "mg_hierarchy_build": max(float(rank["stage_timings"].get("mg_hierarchy_build", 0.0)) for rank in pd),
        "initial_guess_total": max(float(rank["stage_timings"].get("initial_guess_total", 0.0)) for rank in pd),
        "energy_total": max(float(rank["assembly_callbacks"]["energy"]["total"]) for rank in pd),
        "gradient_total": max(float(rank["assembly_callbacks"]["gradient"]["total"]) for rank in pd),
        "hessian_total": max(float(rank["assembly_callbacks"]["hessian"]["total"]) for rank in pd),
        "hessian_hvp": max(float(rank["assembly_callbacks"]["hessian"]["hvp_compute"]) for rank in pd),
        "hessian_extraction": max(float(rank["assembly_callbacks"]["hessian"]["extraction"]) for rank in pd),
        "hessian_coo": max(float(rank["assembly_callbacks"]["hessian"]["coo_assembly"]) for rank in pd),
        "linear1_t_assemble": max(float(rec["t_assemble"]) for rec in first_linear),
        "linear1_t_setup": max(float(rec["t_setup"]) for rec in first_linear),
        "linear1_t_solve": max(float(rec["t_solve"]) for rec in first_linear),
    }


def _collect_rows() -> list[dict]:
    rows: list[dict] = []
    for np_ranks in (1, 2, 4, 8, 16, 32):
        obj = _load_json(FIXED_BASE / f"np{np_ranks}" / "output.json")
        pd = obj["parallel_diagnostics"]
        row = {
            "np": float(np_ranks),
            "dup_factor": sum(int(rank["local_problem"]["local_elements"]) for rank in pd) / GLOBAL_ELEMS,
            "overlap_dof_factor": sum(int(rank["local_problem"]["overlap_total_dofs"]) for rank in pd)
            / sum(int(rank["local_problem"]["owned_free_dofs"]) for rank in pd),
            "local_elem_min": min(int(rank["local_problem"]["local_elements"]) for rank in pd),
            "local_elem_max": max(int(rank["local_problem"]["local_elements"]) for rank in pd),
            "output_path": str(FIXED_BASE / f"np{np_ranks}" / "output.json"),
        }
        row.update(_phase_values(obj))
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


def plot_overview(rows: list[dict]) -> None:
    ranks = np.array([row["np"] for row in rows], dtype=float)
    labels = [str(int(v)) for v in ranks]
    total = np.array([row["total_time"] for row in rows], dtype=float)
    solve = np.array([row["solve_time"] for row in rows], dtype=float)
    ideal_total = total[0] / ranks
    ideal_solve = solve[0] / ranks
    speedup = np.array([row["speedup"] for row in rows], dtype=float)
    efficiency = np.array([row["efficiency"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.loglog(ranks, total, marker="o", linewidth=2.0, label="Total time")
    ax.loglog(ranks, ideal_total, linestyle="--", color="0.35", label="Ideal total")
    ax.set_title("End-to-End Scaling")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    ax.set_xticks(ranks, labels)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.loglog(ranks, solve, marker="s", linewidth=2.0, label="Solve time")
    ax.loglog(ranks, ideal_solve, linestyle="--", color="0.35", label="Ideal solve")
    ax.set_title("Solve-Phase Scaling")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    ax.set_xticks(ranks, labels)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[2]
    ax.loglog(ranks, speedup, marker="o", linewidth=2.0, label="Measured speedup")
    ax.loglog(ranks, ranks, linestyle="--", color="0.35", label="Ideal speedup")
    ax.set_title("Speedup / Efficiency")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Speedup")
    ax.set_xticks(ranks, labels)
    ax.grid(True, which="both", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(ranks, efficiency, marker="d", linewidth=1.8, color="tab:red", label="Efficiency")
    ax2.axhline(0.7, linestyle=":", color="tab:red", alpha=0.8, label="70% target")
    ax2.set_ylabel("Efficiency")
    ax2.set_ylim(0.0, 1.05)
    lines, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels1 + labels2, frameon=False, loc="upper left")

    _save(fig, "scaling_overview_loglog.png")


def plot_linear_breakdown(rows: list[dict]) -> None:
    ranks = np.array([row["np"] for row in rows], dtype=float)
    labels = [str(int(v)) for v in ranks]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    linear_keys = (
        ("linear1_t_assemble", "Assemble"),
        ("linear1_t_setup", "KSP setup"),
        ("linear1_t_solve", "KSP solve"),
    )

    ax = axes[0]
    for key, label in linear_keys:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        ax.loglog(ranks, values, marker="o", linewidth=2.0, label=label)
        ax.loglog(ranks, values[0] / ranks, linestyle="--", alpha=0.25, color=ax.lines[-1].get_color())
    ax.set_title("Linear Solve Sub-Phases")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    ax.set_xticks(ranks, labels)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    width = 0.24
    x = np.arange(len(rows), dtype=float)
    for idx, (key, label) in enumerate(linear_keys):
        values = np.array([float(row[key]) for row in rows], dtype=float)
        ax.bar(x + (idx - 1) * width, values, width=width, label=label)
    ax.set_title("Linear Solve Timing Split")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    ax.set_xticks(x, labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    _save(fig, "linear_solve_breakdown.png")


def plot_overlap(rows: list[dict]) -> None:
    ranks = np.array([row["np"] for row in rows], dtype=float)
    labels = [str(int(v)) for v in ranks]
    eff = np.array([row["efficiency"] for row in rows], dtype=float)
    dup = np.array([row["dup_factor"] for row in rows], dtype=float)
    overlap = np.array([row["overlap_dof_factor"] for row in rows], dtype=float)
    elem_min = np.array([row["local_elem_min"] for row in rows], dtype=float)
    elem_max = np.array([row["local_elem_max"] for row in rows], dtype=float)

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
    ax2.plot(ranks, overlap, marker="d", linewidth=2.0, color="tab:green", label="Overlap DOF factor")
    ax2.set_ylabel("Overlap factor")
    lines, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels1 + labels2, frameon=False, loc="upper left")
    ax.set_title("Efficiency vs Overlap")

    ax = axes[1]
    x = np.arange(len(rows))
    ax.bar(x, elem_max, color="#88CCEE", label="max local elements")
    ax.bar(x, elem_min, color="#4477AA", label="min local elements")
    ax.set_xticks(x, labels)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Local elements per rank")
    ax.set_title("Rank-Local Element Spread")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    _save(fig, "overlap_and_efficiency.png")


def plot_phase_grid(rows: list[dict], phase_records: list[dict]) -> None:
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
        ax.loglog(ranks, values, marker="o", linewidth=2.0, label="Measured")
        ax.loglog(ranks, values[0] / ranks, linestyle="--", color="0.35", label="Ideal")
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


def plot_individual_phases(rows: list[dict], phase_records: list[dict]) -> None:
    PER_PHASE_DIR.mkdir(parents=True, exist_ok=True)
    ranks = np.array([row["np"] for row in rows], dtype=float)
    labels = [str(int(v)) for v in ranks]
    for idx, rec in enumerate(phase_records, start=1):
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        values = np.array(rec["times"], dtype=float)
        ax.loglog(ranks, values, marker="o", linewidth=2.0, label="Measured")
        ax.loglog(ranks, values[0] / ranks, linestyle="--", color="0.35", label="Ideal")
        ax.set_xticks(ranks, labels)
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("Time [s]")
        ax.set_title(
            f"{rec['label']} | 1r={rec['rank1']:.3f}s, 32r={rec['rank32']:.3f}s, eff32={100.0 * rec['eff32']:.1f}%"
        )
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(PER_PHASE_DIR / _phase_plot_name(idx, rec["key"]), dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_summary(rows: list[dict], phase_records: list[dict]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "phase_scaling_summary.json").write_text(
        json.dumps({"rows": rows, "phase_records": phase_records}, indent=2) + "\n"
    )


def _report_text(rows: list[dict], phase_records: list[dict]) -> str:
    outputs = "\n".join(
        f"- [`np{int(row['np'])}/output.json`](./{FIXED_BASE_NAME}/np{int(row['np'])}/output.json)"
        for row in rows
    )
    total_table = "\n".join(
        f"| {int(row['np'])} | {row['total_time']:.3f} | {row['solve_time']:.3f} | {row['speedup']:.3f} | {row['efficiency']:.3f} |"
        for row in rows
    )
    linear_table = "\n".join(
        f"| {int(row['np'])} | {row['linear1_t_assemble']:.3f} | {row['linear1_t_setup']:.3f} | {row['linear1_t_solve']:.3f} |"
        for row in rows
    )
    phase_table = "\n".join(
        f"| {idx} | {rec['label']} | {rec['rank1']:.3f} | {rec['rank32']:.3f} | {rec['speedup32']:.3f} | {rec['eff32']:.3f} | {100.0 * rec['share_rank1']:.1f}% |"
        for idx, rec in enumerate(phase_records, start=1)
    )
    overlap_table = "\n".join(
        f"| {int(row['np'])} | {row['local_elem_min']:.0f} | {row['local_elem_max']:.0f} | {row['dup_factor']:.3f} | {row['overlap_dof_factor']:.3f} |"
        for row in rows
    )
    per_phase = "\n\n".join(
        f"### {idx}. {rec['label']}\n\n![{rec['label']}](./{OUTDIR_NAME}/per_phase/{_phase_plot_name(idx, rec['key'])})"
        for idx, rec in enumerate(phase_records, start=1)
    )
    return f"""# `P4(L1_2), lambda = 1.5` Parallel Scaling Report

## Setup

- Problem: `hetero_ssr_L1_2`
- Hierarchy: `P4(L1_2) -> P2(L1_2) -> P1(L1_2) -> P1(L1)`
- Linear stack: `fgmres + PMG`, Hypre coarse level with near-nullspace
- Nonlinear stack: elastic initial guess, pure plastic tangent, Armijo line search
- Distribution: `overlap_p2p`, `rank_local`, `owned_rows`, `block_xyz`
- Fixed-work definition: `maxit = 1`
- Thread caps: all BLAS/OpenMP/JAX thread counts forced to `1` per MPI rank

Outputs:
{outputs}

## Overview

![overview](./{OUTDIR_NAME}/scaling_overview_loglog.png)

| MPI ranks | total time [s] | solve time [s] | speedup | efficiency |
| ---: | ---: | ---: | ---: | ---: |
{total_table}

## Linear Solve Timing Split

![linear split](./{OUTDIR_NAME}/linear_solve_breakdown.png)

| MPI ranks | assemble [s] | KSP setup [s] | KSP solve [s] |
| ---: | ---: | ---: | ---: |
{linear_table}

## Impact-Sorted Phase Scaling

![phase grid](./{OUTDIR_NAME}/phase_scaling_grid.png)

| Rank | Phase | 1 rank [s] | 32 ranks [s] | speedup | efficiency | 1-rank share of total |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
{phase_table}

## Per-Phase Log-Log Curves

{per_phase}

## Overlap / Duplication

![overlap](./{OUTDIR_NAME}/overlap_and_efficiency.png)

| MPI ranks | local elements min | local elements max | element duplication factor | overlap DOF factor |
| ---: | ---: | ---: | ---: | ---: |
{overlap_table}
"""


def main() -> None:
    rows = _collect_rows()
    phase_records = _phase_records(rows)
    plot_overview(rows)
    plot_linear_breakdown(rows)
    plot_overlap(rows)
    plot_phase_grid(rows, phase_records)
    plot_individual_phases(rows, phase_records)
    write_summary(rows, phase_records)
    REPORT.write_text(_report_text(rows, phase_records), encoding="utf-8")


if __name__ == "__main__":
    main()
