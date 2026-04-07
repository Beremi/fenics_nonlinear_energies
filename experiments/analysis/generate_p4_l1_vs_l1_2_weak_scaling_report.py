from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


H5_BASE = Path("data/meshes/SlopeStability3D/hetero_ssr")

PHASE_SPECS = (
    ("total_time", "Total end-to-end"),
    ("solve_time", "Nonlinear solve"),
    ("problem_load", "Problem load"),
    ("assembler_create", "Assembler create"),
    ("mg_hierarchy_build", "MG hierarchy build"),
    ("initial_guess_total", "Elastic initial guess"),
    ("hessian_total", "Hessian callbacks total"),
    ("hessian_hvp", "Hessian HVP compute"),
    ("hessian_extraction", "Hessian extraction"),
    ("linear1_t_assemble", "First linear assemble"),
    ("linear1_t_setup", "First linear KSP setup"),
    ("linear1_t_solve", "First linear KSP solve"),
)

NORMALIZED_SPECS = (
    ("solve_per_newton", "Nonlinear solve / Newton iter"),
    ("hessian_total_per_newton", "Hessian callbacks / Newton iter"),
    ("hessian_hvp_per_newton", "Hessian HVP / Newton iter"),
    ("hessian_extraction_per_newton", "Hessian extraction / Newton iter"),
    ("linear_assemble_per_ksp", "Cumulative linear assemble / Krylov iter"),
    ("linear_setup_per_ksp", "Cumulative linear setup / Krylov iter"),
    ("linear_solve_per_ksp", "Cumulative linear solve / Krylov iter"),
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _mesh_h5_path(mesh_name: str, degree: int) -> Path:
    return H5_BASE / f"{mesh_name}_p{degree}_same_mesh.h5"


def _global_size(mesh_name: str, degree: int) -> dict[str, int]:
    with h5py.File(_mesh_h5_path(mesh_name, degree), "r") as h5:
        return {
            "nodes": int(h5["nodes"].shape[0]),
            "elems": int(h5["elems_scalar"].shape[0]),
            "freedofs": int(h5["freedofs"].shape[0]),
            "surf_faces": int(h5["surf"].shape[0]),
        }


def _first_linear_records(diags: list[dict]) -> list[dict]:
    return [rank["linear_history"][0] for rank in diags]


def _phase_values(obj: dict) -> dict[str, float]:
    diags = obj["parallel_diagnostics"]
    first_linear = _first_linear_records(diags)
    return {
        "total_time": float(obj["total_time"]),
        "solve_time": float(obj["solve_time"]),
        "problem_load": max(float(rank["stage_timings"].get("problem_load", 0.0)) for rank in diags),
        "assembler_create": max(float(rank["stage_timings"].get("assembler_create", 0.0)) for rank in diags),
        "mg_hierarchy_build": max(float(rank["stage_timings"].get("mg_hierarchy_build", 0.0)) for rank in diags),
        "initial_guess_total": max(float(rank["stage_timings"].get("initial_guess_total", 0.0)) for rank in diags),
        "hessian_total": max(float(rank["assembly_callbacks"]["hessian"]["total"]) for rank in diags),
        "hessian_hvp": max(float(rank["assembly_callbacks"]["hessian"]["hvp_compute"]) for rank in diags),
        "hessian_extraction": max(float(rank["assembly_callbacks"]["hessian"]["extraction"]) for rank in diags),
        "linear1_t_assemble": max(float(rec["t_assemble"]) for rec in first_linear),
        "linear1_t_setup": max(float(rec["t_setup"]) for rec in first_linear),
        "linear1_t_solve": max(float(rec["t_solve"]) for rec in first_linear),
    }


def _progress_summary(path: Path) -> dict[str, float]:
    progress_path = path.with_name("progress.json")
    if not progress_path.exists():
        return {"accepted_steps": 0.0, "mean_alpha": 0.0, "min_alpha": 0.0}
    obj = _load_json(progress_path)
    hist = obj.get("history", [])
    if not hist:
        return {"accepted_steps": 0.0, "mean_alpha": 0.0, "min_alpha": 0.0}
    alphas = np.array([float(entry.get("alpha", 0.0) or 0.0) for entry in hist], dtype=float)
    return {
        "accepted_steps": float(sum(1 for entry in hist if entry.get("accepted_step"))),
        "mean_alpha": float(alphas.mean()),
        "min_alpha": float(alphas.min()),
    }


def _linear_totals(obj: dict) -> dict[str, float]:
    hist = obj.get("linear_history", [])
    return {
        "linear_assemble_total": float(sum(float(entry.get("t_assemble", 0.0) or 0.0) for entry in hist)),
        "linear_setup_total": float(sum(float(entry.get("t_setup", 0.0) or 0.0) for entry in hist)),
        "linear_solve_total": float(sum(float(entry.get("t_solve", 0.0) or 0.0) for entry in hist)),
        "linear_iterations_total": float(sum(int(entry.get("ksp_its", 0) or 0) for entry in hist)),
    }


def _local_summary(obj: dict) -> dict[str, float]:
    diags = obj["parallel_diagnostics"]
    elems = np.array([int(rank["local_problem"]["local_elements"]) for rank in diags], dtype=float)
    owned = np.array([int(rank["local_problem"]["owned_free_dofs"]) for rank in diags], dtype=float)
    overlap = np.array([int(rank["local_problem"]["overlap_total_dofs"]) for rank in diags], dtype=float)
    nnz = np.array([int(rank["local_problem"]["owned_nnz"]) for rank in diags], dtype=float)
    return {
        "ranks": float(len(diags)),
        "local_elem_min": float(elems.min()),
        "local_elem_mean": float(elems.mean()),
        "local_elem_max": float(elems.max()),
        "owned_dofs_min": float(owned.min()),
        "owned_dofs_mean": float(owned.mean()),
        "owned_dofs_max": float(owned.max()),
        "overlap_dofs_min": float(overlap.min()),
        "overlap_dofs_mean": float(overlap.mean()),
        "overlap_dofs_max": float(overlap.max()),
        "overlap_factor_mean": float(overlap.sum() / owned.sum()),
        "owned_nnz_max": float(nnz.max()),
    }


def _case_summary(path: Path) -> dict:
    obj = _load_json(path)
    mesh_name = str(obj["mesh_name"])
    degree = int(obj["elem_degree"])
    return {
        "path": str(path),
        "mesh_name": mesh_name,
        "degree": degree,
        "assembly_backend": str(obj.get("assembly_backend", "")),
        "transfer_backend": str(obj.get("transfer_backend", "")),
        "chunk_size": int(obj.get("p4_hessian_chunk_size", 0) or 0),
        "nit": int(obj.get("nit", 0) or 0),
        "solve_time": float(obj.get("solve_time", 0.0) or 0.0),
        "total_time": float(obj.get("total_time", 0.0) or 0.0),
        "final_grad_norm": float(obj.get("final_grad_norm", 0.0) or 0.0),
        "u_max": float(obj.get("u_max", 0.0) or 0.0),
        "omega": float(obj.get("omega", 0.0) or 0.0),
        "linear_iterations_total": int(obj.get("linear_iterations_total", 0) or 0),
        "global_size": _global_size(mesh_name, degree),
        "local": _local_summary(obj),
        "phases": _phase_values(obj),
        "progress": _progress_summary(path),
        "linear_totals": _linear_totals(obj),
        "status": str(obj["status"]),
        "message": str(obj["message"]),
    }


def _plot_phase_compare(base: dict, weak: dict, outdir: Path) -> None:
    labels = [label for _, label in PHASE_SPECS]
    base_vals = np.array([base["phases"][key] for key, _ in PHASE_SPECS], dtype=float)
    weak_vals = np.array([weak["phases"][key] for key, _ in PHASE_SPECS], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.barh(y - 0.18, base_vals, height=0.34, label=f"{base['mesh_name']} @ 1 rank", color="#4C78A8")
    ax.barh(y + 0.18, weak_vals, height=0.34, label=f"{weak['mesh_name']} @ 8 ranks", color="#F58518")
    ax.set_xscale("log")
    ax.set_xlabel("Time [s]")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(True, axis="x", which="both", alpha=0.25)
    ax.set_title("Weak-Scaling Phase Comparison")
    ax.legend(frameon=False)
    fig.savefig(outdir / "weak_scaling_phase_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_efficiency(base: dict, weak: dict, outdir: Path) -> None:
    labels = [label for _, label in PHASE_SPECS]
    eff = np.array([base["phases"][key] / weak["phases"][key] for key, _ in PHASE_SPECS], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.barh(y, eff, color="#54A24B")
    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.5, label="Ideal weak scaling")
    ax.set_xlabel("Weak-scaling efficiency = T(L1,1) / T(L1_2,8)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("Weak-Scaling Efficiency by Phase")
    ax.legend(frameon=False)
    fig.savefig(outdir / "weak_scaling_efficiency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _normalized_values(case: dict) -> dict[str, float]:
    nit = max(float(case["nit"]), 1.0)
    ksp_its = max(float(case["linear_totals"]["linear_iterations_total"]), 1.0)
    return {
        "solve_per_newton": float(case["phases"]["solve_time"]) / nit,
        "hessian_total_per_newton": float(case["phases"]["hessian_total"]) / nit,
        "hessian_hvp_per_newton": float(case["phases"]["hessian_hvp"]) / nit,
        "hessian_extraction_per_newton": float(case["phases"]["hessian_extraction"]) / nit,
        "linear_assemble_per_ksp": float(case["linear_totals"]["linear_assemble_total"]) / ksp_its,
        "linear_setup_per_ksp": float(case["linear_totals"]["linear_setup_total"]) / ksp_its,
        "linear_solve_per_ksp": float(case["linear_totals"]["linear_solve_total"]) / ksp_its,
    }


def _plot_normalized_efficiency(base: dict, weak: dict, outdir: Path) -> None:
    base_vals_map = _normalized_values(base)
    weak_vals_map = _normalized_values(weak)
    labels = [label for _, label in NORMALIZED_SPECS]
    eff = np.array([base_vals_map[key] / weak_vals_map[key] for key, _ in NORMALIZED_SPECS], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(y, eff, color="#9C755F")
    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.5, label="Ideal normalized weak scaling")
    ax.set_xlabel("Normalized efficiency")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("Work-Normalized Efficiency")
    ax.legend(frameon=False)
    fig.savefig(outdir / "weak_scaling_work_normalized_efficiency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fmt_int(val: float) -> str:
    return f"{int(round(val)):,}"


def _report_text(base: dict, weak: dict) -> str:
    base_size = base["global_size"]
    weak_size = weak["global_size"]
    base_local = base["local"]
    weak_local = weak["local"]

    size_rows = [
        ("nodes", base_size["nodes"], weak_size["nodes"]),
        ("tetrahedra", base_size["elems"], weak_size["elems"]),
        ("free DOFs", base_size["freedofs"], weak_size["freedofs"]),
        ("surface faces", base_size["surf_faces"], weak_size["surf_faces"]),
    ]
    size_table = "\n".join(
        f"| {label} | {_fmt_int(v1)} | {_fmt_int(v2)} | {v2 / v1:.3f}x |"
        for label, v1, v2 in size_rows
    )

    local_table = "\n".join(
        [
            f"| local tetrahedra | {_fmt_int(base_local['local_elem_mean'])} | {_fmt_int(weak_local['local_elem_mean'])} | {weak_local['local_elem_mean'] / base_local['local_elem_mean']:.3f}x |",
            f"| local tetrahedra max | {_fmt_int(base_local['local_elem_max'])} | {_fmt_int(weak_local['local_elem_max'])} | {weak_local['local_elem_max'] / base_local['local_elem_max']:.3f}x |",
            f"| owned free DOFs mean | {_fmt_int(base_local['owned_dofs_mean'])} | {_fmt_int(weak_local['owned_dofs_mean'])} | {weak_local['owned_dofs_mean'] / base_local['owned_dofs_mean']:.3f}x |",
            f"| owned free DOFs max | {_fmt_int(base_local['owned_dofs_max'])} | {_fmt_int(weak_local['owned_dofs_max'])} | {weak_local['owned_dofs_max'] / base_local['owned_dofs_max']:.3f}x |",
            f"| overlap factor | {base_local['overlap_factor_mean']:.3f} | {weak_local['overlap_factor_mean']:.3f} | {weak_local['overlap_factor_mean'] / base_local['overlap_factor_mean']:.3f}x |",
            f"| owned NNZ max | {_fmt_int(base_local['owned_nnz_max'])} | {_fmt_int(weak_local['owned_nnz_max'])} | {weak_local['owned_nnz_max'] / base_local['owned_nnz_max']:.3f}x |",
        ]
    )

    timing_table = "\n".join(
        f"| {label} | {base['phases'][key]:.3f} | {weak['phases'][key]:.3f} | {base['phases'][key] / weak['phases'][key]:.3f} |"
        for key, label in PHASE_SPECS
    )
    base_norm = _normalized_values(base)
    weak_norm = _normalized_values(weak)
    normalized_table = "\n".join(
        f"| {label} | {base_norm[key]:.6f} | {weak_norm[key]:.6f} | {base_norm[key] / weak_norm[key]:.3f} |"
        for key, label in NORMALIZED_SPECS
    )
    linear_iter_ratio = weak["linear_iterations_total"] / max(base["linear_iterations_total"], 1)
    alpha_ratio = weak["progress"]["mean_alpha"] / max(base["progress"]["mean_alpha"], 1e-15)

    if base["nit"] == weak["nit"] and base["nit"] > 1:
        benchmark_mode = f"capped Newton window `maxit = {base['nit']}`"
    else:
        benchmark_mode = "fixed-work `maxit = 1`"

    return f"""# `P4(L1)` vs `P4(L1_2)` Weak Scaling Report

## Setup

This compares the current maintained backend on:

- `P4(L1)` on `1` MPI rank
- `P4(L1_2)` on `8` MPI ranks

The intent is a weak-scaling style comparison because `L1_2` is one uniform
tetra refinement of `L1`, so it has `8x` more macro elements and the `8`-rank
case keeps the local element count in the same ballpark.

Common maintained settings:

- fine assembly backend: `{base['assembly_backend']}`
- transfer backend: `{base['transfer_backend']}`
- `P4` chunk size: `{base['chunk_size']}`
- linear stack: `fgmres + PMG + hypre coarse`
- nonlinear stack: elastic initial guess, pure plastic tangent, Armijo
- benchmark mode: {benchmark_mode}
- thread caps: `OMP/JAX/BLAS = 1` thread per rank

Hierarchy note:

- `L1` uses the same-mesh stack `P4(L1) -> P2(L1) -> P1(L1)`
- `L1_2` uses the maintained refined-tail stack
  `P4(L1_2) -> P2(L1_2) -> P1(L1_2) -> P1(L1)`

So this is a very informative weak-scaling comparison for the fine and repeated
nonlinear work, but not a perfectly identical MG hierarchy comparison.

Inputs:

- base: [{base['path']}]({Path(base['path']).resolve()})
- weak: [{weak['path']}]({Path(weak['path']).resolve()})

## Global Problem Size

| quantity | `L1` | `L1_2` | ratio |
| --- | ---: | ---: | ---: |
{size_table}

## Rank-Local Work

| quantity | `L1` on 1 rank | `L1_2` on 8 ranks | ratio |
| --- | ---: | ---: | ---: |
{local_table}

The average owned DOF count per rank stays very close to flat, but the maximum
local element count on `8` ranks is about `22%` higher than the `L1` single-rank
work unit because of load imbalance plus overlap.

## Outcome Summary

| quantity | `L1` on 1 rank | `L1_2` on 8 ranks |
| --- | ---: | ---: |
| status | `{base['status']}` | `{weak['status']}` |
| nonlinear iterations | `{base['nit']}` | `{weak['nit']}` |
| solve time [s] | `{base['solve_time']:.3f}` | `{weak['solve_time']:.3f}` |
| total time [s] | `{base['total_time']:.3f}` | `{weak['total_time']:.3f}` |
| final gradient norm | `{base['final_grad_norm']:.6e}` | `{weak['final_grad_norm']:.6e}` |
| `u_max` | `{base['u_max']:.6f}` | `{weak['u_max']:.6f}` |
| `omega` | `{base['omega']:.6f}` | `{weak['omega']:.6f}` |
| linear iterations total | `{base['linear_iterations_total']}` | `{weak['linear_iterations_total']}` |
| accepted Newton steps | `{int(base['progress']['accepted_steps'])}` | `{int(weak['progress']['accepted_steps'])}` |
| mean Armijo `alpha` | `{base['progress']['mean_alpha']:.6f}` | `{weak['progress']['mean_alpha']:.6f}` |
| minimum Armijo `alpha` | `{base['progress']['min_alpha']:.6f}` | `{weak['progress']['min_alpha']:.6f}` |

## Timing Comparison

![Weak-scaling phase comparison](assets/weak_scaling_phase_compare.png)

![Weak-scaling efficiency](assets/weak_scaling_efficiency.png)

Weak-scaling efficiency is reported as:

`T(L1, 1 rank) / T(L1_2, 8 ranks)`

An ideal weak-scaling result is therefore `1.0`.

| phase | `L1` 1 rank [s] | `L1_2` 8 ranks [s] | weak-scaling efficiency |
| --- | ---: | ---: | ---: |
{timing_table}

## Convergence-Hardness Nuance

The raw weak-scaling table above mixes two effects:

1. true parallel scaling on the larger weak-scaled problem, and
2. the refined case being harder for the nonlinear and Krylov solvers.

For this `maxit = 20` pair, both runs executed the same number of Newton
iterations and accepted all `20` steps, but the refined `L1_2 @ 8` case still
required more inner work:

- total linear iterations: `{base['linear_iterations_total']} -> {weak['linear_iterations_total']}` (`{linear_iter_ratio:.3f}x`)
- mean Armijo step length: `{base['progress']['mean_alpha']:.6f} -> {weak['progress']['mean_alpha']:.6f}` (`{alpha_ratio:.3f}x`)
- minimum Armijo step length: `{base['progress']['min_alpha']:.6f} -> {weak['progress']['min_alpha']:.6f}`

So the raw timing gap is not purely a weak-scaling story; part of it is that
the refined problem is genuinely a tougher nonlinear solve.

## Work-Normalized Comparison

This second view keeps the raw comparison above, but normalizes repeated work by
the amount of solver effort performed:

- per-Newton iteration for nonlinear repeated phases
- per-Krylov iteration for cumulative linear assembly/setup/solve

![Work-normalized efficiency](assets/weak_scaling_work_normalized_efficiency.png)

| metric | `L1` 1 rank | `L1_2` 8 ranks | normalized efficiency |
| --- | ---: | ---: | ---: |
{normalized_table}

This is the fairer lens for the `maxit = 20` run pair. It shows that the
repeated Hessian path remains relatively healthy, while the biggest remaining
loss is in Krylov-coupled work per linear iteration.

## Summary

- Total end-to-end weak-scaling efficiency: `{base['phases']['total_time'] / weak['phases']['total_time']:.3f}`
- Nonlinear solve weak-scaling efficiency: `{base['phases']['solve_time'] / weak['phases']['solve_time']:.3f}`
- First linear assemble weak-scaling efficiency: `{base['phases']['linear1_t_assemble'] / weak['phases']['linear1_t_assemble']:.3f}`
- First linear KSP solve weak-scaling efficiency: `{base['phases']['linear1_t_solve'] / weak['phases']['linear1_t_solve']:.3f}`
- Hessian extraction weak-scaling efficiency: `{base['phases']['hessian_extraction'] / weak['phases']['hessian_extraction']:.3f}`
- Hessian HVP weak-scaling efficiency: `{base['phases']['hessian_hvp'] / weak['phases']['hessian_hvp']:.3f}`
- Cumulative linear solve time per Krylov iteration efficiency: `{base_norm['linear_solve_per_ksp'] / weak_norm['linear_solve_per_ksp']:.3f}`
- Hessian extraction time per Newton iteration efficiency: `{base_norm['hessian_extraction_per_newton'] / weak_norm['hessian_extraction_per_newton']:.3f}`

The most important interpretation is whether the repeated nonlinear pieces stay
near-flat when the global problem grows by `8x`. On this benchmark, the best
weak-scaling behavior is in extraction-like callback work, while the worst
pressure still comes from Krylov and setup-side metadata. The work-normalized
view also makes clear that part of the raw gap is caused by the refined case
needing more inner solver work, not just by weaker parallel efficiency.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", type=Path, required=True)
    parser.add_argument("--weak-json", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()

    base = _case_summary(args.base_json)
    weak = _case_summary(args.weak_json)

    args.asset_dir.mkdir(parents=True, exist_ok=True)
    _plot_phase_compare(base, weak, args.asset_dir)
    _plot_efficiency(base, weak, args.asset_dir)
    _plot_normalized_efficiency(base, weak, args.asset_dir)
    args.report_path.write_text(_report_text(base, weak), encoding="utf-8")


if __name__ == "__main__":
    main()
