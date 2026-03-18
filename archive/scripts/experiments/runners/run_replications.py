#!/usr/bin/env python3
"""Run the maintained replication campaign into one self-contained folder."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

from src.core.benchmark.replication import (
    now_iso,
    read_json,
    run_logged_command,
    write_json,
)
from src.core.benchmark.results import (
    summarize_load_step_case,
    summarize_pure_jax_load_step_case,
    summarize_single_step_case,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
MPIEXEC = shutil.which("mpiexec") or "mpiexec"
DEFAULT_OUT_DIR = REPO_ROOT / "replications" / "2026-03-16_maintained_refresh"
ONLY_CHOICES = ("examples", "suites", "speed", "model-cards", "reports")
THREAD_ENV = {
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}

TOPOLOGY_SERIAL_CASE = {
    "nx": 192,
    "ny": 96,
    "length": 2.0,
    "height": 1.0,
    "traction": 1.0,
    "load_fraction": 0.2,
    "fixed_pad_cells": 16,
    "load_pad_cells": 16,
    "volume_fraction_target": 0.4,
    "theta_min": 1e-3,
    "solid_latent": 10.0,
    "young": 1.0,
    "poisson": 0.3,
    "alpha_reg": 0.005,
    "ell_pf": 0.08,
    "mu_move": 0.01,
    "beta_lambda": 12.0,
    "volume_penalty": 10.0,
    "p_start": 1.0,
    "p_max": 4.0,
    "p_increment": 0.5,
    "continuation_interval": 20,
    "outer_maxit": 180,
    "outer_tol": 0.02,
    "volume_tol": 0.001,
    "mechanics_maxit": 200,
    "design_maxit": 400,
    "tolf": 1e-6,
    "tolg": 1e-3,
    "ksp_rtol": 1e-2,
    "ksp_max_it": 80,
}

TOPOLOGY_PARALLEL_COMPARISON_CASE = {
    "nx": 192,
    "ny": 96,
    "length": 2.0,
    "height": 1.0,
    "traction": 1.0,
    "load_fraction": 0.2,
    "fixed_pad_cells": 16,
    "load_pad_cells": 16,
    "volume_fraction_target": 0.4,
    "theta_min": 1e-3,
    "solid_latent": 10.0,
    "young": 1.0,
    "poisson": 0.3,
    "alpha_reg": 0.005,
    "ell_pf": 0.08,
    "mu_move": 0.01,
    "beta_lambda": 12.0,
    "volume_penalty": 10.0,
    "p_start": 1.0,
    "p_max": 4.0,
    "p_increment": 0.5,
    "continuation_interval": 20,
    "outer_maxit": 180,
    "outer_tol": 0.02,
    "volume_tol": 0.001,
    "stall_theta_tol": 1e-6,
    "stall_p_min": 4.0,
    "design_maxit": 20,
    "tolf": 1e-6,
    "tolg": 1e-3,
    "linesearch_tol": 0.1,
    "mechanics_ksp_type": "fgmres",
    "mechanics_pc_type": "gamg",
    "mechanics_ksp_rtol": 1e-4,
    "mechanics_ksp_max_it": 100,
    "design_gd_line_search": "golden_adaptive",
    "design_gd_adaptive_window_scale": 2.0,
}

TOPOLOGY_PARALLEL_FINAL_ARGS = [
    "--nx", "768",
    "--ny", "384",
    "--length", "2.0",
    "--height", "1.0",
    "--traction", "1.0",
    "--load_fraction", "0.2",
    "--fixed_pad_cells", "32",
    "--load_pad_cells", "32",
    "--volume_fraction_target", "0.4",
    "--theta_min", "1e-6",
    "--solid_latent", "10.0",
    "--young", "1.0",
    "--poisson", "0.3",
    "--alpha_reg", "0.005",
    "--ell_pf", "0.08",
    "--mu_move", "0.01",
    "--beta_lambda", "12.0",
    "--volume_penalty", "10.0",
    "--p_start", "1.0",
    "--p_max", "10.0",
    "--p_increment", "0.2",
    "--continuation_interval", "1",
    "--outer_maxit", "2000",
    "--outer_tol", "0.02",
    "--volume_tol", "0.001",
    "--stall_theta_tol", "1e-6",
    "--stall_p_min", "4.0",
    "--design_maxit", "20",
    "--tolf", "1e-6",
    "--tolg", "1e-3",
    "--linesearch_tol", "0.1",
    "--linesearch_relative_to_bound",
    "--design_gd_line_search", "golden_adaptive",
    "--design_gd_adaptive_window_scale", "2.0",
    "--mechanics_ksp_type", "fgmres",
    "--mechanics_pc_type", "gamg",
    "--mechanics_ksp_rtol", "1e-4",
    "--mechanics_ksp_max_it", "100",
    "--quiet",
    "--print_outer_iterations",
    "--save_outer_state_history",
    "--outer_snapshot_stride", "2",
]


@dataclass(slots=True)
class TaskSpec:
    id: str
    family: str
    category: str
    kind: str
    source: str
    leaf_dir: Path
    command: list[str]
    outputs: list[Path]
    env: dict[str, str]
    context: dict[str, Any]
    summarize: Callable[[Path, dict[str, Any]], dict[str, Any]]
    notes: str = ""


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _json_rel(path: Path) -> str:
    return str(path.resolve())


def _python_cmd(script: str, *args: str) -> list[str]:
    return [str(PYTHON), "-u", script, *args]


def _mpi_python_cmd(nprocs: int, script: str, *args: str) -> list[str]:
    base = _python_cmd(script, *args)
    if nprocs <= 1:
        return base
    return [MPIEXEC, "-n", str(nprocs), *base]


def _parse_scalar_result_list(json_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(json_path)
    row = payload["results"][0]
    reason = row.get("message", row.get("converged_reason", ""))
    converged_reason = row.get("converged_reason")
    result = "completed"
    if converged_reason is not None and int(converged_reason) <= 0:
        result = "failed"
    if isinstance(reason, str) and "converged" not in reason.lower() and converged_reason is None:
        result = "completed"
    return {
        "implementation": context["implementation"],
        "family": context["family"],
        "case_id": context.get("case_id"),
        "level": int(context.get("level", row.get("mesh_level", 0))),
        "nprocs": int(context.get("nprocs", payload.get("metadata", {}).get("nprocs", 1))),
        "result": result,
        "reason": reason,
        "total_dofs": int(row.get("total_dofs", row.get("dofs", 0))),
        "wall_time_s": float(row.get("solve_time", row.get("time", 0.0))),
        "setup_time_s": float(row.get("setup_time", 0.0)),
        "total_newton_iters": int(row.get("iters", 0)),
        "total_linear_iters": int(row.get("total_ksp_its", 0)),
        "final_energy": float(row["energy"]) if "energy" in row else None,
    }


def _parse_scalar_case_payload(json_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(json_path)
    if "result" not in payload:
        payload = {
            "case": {"backend": context.get("backend", "element")},
            "result": payload,
        }
    row = summarize_single_step_case(
        context["implementation"],
        int(context["level"]),
        int(context["nprocs"]),
        payload,
    )
    return {
        "implementation": context["implementation"],
        "family": context["family"],
        "case_id": context.get("case_id"),
        "level": int(context["level"]),
        "nprocs": int(context["nprocs"]),
        "result": row["result"],
        "total_dofs": int(payload["result"].get("total_dofs", payload["result"].get("dofs", 0))),
        "free_dofs": int(payload["result"].get("free_dofs", 0)),
        "wall_time_s": float(row["total_time_s"]),
        "setup_time_s": float(payload["result"].get("setup_time", 0.0)),
        "total_newton_iters": int(row["total_newton_iters"]),
        "total_linear_iters": int(row["total_linear_iters"]),
        "final_energy": float(row["final_energy"]) if row["final_energy"] is not None else None,
    }


def _parse_load_step_payload(json_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(json_path)
    if "result" not in payload:
        payload = {
            "case": {"backend": context.get("backend", "element")},
            "result": payload,
        }
    row = summarize_load_step_case(
        context["implementation"],
        int(context["total_steps"]),
        int(context["level"]),
        int(context["nprocs"]),
        payload,
    )
    return {
        "implementation": context["implementation"],
        "family": context["family"],
        "case_id": context.get("case_id"),
        "level": int(context["level"]),
        "nprocs": int(context["nprocs"]),
        "total_steps": int(context["total_steps"]),
        "result": row["result"],
        "completed_steps": int(row["completed_steps"]),
        "total_dofs": int(payload["result"].get("total_dofs", payload["result"].get("dofs", 0))),
        "free_dofs": int(payload["result"].get("free_dofs", 0)),
        "wall_time_s": float(row["total_time_s"]),
        "setup_time_s": float(payload["result"].get("setup_time", 0.0)),
        "total_newton_iters": int(row["total_newton_iters"]),
        "total_linear_iters": int(row["total_linear_iters"]),
        "final_energy": float(row["final_energy"]) if row["final_energy"] is not None else None,
    }


def _parse_pure_jax_he_payload(json_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(json_path)
    row = summarize_pure_jax_load_step_case(payload)
    return {
        "implementation": context["implementation"],
        "family": context["family"],
        "case_id": context.get("case_id"),
        "level": int(context["level"]),
        "nprocs": 1,
        "total_steps": int(context["total_steps"]),
        "result": row["result"],
        "completed_steps": int(payload.get("steps_requested", payload.get("total_steps", 0))),
        "total_dofs": int(row["total_dofs"]),
        "free_dofs": int(row["free_dofs"]),
        "wall_time_s": float(row["time"]),
        "setup_time_s": float(payload.get("setup_time", 0.0)),
        "total_newton_iters": int(row["total_newton_iters"]),
        "total_linear_iters": int(row["total_linear_iters"]),
        "final_energy": float(payload["steps"][-1]["energy"]) if payload.get("steps") else None,
    }


def _parse_topology_payload(json_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(json_path)
    final = payload["final_metrics"]
    history = payload.get("history", [])
    total_mechanics_newton = int(sum(int(row.get("mechanics_newton_iters", 0)) for row in history))
    total_design_iters = int(sum(int(row.get("design_iters", 0)) for row in history))
    return {
        "implementation": context["implementation"],
        "family": context["family"],
        "case_id": context.get("case_id"),
        "nx": int(context["nx"]),
        "ny": int(context["ny"]),
        "nprocs": int(context["nprocs"]),
        "result": str(payload["result"]),
        "wall_time_s": float(payload["time"]),
        "setup_time_s": float(payload.get("setup_time", 0.0)),
        "outer_iterations": int(final["outer_iterations"]),
        "total_newton_iters": total_mechanics_newton,
        "total_linear_iters": int(sum(int(row.get("mechanics_ksp_its", 0)) for row in history)),
        "total_design_iters": total_design_iters,
        "final_compliance": float(final["final_compliance"]),
        "final_volume_fraction": float(final["final_volume_fraction"]),
        "final_p_penal": float(final["final_p_penal"]),
        "displacement_free_dofs": int(payload["mesh"].get("displacement_free_dofs", 0)),
        "design_free_dofs": int(payload["mesh"].get("design_free_dofs", 0)),
    }


def _parse_suite_summary(json_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(json_path)
    rows = list(payload.get("rows", []))
    results = Counter(str(row.get("result", "unknown")) for row in rows)
    return {
        "family": context["family"],
        "rows": len(rows),
        "result_counts": dict(sorted(results.items())),
        "completed_rows": int(results.get("completed", 0)),
        "failed_rows": int(results.get("failed", 0) + results.get("kill-switch", 0)),
    }


def _parse_pure_jax_suite_summary(json_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(json_path)
    rows = list(payload.get("rows", []))
    results = Counter(str(row.get("result", "unknown")) for row in rows)
    return {
        "family": context["family"],
        "rows": len(rows),
        "result_counts": dict(sorted(results.items())),
        "completed_rows": int(results.get("completed", 0)),
        "failed_rows": int(results.get("failed", 0)),
    }


def _parse_figure_outputs(asset_dir: Path, context: dict[str, Any]) -> dict[str, Any]:
    files = sorted(path.name for path in asset_dir.glob("*") if path.is_file())
    return {
        "family": context["family"],
        "files": files,
        "count": len(files),
    }


def _parse_scaling_outputs(asset_dir: Path, context: dict[str, Any]) -> dict[str, Any]:
    csv_path = asset_dir if asset_dir.name == "scaling_summary.csv" else asset_dir / "scaling_summary.csv"
    output_dir = csv_path.parent
    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    return {
        "family": context["family"],
        "rows": max(len(rows) - 1, 0),
        "files": sorted(path.name for path in output_dir.glob("*") if path.is_file()),
    }


def _build_example_tasks(out_dir: Path) -> list[TaskSpec]:
    runs_dir = out_dir / "runs" / "examples"
    tasks: list[TaskSpec] = []

    def add_task(
        *,
        task_id: str,
        family: str,
        source: str,
        command: list[str],
        output_name: str = "output.json",
        summarize: Callable[[Path, dict[str, Any]], dict[str, Any]],
        context: dict[str, Any],
        notes: str = "",
    ) -> None:
        leaf_dir = runs_dir / task_id
        output_path = leaf_dir / output_name
        tasks.append(
            TaskSpec(
                id=task_id,
                family=family,
                category="examples",
                kind="example",
                source=source,
                leaf_dir=leaf_dir,
                command=command,
                outputs=[output_path],
                env=dict(THREAD_ENV),
                context=context,
                summarize=summarize,
                notes=notes,
            )
        )

    add_task(
        task_id="plaplace_fenics_custom",
        family="plaplace",
        source="src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py",
        command=_python_cmd(
            "src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py",
            "--levels", "5",
            "--quiet",
            "--json", _rel(runs_dir / "plaplace_fenics_custom" / "output.json"),
        ),
        summarize=_parse_scalar_result_list,
        context={"family": "plaplace", "implementation": "fenics_custom", "level": 5, "nprocs": 1},
    )
    add_task(
        task_id="plaplace_fenics_snes",
        family="plaplace",
        source="src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py",
        command=_python_cmd(
            "src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py",
            "--levels", "5",
            "--json", _rel(runs_dir / "plaplace_fenics_snes" / "output.json"),
        ),
        summarize=_parse_scalar_result_list,
        context={"family": "plaplace", "implementation": "fenics_snes", "level": 5, "nprocs": 1},
    )
    add_task(
        task_id="plaplace_jax_serial",
        family="plaplace",
        source="src/problems/plaplace/jax/solve_pLaplace_jax_newton.py",
        command=_python_cmd(
            "src/problems/plaplace/jax/solve_pLaplace_jax_newton.py",
            "--levels", "5",
            "--quiet",
            "--json", _rel(runs_dir / "plaplace_jax_serial" / "output.json"),
        ),
        summarize=_parse_scalar_result_list,
        context={"family": "plaplace", "implementation": "jax_serial", "level": 5, "nprocs": 1},
    )
    add_task(
        task_id="plaplace_jax_petsc_element",
        family="plaplace",
        source="src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py",
        command=_python_cmd(
            "src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py",
            "--level", "5",
            "--profile", "reference",
            "--assembly-mode", "element",
            "--local-hessian-mode", "element",
            "--element-reorder-mode", "block_xyz",
            "--local-coloring",
            "--nproc", "1",
            "--quiet",
            "--json", _rel(runs_dir / "plaplace_jax_petsc_element" / "output.json"),
        ),
        summarize=_parse_scalar_result_list,
        context={"family": "plaplace", "implementation": "jax_petsc_element", "level": 5, "nprocs": 1},
    )

    add_task(
        task_id="gl_fenics_custom",
        family="ginzburg_landau",
        source="src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py",
        command=_python_cmd(
            "src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py",
            "--levels", "5",
            "--quiet",
            "--json", _rel(runs_dir / "gl_fenics_custom" / "output.json"),
        ),
        summarize=_parse_scalar_result_list,
        context={"family": "ginzburg_landau", "implementation": "fenics_custom", "level": 5, "nprocs": 1},
    )
    add_task(
        task_id="gl_fenics_snes",
        family="ginzburg_landau",
        source="src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py",
        command=_python_cmd(
            "src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py",
            "--levels", "5",
            "--json", _rel(runs_dir / "gl_fenics_snes" / "output.json"),
        ),
        summarize=_parse_scalar_result_list,
        context={"family": "ginzburg_landau", "implementation": "fenics_snes", "level": 5, "nprocs": 1},
    )
    add_task(
        task_id="gl_jax_petsc_element",
        family="ginzburg_landau",
        source="src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py",
        command=_python_cmd(
            "src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py",
            "--level", "5",
            "--profile", "reference",
            "--assembly_mode", "element",
            "--local_hessian_mode", "element",
            "--element_reorder_mode", "block_xyz",
            "--local_coloring",
            "--nproc", "1",
            "--quiet",
            "--out", _rel(runs_dir / "gl_jax_petsc_element" / "output.json"),
        ),
        summarize=_parse_scalar_case_payload,
        context={"family": "ginzburg_landau", "implementation": "jax_petsc_element", "level": 5, "nprocs": 1},
    )

    add_task(
        task_id="he_fenics_custom",
        family="hyperelasticity",
        source="src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py",
        command=_python_cmd(
            "src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py",
            "--level", "1",
            "--steps", "24",
            "--total-steps", "24",
            "--quiet",
            "--out", _rel(runs_dir / "he_fenics_custom" / "output.json"),
        ),
        summarize=_parse_load_step_payload,
        context={
            "family": "hyperelasticity",
            "implementation": "fenics_custom",
            "level": 1,
            "nprocs": 1,
            "total_steps": 24,
        },
    )
    add_task(
        task_id="he_fenics_snes",
        family="hyperelasticity",
        source="src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py",
        command=_python_cmd(
            "src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py",
            "--level", "1",
            "--steps", "24",
            "--total_steps", "24",
            "--ksp_type", "gmres",
            "--pc_type", "hypre",
            "--ksp_rtol", "1e-1",
            "--ksp_max_it", "500",
            "--snes_atol", "1e-3",
            "--quiet",
            "--out", _rel(runs_dir / "he_fenics_snes" / "output.json"),
        ),
        summarize=_parse_load_step_payload,
        context={
            "family": "hyperelasticity",
            "implementation": "fenics_snes",
            "level": 1,
            "nprocs": 1,
            "total_steps": 24,
        },
    )
    add_task(
        task_id="he_jax_serial",
        family="hyperelasticity",
        source="src/problems/hyperelasticity/jax/solve_HE_jax_newton.py",
        command=_python_cmd(
            "src/problems/hyperelasticity/jax/solve_HE_jax_newton.py",
            "--level", "1",
            "--steps", "24",
            "--total_steps", "24",
            "--quiet",
            "--out", _rel(runs_dir / "he_jax_serial" / "output.json"),
        ),
        summarize=_parse_pure_jax_he_payload,
        context={
            "family": "hyperelasticity",
            "implementation": "jax_serial",
            "level": 1,
            "nprocs": 1,
            "total_steps": 24,
        },
    )
    add_task(
        task_id="he_jax_petsc_element",
        family="hyperelasticity",
        source="src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py",
        command=_python_cmd(
            "src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py",
            "--level", "1",
            "--steps", "24",
            "--total_steps", "24",
            "--profile", "performance",
            "--ksp_type", "stcg",
            "--pc_type", "gamg",
            "--ksp_rtol", "1e-1",
            "--ksp_max_it", "30",
            "--gamg_threshold", "0.05",
            "--gamg_agg_nsmooths", "1",
            "--gamg_set_coordinates",
            "--use_near_nullspace",
            "--assembly_mode", "element",
            "--element_reorder_mode", "block_xyz",
            "--local_hessian_mode", "element",
            "--local_coloring",
            "--use_trust_region",
            "--trust_subproblem_line_search",
            "--linesearch_tol", "1e-1",
            "--trust_radius_init", "0.5",
            "--trust_shrink", "0.5",
            "--trust_expand", "1.5",
            "--trust_eta_shrink", "0.05",
            "--trust_eta_expand", "0.75",
            "--nproc", "1",
            "--quiet",
            "--out", _rel(runs_dir / "he_jax_petsc_element" / "output.json"),
        ),
        summarize=_parse_load_step_payload,
        context={
            "family": "hyperelasticity",
            "implementation": "jax_petsc_element",
            "level": 1,
            "nprocs": 1,
            "total_steps": 24,
        },
    )

    topology_serial_dir = runs_dir / "topology_serial_reference"
    topology_serial_json = topology_serial_dir / "report_run.json"
    topology_serial_npz = topology_serial_dir / "report_state.npz"
    tasks.append(
        TaskSpec(
            id="topology_serial_reference",
            family="topology",
            category="examples",
            kind="example",
            source="src/problems/topology/jax/solve_topopt_jax.py",
            leaf_dir=topology_serial_dir,
            command=_python_cmd(
                "src/problems/topology/jax/solve_topopt_jax.py",
                "--nx", "192",
                "--ny", "96",
                "--length", "2.0",
                "--height", "1.0",
                "--traction", "1.0",
                "--load_fraction", "0.2",
                "--fixed_pad_cells", "16",
                "--load_pad_cells", "16",
                "--volume_fraction_target", "0.4",
                "--theta_min", "0.001",
                "--solid_latent", "10.0",
                "--young", "1.0",
                "--poisson", "0.3",
                "--alpha_reg", "0.005",
                "--ell_pf", "0.08",
                "--mu_move", "0.01",
                "--beta_lambda", "12.0",
                "--volume_penalty", "10.0",
                "--p_start", "1.0",
                "--p_max", "4.0",
                "--p_increment", "0.5",
                "--continuation_interval", "20",
                "--outer_maxit", "180",
                "--outer_tol", "0.02",
                "--volume_tol", "0.001",
                "--mechanics_maxit", "200",
                "--design_maxit", "400",
                "--tolf", "1e-06",
                "--tolg", "0.001",
                "--ksp_rtol", "0.01",
                "--ksp_max_it", "80",
                "--save_outer_state_history",
                "--quiet",
                "--json_out", _rel(topology_serial_json),
                "--state_out", _rel(topology_serial_npz),
            ),
            outputs=[topology_serial_json, topology_serial_npz],
            env=dict(THREAD_ENV),
            context={"family": "topology", "implementation": "jax_serial", "nx": 192, "ny": 96, "nprocs": 1},
            summarize=_parse_topology_payload,
        )
    )

    topology_parallel_dir = runs_dir / "topology_parallel_reference"
    topology_parallel_json = topology_parallel_dir / "report_run.json"
    topology_parallel_npz = topology_parallel_dir / "report_state.npz"
    tasks.append(
        TaskSpec(
            id="topology_parallel_reference",
            family="topology",
            category="examples",
            kind="example",
            source="src/problems/topology/jax/solve_topopt_parallel.py",
            leaf_dir=topology_parallel_dir,
            command=_python_cmd(
                "src/problems/topology/jax/solve_topopt_parallel.py",
                "--nx", "192",
                "--ny", "96",
                "--length", "2.0",
                "--height", "1.0",
                "--traction", "1.0",
                "--load_fraction", "0.2",
                "--fixed_pad_cells", "16",
                "--load_pad_cells", "16",
                "--volume_fraction_target", "0.4",
                "--theta_min", "1e-3",
                "--solid_latent", "10.0",
                "--young", "1.0",
                "--poisson", "0.3",
                "--alpha_reg", "0.005",
                "--ell_pf", "0.08",
                "--mu_move", "0.01",
                "--beta_lambda", "12.0",
                "--volume_penalty", "10.0",
                "--p_start", "1.0",
                "--p_max", "4.0",
                "--p_increment", "0.5",
                "--continuation_interval", "20",
                "--outer_maxit", "180",
                "--outer_tol", "0.02",
                "--volume_tol", "0.001",
                "--stall_theta_tol", "1e-6",
                "--stall_p_min", "4.0",
                "--design_maxit", "20",
                "--tolf", "1e-6",
                "--tolg", "1e-3",
                "--linesearch_tol", "0.1",
                "--linesearch_relative_to_bound",
                "--design_gd_line_search", "golden_adaptive",
                "--design_gd_adaptive_window_scale", "2.0",
                "--mechanics_ksp_type", "fgmres",
                "--mechanics_pc_type", "gamg",
                "--mechanics_ksp_rtol", "1e-4",
                "--mechanics_ksp_max_it", "100",
                "--quiet",
                "--save_outer_state_history",
                "--json_out", _rel(topology_parallel_json),
                "--state_out", _rel(topology_parallel_npz),
            ),
            outputs=[topology_parallel_json, topology_parallel_npz],
            env=dict(THREAD_ENV),
            context={"family": "topology", "implementation": "jax_parallel", "nx": 192, "ny": 96, "nprocs": 1},
            summarize=_parse_topology_payload,
        )
    )

    return tasks


def _build_suite_tasks(out_dir: Path) -> list[TaskSpec]:
    runs_dir = out_dir / "runs"
    tasks: list[TaskSpec] = []

    def add_suite_task(
        *,
        task_id: str,
        family: str,
        source: str,
        command: list[str],
        output_dir: Path,
        expected: list[Path],
        summarize: Callable[[Path, dict[str, Any]], dict[str, Any]],
        context: dict[str, Any],
        notes: str = "",
    ) -> None:
        tasks.append(
            TaskSpec(
                id=task_id,
                family=family,
                category="suites",
                kind="suite" if "summary.json" in {path.name for path in expected} else "report",
                source=source,
                leaf_dir=output_dir,
                command=command,
                outputs=expected,
                env=dict(THREAD_ENV),
                context=context,
                summarize=summarize,
                notes=notes,
            )
        )

    plaplace_suite_dir = runs_dir / "plaplace" / "final_suite"
    add_suite_task(
        task_id="plaplace_final_suite",
        family="plaplace",
        source="experiments/runners/run_plaplace_final_suite.py",
        command=_python_cmd(
            "experiments/runners/run_plaplace_final_suite.py",
            "--out-dir", _rel(plaplace_suite_dir),
        ),
        output_dir=plaplace_suite_dir,
        expected=[plaplace_suite_dir / "summary.json"],
        summarize=_parse_suite_summary,
        context={"family": "plaplace"},
    )
    plaplace_fig_dir = runs_dir / "plaplace" / "final_report_figures"
    add_suite_task(
        task_id="plaplace_final_figures",
        family="plaplace",
        source="experiments/analysis/generate_plaplace_final_report_figures.py",
        command=_python_cmd(
            "experiments/analysis/generate_plaplace_final_report_figures.py",
            "--summary-json", _rel(plaplace_suite_dir / "summary.json"),
            "--asset-dir", _rel(plaplace_fig_dir),
        ),
        output_dir=plaplace_fig_dir,
        expected=[
            plaplace_fig_dir / "plaplace_scaling.png",
            plaplace_fig_dir / "plaplace_dof_runtime_np8.png",
            plaplace_fig_dir / "plaplace_convergence_l9_np32.png",
        ],
        summarize=_parse_figure_outputs,
        context={"family": "plaplace"},
    )

    gl_suite_dir = runs_dir / "ginzburg_landau" / "final_suite"
    add_suite_task(
        task_id="gl_final_suite",
        family="ginzburg_landau",
        source="experiments/runners/run_gl_final_suite.py",
        command=_python_cmd(
            "experiments/runners/run_gl_final_suite.py",
            "--out-dir", _rel(gl_suite_dir),
        ),
        output_dir=gl_suite_dir,
        expected=[gl_suite_dir / "summary.json"],
        summarize=_parse_suite_summary,
        context={"family": "ginzburg_landau"},
    )
    gl_fig_dir = runs_dir / "ginzburg_landau" / "final_report_figures"
    add_suite_task(
        task_id="gl_final_figures",
        family="ginzburg_landau",
        source="experiments/analysis/generate_gl_final_report_figures.py",
        command=_python_cmd(
            "experiments/analysis/generate_gl_final_report_figures.py",
            "--summary-json", _rel(gl_suite_dir / "summary.json"),
            "--asset-dir", _rel(gl_fig_dir),
        ),
        output_dir=gl_fig_dir,
        expected=[
            gl_fig_dir / "gl_scaling.png",
            gl_fig_dir / "gl_dof_runtime_np8.png",
            gl_fig_dir / "gl_convergence_l9_np32.png",
        ],
        summarize=_parse_figure_outputs,
        context={"family": "ginzburg_landau"},
    )

    he_suite_dir = runs_dir / "hyperelasticity" / "final_suite_best"
    add_suite_task(
        task_id="he_final_suite_best",
        family="hyperelasticity",
        source="experiments/runners/run_he_final_suite_best.py",
        command=_python_cmd(
            "experiments/runners/run_he_final_suite_best.py",
            "--out-dir", _rel(he_suite_dir),
            "--no-seed-known-results",
        ),
        output_dir=he_suite_dir,
        expected=[he_suite_dir / "summary.json"],
        summarize=_parse_suite_summary,
        context={"family": "hyperelasticity"},
    )
    he_pure_jax_dir = runs_dir / "hyperelasticity" / "pure_jax_suite_best"
    add_suite_task(
        task_id="he_pure_jax_suite_best",
        family="hyperelasticity",
        source="experiments/runners/run_he_pure_jax_suite_best.py",
        command=_python_cmd(
            "experiments/runners/run_he_pure_jax_suite_best.py",
            "--out-dir", _rel(he_pure_jax_dir),
        ),
        output_dir=he_pure_jax_dir,
        expected=[he_pure_jax_dir / "summary.json"],
        summarize=_parse_pure_jax_suite_summary,
        context={"family": "hyperelasticity"},
    )
    he_fig_dir = runs_dir / "hyperelasticity" / "final_report_figures"
    add_suite_task(
        task_id="he_final_figures",
        family="hyperelasticity",
        source="experiments/analysis/generate_he_final_report_figures.py",
        command=_python_cmd(
            "experiments/analysis/generate_he_final_report_figures.py",
            "--summary-json", _rel(he_suite_dir / "summary.json"),
            "--asset-dir", _rel(he_fig_dir),
        ),
        output_dir=he_fig_dir,
        expected=[
            he_fig_dir / "he_scaling_24.png",
            he_fig_dir / "he_scaling_96.png",
            he_fig_dir / "he_dof_runtime_np8_24.png",
            he_fig_dir / "he_dof_runtime_np8_96.png",
        ],
        summarize=_parse_figure_outputs,
        context={"family": "hyperelasticity"},
    )

    topology_serial_dir = runs_dir / "topology" / "serial_reference"
    add_suite_task(
        task_id="topology_serial_report",
        family="topology",
        source="experiments/analysis/generate_report_assets.py",
        command=_python_cmd(
            "experiments/analysis/generate_report_assets.py",
            "--asset-dir", _rel(topology_serial_dir),
            "--report-path", _rel(topology_serial_dir / "report.md"),
        ),
        output_dir=topology_serial_dir,
        expected=[
            topology_serial_dir / "report_run.json",
            topology_serial_dir / "report_state.npz",
            topology_serial_dir / "report_outer_history.csv",
            topology_serial_dir / "mesh_preview.png",
            topology_serial_dir / "final_state.png",
            topology_serial_dir / "convergence_history.png",
            topology_serial_dir / "density_evolution.gif",
            topology_serial_dir / "report.md",
        ],
        summarize=_parse_topology_payload,
        context={"family": "topology", "implementation": "jax_serial", "nx": 192, "ny": 96, "nprocs": 1},
    )

    topology_parallel_dir = runs_dir / "topology" / "parallel_final"
    topology_parallel_json = topology_parallel_dir / "parallel_full_run.json"
    topology_parallel_npz = topology_parallel_dir / "parallel_full_state.npz"
    tasks.append(
        TaskSpec(
            id="topology_parallel_final_solver",
            family="topology",
            category="suites",
            kind="suite",
            source="src/problems/topology/jax/solve_topopt_parallel.py",
            leaf_dir=topology_parallel_dir,
            command=[
                MPIEXEC,
                "-n",
                "32",
                *(_python_cmd(
                    "src/problems/topology/jax/solve_topopt_parallel.py",
                    *TOPOLOGY_PARALLEL_FINAL_ARGS,
                    "--outer_snapshot_dir", _rel(topology_parallel_dir / "frames"),
                    "--json_out", _rel(topology_parallel_json),
                    "--state_out", _rel(topology_parallel_npz),
                ))
            ],
            outputs=[topology_parallel_json, topology_parallel_npz],
            env=dict(THREAD_ENV),
            context={"family": "topology", "implementation": "jax_parallel", "nx": 768, "ny": 384, "nprocs": 32},
            summarize=_parse_topology_payload,
            notes="Fine-grid maintained parallel benchmark run",
        )
    )
    topology_parallel_report_dir = out_dir / "_tasks" / "topology_parallel_final_report"
    tasks.append(
        TaskSpec(
            id="topology_parallel_final_report",
            family="topology",
            category="suites",
            kind="report",
            source="experiments/analysis/generate_parallel_full_report.py",
            leaf_dir=topology_parallel_report_dir,
            command=_python_cmd(
                "experiments/analysis/generate_parallel_full_report.py",
                "--asset_dir", _rel(topology_parallel_dir),
                "--report_path", _rel(topology_parallel_dir / "report.md"),
            ),
            outputs=[
                topology_parallel_dir / "parallel_full_outer_history.csv",
                topology_parallel_dir / "final_state.png",
                topology_parallel_dir / "convergence_history.png",
                topology_parallel_dir / "density_step_history.png",
                topology_parallel_dir / "density_evolution.gif",
                topology_parallel_dir / "report.md",
            ],
            env=dict(THREAD_ENV),
            context={"family": "topology"},
            summarize=_parse_figure_outputs,
            notes="Report task writes figures into the maintained parallel benchmark directory",
        )
    )

    topology_scaling_dir = runs_dir / "topology" / "parallel_scaling"
    add_suite_task(
        task_id="topology_parallel_scaling",
        family="topology",
        source="experiments/analysis/generate_parallel_scaling_stallstop_report.py",
        command=_python_cmd(
            "experiments/analysis/generate_parallel_scaling_stallstop_report.py",
            "--asset-dir", _rel(topology_scaling_dir),
            "--report-path", _rel(topology_scaling_dir / "report.md"),
        ),
        output_dir=topology_scaling_dir,
        expected=[
            topology_scaling_dir / "scaling_summary.csv",
            topology_scaling_dir / "wall_scaling.png",
            topology_scaling_dir / "phase_scaling.png",
            topology_scaling_dir / "efficiency.png",
            topology_scaling_dir / "quality_vs_ranks.png",
            topology_scaling_dir / "report.md",
            topology_scaling_dir / "run_r01.json",
            topology_scaling_dir / "run_r02.json",
            topology_scaling_dir / "run_r04.json",
            topology_scaling_dir / "run_r08.json",
            topology_scaling_dir / "run_r16.json",
            topology_scaling_dir / "run_r32.json",
        ],
        summarize=_parse_scaling_outputs,
        context={"family": "topology"},
    )

    return tasks


def _build_speed_tasks(out_dir: Path) -> list[TaskSpec]:
    comparisons_root = out_dir / "comparisons"
    tasks: list[TaskSpec] = []

    def add_repeat_task(
        *,
        task_id: str,
        family: str,
        implementation: str,
        source: str,
        leaf_dir: Path,
        command: list[str],
        summarize: Callable[[Path, dict[str, Any]], dict[str, Any]],
        context: dict[str, Any],
    ) -> None:
        output_path = leaf_dir / "output.json"
        tasks.append(
            TaskSpec(
                id=task_id,
                family=family,
                category="speed",
                kind="speed_repeat",
                source=source,
                leaf_dir=leaf_dir,
                command=command,
                outputs=[output_path],
                env=dict(THREAD_ENV),
                context=context,
                summarize=summarize,
            )
        )

    def repeat_dirs(family: str, case_id: str, implementation: str) -> list[Path]:
        return [
            comparisons_root / family / "raw" / case_id / implementation / f"run{repeat:02d}"
            for repeat in range(1, 4)
        ]

    for repeat_index, leaf_dir in enumerate(repeat_dirs("plaplace", "l5_np1", "fenics_custom"), start=1):
        add_repeat_task(
            task_id=f"plaplace_l5_np1_fenics_custom_run{repeat_index:02d}",
            family="plaplace",
            implementation="fenics_custom",
            source="src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py",
            leaf_dir=leaf_dir,
            command=_python_cmd(
                "src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py",
                "--levels", "5",
                "--quiet",
                "--json", _rel(leaf_dir / "output.json"),
            ),
            summarize=_parse_scalar_result_list,
            context={"family": "plaplace", "implementation": "fenics_custom", "level": 5, "nprocs": 1, "case_id": "l5_np1"},
        )
        add_repeat_task(
            task_id=f"plaplace_l5_np1_fenics_snes_run{repeat_index:02d}",
            family="plaplace",
            implementation="fenics_snes",
            source="src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py",
            leaf_dir=repeat_dirs("plaplace", "l5_np1", "fenics_snes")[repeat_index - 1],
            command=_python_cmd(
                "src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py",
                "--levels", "5",
                "--json", _rel(repeat_dirs("plaplace", "l5_np1", "fenics_snes")[repeat_index - 1] / "output.json"),
            ),
            summarize=_parse_scalar_result_list,
            context={"family": "plaplace", "implementation": "fenics_snes", "level": 5, "nprocs": 1, "case_id": "l5_np1"},
        )
        add_repeat_task(
            task_id=f"plaplace_l5_np1_jax_serial_run{repeat_index:02d}",
            family="plaplace",
            implementation="jax_serial",
            source="src/problems/plaplace/jax/solve_pLaplace_jax_newton.py",
            leaf_dir=repeat_dirs("plaplace", "l5_np1", "jax_serial")[repeat_index - 1],
            command=_python_cmd(
                "src/problems/plaplace/jax/solve_pLaplace_jax_newton.py",
                "--levels", "5",
                "--quiet",
                "--json", _rel(repeat_dirs("plaplace", "l5_np1", "jax_serial")[repeat_index - 1] / "output.json"),
            ),
            summarize=_parse_scalar_result_list,
            context={"family": "plaplace", "implementation": "jax_serial", "level": 5, "nprocs": 1, "case_id": "l5_np1"},
        )
        for implementation, local_mode in (("jax_petsc_element", "element"), ("jax_petsc_local_sfd", "sfd_local")):
            leaf = repeat_dirs("plaplace", "l5_np1", implementation)[repeat_index - 1]
            add_repeat_task(
                task_id=f"plaplace_l5_np1_{implementation}_run{repeat_index:02d}",
                family="plaplace",
                implementation=implementation,
                source="src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py",
                leaf_dir=leaf,
                command=_python_cmd(
                    "src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py",
                    "--level", "5",
                    "--profile", "reference",
                    "--assembly-mode", "element",
                    "--local-hessian-mode", local_mode,
                    "--element-reorder-mode", "block_xyz",
                    "--local-coloring",
                    "--nproc", "1",
                    "--quiet",
                    "--json", _rel(leaf / "output.json"),
                ),
                summarize=_parse_scalar_result_list,
                context={"family": "plaplace", "implementation": implementation, "level": 5, "nprocs": 1, "case_id": "l5_np1"},
            )

    for nprocs in (2, 4):
        case_id = f"l5_np{nprocs}"
        for repeat_index in range(1, 4):
            for implementation, source, script, parser_name in (
                ("fenics_custom", "src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py", "src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py", _parse_scalar_result_list),
                ("fenics_snes", "src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py", "src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py", _parse_scalar_result_list),
            ):
                leaf = comparisons_root / "plaplace" / "raw" / case_id / implementation / f"run{repeat_index:02d}"
                flag_name = "--json"
                add_repeat_task(
                    task_id=f"plaplace_{case_id}_{implementation}_run{repeat_index:02d}",
                    family="plaplace",
                    implementation=implementation,
                    source=source,
                    leaf_dir=leaf,
                    command=_mpi_python_cmd(
                        nprocs,
                        script,
                        "--levels", "5",
                        *(("--quiet",) if implementation == "fenics_custom" else ()),
                        flag_name,
                        _rel(leaf / "output.json"),
                    ),
                    summarize=parser_name,
                    context={"family": "plaplace", "implementation": implementation, "level": 5, "nprocs": nprocs, "case_id": case_id},
                )
            for implementation, local_mode in (("jax_petsc_element", "element"), ("jax_petsc_local_sfd", "sfd_local")):
                leaf = comparisons_root / "plaplace" / "raw" / case_id / implementation / f"run{repeat_index:02d}"
                add_repeat_task(
                    task_id=f"plaplace_{case_id}_{implementation}_run{repeat_index:02d}",
                    family="plaplace",
                    implementation=implementation,
                    source="src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py",
                    leaf_dir=leaf,
                    command=_mpi_python_cmd(
                        nprocs,
                        "src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py",
                        "--level", "5",
                        "--profile", "reference",
                        "--assembly-mode", "element",
                        "--local-hessian-mode", local_mode,
                        "--element-reorder-mode", "block_xyz",
                        "--local-coloring",
                        "--nproc", "1",
                        "--quiet",
                        "--json", _rel(leaf / "output.json"),
                    ),
                    summarize=_parse_scalar_result_list,
                    context={"family": "plaplace", "implementation": implementation, "level": 5, "nprocs": nprocs, "case_id": case_id},
                )

    for nprocs in (1, 2, 4):
        case_id = f"l5_np{nprocs}"
        for repeat_index in range(1, 4):
            impls: list[tuple[str, str, list[str], Callable[[Path, dict[str, Any]], dict[str, Any]]]] = [
                (
                    "fenics_custom",
                    "src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py",
                    (["--levels", "5", "--quiet", "--json"]),
                    _parse_scalar_result_list,
                ),
                (
                    "fenics_snes",
                    "src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py",
                    (["--levels", "5", "--json"]),
                    _parse_scalar_result_list,
                ),
                (
                    "jax_petsc_element",
                    "src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py",
                    [
                        "--level", "5",
                        "--profile", "reference",
                        "--assembly_mode", "element",
                        "--local_hessian_mode", "element",
                        "--element_reorder_mode", "block_xyz",
                        "--local_coloring",
                        "--nproc", "1",
                        "--quiet",
                        "--out",
                    ],
                    _parse_scalar_case_payload,
                ),
                (
                    "jax_petsc_local_sfd",
                    "src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py",
                    [
                        "--level", "5",
                        "--profile", "reference",
                        "--assembly_mode", "element",
                        "--local_hessian_mode", "sfd_local",
                        "--element_reorder_mode", "block_xyz",
                        "--local_coloring",
                        "--nproc", "1",
                        "--quiet",
                        "--out",
                    ],
                    _parse_scalar_case_payload,
                ),
            ]
            for implementation, source, base_args, parser_fn in impls:
                leaf = comparisons_root / "ginzburg_landau" / "raw" / case_id / implementation / f"run{repeat_index:02d}"
                add_repeat_task(
                    task_id=f"gl_{case_id}_{implementation}_run{repeat_index:02d}",
                    family="ginzburg_landau",
                    implementation=implementation,
                    source=source,
                    leaf_dir=leaf,
                    command=(
                        _python_cmd(source, *base_args, _rel(leaf / "output.json"))
                        if nprocs == 1
                        else _mpi_python_cmd(nprocs, source, *base_args, _rel(leaf / "output.json"))
                    ),
                    summarize=parser_fn,
                    context={"family": "ginzburg_landau", "implementation": implementation, "level": 5, "nprocs": nprocs, "case_id": case_id},
                )

    for nprocs in (1, 2, 4):
        case_id = f"l1_steps24_np{nprocs}"
        for repeat_index in range(1, 4):
            impls = [
                (
                    "fenics_custom",
                    "src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py",
                    [
                        "--level", "1",
                        "--steps", "24",
                        "--total-steps", "24",
                        "--quiet",
                        "--out",
                    ],
                    _parse_load_step_payload,
                ),
                (
                    "fenics_snes",
                    "src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py",
                    [
                        "--level", "1",
                        "--steps", "24",
                        "--total_steps", "24",
                        "--ksp_type", "gmres",
                        "--pc_type", "hypre",
                        "--ksp_rtol", "1e-1",
                        "--ksp_max_it", "500",
                        "--snes_atol", "1e-3",
                        "--quiet",
                        "--out",
                    ],
                    _parse_load_step_payload,
                ),
                (
                    "jax_petsc_element",
                    "src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py",
                    [
                        "--level", "1",
                        "--steps", "24",
                        "--total_steps", "24",
                        "--profile", "performance",
                        "--ksp_type", "stcg",
                        "--pc_type", "gamg",
                        "--ksp_rtol", "1e-1",
                        "--ksp_max_it", "30",
                        "--gamg_threshold", "0.05",
                        "--gamg_agg_nsmooths", "1",
                        "--gamg_set_coordinates",
                        "--use_near_nullspace",
                        "--assembly_mode", "element",
                        "--element_reorder_mode", "block_xyz",
                        "--local_hessian_mode", "element",
                        "--local_coloring",
                        "--use_trust_region",
                        "--trust_subproblem_line_search",
                        "--linesearch_tol", "1e-1",
                        "--trust_radius_init", "0.5",
                        "--trust_shrink", "0.5",
                        "--trust_expand", "1.5",
                        "--trust_eta_shrink", "0.05",
                        "--trust_eta_expand", "0.75",
                        "--nproc", "1",
                        "--quiet",
                        "--out",
                    ],
                    _parse_load_step_payload,
                ),
            ]
            if nprocs == 1:
                impls.insert(
                    2,
                    (
                        "jax_serial",
                        "src/problems/hyperelasticity/jax/solve_HE_jax_newton.py",
                        [
                            "--level", "1",
                            "--steps", "24",
                            "--total_steps", "24",
                            "--quiet",
                            "--out",
                        ],
                        _parse_pure_jax_he_payload,
                    ),
                )
            for implementation, source, base_args, parser_fn in impls:
                leaf = comparisons_root / "hyperelasticity" / "raw" / case_id / implementation / f"run{repeat_index:02d}"
                command = (
                    _python_cmd(source, *base_args, _rel(leaf / "output.json"))
                    if nprocs == 1
                    else _mpi_python_cmd(nprocs, source, *base_args, _rel(leaf / "output.json"))
                )
                add_repeat_task(
                    task_id=f"he_{case_id}_{implementation}_run{repeat_index:02d}",
                    family="hyperelasticity",
                    implementation=implementation,
                    source=source,
                    leaf_dir=leaf,
                    command=command,
                    summarize=parser_fn,
                    context={
                        "family": "hyperelasticity",
                        "implementation": implementation,
                        "level": 1,
                        "nprocs": nprocs,
                        "total_steps": 24,
                        "case_id": case_id,
                    },
                )

    for nprocs in (1, 2, 4):
        case_id = f"nx192_ny96_np{nprocs}"
        for repeat_index in range(1, 4):
            if nprocs == 1:
                leaf = comparisons_root / "topology" / "raw" / case_id / "jax_serial" / f"run{repeat_index:02d}"
                add_repeat_task(
                    task_id=f"topology_{case_id}_jax_serial_run{repeat_index:02d}",
                    family="topology",
                    implementation="jax_serial",
                    source="src/problems/topology/jax/solve_topopt_jax.py",
                    leaf_dir=leaf,
                    command=_python_cmd(
                        "src/problems/topology/jax/solve_topopt_jax.py",
                        "--nx", "192",
                        "--ny", "96",
                        "--length", "2.0",
                        "--height", "1.0",
                        "--traction", "1.0",
                        "--load_fraction", "0.2",
                        "--fixed_pad_cells", "16",
                        "--load_pad_cells", "16",
                        "--volume_fraction_target", "0.4",
                        "--theta_min", "0.001",
                        "--solid_latent", "10.0",
                        "--young", "1.0",
                        "--poisson", "0.3",
                        "--alpha_reg", "0.005",
                        "--ell_pf", "0.08",
                        "--mu_move", "0.01",
                        "--beta_lambda", "12.0",
                        "--volume_penalty", "10.0",
                        "--p_start", "1.0",
                        "--p_max", "4.0",
                        "--p_increment", "0.5",
                        "--continuation_interval", "20",
                        "--outer_maxit", "180",
                        "--outer_tol", "0.02",
                        "--volume_tol", "0.001",
                        "--mechanics_maxit", "200",
                        "--design_maxit", "400",
                        "--tolf", "1e-06",
                        "--tolg", "0.001",
                        "--ksp_rtol", "0.01",
                        "--ksp_max_it", "80",
                        "--save_outer_state_history",
                        "--quiet",
                        "--json_out", _rel(leaf / "output.json"),
                        "--state_out", _rel(leaf / "state.npz"),
                    ),
                    summarize=_parse_topology_payload,
                    context={"family": "topology", "implementation": "jax_serial", "nx": 192, "ny": 96, "nprocs": 1, "case_id": case_id},
                )

            leaf = comparisons_root / "topology" / "raw" / case_id / "jax_parallel" / f"run{repeat_index:02d}"
            add_repeat_task(
                task_id=f"topology_{case_id}_jax_parallel_run{repeat_index:02d}",
                family="topology",
                implementation="jax_parallel",
                source="src/problems/topology/jax/solve_topopt_parallel.py",
                leaf_dir=leaf,
                command=(
                    _python_cmd(
                        "src/problems/topology/jax/solve_topopt_parallel.py",
                        "--nx", "192",
                        "--ny", "96",
                        "--length", "2.0",
                        "--height", "1.0",
                        "--traction", "1.0",
                        "--load_fraction", "0.2",
                        "--fixed_pad_cells", "16",
                        "--load_pad_cells", "16",
                        "--volume_fraction_target", "0.4",
                        "--theta_min", "1e-3",
                        "--solid_latent", "10.0",
                        "--young", "1.0",
                        "--poisson", "0.3",
                        "--alpha_reg", "0.005",
                        "--ell_pf", "0.08",
                        "--mu_move", "0.01",
                        "--beta_lambda", "12.0",
                        "--volume_penalty", "10.0",
                        "--p_start", "1.0",
                        "--p_max", "4.0",
                        "--p_increment", "0.5",
                        "--continuation_interval", "20",
                        "--outer_maxit", "180",
                        "--outer_tol", "0.02",
                        "--volume_tol", "0.001",
                        "--stall_theta_tol", "1e-6",
                        "--stall_p_min", "4.0",
                        "--design_maxit", "20",
                        "--tolf", "1e-6",
                        "--tolg", "1e-3",
                        "--linesearch_tol", "0.1",
                        "--linesearch_relative_to_bound",
                        "--design_gd_line_search", "golden_adaptive",
                        "--design_gd_adaptive_window_scale", "2.0",
                        "--mechanics_ksp_type", "fgmres",
                        "--mechanics_pc_type", "gamg",
                        "--mechanics_ksp_rtol", "1e-4",
                        "--mechanics_ksp_max_it", "100",
                        "--quiet",
                        "--save_outer_state_history",
                        "--json_out", _rel(leaf / "output.json"),
                        "--state_out", _rel(leaf / "state.npz"),
                    )
                    if nprocs == 1
                    else _mpi_python_cmd(
                        nprocs,
                        "src/problems/topology/jax/solve_topopt_parallel.py",
                        "--nx", "192",
                        "--ny", "96",
                        "--length", "2.0",
                        "--height", "1.0",
                        "--traction", "1.0",
                        "--load_fraction", "0.2",
                        "--fixed_pad_cells", "16",
                        "--load_pad_cells", "16",
                        "--volume_fraction_target", "0.4",
                        "--theta_min", "1e-3",
                        "--solid_latent", "10.0",
                        "--young", "1.0",
                        "--poisson", "0.3",
                        "--alpha_reg", "0.005",
                        "--ell_pf", "0.08",
                        "--mu_move", "0.01",
                        "--beta_lambda", "12.0",
                        "--volume_penalty", "10.0",
                        "--p_start", "1.0",
                        "--p_max", "4.0",
                        "--p_increment", "0.5",
                        "--continuation_interval", "20",
                        "--outer_maxit", "180",
                        "--outer_tol", "0.02",
                        "--volume_tol", "0.001",
                        "--stall_theta_tol", "1e-6",
                        "--stall_p_min", "4.0",
                        "--design_maxit", "20",
                        "--tolf", "1e-6",
                        "--tolg", "1e-3",
                        "--linesearch_tol", "0.1",
                        "--linesearch_relative_to_bound",
                        "--design_gd_line_search", "golden_adaptive",
                        "--design_gd_adaptive_window_scale", "2.0",
                        "--mechanics_ksp_type", "fgmres",
                        "--mechanics_pc_type", "gamg",
                        "--mechanics_ksp_rtol", "1e-4",
                        "--mechanics_ksp_max_it", "100",
                        "--quiet",
                        "--save_outer_state_history",
                        "--json_out", _rel(leaf / "output.json"),
                        "--state_out", _rel(leaf / "state.npz"),
                    )
                ),
                summarize=_parse_topology_payload,
                context={"family": "topology", "implementation": "jax_parallel", "nx": 192, "ny": 96, "nprocs": nprocs, "case_id": case_id},
            )

    return tasks


def build_task_specs(out_dir: Path) -> list[TaskSpec]:
    return [
        *_build_example_tasks(out_dir),
        *_build_suite_tasks(out_dir),
        *_build_speed_tasks(out_dir),
    ]


def _record_environment(out_dir: Path) -> None:
    env_dir = out_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    write_json(
        env_dir / "git.json",
        {"branch": branch, "commit": commit, "captured_at": now_iso()},
    )

    versions_script = (
        "import json, socket, platform, sys\n"
        "payload = {\n"
        "  'python': sys.version.split()[0],\n"
        "  'platform': platform.platform(),\n"
        "  'hostname': socket.gethostname(),\n"
        "}\n"
        "mods = {}\n"
        "for name in ('jax', 'dolfinx', 'petsc4py', 'mpi4py', 'numpy', 'h5py', 'pyamg'):\n"
        "  try:\n"
        "    mod = __import__(name)\n"
        "    mods[name] = getattr(mod, '__version__', 'unknown')\n"
        "  except Exception as exc:\n"
        "    mods[name] = f'unavailable: {exc.__class__.__name__}'\n"
        "payload['modules'] = mods\n"
        "print(json.dumps(payload, indent=2))\n"
    )
    versions = subprocess.run(
        [str(PYTHON), "-c", versions_script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (env_dir / "versions.json").write_text(versions, encoding="utf-8")

    mpi_version = subprocess.run(
        [MPIEXEC, "--version"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    (env_dir / "mpi.txt").write_text((mpi_version.stdout or mpi_version.stderr or "") + "\n", encoding="utf-8")


def _load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    payload = read_json(manifest_path)
    return {entry["id"]: entry for entry in payload.get("entries", [])}


def _write_manifest(manifest_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    ordered = [entries[key] for key in sorted(entries)]
    write_json(manifest_path, {"entries": ordered})


def _write_issue_note(out_dir: Path, task: TaskSpec, error: Exception) -> None:
    issues_dir = out_dir / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    path = issues_dir / f"{task.id}.md"
    path.write_text(
        "\n".join(
            [
                f"# {task.id}",
                "",
                f"- family: `{task.family}`",
                f"- category: `{task.category}`",
                f"- source: `{task.source}`",
                f"- command leaf: `{_rel(task.leaf_dir)}`",
                f"- error: `{error.__class__.__name__}: {error}`",
                "",
                "This note was written automatically when the replication runner stopped on a task failure.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_manifest_entry(task: TaskSpec, status: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task.id,
        "family": task.family,
        "category": task.category,
        "kind": task.kind,
        "command": status["command"],
        "cwd": _rel(Path(status["cwd"])),
        "outputs": [_rel(Path(path)) for path in task.outputs],
        "status": "completed" if status["success"] else "failed",
        "source": task.source,
        "started_at": status["started_at"],
        "finished_at": status["finished_at"],
        "duration_s": float(status["duration_s"]),
        "notes": task.notes,
        "leaf_dir": _rel(task.leaf_dir),
        "summary": summary,
    }


def _run_task(task: TaskSpec, *, out_dir: Path, resume: bool, manifest_entries: dict[str, dict[str, Any]]) -> None:
    result = run_logged_command(
        command=task.command,
        cwd=REPO_ROOT,
        leaf_dir=task.leaf_dir,
        expected_outputs=task.outputs,
        env=task.env,
        resume=resume,
        notes=task.notes,
    )
    summary = task.summarize(task.outputs[0], task.context)
    status_path = task.leaf_dir / "status.json"
    status = read_json(status_path)
    status["summary"] = summary
    write_json(status_path, status)
    issue_path = out_dir / "issues" / f"{task.id}.md"
    if issue_path.exists():
        issue_path.unlink()
    manifest_entries[task.id] = _build_manifest_entry(task, result.to_dict(), summary)


def _run_report_generation(out_dir: Path, *, resume: bool, manifest_entries: dict[str, dict[str, Any]]) -> None:
    leaf_dir = out_dir / "_tasks" / "generate_reports"
    command = _python_cmd(
        "experiments/analysis/generate_replication_reports.py",
        "--out-dir", _rel(out_dir),
    )
    outputs = [
        out_dir / "index.md",
        out_dir / "commands.md",
        out_dir / "model_cards" / "plaplace.md",
        out_dir / "model_cards" / "ginzburg_landau.md",
        out_dir / "model_cards" / "hyperelasticity.md",
        out_dir / "model_cards" / "topology.md",
    ]
    result = run_logged_command(
        command=command,
        cwd=REPO_ROOT,
        leaf_dir=leaf_dir,
        expected_outputs=outputs,
        env=dict(THREAD_ENV),
        resume=resume,
        notes="Generate top-level commands, index, model cards, and comparison summaries",
    )
    summary = {"generated_files": [_rel(path) for path in outputs]}
    status_path = leaf_dir / "status.json"
    status = read_json(status_path)
    status["summary"] = summary
    write_json(status_path, status)
    manifest_entries["generate_reports"] = {
        "id": "generate_reports",
        "family": "campaign",
        "category": "reports",
        "kind": "report_generation",
        "command": result.command,
        "cwd": _rel(REPO_ROOT),
        "outputs": [_rel(path) for path in outputs],
        "status": "completed" if result.success else "failed",
        "source": "experiments/analysis/generate_replication_reports.py",
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_s": result.duration_s,
        "notes": result.notes,
        "leaf_dir": _rel(leaf_dir),
        "summary": summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", nargs="+", choices=ONLY_CHOICES, default=list(ONLY_CHOICES))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model_cards", "runs", "comparisons", "issues", "env", "_tasks"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    _record_environment(out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest_entries = _load_manifest(manifest_path) if args.resume else {}

    selected = set(args.only)
    tasks = build_task_specs(out_dir)
    for task in tasks:
        if task.category not in selected:
            continue
        try:
            _run_task(task, out_dir=out_dir, resume=args.resume, manifest_entries=manifest_entries)
            _write_manifest(manifest_path, manifest_entries)
        except Exception as exc:
            _write_issue_note(out_dir, task, exc)
            _write_manifest(manifest_path, manifest_entries)
            raise

    if selected & {"reports", "model-cards"} or not (selected - {"reports", "model-cards"}):
        _run_report_generation(out_dir, resume=args.resume, manifest_entries=manifest_entries)
        _write_manifest(manifest_path, manifest_entries)


if __name__ == "__main__":
    main()
