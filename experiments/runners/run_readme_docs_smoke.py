#!/usr/bin/env python3
"""Run the explicit README/docs smoke commands used by the maintained docs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

TOPOLOGY_THREAD_ENV = {
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _normalize_command(argv: list[str]) -> str:
    parts = []
    for part in argv:
        text = str(part)
        if text == str(PYTHON):
            text = "./.venv/bin/python"
        elif text.startswith(str(REPO_ROOT) + "/"):
            text = text[len(str(REPO_ROOT)) + 1 :]
        parts.append(text)
    return shlex.join(parts)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _step_linear_iters(step: dict[str, Any]) -> int:
    value = step.get("linear_iters")
    if value is not None:
        return int(value)
    total = 0
    for rec in step.get("history", []):
        if "ksp_its" in rec:
            total += int(rec["ksp_its"])
    return total


def _sum_step_field(steps: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for step in steps:
        value = step.get(key)
        if value is not None:
            total += float(value)
    return total


def _parse_scalar_results(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload["results"][0])
    return {
        "result": "completed",
        "solver_wall_time_s": float(result.get("solve_time", result.get("time", result.get("total_time", 0.0)))),
        "final_energy": float(result["energy"]),
        "newton_iters": int(result.get("iters", 0)),
        "linear_iters": int(result.get("total_ksp_its", 0)),
    }


def _parse_gl_step_payload(payload: dict[str, Any]) -> dict[str, Any]:
    steps = list(payload.get("steps", []))
    final_step = steps[-1] if steps else {}
    return {
        "result": "completed",
        "completed_steps": len(steps),
        "solver_wall_time_s": float(payload.get("solve_time_total", payload.get("time", payload.get("total_time", 0.0)))),
        "final_energy": float(final_step.get("energy", math.nan)),
        "newton_iters": int(sum(int(step.get("nit", step.get("iters", 0))) for step in steps)),
        "linear_iters": int(sum(_step_linear_iters(step) for step in steps)),
    }


def _parse_he_payload(payload: dict[str, Any]) -> dict[str, Any]:
    he_payload = dict(payload.get("result", payload)) if isinstance(payload.get("result"), dict) else payload
    steps = list(he_payload.get("steps", []))
    final_step = steps[-1] if steps else {}
    return {
        "result": str(he_payload.get("result", "completed")),
        "completed_steps": len(steps),
        "solver_wall_time_s": float(he_payload.get("solve_time_total", he_payload.get("time", he_payload.get("total_time", 0.0)))),
        "final_energy": float(final_step.get("energy", math.nan)),
        "newton_iters": int(
            he_payload.get("total_newton_iters", sum(int(step.get("nit", step.get("iters", 0))) for step in steps))
        ),
        "linear_iters": int(
            he_payload.get("total_linear_iters", sum(_step_linear_iters(step) for step in steps))
        ),
    }


def _parse_he_snes_payload(payload: dict[str, Any]) -> dict[str, Any]:
    steps = list(payload.get("steps", []))
    converged_steps = 0
    first_failed_step = None
    first_failed_angle = None
    first_failed_reason = None
    for step in steps:
        reason = int(step.get("reason", 0))
        if reason > 0:
            converged_steps += 1
            continue
        if first_failed_step is None:
            first_failed_step = int(step.get("step", 0))
            first_failed_angle = float(step.get("angle", math.nan))
            first_failed_reason = reason
    return {
        "result": "failed" if first_failed_step is not None else "completed",
        "recorded_steps": len(steps),
        "converged_steps": converged_steps,
        "first_failed_step": first_failed_step,
        "first_failed_angle": first_failed_angle,
        "failure_reason": first_failed_reason,
        "solver_wall_time_s": _sum_step_field(steps, "time"),
        "final_energy": float(steps[-1].get("energy", math.nan)) if steps else math.nan,
        "newton_iters": int(sum(int(step.get("iters", 0)) for step in steps)),
        "linear_iters": int(sum(int(step.get("linear_iters", 0)) for step in steps)),
    }


def _parse_topology_payload(payload: dict[str, Any]) -> dict[str, Any]:
    final = dict(payload.get("final_metrics", {}))
    return {
        "result": str(payload.get("result", "unknown")),
        "solver_wall_time_s": float(payload.get("time", 0.0)),
        "setup_time_s": float(payload.get("setup_time", 0.0)),
        "outer_iterations": int(final.get("outer_iterations", 0)),
        "final_compliance": float(final.get("final_compliance", math.nan)),
        "final_volume_fraction": float(final.get("final_volume_fraction", math.nan)),
        "final_p_penal": float(final.get("final_p_penal", math.nan)),
    }


def _case_output_exists(case: dict[str, Any]) -> bool:
    output_json = case.get("output_json")
    if output_json is not None and not Path(output_json).exists():
        return False
    state_path = case.get("state_path")
    if state_path is not None and not Path(state_path).exists():
        return False
    return True


def _smoke_cases(out_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add_case(
        *,
        case_id: str,
        source_files: list[str],
        argv: list[str],
        output_json: Path,
        parser: str,
        note: str = "",
        state_path: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float = 14400.0,
    ) -> None:
        cases.append(
            {
                "id": case_id,
                "source_files": source_files,
                "argv": argv,
                "command": _normalize_command(argv),
                "output_json": output_json,
                "state_path": state_path,
                "parser": parser,
                "note": note,
                "env": env or {},
                "timeout_s": timeout_s,
            }
        )

    def case_dir(name: str) -> Path:
        path = out_dir / "cases" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # pLaplace
    path = case_dir("plaplace_fenics_custom_l5")
    add_case(
        case_id="plaplace_fenics_custom_l5",
        source_files=["docs/problems/pLaplace.md", "docs/setup/quickstart.md", "docs/results/pLaplace.md"],
        argv=[
            str(PYTHON),
            "-u",
            "src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py",
            "--levels",
            "5",
            "--quiet",
            "--json",
            str(path / "output.json"),
            "--state-out",
            str(path / "state.npz"),
        ],
        output_json=path / "output.json",
        state_path=path / "state.npz",
        parser="scalar",
        note="Showcase sample state plus shared level-5 parity case.",
    )
    path = case_dir("plaplace_fenics_snes_l5")
    add_case(
        case_id="plaplace_fenics_snes_l5",
        source_files=["docs/setup/quickstart.md", "docs/results/pLaplace.md"],
        argv=[
            str(PYTHON),
            "-u",
            "src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py",
            "--levels",
            "5",
            "--json",
            str(path / "output.json"),
        ],
        output_json=path / "output.json",
        parser="scalar",
    )
    path = case_dir("plaplace_jax_serial_l5")
    add_case(
        case_id="plaplace_jax_serial_l5",
        source_files=["docs/setup/quickstart.md", "docs/results/pLaplace.md"],
        argv=[
            str(PYTHON),
            "-u",
            "src/problems/plaplace/jax/solve_pLaplace_jax_newton.py",
            "--levels",
            "5",
            "--quiet",
            "--json",
            str(path / "output.json"),
        ],
        output_json=path / "output.json",
        parser="scalar",
    )
    path = case_dir("plaplace_jax_petsc_element_l5")
    add_case(
        case_id="plaplace_jax_petsc_element_l5",
        source_files=["docs/setup/quickstart.md", "docs/results/pLaplace.md"],
        argv=[
            str(PYTHON),
            "-u",
            "src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py",
            "--level",
            "5",
            "--profile",
            "reference",
            "--assembly-mode",
            "element",
            "--local-hessian-mode",
            "element",
            "--element-reorder-mode",
            "block_xyz",
            "--local-coloring",
            "--nproc",
            "1",
            "--quiet",
            "--json",
            str(path / "output.json"),
        ],
        output_json=path / "output.json",
        parser="scalar",
    )

    # Ginzburg-Landau
    path = case_dir("gl_fenics_custom_l5")
    add_case(
        case_id="gl_fenics_custom_l5",
        source_files=["docs/problems/GinzburgLandau.md", "docs/setup/quickstart.md", "docs/results/GinzburgLandau.md"],
        argv=[
            str(PYTHON),
            "-u",
            "src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py",
            "--levels",
            "5",
            "--quiet",
            "--json",
            str(path / "output.json"),
            "--state-out",
            str(path / "state.npz"),
        ],
        output_json=path / "output.json",
        state_path=path / "state.npz",
        parser="scalar",
        note="Showcase sample state plus shared level-5 parity case.",
    )
    path = case_dir("gl_fenics_snes_l5")
    add_case(
        case_id="gl_fenics_snes_l5",
        source_files=["docs/setup/quickstart.md", "docs/results/GinzburgLandau.md"],
        argv=[
            str(PYTHON),
            "-u",
            "src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py",
            "--levels",
            "5",
            "--json",
            str(path / "output.json"),
        ],
        output_json=path / "output.json",
        parser="scalar",
    )
    path = case_dir("gl_jax_petsc_element_l5")
    add_case(
        case_id="gl_jax_petsc_element_l5",
        source_files=["docs/setup/quickstart.md", "docs/results/GinzburgLandau.md"],
        argv=[
            str(PYTHON),
            "-u",
            "src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py",
            "--level",
            "5",
            "--profile",
            "reference",
            "--assembly_mode",
            "element",
            "--local_hessian_mode",
            "element",
            "--element_reorder_mode",
            "block_xyz",
            "--local_coloring",
            "--nproc",
            "1",
            "--quiet",
            "--out",
            str(path / "output.json"),
        ],
        output_json=path / "output.json",
        parser="gl_steps",
    )

    # HyperElasticity
    path = case_dir("he_fenics_custom_l1_s24")
    add_case(
        case_id="he_fenics_custom_l1_s24",
        source_files=["docs/problems/HyperElasticity.md", "docs/setup/quickstart.md", "docs/results/HyperElasticity.md"],
        argv=[
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            "experiments/runners/run_trust_region_case.py",
            "--problem",
            "he",
            "--backend",
            "fenics",
            "--level",
            "1",
            "--steps",
            "24",
            "--start-step",
            "1",
            "--total_steps",
            "24",
            "--profile",
            "performance",
            "--ksp-type",
            "stcg",
            "--pc-type",
            "gamg",
            "--ksp-rtol",
            "1e-1",
            "--ksp-max-it",
            "30",
            "--gamg-threshold",
            "0.05",
            "--gamg-agg-nsmooths",
            "1",
            "--gamg-set-coordinates",
            "--use-near-nullspace",
            "--no-pc-setup-on-ksp-cap",
            "--tolf",
            "1e-4",
            "--tolg",
            "1e-3",
            "--tolg-rel",
            "1e-3",
            "--tolx-rel",
            "1e-3",
            "--tolx-abs",
            "1e-10",
            "--maxit",
            "100",
            "--linesearch-a",
            "-0.5",
            "--linesearch-b",
            "2.0",
            "--linesearch-tol",
            "1e-1",
            "--use-trust-region",
            "--trust-radius-init",
            "0.5",
            "--trust-radius-min",
            "1e-8",
            "--trust-radius-max",
            "1e6",
            "--trust-shrink",
            "0.5",
            "--trust-expand",
            "1.5",
            "--trust-eta-shrink",
            "0.05",
            "--trust-eta-expand",
            "0.75",
            "--trust-max-reject",
            "6",
            "--trust-subproblem-line-search",
            "--save-history",
            "--save-linear-timing",
            "--quiet",
            "--out",
            str(path / "output.json"),
        ],
        output_json=path / "output.json",
        parser="he",
    )
    path = case_dir("he_jax_serial_l1_s24")
    add_case(
        case_id="he_jax_serial_l1_s24",
        source_files=["docs/problems/HyperElasticity.md", "docs/setup/quickstart.md", "docs/results/HyperElasticity.md"],
        argv=[
            str(PYTHON),
            "-u",
            "src/problems/hyperelasticity/jax/solve_HE_jax_newton.py",
            "--level",
            "1",
            "--steps",
            "24",
            "--total_steps",
            "24",
            "--quiet",
            "--out",
            str(path / "output.json"),
            "--state-out",
            str(path / "state.npz"),
        ],
        output_json=path / "output.json",
        state_path=path / "state.npz",
        parser="he",
    )
    path = case_dir("he_jax_petsc_element_l1_s24")
    add_case(
        case_id="he_jax_petsc_element_l1_s24",
        source_files=["docs/setup/quickstart.md", "docs/results/HyperElasticity.md"],
        argv=[
            str(PYTHON),
            "-u",
            "src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py",
            "--level",
            "1",
            "--steps",
            "24",
            "--total_steps",
            "24",
            "--profile",
            "performance",
            "--ksp_type",
            "stcg",
            "--pc_type",
            "gamg",
            "--ksp_rtol",
            "1e-1",
            "--ksp_max_it",
            "30",
            "--gamg_threshold",
            "0.05",
            "--gamg_agg_nsmooths",
            "1",
            "--gamg_set_coordinates",
            "--use_near_nullspace",
            "--assembly_mode",
            "element",
            "--element_reorder_mode",
            "block_xyz",
            "--local_hessian_mode",
            "element",
            "--local_coloring",
            "--use_trust_region",
            "--trust_subproblem_line_search",
            "--linesearch_tol",
            "1e-1",
            "--trust_radius_init",
            "0.5",
            "--trust_shrink",
            "0.5",
            "--trust_expand",
            "1.5",
            "--trust_eta_shrink",
            "0.05",
            "--trust_eta_expand",
            "0.75",
            "--trust_max_reject",
            "6",
            "--nproc",
            "1",
            "--quiet",
            "--out",
            str(path / "output.json"),
        ],
        output_json=path / "output.json",
        parser="he",
    )
    path = case_dir("he_jax_petsc_element_l4_np32_showcase")
    add_case(
        case_id="he_jax_petsc_element_l4_np32_showcase",
        source_files=["docs/problems/HyperElasticity.md"],
        argv=[
            "mpiexec",
            "-n",
            "32",
            str(PYTHON),
            "-u",
            "src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py",
            "--level",
            "4",
            "--steps",
            "24",
            "--total_steps",
            "24",
            "--profile",
            "performance",
            "--ksp_type",
            "stcg",
            "--pc_type",
            "gamg",
            "--ksp_rtol",
            "1e-1",
            "--ksp_max_it",
            "30",
            "--gamg_threshold",
            "0.05",
            "--gamg_agg_nsmooths",
            "1",
            "--gamg_set_coordinates",
            "--use_near_nullspace",
            "--assembly_mode",
            "element",
            "--element_reorder_mode",
            "block_xyz",
            "--local_hessian_mode",
            "element",
            "--local_coloring",
            "--use_trust_region",
            "--trust_subproblem_line_search",
            "--linesearch_tol",
            "1e-1",
            "--trust_radius_init",
            "0.5",
            "--trust_radius_min",
            "1e-8",
            "--trust_radius_max",
            "1e6",
            "--trust_shrink",
            "0.5",
            "--trust_expand",
            "1.5",
            "--trust_eta_shrink",
            "0.05",
            "--trust_eta_expand",
            "0.75",
            "--trust_max_reject",
            "6",
            "--tolf",
            "1e-4",
            "--tolg",
            "1e-3",
            "--tolg_rel",
            "1e-3",
            "--tolx_rel",
            "1e-3",
            "--tolx_abs",
            "1e-10",
            "--maxit",
            "100",
            "--quiet",
            "--out",
            str(path / "output.json"),
            "--state-out",
            str(path / "state.npz"),
        ],
        output_json=path / "output.json",
        state_path=path / "state.npz",
        parser="he",
        timeout_s=21600.0,
    )

    # Topology explicit docs commands.
    path = case_dir("topology_serial_reference")
    add_case(
        case_id="topology_serial_reference",
        source_files=["docs/setup/quickstart.md", "docs/problems/Topology.md", "docs/results/Topology.md"],
        argv=[
            str(PYTHON),
            "-u",
            "src/problems/topology/jax/solve_topopt_jax.py",
            "--nx",
            "192",
            "--ny",
            "96",
            "--length",
            "2.0",
            "--height",
            "1.0",
            "--traction",
            "1.0",
            "--load_fraction",
            "0.2",
            "--fixed_pad_cells",
            "16",
            "--load_pad_cells",
            "16",
            "--volume_fraction_target",
            "0.4",
            "--theta_min",
            "0.001",
            "--solid_latent",
            "10.0",
            "--young",
            "1.0",
            "--poisson",
            "0.3",
            "--alpha_reg",
            "0.005",
            "--ell_pf",
            "0.08",
            "--mu_move",
            "0.01",
            "--beta_lambda",
            "12.0",
            "--volume_penalty",
            "10.0",
            "--p_start",
            "1.0",
            "--p_max",
            "4.0",
            "--p_increment",
            "0.5",
            "--continuation_interval",
            "20",
            "--outer_maxit",
            "180",
            "--outer_tol",
            "0.02",
            "--volume_tol",
            "0.001",
            "--mechanics_maxit",
            "200",
            "--design_maxit",
            "400",
            "--tolf",
            "1e-6",
            "--tolg",
            "1e-3",
            "--ksp_rtol",
            "1e-2",
            "--ksp_max_it",
            "80",
            "--quiet",
            "--json_out",
            str(path / "output.json"),
            "--state_out",
            str(path / "state.npz"),
        ],
        output_json=path / "output.json",
        state_path=path / "state.npz",
        parser="topology",
        env=TOPOLOGY_THREAD_ENV,
    )
    path = case_dir("topology_parallel_final_np32")
    add_case(
        case_id="topology_parallel_final_np32",
        source_files=["docs/setup/quickstart.md", "docs/problems/Topology.md", "docs/results/Topology.md"],
        argv=[
            "mpiexec",
            "-n",
            "32",
            str(PYTHON),
            "-u",
            "src/problems/topology/jax/solve_topopt_parallel.py",
            "--nx",
            "768",
            "--ny",
            "384",
            "--length",
            "2.0",
            "--height",
            "1.0",
            "--traction",
            "1.0",
            "--load_fraction",
            "0.2",
            "--fixed_pad_cells",
            "32",
            "--load_pad_cells",
            "32",
            "--volume_fraction_target",
            "0.4",
            "--theta_min",
            "1e-6",
            "--solid_latent",
            "10.0",
            "--young",
            "1.0",
            "--poisson",
            "0.3",
            "--alpha_reg",
            "0.005",
            "--ell_pf",
            "0.08",
            "--mu_move",
            "0.01",
            "--beta_lambda",
            "12.0",
            "--volume_penalty",
            "10.0",
            "--p_start",
            "1.0",
            "--p_max",
            "10.0",
            "--p_increment",
            "0.2",
            "--continuation_interval",
            "1",
            "--outer_maxit",
            "2000",
            "--outer_tol",
            "0.02",
            "--volume_tol",
            "0.001",
            "--stall_theta_tol",
            "1e-6",
            "--stall_p_min",
            "4.0",
            "--design_maxit",
            "20",
            "--tolf",
            "1e-6",
            "--tolg",
            "1e-3",
            "--linesearch_tol",
            "0.1",
            "--linesearch_relative_to_bound",
            "--design_gd_line_search",
            "golden_adaptive",
            "--design_gd_adaptive_window_scale",
            "2.0",
            "--mechanics_ksp_type",
            "fgmres",
            "--mechanics_pc_type",
            "gamg",
            "--mechanics_ksp_rtol",
            "1e-4",
            "--mechanics_ksp_max_it",
            "100",
            "--quiet",
            "--print_outer_iterations",
            "--save_outer_state_history",
            "--outer_snapshot_stride",
            "2",
            "--outer_snapshot_dir",
            str(path / "frames"),
            "--json_out",
            str(path / "output.json"),
            "--state_out",
            str(path / "state.npz"),
        ],
        output_json=path / "output.json",
        state_path=path / "state.npz",
        parser="topology",
        env=TOPOLOGY_THREAD_ENV,
        timeout_s=21600.0,
    )

    # README SNES continuation rows.
    for steps in (96, 192, 384):
        path = case_dir(f"he_snes_l3_np16_steps{steps}")
        add_case(
            case_id=f"he_snes_l3_np16_steps{steps}",
            source_files=["README.md"],
            argv=[
                "mpiexec",
                "-n",
                "16",
                str(PYTHON),
                "-u",
                "src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py",
                "--level",
                "3",
                "--steps",
                str(steps),
                "--total_steps",
                str(steps),
                "--stop_on_fail",
                "--quiet",
                "--snes_type",
                "newtonls",
                "--linesearch",
                "basic",
                "--pc_type",
                "gamg",
                "--ksp_type",
                "fgmres",
                "--ksp_rtol",
                "1e-1",
                "--ksp_max_it",
                "2000",
                "--snes_atol",
                "1e-3",
                "--out",
                str(path / "output.json"),
            ],
            output_json=path / "output.json",
            parser="he_snes",
            env={
                "PETSC_OPTIONS": "-he_pc_gamg_threshold 0.05 -he_pc_gamg_agg_nsmooths 1",
            },
            timeout_s=21600.0,
        )

    return cases


def _parse_payload(parser_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if parser_name == "scalar":
        return _parse_scalar_results(payload)
    if parser_name == "gl_steps":
        return _parse_gl_step_payload(payload)
    if parser_name == "he":
        return _parse_he_payload(payload)
    if parser_name == "he_snes":
        return _parse_he_snes_payload(payload)
    if parser_name == "topology":
        return _parse_topology_payload(payload)
    raise KeyError(parser_name)


def _run_case(case: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    case_id = str(case["id"])
    stdout_path = log_dir / f"{case_id}.stdout.txt"
    stderr_path = log_dir / f"{case_id}.stderr.txt"
    env = os.environ.copy()
    env.update(case.get("env", {}))

    t0 = time.perf_counter()
    proc = subprocess.run(
        case["argv"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=float(case.get("timeout_s", 14400.0)),
    )
    elapsed = time.perf_counter() - t0
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    row: dict[str, Any] = {
        "id": case_id,
        "source_files": case["source_files"],
        "command": case["command"],
        "note": case.get("note", ""),
        "output_json": _repo_rel(Path(case["output_json"])),
        "state_path": _repo_rel(Path(case["state_path"])) if case.get("state_path") is not None else None,
        "stdout_path": _repo_rel(stdout_path),
        "stderr_path": _repo_rel(stderr_path),
        "exit_code": int(proc.returncode),
        "external_elapsed_s": elapsed,
    }
    if proc.returncode != 0:
        row["result"] = "command_failed"
        return row

    payload = _read_json(Path(case["output_json"]))
    row.update(_parse_payload(str(case["parser"]), payload))
    return row


def _write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# README/Docs Smoke Summary",
        "",
        f"Campaign root: `{_repo_rel(path.parent)}`",
        "",
        "| Case | Result | Solver wall [s] | External [s] | Energy | Compliance | Volume | Newton | Linear | JSON |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {result} | {solver} | {external} | {energy} | {compliance} | {volume} | {newton} | {linear} | `{json_path}` |".format(
                case=row["id"],
                result=row.get("result", "-"),
                solver="{:.6f}".format(float(row["solver_wall_time_s"])) if row.get("solver_wall_time_s") is not None else "-",
                external="{:.6f}".format(float(row["external_elapsed_s"])),
                energy="{:.9f}".format(float(row["final_energy"])) if row.get("final_energy") is not None and math.isfinite(float(row["final_energy"])) else "-",
                compliance="{:.9f}".format(float(row["final_compliance"])) if row.get("final_compliance") is not None and math.isfinite(float(row["final_compliance"])) else "-",
                volume="{:.9f}".format(float(row["final_volume_fraction"])) if row.get("final_volume_fraction") is not None and math.isfinite(float(row["final_volume_fraction"])) else "-",
                newton=row.get("newton_iters", row.get("outer_iterations", "-")),
                linear=row.get("linear_iters", "-"),
                json_path=row.get("output_json", "-"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reproduction" / "local_readme_docs_smoke" / "runs" / "readme_docs_smoke",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"

    rows_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and summary_path.exists():
        payload = _read_json(summary_path)
        rows_by_id = {str(row["id"]): dict(row) for row in payload.get("rows", [])}

    rows: list[dict[str, Any]] = []
    for case in _smoke_cases(out_dir):
        existing = rows_by_id.get(str(case["id"]))
        if args.resume and existing and int(existing.get("exit_code", 1)) == 0 and _case_output_exists(case):
            rows.append(existing)
            continue
        print(f"[smoke] running {case['id']}", flush=True)
        row = _run_case(case, log_dir)
        rows_by_id[str(case["id"])] = row
        rows.append(row)
        summary_path.write_text(json.dumps({"rows": list(rows_by_id.values())}, indent=2) + "\n", encoding="utf-8")
        _write_summary_markdown(summary_md, sorted(rows_by_id.values(), key=lambda item: item["id"]))

    rows = sorted(rows_by_id.values(), key=lambda item: item["id"])
    summary_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    _write_summary_markdown(summary_md, rows)
    print(json.dumps({"rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
