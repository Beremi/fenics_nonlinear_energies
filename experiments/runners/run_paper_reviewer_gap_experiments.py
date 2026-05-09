#!/usr/bin/env python3
"""Run and summarize local reviewer-gap experiments for the paper."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TRUST_RUNNER = REPO_ROOT / "experiments/runners/run_trust_region_case.py"
P3D_RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_backend_mix_case.py"
TOPO_RUNNER = REPO_ROOT / "src/problems/topology/jax/solve_topopt_parallel.py"
SCALAR_L10_GENERATOR = REPO_ROOT / "experiments/runners/generate_scalar_uniform_l10_meshes.py"
RAW_ROOT = REPO_ROOT / "artifacts/raw_results/paper_reviewer_gap_experiments"
REPORT_ROOT = REPO_ROOT / "artifacts/reports/paper_reviewer_gap_experiments"

SECTIONS = (
    "he_distribution",
    "he_pmg",
    "topology_consistency",
    "gl_globalization",
    "p3d_derivative_degree",
)

THREAD_ENV = {
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1",
    "MPLBACKEND": "Agg",
    "MIX_LOCAL_PMG_MUMPS_REDUNDANT_NUMBER": "1",
    "MIX_LOCAL_PMG_MUMPS_FACTOR_SOLVER": "mumps",
}

COMMON_HE_ARGS = (
    "--problem",
    "he",
    "--backend",
    "element",
    "--profile",
    "performance",
    "--ksp-type",
    "stcg",
    "--ksp-rtol",
    "1e-1",
    "--ksp-max-it",
    "30",
    "--tolf",
    "1e-4",
    "--tolg",
    "1e-3",
    "--tolg-rel",
    "1e-3",
    "--tolx-rel",
    "1e-4",
    "--tolx-abs",
    "1e-10",
    "--linesearch-a",
    "-0.5",
    "--linesearch-b",
    "2.0",
    "--linesearch-tol",
    "1e-1",
    "--line-search",
    "armijo",
    "--use-trust-region",
    "--trust-subproblem-line-search",
    "--trust-radius-init",
    "1.0",
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
    "--element-reorder-mode",
    "block_xyz",
    "--local-hessian-mode",
    "element",
    "--local-coloring",
    "--save-history",
    "--save-linear-timing",
    "--quiet",
)

TOPO_BASE_ARGS = (
    "--length",
    "2.0",
    "--height",
    "1.0",
    "--traction",
    "1.0",
    "--load_fraction",
    "0.2",
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
    "0.1",
    "--continuation_interval",
    "1",
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
    "--fixed_outer_schedule",
    "--save_outer_state_history",
)


@dataclass(frozen=True)
class CaseSpec:
    section: str
    key: str
    label: str
    kind: str
    nprocs: int
    wall_cap_s: float
    args: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


P3D_ROUTE_BACKENDS = (
    ("element_ad", "local"),
    ("colored_sfd", "local_sfd"),
    ("constitutive_ad", "local_constitutiveAD"),
)

P3D_DISCRETIZATION_CASES = (
    {
        "key": "p1_l1",
        "label": "P1(L1)",
        "mesh_name": "hetero_ssr_L1",
        "degree": 1,
        "pmg_strategy": "uniform_refined_p1_chain",
        "free_dofs": 10526,
        "local_element_dofs": 12,
        "routes": ("element_ad", "colored_sfd", "constitutive_ad"),
    },
    {
        "key": "p1_l1_2",
        "label": "P1(L2)",
        "mesh_name": "hetero_ssr_L1_2",
        "degree": 1,
        "pmg_strategy": "uniform_refined_p1_chain",
        "free_dofs": 79024,
        "local_element_dofs": 12,
        "routes": ("element_ad", "colored_sfd", "constitutive_ad"),
    },
    {
        "key": "p2_l1",
        "label": "P2(L1)",
        "mesh_name": "hetero_ssr_L1",
        "degree": 2,
        "pmg_strategy": "same_mesh_p2_p1",
        "free_dofs": 79024,
        "local_element_dofs": 30,
        "routes": ("element_ad", "colored_sfd", "constitutive_ad"),
    },
    {
        "key": "p4_l1",
        "label": "P4(L1)",
        "mesh_name": "hetero_ssr_L1",
        "degree": 4,
        "pmg_strategy": "same_mesh_p4_p2_p1",
        "free_dofs": 610964,
        "local_element_dofs": 105,
        "routes": ("element_ad", "constitutive_ad"),
    },
)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _normalize_command(argv: list[str]) -> str:
    parts = []
    for item in argv:
        text = str(item)
        if text == str(PYTHON):
            text = "./.venv/bin/python"
        elif text.startswith(str(REPO_ROOT) + "/"):
            text = text[len(str(REPO_ROOT)) + 1 :]
        parts.append(text)
    return shlex.join(parts)


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(THREAD_ENV)
    return env


def _run_subprocess(argv: list[str], *, timeout_s: float, log_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        cwd=REPO_ROOT,
        env=_base_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, _ = proc.communicate(timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, _ = proc.communicate()
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout or "", encoding="utf-8")
    return {
        "returncode": int(proc.returncode if proc.returncode is not None else -signal.SIGTERM),
        "timed_out": bool(timed_out),
        "wall_time_s": float(elapsed),
    }


def _he_build_args(*, level: int, steps: int, maxit: int, pc_type: str, build_mode: str) -> tuple[str, ...]:
    mesh_source = "procedural" if build_mode == "rank_local" else "hdf5"
    distribution = "overlap_p2p" if build_mode == "rank_local" else "overlap_allgather"
    assembly_backend = "coo_local" if build_mode == "rank_local" else "coo"
    return (
        *COMMON_HE_ARGS,
        "--level",
        str(level),
        "--steps",
        str(steps),
        "--start-step",
        "1",
        "--total-steps",
        "24",
        "--maxit",
        str(maxit),
        "--pc-type",
        pc_type,
        "--problem-build-mode",
        build_mode,
        "--he-mesh-source",
        mesh_source,
        "--he-element-degree",
        "1",
        "--distribution-strategy",
        distribution,
        "--assembly-backend",
        assembly_backend,
    )


def _he_pmg_args(*, level: int, maxit: int, candidate: str) -> tuple[str, ...]:
    if candidate == "gamg":
        return _he_build_args(level=level, steps=1, maxit=maxit, pc_type="gamg", build_mode="rank_local")
    settings = {
        "pmg_l2_hypre": ("2", "hypre", "0", ""),
        "pmg_l2_redundant_mumps": ("2", "redundant", "1", "mumps"),
        "pmg_l3_redundant_mumps": ("3", "redundant", "1", "mumps"),
    }[candidate]
    coarsest, coarse_pc, redundant, factor = settings
    if int(coarsest) >= int(level):
        coarsest = str(max(1, int(level) - 1))
    args = list(_he_build_args(level=level, steps=1, maxit=maxit, pc_type="mg", build_mode="rank_local"))
    args.extend(
        [
            "--he-pmg-coarsest-level",
            coarsest,
            "--he-pmg-smoother-ksp-type",
            "chebyshev",
            "--he-pmg-smoother-pc-type",
            "jacobi",
            "--he-pmg-smoother-steps",
            "2",
            "--he-pmg-coarse-ksp-type",
            "preonly",
            "--he-pmg-coarse-pc-type",
            coarse_pc,
            "--he-pmg-coarse-redundant-number",
            redundant,
        ]
    )
    if factor:
        args.extend(["--he-pmg-coarse-factor-solver-type", factor])
    return tuple(args)


def _gl_args(*, level: int, maxit: int, method: str) -> tuple[str, ...]:
    common = [
        "--problem",
        "gl",
        "--backend",
        "element",
        "--level",
        str(level),
        "--steps",
        "1",
        "--profile",
        "reference",
        "--pc-type",
        "hypre",
        "--ksp-rtol",
        "1e-3",
        "--ksp-max-it",
        "200",
        "--tolf",
        "1e-6",
        "--tolg",
        "1e-5",
        "--tolg-rel",
        "1e-3",
        "--tolx-rel",
        "1e-3",
        "--tolx-abs",
        "1e-10",
        "--maxit",
        str(maxit),
        "--linesearch-a",
        "-0.5",
        "--linesearch-b",
        "2.0",
        "--linesearch-tol",
        "1e-3",
        "--element-reorder-mode",
        "block_xyz",
        "--local-hessian-mode",
        "element",
        "--local-coloring",
        "--save-history",
        "--save-linear-timing",
        "--quiet",
    ]
    if method == "newton_linesearch":
        common.extend(["--ksp-type", "gmres", "--no-use-trust-region", "--line-search", "armijo"])
    elif method == "steihaug_trust":
        common.extend(["--ksp-type", "stcg", "--use-trust-region", "--no-trust-subproblem-line-search"])
    else:
        common.extend(["--ksp-type", "stcg", "--use-trust-region", "--trust-subproblem-line-search", "--line-search", "armijo"])
    return tuple(common)


def _topology_args(*, nx: int, ny: int, outer_maxit: int) -> tuple[str, ...]:
    pads = max(2, nx // 24)
    return (
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--fixed_pad_cells",
        str(pads),
        "--load_pad_cells",
        str(pads),
        "--outer_maxit",
        str(outer_maxit),
        *TOPO_BASE_ARGS,
    )


def _p3d_args(*, mesh_name: str, degree: int, backend: str, pmg_strategy: str, maxit: int) -> tuple[str, ...]:
    return (
        "--assembly-backend",
        backend,
        "--solver-backend",
        "local_pmg_mumps",
        "--mesh-name",
        mesh_name,
        "--elem-degree",
        str(degree),
        "--constraint-variant",
        "glued_bottom",
        "--lambda-target",
        "1.55",
        "--pmg-strategy",
        pmg_strategy,
        "--ksp-rtol",
        "1e-2",
        "--ksp-max-it",
        "100",
        "--convergence-mode",
        "all",
        "--stop-tol",
        "2e-3",
        "--grad-stop-tol",
        "1e-4",
        "--maxit",
        str(maxit),
        "--line-search",
        "armijo",
        "--linesearch-tol",
        "1e-3",
    )


def build_case_matrix(mode: str) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    if mode == "smoke":
        for build_mode in ("replicated", "rank_local"):
            cases.append(
                CaseSpec(
                    "he_distribution",
                    f"he_l2_{build_mode}_np2",
                    f"HE L2 {build_mode}",
                    "trust",
                    2,
                    120.0,
                    _he_build_args(level=2, steps=1, maxit=20, pc_type="gamg", build_mode=build_mode),
                    {"level": 2, "build_mode": build_mode, "probe": "correctness"},
                )
            )
        for candidate in ("gamg", "pmg_l2_hypre", "pmg_l2_redundant_mumps", "pmg_l3_redundant_mumps"):
            cases.append(
                CaseSpec(
                    "he_pmg",
                    f"he_l3_{candidate}_np2",
                    f"HE L3 {candidate}",
                    "trust",
                    2,
                    180.0,
                    _he_pmg_args(level=3, maxit=2, candidate=candidate),
                    {"level": 3, "candidate": candidate},
                )
            )
        for ranks in (1, 2):
            cases.append(
                CaseSpec(
                    "topology_consistency",
                    f"topo_64x32_np{ranks}",
                    f"Topology 64x32 np{ranks}",
                    "topology",
                    ranks,
                    120.0,
                    _topology_args(nx=64, ny=32, outer_maxit=3),
                    {"nx": 64, "ny": 32, "outer_maxit": 3},
                )
            )
        gl_level, gl_ranks, gl_cap = 5, 2, 80.0
        p3d_ranks = 2
    else:
        for build_mode in ("replicated", "rank_local"):
            cases.append(
                CaseSpec(
                    "he_distribution",
                    f"he_l4_{build_mode}_np4",
                    f"HE L4 {build_mode}",
                    "trust",
                    4,
                    480.0,
                    _he_build_args(level=4, steps=1, maxit=100, pc_type="gamg", build_mode=build_mode),
                    {"level": 4, "build_mode": build_mode, "probe": "correctness"},
                )
            )
        for ranks in (8, 16, 32):
            for build_mode in ("replicated", "rank_local"):
                cases.append(
                    CaseSpec(
                        "he_distribution",
                        f"he_l5_{build_mode}_np{ranks}_maxit1",
                        f"HE L5 {build_mode} np{ranks}",
                        "trust",
                        ranks,
                        900.0,
                        _he_build_args(level=5, steps=1, maxit=1, pc_type="gamg", build_mode=build_mode),
                        {"level": 5, "build_mode": build_mode, "probe": "memory"},
                    )
                )
        for candidate in ("gamg", "pmg_l2_hypre", "pmg_l2_redundant_mumps", "pmg_l3_redundant_mumps"):
            cases.append(
                CaseSpec(
                    "he_pmg",
                    f"he_l5_{candidate}_np32_maxit8",
                    f"HE L5 {candidate}",
                    "trust",
                    32,
                    1200.0,
                    _he_pmg_args(level=5, maxit=8, candidate=candidate),
                    {"level": 5, "candidate": candidate},
                )
            )
        for ranks in (1, 2, 4, 8, 16, 32):
            cases.append(
                CaseSpec(
                    "topology_consistency",
                    f"topo_384x192_np{ranks}",
                    f"Topology 384x192 np{ranks}",
                    "topology",
                    ranks,
                    900.0,
                    _topology_args(nx=384, ny=192, outer_maxit=40),
                    {"nx": 384, "ny": 192, "outer_maxit": 40},
                )
            )
        gl_level, gl_ranks, gl_cap = 10, 8, 300.0
        p3d_ranks = 32

    for method in ("newton_linesearch", "steihaug_trust", "hybrid_trust_linesearch"):
        cases.append(
            CaseSpec(
                "gl_globalization",
                f"gl_l{gl_level}_{method}_np{gl_ranks}",
                f"GL L{gl_level} {method}",
                "trust",
                gl_ranks,
                gl_cap,
                _gl_args(level=gl_level, maxit=100, method=method),
                {"level": gl_level, "method": method},
            )
        )
    p3d_specs = (P3D_DISCRETIZATION_CASES[0],) if mode == "smoke" else P3D_DISCRETIZATION_CASES
    route_to_backend = dict(P3D_ROUTE_BACKENDS)
    for spec in p3d_specs:
        for route in spec["routes"]:
            backend = route_to_backend[str(route)]
            metadata = {
                "discretization": spec["label"],
                "mesh_case": spec["key"],
                "mesh_name": spec["mesh_name"],
                "degree": spec["degree"],
                "pmg_strategy": spec["pmg_strategy"],
                "route": route,
                "assembly_backend": backend,
                "free_dofs": spec["free_dofs"],
                "local_element_dofs": spec["local_element_dofs"],
                "maxit": 1,
            }
            cases.append(
                CaseSpec(
                    "p3d_derivative_degree",
                    f"p3d_{spec['key']}_{route}_np{p3d_ranks}_maxit1",
                    f"P3D {spec['label']} {route}",
                    "p3d",
                    p3d_ranks,
                    600.0 if mode == "full" else 180.0,
                    _p3d_args(
                        mesh_name=str(spec["mesh_name"]),
                        degree=int(spec["degree"]),
                        backend=backend,
                        pmg_strategy=str(spec["pmg_strategy"]),
                        maxit=1,
                    ),
                    metadata,
                )
            )
    return cases


def _selected_sections(value: str) -> set[str]:
    parts = {part.strip() for part in value.split(",") if part.strip()}
    if not parts or "all" in parts:
        return set(SECTIONS)
    unknown = parts.difference(SECTIONS)
    if unknown:
        raise SystemExit(f"Unknown section(s): {', '.join(sorted(unknown))}")
    return parts


def _case_paths(mode: str, case: CaseSpec) -> tuple[Path, Path, Path]:
    case_dir = RAW_ROOT / mode / case.section / case.key
    return case_dir, case_dir / "output.json", case_dir / "run.log"


def build_command(mode: str, case: CaseSpec) -> list[str]:
    case_dir, output_json, _log_path = _case_paths(mode, case)
    if case.kind == "trust":
        return [
            "mpiexec",
            "-n",
            str(case.nprocs),
            str(PYTHON),
            "-u",
            str(TRUST_RUNNER),
            *case.args,
            "--out",
            str(output_json),
        ]
    if case.kind == "p3d":
        return [
            "mpiexec",
            "-n",
            str(case.nprocs),
            str(PYTHON),
            "-u",
            str(P3D_RUNNER),
            *case.args,
            "--out-dir",
            str(case_dir),
            "--output-json",
            str(output_json),
        ]
    if case.kind == "topology":
        return [
            "mpiexec",
            "-n",
            str(case.nprocs),
            str(PYTHON),
            "-u",
            str(TOPO_RUNNER),
            *case.args,
            "--json_out",
            str(output_json),
            "--state_out",
            str(case_dir / "state.npz"),
        ]
    raise ValueError(f"Unsupported case kind {case.kind!r}")


def _ensure_l10_inputs(mode: str, selected: set[str]) -> None:
    if mode != "full" or "gl_globalization" not in selected:
        return
    required = (
        REPO_ROOT / "data/meshes/pLaplace/pLaplace_level10.h5",
        REPO_ROOT / "data/meshes/GinzburgLandau/GL_level10.h5",
    )
    if all(path.exists() for path in required):
        return
    subprocess.run([str(PYTHON), str(SCALAR_L10_GENERATOR)], cwd=REPO_ROOT, check=True, env=_base_env())


def run_cases(
    mode: str,
    selected: set[str],
    *,
    resume: bool,
    campaign_wall_s: float | None,
    allow_oom_risk: bool,
) -> None:
    _ensure_l10_inputs(mode, selected)
    started = time.perf_counter()
    for case in build_case_matrix(mode):
        if case.section not in selected:
            continue
        if campaign_wall_s is not None and time.perf_counter() - started > float(campaign_wall_s):
            raise SystemExit(f"Campaign wall cap {campaign_wall_s}s reached before {case.key}.")
        case_dir, output_json, log_path = _case_paths(mode, case)
        case_dir.mkdir(parents=True, exist_ok=True)
        argv = build_command(mode, case)
        command = _normalize_command(argv)
        (case_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
        _write_json(
            case_dir / "case_metadata.json",
            {
                "mode": mode,
                "section": case.section,
                "key": case.key,
                "label": case.label,
                "kind": case.kind,
                "nprocs": case.nprocs,
                "wall_cap_s": case.wall_cap_s,
                "metadata": case.metadata,
                "command": command,
            },
        )
        oom_guard_reason = str(case.metadata.get("oom_guard_reason", ""))
        if oom_guard_reason and not allow_oom_risk:
            print(f"[skip-oom-guard] {case.key}", flush=True)
            _write_json(
                case_dir / "run_info.json",
                {
                    "returncode": "",
                    "timed_out": False,
                    "wall_time_s": 0.0,
                    "skipped": True,
                    "skip_reason": oom_guard_reason,
                    "command": command,
                },
            )
            if output_json.exists():
                output_json.unlink()
            continue
        if output_json.exists() and resume:
            print(f"[resume] {case.key}", flush=True)
            continue
        if output_json.exists():
            output_json.unlink()
        print(f"[run] {case.key}", flush=True)
        run_info = _run_subprocess(argv, timeout_s=case.wall_cap_s, log_path=log_path)
        _write_json(case_dir / "run_info.json", {**run_info, "command": command})


def _trust_payload_result(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload.get("result", payload))


def _sum_step_key(steps: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for step in steps:
        for record in step.get("linear_timing", []):
            total += float(record.get(key, 0.0))
    return total


def _sum_history_key(steps: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0)) for step in steps for row in step.get("history", [])))


def _trust_common_row(mode: str, case: CaseSpec) -> dict[str, Any]:
    case_dir, output_json, log_path = _case_paths(mode, case)
    metadata = _read_json(case_dir / "case_metadata.json") if (case_dir / "case_metadata.json").exists() else {}
    run_info = _read_json(case_dir / "run_info.json") if (case_dir / "run_info.json").exists() else {}
    row = {
        "mode": mode,
        "section": case.section,
        "case": case.key,
        "label": case.label,
        "nprocs": case.nprocs,
        "result": "missing_json",
        "failure_mode": "missing_json",
        "returncode": run_info.get("returncode", ""),
        "launcher_wall_time_s": run_info.get("wall_time_s", ""),
        "json_path": _repo_rel(output_json),
        "log_path": _repo_rel(log_path),
        "command": metadata.get("command", ""),
        **case.metadata,
    }
    if bool(run_info.get("skipped", False)):
        row.update(
            {
                "result": "skipped_oom_guard",
                "failure_mode": "skipped_oom_guard",
                "message": run_info.get("skip_reason", ""),
            }
        )
        return row
    if bool(run_info.get("timed_out", False)):
        row.update({"result": "timeout", "failure_mode": "timeout"})
    if not output_json.exists():
        return row
    payload = _read_json(output_json)
    result = _trust_payload_result(payload)
    steps = [dict(step) for step in result.get("steps", [])]
    last = steps[-1] if steps else {}
    completed = bool(steps) and ("converged" in str(last.get("message", "")).lower())
    row.update(
        {
            "result": "completed" if completed else ("fixed_work" if steps else "failed"),
            "failure_mode": "",
            "completed_steps": len(steps),
            "setup_time_s": result.get("setup_time", ""),
            "solve_time_s": result.get("solve_time_total", ""),
            "total_time_s": result.get("total_time", ""),
            "free_dofs": result.get("free_dofs", ""),
            "total_dofs": result.get("total_dofs", ""),
            "newton_iters": int(sum(int(step.get("nit", 0)) for step in steps)),
            "krylov_iters": int(sum(int(step.get("linear_iters", 0)) for step in steps)),
            "line_search_evals": _sum_history_key(steps, "ls_evals"),
            "trust_rejects": _sum_history_key(steps, "trust_rejects"),
            "energy": last.get("energy", ""),
            "message": last.get("message", ""),
        }
    )
    meta = dict(result.get("metadata", {})).get("linear_solver", {})
    resource_usage = dict(meta.get("resource_usage", {}))
    row.update(
        {
            "ru_maxrss_mib_max": resource_usage.get("ru_maxrss_mib_max", ""),
            "ru_maxrss_mib_total": resource_usage.get("ru_maxrss_mib_total", ""),
            "ksp_type": meta.get("ksp_type", ""),
            "pc_type": meta.get("pc_type", ""),
        }
    )
    return row


def _he_memory_metrics(row: dict[str, Any], result: dict[str, Any]) -> None:
    meta = dict(result.get("metadata", {})).get("linear_solver", {})
    memory = dict(meta.get("assembler_memory_by_rank", {}))
    free = _safe_float(result.get("free_dofs", 0.0), 0.0)
    row.update(
        {
            "tracked_total_gib_max": memory.get("tracked_total_gib_max", ""),
            "tracked_total_gib_total": memory.get("tracked_total_gib_total", ""),
            "local_elements_max": memory.get("local_elements_max", ""),
            "local_overlap_dofs_max": memory.get("local_overlap_dofs_max", ""),
            "local_overlap_dofs_total": memory.get("local_overlap_dofs_total", ""),
            "owned_nnz_total": memory.get("owned_nnz_total", ""),
            "overlap_owned_ratio": (
                float(memory.get("local_overlap_dofs_total", 0.0)) / free if free > 0 else ""
            ),
            "problem_build_mode": meta.get("problem_build_mode", ""),
            "mesh_source": meta.get("mesh_source", ""),
            "assembly_backend": meta.get("assembly_backend", ""),
        }
    )


def summarize_he_distribution(mode: str, cases: list[CaseSpec]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        if case.section != "he_distribution":
            continue
        row = _trust_common_row(mode, case)
        output_json = _case_paths(mode, case)[1]
        if output_json.exists():
            result = _trust_payload_result(_read_json(output_json))
            _he_memory_metrics(row, result)
        rows.append(row)
    return rows


def summarize_he_pmg(mode: str, cases: list[CaseSpec]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        if case.section != "he_pmg":
            continue
        row = _trust_common_row(mode, case)
        output_json = _case_paths(mode, case)[1]
        if output_json.exists():
            result = _trust_payload_result(_read_json(output_json))
            steps = [dict(step) for step in result.get("steps", [])]
            meta = dict(result.get("metadata", {})).get("linear_solver", {})
            pmg = dict(meta.get("pmg_hierarchy") or {})
            coarse = dict(pmg.get("coarse_solver") or {})
            row.update(
                {
                    "assemble_time_s": _sum_step_key(steps, "assemble_total_time"),
                    "pc_setup_time_s": _sum_step_key(steps, "pc_setup_time"),
                    "linear_solve_time_s": _sum_step_key(steps, "solve_time"),
                    "linear_total_time_s": _sum_step_key(steps, "linear_total_time"),
                    "pmg_coarsest_level": pmg.get("coarsest_level_resolved", ""),
                    "coarse_pc": coarse.get("pc_type", ""),
                    "coarse_redundant_number": coarse.get("redundant_number", ""),
                    "coarse_factor_solver": coarse.get("factor_solver_type", ""),
                }
            )
        rows.append(row)
    return rows


def summarize_gl_globalization(mode: str, cases: list[CaseSpec]) -> list[dict[str, Any]]:
    rows = []
    labels = {
        "newton_linesearch": "Newton + LS",
        "steihaug_trust": "Steihaug TR",
        "hybrid_trust_linesearch": "Hybrid TR+LS",
    }
    for case in cases:
        if case.section != "gl_globalization":
            continue
        row = _trust_common_row(mode, case)
        method = str(case.metadata.get("method", ""))
        row["method"] = method
        row["method_label"] = labels.get(method, method)
        row["level"] = case.metadata.get("level", "")
        rows.append(row)
    return rows


def summarize_topology(mode: str, cases: list[CaseSpec]) -> list[dict[str, Any]]:
    rows = []
    base_theta = None
    base_compliance = None
    for case in cases:
        if case.section != "topology_consistency":
            continue
        case_dir, output_json, log_path = _case_paths(mode, case)
        run_info = _read_json(case_dir / "run_info.json") if (case_dir / "run_info.json").exists() else {}
        row = {
            "mode": mode,
            "case": case.key,
            "label": case.label,
            "nprocs": case.nprocs,
            "nx": case.metadata.get("nx", ""),
            "ny": case.metadata.get("ny", ""),
            "outer_maxit": case.metadata.get("outer_maxit", ""),
            "result": "missing_json",
            "returncode": run_info.get("returncode", ""),
            "launcher_wall_time_s": run_info.get("wall_time_s", ""),
            "json_path": _repo_rel(output_json),
            "log_path": _repo_rel(log_path),
        }
        if bool(run_info.get("skipped", False)):
            row["result"] = "skipped_oom_guard"
            row["skip_reason"] = run_info.get("skip_reason", "")
            rows.append(row)
            continue
        if output_json.exists():
            payload = _read_json(output_json)
            final = dict(payload.get("final_metrics", {}))
            row.update(
                {
                    "result": payload.get("result", ""),
                    "wall_time_s": payload.get("time", ""),
                    "solve_time_s": float(payload.get("time", 0.0)) - float(payload.get("setup_time", 0.0)),
                    "setup_time_s": payload.get("setup_time", ""),
                    "outer_iterations": final.get("outer_iterations", ""),
                    "final_compliance": final.get("final_compliance", ""),
                    "final_volume_fraction": final.get("final_volume_fraction", ""),
                    "final_p": final.get("final_p_penal", ""),
                }
            )
            state_path = case_dir / "state.npz"
            if state_path.exists():
                theta = np.asarray(np.load(state_path)["theta_grid"], dtype=np.float64)
                if case.nprocs == 1:
                    base_theta = theta
                    base_compliance = _safe_float(row.get("final_compliance"))
                    row["density_rel_l2_vs_np1"] = 0.0
                    row["compliance_rel_diff_vs_np1"] = 0.0
                elif base_theta is not None and base_theta.shape == theta.shape:
                    diff = np.linalg.norm(theta - base_theta)
                    denom = max(np.linalg.norm(base_theta), 1e-30)
                    row["density_rel_l2_vs_np1"] = float(diff / denom)
                    if base_compliance and np.isfinite(base_compliance):
                        row["compliance_rel_diff_vs_np1"] = abs(
                            _safe_float(row.get("final_compliance")) - base_compliance
                        ) / max(abs(base_compliance), 1.0)
        rows.append(row)
    return rows


def summarize_p3d(mode: str, cases: list[CaseSpec]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        if case.section != "p3d_derivative_degree":
            continue
        case_dir, output_json, log_path = _case_paths(mode, case)
        run_info = _read_json(case_dir / "run_info.json") if (case_dir / "run_info.json").exists() else {}
        degree = int(case.metadata.get("degree", 0))
        row = {
            "mode": mode,
            "case": case.key,
            "discretization": case.metadata.get("discretization", f"P{degree}"),
            "mesh_case": case.metadata.get("mesh_case", ""),
            "mesh_name": case.metadata.get("mesh_name", ""),
            "degree": degree,
            "route": case.metadata.get("route", ""),
            "assembly_backend": case.metadata.get("assembly_backend", ""),
            "pmg_strategy": case.metadata.get("pmg_strategy", ""),
            "nprocs": case.nprocs,
            "free_dofs": case.metadata.get("free_dofs", ""),
            "local_element_dofs": case.metadata.get("local_element_dofs", ""),
            "result": "missing_json",
            "returncode": run_info.get("returncode", ""),
            "launcher_wall_time_s": run_info.get("wall_time_s", ""),
            "json_path": _repo_rel(output_json),
            "log_path": _repo_rel(log_path),
        }
        if bool(run_info.get("skipped", False)):
            row.update({"result": "skipped_oom_guard", "skip_reason": run_info.get("skip_reason", "")})
            rows.append(row)
            continue
        if output_json.exists():
            payload = _read_json(output_json)
            callbacks = dict(payload.get("assembly_callbacks") or {})
            hessian = dict(callbacks.get("hessian") or {})
            diag = dict(payload.get("assembler_rank_diagnostics") or {})
            coloring = dict(diag.get("sfd_coloring") or {})
            resource_usage = dict(diag.get("resource_usage") or {})
            memory = dict(payload.get("assembler_memory") or {})
            finite_metrics = bool(
                np.isfinite(_safe_float(payload.get("energy")))
                and np.isfinite(_safe_float(payload.get("final_grad_norm")))
            )
            result = str(payload.get("status", ""))
            if int(case.metadata.get("maxit", 0)) == 1 and finite_metrics:
                result = "fixed_work"
            row.update(
                {
                    "result": result,
                    "newton_iters": payload.get("nit", ""),
                    "krylov_iters": payload.get("linear_iterations_total", ""),
                    "solve_time_s": payload.get("solve_time", ""),
                    "total_time_s": payload.get("total_time", ""),
                    "energy": payload.get("energy", ""),
                    "final_grad_norm": payload.get("final_grad_norm", ""),
                    "hessian_hvp_time_s": hessian.get("hvp_compute", ""),
                    "hessian_time_s": hessian.get("total", ""),
                    "sfd_colors_min": coloring.get("colors_min", ""),
                    "sfd_colors_max": coloring.get("colors_max", ""),
                    "ru_maxrss_mib_max": resource_usage.get("ru_maxrss_mib_max", ""),
                    "ru_maxrss_mib_total": resource_usage.get("ru_maxrss_mib_total", ""),
                    "local_elements": memory.get("local_elements", ""),
                    "local_overlap_dofs": memory.get("local_overlap_dofs", ""),
                    "finite_metrics": finite_metrics,
                }
            )
        rows.append(row)
    return rows


def summarize(mode: str, selected: set[str]) -> dict[str, list[dict[str, Any]]]:
    cases = [case for case in build_case_matrix(mode) if case.section in selected]
    by_section = {
        "he_distribution": summarize_he_distribution(mode, cases),
        "he_pmg": summarize_he_pmg(mode, cases),
        "topology_consistency": summarize_topology(mode, cases),
        "gl_globalization": summarize_gl_globalization(mode, cases),
        "p3d_derivative_degree": summarize_p3d(mode, cases),
    }
    fields = {
        "he_distribution": [
            "mode", "case", "label", "probe", "level", "build_mode", "nprocs", "result",
            "completed_steps", "newton_iters", "krylov_iters", "energy", "setup_time_s",
            "solve_time_s", "total_time_s", "ru_maxrss_mib_max", "ru_maxrss_mib_total",
            "tracked_total_gib_max", "tracked_total_gib_total", "local_elements_max",
            "local_overlap_dofs_max", "local_overlap_dofs_total", "overlap_owned_ratio",
            "problem_build_mode", "mesh_source", "assembly_backend", "json_path", "log_path",
        ],
        "he_pmg": [
            "mode", "case", "label", "candidate", "level", "nprocs", "result",
            "newton_iters", "krylov_iters", "energy", "setup_time_s", "solve_time_s",
            "total_time_s", "assemble_time_s", "pc_setup_time_s", "linear_solve_time_s",
            "linear_total_time_s", "pc_type", "pmg_coarsest_level", "coarse_pc",
            "coarse_redundant_number", "coarse_factor_solver", "json_path", "log_path",
        ],
        "topology_consistency": [
            "mode", "case", "label", "nprocs", "nx", "ny", "outer_maxit", "result",
            "outer_iterations", "wall_time_s", "solve_time_s", "setup_time_s",
            "final_compliance", "final_volume_fraction", "final_p",
            "compliance_rel_diff_vs_np1", "density_rel_l2_vs_np1", "json_path", "log_path",
        ],
        "gl_globalization": [
            "mode", "case", "method", "method_label", "level", "nprocs", "result",
            "completed_steps", "newton_iters", "krylov_iters", "line_search_evals",
            "trust_rejects", "setup_time_s", "solve_time_s", "total_time_s", "energy",
            "message", "json_path", "log_path",
        ],
        "p3d_derivative_degree": [
            "mode", "case", "discretization", "mesh_case", "mesh_name", "degree",
            "route", "assembly_backend", "pmg_strategy", "nprocs", "free_dofs",
            "local_element_dofs", "local_elements", "local_overlap_dofs", "result",
            "newton_iters", "krylov_iters", "solve_time_s", "total_time_s",
            "hessian_hvp_time_s", "hessian_time_s", "sfd_colors_min", "sfd_colors_max",
            "ru_maxrss_mib_max", "ru_maxrss_mib_total", "energy", "final_grad_norm",
            "finite_metrics", "skip_reason", "json_path", "log_path",
        ],
    }
    for section, rows in by_section.items():
        if section not in selected:
            continue
        _write_csv(REPORT_ROOT / f"{mode}_{section}.csv", rows, fields[section])
    _write_json(REPORT_ROOT / f"{mode}_summary.json", by_section)
    return by_section


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full", "summarize"), required=True)
    parser.add_argument("--sections", default="all")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--campaign-wall-s", type=float, default=None)
    parser.add_argument(
        "--allow-oom-risk",
        action="store_true",
        help="Run rows that are known to exceed the local workstation memory budget.",
    )
    args = parser.parse_args()

    selected = _selected_sections(str(args.sections))
    run_mode = "full" if args.mode == "summarize" else str(args.mode)
    if args.mode != "summarize":
        run_cases(
            run_mode,
            selected,
            resume=bool(args.resume),
            campaign_wall_s=args.campaign_wall_s,
            allow_oom_risk=bool(args.allow_oom_risk),
        )
    summarize(run_mode, selected)
    print(f"Reports written under {_repo_rel(REPORT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
