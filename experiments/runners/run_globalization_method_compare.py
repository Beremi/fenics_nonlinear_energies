#!/usr/bin/env python3
"""Compare nonlinear globalization policies on maintained benchmark cases.

Full mode uses generated level-10 scalar meshes. They are intentionally ignored
local artifacts; create them from the checked-in level-9 inputs with:

    ./.venv/bin/python experiments/runners/generate_scalar_uniform_l10_meshes.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.core.benchmark.run_record import atomic_write_json, sha256_file
from src.core.benchmark.state_export import (
    export_hyperelasticity_state_npz,
    export_scalar_mesh_state_npz,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_RUNNER = REPO_ROOT / "experiments/runners/run_trust_region_case.py"
PLASTICITY3D_RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_backend_mix_case.py"
RAW_ROOT = REPO_ROOT / "artifacts/raw_results/globalization_method_compare"
REPORT_ROOT = REPO_ROOT / "artifacts/reports/globalization_method_compare"
GENERATE_L10_COMMAND = (
    "./.venv/bin/python experiments/runners/generate_scalar_uniform_l10_meshes.py"
)
GENERATED_FULL_MODE_INPUTS = (
    REPO_ROOT / "data/meshes/pLaplace/pLaplace_level10.h5",
    REPO_ROOT / "data/meshes/GinzburgLandau/GL_level10.h5",
)


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(int(value) for value in array.shape)).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    label: str
    problem: str
    backend: str
    level: int
    nprocs: int
    steps: int
    start_step: int
    total_steps: int
    wall_cap_s: float
    line_ksp_type: str
    trust_ksp_type: str
    common_args: tuple[str, ...]
    runner: str = "trust_region_case"


@dataclass(frozen=True)
class CaseSpec:
    benchmark: BenchmarkSpec
    method: MethodSpec
    comparison_tier: str = "production_bundle"

    @property
    def key(self) -> str:
        return f"{self.benchmark.key}_{self.method.key}"


PRODUCTION_BUNDLE_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="newton_linesearch",
        label="Newton + line search",
        args=("--no-use-trust-region", "--line-search", "armijo"),
    ),
    MethodSpec(
        key="steihaug_trust",
        label="Steihaug trust region",
        args=("--use-trust-region", "--no-trust-subproblem-line-search"),
    ),
    MethodSpec(
        key="hybrid_trust_linesearch",
        label="Hybrid trust region + line search",
        args=("--use-trust-region", "--trust-subproblem-line-search", "--line-search", "armijo"),
    ),
)

# This tier holds the discrete problem, Hessian solve, KSP, preconditioner,
# forcing tolerance, starting state, and stopping contract fixed.  The trust
# method deliberately uses the minimizer's reduced-subspace subproblem so the
# same non-trust KSP can be used for both rows.  Plasticity3D is excluded until
# a branch-stable nonlinear benchmark with the same controlled solve is frozen.
CONTROLLED_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="newton_armijo",
        label="Newton + Armijo line search",
        args=("--no-use-trust-region", "--line-search", "armijo"),
    ),
    MethodSpec(
        key="reduced_trust_armijo",
        label="Reduced-subspace trust region + Armijo",
        args=(
            "--use-trust-region",
            "--no-trust-subproblem-line-search",
            "--line-search",
            "armijo",
        ),
    ),
)

# Backward-compatible public name used by older scripts.
METHODS = PRODUCTION_BUNDLE_METHODS


def _scalar_common(*, ksp_rtol: float, ksp_max_it: int, tolf: float, tolg: float, linesearch_tol: float) -> tuple[str, ...]:
    return (
        "--profile",
        "reference",
        "--pc-type",
        "hypre",
        "--ksp-rtol",
        str(ksp_rtol),
        "--ksp-max-it",
        str(ksp_max_it),
        "--tolf",
        str(tolf),
        "--tolg",
        str(tolg),
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
        str(linesearch_tol),
        "--element-reorder-mode",
        "block_xyz",
        "--local-hessian-mode",
        "element",
        "--local-coloring",
        "--save-history",
        "--save-linear-timing",
        "--quiet",
    )


def _he_common() -> tuple[str, ...]:
    return (
        "--profile",
        "performance",
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
        "1e-4",
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
        "--element-reorder-mode",
        "block_xyz",
        "--local-hessian-mode",
        "element",
        "--problem-build-mode",
        "rank_local",
        "--he-mesh-source",
        "procedural",
        "--he-element-degree",
        "1",
        "--distribution-strategy",
        "overlap_p2p",
        "--assembly-backend",
        "coo_local",
        "--local-coloring",
        "--save-history",
        "--save-linear-timing",
        "--quiet",
    )


def _plasticity3d_common() -> tuple[str, ...]:
    return (
        "--assembly-backend",
        "local",
        "--solver-backend",
        "local_pmg_mumps",
        "--mesh-name",
        "hetero_ssr_L1",
        "--elem-degree",
        "2",
        "--constraint-variant",
        "glued_bottom",
        "--lambda-target",
        "1.55",
        "--pmg-strategy",
        "same_mesh_p2_p1",
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
        "80",
        "--line-search",
        "armijo",
        "--linesearch-tol",
        "1e-3",
    )


SMOKE_BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        key="plaplace_l5_np2",
        label="p-Laplace L5",
        problem="plaplace",
        backend="element",
        level=5,
        nprocs=2,
        steps=1,
        start_step=1,
        total_steps=1,
        wall_cap_s=20.0,
        line_ksp_type="cg",
        trust_ksp_type="stcg",
        common_args=_scalar_common(ksp_rtol=1e-1, ksp_max_it=30, tolf=1e-4, tolg=1e-3, linesearch_tol=1e-1),
    ),
    BenchmarkSpec(
        key="gl_l5_np2",
        label="Ginzburg--Landau L5",
        problem="gl",
        backend="element",
        level=5,
        nprocs=2,
        steps=1,
        start_step=1,
        total_steps=1,
        wall_cap_s=20.0,
        line_ksp_type="gmres",
        trust_ksp_type="stcg",
        common_args=_scalar_common(ksp_rtol=1e-3, ksp_max_it=200, tolf=1e-6, tolg=1e-5, linesearch_tol=1e-3),
    ),
    BenchmarkSpec(
        key="he_l2_np2_steps2",
        label="HyperElasticity L2",
        problem="he",
        backend="element",
        level=2,
        nprocs=2,
        steps=2,
        start_step=1,
        total_steps=24,
        wall_cap_s=80.0,
        line_ksp_type="gmres",
        trust_ksp_type="stcg",
        common_args=_he_common(),
    ),
    BenchmarkSpec(
        key="plasticity3d_p2_l1_np2_lambda155",
        label="Plasticity3D P2(L1)",
        problem="plasticity3d",
        backend="element",
        level=1,
        nprocs=2,
        steps=1,
        start_step=1,
        total_steps=1,
        wall_cap_s=30.0,
        line_ksp_type="fgmres",
        trust_ksp_type="stcg",
        common_args=_plasticity3d_common(),
        runner="plasticity3d_backend_mix",
    ),
)


FULL_BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        key="plaplace_l10_np32",
        label="p-Laplace L10",
        problem="plaplace",
        backend="element",
        level=10,
        nprocs=32,
        steps=1,
        start_step=1,
        total_steps=1,
        wall_cap_s=30.0,
        line_ksp_type="cg",
        trust_ksp_type="stcg",
        common_args=_scalar_common(ksp_rtol=1e-1, ksp_max_it=30, tolf=1e-4, tolg=1e-3, linesearch_tol=1e-1),
    ),
    BenchmarkSpec(
        key="gl_l10_np16",
        label="Ginzburg--Landau L10",
        problem="gl",
        backend="element",
        level=10,
        nprocs=16,
        steps=1,
        start_step=1,
        total_steps=1,
        wall_cap_s=120.0,
        line_ksp_type="gmres",
        trust_ksp_type="stcg",
        common_args=_scalar_common(ksp_rtol=1e-3, ksp_max_it=200, tolf=1e-6, tolg=1e-5, linesearch_tol=1e-3),
    ),
    BenchmarkSpec(
        key="he_l4_np32_steps8",
        label="HyperElasticity L4",
        problem="he",
        backend="element",
        level=4,
        nprocs=32,
        steps=8,
        start_step=1,
        total_steps=24,
        wall_cap_s=270.0,
        line_ksp_type="gmres",
        trust_ksp_type="stcg",
        common_args=_he_common(),
    ),
    BenchmarkSpec(
        key="plasticity3d_p2_l1_np32_lambda155",
        label="Plasticity3D P2(L1)",
        problem="plasticity3d",
        backend="element",
        level=1,
        nprocs=32,
        steps=1,
        start_step=1,
        total_steps=1,
        wall_cap_s=180.0,
        line_ksp_type="fgmres",
        trust_ksp_type="stcg",
        common_args=_plasticity3d_common(),
        runner="plasticity3d_backend_mix",
    ),
)


CSV_FIELDS = (
    "mode",
    "comparison_tier",
    "benchmark",
    "benchmark_label",
    "method",
    "method_label",
    "problem",
    "level",
    "nprocs",
    "steps_requested",
    "completed_steps",
    "result",
    "failure_mode",
    "returncode",
    "wall_time_s",
    "solve_time_s",
    "total_time_s",
    "setup_time_s",
    "newton_iters",
    "krylov_iters",
    "line_search_evals",
    "line_search_time_s",
    "trust_rejects",
    "final_energy",
    "initial_state_file_sha256",
    "initial_state_content_sha256",
    "final_state_file_sha256",
    "final_state_content_sha256",
    "endpoint_state_sha256",
    "independent_dual_residual",
    "independent_coefficient_residual",
    "independent_residual_sha256",
    "json_path",
    "log_path",
    "command",
)


def build_case_matrix(
    mode: str,
    comparison_tier: str = "production_bundle",
) -> list[CaseSpec]:
    benchmarks = SMOKE_BENCHMARKS if mode == "smoke" else FULL_BENCHMARKS
    if comparison_tier == "controlled":
        controlled: list[BenchmarkSpec] = []
        for benchmark in benchmarks:
            # p-Laplace is intentionally excluded: its historical initializer
            # was random and is not a retained prescribed globalization unit.
            if benchmark.problem not in {"gl", "he"}:
                continue
            if benchmark.problem == "he":
                benchmark = replace(
                    benchmark,
                    key=f"he_l{benchmark.level}_np{benchmark.nprocs}_step1",
                    label=f"{benchmark.label}, first load",
                    steps=1,
                )
            controlled.append(benchmark)
        benchmarks = tuple(controlled)
        methods = CONTROLLED_METHODS
    elif comparison_tier == "production_bundle":
        methods = PRODUCTION_BUNDLE_METHODS
    else:
        raise ValueError(f"unknown comparison tier: {comparison_tier!r}")
    return [
        CaseSpec(
            benchmark=benchmark,
            method=method,
            comparison_tier=comparison_tier,
        )
        for benchmark in benchmarks
        for method in methods
    ]


def require_generated_inputs(mode: str) -> None:
    if mode != "full":
        return
    missing = [path for path in GENERATED_FULL_MODE_INPUTS if not path.exists()]
    if not missing:
        return
    formatted = "\n".join(f"  - {_display_path(path)}" for path in missing)
    raise SystemExit(
        "Full globalization comparison requires generated scalar level-10 meshes.\n"
        "Missing:\n"
        f"{formatted}\n"
        "Generate them with:\n"
        f"  {GENERATE_L10_COMMAND}"
    )


def canonical_start_path(raw_dir: Path, benchmark: BenchmarkSpec) -> Path:
    return raw_dir / "_canonical_starts" / f"{benchmark.key}.npz"


def prepare_canonical_start(
    benchmark: BenchmarkSpec,
    destination: Path,
) -> dict[str, Any]:
    """Create the one immutable starting-state artifact shared by both methods."""

    destination = destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite canonical start {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if benchmark.problem == "gl":
        from src.problems.ginzburg_landau.jax.mesh import MeshGL2D

        mesh = MeshGL2D(int(benchmark.level))
        params = mesh.params
        full_state = np.asarray(params["u_0"], dtype=np.float64).copy()
        full_state[np.asarray(params["freedofs"], dtype=np.int64)] = np.asarray(
            mesh.u_init, dtype=np.float64
        )
        state_sha256 = _array_sha256(full_state)
        export_scalar_mesh_state_npz(
            destination,
            coords=np.asarray(params["nodes"], dtype=np.float64),
            triangles=np.asarray(params["elems"], dtype=np.int32),
            u=full_state,
            mesh_level=int(benchmark.level),
            problem_name="GinzburgLandau2D",
            metadata={
                "artifact_role": "EXP-GLOB-001 canonical initial state",
                "state_sha256": state_sha256,
                "initializer": "sin(pi*(x-1)/2)*sin(pi*(y-1)/2) on free nodes",
            },
        )
    elif benchmark.problem == "he":
        from src.problems.hyperelasticity.support.mesh import (
            build_procedural_hyperelasticity_export_params,
        )

        params = build_procedural_hyperelasticity_export_params(int(benchmark.level))
        coords_ref = np.asarray(params["nodes2coord"], dtype=np.float64)
        full_state = np.asarray(params["u_0_ref"], dtype=np.float64).reshape((-1, 3))
        state_sha256 = _array_sha256(full_state)
        export_hyperelasticity_state_npz(
            destination,
            coords_ref=coords_ref,
            x_final=full_state,
            tetrahedra=np.asarray(params["elems_scalar"], dtype=np.int32),
            mesh_level=int(benchmark.level),
            total_steps=int(benchmark.total_steps),
            metadata={
                "artifact_role": "EXP-GLOB-001 canonical initial state",
                "state_sha256": state_sha256,
                "initializer": "reference deformation y(X)=X",
            },
        )
    else:
        raise ValueError(
            f"No retained deterministic controlled initializer for {benchmark.problem!r}"
        )
    return {
        "benchmark": benchmark.key,
        "problem": benchmark.problem,
        "level": int(benchmark.level),
        "path": str(destination),
        "file_sha256": sha256_file(destination),
        "state_sha256": state_sha256,
    }


def prepare_controlled_starts(
    cases: list[CaseSpec], raw_dir: Path
) -> dict[str, dict[str, Any]]:
    starts: dict[str, dict[str, Any]] = {}
    for case in cases:
        benchmark = case.benchmark
        if benchmark.key in starts:
            continue
        starts[benchmark.key] = prepare_canonical_start(
            benchmark, canonical_start_path(raw_dir, benchmark)
        )
    atomic_write_json(
        raw_dir / "_canonical_starts" / "manifest.json",
        {
            "schema_id": "fenics-nonlinear-energies.exp-glob-001-common-starts",
            "schema_version": 1,
            "status": "prepared",
            "benchmarks": starts,
        },
    )
    return starts


def build_command(
    case: CaseSpec,
    out_path: Path,
    *,
    state_in: Path | None = None,
    state_out: Path | None = None,
) -> list[str]:
    benchmark = case.benchmark
    method = case.method
    if benchmark.runner == "plasticity3d_backend_mix":
        return [
            "mpiexec",
            "-n",
            str(benchmark.nprocs),
            sys.executable,
            "-u",
            str(PLASTICITY3D_RUNNER),
            "--out-dir",
            str(out_path.parent),
            "--output-json",
            str(out_path),
            *benchmark.common_args,
            *method.args,
        ]

    if case.comparison_tier == "controlled":
        ksp_type = benchmark.line_ksp_type
    else:
        ksp_type = (
            benchmark.line_ksp_type
            if method.key == "newton_linesearch"
            else benchmark.trust_ksp_type
        )
    cmd = [
        "mpiexec",
        "-n",
        str(benchmark.nprocs),
        sys.executable,
        "-u",
        str(CASE_RUNNER),
        "--problem",
        benchmark.problem,
        "--backend",
        benchmark.backend,
        "--level",
        str(benchmark.level),
        "--out",
        str(out_path),
        "--steps",
        str(benchmark.steps),
        "--start-step",
        str(benchmark.start_step),
        "--total-steps",
        str(benchmark.total_steps),
        "--ksp-type",
        ksp_type,
    ]
    if case.comparison_tier == "controlled":
        if state_in is None or state_out is None:
            raise ValueError("controlled globalization commands require state_in/state_out")
        cmd.extend(
            [
                "--state-in",
                str(Path(state_in).resolve()),
                "--state-out",
                str(Path(state_out).resolve()),
            ]
        )
    cmd.extend([*benchmark.common_args, *method.args])
    return cmd


def _terminate_process_group(proc: subprocess.Popen[str], grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=5.0)


def _child_preexec() -> None:
    os.setsid()


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _sum_step_linear(step: dict[str, Any]) -> int:
    if "linear_iters" in step:
        return int(step.get("linear_iters") or 0)
    return int(sum(int(rec.get("ksp_its", 0)) for rec in step.get("linear_timing", [])))


def _sum_history(step: dict[str, Any], key: str) -> float:
    return float(sum(float(rec.get(key, 0.0)) for rec in step.get("history", [])))


def _sum_flat_history(history: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(rec.get(key, 0.0)) for rec in history))


def _write_fallback_payload(
    *,
    case: CaseSpec,
    out_path: Path,
    command: list[str],
    result: str,
    failure_mode: str,
    returncode: int | None,
    wall_time_s: float,
) -> dict[str, Any]:
    payload = {
        "case": {
            "problem": case.benchmark.problem,
            "backend": case.benchmark.backend,
            "level": case.benchmark.level,
            "nprocs": case.benchmark.nprocs,
            "method": case.method.key,
            "method_label": case.method.label,
            "comparison_tier": case.comparison_tier,
            "command": command,
        },
        "result": {
            "status": result,
            "failure_mode": failure_mode,
            "returncode": returncode,
            "wall_time_s": wall_time_s,
            "steps": [],
        },
    }
    atomic_write_json(out_path, payload, nonfinite_as_null=True)
    return payload


def summarize_payload(
    *,
    mode: str,
    case: CaseSpec,
    payload: dict[str, Any],
    json_path: Path,
    log_path: Path,
    command: list[str],
    returncode: int | None,
    wall_time_s: float,
    launcher_failure: str | None = None,
) -> dict[str, Any]:
    result = payload if case.benchmark.runner == "plasticity3d_backend_mix" else payload.get("result", {})
    steps = list(result.get("steps", []))
    expected_steps = int(case.benchmark.steps)
    completed_steps = len(steps)
    failure_mode = launcher_failure

    plasticity_history = list(result.get("history", []))
    is_plasticity = case.benchmark.runner == "plasticity3d_backend_mix"

    if launcher_failure == "timeout":
        row_result = "timeout"
    elif returncode not in (None, 0):
        row_result = "launcher_failed"
        failure_mode = failure_mode or f"returncode {returncode}"
    elif is_plasticity:
        status = str(result.get("status", ""))
        message = str(result.get("message", ""))
        completed_steps = 1 if status == "completed" else 0
        if status == "completed":
            row_result = "completed"
            failure_mode = ""
        else:
            row_result = "failed"
            failure_mode = failure_mode or message or status or "not converged"
    elif not steps:
        row_result = "failed"
        failure_mode = failure_mode or str(result.get("failure_mode") or "no-steps")
    else:
        last_message = str(steps[-1].get("message", ""))
        if completed_steps < expected_steps:
            row_result = "failed"
            failure_mode = failure_mode or f"incomplete: {completed_steps}/{expected_steps} steps"
        elif any(step.get("kill_switch_exceeded") for step in steps):
            row_result = "timeout"
            failure_mode = failure_mode or "step-time-limit"
        elif "converged" in last_message.lower():
            row_result = "completed"
            failure_mode = ""
        else:
            row_result = "failed"
            failure_mode = failure_mode or last_message or "not converged"

    solve_time = _safe_float(result.get("solve_time_total"))
    if solve_time is None:
        solve_time = _safe_float(result.get("solve_time"))
    total_time = _safe_float(result.get("total_time"))
    setup_time = _safe_float(result.get("setup_time"))
    final_energy = _safe_float(result.get("energy")) if is_plasticity else (_safe_float(steps[-1].get("energy")) if steps else None)
    newton_iters = int(result.get("nit", 0)) if is_plasticity else int(sum(int(step.get("nit", step.get("iters", 0))) for step in steps))
    krylov_iters = (
        int(result.get("linear_iterations_total", 0))
        if is_plasticity
        else int(sum(_sum_step_linear(step) for step in steps))
    )
    line_search_evals = (
        int(_sum_flat_history(plasticity_history, "ls_evals"))
        if is_plasticity
        else int(sum(int(_sum_history(step, "ls_evals")) for step in steps))
    )
    line_search_time = (
        float(_sum_flat_history(plasticity_history, "t_ls"))
        if is_plasticity
        else float(sum(_sum_history(step, "t_ls") for step in steps))
    )
    trust_rejects = (
        int(_sum_flat_history(plasticity_history, "trust_rejects"))
        if is_plasticity
        else int(sum(int(_sum_history(step, "trust_rejects")) for step in steps))
    )
    metadata = result.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    initial_identity = metadata.get("initial_state_input", {})
    initial_identity = initial_identity if isinstance(initial_identity, dict) else {}
    endpoint_identity = metadata.get("endpoint_identity", {})
    endpoint_identity = endpoint_identity if isinstance(endpoint_identity, dict) else {}
    state_output = metadata.get("state_output", {})
    state_output = state_output if isinstance(state_output, dict) else {}
    independent = endpoint_identity.get("independent_residual", {})
    independent = independent if isinstance(independent, dict) else {}

    return {
        "mode": mode,
        "comparison_tier": case.comparison_tier,
        "benchmark": case.benchmark.key,
        "benchmark_label": case.benchmark.label,
        "method": case.method.key,
        "method_label": case.method.label,
        "problem": case.benchmark.problem,
        "level": case.benchmark.level,
        "nprocs": case.benchmark.nprocs,
        "steps_requested": expected_steps,
        "completed_steps": completed_steps,
        "result": row_result,
        "failure_mode": failure_mode or "",
        "returncode": "" if returncode is None else int(returncode),
        "wall_time_s": float(wall_time_s),
        "solve_time_s": "" if solve_time is None else float(solve_time),
        "total_time_s": "" if total_time is None else float(total_time),
        "setup_time_s": "" if setup_time is None else float(setup_time),
        "newton_iters": int(newton_iters),
        "krylov_iters": int(krylov_iters),
        "line_search_evals": int(line_search_evals),
        "line_search_time_s": float(line_search_time),
        "trust_rejects": int(trust_rejects),
        "final_energy": "" if final_energy is None else float(final_energy),
        "initial_state_file_sha256": str(initial_identity.get("file_sha256") or ""),
        "initial_state_content_sha256": str(initial_identity.get("state_sha256") or ""),
        "final_state_file_sha256": str(state_output.get("file_sha256") or ""),
        "final_state_content_sha256": str(state_output.get("state_sha256") or ""),
        "endpoint_state_sha256": str(
            endpoint_identity.get("owned_reordered_state_sha256") or ""
        ),
        "independent_dual_residual": (
            "" if _safe_float(independent.get("dual_norm")) is None
            else float(independent["dual_norm"])
        ),
        "independent_coefficient_residual": (
            "" if _safe_float(independent.get("coefficient_l2_norm")) is None
            else float(independent["coefficient_l2_norm"])
        ),
        "independent_residual_sha256": str(
            independent.get("owned_reordered_gradient_sha256") or ""
        ),
        "json_path": _display_path(json_path),
        "log_path": _display_path(log_path),
        "command": shlex.join(command),
    }


def run_case(
    case: CaseSpec,
    *,
    mode: str,
    raw_dir: Path,
    timeout_s: float,
    canonical_start: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir = raw_dir / case.key
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / "output.json"
    log_path = case_dir / "run.log"
    state_in = (
        Path(str(canonical_start["path"])) if canonical_start is not None else None
    )
    state_out = case_dir / "final_state.npz" if canonical_start is not None else None
    command = build_command(
        case,
        out_path,
        state_in=state_in,
        state_out=state_out,
    )

    start = time.perf_counter()
    returncode: int | None = None
    launcher_failure: str | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=_child_preexec,
        )
        try:
            returncode = proc.wait(timeout=max(1.0, float(timeout_s)))
        except subprocess.TimeoutExpired:
            launcher_failure = "timeout"
            _terminate_process_group(proc)
            returncode = proc.returncode
            log.write(f"\n[runner] timeout after {timeout_s:.3f} s\n")
    wall_time_s = time.perf_counter() - start

    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = _write_fallback_payload(
                case=case,
                out_path=out_path,
                command=command,
                result="failed",
                failure_mode="invalid-json",
                returncode=returncode,
                wall_time_s=wall_time_s,
            )
            launcher_failure = launcher_failure or "invalid-json"
    else:
        payload = _write_fallback_payload(
            case=case,
            out_path=out_path,
            command=command,
            result="timeout" if launcher_failure == "timeout" else "failed",
            failure_mode=launcher_failure or "missing-output",
            returncode=returncode,
            wall_time_s=wall_time_s,
        )
    return summarize_payload(
        mode=mode,
        case=case,
        payload=payload,
        json_path=out_path,
        log_path=log_path,
        command=command,
        returncode=returncode,
        wall_time_s=wall_time_s,
        launcher_failure=launcher_failure,
    )


def controlled_identity_audit(
    rows: list[dict[str, Any]],
    canonical_starts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["benchmark"]), []).append(row)
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    required_methods = {method.key for method in CONTROLLED_METHODS}
    for benchmark, group in sorted(groups.items()):
        methods = {str(row["method"]) for row in group}
        start_files = {
            str(row["initial_state_file_sha256"])
            for row in group
            if row.get("initial_state_file_sha256")
        }
        start_states = {
            str(row["initial_state_content_sha256"])
            for row in group
            if row.get("initial_state_content_sha256")
        }
        if methods != required_methods:
            errors.append(f"{benchmark}: controlled method set is incomplete")
        if len(start_files) != 1 or len(start_states) != 1:
            errors.append(f"{benchmark}: methods do not prove one common hashed start")
        expected_start = (canonical_starts or {}).get(benchmark)
        if expected_start is not None and (
            start_files != {str(expected_start["file_sha256"])}
            or start_states != {str(expected_start["state_sha256"])}
        ):
            errors.append(
                f"{benchmark}: reported common start differs from the canonical manifest"
            )
        endpoint_hashes = {
            str(row["final_state_content_sha256"])
            for row in group
            if row.get("final_state_content_sha256")
        }
        row_summaries: list[dict[str, Any]] = []
        for row in group:
            terminal_output_expected = row.get("result") in {"completed", "failed"}
            missing_terminal = [
                field
                for field in (
                    "final_state_file_sha256",
                    "final_state_content_sha256",
                    "endpoint_state_sha256",
                    "independent_residual_sha256",
                )
                if terminal_output_expected and not row.get(field)
            ]
            dual = _safe_float(row.get("independent_dual_residual"))
            coefficient = _safe_float(row.get("independent_coefficient_residual"))
            if terminal_output_expected and (
                missing_terminal
                or dual is None
                or coefficient is None
                or not np.isfinite(dual)
                or not np.isfinite(coefficient)
            ):
                errors.append(
                    f"{benchmark}/{row['method']}: terminal state or independent residual identity is incomplete"
                )
            row_summaries.append(
                {
                    "method": row["method"],
                    "result": row["result"],
                    "final_state_content_sha256": row.get(
                        "final_state_content_sha256", ""
                    ),
                    "endpoint_state_sha256": row.get("endpoint_state_sha256", ""),
                    "independent_dual_residual": row.get(
                        "independent_dual_residual", ""
                    ),
                    "independent_coefficient_residual": row.get(
                        "independent_coefficient_residual", ""
                    ),
                    "independent_residual_sha256": row.get(
                        "independent_residual_sha256", ""
                    ),
                }
            )
        summaries.append(
            {
                "benchmark": benchmark,
                "common_start_file_sha256": (
                    next(iter(start_files)) if len(start_files) == 1 else None
                ),
                "common_start_content_sha256": (
                    next(iter(start_states)) if len(start_states) == 1 else None
                ),
                "endpoint_content_identity_equal": len(endpoint_hashes) == 1,
                "endpoint_comparison_scope": (
                    "same_canonical_endpoint"
                    if len(endpoint_hashes) == 1
                    else "different_or_incomplete_endpoints"
                ),
                "methods": row_summaries,
            }
        )
    return {
        "schema_id": "fenics-nonlinear-energies.exp-glob-001-identity-audit",
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "timing_claim_admissible": False,
        "errors": errors,
        "benchmarks": summaries,
    }


def write_reports(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    comparison_tier: str,
    report_dir: Path,
    canonical_starts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{mode}_summary.csv"
    json_path = report_dir / f"{mode}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(
        json_path,
        {
            "mode": mode,
            "comparison_tier": comparison_tier,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rows": rows,
        },
        nonfinite_as_null=True,
    )
    identity_audit: dict[str, Any] | None = None
    if comparison_tier == "controlled":
        identity_audit = controlled_identity_audit(rows, canonical_starts)
        atomic_write_json(
            report_dir / f"{mode}_identity_audit.json",
            identity_audit,
            nonfinite_as_null=True,
        )
    return identity_audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--comparison-tier",
        choices=("controlled", "production_bundle"),
        default="production_bundle",
        help=(
            "controlled keeps the Hessian solve/KSP policy fixed; "
            "production_bundle retains the existing three complete solver bundles"
        ),
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--campaign-wall-s", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cases = build_case_matrix(args.mode, args.comparison_tier)
    raw_root = (
        args.raw_root / "controlled"
        if args.comparison_tier == "controlled"
        else args.raw_root
    )
    report_root = (
        args.report_root / "controlled"
        if args.comparison_tier == "controlled"
        else args.report_root
    )
    if args.dry_run:
        for case in cases:
            out_path = raw_root / args.mode / case.key / "output.json"
            state_in = (
                canonical_start_path(raw_root / args.mode, case.benchmark)
                if args.comparison_tier == "controlled"
                else None
            )
            state_out = (
                out_path.parent / "final_state.npz"
                if args.comparison_tier == "controlled"
                else None
            )
            print(
                shlex.join(
                    build_command(
                        case,
                        out_path,
                        state_in=state_in,
                        state_out=state_out,
                    )
                )
            )
        return

    require_generated_inputs(args.mode)

    rows: list[dict[str, Any]] = []
    raw_dir = raw_root / args.mode
    controlled_starts = (
        prepare_controlled_starts(cases, raw_dir)
        if args.comparison_tier == "controlled"
        else {}
    )
    campaign_start = time.perf_counter()
    campaign_cap = args.campaign_wall_s

    for idx, case in enumerate(cases, start=1):
        elapsed = time.perf_counter() - campaign_start
        if campaign_cap is not None and elapsed >= campaign_cap:
            out_path = raw_dir / case.key / "output.json"
            log_path = raw_dir / case.key / "run.log"
            start = controlled_starts.get(case.benchmark.key)
            command = build_command(
                case,
                out_path,
                state_in=(Path(str(start["path"])) if start is not None else None),
                state_out=(out_path.parent / "final_state.npz" if start is not None else None),
            )
            payload = _write_fallback_payload(
                case=case,
                out_path=out_path,
                command=command,
                result="timeout",
                failure_mode="campaign-wall-time",
                returncode=None,
                wall_time_s=0.0,
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("[runner] skipped: campaign wall time exhausted\n", encoding="utf-8")
            rows.append(
                summarize_payload(
                    mode=args.mode,
                    case=case,
                    payload=payload,
                    json_path=out_path,
                    log_path=log_path,
                    command=command,
                    returncode=None,
                    wall_time_s=0.0,
                    launcher_failure="timeout",
                )
            )
            continue

        timeout_s = float(case.benchmark.wall_cap_s)
        if campaign_cap is not None:
            timeout_s = min(timeout_s, max(1.0, campaign_cap - elapsed))
        print(f"[{idx}/{len(cases)}] {case.key} timeout={timeout_s:.1f}s", flush=True)
        row = run_case(
            case,
            mode=args.mode,
            raw_dir=raw_dir,
            timeout_s=timeout_s,
            canonical_start=controlled_starts.get(case.benchmark.key),
        )
        print(
            f"  -> {row['result']} steps={row['completed_steps']}/{row['steps_requested']} "
            f"newton={row['newton_iters']} krylov={row['krylov_iters']} "
            f"solve={row['solve_time_s'] or '-'} wall={row['wall_time_s']:.3f}s",
            flush=True,
        )
        rows.append(row)

    identity_audit = write_reports(
        rows,
        mode=args.mode,
        comparison_tier=args.comparison_tier,
        report_dir=report_root,
        canonical_starts=controlled_starts,
    )
    print(f"Wrote {report_root / (args.mode + '_summary.csv')}")
    if identity_audit is not None and identity_audit["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
