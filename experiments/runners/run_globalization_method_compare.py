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
import importlib.metadata
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.core.benchmark.run_record import (
    RUN_RECORD_SCHEMA_ID,
    RUN_RECORD_SCHEMA_VERSION,
    ExperimentPreflight,
    ExperimentPreflightError,
    atomic_write_json,
    atomic_write_run_record,
    check_experiment_preflight,
    sha256_file,
    utc_now_iso,
)
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
PROTOCOL_PATH = REPO_ROOT / "paper/protocols/EXP-GLOB-001.md"
CAMPAIGN_ID = "paper-revision-exp-glob-001-local-v1"
CAMPAIGN_SCHEMA_ID = "fenics-nonlinear-energies.exp-glob-001-campaign"
CAMPAIGN_SCHEMA_VERSION = 1
COMMON_START_SCHEMA_ID = "fenics-nonlinear-energies.exp-glob-001-common-starts"
COMMON_START_SCHEMA_VERSION = 2
DEFAULT_TIMING_REPETITIONS = 5
MAX_LOCAL_RANKS = 4

SOURCE_PATHS = (
    Path(__file__).resolve(),
    CASE_RUNNER,
    REPO_ROOT / "src/core/benchmark/state_export.py",
    REPO_ROOT / "src/core/petsc/minimizers.py",
    REPO_ROOT / "src/core/petsc/scalar_problem_driver.py",
    REPO_ROOT / "src/problems/ginzburg_landau/jax/mesh.py",
    REPO_ROOT / "src/problems/hyperelasticity/jax_petsc/solver.py",
    REPO_ROOT / "src/problems/hyperelasticity/support/mesh.py",
)


@dataclass(frozen=True)
class RobustnessInstance:
    key: str
    label: str
    signed_amplitude: float
    amplitude_unit: str
    prescription: str

    def parameters(self) -> dict[str, object]:
        return {
            "signed_amplitude": float(self.signed_amplitude),
            "amplitude_unit": self.amplitude_unit,
            "prescription": self.prescription,
        }


# These are deterministic scientific instances, not random seeds.  The two
# perturbed states are deliberately small and satisfy the constrained DOFs by
# construction.  They support a bounded sensitivity comparison only; they do
# not define a probability distribution and therefore cannot support a general
# robustness claim.
ROBUSTNESS_INSTANCES: tuple[RobustnessInstance, ...] = (
    RobustnessInstance(
        key="nominal",
        label="prescribed nominal start",
        signed_amplitude=0.0,
        amplitude_unit="problem-specific",
        prescription="unperturbed prescribed state",
    ),
    RobustnessInstance(
        key="mode_plus",
        label="positive deterministic mode",
        signed_amplitude=1.0,
        amplitude_unit="mode multiplier",
        prescription="positive closed-form constrained perturbation",
    ),
    RobustnessInstance(
        key="mode_minus",
        label="negative deterministic mode",
        signed_amplitude=-1.0,
        amplitude_unit="mode multiplier",
        prescription="negative closed-form constrained perturbation",
    ),
)
ROBUSTNESS_BY_KEY = {instance.key: instance for instance in ROBUSTNESS_INSTANCES}


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
    robustness_instance: RobustnessInstance | None = None
    timing_repetition: int = 1

    @property
    def key(self) -> str:
        if self.comparison_tier != "controlled":
            return f"{self.benchmark.key}_{self.method.key}"
        if self.robustness_instance is None:
            raise ValueError("controlled case is missing a robustness instance")
        return (
            f"{self.benchmark.key}_{self.robustness_instance.key}_"
            f"{self.method.key}_r{self.timing_repetition:02d}"
        )

    @property
    def start_key(self) -> str:
        if self.robustness_instance is None:
            raise ValueError("case is missing a robustness instance")
        return f"{self.benchmark.key}::{self.robustness_instance.key}"


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
    "robustness_instance",
    "robustness_parameters_json",
    "timing_repetition",
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
    "started_at_utc",
    "finished_at_utc",
    "run_record_path",
    "run_record_sha256",
    "json_path",
    "log_path",
    "command",
)


def build_case_matrix(
    mode: str,
    comparison_tier: str = "production_bundle",
    *,
    robustness_instances: Sequence[RobustnessInstance] | None = None,
    timing_repetitions: int = 1,
) -> list[CaseSpec]:
    if isinstance(timing_repetitions, bool) or int(timing_repetitions) < 1:
        raise ValueError("timing_repetitions must be a positive integer")
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
        instances = tuple(robustness_instances or ROBUSTNESS_INSTANCES)
        if not instances:
            raise ValueError("controlled comparisons require a robustness instance")
    elif comparison_tier == "production_bundle":
        methods = PRODUCTION_BUNDLE_METHODS
        instances = ()
    else:
        raise ValueError(f"unknown comparison tier: {comparison_tier!r}")
    if comparison_tier == "controlled":
        return [
            CaseSpec(
                benchmark=benchmark,
                method=method,
                comparison_tier=comparison_tier,
                robustness_instance=instance,
                timing_repetition=repetition,
            )
            for benchmark in benchmarks
            for instance in instances
            for repetition in range(1, int(timing_repetitions) + 1)
            for method in methods
        ]
    return [
        CaseSpec(benchmark=benchmark, method=method, comparison_tier=comparison_tier)
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


def canonical_start_path(
    raw_dir: Path,
    benchmark: BenchmarkSpec,
    instance: RobustnessInstance,
) -> Path:
    return raw_dir / "_canonical_starts" / benchmark.key / f"{instance.key}.npz"


def _instance_parameters(
    benchmark: BenchmarkSpec,
    instance: RobustnessInstance,
) -> dict[str, object]:
    if benchmark.problem == "gl":
        return {
            "instance_id": instance.key,
            "mode": "sin(pi*(x+1)/2)*sin(pi*(y+1)/2)*sin(pi*(x+1))*sin(pi*(y+1))",
            "signed_amplitude": float(0.025 * instance.signed_amplitude),
            "amplitude_unit": "dimensionless order-parameter value",
            "constraint_policy": "mode added only at solver free nodes",
        }
    if benchmark.problem == "he":
        return {
            "instance_id": instance.key,
            "mode": "[0, sin(pi*x/L), -0.5*sin(pi*x/L)]",
            "signed_amplitude": float(1.0e-5 * instance.signed_amplitude),
            "amplitude_unit": "m",
            "constraint_policy": "mode added only at constrained-system free components",
        }
    raise ValueError(
        f"No deterministic robustness-instance prescription for {benchmark.problem!r}"
    )


def prepare_canonical_start(
    benchmark: BenchmarkSpec,
    instance: RobustnessInstance,
    destination: Path,
) -> dict[str, Any]:
    """Create one immutable instance start shared by both methods and all repeats."""

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
        coords = np.asarray(params["nodes"], dtype=np.float64)
        mode = (
            np.sin(np.pi * (coords[:, 0] + 1.0) / 2.0)
            * np.sin(np.pi * (coords[:, 1] + 1.0) / 2.0)
            * np.sin(np.pi * (coords[:, 0] + 1.0))
            * np.sin(np.pi * (coords[:, 1] + 1.0))
        )
        instance_parameters = _instance_parameters(benchmark, instance)
        free = np.asarray(params["freedofs"], dtype=np.int64)
        full_state[free] += float(instance_parameters["signed_amplitude"]) * mode[free]
        state_sha256 = _array_sha256(full_state)
        export_scalar_mesh_state_npz(
            destination,
            coords=coords,
            triangles=np.asarray(params["elems"], dtype=np.int32),
            u=full_state,
            mesh_level=int(benchmark.level),
            problem_name="GinzburgLandau2D",
            metadata={
                "artifact_role": "EXP-GLOB-001 canonical initial state",
                "state_sha256": state_sha256,
                "initializer": "sin(pi*(x-1)/2)*sin(pi*(y-1)/2) on free nodes",
                "robustness_instance": instance.key,
                "robustness_parameters_json": json.dumps(
                    instance_parameters, sort_keys=True, separators=(",", ":")
                ),
            },
        )
    elif benchmark.problem == "he":
        from src.problems.hyperelasticity.support.mesh import (
            build_procedural_hyperelasticity_export_params,
        )

        params = build_procedural_hyperelasticity_export_params(int(benchmark.level))
        coords_ref = np.asarray(params["nodes2coord"], dtype=np.float64)
        full_state = np.asarray(params["u_0_ref"], dtype=np.float64).reshape((-1, 3))
        instance_parameters = _instance_parameters(benchmark, instance)
        beam_length = float(np.max(coords_ref[:, 0]) - np.min(coords_ref[:, 0]))
        if beam_length <= 0.0:
            raise ValueError("Hyperelasticity robustness mode requires positive beam length")
        longitudinal = np.sin(
            np.pi * (coords_ref[:, 0] - np.min(coords_ref[:, 0])) / beam_length
        )
        displacement = np.column_stack(
            [np.zeros_like(longitudinal), longitudinal, -0.5 * longitudinal]
        ) * float(instance_parameters["signed_amplitude"])
        flattened = full_state.reshape(-1)
        displacement_flat = displacement.reshape(-1)
        free = np.asarray(params["freedofs"], dtype=np.int64)
        flattened[free] += displacement_flat[free]
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
                "robustness_instance": instance.key,
                "robustness_parameters_json": json.dumps(
                    instance_parameters, sort_keys=True, separators=(",", ":")
                ),
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
        "robustness_instance": instance.key,
        "robustness_parameters": instance_parameters,
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
        instance = case.robustness_instance
        if instance is None:
            raise ValueError("controlled case is missing its robustness instance")
        if case.start_key in starts:
            continue
        starts[case.start_key] = prepare_canonical_start(
            benchmark,
            instance,
            canonical_start_path(raw_dir, benchmark, instance),
        )
    atomic_write_json(
        raw_dir / "_canonical_starts" / "manifest.json",
        {
            "schema_id": COMMON_START_SCHEMA_ID,
            "schema_version": COMMON_START_SCHEMA_VERSION,
            "status": "prepared",
            "created_at_utc": utc_now_iso(),
            "instances": starts,
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


def _json_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _source_hashes() -> dict[str, str]:
    missing = [path for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Globalization source-hash inventory is incomplete: "
            + ", ".join(_display_path(path) for path in missing)
        )
    return {_display_path(path): sha256_file(path) for path in SOURCE_PATHS}


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("jax", "jaxlib", "mpi4py", "numpy", "petsc4py", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _first_matching_line(path: Path, prefix: str) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[-1].strip() or "unknown"
    except OSError:
        pass
    return "unknown"


def publication_child_environment() -> dict[str, str]:
    """Return the exact low-noise environment inherited by every child run."""

    environment = os.environ.copy()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    environment["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    return environment


def capture_environment(environment: Mapping[str, str]) -> dict[str, object]:
    try:
        from mpi4py import MPI

        mpi_version = MPI.Get_library_version().strip().replace("\n", " ")
    except Exception as exc:  # pragma: no cover - only for incomplete environments
        mpi_version = f"unavailable ({type(exc).__name__})"
    try:
        from petsc4py import PETSc

        petsc_version = ".".join(str(value) for value in PETSc.Sys.getVersion())
    except Exception as exc:  # pragma: no cover - only for incomplete environments
        petsc_version = f"unavailable ({type(exc).__name__})"
    try:
        import jax

        jax_x64: bool | None = bool(jax.config.x64_enabled)
    except Exception:  # pragma: no cover - only for incomplete environments
        jax_x64 = None
    packages = _package_versions()
    affinity = (
        ",".join(str(value) for value in sorted(os.sched_getaffinity(0)))
        if hasattr(os, "sched_getaffinity")
        else "not available"
    )
    controlled = {
        key: str(environment[key])
        for key in (
            "JAX_PLATFORMS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "XLA_FLAGS",
        )
    }
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "platform": platform.platform(),
        "jax": packages["jax"],
        "xla": packages["jaxlib"],
        "jax_enable_x64": jax_x64,
        "petsc": petsc_version,
        "mpi": mpi_version,
        "compiler": platform.python_compiler() or "unknown Python compiler",
        "blas": "NumPy runtime; provider retained in the installed environment",
        "cpu_model": _first_matching_line(Path("/proc/cpuinfo"), "model name"),
        "node_model": platform.node() or "local host",
        "memory_model": _first_matching_line(Path("/proc/meminfo"), "MemTotal"),
        "scheduler": "local",
        "scheduler_job_id": None,
        "affinity": affinity,
        "controlled_environment": controlled,
    }


def require_publication_preflight() -> ExperimentPreflight:
    """Require one clean SHA-1 HEAD before any campaign path is created."""

    preflight = check_experiment_preflight(REPO_ROOT, run_kind="publication")
    if re.fullmatch(r"[0-9a-f]{40}", preflight.git_commit) is None:
        raise ExperimentPreflightError(
            "EXP-GLOB-001 requires one exact 40-character Git commit; "
            f"got {preflight.git_commit!r}"
        )
    return preflight


def assert_same_clean_commit(preflight: ExperimentPreflight) -> None:
    current = require_publication_preflight()
    if current.git_commit != preflight.git_commit:
        raise ExperimentPreflightError(
            "Git HEAD changed during EXP-GLOB-001; refusing a mixed-commit campaign"
        )


def require_fresh_campaign_paths(raw_dir: Path, report_dir: Path) -> None:
    existing = [path for path in (raw_dir, report_dir) if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "Publication campaign paths must be new; refusing to mix or overwrite: "
            + ", ".join(str(path.resolve()) for path in existing)
        )


def _case_configuration(case: CaseSpec, mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "comparison_tier": case.comparison_tier,
        "benchmark": {
            "key": case.benchmark.key,
            "problem": case.benchmark.problem,
            "backend": case.benchmark.backend,
            "level": int(case.benchmark.level),
            "ranks": int(case.benchmark.nprocs),
            "steps": int(case.benchmark.steps),
            "start_step": int(case.benchmark.start_step),
            "total_steps": int(case.benchmark.total_steps),
            "wall_cap_s": float(case.benchmark.wall_cap_s),
            "common_args": list(case.benchmark.common_args),
        },
        "method": {
            "key": case.method.key,
            "args": list(case.method.args),
        },
        "robustness_instance": (
            _instance_parameters(case.benchmark, case.robustness_instance)
            if case.robustness_instance is not None
            else None
        ),
        "timing_repetition": int(case.timing_repetition),
    }


def _input_hashes(case: CaseSpec, canonical_start: Mapping[str, Any] | None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if canonical_start is not None:
        hashes[_display_path(Path(str(canonical_start["path"])))] = str(
            canonical_start["file_sha256"]
        )
    if case.benchmark.problem == "gl":
        mesh_path = (
            REPO_ROOT
            / "data/meshes/GinzburgLandau"
            / f"GL_level{case.benchmark.level}.h5"
        )
        if not mesh_path.is_file():
            raise FileNotFoundError(f"missing retained GL input {_display_path(mesh_path)}")
        hashes[_display_path(mesh_path)] = sha256_file(mesh_path)
    return dict(sorted(hashes.items()))


def _command_for_case(
    case: CaseSpec,
    raw_dir: Path,
    canonical_start: Mapping[str, Any] | None,
) -> list[str]:
    out_path = raw_dir / case.key / "output.json"
    state_in = (
        Path(str(canonical_start["path"])) if canonical_start is not None else None
    )
    state_out = out_path.parent / "final_state.npz" if canonical_start is not None else None
    return build_command(case, out_path, state_in=state_in, state_out=state_out)


def _termination_fields(row: Mapping[str, Any], case: CaseSpec) -> dict[str, object]:
    failure = str(row.get("failure_mode") or row.get("result") or "unknown")
    row_result = str(row.get("result"))
    if row_result == "timeout":
        return {
            "status": "timeout",
            "reason": failure,
            "limit_kind": "wall_time_s",
            "limit_value": float(case.benchmark.wall_cap_s),
            "censored": True,
        }
    if row_result == "failed" and re.search(
        r"max(?:imum)?[- ]?it|iteration cap|incomplete", failure, re.IGNORECASE
    ):
        return {
            "status": "capped",
            "reason": failure,
            "limit_kind": "nonlinear_iterations",
            "limit_value": 100,
            "censored": True,
        }
    identity_fields = (
        "final_state_file_sha256",
        "final_state_content_sha256",
        "endpoint_state_sha256",
        "independent_residual_sha256",
    )
    residual = _safe_float(row.get("independent_dual_residual"))
    identity_complete = all(row.get(field) for field in identity_fields) and (
        residual is not None and np.isfinite(residual)
    )
    if row_result == "completed" and identity_complete:
        return {
            "status": "success",
            "reason": "solver completed and terminal identity gate is complete",
            "limit_kind": None,
            "limit_value": None,
            "censored": False,
        }
    if row_result == "completed":
        failure = "solver reported completion but terminal identity gate is incomplete"
    return {
        "status": "failure",
        "reason": failure,
        "limit_kind": None,
        "limit_value": None,
        "censored": False,
    }


def build_publication_run_record(
    *,
    case: CaseSpec,
    mode: str,
    row: Mapping[str, Any],
    command: Sequence[str],
    preflight: ExperimentPreflight,
    environment: Mapping[str, object],
    source_hashes: Mapping[str, str],
    campaign_configuration_sha256: str,
    canonical_start: Mapping[str, Any] | None,
    raw_dir: Path,
) -> dict[str, object]:
    termination = _termination_fields(row, case)
    success = termination["status"] == "success"
    output_path = raw_dir / case.key / "output.json"
    log_path = raw_dir / case.key / "run.log"
    final_state = raw_dir / case.key / "final_state.npz"
    input_hashes = _input_hashes(case, canonical_start)
    case_configuration = _case_configuration(case, mode)
    dual = _safe_float(row.get("independent_dual_residual"))
    coefficient = _safe_float(row.get("independent_coefficient_residual"))
    returncode = row.get("returncode")
    returncode_value = None if returncode in (None, "") else int(returncode)
    states = []
    if canonical_start is not None:
        states.append(_display_path(Path(str(canonical_start["path"]))))
    if final_state.is_file():
        states.append(_display_path(final_state))
    return {
        "schema": {"id": RUN_RECORD_SCHEMA_ID, "version": RUN_RECORD_SCHEMA_VERSION},
        "record_id": f"{CAMPAIGN_ID}-{case.key}",
        "run_kind": "publication",
        "identifiers": {
            "campaign": CAMPAIGN_ID,
            "experiment": "EXP-GLOB-001",
            "case": f"{case.benchmark.key}-{row['robustness_instance']}",
            "method": case.method.key,
            "route": "controlled-local-common-start",
            "repetition": int(case.timing_repetition),
        },
        "problem": {
            "name": case.benchmark.label,
            "mesh": f"maintained level {case.benchmark.level}",
            "degree": 1,
            "quadrature": "maintained element functional quadrature",
            "total_degrees_of_freedom": None,
            "free_degrees_of_freedom": None,
            "notes": (
                "One deterministic robustness instance; timing repetition is not "
                "an independent robustness unit."
            ),
        },
        "solver": {
            "algorithm": case.method.label,
            "implementation": _display_path(CASE_RUNNER),
            "parameters": case_configuration,
            "preconditioner": {
                "type": (
                    case.benchmark.common_args[
                        case.benchmark.common_args.index("--pc-type") + 1
                    ]
                    if "--pc-type" in case.benchmark.common_args
                    else "unspecified"
                )
            },
            "stopping_contract": "maintained Riesz-scaled nonlinear convergence contract",
        },
        "termination": {
            **termination,
            "exit_code": returncode_value,
            "started_at_utc": str(row["started_at_utc"]),
            "finished_at_utc": str(row["finished_at_utc"]),
        },
        "accuracy": {
            "contract_id": "EXP-GLOB-001-terminal-identity-v1",
            "gate_passed": bool(success),
            "absolute_residual": dual,
            "relative_residual": None,
            "scaled_residual": None,
            "relative_correction": None,
            "energy_change": None,
            "custom_metrics": {
                "coefficient_residual": coefficient,
                "endpoint_state_sha256": str(row.get("endpoint_state_sha256") or ""),
                "final_state_content_sha256": str(
                    row.get("final_state_content_sha256") or ""
                ),
            },
            "notes": "Residual was independently reevaluated at the exported terminal state.",
        },
        "counts": {
            "nonlinear_iterations": int(row.get("newton_iters") or 0),
            "krylov_iterations": int(row.get("krylov_iters") or 0),
            "function_evaluations": None,
            "gradient_evaluations": None,
            "hessian_evaluations": None,
            "hvp_evaluations": None,
            "preconditioner_setups": None,
            "notes": "Unavailable evaluation counters are null; nonlinear and Krylov counts are retained.",
        },
        "timing": {
            "aggregation": "single cold process launch; machine-noise aggregation occurs across repetition records",
            "cold_process": True,
            "barrier_policy": "solver-internal MPI synchronization only",
            "synchronization_policy": "solver synchronizes distributed operations before terminal output",
            "phases_overlap": False,
            "relation_to_total": "solver phase times are nested in process wall time",
            "process_startup_s": None,
            "jit_compilation_s": None,
            "coloring_s": None,
            "derivative_evaluation_s": None,
            "constitutive_contraction_s": None,
            "assembly_s": None,
            "communication_s": None,
            "preconditioner_setup_s": _safe_float(row.get("setup_time_s")),
            "krylov_solve_s": _safe_float(row.get("solve_time_s")),
            "globalization_s": _safe_float(row.get("line_search_time_s")),
            "state_output_s": None,
            "total_s": float(row["wall_time_s"]),
            "notes": "Timing is descriptive until all common-start, endpoint, repetition, and environment gates pass.",
        },
        "resources": {
            "nodes": 1,
            "ranks": int(case.benchmark.nprocs),
            "threads_per_rank": 1,
            "peak_memory_per_rank_bytes": None,
            "peak_memory_per_node_bytes": None,
            "tracked_allocations_bytes": None,
            "measurement_method": "memory not instrumented by this campaign",
            "notes": "Local low-rank execution only; no scaling claim.",
        },
        "diagnostics": {
            "state": {
                "robustness_instance": str(row["robustness_instance"]),
                "robustness_parameters": json.loads(
                    str(row["robustness_parameters_json"])
                ),
                "initial_file_sha256": str(row.get("initial_state_file_sha256") or ""),
                "initial_content_sha256": str(
                    row.get("initial_state_content_sha256") or ""
                ),
                "final_file_sha256": str(row.get("final_state_file_sha256") or ""),
                "final_content_sha256": str(
                    row.get("final_state_content_sha256") or ""
                ),
            },
            "branch": {},
            "feasibility": {"terminal_identity_complete": bool(success)},
            "kkt": {
                "independent_residual_sha256": str(
                    row.get("independent_residual_sha256") or ""
                )
            },
        },
        "environment": dict(environment),
        "provenance": {
            **preflight.provenance_fields(),
            "command_argv": [str(value) for value in command],
            "working_directory": str(REPO_ROOT),
            "code_hashes": dict(sorted(source_hashes.items())),
            "configuration_hashes": {
                _display_path(PROTOCOL_PATH): sha256_file(PROTOCOL_PATH),
                "campaign_configuration": campaign_configuration_sha256,
                "case_configuration": _json_sha256(case_configuration),
            },
            "input_hashes": input_hashes,
            "dirty_patch_sha256": None,
            "seed": None,
            "deterministic_policy": (
                "Closed-form instance state, immutable canonical NPZ, FP64 solver, "
                "and fixed single-thread CPU child environment."
            ),
            "recorded_at_utc": utc_now_iso(),
            "preflight_checked_at_utc": preflight.checked_at_utc,
        },
        "artifacts": {
            "raw_outputs": [_display_path(output_path)],
            "states": states,
            "logs": [_display_path(log_path)],
            "tables": [],
            "figures": [],
            "reports": [],
        },
    }


def campaign_configuration(
    *,
    mode: str,
    comparison_tier: str,
    cases: Sequence[CaseSpec],
    child_environment: Mapping[str, str],
) -> dict[str, object]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "mode": mode,
        "comparison_tier": comparison_tier,
        "maximum_local_ranks": MAX_LOCAL_RANKS,
        "machine_noise_repetitions": max(
            (case.timing_repetition for case in cases), default=1
        ),
        "robustness_instances": [
            {
                "id": instance.key,
                "label": instance.label,
                "generic_parameters": instance.parameters(),
            }
            for instance in ROBUSTNESS_INSTANCES
            if any(case.robustness_instance == instance for case in cases)
        ],
        "controlled_child_environment": {
            key: child_environment[key]
            for key in (
                "JAX_PLATFORMS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "cases": [_case_configuration(case, mode) for case in cases],
    }


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
    started_at_utc: str = "1970-01-01T00:00:00Z",
    finished_at_utc: str = "1970-01-01T00:00:00Z",
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
        "robustness_instance": (
            case.robustness_instance.key
            if case.robustness_instance is not None
            else "not-applicable"
        ),
        "robustness_parameters_json": (
            json.dumps(
                _instance_parameters(case.benchmark, case.robustness_instance),
                sort_keys=True,
                separators=(",", ":"),
            )
            if case.robustness_instance is not None
            else "{}"
        ),
        "timing_repetition": int(case.timing_repetition),
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
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "run_record_path": "",
        "run_record_sha256": "",
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
    child_environment: Mapping[str, str] | None = None,
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

    started_at_utc = utc_now_iso()
    start = time.perf_counter()
    returncode: int | None = None
    launcher_failure: str | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=dict(child_environment) if child_environment is not None else None,
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
    finished_at_utc = utc_now_iso()

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
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        launcher_failure=launcher_failure,
    )


def controlled_identity_audit(
    rows: list[dict[str, Any]],
    canonical_starts: dict[str, dict[str, Any]] | None = None,
    *,
    expected_repetitions: int | None = None,
    expected_instances: Sequence[str] | None = None,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        unit = (
            str(row["benchmark"]),
            str(row.get("robustness_instance") or "nominal"),
            int(row.get("timing_repetition") or 1),
        )
        groups.setdefault(unit, []).append(row)
    errors: list[str] = []
    claim_limitations: list[str] = []
    summaries: list[dict[str, Any]] = []
    required_methods = {method.key for method in CONTROLLED_METHODS}
    per_instance_starts: dict[tuple[str, str], set[tuple[str, str]]] = {}
    all_endpoint_equal = True
    all_completed = True
    for (benchmark, instance, repetition), group in sorted(groups.items()):
        unit_label = f"{benchmark}/{instance}/r{repetition:02d}"
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
            errors.append(f"{unit_label}: controlled method set is incomplete")
        if len(start_files) != 1 or len(start_states) != 1:
            errors.append(f"{unit_label}: methods do not prove one common hashed start")
        per_instance_starts.setdefault((benchmark, instance), set()).update(
            zip(start_files, start_states)
        )
        expected_start = (canonical_starts or {}).get(f"{benchmark}::{instance}")
        if expected_start is None:
            # Compatibility for v1 callers that keyed nominal starts only by benchmark.
            expected_start = (canonical_starts or {}).get(benchmark)
        if expected_start is not None and (
            start_files != {str(expected_start["file_sha256"])}
            or start_states != {str(expected_start["state_sha256"])}
        ):
            errors.append(
                f"{unit_label}: reported common start differs from the canonical manifest"
            )
        endpoint_hashes = {
            str(row["final_state_content_sha256"])
            for row in group
            if row.get("final_state_content_sha256")
        }
        row_summaries: list[dict[str, Any]] = []
        for row in group:
            completed = row.get("result") == "completed"
            all_completed = all_completed and completed
            if not completed:
                errors.append(
                    f"{unit_label}/{row['method']}: solver did not complete successfully"
                )
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
                    f"{unit_label}/{row['method']}: terminal state or independent residual identity is incomplete"
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
                "robustness_instance": instance,
                "timing_repetition": repetition,
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
        all_endpoint_equal = all_endpoint_equal and len(endpoint_hashes) == 1

    for (benchmark, instance), identities in sorted(per_instance_starts.items()):
        if len(identities) != 1:
            errors.append(
                f"{benchmark}/{instance}: timing repetitions do not reuse one immutable canonical start"
            )

    observed_instances = sorted(
        {str(row.get("robustness_instance") or "nominal") for row in rows}
    )
    repetitions_by_instance: dict[str, list[int]] = {}
    for instance in observed_instances:
        repetitions_by_instance[instance] = sorted(
            {
                int(row.get("timing_repetition") or 1)
                for row in rows
                if str(row.get("robustness_instance") or "nominal") == instance
            }
        )
    expected_instance_set = set(expected_instances or observed_instances)
    if set(observed_instances) != expected_instance_set:
        errors.append(
            "controlled instance set differs from the frozen campaign configuration"
        )
    repetition_target = int(expected_repetitions or 1)
    complete_repetition_grid = all(
        repetitions == list(range(1, repetition_target + 1))
        for repetitions in repetitions_by_instance.values()
    )
    timing_admissible = bool(
        not errors
        and all_completed
        and all_endpoint_equal
        and repetition_target >= DEFAULT_TIMING_REPETITIONS
        and complete_repetition_grid
    )
    if repetition_target < DEFAULT_TIMING_REPETITIONS:
        claim_limitations.append(
            f"timing requires at least {DEFAULT_TIMING_REPETITIONS} machine-noise repetitions"
        )
    if not all_endpoint_equal:
        claim_limitations.append(
            "methods reached different or incomplete exact endpoint-identity classes"
        )
    claim_limitations.append(
        "deterministic prescribed instances are a bounded sensitivity set, not a sampled population"
    )
    return {
        "schema_id": "fenics-nonlinear-energies.exp-glob-001-identity-audit",
        "schema_version": 2,
        "status": "passed" if not errors else "failed",
        "timing_claim_admissible": timing_admissible,
        "tested_instance_comparison_admissible": bool(
            not errors and all_completed and all_endpoint_equal and len(observed_instances) >= 2
        ),
        "robustness_generalization_claim_admissible": False,
        "robustness_generalization_refusal": (
            "The three closed-form starts are deterministic sensitivity instances, "
            "not independent samples from a declared target population."
        ),
        "machine_noise_repetitions_required": DEFAULT_TIMING_REPETITIONS,
        "machine_noise_repetitions_observed": repetitions_by_instance,
        "robustness_instances_observed": observed_instances,
        "claim_limitations": claim_limitations,
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
    expected_repetitions: int | None = None,
    expected_instances: Sequence[str] | None = None,
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
        identity_audit = controlled_identity_audit(
            rows,
            canonical_starts,
            expected_repetitions=expected_repetitions,
            expected_instances=expected_instances,
        )
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
    parser.add_argument(
        "--timing-repetitions",
        "--repetitions",
        dest="timing_repetitions",
        type=int,
        default=DEFAULT_TIMING_REPETITIONS,
        help=(
            "cold-process machine-noise repetitions per method and deterministic "
            f"instance (default: {DEFAULT_TIMING_REPETITIONS})"
        ),
    )
    parser.add_argument(
        "--instance",
        action="append",
        choices=tuple(ROBUSTNESS_BY_KEY),
        help="controlled robustness instance; repeat to select a subset (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.timing_repetitions) < 1:
        raise SystemExit("--timing-repetitions must be positive")
    selected_instances = tuple(
        ROBUSTNESS_BY_KEY[key]
        for key in (args.instance or tuple(ROBUSTNESS_BY_KEY))
    )
    if len({instance.key for instance in selected_instances}) != len(selected_instances):
        raise SystemExit("--instance values must be unique")
    cases = build_case_matrix(
        args.mode,
        args.comparison_tier,
        robustness_instances=selected_instances,
        timing_repetitions=(
            int(args.timing_repetitions)
            if args.comparison_tier == "controlled"
            else 1
        ),
    )
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
        print(
            f"# {len(cases)} planned launches; no files created; "
            f"timing repetitions={args.timing_repetitions}"
        )
        for case in cases:
            out_path = raw_root / args.mode / case.key / "output.json"
            state_in = (
                canonical_start_path(
                    raw_root / args.mode,
                    case.benchmark,
                    case.robustness_instance,
                )
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

    if args.comparison_tier != "controlled":
        raise SystemExit(
            "Publication execution in this runner is restricted to --comparison-tier controlled; "
            "production bundles remain preparation-only here."
        )
    preflight = require_publication_preflight()
    if any(case.benchmark.nprocs > MAX_LOCAL_RANKS for case in cases):
        raise SystemExit(
            f"Local EXP-GLOB-001 refuses ranks above {MAX_LOCAL_RANKS}; "
            "use --mode smoke or prepare a separately reviewed cluster campaign."
        )
    require_generated_inputs(args.mode)

    rows: list[dict[str, Any]] = []
    raw_dir = raw_root / args.mode
    report_dir = report_root
    require_fresh_campaign_paths(raw_dir, report_dir)
    child_environment = publication_child_environment()
    environment = capture_environment(child_environment)
    source_hashes = _source_hashes()
    configuration = campaign_configuration(
        mode=args.mode,
        comparison_tier=args.comparison_tier,
        cases=cases,
        child_environment=child_environment,
    )
    configuration_sha256 = _json_sha256(configuration)
    controlled_starts = prepare_controlled_starts(cases, raw_dir)
    common_start_manifest = raw_dir / "_canonical_starts" / "manifest.json"
    started_at_utc = utc_now_iso()
    planned_runs: list[dict[str, object]] = []
    for case in cases:
        start = controlled_starts[case.start_key]
        command = _command_for_case(case, raw_dir, start)
        planned_runs.append(
            {
                "case_id": case.key,
                "benchmark": case.benchmark.key,
                "method": case.method.key,
                "robustness_instance": case.robustness_instance.key,
                "timing_repetition": int(case.timing_repetition),
                "command_argv": command,
                "command_sha256": _json_sha256(command),
                "input_hashes": _input_hashes(case, start),
            }
        )
    campaign_manifest_path = raw_dir / "campaign_manifest.json"
    campaign_manifest: dict[str, object] = {
        "schema": {"id": CAMPAIGN_SCHEMA_ID, "version": CAMPAIGN_SCHEMA_VERSION},
        "campaign_id": CAMPAIGN_ID,
        "status": "running",
        "started_at_utc": started_at_utc,
        "finished_at_utc": None,
        "publication_preflight": {
            **preflight.provenance_fields(),
            "checked_at_utc": preflight.checked_at_utc,
        },
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "source_hashes": source_hashes,
        "protocol_hashes": {_display_path(PROTOCOL_PATH): sha256_file(PROTOCOL_PATH)},
        "environment": environment,
        "common_start_manifest": {
            "path": _display_path(common_start_manifest),
            "sha256": sha256_file(common_start_manifest),
        },
        "planned_runs": planned_runs,
        "run_records": [],
        "reports": {},
        "claim_admission": {
            "timing_claim_admissible": False,
            "tested_instance_comparison_admissible": False,
            "robustness_generalization_claim_admissible": False,
            "reason": "campaign has not completed",
        },
    }
    atomic_write_json(campaign_manifest_path, campaign_manifest)
    campaign_start = time.perf_counter()
    campaign_cap = args.campaign_wall_s

    for idx, case in enumerate(cases, start=1):
        assert_same_clean_commit(preflight)
        elapsed = time.perf_counter() - campaign_start
        start_identity = controlled_starts[case.start_key]
        command = _command_for_case(case, raw_dir, start_identity)
        if campaign_cap is not None and elapsed >= campaign_cap:
            out_path = raw_dir / case.key / "output.json"
            log_path = raw_dir / case.key / "run.log"
            skipped_at_utc = utc_now_iso()
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
                    started_at_utc=skipped_at_utc,
                    finished_at_utc=skipped_at_utc,
                    launcher_failure="timeout",
                )
            )
            row = rows[-1]
        else:
            timeout_s = float(case.benchmark.wall_cap_s)
            if campaign_cap is not None:
                timeout_s = min(timeout_s, max(1.0, campaign_cap - elapsed))
            print(f"[{idx}/{len(cases)}] {case.key} timeout={timeout_s:.1f}s", flush=True)
            row = run_case(
                case,
                mode=args.mode,
                raw_dir=raw_dir,
                timeout_s=timeout_s,
                canonical_start=start_identity,
                child_environment=child_environment,
            )
            rows.append(row)
            print(
                f"  -> {row['result']} steps={row['completed_steps']}/{row['steps_requested']} "
                f"newton={row['newton_iters']} krylov={row['krylov_iters']} "
                f"solve={row['solve_time_s'] or '-'} wall={row['wall_time_s']:.3f}s",
                flush=True,
            )
        run_record = build_publication_run_record(
            case=case,
            mode=args.mode,
            row=row,
            command=command,
            preflight=preflight,
            environment=environment,
            source_hashes=source_hashes,
            campaign_configuration_sha256=configuration_sha256,
            canonical_start=start_identity,
            raw_dir=raw_dir,
        )
        run_record_path = raw_dir / case.key / "run_record.json"
        atomic_write_run_record(
            run_record_path, run_record, require_publication_ready=True
        )
        row["run_record_path"] = _display_path(run_record_path)
        row["run_record_sha256"] = sha256_file(run_record_path)

    identity_audit = write_reports(
        rows,
        mode=args.mode,
        comparison_tier=args.comparison_tier,
        report_dir=report_dir,
        canonical_starts=controlled_starts,
        expected_repetitions=int(args.timing_repetitions),
        expected_instances=[instance.key for instance in selected_instances],
    )
    assert_same_clean_commit(preflight)
    report_paths = (
        report_dir / f"{args.mode}_summary.csv",
        report_dir / f"{args.mode}_summary.json",
        report_dir / f"{args.mode}_identity_audit.json",
    )
    campaign_manifest.update(
        {
            "status": (
                "completed"
                if identity_audit is not None and identity_audit["status"] == "passed"
                else "completed_with_failed_identity_gate"
            ),
            "finished_at_utc": utc_now_iso(),
            "run_records": [
                {
                    "path": str(row["run_record_path"]),
                    "sha256": str(row["run_record_sha256"]),
                }
                for row in rows
            ],
            "reports": {
                _display_path(path): sha256_file(path)
                for path in report_paths
                if path.is_file()
            },
            "claim_admission": {
                "timing_claim_admissible": bool(
                    identity_audit and identity_audit["timing_claim_admissible"]
                ),
                "tested_instance_comparison_admissible": bool(
                    identity_audit
                    and identity_audit["tested_instance_comparison_admissible"]
                ),
                "robustness_generalization_claim_admissible": False,
                "reason": (
                    identity_audit["robustness_generalization_refusal"]
                    if identity_audit
                    else "controlled identity audit was not produced"
                ),
            },
        }
    )
    atomic_write_json(campaign_manifest_path, campaign_manifest)
    print(f"Wrote {report_dir / (args.mode + '_summary.csv')}")
    print(f"Wrote {campaign_manifest_path}")
    if identity_audit is not None and identity_audit["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
