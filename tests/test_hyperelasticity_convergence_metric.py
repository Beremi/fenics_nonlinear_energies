from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from petsc4py import PETSc
import pytest

from experiments.runners import run_trust_region_case
from src.core.petsc.metrics import MatrixRieszMetric
from src.problems.hyperelasticity.jax_petsc.solve_HE_dof import _build_parser
from src.problems.hyperelasticity.jax_petsc.solver import (
    PROFILE_DEFAULTS,
    _build_certified_reference_elastic_metric,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CASE_RUNNER = REPO_ROOT / "experiments/runners/run_trust_region_case.py"


def _matrix(values: np.ndarray) -> PETSc.Mat:
    values = np.asarray(values, dtype=np.float64)
    matrix = PETSc.Mat().createAIJ(values.shape, comm=PETSc.COMM_SELF)
    matrix.setUp()
    rows, columns = np.nonzero(values)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        matrix.setValue(int(row), int(column), float(values[row, column]))
    matrix.assemble()
    return matrix


def _vector(values: list[float]) -> PETSc.Vec:
    vector = PETSc.Vec().createSeq(len(values), comm=PETSc.COMM_SELF)
    vector.setArray(np.asarray(values, dtype=np.float64))
    return vector


def test_direct_and_case_parsers_preserve_legacy_default_and_expose_riesz_controls() -> None:
    direct = _build_parser(PROFILE_DEFAULTS)
    legacy = direct.parse_args([])
    assert legacy.convergence_metric == "coefficient_l2"
    assert legacy.convergence_state_scale is None
    assert legacy.riesz_ksp_max_it == 5000

    configured = direct.parse_args(
        [
            "--convergence-metric",
            "reference_elastic_energy",
            "--convergence-state-scale",
            "2.5",
            "--riesz-ksp-rtol",
            "1e-11",
            "--riesz-true-residual-rtol",
            "1e-9",
            "--riesz-symmetry-tol",
            "1e-13",
        ]
    )
    assert configured.convergence_metric == "reference_elastic_energy"
    assert configured.convergence_state_scale == pytest.approx(2.5)
    assert configured.riesz_ksp_rtol == pytest.approx(1.0e-11)
    assert configured.riesz_true_residual_rtol == pytest.approx(1.0e-9)

    case = run_trust_region_case._build_parser().parse_args(
        [
            "--problem",
            "he",
            "--backend",
            "element",
            "--level",
            "1",
            "--out",
            "unused.json",
            "--convergence-metric",
            "reference_elastic_energy",
            "--riesz-ksp-max-it",
            "700",
        ]
    )
    assert case.convergence_metric == "reference_elastic_energy"
    assert case.riesz_ksp_max_it == 700

    case.backend = "fenics"
    with pytest.raises(ValueError, match="certified only"):
        run_trust_region_case._run_he(case)


def test_reference_metric_uses_initial_deformation_map_norm_as_state_scale() -> None:
    operator = _matrix(np.diag([2.0, 8.0]))
    initial_state = _vector([1.0, 2.0])
    output = _vector([0.0, 0.0])
    metric = None
    args = SimpleNamespace(
        convergence_state_scale=None,
        riesz_ksp_type="cg",
        riesz_pc_type="jacobi",
        riesz_ksp_rtol=1.0e-10,
        riesz_ksp_atol=1.0e-14,
        riesz_ksp_max_it=100,
        riesz_true_residual_rtol=1.0e-8,
        riesz_spd_factor_solver_type="mumps",
        riesz_symmetry_tol=1.0e-12,
    )
    try:
        metric, state_scale, configuration = (
            _build_certified_reference_elastic_metric(
                args=args,
                operator=operator,
                initial_state=initial_state,
                expected_free_dofs=2,
                provenance={"unit_test": True},
            )
        )
        assert isinstance(metric, MatrixRieszMetric)
        assert state_scale == pytest.approx(np.sqrt(34.0))
        assert configuration["state_variable"] == (
            "deformation_map_y_on_constrained_free_dofs"
        )
        assert configuration["state_scale_source"] == (
            "initial_reference_deformation_map_primal_norm"
        )
        assert configuration["metric"]["requested_norm_type"] == "unpreconditioned"
        assert configuration["metric"]["effective_norm_type"] == "unpreconditioned"
        certificate = configuration["metric"]["provenance"]["spd_certificate"]
        assert certificate["certified_spd"] is True
        assert certificate["inertia"] == {"negative": 0, "zero": 0, "positive": 2}

        metric.destroy()
        metric = None
        operator.mult(initial_state, output)
        np.testing.assert_allclose(output.getArray(readonly=True), [2.0, 16.0])
    finally:
        if metric is not None:
            metric.destroy()
        output.destroy()
        initial_state.destroy()
        operator.destroy()


@pytest.mark.parametrize("mpi_ranks", [1, 2])
def test_reference_elastic_metric_p1_mpi_smoke(
    tmp_path: Path,
    mpi_ranks: int,
) -> None:
    output_path = tmp_path / f"he_reference_metric_np{mpi_ranks}.json"
    command = [
        "mpiexec",
        "-n",
        str(mpi_ranks),
        str(PYTHON),
        "-u",
        str(CASE_RUNNER),
        "--problem",
        "he",
        "--backend",
        "element",
        "--level",
        "1",
        "--out",
        str(output_path),
        "--steps",
        "1",
        "--total-steps",
        "24",
        "--maxit",
        "0",
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
        "--element-reorder-mode",
        "block_xyz",
        "--local-hessian-mode",
        "element",
        "--ksp-type",
        "gmres",
        "--pc-type",
        "hypre",
        "--convergence-metric",
        "reference_elastic_energy",
        "--riesz-ksp-type",
        "cg",
        "--riesz-pc-type",
        "jacobi",
        "--riesz-ksp-rtol",
        "1e-10",
        "--riesz-ksp-atol",
        "0",
        "--riesz-ksp-max-it",
        "5000",
        "--riesz-true-residual-rtol",
        "1e-8",
        "--quiet",
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "BLIS_NUM_THREADS": "1",
            "FNE_SKIP_REORDERED_WARMUP": "1",
            "JAX_PLATFORMS": "cpu",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "XLA_FLAGS": (
                "--xla_cpu_multi_thread_eigen=false "
                "intra_op_parallelism_threads=1 "
                "--xla_force_host_platform_device_count=1"
            ),
        }
    )
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    result = payload["result"]
    assert result["convergence_metric_requested"] == "reference_elastic_energy"
    assert result["convergence_metric"] == "reference_elastic_energy"
    configuration = result["nonlinear_convergence"]["configuration"]
    assert configuration["state_scale_source"] == (
        "initial_reference_deformation_map_primal_norm"
    )
    certificate = result["nonlinear_convergence"]["metric"]["provenance"][
        "spd_certificate"
    ]
    assert certificate["certified_spd"] is True
    assert certificate["inertia"]["positive"] == result["free_dofs"]

    step_convergence = result["steps"][0]["convergence"]
    assert step_convergence["metric"]["name"] == (
        "hyperelasticity_reference_elastic_energy"
    )
    assert result["nonlinear_convergence"]["terminal"] == step_convergence
    norm_solve = step_convergence["dual_residual_metadata"]
    assert norm_solve["reason"] > 0
    assert norm_solve["effective_norm_type"] == "unpreconditioned"
    assert norm_solve["relative_true_residual"] <= norm_solve[
        "true_residual_rtol_gate"
    ]
    partitions = step_convergence["metric"]["provenance"]["input_identity"][
        "reference_free_state_owned_partitions"
    ]
    assert len(partitions) == mpi_ranks
    assert all(len(row["values_sha256"]) == 64 for row in partitions)


def test_reference_elastic_metric_survives_nonlinear_residual_directions(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "he_reference_metric_nonlinear.json"
    pinned_cpu = min(os.sched_getaffinity(0))
    command = [
        "taskset",
        "--cpu-list",
        str(pinned_cpu),
        str(PYTHON),
        str(CASE_RUNNER),
        "--problem",
        "he",
        "--backend",
        "element",
        "--level",
        "1",
        "--out",
        str(output_path),
        "--steps",
        "1",
        "--total-steps",
        "24",
        "--maxit",
        "80",
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
        "--element-reorder-mode",
        "block_xyz",
        "--local-hessian-mode",
        "element",
        "--ksp-type",
        "gmres",
        "--pc-type",
        "hypre",
        "--ksp-rtol",
        "1e-8",
        "--ksp-max-it",
        "1000",
        "--convergence-metric",
        "reference_elastic_energy",
        "--riesz-ksp-type",
        "cg",
        "--riesz-pc-type",
        "jacobi",
        "--riesz-ksp-rtol",
        "1e-10",
        "--riesz-ksp-atol",
        "0",
        "--riesz-ksp-max-it",
        "5000",
        "--riesz-true-residual-rtol",
        "1e-8",
        "--tolf",
        "1e300",
        "--tolg",
        "0",
        "--tolg-rel",
        "1e-2",
        "--tolx-rel",
        "1e300",
        "--tolx-abs",
        "1e300",
        "--line-search",
        "armijo",
        "--no-retry-on-failure",
        "--save-history",
        "--quiet",
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "BLIS_NUM_THREADS": "1",
            "FNE_SKIP_REORDERED_WARMUP": "1",
            "JAX_ENABLE_X64": "True",
            "JAX_PLATFORMS": "cpu",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "VECLIB_MAXIMUM_THREADS": "1",
            "XLA_FLAGS": (
                "--xla_cpu_multi_thread_eigen=false "
                "intra_op_parallelism_threads=1 "
                "--xla_force_host_platform_device_count=1"
            ),
        }
    )
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    step = payload["result"]["steps"][0]
    assert step["success"] is True
    assert step["nit"] > 0

    def mappings(value: object):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from mappings(child)
        elif isinstance(value, list):
            for child in value:
                yield from mappings(child)

    norm_solves = [
        item
        for item in mappings(step)
        if "relative_true_residual" in item and "true_residual_rtol_gate" in item
    ]
    assert norm_solves
    assert all(item["ksp_type"] == "cg" for item in norm_solves)
    assert all(item["pc_type"] == "jacobi" for item in norm_solves)
    assert all(item["requested_norm_type"] == "unpreconditioned" for item in norm_solves)
    assert all(item["effective_norm_type"] == "unpreconditioned" for item in norm_solves)
    assert all(
        item["reported_residual_norm_type"] == "unpreconditioned"
        for item in norm_solves
    )
    assert all(item["requested_rtol"] == pytest.approx(1.0e-10) for item in norm_solves)
    assert all(item["effective_rtol"] == pytest.approx(1.0e-10) for item in norm_solves)
    assert all(item["requested_atol"] == 0.0 for item in norm_solves)
    assert all(item["effective_atol"] == 0.0 for item in norm_solves)
    assert all(
        item["true_residual_rtol_gate"] == pytest.approx(1.0e-8)
        for item in norm_solves
    )
    assert all(
        item["relative_true_residual"] <= item["true_residual_rtol_gate"]
        for item in norm_solves
    )
