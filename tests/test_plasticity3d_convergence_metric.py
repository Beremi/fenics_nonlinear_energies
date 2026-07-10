from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from petsc4py import PETSc

from src.core.petsc.metrics import MatrixRieszMetric, certify_spd_by_cholesky
from src.problems.slope_stability_3d.jax_petsc import solver as slope3d_solver
from src.problems.slope_stability_3d.jax_petsc.solve_slope_stability_3d_dof import (
    _build_parser,
)
from src.problems.slope_stability_3d.jax_petsc.solver import (
    PROFILE_DEFAULTS,
    _build_convergence_metric,
    _resolve_endpoint_initial_dual_residual,
)
from src.problems.slope_stability_3d.support.mesh import (
    PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = (
    REPO_ROOT
    / "src/problems/slope_stability_3d/jax_petsc/solve_slope_stability_3d_dof.py"
)


def _matrix(values: np.ndarray) -> PETSc.Mat:
    values = np.asarray(values, dtype=np.float64)
    mat = PETSc.Mat().createAIJ(values.shape, comm=PETSc.COMM_SELF)
    mat.setUp()
    rows, columns = np.nonzero(values)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        mat.setValue(int(row), int(column), float(values[row, column]))
    mat.assemble()
    return mat


def _vector(values: list[float]) -> PETSc.Vec:
    vec = PETSc.Vec().createSeq(len(values), comm=PETSc.COMM_SELF)
    vec.setArray(np.asarray(values, dtype=np.float64))
    return vec


def test_parser_preserves_legacy_metric_defaults_and_accepts_riesz_controls() -> None:
    parser = _build_parser(PROFILE_DEFAULTS)
    legacy = parser.parse_args([])
    assert legacy.convergence_metric == "coefficient_l2"
    assert legacy.convergence_state_scale is None

    configured = parser.parse_args(
        [
            "--convergence-metric",
            "reference_elastic_energy",
            "--convergence-state-scale",
            "2.5",
            "--riesz-ksp-type",
            "cg",
            "--riesz-pc-type",
            "jacobi",
            "--riesz-ksp-rtol",
            "1e-11",
            "--riesz-ksp-atol",
            "1e-15",
            "--riesz-ksp-max-it",
            "700",
            "--riesz-true-residual-rtol",
            "1e-9",
            "--riesz-spd-factor-solver-type",
            "mumps",
            "--riesz-symmetry-tol",
            "1e-13",
        ]
    )
    assert configured.convergence_metric == "reference_elastic_energy"
    assert configured.convergence_state_scale == pytest.approx(2.5)
    assert configured.riesz_ksp_rtol == pytest.approx(1.0e-11)
    assert configured.riesz_true_residual_rtol == pytest.approx(1.0e-9)
    assert configured.riesz_spd_factor_solver_type == "mumps"


def test_linear_failure_normalization_preserves_tracked_initial_residual() -> None:
    tracked = SimpleNamespace(value=7.5)
    assert _resolve_endpoint_initial_dual_residual(
        {}, tracked_initial=tracked, endpoint_value=2.0
    ) == pytest.approx(7.5)
    assert (
        _resolve_endpoint_initial_dual_residual(
            {}, tracked_initial=None, endpoint_value=2.0
        )
        is None
    )
    assert _resolve_endpoint_initial_dual_residual(
        {"convergence_metric": {"name": "unit"}},
        tracked_initial=None,
        endpoint_value=2.0,
    ) == pytest.approx(2.0)


def test_cholesky_inertia_certificate_accepts_spd_and_rejects_indefinite() -> None:
    spd = _matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    indefinite = _matrix(np.diag([1.0, -1.0]))
    asymmetric = _matrix(np.array([[2.0, 1.0], [0.0, 2.0]]))
    try:
        certificate = certify_spd_by_cholesky(
            spd,
            factor_solver_type="mumps",
            options_prefix="test_plasticity3d_spd_",
        )
        assert certificate["certified_spd"] is True
        assert certificate["inertia"] == {"negative": 0, "zero": 0, "positive": 2}

        with pytest.raises(ValueError, match="not positive definite"):
            certify_spd_by_cholesky(
                indefinite,
                factor_solver_type="mumps",
                options_prefix="test_plasticity3d_indefinite_",
            )
        assert "test_plasticity3d_indefinite_mat_mumps_icntl_24" not in PETSc.Options()
        with pytest.raises(ValueError, match="not symmetric"):
            certify_spd_by_cholesky(
                asymmetric,
                factor_solver_type="mumps",
                options_prefix="test_plasticity3d_asymmetric_",
            )
    finally:
        spd.destroy()
        indefinite.destroy()
        asymmetric.destroy()


def test_problem_metric_uses_initial_primal_norm_as_dimensioned_state_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _matrix(np.diag([2.0, 8.0]))
    initial = _vector([1.0, 2.0])
    assembler = SimpleNamespace(
        assembly_backend="coo",
        autodiff_tangent_mode="element",
        local_hessian_mode="element",
    )
    monkeypatch.setattr(
        slope3d_solver,
        "_reference_elastic_input_identity",
        lambda **_kwargs: {"unit_test": True},
    )
    params = {
        "constraint_variant": PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM,
        "freedofs": np.arange(2, dtype=np.int64),
        "mesh_name": "unit_mesh",
        "quadrature_rule_id": "unit_rule",
        "shear_q": np.array([2.0]),
        "bulk_q": np.array([4.0]),
        "lame_q": np.array([3.0]),
    }
    args = SimpleNamespace(
        convergence_metric="reference_elastic_energy",
        convergence_state_scale=None,
        elem_degree=1,
        mesh_name="unit_mesh",
        riesz_ksp_type="cg",
        riesz_pc_type="jacobi",
        riesz_ksp_rtol=1.0e-10,
        riesz_ksp_atol=1.0e-14,
        riesz_ksp_max_it=100,
        riesz_true_residual_rtol=1.0e-8,
        riesz_spd_factor_solver_type="mumps",
        riesz_symmetry_tol=1.0e-12,
    )
    metric = None
    try:
        metric, state_scale, metadata = _build_convergence_metric(
            args=args,
            assembler=assembler,
            params=params,
            regularization_state={"elastic_operator": operator},
            initial_state=initial,
        )
        assert isinstance(metric, MatrixRieszMetric)
        assert state_scale == pytest.approx(np.sqrt(34.0))
        assert metadata["state_scale_source"] == "initial_nonlinear_iterate_primal_norm"
        assert metadata["relative_correction_units"] == "dimensionless"
        assert metadata["metric"]["provenance"]["spd_certificate"]["certified_spd"]
    finally:
        if metric is not None:
            metric.destroy()
        initial.destroy()
        operator.destroy()


def test_problem_metric_rejects_zero_implicit_scale_without_destroying_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _matrix(np.diag([2.0, 8.0]))
    initial = _vector([0.0, 0.0])
    output = _vector([0.0, 0.0])
    args = SimpleNamespace(
        convergence_metric="reference_elastic_energy",
        convergence_state_scale=None,
        elem_degree=1,
        mesh_name="unit_mesh",
        riesz_ksp_type="cg",
        riesz_pc_type="jacobi",
        riesz_ksp_rtol=1.0e-10,
        riesz_ksp_atol=1.0e-14,
        riesz_ksp_max_it=100,
        riesz_true_residual_rtol=1.0e-8,
        riesz_spd_factor_solver_type="mumps",
        riesz_symmetry_tol=1.0e-12,
    )
    params = {
        "constraint_variant": PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM,
        "freedofs": np.arange(2, dtype=np.int64),
        "mesh_name": "unit_mesh",
        "quadrature_rule_id": "unit_rule",
        "shear_q": np.array([2.0]),
        "bulk_q": np.array([4.0]),
        "lame_q": np.array([3.0]),
    }
    monkeypatch.setattr(
        slope3d_solver,
        "_reference_elastic_input_identity",
        lambda **_kwargs: {"unit_test": True},
    )
    try:
        with pytest.raises(ValueError, match="zero reference elastic-energy norm"):
            _build_convergence_metric(
                args=args,
                assembler=SimpleNamespace(
                    assembly_backend="coo",
                    autodiff_tangent_mode="element",
                    local_hessian_mode="element",
                ),
                params=params,
                regularization_state={"elastic_operator": operator},
                initial_state=initial,
            )
        operator.mult(initial, output)
        np.testing.assert_allclose(output.getArray(readonly=True), 0.0)
    finally:
        output.destroy()
        initial.destroy()
        operator.destroy()


def test_reference_elastic_metric_small_solver_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "plasticity3d_riesz.json"
    command = [
        "mpiexec",
        "-n",
        "2",
        str(PYTHON),
        "-u",
        str(SOLVER),
        "--mesh_name",
        "hetero_ssr_L1",
        "--elem_degree",
        "1",
        "--pc_type",
        "hypre",
        "--ksp_type",
        "cg",
        "--ksp_rtol",
        "1e-4",
        "--ksp_max_it",
        "200",
        "--maxit",
        "1",
        "--problem_build_mode",
        "rank_local",
        "--distribution_strategy",
        "overlap_p2p",
        "--elastic_initial_guess",
        "--convergence-metric",
        "reference_elastic_energy",
        "--riesz-ksp-rtol",
        "1e-10",
        "--riesz-true-residual-rtol",
        "1e-8",
        "--quiet",
        "--out",
        str(output_path),
    ]
    environment = dict(os.environ)
    environment["FNE_SKIP_REORDERED_WARMUP"] = "1"
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    convergence = payload["nonlinear_convergence"]
    assert convergence["configuration"]["selection"] == "reference_elastic_energy"
    assert len(payload["parallel_diagnostics"]) == 2
    assert convergence["configuration"]["state_scale_source"] == (
        "initial_nonlinear_iterate_primal_norm"
    )
    certificate = convergence["metric"]["provenance"]["spd_certificate"]
    assert certificate["certified_spd"] is True
    assert certificate["inertia"]["positive"] == payload["mesh"]["free_dofs"]
    provenance = convergence["metric"]["provenance"]
    assert provenance["autodiff_tangent_mode"] == "element"
    assert provenance["input_identity"]["tangent_route"]["constitutive_mode"] == (
        "elastic"
    )
    assert all(
        len(value) == 64
        for value in provenance["input_identity"]["array_sha256"].values()
    )
    assert all(
        len(value) == 64
        for value in provenance["input_identity"]["hdf5"][
            "dataset_sha256"
        ].values()
    )
    assert convergence["initial_relative_dual_residual"]["units"] == "dimensionless"
    assert convergence["relative_correction"]["units"] == "dimensionless"
    assert convergence["residual_gate"]["passed"] is False
    assert convergence["coefficient_gradient_l2"] == pytest.approx(
        payload["final_grad_norm"]
    )
    norm_solve = convergence["last_riesz_solve"]
    assert norm_solve["reason"] > 0
    assert norm_solve["rhs_norm"] == pytest.approx(payload["final_grad_norm"])
    assert norm_solve["relative_true_residual"] <= norm_solve[
        "true_residual_rtol_gate"
    ]
