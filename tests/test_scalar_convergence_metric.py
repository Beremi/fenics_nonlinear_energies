from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from petsc4py import PETSc

from experiments.runners import run_trust_region_case
from src.core.petsc.metrics import DiagonalRieszMetric
from src.core.petsc.scalar_problem_driver import (
    _build_scalar_convergence_metric,
    _scalar_lumped_mass_weights,
)
from src.problems.ginzburg_landau.jax_petsc import solve_GL_dof


REPO_ROOT = Path(__file__).resolve().parents[1]


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-standard JSON constant: {token}")


class _SequentialAssembler:
    def __init__(self, permutation: np.ndarray):
        self.part = SimpleNamespace(perm=np.asarray(permutation, dtype=np.int64))

    @staticmethod
    def create_vec(values: np.ndarray) -> PETSc.Vec:
        vector = PETSc.Vec().createSeq(len(values), comm=PETSc.COMM_SELF)
        vector.setArray(np.asarray(values, dtype=np.float64).copy())
        return vector


def _two_triangle_params() -> dict[str, np.ndarray]:
    return {
        "nodes": np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        "elems": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        "vol": np.array([0.5, 0.5], dtype=np.float64),
        "freedofs": np.array([0, 1, 2, 3], dtype=np.int64),
    }


def test_lumped_mass_weights_preserve_area_and_nodal_incidence() -> None:
    weights = _scalar_lumped_mass_weights(_two_triangle_params())
    np.testing.assert_allclose(weights, np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0]))
    assert float(np.sum(weights)) == pytest.approx(1.0)


def test_scalar_metric_uses_unit_field_scale_and_hashed_provenance() -> None:
    params = _two_triangle_params()
    assembler = _SequentialAssembler(np.array([2, 0, 3, 1]))
    args = SimpleNamespace(
        convergence_metric="lumped_l2",
        convergence_state_scale=None,
    )
    metric = None
    vector = None
    try:
        metric, state_scale, configuration = _build_scalar_convergence_metric(
            args,
            params,
            assembler,
        )
        assert isinstance(metric, DiagonalRieszMetric)
        assert state_scale == pytest.approx(1.0)
        assert configuration["state_scale_source"] == "unit_field_lumped_l2_norm"
        assert configuration["correction_normalization"] == "metric_current_state"
        provenance = configuration["metric"]["provenance"]
        assert all(len(value) == 64 for value in provenance["input_sha256"].values())

        vector = assembler.create_vec(np.ones(4, dtype=np.float64))
        assert metric.primal_norm(vector).value == pytest.approx(1.0)
    finally:
        if vector is not None:
            vector.destroy()
        if metric is not None:
            metric.destroy()


def test_case_parser_exposes_scalar_metric_and_rejects_it_for_he() -> None:
    parser = run_trust_region_case._build_parser()
    args = parser.parse_args(
        [
            "--problem",
            "plaplace",
            "--backend",
            "element",
            "--level",
            "3",
            "--out",
            "unused.json",
            "--convergence-metric",
            "lumped_l2",
        ]
    )
    assert args.convergence_metric == "lumped_l2"
    he_args = parser.parse_args(
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
            "lumped_l2",
        ]
    )
    with pytest.raises(ValueError, match="does not use the scalar"):
        run_trust_region_case._run_he(he_args)

    gl_parser = solve_GL_dof._build_parser({"reference": {}})
    gl_args = gl_parser.parse_args(["--convergence-metric", "lumped_l2"])
    assert gl_args.convergence_metric == "lumped_l2"


def test_plaplace_lumped_metric_small_solver_smoke(tmp_path: Path) -> None:
    output = tmp_path / "plaplace_lumped_l2.json"
    state_output = tmp_path / "plaplace_lumped_l2_state.npz"
    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        "experiments/runners/run_trust_region_case.py",
        "--problem",
        "plaplace",
        "--backend",
        "element",
        "--level",
        "3",
        "--out",
        str(output),
        "--state-out",
        str(state_output),
        "--ksp-type",
        "cg",
        "--pc-type",
        "hypre",
        "--ksp-rtol",
        "1e-3",
        "--ksp-max-it",
        "100",
        "--maxit",
        "1",
        "--line-search",
        "armijo",
        "--convergence-metric",
        "lumped_l2",
        "--save-history",
        "--quiet",
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
    payload = json.loads(
        output.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )
    result = payload["result"]
    assert result["metadata"]["convergence"]["selection"] == "lumped_l2"
    step = result["steps"][0]
    assert step["convergence"]["metric"]["name"] == "scalar_p1_lumped_l2"
    assert step["convergence"]["correction_mode"] == "metric_current_state"
    assert step["convergence"]["dual_residual_norm"] > 0.0
    assert len(step["history"]) == 1
    state_metadata = result["metadata"]["state_output"]
    assert state_output.is_file()
    assert hashlib.sha256(state_output.read_bytes()).hexdigest() == state_metadata[
        "file_sha256"
    ]
    with np.load(state_output, allow_pickle=False) as state:
        assert str(state["state_ordering"]) == "global mesh-node order"
        assert str(state["convergence_metric"]) == "lumped_l2"
        assert state["u"].shape == (result["total_dofs"],)
