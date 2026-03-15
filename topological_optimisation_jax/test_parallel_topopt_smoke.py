from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = REPO_ROOT / "topological_optimisation_jax" / "solve_topopt_parallel.py"

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


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(THREAD_ENV)
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _common_args(out: Path) -> list[str]:
    return [
        "--nx",
        "24",
        "--ny",
        "12",
        "--fixed_pad_cells",
        "4",
        "--load_pad_cells",
        "4",
        "--outer_maxit",
        "3",
        "--design_maxit",
        "20",
        "--quiet",
        "--json_out",
        str(out),
    ]


def test_parallel_topopt_smoke_single_rank(tmp_path: Path) -> None:
    out = tmp_path / "parallel_topopt_r1.json"
    _run([str(PYTHON), str(SOLVER), *_common_args(out)], REPO_ROOT)
    result = json.loads(out.read_text())
    assert result["nprocs"] == 1
    assert result["result"] in {"max_outer_iterations", "completed"}
    assert len(result["history"]) == 3


def test_parallel_topopt_smoke_four_ranks(tmp_path: Path) -> None:
    if shutil.which("mpiexec") is None:
        pytest.skip("mpiexec is required for this smoke test.")
    out = tmp_path / "parallel_topopt_r4.json"
    _run(["mpiexec", "-n", "4", str(PYTHON), str(SOLVER), *_common_args(out)], REPO_ROOT)
    result = json.loads(out.read_text())
    assert result["nprocs"] == 4
    assert result["result"] in {"max_outer_iterations", "completed"}
    assert len(result["history"]) == 3
