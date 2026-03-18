from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from src.core.cli.threading import configure_jax_cpu_threading
from src.core.petsc.gamg import build_gamg_coordinates
from src.core.petsc.trust_ksp import _get_libpetsc


def test_build_gamg_coordinates_returns_owned_reordered_rows():
    part = SimpleNamespace(perm=np.array([4, 0, 3, 1, 2]), lo=1, hi=4)
    freedofs = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float64,
    )

    coords = build_gamg_coordinates(part, freedofs, nodes)

    np.testing.assert_allclose(coords, np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]]))


def test_build_gamg_coordinates_supports_block_triplets():
    part = SimpleNamespace(perm=np.array([0, 1, 2, 3, 4, 5]), lo=0, hi=6)
    freedofs = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    coords = build_gamg_coordinates(part, freedofs, nodes, block_size=3)

    np.testing.assert_allclose(coords, nodes)


def test_configure_jax_cpu_threading_sets_standard_env(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    threads = configure_jax_cpu_threading(3)
    assert threads == 3
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "3"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
    assert "xla_force_host_platform_device_count=1" in os.environ["XLA_FLAGS"]


def test_trust_ksp_loader_uses_active_petsc_shared_object():
    libpetsc = _get_libpetsc()
    assert hasattr(libpetsc, "KSPCGSetRadius")
