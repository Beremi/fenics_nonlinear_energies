from __future__ import annotations

import numpy as np

from src.core.problem_data.hdf5 import load_problem_hdf5, mesh_data_path
from src.problems.ginzburg_landau.jax.mesh import MeshGL2D
from src.problems.hyperelasticity.jax.mesh import MeshHyperElasticity3D
from src.problems.hyperelasticity.support.mesh import (
    MeshHyperElasticity3D as MeshHyperElasticity3DSupport,
)
from src.problems.plaplace.jax.mesh import MeshpLaplace2D


def _assert_sparse_equal(lhs, rhs):
    lhs = lhs.tocoo()
    rhs = rhs.tocoo()
    assert lhs.shape == rhs.shape
    np.testing.assert_array_equal(lhs.row, rhs.row)
    np.testing.assert_array_equal(lhs.col, rhs.col)
    np.testing.assert_allclose(lhs.data, rhs.data)


def test_plaplace_loader_matches_hdf5():
    raw, adjacency = load_problem_hdf5(mesh_data_path("pLaplace", "pLaplace_level5.h5"))
    mesh = MeshpLaplace2D(5)
    for key in ("nodes", "u_0", "freedofs", "elems", "dvx", "dvy", "vol", "f"):
        np.testing.assert_allclose(mesh.params[key], raw[key])
    assert float(mesh.params["p"]) == float(raw["p"])
    _assert_sparse_equal(mesh.adjacency, adjacency)


def test_gl_loader_matches_hdf5():
    raw, adjacency = load_problem_hdf5(mesh_data_path("GinzburgLandau", "GL_level5.h5"))
    mesh = MeshGL2D(5)
    for key in ("nodes", "u_0", "freedofs", "elems", "dvx", "dvy", "vol", "ip", "w"):
        np.testing.assert_allclose(mesh.params[key], raw[key])
    assert float(mesh.params["eps"]) == float(raw["eps"])
    _assert_sparse_equal(mesh.adjacency, adjacency)


def test_hyperelasticity_support_loader_matches_hdf5():
    raw, adjacency = load_problem_hdf5(
        mesh_data_path("HyperElasticity", "HyperElasticity_level1.h5")
    )
    mesh = MeshHyperElasticity3DSupport(1)
    _assert_sparse_equal(mesh.adjacency, adjacency)
    np.testing.assert_allclose(mesh.params["nodes2coord"], raw["nodes2coord"])
    np.testing.assert_array_equal(mesh.params["elems_scalar"], raw["elems2nodes"])
    np.testing.assert_array_equal(mesh.params["freedofs"], raw["dofsMinim"].ravel())
    np.testing.assert_allclose(mesh.params["u_0"], raw["u0"].ravel())


def test_hyperelasticity_jax_loader_matches_hdf5():
    raw, adjacency = load_problem_hdf5(
        mesh_data_path("HyperElasticity", "HyperElasticity_level1.h5")
    )
    mesh = MeshHyperElasticity3D(1)
    _assert_sparse_equal(mesh.adjacency, adjacency)
    for key in ("nodes2coord", "u0", "dofsMinim", "elems2nodes", "dphix", "dphiy", "dphiz", "vol"):
        np.testing.assert_allclose(mesh.params[key], raw[key])
