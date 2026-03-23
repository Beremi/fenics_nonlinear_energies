from __future__ import annotations

import numpy as np

from src.core.problem_data.hdf5 import load_problem_hdf5, mesh_data_path
from src.problems.ginzburg_landau.jax.mesh import MeshGL2D
from src.problems.hyperelasticity.jax.mesh import MeshHyperElasticity3D
from src.problems.hyperelasticity.support.mesh import (
    MeshHyperElasticity3D as MeshHyperElasticity3DSupport,
)
from src.problems.plaplace.jax.mesh import MeshpLaplace2D
from src.problems.slope_stability.jax.mesh import MeshSlopeStability2D
from src.problems.slope_stability.support import (
    DEFAULT_LEVEL,
    case_name_for_level,
    ensure_same_mesh_case_hdf5,
    load_case_hdf5,
)


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


def test_slope_stability_loader_matches_hdf5():
    raw, adjacency = load_problem_hdf5(
        mesh_data_path("SlopeStability", "ssr_homo_capture_p2_h1.h5")
    )
    mesh = MeshSlopeStability2D()
    _assert_sparse_equal(mesh.adjacency, adjacency)
    for key in (
        "nodes",
        "elems_scalar",
        "elems",
        "surf",
        "q_mask",
        "freedofs",
        "elem_B",
        "quad_weight",
        "force",
        "u_0",
        "eps_p_old",
    ):
        np.testing.assert_allclose(mesh.params[key], raw[key])
    for key in (
        "h",
        "x1",
        "x2",
        "x3",
        "y1",
        "y2",
        "beta_deg",
        "E",
        "nu",
        "c0",
        "phi_deg",
        "psi_deg",
        "gamma",
    ):
        assert float(mesh.params[key]) == float(raw[key])


def test_slope_stability_level_loaders_match_hdf5():
    for level in range(1, 6):
        raw, adjacency = load_problem_hdf5(
            mesh_data_path("SlopeStability", f"ssr_homo_capture_p2_level{level}.h5")
        )
        mesh = MeshSlopeStability2D(level=level)
        assert mesh.level == level
        assert mesh.case == case_name_for_level(level)
        _assert_sparse_equal(mesh.adjacency, adjacency)
        for key in ("nodes", "elems_scalar", "freedofs", "elem_B", "quad_weight", "force"):
            np.testing.assert_allclose(mesh.params[key], raw[key])
        assert int(mesh.params["level"]) == level
        assert mesh.u_init.shape == (mesh.params["freedofs"].shape[0],)
        assert mesh.elastic_kernel.shape == (mesh.params["freedofs"].shape[0], 3)


def test_slope_stability_legacy_alias_matches_level3():
    alias = MeshSlopeStability2D(case="ssr_homo_capture_p2_h1")
    level3 = MeshSlopeStability2D(level=DEFAULT_LEVEL)
    assert alias.level == DEFAULT_LEVEL == level3.level
    np.testing.assert_allclose(alias.params["nodes"], level3.params["nodes"])
    np.testing.assert_array_equal(alias.params["elems_scalar"], level3.params["elems_scalar"])
    np.testing.assert_array_equal(alias.params["freedofs"], level3.params["freedofs"])
    np.testing.assert_allclose(alias.u_init, level3.u_init)
    np.testing.assert_allclose(alias.elastic_kernel, level3.elastic_kernel)


def test_slope_stability_same_mesh_p4_level4_hdf5_roundtrip():
    path = ensure_same_mesh_case_hdf5(4, 4)
    case = load_case_hdf5(path)
    assert str(case.case_name).endswith("_p4_same_mesh")
    assert int(case.level) == 4
    assert case.nodes.ndim == 2
    assert case.elems_scalar.shape[1] == 15
    assert case.freedofs.ndim == 1
    assert case.elem_B.shape[0] == case.elems_scalar.shape[0]
