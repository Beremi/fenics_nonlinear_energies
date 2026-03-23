from __future__ import annotations

import jax
import numpy as np

from src.core.serial.jax_diff import EnergyDerivator
from src.problems.slope_stability.jax.jax_energy import J, element_energy
from src.problems.slope_stability.jax.mesh import MeshSlopeStability2D
from src.problems.slope_stability.support import davis_b_reduction


def test_davis_b_reduction_is_finite_at_lambda_121():
    cohesion, phi_deg = davis_b_reduction(6.0, 45.0, 0.0, 1.21)
    assert np.isfinite(cohesion)
    assert np.isfinite(phi_deg)
    assert cohesion > 0.0
    assert 0.0 < phi_deg < 45.0


def test_davis_b_reduction_returns_raw_cohesion_not_c_bar():
    cohesion, phi_deg = davis_b_reduction(6.0, 45.0, 0.0, 1.0)
    assert np.isclose(cohesion, 4.242640687119286)
    assert np.isclose(phi_deg, 35.264389682754654)
    assert np.isclose(2.0 * cohesion * np.cos(np.deg2rad(phi_deg)), 6.928203230275509)


def test_energy_gradient_and_hessian_are_finite_for_zero_and_elastic_guess():
    mesh = MeshSlopeStability2D()
    params, adjacency, u_init = mesh.get_data_jax()
    cohesion, phi_deg = davis_b_reduction(
        float(mesh.params["c0"]),
        float(mesh.params["phi_deg"]),
        float(mesh.params["psi_deg"]),
        1.21,
    )
    params["cohesion"] = float(cohesion)
    params["phi_deg"] = float(phi_deg)
    params["reg"] = 1.0e-12

    energy = EnergyDerivator(J, params, adjacency, u_init)
    F, dF, ddF = energy.get_derivatives()

    zero = np.zeros_like(np.asarray(u_init))
    f0 = float(F(zero))
    g0 = np.asarray(dF(zero), dtype=np.float64)
    H0 = ddF(zero)

    u_init_np = np.asarray(u_init, dtype=np.float64)
    f1 = float(F(u_init_np))
    g1 = np.asarray(dF(u_init_np), dtype=np.float64)
    H1 = ddF(u_init_np)

    assert np.isfinite(f0)
    assert np.isfinite(f1)
    assert np.all(np.isfinite(g0))
    assert np.all(np.isfinite(g1))
    assert np.all(np.isfinite(H0.data))
    assert np.all(np.isfinite(H1.data))
    assert H0.shape == (5220, 5220)
    assert H1.shape == (5220, 5220)


def test_element_energy_grad_and_hessian_are_finite():
    mesh = MeshSlopeStability2D()
    params, _, _ = mesh.get_data_jax()
    cohesion, phi_deg = davis_b_reduction(
        float(mesh.params["c0"]),
        float(mesh.params["phi_deg"]),
        float(mesh.params["psi_deg"]),
        1.21,
    )

    u_elem = params["u_0"][params["elems"][0]]
    elem_B_elem = params["elem_B"][0]
    quad_weight_elem = params["quad_weight"][0]
    eps_p_old_elem = params["eps_p_old"][0]

    value = element_energy(
        u_elem,
        elem_B_elem,
        quad_weight_elem,
        eps_p_old_elem,
        float(params["E"]),
        float(params["nu"]),
        float(phi_deg),
        float(cohesion),
    )
    grad = jax.grad(element_energy)(
        u_elem,
        elem_B_elem,
        quad_weight_elem,
        eps_p_old_elem,
        float(params["E"]),
        float(params["nu"]),
        float(phi_deg),
        float(cohesion),
    )
    hess = jax.hessian(element_energy)(
        u_elem,
        elem_B_elem,
        quad_weight_elem,
        eps_p_old_elem,
        float(params["E"]),
        float(params["nu"]),
        float(phi_deg),
        float(cohesion),
    )

    assert np.isfinite(float(value))
    assert grad.shape == (12,)
    assert hess.shape == (12, 12)
    assert np.all(np.isfinite(np.asarray(grad, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(hess, dtype=np.float64)))
