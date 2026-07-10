from __future__ import annotations

import jax
import numpy as np

from src.core.serial.jax_diff import EnergyDerivator
from src.problems.slope_stability.jax.jax_energy import J, _mc_energy_density, element_energy
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


def _reported_mc_density():
    cohesion, phi_deg = davis_b_reduction(6.0, 45.0, 0.0, 1.0)

    def density(strain):
        return _mc_energy_density(
            strain,
            jax.numpy.zeros(3, dtype=jax.numpy.float64),
            40000.0,
            0.3,
            float(phi_deg),
            float(cohesion),
            1.0e-12,
        )

    return density


def _rotate_engineering_strain(
    strain: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(float(angle))
    sine = np.sin(float(angle))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    tensor = np.array(
        [
            [strain[0], 0.5 * strain[2]],
            [0.5 * strain[2], strain[1]],
        ],
        dtype=np.float64,
    )
    rotated = rotation @ tensor @ rotation.T
    return (
        np.array([rotated[0, 0], rotated[1, 1], 2.0 * rotated[0, 1]]),
        rotation,
    )


def _rotate_engineering_dual(dual: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # The dual of [eps_xx, eps_yy, gamma_xy] is [sig_xx, sig_yy, tau_xy].
    tensor = np.array(
        [[dual[0], dual[2]], [dual[2], dual[1]]],
        dtype=np.float64,
    )
    rotated = rotation @ tensor @ rotation.T
    return np.array([rotated[0, 0], rotated[1, 1], rotated[0, 1]])


def test_hydrostatic_plastic_apex_has_finite_invariant_gradient():
    density = _reported_mc_density()
    strain = jax.numpy.array([1.0e-2, 1.0e-2, 0.0], dtype=jax.numpy.float64)

    value = float(density(strain))
    gradient = np.asarray(jax.grad(density)(strain), dtype=np.float64)
    hessian = np.asarray(jax.hessian(density)(strain), dtype=np.float64)

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    np.testing.assert_allclose(value, 0.119532, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        gradient,
        np.array([6.0, 6.0, 0.0]),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(hessian, np.zeros((3, 3)), rtol=0.0, atol=1.0e-12)

    # The apex branch is locally affine here.  This directional check guards
    # the energy/gradient contract without asserting smoothness at the separate
    # yield or line--apex switching surfaces.
    direction = np.array([0.3, -0.4, 0.5], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    step = 1.0e-7
    fd = (
        float(density(strain + step * direction))
        - float(density(strain - step * direction))
    ) / (2.0 * step)
    np.testing.assert_allclose(fd, gradient @ direction, rtol=1.0e-9, atol=1.0e-10)


def test_non_degenerate_line_branch_preserves_reference_energy_and_gradient():
    density = _reported_mc_density()
    strain = jax.numpy.array([1.0e-3, -5.0e-4, 2.0e-4], dtype=jax.numpy.float64)

    value = float(density(strain))
    gradient = np.asarray(jax.grad(density)(strain), dtype=np.float64)

    # These values are from the original nondegenerate atan2 reconstruction.
    # The hydrostatic reformulation must not alter the regular line branch.
    np.testing.assert_allclose(value, 0.005475052037488231, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        gradient,
        np.array([1.2470816646656406, -11.47181571244509, 0.8479264918073821]),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_mc_energy_and_gradient_are_rotation_covariant_on_line_and_apex_branches():
    density = _reported_mc_density()
    states = (
        np.array([1.0e-3, -5.0e-4, 2.0e-4], dtype=np.float64),
        np.array([1.0e-2, 1.0e-2, 0.0], dtype=np.float64),
    )

    for strain in states:
        value = float(density(jax.numpy.asarray(strain)))
        gradient = np.asarray(jax.grad(density)(jax.numpy.asarray(strain)), dtype=np.float64)
        for angle in (0.37, 1.13):
            rotated_strain, rotation = _rotate_engineering_strain(strain, angle)
            rotated_value = float(density(jax.numpy.asarray(rotated_strain)))
            rotated_gradient = np.asarray(
                jax.grad(density)(jax.numpy.asarray(rotated_strain)),
                dtype=np.float64,
            )
            expected_gradient = _rotate_engineering_dual(gradient, rotation)

            np.testing.assert_allclose(
                rotated_value,
                value,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            np.testing.assert_allclose(
                rotated_gradient,
                expected_gradient,
                rtol=1.0e-11,
                atol=5.0e-12,
            )


def test_plastic_apex_remains_finite_and_direction_independent_near_hydrostaticity():
    density = _reported_mc_density()
    mean_strain = 1.0e-2

    for radius in (0.0, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6):
        for angle in (0.0, 0.29, 0.91):
            strain = jax.numpy.array(
                [
                    mean_strain + radius * np.cos(2.0 * angle),
                    mean_strain - radius * np.cos(2.0 * angle),
                    2.0 * radius * np.sin(2.0 * angle),
                ],
                dtype=jax.numpy.float64,
            )
            value = float(density(strain))
            gradient = np.asarray(jax.grad(density)(strain), dtype=np.float64)
            hessian = np.asarray(jax.hessian(density)(strain), dtype=np.float64)

            assert np.isfinite(value)
            assert np.all(np.isfinite(gradient))
            assert np.all(np.isfinite(hessian))
            np.testing.assert_allclose(
                gradient,
                np.array([6.0, 6.0, 0.0]),
                rtol=0.0,
                atol=1.0e-11,
            )


def test_exact_repeated_principal_yield_neighborhood_has_finite_selected_derivatives():
    density = _reported_mc_density()
    # The hydrostatic trial stress is 6 at this strain.  The regularized yield
    # test can select different neighboring branches within an O(reg) interval.
    # Finite selected-branch AD is required, but Hessian continuity is neither
    # expected nor asserted at this nonsmooth yield/line--apex neighborhood.
    hydrostatic_modulus = 76923.07692307692
    for stress_offset in (-1.0e-10, -1.0e-12, 0.0, 1.0e-12, 1.0e-10):
        scalar_strain = (6.0 + stress_offset) / hydrostatic_modulus
        strain = jax.numpy.array(
            [scalar_strain, scalar_strain, 0.0],
            dtype=jax.numpy.float64,
        )
        value = float(density(strain))
        gradient = np.asarray(jax.grad(density)(strain), dtype=np.float64)
        hessian = np.asarray(jax.hessian(density)(strain), dtype=np.float64)

        assert np.isfinite(value)
        assert np.all(np.isfinite(gradient))
        assert np.all(np.isfinite(hessian))
