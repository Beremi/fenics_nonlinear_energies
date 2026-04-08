"""JAX scalar-energy kernels for the 3D heterogeneous Mohr-Coulomb benchmark."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import config, lax

config.update("jax_enable_x64", True)


def strain6_from_local_gradients(
    u_elem: jnp.ndarray,
    dphix_q: jnp.ndarray,
    dphiy_q: jnp.ndarray,
    dphiz_q: jnp.ndarray,
) -> jnp.ndarray:
    """Return the source-consistent 3D strain ordering ``[xx, yy, zz, xy, yz, xz]``."""

    ux = u_elem[0::3]
    uy = u_elem[1::3]
    uz = u_elem[2::3]

    e_xx = jnp.dot(ux, dphix_q)
    e_yy = jnp.dot(uy, dphiy_q)
    e_zz = jnp.dot(uz, dphiz_q)
    g_xy = jnp.dot(ux, dphiy_q) + jnp.dot(uy, dphix_q)
    g_yz = jnp.dot(uy, dphiz_q) + jnp.dot(uz, dphiy_q)
    g_xz = jnp.dot(ux, dphiz_q) + jnp.dot(uz, dphix_q)
    return jnp.array([e_xx, e_yy, e_zz, g_xy, g_yz, g_xz], dtype=jnp.float64)


def _safe_signed_denom(x: jnp.ndarray, tiny: float = 1.0e-15) -> jnp.ndarray:
    sign = jnp.where(x >= 0.0, 1.0, -1.0)
    return jnp.where(jnp.abs(x) < tiny, sign * tiny, x)


def principal_values_from_sym6(
    eps6: jnp.ndarray,
    *,
    tiny: float = 1.0e-15,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return source-ordered principal strains ``eig_1 >= eig_2 >= eig_3``.

    The input uses engineering shear components. The source 3D benchmark first
    maps those to the tensor-shear representation with ``IDENT =
    diag(1, 1, 1, 1/2, 1/2, 1/2)`` before building invariants and principal
    values, so we must do the same here.
    """

    e11, e22, e33, g12, g23, g13 = eps6
    e12 = 0.5 * g12
    e23 = 0.5 * g23
    e13 = 0.5 * g13
    mat = jnp.array(
        [
            [e11, e12, e13],
            [e12, e22, e23],
            [e13, e23, e33],
        ],
        dtype=jnp.float64,
    )
    tie_break = jnp.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=jnp.float64,
    )
    eigvals = jnp.linalg.eigvalsh(mat + float(tiny) * tie_break)
    eig_3, eig_2, eig_1 = eigvals[0], eigvals[1], eigvals[2]
    I1 = e11 + e22 + e33
    return eig_1, eig_2, eig_3, I1


def mc_potential_density_3d(
    eps6: jnp.ndarray,
    c_bar: jnp.ndarray,
    sin_phi: jnp.ndarray,
    shear: jnp.ndarray,
    bulk: jnp.ndarray,
    lame: jnp.ndarray,
    *,
    tiny: float = 1.0e-15,
) -> jnp.ndarray:
    """Source-style 3D Mohr-Coulomb scalar potential density."""

    e11, e22, e33, e12, e23, e13 = eps6
    elastic_quadratic = (
        e11 * e11
        + e22 * e22
        + e33 * e33
        + 0.5 * (e12 * e12 + e23 * e23 + e13 * e13)
    )
    eig_1, eig_2, eig_3, I1 = principal_values_from_sym6(eps6, tiny=tiny)

    f_tr = (
        2.0 * shear * ((1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * eig_3)
        + 2.0 * lame * sin_phi * I1
        - c_bar
    )
    gamma_sl = (eig_1 - eig_2) / jnp.maximum(tiny, 1.0 + sin_phi)
    gamma_sr = (eig_2 - eig_3) / jnp.maximum(tiny, 1.0 - sin_phi)
    gamma_la = (eig_1 + eig_2 - 2.0 * eig_3) / jnp.maximum(tiny, 3.0 - sin_phi)
    gamma_ra = (2.0 * eig_1 - eig_2 - eig_3) / jnp.maximum(tiny, 3.0 + sin_phi)

    denom_s = 4.0 * lame * sin_phi**2 + 4.0 * shear * (1.0 + sin_phi**2)
    denom_l = (
        4.0 * lame * sin_phi**2
        + shear * (1.0 + sin_phi) ** 2
        + 2.0 * shear * (1.0 - sin_phi) ** 2
    )
    denom_r = (
        4.0 * lame * sin_phi**2
        + 2.0 * shear * (1.0 + sin_phi) ** 2
        + shear * (1.0 - sin_phi) ** 2
    )
    denom_a = 4.0 * bulk * sin_phi**2

    lambda_s = f_tr / _safe_signed_denom(denom_s, tiny)
    lambda_l = (
        shear * ((1.0 + sin_phi) * (eig_1 + eig_2) - 2.0 * (1.0 - sin_phi) * eig_3)
        + 2.0 * lame * sin_phi * I1
        - c_bar
    ) / _safe_signed_denom(denom_l, tiny)
    lambda_r = (
        shear * (2.0 * (1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * (eig_2 + eig_3))
        + 2.0 * lame * sin_phi * I1
        - c_bar
    ) / _safe_signed_denom(denom_r, tiny)
    lambda_a = (
        2.0 * bulk * sin_phi * I1 - c_bar
    ) / _safe_signed_denom(denom_a, tiny)

    psi_el = 0.5 * lame * I1**2 + shear * elastic_quadratic
    psi_s = 0.5 * lame * I1**2 + shear * elastic_quadratic - 0.5 * denom_s * lambda_s**2
    psi_l = (
        0.5 * lame * I1**2
        + shear * (eig_3**2 + 0.5 * (eig_1 + eig_2) ** 2)
        - 0.5 * denom_l * lambda_l**2
    )
    psi_r = (
        0.5 * lame * I1**2
        + shear * (eig_1**2 + 0.5 * (eig_2 + eig_3) ** 2)
        - 0.5 * denom_r * lambda_r**2
    )
    psi_a = 0.5 * bulk * I1**2 - 0.5 * denom_a * lambda_a**2

    f_tr_test = lax.stop_gradient(f_tr)
    gamma_sl_test = lax.stop_gradient(gamma_sl)
    gamma_sr_test = lax.stop_gradient(gamma_sr)
    gamma_la_test = lax.stop_gradient(gamma_la)
    gamma_ra_test = lax.stop_gradient(gamma_ra)
    lambda_s_test = lax.stop_gradient(lambda_s)
    lambda_l_test = lax.stop_gradient(lambda_l)
    lambda_r_test = lax.stop_gradient(lambda_r)

    test_el = f_tr_test <= 0.0
    test_s = (~test_el) & (lambda_s_test <= jnp.minimum(gamma_sl_test, gamma_sr_test))
    test_l = (
        (~(test_el | test_s))
        & (gamma_sl_test < gamma_sr_test)
        & (lambda_l_test >= gamma_sl_test)
        & (lambda_l_test <= gamma_la_test)
    )
    test_r = (
        (~(test_el | test_s | test_l))
        & (gamma_sl_test > gamma_sr_test)
        & (lambda_r_test >= gamma_sr_test)
        & (lambda_r_test <= gamma_ra_test)
    )

    def _elastic(_):
        return psi_el

    def _plastic_after_elastic(_):
        def _shear_return(_):
            return psi_s

        def _after_shear(_):
            def _left_return(_):
                return psi_l

            def _after_left(_):
                def _right_return(_):
                    return psi_r

                return lax.cond(test_r, _right_return, lambda __: psi_a, operand=None)

            return lax.cond(test_l, _left_return, _after_left, operand=None)

        return lax.cond(test_s, _shear_return, _after_shear, operand=None)

    return lax.cond(test_el, _elastic, _plastic_after_elastic, operand=None)


def elastic_potential_density_3d(
    eps6: jnp.ndarray,
    shear: jnp.ndarray,
    bulk: jnp.ndarray,
    lame: jnp.ndarray,
) -> jnp.ndarray:
    """Linear-elastic 3D potential density using the source strain convention."""

    del bulk
    e11, e22, e33, e12, e23, e13 = eps6
    elastic_quadratic = (
        e11 * e11
        + e22 * e22
        + e33 * e33
        + 0.5 * (e12 * e12 + e23 * e23 + e13 * e13)
    )
    I1 = e11 + e22 + e33
    return 0.5 * lame * I1**2 + shear * elastic_quadratic


def element_energy_3d(
    u_elem: jnp.ndarray,
    dphix_e: jnp.ndarray,
    dphiy_e: jnp.ndarray,
    dphiz_e: jnp.ndarray,
    quad_weight_e: jnp.ndarray,
    c_bar_e: jnp.ndarray,
    sin_phi_e: jnp.ndarray,
    shear_e: jnp.ndarray,
    bulk_e: jnp.ndarray,
    lame_e: jnp.ndarray,
) -> jnp.ndarray:
    """Element potential energy for one 3D vector-valued tetrahedron."""

    eps_q = jax.vmap(
        strain6_from_local_gradients,
        in_axes=(None, 0, 0, 0),
    )(u_elem, dphix_e, dphiy_e, dphiz_e)
    psi_q = jax.vmap(
        mc_potential_density_3d,
        in_axes=(0, 0, 0, 0, 0, 0),
    )(eps_q, c_bar_e, sin_phi_e, shear_e, bulk_e, lame_e)
    return jnp.sum(quad_weight_e * psi_q)


def elastic_element_energy_3d(
    u_elem: jnp.ndarray,
    dphix_e: jnp.ndarray,
    dphiy_e: jnp.ndarray,
    dphiz_e: jnp.ndarray,
    quad_weight_e: jnp.ndarray,
    shear_e: jnp.ndarray,
    bulk_e: jnp.ndarray,
    lame_e: jnp.ndarray,
) -> jnp.ndarray:
    """Element potential energy for one linear-elastic 3D tetrahedron."""

    eps_q = jax.vmap(
        strain6_from_local_gradients,
        in_axes=(None, 0, 0, 0),
    )(u_elem, dphix_e, dphiy_e, dphiz_e)
    psi_q = jax.vmap(
        elastic_potential_density_3d,
        in_axes=(0, 0, 0, 0),
    )(eps_q, shear_e, bulk_e, lame_e)
    return jnp.sum(quad_weight_e * psi_q)


def strain6_matrix_from_local_gradients(
    dphix_q: jnp.ndarray,
    dphiy_q: jnp.ndarray,
    dphiz_q: jnp.ndarray,
) -> jnp.ndarray:
    """Return the linear strain operator ``B_q`` in source strain ordering."""

    zeros = jnp.zeros_like(dphix_q)
    row_xx = jnp.stack((dphix_q, zeros, zeros), axis=1).reshape(-1)
    row_yy = jnp.stack((zeros, dphiy_q, zeros), axis=1).reshape(-1)
    row_zz = jnp.stack((zeros, zeros, dphiz_q), axis=1).reshape(-1)
    row_xy = jnp.stack((dphiy_q, dphix_q, zeros), axis=1).reshape(-1)
    row_yz = jnp.stack((zeros, dphiz_q, dphiy_q), axis=1).reshape(-1)
    row_xz = jnp.stack((dphiz_q, zeros, dphix_q), axis=1).reshape(-1)
    return jnp.stack((row_xx, row_yy, row_zz, row_xy, row_yz, row_xz), axis=0)


element_residual_3d = jax.grad(element_energy_3d, argnums=0)
element_hessian_3d = jax.hessian(element_energy_3d, argnums=0)
mc_stress_density_3d = jax.grad(mc_potential_density_3d, argnums=0)
mc_tangent_density_3d = jax.hessian(mc_potential_density_3d, argnums=0)
elastic_element_residual_3d = jax.grad(elastic_element_energy_3d, argnums=0)
elastic_element_hessian_3d = jax.hessian(elastic_element_energy_3d, argnums=0)
elastic_stress_density_3d = jax.grad(elastic_potential_density_3d, argnums=0)
elastic_tangent_density_3d = jax.hessian(elastic_potential_density_3d, argnums=0)


def constitutive_element_residual_3d(
    u_elem: jnp.ndarray,
    dphix_e: jnp.ndarray,
    dphiy_e: jnp.ndarray,
    dphiz_e: jnp.ndarray,
    quad_weight_e: jnp.ndarray,
    c_bar_e: jnp.ndarray,
    sin_phi_e: jnp.ndarray,
    shear_e: jnp.ndarray,
    bulk_e: jnp.ndarray,
    lame_e: jnp.ndarray,
) -> jnp.ndarray:
    """Element residual assembled from constitutive autodiff in strain space."""

    eps_q = jax.vmap(
        strain6_from_local_gradients,
        in_axes=(None, 0, 0, 0),
    )(u_elem, dphix_e, dphiy_e, dphiz_e)
    bmat_q = jax.vmap(
        strain6_matrix_from_local_gradients,
        in_axes=(0, 0, 0),
    )(dphix_e, dphiy_e, dphiz_e)
    sigma_q = jax.vmap(
        mc_stress_density_3d,
        in_axes=(0, 0, 0, 0, 0, 0),
    )(eps_q, c_bar_e, sin_phi_e, shear_e, bulk_e, lame_e)
    return jnp.einsum(
        "qsi,qs,q->i",
        bmat_q,
        sigma_q,
        quad_weight_e,
        optimize="optimal",
    )


def constitutive_element_hessian_3d(
    u_elem: jnp.ndarray,
    dphix_e: jnp.ndarray,
    dphiy_e: jnp.ndarray,
    dphiz_e: jnp.ndarray,
    quad_weight_e: jnp.ndarray,
    c_bar_e: jnp.ndarray,
    sin_phi_e: jnp.ndarray,
    shear_e: jnp.ndarray,
    bulk_e: jnp.ndarray,
    lame_e: jnp.ndarray,
) -> jnp.ndarray:
    """Element tangent assembled from constitutive autodiff in strain space."""

    eps_q = jax.vmap(
        strain6_from_local_gradients,
        in_axes=(None, 0, 0, 0),
    )(u_elem, dphix_e, dphiy_e, dphiz_e)
    bmat_q = jax.vmap(
        strain6_matrix_from_local_gradients,
        in_axes=(0, 0, 0),
    )(dphix_e, dphiy_e, dphiz_e)
    tangent_q = jax.vmap(
        mc_tangent_density_3d,
        in_axes=(0, 0, 0, 0, 0, 0),
    )(eps_q, c_bar_e, sin_phi_e, shear_e, bulk_e, lame_e)
    return jnp.einsum(
        "qsi,qst,qtj,q->ij",
        bmat_q,
        tangent_q,
        bmat_q,
        quad_weight_e,
        optimize="optimal",
    )


def constitutive_elastic_element_hessian_3d(
    u_elem: jnp.ndarray,
    dphix_e: jnp.ndarray,
    dphiy_e: jnp.ndarray,
    dphiz_e: jnp.ndarray,
    quad_weight_e: jnp.ndarray,
    shear_e: jnp.ndarray,
    bulk_e: jnp.ndarray,
    lame_e: jnp.ndarray,
) -> jnp.ndarray:
    """Elastic element tangent assembled from constitutive autodiff."""

    eps_q = jax.vmap(
        strain6_from_local_gradients,
        in_axes=(None, 0, 0, 0),
    )(u_elem, dphix_e, dphiy_e, dphiz_e)
    bmat_q = jax.vmap(
        strain6_matrix_from_local_gradients,
        in_axes=(0, 0, 0),
    )(dphix_e, dphiy_e, dphiz_e)
    tangent_q = jax.vmap(
        elastic_tangent_density_3d,
        in_axes=(0, 0, 0, 0),
    )(eps_q, shear_e, bulk_e, lame_e)
    return jnp.einsum(
        "qsi,qst,qtj,q->ij",
        bmat_q,
        tangent_q,
        bmat_q,
        quad_weight_e,
        optimize="optimal",
    )


element_constitutive_hessian_3d = constitutive_element_hessian_3d
elastic_element_constitutive_hessian_3d = constitutive_elastic_element_hessian_3d

vmapped_element_residual_3d = jax.jit(
    jax.vmap(
        element_residual_3d,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
)

vmapped_element_hessian_3d = jax.jit(
    jax.vmap(
        element_hessian_3d,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
)

vmapped_elastic_element_hessian_3d = jax.jit(
    jax.vmap(
        elastic_element_hessian_3d,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    )
)

vmapped_element_constitutive_hessian_3d = jax.jit(
    jax.vmap(
        element_constitutive_hessian_3d,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
)

vmapped_elastic_element_constitutive_hessian_3d = jax.jit(
    jax.vmap(
        elastic_element_constitutive_hessian_3d,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    )
)

vmapped_mc_stress_density_3d = jax.jit(
    jax.vmap(
        mc_stress_density_3d,
        in_axes=(0, 0, 0, 0, 0, 0),
    )
)


def chunked_vmapped_element_hessian_3d(
    u_elem_batch: jnp.ndarray,
    dphix_batch: jnp.ndarray,
    dphiy_batch: jnp.ndarray,
    dphiz_batch: jnp.ndarray,
    quad_weight_batch: jnp.ndarray,
    c_bar_batch: jnp.ndarray,
    sin_phi_batch: jnp.ndarray,
    shear_batch: jnp.ndarray,
    bulk_batch: jnp.ndarray,
    lame_batch: jnp.ndarray,
    *,
    chunk_size: int,
) -> jnp.ndarray:
    """Evaluate vmapped element Hessians in chunks to keep ``P4`` memory bounded."""

    n_elem = int(u_elem_batch.shape[0])
    if n_elem == 0:
        n_dof = int(u_elem_batch.shape[1]) if u_elem_batch.ndim == 2 else 0
        return jnp.zeros((0, n_dof, n_dof), dtype=jnp.float64)

    chunk_size = max(1, int(chunk_size))
    chunks = []
    for start in range(0, n_elem, chunk_size):
        stop = min(start + chunk_size, n_elem)
        chunks.append(
            vmapped_element_hessian_3d(
                u_elem_batch[start:stop],
                dphix_batch[start:stop],
                dphiy_batch[start:stop],
                dphiz_batch[start:stop],
                quad_weight_batch[start:stop],
                c_bar_batch[start:stop],
                sin_phi_batch[start:stop],
                shear_batch[start:stop],
                bulk_batch[start:stop],
                lame_batch[start:stop],
            )
        )
    if len(chunks) == 1:
        return chunks[0]
    return jnp.concatenate(chunks, axis=0)


def chunked_vmapped_element_constitutive_hessian_3d(
    u_elem_batch: jnp.ndarray,
    dphix_batch: jnp.ndarray,
    dphiy_batch: jnp.ndarray,
    dphiz_batch: jnp.ndarray,
    quad_weight_batch: jnp.ndarray,
    c_bar_batch: jnp.ndarray,
    sin_phi_batch: jnp.ndarray,
    shear_batch: jnp.ndarray,
    bulk_batch: jnp.ndarray,
    lame_batch: jnp.ndarray,
    *,
    chunk_size: int,
) -> jnp.ndarray:
    """Evaluate constitutive-autodiff element Hessians in chunks."""

    n_elem = int(u_elem_batch.shape[0])
    if n_elem == 0:
        n_dof = int(u_elem_batch.shape[1]) if u_elem_batch.ndim == 2 else 0
        return jnp.zeros((0, n_dof, n_dof), dtype=jnp.float64)

    chunk_size = max(1, int(chunk_size))
    chunks = []
    for start in range(0, n_elem, chunk_size):
        stop = min(start + chunk_size, n_elem)
        chunks.append(
            vmapped_element_constitutive_hessian_3d(
                u_elem_batch[start:stop],
                dphix_batch[start:stop],
                dphiy_batch[start:stop],
                dphiz_batch[start:stop],
                quad_weight_batch[start:stop],
                c_bar_batch[start:stop],
                sin_phi_batch[start:stop],
                shear_batch[start:stop],
                bulk_batch[start:stop],
                lame_batch[start:stop],
            )
        )
    if len(chunks) == 1:
        return chunks[0]
    return jnp.concatenate(chunks, axis=0)


def chunked_vmapped_elastic_element_hessian_3d(
    u_elem_batch: jnp.ndarray,
    dphix_batch: jnp.ndarray,
    dphiy_batch: jnp.ndarray,
    dphiz_batch: jnp.ndarray,
    quad_weight_batch: jnp.ndarray,
    shear_batch: jnp.ndarray,
    bulk_batch: jnp.ndarray,
    lame_batch: jnp.ndarray,
    *,
    chunk_size: int,
) -> jnp.ndarray:
    """Evaluate vmapped elastic element Hessians in chunks for ``P4``."""

    n_elem = int(u_elem_batch.shape[0])
    if n_elem == 0:
        n_dof = int(u_elem_batch.shape[1]) if u_elem_batch.ndim == 2 else 0
        return jnp.zeros((0, n_dof, n_dof), dtype=jnp.float64)

    chunk_size = max(1, int(chunk_size))
    chunks = []
    for start in range(0, n_elem, chunk_size):
        stop = min(start + chunk_size, n_elem)
        chunks.append(
            vmapped_elastic_element_hessian_3d(
                u_elem_batch[start:stop],
                dphix_batch[start:stop],
                dphiy_batch[start:stop],
                dphiz_batch[start:stop],
                quad_weight_batch[start:stop],
                shear_batch[start:stop],
                bulk_batch[start:stop],
                lame_batch[start:stop],
            )
        )
    return jnp.concatenate(chunks, axis=0)


def chunked_vmapped_elastic_element_constitutive_hessian_3d(
    u_elem_batch: jnp.ndarray,
    dphix_batch: jnp.ndarray,
    dphiy_batch: jnp.ndarray,
    dphiz_batch: jnp.ndarray,
    quad_weight_batch: jnp.ndarray,
    shear_batch: jnp.ndarray,
    bulk_batch: jnp.ndarray,
    lame_batch: jnp.ndarray,
    *,
    chunk_size: int,
) -> jnp.ndarray:
    """Evaluate constitutive-autodiff elastic element Hessians in chunks."""

    n_elem = int(u_elem_batch.shape[0])
    if n_elem == 0:
        n_dof = int(u_elem_batch.shape[1]) if u_elem_batch.ndim == 2 else 0
        return jnp.zeros((0, n_dof, n_dof), dtype=jnp.float64)

    chunk_size = max(1, int(chunk_size))
    chunks = []
    for start in range(0, n_elem, chunk_size):
        stop = min(start + chunk_size, n_elem)
        chunks.append(
            vmapped_elastic_element_constitutive_hessian_3d(
                u_elem_batch[start:stop],
                dphix_batch[start:stop],
                dphiy_batch[start:stop],
                dphiz_batch[start:stop],
                quad_weight_batch[start:stop],
                shear_batch[start:stop],
                bulk_batch[start:stop],
                lame_batch[start:stop],
            )
        )
    if len(chunks) == 1:
        return chunks[0]
    return jnp.concatenate(chunks, axis=0)
