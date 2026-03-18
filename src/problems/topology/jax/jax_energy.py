from __future__ import annotations

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


def expand_free_dofs(free_values, template, freedofs):
    return template.at[freedofs].set(free_values)


def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def theta_from_latent(z, theta_min):
    return theta_min + (1.0 - theta_min) * sigmoid(z)


def element_theta(theta_full, elems):
    return jnp.mean(theta_full[elems], axis=1)


def material_scale_from_design(z_full, elems, theta_min, p_penal):
    theta_full = theta_from_latent(z_full, theta_min)
    return element_theta(theta_full, elems) ** p_penal


def mechanics_energy(
    u_free,
    u_0,
    freedofs,
    elems,
    elem_B,
    elem_area,
    material_scale,
    constitutive,
    force,
):
    u_full = expand_free_dofs(u_free, u_0, freedofs)
    u_elem = u_full[elems]
    strain = jnp.einsum("eij,ej->ei", elem_B, u_elem)
    elastic_density = 0.5 * jnp.einsum("ei,ij,ej->e", strain, constitutive, strain)
    return jnp.sum(elem_area * material_scale * elastic_density) - jnp.dot(force, u_full)


def design_energy(
    z_free,
    z_0,
    freedofs,
    elems,
    elem_grad_phi,
    elem_area,
    e_frozen,
    z_old_full,
    lambda_volume,
    alpha_reg,
    ell_pf,
    mu_move,
    theta_min,
    p_penal,
):
    z_full = expand_free_dofs(z_free, z_0, freedofs)
    theta_full = theta_from_latent(z_full, theta_min)

    theta_elem = theta_full[elems]
    theta_centroid = jnp.mean(theta_elem, axis=1)
    grad_theta = jnp.einsum("eia,ei->ea", elem_grad_phi, theta_elem)

    z_elem = z_full[elems]
    z_old_elem = z_old_full[elems]
    z_delta_centroid = jnp.mean(z_elem - z_old_elem, axis=1)

    double_well = theta_centroid**2 * (1.0 - theta_centroid) ** 2
    reg_density = 0.5 * ell_pf * jnp.sum(grad_theta * grad_theta, axis=1) + double_well / ell_pf
    proximal_density = 0.5 * mu_move * z_delta_centroid**2
    design_density = e_frozen * theta_centroid ** (-p_penal) + lambda_volume * theta_centroid

    return jnp.sum(elem_area * (design_density + alpha_reg * reg_density + proximal_density))


def element_strain(u_full, elems, elem_B):
    return jnp.einsum("eij,ej->ei", elem_B, u_full[elems])


def element_eps_ceps(u_full, elems, elem_B, constitutive):
    strain = element_strain(u_full, elems, elem_B)
    return jnp.einsum("ei,ij,ej->e", strain, constitutive, strain)


def frozen_design_density(
    u_full,
    z_full,
    scalar_elems,
    vector_elems,
    elem_B,
    constitutive,
    theta_min,
    p_penal,
):
    theta_full = theta_from_latent(z_full, theta_min)
    theta_elem = element_theta(theta_full, scalar_elems)
    material_scale = theta_elem**p_penal
    eps_ceps = element_eps_ceps(u_full, vector_elems, elem_B, constitutive)
    return (material_scale**2) * eps_ceps


def compliance(u_full, force):
    return jnp.dot(force, u_full)


def volume_fraction(theta_full, nodal_volume_weights, domain_area):
    return jnp.dot(nodal_volume_weights, theta_full) / domain_area
