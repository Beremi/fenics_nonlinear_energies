"""GL-specific configuration for the generic JAX/PETSc assemblers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from src.core.petsc.jax_tools.parallel_assembler import (  # noqa: E402
    JaxProblemSpec,
    ProblemDOFHessianAssembler,
    ProblemLocalColoringAssembler,
)


def _build_gl_state(params, options):
    del options
    return {
        "eps": float(params["eps"]),
        "ip": jnp.array(params["ip"], dtype=jnp.float64),
        "w": jnp.array(params["w"], dtype=jnp.float64),
    }


def _gl_integrand(v_e, dvx_e, dvy_e, ip, w, eps):
    fx = jnp.sum(v_e * dvx_e, axis=-1)
    fy = jnp.sum(v_e * dvy_e, axis=-1)
    grad_term = 0.5 * eps * (fx**2 + fy**2)
    nodal_vals = v_e @ ip
    potential_term = 0.25 * (((nodal_vals**2) - 1.0) ** 2) @ w
    return grad_term + potential_term


def _make_gl_local_energy_fns(part, state):
    eps = state["eps"]
    ip = state["ip"]
    w = state["w"]

    elems = jnp.array(part.elems_local_np, dtype=jnp.int32)
    dvx = jnp.array(part.local_elem_data["dvx"], dtype=jnp.float64)
    dvy = jnp.array(part.local_elem_data["dvy"], dtype=jnp.float64)
    vol = jnp.array(part.local_elem_data["vol"], dtype=jnp.float64)
    vol_w = jnp.array(
        part.local_elem_data["vol"] * part.elem_weights,
        dtype=jnp.float64,
    )

    def energy_weighted(v_local):
        v_e = v_local[elems]
        return jnp.sum(_gl_integrand(v_e, dvx, dvy, ip, w, eps) * vol_w)

    def energy_full(v_local):
        v_e = v_local[elems]
        return jnp.sum(_gl_integrand(v_e, dvx, dvy, ip, w, eps) * vol)

    return energy_weighted, energy_full


def _make_gl_element_hessian_jit(part, state):
    eps = state["eps"]
    ip = state["ip"]
    w = state["w"]

    dvx_jnp = jnp.array(part.local_elem_data["dvx"], dtype=jnp.float64)
    dvy_jnp = jnp.array(part.local_elem_data["dvy"], dtype=jnp.float64)
    vol_jnp = jnp.array(part.local_elem_data["vol"], dtype=jnp.float64)
    elems_jnp = jnp.array(part.elems_local_np, dtype=jnp.int32)

    def element_energy(v_e, dvx_e, dvy_e, vol_e):
        return _gl_integrand(v_e, dvx_e, dvy_e, ip, w, eps) * vol_e

    vmapped_hess = jax.vmap(jax.hessian(element_energy), in_axes=(0, 0, 0, 0))

    @jax.jit
    def compute_elem_hessians(v_local):
        v_e = v_local[elems_jnp]
        return vmapped_hess(v_e, dvx_jnp, dvy_jnp, vol_jnp)

    return compute_elem_hessians


GL_PROBLEM_SPEC = JaxProblemSpec(
    elem_data_keys=("dvx", "dvy", "vol"),
    make_local_energy_fns=_make_gl_local_energy_fns,
    make_element_hessian_jit=_make_gl_element_hessian_jit,
    build_state=_build_gl_state,
    rhs_key=None,
)


class ParallelDOFHessianAssembler(ProblemDOFHessianAssembler):
    """GL assembler with global multi-start coloring."""

    problem_spec = GL_PROBLEM_SPEC

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-3,
        ksp_type="gmres",
        pc_type="hypre",
    ):
        super().__init__(
            params=params,
            comm=comm,
            adjacency=adjacency,
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
        )


class LocalColoringAssembler(ProblemLocalColoringAssembler):
    """GL assembler with per-rank local coloring + vmap."""

    problem_spec = GL_PROBLEM_SPEC

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-3,
        ksp_type="gmres",
        pc_type="hypre",
        hvp_eval_mode="batched",
    ):
        super().__init__(
            params=params,
            comm=comm,
            adjacency=adjacency,
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
            hvp_eval_mode=hvp_eval_mode,
        )
