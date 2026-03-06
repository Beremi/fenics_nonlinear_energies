"""
pLaplace-specific configuration for the generic JAX/PETSc assemblers.

Thin subclasses of the shared generic assemblers keep only the problem
definition here: local energy kernels and problem metadata.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from tools_petsc4py.jax_tools.parallel_assembler import (  # noqa: E402
    JaxProblemSpec,
    ProblemDOFHessianAssembler,
    ProblemLocalColoringAssembler,
)


def _build_plaplace_state(params, options):
    """Extract the p exponent for the shared generic layer."""
    del options
    return {"p": float(params["p"])}


def _plaplace_integrand(v_e, dvx_e, dvy_e, p):
    """Elementwise p-Laplace integrand."""
    Fx = jnp.sum(v_e * dvx_e, axis=-1)
    Fy = jnp.sum(v_e * dvy_e, axis=-1)
    return (1.0 / p) * (Fx**2 + Fy**2) ** (p / 2.0)


def _make_plaplace_local_energy_fns(part, state):
    """Return weighted/full local energy closures for the generic assembler."""
    p = state["p"]
    elems = jnp.array(part.elems_local_np, dtype=jnp.int32)
    dvx = jnp.array(part.local_elem_data["dvx"], dtype=jnp.float64)
    dvy = jnp.array(part.local_elem_data["dvy"], dtype=jnp.float64)
    vol = jnp.array(part.local_elem_data["vol"], dtype=jnp.float64)
    vol_w = jnp.array(part.local_elem_data["vol"] * part.elem_weights, dtype=jnp.float64)

    def energy_weighted(v_local):
        v_e = v_local[elems]
        return jnp.sum(_plaplace_integrand(v_e, dvx, dvy, p) * vol_w)

    def energy_full(v_local):
        v_e = v_local[elems]
        return jnp.sum(_plaplace_integrand(v_e, dvx, dvy, p) * vol)

    return energy_weighted, energy_full


def _make_plaplace_element_hessian_jit(part, state):
    """Return a JIT-compiled exact per-element Hessian operator."""
    p = state["p"]

    dvx_jnp = jnp.array(part.local_elem_data["dvx"], dtype=jnp.float64)
    dvy_jnp = jnp.array(part.local_elem_data["dvy"], dtype=jnp.float64)
    vol_jnp = jnp.array(part.local_elem_data["vol"], dtype=jnp.float64)
    elems_jnp = jnp.array(part.elems_local_np, dtype=jnp.int32)

    def element_energy(v_e, dvx_e, dvy_e, vol_e):
        return _plaplace_integrand(v_e, dvx_e, dvy_e, p) * vol_e

    vmapped_hess = jax.vmap(jax.hessian(element_energy), in_axes=(0, 0, 0, 0))

    @jax.jit
    def compute_elem_hessians(v_local):
        v_e = v_local[elems_jnp]
        return vmapped_hess(v_e, dvx_jnp, dvy_jnp, vol_jnp)

    return compute_elem_hessians


PLAPLACE_PROBLEM_SPEC = JaxProblemSpec(
    elem_data_keys=("dvx", "dvy", "vol"),
    make_local_energy_fns=_make_plaplace_local_energy_fns,
    make_element_hessian_jit=_make_plaplace_element_hessian_jit,
    build_state=_build_plaplace_state,
    rhs_key="f",
)


class ParallelDOFHessianAssembler(ProblemDOFHessianAssembler):
    """pLaplace assembler with global multi-start coloring."""

    problem_spec = PLAPLACE_PROBLEM_SPEC

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="gamg",
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
    """pLaplace assembler with per-rank local coloring + vmap."""

    problem_spec = PLAPLACE_PROBLEM_SPEC

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="gamg",
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
