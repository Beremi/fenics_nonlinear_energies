"""Reordered PETSc element assembler for 3D heterogeneous slope-stability."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from petsc4py import PETSc

from src.core.petsc.reordered_element_base import ReorderedElementAssemblerBase
from src.problems.slope_stability_3d.jax.jax_energy_3d import (
    chunked_vmapped_elastic_element_hessian_3d,
    chunked_vmapped_element_hessian_3d,
    elastic_element_energy_3d,
    element_energy_3d,
)


config.update("jax_enable_x64", True)


class SlopeStability3DReorderedElementAssembler(ReorderedElementAssemblerBase):
    """3D vector-valued overlap-domain assembler backed by JAX autodiff."""

    block_size = 3
    coordinate_key = "nodes"
    dirichlet_key = "u_0"
    local_elem_data_keys = (
        "dphix",
        "dphiy",
        "dphiz",
        "quad_weight",
        "c_bar_q",
        "sin_phi_q",
        "shear_q",
        "bulk_q",
        "lame_q",
    )
    near_nullspace_key = "elastic_kernel"

    def __init__(
        self,
        params,
        comm,
        adjacency,
        *,
        ksp_rtol=1.0e-3,
        ksp_type="cg",
        pc_type="hypre",
        ksp_max_it=200,
        use_near_nullspace=True,
        pc_options=None,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
        perm_override=None,
        block_size_override=None,
        distribution_strategy=None,
        reuse_hessian_value_buffers=True,
        p4_hessian_chunk_size=32,
    ):
        self.constitutive_mode = "plastic"
        self._kernel_cache: dict[str, dict[str, object]] = {}
        self.p4_hessian_chunk_size = max(1, int(p4_hessian_chunk_size))
        if block_size_override is not None:
            self.block_size = int(block_size_override)
        super().__init__(
            params,
            comm,
            adjacency,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_max_it=ksp_max_it,
            pc_options=pc_options,
            reorder_mode=reorder_mode,
            local_hessian_mode=local_hessian_mode,
            use_near_nullspace=use_near_nullspace,
            perm_override=perm_override,
            distribution_strategy=distribution_strategy,
            reuse_hessian_value_buffers=reuse_hessian_value_buffers,
        )

    def _build_local_element_kernels(self, constitutive_mode: str) -> dict[str, object]:
        elems = jnp.asarray(self.local_data.elems_local_np, dtype=jnp.int32)
        dphix = jnp.asarray(self.local_data.local_elem_data["dphix"], dtype=jnp.float64)
        dphiy = jnp.asarray(self.local_data.local_elem_data["dphiy"], dtype=jnp.float64)
        dphiz = jnp.asarray(self.local_data.local_elem_data["dphiz"], dtype=jnp.float64)
        quad_weight = jnp.asarray(
            self.local_data.local_elem_data["quad_weight"], dtype=jnp.float64
        )
        c_bar_q = jnp.asarray(
            self.local_data.local_elem_data["c_bar_q"], dtype=jnp.float64
        )
        sin_phi_q = jnp.asarray(
            self.local_data.local_elem_data["sin_phi_q"], dtype=jnp.float64
        )
        shear_q = jnp.asarray(
            self.local_data.local_elem_data["shear_q"], dtype=jnp.float64
        )
        bulk_q = jnp.asarray(
            self.local_data.local_elem_data["bulk_q"], dtype=jnp.float64
        )
        lame_q = jnp.asarray(
            self.local_data.local_elem_data["lame_q"], dtype=jnp.float64
        )
        energy_weights = jnp.asarray(self.local_data.energy_weights, dtype=jnp.float64)
        degree = int(self.params.get("element_degree", 2))
        mode = str(constitutive_mode)

        if mode == "elastic":

            hess_elem_batch = jax.jit(
                jax.vmap(
                    jax.hessian(elastic_element_energy_3d),
                    in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
                )
            )

            def _eval_elem_energy(v_elem):
                return jax.vmap(
                    elastic_element_energy_3d,
                    in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
                )(
                    v_elem,
                    dphix,
                    dphiy,
                    dphiz,
                    quad_weight,
                    shear_q,
                    bulk_q,
                    lame_q,
                )

        elif mode == "plastic":

            hess_elem_batch = jax.jit(
                jax.vmap(
                    jax.hessian(element_energy_3d),
                    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                )
            )

            def _eval_elem_energy(v_elem):
                return jax.vmap(
                    element_energy_3d,
                    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                )(
                    v_elem,
                    dphix,
                    dphiy,
                    dphiz,
                    quad_weight,
                    c_bar_q,
                    sin_phi_q,
                    shear_q,
                    bulk_q,
                    lame_q,
                )

        else:
            raise ValueError(f"Unsupported 3D constitutive mode {constitutive_mode!r}")

        def local_full_energy(v_local):
            v_elem = v_local[elems]
            return jnp.sum(_eval_elem_energy(v_elem))

        grad_local = jax.grad(local_full_energy)

        @jax.jit
        def energy_fn(v_local):
            v_elem = v_local[elems]
            e = _eval_elem_energy(v_elem)
            return jnp.sum(e * energy_weights)

        @jax.jit
        def local_grad_fn(v_local):
            return grad_local(v_local)

        if degree == 4 and mode == "plastic":

            def elem_hess_fn(v_local):
                v_elem = v_local[elems]
                return chunked_vmapped_element_hessian_3d(
                    v_elem,
                    dphix,
                    dphiy,
                    dphiz,
                    quad_weight,
                    c_bar_q,
                    sin_phi_q,
                    shear_q,
                    bulk_q,
                    lame_q,
                    chunk_size=self.p4_hessian_chunk_size,
                )

        elif degree == 4 and mode == "elastic":

            def elem_hess_fn(v_local):
                v_elem = v_local[elems]
                return chunked_vmapped_elastic_element_hessian_3d(
                    v_elem,
                    dphix,
                    dphiy,
                    dphiz,
                    quad_weight,
                    shear_q,
                    bulk_q,
                    lame_q,
                    chunk_size=self.p4_hessian_chunk_size,
                )

        else:

            @jax.jit
            def elem_hess_fn(v_local):
                v_elem = v_local[elems]
                if mode == "elastic":
                    return hess_elem_batch(
                        v_elem,
                        dphix,
                        dphiy,
                        dphiz,
                        quad_weight,
                        shear_q,
                        bulk_q,
                        lame_q,
                    )
                return hess_elem_batch(
                    v_elem,
                    dphix,
                    dphiy,
                    dphiz,
                    quad_weight,
                    c_bar_q,
                    sin_phi_q,
                    shear_q,
                    bulk_q,
                    lame_q,
                )

        return {
            "elems": elems,
            "energy_fn": energy_fn,
            "local_grad_fn": local_grad_fn,
            "elem_hess_fn": elem_hess_fn,
            "grad_local": grad_local,
        }

    def _get_local_element_kernels(self, constitutive_mode: str) -> dict[str, object]:
        key = str(constitutive_mode)
        kernels = self._kernel_cache.get(key)
        if kernels is None:
            kernels = self._build_local_element_kernels(key)
            self._kernel_cache[key] = kernels
        return kernels

    def _make_local_element_kernels(self):
        kernels = self._get_local_element_kernels(self.constitutive_mode)
        return (
            kernels["energy_fn"],
            kernels["local_grad_fn"],
            kernels["elem_hess_fn"],
            kernels["grad_local"],
        )

    def _assemble_hessian_with_elem_hess(self, u_owned, elem_hess_jit, *, assembly_mode: str):
        timings = {}
        t_total = time.perf_counter()
        v_local, exchange = self._owned_to_local(
            np.asarray(u_owned, dtype=np.float64),
            zero_dirichlet=False,
        )

        t0 = time.perf_counter()
        elem_hess = np.asarray(elem_hess_jit(jnp.asarray(v_local)).block_until_ready())
        timings["elem_hessian_compute"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        contrib = elem_hess[self._scatter.hess_e, self._scatter.hess_i, self._scatter.hess_j]
        owned_vals = self._reset_owned_hessian_values()
        np.add.at(owned_vals, self._scatter.hess_positions, contrib)
        timings["scatter"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.A.setValuesCOO(
            self._owned_hessian_values_for_petsc(owned_vals),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        self.A.assemble()
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["allgatherv"] = float(exchange["allgatherv"])
        timings["ghost_exchange"] = float(exchange["ghost_exchange"])
        timings["build_v_local"] = float(exchange["build_v_local"])
        timings["p2p_exchange"] = float(exchange["exchange_total"])
        timings["hvp_compute"] = float(timings["elem_hessian_compute"])
        timings["extraction"] = float(timings["scatter"])
        timings["n_hvps"] = 0
        timings["assembly_mode"] = str(assembly_mode)
        timings["constitutive_mode"] = str(assembly_mode).removeprefix("element_overlap_")
        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        self._record_hessian_iteration(timings)
        return timings

    def assemble_hessian_with_mode(self, u_owned, *, constitutive_mode: str):
        kernels = self._get_local_element_kernels(str(constitutive_mode))
        return self._assemble_hessian_with_elem_hess(
            u_owned,
            kernels["elem_hess_fn"],
            assembly_mode=f"element_overlap_{constitutive_mode}",
        )

    def _build_rhs_owned(self) -> np.ndarray:
        force = np.asarray(self.params["force"], dtype=np.float64)
        freedofs = np.asarray(self.params["freedofs"], dtype=np.int64)
        rhs_free = force[freedofs]
        rhs_reordered = rhs_free[self.layout.perm]
        return np.asarray(rhs_reordered[self.layout.lo : self.layout.hi], dtype=np.float64)
