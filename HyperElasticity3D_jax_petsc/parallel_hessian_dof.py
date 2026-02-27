"""HyperElasticity-specific parallel Hessian assemblers (DOF-partitioned)."""

import numpy as np
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from tools_petsc4py.parallel_assembler import (  # noqa: E402
    DOFHessianAssemblerBase,
    LocalColoringAssemblerBase,
)


class _HyperElasticityMixin:
    """Mixin providing local Neo-Hookean energy functions."""

    def _make_local_energy_fns(self):
        elems = jnp.array(self.part.elems_local_np, dtype=jnp.int32)  # (n_elem, 12)
        dphix = jnp.array(self.part.local_elem_data["dphix"], dtype=jnp.float64)  # (n_elem, 4)
        dphiy = jnp.array(self.part.local_elem_data["dphiy"], dtype=jnp.float64)  # (n_elem, 4)
        dphiz = jnp.array(self.part.local_elem_data["dphiz"], dtype=jnp.float64)  # (n_elem, 4)
        vol = jnp.array(self.part.local_elem_data["vol"], dtype=jnp.float64)
        vol_w = jnp.array(
            self.part.local_elem_data["vol"] * self.part.elem_weights,
            dtype=jnp.float64,
        )

        C1 = self._C1
        D1 = self._D1
        use_abs_det = self._use_abs_det

        def _energy_with_volume(v_local, elem_vol):
            v_e = v_local[elems]  # (n_elem, 12)
            vx = v_e[:, 0::3]  # (n_elem, 4)
            vy = v_e[:, 1::3]  # (n_elem, 4)
            vz = v_e[:, 2::3]  # (n_elem, 4)

            F11 = jnp.sum(vx * dphix, axis=1)
            F12 = jnp.sum(vx * dphiy, axis=1)
            F13 = jnp.sum(vx * dphiz, axis=1)
            F21 = jnp.sum(vy * dphix, axis=1)
            F22 = jnp.sum(vy * dphiy, axis=1)
            F23 = jnp.sum(vy * dphiz, axis=1)
            F31 = jnp.sum(vz * dphix, axis=1)
            F32 = jnp.sum(vz * dphiy, axis=1)
            F33 = jnp.sum(vz * dphiz, axis=1)

            I1 = (
                F11**2
                + F12**2
                + F13**2
                + F21**2
                + F22**2
                + F23**2
                + F31**2
                + F32**2
                + F33**2
            )
            detF = (
                F11 * (F22 * F33 - F23 * F32)
                - F12 * (F21 * F33 - F23 * F31)
                + F13 * (F21 * F32 - F22 * F31)
            )
            det_for_energy = jnp.abs(detF) if use_abs_det else detF
            W = C1 * (I1 - 3.0 - 2.0 * jnp.log(det_for_energy)) + D1 * (det_for_energy - 1.0) ** 2
            return jnp.sum(W * elem_vol)

        def energy_weighted(v_local):
            return _energy_with_volume(v_local, vol_w)

        def energy_full(v_local):
            return _energy_with_volume(v_local, vol)

        return energy_weighted, energy_full


class ParallelDOFHessianAssembler(_HyperElasticityMixin, DOFHessianAssemblerBase):
    """Global-coloring HE assembler."""

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-1,
        ksp_type="gmres",
        pc_type="hypre",
        ksp_max_it=30,
        use_near_nullspace=True,
        pc_options=None,
        reorder=False,
        use_abs_det=False,
    ):
        self._C1 = float(params["C1"])
        self._D1 = float(params["D1"])
        self._use_abs_det = bool(use_abs_det)

        near_nullspace_vecs = None
        if use_near_nullspace:
            kernel = np.asarray(params["elastic_kernel"], dtype=np.float64)
            near_nullspace_vecs = [kernel[:, i] for i in range(kernel.shape[1])]

        super().__init__(
            freedofs=np.asarray(params["freedofs"]),
            elems=np.asarray(params["elems"]),
            u_0=np.asarray(params["u_0"]),
            comm=comm,
            adjacency=adjacency,
            f=None,
            elem_data={
                "dphix": np.asarray(params["dphix"]),
                "dphiy": np.asarray(params["dphiy"]),
                "dphiz": np.asarray(params["dphiz"]),
                "vol": np.asarray(params["vol"]),
            },
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_max_it=ksp_max_it,
            near_nullspace_vecs=near_nullspace_vecs,
            pc_options=pc_options,
            reorder=reorder,
            ownership_block_size=3,
        )


class LocalColoringAssembler(_HyperElasticityMixin, LocalColoringAssemblerBase):
    """Per-rank local-coloring HE assembler (preferred for performance)."""

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-1,
        ksp_type="gmres",
        pc_type="hypre",
        ksp_max_it=30,
        use_near_nullspace=True,
        pc_options=None,
        reorder=False,
        use_abs_det=False,
        hvp_eval_mode="sequential",
    ):
        self._C1 = float(params["C1"])
        self._D1 = float(params["D1"])
        self._use_abs_det = bool(use_abs_det)
        self._hvp_eval_mode = str(hvp_eval_mode)

        near_nullspace_vecs = None
        if use_near_nullspace:
            kernel = np.asarray(params["elastic_kernel"], dtype=np.float64)
            near_nullspace_vecs = [kernel[:, i] for i in range(kernel.shape[1])]

        super().__init__(
            freedofs=np.asarray(params["freedofs"]),
            elems=np.asarray(params["elems"]),
            u_0=np.asarray(params["u_0"]),
            comm=comm,
            adjacency=adjacency,
            f=None,
            elem_data={
                "dphix": np.asarray(params["dphix"]),
                "dphiy": np.asarray(params["dphiy"]),
                "dphiz": np.asarray(params["dphiz"]),
                "vol": np.asarray(params["vol"]),
            },
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_max_it=ksp_max_it,
            near_nullspace_vecs=near_nullspace_vecs,
            pc_options=pc_options,
            reorder=reorder,
            ownership_block_size=3,
        )
