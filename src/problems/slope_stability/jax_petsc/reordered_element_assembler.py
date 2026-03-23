"""Reordered PETSc element assembler for experimental slope-stability."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from petsc4py import PETSc
from scipy import sparse
from scipy.sparse.linalg import splu

from src.core.petsc.reordered_element_base import ReorderedElementAssemblerBase
from src.problems.slope_stability.jax.jax_energy import (
    elastic_element_energy,
    element_energy,
)


config.update("jax_enable_x64", True)


class _MatrixFreeHessianContext:
    """PETSc Python-matrix context for cached Hessian-vector products."""

    def __init__(self, assembler: "SlopeStabilityReorderedElementAssembler", mode: str):
        self.assembler = assembler
        self.mode = str(mode)

    def mult(self, mat, x, y):
        del mat
        self.assembler.matrix_free_mult(self.mode, x, y)

    def multTranspose(self, mat, x, y):
        del mat
        self.assembler.matrix_free_mult(self.mode, x, y)

    def createVecs(self, mat):
        del mat
        return self.assembler.create_vec(), self.assembler.create_vec()

    def getDiagonal(self, mat, d):
        del mat
        self.assembler.matrix_free_get_diagonal(self.mode, d)

    def duplicate(self, mat, op):
        del mat, op
        return self.assembler._create_matrix_free_operator(self.mode)


class _MatrixFreePythonPCContext:
    """PETSc Python preconditioner context backed by overlap-local P4 data."""

    def __init__(self, assembler: "SlopeStabilityReorderedElementAssembler", variant: str):
        self.assembler = assembler
        self.variant = str(variant)

    def setUp(self, pc):
        del pc
        self.assembler.prepare_matrix_free_python_pc(self.variant)

    def apply(self, pc, b, x):
        del pc
        self.assembler.apply_matrix_free_python_pc(self.variant, b, x)

    def view(self, pc, viewer):
        del pc
        viewer.printfASCII(
            f"Matrix-free Python PC variant: {self.variant}\n"
        )


class SlopeStabilityReorderedElementAssembler(ReorderedElementAssemblerBase):
    """P2 plane-strain slope-stability element assembler."""

    block_size = 2
    coordinate_key = "nodes"
    dirichlet_key = "u_0"
    local_elem_data_keys = ("elem_B", "quad_weight", "eps_p_old")
    near_nullspace_key = "elastic_kernel"

    def __init__(
        self,
        params,
        comm,
        adjacency,
        *,
        ksp_rtol=1e-1,
        ksp_type="stcg",
        pc_type="gamg",
        ksp_max_it=30,
        use_near_nullspace=True,
        pc_options=None,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
        perm_override=None,
        block_size_override=None,
        constitutive_mode="mc",
        distribution_strategy=None,
        reuse_hessian_value_buffers=True,
    ):
        self.constitutive_mode = str(constitutive_mode)
        self._kernel_cache: dict[str, dict[str, object]] = {}
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
        self._matrix_free_ops: dict[str, PETSc.Mat] = {}
        self._matrix_free_linearization_mode: str | None = None
        self._matrix_free_linearization = None
        self._matrix_free_stats: dict[str, object] = {}
        self._matrix_free_state_id = 0
        self._matrix_free_v_local: np.ndarray | None = None
        self._matrix_free_elem_hess_cache = None
        self._matrix_free_diag_cache: dict[str, object] | None = None
        self._matrix_free_patch_cache: dict[str, object] = {}
        local_reord = np.asarray(
            self.layout.total_to_free_reord[self.local_data.local_total_nodes],
            dtype=np.int64,
        )
        self._local_reord = local_reord
        self._local_free_pos = np.where(local_reord >= 0)[0].astype(np.int64)
        self._local_free_reord = np.asarray(
            local_reord[self._local_free_pos], dtype=np.int64
        )
        self._local_pos_to_free_index = np.full(len(local_reord), -1, dtype=np.int64)
        self._local_pos_to_free_index[self._local_free_pos] = np.arange(
            len(self._local_free_pos), dtype=np.int64
        )
        self._owned_free_indices = np.asarray(
            self._local_pos_to_free_index[self._scatter.owned_local_pos], dtype=np.int64
        )
        if np.any(self._owned_free_indices < 0):
            raise RuntimeError("Owned local rows are missing from the free overlap set")

    def _build_local_element_kernels(self, constitutive_mode: str) -> dict[str, object]:
        elems = jnp.asarray(self.local_data.elems_local_np, dtype=jnp.int32)
        elem_B = jnp.asarray(self.local_data.local_elem_data["elem_B"], dtype=jnp.float64)
        quad_weight = jnp.asarray(
            self.local_data.local_elem_data["quad_weight"], dtype=jnp.float64
        )
        eps_p_old = jnp.asarray(
            self.local_data.local_elem_data["eps_p_old"], dtype=jnp.float64
        )
        energy_weights = jnp.asarray(self.local_data.energy_weights, dtype=jnp.float64)

        E = float(self.params["E"])
        nu = float(self.params["nu"])
        phi_deg = float(self.params["phi_deg"])
        cohesion = float(self.params["cohesion"])
        reg = float(self.params.get("reg", 1.0e-12))

        constitutive_mode = str(constitutive_mode)
        if constitutive_mode == "mc":
            def element_energy_local(v_e, elem_B_e, quad_weight_e, eps_p_old_e):
                return element_energy(
                    v_e,
                    elem_B_e,
                    quad_weight_e,
                    eps_p_old_e,
                    E,
                    nu,
                    phi_deg,
                    cohesion,
                    reg,
                )
        elif constitutive_mode == "elastic":
            def element_energy_local(v_e, elem_B_e, quad_weight_e, eps_p_old_e):
                return elastic_element_energy(
                    v_e,
                    elem_B_e,
                    quad_weight_e,
                    eps_p_old_e,
                    E,
                    nu,
                    reg,
                )
        else:
            raise ValueError(f"Unsupported constitutive_mode {constitutive_mode!r}")

        hess_elem = jax.vmap(
            jax.hessian(element_energy_local),
            in_axes=(0, 0, 0, 0),
        )

        def local_full_energy(v_local):
            v_e = v_local[elems]
            e = jax.vmap(element_energy_local, in_axes=(0, 0, 0, 0))(
                v_e, elem_B, quad_weight, eps_p_old
            )
            return jnp.sum(e)

        grad_local = jax.grad(local_full_energy)
        elem_grad = jax.vmap(
            jax.grad(element_energy_local),
            in_axes=(0, 0, 0, 0),
        )

        @jax.jit
        def energy_fn(v_local):
            v_e = v_local[elems]
            e = jax.vmap(element_energy_local, in_axes=(0, 0, 0, 0))(
                v_e, elem_B, quad_weight, eps_p_old
            )
            return jnp.sum(e * energy_weights)

        @jax.jit
        def local_grad_fn(v_local):
            return grad_local(v_local)

        @jax.jit
        def elem_hess_fn(v_local):
            v_e = v_local[elems]
            return hess_elem(v_e, elem_B, quad_weight, eps_p_old)

        @jax.jit
        def elem_grad_fn(v_elem):
            return elem_grad(v_elem, elem_B, quad_weight, eps_p_old)

        return {
            "elems": elems,
            "energy_fn": energy_fn,
            "local_grad_fn": local_grad_fn,
            "elem_hess_fn": elem_hess_fn,
            "grad_local": grad_local,
            "elem_grad_fn": elem_grad_fn,
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
        self._local_elems_jax = kernels["elems"]
        self._elem_grad_batch_jit = kernels["elem_grad_fn"]
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

    def _create_matrix_free_operator(self, mode: str) -> PETSc.Mat:
        operator = PETSc.Mat().createPython(
            (
                (self.layout.hi - self.layout.lo, self.layout.n_free),
                (self.layout.hi - self.layout.lo, self.layout.n_free),
            ),
            context=_MatrixFreeHessianContext(self, mode),
            comm=self.comm,
        )
        if int(self.block_size) > 1:
            operator.setBlockSize(int(self.block_size))
        if self._nullspace is not None:
            operator.setNearNullSpace(self._nullspace)
        operator.setUp()
        return operator

    def get_matrix_free_operator(self, mode: str) -> PETSc.Mat:
        mode = str(mode)
        if mode not in self._matrix_free_ops:
            self._matrix_free_ops[mode] = self._create_matrix_free_operator(mode)
        return self._matrix_free_ops[mode]

    def prepare_matrix_free_operator(self, u_owned: np.ndarray, *, mode: str) -> PETSc.Mat:
        mode = str(mode)
        if mode not in {"matfree_element", "matfree_overlap"}:
            raise ValueError(f"Unsupported matrix-free operator mode {mode!r}")

        t_total = time.perf_counter()
        v_local, exchange = self._owned_to_local(
            np.asarray(u_owned, dtype=np.float64),
            zero_dirichlet=False,
        )
        v_local_j = jnp.asarray(v_local, dtype=jnp.float64)

        t_lin0 = time.perf_counter()
        if mode == "matfree_overlap":
            _, linearized = jax.linearize(self._grad_jit, v_local_j)
            # Warm one call so the first Krylov iteration does not pay the trace cost.
            linearized(jnp.zeros_like(v_local_j)).block_until_ready()
        else:
            v_elem = v_local_j[self._local_elems_jax]
            _, linearized = jax.linearize(self._elem_grad_batch_jit, v_elem)
            linearized(jnp.zeros_like(v_elem)).block_until_ready()
        t_linearize = time.perf_counter() - t_lin0

        self._matrix_free_linearization_mode = mode
        self._matrix_free_linearization = linearized
        self._matrix_free_state_id += 1
        self._matrix_free_v_local = np.asarray(v_local, dtype=np.float64)
        self._matrix_free_elem_hess_cache = None
        self._matrix_free_diag_cache = None
        self._matrix_free_patch_cache.clear()
        self._matrix_free_stats = {
            "mode": mode,
            "prepare_allgatherv": float(exchange["allgatherv"]),
            "prepare_ghost_exchange": float(exchange["ghost_exchange"]),
            "prepare_build_v_local": float(exchange["build_v_local"]),
            "prepare_linearize": float(t_linearize),
            "prepare_total": float(time.perf_counter() - t_total),
            "diagonal_source": "not_requested",
            "diagonal_prepare_total": 0.0,
            "python_pc_variant": "none",
            "python_pc_prepare_total": 0.0,
            "python_pc_apply_calls": 0,
            "python_pc_apply_total": 0.0,
            "mult_calls": 0,
            "mult_allgatherv": 0.0,
            "mult_ghost_exchange": 0.0,
            "mult_build_v_local": 0.0,
            "mult_apply": 0.0,
            "mult_scatter": 0.0,
            "mult_total": 0.0,
        }
        operator = self.get_matrix_free_operator(mode)
        operator.stateIncrease()
        return operator

    def matrix_free_summary(self) -> dict[str, object]:
        return dict(self._matrix_free_stats)

    def make_matrix_free_python_pc_context(self, variant: str) -> _MatrixFreePythonPCContext:
        return _MatrixFreePythonPCContext(self, variant)

    def _require_matrix_free_state(self, mode: str | None = None) -> None:
        if self._matrix_free_linearization is None or self._matrix_free_v_local is None:
            raise RuntimeError("Matrix-free state has not been prepared")
        if mode is not None and self._matrix_free_linearization_mode != str(mode):
            raise RuntimeError(
                f"Matrix-free state {self._matrix_free_linearization_mode!r} "
                f"does not match requested mode {mode!r}"
            )

    def _matrix_free_element_hessian(self) -> np.ndarray:
        self._require_matrix_free_state()
        if self._matrix_free_elem_hess_cache is None:
            elem_hess = self._elem_hess_jit(
                jnp.asarray(self._matrix_free_v_local, dtype=jnp.float64)
            ).block_until_ready()
            self._matrix_free_elem_hess_cache = np.asarray(elem_hess, dtype=np.float64)
        return np.asarray(self._matrix_free_elem_hess_cache, dtype=np.float64)

    def _build_matrix_free_diagonal(self, mode: str) -> np.ndarray:
        self._require_matrix_free_state(mode)
        t0 = time.perf_counter()
        elem_hess = self._matrix_free_element_hessian()
        contrib = elem_hess[self._scatter.vec_e, self._scatter.vec_i, self._scatter.vec_i]
        owned_diag = np.zeros(self.layout.hi - self.layout.lo, dtype=np.float64)
        np.add.at(owned_diag, self._scatter.vec_positions, contrib)
        self._matrix_free_diag_cache = {
            "state_id": int(self._matrix_free_state_id),
            "owned_diag": np.asarray(owned_diag, dtype=np.float64),
        }
        self._matrix_free_stats["diagonal_source"] = "element_hessian_diagonal"
        self._matrix_free_stats["diagonal_prepare_total"] = float(
            time.perf_counter() - t0
        )
        return np.asarray(owned_diag, dtype=np.float64)

    def matrix_free_get_diagonal(self, mode: str, diag_vec) -> None:
        mode = str(mode)
        self._require_matrix_free_state(mode)
        cache = self._matrix_free_diag_cache
        if cache is None or int(cache.get("state_id", -1)) != int(self._matrix_free_state_id):
            owned_diag = self._build_matrix_free_diagonal(mode)
        else:
            owned_diag = np.asarray(cache["owned_diag"], dtype=np.float64)
        diag_arr = diag_vec.getArray()
        diag_arr[:] = owned_diag

    def _build_local_free_operator(self) -> sparse.csr_matrix:
        elem_hess = self._matrix_free_element_hessian()
        elems_local = np.asarray(self.local_data.elems_local_np, dtype=np.int64)
        rows_local = elems_local[:, :, None]
        cols_local = elems_local[:, None, :]
        rows_free = self._local_pos_to_free_index[rows_local]
        cols_free = self._local_pos_to_free_index[cols_local]
        valid = (rows_free >= 0) & (cols_free >= 0)
        vi = np.where(valid)
        data = elem_hess[vi[0], vi[1], vi[2]]
        matrix = sparse.coo_matrix(
            (
                data,
                (
                    np.asarray(rows_free[vi[0], vi[1], 0], dtype=np.int64),
                    np.asarray(cols_free[vi[0], 0, vi[2]], dtype=np.int64),
                ),
            ),
            shape=(len(self._local_free_pos), len(self._local_free_pos)),
        ).tocsr()
        return matrix

    def prepare_matrix_free_python_pc(self, variant: str) -> None:
        variant = str(variant)
        self._require_matrix_free_state()
        cache = self._matrix_free_patch_cache.get(variant)
        if cache is not None and int(cache.get("state_id", -1)) == int(self._matrix_free_state_id):
            return
        t0 = time.perf_counter()
        if variant != "overlap_lu":
            raise ValueError(f"Unsupported matrix-free python PC variant {variant!r}")
        local_operator = self._build_local_free_operator().tocsc()
        factor = splu(local_operator)
        self._matrix_free_patch_cache[variant] = {
            "state_id": int(self._matrix_free_state_id),
            "factor": factor,
            "shape": tuple(local_operator.shape),
        }
        self._matrix_free_stats["python_pc_variant"] = variant
        self._matrix_free_stats["python_pc_prepare_total"] = float(
            time.perf_counter() - t0
        )

    def apply_matrix_free_python_pc(self, variant: str, b, x) -> None:
        variant = str(variant)
        self.prepare_matrix_free_python_pc(variant)
        cache = self._matrix_free_patch_cache[variant]
        factor = cache["factor"]
        t0 = time.perf_counter()
        local_rhs, _ = self._owned_to_local(
            np.asarray(b.getArray(readonly=True), dtype=np.float64),
            zero_dirichlet=True,
        )
        local_rhs = np.asarray(local_rhs[self._local_free_pos], dtype=np.float64)
        local_sol = np.asarray(factor.solve(local_rhs), dtype=np.float64)
        x_arr = x.getArray()
        x_arr[:] = local_sol[self._owned_free_indices]
        self._matrix_free_stats["python_pc_apply_calls"] = int(
            self._matrix_free_stats.get("python_pc_apply_calls", 0)
        ) + 1
        self._matrix_free_stats["python_pc_apply_total"] = float(
            self._matrix_free_stats.get("python_pc_apply_total", 0.0)
        ) + float(time.perf_counter() - t0)

    def matrix_free_mult(self, mode: str, x, y) -> None:
        mode = str(mode)
        if self._matrix_free_linearization is None or self._matrix_free_linearization_mode != mode:
            raise RuntimeError(
                f"Matrix-free operator {mode!r} is missing a matching linearization point"
            )

        t_total = time.perf_counter()
        tangent_local, exchange = self._owned_to_local(
            np.asarray(x.getArray(readonly=True), dtype=np.float64),
            zero_dirichlet=True,
        )
        tangent_local_j = jnp.asarray(tangent_local, dtype=jnp.float64)

        t_apply0 = time.perf_counter()
        if mode == "matfree_overlap":
            hvp_local = np.asarray(
                self._matrix_free_linearization(tangent_local_j).block_until_ready(),
                dtype=np.float64,
            )
            t_scatter0 = time.perf_counter()
            owned = np.asarray(hvp_local[self._scatter.owned_local_pos], dtype=np.float64)
            t_scatter = time.perf_counter() - t_scatter0
        elif mode == "matfree_element":
            tangent_elem = tangent_local_j[self._local_elems_jax]
            hvp_elem = np.asarray(
                self._matrix_free_linearization(tangent_elem).block_until_ready(),
                dtype=np.float64,
            )
            t_scatter0 = time.perf_counter()
            owned = np.zeros(self.layout.hi - self.layout.lo, dtype=np.float64)
            contrib = hvp_elem[self._scatter.vec_e, self._scatter.vec_i]
            np.add.at(owned, self._scatter.vec_positions, contrib)
            t_scatter = time.perf_counter() - t_scatter0
        else:
            raise ValueError(f"Unsupported matrix-free operator mode {mode!r}")
        t_apply = time.perf_counter() - t_apply0

        y_arr = y.getArray()
        y_arr[:] = owned

        stats = self._matrix_free_stats
        stats["mult_calls"] = int(stats.get("mult_calls", 0)) + 1
        stats["mult_allgatherv"] = float(stats.get("mult_allgatherv", 0.0)) + float(
            exchange["allgatherv"]
        )
        stats["mult_ghost_exchange"] = float(
            stats.get("mult_ghost_exchange", 0.0)
        ) + float(exchange["ghost_exchange"])
        stats["mult_build_v_local"] = float(stats.get("mult_build_v_local", 0.0)) + float(
            exchange["build_v_local"]
        )
        stats["mult_apply"] = float(stats.get("mult_apply", 0.0)) + float(t_apply)
        stats["mult_scatter"] = float(stats.get("mult_scatter", 0.0)) + float(t_scatter)
        stats["mult_total"] = float(stats.get("mult_total", 0.0)) + float(
            time.perf_counter() - t_total
        )

    def _build_rhs_owned(self) -> np.ndarray:
        rhs = np.asarray(self.params["force"], dtype=np.float64)
        freedofs = np.asarray(self.params["freedofs"], dtype=np.int64)
        rhs_free = rhs[freedofs]
        rhs_reordered = rhs_free[self.layout.perm]
        return np.asarray(rhs_reordered[self.layout.lo : self.layout.hi], dtype=np.float64)

    def cleanup(self):
        for operator in self._matrix_free_ops.values():
            operator.destroy()
        self._matrix_free_ops.clear()
        self._matrix_free_linearization = None
        self._matrix_free_stats = {}
        super().cleanup()
