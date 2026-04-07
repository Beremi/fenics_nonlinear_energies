"""Reordered PETSc element assembler for 3D heterogeneous slope-stability."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import resource
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


_PLASTIC_ELEMENT_ENERGY_BATCH = jax.jit(
    jax.vmap(
        element_energy_3d,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
)
_ELASTIC_ELEMENT_ENERGY_BATCH = jax.jit(
    jax.vmap(
        elastic_element_energy_3d,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    )
)
_PLASTIC_ELEMENT_HESSIAN_BATCH = jax.jit(
    jax.vmap(
        jax.hessian(element_energy_3d),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
)
_ELASTIC_ELEMENT_HESSIAN_BATCH = jax.jit(
    jax.vmap(
        jax.hessian(elastic_element_energy_3d),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    )
)


def _plastic_local_full_energy(
    v_local,
    elems,
    dphix,
    dphiy,
    dphiz,
    quad_weight,
    c_bar_q,
    sin_phi_q,
    shear_q,
    bulk_q,
    lame_q,
):
    v_elem = v_local[elems]
    return jnp.sum(
        _PLASTIC_ELEMENT_ENERGY_BATCH(
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
    )


def _plastic_weighted_energy(
    v_local,
    elems,
    dphix,
    dphiy,
    dphiz,
    quad_weight,
    c_bar_q,
    sin_phi_q,
    shear_q,
    bulk_q,
    lame_q,
    energy_weights,
):
    v_elem = v_local[elems]
    return jnp.sum(
        _PLASTIC_ELEMENT_ENERGY_BATCH(
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
        * energy_weights
    )


def _elastic_local_full_energy(
    v_local,
    elems,
    dphix,
    dphiy,
    dphiz,
    quad_weight,
    shear_q,
    bulk_q,
    lame_q,
):
    v_elem = v_local[elems]
    return jnp.sum(
        _ELASTIC_ELEMENT_ENERGY_BATCH(
            v_elem,
            dphix,
            dphiy,
            dphiz,
            quad_weight,
            shear_q,
            bulk_q,
            lame_q,
        )
    )


def _elastic_weighted_energy(
    v_local,
    elems,
    dphix,
    dphiy,
    dphiz,
    quad_weight,
    shear_q,
    bulk_q,
    lame_q,
    energy_weights,
):
    v_elem = v_local[elems]
    return jnp.sum(
        _ELASTIC_ELEMENT_ENERGY_BATCH(
            v_elem,
            dphix,
            dphiy,
            dphiz,
            quad_weight,
            shear_q,
            bulk_q,
            lame_q,
        )
        * energy_weights
    )


_PLASTIC_LOCAL_GRAD = jax.jit(jax.grad(_plastic_local_full_energy, argnums=0))
_ELASTIC_LOCAL_GRAD = jax.jit(jax.grad(_elastic_local_full_energy, argnums=0))
_PLASTIC_WEIGHTED_ENERGY = jax.jit(_plastic_weighted_energy)
_ELASTIC_WEIGHTED_ENERGY = jax.jit(_elastic_weighted_energy)


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
        assembly_backend="coo",
        petsc_log_events=False,
        jax_trace_dir="",
    ):
        self.constitutive_mode = "plastic"
        self._kernel_cache: dict[str, dict[str, object]] = {}
        self.jax_trace_dir = str(jax_trace_dir or "").strip()
        self._jax_trace_enabled = bool(self.jax_trace_dir)
        self.p4_hessian_chunk_size = max(1, int(p4_hessian_chunk_size))
        self.p4_chunk_autotune_meta: dict[str, object] | None = None
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
            assembly_backend=assembly_backend,
            petsc_log_events=petsc_log_events,
        )

    def _resolve_assembly_backend(self, backend: str) -> str:
        backend_name = str(backend or "coo")
        if backend_name not in {"coo", "coo_local", "blocked_local"}:
            raise ValueError(f"Unsupported 3D assembly backend {backend_name!r}")
        if backend_name == "blocked_local":
            n_free = int(np.asarray(self.params["freedofs"], dtype=np.int64).size)
            if int(getattr(self, "block_size", 1)) != 3 or (n_free % 3) != 0:
                raise ValueError(
                    "assembly_backend='blocked_local' requires a full 3-block free-DOF layout; "
                    "the current reduced system has component-wise constraints and is not block-compatible"
                )
        return backend_name

    def _chunk_trace(self, name: str):
        if not self._jax_trace_enabled:
            return nullcontext()
        return jax.profiler.TraceAnnotation(str(name))

    @staticmethod
    def _rss_gib() -> float:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(int(getattr(usage, "ru_maxrss", 0)) * 1024) / float(1024**3)

    def autotune_p4_hessian_chunk_size(
        self,
        *,
        u_owned: np.ndarray,
        candidates: tuple[int, ...] = (32, 64, 128, 256),
        rss_budget_gib: float = 64.0,
    ) -> dict[str, object] | None:
        if int(self.params.get("element_degree", 2)) != 4:
            return None
        if self.assembly_backend == "blocked_local":
            return None
        original_chunk = int(self.p4_hessian_chunk_size)
        original_iter_timings = list(self.iter_timings)
        original_callback_stats = {
            key: dict(value) for key, value in self._callback_stats.items()
        }
        results: list[dict[str, object]] = []
        try:
            for cand in tuple(int(v) for v in candidates):
                self.p4_hessian_chunk_size = max(1, int(cand))
                rss_before = self._rss_gib()
                t0 = time.perf_counter()
                timing = self.assemble_hessian_with_mode(
                    np.asarray(u_owned, dtype=np.float64),
                    constitutive_mode=self.constitutive_mode,
                )
                total = float(time.perf_counter() - t0)
                rss_after = self._rss_gib()
                results.append(
                    {
                        "chunk_size": int(cand),
                        "assemble_total": float(total),
                        "hvp_compute": float(timing.get("hvp_compute", 0.0)),
                        "extraction": float(timing.get("extraction", 0.0)),
                        "coo_assembly": float(timing.get("coo_assembly", 0.0)),
                        "rss_hwm_gib": float(rss_after),
                        "rss_delta_gib": float(max(0.0, rss_after - rss_before)),
                        "within_budget": bool(float(rss_after) <= float(rss_budget_gib)),
                    }
                )
            feasible = [rec for rec in results if bool(rec["within_budget"])]
            best = min(feasible or results, key=lambda rec: float(rec["assemble_total"]))
            self.p4_hessian_chunk_size = int(best["chunk_size"])
            self.p4_chunk_autotune_meta = {
                "enabled": True,
                "rss_budget_gib": float(rss_budget_gib),
                "candidates": list(results),
                "selected_chunk_size": int(best["chunk_size"]),
            }
            return dict(self.p4_chunk_autotune_meta)
        finally:
            self.iter_timings = list(original_iter_timings)
            self._callback_stats = {
                key: dict(value) for key, value in original_callback_stats.items()
            }
            if self.p4_chunk_autotune_meta is None:
                self.p4_hessian_chunk_size = int(original_chunk)

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
            def energy_fn(v_local):
                return _ELASTIC_WEIGHTED_ENERGY(
                    v_local,
                    elems,
                    dphix,
                    dphiy,
                    dphiz,
                    quad_weight,
                    shear_q,
                    bulk_q,
                    lame_q,
                    energy_weights,
                )

            def local_grad_fn(v_local):
                return _ELASTIC_LOCAL_GRAD(
                    v_local,
                    elems,
                    dphix,
                    dphiy,
                    dphiz,
                    quad_weight,
                    shear_q,
                    bulk_q,
                    lame_q,
                )

            def grad_local(v_local):
                return _ELASTIC_LOCAL_GRAD(
                    v_local,
                    elems,
                    dphix,
                    dphiy,
                    dphiz,
                    quad_weight,
                    shear_q,
                    bulk_q,
                    lame_q,
                )

        elif mode == "plastic":
            def energy_fn(v_local):
                return _PLASTIC_WEIGHTED_ENERGY(
                    v_local,
                    elems,
                    dphix,
                    dphiy,
                    dphiz,
                    quad_weight,
                    c_bar_q,
                    sin_phi_q,
                    shear_q,
                    bulk_q,
                    lame_q,
                    energy_weights,
                )

            def local_grad_fn(v_local):
                return _PLASTIC_LOCAL_GRAD(
                    v_local,
                    elems,
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

            def grad_local(v_local):
                return _PLASTIC_LOCAL_GRAD(
                    v_local,
                    elems,
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

            def elem_hess_chunk_fn(v_local, start: int, stop: int):
                chunk = slice(int(start), int(stop))
                v_elem = v_local[elems[chunk]]
                return chunked_vmapped_element_hessian_3d(
                    v_elem,
                    dphix[chunk],
                    dphiy[chunk],
                    dphiz[chunk],
                    quad_weight[chunk],
                    c_bar_q[chunk],
                    sin_phi_q[chunk],
                    shear_q[chunk],
                    bulk_q[chunk],
                    lame_q[chunk],
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

            def elem_hess_chunk_fn(v_local, start: int, stop: int):
                chunk = slice(int(start), int(stop))
                v_elem = v_local[elems[chunk]]
                return chunked_vmapped_elastic_element_hessian_3d(
                    v_elem,
                    dphix[chunk],
                    dphiy[chunk],
                    dphiz[chunk],
                    quad_weight[chunk],
                    shear_q[chunk],
                    bulk_q[chunk],
                    lame_q[chunk],
                    chunk_size=self.p4_hessian_chunk_size,
                )

        else:
            def elem_hess_fn(v_local):
                v_elem = v_local[elems]
                if mode == "elastic":
                    return _ELASTIC_ELEMENT_HESSIAN_BATCH(
                        v_elem,
                        dphix,
                        dphiy,
                        dphiz,
                        quad_weight,
                        shear_q,
                        bulk_q,
                        lame_q,
                    )
                return _PLASTIC_ELEMENT_HESSIAN_BATCH(
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

            def elem_hess_chunk_fn(v_local, start: int, stop: int):
                chunk = slice(int(start), int(stop))
                v_elem = v_local[elems[chunk]]
                if mode == "elastic":
                    return _ELASTIC_ELEMENT_HESSIAN_BATCH(
                        v_elem,
                        dphix[chunk],
                        dphiy[chunk],
                        dphiz[chunk],
                        quad_weight[chunk],
                        shear_q[chunk],
                        bulk_q[chunk],
                        lame_q[chunk],
                    )
                return _PLASTIC_ELEMENT_HESSIAN_BATCH(
                    v_elem,
                    dphix[chunk],
                    dphiy[chunk],
                    dphiz[chunk],
                    quad_weight[chunk],
                    c_bar_q[chunk],
                    sin_phi_q[chunk],
                    shear_q[chunk],
                    bulk_q[chunk],
                    lame_q[chunk],
                )

        return {
            "elems": elems,
            "degree": int(degree),
            "energy_fn": energy_fn,
            "local_grad_fn": local_grad_fn,
            "elem_hess_fn": elem_hess_fn,
            "elem_hess_chunk_fn": elem_hess_chunk_fn,
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

    def _needs_prebuilt_hessian_scatter(self) -> bool:
        return int(self.params.get("element_degree", 2)) != 4

    def _warmup_hessian(self, v_local: np.ndarray) -> None:
        kernels = self._get_local_element_kernels(self.constitutive_mode)
        chunk_fn = kernels.get("elem_hess_chunk_fn")
        n_local_elem = int(self.local_data.elems_local_np.shape[0])
        if int(self.params.get("element_degree", 2)) == 4 and chunk_fn is not None and n_local_elem > 0:
            stop = min(max(1, int(self.p4_hessian_chunk_size)), n_local_elem)
            chunk_fn(jnp.asarray(v_local), 0, stop).block_until_ready()
            return
        kernels["elem_hess_fn"](jnp.asarray(v_local)).block_until_ready()

    def _chunk_scatter_pattern(
        self,
        start: int,
        stop: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.assembly_backend == "coo_local":
            elems_lookup = np.asarray(
                self._local_elems_free[int(start) : int(stop)],
                dtype=np.int64,
            )
            rows = elems_lookup[:, :, None]
            cols = elems_lookup[:, None, :]
            valid = (rows >= 0) & (cols >= 0) & self._local_owned_free_mask[rows]
            key_base = np.int64(max(1, int(self._local_free_global_indices.size)))
            key_table = self._local_owned_keys_sorted
            pos_table = self._local_owned_pos_sorted
        else:
            elems_lookup = np.asarray(
                self.local_data.elems_reordered[int(start) : int(stop)], dtype=np.int64
            )
            rows = elems_lookup[:, :, None]
            cols = elems_lookup[:, None, :]
            valid = (rows >= self.layout.lo) & (rows < self.layout.hi) & (cols >= 0)
            key_base = np.int64(self.layout.n_free)
            key_table = self.layout.owned_keys_sorted
            pos_table = self.layout.owned_pos_sorted
        vi = np.where(valid)
        if vi[0].size == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, empty
        row_vals = elems_lookup[vi[0], vi[1]]
        col_vals = elems_lookup[vi[0], vi[2]]
        keys = row_vals.astype(np.int64) * key_base + col_vals.astype(np.int64)
        key_pos = np.searchsorted(key_table, keys)
        if np.any(key_pos >= key_table.size):
            raise RuntimeError("Chunk scatter lookup exceeded owned COO pattern size")
        matched = key_table[key_pos]
        if not np.array_equal(matched, keys):
            raise RuntimeError("Chunk scatter lookup found mismatched owned COO entries")
        positions = np.asarray(pos_table[key_pos], dtype=np.int64)
        return (
            np.asarray(vi[0], dtype=np.int64),
            np.asarray(vi[1], dtype=np.int64),
            np.asarray(vi[2], dtype=np.int64),
            positions,
        )

    def _accumulate_owned_contrib(
        self,
        owned_vals: np.ndarray,
        positions: np.ndarray,
        contrib: np.ndarray,
    ) -> None:
        pos = np.asarray(positions, dtype=np.int64).ravel()
        if pos.size == 0:
            return
        vals = np.asarray(contrib, dtype=np.float64).ravel()
        np.add.at(owned_vals, pos, vals)

    def _assemble_hessian_with_elem_hess(
        self,
        u_owned,
        elem_hess_jit,
        *,
        assembly_mode: str,
        elem_hess_chunk_fn=None,
        hessian_chunk_elems: int | None = None,
    ):
        timings = {}
        t_total = time.perf_counter()
        with self._petsc_event("slope3d:overlap_exchange"):
            v_local, exchange = self._owned_to_local(
                np.asarray(u_owned, dtype=np.float64),
                zero_dirichlet=False,
            )

        owned_vals = self._reset_owned_hessian_values()
        chunk_elems = int(hessian_chunk_elems or 0)
        if elem_hess_chunk_fn is not None and chunk_elems > 0:
            v_local_jax = jnp.asarray(v_local)
            n_local_elem = int(self.local_data.elems_local_np.shape[0])
            timings["elem_hessian_compute"] = 0.0
            timings["scatter"] = 0.0
            timings["chunk_count"] = int((n_local_elem + chunk_elems - 1) // chunk_elems)
            timings["chunk_size"] = int(chunk_elems)
            chunk_rows: list[int] = []
            chunk_kernel_times: list[float] = []
            chunk_scatter_times: list[float] = []
            for start in range(0, n_local_elem, chunk_elems):
                stop = min(start + chunk_elems, n_local_elem)
                t0 = time.perf_counter()
                with self._petsc_event("slope3d:hessian_kernel"):
                    with self._chunk_trace("slope3d_p4_chunk_hessian"):
                        elem_hess_chunk = np.asarray(
                            elem_hess_chunk_fn(v_local_jax, start, stop).block_until_ready()
                        )
                kernel_dt = float(time.perf_counter() - t0)
                timings["elem_hessian_compute"] += kernel_dt
                t0 = time.perf_counter()
                with self._petsc_event("slope3d:hessian_scatter"):
                    with self._chunk_trace("slope3d_p4_chunk_scatter"):
                        hess_e, hess_i, hess_j, hess_positions = self._chunk_scatter_pattern(
                            start, stop
                        )
                        if hess_e.size != 0:
                            contrib = elem_hess_chunk[
                                hess_e,
                                hess_i,
                                hess_j,
                            ]
                            self._accumulate_owned_contrib(
                                owned_vals,
                                hess_positions,
                                contrib,
                            )
                scatter_dt = float(time.perf_counter() - t0)
                if hess_e.size == 0:
                    timings["scatter"] += scatter_dt
                    chunk_rows.append(0)
                    chunk_kernel_times.append(kernel_dt)
                    chunk_scatter_times.append(scatter_dt)
                    continue
                timings["scatter"] += scatter_dt
                chunk_rows.append(int(hess_positions.size))
                chunk_kernel_times.append(kernel_dt)
                chunk_scatter_times.append(scatter_dt)
            timings["chunk_rows_max"] = int(max(chunk_rows) if chunk_rows else 0)
            timings["chunk_kernel_time_max"] = float(max(chunk_kernel_times) if chunk_kernel_times else 0.0)
            timings["chunk_scatter_time_max"] = float(max(chunk_scatter_times) if chunk_scatter_times else 0.0)
        else:
            if (
                self._scatter.hess_e is None
                or self._scatter.hess_i is None
                or self._scatter.hess_j is None
                or self._scatter.hess_positions is None
            ):
                raise RuntimeError(
                    "Full-element Hessian assembly requires prebuilt Hessian scatter data"
                )
            t0 = time.perf_counter()
            with self._petsc_event("slope3d:hessian_kernel"):
                elem_hess = np.asarray(elem_hess_jit(jnp.asarray(v_local)).block_until_ready())
            timings["elem_hessian_compute"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            with self._petsc_event("slope3d:hessian_scatter"):
                contrib = elem_hess[self._scatter.hess_e, self._scatter.hess_i, self._scatter.hess_j]
                self._accumulate_owned_contrib(
                    owned_vals,
                    self._scatter.hess_positions,
                    contrib,
                )
            timings["scatter"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with self._petsc_event("slope3d:hessian_matrix_insert"):
            self._insert_owned_hessian_values(owned_vals)
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

    def assemble_hessian_element(self, u_owned):
        if int(self.params.get("element_degree", 2)) != 4:
            return super().assemble_hessian_element(u_owned)
        return self.assemble_hessian_with_mode(u_owned, constitutive_mode=self.constitutive_mode)

    def assemble_hessian_with_mode(self, u_owned, *, constitutive_mode: str):
        kernels = self._get_local_element_kernels(str(constitutive_mode))
        return self._assemble_hessian_with_elem_hess(
            u_owned,
            kernels["elem_hess_fn"],
            assembly_mode=f"element_overlap_{constitutive_mode}",
            elem_hess_chunk_fn=(
                kernels["elem_hess_chunk_fn"] if int(kernels["degree"]) == 4 else None
            ),
            hessian_chunk_elems=(
                int(self.p4_hessian_chunk_size) if int(kernels["degree"]) == 4 else 0
            ),
        )

    def _build_rhs_owned(self) -> np.ndarray:
        force = np.asarray(self.params["force"], dtype=np.float64)
        freedofs = np.asarray(self.params["freedofs"], dtype=np.int64)
        rhs_free = force[freedofs]
        rhs_reordered = rhs_free[self.layout.perm]
        return np.asarray(rhs_reordered[self.layout.lo : self.layout.hi], dtype=np.float64)
