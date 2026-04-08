"""Reordered PETSc element assembler for 3D heterogeneous slope-stability."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
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
    chunked_vmapped_elastic_element_constitutive_hessian_3d,
    chunked_vmapped_elastic_element_hessian_3d,
    chunked_vmapped_element_constitutive_hessian_3d,
    chunked_vmapped_element_hessian_3d,
    elastic_element_energy_3d,
    element_energy_3d,
    vmapped_elastic_element_constitutive_hessian_3d,
    vmapped_element_constitutive_hessian_3d,
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


@dataclass(frozen=True)
class P4ChunkScatterCacheEntry:
    """Compressed fixed scatter metadata for one P4 Hessian chunk."""

    flat_elem_idx: np.ndarray
    owned_pos: np.ndarray


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
        autodiff_tangent_mode="element",
        perm_override=None,
        block_size_override=None,
        distribution_strategy=None,
        reuse_hessian_value_buffers=True,
        p4_hessian_chunk_size=32,
        p4_chunk_scatter_cache="auto",
        p4_chunk_scatter_cache_max_gib=0.5,
        assembly_backend="coo",
        petsc_log_events=False,
        jax_trace_dir="",
        memory_guard_total_gib=None,
    ):
        self.constitutive_mode = "plastic"
        self._kernel_cache: dict[str, dict[str, object]] = {}
        self.jax_trace_dir = str(jax_trace_dir or "").strip()
        self._jax_trace_enabled = bool(self.jax_trace_dir)
        self.p4_hessian_chunk_size = max(1, int(p4_hessian_chunk_size))
        self.p4_chunk_autotune_meta: dict[str, object] | None = None
        self.autodiff_tangent_mode = str(autodiff_tangent_mode or "element").strip().lower()
        if self.autodiff_tangent_mode not in {"element", "constitutive"}:
            raise ValueError(
                "autodiff_tangent_mode must be one of {'element', 'constitutive'}"
            )
        self.p4_chunk_scatter_cache_requested = str(
            p4_chunk_scatter_cache or "auto"
        ).strip().lower()
        if self.p4_chunk_scatter_cache_requested not in {"auto", "on", "off"}:
            raise ValueError(
                "p4_chunk_scatter_cache must be one of {'auto', 'on', 'off'}"
            )
        self.p4_chunk_scatter_cache_max_gib = float(p4_chunk_scatter_cache_max_gib)
        self.p4_chunk_scatter_cache_meta: dict[str, object] = {
            "requested": str(self.p4_chunk_scatter_cache_requested),
            "enabled": False,
            "reason": "not_built",
            "chunk_size": 0,
            "chunk_count": 0,
            "bytes": 0,
            "gib": 0.0,
            "index_dtype": "",
            "build_time_s": 0.0,
            "max_chunk_entries": 0,
        }
        self._p4_chunk_scatter_cache: tuple[P4ChunkScatterCacheEntry, ...] | None = None
        self._p4_chunk_scatter_cache_chunk_size = 0
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
            memory_guard_total_gib=memory_guard_total_gib,
        )
        self._update_p4_chunk_scatter_memory_summary(0)

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
        original_cache_requested = str(self.p4_chunk_scatter_cache_requested)
        original_iter_timings = list(self.iter_timings)
        original_callback_stats = {
            key: dict(value) for key, value in self._callback_stats.items()
        }
        results: list[dict[str, object]] = []
        try:
            self._invalidate_p4_chunk_scatter_cache(reason="autotune")
            self.p4_chunk_scatter_cache_requested = "off"
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
            self.p4_chunk_scatter_cache_requested = str(original_cache_requested)
            if self.p4_chunk_autotune_meta is None:
                self.p4_hessian_chunk_size = int(original_chunk)
            self._invalidate_p4_chunk_scatter_cache(reason="not_built")

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
        tangent_mode = str(self.autodiff_tangent_mode)

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

        if degree == 4 and mode == "plastic" and tangent_mode == "constitutive":

            def elem_hess_fn(v_local):
                v_elem = v_local[elems]
                return chunked_vmapped_element_constitutive_hessian_3d(
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
                return chunked_vmapped_element_constitutive_hessian_3d(
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

        elif degree == 4 and mode == "plastic":

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

        elif degree == 4 and mode == "elastic" and tangent_mode == "constitutive":

            def elem_hess_fn(v_local):
                v_elem = v_local[elems]
                return chunked_vmapped_elastic_element_constitutive_hessian_3d(
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
                return chunked_vmapped_elastic_element_constitutive_hessian_3d(
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
                if mode == "elastic" and tangent_mode == "constitutive":
                    return vmapped_elastic_element_constitutive_hessian_3d(
                        v_elem,
                        dphix,
                        dphiy,
                        dphiz,
                        quad_weight,
                        shear_q,
                        bulk_q,
                        lame_q,
                    )
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
                if tangent_mode == "constitutive":
                    return vmapped_element_constitutive_hessian_3d(
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
                if mode == "elastic" and tangent_mode == "constitutive":
                    return vmapped_elastic_element_constitutive_hessian_3d(
                        v_elem,
                        dphix[chunk],
                        dphiy[chunk],
                        dphiz[chunk],
                        quad_weight[chunk],
                        shear_q[chunk],
                        bulk_q[chunk],
                        lame_q[chunk],
                    )
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
                if tangent_mode == "constitutive":
                    return vmapped_element_constitutive_hessian_3d(
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

    def _update_p4_chunk_scatter_memory_summary(self, cache_bytes: int) -> None:
        if not hasattr(self, "_memory_summary"):
            return
        bytes_count = int(max(0, int(cache_bytes)))
        gib = float(bytes_count) / float(1024.0**3)
        self._memory_summary["p4_chunk_scatter_cache_bytes"] = bytes_count
        self._memory_summary["p4_chunk_scatter_cache_gib"] = gib
        self._memory_summary["tracked_total_gib"] = (
            float(self._memory_summary.get("layout_gib", 0.0))
            + float(self._memory_summary.get("local_overlap_gib", 0.0))
            + float(self._memory_summary.get("scatter_gib", 0.0))
            + float(self._memory_summary.get("owned_hessian_values_gib", 0.0))
            + float(self._memory_summary.get("local_backend_gib", 0.0))
            + gib
        )

    def _invalidate_p4_chunk_scatter_cache(self, *, reason: str) -> None:
        self._p4_chunk_scatter_cache = None
        self._p4_chunk_scatter_cache_chunk_size = 0
        self.p4_chunk_scatter_cache_meta = {
            "requested": str(self.p4_chunk_scatter_cache_requested),
            "enabled": False,
            "reason": str(reason),
            "chunk_size": 0,
            "chunk_count": 0,
            "bytes": 0,
            "gib": 0.0,
            "index_dtype": "",
            "build_time_s": 0.0,
            "max_chunk_entries": 0,
        }
        self._update_p4_chunk_scatter_memory_summary(0)

    @staticmethod
    def _best_index_dtype(max_value: int) -> np.dtype:
        return np.int32 if int(max_value) <= int(np.iinfo(np.int32).max) else np.int64

    def _supports_p4_chunk_scatter_cache(self, *, chunk_elems: int) -> bool:
        return (
            int(self.params.get("element_degree", 2)) == 4
            and int(chunk_elems) > 0
            and self.assembly_backend in {"coo", "coo_local"}
        )

    def _build_p4_chunk_scatter_cache(
        self,
        *,
        chunk_elems: int,
    ) -> tuple[P4ChunkScatterCacheEntry, ...] | None:
        if not self._supports_p4_chunk_scatter_cache(chunk_elems=chunk_elems):
            self._invalidate_p4_chunk_scatter_cache(reason="unsupported")
            return None
        request = str(self.p4_chunk_scatter_cache_requested)
        if request == "off":
            self._invalidate_p4_chunk_scatter_cache(reason="requested_off")
            return None

        n_local_elem = int(self.local_data.elems_local_np.shape[0])
        if n_local_elem <= 0:
            self._invalidate_p4_chunk_scatter_cache(reason="empty")
            return None

        dofs_per_elem = int(self.local_data.elems_local_np.shape[1])
        max_flat_idx = max(0, int(chunk_elems * dofs_per_elem * dofs_per_elem - 1))
        max_owned_pos = max(0, int(self.layout.owned_rows.size - 1))
        flat_dtype = self._best_index_dtype(max_flat_idx)
        pos_dtype = self._best_index_dtype(max_owned_pos)
        cache_limit_bytes = int(max(0.0, self.p4_chunk_scatter_cache_max_gib) * (1024.0**3))

        entries: list[P4ChunkScatterCacheEntry] = []
        cache_bytes = 0
        max_chunk_entries = 0
        t0 = time.perf_counter()
        for start in range(0, n_local_elem, int(chunk_elems)):
            stop = min(start + int(chunk_elems), n_local_elem)
            hess_e, hess_i, hess_j, hess_positions = self._chunk_scatter_pattern(
                start, stop
            )
            if hess_positions.size == 0:
                empty_flat = np.zeros(0, dtype=flat_dtype)
                empty_pos = np.zeros(0, dtype=pos_dtype)
                entries.append(
                    P4ChunkScatterCacheEntry(
                        flat_elem_idx=empty_flat,
                        owned_pos=empty_pos,
                    )
                )
                continue
            flat_idx = (
                (
                    np.asarray(hess_e, dtype=np.int64) * np.int64(dofs_per_elem)
                    + np.asarray(hess_i, dtype=np.int64)
                )
                * np.int64(dofs_per_elem)
                + np.asarray(hess_j, dtype=np.int64)
            )
            flat_arr = np.asarray(flat_idx, dtype=flat_dtype)
            pos_arr = np.asarray(hess_positions, dtype=pos_dtype)
            cache_bytes += int(flat_arr.nbytes + pos_arr.nbytes)
            max_chunk_entries = max(max_chunk_entries, int(pos_arr.size))
            if request == "auto" and cache_limit_bytes > 0 and cache_bytes > cache_limit_bytes:
                self._invalidate_p4_chunk_scatter_cache(reason="max_gib_exceeded")
                self.p4_chunk_scatter_cache_meta.update(
                    {
                        "requested": str(request),
                        "chunk_size": int(chunk_elems),
                        "chunk_count": int(len(entries) + 1),
                        "bytes": int(cache_bytes),
                        "gib": float(cache_bytes) / float(1024.0**3),
                        "index_dtype": f"{np.dtype(flat_dtype).name}/{np.dtype(pos_dtype).name}",
                        "build_time_s": float(time.perf_counter() - t0),
                        "max_chunk_entries": int(max_chunk_entries),
                    }
                )
                self._update_p4_chunk_scatter_memory_summary(0)
                return None
            entries.append(
                P4ChunkScatterCacheEntry(
                    flat_elem_idx=flat_arr,
                    owned_pos=pos_arr,
                )
            )

        self._p4_chunk_scatter_cache = tuple(entries)
        self._p4_chunk_scatter_cache_chunk_size = int(chunk_elems)
        self.p4_chunk_scatter_cache_meta = {
            "requested": str(request),
            "enabled": True,
            "reason": "built",
            "chunk_size": int(chunk_elems),
            "chunk_count": int(len(entries)),
            "bytes": int(cache_bytes),
            "gib": float(cache_bytes) / float(1024.0**3),
            "index_dtype": f"{np.dtype(flat_dtype).name}/{np.dtype(pos_dtype).name}",
            "build_time_s": float(time.perf_counter() - t0),
            "max_chunk_entries": int(max_chunk_entries),
        }
        self._update_p4_chunk_scatter_memory_summary(cache_bytes)
        return self._p4_chunk_scatter_cache

    def _get_p4_chunk_scatter_cache(
        self,
        *,
        chunk_elems: int,
    ) -> tuple[P4ChunkScatterCacheEntry, ...] | None:
        if not self._supports_p4_chunk_scatter_cache(chunk_elems=chunk_elems):
            return None
        if (
            self._p4_chunk_scatter_cache is not None
            and int(self._p4_chunk_scatter_cache_chunk_size) == int(chunk_elems)
        ):
            return self._p4_chunk_scatter_cache
        if self._p4_chunk_scatter_cache is not None:
            self._invalidate_p4_chunk_scatter_cache(reason="chunk_size_changed")
        if (
            not bool(self.p4_chunk_scatter_cache_meta.get("enabled", False))
            and int(self.p4_chunk_scatter_cache_meta.get("chunk_size", 0))
            == int(chunk_elems)
            and str(self.p4_chunk_scatter_cache_meta.get("reason", ""))
            in {"max_gib_exceeded", "requested_off", "empty"}
        ):
            return None
        if str(self.p4_chunk_scatter_cache_requested) == "off":
            return None
        return self._build_p4_chunk_scatter_cache(chunk_elems=int(chunk_elems))

    def _accumulate_owned_contrib(
        self,
        owned_vals: np.ndarray,
        positions: np.ndarray,
        contrib: np.ndarray,
    ) -> None:
        pos = np.asarray(positions).ravel()
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
            chunk_cache = self._get_p4_chunk_scatter_cache(chunk_elems=chunk_elems)
            timings["elem_hessian_compute"] = 0.0
            timings["scatter"] = 0.0
            timings["pattern_lookup"] = 0.0
            timings["accumulate"] = 0.0
            timings["chunk_count"] = int((n_local_elem + chunk_elems - 1) // chunk_elems)
            timings["chunk_size"] = int(chunk_elems)
            chunk_rows: list[int] = []
            chunk_kernel_times: list[float] = []
            chunk_scatter_times: list[float] = []
            for chunk_idx, start in enumerate(range(0, n_local_elem, chunk_elems)):
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
                        if chunk_cache is not None:
                            cache_entry = chunk_cache[int(chunk_idx)]
                            hess_positions = cache_entry.owned_pos
                            pattern_dt = 0.0
                            has_entries = bool(hess_positions.size != 0)
                            if has_entries:
                                t_acc0 = time.perf_counter()
                                contrib = elem_hess_chunk.reshape(-1)[
                                    cache_entry.flat_elem_idx
                                ]
                                self._accumulate_owned_contrib(
                                    owned_vals,
                                    hess_positions,
                                    contrib,
                                )
                                accumulate_dt = float(time.perf_counter() - t_acc0)
                            else:
                                accumulate_dt = 0.0
                        else:
                            t_pat0 = time.perf_counter()
                            hess_e, hess_i, hess_j, hess_positions = self._chunk_scatter_pattern(
                                start, stop
                            )
                            pattern_dt = float(time.perf_counter() - t_pat0)
                            has_entries = bool(hess_positions.size != 0)
                            if has_entries:
                                t_acc0 = time.perf_counter()
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
                                accumulate_dt = float(time.perf_counter() - t_acc0)
                            else:
                                accumulate_dt = 0.0
                        timings["pattern_lookup"] += float(pattern_dt)
                        timings["accumulate"] += float(accumulate_dt)
                scatter_dt = float(time.perf_counter() - t0)
                if not has_entries:
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
            timings["pattern_lookup"] = 0.0
            timings["accumulate"] = float(timings["scatter"])

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
