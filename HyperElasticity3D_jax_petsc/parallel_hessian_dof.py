"""HyperElasticity-specific parallel Hessian assemblers (DOF-partitioned)."""

import time

import numpy as np
import jax
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

    # ------------------------------------------------------------------
    # Element-level Hessian assembly (analytical, no graph coloring)
    # ------------------------------------------------------------------

    def setup_element_hessian(self):
        """Set up element-level analytical Hessian assembly.

        Creates a per-element energy function, vmapped element Hessian,
        and a mapping from element contributions to the SFD COO pattern.
        Reuses ``self.A`` (same sparsity, same internal storage) to avoid
        KSP solve regressions from a separate matrix with duplicate COO entries.
        Call this after construction to enable ``assemble_hessian_element()``.
        """
        t_setup_start = time.perf_counter()
        part = self.part
        comm = self.comm

        C1 = self._C1
        D1 = self._D1
        use_abs_det = self._use_abs_det

        # --- 1. Per-element energy function (scalar) ---
        def element_energy(v_e, dphix_e, dphiy_e, dphiz_e, vol_e):
            """Neo-Hookean energy for a single element.

            v_e: (12,)  dphix_e/dphiy_e/dphiz_e: (4,)  vol_e: scalar
            """
            vx = v_e[0::3]  # (4,)
            vy = v_e[1::3]
            vz = v_e[2::3]

            F11 = jnp.dot(vx, dphix_e)
            F12 = jnp.dot(vx, dphiy_e)
            F13 = jnp.dot(vx, dphiz_e)
            F21 = jnp.dot(vy, dphix_e)
            F22 = jnp.dot(vy, dphiy_e)
            F23 = jnp.dot(vy, dphiz_e)
            F31 = jnp.dot(vz, dphix_e)
            F32 = jnp.dot(vz, dphiy_e)
            F33 = jnp.dot(vz, dphiz_e)

            I1 = (F11**2 + F12**2 + F13**2
                  + F21**2 + F22**2 + F23**2
                  + F31**2 + F32**2 + F33**2)
            detF = (F11 * (F22 * F33 - F23 * F32)
                    - F12 * (F21 * F33 - F23 * F31)
                    + F13 * (F21 * F32 - F22 * F31))
            det_for_energy = jnp.abs(detF) if use_abs_det else detF
            W = C1 * (I1 - 3.0 - 2.0 * jnp.log(det_for_energy)) + D1 * (det_for_energy - 1.0) ** 2
            return W * vol_e

        # --- 2. Vmapped element Hessian ---
        elem_hess_fn = jax.hessian(element_energy)
        vmapped_hess = jax.vmap(elem_hess_fn, in_axes=(0, 0, 0, 0, 0))

        elems_jnp = jnp.array(part.elems_local_np, dtype=jnp.int32)
        dphix_jnp = jnp.array(part.local_elem_data["dphix"], dtype=jnp.float64)
        dphiy_jnp = jnp.array(part.local_elem_data["dphiy"], dtype=jnp.float64)
        dphiz_jnp = jnp.array(part.local_elem_data["dphiz"], dtype=jnp.float64)
        vol_jnp = jnp.array(part.local_elem_data["vol"], dtype=jnp.float64)

        @jax.jit
        def compute_elem_hessians(v_local):
            v_e = v_local[elems_jnp]  # (n_elem, 12)
            return vmapped_hess(v_e, dphix_jnp, dphiy_jnp, dphiz_jnp, vol_jnp)

        self._elem_hessian_jit = compute_elem_hessians

        # --- 3. Map element contributions → SFD COO positions ---
        # Reuse self.A's COO pattern (unique entries, INSERT_VALUES) to keep
        # the same PETSc internal storage → no KSP matvec regression.
        lo, hi = part.lo, part.hi
        freedofs_np = part.freedofs_np
        iperm = part.iperm

        # Map local node indices → original free-DOF indices
        total_to_freedof = np.full(part.n_total, -1, dtype=np.int64)
        total_to_freedof[freedofs_np] = np.arange(part.n_free, dtype=np.int64)

        elems_local = part.elems_local_np  # (n_elem, 12) local node indices
        n_elem = len(elems_local)

        # Map element local-node DOFs to reordered global indices
        elems_total = part.local_to_total[elems_local]  # (n_elem, 12) total node
        elems_freedof = total_to_freedof[elems_total]  # (n_elem, 12) orig free-DOF, -1=Dirichlet
        elems_reordered = np.where(
            elems_freedof >= 0,
            iperm[np.clip(elems_freedof, 0, None)],
            -1,
        )  # (n_elem, 12) reordered global index, -1 for Dirichlet

        # Reconstruct original COO arrays (before PETSc's setPreallocationCOO
        # modified them in-place for MPIAIJ off-process column remapping).
        iperm = part.iperm
        row_adj = self._row_adj
        col_adj = self._col_adj
        eff_rows_orig = iperm[row_adj]
        eff_cols_orig = iperm[col_adj]
        owned_mask_orig = (eff_rows_orig >= lo) & (eff_rows_orig < hi)
        sfd_coo_rows = eff_rows_orig[owned_mask_orig].astype(np.int64)
        sfd_coo_cols = eff_cols_orig[owned_mask_orig].astype(np.int64)
        n_sfd_nnz = len(sfd_coo_rows)

        # Pack (row, col) into a single int64 for fast lookup
        n_global = part.n_free
        sfd_keys = sfd_coo_rows * n_global + sfd_coo_cols
        sfd_pos_map = {}
        for k in range(n_sfd_nnz):
            sfd_pos_map[int(sfd_keys[k])] = k

        # For each element contribution (owned row, non-Dirichlet col),
        # find the corresponding SFD COO position and record the mapping.
        all_sfd_pos = []
        all_scatter_e = []
        all_scatter_i = []
        all_scatter_j = []

        n_dropped = 0
        chunk_size = 10000
        for start in range(0, n_elem, chunk_size):
            end = min(start + chunk_size, n_elem)
            chunk = elems_reordered[start:end]  # (chunk, 12)

            row_all = chunk[:, :, None]  # (chunk, 12, 1)
            col_all = chunk[:, None, :]  # (chunk, 1, 12)

            row_valid = (row_all >= lo) & (row_all < hi)
            col_valid = col_all >= 0
            pair_valid = row_valid & col_valid  # (chunk, 12, 12)

            vi = np.where(pair_valid)
            rows = chunk[vi[0], vi[1]]
            cols = chunk[vi[0], vi[2]]
            keys = rows.astype(np.int64) * n_global + cols.astype(np.int64)

            for idx in range(len(keys)):
                pos = sfd_pos_map.get(int(keys[idx]), -1)
                if pos >= 0:
                    all_sfd_pos.append(pos)
                    all_scatter_e.append(int(vi[0][idx]) + start)
                    all_scatter_i.append(int(vi[1][idx]))
                    all_scatter_j.append(int(vi[2][idx]))
                else:
                    n_dropped += 1

        n_dropped_total = comm.allreduce(n_dropped)
        if n_dropped_total > 0 and comm.Get_rank() == 0:
            print(
                f"  WARNING: {n_dropped_total} element contributions not "
                f"found in SFD COO pattern",
                flush=True,
            )

        self._elem_to_sfd_pos = np.array(all_sfd_pos, dtype=np.int64)
        self._elem_scatter_e = np.array(all_scatter_e, dtype=np.int64)
        self._elem_scatter_i = np.array(all_scatter_i, dtype=np.int64)
        self._elem_scatter_j = np.array(all_scatter_j, dtype=np.int64)

        n_mapped = len(self._elem_to_sfd_pos)
        if comm.Get_rank() == 0:
            print(
                f"  Element Hessian: {n_mapped} contributions → "
                f"{n_sfd_nnz} unique SFD COO entries",
                flush=True,
            )

        # Store SFD nnz count for assembly
        self._n_sfd_nnz = n_sfd_nnz

        # --- 4. JIT warmup ---
        t_jit0 = time.perf_counter()
        v_dummy = jnp.zeros(part.n_local, dtype=jnp.float64)
        _ = self._elem_hessian_jit(v_dummy).block_until_ready()
        t_jit = time.perf_counter() - t_jit0

        self._elem_hessian_setup_time = time.perf_counter() - t_setup_start
        if comm.Get_rank() == 0:
            print(
                f"  Element Hessian setup: {self._elem_hessian_setup_time:.3f}s "
                f"(JIT warmup: {t_jit:.3f}s)",
                flush=True,
            )

    def assemble_hessian_element(self, u_owned):
        """Assemble Hessian via element-level analytical Hessians.

        Pre-aggregates element contributions into the SFD COO pattern and
        uses INSERT_VALUES on self.A to preserve PETSc internal storage layout.
        Returns timing dict compatible with SFD assembly interface.
        """
        from petsc4py import PETSc as _PETSc

        timings = {}
        t_total = time.perf_counter()

        # 1. P2P ghost exchange → v_local
        t0 = time.perf_counter()
        v_local = self._get_v_local(u_owned)
        timings["p2p_exchange"] = time.perf_counter() - t0

        # 2. Compute all element Hessians via vmapped jax.hessian
        t0 = time.perf_counter()
        elem_hess = self._elem_hessian_jit(v_local).block_until_ready()
        timings["elem_hessian_compute"] = time.perf_counter() - t0

        # 3. Scatter + aggregate element contributions into SFD COO values
        t0 = time.perf_counter()
        elem_hess_np = np.asarray(elem_hess)  # (n_elem, 12, 12)
        contrib = elem_hess_np[
            self._elem_scatter_e, self._elem_scatter_i, self._elem_scatter_j
        ]
        # Pre-aggregate duplicates: sum element contributions per SFD COO entry
        owned_vals = np.zeros(self._n_sfd_nnz, dtype=np.float64)
        np.add.at(owned_vals, self._elem_to_sfd_pos, contrib)
        timings["scatter"] = time.perf_counter() - t0

        # 4. COO assembly on self.A (same matrix as SFD)
        t0 = time.perf_counter()
        self.A.zeroEntries()
        self.A.setValuesCOO(
            owned_vals.astype(_PETSc.ScalarType),
            addv=_PETSc.InsertMode.ADD_VALUES,
        )
        self.A.assemble()
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_total
        timings["hvp_compute"] = timings["elem_hessian_compute"]  # compat key
        timings["n_hvps"] = 0  # no SFD HVPs
        timings["assembly_mode"] = "element"
        self.iter_timings.append(timings)
        return timings
