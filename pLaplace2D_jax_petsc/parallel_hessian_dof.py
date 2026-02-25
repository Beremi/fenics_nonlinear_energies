"""
Parallel Hessian assembly using DOF-partitioned overlapping domains + local SFD.

Two assembly variants:

  **Variant 2 — Zero-Communication Assembly** (default, preferred):
    Each MPI rank computes ALL n_colors HVPs on its local domain
    (overlapping elements from DOFPartition).  The HVP at owned DOFs is
    EXACT from local computation alone, so no Allreduce of Hessian values
    is needed.  Each rank directly fills its owned rows into the PETSc
    MPIAIJ matrix via COO fast-path.

    Only communication: Allgatherv of u (free DOFs, ~6 MB for 784K DOFs)
    before HVP computation to fill ghost DOF values.

  **Variant 1 — Communication Hiding**:
    Same computation as Variant 2, but the Allgatherv is started
    non-blocking (Iallgatherv) and overlapped with buffer preparation
    and owned-value filling.

Key difference from parallel_sfd.py (replicated SFD approach):
  - SFD:  n_colors/nprocs HVPs on FULL domain + Allreduce(nnz values, ~40-50 MB)
  - DOF:  n_colors HVPs on LOCAL domain     + NO Allreduce
  The local domain is ~1/nprocs of the full domain (+ boundary overlap), so
  total FLOPs are similar, but communication is dramatically reduced.

Correctness guarantee:
  For each owned row i and nonzero column j in the Hessian sparsity pattern:
  - DOFs i and j share at least one element (by definition of adjacency A)
  - That element touches owned DOF i, so it is in the local domain
  - Therefore DOF j is also in the local domain (as a node of that element)
  - The local HVP at owned DOF i with indicator for color(j) gives exact H[i,j]
  - The A² coloring ensures no interference between same-color columns

Provides three callbacks compatible with ``tools_petsc4py.minimizers.newton``:
    energy_fn(PETSc.Vec) -> float
    gradient_fn(PETSc.Vec, PETSc.Vec) -> None
    hessian_solve_fn(PETSc.Vec, PETSc.Vec, PETSc.Vec) -> int
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from mpi4py import MPI
from petsc4py import PETSc
from jax import config

config.update("jax_enable_x64", True)


class ParallelDOFHessianAssembler:
    """DOF-partitioned local SFD Hessian assembly + PETSc KSP.

    Parameters
    ----------
    params : dict
        JAX mesh params (u_0, freedofs, elems, dvx, dvy, vol, p, f).
    comm : MPI.Comm
        MPI communicator.
    adjacency : scipy.sparse matrix or None
        DOF-DOF adjacency (Hessian sparsity pattern), (n_free, n_free).
        Required on rank 0; other ranks may pass None.
    coloring_trials_per_rank : int
        Multi-start coloring trials per rank (default 10).
    ksp_rtol : float
        KSP relative tolerance.
    ksp_type : str
        PETSc KSP type (``"cg"`` for SPD).
    pc_type : str
        PETSc PC type (``"gamg"`` recommended; ``"hypre"`` also works since
        DOF ordering reflects mesh locality via RCM).
    """

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
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.timings = {}
        self.iter_timings = []

        # ---- 1. Build DOF partition (RCM reordering, local domain) ----
        t0 = time.perf_counter()
        from pLaplace2D_jax_petsc.dof_partition import DOFPartition
        self.part = DOFPartition(params, comm, adjacency=adjacency, reorder=True)
        self.timings["partition"] = time.perf_counter() - t0

        # ---- 2. Graph coloring of A² ----
        t0 = time.perf_counter()
        self._setup_coloring(adjacency, comm, coloring_trials_per_rank)
        self.timings["coloring"] = time.perf_counter() - t0

        # ---- 3. Precompute SFD index mappings (local extraction) ----
        t0 = time.perf_counter()
        self._precompute_indices(params, adjacency)
        self.timings["precompute"] = time.perf_counter() - t0

        # ---- 4. Create PETSc objects (Mat, KSP) ----
        t0 = time.perf_counter()
        self._setup_petsc(ksp_rtol, ksp_type, pc_type)
        self.timings["petsc_setup"] = time.perf_counter() - t0

        # ---- 5. JIT compile local JAX functions ----
        t0 = time.perf_counter()
        self._compile_jax()
        self.timings["jit"] = time.perf_counter() - t0

        # ---- 6. Compute aggregate stats (collective) ----
        self._sum_local_elems = comm.allreduce(self.part.n_local_elems, op=MPI.SUM)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _setup_coloring(self, adjacency, comm, trials_per_rank):
        """Run MPI multi-start randomised graph coloring on A²."""
        from graph_coloring.multistart_coloring import multistart_color

        adj_rank0 = adjacency if self.rank == 0 else None
        n_colors, coloring, color_info = multistart_color(
            adj_rank0, comm, trials_per_rank=trials_per_rank,
        )
        self.n_colors = n_colors
        self.coloring = coloring  # int32 array, length n_free, original ordering
        self.color_info = color_info

    def _precompute_indices(self, params, adjacency):
        """Precompute all index arrays for local-domain SFD extraction.

        After this method, every rank has:
        - ``_coo_rows, _coo_cols``:  PETSc COO pattern (reordered, owned rows only)
        - ``_color_nz[c]``:  (positions, local_rows) per colour
        - ``_indicators_local[c]``:  jnp indicator vectors in local node space
        """
        part = self.part
        comm = self.comm

        freedofs_np = np.asarray(params["freedofs"], dtype=np.int64)
        lo, hi = part.lo, part.hi
        iperm = part.iperm

        # ---- Broadcast adjacency nonzeros from rank 0 ----
        if self.rank == 0:
            adj_csr = adjacency.tocsr()
            row_adj, col_adj = adj_csr.nonzero()
            row_adj = np.ascontiguousarray(row_adj, dtype=np.int64)
            col_adj = np.ascontiguousarray(col_adj, dtype=np.int64)
            nnz = np.int64(len(row_adj))
        else:
            nnz = np.int64(0)

        nnz = int(comm.bcast(int(nnz), root=0))
        if self.rank != 0:
            row_adj = np.empty(nnz, dtype=np.int64)
            col_adj = np.empty(nnz, dtype=np.int64)
        comm.Bcast(row_adj, root=0)
        comm.Bcast(col_adj, root=0)

        self.nnz_global = nnz

        # ---- Identify owned entries in reordered space ----
        eff_rows = iperm[row_adj]
        eff_cols = iperm[col_adj]
        owned_mask = (eff_rows >= lo) & (eff_rows < hi)

        # COO arrays for PETSc (reordered indices)
        self._coo_rows = eff_rows[owned_mask].astype(PETSc.IntType)
        self._coo_cols = eff_cols[owned_mask].astype(PETSc.IntType)
        self._n_owned_nnz = int(owned_mask.sum())

        # Original-space indices of owned entries
        owned_row_orig = row_adj[owned_mask]   # original free-DOF row indices
        owned_col_orig = col_adj[owned_mask]   # original free-DOF col indices

        # ---- Build total_to_local mapping ----
        total_to_local = np.full(part.n_total, -1, dtype=np.int64)
        total_to_local[part.local_to_total] = np.arange(part.n_local, dtype=np.int64)

        # Map owned NZ rows to local node indices
        owned_row_total = freedofs_np[owned_row_orig]   # total node indices
        nz_local_rows = total_to_local[owned_row_total]   # local node indices
        assert np.all(nz_local_rows >= 0), (
            "BUG: some owned NZ rows map to nodes outside local domain"
        )

        # ---- Group owned NZ entries by colour ----
        owned_col_colors = self.coloring[owned_col_orig]

        self._color_nz = {}
        for c in range(self.n_colors):
            mask_c = owned_col_colors == c
            positions = np.where(mask_c)[0].astype(np.int64)
            local_rows = nz_local_rows[positions].astype(np.int64)
            self._color_nz[c] = (positions, local_rows)

        # ---- Build local indicator vectors per colour ----
        # orig_freedof → local_node  (or -1 if not in local domain)
        orig_to_local = total_to_local[freedofs_np]   # (n_free,) → local or -1

        self._indicators_local = []
        for c in range(self.n_colors):
            indicator = np.zeros(part.n_local, dtype=np.float64)
            dofs_of_c = np.where(self.coloring == c)[0]   # original free-DOF indices
            local_idx = orig_to_local[dofs_of_c]
            valid = local_idx >= 0
            indicator[local_idx[valid]] = 1.0
            self._indicators_local.append(jnp.array(indicator))

        # ---- Store for energy / gradient callbacks ----
        self._owned_idx = jnp.array(part.owned_local_indices, dtype=jnp.int32)
        self._f_owned = jnp.array(part.f_owned, dtype=jnp.float64)

    def _setup_petsc(self, ksp_rtol, ksp_type, pc_type):
        """Create PETSc MPIAIJ matrix via COO preallocation and KSP."""
        n = self.part.n_free
        lo, hi = self.part.lo, self.part.hi
        n_local = hi - lo

        # COO preallocation — each rank registers only its owned (row, col)
        self.A = PETSc.Mat().create(comm=self.comm)
        self.A.setType(PETSc.Mat.Type.MPIAIJ)
        self.A.setSizes(((n_local, n), (n_local, n)))
        self.A.setPreallocationCOO(self._coo_rows, self._coo_cols)

        # KSP (CG + GAMG by default)
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setType(ksp_type)
        self.ksp.getPC().setType(pc_type)
        self.ksp.setTolerances(rtol=ksp_rtol)
        self.ksp.setFromOptions()

    def _compile_jax(self):
        """JIT-compile energy, gradient, and HVP for the local domain."""
        p = self.part.p
        elems = jnp.array(self.part.elems_local_np, dtype=jnp.int32)
        dvx = jnp.array(self.part.dvx_np, dtype=jnp.float64)
        dvy = jnp.array(self.part.dvy_np, dtype=jnp.float64)
        vol = jnp.array(self.part.vol_np, dtype=jnp.float64)
        vol_w = jnp.array(
            self.part.vol_np * self.part.elem_weights, dtype=jnp.float64
        )

        # --- Energy: weighted (unique element assignment for Allreduce) ---
        def energy_weighted(v_local):
            v_e = v_local[elems]
            Fx = jnp.sum(v_e * dvx, axis=1)
            Fy = jnp.sum(v_e * dvy, axis=1)
            intgrds = (1.0 / p) * (Fx**2 + Fy**2) ** (p / 2.0)
            return jnp.sum(intgrds * vol_w)

        self._energy_jit = jax.jit(energy_weighted)

        # --- Gradient / HVP: unweighted (exact at owned DOFs) ---
        def energy_full(v_local):
            v_e = v_local[elems]
            Fx = jnp.sum(v_e * dvx, axis=1)
            Fy = jnp.sum(v_e * dvy, axis=1)
            intgrds = (1.0 / p) * (Fx**2 + Fy**2) ** (p / 2.0)
            return jnp.sum(intgrds * vol)

        self._grad_jit = jax.jit(jax.grad(energy_full))

        def hvp_fn(v_local, tangent_local):
            return jax.jvp(jax.grad(energy_full), (v_local,), (tangent_local,))[1]

        self._hvp_jit = jax.jit(hvp_fn)

        # Warmup (trigger JIT compilation)
        v_dummy = jnp.zeros(self.part.n_local, dtype=jnp.float64)
        _ = self._energy_jit(v_dummy).block_until_ready()
        _ = self._grad_jit(v_dummy).block_until_ready()
        _ = self._hvp_jit(v_dummy, v_dummy).block_until_ready()

    # ------------------------------------------------------------------
    # Assembly methods
    # ------------------------------------------------------------------

    def assemble_hessian(self, u_owned, variant=2):
        """Assemble Hessian matrix into PETSc Mat.

        Parameters
        ----------
        u_owned : (n_owned,) numpy array
            This rank's free DOFs in reordered space.
        variant : int
            2 = zero-communication (default), 1 = communication hiding.

        Returns
        -------
        timings : dict
            Detailed timing breakdown of the assembly.
        """
        if variant == 2:
            return self._assemble_variant2(u_owned)
        elif variant == 1:
            return self._assemble_variant1(u_owned)
        else:
            raise ValueError(f"Unknown variant {variant}")

    def _assemble_variant2(self, u_owned):
        """Variant 2: blocking Allgatherv + all-colors local HVP.

        No Allreduce needed — each rank fills only its owned rows.
        """
        timings = {}
        t_total = time.perf_counter()

        # 1. Allgatherv u → build v_local
        t0 = time.perf_counter()
        u_full = self.part.get_u_full(np.asarray(u_owned, dtype=np.float64))
        v_local = jnp.array(self.part.build_v_local(u_full))
        timings["allgatherv"] = time.perf_counter() - t0

        # 2. Compute ALL n_colors HVPs locally, extract owned entries
        t0 = time.perf_counter()
        owned_vals = np.zeros(self._n_owned_nnz, dtype=np.float64)
        n_hvps = 0

        for c in range(self.n_colors):
            hvp_result = np.asarray(
                self._hvp_jit(v_local, self._indicators_local[c]).block_until_ready(),
                dtype=np.float64,
            )
            positions, local_rows = self._color_nz[c]
            if len(positions) > 0:
                owned_vals[positions] = hvp_result[local_rows]
            n_hvps += 1

        timings["hvp_compute"] = time.perf_counter() - t0
        timings["n_hvps"] = n_hvps

        # 3. COO assembly — NO Allreduce!
        t0 = time.perf_counter()
        self.A.setValuesCOO(
            owned_vals.astype(PETSc.ScalarType),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        return timings

    def _assemble_variant1(self, u_owned):
        """Variant 1: non-blocking Allgatherv overlapped with preparation.

        Same final result as Variant 2. The Iallgatherv is overlapped with
        owned-value filling and buffer allocation.
        """
        timings = {}
        t_total = time.perf_counter()

        # 1. Start non-blocking Allgatherv
        t0 = time.perf_counter()
        u_full = np.empty(self.part.n_free, dtype=np.float64)
        u_owned_buf = np.ascontiguousarray(u_owned, dtype=np.float64)
        req = self.comm.Iallgatherv(
            u_owned_buf,
            [u_full, self.part._gather_sizes, self.part._gather_displs, MPI.DOUBLE],
        )
        timings["iallgatherv_start"] = time.perf_counter() - t0

        # 2. Overlap: prepare output buffer + pre-fill owned DOFs in template
        t0 = time.perf_counter()
        owned_vals = np.zeros(self._n_owned_nnz, dtype=np.float64)
        v_np = self.part.v_template.copy()
        # Owned DOFs are known already (u_owned) — fill them now
        # This avoids re-touching these cache lines after the wait
        part = self.part
        # owned DOFs in free_local/free_global that map to [lo, hi):
        # we can precompute the owned subset of free_local_indices
        v_np[part.free_local_indices[
            (part.free_global_indices >= part.lo) &
            (part.free_global_indices < part.hi)
        ]] = u_owned_buf[
            part.free_global_indices[
                (part.free_global_indices >= part.lo) &
                (part.free_global_indices < part.hi)
            ] - part.lo
        ]
        timings["overlap_prep"] = time.perf_counter() - t0

        # 3. Wait for Allgatherv
        t0 = time.perf_counter()
        req.Wait()
        timings["iallgatherv_wait"] = time.perf_counter() - t0
        timings["allgatherv"] = (
            timings["iallgatherv_start"] + timings["iallgatherv_wait"]
        )

        # 4. Fill ghost DOFs from u_full
        t0 = time.perf_counter()
        # Re-fill ALL free DOFs (simpler than computing ghost-only subset)
        v_np[part.free_local_indices] = u_full[part.free_global_indices]
        v_local = jnp.array(v_np)
        timings["build_v_local"] = time.perf_counter() - t0

        # 5. Compute all HVPs locally
        t0 = time.perf_counter()
        n_hvps = 0
        for c in range(self.n_colors):
            hvp_result = np.asarray(
                self._hvp_jit(v_local, self._indicators_local[c]).block_until_ready(),
                dtype=np.float64,
            )
            positions, local_rows = self._color_nz[c]
            if len(positions) > 0:
                owned_vals[positions] = hvp_result[local_rows]
            n_hvps += 1
        timings["hvp_compute"] = time.perf_counter() - t0
        timings["n_hvps"] = n_hvps

        # 6. COO assembly — NO Allreduce!
        t0 = time.perf_counter()
        self.A.setValuesCOO(
            owned_vals.astype(PETSc.ScalarType),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        return timings

    # ------------------------------------------------------------------
    # Newton callbacks (compatible with tools_petsc4py.minimizers.newton)
    # ------------------------------------------------------------------

    def energy_fn(self, vec):
        """Evaluate energy J(u) from a distributed PETSc Vec (reordered)."""
        u_owned = np.array(vec.array[:], dtype=np.float64)
        v_local = self._get_v_local(u_owned)

        local_e = float(self._energy_jit(v_local).block_until_ready())
        total_e = self.comm.allreduce(local_e, op=MPI.SUM)

        local_fu = float(np.dot(np.asarray(self._f_owned), u_owned))
        total_fu = self.comm.allreduce(local_fu, op=MPI.SUM)

        return total_e - total_fu

    def gradient_fn(self, vec, g):
        """Compute ∇J(u) at owned DOFs → distributed PETSc Vec g (reordered)."""
        u_owned = np.array(vec.array[:], dtype=np.float64)
        v_local = self._get_v_local(u_owned)

        g_local = self._grad_jit(v_local).block_until_ready()
        g_owned = np.asarray(g_local[self._owned_idx]) - np.asarray(self._f_owned)
        g.array[:] = g_owned

    def hessian_solve_fn(self, vec, rhs, sol):
        """Assemble Hessian via local SFD + KSP solve.

        All vectors are in reordered space (matching PETSc block dist).
        Returns the number of KSP iterations.
        """
        t_total_start = time.perf_counter()

        u_owned = np.array(vec.array[:], dtype=np.float64)
        assembly_timings = self.assemble_hessian(u_owned, variant=2)

        # KSP solve (already in reordered space)
        t0 = time.perf_counter()
        self.ksp.setOperators(self.A)
        self.ksp.solve(rhs, sol)
        ksp_its = self.ksp.getIterationNumber()
        t_ksp = time.perf_counter() - t0

        assembly_timings["ksp"] = t_ksp
        assembly_timings["ksp_its"] = ksp_its
        assembly_timings["total_with_ksp"] = time.perf_counter() - t_total_start

        return ksp_its

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_v_local(self, u_owned):
        """Build v_local from owned DOFs via P2P ghost exchange."""
        v_np = self.part.build_v_local_p2p(np.asarray(u_owned, dtype=np.float64))
        return jnp.array(v_np)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def create_vec(self, full_array_reordered=None):
        """Create distributed PETSc Vec with correct ownership.

        Parameters
        ----------
        full_array_reordered : (n_free,) array or None
            If provided, scatter [lo:hi) slice into the Vec.
            Must be in **reordered** DOF ordering.
        """
        n = self.part.n_free
        v = PETSc.Vec().createMPI(n, comm=self.comm)
        if full_array_reordered is not None:
            lo, hi = self.part.lo, self.part.hi
            arr = np.asarray(full_array_reordered, dtype=np.float64)
            v.array[:] = arr[lo:hi]
            v.assemble()
        return v

    def cleanup(self):
        """Destroy PETSc objects."""
        self.ksp.destroy()
        self.A.destroy()

    def get_timing_report(self):
        """Return summary dict of setup and per-assembly timings.

        Safe to call on a single rank (no collective ops).
        """
        report = {
            "setup": dict(self.timings),
            "n_colors": self.n_colors,
            "n_free": self.part.n_free,
            "n_local": self.part.n_local,
            "n_local_elems": self.part.n_local_elems,
            "lo": self.part.lo,
            "hi": self.part.hi,
            "n_owned": self.part.n_owned,
            "n_owned_nnz": self._n_owned_nnz,
            "nnz_global": self.nnz_global,
            "sum_local_elems_all_ranks": self._sum_local_elems,
        }
        if self.iter_timings:
            report["assembly_details"] = list(self.iter_timings)
        return report
