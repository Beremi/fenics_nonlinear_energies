"""
Generic parallel Hessian assembly using DOF-partitioned overlapping domains + local SFD.

Two assembler classes:

  **DOFHessianAssemblerBase** — global multi-start A² graph coloring,
    sequential per-color HVP computation, COO fast-path assembly.

  **LocalColoringAssemblerBase** — per-rank local coloring on A²|_J,
    vmap-batched all-color HVP, P2P ghost exchange.  Preferred.

Both classes provide Newton callbacks compatible with
``tools_petsc4py.minimizers.newton``:
    energy_fn(PETSc.Vec) -> float
    gradient_fn(PETSc.Vec, PETSc.Vec) -> None
    hessian_solve_fn(PETSc.Vec, PETSc.Vec, PETSc.Vec) -> int

Subclasses MUST override ``_make_local_energy_fns()`` to provide
problem-specific energy functions.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from mpi4py import MPI
from petsc4py import PETSc
from jax import config

config.update("jax_enable_x64", True)


def _empty_problem_state(params, options):
    """Default problem-state builder."""
    del params, options
    return {}


@dataclass(frozen=True)
class JaxProblemSpec:
    """Problem-specific hooks for generic JAX/PETSc assemblers.

    Only the energy kernels and optional problem metadata should vary per
    problem. The distributed assembly, coloring, PETSc setup, and Newton
    callback plumbing stay generic.
    """

    elem_data_keys: tuple[str, ...]
    make_local_energy_fns: Callable[[object, Mapping[str, object]], tuple[Callable, Callable]]
    make_element_hessian_jit: Callable[[object, Mapping[str, object]], Callable] | None = None
    build_state: Callable[[Mapping[str, object], Mapping[str, object]], Mapping[str, object]] = _empty_problem_state
    make_near_nullspace: Callable[[Mapping[str, object], Mapping[str, object]], list[np.ndarray] | None] | None = None
    rhs_key: str | None = None
    ownership_block_size: int = 1
    default_reorder: bool = True


class DOFHessianAssemblerBase:
    """DOF-partitioned local SFD Hessian assembly + PETSc KSP.

    Subclasses MUST override ``_make_local_energy_fns()`` to return
    problem-specific energy functions.

    Parameters
    ----------
    freedofs, elems, u_0 : array-like
        Free DOF indices, element connectivity, full DOF vector.
    comm : MPI.Comm
        MPI communicator.
    adjacency : scipy.sparse matrix or None
        DOF-DOF adjacency (Hessian sparsity), (n_free, n_free).
        Required on rank 0; other ranks may pass None.
    f : array-like or None
        RHS/load vector (same length as u_0). None → zeros.
    elem_data : dict or None
        Element-level arrays to slice by local elements.
    coloring_trials_per_rank : int
        Multi-start coloring trials per rank (default 10).
    ksp_rtol : float
        KSP relative tolerance.
    ksp_type : str
        PETSc KSP type.
    pc_type : str
        PETSc PC type.
    ksp_max_it : int or None
        KSP maximum iterations (None → PETSc default).
    near_nullspace_vecs : list of (n_free,) arrays or None
        Vectors for the near-nullspace (e.g., rigid body modes for
        elasticity).  Must be in **original** free-DOF ordering.
    pc_options : dict or None
        Additional PETSc options for the preconditioner
        (e.g., ``{"pc_hypre_boomeramg_nodal_coarsen": 6}``).
    reorder : bool
        Whether to apply RCM DOF reordering in the partition.  Keep
        ``True`` for scalar problems; vector elasticity may prefer
        ``False`` to preserve natural block ordering.
    ownership_block_size : int
        Ownership distribution block size.  Use 1 for scalar unknowns,
        3 for 3D vector DOFs when local ownership must preserve xyz blocks.
    """

    def __init__(
        self,
        freedofs,
        elems,
        u_0,
        comm,
        adjacency=None,
        f=None,
        elem_data=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="gamg",
        ksp_max_it=None,
        near_nullspace_vecs=None,
        pc_options=None,
        reorder=True,
        ownership_block_size=1,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.timings = {}
        self.iter_timings = []

        # Store adjacency for _precompute_indices (parent class broadcasts)
        self._adjacency = adjacency

        # ---- 1. Build DOF partition (RCM reordering, local domain) ----
        t0 = time.perf_counter()
        from tools_petsc4py.dof_partition import DOFPartition
        self.part = DOFPartition(
            freedofs, elems, u_0, comm,
            adjacency=adjacency, reorder=reorder,
            f=f, elem_data=elem_data,
            ownership_block_size=ownership_block_size,
        )
        self.timings["partition"] = time.perf_counter() - t0

        # ---- 2. Graph coloring of A² ----
        t0 = time.perf_counter()
        self._setup_coloring(comm, coloring_trials_per_rank)
        self.timings["coloring"] = time.perf_counter() - t0

        # ---- 3. Precompute SFD index mappings (local extraction) ----
        t0 = time.perf_counter()
        self._precompute_indices()
        self.timings["precompute"] = time.perf_counter() - t0

        # ---- 4. Create PETSc objects (Mat, KSP) ----
        t0 = time.perf_counter()
        self._setup_petsc(
            ksp_rtol, ksp_type, pc_type,
            ksp_max_it=ksp_max_it,
            near_nullspace_vecs=near_nullspace_vecs,
            pc_options=pc_options,
        )
        self.timings["petsc_setup"] = time.perf_counter() - t0

        # ---- 5. JIT compile local JAX functions ----
        t0 = time.perf_counter()
        self._compile_jax()
        self.timings["jit"] = time.perf_counter() - t0

        # ---- 6. Compute aggregate stats (collective) ----
        self._sum_local_elems = comm.allreduce(self.part.n_local_elems, op=MPI.SUM)

    # ------------------------------------------------------------------
    # Abstract: subclass must override
    # ------------------------------------------------------------------

    def _make_local_energy_fns(self):
        """Return (energy_weighted_fn, energy_full_fn) for JIT compilation.

        Both functions:  ``v_local -> scalar``

        ``energy_weighted_fn`` uses element weights for unique energy assignment
        (each element counted once across ranks for the global energy sum).

        ``energy_full_fn`` uses all local elements without weighting (for exact
        gradient/HVP at owned DOFs).

        Subclasses MUST override this method.
        """
        raise NotImplementedError(
            "Subclass must implement _make_local_energy_fns() to provide "
            "problem-specific energy functions."
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _setup_coloring(self, comm, trials_per_rank):
        """Run MPI multi-start randomised graph coloring on A²."""
        from graph_coloring.multistart_coloring import multistart_color

        adj_rank0 = self._adjacency if self.rank == 0 else None
        n_colors, coloring, color_info = multistart_color(
            adj_rank0, comm, trials_per_rank=trials_per_rank,
        )
        self.n_colors = n_colors
        self.coloring = coloring  # int32 array, length n_free, original ordering
        self.color_info = color_info

    def _precompute_indices(self):
        """Precompute all index arrays for local-domain SFD extraction.

        After this method, every rank has:
        - ``_coo_rows, _coo_cols``:  PETSc COO pattern (reordered, owned rows only)
        - ``_color_nz[c]``:  (positions, local_rows) per colour
        - ``_indicators_local[c]``:  jnp indicator vectors in local node space
        """
        part = self.part
        comm = self.comm

        freedofs_np = part.freedofs_np
        lo, hi = part.lo, part.hi
        iperm = part.iperm

        # ---- Broadcast adjacency nonzeros from rank 0 ----
        if self.rank == 0:
            adj_csr = self._adjacency.tocsr()
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

    def _setup_petsc(self, ksp_rtol, ksp_type, pc_type,
                     ksp_max_it=None, near_nullspace_vecs=None,
                     pc_options=None):
        """Create PETSc MPIAIJ matrix via COO preallocation and KSP."""
        n = self.part.n_free
        lo, hi = self.part.lo, self.part.hi
        n_local = hi - lo

        # COO preallocation — each rank registers only its owned (row, col)
        self.A = PETSc.Mat().create(comm=self.comm)
        self.A.setType(PETSc.Mat.Type.MPIAIJ)
        self.A.setSizes(((n_local, n), (n_local, n)))
        self.A.setPreallocationCOO(self._coo_rows, self._coo_cols)

        # Near-nullspace (e.g., rigid body modes for elasticity)
        self._nullspace = None
        if near_nullspace_vecs is not None:
            perm = self.part.perm
            petsc_vecs = []
            for mode_orig in near_nullspace_vecs:
                mode_orig = np.asarray(mode_orig, dtype=np.float64)
                mode_reord = mode_orig[perm]
                pv = PETSc.Vec().createMPI((hi - lo, n), comm=self.comm)
                pv.array[:] = mode_reord[lo:hi]
                pv.assemble()
                petsc_vecs.append(pv)
            self._nullspace = PETSc.NullSpace().create(vectors=petsc_vecs)
            self.A.setNearNullSpace(self._nullspace)

        # KSP
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setType(ksp_type)
        self.ksp.getPC().setType(pc_type)

        # PC options (e.g., Hypre BoomerAMG settings)
        if pc_options:
            opts = PETSc.Options()
            for key, val in pc_options.items():
                opts[key] = val

        self.ksp.setTolerances(rtol=ksp_rtol)
        if ksp_max_it is not None:
            self.ksp.setTolerances(max_it=ksp_max_it)
        self.ksp.setFromOptions()

    def _compile_jax(self):
        """JIT-compile energy, gradient, and HVP for the local domain."""
        energy_weighted, energy_full = self._make_local_energy_fns()

        self._energy_jit = jax.jit(energy_weighted)

        grad_fn = jax.grad(energy_full)
        self._grad_jit = jax.jit(grad_fn)

        def hvp_fn(v_local, tangent_local):
            return jax.jvp(grad_fn, (v_local,), (tangent_local,))[1]

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
        # Finalise matrix values explicitly to avoid hidden assembly cost in KSP.solve.
        self.A.assemble()
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        return timings

    def _assemble_variant1(self, u_owned):
        """Variant 1: non-blocking Allgatherv overlapped with preparation."""
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
        part = self.part
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
        # Finalise matrix values explicitly to avoid hidden assembly cost in KSP.solve.
        self.A.assemble()
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

    def update_dirichlet(self, u_0_new):
        """Update Dirichlet boundary values for next timestep.

        Parameters
        ----------
        u_0_new : array-like
            New full DOF vector with updated Dirichlet boundary values.
        """
        self.part.update_dirichlet(u_0_new)

    def create_vec(self, full_array_reordered=None):
        """Create distributed PETSc Vec with correct ownership.

        Parameters
        ----------
        full_array_reordered : (n_free,) array or None
            If provided, scatter [lo:hi) slice into the Vec.
            Must be in **reordered** DOF ordering.
        """
        n = self.part.n_free
        lo, hi = self.part.lo, self.part.hi
        v = PETSc.Vec().createMPI((hi - lo, n), comm=self.comm)
        if full_array_reordered is not None:
            arr = np.asarray(full_array_reordered, dtype=np.float64)
            v.array[:] = arr[lo:hi]
            v.assemble()
        return v

    def cleanup(self):
        """Destroy PETSc objects."""
        self.ksp.destroy()
        self.A.destroy()
        if self._nullspace is not None:
            self._nullspace.destroy()

    def get_timing_report(self):
        """Return summary dict of setup and per-assembly timings."""
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


class LocalColoringAssemblerBase(DOFHessianAssemblerBase):
    """DOF-partitioned local SFD Hessian assembly with per-rank local
    coloring + vmap.

    Replaces global graph coloring with local per-rank coloring on A²|_J
    and uses JAX ``vmap`` for batched all-color HVP computation.

    Key benefits over :class:`DOFHessianAssemblerBase`:
      - No global A² computation/broadcast (coloring is purely local)
      - Typically fewer colors (smaller subgraph → lower max degree)
      - Batched HVP via ``vmap`` (single JIT call instead of n_colors sequential)
      - P2P ghost exchange for assembly (no Allgatherv)

    Same public interface.  Subclasses MUST still override
    ``_make_local_energy_fns()``.
    """

    def _make_element_hessian_jit(self):
        """Return a JIT-compiled ``v_local -> (n_elem, npe, npe)`` Hessian."""
        raise NotImplementedError(
            "Subclass must implement _make_element_hessian_jit() to provide "
            "problem-specific per-element Hessians."
        )

    def _setup_coloring(self, comm, trials_per_rank):
        """Per-rank local graph coloring on A²|_J (Variant B) - no broadcast.

        J = owned_DOFs ∪ N_A(owned_DOFs) in original free-DOF space.
        Each rank builds A|_J directly from its local element connectivity,
        computes A²|_J, and colors locally with igraph greedy coloring.
        """
        import scipy.sparse as sp
        import igraph

        part = self.part
        lo, hi = part.lo, part.hi
        perm = part.perm
        n_free = part.n_free

        # ---- Map local nodes → original free-DOF indices ----
        local_to_orig_freedof = np.full(part.n_local, -1, dtype=np.int64)
        local_to_orig_freedof[part.free_local_indices] = perm[part.free_global_indices]

        # ---- Build A|_J from local elements (no broadcast) ----
        elems = part.elems_local_np           # (n_local_elems, npe) local node indices
        npe = elems.shape[1]
        elem_fdof = local_to_orig_freedof[elems]  # (n_local_elems, npe); -1 = Dirichlet

        # All ordered pairs within each element (vectorised)
        rows_2d = np.repeat(elem_fdof, npe, axis=1)    # (n_elems, npe*npe)
        cols_2d = np.tile(elem_fdof, (1, npe))          # (n_elems, npe*npe)
        rows_flat = rows_2d.ravel()
        cols_flat = cols_2d.ravel()
        valid = (rows_flat >= 0) & (cols_flat >= 0)
        row_arr = rows_flat[valid]
        col_arr = cols_flat[valid]

        # J = unique original free-DOF indices in local elements
        J_arr = np.unique(local_to_orig_freedof[local_to_orig_freedof >= 0]).astype(np.int64)
        n_J = len(J_arr)

        # ---- J-indexed lookup ----
        J_to_idx = np.full(n_free, -1, dtype=np.int64)
        J_to_idx[J_arr] = np.arange(n_J, dtype=np.int64)

        # ---- Build A|_J as CSR ----
        A_J = sp.csr_matrix(
            (np.ones(len(row_arr), dtype=np.float64),
             (J_to_idx[row_arr], J_to_idx[col_arr])),
            shape=(n_J, n_J),
        )
        A_J.data[:] = 1.0   # binarise

        # ---- Store edges in original free-DOF space for _precompute_indices ----
        A_J_coo = A_J.tocoo()
        self._row_adj = J_arr[A_J_coo.row].astype(np.int64)
        self._col_adj = J_arr[A_J_coo.col].astype(np.int64)
        self.nnz_global = 0   # updated in _precompute_indices

        # ---- A²|_J = (A|_J)² ----
        A2_J = sp.csr_matrix(A_J @ A_J)

        # ---- igraph greedy coloring on A²|_J ----
        A2_J_coo = A2_J.tocoo()
        lo_tri = A2_J_coo.row > A2_J_coo.col
        edges = np.column_stack((A2_J_coo.row[lo_tri], A2_J_coo.col[lo_tri]))
        g = igraph.Graph(n_J, edges.tolist() if len(edges) > 0 else [], directed=False)
        coloring_raw = g.vertex_coloring_greedy()
        best_coloring = np.array(coloring_raw, dtype=np.int32).ravel()
        best_nc = int(best_coloring.max() + 1) if n_J > 0 else 0

        self.n_colors = best_nc
        self._local_coloring = best_coloring  # int32, length |J|
        self._J_dofs = J_arr                  # original free-DOF indices in J
        self._J_to_idx = J_to_idx             # n_free → J-index (or -1)
        self.coloring = None                  # no global coloring
        self.color_info = {
            "method": "local_variant_B_igraph_no_bcast",
            "n_J": n_J,
            "n_owned": hi - lo,
            "n_colors": best_nc,
            "trials": 1,
        }

    def _precompute_indices(self):
        """Precompute SFD index arrays using local coloring."""
        part = self.part
        freedofs_np = part.freedofs_np
        lo, hi = part.lo, part.hi
        iperm = part.iperm
        row_adj = self._row_adj
        col_adj = self._col_adj

        # ---- COO pattern (owned rows of A in reordered space) ----
        eff_rows = iperm[row_adj]
        eff_cols = iperm[col_adj]
        owned_mask = (eff_rows >= lo) & (eff_rows < hi)

        self._coo_rows = eff_rows[owned_mask].astype(PETSc.IntType)
        self._coo_cols = eff_cols[owned_mask].astype(PETSc.IntType)
        self._n_owned_nnz = int(owned_mask.sum())

        # Compute global NNZ via allreduce
        self.nnz_global = int(self.comm.allreduce(self._n_owned_nnz, op=MPI.SUM))

        # Original free-DOF indices of owned NZ entries
        owned_row_orig = row_adj[owned_mask]
        owned_col_orig = col_adj[owned_mask]

        # ---- Map owned NZ rows to local node indices ----
        total_to_local = np.full(part.n_total, -1, dtype=np.int64)
        total_to_local[part.local_to_total] = np.arange(
            part.n_local, dtype=np.int64
        )

        owned_row_total = freedofs_np[owned_row_orig]
        nz_local_rows = total_to_local[owned_row_total]
        assert np.all(nz_local_rows >= 0), (
            "BUG: some owned NZ rows map to nodes outside local domain"
        )

        # ---- Group owned NZ by LOCAL color of column DOF ----
        owned_col_J_idx = self._J_to_idx[owned_col_orig]
        assert np.all(owned_col_J_idx >= 0), (
            "BUG: some owned NZ columns not in J"
        )
        owned_col_colors = self._local_coloring[owned_col_J_idx]

        self._color_nz = {}
        for c in range(self.n_colors):
            mask_c = owned_col_colors == c
            positions = np.where(mask_c)[0].astype(np.int64)
            local_rows = nz_local_rows[positions].astype(np.int64)
            self._color_nz[c] = (positions, local_rows)

        # ---- Build local indicator vectors per LOCAL color ----
        orig_to_local = total_to_local[freedofs_np]  # (n_free,) → local or -1

        self._indicators_local = []
        for c in range(self.n_colors):
            indicator = np.zeros(part.n_local, dtype=np.float64)
            J_dofs_c = self._J_dofs[self._local_coloring == c]
            local_idx = orig_to_local[J_dofs_c]
            valid = local_idx >= 0
            indicator[local_idx[valid]] = 1.0
            self._indicators_local.append(jnp.array(indicator))

        # Stack for vmap
        self._indicators_stacked = jnp.stack(self._indicators_local)

        # ---- Store for energy / gradient callbacks ----
        self._owned_idx = jnp.array(part.owned_local_indices, dtype=jnp.int32)
        self._f_owned = jnp.array(part.f_owned, dtype=jnp.float64)

    def _compile_jax(self):
        """JIT-compile energy, gradient, and vmapped HVP for the local domain."""
        energy_weighted, energy_full = self._make_local_energy_fns()
        hvp_mode = getattr(self, "_hvp_eval_mode", "batched")
        if hvp_mode not in {"batched", "sequential"}:
            raise ValueError(f"Unsupported _hvp_eval_mode={hvp_mode!r}")
        self._hvp_eval_mode = hvp_mode

        self._energy_jit = jax.jit(energy_weighted)

        grad_fn = jax.grad(energy_full)
        self._grad_jit = jax.jit(grad_fn)

        def hvp_fn(v_local, tangent_local):
            return jax.jvp(grad_fn, (v_local,), (tangent_local,))[1]

        self._hvp_jit = jax.jit(hvp_fn)  # kept for variant 1 compatibility

        # Warmup (trigger JIT compilation)
        v_dummy = jnp.zeros(self.part.n_local, dtype=jnp.float64)
        _ = self._energy_jit(v_dummy).block_until_ready()
        _ = self._grad_jit(v_dummy).block_until_ready()
        if self._hvp_eval_mode == "batched":
            # --- Batched HVP via vmap (primary for assembly) ---
            def hvp_batched(v_local, tangents):
                """tangents: (n_colors, n_local) → (n_colors, n_local)"""
                return jax.vmap(lambda t: hvp_fn(v_local, t))(tangents)

            self._hvp_batched_jit = jax.jit(hvp_batched)
            dummy_tangents = jnp.zeros(
                (self.n_colors, self.part.n_local), dtype=jnp.float64
            )
            _ = self._hvp_batched_jit(v_dummy, dummy_tangents).block_until_ready()
        else:
            self._hvp_batched_jit = None
            _ = self._hvp_jit(v_dummy, v_dummy).block_until_ready()

    # ------------------------------------------------------------------
    # Assembly (vmap-based, P2P ghost exchange)
    # ------------------------------------------------------------------

    def _assemble_variant2(self, u_owned):
        """P2P ghost exchange + vmap all-colors HVP + COO assembly."""
        timings = {}
        t_total = time.perf_counter()

        # 1. P2P ghost exchange → build v_local
        t0 = time.perf_counter()
        v_local = self._get_v_local(u_owned)
        timings["p2p_exchange"] = time.perf_counter() - t0
        timings["allgatherv"] = timings["p2p_exchange"]

        # 2. HVP + extraction (batched or sequential, both local-only)
        owned_vals = np.zeros(self._n_owned_nnz, dtype=np.float64)
        if self._hvp_eval_mode == "batched":
            t0 = time.perf_counter()
            all_hvps = self._hvp_batched_jit(
                v_local, self._indicators_stacked
            ).block_until_ready()
            all_hvps_np = np.asarray(all_hvps)
            timings["hvp_compute"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            for c in range(self.n_colors):
                positions, local_rows = self._color_nz[c]
                if len(positions) > 0:
                    owned_vals[positions] = all_hvps_np[c, local_rows]
            timings["extraction"] = time.perf_counter() - t0
        else:
            t_hvp = 0.0
            t_extract = 0.0
            for c in range(self.n_colors):
                t0 = time.perf_counter()
                hvp_result = np.asarray(
                    self._hvp_jit(v_local, self._indicators_local[c]).block_until_ready(),
                    dtype=np.float64,
                )
                t_hvp += time.perf_counter() - t0

                t0 = time.perf_counter()
                positions, local_rows = self._color_nz[c]
                if len(positions) > 0:
                    owned_vals[positions] = hvp_result[local_rows]
                t_extract += time.perf_counter() - t0
            timings["hvp_compute"] = t_hvp
            timings["extraction"] = t_extract

        timings["n_hvps"] = self.n_colors
        timings["hvp_mode"] = self._hvp_eval_mode

        # 4. COO assembly — NO Allreduce!
        t0 = time.perf_counter()
        self.A.setValuesCOO(
            owned_vals.astype(PETSc.ScalarType),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        # Finalise matrix values explicitly to avoid hidden assembly cost in KSP.solve.
        self.A.assemble()
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        return timings

    # ------------------------------------------------------------------
    # Optional element-Hessian assembly (generic mapping, problem kernel only)
    # ------------------------------------------------------------------

    def setup_element_hessian(self):
        """Prepare analytical element-Hessian assembly on top of the SFD COO pattern."""
        t_setup_start = time.perf_counter()
        part = self.part
        comm = self.comm

        self._elem_hessian_jit = self._make_element_hessian_jit()

        lo, hi = part.lo, part.hi
        freedofs_np = part.freedofs_np
        iperm = part.iperm

        total_to_freedof = np.full(part.n_total, -1, dtype=np.int64)
        total_to_freedof[freedofs_np] = np.arange(part.n_free, dtype=np.int64)

        elems_local = part.elems_local_np
        n_elem = len(elems_local)

        elems_total = part.local_to_total[elems_local]
        elems_freedof = total_to_freedof[elems_total]
        elems_reordered = np.where(
            elems_freedof >= 0,
            iperm[np.maximum(elems_freedof, 0)],
            -1,
        )

        eff_rows_orig = iperm[self._row_adj]
        eff_cols_orig = iperm[self._col_adj]
        owned_mask_orig = (eff_rows_orig >= lo) & (eff_rows_orig < hi)
        sfd_coo_rows = eff_rows_orig[owned_mask_orig].astype(np.int64)
        sfd_coo_cols = eff_cols_orig[owned_mask_orig].astype(np.int64)
        n_sfd_nnz = len(sfd_coo_rows)

        n_global = part.n_free
        sfd_keys = sfd_coo_rows * n_global + sfd_coo_cols
        sfd_pos_map = {int(sfd_keys[k]): k for k in range(n_sfd_nnz)}

        all_sfd_pos = []
        all_scatter_e = []
        all_scatter_i = []
        all_scatter_j = []
        n_dropped = 0

        chunk_size = 10000
        for start in range(0, n_elem, chunk_size):
            end = min(start + chunk_size, n_elem)
            chunk = elems_reordered[start:end]

            row_all = chunk[:, :, None]
            col_all = chunk[:, None, :]
            row_valid = (row_all >= lo) & (row_all < hi)
            col_valid = col_all >= 0
            pair_valid = row_valid & col_valid

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
        self._n_sfd_nnz = n_sfd_nnz

        n_mapped = len(self._elem_to_sfd_pos)
        if comm.Get_rank() == 0:
            print(
                f"  Element Hessian: {n_mapped} contributions -> "
                f"{n_sfd_nnz} unique SFD COO entries",
                flush=True,
            )

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
        """Assemble Hessian via exact per-element Hessians into the SFD matrix."""
        timings = {}
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        v_local = self._get_v_local(u_owned)
        timings["p2p_exchange"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        elem_hess = self._elem_hessian_jit(v_local).block_until_ready()
        timings["elem_hessian_compute"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        elem_hess_np = np.asarray(elem_hess)
        contrib = elem_hess_np[
            self._elem_scatter_e, self._elem_scatter_i, self._elem_scatter_j
        ]
        owned_vals = np.zeros(self._n_sfd_nnz, dtype=np.float64)
        np.add.at(owned_vals, self._elem_to_sfd_pos, contrib)
        timings["scatter"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.A.zeroEntries()
        self.A.setValuesCOO(
            owned_vals.astype(PETSc.ScalarType),
            addv=PETSc.InsertMode.ADD_VALUES,
        )
        self.A.assemble()
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_total
        timings["hvp_compute"] = timings["elem_hessian_compute"]
        timings["n_hvps"] = 0
        timings["assembly_mode"] = "element"
        self.iter_timings.append(timings)
        return timings

    def get_timing_report(self):
        """Return summary dict with local coloring info."""
        report = super().get_timing_report()
        report["coloring_method"] = "local_variant_B"
        report["n_J_dofs"] = len(self._J_dofs)
        return report


class _ProblemAssemblerMixin:
    """Shared adapter from a :class:`JaxProblemSpec` to the generic bases."""

    problem_spec: JaxProblemSpec | None = None

    def _get_problem_spec(self):
        spec = self.problem_spec
        if spec is None:
            raise RuntimeError(
                f"{type(self).__name__} must define a 'problem_spec' class attribute"
            )
        return spec

    def _init_problem_state(self, params, problem_options):
        spec = self._get_problem_spec()
        options = dict(problem_options or {})
        self._problem_state = dict(spec.build_state(params, options))
        return spec

    def _problem_rhs(self, params, spec):
        if spec.rhs_key is None:
            return None
        return np.asarray(params[spec.rhs_key], dtype=np.float64)

    def _problem_elem_data(self, params, spec):
        return {
            key: np.asarray(params[key], dtype=np.float64)
            for key in spec.elem_data_keys
        }

    def _problem_near_nullspace(self, params, spec):
        if spec.make_near_nullspace is None:
            return None
        return spec.make_near_nullspace(params, self._problem_state)

    def _make_local_energy_fns(self):
        spec = self._get_problem_spec()
        return spec.make_local_energy_fns(self.part, self._problem_state)


class ProblemDOFHessianAssembler(_ProblemAssemblerMixin, DOFHessianAssemblerBase):
    """Generic global-coloring assembler driven by a :class:`JaxProblemSpec`."""

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="gamg",
        ksp_max_it=None,
        pc_options=None,
        reorder=None,
        problem_options=None,
    ):
        spec = self._init_problem_state(params, problem_options)
        if reorder is None:
            reorder = spec.default_reorder

        super().__init__(
            freedofs=np.asarray(params["freedofs"], dtype=np.int64),
            elems=np.asarray(params["elems"], dtype=np.int64),
            u_0=np.asarray(params["u_0"], dtype=np.float64),
            comm=comm,
            adjacency=adjacency,
            f=self._problem_rhs(params, spec),
            elem_data=self._problem_elem_data(params, spec),
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_max_it=ksp_max_it,
            near_nullspace_vecs=self._problem_near_nullspace(params, spec),
            pc_options=pc_options,
            reorder=reorder,
            ownership_block_size=spec.ownership_block_size,
        )


class ProblemLocalColoringAssembler(_ProblemAssemblerMixin, LocalColoringAssemblerBase):
    """Generic local-coloring assembler driven by a :class:`JaxProblemSpec`."""

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="gamg",
        ksp_max_it=None,
        pc_options=None,
        reorder=None,
        hvp_eval_mode="batched",
        problem_options=None,
    ):
        spec = self._init_problem_state(params, problem_options)
        if reorder is None:
            reorder = spec.default_reorder
        self._hvp_eval_mode = str(hvp_eval_mode)

        super().__init__(
            freedofs=np.asarray(params["freedofs"], dtype=np.int64),
            elems=np.asarray(params["elems"], dtype=np.int64),
            u_0=np.asarray(params["u_0"], dtype=np.float64),
            comm=comm,
            adjacency=adjacency,
            f=self._problem_rhs(params, spec),
            elem_data=self._problem_elem_data(params, spec),
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_max_it=ksp_max_it,
            near_nullspace_vecs=self._problem_near_nullspace(params, spec),
            pc_options=pc_options,
            reorder=reorder,
            ownership_block_size=spec.ownership_block_size,
        )

    def _make_element_hessian_jit(self):
        spec = self._get_problem_spec()
        if spec.make_element_hessian_jit is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not define an element-Hessian kernel"
            )
        return spec.make_element_hessian_jit(self.part, self._problem_state)
