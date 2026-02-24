"""
Parallel sparse-finite-difference (SFD) Hessian assembly for JAX energy functionals.

Architecture (replicated-data model)
-------------------------------------
Every MPI rank holds the full mesh data and JAX JIT-compiled functions.
Parallelism comes from two sources:

  1. **Distributed HVP computation** — graph coloring produces ``n_colors``
     colour groups.  These are assigned round-robin to ranks, so each rank
     computes only ``⌈n_colors / nprocs⌉`` Hessian-vector products per
     Newton step.
  2. **Distributed KSP solve** — the assembled Hessian is stored as a
     PETSc MPIAIJ matrix; CG + HYPRE AMG runs in parallel.

Data flow per Newton iteration
------------------------------
1.  ``allgather_vec`` — collect distributed PETSc Vec → full numpy array.
2.  Energy / gradient — each rank evaluates JAX on the full vector (identical
    result; no communication needed beyond the initial gather).
3.  HVP per colour — each rank computes its assigned HVPs.  Results are
    written into a shared ``data`` array (length = nnz of Hessian).
4.  ``MPI_Allreduce(SUM)`` on ``data`` — combines partial contributions.
5.  Each rank fills its owned rows of the PETSc MPIAIJ matrix from ``data``.
6.  PETSc KSP solve (CG + HYPRE AMG) — fully distributed.

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


class ParallelSFDSolver:
    """Parallel SFD Hessian assembly + PETSc KSP wrapper.

    Parameters
    ----------
    energy_jax_fn : callable
        JAX energy function ``J(u, **params) -> scalar``.
    params : dict
        JAX parameters (jnp arrays + scalars) passed as ``**params``.
    adjacency : scipy.sparse matrix or None
        DOF–DOF adjacency matrix (= Hessian sparsity pattern), size
        ``(n_freedofs, n_freedofs)``.  Must be available on **rank 0**;
        other ranks may pass ``None``.
    u_init : jnp.ndarray
        Initial guess — full vector of size ``n_freedofs``.
        Must be **identical** on all ranks (used for JIT warm-up).
    comm : MPI.Comm
        MPI communicator.
    coloring_trials_per_rank : int
        Multi-start coloring trials per rank (default 10).
    ksp_rtol : float
        KSP relative tolerance.
    ksp_type : str
        PETSc KSP type (``"cg"`` for SPD, ``"gmres"`` for indefinite).
    pc_type : str
        PETSc PC type (``"hypre"`` for BoomerAMG, ``"gamg"``, …).
    """

    def __init__(
        self,
        energy_jax_fn,
        params,
        adjacency,
        u_init,
        comm,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="hypre",
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.params = params
        self.timings = {}

        # ---- 1. JIT compile JAX functions --------------------------------
        t0 = time.perf_counter()
        self._compile_jax(energy_jax_fn, params, u_init)
        self.timings["jit"] = time.perf_counter() - t0

        # ---- 2. Parallel multi-start coloring ----------------------------
        t0 = time.perf_counter()
        self._setup_coloring(adjacency, comm, coloring_trials_per_rank)
        self.timings["coloring"] = time.perf_counter() - t0

        # ---- 3. Precompute SFD index mappings ----------------------------
        t0 = time.perf_counter()
        self._precompute_indices(adjacency)
        self.timings["precompute"] = time.perf_counter() - t0

        # ---- 4. Create PETSc objects (Mat, KSP, Vec helpers) -------------
        t0 = time.perf_counter()
        self._setup_petsc(ksp_rtol, ksp_type, pc_type)
        self.timings["petsc_setup"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _compile_jax(self, energy_fn, params, u_init):
        """JIT-compile energy, gradient, and Hessian-vector product."""
        self.f_jit = jax.jit(energy_fn)
        df_raw = jax.grad(energy_fn, argnums=0)
        self.df_jit = jax.jit(df_raw)

        # HVP via forward-over-reverse: JVP of the gradient function.
        # Follows the same pattern as tools/jax_diff.py EnergyDerivator.
        def _hvp_raw(x, tangent, params):
            def grad_fn(x):
                return self.df_jit(x, **params)

            return jax.jvp(grad_fn, (x,), (tangent,))[1]

        self.hvp_jit = jax.jit(_hvp_raw)

        # Warmup: trigger compilation so first solve is not penalised.
        _ = self.f_jit(u_init, **params)
        _ = self.df_jit(u_init, **params)
        _ = self.hvp_jit(u_init, u_init, params)

    def _setup_coloring(self, adjacency, comm, trials_per_rank):
        """Run MPI multi-start randomised graph coloring."""
        from graph_coloring.multistart_coloring import multistart_color

        adj_rank0 = adjacency if self.rank == 0 else None
        n_colors, coloring, color_info = multistart_color(
            adj_rank0, comm, trials_per_rank=trials_per_rank
        )
        self.n_colors = n_colors
        self.coloring = coloring  # int32 array, length n
        self.n = len(coloring)
        self.color_info = color_info

    def _precompute_indices(self, adjacency):
        """Precompute all index arrays needed for distributed SFD assembly.

        After this method, every rank has:
        - ``row_adj, col_adj`` — adjacency (= Hessian sparsity) nonzeros
        - ``color_indicators`` — one jnp indicator vector per colour
        - ``my_colors`` — colours assigned to this rank
        - ``color_nz_map[c]`` — (nz_indices, nz_rows) for each colour
        """
        # --- broadcast adjacency nonzeros from rank 0 ---
        if self.rank == 0:
            adj_csr = adjacency.tocsr()
            row_adj, col_adj = adj_csr.nonzero()
            row_adj = np.ascontiguousarray(row_adj, dtype=np.int64)
            col_adj = np.ascontiguousarray(col_adj, dtype=np.int64)
            nnz = np.int64(len(row_adj))
        else:
            nnz = np.int64(0)

        nnz = self.comm.bcast(int(nnz), root=0)
        if self.rank != 0:
            row_adj = np.empty(nnz, dtype=np.int64)
            col_adj = np.empty(nnz, dtype=np.int64)
        self.comm.Bcast(row_adj, root=0)
        self.comm.Bcast(col_adj, root=0)

        self.row_adj = row_adj
        self.col_adj = col_adj
        self.nnz = nnz

        # --- colour indicator vectors (jnp, one per colour) ---
        self.color_indicators = []
        for c in range(self.n_colors):
            v = np.zeros(self.n, dtype=np.float64)
            v[self.coloring == c] = 1.0
            self.color_indicators.append(jnp.array(v))

        # --- round-robin colour assignment ---
        self.my_colors = list(range(self.rank, self.n_colors, self.size))

        # --- for each colour: which nnz entries does it produce? ---
        #
        # For nonzero (row_adj[k], col_adj[k]), the value comes from the
        # HVP for colour c = coloring[col_adj[k]].  So colour c contributes
        # all entries k where coloring[col_adj[k]] == c.
        color_of_col = self.coloring[col_adj]
        self.color_nz_map = {}
        for c in range(self.n_colors):
            nz_indices = np.where(color_of_col == c)[0]
            nz_rows = row_adj[nz_indices]
            self.color_nz_map[c] = (nz_indices, nz_rows)

    def _setup_petsc(self, ksp_rtol, ksp_type, pc_type):
        """Create PETSc MPIAIJ matrix, determine ownership, build KSP."""
        n = self.n

        # Determine row ownership range (PETSc default block distribution).
        tmp = PETSc.Vec().createMPI(n, comm=self.comm)
        self.lo, self.hi = tmp.getOwnershipRange()
        self.n_local = self.hi - self.lo
        tmp.destroy()

        # --- MPIAIJ preallocation (d_nnz / o_nnz per owned row) ---
        owned_mask = (self.row_adj >= self.lo) & (self.row_adj < self.hi)
        local_rows = self.row_adj[owned_mask] - self.lo
        owned_cols = self.col_adj[owned_mask]
        diag_mask = (owned_cols >= self.lo) & (owned_cols < self.hi)

        d_nnz = np.bincount(
            local_rows[diag_mask], minlength=self.n_local
        ).astype(np.int32)
        o_nnz = np.bincount(
            local_rows[~diag_mask], minlength=self.n_local
        ).astype(np.int32)

        self.A = PETSc.Mat().createAIJ(
            size=(n, n), nnz=(d_nnz, o_nnz), comm=self.comm
        )
        self.A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        self.A.setUp()

        # --- precompute owned-row CSR for fast value insertion ---
        #
        # For each owned row, store the column indices and their positions
        # in the global ``data`` array so that ``setValues`` can be called
        # row-by-row with precomputed slicing.
        order = np.argsort(local_rows)
        self._owned_local_rows = local_rows[order]
        self._owned_cols_sorted = owned_cols[order]
        self._owned_data_idx = np.where(owned_mask)[0][order]

        # CSR-style indptr over owned rows
        indptr = np.zeros(self.n_local + 1, dtype=np.int64)
        np.add.at(indptr[1:], self._owned_local_rows, 1)
        self._owned_indptr = np.cumsum(indptr)

        # --- Allgatherv displacement table (reused every gather) ---
        self._gather_sizes = np.array(
            self.comm.allgather(self.n_local), dtype=np.int64
        )
        self._gather_displs = np.zeros(len(self._gather_sizes), dtype=np.int64)
        np.cumsum(self._gather_sizes[:-1], out=self._gather_displs[1:])

        # --- KSP (CG + HYPRE by default) ---
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setType(ksp_type)
        self.ksp.getPC().setType(pc_type)
        self.ksp.setTolerances(rtol=ksp_rtol)
        self.ksp.setFromOptions()

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def allgather_vec(self, vec):
        """Gather distributed PETSc Vec → full numpy array on every rank."""
        local = np.asarray(vec.array, dtype=np.float64).copy()
        full = np.empty(self.n, dtype=np.float64)
        self.comm.Allgatherv(
            local,
            [full, self._gather_sizes, self._gather_displs, MPI.DOUBLE],
        )
        return full

    def create_vec(self, full_array=None):
        """Create a distributed PETSc Vec (optionally filled from a full array).

        The ownership range matches ``self.lo : self.hi``.
        """
        v = PETSc.Vec().createMPI(self.n, comm=self.comm)
        if full_array is not None:
            arr = np.asarray(full_array, dtype=np.float64)
            v.array[:] = arr[self.lo:self.hi]
            v.assemble()
        return v

    # ------------------------------------------------------------------
    # Newton callbacks
    # ------------------------------------------------------------------

    def energy_fn(self, vec):
        """Evaluate energy J(u) at a distributed PETSc Vec.

        Every rank computes the *same* scalar (replicated data), so no
        ``allreduce`` is needed — only an ``Allgatherv`` to collect
        the full *u* vector.
        """
        u_full = self.allgather_vec(vec)
        return float(self.f_jit(jnp.array(u_full), **self.params))

    def gradient_fn(self, vec, g):
        """Assemble ∇J(u) into distributed PETSc Vec *g*.

        Every rank computes the full gradient via ``jax.grad`` and copies
        its owned slice into *g*.
        """
        u_full = self.allgather_vec(vec)
        grad_full = np.asarray(
            self.df_jit(jnp.array(u_full), **self.params), dtype=np.float64
        )
        g.array[:] = grad_full[self.lo:self.hi]

    def hessian_solve_fn(self, vec, rhs, sol):
        """Assemble Hessian via distributed SFD + KSP solve.

        Steps:
            1. Allgather *u* from distributed ``vec``.
            2. Each rank computes HVPs for its assigned colours.
            3. ``MPI_Allreduce(SUM)`` combines partial ``data`` arrays.
            4. Each rank fills its owned MPIAIJ rows.
            5. PETSc KSP solve: ``H · sol = rhs``.

        Returns the number of KSP iterations.
        """
        u_full = self.allgather_vec(vec)
        u_jnp = jnp.array(u_full)

        # --- 1. distributed HVP computation ---
        data = np.zeros(self.nnz, dtype=np.float64)
        for c in self.my_colors:
            hvp_result = np.asarray(
                self.hvp_jit(u_jnp, self.color_indicators[c], self.params),
                dtype=np.float64,
            )
            nz_indices, nz_rows = self.color_nz_map[c]
            data[nz_indices] = hvp_result[nz_rows]

        # --- 2. combine across all ranks ---
        self.comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)

        # --- 3. fill PETSc MPIAIJ matrix (owned rows only) ---
        self.A.zeroEntries()
        for li in range(self.n_local):
            s = self._owned_indptr[li]
            e = self._owned_indptr[li + 1]
            if s < e:
                gi = self.lo + li
                cols = self._owned_cols_sorted[s:e]
                vals = data[self._owned_data_idx[s:e]]
                self.A.setValues(
                    [gi], cols.tolist(), vals.tolist(), addv=PETSc.InsertMode.INSERT
                )
        self.A.assemble()

        # --- 4. KSP solve ---
        self.ksp.setOperators(self.A)
        self.ksp.solve(rhs, sol)
        return self.ksp.getIterationNumber()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Destroy PETSc objects."""
        self.ksp.destroy()
        self.A.destroy()
