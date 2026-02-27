"""
Generic DOF-based domain decomposition for parallel computation.

Each MPI rank owns a contiguous range of free DOFs (matching PETSc's
default block distribution).  The rank's "local domain" consists of ALL
elements that touch at least one owned DOF.  This ensures:

  - gradient at owned DOFs is EXACT  (no Allreduce needed)
  - HVP at owned DOFs is EXACT       (no Allreduce needed)
  - energy needs only a SCALAR Allreduce (trivial cost)

Communication: P2P ghost exchange of free-DOF values before each
computation (to fill ghost values in local domain).

**RCM reordering** (Reverse Cuthill-McKee) is applied by default so that
the PETSc block distribution corresponds to spatially compact subdomains,
minimising the element overlap between ranks.

Aligns with the PETSc MPIAIJ block distribution used for AMG KSP solves.
"""

import numpy as np
import time
from mpi4py import MPI


def petsc_ownership_range(n, rank, size, block_size=1):
    """Compute PETSc-style ownership range [lo, hi).

    PETSc distributes N items across P ranks:
      floor(N/P)+1  items to the first  N%P ranks,
      floor(N/P)    items to the remaining ranks.

    If ``block_size > 1``, distribution is performed in block space first
    (``n / block_size`` blocks), then mapped back to scalar indices.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if block_size > 1:
        if n % block_size != 0:
            raise ValueError(
                f"n={n} not divisible by block_size={block_size} for ownership split"
            )
        n_blocks = n // block_size
        lo_b, hi_b = petsc_ownership_range(n_blocks, rank, size, block_size=1)
        return lo_b * block_size, hi_b * block_size

    q, r = divmod(n, size)
    if rank < r:
        lo = rank * (q + 1)
        hi = lo + q + 1
    else:
        lo = rank * q + r
        hi = lo + q
    return lo, hi


class DOFPartition:
    """Generic DOF-based domain decomposition for one MPI rank.

    After construction, the rank has:
      - ``lo, hi``:  owned free-DOF range [lo, hi) in **reordered** space
      - ``elems_local_np``:  local element connectivity (local node indices)
      - ``local_elem_data``:  dict of locally-sliced element arrays
      - ``elem_weights``:  1.0 for uniquely-owned elements, 0.0 otherwise
      - ``owned_local_indices``:  local indices of owned free DOFs
      - ``v_template``:  pre-filled Dirichlet values (numpy)
      - methods to build v_local from u_owned via P2P ghost exchange

    Parameters
    ----------
    freedofs : array-like
        Free DOF indices into the full node/DOF vector.
    elems : array-like
        Element connectivity (total-node indices), shape (n_elems, npe).
    u_0 : array-like
        Full DOF vector with Dirichlet boundary values.
    comm : MPI.Comm
        MPI communicator.
    adjacency : scipy.sparse matrix or None
        DOF-DOF adjacency (Hessian sparsity pattern), (n_free, n_free).
        Required on rank 0 for RCM reordering; other ranks may pass None.
    reorder : bool
        Whether to apply RCM reordering (default True).
    f : array-like or None
        RHS/load vector (same length as u_0). If None, f_owned is zeros.
    elem_data : dict or None
        Element-level arrays to slice by local elements.
        Keys are names, values are (n_elems, ...) arrays.
    """

    def __init__(self, freedofs, elems, u_0, comm, adjacency=None, reorder=True,
                 f=None, elem_data=None, ownership_block_size=1):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.timings = {}

        t_total = time.perf_counter()

        # ---- 1. Convert to numpy ----
        t0 = time.perf_counter()
        freedofs = np.asarray(freedofs, dtype=np.int64)
        elems = np.asarray(elems, dtype=np.int64)
        u_0 = np.asarray(u_0, dtype=np.float64)
        self.timings["numpy_convert"] = time.perf_counter() - t0

        n_free = len(freedofs)
        n_total = len(u_0)
        self.n_free = n_free
        self.n_total = n_total
        self.freedofs_np = freedofs
        self.ownership_block_size = int(ownership_block_size)
        if self.ownership_block_size < 1:
            raise ValueError("ownership_block_size must be >= 1")

        # ---- 2. RCM reordering (rank 0 computes, broadcast) ----
        t0 = time.perf_counter()
        if reorder:
            if self.rank == 0:
                from scipy.sparse.csgraph import reverse_cuthill_mckee
                perm = reverse_cuthill_mckee(adjacency.tocsr())
                perm = np.ascontiguousarray(perm, dtype=np.int64)
            else:
                perm = np.empty(n_free, dtype=np.int64)
            comm.Bcast(perm, root=0)
            # perm[reordered_idx] = original_idx
            iperm = np.empty_like(perm)
            iperm[perm] = np.arange(n_free, dtype=np.int64)
            # iperm[original_idx] = reordered_idx
        else:
            perm = np.arange(n_free, dtype=np.int64)
            iperm = np.arange(n_free, dtype=np.int64)

        self.perm = perm    # perm[new] = old
        self.iperm = iperm  # iperm[old] = new
        self.timings["rcm_reorder"] = time.perf_counter() - t0

        # ---- 3. DOF ownership (PETSc block distribution in reordered space) ----
        t0 = time.perf_counter()
        lo, hi = petsc_ownership_range(
            n_free, self.rank, self.size, block_size=self.ownership_block_size
        )
        self.lo = lo
        self.hi = hi
        self.n_owned = hi - lo

        # ---- 4. Total-node ↔ reordered-free-DOF mapping ----
        total_to_free_orig = np.full(n_total, -1, dtype=np.int64)
        total_to_free_orig[freedofs] = np.arange(n_free, dtype=np.int64)

        total_to_free_reord = np.full(n_total, -1, dtype=np.int64)
        free_mask_total = total_to_free_orig >= 0
        total_to_free_reord[free_mask_total] = iperm[total_to_free_orig[free_mask_total]]

        # ---- 5. Find local elements (touch at least one owned reordered DOF) ----
        elem_reord_idx = total_to_free_reord[elems]  # (n_elems, npe); -1 = Dirichlet
        has_owned = np.any(
            (elem_reord_idx >= lo) & (elem_reord_idx < hi), axis=1
        )
        local_elem_idx = np.where(has_owned)[0]
        n_local_elems = len(local_elem_idx)
        self.local_elem_idx = local_elem_idx

        # ---- 6. Local element data (generic slicing) ----
        local_elems_total = elems[local_elem_idx]
        self.local_elem_data = {}
        if elem_data is not None:
            for name, arr in elem_data.items():
                arr_np = np.asarray(arr, dtype=np.float64)
                self.local_elem_data[name] = arr_np[local_elem_idx]

        # ---- 7. Element weights for energy (unique assignment) ----
        # Assign element to rank owning its minimum reordered free DOF.
        local_elem_reord = total_to_free_reord[local_elems_total]
        masked = np.where(local_elem_reord >= 0, local_elem_reord,
                          np.int64(n_free))
        elem_min_reord = np.min(masked, axis=1)

        # Vectorised rank assignment
        self.elem_weights = np.zeros(n_local_elems, dtype=np.float64)
        valid = elem_min_reord < n_free
        if np.any(valid):
            owners = _rank_of_dof_vec(
                elem_min_reord[valid],
                n_free,
                self.size,
                block_size=self.ownership_block_size,
            )
            self.elem_weights[valid] = np.where(owners == self.rank, 1.0, 0.0)

        # ---- 8. Local node space ----
        local_total_nodes, inverse = np.unique(
            local_elems_total.ravel(), return_inverse=True
        )
        self.n_local = len(local_total_nodes)
        self.local_to_total = local_total_nodes
        self.elems_local_np = inverse.reshape(n_local_elems, elems.shape[1]).astype(np.int32)
        self.n_local_elems = n_local_elems

        # ---- 9. Identify DOF types in local node space ----
        local_reord_idx = total_to_free_reord[local_total_nodes]  # -1 = Dirichlet

        # Owned free DOFs (reordered idx in [lo, hi))
        # CRITICAL: must be sorted by reordered index so that
        # extracted gradient[i] corresponds to reordered DOF (lo+i).
        owned_mask = (local_reord_idx >= lo) & (local_reord_idx < hi)
        owned_positions = np.where(owned_mask)[0]
        owned_reord_vals = local_reord_idx[owned_positions]
        sort_order = np.argsort(owned_reord_vals)
        self.owned_local_indices = owned_positions[sort_order].astype(np.int32)
        assert len(self.owned_local_indices) == self.n_owned, (
            f"Expected {self.n_owned} owned DOFs, found {len(self.owned_local_indices)}"
        )

        # All free DOFs in local space: their reordered-free-DOF indices
        free_mask = local_reord_idx >= 0
        self.free_local_indices = np.where(free_mask)[0]
        self.free_global_indices = local_reord_idx[free_mask]  # reordered free-DOF indices

        # ---- 10. Pre-fill Dirichlet values in v_template ----
        v_template = np.zeros(self.n_local, dtype=np.float64)
        dirichlet_mask = ~free_mask
        if np.any(dirichlet_mask):
            v_template[dirichlet_mask] = u_0[local_total_nodes[dirichlet_mask]]
        self.v_template = v_template
        self._dirichlet_local_mask = dirichlet_mask

        # ---- 11. f vector at owned free DOFs (reordered order) ----
        owned_orig_free = perm[lo:hi]
        if f is not None:
            f_np = np.asarray(f, dtype=np.float64)
            self.f_owned = f_np[freedofs[owned_orig_free]].astype(np.float64)
        else:
            self.f_owned = np.zeros(self.n_owned, dtype=np.float64)

        # ---- 12. Allgatherv sizes and displacements ----
        self._gather_sizes = np.array(
            comm.allgather(self.n_owned), dtype=np.int64
        )
        self._gather_displs = np.zeros(self.size, dtype=np.int64)
        np.cumsum(self._gather_sizes[:-1], out=self._gather_displs[1:])

        # ---- 13. Point-to-point ghost exchange setup ----
        self._setup_ghost_exchange()

        self.timings["partition_build"] = time.perf_counter() - t0
        self.timings["total_setup_partition"] = time.perf_counter() - t_total

    # ------------------------------------------------------------------
    # Ghost exchange setup and P2P methods
    # ------------------------------------------------------------------

    def _setup_ghost_exchange(self):
        """Pre-compute point-to-point ghost exchange plans.

        Instead of Allgatherv of the full free-DOF vector, each rank
        exchanges ONLY the ghost DOF values it needs from neighbour ranks.
        """
        lo, hi = self.lo, self.hi
        n_free = self.n_free
        comm = self.comm

        # --- Split local free DOFs into owned and ghost ---
        free_global = self.free_global_indices  # reordered free-DOF indices
        owned_mask = (free_global >= lo) & (free_global < hi)
        ghost_mask = ~owned_mask

        # Owned: fill directly from u_owned[offset]
        self._p2p_owned_local = self.free_local_indices[owned_mask]
        self._p2p_owned_offset = (free_global[owned_mask] - lo).astype(np.int64)

        # Ghost: group by owning rank
        ghost_local = self.free_local_indices[ghost_mask]
        ghost_global = free_global[ghost_mask]

        # _ghost_recv[r] = local-node indices where values from rank r go
        self._ghost_recv = {}
        # send_requests[r] = offsets into rank r's u_owned that we need
        send_requests = {}

        if len(ghost_global) > 0:
            ghost_owners = _rank_of_dof_vec(
                ghost_global,
                n_free,
                self.size,
                block_size=self.ownership_block_size,
            )
            for r in np.unique(ghost_owners):
                r = int(r)
                if r == self.rank:
                    continue
                mask_r = ghost_owners == r
                r_lo, _ = petsc_ownership_range(
                    n_free, r, self.size, block_size=self.ownership_block_size
                )
                self._ghost_recv[r] = ghost_local[mask_r]
                send_requests[r] = (ghost_global[mask_r] - r_lo).astype(np.int64)

        # --- Exchange counts (how many DOFs each rank needs from each other) ---
        n_we_need = np.zeros(self.size, dtype=np.int64)
        for r, offsets in send_requests.items():
            n_we_need[r] = len(offsets)
        n_others_need = np.zeros(self.size, dtype=np.int64)
        comm.Alltoall(n_we_need, n_others_need)

        # --- Exchange index lists (which offsets into our u_owned do others need?) ---
        # Receive requests from other ranks
        recv_reqs = []
        self._ghost_send_offsets = {}
        for r in range(self.size):
            if r == self.rank or n_others_need[r] == 0:
                continue
            buf = np.empty(int(n_others_need[r]), dtype=np.int64)
            req = comm.Irecv(buf, source=r, tag=200)
            recv_reqs.append((req, r, buf))

        # Send our requests to owner ranks
        send_reqs = []
        for r, offsets in send_requests.items():
            req = comm.Isend(np.ascontiguousarray(offsets), dest=r, tag=200)
            send_reqs.append(req)

        # Complete receives: now we know which of our DOFs each rank needs
        for req, r, buf in recv_reqs:
            req.Wait()
            self._ghost_send_offsets[r] = buf

        for req in send_reqs:
            req.Wait()

        # --- Pre-allocate value send/recv buffers ---
        self._ghost_send_bufs = {
            r: np.empty(len(offsets), dtype=np.float64)
            for r, offsets in self._ghost_send_offsets.items()
        }
        self._ghost_recv_bufs = {
            r: np.empty(len(local_idx), dtype=np.float64)
            for r, local_idx in self._ghost_recv.items()
        }

        self.n_ghost = int(ghost_mask.sum())
        self.n_neighbors = len(
            set(self._ghost_recv.keys()) | set(self._ghost_send_offsets.keys())
        )

    def _p2p_fill(self, template, u_owned, tag):
        """Fill a local-node vector: owned DOFs from u_owned, ghosts via P2P.

        Parameters
        ----------
        template : ndarray or None
            If not None, start from template.copy() (use v_template for
            Dirichlet values).  If None, start from zeros.
        u_owned : ndarray, shape (n_owned,)
            This rank's owned free-DOF values (contiguous float64).
        tag : int
            MPI tag for the P2P exchange.
        """
        u_owned = np.ascontiguousarray(u_owned, dtype=np.float64)

        if template is not None:
            v = template.copy()
        else:
            v = np.zeros(self.n_local, dtype=np.float64)

        # Fill owned free DOFs directly from u_owned
        v[self._p2p_owned_local] = u_owned[self._p2p_owned_offset]

        if not self._ghost_recv and not self._ghost_send_offsets:
            return v  # single rank or no ghosts

        # Post non-blocking receives
        recv_reqs = []
        for r, buf in self._ghost_recv_bufs.items():
            req = self.comm.Irecv(buf, source=r, tag=tag)
            recv_reqs.append((req, r))

        # Pack and send
        send_reqs = []
        for r, offsets in self._ghost_send_offsets.items():
            self._ghost_send_bufs[r][:] = u_owned[offsets]
            req = self.comm.Isend(self._ghost_send_bufs[r], dest=r, tag=tag)
            send_reqs.append(req)

        # Complete receives and scatter into v
        for req, r in recv_reqs:
            req.Wait()
            v[self._ghost_recv[r]] = self._ghost_recv_bufs[r]

        # Complete sends
        for req in send_reqs:
            req.Wait()

        return v

    def build_v_local_p2p(self, u_owned):
        """Build local node-value vector using P2P ghost exchange.

        Fills Dirichlet values from v_template, owned free DOFs from u_owned,
        ghost free DOFs via point-to-point Isend/Irecv.
        """
        return self._p2p_fill(self.v_template, u_owned, tag=42)

    def build_zero_local_p2p(self, u_owned):
        """Build tangent-like local vector using P2P ghost exchange.

        Zero at Dirichlet DOFs, owned free DOFs from u_owned,
        ghost free DOFs via point-to-point Isend/Irecv.
        """
        return self._p2p_fill(None, u_owned, tag=43)

    def update_dirichlet(self, u_0_new):
        """Update Dirichlet boundary values in v_template.

        Call this when boundary conditions change between solves
        (e.g., load stepping in hyperelasticity).

        Parameters
        ----------
        u_0_new : array-like
            New full DOF vector with updated Dirichlet boundary values.
        """
        u_0_new = np.asarray(u_0_new, dtype=np.float64)
        mask = self._dirichlet_local_mask
        self.v_template[mask] = u_0_new[self.local_to_total[mask]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_u_full(self, u_owned):
        """Allgather owned free-DOFs → full free-DOF vector (reordered order)."""
        u_full = np.empty(self.n_free, dtype=np.float64)
        self.comm.Allgatherv(
            np.ascontiguousarray(u_owned, dtype=np.float64),
            [u_full, self._gather_sizes, self._gather_displs, MPI.DOUBLE],
        )
        return u_full

    def build_v_local(self, u_full):
        """Build local node-value vector from full (reordered) free-DOF vector.

        Starts from Dirichlet template, fills in free-DOF values.
        Returns numpy array of shape (n_local,).
        """
        v = self.v_template.copy()
        v[self.free_local_indices] = u_full[self.free_global_indices]
        return v

    def original_to_reordered(self, u_orig):
        """Convert full free-DOF vector from original → reordered order."""
        return u_orig[self.perm]

    def reordered_to_original(self, u_reord):
        """Convert full free-DOF vector from reordered → original order."""
        u_orig = np.empty_like(u_reord)
        u_orig[self.perm] = u_reord
        return u_orig


def _rank_of_dof_vec(dof_indices, n_total, size, block_size=1):
    """Vectorised: determine owning rank for each DOF index (PETSc dist)."""
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if block_size > 1:
        if n_total % block_size != 0:
            raise ValueError(
                f"n_total={n_total} not divisible by block_size={block_size}"
            )
        dof_indices = np.asarray(dof_indices, dtype=np.int64)
        block_idx = dof_indices // block_size
        return _rank_of_dof_vec(block_idx, n_total // block_size, size, block_size=1)

    q, r = divmod(n_total, size)
    boundary = r * (q + 1)
    result = np.empty(len(dof_indices), dtype=np.int64)
    below = dof_indices < boundary
    result[below] = dof_indices[below] // (q + 1)
    result[~below] = r + (dof_indices[~below] - boundary) // q
    return result
