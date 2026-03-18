"""Shared reordered overlap-domain element assembler scaffold."""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee

from src.core.petsc.dof_partition import _rank_of_dof_vec, petsc_ownership_range


@dataclass
class GlobalLayout:
    perm: np.ndarray
    iperm: np.ndarray
    lo: int
    hi: int
    n_free: int
    total_to_free_reord: np.ndarray
    coo_rows: np.ndarray
    coo_cols: np.ndarray
    owned_mask: np.ndarray
    owned_rows: np.ndarray
    owned_cols: np.ndarray
    global_key_to_pos: dict[int, int]
    owned_key_to_pos: dict[int, int]
    elem_owner: np.ndarray


@dataclass
class LocalOverlapData:
    local_elem_idx: np.ndarray
    local_total_nodes: np.ndarray
    elems_local_np: np.ndarray
    elems_reordered: np.ndarray
    local_elem_data: dict[str, np.ndarray]
    energy_weights: np.ndarray


@dataclass
class ScatterData:
    owned_local_pos: np.ndarray
    hess_e: np.ndarray
    hess_i: np.ndarray
    hess_j: np.ndarray
    hess_positions: np.ndarray


def _build_block_graph(adjacency: sparse.spmatrix, block_size: int) -> sparse.csr_matrix:
    rows, cols = adjacency.nonzero()
    br = rows // block_size
    bc = cols // block_size
    n_blocks = adjacency.shape[0] // block_size
    block = sparse.coo_matrix(
        (np.ones_like(br, dtype=np.int8), (br, bc)),
        shape=(n_blocks, n_blocks),
    ).tocsr()
    block.data[:] = 1
    block.eliminate_zeros()
    return block


def _expand_block_perm(block_perm: np.ndarray, block_size: int) -> np.ndarray:
    if int(block_size) <= 1:
        return np.asarray(block_perm, dtype=np.int64)
    perm = np.empty(block_perm.size * block_size, dtype=np.int64)
    for comp in range(block_size):
        perm[comp::block_size] = block_perm * block_size + comp
    return perm


def perm_identity(n_free: int) -> np.ndarray:
    return np.arange(n_free, dtype=np.int64)


def perm_block_rcm(adjacency: sparse.spmatrix, block_size: int) -> np.ndarray:
    graph = (
        adjacency.tocsr()
        if int(block_size) <= 1
        else _build_block_graph(adjacency, int(block_size))
    )
    block_perm = np.asarray(
        reverse_cuthill_mckee(graph, symmetric_mode=True), dtype=np.int64
    )
    return _expand_block_perm(block_perm, int(block_size))


def perm_block_xyz(
    coords_all: np.ndarray,
    freedofs: np.ndarray,
    block_size: int,
) -> np.ndarray:
    block_size = int(block_size)
    freedofs_arr = np.asarray(freedofs, dtype=np.int64)
    node_ids = freedofs_arr[::block_size] // block_size
    coords = np.asarray(coords_all[node_ids], dtype=np.float64)
    sort_keys = tuple(coords[:, dim] for dim in reversed(range(coords.shape[1])))
    block_perm = np.lexsort(sort_keys).astype(np.int64)
    return _expand_block_perm(block_perm, block_size)


def perm_block_metis(
    adjacency: sparse.spmatrix,
    block_size: int,
    n_parts: int,
) -> np.ndarray:
    import pymetis

    graph = (
        adjacency.tocsr()
        if int(block_size) <= 1
        else _build_block_graph(adjacency, int(block_size))
    )
    _, part = pymetis.part_graph(n_parts, xadj=graph.indptr, adjncy=graph.indices)
    part = np.asarray(part, dtype=np.int64)
    block_ids = np.arange(graph.shape[0], dtype=np.int64)
    block_perm = np.lexsort((block_ids, part)).astype(np.int64)
    return _expand_block_perm(block_perm, int(block_size))


def select_permutation(
    reorder_mode: str,
    *,
    adjacency: sparse.spmatrix,
    coords_all: np.ndarray,
    freedofs: np.ndarray,
    n_parts: int,
    block_size: int,
) -> np.ndarray:
    if reorder_mode == "none":
        return perm_identity(len(freedofs))
    if reorder_mode == "block_rcm":
        return perm_block_rcm(adjacency, block_size)
    if reorder_mode == "block_xyz":
        return perm_block_xyz(coords_all, freedofs, block_size)
    if reorder_mode == "block_metis":
        return perm_block_metis(adjacency, block_size, n_parts)
    raise ValueError(f"Unsupported element reorder mode: {reorder_mode!r}")


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    iperm = np.empty_like(perm)
    iperm[perm] = np.arange(perm.size, dtype=np.int64)
    return iperm


def local_vec_from_full(
    full_reordered: np.ndarray,
    total_to_free_reord: np.ndarray,
    local_total_nodes: np.ndarray,
    dirichlet_full: np.ndarray,
) -> np.ndarray:
    local_reord = total_to_free_reord[local_total_nodes]
    v_local = np.asarray(dirichlet_full[local_total_nodes], dtype=np.float64).copy()
    free_mask = local_reord >= 0
    if np.any(free_mask):
        v_local[free_mask] = full_reordered[local_reord[free_mask]]
    return v_local


def build_global_layout(
    params: dict,
    adjacency: sparse.spmatrix,
    perm: np.ndarray,
    comm: MPI.Comm,
    *,
    block_size: int,
    dirichlet_key: str,
) -> GlobalLayout:
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    elems = np.asarray(params["elems"], dtype=np.int64)
    n_total = int(len(np.asarray(params[dirichlet_key], dtype=np.float64)))
    n_free = int(freedofs.size)
    iperm = inverse_permutation(perm)
    lo, hi = petsc_ownership_range(
        n_free, comm.rank, comm.size, block_size=int(block_size)
    )

    total_to_free_orig = np.full(n_total, -1, dtype=np.int64)
    total_to_free_orig[freedofs] = np.arange(n_free, dtype=np.int64)
    total_to_free_reord = np.full(n_total, -1, dtype=np.int64)
    mask = total_to_free_orig >= 0
    total_to_free_reord[mask] = iperm[total_to_free_orig[mask]]

    row_adj, col_adj = adjacency.tocsr().nonzero()
    coo_rows = iperm[np.asarray(row_adj, dtype=np.int64)]
    coo_cols = iperm[np.asarray(col_adj, dtype=np.int64)]
    owned_mask = (coo_rows >= lo) & (coo_rows < hi)
    owned_rows = np.asarray(coo_rows[owned_mask], dtype=np.int64)
    owned_cols = np.asarray(coo_cols[owned_mask], dtype=np.int64)

    key_base = np.int64(n_free)
    global_keys = (
        np.asarray(coo_rows, dtype=np.int64) * key_base
        + np.asarray(coo_cols, dtype=np.int64)
    )
    owned_keys = (
        np.asarray(owned_rows, dtype=np.int64) * key_base
        + np.asarray(owned_cols, dtype=np.int64)
    )
    global_key_to_pos = {int(k): i for i, k in enumerate(global_keys.tolist())}
    owned_key_to_pos = {int(k): i for i, k in enumerate(owned_keys.tolist())}

    elems_reordered = total_to_free_reord[elems]
    masked = np.where(elems_reordered >= 0, elems_reordered, np.int64(n_free))
    elem_min = np.min(masked, axis=1)
    valid = elem_min < n_free
    elem_owner = np.full(len(elems), -1, dtype=np.int64)
    if np.any(valid):
        elem_owner[valid] = _rank_of_dof_vec(
            elem_min[valid],
            n_free,
            comm.size,
            block_size=int(block_size),
        )

    return GlobalLayout(
        perm=perm,
        iperm=iperm,
        lo=lo,
        hi=hi,
        n_free=n_free,
        total_to_free_reord=total_to_free_reord,
        coo_rows=np.asarray(coo_rows, dtype=np.int64),
        coo_cols=np.asarray(coo_cols, dtype=np.int64),
        owned_mask=np.asarray(owned_mask, dtype=bool),
        owned_rows=owned_rows,
        owned_cols=owned_cols,
        global_key_to_pos=global_key_to_pos,
        owned_key_to_pos=owned_key_to_pos,
        elem_owner=elem_owner,
    )


def build_local_overlap_data(
    params: dict,
    layout: GlobalLayout,
    comm: MPI.Comm,
    *,
    elem_data_keys: tuple[str, ...],
) -> LocalOverlapData:
    elems = np.asarray(params["elems"], dtype=np.int64)
    elem_reordered = layout.total_to_free_reord[elems]
    local_mask = np.any(
        (elem_reordered >= layout.lo) & (elem_reordered < layout.hi), axis=1
    )
    local_elem_idx = np.where(local_mask)[0].astype(np.int64)
    local_energy_weights = (layout.elem_owner[local_elem_idx] == comm.rank).astype(
        np.float64
    )

    local_elems_total = elems[local_elem_idx]
    local_total_nodes, inverse = np.unique(
        local_elems_total.ravel(), return_inverse=True
    )
    elems_local_np = inverse.reshape(local_elems_total.shape).astype(np.int32)

    local_elem_data = {
        key: np.asarray(params[key], dtype=np.float64)[local_elem_idx]
        for key in elem_data_keys
    }

    return LocalOverlapData(
        local_elem_idx=local_elem_idx,
        local_total_nodes=np.asarray(local_total_nodes, dtype=np.int64),
        elems_local_np=elems_local_np,
        elems_reordered=np.asarray(
            layout.total_to_free_reord[local_elems_total], dtype=np.int64
        ),
        local_elem_data=local_elem_data,
        energy_weights=local_energy_weights,
    )


def build_near_nullspace(
    layout: GlobalLayout,
    params: dict,
    comm: MPI.Comm,
    *,
    kernel_key: str,
) -> PETSc.NullSpace:
    kernel = np.asarray(params[kernel_key], dtype=np.float64)
    vecs = []
    for i in range(kernel.shape[1]):
        mode = kernel[:, i][layout.perm]
        vec = PETSc.Vec().createMPI((layout.hi - layout.lo, layout.n_free), comm=comm)
        vec.array[:] = mode[layout.lo : layout.hi]
        vec.assemble()
        vecs.append(vec)
    return PETSc.NullSpace().create(vectors=vecs)


class ReorderedElementAssemblerBase:
    """Generic reordered overlap-domain PETSc assembler."""

    distribution_strategy = "overlap_allgather"
    block_size = 1
    coordinate_key = "nodes"
    dirichlet_key = "u_0"
    local_elem_data_keys: tuple[str, ...] = ()
    near_nullspace_key: str | None = None

    def __init__(
        self,
        params,
        comm,
        adjacency,
        *,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="gamg",
        ksp_max_it=10000,
        pc_options=None,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
        use_near_nullspace=False,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.params = params
        self.reorder_mode = str(reorder_mode)
        self.local_hessian_mode = str(local_hessian_mode)
        self.use_near_nullspace = bool(use_near_nullspace)
        self.iter_timings = []
        self._hvp_eval_mode = "element_overlap"

        perm = select_permutation(
            self.reorder_mode,
            adjacency=adjacency,
            coords_all=np.asarray(params[self.coordinate_key], dtype=np.float64),
            freedofs=np.asarray(params["freedofs"], dtype=np.int64),
            n_parts=self.size,
            block_size=int(self.block_size),
        )
        self.layout = build_global_layout(
            params,
            adjacency,
            perm,
            comm,
            block_size=int(self.block_size),
            dirichlet_key=self.dirichlet_key,
        )
        self.part = SimpleNamespace(
            perm=self.layout.perm,
            iperm=self.layout.iperm,
            lo=self.layout.lo,
            hi=self.layout.hi,
            n_free=self.layout.n_free,
            n_owned=self.layout.hi - self.layout.lo,
        )
        self.local_data = build_local_overlap_data(
            params,
            self.layout,
            comm,
            elem_data_keys=self.local_elem_data_keys,
        )
        (
            self._energy_jit,
            self._grad_jit,
            self._elem_hess_jit,
            self._local_grad_raw,
        ) = self._make_local_element_kernels()
        self._scatter = self._build_scatter_data()
        self._f_owned = np.asarray(self._build_rhs_owned(), dtype=np.float64)

        if self.local_hessian_mode == "sfd_local":
            self._setup_local_sfd()
            self._hvp_eval_mode = "sfd_local_batched"
        elif self.local_hessian_mode == "sfd_local_vmap":
            self._setup_local_sfd()
            self._hvp_eval_mode = "sfd_local_vmap_hvpjit"
        elif self.local_hessian_mode != "element":
            raise ValueError(
                f"Unsupported local_hessian_mode={self.local_hessian_mode!r}"
            )

        self._gather_sizes = np.asarray(
            comm.allgather(self.layout.hi - self.layout.lo), dtype=np.int64
        )
        self._gather_displs = np.zeros_like(self._gather_sizes)
        if len(self._gather_displs) > 1:
            self._gather_displs[1:] = np.cumsum(self._gather_sizes[:-1])

        self.dirichlet_full = np.asarray(params[self.dirichlet_key], dtype=np.float64)

        self.A = PETSc.Mat().create(comm=comm)
        self.A.setType(PETSc.Mat.Type.MPIAIJ)
        self.A.setSizes(
            ((self.layout.hi - self.layout.lo, self.layout.n_free),) * 2
        )
        self.A.setPreallocationCOO(
            self.layout.owned_rows.astype(PETSc.IntType),
            self.layout.owned_cols.astype(PETSc.IntType),
        )
        if int(self.block_size) > 1:
            self.A.setBlockSize(int(self.block_size))
        self._nullspace = None
        if self.use_near_nullspace and self.near_nullspace_key is not None:
            self._nullspace = build_near_nullspace(
                self.layout,
                params,
                comm,
                kernel_key=self.near_nullspace_key,
            )
            self.A.setNearNullSpace(self._nullspace)

        self.ksp = PETSc.KSP().create(comm)
        self.ksp.setType(ksp_type)
        self.ksp.getPC().setType(pc_type)
        if pc_options:
            opts = PETSc.Options()
            for key, value in pc_options.items():
                opts[key] = value
        self.ksp.setTolerances(rtol=float(ksp_rtol), max_it=int(ksp_max_it))
        self.ksp.setFromOptions()

        self._warmup()

    def _make_local_element_kernels(self):
        raise NotImplementedError

    def _build_rhs_owned(self) -> np.ndarray:
        return np.zeros(self.layout.hi - self.layout.lo, dtype=np.float64)

    def _warmup(self):
        v_local = np.asarray(
            self.dirichlet_full[self.local_data.local_total_nodes], dtype=np.float64
        )
        self._energy_jit(jnp.asarray(v_local)).block_until_ready()
        self._grad_jit(jnp.asarray(v_local)).block_until_ready()
        self._elem_hess_jit(jnp.asarray(v_local)).block_until_ready()

    def _setup_local_sfd(self):
        import igraph

        local_reord = self.layout.total_to_free_reord[self.local_data.local_total_nodes]
        rows = self.layout.coo_rows
        cols = self.layout.coo_cols
        mask = np.isin(rows, local_reord) & np.isin(cols, local_reord)
        row_arr = rows[mask]
        col_arr = cols[mask]
        valid = (row_arr >= 0) & (col_arr >= 0)
        row_arr = row_arr[valid]
        col_arr = col_arr[valid]

        J_arr = np.unique(local_reord[local_reord >= 0]).astype(np.int64)
        n_J = len(J_arr)
        J_to_idx = np.full(self.layout.n_free, -1, dtype=np.int64)
        J_to_idx[J_arr] = np.arange(n_J, dtype=np.int64)

        A_J = sparse.csr_matrix(
            (
                np.ones(len(row_arr), dtype=np.float64),
                (J_to_idx[row_arr], J_to_idx[col_arr]),
            ),
            shape=(n_J, n_J),
        )
        A_J.data[:] = 1.0
        A_J.eliminate_zeros()

        A2_J = sparse.csr_matrix(A_J @ A_J)
        A2_J.data[:] = 1.0
        A2_J.eliminate_zeros()

        A2_J_coo = A2_J.tocoo()
        lo_tri = A2_J_coo.row > A2_J_coo.col
        edges = np.column_stack((A2_J_coo.row[lo_tri], A2_J_coo.col[lo_tri]))
        graph = igraph.Graph(
            n_J, edges.tolist() if len(edges) > 0 else [], directed=False
        )
        coloring_raw = graph.vertex_coloring_greedy()
        self._sfd_local_coloring = np.array(coloring_raw, dtype=np.int32).ravel()
        self._sfd_n_colors = (
            int(self._sfd_local_coloring.max() + 1) if n_J > 0 else 0
        )
        self._sfd_J_dofs = J_arr
        self._sfd_J_to_idx = J_to_idx

        reord_to_local = np.full(self.layout.n_free, -1, dtype=np.int64)
        free_mask = local_reord >= 0
        reord_to_local[local_reord[free_mask]] = np.nonzero(free_mask)[0]

        owned_local_rows = reord_to_local[self.layout.owned_rows]
        if np.any(owned_local_rows < 0):
            raise RuntimeError("Owned reordered rows are missing from the overlap domain")
        owned_col_J_idx = J_to_idx[self.layout.owned_cols]
        if np.any(owned_col_J_idx < 0):
            raise RuntimeError("Owned reordered columns are missing from the local SFD set")
        owned_col_colors = self._sfd_local_coloring[owned_col_J_idx]

        self._sfd_color_nz = {}
        for c in range(self._sfd_n_colors):
            mask_c = owned_col_colors == c
            positions = np.where(mask_c)[0].astype(np.int64)
            local_rows = owned_local_rows[positions].astype(np.int64)
            self._sfd_color_nz[c] = (positions, local_rows)

        indicators_local = []
        for c in range(self._sfd_n_colors):
            indicator = np.zeros(len(local_reord), dtype=np.float64)
            J_dofs_c = self._sfd_J_dofs[self._sfd_local_coloring == c]
            local_idx = reord_to_local[J_dofs_c]
            indicator[local_idx] = 1.0
            indicators_local.append(jnp.array(indicator))
        self._sfd_indicators_local = indicators_local
        self._sfd_indicators_stacked = (
            jnp.stack(indicators_local)
            if len(indicators_local) > 0
            else jnp.zeros((0, len(local_reord)), dtype=jnp.float64)
        )

        def hvp_fn(v_local, tangent):
            return jax.jvp(self._local_grad_raw, (v_local,), (tangent,))[1]

        self._sfd_hvp_jit = jax.jit(hvp_fn)

        def hvp_batched(v_local, tangents):
            return jax.vmap(lambda t: hvp_fn(v_local, t))(tangents)

        self._sfd_hvp_batched_jit = jax.jit(hvp_batched)

        def hvp_vmap(v_local, tangents):
            return jax.vmap(lambda t: self._sfd_hvp_jit(v_local, t))(tangents)

        self._sfd_hvp_vmap = hvp_vmap

    def _build_scatter_data(self) -> ScatterData:
        elems_reordered = self.local_data.elems_reordered
        local_reord = self.layout.total_to_free_reord[self.local_data.local_total_nodes]
        owned_mask_local = (local_reord >= self.layout.lo) & (local_reord < self.layout.hi)
        owned_rows = local_reord[owned_mask_local] - self.layout.lo
        owned_local_pos = np.full(self.layout.hi - self.layout.lo, -1, dtype=np.int64)
        owned_local_pos[owned_rows] = np.where(owned_mask_local)[0].astype(np.int64)
        if np.any(owned_local_pos < 0):
            raise RuntimeError(
                "Failed to map all owned reordered DOFs to overlap-local indices"
            )

        rows = elems_reordered[:, :, None]
        cols = elems_reordered[:, None, :]
        valid = (rows >= self.layout.lo) & (rows < self.layout.hi) & (cols >= 0)
        vi = np.where(valid)
        row_vals = elems_reordered[vi[0], vi[1]]
        col_vals = elems_reordered[vi[0], vi[2]]
        keys = row_vals.astype(np.int64) * np.int64(self.layout.n_free) + col_vals.astype(
            np.int64
        )
        positions = np.fromiter(
            (self.layout.owned_key_to_pos[int(k)] for k in keys),
            dtype=np.int64,
            count=len(keys),
        )
        return ScatterData(
            owned_local_pos=owned_local_pos,
            hess_e=np.asarray(vi[0], dtype=np.int64),
            hess_i=np.asarray(vi[1], dtype=np.int64),
            hess_j=np.asarray(vi[2], dtype=np.int64),
            hess_positions=positions,
        )

    def _allgather_full_owned(self, owned_values: np.ndarray) -> tuple[np.ndarray, float]:
        full = np.empty(self.layout.n_free, dtype=np.float64)
        t0 = time.perf_counter()
        self.comm.Allgatherv(
            np.asarray(owned_values, dtype=np.float64),
            [full, self._gather_sizes, self._gather_displs, MPI.DOUBLE],
        )
        return full, time.perf_counter() - t0

    def _build_v_local(self, full_reordered: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        v_local = local_vec_from_full(
            full_reordered,
            self.layout.total_to_free_reord,
            self.local_data.local_total_nodes,
            self.dirichlet_full,
        )
        return v_local, time.perf_counter() - t0

    def update_dirichlet(self, u_0_new):
        self.dirichlet_full = np.asarray(u_0_new, dtype=np.float64)

    def create_vec(self, full_array_reordered=None):
        vec = PETSc.Vec().createMPI(
            (self.layout.hi - self.layout.lo, self.layout.n_free),
            comm=self.comm,
        )
        if full_array_reordered is not None:
            arr = np.asarray(full_array_reordered, dtype=np.float64)
            vec.array[:] = arr[self.layout.lo : self.layout.hi]
            vec.assemble()
        return vec

    def energy_fn(self, vec):
        full, _ = self._allgather_full_owned(np.asarray(vec.array[:], dtype=np.float64))
        v_local, _ = self._build_v_local(full)
        val_local = float(self._energy_jit(jnp.asarray(v_local)).block_until_ready())
        energy = float(self.comm.allreduce(val_local, op=MPI.SUM))
        if self._f_owned.size == 0:
            return energy
        load = float(self.comm.allreduce(np.dot(self._f_owned, vec.array[:]), op=MPI.SUM))
        return energy - load

    def gradient_fn(self, vec, g):
        full, _ = self._allgather_full_owned(np.asarray(vec.array[:], dtype=np.float64))
        v_local, _ = self._build_v_local(full)
        grad_local = np.asarray(self._grad_jit(jnp.asarray(v_local)).block_until_ready())
        grad_owned = grad_local[self._scatter.owned_local_pos]
        if self._f_owned.size:
            grad_owned = grad_owned - self._f_owned
        g.array[:] = grad_owned

    def assemble_hessian(self, u_owned, variant=2):
        del variant
        if self.local_hessian_mode == "sfd_local":
            return self._assemble_hessian_sfd_local(u_owned)
        if self.local_hessian_mode == "sfd_local_vmap":
            return self._assemble_hessian_sfd_local_vmap(u_owned)
        return self.assemble_hessian_element(u_owned)

    def _finalize_sfd_local_hessian(self, all_hvps_np, t_comm, t_build, t_total):
        timings = {}

        t0 = time.perf_counter()
        owned_vals = np.zeros(self.layout.owned_rows.size, dtype=np.float64)
        for c in range(self._sfd_n_colors):
            positions, local_rows = self._sfd_color_nz[c]
            if len(positions) > 0:
                owned_vals[positions] = all_hvps_np[c, local_rows]
        timings["extraction"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.A.setValuesCOO(
            owned_vals.astype(PETSc.ScalarType),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        self.A.assemble()
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["allgatherv"] = float(t_comm)
        timings["build_v_local"] = float(t_build)
        timings["p2p_exchange"] = float(t_comm + t_build)
        timings["n_hvps"] = int(self._sfd_n_colors)
        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        return timings

    def _assemble_hessian_sfd_local(self, u_owned):
        t_total = time.perf_counter()

        full, t_comm = self._allgather_full_owned(np.asarray(u_owned, dtype=np.float64))
        v_local, t_build = self._build_v_local(full)

        timings = {}
        if self._sfd_n_colors > 0:
            t0 = time.perf_counter()
            all_hvps = self._sfd_hvp_batched_jit(
                jnp.asarray(v_local), self._sfd_indicators_stacked
            ).block_until_ready()
            all_hvps_np = np.asarray(all_hvps)
            timings["hvp_compute"] = time.perf_counter() - t0
        else:
            all_hvps_np = np.zeros((0, len(v_local)), dtype=np.float64)
            timings["hvp_compute"] = 0.0

        timings["assembly_mode"] = "sfd_overlap_local"
        finalize = self._finalize_sfd_local_hessian(
            all_hvps_np, t_comm, t_build, t_total
        )
        finalize.update(timings)
        return finalize

    def _assemble_hessian_sfd_local_vmap(self, u_owned):
        t_total = time.perf_counter()

        full, t_comm = self._allgather_full_owned(np.asarray(u_owned, dtype=np.float64))
        v_local, t_build = self._build_v_local(full)

        timings = {}
        if self._sfd_n_colors > 0:
            t0 = time.perf_counter()
            all_hvps = self._sfd_hvp_vmap(
                jnp.asarray(v_local), self._sfd_indicators_stacked
            )
            all_hvps_np = np.asarray(all_hvps.block_until_ready())
            timings["hvp_compute"] = time.perf_counter() - t0
        else:
            all_hvps_np = np.zeros((0, len(v_local)), dtype=np.float64)
            timings["hvp_compute"] = 0.0

        timings["assembly_mode"] = "sfd_overlap_local_vmap_hvpjit"
        finalize = self._finalize_sfd_local_hessian(
            all_hvps_np, t_comm, t_build, t_total
        )
        finalize.update(timings)
        return finalize

    def assemble_hessian_element(self, u_owned):
        timings = {}
        t_total = time.perf_counter()

        full, t_comm = self._allgather_full_owned(np.asarray(u_owned, dtype=np.float64))
        v_local, t_build = self._build_v_local(full)

        t0 = time.perf_counter()
        elem_hess = np.asarray(self._elem_hess_jit(jnp.asarray(v_local)).block_until_ready())
        timings["elem_hessian_compute"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        contrib = elem_hess[self._scatter.hess_e, self._scatter.hess_i, self._scatter.hess_j]
        owned_vals = np.zeros(self.layout.owned_rows.size, dtype=np.float64)
        np.add.at(owned_vals, self._scatter.hess_positions, contrib)
        timings["scatter"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.A.setValuesCOO(
            owned_vals.astype(PETSc.ScalarType),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        self.A.assemble()
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["allgatherv"] = float(t_comm)
        timings["build_v_local"] = float(t_build)
        timings["p2p_exchange"] = float(t_comm + t_build)
        timings["hvp_compute"] = float(timings["elem_hessian_compute"])
        timings["extraction"] = float(timings["scatter"])
        timings["n_hvps"] = 0
        timings["assembly_mode"] = "element_overlap"
        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        return timings

    def cleanup(self):
        self.ksp.destroy()
        self.A.destroy()
        if self._nullspace is not None:
            self._nullspace.destroy()
