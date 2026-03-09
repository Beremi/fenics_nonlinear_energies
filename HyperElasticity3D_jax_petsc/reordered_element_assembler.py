"""Production HE element assembler using reordered PETSc ownership + overlap domains."""

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
from jax import config

from HyperElasticity3D_jax_petsc.parallel_hessian_dof import _he_energy_density
from tools_petsc4py.dof_partition import _rank_of_dof_vec, petsc_ownership_range


config.update("jax_enable_x64", True)


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
    perm = np.empty(block_perm.size * block_size, dtype=np.int64)
    for comp in range(block_size):
        perm[comp::block_size] = block_perm * block_size + comp
    return perm


def _perm_identity(n_free: int) -> np.ndarray:
    return np.arange(n_free, dtype=np.int64)


def _perm_block_rcm(adjacency: sparse.spmatrix, block_size: int) -> np.ndarray:
    block = _build_block_graph(adjacency, block_size)
    block_perm = np.asarray(
        reverse_cuthill_mckee(block, symmetric_mode=True), dtype=np.int64
    )
    return _expand_block_perm(block_perm, block_size)


def _perm_block_xyz(
    nodes2coord: np.ndarray,
    freedofs: np.ndarray,
    block_size: int,
) -> np.ndarray:
    node_ids = np.asarray(freedofs[::block_size], dtype=np.int64) // block_size
    coords = np.asarray(nodes2coord[node_ids], dtype=np.float64)
    block_perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0])).astype(np.int64)
    return _expand_block_perm(block_perm, block_size)


def _perm_block_metis(
    adjacency: sparse.spmatrix,
    block_size: int,
    n_parts: int,
) -> np.ndarray:
    import pymetis

    block = _build_block_graph(adjacency, block_size)
    _, part = pymetis.part_graph(n_parts, xadj=block.indptr, adjncy=block.indices)
    part = np.asarray(part, dtype=np.int64)
    block_ids = np.arange(block.shape[0], dtype=np.int64)
    block_perm = np.lexsort((block_ids, part)).astype(np.int64)
    return _expand_block_perm(block_perm, block_size)


def _select_perm(
    reorder_mode: str,
    params: dict,
    adjacency: sparse.spmatrix,
    n_parts: int,
    block_size: int,
) -> np.ndarray:
    if reorder_mode == "none":
        return _perm_identity(len(params["freedofs"]))
    if reorder_mode == "block_rcm":
        return _perm_block_rcm(adjacency, block_size)
    if reorder_mode == "block_xyz":
        return _perm_block_xyz(params["nodes2coord"], params["freedofs"], block_size)
    if reorder_mode == "block_metis":
        return _perm_block_metis(adjacency, block_size, n_parts)
    raise ValueError(f"Unsupported element reorder mode: {reorder_mode!r}")


def _inverse_perm(perm: np.ndarray) -> np.ndarray:
    iperm = np.empty_like(perm)
    iperm[perm] = np.arange(perm.size, dtype=np.int64)
    return iperm


def _local_vec_from_full(
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


def _build_global_layout(
    params: dict,
    adjacency: sparse.spmatrix,
    perm: np.ndarray,
    comm: MPI.Comm,
    block_size: int,
) -> GlobalLayout:
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    elems = np.asarray(params["elems"], dtype=np.int64)
    n_total = int(len(np.asarray(params["u_0_ref"], dtype=np.float64)))
    n_free = int(freedofs.size)
    iperm = _inverse_perm(perm)
    lo, hi = petsc_ownership_range(n_free, comm.rank, comm.size, block_size=block_size)

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
            block_size=block_size,
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


def _build_local_overlap_data(
    params: dict,
    layout: GlobalLayout,
    comm: MPI.Comm,
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
    local_total_nodes, inverse = np.unique(local_elems_total.ravel(), return_inverse=True)
    elems_local_np = inverse.reshape(local_elems_total.shape).astype(np.int32)

    local_elem_data = {
        key: np.asarray(params[key], dtype=np.float64)[local_elem_idx]
        for key in ("dphix", "dphiy", "dphiz", "vol")
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


def _build_near_nullspace(layout: GlobalLayout, params: dict, comm: MPI.Comm):
    kernel = np.asarray(params["elastic_kernel"], dtype=np.float64)
    vecs = []
    for i in range(kernel.shape[1]):
        mode = kernel[:, i][layout.perm]
        v = PETSc.Vec().createMPI((layout.hi - layout.lo, layout.n_free), comm=comm)
        v.array[:] = mode[layout.lo:layout.hi]
        v.assemble()
        vecs.append(v)
    return PETSc.NullSpace().create(vectors=vecs)


class HEReorderedElementAssembler:
    """Production HE element assembler using overlap domains and reordered ownership."""

    distribution_strategy = "overlap_allgather"

    def __init__(
        self,
        params,
        comm,
        adjacency,
        ksp_rtol=1e-1,
        ksp_type="gmres",
        pc_type="gamg",
        ksp_max_it=30,
        use_near_nullspace=True,
        pc_options=None,
        reorder_mode="block_xyz",
        use_abs_det=False,
        local_hessian_mode="element",
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.block_size = 3
        self.params = params
        self.use_abs_det = bool(use_abs_det)
        self.use_near_nullspace = bool(use_near_nullspace)
        self.reorder_mode = str(reorder_mode)
        self.local_hessian_mode = str(local_hessian_mode)
        self.iter_timings = []
        self._f_owned = np.zeros(0, dtype=np.float64)
        self._hvp_eval_mode = "element_overlap"

        perm = _select_perm(
            self.reorder_mode,
            params,
            adjacency,
            self.size,
            self.block_size,
        )
        self.layout = _build_global_layout(params, adjacency, perm, comm, self.block_size)
        self.part = SimpleNamespace(
            perm=self.layout.perm,
            iperm=self.layout.iperm,
            lo=self.layout.lo,
            hi=self.layout.hi,
            n_free=self.layout.n_free,
            n_owned=self.layout.hi - self.layout.lo,
        )
        self.local_data = _build_local_overlap_data(params, self.layout, comm)
        (
            self._energy_jit,
            self._grad_jit,
            self._elem_hess_jit,
            self._local_grad_raw,
        ) = self._make_local_element_kernels()
        self._scatter = self._build_scatter_data()
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

        self.dirichlet_full = np.asarray(params["u_0_ref"], dtype=np.float64)

        self.A = PETSc.Mat().create(comm=comm)
        self.A.setType(PETSc.Mat.Type.MPIAIJ)
        self.A.setSizes(
            ((self.layout.hi - self.layout.lo, self.layout.n_free),)
            * 2
        )
        self.A.setPreallocationCOO(
            self.layout.owned_rows.astype(PETSc.IntType),
            self.layout.owned_cols.astype(PETSc.IntType),
        )
        self.A.setBlockSize(self.block_size)
        self._nullspace = None
        if self.use_near_nullspace:
            self._nullspace = _build_near_nullspace(self.layout, params, comm)
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
        elems = jnp.asarray(self.local_data.elems_local_np, dtype=jnp.int32)
        dphix = jnp.asarray(self.local_data.local_elem_data["dphix"], dtype=jnp.float64)
        dphiy = jnp.asarray(self.local_data.local_elem_data["dphiy"], dtype=jnp.float64)
        dphiz = jnp.asarray(self.local_data.local_elem_data["dphiz"], dtype=jnp.float64)
        vol = jnp.asarray(self.local_data.local_elem_data["vol"], dtype=jnp.float64)
        energy_weights = jnp.asarray(self.local_data.energy_weights, dtype=jnp.float64)

        C1 = float(self.params["C1"])
        D1 = float(self.params["D1"])
        use_abs_det = self.use_abs_det

        def element_energy(v_e, dphix_e, dphiy_e, dphiz_e, vol_e):
            return (
                _he_energy_density(v_e, dphix_e, dphiy_e, dphiz_e, C1, D1, use_abs_det)
                * vol_e
            )

        hess_elem = jax.vmap(jax.hessian(element_energy), in_axes=(0, 0, 0, 0, 0))

        def local_full_energy(v_local):
            v_e = v_local[elems]
            e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0, 0))(
                v_e, dphix, dphiy, dphiz, vol
            )
            return jnp.sum(e)

        grad_local = jax.grad(local_full_energy)

        @jax.jit
        def energy_fn(v_local):
            v_e = v_local[elems]
            e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0, 0))(
                v_e, dphix, dphiy, dphiz, vol
            )
            return jnp.sum(e * energy_weights)

        @jax.jit
        def local_grad_fn(v_local):
            return grad_local(v_local)

        @jax.jit
        def elem_hess_fn(v_local):
            v_e = v_local[elems]
            return hess_elem(v_e, dphix, dphiy, dphiz, vol)

        return energy_fn, local_grad_fn, elem_hess_fn, grad_local

    def _warmup(self):
        v_local = np.asarray(
            self.dirichlet_full[self.local_data.local_total_nodes], dtype=np.float64
        )
        v_local_j = jnp.asarray(v_local, dtype=jnp.float64)
        _ = self._energy_jit(v_local_j).block_until_ready()
        _ = self._grad_jit(v_local_j).block_until_ready()
        _ = self._elem_hess_jit(v_local_j).block_until_ready()
        if self.local_hessian_mode == "sfd_local":
            dummy_tangents = jnp.zeros(
                (self._sfd_n_colors, v_local.shape[0]), dtype=jnp.float64
            )
            _ = self._sfd_hvp_batched_jit(v_local_j, dummy_tangents).block_until_ready()
        elif self.local_hessian_mode == "sfd_local_vmap":
            dummy_tangents = jnp.zeros(
                (self._sfd_n_colors, v_local.shape[0]), dtype=jnp.float64
            )
            _ = self._sfd_hvp_vmap(v_local_j, dummy_tangents).block_until_ready()

    def _setup_local_sfd(self):
        import igraph

        elems_local = self.local_data.elems_local_np
        local_reord = self.layout.total_to_free_reord[self.local_data.local_total_nodes]
        elem_fdof = local_reord[elems_local]
        npe = elems_local.shape[1]

        rows_2d = np.repeat(elem_fdof, npe, axis=1)
        cols_2d = np.tile(elem_fdof, (1, npe))
        rows_flat = rows_2d.ravel()
        cols_flat = cols_2d.ravel()
        valid = (rows_flat >= 0) & (cols_flat >= 0)
        row_arr = rows_flat[valid]
        col_arr = cols_flat[valid]

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
        reord_to_local[local_reord[free_mask]] = np.where(free_mask)[0]

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
        if indicators_local:
            self._sfd_indicators_stacked = jnp.stack(indicators_local)
        else:
            self._sfd_indicators_stacked = jnp.zeros(
                (0, len(local_reord)), dtype=jnp.float64
            )

        def hvp_fn(v_local, tangent_local):
            return jax.jvp(self._local_grad_raw, (v_local,), (tangent_local,))[1]

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
        v_local = _local_vec_from_full(
            full_reordered,
            self.layout.total_to_free_reord,
            self.local_data.local_total_nodes,
            self.dirichlet_full,
        )
        return v_local, time.perf_counter() - t0

    def update_dirichlet(self, u_0_new):
        self.dirichlet_full = np.asarray(u_0_new, dtype=np.float64)

    def create_vec(self, full_array_reordered=None):
        v = PETSc.Vec().createMPI(
            (self.layout.hi - self.layout.lo, self.layout.n_free),
            comm=self.comm,
        )
        if full_array_reordered is not None:
            arr = np.asarray(full_array_reordered, dtype=np.float64)
            v.array[:] = arr[self.layout.lo:self.layout.hi]
            v.assemble()
        return v

    def energy_fn(self, vec):
        full, _ = self._allgather_full_owned(np.asarray(vec.array[:], dtype=np.float64))
        v_local, _ = self._build_v_local(full)
        val_local = float(self._energy_jit(jnp.asarray(v_local)).block_until_ready())
        return float(self.comm.allreduce(val_local, op=MPI.SUM))

    def gradient_fn(self, vec, g):
        full, _ = self._allgather_full_owned(np.asarray(vec.array[:], dtype=np.float64))
        v_local, _ = self._build_v_local(full)
        grad_local = np.asarray(self._grad_jit(jnp.asarray(v_local)).block_until_ready())
        grad_owned = grad_local[self._scatter.owned_local_pos]
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
