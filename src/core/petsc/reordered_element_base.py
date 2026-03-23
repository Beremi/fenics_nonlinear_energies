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
    vec_e: np.ndarray
    vec_i: np.ndarray
    vec_positions: np.ndarray
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


def _validate_permutation(perm: np.ndarray, n_free: int) -> np.ndarray:
    perm = np.asarray(perm, dtype=np.int64).ravel()
    if perm.size != int(n_free):
        raise ValueError(
            f"Permutation size {perm.size} does not match number of free DOFs {n_free}"
        )
    if perm.size and (
        int(np.min(perm)) != 0
        or int(np.max(perm)) != int(n_free) - 1
        or np.unique(perm).size != perm.size
    ):
        raise ValueError("Permutation override must contain each free DOF exactly once")
    return perm


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


def _owned_pattern_from_local_elems(
    local_elems_reordered: np.ndarray,
    *,
    lo: int,
    hi: int,
    n_free: int,
    chunk_elems: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    key_chunks: list[np.ndarray] = []
    elems_arr = np.asarray(local_elems_reordered, dtype=np.int64)
    for start in range(0, int(elems_arr.shape[0]), int(chunk_elems)):
        block = elems_arr[start : start + int(chunk_elems)]
        rows = block[:, :, None]
        cols = block[:, None, :]
        valid = (rows >= int(lo)) & (rows < int(hi)) & (cols >= 0)
        if not np.any(valid):
            continue
        row_vals = np.broadcast_to(rows, valid.shape)[valid].astype(np.int64, copy=False)
        col_vals = np.broadcast_to(cols, valid.shape)[valid].astype(np.int64, copy=False)
        keys = np.unique(row_vals * np.int64(n_free) + col_vals)
        if keys.size:
            key_chunks.append(np.asarray(keys, dtype=np.int64))
    if not key_chunks:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    all_keys = np.unique(np.concatenate(key_chunks).astype(np.int64, copy=False))
    return (
        np.asarray(all_keys // np.int64(n_free), dtype=np.int64),
        np.asarray(all_keys % np.int64(n_free), dtype=np.int64),
    )


def build_global_layout(
    params: dict,
    adjacency: sparse.spmatrix | None,
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
    key_base = np.int64(n_free)
    if adjacency is not None:
        row_adj, col_adj = adjacency.tocsr().nonzero()
        coo_rows = iperm[np.asarray(row_adj, dtype=np.int64)]
        coo_cols = iperm[np.asarray(col_adj, dtype=np.int64)]
        owned_mask = (coo_rows >= lo) & (coo_rows < hi)
        owned_rows = np.asarray(coo_rows[owned_mask], dtype=np.int64)
        owned_cols = np.asarray(coo_cols[owned_mask], dtype=np.int64)

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
    elif "_distributed_local_elem_idx" in params:
        local_elem_idx = np.asarray(params["_distributed_local_elem_idx"], dtype=np.int64)
        local_elems_total = np.asarray(elems[local_elem_idx], dtype=np.int64)
        local_elems_reordered = np.asarray(
            total_to_free_reord[local_elems_total], dtype=np.int64
        )
        owned_rows, owned_cols = _owned_pattern_from_local_elems(
            local_elems_reordered,
            lo=int(lo),
            hi=int(hi),
            n_free=int(n_free),
        )
        coo_rows = np.asarray(owned_rows, dtype=np.int64)
        coo_cols = np.asarray(owned_cols, dtype=np.int64)
        owned_mask = np.ones(len(owned_rows), dtype=bool)
        owned_keys = (
            np.asarray(owned_rows, dtype=np.int64) * key_base
            + np.asarray(owned_cols, dtype=np.int64)
        )
        global_key_to_pos = {int(k): i for i, k in enumerate(owned_keys.tolist())}
        owned_key_to_pos = dict(global_key_to_pos)
        elem_owner = np.zeros(0, dtype=np.int64)
    else:
        raise ValueError(
            "A global adjacency matrix is required unless distributed local element "
            "data is provided via '_distributed_local_elem_idx'"
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
    block_size: int,
) -> LocalOverlapData:
    elems = np.asarray(params["elems"], dtype=np.int64)
    if "_distributed_local_elem_idx" in params:
        local_elem_idx = np.asarray(params["_distributed_local_elem_idx"], dtype=np.int64)
        local_elems_total = np.asarray(elems[local_elem_idx], dtype=np.int64)
        elem_reordered_local = np.asarray(
            layout.total_to_free_reord[local_elems_total], dtype=np.int64
        )
        masked = np.where(
            elem_reordered_local >= 0,
            elem_reordered_local,
            np.int64(layout.n_free),
        )
        elem_min = np.min(masked, axis=1)
        valid = elem_min < int(layout.n_free)
        local_elem_owner = np.full(len(local_elem_idx), -1, dtype=np.int64)
        if np.any(valid):
            local_elem_owner[valid] = _rank_of_dof_vec(
                elem_min[valid],
                int(layout.n_free),
                int(comm.size),
                block_size=int(block_size),
            )
        local_energy_weights = (local_elem_owner == int(comm.rank)).astype(np.float64)
        local_elem_data = {}
        for key in elem_data_keys:
            local_key = f"_distributed_{key}"
            if local_key in params:
                local_elem_data[key] = np.asarray(params[local_key], dtype=np.float64)
            else:
                local_elem_data[key] = np.asarray(params[key], dtype=np.float64)[local_elem_idx]
    else:
        elem_reordered = layout.total_to_free_reord[elems]
        local_mask = np.any(
            (elem_reordered >= layout.lo) & (elem_reordered < layout.hi), axis=1
        )
        local_elem_idx = np.where(local_mask)[0].astype(np.int64)
        local_energy_weights = (layout.elem_owner[local_elem_idx] == comm.rank).astype(
            np.float64
        )
        local_elems_total = elems[local_elem_idx]
        local_elem_data = {
            key: np.asarray(params[key], dtype=np.float64)[local_elem_idx]
            for key in elem_data_keys
        }

    local_total_nodes, inverse = np.unique(
        local_elems_total.ravel(), return_inverse=True
    )
    elems_local_np = inverse.reshape(local_elems_total.shape).astype(np.int32)

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
    if kernel_key in params:
        kernel = np.asarray(params[kernel_key], dtype=np.float64)
    elif "nodes" in params and "freedofs" in params:
        nodes = np.asarray(params["nodes"], dtype=np.float64)
        freedofs = np.asarray(params["freedofs"], dtype=np.int64)
        owned_orig_free = np.asarray(layout.perm[layout.lo : layout.hi], dtype=np.int64)
        owned_total_dofs = np.asarray(freedofs[owned_orig_free], dtype=np.int64)
        comps = owned_total_dofs % 2
        node_ids = owned_total_dofs // 2
        center = np.mean(nodes, axis=0)
        x = np.asarray(nodes[node_ids, 0], dtype=np.float64) - float(center[0])
        y = np.asarray(nodes[node_ids, 1], dtype=np.float64) - float(center[1])
        kernel = np.zeros((layout.hi - layout.lo, 3), dtype=np.float64)
        kernel[comps == 0, 0] = 1.0
        kernel[comps == 1, 1] = 1.0
        kernel[comps == 0, 2] = -y[comps == 0]
        kernel[comps == 1, 2] = x[comps == 1]
    else:
        raise KeyError(
            f"Missing near-nullspace source {kernel_key!r} and could not derive "
            "rigid modes from nodes/freedofs"
        )
    vecs = []
    for i in range(kernel.shape[1]):
        vec = PETSc.Vec().createMPI((layout.hi - layout.lo, layout.n_free), comm=comm)
        if kernel.shape[0] == int(layout.hi - layout.lo):
            vec.array[:] = np.asarray(kernel[:, i], dtype=np.float64)
        else:
            mode = kernel[:, i][layout.perm]
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
        perm_override=None,
        distribution_strategy=None,
        reuse_hessian_value_buffers=True,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.params = params
        self.reorder_mode = str(reorder_mode)
        self.local_hessian_mode = str(local_hessian_mode)
        self.use_near_nullspace = bool(use_near_nullspace)
        self.reuse_hessian_value_buffers = bool(reuse_hessian_value_buffers)
        self.distribution_strategy = str(
            distribution_strategy or getattr(self, "distribution_strategy", "overlap_allgather")
        )
        self.iter_timings = []
        self._hvp_eval_mode = "element_overlap"
        self._setup_timings: dict[str, float] = {}
        self._callback_stats = {
            "energy": {
                "calls": 0,
                "allgatherv": 0.0,
                "ghost_exchange": 0.0,
                "build_v_local": 0.0,
                "kernel": 0.0,
                "allreduce": 0.0,
                "load": 0.0,
                "total": 0.0,
            },
            "gradient": {
                "calls": 0,
                "allgatherv": 0.0,
                "ghost_exchange": 0.0,
                "build_v_local": 0.0,
                "kernel": 0.0,
                "total": 0.0,
            },
            "hessian": {
                "calls": 0,
                "allgatherv": 0.0,
                "ghost_exchange": 0.0,
                "build_v_local": 0.0,
                "hvp_compute": 0.0,
                "extraction": 0.0,
                "coo_assembly": 0.0,
                "total": 0.0,
            },
        }

        t_setup_total = time.perf_counter()

        freedofs = np.asarray(params["freedofs"], dtype=np.int64)
        t0 = time.perf_counter()
        if perm_override is None:
            if adjacency is None and self.reorder_mode not in {"none", "block_xyz"}:
                raise ValueError(
                    "Distributed local element mode currently supports only "
                    "reorder modes 'none' and 'block_xyz' without a global adjacency"
                )
            perm = select_permutation(
                self.reorder_mode,
                adjacency=adjacency,
                coords_all=np.asarray(params[self.coordinate_key], dtype=np.float64),
                freedofs=freedofs,
                n_parts=self.size,
                block_size=int(self.block_size),
            )
        else:
            perm = _validate_permutation(perm_override, len(freedofs))
        self._setup_timings["permutation"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.layout = build_global_layout(
            params,
            adjacency,
            perm,
            comm,
            block_size=int(self.block_size),
            dirichlet_key=self.dirichlet_key,
        )
        self._setup_timings["global_layout"] = time.perf_counter() - t0
        self.part = SimpleNamespace(
            perm=self.layout.perm,
            iperm=self.layout.iperm,
            lo=self.layout.lo,
            hi=self.layout.hi,
            n_free=self.layout.n_free,
            n_owned=self.layout.hi - self.layout.lo,
        )
        t0 = time.perf_counter()
        self.local_data = build_local_overlap_data(
            params,
            self.layout,
            comm,
            elem_data_keys=self.local_elem_data_keys,
            block_size=int(self.block_size),
        )
        self._setup_timings["local_overlap"] = time.perf_counter() - t0
        self.dirichlet_full = np.asarray(params[self.dirichlet_key], dtype=np.float64)

        t0 = time.perf_counter()
        self._setup_distribution_exchange()
        self._setup_timings["distribution_setup"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        (
            self._energy_jit,
            self._grad_jit,
            self._elem_hess_jit,
            self._local_grad_raw,
        ) = self._make_local_element_kernels()
        self._setup_timings["kernel_build"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._scatter = self._build_scatter_data()
        self._setup_timings["scatter_build"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._f_owned = np.asarray(self._build_rhs_owned(), dtype=np.float64)
        self._setup_timings["rhs_build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
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
        self._setup_timings["local_hessian_setup"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._gather_sizes = np.asarray(
            comm.allgather(self.layout.hi - self.layout.lo), dtype=np.int64
        )
        self._gather_displs = np.zeros_like(self._gather_sizes)
        if len(self._gather_displs) > 1:
            self._gather_displs[1:] = np.cumsum(self._gather_sizes[:-1])
        self._setup_timings["allgather_plan"] = time.perf_counter() - t0

        t0 = time.perf_counter()
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
        self._setup_timings["matrix_create"] = time.perf_counter() - t0
        owned_nnz = int(self.layout.owned_rows.size)
        self._owned_hessian_values = np.zeros(owned_nnz, dtype=np.float64)
        if np.dtype(PETSc.ScalarType) == np.dtype(np.float64):
            self._owned_hessian_values_petsc = self._owned_hessian_values
        else:
            self._owned_hessian_values_petsc = np.zeros(
                owned_nnz, dtype=PETSc.ScalarType
            )
        self._nullspace = None
        t0 = time.perf_counter()
        if self.use_near_nullspace and self.near_nullspace_key is not None:
            self._nullspace = build_near_nullspace(
                self.layout,
                params,
                comm,
                kernel_key=self.near_nullspace_key,
            )
            self.A.setNearNullSpace(self._nullspace)
        self._setup_timings["nullspace_build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.ksp = PETSc.KSP().create(comm)
        self.ksp.setType(ksp_type)
        self.ksp.getPC().setType(pc_type)
        if pc_options:
            opts = PETSc.Options()
            for key, value in pc_options.items():
                opts[key] = value
        self.ksp.setTolerances(rtol=float(ksp_rtol), max_it=int(ksp_max_it))
        self.ksp.setFromOptions()
        self._setup_timings["ksp_create"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._warmup()
        self._setup_timings["warmup"] = time.perf_counter() - t0
        self._setup_timings["total"] = time.perf_counter() - t_setup_total

    def _make_local_element_kernels(self):
        raise NotImplementedError

    def _build_rhs_owned(self) -> np.ndarray:
        return np.zeros(self.layout.hi - self.layout.lo, dtype=np.float64)

    def setup_summary(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in self._setup_timings.items()}

    def _reset_owned_hessian_values(self) -> np.ndarray:
        if not self.reuse_hessian_value_buffers:
            return np.zeros(int(self.layout.owned_rows.size), dtype=np.float64)
        self._owned_hessian_values.fill(0.0)
        return self._owned_hessian_values

    def _owned_hessian_values_for_petsc(self, owned_values: np.ndarray | None = None) -> np.ndarray:
        values = self._owned_hessian_values if owned_values is None else np.asarray(
            owned_values, dtype=np.float64
        )
        if not self.reuse_hessian_value_buffers:
            return np.asarray(values, dtype=PETSc.ScalarType)
        if self._owned_hessian_values_petsc is self._owned_hessian_values:
            return self._owned_hessian_values_petsc
        np.copyto(
            self._owned_hessian_values_petsc,
            values,
            casting="unsafe",
        )
        return self._owned_hessian_values_petsc

    def callback_summary(self) -> dict[str, dict[str, float | int]]:
        summary = {}
        for phase, stats in self._callback_stats.items():
            summary[phase] = {}
            for key, value in stats.items():
                if key == "calls":
                    summary[phase][key] = int(value)
                else:
                    summary[phase][key] = float(value)
        return summary

    def _record_callback(self, phase: str, **timings) -> None:
        stats = self._callback_stats[str(phase)]
        stats["calls"] = int(stats.get("calls", 0)) + 1
        for key, value in timings.items():
            stats[key] = float(stats.get(key, 0.0)) + float(value)

    def _record_hessian_iteration(self, timings: dict[str, object]) -> None:
        self._record_callback(
            "hessian",
            allgatherv=float(timings.get("allgatherv", 0.0)),
            ghost_exchange=float(timings.get("ghost_exchange", 0.0)),
            build_v_local=float(timings.get("build_v_local", 0.0)),
            hvp_compute=float(timings.get("hvp_compute", 0.0)),
            extraction=float(timings.get("extraction", 0.0)),
            coo_assembly=float(timings.get("coo_assembly", 0.0)),
            total=float(timings.get("total", 0.0)),
        )

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

    def _setup_distribution_exchange(self) -> None:
        local_reord = np.asarray(
            self.layout.total_to_free_reord[self.local_data.local_total_nodes],
            dtype=np.int64,
        )
        free_mask = local_reord >= 0
        self._dist_local_reord = local_reord
        self._dist_free_local_indices = np.where(free_mask)[0].astype(np.int64)
        self._dist_free_global_indices = np.asarray(
            local_reord[self._dist_free_local_indices],
            dtype=np.int64,
        )
        self._dist_dirichlet_template = np.zeros(len(local_reord), dtype=np.float64)
        dirichlet_mask = ~free_mask
        if np.any(dirichlet_mask):
            self._dist_dirichlet_template[dirichlet_mask] = self.dirichlet_full[
                self.local_data.local_total_nodes[dirichlet_mask]
            ]
        self._p2p_owned_local = np.zeros(0, dtype=np.int64)
        self._p2p_owned_offset = np.zeros(0, dtype=np.int64)
        self._ghost_recv: dict[int, np.ndarray] = {}
        self._ghost_send_offsets: dict[int, np.ndarray] = {}
        self._ghost_send_bufs: dict[int, np.ndarray] = {}
        self._ghost_recv_bufs: dict[int, np.ndarray] = {}
        if self.distribution_strategy != "overlap_p2p":
            return

        lo, hi = self.layout.lo, self.layout.hi
        free_global = self._dist_free_global_indices
        owned_mask = (free_global >= lo) & (free_global < hi)
        ghost_mask = ~owned_mask
        self._p2p_owned_local = self._dist_free_local_indices[owned_mask]
        self._p2p_owned_offset = (free_global[owned_mask] - lo).astype(np.int64)

        ghost_local = self._dist_free_local_indices[ghost_mask]
        ghost_global = free_global[ghost_mask]
        send_requests: dict[int, np.ndarray] = {}
        if len(ghost_global) > 0:
            ghost_owners = _rank_of_dof_vec(
                ghost_global,
                self.layout.n_free,
                self.size,
                block_size=int(self.block_size),
            )
            for owner in np.unique(ghost_owners):
                owner = int(owner)
                if owner == self.rank:
                    continue
                mask_owner = ghost_owners == owner
                owner_lo, _ = petsc_ownership_range(
                    self.layout.n_free,
                    owner,
                    self.size,
                    block_size=int(self.block_size),
                )
                self._ghost_recv[owner] = ghost_local[mask_owner]
                send_requests[owner] = (ghost_global[mask_owner] - owner_lo).astype(
                    np.int64
                )

        n_we_need = np.zeros(self.size, dtype=np.int64)
        for owner, offsets in send_requests.items():
            n_we_need[int(owner)] = len(offsets)
        n_others_need = np.zeros(self.size, dtype=np.int64)
        self.comm.Alltoall(n_we_need, n_others_need)

        recv_reqs = []
        for owner in range(self.size):
            if owner == self.rank or int(n_others_need[owner]) == 0:
                continue
            buf = np.empty(int(n_others_need[owner]), dtype=np.int64)
            recv_reqs.append((self.comm.Irecv(buf, source=owner, tag=4200), owner, buf))

        send_reqs = []
        for owner, offsets in send_requests.items():
            send_reqs.append(
                self.comm.Isend(np.ascontiguousarray(offsets), dest=int(owner), tag=4200)
            )

        for req, owner, buf in recv_reqs:
            req.Wait()
            self._ghost_send_offsets[int(owner)] = buf
        for req in send_reqs:
            req.Wait()

        self._ghost_send_bufs = {
            int(owner): np.empty(len(offsets), dtype=np.float64)
            for owner, offsets in self._ghost_send_offsets.items()
        }
        self._ghost_recv_bufs = {
            int(owner): np.empty(len(local_idx), dtype=np.float64)
            for owner, local_idx in self._ghost_recv.items()
        }

    def _p2p_fill_local(
        self,
        owned_values: np.ndarray,
        *,
        zero_dirichlet: bool,
        tag: int,
    ) -> np.ndarray:
        owned_values = np.ascontiguousarray(owned_values, dtype=np.float64)
        if zero_dirichlet:
            v_local = np.zeros(len(self._dist_local_reord), dtype=np.float64)
        else:
            v_local = self._dist_dirichlet_template.copy()
        if len(self._p2p_owned_local) > 0:
            v_local[self._p2p_owned_local] = owned_values[self._p2p_owned_offset]
        if not self._ghost_recv and not self._ghost_send_offsets:
            return v_local

        recv_reqs = []
        for owner, buf in self._ghost_recv_bufs.items():
            recv_reqs.append((self.comm.Irecv(buf, source=owner, tag=tag), owner))

        send_reqs = []
        for owner, offsets in self._ghost_send_offsets.items():
            self._ghost_send_bufs[owner][:] = owned_values[offsets]
            send_reqs.append(self.comm.Isend(self._ghost_send_bufs[owner], dest=owner, tag=tag))

        for req, owner in recv_reqs:
            req.Wait()
            v_local[self._ghost_recv[owner]] = self._ghost_recv_bufs[owner]
        for req in send_reqs:
            req.Wait()
        return v_local

    def _owned_to_local(
        self,
        owned_values: np.ndarray,
        *,
        zero_dirichlet: bool = False,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self.distribution_strategy == "overlap_p2p":
            t0 = time.perf_counter()
            v_local = self._p2p_fill_local(
                owned_values,
                zero_dirichlet=bool(zero_dirichlet),
                tag=4202 if zero_dirichlet else 4201,
            )
            t_exchange = time.perf_counter() - t0
            return v_local, {
                "allgatherv": 0.0,
                "ghost_exchange": float(t_exchange),
                "build_v_local": 0.0,
                "exchange_total": float(t_exchange),
            }
        full_reordered, t_comm = self._allgather_full_owned(
            np.asarray(owned_values, dtype=np.float64)
        )
        v_local, t_build = self._build_v_local(
            full_reordered,
            zero_dirichlet=bool(zero_dirichlet),
        )
        return v_local, {
            "allgatherv": float(t_comm),
            "ghost_exchange": 0.0,
            "build_v_local": float(t_build),
            "exchange_total": float(t_comm + t_build),
        }

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

        vec_valid = (elems_reordered >= self.layout.lo) & (elems_reordered < self.layout.hi)
        vec_vi = np.where(vec_valid)
        vec_positions = elems_reordered[vec_vi] - self.layout.lo

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
            vec_e=np.asarray(vec_vi[0], dtype=np.int64),
            vec_i=np.asarray(vec_vi[1], dtype=np.int64),
            vec_positions=np.asarray(vec_positions, dtype=np.int64),
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

    def _build_v_local(
        self,
        full_reordered: np.ndarray,
        *,
        zero_dirichlet: bool = False,
    ) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        v_local = local_vec_from_full(
            full_reordered,
            self.layout.total_to_free_reord,
            self.local_data.local_total_nodes,
            np.zeros_like(self.dirichlet_full) if zero_dirichlet else self.dirichlet_full,
        )
        return v_local, time.perf_counter() - t0

    def update_dirichlet(self, u_0_new):
        self.dirichlet_full = np.asarray(u_0_new, dtype=np.float64)
        dirichlet_mask = self._dist_local_reord < 0
        if np.any(dirichlet_mask):
            self._dist_dirichlet_template[dirichlet_mask] = self.dirichlet_full[
                self.local_data.local_total_nodes[dirichlet_mask]
            ]

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
        t_total = time.perf_counter()
        v_local, exchange = self._owned_to_local(
            np.asarray(vec.array[:], dtype=np.float64),
            zero_dirichlet=False,
        )
        t_kernel0 = time.perf_counter()
        val_local = float(self._energy_jit(jnp.asarray(v_local)).block_until_ready())
        t_kernel = time.perf_counter() - t_kernel0
        t_red0 = time.perf_counter()
        energy = float(self.comm.allreduce(val_local, op=MPI.SUM))
        t_allreduce = time.perf_counter() - t_red0
        t_load = 0.0
        if self._f_owned.size == 0:
            result = energy
        else:
            t_load0 = time.perf_counter()
            load = float(
                self.comm.allreduce(np.dot(self._f_owned, vec.array[:]), op=MPI.SUM)
            )
            t_load = time.perf_counter() - t_load0
            result = energy - load
        self._record_callback(
            "energy",
            allgatherv=float(exchange["allgatherv"]),
            ghost_exchange=float(exchange["ghost_exchange"]),
            build_v_local=float(exchange["build_v_local"]),
            kernel=float(t_kernel),
            allreduce=float(t_allreduce),
            load=float(t_load),
            total=float(time.perf_counter() - t_total),
        )
        return result

    def gradient_fn(self, vec, g):
        t_total = time.perf_counter()
        v_local, exchange = self._owned_to_local(
            np.asarray(vec.array[:], dtype=np.float64),
            zero_dirichlet=False,
        )
        t_kernel0 = time.perf_counter()
        grad_local = np.asarray(self._grad_jit(jnp.asarray(v_local)).block_until_ready())
        t_kernel = time.perf_counter() - t_kernel0
        grad_owned = grad_local[self._scatter.owned_local_pos]
        if self._f_owned.size:
            grad_owned = grad_owned - self._f_owned
        g.array[:] = grad_owned
        self._record_callback(
            "gradient",
            allgatherv=float(exchange["allgatherv"]),
            ghost_exchange=float(exchange["ghost_exchange"]),
            build_v_local=float(exchange["build_v_local"]),
            kernel=float(t_kernel),
            total=float(time.perf_counter() - t_total),
        )

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
        owned_vals = self._reset_owned_hessian_values()
        for c in range(self._sfd_n_colors):
            positions, local_rows = self._sfd_color_nz[c]
            if len(positions) > 0:
                owned_vals[positions] = all_hvps_np[c, local_rows]
        timings["extraction"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.A.setValuesCOO(
            self._owned_hessian_values_for_petsc(owned_vals),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        self.A.assemble()
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["allgatherv"] = float(t_comm)
        timings["ghost_exchange"] = 0.0
        timings["build_v_local"] = float(t_build)
        timings["p2p_exchange"] = float(t_comm + t_build)
        timings["n_hvps"] = int(self._sfd_n_colors)
        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        self._record_hessian_iteration(timings)
        return timings

    def _assemble_hessian_sfd_local(self, u_owned):
        t_total = time.perf_counter()

        v_local, exchange = self._owned_to_local(
            np.asarray(u_owned, dtype=np.float64),
            zero_dirichlet=False,
        )

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
            all_hvps_np,
            float(exchange["allgatherv"] + exchange["ghost_exchange"]),
            float(exchange["build_v_local"]),
            t_total,
        )
        finalize["allgatherv"] = float(exchange["allgatherv"])
        finalize["ghost_exchange"] = float(exchange["ghost_exchange"])
        finalize.update(timings)
        return finalize

    def _assemble_hessian_sfd_local_vmap(self, u_owned):
        t_total = time.perf_counter()

        v_local, exchange = self._owned_to_local(
            np.asarray(u_owned, dtype=np.float64),
            zero_dirichlet=False,
        )

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
            all_hvps_np,
            float(exchange["allgatherv"] + exchange["ghost_exchange"]),
            float(exchange["build_v_local"]),
            t_total,
        )
        finalize["allgatherv"] = float(exchange["allgatherv"])
        finalize["ghost_exchange"] = float(exchange["ghost_exchange"])
        finalize.update(timings)
        return finalize

    def assemble_hessian_element(self, u_owned):
        timings = {}
        t_total = time.perf_counter()

        v_local, exchange = self._owned_to_local(
            np.asarray(u_owned, dtype=np.float64),
            zero_dirichlet=False,
        )

        t0 = time.perf_counter()
        elem_hess = np.asarray(self._elem_hess_jit(jnp.asarray(v_local)).block_until_ready())
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
        timings["assembly_mode"] = "element_overlap"
        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        self._record_hessian_iteration(timings)
        return timings

    def cleanup(self):
        self.ksp.destroy()
        self.A.destroy()
        if self._nullspace is not None:
            self._nullspace.destroy()
