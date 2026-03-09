"""Production pLaplace element assembler using reordered PETSc ownership."""

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
    owned_rows: np.ndarray
    owned_cols: np.ndarray
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


def _perm_identity(n_free: int) -> np.ndarray:
    return np.arange(n_free, dtype=np.int64)


def _perm_block_rcm(adjacency: sparse.spmatrix) -> np.ndarray:
    return np.asarray(
        reverse_cuthill_mckee(adjacency.tocsr(), symmetric_mode=True),
        dtype=np.int64,
    )


def _perm_block_xyz(nodes: np.ndarray, freedofs: np.ndarray) -> np.ndarray:
    coords = np.asarray(nodes[np.asarray(freedofs, dtype=np.int64)], dtype=np.float64)
    if coords.shape[1] >= 3:
        return np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0])).astype(np.int64)
    return np.lexsort((coords[:, 1], coords[:, 0])).astype(np.int64)


def _perm_block_metis(adjacency: sparse.spmatrix, n_parts: int) -> np.ndarray:
    import pymetis

    adj = adjacency.tocsr()
    _, part = pymetis.part_graph(n_parts, xadj=adj.indptr, adjncy=adj.indices)
    part = np.asarray(part, dtype=np.int64)
    dof_ids = np.arange(adj.shape[0], dtype=np.int64)
    return np.lexsort((dof_ids, part)).astype(np.int64)


def _select_perm(
    reorder_mode: str,
    params: dict,
    adjacency: sparse.spmatrix,
    n_parts: int,
) -> np.ndarray:
    if reorder_mode == "none":
        return _perm_identity(len(params["freedofs"]))
    if reorder_mode == "block_rcm":
        return _perm_block_rcm(adjacency)
    if reorder_mode == "block_xyz":
        return _perm_block_xyz(params["nodes"], params["freedofs"])
    if reorder_mode == "block_metis":
        return _perm_block_metis(adjacency, n_parts)
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
) -> GlobalLayout:
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    elems = np.asarray(params["elems"], dtype=np.int64)
    n_total = int(len(np.asarray(params["u_0"], dtype=np.float64)))
    n_free = int(freedofs.size)
    iperm = _inverse_perm(perm)
    lo, hi = petsc_ownership_range(n_free, comm.rank, comm.size, block_size=1)

    total_to_free_orig = np.full(n_total, -1, dtype=np.int64)
    total_to_free_orig[freedofs] = np.arange(n_free, dtype=np.int64)
    total_to_free_reord = np.full(n_total, -1, dtype=np.int64)
    free_mask = total_to_free_orig >= 0
    total_to_free_reord[free_mask] = iperm[total_to_free_orig[free_mask]]

    row_adj, col_adj = adjacency.tocsr().nonzero()
    coo_rows = iperm[np.asarray(row_adj, dtype=np.int64)]
    coo_cols = iperm[np.asarray(col_adj, dtype=np.int64)]
    owned_mask = (coo_rows >= lo) & (coo_rows < hi)
    owned_rows = np.asarray(coo_rows[owned_mask], dtype=np.int64)
    owned_cols = np.asarray(coo_cols[owned_mask], dtype=np.int64)

    key_base = np.int64(n_free)
    owned_keys = (
        owned_rows.astype(np.int64) * key_base + owned_cols.astype(np.int64)
    )
    owned_key_to_pos = {int(k): i for i, k in enumerate(owned_keys.tolist())}

    elems_reordered = total_to_free_reord[elems]
    masked = np.where(elems_reordered >= 0, elems_reordered, np.int64(n_free))
    elem_min = np.min(masked, axis=1)
    valid = elem_min < n_free
    elem_owner = np.full(len(elems), -1, dtype=np.int64)
    if np.any(valid):
        elem_owner[valid] = _rank_of_dof_vec(
            elem_min[valid], n_free, comm.size, block_size=1
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
        owned_rows=owned_rows,
        owned_cols=owned_cols,
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
        for key in ("dvx", "dvy", "vol")
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


def _plaplace_integrand(v_e, dvx_e, dvy_e, p):
    fx = jnp.sum(v_e * dvx_e, axis=-1)
    fy = jnp.sum(v_e * dvy_e, axis=-1)
    return (1.0 / p) * (fx**2 + fy**2) ** (p / 2.0)


class PLaplaceReorderedElementAssembler:
    """Production pLaplace assembler using overlap domains and reordered ownership."""

    distribution_strategy = "overlap_allgather"

    def __init__(
        self,
        params,
        comm,
        adjacency,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="gamg",
        ksp_max_it=10000,
        pc_options=None,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.params = params
        self.reorder_mode = str(reorder_mode)
        self.local_hessian_mode = str(local_hessian_mode)
        self.iter_timings = []
        self._hvp_eval_mode = "element_overlap"

        perm = _select_perm(self.reorder_mode, params, adjacency, self.size)
        self.layout = _build_global_layout(params, adjacency, perm, comm)
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
        self._f_owned = self._build_rhs_owned()

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

        self.dirichlet_full = np.asarray(params["u_0"], dtype=np.float64)

        self.A = PETSc.Mat().create(comm=comm)
        self.A.setType(PETSc.Mat.Type.MPIAIJ)
        self.A.setSizes(
            ((self.layout.hi - self.layout.lo, self.layout.n_free),) * 2
        )
        self.A.setPreallocationCOO(
            self.layout.owned_rows.astype(PETSc.IntType),
            self.layout.owned_cols.astype(PETSc.IntType),
        )

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
        dvx = jnp.asarray(self.local_data.local_elem_data["dvx"], dtype=jnp.float64)
        dvy = jnp.asarray(self.local_data.local_elem_data["dvy"], dtype=jnp.float64)
        vol = jnp.asarray(self.local_data.local_elem_data["vol"], dtype=jnp.float64)
        energy_weights = jnp.asarray(self.local_data.energy_weights, dtype=jnp.float64)
        p = float(self.params["p"])

        def element_energy(v_e, dvx_e, dvy_e, vol_e):
            return _plaplace_integrand(v_e, dvx_e, dvy_e, p) * vol_e

        hess_elem = jax.vmap(jax.hessian(element_energy), in_axes=(0, 0, 0, 0))

        def local_full_energy(v_local):
            v_e = v_local[elems]
            e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0))(v_e, dvx, dvy, vol)
            return jnp.sum(e)

        grad_local = jax.grad(local_full_energy)

        @jax.jit
        def energy_fn(v_local):
            v_e = v_local[elems]
            e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0))(v_e, dvx, dvy, vol)
            return jnp.sum(e * energy_weights)

        @jax.jit
        def local_grad_fn(v_local):
            return grad_local(v_local)

        @jax.jit
        def elem_hess_fn(v_local):
            v_e = v_local[elems]
            return hess_elem(v_e, dvx, dvy, vol)

        return energy_fn, local_grad_fn, elem_hess_fn, grad_local

    def _build_rhs_owned(self) -> np.ndarray:
        rhs = np.asarray(self.params["f"], dtype=np.float64)
        freedofs = np.asarray(self.params["freedofs"], dtype=np.int64)
        rhs_free = rhs[freedofs]
        rhs_reordered = rhs_free[self.layout.perm]
        return np.asarray(rhs_reordered[self.layout.lo : self.layout.hi], dtype=np.float64)

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

        j_arr = np.unique(local_reord[local_reord >= 0]).astype(np.int64)
        n_j = len(j_arr)
        j_to_idx = np.full(self.layout.n_free, -1, dtype=np.int64)
        j_to_idx[j_arr] = np.arange(n_j, dtype=np.int64)

        a_j = sparse.csr_matrix(
            (
                np.ones(len(row_arr), dtype=np.float64),
                (j_to_idx[row_arr], j_to_idx[col_arr]),
            ),
            shape=(n_j, n_j),
        )
        a_j.data[:] = 1.0
        a_j.eliminate_zeros()

        a2_j = sparse.csr_matrix(a_j @ a_j)
        a2_j.data[:] = 1.0
        a2_j.eliminate_zeros()

        a2_j_coo = a2_j.tocoo()
        lo_tri = a2_j_coo.row > a2_j_coo.col
        edges = np.column_stack((a2_j_coo.row[lo_tri], a2_j_coo.col[lo_tri]))
        graph = igraph.Graph(
            n_j, edges.tolist() if len(edges) > 0 else [], directed=False
        )
        coloring_raw = graph.vertex_coloring_greedy()
        self._sfd_local_coloring = np.array(coloring_raw, dtype=np.int32).ravel()
        self._sfd_n_colors = int(self._sfd_local_coloring.max() + 1) if n_j > 0 else 0
        self._sfd_j_dofs = j_arr

        reord_to_local = np.full(self.layout.n_free, -1, dtype=np.int64)
        free_mask = local_reord >= 0
        reord_to_local[local_reord[free_mask]] = np.where(free_mask)[0]

        owned_local_rows = reord_to_local[self.layout.owned_rows]
        if np.any(owned_local_rows < 0):
            raise RuntimeError("Owned reordered rows are missing from the overlap domain")
        owned_col_j_idx = j_to_idx[self.layout.owned_cols]
        if np.any(owned_col_j_idx < 0):
            raise RuntimeError("Owned reordered columns are missing from the local SFD set")
        owned_col_colors = self._sfd_local_coloring[owned_col_j_idx]

        self._sfd_color_nz = {}
        for color in range(self._sfd_n_colors):
            mask_c = owned_col_colors == color
            positions = np.where(mask_c)[0].astype(np.int64)
            local_rows = owned_local_rows[positions].astype(np.int64)
            self._sfd_color_nz[color] = (positions, local_rows)

        indicators_local = []
        for color in range(self._sfd_n_colors):
            indicator = np.zeros(len(local_reord), dtype=np.float64)
            j_dofs_c = self._sfd_j_dofs[self._sfd_local_coloring == color]
            local_idx = reord_to_local[j_dofs_c]
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
            raise RuntimeError("Failed to map all owned DOFs to overlap-local indices")

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
        load = float(self.comm.allreduce(np.dot(self._f_owned, vec.array[:]), op=MPI.SUM))
        return energy - load

    def gradient_fn(self, vec, g):
        full, _ = self._allgather_full_owned(np.asarray(vec.array[:], dtype=np.float64))
        v_local, _ = self._build_v_local(full)
        grad_local = np.asarray(self._grad_jit(jnp.asarray(v_local)).block_until_ready())
        g.array[:] = grad_local[self._scatter.owned_local_pos] - self._f_owned

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
        for color in range(self._sfd_n_colors):
            positions, local_rows = self._sfd_color_nz[color]
            if len(positions) > 0:
                owned_vals[positions] = all_hvps_np[color, local_rows]
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
            all_hvps = self._sfd_hvp_vmap(jnp.asarray(v_local), self._sfd_indicators_stacked)
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
