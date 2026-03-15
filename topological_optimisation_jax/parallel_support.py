from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from tools_petsc4py.reasons import ksp_reason_name
from topological_optimisation_jax.jax_energy import theta_from_latent


jax.config.update("jax_enable_x64", True)


def _golden_section_scalar(f, a: float, b: float, tol: float = 1e-2, max_it: int = 64) -> tuple[float, float]:
    """Minimise a scalar function on [a, b] with a lightweight golden search."""
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    a = float(a)
    b = float(b)
    tol = max(float(tol), 1e-12)
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = float(f(c))
    fd = float(f(d))
    for _ in range(max_it):
        if abs(b - a) <= tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / phi
            fc = float(f(c))
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / phi
            fd = float(f(d))
    x = 0.5 * (a + b)
    return x, float(f(x))


@dataclass
class ColumnOwnership:
    rank_x_start: np.ndarray
    rank_x_end: np.ndarray
    rank_n_cols: np.ndarray


@dataclass
class ExchangeTimings:
    build_v_local: float


def _partition_columns(nx: int, size: int) -> ColumnOwnership:
    counts = np.full(size, nx // size, dtype=np.int64)
    counts[: nx % size] += 1
    starts = np.empty(size, dtype=np.int64)
    ends = np.empty(size, dtype=np.int64)
    cursor = 1
    for rank in range(size):
        starts[rank] = cursor
        ends[rank] = cursor + counts[rank] - 1
        cursor += counts[rank]
    return ColumnOwnership(rank_x_start=starts, rank_x_end=ends, rank_n_cols=counts)


def _build_free_design_index_map(
    *,
    nx: int,
    ny: int,
    fixed_pad_cells: int,
    load_pad_cells: int,
    load_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_idx = np.arange(nx + 1, dtype=np.int32)[:, None]
    y_idx = np.arange(ny + 1, dtype=np.int32)[None, :]

    left_solid = x_idx <= fixed_pad_cells
    right_zone = x_idx >= (nx - fixed_pad_cells)

    load_center = 0.5 * ny
    load_half = 0.5 * load_fraction * ny
    y_min = int(np.floor(load_center - load_half - load_pad_cells - 1e-12))
    y_max = int(np.ceil(load_center + load_half + load_pad_cells + 1e-12))
    y_min = max(0, y_min)
    y_max = min(ny, y_max)
    load_pad = right_zone & (y_idx >= y_min) & (y_idx <= y_max)

    fixed_mask = np.asarray(left_solid | load_pad, dtype=bool)
    index_map = np.full((nx + 1, ny + 1), -1, dtype=np.int64)
    cursor = 0
    for ix in range(nx + 1):
        free_y = np.flatnonzero(~fixed_mask[ix])
        n_free = int(free_y.size)
        if n_free:
            index_map[ix, free_y] = cursor + np.arange(n_free, dtype=np.int64)
            cursor += n_free
    return index_map, fixed_mask


def _owner_of_indices(indices: np.ndarray, rank_hi: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.searchsorted(rank_hi, np.asarray(indices, dtype=np.int64), side="right").astype(np.int64)


class OwnedGhostLayout:
    def __init__(
        self,
        *,
        global_ids_local: np.ndarray,
        template_local: np.ndarray,
        n_free: int,
        lo: int,
        hi: int,
        rank_lo: np.ndarray,
        rank_hi: np.ndarray,
        comm: MPI.Comm,
        tag_base: int = 1000,
    ) -> None:
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.global_ids_local = np.asarray(global_ids_local, dtype=np.int64)
        self.v_template = np.asarray(template_local, dtype=np.float64)
        self.n_local = int(self.global_ids_local.size)
        self.n_free = int(n_free)
        self.lo = int(lo)
        self.hi = int(hi)
        self.n_owned = int(max(0, self.hi - self.lo))
        self.rank_lo = np.asarray(rank_lo, dtype=np.int64)
        self.rank_hi = np.asarray(rank_hi, dtype=np.int64)
        self.tag_base = int(tag_base)

        free_mask = self.global_ids_local >= 0
        self.free_local_indices = np.flatnonzero(free_mask).astype(np.int64)
        self.free_global_indices = self.global_ids_local[free_mask]

        owned_mask = (self.free_global_indices >= self.lo) & (self.free_global_indices < self.hi)
        owned_local = self.free_local_indices[owned_mask]
        owned_global = self.free_global_indices[owned_mask]
        owned_order = np.argsort(owned_global)
        self.owned_local_indices = owned_local[owned_order].astype(np.int64)
        self._owned_offsets = (owned_global[owned_order] - self.lo).astype(np.int64)

        self._gather_sizes = (self.rank_hi - self.rank_lo).astype(np.int64)
        self._gather_displs = self.rank_lo.copy()
        self._gather_sizes_list = [int(v) for v in self._gather_sizes]
        self._gather_displs_list = [int(v) for v in self._gather_displs]
        self._setup_ghost_exchange()

    def _setup_ghost_exchange(self) -> None:
        free_global = self.free_global_indices
        owned_mask = (free_global >= self.lo) & (free_global < self.hi)
        ghost_mask = ~owned_mask

        self._p2p_owned_local = self.free_local_indices[owned_mask]
        self._p2p_owned_offset = (free_global[owned_mask] - self.lo).astype(np.int64)

        ghost_local = self.free_local_indices[ghost_mask]
        ghost_global = free_global[ghost_mask]

        self._ghost_recv: dict[int, np.ndarray] = {}
        send_requests: dict[int, np.ndarray] = {}
        if ghost_global.size:
            ghost_owners = _owner_of_indices(ghost_global, self.rank_hi)
            for r in np.unique(ghost_owners):
                r = int(r)
                if r == self.rank:
                    continue
                mask_r = ghost_owners == r
                self._ghost_recv[r] = ghost_local[mask_r]
                send_requests[r] = (ghost_global[mask_r] - self.rank_lo[r]).astype(np.int64)

        n_we_need = np.zeros(self.size, dtype=np.int64)
        for r, offsets in send_requests.items():
            n_we_need[r] = len(offsets)
        n_others_need = np.zeros(self.size, dtype=np.int64)
        self.comm.Alltoall(n_we_need, n_others_need)

        recv_reqs = []
        self._ghost_send_offsets: dict[int, np.ndarray] = {}
        for r in range(self.size):
            if r == self.rank or n_others_need[r] == 0:
                continue
            buf = np.empty(int(n_others_need[r]), dtype=np.int64)
            req = self.comm.Irecv(buf, source=r, tag=200)
            recv_reqs.append((req, r, buf))

        send_reqs = []
        for r, offsets in send_requests.items():
            req = self.comm.Isend(np.ascontiguousarray(offsets), dest=r, tag=200)
            send_reqs.append(req)

        for req, r, buf in recv_reqs:
            req.Wait()
            self._ghost_send_offsets[r] = buf
        for req in send_reqs:
            req.Wait()

        self._ghost_send_counts = np.zeros(self.size, dtype=np.int32)
        self._ghost_recv_counts = np.zeros(self.size, dtype=np.int32)
        send_offsets_concat: list[np.ndarray] = []
        recv_local_concat: list[np.ndarray] = []
        for r in range(self.size):
            send_offsets_r = self._ghost_send_offsets.get(r)
            recv_local_r = self._ghost_recv.get(r)
            if send_offsets_r is not None:
                self._ghost_send_counts[r] = int(len(send_offsets_r))
                if len(send_offsets_r):
                    send_offsets_concat.append(np.asarray(send_offsets_r, dtype=np.int64))
            if recv_local_r is not None:
                self._ghost_recv_counts[r] = int(len(recv_local_r))
                if len(recv_local_r):
                    recv_local_concat.append(np.asarray(recv_local_r, dtype=np.int64))

        self._ghost_send_displs = np.zeros(self.size, dtype=np.int32)
        self._ghost_recv_displs = np.zeros(self.size, dtype=np.int32)
        if self.size > 1:
            self._ghost_send_displs[1:] = np.cumsum(self._ghost_send_counts[:-1], dtype=np.int32)
            self._ghost_recv_displs[1:] = np.cumsum(self._ghost_recv_counts[:-1], dtype=np.int32)

        self._ghost_send_offsets_concat = (
            np.concatenate(send_offsets_concat).astype(np.int64)
            if send_offsets_concat
            else np.zeros(0, dtype=np.int64)
        )
        self._ghost_recv_local_concat = (
            np.concatenate(recv_local_concat).astype(np.int64)
            if recv_local_concat
            else np.zeros(0, dtype=np.int64)
        )
        self._ghost_send_buf = np.empty(int(np.sum(self._ghost_send_counts)), dtype=np.float64)
        self._ghost_recv_buf = np.empty(int(np.sum(self._ghost_recv_counts)), dtype=np.float64)
        self._ghost_send_counts_list = [int(v) for v in self._ghost_send_counts]
        self._ghost_recv_counts_list = [int(v) for v in self._ghost_recv_counts]
        self._ghost_send_displs_list = [int(v) for v in self._ghost_send_displs]
        self._ghost_recv_displs_list = [int(v) for v in self._ghost_recv_displs]

    def _p2p_fill(self, template: np.ndarray | None, u_owned: np.ndarray, tag: int) -> np.ndarray:
        if template is not None:
            v = template.copy()
        else:
            v = np.zeros(self.n_local, dtype=np.float64)
        if self.n_owned:
            v[self._p2p_owned_local] = u_owned[self._p2p_owned_offset]
        if not self._ghost_recv_local_concat.size and not self._ghost_send_offsets_concat.size:
            return v

        if self._ghost_send_offsets_concat.size:
            self._ghost_send_buf[:] = u_owned[self._ghost_send_offsets_concat]
        if self._ghost_recv_buf.size:
            self._ghost_recv_buf.fill(0.0)
        self.comm.Alltoallv(
            [
                self._ghost_send_buf,
                self._ghost_send_counts_list,
                self._ghost_send_displs_list,
                MPI.DOUBLE,
            ],
            [
                self._ghost_recv_buf,
                self._ghost_recv_counts_list,
                self._ghost_recv_displs_list,
                MPI.DOUBLE,
            ],
        )
        if self._ghost_recv_local_concat.size:
            v[self._ghost_recv_local_concat] = self._ghost_recv_buf
        return v

    def build_v_local_p2p(self, u_owned: np.ndarray) -> tuple[np.ndarray, ExchangeTimings]:
        t0 = time.perf_counter()
        v = self._p2p_fill(
            self.v_template,
            np.asarray(u_owned, dtype=np.float64),
            tag=self.tag_base + 1,
        )
        return v, ExchangeTimings(build_v_local=time.perf_counter() - t0)

    def build_zero_local_p2p(self, u_owned: np.ndarray) -> tuple[np.ndarray, ExchangeTimings]:
        t0 = time.perf_counter()
        v = self._p2p_fill(
            None,
            np.asarray(u_owned, dtype=np.float64),
            tag=self.tag_base + 2,
        )
        return v, ExchangeTimings(build_v_local=time.perf_counter() - t0)

    def create_vec(self, fill: float = 0.0) -> PETSc.Vec:
        v = PETSc.Vec().createMPI((self.n_owned, self.n_free), comm=self.comm)
        arr = v.getArray(readonly=False)
        arr[:] = float(fill)
        del arr
        v.assemble()
        return v

    def gather_full(self, u_owned: np.ndarray) -> np.ndarray:
        send = np.ascontiguousarray(u_owned, dtype=np.float64)
        out = np.empty(self.n_free, dtype=np.float64)
        recvbuf = None
        if self.rank == 0:
            recvbuf = [out, self._gather_sizes_list, self._gather_displs_list, MPI.DOUBLE]
        self.comm.Gatherv(send, recvbuf, root=0)
        self.comm.Bcast(out, root=0)
        return out


@dataclass
class StructuredTopologyPartition:
    nx: int
    ny: int
    length: float
    height: float
    traction: float
    load_fraction: float
    fixed_pad_cells: int
    load_pad_cells: int
    theta_min: float
    solid_latent: float
    comm: MPI.Comm
    ownership: ColumnOwnership
    owned_x_start: int
    owned_x_end: int
    local_node_x_start: int
    local_node_x_end: int
    local_cell_x_start: int
    local_cell_x_end: int
    node_ix_local: np.ndarray
    node_iy_local: np.ndarray
    coords_local: np.ndarray
    scalar_elems_local: np.ndarray
    vector_elems_local: np.ndarray
    elem_area: np.ndarray
    elem_grad_phi: np.ndarray
    elem_B: np.ndarray
    elem_cell_x: np.ndarray
    owned_cell_mask: np.ndarray
    design_energy_weights: np.ndarray
    scalar_layout: OwnedGhostLayout
    vector_layout: OwnedGhostLayout
    force_owned: np.ndarray
    free_count_per_col: np.ndarray
    design_index_map: np.ndarray
    design_fixed_mask: np.ndarray
    design_global_ids_local: np.ndarray
    vector_global_ids_local: np.ndarray
    scalar_rank_lo: np.ndarray
    scalar_rank_hi: np.ndarray
    vector_rank_lo: np.ndarray
    vector_rank_hi: np.ndarray
    domain_area: float

    @property
    def n_nodes(self) -> int:
        return int((self.nx + 1) * (self.ny + 1))

    @property
    def n_disp_dofs(self) -> int:
        return int(2 * self.n_nodes)

    @property
    def n_scalar_free(self) -> int:
        return int(self.scalar_layout.n_free)

    @property
    def n_vector_free(self) -> int:
        return int(self.vector_layout.n_free)


def _local_node_index(ix: int, iy: int, local_node_x_start: int, ny: int) -> int:
    return (ix - local_node_x_start) * (ny + 1) + iy


def _build_local_geometry(
    *,
    nx: int,
    ny: int,
    length: float,
    height: float,
    owned_x_start: int,
    owned_x_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    if owned_x_end < owned_x_start:
        local_node_x_start = 0
        local_node_x_end = -1
        local_cell_x_start = 0
        local_cell_x_end = -1
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 6), dtype=np.int32),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 3, 2), dtype=np.float64),
            np.zeros((0, 3, 6), dtype=np.float64),
            local_node_x_start,
            local_node_x_end,
        )

    hx = length / float(nx)
    hy = height / float(ny)
    local_cell_x_start = max(0, owned_x_start - 1)
    local_cell_x_end = min(nx - 1, owned_x_end)
    local_node_x_start = local_cell_x_start
    local_node_x_end = local_cell_x_end + 1

    x_cols = np.arange(local_node_x_start, local_node_x_end + 1, dtype=np.int32)
    y_rows = np.arange(ny + 1, dtype=np.int32)
    node_ix = np.repeat(x_cols, ny + 1)
    node_iy = np.tile(y_rows, len(x_cols))
    coords = np.column_stack((node_ix.astype(np.float64) * hx, node_iy.astype(np.float64) * hy))

    scalar_elems = []
    vector_elems = []
    elem_cell_x = []
    for ix in range(local_cell_x_start, local_cell_x_end + 1):
        for iy in range(ny):
            n00 = _local_node_index(ix, iy, local_node_x_start, ny)
            n10 = _local_node_index(ix + 1, iy, local_node_x_start, ny)
            n01 = _local_node_index(ix, iy + 1, local_node_x_start, ny)
            n11 = _local_node_index(ix + 1, iy + 1, local_node_x_start, ny)
            if (ix + iy) % 2 == 0:
                tris = [(n00, n10, n11), (n00, n11, n01)]
            else:
                tris = [(n00, n10, n01), (n10, n11, n01)]
            for tri in tris:
                scalar_elems.append(tri)
                vector_elems.append(
                    (
                        2 * tri[0],
                        2 * tri[0] + 1,
                        2 * tri[1],
                        2 * tri[1] + 1,
                        2 * tri[2],
                        2 * tri[2] + 1,
                    )
                )
                elem_cell_x.append(ix)

    scalar_elems_np = np.asarray(scalar_elems, dtype=np.int32)
    vector_elems_np = np.asarray(vector_elems, dtype=np.int32)
    elem_cell_x_np = np.asarray(elem_cell_x, dtype=np.int32)

    if scalar_elems_np.size == 0:
        return (
            node_ix,
            node_iy,
            coords,
            scalar_elems_np.reshape(0, 3),
            vector_elems_np.reshape(0, 6),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 3, 2), dtype=np.float64),
            np.zeros((0, 3, 6), dtype=np.float64),
            local_node_x_start,
            local_node_x_end,
        )

    xy = coords[scalar_elems_np]
    x0 = xy[:, 0, 0]
    y0 = xy[:, 0, 1]
    x1 = xy[:, 1, 0]
    y1 = xy[:, 1, 1]
    x2 = xy[:, 2, 0]
    y2 = xy[:, 2, 1]
    twice_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    elem_area = 0.5 * np.abs(twice_area)
    grad_x = np.column_stack((y1 - y2, y2 - y0, y0 - y1)) / twice_area[:, None]
    grad_y = np.column_stack((x2 - x1, x0 - x2, x1 - x0)) / twice_area[:, None]
    elem_grad_phi = np.stack((grad_x, grad_y), axis=2)
    elem_B = np.zeros((scalar_elems_np.shape[0], 3, 6), dtype=np.float64)
    for local_node in range(3):
        dphix = elem_grad_phi[:, local_node, 0]
        dphiy = elem_grad_phi[:, local_node, 1]
        elem_B[:, 0, 2 * local_node] = dphix
        elem_B[:, 1, 2 * local_node + 1] = dphiy
        elem_B[:, 2, 2 * local_node] = dphiy
        elem_B[:, 2, 2 * local_node + 1] = dphix

    return (
        node_ix,
        node_iy,
        coords,
        scalar_elems_np,
        vector_elems_np,
        elem_area,
        elem_grad_phi,
        elem_B,
        local_node_x_start,
        local_node_x_end,
    )


def _build_vector_global_ids(node_ix: np.ndarray, node_iy: np.ndarray, ny: int) -> np.ndarray:
    out = np.full(2 * node_ix.size, -1, dtype=np.int64)
    free_mask = node_ix > 0
    if np.any(free_mask):
        block = (node_ix[free_mask].astype(np.int64) - 1) * (ny + 1) + node_iy[free_mask].astype(np.int64)
        out[2 * np.flatnonzero(free_mask)] = 2 * block
        out[2 * np.flatnonzero(free_mask) + 1] = 2 * block + 1
    return out


def _build_force_owned(
    *,
    nx: int,
    ny: int,
    height: float,
    traction: float,
    load_fraction: float,
    owned_x_start: int,
    owned_x_end: int,
    lo: int,
    hi: int,
) -> np.ndarray:
    n_owned = max(0, hi - lo)
    force = np.zeros(n_owned, dtype=np.float64)
    if owned_x_end < owned_x_start or nx < owned_x_start or nx > owned_x_end:
        return force

    load_center = 0.5 * height
    load_half = 0.5 * load_fraction * height
    load_min = load_center - load_half - 1e-12
    load_max = load_center + load_half + 1e-12
    hy = height / float(ny)
    traction_val = -abs(traction)

    for iy in range(ny):
        y0 = iy * hy
        y1 = (iy + 1) * hy
        overlap = max(0.0, min(y1, load_max) - max(y0, load_min))
        if overlap <= 0.0:
            continue
        nodal = 0.5 * overlap * traction_val
        for node_y in (iy, iy + 1):
            block = (nx - 1) * (ny + 1) + node_y
            dof = 2 * block + 1
            force[dof - lo] += nodal
    return force


def build_structured_topology_partition(
    *,
    nx: int,
    ny: int,
    length: float,
    height: float,
    traction: float,
    load_fraction: float,
    fixed_pad_cells: int,
    load_pad_cells: int,
    theta_min: float,
    solid_latent: float,
    comm: MPI.Comm,
) -> StructuredTopologyPartition:
    ownership = _partition_columns(nx, comm.size)
    rank = comm.Get_rank()
    owned_x_start = int(ownership.rank_x_start[rank])
    owned_x_end = int(ownership.rank_x_end[rank])

    (
        node_ix_local,
        node_iy_local,
        coords_local,
        scalar_elems_local,
        vector_elems_local,
        elem_area,
        elem_grad_phi,
        elem_B,
        local_node_x_start,
        local_node_x_end,
    ) = _build_local_geometry(
        nx=nx,
        ny=ny,
        length=length,
        height=height,
        owned_x_start=owned_x_start,
        owned_x_end=owned_x_end,
    )
    local_cell_x_start = max(0, owned_x_start - 1) if owned_x_end >= owned_x_start else 0
    local_cell_x_end = min(nx - 1, owned_x_end) if owned_x_end >= owned_x_start else -1
    elem_cell_x = np.repeat(np.arange(local_cell_x_start, local_cell_x_end + 1, dtype=np.int32), 2 * ny)
    if elem_area.size == 0:
        elem_cell_x = np.zeros(0, dtype=np.int32)

    design_index_map, design_fixed_mask = _build_free_design_index_map(
        nx=nx,
        ny=ny,
        fixed_pad_cells=fixed_pad_cells,
        load_pad_cells=load_pad_cells,
        load_fraction=load_fraction,
    )
    free_count_per_col = np.count_nonzero(design_index_map >= 0, axis=1).astype(np.int64)

    vector_rank_sizes = 2 * ownership.rank_n_cols * (ny + 1)
    vector_rank_lo = np.zeros(comm.size, dtype=np.int64)
    if comm.size > 1:
        vector_rank_lo[1:] = np.cumsum(vector_rank_sizes[:-1])
    vector_rank_hi = vector_rank_lo + vector_rank_sizes

    scalar_rank_sizes = np.zeros(comm.size, dtype=np.int64)
    for r in range(comm.size):
        x0 = ownership.rank_x_start[r]
        x1 = ownership.rank_x_end[r]
        if x1 >= x0:
            scalar_rank_sizes[r] = int(np.sum(free_count_per_col[x0 : x1 + 1]))
    scalar_rank_lo = np.zeros(comm.size, dtype=np.int64)
    if comm.size > 1:
        scalar_rank_lo[1:] = np.cumsum(scalar_rank_sizes[:-1])
    scalar_rank_hi = scalar_rank_lo + scalar_rank_sizes

    if local_node_x_end >= local_node_x_start:
        x_slice = slice(local_node_x_start, local_node_x_end + 1)
        design_global_ids_local = design_index_map[x_slice].reshape(-1)
        design_fixed_local = design_fixed_mask[x_slice].reshape(-1)
    else:
        design_global_ids_local = np.zeros(0, dtype=np.int64)
        design_fixed_local = np.zeros(0, dtype=bool)

    design_template_local = np.zeros(design_global_ids_local.size, dtype=np.float64)
    design_template_local[design_fixed_local] = float(solid_latent)
    vector_global_ids_local = _build_vector_global_ids(node_ix_local, node_iy_local, ny)

    scalar_lo = int(scalar_rank_lo[rank])
    scalar_hi = int(scalar_rank_hi[rank])
    vector_lo = int(vector_rank_lo[rank])
    vector_hi = int(vector_rank_hi[rank])

    scalar_layout = OwnedGhostLayout(
        global_ids_local=design_global_ids_local,
        template_local=design_template_local,
        n_free=int(np.max(design_index_map) + 1 if np.any(design_index_map >= 0) else 0),
        lo=scalar_lo,
        hi=scalar_hi,
        rank_lo=scalar_rank_lo,
        rank_hi=scalar_rank_hi,
        comm=comm,
        tag_base=1000,
    )
    vector_layout = OwnedGhostLayout(
        global_ids_local=vector_global_ids_local,
        template_local=np.zeros(vector_global_ids_local.size, dtype=np.float64),
        n_free=int(vector_rank_hi[-1] if vector_rank_hi.size else 0),
        lo=vector_lo,
        hi=vector_hi,
        rank_lo=vector_rank_lo,
        rank_hi=vector_rank_hi,
        comm=comm,
        tag_base=2000,
    )

    if elem_cell_x.size:
        owned_cell_rank = np.empty_like(elem_cell_x, dtype=np.int64)
        for i, cell_x in enumerate(elem_cell_x.tolist()):
            owner_node_col = 1 if cell_x == 0 else cell_x
            owned_cell_rank[i] = int(np.searchsorted(ownership.rank_x_end, owner_node_col, side="left"))
        owned_cell_mask = owned_cell_rank == rank
    else:
        owned_cell_mask = np.zeros(0, dtype=bool)

    if scalar_elems_local.size:
        elem_scalar_ids = design_global_ids_local[scalar_elems_local]
        masked = np.where(elem_scalar_ids >= 0, elem_scalar_ids, np.iinfo(np.int64).max)
        elem_min = masked.min(axis=1)
        valid = elem_min < np.iinfo(np.int64).max
        design_owner = np.full(elem_min.shape[0], -1, dtype=np.int64)
        if np.any(valid):
            design_owner[valid] = _owner_of_indices(elem_min[valid], scalar_rank_hi)
        design_energy_weights = (design_owner == rank).astype(np.float64)
    else:
        design_energy_weights = np.zeros(0, dtype=np.float64)

    force_owned = _build_force_owned(
        nx=nx,
        ny=ny,
        height=height,
        traction=traction,
        load_fraction=load_fraction,
        owned_x_start=owned_x_start,
        owned_x_end=owned_x_end,
        lo=vector_lo,
        hi=vector_hi,
    )

    return StructuredTopologyPartition(
        nx=nx,
        ny=ny,
        length=length,
        height=height,
        traction=traction,
        load_fraction=load_fraction,
        fixed_pad_cells=fixed_pad_cells,
        load_pad_cells=load_pad_cells,
        theta_min=theta_min,
        solid_latent=solid_latent,
        comm=comm,
        ownership=ownership,
        owned_x_start=owned_x_start,
        owned_x_end=owned_x_end,
        local_node_x_start=local_node_x_start,
        local_node_x_end=local_node_x_end,
        local_cell_x_start=local_cell_x_start,
        local_cell_x_end=local_cell_x_end,
        node_ix_local=node_ix_local,
        node_iy_local=node_iy_local,
        coords_local=coords_local,
        scalar_elems_local=scalar_elems_local,
        vector_elems_local=vector_elems_local,
        elem_area=np.asarray(elem_area, dtype=np.float64),
        elem_grad_phi=np.asarray(elem_grad_phi, dtype=np.float64),
        elem_B=np.asarray(elem_B, dtype=np.float64),
        elem_cell_x=np.asarray(elem_cell_x, dtype=np.int32),
        owned_cell_mask=np.asarray(owned_cell_mask, dtype=bool),
        design_energy_weights=np.asarray(design_energy_weights, dtype=np.float64),
        scalar_layout=scalar_layout,
        vector_layout=vector_layout,
        force_owned=force_owned,
        free_count_per_col=free_count_per_col,
        design_index_map=design_index_map,
        design_fixed_mask=design_fixed_mask,
        design_global_ids_local=design_global_ids_local,
        vector_global_ids_local=vector_global_ids_local,
        scalar_rank_lo=scalar_rank_lo,
        scalar_rank_hi=scalar_rank_hi,
        vector_rank_lo=vector_rank_lo,
        vector_rank_hi=vector_rank_hi,
        domain_area=float(length * height),
    )


def _compute_initial_latent(
    partition: StructuredTopologyPartition,
    *,
    target_volume_fraction: float,
    theta_min: float,
    solid_latent: float,
) -> float:
    theta_solid = float(theta_from_latent(jnp.asarray([solid_latent], dtype=jnp.float64), theta_min)[0])
    if partition.scalar_elems_local.size:
        fixed_count = np.sum(partition.design_fixed_mask[partition.node_ix_local, partition.node_iy_local][partition.scalar_elems_local], axis=1)
        local_fixed = float(np.sum(partition.elem_area[partition.owned_cell_mask] * (fixed_count[partition.owned_cell_mask] / 3.0)))
        local_free = float(np.sum(partition.elem_area[partition.owned_cell_mask] * ((3.0 - fixed_count[partition.owned_cell_mask]) / 3.0)))
    else:
        local_fixed = 0.0
        local_free = 0.0
    total_fixed = float(partition.comm.allreduce(local_fixed, op=MPI.SUM))
    total_free = float(partition.comm.allreduce(local_free, op=MPI.SUM))
    min_volume = total_fixed * theta_solid + total_free * theta_min
    target_volume = target_volume_fraction * partition.domain_area
    if target_volume < min_volume - 1e-12:
        raise ValueError("Target volume fraction is infeasible for the fixed design pads.")
    theta_free = (target_volume - theta_solid * total_fixed) / max(total_free, 1e-12)
    theta_free = float(np.clip(theta_free, theta_min + 1e-6, 1.0 - 1e-6))
    scaled = (theta_free - theta_min) / (1.0 - theta_min)
    return float(np.log(scaled / (1.0 - scaled)))


def _build_near_nullspace(partition: StructuredTopologyPartition, comm: MPI.Comm) -> PETSc.NullSpace:
    n_owned_nodes = partition.vector_layout.n_owned // 2
    if n_owned_nodes == 0:
        return PETSc.NullSpace().create(vectors=[], comm=comm)
    coords = np.empty((n_owned_nodes, 2), dtype=np.float64)
    for local_k in range(n_owned_nodes):
        global_block = partition.vector_layout.lo // 2 + local_k
        x_col = global_block // (partition.ny + 1) + 1
        y_row = global_block % (partition.ny + 1)
        coords[local_k, 0] = (partition.length / partition.nx) * x_col
        coords[local_k, 1] = (partition.height / partition.ny) * y_row

    modes = []
    tx = PETSc.Vec().createMPI((partition.vector_layout.n_owned, partition.vector_layout.n_free), comm=comm)
    ty = tx.duplicate()
    rot = tx.duplicate()
    tx_arr = tx.array
    ty_arr = ty.array
    rot_arr = rot.array
    tx_arr[0::2] = 1.0
    ty_arr[1::2] = 1.0
    rot_arr[0::2] = -coords[:, 1]
    rot_arr[1::2] = coords[:, 0]
    for vec in (tx, ty, rot):
        vec.assemble()
        modes.append(vec)
    return PETSc.NullSpace().create(vectors=modes, comm=comm)


class TopologyMechanicsAssembler:
    distribution_strategy = "structured_column_local"

    def __init__(
        self,
        *,
        partition: StructuredTopologyPartition,
        constitutive: np.ndarray,
        comm: MPI.Comm,
        ksp_rtol: float = 1e-2,
        ksp_type: str = "cg",
        pc_type: str = "gamg",
        ksp_max_it: int = 80,
        use_near_nullspace: bool = True,
        pc_options: dict[str, object] | None = None,
        gamg_set_coordinates: bool = True,
        fallback_solvers: list[dict[str, object]] | None = None,
    ) -> None:
        self.partition = partition
        self.comm = comm
        self.rank = comm.Get_rank()
        self.constitutive = jnp.asarray(constitutive, dtype=jnp.float64)
        self.gamg_set_coordinates = bool(gamg_set_coordinates)
        self.iter_timings: list[dict[str, object]] = []
        self._nullspace = None
        self._primary_ksp_type = str(ksp_type)
        self._primary_pc_type = str(pc_type)
        self._primary_ksp_rtol = float(ksp_rtol)
        self._primary_ksp_max_it = int(ksp_max_it)
        self._primary_pc_options = dict(pc_options or {})
        self._fallback_solvers = [dict(spec) for spec in (fallback_solvers or [])]

        self._material_scale_local = jnp.ones(partition.elem_area.shape[0], dtype=jnp.float64)
        self._theta_elem_local_np = np.ones(partition.elem_area.shape[0], dtype=np.float64)
        self._energy_jit, self._grad_jit, self._elem_hess_jit = self._make_local_kernels()
        self._coo_rows, self._coo_cols, self._scatter_e, self._scatter_i, self._scatter_j, self._scatter_pos = self._build_scatter_data()
        self._warmup()

        self.A = PETSc.Mat().create(comm=comm)
        self.A.setType(PETSc.Mat.Type.MPIAIJ)
        self.A.setSizes(((partition.vector_layout.n_owned, partition.vector_layout.n_free),) * 2)
        self.A.setPreallocationCOO(
            self._coo_rows.astype(PETSc.IntType),
            self._coo_cols.astype(PETSc.IntType),
        )
        self.A.setBlockSize(2)
        self.P = PETSc.Mat().create(comm=comm)
        self.P.setType(PETSc.Mat.Type.MPIAIJ)
        self.P.setSizes(((partition.vector_layout.n_owned, partition.vector_layout.n_free),) * 2)
        self.P.setPreallocationCOO(
            self._coo_rows.astype(PETSc.IntType),
            self._coo_cols.astype(PETSc.IntType),
        )
        self.P.setBlockSize(2)
        if use_near_nullspace:
            self._nullspace = _build_near_nullspace(partition, comm)
            self.A.setNearNullSpace(self._nullspace)
            self.P.setNearNullSpace(self._nullspace)

        self.ksp = PETSc.KSP().create(comm)
        self._configure_ksp(
            self.ksp,
            ksp_type=self._primary_ksp_type,
            pc_type=self._primary_pc_type,
            ksp_rtol=self._primary_ksp_rtol,
            ksp_max_it=self._primary_ksp_max_it,
            pc_options=self._primary_pc_options,
            prefix_tag="primary",
        )
        self._diag_positions = np.flatnonzero(self._coo_rows == self._coo_cols).astype(np.int64)
        self._latest_owned_vals = np.zeros(self._coo_rows.size, dtype=np.float64)
        self._latest_diag_abs_max = 0.0

    def _make_local_kernels(self):
        elems = jnp.asarray(self.partition.vector_elems_local, dtype=jnp.int32)
        elem_B = jnp.asarray(self.partition.elem_B, dtype=jnp.float64)
        elem_area = jnp.asarray(self.partition.elem_area, dtype=jnp.float64)
        constitutive = self.constitutive

        def element_energy(u_e, elem_B_e, elem_area_e, scale_e):
            strain = jnp.einsum("ij,j->i", elem_B_e, u_e)
            elastic_density = 0.5 * jnp.einsum("i,ij,j->", strain, constitutive, strain)
            return elem_area_e * scale_e * elastic_density

        vmapped_hess = jax.vmap(jax.hessian(element_energy), in_axes=(0, 0, 0, 0))

        @jax.jit
        def grad_fn(v_local, material_scale_local):
            u_e = v_local[elems]
            def local_full_energy(v):
                u_e2 = v[elems]
                e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0))(u_e2, elem_B, elem_area, material_scale_local)
                return jnp.sum(e)
            return jax.grad(local_full_energy)(v_local)

        @jax.jit
        def elem_hess_fn(v_local, material_scale_local):
            return vmapped_hess(v_local[elems], elem_B, elem_area, material_scale_local)

        @jax.jit
        def energy_fn(v_local, material_scale_local):
            u_e = v_local[elems]
            e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0))(u_e, elem_B, elem_area, material_scale_local)
            return jnp.sum(e)

        return energy_fn, grad_fn, elem_hess_fn

    def _build_scatter_data(self):
        elem_global = self.partition.vector_global_ids_local[self.partition.vector_elems_local]
        rows = elem_global[:, :, None]
        cols = elem_global[:, None, :]
        valid = (rows >= self.partition.vector_layout.lo) & (rows < self.partition.vector_layout.hi) & (cols >= 0)
        vi = np.where(valid)
        coo_rows = elem_global[vi[0], vi[1]].astype(np.int64)
        coo_cols = elem_global[vi[0], vi[2]].astype(np.int64)
        keys = coo_rows * np.int64(self.partition.vector_layout.n_free) + coo_cols
        uniq_keys, inverse = np.unique(keys, return_inverse=True)
        return (
            (uniq_keys // np.int64(self.partition.vector_layout.n_free)).astype(np.int64),
            (uniq_keys % np.int64(self.partition.vector_layout.n_free)).astype(np.int64),
            np.asarray(vi[0], dtype=np.int64),
            np.asarray(vi[1], dtype=np.int64),
            np.asarray(vi[2], dtype=np.int64),
            np.asarray(inverse, dtype=np.int64),
        )

    def _warmup(self) -> None:
        dummy = np.zeros(self.partition.vector_layout.n_local, dtype=np.float64)
        dummy_j = jnp.asarray(dummy, dtype=jnp.float64)
        _ = self._energy_jit(dummy_j, self._material_scale_local).block_until_ready()
        _ = self._grad_jit(dummy_j, self._material_scale_local).block_until_ready()
        _ = self._elem_hess_jit(dummy_j, self._material_scale_local).block_until_ready()

    def create_vec(self) -> PETSc.Vec:
        return self.partition.vector_layout.create_vec(0.0)

    @property
    def part(self):
        return self.partition.vector_layout

    def update_material_scale_from_design(self, z_vec: PETSc.Vec, p_penal: float) -> None:
        z_owned = np.asarray(z_vec.array[:], dtype=np.float64)
        z_local, _ = self.partition.scalar_layout.build_v_local_p2p(z_owned)
        theta_local = np.asarray(theta_from_latent(jnp.asarray(z_local, dtype=jnp.float64), self.partition.theta_min))
        if self.partition.scalar_elems_local.size:
            theta_elem = theta_local[self.partition.scalar_elems_local].mean(axis=1)
            self._theta_elem_local_np = np.asarray(theta_elem, dtype=np.float64)
            self._material_scale_local = jnp.asarray(self._theta_elem_local_np**p_penal, dtype=jnp.float64)
        else:
            self._material_scale_local = jnp.zeros(0, dtype=jnp.float64)
            self._theta_elem_local_np = np.zeros(0, dtype=np.float64)

    def assemble_matrix(self, vec: PETSc.Vec) -> dict[str, float]:
        timings: dict[str, float] = {}
        t_total = time.perf_counter()
        u_owned = np.asarray(vec.array[:], dtype=np.float64)
        u_local, exch = self.partition.vector_layout.build_v_local_p2p(u_owned)
        timings["build_v_local"] = float(exch.build_v_local)

        t0 = time.perf_counter()
        elem_hess = np.asarray(self._elem_hess_jit(jnp.asarray(u_local, dtype=jnp.float64), self._material_scale_local).block_until_ready())
        timings["elem_hessian_compute"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        contrib = elem_hess[self._scatter_e, self._scatter_i, self._scatter_j]
        owned_vals = np.zeros(self._coo_rows.size, dtype=np.float64)
        np.add.at(owned_vals, self._scatter_pos, contrib)
        self._latest_owned_vals = owned_vals.copy()
        diag_local = 0.0
        if self._diag_positions.size:
            diag_local = float(np.max(np.abs(owned_vals[self._diag_positions])))
        self._latest_diag_abs_max = float(self.comm.allreduce(diag_local, op=MPI.MAX))
        timings["scatter"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.A.setValuesCOO(owned_vals.astype(PETSc.ScalarType), addv=PETSc.InsertMode.INSERT_VALUES)
        self.A.assemble()
        self.P.setValuesCOO(owned_vals.astype(PETSc.ScalarType), addv=PETSc.InsertMode.INSERT_VALUES)
        self.P.assemble()
        if self._nullspace is not None:
            self.A.setNearNullSpace(self._nullspace)
            self.P.setNearNullSpace(self._nullspace)
        timings["coo_assembly"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(dict(timings))
        return timings

    def _configure_ksp(
        self,
        ksp: PETSc.KSP,
        *,
        ksp_type: str,
        pc_type: str,
        ksp_rtol: float,
        ksp_max_it: int,
        pc_options: dict[str, object] | None,
        prefix_tag: str,
    ) -> None:
        ksp.setType(str(ksp_type))
        ksp.getPC().setType(str(pc_type))
        ksp.setTolerances(rtol=float(ksp_rtol), max_it=int(ksp_max_it))
        if pc_options:
            opts = PETSc.Options()
            safe_tag = "".join(ch if ch.isalnum() else "_" for ch in str(prefix_tag))
            prefix = f"topopt_mech_{id(self)}_{safe_tag}_"
            ksp.setOptionsPrefix(prefix)
            for key, value in pc_options.items():
                opts[f"{prefix}{key}"] = value
        ksp.setFromOptions()

    def _assemble_shifted_pc(self, rho: float) -> None:
        vals = self._latest_owned_vals.copy()
        if self._diag_positions.size:
            vals[self._diag_positions] += float(rho)
        self.P.setValuesCOO(vals.astype(PETSc.ScalarType), addv=PETSc.InsertMode.INSERT_VALUES)
        self.P.assemble()
        if self._nullspace is not None:
            self.P.setNearNullSpace(self._nullspace)

    def _apply_pc_coordinates(self, ksp: PETSc.KSP, *, pc_type: str) -> None:
        if pc_type != "gamg" or not self.gamg_set_coordinates or not self.part.n_owned:
            return
        n_owned_nodes = self.part.n_owned // 2
        coords = np.empty((n_owned_nodes, 2), dtype=np.float64)
        for local_k in range(n_owned_nodes):
            global_block = self.part.lo // 2 + local_k
            x_col = global_block // (self.partition.ny + 1) + 1
            y_row = global_block % (self.partition.ny + 1)
            coords[local_k, 0] = (self.partition.length / self.partition.nx) * x_col
            coords[local_k, 1] = (self.partition.height / self.partition.ny) * y_row
        ksp.getPC().setCoordinates(coords)

    def _solve_with_ksp(
        self,
        *,
        ksp: PETSc.KSP,
        rhs: PETSc.Vec,
        vec_guess: PETSc.Vec,
        label: str,
        pc_type: str,
        op_mat: PETSc.Mat | None = None,
        pc_mat: PETSc.Mat | None = None,
        capture_history: bool = False,
    ) -> dict[str, object]:
        sol = rhs.duplicate()
        op = self.A if op_mat is None else op_mat
        if pc_mat is None:
            ksp.setOperators(op)
        else:
            ksp.setOperators(op, pc_mat)
        self._apply_pc_coordinates(ksp, pc_type=pc_type)
        vec_guess.copy(sol)
        ksp.setInitialGuessNonzero(True)
        residual_history: list[float] = []
        if capture_history:
            def _monitor(_ksp, _its, rnorm):
                residual_history.append(float(rnorm))
            ksp.setMonitor(_monitor)
        t0 = time.perf_counter()
        ksp.solve(rhs, sol)
        solve_time = time.perf_counter() - t0
        if capture_history:
            try:
                ksp.cancelMonitor()
            except Exception:
                pass
        its = int(ksp.getIterationNumber())
        reason_code = int(ksp.getConvergedReason())
        reason = ksp_reason_name(reason_code)
        return {
            "x": sol,
            "ksp_its": its,
            "reason_code": reason_code,
            "reason": reason,
            "solver_label": str(label),
            "solve_time": float(solve_time),
            "residual_history": residual_history,
        }


    def _potential_energy(self, vec: PETSc.Vec) -> float:
        u_owned = np.asarray(vec.array[:], dtype=np.float64)
        u_local, _ = self.partition.vector_layout.build_v_local_p2p(u_owned)
        elastic_local = float(
            self._energy_jit(jnp.asarray(u_local, dtype=jnp.float64), self._material_scale_local).block_until_ready()
        )
        elastic = float(self.comm.allreduce(elastic_local, op=MPI.SUM))
        work_local = float(np.dot(self.partition.force_owned, u_owned))
        work = float(self.comm.allreduce(work_local, op=MPI.SUM))
        return elastic - work

    def _accept_maxit_via_line_search(
        self,
        *,
        rhs: PETSc.Vec,
        vec_guess: PETSc.Vec,
        candidate: PETSc.Vec,
        raw_reason: str,
        label: str,
    ) -> dict[str, object] | None:
        phi0 = self._potential_energy(vec_guess)
        guess_arr = np.asarray(vec_guess.array[:], dtype=np.float64)
        cand_arr = np.asarray(candidate.array[:], dtype=np.float64)
        direction = cand_arr - guess_arr
        if not np.any(np.isfinite(direction)) or float(np.linalg.norm(direction)) <= 1e-16:
            return None

        trial = vec_guess.duplicate()

        def _phi(alpha: float) -> float:
            arr = trial.getArray(readonly=False)
            arr[:] = guess_arr + float(alpha) * direction
            del arr
            trial.assemble()
            return self._potential_energy(trial)

        alpha_opt, phi_opt = _golden_section_scalar(_phi, -1.0, 1.0, tol=1e-2)
        candidates = [(0.0, phi0), (-1.0, _phi(-1.0)), (1.0, _phi(1.0)), (alpha_opt, phi_opt)]
        alpha_best, phi_best = min(candidates, key=lambda item: item[1])
        if not np.isfinite(phi_best) or not (phi_best < phi0 - 1e-12):
            trial.destroy()
            return None

        arr = trial.getArray(readonly=False)
        arr[:] = guess_arr + float(alpha_best) * direction
        del arr
        trial.assemble()
        candidate.destroy()
        return {
            "x": trial,
            "ksp_its": 0,
            "reason_code": 1,
            "reason": f"ACCEPTED_MAX_IT_LINESEARCH(raw={raw_reason}, alpha={alpha_best:.4f})",
            "solver_label": f"{label}+energy_ls",
            "solve_time": 0.0,
            "line_search_alpha": float(alpha_best),
            "potential_energy_before": float(phi0),
            "potential_energy_after": float(phi_best),
        }

    def solve(
        self,
        vec_guess: PETSc.Vec,
        *,
        capture_residual_history: bool = False,
    ) -> dict[str, object]:
        assembly = self.assemble_matrix(vec_guess)
        rhs = self.part.create_vec(0.0)
        rhs_arr = rhs.getArray(readonly=False)
        rhs_arr[:] = self.partition.force_owned
        del rhs_arr
        rhs.assemble()
        attempts: list[dict[str, object]] = []
        result = self._solve_with_ksp(
            ksp=self.ksp,
            rhs=rhs,
            vec_guess=vec_guess,
            label=f"{self._primary_ksp_type}+{self._primary_pc_type}",
            pc_type=self._primary_pc_type,
            op_mat=self.A,
            pc_mat=self.A,
            capture_history=capture_residual_history,
        )
        attempts.append(
            {
                "solver_label": str(result["solver_label"]),
                "reason_code": int(result["reason_code"]),
                "reason": str(result["reason"]),
                "ksp_its": int(result["ksp_its"]),
                "solve_time": float(result["solve_time"]),
                "residual_history": list(result.get("residual_history", [])),
            }
        )

        if int(result["reason_code"]) <= 0 and str(result["reason"]) == "DIVERGED_MAX_IT":
            accepted = self._accept_maxit_via_line_search(
                rhs=rhs,
                vec_guess=vec_guess,
                candidate=result["x"],
                raw_reason=str(result["reason"]),
                label=str(result["solver_label"]),
            )
            if accepted is not None:
                attempts.append(
                    {
                        "solver_label": str(accepted["solver_label"]),
                        "reason_code": int(accepted["reason_code"]),
                        "reason": str(accepted["reason"]),
                        "ksp_its": int(accepted["ksp_its"]),
                        "solve_time": float(accepted["solve_time"]),
                    }
                )
                result = accepted

        if int(result["reason_code"]) <= 0:
            reason_msg = str(result["reason"])
            needs_shift = "INDEFINITE_PC" in reason_msg or "BREAKDOWN" in reason_msg
            if needs_shift:
                result["x"].destroy()
                base_shift = max(1e-16, 0.01 * float(self._latest_diag_abs_max))
                shift_tries = 4
                for idx in range(shift_tries):
                    rho = base_shift * (10.0 ** idx)
                    self._assemble_shifted_pc(rho)
                    fallback_label = f"fgmres+gamg_shifted_rho{rho:.3e}"
                    fallback_result = self._solve_with_ksp(
                        ksp=self.ksp,
                        rhs=rhs,
                        vec_guess=vec_guess,
                        label=fallback_label,
                        pc_type="gamg",
                        op_mat=self.A,
                        pc_mat=self.P,
                        capture_history=capture_residual_history,
                    )
                    attempts.append(
                        {
                            "solver_label": str(fallback_result["solver_label"]),
                            "reason_code": int(fallback_result["reason_code"]),
                            "reason": str(fallback_result["reason"]),
                            "ksp_its": int(fallback_result["ksp_its"]),
                            "solve_time": float(fallback_result["solve_time"]),
                            "residual_history": list(fallback_result.get("residual_history", [])),
                        }
                    )
                    result = fallback_result
                    if int(result["reason_code"]) > 0:
                        break
                    if str(result["reason"]) == "DIVERGED_MAX_IT":
                        accepted = self._accept_maxit_via_line_search(
                            rhs=rhs,
                            vec_guess=vec_guess,
                            candidate=result["x"],
                            raw_reason=str(result["reason"]),
                            label=str(result["solver_label"]),
                        )
                        if accepted is not None:
                            attempts.append(
                                {
                                    "solver_label": str(accepted["solver_label"]),
                                    "reason_code": int(accepted["reason_code"]),
                                    "reason": str(accepted["reason"]),
                                    "ksp_its": int(accepted["ksp_its"]),
                                    "solve_time": float(accepted["solve_time"]),
                                }
                            )
                            result = accepted
                            break
                    if idx < shift_tries - 1:
                        result["x"].destroy()

        sol = result["x"]
        solve_time = float(sum(float(entry["solve_time"]) for entry in attempts))
        its = int(sum(int(entry["ksp_its"]) for entry in attempts))
        reason_code = int(result["reason_code"])
        reason = str(result["reason"])
        rhs.destroy()
        return {
            "x": sol,
            "ksp_its": its,
            "reason_code": reason_code,
            "reason": reason,
            "solver_label": str(result["solver_label"]),
            "attempts": attempts,
            "build_v_local_time": float(assembly["build_v_local"]),
            "elem_hessian_time": float(assembly["elem_hessian_compute"]),
            "scatter_time": float(assembly["scatter"]),
            "coo_assembly_time": float(assembly["coo_assembly"]),
            "assemble_time": float(assembly["total"]),
            "solve_time": float(solve_time),
            "time": float(assembly["total"] + solve_time),
        }

    def compliance(self, vec: PETSc.Vec) -> float:
        local = float(np.dot(self.partition.force_owned, np.asarray(vec.array[:], dtype=np.float64)))
        return float(self.comm.allreduce(local, op=MPI.SUM))

    def cleanup(self) -> None:
        self.ksp.destroy()
        self.A.destroy()
        self.P.destroy()
        if self._nullspace is not None:
            self._nullspace.destroy()


class TopologyDesignEvaluator:
    def __init__(
        self,
        *,
        partition: StructuredTopologyPartition,
        constitutive: np.ndarray,
        alpha_reg: float,
        ell_pf: float,
        mu_move: float,
        comm: MPI.Comm,
    ) -> None:
        self.partition = partition
        self.comm = comm
        self.rank = comm.Get_rank()
        self.theta_min = float(partition.theta_min)
        self.alpha_reg = float(alpha_reg)
        self.ell_pf = float(ell_pf)
        self.mu_move = float(mu_move)
        self.constitutive = jnp.asarray(constitutive, dtype=jnp.float64)
        self._lambda_volume = 0.0
        self._p_penal = 1.0
        self._e_frozen_local = jnp.ones(partition.elem_area.shape[0], dtype=jnp.float64)
        self._z_old_local = jnp.asarray(partition.scalar_layout.v_template, dtype=jnp.float64)
        self._theta_elem_local_np = np.ones(partition.elem_area.shape[0], dtype=np.float64)
        self._energy_jit, self._grad_jit, self._frozen_jit = self._make_local_kernels()
        self._warmup()

    @property
    def part(self):
        return self.partition.scalar_layout

    def _make_local_kernels(self):
        elems = jnp.asarray(self.partition.scalar_elems_local, dtype=jnp.int32)
        vector_elems = jnp.asarray(self.partition.vector_elems_local, dtype=jnp.int32)
        elem_grad_phi = jnp.asarray(self.partition.elem_grad_phi, dtype=jnp.float64)
        elem_area = jnp.asarray(self.partition.elem_area, dtype=jnp.float64)
        energy_weights = jnp.asarray(self.partition.design_energy_weights, dtype=jnp.float64)
        owned_cell_weights = jnp.asarray(self.partition.owned_cell_mask.astype(np.float64), dtype=jnp.float64)
        constitutive = self.constitutive
        theta_min = self.theta_min
        alpha_reg = self.alpha_reg
        ell_pf = self.ell_pf
        mu_move = self.mu_move

        def local_energy(v_local, e_frozen_local, z_old_local, lambda_volume, p_penal):
            theta_full = theta_from_latent(v_local, theta_min)
            theta_elem = theta_full[elems]
            theta_centroid = jnp.mean(theta_elem, axis=1)
            grad_theta = jnp.einsum("eia,ei->ea", elem_grad_phi, theta_elem)
            z_elem = v_local[elems]
            z_old_elem = z_old_local[elems]
            z_delta_centroid = jnp.mean(z_elem - z_old_elem, axis=1)

            double_well = theta_centroid**2 * (1.0 - theta_centroid) ** 2
            reg_density = 0.5 * ell_pf * jnp.sum(grad_theta * grad_theta, axis=1) + double_well / ell_pf
            proximal_density = 0.5 * mu_move * z_delta_centroid**2
            design_density = e_frozen_local * theta_centroid ** (-p_penal) + lambda_volume * theta_centroid
            return jnp.sum(elem_area * (design_density + alpha_reg * reg_density + proximal_density) * energy_weights)

        @jax.jit
        def energy_fn(v_local, e_frozen_local, z_old_local, lambda_volume, p_penal):
            return local_energy(v_local, e_frozen_local, z_old_local, lambda_volume, p_penal)

        @jax.jit
        def grad_fn(v_local, e_frozen_local, z_old_local, lambda_volume, p_penal):
            return jax.grad(local_energy, argnums=0)(v_local, e_frozen_local, z_old_local, lambda_volume, p_penal)

        @jax.jit
        def frozen_fn(u_local, z_local, p_penal):
            theta_full = theta_from_latent(z_local, theta_min)
            theta_elem = jnp.mean(theta_full[elems], axis=1)
            strain = jnp.einsum("eij,ej->ei", jnp.asarray(self.partition.elem_B, dtype=jnp.float64), u_local[vector_elems])
            eps_ceps = jnp.einsum("ei,ij,ej->e", strain, constitutive, strain)
            e_frozen = (theta_elem**p_penal) ** 2 * eps_ceps
            volume_local = jnp.sum(elem_area * theta_elem * owned_cell_weights) / self.partition.domain_area
            return e_frozen, theta_elem, volume_local

        return energy_fn, grad_fn, frozen_fn

    def _warmup(self) -> None:
        z_local = np.asarray(self.partition.scalar_layout.v_template, dtype=np.float64)
        u_local = np.zeros(self.partition.vector_layout.n_local, dtype=np.float64)
        _ = self._energy_jit(
            jnp.asarray(z_local, dtype=jnp.float64),
            self._e_frozen_local,
            self._z_old_local,
            self._lambda_volume,
            self._p_penal,
        ).block_until_ready()
        _ = self._grad_jit(
            jnp.asarray(z_local, dtype=jnp.float64),
            self._e_frozen_local,
            self._z_old_local,
            self._lambda_volume,
            self._p_penal,
        ).block_until_ready()
        frozen = self._frozen_jit(
            jnp.asarray(u_local, dtype=jnp.float64),
            jnp.asarray(z_local, dtype=jnp.float64),
            self._p_penal,
        )
        _ = frozen[0].block_until_ready()

    def create_vec(self, latent_free: float) -> PETSc.Vec:
        v = self.part.create_vec(float(latent_free))
        return v

    def _z_local_from_owned(self, z_owned: np.ndarray) -> np.ndarray:
        z_local, _ = self.part.build_v_local_p2p(np.asarray(z_owned, dtype=np.float64))
        return z_local

    def update_state_from_current(
        self,
        *,
        z_vec: PETSc.Vec,
        u_vec: PETSc.Vec,
        lambda_volume: float,
        p_penal: float,
    ) -> None:
        z_owned = np.asarray(z_vec.array[:], dtype=np.float64)
        u_owned = np.asarray(u_vec.array[:], dtype=np.float64)
        z_local = self._z_local_from_owned(z_owned)
        u_local, _ = self.partition.vector_layout.build_v_local_p2p(u_owned)
        e_frozen_local, theta_elem_local, _ = self._frozen_jit(
            jnp.asarray(u_local, dtype=jnp.float64),
            jnp.asarray(z_local, dtype=jnp.float64),
            float(p_penal),
        )
        self._e_frozen_local = jnp.asarray(np.asarray(e_frozen_local), dtype=jnp.float64)
        self._theta_elem_local_np = np.asarray(theta_elem_local, dtype=np.float64)
        self._z_old_local = jnp.asarray(z_local, dtype=jnp.float64)
        self._lambda_volume = float(lambda_volume)
        self._p_penal = float(p_penal)

    def energy_fn(self, vec: PETSc.Vec) -> float:
        z_owned = np.asarray(vec.array[:], dtype=np.float64)
        z_local = self._z_local_from_owned(z_owned)
        local = float(
            self._energy_jit(
                jnp.asarray(z_local, dtype=jnp.float64),
                self._e_frozen_local,
                self._z_old_local,
                self._lambda_volume,
                self._p_penal,
            ).block_until_ready()
        )
        total = float(self.comm.allreduce(local, op=MPI.SUM))
        return total

    def gradient_fn(self, vec: PETSc.Vec, g: PETSc.Vec) -> None:
        z_owned = np.asarray(vec.array[:], dtype=np.float64)
        z_local = self._z_local_from_owned(z_owned)
        grad_local = np.asarray(
            self._grad_jit(
                jnp.asarray(z_local, dtype=jnp.float64),
                self._e_frozen_local,
                self._z_old_local,
                self._lambda_volume,
                self._p_penal,
            ).block_until_ready()
        )
        g_arr = g.getArray(readonly=False)
        g_arr[:] = grad_local[self.part.owned_local_indices]
        del g_arr

    def volume_fraction(self, vec: PETSc.Vec) -> float:
        z_owned = np.asarray(vec.array[:], dtype=np.float64)
        z_local, _ = self.partition.scalar_layout.build_v_local_p2p(z_owned)
        theta_local = np.asarray(
            theta_from_latent(jnp.asarray(z_local, dtype=jnp.float64), self.theta_min),
            dtype=np.float64,
        )
        theta_elem = theta_local[self.partition.scalar_elems_local].mean(axis=1)
        volume_local = float(
            np.sum(
                self.partition.elem_area[self.partition.owned_cell_mask]
                * theta_elem[self.partition.owned_cell_mask]
            )
        )
        return float(self.comm.allreduce(volume_local, op=MPI.SUM))

    def theta_owned(self, vec: PETSc.Vec) -> np.ndarray:
        z_owned = np.asarray(vec.array[:], dtype=np.float64)
        return np.asarray(theta_from_latent(jnp.asarray(z_owned, dtype=jnp.float64), self.theta_min), dtype=np.float64)

    def sensitivity_scale_owned_cells(self) -> np.ndarray:
        if not self.partition.owned_cell_mask.size:
            return np.zeros(0, dtype=np.float64)
        scale = self._p_penal * np.asarray(self._e_frozen_local, dtype=np.float64) / np.maximum(
            self._theta_elem_local_np, self.theta_min
        ) ** (self._p_penal + 1.0)
        return scale[self.partition.owned_cell_mask]


def gather_quantile(local_values: np.ndarray, q: float, comm: MPI.Comm) -> float:
    rank = comm.Get_rank()
    counts = comm.gather(int(local_values.size), root=0)
    result = None
    if rank == 0:
        displs = np.zeros(len(counts), dtype=np.int64)
        if len(displs) > 1:
            displs[1:] = np.cumsum(np.asarray(counts[:-1], dtype=np.int64))
        gathered = np.empty(int(np.sum(counts)), dtype=np.float64)
    else:
        displs = None
        gathered = None
    comm.Gatherv(
        np.asarray(local_values, dtype=np.float64),
        [gathered, counts, displs, MPI.DOUBLE] if rank == 0 else None,
        root=0,
    )
    if rank == 0:
        result = float(np.quantile(gathered, q)) if gathered.size else 0.0
    return float(comm.bcast(result, root=0))


def distributed_relative_change(current_owned: np.ndarray, previous_owned: np.ndarray | None, comm: MPI.Comm) -> float:
    if previous_owned is None:
        return np.inf
    local_num = float(np.dot(current_owned - previous_owned, current_owned - previous_owned))
    local_den = float(np.dot(previous_owned, previous_owned))
    num = float(comm.allreduce(local_num, op=MPI.SUM))
    den = float(comm.allreduce(local_den, op=MPI.SUM))
    return float(np.sqrt(num) / max(1.0, np.sqrt(den)))


def create_initial_design_vec(
    *,
    partition: StructuredTopologyPartition,
    target_volume_fraction: float,
    theta_min: float,
    solid_latent: float,
) -> tuple[PETSc.Vec, float]:
    latent_free = _compute_initial_latent(
        partition,
        target_volume_fraction=target_volume_fraction,
        theta_min=theta_min,
        solid_latent=solid_latent,
    )
    vec = partition.scalar_layout.create_vec(latent_free)
    return vec, latent_free
