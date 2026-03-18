#!/usr/bin/env python3
"""Experimental HE element-assembly benchmark for PETSc distribution strategies."""

from __future__ import annotations

import argparse
import json
import os
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

from HyperElasticity3D_jax_petsc.parallel_hessian_dof import _he_energy_density
from HyperElasticity3D_petsc_support.mesh import MeshHyperElasticity3D
from HyperElasticity3D_petsc_support.rotate_boundary import rotate_right_face_from_reference
from tools_petsc4py.dof_partition import _rank_of_dof_vec, petsc_ownership_range


jax.config.update("jax_enable_x64", True)


def _configure_thread_env(nproc_threads: int) -> None:
    threads = max(1, int(nproc_threads))
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        "--xla_force_host_platform_device_count=1"
    )
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)


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
    block_perm = np.asarray(reverse_cuthill_mckee(block, symmetric_mode=True), dtype=np.int64)
    return _expand_block_perm(block_perm, block_size)


def _perm_block_xyz(nodes2coord: np.ndarray, freedofs: np.ndarray, block_size: int) -> np.ndarray:
    node_ids = freedofs[::block_size] // block_size
    coords = np.asarray(nodes2coord[node_ids], dtype=np.float64)
    block_perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0])).astype(np.int64)
    return _expand_block_perm(block_perm, block_size)


def _perm_block_metis(adjacency: sparse.spmatrix, block_size: int, n_parts: int) -> np.ndarray:
    import pymetis

    block = _build_block_graph(adjacency, block_size)
    _, part = pymetis.part_graph(n_parts, xadj=block.indptr, adjncy=block.indices)
    part = np.asarray(part, dtype=np.int64)
    block_ids = np.arange(block.shape[0], dtype=np.int64)
    block_perm = np.lexsort((block_ids, part)).astype(np.int64)
    return _expand_block_perm(block_perm, block_size)


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
class LocalStrategyData:
    local_elem_idx: np.ndarray
    local_total_nodes: np.ndarray
    elems_local_np: np.ndarray
    elems_reordered: np.ndarray
    local_elem_data: dict[str, np.ndarray]
    energy_weights: np.ndarray


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
    n_free = freedofs.size
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
    global_keys = np.asarray(coo_rows, dtype=np.int64) * key_base + np.asarray(coo_cols, dtype=np.int64)
    owned_keys = np.asarray(owned_rows, dtype=np.int64) * key_base + np.asarray(owned_cols, dtype=np.int64)
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


def _build_local_strategy_data(
    params: dict,
    layout: GlobalLayout,
    strategy: str,
    comm: MPI.Comm,
) -> LocalStrategyData:
    elems = np.asarray(params["elems"], dtype=np.int64)
    elem_reordered = layout.total_to_free_reord[elems]

    if strategy == "overlap_allgather":
        local_mask = np.any((elem_reordered >= layout.lo) & (elem_reordered < layout.hi), axis=1)
        local_elem_idx = np.where(local_mask)[0].astype(np.int64)
        local_energy_weights = (layout.elem_owner[local_elem_idx] == comm.rank).astype(np.float64)
    elif strategy == "nonoverlap_allreduce":
        local_elem_idx = np.where(layout.elem_owner == comm.rank)[0].astype(np.int64)
        local_energy_weights = np.ones(local_elem_idx.size, dtype=np.float64)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    local_elems_total = elems[local_elem_idx]
    local_total_nodes, inverse = np.unique(local_elems_total.ravel(), return_inverse=True)
    elems_local_np = inverse.reshape(local_elems_total.shape).astype(np.int32)

    local_elem_data = {
        key: np.asarray(params[key], dtype=np.float64)[local_elem_idx]
        for key in ("dphix", "dphiy", "dphiz", "vol")
    }

    return LocalStrategyData(
        local_elem_idx=local_elem_idx,
        local_total_nodes=np.asarray(local_total_nodes, dtype=np.int64),
        elems_local_np=elems_local_np,
        elems_reordered=np.asarray(layout.total_to_free_reord[local_elems_total], dtype=np.int64),
        local_elem_data=local_elem_data,
        energy_weights=local_energy_weights,
    )


def _make_local_element_kernels(
    local_data: LocalStrategyData,
    params: dict,
):
    elems = jnp.asarray(local_data.elems_local_np, dtype=jnp.int32)
    dphix = jnp.asarray(local_data.local_elem_data["dphix"], dtype=jnp.float64)
    dphiy = jnp.asarray(local_data.local_elem_data["dphiy"], dtype=jnp.float64)
    dphiz = jnp.asarray(local_data.local_elem_data["dphiz"], dtype=jnp.float64)
    vol = jnp.asarray(local_data.local_elem_data["vol"], dtype=jnp.float64)
    energy_weights = jnp.asarray(local_data.energy_weights, dtype=jnp.float64)

    C1 = float(params["C1"])
    D1 = float(params["D1"])

    def element_energy(v_e, dphix_e, dphiy_e, dphiz_e, vol_e):
        return _he_energy_density(v_e, dphix_e, dphiy_e, dphiz_e, C1, D1, False) * vol_e

    grad_elem = jax.vmap(jax.grad(element_energy), in_axes=(0, 0, 0, 0, 0))
    hess_elem = jax.vmap(jax.hessian(element_energy), in_axes=(0, 0, 0, 0, 0))

    @jax.jit
    def energy_fn(v_local):
        v_e = v_local[elems]
        e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0, 0))(v_e, dphix, dphiy, dphiz, vol)
        return jnp.sum(e * energy_weights)

    @jax.jit
    def elem_grad_fn(v_local):
        v_e = v_local[elems]
        return grad_elem(v_e, dphix, dphiy, dphiz, vol)

    @jax.jit
    def elem_hess_fn(v_local):
        v_e = v_local[elems]
        return hess_elem(v_e, dphix, dphiy, dphiz, vol)

    return energy_fn, elem_grad_fn, elem_hess_fn


def _allgather_full_free_vector(
    u_full_reordered: np.ndarray,
    lo: int,
    hi: int,
    comm: MPI.Comm,
) -> tuple[np.ndarray, float]:
    send = np.asarray(u_full_reordered[lo:hi], dtype=np.float64)
    sizes = np.asarray(comm.allgather(hi - lo), dtype=np.int64)
    displs = np.zeros_like(sizes)
    if len(displs) > 1:
        displs[1:] = np.cumsum(sizes[:-1])
    recv = np.empty(np.sum(sizes), dtype=np.float64)
    t0 = time.perf_counter()
    comm.Allgatherv(send, [recv, sizes, displs, MPI.DOUBLE])
    return recv, time.perf_counter() - t0


def _build_near_nullspace(layout: GlobalLayout, params: dict, comm: MPI.Comm) -> PETSc.NullSpace:
    kernel = np.asarray(params["elastic_kernel"], dtype=np.float64)
    lo, hi = layout.lo, layout.hi
    vecs = []
    for i in range(kernel.shape[1]):
        mode = kernel[:, i][layout.perm]
        v = PETSc.Vec().createMPI((hi - lo, layout.n_free), comm=comm)
        v.array[:] = mode[lo:hi]
        v.assemble()
        vecs.append(v)
    return PETSc.NullSpace().create(vectors=vecs)


def _build_gamg_coordinates(layout: GlobalLayout, params: dict) -> np.ndarray:
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    nodes2coord = np.asarray(params["nodes2coord"], dtype=np.float64)
    owned_orig_free = layout.perm[layout.lo:layout.hi]
    owned_total_dofs = freedofs[owned_orig_free]
    blocks = owned_total_dofs.reshape(-1, 3)
    node_ids = blocks[:, 0] // 3
    return np.asarray(nodes2coord[node_ids], dtype=np.float64)


def _create_matrix(layout: GlobalLayout, params: dict, comm: MPI.Comm) -> tuple[PETSc.Mat, PETSc.NullSpace]:
    A = PETSc.Mat().create(comm=comm)
    A.setType(PETSc.Mat.Type.MPIAIJ)
    A.setSizes(((layout.hi - layout.lo, layout.n_free), (layout.hi - layout.lo, layout.n_free)))
    A.setPreallocationCOO(
        layout.owned_rows.astype(PETSc.IntType),
        layout.owned_cols.astype(PETSc.IntType),
    )
    A.setBlockSize(3)
    nullspace = _build_near_nullspace(layout, params, comm)
    A.setNearNullSpace(nullspace)
    return A, nullspace


def _create_ksp(A: PETSc.Mat, params: dict, layout: GlobalLayout, args, comm: MPI.Comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setType(args.ksp_type)
    pc = ksp.getPC()
    pc.setType(args.pc_type)
    opts = PETSc.Options()
    opts["pc_gamg_threshold"] = float(args.gamg_threshold)
    opts["pc_gamg_agg_nsmooths"] = int(args.gamg_agg_nsmooths)
    ksp.setTolerances(rtol=float(args.ksp_rtol), max_it=int(args.ksp_max_it))
    ksp.setFromOptions()

    coords = _build_gamg_coordinates(layout, params)
    t0 = time.perf_counter()
    ksp.setOperators(A)
    pc.setCoordinates(coords)
    setop_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    ksp.setUp()
    pc_setup_time = time.perf_counter() - t0
    return ksp, setop_time, pc_setup_time


def _assemble_overlap_allgather(
    layout: GlobalLayout,
    local_data: LocalStrategyData,
    params: dict,
    dirichlet_full: np.ndarray,
    u_init_reordered: np.ndarray,
    energy_fn,
    elem_grad_fn,
    elem_hess_fn,
    comm: MPI.Comm,
):
    full_reordered, t_comm = _allgather_full_free_vector(u_init_reordered, layout.lo, layout.hi, comm)
    t0 = time.perf_counter()
    v_local = _local_vec_from_full(
        full_reordered,
        layout.total_to_free_reord,
        local_data.local_total_nodes,
        dirichlet_full,
    )
    t_build = time.perf_counter() - t0

    v_local_j = jnp.asarray(v_local, dtype=jnp.float64)

    t0 = time.perf_counter()
    energy_local = float(energy_fn(v_local_j).block_until_ready())
    energy = float(comm.allreduce(energy_local, op=MPI.SUM))
    t_energy = time.perf_counter() - t0

    t0 = time.perf_counter()
    elem_grad = np.asarray(elem_grad_fn(v_local_j).block_until_ready())
    grad_owned = np.zeros(layout.hi - layout.lo, dtype=np.float64)
    grad_dofs = local_data.elems_reordered.reshape(-1)
    grad_vals = elem_grad.reshape(-1)
    mask = (grad_dofs >= layout.lo) & (grad_dofs < layout.hi)
    np.add.at(grad_owned, grad_dofs[mask] - layout.lo, grad_vals[mask])
    t_grad = time.perf_counter() - t0

    t0 = time.perf_counter()
    elem_hess = np.asarray(elem_hess_fn(v_local_j).block_until_ready())
    rows = local_data.elems_reordered[:, :, None]
    cols = local_data.elems_reordered[:, None, :]
    valid = (rows >= layout.lo) & (rows < layout.hi) & (cols >= 0)
    vi = np.where(valid)
    row_vals = local_data.elems_reordered[vi[0], vi[1]]
    col_vals = local_data.elems_reordered[vi[0], vi[2]]
    keys = row_vals.astype(np.int64) * np.int64(layout.n_free) + col_vals.astype(np.int64)
    owned_vals = np.zeros(layout.owned_rows.size, dtype=np.float64)
    positions = np.fromiter((layout.owned_key_to_pos[int(k)] for k in keys), dtype=np.int64, count=len(keys))
    contrib = elem_hess[vi[0], vi[1], vi[2]]
    np.add.at(owned_vals, positions, contrib)
    t_hess = time.perf_counter() - t0

    return {
        "energy": energy,
        "grad_owned": grad_owned,
        "owned_matrix_values": owned_vals,
        "global_matrix_values": None,
        "timings": {
            "state_allgather": t_comm,
            "state_build": t_build,
            "energy": t_energy,
            "grad": t_grad,
            "hessian": t_hess,
        },
    }


def _assemble_nonoverlap_allreduce(
    layout: GlobalLayout,
    local_data: LocalStrategyData,
    params: dict,
    dirichlet_full: np.ndarray,
    u_init_reordered: np.ndarray,
    energy_fn,
    elem_grad_fn,
    elem_hess_fn,
    comm: MPI.Comm,
):
    full_reordered, t_comm = _allgather_full_free_vector(u_init_reordered, layout.lo, layout.hi, comm)
    t0 = time.perf_counter()
    v_local = _local_vec_from_full(
        full_reordered,
        layout.total_to_free_reord,
        local_data.local_total_nodes,
        dirichlet_full,
    )
    t_build = time.perf_counter() - t0

    v_local_j = jnp.asarray(v_local, dtype=jnp.float64)

    t0 = time.perf_counter()
    energy_local = float(energy_fn(v_local_j).block_until_ready())
    energy = float(comm.allreduce(energy_local, op=MPI.SUM))
    t_energy = time.perf_counter() - t0

    t0 = time.perf_counter()
    elem_grad = np.asarray(elem_grad_fn(v_local_j).block_until_ready())
    grad_partial = np.zeros(layout.n_free, dtype=np.float64)
    grad_dofs = local_data.elems_reordered.reshape(-1)
    grad_vals = elem_grad.reshape(-1)
    mask = grad_dofs >= 0
    np.add.at(grad_partial, grad_dofs[mask], grad_vals[mask])
    grad_global = np.zeros_like(grad_partial)
    comm.Allreduce(grad_partial, grad_global, op=MPI.SUM)
    grad_owned = grad_global[layout.lo:layout.hi]
    t_grad = time.perf_counter() - t0

    t0 = time.perf_counter()
    elem_hess = np.asarray(elem_hess_fn(v_local_j).block_until_ready())
    rows = local_data.elems_reordered[:, :, None]
    cols = local_data.elems_reordered[:, None, :]
    valid = (rows >= 0) & (cols >= 0)
    vi = np.where(valid)
    row_vals = local_data.elems_reordered[vi[0], vi[1]]
    col_vals = local_data.elems_reordered[vi[0], vi[2]]
    keys = row_vals.astype(np.int64) * np.int64(layout.n_free) + col_vals.astype(np.int64)
    positions = np.fromiter((layout.global_key_to_pos[int(k)] for k in keys), dtype=np.int64, count=len(keys))
    contrib = elem_hess[vi[0], vi[1], vi[2]]
    local_global_vals = np.zeros(layout.coo_rows.size, dtype=np.float64)
    np.add.at(local_global_vals, positions, contrib)
    global_vals = np.zeros_like(local_global_vals)
    comm.Allreduce(local_global_vals, global_vals, op=MPI.SUM)
    owned_vals = global_vals[layout.owned_mask]
    t_hess = time.perf_counter() - t0

    return {
        "energy": energy,
        "grad_owned": grad_owned,
        "owned_matrix_values": owned_vals,
        "global_matrix_values": global_vals,
        "timings": {
            "state_allgather": t_comm,
            "state_build": t_build,
            "energy": t_energy,
            "grad": t_grad,
            "hessian": t_hess,
        },
    }


def _run_case(args, reorder_name: str, strategy: str):
    comm = MPI.COMM_WORLD
    mesh = MeshHyperElasticity3D(args.level)
    params, adjacency, u_init = mesh.get_data()

    block_size = 3
    if reorder_name == "none":
        perm = _perm_identity(len(params["freedofs"]))
    elif reorder_name == "block_rcm":
        perm = _perm_block_rcm(adjacency, block_size)
    elif reorder_name == "block_xyz":
        perm = _perm_block_xyz(params["nodes2coord"], params["freedofs"], block_size)
    elif reorder_name == "block_metis":
        perm = _perm_block_metis(adjacency, block_size, comm.size)
    else:
        raise ValueError(f"Unknown reorder: {reorder_name}")

    layout = _build_global_layout(params, adjacency, perm, comm, block_size)
    local_data = _build_local_strategy_data(params, layout, strategy, comm)
    energy_fn, elem_grad_fn, elem_hess_fn = _make_local_element_kernels(local_data, params)

    angle = float(args.step) * (4.0 * 2.0 * np.pi / float(args.total_steps))
    dirichlet_full = rotate_right_face_from_reference(
        params["u_0_ref"],
        params["nodes2coord"],
        angle,
        params["right_nodes"],
    )
    u_init_reordered = np.asarray(u_init, dtype=np.float64)[layout.perm]

    if strategy == "overlap_allgather":
        assembled = _assemble_overlap_allgather(
            layout,
            local_data,
            params,
            dirichlet_full,
            u_init_reordered,
            energy_fn,
            elem_grad_fn,
            elem_hess_fn,
            comm,
        )
    elif strategy == "nonoverlap_allreduce":
        assembled = _assemble_nonoverlap_allreduce(
            layout,
            local_data,
            params,
            dirichlet_full,
            u_init_reordered,
            energy_fn,
            elem_grad_fn,
            elem_hess_fn,
            comm,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    A, nullspace = _create_matrix(layout, params, comm)
    t0 = time.perf_counter()
    A.setValuesCOO(assembled["owned_matrix_values"].astype(PETSc.ScalarType), addv=PETSc.InsertMode.INSERT_VALUES)
    A.assemble()
    matrix_assemble_time = time.perf_counter() - t0

    rhs = PETSc.Vec().createMPI((layout.hi - layout.lo, layout.n_free), comm=comm)
    rhs.array[:] = -assembled["grad_owned"]
    rhs.assemble()
    sol = rhs.duplicate()
    sol.set(0.0)

    ksp, setop_time, pc_setup_time = _create_ksp(A, params, layout, args, comm)
    t0 = time.perf_counter()
    ksp.solve(rhs, sol)
    solve_time = time.perf_counter() - t0

    local_stats = {
        "local_elems": int(local_data.local_elem_idx.size),
        "local_nodes": int(local_data.local_total_nodes.size),
        "owned_dofs": int(layout.hi - layout.lo),
        "owned_nnz": int(layout.owned_rows.size),
    }
    global_stats = {
        key: int(comm.allreduce(value, op=MPI.SUM))
        for key, value in local_stats.items()
        if key in {"local_elems", "local_nodes", "owned_dofs", "owned_nnz"}
    }
    global_stats["global_nnz"] = int(layout.coo_rows.size)

    energy = float(assembled["energy"])
    grad_norm_local_sq = float(np.dot(assembled["grad_owned"], assembled["grad_owned"]))
    grad_norm = float(np.sqrt(comm.allreduce(grad_norm_local_sq, op=MPI.SUM)))

    result = {
        "reorder": reorder_name,
        "strategy": strategy,
        "mesh_level": int(args.level),
        "step": int(args.step),
        "total_steps": int(args.total_steps),
        "nprocs": int(comm.size),
        "energy": energy,
        "grad_norm": grad_norm,
        "ksp_its": int(ksp.getIterationNumber()),
        "ksp_reason": int(ksp.getConvergedReason()),
        "timings": {
            **assembled["timings"],
            "matrix_setvalues_assemble": matrix_assemble_time,
            "ksp_setoperators": setop_time,
            "pc_setup": pc_setup_time,
            "ksp_solve": solve_time,
        },
        "distribution": {
            "owned_range": [int(layout.lo), int(layout.hi)],
            "local_elems_sum": global_stats["local_elems"],
            "local_nodes_sum": global_stats["local_nodes"],
            "owned_dofs_sum": global_stats["owned_dofs"],
            "owned_nnz_sum": global_stats["owned_nnz"],
            "global_nnz": global_stats["global_nnz"],
            "elem_duplication_factor": (
                float(global_stats["local_elems"]) / float(len(params["elems"]))
            ),
            "node_duplication_factor": (
                float(global_stats["local_nodes"]) / float(len(np.unique(params["elems"])))
            ),
        },
    }

    if assembled["global_matrix_values"] is not None:
        result["matrix_norms"] = {
            "global_frobenius": float(np.linalg.norm(assembled["global_matrix_values"])),
        }

    rhs.destroy()
    sol.destroy()
    ksp.destroy()
    nullspace.destroy()
    A.destroy()
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark HE element-assembly distribution experiments",
    )
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=96)
    parser.add_argument("--reorders", nargs="+", default=["none", "block_rcm", "block_xyz", "block_metis"])
    parser.add_argument("--strategies", nargs="+", default=["overlap_allgather", "nonoverlap_allreduce"])
    parser.add_argument("--ksp-type", type=str, default="gmres")
    parser.add_argument("--pc-type", type=str, default="gamg")
    parser.add_argument("--ksp-rtol", type=float, default=1e-1)
    parser.add_argument("--ksp-max-it", type=int, default=30)
    parser.add_argument("--gamg-threshold", type=float, default=0.05)
    parser.add_argument("--gamg-agg-nsmooths", type=int, default=1)
    parser.add_argument("--nproc-threads", type=int, default=1)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_thread_env(args.nproc_threads)

    comm = MPI.COMM_WORLD
    cases = []
    for reorder_name in args.reorders:
        for strategy in args.strategies:
            if comm.rank == 0 and not args.quiet:
                print(
                    f"[bench] level={args.level} step={args.step}/{args.total_steps} "
                    f"reorder={reorder_name} strategy={strategy}",
                    flush=True,
                )
            result = _run_case(args, reorder_name, strategy)
            gathered = comm.gather(result, root=0)
            if comm.rank == 0:
                cases.append(gathered[0])

    if comm.rank == 0:
        payload = {"cases": cases}

        if "overlap_allgather" in args.strategies and "nonoverlap_allreduce" in args.strategies:
            comparisons = []
            by_reorder = {case["reorder"]: case for case in cases if case["strategy"] == "overlap_allgather"}
            for case in cases:
                if case["strategy"] != "nonoverlap_allreduce":
                    continue
                other = by_reorder.get(case["reorder"])
                if other is None:
                    continue
                comparisons.append(
                    {
                        "reorder": case["reorder"],
                        "energy_abs_diff": abs(case["energy"] - other["energy"]),
                        "grad_norm_abs_diff": abs(case["grad_norm"] - other["grad_norm"]),
                    }
                )
            payload["cross_strategy_comparison"] = comparisons

        text = json.dumps(payload, indent=2)
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    main()
