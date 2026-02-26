#!/usr/bin/env python3
"""Detailed setup timing breakdown for original vs local-coloring assembler.

Usage:
  mpirun -np 16 python -m experiment_scripts.bench_setup_breakdown --level 9
"""
import igraph
import scipy.sparse as sp
from graph_coloring.multistart_coloring import multistart_color
from pLaplace2D_jax_petsc.dof_partition import DOFPartition
from pLaplace2D_jax_petsc.mesh import MeshpLaplace2D
import argparse
from petsc4py import PETSc
from mpi4py import MPI
import jax.numpy as jnp
import jax
import sys
import os
import time
import numpy as np

# Must set before JAX import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# --- Parse args ---
parser = argparse.ArgumentParser()
parser.add_argument("--level", type=int, default=9)
args, _ = parser.parse_known_args()

# --- 1. Mesh load ---
t0 = time.perf_counter()
mesh_obj = MeshpLaplace2D(mesh_level=args.level)
params, adjacency, u_init = mesh_obj.get_data_jax()
t_mesh = time.perf_counter() - t0

n_dofs = len(u_init)
n_elems = params["elems"].shape[0]

if rank == 0:
    print(f"{'=' * 72}")
    print(f"Setup Breakdown: level={args.level}  DOFs={n_dofs}  elems={n_elems}  np={nprocs}")
    print(f"{'=' * 72}")
    print(f"\n  Mesh load: {t_mesh:.4f}s")
    sys.stdout.flush()

# ====================================================================
# ORIGINAL assembler
# ====================================================================
if rank == 0:
    print(f"\n{'─' * 72}")
    print(f"  ORIGINAL assembler (global multistart coloring)")
    print(f"{'─' * 72}")
    sys.stdout.flush()


# Step 1: DOFPartition
comm.Barrier()
t0 = time.perf_counter()
part_orig = DOFPartition(params, comm, adjacency=adjacency, reorder=True)
t_partition_orig = time.perf_counter() - t0

if rank == 0:
    pt = part_orig.timings
    print(f"  1. DOFPartition:     {t_partition_orig:.4f}s")
    print(f"     - numpy_convert:  {pt.get('numpy_convert', 0):.4f}s")
    print(f"     - rcm_reorder:    {pt.get('rcm_reorder', 0):.4f}s")
    print(f"     - partition_build:{pt.get('partition_build', 0):.4f}s")
    print(f"     (n_local_elems={part_orig.n_local_elems}  n_local_nodes={part_orig.n_local}")
    print(f"      n_owned={part_orig.n_owned}  n_ghost={part_orig.n_ghost}  n_neighbors={part_orig.n_neighbors})")
    sys.stdout.flush()

# Step 2: Global coloring (multistart)
comm.Barrier()
t0 = time.perf_counter()
adj_rank0 = adjacency if rank == 0 else None
n_colors_orig, coloring_orig, color_info_orig = multistart_color(
    adj_rank0, comm, trials_per_rank=10,
)
t_coloring_orig = time.perf_counter() - t0

if rank == 0:
    print(f"  2. Global coloring:  {t_coloring_orig:.4f}s  →  {n_colors_orig} colors")
    ci = color_info_orig
    print(f"     - A² compute:     {ci.get('a2_time', 0):.4f}s")
    print(f"     - A² bcast:       {ci.get('bcast_time', 0):.4f}s")
    print(f"     - coloring:       {ci.get('color_time', 0):.4f}s")
    print(f"     - allreduce+bcast:{ci.get('total_time', 0) -
                                    ci.get('a2_time', 0) -
                                    ci.get('bcast_time', 0) -
                                    ci.get('color_time', 0):.4f}s")
    sys.stdout.flush()

# Step 3: Precompute indices (simulate what happens in __init__)
# We need to replicate the logic but with timing
comm.Barrier()
t0 = time.perf_counter()

freedofs_np = np.asarray(params["freedofs"], dtype=np.int64)
lo, hi = part_orig.lo, part_orig.hi
iperm = part_orig.iperm

# Broadcast A NZ
t_bcast_start = time.perf_counter()
if rank == 0:
    adj_csr = adjacency.tocsr()
    row_adj, col_adj = adj_csr.nonzero()
    row_adj = np.ascontiguousarray(row_adj, dtype=np.int64)
    col_adj = np.ascontiguousarray(col_adj, dtype=np.int64)
    nnz = np.int64(len(row_adj))
else:
    nnz = np.int64(0)
nnz = int(comm.bcast(int(nnz), root=0))
if rank != 0:
    row_adj = np.empty(nnz, dtype=np.int64)
    col_adj = np.empty(nnz, dtype=np.int64)
comm.Bcast(row_adj, root=0)
comm.Bcast(col_adj, root=0)
t_bcast_A = time.perf_counter() - t_bcast_start

# COO pattern
t_coo_start = time.perf_counter()
eff_rows = iperm[row_adj]
eff_cols = iperm[col_adj]
owned_mask = (eff_rows >= lo) & (eff_rows < hi)
coo_rows = eff_rows[owned_mask].astype(PETSc.IntType)
coo_cols = eff_cols[owned_mask].astype(PETSc.IntType)
n_owned_nnz = int(owned_mask.sum())
owned_row_orig = row_adj[owned_mask]
owned_col_orig = col_adj[owned_mask]
t_coo_pattern = time.perf_counter() - t_coo_start

# Group by color
t_group_start = time.perf_counter()
total_to_local = np.full(part_orig.n_total, -1, dtype=np.int64)
total_to_local[part_orig.local_to_total] = np.arange(part_orig.n_local, dtype=np.int64)
owned_row_total = freedofs_np[owned_row_orig]
nz_local_rows = total_to_local[owned_row_total]
owned_col_colors = coloring_orig[owned_col_orig]
color_nz = {}
for c in range(n_colors_orig):
    mask_c = owned_col_colors == c
    positions = np.where(mask_c)[0].astype(np.int64)
    local_rows = nz_local_rows[positions].astype(np.int64)
    color_nz[c] = (positions, local_rows)
t_group = time.perf_counter() - t_group_start

# Indicators
t_ind_start = time.perf_counter()
orig_to_local = total_to_local[freedofs_np]
indicators = []
for c in range(n_colors_orig):
    indicator = np.zeros(part_orig.n_local, dtype=np.float64)
    dofs_of_c = np.where(coloring_orig == c)[0]
    local_idx = orig_to_local[dofs_of_c]
    valid = local_idx >= 0
    indicator[local_idx[valid]] = 1.0
    indicators.append(jnp.array(indicator))
t_indicators = time.perf_counter() - t_ind_start

t_precompute_orig = time.perf_counter() - t0

if rank == 0:
    print(f"  3. Precompute idx:   {t_precompute_orig:.4f}s")
    print(f"     - bcast A NZ:     {t_bcast_A:.4f}s  (nnz={nnz})")
    print(f"     - COO pattern:    {t_coo_pattern:.4f}s  (owned_nnz={n_owned_nnz})")
    print(f"     - group by color: {t_group:.4f}s")
    print(f"     - indicators:     {t_indicators:.4f}s")
    sys.stdout.flush()

# Step 4: PETSc setup
comm.Barrier()
t0 = time.perf_counter()
n = part_orig.n_free
n_local_petsc = hi - lo
A_mat = PETSc.Mat().create(comm=comm)
A_mat.setType(PETSc.Mat.Type.MPIAIJ)
A_mat.setSizes(((n_local_petsc, n), (n_local_petsc, n)))
t_pre = time.perf_counter()
A_mat.setPreallocationCOO(coo_rows, coo_cols)
t_prealloc = time.perf_counter() - t_pre
ksp = PETSc.KSP().create(comm)
ksp.setType("cg")
ksp.getPC().setType("gamg")
ksp.setTolerances(rtol=1e-3)
ksp.setFromOptions()
t_petsc_orig = time.perf_counter() - t0

if rank == 0:
    print(f"  4. PETSc setup:      {t_petsc_orig:.4f}s")
    print(f"     - COO prealloc:   {t_prealloc:.4f}s")
    sys.stdout.flush()

# Cleanup
A_mat.destroy()
ksp.destroy()

# Step 5: JIT compile
comm.Barrier()
t0 = time.perf_counter()
p = part_orig.p
elems_j = jnp.array(part_orig.elems_local_np, dtype=jnp.int32)
dvx_j = jnp.array(part_orig.dvx_np, dtype=jnp.float64)
dvy_j = jnp.array(part_orig.dvy_np, dtype=jnp.float64)
vol_j = jnp.array(part_orig.vol_np, dtype=jnp.float64)
vol_w_j = jnp.array(part_orig.vol_np * part_orig.elem_weights, dtype=jnp.float64)


def energy_weighted(v_local):
    v_e = v_local[elems_j]
    Fx = jnp.sum(v_e * dvx_j, axis=1)
    Fy = jnp.sum(v_e * dvy_j, axis=1)
    return jnp.sum((1.0 / p) * (Fx**2 + Fy**2)**(p / 2.0) * vol_w_j)


def energy_full(v_local):
    v_e = v_local[elems_j]
    Fx = jnp.sum(v_e * dvx_j, axis=1)
    Fy = jnp.sum(v_e * dvy_j, axis=1)
    return jnp.sum((1.0 / p) * (Fx**2 + Fy**2)**(p / 2.0) * vol_j)


energy_jit = jax.jit(energy_weighted)
grad_jit = jax.jit(jax.grad(energy_full))


def hvp_fn(v, t):
    return jax.jvp(jax.grad(energy_full), (v,), (t,))[1]


hvp_jit = jax.jit(hvp_fn)

v_dummy = jnp.zeros(part_orig.n_local, dtype=jnp.float64)

t_jit_e = time.perf_counter()
_ = energy_jit(v_dummy).block_until_ready()
t_jit_energy = time.perf_counter() - t_jit_e

t_jit_g = time.perf_counter()
_ = grad_jit(v_dummy).block_until_ready()
t_jit_grad = time.perf_counter() - t_jit_g

t_jit_h = time.perf_counter()
_ = hvp_jit(v_dummy, v_dummy).block_until_ready()
t_jit_hvp = time.perf_counter() - t_jit_h

t_jit_orig = time.perf_counter() - t0

if rank == 0:
    print(f"  5. JIT compile:      {t_jit_orig:.4f}s")
    print(f"     - energy JIT:     {t_jit_energy:.4f}s")
    print(f"     - gradient JIT:   {t_jit_grad:.4f}s")
    print(f"     - HVP JIT:        {t_jit_hvp:.4f}s")
    print(f"     (n_local_elems={part_orig.n_local_elems}  n_local_nodes={part_orig.n_local})")
    t_total_orig = t_partition_orig + t_coloring_orig + t_precompute_orig + t_petsc_orig + t_jit_orig
    print(f"\n  TOTAL assembler:     {t_total_orig:.4f}s")
    print(f"  TOTAL setup (+ mesh):{t_mesh + t_total_orig:.4f}s")
    sys.stdout.flush()

# cleanup
del part_orig

# ====================================================================
# LOCAL COLORING assembler
# ====================================================================
if rank == 0:
    print(f"\n{'─' * 72}")
    print(f"  LOCAL COLORING assembler (igraph per-rank, vmap)")
    print(f"{'─' * 72}")
    sys.stdout.flush()

# Step 1: DOFPartition (same)
comm.Barrier()
t0 = time.perf_counter()
part_loc = DOFPartition(params, comm, adjacency=adjacency, reorder=True)
t_partition_loc = time.perf_counter() - t0

if rank == 0:
    pt = part_loc.timings
    print(f"  1. DOFPartition:     {t_partition_loc:.4f}s")
    print(f"     - numpy_convert:  {pt.get('numpy_convert', 0):.4f}s")
    print(f"     - rcm_reorder:    {pt.get('rcm_reorder', 0):.4f}s")
    print(f"     - partition_build:{pt.get('partition_build', 0):.4f}s")
    sys.stdout.flush()

# Step 2: Local coloring (igraph)

lo2, hi2 = part_loc.lo, part_loc.hi
perm2 = part_loc.perm
iperm2 = part_loc.iperm
n_free2 = part_loc.n_free

comm.Barrier()
t0 = time.perf_counter()

# 2a: Broadcast A NZ
t_sub = time.perf_counter()
if rank == 0:
    adj_csr2 = adjacency.tocsr()
    row_adj2, col_adj2 = adj_csr2.nonzero()
    row_adj2 = np.ascontiguousarray(row_adj2, dtype=np.int64)
    col_adj2 = np.ascontiguousarray(col_adj2, dtype=np.int64)
    nnz2 = np.int64(len(row_adj2))
else:
    nnz2 = np.int64(0)
nnz2 = int(comm.bcast(int(nnz2), root=0))
if rank != 0:
    row_adj2 = np.empty(nnz2, dtype=np.int64)
    col_adj2 = np.empty(nnz2, dtype=np.int64)
comm.Bcast(row_adj2, root=0)
comm.Bcast(col_adj2, root=0)
t_bcast_A2 = time.perf_counter() - t_sub

# 2b: Build A_csr
t_sub = time.perf_counter()
A_csr2 = sp.csr_matrix(
    (np.ones(nnz2, dtype=np.float64), (row_adj2, col_adj2)),
    shape=(n_free2, n_free2))
t_build_A = time.perf_counter() - t_sub

# 2c: Compute J = owned ∪ N_A(owned)
t_sub = time.perf_counter()
owned_orig2 = perm2[lo2:hi2]
slices = [A_csr2.indices[A_csr2.indptr[d]:A_csr2.indptr[d + 1]] for d in owned_orig2]
if slices:
    all_nbrs = np.unique(np.concatenate(slices))
else:
    all_nbrs = np.array([], dtype=np.int64)
J_arr = np.union1d(owned_orig2, all_nbrs).astype(np.int64)
n_J = len(J_arr)
t_compute_J = time.perf_counter() - t_sub

# 2d: Build A|_J
t_sub = time.perf_counter()
J_to_idx = np.full(n_free2, -1, dtype=np.int64)
J_to_idx[J_arr] = np.arange(n_J, dtype=np.int64)
mask_J = (J_to_idx[row_adj2] >= 0) & (J_to_idx[col_adj2] >= 0)
A_J = sp.csr_matrix(
    (np.ones(int(mask_J.sum()), dtype=np.float64),
     (J_to_idx[row_adj2[mask_J]], J_to_idx[col_adj2[mask_J]])),
    shape=(n_J, n_J))
t_build_AJ = time.perf_counter() - t_sub

# 2e: A²|_J = (A|_J)²
t_sub = time.perf_counter()
A2_J = sp.csr_matrix(A_J @ A_J)
t_A2J = time.perf_counter() - t_sub

# 2f: igraph coloring
t_sub = time.perf_counter()
A2_J_coo = A2_J.tocoo()
lo_tri = A2_J_coo.row > A2_J_coo.col
edges = np.column_stack((A2_J_coo.row[lo_tri], A2_J_coo.col[lo_tri]))
g = igraph.Graph(n_J, edges.tolist() if len(edges) > 0 else [], directed=False)
coloring_raw = g.vertex_coloring_greedy()
best_coloring = np.array(coloring_raw, dtype=np.int32).ravel()
best_nc = int(best_coloring.max() + 1) if n_J > 0 else 0
t_igraph_color = time.perf_counter() - t_sub

t_coloring_loc = time.perf_counter() - t0

if rank == 0:
    A2_J_nnz = A2_J.nnz
    print(f"  2. Local coloring:   {t_coloring_loc:.4f}s  →  {best_nc} colors")
    print(f"     - bcast A NZ:     {t_bcast_A2:.4f}s  (nnz={nnz2})")
    print(f"     - build A_csr:    {t_build_A:.4f}s")
    print(f"     - compute J:      {t_compute_J:.4f}s  (|J|={n_J})")
    print(f"     - build A|_J:     {t_build_AJ:.4f}s")
    print(f"     - (A|_J)²:        {t_A2J:.4f}s  (A²|_J nnz={A2_J_nnz})")
    print(f"     - igraph greedy:  {t_igraph_color:.4f}s")
    sys.stdout.flush()

# Gather stats from all ranks
all_nJ = comm.gather(n_J, root=0)
all_nc = comm.gather(best_nc, root=0)
all_A2_nnz = comm.gather(A2_J.nnz, root=0)
all_t_col = comm.gather(t_coloring_loc, root=0)
all_t_J = comm.gather(t_compute_J, root=0)
all_t_A2 = comm.gather(t_A2J, root=0)
all_t_ig = comm.gather(t_igraph_color, root=0)

if rank == 0:
    print(f"\n     Per-rank stats:")
    print(
        f"     {
            'rank':>4}  {
            '|J|':>7}  {
                'colors':>6}  {
                    'A²|_J nnz':>10}  {
                        't_color':>7}  {
                            't_J':>7}  {
                                't_A²':>7}  {
                                    't_igraph':>8}")
    for r in range(nprocs):
        print(
            f"     {
                r:4d}  {
                all_nJ[r]:7d}  {
                all_nc[r]:6d}  {
                    all_A2_nnz[r]:10d}  {
                        all_t_col[r]:7.4f}  {
                            all_t_J[r]:7.4f}  {
                                all_t_A2[r]:7.4f}  {
                                    all_t_ig[r]:8.4f}")
    sys.stdout.flush()

# Step 3: Precompute indices
comm.Barrier()
t0 = time.perf_counter()

# COO pattern
t_sub = time.perf_counter()
eff_rows2 = iperm2[row_adj2]
eff_cols2 = iperm2[col_adj2]
owned_mask2 = (eff_rows2 >= lo2) & (eff_rows2 < hi2)
coo_rows2 = eff_rows2[owned_mask2].astype(PETSc.IntType)
coo_cols2 = eff_cols2[owned_mask2].astype(PETSc.IntType)
n_owned_nnz2 = int(owned_mask2.sum())
owned_row_orig2 = row_adj2[owned_mask2]
owned_col_orig2 = col_adj2[owned_mask2]
t_coo_pattern2 = time.perf_counter() - t_sub

# Group by LOCAL color
t_sub = time.perf_counter()
total_to_local2 = np.full(part_loc.n_total, -1, dtype=np.int64)
total_to_local2[part_loc.local_to_total] = np.arange(part_loc.n_local, dtype=np.int64)
owned_row_total2 = freedofs_np[owned_row_orig2]
nz_local_rows2 = total_to_local2[owned_row_total2]
owned_col_J_idx = J_to_idx[owned_col_orig2]
owned_col_colors2 = best_coloring[owned_col_J_idx]
color_nz2 = {}
for c in range(best_nc):
    mask_c = owned_col_colors2 == c
    positions = np.where(mask_c)[0].astype(np.int64)
    local_rows = nz_local_rows2[positions].astype(np.int64)
    color_nz2[c] = (positions, local_rows)
t_group2 = time.perf_counter() - t_sub

# Indicators + stacking
t_sub = time.perf_counter()
orig_to_local2 = total_to_local2[freedofs_np]
indicators2 = []
for c in range(best_nc):
    indicator = np.zeros(part_loc.n_local, dtype=np.float64)
    J_dofs_c = J_arr[best_coloring == c]
    local_idx = orig_to_local2[J_dofs_c]
    valid = local_idx >= 0
    indicator[local_idx[valid]] = 1.0
    indicators2.append(jnp.array(indicator))
indicators_stacked = jnp.stack(indicators2)
t_indicators2 = time.perf_counter() - t_sub

t_precompute_loc = time.perf_counter() - t0

if rank == 0:
    print(f"  3. Precompute idx:   {t_precompute_loc:.4f}s")
    print(f"     - COO pattern:    {t_coo_pattern2:.4f}s  (owned_nnz={n_owned_nnz2})")
    print(f"     - group by color: {t_group2:.4f}s")
    print(f"     - indicators+stk: {t_indicators2:.4f}s")
    sys.stdout.flush()

# Step 4: PETSc setup (same)
comm.Barrier()
t0 = time.perf_counter()
A_mat2 = PETSc.Mat().create(comm=comm)
A_mat2.setType(PETSc.Mat.Type.MPIAIJ)
A_mat2.setSizes(((hi2 - lo2, n_free2), (hi2 - lo2, n_free2)))
t_pre2 = time.perf_counter()
A_mat2.setPreallocationCOO(coo_rows2, coo_cols2)
t_prealloc2 = time.perf_counter() - t_pre2
ksp2 = PETSc.KSP().create(comm)
ksp2.setType("cg")
ksp2.getPC().setType("gamg")
ksp2.setTolerances(rtol=1e-3)
ksp2.setFromOptions()
t_petsc_loc = time.perf_counter() - t0

if rank == 0:
    print(f"  4. PETSc setup:      {t_petsc_loc:.4f}s")
    print(f"     - COO prealloc:   {t_prealloc2:.4f}s")
    sys.stdout.flush()

A_mat2.destroy()
ksp2.destroy()

# Step 5: JIT compile (with vmap)
comm.Barrier()
t0 = time.perf_counter()

elems_j2 = jnp.array(part_loc.elems_local_np, dtype=jnp.int32)
dvx_j2 = jnp.array(part_loc.dvx_np, dtype=jnp.float64)
dvy_j2 = jnp.array(part_loc.dvy_np, dtype=jnp.float64)
vol_j2 = jnp.array(part_loc.vol_np, dtype=jnp.float64)
vol_w_j2 = jnp.array(part_loc.vol_np * part_loc.elem_weights, dtype=jnp.float64)


def energy_weighted2(v_local):
    v_e = v_local[elems_j2]
    Fx = jnp.sum(v_e * dvx_j2, axis=1)
    Fy = jnp.sum(v_e * dvy_j2, axis=1)
    return jnp.sum((1.0 / p) * (Fx**2 + Fy**2)**(p / 2.0) * vol_w_j2)


def energy_full2(v_local):
    v_e = v_local[elems_j2]
    Fx = jnp.sum(v_e * dvx_j2, axis=1)
    Fy = jnp.sum(v_e * dvy_j2, axis=1)
    return jnp.sum((1.0 / p) * (Fx**2 + Fy**2)**(p / 2.0) * vol_j2)


energy_jit2 = jax.jit(energy_weighted2)
grad_jit2 = jax.jit(jax.grad(energy_full2))


def hvp_fn2(v, t):
    return jax.jvp(jax.grad(energy_full2), (v,), (t,))[1]


hvp_jit2 = jax.jit(hvp_fn2)


def hvp_batched2(v_local, tangents):
    return jax.vmap(lambda t: hvp_fn2(v_local, t))(tangents)


hvp_batched_jit2 = jax.jit(hvp_batched2)

v_dummy2 = jnp.zeros(part_loc.n_local, dtype=jnp.float64)

t_jit_e2 = time.perf_counter()
_ = energy_jit2(v_dummy2).block_until_ready()
t_jit_energy2 = time.perf_counter() - t_jit_e2

t_jit_g2 = time.perf_counter()
_ = grad_jit2(v_dummy2).block_until_ready()
t_jit_grad2 = time.perf_counter() - t_jit_g2

t_jit_hv = time.perf_counter()
_ = hvp_jit2(v_dummy2, v_dummy2).block_until_ready()
t_jit_hvp2 = time.perf_counter() - t_jit_hv

t_jit_vmap = time.perf_counter()
dummy_tangents2 = jnp.zeros((best_nc, part_loc.n_local), dtype=jnp.float64)
_ = hvp_batched_jit2(v_dummy2, dummy_tangents2).block_until_ready()
t_jit_vmap_hvp = time.perf_counter() - t_jit_vmap

t_jit_loc = time.perf_counter() - t0

if rank == 0:
    print(f"  5. JIT compile:      {t_jit_loc:.4f}s")
    print(f"     - energy JIT:     {t_jit_energy2:.4f}s")
    print(f"     - gradient JIT:   {t_jit_grad2:.4f}s")
    print(f"     - HVP JIT:        {t_jit_hvp2:.4f}s")
    print(f"     - vmap HVP JIT:   {t_jit_vmap_hvp:.4f}s  ({best_nc} colors × {part_loc.n_local} local nodes)")
    t_total_loc = t_partition_loc + t_coloring_loc + t_precompute_loc + t_petsc_loc + t_jit_loc
    print(f"\n  TOTAL assembler:     {t_total_loc:.4f}s")
    print(f"  TOTAL setup (+ mesh):{t_mesh + t_total_loc:.4f}s")
    sys.stdout.flush()

# ====================================================================
# Summary comparison
# ====================================================================
if rank == 0:
    print(f"\n{'=' * 72}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 72}")
    t_total_orig = t_partition_orig + t_coloring_orig + t_precompute_orig + t_petsc_orig + t_jit_orig
    t_total_loc = t_partition_loc + t_coloring_loc + t_precompute_loc + t_petsc_loc + t_jit_loc

    print(f"  {'Phase':<20} {'Original':>10} {'Local':>10} {'Δ':>10}")
    print(f"  {'─' * 52}")
    print(f"  {'Mesh load':<20} {t_mesh:10.4f} {t_mesh:10.4f} {'':>10}")
    print(
        f"  {
            'DOFPartition':<20} {
            t_partition_orig:10.4f} {
                t_partition_loc:10.4f} {
                    t_partition_loc -
            t_partition_orig:+10.4f}")
    print(
        f"  {
            'Coloring':<20} {
            t_coloring_orig:10.4f} {
                t_coloring_loc:10.4f} {
                    t_coloring_loc -
            t_coloring_orig:+10.4f}")
    print(
        f"  {
            'Precompute indices':<20} {
            t_precompute_orig:10.4f} {
                t_precompute_loc:10.4f} {
                    t_precompute_loc -
            t_precompute_orig:+10.4f}")
    print(f"  {'PETSc setup':<20} {t_petsc_orig:10.4f} {t_petsc_loc:10.4f} {t_petsc_loc - t_petsc_orig:+10.4f}")
    print(f"  {'JIT compile':<20} {t_jit_orig:10.4f} {t_jit_loc:10.4f} {t_jit_loc - t_jit_orig:+10.4f}")
    print(f"  {'─' * 52}")
    print(f"  {'Assembler total':<20} {t_total_orig:10.4f} {t_total_loc:10.4f} {t_total_loc - t_total_orig:+10.4f}")
    print(f"  {'full setup (w/mesh)':<20} {t_mesh +
                                           t_total_orig:10.4f} {t_mesh +
                                                                t_total_loc:10.4f} {t_total_loc -
                                                                                    t_total_orig:+10.4f}")
    print()
    sys.stdout.flush()
