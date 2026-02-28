#!/usr/bin/env python3
"""HyperElasticity3D matrix-layout experiment.

Builds a FEniCS-partitioned Hessian matrix (DOLFINx create_matrix + assemble_matrix),
then overwrites the free-free values with JAX SFD values mapped into FEniCS global DOF
numbering. Runs identical PETSc KSP solves to isolate matrix-layout effects and records
per-rank JAX HVP compute timings.

Run example:
  mpirun -np 16 python3 experiment_scripts/bench_he_fenics_partition_jax_values.py \
      --level 3 --step 1 --total_steps 24 \
      --out tmp_work/he_fenics_partition_jax_values_l3_np16.json

Notes:
  - `--fenics_mesh_source h5` (default) builds DOLFINx mesh from the same HDF5
    cells/coordinates used by the JAX pipeline, so sparsity patterns are exactly
    mappable.
  - `--fenics_mesh_source create_box` mirrors the current FEniCS solver mesh
    construction, but this may be topologically different from the HDF5 mesh and
    can fail strict nonzero-pattern checks during JAX value insertion.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from dolfinx import fem, mesh
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    set_bc,
)


def _ghost_update(v: PETSc.Vec) -> None:
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def _configure_thread_env(nproc: int) -> None:
    threads = max(1, int(nproc))
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        "--xla_force_host_platform_device_count=1"
    )
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)


def build_nullspace(V: fem.FunctionSpace, A: PETSc.Mat) -> PETSc.NullSpace:
    """Build 6 rigid-body near-nullspace vectors for 3D elasticity."""
    x = V.tabulate_dof_coordinates()
    imap = V.dofmap.index_map
    x_owned = x[: imap.size_local, :]

    vecs = [A.createVecLeft() for _ in range(6)]
    for vec in vecs:
        vec.getArray()[:] = 0.0

    for i in range(3):
        vecs[i].getArray()[i::3] = 1.0

    vecs[3].getArray()[1::3] = -x_owned[:, 2]
    vecs[3].getArray()[2::3] = x_owned[:, 1]

    vecs[4].getArray()[0::3] = x_owned[:, 2]
    vecs[4].getArray()[2::3] = -x_owned[:, 0]

    vecs[5].getArray()[0::3] = -x_owned[:, 1]
    vecs[5].getArray()[1::3] = x_owned[:, 0]

    return PETSc.NullSpace().create(vectors=vecs)


@dataclass
class FenicsProblem:
    comm: MPI.Comm
    msh: mesh.Mesh
    V: fem.FunctionSpace
    u: fem.Function
    u_right: fem.Function
    bcs: list
    grad_form: fem.Form
    hessian_form: fem.Form
    x: PETSc.Vec
    rhs: PETSc.Vec
    A_fenics: PETSc.Mat
    nullspace: PETSc.NullSpace
    angle: float



def _build_fenics_problem(
    level: int,
    step: int,
    total_steps: int,
    C1: float,
    D1: float,
    mesh_source: str,
    mesh_params: dict,
) -> FenicsProblem:
    comm = MPI.COMM_WORLD

    if mesh_source == "h5":
        from basix.ufl import element

        coords = np.asarray(mesh_params["nodes2coord"], dtype=np.float64)
        all_cells = np.asarray(mesh_params["elems_scalar"], dtype=np.int64)
        n_cells = int(all_cells.shape[0])
        q, r = divmod(n_cells, comm.size)
        c_lo = comm.rank * q + min(comm.rank, r)
        c_hi = c_lo + q + (1 if comm.rank < r else 0)
        cells = all_cells[c_lo:c_hi]
        domain = ufl.Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,)))
        msh = mesh.create_mesh(comm, cells, domain, coords)
    elif mesh_source == "create_box":
        Nx = 80 * 2 ** (level - 1)
        Ny = 2 * 2 ** (level - 1)
        Nz = 2 * 2 ** (level - 1)
        msh = mesh.create_box(
            comm,
            [[0.0, -0.005, -0.005], [0.4, 0.005, 0.005]],
            [Nx, Ny, Nz],
            cell_type=mesh.CellType.tetrahedron,
        )
    else:
        raise ValueError(f"Unsupported mesh_source={mesh_source!r}")
    V = fem.functionspace(msh, ("Lagrange", 1, (3,)))

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    left_facets = mesh.locate_entities_boundary(msh, 2, left_boundary)
    left_dofs = fem.locate_dofs_topological(V, 2, left_facets)
    bc_left = fem.dirichletbc(np.zeros(3, dtype=ScalarType), left_dofs, V)

    def right_boundary(x):
        return np.isclose(x[0], 0.4)

    right_facets = mesh.locate_entities_boundary(msh, 2, right_boundary)
    right_dofs = fem.locate_dofs_topological(V, 2, right_facets)
    u_right = fem.Function(V)
    bc_right = fem.dirichletbc(u_right, right_dofs)

    bcs = [bc_left, bc_right]

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)

    d = len(u)
    I = ufl.Identity(d)
    F_def = I + ufl.grad(u)
    J_det = ufl.det(F_def)
    I1 = ufl.inner(F_def, F_def)

    W = C1 * (I1 - 3 - 2 * ufl.ln(J_det)) + D1 * (J_det - 1) ** 2
    J_energy = W * ufl.dx
    dJ = ufl.derivative(J_energy, u, v)
    ddJ = ufl.derivative(dJ, u, w)

    grad_form = fem.form(dJ)
    hessian_form = fem.form(ddJ)

    rotation_per_iter = 4.0 * 2.0 * np.pi / float(total_steps)
    angle = float(step) * rotation_per_iter

    def right_rotation(x_coords):
        vals = np.zeros_like(x_coords)
        vals[0] = 0.0
        vals[1] = (
            np.cos(angle) * x_coords[1]
            + np.sin(angle) * x_coords[2]
            - x_coords[1]
        )
        vals[2] = (
            -np.sin(angle) * x_coords[1]
            + np.cos(angle) * x_coords[2]
            - x_coords[2]
        )
        return vals

    u_right.interpolate(right_rotation)

    u.x.array[:] = 0.0
    x = u.x.petsc_vec
    set_bc(x, bcs)
    _ghost_update(x)
    x.assemble()

    A_fenics = create_matrix(hessian_form)
    A_fenics.setBlockSize(3)
    nullspace = build_nullspace(V, A_fenics)
    A_fenics.setNearNullSpace(nullspace)
    A_fenics.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)

    A_fenics.zeroEntries()
    assemble_matrix(A_fenics, hessian_form, bcs=bcs)
    A_fenics.assemble()

    rhs = x.duplicate()
    with rhs.localForm() as rhs_loc:
        rhs_loc.set(0.0)
    assemble_vector(rhs, grad_form)
    apply_lifting(rhs, [hessian_form], [bcs], x0=[x])
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(rhs, bcs, x)
    rhs.scale(-1.0)

    return FenicsProblem(
        comm=comm,
        msh=msh,
        V=V,
        u=u,
        u_right=u_right,
        bcs=bcs,
        grad_form=grad_form,
        hessian_form=hessian_form,
        x=x,
        rhs=rhs,
        A_fenics=A_fenics,
        nullspace=nullspace,
        angle=angle,
    )



def _round_coord_key(arr: np.ndarray, decimals: int) -> tuple:
    return tuple(np.round(np.asarray(arr, dtype=np.float64), decimals=decimals))



def _build_fenics_jax_dof_mapping(
    V: fem.FunctionSpace,
    jax_nodes2coord: np.ndarray,
    jax_freedofs: np.ndarray,
    jax_u0: np.ndarray,
    coord_decimals: int,
):
    """Build global scalar-DOF map between FEniCS and JAX.

    Returns
    -------
    fenics_to_jax_free : (n_scalar_fenics,) int64
        -1 for constrained/non-JAX-free scalar DOFs.
    jax_free_to_fenics : (n_free,) int64
        Global scalar DOF id in FEniCS numbering for each JAX free index.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    imap = V.dofmap.index_map
    size_local = int(imap.size_local)

    local_blocks = np.arange(size_local, dtype=np.int32)
    local_global_blocks = imap.local_to_global(local_blocks).astype(np.int64)
    local_coords = np.asarray(V.tabulate_dof_coordinates()[:size_local], dtype=np.float64)

    all_blocks = comm.gather(local_global_blocks, root=0)
    all_coords = comm.gather(local_coords, root=0)

    if rank == 0:
        n_blocks_global = int(imap.size_global)
        fenics_block_coords = np.empty((n_blocks_global, 3), dtype=np.float64)

        for blk, crd in zip(all_blocks, all_coords):
            fenics_block_coords[blk] = crd

        jax_nodes2coord = np.asarray(jax_nodes2coord, dtype=np.float64)
        key_to_jax_node = {
            _round_coord_key(jax_nodes2coord[i], coord_decimals): int(i)
            for i in range(jax_nodes2coord.shape[0])
        }

        block_to_jax_node = np.full(n_blocks_global, -1, dtype=np.int64)
        missing = []
        for g in range(n_blocks_global):
            key = _round_coord_key(fenics_block_coords[g], coord_decimals)
            node = key_to_jax_node.get(key, -1)
            if node < 0:
                missing.append((g, fenics_block_coords[g].tolist()))
            block_to_jax_node[g] = node

        if missing:
            preview = missing[:5]
            raise RuntimeError(
                f"Failed to map {len(missing)} FEniCS nodes to JAX nodes. "
                f"First mismatches: {preview}"
            )

        jax_u0 = np.asarray(jax_u0, dtype=np.float64).ravel()
        jax_freedofs = np.asarray(jax_freedofs, dtype=np.int64).ravel()
        jax_scalar_to_free = np.full(len(jax_u0), -1, dtype=np.int64)
        jax_scalar_to_free[jax_freedofs] = np.arange(len(jax_freedofs), dtype=np.int64)

        fenics_to_jax_free = np.full(n_blocks_global * 3, -1, dtype=np.int64)
        jax_free_to_fenics = np.full(len(jax_freedofs), -1, dtype=np.int64)

        for gblock in range(n_blocks_global):
            jnode = int(block_to_jax_node[gblock])
            for comp in range(3):
                fenics_scalar = 3 * gblock + comp
                jax_scalar = 3 * jnode + comp
                free_idx = int(jax_scalar_to_free[jax_scalar])
                if free_idx >= 0:
                    fenics_to_jax_free[fenics_scalar] = free_idx
                    jax_free_to_fenics[free_idx] = fenics_scalar

        if np.any(jax_free_to_fenics < 0):
            raise RuntimeError("Incomplete JAX→FEniCS free DOF mapping")

        roundtrip = fenics_to_jax_free[jax_free_to_fenics]
        expected = np.arange(len(jax_freedofs), dtype=np.int64)
        if not np.array_equal(roundtrip, expected):
            raise RuntimeError("FEniCS↔JAX mapping roundtrip check failed")

    else:
        fenics_to_jax_free = None
        jax_free_to_fenics = None

    fenics_to_jax_free = comm.bcast(fenics_to_jax_free, root=0)
    jax_free_to_fenics = comm.bcast(jax_free_to_fenics, root=0)

    return fenics_to_jax_free, jax_free_to_fenics



def _compute_jax_owned_values(assembler, u_owned_reordered: np.ndarray):
    """Compute local owned COO values using sequential local-coloring HVPs."""
    timings = {}

    t0 = time.perf_counter()
    v_local = assembler._get_v_local(np.asarray(u_owned_reordered, dtype=np.float64))
    timings["p2p_exchange"] = time.perf_counter() - t0

    owned_vals = np.zeros(assembler._n_owned_nnz, dtype=np.float64)

    t0 = time.perf_counter()
    for c in range(assembler.n_colors):
        hvp_result = np.asarray(
            assembler._hvp_jit(v_local, assembler._indicators_local[c]).block_until_ready(),
            dtype=np.float64,
        )
        positions, local_rows = assembler._color_nz[c]
        if len(positions) > 0:
            owned_vals[positions] = hvp_result[local_rows]
    timings["hvp_compute"] = time.perf_counter() - t0
    timings["n_hvps"] = int(assembler.n_colors)

    return (
        np.asarray(assembler._coo_rows, dtype=np.int64),
        np.asarray(assembler._coo_cols, dtype=np.int64),
        owned_vals,
        timings,
    )



def _zero_local_free_free_entries(
    A: PETSc.Mat,
    fenics_to_jax_free: np.ndarray,
) -> None:
    """Zero free-free entries in owned rows, keep BC rows/entries unchanged."""
    lo, hi = A.getOwnershipRange()
    ai, aj, av = A.getValuesCSR()
    av = np.asarray(av, dtype=np.float64).copy()

    row_counts = np.diff(ai)
    local_rows = np.repeat(np.arange(hi - lo, dtype=np.int64), row_counts)
    global_rows = local_rows + lo

    free_row = fenics_to_jax_free[global_rows] >= 0
    free_col = fenics_to_jax_free[np.asarray(aj, dtype=np.int64)] >= 0
    free_free_mask = free_row & free_col

    av[free_free_mask] = 0.0
    A.setValuesCSR(ai, aj, av, addv=PETSc.InsertMode.INSERT_VALUES)



def _insert_coo_grouped_by_row(
    A: PETSc.Mat,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
) -> int:
    """Insert COO entries via row-grouped MatSetValues calls."""
    if len(rows) == 0:
        return 0

    order = np.argsort(rows, kind="mergesort")
    rows_s = np.asarray(rows[order], dtype=np.int64)
    cols_s = np.asarray(cols[order], dtype=PETSc.IntType)
    vals_s = np.asarray(vals[order], dtype=PETSc.ScalarType)

    n_calls = 0
    start = 0
    n = len(rows_s)
    while start < n:
        r = int(rows_s[start])
        end = start + 1
        while end < n and rows_s[end] == r:
            end += 1
        A.setValues(r, cols_s[start:end], vals_s[start:end], addv=PETSc.InsertMode.INSERT_VALUES)
        n_calls += 1
        start = end

    return n_calls



def _get_local_owned_coords(V: fem.FunctionSpace) -> np.ndarray:
    imap = V.dofmap.index_map
    return np.asarray(V.tabulate_dof_coordinates()[: int(imap.size_local), :], dtype=np.float64)



def _solve_with_ksp(
    A: PETSc.Mat,
    rhs: PETSc.Vec,
    coords_owned: np.ndarray,
    args,
    prefix: str,
):
    """Solve A x = rhs with configured KSP and return timing/iteration info."""
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOptionsPrefix(prefix)
    ksp.setType(args.ksp_type)

    pc = ksp.getPC()
    pc.setType(args.pc_type)

    opts = PETSc.Options()
    if args.pc_type == "gamg":
        opts[f"{prefix}pc_gamg_threshold"] = float(args.gamg_threshold)
        opts[f"{prefix}pc_gamg_agg_nsmooths"] = int(args.gamg_agg_nsmooths)

    ksp.setTolerances(rtol=float(args.ksp_rtol), max_it=int(args.ksp_max_it))
    ksp.setFromOptions()

    sol = rhs.duplicate()
    sol.set(0.0)

    t0 = time.perf_counter()
    ksp.setOperators(A)
    if args.pc_type == "gamg":
        pc.setCoordinates(coords_owned)
    t_setop = time.perf_counter() - t0

    t0 = time.perf_counter()
    ksp.setUp()
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    ksp.solve(rhs, sol)
    t_solve = time.perf_counter() - t0

    result = {
        "setop_time": float(t_setop),
        "pc_setup_time": float(t_setup),
        "solve_time": float(t_solve),
        "ksp_its": int(ksp.getIterationNumber()),
        "ksp_reason": int(ksp.getConvergedReason()),
        "solution_norm": float(sol.norm(PETSc.NormType.NORM_2)),
        "residual_norm": float(ksp.getResidualNorm()),
    }

    sol.destroy()
    ksp.destroy()
    return result



def _free_free_diff_norms(
    A_ref: PETSc.Mat,
    A_test: PETSc.Mat,
    fenics_to_jax_free: np.ndarray,
):
    """Compute free-free Frobenius and max-abs norms of A_ref - A_test."""
    lo, hi = A_ref.getOwnershipRange()
    ai, aj, av_ref = A_ref.getValuesCSR()
    _, _, av_test = A_test.getValuesCSR()

    av_ref = np.asarray(av_ref, dtype=np.float64)
    av_test = np.asarray(av_test, dtype=np.float64)

    row_counts = np.diff(ai)
    local_rows = np.repeat(np.arange(hi - lo, dtype=np.int64), row_counts)
    global_rows = local_rows + lo
    global_cols = np.asarray(aj, dtype=np.int64)

    free_row = fenics_to_jax_free[global_rows] >= 0
    free_col = fenics_to_jax_free[global_cols] >= 0
    mask = free_row & free_col

    diff = av_ref - av_test
    local_sq = float(np.dot(diff[mask], diff[mask])) if np.any(mask) else 0.0
    local_max = float(np.max(np.abs(diff[mask]))) if np.any(mask) else 0.0

    comm = A_ref.comm.tompi4py()
    global_sq = comm.allreduce(local_sq, op=MPI.SUM)
    global_max = comm.allreduce(local_max, op=MPI.MAX)

    return {
        "free_free_frobenius": float(np.sqrt(global_sq)),
        "free_free_max_abs": float(global_max),
    }



def _nnz_global(A: PETSc.Mat) -> int:
    ai, _, _ = A.getValuesCSR()
    local_nnz = int(ai[-1]) if len(ai) > 0 else 0
    return int(A.comm.tompi4py().allreduce(local_nnz, op=MPI.SUM))



def main() -> None:
    parser = argparse.ArgumentParser(description="FEniCS-layout vs JAX-values HE matrix benchmark")
    parser.add_argument("--level", type=int, default=3, help="Mesh level (default: 3)")
    parser.add_argument("--step", type=int, default=1, help="Load step index (default: 1)")
    parser.add_argument("--total_steps", type=int, default=24, help="Total load steps for full rotation")
    parser.add_argument(
        "--fenics_mesh_source",
        type=str,
        default="h5",
        choices=["h5", "create_box"],
        help="FEniCS mesh source: h5 (same mesh as JAX) or create_box",
    )
    parser.add_argument("--ksp_type", type=str, default="gmres", help="KSP type")
    parser.add_argument("--pc_type", type=str, default="gamg", help="PC type")
    parser.add_argument("--ksp_rtol", type=float, default=1e-1, help="KSP relative tolerance")
    parser.add_argument("--ksp_max_it", type=int, default=30, help="KSP max iterations")
    parser.add_argument("--gamg_threshold", type=float, default=0.05, help="GAMG threshold")
    parser.add_argument("--gamg_agg_nsmooths", type=int, default=1, help="GAMG agg_nsmooths")
    parser.add_argument("--coloring_trials", type=int, default=10, help="JAX coloring trials per rank")
    parser.add_argument(
        "--coord_decimals",
        type=int,
        default=12,
        help="Coordinate rounding decimals for DOF mapping",
    )
    parser.add_argument("--nproc", type=int, default=1, help="Threads per MPI rank for JAX")
    parser.add_argument("--out", type=str, default="", help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Reduce rank-0 logs")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    _configure_thread_env(args.nproc)

    from HyperElasticity3D_jax_petsc.mesh import MeshHyperElasticity3D
    from HyperElasticity3D_jax_petsc.parallel_hessian_dof import LocalColoringAssembler
    from HyperElasticity3D_jax_petsc.rotate_boundary import rotate_right_face_from_reference

    mesh_obj = MeshHyperElasticity3D(args.level)
    params, adjacency, u_init = mesh_obj.get_data()

    C1 = float(params["C1"])
    D1 = float(params["D1"])

    if rank == 0 and not args.quiet:
        print(
            f"[exp] level={args.level} step={args.step} np={size} "
            f"ksp={args.ksp_type}/{args.pc_type} mesh_source={args.fenics_mesh_source}",
            flush=True,
        )

    problem = _build_fenics_problem(
        level=args.level,
        step=args.step,
        total_steps=args.total_steps,
        C1=C1,
        D1=D1,
        mesh_source=args.fenics_mesh_source,
        mesh_params=params,
    )

    fenics_to_jax_free, jax_free_to_fenics = _build_fenics_jax_dof_mapping(
        problem.V,
        params["nodes2coord"],
        params["freedofs"],
        params["u_0"],
        args.coord_decimals,
    )

    fenics_owned_range = tuple(map(int, problem.A_fenics.getOwnershipRange()))
    fenics_owned_ranges = comm.gather(fenics_owned_range, root=0)

    fenics_nnz_global = _nnz_global(problem.A_fenics)
    fenics_coords_owned = _get_local_owned_coords(problem.V)

    if rank == 0 and not args.quiet:
        print("[exp] Assembled baseline FEniCS matrix", flush=True)

    jax_setup_t0 = time.perf_counter()
    assembler = LocalColoringAssembler(
        params=params,
        comm=comm,
        adjacency=adjacency,
        coloring_trials_per_rank=int(args.coloring_trials),
        ksp_rtol=float(args.ksp_rtol),
        ksp_type=str(args.ksp_type),
        pc_type=str(args.pc_type),
        ksp_max_it=int(args.ksp_max_it),
        use_near_nullspace=True,
        pc_options={
            "pc_gamg_threshold": float(args.gamg_threshold),
            "pc_gamg_agg_nsmooths": int(args.gamg_agg_nsmooths),
        }
        if args.pc_type == "gamg"
        else None,
        reorder=False,
        use_abs_det=False,
        hvp_eval_mode="sequential",
    )
    jax_setup_time = time.perf_counter() - jax_setup_t0

    try:
        angle = problem.angle
        u0_step = rotate_right_face_from_reference(
            params["u_0_ref"],
            params["nodes2coord"],
            angle,
            params["right_nodes"],
        )
        assembler.update_dirichlet(u0_step)

        # LocalColoringAssembler expects this rank's owned free-DOF slice.
        u_owned = np.asarray(u_init, dtype=np.float64)[assembler.part.lo:assembler.part.hi]

        rows_jax, cols_jax, vals_jax, local_jax_timings = _compute_jax_owned_values(
            assembler, u_owned
        )

        hvp_time_local = float(local_jax_timings["hvp_compute"])
        p2p_time_local = float(local_jax_timings["p2p_exchange"])
        n_owned_nnz_local = int(len(rows_jax))

        hvp_times = comm.gather(hvp_time_local, root=0)
        p2p_times = comm.gather(p2p_time_local, root=0)
        owned_nnz = comm.gather(n_owned_nnz_local, root=0)
        jax_owned_ranges = comm.gather((int(assembler.part.lo), int(assembler.part.hi)), root=0)

        rows_f = np.asarray(jax_free_to_fenics[rows_jax], dtype=np.int64)
        cols_f = np.asarray(jax_free_to_fenics[cols_jax], dtype=np.int64)

        if np.any(rows_f < 0) or np.any(cols_f < 0):
            bad_rows = int(np.count_nonzero(rows_f < 0))
            bad_cols = int(np.count_nonzero(cols_f < 0))
            raise RuntimeError(
                f"Mapped JAX COO contains constrained DOFs (rows={bad_rows}, cols={bad_cols})"
            )

        A_jax_in_fenics = problem.A_fenics.copy()
        A_jax_in_fenics.setNearNullSpace(problem.nullspace)
        A_jax_in_fenics.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)

        t0 = time.perf_counter()
        _zero_local_free_free_entries(A_jax_in_fenics, fenics_to_jax_free)
        t_zero = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            n_row_calls = _insert_coo_grouped_by_row(A_jax_in_fenics, rows_f, cols_f, vals_jax)
        except PETSc.Error as exc:
            if args.fenics_mesh_source == "create_box":
                raise RuntimeError(
                    "JAX->FEniCS insertion hit new nonzero allocations with "
                    "--fenics_mesh_source create_box. This usually means the create_box "
                    "tetrahedralization differs from the HDF5 mesh used by JAX. "
                    "Use --fenics_mesh_source h5 for exact pattern mapping."
                ) from exc
            raise
        t_insert_local = time.perf_counter() - t0

        comm.Barrier()
        t0 = time.perf_counter()
        A_jax_in_fenics.assemble()
        t_assemble = time.perf_counter() - t0

        row_calls = comm.gather(int(n_row_calls), root=0)
        insert_local_times = comm.gather(float(t_insert_local), root=0)

        fenics_ksp = _solve_with_ksp(
            A=problem.A_fenics,
            rhs=problem.rhs,
            coords_owned=fenics_coords_owned,
            args=args,
            prefix="hefenics_",
        )

        jax_layout_ksp = _solve_with_ksp(
            A=A_jax_in_fenics,
            rhs=problem.rhs,
            coords_owned=fenics_coords_owned,
            args=args,
            prefix="hejaxmapped_",
        )

        A_diff = problem.A_fenics.copy()
        A_diff.axpy(-1.0, A_jax_in_fenics, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)

        diff_fro = float(A_diff.norm(PETSc.NormType.FROBENIUS))
        diff_inf = float(A_diff.norm(PETSc.NormType.NORM_INFINITY))
        free_norms = _free_free_diff_norms(problem.A_fenics, A_jax_in_fenics, fenics_to_jax_free)

        jax_mapped_nnz_global = _nnz_global(A_jax_in_fenics)

        if rank == 0:
            free_count = int(np.count_nonzero(fenics_to_jax_free >= 0))
            constrained_count = int(len(fenics_to_jax_free) - free_count)

            hvp_stats = {
                "min": float(np.min(hvp_times)),
                "max": float(np.max(hvp_times)),
                "mean": float(np.mean(hvp_times)),
                "std": float(np.std(hvp_times)),
            }
            p2p_stats = {
                "min": float(np.min(p2p_times)),
                "max": float(np.max(p2p_times)),
                "mean": float(np.mean(p2p_times)),
                "std": float(np.std(p2p_times)),
            }

            result = {
                "mesh_level": int(args.level),
                "step": int(args.step),
                "total_steps": int(args.total_steps),
                "angle": float(problem.angle),
                "nprocs": int(size),
                "fenics_mesh_source": str(args.fenics_mesh_source),
                "total_dofs": int(len(fenics_to_jax_free)),
                "free_dofs": int(free_count),
                "constrained_dofs": int(constrained_count),
                "linear_solver": {
                    "ksp_type": str(args.ksp_type),
                    "pc_type": str(args.pc_type),
                    "ksp_rtol": float(args.ksp_rtol),
                    "ksp_max_it": int(args.ksp_max_it),
                    "gamg_threshold": float(args.gamg_threshold),
                    "gamg_agg_nsmooths": int(args.gamg_agg_nsmooths),
                },
                "matrix_layout": {
                    "fenics_row_ownership_ranges": fenics_owned_ranges,
                    "jax_row_ownership_ranges": jax_owned_ranges,
                    "fenics_nnz_global": int(fenics_nnz_global),
                    "jax_values_in_fenics_nnz_global": int(jax_mapped_nnz_global),
                },
                "timings": {
                    "jax_setup_time": float(jax_setup_time),
                    "jax_value_compute_zero_local_free_free": float(t_zero),
                    "jax_value_insert_local_total": float(sum(insert_local_times)),
                    "jax_value_insert_local_times": [float(x) for x in insert_local_times],
                    "jax_value_insert_row_calls_per_rank": [int(x) for x in row_calls],
                    "jax_value_matrix_assemble_time": float(t_assemble),
                },
                "ksp_solve": {
                    "fenics_matrix": fenics_ksp,
                    "jax_values_in_fenics_matrix": jax_layout_ksp,
                },
                "per_rank_hvp_compute_times": [float(x) for x in hvp_times],
                "per_rank_hvp_compute_stats": hvp_stats,
                "per_rank_p2p_exchange_times": [float(x) for x in p2p_times],
                "per_rank_p2p_exchange_stats": p2p_stats,
                "per_rank_owned_jax_nnz": [int(x) for x in owned_nnz],
                "matrix_difference": {
                    "frobenius_norm": diff_fro,
                    "infinity_norm": diff_inf,
                    **free_norms,
                },
                "newton_iteration_count_comparison": {
                    "fenics": None,
                    "jax_values_in_fenics": None,
                    "note": "This experiment compares one linearized Hessian/KSP solve at fixed state; full Newton loops are not run.",
                },
            }

            out_path = args.out
            if not out_path:
                out_path = (
                    f"tmp_work/he_fenics_partition_jax_values_"
                    f"l{args.level}_s{args.step}_np{size}.json"
                )

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            if not args.quiet:
                print(json.dumps(result, indent=2))
                print(f"[exp] wrote {out_path}", flush=True)

        A_diff.destroy()
        A_jax_in_fenics.destroy()

    finally:
        assembler.cleanup()

    problem.rhs.destroy()
    problem.nullspace.destroy()
    problem.A_fenics.destroy()


if __name__ == "__main__":
    main()
