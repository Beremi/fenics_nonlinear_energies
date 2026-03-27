"""Scalar structured P1 PMG support for the arctan PETSc backend."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.petsc.dof_partition import petsc_ownership_range
from src.core.petsc.reordered_element_base import inverse_permutation, select_permutation
from src.problems.plaplace_up_arctan.support.mesh import build_problem_data


@dataclass(frozen=True)
class ScalarMGLevelSpace:
    level: int
    params: dict[str, object]
    perm: np.ndarray
    iperm: np.ndarray
    lo: int
    hi: int
    n_free: int
    total_to_free_orig: np.ndarray


@dataclass
class ScalarStructuredMGHierarchy:
    levels: list[ScalarMGLevelSpace]
    prolongations: list[PETSc.Mat]
    restrictions: list[PETSc.Mat]
    build_metadata: dict[str, float | int] | None = None

    def cleanup(self) -> None:
        for mat in self.restrictions:
            mat.destroy()
        for mat in self.prolongations:
            mat.destroy()


def _build_matrix(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    *,
    row_lo: int,
    row_hi: int,
    n_rows: int,
    col_lo: int,
    col_hi: int,
    n_cols: int,
    comm: MPI.Comm,
) -> PETSc.Mat:
    owned_mask = (rows >= int(row_lo)) & (rows < int(row_hi))
    owned_rows = np.asarray(rows[owned_mask], dtype=np.int64)
    owned_cols = np.asarray(cols[owned_mask], dtype=np.int64)
    owned_vals = np.asarray(data[owned_mask], dtype=np.float64)
    mat = PETSc.Mat().create(comm=comm)
    mat.setType(PETSc.Mat.Type.MPIAIJ)
    mat.setSizes(((int(row_hi) - int(row_lo), int(n_rows)), (int(col_hi) - int(col_lo), int(n_cols))))
    mat.setPreallocationCOO(
        owned_rows.astype(PETSc.IntType),
        owned_cols.astype(PETSc.IntType),
    )
    mat.setBlockSize(1)
    mat.setValuesCOO(
        owned_vals.astype(PETSc.ScalarType),
        addv=PETSc.InsertMode.INSERT_VALUES,
    )
    mat.assemble()
    return mat


def _create_level_template_vec(space: ScalarMGLevelSpace, comm: MPI.Comm) -> PETSc.Vec:
    return PETSc.Vec().createMPI((int(space.hi) - int(space.lo), int(space.n_free)), comm=comm)


def _structured_prolongation_entries(
    coarse: ScalarMGLevelSpace,
    fine: ScalarMGLevelSpace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse_params = coarse.params
    fine_params = fine.params
    coarse_h = float(coarse_params["h"])
    coarse_n = int(round(1.0 / coarse_h))

    coarse_freedofs = np.asarray(coarse_params["freedofs"], dtype=np.int64)
    fine_freedofs = np.asarray(fine_params["freedofs"], dtype=np.int64)
    fine_nodes = np.asarray(fine_params["nodes"], dtype=np.float64)

    n_total_coarse = int(np.asarray(coarse_params["u_0"], dtype=np.float64).size)
    coarse_full_to_free = np.full(n_total_coarse, -1, dtype=np.int64)
    coarse_full_to_free[coarse_freedofs] = np.arange(coarse_freedofs.size, dtype=np.int64)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    stride = coarse_n + 1

    for fine_free_orig, fine_full in enumerate(fine_freedofs.tolist()):
        x, y = fine_nodes[int(fine_full)]
        cell_i = int(np.clip(np.floor(x / coarse_h), 0, coarse_n - 1))
        cell_j = int(np.clip(np.floor(y / coarse_h), 0, coarse_n - 1))
        x0 = float(cell_i) * coarse_h
        y0 = float(cell_j) * coarse_h
        rx = float(np.clip((x - x0) / coarse_h, 0.0, 1.0))
        ry = float(np.clip((y - y0) / coarse_h, 0.0, 1.0))

        ll = cell_j * stride + cell_i
        lr = cell_j * stride + (cell_i + 1)
        ul = (cell_j + 1) * stride + cell_i
        ur = (cell_j + 1) * stride + (cell_i + 1)
        if rx + ry <= 1.0:
            support = (
                (ll, 1.0 - rx - ry),
                (lr, rx),
                (ul, ry),
            )
        else:
            support = (
                (ur, rx + ry - 1.0),
                (ul, 1.0 - rx),
                (lr, 1.0 - ry),
            )

        fine_reordered = int(fine.iperm[int(fine_free_orig)])
        for coarse_full, weight in support:
            if abs(float(weight)) <= 1.0e-14:
                continue
            coarse_free_orig = int(coarse_full_to_free[int(coarse_full)])
            if coarse_free_orig < 0:
                continue
            rows.append(fine_reordered)
            cols.append(int(coarse.iperm[coarse_free_orig]))
            data.append(float(weight))

    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(data, dtype=np.float64),
    )


def _build_free_reordered_prolongation(
    coarse: ScalarMGLevelSpace,
    fine: ScalarMGLevelSpace,
    comm: MPI.Comm,
) -> tuple[PETSc.Mat, PETSc.Mat]:
    rows, cols, data = _structured_prolongation_entries(coarse, fine)
    prolong = _build_matrix(
        rows,
        cols,
        data,
        row_lo=int(fine.lo),
        row_hi=int(fine.hi),
        n_rows=int(fine.n_free),
        col_lo=int(coarse.lo),
        col_hi=int(coarse.hi),
        n_cols=int(coarse.n_free),
        comm=comm,
    )
    restrict = prolong.copy()
    restrict.transpose()
    return prolong, restrict


def build_structured_scalar_pmg_hierarchy(
    *,
    finest_level: int,
    coarsest_level: int,
    p: float,
    geometry: str,
    comm: MPI.Comm,
    reorder_mode: str = "block_xyz",
) -> ScalarStructuredMGHierarchy:
    if int(coarsest_level) >= int(finest_level):
        raise ValueError("coarsest_level must be strictly smaller than finest_level")

    t0 = time.perf_counter()
    spaces: list[ScalarMGLevelSpace] = []
    for level in range(int(coarsest_level), int(finest_level) + 1):
        params, adjacency = build_problem_data(level, geometry=str(geometry), p=float(p))
        freedofs = np.asarray(params["freedofs"], dtype=np.int64)
        perm = select_permutation(
            str(reorder_mode),
            adjacency=adjacency,
            coords_all=np.asarray(params["nodes"], dtype=np.float64),
            freedofs=freedofs,
            n_parts=int(comm.size),
            block_size=1,
        )
        iperm = inverse_permutation(perm)
        lo, hi = petsc_ownership_range(int(freedofs.size), int(comm.rank), int(comm.size), block_size=1)
        total_to_free_orig = np.full(int(np.asarray(params["u_0"], dtype=np.float64).size), -1, dtype=np.int64)
        total_to_free_orig[freedofs] = np.arange(freedofs.size, dtype=np.int64)
        spaces.append(
            ScalarMGLevelSpace(
                level=int(level),
                params=params,
                perm=np.asarray(perm, dtype=np.int64),
                iperm=np.asarray(iperm, dtype=np.int64),
                lo=int(lo),
                hi=int(hi),
                n_free=int(freedofs.size),
                total_to_free_orig=total_to_free_orig,
            )
        )

    prolongations: list[PETSc.Mat] = []
    restrictions: list[PETSc.Mat] = []
    for coarse, fine in zip(spaces[:-1], spaces[1:]):
        prolong, restrict = _build_free_reordered_prolongation(coarse, fine, comm)
        prolongations.append(prolong)
        restrictions.append(restrict)

    return ScalarStructuredMGHierarchy(
        levels=spaces,
        prolongations=prolongations,
        restrictions=restrictions,
        build_metadata={
            "build_time": float(time.perf_counter() - t0),
            "coarsest_level": int(coarsest_level),
            "finest_level": int(finest_level),
            "n_levels": int(len(spaces)),
        },
    )


def configure_scalar_pmg(
    ksp: PETSc.KSP,
    hierarchy: ScalarStructuredMGHierarchy,
    *,
    coarse_ksp_type: str = "preonly",
    coarse_pc_type: str = "lu",
    smoother_ksp_type: str = "richardson",
    smoother_pc_type: str = "jacobi",
    smoother_steps: int = 2,
) -> dict[str, object]:
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.MG)
    pc.setMGLevels(len(hierarchy.levels))
    pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
    pc.setMGCycleType(PETSc.PC.MGCycleType.V)

    for level_idx, (prolong, restrict) in enumerate(zip(hierarchy.prolongations, hierarchy.restrictions), start=1):
        pc.setMGInterpolation(level_idx, prolong)
        pc.setMGRestriction(level_idx, restrict)

    finest_level_idx = len(hierarchy.levels) - 1
    for level_idx, space in enumerate(hierarchy.levels):
        if level_idx < finest_level_idx:
            pc.setMGX(level_idx, _create_level_template_vec(space, ksp.comm))
            pc.setMGRhs(level_idx, _create_level_template_vec(space, ksp.comm))
        if 0 < level_idx < finest_level_idx:
            pc.setMGR(level_idx, _create_level_template_vec(space, ksp.comm))

    for level_idx in range(1, len(hierarchy.levels)):
        for level_ksp in (
            pc.getMGSmoother(level_idx),
            pc.getMGSmootherDown(level_idx),
            pc.getMGSmootherUp(level_idx),
        ):
            level_ksp.setType(str(smoother_ksp_type))
            level_ksp.setTolerances(max_it=int(smoother_steps))
            level_ksp.getPC().setType(str(smoother_pc_type))

    coarse = pc.getMGCoarseSolve()
    coarse.setType(str(coarse_ksp_type))
    coarse.getPC().setType(str(coarse_pc_type))

    return {"level_operators": [None] * len(hierarchy.levels)}


def refresh_scalar_pmg_operators(
    ksp: PETSc.KSP,
    hierarchy: ScalarStructuredMGHierarchy,
    *,
    fine_operator: PETSc.Mat,
    cache: dict[str, object],
) -> None:
    level_operators = list(cache.setdefault("level_operators", [None] * len(hierarchy.levels)))
    level_operators[-1] = fine_operator
    current_operator = fine_operator
    for fine_idx in range(len(hierarchy.levels) - 1, 0, -1):
        coarse_idx = fine_idx - 1
        prolong = hierarchy.prolongations[coarse_idx]
        coarse_operator = current_operator.ptap(
            prolong,
            result=level_operators[coarse_idx],
        )
        level_operators[coarse_idx] = coarse_operator
        current_operator = coarse_operator
    cache["level_operators"] = level_operators

    pc = ksp.getPC()
    coarse = pc.getMGCoarseSolve()
    coarse.setOperators(level_operators[0])
    for level_idx in range(1, len(hierarchy.levels)):
        operator = level_operators[level_idx]
        for smoother in (
            pc.getMGSmoother(level_idx),
            pc.getMGSmootherDown(level_idx),
            pc.getMGSmootherUp(level_idx),
        ):
            smoother.setOperators(operator)
