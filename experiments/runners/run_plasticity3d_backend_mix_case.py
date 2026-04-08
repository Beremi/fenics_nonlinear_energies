#!/usr/bin/env python3
"""Run one Plasticity3D backend-mix comparison case."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.petsc.minimizers import newton as local_newton
from src.core.petsc.reasons import ksp_reason_name
from src.problems.slope_stability_3d.jax_petsc.reordered_element_assembler import (
    SlopeStability3DReorderedElementAssembler,
)
from src.problems.slope_stability_3d.jax_petsc.solver import (
    _apply_strength_reduction,
    _load_problem_data,
)
from src.problems.slope_stability_3d.support.mesh import (
    ownership_block_size_3d,
    select_reordered_perm_3d,
)

SOURCE_IMPORT_ERROR: Exception | None = None
try:
    from slope_stability.constitutive import ConstitutiveOperator
    from slope_stability.export import write_history_json
    from slope_stability.fem import (
        assemble_owned_elastic_rows_for_comm,
        assemble_strain_operator,
        prepare_owned_tangent_pattern,
        quadrature_volume_3d,
    )
    from slope_stability.linear import SolverFactory
    from slope_stability.mesh import (
        MaterialSpec,
        heterogenous_materials,
        load_mesh_from_file,
        reorder_mesh_nodes,
    )
    from slope_stability.nonlinear.newton import (
        _collector_delta,
        _collector_snapshot,
        _setup_linear_system,
        _solve_linear_system,
        newton as source_newton,
    )
    from slope_stability.problem_assets import load_material_rows_for_path
    from slope_stability.utils import (
        extract_submatrix_free,
        full_field_from_free_values,
        owned_block_range,
        q_to_free_indices,
        release_petsc_aij_matrix,
    )
except Exception as exc:  # pragma: no cover - exercised in real runs
    SOURCE_IMPORT_ERROR = exc

DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"


def _require_source_imports() -> None:
    if SOURCE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The slope_stability source package is not importable. "
            "Launch this script with PYTHONPATH including <source-root>/src."
        ) from SOURCE_IMPORT_ERROR


def _local_hypre_options(prefix: str) -> None:
    opts = PETSc.Options()
    opts[f"{prefix}pc_hypre_type"] = "boomeramg"
    opts[f"{prefix}pc_hypre_boomeramg_nodal_coarsen"] = 6
    opts[f"{prefix}pc_hypre_boomeramg_vec_interp_variant"] = 3
    opts[f"{prefix}pc_hypre_boomeramg_strong_threshold"] = 0.5
    opts[f"{prefix}pc_hypre_boomeramg_coarsen_type"] = "HMIS"
    opts[f"{prefix}pc_hypre_boomeramg_max_iter"] = 1
    opts[f"{prefix}pc_hypre_boomeramg_tol"] = 0.0


def _append_stage_event(
    path: Path | None,
    *,
    stage: str,
    started: float,
    **fields: object,
) -> None:
    if path is None or PETSc.COMM_WORLD.getRank() != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": str(stage),
        "elapsed_s": float(time.perf_counter() - started),
        "wall_time_unix": float(time.time()),
    }
    payload.update(fields)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _set_vec_from_global(vec: PETSc.Vec, global_arr: np.ndarray) -> None:
    ownership = vec.getOwnershipRange()
    vec.array[:] = np.asarray(global_arr, dtype=np.float64)[ownership[0] : ownership[1]]
    vec.assemble()


def _global_from_vec(vec: PETSc.Vec) -> np.ndarray:
    comm = vec.getComm().tompi4py()
    local = np.asarray(vec.array[:], dtype=np.float64).copy()
    parts = comm.allgather(local)
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts)


def _copy_mat(A: PETSc.Mat) -> PETSc.Mat:
    B = A.copy()
    if int(A.getBlockSize() or 1) > 1:
        B.setBlockSize(int(A.getBlockSize() or 1))
    return B


def _destroy_mat(A: PETSc.Mat | None) -> None:
    if A is None:
        return
    release = getattr(sys.modules.get("slope_stability.utils", None), "release_petsc_aij_matrix", None)
    if callable(release):
        try:
            release(A)
        except Exception:
            pass
    A.destroy()


@dataclass
class LocalAssemblyBackend:
    assembler: SlopeStability3DReorderedElementAssembler
    params: dict[str, object]

    def __post_init__(self) -> None:
        self.comm = self.assembler.comm
        self.layout = self.assembler.layout
        self.rhs_global = self._gather_owned(np.asarray(self.assembler._f_owned, dtype=np.float64))
        self.freedofs = np.asarray(self.params["freedofs"], dtype=np.int64)
        self.perm = np.asarray(self.layout.perm, dtype=np.int64)
        self.force = np.asarray(self.params["force"], dtype=np.float64)
        self.coords_ref = np.asarray(self.params["nodes"], dtype=np.float64)
        self._elastic_mat: PETSc.Mat | None = None
        self._owned_tangent_mat: PETSc.Mat | None = None
        self._owned_regularized_mat: PETSc.Mat | None = None

    @property
    def n_free(self) -> int:
        return int(self.layout.n_free)

    @property
    def source_q(self) -> np.ndarray:
        return np.ones((1, self.n_free), dtype=bool)

    @property
    def source_f(self) -> np.ndarray:
        return np.asarray(self.rhs_global, dtype=np.float64).reshape((1, self.n_free), order="F")

    def _gather_owned(self, owned: np.ndarray) -> np.ndarray:
        parts = self.comm.allgather(np.asarray(owned, dtype=np.float64))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    def create_vec(self, global_arr: np.ndarray | None = None) -> PETSc.Vec:
        return self.assembler.create_vec(global_arr)

    def global_from_vec(self, vec: PETSc.Vec) -> np.ndarray:
        return self._gather_owned(np.asarray(vec.array[:], dtype=np.float64))

    def vec_energy(self, vec: PETSc.Vec) -> float:
        return float(self.assembler.energy_fn(vec))

    def vec_gradient(self, vec: PETSc.Vec, g: PETSc.Vec) -> None:
        self.assembler.gradient_fn(vec, g)

    def vec_tangent(self, vec: PETSc.Vec) -> PETSc.Mat:
        local_owned = np.asarray(vec.array[:], dtype=np.float64)
        self.assembler.assemble_hessian_with_mode(local_owned, constitutive_mode="plastic")
        self._owned_tangent_mat = self.assembler.A
        return self.assembler.A

    def elastic_matrix(self) -> PETSc.Mat:
        if self._elastic_mat is not None:
            return self._elastic_mat
        zero = np.zeros(self.layout.hi - self.layout.lo, dtype=np.float64)
        self.assembler.assemble_hessian_with_mode(zero, constitutive_mode="elastic")
        self._elastic_mat = _copy_mat(self.assembler.A)
        return self._elastic_mat

    def source_u0(self) -> np.ndarray:
        return np.zeros((1, self.n_free), dtype=np.float64)

    def _u_global(self, U: np.ndarray) -> np.ndarray:
        return np.asarray(U, dtype=np.float64).reshape(-1, order="F")

    def build_F_reduced(self, U: np.ndarray) -> np.ndarray:
        u_global = self._u_global(U)
        vec = self.create_vec(u_global)
        grad = vec.duplicate()
        try:
            self.assembler.gradient_fn(vec, grad)
            total_grad = self.global_from_vec(grad)
        finally:
            grad.destroy()
            vec.destroy()
        internal = total_grad + self.rhs_global
        return internal.reshape((1, self.n_free), order="F")

    def build_F_reduced_free(self, U: np.ndarray) -> np.ndarray:
        return np.asarray(self.build_F_reduced(U), dtype=np.float64).reshape(-1, order="F")

    def build_F_K_tangent_reduced(self, U: np.ndarray):
        F = self.build_F_reduced(U)
        _F_free, K_tangent = self.build_F_K_tangent_reduced_free(U)
        return F, K_tangent

    def build_F_K_tangent_reduced_free(self, U: np.ndarray):
        u_global = self._u_global(U)
        F_free = self.build_F_reduced_free(U)
        vec = self.create_vec(u_global)
        try:
            self.assembler.assemble_hessian_with_mode(
                np.asarray(vec.array[:], dtype=np.float64),
                constitutive_mode="plastic",
            )
            self._owned_tangent_mat = self.assembler.A
        finally:
            vec.destroy()
        return F_free, self._owned_tangent_mat

    def build_K_regularized(self, r: float):
        if self._owned_tangent_mat is None:
            raise RuntimeError("Tangent matrix is not available before build_K_regularized().")
        elastic = self.elastic_matrix()
        if self._owned_regularized_mat is None:
            self._owned_regularized_mat = _copy_mat(elastic)
        self._owned_regularized_mat.zeroEntries()
        self._owned_regularized_mat.axpy(
            float(1.0 - r),
            self._owned_tangent_mat,
            structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN,
        )
        self._owned_regularized_mat.axpy(
            float(r),
            elastic,
            structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN,
        )
        self._owned_regularized_mat.assemble()
        return self._owned_regularized_mat

    def build_F_K_regularized_reduced_free(self, U: np.ndarray, r: float):
        F_free, _ = self.build_F_K_tangent_reduced_free(U)
        return F_free, self.build_K_regularized(r)

    def build_F_K_regularized_reduced(self, U: np.ndarray, r: float):
        F = self.build_F_reduced(U)
        _F_free, K_r = self.build_F_K_regularized_reduced_free(U, r)
        return F, K_r

    def energy_global(self, u_global: np.ndarray) -> float:
        vec = self.create_vec(u_global)
        try:
            return float(self.assembler.energy_fn(vec))
        finally:
            vec.destroy()

    def gradient_global(self, u_global: np.ndarray) -> np.ndarray:
        vec = self.create_vec(u_global)
        grad = vec.duplicate()
        try:
            self.assembler.gradient_fn(vec, grad)
            return self.global_from_vec(grad)
        finally:
            grad.destroy()
            vec.destroy()

    def tangent_global(self, u_global: np.ndarray) -> PETSc.Mat:
        vec = self.create_vec(u_global)
        try:
            self.assembler.assemble_hessian_with_mode(
                np.asarray(vec.array[:], dtype=np.float64),
                constitutive_mode="plastic",
            )
            self._owned_tangent_mat = self.assembler.A
            return self.assembler.A
        finally:
            vec.destroy()

    def final_observables(self, u_global: np.ndarray) -> dict[str, float]:
        full_original = np.empty_like(np.asarray(u_global, dtype=np.float64))
        full_original[self.perm] = np.asarray(u_global, dtype=np.float64)
        u_full = np.asarray(self.params["u_0"], dtype=np.float64).copy()
        u_full[self.freedofs] = full_original
        coords_final = self.coords_ref + u_full.reshape((-1, 3))
        displacement = coords_final - self.coords_ref
        return {
            "energy": float(self.energy_global(np.asarray(u_global, dtype=np.float64))),
            "omega": float(np.dot(self.force, u_full)),
            "u_max": float(np.max(np.linalg.norm(displacement, axis=1))),
        }

    def close(self) -> None:
        _destroy_mat(self._elastic_mat)
        _destroy_mat(self._owned_regularized_mat)
        self.assembler.cleanup()


@dataclass
class SourceAssemblyBackend:
    const_builder: object
    tangent_pattern: object
    coord: np.ndarray
    q_mask_actual: np.ndarray
    free_idx_actual: np.ndarray
    rhs_free: np.ndarray
    elastic_full_mat: PETSc.Mat
    elastic_free_mat: PETSc.Mat
    data_dir: Path
    has_energy_operator: bool = False

    def __post_init__(self) -> None:
        self.comm = PETSc.COMM_WORLD.tompi4py()
        self.n_free = int(self.free_idx_actual.size)
        self.source_q = np.ones((1, self.n_free), dtype=bool)
        self.source_f = np.asarray(self.rhs_free, dtype=np.float64).reshape((1, self.n_free), order="F")
        self.owned_tangent_pattern = self.tangent_pattern
        self._owned_tangent_mat: PETSc.Mat | None = None
        self._owned_regularized_mat: PETSc.Mat | None = None

    def _to_full(self, values: np.ndarray) -> np.ndarray:
        return full_field_from_free_values(
            np.asarray(values, dtype=np.float64),
            self.free_idx_actual,
            tuple(int(v) for v in self.q_mask_actual.shape),
        )

    def create_vec(self, global_arr: np.ndarray | None = None) -> PETSc.Vec:
        if global_arr is None:
            global_arr = np.zeros(self.n_free, dtype=np.float64)
        vec = self.elastic_free_mat.createVecLeft()
        _set_vec_from_global(vec, np.asarray(global_arr, dtype=np.float64))
        return vec

    def _reduce_full_matrix(self, mat_full: PETSc.Mat) -> PETSc.Mat:
        return extract_submatrix_free(mat_full, self.free_idx_actual)

    def global_from_vec(self, vec: PETSc.Vec) -> np.ndarray:
        return _global_from_vec(vec)

    def vec_energy(self, vec: PETSc.Vec) -> float:
        return float(self.energy_global(self.global_from_vec(vec)))

    def vec_gradient(self, vec: PETSc.Vec, g: PETSc.Vec) -> None:
        grad = self.gradient_global(self.global_from_vec(vec))
        ownership = g.getOwnershipRange()
        g.array[:] = grad[ownership[0] : ownership[1]]
        g.assemble()

    def vec_tangent(self, vec: PETSc.Vec) -> PETSc.Mat:
        return self.tangent_global(self.global_from_vec(vec))

    def elastic_matrix(self) -> PETSc.Mat:
        return self.elastic_free_mat

    def source_u0(self) -> np.ndarray:
        return np.zeros((1, self.n_free), dtype=np.float64)

    def build_F_reduced(self, U: np.ndarray) -> np.ndarray:
        full = self._to_full(np.asarray(U, dtype=np.float64).reshape(-1, order="F"))
        F_free = np.asarray(self.const_builder.build_F_reduced_free(full), dtype=np.float64).reshape(-1)
        return F_free.reshape((1, self.n_free), order="F")

    def build_F_reduced_free(self, U: np.ndarray) -> np.ndarray:
        full = self._to_full(np.asarray(U, dtype=np.float64).reshape(-1, order="F"))
        return np.asarray(self.const_builder.build_F_reduced_free(full), dtype=np.float64).reshape(-1)

    def build_F_K_tangent_reduced_free(self, U: np.ndarray):
        full = self._to_full(np.asarray(U, dtype=np.float64).reshape(-1, order="F"))
        F_free, K_tangent_full = self.const_builder.build_F_K_tangent_reduced_free(full)
        _destroy_mat(self._owned_tangent_mat)
        self._owned_tangent_mat = self._reduce_full_matrix(K_tangent_full)
        return np.asarray(F_free, dtype=np.float64).reshape(-1), self._owned_tangent_mat

    def build_F_K_tangent_reduced(self, U: np.ndarray):
        F_free, K_tangent = self.build_F_K_tangent_reduced_free(U)
        return np.asarray(F_free, dtype=np.float64).reshape((1, self.n_free), order="F"), K_tangent

    def build_K_regularized(self, r: float):
        K_r_full = self.const_builder.build_K_regularized(float(r))
        _destroy_mat(self._owned_regularized_mat)
        self._owned_regularized_mat = self._reduce_full_matrix(K_r_full)
        return self._owned_regularized_mat

    def build_F_K_regularized_reduced_free(self, U: np.ndarray, r: float):
        full = self._to_full(np.asarray(U, dtype=np.float64).reshape(-1, order="F"))
        F_free, K_r_full = self.const_builder.build_F_K_regularized_reduced_free(full, float(r))
        _destroy_mat(self._owned_regularized_mat)
        self._owned_regularized_mat = self._reduce_full_matrix(K_r_full)
        return np.asarray(F_free, dtype=np.float64).reshape(-1), self._owned_regularized_mat

    def energy_global(self, u_global: np.ndarray) -> float:
        if not bool(self.has_energy_operator):
            return float("nan")
        full = self._to_full(np.asarray(u_global, dtype=np.float64))
        potential = float(self.const_builder.potential_energy(full))
        return potential - float(np.dot(self.rhs_free, np.asarray(u_global, dtype=np.float64)))

    def gradient_global(self, u_global: np.ndarray) -> np.ndarray:
        full = self._to_full(np.asarray(u_global, dtype=np.float64))
        F_free = np.asarray(self.const_builder.build_F_reduced_free(full), dtype=np.float64).reshape(-1)
        return F_free - self.rhs_free

    def tangent_global(self, u_global: np.ndarray) -> PETSc.Mat:
        _F_free, K_tangent = self.build_F_K_tangent_reduced_free(
            np.asarray(u_global, dtype=np.float64)
        )
        return K_tangent

    def final_observables(self, u_global: np.ndarray) -> dict[str, float]:
        full = self._to_full(np.asarray(u_global, dtype=np.float64))
        coords_ref = np.asarray(self.coord.T, dtype=np.float64)
        coords_final = coords_ref + np.asarray(full.T, dtype=np.float64)
        displacement = coords_final - coords_ref
        return {
            "energy": float(self.energy_global(np.asarray(u_global, dtype=np.float64))),
            "omega": float(np.dot(self.rhs_free, np.asarray(u_global, dtype=np.float64))),
            "u_max": float(np.max(np.linalg.norm(displacement, axis=1))),
        }

    def close(self) -> None:
        history_path = self.data_dir / "source_builder_history.json"
        if PETSc.COMM_WORLD.getRank() == 0:
            try:
                write_history_json(history_path, self.const_builder.get_total_time())
            except Exception:
                pass
        self.const_builder.release_petsc_caches()
        _destroy_mat(self.elastic_full_mat)
        _destroy_mat(self.elastic_free_mat)


def _build_local_assembly_backend(
    *,
    autodiff_tangent_mode: str = "element",
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> LocalAssemblyBackend:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    comm = MPI.COMM_WORLD
    args = SimpleNamespace(
        mesh_name="hetero_ssr_L1",
        elem_degree=4,
        problem_build_mode="rank_local",
        element_reorder_mode="block_xyz",
        distribution_strategy="overlap_p2p",
        local_hessian_mode="element",
        autodiff_tangent_mode=str(autodiff_tangent_mode),
        reuse_hessian_value_buffers=True,
        assembly_backend="coo",
        profile="performance",
        ksp_type="fgmres",
        pc_type="hypre",
        ksp_rtol=1.0e-2,
        ksp_max_it=100,
        use_near_nullspace=False,
        jax_trace_dir="",
        enable_petsc_log_events=False,
    )
    _mesh_name, params, adjacency = _load_problem_data(args, comm)
    _append_stage_event(stage_path, stage="local_problem_loaded", started=started)
    _apply_strength_reduction(params, 1.5)
    perm_override = (
        np.asarray(params["_distributed_perm"], dtype=np.int64)
        if "_distributed_perm" in params
        else select_reordered_perm_3d(
            "block_xyz",
            adjacency=adjacency,
            coords_all=np.asarray(params["nodes"], dtype=np.float64),
            freedofs=np.asarray(params["freedofs"], dtype=np.int64),
            n_parts=int(comm.size),
        )
    )
    assembler = SlopeStability3DReorderedElementAssembler(
        params,
        comm,
        adjacency,
        ksp_rtol=1.0e-2,
        ksp_type="fgmres",
        pc_type="hypre",
        ksp_max_it=100,
        use_near_nullspace=False,
        pc_options={},
        reorder_mode="block_xyz",
        local_hessian_mode="element",
        autodiff_tangent_mode=str(autodiff_tangent_mode),
        perm_override=perm_override,
        block_size_override=ownership_block_size_3d(
            np.asarray(params["freedofs"], dtype=np.int64)
        ),
        distribution_strategy="overlap_p2p",
        reuse_hessian_value_buffers=True,
        assembly_backend="coo",
        petsc_log_events=False,
        jax_trace_dir="",
    )
    _append_stage_event(stage_path, stage="local_assembler_ready", started=started)
    return LocalAssemblyBackend(assembler=assembler, params=params)


def _build_source_assembly_backend(
    *,
    source_root: Path,
    data_dir: Path,
    need_energy_operator: bool,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> SourceAssemblyBackend:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    _require_source_imports()
    mesh_path = source_root / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
    material_rows = load_material_rows_for_path(mesh_path)
    if material_rows is None:
        raise FileNotFoundError(f"Could not load material rows for {mesh_path}")
    materials = [
        MaterialSpec(
            c0=float(row[0]),
            phi=float(row[1]),
            psi=float(row[2]),
            young=float(row[3]),
            poisson=float(row[4]),
            gamma_sat=float(row[5]),
            gamma_unsat=float(row[6]),
        )
        for row in np.asarray(material_rows, dtype=np.float64)
    ]
    mesh = load_mesh_from_file(mesh_path, boundary_type=0, elem_type="P4")
    reordered = reorder_mesh_nodes(
        mesh.coord,
        mesh.elem,
        mesh.surf,
        mesh.q_mask,
        strategy="block_xyz",
        n_parts=None,
    )
    coord = reordered.coord.astype(np.float64)
    elem = reordered.elem.astype(np.int64)
    q_mask = reordered.q_mask.astype(bool)
    material_identifier = mesh.material.astype(np.int64).ravel()
    _append_stage_event(stage_path, stage="source_mesh_ready", started=started)

    n_q = int(quadrature_volume_3d("P4")[0].shape[1])
    n_int = int(elem.shape[1] * n_q)
    c0, phi, psi, shear, bulk, lame, gamma = heterogenous_materials(
        material_identifier,
        np.ones(n_int, dtype=bool),
        n_q,
        materials,
    )

    B = None
    weight = np.zeros(n_int, dtype=np.float64)
    if bool(need_energy_operator):
        assembly = assemble_strain_operator(coord, elem, "P4", dim=3)
        B = assembly.B
        weight = np.asarray(assembly.weight, dtype=np.float64)
        del assembly

    const_builder = ConstitutiveOperator(
        B=B,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type="B",
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=weight,
        n_strain=6,
        n_int=n_int,
        dim=3,
        q_mask=q_mask,
    )
    _append_stage_event(stage_path, stage="source_constitutive_ready", started=started)
    row0, row1 = owned_block_range(coord.shape[1], coord.shape[0], PETSc.COMM_WORLD)
    pattern = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        material_identifier,
        materials,
        (row0 // coord.shape[0], row1 // coord.shape[0]),
        elem_type="P4",
        include_unique=False,
        include_legacy_scatter=False,
        include_overlap_B=False,
        elastic_rows=None,
    )
    _append_stage_event(stage_path, stage="source_tangent_pattern_ready", started=started)
    const_builder.set_owned_tangent_pattern(
        pattern,
        use_compiled=True,
        tangent_kernel="rows",
        constitutive_mode="overlap",
        use_compiled_constitutive=True,
    )
    const_builder.reduction(1.5)
    free_idx = q_to_free_indices(q_mask)

    elastic_rows = assemble_owned_elastic_rows_for_comm(
        coord,
        elem,
        q_mask,
        material_identifier,
        materials,
        PETSc.COMM_WORLD,
        elem_type="P4",
    )
    rhs_full_parts = PETSc.COMM_WORLD.tompi4py().allgather(
        np.asarray(elastic_rows.local_rhs, dtype=np.float64)
    )
    rhs_full = np.concatenate(rhs_full_parts) if rhs_full_parts else np.empty(0, dtype=np.float64)
    rhs_free = np.asarray(rhs_full[free_idx], dtype=np.float64)
    zero_full = np.zeros(tuple(int(v) for v in q_mask.shape), dtype=np.float64, order="F")
    _unused_F_free, elastic_owned_live = const_builder.build_F_K_tangent_reduced_free(zero_full)
    elastic_full = elastic_owned_live.duplicate(copy=True)
    elastic_free = extract_submatrix_free(elastic_full, free_idx)
    _append_stage_event(stage_path, stage="source_elastic_matrix_ready", started=started)
    return SourceAssemblyBackend(
        const_builder=const_builder,
        tangent_pattern=pattern,
        coord=coord,
        q_mask_actual=q_mask,
        free_idx_actual=free_idx,
        rhs_free=rhs_free,
        elastic_full_mat=elastic_full,
        elastic_free_mat=elastic_free,
        data_dir=data_dir,
        has_energy_operator=bool(need_energy_operator),
    )


def _make_local_ksp(*, prefix: str, comm: PETSc.Comm) -> PETSc.KSP:
    _local_hypre_options(prefix)
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix(prefix)
    ksp.setType("fgmres")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1.0e-2, max_it=100)
    ksp.setFromOptions()
    return ksp


def _local_initial_guess(
    backend,
    *,
    out_dir: Path,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> tuple[PETSc.Vec, dict[str, object]]:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    _append_stage_event(stage_path, stage="local_initial_guess_start", started=started)
    ksp = _make_local_ksp(prefix="mix_init_", comm=backend.elastic_matrix().getComm())
    rhs = backend.create_vec(backend.rhs_global if hasattr(backend, "rhs_global") else backend.rhs_free)
    sol = backend.create_vec()
    sol_right = None
    ksp.setOperators(backend.elastic_matrix())
    t0 = time.perf_counter()
    if isinstance(backend, SourceAssemblyBackend):
        sol_right = backend.elastic_matrix().createVecRight()
        ksp.solve(rhs, sol_right)
        _set_vec_from_global(sol, backend.global_from_vec(sol_right))
    else:
        ksp.solve(rhs, sol)
    elapsed = time.perf_counter() - t0
    meta = {
        "enabled": True,
        "success": bool(int(ksp.getConvergedReason()) > 0),
        "ksp_type": "fgmres",
        "pc_type": "hypre",
        "ksp_iterations": int(ksp.getIterationNumber()),
        "ksp_reason": str(ksp_reason_name(int(ksp.getConvergedReason()))),
        "ksp_reason_code": int(ksp.getConvergedReason()),
        "rhs_norm": float(rhs.norm(PETSc.NormType.NORM_2)),
        "residual_norm": float(ksp.getResidualNorm()),
        "solve_time": float(elapsed),
        "vector_norm": float(np.linalg.norm(backend.global_from_vec(sol))),
    }
    rhs.destroy()
    if sol_right is not None:
        sol_right.destroy()
    ksp.destroy()
    _append_stage_event(
        stage_path,
        stage="local_initial_guess_done",
        started=started,
        ksp_iterations=int(meta["ksp_iterations"]),
        success=bool(meta["success"]),
    )
    return sol, meta


def _run_local_solver_backend(
    backend,
    *,
    out_dir: Path,
    stop_tol: float,
    maxit: int,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> dict[str, object]:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    x, init_meta = _local_initial_guess(
        backend,
        out_dir=out_dir,
        stage_path=stage_path,
        stage_started=started,
    )
    linear_records: list[dict[str, object]] = []
    ksp = _make_local_ksp(prefix="mix_newton_", comm=backend.elastic_matrix().getComm())

    def hessian_solve_fn(vec, rhs, sol):
        t0 = time.perf_counter()
        A = backend.vec_tangent(vec)
        t_assemble = time.perf_counter() - t0
        t1 = time.perf_counter()
        ksp.setOperators(A)
        ksp.setUp()
        t_setup = time.perf_counter() - t1
        t2 = time.perf_counter()
        sol_right = None
        if isinstance(backend, SourceAssemblyBackend):
            sol_right = A.createVecRight()
            ksp.solve(rhs, sol_right)
            _set_vec_from_global(sol, backend.global_from_vec(sol_right))
        else:
            ksp.solve(rhs, sol)
        t_solve = time.perf_counter() - t2
        if sol_right is not None:
            sol_right.destroy()
        rec = {
            "newton_iteration": int(len(linear_records) + 1),
            "ksp_its": int(ksp.getIterationNumber()),
            "ksp_reason_code": int(ksp.getConvergedReason()),
            "ksp_reason_name": str(ksp_reason_name(int(ksp.getConvergedReason()))),
            "ksp_residual_norm": float(ksp.getResidualNorm()),
            "t_assemble": float(t_assemble),
            "t_setup": float(t_setup),
            "t_solve": float(t_solve),
        }
        linear_records.append(rec)
        _append_stage_event(
            stage_path,
            stage="local_linear_iteration_done",
            started=started,
            newton_iteration=int(rec["newton_iteration"]),
            ksp_iterations=int(rec["ksp_its"]),
            ksp_reason=str(rec["ksp_reason_name"]),
        )
        return int(rec["ksp_its"])

    solve_t0 = time.perf_counter()
    _append_stage_event(stage_path, stage="local_newton_start", started=started)
    def _energy_placeholder(_vec: PETSc.Vec) -> float:
        return 0.0

    result = local_newton(
        energy_fn=_energy_placeholder,
        gradient_fn=backend.vec_gradient,
        hessian_solve_fn=hessian_solve_fn,
        x=x,
        tolf=1.0e100,
        tolg=1.0e100,
        tolg_rel=0.0,
        line_search="residual_bisection",
        armijo_alpha0=1.0,
        armijo_c1=1.0e-4,
        armijo_shrink=0.5,
        armijo_max_ls=40,
        maxit=int(maxit),
        tolx_rel=float(stop_tol),
        tolx_abs=0.0,
        require_all_convergence=True,
        fail_on_nonfinite=True,
        verbose=False,
        comm=PETSc.COMM_WORLD.tompi4py(),
        save_history=True,
    )
    solve_time = time.perf_counter() - solve_t0
    final_global = backend.global_from_vec(x)
    final_metric = (
        float(result["history"][-1]["step_rel"])
        if result.get("history")
        else float("nan")
    )
    observables = backend.final_observables(final_global)
    status = "completed" if str(result.get("message", "")).lower().startswith("converged") else "failed"
    x.destroy()
    ksp.destroy()
    _append_stage_event(
        stage_path,
        stage="local_newton_done",
        started=started,
        nit=int(result.get("nit", 0)),
        status=str(status),
        final_metric=float(final_metric),
    )
    return {
        "status": str(status),
        "solver_success": bool(status == "completed"),
        "message": str(result.get("message", "")),
        "nit": int(result.get("nit", 0)),
        "solve_time": float(solve_time),
        "total_time": float(solve_time),
        "linear_iterations_total": int(sum(int(row["ksp_its"]) for row in linear_records)),
        "linear_history": list(linear_records),
        "history": list(result.get("history", [])),
        "final_metric": float(final_metric),
        "final_metric_name": "relative_correction",
        "initial_guess": init_meta,
        **observables,
    }


def _make_source_solver():
    _require_source_imports()
    return SolverFactory.create(
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        tolerance=1.0e-2,
        max_iterations=100,
        deflation_basis_tolerance=1.0e-3,
        verbose=False,
        q_mask=np.array([], dtype=bool),
        coord=None,
        preconditioner_options={
            "pc_backend": "hypre",
            "pc_hypre_boomeramg_coarsen_type": "HMIS",
            "pc_hypre_boomeramg_interp_type": "ext+i",
            "pc_hypre_boomeramg_strong_threshold": 0.5,
            "pc_hypre_boomeramg_max_iter": 1,
            "mpi_distribute_by_nodes": True,
        },
    )


def _run_source_solver_backend(
    backend,
    *,
    out_dir: Path,
    stop_tol: float,
    maxit: int,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> dict[str, object]:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    linear_solver = _make_source_solver()
    snap0 = _collector_snapshot(linear_solver)
    _append_stage_event(stage_path, stage="source_initial_guess_start", started=started)
    _setup_linear_system(linear_solver, backend.elastic_matrix())
    init = _solve_linear_system(
        linear_solver,
        backend.elastic_matrix(),
        np.asarray(backend.source_f, dtype=np.float64).reshape(-1, order="F"),
    )
    snap1 = _collector_snapshot(linear_solver)
    init_delta = _collector_delta(snap0, snap1)
    init_meta = {
        "enabled": True,
        "success": True,
        "ksp_iterations": int(init_delta["iterations"]),
        "solve_time": float(init_delta["solve_time"]),
        "vector_norm": float(np.linalg.norm(np.asarray(init, dtype=np.float64).reshape(-1))),
    }
    _append_stage_event(
        stage_path,
        stage="source_initial_guess_done",
        started=started,
        ksp_iterations=int(init_meta["ksp_iterations"]),
    )
    progress_events: list[dict[str, object]] = []

    def progress_callback(event: dict[str, object]) -> None:
        if PETSc.COMM_WORLD.getRank() == 0:
            progress_events.append(dict(event))
            if str(event.get("event", "")) == "newton_iteration":
                _append_stage_event(
                    stage_path,
                    stage="source_newton_iteration_done",
                    started=started,
                    iteration=int(event.get("iteration", 0)),
                    linear_iterations=int(event.get("linear_iterations", 0)),
                    stopping_value=float(event.get("stopping_value", np.nan)),
                    status=str(event.get("status", "")),
                )

    solve_t0 = time.perf_counter()
    _append_stage_event(stage_path, stage="source_newton_start", started=started)
    U_final, flag_N, nit = source_newton(
        np.asarray(init, dtype=np.float64).reshape((1, backend.n_free), order="F"),
        tol=1.0e-4,
        it_newt_max=int(maxit),
        it_damp_max=10,
        r_min=1.0e-4,
        K_elast=backend.elastic_matrix(),
        Q=backend.source_q,
        f=backend.source_f,
        constitutive_matrix_builder=backend,
        linear_system_solver=linear_solver,
        progress_callback=progress_callback,
        stopping_criterion="relative_correction",
        stopping_tol=float(stop_tol),
    )
    solve_time = time.perf_counter() - solve_t0
    final_global = np.asarray(U_final, dtype=np.float64).reshape(-1, order="F")
    history = [
        {
            "iteration": int(event.get("iteration", 0)),
            "metric": float(event.get("stopping_value", np.nan)),
            "metric_name": str(event.get("stop_criterion", "relative_correction")),
            "alpha": None if event.get("alpha") is None else float(event.get("alpha")),
            "linear_iterations": int(event.get("linear_iterations", 0)),
            "linear_solve_time": float(event.get("linear_solve_time", 0.0)),
            "iteration_wall_time": float(event.get("iteration_wall_time", 0.0)),
            "accepted_relative_correction_norm": (
                None
                if event.get("accepted_relative_correction_norm") is None
                else float(event.get("accepted_relative_correction_norm"))
            ),
            "status": str(event.get("status", "")),
        }
        for event in progress_events
        if str(event.get("event", "")) == "newton_iteration"
    ]
    final_metric = float(history[-1]["metric"]) if history else float("nan")
    observables = backend.final_observables(final_global)
    status = "completed" if int(flag_N) == 0 and np.isfinite(final_metric) else "failed"
    close = getattr(linear_solver, "close", None)
    if callable(close):
        close()
    _append_stage_event(
        stage_path,
        stage="source_newton_done",
        started=started,
        nit=int(nit),
        status=str(status),
        final_metric=float(final_metric),
    )
    return {
        "status": str(status),
        "solver_success": bool(status == "completed"),
        "message": "Converged" if status == "completed" else "Newton failed or hit maxit",
        "nit": int(nit),
        "solve_time": float(solve_time),
        "total_time": float(solve_time),
        "linear_iterations_total": int(
            sum(int(row["linear_iterations"]) for row in history)
        ),
        "history": history,
        "final_metric": float(final_metric),
        "final_metric_name": "relative_correction",
        "initial_guess": init_meta,
        **observables,
    }


def _case_payload(
    *,
    assembly_backend: str,
    solver_backend: str,
    stop_tol: float,
    maxit: int,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "assembly_backend": str(assembly_backend),
        "solver_backend": str(solver_backend),
        "ranks": int(PETSc.COMM_WORLD.getSize()),
        "stop_metric_name": "relative_correction",
        "stop_tol": float(stop_tol),
        "maxit": int(maxit),
        **result,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Plasticity3D backend-mix case."
    )
    parser.add_argument(
        "--assembly-backend",
        choices=("local", "local_constitutiveAD", "source"),
        required=True,
    )
    parser.add_argument("--solver-backend", choices=("local", "source"), required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--stop-tol", type=float, default=2.0e-3)
    parser.add_argument("--maxit", type=int, default=80)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    output_json = Path(args.output_json).resolve()
    case_started = time.perf_counter()
    stage_path = out_dir / "data" / "stage.jsonl"
    if PETSc.COMM_WORLD.getRank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "data").mkdir(parents=True, exist_ok=True)
    _append_stage_event(
        stage_path,
        stage="case_start",
        started=case_started,
        assembly_backend=str(args.assembly_backend),
        solver_backend=str(args.solver_backend),
    )

    if str(args.assembly_backend) in {"local", "local_constitutiveAD"}:
        backend = _build_local_assembly_backend(
            autodiff_tangent_mode=(
                "constitutive"
                if str(args.assembly_backend) == "local_constitutiveAD"
                else "element"
            ),
            stage_path=stage_path,
            stage_started=case_started,
        )
    else:
        backend = _build_source_assembly_backend(
            source_root=Path(args.source_root).resolve(),
            data_dir=out_dir / "data",
            need_energy_operator=False,
            stage_path=stage_path,
            stage_started=case_started,
        )
    _append_stage_event(stage_path, stage="backend_ready", started=case_started)

    total_t0 = time.perf_counter()
    if str(args.solver_backend) == "local":
        result = _run_local_solver_backend(
            backend,
            out_dir=out_dir,
            stop_tol=float(args.stop_tol),
            maxit=int(args.maxit),
            stage_path=stage_path,
            stage_started=case_started,
        )
    else:
        result = _run_source_solver_backend(
            backend,
            out_dir=out_dir,
            stop_tol=float(args.stop_tol),
            maxit=int(args.maxit),
            stage_path=stage_path,
            stage_started=case_started,
        )
    result["total_time"] = float(time.perf_counter() - total_t0)
    payload = _case_payload(
        assembly_backend=str(args.assembly_backend),
        solver_backend=str(args.solver_backend),
        stop_tol=float(args.stop_tol),
        maxit=int(args.maxit),
        result=result,
    )

    if PETSc.COMM_WORLD.getRank() == 0:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
    _append_stage_event(
        stage_path,
        stage="payload_written",
        started=case_started,
        status=str(payload.get("status", "")),
    )
    backend.close()


if __name__ == "__main__":
    main()
