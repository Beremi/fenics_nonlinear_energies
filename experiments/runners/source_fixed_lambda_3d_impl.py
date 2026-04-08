#!/usr/bin/env python3
"""Run the source PETSc4py 3D slope-stability plasticity case at a fixed lambda."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from slope_stability.cli import run_3D_hetero_SSR_capture as capture
from slope_stability.cli.progress import make_progress_logger
from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.core.elements import normalize_elem_type, validate_supported_elem_type
from slope_stability.export import write_debug_bundle_h5, write_history_json, write_vtu
from slope_stability.fem import (
    assemble_owned_elastic_rows_for_comm,
    assemble_strain_operator,
    prepare_owned_tangent_pattern,
    quadrature_volume_3d,
    vector_volume,
)
from slope_stability.fem.distributed_tangent import prepare_bddc_subdomain_pattern
from slope_stability.linear import SolverFactory
from slope_stability.linear.pmg import (
    build_3d_mixed_pmg_hierarchy,
    build_3d_mixed_pmg_hierarchy_with_intermediate_p2,
    build_3d_pmg_hierarchy,
    build_3d_same_mesh_pmg_hierarchy,
)
from slope_stability.mesh import (
    MaterialSpec,
    heterogenous_materials,
    load_mesh_from_file,
    reorder_mesh_nodes,
)
from slope_stability.nonlinear.newton import (
    _destroy_petsc_mat,
    _prefers_full_system_operator,
    _setup_linear_system,
    _solve_linear_system,
    newton,
)
from slope_stability.problem_assets import load_material_rows_for_path
from slope_stability.utils import (
    extract_submatrix_free,
    flatten_field,
    full_field_from_free_values,
    local_csr_to_petsc_aij_matrix,
    owned_block_range,
    q_to_free_indices,
)


def _free_dot(a: np.ndarray, b: np.ndarray, q_mask: np.ndarray) -> float:
    free_idx = q_to_free_indices(np.asarray(q_mask, dtype=bool))
    a_free = flatten_field(np.asarray(a, dtype=np.float64))[free_idx]
    b_free = flatten_field(np.asarray(b, dtype=np.float64))[free_idx]
    return float(np.dot(a_free, b_free))


def _vtk_cell_type(elem: np.ndarray) -> str:
    width = int(np.asarray(elem, dtype=np.int64).shape[0])
    if width == 4:
        return "tetra"
    if width == 10:
        return "tetra10"
    if width == 35:
        return "VTK_LAGRANGE_TETRAHEDRON"
    raise ValueError(f"Unsupported tetrahedral connectivity width {width}.")


def _progress_history(events: list[dict[str, object]]) -> list[dict[str, object]]:
    history: list[dict[str, object]] = []
    for event in events:
        if str(event.get("event", "")) != "newton_iteration":
            continue
        history.append(
            {
                "iteration": int(event.get("iteration", 0)),
                "status": str(event.get("status", "")),
                "metric": float(event.get("stopping_value", np.nan)),
                "metric_name": str(event.get("stop_criterion", "relative_residual")),
                "rel_residual": float(event.get("rel_residual", np.nan)),
                "criterion": float(event.get("criterion", np.nan)),
                "alpha": (
                    None
                    if event.get("alpha") is None
                    else float(event.get("alpha", np.nan))
                ),
                "r": float(event.get("r", np.nan)),
                "linear_iterations": int(event.get("linear_iterations", 0)),
                "linear_solve_time": float(event.get("linear_solve_time", 0.0)),
                "linear_preconditioner_time": float(
                    event.get("linear_preconditioner_time", 0.0)
                ),
                "linear_orthogonalization_time": float(
                    event.get("linear_orthogonalization_time", 0.0)
                ),
                "iteration_wall_time": float(event.get("iteration_wall_time", 0.0)),
                "accepted_relative_correction_norm": (
                    None
                    if event.get("accepted_relative_correction_norm") is None
                    else float(event.get("accepted_relative_correction_norm", np.nan))
                ),
            }
        )
    return history


def _material_specs(material_rows: list[list[float]] | np.ndarray) -> tuple[np.ndarray, list[MaterialSpec]]:
    mat_props = np.asarray(material_rows, dtype=np.float64)
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
        for row in mat_props
    ]
    return mat_props, materials


def _append_stage_event(
    path: Path | None,
    *,
    stage: str,
    started: float,
    **fields: object,
) -> None:
    if path is None:
        return
    payload = {
        "stage": str(stage),
        "elapsed_s": float(perf_counter() - started),
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def run_fixed_lambda(
    output_dir: Path,
    *,
    mesh_path: Path | None = None,
    mesh_boundary_type: int = 0,
    elem_type: str = "P4",
    davis_type: str = "B",
    material_rows: list[list[float]] | np.ndarray | None = None,
    node_ordering: str = "block_xyz",
    lambda_target: float = 1.5,
    it_newt_max: int = 20,
    it_damp_max: int = 10,
    tol: float = 1.0e-4,
    r_min: float = 1.0e-4,
    stopping_criterion: str = "relative_residual",
    stopping_tol: float | None = None,
    linear_tolerance: float = 1.0e-2,
    linear_max_iter: int = 100,
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
    factor_solver_type: str | None = None,
    pc_backend: str | None = "pmg",
    pmg_coarse_mesh_path: Path | None = None,
    pmg_fine_hierarchy_mode: str = "default",
    preconditioner_matrix_source: str = "tangent",
    preconditioner_matrix_policy: str = "current",
    preconditioner_rebuild_policy: str = "every_newton",
    preconditioner_rebuild_interval: int = 1,
    mpi_distribute_by_nodes: bool = True,
    pc_gamg_process_eq_limit: int | None = None,
    pc_gamg_threshold: float | None = None,
    pc_gamg_aggressive_coarsening: int | None = None,
    pc_gamg_aggressive_square_graph: bool | None = None,
    pc_gamg_aggressive_mis_k: int | None = None,
    pc_hypre_coarsen_type: str | None = "HMIS",
    pc_hypre_interp_type: str | None = "ext+i",
    pc_hypre_strong_threshold: float | None = None,
    pc_hypre_boomeramg_max_iter: int | None = 1,
    pc_hypre_P_max: int | None = None,
    pc_hypre_agg_nl: int | None = None,
    pc_hypre_nongalerkin_tol: float | None = None,
    mg_coarse_ksp_type: str | None = "cg",
    mg_coarse_pc_type: str | None = "hypre",
    petsc_opt: list[str] | None = None,
    compiled_outer: bool = False,
    recycle_preconditioner: bool = True,
    constitutive_mode: str = "overlap",
    tangent_kernel: str = "rows",
    max_deflation_basis_vectors: int = 48,
    threads: int | None = None,
    elastic_initial_guess: bool = True,
    write_debug_bundle: bool = True,
    write_history_json_file: bool = True,
    write_solution_vtu: bool = True,
    write_plots: bool = True,
    quiet: bool = False,
    output_json: Path | None = None,
    fixed_work_mode: bool = False,
) -> dict[str, object]:
    rank = int(PETSc.COMM_WORLD.getRank())
    out_dir = Path(output_dir)
    data_dir = out_dir / "data"
    exports_dir = out_dir / "exports"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        exports_dir.mkdir(parents=True, exist_ok=True)
    stage_path = data_dir / "stage.jsonl" if rank == 0 else None
    stage_started = perf_counter()
    _append_stage_event(
        stage_path,
        stage="start",
        started=stage_started,
        mesh_path=str(mesh_path) if mesh_path is not None else None,
        elem_type=str(elem_type),
        lambda_target=float(lambda_target),
    )

    progress_events: list[dict[str, object]] = []
    base_progress = None
    if rank == 0:
        progress_console = io.StringIO() if quiet else None
        base_progress = make_progress_logger(
            data_dir,
            console=progress_console,
        )

    def progress_callback(event: dict[str, object]) -> None:
        if rank != 0:
            return
        payload = dict(event)
        progress_events.append(payload)
        if base_progress is not None:
            base_progress(payload)

    elem_type = validate_supported_elem_type(3, elem_type)
    if mesh_path is None:
        mesh_path = (
            Path(capture.__file__).resolve().parents[3]
            / "meshes"
            / "3d_hetero_ssr"
            / "SSR_hetero_ada_L1.msh"
        )
    mesh_path = Path(mesh_path)

    solver_type_upper = str(solver_type).upper()
    effective_pc_backend = None if pc_backend is None else str(pc_backend).strip().lower()
    if effective_pc_backend is None:
        if "HYPRE" in solver_type_upper:
            effective_pc_backend = "hypre"
        elif "GAMG" in solver_type_upper:
            effective_pc_backend = "gamg"

    if material_rows is None:
        material_rows = load_material_rows_for_path(mesh_path)
    if material_rows is None:
        material_rows = [
            [15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0],
            [15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0],
            [10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 21.0],
            [18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0],
        ]
    mat_props, materials = _material_specs(material_rows)

    partition_count = (
        int(PETSc.COMM_WORLD.getSize())
        if str(node_ordering).lower() == "block_metis"
        else None
    )
    pmg_hierarchy = None
    if effective_pc_backend in {"pmg", "pmg_shell"}:
        fine_hierarchy_mode = str(pmg_fine_hierarchy_mode).strip().lower()
        if pmg_coarse_mesh_path is None:
            if effective_pc_backend == "pmg_shell" and str(elem_type).upper() == "P2":
                pmg_hierarchy = build_3d_same_mesh_pmg_hierarchy(
                    mesh_path,
                    fine_elem_type=elem_type,
                    boundary_type=int(mesh_boundary_type),
                    node_ordering=node_ordering,
                    reorder_parts=partition_count,
                    material_rows=np.asarray(mat_props, dtype=np.float64).tolist(),
                    comm=PETSc.COMM_WORLD,
                )
            else:
                pmg_hierarchy = build_3d_pmg_hierarchy(
                    mesh_path,
                    boundary_type=int(mesh_boundary_type),
                    node_ordering=node_ordering,
                    reorder_parts=partition_count,
                    material_rows=np.asarray(mat_props, dtype=np.float64).tolist(),
                    comm=PETSc.COMM_WORLD,
                )
        else:
            if fine_hierarchy_mode == "p4_p2_intermediate":
                pmg_hierarchy = build_3d_mixed_pmg_hierarchy_with_intermediate_p2(
                    mesh_path,
                    pmg_coarse_mesh_path,
                    boundary_type=int(mesh_boundary_type),
                    node_ordering=node_ordering,
                    reorder_parts=partition_count,
                    material_rows=np.asarray(mat_props, dtype=np.float64).tolist(),
                    comm=PETSc.COMM_WORLD,
                )
            else:
                pmg_hierarchy = build_3d_mixed_pmg_hierarchy(
                    mesh_path,
                    pmg_coarse_mesh_path,
                    fine_elem_type=elem_type,
                    boundary_type=int(mesh_boundary_type),
                    node_ordering=node_ordering,
                    reorder_parts=partition_count,
                    material_rows=np.asarray(mat_props, dtype=np.float64).tolist(),
                    comm=PETSc.COMM_WORLD,
                )
        coord = pmg_hierarchy.fine_level.coord.astype(np.float64)
        elem = pmg_hierarchy.fine_level.elem.astype(np.int64)
        surf = pmg_hierarchy.fine_level.surf.astype(np.int64)
        q_mask = pmg_hierarchy.fine_level.q_mask.astype(bool)
        material_identifier = pmg_hierarchy.fine_level.material_identifier.astype(np.int64).ravel()
    else:
        mesh = load_mesh_from_file(
            mesh_path,
            boundary_type=int(mesh_boundary_type),
            elem_type=elem_type,
        )
        if mesh.elem_type is not None and normalize_elem_type(mesh.elem_type) != elem_type:
            raise ValueError(
                f"Requested elem_type {elem_type!r}, but mesh contains {mesh.elem_type!r}."
            )
        reordered = reorder_mesh_nodes(
            mesh.coord,
            mesh.elem,
            mesh.surf,
            mesh.q_mask,
            strategy=node_ordering,
            n_parts=partition_count,
        )
        coord = reordered.coord.astype(np.float64)
        elem = reordered.elem.astype(np.int64)
        surf = reordered.surf.astype(np.int64)
        q_mask = reordered.q_mask.astype(bool)
        material_identifier = mesh.material.astype(np.int64).ravel()
    _append_stage_event(
        stage_path,
        stage="mesh_ready",
        started=stage_started,
        mesh_nodes=int(coord.shape[1]),
        mesh_elements=int(elem.shape[1]),
        reorder=str(node_ordering),
        pc_backend=str(effective_pc_backend),
    )

    n_q = int(quadrature_volume_3d(elem_type)[0].shape[1])
    n_int = int(elem.shape[1] * n_q)
    c0, phi, psi, shear, bulk, lame, gamma = heterogenous_materials(
        material_identifier,
        np.ones(n_int, dtype=bool),
        n_q,
        materials,
    )

    use_owned_mpi_tangent_path = capture.use_owned_tangent_path(
        solver_type=solver_type,
        mpi_distribute_by_nodes=mpi_distribute_by_nodes,
    )
    use_lightweight_mpi_path = capture.use_lightweight_mpi_elastic_path(
        solver_type=solver_type,
        mpi_distribute_by_nodes=mpi_distribute_by_nodes,
        constitutive_mode=constitutive_mode,
    )

    B = None
    weight = np.zeros(n_int, dtype=np.float64)
    elastic_rows = None
    tangent_pattern = None
    bddc_pattern = None

    if use_lightweight_mpi_path:
        elastic_rows = assemble_owned_elastic_rows_for_comm(
            coord,
            elem,
            q_mask,
            material_identifier,
            materials,
            PETSc.COMM_WORLD,
            elem_type=elem_type,
        )
        global_size = int(coord.shape[0] * coord.shape[1])
        K_elast = local_csr_to_petsc_aij_matrix(
            elastic_rows.local_matrix,
            global_shape=(global_size, global_size),
            comm=PETSc.COMM_WORLD,
            block_size=coord.shape[0],
        )
        rhs_parts = MPI.COMM_WORLD.allgather(
            np.asarray(elastic_rows.local_rhs, dtype=np.float64)
        )
        f_V = np.concatenate(rhs_parts).reshape(coord.shape[0], coord.shape[1], order="F")
    else:
        assembly = assemble_strain_operator(coord, elem, elem_type, dim=3)
        from slope_stability.fem.assembly import build_elastic_stiffness_matrix

        K_elast, weight, B = build_elastic_stiffness_matrix(assembly, shear, lame, bulk)
        f_v_int = np.vstack(
            (
                np.zeros(assembly.n_int, dtype=np.float64),
                -gamma.astype(np.float64),
                np.zeros(assembly.n_int, dtype=np.float64),
            )
        )
        f_V = vector_volume(assembly, f_v_int, weight)
    _append_stage_event(
        stage_path,
        stage="elastic_operator_ready",
        started=stage_started,
        lightweight_path=bool(use_lightweight_mpi_path),
        owned_tangent_path=bool(use_owned_mpi_tangent_path),
    )

    const_builder = ConstitutiveOperator(
        B=B,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type=str(davis_type),
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=weight,
        n_strain=6,
        n_int=n_int,
        dim=3,
        q_mask=q_mask,
    )

    if use_owned_mpi_tangent_path:
        row0, row1 = owned_block_range(coord.shape[1], coord.shape[0], PETSc.COMM_WORLD)
        tangent_pattern = prepare_owned_tangent_pattern(
            coord,
            elem,
            q_mask,
            material_identifier,
            materials,
            (row0 // coord.shape[0], row1 // coord.shape[0]),
            elem_type=elem_type,
            include_unique=(str(constitutive_mode).lower() != "overlap"),
            include_legacy_scatter=(str(tangent_kernel).lower() == "legacy"),
            include_overlap_B=(str(tangent_kernel).lower() == "legacy"),
            elastic_rows=elastic_rows if use_lightweight_mpi_path else None,
        )
        const_builder.set_owned_tangent_pattern(
            tangent_pattern,
            use_compiled=True,
            tangent_kernel=tangent_kernel,
            constitutive_mode=constitutive_mode,
            use_compiled_constitutive=True,
        )
        if effective_pc_backend == "bddc":
            bddc_pattern = prepare_bddc_subdomain_pattern(
                coord,
                elem,
                q_mask,
                material_identifier,
                materials,
                (row0 // coord.shape[0], row1 // coord.shape[0]),
                elem_type=elem_type,
                overlap_local_int_indices=tangent_pattern.local_int_indices,
            )
            const_builder.set_bddc_subdomain_pattern(bddc_pattern)
    _append_stage_event(
        stage_path,
        stage="constitutive_ready",
        started=stage_started,
        constitutive_mode=str(constitutive_mode),
        tangent_kernel=str(tangent_kernel),
    )

    effective_threads = int(
        threads
        if threads is not None
        else os.environ.get("OMP_NUM_THREADS", "1")
    )
    preconditioner_options: dict[str, object] = {
        "threads": max(1, effective_threads),
        "print_level": 0,
        "use_as_preconditioner": True,
        "factor_solver_type": factor_solver_type,
        "pc_backend": effective_pc_backend,
        "pmg_coarse_mesh_path": None
        if pmg_coarse_mesh_path is None
        else str(pmg_coarse_mesh_path),
        "pmg_fine_hierarchy_mode": str(pmg_fine_hierarchy_mode),
        "preconditioner_matrix_source": str(preconditioner_matrix_source),
        "preconditioner_matrix_policy": str(preconditioner_matrix_policy),
        "preconditioner_rebuild_policy": str(preconditioner_rebuild_policy),
        "preconditioner_rebuild_interval": int(preconditioner_rebuild_interval),
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "use_coordinates": True,
        "max_deflation_basis_vectors": int(max_deflation_basis_vectors),
    }
    if compiled_outer:
        preconditioner_options["compiled_outer"] = True
    if recycle_preconditioner:
        preconditioner_options["recycle_preconditioner"] = True
    if effective_pc_backend == "pmg":
        preconditioner_options.update(
            {
                "full_system_preconditioner": False,
                "pc_mg_galerkin": "both",
                "pc_mg_cycle_type": "v",
                "mg_levels_ksp_type": "richardson",
                "mg_levels_ksp_max_it": 3,
                "mg_levels_pc_type": "sor",
                "mg_coarse_ksp_type": str(mg_coarse_ksp_type or "cg"),
                "mg_coarse_pc_type": str(mg_coarse_pc_type or "hypre"),
                "pmg_hierarchy": pmg_hierarchy,
            }
        )
    if effective_pc_backend == "pmg_shell":
        preconditioner_options.update(
            {
                "full_system_preconditioner": False,
                "mg_levels_ksp_type": "richardson",
                "mg_levels_ksp_max_it": 3,
                "mg_levels_pc_type": "sor",
                "mg_coarse_ksp_type": str(mg_coarse_ksp_type or "cg"),
                "mg_coarse_pc_type": str(mg_coarse_pc_type or "hypre"),
                "mg_coarse_pc_hypre_type": "boomeramg",
                "pmg_hierarchy": pmg_hierarchy,
            }
        )
    if effective_pc_backend == "gamg":
        preconditioner_options.update(
            {
                "mg_levels_ksp_type": "richardson",
                "mg_levels_pc_type": "jacobi",
                "mg_levels_pc_jacobi_type": "rowl1",
                "mg_levels_pc_jacobi_rowl1_scale": 0.5,
                "mg_levels_pc_jacobi_fixdiagonal": True,
                "pc_gamg_agg_nsmooths": 1,
                "pc_gamg_esteig_ksp_max_it": 10,
                "pc_gamg_use_sa_esteig": False,
                "pc_gamg_coarse_eq_limit": 1000,
                "pc_mg_cycle_type": "v",
            }
        )
        if pc_gamg_process_eq_limit is not None:
            preconditioner_options["pc_gamg_process_eq_limit"] = int(
                pc_gamg_process_eq_limit
            )
        if pc_gamg_threshold is not None:
            preconditioner_options["pc_gamg_threshold"] = float(pc_gamg_threshold)
        if pc_gamg_aggressive_coarsening is not None:
            preconditioner_options["pc_gamg_aggressive_coarsening"] = int(
                pc_gamg_aggressive_coarsening
            )
        if pc_gamg_aggressive_square_graph is not None:
            preconditioner_options["pc_gamg_aggressive_square_graph"] = bool(
                pc_gamg_aggressive_square_graph
            )
        if pc_gamg_aggressive_mis_k is not None:
            preconditioner_options["pc_gamg_aggressive_mis_k"] = int(
                pc_gamg_aggressive_mis_k
            )
    if effective_pc_backend == "hypre":
        if pc_hypre_coarsen_type is not None:
            preconditioner_options["pc_hypre_boomeramg_coarsen_type"] = str(
                pc_hypre_coarsen_type
            )
        if pc_hypre_interp_type is not None:
            preconditioner_options["pc_hypre_boomeramg_interp_type"] = str(
                pc_hypre_interp_type
            )
        if pc_hypre_strong_threshold is not None:
            preconditioner_options["pc_hypre_boomeramg_strong_threshold"] = float(
                pc_hypre_strong_threshold
            )
        if pc_hypre_boomeramg_max_iter is not None:
            preconditioner_options["pc_hypre_boomeramg_max_iter"] = int(
                pc_hypre_boomeramg_max_iter
            )
        if pc_hypre_P_max is not None:
            preconditioner_options["pc_hypre_boomeramg_P_max"] = int(pc_hypre_P_max)
        if pc_hypre_agg_nl is not None:
            preconditioner_options["pc_hypre_boomeramg_agg_nl"] = int(pc_hypre_agg_nl)
        if pc_hypre_nongalerkin_tol is not None:
            preconditioner_options["pc_hypre_boomeramg_nongalerkin_tol"] = float(
                pc_hypre_nongalerkin_tol
            )
    preconditioner_options.update(capture._parse_petsc_opt_entries(petsc_opt))

    linear_system_solver = SolverFactory.create(
        solver_type,
        tolerance=linear_tolerance,
        max_iterations=linear_max_iter,
        deflation_basis_tolerance=1.0e-3,
        verbose=False,
        q_mask=q_mask,
        coord=coord,
        preconditioner_options=preconditioner_options,
    )

    initial_guess_meta: dict[str, object] = {"enabled": bool(elastic_initial_guess)}
    U_ini = np.zeros_like(np.asarray(f_V, dtype=np.float64))
    init_linear: dict[str, float | int] = {
        "init_linear_iterations": 0,
        "init_linear_solve_time": 0.0,
        "init_linear_preconditioner_time": 0.0,
        "init_linear_orthogonalization_time": 0.0,
    }

    t_total_start = perf_counter()
    if elastic_initial_guess:
        _append_stage_event(
            stage_path,
            stage="initial_guess_start",
            started=stage_started,
            solver_type=str(solver_type),
        )
        free_idx = q_to_free_indices(q_mask)
        f_full = np.asarray(f_V, dtype=np.float64).reshape(-1, order="F")
        f_free = f_full[free_idx]
        init_linear_solver = linear_system_solver
        if effective_pc_backend in {"pmg", "pmg_shell"}:
            init_preconditioner_options = dict(preconditioner_options)
            init_preconditioner_options["pc_backend"] = "hypre"
            init_preconditioner_options.pop("pmg_hierarchy", None)
            for key in tuple(init_preconditioner_options.keys()):
                if key.startswith("mg_") or key.startswith("pc_mg_"):
                    init_preconditioner_options.pop(key, None)
            init_linear_solver = SolverFactory.create(
                solver_type,
                tolerance=linear_tolerance,
                max_iterations=linear_max_iter,
                deflation_basis_tolerance=1.0e-3,
                verbose=False,
                q_mask=q_mask,
                coord=coord,
                preconditioner_options=init_preconditioner_options,
            )

        snap_init_0 = capture._collector_snapshot(init_linear_solver)
        U_elast_free = None
        K_free = None
        try:
            if _prefers_full_system_operator(init_linear_solver, K_elast):
                _setup_linear_system(
                    init_linear_solver,
                    K_elast,
                    A_full=K_elast,
                    free_idx=free_idx,
                )
                U_elast_free = _solve_linear_system(
                    init_linear_solver,
                    K_elast,
                    f_free,
                    b_full=f_full,
                    free_idx=free_idx,
                )
            else:
                K_free = extract_submatrix_free(K_elast, free_idx)
                _setup_linear_system(
                    init_linear_solver,
                    K_free,
                    A_full=K_elast,
                    free_idx=free_idx,
                )
                U_elast_free = _solve_linear_system(
                    init_linear_solver,
                    K_free,
                    f_free,
                    b_full=f_full,
                    free_idx=free_idx,
                )
        finally:
            release = getattr(init_linear_solver, "release_iteration_resources", None)
            if callable(release):
                release()
            _destroy_petsc_mat(K_free)

        snap_init_1 = capture._collector_snapshot(init_linear_solver)
        init_delta = capture._collector_delta(snap_init_0, snap_init_1)
        init_linear = {
            "init_linear_iterations": int(init_delta["iterations"]),
            "init_linear_solve_time": float(init_delta["solve_time"]),
            "init_linear_preconditioner_time": float(
                init_delta["preconditioner_time"]
            ),
            "init_linear_orthogonalization_time": float(
                init_delta["orthogonalization_time"]
            ),
        }
        U_ini = full_field_from_free_values(
            np.asarray(U_elast_free, dtype=np.float64),
            free_idx,
            f_V.shape,
        )
        if getattr(
            linear_system_solver,
            "supports_dynamic_deflation_basis",
            lambda: True,
        )():
            linear_system_solver.expand_deflation_basis(
                np.asarray(U_elast_free, dtype=np.float64)
            )
        initial_guess_meta.update(
            {
                "success": True,
                "ksp_iterations": int(init_linear["init_linear_iterations"]),
                "solve_time": float(init_linear["init_linear_solve_time"]),
                "vector_norm": float(np.linalg.norm(np.asarray(U_ini, dtype=np.float64))),
            }
        )
        _append_stage_event(
            stage_path,
            stage="initial_guess_done",
            started=stage_started,
            ksp_iterations=int(init_linear["init_linear_iterations"]),
            solve_time=float(init_linear["init_linear_solve_time"]),
        )

    solve_start = perf_counter()
    _append_stage_event(
        stage_path,
        stage="newton_start",
        started=stage_started,
        stopping_criterion=str(stopping_criterion),
        fixed_work_mode=bool(fixed_work_mode),
        it_newt_max=int(it_newt_max),
    )
    const_builder.reduction(float(lambda_target))
    U_final, flag_N, nit = newton(
        np.asarray(U_ini, dtype=np.float64),
        float(tol),
        int(it_newt_max),
        int(it_damp_max),
        float(r_min),
        K_elast,
        q_mask,
        f_V,
        const_builder,
        linear_system_solver,
        progress_callback=progress_callback,
        stopping_criterion=str(stopping_criterion),
        stopping_tol=stopping_tol,
    )
    solve_time = float(perf_counter() - solve_start)
    total_time = float(perf_counter() - t_total_start)
    _append_stage_event(
        stage_path,
        stage="newton_done",
        started=stage_started,
        nit=int(nit),
        flag_N=int(flag_N),
        solve_time=float(solve_time),
        total_time=float(total_time),
    )

    history = _progress_history(progress_events)
    final_metric = float(history[-1]["metric"]) if history else np.nan
    final_rel_residual = float(history[-1]["rel_residual"]) if history else np.nan
    linear_iterations_total = int(
        sum(int(row["linear_iterations"]) for row in history)
    )
    linear_solve_time_total = float(
        sum(float(row["linear_solve_time"]) for row in history)
    )
    linear_preconditioner_time_total = float(
        sum(float(row["linear_preconditioner_time"]) for row in history)
    )
    linear_orthogonalization_time_total = float(
        sum(float(row["linear_orthogonalization_time"]) for row in history)
    )

    omega = _free_dot(f_V, U_final, q_mask)
    internal_energy = float("nan")
    energy = float("nan")
    try:
        internal_energy = float(const_builder.potential_energy(U_final))
        energy = float(internal_energy - omega)
    except RuntimeError as exc:
        if "Global strain operator B is not available" not in str(exc):
            raise
    u_max = float(np.max(np.linalg.norm(np.asarray(U_final, dtype=np.float64), axis=0)))

    if int(flag_N) == 0:
        status = "completed"
        message = "Converged"
    elif fixed_work_mode and int(nit) >= int(it_newt_max):
        status = "completed_fixed_work"
        message = f"Reached fixed Newton cap ({int(it_newt_max)})"
    else:
        status = "failed"
        message = "Maximum number of iterations reached"

    mpi_comm = PETSc.COMM_WORLD.tompi4py()
    const_times = const_builder.get_total_time()
    const_times_max = {
        key: float(mpi_comm.allreduce(float(val), op=MPI.MAX))
        for key, val in const_times.items()
    }
    tangent_pattern_stats_max = None
    tangent_pattern_stats_sum = None
    tangent_pattern_timings_max = None
    if tangent_pattern is not None:
        tangent_pattern_stats_max = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.MAX))
            for key, val in tangent_pattern.stats.items()
        }
        tangent_pattern_stats_sum = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.SUM))
            for key, val in tangent_pattern.stats.items()
        }
        tangent_pattern_timings_max = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.MAX))
            for key, val in tangent_pattern.timings.items()
        }
    bddc_pattern_stats_max = None
    bddc_pattern_stats_sum = None
    bddc_pattern_timings_max = None
    if bddc_pattern is not None:
        bddc_pattern_stats_max = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.MAX))
            for key, val in bddc_pattern.stats.items()
        }
        bddc_pattern_stats_sum = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.SUM))
            for key, val in bddc_pattern.stats.items()
        }
        bddc_pattern_timings_max = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.MAX))
            for key, val in bddc_pattern.timings.items()
        }

    preconditioner_diag_fn = getattr(
        linear_system_solver,
        "get_preconditioner_diagnostics",
        None,
    )
    preconditioner_diagnostics = (
        dict(preconditioner_diag_fn()) if callable(preconditioner_diag_fn) else {}
    )
    linear_summary = {
        "init_linear_iterations": int(init_linear["init_linear_iterations"]),
        "init_linear_solve_time": float(init_linear["init_linear_solve_time"]),
        "init_linear_preconditioner_time": float(
            init_linear["init_linear_preconditioner_time"]
        ),
        "init_linear_orthogonalization_time": float(
            init_linear["init_linear_orthogonalization_time"]
        ),
        "solve_linear_iterations_total": int(linear_iterations_total),
        "solve_linear_solve_time_total": float(linear_solve_time_total),
        "solve_linear_preconditioner_time_total": float(
            linear_preconditioner_time_total
        ),
        "solve_linear_orthogonalization_time_total": float(
            linear_orthogonalization_time_total
        ),
        **preconditioner_diagnostics,
    }

    lambda_hist = np.asarray([float(lambda_target)], dtype=np.float64)
    omega_hist = np.asarray([float(omega)], dtype=np.float64)
    umax_hist = np.asarray([float(u_max)], dtype=np.float64)

    stats_payload = {
        "nit": int(nit),
        "linear_iterations": np.asarray(
            [int(row["linear_iterations"]) for row in history],
            dtype=np.int64,
        ),
        "linear_solve_time": np.asarray(
            [float(row["linear_solve_time"]) for row in history],
            dtype=np.float64,
        ),
        "linear_preconditioner_time": np.asarray(
            [float(row["linear_preconditioner_time"]) for row in history],
            dtype=np.float64,
        ),
        "linear_orthogonalization_time": np.asarray(
            [float(row["linear_orthogonalization_time"]) for row in history],
            dtype=np.float64,
        ),
        "iteration_wall_time": np.asarray(
            [float(row["iteration_wall_time"]) for row in history],
            dtype=np.float64,
        ),
        "rel_residual": np.asarray(
            [float(row["rel_residual"]) for row in history],
            dtype=np.float64,
        ),
        "criterion": np.asarray(
            [float(row["criterion"]) for row in history],
            dtype=np.float64,
        ),
        "alpha": np.asarray(
            [
                np.nan if row["alpha"] is None else float(row["alpha"])
                for row in history
            ],
            dtype=np.float64,
        ),
        "accepted_relative_correction_norm": np.asarray(
            [
                np.nan
                if row["accepted_relative_correction_norm"] is None
                else float(row["accepted_relative_correction_norm"])
                for row in history
            ],
            dtype=np.float64,
        ),
        "stopping_metric_name": np.asarray(
            [str(row["metric_name"]) for row in history],
            dtype=object,
        ),
    }

    run_payload = {
        "run_info": {
            "timestamp": np.datetime64("now").astype(str),
            "runtime_seconds": float(total_time),
            "solve_seconds": float(solve_time),
            "mpi_size": int(PETSc.COMM_WORLD.getSize()),
            "mesh_nodes": int(coord.shape[1]),
            "mesh_elements": int(elem.shape[1]),
            "unknowns": int(np.asarray(q_mask, dtype=bool).sum()),
            "analysis": "fixed_lambda_ssr",
            "solver_type": str(solver_type),
            "step_count": 1,
        },
        "params": {
            "mesh_path": str(mesh_path),
            "mesh_boundary_type": int(mesh_boundary_type),
            "elem_type": str(elem_type),
            "davis_type": str(davis_type),
            "material_rows": np.asarray(mat_props, dtype=np.float64).tolist(),
            "node_ordering": str(node_ordering),
            "lambda_target": float(lambda_target),
            "it_newt_max": int(it_newt_max),
            "it_damp_max": int(it_damp_max),
            "tol": float(tol),
            "r_min": float(r_min),
            "stopping_criterion": str(stopping_criterion),
            "stopping_tol": None if stopping_tol is None else float(stopping_tol),
            "fixed_work_mode": bool(fixed_work_mode),
            "linear_tolerance": float(linear_tolerance),
            "linear_max_iter": int(linear_max_iter),
            "solver_type": str(solver_type),
            "pc_backend": str(effective_pc_backend),
            "pmg_fine_hierarchy_mode": str(pmg_fine_hierarchy_mode),
            "preconditioner_matrix_source": str(preconditioner_matrix_source),
            "preconditioner_matrix_policy": str(preconditioner_matrix_policy),
            "preconditioner_rebuild_policy": str(preconditioner_rebuild_policy),
            "preconditioner_rebuild_interval": int(preconditioner_rebuild_interval),
            "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
            "compiled_outer": bool(compiled_outer),
            "recycle_preconditioner": bool(recycle_preconditioner),
            "constitutive_mode": str(constitutive_mode),
            "tangent_kernel": str(tangent_kernel),
            "petsc_opt": list(petsc_opt or []),
            "threads": int(max(1, effective_threads)),
        },
        "mesh": {
            "mesh_file": str(mesh_path),
            "coord_shape": tuple(int(v) for v in coord.shape),
            "elem_shape": tuple(int(v) for v in elem.shape),
            "surf_shape": tuple(int(v) for v in surf.shape),
        },
        "timings": {
            "constitutive": const_times_max,
            "linear": linear_summary,
            "solve_wall_time": float(solve_time),
            "total_wall_time": float(total_time),
        },
        "owned_tangent_pattern": None
        if tangent_pattern is None
        else {
            "stats_max": tangent_pattern_stats_max,
            "stats_sum": tangent_pattern_stats_sum,
            "timings_max": tangent_pattern_timings_max,
        },
        "bddc_subdomain_pattern": None
        if bddc_pattern is None
        else {
            "stats_max": bddc_pattern_stats_max,
            "stats_sum": bddc_pattern_stats_sum,
            "timings_max": bddc_pattern_timings_max,
        },
    }

    result_payload = {
        "mesh_path": str(mesh_path),
        "elem_type": str(elem_type),
        "lambda_target": float(lambda_target),
        "nit": int(nit),
        "energy": float(energy),
        "internal_energy": float(internal_energy),
        "omega": float(omega),
        "u_max": float(u_max),
        "message": str(message),
        "status": str(status),
        "solver_success": bool(status in {"completed", "completed_fixed_work"}),
        "solve_time": float(solve_time),
        "total_time": float(total_time),
        "linear_iterations_total": int(linear_iterations_total),
        "final_metric": float(final_metric),
        "final_metric_name": str(stopping_criterion),
        "final_rel_residual": float(final_rel_residual),
        "history_metric_name": str(stopping_criterion),
        "history": history,
        "initial_guess": dict(initial_guess_meta),
        "linear_solver": {
            "solver_type": str(solver_type),
            "pc_backend": str(effective_pc_backend),
            "linear_tolerance": float(linear_tolerance),
            "linear_max_iter": int(linear_max_iter),
            **preconditioner_diagnostics,
        },
    }

    if rank == 0:
        np.savez_compressed(
            data_dir / "petsc_run.npz",
            U=np.asarray(U_final, dtype=np.float64),
            lambda_hist=lambda_hist,
            load_factor_hist=lambda_hist,
            omega_hist=omega_hist,
            Umax_hist=umax_hist,
            **{
                "stats_" + key: np.asarray(value)
                for key, value in stats_payload.items()
            },
        )
        (data_dir / "run_info.json").write_text(
            json.dumps(run_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        try:
            close_solver = getattr(linear_system_solver, "close", None)
            if callable(close_solver):
                close_solver()
            else:
                linear_system_solver.release_iteration_resources()
        except Exception:
            pass
        try:
            const_builder.release_petsc_caches()
        except Exception:
            pass

        if write_plots:
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_B = B
            if plot_B is None:
                plot_B = assemble_strain_operator(coord, elem, elem_type, dim=3).B
            capture._save_plots(
                coord,
                surf,
                np.asarray(U_final, dtype=np.float64),
                lambda_hist,
                omega_hist,
                plot_B,
                plots_dir,
                step_u=np.asarray([U_final], dtype=np.float64),
                elem=elem,
                n_q=n_q,
                load_label=r"$\lambda$",
                title_prefix=r"Fixed $\lambda$: $\omega$ vs $\lambda$",
            )

        progress_path = data_dir / "progress.jsonl"
        run_info_path = data_dir / "run_info.json"
        npz_path = data_dir / "petsc_run.npz"
        if write_debug_bundle and progress_path.exists():
            write_debug_bundle_h5(
                out_path=exports_dir / "debug_bundle.h5",
                config_text=json.dumps(run_payload["params"], indent=2),
                run_info_path=run_info_path,
                npz_path=npz_path,
                progress_path=progress_path,
            )
        if write_history_json_file and progress_path.exists():
            write_history_json(
                out_path=exports_dir / "history.json",
                run_info_path=run_info_path,
                npz_path=npz_path,
                progress_path=progress_path,
            )
        if write_solution_vtu:
            write_vtu(
                exports_dir / "solution.vtu",
                points=np.asarray(coord.T, dtype=np.float64),
                cell_blocks=[(_vtk_cell_type(elem), np.asarray(elem.T, dtype=np.int64))],
                point_data={
                    "displacement": np.asarray(U_final.T, dtype=np.float64),
                    "displacement_magnitude": np.linalg.norm(
                        np.asarray(U_final.T, dtype=np.float64),
                        axis=1,
                    ),
                },
                cell_data={"material_id": np.asarray(material_identifier, dtype=np.int64)},
            )

        result_payload.update(
            {
                "native_run_info": str(run_info_path),
                "native_npz": str(npz_path),
                "native_progress_jsonl": str(progress_path),
                "native_history_json": str(exports_dir / "history.json")
                if (exports_dir / "history.json").exists()
                else "",
                "native_debug_bundle": str(exports_dir / "debug_bundle.h5")
                if (exports_dir / "debug_bundle.h5").exists()
                else "",
                "native_vtu": str(exports_dir / "solution.vtu")
                if (exports_dir / "solution.vtu").exists()
                else "",
            }
        )

        output_path = out_dir / "output.json" if output_json is None else Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result_payload, indent=2) + "\n",
            encoding="utf-8",
        )

    return result_payload if rank == 0 else {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the source PETSc4py 3D heterogeneous slope case at a fixed lambda."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--mesh-path", type=Path, default=None)
    parser.add_argument("--mesh-boundary-type", type=int, default=0)
    parser.add_argument("--elem-type", type=str, default="P4", choices=["P1", "P2", "P4"])
    parser.add_argument("--davis-type", type=str, default="B")
    parser.add_argument(
        "--node-ordering",
        type=str,
        default="block_xyz",
        choices=["original", "xyz", "block_xyz", "morton", "rcm", "block_rcm", "block_metis"],
    )
    parser.add_argument("--lambda-target", type=float, default=1.5)
    parser.add_argument("--it-newt-max", type=int, default=20)
    parser.add_argument("--it-damp-max", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--r-min", type=float, default=1.0e-4)
    parser.add_argument(
        "--stopping-criterion",
        type=str,
        default="relative_residual",
        choices=["relative_residual", "relative_correction"],
    )
    parser.add_argument("--stopping-tol", type=float, default=None)
    parser.add_argument("--linear-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--linear-max-iter", type=int, default=100)
    parser.add_argument(
        "--solver-type",
        type=str,
        default="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
    )
    parser.add_argument("--factor-solver-type", type=str, default=None)
    parser.add_argument(
        "--pc-backend",
        type=str,
        default="pmg",
        choices=["hypre", "gamg", "bddc", "pmg", "pmg_shell"],
    )
    parser.add_argument("--pmg-coarse-mesh-path", type=Path, default=None)
    parser.add_argument(
        "--pmg-fine-hierarchy-mode",
        type=str,
        default="default",
        choices=["default", "p4_p2_intermediate"],
    )
    parser.add_argument(
        "--preconditioner-matrix-source",
        type=str,
        default="tangent",
        choices=["tangent", "regularized", "elastic"],
    )
    parser.add_argument(
        "--preconditioner-matrix-policy",
        type=str,
        default="current",
        choices=["current", "lagged"],
    )
    parser.add_argument(
        "--preconditioner-rebuild-policy",
        type=str,
        default="every_newton",
        choices=["every_newton", "every_n_newton", "accepted_step", "accepted_or_rejected_step"],
    )
    parser.add_argument("--preconditioner-rebuild-interval", type=int, default=1)
    parser.add_argument(
        "--mpi-distribute-by-nodes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pc-gamg-process-eq-limit", type=int, default=None)
    parser.add_argument("--pc-gamg-threshold", type=float, default=None)
    parser.add_argument("--pc-gamg-aggressive-coarsening", type=int, default=None)
    parser.add_argument(
        "--pc-gamg-aggressive-square-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--pc-gamg-aggressive-mis-k", type=int, default=None)
    parser.add_argument("--pc-hypre-coarsen-type", type=str, default="HMIS")
    parser.add_argument("--pc-hypre-interp-type", type=str, default="ext+i")
    parser.add_argument("--pc-hypre-strong-threshold", type=float, default=None)
    parser.add_argument("--pc-hypre-boomeramg-max-iter", type=int, default=1)
    parser.add_argument("--pc-hypre-p-max", type=int, default=None)
    parser.add_argument("--pc-hypre-agg-nl", type=int, default=None)
    parser.add_argument("--pc-hypre-nongalerkin-tol", type=float, default=None)
    parser.add_argument("--mg-coarse-ksp-type", type=str, default="cg")
    parser.add_argument("--mg-coarse-pc-type", type=str, default="hypre")
    parser.add_argument("--petsc-opt", action="append", default=[], dest="petsc_opt")
    parser.add_argument("--compiled-outer", action="store_true", default=False)
    parser.add_argument(
        "--recycle-preconditioner",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-deflation-basis-vectors", type=int, default=48)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--elastic-initial-guess",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write-debug-bundle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write-history-json",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write-solution-vtu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--fixed-work-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    result = run_fixed_lambda(
        args.out_dir,
        output_json=args.output_json,
        mesh_path=args.mesh_path,
        mesh_boundary_type=args.mesh_boundary_type,
        elem_type=args.elem_type,
        davis_type=args.davis_type,
        node_ordering=args.node_ordering,
        lambda_target=args.lambda_target,
        it_newt_max=args.it_newt_max,
        it_damp_max=args.it_damp_max,
        tol=args.tol,
        r_min=args.r_min,
        stopping_criterion=args.stopping_criterion,
        stopping_tol=args.stopping_tol,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_iter,
        solver_type=args.solver_type,
        factor_solver_type=args.factor_solver_type,
        pc_backend=args.pc_backend,
        pmg_coarse_mesh_path=args.pmg_coarse_mesh_path,
        pmg_fine_hierarchy_mode=args.pmg_fine_hierarchy_mode,
        preconditioner_matrix_source=args.preconditioner_matrix_source,
        preconditioner_matrix_policy=args.preconditioner_matrix_policy,
        preconditioner_rebuild_policy=args.preconditioner_rebuild_policy,
        preconditioner_rebuild_interval=args.preconditioner_rebuild_interval,
        mpi_distribute_by_nodes=args.mpi_distribute_by_nodes,
        pc_gamg_process_eq_limit=args.pc_gamg_process_eq_limit,
        pc_gamg_threshold=args.pc_gamg_threshold,
        pc_gamg_aggressive_coarsening=args.pc_gamg_aggressive_coarsening,
        pc_gamg_aggressive_square_graph=args.pc_gamg_aggressive_square_graph,
        pc_gamg_aggressive_mis_k=args.pc_gamg_aggressive_mis_k,
        pc_hypre_coarsen_type=args.pc_hypre_coarsen_type,
        pc_hypre_interp_type=args.pc_hypre_interp_type,
        pc_hypre_strong_threshold=args.pc_hypre_strong_threshold,
        pc_hypre_boomeramg_max_iter=args.pc_hypre_boomeramg_max_iter,
        pc_hypre_P_max=args.pc_hypre_p_max,
        pc_hypre_agg_nl=args.pc_hypre_agg_nl,
        pc_hypre_nongalerkin_tol=args.pc_hypre_nongalerkin_tol,
        mg_coarse_ksp_type=args.mg_coarse_ksp_type,
        mg_coarse_pc_type=args.mg_coarse_pc_type,
        petsc_opt=args.petsc_opt,
        compiled_outer=args.compiled_outer,
        recycle_preconditioner=args.recycle_preconditioner,
        constitutive_mode="overlap",
        tangent_kernel="rows",
        max_deflation_basis_vectors=args.max_deflation_basis_vectors,
        threads=args.threads,
        elastic_initial_guess=args.elastic_initial_guess,
        write_debug_bundle=args.write_debug_bundle,
        write_history_json_file=args.write_history_json,
        write_solution_vtu=args.write_solution_vtu,
        write_plots=args.write_plots,
        quiet=args.quiet,
        fixed_work_mode=args.fixed_work_mode,
    )
    if PETSc.COMM_WORLD.getRank() == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
