"""
HyperElasticity 3D — solver logic (DOF-partitioned JAX + PETSc).

Provides ``PROFILE_DEFAULTS`` and ``run(args)`` which runs all load steps.
CLI entry point (argparse) is in ``solve_HE_dof.py``.
"""

import gc
import hashlib
from pathlib import Path
import resource
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.benchmark.repair import build_retry_attempts, needs_solver_repair
from src.core.benchmark.run_record import sha256_file
from src.core.benchmark.state_export import export_hyperelasticity_state_npz
from src.core.petsc.minimizers import newton
from src.core.petsc.metrics import (
    EuclideanMetric,
    MatrixRieszMetric,
    certify_spd_by_cholesky,
)
from src.core.petsc.load_step_driver import (
    attempts_from_tuples,
    build_load_step_result,
    run_load_steps,
)
from src.core.petsc.trust_ksp import ksp_cg_set_radius
from src.problems.hyperelasticity.jax_petsc.parallel_hessian_dof import (
    LocalColoringAssembler,
    ParallelDOFHessianAssembler,
)
from src.problems.hyperelasticity.jax_petsc.reordered_element_assembler import (
    HEReorderedElementAssembler,
)
from src.problems.hyperelasticity.jax_petsc.multigrid import (
    HEPmgSmootherConfig,
    build_he_pmg_hierarchy,
    choose_he_pmg_coarsest_level,
    configure_he_pmg,
)
from src.problems.hyperelasticity.support.mesh import (
    MeshHyperElasticity3D,
    build_procedural_hyperelasticity_export_params,
    load_rank_local_hyperelasticity,
    local_dirichlet_values_from_reference,
    reordered_free_to_total_dofs,
)
from src.problems.hyperelasticity.support.rotate_boundary import (
    rotate_right_face_from_reference,
)
from src.core.petsc.gamg import build_gamg_coordinates


PROFILE_DEFAULTS = {
    "reference": {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "pc_setup_on_ksp_cap": True,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
        "reorder": False,
    },
    "performance": {
        "ksp_type": "gmres",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "pc_setup_on_ksp_cap": True,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
        "reorder": False,
    },
}


def _gather_full_free_original(assembler, vec) -> np.ndarray:
    owned = np.asarray(vec.array[:], dtype=np.float64)
    if hasattr(assembler, "part") and hasattr(assembler.part, "get_u_full"):
        full_reordered = assembler.part.get_u_full(owned)
        return np.asarray(assembler.part.reordered_to_original(full_reordered), dtype=np.float64)
    if hasattr(assembler, "_allgather_full_owned") and hasattr(assembler, "layout"):
        full_reordered, _ = assembler._allgather_full_owned(owned)
        if bool(getattr(assembler, "_formula_layout", False)):
            grid = assembler.params["_he_grid"]
            mode = str(assembler.params["_distributed_reorder_mode"])
            total_dofs = reordered_free_to_total_dofs(
                np.arange(int(assembler.layout.n_free), dtype=np.int64),
                grid,
                mode,
            )
            original = np.empty_like(full_reordered)
            node = total_dofs // 3
            comp = total_dofs % 3
            ix = node % int(grid.nx1)
            plane = node // int(grid.nx1)
            iy = plane % int(grid.ny1)
            iz = plane // int(grid.ny1)
            block = (iz * int(grid.ny1) + iy) * (int(grid.nx) - 1) + (ix - 1)
            original_index = 3 * block + comp
            original[original_index] = full_reordered
            return original
        full_original = np.empty_like(full_reordered)
        full_original[np.asarray(assembler.layout.perm, dtype=np.int64)] = full_reordered
        return full_original
    raise TypeError(f"Unsupported assembler type for state export: {type(assembler).__name__}")


def _distributed_vector_sha256(vec: PETSc.Vec, comm: MPI.Comm) -> str:
    chunks = comm.gather(
        np.asarray(vec.getArray(readonly=True), dtype=np.float64).copy(),
        root=0,
    )
    digest = (
        _sha256_array(np.concatenate(chunks or [])) if comm.rank == 0 else None
    )
    return str(comm.bcast(digest, root=0))


def _load_hyperelasticity_initial_state(
    *,
    path: str,
    args,
    params: dict[str, object],
    assembler,
    comm: MPI.Comm,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a canonical full deformation map and convert it to solver ordering."""

    destination = Path(path).resolve()
    root_error: str | None = None
    coords_final: np.ndarray | None = None
    file_sha256: str | None = None
    if bool(params.get("_distributed_local_data", False)):
        mesh_source = str(params.get("_distributed_mesh_source", "hdf5"))
        if mesh_source == "procedural":
            export_params = build_procedural_hyperelasticity_export_params(int(args.level))
        else:
            export_params, _, _ = MeshHyperElasticity3D(int(args.level)).get_data()
    else:
        export_params = params

    if comm.rank == 0:
        try:
            if destination.suffix != ".npz" or not destination.is_file():
                raise ValueError(
                    "Hyperelasticity initial state must be an existing .npz file"
                )
            with np.load(destination, allow_pickle=False) as archive:
                required = {"coords_ref", "coords_final", "tetrahedra", "mesh_level"}
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(
                        f"Hyperelasticity initial state is missing datasets {sorted(missing)}"
                    )
                coords_ref = np.asarray(archive["coords_ref"], dtype=np.float64)
                coords_final = np.asarray(archive["coords_final"], dtype=np.float64)
                tetrahedra = np.asarray(archive["tetrahedra"], dtype=np.int32)
                stored_level = int(np.asarray(archive["mesh_level"]).item())
            expected_coords = np.asarray(export_params["nodes2coord"], dtype=np.float64)
            expected_tetrahedra = np.asarray(
                export_params["elems_scalar"], dtype=np.int32
            )
            if stored_level != int(args.level):
                raise ValueError("Hyperelasticity initial-state mesh level does not match")
            if not np.array_equal(coords_ref, expected_coords):
                raise ValueError(
                    "Hyperelasticity initial-state reference coordinates do not match"
                )
            if not np.array_equal(tetrahedra, expected_tetrahedra):
                raise ValueError("Hyperelasticity initial-state connectivity does not match")
            if coords_final.shape != expected_coords.shape:
                raise ValueError("Hyperelasticity initial-state deformation has wrong shape")
            if not np.all(np.isfinite(coords_final)):
                raise ValueError(
                    "Hyperelasticity initial-state deformation contains nonfinite values"
                )
            file_sha256 = sha256_file(destination)
        except Exception as exc:
            root_error = f"{type(exc).__name__}: {exc}"
    root_error = comm.bcast(root_error, root=0)
    if root_error is not None:
        raise ValueError(
            f"Canonical Hyperelasticity initial-state validation failed: {root_error}"
        )
    coords_final = comm.bcast(coords_final, root=0)
    file_sha256 = comm.bcast(file_sha256, root=0)
    if coords_final is None or file_sha256 is None:
        raise RuntimeError("Canonical Hyperelasticity initial-state broadcast failed")

    flattened = np.asarray(coords_final, dtype=np.float64).reshape(-1)
    if bool(params.get("_distributed_local_data", False)):
        lo, hi = int(assembler.layout.lo), int(assembler.layout.hi)
        total_dofs = reordered_free_to_total_dofs(
            np.arange(lo, hi, dtype=np.int64),
            params["_he_grid"],
            str(params["_distributed_reorder_mode"]),
            int(params.get("element_degree", 1)),
        )
        solver_values = np.asarray(flattened[total_dofs], dtype=np.float64)
    else:
        free_original = flattened[np.asarray(params["freedofs"], dtype=np.int64)]
        solver_values = np.asarray(
            free_original[np.asarray(assembler.part.perm, dtype=np.int64)],
            dtype=np.float64,
        )
    return solver_values, {
        "source": "canonical_npz",
        "path": str(destination),
        "file_sha256": str(file_sha256),
        "state_sha256": _sha256_array(coords_final),
        "ordering": "global mesh-node vector components",
        "mesh_identity_verified": True,
    }


def _export_state_if_requested(
    args, assembler, params, vec, step_records, comm
) -> dict[str, object] | None:
    state_out = str(getattr(args, "state_out", "") or "")
    if not state_out:
        return None

    full_free_original = _gather_full_free_original(assembler, vec)
    export_params = params
    if bool(params.get("_distributed_local_data", False)):
        if int(params.get("element_degree", 1)) != 1:
            raise ValueError(
                "HyperElasticity state export is currently implemented for "
                "element_degree=1 only"
            )
        if comm.rank != 0:
            return None
        mesh_source = str(params.get("_distributed_mesh_source", "hdf5"))
        if mesh_source == "procedural":
            export_params = build_procedural_hyperelasticity_export_params(args.level)
        else:
            mesh_obj = MeshHyperElasticity3D(args.level)
            export_params, _, _ = mesh_obj.get_data()

    if step_records:
        final_angle = float(step_records[-1]["angle"])
        final_energy = float(step_records[-1]["energy"])
        completed_steps = int(len(step_records))
        full_state = rotate_right_face_from_reference(
            export_params["u_0_ref"],
            export_params["nodes2coord"],
            final_angle,
            export_params["right_nodes"],
        )
    else:
        final_energy = None
        completed_steps = 0
        full_state = np.asarray(export_params["u_0_ref"], dtype=np.float64).copy()

    full_state = np.asarray(full_state, dtype=np.float64).copy()
    full_state[np.asarray(export_params["freedofs"], dtype=np.int64)] = full_free_original

    if comm.rank == 0:
        export_hyperelasticity_state_npz(
            state_out,
            coords_ref=np.asarray(export_params["nodes2coord"], dtype=np.float64),
            x_final=full_state.reshape((-1, 3)),
            tetrahedra=np.asarray(export_params["elems_scalar"], dtype=np.int32),
            mesh_level=int(args.level),
            total_steps=int(args.total_steps),
            energy=final_energy,
            metadata={
                "solver_family": "hyperelasticity_jax_petsc_element",
                "mpi_ranks": int(comm.Get_size()),
                "completed_steps": int(completed_steps),
                "convergence_metric": str(_convergence_metric_name(args)),
            },
        )
        destination = Path(state_out).resolve()
        return {
            "path": str(destination),
            "file_sha256": sha256_file(destination),
            "state_sha256": _sha256_array(full_state),
            "free_state_sha256": _sha256_array(full_free_original),
            "ordering": "global mesh-node vector components",
        }
    return None


def _summarize_rank_memory(rank_summaries) -> dict[str, float | int | list[dict[str, object]]]:
    if not rank_summaries:
        return {"ranks": 0, "rank_summaries": []}

    rows: list[dict[str, object]] = []
    for rank, summary in enumerate(rank_summaries):
        row = dict(summary)
        row["rank"] = int(rank)
        rows.append(row)

    aggregate_keys = (
        "local_elements",
        "local_overlap_dofs",
        "owned_nnz",
        "layout_gib",
        "local_overlap_gib",
        "scatter_gib",
        "owned_hessian_values_gib",
        "petsc_owned_values_gib",
        "local_backend_gib",
        "tracked_total_gib",
    )
    out: dict[str, float | int | list[dict[str, object]]] = {
        "ranks": int(len(rows)),
        "rank_summaries": rows,
    }
    for key in aggregate_keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        out[f"{key}_min"] = float(min(values))
        out[f"{key}_max"] = float(max(values))
        out[f"{key}_total"] = float(sum(values))
    return out


def _summarize_rank_rss(comm: MPI.Comm) -> dict[str, float | list[float]]:
    local_rss_mib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    gathered = comm.gather(local_rss_mib, root=0)
    if comm.rank != 0 or not gathered:
        return {}
    values = [float(value) for value in gathered]
    return {
        "rank_ru_maxrss_mib": values,
        "ru_maxrss_mib_min": float(min(values)),
        "ru_maxrss_mib_max": float(max(values)),
        "ru_maxrss_mib_total": float(sum(values)),
    }


def _collect_publication_timing(
    comm: MPI.Comm,
    *,
    setup_s: float,
    first_step_s: float,
    solve_s: float,
    total_s: float,
) -> dict[str, object]:
    """Gather rank-local timings and retain proof of each collective maximum.

    Values are sampled before this gather, so the measured total excludes the
    reporting collective itself.  Publication analysis can independently
    recompute each maximum from the retained rank vector.
    """

    local = {
        "rank": int(comm.Get_rank()),
        "setup_s": float(setup_s),
        "first_step_s": float(first_step_s),
        "solve_s": float(solve_s),
        "total_s": float(total_s),
    }
    if any(
        not np.isfinite(float(local[key])) or float(local[key]) <= 0.0
        for key in ("setup_s", "first_step_s", "solve_s", "total_s")
    ):
        raise ValueError("publication timing values must be finite and positive")
    gathered = comm.gather(local, root=0)
    if int(comm.Get_rank()) != 0:
        return {}
    rows = sorted(
        (dict(row) for row in (gathered or [])),
        key=lambda row: int(row["rank"]),
    )
    expected_ranks = list(range(int(comm.Get_size())))
    if [int(row["rank"]) for row in rows] != expected_ranks:
        raise ValueError(
            "publication timing gather did not return every MPI rank exactly once"
        )

    phases: dict[str, object] = {}
    for phase in ("setup", "first_step", "solve", "total"):
        values = [float(row[f"{phase}_s"]) for row in rows]
        phases[phase] = {
            "collective_max_s": float(max(values)),
            "per_rank_s": values,
        }
    return {
        "schema_id": "fenics-nonlinear-energies.mpi-phase-timing",
        "schema_version": 1,
        "reduction": "mpi_collective_max",
        "rank_count": int(comm.Get_size()),
        "measured_region_excludes_reporting_collective": True,
        "phases": phases,
    }


def _resolve_linear_settings(args):
    settings = dict(PROFILE_DEFAULTS[args.profile])
    overrides = {
        "ksp_type": args.ksp_type,
        "pc_type": args.pc_type,
        "ksp_rtol": args.ksp_rtol,
        "ksp_max_it": args.ksp_max_it,
        "pc_setup_on_ksp_cap": args.pc_setup_on_ksp_cap,
        "gamg_threshold": args.gamg_threshold,
        "gamg_agg_nsmooths": args.gamg_agg_nsmooths,
        "use_near_nullspace": args.use_near_nullspace,
        "gamg_set_coordinates": args.gamg_set_coordinates,
        "reorder": args.reorder,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings


def _pc_options(settings):
    opts = {}
    if settings["pc_type"] == "gamg":
        opts["pc_gamg_threshold"] = float(settings["gamg_threshold"])
        opts["pc_gamg_agg_nsmooths"] = int(settings["gamg_agg_nsmooths"])
    return opts


def _sha256_array(values: object) -> str:
    """Hash an array together with its dtype and shape."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(int(value) for value in array.shape)).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _convergence_metric_name(args) -> str:
    name = str(
        getattr(args, "convergence_metric", "coefficient_l2") or "coefficient_l2"
    )
    if name not in {"coefficient_l2", "reference_elastic_energy"}:
        raise ValueError(f"Unsupported HyperElasticity convergence metric {name!r}")
    return name


def _positive_explicit_state_scale(args) -> float | None:
    explicit_scale = getattr(args, "convergence_state_scale", None)
    if explicit_scale is None:
        return None
    explicit_scale = float(explicit_scale)
    if not np.isfinite(explicit_scale) or explicit_scale <= 0.0:
        raise ValueError(
            "--convergence-state-scale must be finite and strictly positive"
        )
    return explicit_scale


def _owned_partition_identity(vec: PETSc.Vec, comm: MPI.Comm) -> list[dict[str, object]]:
    lo, hi = (int(value) for value in vec.getOwnershipRange())
    local = np.asarray(vec.getArray(readonly=True), dtype=np.float64)
    return list(
        comm.allgather(
            {
                "rank": int(comm.rank),
                "ownership_range": [int(lo), int(hi)],
                "values_sha256": _sha256_array(local),
            }
        )
    )


def _reference_owned_values(params: dict[str, object], assembler) -> np.ndarray:
    """Return the undeformed reference map in this rank's solver ordering."""

    if "_distributed_u_init_owned" in params:
        return np.asarray(params["_distributed_u_init_owned"], dtype=np.float64)
    free_original = np.asarray(params["u_0_ref"], dtype=np.float64)[
        np.asarray(params["freedofs"], dtype=np.int64)
    ]
    reordered = free_original[np.asarray(assembler.part.perm, dtype=np.int64)]
    return np.asarray(
        reordered[int(assembler.part.lo) : int(assembler.part.hi)],
        dtype=np.float64,
    )


def _reference_elastic_input_identity(
    *,
    args,
    params: dict[str, object],
    assembler,
    initial_state: PETSc.Vec,
    comm: MPI.Comm,
) -> dict[str, object]:
    """Describe the exact mesh/free-space payload used for the reference map."""

    identity: dict[str, object] = {
        "reference_free_state_owned_partitions": _owned_partition_identity(
            initial_state, comm
        ),
    }
    if bool(params.get("_distributed_local_data", False)):
        grid = params["_he_grid"]
        local_array_sha256 = {
            key: _sha256_array(params[key])
            for key in (
                "_distributed_local_elem_idx",
                "_distributed_local_elems_total",
                "_distributed_local_elems_reordered",
                "_distributed_dphix",
                "_distributed_dphiy",
                "_distributed_dphiz",
                "_distributed_vol",
            )
        }
        identity.update(
            {
                "mesh_representation": "rank_local_structured_overlap",
                "structured_grid": {
                    "nx": int(grid.nx),
                    "ny": int(grid.ny),
                    "nz": int(grid.nz),
                    "x_extent": [float(grid.x_min), float(grid.x_max)],
                    "y_extent": [float(grid.y_min), float(grid.y_max)],
                    "z_extent": [float(grid.z_min), float(grid.z_max)],
                },
                "local_payload_by_rank": list(
                    comm.allgather(
                        {
                            "rank": int(comm.rank),
                            "array_sha256": local_array_sha256,
                        }
                    )
                ),
            }
        )
    else:
        permutation = np.asarray(assembler.part.perm, dtype=np.int64)
        identity.update(
            {
                "mesh_representation": "replicated_hdf5",
                "array_sha256": {
                    "nodes": _sha256_array(params["nodes2coord"]),
                    "elements_scalar": _sha256_array(params["elems_scalar"]),
                    "free_dofs": _sha256_array(params["freedofs"]),
                    "free_dof_permutation": _sha256_array(permutation),
                    "dphix": _sha256_array(params["dphix"]),
                    "dphiy": _sha256_array(params["dphiy"]),
                    "dphiz": _sha256_array(params["dphiz"]),
                    "element_weights": _sha256_array(params["vol"]),
                },
            }
        )
    identity["tangent_route"] = {
        "assembler": type(assembler).__name__,
        "assembly_mode": str(args.assembly_mode),
        "local_hessian_mode": str(
            getattr(assembler, "local_hessian_mode", "structural_coloring_hvp")
        ),
        "assembly_backend": str(getattr(assembler, "assembly_backend", "coo")),
        "element_reorder_mode": str(
            getattr(args, "element_reorder_mode", None) or "not_applicable"
        ),
    }
    return identity


def _build_certified_reference_elastic_metric(
    *,
    args,
    operator: PETSc.Mat,
    initial_state: PETSc.Vec,
    expected_free_dofs: int,
    provenance: dict[str, object],
) -> tuple[MatrixRieszMetric, float, dict[str, object]]:
    """Build the certified HyperElasticity reference-energy stopping metric."""

    rows, columns = (int(value) for value in operator.getSize())
    expected_free_dofs = int(expected_free_dofs)
    if rows != expected_free_dofs or columns != expected_free_dofs:
        raise ValueError(
            "Reference elastic operator/free-space mismatch: "
            f"matrix={rows}x{columns}, free_dofs={expected_free_dofs}."
        )
    explicit_scale = _positive_explicit_state_scale(args)
    certificate = certify_spd_by_cholesky(
        operator,
        factor_solver_type=str(
            getattr(args, "riesz_spd_factor_solver_type", "mumps") or "mumps"
        ),
        symmetry_tol=float(getattr(args, "riesz_symmetry_tol", 1.0e-12)),
        options_prefix="hyperelasticity_riesz_spd_certificate_",
    )
    matrix_info = operator.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
    complete_provenance = {
        **dict(provenance),
        "free_space": "interior_dofs_after_both_end_face_dirichlet_constraints",
        "free_dofs": int(expected_free_dofs),
        "matrix_type": str(operator.getType()),
        "matrix_nonzeros": int(matrix_info.get("nz_used", 0.0)),
        "spd_certificate": dict(certificate),
    }

    metric: MatrixRieszMetric | None = None
    try:
        metric = MatrixRieszMetric(
            operator,
            name="hyperelasticity_reference_elastic_energy",
            provenance=complete_provenance,
            ksp_type=str(getattr(args, "riesz_ksp_type", "cg") or "cg"),
            pc_type=str(getattr(args, "riesz_pc_type", "jacobi") or "jacobi"),
            rtol=float(getattr(args, "riesz_ksp_rtol", 1.0e-10)),
            atol=float(getattr(args, "riesz_ksp_atol", 1.0e-14)),
            max_it=int(getattr(args, "riesz_ksp_max_it", 5000)),
            require_symmetric=False,
            true_residual_rtol=float(
                getattr(args, "riesz_true_residual_rtol", 1.0e-8)
            ),
            set_from_options=False,
        )
        initial_reference_norm = float(metric.primal_norm(initial_state).value)
        if not np.isfinite(initial_reference_norm) or initial_reference_norm <= 0.0:
            raise ValueError(
                "The initial deformation map must have a finite, strictly positive "
                "reference elastic-energy norm"
            )
        state_scale = (
            float(initial_reference_norm)
            if explicit_scale is None
            else float(explicit_scale)
        )
        return metric, state_scale, {
            "selection": "reference_elastic_energy",
            "legacy_default": False,
            "state_variable": "deformation_map_y_on_constrained_free_dofs",
            "state_scale": float(state_scale),
            "state_scale_source": (
                "initial_reference_deformation_map_primal_norm"
                if explicit_scale is None
                else "explicit_cli"
            ),
            "correction_normalization": "metric_current_state",
            "initial_state_primal_norm": float(initial_reference_norm),
            "primal_norm_definition": "sqrt(y^T K_ref y)",
            "absolute_dual_residual_definition": "sqrt(g^T K_ref^{-1} g)",
            "absolute_dual_residual_units": (
                "sqrt(discrete_reference_elastic_energy_unit)"
            ),
            "initial_relative_dual_residual_definition": (
                "absolute_dual_residual/initial_absolute_dual_residual_per_load_step"
            ),
            "relative_correction_definition": (
                "primal_correction_norm/max(primal_state_norm,state_scale)"
            ),
            "dimensionless_quantities": [
                "initial_relative_dual_residual",
                "relative_correction",
            ],
            "metric": metric.describe(),
        }
    except Exception:
        if metric is not None:
            metric.destroy()
        raise


def _build_convergence_metric(
    *,
    args,
    assembler,
    params: dict[str, object],
    initial_state: PETSc.Vec,
    use_element_assembly: bool,
    comm: MPI.Comm,
) -> tuple[
    EuclideanMetric | MatrixRieszMetric,
    float,
    dict[str, object],
    PETSc.Mat | None,
]:
    """Construct the selected stopping metric without changing solver geometry."""

    selection = _convergence_metric_name(args)
    explicit_scale = _positive_explicit_state_scale(args)
    if selection == "coefficient_l2":
        state_scale = 1.0 if explicit_scale is None else float(explicit_scale)
        metric = EuclideanMetric()
        return metric, float(state_scale), {
            "selection": "coefficient_l2",
            "legacy_default": bool(explicit_scale is None),
            "state_variable": "deformation_map_y_on_constrained_free_dofs",
            "state_scale": float(state_scale),
            "state_scale_source": (
                "legacy_unit_coefficient" if explicit_scale is None else "explicit_cli"
            ),
            "correction_normalization": "legacy_coefficient",
            "absolute_dual_residual_units": "coefficient_l2",
            "dimensionless_quantities": [
                "initial_relative_dual_residual",
                "relative_correction",
            ],
            "metric": metric.describe(),
        }, None

    reference_setup_start = time.perf_counter()
    reference_owned = np.asarray(
        initial_state.getArray(readonly=True), dtype=np.float64
    ).copy()
    expected_reference_owned = _reference_owned_values(params, assembler)
    local_match = bool(
        reference_owned.shape == expected_reference_owned.shape
        and np.array_equal(reference_owned, expected_reference_owned)
    )
    local_error = (
        np.inf
        if reference_owned.shape != expected_reference_owned.shape
        else (
            float(np.max(np.abs(reference_owned - expected_reference_owned)))
            if reference_owned.size
            else 0.0
        )
    )
    global_match = bool(comm.allreduce(local_match, op=MPI.LAND))
    global_error = float(comm.allreduce(local_error, op=MPI.MAX))
    if not global_match:
        raise ValueError(
            "The reference elastic tangent must be assembled at y(X)=X exactly; "
            f"the current solver state differs by {global_error:.3e}."
        )
    if use_element_assembly:
        assembly_timing = assembler.assemble_hessian(reference_owned)
    else:
        assembly_timing = assembler.assemble_hessian(reference_owned, variant=2)
    operator = assembler.A.copy()
    operator.setBlockSize(3)
    try:
        input_identity = _reference_elastic_input_identity(
            args=args,
            params=params,
            assembler=assembler,
            initial_state=initial_state,
            comm=comm,
        )
        provenance = {
            "problem": "HyperElasticity",
            "operator_source": (
                "exact_discrete_neo_hookean_hessian_at_y_equal_reference_coordinates"
            ),
            "reference_configuration": "y(X)=X (zero displacement)",
            "reference_state_exact_match_verified": True,
            "state_variable": "deformation_map_y",
            "ordering": "assembler_reordered_constrained_free_dofs",
            "ownership": "petsc_distributed_rows",
            "mesh_level": int(args.level),
            "element_degree": int(
                getattr(args, "he_element_degree", getattr(args, "element_degree", 1))
                or 1
            ),
            "material_parameters": {
                "C1": float(params["C1"]),
                "D1": float(params["D1"]),
            },
            "boundary_treatment": (
                "left_and_right_end_faces_eliminated_before_matrix_assembly"
            ),
            "use_abs_det_debug_functional": bool(getattr(args, "use_abs_det", False)),
            "reference_assembly": {
                "elapsed_seconds": float(time.perf_counter() - reference_setup_start),
                "reported_total_seconds": float(assembly_timing.get("total", 0.0)),
                "reported_hvp_count": int(assembly_timing.get("n_hvps", 0)),
            },
            "input_identity": input_identity,
        }
        metric, state_scale, configuration = (
            _build_certified_reference_elastic_metric(
                args=args,
                operator=operator,
                initial_state=initial_state,
                expected_free_dofs=int(assembler.part.n_free),
                provenance=provenance,
            )
        )
        configuration["setup_time_seconds"] = float(
            time.perf_counter() - reference_setup_start
        )
        return metric, state_scale, configuration, operator
    except Exception:
        operator.destroy()
        raise


def run(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    total_runtime_start = time.perf_counter()

    settings = _resolve_linear_settings(args)
    pc_options = _pc_options(settings)
    use_element_assembly = args.assembly_mode == "element"
    element_reorder_mode = str(
        getattr(args, "element_reorder_mode", None) or "block_xyz"
    )
    local_hessian_mode = str(
        getattr(args, "local_hessian_mode", None) or "element"
    )
    problem_build_mode = str(
        getattr(
            args,
            "problem_build_mode",
            "rank_local" if use_element_assembly else "replicated",
        )
        or ("rank_local" if use_element_assembly else "replicated")
    )
    distribution_strategy = str(
        getattr(
            args,
            "distribution_strategy",
            "overlap_p2p" if problem_build_mode == "rank_local" else "overlap_allgather",
        )
        or ("overlap_p2p" if problem_build_mode == "rank_local" else "overlap_allgather")
    )
    assembly_backend = str(
        getattr(
            args,
            "assembly_backend",
            "coo_local" if problem_build_mode == "rank_local" else "coo",
        )
        or ("coo_local" if problem_build_mode == "rank_local" else "coo")
    )
    mesh_source = str(
        getattr(
            args,
            "mesh_source",
            "procedural" if problem_build_mode == "rank_local" else "hdf5",
        )
        or ("procedural" if problem_build_mode == "rank_local" else "hdf5")
    )
    element_degree = int(
        getattr(args, "he_element_degree", getattr(args, "element_degree", 1)) or 1
    )
    if element_degree not in {1, 4}:
        raise ValueError(
            f"HyperElasticity element degree {element_degree!r} is unsupported; "
            "expected 1 or 4"
        )

    mesh_obj = None
    if problem_build_mode == "rank_local":
        if not use_element_assembly:
            raise ValueError("problem_build_mode='rank_local' is supported only for element assembly")
        if distribution_strategy != "overlap_p2p":
            raise ValueError("rank-local HyperElasticity requires distribution_strategy='overlap_p2p'")
        if assembly_backend != "coo_local":
            raise ValueError("rank-local HyperElasticity requires assembly_backend='coo_local'")
        if local_hessian_mode != "element":
            raise ValueError("rank-local HyperElasticity requires local_hessian_mode='element'")
        params, adjacency, u_init = load_rank_local_hyperelasticity(
            int(args.level),
            comm=comm,
            reorder_mode=element_reorder_mode,
            mesh_source=mesh_source,
            element_degree=element_degree,
        )
    elif problem_build_mode == "replicated":
        if element_degree != 1:
            raise ValueError(
                "problem_build_mode='replicated' currently supports only "
                "HyperElasticity element_degree=1"
            )
        if mesh_source != "hdf5":
            raise ValueError(
                "problem_build_mode='replicated' currently supports only "
                "mesh_source='hdf5'"
            )
        mesh_obj = MeshHyperElasticity3D(args.level)
        params, adjacency, u_init = mesh_obj.get_data()
    else:
        raise ValueError(
            f"Unsupported HyperElasticity problem_build_mode={problem_build_mode!r}"
        )

    setup_start = time.perf_counter()
    if use_element_assembly:
        if not args.local_coloring:
            raise ValueError("--assembly_mode element requires --local_coloring")
        assembler = HEReorderedElementAssembler(
            params=params,
            comm=comm,
            adjacency=adjacency,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_type=str(settings["ksp_type"]),
            pc_type=str(settings["pc_type"]),
            ksp_max_it=int(settings["ksp_max_it"]),
            use_near_nullspace=bool(settings["use_near_nullspace"]),
            pc_options=pc_options,
            reorder_mode=element_reorder_mode,
            local_hessian_mode=local_hessian_mode,
            use_abs_det=bool(args.use_abs_det),
            distribution_strategy=distribution_strategy,
            assembly_backend=assembly_backend,
        )
    else:
        assembler_cls = (
            LocalColoringAssembler if args.local_coloring else ParallelDOFHessianAssembler
        )
        assembler_kwargs = dict(
            params=params,
            comm=comm,
            adjacency=adjacency,
            coloring_trials_per_rank=args.coloring_trials,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_type=str(settings["ksp_type"]),
            pc_type=str(settings["pc_type"]),
            ksp_max_it=int(settings["ksp_max_it"]),
            use_near_nullspace=bool(settings["use_near_nullspace"]),
            pc_options=pc_options,
            reorder=bool(settings["reorder"]),
            use_abs_det=bool(args.use_abs_det),
        )
        if args.local_coloring:
            assembler_kwargs["hvp_eval_mode"] = str(args.hvp_eval_mode)
        assembler = assembler_cls(**assembler_kwargs)
        assembler.A.setBlockSize(3)

    # The assembler has extracted its local/owned sparsity pattern by here.
    # Keep the problem arrays in ``params``, but release the replicated mesh
    # wrapper and global HDF5 adjacency before PETSc/JAX setup grows memory.
    del mesh_obj, adjacency
    gc.collect()

    setup_time = time.perf_counter() - setup_start
    local_assembler_setup = assembler.setup_summary() if use_element_assembly else {}
    local_assembler_memory = assembler.memory_summary() if use_element_assembly else {}
    gathered_assembler_setup = comm.gather(local_assembler_setup, root=0)
    gathered_assembler_memory = comm.gather(local_assembler_memory, root=0)
    if rank == 0:
        assembler_memory_report = _summarize_rank_memory(gathered_assembler_memory)
        assembler_setup_report = [
            {"rank": int(idx), **dict(summary)}
            for idx, summary in enumerate(gathered_assembler_setup or [])
        ]
    else:
        assembler_memory_report = {"ranks": 0, "rank_summaries": []}
        assembler_setup_report = []

    state_in = str(getattr(args, "state_in", "") or "").strip()
    if state_in:
        initial_values, initial_state_input = _load_hyperelasticity_initial_state(
            path=state_in,
            args=args,
            params=params,
            assembler=assembler,
            comm=comm,
        )
    elif "_distributed_u_init_owned" in params:
        initial_values = np.asarray(
            params["_distributed_u_init_owned"], dtype=np.float64
        )
        initial_state_input = {
            "source": "solver_default",
            "file_sha256": None,
            "ordering": "owned reordered constrained free DOFs",
            "mesh_identity_verified": True,
        }
    else:
        initial_values = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
        initial_state_input = {
            "source": "solver_default",
            "file_sha256": None,
            "ordering": "global reordered constrained free DOFs",
            "mesh_identity_verified": True,
        }
    x = assembler.create_vec(initial_values)
    initial_state_input["owned_partitions"] = _owned_partition_identity(x, comm)
    x_step_start = x.duplicate()

    ksp = assembler.ksp
    A = assembler.A
    pc = ksp.getPC()
    pmg_hierarchy = None
    pmg_metadata: dict[str, object] | None = None
    if use_element_assembly and str(settings["pc_type"]) == "mg":
        if element_degree != 1:
            raise ValueError(
                "HyperElasticity PCMG currently supports element_degree=1 only. "
                "Use --pc-type gamg for P4 smoke/scaling until the P4->P2->P1 "
                "same-mesh hierarchy is added."
            )
        coarsest_level = choose_he_pmg_coarsest_level(
            finest_level=int(args.level),
            n_ranks=int(nprocs),
            requested=getattr(args, "he_pmg_coarsest_level", 1),
            min_dofs_per_rank=int(getattr(args, "he_pmg_auto_min_dofs_per_rank", 128)),
        )
        t_pmg0 = time.perf_counter()
        pmg_hierarchy = build_he_pmg_hierarchy(
            finest_level=int(args.level),
            coarsest_level=int(coarsest_level),
            reorder_mode=element_reorder_mode,
            comm=comm,
        )
        configure_he_pmg(
            ksp,
            pmg_hierarchy,
            smoother=HEPmgSmootherConfig(
                ksp_type=str(getattr(args, "he_pmg_smoother_ksp_type", "chebyshev")),
                pc_type=str(getattr(args, "he_pmg_smoother_pc_type", "jacobi")),
                steps=int(getattr(args, "he_pmg_smoother_steps", 2)),
            ),
            coarse_ksp_type=(
                None
                if getattr(args, "he_pmg_coarse_ksp_type", None) in {None, ""}
                else str(getattr(args, "he_pmg_coarse_ksp_type"))
            ),
            coarse_pc_type=str(getattr(args, "he_pmg_coarse_pc_type", "hypre")),
            coarse_redundant_number=int(
                getattr(args, "he_pmg_coarse_redundant_number", 0)
            ),
            coarse_telescope_reduction_factor=int(
                getattr(args, "he_pmg_coarse_telescope_reduction_factor", 0)
            ),
            coarse_factor_solver_type=(
                None
                if getattr(args, "he_pmg_coarse_factor_solver_type", None) in {None, ""}
                else str(getattr(args, "he_pmg_coarse_factor_solver_type"))
            ),
            coarse_hypre_nodal_coarsen=int(
                getattr(args, "he_pmg_coarse_hypre_nodal_coarsen", 6)
            ),
            coarse_hypre_vec_interp_variant=int(
                getattr(args, "he_pmg_coarse_hypre_vec_interp_variant", 3)
            ),
            coarse_hypre_strong_threshold=getattr(
                args, "he_pmg_coarse_hypre_strong_threshold", None
            ),
            coarse_hypre_coarsen_type=getattr(
                args, "he_pmg_coarse_hypre_coarsen_type", None
            ),
            coarse_hypre_max_iter=int(
                getattr(args, "he_pmg_coarse_hypre_max_iter", 2)
            ),
            coarse_hypre_tol=float(getattr(args, "he_pmg_coarse_hypre_tol", 0.0)),
            coarse_hypre_relax_type_all=getattr(
                args,
                "he_pmg_coarse_hypre_relax_type_all",
                "symmetric-SOR/Jacobi",
            ),
            galerkin=str(getattr(args, "he_pmg_galerkin", "both")),
        )
        pmg_metadata = dict(pmg_hierarchy.build_metadata)
        pmg_metadata["configure_time"] = float(time.perf_counter() - t_pmg0)
        pmg_metadata["coarsest_level_resolved"] = int(coarsest_level)
        pmg_metadata["coarsest_level_requested"] = str(
            getattr(args, "he_pmg_coarsest_level", 1)
        )
        pmg_metadata["auto_min_dofs_per_rank"] = int(
            getattr(args, "he_pmg_auto_min_dofs_per_rank", 128)
        )
        pmg_metadata["smoother"] = {
            "ksp_type": str(getattr(args, "he_pmg_smoother_ksp_type", "chebyshev")),
            "pc_type": str(getattr(args, "he_pmg_smoother_pc_type", "jacobi")),
            "steps": int(getattr(args, "he_pmg_smoother_steps", 2)),
        }
        pmg_metadata["coarse_solver"] = {
            "ksp_type": str(getattr(args, "he_pmg_coarse_ksp_type", "") or ""),
            "pc_type": str(getattr(args, "he_pmg_coarse_pc_type", "hypre")),
            "redundant_number": int(
                getattr(args, "he_pmg_coarse_redundant_number", 0)
            ),
            "telescope_reduction_factor": int(
                getattr(args, "he_pmg_coarse_telescope_reduction_factor", 0)
            ),
            "factor_solver_type": str(
                getattr(args, "he_pmg_coarse_factor_solver_type", "") or ""
            ),
        }

    gamg_coords = None
    if settings["pc_type"] == "gamg" and settings["gamg_set_coordinates"]:
        if "_distributed_owned_block_coordinates" in params:
            gamg_coords = np.asarray(
                params["_distributed_owned_block_coordinates"], dtype=np.float64
            )
        else:
            gamg_coords = build_gamg_coordinates(
                assembler.part,
                params["freedofs"],
                params["nodes2coord"],
                block_size=3,
            )

    rotation_per_iter = 4.0 * 2.0 * np.pi / float(args.total_steps)
    ls_primary = (float(args.linesearch_a), float(args.linesearch_b))
    line_search = str(getattr(args, "line_search", "golden_fixed"))
    use_trust_region = bool(getattr(args, "use_trust_region", False))
    trust_radius_init = float(getattr(args, "trust_radius_init", 1.0))
    trust_radius_min = float(getattr(args, "trust_radius_min", 1e-8))
    trust_radius_max = float(getattr(args, "trust_radius_max", 1e6))
    trust_shrink = float(getattr(args, "trust_shrink", 0.5))
    trust_expand = float(getattr(args, "trust_expand", 1.5))
    trust_eta_shrink = float(getattr(args, "trust_eta_shrink", 0.05))
    trust_eta_expand = float(getattr(args, "trust_eta_expand", 0.75))
    trust_max_reject = int(getattr(args, "trust_max_reject", 6))
    trust_subproblem_line_search = bool(
        getattr(args, "trust_subproblem_line_search", False)
    )
    step_time_limit_s = getattr(args, "step_time_limit_s", None)
    trust_ksp_subproblem = bool(
        use_trust_region and str(settings["ksp_type"]).lower() in {"stcg", "nash", "gltr"}
    )

    convergence_metric: EuclideanMetric | MatrixRieszMetric | None = None
    reference_elastic_operator: PETSc.Mat | None = None
    try:
        (
            convergence_metric,
            convergence_state_scale,
            convergence_configuration,
            reference_elastic_operator,
        ) = _build_convergence_metric(
            args=args,
            assembler=assembler,
            params=params,
            initial_state=x,
            use_element_assembly=use_element_assembly,
            comm=comm,
        )
    except Exception:
        x_step_start.destroy()
        x.destroy()
        assembler.cleanup()
        if pmg_hierarchy is not None:
            pmg_hierarchy.cleanup()
        raise

    if rank == 0 and not args.quiet:
        print(
            f"HE 3D DOF solver | level={args.level} np={nprocs} profile={args.profile} "
            f"ksp={settings['ksp_type']} pc={settings['pc_type']} setup={setup_time:.3f}s",
            flush=True,
        )

    linear_timing_records: list[dict[str, object]] = []
    linear_iters_this_attempt: list[int] = []
    force_pc_setup_next = True
    used_ksp_rtol = float(settings["ksp_rtol"])
    used_ksp_max_it = int(settings["ksp_max_it"])

    def _assemble_and_solve(vec, rhs, sol, trust_radius=None):
        nonlocal force_pc_setup_next, gamg_coords

        if trust_radius is not None:
            ksp_cg_set_radius(ksp, float(trust_radius))

        t_asm0 = time.perf_counter()
        u_owned = np.array(vec.array[:], dtype=np.float64)
        if use_element_assembly:
            assembler.assemble_hessian(u_owned)
        else:
            assembler.assemble_hessian(u_owned, variant=2)
        asm_total_time = time.perf_counter() - t_asm0

        asm_details = {}
        if assembler.iter_timings:
            asm_details = dict(assembler.iter_timings[-1])
        asm_details["assembly_total_time"] = float(asm_total_time)

        t_setop0 = time.perf_counter()
        ksp.setOperators(A)
        if gamg_coords is not None:
            pc.setCoordinates(gamg_coords)
            gamg_coords = None
        t_setop = time.perf_counter() - t_setop0

        t_tol0 = time.perf_counter()
        ksp.setTolerances(rtol=float(used_ksp_rtol), max_it=int(used_ksp_max_it))
        t_tol = time.perf_counter() - t_tol0

        t_setup0 = time.perf_counter()
        if settings["pc_setup_on_ksp_cap"]:
            if force_pc_setup_next:
                ksp.setUp()
                force_pc_setup_next = False
        else:
            ksp.setUp()
        t_setup = time.perf_counter() - t_setup0

        t_solve0 = time.perf_counter()
        ksp.solve(rhs, sol)
        t_solve = time.perf_counter() - t_solve0
        ksp_its = int(ksp.getIterationNumber())
        linear_iters_this_attempt.append(ksp_its)

        if settings["pc_setup_on_ksp_cap"] and ksp_its >= int(used_ksp_max_it):
            force_pc_setup_next = True

        if args.save_linear_timing:
            record = {
                "assemble_total_time": float(asm_total_time),
                "assemble_p2p_exchange": float(asm_details.get("p2p_exchange", 0.0)),
                "assemble_hvp_compute": float(asm_details.get("hvp_compute", 0.0)),
                "assemble_extraction": float(asm_details.get("extraction", 0.0)),
                "assemble_coo_assembly": float(asm_details.get("coo_assembly", 0.0)),
                "assemble_n_hvps": int(asm_details.get("n_hvps", 0)),
                "setop_time": float(t_setop),
                "set_tolerances_time": float(t_tol),
                "pc_setup_time": float(t_setup),
                "solve_time": float(t_solve),
                "linear_total_time": float(
                    asm_total_time + t_setop + t_tol + t_setup + t_solve
                ),
                "ksp_its": int(ksp_its),
            }
            if trust_radius is not None:
                record["trust_radius"] = float(trust_radius)
            linear_timing_records.append(record)

        return ksp_its

    def hessian_solve_fn(vec, rhs, sol):
        return _assemble_and_solve(vec, rhs, sol, trust_radius=None)

    def trust_subproblem_solve_fn(vec, rhs, sol, trust_radius):
        return _assemble_and_solve(vec, rhs, sol, trust_radius=float(trust_radius))

    attempt_specs = attempts_from_tuples(
        build_retry_attempts(
            retry_on_failure=bool(args.retry_on_failure),
            linesearch_interval=ls_primary,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_max_it=int(settings["ksp_max_it"]),
        )
    )

    def prepare_step(step_ctx):
        if "_distributed_local_data" in params:
            u0_step = local_dirichlet_values_from_reference(params, step_ctx.angle)
        else:
            u0_step = rotate_right_face_from_reference(
                params["u_0_ref"],
                params["nodes2coord"],
                step_ctx.angle,
                params["right_nodes"],
            )
        assembler.update_dirichlet(u0_step)
        x.copy(x_step_start)

    def build_attempts(_step_ctx):
        return attempt_specs

    def solve_attempt(step_ctx, attempt):
        nonlocal force_pc_setup_next, used_ksp_rtol, used_ksp_max_it

        x_step_start.copy(x)
        force_pc_setup_next = True
        linear_iters_this_attempt.clear()
        linear_timing_records.clear()

        used_ksp_rtol = float(attempt.linear_rtol)
        used_ksp_max_it = int(attempt.linear_max_it)

        t0 = time.perf_counter()
        result = newton(
            energy_fn=assembler.energy_fn,
            gradient_fn=assembler.gradient_fn,
            hessian_solve_fn=hessian_solve_fn,
            x=x,
            tolf=float(args.tolf),
            tolg=float(args.tolg),
            tolg_rel=float(args.tolg_rel),
            linesearch_tol=float(args.linesearch_tol),
            linesearch_interval=attempt.linesearch_interval,
            line_search=line_search,
            maxit=int(args.maxit),
            tolx_rel=float(args.tolx_rel),
            tolx_abs=float(args.tolx_abs),
            require_all_convergence=True,
            fail_on_nonfinite=True,
            verbose=(not args.quiet),
            comm=comm,
            ghost_update_fn=None,
            hessian_matvec_fn=lambda _x, vin, vout: assembler.A.mult(vin, vout),
            trust_subproblem_solve_fn=(
                trust_subproblem_solve_fn if trust_ksp_subproblem else None
            ),
            trust_subproblem_line_search=trust_subproblem_line_search,
            save_history=bool(args.save_history),
            trust_region=use_trust_region,
            trust_radius_init=trust_radius_init,
            trust_radius_min=trust_radius_min,
            trust_radius_max=trust_radius_max,
            trust_shrink=trust_shrink,
            trust_expand=trust_expand,
            trust_eta_shrink=trust_eta_shrink,
            trust_eta_expand=trust_eta_expand,
            trust_max_reject=trust_max_reject,
            step_time_limit_s=step_time_limit_s,
            convergence_metric=convergence_metric,
            convergence_state_scale=convergence_state_scale,
        )
        step_ctx.state["step_time_raw"] = time.perf_counter() - t0
        return result, float(step_ctx.state["step_time_raw"])

    def should_retry(result, _step_ctx):
        return needs_solver_repair(result)

    def build_step_record(step_ctx, result, step_time, attempt):
        initial_dual_residual = result.get("initial_dual_residual_norm")
        dual_residual_target = None
        if initial_dual_residual is not None and np.isfinite(
            float(initial_dual_residual)
        ):
            dual_residual_target = max(
                float(args.tolg),
                float(args.tolg_rel) * float(initial_dual_residual),
            )
        dual_residual = float(result["dual_residual_norm"])
        step_record = {
            "step": int(step_ctx.step),
            "angle": float(step_ctx.angle),
            "time": float(round(step_time, 6)),
            "rank_local_time_s": float(step_time),
            "nit": int(result["nit"]),
            "linear_iters": int(sum(linear_iters_this_attempt)),
            "energy": float(result["fun"]),
            "message": str(result["message"]),
            "success": bool(result["success"]),
            "convergence": {
                "metric": dict(result["convergence_metric"]),
                "initial_dual_residual_norm": initial_dual_residual,
                "dual_residual_norm": dual_residual,
                "dual_residual_relative": float(result["dual_residual_relative"]),
                "dual_residual_target": dual_residual_target,
                "dual_residual_gate_pass": bool(
                    dual_residual_target is not None
                    and np.isfinite(dual_residual)
                    and dual_residual < float(dual_residual_target)
                ),
                "dual_residual_metadata": dict(result["dual_residual_metadata"]),
                "coefficient_gradient_l2": float(
                    result["grad_norm_coefficient_l2"]
                ),
                "correction_norm": float(result["correction_norm"]),
                "relative_correction": float(result["relative_correction"]),
                "state_norm": float(result["state_norm"]),
                "state_scale": float(result["convergence_state_scale"]),
                "correction_mode": str(result["convergence_correction_mode"]),
            },
            "attempt": str(attempt.name),
            "ksp_rtol_used": float(attempt.linear_rtol),
            "ksp_max_it_used": int(attempt.linear_max_it),
            "linesearch_interval_used": [
                float(attempt.linesearch_interval[0]),
                float(attempt.linesearch_interval[1]),
            ],
        }
        if step_time_limit_s is not None:
            step_record["step_time_limit_s"] = float(step_time_limit_s)
            step_record["kill_switch_exceeded"] = bool(
                step_time > float(step_time_limit_s)
            )
        if args.save_history:
            step_record["history"] = result.get("history", [])
        if args.save_linear_timing:
            step_record["linear_timing"] = list(linear_timing_records)
        return step_record

    def on_retry(step_ctx, attempt, _attempt_idx, _total_attempts):
        if rank == 0 and not args.quiet:
            print(
                f"Step {step_ctx.step}: retrying with repair settings "
                f"(rtol={float(attempt.linear_rtol):.3e}, "
                f"ksp_max_it={int(attempt.linear_max_it)}, "
                f"ls=[{attempt.linesearch_interval[0]:.3g},"
                f"{attempt.linesearch_interval[1]:.3g}])",
                flush=True,
            )

    def on_step_complete(step_record, _step_ctx):
        if rank == 0 and not args.quiet:
            print(
                f"step={step_record['step']:3d} angle={step_record['angle']:.6f} "
                f"time={step_record['time']:.3f}s nit={step_record['nit']:3d} "
                f"ksp={step_record['linear_iters']:5d} "
                f"energy={step_record['energy']:.6e} "
                f"[{step_record['message']}]",
                flush=True,
            )

    def should_stop(step_record, _result, step_ctx):
        if args.stop_on_fail and "converged" not in step_record["message"].lower():
            if rank == 0 and not args.quiet:
                print(
                    f"Stopping at step {step_ctx.step} due to failure message.",
                    flush=True,
                )
            return True
        if step_time_limit_s is not None and bool(step_record.get("kill_switch_exceeded")):
            if rank == 0 and not args.quiet:
                print(
                    f"Stopping at step {step_ctx.step} because step time "
                    f"{float(step_ctx.state['step_time_raw']):.3f}s exceeded limit "
                    f"{float(step_time_limit_s):.3f}s",
                    flush=True,
                )
            return True
        return False

    step_records = []
    solve_runtime_start = time.perf_counter()
    publication_setup_time = solve_runtime_start - total_runtime_start
    state_output: dict[str, object] | None = None
    endpoint_identity: dict[str, object] | None = None

    try:
        step_records = run_load_steps(
            start_step=int(args.start_step),
            num_steps=int(args.steps),
            rotation_per_step=float(rotation_per_iter),
            prepare_step=prepare_step,
            build_attempts=build_attempts,
            solve_attempt=solve_attempt,
            should_retry=should_retry,
            build_step_record=build_step_record,
            should_stop=should_stop,
            on_retry=on_retry,
            on_step_complete=on_step_complete,
        )
        publication_solve_time = time.perf_counter() - solve_runtime_start
        independent_gradient = x.duplicate()
        try:
            assembler.gradient_fn(x, independent_gradient)
            independent_dual = convergence_metric.dual_norm(independent_gradient)
            endpoint_identity = {
                "owned_reordered_state_sha256": _distributed_vector_sha256(x, comm),
                "independent_residual": {
                    "dual_norm": float(independent_dual.value),
                    "coefficient_l2_norm": float(
                        independent_gradient.norm(PETSc.NormType.NORM_2)
                    ),
                    "owned_reordered_gradient_sha256": _distributed_vector_sha256(
                        independent_gradient, comm
                    ),
                    "evaluation": dict(independent_dual.metadata),
                    "evaluated_after_solver_termination": True,
                },
            }
        finally:
            independent_gradient.destroy()
    finally:
        try:
            state_output = _export_state_if_requested(
                args, assembler, params, x, step_records, comm
            )
        finally:
            if isinstance(convergence_metric, MatrixRieszMetric):
                convergence_metric.destroy()
            if reference_elastic_operator is not None:
                reference_elastic_operator.destroy()
            x_step_start.destroy()
            x.destroy()
            assembler.cleanup()
            if pmg_hierarchy is not None:
                pmg_hierarchy.cleanup()

    resource_usage_report = _summarize_rank_rss(comm)
    publication_first_step_time = (
        float(step_records[0].get("rank_local_time_s", 0.0))
        if step_records
        else 0.0
    )
    if publication_first_step_time <= 0.0:
        publication_first_step_time = float(publication_solve_time)
    publication_timing = _collect_publication_timing(
        comm,
        setup_s=float(publication_setup_time),
        first_step_s=float(publication_first_step_time),
        solve_s=float(publication_solve_time),
        total_s=float(time.perf_counter() - total_runtime_start),
    )
    terminal_convergence = (
        dict(step_records[-1]["convergence"]) if step_records else None
    )

    return build_load_step_result(
        mesh_level=int(args.level),
        total_dofs=int(params.get("_distributed_total_dofs", len(params.get("u_0", [])))),
        setup_time=setup_time,
        total_runtime_start=total_runtime_start,
        steps=step_records,
        extra={
            "free_dofs": int(assembler.part.n_free),
            "publication_timing": publication_timing,
            "convergence_metric_requested": str(_convergence_metric_name(args)),
            "convergence_metric": str(convergence_configuration["selection"]),
            "nonlinear_convergence": {
                "configuration": dict(convergence_configuration),
                "metric": dict(convergence_configuration["metric"]),
                "terminal": terminal_convergence,
                "per_step_records": int(len(step_records)),
            },
            "metadata": {
                "profile": args.profile,
                "nprocs": nprocs,
                "nproc_threads": max(1, int(args.nproc)),
                "initial_state_input": dict(initial_state_input),
                "endpoint_identity": endpoint_identity,
                "state_output": state_output,
                "convergence": dict(convergence_configuration),
                "linear_solver": {
                    "ksp_type": str(settings["ksp_type"]),
                    "pc_type": str(settings["pc_type"]),
                    "ksp_rtol": float(settings["ksp_rtol"]),
                    "ksp_max_it": int(settings["ksp_max_it"]),
                    "pc_setup_on_ksp_cap": bool(settings["pc_setup_on_ksp_cap"]),
                    "gamg_threshold": float(settings["gamg_threshold"]),
                    "gamg_agg_nsmooths": int(settings["gamg_agg_nsmooths"]),
                    "gamg_set_coordinates": bool(settings["gamg_set_coordinates"]),
                    "use_near_nullspace": bool(settings["use_near_nullspace"]),
                    "matrix_block_size": 3,
                    "reorder": bool(settings["reorder"]),
                    "hvp_eval_mode": str(
                        getattr(assembler, "_hvp_eval_mode", "batched")
                    ),
                    "assembly_mode": str(args.assembly_mode),
                    "element_reorder_mode": (
                        element_reorder_mode if use_element_assembly else None
                    ),
                    "local_hessian_mode": (
                        local_hessian_mode if use_element_assembly else None
                    ),
                    "distribution_strategy": str(
                        getattr(assembler, "distribution_strategy", "reduced_free_dofs")
                    ),
                    "assembly_backend": str(getattr(assembler, "assembly_backend", "")),
                    "assembly_backend_requested": str(
                        getattr(assembler, "assembly_backend_requested", "")
                    ),
                    "problem_build_mode": str(problem_build_mode),
                    "mesh_source": str(mesh_source),
                    "element_degree": int(element_degree),
                    "rank_local_formula_layout": bool(
                        getattr(assembler, "_formula_layout", False)
                    ),
                    "pmg_hierarchy": pmg_metadata,
                    "assembler_setup_by_rank": assembler_setup_report,
                    "assembler_memory_by_rank": assembler_memory_report,
                    "resource_usage": resource_usage_report,
                    "assembler": assembler.__class__.__name__,
                    "trust_subproblem_solver": (
                        "petsc_ksp" if trust_ksp_subproblem else "reduced_2d"
                    ),
                    "trust_subproblem_line_search": bool(
                        trust_subproblem_line_search
                    ),
                },
                "newton": {
                    "tolf": float(args.tolf),
                    "tolg": float(args.tolg),
                    "tolg_rel": float(args.tolg_rel),
                    "tolx_rel": float(args.tolx_rel),
                    "tolx_abs": float(args.tolx_abs),
                    "maxit": int(args.maxit),
                    "require_all_convergence": True,
                    "fail_on_nonfinite": True,
                    "linesearch_interval": [
                        float(args.linesearch_a),
                        float(args.linesearch_b),
                    ],
                    "linesearch_tol": float(args.linesearch_tol),
                    "line_search": str(line_search),
                    "trust_region": bool(use_trust_region),
                    "trust_radius_init": float(trust_radius_init),
                    "trust_radius_min": float(trust_radius_min),
                    "trust_radius_max": float(trust_radius_max),
                    "trust_subproblem_line_search": bool(
                        trust_subproblem_line_search
                    ),
                    "step_time_limit_s": (
                        None if step_time_limit_s is None else float(step_time_limit_s)
                    ),
                },
                "load_stepping": {
                    "start_step": int(args.start_step),
                    "steps": int(args.steps),
                    "total_steps": int(args.total_steps),
                    "rotation_per_iter": float(rotation_per_iter),
                    "retry_on_failure": bool(args.retry_on_failure),
                },
            },
        },
    )
