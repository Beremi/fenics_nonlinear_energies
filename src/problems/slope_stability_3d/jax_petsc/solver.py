"""3D heterogeneous slope-stability solver using JAX autodiff + PETSc."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import resource
import time

import jax
import h5py
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.benchmark.run_record import atomic_write_json
from src.core.benchmark.state_export import export_plasticity3d_state_npz
from src.core.petsc.metrics import (
    EuclideanMetric,
    MatrixRieszMetric,
    certify_spd_by_cholesky,
)
from src.core.petsc.minimizers import newton
from src.core.petsc.reasons import ksp_reason_name
from src.core.petsc.trust_ksp import ksp_cg_set_radius
from src.problems.slope_stability_3d.jax_petsc.multigrid import (
    LegacyPMGLevelSmootherConfig,
    attach_pmg_level_metadata,
    build_mixed_pmg_hierarchy,
    configure_pmg,
    mixed_hierarchy_specs,
)
from src.problems.slope_stability_3d.jax_petsc.reordered_element_assembler import (
    SlopeStability3DReorderedElementAssembler,
)
from src.problems.slope_stability_3d.support.mesh import (
    DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    DEFAULT_MESH_NAME,
    PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM,
    base_mesh_name_for_name,
    build_same_mesh_lagrange_case_data,
    ensure_same_mesh_case_hdf5,
    load_same_mesh_case_hdf5_rank_local,
    normalize_tetra_quadrature_rule_id,
    ownership_block_size_3d,
    same_mesh_case_hdf5_path,
    select_reordered_perm_3d,
)
from src.problems.slope_stability_3d.support.reduction import davis_b_reduction_qp


PROFILE_DEFAULTS = {
    "reference": {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-3,
        "ksp_max_it": 200,
        "hypre_nodal_coarsen": -1,
        "hypre_vec_interp_variant": -1,
        "hypre_strong_threshold": None,
        "hypre_coarsen_type": "",
        "hypre_max_iter": 1,
        "hypre_tol": 0.0,
        "hypre_relax_type_all": "",
        "pc_setup_on_ksp_cap": False,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
    },
    "performance": {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-2,
        "ksp_max_it": 80,
        "hypre_nodal_coarsen": -1,
        "hypre_vec_interp_variant": -1,
        "hypre_strong_threshold": None,
        "hypre_coarsen_type": "",
        "hypre_max_iter": 1,
        "hypre_tol": 0.0,
        "hypre_relax_type_all": "",
        "pc_setup_on_ksp_cap": False,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
    },
}


def _write_progress_payload(path: str | Path, payload: dict[str, object]) -> None:
    atomic_write_json(path, payload, nonfinite_as_null=True)


@contextmanager
def _petsc_stage(name: str, enabled: bool):
    if not bool(enabled):
        yield
        return
    stage = PETSc.Log.Stage(str(name))
    stage.push()
    try:
        yield
    finally:
        stage.pop()


def _configure_petsc_logging(args, comm: MPI.Comm) -> str:
    log_view_path = str(getattr(args, "petsc_log_view_path", "") or "").strip()
    if not (bool(getattr(args, "enable_petsc_log_events", False)) or log_view_path):
        return ""
    PETSc.Log.begin()
    if log_view_path:
        target = Path(log_view_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(target)
    return ""


def _flush_petsc_log_view(path: str, comm: MPI.Comm) -> None:
    if not str(path).strip():
        return
    viewer = PETSc.Viewer().createASCII(str(path), mode="w", comm=comm)
    try:
        PETSc.Log.view(viewer)
    finally:
        viewer.destroy()


@contextmanager
def _jax_trace_context(trace_dir: str, *, rank: int):
    root = str(trace_dir or "").strip()
    if not root:
        yield
        return
    rank_dir = Path(root) / f"rank{int(rank):03d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(rank_dir)):
        yield


def _parse_p4_chunk_size_arg(value: object) -> tuple[str, int]:
    raw = str(value if value is not None else "32").strip()
    if raw.lower() == "auto":
        return "auto", 32
    return "fixed", max(1, int(raw))


def _parse_chunk_candidates(value: object) -> tuple[int, ...]:
    raw = str(value if value is not None else "32,64,128,256").strip()
    if not raw:
        return (32, 64, 128, 256)
    candidates = tuple(
        max(1, int(part.strip()))
        for part in raw.split(",")
        if str(part).strip()
    )
    if not candidates:
        raise ValueError("p4 chunk autotune candidate list may not be empty")
    return candidates


def _format_gib(value: object) -> str:
    return f"{float(value):.3f} GiB"


def _process_memory_snapshot() -> dict[str, float | int]:
    rss_current_bytes = 0
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    rss_current_bytes = int(parts[1]) * 1024
                break
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_hwm_bytes = int(getattr(usage, "ru_maxrss", 0)) * 1024
    return {
        "rss_current_bytes": int(rss_current_bytes),
        "rss_hwm_bytes": int(rss_hwm_bytes),
        "rss_current_gib": float(rss_current_bytes) / float(1024**3),
        "rss_hwm_gib": float(rss_hwm_bytes) / float(1024**3),
    }


def _aggregate_iteration_memory(
    gathered_memory: list[dict[str, object]] | None,
) -> dict[str, object]:
    records = list(gathered_memory or [])
    if not records:
        return {}
    current_vals = [float(rec.get("rss_current_bytes", 0.0)) for rec in records]
    hwm_vals = [float(rec.get("rss_hwm_bytes", 0.0)) for rec in records]
    current_rank = int(np.argmax(current_vals))
    hwm_rank = int(np.argmax(hwm_vals))
    return {
        "rss_current_max_gib": float(max(current_vals)) / float(1024**3),
        "rss_current_mean_gib": float(np.mean(current_vals)) / float(1024**3),
        "rss_current_min_gib": float(min(current_vals)) / float(1024**3),
        "rss_hwm_max_gib": float(max(hwm_vals)) / float(1024**3),
        "rss_hwm_mean_gib": float(np.mean(hwm_vals)) / float(1024**3),
        "rss_hwm_min_gib": float(min(hwm_vals)) / float(1024**3),
        "rss_current_worst_rank": int(current_rank),
        "rss_hwm_worst_rank": int(hwm_rank),
        "per_rank": [
            {
                "rank": int(idx),
                "rss_current_gib": float(rec.get("rss_current_gib", 0.0)),
                "rss_hwm_gib": float(rec.get("rss_hwm_gib", 0.0)),
            }
            for idx, rec in enumerate(records)
        ],
    }


def _aggregate_iteration_linear(
    gathered_linear: list[dict[str, object] | None] | None,
) -> dict[str, object]:
    records = [dict(rec) for rec in list(gathered_linear or []) if rec is not None]
    if not records:
        return {}
    assemble_vals = [float(rec.get("t_assemble", 0.0)) for rec in records]
    setup_vals = [float(rec.get("t_setup", 0.0)) for rec in records]
    solve_vals = [float(rec.get("t_solve", 0.0)) for rec in records]
    ksp_vals = [int(rec.get("ksp_its", 0)) for rec in records]
    true_rel_vals = [float(rec.get("true_relative_residual", 0.0)) for rec in records]
    return {
        "ranks": int(len(records)),
        "t_assemble_max": float(max(assemble_vals)),
        "t_assemble_mean": float(np.mean(assemble_vals)),
        "t_setup_max": float(max(setup_vals)),
        "t_setup_mean": float(np.mean(setup_vals)),
        "t_solve_max": float(max(solve_vals)),
        "t_solve_mean": float(np.mean(solve_vals)),
        "ksp_its_max": int(max(ksp_vals)),
        "ksp_its_mean": float(np.mean(ksp_vals)),
        "ksp_its_sum": int(sum(ksp_vals)),
        "true_relative_residual_max": float(max(true_rel_vals)),
        "true_relative_residual_mean": float(np.mean(true_rel_vals)),
        "accepted_via_true_residual_any": bool(
            any(bool(rec.get("accepted_via_true_residual", False)) for rec in records)
        ),
        "accepted_via_maxit_direction_any": bool(
            any(bool(rec.get("accepted_via_maxit_direction", False)) for rec in records)
        ),
        "per_rank": [
            {
                "rank": int(idx),
                "ksp_its": int(rec.get("ksp_its", 0)),
                "t_assemble": float(rec.get("t_assemble", 0.0)),
                "t_setup": float(rec.get("t_setup", 0.0)),
                "t_solve": float(rec.get("t_solve", 0.0)),
                "true_relative_residual": float(
                    rec.get("true_relative_residual", 0.0)
                ),
                "ksp_reason_name": str(rec.get("ksp_reason_name", "")),
            }
            for idx, rec in enumerate(records)
        ],
    }


@dataclass(frozen=True)
class _LinearSolveFailure(RuntimeError):
    reason_code: int
    reason_name: str
    ksp_its: int
    true_residual_norm: float
    true_relative_residual: float

    def __str__(self) -> str:
        return (
            "Linear solve failed with "
            f"{self.reason_name} after {self.ksp_its} iterations "
            f"(true rel residual={self.true_relative_residual:.3e})"
        )


def _resolve_linear_settings(args):
    settings = dict(PROFILE_DEFAULTS[str(args.profile)])
    overrides = {
        "ksp_type": getattr(args, "ksp_type", None),
        "pc_type": getattr(args, "pc_type", None),
        "ksp_rtol": getattr(args, "ksp_rtol", None),
        "ksp_max_it": getattr(args, "ksp_max_it", None),
        "pc_setup_on_ksp_cap": getattr(args, "pc_setup_on_ksp_cap", None),
        "hypre_nodal_coarsen": getattr(args, "hypre_nodal_coarsen", None),
        "hypre_vec_interp_variant": getattr(args, "hypre_vec_interp_variant", None),
        "hypre_strong_threshold": getattr(args, "hypre_strong_threshold", None),
        "hypre_coarsen_type": getattr(args, "hypre_coarsen_type", None),
        "hypre_max_iter": getattr(args, "hypre_max_iter", None),
        "hypre_tol": getattr(args, "hypre_tol", None),
        "hypre_relax_type_all": getattr(args, "hypre_relax_type_all", None),
        "gamg_threshold": getattr(args, "gamg_threshold", None),
        "gamg_agg_nsmooths": getattr(args, "gamg_agg_nsmooths", None),
        "use_near_nullspace": getattr(args, "use_near_nullspace", None),
        "gamg_set_coordinates": getattr(args, "gamg_set_coordinates", None),
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings


def _pc_options(settings):
    opts = {}
    if str(settings["pc_type"]) == "gamg":
        opts["pc_gamg_threshold"] = float(settings["gamg_threshold"])
        opts["pc_gamg_agg_nsmooths"] = int(settings["gamg_agg_nsmooths"])
    if str(settings["pc_type"]) == "mg":
        opts["pc_mg_galerkin"] = "both"
    if str(settings["pc_type"]) == "hypre":
        opts["pc_hypre_type"] = "boomeramg"
        if int(settings["hypre_nodal_coarsen"]) >= 0:
            opts["pc_hypre_boomeramg_nodal_coarsen"] = int(settings["hypre_nodal_coarsen"])
        if int(settings["hypre_vec_interp_variant"]) >= 0:
            opts["pc_hypre_boomeramg_vec_interp_variant"] = int(
                settings["hypre_vec_interp_variant"]
            )
        if settings["hypre_strong_threshold"] is not None:
            opts["pc_hypre_boomeramg_strong_threshold"] = float(
                settings["hypre_strong_threshold"]
            )
        if str(settings["hypre_coarsen_type"]):
            opts["pc_hypre_boomeramg_coarsen_type"] = str(settings["hypre_coarsen_type"])
        if int(settings["hypre_max_iter"]) >= 0:
            opts["pc_hypre_boomeramg_max_iter"] = int(settings["hypre_max_iter"])
        if settings["hypre_tol"] is not None:
            opts["pc_hypre_boomeramg_tol"] = float(settings["hypre_tol"])
        if str(settings["hypre_relax_type_all"]):
            opts["pc_hypre_boomeramg_relax_type_all"] = str(
                settings["hypre_relax_type_all"]
            )
    return opts


def _load_problem_data(args, comm: MPI.Comm):
    mesh_name = str(getattr(args, "mesh_name", None) or DEFAULT_MESH_NAME)
    degree = int(args.elem_degree)
    constraint_variant = str(
        getattr(args, "constraint_variant", DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT)
    )
    quadrature_rule_id = normalize_tetra_quadrature_rule_id(
        getattr(args, "quadrature_rule", None),
        element_degree=degree,
    )
    ensure_same_mesh_case_hdf5(
        mesh_name,
        degree,
        constraint_variant=constraint_variant,
        quadrature_rule_id=quadrature_rule_id,
    )

    build_mode = str(getattr(args, "problem_build_mode", "root_bcast"))
    reorder_mode = str(getattr(args, "element_reorder_mode", None) or "block_xyz")
    if build_mode == "rank_local":
        params = load_same_mesh_case_hdf5_rank_local(
            mesh_name,
            degree,
            constraint_variant=constraint_variant,
            quadrature_rule_id=quadrature_rule_id,
            reorder_mode=reorder_mode,
            comm=comm,
            block_size=3,
        )
        adjacency = None
    else:
        case_data = build_same_mesh_lagrange_case_data(
            mesh_name,
            degree=degree,
            constraint_variant=constraint_variant,
            quadrature_rule_id=quadrature_rule_id,
            build_mode=build_mode,
            comm=comm,
        )
        params = dict(case_data.__dict__)
        adjacency = case_data.adjacency

    params["elem_type"] = f"P{degree}"
    params["element_degree"] = int(degree)
    return mesh_name, params, adjacency


def _apply_strength_reduction(params: dict[str, object], lambda_target: float) -> None:
    if "_distributed_c0_q" in params:
        c_bar_q, sin_phi_q = davis_b_reduction_qp(
            np.asarray(params["_distributed_c0_q"], dtype=np.float64),
            np.asarray(params["_distributed_phi_q"], dtype=np.float64),
            np.asarray(params["_distributed_psi_q"], dtype=np.float64),
            float(lambda_target),
        )
        params["_distributed_c_bar_q"] = np.asarray(c_bar_q, dtype=np.float64)
        params["_distributed_sin_phi_q"] = np.asarray(sin_phi_q, dtype=np.float64)
    else:
        c_bar_q, sin_phi_q = davis_b_reduction_qp(
            np.asarray(params["c0_q"], dtype=np.float64),
            np.asarray(params["phi_q"], dtype=np.float64),
            np.asarray(params["psi_q"], dtype=np.float64),
            float(lambda_target),
        )
        params["c_bar_q"] = np.asarray(c_bar_q, dtype=np.float64)
        params["sin_phi_q"] = np.asarray(sin_phi_q, dtype=np.float64)


def _build_gamg_coordinates_owned_blocks(
    assembler: SlopeStability3DReorderedElementAssembler,
    params: dict[str, object],
) -> np.ndarray:
    if int(getattr(assembler, "block_size", 1)) != 3:
        return np.empty((0, 3), dtype=np.float64)
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    nodes = np.asarray(params["nodes"], dtype=np.float64)
    owned_orig_free = np.asarray(
        assembler.layout.perm[assembler.layout.lo : assembler.layout.hi],
        dtype=np.int64,
    )
    owned_total_dofs = np.asarray(freedofs[owned_orig_free], dtype=np.int64)
    if owned_total_dofs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    owned_total_dofs = owned_total_dofs.reshape((-1, 3))
    node_ids = owned_total_dofs[:, 0] // 3
    return np.asarray(nodes[node_ids], dtype=np.float64)


def _legacy_mg_settings(args) -> dict[str, LegacyPMGLevelSmootherConfig]:
    def _cfg(
        ksp_type: str | None,
        pc_type: str | None,
        steps: int | None,
        *,
        default_pc: str = "sor",
        default_steps: int = 3,
    ) -> LegacyPMGLevelSmootherConfig:
        return LegacyPMGLevelSmootherConfig(
            ksp_type=str(ksp_type or "richardson"),
            pc_type=str(pc_type or default_pc),
            steps=int(steps if steps is not None else default_steps),
        )

    degree4_cfg = _cfg(
        getattr(args, "mg_p4_smoother_ksp_type", None),
        getattr(args, "mg_p4_smoother_pc_type", None),
        getattr(args, "mg_p4_smoother_steps", None),
        default_pc="sor",
        default_steps=3,
    )
    degree2_cfg = _cfg(
        getattr(args, "mg_p2_smoother_ksp_type", None),
        getattr(args, "mg_p2_smoother_pc_type", None),
        getattr(args, "mg_p2_smoother_steps", None),
        default_pc="sor",
        default_steps=3,
    )
    degree1_cfg = _cfg(
        getattr(args, "mg_p1_smoother_ksp_type", None),
        getattr(args, "mg_p1_smoother_pc_type", None),
        getattr(args, "mg_p1_smoother_steps", None),
        default_pc="sor",
        default_steps=3,
    )
    fine_degree = int(getattr(args, "elem_degree", 2))
    if fine_degree == 4:
        fine_cfg = degree4_cfg
    elif fine_degree == 2:
        fine_cfg = degree2_cfg
    else:
        fine_cfg = degree1_cfg
    return {
        "fine": fine_cfg,
        "degree2": degree2_cfg,
        "degree1": degree1_cfg,
    }


def _resolve_mg_strategy(args) -> str:
    strategy = str(getattr(args, "mg_strategy", "auto") or "auto")
    if strategy != "auto":
        return strategy
    degree = int(getattr(args, "elem_degree", 2))
    mesh_name = str(getattr(args, "mesh_name", DEFAULT_MESH_NAME) or DEFAULT_MESH_NAME)
    if degree == 1:
        if base_mesh_name_for_name(mesh_name) != mesh_name:
            return "uniform_refined_p1_chain"
        raise ValueError("3D PMG requires a refined mesh name for degree-1 auto strategy")
    if degree == 2:
        return "same_mesh_p2_p1"
    if degree == 4:
        if base_mesh_name_for_name(mesh_name) != mesh_name:
            return "uniform_refined_p4_p2_p1_p1"
        return "same_mesh_p4_p2_p1"
    raise ValueError("3D PMG requires a fine degree of 2 or 4")


def _apply_3d_stack_defaults(args, settings: dict[str, object]) -> dict[str, object]:
    resolved = dict(settings)
    elem_degree = int(getattr(args, "elem_degree", 2))
    if getattr(args, "pc_type", None) is None and elem_degree in {2, 4}:
        resolved["pc_type"] = "mg"
    if getattr(args, "ksp_type", None) is None:
        if str(resolved["pc_type"]) == "mg":
            resolved["ksp_type"] = "fgmres"
        elif str(resolved["pc_type"]) == "hypre":
            resolved["ksp_type"] = "cg"
    return resolved


def _solve_elastic_initial_guess(
    *,
    assembler: SlopeStability3DReorderedElementAssembler,
    settings: dict[str, object],
    args,
    mg_hierarchy,
    mg_nullspace_meta,
    gamg_coords: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, object] | None, dict[str, object] | None, np.ndarray | None]:
    zero_owned = np.zeros(assembler.layout.hi - assembler.layout.lo, dtype=np.float64)
    assembler.assemble_hessian_with_mode(zero_owned, constitutive_mode="elastic")
    elastic_ksp = PETSc.KSP().create(assembler.comm)
    elastic_ksp.setType(str(settings["ksp_type"]))
    elastic_ksp.getPC().setType(str(settings["pc_type"]))
    if str(settings["pc_type"]) == "mg":
        configure_pmg(
            elastic_ksp,
            mg_hierarchy,
            level_smoothers=_legacy_mg_settings(args),
            coarse_backend=str(getattr(args, "mg_coarse_backend", None) or "hypre"),
            coarse_ksp_type=str(getattr(args, "mg_coarse_ksp_type", None) or "cg"),
            coarse_pc_type=str(getattr(args, "mg_coarse_pc_type", None) or "hypre"),
            coarse_hypre_nodal_coarsen=int(
                getattr(args, "mg_coarse_hypre_nodal_coarsen", 6)
            ),
            coarse_hypre_vec_interp_variant=int(
                getattr(args, "mg_coarse_hypre_vec_interp_variant", 3)
            ),
            coarse_hypre_strong_threshold=getattr(
                args, "mg_coarse_hypre_strong_threshold", 0.5
            ),
            coarse_hypre_coarsen_type=str(
                getattr(args, "mg_coarse_hypre_coarsen_type", None) or "HMIS"
            ),
            coarse_hypre_max_iter=int(getattr(args, "mg_coarse_hypre_max_iter", 2)),
            coarse_hypre_tol=float(getattr(args, "mg_coarse_hypre_tol", 0.0)),
            coarse_hypre_relax_type_all=str(
                getattr(args, "mg_coarse_hypre_relax_type_all", "symmetric-SOR/Jacobi")
            ),
        )
    elastic_ksp.setOperators(assembler.A)
    if gamg_coords is not None and int(np.asarray(gamg_coords).size) > 0:
        elastic_ksp.getPC().setCoordinates(np.asarray(gamg_coords, dtype=np.float64))
    elastic_ksp.setTolerances(
        rtol=float(settings["ksp_rtol"]),
        max_it=int(settings["ksp_max_it"]),
    )
    elastic_ksp.setFromOptions()
    elastic_ksp.setUp()
    if mg_hierarchy is not None:
        mg_nullspace_meta = attach_pmg_level_metadata(
            elastic_ksp,
            mg_hierarchy,
            use_near_nullspace=bool(settings["use_near_nullspace"]),
            coarse_pc_type=str(getattr(args, "mg_coarse_pc_type", None) or "hypre"),
            coarse_hypre_nodal_coarsen=int(
                getattr(args, "mg_coarse_hypre_nodal_coarsen", 6)
            ),
            coarse_hypre_vec_interp_variant=int(
                getattr(args, "mg_coarse_hypre_vec_interp_variant", 3)
            ),
            coarse_hypre_strong_threshold=getattr(
                args, "mg_coarse_hypre_strong_threshold", 0.5
            ),
            coarse_hypre_coarsen_type=str(
                getattr(args, "mg_coarse_hypre_coarsen_type", None) or "HMIS"
            ),
            coarse_hypre_max_iter=int(getattr(args, "mg_coarse_hypre_max_iter", 2)),
            coarse_hypre_tol=float(getattr(args, "mg_coarse_hypre_tol", 0.0)),
            coarse_hypre_relax_type_all=str(
                getattr(args, "mg_coarse_hypre_relax_type_all", "symmetric-SOR/Jacobi")
            ),
        )

    rhs = assembler.create_vec()
    sol = assembler.create_vec()
    rhs.array[:] = np.asarray(assembler._f_owned, dtype=np.float64)
    rhs.assemble()
    t0 = time.perf_counter()
    elastic_ksp.solve(rhs, sol)
    solve_time = time.perf_counter() - t0
    reason_code = int(elastic_ksp.getConvergedReason())
    reason_name = str(ksp_reason_name(reason_code))
    residual_norm = float(elastic_ksp.getResidualNorm())
    ksp_its = int(elastic_ksp.getIterationNumber())
    rhs_norm = float(rhs.norm(PETSc.NormType.NORM_2))
    success = bool(reason_code > 0 and np.all(np.isfinite(np.asarray(sol.array[:], dtype=np.float64))))
    result = np.asarray(sol.array[:], dtype=np.float64).copy() if success else np.zeros_like(
        np.asarray(sol.array[:], dtype=np.float64)
    )
    meta = {
        "enabled": True,
        "success": bool(success),
        "ksp_type": str(settings["ksp_type"]),
        "pc_type": str(settings["pc_type"]),
        "ksp_iterations": int(ksp_its),
        "ksp_reason": reason_name,
        "ksp_reason_code": int(reason_code),
        "rhs_norm": float(rhs_norm),
        "residual_norm": float(residual_norm),
        "solve_time": float(solve_time),
        "vector_norm": float(np.linalg.norm(result)),
    }
    if not success:
        meta["message"] = (
            "Elastic initial-guess solve failed with "
            f"{reason_name} after {ksp_its} iterations"
        )
    rhs.destroy()
    sol.destroy()
    elastic_ksp.destroy()
    return result, mg_nullspace_meta, meta, gamg_coords


def _load_initial_state_guess(
    args,
    *,
    params: dict[str, object],
    freedofs: np.ndarray,
    perm: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, object] | None]:
    path_raw = str(getattr(args, "initial_state", "") or "").strip()
    if not path_raw:
        return None, None

    path = Path(path_raw)
    if not path.exists():
        raise FileNotFoundError(f"Initial-state file not found: {path}")

    state = np.load(path)
    try:
        if "displacement" in state.files:
            disp = np.asarray(state["displacement"], dtype=np.float64)
        elif "coords_ref" in state.files and "coords_final" in state.files:
            disp = np.asarray(state["coords_final"], dtype=np.float64) - np.asarray(
                state["coords_ref"], dtype=np.float64
            )
        else:
            raise ValueError(
                f"{path} must contain either 'displacement' or both 'coords_ref' and 'coords_final'"
            )

        disp = np.asarray(disp, dtype=np.float64)
        nodes = np.asarray(params["nodes"], dtype=np.float64)
        if disp.shape != nodes.shape:
            raise ValueError(
                f"{path} displacement shape {disp.shape} does not match mesh nodes {nodes.shape}"
            )
        if not np.all(np.isfinite(disp)):
            raise ValueError(f"{path} contains non-finite displacement values")

        coords_ref_max_abs_diff = None
        if "coords_ref" in state.files:
            coords_ref = np.asarray(state["coords_ref"], dtype=np.float64)
            if coords_ref.shape != nodes.shape:
                raise ValueError(
                    f"{path} coords_ref shape {coords_ref.shape} does not match mesh nodes {nodes.shape}"
                )
            coords_ref_max_abs_diff = float(np.max(np.abs(coords_ref - nodes)))

        full = np.asarray(params["u_0"], dtype=np.float64).copy()
        flat = disp.reshape(-1)
        full[np.asarray(freedofs, dtype=np.int64)] = flat[np.asarray(freedofs, dtype=np.int64)]
        owned = np.asarray(full[np.asarray(freedofs, dtype=np.int64)], dtype=np.float64)
        reordered = np.asarray(owned[np.asarray(perm, dtype=np.int64)], dtype=np.float64)
        meta = {
            "enabled": True,
            "success": True,
            "source": "state_npz",
            "path": str(path),
            "vector_norm": float(np.linalg.norm(reordered)),
        }
        if coords_ref_max_abs_diff is not None:
            meta["coords_ref_max_abs_diff"] = float(coords_ref_max_abs_diff)
        return reordered, meta
    finally:
        state.close()


def _should_use_elastic_initial_guess(args, settings: dict[str, object]) -> bool:
    if str(getattr(args, "initial_state", "") or "").strip():
        return False
    explicit = getattr(args, "elastic_initial_guess", None)
    if explicit is not None:
        return bool(explicit)
    return (
        str(settings["pc_type"]) == "mg"
        and str(settings["ksp_type"]).lower() == "fgmres"
        and not bool(getattr(args, "use_trust_region", False))
    )


def _newton_regularization_settings(args) -> dict[str, object]:
    return {
        "enabled": bool(getattr(args, "regularized_newton_tangent", True)),
        "r_min": float(getattr(args, "newton_r_min", 1.0e-4)),
        "r_initial": float(getattr(args, "newton_r_initial", 1.0)),
        "r_max": float(getattr(args, "newton_r_max", 2.0)),
        "fail_growth": float(getattr(args, "newton_r_fail_growth", 2.0)),
        "small_alpha_growth": float(
            getattr(args, "newton_r_small_alpha_growth", 2.0 ** 0.25)
        ),
        "decay": float(getattr(args, "newton_r_decay", 2.0 ** 0.5)),
        "retry_max": int(getattr(args, "newton_r_retry_max", 16)),
        "alpha_increase_threshold": 1.0e-1,
        "alpha_decrease_threshold": 0.5,
    }


def _init_newton_regularization_state(args) -> dict[str, object]:
    settings = _newton_regularization_settings(args)
    r_min = max(float(settings["r_min"]), 0.0)
    r_initial = max(float(settings["r_initial"]), r_min)
    r_max = max(float(settings["r_max"]), r_min)
    return {
        "enabled": bool(settings["enabled"]),
        "r": float(min(r_initial, r_max)),
        "r_min": float(r_min),
        "r_initial": float(min(r_initial, r_max)),
        "r_max": float(r_max),
        "fail_growth": float(max(settings["fail_growth"], 1.0)),
        "small_alpha_growth": float(max(settings["small_alpha_growth"], 1.0)),
        "decay": float(max(settings["decay"], 1.0)),
        "retry_max": int(max(settings["retry_max"], 0)),
        "alpha_increase_threshold": float(settings["alpha_increase_threshold"]),
        "alpha_decrease_threshold": float(settings["alpha_decrease_threshold"]),
        "elastic_operator": None,
        "history": [],
        "last_step": None,
    }


def _convergence_metric_name(args) -> str:
    name = str(getattr(args, "convergence_metric", "coefficient_l2") or "coefficient_l2")
    if name not in {"coefficient_l2", "reference_elastic_energy"}:
        raise ValueError(f"Unsupported Plasticity3D convergence metric {name!r}")
    return name


def _elastic_operator_required(args, regularization_state: dict[str, object]) -> bool:
    return bool(regularization_state["enabled"]) or (
        _convergence_metric_name(args) == "reference_elastic_energy"
    )


def _capture_elastic_operator(
    assembler: SlopeStability3DReorderedElementAssembler,
    regularization_state: dict[str, object],
) -> None:
    elastic_operator = regularization_state.get("elastic_operator")
    if elastic_operator is not None:
        return
    copied = assembler.A.copy()
    if int(getattr(assembler, "block_size", 1)) > 1:
        copied.setBlockSize(int(getattr(assembler, "block_size", 1)))
    regularization_state["elastic_operator"] = copied


def _reference_material_range(
    params: dict[str, object],
    key: str,
    *,
    comm: MPI.Comm,
) -> dict[str, float]:
    source_key = key if key in params else f"_distributed_{key}"
    if source_key not in params:
        raise KeyError(f"Plasticity3D metric provenance is missing {key!r}")
    values = np.asarray(params[source_key], dtype=np.float64)
    local_minimum = float(np.min(values)) if values.size else np.inf
    local_maximum = float(np.max(values)) if values.size else -np.inf
    return {
        "minimum": float(comm.allreduce(local_minimum, op=MPI.MIN)),
        "maximum": float(comm.allreduce(local_maximum, op=MPI.MAX)),
    }


def _sha256_array(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(int(value) for value in array.shape)).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _sha256_hdf5_dataset(dataset: h5py.Dataset) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset.dtype).encode("utf-8"))
    digest.update(str(tuple(int(value) for value in dataset.shape)).encode("utf-8"))
    if dataset.ndim == 0:
        digest.update(np.ascontiguousarray(dataset[()]).view(np.uint8))
        return digest.hexdigest()
    block_rows = max(1, min(int(dataset.shape[0]), 4096))
    for start in range(0, int(dataset.shape[0]), block_rows):
        stop = min(int(dataset.shape[0]), start + block_rows)
        digest.update(np.ascontiguousarray(dataset[start:stop]).view(np.uint8))
    return digest.hexdigest()


def _reference_elastic_input_identity(
    *,
    args,
    params: dict[str, object],
    assembler: SlopeStability3DReorderedElementAssembler,
    comm: MPI.Comm,
) -> dict[str, object]:
    mesh_name = str(params.get("mesh_name", getattr(args, "mesh_name", "")))
    degree = int(getattr(args, "elem_degree", 0))
    constraint_variant = str(params["constraint_variant"])
    quadrature_rule_id = str(params["quadrature_rule_id"])
    input_path = same_mesh_case_hdf5_path(
        mesh_name,
        degree,
        constraint_variant,
        quadrature_rule_id=quadrature_rule_id,
    )
    root_dataset_identity = None
    if int(comm.rank) == 0:
        with h5py.File(input_path, "r") as handle:
            root_dataset_identity = {
                "path": str(input_path),
                "size_bytes": int(input_path.stat().st_size),
                "dataset_sha256": {
                    key: _sha256_hdf5_dataset(handle[key])
                    for key in ("shear_q", "bulk_q", "lame_q", "quad_weight")
                },
            }
    hdf5_identity = comm.bcast(root_dataset_identity, root=0)
    return {
        "hdf5": dict(hdf5_identity),
        "array_sha256": {
            "nodes": _sha256_array(np.asarray(params["nodes"], dtype=np.float64)),
            "elements_scalar": _sha256_array(
                np.asarray(params["elems_scalar"], dtype=np.int64)
            ),
            "material_id": _sha256_array(
                np.asarray(params["material_id"], dtype=np.int64)
            ),
            "free_dofs": _sha256_array(
                np.asarray(params["freedofs"], dtype=np.int64)
            ),
            "free_mask": _sha256_array(np.asarray(params["q_mask"], dtype=bool)),
            "free_dof_permutation": _sha256_array(
                np.asarray(assembler.layout.perm, dtype=np.int64)
            ),
        },
        "tangent_route": {
            "constitutive_mode": "elastic",
            "autodiff_tangent_mode": str(assembler.autodiff_tangent_mode),
            "local_hessian_mode": str(assembler.local_hessian_mode),
            "assembly_backend": str(assembler.assembly_backend),
        },
    }


def _build_certified_reference_elastic_metric(
    *,
    args,
    operator: PETSc.Mat,
    initial_state: PETSc.Vec,
    constraint_variant: str,
    expected_free_dofs: int,
    provenance: dict[str, object],
) -> tuple[MatrixRieszMetric, float, dict[str, object]]:
    """Build the shared Plasticity3D elastic Riesz stopping contract."""

    constraint_variant = str(constraint_variant)
    if constraint_variant != PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM:
        raise ValueError(
            "The reference elastic-energy convergence metric is certified only "
            "for the glued_bottom constrained free space; got "
            f"{constraint_variant!r}."
        )
    rows, columns = (int(value) for value in operator.getSize())
    expected_free_dofs = int(expected_free_dofs)
    if rows != expected_free_dofs or columns != expected_free_dofs:
        raise ValueError(
            "Reference elastic operator/free-space mismatch: "
            f"matrix={rows}x{columns}, free_dofs={expected_free_dofs}."
        )

    explicit_scale = getattr(args, "convergence_state_scale", None)
    if explicit_scale is not None:
        explicit_scale = float(explicit_scale)
        if not np.isfinite(explicit_scale) or explicit_scale <= 0.0:
            raise ValueError(
                "--convergence-state-scale must be finite and strictly positive"
            )

    certificate = certify_spd_by_cholesky(
        operator,
        factor_solver_type=str(
            getattr(args, "riesz_spd_factor_solver_type", "mumps") or "mumps"
        ),
        symmetry_tol=float(getattr(args, "riesz_symmetry_tol", 1.0e-12)),
        options_prefix="plasticity3d_riesz_spd_certificate_",
    )
    matrix_info = operator.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
    complete_provenance = {
        **dict(provenance),
        "constraint_variant": str(constraint_variant),
        "free_space": "glued_free_dofs",
        "free_dofs": int(expected_free_dofs),
        "matrix_type": str(operator.getType()),
        "matrix_nonzeros": int(matrix_info.get("nz_used", 0.0)),
        "spd_certificate": dict(certificate),
    }

    metric: MatrixRieszMetric | None = None
    try:
        metric = MatrixRieszMetric(
            operator,
            name="plasticity3d_reference_elastic_energy",
            provenance=complete_provenance,
            ksp_type=str(getattr(args, "riesz_ksp_type", "cg") or "cg"),
            pc_type=str(getattr(args, "riesz_pc_type", "jacobi") or "jacobi"),
            rtol=float(getattr(args, "riesz_ksp_rtol", 1.0e-10)),
            atol=float(getattr(args, "riesz_ksp_atol", 1.0e-14)),
            max_it=int(getattr(args, "riesz_ksp_max_it", 1000)),
            require_symmetric=False,
            true_residual_rtol=float(
                getattr(args, "riesz_true_residual_rtol", 1.0e-8)
            ),
            set_from_options=False,
        )
        initial_reference_norm = float(metric.primal_norm(initial_state).value)
        if not np.isfinite(initial_reference_norm) or initial_reference_norm < 0.0:
            raise ValueError("Initial reference elastic-energy norm is not finite")
        if explicit_scale is None:
            if initial_reference_norm <= 0.0:
                raise ValueError(
                    "The initial nonlinear iterate has zero reference elastic-energy "
                    "norm. Provide a physically justified positive "
                    "--convergence-state-scale or use a nonzero initial state."
                )
            state_scale = float(initial_reference_norm)
            scale_source = "initial_nonlinear_iterate_primal_norm"
        else:
            state_scale = float(explicit_scale)
            scale_source = "explicit_cli"
        return (
            metric,
            float(state_scale),
            {
                "selection": "reference_elastic_energy",
                "legacy_default": False,
                "state_scale": float(state_scale),
                "state_scale_source": str(scale_source),
                "correction_normalization": "metric_current_state",
                "initial_state_primal_norm": float(initial_reference_norm),
                "primal_norm_definition": "sqrt(u^T K_el u)",
                "absolute_dual_residual_definition": "sqrt(g^T K_el^{-1} g)",
                "absolute_dual_residual_units": (
                    "sqrt(discrete_reference_elastic_energy_unit)"
                ),
                "initial_relative_dual_residual_definition": (
                    "absolute_dual_residual/initial_absolute_dual_residual"
                ),
                "initial_relative_dual_residual_units": "dimensionless",
                "relative_correction_definition": (
                    "primal_correction_norm/max(primal_state_norm,state_scale)"
                ),
                "relative_correction_units": "dimensionless",
                "metric": metric.describe(),
            },
        )
    except Exception:
        if metric is not None:
            metric.destroy()
        raise


def _build_convergence_metric(
    *,
    args,
    assembler: SlopeStability3DReorderedElementAssembler,
    params: dict[str, object],
    regularization_state: dict[str, object],
    initial_state: PETSc.Vec,
) -> tuple[EuclideanMetric | MatrixRieszMetric, float, dict[str, object]]:
    """Construct the selected stopping metric and its dimensionally valid scale."""

    metric_name = _convergence_metric_name(args)
    explicit_scale = getattr(args, "convergence_state_scale", None)
    if explicit_scale is not None:
        explicit_scale = float(explicit_scale)
        if not np.isfinite(explicit_scale) or explicit_scale <= 0.0:
            raise ValueError(
                "--convergence-state-scale must be finite and strictly positive"
            )

    if metric_name == "coefficient_l2":
        state_scale = 1.0 if explicit_scale is None else float(explicit_scale)
        return (
            EuclideanMetric(),
            float(state_scale),
            {
                "selection": "coefficient_l2",
                "legacy_default": bool(explicit_scale is None),
                "state_scale": float(state_scale),
                "state_scale_source": (
                    "legacy_unit_coefficient" if explicit_scale is None else "explicit_cli"
                ),
                "correction_normalization": "legacy_coefficient",
                "absolute_dual_residual_units": "coefficient_l2",
                "initial_relative_dual_residual_units": "dimensionless",
                "relative_correction_units": "dimensionless",
                "metric": EuclideanMetric().describe(),
            },
        )

    constraint_variant = str(params.get("constraint_variant", ""))
    if constraint_variant != PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM:
        raise ValueError(
            "The reference elastic-energy convergence metric is certified only "
            "for the glued_bottom constrained free space; got "
            f"{constraint_variant!r}."
        )
    operator = regularization_state.get("elastic_operator")
    if operator is None:
        raise RuntimeError(
            "Reference elastic operator was not captured before convergence-metric setup"
        )
    expected_rows = int(np.asarray(params["freedofs"], dtype=np.int64).size)
    mpi_comm = operator.getComm().tompi4py()
    input_identity = _reference_elastic_input_identity(
        args=args,
        params=params,
        assembler=assembler,
        comm=mpi_comm,
    )
    provenance = {
        "problem": "Plasticity3D",
        "operator_source": "elastic_tangent_at_zero_displacement",
        "constitutive_mode": "elastic",
        "ordering": "reordered_constrained_free_dofs",
        "ownership": "petsc_distributed_rows",
        "mesh_name": str(params.get("mesh_name", getattr(args, "mesh_name", ""))),
        "element_degree": int(getattr(args, "elem_degree", 0)),
        "quadrature_rule_id": str(params.get("quadrature_rule_id", "")),
        "assembly_backend": str(assembler.assembly_backend),
        "autodiff_tangent_mode": str(assembler.autodiff_tangent_mode),
        "local_hessian_mode": str(assembler.local_hessian_mode),
        "material_parameter_ranges": {
            "shear": _reference_material_range(params, "shear_q", comm=mpi_comm),
            "bulk": _reference_material_range(params, "bulk_q", comm=mpi_comm),
            "lame": _reference_material_range(params, "lame_q", comm=mpi_comm),
        },
        "input_identity": dict(input_identity),
    }
    return _build_certified_reference_elastic_metric(
        args=args,
        operator=operator,
        initial_state=initial_state,
        constraint_variant=constraint_variant,
        expected_free_dofs=expected_rows,
        provenance=provenance,
    )


def _resolve_endpoint_initial_dual_residual(
    result: dict[str, object],
    *,
    tracked_initial: object | None,
    endpoint_value: float,
) -> float | None:
    recorded = result.get("initial_dual_residual_norm")
    if recorded is not None and np.isfinite(float(recorded)):
        return float(recorded)
    if tracked_initial is not None:
        value = float(getattr(tracked_initial, "value"))
        return value if np.isfinite(value) else None
    if "convergence_metric" in result:
        # A normal zero-iteration return has no prior evaluation; its terminal
        # state is also the initial state.
        return float(endpoint_value)
    return None


def _blend_regularized_operator(
    plastic_snapshot: PETSc.Mat,
    assembler: SlopeStability3DReorderedElementAssembler,
    regularization_state: dict[str, object],
    r_value: float,
) -> PETSc.Mat:
    elastic_operator = regularization_state.get("elastic_operator")
    if elastic_operator is None:
        raise RuntimeError("Elastic operator is not available for Newton regularization")
    target = assembler.A
    target.zeroEntries()
    target.axpy(
        float(1.0 - r_value),
        plastic_snapshot,
        structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN,
    )
    target.axpy(
        float(r_value),
        elastic_operator,
        structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN,
    )
    target.assemble()
    return target


def _update_newton_regularization_after_step(
    regularization_state: dict[str, object],
    *,
    alpha: float,
    accepted_step: bool,
    iteration: int,
) -> None:
    if not bool(regularization_state["enabled"]):
        return
    r_old = float(regularization_state["r"])
    r_new = float(r_old)
    reason = "keep"
    alpha = float(alpha)
    if (not bool(accepted_step)) or alpha <= 0.0:
        r_new = min(float(regularization_state["r_max"]), r_old * float(regularization_state["fail_growth"]))
        reason = "rejected"
    elif alpha < float(regularization_state["alpha_increase_threshold"]):
        r_new = min(
            float(regularization_state["r_max"]),
            r_old * float(regularization_state["small_alpha_growth"]),
        )
        reason = "small_alpha"
    elif alpha > float(regularization_state["alpha_decrease_threshold"]):
        r_new = max(
            float(regularization_state["r_min"]),
            r_old / float(regularization_state["decay"]),
        )
        reason = "good_alpha"
    regularization_state["r"] = float(r_new)
    regularization_state["last_step"] = {
        "it": int(iteration),
        "r_before": float(r_old),
        "r_after": float(r_new),
        "alpha": float(alpha),
        "accepted_step": bool(accepted_step),
        "reason": str(reason),
    }
    regularization_state["history"].append(dict(regularization_state["last_step"]))


def run(args):
    comm = MPI.COMM_WORLD
    rank = int(comm.rank)
    total_runtime_start = time.perf_counter()
    stage_timings: dict[str, float] = {}
    debug_setup = bool(getattr(args, "debug_setup", False))
    petsc_log_view_path = _configure_petsc_logging(args, comm)
    petsc_stage_enabled = bool(
        bool(getattr(args, "enable_petsc_log_events", False)) or petsc_log_view_path
    )

    settings = _apply_3d_stack_defaults(args, _resolve_linear_settings(args))
    regularization_state = _init_newton_regularization_state(args)
    pc_options = _pc_options(settings)
    chunk_mode, p4_chunk_size_initial = _parse_p4_chunk_size_arg(
        getattr(args, "p4_hessian_chunk_size", 32)
    )
    chunk_autotune_candidates = _parse_chunk_candidates(
        getattr(args, "p4_chunk_autotune_candidates", "32,64,128,256")
    )
    if rank == 0 and debug_setup:
        print("setup: problem load begin", flush=True)
    t_stage = time.perf_counter()
    with _petsc_stage("slope3d_problem_load", petsc_stage_enabled):
        mesh_name, params, adjacency = _load_problem_data(args, comm)
    stage_timings["problem_load"] = float(time.perf_counter() - t_stage)
    if rank == 0 and debug_setup:
        print(
            f"setup: problem load done, t={stage_timings['problem_load']:.3f}s",
            flush=True,
        )
    lambda_target = float(
        getattr(args, "lambda_target", None)
        if getattr(args, "lambda_target", None) is not None
        else params.get("lambda_target_default", 1.0)
    )
    _apply_strength_reduction(params, lambda_target)

    if rank == 0 and debug_setup:
        print("setup: assembler create begin", flush=True)
    t_stage = time.perf_counter()
    with _petsc_stage("slope3d_assembler_create", petsc_stage_enabled):
        assembler = SlopeStability3DReorderedElementAssembler(
            params=params,
            comm=comm,
            adjacency=adjacency,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_type=str(settings["ksp_type"]),
            pc_type=str(settings["pc_type"]),
            ksp_max_it=int(settings["ksp_max_it"]),
            use_near_nullspace=bool(settings["use_near_nullspace"]),
            pc_options=pc_options,
            reorder_mode=str(getattr(args, "element_reorder_mode", None) or "block_xyz"),
            local_hessian_mode=str(getattr(args, "local_hessian_mode", None) or "element"),
            autodiff_tangent_mode=str(
                getattr(args, "autodiff_tangent_mode", None) or "element"
            ),
            perm_override=(
                np.asarray(params["_distributed_perm"], dtype=np.int64)
                if "_distributed_perm" in params
                else select_reordered_perm_3d(
                    str(getattr(args, "element_reorder_mode", None) or "block_xyz"),
                    adjacency=adjacency,
                    coords_all=np.asarray(params["nodes"], dtype=np.float64),
                    freedofs=np.asarray(params["freedofs"], dtype=np.int64),
                    n_parts=int(comm.size),
                )
            ),
            block_size_override=ownership_block_size_3d(
                np.asarray(params["freedofs"], dtype=np.int64)
            ),
            distribution_strategy=str(getattr(args, "distribution_strategy", "overlap_p2p")),
            reuse_hessian_value_buffers=bool(
                getattr(args, "reuse_hessian_value_buffers", True)
            ),
            p4_hessian_chunk_size=int(p4_chunk_size_initial),
            p4_chunk_scatter_cache=str(
                getattr(args, "p4_chunk_scatter_cache", "auto")
            ),
            p4_chunk_scatter_cache_max_gib=float(
                getattr(args, "p4_chunk_scatter_cache_max_gib", 0.5)
            ),
            assembly_backend=str(getattr(args, "assembly_backend", "coo")),
            petsc_log_events=bool(petsc_stage_enabled),
            jax_trace_dir=str(getattr(args, "jax_trace_dir", "") or ""),
        )
    stage_timings["assembler_create"] = float(time.perf_counter() - t_stage)
    if int(args.elem_degree) == 4 and chunk_mode == "auto":
        if rank == 0 and debug_setup:
            print("setup: p4 chunk autotune begin", flush=True)
        t_stage = time.perf_counter()
        with _petsc_stage("slope3d_p4_chunk_autotune", petsc_stage_enabled):
            assembler.autotune_p4_hessian_chunk_size(
                u_owned=np.zeros(int(assembler.layout.hi - assembler.layout.lo), dtype=np.float64),
                candidates=chunk_autotune_candidates,
                rss_budget_gib=float(
                    getattr(args, "p4_chunk_autotune_rss_budget_gib", 64.0)
                ),
            )
        stage_timings["p4_chunk_autotune"] = float(time.perf_counter() - t_stage)
        if rank == 0 and debug_setup:
            print(
                "setup: p4 chunk autotune done, "
                f"t={stage_timings['p4_chunk_autotune']:.3f}s "
                f"selected={int(assembler.p4_hessian_chunk_size)}",
                flush=True,
            )
    local_setup_summary = assembler.setup_summary()
    local_memory_summary = assembler.memory_summary()
    gathered_setup = comm.gather(local_setup_summary, root=0)
    gathered_memory = comm.gather(local_memory_summary, root=0)
    progress_path = str(getattr(args, "progress_out", "") or "").strip()
    if rank == 0 and debug_setup:
        print(
            f"setup: assembler create done, t={stage_timings['assembler_create']:.3f}s",
            flush=True,
        )
        if gathered_setup:
            worst_setup = max(
                list(gathered_setup),
                key=lambda entry: float(entry.get("total", 0.0)),
            )
            print("setup: worst-rank timings", json.dumps(worst_setup, indent=2), flush=True)
        if gathered_memory:
            worst_memory = max(
                list(gathered_memory),
                key=lambda entry: float(entry.get("tracked_total_gib", 0.0)),
            )
            print(
                "setup: worst-rank tracked memory "
                f"{_format_gib(worst_memory.get('tracked_total_gib', 0.0))} "
                f"(layout={_format_gib(worst_memory.get('layout_gib', 0.0))}, "
                f"local={_format_gib(worst_memory.get('local_overlap_gib', 0.0))}, "
                f"scatter={_format_gib(worst_memory.get('scatter_gib', 0.0))}, "
                f"owned_vals={_format_gib(worst_memory.get('owned_hessian_values_gib', 0.0))})",
                flush=True,
            )
    if rank == 0 and progress_path:
        _write_progress_payload(
            progress_path,
            {
                "status": "setup_complete",
                "mesh_name": str(mesh_name),
                "elem_degree": int(args.elem_degree),
                "lambda_target": float(lambda_target),
                "assembly_backend": str(assembler.assembly_backend),
                "assembly_backend_requested": str(
                    assembler.assembly_backend_requested
                ),
                "autodiff_tangent_mode": str(assembler.autodiff_tangent_mode),
                "matrix_type": str(assembler.A.getType()),
                "p4_hessian_chunk_size": int(assembler.p4_hessian_chunk_size),
                "p4_chunk_autotune": dict(assembler.p4_chunk_autotune_meta or {}),
                "p4_chunk_scatter_cache": dict(
                    assembler.p4_chunk_scatter_cache_meta or {}
                ),
                "petsc_log_view_path": str(petsc_log_view_path or ""),
                "stage_timings": dict(stage_timings),
                "assembler_setup": dict(local_setup_summary),
                "assembler_memory": dict(local_memory_summary),
                "parallel_setup": list(gathered_setup or []),
                "parallel_memory": list(gathered_memory or []),
            },
        )

    mg_hierarchy = None
    mg_nullspace_meta = None
    mg_nullspaces_live: list[PETSc.NullSpace] = []
    initial_guess_meta: dict[str, object] | None = None
    resolved_mg_strategy: str | None = None
    legacy_mg_settings = _legacy_mg_settings(args)
    if str(settings["pc_type"]) == "mg":
        if int(args.elem_degree) == 1:
            raise ValueError("3D PMG requires a fine degree of 2 or 4")
        strategy = _resolve_mg_strategy(args)
        resolved_mg_strategy = str(strategy)
        specs = mixed_hierarchy_specs(
            mesh_name=mesh_name,
            finest_degree=int(args.elem_degree),
            strategy=strategy,
        )
        if rank == 0 and debug_setup:
            print("setup: mg hierarchy build begin", flush=True)
        t_stage = time.perf_counter()
        with _petsc_stage("slope3d_mg_hierarchy_build", petsc_stage_enabled):
            mg_hierarchy = build_mixed_pmg_hierarchy(
                specs=specs,
                finest_params=params,
                finest_adjacency=adjacency,
                finest_perm=np.asarray(assembler.layout.perm, dtype=np.int64),
                reorder_mode=str(getattr(args, "element_reorder_mode", None) or "block_xyz"),
                comm=comm,
                level_build_mode=str(getattr(args, "mg_level_build_mode", "root_bcast")),
                transfer_build_mode=str(
                    getattr(args, "mg_transfer_build_mode", "owned_rows")
                ),
            )
        stage_timings["mg_hierarchy_build"] = float(time.perf_counter() - t_stage)
        if rank == 0 and debug_setup:
            print(
                f"setup: mg hierarchy build done, t={stage_timings['mg_hierarchy_build']:.3f}s",
                flush=True,
            )
        configure_pmg(
            assembler.ksp,
            mg_hierarchy,
            level_smoothers=legacy_mg_settings,
            coarse_backend=str(getattr(args, "mg_coarse_backend", None) or "hypre"),
            coarse_ksp_type=str(getattr(args, "mg_coarse_ksp_type", None) or "cg"),
            coarse_pc_type=str(getattr(args, "mg_coarse_pc_type", None) or "hypre"),
            coarse_hypre_nodal_coarsen=int(
                getattr(args, "mg_coarse_hypre_nodal_coarsen", 6)
            ),
            coarse_hypre_vec_interp_variant=int(
                getattr(args, "mg_coarse_hypre_vec_interp_variant", 3)
            ),
            coarse_hypre_strong_threshold=getattr(
                args, "mg_coarse_hypre_strong_threshold", 0.5
            ),
            coarse_hypre_coarsen_type=str(
                getattr(args, "mg_coarse_hypre_coarsen_type", None) or "HMIS"
            ),
            coarse_hypre_max_iter=int(getattr(args, "mg_coarse_hypre_max_iter", 2)),
            coarse_hypre_tol=float(getattr(args, "mg_coarse_hypre_tol", 0.0)),
            coarse_hypre_relax_type_all=str(
                getattr(args, "mg_coarse_hypre_relax_type_all", "symmetric-SOR/Jacobi")
            ),
        )

    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    u_0 = np.asarray(params["u_0"], dtype=np.float64)
    u_init_free = np.asarray(u_0[freedofs], dtype=np.float64)
    u_init_reordered = np.asarray(u_init_free[assembler.layout.perm], dtype=np.float64)
    external_init_reordered, external_init_meta = _load_initial_state_guess(
        args,
        params=params,
        freedofs=freedofs,
        perm=np.asarray(assembler.layout.perm, dtype=np.int64),
    )
    if external_init_reordered is not None:
        u_init_reordered = np.asarray(external_init_reordered, dtype=np.float64)
    x = assembler.create_vec(u_init_reordered)
    ksp = assembler.ksp
    A = assembler.A
    active_operator = {"mat": A, "label": "plastic"}
    use_fresh_linear_ksp = str(settings["pc_type"]) == "mg"

    gamg_coords = None
    if str(settings["pc_type"]) == "gamg" and bool(settings["gamg_set_coordinates"]):
        gamg_coords = _build_gamg_coordinates_owned_blocks(assembler, params)

    linear_iters: list[int] = []
    linear_records: list[dict[str, object]] = []
    residual_ax = x.duplicate()
    residual_vec = x.duplicate()
    ksp_accept_true_rel = getattr(args, "ksp_accept_true_rel", None)
    verbose_linear_debug = not bool(getattr(args, "quiet", False))

    if external_init_reordered is not None:
        initial_guess_meta = dict(external_init_meta or {})
    elif _should_use_elastic_initial_guess(args, settings):
        if rank == 0 and debug_setup:
            print("setup: elastic initial guess begin", flush=True)
        t_stage = time.perf_counter()
        with _petsc_stage("slope3d_elastic_initial_guess", petsc_stage_enabled):
            (
                u_init_reordered,
                mg_nullspace_meta,
                initial_guess_meta,
                gamg_coords,
            ) = _solve_elastic_initial_guess(
                assembler=assembler,
                settings=settings,
                args=args,
                mg_hierarchy=mg_hierarchy,
                mg_nullspace_meta=mg_nullspace_meta,
                gamg_coords=gamg_coords,
            )
        stage_timings["initial_guess_total"] = float(time.perf_counter() - t_stage)
        if rank == 0 and debug_setup:
            print(
                f"setup: elastic initial guess done, t={stage_timings['initial_guess_total']:.3f}s",
                flush=True,
            )
        if not bool(initial_guess_meta.get("success", False)):
            raise RuntimeError(str(initial_guess_meta.get("message", "Elastic initial-guess solve failed")))
        if mg_nullspace_meta is not None:
            mg_nullspaces_live.extend(list(mg_nullspace_meta.get("nullspaces", [])))
        if _elastic_operator_required(args, regularization_state):
            _capture_elastic_operator(assembler, regularization_state)
        x.array[:] = np.asarray(u_init_reordered, dtype=np.float64)
        x.assemble()
    else:
        initial_guess_meta = {
            "enabled": False,
            "success": False,
            "message": "Elastic initial guess disabled for this solver stack",
            "vector_norm": float(np.linalg.norm(np.asarray(x.array[:], dtype=np.float64))),
        }
        stage_timings["initial_guess_total"] = 0.0

    if (
        _elastic_operator_required(args, regularization_state)
        and regularization_state.get("elastic_operator") is None
    ):
        zero_owned = np.zeros(assembler.layout.hi - assembler.layout.lo, dtype=np.float64)
        assembler.assemble_hessian_with_mode(zero_owned, constitutive_mode="elastic")
        _capture_elastic_operator(assembler, regularization_state)

    convergence_metric: EuclideanMetric | MatrixRieszMetric | None = None
    try:
        t_stage = time.perf_counter()
        (
            convergence_metric,
            convergence_state_scale,
            convergence_configuration,
        ) = _build_convergence_metric(
            args=args,
            assembler=assembler,
            params=params,
            regularization_state=regularization_state,
            initial_state=x,
        )
        stage_timings["convergence_metric_setup"] = float(
            time.perf_counter() - t_stage
        )
    except Exception:
        if isinstance(convergence_metric, MatrixRieszMetric):
            convergence_metric.destroy()
        for nullspace in mg_nullspaces_live:
            nullspace.destroy()
        elastic_operator = regularization_state.get("elastic_operator")
        if elastic_operator is not None:
            elastic_operator.destroy()
            regularization_state["elastic_operator"] = None
        residual_ax.destroy()
        residual_vec.destroy()
        x.destroy()
        assembler.cleanup()
        if mg_hierarchy is not None:
            mg_hierarchy.cleanup()
        raise

    def _make_linear_ksp() -> PETSc.KSP:
        linear_ksp = PETSc.KSP().create(comm)
        linear_ksp.setType(str(settings["ksp_type"]))
        linear_pc = linear_ksp.getPC()
        linear_pc.setType(str(settings["pc_type"]))
        if str(settings["pc_type"]) == "mg":
            configure_pmg(
                linear_ksp,
                mg_hierarchy,
                level_smoothers=legacy_mg_settings,
                coarse_backend=str(getattr(args, "mg_coarse_backend", None) or "hypre"),
                coarse_ksp_type=str(getattr(args, "mg_coarse_ksp_type", None) or "cg"),
                coarse_pc_type=str(getattr(args, "mg_coarse_pc_type", None) or "hypre"),
                coarse_hypre_nodal_coarsen=int(
                    getattr(args, "mg_coarse_hypre_nodal_coarsen", 6)
                ),
                coarse_hypre_vec_interp_variant=int(
                    getattr(args, "mg_coarse_hypre_vec_interp_variant", 3)
                ),
                coarse_hypre_strong_threshold=getattr(
                    args, "mg_coarse_hypre_strong_threshold", 0.5
                ),
                coarse_hypre_coarsen_type=str(
                    getattr(args, "mg_coarse_hypre_coarsen_type", None) or "HMIS"
                ),
                coarse_hypre_max_iter=int(getattr(args, "mg_coarse_hypre_max_iter", 2)),
                coarse_hypre_tol=float(getattr(args, "mg_coarse_hypre_tol", 0.0)),
                coarse_hypre_relax_type_all=str(
                    getattr(args, "mg_coarse_hypre_relax_type_all", "symmetric-SOR/Jacobi")
                ),
            )
        linear_ksp.setFromOptions()
        return linear_ksp

    root_iteration_history: list[dict[str, object]] = []

    def _assemble_and_solve(vec, rhs, sol, trust_radius=None):
        nonlocal gamg_coords, mg_nullspace_meta
        if verbose_linear_debug and rank == 0:
            print("linear: assemble begin", flush=True)
        t_asm0 = time.perf_counter()
        if trust_radius is not None and str(settings["ksp_type"]).lower() in {
            "stcg",
            "nash",
            "gltr",
        }:
            ksp_cg_set_radius(ksp, float(trust_radius))

        assembler.assemble_hessian(np.asarray(vec.array[:], dtype=np.float64))
        t_asm = time.perf_counter() - t_asm0
        if verbose_linear_debug and rank == 0:
            print(f"linear: assemble done, t_asm={t_asm:.3f}s", flush=True)
        r_current = float(regularization_state["r"])
        plastic_snapshot = A.copy() if bool(regularization_state["enabled"]) else None
        attempt_records: list[dict[str, object]] = []
        max_attempts = 1 + (
            int(regularization_state["retry_max"])
            if bool(regularization_state["enabled"])
            else 0
        )
        final_record: dict[str, object] | None = None
        linear_ksp = _make_linear_ksp() if use_fresh_linear_ksp else ksp
        for attempt_index in range(1, max_attempts + 1):
            active_mat = A
            operator_label = "plastic"
            if bool(regularization_state["enabled"]):
                active_mat = _blend_regularized_operator(
                    plastic_snapshot,
                    assembler,
                    regularization_state,
                    r_current,
                )
                operator_label = "regularized"
            active_operator["mat"] = active_mat
            active_operator["label"] = str(operator_label)
            linear_ksp.setOperators(active_mat)
            if gamg_coords is not None and int(np.asarray(gamg_coords).size) > 0:
                linear_ksp.getPC().setCoordinates(np.asarray(gamg_coords, dtype=np.float64))
                gamg_coords = None
            linear_ksp.setTolerances(
                rtol=float(settings["ksp_rtol"]),
                max_it=int(settings["ksp_max_it"]),
            )
            if verbose_linear_debug and rank == 0:
                print(
                    "linear: ksp setup begin "
                    f"(attempt={attempt_index}, mode={operator_label}, r={r_current:.5e})",
                    flush=True,
                )
            t_setup0 = time.perf_counter()
            linear_ksp.setUp()
            t_setup = time.perf_counter() - t_setup0
            if verbose_linear_debug and rank == 0:
                print(f"linear: ksp setup done, t_setup={t_setup:.3f}s", flush=True)
            if mg_hierarchy is not None:
                mg_nullspace_meta = attach_pmg_level_metadata(
                    linear_ksp,
                    mg_hierarchy,
                    use_near_nullspace=bool(settings["use_near_nullspace"]),
                    coarse_pc_type=str(getattr(args, "mg_coarse_pc_type", None) or "hypre"),
                    coarse_hypre_nodal_coarsen=int(
                        getattr(args, "mg_coarse_hypre_nodal_coarsen", 6)
                    ),
                    coarse_hypre_vec_interp_variant=int(
                        getattr(args, "mg_coarse_hypre_vec_interp_variant", 3)
                    ),
                    coarse_hypre_strong_threshold=getattr(
                        args, "mg_coarse_hypre_strong_threshold", 0.5
                    ),
                    coarse_hypre_coarsen_type=str(
                        getattr(args, "mg_coarse_hypre_coarsen_type", None) or "HMIS"
                    ),
                    coarse_hypre_max_iter=int(getattr(args, "mg_coarse_hypre_max_iter", 2)),
                    coarse_hypre_tol=float(getattr(args, "mg_coarse_hypre_tol", 0.0)),
                    coarse_hypre_relax_type_all=str(
                        getattr(args, "mg_coarse_hypre_relax_type_all", "symmetric-SOR/Jacobi")
                    ),
                )
                for ns in list(mg_nullspace_meta.get("nullspaces", [])):
                    if ns is None:
                        continue
                    if not any(existing is ns for existing in mg_nullspaces_live):
                        mg_nullspaces_live.append(ns)
            if verbose_linear_debug and rank == 0:
                print("linear: ksp solve begin", flush=True)
            t_solve0 = time.perf_counter()
            if verbose_linear_debug and rank == 0:
                def _ksp_monitor(_ksp, its, rnorm):
                    its_i = int(its)
                    if its_i < 10 or its_i % 10 == 0:
                        print(
                            f"linear: ksp iter={its_i}, residual={float(rnorm):.5e}",
                            flush=True,
                        )
                linear_ksp.setMonitor(_ksp_monitor)
            linear_ksp.solve(rhs, sol)
            if verbose_linear_debug and rank == 0:
                try:
                    linear_ksp.cancelMonitor()
                except PETSc.Error:
                    pass
            t_solve = time.perf_counter() - t_solve0
            ksp_its = int(linear_ksp.getIterationNumber())
            reason_code = int(linear_ksp.getConvergedReason())
            reason_name = ksp_reason_name(reason_code)
            rhs_norm = float(rhs.norm(PETSc.NormType.NORM_2))
            active_mat.mult(sol, residual_ax)
            rhs.copy(residual_vec)
            residual_vec.axpy(-1.0, residual_ax)
            true_residual_norm = float(residual_vec.norm(PETSc.NormType.NORM_2))
            true_relative_residual = true_residual_norm / max(rhs_norm, 1.0e-16)
            directional_derivative = float(-rhs.dot(sol))
            if verbose_linear_debug and rank == 0:
                print(
                    "linear: ksp solve done, "
                    f"t_solve={t_solve:.3f}s, its={ksp_its}, "
                    f"reason={reason_name}, true_rel={true_relative_residual:.5e}",
                    flush=True,
                )
            accepted_via_true_residual = bool(
                reason_code <= 0
                and ksp_accept_true_rel is not None
                and np.isfinite(float(true_relative_residual))
                and float(true_relative_residual) <= float(ksp_accept_true_rel)
            )
            maxit_direction_true_rel_cap = float(
                getattr(args, "ksp_maxit_direction_true_rel_cap", 6.0e-2)
            )
            guard_ksp_maxit_direction = bool(
                getattr(args, "guard_ksp_maxit_direction", False)
            )
            accepted_via_maxit_direction = bool(
                reason_code <= 0
                and bool(getattr(args, "accept_ksp_maxit_direction", True))
                and str(reason_name) == "DIVERGED_MAX_IT"
                and np.isfinite(float(true_relative_residual))
                and (
                    not guard_ksp_maxit_direction
                    or (
                        float(true_relative_residual) <= maxit_direction_true_rel_cap
                        and np.isfinite(float(directional_derivative))
                        and float(directional_derivative) < 0.0
                    )
                )
            )
            attempt_record = {
                "newton_iteration": int(len(linear_records) + 1),
                "attempt": int(attempt_index),
                "operator_mode": str(operator_label),
                "newton_regularization_r": float(r_current),
                "t_assemble": float(t_asm),
                "ksp_its": int(ksp_its),
                "ksp_reason_code": int(reason_code),
                "ksp_reason_name": str(reason_name),
                "ksp_residual_norm": float(linear_ksp.getResidualNorm()),
                "rhs_norm": float(rhs_norm),
                "true_residual_norm": float(true_residual_norm),
                "true_relative_residual": float(true_relative_residual),
                "directional_derivative": float(directional_derivative),
                "accepted_via_true_residual": bool(accepted_via_true_residual),
                "accepted_via_maxit_direction": bool(accepted_via_maxit_direction),
                "guard_ksp_maxit_direction": bool(guard_ksp_maxit_direction),
                "ksp_maxit_direction_true_rel_cap": float(maxit_direction_true_rel_cap),
                "t_setup": float(t_setup),
                "t_solve": float(t_solve),
            }
            attempt_records.append(dict(attempt_record))
            final_record = dict(attempt_record)
            success = bool(
                reason_code > 0
                or accepted_via_true_residual
                or accepted_via_maxit_direction
            )
            if success:
                linear_iters.append(ksp_its)
                regularization_state["r"] = float(r_current)
                break
            if (
                not bool(regularization_state["enabled"])
                or attempt_index >= max_attempts
                or float(r_current) >= float(regularization_state["r_max"])
            ):
                linear_iters.append(ksp_its)
                break
            r_current = min(
                float(regularization_state["r_max"]),
                float(r_current) * float(regularization_state["fail_growth"]),
            )
            if verbose_linear_debug and rank == 0:
                print(
                    "linear: retrying with stronger regularization "
                    f"r={r_current:.5e}",
                    flush=True,
                )
        if final_record is None:
            if plastic_snapshot is not None:
                plastic_snapshot.destroy()
            if use_fresh_linear_ksp:
                linear_ksp.destroy()
            raise RuntimeError("Linear solve attempt loop finished without a final record")
        final_record["regularization_attempts"] = list(attempt_records)
        linear_records.append(dict(final_record))
        if plastic_snapshot is not None:
            plastic_snapshot.destroy()
        if use_fresh_linear_ksp:
            linear_ksp.destroy()
        elif settings["pc_setup_on_ksp_cap"] and final_record["ksp_its"] >= int(settings["ksp_max_it"]):
            ksp.setUp()
        if (
            int(final_record["ksp_reason_code"]) <= 0
            and not bool(final_record["accepted_via_true_residual"])
            and not bool(final_record["accepted_via_maxit_direction"])
        ):
            raise _LinearSolveFailure(
                reason_code=int(final_record["ksp_reason_code"]),
                reason_name=str(final_record["ksp_reason_name"]),
                ksp_its=int(final_record["ksp_its"]),
                true_residual_norm=float(final_record["true_residual_norm"]),
                true_relative_residual=float(final_record["true_relative_residual"]),
            )
        return int(final_record["ksp_its"])

    def hessian_solve_fn(vec, rhs, sol):
        return _assemble_and_solve(vec, rhs, sol, trust_radius=None)

    def trust_subproblem_solve_fn(vec, rhs, sol, trust_radius):
        return _assemble_and_solve(vec, rhs, sol, trust_radius=float(trust_radius))

    def _iteration_callback(entry: dict[str, object], history: list[dict[str, object]]) -> None:
        _update_newton_regularization_after_step(
            regularization_state,
            alpha=float(entry.get("alpha", 0.0)),
            accepted_step=bool(entry.get("accepted_step", False)),
            iteration=int(entry.get("it", len(history))),
        )
        latest_linear_local = dict(linear_records[-1]) if linear_records else None
        memory_local = _process_memory_snapshot()
        gathered_memory = comm.gather(memory_local, root=0)
        gathered_linear = comm.gather(latest_linear_local, root=0)
        if rank != 0 or not progress_path:
            return
        enriched_entry = dict(entry)
        enriched_entry["linear_iteration"] = _aggregate_iteration_linear(gathered_linear)
        enriched_entry["memory_profile"] = _aggregate_iteration_memory(gathered_memory)
        root_iteration_history.append(dict(enriched_entry))
        if root_iteration_history:
            history_payload = list(root_iteration_history)
        else:
            history_payload = list(history)
        if not progress_path:
            return
        _write_progress_payload(
            progress_path,
            {
                "status": "running",
                "mesh_name": str(mesh_name),
                "elem_degree": int(args.elem_degree),
                "lambda_target": float(lambda_target),
                "assembly_backend": str(assembler.assembly_backend),
                "matrix_type": str(assembler.A.getType()),
                "p4_hessian_chunk_size": int(assembler.p4_hessian_chunk_size),
                "p4_chunk_autotune": dict(assembler.p4_chunk_autotune_meta or {}),
                "p4_chunk_scatter_cache": dict(
                    assembler.p4_chunk_scatter_cache_meta or {}
                ),
                "iterations_completed": int(enriched_entry.get("it", len(history_payload))),
                "last_iteration": dict(enriched_entry),
                "history": history_payload,
                "convergence": dict(convergence_configuration),
                "newton_regularization": {
                    "enabled": bool(regularization_state["enabled"]),
                    "current_r": float(regularization_state["r"]),
                    "last_step": dict(regularization_state["last_step"] or {}),
                    "history": list(regularization_state["history"]),
                },
            },
        )

    solve_start = time.perf_counter()
    try:
        with _petsc_stage("slope3d_newton_solve", petsc_stage_enabled):
            with _jax_trace_context(
                str(getattr(args, "jax_trace_dir", "") or ""),
                rank=rank,
            ):
                result = newton(
                    energy_fn=assembler.energy_fn,
                    gradient_fn=assembler.gradient_fn,
                    hessian_solve_fn=hessian_solve_fn,
                    x=x,
                    tolf=float(args.tolf),
                    tolg=float(args.tolg),
                    tolg_rel=float(args.tolg_rel),
                    linesearch_tol=float(args.linesearch_tol),
                    linesearch_interval=(float(args.linesearch_a), float(args.linesearch_b)),
                    line_search=str(getattr(args, "line_search", "golden_fixed")),
                    armijo_alpha0=float(getattr(args, "armijo_alpha0", 1.0)),
                    armijo_c1=float(getattr(args, "armijo_c1", 1.0e-4)),
                    armijo_shrink=float(getattr(args, "armijo_shrink", 0.5)),
                    armijo_max_ls=int(getattr(args, "armijo_max_ls", 40)),
                    maxit=int(args.maxit),
                    tolx_rel=float(args.tolx_rel),
                    tolx_abs=float(args.tolx_abs),
                    require_all_convergence=True,
                    fail_on_nonfinite=True,
                    verbose=(not bool(getattr(args, "quiet", False))),
                    comm=comm,
                    hessian_matvec_fn=lambda _x, vin, vout: active_operator["mat"].mult(vin, vout),
                    trust_subproblem_solve_fn=(
                        trust_subproblem_solve_fn
                        if bool(getattr(args, "use_trust_region", False))
                        else None
                    ),
                    trust_subproblem_line_search=bool(
                        getattr(args, "trust_subproblem_line_search", False)
                    ),
                    save_history=bool(getattr(args, "save_history", False)),
                    trust_region=bool(getattr(args, "use_trust_region", False)),
                    trust_radius_init=float(getattr(args, "trust_radius_init", 0.5)),
                    trust_radius_min=float(getattr(args, "trust_radius_min", 1.0e-8)),
                    trust_radius_max=float(getattr(args, "trust_radius_max", 1.0e6)),
                    trust_shrink=float(getattr(args, "trust_shrink", 0.5)),
                    trust_expand=float(getattr(args, "trust_expand", 1.5)),
                    trust_eta_shrink=float(getattr(args, "trust_eta_shrink", 0.05)),
                    trust_eta_expand=float(getattr(args, "trust_eta_expand", 0.75)),
                    trust_max_reject=int(getattr(args, "trust_max_reject", 6)),
                    step_time_limit_s=getattr(args, "step_time_limit_s", None),
                    iteration_callback=_iteration_callback,
                    convergence_metric=convergence_metric,
                    convergence_state_scale=float(convergence_state_scale),
                    convergence_correction_mode=str(
                        convergence_configuration["correction_normalization"]
                    ),
                )
    except _LinearSolveFailure as exc:
        result = {
            "nit": int(len(linear_iters)),
            "fun": float(assembler.energy_fn(x)),
            "message": str(exc),
            "history": [],
        }
    except Exception:
        if isinstance(convergence_metric, MatrixRieszMetric):
            convergence_metric.destroy()
        for nullspace in mg_nullspaces_live:
            nullspace.destroy()
        elastic_operator = regularization_state.get("elastic_operator")
        if elastic_operator is not None:
            elastic_operator.destroy()
            regularization_state["elastic_operator"] = None
        residual_ax.destroy()
        residual_vec.destroy()
        x.destroy()
        assembler.cleanup()
        if mg_hierarchy is not None:
            mg_hierarchy.cleanup()
        raise

    tracked_initial_before_endpoint = getattr(
        convergence_metric,
        "first_dual_evaluation",
        None,
    )
    final_grad_vec: PETSc.Vec | None = None
    endpoint_convergence_start = time.perf_counter()
    try:
        recorded_endpoint_values = (
            result.get("grad_norm_coefficient_l2"),
            result.get("dual_residual_norm"),
            result.get("state_norm"),
        )
        has_recorded_endpoint = all(
            value is not None and np.isfinite(float(value))
            for value in recorded_endpoint_values
        ) and bool(result.get("dual_residual_metadata"))
        if has_recorded_endpoint:
            final_grad_norm = float(result["grad_norm_coefficient_l2"])
            endpoint_dual_value = float(result["dual_residual_norm"])
            endpoint_dual_metadata = dict(result["dual_residual_metadata"])
            endpoint_state_value = float(result["state_norm"])
        else:
            final_grad_vec = x.duplicate()
            assembler.gradient_fn(x, final_grad_vec)
            final_grad_norm = float(final_grad_vec.norm(PETSc.NormType.NORM_2))
            endpoint_dual = convergence_metric.dual_norm(final_grad_vec)
            endpoint_state = convergence_metric.primal_norm(x)
            endpoint_dual_value = float(endpoint_dual.value)
            endpoint_dual_metadata = dict(endpoint_dual.metadata)
            endpoint_state_value = float(endpoint_state.value)
        if not (
            np.isfinite(final_grad_norm)
            and np.isfinite(endpoint_dual_value)
            and np.isfinite(endpoint_state_value)
        ):
            raise RuntimeError("Plasticity3D endpoint convergence norms are nonfinite")
        initial_dual_residual = _resolve_endpoint_initial_dual_residual(
            result,
            tracked_initial=tracked_initial_before_endpoint,
            endpoint_value=endpoint_dual_value,
        )
        result["convergence_metric"] = convergence_metric.describe()
        result["initial_dual_residual_norm"] = (
            None
            if initial_dual_residual is None
            else float(initial_dual_residual)
        )
        result["dual_residual_norm"] = float(endpoint_dual_value)
        if initial_dual_residual is None:
            result["dual_residual_relative"] = None
            endpoint_dual_target = (
                float(args.tolg) if float(args.tolg_rel) <= 0.0 else None
            )
        else:
            result["dual_residual_relative"] = float(
                float(endpoint_dual_value)
                / max(float(initial_dual_residual), np.finfo(np.float64).tiny)
            )
            endpoint_dual_target = max(
                float(args.tolg),
                float(args.tolg_rel) * float(initial_dual_residual),
            )
        result["dual_residual_target"] = (
            None if endpoint_dual_target is None else float(endpoint_dual_target)
        )
        result["dual_residual_gate_pass"] = bool(
            endpoint_dual_target is not None
            and float(endpoint_dual_value) < float(endpoint_dual_target)
        )
        result["dual_residual_metadata"] = dict(endpoint_dual_metadata)
        result["grad_norm_coefficient_l2"] = float(final_grad_norm)
        result["state_norm"] = float(endpoint_state_value)
        result["convergence_state_scale"] = float(convergence_state_scale)
        result["convergence_correction_mode"] = str(
            convergence_configuration["correction_normalization"]
        )
    except Exception:
        if isinstance(convergence_metric, MatrixRieszMetric):
            convergence_metric.destroy()
        if final_grad_vec is not None:
            final_grad_vec.destroy()
            final_grad_vec = None
        for nullspace in mg_nullspaces_live:
            nullspace.destroy()
        elastic_operator = regularization_state.get("elastic_operator")
        if elastic_operator is not None:
            elastic_operator.destroy()
            regularization_state["elastic_operator"] = None
        residual_ax.destroy()
        residual_vec.destroy()
        x.destroy()
        assembler.cleanup()
        if mg_hierarchy is not None:
            mg_hierarchy.cleanup()
        raise
    finally:
        if final_grad_vec is not None:
            final_grad_vec.destroy()
        if isinstance(convergence_metric, MatrixRieszMetric):
            convergence_metric.destroy()
    stage_timings["endpoint_convergence"] = float(
        time.perf_counter() - endpoint_convergence_start
    )
    solve_time = time.perf_counter() - solve_start

    full_reordered, _ = assembler._allgather_full_owned(
        np.asarray(x.array[:], dtype=np.float64)
    )
    full_original = np.empty_like(full_reordered)
    full_original[np.asarray(assembler.layout.perm, dtype=np.int64)] = full_reordered
    u_full = np.asarray(params["u_0"], dtype=np.float64).copy()
    u_full[freedofs] = full_original
    coords_ref = np.asarray(params["nodes"], dtype=np.float64)
    coords_final = coords_ref + u_full.reshape((-1, 3))
    displacement = coords_final - coords_ref
    u_max = float(np.max(np.linalg.norm(displacement, axis=1)))
    omega = float(np.dot(np.asarray(params["force"], dtype=np.float64), u_full))
    solver_success = bool(
        str(result["message"]).lower().startswith("converged")
        and bool(result.get("dual_residual_gate_pass", False))
        and np.isfinite(float(result["fun"]))
        and np.all(np.isfinite(full_original))
    )
    result_status = "completed" if solver_success else "failed"
    local_total_time = float(time.perf_counter() - total_runtime_start)
    matrix_type = str(assembler.A.getType())
    transfer_backend = str(
        ((mg_hierarchy.build_metadata or {}) if mg_hierarchy is not None else {}).get(
            "transfer_backend",
            "coo_vectorized",
        )
    )
    local_parallel_diag = {
        "rank": int(rank),
        "stage_timings": dict(stage_timings),
        "local_problem": {
            "owned_free_dofs": int(assembler.layout.hi - assembler.layout.lo),
            "overlap_total_dofs": int(assembler.local_data.local_total_nodes.size),
            "local_elements": int(assembler.local_data.local_elem_idx.size),
            "owned_nnz": int(assembler.layout.owned_rows.size),
            "vector_block_size": int(getattr(assembler, "block_size", 1)),
        },
        "assembler_setup": assembler.setup_summary(),
        "assembler_memory": assembler.memory_summary(),
        "assembly_callbacks": assembler.callback_summary(),
        "linear_history": list(linear_records),
        "solve_time_local": float(solve_time),
        "total_time_local": float(local_total_time),
    }
    parallel_diagnostics = comm.gather(local_parallel_diag, root=0)
    summary_diagnostics = (
        list(parallel_diagnostics)
        if rank == 0 and parallel_diagnostics is not None
        else [local_parallel_diag]
    )

    if getattr(args, "state_out", "") and rank == 0:
        export_plasticity3d_state_npz(
            args.state_out,
            coords_ref=coords_ref,
            x_final=coords_final,
            tetrahedra=np.asarray(params["elems_scalar"], dtype=np.int32),
            surface_faces=np.asarray(params["surf"], dtype=np.int32),
            boundary_label=np.asarray(params["boundary_label"], dtype=np.int32),
            mesh_name=str(mesh_name),
            element_degree=int(args.elem_degree),
            lambda_target=float(lambda_target),
            energy=float(result["fun"]),
            metadata={
                "solver_family": "jax_petsc",
                "prototype_mode": "zero_history_endpoint",
                "assembly_backend": str(assembler.assembly_backend),
                "local_hessian_mode": str(assembler.local_hessian_mode),
                "autodiff_tangent_mode": str(assembler.autodiff_tangent_mode),
                "davis_type": str(params["davis_type"]),
                "mpi_ranks": int(comm.size),
                "constraint_variant": str(params["constraint_variant"]),
                "quadrature_rule_id": str(params["quadrature_rule_id"]),
            },
        )

    def _finite_result_value(key: str) -> float | None:
        value = result.get(key)
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    last_dual_metadata = dict(result.get("dual_residual_metadata", {}))
    convergence_payload = {
        "configuration": dict(convergence_configuration),
        "metric": dict(
            result.get("convergence_metric", convergence_configuration["metric"])
        ),
        "absolute_dual_residual": {
            "value": _finite_result_value("dual_residual_norm"),
            "units": str(convergence_configuration["absolute_dual_residual_units"]),
        },
        "initial_absolute_dual_residual": {
            "value": _finite_result_value("initial_dual_residual_norm"),
            "units": str(convergence_configuration["absolute_dual_residual_units"]),
        },
        "initial_relative_dual_residual": {
            "value": _finite_result_value("dual_residual_relative"),
            "units": "dimensionless",
        },
        "residual_gate": {
            "absolute_tolerance": float(args.tolg),
            "absolute_tolerance_units": str(
                convergence_configuration["absolute_dual_residual_units"]
            ),
            "initial_relative_tolerance": float(args.tolg_rel),
            "initial_relative_tolerance_units": "dimensionless",
            "effective_absolute_target": _finite_result_value(
                "dual_residual_target"
            ),
            "passed": bool(result.get("dual_residual_gate_pass", False)),
        },
        "absolute_correction": {
            "value": _finite_result_value("correction_norm"),
            "units": str(convergence_configuration["absolute_dual_residual_units"]),
        },
        "relative_correction": {
            "value": _finite_result_value("relative_correction"),
            "units": "dimensionless",
        },
        "state_norm": {
            "value": _finite_result_value("state_norm"),
            "units": str(convergence_configuration["absolute_dual_residual_units"]),
        },
        "state_scale": {
            "value": float(convergence_state_scale),
            "units": str(convergence_configuration["absolute_dual_residual_units"]),
            "source": str(convergence_configuration["state_scale_source"]),
        },
        "coefficient_gradient_l2": _finite_result_value("grad_norm_coefficient_l2"),
        "last_dual_norm_evaluation": dict(last_dual_metadata),
        "last_riesz_solve": (
            dict(last_dual_metadata)
            if last_dual_metadata.get("riesz_solve") == "iterative"
            else None
        ),
    }

    for nullspace in mg_nullspaces_live:
        nullspace.destroy()
    elastic_operator = regularization_state.get("elastic_operator")
    if elastic_operator is not None:
        elastic_operator.destroy()
    residual_ax.destroy()
    residual_vec.destroy()
    result.pop("x", None)
    x.destroy()
    assembler.cleanup()
    if mg_hierarchy is not None:
        mg_hierarchy.cleanup()

    payload = {
        "mesh_name": str(mesh_name),
        "elem_degree": int(args.elem_degree),
        "quadrature_rule_id": str(params["quadrature_rule_id"]),
        "quadrature_points": int(
            np.asarray(
                params.get("_distributed_quad_weight", params.get("quad_weight")),
                dtype=np.float64,
            ).shape[1]
        ),
        "lambda_target": float(lambda_target),
        "profile": str(args.profile),
        "pc_type": str(settings["pc_type"]),
        "ksp_type": str(settings["ksp_type"]),
        "nit": int(result["nit"]),
        "energy": float(result["fun"]),
        "message": str(result["message"]),
        "status": str(result_status),
        "solver_success": bool(solver_success),
        "solve_time": float(solve_time),
        "total_time": float(local_total_time),
        "assembly_backend": str(assembler.assembly_backend),
        "assembly_backend_requested": str(assembler.assembly_backend_requested),
        "autodiff_tangent_mode": str(assembler.autodiff_tangent_mode),
        "matrix_type": str(matrix_type),
        "p4_hessian_chunk_size": int(assembler.p4_hessian_chunk_size),
        "p4_chunk_autotune": dict(assembler.p4_chunk_autotune_meta or {}),
        "p4_chunk_scatter_cache": dict(assembler.p4_chunk_scatter_cache_meta or {}),
        "petsc_log_view_path": str(petsc_log_view_path or ""),
        "jax_trace_dir": str(getattr(args, "jax_trace_dir", "") or ""),
        "linear_iterations_total": int(sum(linear_iters)),
        "linear_iterations_last": int(linear_iters[-1] if linear_iters else 0),
        "linear_history": list(linear_records),
        "initial_guess": dict(initial_guess_meta or {}),
        "u_max": float(u_max),
        "omega": float(omega),
        "final_grad_norm": float(final_grad_norm),
        "final_grad_norm_kind": "coefficient_l2_diagnostic",
        "nonlinear_convergence": dict(convergence_payload),
        "assembly_callbacks": assembler.callback_summary(),
        "assembler_setup": assembler.setup_summary(),
        "assembler_memory": assembler.memory_summary(),
        "stage_timings": dict(stage_timings),
        "local_problem_summary": {
            "owned_free_dofs_min": int(
                min(int(r["local_problem"]["owned_free_dofs"]) for r in summary_diagnostics)
            ),
            "owned_free_dofs_max": int(
                max(int(r["local_problem"]["owned_free_dofs"]) for r in summary_diagnostics)
            ),
            "overlap_total_dofs_min": int(
                min(int(r["local_problem"]["overlap_total_dofs"]) for r in summary_diagnostics)
            ),
            "overlap_total_dofs_max": int(
                max(int(r["local_problem"]["overlap_total_dofs"]) for r in summary_diagnostics)
            ),
            "local_elements_min": int(
                min(int(r["local_problem"]["local_elements"]) for r in summary_diagnostics)
            ),
            "local_elements_max": int(
                max(int(r["local_problem"]["local_elements"]) for r in summary_diagnostics)
            ),
            "owned_nnz_min": int(
                min(int(r["local_problem"]["owned_nnz"]) for r in summary_diagnostics)
            ),
            "owned_nnz_max": int(
                max(int(r["local_problem"]["owned_nnz"]) for r in summary_diagnostics)
            ),
        },
        "mesh": {
            "nodes": int(np.asarray(params["nodes"]).shape[0]),
            "elements": int(np.asarray(params["elems_scalar"]).shape[0]),
            "free_dofs": int(freedofs.size),
            "free_x_dofs": int(np.asarray(params["q_mask"], dtype=bool)[:, 0].sum()),
            "free_y_dofs": int(np.asarray(params["q_mask"], dtype=bool)[:, 1].sum()),
            "free_z_dofs": int(np.asarray(params["q_mask"], dtype=bool)[:, 2].sum()),
        },
        "linear_solver": {
            "ksp_type": str(settings["ksp_type"]),
            "pc_type": str(settings["pc_type"]),
            "ksp_rtol": float(settings["ksp_rtol"]),
            "ksp_max_it": int(settings["ksp_max_it"]),
            "ksp_accept_true_rel": (
                None if ksp_accept_true_rel is None else float(ksp_accept_true_rel)
            ),
            "accept_ksp_maxit_direction": bool(
                getattr(args, "accept_ksp_maxit_direction", True)
            ),
            "guard_ksp_maxit_direction": bool(
                getattr(args, "guard_ksp_maxit_direction", False)
            ),
            "ksp_maxit_direction_true_rel_cap": float(
                getattr(args, "ksp_maxit_direction_true_rel_cap", 6.0e-2)
            ),
            "pc_setup_on_ksp_cap": bool(settings["pc_setup_on_ksp_cap"]),
            "distribution_strategy": str(
                getattr(args, "distribution_strategy", "overlap_p2p")
            ),
            "assembly_backend": str(assembler.assembly_backend),
            "assembly_backend_requested": str(
                assembler.assembly_backend_requested
            ),
            "autodiff_tangent_mode": str(assembler.autodiff_tangent_mode),
            "matrix_type": str(matrix_type),
            "p4_hessian_chunk_size": int(assembler.p4_hessian_chunk_size),
            "p4_chunk_autotune": dict(assembler.p4_chunk_autotune_meta or {}),
            "p4_chunk_scatter_cache": dict(
                assembler.p4_chunk_scatter_cache_meta or {}
            ),
            "problem_build_mode": str(
                getattr(args, "problem_build_mode", "root_bcast")
            ),
            "mg_level_build_mode": str(
                getattr(args, "mg_level_build_mode", "root_bcast")
            ),
            "mg_transfer_build_mode": str(
                getattr(args, "mg_transfer_build_mode", "owned_rows")
            ),
            "element_reorder_mode": str(
                getattr(args, "element_reorder_mode", None) or "block_xyz"
            ),
            "use_near_nullspace": bool(settings["use_near_nullspace"]),
            "mg_strategy": (
                str(resolved_mg_strategy or "")
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_coarse_backend": (
                str(getattr(args, "mg_coarse_backend", None) or "hypre")
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_coarse_ksp_type": (
                str(getattr(args, "mg_coarse_ksp_type", None) or "cg")
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_coarse_pc_type": (
                str(getattr(args, "mg_coarse_pc_type", None) or "hypre")
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_coarse_hypre_nodal_coarsen": (
                int(getattr(args, "mg_coarse_hypre_nodal_coarsen", 6))
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_coarse_hypre_vec_interp_variant": (
                int(getattr(args, "mg_coarse_hypre_vec_interp_variant", 3))
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_coarse_hypre_strong_threshold": (
                getattr(args, "mg_coarse_hypre_strong_threshold", 0.5)
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_coarse_hypre_coarsen_type": (
                str(getattr(args, "mg_coarse_hypre_coarsen_type", None) or "HMIS")
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_coarse_hypre_max_iter": (
                int(getattr(args, "mg_coarse_hypre_max_iter", 2))
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_coarse_hypre_tol": (
                float(getattr(args, "mg_coarse_hypre_tol", 0.0))
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_coarse_hypre_relax_type_all": (
                str(
                    getattr(
                        args,
                        "mg_coarse_hypre_relax_type_all",
                        "symmetric-SOR/Jacobi",
                    )
                )
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_p1_smoother_ksp_type": (
                str(legacy_mg_settings["degree1"].ksp_type)
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_p1_smoother_pc_type": (
                str(legacy_mg_settings["degree1"].pc_type)
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_p1_smoother_steps": (
                int(legacy_mg_settings["degree1"].steps)
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_p2_smoother_ksp_type": (
                str(legacy_mg_settings["degree2"].ksp_type)
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_p2_smoother_pc_type": (
                str(legacy_mg_settings["degree2"].pc_type)
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_p2_smoother_steps": (
                int(legacy_mg_settings["degree2"].steps)
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_p4_smoother_ksp_type": (
                str(legacy_mg_settings["fine"].ksp_type)
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_p4_smoother_pc_type": (
                str(legacy_mg_settings["fine"].pc_type)
                if str(settings["pc_type"]) == "mg"
                else None
            ),
            "mg_p4_smoother_steps": (
                int(legacy_mg_settings["fine"].steps)
                if str(settings["pc_type"]) == "mg"
                else None
            ),
        },
        "newton_regularization": {
            "enabled": bool(regularization_state["enabled"]),
            "r_min": float(regularization_state["r_min"]),
            "r_initial": float(regularization_state["r_initial"]),
            "r_max": float(regularization_state["r_max"]),
            "r_final": float(regularization_state["r"]),
            "retry_max": int(regularization_state["retry_max"]),
            "history": list(regularization_state["history"]),
        },
    }
    if rank == 0:
        payload["parallel_diagnostics"] = list(parallel_diagnostics)
    if bool(getattr(args, "save_history", False)):
        payload["history"] = (
            list(root_iteration_history)
            if rank == 0 and root_iteration_history
            else list(result.get("history", []))
        )
    if mg_hierarchy is not None:
        payload["mg_hierarchy"] = dict(mg_hierarchy.build_metadata or {})
        payload["transfer_backend"] = str(transfer_backend)
    if mg_nullspace_meta is not None:
        payload["mg_level_metadata"] = list(mg_nullspace_meta.get("levels", []))
    if rank == 0 and not bool(getattr(args, "quiet", False)):
        print(
            f"3D slope-stability solve | mesh={mesh_name} degree={int(args.elem_degree)} "
            f"lambda={lambda_target:.4f} nit={int(result['nit'])} "
            f"energy={float(result['fun']):.6e}",
            flush=True,
        )
    if progress_path and rank == 0:
        final_history = (
            list(root_iteration_history)
            if root_iteration_history
            else list(result.get("history", []))
        )
        _write_progress_payload(
            progress_path,
            {
                "status": str(result_status),
                "message": str(result["message"]),
                "mesh_name": str(mesh_name),
                "elem_degree": int(args.elem_degree),
                "lambda_target": float(lambda_target),
                "assembly_backend": str(assembler.assembly_backend),
                "matrix_type": str(matrix_type),
                "p4_hessian_chunk_size": int(assembler.p4_hessian_chunk_size),
                "p4_chunk_autotune": dict(assembler.p4_chunk_autotune_meta or {}),
                "p4_chunk_scatter_cache": dict(
                    assembler.p4_chunk_scatter_cache_meta or {}
                ),
                "iterations_completed": int(result["nit"]),
                "energy": float(result["fun"]),
                "history": final_history,
                "convergence": dict(convergence_payload),
                "newton_regularization": {
                    "enabled": bool(regularization_state["enabled"]),
                    "current_r": float(regularization_state["r"]),
                    "history": list(regularization_state["history"]),
                },
            },
        )
    _flush_petsc_log_view(petsc_log_view_path, comm)
    return payload
