"""3D heterogeneous slope-stability solver using JAX autodiff + PETSc."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.benchmark.state_export import export_plasticity3d_state_npz
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
    DEFAULT_MESH_NAME,
    base_mesh_name_for_name,
    build_same_mesh_lagrange_case_data,
    ensure_same_mesh_case_hdf5,
    load_same_mesh_case_hdf5_rank_local,
    ownership_block_size_3d,
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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _format_gib(value: object) -> str:
    return f"{float(value):.3f} GiB"


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
    ensure_same_mesh_case_hdf5(mesh_name, degree)

    build_mode = str(getattr(args, "problem_build_mode", "root_bcast"))
    reorder_mode = str(getattr(args, "element_reorder_mode", None) or "block_xyz")
    if build_mode == "rank_local":
        params = load_same_mesh_case_hdf5_rank_local(
            mesh_name,
            degree,
            reorder_mode=reorder_mode,
            comm=comm,
            block_size=3,
        )
        adjacency = None
    else:
        case_data = build_same_mesh_lagrange_case_data(
            mesh_name,
            degree=degree,
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


def _capture_elastic_operator(
    assembler: SlopeStability3DReorderedElementAssembler,
    regularization_state: dict[str, object],
) -> None:
    if not bool(regularization_state["enabled"]):
        return
    elastic_operator = regularization_state.get("elastic_operator")
    if elastic_operator is not None:
        return
    copied = assembler.A.copy()
    if int(getattr(assembler, "block_size", 1)) > 1:
        copied.setBlockSize(int(getattr(assembler, "block_size", 1)))
    regularization_state["elastic_operator"] = copied


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

    settings = _apply_3d_stack_defaults(args, _resolve_linear_settings(args))
    regularization_state = _init_newton_regularization_state(args)
    pc_options = _pc_options(settings)
    if rank == 0 and debug_setup:
        print("setup: problem load begin", flush=True)
    t_stage = time.perf_counter()
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
        p4_hessian_chunk_size=int(getattr(args, "p4_hessian_chunk_size", 32)),
    )
    stage_timings["assembler_create"] = float(time.perf_counter() - t_stage)
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
        if bool(regularization_state["enabled"]):
            zero_owned = np.zeros(assembler.layout.hi - assembler.layout.lo, dtype=np.float64)
            assembler.assemble_hessian_with_mode(zero_owned, constitutive_mode="elastic")
            _capture_elastic_operator(assembler, regularization_state)

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
        if rank != 0 or not progress_path:
            return
        _write_progress_payload(
            progress_path,
            {
                "status": "running",
                "mesh_name": str(mesh_name),
                "elem_degree": int(args.elem_degree),
                "lambda_target": float(lambda_target),
                "iterations_completed": int(entry.get("it", len(history))),
                "last_iteration": dict(entry),
                "history": list(history),
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
                trust_subproblem_solve_fn if bool(getattr(args, "use_trust_region", False)) else None
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
        )
    except _LinearSolveFailure as exc:
        result = {
            "nit": int(len(linear_iters)),
            "fun": float(assembler.energy_fn(x)),
            "message": str(exc),
            "history": [],
        }
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
    final_grad_vec = x.duplicate()
    try:
        assembler.gradient_fn(x, final_grad_vec)
        final_grad_norm = float(final_grad_vec.norm(PETSc.NormType.NORM_2))
    finally:
        final_grad_vec.destroy()
    solver_success = bool(
        str(result["message"]).lower().startswith("converged")
        and np.isfinite(float(result["fun"]))
        and np.all(np.isfinite(full_original))
    )
    result_status = "completed" if solver_success else "failed"
    local_total_time = float(time.perf_counter() - total_runtime_start)
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
                "davis_type": str(params["davis_type"]),
                "mpi_ranks": int(comm.size),
            },
        )

    for nullspace in mg_nullspaces_live:
        nullspace.destroy()
    elastic_operator = regularization_state.get("elastic_operator")
    if elastic_operator is not None:
        elastic_operator.destroy()
    residual_ax.destroy()
    residual_vec.destroy()
    assembler.cleanup()
    if mg_hierarchy is not None:
        mg_hierarchy.cleanup()

    payload = {
        "mesh_name": str(mesh_name),
        "elem_degree": int(args.elem_degree),
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
        "linear_iterations_total": int(sum(linear_iters)),
        "linear_iterations_last": int(linear_iters[-1] if linear_iters else 0),
        "linear_history": list(linear_records),
        "initial_guess": dict(initial_guess_meta or {}),
        "u_max": float(u_max),
        "omega": float(omega),
        "final_grad_norm": float(final_grad_norm),
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
        payload["history"] = list(result.get("history", []))
    if mg_hierarchy is not None:
        payload["mg_hierarchy"] = dict(mg_hierarchy.build_metadata or {})
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
        _write_progress_payload(
            progress_path,
            {
                "status": str(result_status),
                "message": str(result["message"]),
                "mesh_name": str(mesh_name),
                "elem_degree": int(args.elem_degree),
                "lambda_target": float(lambda_target),
                "iterations_completed": int(result["nit"]),
                "energy": float(result["fun"]),
                "history": list(result.get("history", [])),
                "newton_regularization": {
                    "enabled": bool(regularization_state["enabled"]),
                    "current_r": float(regularization_state["r"]),
                    "history": list(regularization_state["history"]),
                },
            },
        )
    return payload
