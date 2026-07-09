#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import REPO_ROOT


BUNDLE_ROOT = REPO_ROOT / "artifacts" / "reproduction" / "paper_submission_2026_07_08"
INPUT_ROOT = BUNDLE_ROOT / "inputs"

P3D_VALIDATION_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_validation"
P3D_ABLATION_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_derivative_ablation"
JAX_FEM_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "jax_fem_hyperelastic_baseline"
P3D_SCALING_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "plasticity3d_p4_l1_2_mumps_pmg_step_grad_local_karolina_scaling"
)
GLOBALIZATION_REPORT = REPO_ROOT / "artifacts" / "reports" / "globalization_method_compare" / "full_summary.csv"
GLOBALIZATION_RAW_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "globalization_method_compare" / "full"
P3D_GLOBALIZATION_OUTPUTS = (
    (
        GLOBALIZATION_RAW_ROOT / "plasticity3d_p2_l1_np32_lambda155_newton_linesearch" / "output.json",
        INPUT_ROOT
        / "globalization_method_compare"
        / "plasticity3d_p2_l1_np32_lambda155_newton_linesearch"
        / "output.json",
    ),
    (
        GLOBALIZATION_RAW_ROOT / "plasticity3d_p2_l1_np32_lambda155_steihaug_trust" / "output.json",
        INPUT_ROOT / "globalization_method_compare" / "plasticity3d_p2_l1_np32_lambda155_steihaug_trust" / "output.json",
    ),
    (
        GLOBALIZATION_RAW_ROOT / "plasticity3d_p2_l1_np32_lambda155_hybrid_trust_linesearch" / "output.json",
        INPUT_ROOT
        / "globalization_method_compare"
        / "plasticity3d_p2_l1_np32_lambda155_hybrid_trust_linesearch"
        / "output.json",
    ),
)
DERIVATIVE_ROUTE_REPORT = REPO_ROOT / "artifacts" / "reports" / "derivative_route_compare" / "full_summary.csv"
SUPPLEMENTAL_REPORT_ROOT = REPO_ROOT / "artifacts" / "reports" / "paper_reviewer_gap_experiments"
SUPPLEMENTAL_GL_TIMEOUT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "paper_reviewer_gap_experiments"
    / "full"
    / "gl_globalization"
    / "gl_l10_newton_linesearch_np8"
)
P2D_SHOWCASE_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "docs_showcase" / "mc_plasticity_p4_l5"
P2D_L6_SUMMARY = (
    REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_l6_p4_deep_p1_tail_scaling_lambda1_maxit20" / "summary.json"
)
P2D_L7_SUMMARY = (
    REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_l7_p4_deep_p1_tail_scaling_lambda1_maxit20" / "summary.json"
)
SOURCE_CONT_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "source_compare"
P3D_DEGREE_ENERGY_STUDY_SUMMARY = (
    REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_lambda1p55_degree_mesh_energy_study" / "comparison_summary.json"
)
P3D_DEGREE_ENERGY_PLOT_FIELDS = (
    "degree_line",
    "mesh_alias",
    "free_dofs",
    "total_time_s",
    "energy",
    "status",
)
P3D_STATE_PAIR_ARRAYS = INPUT_ROOT / "plasticity3d_figure_derived" / "state_pair_surface.npz"
P3D_STATE_PAIR_METADATA = INPUT_ROOT / "plasticity3d_figure_derived" / "state_pair_surface.json"
P3D_SLICE_ARRAYS = INPUT_ROOT / "plasticity3d_figure_derived" / "highest_y_slice_panels.npz"
P3D_SLICE_METADATA = INPUT_ROOT / "plasticity3d_figure_derived" / "highest_y_slice_panels.json"
P3D_RECOMMENDED_SCALING_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling"
    / "comparison_summary.json"
)
P3D_REFERENCE_FORMULA_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_l1_2_lambda1_grad1e2_scaling"
    / "comparison_summary.json"
)
P3D_FIXED_REFERENCE_OPERATOR_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_l1_2_lambda1_grad1e2_scaling_all_pmg"
    / "comparison_summary.json"
)
P3D_FIXED_REFERENCE_TABLE_IMPLS = {
    "local_constitutiveAD_local_pmg_sourcefixed_armijo": "constitutive_ad_fixed_reference_pmg",
    "source_local_pmg_sourcefixed_armijo": "reference_formula_fixed_reference_pmg",
}
P3D_FIXED_REFERENCE_TABLE_FIELDS = (
    "ranks",
    "implementation",
    "wall_time_s",
    "nit",
    "linear_iterations_total",
    "final_metric",
    "final_metric_name",
    "status",
)
P3D_LAMBDA155_STOP_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "example_runs"
    / "plasticity3d_p4_l1_2_lambda1p55_mumps_pmg_step_grad_convergence_20260507_190225"
    / "step_grad_convergence_summary.csv"
)
P3D_DIRECT_BRANCH_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "debug" / "p2_direct_branch_lambda1p6_merged"
SOURCE_BRANCH_ROOT = (
    REPO_ROOT
    / "tmp"
    / "source_compare"
    / "slope_stability_octave_ref"
    / "slope_stability"
    / "artifacts"
    / "compare_direct_branch_lambda1p6"
)
_SHA256_CACHE: dict[Path, str] = {}
P3D_RECOMMENDED_SCALING_OUTPUTS = (
    (
        REPO_ROOT
        / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling/runs/np1/solver_local_pmg/assembly_local_constitutiveAD/output.json",
        INPUT_ROOT / "plasticity3d_recommended_scaling/runs/np1/output.json",
    ),
    (
        REPO_ROOT
        / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling/runs/np2/solver_local_pmg/assembly_local_constitutiveAD/output.json",
        INPUT_ROOT / "plasticity3d_recommended_scaling/runs/np2/output.json",
    ),
    (
        REPO_ROOT
        / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_scaling/runs/np4/solver_local_pmg/assembly_local_constitutiveAD/output.json",
        INPUT_ROOT / "plasticity3d_recommended_scaling/runs/np4/output.json",
    ),
    (
        REPO_ROOT
        / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_scaling/runs/np8/solver_local_pmg/assembly_local_constitutiveAD/output.json",
        INPUT_ROOT / "plasticity3d_recommended_scaling/runs/np8/output.json",
    ),
    (
        REPO_ROOT
        / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_scaling/runs/np16/solver_local_pmg/assembly_local_constitutiveAD/output.json",
        INPUT_ROOT / "plasticity3d_recommended_scaling/runs/np16/output.json",
    ),
    (
        REPO_ROOT
        / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_scaling/runs/np32/solver_local_pmg/assembly_local_constitutiveAD/output.json",
        INPUT_ROOT / "plasticity3d_recommended_scaling/runs/np32/output.json",
    ),
)


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _source_id(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative_to_source_branch = resolved.relative_to(SOURCE_BRANCH_ROOT.resolve())
    except ValueError:
        rel = _repo_rel(path)
        if rel.startswith("artifacts/reports/paper_reviewer_gap_experiments/"):
            return rel.replace(
                "artifacts/reports/paper_reviewer_gap_experiments/",
                "artifacts/reports/supplemental_solver_evidence/",
                1,
            )
        if rel.startswith("artifacts/raw_results/paper_reviewer_gap_experiments/"):
            return rel.replace(
                "artifacts/raw_results/paper_reviewer_gap_experiments/",
                "artifacts/raw_results/supplemental_solver_evidence/",
                1,
            )
        return rel
    return f"external_reference/slope_stability_octave_ref/compare_direct_branch_lambda1p6/{relative_to_source_branch.as_posix()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_memo(path: Path) -> str:
    resolved = path.resolve()
    if resolved not in _SHA256_CACHE:
        _SHA256_CACHE[resolved] = _sha256(resolved)
    return _SHA256_CACHE[resolved]


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sanitize_string(value: str) -> str:
    repo_prefix = REPO_ROOT.resolve().as_posix() + "/"
    text = value.replace(repo_prefix, "")
    bundle_prefix = _repo_rel(BUNDLE_ROOT)
    replacements = {
        "artifacts/raw_results/jax_fem_hyperelastic_baseline/comparison_summary.json": (
            f"{bundle_prefix}/inputs/jax_fem_hyperelastic_baseline/comparison_summary.json"
        ),
        "artifacts/raw_results/jax_fem_hyperelastic_baseline/parity/repo_serial_direct_state.npz": (
            f"{bundle_prefix}/inputs/jax_fem_hyperelastic_baseline/parity/repo_serial_direct_state.npz"
        ),
        "artifacts/raw_results/jax_fem_hyperelastic_baseline/parity/jax_fem_umfpack_serial_state.npz": (
            f"{bundle_prefix}/inputs/jax_fem_hyperelastic_baseline/parity/jax_fem_umfpack_serial_state.npz"
        ),
        "artifacts/raw_results/jax_fem_hyperelastic_baseline/parity/repo_serial_direct.json": (
            f"{bundle_prefix}/inputs/jax_fem_hyperelastic_baseline/parity/repo_serial_direct.json"
        ),
        "artifacts/raw_results/jax_fem_hyperelastic_baseline/parity/jax_fem_umfpack_serial.json": (
            f"{bundle_prefix}/inputs/jax_fem_hyperelastic_baseline/parity/jax_fem_umfpack_serial.json"
        ),
        "tmp/source_compare/slope_stability_octave_ref/slope_stability/artifacts/compare_direct_branch_lambda1p6/final_source_state.mat": (
            f"{bundle_prefix}/inputs/plasticity3d_validation/source_branch/final_source_state.mat"
        ),
        "tmp/source_compare/slope_stability_octave_ref/slope_stability/artifacts/compare_direct_branch_lambda1p6": (
            f"{bundle_prefix}/inputs/plasticity3d_validation/source_branch"
        ),
        "artifacts/raw_results/debug/p2_direct_branch_lambda1p6_merged": (
            f"{bundle_prefix}/inputs/plasticity3d_validation/maintained_branch"
        ),
        "tmp/source_compare/slope_stability_petsc4py": "external_reference/slope_stability_petsc4py",
        "artifacts/raw_results/paper_reviewer_gap_experiments": "archive_source/supplemental_solver_evidence",
        "artifacts/reports/paper_reviewer_gap_experiments": "archive_source/supplemental_solver_evidence",
        ".venv/bin/python": "python",
        ".venv/lib/python3.12/site-packages": "python-environment/site-packages",
        "local_env/python/bin/python3.12": "python",
        "local_env": "python-environment",
        "tmp_work/jax_fem_0_0_10_py312/bin/python": "python",
        "tmp_work/jax_fem_0_0_10_py312": "external_environment/jax_fem_0_0_10_py312",
    }
    for source, dest in P3D_RECOMMENDED_SCALING_OUTPUTS:
        replacements[_repo_rel(source)] = _repo_rel(dest)
    for source, dest in P3D_GLOBALIZATION_OUTPUTS:
        replacements[_repo_rel(source)] = _repo_rel(dest)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _copy_json(source: Path, dest: Path, copied: list[dict[str, str]]) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    sanitized = _sanitize_json_value(payload)
    _write_json(dest, sanitized)
    copied.append(
        {
            "source_path": _source_id(source),
            "bundle_path": _repo_rel(dest),
            "source_sha256": _sha256(source),
            "bundle_sha256": _sha256(dest),
            "sanitization": "local paths rewritten to archive-neutral references; non-finite JSON numbers written as null",
        }
    )


def _copy_binary(source: Path, dest: Path, copied: list[dict[str, str]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    copied.append(
        {
            "source_path": _source_id(source),
            "bundle_path": _repo_rel(dest),
            "source_sha256": _sha256(source),
            "bundle_sha256": _sha256(dest),
            "sanitization": "byte-for-byte copy",
        }
    )


def _copy_csv(source: Path, dest: Path, copied: list[dict[str, str]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        rows = [
            {key: _sanitize_string(value) if isinstance(value, str) else value for key, value in row.items()}
            for row in reader
        ]
        fieldnames = list(reader.fieldnames or [])
    with dest.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    copied.append(
        {
            "source_path": _source_id(source),
            "bundle_path": _repo_rel(dest),
            "source_sha256": _sha256(source),
            "bundle_sha256": _sha256(dest),
            "sanitization": "local paths rewritten to archive-neutral references",
        }
    )


def _copy_fixed_reference_table_summary(source: Path, dest: Path, copied: list[dict[str, str]]) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for raw_row in payload.get("rows", []):
        if not isinstance(raw_row, dict):
            continue
        implementation = str(raw_row.get("implementation", ""))
        if implementation not in P3D_FIXED_REFERENCE_TABLE_IMPLS:
            continue
        row = {field: _sanitize_json_value(raw_row[field]) for field in P3D_FIXED_REFERENCE_TABLE_FIELDS if field in raw_row}
        row["implementation"] = P3D_FIXED_REFERENCE_TABLE_IMPLS[implementation]
        rows.append(row)
    rows.sort(key=lambda row: (int(row.get("ranks", 10**6)), str(row.get("implementation", ""))))
    _write_json(
        dest,
        {
            "description": (
                "Table-specific release summary for the Plasticity3D fixed-reference "
                "operator PMG diagnostic. Route identifiers are paper-facing aliases "
                "of the two compared implementations; numerical values are copied "
                "from the full comparison summary."
            ),
            "rows": rows,
        },
    )
    copied.append(
        {
            "source_path": _source_id(source),
            "bundle_path": _repo_rel(dest),
            "source_sha256": _sha256(source),
            "bundle_sha256": _sha256(dest),
            "sanitization": "table-specific fields copied; implementation route identifiers rewritten to paper-facing aliases",
        }
    )


def _finite_json_number(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _copy_p3d_degree_plot_summary(source: Path, dest: Path, copied: list[dict[str, str]]) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    dependencies: list[dict[str, str]] = [{"source_path": _source_id(source), "source_sha256": _sha256(source)}]
    for raw_row in payload.get("rows", []):
        if not isinstance(raw_row, dict):
            continue
        row = {field: _sanitize_json_value(raw_row[field]) for field in P3D_DEGREE_ENERGY_PLOT_FIELDS if field in raw_row}
        result_path = REPO_ROOT / str(raw_row["result_json"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        dependencies.append({"source_path": _source_id(result_path), "source_sha256": _sha256(result_path)})
        history: list[dict[str, object]] = []
        for index, item in enumerate(result.get("history", [])):
            if not isinstance(item, dict):
                continue
            grad_norm = _finite_json_number(item.get("grad_norm_post"))
            if grad_norm is None:
                grad_norm = _finite_json_number(item.get("grad_norm"))
            history.append(
                {
                    "it": int(item.get("it", index + 1)),
                    "energy": _finite_json_number(item.get("energy")),
                    "grad_norm": grad_norm,
                }
            )
        row["history"] = history
        rows.append(row)
    rows.sort(
        key=lambda row: (
            int(str(row.get("degree_line", "P0")).replace("P", "")),
            int(row.get("free_dofs", 0)),
        )
    )
    _write_json(
        dest,
        {
            "description": (
                "Plot-specific release summary for Plasticity3D degree/mesh energy "
                "and convergence figures. It contains scalar study rows and "
                "Newton-history traces needed by the submitted plots, without raw "
                "state-array or mesh-file paths."
            ),
            "rows": rows,
            "schema": "paper_figure/plasticity3d_degree_energy_plot_summary/v1",
        },
    )
    copied.append(
        {
            "source_path": _source_id(source),
            "bundle_path": _repo_rel(dest),
            "source_sha256": _sha256(source),
            "bundle_sha256": _sha256(dest),
            "sanitization": "plot-specific scalar rows and histories copied without raw state or mesh paths",
            "source_dependencies": dependencies,
        }
    )


def _source_dependency(path: Path) -> dict[str, str]:
    return {
        "source_path": _source_id(path),
        "source_sha256": _sha256_memo(path),
    }


def _row_repo_path(row: dict[str, Any], key: str) -> Path:
    path = Path(str(row[key]))
    return path if path.is_absolute() else REPO_ROOT / path


def _copy_p3d_state_pair_surface(source: Path, metadata_dest: Path, arrays_dest: Path, copied: list[dict[str, str]]) -> None:
    import numpy as np
    from generate_paper_figures import (
        P3D_BENCHMARK_MESH_ALIAS,
        P3D_BENCHMARK_DEGREE_LINE,
        P3D_BENCHMARK_SURFACE_SUBDIVISIONS,
        P3D_CAMERA_POSITION,
        P3D_CAMERA_TARGET,
        _plasticity3d_dev_strain_data,
        plasticity3d_surface_plot_arrays,
    )
    from src.problems.slope_stability_3d.support.mesh import load_case_hdf5

    payload = json.loads(source.read_text(encoding="utf-8"))
    row = next(
        dict(item)
        for item in payload.get("rows", [])
        if isinstance(item, dict)
        and str(item.get("degree_line", "")) == P3D_BENCHMARK_DEGREE_LINE
        and str(item.get("mesh_alias", "")) == P3D_BENCHMARK_MESH_ALIAS
    )
    state_path = _row_repo_path(row, "state_npz")
    case_path = _row_repo_path(row, "same_mesh_case_path")
    with np.load(state_path) as state:
        coords_final = np.asarray(state["coords_final"], dtype=np.float64)
        displacement = np.asarray(state["displacement"], dtype=np.float64)
        surface_faces = np.asarray(state["surface_faces"], dtype=np.int64)
    degree = int(row["elem_degree"])
    nodal_disp_mag = np.linalg.norm(displacement, axis=1)
    case = load_case_hdf5(case_path)
    nodal_dev, _, _ = _plasticity3d_dev_strain_data(
        coords_final=coords_final,
        displacement=displacement,
        case=case,
        degree=degree,
        chunk_size=256,
    )
    coords_plot, tri_plot, disp_values = plasticity3d_surface_plot_arrays(
        coords_final,
        surface_faces,
        nodal_disp_mag,
        degree=degree,
        subdivisions=P3D_BENCHMARK_SURFACE_SUBDIVISIONS,
    )
    _, _, dev_values = plasticity3d_surface_plot_arrays(
        coords_final,
        surface_faces,
        nodal_dev,
        degree=degree,
        subdivisions=P3D_BENCHMARK_SURFACE_SUBDIVISIONS,
    )
    tri_indices = np.asarray(tri_plot, dtype=np.int64)
    tri_xyz = coords_plot[tri_indices]
    disp_tri_vals = np.mean(disp_values[tri_indices], axis=1)
    dev_tri_vals = np.mean(dev_values[tri_indices], axis=1)
    arrays_dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_dest,
        tri_xyz=np.asarray(tri_xyz, dtype=np.float64),
        disp_tri_vals=np.asarray(disp_tri_vals, dtype=np.float64),
        dev_tri_vals=np.asarray(dev_tri_vals, dtype=np.float64),
    )
    metadata = {
        "schema": "paper_figure/plasticity3d_state_pair_surface/v1",
        "description": (
            "Derived plotted surface triangles for the Plasticity3D state-pair figure. "
            "The arrays contain the final triangulated geometry and per-triangle "
            "displacement/deviatoric-strain values used for rendering."
        ),
        "degree_line": str(row["degree_line"]),
        "mesh_alias": str(row["mesh_alias"]),
        "degree": degree,
        "subdivisions": P3D_BENCHMARK_SURFACE_SUBDIVISIONS,
        "disp_norm_vmax": float(np.max(nodal_disp_mag)),
        "dev_norm_vmax": float(max(float(np.quantile(np.asarray(nodal_dev, dtype=np.float64), 0.995)), 1.0e-12)),
        "camera_target": [float(value) for value in np.asarray(P3D_CAMERA_TARGET, dtype=np.float64)],
        "camera_position": [float(value) for value in np.asarray(P3D_CAMERA_POSITION, dtype=np.float64)],
        "source_dependencies": [
            _source_dependency(source),
            _source_dependency(state_path),
            _source_dependency(case_path),
        ],
    }
    _write_json(metadata_dest, metadata)
    for dest, label in ((metadata_dest, "metadata"), (arrays_dest, "arrays")):
        copied.append(
            {
                "source_path": _source_id(source),
                "bundle_path": _repo_rel(dest),
                "source_sha256": _sha256_memo(source),
                "bundle_sha256": _sha256(dest),
                "sanitization": f"derived Plasticity3D state-pair surface {label} with source hashes",
            }
        )


def _copy_p3d_highest_y_slice_panels(
    source: Path, metadata_dest: Path, arrays_dest: Path, copied: list[dict[str, str]]
) -> None:
    import numpy as np
    from generate_paper_figures import _build_highest_y_slice, _highest_rows_by_degree

    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload.get("rows", []) if isinstance(row, dict)]
    selected_rows = _highest_rows_by_degree(rows)
    slices = [_build_highest_y_slice(row) for row in selected_rows]
    image_arrays = [np.asarray(item["image"], dtype=np.float64) for item in slices]
    extents = np.asarray([np.asarray(item["extent"], dtype=np.float64) for item in slices], dtype=np.float64)
    finite_arrays = [image[np.isfinite(image)] for image in image_arrays if np.any(np.isfinite(image))]
    vmax = float(np.quantile(np.concatenate(finite_arrays), 0.995))
    zlim = [float(np.min(extents[:, 2])), float(np.max(extents[:, 3]))]
    arrays_dest.parent.mkdir(parents=True, exist_ok=True)
    image_payload = {
        f"image_{str(row['degree_line']).lower()}": image
        for row, image in zip(selected_rows, image_arrays, strict=True)
    }
    np.savez_compressed(arrays_dest, extents=extents, **image_payload)
    dependencies: list[dict[str, str]] = [_source_dependency(source)]
    panels: list[dict[str, object]] = []
    for row, item in zip(selected_rows, slices, strict=True):
        state_path = _row_repo_path(row, "state_npz")
        case_path = _row_repo_path(row, "same_mesh_case_path")
        dependencies.extend([_source_dependency(state_path), _source_dependency(case_path)])
        panels.append(
            {
                "degree_line": str(row["degree_line"]),
                "image_key": f"image_{str(row['degree_line']).lower()}",
                "mesh_alias": str(row["mesh_alias"]),
                "free_dofs": int(row["free_dofs"]),
                "energy": float(row["energy"]),
                "extent": [float(value) for value in item["extent"]],
            }
        )
    metadata = {
        "schema": "paper_figure/plasticity3d_highest_y_slice_panels/v1",
        "description": (
            "Derived interpolated y-slice panels for the Plasticity3D highest-mesh "
            "comparison figure. The arrays contain the final images and extents "
            "used for rendering."
        ),
        "axis": 1,
        "center_fraction": 0.62,
        "half_thickness_fraction": 0.02,
        "resolution": 900,
        "smooth_sigma": 1.0,
        "xlim": [-150.0, -50.0],
        "zlim": zlim,
        "global_vmax": vmax,
        "panels": panels,
        "source_dependencies": dependencies,
    }
    _write_json(metadata_dest, metadata)
    for dest, label in ((metadata_dest, "metadata"), (arrays_dest, "arrays")):
        copied.append(
            {
                "source_path": _source_id(source),
                "bundle_path": _repo_rel(dest),
                "source_sha256": _sha256_memo(source),
                "bundle_sha256": _sha256(dest),
                "sanitization": f"derived Plasticity3D highest-y-slice {label} with source hashes",
            }
        )


def main() -> None:
    if BUNDLE_ROOT.exists():
        shutil.rmtree(BUNDLE_ROOT)
    copied: list[dict[str, str]] = []
    _copy_json(
        P3D_VALIDATION_ROOT / "validation_manifest.json",
        INPUT_ROOT / "plasticity3d_validation" / "validation_manifest.json",
        copied,
    )
    _copy_json(
        P3D_VALIDATION_ROOT / "comparison_summary.json",
        INPUT_ROOT / "plasticity3d_validation" / "comparison_summary.json",
        copied,
    )
    _copy_json(
        SOURCE_BRANCH_ROOT / "branch_summary.json",
        INPUT_ROOT / "plasticity3d_validation" / "source_branch" / "branch_summary.json",
        copied,
    )
    _copy_binary(
        SOURCE_BRANCH_ROOT / "final_source_state.mat",
        INPUT_ROOT / "plasticity3d_validation" / "source_branch" / "final_source_state.mat",
        copied,
    )
    _copy_json(
        P3D_DIRECT_BRANCH_ROOT / "branch_summary.json",
        INPUT_ROOT / "plasticity3d_validation" / "maintained_branch" / "branch_summary.json",
        copied,
    )
    _copy_json(
        P3D_ABLATION_ROOT / "comparison_summary.json",
        INPUT_ROOT / "plasticity3d_derivative_ablation" / "comparison_summary.json",
        copied,
    )
    _copy_json(
        JAX_FEM_ROOT / "comparison_summary.json",
        INPUT_ROOT / "jax_fem_hyperelastic_baseline" / "comparison_summary.json",
        copied,
    )
    _copy_json(
        JAX_FEM_ROOT / "run_manifest.json",
        INPUT_ROOT / "jax_fem_hyperelastic_baseline" / "run_manifest.json",
        copied,
    )
    for name in (
        "repo_serial_direct.json",
        "jax_fem_umfpack_serial.json",
        "repo_serial_direct_state.npz",
        "jax_fem_umfpack_serial_state.npz",
    ):
        source = JAX_FEM_ROOT / "parity" / name
        dest = INPUT_ROOT / "jax_fem_hyperelastic_baseline" / "parity" / name
        if source.suffix == ".json":
            _copy_json(source, dest, copied)
        else:
            _copy_binary(source, dest, copied)
    for name in ("local_solver_total_scaling.csv", "karolina_rpn16_solver_total_scaling.csv"):
        _copy_csv(
            P3D_SCALING_ROOT / name,
            INPUT_ROOT / "plasticity3d_lambda155_scaling" / name,
            copied,
        )
    _copy_csv(
        P3D_LAMBDA155_STOP_SUMMARY,
        INPUT_ROOT / "plasticity3d_lambda155_scaling" / "step_grad_convergence_summary.csv",
        copied,
    )
    _copy_csv(GLOBALIZATION_REPORT, INPUT_ROOT / "globalization_method_compare" / "full_summary.csv", copied)
    for source, dest in P3D_GLOBALIZATION_OUTPUTS:
        _copy_json(source, dest, copied)
    _copy_csv(DERIVATIVE_ROUTE_REPORT, INPUT_ROOT / "derivative_route_compare" / "full_summary.csv", copied)
    for name in (
        "full_he_distribution.csv",
        "full_he_pmg.csv",
        "full_topology_consistency.csv",
        "full_gl_globalization.csv",
        "full_p3d_derivative_degree.csv",
    ):
        _copy_csv(SUPPLEMENTAL_REPORT_ROOT / name, INPUT_ROOT / "supplemental_solver_evidence" / name, copied)
    for name in ("run_info.json", "case_metadata.json"):
        _copy_json(
            SUPPLEMENTAL_GL_TIMEOUT_ROOT / name,
            INPUT_ROOT / "supplemental_solver_evidence" / "gl_globalization" / "gl_l10_newton_linesearch_np8" / name,
            copied,
        )
    _copy_json(P2D_SHOWCASE_ROOT / "output.json", INPUT_ROOT / "plasticity2d_resolution" / "output.json", copied)
    _copy_binary(P2D_SHOWCASE_ROOT / "state.npz", INPUT_ROOT / "plasticity2d_resolution" / "state.npz", copied)
    _copy_json(P2D_L6_SUMMARY, INPUT_ROOT / "plasticity2d_resolution" / "slope_stability_l6_p4" / "summary.json", copied)
    _copy_json(P2D_L7_SUMMARY, INPUT_ROOT / "plasticity2d_resolution" / "slope_stability_l7_p4" / "summary.json", copied)
    for ranks, root_name in (
        (8, "ssr_indirect_p4_l1_omega6p7e6_np8_shell_default_afterfix"),
        (32, "ssr_indirect_p4_l1_omega6p7e6_np32_shell_default_afterfix"),
    ):
        source_dir = SOURCE_CONT_ROOT / root_name / "data"
        dest_dir = INPUT_ROOT / "plasticity2d_reference_continuation" / f"np{ranks}"
        _copy_json(source_dir / "run_info.json", dest_dir / "run_info.json", copied)
        _copy_json(source_dir / "progress_latest.json", dest_dir / "progress_latest.json", copied)
    _copy_json(
        P3D_DEGREE_ENERGY_STUDY_SUMMARY,
        INPUT_ROOT / "plasticity3d_degree_energy_study" / "comparison_summary.json",
        copied,
    )
    _copy_p3d_degree_plot_summary(
        P3D_DEGREE_ENERGY_STUDY_SUMMARY,
        INPUT_ROOT / "plasticity3d_degree_energy_study" / "plot_summary.json",
        copied,
    )
    _copy_p3d_state_pair_surface(P3D_DEGREE_ENERGY_STUDY_SUMMARY, P3D_STATE_PAIR_METADATA, P3D_STATE_PAIR_ARRAYS, copied)
    _copy_p3d_highest_y_slice_panels(P3D_DEGREE_ENERGY_STUDY_SUMMARY, P3D_SLICE_METADATA, P3D_SLICE_ARRAYS, copied)
    _copy_json(
        P3D_RECOMMENDED_SCALING_SUMMARY,
        INPUT_ROOT / "plasticity3d_recommended_scaling" / "comparison_summary.json",
        copied,
    )
    for source, dest in P3D_RECOMMENDED_SCALING_OUTPUTS:
        _copy_json(source, dest, copied)
    _copy_json(
        P3D_REFERENCE_FORMULA_SUMMARY,
        INPUT_ROOT / "plasticity3d_reference_formula" / "comparison_summary.json",
        copied,
    )
    _copy_json(
        P3D_FIXED_REFERENCE_OPERATOR_SUMMARY,
        INPUT_ROOT / "plasticity3d_fixed_reference_operator" / "comparison_summary.json",
        copied,
    )
    _copy_fixed_reference_table_summary(
        P3D_FIXED_REFERENCE_OPERATOR_SUMMARY,
        INPUT_ROOT / "plasticity3d_fixed_reference_operator" / "table_summary.json",
        copied,
    )

    manifest = {
        "id": "paper_submission_2026_07_08",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_head(),
        "purpose": "Archive-neutral provenance bundle for manuscript-critical paper figure and table inputs.",
        "scope": [
            "Plasticity3D endpoint-surrogate validation inputs",
            "Plasticity3D derivative-route comparison summary",
            "Hyperelastic JAX-FEM comparison summary and terminal states",
            "Plasticity3D lambda=1.55 local/multi-node scaling summaries",
            "Small generated-table report summaries for globalization, derivative-route, and supplemental solver evidence",
            "Plasticity2D endpoint, resolution, and reference-continuation inputs",
            "Plasticity3D degree/energy, recommended-scaling, reference-formula, and fixed-reference summary inputs",
            "Derived Plasticity3D surface, slice, degree-energy, and convergence inputs for submitted figures",
        ],
        "source_files": copied,
        "known_limitations": [
            "This bundle normalizes existing paper-critical provenance without rerunning MPI campaigns.",
            "Large Plasticity3D state arrays and same-mesh HDF5 files are not bundled; derived arrays reproduce the submitted figures, while full raw-state recomputation requires the original raw-results archive.",
            "The full fixed-reference operator summary remains bundled for traceability; the manuscript table reads a table-specific release summary with paper-facing route identifiers.",
            "Target-journal metadata, repository license, and permanent archive DOI remain outside this bundle.",
        ],
        "validation": {
            "paper_asset_check": "python paper/scripts/validate_paper_assets.py",
            "archive_neutral_check": "python paper/scripts/validate_paper_assets.py --archive-neutral",
        },
    }
    _write_json(BUNDLE_ROOT / "manifest.json", manifest)
    print(BUNDLE_ROOT / "manifest.json")


if __name__ == "__main__":
    main()
