#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import scipy.io
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import (
    FIGURES_ROOT,
    PDF_METADATA,
    PNG_METADATA,
    REPO_ROOT,
    configure_paper_matplotlib,
    ensure_paper_dirs,
    load_layout,
    paper_figure_size,
    read_csv_rows,
    save_pdf_and_png,
    text_figure_size,
    write_json,
)
from experiments.analysis import generate_plasticity3d_impl_scaling_assets as impl_assets
from experiments.analysis.docs_assets.common import implementation_style, ideal_strong_scaling
from experiments.analysis.generate_mc_plasticity_p4_docs_assets import _build_refined_plot_data as plasticity2d_build_refined_plot_data
from experiments.analysis.generate_plasticity3d_p2_lambda1p6_docs_assets import _interpolate_planar_slice as plasticity3d_interpolate_planar_slice
from experiments.analysis.generate_plasticity3d_p4_l1_docs_assets import (
    _quadrature_points_tetra as plasticity3d_quadrature_points_tetra,
    _surface_plot_arrays as plasticity3d_surface_plot_arrays,
)
from experiments.analysis.hyperelastic_companion_common import centerline_profile
from src.problems.slope_stability.support.mesh import build_same_mesh_lagrange_case_data
from src.problems.slope_stability_3d.support.mesh import (
    load_case_hdf5,
    same_mesh_case_hdf5_path,
)
from src.problems.slope_stability_3d.support.simplex_lagrange import evaluate_tetra_lagrange_basis


matplotlib.use("Agg")

PAPER_SUBMISSION_BUNDLE_ROOT = REPO_ROOT / "artifacts/reproduction/paper_submission_2026_07_08"
PAPER_SUBMISSION_INPUT_ROOT = PAPER_SUBMISSION_BUNDLE_ROOT / "inputs"
LOCAL_P3D_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_recommended_scaling/comparison_summary.json"
MIXED_P3D_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_reference_formula/comparison_summary.json"
SOURCEFIXED_P3D_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_fixed_reference_operator/comparison_summary.json"
P3D_DEGREE_ENERGY_STUDY_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity3d_degree_energy_study/comparison_summary.json"
)
P3D_DEGREE_ENERGY_STUDY_PLOT_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity3d_degree_energy_study/plot_summary.json"
)
P3D_STATE_PAIR_SURFACE_METADATA = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity3d_figure_derived/state_pair_surface.json"
)
P3D_STATE_PAIR_SURFACE_ARRAYS = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity3d_figure_derived/state_pair_surface.npz"
)
P3D_HIGHEST_Y_SLICE_METADATA = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity3d_figure_derived/highest_y_slice_panels.json"
)
P3D_HIGHEST_Y_SLICE_ARRAYS = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity3d_figure_derived/highest_y_slice_panels.npz"
)
P3D_RECOMMENDED_SCALING_OUTPUTS = tuple(
    PAPER_SUBMISSION_INPUT_ROOT / f"plasticity3d_recommended_scaling/runs/np{ranks}/output.json"
    for ranks in (1, 2, 4, 8, 16, 32)
)
SOURCE_CONT_NP8 = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np8_shell_default_afterfix/data/run_info.json"
)
SOURCE_CONT_NP32 = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np32_shell_default_afterfix/data/run_info.json"
)
PLAPLACE_STATE = REPO_ROOT / "experiments/analysis/docs_assets/data/plaplace/sample_state.npz"
PLAPLACE_ENERGY = REPO_ROOT / "experiments/analysis/docs_assets/data/plaplace/energy_levels.csv"
PLAPLACE_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/plaplace/strong_scaling.csv"
GL_STATE = REPO_ROOT / "experiments/analysis/docs_assets/data/ginzburg_landau/sample_state.npz"
GL_ENERGY = REPO_ROOT / "experiments/analysis/docs_assets/data/ginzburg_landau/energy_levels.csv"
GL_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/ginzburg_landau/strong_scaling.csv"
HYPER_STATE = REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/sample_state.npz"
HYPER_ENERGY = REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/energy_levels.csv"
HYPER_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/strong_scaling.csv"
HYPER_CPU_PMG_SCALING = (
    REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/karolina_l5_pmg_scaling.csv"
)
PLASTICITY2D_STATE = PAPER_SUBMISSION_INPUT_ROOT / "plasticity2d_resolution/state.npz"
PLASTICITY2D_RESULT = PAPER_SUBMISSION_INPUT_ROOT / "plasticity2d_resolution/output.json"
PLASTICITY2D_L6_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity2d_resolution/slope_stability_l6_p4/summary.json"
PLASTICITY2D_L7_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity2d_resolution/slope_stability_l7_p4/summary.json"
TOPOLOGY_STATE = REPO_ROOT / "experiments/analysis/docs_assets/data/topology/parallel_final_state.npz"
TOPOLOGY_HISTORY = REPO_ROOT / "experiments/analysis/docs_assets/data/topology/objective_history.csv"
TOPOLOGY_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/topology/strong_scaling.csv"
P3D_VALIDATION_ROOT = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_validation"
P3D_DERIVATIVE_ABLATION_ROOT = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_derivative_ablation"
JAX_FEM_BASELINE_ROOT = PAPER_SUBMISSION_INPUT_ROOT / "jax_fem_hyperelastic_baseline"
P3D_LOCAL_LAMBDA155_SCALING = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_lambda155_scaling/local_solver_total_scaling.csv"
P3D_MULTINODE_LAMBDA155_SCALING = (
    PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_lambda155_scaling/karolina_rpn16_solver_total_scaling.csv"
)

LOCAL_IMPL = "local_constitutiveAD_local_pmg_armijo"
SOURCE_IMPL = "source_local_pmg_armijo"
LOCAL_SOURCEFIXED_IMPL = "local_constitutiveAD_local_pmg_sourcefixed_armijo"
SOURCE_SOURCEFIXED_IMPL = "source_local_pmg_sourcefixed_armijo"
P3D_CAMERA_TARGET = np.asarray([-100.0, 0.0, 50.0], dtype=np.float64)
P3D_CAMERA_POSITION = np.asarray([-200.0, 100.0, -50.0], dtype=np.float64)
P3D_BENCHMARK_DEGREE_LINE = "P4"
P3D_BENCHMARK_MESH_ALIAS = "L1_2"
P3D_BENCHMARK_SURFACE_SUBDIVISIONS = 2
MESH_ALIAS_MATH = {
    "L1": "L_1",
    "L1_2": "L_2",
    "L1_2_3": "L_3",
    "L1_2_3_4": "L_4",
}


def _manifest_repo_input(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Paper figure manifest input is outside the repository: {path}") from exc
    return {"kind": "repository_path", "path": relative.as_posix()}


PAPER_IMPLEMENTATION_LABELS = {
    "fenics_custom": "FEniCS Newton reference",
    "jax_petsc_element": "JAX+PETSc element AD",
    "jax_petsc_local_sfd": "JAX+PETSc colored recovery",
}


def paper_implementation_style(implementation: str) -> dict[str, object]:
    style = dict(implementation_style(implementation))
    if implementation in PAPER_IMPLEMENTATION_LABELS:
        style["label"] = PAPER_IMPLEMENTATION_LABELS[implementation]
    return style


def _mesh_math(alias: object) -> str:
    key = str(alias)
    if key in MESH_ALIAS_MATH:
        return MESH_ALIAS_MATH[key]
    if key.startswith("L") and key[1:].isdigit():
        return f"L_{key[1:]}"
    return key.replace("_", r"\_")


def _degree_math(degree: object) -> str:
    key = str(degree)
    if key.startswith("P") and key[1:].isdigit():
        return f"P_{key[1:]}"
    return key.replace("_", r"\_")


def _element_math_label(degree: object, mesh_alias: object) -> str:
    return rf"${_degree_math(degree)}({_mesh_math(mesh_alias)})$"


def _degree_math_label(degree: object) -> str:
    return rf"${_degree_math(degree)}$"


def _latex_scientific(value: float, digits: int = 2) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    rounded = round(mantissa, digits)
    if abs(rounded) >= 10.0:
        rounded /= 10.0
        exponent += 1
    return rf"{rounded:.{digits}f}\times 10^{{{exponent}}}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_pdf_trailer_id(path: Path) -> None:
    # pdfcrop preserves a run-dependent first trailer ID even when dates are fixed.
    stable_id = b"31415926535897932384626433832795"
    data = path.read_bytes()
    pattern = rb"/ID \[<[0-9A-Fa-f]{32}><[0-9A-Fa-f]{32}>\]"
    replacement = b"/ID [<" + stable_id + b"><" + stable_id + b">]"
    data, count = re.subn(pattern, replacement, data, count=1)
    if count != 1:
        raise RuntimeError(f"Could not normalize PDF trailer ID in {path}")
    path.write_bytes(data)


def _repo_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _plasticity3d_benchmark_row() -> dict[str, object]:
    summary = _read_json(P3D_DEGREE_ENERGY_STUDY_SUMMARY)
    for row in summary.get("rows", []):
        if not isinstance(row, dict):
            continue
        if (
            str(row.get("degree_line", "")) == P3D_BENCHMARK_DEGREE_LINE
            and str(row.get("mesh_alias", "")) == P3D_BENCHMARK_MESH_ALIAS
        ):
            return dict(row)
    raise RuntimeError(
        f"Could not find Plasticity3D benchmark row "
        f"{P3D_BENCHMARK_DEGREE_LINE}({P3D_BENCHMARK_MESH_ALIAS}) in {P3D_DEGREE_ENERGY_STUDY_SUMMARY}"
    )


def _plasticity3d_study_rows() -> list[dict[str, object]]:
    summary = _read_json(P3D_DEGREE_ENERGY_STUDY_PLOT_SUMMARY)
    rows = [dict(row) for row in summary.get("rows", []) if isinstance(row, dict)]
    rows.sort(
        key=lambda row: (
            int(str(row.get("degree_line", "P0")).replace("P", "")),
            int(row.get("free_dofs", 0)),
        )
    )
    return rows


def _plasticity3d_benchmark_context() -> dict[str, object]:
    row = _plasticity3d_benchmark_row()
    state_path = _repo_path(row["state_npz"])
    result_path = _repo_path(row["result_json"])
    case_path_value = str(row.get("same_mesh_case_path", ""))
    case_path = _repo_path(case_path_value) if case_path_value else None
    if case_path is None or not case_path.exists():
        case_path = same_mesh_case_hdf5_path(
            str(row["mesh_name"]),
            int(row["elem_degree"]),
            str(row.get("constraint_variant", "glued_bottom")),
        )
    return {
        "row": row,
        "state": _load_npz(state_path),
        "result": _read_json(result_path),
        "case": load_case_hdf5(case_path),
        "degree": int(row["elem_degree"]),
    }


def _load_impl_rows(summary_path: Path) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    summary = _read_json(summary_path)
    implementation_order = impl_assets._implementation_order(summary)
    enriched_rows = []
    for row in list(summary.get("rows", [])):
        if not isinstance(row, dict):
            continue
        result_path = impl_assets._repo_path(str(row.get("result_json", "") or ""))
        if result_path.exists():
            enriched_rows.append(impl_assets._enrich_row(dict(row)))
    rows_by_impl = impl_assets._rows_by_impl(enriched_rows, implementation_order)
    return summary, rows_by_impl


def _comparison_rows(summary_path: Path) -> list[dict[str, object]]:
    summary = _read_json(summary_path)
    rows = [dict(row) for row in summary.get("rows", []) if isinstance(row, dict)]
    rows.sort(key=lambda row: (int(row.get("ranks", 10**6)), str(row.get("implementation", ""))))
    return rows


def _find_rows(rows: list[dict[str, object]], implementation: str) -> list[dict[str, object]]:
    selected = [dict(row) for row in rows if str(row.get("implementation", "")) == implementation]
    selected.sort(key=lambda row: int(row.get("ranks", 10**6)))
    return selected


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _boundary_faces_with_owner(tetrahedra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from collections import Counter

    face_rows: list[np.ndarray] = []
    owners: list[int] = []
    keys: list[tuple[int, int, int]] = []
    for elem_id, tet in enumerate(np.asarray(tetrahedra, dtype=np.int32)):
        local_faces = (
            tet[[0, 1, 2]],
            tet[[0, 1, 3]],
            tet[[0, 2, 3]],
            tet[[1, 2, 3]],
        )
        for face in local_faces:
            face_rows.append(np.asarray(face, dtype=np.int32))
            owners.append(elem_id)
            keys.append(tuple(sorted(int(v) for v in face.tolist())))
    counts = Counter(keys)
    keep = [idx for idx, key in enumerate(keys) if counts[key] == 1]
    return np.asarray([face_rows[idx] for idx in keep], dtype=np.int32), np.asarray([owners[idx] for idx in keep], dtype=np.int32)


def _tet_energy_density(coords_ref: np.ndarray, coords_final: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    c1 = 38461538.461538464
    d1 = 83333333.33333333
    x_ref = coords_ref[np.asarray(tetrahedra, dtype=np.int32)]
    x_def = coords_final[np.asarray(tetrahedra, dtype=np.int32)]
    dx_ref = np.transpose(x_ref[:, 1:, :] - x_ref[:, :1, :], (0, 2, 1))
    dx_def = np.transpose(x_def[:, 1:, :] - x_def[:, :1, :], (0, 2, 1))
    f = dx_def @ np.linalg.inv(dx_ref)
    i1 = np.sum(f * f, axis=(1, 2))
    detf = np.abs(np.linalg.det(f))
    detf = np.maximum(detf, 1.0e-12)
    return c1 * (i1 - 3.0 - 2.0 * np.log(detf)) + d1 * (detf - 1.0) ** 2


def _beam_corner_curves(coords_ref: np.ndarray, coords_final: np.ndarray) -> list[np.ndarray]:
    mins = coords_ref.min(axis=0)
    maxs = coords_ref.max(axis=0)
    curves: list[np.ndarray] = []
    for y in (mins[1], maxs[1]):
        for z in (mins[2], maxs[2]):
            mask = np.isclose(coords_ref[:, 1], y) & np.isclose(coords_ref[:, 2], z)
            curve_ref = coords_ref[mask]
            curve_final = coords_final[mask]
            order = np.argsort(curve_ref[:, 0])
            curves.append(curve_final[order])
    return curves


def _beam_prism_edges(coords_ref: np.ndarray, coords_final: np.ndarray) -> list[np.ndarray]:
    corner_curves = _beam_corner_curves(coords_ref, coords_final)
    edges: list[np.ndarray] = list(corner_curves)
    for end in (0, -1):
        corners = [curve[end] for curve in corner_curves]
        ordered = [corners[idx] for idx in (0, 1, 3, 2)]
        for idx in range(4):
            edges.append(np.vstack([ordered[idx], ordered[(idx + 1) % 4]]))
    return edges


def _box_edges(mins: np.ndarray, maxs: np.ndarray) -> list[np.ndarray]:
    x0, y0, z0 = np.asarray(mins, dtype=np.float64)
    x1, y1, z1 = np.asarray(maxs, dtype=np.float64)
    corners = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ],
        dtype=np.float64,
    )
    return [
        corners[[0, 1]],
        corners[[0, 2]],
        corners[[1, 3]],
        corners[[2, 3]],
        corners[[4, 5]],
        corners[[4, 6]],
        corners[[5, 7]],
        corners[[6, 7]],
        corners[[0, 4]],
        corners[[1, 5]],
        corners[[2, 6]],
        corners[[3, 7]],
    ]


def _set_equal_3d_axes_tight(ax, xyz: np.ndarray, *, radius_scale: float = 0.43) -> None:
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    radius = radius_scale * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def _set_3d_axes_from_bounds(ax, xyz: np.ndarray, *, pad_fraction: float = 0.04) -> None:
    mins = np.min(xyz, axis=0).astype(np.float64)
    maxs = np.max(xyz, axis=0).astype(np.float64)
    spans = np.maximum(maxs - mins, 1.0e-12)
    pads = pad_fraction * spans
    mins -= pads
    maxs += pads

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    try:
        ax.set_box_aspect(tuple(spans.tolist()), zoom=0.88)
    except Exception:
        pass


def _apply_hyperelasticity_camera(ax, bounds_min: np.ndarray, bounds_max: np.ndarray) -> None:
    from mpl_toolkits.mplot3d import proj3d

    spans = np.maximum(np.asarray(bounds_max, dtype=np.float64) - np.asarray(bounds_min, dtype=np.float64), 1.0e-12)
    ax.set_xlim(float(bounds_min[0]), float(bounds_max[0]))
    ax.set_ylim(float(bounds_min[1]), float(bounds_max[1]))
    ax.set_zlim(float(bounds_min[2]), float(bounds_max[2]))
    try:
        ax.set_box_aspect(tuple(spans.tolist()), zoom=1.25)
    except Exception:
        pass
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    elev = 12.0
    azim = -72.0
    roll = 0.0
    center = 0.5 * (np.asarray(bounds_min, dtype=np.float64) + np.asarray(bounds_max, dtype=np.float64))
    p0 = center
    p1 = center + np.array([1.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(3):
        ax.view_init(elev=elev, azim=azim, roll=roll)
        x0, y0, _ = proj3d.proj_transform(*p0, ax.get_proj())
        x1, y1, _ = proj3d.proj_transform(*p1, ax.get_proj())
        angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        roll -= angle
    ax.view_init(elev=elev, azim=azim, roll=roll)


def _draw_hyperelasticity_box_annotations(ax, bounds_min: np.ndarray, bounds_max: np.ndarray) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)
    spans = np.maximum(bounds_max - bounds_min, 1.0e-12)
    x0, y0, z0 = bounds_min
    x1, y1, z1 = bounds_max
    dx, dy, dz = spans

    box = Line3DCollection(_box_edges(bounds_min, bounds_max), colors="black", linewidths=0.85)
    ax.add_collection3d(box)

    tick_color = "black"
    tick_lw = 0.7
    fs = 7.0
    x_tick_len = 0.10 * dy
    yz_tick_len = 0.035 * dx

    for x in np.linspace(x0, x1, 5):
        ax.plot([x, x], [y0, y0 - x_tick_len], [z0, z0], color=tick_color, linewidth=tick_lw)
        ax.text(x, y0 - 2.2 * x_tick_len, z0 - 0.12 * dz, f"{x:.1f}", ha="center", va="top", fontsize=fs)

    for y in np.linspace(y0, y1, 3):
        ax.plot([x1, x1 + yz_tick_len], [y, y], [z0, z0], color=tick_color, linewidth=tick_lw)
        ax.text(x1 + 2.0 * yz_tick_len, y, z0 - 0.10 * dz, f"{y:.2f}", ha="left", va="center", fontsize=fs)

    for z in np.linspace(z0, z1, 3):
        ax.plot([x0 - yz_tick_len, x0], [y0, y0], [z, z], color=tick_color, linewidth=tick_lw)
        ax.text(x0 - 2.4 * yz_tick_len, y0 - 0.12 * dy, z, f"{z:.2f}", ha="right", va="center", fontsize=fs)


def _apply_plasticity3d_camera(ax, xyz: np.ndarray) -> None:
    from mpl_toolkits.mplot3d import proj3d

    rel = np.asarray(P3D_CAMERA_POSITION - P3D_CAMERA_TARGET, dtype=np.float64)
    # Rotate the viewing position by +90 degrees about x so the rendered body
    # is turned anti-clockwise around x before the final roll alignment.
    rot_x_90 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    rel = rot_x_90 @ rel
    azim = float(np.degrees(np.arctan2(rel[1], rel[0])))
    elev = float(np.degrees(np.arctan2(rel[2], np.hypot(rel[0], rel[1]))))
    _set_3d_axes_from_bounds(ax, xyz, pad_fraction=0.04)

    # Align the projected x-axis with the horizontal frame direction by
    # correcting the camera roll after the target/position-based view is set.
    roll = 0.0
    p0 = np.asarray(P3D_CAMERA_TARGET, dtype=np.float64)
    p1 = p0 + np.array([1.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(3):
        ax.view_init(elev=elev, azim=azim, roll=roll)
        x0, y0, _ = proj3d.proj_transform(*p0, ax.get_proj())
        x1, y1, _ = proj3d.proj_transform(*p1, ax.get_proj())
        angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        roll -= angle
    ax.view_init(elev=elev, azim=azim, roll=roll)

    x_min, x_max = ax.get_xlim()
    x_tick_start = 50.0 * np.floor(x_min / 50.0)
    x_tick_stop = 50.0 * np.ceil(x_max / 50.0)
    ax.set_xticks(np.arange(x_tick_start, x_tick_stop + 1.0, 50.0))
    locator = matplotlib.ticker.MaxNLocator(4)
    ax.yaxis.set_major_locator(locator)
    ax.zaxis.set_major_locator(locator)
    ax.tick_params(pad=1, labelsize=8.0)


def _deviatoric_strain_norm_3d(strain6: np.ndarray) -> np.ndarray:
    strain6 = np.asarray(strain6, dtype=np.float64)
    volumetric = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    metric = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(volumetric, volumetric) / 3.0
    projected = strain6 @ metric.T
    return np.sqrt(np.maximum(0.0, np.sum(strain6 * projected, axis=1)))


def _plasticity3d_dev_strain_data(
    *,
    coords_final: np.ndarray,
    displacement: np.ndarray,
    case,
    degree: int,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = int(degree)
    xi = plasticity3d_quadrature_points_tetra(degree)
    hatp = np.asarray(evaluate_tetra_lagrange_basis(degree, xi)[0], dtype=np.float64)
    basis_weight = np.abs(hatp.T)[None, :, :]

    elems = np.asarray(case.elems_scalar, dtype=np.int64)
    n_nodes = int(coords_final.shape[0])
    accum = np.zeros(n_nodes, dtype=np.float64)
    mass = np.zeros(n_nodes, dtype=np.float64)
    qcoords_blocks: list[np.ndarray] = []
    dev_blocks: list[np.ndarray] = []

    for start in range(0, elems.shape[0], chunk_size):
        stop = min(start + chunk_size, elems.shape[0])
        elem_nodes = elems[start:stop]
        u_elem = np.asarray(displacement[elem_nodes], dtype=np.float64)

        ux = u_elem[:, :, 0]
        uy = u_elem[:, :, 1]
        uz = u_elem[:, :, 2]
        dphix = np.asarray(case.dphix[start:stop], dtype=np.float64)
        dphiy = np.asarray(case.dphiy[start:stop], dtype=np.float64)
        dphiz = np.asarray(case.dphiz[start:stop], dtype=np.float64)
        e_xx = np.einsum("eqp,ep->eq", dphix, ux)
        e_yy = np.einsum("eqp,ep->eq", dphiy, uy)
        e_zz = np.einsum("eqp,ep->eq", dphiz, uz)
        g_xy = np.einsum("eqp,ep->eq", dphiy, ux) + np.einsum("eqp,ep->eq", dphix, uy)
        g_yz = np.einsum("eqp,ep->eq", dphiz, uy) + np.einsum("eqp,ep->eq", dphiy, uz)
        g_xz = np.einsum("eqp,ep->eq", dphiz, ux) + np.einsum("eqp,ep->eq", dphix, uz)
        eps6 = np.stack((e_xx, e_yy, e_zz, g_xy, g_yz, g_xz), axis=-1).reshape((-1, 6))
        dev = _deviatoric_strain_norm_3d(eps6)
        dev_elem = dev.reshape((stop - start, -1))
        qcoords = np.einsum("pq,epd->eqd", hatp, np.asarray(coords_final[elem_nodes], dtype=np.float64)).reshape((-1, 3))

        quad_weight = np.asarray(case.quad_weight[start:stop], dtype=np.float64)[:, :, None]
        local_weight = basis_weight * quad_weight
        local_value = np.einsum("eqp,eq->ep", local_weight, dev_elem)
        local_mass = np.sum(local_weight, axis=1)
        np.add.at(accum, elem_nodes.reshape(-1), local_value.reshape(-1))
        np.add.at(mass, elem_nodes.reshape(-1), local_mass.reshape(-1))

        qcoords_blocks.append(qcoords)
        dev_blocks.append(dev)

    nodal = np.zeros_like(accum)
    mask = mass > 0.0
    nodal[mask] = accum[mask] / mass[mask]
    return nodal, np.vstack(qcoords_blocks), np.concatenate(dev_blocks)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _set_rank_axis(ax, ranks: np.ndarray) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Wall time [s]")


def generate_scalar_state_figure(
    layout: dict[str, float],
    *,
    npz_path: Path,
    out_name: str,
    cbar_label: str,
    cmap: str = "viridis",
) -> str:
    plt = configure_paper_matplotlib()
    from matplotlib.tri import Triangulation

    data = _load_npz(npz_path)
    coords = np.asarray(data["coords"], dtype=np.float64)
    triangles = np.asarray(data["triangles"], dtype=np.int32)
    values = np.asarray(data["u"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="narrow", height_ratio=0.62))
    tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
    artist = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap)
    artist.set_rasterized(True)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    cbar = fig.colorbar(artist, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.14, top=0.96)

    out = FIGURES_ROOT / out_name
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def generate_energy_levels_figure(
    layout: dict[str, float],
    *,
    csv_path: Path,
    implementations: tuple[str, ...],
    out_name: str,
    ylabel: str,
) -> str:
    plt = configure_paper_matplotlib()
    rows = read_csv_rows(csv_path)
    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="narrow", height_ratio=0.44))
    for implementation in implementations:
        style = paper_implementation_style(implementation)
        x = [int(row["level"]) for row in rows if row.get(implementation)]
        y = [float(row[implementation]) for row in rows if row.get(implementation)]
        if not x:
            continue
        ax.plot(
            x,
            y,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=style["label"],
        )
    ax.set_xlabel("Mesh level")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc="best")
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.96)
    out = FIGURES_ROOT / out_name
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def generate_family_scaling_figure(
    layout: dict[str, float],
    *,
    csv_path: Path,
    implementations: tuple[str, ...],
    out_name: str,
) -> str:
    plt = configure_paper_matplotlib()
    rows = [row for row in read_csv_rows(csv_path) if row.get("result", "completed") == "completed"]
    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="medium", height_ratio=0.52))
    ideal_drawn = False
    for implementation in implementations:
        impl_rows = [row for row in rows if row.get("solver") == implementation]
        if not impl_rows:
            continue
        impl_rows.sort(key=lambda item: int(item["nprocs"] if "nprocs" in item else item["ranks"]))
        style = paper_implementation_style(implementation)
        ranks = np.asarray([int(row["nprocs"] if "nprocs" in row else row["ranks"]) for row in impl_rows], dtype=np.int64)
        times = np.asarray([float(row["total_time_s"] if "total_time_s" in row else row["wall_time_s"]) for row in impl_rows], dtype=np.float64)
        ax.plot(
            ranks,
            times,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=style["label"],
        )
        if not ideal_drawn:
            ax.plot(ranks, ideal_strong_scaling(ranks, times), color="#000000", linestyle="--", linewidth=1.1, label=r"ideal $1/r$")
            ideal_drawn = True
    _set_rank_axis(ax, np.asarray(sorted({int(row["nprocs"] if "nprocs" in row else row["ranks"]) for row in rows}), dtype=np.int64))
    ax.legend(frameon=True, loc="best")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.96)

    out = FIGURES_ROOT / out_name
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def generate_hyperelasticity_cpu_pmg_scaling(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    rows = [
        row
        for row in read_csv_rows(HYPER_CPU_PMG_SCALING)
        if row.get("result", "completed") == "completed"
    ]
    rows.sort(key=lambda row: int(row["ranks"]))
    ranks = np.asarray([int(row["ranks"]) for row in rows], dtype=np.int64)
    times = np.asarray([float(row["solver_total_s"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="medium", height_ratio=0.52))
    ax.plot(
        ranks,
        times,
        marker="o",
        color="#1f77b4",
        linestyle="-",
        linewidth=1.8,
        label="PMG coarse level $L_2$",
    )
    ax.plot(
        ranks,
        ideal_strong_scaling(ranks, times),
        color="#000000",
        linestyle="--",
        linewidth=1.1,
        label=r"ideal $1/r$",
    )
    _set_rank_axis(ax, ranks)
    ax.set_ylabel("Solver total time [s]")
    ax.margins(x=0.05, y=0.12)
    for row in rows:
        nodes = int(row["nodes"])
        if nodes <= 1:
            continue
        ax.annotate(
            f"{nodes}n",
            (int(row["ranks"]), float(row["solver_total_s"])),
            xytext=(4.0, 5.0),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.legend(frameon=True, loc="best")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.96)

    out = FIGURES_ROOT / "hyperelasticity_cpu_pmg_scaling.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def generate_hyperelasticity_state(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    data = _load_npz(HYPER_STATE)
    coords_ref = np.asarray(data["coords_ref"], dtype=np.float64)
    coords_final = np.asarray(data["coords_final"], dtype=np.float64)
    tetrahedra = np.asarray(data["tetrahedra"], dtype=np.int32)
    boundary_faces, boundary_owners = _boundary_faces_with_owner(tetrahedra)
    face_values = _tet_energy_density(coords_ref, coords_final, tetrahedra)[boundary_owners]
    ref_mins = coords_ref.min(axis=0)
    ref_maxs = coords_ref.max(axis=0)
    tri_xyz = coords_final[np.asarray(boundary_faces, dtype=np.int64)]
    norm = Normalize(vmin=float(np.min(face_values)), vmax=float(np.max(face_values)))
    cmap = matplotlib.colormaps["viridis"]

    # Approved 2026-04-16 paper layout for Figure 10:
    # pure Matplotlib 3D, vector PDF except rasterized surface, cropped tightly.
    # The figure is intentionally generated on a large canvas so the final
    # cropped asset remains clean after inclusion in the manuscript.
    fig_rc = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with matplotlib.rc_context(fig_rc):
        # The tightly cropped asset is included at \PaperFigureMedium. The
        # historical large canvas cropped to about 549 pt, which LaTeX then
        # downscaled. Scale the canvas so the cropped PDF is produced near its
        # final physical include width and figure fonts retain their point size.
        final_width_in = 0.84 * layout["textwidth_in"]
        canvas_scale = final_width_in / (549.0 / 72.27)
        fig = plt.figure(figsize=(16.9 * canvas_scale, 11.999 * canvas_scale))
        ax = fig.add_axes([0.06, 0.34, 0.88, 0.56], projection="3d")

        poly = Poly3DCollection(tri_xyz, linewidths=0.0, antialiased=False)
        poly.set_facecolor(cmap(norm(face_values)))
        poly.set_edgecolor("none")
        poly.set_alpha(1.0)
        poly.set_clip_on(False)
        poly.set_rasterized(True)
        ax.add_collection3d(poly)

        eye = np.asarray([0.2, 0.20, -0.05], dtype=np.float64)
        target = np.asarray([0.2, 0.00, 0.0], dtype=np.float64)
        direction = target - eye
        azim = float(np.degrees(np.arctan2(direction[1], direction[0])))
        elev = float(np.degrees(np.arctan2(direction[2], np.hypot(direction[0], direction[1]))))

        ax.set_xlim(0.0, 0.4)
        ax.set_ylim(-0.02, 0.02)
        ax.set_zlim(-0.02, 0.02)
        ax.set_box_aspect((0.4, 0.04, 0.04), zoom=1.03)
        ax.set_proj_type("persp", focal_length=1.2)
        ax.view_init(elev=elev, azim=azim, roll=0.0)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
            axis.line.set_color((0.0, 0.0, 0.0, 1.0))
            axis.line.set_linewidth(1.0)
            axis._axinfo["grid"]["color"] = (0.0, 0.0, 0.0, 1.0)
            axis._axinfo["grid"]["linewidth"] = 0.8
            axis._axinfo["grid"]["linestyle"] = "-"
            axis._axinfo["axisline"]["color"] = (0.0, 0.0, 0.0, 1.0)

        ax.xaxis._axinfo["tick"]["inward_factor"] = 0.0
        ax.xaxis._axinfo["tick"]["outward_factor"] = 0.45
        ax.zaxis._axinfo["tick"]["inward_factor"] = 0.0
        ax.zaxis._axinfo["tick"]["outward_factor"] = 0.25
        ax.xaxis._axinfo["tick"]["linewidth"] = {True: 1.0, False: 0.8}
        ax.zaxis._axinfo["tick"]["linewidth"] = {True: 1.0, False: 0.8}

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks(np.linspace(0.0, 0.4, 5))
        ax.set_yticks([])
        ax.set_zticks([-0.02, 0.02])
        ax.tick_params(labelsize=12, length=6, width=1.0, pad=1.8, colors="black")
        ax.grid(True)

        sm = cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array(face_values)
        cax = fig.add_axes([0.3775, 0.54, 0.285, 0.012845])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label(r"energy density $W(F)$")
        cbar.ax.tick_params(labelsize=12, width=1.0, length=6, colors="black")

        out = FIGURES_ROOT / "hyperelasticity_state.pdf"
        raw_out = out.with_name("hyperelasticity_state_raw.pdf")
        fig.savefig(raw_out, format="pdf", dpi=660, metadata=PDF_METADATA)
        plt.close(fig)

    crop_env = dict(os.environ)
    crop_env.setdefault("SOURCE_DATE_EPOCH", "946684800")
    subprocess.run(["pdfcrop", "--margins", "0", str(raw_out), str(out)], check=True, env=crop_env)
    subprocess.run(["qpdf", "--deterministic-id", str(out), "--replace-input"], check=True)
    _normalize_pdf_trailer_id(out)
    subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile", str(out), str(out.with_suffix(""))], check=True)
    if raw_out.exists():
        raw_out.unlink()
    return out.name


def generate_plasticity2d_figures(layout: dict[str, float]) -> list[str]:
    plt = configure_paper_matplotlib()
    import matplotlib.tri as mtri

    state = _load_npz(PLASTICITY2D_STATE)
    case = build_same_mesh_lagrange_case_data("ssr_homo_capture_p2_level5", degree=4)
    coords_ref = np.asarray(state["coords_ref"], dtype=np.float64)
    coords_final = np.asarray(state["coords_final"], dtype=np.float64)
    displacement = np.asarray(state["displacement"], dtype=np.float64)
    elems_scalar = np.asarray(case.elems_scalar, dtype=np.int32)
    coords_plot, tri_plot, dispmag, dev, macro_tri = plasticity2d_build_refined_plot_data(
        coords_ref=coords_ref,
        coords_final=coords_final,
        displacement=displacement,
        elems_scalar=elems_scalar,
        degree=4,
        subdivisions=6,
    )
    triangulation = mtri.Triangulation(coords_plot[:, 0], coords_plot[:, 1], triangles=tri_plot)
    macro = mtri.Triangulation(coords_final[:, 0], coords_final[:, 1], triangles=macro_tri)
    outputs: list[str] = []

    fig = plt.figure(figsize=paper_figure_size(layout, preset="medium", height_ratio=0.42))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.07],
        left=0.08,
        right=0.98,
        bottom=0.18,
        top=0.97,
        wspace=0.22,
        hspace=0.16,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    caxes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    panel_specs = (
        (dispmag, "viridis", r"$\|u\|$", None),
        (dev, "magma", r"$\|\varepsilon_{\mathrm{dev}}\|$", float(np.quantile(np.asarray(dev, dtype=np.float64), 0.995))),
    )
    for idx, (ax, cax, (values, cmap_name, cbar_label, vmax)) in enumerate(zip(axes, caxes, panel_specs, strict=True)):
        plot_kwargs = {"shading": "gouraud", "cmap": cmap_name}
        if vmax is not None:
            plot_kwargs.update({"vmin": 0.0, "vmax": max(vmax, 1.0e-12)})
        pc = ax.tripcolor(triangulation, values, **plot_kwargs)
        pc.set_rasterized(True)
        mesh_lines = ax.triplot(macro, color="black", linewidth=0.12, alpha=0.10)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$" if idx == 0 else "")
        cbar = fig.colorbar(pc, cax=cax, orientation="horizontal")
        cbar.set_label(cbar_label, labelpad=1.0)
    out = FIGURES_ROOT / "plasticity2d_state_pair.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    outputs.append(out.name)

    l5_result = _read_json(PLASTICITY2D_RESULT)["result"]["steps"][0]
    l6_rows = _read_json(PLASTICITY2D_L6_SUMMARY)
    l7_rows = _read_json(PLASTICITY2D_L7_SUMMARY)
    l6 = next(row for row in l6_rows if int(row["ranks"]) == 8)
    l7 = next(row for row in l7_rows if int(row["ranks"]) == 16)
    dofs = np.asarray(
        [
            int(_read_json(PLASTICITY2D_RESULT)["mesh"]["free_dofs"]),
            int(l6["free_dofs"]),
            int(l7["free_dofs"]),
        ],
        dtype=np.float64,
    )
    energy = np.asarray(
        [
            float(l5_result["energy"]),
            float(l6["energy"]),
            float(l7["energy"]),
        ],
        dtype=np.float64,
    )
    labels = [
        _element_math_label("P4", "L5"),
        _element_math_label("P4", "L6"),
        _element_math_label("P4", "L7"),
    ]
    energy_offset = -212.538
    energy_scaled = (energy - energy_offset) / 1.0e-4
    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="narrow", height_ratio=0.50))
    ax.plot(dofs, energy_scaled, color="#7f3c8d", marker="o", linewidth=2.0)
    for x, y, label in zip(dofs, energy_scaled, labels, strict=True):
        annotate_kwargs = {
            "xy": (x, y),
            "textcoords": "offset points",
            "fontsize": 7.5,
            "clip_on": False,
        }
        if label.endswith("(L_5)$"):
            ax.annotate(label, xytext=(6, 0), ha="left", va="center", **annotate_kwargs)
        elif label.endswith("(L_7)$"):
            ax.annotate(label, xytext=(0, 14), ha="center", va="bottom", **annotate_kwargs)
        else:
            ax.annotate(label, xytext=(4, 6), ha="left", va="bottom", **annotate_kwargs)
    ax.set_xscale("log")
    ax.set_xlabel("Free DOFs")
    ax.set_ylabel(r"Shifted endpoint energy [$10^{-4}$]")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.margins(x=0.08, y=0.08)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(
        0.0,
        1.02,
        r"Energy = $-212.538 + 10^{-4}\times$ ordinate",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        clip_on=False,
    )
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.18, top=0.92)
    out = FIGURES_ROOT / "plasticity2d_resolution_energy.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    outputs.append(out.name)
    return outputs


def _plot_plasticity3d_surface(
    layout: dict[str, float],
    *,
    nodal_values: np.ndarray,
    out_name: str,
    cbar_label: str,
    cmap_name: str,
    view_azim: float,
    norm_override=None,
) -> str:
    plt = configure_paper_matplotlib()
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ctx = _plasticity3d_benchmark_context()
    state = ctx["state"]
    degree = int(ctx["degree"])
    coords_final = np.asarray(state["coords_final"], dtype=np.float64)
    surface_faces = np.asarray(state["surface_faces"], dtype=np.int64)
    coords_plot, tri_plot, values = plasticity3d_surface_plot_arrays(
        coords_final,
        surface_faces,
        np.asarray(nodal_values, dtype=np.float64),
        degree=degree,
        subdivisions=P3D_BENCHMARK_SURFACE_SUBDIVISIONS,
    )
    tri_xyz = coords_plot[np.asarray(tri_plot, dtype=np.int64)]
    tri_vals = np.mean(values[np.asarray(tri_plot, dtype=np.int64)], axis=1)
    if norm_override is None:
        norm = Normalize(vmin=float(np.min(tri_vals)), vmax=float(np.max(tri_vals)))
    else:
        norm = norm_override
    cmap = matplotlib.colormaps[cmap_name]

    fig = plt.figure(figsize=paper_figure_size(layout, preset="medium", height_ratio=0.48))
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(tri_xyz, linewidths=0.0, antialiased=False)
    poly.set_facecolor(cmap(norm(tri_vals)))
    poly.set_edgecolor("none")
    poly.set_rasterized(True)
    ax.add_collection3d(poly)
    _apply_plasticity3d_camera(ax, coords_plot)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(tri_vals)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label(cbar_label)
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.05, top=0.97)

    out = FIGURES_ROOT / out_name
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def _plasticity3d_surface_tri_data(nodal_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ctx = _plasticity3d_benchmark_context()
    state = ctx["state"]
    degree = int(ctx["degree"])
    coords_final = np.asarray(state["coords_final"], dtype=np.float64)
    surface_faces = np.asarray(state["surface_faces"], dtype=np.int64)
    coords_plot, tri_plot, values = plasticity3d_surface_plot_arrays(
        coords_final,
        surface_faces,
        np.asarray(nodal_values, dtype=np.float64),
        degree=degree,
        subdivisions=P3D_BENCHMARK_SURFACE_SUBDIVISIONS,
    )
    tri_xyz = coords_plot[np.asarray(tri_plot, dtype=np.int64)]
    tri_vals = np.mean(values[np.asarray(tri_plot, dtype=np.int64)], axis=1)
    return tri_xyz, tri_vals


def _plot_plasticity3d_surface_panel(
    ax,
    *,
    nodal_values: np.ndarray,
    cmap_name: str,
    norm,
    view_azim: float,
) -> np.ndarray:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tri_xyz, tri_vals = _plasticity3d_surface_tri_data(nodal_values)
    cmap = matplotlib.colormaps[cmap_name]
    poly = Poly3DCollection(tri_xyz, linewidths=0.0, antialiased=False)
    poly.set_facecolor(cmap(norm(tri_vals)))
    poly.set_edgecolor("none")
    poly.set_rasterized(True)
    ax.add_collection3d(poly)
    _apply_plasticity3d_camera(ax, tri_xyz.reshape(-1, 3))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    return tri_vals


def _plot_plasticity3d_tri_panel(
    ax,
    *,
    tri_xyz: np.ndarray,
    tri_vals: np.ndarray,
    cmap_name: str,
    norm,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cmap = matplotlib.colormaps[cmap_name]
    poly = Poly3DCollection(np.asarray(tri_xyz, dtype=np.float64), linewidths=0.0, antialiased=False)
    poly.set_facecolor(cmap(norm(np.asarray(tri_vals, dtype=np.float64))))
    poly.set_edgecolor("none")
    poly.set_rasterized(True)
    ax.add_collection3d(poly)
    _apply_plasticity3d_camera(ax, np.asarray(tri_xyz, dtype=np.float64).reshape(-1, 3))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


def generate_plasticity3d_state_figures(layout: dict[str, float]) -> list[str]:
    plt = configure_paper_matplotlib()
    from matplotlib import cm
    from matplotlib.colors import Normalize

    metadata = _read_json(P3D_STATE_PAIR_SURFACE_METADATA)
    with np.load(P3D_STATE_PAIR_SURFACE_ARRAYS) as data:
        tri_xyz = np.asarray(data["tri_xyz"], dtype=np.float64)
        disp_tri_vals = np.asarray(data["disp_tri_vals"], dtype=np.float64)
        dev_tri_vals = np.asarray(data["dev_tri_vals"], dtype=np.float64)
    disp_norm = Normalize(vmin=0.0, vmax=float(metadata["disp_norm_vmax"]))
    dev_norm = Normalize(vmin=0.0, vmax=float(metadata["dev_norm_vmax"]))

    fig = plt.figure(figsize=paper_figure_size(layout, preset="medium", height_ratio=0.60))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.08], left=0.03, right=0.96, bottom=0.13, top=0.96, wspace=0.08, hspace=0.04)
    ax_disp = fig.add_subplot(gs[0, 0], projection="3d")
    ax_dev = fig.add_subplot(gs[0, 1], projection="3d")
    _plot_plasticity3d_tri_panel(
        ax_disp,
        tri_xyz=tri_xyz,
        tri_vals=disp_tri_vals,
        cmap_name="viridis",
        norm=disp_norm,
    )
    _plot_plasticity3d_tri_panel(
        ax_dev,
        tri_xyz=tri_xyz,
        tri_vals=dev_tri_vals,
        cmap_name="magma",
        norm=dev_norm,
    )
    disp_sm = cm.ScalarMappable(norm=disp_norm, cmap="viridis")
    disp_sm.set_array(np.asarray(disp_tri_vals, dtype=np.float64))
    dev_sm = cm.ScalarMappable(norm=dev_norm, cmap="magma")
    dev_sm.set_array(np.asarray(dev_tri_vals, dtype=np.float64))
    cax1 = fig.add_subplot(gs[1, 0])
    cax2 = fig.add_subplot(gs[1, 1])
    cbar1 = fig.colorbar(disp_sm, cax=cax1, orientation="horizontal")
    cbar1.set_label(r"$\|u\|$", labelpad=0.3)
    cbar1.ax.tick_params(pad=1.5)
    cbar2 = fig.colorbar(dev_sm, cax=cax2, orientation="horizontal")
    cbar2.set_label(r"$\|\varepsilon_{\mathrm{dev}}\|$", labelpad=0.3)
    cbar2.ax.tick_params(pad=1.5)
    out = FIGURES_ROOT / "plasticity3d_state_pair.pdf"
    save_pdf_and_png(fig, out, png_dpi=260)
    plt.close(fig)
    conv_out = _generate_plasticity3d_convergence_figure(layout)
    return [out.name, conv_out]


def _generate_plasticity3d_convergence_figure(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    study_rows = _plasticity3d_study_rows()
    style_by_degree = {
        "P1": {"color": "#1f77b4"},
        "P2": {"color": "#ff7f0e"},
        "P4": {"color": "#2ca02c"},
    }
    linestyle_by_mesh = {
        "L1": "-",
        "L1_2": "--",
        "L1_2_3": "-.",
        "L1_2_3_4": ":",
    }
    fig, axes = plt.subplots(2, 1, figsize=paper_figure_size(layout, preset="medium", height_ratio=0.68), sharex=True)
    legend_handles: list[object] = []
    legend_labels: list[str] = []
    for row in study_rows:
        history = list(row.get("history", []))
        if not history:
            continue
        its = np.asarray([int(item.get("it", idx + 1)) for idx, item in enumerate(history)], dtype=np.int64)
        energy = np.asarray([_safe_float(item.get("energy")) / 1.0e6 for item in history], dtype=np.float64)
        grad = []
        for item in history:
            grad.append(_safe_float(item.get("grad_norm")))
        grad_arr = np.maximum(np.asarray(grad, dtype=np.float64), 1.0e-16)
        degree_line = str(row.get("degree_line", ""))
        mesh_alias = str(row.get("mesh_alias", ""))
        style = style_by_degree.get(degree_line, {"color": "#333333"})
        handle = axes[0].plot(
            its,
            energy,
            color=style["color"],
            linestyle=linestyle_by_mesh.get(mesh_alias, "-"),
            linewidth=1.7,
            alpha=0.95,
        )[0]
        axes[1].semilogy(
            its,
            grad_arr,
            color=style["color"],
            linestyle=linestyle_by_mesh.get(mesh_alias, "-"),
            linewidth=1.7,
            alpha=0.95,
        )
        legend_handles.append(handle)
        legend_labels.append(_element_math_label(degree_line, mesh_alias))

    axes[0].set_ylabel(r"Energy [$10^{6}$]")
    axes[0].grid(True, alpha=0.25)
    axes[1].axhline(1.0e-2, color="#555555", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Newton iteration")
    axes[1].set_ylabel(r"$\|g\|$")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    axes[0].set_title("All nine bottom-clamped endpoint configurations", pad=4)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        handlelength=2.8,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.30, top=0.92, hspace=0.10)
    conv_out = FIGURES_ROOT / "plasticity3d_convergence.pdf"
    save_pdf_and_png(fig, conv_out)
    plt.close(fig)
    return conv_out.name


def _highest_rows_by_degree(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for degree_line in ("P1", "P2", "P4"):
        selected = [dict(row) for row in rows if str(row.get("degree_line", "")) == degree_line]
        selected.sort(key=lambda row: int(row.get("free_dofs", 0)))
        if not selected:
            raise RuntimeError(f"No highest-mesh row found for {degree_line}")
        out.append(selected[-1])
    return out


def _build_highest_y_slice(row: dict[str, object]) -> dict[str, object]:
    state_path = _repo_path(row["state_npz"])
    mesh_name = str(row["mesh_name"])
    degree = int(row["elem_degree"])
    constraint_variant = str(row.get("constraint_variant", "glued_bottom"))
    case_path = same_mesh_case_hdf5_path(mesh_name, degree, constraint_variant)

    state = np.load(state_path)
    coords_final = np.asarray(state["coords_final"], dtype=np.float64)
    displacement = np.asarray(state["displacement"], dtype=np.float64)
    bounds_min = np.min(coords_final, axis=0)
    bounds_max = np.max(coords_final, axis=0)
    spans = np.maximum(bounds_max - bounds_min, 1.0e-12)
    y_center = bounds_min[1] + 0.62 * spans[1]
    half_thickness = 0.02 * spans[1]

    xi = plasticity3d_quadrature_points_tetra(degree)
    hatp = np.asarray(evaluate_tetra_lagrange_basis(degree, xi)[0], dtype=np.float64)
    qcoords_blocks: list[np.ndarray] = []
    qdev_blocks: list[np.ndarray] = []
    chunk_size = 4096 if degree == 1 else (1024 if degree == 2 else 256)
    with h5py.File(case_path, "r") as handle:
        elems = np.asarray(handle["elems_scalar"], dtype=np.int64)
        dphix_ds = handle["dphix"]
        dphiy_ds = handle["dphiy"]
        dphiz_ds = handle["dphiz"]
        for start in range(0, elems.shape[0], chunk_size):
            stop = min(start + chunk_size, elems.shape[0])
            elem_nodes = elems[start:stop]
            x_def = np.asarray(coords_final[elem_nodes], dtype=np.float64)
            u_elem = np.asarray(displacement[elem_nodes], dtype=np.float64)

            ux = u_elem[:, :, 0]
            uy = u_elem[:, :, 1]
            uz = u_elem[:, :, 2]
            dphix = np.asarray(dphix_ds[start:stop], dtype=np.float64)
            dphiy = np.asarray(dphiy_ds[start:stop], dtype=np.float64)
            dphiz = np.asarray(dphiz_ds[start:stop], dtype=np.float64)
            e_xx = np.einsum("eqp,ep->eq", dphix, ux)
            e_yy = np.einsum("eqp,ep->eq", dphiy, uy)
            e_zz = np.einsum("eqp,ep->eq", dphiz, uz)
            g_xy = np.einsum("eqp,ep->eq", dphiy, ux) + np.einsum("eqp,ep->eq", dphix, uy)
            g_yz = np.einsum("eqp,ep->eq", dphiz, uy) + np.einsum("eqp,ep->eq", dphiy, uz)
            g_xz = np.einsum("eqp,ep->eq", dphiz, ux) + np.einsum("eqp,ep->eq", dphix, uz)
            eps6 = np.stack((e_xx, e_yy, e_zz, g_xy, g_yz, g_xz), axis=-1).reshape((-1, 6))
            qcoords_chunk = np.einsum("pq,epd->eqd", hatp, x_def)
            qdev_chunk = _deviatoric_strain_norm_3d(eps6).reshape(qcoords_chunk.shape[:2])
            mask = np.abs(qcoords_chunk[:, :, 1] - float(y_center)) <= float(half_thickness)
            if np.any(mask):
                qcoords_blocks.append(np.asarray(qcoords_chunk[mask], dtype=np.float64))
                qdev_blocks.append(np.asarray(qdev_chunk[mask], dtype=np.float64))

    qcoords = np.vstack(qcoords_blocks)
    qdev = np.concatenate(qdev_blocks)
    with h5py.File(case_path, "r") as handle:
        elems = np.asarray(handle["elems_scalar"], dtype=np.int64)
    slice_data = plasticity3d_interpolate_planar_slice(
        qcoords,
        qdev,
        axis=1,
        center=float(y_center),
        half_thickness=float(half_thickness),
        footprint_points=coords_final,
        footprint_tetrahedra=elems,
        resolution=900,
        smooth_sigma=1.0,
    )
    slice_data["degree_line"] = str(row["degree_line"])
    slice_data["free_dofs"] = int(row["free_dofs"])
    slice_data["energy"] = float(row["energy"])
    return slice_data


def generate_plasticity3d_highest_y_slice_comparison(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    metadata = _read_json(P3D_HIGHEST_Y_SLICE_METADATA)
    with np.load(P3D_HIGHEST_Y_SLICE_ARRAYS) as data:
        extents = np.asarray(data["extents"], dtype=np.float64)
        image_by_key = {key: np.asarray(data[key], dtype=np.float64) for key in data.files if key.startswith("image_")}
    panels = [dict(panel) for panel in metadata["panels"]]
    vmax = float(metadata["global_vmax"])
    zlim = tuple(float(value) for value in metadata["zlim"])

    fig = plt.figure(figsize=paper_figure_size(layout, preset="full", height_ratio=0.41))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.08],
        left=0.06,
        right=0.98,
        bottom=0.13,
        top=0.88,
        hspace=0.52,
        wspace=0.14,
    )
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    cax = fig.add_subplot(gs[1, :])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="white")
    mappable = None
    for idx, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
        image = image_by_key[str(panel["image_key"])]
        mappable = ax.imshow(
            image,
            origin="lower",
            extent=tuple(float(v) for v in extents[idx]),
            cmap=cmap,
            vmin=0.0,
            vmax=max(vmax, 1.0e-12),
            interpolation="bilinear",
            aspect="equal",
        )
        mappable.set_rasterized(True)
        ax.set_title(_degree_math_label(panel["degree_line"]), pad=6)
        ax.set_xlabel("x")
        if idx == 0:
            ax.set_ylabel("z")
        else:
            ax.set_ylabel("")
        ax.set_xlim(*(float(value) for value in metadata["xlim"]))
        ax.set_ylim(*zlim)
        ax.grid(False)
    if mappable is not None:
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cbar.set_label(r"$\|\varepsilon_{\mathrm{dev}}\|$", labelpad=0.4)
        cbar.set_ticks([0.0, 0.5 * max(vmax, 1.0e-12), max(vmax, 1.0e-12)])
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.03f"))

    out = FIGURES_ROOT / "plasticity3d_highest_mesh_y_slice_comparison.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def generate_topology_density(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    data = _load_npz(TOPOLOGY_STATE)
    theta = np.asarray(data["theta_grid"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="medium", height_ratio=0.46))
    artist = ax.imshow(
        theta.T,
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        extent=(0.0, 2.0, 0.0, 1.0),
        aspect="equal",
    )
    artist.set_rasterized(True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.16, top=0.96)
    cax = fig.add_axes([0.88, 0.22, 0.024, 0.60])
    cbar = fig.colorbar(artist, cax=cax, orientation="vertical")
    cbar.set_label(r"density $\theta_h$", labelpad=2.0)
    out = FIGURES_ROOT / "topology_density.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def generate_topology_history(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    rows = read_csv_rows(TOPOLOGY_HISTORY)
    outer_key = "outer_iteration" if rows and "outer_iteration" in rows[0] else "outer_iter"
    outer = np.asarray([int(row[outer_key]) for row in rows], dtype=np.int64)
    compliance = np.asarray([float(row["compliance"]) for row in rows], dtype=np.float64)
    volume = np.asarray([float(row["volume_fraction"]) for row in rows], dtype=np.float64)
    p_penal = np.asarray([float(row["p_penal"]) for row in rows], dtype=np.float64)
    p_ratio = p_penal / max(float(np.max(p_penal)), 1.0e-12)

    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="narrow", height_ratio=0.54))
    ax_right = ax.twinx()

    compliance_line = ax.plot(outer, compliance, color="#111111", linewidth=1.8, label="compliance")[0]
    volume_line = ax_right.plot(outer, volume, color="#555555", linewidth=1.5, linestyle="--", label="volume fraction")[0]
    penal_line = ax_right.plot(outer, p_ratio, color="#888888", linewidth=1.5, linestyle=":", label=r"$p / p_{\max}$")[0]

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Compliance")
    ax_right.set_ylabel(r"Volume fraction / $p$ ratio")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(float(np.min(outer)), float(np.max(outer)))
    ax_right.set_ylim(0.0, max(1.02, float(np.nanmax(volume)) * 1.08))
    ax.legend(
        [compliance_line, volume_line, penal_line],
        [line.get_label() for line in (compliance_line, volume_line, penal_line)],
        frameon=True,
        loc="lower right",
        ncol=2,
        columnspacing=1.0,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.12, right=0.87, bottom=0.18, top=0.96)
    out = FIGURES_ROOT / "topology_history.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def generate_topology_scaling(layout: dict[str, float]) -> str:
    return generate_family_scaling_figure(
        layout,
        csv_path=TOPOLOGY_SCALING,
        implementations=("jax_parallel",),
        out_name="topology_scaling.pdf",
    )


def _draw_box(
    ax,
    xy,
    wh,
    *,
    title: str,
    body: list[str],
    facecolor: str,
    title_size: float = 10.0,
    body_size: float = 9.0,
) -> None:
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor="#303030",
        linewidth=1.0,
    )
    ax.add_patch(patch)
    if title:
        ax.text(
            x + 0.022 * w,
            y + h - 0.12 * h,
            title,
            fontsize=title_size,
            fontweight="bold",
            va="top",
            linespacing=1.0,
        )
    if body:
        body_top = y + h - (0.31 * h if title else 0.14 * h)
        ax.text(
            x + 0.022 * w,
            body_top,
            "\n".join(body),
            fontsize=body_size,
            va="top",
            linespacing=1.0,
        )


def _draw_arrow(
    ax,
    start,
    end,
    text: str = "",
    *,
    shrink_a: float = 8.0,
    shrink_b: float = 8.0,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            lw=1.3,
            color="#3a3a3a",
            shrinkA=shrink_a,
            shrinkB=shrink_b,
        ),
    )
    if text:
        mx = 0.5 * (start[0] + end[0])
        my = 0.5 * (start[1] + end[1])
        ax.text(mx, my + 0.03, text, fontsize=9, ha="center", va="bottom")


def generate_framework_overview(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    fig, ax = plt.subplots(figsize=text_figure_size(layout, height_ratio=0.50))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _draw_box(
        ax,
        (0.04, 0.61),
        (0.25, 0.15),
        title="",
        body=[
            "Reference",
            "implementations",
            "pure JAX, FEniCS",
        ],
        facecolor="#f4efe6",
        title_size=8.7,
        body_size=7.2,
    )
    _draw_box(
        ax,
        (0.375, 0.61),
        (0.25, 0.15),
        title="",
        body=[
            "JAX+PETSc realization",
            "Newton--Krylov + PMG",
        ],
        facecolor="#e6f0fb",
        title_size=8.7,
        body_size=7.2,
    )
    _draw_box(
        ax,
        (0.71, 0.61),
        (0.25, 0.15),
        title="",
        body=[
            "Benchmark families",
            "elliptic, elasticity, topology",
        ],
        facecolor="#eef5e7",
        title_size=8.7,
        body_size=7.2,
    )
    _draw_box(
        ax,
        (0.14, 0.36),
        (0.27, 0.11),
        title="",
        body=[
            "Derivative modes",
            "element AD, constitutive, colored recovery",
        ],
        facecolor="#f7f4fb",
        title_size=8.5,
        body_size=6.9,
    )
    _draw_box(
        ax,
        (0.59, 0.36),
        (0.27, 0.11),
        title="",
        body=[
            "Solver modes",
            "Armijo, Krylov, PMG",
        ],
        facecolor="#fbf1eb",
        title_size=8.5,
        body_size=6.9,
    )
    _draw_arrow(ax, (0.29, 0.685), (0.375, 0.685), shrink_a=4.0, shrink_b=4.0)
    _draw_arrow(ax, (0.625, 0.685), (0.71, 0.685), shrink_a=4.0, shrink_b=4.0)
    _draw_arrow(ax, (0.50, 0.61), (0.285, 0.47), shrink_a=6.0, shrink_b=8.0)
    _draw_arrow(ax, (0.50, 0.61), (0.715, 0.47), shrink_a=6.0, shrink_b=8.0)
    out = FIGURES_ROOT / "framework_overview.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def _surface_compare_panel(
    ax,
    *,
    coords_final: np.ndarray,
    surface_faces: np.ndarray,
    nodal_values: np.ndarray,
    cmap_name: str,
    norm,
) -> np.ndarray:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    coords_plot, tri_plot, values = plasticity3d_surface_plot_arrays(
        np.asarray(coords_final, dtype=np.float64),
        np.asarray(surface_faces, dtype=np.int64),
        np.asarray(nodal_values, dtype=np.float64),
        degree=2,
        subdivisions=1,
    )
    tri_plot = np.asarray(tri_plot, dtype=np.int64)
    tri_vals = np.mean(np.asarray(values, dtype=np.float64)[tri_plot], axis=1)
    poly = Poly3DCollection(coords_plot[tri_plot], linewidths=0.0, antialiased=False)
    poly.set_facecolor(matplotlib.colormaps[cmap_name](norm(tri_vals)))
    poly.set_edgecolor("none")
    poly.set_rasterized(True)
    ax.add_collection3d(poly)
    _apply_plasticity3d_camera(ax, coords_plot)
    ax.set_axis_off()
    return tri_vals


def _plot_plasticity3d_validation_surface_compare(
    layout: dict[str, float],
    *,
    coords_source: np.ndarray,
    coords_candidate: np.ndarray,
    surface_faces: np.ndarray,
    values_source: np.ndarray,
    values_candidate: np.ndarray,
    out_name: str,
) -> str:
    plt = configure_paper_matplotlib(font_size=9.0)
    from matplotlib import cm
    from matplotlib.colors import Normalize

    values_source = np.asarray(values_source, dtype=np.float64)
    values_candidate = np.asarray(values_candidate, dtype=np.float64)
    diff = np.abs(values_candidate - values_source)
    main_vmax = float(np.quantile(np.concatenate([values_source, values_candidate]), 0.995))
    diff_vmax = float(max(np.quantile(diff, 0.995), 1.0e-12))
    main_norm = Normalize(vmin=0.0, vmax=max(main_vmax, 1.0e-12))
    diff_norm = Normalize(vmin=0.0, vmax=diff_vmax)

    fig = plt.figure(figsize=paper_figure_size(layout, preset="medium", height_ratio=0.50))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.13],
        left=0.01,
        right=0.99,
        bottom=0.14,
        top=0.94,
        wspace=0.02,
        hspace=0.02,
    )
    axes = [fig.add_subplot(gs[0, idx], projection="3d") for idx in range(3)]
    _surface_compare_panel(
        axes[0],
        coords_final=coords_source,
        surface_faces=surface_faces,
        nodal_values=values_source,
        cmap_name="viridis",
        norm=main_norm,
    )
    _surface_compare_panel(
        axes[1],
        coords_final=coords_candidate,
        surface_faces=surface_faces,
        nodal_values=values_candidate,
        cmap_name="viridis",
        norm=main_norm,
    )
    _surface_compare_panel(
        axes[2],
        coords_final=coords_candidate,
        surface_faces=surface_faces,
        nodal_values=diff,
        cmap_name="cividis",
        norm=diff_norm,
    )
    for ax, title in zip(axes, ("matched comparator", "JAX+PETSc", r"$|\Delta \|u\||$"), strict=True):
        ax.set_title(title, pad=2.0, fontsize=9.0)

    cax_main = fig.add_subplot(gs[1, :2])
    cax_diff = fig.add_subplot(gs[1, 2])
    main_pos = cax_main.get_position()
    diff_pos = cax_diff.get_position()
    cbar_gap = 0.022
    cbar_left_pad = 0.012
    cax_main.set_position(
        [
            main_pos.x0 + cbar_left_pad,
            main_pos.y0,
            main_pos.width - cbar_gap - cbar_left_pad,
            main_pos.height,
        ]
    )
    cax_diff.set_position([diff_pos.x0 + cbar_gap, diff_pos.y0, diff_pos.width - cbar_gap, diff_pos.height])
    sm_main = cm.ScalarMappable(norm=main_norm, cmap="viridis")
    sm_main.set_array(np.concatenate([values_source, values_candidate]))
    sm_diff = cm.ScalarMappable(norm=diff_norm, cmap="cividis")
    sm_diff.set_array(diff)
    cbar_main = fig.colorbar(sm_main, cax=cax_main, orientation="horizontal")
    cbar_main.set_label(r"$\|u\|$", labelpad=0.2)
    cbar_diff = fig.colorbar(sm_diff, cax=cax_diff, orientation="horizontal")
    cbar_diff.set_label(r"abs. diff.", labelpad=0.2)
    for cbar in (cbar_main, cbar_diff):
        cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        cbar.ax.tick_params(labelsize=7.5, pad=1.0)
        cbar.update_ticks()

    out = FIGURES_ROOT / out_name
    save_pdf_and_png(fig, out, png_dpi=260)
    plt.close(fig)
    return out.name


def generate_plasticity3d_validation_layer1a_boundary(layout: dict[str, float]) -> str:
    manifest = _read_json(P3D_VALIDATION_ROOT / "validation_manifest.json")
    layer_cfg = dict(manifest["layer1a"])
    source_summary = _read_json(_repo_path(layer_cfg["source_branch_dir"]) / "branch_summary.json")
    jax_summary = _read_json(_repo_path(layer_cfg["jax_branch_dir"]) / "branch_summary.json")
    final_jax_step = dict(jax_summary["steps"][-1])
    jax_state = _load_npz(_repo_path(final_jax_step["state_npz"]))
    source_mat = scipy.io.loadmat(_repo_path(source_summary["final_state_mat"]))
    source_coords_ref = np.asarray(source_mat["coord"], dtype=np.float64).T
    coords_ref = np.asarray(jax_state["coords_ref"], dtype=np.float64)
    _, src_to_jax = cKDTree(coords_ref).query(source_coords_ref, k=1)
    inv = np.empty_like(src_to_jax)
    inv[src_to_jax] = np.arange(src_to_jax.size, dtype=np.int64)

    source_disp = np.asarray(source_mat["U_final"], dtype=np.float64).T[np.asarray(inv, dtype=np.int64)]
    source_coords_final = coords_ref + source_disp
    candidate_disp = np.asarray(jax_state["displacement"], dtype=np.float64)
    candidate_coords_final = np.asarray(jax_state["coords_final"], dtype=np.float64)
    return _plot_plasticity3d_validation_surface_compare(
        layout,
        coords_source=source_coords_final,
        coords_candidate=candidate_coords_final,
        surface_faces=np.asarray(jax_state["surface_faces"], dtype=np.int64),
        values_source=np.linalg.norm(source_disp, axis=1),
        values_candidate=np.linalg.norm(candidate_disp, axis=1),
        out_name="plasticity3d_validation_layer1a_boundary.pdf",
    )


def generate_plasticity3d_validation_umax_curve(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib(font_size=9.0)
    summary = _read_json(P3D_VALIDATION_ROOT / "comparison_summary.json")
    rows = [dict(row) for row in dict(summary["layer2"])["rows"]]
    x = np.asarray([float(row["lambda_value"]) for row in rows], dtype=np.float64)
    source = np.asarray([float(row["source_u_max"]) for row in rows], dtype=np.float64)
    maintained = np.asarray([float(row["maintained_u_max"]) for row in rows], dtype=np.float64)
    rel_l2 = float(dict(summary["layer2"])["umax_curve_relative_l2"])

    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="medium", height_ratio=0.42))
    ax.plot(x, source, marker="o", linewidth=1.8, markersize=4.5, color="#111111", label="matched comparator")
    ax.plot(x, maintained, marker="s", linewidth=1.6, markersize=4.3, color="#777777", linestyle="--", label="JAX+PETSc")
    ax.set_xlabel(r"$\lambda_{\mathrm{sr}}$")
    ax.set_ylabel(r"$u_{\max}$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left", handlelength=1.7)
    ax.text(
        0.98,
        0.05,
        rf"rel. Eucl. $={_latex_scientific(rel_l2)}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.17, top=0.96)
    out = FIGURES_ROOT / "plasticity3d_validation_umax_curve.pdf"
    save_pdf_and_png(fig, out, png_dpi=260)
    plt.close(fig)
    return out.name


def generate_plasticity3d_derivative_ablation_bars(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib(font_size=9.0)
    summary = _read_json(P3D_DERIVATIVE_ABLATION_ROOT / "comparison_summary.json")
    rows = [dict(row) for row in summary["rows"]]
    label_by_route = {
        "element_ad": "Element AD",
        "constitutive_ad": "Constitutive AD",
        "colored_sfd": "Colored recovery",
    }
    labels = [label_by_route.get(str(row.get("route", "")), str(row["display_label"])) for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, axes = plt.subplots(1, 2, figsize=paper_figure_size(layout, preset="medium", height_ratio=0.42))
    axes[0].bar(x, [float(row["median_wall_time_s"]) for row in rows], color=colors, width=0.70)
    axes[0].set_ylabel("Median wall time [s]")
    axes[1].bar(x, [float(row["median_linear_iterations_total"]) for row in rows], color=colors, width=0.70)
    axes[1].set_ylabel("Median linear iterations")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.margins(x=0.05)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.29, top=0.94, wspace=0.34)
    out = FIGURES_ROOT / "plasticity3d_derivative_ablation_bars.pdf"
    save_pdf_and_png(fig, out, png_dpi=260)
    plt.close(fig)
    return out.name


def generate_jax_fem_hyperelastic_baseline_energy_history(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib(font_size=9.0)
    summary = _read_json(JAX_FEM_BASELINE_ROOT / "comparison_summary.json")
    step_rows = [dict(row) for row in summary["step_rows"]]
    steps = np.asarray([int(row["step"]) for row in step_rows], dtype=np.int64)
    repo = np.asarray([float(row["repo_energy"]) for row in step_rows], dtype=np.float64)
    jax_fem = np.asarray([float(row["jax_fem_energy"]) for row in step_rows], dtype=np.float64)
    rel_diff = float(dict(summary["final_metrics"])["energy_rel_diff"])

    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="subfigure", height_ratio=0.72))
    ax.plot(steps, repo, marker="o", linewidth=1.6, markersize=4.0, color="#111111", label="JAX+PETSc")
    ax.plot(steps, jax_fem, marker="s", linewidth=1.4, markersize=3.8, color="#777777", linestyle="--", label="JAX-FEM")
    ax.set_xlabel("Load step")
    ax.set_ylabel("Stored energy")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best", handlelength=1.7)
    ax.text(0.03, 0.95, rf"endpoint rel. diff. $={_latex_scientific(rel_diff)}$", transform=ax.transAxes, ha="left", va="top", fontsize=8.0)
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.23, top=0.95)
    out = FIGURES_ROOT / "jax_fem_hyperelastic_baseline_energy_history.pdf"
    save_pdf_and_png(fig, out, png_dpi=240)
    plt.close(fig)
    return out.name


def generate_jax_fem_hyperelastic_baseline_centerline(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib(font_size=9.0)
    summary = _read_json(JAX_FEM_BASELINE_ROOT / "comparison_summary.json")
    impls = [dict(row) for row in summary["implementations"]]
    repo_state = _load_npz(_repo_path(impls[0]["state_npz"]))
    jax_state = _load_npz(_repo_path(impls[1]["state_npz"]))
    coords_ref = np.asarray(repo_state["coords_ref"], dtype=np.float64)
    repo_profile = centerline_profile(coords_ref, np.asarray(repo_state["displacement"], dtype=np.float64))
    jax_profile = centerline_profile(coords_ref, np.asarray(jax_state["displacement"], dtype=np.float64))
    rel_l2 = float(dict(summary["final_metrics"])["centerline_relative_l2"])

    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="subfigure", height_ratio=0.72))
    ax.plot(repo_profile["x"], repo_profile["ux"], marker="o", linewidth=1.6, markersize=4.0, color="#111111", label="JAX+PETSc")
    ax.plot(jax_profile["x"], jax_profile["ux"], marker="s", linewidth=1.4, markersize=3.8, color="#777777", linestyle="--", label="JAX-FEM")
    ax.set_xlabel(r"Reference $x$")
    ax.set_ylabel(r"Centerline $u_x$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best", handlelength=1.7)
    ax.text(0.03, 0.95, rf"rel. Eucl. $={_latex_scientific(rel_l2)}$", transform=ax.transAxes, ha="left", va="top", fontsize=8.0)
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.23, top=0.95)
    out = FIGURES_ROOT / "jax_fem_hyperelastic_baseline_centerline.pdf"
    save_pdf_and_png(fig, out, png_dpi=240)
    plt.close(fig)
    return out.name


def generate_derivative_path_diagram(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    fig, ax = plt.subplots(figsize=text_figure_size(layout, height_ratio=0.46))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _draw_box(ax, (0.03, 0.58), (0.25, 0.13), title="", body=["Weak form", "energy / residual"], facecolor="#f4efe6", title_size=8.3, body_size=7.1)
    _draw_box(ax, (0.375, 0.58), (0.25, 0.13), title="", body=["JAX AD", "grad, Hessian, HVP"], facecolor="#e6f0fb", title_size=8.3, body_size=7.1)
    _draw_box(ax, (0.72, 0.58), (0.25, 0.13), title="", body=["PETSc solve", "distributed Newton"], facecolor="#eef5e7", title_size=8.3, body_size=7.1)
    _draw_box(ax, (0.09, 0.31), (0.20, 0.09), title="", body=["Element AD"], facecolor="#f7f4fb", title_size=8.1, body_size=7.0)
    _draw_box(ax, (0.40, 0.31), (0.20, 0.09), title="", body=["Constitutive AD"], facecolor="#fbf1eb", title_size=8.1, body_size=7.0)
    _draw_box(ax, (0.71, 0.31), (0.20, 0.09), title="", body=["Colored recovery"], facecolor="#f2f2f2", title_size=8.1, body_size=7.0)
    _draw_arrow(ax, (0.28, 0.645), (0.375, 0.645), shrink_a=4.0, shrink_b=4.0)
    _draw_arrow(ax, (0.625, 0.645), (0.72, 0.645), shrink_a=4.0, shrink_b=4.0)
    _draw_arrow(ax, (0.50, 0.58), (0.19, 0.40), shrink_a=6.0, shrink_b=8.0)
    _draw_arrow(ax, (0.50, 0.58), (0.50, 0.40), shrink_a=6.0, shrink_b=8.0)
    _draw_arrow(ax, (0.50, 0.58), (0.81, 0.40), shrink_a=6.0, shrink_b=8.0)
    out = FIGURES_ROOT / "derivative_paths.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def generate_globalization_schematic(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=text_figure_size(layout, height_ratio=0.44))
    panels = [
        ("Line search", ["Newton direction", "1D merit decrease", "accept by energy/residual drop"], "#e6f0fb"),
        ("Trust region", ["reduced model step", "$\\rho$-based acceptance", "radius shrink/expand"], "#eef5e7"),
        ("Hybrid", ["trust-model direction", "bounded line search", "fallback to gradient-safe step"], "#fbf1eb"),
    ]
    for ax, (title, lines, facecolor) in zip(axes, panels, strict=True):
        ax.set_axis_off()
        _draw_box(ax, (0.08, 0.16), (0.84, 0.68), title=title, body=lines, facecolor=facecolor, title_size=9.0, body_size=8.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.subplots_adjust(top=0.96, bottom=0.08, hspace=0.08)
    out = FIGURES_ROOT / "globalization_schematic.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def generate_coloring_schematic(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=text_figure_size(layout, height_ratio=0.66))
    rng = np.random.default_rng(0)
    base = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            if abs(i - j) <= 2 or rng.random() < 0.08:
                base[i, j] = 1.0
    color_map = np.array([0, 1, 2, 0, 3, 1, 2, 4, 0, 3, 1, 2, 4, 0], dtype=float)
    axes[0].imshow(base, cmap="Greys", interpolation="nearest", aspect="auto")
    axes[0].set_title("Sparse Hessian pattern")
    axes[1].imshow(color_map[None, :], cmap="tab20", aspect="auto", interpolation="nearest")
    axes[1].set_title("Distance-2 coloring")
    grouped = np.zeros((5, 14))
    for idx, c in enumerate(color_map.astype(int)):
        grouped[c, idx] = 1.0
    axes[2].imshow(grouped, cmap="Blues", aspect="auto", interpolation="nearest")
    axes[2].set_title("Probe groups / recovery")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    out = FIGURES_ROOT / "coloring_schematic.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def generate_autodiff_modes(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=text_figure_size(layout, height_ratio=0.44))
    panels = [
        ("Element AD", ["$\\Pi_e(u_e)$", "exact local gradient and Hessian", "higher-order elements can be costly"], "#e6f0fb"),
        ("Constitutive AD", ["$\\psi(\\varepsilon_q)$", "quadrature-point tangent assembly", "exact local constitutive derivatives"], "#eef5e7"),
        ("Colored recovery", ["rank-local probe HVPs only where needed", "parallel coloring / recovery", "useful when exact Hessians are too expensive"], "#fbf1eb"),
    ]
    for ax, (title, lines, color) in zip(axes, panels, strict=True):
        ax.set_axis_off()
        _draw_box(ax, (0.08, 0.16), (0.84, 0.68), title=title, body=lines, facecolor=color, title_size=9.0, body_size=8.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.subplots_adjust(top=0.96, bottom=0.08, hspace=0.08)
    out = FIGURES_ROOT / "autodiff_modes.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def _plot_plasticity_scaling(layout: dict[str, float], rows_by_impl: dict[str, list[dict[str, object]]]) -> list[str]:
    plt = configure_paper_matplotlib()
    outputs: list[str] = []
    rows = rows_by_impl[LOCAL_IMPL]
    ranks = np.asarray([int(row["ranks"]) for row in rows], dtype=np.int64)
    wall = np.asarray([_safe_float(row["wall_time_s"]) for row in rows], dtype=np.float64)
    max_local_elements = np.asarray(
        [
            int(
                _read_json(_repo_path(row["result_json"]))["parallel_setup"]["local_elements_max"]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="medium", height_ratio=0.52))
    ax.plot(ranks, wall, marker="o", color="#1f77b4", linewidth=2.0, label="Measured")
    impl_assets._plot_ideal_reference(ax, ranks, wall, color="#1f77b4")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ranks)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Wall time [s]", labelpad=2.0)
    ax.plot([], [], linewidth=1.2, linestyle=(0, (4, 3)), color="#555555", alpha=0.7, label="Ideal $1/p$")
    left_ylim = ax.get_ylim()
    align_scale = float(max_local_elements[0] / wall[0]) if float(wall[0]) > 0.0 else 1.0
    ax_right = ax.twinx()
    ax_right.plot(
        ranks,
        max_local_elements,
        marker="s",
        color="#d62728",
        linewidth=1.8,
        label="Max local elements",
    )
    ax_right.set_yscale("log")
    ax_right.set_ylim(left_ylim[0] * align_scale, left_ylim[1] * align_scale)
    ax_right.set_ylabel("Max local elements", color="#d62728", labelpad=4.0)
    ax_right.tick_params(axis="y", colors="#d62728")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_right.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.subplots_adjust(top=0.94, bottom=0.23, left=0.21, right=0.76)
    out = FIGURES_ROOT / "plasticity3d_recommended_scaling.pdf"
    save_pdf_and_png(fig, out)
    outputs.append(out.name)
    plt.close(fig)
    return outputs


def generate_plasticity3d_cpu_scaling(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    local_rows = read_csv_rows(P3D_LOCAL_LAMBDA155_SCALING)
    multinode_rows = [
        row
        for row in read_csv_rows(P3D_MULTINODE_LAMBDA155_SCALING)
        if row.get("output") == "yes" and row.get("solver_total")
    ]
    local_rows.sort(key=lambda row: int(row["ranks"]))
    multinode_rows.sort(key=lambda row: int(row["ranks"]))

    fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="medium", height_ratio=0.52))
    series = [
        (
            "single-node CPU",
            np.asarray([int(row["ranks"]) for row in local_rows], dtype=np.int64),
            np.asarray([float(row["solver_total_s"]) for row in local_rows], dtype=np.float64),
            "#1f77b4",
            "o",
        ),
        (
            "multi-node CPU, 16 ranks/node",
            np.asarray([int(row["ranks"]) for row in multinode_rows], dtype=np.int64),
            np.asarray([float(row["solver_total"]) for row in multinode_rows], dtype=np.float64),
            "#d62728",
            "s",
        ),
    ]
    all_ranks: list[int] = []
    for label, ranks, times, color, marker in series:
        all_ranks.extend(int(rank) for rank in ranks)
        ax.plot(ranks, times, marker=marker, color=color, linewidth=1.9, markersize=4.6, label=label)
        ax.plot(ranks, ideal_strong_scaling(ranks, times), color=color, linestyle="--", linewidth=1.0, alpha=0.42)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(np.asarray(sorted(set(all_ranks)), dtype=np.int64))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Solver total time [s]")
    ax.margins(x=0.04, y=0.12)
    ax.legend(frameon=True, loc="best")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.96)

    out = FIGURES_ROOT / "plasticity3d_cpu_scaling.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def _plot_local_vs_source(layout: dict[str, float], rows: list[dict[str, object]]) -> str:
    plt = configure_paper_matplotlib()
    local_rows = _find_rows(rows, LOCAL_IMPL)
    source_rows = _find_rows(rows, SOURCE_IMPL)
    fig, axes = plt.subplots(2, 1, figsize=text_figure_size(layout, height_ratio=0.95))
    for selected, label, color in (
        (local_rows, "local_constitutiveAD + local_pmg", "#1f77b4"),
        (source_rows, "source + local_pmg", "#d62728"),
    ):
        ranks = np.asarray([int(row["ranks"]) for row in selected], dtype=np.int64)
        wall = np.asarray([_safe_float(row.get("wall_time_s")) for row in selected], dtype=np.float64)
        solve = np.asarray([_safe_float(row.get("solve_time_s")) for row in selected], dtype=np.float64)
        axes[0].plot(ranks, wall, marker="o", linewidth=2.0, color=color, label=label)
        axes[1].plot(ranks, solve, marker="o", linewidth=2.0, color=color, label=label)
        impl_assets._plot_ideal_reference(axes[0], ranks, wall, color=color)
        impl_assets._plot_ideal_reference(axes[1], ranks, solve, color=color)
    for ax, title, ylabel in (
        (axes[0], "Matched wall time", "Wall time [s]"),
        (axes[1], "Matched solve time", "Solve time [s]"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel(ylabel)
    axes[1].plot([], [], linewidth=1.2, linestyle=(0, (4, 3)), color="#555555", alpha=0.7, label="Ideal $1/p$")
    axes[1].legend(loc="best")
    fig.subplots_adjust(top=0.95, bottom=0.10, hspace=0.34)
    out = FIGURES_ROOT / "plasticity3d_local_vs_source.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def _plot_sourcefixed(layout: dict[str, float], rows: list[dict[str, object]]) -> str:
    plt = configure_paper_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=text_figure_size(layout, height_ratio=0.95))
    series = (
        (LOCAL_IMPL, "local + local_pmg", "#1f77b4"),
        (SOURCE_IMPL, "source + local_pmg", "#d62728"),
        (LOCAL_SOURCEFIXED_IMPL, "AD tangent + frozen PMG", "#2ca02c"),
        (SOURCE_SOURCEFIXED_IMPL, "reference tangent + frozen PMG", "#ff7f0e"),
    )
    for implementation, label, color in series:
        selected = _find_rows(rows, implementation)
        if not selected:
            continue
        converged = [row for row in selected if str(row.get("status", "")) == "completed"]
        failed = [row for row in selected if str(row.get("status", "")) != "completed"]
        if converged:
            ranks = np.asarray([int(row["ranks"]) for row in converged], dtype=np.int64)
            wall = np.asarray([_safe_float(row.get("wall_time_s")) for row in converged], dtype=np.float64)
            solve = np.asarray([_safe_float(row.get("solve_time_s")) for row in converged], dtype=np.float64)
            axes[0].plot(ranks, wall, marker="o", linewidth=2.0, color=color, label=label)
            axes[1].plot(ranks, solve, marker="o", linewidth=2.0, color=color, label=label)
            impl_assets._plot_ideal_reference(axes[0], ranks, wall, color=color)
            impl_assets._plot_ideal_reference(axes[1], ranks, solve, color=color)
        for ax, key in ((axes[0], "wall_time_s"), (axes[1], "solve_time_s")):
            if not failed:
                continue
            x = np.asarray([int(row["ranks"]) for row in failed], dtype=np.int64)
            y = np.asarray([_safe_float(row.get(key)) for row in failed], dtype=np.float64)
            ax.scatter(x, y, marker="x", s=60, color=color, alpha=0.9)
            for row in failed:
                ax.annotate("maxit", (int(row["ranks"]), _safe_float(row.get(key))), xytext=(4, 4), textcoords="offset points", fontsize=8, color=color)
    for ax, title, ylabel in (
        (axes[0], "Converged wall time", "Wall time [s]"),
        (axes[1], "Converged solve time", "Solve time [s]"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel(ylabel)
    axes[1].plot([], [], linewidth=1.2, linestyle=(0, (4, 3)), color="#555555", alpha=0.7, label="Ideal $1/p$")
    axes[1].legend(loc="best", fontsize=8)
    fig.subplots_adjust(top=0.95, bottom=0.10, hspace=0.34)
    out = FIGURES_ROOT / "plasticity3d_sourcefixed_compare.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def _plot_plasticity3d_degree_energy_study(layout: dict[str, float]) -> list[str]:
    plt = configure_paper_matplotlib()
    summary = _read_json(P3D_DEGREE_ENERGY_STUDY_PLOT_SUMMARY)
    rows = sorted(
        (dict(row) for row in summary.get("rows", []) if isinstance(row, dict)),
        key=lambda row: (int(str(row.get("degree_line", "P0")).replace("P", "")), int(row.get("free_dofs", 0))),
    )
    styles = {
        "P1": {"label": _degree_math_label("P1"), "color": "#1f77b4", "marker": "o"},
        "P2": {"label": _degree_math_label("P2"), "color": "#ff7f0e", "marker": "s"},
        "P4": {"label": _degree_math_label("P4"), "color": "#2ca02c", "marker": "^"},
    }

    def _plot_panel(
        degree_lines: tuple[str, ...],
        *,
        x_key: str,
        xlabel: str,
        out_name: str,
        legend: bool = False,
        legend_loc: str = "upper right",
        legend_ncol: int = 2,
        show_ylabel: bool = True,
        y_formatter: str = "%.2f",
    ) -> str:
        fig, ax = plt.subplots(figsize=paper_figure_size(layout, preset="subfigure", height_ratio=0.58))
        for degree_line in degree_lines:
            style = styles[degree_line]
            selected = [row for row in rows if str(row.get("degree_line", "")) == degree_line]
            converged = [row for row in selected if str(row.get("status", "")) == "completed"]
            converged_sorted = sorted(converged, key=lambda row: _safe_float(row.get(x_key)))
            if not converged_sorted:
                continue
            x = np.asarray([_safe_float(row.get(x_key)) for row in converged_sorted], dtype=np.float64)
            y = np.asarray([_safe_float(row.get("energy")) / 1.0e6 for row in converged_sorted], dtype=np.float64)
            ax.plot(
                x,
                y,
                marker=style["marker"],
                color=style["color"],
                linewidth=2.0,
                markersize=6.0,
                label=style["label"] if legend else None,
            )
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel(xlabel)
        if show_ylabel:
            ax.set_ylabel(r"Final energy [$10^{6}$]", labelpad=1.5)
        if legend:
            ax.legend(
                loc=legend_loc,
                ncol=legend_ncol,
                frameon=False,
                handlelength=2.0,
                borderaxespad=0.25,
                columnspacing=1.0,
            )
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(y_formatter))
        fig.subplots_adjust(
            top=0.90,
            bottom=0.25,
            left=0.25 if show_ylabel else 0.18,
            right=0.97,
            hspace=0.0,
            wspace=0.0,
        )
        out = FIGURES_ROOT / out_name
        save_pdf_and_png(fig, out)
        plt.close(fig)
        return out.name

    outputs = [
        _plot_panel(
            ("P1", "P2", "P4"),
            x_key="free_dofs",
            xlabel="Free DOFs",
            out_name="plasticity3d_degree_energy_all_dofs.pdf",
            legend=True,
            legend_loc="upper right",
            legend_ncol=2,
            show_ylabel=True,
            y_formatter="%.2f",
        ),
        _plot_panel(
            ("P1", "P2", "P4"),
            x_key="total_time_s",
            xlabel="Total wall time [s]",
            out_name="plasticity3d_degree_energy_all_time.pdf",
            legend=False,
            legend_loc="upper right",
            legend_ncol=2,
            show_ylabel=False,
            y_formatter="%.2f",
        ),
        _plot_panel(
            ("P2", "P4"),
            x_key="free_dofs",
            xlabel="Free DOFs",
            out_name="plasticity3d_degree_energy_zoom_dofs.pdf",
            legend=True,
            legend_loc="lower left",
            legend_ncol=1,
            show_ylabel=True,
            y_formatter="%.4f",
        ),
        _plot_panel(
            ("P2", "P4"),
            x_key="total_time_s",
            xlabel="Total wall time [s]",
            out_name="plasticity3d_degree_energy_zoom_time.pdf",
            legend=False,
            legend_loc="lower left",
            legend_ncol=1,
            show_ylabel=False,
            y_formatter="%.4f",
        ),
    ]
    return outputs


def _plot_source_continuation_compare(layout: dict[str, float]) -> str:
    plt = configure_paper_matplotlib()
    np8 = _read_json(SOURCE_CONT_NP8)
    np32 = _read_json(SOURCE_CONT_NP32)
    labels = ["runtime", "init linear", "continuation linear"]
    v8 = np.array(
        [
            float(np8["run_info"]["runtime_seconds"]),
            float(np8["timings"]["linear"]["init_linear_iterations"]),
            float(np8["timings"]["linear"]["attempt_linear_iterations_total"]),
        ]
    )
    v32 = np.array(
        [
            float(np32["run_info"]["runtime_seconds"]),
            float(np32["timings"]["linear"]["init_linear_iterations"]),
            float(np32["timings"]["linear"]["attempt_linear_iterations_total"]),
        ]
    )
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=text_figure_size(layout, height_ratio=0.62))
    ax.bar(x - width / 2, v8, width=width, label="8 ranks", color="#1f77b4")
    ax.bar(x + width / 2, v32, width=width, label="32 ranks", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Value")
    ax.set_title("Reference-operator continuation comparison")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(loc="best")
    out = FIGURES_ROOT / "source_continuation_pmg_shell_compare.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)
    return out.name


def _figure_source(
    *,
    generator: str,
    data_inputs: list[dict[str, str]] | None = None,
    archive_status: str = "archive_neutral",
    note: str = "",
) -> dict[str, object]:
    source: dict[str, object] = {
        "generator": {
            "kind": "repository_path",
            "path": "paper/scripts/generate_paper_figures.py",
            "function": generator,
        },
        "archive_status": archive_status,
    }
    if data_inputs:
        source["data_inputs"] = data_inputs
    if note:
        source["note"] = note
    return source


def _figure_sources() -> dict[str, dict[str, object]]:
    tracked_or_bundle = "archive_neutral"
    needs_archive = "needs_final_archive"
    return {
        "plaplace_state.pdf": _figure_source(
            generator="generate_scalar_state_figure",
            data_inputs=[_manifest_repo_input(PLAPLACE_STATE)],
        ),
        "plaplace_energy_levels.pdf": _figure_source(
            generator="generate_energy_levels_figure",
            data_inputs=[_manifest_repo_input(PLAPLACE_ENERGY)],
        ),
        "plaplace_scaling.pdf": _figure_source(
            generator="generate_family_scaling_figure",
            data_inputs=[_manifest_repo_input(PLAPLACE_SCALING)],
        ),
        "ginzburg_landau_state.pdf": _figure_source(
            generator="generate_scalar_state_figure",
            data_inputs=[_manifest_repo_input(GL_STATE)],
        ),
        "ginzburg_landau_energy_levels.pdf": _figure_source(
            generator="generate_energy_levels_figure",
            data_inputs=[_manifest_repo_input(GL_ENERGY)],
        ),
        "ginzburg_landau_scaling.pdf": _figure_source(
            generator="generate_family_scaling_figure",
            data_inputs=[_manifest_repo_input(GL_SCALING)],
        ),
        "hyperelasticity_state.pdf": _figure_source(
            generator="generate_hyperelasticity_state",
            data_inputs=[_manifest_repo_input(HYPER_STATE)],
        ),
        "hyperelasticity_energy_levels.pdf": _figure_source(
            generator="generate_energy_levels_figure",
            data_inputs=[_manifest_repo_input(HYPER_ENERGY)],
        ),
        "hyperelasticity_scaling.pdf": _figure_source(
            generator="generate_family_scaling_figure",
            data_inputs=[_manifest_repo_input(HYPER_SCALING)],
        ),
        "hyperelasticity_cpu_pmg_scaling.pdf": _figure_source(
            generator="generate_hyperelasticity_cpu_pmg_scaling",
            data_inputs=[_manifest_repo_input(HYPER_CPU_PMG_SCALING)],
        ),
        "plasticity2d_state_pair.pdf": _figure_source(
            generator="generate_plasticity2d_figures",
            data_inputs=[_manifest_repo_input(PLASTICITY2D_STATE)],
            note="Uses the bundled Plasticity2D endpoint state plus deterministic same-mesh case construction.",
        ),
        "plasticity2d_resolution_energy.pdf": _figure_source(
            generator="generate_plasticity2d_figures",
            data_inputs=[
                _manifest_repo_input(PLASTICITY2D_RESULT),
                _manifest_repo_input(PLASTICITY2D_L6_SUMMARY),
                _manifest_repo_input(PLASTICITY2D_L7_SUMMARY),
            ],
        ),
        "plasticity3d_state_pair.pdf": _figure_source(
            generator="generate_plasticity3d_state_figures",
            data_inputs=[
                _manifest_repo_input(P3D_STATE_PAIR_SURFACE_METADATA),
                _manifest_repo_input(P3D_STATE_PAIR_SURFACE_ARRAYS),
            ],
            note="Uses derived plotted surface arrays with source hashes recorded in the bundle metadata.",
        ),
        "plasticity3d_convergence.pdf": _figure_source(
            generator="_generate_plasticity3d_convergence_figure",
            data_inputs=[_manifest_repo_input(P3D_DEGREE_ENERGY_STUDY_PLOT_SUMMARY)],
        ),
        "plasticity3d_highest_mesh_y_slice_comparison.pdf": _figure_source(
            generator="generate_plasticity3d_highest_y_slice_comparison",
            data_inputs=[
                _manifest_repo_input(P3D_HIGHEST_Y_SLICE_METADATA),
                _manifest_repo_input(P3D_HIGHEST_Y_SLICE_ARRAYS),
            ],
            note="Uses derived interpolated y-slice arrays with source hashes recorded in the bundle metadata.",
        ),
        "plasticity3d_degree_energy_all_dofs.pdf": _figure_source(
            generator="_plot_plasticity3d_degree_energy_study",
            data_inputs=[_manifest_repo_input(P3D_DEGREE_ENERGY_STUDY_PLOT_SUMMARY)],
        ),
        "plasticity3d_degree_energy_all_time.pdf": _figure_source(
            generator="_plot_plasticity3d_degree_energy_study",
            data_inputs=[_manifest_repo_input(P3D_DEGREE_ENERGY_STUDY_PLOT_SUMMARY)],
        ),
        "plasticity3d_degree_energy_zoom_dofs.pdf": _figure_source(
            generator="_plot_plasticity3d_degree_energy_study",
            data_inputs=[_manifest_repo_input(P3D_DEGREE_ENERGY_STUDY_PLOT_SUMMARY)],
        ),
        "plasticity3d_degree_energy_zoom_time.pdf": _figure_source(
            generator="_plot_plasticity3d_degree_energy_study",
            data_inputs=[_manifest_repo_input(P3D_DEGREE_ENERGY_STUDY_PLOT_SUMMARY)],
        ),
        "topology_density.pdf": _figure_source(
            generator="generate_topology_density",
            data_inputs=[_manifest_repo_input(TOPOLOGY_STATE)],
        ),
        "topology_history.pdf": _figure_source(
            generator="generate_topology_history",
            data_inputs=[_manifest_repo_input(TOPOLOGY_HISTORY)],
        ),
        "topology_scaling.pdf": _figure_source(
            generator="generate_topology_scaling",
            data_inputs=[_manifest_repo_input(TOPOLOGY_SCALING)],
        ),
        "plasticity3d_validation_layer1a_boundary.pdf": _figure_source(
            generator="generate_plasticity3d_validation_layer1a_boundary",
            data_inputs=[
                _manifest_repo_input(P3D_VALIDATION_ROOT / "validation_manifest.json"),
                _manifest_repo_input(P3D_VALIDATION_ROOT / "comparison_summary.json"),
                _manifest_repo_input(P3D_VALIDATION_ROOT / "source_branch/final_source_state.mat"),
                _manifest_repo_input(P3D_VALIDATION_ROOT / "source_branch/branch_summary.json"),
                _manifest_repo_input(P3D_VALIDATION_ROOT / "maintained_branch/branch_summary.json"),
            ],
            archive_status=tracked_or_bundle,
        ),
        "plasticity3d_validation_umax_curve.pdf": _figure_source(
            generator="generate_plasticity3d_validation_umax_curve",
            data_inputs=[_manifest_repo_input(P3D_VALIDATION_ROOT / "comparison_summary.json")],
            archive_status=tracked_or_bundle,
        ),
        "plasticity3d_derivative_ablation_bars.pdf": _figure_source(
            generator="generate_plasticity3d_derivative_ablation_bars",
            data_inputs=[_manifest_repo_input(P3D_DERIVATIVE_ABLATION_ROOT / "comparison_summary.json")],
            archive_status=tracked_or_bundle,
        ),
        "jax_fem_hyperelastic_baseline_energy_history.pdf": _figure_source(
            generator="generate_jax_fem_hyperelastic_baseline_energy_history",
            data_inputs=[_manifest_repo_input(JAX_FEM_BASELINE_ROOT / "comparison_summary.json")],
            archive_status=tracked_or_bundle,
        ),
        "jax_fem_hyperelastic_baseline_centerline.pdf": _figure_source(
            generator="generate_jax_fem_hyperelastic_baseline_centerline",
            data_inputs=[
                _manifest_repo_input(JAX_FEM_BASELINE_ROOT / "comparison_summary.json"),
                _manifest_repo_input(JAX_FEM_BASELINE_ROOT / "parity/repo_serial_direct_state.npz"),
                _manifest_repo_input(JAX_FEM_BASELINE_ROOT / "parity/jax_fem_umfpack_serial_state.npz"),
            ],
            archive_status=tracked_or_bundle,
        ),
        "plasticity3d_recommended_scaling.pdf": _figure_source(
            generator="_plot_plasticity_scaling",
            data_inputs=[
                _manifest_repo_input(LOCAL_P3D_SUMMARY),
                *[_manifest_repo_input(path) for path in P3D_RECOMMENDED_SCALING_OUTPUTS],
            ],
            archive_status=tracked_or_bundle,
            note="The bundled scaling summary references bundled per-rank result JSON files used for local-element counts.",
        ),
        "plasticity3d_cpu_scaling.pdf": _figure_source(
            generator="generate_plasticity3d_cpu_scaling",
            data_inputs=[
                _manifest_repo_input(P3D_LOCAL_LAMBDA155_SCALING),
                _manifest_repo_input(P3D_MULTINODE_LAMBDA155_SCALING),
            ],
            archive_status=tracked_or_bundle,
        ),
    }


def _archive_neutral_figure_inputs(sources: dict[str, dict[str, object]]) -> dict[str, list[dict[str, str]]]:
    inputs: dict[str, list[dict[str, str]]] = {}
    for name, source in sorted(sources.items()):
        if source.get("archive_status") != "archive_neutral":
            continue
        data_inputs = source.get("data_inputs", [])
        if isinstance(data_inputs, list) and data_inputs:
            inputs[name] = data_inputs
    return inputs


def _write_figure_manifest(out_dir: Path, layout: dict[str, float], generated: list[str]) -> None:
    sources = _figure_sources()
    notes = (
        "All manuscript-included PDFs in this manifest are generated by "
        "paper/scripts/generate_paper_figures.py; no external PDFs are copied "
        "into paper/figures/generated."
    )
    if any(source["archive_status"] == "needs_final_archive" for source in sources.values()):
        notes = (
            notes
            + " Assets marked needs_final_archive require additional raw/state inputs "
            + "in the final durable archive or an explicit submission-scope exception."
        )
    manifest = {
        "copied_assets": [],
        "generated_assets": generated,
        "layout": {
            "textwidth_in": layout["textwidth_in"],
            "subfigure_width_in": 0.46 * layout["textwidth_in"],
            "narrow_width_in": 0.72 * layout["textwidth_in"],
            "medium_width_in": 0.84 * layout["textwidth_in"],
            "full_width_in": layout["textwidth_in"],
        },
        "generated_asset_sources": {name: sources[name] for name in generated if name in sources},
        "generated_asset_inputs": _archive_neutral_figure_inputs(sources),
        "notes": notes,
    }
    write_json(out_dir / "manifest.json", manifest)


def _expected_generated_assets() -> list[str]:
    return [
        "plaplace_state.pdf",
        "plaplace_energy_levels.pdf",
        "plaplace_scaling.pdf",
        "ginzburg_landau_state.pdf",
        "ginzburg_landau_energy_levels.pdf",
        "ginzburg_landau_scaling.pdf",
        "hyperelasticity_state.pdf",
        "hyperelasticity_energy_levels.pdf",
        "hyperelasticity_scaling.pdf",
        "hyperelasticity_cpu_pmg_scaling.pdf",
        "plasticity2d_state_pair.pdf",
        "plasticity2d_resolution_energy.pdf",
        "plasticity3d_state_pair.pdf",
        "plasticity3d_convergence.pdf",
        "plasticity3d_highest_mesh_y_slice_comparison.pdf",
        "plasticity3d_degree_energy_all_dofs.pdf",
        "plasticity3d_degree_energy_all_time.pdf",
        "plasticity3d_degree_energy_zoom_dofs.pdf",
        "plasticity3d_degree_energy_zoom_time.pdf",
        "topology_density.pdf",
        "topology_history.pdf",
        "topology_scaling.pdf",
        "plasticity3d_validation_layer1a_boundary.pdf",
        "plasticity3d_validation_umax_curve.pdf",
        "plasticity3d_derivative_ablation_bars.pdf",
        "jax_fem_hyperelastic_baseline_energy_history.pdf",
        "jax_fem_hyperelastic_baseline_centerline.pdf",
        "plasticity3d_recommended_scaling.pdf",
        "plasticity3d_cpu_scaling.pdf",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from curated repo assets and raw summaries.")
    parser.add_argument("--out-dir", type=Path, default=FIGURES_ROOT)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="refresh the figure manifest without regenerating figure PDFs",
    )
    args = parser.parse_args()
    ensure_paper_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layout = load_layout()
    if args.manifest_only:
        _write_figure_manifest(args.out_dir, layout, _expected_generated_assets())
        print(f"Wrote figure manifest to {args.out_dir / 'manifest.json'}")
        return

    generated: list[str] = []
    generated.append(
        generate_scalar_state_figure(
            layout,
            npz_path=PLAPLACE_STATE,
            out_name="plaplace_state.pdf",
            cbar_label=r"$u_h(x,y)$",
            cmap="Greys",
        )
    )
    generated.append(
        generate_energy_levels_figure(
            layout,
            csv_path=PLAPLACE_ENERGY,
            implementations=("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"),
            out_name="plaplace_energy_levels.pdf",
            ylabel=r"Final energy $\mathcal{E}_h$",
        )
    )
    generated.append(
        generate_family_scaling_figure(
            layout,
            csv_path=PLAPLACE_SCALING,
            implementations=("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"),
            out_name="plaplace_scaling.pdf",
        )
    )
    generated.append(
        generate_scalar_state_figure(
            layout,
            npz_path=GL_STATE,
            out_name="ginzburg_landau_state.pdf",
            cbar_label=r"$u_h(x,y)$",
            cmap="Greys",
        )
    )
    generated.append(
        generate_energy_levels_figure(
            layout,
            csv_path=GL_ENERGY,
            implementations=("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"),
            out_name="ginzburg_landau_energy_levels.pdf",
            ylabel=r"Final energy $\mathcal{E}_h$",
        )
    )
    generated.append(
        generate_family_scaling_figure(
            layout,
            csv_path=GL_SCALING,
            implementations=("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"),
            out_name="ginzburg_landau_scaling.pdf",
        )
    )
    generated.append(generate_hyperelasticity_state(layout))
    generated.append(
        generate_energy_levels_figure(
            layout,
            csv_path=HYPER_ENERGY,
            implementations=("fenics_custom", "jax_serial", "jax_petsc_element"),
            out_name="hyperelasticity_energy_levels.pdf",
            ylabel=r"Endpoint energy $\Pi_h$",
        )
    )
    generated.append(
        generate_family_scaling_figure(
            layout,
            csv_path=HYPER_SCALING,
            implementations=("fenics_custom", "jax_serial", "jax_petsc_element"),
            out_name="hyperelasticity_scaling.pdf",
        )
    )
    generated.append(generate_hyperelasticity_cpu_pmg_scaling(layout))
    generated.extend(generate_plasticity2d_figures(layout))
    generated.extend(generate_plasticity3d_state_figures(layout))
    generated.append(generate_plasticity3d_highest_y_slice_comparison(layout))
    generated.extend(_plot_plasticity3d_degree_energy_study(layout))
    generated.append(generate_topology_density(layout))
    generated.append(generate_topology_history(layout))
    generated.append(generate_topology_scaling(layout))
    generated.append(generate_plasticity3d_validation_layer1a_boundary(layout))
    generated.append(generate_plasticity3d_validation_umax_curve(layout))
    generated.append(generate_plasticity3d_derivative_ablation_bars(layout))
    generated.append(generate_jax_fem_hyperelastic_baseline_energy_history(layout))
    generated.append(generate_jax_fem_hyperelastic_baseline_centerline(layout))

    _, local_rows_by_impl = _load_impl_rows(LOCAL_P3D_SUMMARY)
    generated.extend(_plot_plasticity_scaling(layout, local_rows_by_impl))
    generated.append(generate_plasticity3d_cpu_scaling(layout))

    _write_figure_manifest(args.out_dir, layout, generated)
    print(f"Wrote figure manifest to {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
