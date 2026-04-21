#!/usr/bin/env python3
"""Generate Plasticity3D docs assets for the converged `P2(L1)` `lambda=1.6` card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from experiments.analysis.generate_plasticity3d_p4_l1_docs_assets import (
    _plot_convergence,
    _plot_surface_scalar_field,
)
from src.problems.slope_stability_3d.support.mesh import (
    PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM,
    load_case_hdf5,
    same_mesh_case_hdf5_path,
)
from src.problems.slope_stability_3d.support.simplex_lagrange import evaluate_tetra_lagrange_basis


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "docs_showcase"
    / "plasticity3d_p2_l1_lambda1p6_from_scratch"
    / "state.npz"
)
DEFAULT_RESULT = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "docs_showcase"
    / "plasticity3d_p2_l1_lambda1p6_from_scratch"
    / "output.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "assets" / "plasticity3d"
DEFAULT_MESH_NAME = "hetero_ssr_L1"
DEFAULT_DEGREE = 2
DEFAULT_SLUG = "plasticity3d_p2_l1_lambda1p6_from_scratch"
DEFAULT_CONSTRAINT_VARIANT = PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM


def _save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _quadrature_points_tetra_p2() -> np.ndarray:
    return np.array(
        [
            [
                1.0 / 4.0,
                0.0714285714285714,
                0.785714285714286,
                0.0714285714285714,
                0.0714285714285714,
                0.399403576166799,
                0.100596423833201,
                0.100596423833201,
                0.399403576166799,
                0.399403576166799,
                0.100596423833201,
            ],
            [
                1.0 / 4.0,
                0.0714285714285714,
                0.0714285714285714,
                0.785714285714286,
                0.0714285714285714,
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
            ],
            [
                1.0 / 4.0,
                0.0714285714285714,
                0.0714285714285714,
                0.0714285714285714,
                0.785714285714286,
                0.100596423833201,
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
                0.399403576166799,
            ],
        ],
        dtype=np.float64,
    )


def _dev_strain_norm(strain6: np.ndarray) -> np.ndarray:
    dev = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    ) / 3.0
    arr = np.asarray(strain6, dtype=np.float64)
    proj = arr @ dev.T
    return np.sqrt(np.maximum(0.0, np.sum(arr * proj, axis=1)))


def _slice_axis_metadata(axis: int) -> dict[str, object]:
    axis = int(axis)
    if axis == 0:
        return {"axis_name": "x", "proj": (1, 2), "xlabel": "y", "ylabel": "z"}
    if axis == 1:
        return {"axis_name": "y", "proj": (0, 2), "xlabel": "x", "ylabel": "z"}
    if axis == 2:
        return {"axis_name": "z", "proj": (0, 1), "xlabel": "x", "ylabel": "y"}
    raise ValueError(f"Unsupported axis {axis}")


def _tetra_plane_polygon_projected(
    points: np.ndarray,
    tet4: np.ndarray,
    *,
    axis: int,
    center: float,
) -> np.ndarray | None:
    tet_pts = np.asarray(points[np.asarray(tet4[:4], dtype=np.int64)], dtype=np.float64)
    distances = tet_pts[:, int(axis)] - float(center)
    tol = 1.0e-10 * max(1.0, float(np.max(np.abs(tet_pts[:, int(axis)]))))
    if np.all(distances > tol) or np.all(distances < -tol):
        return None

    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    intersections: list[np.ndarray] = []

    def _append_unique(pt: np.ndarray) -> None:
        for prev in intersections:
            if np.linalg.norm(prev - pt) <= 1.0e-9:
                return
        intersections.append(pt)

    for idx in range(4):
        if abs(distances[idx]) <= tol:
            _append_unique(tet_pts[idx])

    for i, j in edge_pairs:
        di = distances[i]
        dj = distances[j]
        if di * dj < -tol * tol:
            t = float(di / (di - dj))
            _append_unique(tet_pts[i] + t * (tet_pts[j] - tet_pts[i]))
        elif abs(di) <= tol and abs(dj) > tol:
            _append_unique(tet_pts[i])
        elif abs(dj) <= tol and abs(di) > tol:
            _append_unique(tet_pts[j])

    if len(intersections) < 3:
        return None

    proj0, proj1 = _slice_axis_metadata(int(axis))["proj"]
    projected = np.asarray([[pt[int(proj0)], pt[int(proj1)]] for pt in intersections], dtype=np.float64)
    centroid = np.mean(projected, axis=0)
    angles = np.arctan2(projected[:, 1] - centroid[1], projected[:, 0] - centroid[0])
    return projected[np.argsort(angles)]


def _slice_footprint_mask(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    axis: int,
    center: float,
    extent: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    xmin, xmax, ymin, ymax = (float(v) for v in extent)
    ny, nx = (int(shape[0]), int(shape[1]))
    image = Image.new("L", (nx, ny), 0)
    draw = ImageDraw.Draw(image)
    scale_x = (nx - 1) / max(xmax - xmin, 1.0e-12)
    scale_y = (ny - 1) / max(ymax - ymin, 1.0e-12)
    for tet in np.asarray(tetrahedra, dtype=np.int64):
        polygon = _tetra_plane_polygon_projected(points, tet, axis=int(axis), center=float(center))
        if polygon is None:
            continue
        pixels = [((px - xmin) * scale_x, (py - ymin) * scale_y) for px, py in polygon]
        if len(pixels) >= 3:
            draw.polygon(pixels, fill=1, outline=1)
    return np.asarray(image, dtype=bool)


def _interpolate_planar_slice(
    points: np.ndarray,
    values: np.ndarray,
    *,
    axis: int,
    center: float,
    half_thickness: float,
    footprint_points: np.ndarray,
    footprint_tetrahedra: np.ndarray,
    resolution: int = 900,
    smooth_sigma: float = 1.0,
) -> dict[str, object]:
    pts = np.asarray(points, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    meta = _slice_axis_metadata(int(axis))
    proj0, proj1 = (int(v) for v in meta["proj"])
    mask = np.abs(pts[:, int(axis)] - float(center)) <= float(half_thickness)
    pts_sel = np.asarray(pts[mask][:, [proj0, proj1]], dtype=np.float64)
    vals_sel = np.asarray(vals[mask], dtype=np.float64)
    if pts_sel.shape[0] < 3:
        raise RuntimeError(f"Not enough slice samples for axis {axis}")

    xmin = float(np.min(pts[:, proj0]))
    xmax = float(np.max(pts[:, proj0]))
    ymin = float(np.min(pts[:, proj1]))
    ymax = float(np.max(pts[:, proj1]))
    width = max(xmax - xmin, 1.0e-12)
    height = max(ymax - ymin, 1.0e-12)
    nx = int(resolution)
    ny = max(int(round(nx * height / width)), 240)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    image = griddata(pts_sel, vals_sel, (grid_x, grid_y), method="linear")
    if np.any(~np.isfinite(image)):
        nearest = griddata(pts_sel, vals_sel, (grid_x, grid_y), method="nearest")
        image = np.where(np.isfinite(image), image, nearest)
    mask_img = _slice_footprint_mask(
        np.asarray(footprint_points, dtype=np.float64),
        np.asarray(footprint_tetrahedra, dtype=np.int64),
        axis=int(axis),
        center=float(center),
        extent=(xmin, xmax, ymin, ymax),
        shape=(ny, nx),
    )
    image = np.where(mask_img, image, np.nan)
    sigma = float(max(smooth_sigma, 0.0))
    if sigma > 0.0:
        valid = np.isfinite(image)
        if np.any(valid):
            filled = np.where(valid, image, 0.0)
            weights = valid.astype(np.float64)
            smooth_filled = gaussian_filter(filled, sigma=sigma, mode="nearest")
            smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
            image_out = np.full_like(smooth_filled, np.nan, dtype=np.float64)
            np.divide(smooth_filled, smooth_weights, out=image_out, where=smooth_weights > 1.0e-12)
            image = np.where(mask_img, image_out, np.nan)
    return {
        "image": image,
        "extent": (xmin, xmax, ymin, ymax),
        "xlabel": str(meta["xlabel"]),
        "ylabel": str(meta["ylabel"]),
        "axis_name": str(meta["axis_name"]),
    }


def _plot_single_slice(
    slice_data: dict[str, object],
    out_base: Path,
    *,
    title: str,
    scalar_label: str,
    cmap: str = "magma",
) -> None:
    image = np.asarray(slice_data["image"], dtype=np.float64)
    finite = np.isfinite(image)
    vmax = float(np.quantile(image[finite], 0.995)) if np.any(finite) else 1.0
    vmax = max(vmax, 1.0e-12)
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=180)
    im = ax.imshow(
        image,
        origin="lower",
        extent=tuple(slice_data["extent"]),
        cmap=str(cmap),
        vmin=0.0,
        vmax=float(vmax),
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlabel(str(slice_data["xlabel"]))
    ax.set_ylabel(str(slice_data["ylabel"]))
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(str(scalar_label))
    fig.tight_layout()
    _save_figure(fig, out_base)


def _compute_qfields(case, coords_final: np.ndarray, displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    elems = np.asarray(case.elems_scalar, dtype=np.int64)
    u_elem = np.asarray(displacement[elems], dtype=np.float64)
    ux = u_elem[:, :, 0]
    uy = u_elem[:, :, 1]
    uz = u_elem[:, :, 2]
    dphix = np.asarray(case.dphix, dtype=np.float64)
    dphiy = np.asarray(case.dphiy, dtype=np.float64)
    dphiz = np.asarray(case.dphiz, dtype=np.float64)
    e_xx = np.einsum("eqp,ep->eq", dphix, ux)
    e_yy = np.einsum("eqp,ep->eq", dphiy, uy)
    e_zz = np.einsum("eqp,ep->eq", dphiz, uz)
    g_xy = np.einsum("eqp,ep->eq", dphiy, ux) + np.einsum("eqp,ep->eq", dphix, uy)
    g_yz = np.einsum("eqp,ep->eq", dphiz, uy) + np.einsum("eqp,ep->eq", dphiy, uz)
    g_xz = np.einsum("eqp,ep->eq", dphiz, ux) + np.einsum("eqp,ep->eq", dphix, uz)
    strain = np.stack((e_xx, e_yy, e_zz, g_xy, g_yz, g_xz), axis=-1)
    dev = _dev_strain_norm(strain.reshape(-1, 6)).reshape(strain.shape[:2])
    xi = _quadrature_points_tetra_p2()
    hatp = np.asarray(evaluate_tetra_lagrange_basis(2, xi)[0], dtype=np.float64)
    qcoords = np.einsum("pq,epd->eqd", hatp, np.asarray(coords_final[elems], dtype=np.float64))
    return qcoords, dev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mesh-name", type=str, default=DEFAULT_MESH_NAME)
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--slug", type=str, default=DEFAULT_SLUG)
    parser.add_argument("--constraint-variant", type=str, default=DEFAULT_CONSTRAINT_VARIANT)
    parser.add_argument("--surface-subdivisions", type=int, default=4)
    args = parser.parse_args()

    state = np.load(args.state)
    result_payload = json.loads(args.result.read_text(encoding="utf-8"))
    case = load_case_hdf5(
        same_mesh_case_hdf5_path(
            str(args.mesh_name),
            int(args.degree),
            str(args.constraint_variant),
        )
    )

    coords_final = np.asarray(state["coords_final"], dtype=np.float64)
    displacement = np.asarray(state["displacement"], dtype=np.float64)
    surface_faces = np.asarray(state["surface_faces"], dtype=np.int64)
    nodal_disp_mag = np.linalg.norm(displacement, axis=1)
    qcoords, dev_strain = _compute_qfields(case, coords_final, displacement)
    qpoints = np.asarray(qcoords.reshape(-1, 3), dtype=np.float64)
    qvalues = np.asarray(dev_strain.reshape(-1), dtype=np.float64)
    y_min = float(np.min(qpoints[:, 1]))
    y_max = float(np.max(qpoints[:, 1]))
    y_center_up = y_min + 0.62 * (y_max - y_min)
    y_half = 0.02 * max(y_max - y_min, 1.0e-12)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    conv_base = args.out_dir / f"{args.slug}_convergence"
    disp_base = args.out_dir / f"{args.slug}_displacement"
    slice_x_base = args.out_dir / f"{args.slug}_deviatoric_strain_slice_x"
    slice_y_base = args.out_dir / f"{args.slug}_deviatoric_strain_slice_y"
    slice_z_base = args.out_dir / f"{args.slug}_deviatoric_strain_slice_z"

    _plot_convergence(
        result_payload,
        conv_base,
        title="P2(L1), lambda = 1.6, from-scratch Newton solve convergence",
    )
    _plot_surface_scalar_field(
        coords_final,
        surface_faces,
        nodal_disp_mag,
        disp_base,
        degree=int(args.degree),
        subdivisions=int(args.surface_subdivisions),
        title="P2(L1), lambda = 1.6, displacement magnitude on the deformed boundary",
        cbar_label=r"$||u||$",
        cmap_name="viridis",
    )

    slices = {
        "x": _interpolate_planar_slice(
            qpoints,
            qvalues,
            axis=0,
            center=float(0.5 * (np.min(qpoints[:, 0]) + np.max(qpoints[:, 0]))),
            half_thickness=float(0.025 * max(np.max(qpoints[:, 0]) - np.min(qpoints[:, 0]), 1.0e-12)),
            footprint_points=coords_final,
            footprint_tetrahedra=np.asarray(case.elems_scalar, dtype=np.int64),
        ),
        "y": _interpolate_planar_slice(
            qpoints,
            qvalues,
            axis=1,
            center=float(y_center_up),
            half_thickness=float(y_half),
            footprint_points=coords_final,
            footprint_tetrahedra=np.asarray(case.elems_scalar, dtype=np.int64),
        ),
        "z": _interpolate_planar_slice(
            qpoints,
            qvalues,
            axis=2,
            center=float(0.5 * (np.min(qpoints[:, 2]) + np.max(qpoints[:, 2]))),
            half_thickness=float(0.025 * max(np.max(qpoints[:, 2]) - np.min(qpoints[:, 2]), 1.0e-12)),
            footprint_points=coords_final,
            footprint_tetrahedra=np.asarray(case.elems_scalar, dtype=np.int64),
        ),
    }

    _plot_single_slice(
        slices["x"],
        slice_x_base,
        title="P2(L1), lambda = 1.6, deviatoric-strain x-slice",
        scalar_label="deviatoric strain",
    )
    _plot_single_slice(
        slices["y"],
        slice_y_base,
        title="P2(L1), lambda = 1.6, deviatoric-strain y-slice",
        scalar_label="deviatoric strain",
    )
    _plot_single_slice(
        slices["z"],
        slice_z_base,
        title="P2(L1), lambda = 1.6, deviatoric-strain z-slice",
        scalar_label="deviatoric strain",
    )

    summary = {
        "state": str(args.state),
        "result": str(args.result),
        "out_dir": str(args.out_dir),
        "mesh_name": str(args.mesh_name),
        "degree": int(args.degree),
        "lambda_target": float(result_payload.get("lambda_target", float("nan"))),
        "energy": float(result_payload.get("energy", float("nan"))),
        "u_max": float(result_payload.get("u_max", float("nan"))),
        "omega": float(result_payload.get("omega", float("nan"))),
        "nit": int(result_payload.get("nit", 0)),
        "solve_time": float(result_payload.get("solve_time", float("nan"))),
        "convergence_png": str(conv_base.with_suffix(".png")),
        "displacement_png": str(disp_base.with_suffix(".png")),
        "slice_x_png": str(slice_x_base.with_suffix(".png")),
        "slice_y_png": str(slice_y_base.with_suffix(".png")),
        "slice_z_png": str(slice_z_base.with_suffix(".png")),
    }
    (args.out_dir / f"{args.slug}_assets_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
