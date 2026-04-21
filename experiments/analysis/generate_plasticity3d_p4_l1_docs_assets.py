#!/usr/bin/env python3
"""Generate Plasticity3D docs assets for the `hetero_ssr_L1` `P4` showcase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from src.problems.slope_stability_3d.support.mesh import (
    PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM,
    load_case_hdf5,
    same_mesh_case_hdf5_path,
)
from src.problems.slope_stability_3d.support.simplex_lagrange import (
    evaluate_tetra_lagrange_basis,
    evaluate_triangle_lagrange_basis,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = (
    REPO_ROOT / "artifacts" / "raw_results" / "docs_showcase" / "plasticity3d_p4_l1" / "state.npz"
)
DEFAULT_RESULT = (
    REPO_ROOT / "artifacts" / "raw_results" / "docs_showcase" / "plasticity3d_p4_l1" / "output.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "assets" / "plasticity3d"
DEFAULT_MESH_NAME = "hetero_ssr_L1"
DEFAULT_DEGREE = 4
DEFAULT_CONSTRAINT_VARIANT = PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM


def _slug_for_case(mesh_name: str, degree: int) -> str:
    mesh_label = str(mesh_name).replace("hetero_ssr_", "").lower()
    return f"plasticity3d_p{int(degree)}_{mesh_label}"


def _reference_triangle_submesh(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    if subdivisions < 1:
        raise ValueError("subdivisions must be >= 1")
    pts: list[tuple[float, float]] = []
    index: dict[tuple[int, int], int] = {}
    for i in range(subdivisions + 1):
        for j in range(subdivisions + 1 - i):
            index[(i, j)] = len(pts)
            pts.append((i / subdivisions, j / subdivisions))

    tri: list[tuple[int, int, int]] = []
    for i in range(subdivisions):
        for j in range(subdivisions - i):
            a = index[(i, j)]
            b = index[(i + 1, j)]
            c = index[(i, j + 1)]
            tri.append((a, b, c))
            if i + j <= subdivisions - 2:
                d = index[(i + 1, j + 1)]
                tri.append((b, d, c))
    return np.asarray(pts, dtype=np.float64), np.asarray(tri, dtype=np.int32)


def _quadrature_points_tetra(degree: int) -> np.ndarray:
    degree = int(degree)
    if degree == 1:
        return np.array([[0.25], [0.25], [0.25]], dtype=np.float64)
    if degree == 2:
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
    if degree == 4:
        return np.array(
            [
                [
                    0.3561913862225449,
                    0.2146028712591517,
                    0.2146028712591517,
                    0.2146028712591517,
                    0.8779781243961660,
                    0.0406739585346113,
                    0.0406739585346113,
                    0.0406739585346113,
                    0.0329863295731731,
                    0.3223378901422757,
                    0.3223378901422757,
                    0.3223378901422757,
                    0.2696723314583159,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.6030056647916491,
                ],
                [
                    0.2146028712591517,
                    0.2146028712591517,
                    0.2146028712591517,
                    0.3561913862225449,
                    0.0406739585346113,
                    0.0406739585346113,
                    0.0406739585346113,
                    0.8779781243961660,
                    0.3223378901422757,
                    0.3223378901422757,
                    0.3223378901422757,
                    0.0329863295731731,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.2696723314583159,
                ],
                [
                    0.2146028712591517,
                    0.2146028712591517,
                    0.3561913862225449,
                    0.2146028712591517,
                    0.0406739585346113,
                    0.0406739585346113,
                    0.8779781243961660,
                    0.0406739585346113,
                    0.3223378901422757,
                    0.3223378901422757,
                    0.0329863295731731,
                    0.3223378901422757,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.6030056647916491,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.2696723314583159,
                    0.6030056647916491,
                    0.0636610018750175,
                ],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported tetra degree {degree!r}")


def _save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def _set_equal_3d_axes(ax, xyz: np.ndarray) -> None:
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def _apply_showcase_camera(ax, xyz: np.ndarray) -> None:
    _set_equal_3d_axes(ax, xyz)
    ax.view_init(elev=22.0, azim=56.0)


def _deviatoric_strain_norm_3d(strain6: np.ndarray) -> np.ndarray:
    dev_metric = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    ) / 3.0
    arr = np.asarray(strain6, dtype=np.float64)
    proj = arr @ dev_metric.T
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


def _surface_plot_arrays(
    coords_final: np.ndarray,
    surface_faces: np.ndarray,
    nodal_values: np.ndarray,
    *,
    degree: int,
    subdivisions: int,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_pts, tri_local = _reference_triangle_submesh(subdivisions)
    hatp = evaluate_triangle_lagrange_basis(int(degree), ref_pts.T)[0]
    n_face = int(surface_faces.shape[0])
    n_ref = int(ref_pts.shape[0])
    coords_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []
    tri_blocks: list[np.ndarray] = []
    point_offset = 0
    for start in range(0, n_face, chunk_size):
        stop = min(start + chunk_size, n_face)
        face_nodes = np.asarray(surface_faces[start:stop], dtype=np.int64)
        face_coords = np.asarray(coords_final[face_nodes], dtype=np.float64)
        face_values = np.asarray(nodal_values[face_nodes], dtype=np.float64)
        coords_chunk = np.einsum("pq,fpd->fqd", hatp, face_coords)
        values_chunk = np.einsum("pq,fp->fq", hatp, face_values)
        tri_chunk = tri_local[None, :, :] + (
            point_offset + np.arange(stop - start, dtype=np.int32)[:, None, None] * n_ref
        )
        coords_blocks.append(coords_chunk.reshape(-1, 3))
        value_blocks.append(values_chunk.reshape(-1))
        tri_blocks.append(tri_chunk.reshape(-1, 3))
        point_offset += (stop - start) * n_ref
    return np.vstack(coords_blocks), np.vstack(tri_blocks), np.concatenate(value_blocks)


def _plot_surface_scalar_field(
    coords_final: np.ndarray,
    surface_faces: np.ndarray,
    nodal_values: np.ndarray,
    out_base: Path,
    *,
    degree: int,
    subdivisions: int,
    title: str,
    cbar_label: str,
    cmap_name: str,
    upper_quantile: float | None = None,
) -> None:
    coords_plot, tri_plot, values = _surface_plot_arrays(
        coords_final,
        surface_faces,
        nodal_values,
        degree=degree,
        subdivisions=subdivisions,
    )
    tri_vals = np.mean(values[np.asarray(tri_plot, dtype=np.int64)], axis=1)
    if upper_quantile is not None:
        vmax = float(np.quantile(tri_vals, float(upper_quantile)))
        vmin = 0.0
    else:
        vmin = float(np.min(tri_vals))
        vmax = float(np.max(tri_vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Surface scalar field contains non-finite values")
    if abs(vmax - vmin) < 1.0e-14:
        vmax = vmin + 1.0e-14
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]
    facecolors = cmap(norm(tri_vals))

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(9.0, 7.4), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(
        coords_plot[np.asarray(tri_plot, dtype=np.int64)],
        facecolors=facecolors,
        linewidths=0.0,
        edgecolors="none",
        alpha=1.0,
    )
    poly.set_rasterized(True)
    ax.add_collection3d(poly)
    _apply_showcase_camera(ax, coords_plot)
    ax.set_axis_off()
    ax.set_title(title, pad=8.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_name)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label=cbar_label)
    fig.tight_layout()
    _save_figure(fig, out_base)


def _compute_nodal_deviatoric_strain(
    *,
    coords_final: np.ndarray,
    displacement: np.ndarray,
    case,
    degree: int,
    chunk_size: int = 256,
) -> np.ndarray:
    degree = int(degree)
    xi = _quadrature_points_tetra(degree)
    hatp = np.asarray(evaluate_tetra_lagrange_basis(degree, xi)[0], dtype=np.float64)
    basis_weight = np.abs(hatp.T)[None, :, :]

    elems = np.asarray(case.elems_scalar, dtype=np.int64)
    n_nodes = int(coords_final.shape[0])
    accum = np.zeros(n_nodes, dtype=np.float64)
    mass = np.zeros(n_nodes, dtype=np.float64)

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
        dev = _deviatoric_strain_norm_3d(eps6).reshape((stop - start, -1))
        quad_weight = np.asarray(case.quad_weight[start:stop], dtype=np.float64)[:, :, None]
        local_weight = basis_weight * quad_weight
        local_value = np.einsum("eqp,eq->ep", local_weight, dev)
        local_mass = np.sum(local_weight, axis=1)
        np.add.at(accum, elem_nodes.reshape(-1), local_value.reshape(-1))
        np.add.at(mass, elem_nodes.reshape(-1), local_mass.reshape(-1))

    nodal = np.zeros_like(accum)
    mask = mass > 0.0
    nodal[mask] = accum[mask] / mass[mask]
    return nodal


def _compute_qcoords_and_deviatoric_strain(
    *,
    coords_final: np.ndarray,
    displacement: np.ndarray,
    case,
    degree: int,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    degree = int(degree)
    xi = _quadrature_points_tetra(degree)
    hatp = np.asarray(evaluate_tetra_lagrange_basis(degree, xi)[0], dtype=np.float64)
    elems = np.asarray(case.elems_scalar, dtype=np.int64)
    qcoords_blocks: list[np.ndarray] = []
    qdev_blocks: list[np.ndarray] = []
    for start in range(0, elems.shape[0], chunk_size):
        stop = min(start + chunk_size, elems.shape[0])
        elem_nodes = elems[start:stop]
        x_def = np.asarray(coords_final[elem_nodes], dtype=np.float64)
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
        qcoords = np.einsum("pq,epd->eqd", hatp, x_def).reshape((-1, 3))
        qcoords_blocks.append(np.asarray(qcoords, dtype=np.float64))
        qdev_blocks.append(np.asarray(_deviatoric_strain_norm_3d(eps6), dtype=np.float64))
    return np.vstack(qcoords_blocks), np.concatenate(qdev_blocks)


def _compute_slice_datasets(
    *,
    coords_final: np.ndarray,
    qcoords: np.ndarray,
    qdev: np.ndarray,
    case,
) -> list[dict[str, object]]:
    bounds_min = np.min(qcoords, axis=0)
    bounds_max = np.max(qcoords, axis=0)
    centers = 0.5 * (bounds_min + bounds_max)
    spans = np.maximum(bounds_max - bounds_min, 1.0e-12)
    y_center_up = bounds_min[1] + 0.62 * spans[1]
    slices = [
        {
            "axis": 0,
            "center": float(centers[0]),
            "half_thickness": float(0.025 * spans[0]),
        },
        {
            "axis": 1,
            "center": float(y_center_up),
            "half_thickness": float(0.02 * spans[1]),
        },
        {
            "axis": 2,
            "center": float(centers[2]),
            "half_thickness": float(0.025 * spans[2]),
        },
    ]
    out: list[dict[str, object]] = []
    for item in slices:
        axis = int(item["axis"])
        slice_image = _interpolate_planar_slice(
            qcoords,
            qdev,
            axis=axis,
            center=float(item["center"]),
            half_thickness=float(item["half_thickness"]),
            footprint_points=np.asarray(coords_final, dtype=np.float64),
            footprint_tetrahedra=np.asarray(case.elems_scalar, dtype=np.int64),
            resolution=900,
            smooth_sigma=1.0,
        )
        out.append(slice_image)
    return out


def _plot_deviatoric_slices(
    slice_data: list[dict[str, object]],
    out_base: Path,
    *,
    title: str,
) -> None:
    finite_arrays = []
    for item in slice_data:
        image = np.asarray(item["image"], dtype=np.float64)
        finite = np.isfinite(image)
        if np.any(finite):
            finite_arrays.append(image[finite])
    vmax = 1.0
    if finite_arrays:
        vmax = float(np.quantile(np.concatenate(finite_arrays), 0.995))
        vmax = max(vmax, 1.0e-12)
    fig = plt.figure(figsize=(14.2, 4.9), dpi=180)
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.06],
        left=0.05,
        right=0.98,
        bottom=0.13,
        top=0.84,
        wspace=0.30,
    )
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    cmap = "magma"
    mappable = None
    axis_titles = ("central x-slice", "upper y-slice", "central z-slice")
    for ax, item, axis_title in zip(axes, slice_data, axis_titles, strict=False):
        image = np.asarray(item["image"], dtype=np.float64)
        mappable = ax.imshow(
            image,
            origin="lower",
            extent=tuple(item["extent"]),
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )
        mappable.set_rasterized(True)
        ax.set_title(axis_title)
        ax.set_xlabel(str(item["xlabel"]))
        ax.set_ylabel(str(item["ylabel"]))
    if mappable is not None:
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label("deviatoric strain (99.5% clip)")
    fig.suptitle(title, y=0.94)
    _save_figure(fig, out_base)


def _plot_convergence(
    result_payload: dict[str, object],
    out_base: Path,
    *,
    title: str,
) -> None:
    history = list(result_payload.get("history", []))
    nit = int(result_payload.get("nit", 0))
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), dpi=180)
    if history:
        its = np.asarray([int(item.get("it", idx + 1)) for idx, item in enumerate(history)], dtype=np.int64)
        energy = np.asarray([float(item.get("energy", np.nan)) for item in history], dtype=np.float64)
        grad = np.asarray([float(item.get("grad_norm", np.nan)) for item in history], dtype=np.float64)
        step = np.asarray([float(item.get("step_norm", np.nan)) for item in history], dtype=np.float64)
        ksp = np.asarray([float(item.get("ksp_its", np.nan)) for item in history], dtype=np.float64)
        ls = np.asarray([float(item.get("ls_evals", np.nan)) for item in history], dtype=np.float64)

        axes[0, 0].plot(its, energy, marker="o", color="#1f77b4")
        axes[0, 0].set_title("Energy")
        axes[0, 0].set_xlabel("Newton iteration")

        axes[0, 1].semilogy(its, np.maximum(grad, 1.0e-16), marker="o", color="#d62728", label="||g||")
        axes[0, 1].semilogy(its, np.maximum(step, 1.0e-16), marker="s", color="#2ca02c", label="||dx||")
        axes[0, 1].set_title("Gradient / step norms")
        axes[0, 1].set_xlabel("Newton iteration")
        axes[0, 1].legend(loc="best")

        axes[1, 0].bar(its, ksp, color="#9467bd")
        axes[1, 0].set_title("Linear iterations")
        axes[1, 0].set_xlabel("Newton iteration")

        axes[1, 1].bar(its, ls, color="#ff7f0e")
        axes[1, 1].set_title("Line-search evaluations")
        axes[1, 1].set_xlabel("Newton iteration")
    else:
        for ax in axes.ravel():
            ax.axis("off")
        axes[0, 0].axis("on")
        axes[0, 0].text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"nit = {nit}",
                    f"status = {result_payload.get('status', '')}",
                    f"message = {result_payload.get('message', '')}",
                    f"energy = {float(result_payload.get('energy', float('nan'))):.6e}",
                    f"final_grad_norm = {float(result_payload.get('final_grad_norm', float('nan'))):.6e}",
                ]
            ),
            ha="left",
            va="top",
            transform=axes[0, 0].transAxes,
            family="monospace",
        )
        axes[0, 0].set_title("No saved Newton history")
    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    _save_figure(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mesh-name", type=str, default=DEFAULT_MESH_NAME)
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--constraint-variant", type=str, default=DEFAULT_CONSTRAINT_VARIANT)
    parser.add_argument("--surface-subdivisions", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
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

    coords_ref = np.asarray(state["coords_ref"], dtype=np.float64)
    coords_final = np.asarray(state["coords_final"], dtype=np.float64)
    displacement = np.asarray(state["displacement"], dtype=np.float64)
    surface_faces = np.asarray(state["surface_faces"], dtype=np.int64)
    nodal_disp_mag = np.linalg.norm(displacement, axis=1)
    lambda_target = float(result_payload.get("lambda_target", float(state["lambda_target"])))
    label = f"P{int(args.degree)}({str(args.mesh_name).replace('hetero_ssr_', '')})"
    slug = _slug_for_case(str(args.mesh_name), int(args.degree))
    nodal_dev = _compute_nodal_deviatoric_strain(
        coords_final=coords_final,
        displacement=displacement,
        case=case,
        degree=int(args.degree),
        chunk_size=int(args.chunk_size),
    )
    qcoords, qdev = _compute_qcoords_and_deviatoric_strain(
        coords_final=coords_final,
        displacement=displacement,
        case=case,
        degree=int(args.degree),
        chunk_size=int(args.chunk_size),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    conv_base = args.out_dir / f"{slug}_convergence"
    disp_base = args.out_dir / f"{slug}_displacement"
    dev_base = args.out_dir / f"{slug}_deviatoric_strain"
    slice_base = args.out_dir / f"{slug}_deviatoric_strain_slices"

    _plot_convergence(
        result_payload,
        conv_base,
        title=f"{label} solve convergence summary",
    )
    _plot_surface_scalar_field(
        coords_final,
        surface_faces,
        nodal_disp_mag,
        disp_base,
        degree=int(args.degree),
        subdivisions=int(args.surface_subdivisions),
        title=f"{label} displacement magnitude on the deformed boundary",
        cbar_label=r"$||u||$",
        cmap_name="viridis",
    )
    _plot_surface_scalar_field(
        coords_final,
        surface_faces,
        nodal_dev,
        dev_base,
        degree=int(args.degree),
        subdivisions=int(args.surface_subdivisions),
        title=f"{label} surface-projected deviatoric-strain magnitude",
        cbar_label="deviatoric strain",
        cmap_name="magma",
        upper_quantile=0.995,
    )
    slice_data = _compute_slice_datasets(
        coords_final=coords_final,
        qcoords=qcoords,
        qdev=qdev,
        case=case,
    )
    _plot_deviatoric_slices(
        slice_data,
        slice_base,
        title=f"{label} deviatoric-strain slab slices on the deformed configuration",
    )

    summary = {
        "state": str(args.state),
        "result": str(args.result),
        "out_dir": str(args.out_dir),
        "mesh_name": str(args.mesh_name),
        "degree": int(args.degree),
        "lambda_target": float(lambda_target),
        "energy": float(result_payload.get("energy", float("nan"))),
        "u_max": float(result_payload.get("u_max", float("nan"))),
        "omega": float(result_payload.get("omega", float("nan"))),
        "nit": int(result_payload.get("nit", 0)),
        "solve_time": float(result_payload.get("solve_time", float("nan"))),
        "convergence_png": str(conv_base.with_suffix(".png")),
        "displacement_png": str(disp_base.with_suffix(".png")),
        "deviatoric_strain_png": str(dev_base.with_suffix(".png")),
        "slice_png": str(slice_base.with_suffix(".png")),
    }
    (args.out_dir / f"{slug}_assets_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
