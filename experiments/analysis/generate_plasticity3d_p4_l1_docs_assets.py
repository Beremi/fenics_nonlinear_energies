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

from src.problems.slope_stability_3d.jax.jax_energy_3d import vmapped_mc_stress_density_3d
from src.problems.slope_stability_3d.support.mesh import load_case_hdf5, same_mesh_case_hdf5_path
from src.problems.slope_stability_3d.support.reduction import davis_b_reduction_qp
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
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
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
) -> None:
    coords_plot, tri_plot, values = _surface_plot_arrays(
        coords_final,
        surface_faces,
        nodal_values,
        degree=degree,
        subdivisions=subdivisions,
    )
    tri_vals = np.mean(values[np.asarray(tri_plot, dtype=np.int64)], axis=1)
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
    ax.add_collection3d(poly)
    _set_equal_3d_axes(ax, coords_plot)
    ax.view_init(elev=22.0, azim=-56.0)
    ax.set_axis_off()
    ax.set_title(title, pad=8.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_name)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label=cbar_label)
    fig.tight_layout()
    _save_figure(fig, out_base)


def _deviatoric_stress_norm_3d(stress6: np.ndarray) -> np.ndarray:
    stress6 = np.asarray(stress6, dtype=np.float64)
    out = np.empty(stress6.shape[0], dtype=np.float64)
    mean = np.mean(stress6[:, :3], axis=1)
    sxx = stress6[:, 0] - mean
    syy = stress6[:, 1] - mean
    szz = stress6[:, 2] - mean
    sxy = stress6[:, 3]
    syz = stress6[:, 4]
    sxz = stress6[:, 5]
    out[:] = np.sqrt(
        np.maximum(
            0.0,
            sxx * sxx + syy * syy + szz * szz + 2.0 * (sxy * sxy + syz * syz + sxz * sxz),
        )
    )
    return out


def _compute_nodal_deviatoric_stress(
    *,
    coords_final: np.ndarray,
    displacement: np.ndarray,
    case,
    degree: int,
    lambda_target: float,
    chunk_size: int = 256,
) -> np.ndarray:
    c_bar_q, sin_phi_q = davis_b_reduction_qp(case.c0_q, case.phi_q, case.psi_q, lambda_target)
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

        stress6 = np.asarray(
            vmapped_mc_stress_density_3d(
                eps6,
                np.asarray(c_bar_q[start:stop], dtype=np.float64).reshape(-1),
                np.asarray(sin_phi_q[start:stop], dtype=np.float64).reshape(-1),
                np.asarray(case.shear_q[start:stop], dtype=np.float64).reshape(-1),
                np.asarray(case.bulk_q[start:stop], dtype=np.float64).reshape(-1),
                np.asarray(case.lame_q[start:stop], dtype=np.float64).reshape(-1),
            ),
            dtype=np.float64,
        )
        dev = _deviatoric_stress_norm_3d(stress6).reshape((stop - start, -1))
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


def _compute_slice_datasets(
    *,
    coords_final: np.ndarray,
    displacement: np.ndarray,
    case,
    degree: int,
    lambda_target: float,
    chunk_size: int = 256,
) -> list[dict[str, object]]:
    c_bar_q, sin_phi_q = davis_b_reduction_qp(case.c0_q, case.phi_q, case.psi_q, lambda_target)
    degree = int(degree)
    xi = _quadrature_points_tetra(degree)
    hatp = evaluate_tetra_lagrange_basis(degree, xi)[0]

    bounds_min = np.min(coords_final, axis=0)
    bounds_max = np.max(coords_final, axis=0)
    centers = 0.5 * (bounds_min + bounds_max)
    thickness = 0.04 * np.maximum(bounds_max - bounds_min, 1.0)
    datasets = [
        {
            "axis": 0,
            "center": float(centers[0]),
            "thickness": float(thickness[0]),
            "label": f"x ~= {centers[0]:.2f}",
            "xlabel": "y",
            "ylabel": "z",
            "project": (1, 2),
            "x": [],
            "y": [],
            "v": [],
        },
        {
            "axis": 1,
            "center": float(centers[1]),
            "thickness": float(thickness[1]),
            "label": f"y ~= {centers[1]:.2f}",
            "xlabel": "x",
            "ylabel": "z",
            "project": (0, 2),
            "x": [],
            "y": [],
            "v": [],
        },
        {
            "axis": 2,
            "center": float(centers[2]),
            "thickness": float(thickness[2]),
            "label": f"z ~= {centers[2]:.2f}",
            "xlabel": "x",
            "ylabel": "y",
            "project": (0, 1),
            "x": [],
            "y": [],
            "v": [],
        },
    ]

    elems = np.asarray(case.elems_scalar, dtype=np.int64)
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

        stress6 = np.asarray(
            vmapped_mc_stress_density_3d(
                eps6,
                np.asarray(c_bar_q[start:stop], dtype=np.float64).reshape(-1),
                np.asarray(sin_phi_q[start:stop], dtype=np.float64).reshape(-1),
                np.asarray(case.shear_q[start:stop], dtype=np.float64).reshape(-1),
                np.asarray(case.bulk_q[start:stop], dtype=np.float64).reshape(-1),
                np.asarray(case.lame_q[start:stop], dtype=np.float64).reshape(-1),
            ),
            dtype=np.float64,
        )
        dev = _deviatoric_stress_norm_3d(stress6)
        qcoords = np.einsum("pq,epd->eqd", hatp, x_def).reshape((-1, 3))

        for data in datasets:
            axis = int(data["axis"])
            center = float(data["center"])
            half = 0.5 * float(data["thickness"])
            mask = np.abs(qcoords[:, axis] - center) <= half
            if not np.any(mask):
                continue
            p0, p1 = data["project"]
            data["x"].append(np.asarray(qcoords[mask, p0], dtype=np.float64))
            data["y"].append(np.asarray(qcoords[mask, p1], dtype=np.float64))
            data["v"].append(np.asarray(dev[mask], dtype=np.float64))

    out: list[dict[str, object]] = []
    for data in datasets:
        out.append(
            {
                **{k: v for k, v in data.items() if k not in {"x", "y", "v"}},
                "x": np.concatenate(data["x"]) if data["x"] else np.empty(0, dtype=np.float64),
                "y": np.concatenate(data["y"]) if data["y"] else np.empty(0, dtype=np.float64),
                "v": np.concatenate(data["v"]) if data["v"] else np.empty(0, dtype=np.float64),
            }
        )
    return out


def _plot_deviatoric_slices(
    slice_data: list[dict[str, object]],
    out_base: Path,
    *,
    title: str,
) -> None:
    nonempty = [item for item in slice_data if int(item["v"].size) > 0]
    vmax = 1.0
    if nonempty:
        vmax = float(np.quantile(np.concatenate([item["v"] for item in nonempty]), 0.995))
        vmax = max(vmax, 1.0e-12)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=180)
    cmap = "magma"
    sc = None
    for ax, item in zip(axes, slice_data, strict=False):
        x = np.asarray(item["x"], dtype=np.float64)
        y = np.asarray(item["y"], dtype=np.float64)
        v = np.asarray(item["v"], dtype=np.float64)
        if v.size == 0:
            ax.text(0.5, 0.5, "No points in slice", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(str(item["label"]))
            ax.set_xlabel(str(item["xlabel"]))
            ax.set_ylabel(str(item["ylabel"]))
            continue
        sc = ax.scatter(x, y, c=v, s=3.0, cmap=cmap, vmin=0.0, vmax=vmax, linewidths=0.0)
        ax.set_aspect("equal")
        ax.set_title(str(item["label"]))
        ax.set_xlabel(str(item["xlabel"]))
        ax.set_ylabel(str(item["ylabel"]))
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label(r"$||s_{\mathrm{dev}}||$ (99.5% clip)")
    fig.suptitle(title, y=0.98)
    fig.tight_layout()
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
    parser.add_argument("--surface-subdivisions", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    state = np.load(args.state)
    result_payload = json.loads(args.result.read_text(encoding="utf-8"))
    case = load_case_hdf5(same_mesh_case_hdf5_path(str(args.mesh_name), int(args.degree)))

    coords_ref = np.asarray(state["coords_ref"], dtype=np.float64)
    coords_final = np.asarray(state["coords_final"], dtype=np.float64)
    displacement = np.asarray(state["displacement"], dtype=np.float64)
    surface_faces = np.asarray(state["surface_faces"], dtype=np.int64)
    nodal_disp_mag = np.linalg.norm(displacement, axis=1)
    lambda_target = float(result_payload.get("lambda_target", float(state["lambda_target"])))
    label = f"P{int(args.degree)}({str(args.mesh_name).replace('hetero_ssr_', '')})"
    slug = f"plasticity3d_{str(args.mesh_name).lower()}_p{int(args.degree)}"
    nodal_dev = _compute_nodal_deviatoric_stress(
        coords_final=coords_final,
        displacement=displacement,
        case=case,
        degree=int(args.degree),
        lambda_target=float(lambda_target),
        chunk_size=int(args.chunk_size),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    conv_base = args.out_dir / f"{slug}_convergence"
    disp_base = args.out_dir / f"{slug}_displacement"
    dev_base = args.out_dir / f"{slug}_deviatoric_stress"
    slice_base = args.out_dir / f"{slug}_deviatoric_stress_slices"

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
        title=f"{label} surface-projected deviatoric-stress magnitude",
        cbar_label=r"$||s_{\mathrm{dev}}||$",
        cmap_name="magma",
    )
    slice_data = _compute_slice_datasets(
        coords_final=coords_final,
        displacement=displacement,
        case=case,
        degree=int(args.degree),
        lambda_target=float(lambda_target),
        chunk_size=int(args.chunk_size),
    )
    _plot_deviatoric_slices(
        slice_data,
        slice_base,
        title=f"{label} deviatoric-stress slab slices on the deformed configuration",
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
        "deviatoric_stress_png": str(dev_base.with_suffix(".png")),
        "slice_png": str(slice_base.with_suffix(".png")),
    }
    (args.out_dir / f"{slug}_assets_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
