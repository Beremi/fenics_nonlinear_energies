#!/usr/bin/env python3
"""Generate a shared-color Plasticity3D y-slice comparison for the highest-resolution P1/P2/P4 runs."""

from __future__ import annotations

import argparse
import h5py
import json
from pathlib import Path

import numpy as np

from experiments.analysis.docs_assets import common as docs_common
from experiments.analysis.generate_plasticity3d_p4_l1_docs_assets import (
    _deviatoric_strain_norm_3d,
    _interpolate_planar_slice,
    _quadrature_points_tetra,
)
from src.problems.slope_stability_3d.support.mesh import same_mesh_case_hdf5_path
from src.problems.slope_stability_3d.support.simplex_lagrange import evaluate_tetra_lagrange_basis


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_lambda1p55_degree_mesh_energy_study" / "comparison_summary.json"
)
DEFAULT_DOCS_OUT_DIR = REPO_ROOT / "docs" / "assets" / "plasticity3d"
DEFAULT_STUDY_OUT_DIR = DEFAULT_SUMMARY.parent

FIGURE_STEM = "plasticity3d_lambda1p55_highest_mesh_y_slice_comparison"
SUMMARY_NAME = "plasticity3d_lambda1p55_highest_mesh_y_slice_comparison_summary.json"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _highest_rows_by_degree(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for degree_line in ("P1", "P2", "P4"):
        selected = [
            dict(row)
            for row in rows
            if isinstance(row, dict) and str(row.get("degree_line", "")) == degree_line
        ]
        if not selected:
            raise RuntimeError(f"No rows found for {degree_line}")
        selected.sort(key=lambda row: int(row.get("free_dofs", 0)))
        out.append(selected[-1])
    return out


def _build_y_slice(row: dict[str, object]) -> dict[str, object]:
    state_path = REPO_ROOT / str(row["state_npz"])
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
    y_center_up = bounds_min[1] + 0.62 * spans[1]
    half_thickness = 0.02 * spans[1]

    xi = _quadrature_points_tetra(degree)
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
            mask = np.abs(qcoords_chunk[:, :, 1] - float(y_center_up)) <= float(half_thickness)
            if np.any(mask):
                qcoords_blocks.append(np.asarray(qcoords_chunk[mask], dtype=np.float64))
                qdev_blocks.append(np.asarray(qdev_chunk[mask], dtype=np.float64))

    if not qcoords_blocks:
        raise RuntimeError(f"No y-slice quadrature samples retained for {row['degree_line']}({row['mesh_alias']})")
    qcoords = np.vstack(qcoords_blocks)
    qdev = np.concatenate(qdev_blocks)

    slice_data = _interpolate_planar_slice(
        qcoords,
        qdev,
        axis=1,
        center=float(y_center_up),
        half_thickness=float(half_thickness),
        footprint_points=coords_final,
        footprint_tetrahedra=elems,
        resolution=900,
        smooth_sigma=1.0,
    )
    slice_data["label"] = f"{row['degree_line']}({row['mesh_alias']})"
    slice_data["degree_line"] = str(row["degree_line"])
    slice_data["mesh_alias"] = str(row["mesh_alias"])
    slice_data["free_dofs"] = int(row["free_dofs"])
    slice_data["energy"] = float(row["energy"])
    slice_data["state_npz"] = str(row["state_npz"])
    slice_data["result_json"] = str(row["result_json"])
    slice_data["constraint_variant"] = str(constraint_variant)
    return slice_data


def _save_comparison(
    slice_panels: list[dict[str, object]],
    *,
    out_base: Path,
    xlim: tuple[float, float],
) -> dict[str, object]:
    plt = docs_common.configure_matplotlib()

    finite_arrays = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    for item in slice_panels:
        image = np.asarray(item["image"], dtype=np.float64)
        finite = np.isfinite(image)
        if np.any(finite):
            finite_arrays.append(image[finite])
        _, _, ymin, ymax = (float(v) for v in item["extent"])
        ymins.append(ymin)
        ymaxs.append(ymax)
    if not finite_arrays:
        raise RuntimeError("No finite slice data available for comparison figure")

    vmax = float(np.quantile(np.concatenate(finite_arrays), 0.995))
    vmax = max(vmax, 1.0e-12)
    zlim = (float(min(ymins)), float(max(ymaxs)))

    fig = plt.figure(figsize=(2.45 * docs_common.figure_width_in(), 0.84 * docs_common.figure_width_in()))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.06],
        left=0.055,
        right=0.985,
        bottom=0.16,
        top=0.96,
        wspace=0.18,
    )
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="white")
    mappable = None

    for idx, (ax, item) in enumerate(zip(axes, slice_panels, strict=True)):
        image = np.asarray(item["image"], dtype=np.float64)
        mappable = ax.imshow(
            image,
            origin="lower",
            extent=tuple(float(v) for v in item["extent"]),
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )
        mappable.set_rasterized(True)
        ax.set_title(str(item["degree_line"]))
        ax.set_xlabel("x")
        if idx == 0:
            ax.set_ylabel("z")
        else:
            ax.set_ylabel("")
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
        ax.set_ylim(*zlim)
        ax.grid(False)

    if mappable is not None:
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(r"$\|\varepsilon_{\mathrm{dev}}\|$")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

    return {
        "xlim": [float(xlim[0]), float(xlim[1])],
        "zlim": [float(zlim[0]), float(zlim[1])],
        "global_vmax": float(vmax),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--docs-out-dir", type=Path, default=DEFAULT_DOCS_OUT_DIR)
    parser.add_argument("--study-out-dir", type=Path, default=DEFAULT_STUDY_OUT_DIR)
    parser.add_argument("--xmin", type=float, default=-150.0)
    parser.add_argument("--xmax", type=float, default=-50.0)
    args = parser.parse_args()

    summary = _read_json(Path(args.summary_json).resolve())
    rows = [dict(row) for row in summary.get("rows", []) if isinstance(row, dict)]
    selected = _highest_rows_by_degree(rows)
    slices = [_build_y_slice(row) for row in selected]

    docs_out_base = Path(args.docs_out_dir).resolve() / FIGURE_STEM
    study_out_base = Path(args.study_out_dir).resolve() / FIGURE_STEM
    meta = _save_comparison(
        slices,
        out_base=docs_out_base,
        xlim=(float(args.xmin), float(args.xmax)),
    )
    _save_comparison(
        slices,
        out_base=study_out_base,
        xlim=(float(args.xmin), float(args.xmax)),
    )

    payload = {
        "summary_json": docs_common.repo_rel(Path(args.summary_json).resolve()),
        "rows": [
            {
                "degree_line": str(item["degree_line"]),
                "mesh_alias": str(item["mesh_alias"]),
                "free_dofs": int(item["free_dofs"]),
                "energy": float(item["energy"]),
                "constraint_variant": str(item["constraint_variant"]),
                "state_npz": str(item["state_npz"]),
                "result_json": str(item["result_json"]),
            }
            for item in slices
        ],
        "comparison": meta,
        "assets": {
            "docs_png": docs_common.repo_rel(docs_out_base.with_suffix(".png")),
            "docs_pdf": docs_common.repo_rel(docs_out_base.with_suffix(".pdf")),
            "study_png": docs_common.repo_rel(study_out_base.with_suffix(".png")),
            "study_pdf": docs_common.repo_rel(study_out_base.with_suffix(".pdf")),
        },
    }
    summary_path = Path(args.docs_out_dir).resolve() / SUMMARY_NAME
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
