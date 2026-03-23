#!/usr/bin/env python3
"""Generate P4 publication assets for the Mohr-Coulomb plasticity model card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import basix
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from src.problems.slope_stability.support.mesh import build_same_mesh_lagrange_case_data


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = REPO_ROOT / "artifacts" / "raw_results" / "docs_showcase" / "mc_plasticity_p4_l5" / "state.npz"
DEFAULT_RESULT = REPO_ROOT / "artifacts" / "raw_results" / "docs_showcase" / "mc_plasticity_p4_l5" / "output.json"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "assets" / "plasticity"


def _deviatoric_strain_norm_2d(E: np.ndarray) -> np.ndarray:
    iota = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    dev = np.diag([1.0, 1.0, 0.5]) - np.outer(iota, iota) / 2.0
    dev_e = dev @ E
    return np.sqrt(np.maximum(0.0, np.sum(E * dev_e, axis=0)))


def _reference_submesh(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
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


def _build_refined_plot_data(
    *,
    coords_ref: np.ndarray,
    coords_final: np.ndarray,
    displacement: np.ndarray,
    elems_scalar: np.ndarray,
    degree: int,
    subdivisions: int,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.triangle,
        int(degree),
        basix.LagrangeVariant.equispaced,
    )
    ref_points, tri_local = _reference_submesh(subdivisions)
    tab = element.tabulate(1, ref_points)
    phi_qp = np.asarray(tab[0, :, :, 0], dtype=np.float64)
    dphi_dxi_qp = np.asarray(tab[1, :, :, 0], dtype=np.float64)
    dphi_deta_qp = np.asarray(tab[2, :, :, 0], dtype=np.float64)

    n_elem = int(elems_scalar.shape[0])
    n_ref = int(ref_points.shape[0])

    coords_blocks: list[np.ndarray] = []
    dispmag_blocks: list[np.ndarray] = []
    dev_blocks: list[np.ndarray] = []
    tri_blocks: list[np.ndarray] = []
    point_offset = 0

    for start in range(0, n_elem, chunk_size):
        stop = min(start + chunk_size, n_elem)
        elem_nodes = np.asarray(elems_scalar[start:stop], dtype=np.int64)
        x_ref = np.asarray(coords_ref[elem_nodes], dtype=np.float64)
        x_def = np.asarray(coords_final[elem_nodes], dtype=np.float64)
        u_elem = np.asarray(displacement[elem_nodes], dtype=np.float64)

        plot_coords = np.einsum("qp,epd->eqd", phi_qp, x_def)
        plot_disp = np.einsum("qp,epd->eqd", phi_qp, u_elem)
        plot_disp_mag = np.linalg.norm(plot_disp, axis=-1)

        xr = x_ref[:, :, 0]
        yr = x_ref[:, :, 1]
        j11 = np.einsum("qp,ep->eq", dphi_dxi_qp, xr)
        j12 = np.einsum("qp,ep->eq", dphi_dxi_qp, yr)
        j21 = np.einsum("qp,ep->eq", dphi_deta_qp, xr)
        j22 = np.einsum("qp,ep->eq", dphi_deta_qp, yr)
        det = j11 * j22 - j12 * j21
        inv11 = j22 / det
        inv12 = -j12 / det
        inv21 = -j21 / det
        inv22 = j11 / det
        dphix = inv11[:, :, None] * dphi_dxi_qp[None, :, :] + inv12[:, :, None] * dphi_deta_qp[None, :, :]
        dphiy = inv21[:, :, None] * dphi_dxi_qp[None, :, :] + inv22[:, :, None] * dphi_deta_qp[None, :, :]

        ux = u_elem[:, :, 0]
        uy = u_elem[:, :, 1]
        eps_xx = np.einsum("eqp,ep->eq", dphix, ux)
        eps_yy = np.einsum("eqp,ep->eq", dphiy, uy)
        gamma_xy = np.einsum("eqp,ep->eq", dphiy, ux) + np.einsum("eqp,ep->eq", dphix, uy)
        strain = np.stack((eps_xx, eps_yy, gamma_xy), axis=0).reshape(3, -1)
        plot_dev = _deviatoric_strain_norm_2d(strain).reshape(stop - start, n_ref)

        tri_chunk = tri_local[None, :, :] + (
            point_offset + np.arange(stop - start, dtype=np.int32)[:, None, None] * n_ref
        )

        coords_blocks.append(plot_coords.reshape(-1, 2))
        dispmag_blocks.append(plot_disp_mag.reshape(-1))
        dev_blocks.append(plot_dev.reshape(-1))
        tri_blocks.append(tri_chunk.reshape(-1, 3))
        point_offset += (stop - start) * n_ref

    coords_plot = np.vstack(coords_blocks)
    dispmag = np.concatenate(dispmag_blocks)
    dev = np.concatenate(dev_blocks)
    tri_plot = np.vstack(tri_blocks)
    macro_tri = np.asarray(elems_scalar[:, :3], dtype=np.int32)
    return coords_plot, tri_plot, dispmag, dev, macro_tri


def _save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_displacement(
    *,
    coords_plot: np.ndarray,
    tri_plot: np.ndarray,
    values: np.ndarray,
    coords_final: np.ndarray,
    macro_tri: np.ndarray,
    out_base: Path,
) -> None:
    triangulation = mtri.Triangulation(coords_plot[:, 0], coords_plot[:, 1], triangles=tri_plot)
    macro = mtri.Triangulation(coords_final[:, 0], coords_final[:, 1], triangles=macro_tri)

    fig, ax = plt.subplots(figsize=(10.4, 5.8), dpi=180)
    pc = ax.tripcolor(triangulation, values, shading="gouraud", cmap="viridis")
    ax.triplot(macro, color="black", linewidth=0.12, alpha=0.12)
    ax.set_aspect("equal")
    ax.set_title("L5 P4 solution: displacement magnitude on the deformed mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(pc, ax=ax)
    cbar.set_label(r"$||u||$")
    fig.tight_layout()
    _save_figure(fig, out_base)


def _plot_deviatoric(
    *,
    coords_plot: np.ndarray,
    tri_plot: np.ndarray,
    values: np.ndarray,
    coords_final: np.ndarray,
    macro_tri: np.ndarray,
    out_base: Path,
) -> None:
    triangulation = mtri.Triangulation(coords_plot[:, 0], coords_plot[:, 1], triangles=tri_plot)
    macro = mtri.Triangulation(coords_final[:, 0], coords_final[:, 1], triangles=macro_tri)
    vmax = float(np.quantile(values, 0.995))

    fig, ax = plt.subplots(figsize=(10.4, 5.8), dpi=180)
    pc = ax.tripcolor(triangulation, values, shading="gouraud", cmap="magma", vmin=0.0, vmax=vmax)
    ax.triplot(macro, color="black", linewidth=0.12, alpha=0.10)
    ax.set_aspect("equal")
    ax.set_title("L5 P4 solution: deviatoric strain on the deformed mesh (99.5% clip)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(pc, ax=ax)
    cbar.set_label("deviatoric strain")
    fig.tight_layout()
    _save_figure(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--subdivisions", type=int, default=6)
    args = parser.parse_args()

    state = np.load(args.state)
    result_payload = json.loads(args.result.read_text(encoding="utf-8"))
    case = build_same_mesh_lagrange_case_data("ssr_homo_capture_p2_level5", degree=4)

    coords_ref = np.asarray(state["coords_ref"], dtype=np.float64)
    coords_final = np.asarray(state["coords_final"], dtype=np.float64)
    displacement = np.asarray(state["displacement"], dtype=np.float64)
    elems_scalar = np.asarray(case.elems_scalar, dtype=np.int32)

    coords_plot, tri_plot, dispmag, dev, macro_tri = _build_refined_plot_data(
        coords_ref=coords_ref,
        coords_final=coords_final,
        displacement=displacement,
        elems_scalar=elems_scalar,
        degree=4,
        subdivisions=int(args.subdivisions),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    disp_base = args.out_dir / "mc_plasticity_p4_l5_displacement"
    dev_base = args.out_dir / "mc_plasticity_p4_l5_deviatoric_strain_robust"
    _plot_displacement(
        coords_plot=coords_plot,
        tri_plot=tri_plot,
        values=dispmag,
        coords_final=coords_final,
        macro_tri=macro_tri,
        out_base=disp_base,
    )
    _plot_deviatoric(
        coords_plot=coords_plot,
        tri_plot=tri_plot,
        values=dev,
        coords_final=coords_final,
        macro_tri=macro_tri,
        out_base=dev_base,
    )

    summary = {
        "state": str(args.state),
        "result": str(args.result),
        "out_dir": str(args.out_dir),
        "subdivisions": int(args.subdivisions),
        "level": 5,
        "degree": 4,
        "nodes": int(case.nodes.shape[0]),
        "elements": int(case.elems_scalar.shape[0]),
        "free_dofs": int(case.freedofs.size),
        "quad_points_per_element": int(case.quad_weight.shape[1]),
        "lambda_target": float(result_payload["case"]["lambda_target"]),
        "energy": float(result_payload["result"]["steps"][0]["energy"]),
        "omega": float(result_payload["result"]["steps"][0]["omega"]),
        "u_max": float(result_payload["result"]["steps"][0]["u_max"]),
        "displacement_png": str(disp_base.with_suffix(".png")),
        "deviatoric_png": str(dev_base.with_suffix(".png")),
    }
    (args.out_dir / "mc_plasticity_p4_l5_assets_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
