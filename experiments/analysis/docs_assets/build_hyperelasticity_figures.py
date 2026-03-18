#!/usr/bin/env python3
"""Generate HyperElasticity overview PDF figures."""

from __future__ import annotations

import argparse

import numpy as np

from common import (
    family_data,
    family_pdf,
    implementation_style,
    configure_matplotlib,
    ideal_mesh_scaling,
    ideal_strong_scaling,
    load_npz,
    read_csv_rows,
    record_provenance,
    repo_rel,
    save_pdf,
    springer_figure_size,
)


FAMILY = "hyperelasticity"
C1 = 38461538.461538464
D1 = 83333333.33333333


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
    X = coords_ref[np.asarray(tetrahedra, dtype=np.int32)]
    x = coords_final[np.asarray(tetrahedra, dtype=np.int32)]
    dX = np.transpose(X[:, 1:, :] - X[:, :1, :], (0, 2, 1))
    dx = np.transpose(x[:, 1:, :] - x[:, :1, :], (0, 2, 1))
    F = dx @ np.linalg.inv(dX)
    I1 = np.sum(F * F, axis=(1, 2))
    detF = np.abs(np.linalg.det(F))
    detF = np.maximum(detF, 1e-12)
    return C1 * (I1 - 3.0 - 2.0 * np.log(detF)) + D1 * (detF - 1.0) ** 2


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
            edge = np.vstack([ordered[idx], ordered[(idx + 1) % 4]])
            edges.append(edge)
    return edges


def _project_view(coords: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beam_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    camera_dir = np.array([0.18, 1.0, 0.38], dtype=np.float64)
    camera_dir /= np.linalg.norm(camera_dir)
    forward = -camera_dir
    horizontal = beam_axis - np.dot(beam_axis, forward) * forward
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(forward, horizontal)
    vertical /= np.linalg.norm(vertical)
    coords = np.asarray(coords, dtype=np.float64)
    span = max(float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))), 1e-12)
    camera = center + 1.6 * span * camera_dir
    rel = coords - camera[None, :]
    depth = rel @ forward
    depth = np.maximum(depth, 1e-9)
    focal = 1.0
    u = focal * (rel @ horizontal) / depth
    v = focal * (rel @ vertical) / depth
    return u, v, depth


def _state_figure():
    plt = configure_matplotlib()
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import pyvista as pv

    data = load_npz(family_data(f"{FAMILY}/sample_state.npz"))
    coords_ref = data["coords_ref"]
    coords_final = data["coords_final"]
    tetrahedra = data["tetrahedra"]
    boundary_faces, boundary_owners = _boundary_faces_with_owner(tetrahedra)
    prism_edges = _beam_prism_edges(coords_ref, coords_final)
    face_values = _tet_energy_density(coords_ref, coords_final, tetrahedra)[boundary_owners]
    mins = coords_final.min(axis=0)
    maxs = coords_final.max(axis=0)
    center = 0.5 * (mins + maxs)

    pv.OFF_SCREEN = True
    faces = np.empty((boundary_faces.shape[0], 4), dtype=np.int32)
    faces[:, 0] = 3
    faces[:, 1:] = boundary_faces
    surface = pv.PolyData(coords_final, faces.ravel())
    surface.cell_data["energy_density"] = face_values

    edge_points: list[np.ndarray] = []
    edge_lines: list[int] = []
    point_offset = 0
    for edge in prism_edges:
        edge = np.asarray(edge, dtype=np.float64)
        edge_points.append(edge)
        edge_lines.extend([edge.shape[0], *range(point_offset, point_offset + edge.shape[0])])
        point_offset += edge.shape[0]
    edge_mesh = pv.PolyData(np.vstack(edge_points), lines=np.asarray(edge_lines, dtype=np.int32))

    plotter = pv.Plotter(off_screen=True, window_size=(4800, 1600))
    plotter.set_background("white")
    plotter.add_mesh(
        surface,
        scalars="energy_density",
        cmap="viridis",
        preference="cell",
        interpolate_before_map=False,
        show_edges=False,
        lighting=False,
        show_scalar_bar=False,
    )
    plotter.add_mesh(edge_mesh, color="black", line_width=2.4, render_lines_as_tubes=False)
    extents = maxs - mins
    distance = 1.9 * float(extents[0])
    camera_position = [
        (float(center[0]), float(center[1]) - distance, float(center[2]) + 0.22 * distance),
        tuple(center.tolist()),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera_position = camera_position
    plotter.camera.view_angle = 28.0
    plotter.camera.parallel_projection = False
    plotter.show_bounds(
        bounds=(
            float(mins[0]),
            float(maxs[0]),
            float(mins[1]),
            float(maxs[1]),
            float(mins[2]),
            float(maxs[2]),
        ),
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True,
        show_xlabels=True,
        show_ylabels=False,
        show_zlabels=False,
        xtitle=" ",
        ytitle=" ",
        ztitle=" ",
        n_xlabels=3,
        fmt="%.1f",
        use_2d=True,
        all_edges=True,
        location="outer",
        ticks="outside",
        minor_ticks=False,
        font_size=24,
        color="black",
    )
    vector_out = family_pdf(f"{FAMILY}/hyperelasticity_sample_state_render_vector.pdf")
    try:
        vector_out.parent.mkdir(parents=True, exist_ok=True)
        plotter.save_graphic(vector_out, title="HyperElasticity beam render", raster=False, painter=True)
    except Exception:
        pass
    image = plotter.screenshot(return_img=True, scale=2)
    plotter.close()

    rgb = image[..., :3]
    non_white = np.any(rgb < 248, axis=2)
    rows, cols = np.where(non_white)
    if rows.size:
        y0 = max(int(rows.min()) - 18, 0)
        y1 = min(int(rows.max()) + 19, image.shape[0])
        x0 = max(int(cols.min()) - 24, 0)
        x1 = min(int(cols.max()) + 25, image.shape[1])
        image = image[y0:y1, x0:x1]

    fig = plt.figure(figsize=springer_figure_size(0.56))
    ax = fig.add_axes([0.04, 0.24, 0.92, 0.60])
    norm = Normalize(vmin=float(np.min(face_values)), vmax=float(np.max(face_values)))
    ax.imshow(image)
    ax.axis("off")
    scalar_map = cm.ScalarMappable(norm=norm, cmap="viridis")
    scalar_map.set_array(face_values)
    cax = fig.add_axes([0.16, 0.205, 0.68, 0.03])
    cbar = fig.colorbar(scalar_map, cax=cax, orientation="horizontal")
    cbar.set_label(r"energy density $W(F)$", labelpad=1.0)

    out = family_pdf(f"{FAMILY}/hyperelasticity_sample_state.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/hyperelasticity_sample_state.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_hyperelasticity_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/sample_state.npz"))],
        outputs=[repo_rel(out)],
        notes="True 3D perspective render of the level-4, 32-rank JAX+PETSc element publication state, looking at the mesh center with the beam axis shown horizontally, viridis energy-density coloring, and literal prism edges drawn from the undeformed box after deformation. The composed publication figure is saved as a high-resolution raster-backed PDF/PNG; a best-effort vector-only beam render sidecar is also attempted via PyVista.",
    )


def _energy_figure():
    plt = configure_matplotlib()
    rows = read_csv_rows(family_data(f"{FAMILY}/energy_levels.csv"))
    fig, ax = plt.subplots(figsize=springer_figure_size(0.78), constrained_layout=True)

    for implementation in ("fenics_custom", "jax_petsc_element", "jax_serial"):
        style = implementation_style(implementation)
        x = [int(row["level"]) for row in rows if row[implementation]]
        y = [float(row[implementation]) for row in rows if row[implementation]]
        ax.plot(
            x,
            y,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=style["label"],
        )

    ax.set_xlabel("mesh level")
    ax.set_ylabel(r"final energy $\Pi_h$")
    ax.legend(frameon=True, loc="upper right")

    out = family_pdf(f"{FAMILY}/hyperelasticity_energy_levels.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/hyperelasticity_energy_levels.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_hyperelasticity_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/energy_levels.csv"))],
        outputs=[repo_rel(out)],
        notes="Final-energy overview across levels for the maintained 24-step HyperElasticity runs.",
    )


def _strong_scaling_figure():
    plt = configure_matplotlib()
    rows = [row for row in read_csv_rows(family_data(f"{FAMILY}/strong_scaling.csv")) if row["result"] == "completed"]
    fig, ax = plt.subplots(figsize=springer_figure_size(height_in=3.25), constrained_layout=True)

    by_impl = {}
    for row in rows:
        by_impl.setdefault(row["solver"], []).append(row)
    ideal_drawn = False
    for implementation, impl_rows in sorted(by_impl.items()):
        style = implementation_style(implementation)
        impl_rows = sorted(impl_rows, key=lambda item: int(item["nprocs"]))
        ranks = [int(row["nprocs"]) for row in impl_rows]
        times = [float(row["total_time_s"]) for row in impl_rows]
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
            ax.plot(ranks, ideal_strong_scaling(ranks, times), color="#000000", linestyle="--", linewidth=1.2, label=r"ideal $1/r$")
            ideal_drawn = True
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("wall time [s]")
    ax.legend(frameon=True, loc="upper right")

    out = family_pdf(f"{FAMILY}/hyperelasticity_strong_scaling.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/hyperelasticity_strong_scaling.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_hyperelasticity_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/strong_scaling.csv"))],
        outputs=[repo_rel(out)],
        notes="Strong scaling on the finest maintained HyperElasticity mesh (level 4, 24 steps) with an ideal 1/r reference.",
    )


def _mesh_timing_figure():
    plt = configure_matplotlib()
    rows = [row for row in read_csv_rows(family_data(f"{FAMILY}/mesh_timing.csv")) if row["result"] == "completed"]
    fig, ax = plt.subplots(figsize=springer_figure_size(height_in=3.25), constrained_layout=True)

    by_impl = {}
    for row in rows:
        by_impl.setdefault(row["solver"], []).append(row)
    ideal_drawn = False
    for implementation, impl_rows in sorted(by_impl.items()):
        style = implementation_style(implementation)
        impl_rows = sorted(impl_rows, key=lambda item: int(item["problem_size"]))
        sizes = [int(row["problem_size"]) for row in impl_rows]
        times = [float(row["total_time_s"]) for row in impl_rows]
        ax.plot(
            sizes,
            times,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=style["label"],
        )
        if not ideal_drawn:
            ax.plot(sizes, ideal_mesh_scaling(sizes, times), color="#000000", linestyle="--", linewidth=1.2, label="ideal linear")
            ideal_drawn = True
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total DOFs")
    ax.set_ylabel("wall time [s]")
    ax.legend(frameon=True, loc="upper left")

    out = family_pdf(f"{FAMILY}/hyperelasticity_mesh_timing.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/hyperelasticity_mesh_timing.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_hyperelasticity_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/mesh_timing.csv"))],
        outputs=[repo_rel(out)],
        notes="Time-vs-mesh-size scaling on the maintained HyperElasticity suite at 32 MPI ranks with an ideal linear reference.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    _state_figure()
    _energy_figure()
    _strong_scaling_figure()
    _mesh_timing_figure()


if __name__ == "__main__":
    main()
