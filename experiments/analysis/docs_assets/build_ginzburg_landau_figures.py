#!/usr/bin/env python3
"""Generate GinzburgLandau overview PDF figures."""

from __future__ import annotations

import argparse

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


FAMILY = "ginzburg_landau"


def _state_figure():
    plt = configure_matplotlib()
    from matplotlib.tri import Triangulation

    data = load_npz(family_data(f"{FAMILY}/sample_state.npz"))
    coords = data["coords"]
    triangles = data["triangles"]
    values = data["u"]

    fig, ax = plt.subplots(figsize=springer_figure_size(0.92), constrained_layout=True)
    tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
    artist = ax.tripcolor(tri, values, shading="flat", cmap="Greys")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    cbar = fig.colorbar(artist, ax=ax, pad=0.03)
    cbar.set_label(r"$u_h(x,y)$")

    out = family_pdf(f"{FAMILY}/ginzburg_landau_sample_state.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/ginzburg_landau_sample_state.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_ginzburg_landau_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/sample_state.npz"))],
        outputs=[repo_rel(out)],
        notes="Scalar showcase state rendered from the serial FEniCS custom publication rerun.",
    )


def _energy_figure():
    plt = configure_matplotlib()
    rows = read_csv_rows(family_data(f"{FAMILY}/energy_levels.csv"))
    fig, ax = plt.subplots(figsize=springer_figure_size(0.78), constrained_layout=True)

    for implementation in ("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"):
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
    ax.set_ylabel(r"final energy $E_h$")
    ax.legend(frameon=True, loc="upper right")

    out = family_pdf(f"{FAMILY}/ginzburg_landau_energy_levels.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/ginzburg_landau_energy_levels.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_ginzburg_landau_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/energy_levels.csv"))],
        outputs=[repo_rel(out)],
        notes="Energy-vs-level figure from the authoritative maintained GinzburgLandau suite.",
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

    out = family_pdf(f"{FAMILY}/ginzburg_landau_strong_scaling.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/ginzburg_landau_strong_scaling.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_ginzburg_landau_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/strong_scaling.csv"))],
        outputs=[repo_rel(out)],
        notes="Strong scaling on the finest maintained GinzburgLandau mesh (level 9) with an ideal 1/r reference.",
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
    ax.set_xlabel("free DOFs")
    ax.set_ylabel("wall time [s]")
    ax.legend(frameon=True, loc="upper left")

    out = family_pdf(f"{FAMILY}/ginzburg_landau_mesh_timing.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/ginzburg_landau_mesh_timing.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_ginzburg_landau_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/mesh_timing.csv"))],
        outputs=[repo_rel(out)],
        notes="Time-vs-mesh-size scaling on the maintained GinzburgLandau suite at 32 MPI ranks with an ideal linear reference.",
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
