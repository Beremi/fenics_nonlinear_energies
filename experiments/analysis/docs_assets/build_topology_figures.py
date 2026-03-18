#!/usr/bin/env python3
"""Generate Topology overview PDF figures."""

from __future__ import annotations

import argparse

from common import (
    family_data,
    family_pdf,
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


FAMILY = "topology"


def _state_figure():
    plt = configure_matplotlib()
    data = load_npz(family_data(f"{FAMILY}/parallel_final_state.npz"))
    theta = data["theta_grid"]

    fig, ax = plt.subplots(figsize=springer_figure_size(0.86), constrained_layout=True)
    artist = ax.imshow(
        theta.T,
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        extent=(0.0, 2.0, 0.0, 1.0),
        aspect="equal",
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    cbar = fig.colorbar(artist, ax=ax, orientation="horizontal", fraction=0.08, pad=0.08)
    cbar.set_label(r"density $\theta_h$")

    out = family_pdf(f"{FAMILY}/topology_final_density.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/topology_final_density.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_topology_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/parallel_final_state.npz"))],
        outputs=[repo_rel(out)],
        notes="Finished parallel topology density rendered from the validated 768x384, 32-rank final run.",
    )


def _objective_history_figure():
    plt = configure_matplotlib()
    rows = read_csv_rows(family_data(f"{FAMILY}/objective_history.csv"))

    outer_key = "outer_iteration" if rows and "outer_iteration" in rows[0] else "outer_iter"
    outer = [int(row[outer_key]) for row in rows]
    compliance = [float(row["compliance"]) for row in rows]
    volume = [float(row["volume_fraction"]) for row in rows]
    p_penal = [float(row["p_penal"]) for row in rows]
    compliance_norm = [value / max(compliance[0], 1e-12) for value in compliance]
    volume_target = volume[-1] if volume else 1.0
    volume_norm = [value / max(volume_target, 1e-12) for value in volume]
    p_norm = [value / max(p_penal[-1], 1e-12) for value in p_penal]

    fig, ax = plt.subplots(figsize=springer_figure_size(0.9), constrained_layout=True)
    ax.plot(outer, compliance_norm, color="#111111", linewidth=1.8, linestyle="-", marker="o", markevery=max(len(outer)//10, 1), label="compliance / initial")
    ax.plot(outer, volume_norm, color="#555555", linewidth=1.5, linestyle="--", marker="s", markevery=max(len(outer)//10, 1), label="volume / final")
    ax.plot(outer, p_norm, color="#888888", linewidth=1.5, linestyle=":", marker="^", markevery=max(len(outer)//10, 1), label=r"$p / p_{\max}$")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("normalised value")
    ax.legend(frameon=True, ncol=1, loc="upper right")

    out = family_pdf(f"{FAMILY}/topology_objective_history.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/topology_objective_history.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_topology_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/objective_history.csv"))],
        outputs=[repo_rel(out)],
        notes="Single-axis normalised compliance, volume, and continuation history for the finished parallel topology run.",
    )


def _strong_scaling_figure():
    plt = configure_matplotlib()
    rows = [row for row in read_csv_rows(family_data(f"{FAMILY}/strong_scaling.csv")) if row["result"] == "completed"]

    ranks = [int(row["ranks"]) for row in rows]
    wall = [float(row["wall_time_s"]) for row in rows]
    ideal = ideal_strong_scaling(ranks, wall)

    fig, ax = plt.subplots(figsize=springer_figure_size(height_in=3.25), constrained_layout=True)
    ax.plot(ranks, wall, marker="o", color="#222222", linestyle="-", linewidth=1.8, label="measured")
    ax.plot(ranks, ideal, linestyle="--", color="#4c4c4c", linewidth=1.2, label=r"ideal $1/r$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("wall time [s]")
    ax.legend(frameon=True, loc="upper right")

    out = family_pdf(f"{FAMILY}/topology_strong_scaling.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/topology_strong_scaling.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_topology_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/strong_scaling.csv"))],
        outputs=[repo_rel(out)],
        notes="Single-axis topology parallel wall-time scaling figure with an ideal 1/r reference.",
    )


def _mesh_timing_figure():
    plt = configure_matplotlib()
    rows = [row for row in read_csv_rows(family_data(f"{FAMILY}/mesh_timing.csv")) if row["result"] == "completed"]
    sizes = [int(row["problem_size"]) for row in rows]
    wall = [float(row["wall_time_s"]) for row in rows]
    ideal = ideal_mesh_scaling(sizes, wall)

    fig, ax = plt.subplots(figsize=springer_figure_size(height_in=3.25), constrained_layout=True)
    ax.plot(sizes, wall, marker="o", color="#222222", linestyle="-", linewidth=1.8, label="measured")
    ax.plot(sizes, ideal, color="#4c4c4c", linestyle="--", linewidth=1.2, label="ideal linear")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total free DOFs")
    ax.set_ylabel("wall time [s]")
    ax.legend(frameon=True, loc="upper left")

    out = family_pdf(f"{FAMILY}/topology_mesh_timing.pdf")
    save_pdf(fig, out)
    record_provenance(
        family_data(f"{FAMILY}/topology_mesh_timing.provenance.json"),
        script_name="experiments/analysis/docs_assets/build_topology_figures.py",
        inputs=[repo_rel(family_data(f"{FAMILY}/mesh_timing.csv"))],
        outputs=[repo_rel(out)],
        notes="Time-vs-mesh-size scaling for the maintained parallel topology solver at 8 MPI ranks with an ideal linear reference.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    _state_figure()
    _objective_history_figure()
    _strong_scaling_figure()
    _mesh_timing_figure()


if __name__ == "__main__":
    main()
