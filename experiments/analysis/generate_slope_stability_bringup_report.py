#!/usr/bin/env python3
"""Generate a visual markdown report for the slope-stability JAX bring-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from src.problems.slope_stability.jax.mesh import MeshSlopeStability2D


REPO_ROOT = Path(__file__).resolve().parents[2]


def _split_triangles_for_plot(elems_scalar: np.ndarray) -> np.ndarray:
    elem = np.asarray(elems_scalar, dtype=np.int64)
    split = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2], [5, 3, 4]], dtype=np.int64)
    tri = np.empty((4 * elem.shape[0], 3), dtype=np.int64)
    for i in range(elem.shape[0]):
        tri[4 * i : 4 * (i + 1), :] = elem[i][split]
    return tri


def _deviatoric_strain_norm_2d(E: np.ndarray) -> np.ndarray:
    iota = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    dev = np.diag([1.0, 1.0, 0.5]) - np.outer(iota, iota) / 2.0
    dev_e = dev @ E
    return np.sqrt(np.maximum(0.0, np.sum(E * dev_e, axis=0)))


def _plot_mesh(mesh: MeshSlopeStability2D, out_path: Path) -> None:
    coords = np.asarray(mesh.params["nodes"], dtype=np.float64)
    elems = np.asarray(mesh.params["elems_scalar"], dtype=np.int64)
    q_mask = np.asarray(mesh.params["q_mask"], dtype=bool)
    tri = _split_triangles_for_plot(elems)
    triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles=tri)

    fixed_x = ~q_mask[:, 0]
    fixed_y = ~q_mask[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    ax.triplot(triangulation, color="#36454F", linewidth=0.30, alpha=0.65)
    ax.scatter(coords[fixed_x, 0], coords[fixed_x, 1], s=8, color="#005F73", label="x constrained")
    ax.scatter(coords[fixed_y, 0], coords[fixed_y, 1], s=8, color="#BB3E03", label="y constrained")
    ax.annotate(
        "gravity",
        xy=(20.0, 18.0),
        xytext=(20.0, 20.0),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#222222"},
        ha="center",
        va="bottom",
    )
    ax.set_aspect("equal")
    ax.set_title("Reference P2 homogeneous SSR mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.02,
        f"{coords.shape[0]} nodes, {elems.shape[0]} P2 triangles",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#BBBBBB"},
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_displacement(state: dict[str, np.ndarray], elems_scalar: np.ndarray, out_path: Path) -> None:
    coords_ref = np.asarray(state["coords_ref"], dtype=np.float64)
    coords_final = np.asarray(state["coords_final"], dtype=np.float64)
    disp = np.asarray(state["displacement"], dtype=np.float64)
    disp_mag = np.linalg.norm(disp, axis=1)
    tri = _split_triangles_for_plot(elems_scalar)
    triangulation = mtri.Triangulation(coords_final[:, 0], coords_final[:, 1], triangles=tri)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    pc = ax.tripcolor(triangulation, disp_mag, shading="gouraud", cmap="viridis")
    ax.triplot(triangulation, color="black", linewidth=0.20, alpha=0.18)
    ax.set_aspect("equal")
    ax.set_title("Reached solution: displacement magnitude on deformed mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(pc, ax=ax, label=r"$||u||$")

    scale = 0.15
    sample = np.arange(0, coords_ref.shape[0], 40, dtype=np.int64)
    ax.quiver(
        coords_ref[sample, 0],
        coords_ref[sample, 1],
        disp[sample, 0],
        disp[sample, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / scale,
        width=0.0015,
        color="#111111",
        alpha=0.55,
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_deviatoric_strain(mesh: MeshSlopeStability2D, state: dict[str, np.ndarray], out_path: Path) -> None:
    coords_final = np.asarray(state["coords_final"], dtype=np.float64)
    disp = np.asarray(state["displacement"], dtype=np.float64).reshape(-1)
    elems_scalar = np.asarray(mesh.params["elems_scalar"], dtype=np.int64)
    elems_dof = np.asarray(mesh.params["elems"], dtype=np.int64)
    elem_B = np.asarray(mesh.params["elem_B"], dtype=np.float64)

    strain = np.einsum("eqij,ej->eqi", elem_B, disp[elems_dof])
    dev_q = _deviatoric_strain_norm_2d(strain.reshape(-1, 3).T).reshape(strain.shape[0], strain.shape[1])
    dev_elem = np.mean(dev_q, axis=1)

    tri = _split_triangles_for_plot(elems_scalar)
    triangulation = mtri.Triangulation(coords_final[:, 0], coords_final[:, 1], triangles=tri)
    facecolors = np.repeat(dev_elem, 4)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    pc = ax.tripcolor(triangulation, facecolors=facecolors, shading="flat", cmap="magma")
    ax.triplot(triangulation, color="black", linewidth=0.20, alpha=0.16)
    ax.set_aspect("equal")
    ax.set_title("Reached solution: mean deviatoric strain on deformed mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(pc, ax=ax, label="deviatoric strain")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_convergence(payload: dict, out_path: Path) -> None:
    history = list(payload.get("history", []))
    if not history:
        raise RuntimeError("No nonlinear history found in payload")

    it = np.array([int(row["it"]) for row in history], dtype=np.int32)
    energy = np.array([float(row["energy"]) for row in history], dtype=np.float64)
    grad = np.array([max(float(row.get("grad_norm", np.nan)), 1.0e-16) for row in history], dtype=np.float64)
    grad_target = np.array([max(float(row.get("grad_target", np.nan)), 1.0e-16) for row in history], dtype=np.float64)
    step_norm = np.array([max(float(row.get("step_norm", np.nan)), 1.0e-16) for row in history], dtype=np.float64)
    linear = np.array([int(row.get("ksp_its", 0)) for row in history], dtype=np.int32)
    trust_ratio = np.array([float(row.get("trust_ratio", np.nan)) for row in history], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), dpi=180)
    ax = axes[0, 0]
    ax.plot(it, energy, marker="o", color="#005F73")
    ax.set_title("Energy")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("total potential")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.semilogy(it, grad, marker="o", color="#9B2226", label="grad norm")
    ax.semilogy(it, grad_target, linestyle="--", color="#EE9B00", label="target")
    ax.set_title("Gradient convergence")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("norm")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    ax.semilogy(it, step_norm, marker="o", color="#3A5A40")
    ax.set_title("Step norm")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("norm")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(it, linear, marker="o", color="#6A4C93", label="linear its")
    ax2 = ax.twinx()
    ax2.plot(it, trust_ratio, marker="s", color="#C1121F", alpha=0.75, label="trust ratio")
    ax.set_title("Linear solve and trust ratio")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("KSP its")
    ax2.set_ylabel("trust ratio")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_markdown(
    out_path: Path,
    *,
    payload: dict,
    mesh_png: str,
    disp_png: str,
    dev_png: str,
    conv_png: str,
) -> None:
    result = payload["result"]
    material = payload["material"]
    mesh = payload["mesh"]

    lines = [
        "# Slope Stability P2 JAX Bring-Up Report",
        "",
        "Prototype status: zero-history endpoint solve on the external homogeneous SSR P2 mesh.",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Case | `{payload['case']['name']}` |",
        f"| Lambda target | `{payload['case']['lambda_target']}` |",
        f"| Nodes | `{mesh['nodes']}` |",
        f"| P2 triangles | `{mesh['elements']}` |",
        f"| Free DOFs | `{mesh['free_dofs']}` |",
        f"| Reduced cohesion | `{material['reduced']['cohesion']:.9f}` |",
        f"| Reduced phi [deg] | `{material['reduced']['phi_deg']:.9f}` |",
        f"| Final energy | `{result['final_energy']:.12f}` |",
        f"| Umax | `{result['u_max']:.12f}` |",
        f"| Newton iterations | `{result['newton_iters']}` |",
        f"| Linear iterations | `{result['linear_iters']}` |",
        f"| Status | `{result['status']}` |",
        "",
        "## Mesh",
        "",
        f"![Reference mesh]({mesh_png})",
        "",
        "## Solution Reached",
        "",
        "### Displacement",
        "",
        f"![Displacement magnitude]({disp_png})",
        "",
        "### Deviatoric Strain",
        "",
        f"![Deviatoric strain]({dev_png})",
        "",
        "## Convergence",
        "",
        f"![Convergence history]({conv_png})",
        "",
        "## Notes",
        "",
        "- The displacement and deviatoric-strain figures use the deformed configuration.",
        "- Deviatoric strain is shown as an elementwise quadrature-mean value using the prototype engineering-shear convention.",
        "- This is not a path-consistent SSR continuation result; `eps_p_old` is fixed to zero in this bring-up.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=str,
        default="artifacts/tmp_slope_stability_smoke/output.json",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="artifacts/tmp_slope_stability_smoke/state.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/reports/slope_stability_p2_jax_bringup",
    )
    args = parser.parse_args()

    json_path = (REPO_ROOT / args.json).resolve()
    state_path = (REPO_ROOT / args.state).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    state_npz = np.load(state_path)
    state = {key: state_npz[key] for key in state_npz.files}

    mesh = MeshSlopeStability2D(case=str(payload["case"]["name"]))

    mesh_png = out_dir / "mesh.png"
    disp_png = out_dir / "solution_displacement.png"
    dev_png = out_dir / "solution_deviatoric_strain.png"
    conv_png = out_dir / "convergence.png"
    md_path = out_dir / "report.md"

    _plot_mesh(mesh, mesh_png)
    _plot_displacement(state, np.asarray(mesh.params["elems_scalar"], dtype=np.int64), disp_png)
    _plot_deviatoric_strain(mesh, state, dev_png)
    _plot_convergence(payload, conv_png)
    _write_markdown(
        md_path,
        payload=payload,
        mesh_png=mesh_png.name,
        disp_png=disp_png.name,
        dev_png=dev_png.name,
        conv_png=conv_png.name,
    )

    print(md_path)


if __name__ == "__main__":
    main()
