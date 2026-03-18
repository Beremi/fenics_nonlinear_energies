#!/usr/bin/env python3
"""Run the small publication reruns used by the overview package."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from src.core.benchmark.replication import read_json, run_logged_command, write_json

from common import (
    PYTHON,
    REPO_ROOT,
    RUNS_ROOT,
    THREAD_ENV,
    ensure_overview_dirs,
    publication_run_dir,
    record_provenance,
    repo_rel,
    shell_join,
)


MPIEXEC = shutil.which("mpiexec") or "mpiexec"


@dataclass(slots=True)
class OverviewRun:
    id: str
    family: str
    implementation: str
    command: list[str]
    run_dir: Path
    expected_outputs: list[Path]
    notes: str = ""


def _python_cmd(script: str, *args: str) -> list[str]:
    return [str(PYTHON), "-u", script, *args]


def _read_status(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _plaplace_runs() -> list[OverviewRun]:
    family = "plaplace"
    base = publication_run_dir(family, "showcase")
    return [
        OverviewRun(
            id="plaplace_fenics_custom",
            family=family,
            implementation="fenics_custom",
            command=_python_cmd(
                "src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py",
                "--levels", "5",
                "--quiet",
                "--json", repo_rel(base / "fenics_custom" / "output.json"),
                "--state-out", repo_rel(base / "fenics_custom" / "state.npz"),
            ),
            run_dir=base / "fenics_custom",
            expected_outputs=[base / "fenics_custom" / "output.json", base / "fenics_custom" / "state.npz"],
            notes="Showcase parity and sample-state rerun",
        ),
        OverviewRun(
            id="plaplace_fenics_snes",
            family=family,
            implementation="fenics_snes",
            command=_python_cmd(
                "src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py",
                "--levels", "5",
                "--json", repo_rel(base / "fenics_snes" / "output.json"),
            ),
            run_dir=base / "fenics_snes",
            expected_outputs=[base / "fenics_snes" / "output.json"],
        ),
        OverviewRun(
            id="plaplace_jax_serial",
            family=family,
            implementation="jax_serial",
            command=_python_cmd(
                "src/problems/plaplace/jax/solve_pLaplace_jax_newton.py",
                "--levels", "5",
                "--quiet",
                "--json", repo_rel(base / "jax_serial" / "output.json"),
            ),
            run_dir=base / "jax_serial",
            expected_outputs=[base / "jax_serial" / "output.json"],
        ),
        OverviewRun(
            id="plaplace_jax_petsc_element",
            family=family,
            implementation="jax_petsc_element",
            command=_python_cmd(
                "src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py",
                "--level", "5",
                "--profile", "reference",
                "--assembly-mode", "element",
                "--local-hessian-mode", "element",
                "--element-reorder-mode", "block_xyz",
                "--local-coloring",
                "--nproc", "1",
                "--quiet",
                "--json", repo_rel(base / "jax_petsc_element" / "output.json"),
            ),
            run_dir=base / "jax_petsc_element",
            expected_outputs=[base / "jax_petsc_element" / "output.json"],
        ),
        OverviewRun(
            id="plaplace_jax_petsc_local_sfd",
            family=family,
            implementation="jax_petsc_local_sfd",
            command=_python_cmd(
                "src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py",
                "--level", "5",
                "--profile", "reference",
                "--assembly-mode", "element",
                "--local-hessian-mode", "sfd_local",
                "--element-reorder-mode", "block_xyz",
                "--local-coloring",
                "--nproc", "1",
                "--quiet",
                "--json", repo_rel(base / "jax_petsc_local_sfd" / "output.json"),
            ),
            run_dir=base / "jax_petsc_local_sfd",
            expected_outputs=[base / "jax_petsc_local_sfd" / "output.json"],
        ),
    ]


def _gl_runs() -> list[OverviewRun]:
    family = "ginzburg_landau"
    base = publication_run_dir(family, "showcase")
    return [
        OverviewRun(
            id="gl_fenics_custom",
            family=family,
            implementation="fenics_custom",
            command=_python_cmd(
                "src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py",
                "--levels", "5",
                "--quiet",
                "--json", repo_rel(base / "fenics_custom" / "output.json"),
                "--state-out", repo_rel(base / "fenics_custom" / "state.npz"),
            ),
            run_dir=base / "fenics_custom",
            expected_outputs=[base / "fenics_custom" / "output.json", base / "fenics_custom" / "state.npz"],
            notes="Showcase parity and sample-state rerun",
        ),
        OverviewRun(
            id="gl_fenics_snes",
            family=family,
            implementation="fenics_snes",
            command=_python_cmd(
                "src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py",
                "--levels", "5",
                "--json", repo_rel(base / "fenics_snes" / "output.json"),
            ),
            run_dir=base / "fenics_snes",
            expected_outputs=[base / "fenics_snes" / "output.json"],
        ),
        OverviewRun(
            id="gl_jax_petsc_element",
            family=family,
            implementation="jax_petsc_element",
            command=_python_cmd(
                "src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py",
                "--level", "5",
                "--profile", "reference",
                "--assembly_mode", "element",
                "--local_hessian_mode", "element",
                "--element_reorder_mode", "block_xyz",
                "--local_coloring",
                "--nproc", "1",
                "--quiet",
                "--out", repo_rel(base / "jax_petsc_element" / "output.json"),
            ),
            run_dir=base / "jax_petsc_element",
            expected_outputs=[base / "jax_petsc_element" / "output.json"],
        ),
        OverviewRun(
            id="gl_jax_petsc_local_sfd",
            family=family,
            implementation="jax_petsc_local_sfd",
            command=_python_cmd(
                "src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py",
                "--level", "5",
                "--profile", "reference",
                "--assembly_mode", "element",
                "--local_hessian_mode", "sfd_local",
                "--element_reorder_mode", "block_xyz",
                "--local_coloring",
                "--nproc", "1",
                "--quiet",
                "--out", repo_rel(base / "jax_petsc_local_sfd" / "output.json"),
            ),
            run_dir=base / "jax_petsc_local_sfd",
            expected_outputs=[base / "jax_petsc_local_sfd" / "output.json"],
        ),
    ]


def _he_runs() -> list[OverviewRun]:
    family = "hyperelasticity"
    base = publication_run_dir(family, "showcase")
    render_base = publication_run_dir(family, "sample_render")
    return [
        OverviewRun(
            id="he_fenics_custom",
            family=family,
            implementation="fenics_custom",
            command=_python_cmd(
                "src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py",
                "--level", "1",
                "--steps", "24",
                "--total-steps", "24",
                "--quiet",
                "--out", repo_rel(base / "fenics_custom" / "output.json"),
            ),
            run_dir=base / "fenics_custom",
            expected_outputs=[base / "fenics_custom" / "output.json"],
        ),
        OverviewRun(
            id="he_jax_petsc_element",
            family=family,
            implementation="jax_petsc_element",
            command=_python_cmd(
                "src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py",
                "--level", "1",
                "--steps", "24",
                "--total_steps", "24",
                "--profile", "performance",
                "--ksp_type", "stcg",
                "--pc_type", "gamg",
                "--ksp_rtol", "1e-1",
                "--ksp_max_it", "30",
                "--gamg_threshold", "0.05",
                "--gamg_agg_nsmooths", "1",
                "--gamg_set_coordinates",
                "--use_near_nullspace",
                "--assembly_mode", "element",
                "--element_reorder_mode", "block_xyz",
                "--local_hessian_mode", "element",
                "--local_coloring",
                "--use_trust_region",
                "--trust_subproblem_line_search",
                "--linesearch_tol", "1e-1",
                "--trust_radius_init", "0.5",
                "--trust_shrink", "0.5",
                "--trust_expand", "1.5",
                "--trust_eta_shrink", "0.05",
                "--trust_eta_expand", "0.75",
                "--trust_max_reject", "6",
                "--nproc", "1",
                "--quiet",
                "--out", repo_rel(base / "jax_petsc_element" / "output.json"),
            ),
            run_dir=base / "jax_petsc_element",
            expected_outputs=[base / "jax_petsc_element" / "output.json"],
        ),
        OverviewRun(
            id="he_jax_serial",
            family=family,
            implementation="jax_serial",
            command=_python_cmd(
                "src/problems/hyperelasticity/jax/solve_HE_jax_newton.py",
                "--level", "1",
                "--steps", "24",
                "--total_steps", "24",
                "--quiet",
                "--out", repo_rel(base / "jax_serial" / "output.json"),
                "--state-out", repo_rel(base / "jax_serial" / "state.npz"),
            ),
            run_dir=base / "jax_serial",
            expected_outputs=[base / "jax_serial" / "output.json", base / "jax_serial" / "state.npz"],
            notes="Showcase parity and sample-state rerun",
        ),
        OverviewRun(
            id="he_jax_petsc_element_l4_np32",
            family=family,
            implementation="jax_petsc_element",
            command=[
                MPIEXEC,
                "-n", "32",
                * _python_cmd(
                    "src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py",
                    "--level", "4",
                    "--steps", "24",
                    "--total_steps", "24",
                    "--profile", "performance",
                    "--ksp_type", "stcg",
                    "--pc_type", "gamg",
                    "--ksp_rtol", "1e-1",
                    "--ksp_max_it", "30",
                    "--gamg_threshold", "0.05",
                    "--gamg_agg_nsmooths", "1",
                    "--gamg_set_coordinates",
                    "--use_near_nullspace",
                    "--no-pc_setup_on_ksp_cap",
                    "--assembly_mode", "element",
                    "--element_reorder_mode", "block_xyz",
                    "--local_hessian_mode", "element",
                    "--local_coloring",
                    "--tolf", "1e-4",
                    "--tolg", "1e-3",
                    "--tolg_rel", "1e-3",
                    "--tolx_rel", "1e-3",
                    "--tolx_abs", "1e-10",
                    "--maxit", "100",
                    "--linesearch_a", "-0.5",
                    "--linesearch_b", "2.0",
                    "--linesearch_tol", "1e-1",
                    "--use_trust_region",
                    "--trust_radius_init", "0.5",
                    "--trust_radius_min", "1e-8",
                    "--trust_radius_max", "1e6",
                    "--trust_shrink", "0.5",
                    "--trust_expand", "1.5",
                    "--trust_eta_shrink", "0.05",
                    "--trust_eta_expand", "0.75",
                    "--trust_max_reject", "6",
                    "--trust_subproblem_line_search",
                    "--nproc", "1",
                    "--quiet",
                    "--out", repo_rel(render_base / "jax_petsc_element_l4_np32" / "output.json"),
                    "--state-out", repo_rel(render_base / "jax_petsc_element_l4_np32" / "state.npz"),
                ),
            ],
            run_dir=render_base / "jax_petsc_element_l4_np32",
            expected_outputs=[
                render_base / "jax_petsc_element_l4_np32" / "output.json",
                render_base / "jax_petsc_element_l4_np32" / "state.npz",
            ],
            notes="High-resolution HyperElasticity publication render on the validated L4, np32 JAX+PETSc element case",
        ),
    ]


def _topology_mesh_scaling_runs() -> list[OverviewRun]:
    family = "topology"
    base = publication_run_dir(family, "mesh_scaling")
    cases = [
        ("nx192_ny96_np8", 192, 96, 8, 8),
        ("nx384_ny192_np8", 384, 192, 16, 16),
        ("nx768_ny384_np8", 768, 384, 32, 32),
    ]
    runs: list[OverviewRun] = []
    for case_id, nx, ny, fixed_pad, load_pad in cases:
        run_dir = base / case_id
        runs.append(
            OverviewRun(
                id=f"topology_{case_id}",
                family=family,
                implementation="jax_parallel",
                command=[
                    MPIEXEC,
                    "-n", "8",
                    *_python_cmd(
                        "src/problems/topology/jax/solve_topopt_parallel.py",
                        "--nx", str(nx),
                        "--ny", str(ny),
                        "--length", "2.0",
                        "--height", "1.0",
                        "--traction", "1.0",
                        "--load_fraction", "0.2",
                        "--fixed_pad_cells", str(fixed_pad),
                        "--load_pad_cells", str(load_pad),
                        "--volume_fraction_target", "0.4",
                        "--theta_min", "1e-6",
                        "--solid_latent", "10.0",
                        "--young", "1.0",
                        "--poisson", "0.3",
                        "--alpha_reg", "0.005",
                        "--ell_pf", "0.08",
                        "--mu_move", "0.01",
                        "--beta_lambda", "12.0",
                        "--volume_penalty", "10.0",
                        "--p_start", "1.0",
                        "--p_max", "10.0",
                        "--p_increment", "0.2",
                        "--continuation_interval", "1",
                        "--outer_maxit", "2000",
                        "--outer_tol", "0.02",
                        "--volume_tol", "0.001",
                        "--stall_theta_tol", "1e-6",
                        "--stall_p_min", "4.0",
                        "--design_maxit", "20",
                        "--tolf", "1e-6",
                        "--tolg", "1e-3",
                        "--linesearch_tol", "0.1",
                        "--linesearch_relative_to_bound",
                        "--design_gd_line_search", "golden_adaptive",
                        "--design_gd_adaptive_window_scale", "2.0",
                        "--mechanics_ksp_type", "fgmres",
                        "--mechanics_pc_type", "gamg",
                        "--mechanics_ksp_rtol", "1e-4",
                        "--mechanics_ksp_max_it", "100",
                        "--quiet",
                        "--json_out", repo_rel(run_dir / "output.json"),
                    ),
                ],
                run_dir=run_dir,
                expected_outputs=[run_dir / "output.json"],
                notes="Topology fixed-rank mesh-size sweep on the maintained parallel solver path",
            )
        )
    return runs


def build_runs() -> list[OverviewRun]:
    return [*_plaplace_runs(), *_gl_runs(), *_he_runs(), *_topology_mesh_scaling_runs()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Skip already successful publication reruns")
    args = parser.parse_args()

    ensure_overview_dirs()
    entries: list[dict[str, Any]] = []

    for task in build_runs():
        result = run_logged_command(
            command=task.command,
            cwd=REPO_ROOT,
            leaf_dir=task.run_dir,
            expected_outputs=task.expected_outputs,
            env=THREAD_ENV,
            resume=args.resume,
            notes=task.notes,
        )
        status = _read_status(task.run_dir / "status.json")
        entries.append(
            {
                "id": task.id,
                "family": task.family,
                "implementation": task.implementation,
                "run_dir": repo_rel(task.run_dir),
                "command": shell_join(task.command),
                "outputs": [repo_rel(path) for path in task.expected_outputs],
                "success": bool(result.success),
                "skipped": bool(result.skipped),
                "notes": task.notes,
                "status": status,
            }
        )

    manifest = {
        "entries": entries,
        "success_count": sum(1 for entry in entries if entry["success"]),
        "total_runs": len(entries),
    }
    write_json(RUNS_ROOT / "manifest.json", manifest)
    record_provenance(
        RUNS_ROOT / "build_publication_reruns.provenance.json",
        script_name="overview/img/scripts/build_publication_reruns.py",
        inputs=[],
        outputs=[repo_rel(RUNS_ROOT / "manifest.json")],
        notes="Runs the small showcase commands and topology mesh-scaling commands used by the overview package.",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
