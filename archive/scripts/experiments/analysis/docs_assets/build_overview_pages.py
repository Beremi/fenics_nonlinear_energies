#!/usr/bin/env python3
"""Generate the publication-style overview markdown pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DATA_ROOT,
    OVERVIEW_ROOT,
    REPLICATION_ROOT,
    FAMILY_TITLES,
    format_float,
    format_int,
    implementation_style,
    load_publication_manifest,
    markdown_table,
    normalize_command,
    read_csv_rows,
    read_json,
    record_provenance,
    repo_rel,
    shell_join,
)


ISSUE_MAP = {
    "plaplace": [
        "replications/2026-03-16_maintained_refresh/issues/plaplace_fenics_snes_parallel_mesh_construction.md",
    ],
    "ginzburg_landau": [
        "replications/2026-03-16_maintained_refresh/issues/gl_jax_petsc_direct_output_schema.md",
    ],
    "hyperelasticity": [
        "replications/2026-03-16_maintained_refresh/issues/he_jax_petsc_trust_region_cli_flags.md",
        "replications/2026-03-16_maintained_refresh/issues/he_suite_resume_restart.md",
    ],
    "topology": [],
}


def _load_replication_manifest() -> dict:
    return read_json(REPLICATION_ROOT / "manifest.json")


def _manifest_command(manifest: dict, task_id: str) -> str:
    for entry in manifest.get("entries", []):
        if entry.get("id") == task_id:
            return normalize_command(str(entry.get("command", "")))
    return ""


def _provenance_command(path: Path) -> str:
    return normalize_command(str(read_json(path).get("command", "")))


def _publication_command(family: str, implementation: str) -> str:
    manifest = load_publication_manifest()
    run_dir = f"overview/img/runs/{family}/showcase/{implementation}"
    for entry in manifest.get("entries", []):
        if entry.get("run_dir") == run_dir:
            return normalize_command(str(entry.get("command", "")))
    return ""


def _build_pages_command() -> str:
    return shell_join(["./.venv/bin/python", "overview/img/scripts/build_overview_pages.py"])


def _commands_block(commands: list[str]) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for command in commands:
        if not command or command in seen:
            continue
        seen.add(command)
        unique.append(command)
    return "\n\n".join(f"```bash\n{cmd}\n```" for cmd in unique)


def _topology_source_command(key: str) -> str:
    return normalize_command(str(read_json(DATA_ROOT / "topology" / "sources.json").get(key, "")))


def _family_source_command(family: str, key: str) -> str:
    return normalize_command(str(read_json(DATA_ROOT / family / "sources.json").get(key, "")))


def _topology_mesh_scaling_commands() -> list[str]:
    payload = read_json(DATA_ROOT / "topology" / "sources.json")
    commands = payload.get("mesh_scaling_commands", {})
    return [normalize_command(str(command)) for _, command in sorted(commands.items())]


def _issue_lines(family: str) -> str:
    issues = ISSUE_MAP[family]
    if not issues:
        return "- No new solver/path issues were discovered during the maintained replication refresh."
    return "\n".join(f"- `{issue}`" for issue in issues)


def _figure_link(rel_path: str, label: str) -> str:
    return f"[{label}]({rel_path})"


def _inline_figure(png_rel: str, pdf_rel: str, alt: str, pdf_label: str) -> str:
    return f"![{alt}]({png_rel})\n\nPDF: [{pdf_label}]({pdf_rel})"


def _table_from_csv(path: Path, headers: list[str], mapping) -> str:
    rows = []
    for row in read_csv_rows(path):
        rows.append(mapping(row))
    return markdown_table(headers, rows)


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _plaplace_problem(manifest: dict) -> str:
    energy_table = _table_from_csv(
        DATA_ROOT / "plaplace" / "energy_levels.csv",
        ["level", "FEniCS custom", "JAX+PETSc element", "JAX+PETSc local-SFD"],
        lambda row: [
            row["level"],
            format_float(row["fenics_custom"]),
            format_float(row["jax_petsc_element"]),
            format_float(row["jax_petsc_local_sfd"]),
        ],
    )
    parity_rows = read_csv_rows(DATA_ROOT / "plaplace" / "parity_showcase.csv")
    max_rel = max(float(row["rel_energy_delta_vs_ref"]) for row in parity_rows)
    commands = [
        _manifest_command(manifest, "plaplace_final_suite"),
        *[_publication_command("plaplace", impl) for impl in (
            "fenics_custom", "fenics_snes", "jax_serial", "jax_petsc_element", "jax_petsc_local_sfd"
        )],
        _provenance_command(DATA_ROOT / "plaplace" / "build_plaplace_data.provenance.json"),
        _provenance_command(DATA_ROOT / "plaplace" / "plaplace_sample_state.provenance.json"),
        _provenance_command(DATA_ROOT / "plaplace" / "plaplace_energy_levels.provenance.json"),
        _provenance_command(DATA_ROOT / "plaplace" / "plaplace_strong_scaling.provenance.json"),
        _provenance_command(DATA_ROOT / "plaplace" / "plaplace_mesh_timing.provenance.json"),
    ]
    commands_block = _commands_block(commands)
    return f"""# pLaplace Problem Overview

## Mathematical Model

We solve the nonlinear scalar p-Laplace minimisation problem

$$
E(u)=\\int_\\Omega \\frac{{1}}{{p}}\\lvert \\nabla u \\rvert^p\\,dx - \\int_\\Omega f u\\,dx,
\\qquad p=3,\\quad f=-10,
$$

on the unit square $\\Omega=(0,1)^2$ with homogeneous Dirichlet data
$u=0$ on $\\partial\\Omega$. The Euler-Lagrange equation is the nonlinear
diffusion problem

$$
-\\nabla \\cdot \\left(\\lvert \\nabla u \\rvert^{{p-2}} \\nabla u \\right) = f,
$$

so the benchmark couples a strongly nonlinear constitutive law with a fixed
second-order elliptic geometry.

## Geometry, Boundary Conditions, And Setup

- domain: unit square
- boundary condition: homogeneous Dirichlet on the full boundary
- forcing: constant negative load $f=-10$
- benchmark hierarchy: maintained mesh levels $5,6,7,8,9$
- benchmark intent: compare custom nonlinear solvers against JAX-derived
  Hessian paths on the same finite-element problem

## Discretization And Mesh Source

The maintained implementation uses first-order Lagrange finite elements on the
canonical triangular meshes in `data/meshes/pLaplace/`. The problem is one
degree of freedom per node, so mesh refinement directly increases the free-DOF
count used in the solver and scaling figures.

## Maintained Implementations

| implementation | role in overview |
| --- | --- |
| FEniCS custom Newton | authoritative suite + showcase state export |
| FEniCS SNES | showcase comparison only |
| pure JAX serial | showcase comparison only |
| JAX+PETSc element Hessian | authoritative suite + showcase comparison |
| JAX+PETSc local-SFD Hessian | authoritative suite + showcase comparison |

## Showcase Sample Result

The publication showcase field is exported from the serial FEniCS custom Newton
rerun at level `5`. The converged maintained implementations agree on the final
scalar energy on this shared showcase case to a relative tolerance of
approximately `{max_rel:.3e}`.

{_inline_figure('img/png/plaplace/plaplace_sample_state.png', 'img/pdf/plaplace/plaplace_sample_state.pdf', 'pLaplace showcase solution preview', 'Showcase solution PDF')}

{_inline_figure('img/png/plaplace/plaplace_energy_levels.png', 'img/pdf/plaplace/plaplace_energy_levels.pdf', 'pLaplace energy-vs-level preview', 'Energy-vs-level PDF')}

## Energy Table Across Levels

The level table below uses the authoritative maintained benchmark suite at
`np=1`.

{energy_table}

## Caveats And Repaired Issues

{_issue_lines('plaplace')}

## Commands Used

{commands_block}
"""


def _plaplace_comparison(manifest: dict) -> str:
    parity_table = _table_from_csv(
        DATA_ROOT / "plaplace" / "parity_showcase.csv",
        ["implementation", "energy", "rel. diff vs ref", "Newton", "linear", "wall [s]"],
        lambda row: [
            implementation_style(row["implementation"])["label"],
            format_float(row["final_energy"]),
            format_float(row["rel_energy_delta_vs_ref"], 3),
            format_int(row["newton_iters"]),
            format_int(row["linear_iters"]),
            format_float(row["wall_time_s"], 4),
        ],
    )
    strong_scaling_table = _table_from_csv(
        DATA_ROOT / "plaplace" / "strong_scaling.csv",
        ["implementation", "ranks", "time [s]", "Newton", "linear", "energy"],
        lambda row: [
            implementation_style(row["solver"])["label"],
            format_int(row["nprocs"]),
            format_float(row["total_time_s"], 4),
            format_int(row["total_newton_iters"]),
            format_int(row["total_linear_iters"]),
            format_float(row["final_energy"]),
        ],
    )
    mesh_table = _table_from_csv(
        DATA_ROOT / "plaplace" / "mesh_timing.csv",
        ["implementation", "level", "free DOFs", "ranks", "time [s]", "energy"],
        lambda row: [
            implementation_style(row["solver"])["label"],
            format_int(row["level"]),
            format_int(row["problem_size"]),
            format_int(row["nprocs"]),
            format_float(row["total_time_s"], 4),
            format_float(row["final_energy"]),
        ],
    )
    commands = [
        _manifest_command(manifest, "plaplace_final_suite"),
        *[_publication_command("plaplace", impl) for impl in (
            "fenics_custom", "fenics_snes", "jax_serial", "jax_petsc_element", "jax_petsc_local_sfd"
        )],
        _provenance_command(DATA_ROOT / "plaplace" / "build_plaplace_data.provenance.json"),
        _provenance_command(DATA_ROOT / "plaplace" / "plaplace_strong_scaling.provenance.json"),
        _provenance_command(DATA_ROOT / "plaplace" / "plaplace_mesh_timing.provenance.json"),
    ]
    commands_block = _commands_block(commands)
    return f"""# pLaplace Implementation Comparison

## Maintained Implementation Roster

| implementation | comparison role |
| --- | --- |
| FEniCS custom Newton | authoritative suite + showcase parity |
| FEniCS SNES | showcase parity only |
| pure JAX serial | showcase parity only |
| JAX+PETSc element Hessian | authoritative suite + showcase parity |
| JAX+PETSc local-SFD Hessian | authoritative suite + showcase parity |

## Shared-Case Result Equivalence

The shared showcase case is level `5` at `np=1`. Only converged
implementations are included below.

{parity_table}

## Scaling And Speed Comparison

{_inline_figure('img/png/plaplace/plaplace_strong_scaling.png', 'img/pdf/plaplace/plaplace_strong_scaling.pdf', 'pLaplace finest-mesh strong scaling preview', 'pLaplace strong-scaling PDF')}

{_inline_figure('img/png/plaplace/plaplace_mesh_timing.png', 'img/pdf/plaplace/plaplace_mesh_timing.pdf', 'pLaplace time-vs-mesh-size preview', 'pLaplace time-vs-mesh-size PDF')}
- raw direct speed source: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/direct_speed.csv`
- authoritative maintained scaling source: `replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite/summary.json`

Finest maintained suite scaling (`level 9`):

{strong_scaling_table}

Fixed-rank mesh-size timing (`32` ranks):

{mesh_table}

## Notes On Exclusions

- No pLaplace showcase implementation had to be excluded: all five maintained
  paths converged on the shared level-`5` serial comparison case.

## Raw Outputs And Figures

- replication suite: `replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite`
- publication reruns: `overview/img/runs/plaplace/showcase`
- curated figures: `overview/img/pdf/plaplace/` and `overview/img/png/plaplace/`

## Commands Used

{commands_block}
"""


def _gl_problem(manifest: dict) -> str:
    energy_table = _table_from_csv(
        DATA_ROOT / "ginzburg_landau" / "energy_levels.csv",
        ["level", "FEniCS custom", "JAX+PETSc element", "JAX+PETSc local-SFD"],
        lambda row: [
            row["level"],
            format_float(row["fenics_custom"]),
            format_float(row["jax_petsc_element"]),
            format_float(row["jax_petsc_local_sfd"]),
        ],
    )
    parity_rows = read_csv_rows(DATA_ROOT / "ginzburg_landau" / "parity_showcase.csv")
    max_rel = max(float(row["rel_energy_delta_vs_ref"]) for row in parity_rows)
    commands = [
        _manifest_command(manifest, "gl_final_suite"),
        *[_publication_command("ginzburg_landau", impl) for impl in (
            "fenics_custom", "fenics_snes", "jax_petsc_element", "jax_petsc_local_sfd"
        )],
        _provenance_command(DATA_ROOT / "ginzburg_landau" / "build_ginzburg_landau_data.provenance.json"),
        _provenance_command(DATA_ROOT / "ginzburg_landau" / "ginzburg_landau_sample_state.provenance.json"),
        _provenance_command(DATA_ROOT / "ginzburg_landau" / "ginzburg_landau_energy_levels.provenance.json"),
        _provenance_command(DATA_ROOT / "ginzburg_landau" / "ginzburg_landau_strong_scaling.provenance.json"),
        _provenance_command(DATA_ROOT / "ginzburg_landau" / "ginzburg_landau_mesh_timing.provenance.json"),
    ]
    commands_block = _commands_block(commands)
    return f"""# GinzburgLandau Problem Overview

## Mathematical Model

We solve the non-convex scalar Ginzburg-Landau energy minimisation problem

$$
E(u)=\\int_\\Omega \\frac{{\\varepsilon}}{{2}}\\lvert \\nabla u \\rvert^2
+ \\frac{{1}}{{4}}(u^2-1)^2\\,dx,
\\qquad \\varepsilon = 10^{{-2}},
$$

on $\\Omega=[-1,1]^2$ with homogeneous Dirichlet boundary data $u=0$ on
$\\partial\\Omega$. The benchmark sits in the non-convex regime where the
double-well potential competes against the gradient regularisation, so the
nonlinear solves must pass through indefinite local curvature while still
converging to the same discrete minimiser.

## Geometry, Boundary Conditions, And Setup

- domain: square $[-1,1]^2$
- boundary condition: homogeneous Dirichlet on the full boundary
- benchmark hierarchy: maintained mesh levels $5,6,7,8,9$
- difficulty: non-convex double-well potential with indefinite Hessian regions
- benchmark intent: compare custom Newton logic against JAX-derived PETSc
  Hessian paths on a fixed non-convex scalar model

## Discretization And Mesh Source

All maintained implementations use first-order Lagrange finite elements on the
canonical triangular meshes in `data/meshes/GinzburgLandau/`. As in pLaplace,
the problem is one degree of freedom per node, so the mesh-size timing figure
tracks free-DOF growth directly.

## Maintained Implementations

| implementation | role in overview |
| --- | --- |
| FEniCS custom Newton | authoritative suite + showcase state export |
| FEniCS SNES trust-region | showcase comparison only |
| JAX+PETSc element Hessian | authoritative suite + showcase comparison |
| JAX+PETSc local-SFD Hessian | authoritative suite + showcase comparison |

## Showcase Sample Result

The publication showcase field is exported from the serial FEniCS custom Newton
rerun at level `5`. The converged maintained implementations agree on the final
showcase energy to a relative tolerance of approximately `{max_rel:.3e}`.

{_inline_figure('img/png/ginzburg_landau/ginzburg_landau_sample_state.png', 'img/pdf/ginzburg_landau/ginzburg_landau_sample_state.pdf', 'GinzburgLandau showcase solution preview', 'Showcase solution PDF')}

{_inline_figure('img/png/ginzburg_landau/ginzburg_landau_energy_levels.png', 'img/pdf/ginzburg_landau/ginzburg_landau_energy_levels.pdf', 'GinzburgLandau energy-vs-level preview', 'Energy-vs-level PDF')}

## Energy Table Across Levels

The level table below uses the authoritative maintained benchmark suite at
`np=1`.

{energy_table}

## Caveats And Repaired Issues

{_issue_lines('ginzburg_landau')}

## Commands Used

{commands_block}
"""


def _gl_comparison(manifest: dict) -> str:
    parity_table = _table_from_csv(
        DATA_ROOT / "ginzburg_landau" / "parity_showcase.csv",
        ["implementation", "energy", "rel. diff vs ref", "Newton", "linear", "wall [s]"],
        lambda row: [
            implementation_style(row["implementation"])["label"],
            format_float(row["final_energy"]),
            format_float(row["rel_energy_delta_vs_ref"], 3),
            format_int(row["newton_iters"]),
            format_int(row["linear_iters"]),
            format_float(row["wall_time_s"], 4),
        ],
    )
    strong_scaling_table = _table_from_csv(
        DATA_ROOT / "ginzburg_landau" / "strong_scaling.csv",
        ["implementation", "ranks", "time [s]", "Newton", "linear", "energy"],
        lambda row: [
            implementation_style(row["solver"])["label"],
            format_int(row["nprocs"]),
            format_float(row["total_time_s"], 4),
            format_int(row["total_newton_iters"]),
            format_int(row["total_linear_iters"]),
            format_float(row["final_energy"]),
        ],
    )
    mesh_table = _table_from_csv(
        DATA_ROOT / "ginzburg_landau" / "mesh_timing.csv",
        ["implementation", "level", "free DOFs", "ranks", "time [s]", "energy"],
        lambda row: [
            implementation_style(row["solver"])["label"],
            format_int(row["level"]),
            format_int(row["problem_size"]),
            format_int(row["nprocs"]),
            format_float(row["total_time_s"], 4),
            format_float(row["final_energy"]),
        ],
    )
    commands = [
        _manifest_command(manifest, "gl_final_suite"),
        *[_publication_command("ginzburg_landau", impl) for impl in (
            "fenics_custom", "fenics_snes", "jax_petsc_element", "jax_petsc_local_sfd"
        )],
        _provenance_command(DATA_ROOT / "ginzburg_landau" / "build_ginzburg_landau_data.provenance.json"),
        _provenance_command(DATA_ROOT / "ginzburg_landau" / "ginzburg_landau_strong_scaling.provenance.json"),
        _provenance_command(DATA_ROOT / "ginzburg_landau" / "ginzburg_landau_mesh_timing.provenance.json"),
    ]
    commands_block = _commands_block(commands)
    return f"""# GinzburgLandau Implementation Comparison

## Maintained Implementation Roster

| implementation | comparison role |
| --- | --- |
| FEniCS custom Newton | authoritative suite + showcase parity |
| FEniCS SNES trust-region | showcase parity only |
| JAX+PETSc element Hessian | authoritative suite + showcase parity |
| JAX+PETSc local-SFD Hessian | authoritative suite + showcase parity |

## Shared-Case Result Equivalence

The shared showcase case is level `5` at `np=1`. Only converged
implementations are included below.

{parity_table}

## Scaling And Speed Comparison

{_inline_figure('img/png/ginzburg_landau/ginzburg_landau_strong_scaling.png', 'img/pdf/ginzburg_landau/ginzburg_landau_strong_scaling.pdf', 'GinzburgLandau finest-mesh strong scaling preview', 'GinzburgLandau strong-scaling PDF')}

{_inline_figure('img/png/ginzburg_landau/ginzburg_landau_mesh_timing.png', 'img/pdf/ginzburg_landau/ginzburg_landau_mesh_timing.pdf', 'GinzburgLandau time-vs-mesh-size preview', 'GinzburgLandau time-vs-mesh-size PDF')}
- raw direct speed source: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/direct_speed.csv`
- authoritative maintained scaling source: `replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_suite/summary.json`

Finest maintained suite scaling (`level 9`):

{strong_scaling_table}

Fixed-rank mesh-size timing (`32` ranks):

{mesh_table}

## Notes On Exclusions

- All four maintained GinzburgLandau implementations converged on the shared
  level-`5` serial showcase case, so the parity table contains the full working
  roster.
- The authoritative maintained final suite still tracks only the custom FEniCS
  and JAX+PETSc paths; FEniCS SNES remains a showcase-only reference here.

## Raw Outputs And Figures

- replication suite: `replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_suite`
- publication reruns: `overview/img/runs/ginzburg_landau/showcase`
- curated figures: `overview/img/pdf/ginzburg_landau/` and `overview/img/png/ginzburg_landau/`

## Commands Used

{commands_block}
"""


def _he_problem(manifest: dict) -> str:
    energy_table = _table_from_csv(
        DATA_ROOT / "hyperelasticity" / "energy_levels.csv",
        ["level", "FEniCS custom", "JAX+PETSc element", "pure JAX serial"],
        lambda row: [
            row["level"],
            format_float(row["fenics_custom"], 3),
            format_float(row["jax_petsc_element"], 3),
            format_float(row["jax_serial"], 3),
        ],
    )
    parity_rows = read_csv_rows(DATA_ROOT / "hyperelasticity" / "parity_showcase.csv")
    max_rel = max(float(row["rel_energy_delta_vs_ref"]) for row in parity_rows)
    commands = [
        _manifest_command(manifest, "he_final_suite_best"),
        _manifest_command(manifest, "he_pure_jax_suite_best"),
        *[_publication_command("hyperelasticity", impl) for impl in (
            "fenics_custom", "jax_petsc_element", "jax_serial"
        )],
        _family_source_command("hyperelasticity", "sample_render_command"),
        _provenance_command(DATA_ROOT / "hyperelasticity" / "build_hyperelasticity_data.provenance.json"),
        _provenance_command(DATA_ROOT / "hyperelasticity" / "hyperelasticity_sample_state.provenance.json"),
        _provenance_command(DATA_ROOT / "hyperelasticity" / "hyperelasticity_energy_levels.provenance.json"),
        _provenance_command(DATA_ROOT / "hyperelasticity" / "hyperelasticity_strong_scaling.provenance.json"),
        _provenance_command(DATA_ROOT / "hyperelasticity" / "hyperelasticity_mesh_timing.provenance.json"),
    ]
    commands_block = _commands_block(commands)
    return f"""# HyperElasticity Problem Overview

## Mathematical Model

The maintained HyperElasticity benchmark minimises the compressible Neo-Hookean
stored energy

$$
\\Pi(y)=\\int_\\Omega C_1\\bigl(\\operatorname{{tr}}(F^T F)-3-2\\ln J\\bigr)
+ D_1(J-1)^2\\,dx,
\\qquad F = \\nabla y, \\quad J = \\det F,
$$

with $C_1 = 38461538.461538464$ and $D_1 = 83333333.33333333$.
The unknown is the deformation map $y = X + u$, so the solver tracks a
three-component displacement field while preserving the nonlinear elastic
energy structure exactly at the discrete level.

## Geometry, Boundary Conditions, And Setup

- geometry: 3D cantilever beam on $[0,0.4] \\times [-0.005,0.005]^2$
- left face: clamped
- right face: prescribed rotating boundary motion
- load path: maintained `24`-step and `96`-step trajectories
- benchmark intent: compare two maintained distributed trust-region paths on a
  genuinely vector-valued large-deformation problem while keeping a pure-JAX
  serial reference on the same energy model

## Discretization And Mesh Source

The maintained benchmark uses vector-valued first-order tetrahedral finite
elements on the canonical meshes under `data/meshes/HyperElasticity/`. The
distributed PETSc paths use block-aware `xyz`-grouped free-DOF ordering,
three-by-three block structure, GAMG coordinates, and rigid-body near-nullspace
vectors so that the elasticity-like linear systems remain scalable on the
largest maintained meshes.

## Maintained Implementations

| implementation | role in overview |
| --- | --- |
| FEniCS custom trust-region Newton | authoritative suite + showcase parity |
| FEniCS SNES | retained comparison point, excluded from parity due failure |
| JAX+PETSc element Hessian | authoritative suite + showcase parity + sample render |
| pure JAX serial | authoritative serial reference + showcase parity |

## Showcase Sample Result

The publication image is exported from a dedicated maintained JAX+PETSc element
render run at level `4` on `32` MPI ranks. The converged maintained
implementations still agree on the shared parity showcase case at level `1`
with `24` load steps to a relative tolerance of approximately `{max_rel:.3e}`.
The viewpoint is chosen automatically from the beam's principal-extent axis, so
the render looks straight down the beam length while preserving equal aspect
ratio and orthographic projection.

{_inline_figure('img/png/hyperelasticity/hyperelasticity_sample_state.png', 'img/pdf/hyperelasticity/hyperelasticity_sample_state.pdf', 'HyperElasticity showcase deformed-shape preview', 'Showcase deformed-shape PDF')}

{_inline_figure('img/png/hyperelasticity/hyperelasticity_energy_levels.png', 'img/pdf/hyperelasticity/hyperelasticity_energy_levels.pdf', 'HyperElasticity energy-vs-level preview', 'Energy-vs-level PDF')}

## Energy Table Across Levels

The table below uses the maintained `24`-step reference path at `np=1`.

{energy_table}

## Caveats And Repaired Issues

{_issue_lines('hyperelasticity')}

## Commands Used

{commands_block}
"""


def _he_comparison(manifest: dict) -> str:
    parity_table = _table_from_csv(
        DATA_ROOT / "hyperelasticity" / "parity_showcase.csv",
        ["implementation", "completed steps", "energy", "rel. diff vs ref", "Newton", "linear", "wall [s]"],
        lambda row: [
            implementation_style(row["implementation"])["label"],
            format_int(row["completed_steps"]),
            format_float(row["final_energy"], 3),
            format_float(row["rel_energy_delta_vs_ref"], 3),
            format_int(row["total_newton_iters"]),
            format_int(row["total_linear_iters"]),
            format_float(row["wall_time_s"], 3),
        ],
    )
    strong_scaling_table = _table_from_csv(
        DATA_ROOT / "hyperelasticity" / "strong_scaling.csv",
        ["implementation", "ranks", "time [s]", "Newton", "linear", "energy"],
        lambda row: [
            implementation_style(row["solver"])["label"],
            format_int(row["nprocs"]),
            format_float(row["total_time_s"], 3),
            format_int(row["total_newton_iters"]),
            format_int(row["total_linear_iters"]),
            format_float(row["final_energy"], 3),
        ],
    )
    mesh_table = _table_from_csv(
        DATA_ROOT / "hyperelasticity" / "mesh_timing.csv",
        ["implementation", "level", "total DOFs", "ranks", "time [s]", "energy"],
        lambda row: [
            implementation_style(row["solver"])["label"],
            format_int(row["level"]),
            format_int(row["problem_size"]),
            format_int(row["nprocs"]),
            format_float(row["total_time_s"], 3),
            format_float(row["final_energy"], 3),
        ],
    )
    commands = [
        _manifest_command(manifest, "he_final_suite_best"),
        _manifest_command(manifest, "he_pure_jax_suite_best"),
        *[_publication_command("hyperelasticity", impl) for impl in (
            "fenics_custom", "jax_petsc_element", "jax_serial"
        )],
        _family_source_command("hyperelasticity", "sample_render_command"),
        _provenance_command(DATA_ROOT / "hyperelasticity" / "build_hyperelasticity_data.provenance.json"),
        _provenance_command(DATA_ROOT / "hyperelasticity" / "hyperelasticity_strong_scaling.provenance.json"),
        _provenance_command(DATA_ROOT / "hyperelasticity" / "hyperelasticity_mesh_timing.provenance.json"),
    ]
    commands_block = _commands_block(commands)
    return f"""# HyperElasticity Implementation Comparison

## Maintained Implementation Roster

| implementation | comparison role |
| --- | --- |
| FEniCS custom trust-region Newton | authoritative suite + showcase parity |
| FEniCS SNES | excluded from parity; fails on the showcase case |
| JAX+PETSc element Hessian | authoritative suite + showcase parity + fine-mesh scaling |
| pure JAX serial | authoritative serial reference + showcase parity only |

## Shared-Case Result Equivalence

The shared showcase case is level `1` with `24` load steps at `np=1`. Only
working implementations are included in the parity table.

{parity_table}

## Scaling And Speed Comparison

{_inline_figure('img/png/hyperelasticity/hyperelasticity_strong_scaling.png', 'img/pdf/hyperelasticity/hyperelasticity_strong_scaling.pdf', 'HyperElasticity finest-mesh strong scaling preview', 'HyperElasticity strong-scaling PDF')}

{_inline_figure('img/png/hyperelasticity/hyperelasticity_mesh_timing.png', 'img/pdf/hyperelasticity/hyperelasticity_mesh_timing.pdf', 'HyperElasticity time-vs-mesh-size preview', 'HyperElasticity time-vs-mesh-size PDF')}
- raw direct speed source: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/direct_speed.csv`
- authoritative maintained scaling source: `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best/summary.json`

Finest maintained strong scaling (`level 4`, `24` steps):

{strong_scaling_table}

Fixed-rank mesh-size timing (`32` ranks, `24` steps):

{mesh_table}

## Notes On Exclusions

- FEniCS SNES is excluded from the parity table because it fails on the
  shared showcase case in the maintained direct comparison data.
- Pure JAX is intentionally excluded from the scaling figures: the maintained
  pure-JAX path is single-process only and a new `level 4` serial rerun would
  add a large extra cost without changing the distributed comparison.

## Raw Outputs And Figures

- MPI suite: `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best`
- pure-JAX suite: `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/pure_jax_suite_best`
- publication reruns: `overview/img/runs/hyperelasticity/showcase`
- curated figures: `overview/img/pdf/hyperelasticity/` and `overview/img/png/hyperelasticity/`

## Commands Used

{commands_block}
"""


def _topology_problem(manifest: dict) -> str:
    resolution_table = _table_from_csv(
        DATA_ROOT / "topology" / "resolution_objectives.csv",
        ["label", "mesh", "ranks", "result", "outer", "compliance", "volume", "wall [s]"],
        lambda row: [
            row["label"],
            row["mesh"],
            format_int(row["ranks"]),
            row["result"],
            format_int(row["outer_iterations"]),
            format_float(row["final_compliance"], 4),
            format_float(row["final_volume_fraction"], 4),
            format_float(row["wall_time_s"], 3),
        ],
    )
    commands = [
        _topology_source_command("serial_command"),
        _topology_source_command("parallel_final_command"),
        _topology_source_command("parallel_scaling_command"),
        *_topology_mesh_scaling_commands(),
        _provenance_command(DATA_ROOT / "topology" / "build_topology_data.provenance.json"),
        _provenance_command(DATA_ROOT / "topology" / "topology_final_density.provenance.json"),
        _provenance_command(DATA_ROOT / "topology" / "topology_objective_history.provenance.json"),
        _provenance_command(DATA_ROOT / "topology" / "topology_strong_scaling.provenance.json"),
        _provenance_command(DATA_ROOT / "topology" / "topology_mesh_timing.provenance.json"),
    ]
    commands_block = _commands_block(commands)
    return f"""# Topology Problem Overview

## Mathematical Model

The maintained topology workflow solves a reduced compliance-minimisation
problem with a density field, phase-field regularisation, a proximal move
penalty, and staircase SIMP continuation. The mechanics step solves the
equilibrium problem for a fixed density field, then the design step minimises a
scalar FE energy with frozen mechanics data. In shorthand, the design update
minimises

$$
\\mathcal{{J}}(\\theta, z)
= C(\\theta, u)
+ \\lambda_{{V}} \\int_\\Omega \\theta\\,dx
+ \\alpha \\int_\\Omega \\left(\\frac{{\\ell}}{{2}}\\lvert \\nabla \\theta \\rvert^2
+ \\frac{{1}}{{\\ell}}W(\\theta)\\right) dx
+ \\frac{{\\mu}}{{2}}\\| z-z_{{\\mathrm{{old}}}} \\|^2,
$$

with a target volume fraction of `0.4` and maintained continuation in the SIMP
penalisation parameter $p$. This keeps the mechanics subproblem in the same
energy-minimisation setting as the repository's elasticity solvers while the
design update looks like a data-driven Ginzburg-Landau problem on a fixed
scalar graph.

## Geometry, Boundary Conditions, And Setup

- geometry: 2D cantilever beam of length `2.0` and height `1.0`
- left boundary: clamped support
- right-edge traction patch: `load_fraction = 0.2`
- maintained reference meshes:
  - serial reference: `192 x 96`
  - parallel fine benchmark / scaling: `768 x 384`
- maintained active path: distributed JAX+PETSc topology solve
- maintained reference path: pure-JAX serial solve retained for smaller
  comparison and formulation sanity checks

## Discretization And Mesh Source

The maintained topology path uses structured triangular displacement/design
meshes with separate free-DOF layouts for mechanics and design. The mechanics
phase uses a vector elasticity operator with PETSc `fgmres + gamg`, rigid-body
near-nullspace enrichment, and fixed-rank distributed assembly. The design
phase uses a distributed gradient-based solve on the same mesh family with a
fixed continuation schedule in $p$.

## Maintained Implementations

| implementation | role in overview |
| --- | --- |
| pure JAX serial | serial reference benchmark and showcase state |
| parallel JAX+PETSc | fine-grid final benchmark and scaling study |

## Showcase Sample Result

The finished parallel fine-grid campaign now provides the publication density
figure, the objective-history figure, and the curated density-evolution
animation, while the serial reference remains the smaller direct-comparison
baseline. The finished parallel final state shown below is the maintained
`768 x 384`, `32`-rank run rather than a partially converged demonstration.

{_inline_figure('img/png/topology/topology_final_density.png', 'img/pdf/topology/topology_final_density.pdf', 'Topology final density preview', 'Final density PDF')}

{_inline_figure('img/png/topology/topology_objective_history.png', 'img/pdf/topology/topology_objective_history.pdf', 'Topology objective-history preview', 'Objective-history PDF')}

![Topology parallel final density evolution](img/gif/topology/topology_parallel_final_evolution.gif)

## Resolution / Objective Table

{resolution_table}

## Caveats And Repaired Issues

{_issue_lines('topology')}

## Commands Used

{commands_block}
"""


def _topology_comparison(manifest: dict) -> str:
    parity_rows = [
        row for row in read_csv_rows(DATA_ROOT / "topology" / "direct_comparison.csv")
        if row["status"] == "completed"
    ]
    parity_table = markdown_table(
        ["implementation", "ranks", "status", "compliance", "volume", "wall [s]"],
        [
            [
                implementation_style(row["implementation"])["label"],
                format_int(row["mpi_ranks"]),
                row["status"],
                format_float(row["median_final_compliance"], 4),
                format_float(row["median_final_volume_fraction"], 4),
                format_float(row["median_wall_time_s"], 3),
            ]
            for row in parity_rows
        ],
    )
    direct_status_table = _table_from_csv(
        DATA_ROOT / "topology" / "direct_comparison.csv",
        ["implementation", "ranks", "status", "wall [s]", "compliance", "volume"],
        lambda row: [
            implementation_style(row["implementation"])["label"],
            format_int(row["mpi_ranks"]),
            row["status"],
            format_float(row["median_wall_time_s"], 3),
            format_float(row["median_final_compliance"], 4),
            format_float(row["median_final_volume_fraction"], 4),
        ],
    )
    strong_scaling_table = _table_from_csv(
        DATA_ROOT / "topology" / "strong_scaling.csv",
        ["ranks", "result", "outer", "p", "volume", "compliance", "wall [s]", "speedup"],
        lambda row: [
            format_int(row["ranks"]),
            row["result"],
            format_int(row["outer_iterations"]),
            format_float(row["final_p_penal"], 2),
            format_float(row["final_volume_fraction"], 4),
            format_float(row["final_compliance"], 4),
            format_float(row["wall_time_s"], 3),
            format_float(row["speedup_vs_1"], 3),
        ],
    )
    mesh_table = _table_from_csv(
        DATA_ROOT / "topology" / "mesh_timing.csv",
        ["mesh", "free DOFs", "ranks", "result", "outer", "compliance", "volume", "wall [s]"],
        lambda row: [
            row["mesh_label"],
            format_int(row["problem_size"]),
            format_int(row["nprocs"]),
            row["result"],
            format_int(row["outer_iterations"]),
            format_float(row["final_compliance"], 4),
            format_float(row["final_volume_fraction"], 4),
            format_float(row["wall_time_s"], 3),
        ],
    )
    commands = [
        _topology_source_command("serial_command"),
        _topology_source_command("parallel_final_command"),
        _topology_source_command("parallel_scaling_command"),
        *_topology_mesh_scaling_commands(),
        _provenance_command(DATA_ROOT / "topology" / "build_topology_data.provenance.json"),
        _provenance_command(DATA_ROOT / "topology" / "topology_strong_scaling.provenance.json"),
        _provenance_command(DATA_ROOT / "topology" / "topology_mesh_timing.provenance.json"),
    ]
    commands_block = _commands_block(commands)
    return f"""# Topology Implementation Comparison

## Maintained Implementation Roster

| implementation | comparison role |
| --- | --- |
| pure JAX serial | shared-case parity reference on `192 x 96` |
| parallel JAX+PETSc | direct-status comparison on `192 x 96`; fine-grid scaling on `768 x 384` |

## Shared-Case Result Equivalence

Only the serial pure-JAX implementation completes the maintained shared
`192 x 96` direct-comparison case. The parallel implementation is therefore
excluded from the parity table and reported in the status table below instead.

Completed shared-case results:

{parity_table}

## Direct Comparison Status Table

{direct_status_table}

## Scaling And Speed Comparison

{_inline_figure('img/png/topology/topology_strong_scaling.png', 'img/pdf/topology/topology_strong_scaling.pdf', 'Topology finest-mesh strong scaling preview', 'Topology strong-scaling PDF')}

{_inline_figure('img/png/topology/topology_mesh_timing.png', 'img/pdf/topology/topology_mesh_timing.pdf', 'Topology time-vs-mesh-size preview', 'Topology time-vs-mesh-size PDF')}
- raw direct comparison source: `replications/2026-03-16_maintained_refresh/comparisons/topology/direct_speed.csv`
- maintained fine-grid scaling source: `replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/scaling_summary.csv`
- maintained fixed-rank mesh-size sweep source: `overview/img/runs/topology/mesh_scaling`

Fine-grid parallel scaling (`768 x 384`):

{strong_scaling_table}

Fixed-rank mesh-size timing (`8` ranks):

{mesh_table}

## Notes On Exclusions

- The parallel JAX+PETSc path is excluded from the shared-case parity table
  because the maintained `192 x 96` direct-comparison runs terminate at
  `max_outer_iterations` for ranks `1`, `2`, and `4`.
- The parallel path is still the maintained fine-grid benchmark and scaling
  implementation; its validated `768 x 384` results are reported here
  separately from the serial reference parity case.

## Raw Outputs And Figures

- serial reference: `replications/2026-03-16_maintained_refresh/runs/topology/serial_reference`
- parallel final: `replications/2026-03-16_maintained_refresh/runs/topology/parallel_final`
- parallel scaling: `replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling`
- curated figures: `overview/img/pdf/topology/` and `overview/img/png/topology/`

## Commands Used

{commands_block}
"""


def _commands_md(manifest: dict) -> str:
    lines = [
        "# Overview Command Catalog",
        "",
        "These commands produced the publication overview package under `overview/`.",
        "",
        "## Publication Reruns",
        "",
        f"```bash\n{_provenance_command(OVERVIEW_ROOT / 'img' / 'runs' / 'build_publication_reruns.provenance.json')}\n```",
        "",
        "## Data Extraction",
        "",
    ]
    for family in ("plaplace", "ginzburg_landau", "hyperelasticity", "topology"):
        lines.extend(
            [
                f"### {FAMILY_TITLES[family]}",
                "",
                f"```bash\n{_provenance_command(DATA_ROOT / family / f'build_{family}_data.provenance.json')}\n```",
                "",
            ]
        )
    lines.extend(["## Figure Generation", ""])
    figure_provenance = {
        "plaplace": DATA_ROOT / "plaplace" / "plaplace_strong_scaling.provenance.json",
        "ginzburg_landau": DATA_ROOT / "ginzburg_landau" / "ginzburg_landau_strong_scaling.provenance.json",
        "hyperelasticity": DATA_ROOT / "hyperelasticity" / "hyperelasticity_strong_scaling.provenance.json",
        "topology": DATA_ROOT / "topology" / "topology_strong_scaling.provenance.json",
    }
    for family, provenance_path in figure_provenance.items():
        lines.extend(
            [
                f"### {FAMILY_TITLES[family]}",
                "",
                f"```bash\n{_provenance_command(provenance_path)}\n```",
                "",
            ]
        )
    lines.extend(
        [
            "## Overview Page Generation",
            "",
            f"```bash\n{_build_pages_command()}\n```",
            "",
            "## Source Replication Campaign",
            "",
            f"- replication root: `{repo_rel(REPLICATION_ROOT)}`",
            f"- pLaplace suite: `{_manifest_command(manifest, 'plaplace_final_suite')}`",
            f"- GinzburgLandau suite: `{_manifest_command(manifest, 'gl_final_suite')}`",
            f"- HyperElasticity MPI suite: `{_manifest_command(manifest, 'he_final_suite_best')}`",
            f"- HyperElasticity pure-JAX suite: `{_manifest_command(manifest, 'he_pure_jax_suite_best')}`",
            f"- topology serial report: `{_topology_source_command('serial_command')}`",
            f"- topology parallel final report: `{_topology_source_command('parallel_final_command')}`",
            f"- topology parallel scaling report: `{_topology_source_command('parallel_scaling_command')}`",
        ]
    )
    return "\n".join(lines)


def _index_md() -> str:
    return """# Overview Index

This folder is the publication-style summary layer built on top of the finished
maintained replication campaign in `replications/2026-03-16_maintained_refresh/`.

## Problem Overviews

- [pLaplace problem](pLaplace_problem.md)
- [GinzburgLandau problem](GinzburgLandau_problem.md)
- [HyperElasticity problem](HyperElasticity_problem.md)
- [Topology problem](Topology_problem.md)

## Implementation Comparisons

- [pLaplace comparison](pLaplace_comparison.md)
- [GinzburgLandau comparison](GinzburgLandau_comparison.md)
- [HyperElasticity comparison](HyperElasticity_comparison.md)
- [Topology comparison](Topology_comparison.md)

## Build And Provenance

- [command catalog](commands.md)
- PDF assets: `overview/img/pdf/`
- PNG previews: `overview/img/png/`
- figure source data: `overview/img/data/`
- publication reruns: `overview/img/runs/`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    manifest = _load_replication_manifest()

    _write(OVERVIEW_ROOT / "pLaplace_problem.md", _plaplace_problem(manifest))
    _write(OVERVIEW_ROOT / "pLaplace_comparison.md", _plaplace_comparison(manifest))
    _write(OVERVIEW_ROOT / "GinzburgLandau_problem.md", _gl_problem(manifest))
    _write(OVERVIEW_ROOT / "GinzburgLandau_comparison.md", _gl_comparison(manifest))
    _write(OVERVIEW_ROOT / "HyperElasticity_problem.md", _he_problem(manifest))
    _write(OVERVIEW_ROOT / "HyperElasticity_comparison.md", _he_comparison(manifest))
    _write(OVERVIEW_ROOT / "Topology_problem.md", _topology_problem(manifest))
    _write(OVERVIEW_ROOT / "Topology_comparison.md", _topology_comparison(manifest))
    _write(OVERVIEW_ROOT / "commands.md", _commands_md(manifest))
    _write(OVERVIEW_ROOT / "index.md", _index_md())
    record_provenance(
        OVERVIEW_ROOT / "build_overview_pages.provenance.json",
        script_name="overview/img/scripts/build_overview_pages.py",
        inputs=[
            repo_rel(DATA_ROOT / "plaplace"),
            repo_rel(DATA_ROOT / "ginzburg_landau"),
            repo_rel(DATA_ROOT / "hyperelasticity"),
            repo_rel(DATA_ROOT / "topology"),
            repo_rel(REPLICATION_ROOT / "manifest.json"),
        ],
        outputs=[
            repo_rel(OVERVIEW_ROOT / "index.md"),
            repo_rel(OVERVIEW_ROOT / "commands.md"),
            repo_rel(OVERVIEW_ROOT / "pLaplace_problem.md"),
            repo_rel(OVERVIEW_ROOT / "pLaplace_comparison.md"),
            repo_rel(OVERVIEW_ROOT / "GinzburgLandau_problem.md"),
            repo_rel(OVERVIEW_ROOT / "GinzburgLandau_comparison.md"),
            repo_rel(OVERVIEW_ROOT / "HyperElasticity_problem.md"),
            repo_rel(OVERVIEW_ROOT / "HyperElasticity_comparison.md"),
            repo_rel(OVERVIEW_ROOT / "Topology_problem.md"),
            repo_rel(OVERVIEW_ROOT / "Topology_comparison.md"),
        ],
        notes="Generates the publication-ready overview markdown package.",
    )


if __name__ == "__main__":
    main()
