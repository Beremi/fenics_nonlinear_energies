# FEniCS Nonlinear Energies

This repository contains FEniCSx (DOLFINx) solvers for nonlinear energy minimization problems and infrastructure for reproducible benchmarking.

## Problem: p-Laplacian 2D

We solve the p-Laplacian problem ($p = 3$) on the unit square with homogeneous Dirichlet boundary conditions:

$$\min_u J(u) = \int_\Omega \frac{1}{p} |\nabla u|^p \, dx - \int_\Omega f \cdot u \, dx, \quad u|_{\partial\Omega} = 0$$

with $f = -10$. Two solver variants are provided — see [Solver Scripts](#solver-scripts) below.

Benchmark results are in [results_pLaplace.md](results_pLaplace.md).
Instructions for running solvers, storing results, and generating tables are in [instructions.md](instructions.md).

## Prerequisites

All solvers require **DOLFINx >= 0.10** with PETSc support. The easiest way is via the included devcontainer (see [instructions.md](instructions.md)).

## Solver Scripts

### `solve_pLaplace_snes_newton.py` — Built-in SNES Newton

Uses PETSc's SNES with Newton line search (`newtonls`), full steps (`basic`), CG + HYPRE AMG.
This is the **recommended solver** — matches the "FEniCS serial/parallel" columns in the paper.

### `solve_pLaplace_custom_newton.py` — Custom Newton with Line Search

Manual Newton method with golden-section line search on the energy functional, CG + HYPRE AMG.
Slower due to line search overhead but provides more control and debugging output.

### `solve_pLaplace_jax_newton.py` — JAX Newton (no MPI)

Pure-JAX solver using automatic differentiation for gradients, sparse finite differences (graph coloring) for Hessian assembly, and PyAMG smoothed-aggregation CG. Runs on a single CPU; no MPI parallelism. Uses the modules in `pLaplace2D/` and `tools/`.

## Repository Structure

```
.
├── README.md                          # This file
├── instructions.md                    # How to run, store results, generate tables
├── results_pLaplace.md                # Compiled benchmark results
│
├── solve_pLaplace_snes_newton.py      # Main solver: SNES Newton (DOLFINx 0.10+)
├── solve_pLaplace_custom_newton.py    # Custom Newton with line search (DOLFINx 0.10+)
├── solve_pLaplace_jax_newton.py       # JAX Newton solver (single CPU, no MPI)
├── run_experiments.py                 # Automated experiment runner
├── generate_latex_tables.py           # Generate LaTeX/Markdown tables from results
├── generate_scaling_plot.py           # Generate strong scaling plots from results
├── export_pLaplace_meshes.py          # Generate FEniCS mesh files from pLaplace2D data
│
├── pLaplace_fenics_mesh/              # Pre-generated mesh files (XDMF + HDF5), levels 1–9
├── pLaplace2D/                        # Mesh generation module (JAX-based mesh data)
├── results/                           # Experiment results (JSON + LaTeX tables)
│
├── GinzburgLandau2D/                  # Ginzburg-Landau 2D problem (JAX energy + mesh)
├── HyperElasticity3D/                 # Hyperelasticity 3D problem (JAX energy + mesh)
├── tools/                             # Shared utilities (JAX diff, minimizers, sparse solvers)
│
├── test_fenics_pLaplace.py            # Legacy: old DOLFINx API (for reference only)
├── test_fenics_pLaplace2.py           # Legacy: old DOLFINx API (for reference only)
└── .devcontainer/                     # Docker/devcontainer configuration
```

## Mesh Levels

The meshes in `pLaplace_fenics_mesh/` (levels 1–9) are pre-generated. The paper table uses levels 4–8, which correspond to **mesh files 5–9**:

| Table Level |  Mesh File   | Total DOFs | Free DOFs |
| :---------: | :----------: | :--------: | :-------: |
|      4      | mesh_level_5 |    3201    |   2945    |
|      5      | mesh_level_6 |   12545    |   12033   |
|      6      | mesh_level_7 |   49665    |   48641   |
|      7      | mesh_level_8 |   197633   |  195585   |
|      8      | mesh_level_9 |   788481   |  784385   |

To regenerate meshes: `python3 export_pLaplace_meshes.py`

## Legacy Scripts

The original scripts (`test_fenics_pLaplace.py`, `test_fenics_pLaplace2.py`) are kept for reference but use the **old DOLFINx API** (`u.vector` instead of `u.x.petsc_vec`, old `NonlinearProblem` / `NewtonSolver` constructors). They will **not run** with DOLFINx >= 0.10.
