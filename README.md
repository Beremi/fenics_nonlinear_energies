# Nonlinear Energies — FEniCS & JAX Solvers

This repository contains solvers for nonlinear energy minimization problems using both **FEniCSx (DOLFINx)** and **JAX**, with infrastructure for reproducible benchmarking.

## Quick Start — Example Notebooks

The best way to explore the solvers is through the **Jupyter notebooks in the repository root**:

| Notebook                                                                   | Problem            | Framework        |
| -------------------------------------------------------------------------- | ------------------ | ---------------- |
| [example_pLaplace2D_jax.ipynb](example_pLaplace2D_jax.ipynb)               | p-Laplacian 2D     | JAX              |
| [example_GinzburgLandau2D_jax.ipynb](example_GinzburgLandau2D_jax.ipynb)   | Ginzburg-Landau 2D | JAX              |
| [example_HyperElasticity3D_jax.ipynb](example_HyperElasticity3D_jax.ipynb) | Hyperelasticity 3D | JAX              |
| [benchmark_pLaplace2D_fenics.ipynb](benchmark_pLaplace2D_fenics.ipynb)     | p-Laplacian 2D     | FEniCS (DOLFINx) |

Open in VS Code devcontainer and run — everything is pre-configured.

## Problem: p-Laplacian 2D

$$\min_u J(u) = \int_\Omega \frac{1}{p} |\nabla u|^p \, dx - \int_\Omega f \cdot u \, dx, \quad u|_{\partial\Omega} = 0$$

with $p = 3$, $f = -10$ on the unit square. Three solver variants are provided:

| Solver                             | Location                                                | Parallelism |
| ---------------------------------- | ------------------------------------------------------- | ----------- |
| **SNES Newton** (recommended)      | `pLaplace2D_fenics/solve_pLaplace_snes_newton.py`       | MPI         |
| **Custom Newton** (JAX algorithm)  | `pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py` | MPI         |
| **JAX Newton** (auto-diff + PyAMG) | `pLaplace2D_jax/solve_pLaplace_jax_newton.py`           | single CPU  |

The **Custom Newton** re-implements the JAX minimisation algorithm (golden-section on [−0.5, 2], tighter tolerances) on top of PETSc, matching JAX iteration counts while supporting MPI parallelism.

Benchmark results: [results_pLaplace.md](results_pLaplace.md)
How to run: [instructions.md](instructions.md)

## Problem: Ginzburg-Landau 2D

$$\min_u J(u) = \int_\Omega \frac{\varepsilon}{2} |\nabla u|^2 + \frac{1}{4}(u^2 - 1)^2 \, dx, \quad u|_{\partial\Omega} = 0$$

with $\varepsilon = 0.01$ on $[-1,1]^2$. This is a **non-convex** energy (indefinite Hessian). Three solver variants are provided:

| Solver                             | Location                                                    | Parallelism |
| ---------------------------------- | ----------------------------------------------------------- | ----------- |
| **Custom Newton** (recommended)    | `GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py`     | MPI         |
| **SNES Newton**                    | `GinzburgLandau2D_fenics/solve_GL_snes_newton.py`           | MPI         |
| **JAX Newton** (auto-diff + PyAMG) | `GinzburgLandau2D_jax/` + `example_GinzburgLandau2D_jax.ipynb` | single CPU  |

The **SNES Newton** uses trust-region Newton (`newtontr`) with FGMRES + ASM/ILU preconditioning and loose inner tolerance (`ksp_rtol=1e-1`). This was the only SNES configuration found to be reliable across all mesh sizes and MPI decompositions (see [results_GinzburgLandau2D.md](results_GinzburgLandau2D.md) for the full configuration survey). The **Custom Newton** uses a golden-section energy line search and remains the fastest option (6 iterations at all levels).

Benchmark results: [results_GinzburgLandau2D.md](results_GinzburgLandau2D.md)
How to run: [instructions.md](instructions.md)

## Prerequisites

FEniCS solvers require **DOLFINx >= 0.10** with PETSc. JAX solvers require **JAX**, **h5py**, **PyAMG**. The included devcontainer provides everything — see [instructions.md](instructions.md).

## Repository Structure

```
.
├── example_pLaplace2D_jax.ipynb       # ★ JAX p-Laplace example notebook
├── example_GinzburgLandau2D_jax.ipynb # ★ JAX Ginzburg-Landau example notebook
├── example_HyperElasticity3D_jax.ipynb# ★ JAX Hyperelasticity example notebook
├── benchmark_pLaplace2D_fenics.ipynb  # ★ FEniCS p-Laplace benchmark notebook
│
├── README.md                          # This file
├── instructions.md                    # How to run, store results, generate tables
├── results_pLaplace.md                # Compiled p-Laplace benchmark results
├── results_GinzburgLandau2D.md        # Compiled Ginzburg-Landau benchmark results
│
├── mesh_data/                         # Shared mesh files (all problems)
│   ├── pLaplace/                      #   HDF5 source + FEniCS XDMF, levels 1–9
│   ├── GinzburgLandau/                #   HDF5, levels 2–9
│   └── HyperElasticity/              #   HDF5, levels 1–4
│
├── pLaplace2D_fenics/                 # FEniCS solvers for p-Laplace
│   ├── solve_pLaplace_snes_newton.py  #   SNES Newton (DOLFINx 0.10+)
│   ├── solve_pLaplace_custom_jaxversion.py # Custom Newton (JAX algorithm via PETSc)
│   └── export_pLaplace_meshes.py      #   HDF5 → XDMF mesh converter
│
├── pLaplace2D_jax/                    # JAX solver for p-Laplace
│   ├── jax_energy.py                  #   Energy functional in JAX
│   ├── mesh.py                        #   Mesh loader (HDF5 → JAX arrays)
│   └── solve_pLaplace_jax_newton.py   #   Benchmark script
│
├── GinzburgLandau2D_jax/              # JAX solver for Ginzburg-Landau 2D
│   └── jax_energy.py, mesh.py         #   Energy + mesh loader
│
├── GinzburgLandau2D_fenics/           # FEniCS solvers for Ginzburg-Landau 2D
│   ├── solve_GL_snes_newton.py        #   SNES Newton (newtontr + ASM/ILU)
│   ├── solve_GL_custom_jaxversion.py  #   Custom Newton (recommended, uses tools_petsc4py)
│   └── export_GL_meshes.py            #   HDF5 → XDMF mesh converter
│
├── HyperElasticity3D_jax/             # JAX solver for Hyperelasticity 3D
│   ├── jax_energy.py, mesh.py         #   Energy + mesh loader
│   └── rotate_boundary.py             #   Boundary rotation utility
│
├── tools/                             # Shared JAX utilities
│   ├── jax_diff.py                    #   Auto-diff + sparse Hessian assembly
│   ├── minimizers.py                  #   Newton solver with line search
│   ├── sparse_solvers.py              #   PyAMG / direct linear solvers
│   └── graph_sfd.py                   #   Graph coloring for sparse finite differences
│
├── tools_petsc4py/                    # Shared PETSc Newton utilities
│   └── minimizers.py                  #   Newton solver (PETSc Vec interface)
│
├── results/                           # p-Laplace experiment results + processing scripts
│   ├── run_experiments.py             #   Automated FEniCS benchmark runner
│   ├── generate_latex_tables.py       #   Generate LaTeX/Markdown tables from results
│   ├── generate_scaling_plot.py       #   Generate strong scaling plots
│   └── experiment_001/                #   Stored benchmark results (JSON + tables + plots)
│
├── results_GL/                        # Ginzburg-Landau experiment results
│   ├── run_experiments.py             #   Automated GL benchmark runner
│   └── experiment_001/                #   Stored benchmark results (JSON)
│
└── .devcontainer/                     # Docker/devcontainer configuration
```

## Mesh Levels

The meshes in `mesh_data/pLaplace/` (levels 1–9) are pre-generated. The paper table uses levels 4–8, which correspond to **mesh files 5–9**:

| Table Level |  Mesh File   | Total DOFs | Free DOFs |
| :---------: | :----------: | :--------: | :-------: |
|      4      | mesh_level_5 |    3201    |   2945    |
|      5      | mesh_level_6 |   12545    |   12033   |
|      6      | mesh_level_7 |   49665    |   48641   |
|      7      | mesh_level_8 |   197633   |  195585   |
|      8      | mesh_level_9 |   788481   |  784385   |

To regenerate FEniCS XDMF meshes from HDF5 source: `python3 pLaplace2D_fenics/export_pLaplace_meshes.py`
