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

| Solver                             | Location                                                       | Parallelism |
| ---------------------------------- | -------------------------------------------------------------- | ----------- |
| **Custom Newton** (recommended)    | `GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py`        | MPI         |
| **SNES Newton**                    | `GinzburgLandau2D_fenics/solve_GL_snes_newton.py`              | MPI         |
| **JAX Newton** (auto-diff + PyAMG) | `GinzburgLandau2D_jax/` + `example_GinzburgLandau2D_jax.ipynb` | single CPU  |

The **SNES Newton** uses trust-region Newton (`newtontr`) with FGMRES + ASM/ILU preconditioning and loose inner tolerance (`ksp_rtol=1e-1`). This was the only SNES configuration found to be reliable across all mesh sizes and MPI decompositions (see [results_GinzburgLandau2D.md](results_GinzburgLandau2D.md) for the full configuration survey). The **Custom Newton** uses a golden-section energy line search and remains the fastest option (6 iterations at all levels).

Benchmark results: [results_GinzburgLandau2D.md](results_GinzburgLandau2D.md)
How to run: [instructions.md](instructions.md)

## Problem: HyperElasticity 3D (current status)

The HyperElasticity setup uses a rotating right boundary and Neo-Hookean-type energy. Current benchmark and debugging status is tracked in:

- [results_HyperElasticity3D.md](results_HyperElasticity3D.md)

### Level-1 custom Newton result (updated)

- With corrected JAX->FEniCS restart mapping, custom FEniCS converges at step 20 from JAX step 19 restart:
	- energy `137.392333`, Newton iters `22`
	- artifact: [experiment_scripts/he_custom_restart_step20_maxit1000_fixedinit.json](experiment_scripts/he_custom_restart_step20_maxit1000_fixedinit.json)

### Step-24 inner-precision sweep (`ksp_rtol = 1e-1 ... 1e-6`)

For level 1, step 24 (restart from step 23), custom Newton was profiled with per-iteration convergence history.

Sweep artifacts:
- Summary table: [experiment_scripts/he_step24_precision_sweep/step24_precision_summary.md](experiment_scripts/he_step24_precision_sweep/step24_precision_summary.md)
- Full JSON: [experiment_scripts/he_step24_precision_sweep/step24_precision_summary.json](experiment_scripts/he_step24_precision_sweep/step24_precision_summary.json)
- Convergence profiles (all iterations): [experiment_scripts/he_step24_precision_sweep/step24_convergence_profiles.csv](experiment_scripts/he_step24_precision_sweep/step24_convergence_profiles.csv)

Observed trend:
- `CG + HYPRE` fails for all tested `ksp_rtol` values (`1e-1 ... 1e-6`) at step 24.
- `GMRES + HYPRE` converges for all tested `ksp_rtol` values, with final energies around `197.7484` (close to JAX reference `197.748635`).
- This indicates late-step robustness is dominated by linear solver choice, not only tolerance tightening.

### Step-24 detailed settings used (for current diagnosis)

This section captures the exact settings for the **single-step-24** examination (level 1, restart from step 23).

Current fast variant used in latest reruns (step-24 and full trajectory):
- Linear solver: `gmres + hypre (boomeramg)`
- `ksp_rtol = 1e-1`
- `ksp_max_it = 30`
- skip explicit `hypre_nodal_coarsen` / `hypre_vec_interp_variant` (HYPRE defaults)
- `--pc_setup_on_ksp_cap` enabled (PC setup only after previous solve hits cap; first solve always sets up)

#### Custom FEniCS (PETSc) — settings

- Nonlinear solver: custom Newton (`tools_petsc4py/minimizers.py`)
- Outer tolerances:
	- energy change tolerance `tolf = 1e-4`
	- gradient norm tolerance `tolg = 1e-3`
	- max Newton iterations `maxit = 300` (for sweep runs)
- Line search:
	- method: golden-section
	- tolerance `linesearch_tol = 1e-3`
	- interval `[-0.5, 2.0]`
	- non-finite trial energies are treated as `+inf` (guard enabled)
- Linear solver (inner):
	- KSP type: `gmres` (or `cg` in failing comparisons)
	- PC type: `hypre` (`boomeramg`)
	- `ksp_rtol` swept over `1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6`
	- latest skip-variant reruns use `ksp_rtol = 1e-1`
	- HYPRE options: `pc_hypre_boomeramg_nodal_coarsen=6`, `pc_hypre_boomeramg_vec_interp_variant=3`
	- latest skip-variant reruns set `ksp_max_it = 30` (cap-triggered setup mode)
	- earlier sweeps used larger caps (including observed `10000` cap-hit behavior)

#### JAX reference — settings (for comparability)

- Nonlinear solver: `tools/minimizers.py::newton`
- Outer tolerances:
	- energy change tolerance `tolf = 1e-4`
	- gradient norm tolerance `tolg = 1e-3` (default)
	- max Newton iterations `maxit = 100`
- Line search:
	- method: golden-section
	- tolerance `linesearch_tol = 1e-3`
	- interval `[-0.5, 2.0]`
- Linear solver (inner):
	- Krylov: SciPy `cg`
	- preconditioner: PyAMG smoothed aggregation (`smooth='energy'`)
	- tolerance `tol = 1e-3`
	- max inner iterations `maxiter = 100`

#### Why step 24 is slow (measured)

For `gmres+hypre` at step 24:

- `ksp_rtol = 1e-3`:
	- wall time `70.1249 s`, Newton iterations `23`
	- total inner iterations across Newton steps: `43944`
	- average inner iterations per Newton step: `1910.61`
	- Newton steps hitting inner cap (`ksp_its = 10000`): `4`
- `ksp_rtol = 1e-6`:
	- wall time `90.1326 s`, Newton iterations `23`
	- total inner iterations: `57177`
	- average inner iterations per Newton step: `2485.96`
	- Newton steps hitting cap: `4`

Interpretation:
- The long runtime is dominated by expensive inner GMRES solves in several Newton steps (including cap hits), not by a large number of outer Newton iterations.
- Tightening `ksp_rtol` from `1e-3` to `1e-6` increases total inner work substantially and therefore runtime.

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
