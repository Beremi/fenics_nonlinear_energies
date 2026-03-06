# Nonlinear Energies — FEniCS & JAX Solvers

This repository contains solvers for nonlinear energy minimization problems using both **FEniCSx (DOLFINx)** and **JAX**, with infrastructure for reproducible benchmarking.

## Quick Start — Example Notebooks

The best way to explore the solvers is through the **Jupyter notebooks in the repository root**:

| Notebook                                                                             | Problem            | Framework        |
| ------------------------------------------------------------------------------------ | ------------------ | ---------------- |
| [example_pLaplace2D_jax.ipynb](example_pLaplace2D_jax.ipynb)                         | p-Laplacian 2D     | JAX              |
| [example_GinzburgLandau2D_jax.ipynb](example_GinzburgLandau2D_jax.ipynb)             | Ginzburg-Landau 2D | JAX              |
| [example_HyperElasticity3D_jax.ipynb](example_HyperElasticity3D_jax.ipynb)           | Hyperelasticity 3D | JAX              |
| [benchmark_pLaplace2D_fenics.ipynb](benchmark_pLaplace2D_fenics.ipynb)               | p-Laplacian 2D     | FEniCS (DOLFINx) |
| [benchmark_GinzburgLandau2D_fenics.ipynb](benchmark_GinzburgLandau2D_fenics.ipynb)   | Ginzburg-Landau 2D | FEniCS (DOLFINx) |
| [benchmark_HyperElasticity3D_fenics.ipynb](benchmark_HyperElasticity3D_fenics.ipynb) | Hyperelasticity 3D | FEniCS (DOLFINx) |

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

## Problem: HyperElasticity 3D

Neo-Hookean energy on a 3D beam with a rotating right-face boundary condition (0°–360° in 96 quarter-degree steps). FEniCS and JAX+PETSc solver variants are provided:

| Solver                          | Location                                                 | Status        |
| ------------------------------- | -------------------------------------------------------- | ------------- |
| **Custom Newton** (recommended) | `HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py` | 96/96 steps ✓ |
| **SNES Newton**                 | `HyperElasticity3D_fenics/solve_HE_snes_newton.py`       | 93/96 steps ✓ |
| **JAX+PETSc Custom Newton**     | `HyperElasticity3D_jax_petsc/solve_HE_dof.py`           | MPI parallel  |

FEniCS solvers use:
- GMRES + AMG preconditioner (HYPRE BoomerAMG or PETSc GAMG)
- Near-nullspace: 6 rigid-body modes (3 translations + 3 rotations)
- `ksp_rtol = 1e-1`

The **Custom Newton** uses a golden-section energy line search (`tools_petsc4py/minimizers.py`) and
`--pc_setup_on_ksp_cap` to reuse the AMG preconditioner across Newton steps.

Current practical tuning notes:

- large HE cases strongly prefer a forward-only line-search interval `[0, 1]`
- if the optional trust-region path is enabled, large-case HE currently works best with
  `trust_radius_init` around `0.5` to `1.0`
- small-case and large-case trust-region settings are not the same

See [TRUST_REGION_LINESEARCH_TUNING.md](TRUST_REGION_LINESEARCH_TUNING.md) for the current tested
settings and large-case sweep results.

Recent update: the PETSc minimizer was hardened against false convergence / NaN propagation.
The HE path now uses a 3-part nonlinear stop criterion (energy + step + gradient), non-finite
rollback checks, and per-step repair retries with tighter linear settings on failure.
See [results_HyperElasticity3D.md](results_HyperElasticity3D.md) Annex F.8–F.10 for the full
before/after sweep comparisons and tolerance sensitivity (`tolg_rel=1e-3`, `1e-4`, `1e-2`).

**Preconditioner comparison (level 3, 78k DOFs, 16 MPI, 24 load steps):**

| PC                                            | Total time | Newton iters | KSP iters |
| --------------------------------------------- | ---------: | -----------: | --------: |
| GAMG (`--pc_type gamg --gamg_threshold 0.05`) | **62.4 s** |        1,123 |    17,868 |
| HYPRE BoomerAMG (`--pc_type hypre`)           |    135.5 s |          669 |    10,347 |

GAMG with `pc_gamg_threshold=0.05` is **2.2× faster** than HYPRE for this problem. The threshold
is critical for correctness — without it, GAMG converges to wrong solutions for 3D elasticity.
See [results_HyperElasticity3D.md, Annex F](results_HyperElasticity3D.md#annex-f-gamg-vs-hypre-preconditioner-comparison-level-3-16-mpi-processes)
for the full investigation.

### SNES+GAMG continuation check (level 3, 16 MPI)

Using the best SNES+GAMG base setting found in the latest sweep:

- `--snes_type newtonls --linesearch basic`
- `--ksp_type fgmres --pc_type gamg`
- `--ksp_rtol 1e-1 --ksp_max_it 2000 --snes_atol 1e-3`
- `PETSC_OPTIONS='-he_pc_gamg_threshold 0.05 -he_pc_gamg_agg_nsmooths 1'`
- near-nullspace ON (default), `--stop_on_fail`

We then reduced the load increment by increasing the number of steps while keeping the same horizon (`--total_steps = --steps`):

| Requested steps | Recorded | Converged | First fail step | First fail angle [rad] | Reason | Time [s] | Newton | Linear |
| --------------: | -------: | --------: | --------------: | ---------------------: | -----: | -------: | -----: | -----: |
|          `96/96` |       60 |        59 |              60 |              15.707963 |     -3 |    78.69 |   1092 |  48696 |
|        `192/192` |      119 |       118 |             119 |              15.577064 |     -3 |   131.68 |   1820 |  82410 |
|        `384/384` |      237 |       236 |             237 |              15.511614 |     -3 |   225.93 |   3070 | 138544 |

Finding: smaller increments significantly delay failure in step index, but all tested runs still end with `SNES_DIVERGED_LINEAR_SOLVE` (`reason=-3`) near the same physical angle.

Artifacts:
- [experiment_scripts/he_snes_l3_np16_gamg_96_fgmres_r1e1_k2000_a1e3_basic.json](experiment_scripts/he_snes_l3_np16_gamg_96_fgmres_r1e1_k2000_a1e3_basic.json)
- [experiment_scripts/he_snes_l3_np16_gamg_192_fgmres_base.json](experiment_scripts/he_snes_l3_np16_gamg_192_fgmres_base.json)
- [experiment_scripts/he_snes_l3_np16_gamg_384_fgmres_base.json](experiment_scripts/he_snes_l3_np16_gamg_384_fgmres_base.json)

Run template (set `<N>=96|192|384`):
```bash
docker exec -u ubuntu bench_container bash -lc '
cd /workdir &&
PETSC_OPTIONS="-he_pc_gamg_threshold 0.05 -he_pc_gamg_agg_nsmooths 1" \
mpirun -n 16 python3 HyperElasticity3D_fenics/solve_HE_snes_newton.py \
  --level 3 --steps <N> --total_steps <N> --stop_on_fail --quiet \
  --snes_type newtonls --linesearch basic \
  --pc_type gamg --ksp_type fgmres \
  --ksp_rtol 1e-1 --ksp_max_it 2000 --snes_atol 1e-3 \
  --out experiment_scripts/he_snes_l3_np16_gamg_<N>_fgmres_base.json
'
```

### How to run (96 quarter-steps, level 1, single process)

**Custom Newton:**
```bash
python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
    --level 1 --steps 96 --total_steps 96 \
    --ksp_rtol 1e-1 --ksp_max_it 30 \
    --pc_setup_on_ksp_cap \
    --quiet --out experiment_scripts/out_custom.json
```

**SNES Newton:**
```bash
python3 HyperElasticity3D_fenics/solve_HE_snes_newton.py \
    --level 1 --steps 96 \
    --ksp_type gmres --pc_type hypre \
    --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 \
    --quiet --out experiment_scripts/out_snes.json
```

**With Docker (from repository root):**
```bash
# Serial (no special flags needed)
docker run --rm --entrypoint "" -v "$PWD":/work -w /work fenics_test \
    python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
    --level 1 --steps 96 --total_steps 96 \
    --ksp_rtol 1e-1 --ksp_max_it 30 --pc_setup_on_ksp_cap \
    --quiet --out experiment_scripts/out_custom.json

# Parallel — MUST use --shm-size for multi-process MPI (MPICH needs shared memory)
docker run --rm --shm-size=8g --entrypoint mpirun -v "$PWD":/work -w /work fenics_test \
    -n 16 python3 /work/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
    --level 3 --steps 24 --total_steps 24 \
    --ksp_rtol 1e-1 --ksp_max_it 30 --pc_setup_on_ksp_cap \
    --quiet --out experiment_scripts/out_custom.json
```

> **⚠ Docker shared-memory**: The image uses MPICH, which requires shared memory
> for inter-process communication. Docker defaults to 64 MB, which causes
> **SIGBUS (exit 135)** or OOM kills with ≥8 MPI processes. Always pass
> `--shm-size=8g` for parallel runs. See [instructions.md](instructions.md) for
> details and a persistent-container pattern for long benchmarks.

### Summary results (level 1, 96 quarter-steps)

| Solver        | Converged | Newton iters | KSP iters | Avg KSP/Newton | Wall time |
| ------------- | --------: | -----------: | --------: | -------------: | --------: |
| Custom Newton |     96/96 |         1209 |     24872 |           20.6 |    72.6 s |
| SNES Newton   |     93/96 |         1175 |     22490 |           19.1 |    15.0 s |

Steps 94–96 fail in the SNES solver due to AMG degradation at extreme deformation (the near-nullspace
is working correctly — confirmed by matching KSP/Newton ratio). See
[results_HyperElasticity3D.md](results_HyperElasticity3D.md) for full per-step tables and all
experimental details (Annexes A–D).

### JAX+PETSc solver for HyperElasticity 3D

`HyperElasticity3D_jax_petsc/solve_HE_dof.py` is an MPI-parallel re-implementation of the
Custom Newton solver using JAX automatic differentiation and PETSc linear algebra. It targets
full parity with the FEniCS custom solver.

#### Hessian assembly modes

The solver supports two Hessian assembly strategies selected via `--assembly_mode`:

| Mode | Flag | Method | Per-Newton-step cost |
| --- | --- | --- | --- |
| **SFD** (default) | `--assembly_mode sfd` | Graph coloring + HVPs | 63 sequential JAX HVP evaluations (= 63 graph colors for HE 3D vector P1) |
| **Element** | `--assembly_mode element` | Analytical element Hessians | 1 vmapped `jax.hessian` call over all local elements |

**SFD (Sparse Finite Differences):** builds a distance-2 graph coloring of the DOF adjacency,
then recovers each column group of the sparse Hessian via one Hessian-vector product (HVP) per
color. The HVP is computed as `jax.jvp(grad_energy, v, indicator)`. The coloring has 63 colors
for 3D vector P1 elements (vs 8 for 2D scalar P1 pLaplace), making this approach relatively
expensive for HE.

**Element assembly:** the production HE path now uses a dedicated reordered-overlap assembler.
It computes exact 12×12 element Hessians analytically via `jax.hessian(element_energy)`,
vmapped over all local elements in a single JIT-compiled call, but it also:

- reorders free DOFs before PETSc ownership is assigned,
- builds larger overlap subdomains from that ownership,
- rebuilds the current state by `Allgatherv`,
- assembles owned rows locally without a matrix-value reduction step.

This change was needed because the older HE element path still suffered from a
PETSc matrix ownership/layout regression even after the SFD coloring cost was
removed.

> **Implementation note:** PETSc's `MatSetPreallocationCOO` modifies the column index array
> in-place for MPIAIJ matrices (off-process column remapping). The element→COO position mapping
> must be built from the original adjacency arrays (`_row_adj`, `_col_adj`) before this
> modification, not from `_coo_cols` after it.

#### Current HE status

On the main fine check (`level 4`, `step 1 / 96`, `np=32`, `GMRES + GAMG`):

| Variant | Setup [s] | Step [s] | Assembly [s] | PC setup [s] | Solve [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| FEniCS custom | 0.192 | 5.142 | 1.141 | 0.299 | 2.774 |
| Old JAX+PETSc element | 6.722 | 16.397 | 2.163 | 1.267 | 9.550 |
| Production reordered element | 7.499 | 5.856 | 1.978 | 0.346 | 1.818 |

So the old HE JAX+PETSc solve gap is largely gone in the production element
path. The main remaining difference is end-to-end setup/JIT cost rather than
the linear solve itself.

Detailed notes:
- [investigation_jaxpetsc_performance_gap.md](investigation_jaxpetsc_performance_gap.md)
- [HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md](HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md)

#### How to run

```bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

# Element Hessian assembly (recommended — 2–3× faster than SFD)
mpirun -n 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py \
    --level 3 --steps 24 --total_steps 24 \
    --profile performance --assembly_mode element \
    --element_reorder_mode block_xyz --quiet

# SFD assembly (baseline)
mpirun -n 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py \
    --level 3 --steps 24 --total_steps 24 \
    --profile performance --assembly_mode sfd --quiet

# With detailed linear timing breakdown
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py \
    --level 4 --steps 1 --total_steps 96 \
    --profile performance --assembly_mode element \
    --element_reorder_mode block_xyz \
    --save_linear_timing --quiet
```

Benchmark results and investigation: [results_HyperElasticity3D.md](results_HyperElasticity3D.md),
[investigation_jaxpetsc_performance_gap.md](investigation_jaxpetsc_performance_gap.md),
[HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md](HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md)

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
├── HyperElasticity3D_jax_petsc/       # JAX + PETSc Hyperelasticity solver
│   ├── parallel_hessian_dof.py        #   Problem-specific assembler glue
│   ├── solver.py                      #   Solver logic
│   └── solve_HE_dof.py                #   CLI wrapper
│
├── HyperElasticity3D_petsc_support/   # Shared HE mesh / BC helpers
│   ├── mesh.py                        #   PETSc-compatible HE mesh loader
│   └── rotate_boundary.py             #   Boundary rotation utility
│
├── pLaplace2D_jax_petsc/              # JAX + PETSc p-Laplace solver
│   ├── parallel_hessian_dof.py        #   Problem-specific assembler glue
│   ├── solver.py                      #   Solver logic
│   └── solve_pLaplace_dof.py          #   CLI wrapper
│
├── pLaplace2D_petsc_support/          # Shared p-Laplace PETSc support
│   └── mesh.py                        #   PETSc-compatible p-Laplace mesh loader
│
├── tools/                             # Shared JAX utilities
│   ├── jax_diff.py                    #   Auto-diff + sparse Hessian assembly
│   ├── minimizers.py                  #   Newton solver with line search
│   ├── sparse_solvers.py              #   PyAMG / direct linear solvers
│   └── graph_sfd.py                   #   Graph coloring for sparse finite differences
│
├── tools_petsc4py/                    # Shared PETSc Newton utilities
│   ├── fenics_tools/                  #   FEniCS-side helpers
│   ├── jax_tools/                     #   Shared JAX + PETSc assembler layer
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
