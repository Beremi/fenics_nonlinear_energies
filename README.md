# Nonlinear Energies — FEniCS & JAX Solvers

This repository contains solvers for nonlinear energy minimization problems using both **FEniCSx (DOLFINx)** and **JAX**, with infrastructure for reproducible benchmarking.

## Canonical Layout

- `src/` — core solver implementations and shared backend infrastructure
- `experiments/` — benchmark runners, sweeps, diagnostics, and analysis scripts
- `notebooks/` — demos and benchmark walkthrough notebooks
- `docs/` — canonical setup, problem, result, implementation, and reference docs
- `data/meshes/` — checked-in mesh/problem inputs
- `artifacts/` — generated reproduction outputs, raw result caches, and figures

## Documentation

Current documentation now lives under `docs/`:

- index: [docs/README.md](docs/README.md)
- quickstart: [docs/setup/quickstart.md](docs/setup/quickstart.md)
- problem pages:
  - [docs/problems/pLaplace.md](docs/problems/pLaplace.md)
  - [docs/problems/pLaplace_u3_thesis_replications.md](docs/problems/pLaplace_u3_thesis_replications.md)
  - [docs/problems/GinzburgLandau.md](docs/problems/GinzburgLandau.md)
  - [docs/problems/HyperElasticity.md](docs/problems/HyperElasticity.md)
  - [docs/problems/Plasticity.md](docs/problems/Plasticity.md)
  - [docs/problems/Topology.md](docs/problems/Topology.md)
- current maintained results:
  - [docs/results/pLaplace.md](docs/results/pLaplace.md)
  - [docs/results/GinzburgLandau.md](docs/results/GinzburgLandau.md)
  - [docs/results/HyperElasticity.md](docs/results/HyperElasticity.md)
  - [docs/results/Plasticity.md](docs/results/Plasticity.md)
  - [docs/results/Topology.md](docs/results/Topology.md)

## Quick Start — Example Notebooks

The best way to explore the solvers is through the **Jupyter notebooks in `notebooks/`**:

| Notebook                                                                             | Problem            | Framework        |
| ------------------------------------------------------------------------------------ | ------------------ | ---------------- |
| [plaplace_jax_api.ipynb](notebooks/demos/plaplace_jax_api.ipynb) | p-Laplacian 2D | JAX |
| [ginzburg_landau_jax_api.ipynb](notebooks/demos/ginzburg_landau_jax_api.ipynb) | Ginzburg-Landau 2D | JAX |
| [hyperelasticity_jax_api.ipynb](notebooks/demos/hyperelasticity_jax_api.ipynb) | Hyperelasticity 3D | JAX |
| [plaplace_jax_petsc_api.ipynb](notebooks/demos/plaplace_jax_petsc_api.ipynb) | p-Laplacian 2D | JAX + PETSc |
| [plaplace_fenics_benchmark.ipynb](notebooks/benchmarks/plaplace_fenics_benchmark.ipynb) | p-Laplacian 2D | FEniCS (DOLFINx) |
| [ginzburg_landau_fenics_benchmark.ipynb](notebooks/benchmarks/ginzburg_landau_fenics_benchmark.ipynb) | Ginzburg-Landau 2D | FEniCS (DOLFINx) |
| [hyperelasticity_fenics_benchmark.ipynb](notebooks/benchmarks/hyperelasticity_fenics_benchmark.ipynb) | Hyperelasticity 3D | FEniCS (DOLFINx) |
| [topology_parallel_benchmark.ipynb](notebooks/benchmarks/topology_parallel_benchmark.ipynb) | Topology optimisation | JAX + PETSc |

Canonical implementation code lives under `src/core/` and `src/problems/`.
Use the canonical `src/problems/...` paths directly in scripts, notebooks, and
reproduction commands.

Open in VS Code devcontainer and run — everything is pre-configured.

## Problem: Mohr-Coulomb Plasticity 2D

The current Mohr-Coulomb plasticity implementation lives under
`src/problems/slope_stability/` because the benchmark geometry is a homogeneous
2D slope. The canonical docs surface now treats it as the repository's
plane-strain plasticity model, with a model page and a maintained results page
covering the deep-tail PMG hierarchy, backend context, and current large-scale
scaling campaign.

Model card: [docs/problems/Plasticity.md](docs/problems/Plasticity.md)  
Current maintained results: [docs/results/Plasticity.md](docs/results/Plasticity.md)

```bash
./.venv/bin/python -u src/problems/slope_stability/jax_petsc/solve_slope_stability_dof.py \
  --level 5 --elem_degree 4 --lambda-target 1.0 \
  --profile performance --pc_type mg \
  --mg_strategy same_mesh_p4_p2_p1_lminus1_p1 --mg_variant legacy_pmg \
  --ksp_type fgmres --ksp_rtol 1e-2 --ksp_max_it 100 \
  --save-linear-timing --no-use_trust_region --quiet \
  --out artifacts/raw_results/example_runs/mc_plasticity_p4_l5/output.json \
  --state-out artifacts/raw_results/example_runs/mc_plasticity_p4_l5/state.npz
```

## Problem: p-Laplacian 2D

$$\min_u J(u) = \int_\Omega \frac{1}{p} |\nabla u|^p \, dx - \int_\Omega f \cdot u \, dx, \quad u|_{\partial\Omega} = 0$$

with $p = 3$, $f = -10$ on the unit square. Three solver variants are provided:

| Solver                             | Location                                                | Parallelism |
| ---------------------------------- | ------------------------------------------------------- | ----------- |
| **SNES Newton** (recommended)      | `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`       | MPI         |
| **Custom Newton** (JAX algorithm)  | `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py` | MPI         |
| **JAX Newton** (auto-diff + PyAMG) | `src/problems/plaplace/jax/solve_pLaplace_jax_newton.py`           | single CPU  |

The **Custom Newton** re-implements the JAX minimisation algorithm (golden-section on [−0.5, 2], tighter tolerances) on top of PETSc, matching JAX iteration counts while supporting MPI parallelism.

Problem overview: [docs/problems/pLaplace.md](docs/problems/pLaplace.md)  
Current results: [docs/results/pLaplace.md](docs/results/pLaplace.md)  
How to run: [docs/setup/quickstart.md](docs/setup/quickstart.md)

## Problem: Ginzburg-Landau 2D

$$\min_u J(u) = \int_\Omega \frac{\varepsilon}{2} |\nabla u|^2 + \frac{1}{4}(u^2 - 1)^2 \, dx, \quad u|_{\partial\Omega} = 0$$

with $\varepsilon = 0.01$ on $[-1,1]^2$. This is a **non-convex** energy (indefinite Hessian). Three solver variants are provided:

| Solver                             | Location                                                       | Parallelism |
| ---------------------------------- | -------------------------------------------------------------- | ----------- |
| **Custom Newton** (recommended)    | `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`        | MPI         |
| **SNES Newton**                    | `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`              | MPI         |
| **JAX Newton** (auto-diff + PyAMG) | `src/problems/ginzburg_landau/jax/` + `notebooks/demos/ginzburg_landau_jax_api.ipynb` | single CPU  |

The **SNES Newton** uses trust-region Newton (`newtontr`) with FGMRES + ASM/ILU preconditioning and loose inner tolerance (`ksp_rtol=1e-1`). This was the only SNES configuration found to be reliable across all mesh sizes and MPI decompositions (see [results_GinzburgLandau2D.md](archive/results_GinzburgLandau2D.md) for the full configuration survey). The **Custom Newton** uses a golden-section energy line search and remains the fastest option (6 iterations at all levels).

Problem overview: [docs/problems/GinzburgLandau.md](docs/problems/GinzburgLandau.md)  
Current results: [docs/results/GinzburgLandau.md](docs/results/GinzburgLandau.md)  
How to run: [docs/setup/quickstart.md](docs/setup/quickstart.md)

## Problem: HyperElasticity 3D

Neo-Hookean energy on a 3D beam with a rotating right-face boundary condition
(0°–360° in 96 quarter-degree steps). FEniCS, pure JAX, and JAX+PETSc solver
variants are provided:

| Solver                          | Location                                                 | Status        |
| ------------------------------- | -------------------------------------------------------- | ------------- |
| **Custom Newton** (recommended) | `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py` | 96/96 steps ✓ |
| **SNES Newton**                 | `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`       | 93/96 steps ✓ |
| **Pure JAX Newton**             | `src/problems/hyperelasticity/jax/solve_HE_jax_newton.py`           | serial, levels 1–3 |
| **JAX+PETSc Custom Newton**     | `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`            | MPI parallel  |

Current default PETSc HE setting for both `fenics_custom` and
`jax_petsc_element`:

- `--ksp_type stcg --pc_type gamg`
- `--use_trust_region --trust_subproblem_line_search`
- `--linesearch_tol 1e-1`
- `--trust_radius_init 0.5`
- `--trust_shrink 0.5 --trust_expand 1.5`
- `--trust_eta_shrink 0.05 --trust_eta_expand 0.75`
- `--ksp_rtol 1e-1 --ksp_max_it 30`
- rebuild the PC every Newton iteration:
  leave `--pc_setup_on_ksp_cap` **off**
- near-nullspace ON
- GAMG coordinates ON

JAX + PETSc implementation defaults on top of that:

- `--assembly_mode element`
- `--element_reorder_mode block_xyz`
- `--local_hessian_mode element`
- `--local_coloring`

The backend-specific best radii from the STCG sweep were:

- `fenics_custom`: `trust_radius_init=1.0`
- `jax_petsc_element`: `trust_radius_init=0.5`

The shared campaign default is `0.5` because it is the best JAX setting, still
near-best on FEniCS, and avoids introducing a backend-specific nonlinear policy
into the like-for-like comparison.

See [docs/results/HyperElasticity.md](docs/results/HyperElasticity.md) for the current maintained benchmark
summary and [archive/docs/tuning/trust_region_linesearch_tuning.md](archive/docs/tuning/trust_region_linesearch_tuning.md)
for the archived tuning trail that led to this default.

Pure JAX serial HE now also uses the same outer trust-region policy through
`src/problems/hyperelasticity/jax/solve_HE_jax_newton.py`:

- serial Steihaug-Toint trust subproblem solve
- post trust-subproblem line search ON
- `linesearch_tol=1e-1`
- `trust_radius_init=0.5`
- `trust_shrink=0.5 --trust_expand=1.5`
- `trust_eta_shrink=0.05 --trust_eta_expand=0.75`
- PyAMG energy-smoothed aggregation preconditioner
- linear tolerance `1e-1`, linear max it `30`

It is used in the final HE report as the serial reference path up to level `3`.

Recent update: the PETSc minimizer was hardened against false convergence / NaN propagation.
The HE path now uses a 3-part nonlinear stop criterion (energy + step + gradient), non-finite
rollback checks, and per-step repair retries with tighter linear settings on failure.
See [results_HyperElasticity3D.md](archive/results_HyperElasticity3D.md) Annex F.8–F.10 for the full
before/after sweep comparisons and tolerance sensitivity (`tolg_rel=1e-3`, `1e-4`, `1e-2`).

**Preconditioner comparison (level 3, 78k DOFs, 16 MPI, 24 load steps):**

| PC                                            | Total time | Newton iters | KSP iters |
| --------------------------------------------- | ---------: | -----------: | --------: |
| GAMG (`--pc_type gamg --gamg_threshold 0.05`) | **62.4 s** |        1,123 |    17,868 |
| HYPRE BoomerAMG (`--pc_type hypre`)           |    135.5 s |          669 |    10,347 |

GAMG with `pc_gamg_threshold=0.05` is **2.2× faster** than HYPRE for this problem. The threshold
is critical for correctness — without it, GAMG converges to wrong solutions for 3D elasticity.
See [results_HyperElasticity3D.md, Annex F](archive/results_HyperElasticity3D.md#annex-f-gamg-vs-hypre-preconditioner-comparison-level-3-16-mpi-processes)
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
|          `96/96` |       60 |        59 |              60 |              15.707963 |     -3 |   118.47 |   1079 |  50010 |
|        `192/192` |      117 |       116 |             117 |              15.315264 |     -3 |   193.51 |   1783 |  78389 |
|        `384/384` |      236 |       235 |             236 |              15.446164 |     -3 |   320.65 |   3055 | 137445 |

Finding: smaller increments significantly delay failure in step index, but all tested runs still end with `SNES_DIVERGED_LINEAR_SOLVE` (`reason=-3`) near the same physical angle.

Archived sweep artifacts:
- [he_snes_l3_np16_gamg_96_fgmres_r1e1_k2000_a1e3_basic.json](experiments/legacy/he_snes_l3_np16_gamg_96_fgmres_r1e1_k2000_a1e3_basic.json)
- [he_snes_l3_np16_gamg_192_fgmres_base.json](experiments/legacy/he_snes_l3_np16_gamg_192_fgmres_base.json)
- [he_snes_l3_np16_gamg_384_fgmres_base.json](experiments/legacy/he_snes_l3_np16_gamg_384_fgmres_base.json)

Run template (set `<N>=96|192|384`):
```bash
PETSC_OPTIONS="-he_pc_gamg_threshold 0.05 -he_pc_gamg_agg_nsmooths 1" \
mpiexec -n 16 ./.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py \
  --level 3 --steps <N> --total_steps <N> --stop_on_fail --quiet \
  --snes_type newtonls --linesearch basic \
  --pc_type gamg --ksp_type fgmres \
  --ksp_rtol 1e-1 --ksp_max_it 2000 --snes_atol 1e-3 \
  --out artifacts/raw_results/example_runs/he_snes_l3_np16_gamg_<N>_fgmres_base.json
```

### How to run (96 quarter-steps, level 1, single process)

**Custom Newton, current default PETSc setting:**
```bash
python3 src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py \
    --level 1 --steps 96 --total_steps 96 \
    --ksp_type stcg --pc_type gamg \
    --ksp_rtol 1e-1 --ksp_max_it 30 \
    --use_trust_region --trust_subproblem_line_search \
    --linesearch_tol 1e-1 \
    --trust_radius_init 0.5 \
    --trust_shrink 0.5 --trust_expand 1.5 \
    --trust_eta_shrink 0.05 --trust_eta_expand 0.75 \
    --quiet --out artifacts/raw_results/example_runs/he/out_custom.json
```

**JAX + PETSc, current default PETSc setting:**
```bash
python3 src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py \
    --level 1 --steps 96 --total_steps 96 \
    --ksp_type stcg --pc_type gamg \
    --ksp_rtol 1e-1 --ksp_max_it 30 \
    --use_trust_region --trust_subproblem_line_search \
    --linesearch_tol 1e-1 \
    --trust_radius_init 0.5 \
    --trust_shrink 0.5 --trust_expand 1.5 \
    --trust_eta_shrink 0.05 --trust_eta_expand 0.75 \
    --assembly_mode element \
    --element_reorder_mode block_xyz \
    --local_hessian_mode element \
    --local_coloring \
    --quiet --out artifacts/raw_results/example_runs/he/out_jax_petsc.json
```

**Pure JAX, current serial reference setting:**
```bash
python3 src/problems/hyperelasticity/jax/solve_HE_jax_newton.py \
    --level 1 --steps 96 --total_steps 96 \
    --linesearch_tol 1e-1 \
    --trust_radius_init 0.5 \
    --trust_shrink 0.5 --trust_expand 1.5 \
    --trust_eta_shrink 0.05 --trust_eta_expand 0.75 \
    --ksp_rtol 1e-1 --ksp_max_it 30 \
    --quiet --out artifacts/raw_results/example_runs/he/out_jax_serial.json
```

**SNES Newton:**
```bash
python3 src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py \
    --level 1 --steps 96 \
    --ksp_type gmres --pc_type hypre \
    --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 \
    --quiet --out artifacts/raw_results/example_runs/he/out_snes.json
```

**With Docker (from repository root):**
```bash
# Serial (no special flags needed)
docker run --rm --entrypoint "" -v "$PWD":/repo -w /repo fenics_test \
    python3 src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py \
    --level 1 --steps 96 --total_steps 96 \
    --quiet --out artifacts/raw_results/example_runs/he/he_custom_serial.json

# Parallel — MUST use --shm-size for multi-process MPI (MPICH needs shared memory)
docker run --rm --shm-size=8g --entrypoint mpirun -v "$PWD":/repo -w /repo fenics_test \
    -n 16 python3 src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py \
    --level 3 --steps 24 --total_steps 24 \
    --quiet --out artifacts/raw_results/example_runs/he/he_custom_parallel.json
```

> **⚠ Docker shared-memory**: The image uses MPICH, which requires shared memory
> for inter-process communication. Docker defaults to 64 MB, which causes
> **SIGBUS (exit 135)** or OOM kills with ≥8 MPI processes. Always pass
> `--shm-size=8g` for parallel runs. See [docs/setup/quickstart.md](docs/setup/quickstart.md) for
> details and a persistent-container pattern for long benchmarks.

### Summary results (level 1, 96 quarter-steps)

| Solver        | Converged | Newton iters | KSP iters | Avg KSP/Newton | Wall time |
| ------------- | --------: | -----------: | --------: | -------------: | --------: |
| Custom Newton |     96/96 |         1209 |     24872 |           20.6 |    72.6 s |
| SNES Newton   |     93/96 |         1175 |     22490 |           19.1 |    15.0 s |

Steps 94–96 fail in the SNES solver due to AMG degradation at extreme deformation (the near-nullspace
is working correctly — confirmed by matching KSP/Newton ratio). See
[results_HyperElasticity3D.md](archive/results_HyperElasticity3D.md) for full per-step tables and all
experimental details (Annexes A–D).

### JAX+PETSc solver for HyperElasticity 3D

`src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py` is an MPI-parallel re-implementation of the
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
| Production reordered element | 7.556 | 5.473 | 1.937 | 0.333 | 1.653 |

So the old HE JAX+PETSc solve gap is largely gone in the production element
path. The main remaining difference is end-to-end setup/JIT cost rather than
the linear solve itself.

The current production element path also uses a whole-local overlap gradient
rather than per-element gradient scatter, which gives a small additional win on
top of the reordered ownership change.

Detailed notes:
- [investigation_jaxpetsc_performance_gap.md](archive/investigation_jaxpetsc_performance_gap.md)
- [HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md](archive/HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md)

#### How to run

```bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

# Element Hessian assembly (recommended — 2–3× faster than SFD)
mpirun -n 16 python3 src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py \
    --level 3 --steps 24 --total_steps 24 \
    --profile performance --assembly_mode element \
    --element_reorder_mode block_xyz --quiet

# SFD assembly (baseline)
mpirun -n 16 python3 src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py \
    --level 3 --steps 24 --total_steps 24 \
    --profile performance --assembly_mode sfd --quiet

# With detailed linear timing breakdown
mpirun -n 32 python3 src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py \
    --level 4 --steps 1 --total_steps 96 \
    --profile performance --assembly_mode element \
    --element_reorder_mode block_xyz \
    --save_linear_timing --quiet
```

Benchmark results and investigation: [results_HyperElasticity3D.md](archive/results_HyperElasticity3D.md),
[investigation_jaxpetsc_performance_gap.md](archive/investigation_jaxpetsc_performance_gap.md),
[HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md](archive/HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md)

## Prerequisites

FEniCS solvers require **DOLFINx >= 0.10** with PETSc. JAX solvers require **JAX**, **h5py**, **PyAMG**. The included devcontainer provides everything — see [docs/setup/quickstart.md](docs/setup/quickstart.md).

## Repository Structure

```
.
├── src/
│   ├── core/                          # Shared serial, PETSc, CLI, and benchmark helpers
│   └── problems/                      # Canonical problem implementations
│       ├── plaplace/
│       ├── ginzburg_landau/
│       ├── hyperelasticity/
│       └── topology/
├── experiments/
│   ├── runners/                       # Authoritative benchmark runners
│   ├── analysis/                      # Report generators and aggregation scripts
│   ├── diagnostics/                   # Focused investigation helpers
│   └── sweeps/                        # Parameter sweeps
├── notebooks/
│   ├── demos/
│   └── benchmarks/
├── docs/
│   ├── problems/                      # Canonical problem overviews
│   ├── results/                       # Current maintained results and scaling
│   ├── setup/                         # Environment and run instructions
│   ├── implementation/                # Current implementation notes
│   ├── reference/                     # Cross-problem formulation/reference notes
│   └── assets/                        # Curated current figures
├── data/
│   └── meshes/                        # Checked-in mesh/problem inputs
├── artifacts/
│   ├── raw_results/                   # Stored benchmark outputs
│   ├── reproduction/                  # Reproduction campaign runs
│   └── figures/                       # Generated figures before curation
├── archive/                           # Historical reports and retired material
└── scratch/                           # Local scratch space (gitignored)
```

## Mesh Levels

The meshes in `data/meshes/pLaplace/` (levels 1–9) are pre-generated. The paper table uses levels 4–8, which correspond to **mesh files 5–9**:

| Table Level |  Mesh File   | Total DOFs | Free DOFs |
| :---------: | :----------: | :--------: | :-------: |
|      4      | mesh_level_5 |    3201    |   2945    |
|      5      | mesh_level_6 |   12545    |   12033   |
|      6      | mesh_level_7 |   49665    |   48641   |
|      7      | mesh_level_8 |   197633   |  195585   |
|      8      | mesh_level_9 |   788481   |  784385   |

To regenerate FEniCS XDMF meshes from HDF5 source: `python3 src/problems/plaplace/fenics/export_pLaplace_meshes.py`
