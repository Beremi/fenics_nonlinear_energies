# FEniCS Nonlinear Energies — p-Laplace 2D Benchmark

This repository contains FEniCSx (DOLFINx) solvers for the 2D p-Laplacian problem and infrastructure for reproducible benchmarking.

## Problem Description

We solve the p-Laplacian problem (p=3) on the unit square with homogeneous Dirichlet boundary conditions:

$$\min_u J(u) = \int_\Omega \frac{1}{p} |\nabla u|^p \, dx - \int_\Omega f \cdot u \, dx, \quad u|_{\partial\Omega} = 0$$

with $p = 3$ and $f = -10$.

## Solver Scripts

### `solve_pLaplace_snes_newton.py` — Built-in SNES Newton

Uses PETSc's SNES (Scalable Nonlinear Equations Solvers) with:
- Newton line search (`newtonls`) with full steps (`basic`)
- CG linear solver with HYPRE AMG preconditioner
- `rtol = 1e-1` for the linear solver (inexact Newton)
- Convergence: `atol = 1e-6`, `rtol = 1e-8`

This is the **recommended solver** — matches the "FEniCS serial/parallel" columns in the paper.

### `solve_pLaplace_custom_newton.py` — Custom Newton with Line Search

Manually implements the Newton method with:
- Explicit Hessian and gradient assembly
- Golden-section line search on the energy functional
- Same CG + HYPRE AMG linear solver
- Convergence: gradient infinity norm < 1e-6

This solver is slower due to the line search overhead (multiple energy evaluations per iteration) but provides more control and debugging output.

## Prerequisites

All solvers require **DOLFINx >= 0.10** with PETSc support. The easiest way to run them is via the included devcontainer.

### Using the Devcontainer (Docker)

1. Build the Docker image:
   ```bash
   docker build -t fenics_test -f .devcontainer/Dockerfile .devcontainer/
   ```

2. Run a solver (serial):
   ```bash
   docker run --rm --entrypoint python3 -e PYTHONUNBUFFERED=1 \
     -v "$PWD":/work -w /work fenics_test \
     /work/solve_pLaplace_snes_newton.py
   ```

3. Run a solver (parallel with N processes):
   ```bash
   docker run --rm --entrypoint mpirun -e PYTHONUNBUFFERED=1 \
     -v "$PWD":/work -w /work fenics_test \
     -n 4 python3 /work/solve_pLaplace_snes_newton.py
   ```

### VS Code Devcontainer

Open this folder in VS Code and use "Reopen in Container" to get a fully configured environment. Then you can run directly:
```bash
python3 solve_pLaplace_snes_newton.py
mpirun -n 4 python3 solve_pLaplace_snes_newton.py
```

## Running Benchmarks

### Quick Run

Both solver scripts accept `--levels` and `--json` arguments:

```bash
# Run specific mesh levels
python3 solve_pLaplace_snes_newton.py --levels 5 6 7 8 9

# Save results to JSON
python3 solve_pLaplace_snes_newton.py --json results/my_run.json

# Custom Newton with quiet mode
python3 solve_pLaplace_custom_newton.py --quiet --json results/custom.json
```

### Automated Experiment Runner

Use `run_experiments.py` inside the devcontainer to run a complete benchmark suite:

```bash
python3 run_experiments.py --nprocs 1 4 8 --levels 5 6 7 8 9 --repeat 3 --tag my_machine
```

This creates a timestamped directory under `results/` with:
- `metadata.json` — system info, git commit, DOLFINx version, etc.
- `<solver>_np<N>_run<R>.json` — individual run results

### Generating LaTeX Tables

After running experiments:

```bash
python3 generate_latex_tables.py results/experiment_001/
python3 generate_latex_tables.py results/experiment_001/ --output results/experiment_001/tables.tex
python3 generate_latex_tables.py results/experiment_001/ --markdown
```

## Mesh Levels

The meshes are pre-generated in `pLaplace_fenics_mesh/` (mesh levels 1–9). The table in the paper uses levels 4–8, which correspond to **mesh files 5–9**:

| Table Level | Mesh File | Total DOFs | Free DOFs |
|:-----------:|:---------:|:----------:|:---------:|
| 4 | mesh_level_5 | 3201 | 2945 |
| 5 | mesh_level_6 | 12545 | 12033 |
| 6 | mesh_level_7 | 49665 | 48641 |
| 7 | mesh_level_8 | 197633 | 195585 |
| 8 | mesh_level_9 | 788481 | 784385 |

To regenerate meshes (requires pLaplace2D mesh data): `python3 export_pLaplace_meshes.py`

---

## Results

### Experiment `experiment_001`

- **Date**: 2026-02-21
- **CPU**: AMD Ryzen 9 9950X3D 16-Core Processor (32 threads)
- **DOLFINx**: 0.10.0.post2
- **Git commit**: `7dce8760`
- **Repetitions**: 3 (median time reported)

#### FEniCS SNES Newton (serial vs parallel)

| lvl | dofs | time (serial) | iters | J(u) | time (4 proc) | iters | J(u) | time (8 proc) | iters | J(u) |
|-----|------|------------|-------|------|------------|-------|------|------------|-------|------|
| 4 | 2945 | 0.043 | 10 | -7.9430 | 0.029 | 11 | -7.9430 | 0.026 | 10 | -7.9430 |
| 5 | 12033 | 0.167 | 10 | -7.9546 | 0.071 | 10 | -7.9546 | 0.050 | 10 | -7.9546 |
| 6 | 48641 | 0.478 | 7 | -7.9583 | 0.169 | 7 | -7.9583 | 0.110 | 7 | -7.9583 |
| 7 | 195585 | 2.152 | 8 | -7.9596 | 0.768 | 8 | -7.9596 | 0.463 | 8 | -7.9596 |
| 8 | 784385 | 10.026 | 9 | -7.9600 | 3.772 | 9 | -7.9600 | 2.430 | 9 | -7.9600 |

#### All Solver Configurations

| lvl | dofs | SNES serial | iters | Custom serial | iters | SNES 4-proc | iters | Custom 4-proc | iters | SNES 8-proc | iters | Custom 8-proc | iters | J(u) |
|-----|------|------------|-------|------------|-------|------------|-------|------------|-------|------------|-------|------------|-------|------|
| 4 | 2945 | 0.043 | 10 | 0.043 | 8 | 0.029 | 11 | 0.025 | 8 | 0.026 | 10 | 0.026 | 9 | -7.9430 |
| 5 | 12033 | 0.167 | 10 | 0.193 | 9 | 0.071 | 10 | 0.075 | 9 | 0.050 | 10 | 0.055 | 9 | -7.9546 |
| 6 | 48641 | 0.478 | 7 | 0.754 | 9 | 0.169 | 7 | 0.259 | 9 | 0.110 | 7 | 0.164 | 9 | -7.9583 |
| 7 | 195585 | 2.152 | 8 | 3.373 | 10 | 0.768 | 8 | 1.014 | 9 | 0.463 | 8 | 0.681 | 10 | -7.9596 |
| 8 | 784385 | 10.026 | 9 | 15.162 | 11 | 3.772 | 9 | 5.337 | 11 | 2.430 | 9 | 3.508 | 11 | -7.9600 |

**Note**: The Custom Newton solver is slower because each iteration includes a golden-section line search (multiple energy evaluations). The SNES Newton with full steps is more efficient for this problem.

---

## Results Directory Structure

```
results/
├── .gitkeep
└── experiment_001/
    ├── metadata.json                    # System/environment metadata
    ├── snes_newton_np1_run1.json        # SNES Newton, serial, run 1
    ├── snes_newton_np1_run2.json        # SNES Newton, serial, run 2
    ├── snes_newton_np1_run3.json        # SNES Newton, serial, run 3
    ├── snes_newton_np4_run1.json        # SNES Newton, 4 proc, run 1
    ├── ...
    ├── custom_newton_np1_run1.json      # Custom Newton, serial, run 1
    ├── ...
    ├── custom_newton_np8_run3.json      # Custom Newton, 8 proc, run 3
    └── tables.tex                       # Generated LaTeX tables
```

### JSON Result Format

Each result file (`<solver>_np<N>_run<R>.json`) contains:

```json
{
  "metadata": {
    "solver": "snes_newton",
    "description": "Built-in PETSc SNES Newton solver with CG + HYPRE AMG",
    "dolfinx_version": "0.10.0.post2",
    "nprocs": 1,
    "petsc_options": {
      "snes_type": "newtonls",
      "snes_linesearch_type": "basic",
      "ksp_type": "cg",
      "pc_type": "hypre",
      "ksp_rtol": 0.1
    },
    "p": 3,
    "rhs_f": -10.0
  },
  "results": [
    {
      "mesh_level": 5,
      "total_dofs": 3201,
      "time": 0.043,
      "iters": 10,
      "energy": -7.943,
      "converged_reason": 2
    },
    ...
  ]
}
```

### `metadata.json` Format

```json
{
  "experiment_id": "experiment_001",
  "timestamp": "2026-02-21T13:36:59.405339",
  "git_commit": "7dce8760105ef28d9c2213374c23247ce3a33098",
  "hostname": "beremifeipc",
  "cpu": "AMD Ryzen 9 9950X3D 16-Core Processor",
  "cpu_count": 32,
  "dolfinx_version": "0.10.0.post2",
  "mesh_levels": [5, 6, 7, 8, 9],
  "nprocs_tested": [1, 4, 8],
  "repetitions": 3
}
```

### How to Read Results Programmatically

```python
import json
from collections import defaultdict

# Load a single run
with open("results/experiment_001/snes_newton_np1_run1.json") as f:
    data = json.load(f)
    for r in data["results"]:
        print(f"Level {r['mesh_level']}: {r['time']:.3f}s, {r['iters']} iters, J={r['energy']}")

# Load and aggregate all runs for a configuration
import statistics
times = defaultdict(list)
for run in range(1, 4):
    with open(f"results/experiment_001/snes_newton_np1_run{run}.json") as f:
        for r in json.load(f)["results"]:
            times[r["mesh_level"]].append(r["time"])

for lvl in sorted(times):
    print(f"Level {lvl}: median={statistics.median(times[lvl]):.3f}s")
```

### How to Add New Experiments

1. **Commit your code changes first** — so the git commit hash in metadata is meaningful.

2. **Create a new experiment directory**:
   ```bash
   mkdir results/experiment_002
   ```

3. **Run experiments** (inside devcontainer or Docker):
   ```bash
   # Option A: Use the experiment runner
   python3 run_experiments.py --nprocs 1 4 8 --tag experiment_002

   # Option B: Run manually
   python3 solve_pLaplace_snes_newton.py --json results/experiment_002/snes_newton_np1_run1.json
   mpirun -n 4 python3 solve_pLaplace_snes_newton.py --json results/experiment_002/snes_newton_np4_run1.json
   ```

4. **Save metadata** (the experiment runner does this automatically, or create manually).

5. **Generate tables**:
   ```bash
   python3 generate_latex_tables.py results/experiment_002/ --output results/experiment_002/tables.tex
   ```

6. **Commit results**:
   ```bash
   git add results/experiment_002/
   git commit -m "Add experiment_002 results on <machine>"
   ```

### Naming Convention

```
results/<experiment_id>/<solver>_np<nprocs>_run<repetition>.json
```

- `experiment_id`: Sequential (`experiment_001`) or timestamped (`20260221_133000_mytag`)
- `solver`: `snes_newton` or `custom_newton`
- `nprocs`: Number of MPI processes (1 = serial)
- `repetition`: Run number (1, 2, 3...)

## Legacy Scripts

The original scripts (`test_fenics_pLaplace.py`, `test_fenics_pLaplace2.py`) are kept for reference but use the **old DOLFINx API** (`u.vector` instead of `u.x.petsc_vec`, old `NonlinearProblem` / `NewtonSolver` constructors). They will **not run** with DOLFINx >= 0.10.

## File Overview

| File | Description |
|------|-------------|
| `solve_pLaplace_snes_newton.py` | **Main solver**: SNES Newton (DOLFINx 0.10+) |
| `solve_pLaplace_custom_newton.py` | Custom Newton with golden-section line search (DOLFINx 0.10+) |
| `run_experiments.py` | Automated experiment runner (collects metadata, runs multiple configs) |
| `generate_latex_tables.py` | Generate LaTeX/Markdown tables from result JSON files |
| `export_pLaplace_meshes.py` | Generate FEniCS mesh files from pLaplace2D mesh data |
| `test_fenics_pLaplace.py` | Legacy: old DOLFINx API SNES Newton (for reference) |
| `test_fenics_pLaplace2.py` | Legacy: old DOLFINx API custom Newton (for reference) |
| `pLaplace_fenics_mesh/` | Pre-generated mesh files (XDMF + HDF5) for levels 1–9 |
| `pLaplace2D/` | Mesh generation module (JAX-based mesh data) |
| `results/` | Experiment results (JSON + LaTeX tables) |
