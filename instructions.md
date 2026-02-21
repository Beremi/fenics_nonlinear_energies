# Instructions

How to set up the environment, run solvers, store results, and generate tables.

## Environment Setup

All solvers require **DOLFINx >= 0.10** with PETSc support.

### Using Docker Directly

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

Open this folder in VS Code and use **"Reopen in Container"**. Then run directly in the terminal:

```bash
python3 solve_pLaplace_snes_newton.py
mpirun -n 4 python3 solve_pLaplace_snes_newton.py
```

## Running Solvers

### SNES Newton (recommended)

```bash
# Default (all mesh levels)
python3 solve_pLaplace_snes_newton.py

# Specific mesh levels
python3 solve_pLaplace_snes_newton.py --levels 5 6 7 8 9

# Save results to JSON
python3 solve_pLaplace_snes_newton.py --json results/my_run.json

# Parallel
mpirun -n 4 python3 solve_pLaplace_snes_newton.py --json results/parallel.json
```

### Custom Newton

```bash
# With quiet mode (suppress per-iteration output)
python3 solve_pLaplace_custom_newton.py --quiet --json results/custom.json

# Parallel
mpirun -n 8 python3 solve_pLaplace_custom_newton.py --quiet --json results/custom_8proc.json
```

### JAX Newton (no MPI)

Pure-JAX solver using auto-diff gradients, sparse finite-difference Hessian (graph coloring), and PyAMG CG. Runs on a single CPU — no MPI parallelism. Requires `jax`, `h5py`, and `pyamg`.

```bash
# Default (mesh levels 5–9)
python3 solve_pLaplace_jax_newton.py

# Specific levels
python3 solve_pLaplace_jax_newton.py --levels 5 6 7

# Save results to JSON
python3 solve_pLaplace_jax_newton.py --json results/jax.json
```

## Automated Experiment Runner

`run_experiments.py` runs a complete benchmark suite with multiple solvers, MPI configurations, and repetitions:

```bash
python3 run_experiments.py --nprocs 1 4 8 --levels 5 6 7 8 9 --repeat 3 --tag my_machine
```

This creates a timestamped directory under `results/` with:
- `metadata.json` — system info, git commit, DOLFINx version, CPU, etc.
- `<solver>_np<N>_run<R>.json` — individual run results

## Storing Results

### Directory Structure

```
results/
└── experiment_001/
    ├── metadata.json                    # System/environment metadata
    ├── snes_newton_np1_run1.json        # SNES Newton, serial, run 1
    ├── snes_newton_np1_run2.json
    ├── snes_newton_np1_run3.json
    ├── snes_newton_np4_run1.json        # SNES Newton, 4 proc, run 1
    ├── ...
    ├── custom_newton_np1_run1.json      # Custom Newton, serial, run 1
    ├── ...
    ├── custom_newton_np8_run3.json
    └── tables.tex                       # Generated LaTeX tables
```

### File Naming Convention

```
results/<experiment_id>/<solver>_np<nprocs>_run<repetition>.json
```

| Field           | Values                                                                 |
| --------------- | ---------------------------------------------------------------------- |
| `experiment_id` | Sequential (`experiment_001`) or timestamped (`20260221_133000_mytag`) |
| `solver`        | `snes_newton`, `custom_newton`, or `jax_newton`                        |
| `nprocs`        | Number of MPI processes (1 = serial)                                   |
| `repetition`    | Run number (1, 2, 3…)                                                  |

### JSON Result Format

Each result file contains solver metadata and per-level results:

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
    }
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

## Reading Results Programmatically

```python
import json
from collections import defaultdict
import statistics

# Load a single run
with open("results/experiment_001/snes_newton_np1_run1.json") as f:
    data = json.load(f)
    for r in data["results"]:
        print(f"Level {r['mesh_level']}: {r['time']:.3f}s, {r['iters']} iters, J={r['energy']}")

# Load and aggregate all runs for a configuration (median time)
times = defaultdict(list)
for run in range(1, 4):
    with open(f"results/experiment_001/snes_newton_np1_run{run}.json") as f:
        for r in json.load(f)["results"]:
            times[r["mesh_level"]].append(r["time"])

for lvl in sorted(times):
    print(f"Level {lvl}: median={statistics.median(times[lvl]):.3f}s")
```

## Adding New Experiments

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

4. **Save metadata** — the experiment runner does this automatically, or create `metadata.json` manually.

5. **Generate tables**:
   ```bash
   python3 generate_latex_tables.py results/experiment_002/ --output results/experiment_002/tables.tex
   ```

6. **Commit results**:
   ```bash
   git add results/experiment_002/
   git commit -m "Add experiment_002 results on <machine>"
   ```

## Generating LaTeX Tables

`generate_latex_tables.py` reads all JSON files in an experiment directory, aggregates repeated runs (median time), and produces formatted tables.

```bash
# Print LaTeX to stdout
python3 generate_latex_tables.py results/experiment_001/

# Save LaTeX to file
python3 generate_latex_tables.py results/experiment_001/ --output results/experiment_001/tables.tex

# Print Markdown tables to stdout
python3 generate_latex_tables.py results/experiment_001/ --markdown
```

The script produces two tables:
1. **FEniCS SNES Newton comparison** — serial vs parallel timings (matches the paper format)
2. **All solver configurations** — both SNES and Custom Newton across all MPI counts

The generated `.tex` file can be `\input{}`-ed directly into a LaTeX document.

## Generating Scaling Plots

`generate_scaling_plot.py` reads SNES Newton JSON results across process counts and produces strong scaling and speedup plots.

```bash
# Default output: <exp_dir>/scaling.png
python3 generate_scaling_plot.py results/experiment_001/

# Custom output path
python3 generate_scaling_plot.py results/experiment_001/ --output my_scaling.png

# Plot a different solver
python3 generate_scaling_plot.py results/experiment_001/ --solver custom_newton
```

**Note**: Requires `matplotlib`. If running outside the devcontainer, use Docker:
```bash
docker run --rm --entrypoint python3 -v "$PWD":/work -w /work fenics_test \
  /work/generate_scaling_plot.py results/experiment_001/
```
