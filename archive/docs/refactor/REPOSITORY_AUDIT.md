# Repository Audit

Date: 2026-03-15

## What This Repository Is

This is a research and benchmarking repository for nonlinear finite-element
energy minimisation problems.

The main implementation styles are:

- FEniCSx + PETSc solvers
- pure JAX serial solvers
- JAX + PETSc distributed solvers

It is not organised like a small reusable Python package. It behaves more like
an active lab notebook plus solver codebase:

- solver implementations live beside benchmark runners,
- current docs live beside archived investigations,
- large raw result caches are committed,
- and temporary analysis folders are still present.

## The Main Problem Families

### 1. `pLaplace`

Purpose:
- scalar 2D p-Laplacian benchmark

Main directories:
- `pLaplace2D_fenics/`
  - FEniCSx versions
  - includes SNES Newton and custom PETSc Newton
- `pLaplace2D_jax/`
  - pure JAX serial reference path
- `pLaplace2D_jax_petsc/`
  - distributed JAX + PETSc path
  - current main CLI is `solve_pLaplace_dof.py`
- `pLaplace2D_petsc_support/`
  - mesh loading helpers for PETSc/JAX+PETSc paths

What it should do:
- provide like-for-like solver variants for the same scalar benchmark
- support benchmarking across mesh levels and MPI counts
- serve as the cleanest scalar test case for the JAX+PETSc infrastructure

### 2. `GinzburgLandau`

Purpose:
- scalar 2D non-convex benchmark with indefinite Hessian

Main directories:
- `GinzburgLandau2D_fenics/`
  - FEniCSx custom Newton and SNES paths
- `GinzburgLandau2D_jax/`
  - pure JAX energy and mesh support used by the notebook/reference path
- `GinzburgLandau2D_jax_petsc/`
  - distributed JAX + PETSc path

What it should do:
- stress-test the nonlinear solver stack on a harder non-convex scalar energy
- compare line-search and trust-region behaviour
- validate the reordered overlap JAX+PETSc infrastructure on a scalar problem

### 3. `HyperElasticity`

Purpose:
- 3D neo-Hookean beam under incremental rotating boundary load

Main directories:
- `HyperElasticity3D_fenics/`
  - FEniCSx custom Newton and SNES paths
- `HyperElasticity3D_jax/`
  - pure JAX serial reference path
- `HyperElasticity3D_jax_petsc/`
  - distributed JAX + PETSc path
- `HyperElasticity3D_petsc_support/`
  - mesh loading and boundary rotation helpers

What it should do:
- act as the flagship hard benchmark in the repo
- compare FEniCS custom, JAX+PETSc, and pure-JAX trust-region workflows
- exercise vector-valued assembly, near-nullspace handling, and PETSc AMG

### 4. `topological_optimisation_jax`

Purpose:
- staggered SIMP + phase-field style topology optimisation benchmark

Current status:
- the maintained path is the MPI-parallel JAX+PETSc solver
- the pure-JAX solver is retained as a historical/reference implementation

Main files:
- `topological_optimisation_jax/solve_topopt_parallel.py`
  - current active path
- `topological_optimisation_jax/solve_topopt_jax.py`
  - serial reference path
- `topological_optimisation_jax/parallel_support.py`
  - distributed mechanics/design support
- report/test helpers in the same folder

What it should do:
- be the current topology benchmark path
- produce final benchmark figures and scaling reports
- remain separate from the older exploratory topology markdown clutter

## Shared Infrastructure

### `tools/`

Purpose:
- serial/shared numerical infrastructure

Contains:
- `minimizers.py`
  - serial Newton and gradient-descent style outer solvers
- `jax_diff.py`
  - JAX energy/gradient/Hessian derivation on fixed sparsity graphs
- `sparse_solvers.py`
  - PyAMG / sparse direct / trust-subproblem helpers
- `graph_sfd.py`
  - serial graph-coloring-based sparse finite difference support

What it should do:
- hold backend-agnostic serial solver infrastructure

### `tools_petsc4py/`

Purpose:
- PETSc-side shared infrastructure

Contains:
- `minimizers.py`
  - PETSc-vector Newton and gradient-descent methods
- `dof_partition.py`
  - distributed ownership/ghost/overlap partitioning
- `trust_ksp.py`
  - PETSc trust-region KSP compatibility helpers
- `jax_tools/parallel_assembler.py`
  - generic base classes for JAX+PETSc distributed assembly

What it should do:
- be the common PETSc backend for the distributed solver families

### `graph_coloring/`

Purpose:
- distance-2 coloring for sparse finite-difference Hessian recovery

Contains:
- Python wrappers
- multiple coloring backends
- `custom_coloring.c`
- prebuilt `custom_coloring.so`

What it should do:
- provide fast coloring for the serial and distributed SFD Hessian paths

## Data, Docs, and Entry Points

### `mesh_data/`

Purpose:
- pre-generated HDF5 problem data and adjacency structures

What it should do:
- be the canonical mesh/problem-data source for all solver families

Notes:
- this is real project data, not clutter
- size is significant but justified

### Root notebooks

Files:
- `example_pLaplace2D_jax.ipynb`
- `example_GinzburgLandau2D_jax.ipynb`
- `example_HyperElasticity3D_jax.ipynb`
- `example_pLaplace2D_jax_petsc.ipynb`
- `benchmark_*_fenics.ipynb`

What they should do:
- be the easiest interactive entry points for exploring the repo

### `docs/`

Purpose:
- current documentation and final reports

What looks authoritative now:
- `docs/README.md`
- `docs/final_pLaplace_results.md`
- `docs/final_GL_results.md`
- `docs/final_HE_results.md`
- `docs/JAX_TOPOLOGY_CURRENT_STATE.md`
- `docs/final_JAX_TOPOLOGY_parallel_results.md`
- implementation notes and local build docs

What it should do:
- be the first place to look for current state
- absorb the important information that is currently scattered elsewhere

### `archive/`

Purpose:
- historical benchmark reports and investigations

What it should do:
- hold superseded reports and handoff notes
- stay clearly non-authoritative compared with `docs/`

This folder is doing the right kind of job already.

## Benchmark Automation and Result Storage

### `experiment_scripts/`

Purpose:
- campaign runners, sweeps, diagnostics, parsers, and one-off analysis scripts

What is actually in here:
- benchmark runners (`run_*`)
- sweeps (`sweep_*`)
- diagnostics/debug scripts (`diag_*`, `debug_*`, `check_*`)
- many benchmark/investigation scripts (`bench_*`)
- many scripts named `test_*` that are really experiment scripts, not a clean test suite
- some generated summaries and ad hoc data files

What it should do:
- contain reproducible benchmark automation

What it currently also does:
- act as a dumping ground for investigations

### `experiment_results_cache/`

Purpose:
- raw benchmark outputs and final-suite caches

Notable subtrees:
- `gl_final_suite/`
- `plaplace_final_suite/`
- `he_final_suite_*`
- minimizer/trust-region sweep caches

What it should do:
- preserve raw campaign outputs used by final reports

What makes it messy:
- it is very large
- it mixes canonical final data with exploratory reruns and loose logs
- it dominates the repo size

### `results/` and `results_GL/`

Purpose:
- older benchmark output folders plus helper scripts

What they appear to be now:
- older experiment-layout directories from earlier benchmarking workflows
- no longer the main canonical location for the latest final reports

What they should do:
- either become clearly “legacy benchmark outputs”
- or be merged into one consistent experiment-output story later

## Likely Scratch or Local-Only Areas

### `tmp_scripts/`

Purpose:
- temporary shell/python helpers for one-off reruns and parsing

What it should do:
- probably not live as a first-class committed project area

### `tmp_work/`

Purpose:
- scratch logs, quick parsers, temporary analyses, ad hoc tests

What it should do:
- probably be treated as disposable workspace, not repository structure

### `.venv/` and `local_env/`

Purpose:
- local Python / local full FEniCS build environments

What they should do:
- stay local-only

Current state:
- `.gitignore` excludes them
- they are still physically present in the working tree
- they are among the largest things in the repo, but they are environment clutter, not source

### `.devcontainer/`, `.vscode/`, `.claude/`

Purpose:
- local/editor/container setup

What they should do:
- support development only

These are normal support folders and not a major source of conceptual mess.

## Current Authoritative Reading Order

If I had to understand the repository quickly, I would read in this order:

1. `README.md`
2. `docs/README.md`
3. the relevant final report in `docs/`
4. the matching solver directories for that problem family
5. `tools/` and `tools_petsc4py/`

For topology specifically:

1. `docs/JAX_TOPOLOGY_CURRENT_STATE.md`
2. `docs/JAX_TOPOLOGY_jax_petsc_IMPLEMENTATION.md`
3. `docs/final_JAX_TOPOLOGY_parallel_results.md`
4. `topological_optimisation_jax/solve_topopt_parallel.py`

## Places Where The Repo Is Currently Inconsistent

These are not reorganisation proposals yet, just observed mismatches.

### 1. Some docs/scripts still point to stale pLaplace JAX+PETSc entry points

Observed:
- `docs/instructions.md` refers to `pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py`
- `results/run_experiments.py` also refers to that path

Current actual CLI:
- `pLaplace2D_jax_petsc/solve_pLaplace_dof.py`

### 2. Topology has both a clean current doc trail and leftover historical material

Observed:
- `docs/` clearly treats the parallel JAX+PETSc topology path as current
- `jax_topology_optimisation_benchmark.md` is explicitly historical
- `topological_optimisation_jax/description.md` reads like an old planning/prompt artifact

### 3. “Tests” are split between real tests and experiment scripts

Observed:
- the cleanest actual tests are in `topological_optimisation_jax/`
- many `experiment_scripts/test_*.py` files are experiment drivers, not a maintained test suite
- `tmp_*` folders also contain ad hoc test files

### 4. Result storage is fragmented

Observed:
- current final reports point heavily at `experiment_results_cache/`
- older benchmark runners write to `results/` and `results_GL/`
- images live in `img/`
- topology final assets live under `docs/assets/`

## My High-Level Read On The Mess

The repository is not random. It has a real structure, but it is layered by
project age:

- clean current solver code
- clean-ish current docs
- large raw benchmark history
- many one-off investigation scripts
- some leftover temporary workspace

So the main issue is not “nobody knows what anything is”.
The main issue is “active, historical, generated, and scratch material still
live too close together”.

That means a cleanup/reorganisation pass should probably focus on:

- making the active paths unmistakable,
- moving historical material behind clearer boundaries,
- and separating reproducible benchmark assets from temporary analysis.

## Proposed Clean Organisation

The main goal should be that the tree explains itself.
Someone should be able to open the repository and infer, from directory names
alone:

- where the reusable solver code lives,
- where to run experiments from,
- where to find human-facing docs and demos,
- where static input data lives,
- where generated outputs go,
- and what is historical or scratch.

### Directory Contracts

These are the rules I would use.

#### `src/`

Contains:
- importable project code only

Must not contain:
- benchmark outputs
- notebooks
- final report markdown
- scratch analysis

#### `experiments/`

Contains:
- runnable benchmark/sweep/diagnostic scripts only

Must not contain:
- core solver implementation
- final documentation
- cached raw outputs mixed beside scripts

#### `docs/`

Contains:
- current human-facing markdown only

Should be split by reader need:
- setup
- demos
- problem setup
- implementation notes
- benchmark reports

Must not contain:
- archived superseded reports mixed beside current ones

#### `data/`

Contains:
- checked-in static inputs
- small curated outputs that are intentionally versioned

Good fits:
- meshes
- small benchmark summary tables
- curated CSV/JSON used in final reports

#### `artifacts/`

Contains:
- generated outputs
- large raw result caches
- figures produced by scripts

This is where bulky benchmark runs should live.

#### `archive/`

Contains:
- superseded docs
- old investigations
- retired benchmark snapshots
- one-off planning notes worth keeping

#### `scratch/`

Contains:
- throwaway local work

Should be gitignored.

### Proposed Top-Level Tree

```text
fenics_nonlinear_energies/
├── src/
│   ├── core/
│   │   ├── serial/
│   │   ├── petsc/
│   │   └── coloring/
│   └── problems/
│       ├── plaplace/
│       │   ├── fenics/
│       │   ├── jax/
│       │   ├── jax_petsc/
│       │   └── support/
│       ├── ginzburg_landau/
│       │   ├── fenics/
│       │   ├── jax/
│       │   ├── jax_petsc/
│       │   └── support/
│       ├── hyperelasticity/
│       │   ├── fenics/
│       │   ├── jax/
│       │   ├── jax_petsc/
│       │   └── support/
│       └── topology/
│           ├── jax/
│           ├── jax_petsc/
│           └── support/
├── experiments/
│   ├── runners/
│   ├── sweeps/
│   ├── diagnostics/
│   ├── analysis/
│   └── legacy/
├── notebooks/
│   ├── demos/
│   └── benchmarks/
├── docs/
│   ├── overview/
│   ├── setup/
│   ├── demos/
│   ├── problem_setup/
│   ├── implementation/
│   ├── benchmarks/
│   └── assets/
├── data/
│   ├── meshes/
│   └── curated_results/
├── artifacts/
│   ├── raw_results/
│   ├── figures/
│   └── reports/
├── archive/
│   ├── docs/
│   ├── scripts/
│   └── results/
├── scratch/
├── .devcontainer/
├── README.md
└── Makefile
```

## Why This Structure Fits This Repo

### 1. It separates code from campaigns

Right now the core solver code is mixed at the root with experiment-heavy
folders. Moving all reusable code under `src/` makes the architecture obvious:

- `src/core/` is backend infrastructure
- `src/problems/` is problem-specific implementation

### 2. It separates “how to run” from “what the code is”

`experiments/` becomes the place for:

- reproducing papers/reports
- sweeps
- benchmark automation
- diagnosis scripts

That keeps run machinery away from the importable solver implementation.

### 3. It separates human docs by purpose

This repo has several kinds of markdown, and they should not sit flat in one
folder.

Suggested `docs/` split:

- `docs/overview/`
  - repo map, current state, index
- `docs/setup/`
  - environment setup, local build, how to run
- `docs/demos/`
  - notebook-oriented walkthroughs and simple usage guides
- `docs/problem_setup/`
  - mathematical problem definitions and formulation notes
- `docs/implementation/`
  - internal design notes, partitioning, trust region, coloring, PETSc setup
- `docs/benchmarks/`
  - final benchmark reports and curated result summaries

This directly matches your goal of keeping:

- demonstration material separate,
- problem/setup material separate,
- and result-generation scripts separate.

### 4. It makes generated material legible

Today `img/`, `docs/assets/`, `results/`, `results_GL/`, and
`experiment_results_cache/` all play overlapping output roles.

The cleaned model should be:

- small curated inputs/results in `data/`
- bulky generated outputs in `artifacts/`
- current human-readable reports in `docs/benchmarks/`

### 5. It gives scratch work an explicit home

`tmp_scripts/` and `tmp_work/` are signs that the repo needs a sanctioned
scratch area.

If `scratch/` exists and is gitignored, ad hoc work no longer has to become
part of the visible permanent structure.

## Proposed Code Layout Inside `src/`

The current problem-family split is already good.
What is missing is a shared parent layout and more consistent naming.

### `src/core/`

Suggested mapping:

- `tools/` -> `src/core/serial/`
- `tools_petsc4py/` -> `src/core/petsc/`
- `graph_coloring/` -> `src/core/coloring/`

This creates one obvious place for:

- serial minimizers and JAX derivative helpers
- PETSc minimizers and partitioning
- graph coloring / SFD infrastructure

### `src/problems/`

Suggested mapping:

- `pLaplace2D_*` -> `src/problems/plaplace/...`
- `GinzburgLandau2D_*` -> `src/problems/ginzburg_landau/...`
- `HyperElasticity3D_*` -> `src/problems/hyperelasticity/...`
- `topological_optimisation_jax/` -> `src/problems/topology/...`

Inside each problem:

- `fenics/`
  - FEniCS solver implementations
- `jax/`
  - pure JAX reference path
- `jax_petsc/`
  - distributed JAX+PETSc path
- `support/`
  - mesh loaders, BC helpers, shared problem utilities

This keeps the good part of the current structure while removing the current
root-level sprawl.

### CLI Convention

One thing that would help a lot is a consistent per-backend CLI naming scheme.

For example:

- `src/problems/plaplace/fenics/run_snes.py`
- `src/problems/plaplace/fenics/run_custom.py`
- `src/problems/plaplace/jax/run.py`
- `src/problems/plaplace/jax_petsc/run.py`

The exact names are flexible, but the pattern should be predictable across all
problem families.

## Proposed Documentation Layout

Here is the concrete split I would use.

### `docs/overview/`

Use for:
- repo map
- documentation index
- current-state summaries

Move here:
- `docs/README.md`
- `docs/REPOSITORY_AUDIT.md`
- `docs/JAX_TOPOLOGY_CURRENT_STATE.md`

### `docs/setup/`

Use for:
- environment setup
- local build
- run instructions

Move here:
- `docs/instructions.md`
- `docs/LOCAL_BUILD_GUIDE.md`

### `docs/problem_setup/`

Use for:
- mathematical formulations
- problem assumptions
- benchmark definitions

Move here:
- `docs/problem_formulation_brief.md`
- root historical benchmark-definition markdowns if still useful as references

### `docs/implementation/`

Use for:
- solver design notes
- algorithm notes
- partitioning and assembly internals

Move here:
- `docs/HyperElasticity3D_jax_petsc_IMPLEMENTATION.md`
- `docs/JAX_TOPOLOGY_jax_petsc_IMPLEMENTATION.md`
- `docs/TRUST_REGION_LINESEARCH_ALGORITHM.md`
- `docs/TRUST_REGION_LINESEARCH_TUNING.md`
- `docs/jax_parallel_partitioning.md`
- `docs/graph_coloring_implementation.md`
- `docs/HE_GAMG_ELASTICITY_SETUP.md`

### `docs/benchmarks/`

Use for:
- final benchmark reports only

Move here:
- `docs/final_pLaplace_results.md`
- `docs/final_GL_results.md`
- `docs/final_HE_results.md`
- `docs/final_JAX_TOPOLOGY_parallel_results.md`

And put benchmark assets under:
- `docs/assets/plaplace/`
- `docs/assets/ginzburg_landau/`
- `docs/assets/hyperelasticity/`
- `docs/assets/topology/`

That is clearer than the current split between `img/` and `docs/assets/`.

## Proposed Experiments Layout

The current `experiment_scripts/` folder really contains several different
classes of things.

I would split it into:

- `experiments/runners/`
  - stable scripts that launch standard suites
- `experiments/sweeps/`
  - parameter sweeps and campaign variants
- `experiments/diagnostics/`
  - focused debugging or verification scripts
- `experiments/analysis/`
  - parsing, summarising, plotting, table generation
- `experiments/legacy/`
  - one-off scripts kept only for provenance

This way a person can find the right class of script from the tree alone.

### Suggested classification of current script folders

- `results/run_experiments.py` and `results_GL/run_experiments.py`
  - move to `experiments/runners/legacy/` or replace with one unified runner
- `experiment_scripts/run_*`
  - mostly `experiments/runners/`
- `experiment_scripts/sweep_*`
  - `experiments/sweeps/`
- `experiment_scripts/bench_*`
  - split between `experiments/diagnostics/` and `experiments/legacy/`
- `experiment_scripts/parse_*`, `analyze_*`, `aggregate_*`, `summarize_*`
  - `experiments/analysis/`
- `experiment_scripts/test_*`
  - either move to real tests or reclassify as diagnostics

## Proposed Data and Output Layout

### `data/meshes/`

Suggested mapping:
- `mesh_data/` -> `data/meshes/`

This is checked-in, canonical, and part of the actual project inputs.

### `data/curated_results/`

Use for:
- small checked-in benchmark summaries
- CSV/JSON that final docs directly depend on

Only the minimal curated subset should live here.

### `artifacts/raw_results/`

Suggested mapping:
- `experiment_results_cache/` -> `artifacts/raw_results/`
- `results/` and `results_GL/` legacy outputs -> `artifacts/raw_results/legacy/`

This makes it explicit that these are generated outputs, not source code.

### `artifacts/figures/`

Suggested mapping:
- `img/` -> `artifacts/figures/`

If figures are embedded in docs, a curated published subset can still be copied
into `docs/assets/`.

## Proposed Treatment of Historical and Temporary Material

### Keep, but move behind a clear boundary

These should remain available, but not compete visually with active code:

- `archive/`
- old benchmark markdown in the repo root
- older result folders
- retired experiment scripts

### Remove from the visible permanent structure

These should move to a dedicated scratch area or be dropped:

- `tmp_scripts/`
- `tmp_work/`

## Current-To-Target Mapping Summary

Here is the practical directory mapping in one place.

- `tools/` -> `src/core/serial/`
- `tools_petsc4py/` -> `src/core/petsc/`
- `graph_coloring/` -> `src/core/coloring/`
- `pLaplace2D_*` -> `src/problems/plaplace/...`
- `GinzburgLandau2D_*` -> `src/problems/ginzburg_landau/...`
- `HyperElasticity3D_*` -> `src/problems/hyperelasticity/...`
- `topological_optimisation_jax/` -> `src/problems/topology/...`
- root notebooks -> `notebooks/demos/` and `notebooks/benchmarks/`
- `mesh_data/` -> `data/meshes/`
- `experiment_scripts/` -> `experiments/...`
- `results/`, `results_GL/`, `experiment_results_cache/` -> `artifacts/raw_results/...`
- `img/` -> `artifacts/figures/`
- current docs -> `docs/{overview,setup,problem_setup,implementation,benchmarks}/`
- `archive/` stays `archive/`
- `tmp_scripts/`, `tmp_work/` -> `scratch/`

## Minimal Reorganisation Principles

If the cleanup is done gradually, I would keep these rules:

1. No core solver code at repo root.
2. No generated results beside runnable scripts.
3. No current and archived markdown mixed in one folder.
4. No scratch folders pretending to be stable project structure.
5. One obvious place for code, one for docs, one for experiments, one for data, one for generated outputs.

That would make the repository understandable from the directory tree itself,
which is the main thing you asked for.
