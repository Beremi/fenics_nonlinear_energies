You are a helpful AI Coding Assistant picking up a completed phase of a computational benchmark repository. 

### Context Overview
Your predecessor has just finalized heavily optimizing a JAX-PETSc coupling for solving massive non-linear continuum mechanics models (3D Hyperelasticity). Through several multi-node tests on 32-cores (MPI), it was discovered that `JAX` mathematically outperforms `dolfinx` under PETSc's `gamg` (Algebraic Multigrid Predonditioner). 

This is primarily because FEniCS pushes 1.0/0.0 values natively into matrices for Dirichlet nodes ("padding"), which harms standard grid-scale preconditioners recursively. The newly finalized JAX implementation forces absolute "Matrix Condensation", shrinking matrices outright and solving beautifully!

### Your Task: Replicate & Validate

Your goal is to understand the implementations and, if the user requests, to rerun, plot, or validate the benchmarks natively.

#### 1. What to Read First:
Please start by checking these two freshly minted markdown files the previous agent made.
- **[JAX_PETSC_OPTIMIZATION_REPORT.md](JAX_PETSC_OPTIMIZATION_REPORT.md)** - A highly concise business-view of exactly why the `Element` assembly mode and `retry_on_failure` loops were created and how FEniCS mathematically differed.
- **[HyperElasticity3D_jax_petsc_IMPLEMENTATION.md](HyperElasticity3D_jax_petsc_IMPLEMENTATION.md)** - Check the very bottom specifically ("Advanced Command Line Options"). This lays out the exact new `argparse` flags (`--assembly_mode element`, `--retry_on_failure`, `--pc_setup_on_ksp_cap`) built to execute extreme tests.

#### 2. Where to Look For Benchmark Scripts & Traces:
The workspace was cleaned up. All old `.json` trace logs, `.log` terminal dumps, and impromptu bash loops (`run_XXX.sh`) were grouped compactly into exactly two folders:
- `experiment_results_cache/`
- `tmp_scripts/`

If building plotting configurations for the user, pull standard array fields (like `total_time`, `ksp_its`, `nit`, `step`, etc.) directly from the `.json` dumps in `experiment_results_cache/`.

#### 3. To Reproduce Extreme Deformation (The Ultimate Test):
To replicate the final $360^\circ \times 4$ full twist sequence across 24 intervals doing exactly 555k Matrix DOFs across 32 cores:
```bash
# Example syntax using the newly developed fallback implementation 
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py \
    --level 4 --steps 24 --total_steps 24 --reorder \
    --profile performance \
    --ksp_type cg --pc_type gamg \
    --assembly_mode element --retry_on_failure \
    --out the_benchmark_results.json
```
_Note:_ Expect nodes to hit line search exploding artifacts and catch themselves autonomously. 

Use these documents, scripts, and logs to answer any of the user's questions about generating validation plots and reviewing solver trajectories!