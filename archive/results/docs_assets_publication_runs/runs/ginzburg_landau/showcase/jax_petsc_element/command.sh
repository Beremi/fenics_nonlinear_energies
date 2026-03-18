#!/usr/bin/env bash
set -euo pipefail
cd /home/michal/repos/fenics_nonlinear_energies
export BLIS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out overview/img/runs/ginzburg_landau/showcase/jax_petsc_element/output.json
