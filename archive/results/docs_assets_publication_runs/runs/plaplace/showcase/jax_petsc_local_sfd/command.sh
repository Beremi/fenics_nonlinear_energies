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
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json overview/img/runs/plaplace/showcase/jax_petsc_local_sfd/output.json
