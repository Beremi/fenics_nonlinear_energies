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
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py --level 1 --steps 24 --total_steps 24 --quiet --out overview/img/runs/hyperelasticity/showcase/jax_serial/output.json --state-out overview/img/runs/hyperelasticity/showcase/jax_serial/state.npz
