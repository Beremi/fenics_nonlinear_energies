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
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --trust_max_reject 6 --nproc 1 --quiet --out overview/img/runs/hyperelasticity/showcase/jax_petsc_element/output.json
