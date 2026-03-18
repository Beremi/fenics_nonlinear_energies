#!/bin/bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

echo "Running JAX Element GAMG (24 steps) - lvl 4, n=32, 1e-3, with retry on fail"
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 4 --steps 24 --total_steps 24 --reorder --profile performance --save_linear_timing --ksp_type cg --pc_type gamg --ksp_rtol 1e-3 --ksp_max_it 10000 --assembly_mode element --retry_on_failure --out l4_jax_element_gamg_1e-3_24steps_total24.json > l4_jax_element_24steps_total24.log 2>&1

echo "Done!"
