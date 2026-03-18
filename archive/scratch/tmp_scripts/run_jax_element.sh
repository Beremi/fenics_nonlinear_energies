#!/bin/bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

echo "Running JAX GAMG (Element Assembly) - lvl 4, n=32, 1e-3"
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 4 --steps 1 --total_steps 96 --reorder --profile performance --save_linear_timing --ksp_type cg --pc_type gamg --ksp_rtol 1e-3 --ksp_max_it 10000 --assembly_mode element --out l4_jax_element_gamg_1e-3.json > l4_jax_element_1e-3.log 2>&1

echo "Done!"
