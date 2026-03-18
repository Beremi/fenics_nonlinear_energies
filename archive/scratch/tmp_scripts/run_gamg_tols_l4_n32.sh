#!/bin/bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

echo "Level 4, n=32 Tests (Tol Variants)..."

echo "Running JAX GAMG (1e-1)"
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 4 --steps 1 --total_steps 96 --reorder --profile performance --save_linear_timing --ksp_type cg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 10000 --out l4_jax_gamg_1e-1.json > l4_jax_1e-1.log 2>&1

echo "Running FEniCS GAMG (1e-1)"
mpirun -n 32 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 4 --steps 1 --total_steps 96 --save_linear_timing --ksp_type cg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 10000 --maxit 50 --disable_repair --out l4_fenics_gamg_1e-1.json > l4_fenics_1e-1.log 2>&1

echo "Running FEniCS GAMG (1e-6)"
mpirun -n 32 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 4 --steps 1 --total_steps 96 --save_linear_timing --ksp_type cg --pc_type gamg --ksp_rtol 1e-6 --ksp_max_it 10000 --maxit 100 --out l4_fenics_gamg_1e-6.json > l4_fenics_1e-6.log 2>&1

echo "Running JAX GAMG (1e-6)"
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 4 --steps 1 --total_steps 96 --reorder --profile performance --save_linear_timing --ksp_type cg --pc_type gamg --ksp_rtol 1e-6 --ksp_max_it 10000 --maxit 100 --out l4_jax_gamg_1e-6.json > l4_jax_1e-6.log 2>&1

echo "Done!"
