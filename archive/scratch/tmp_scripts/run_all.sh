#!/bin/bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

echo "3/6 FEniCS 16"
mpirun -n 16 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 3 --steps 1 --total_steps 96 --save_linear_timing --ksp_type cg --pc_type gamg --out fenics_detailed_16.json

echo "6/6 FEniCS 32"
mpirun -n 32 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 3 --steps 1 --total_steps 96 --save_linear_timing --ksp_type cg --pc_type gamg --out fenics_detailed_32.json

echo "1/6 Old JAX 16"
mpirun -n 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 3 --steps 1 --total_steps 96 --no-reorder --profile performance --save_linear_timing --ksp_type cg --pc_type gamg --out old_detailed_16.json

echo "2/6 New JAX 16"
mpirun -n 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 3 --steps 1 --total_steps 96 --reorder --profile performance --save_linear_timing --ksp_type cg --pc_type gamg --out new_detailed_16.json

echo "4/6 Old JAX 32"
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 3 --steps 1 --total_steps 96 --no-reorder --profile performance --save_linear_timing --ksp_type cg --pc_type gamg --out old_detailed_32.json

echo "5/6 New JAX 32"
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 3 --steps 1 --total_steps 96 --reorder --profile performance --save_linear_timing --ksp_type cg --pc_type gamg --out new_detailed_32.json
