#!/bin/bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

echo "HYPRE: Old JAX 32"
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 3 --steps 1 --total_steps 96 --no-reorder --save_linear_timing --ksp_type gmres --pc_type hypre --out old_jax_hypre_32.json > /dev/null

echo "HYPRE: New JAX 32"
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 3 --steps 1 --total_steps 96 --reorder --save_linear_timing --ksp_type gmres --pc_type hypre --out new_jax_hypre_32.json > /dev/null

echo "HYPRE: Old JAX 16"
mpirun -n 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 3 --steps 1 --total_steps 96 --no-reorder --save_linear_timing --ksp_type gmres --pc_type hypre --out old_jax_hypre_16.json > /dev/null
