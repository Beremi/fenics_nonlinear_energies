#!/bin/bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

# GAMG comparison
echo "Running FEniCS GAMG (16)"
mpirun -n 16 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 3 --steps 1 --total_steps 96 --save_linear_timing --ksp_type cg --pc_type gamg --ksp_rtol 1e-3 --ksp_max_it 10000 --out fenics_exact_gamg.json > /dev/null

echo "Running New JAX GAMG (16)"
mpirun -n 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 3 --steps 1 --total_steps 96 --reorder --profile performance --save_linear_timing --ksp_type cg --pc_type gamg --ksp_rtol 1e-3 --ksp_max_it 10000 --out jax_exact_gamg.json > /dev/null

# HYPRE comparison
echo "Running FEniCS HYPRE (16)"
mpirun -n 16 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 3 --steps 1 --total_steps 96 --save_linear_timing --ksp_type gmres --pc_type hypre --hypre_nodal_coarsen -1 --hypre_vec_interp_variant -1 --ksp_rtol 1e-3 --ksp_max_it 10000 --out fenics_exact_hypre.json > /dev/null

echo "Running New JAX HYPRE (16)"
mpirun -n 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 3 --steps 1 --total_steps 96 --reorder --profile performance --save_linear_timing --ksp_type gmres --pc_type hypre --ksp_rtol 1e-3 --ksp_max_it 10000 --out jax_exact_hypre.json > /dev/null

