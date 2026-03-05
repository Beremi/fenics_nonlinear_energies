#!/bin/bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

echo "Level 4, n=32 Tests..."

# GAMG comparison
echo "Running FEniCS GAMG (32 cores, L4)"
mpirun -n 32 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 4 --steps 1 --total_steps 96 --save_linear_timing --ksp_type cg --pc_type gamg --ksp_rtol 1e-3 --ksp_max_it 10000 --out l4_fenics_exact_gamg.json > /dev/null

echo "Running New JAX GAMG (32 cores, L4)"
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 4 --steps 1 --total_steps 96 --reorder --profile performance --save_linear_timing --ksp_type cg --pc_type gamg --ksp_rtol 1e-3 --ksp_max_it 10000 --out l4_jax_exact_gamg.json > /dev/null

# HYPRE comparison
echo "Running FEniCS HYPRE (32 cores, L4)"
mpirun -n 32 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 4 --steps 1 --total_steps 96 --save_linear_timing --ksp_type gmres --pc_type hypre --hypre_nodal_coarsen -1 --hypre_vec_interp_variant -1 --ksp_rtol 1e-3 --ksp_max_it 10000 --out l4_fenics_exact_hypre.json > /dev/null

echo "Running New JAX HYPRE (32 cores, L4)"
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 4 --steps 1 --total_steps 96 --reorder --profile performance --save_linear_timing --ksp_type gmres --pc_type hypre --ksp_rtol 1e-3 --ksp_max_it 10000 --out l4_jax_exact_hypre.json > /dev/null

echo "Done!"
