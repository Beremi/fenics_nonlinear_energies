#!/bin/bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

echo "FEniCS HYPRE 16"
mpirun -n 16 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 3 --steps 1 --total_steps 96 --save_linear_timing --ksp_type gmres --pc_type hypre --hypre_nodal_coarsen -1 --hypre_vec_interp_variant -1 --out fenics_hypre_16.json > /dev/null

echo "FEniCS GAMG 16"
mpirun -n 16 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 3 --steps 1 --total_steps 96 --save_linear_timing --ksp_type cg --pc_type gamg --out fenics_gamg_16.json > /dev/null

