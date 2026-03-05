#!/bin/bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

echo "3/6 FEniCS 16"
mpirun -n 16 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 3 --steps 1 --total_steps 96 --save_linear_timing --ksp_type gmres --pc_type hypre --hypre_nodal_coarsen -1 --hypre_vec_interp_variant -1 --out fenics_detailed_16.json

echo "6/6 FEniCS 32"
mpirun -n 32 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 3 --steps 1 --total_steps 96 --save_linear_timing --ksp_type gmres --pc_type hypre --hypre_nodal_coarsen -1 --hypre_vec_interp_variant -1 --out fenics_detailed_32.json
