#!/bin/bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

echo "Running JAX+PETSc SFD..."
mpirun -n 32 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py --level 3 --steps 1 --total_steps 24 --profile performance --assembly_mode sfd --save_linear_timing > tmp_work/jax_sfd_n32.log 2>&1

echo "Parsing..."
python3 tools_petsc4py/fix_json.py < tmp_work/jax_sfd_n32.log
