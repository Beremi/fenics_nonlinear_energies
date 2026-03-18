#!/usr/bin/env python3
"""Debug: compare SFD Hessian vs element-level Hessian at initial state."""

import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from HyperElasticity3D_jax_petsc.mesh import MeshHyperElasticity3D
from HyperElasticity3D_jax_petsc.parallel_hessian_dof import LocalColoringAssembler
from HyperElasticity3D_jax_petsc.rotate_boundary import rotate_right_face_from_reference

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

level = 3
total_steps = 24
rotation_per_iter = 4.0 * 2.0 * np.pi / float(total_steps)
angle = 1 * rotation_per_iter

mesh_obj = MeshHyperElasticity3D(level)
params, adjacency, u_init = mesh_obj.get_data()

assembler = LocalColoringAssembler(
    params=params,
    comm=comm,
    adjacency=adjacency,
    ksp_rtol=1e-1,
    ksp_type="gmres",
    pc_type="gamg",
    ksp_max_it=30,
    use_near_nullspace=True,
    pc_options={"pc_gamg_threshold": 0.05, "pc_gamg_agg_nsmooths": 1},
    reorder=False,
    hvp_eval_mode="sequential",
)
assembler.A.setBlockSize(3)
assembler.setup_element_hessian()

# Apply BC
u0_step = rotate_right_face_from_reference(
    params["u_0_ref"], params["nodes2coord"], angle, params["right_nodes"]
)
assembler.update_dirichlet(u0_step)

# Initial state
u_init_reordered = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
lo, hi = assembler.part.lo, assembler.part.hi
u_owned = u_init_reordered[lo:hi]

# Assemble SFD Hessian first, save a copy
assembler.assemble_hessian(u_owned, variant=2)
A_sfd_copy = assembler.A.duplicate()
assembler.A.copy(A_sfd_copy)
sfd_norm = A_sfd_copy.norm(PETSc.NormType.FROBENIUS)

# Now assemble element Hessian (overwrites self.A)
assembler.assemble_hessian_element(u_owned)
elem_norm = assembler.A.norm(PETSc.NormType.FROBENIUS)

if rank == 0:
    print(f"SFD  Hessian Frobenius norm: {sfd_norm:.6e}")
    print(f"Elem Hessian Frobenius norm: {elem_norm:.6e}")
    print(f"Ratio (elem/sfd): {elem_norm/sfd_norm:.6f}")

# Compute difference: A_diff = A_elem - A_sfd
A_diff = assembler.A.duplicate()
assembler.A.copy(A_diff)
A_diff.axpy(-1.0, A_sfd_copy)
diff_norm = A_diff.norm(PETSc.NormType.FROBENIUS)

if rank == 0:
    print(f"Difference norm: {diff_norm:.6e}")
    print(f"Relative diff: {diff_norm/sfd_norm:.6e}")

# Check a few diagonal entries
for i in range(min(6, hi - lo)):
    gi = lo + i
    sfd_val = A_sfd_copy.getValue(gi, gi)
    elem_val = assembler.A.getValue(gi, gi)
    if rank == 0:
        print(f"  diag[{gi}]: sfd={sfd_val:.6e}  elem={elem_val:.6e}  diff={abs(elem_val-sfd_val):.6e}")

A_diff.destroy()
A_sfd_copy.destroy()
assembler.cleanup()
