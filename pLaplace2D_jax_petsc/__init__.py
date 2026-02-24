"""
p-Laplace 2D solver — parallel JAX + PETSc (MPI).

Combines JAX automatic differentiation with MPI-parallel sparse-finite-difference
(SFD) Hessian assembly and distributed PETSc KSP solve.  Every rank holds the
full mesh data and JAX JIT-compiled functions; parallelism comes from:

  1. Distributed graph coloring (multistart randomised, graph_coloring/)
  2. Round-robin distribution of colors for Hessian-vector products
  3. PETSc MPIAIJ distributed matrix + KSP (CG + HYPRE AMG)

The Newton solver reuses tools_petsc4py.minimizers.newton (golden-section
line search on [-0.5, 2], same algorithm as the serial JAX solver).
"""
