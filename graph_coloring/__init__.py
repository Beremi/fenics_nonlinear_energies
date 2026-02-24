"""
Graph coloring utilities for sparse-finite-difference (SFD) Hessian recovery.

Provides three backends:
  1. **igraph**    – sequential greedy coloring on explicit A² (best color quality)
  2. **PETSc**     – distance-2 MatColoring via petsc4py + ctypes (serial & parallel,
                     recommended types: 'sl' or 'id' for quality, 'greedy' for speed)
  3. **NetworkX**  – pure-Python greedy coloring on explicit A² (slow, reference only)
"""
