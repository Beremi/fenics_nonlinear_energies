"""
Graph coloring utilities for sparse-finite-difference (SFD) Hessian recovery.

Provides four backends:
  1. **Custom**    – fast C greedy coloring on explicit A² (fastest serial, good quality)
  2. **igraph**    – sequential greedy coloring on explicit A² (best color quality)
  3. **PETSc**     – distance-2 MatColoring via petsc4py + ctypes (serial & parallel,
                     recommended types: 'sl' or 'id' for quality, 'greedy' for speed)
  4. **NetworkX**  – pure-Python greedy coloring on explicit A² (slow, reference only)
"""
