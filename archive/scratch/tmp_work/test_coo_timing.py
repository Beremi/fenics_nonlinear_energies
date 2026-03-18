import time
import numpy as np
from petsc4py import PETSc

# simple test to see petsc4py COO overhead
n = 1000000
rows = np.arange(n, dtype=np.int32)
cols = np.arange(n, dtype=np.int32)
vals = np.ones(n, dtype=np.float64)

A = PETSc.Mat().create()
A.setSizes((n, n))
A.setType(PETSc.Mat.Type.SEQAIJ)
A.setPreallocationCOO(rows, cols)

t0 = time.perf_counter()
A.setValuesCOO(vals, PETSc.InsertMode.INSERT_VALUES)
t1 = time.perf_counter()
A.assemble()
t2 = time.perf_counter()

print(f"setValuesCOO: {t1-t0:.4f} s")
print(f"assemble:    {t2-t1:.4f} s")
