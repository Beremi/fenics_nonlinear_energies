import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_local = 100000
n = n_local * size
lo = rank * n_local
hi = lo + n_local

# Ranks insert into their own rows, but to ANY column
np.random.seed(rank)
rows = np.repeat(np.arange(lo, hi, dtype=np.int32), 20)
cols = np.random.randint(0, n, size=n_local * 20, dtype=np.int32)
vals = np.ones(len(rows), dtype=np.float64)

A = PETSc.Mat().create(comm)
A.setSizes(((n_local, n), (n_local, n)))
A.setType(PETSc.Mat.Type.MPIAIJ)
A.setPreallocationCOO(rows, cols)

# Warm up
A.setValuesCOO(vals, PETSc.InsertMode.ADD_VALUES)
A.assemble()

comm.barrier()
t0 = time.perf_counter()
A.setValuesCOO(vals, PETSc.InsertMode.ADD_VALUES)
t1 = time.perf_counter()
A.assemble()
t2 = time.perf_counter()

if rank == 0:
    print(f"setValuesCOO: {t1-t0:.4f} s")
    print(f"assemble:    {t2-t1:.4f} s")
