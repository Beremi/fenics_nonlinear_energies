"""Test: can we implement a custom golden-section energy line search in PETSc4py?"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import ufl

# ---------- tiny mesh + GL problem ----------
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)
eps_val = 0.01
F_form = (eps_val * ufl.inner(ufl.grad(u), ufl.grad(v)) + (u**2 - 1) * u * v) * ufl.dx

# Check if SNESLineSearch has shell type support
snes = PETSc.SNES().create(MPI.COMM_WORLD)
ls = snes.getLineSearch()
print("LineSearch type before:", ls.getType())
print("LineSearch methods:", [m for m in dir(ls) if not m.startswith('_')])

# Try shell type
try:
    ls.setType("shell")
    print("Shell type set OK:", ls.getType())
except Exception as e:
    print(f"Shell type FAIL: {e}")

# Check if we can set a custom apply function
if hasattr(ls, 'setShell'):
    print("setShell exists")
else:
    print("setShell NOT found")

# Try alternative: precheck/postcheck
try:
    def precheck(ls, x, y):
        print("  precheck called")
        return False
    ls.setPreCheck(precheck)
    print("setPreCheck OK")
except Exception as e:
    print(f"setPreCheck FAIL: {e}")

snes.destroy()
print("\nDone.")
