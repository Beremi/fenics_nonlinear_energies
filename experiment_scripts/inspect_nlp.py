#!/usr/bin/env python3
import sys
from dolfinx.fem.petsc import NonlinearProblem
import inspect
src = inspect.getsource(NonlinearProblem)
sys.stdout.write(src[4000:9000])
sys.stdout.write("\n---END---\n")
sys.stdout.flush()
