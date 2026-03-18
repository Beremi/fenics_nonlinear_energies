"""
Export Ginzburg-Landau 2D meshes from HDF5 (JAX format) to XDMF (DOLFINx format).

Reads the pre-generated HDF5 mesh files used by the JAX solver and converts
them to XDMF files that DOLFINx can read directly.

Usage (inside devcontainer or Docker):
  python3 src/problems/ginzburg_landau/fenics/export_GL_meshes.py

Produces mesh_data/GinzburgLandau/GL_mesh_level_{n}.xdmf for n = 2..9.
"""
import os
import numpy as np
from mpi4py import MPI
from src.problems.ginzburg_landau.jax.mesh import MeshGL2D
from src.core.problem_data.hdf5 import MESH_DATA_ROOT
import ufl
import basix.ufl
import dolfinx
from dolfinx.io import XDMFFile

# Output directory (same as source HDF5 files)
output_dir = MESH_DATA_ROOT / "GinzburgLandau"
os.makedirs(output_dir, exist_ok=True)

# P1 triangle mesh element
c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))

# Loop over all available mesh levels (2–9)
for mesh_level in range(2, 10):
    mesh_gl = MeshGL2D(mesh_level=mesh_level)
    triangles = np.asarray(mesh_gl.params["elems"], dtype=np.int64)
    points = np.asarray(mesh_gl.params["nodes"], dtype=np.float64)

    # Create DOLFINx mesh from raw connectivity + coordinates
    msh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, triangles, points, c_el)

    # Save as XDMF
    filename = os.path.join(str(output_dir), f"GL_mesh_level_{mesh_level}.xdmf")
    with XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf_file:
        xdmf_file.write_mesh(msh)
        print(f"Saved mesh for level {mesh_level} to {filename}")

print("All GL meshes have been created and saved.")
