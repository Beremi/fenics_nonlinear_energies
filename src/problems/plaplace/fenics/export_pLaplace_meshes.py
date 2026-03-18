"""
[OBSOLETE] Export pLaplace meshes to XDMF format.

This script was previously used to derive mesh_level_N.{xdmf,h5} from the JAX
source files pLaplace_levelN.h5 for use with FEniCS XDMFFile.read_mesh().

The FEniCS solvers now load meshes directly from pLaplace_levelN.h5 via h5py +
dolfinx.mesh.create_mesh, so this script and the derived XDMF/H5 files are no
longer needed. This file is kept for reference only.
"""
import os
from mpi4py import MPI
import dolfinx
from src.problems.plaplace.jax.mesh import MeshpLaplace2D
from src.core.problem_data.hdf5 import MESH_DATA_ROOT
import ufl
import basix.ufl
from dolfinx.io import XDMFFile

# Create output directory if it doesn't exist
output_dir = MESH_DATA_ROOT / "pLaplace"
os.makedirs(output_dir, exist_ok=True)

# Define number of spatial dimensions and element type
c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))

# Loop over mesh levels
for mesh_level in range(1, 10):
    # Generate mesh from pLaplace2D
    mesh_plaplace2d = MeshpLaplace2D(mesh_level=mesh_level)
    triangles = mesh_plaplace2d.params["elems"]
    points = mesh_plaplace2d.params["nodes"]

    # Create DOLFINx mesh
    msh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, triangles, points, c_el)

    # Define file name for each mesh level
    filename = os.path.join(str(output_dir), f"mesh_level_{mesh_level}.xdmf")

    # Save mesh in XDMF format
    with XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf_file:
        xdmf_file.write_mesh(msh)
        print(f"Saved mesh for level {mesh_level} to {filename}")

print("All meshes have been created and saved.")
