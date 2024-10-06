import os
from mpi4py import MPI
import dolfinx
from pLaplace2D.mesh import MeshpLaplace2D
import ufl
import basix.ufl
from dolfinx.io import XDMFFile

# Create output directory if it doesn't exist
output_dir = "pLaplace_fenics_mesh"
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
    filename = os.path.join(output_dir, f"mesh_level_{mesh_level}.xdmf")

    # Save mesh in XDMF format
    with XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf_file:
        xdmf_file.write_mesh(msh)
        print(f"Saved mesh for level {mesh_level} to {filename}")

print("All meshes have been created and saved.")
