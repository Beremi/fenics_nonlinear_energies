import sys
sys.path.insert(0, ".")
from HyperElasticity3D_jax_petsc.mesh import MeshHyperElasticity3D
m = MeshHyperElasticity3D(3)
p, adj, u = m.get_data()
print(f"n_free: {len(p['freedofs'])}, adj shape: {adj.shape}")
