import re

file = "HyperElasticity3D_jax_petsc/parallel_hessian_dof.py"
with open(file, 'r') as f: text = f.read()

text = re.sub(r'reorder=reorder,', 'reorder=reorder,\n            ownership_block_size=3,', text)

with open(file, 'w') as f: f.write(text)

file2 = "tools_petsc4py/parallel_assembler.py"
with open(file2, 'r') as f: text2 = f.read()

text2 = re.sub(r'reorder=True,', 'reorder=True,\n        ownership_block_size=1,', text2)
text2 = re.sub(r'reorder=reorder,', 'reorder=reorder,\n            ownership_block_size=ownership_block_size,', text2)

with open(file2, 'w') as f: f.write(text2)

