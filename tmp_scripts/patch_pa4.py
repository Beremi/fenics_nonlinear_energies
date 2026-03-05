import re

# In parallel_hessian_dof.py, there's a base constructor and a subclass.
file = "HyperElasticity3D_jax_petsc/parallel_hessian_dof.py"
with open(file, 'r') as f: lines = f.readlines()

for i, line in enumerate(lines):
    if "reorder=reorder" in line:
        if "ownership_block_size" not in lines[i+1]:
            lines.insert(i+1, "            ownership_block_size=3,\n")

with open(file, 'w') as f: f.writelines(lines)

file2 = "tools_petsc4py/parallel_assembler.py"
with open(file2, 'r') as f: lines2 = f.readlines()

for i, line in enumerate(lines2):
    if "def __init__(" in line and "LocalColoringAssemblerBase" in "".join(lines2[i-5:i]):
        for j in range(i, i+30):
            if "reorder=True," in lines2[j]:
                if "ownership_block_size" not in lines2[j+1]:
                    lines2.insert(j+1, "        ownership_block_size=1,\n")
                break
    if "super().__init__(" in line:
        for j in range(i, i+30):
            if "reorder=reorder," in lines2[j]:
                if "ownership_block_size" not in lines2[j+1]:
                    lines2.insert(j+1, "            ownership_block_size=ownership_block_size,\n")
                break

with open(file2, 'w') as f: f.writelines(lines2)

