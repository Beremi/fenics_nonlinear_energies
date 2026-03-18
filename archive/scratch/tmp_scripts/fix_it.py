import sys

with open("HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py", "r") as f:
    text = f.read()

import re
old_func = re.search(r"def build_nullspace\(.*?\n# ---------------------------------------------------------------------------", text, re.DOTALL)

with open("HyperElasticity3D_fenics/solve_HE_snes_newton.py", "r") as f:
    text2 = f.read()
new_func = re.search(r"def build_nullspace\(.*?\n# ---------------------------------------------------------------------------", text2, re.DOTALL)

text = text.replace(old_func.group(0), new_func.group(0))

with open("HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py", "w") as f:
    f.write(text)
