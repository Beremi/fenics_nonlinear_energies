import json

file = "HyperElasticity3D_jax_petsc/solve_HE_dof.py"
with open(file, 'r') as f:
    text = f.read()

# after assembly
text = text.replace('ksp.setOperators(A)', """
                    A.setOption(A.Option.NEW_NONZERO_LOCATIONS, False)
                    A.setOption(A.Option.NEW_NONZERO_ALLOCATION_ERR, True)
                    ksp.setOperators(A)
                    if hasattr(ksp, '_set_first'):
                        ksp.setReusePreconditioner(True)
                    ksp._set_first = False
""")

with open('HyperElasticity3D_jax_petsc/solve_HE_dof_test.py', 'w') as f:
    f.write(text)
