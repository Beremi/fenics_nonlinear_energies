# GL Minimizer Sweep Summary

| Solver | Config | Total time [s] | Newton | Linear | Final energy | Result |
|---|---|---:|---:|---:|---:|---|
| fenics_custom | ls_loose | 0.995 | 6 | 8 | 0.345626 | completed |
| fenics_custom | ls_ref | 2.299 | 8 | 37 | 0.345626 | completed |
| fenics_custom | tr_stcg_r0_5 | 15.187 | 100 | 100 | 0.664550 | failed |
| fenics_custom | tr_stcg_r1_0 | 20.812 | 100 | 100 | 0.664112 | failed |
| fenics_custom | tr_stcg_r2_0 | 13.878 | 100 | 100 | 0.664195 | failed |
| jax_petsc_element | ls_loose | 1.257 | 6 | 9 | 0.345626 | completed |
| jax_petsc_element | ls_ref | 2.126 | 7 | 39 | 0.345626 | completed |
| jax_petsc_element | tr_stcg_r0_5 | 16.525 | 100 | 100 | 0.664494 | failed |
| jax_petsc_element | tr_stcg_r1_0 | 16.444 | 100 | 100 | 0.664392 | failed |
| jax_petsc_element | tr_stcg_r2_0 | 16.602 | 100 | 100 | 0.664678 | failed |
