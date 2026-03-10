# GL Trust-Region Diagnostic Sweep

Benchmark: `level 9`, `np=32`.

| Solver | Config | Kind | Result | Time [s] | Newton | Linear | Final energy | Energy gap vs LS ref | Final ||g|| | Grad ratio | Accepted | TR rejects | Max KSP it | Dominant KSP reason | Tail dir norm | Message |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| fenics_custom | ls_ref | line_search | completed | 2.276 | 8 | 37 | 0.345626 | 0.000000 | 0.000000 | 0.000 | 8 | 0 | 11 | CONVERGED_RTOL | 578.372023 | Converged (energy, step, gradient) |
| fenics_custom | ls_loose | line_search | completed | 0.882 | 6 | 8 | 0.345626 | 0.000000 | 0.000014 | 0.015 | 6 | 0 | 2 | CONVERGED_RTOL | 465.154407 | Converged (energy, step, gradient) |
| fenics_custom | tr_stcg_postls_r0_05 | trust_stcg | failed | 11.689 | 100 | 100 | 0.663851 | 0.318224 | 0.000928 | 1.002 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000928 | Maximum number of iterations reached |
| fenics_custom | tr_stcg_postls_r0_2 | trust_stcg | failed | 14.807 | 100 | 100 | 0.664390 | 0.318764 | 0.000927 | 1.001 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| fenics_custom | tr_stcg_postls_r0_5 | trust_stcg | failed | 12.795 | 100 | 100 | 0.664550 | 0.318924 | 0.000927 | 1.001 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| fenics_custom | tr_stcg_postls_r1_0 | trust_stcg | failed | 13.601 | 100 | 100 | 0.664112 | 0.318486 | 0.000928 | 1.002 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000928 | Maximum number of iterations reached |
| fenics_custom | tr_stcg_postls_r2_0 | trust_stcg | failed | 11.690 | 100 | 100 | 0.664195 | 0.318569 | 0.000928 | 1.002 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000928 | Maximum number of iterations reached |
| fenics_custom | tr_stcg_postls_r4_0 | trust_stcg | failed | 12.029 | 100 | 100 | 0.663251 | 0.317625 | 0.000929 | 1.003 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000929 | Maximum number of iterations reached |
| fenics_custom | tr_stcg_only_r1_0 | trust_stcg | failed | 11.905 | 100 | 100 | 0.662634 | 0.317008 | 0.000930 | 1.004 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000930 | Maximum number of iterations reached |
| fenics_custom | tr_stcg_postls_r1_0_ls1e_3 | trust_stcg | failed | 15.204 | 100 | 100 | 0.662542 | 0.316916 | 0.000930 | 1.004 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000930 | Maximum number of iterations reached |
| fenics_custom | tr_stcg_postls_r1_0_ksp1e_8_it200 | trust_stcg | failed | 15.588 | 100 | 100 | 0.664112 | 0.318486 | 0.000928 | 1.002 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000928 | Maximum number of iterations reached |
| fenics_custom | tr_stcg_postls_r1_0_int0_1 | trust_stcg | failed | 11.708 | 100 | 100 | 0.664186 | 0.318560 | 0.000928 | 1.002 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000928 | Maximum number of iterations reached |
| fenics_custom | tr_stcg_postls_r1_0_maxit300 | trust_stcg | failed | 36.894 | 300 | 300 | 0.663776 | 0.318149 | 0.000928 | 1.002 | 300 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000928 | Maximum number of iterations reached |
| fenics_custom | tr_2d_gmres_hypre_r0_2 | trust_2d | failed | 21.215 | 100 | 567 | 0.537402 | 0.191776 | 0.000805 | 0.870 | 100 | 0 | 12 | CONVERGED_RTOL | 0.020112 | Maximum number of iterations reached |
| fenics_custom | tr_2d_gmres_hypre_r1_0 | trust_2d | failed | 24.555 | 100 | 746 | 0.516303 | 0.170677 | 0.000854 | 0.922 | 100 | 0 | 16 | CONVERGED_RTOL | 0.023957 | Maximum number of iterations reached |
| fenics_custom | tr_2d_gmres_hypre_r2_0 | trust_2d | failed | 21.595 | 100 | 549 | 0.530701 | 0.185075 | 0.000738 | 0.797 | 100 | 0 | 11 | CONVERGED_RTOL | 0.019263 | Maximum number of iterations reached |
| jax_petsc_element | ls_ref | line_search | completed | 1.948 | 7 | 39 | 0.345626 | 0.000000 | 0.000002 | 0.002 | 7 | 0 | 12 | CONVERGED_RTOL | 705.512514 | Converged (energy, step, gradient) |
| jax_petsc_element | ls_loose | line_search | completed | 1.123 | 6 | 9 | 0.345626 | 0.000000 | 0.000004 | 0.004 | 6 | 0 | 3 | CONVERGED_RTOL | 322.720392 | Converged (energy, step, gradient) |
| jax_petsc_element | tr_stcg_postls_r0_05 | trust_stcg | failed | 16.033 | 100 | 100 | 0.665122 | 0.319496 | 0.000926 | 1.000 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000926 | Maximum number of iterations reached |
| jax_petsc_element | tr_stcg_postls_r0_2 | trust_stcg | failed | 15.559 | 100 | 100 | 0.664823 | 0.319197 | 0.000927 | 1.001 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| jax_petsc_element | tr_stcg_postls_r0_5 | trust_stcg | failed | 16.014 | 100 | 100 | 0.664494 | 0.318868 | 0.000927 | 1.001 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| jax_petsc_element | tr_stcg_postls_r1_0 | trust_stcg | failed | 15.916 | 100 | 100 | 0.664392 | 0.318766 | 0.000927 | 1.001 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| jax_petsc_element | tr_stcg_postls_r2_0 | trust_stcg | failed | 15.703 | 100 | 100 | 0.664678 | 0.319052 | 0.000927 | 1.001 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| jax_petsc_element | tr_stcg_postls_r4_0 | trust_stcg | failed | 15.768 | 100 | 100 | 0.664217 | 0.318591 | 0.000928 | 1.002 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000928 | Maximum number of iterations reached |
| jax_petsc_element | tr_stcg_only_r1_0 | trust_stcg | failed | 9.855 | 100 | 100 | 0.664428 | 0.318801 | 0.000927 | 1.001 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| jax_petsc_element | tr_stcg_postls_r1_0_ls1e_3 | trust_stcg | failed | 21.813 | 100 | 100 | 0.664344 | 0.318717 | 0.000927 | 1.001 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| jax_petsc_element | tr_stcg_postls_r1_0_ksp1e_8_it200 | trust_stcg | failed | 16.036 | 100 | 100 | 0.664392 | 0.318766 | 0.000927 | 1.001 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| jax_petsc_element | tr_stcg_postls_r1_0_int0_1 | trust_stcg | failed | 14.603 | 100 | 100 | 0.664481 | 0.318854 | 0.000927 | 1.001 | 100 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| jax_petsc_element | tr_stcg_postls_r1_0_maxit300 | trust_stcg | failed | 46.736 | 300 | 300 | 0.664056 | 0.318429 | 0.000927 | 1.001 | 300 | 0 | 1 | DIVERGED_INDEFINITE_PC | 0.000927 | Maximum number of iterations reached |
| jax_petsc_element | tr_2d_gmres_hypre_r0_2 | trust_2d | failed | 22.415 | 100 | 580 | 0.525262 | 0.179636 | 0.000739 | 0.798 | 100 | 0 | 15 | CONVERGED_RTOL | 0.018590 | Maximum number of iterations reached |
| jax_petsc_element | tr_2d_gmres_hypre_r1_0 | trust_2d | failed | 22.562 | 100 | 555 | 0.535560 | 0.189934 | 0.000699 | 0.755 | 100 | 0 | 12 | CONVERGED_RTOL | 0.021854 | Maximum number of iterations reached |
| jax_petsc_element | tr_2d_gmres_hypre_r2_0 | trust_2d | failed | 22.547 | 100 | 556 | 0.529731 | 0.184104 | 0.000750 | 0.810 | 100 | 0 | 17 | CONVERGED_RTOL | 0.018831 | Maximum number of iterations reached |

## Notes

- `Kind = trust_stcg` means PETSc trust-subproblem KSP (`ksp_type=stcg`).
- `Kind = trust_2d` means the older reduced 2D trust hybrid: trust region stays on, but the inner linear solve uses the standard Newton direction (`ksp_type=gmres`) and the trust step is built in the reduced subspace.
- `Tail dir norm` is the median of `||p||` over the last 10 iterations, reconstructed from `step_norm / |alpha|`.
