# HE STCG KSP Max-It Comparison

Level `4`, `32` MPI ranks, full `24/24` trajectory.

Shared settings:

- trust region on
- post-STCG line search on
- `linesearch_tol=1e-1`
- `trust_radius_init=0.5`
- `trust_shrink=0.5`
- `trust_expand=1.5`
- `trust_eta_shrink=0.05`
- `trust_eta_expand=0.75`
- rebuild PC every Newton iteration (`pc_setup_on_ksp_cap=False`)
- `ksp_type=stcg`, `pc_type=gamg`
- `ksp_rtol=1e-1`

| Backend | `ksp_max_it` | All 24 converged | Total [s] | Newton | Linear | Final energy | Max step [s] | Max KSP it | KSP cap hits |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `fenics_custom` | `30` | yes | 360.318 | 708 | 16046 | 87.722387 | 17.710 | 30 | 415 |
| `fenics_custom` | `100` | yes | 430.905 | 708 | 22611 | 87.722017 | 20.555 | 97 | 0 |
| `jax_petsc_element` | `30` | yes | 479.569 | 825 | 18925 | 87.721975 | 23.191 | 30 | 496 |
| `jax_petsc_element` | `100` | yes | 557.517 | 814 | 27187 | 87.721835 | 37.287 | 100 | 20 |

Readout:

- `ksp_max_it=30` is faster than `100` for both backends on this fine case.
- FEniCS custom keeps the same Newton count at `30` and `100`, so the slower
  `100` run is purely extra linear-solver work.
- JAX + PETSc reduces Newton iterations slightly at `100`, but the extra linear
  work dominates and the run is still clearly slower.
- Final energies match closely across both `ksp_max_it` choices.
