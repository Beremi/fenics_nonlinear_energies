# Representative Matrix Post-Cleanup

| Case | Pre-finish | Post-cleanup | Notes |
|---|---|---|---|
| `plaplace_jax_l5_np1.json` | `{'status': 'completed', 'iters': 6, 'energy': -7.943}` | `{'status': 'completed', 'iters': 6, 'energy': -7.943}` | energy_rel_diff=0.000e+00 |
| `plaplace_fenics_custom_l5_np2.json` | `{'status': 'completed', 'iters': 5, 'energy': -7.9429687182}` | `{'status': 'completed', 'iters': 5, 'energy': -7.9429687182}` | energy_rel_diff=0.000e+00 |
| `plaplace_jax_petsc_l5_np2.json` | `{'status': 'completed', 'iters': 6, 'energy': -7.9429687193}` | `{'status': 'completed', 'iters': 6, 'energy': -7.9429687193}` | energy_rel_diff=0.000e+00 |
| `gl_fenics_custom_l5_np2.json` | `{'status': 'completed', 'iters': 7, 'energy': 0.3462323651}` | `{'status': 'completed', 'iters': 7, 'energy': 0.3462323651}` | energy_rel_diff=0.000e+00 |
| `gl_jax_petsc_l5_np2.json` | `{'status': 'completed', 'steps': 1, 'newton_iters': 12, 'energy': 0.34598730315972287}` | `{'status': 'completed', 'steps': 1, 'newton_iters': 12, 'energy': 0.34598730315972287}` | energy_rel_diff=0.000e+00 |
| `he_jax_l1_steps24_np1.json` | `{'status': 'completed', 'steps': 24, 'newton_iters': 559, 'energy': 197.74950930751214}` | `{'status': 'completed', 'steps': 24, 'newton_iters': 559, 'energy': 197.74950930751214}` | energy_rel_diff=0.000e+00 |
| `he_fenics_custom_l1_steps24_np2.json` | `{'status': 'failed', 'note': 'no-json'}` | `{'status': 'completed', 'steps': 24, 'newton_iters': 795, 'energy': 197.754815}` | new JSON after previously crashing case |
| `he_jax_petsc_l1_steps24_np2.json` | `{'status': 'completed', 'steps': 24, 'newton_iters': 528, 'energy': 197.75512338531667}` | `{'status': 'completed', 'steps': 24, 'newton_iters': 528, 'energy': 197.75512338531667}` | energy_rel_diff=0.000e+00 |
| `topology_serial_nx192_ny96_np1.json` | `{'status': 'completed', 'outer_iterations': 121, 'compliance': 4.155705577015535, 'volume_fraction': 0.40000000000006014, 'final_p': nan}` | `{'status': 'completed', 'outer_iterations': 121, 'compliance': 4.155705577015535, 'volume_fraction': 0.40000000000006014, 'final_p': nan}` | compliance_rel_diff=0.000e+00 |
| `topology_parallel_nx768_ny384_np2.json` | `{'status': 'completed', 'outer_iterations': 72, 'compliance': 8.947270662878754, 'volume_fraction': 0.39320397728290735, 'final_p': nan}` | `{'status': 'completed', 'outer_iterations': 72, 'compliance': 8.947270662878754, 'volume_fraction': 0.39320397728290735, 'final_p': nan}` | compliance_rel_diff=0.000e+00 |
