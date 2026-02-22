# Step 24 precision sweep (custom HE solver)

- Level: 1
- Step: 24 (restart from step 23)
- PC: hypre
- Max Newton iterations: 300

| KSP | ksp_rtol | Return | Newton iters | Final energy | Message |
|---|---:|---:|---:|---:|---|
| cg | 0.1 | 0 | 300 | 3169630.917585 | Maximum number of iterations reached |
| cg | 0.01 | 0 | 300 | 2372625.538376 | Maximum number of iterations reached |
| cg | 0.001 | 0 | 300 | 17876779.349843 | Maximum number of iterations reached |
| cg | 0.0001 | 0 | 300 | 27355402.06771 | Maximum number of iterations reached |
| cg | 1e-05 | 0 | 300 | 68287873.676586 | Maximum number of iterations reached |
| cg | 1e-06 | 0 | 300 | 64779774.238031 | Maximum number of iterations reached |
| gmres | 0.1 | 0 | 24 | 197.749038 | Energy change converged |
| gmres | 0.01 | 0 | 22 | 197.748883 | Energy change converged |
| gmres | 0.001 | 0 | 23 | 197.748452 | Energy change converged |
| gmres | 0.0001 | 0 | 23 | 197.748432 | Energy change converged |
| gmres | 1e-05 | 0 | 23 | 197.748427 | Energy change converged |
| gmres | 1e-06 | 0 | 23 | 197.748436 | Energy change converged |

Profile CSV: `experiment_scripts/he_step24_precision_sweep/step24_convergence_profiles.csv`
Summary JSON: `experiment_scripts/he_step24_precision_sweep/step24_precision_summary.json`
