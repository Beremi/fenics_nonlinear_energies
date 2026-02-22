# Step 24 precision sweep (custom HE solver)

- Level: 1
- Step: 24 (restart from step 23)
- PC: hypre
- Max Newton iterations: 100

| KSP | ksp_rtol | Return | Newton iters | Final energy | Message |
|---|---:|---:|---:|---:|---|
| gmres | 0.1 | 0 | 24 | 197.749038 | Energy change converged |
| gmres | 0.01 | 0 | 22 | 197.748883 | Energy change converged |
| gmres | 0.001 | 0 | 23 | 197.74843 | Energy change converged |
| gmres | 0.0001 | 0 | 22 | 197.74842 | Energy change converged |
| gmres | 1e-05 | 0 | 23 | 197.748427 | Energy change converged |
| gmres | 1e-06 | 0 | 23 | 197.748428 | Energy change converged |

Profile CSV: `experiment_scripts/he_step24_precision_sweep_maxit100_gmres/step24_convergence_profiles.csv`
Summary JSON: `experiment_scripts/he_step24_precision_sweep_maxit100_gmres/step24_precision_summary.json`
