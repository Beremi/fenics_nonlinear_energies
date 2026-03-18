# Parallel JAX Topology Benchmark

This report documents the current `32`-rank parallel topology run on the distributed mesh `768 x 384`.

The implementation follows the current stable parallel path in the repository:

- mechanics solved by PETSc `fgmres + gamg`
- rigid-body near-nullspace supplied to GAMG
- design updated by distributed gradient descent with adaptive golden-section line search (`design_maxit = 20`, `linesearch_tol = 0.1`)
- fixed staircase SIMP continuation up to `p = 10.0`

## Configuration

| Knob | Value |
| --- | --- |
| MPI ranks | 32 |
| Mesh | 768 x 384 |
| Elements | 589824 |
| Free displacement DOFs | 591360 |
| Free design DOFs | 278641 |
| Target volume fraction | 0.4000 |
| SIMP schedule | p = p + 0.2 every 1 outer iterations |
| Final p target | 10.00 |
| Mechanics solver | fgmres + gamg |
| Near-nullspace | True |
| Design LS policy | golden_adaptive |
| Design LS scale | 2.000 |
| Design LS tol mode | relative to bound |
| Graceful stall tol | 0.00 |
| Graceful stall p min | 4.00 |

## Final State

![Final state](artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/final_state.png)

## Convergence History

![Convergence history](artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/convergence_history.png)

## Density Step Size

![Outer density step size](artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/density_step_history.png)

## Density Evolution

![Density evolution](artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/density_evolution.gif)

## Run Summary

| Metric | Value |
| --- | --- |
| Result | completed |
| Outer iterations | 66 |
| Final p | 6.6000 |
| Wall time [s] | 25.291 |
| Setup time [s] | 1.258 |
| Solve time [s] | 24.033 |
| Final compliance | 9.907413 |
| Final volume fraction | 0.374866 |
| Final volume error | -0.025134 |
| Final design change | 0.000000 |
| Final compliance change | 0.001426 |
| Gray fraction (cell 0.1-0.9) | 0.0394 |

## Status Notes

- No zero-step design stall was detected in the saved outer history.

- The run stopped gracefully once both `dtheta` and `dtheta_state` fell below `1.0e-06` at `p >= 4.00`.

## Parallel Work Summary

| Metric | Value |
| --- | --- |
| Total mechanics KSP time [s] | 11.789 |
| Total mechanics scatter time [s] | 1.037 |
| Total design grad time [s] | 1.069 |
| Total design line-search time [s] | 6.921 |
| Total mechanics KSP iterations | 2235 |
| Total design GD iterations | 632 |
| Total design line-search evals | 3792 |

## Outer Iteration Table

| k | p | mech KSP | GD | LS evals | compliance | volume | vol error | dtheta | dC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 9 | 20 | 120 | 3.895134 | 0.669430 | 0.269430 | 0.394912 | nan |
| 2 | 1.00 | 9 | 11 | 66 | 3.620039 | 0.436323 | 0.036323 | 0.399828 | 0.070625 |
| 3 | 1.00 | 9 | 7 | 42 | 5.619814 | 0.364361 | -0.035639 | 0.211775 | 0.552418 |
| 4 | 1.00 | 9 | 5 | 30 | 6.924966 | 0.351152 | -0.048848 | 0.095624 | 0.232241 |
| 5 | 1.00 | 8 | 6 | 36 | 7.165631 | 0.397511 | -0.002489 | 0.248271 | 0.034753 |
| 6 | 1.00 | 9 | 9 | 54 | 5.990178 | 0.469775 | 0.069775 | 0.251593 | 0.164040 |
| 7 | 1.00 | 9 | 7 | 42 | 4.842148 | 0.406107 | 0.006107 | 0.166344 | 0.191652 |
| 8 | 1.00 | 9 | 5 | 30 | 5.785315 | 0.366319 | -0.033681 | 0.127993 | 0.194783 |
| 9 | 1.00 | 9 | 1 | 6 | 6.616244 | 0.366538 | -0.033462 | 0.026154 | 0.143627 |
| 10 | 1.00 | 7 | 5 | 30 | 6.601146 | 0.414901 | 0.014901 | 0.186973 | 0.002282 |
| 11 | 1.00 | 9 | 5 | 30 | 5.603612 | 0.439035 | 0.039035 | 0.077433 | 0.151115 |
| 12 | 1.20 | 9 | 3 | 18 | 6.730707 | 0.413782 | 0.013782 | 0.090416 | 0.201137 |
| 13 | 1.40 | 10 | 6 | 36 | 9.617040 | 0.384827 | -0.015173 | 0.127827 | 0.428831 |
| 14 | 1.60 | 11 | 8 | 48 | 14.487060 | 0.373850 | -0.026150 | 0.106363 | 0.506395 |
| 15 | 1.80 | 13 | 11 | 66 | 20.321962 | 0.381333 | -0.018667 | 0.101454 | 0.402766 |
| 16 | 2.00 | 14 | 18 | 108 | 25.190585 | 0.401062 | 0.001062 | 0.108698 | 0.239574 |
| 17 | 2.20 | 14 | 20 | 120 | 28.173107 | 0.415297 | 0.015297 | 0.090434 | 0.118398 |
| 18 | 2.20 | 13 | 20 | 120 | 24.480578 | 0.420271 | 0.020271 | 0.063285 | 0.131066 |
| 19 | 2.20 | 13 | 14 | 84 | 23.068583 | 0.406798 | 0.006798 | 0.059654 | 0.057678 |
| 20 | 2.20 | 13 | 13 | 78 | 24.626774 | 0.391311 | -0.008689 | 0.063602 | 0.067546 |
| 21 | 2.20 | 14 | 18 | 108 | 26.808986 | 0.383919 | -0.016081 | 0.054630 | 0.088611 |
| 22 | 2.20 | 14 | 14 | 84 | 27.652560 | 0.390702 | -0.009298 | 0.057997 | 0.031466 |
| 23 | 2.20 | 14 | 10 | 60 | 25.771090 | 0.411135 | 0.011135 | 0.085963 | 0.068040 |
| 24 | 2.20 | 15 | 8 | 48 | 22.087414 | 0.423670 | 0.023670 | 0.070530 | 0.142938 |
| 25 | 2.20 | 15 | 9 | 54 | 20.070541 | 0.412469 | 0.012469 | 0.056492 | 0.091313 |
| 26 | 2.20 | 15 | 12 | 72 | 20.869007 | 0.394319 | -0.005681 | 0.065448 | 0.039783 |
| 27 | 2.20 | 17 | 15 | 90 | 22.661703 | 0.383431 | -0.016569 | 0.065631 | 0.085902 |
| 28 | 2.20 | 19 | 10 | 60 | 23.367333 | 0.384890 | -0.015110 | 0.077441 | 0.031138 |
| 29 | 2.40 | 27 | 19 | 114 | 27.453624 | 0.392247 | -0.007753 | 0.101645 | 0.174872 |
| 30 | 2.60 | 40 | 20 | 120 | 29.119175 | 0.406238 | 0.006238 | 0.124342 | 0.060668 |
| 31 | 2.60 | 46 | 20 | 120 | 23.013832 | 0.429496 | 0.029496 | 0.141693 | 0.209667 |
| 32 | 2.60 | 51 | 20 | 120 | 18.046955 | 0.427248 | 0.027248 | 0.089655 | 0.215821 |
| 33 | 2.60 | 48 | 20 | 120 | 16.857512 | 0.410911 | 0.010911 | 0.080322 | 0.065908 |
| 34 | 2.60 | 57 | 20 | 120 | 17.044426 | 0.396793 | -0.003207 | 0.078206 | 0.011088 |
| 35 | 2.60 | 48 | 20 | 120 | 17.066677 | 0.389676 | -0.010324 | 0.080037 | 0.001305 |
| 36 | 2.60 | 61 | 20 | 120 | 16.407070 | 0.389802 | -0.010198 | 0.089046 | 0.038649 |
| 37 | 2.60 | 88 | 13 | 78 | 15.193395 | 0.395112 | -0.004888 | 0.098123 | 0.073973 |
| 38 | 2.60 | 95 | 12 | 72 | 13.868607 | 0.399330 | -0.000670 | 0.094642 | 0.087195 |
| 39 | 2.60 | 88 | 10 | 60 | 12.841521 | 0.400035 | 0.000035 | 0.082890 | 0.074058 |
| 40 | 2.60 | 77 | 10 | 60 | 12.147982 | 0.399344 | -0.000656 | 0.073451 | 0.054008 |
| 41 | 2.60 | 56 | 11 | 66 | 11.631965 | 0.397810 | -0.002190 | 0.064483 | 0.042478 |
| 42 | 2.60 | 59 | 12 | 72 | 11.234731 | 0.394714 | -0.005286 | 0.057354 | 0.034150 |
| 43 | 2.60 | 63 | 13 | 78 | 10.960622 | 0.392142 | -0.007858 | 0.052058 | 0.024398 |
| 44 | 2.60 | 58 | 10 | 60 | 10.744694 | 0.389367 | -0.010633 | 0.049048 | 0.019700 |
| 45 | 2.60 | 43 | 6 | 36 | 10.587484 | 0.387549 | -0.012451 | 0.042341 | 0.014631 |
| 46 | 2.60 | 44 | 8 | 48 | 10.454261 | 0.385401 | -0.014599 | 0.042762 | 0.012583 |
| 47 | 2.80 | 47 | 6 | 36 | 10.461933 | 0.385901 | -0.014099 | 0.045118 | 0.000734 |
| 48 | 3.00 | 64 | 9 | 54 | 10.364817 | 0.385619 | -0.014381 | 0.047083 | 0.009283 |
| 49 | 3.20 | 50 | 6 | 36 | 10.244534 | 0.384347 | -0.015653 | 0.041689 | 0.011605 |
| 50 | 3.40 | 38 | 8 | 48 | 10.179006 | 0.382463 | -0.017537 | 0.040404 | 0.006396 |
| 51 | 3.60 | 57 | 8 | 48 | 10.123245 | 0.380492 | -0.019508 | 0.037644 | 0.005478 |
| 52 | 3.80 | 56 | 6 | 36 | 10.080452 | 0.378738 | -0.021262 | 0.033353 | 0.004227 |
| 53 | 4.00 | 44 | 6 | 36 | 10.051396 | 0.377144 | -0.022856 | 0.030851 | 0.002882 |
| 54 | 4.20 | 43 | 5 | 30 | 10.024040 | 0.376021 | -0.023979 | 0.027747 | 0.002722 |
| 55 | 4.40 | 42 | 4 | 24 | 10.001256 | 0.375189 | -0.024811 | 0.023411 | 0.002273 |
| 56 | 4.60 | 43 | 4 | 24 | 9.984857 | 0.374590 | -0.025410 | 0.020489 | 0.001640 |
| 57 | 4.80 | 47 | 3 | 18 | 9.972323 | 0.374149 | -0.025851 | 0.018943 | 0.001255 |
| 58 | 5.00 | 41 | 3 | 18 | 9.964404 | 0.373960 | -0.026040 | 0.017687 | 0.000794 |
| 59 | 5.20 | 44 | 2 | 12 | 9.952028 | 0.373922 | -0.026078 | 0.011516 | 0.001242 |
| 60 | 5.40 | 39 | 2 | 12 | 9.948679 | 0.373981 | -0.026019 | 0.010543 | 0.000336 |
| 61 | 5.60 | 39 | 1 | 6 | 9.943750 | 0.374132 | -0.025868 | 0.013316 | 0.000495 |
| 62 | 5.80 | 36 | 2 | 12 | 9.938642 | 0.374275 | -0.025725 | 0.012931 | 0.000514 |
| 63 | 6.00 | 37 | 1 | 6 | 9.925764 | 0.374607 | -0.025393 | 0.013346 | 0.001296 |
| 64 | 6.20 | 35 | 2 | 12 | 9.910433 | 0.374866 | -0.025134 | 0.012630 | 0.001545 |
| 65 | 6.40 | 34 | 0 | 0 | 9.893304 | 0.374866 | -0.025134 | 0.000000 | 0.001728 |
| 66 | 6.60 | 21 | 0 | 0 | 9.907413 | 0.374866 | -0.025134 | 0.000000 | 0.001426 |

## Artifacts

- JSON result: `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/parallel_full_run.json`
- Final state: `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/parallel_full_state.npz`
- Outer-history CSV: `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/parallel_full_outer_history.csv`
- Final-state figure: `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/final_state.png`
- Convergence figure: `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/convergence_history.png`
- Density-step figure: `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/density_step_history.png`

- Density-evolution GIF: `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/density_evolution.gif`
## Reproduction

```bash
mpiexec -n 32 ./.venv/bin/python src/problems/topology/jax/solve_topopt_parallel.py \
    --nx 768 --ny 384 --length 2.0 --height 1.0 \
    --traction 1.0 --load_fraction 0.2 \
    --fixed_pad_cells 32 --load_pad_cells 32 \
    --volume_fraction_target 0.4 --theta_min 1e-06 \
    --solid_latent 10.0 --young 1.0 --poisson 0.3 \
    --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 \
    --beta_lambda 12.0 --volume_penalty 10.0 \
    --p_start 1.0 --p_max 10.0 --p_increment 0.2 \
    --continuation_interval 1 --outer_maxit 2000 \
    --outer_tol 0.02 --volume_tol 0.001 \
    --design_maxit 20 --tolf 1e-06 --tolg 0.001 \
    --linesearch_tol 0.1 --mechanics_ksp_rtol 0.0001 \
    --linesearch_relative_to_bound \
    --design_gd_line_search golden_adaptive \
    --design_gd_adaptive_window_scale 2.0 \
    --mechanics_ksp_max_it 100 --quiet --print_outer_iterations \
    --save_outer_state_history --outer_snapshot_stride 2 \
    --outer_snapshot_dir artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/frames \
    --json_out artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/parallel_full_run.json --state_out artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/parallel_full_state.npz
```
