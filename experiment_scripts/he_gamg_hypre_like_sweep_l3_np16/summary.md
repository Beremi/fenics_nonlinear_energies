# GAMG sweep for HYPRE-like iteration profile (L3, np=16)

## Baseline (HYPRE reference)

- Source: `experiment_scripts/he_custom_l3_np16_bench.json`
- Steps: 24 (24 converged)
- Total time: 135.5444 s
- Total Newton iters: 669
- Total KSP iters: 10347
- Avg KSP/Newton: 15.4664

## Sweep results

| Case | rtol | ksp_max_it | PC reuse on cap | RC | Conv steps | Time [s] | Newton | KSP | Avg KSP/Newton | Score to HYPRE | Speedup vs HYPRE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r2e-2_k30_fresh | 2e-02 | 30 | No | 0 | 24/24 | 47.0968 | 720 | 19027 | 26.4264 | 0.9151 | 2.878x |
| r5e-2_k30_reuse | 5e-02 | 30 | Yes | 0 | 24/24 | 51.5617 | 849 | 17447 | 20.5501 | 0.9552 | 2.629x |
| r1e-2_k30_reuse | 1e-02 | 30 | Yes | 0 | 24/24 | 50.4301 | 730 | 20484 | 28.0603 | 1.0709 | 2.688x |
| r1e-2_k30_fresh | 1e-02 | 30 | No | 0 | 24/24 | 50.2838 | 756 | 21168 | 28.0000 | 1.1759 | 2.696x |
| r1e-1_k30_reuse | 1e-01 | 30 | Yes | 0 | 24/24 | 58.5355 | 1044 | 16725 | 16.0201 | 1.1769 | 2.316x |
| r2e-2_k30_reuse | 2e-02 | 30 | Yes | 0 | 24/24 | 51.0453 | 779 | 20828 | 26.7368 | 1.1774 | 2.655x |
| r5e-3_k30_reuse | 5e-03 | 30 | Yes | 0 | 24/24 | 52.7780 | 752 | 21809 | 29.0013 | 1.2318 | 2.568x |
| r2e-2_k50_reuse | 2e-02 | 50 | Yes | 0 | 24/24 | 54.3331 | 700 | 27912 | 39.8743 | 1.7439 | 2.495x |
| r1e-2_k50_reuse | 1e-02 | 50 | Yes | 0 | 24/24 | 55.0011 | 692 | 29905 | 43.2153 | 1.9246 | 2.464x |
| r5e-3_k50_reuse | 5e-03 | 50 | Yes | 0 | 24/24 | 58.4888 | 675 | 31223 | 46.2563 | 2.0266 | 2.317x |

## Top 3 closest to HYPRE iteration totals

| Rank | Case | Newton | KSP | Time [s] | Score |
|---:|---|---:|---:|---:|---:|
| 1 | r2e-2_k30_fresh | 720 | 19027 | 47.0968 | 0.9151 |
| 2 | r5e-2_k30_reuse | 849 | 17447 | 51.5617 | 0.9552 |
| 3 | r1e-2_k30_reuse | 730 | 20484 | 50.4301 | 1.0709 |

Summary JSON: `/home/beremi/repos/nonlinear_energies_all/fenics_nonlinear_energies/experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/summary.json`
Summary CSV: `/home/beremi/repos/nonlinear_energies_all/fenics_nonlinear_energies/experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/summary.csv`
