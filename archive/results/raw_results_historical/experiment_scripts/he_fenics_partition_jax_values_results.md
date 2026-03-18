# HyperElasticity3D: FEniCS-Partitioned Matrix + JAX Values (np=16)

Date: 2026-02-28

## Commands run

```bash
# Main experiment (FEniCS matrix layout, JAX-computed values)
docker exec bench_container bash -lc "cd /workdir && \
  mpirun -np 16 python3 experiment_scripts/bench_he_fenics_partition_jax_values.py \
    --level 3 --step 1 --total_steps 24 --quiet \
    --out tmp_work/he_fenics_partition_jax_values_l3_np16_run2.json"

# Control experiment (same values, same layout: FEniCS matrix vs direct copy)
docker exec bench_container bash -lc "cd /workdir && \
  mpirun -np 16 python3 /tmp/he_ctrl_same_values_tojson.py"
```

## Experiment A: FEniCS layout, JAX values mapped into it

Source JSON: `tmp_work/he_fenics_partition_jax_values_l3_np16_run2.json`

### Matrix layout / ownership

- Total DOFs: `78003`
- Free DOFs: `77517`
- Global NNZ (FEniCS matrix): `3100473`
- Global NNZ (JAX values in FEniCS matrix): `3100473`

FEniCS row ownership ranges:

```text
[0, 4764], [4764, 9507], [9507, 14394], [14394, 19344],
[19344, 24105], [24105, 29031], [29031, 34023], [34023, 38937],
[38937, 43824], [43824, 48699], [48699, 53646], [53646, 58611],
[58611, 63363], [63363, 68187], [68187, 73068], [73068, 78003]
```

JAX custom-partition ownership ranges (for reference):

```text
[0, 4845], [4845, 9690], [9690, 14535], [14535, 19380],
[19380, 24225], [24225, 29070], [29070, 33915], [33915, 38760],
[38760, 43605], [43605, 48450], [48450, 53295], [53295, 58140],
[58140, 62985], [62985, 67830], [67830, 72675], [72675, 77517]
```

### KSP solve timings (GMRES+GAMG, rtol=1e-1, max_it=30)

- FEniCS matrix:
  - `ksp_its = 1`
  - `solve_time = 0.003520325 s`
  - `pc_setup_time = 0.051652770 s`
- JAX values in FEniCS matrix:
  - `ksp_its = 4`
  - `solve_time = 0.006525067 s`
  - `pc_setup_time = 0.056492914 s`

### Matrix difference

- `||A_fenics - A_jax_mapped||_F (free-free) = 4.486492398088106e+08`
- `max |A_fenics - A_jax_mapped| (free-free) = 1.2464058348025832e+07`

### Per-rank Hessian value compute time (JAX HVP only, seconds)

| Rank | Time (s) |
|---:|---:|
| 0 | 0.287502625 |
| 1 | 0.142633791 |
| 2 | 0.458094558 |
| 3 | 0.109057723 |
| 4 | 0.461781401 |
| 5 | 0.161493307 |
| 6 | 0.411598875 |
| 7 | 0.437503479 |
| 8 | 0.260527229 |
| 9 | 0.204617477 |
| 10 | 0.446385652 |
| 11 | 0.459080028 |
| 12 | 0.332705662 |
| 13 | 0.458339856 |
| 14 | 0.343311384 |
| 15 | 0.085222216 |

Summary:

- `min = 0.085222216 s`
- `max = 0.461781401 s`
- `mean = 0.316240954 s`
- `std = 0.134724111 s`
- Imbalance ratio `max/min = 5.42x`

## Experiment B (control): identical values + identical layout

Source JSON: `tmp_work/he_fenics_same_values_control_l3_np16.json`

This checks the hypothesis: if matrix partition/layout and values are both the same, solve time should match.

- Base FEniCS matrix solve:
  - `ksp_its = 2`
  - `solve_time = 0.004137944 s`
  - `pc_setup_time = 0.047178832 s`
- Direct copy of the same matrix values/layout:
  - `ksp_its = 2`
  - `solve_time = 0.004382260 s`
  - `pc_setup_time = 0.048293508 s`

Observed result: solve behavior is effectively the same (same iteration count, very close timings).

## Conclusions

- Using FEniCS matrix partition/layout alone does not make JAX matrix solves match FEniCS solves.
- The mapped JAX matrix still solves slower (`4` vs `1` KSP iterations in this run), consistent with value differences.
- Per-rank Hessian value computation shows strong imbalance at `np=16` (`5.42x` max/min), which can contribute to assembly barriers.
- Control case confirms the expected baseline: same layout + same values gives matching KSP behavior.
