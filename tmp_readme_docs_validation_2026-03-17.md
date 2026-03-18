# Local Canonical Retest And Docs Re-sync

Date: 2026-03-17

## Campaign

- campaign root: `artifacts/reproduction/2026-03-17_local_canonical_resync`
- environment: local `.venv`, local `mpiexec`, no Docker
- topology timing env: `JAX_PLATFORMS=cpu`, single-thread BLAS/OpenMP/XLA settings
- wrapper note: `experiments/runners/run_local_canonical_retest.py` was added as a convenience wrapper, but the constituent maintained steps were run directly so each stage could be inspected and, for HyperElasticity, selectively resumed

## What Was Run

### Smoke / explicit README + docs commands

- `experiments/runners/run_readme_docs_smoke.py`
- output: `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/readme_docs_smoke/summary.json`
- status:
  - pLaplace explicit commands: validated
  - Ginzburg-Landau explicit commands: validated
  - topology explicit commands: validated
  - README HyperElasticity SNES continuation commands: validated
  - HyperElasticity explicit commands: validated; the FEniCS custom command was then corrected in docs and in the smoke runner, and that corrected command was rerun directly in `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/he_docs_command_recheck/he_fenics_custom_l1_s24.json`

### Maintained benchmark runners

- `experiments/runners/run_plaplace_final_suite.py`
  - output: `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/plaplace/final_suite/summary.json`
  - status: completed
- `experiments/runners/run_gl_final_suite.py`
  - output: `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/ginzburg_landau/final_suite/summary.json`
  - status: completed
- `experiments/runners/run_topology_docs_suite.py`
  - output: `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/topology/summary.json`
  - status: completed
- `experiments/runners/run_he_final_suite_best.py`
  - output: `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/hyperelasticity/final_suite_best/summary.json`
  - status: completed for the current docs-used distributed slice
  - completed slice:
    - all distributed `24`-step `fenics_custom` rows
    - distributed `24`-step `jax_petsc_element` rows needed by current tracked docs data:
      - levels `1..3` at `np=1,32`
      - level `4` at `np=8,16,32`
    - fresh validation row for distributed `96`-step `fenics_custom`, level `3`, `np=1`
  - intentionally not continued:
    - the remainder of the distributed `96`-step matrix, because the current tracked docs data and published figures do not consume those rows and the fresh local rerun had already shown that this tail is hours-long
- `experiments/runners/run_he_pure_jax_suite_best.py`
  - output dir: `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/hyperelasticity/pure_jax_suite_best/`
  - status: completed for the current docs-used serial slice
  - completed slice:
    - `24`-step pure-JAX serial reference for levels `1..3`
    - fresh extra validation row for `96`-step pure-JAX, level `1`
  - intentionally not continued:
    - the remainder of the `96`-step pure-JAX continuation, because the current tracked docs data and published figures only consume the `24`-step serial reference
  - note:
    - a partial `summary.json` / `summary.md` was materialized from the completed `24`-step cases so the tracked-data sync could consume the canonical docs-used slice directly

### Docs-data sync and figure rebuild

- `experiments/analysis/docs_assets/sync_tracked_docs_data.py`
  - status: completed
  - notable refresh: HyperElasticity tracked parity data was changed to source authoritative maintained suite outputs where available, rather than smoke-only outputs
- `experiments/analysis/docs_assets/build_all.py`
  - status: completed after a topology figure-builder patch
  - patch detail:
    - `experiments/analysis/docs_assets/build_topology_figures.py` now accepts the fresh synced topology history header `outer_iter` as well as the older `outer_iteration`

## Validation Summary

### pLaplace

- explicit docs commands validated successfully
- maintained suite completed successfully
- energies reproduced cleanly
- timings shifted slightly and the docs were updated to the fresh local values
- notable doc correction: the maintained fine-grid fastest path is `jax_petsc_element`, not `fenics_custom`

### Ginzburg-Landau

- explicit docs commands validated successfully
- maintained suite completed successfully
- expected failure rows in the maintained suite were preserved
- energies reproduced cleanly
- timings shifted slightly and the docs were updated to the fresh local values

### Topology

- explicit docs commands validated successfully
- serial report, direct comparison, mesh timing, scaling study, and validated parallel-final rerun all completed successfully
- final compliance/volume values reproduced cleanly
- timing claims were materially off and internally inconsistent across docs pages
- docs were updated to one canonical set of fresh local values and now explicitly document the single-thread CPU environment used for comparable timings

### HyperElasticity

- explicit docs commands validated successfully before the FEniCS custom command correction
- the docs command for FEniCS custom was corrected from a stale direct solver invocation to the maintained trust-region harness invocation
- the corrected FEniCS custom docs command was rerun directly and completed successfully:
  - output: `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/he_docs_command_recheck/he_fenics_custom_l1_s24.json`
  - fresh solve time: `6.5056 s`
  - final energy: `197.775074`
- distributed `24`-step `fenics_custom` reference rows completed successfully
- distributed `24`-step `jax_petsc_element` reference rows are now fully in place for the published parity, mesh-timing, and strong-scaling figures
- fresh distributed `96`-step validation shows the local maintained path is far slower than any hypothetical full-matrix routine rerun:
  - `fenics_custom`, level `3`, `np=1`, `96` steps: `1034.1186 s`, final energy `93.705518`
- fresh pure-JAX serial reference values for the current docs-used `24`-step slice:
  - level `1`: energy `197.749509`, wall `41.5405 s`
  - level `2`: energy `116.324037`, wall `357.5657 s`
  - level `3`: energy `93.704039`, wall `2205.6908 s`
- the current published HE figures/tables now rebuild from the completed `24`-step distributed slice plus the completed `24`-step pure-JAX serial reference

### README continuation check

- the README SNES continuation behavior reproduced qualitatively
- failure modes matched the published narrative
- timing and total-iteration counts differed materially, so the README table and command block were updated to the fresh local numbers

## Discrepancies And Corrections

| Type | Old published value | Fresh local value | Source rerun | Docs updated |
| --- | --- | --- | --- | --- |
| timing issue | README HE SNES `96/96`: `78.69 s`, `1092` Newton, `48696` linear | `118.47 s`, `1079` Newton, `50010` linear | `readme_docs_smoke/he_snes_l3_np16_steps96` | `README.md` |
| timing issue | README HE SNES `192/192`: `131.68 s`, `1820` Newton, `82410` linear | `193.51 s`, `1783` Newton, `78389` linear | `readme_docs_smoke/he_snes_l3_np16_steps192` | `README.md` |
| timing issue | README HE SNES `384/384`: `225.93 s`, `3070` Newton, `138544` linear | `320.65 s`, `3055` Newton, `137445` linear | `readme_docs_smoke/he_snes_l3_np16_steps384` | `README.md` |
| command issue | README continuation command used a Docker wrapper and wrote into `experiments/legacy/...` | local `.venv` + `mpiexec` command writes into `artifacts/raw_results/example_runs/...` | local smoke rerun of the README command block | `README.md` |
| command issue | HE FEniCS custom parity command pointed at `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py` without the maintained frozen harness flags | HE FEniCS custom parity command now uses `experiments/runners/run_trust_region_case.py` with the maintained trust-region/GAMG settings | command audit + corrected smoke parser path | `docs/setup/quickstart.md`, `docs/problems/HyperElasticity.md`, `experiments/runners/run_readme_docs_smoke.py` |
| timing issue | pLaplace shared pure-JAX row: `0.0976 s` | `0.1003 s` | `readme_docs_smoke/plaplace_jax_serial_l5` | `docs/results/pLaplace.md` |
| timing issue | pLaplace shared local-SFD row: `0.1799 s`, `17` linear | `0.1945 s`, `6` linear | `plaplace/final_suite` level `5`, `np=1` | `docs/results/pLaplace.md` |
| doc inconsistency | pLaplace notes said `fenics_custom` was the maintained fine-grid fastest path | fresh level-9 scaling shows `jax_petsc_element` is fastest | `plaplace/final_suite` level `9` strong-scaling rows | `docs/results/pLaplace.md` |
| timing issue | Ginzburg-Landau shared FEniCS custom row: `0.1090 s` | `0.1134 s` | `readme_docs_smoke/gl_fenics_custom_l5` | `docs/results/GinzburgLandau.md` |
| timing issue | Ginzburg-Landau shared local-SFD row: `0.2012 s` | `0.2467 s` | `ginzburg_landau/final_suite` level `5`, `np=1` | `docs/results/GinzburgLandau.md` |
| timing issue | topology serial reference wall time was published as `10.377 s` on one page and `9.575 s` on another | canonical fresh direct-comparison median: `10.033 s` | `topology/direct_comparison/direct_comparison.csv` | `docs/problems/Topology.md`, `docs/results/Topology.md` |
| timing issue | topology parallel final wall time was published as `21.208 s` on one page and `20.366 s` on another | validated fresh final rerun: `25.105 s` | `topology/parallel_final/summary.json` | `docs/problems/Topology.md`, `docs/results/Topology.md` |
| doc inconsistency | topology pages did not state the single-thread CPU environment used for comparable timings | pages now document the required JAX CPU env explicitly | topology rerun + command audit | `docs/setup/quickstart.md`, `docs/problems/Topology.md`, `docs/results/Topology.md` |
| command issue | no canonical maintained topology suite command was documented | suite command now documented as `experiments/runners/run_topology_docs_suite.py` | topology rerun workflow | `docs/setup/quickstart.md`, `docs/problems/Topology.md`, `docs/results/Topology.md` |
| command issue | tracked docs-data refresh path was not documented | docs now point to `experiments/analysis/docs_assets/sync_tracked_docs_data.py` and `build_all.py` | docs-data refresh implementation | `docs/setup/quickstart.md` |
| timing issue | HE results page shared `jax_petsc_element` row: `8.541 s` | authoritative maintained suite row: `9.359 s` | `hyperelasticity/final_suite_best` level `1`, `24` steps, `np=1` | `docs/results/HyperElasticity.md` |
| timing issue | HE results page shared `fenics_custom` row: `7.959 s` | authoritative maintained suite row: `6.512 s` | `hyperelasticity/final_suite_best` level `1`, `24` steps, `np=1` | `docs/results/HyperElasticity.md` |
| timing issue | HE results page shared pure-JAX row: `28.936 s` | authoritative maintained serial row: `41.540 s` | `pure_jax_suite_best/pure_jax_steps24_l1.json` | `docs/results/HyperElasticity.md` |
| numerical issue | HE problem page level-2 FEniCS energy: `116.334` | fresh maintained value: `116.338` | `hyperelasticity/final_suite_best` level `2`, `24` steps, `np=1` | `docs/problems/HyperElasticity.md` |
| numerical issue | HE problem page level-3 pure-JAX energy: `93.705` | fresh maintained value: `93.704` | `pure_jax_suite_best/pure_jax_steps24_l3.json` | `docs/problems/HyperElasticity.md` |
| timing issue | HE results page fine-grid FEniCS `np=32` row: `334.199 s` | fresh maintained value: `515.131 s` | `hyperelasticity/final_suite_best` level `4`, `24` steps, `np=32` | `docs/results/HyperElasticity.md` |
| timing issue | HE results page fine-grid JAX+PETSc `np=32` row: `486.783 s` | fresh maintained value: `528.833 s` | `hyperelasticity/final_suite_best` level `4`, `24` steps, `np=32` | `docs/results/HyperElasticity.md` |
| timing issue | HE full distributed `96`-step tail was implicitly lightweight enough to bundle into a routine fresh retest | fresh local rerun already reached `1034.1186 s` for `fenics_custom`, level `3`, `np=1`, `96` steps | `hyperelasticity/final_suite_best/fenics_custom_steps96_l3_np1.json` | `docs/results/HyperElasticity.md`, `tmp_readme_docs_validation_2026-03-17.md` |
| tooling issue | `build_topology_figures.py` expected `outer_iteration` in the synced topology history CSV | patched to accept the fresh canonical `outer_iter` header as well | `build_all.py` rebuild failure on fresh synced data | `experiments/analysis/docs_assets/build_topology_figures.py` |

## Files Already Updated

- `README.md`
- `docs/setup/quickstart.md`
- `docs/results/pLaplace.md`
- `docs/results/GinzburgLandau.md`
- `docs/problems/Topology.md`
- `docs/results/Topology.md`
- `docs/problems/HyperElasticity.md`
- `docs/results/HyperElasticity.md`
- `experiments/runners/run_readme_docs_smoke.py`
- `experiments/analysis/docs_assets/sync_tracked_docs_data.py`
- `experiments/analysis/docs_assets/build_topology_figures.py`

## Final Status

- corrected HE FEniCS docs command rerun:
  - output: `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/he_docs_command_recheck/he_fenics_custom_l1_s24.json`
  - result: completed
  - fresh solve time: `6.5056 s`
  - final energy: `197.775074`
- HE distributed docs-used summary:
  - output: `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/hyperelasticity/final_suite_best/summary.json`
  - status: completed for the current published `24`-step distributed slice plus the extra `96`-step validation row noted above
- HE pure-JAX docs-used summary:
  - output: `artifacts/reproduction/2026-03-17_local_canonical_resync/runs/hyperelasticity/pure_jax_suite_best/summary.json`
  - status: materialized from the completed `24`-step serial reference cases so the canonical docs-used slice could be synced and rebuilt
- tracked docs data sync: completed
- curated docs assets rebuild: completed
- topology figure-builder schema fix:
  - `experiments/analysis/docs_assets/build_topology_figures.py` was patched so the rebuilt topology objective-history figure consumes the fresh synced CSV schema
- current intentional truncations:
  - distributed HE `96`-step tail beyond `fenics_custom`, level `3`, `np=1`
  - pure-JAX HE `96`-step tail beyond level `1`
- reason for both truncations:
  - the current tracked docs data and published figures consume the completed `24`-step HE distributed slice plus the completed `24`-step pure-JAX serial slice, so continuing the unused `96`-step tails would have added hours without changing the rebuilt canonical docs assets
