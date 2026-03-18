# Cleanup Closeout Comparison

This file summarizes the final cleanup campaign rooted at
`artifacts/reproduction/2026-03-15_refactor_stage2b_final/`.

## Representative Validation

- pre-finish reference:
  `artifacts/reproduction/2026-03-15_refactor_stage2b_final/checkpoints/pre_finish_reference/summary.json`
- post-cleanup reference:
  `artifacts/reproduction/2026-03-15_refactor_stage2b_final/checkpoints/representative_matrix_post_cleanup/summary.json`
- result:
  representative coverage is green after the cleanup pass, including the
  previously failing HE FEniCS direct-CLI `24/24` case

## Full Campaign Status

- pLaplace final suite:
  `90 / 90` rows completed
- GL final suite:
  `90` rows total, `88` completed, `2` expected benchmark-documented failures
- HE final suite best:
  `84 / 84` rows completed
- HE pure-JAX suite best:
  `6 / 6` rows completed
- topology serial reference:
  canonical rerun completed and benchmark doc refreshed
- topology parallel fine benchmark:
  canonical rerun completed and curated assets refreshed
- topology parallel scaling:
  validated scaling CSV/figures promoted into the final campaign from curated
  benchmark assets because the scaling solver path was unchanged in this
  cleanup closeout

## Canonical Drift Updates

- `docs/benchmarks/jax_topology_optimisation_benchmark.md` was refreshed to the
  canonical rerun because the older serial benchmark text was stale
- `docs/benchmarks/final_pLaplace_results.md` and
  `docs/benchmarks/final_GL_results.md` were refreshed to the validated final
  reruns because their fine-mesh headline timings had drifted
- `docs/benchmarks/final_HE_results.md` kept the same benchmark content, but its
  canonical data and figure-generator paths were updated to the final campaign

## README Example Outputs

The maintained README JSON outputs now exist under:

- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/he_examples/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/readme_examples/`

## Validation

- `python -m compileall -q src tests experiments/runners experiments/analysis`
- `./.venv/bin/pytest -q tests/test_benchmark_results_helpers.py tests/test_runner_summary_contracts.py tests/test_topology_report_generators.py tests/test_import_hygiene.py tests/test_load_step_driver.py tests/test_reordered_element_base.py tests/test_wrapper_parity.py tests/test_he_fenics_direct_cli.py tests/test_shared_helpers.py tests/test_topology_parallel_topopt_smoke.py tests/test_final_report_figure_generators.py tests/test_current_doc_paths.py`

Both validation steps passed on the closeout state.
