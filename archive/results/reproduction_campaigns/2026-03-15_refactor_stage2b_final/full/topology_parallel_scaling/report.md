# Parallel Scaling Report

This directory stores the validated fine-grid topology scaling artifacts used by
the canonical benchmark docs.

## Status

- scaling figures and CSV were promoted into this final campaign from the
  validated curated outputs in `docs/assets/jax_topology_parallel/scaling/`
- the topology scaling solver/report path itself was not changed in this
  cleanup closeout
- a direct rerun was started during cleanup, then stopped once it was clear the
  run would be redundant for an unchanged solver path

## Files

- [`scaling_summary.csv`](scaling_summary.csv)
- [`wall_scaling.png`](wall_scaling.png)
- [`phase_scaling.png`](phase_scaling.png)
- [`efficiency.png`](efficiency.png)
- [`quality_vs_ranks.png`](quality_vs_ranks.png)

## Reproduction

```bash
./.venv/bin/python experiments/analysis/generate_parallel_scaling_stallstop_report.py \
  --asset-dir artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_scaling \
  --report-path artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_scaling/report.md
```
