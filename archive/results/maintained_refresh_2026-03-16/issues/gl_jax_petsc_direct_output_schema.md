# GL JAX+PETSc Direct Output Schema

- status: `resolved`
- family: `ginzburg_landau`
- affected area: `experiments/runners/run_replications.py`

## Symptom

The first maintained example run for the direct JAX+PETSc Ginzburg-Landau solver stopped with:

`KeyError: 'result'`

The solver itself completed successfully. The failure was in the replication runner's summary extraction.

## Smallest reproducer

```bash
./.venv/bin/python experiments/runners/run_replications.py \
  --out-dir replications/2026-03-16_maintained_refresh \
  --only examples \
  --resume
```

## Cause

The replication runner initially assumed every direct JSON payload already used the normalized
`{"case": ..., "result": ...}` schema. The maintained GL JAX+PETSc direct CLI emits the result payload directly.

## Repair

- taught the replication runner to normalize direct scalar payloads before summary extraction
- reran the maintained examples and the direct speed rows that use the same solver

## Validation

- `tests/test_replication_runner.py`
- `replications/2026-03-16_maintained_refresh/runs/examples/gl_jax_petsc_element/`
- `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/`
