# HE JAX+PETSc Trust-Region CLI Flags

- status: `resolved`
- family: `hyperelasticity`
- affected area: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`

## Symptom

The maintained direct JAX+PETSc hyperelasticity example rejected the trust-region options used by the
maintained workflows and exited with argument-parsing failure.

## Smallest reproducer

```bash
./.venv/bin/python src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py \
  --level 1 \
  --steps 24 \
  --total_steps 24 \
  --profile performance \
  --ksp_type stcg \
  --pc_type gamg \
  --use_trust_region \
  --trust_subproblem_line_search \
  --trust_radius_init 0.5 \
  --trust_shrink 0.5 \
  --trust_expand 1.5 \
  --trust_eta_shrink 0.05 \
  --trust_eta_expand 0.75 \
  --out /tmp/he_jax_petsc.json
```

## Cause

The maintained direct CLI exposed the trust-subproblem line-search switch but was missing the rest of the
trust-region argument family used by the maintained HE settings.

## Repair

- added the missing trust-region parser options to the canonical HE JAX+PETSc CLI
- added regression coverage for the maintained argument set
- reran the maintained example and direct speed rows

## Validation

- `tests/test_he_jax_petsc_cli.py`
- `replications/2026-03-16_maintained_refresh/runs/examples/he_jax_petsc_element/`
- `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/`
