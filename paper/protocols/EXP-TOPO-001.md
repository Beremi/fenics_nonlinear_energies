# EXP-TOPO-001: Topology Scope And Corrected-Unit Diagnostic

## Scope Decision

For the selected scientific-computing paper, topology is a supplementary
distributed software demonstration. The manuscript will not claim a new
optimization algorithm, an optimal design, convergence to a fixed problem, or
a KKT point. A future SIOPT route would require the full fixed-problem,
gradient, feasibility, complementarity, baseline, and convergence campaign in
`paper/publication_action_plan.md`; that campaign is outside the selected core
scope.

## Frozen Unit Contract

The rectangular domain has area 2. Define

\[
M_h(\theta)=\sum_e |e|\theta_e,
\qquad
\bar\theta_h=M_h(\theta)/2.
\]

The paper demonstration uses target material measure `0.4`, hence target
normalized fraction `0.2`. Its historical initialization is normalized
fraction `0.4`, hence initial material measure `0.8`. Maintained commands must
spell these as

```text
--target-material-measure 0.4 --initial-normalized-fraction 0.4
```

Serial examples that intentionally target normalized fraction `0.4` must use
`--target-normalized-fraction 0.4`. Result schema version 2 records both units;
historical version-1 parallel fields are interpreted as material measures by
the report generators and are never silently combined with version-2 values.

## Corrected-Unit Smoke

Run the same `64 x 32`, three-outer-iteration fixed schedule at one and two MPI
ranks. The smoke checks:

- exact target/initial unit metadata and initialization parity;
- mechanics completion at every outer step;
- finite compliance and density output;
- rank differences in compliance, material measure, and weighted density;
- isolated campaign identity, UUID, repetition, commands, states, and logs.

This is deliberately `fixed_work_completed`. It is not successful optimization
termination because the schedule stops at SIMP exponent 1.3, design inner
iterations reach their cap, material feasibility is not met, and no projected
KKT or complementarity residual is evaluated.

## Future Upgrade Gate

Topology may return to core optimization evidence only after one fixed
constrained problem has:

1. a chosen physical- or latent-variable KKT system;
2. exact reduced-gradient and volume Taylor tests;
3. relative feasibility, bound violation, projected Lagrangian-gradient,
   complementarity, and mechanics residuals;
4. final continuation parameters reached;
5. an accepted constrained baseline on identical problems;
6. multiple meshes/initializations and rank-consistent KKT-quality endpoints.

Until then, permitted labels are `reduced-design endpoint`, `fixed-work
distributed diagnostic`, and `material-measure/fraction consistency check`.
