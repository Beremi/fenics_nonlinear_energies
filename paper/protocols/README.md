# Publication Revision Experiment Protocols

These cards freeze the scientific question, comparison, gate, and evidence
status before a publication-grade campaign is run. The authoritative result
shape is documented in `docs/reference/publication_run_records.md`. A result
from a dirty worktree is a pilot even when every numerical check passes.

| Experiment | Scope decision | Protocol | Evidence status |
| --- | --- | --- | --- |
| EXP-DERIV-001 | central | [smooth and fixed-branch derivative correctness](EXP-DERIV-001.md) | enhanced Plasticity3D fixed-element pilot passed; smooth FE/distributed matrix gates remain |
| EXP-MC-001 | supplementary diagnostic | [Plasticity3D material-point branch verification](EXP-MC-001.md) | complete dirty-worktree CPU pilot covers five interiors, five two-sided interfaces, rotations, and repeated spectra; clean rerun remains |
| EXP-STOP-001 | central prerequisite | [stopping and inexact-solve calibration](EXP-STOP-001.md) | narrow P1 route-sensitivity diagnostic complete; problem-specific Riesz calibration and converged tolerance matrix remain |
| EXP-DIST-001 | central prerequisite | [distributed equivalence](EXP-DIST-001.md) | controlled one-/two-rank hyperelastic fixed-state pilot passes; four-rank, one-factor-at-a-time, clean-rerun, and nonlinear endpoint gates remain |
| EXP-VAL-001 | central independent checks; optional comparator | [independent verification and optional matched comparator](EXP-VAL-001.md) | independent $p$-Laplace and branch-controlled Ginzburg--Landau rates plus analytic and nonaffine hyperelastic checks pass as diagnostics; DOLFINx is outside required scope and its optional ABI repair remains unauthorized |
| EXP-P3D-ROUTE-001 | merged into `EXP-ROUTE-001` Tier B | [route study](EXP-ROUTE-001.md) | exact two-route high-order and low-order confirmation matrix prepared; no publication-grade repetitions |
| EXP-ROUTE-001 | central | [second-architecture fixed-state and full-solve route study](EXP-ROUTE-001.md) | paired balanced all-route blocks, replicated descriptive factor diagnostics, four-probe/gradient/direct checks, and strict high/low-order endpoint analysis are implemented; local legacy rows remain diagnostic and Karolina is unsubmitted |
| EXP-GLOB-001 | supplementary unless retained by claim audit | [controlled globalization evidence](EXP-GLOB-001.md) | common-start GL/first-load HE smoke prepared but unrun; algorithm freeze, multiple instances, and repetitions remain |
| EXP-DISC-001 | conditional | [separated discretization study](EXP-DISC-001.md) | named-rule implementation, fixed-state pilot, failure propagation, and strict adjudicator complete; Karolina high-order matrix prepared but not submitted |
| EXP-TOPO-001 | supplementary software demonstration | [scope and corrected-unit diagnostic](EXP-TOPO-001.md) | corrected 1/2-rank fixed-work pilot complete; no KKT-quality result and no optimization-solution claim allowed |
| EXP-SCALE-001 | central distributed-viability evidence | [fixed-policy Karolina viability](EXP-SCALE-001.md) | required/optional matrices, frozen analysis contract, strict analyzer, and post-job accounting collector are prepared and tested; not submitted; allocation revalidation required |

The table is an execution index, not an evidence table. Numerical claims are
adjudicated separately after every retained card has a terminal outcome.
