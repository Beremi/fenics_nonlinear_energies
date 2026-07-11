# Publication Revision Experiment Protocols

These cards freeze the scientific question, comparison, gate, and evidence
status before a publication-grade campaign is run. The authoritative result
shape is documented in `docs/reference/publication_run_records.md`. A result
from a dirty worktree is a pilot even when every numerical check passes.

| Experiment | Scope decision | Protocol | Evidence status |
| --- | --- | --- | --- |
| EXP-DERIV-001 | central | [smooth and fixed-branch derivative correctness](EXP-DERIV-001.md) | clean smooth and assembled $P_1$/$P_2$/$P_4$ checks pass for all 15 frozen states, including exact CSR structure and route equivalence |
| EXP-MC-001 | supplementary diagnostic | [Plasticity3D material-point branch verification](EXP-MC-001.md) | clean campaign passes five branch interiors, 15 two-sided interface pairs, 15 rotations, and seven repeated-spectrum cases |
| EXP-STOP-001 | central prerequisite | [stopping and inexact-solve calibration](EXP-STOP-001.md) | all 45 local rows completed and were hash-checked; seven cluster-deferred rows are prepared offline but unsubmitted |
| EXP-DIST-001 | central prerequisite | [distributed equivalence](EXP-DIST-001.md) | clean one-/two-/four-rank hyperelastic equivalence and 12-block distributed colored recovery pass correctness gates; timing is excluded |
| EXP-VAL-001 | central independent checks; optional comparator | [independent verification and optional matched comparator](EXP-VAL-001.md) | clean $p$-Laplace, Ginzburg--Landau, affine-patch, and nonaffine hyperelastic verification passes for the tested smooth branches |
| EXP-P3D-ROUTE-001 | merged into `EXP-ROUTE-001` Tier B | [route study](EXP-ROUTE-001.md) | exact high-/low-order confirmation matrix is prepared; cluster repetitions remain unsubmitted |
| EXP-ROUTE-001 | central | [second-architecture fixed-state and full-solve route study](EXP-ROUTE-001.md) | common-commit workstation calibration completed and independently admitted 12 blocks/36 routes; Karolina training, holdout, and Tier-B rows remain unsubmitted |
| EXP-GLOB-001 | supplementary unless retained by claim audit | [controlled globalization evidence](EXP-GLOB-001.md) | all 60 bounded local executions completed with a failed endpoint-identity gate; the 60-row full-rank archive is prepared offline but unsubmitted |
| EXP-DISC-001 | conditional | [separated discretization study](EXP-DISC-001.md) | clean named-rule fixed-state $P_1$/$P_2$/$P_4$ study is complete; nonlinear high-order endpoints remain in the unsubmitted Karolina matrix |
| EXP-TOPO-001 | supplementary software demonstration | [scope and corrected-unit diagnostic](EXP-TOPO-001.md) | corrected 1/2-rank fixed-work pilot complete; no KKT-quality result and no optimization-solution claim allowed |
| EXP-SCALE-001 | central distributed-viability evidence | [fixed-policy Karolina viability](EXP-SCALE-001.md) | required/optional matrices, frozen analysis contract, strict analyzer, and post-job accounting collector are prepared and tested; not submitted; allocation revalidation required |

The table is an execution index, not an evidence table. Numerical claims are
adjudicated separately after every retained card has a terminal outcome.
