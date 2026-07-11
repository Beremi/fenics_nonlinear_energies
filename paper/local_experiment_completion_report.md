# Local Experiment Completion Report

## Purpose and decision

This report records the terminal local experiment status for the paper
revision. We distinguish completed local evidence from computations that
require an authorized parallel cluster. All locally feasible publication
experiments are now complete. The remaining numerical gaps are cluster-only,
and no scheduler, remote shell, or cluster command was invoked during this
work.

The current managed source campaign was executed from clean commit
`d71ba78aa29259a86296dfac0eb9ce86166bed23`. Its canonical 17-command plan has
SHA-256
`d4ad05dd0c9d646d53c703b2ab7553ca2dfc6b307384de36e526431eec405c5d`.
We executed the 16 commands whose inputs and outputs are local. All 16 receipts
passed the finalizer's independent fingerprint, clean-commit, producer,
configuration, input, output, and referenced-artifact checks. The remaining
command, `route_cost_analysis`, was not executed because its Karolina source
archive and Tier-B endpoint evidence do not exist.

## Resource-safety contract

We ran one managed command at a time. Each command inherited one thread for
OpenMP, OpenBLAS, and MKL, used the CPU JAX backend, and ran inside a systemd
scope with `MemoryHigh=28G`, `MemoryMax=32G`, no swap, and `OOMPolicy=kill`.
The process address-space ceiling was 64 GiB. The P4 derivative check was the
only material memory risk; its three route high-water marks were 15.61, 24.65,
and 27.04 GiB. The scope completed without an out-of-memory event, and host
memory returned to 15 GiB used after the campaign.

The managed archive is
`artifacts/reproduction/paper_revision_local_d71ba78/publication_campaign`.
It occupies approximately 75 MiB and contains the immutable plan, command
logs, 16 execution receipts, raw source payloads, prescribed states, and
quadrature arrays. We did not run the final bundle decoration because that
operation correctly requires the missing route source and dependency receipts.

## Clean managed source campaign

Table 1 summarizes the locally produced table-facing sources. A `passed`
entry denotes that both the producer's numerical gate and the managed receipt
validation succeeded.

| Experiment | Managed commands | Terminal result | Evidence-supported conclusion |
| --- | --- | --- | --- |
| `EXP-VAL-001` | `val_plaplace`, `val_ginzburg_landau`, `val_hyperelastic_patch`, `val_hyperelastic_nonaffine` | 4/4 passed | Independent manufactured and analytic checks pass for the tested smooth branches and meshes. |
| `EXP-DERIV-001` | `deriv_smooth`, `deriv_p1`, `deriv_p2`, `deriv_p4` | 4/4 passed | Fixed-element differentiation and assembled element-AD, constitutive-AD, and colored-recovery routes agree at the frozen states. |
| `EXP-MC-001` | `mc_material_point` | passed | The five branch interiors, all two-sided interfaces, rotation checks, and repeated-spectrum probes pass the declared finite-precision gates. |
| `EXP-DIST-001` | `dist_hyperelastic` | passed | The one-, two-, and four-rank canonical hyperelastic constructions are algebraically equivalent at the tested fixed state. |
| `EXP-DISC-001` | three prescribed-state producers and `disc_p1`, `disc_p2`, `disc_p4` | 6/6 completed; all source validations passed | The named-rule fixed-state study quantifies the quadrature sensitivity of energy, residual, action, and branch measures. It is not a nonlinear endpoint study. |

### Independent validation

The final p-Laplace refinement step gives rates 1.999 in $L^2$ and 0.999 in
the $H^1$ seminorm. The corresponding Ginzburg--Landau rates are 2.001 and
1.000. The affine hyperelastic patch has relative energy, residual, and Hessian
errors of $4.58\times10^{-15}$, $2.67\times10^{-15}$, and
$4.38\times10^{-16}$, respectively.

For the nonaffine hyperelastic problem, the final displacement, deformation,
and first-Piola rates are 1.887, 1.006, and 0.983. The minimum discrete
determinant over all levels is 0.844. The maximum tested load-quadrature error
is $8.29\times10^{-6}$ of the corresponding finite element error. These
results support smooth-branch formulation consistency; they do not establish
multi-basin robustness or a general nonlinear convergence rate.

### Derivative and branch verification

The smooth fixed-element test has maximum relative gradient and Hessian errors
of $1.29\times10^{-15}$ and $2.23\times10^{-16}$. The assembled P1, P2, and P4
comparisons all pass exact CSR-structure and frozen value tolerances. Across
the P4 checks, the maximum residual and Hessian relative errors are
$2.26\times10^{-16}$ and $2.34\times10^{-16}$, and the maximum symmetry defect
is $1.24\times10^{-16}$.

The material-point campaign contains five strict branch interiors, 15
two-sided interface pairs, 15 rotation checks, and seven repeated-principal-
value cases. All gates pass. The maximum centered Hessian-action error at the
selected finite-difference scale is $1.54\times10^{-9}$; the maximum Hessian
symmetry defect is $8.68\times10^{-17}$. This is a finite-precision branch
program check, not a proof of differentiability at branch interfaces.

### Distributed equivalence

The one-rank reference agrees with the two- and four-rank constructions in
topology, maps, state, direction, and CSR structure. The maximum relative
energy, residual, matrix, action, and linear-correction errors are
$8.67\times10^{-19}$, zero, zero, $3.35\times10^{-16}$, and
$2.33\times10^{-16}$, respectively. Timing remains descriptive and is not
admitted by this correctness experiment.

### Fixed-state quadrature sensitivity

Table 2 compares the 24-point tetrahedral rule with the enriched 125-point
reference at the same saved state and direction. The action column uses the
free-DOF vector relative $L^2$ difference, while the branch column gives the
$L^1$ difference of absolute-weight branch fractions.

| Element | Energy | Free residual | Free Hessian action | Branch measure |
| --- | ---: | ---: | ---: | ---: |
| P1 | $1.76\times10^{-16}$ | $6.90\times10^{-16}$ | $6.55\times10^{-16}$ | $7.46\times10^{-17}$ |
| P2 | $5.90\times10^{-5}$ | $1.40\times10^{-3}$ | $2.11\times10^{-2}$ | $3.33\times10^{-3}$ |
| P4 | $6.28\times10^{-4}$ | $2.68\times10^{-2}$ | $6.91\times10^{-2}$ | $8.51\times10^{-3}$ |

The energy differences alone substantially understate the residual and action
sensitivity for P2 and P4. Consequently, the nonlinear P4 comparison must use
separate 24- and 125-point solves with admitted stopping evidence; the local
fixed-state table cannot replace those cluster endpoints.

## Other completed local campaigns

The following clean campaigns were completed and admitted earlier on this
branch. We retain their immutable experiment commits instead of rerunning them
after unrelated documentation changes.

- `EXP-STOP-001`: all 45 locally feasible rows completed at commit `5b2f3b5`;
  no local row is missing, invalid, failed, or censored. Seven publication-rank
  rows remain cluster-only.
- Distributed colored recovery: all 12 blocks and 36 route processes completed
  at commit `8be360e`; the archive passed its correctness-only admission gate.
- `EXP-GLOB-001`: all 60 prescribed two-rank solves completed at commit
  `692bc6d`. The terminal result is a valid negative outcome,
  `completed_with_failed_identity_gate`; therefore no timing or population-
  robustness claim is released.
- `EXP-TOPO-001`: the corrected-unit one-/two-rank fixed-work diagnostic is
  complete. Topology optimization remains supplementary, with no KKT-quality
  or optimization-solution claim.
- `EXP-ROUTE-001` workstation calibration: all 12 balanced blocks and 36
  sequential route processes completed at commit `d71ba78`; the 273-file hash
  closure and independent 12-row archive validator passed. The manifest
  SHA-256 is
  `9ff54f117f34dfb03e93fee7dbd88243278ecd1fe989604f7e7b8fc042cb4dab`.
  All six route permutations occurred twice; the maximum action discrepancy
  was $2.34\times10^{-16}$, the saved-state and gradient/residual discrepancies
  were zero at stored precision, and the maximum process RSS was 6.09 GiB.
  Every block retains `timing_claim_released: false`.

## Computations that remain cluster-only

The following work must not be approximated by workstation runs. Scripts and
scheduler-free inventories exist, but execution requires a current allocation,
independent protocol sign-off, explicit human release, and a clean frozen
commit.

1. Complete the seven `EXP-STOP-001` parallel rows: four P4 nonlinear
   calibrations and three publication-rank MPI-consistency checks. The
   scheduler-free archive
   `artifacts/reproduction/EXP-STOP-001-karolina-5b2f3b5` is bound to the
   matching 45-row local campaign, contains a 23.0-node-hour ceiling, and
   passed offline preflight without scheduler contact. It is intentionally
   non-submittable until an environment contract and human release exist.
   These rows produce the detached STOP adjudication required by Tier B.
2. Run the required Karolina route, factor, discretization, and
   hyperelastic-scaling tranches. The current offline-preflighted inventory is
   `paper_revision_karolina_prepared_v12_d71ba78`; it contains 115 required
   rows with a 99.95-node-hour ceiling, plan SHA-256
   `d84c798cbf19dfc86dfe1a558fee5db5f0e0cf5a6f084a034e629d733f4fd08c`,
   and source-freeze SHA-256
   `3b3f4ab511395d956e31818627eff157203c7fa774d51a53224ff9ed061dc420`.
3. Run the 30 optional Tier-B route rows only after the detached version-3
   STOP adjudication passes. The archive
   `paper_revision_karolina_route_optional_v12_d71ba78` has a 45.0-node-hour
   ceiling and is explicitly non-submittable.
4. Run the three optional Plasticity3D scaling rows only as the separate
   `paper_revision_karolina_p3d_scaling_optional_v12_d71ba78` tranche, whose
   ceiling is 17.5 node-hours.
5. If the globalization result remains in the paper, run its prescribed
   full-rank matrix. The scheduler-free archive
   `artifacts/reproduction/EXP-GLOB-001-karolina-d71ba78` contains 60 rows
   (30 Ginzburg--Landau and 30 hyperelasticity) and a 12.5-node-hour ceiling;
   it passed offline preflight and is explicitly non-submittable. The local
   negative outcome does not establish a probability or performance
   comparison.
6. After copy-back and checksum closure, execute the route endpoint and cost-
   model analyses locally. Fit the frozen model only after its training,
   holdout, endpoint, and coverage gates pass.

Until these items are complete, we may publish the verified local correctness
results and their explicit limitations, but not a Karolina route ordering,
crossover predictor, parallel-scaling conclusion, or nonlinear high-order
quadrature endpoint claim.
