# Prepared Karolina Publication-Revision Campaigns

This directory contains execution-prepared, but **not submitted or
scientifically released**, Karolina
CPU artifacts for `EXP-ROUTE-001`, `EXP-DISC-001`, and `EXP-SCALE-001`. The
authoritative row-level specification is
[`campaign_matrix.csv`](campaign_matrix.csv); its SHA-256 at preparation time is
`0010453084157f5ccf0ba307ed26b377c4aff80ad9f095fa359270e90ad5b1a5`.

The recorded IT4I allocation ends on **2026-07-11**. No command in this bundle
may be admitted or submitted from that historical record. Before even a Slurm
admission test, recheck SCS and `sacctmgr` and provide a new validity date. The
default workflow only writes commands and never invokes `sbatch`.

## Frozen campaign scope

| Experiment/tranche | Slurm rows/jobs | In-allocation process executions | Ceiling |
| --- | ---: | ---: | ---: |
| `EXP-ROUTE-001` paired baseline blocks | 78 | 210 route processes | 51.00 node-hours |
| `EXP-ROUTE-001` paired quadrature-factor blocks | 18 | 54 route processes | 12.00 node-hours |
| `EXP-ROUTE-001` replicated synthetic mechanism diagnostic | 9 | 9 MPI processes | 1.20 node-hours |
| `EXP-DISC-001` | 6 | 6 solve processes | 13.25 node-hours |
| `EXP-SCALE-001` Hyperelasticity | 4 | 20 repeated solve processes | 22.50 node-hours |
| Required total | 115 | 299 | 99.95 node-hours |
| Optional paired high/low-order route confirmations | 30 | 60 route processes | 45.00 node-hours |
| Optional Plasticity3D viability | 3 | 15 repeated solve processes | 17.50 node-hours |
| All optional rows | 33 | 75 | 62.50 node-hours |
| Full matrix (must not be submitted as one tranche) | 148 | 374 | 162.45 node-hours |

A “Slurm job” in this table is one matrix row and one `sbatch` submission. A
paired block deliberately launches two or three route processes sequentially
inside that allocation; Hyperelasticity/scaling rows launch five repeated
processes. Process executions are therefore not additional scheduler jobs.

All jobs use account `fta-26-40`, QoS `3571_6328`, one thread per MPI rank,
`block:block` placement, a sequential explicit CPU map, and local NUMA binding.
Rows with one or two nodes use `qcpu_exp`; larger rows use `qcpu`. The commands
contain no `--exclusive`, `--mem`, or `--mem-per-cpu` option.

## Safe local preparation

The following is the only command exercised while preparing the paper:

```bash
DRY_RUN=1 \
CAMPAIGN_ID=paper_revision_karolina_prepared_v9 \
bash experiments/runners/paper_revision_karolina/submit_prepared_campaigns.sh
```

Optional rows are always prepared as separate tranches. The 45-node-hour
route-confirmation scope is split before any real submission: use
`ONLY_OPTIONAL=1 EXPERIMENTS=EXP-ROUTE-001 ROUTE_PHASE=training` for the exact
20-row rank-1/8 training phase, then use `ROUTE_PHASE=holdout` only after the
training model has been frozen; the latter selects the exact 10-row rank-32
holdout phase. A phase-incomplete or overlapping master archive is rejected.
For the 17.5-node-hour
optional scaling tranche use `ONLY_OPTIONAL=1 EXPERIMENTS=EXP-SCALE-001`; this
selects exactly the three Plasticity3D rows. Hyperelasticity and Plasticity3D
scaling can never share a real submission root. The default dry run can inventory
the 99.95-node-hour required rows, but that inventory is not a scientifically
safe real-submission tranche. Real submission requires exactly one explicit
`EXPERIMENTS` value and explicit comma-separated `TIERS`; downstream
discretization, scaling, and Tier-B tiers also require a hash-checked human
release record through `ADMISSION_GATE`.
Generated plans are isolated below
`artifacts/reproduction/paper_revision_karolina/<campaign-id>/` and contain the
selected matrix, exact shell-quoted commands, a prepared manifest, and
`reviewed_source_freeze.json`. Each queued command carries the preparation-time
Git commit, matrix SHA-256, and the SHA-256 of that complete reviewed-source
freeze. The batch script rejects a different clean commit, changed matrix,
changed freeze, missing reviewed source, or changed reviewed-source hash before
it starts a scientific runner.
Internal manifest paths are archive-relative, so a completed tranche can be
copied back or renamed without invalidating its plan, command, source-freeze,
or release-authorization references. Historical `sbatch` command lines retain
their original execution paths as provenance and are not rerun after copy-back.
Every preparation requires a fresh output root and verifies the complete
reviewed source-hash map before writing commands. Real execution records
`submitting` before its first scheduler call, atomically persists progress after
each response, and records `partial_submission` on interruption or failure.

Preparation also runs a scheduler-free integrity preflight. It validates the
selected matrix rows, node-hour total, optional-tranche boundaries, exact
resource arguments, forbidden options, source freeze, and hashes of the plan
and command file. It can be repeated after copy-back without contacting Slurm:

```bash
./.venv/bin/python \
  experiments/runners/paper_revision_karolina/preflight_prepared_campaign.py \
  --campaign-root artifacts/reproduction/paper_revision_karolina/<campaign-id>
```

Real route submissions must additionally set `ROUTE_PHASE=training` for ranks
1 and 8 or `ROUTE_PHASE=holdout` for rank 32. The holdout path is rejected
unless `MODEL_FREEZE_RECEIPT` names a receipt conforming to
`paper/protocols/route-model-freeze-v1.schema.json`; the receipt binds the
complete 76-case training plan, clean commit, training analysis, and frozen
model hashes before any holdout scheduler call. `ENV_SETUP` and `ENV_LOCK` are
also mandatory for scheduler admission. Both files are copied into the tranche,
hash-bound in every command, verified before the setup is sourced, and linked
to the compute-node compiler, MPI/mpi4py, JAX/jaxlib, backend, and XLA identity
record.

After all training jobs finish, settle the complete tranche in one operation from an
offline accounting index (default) or the explicitly opted-in live mode:

```bash
./.venv/bin/python experiments/analysis/finalize_karolina_campaign_archive.py \
  --campaign-root <campaign-root> --offline-index <accounting-index.json> \
  --receipt <detached-receipt.json>
```

The detached receipt contains the checksum-manifest digest. After copy-back,
use `--verify-only --expected-checksum-manifest-sha256 <pre-copy-digest>`.
Verification rejects missing, additional, changed, or symlinked archive files.

Before preparing a real rank-32 route phase, freeze the prespecified model from
the complete local workstation archive and the checksum-sealed Karolina
training archive:

```bash
./.venv/bin/python experiments/analysis/freeze_route_training_model.py \
  --workstation-root <workstation-root> \
  --karolina-training-root <training-root> \
  --output-dir <training-freeze-output>
```

The utility opens only the 76 planned rank-1/8 jobs, requires 74
equivalence-admitted model rows, checks the full-rank 13-feature design, and
writes `training_analysis.json` plus `frozen_model.json`, both explicitly
recording that zero holdout rows were seen. A human then records those hashes,
the checksum-sealed training manifest, and the review decision in a copy of
`paper/protocols/route-model-freeze-v1.example.json`. The holdout preparer
revalidates the complete training archive and both output schemas before its
first scheduler call.

After both optional phases have completed and been settled, create the only
Tier-B manifest accepted by the endpoint analyzer:

```bash
./.venv/bin/python experiments/analysis/aggregate_route_tier_b_manifests.py \
  --training-manifest <training-root>/prepared_manifest.json \
  --holdout-manifest <holdout-root>/prepared_manifest.json \
  --archive-root <common-copy-back-root> \
  --output <common-copy-back-root>/route_tier_b_campaign_master_manifest.json
```

This step is scheduler-free. It recomputes exact 20/10 phase coverage, checks
the common clean commit and environment, binds the holdout receipt back to the
admitted training manifest, and emits only archive-relative paths.

A real submission now journals and fsyncs an intent before every `sbatch`
call and a result afterward. `resume_partial_submission.py` submits only case
IDs without accepted job IDs. An unmatched intent is ambiguous external state
and blocks automatic resume until a human reconciles it with the scheduler.

`SBATCH_TEST_ONLY=1` can be combined with `DRY_RUN=1` to inspect the exact
admission-test commands without contacting Slurm:

```bash
DRY_RUN=1 SBATCH_TEST_ONLY=1 \
bash experiments/runners/paper_revision_karolina/submit_prepared_campaigns.sh
```

## Revalidation procedure for a future authorized session

Do not perform these steps as part of the current revision session. In a future
session with explicit submission authorization:

1. Verify that the Karolina CPU allocation is active and record its new end
   date.
2. Verify account `fta-26-40`, QoS `3571_6328`, and both `qcpu_exp` and `qcpu`.
   The compute-node guard independently checks account, QoS, partition, nodes,
   tasks, tasks per node (including Slurm's compact `128(xN)` spelling), and
   CPUs per task against the selected matrix row before creating solver output.
3. Verify the code is a clean, committed checkout and that the matrix hash
   matches the reviewed manifest.
4. Verify the private PETSc/petsc4py environment reports PETSc 3.24.x and
   `Mat.setPreallocationCOO`, with both Hypre and MUMPS external packages.
5. Run one-node admission tests first by setting all of:
   `DRY_RUN=0`, `SBATCH_TEST_ONLY=1`, `ALLOCATION_REVALIDATED=YES`,
   `ACCOUNT_QOS_REVALIDATED=YES`, and a future
   `ALLOCATION_VALID_UNTIL=YYYY-MM-DD`.
6. Review the resulting `test_only_results.jsonl`. A real run additionally
   requires `SUBMIT_CONFIRMED=YES`, one experiment, explicit tiers, an explicit
   user decision, and a clean worktree. Downstream tiers require a versioned
   human release-authorization record enumerating reviewed artifact paths and
   SHA-256 values; it is a release decision, not automated numerical evidence.

The maintained record definition is
[`human-release-authorization-v1.schema.json`](../../../paper/protocols/human-release-authorization-v1.schema.json).
Copy
[`human-release-authorization-v1.example.json`](../../../paper/protocols/human-release-authorization-v1.example.json)
and replace every example value. The example deliberately uses zero commit and
SHA-256 values and an `EXAMPLE_ONLY` reviewer, so it cannot pass the contextual
matrix, source-commit, or artifact checks and is not itself an authorization.

The program repeats these checks immediately before invoking `sbatch`, and the
batch script repeats the allocation and clean-commit checks on the compute node.

## Scientific gates before submission

- `EXP-ROUTE-001` covers `P1(L1)`, `P1(L2)`, `P2(L1)`, and `P4(L1)` at 1, 8,
  and 32 ranks for elastic-target and mixed-branch analytic states. Ranks 1 and
  8 are frozen training points; rank 32 is holdout and must not be inspected
  before fitting. The exact split, features, error bands, and ordering gates are
  in
  [`EXP-ROUTE-001-analysis-contract.json`](../../../paper/protocols/EXP-ROUTE-001-analysis-contract.json).
- The two analytic fixed states passed local `P1(L1)` and `P2(L1)` branch and
  action diagnostics. Karolina repeats the diagnostic for every configuration;
  a changed classification blocks timing use.
- The fixed-state tangent-action arrays must agree across routes at the
  prespecified tolerance; failed routes remain censored rows.
- Each Slurm row is now one paired all-route allocation block. Three-route
  comparisons use a hash-seeded randomized base order and three cyclic
  rotations; two-route comparisons use a hash-seeded base and four or ten
  alternating blocks. Route position is exactly balanced, and every timing is
  the maximum of saved per-rank elapsed values.
- Fixed-state admission uses four deterministic actions, the saved
  gradient/residual, exact state and branch counts, and direct CSR comparison
  at feasible one-rank `P1` points before timing is exposed.
- Factorized rows separately vary quadrature at fixed `P2(L1)` and vary kernel,
  color, insertion, communication, and imbalance factors at 1, 8, and 32
  ranks. Cold/warm timing and memory are measured, not independently varied.
  The analyzer fits a descriptive shared-stage log-time diagnostic on ranks 1
  and 8 and validates it at rank 32 without refitting. These synthetic,
  non-route-faithful values are never inserted into the production selector.
- Colored SFD at `P4(L1)` is a prespecified non-attempt motivated by pilot
  memory risk. It has no Slurm row, carries no measured memory threshold,
  remains visible in the finite empirical map, and is never imputed.
- Tier-B route confirmations use `ksp_rtol=1e-8` and KSP cap 500. The matrix
  validator rejects the obsolete loose `1e-2` policy that produced
  route-sensitive one-step states in the local pilot.
- Every `p3d_solve` row, including discretization and optional scaling rows,
  uses the numerically checked reference-elastic-energy Riesz stopping metric. The
  executor rejects coefficient stopping, invalid SPD inertia, stale endpoint
  Riesz evidence, changed GMRES/Hypre tolerances, and completed rows that fail
  the residual gate.
- `EXP-DISC-001` has five explicit sequential release stages: P4(L1) smoke;
  the paired P4(L1) 24/125-point quadrature stage; P4(L2) 24-point mesh stage;
  P4(L2) 125-point mesh-quadrature stage; and the tight-tolerance stage. A
  later stage needs a fresh authorization after the preceding stage is
  inspected. Each successful state is
  re-evaluated with the common 24-point and positive 125-point rules.
  `analyze_plasticity3d_discretization.py` must admit the complete six-row
  evidence before any discretization interpretation is released.
- `EXP-SCALE-001` is a fixed-policy first-step Hyperelasticity series. The PMG
  hierarchy, Hypre coarse policy, tolerances, mesh, and output scope do not
  change with rank. The optional Plasticity3D series is separate.
  `analyze_exp_scale_001.py` enforces endpoint and raw-rank timing gates;
  `collect_slurm_accounting.py` creates the required settled
  `sacct_final.json` after completion and is offline by default.
- Numerical rows from a dirty checkout, a changed matrix, unequal stopping
  criteria, or a failed endpoint-equivalence check are pilots, not paper data.
- The strict finite-map analysis is
  [`analyze_plasticity3d_route_cost_model.py`](../../analysis/analyze_plasticity3d_route_cost_model.py).
  It reconstructs saved states/actions before admitting timing and refuses to
  fit the frozen model until every minimum train/holdout gate is satisfied.
- The replicated factor study is a descriptive mechanism diagnostic, not a
  selector feature or selector gate. The production selector remains
  fail-closed until clean submitted route records, complete paired blocks,
  production holdout gates, and the strict high/low-order endpoint analyzer all
  pass. No cluster result currently exists.
- Tier-B endpoint admission requires, for every paired block, the complete
  compute-node environment sections and settled raw `sacct --parsable2`
  evidence. The analyzer reparses that raw record and checks job identity,
  account, QoS, partition, node/CPU allocation, terminal state, and exit code
  against the matrix before exposing timing.

Protocol cards: [`EXP-ROUTE-001`](../../../paper/protocols/EXP-ROUTE-001.md),
[`EXP-DISC-001`](../../../paper/protocols/EXP-DISC-001.md), and
[`EXP-SCALE-001`](../../../paper/protocols/EXP-SCALE-001.md). The static
handoff is [`handoff.yaml`](handoff.yaml), and the preparation manifest is
[`campaign_manifest.yaml`](campaign_manifest.yaml).
