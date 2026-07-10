# Revision Table Publication-Evidence Admission

## Purpose

The four revision tables consume 14 JSON inputs with different schemas.  A
table-input hash by itself does not establish that an input was produced by a
clean publication run, terminated correctly, satisfied its scientific gates,
or still corresponds to the checked-out producer.  The admission boundary is
therefore implemented by
`paper/scripts/admit_revision_publication_evidence.py` and is repeated by both
the table generator and final evidence checker.

The workflow has two modes:

1. `audit` is read-only.  It returns a versioned diagnostic object and prints
   every global and per-input blocker.  An audit object is never publication
   evidence.
2. `admit` runs the identical audit and writes a schema-version-2 source
   manifest only if all 14 inputs pass.  It then reads that manifest back and
   repeats the complete audit before returning success.

No manifest is created for a partly successful campaign.  Do not hand-author,
copy, or patch a source manifest: downstream tools independently re-evaluate
the source files and reject a manifest whose recorded audit differs.

## Configured inputs and producer identities

| Key | Input relative to the evidence root | Required producer or analyzer |
| --- | --- | --- |
| `plaplace` | `EXP-VAL-001/plaplace_manufactured.json` | `run_manufactured_plaplace_verification.py` |
| `ginzburg_landau` | `EXP-VAL-001/ginzburg_landau_manufactured.json` | `run_manufactured_ginzburg_landau_verification.py` |
| `hyperelastic_patch` | `EXP-VAL-001/hyperelastic_affine_patch.json` | `run_hyperelastic_affine_patch_verification.py` |
| `hyperelastic_nonaffine` | `EXP-VAL-001/hyperelastic_nonaffine_quadrature_refinement_v2/result.json` | `run_manufactured_hyperelastic_verification.py` |
| `smooth_derivatives` | `EXP-DERIV-001/smooth_fixed_element_v1.json` | `run_smooth_element_derivative_verification.py` |
| `p1_derivatives` | `EXP-DERIV-001/p1_l1_fixed_element_v2.json` | `run_paper_derivative_verification.py` |
| `p2_derivatives` | `EXP-DERIV-001/p2_l1_fixed_element_v2.json` | `run_paper_derivative_verification.py` |
| `p4_derivatives` | `EXP-DERIV-001/p4_l1_fixed_element_v2.json` | `run_paper_derivative_verification.py` |
| `material_point` | `EXP-MC-001/material_point_verification.json` | `run_plasticity3d_material_point_verification.py` |
| `distribution` | `EXP-DIST-001/distribution_equivalence.json` | `run_hyperelasticity_distribution_equivalence.py` |
| `p1_quadrature` | `EXP-DISC-001/p1_l1_fixed_state_quadrature_v2.json` | `run_plasticity3d_fixed_state_quadrature.py` |
| `p2_quadrature` | `EXP-DISC-001/p2_l1_fixed_state_quadrature_v2.json` | `run_plasticity3d_fixed_state_quadrature.py` |
| `p4_quadrature` | `EXP-DISC-001/p4_l1_fixed_state_quadrature_v2.json` | `run_plasticity3d_fixed_state_quadrature.py` |
| `route_analysis` | `EXP-ROUTE-001/analysis_contract_v1/analysis.json` | `analyze_plasticity3d_route_cost_model.py` |

The configured producer must exist, and at least one source payload, companion
manifest, or validated run record must bind it by its current SHA-256 digest.
The admission manifest also binds the admission tool and downstream table
generator by path and SHA-256.

## Checks applied independently to every input

For each key, the tool checks all of the following before setting `admitted`:

1. The source and its experiment-level companion manifest exist and parse as
   JSON objects containing no non-finite scientific values.
2. An input-level `publication_evidence` field, when present, is `true`.  The
   companion manifest must always say `publication_evidence: true`; a `run_kind`
   field, when present, must be `publication`.
3. Every Git declaration in the payload, companion, and required run records
   says clean and names one common immutable experiment commit.  That commit
   must be an ancestor of the clean release `HEAD`; it need not equal the later
   table/release commit.  Producer and input hashes must still match the release
   checkout.  This avoids a circular generate--commit--invalidate workflow.
4. A non-empty command (or argument vector) and non-empty environment/version
   record are retained.
5. The source-specific terminal and scientific gates below pass.
6. Required publication run records validate against the versioned strict run
   record contract.  Material-point evidence requires one record; distribution
   evidence requires the one-rank, two-rank, and four-rank records.
7. The companion manifest binds the table input by SHA-256.  Every resolvable
   declared code, input, output, and artifact hash is recomputed; a missing or
   stale file blocks admission.
8. The configured producer/analyzer digest matches the current file.  Route
   analysis additionally binds its analysis contract by path and SHA-256.

Publication payloads also require a versioned family `source_schema` and a
versioned `publication_provenance` block produced by the managed campaign
finalizer.  Experiment companions use the versioned
`revision-publication-companion` schema.  The latter binds canonical code,
configuration, input, raw-output, execution-receipt, and final-output paths;
absolute paths, `..`, symlink escapes, basename aliases, and malformed digests
are rejected.  Families without external file inputs must state the explicit
`no_external_file_inputs` policy rather than omit provenance silently.

The source manifest stores every check and blocker, not only a Boolean result.
During table generation and submission checking, a fresh audit must reproduce
the stored input records and deterministic audit digest exactly.

## Source-specific terminal and scientific gates

- Manufactured scalar studies require terminal `passed`, at least one
  convergence-rate row, and every mesh level marked `converged`.  Admission
  fixes the four-level `[8,16,32,64]` protocol, residual tolerance bounds,
  nonnegative errors, and final rates of at least 1.75 in $L^2$ and 0.85 in
  $H^1$.
- The affine hyperelastic patch requires terminal `passed`, a positive declared
  tolerance, and every recorded scalar/vector defect below that tolerance.
- The nonaffine hyperelastic study requires terminal `passed`, every Boolean
  gate true, and every mesh level converged.
- Derivative studies require terminal `passed`; route, symmetry, and centered
  finite-difference metrics are compared again with their declared tolerances.
  A recorded fixed-branch gate must be true.
- Material-point verification requires terminal `passed` and the FP64,
  interior, interface, degeneracy-finiteness, and rotation gates all true.
- Distribution equivalence requires terminal `passed`, the algebraic gate, and
  every derivative, exact-object, exact-topology, and linear-solve gate true.
- Fixed-state quadrature studies require terminal `completed`, a common free-DOF
  set, and both the solve and reference rules in a nontrivial evaluation list.
- Route analysis has two equally admissible, prespecified terminals.  A passed
  model uses `predictive_selector_admissible`.  A failed frozen validation gate
  or a rank-deficient frozen design uses exactly `finite_empirical_map_only`.
  Both terminals require the complete 102-slot map, all 96 active rows admitted
  and publication-model eligible, exactly 74 training and 22 untouched holdout
  rows, the six frozen $P_4$ colored-recovery non-attempts with null timing
  fields, no invalid records, and a hash-bound 30-of-30 Tier-B endpoint analysis.
  The negative branch contains only its status, selector rejection, frozen
  feature order, row counts, and named failure decision.  It contains no
  coefficients, predictions, winner/order rows, recommendation, crossover, or
  post-fit confirmation.  `not_fit_insufficient_data` and an unreleased design
  are evidence defects, not publishable negative outcomes.
- The replicated synthetic factor study is descriptive and reportable on
  either route terminal.  A consistent diagnostic failure does not block the
  finite map or selector because the frozen contract sets
  `required_for_selector_claim: false`.  Its design identity and internal
  pass/failure consistency remain integrity gates.  It must keep
  `calibration_integrated: false`, `selector_blockers: []`, and selector use
  `descriptive_replicated_synthetic_non_route_faithful_proxy`; it is never a
  selector feature or predictor.
- Tier-B endpoint terminal names and the comparative-ranking Boolean are bound
  bidirectionally: descriptive timing means `false`, while comparative ranking
  means `true`.  An observed comparative endpoint may coexist with a failed
  predictive selector.

The evidence directory component `analysis_contract_v1` is the stable layout
name for `analysis_schema_version: 1`; it is not the scientific route-contract
version.  The hash-bound scientific contract is currently `contract_version: 2`
in `EXP-ROUTE-001-analysis-contract.json`.  Keeping these namespaces distinct
avoids moving the exact 14-source interface whenever the frozen scientific
contract is revised before execution.

## Offline route-dependency staging

The workstation campaign and the copied-back Karolina archive enter the
managed finalization root through a separate dependency plan.  Generate this
plan only at the clean experiment commit shared by both archives:

```bash
HEAD=<full-40-digit-experiment-commit>
EVIDENCE=artifacts/reproduction/<campaign>
./.venv/bin/python experiments/analysis/stage_route_publication_dependencies.py plan \
  --expected-commit "$HEAD" \
  --workstation-source /path/to/completed/workstation/archive \
  --karolina-source /path/to/copied-back/karolina/archive \
  --endpoint-relative path/inside/karolina/to/tier_b_endpoint_analysis.json \
  --output "$EVIDENCE/route_dependency_plan.json"
```

Plan generation independently applies the frozen workstation, Karolina, and
Tier-B endpoint gates.  It also records every regular file and SHA-256 digest
in each source tree.  Symbolic links, incomplete workstation closure,
nonadmissible route rows, stale endpoint evidence, mixed commits, or later
source-tree changes block staging.

Execute the three local preparation commands in the recorded order:

```bash
for COMMAND_ID in \
  prepare_workstation_archive \
  prepare_route_campaign_master \
  prepare_tier_b_endpoint_analysis
do
  ./.venv/bin/python experiments/analysis/finalize_revision_publication_campaign.py execute \
    --plan "$EVIDENCE/route_dependency_plan.json" \
    --command-id "$COMMAND_ID" \
    --evidence-root "$EVIDENCE"
done
```

These commands perform local validation and file copying only.  They neither
submit work nor contact a remote system.  The managed executor writes the
three fingerprinted receipts required by the canonical source plan.  The
canonical Tier-B endpoint copy remains inside the relocated Karolina archive,
so the route analyzer can enforce its independent archive-confinement gate.
The dependency plan and receipts must be retained with the final evidence.

These checks establish admission for the quantities consumed by the revision
tables.  They do not broaden the scientific scope of an experiment.

## Commands and expected failure behavior

Run a diagnostic audit:

```bash
./.venv/bin/python paper/scripts/admit_revision_publication_evidence.py audit \
  --evidence-root artifacts/reproduction/<campaign>/publication \
  --audit-json /tmp/revision-evidence-audit.json
```

`audit` exits successfully after reporting blockers so it can be used during a
campaign.  The JSON output has
`publication_evidence: false` and status `publication_admission_blocked` or
`eligible_for_manifest_creation`.

Attempt final admission only after the audit is clean:

```bash
./.venv/bin/python paper/scripts/admit_revision_publication_evidence.py admit \
  --evidence-root artifacts/reproduction/<campaign>/publication \
  --manifest-out artifacts/reproduction/<campaign>/publication/publication_evidence_manifest.json
```

`admit` exits nonzero and writes no source manifest if any global or per-input
check fails.  The current `paper_revision_2026_07_10/pilots` tree is expected to
fail: it is dirty diagnostic evidence, its experiment manifests are pilots,
its strict run records are not publication records, several producer hashes
are stale after implementation work, the strengthened nonaffine input is not
bound by its old group manifest, and route analysis has no admitted map or
selector.

After successful admission, pass the exact emitted path to the table generator:

```bash
make -C paper tables \
  REVISION_EVIDENCE_ROOT=../artifacts/reproduction/<campaign>/publication \
  REVISION_EVIDENCE_CLASS=publication \
  REVISION_EVIDENCE_MANIFEST=../artifacts/reproduction/<campaign>/publication/publication_evidence_manifest.json
```

Any subsequent source, producer, admission-tool, or table-generator change
invalidates the manifest and requires a new clean audit/admission cycle.

The final table checker additionally requires exactly the 14 configured input
keys and canonical paths, exactly the four TeX outputs consumed by the
manuscript, and the four literal manuscript `\input` bindings.  It rejects path
escape and unrelated-file substitutions, reruns the hash-bound table generator
in a temporary directory, and requires byte-identical output.
