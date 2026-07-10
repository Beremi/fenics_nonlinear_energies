# Paper Workflow

This directory contains the manuscript source, generated paper assets, and
local checks for the nonlinear finite-element energy problems paper. The
paper-facing story is the conditional equivalence and verification of element,
constitutive, and colored-recovery derivative placement in distributed
nonlinear finite elements. JAX supplies local differentiated kernels and PETSc
supplies owned sparse algebra. Performance, crossover, and scaling claims are
excluded until the paired distributed experiment satisfies the admission
contract.

## Writing Style

Use the ignored local snapshot in `paper/style_guide/` before manuscript edits:

1. Read `paper/style_guide/AGENTS.md`.
2. Read `paper/style_guide/style_fingerprint/agent_quick_reference.md`.
3. Preserve this paper's current LaTeX conventions, especially `cleveref` and
   `\Cref`, while applying the guide's prose, structure, mathematical
   exposition, and claim-scoping rules.

The style-guide snapshot is local helper material. It is ignored through
`.git/info/exclude` and must not be staged.

## Main Commands

Run commands from the repository root unless noted otherwise.

- `make -C paper figures`: regenerate generated figures after figure-source or
  plotting changes.
- `make -C paper tables`: regenerate generated tables after table-source or
  table-generator changes, including the revision evidence tables and their
  diagnostic manifest. This default intentionally cannot create publication
  evidence.
- `make -C paper literature`: regenerate the literature source index after
  bibliography or literature-manifest changes.
- `make -C paper literature-check`: validate bibliography metadata and require
  the generated literature source index to be current without rewriting it.
- `make -C paper submission-bundle`: refresh the local submission bundle after
  manuscript, bibliography, table, figure, or provenance-source changes.
- `make -C paper publish-check`: verify archive-neutral paper provenance,
  including submission-bundle manifest SHA-256 records.
- `make -C paper submission-check`: build the PDF and run the LaTeX-log scan,
  `qpdf`, figure/table aux-order check, hard-float placement allowlist,
  PDF-text manuscript hygiene check, literature-source check, submission-bundle
  hash check, revision-evidence admission check, and archive-neutral asset
  validation.
- `make -C paper release-blockers`: print final-release blockers that local
  build checks cannot resolve.
- `make -C paper release-check`: run `submission-check` and then fail unless
  final-release blockers are resolved.

Create clean table sources through the managed producer workflow; do not copy
or relabel pilot outputs:

```bash
./.venv/bin/python experiments/analysis/finalize_revision_publication_campaign.py \
  init-plan --output artifacts/reproduction/<clean-campaign>/execution_plan.json
```

The ordered dependency, execution, finalization, relocation-verification, and
admission commands are documented in
[`docs/reference/revision_publication_campaign_finalization.md`](../docs/reference/revision_publication_campaign_finalization.md).

Audit a candidate evidence root before trying to generate final tables:

```bash
./.venv/bin/python paper/scripts/admit_revision_publication_evidence.py audit \
  --evidence-root artifacts/reproduction/<clean-campaign>/publication \
  --audit-json /tmp/revision-evidence-audit.json
```

The audit is read-only and reports blockers separately for all 14 configured
inputs.  After the managed finalizer has verified every clean source from one
immutable experiment commit, request the versioned source manifest explicitly:

```bash
./.venv/bin/python paper/scripts/admit_revision_publication_evidence.py admit \
  --evidence-root artifacts/reproduction/<clean-campaign>/publication \
  --manifest-out artifacts/reproduction/<clean-campaign>/publication/publication_evidence_manifest.json
```

`admit` writes nothing when any source is diagnostic, dirty, stale, missing a
terminal gate, or missing command/environment/hash provenance.  Do not create
or edit the source manifest manually.  With the emitted manifest, generate the
final tables explicitly:

```bash
make -C paper tables \
  REVISION_EVIDENCE_ROOT=../artifacts/reproduction/<clean-campaign>/publication \
  REVISION_EVIDENCE_CLASS=publication \
  REVISION_EVIDENCE_MANIFEST=../artifacts/reproduction/<clean-campaign>/publication/publication_evidence_manifest.json
```

Publication mode verifies the immutable experiment commit, a clean descendant
release commit, and every admitted input hash.  Both the table generator and
submission checker repeat the semantic source audit; an `admitted=true` flag
alone is insufficient.  The full contract and source-specific gates are documented in
`paper/protocols/REVISION-EVIDENCE-ADMISSION.md`. `submission-check` and
`release-check` are expected to fail while the tables remain diagnostic, the
bundle is stale, or final release metadata is absent.

## Current Release Blockers

The working manuscript builds, but final submission still requires:

- Clean immutable reruns and a publication evidence-source manifest for every
  displayed revision table.
- The executed, admitted paired distributed route/crossover campaign, or a
  further scope reduction that removes that central empirical claim.
- A refreshed clean submission bundle.
- Target venue/template and required declarations; template accommodation is
  outside the current requested scope.
- Root repository license.
- Durable software/artifact archive and archival DOI.
- Final integration of the local submission bundle into that licensed archive.

Do not remove these blockers from `paper/todo.md` or
`paper/publish_readiness_knowledge_graph.md` until the corresponding evidence
is present in the current repository state.

## Manuscript Guardrails

- Keep manuscript claims tied to assumptions, citations, proofs, or reported
  numerical evidence.
- Interpret figures and tables in body text, not only captions.
- Avoid process-local wording such as local paths, run tags, repository-local
  labels, and defensive software-ranking phrasing.
- Do not introduce new hard `[H]` floats without a reasoned allowlist entry in
  `paper/scripts/check_float_placements.py`.
- Keep validation and performance evidence conceptually separate.
- Do not change final figure/table assets without finding or updating the
  generation path first.
