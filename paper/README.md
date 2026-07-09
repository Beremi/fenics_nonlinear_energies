# Paper Workflow

This directory contains the manuscript source, generated paper assets, and
local checks for the nonlinear finite-element energy-minimization paper. The
paper-facing story is the JAX+PETSc toolset for local automatic
differentiation, sparse distributed assembly, nonlinear globalization, Krylov
linear solvers, and preconditioner policy, with pure JAX, FEniCS, and selected
external/reference-model comparisons used only within their stated evidence
scope.

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
  table-generator changes.
- `make -C paper submission-bundle`: refresh the local submission bundle after
  manuscript, table, figure, or provenance-source changes.
- `make -C paper publish-check`: verify archive-neutral paper provenance,
  including submission-bundle manifest SHA-256 records.
- `make -C paper submission-check`: build the PDF and run the LaTeX-log scan,
  `qpdf`, figure/table aux-order check, hard-float placement allowlist,
  PDF-text manuscript hygiene check, submission-bundle hash check, and
  archive-neutral asset validation.
- `make -C paper release-blockers`: print final-release blockers that local
  build checks cannot resolve.
- `make -C paper release-check`: run `submission-check` and then fail unless
  final-release blockers are resolved.

`publish-check` and `submission-check` are expected to pass on the current
paper-readiness branch. `release-check` is expected to fail until the final
external submission decisions are made.

## Current Release Blockers

The manuscript and local provenance checks are currently green, but final
submission still requires decisions and artifacts outside the local paper build:

- Target venue/template and required declarations.
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
