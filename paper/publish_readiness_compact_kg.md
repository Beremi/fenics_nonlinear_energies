# Compact Publish-Readiness Knowledge Graph

Last updated: 2026-07-09.

This is the short navigation graph for paper-readiness work. Use it beside the
long audit log in `publish_readiness_knowledge_graph.md`; keep this file compact
enough that it can be read before each manuscript-editing pass.

## Core Message

The paper presents a scientific JAX+PETSc toolset and accompanying numerical
expertise for nonlinear finite-element energy minimization. The mainline object
is the JAX+PETSc realization: local JAX differentiation and constitutive
evaluation coupled to PETSc sparse assembly, Newton globalization, Krylov linear
solvers, and preconditioner policy.

Comparisons are deliberately scoped:

- pure JAX: serial reference and design/reference path where available;
- FEniCS: reference implementation for scalar and hyperelastic families;
- JAX-FEM: narrow hyperelastic external comparison on matched data;
- Sysala-family literature: slope-stability, strength-reduction, and
  reference-model context for plasticity observables;
- \v{C}ermak--Sysala--Valdman MATLAB literature: elastoplastic finite-element
  implementation context, not a validated numerical baseline.

The paper should not read as a repository report. Avoid process-local language,
local paths, campaign tags, machine names, defensive rankings, and broad
software superiority claims.

## Style Anchors

- Local style guide: `paper/style_guide/AGENTS.md`,
  `paper/style_guide/style_fingerprint/agent_quick_reference.md`, and
  `paper/style_guide/style_fingerprint/agent_cookbook.md`.
- Preserve the paper's current LaTeX convention: `cleveref`/`\Cref` for
  non-equation references and `\eqref` for equations.
- Follow the 2025 accepted KL-paper style: motivation before construction,
  definitions before dense notation, authorial `we`, punctuated displays, and
  body-text interpretation of numerical evidence.
- Current SIAM optimization style cues are method-first and evidence-scoped:
  application motivation, assumptions/problem class, computational method,
  numerical evidence, and discussion/conclusion.

The local style-guide snapshot is ignored through `.git/info/exclude`; do not
stage or commit `paper/style_guide/`.

## Section Roles

- `abstract.tex`: state the nonlinear FEM bottleneck, the JAX+PETSc toolset,
  derivative/assembly/solver ingredients, scoped comparator surface, and
  evidence summary.
- `introduction.tex`: motivate nonlinear FEM energy solves, position related
  ecosystems, state contributions, and preview the evidence without claiming a
  broad framework ranking.
- `related_work.tex`: group literature by role: FEM/PDE automation,
  differentiable FEM/JAX bridges, mechanics/topology context, sparse recovery,
  and PETSc nonlinear infrastructure.
- `methodology.tex`: define finite-element energies and derivative routes, then
  explain globalization, solver reporting, colored sparse recovery, and
  constitutive AD.
- `implementation.tex`: explain pure JAX/FEniCS reference roles and the primary
  JAX+PETSc path, including autodiff modes, linear solvers/preconditioners, and
  distributed assembly.
- `benchmarks.tex`: define problem families and discrete objectives. Put model,
  boundary data, discretization labels, and representative evidence here before
  performance claims.
- `validation.tex`: keep external/reference-model agreement separate from
  performance. Hyperelasticity is the JAX-FEM comparison; Plasticity3D is
  endpoint-surrogate agreement.
- `results.tex`: report globalization, derivative-route, scaling, memory,
  fixed-work, and topology evidence. Each block must state timing scope and
  limitation.
- `discussion.tex`: synthesize interactions among derivative routes,
  PETSc-owned sparse algebra, solver policy, validation scope, and evidence
  limits.
- `conclusion.tex`: close on the scientific toolset and next research steps,
  not on implementation housekeeping.
- `appendix.tex`: keep supporting solver-policy diagnostics and reference-formula
  or fixed-reference comparisons out of the main evidence line.

## Evidence Surface

| Family | Primary role | Comparators | Main evidence |
| --- | --- | --- | --- |
| p-Laplace | scalar nonlinear PDE and derivative-route check | FEniCS, pure JAX, JAX+PETSc | formulation agreement and strong scaling |
| Ginzburg--Landau | indefinite scalar energy/globalization | FEniCS, JAX+PETSc | energy agreement, globalization diagnostics, scaling |
| Hyperelasticity | finite-strain mechanics and external comparison | FEniCS, pure JAX, JAX+PETSc, JAX-FEM | terminal-energy agreement, distribution/memory, GAMG/PMG sensitivity |
| Plasticity2D | Davis-B reduced slope-stability endpoint/fixed-work study | JAX+PETSc only | endpoint result, fixed-work diagnostics, reference-continuation appendix |
| Plasticity3D | high-order Mohr-Coulomb endpoint surrogate and solver-policy study | JAX+PETSc; Sysala-family reference-model context | endpoint observables, derivative-route cost, degree-energy, scaling |
| Topology | coupled design-mechanics distributed workflow | pure JAX serial reference, JAX+PETSc parallel run | objective history, timing, rank-consistency check |

## Current Subagent Findings

- Message/front matter audit: the scientific-toolset message is clear; unresolved
  blockers remain target template/declarations, final archive DOI/license,
  durable evidence packaging, and page-budget strategy. SIAM/SIOPT pages were
  not fetchable from this environment because of Cloudflare, so target-template
  details require browser/manual verification.
- Math audit: recent fixes include Plasticity3D elastic-first branch selection,
  return-test notation, generic energy-density notation, KSP target wording,
  Plasticity2D/Plasticity3D surrogate-definition order, `u_{\max}` definition
  before validation thresholds, Plasticity2D trial-function definition order,
  consistent Plasticity3D `s_{\lambda}` notation, and topology notation that
  avoids stress/density and hyperelastic/double-well collisions.
- Evidence/provenance audit: no missing numerical experiment is required for
  the current scoped claims. The local submission-bundle manifest is refreshed
  to paper-cleanup commit `3febc92239be9f0c9fc8f129459377cb5fb9340a`.
  Remaining evidence risks are release-packaging risks: source data split
  between the curated paper bundle and broader repo assets, derived rather than
  full raw Plasticity3D figure data in the local bundle, and final DOI/archive
  integration.
- Citation/comparator audit: recent fixes narrow JAX-FEM wording to a
  hyperelastic companion terminal-state comparison, keep the
  \v{C}ermak--Sysala--Valdman role to elastoplastic implementation context,
  add Plasticity3D endpoint-comparator provenance to the claim audit, add Conn
  trust-region metadata, and narrow Sysala 2017 source-record claims to the
  locally recorded DOI/CAS metadata.
- Layout audit: no hard rendering failures. Current A4 risks are float-loop
  breaks around Plasticity2D and Plasticity3D benchmark figures, dense
  derivative-route and hyperelasticity result pages, early schematic-heavy
  implementation pages, and final conclusion/appendix compression. Major layout
  moves are best paired with a target-template/page-budget decision.
- Final-section synthesis pass: Results now ends with a compact evidence-block
  handoff, Discussion carries the cross-cutting interpretation of derivative
  routes, matched agreement contracts, and solver-policy scope, and Conclusions
  close on the scientific-computing contribution plus next research steps. This
  reduces repeated endpoint-scope and coupled-solver language without changing
  numerical evidence.
- Current audit chunk: tightened Plasticity2D/3D notation and nonsmooth AD
  scope, split long validation/topology prose, made benchmark and
  hyperelastic-memory table labels paper-facing, narrowed Sysala 2017 claim-audit
  evidence to source-record context, added a Karolina hardware audit row, and
  made figure/table asset validators reject stale non-manuscript generated
  outputs.
- Current provenance-refresh chunk: rebuilt the local submission-bundle manifest
  to commit `3febc92239be9f0c9fc8f129459377cb5fb9340a`; manifest-hash,
  archive-neutral asset, submission, release-blocker, diff-whitespace, and
  ignored-style-guide checks pass.
- Current citation/comparator chunk: replaced the PETSc web citation with the
  official PETSc/TAO Users Manual DOI record, narrowed Plasticity3D validation
  wording and figure labels to matched-comparator diagnostics, and recast the
  topology parallel evidence as rank-varied adaptive timing plus a controlled
  rank-consistency check. The current A4 submission gate passes and produces a
  42-page PDF.
- Current message/math/layout chunk: removed draft availability prose that
  called out the absent DOI, put the primary JAX+PETSc implementation path
  before reference paths, defined AD/AD-HVP on first use, added the topology
  volume-multiplier recurrence, normalized equation references/differentials,
  and moved captions above the most fragile long result tables. Release checks
  still flag the unresolved DOI/archive/license/template blockers.

## Figure And Table Rules

- Figure/table evidence must be interpreted in body text, not only captions.
- Figure fonts should match the LaTeX family and be readable at final included
  size. Regenerate through `paper/scripts/generate_paper_figures.py` rather than
  editing final assets directly.
- Tables are generated through `paper/scripts/generate_paper_tables.py` unless a
  source note says otherwise. Preserve relative widths, numeric alignment,
  units in headers, and paper-facing labels.
- Avoid adding hard `[H]` floats. If one is unavoidable, update
  `paper/scripts/check_float_placements.py` with a reason.
- After figure/table changes, run the narrow generator first and then
  `make -C paper submission-check`.

## Claim Safety Rules

- Tie every novelty, validation, timing, scaling, and agreement claim to a
  citation, equation, table, figure, artifact, or explicit assumption.
- Validation evidence and performance evidence stay conceptually separate.
- Direct Plasticity3D claims are endpoint-surrogate claims unless incremental
  plastic-history evidence is explicitly generated.
- Pure JAX, FEniCS, JAX-FEM, and MATLAB references are scoped comparators, not
  universal baselines.
- Do not introduce a new experiment result without command, input, environment,
  output, and generator provenance.

## Current Blockers

Local build and provenance gates pass, but final submission is blocked by
external release decisions:

- target venue/template and required declarations;
- root repository license;
- durable software/artifact archive;
- archival DOI in the availability statement.

The current PDF is a 42-page A4 article. Page budget and float behavior must be
rechecked after target-template conversion.

Deferred layout risks from the 2026-07-09 audit: the hyperelasticity diagnostic
cluster around PDF pages 31--33, the Plasticity3D degree/scaling block around
PDF pages 34--38, dense Table 19, early schematic-heavy implementation pages,
and target-template sensitivity in the benchmark float groups. Stale
non-manuscript generated figure/table outputs are now removed and guarded by
`validate_paper_assets.py`.

## Standard Validation

- `./.venv/bin/python -m py_compile paper/scripts/generate_paper_tables.py paper/scripts/generate_paper_figures.py`
- `make -C paper tables` after table-generator edits
- `make -C paper figures` after figure-generator edits
- `make -C paper submission-check`
- `./.venv/bin/python paper/scripts/check_release_blockers.py --expect-blockers`
- `git check-ignore -v paper/style_guide/README.md`
- `git status --short --ignored paper/style_guide`
