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
  blockers remain target template/declarations, final archive DOI/license, and
  page-budget strategy.
- Math audit: fixed in current chunk by adding Plasticity3D elastic-first branch
  selection, separating return-test ratios from material weight and engineering
  shear, replacing generic `\psi_q` energy-density notation with
  `\mathcal{W}_q`, making the KSP target column explicit, and moving the
  Plasticity2D/Plasticity3D discrete surrogate definitions after their local
  Davis-B and branch-potential definitions.
- Evidence audit: fixed in current chunk by interpreting the p-Laplace,
  Plasticity2D, hyperelastic validation, and Plasticity3D validation figures in
  body text.
- Citation audit: fixed in current chunk by narrowing the
  \v{C}ermak--Sysala--Valdman role to elastoplastic implementation context and
  making JAX-native / bridge rows source-specific in the SOTA table.
- Layout audit: no hard rendering failures; defer target-template-sensitive
  float moves for dense hyperelasticity and Plasticity3D result pages until the
  venue/template split is known.

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

The current PDF is a 44-page A4 article. Page budget and float behavior must be
rechecked after target-template conversion.

Deferred layout risks from the 2026-07-09 audit: the hyperelasticity diagnostic
cluster around PDF page 31, the Plasticity3D degree/scaling block around PDF
page 34, dense Table 19, early schematic-heavy implementation pages, and unused
generated assets that should be excluded or cleaned before the final submission
bundle.

## Standard Validation

- `./.venv/bin/python -m py_compile paper/scripts/generate_paper_tables.py paper/scripts/generate_paper_figures.py`
- `make -C paper tables` after table-generator edits
- `make -C paper figures` after figure-generator edits
- `make -C paper submission-check`
- `./.venv/bin/python paper/scripts/check_release_blockers.py --expect-blockers`
- `git check-ignore -v paper/style_guide/README.md`
- `git status --short --ignored paper/style_guide`
