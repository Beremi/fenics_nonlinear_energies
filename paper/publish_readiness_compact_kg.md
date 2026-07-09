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

- pure JAX: serial formulation checks where matched compact formulations exist;
  topology is a serial design demonstration, not a validation baseline;
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
- Recent SIOPT-style cues checked on 2026-07-09 are method-first and
  evidence-scoped: application motivation, assumptions/problem class, numerical
  method, experiment parameters, numerical evidence, and discussion/conclusion.
  The SIAM DOI page for Sysala--Beres--Beresova--Haslinger--Kruzik--Luber
  (2025) is accessible from this environment and confirms the open-access
  SIOPT structure; secondary open anchors remain the arXiv full text of
  Keith--Kim--Lazarov--Surowiec, "Analysis of the SiMPL method for
  density-based topology optimization" (current arXiv version dated
  2025-02-23), and an open Optimization Online PDF for a SIOPT-style
  derivative-free optimization manuscript.

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
| Topology | coupled design-mechanics distributed workflow | pure JAX serial demonstration, JAX+PETSc parallel run | objective history, timing, rank-consistency check |

## Recent Audit State

- Message and comparator scope: the scientific-toolset message is clear.
  Comparators remain scoped to internal consistency checks, narrow external
  companion comparisons, or literature context rather than universal baselines.
- Mathematical exposition: recent fixes define Plasticity2D/3D surrogate
  notation before use, keep topology notation separate from mechanics notation,
  define AD/AD-HVP and distributed ownership explicitly, and treat displayed
  equations as sentence parts.
- Evidence scope: no small missing experiment is required for the current claims.
  Plasticity3D remains endpoint-surrogate evidence; topology reports adaptive
  end-to-end timing plus a controlled rank-consistency check; scalar FEniCS
  curves are reference-implementation context for derivative-route results.
- Provenance: the local submission-bundle manifest records the earlier
  paper-cleanup commit `3febc92239be9f0c9fc8f129459377cb5fb9340a`; treat this
  as bundle provenance, not as a claim that the manifest reflects the latest
  manuscript HEAD. Final archive/DOI integration remains unresolved.
- Layout: the current A4 PDF has no known hard rendering blocker after the last
  completed build. Only the three method algorithms remain intentionally
  hard-pinned with `[H]`; problem-specific Results, Validation, Benchmark, and
  Appendix floats are now broadly flexible. Remaining layout risks are
  target-template driven: dense result pages, figure/table compression, and page
  budget after venue conversion.
- Current page-budget/evidence pass: the roadmap and related-work transition
  are shorter, the standalone Armijo pseudocode was removed while preserving the
  solver-policy prose and citations, comparator-role citations were moved into
  the opening scope paragraph, and topology rank-consistency wording now uses a
  one-rank reference run. The rebuilt PDF is a 44-page A4 article.
- Current thematic polish pass: Results now opens with a shorter protocol
  transition, generated protocol-table cells use paper-facing wording, Methods
  defines the Armijo, bounded-merit, and residual-bisection line-search
  acceptance policies, Implementation includes an owned-row residual/matrix
  assembly display, and Discussion/Conclusion synthesize rather than re-list
  exact result numbers. No small missing experiment is required for current
  claims; path-history plasticity validation remains future work.
- Current last-mile pass: rendered/source scans found no high-priority
  paper-facing leaks. Recent fixes expand hyperelasticity scaling acronyms,
  sharpen residual-bisection and owned-row assembly notation, standardize nearby
  `\lambda_{\mathrm{sr}}` numeric styling, and clarify Plasticity3D validation
  table status text. The rebuilt PDF remains a 44-page A4 article.
- Current paper-facing tone/notation pass: subagent audits found no hard local
  path or campaign-tag leaks. Integrated fixes define the owned-row assembly
  residual/tangent symbols before use, define
  `\varepsilon_{eq}(u_e)=B_{eq}u_e` in the Plasticity3D surrogate, name the
  matched comparator as an independent reference-formula assembly path, replace
  residual run-log wording with configuration/comparison language, and narrow the
  conclusion to the tested nonlinear energy families. The rebuilt PDF is now a
  45-page A4 article.
- Current float-concision and notation pass: subagent audits reduced the hard
  float allowlist from 12 problem/method floats to the three method algorithms,
  shortened dense Results and Conclusion prose without changing evidence, added
  the Plasticity2D deviatoric engineering-strain invariant, clarified frozen
  design-state notation, cleaned generated figure/table wording, and narrowed
  topology rank-consistency language to observed discrepancies in the controlled
  40-iteration case. The rebuilt PDF remains a 45-page A4 article; rendered
  pages 16--19 and 25--42 were visually checked.

## Figure And Table Rules

- Figure/table evidence must be interpreted in body text, not only captions.
- Figure fonts should match the LaTeX family and be readable at final included
  size. Regenerate through `paper/scripts/generate_paper_figures.py` rather than
  editing final assets directly.
- Tables are generated through `paper/scripts/generate_paper_tables.py` unless a
  source note says otherwise. Preserve relative widths, numeric alignment,
  units in headers, and publication labels.
- Avoid adding hard `[H]` floats. If one is unavoidable, update
  `paper/scripts/check_float_placements.py` with a reason. The current
  allowlist contains only the three methodology algorithms.
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

The current PDF is a 45-page A4 article. Page budget and float behavior must be
rechecked after target-template conversion.

Deferred layout risks from the 2026-07-09 audits are now target-template risks
rather than current A4 blockers: dense hyperelasticity and Plasticity3D result
clusters, long result tables, the four-panel Plasticity3D degree figure, and
the Plasticity3D CPU-scaling table. Latest rendered-page review found no A4
clipping or illegible figures; revisit these risks only after a target template
changes text width or the paper is split into main/supplement material. Stale
non-manuscript generated figure/table outputs are removed and guarded by
`validate_paper_assets.py`; figure sizing now fails fast if the current A4
article layout contract changes without updating the measurement policy.

## Standard Validation

- `./.venv/bin/python -m py_compile paper/scripts/generate_paper_tables.py paper/scripts/generate_paper_figures.py`
- `make -C paper tables` after table-generator edits
- `make -C paper figures` after figure-generator edits
- `make -C paper submission-check`
- `./.venv/bin/python paper/scripts/check_release_blockers.py --expect-blockers`
- `git check-ignore -v paper/style_guide/README.md`
- `git status --short --ignored paper/style_guide`
