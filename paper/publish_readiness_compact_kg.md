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
- FEniCS: reference-formulation checks for scalar and hyperelastic families;
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
- Additional 2025 SIOPT metadata and open arXiv mirrors checked on
  2026-07-09 reinforce the same pattern: abstracts state the problem class and
  method directly, qualify assumptions or stationarity notions, and name the
  numerical or theoretical evidence. Concrete anchors: "Consistency of
  sample-based stationary points for infinite-dimensional stochastic
  optimization" (SIAM J. Optim. DOI `10.1137/23M1600608`, arXiv
  `2306.17032`), "Splitting the Conditional Gradient Algorithm" (DOI
  `10.1137/24M1638008`, arXiv `2311.05381`), "On Squared-Variable
  Formulations" (DOI `10.1137/23M1608343`, arXiv `2310.01784`), and
  "TS-RSR: A provably efficient approach for batch Bayesian Optimization" (DOI
  `10.1137/24M1675102`, arXiv `2403.04764`).

The local style-guide snapshot is ignored through `.git/info/exclude`; do not
stage or commit `paper/style_guide/`.

Local `archive_neutral` figure/table manifest checks mean that submitted assets
point to the curated local submission bundle rather than raw run paths. They do
not resolve the final durable archive/DOI blocker.

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
- `appendix.tex`: keep supporting solver-policy diagnostics, reference-formula
  branch-tangent comparisons, and frozen-preconditioner PMG diagnostics out of
  the main evidence line.

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

- Current message: the manuscript now presents a scientific nonlinear FEM
  toolset plus numerical expertise, not a codebase report. The mainline is the
  JAX+PETSc realization; pure JAX, FEniCS, JAX-FEM, Sysala-family papers, and
  the \v{C}ermak--Sysala--Valdman MATLAB work are scoped comparators or
  context, not universal baselines.
- Current mathematical state: recent passes define the p-Laplace assumptions and
  discrete space before use, separate plasticity source-model notation from
  endpoint surrogates, align plasticity work with free coefficient-vector
  notation, state homogeneous endpoint displacement data, use element index `e`
  in distributed assembly to avoid conflict with bulk modulus `K`, define
  topology diagnostics before the stopping rule, reference the effective-volume
  multiplier explicitly, and reserve `p_{\mathrm{SIMP}}` for the topology
  continuation exponent.
- Current evidence state: subagent evidence audits found no small missing
  experiment needed for the current evidence surface. Wording is scoped so
  p-Laplace energy agreement is plotted-scale across levels and about
  `1e-8` only on the selected table case, Ginzburg--Landau globalization reports
  matching energy and Newton/Krylov work, Plasticity2D state wording is visual
  rather than externally validated, Plasticity3D remains endpoint-surrogate
  evidence, and topology keeps relying on the JAX+PETSc rank-consistency check.
- Current layout state: the latest full build is a 46-page A4 article with a
  clean LaTeX warning scan, embedded Type 1 fonts, clean aux-order and hard-float
  checks, and `qpdf --check` passing. Polish/risk items remain target-template
  dependent: an under-filled page after the Plasticity3D state figure, dense
  protocol and late-results tables, and full-width multi-panel figures that
  should not be shrunk blindly.
- Current provenance rule: manuscript, generated-table, generated-figure, or
  bundle-generator edits make the submission-bundle manifest stale until those
  edits are committed, `make -C paper submission-bundle` is rerun, and the
  refreshed manifest is committed. Final archive/DOI/license/template decisions
  remain external blockers.
- Current generator state: the Plasticity3D validation table summary is
  data-driven rather than hard-coded. Regenerate generated tables after any
  table-source or table-generator edit.
- Prior completed passes: older audits already removed process-local wording,
  tightened comparator scope, reduced hard floats to the three method
  algorithms, clarified validation norms and topology diagnostics, refreshed
  generated figures/tables, added literature and bundle freshness gates, and
  rebuilt clean 44--45 page A4 PDFs. Use git history or the long audit log for
  details; do not re-expand this compact section.

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
- After figure/table changes, run the narrow generator first, refresh the local
  bundle with `make -C paper submission-bundle`, and then run
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

The current PDF is a 46-page A4 article. Page budget and float behavior must be
rechecked after target-template conversion.

Deferred layout risks from the 2026-07-09 audits are now target-template risks
rather than current A4 blockers: dense hyperelasticity and Plasticity3D result
clusters, long result tables, the four-panel Plasticity3D degree figure, and
the Plasticity3D CPU-scaling table. Latest rendered-page review found no A4
clipping, isolated float page, or illegible figures after barrier relaxation;
revisit these risks only after a target template changes text width or the paper
is split into main/supplement material. Stale non-manuscript generated
figure/table outputs are removed and guarded by `validate_paper_assets.py`;
figure sizing now fails fast if the current A4 article layout contract changes
without updating the measurement policy.

## Standard Validation

- `./.venv/bin/python -m py_compile paper/scripts/check_submission_bundle_manifest.py paper/scripts/generate_literature_sources.py paper/scripts/generate_paper_tables.py paper/scripts/generate_paper_figures.py paper/scripts/build_submission_bundle.py`
- `make -C paper tables` after table-generator edits
- `make -C paper figures` after figure-generator edits
- `make -C paper literature` after bibliography or literature-manifest edits
- `make -C paper literature-check`
- commit manuscript/generated-paper/provenance-checker changes
- `make -C paper submission-bundle` after manuscript, generated-table,
  generated-figure, bibliography, literature-source, or provenance-source changes
- `make -C paper submission-check`
- `./.venv/bin/python paper/scripts/check_release_blockers.py --expect-blockers`
- `git check-ignore -v paper/style_guide/README.md`
- `git status --short --ignored paper/style_guide`
