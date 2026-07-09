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

- Message and comparator scope: the scientific-toolset message is clear.
  Comparators remain scoped to internal consistency checks, narrow external
  companion comparisons, or literature context rather than universal baselines.
- Mathematical exposition: recent fixes define Plasticity2D/3D surrogate
  notation before use, split true reduced objectives from frozen design
  subproblems, define topology design/pad node sets, keep topology notation
  separate from mechanics notation, define AD/AD-HVP and distributed ownership
  explicitly, and treat displayed equations as sentence parts.
- Evidence scope: no small missing experiment is required for the current claims.
  Plasticity3D remains endpoint-surrogate evidence; topology reports adaptive
  end-to-end timing plus a controlled rank-consistency check; scalar FEniCS
  curves are reference-implementation context for derivative-route results.
- Provenance: the local submission-bundle manifest is a committed paper-source
  anchor. `paper/scripts/check_submission_bundle_manifest.py` now fails if
  manuscript, generated-table, generated-figure, or bundle-generator paths have
  changed after the manifest's recorded `git_commit`. Refresh the bundle after
  committing such changes. Final archive/DOI integration remains unresolved.
- Layout: the current A4 PDF has no known hard rendering blocker after the last
  completed build. Only the three method algorithms remain intentionally
  hard-pinned with `[H]`; problem-specific Results, Validation, Benchmark, and
  Appendix floats are now broadly flexible. Remaining layout risks are
  target-template driven: dense result pages, figure/table compression, and page
  budget after venue conversion.
- Earlier page-budget/evidence pass: the roadmap and related-work transition
  are shorter, the standalone Armijo pseudocode was removed while preserving the
  solver-policy prose and citations, comparator-role citations were moved into
  the opening scope paragraph, and topology rank-consistency wording now uses a
  one-rank reference solve. That pass rebuilt a 44-page A4 article.
- Earlier thematic polish pass: Results now opens with a shorter protocol
  transition, generated protocol-table cells use paper-facing wording, Methods
  defines the Armijo, bounded-merit, and residual-bisection line-search
  acceptance policies, Implementation includes an owned-row residual/matrix
  assembly display, and Discussion/Conclusion synthesize rather than re-list
  exact result numbers. No small missing experiment is required for current
  claims; path-history plasticity validation remains future work.
- Earlier last-mile pass: rendered/source scans found no high-priority
  paper-facing leaks. Recent fixes expand hyperelasticity scaling acronyms,
  sharpen residual-bisection and owned-row assembly notation, standardize nearby
  `\lambda_{\mathrm{sr}}` numeric styling, and clarify Plasticity3D validation
  table status text. That pass rebuilt a 44-page A4 article.
- Earlier paper-facing tone/notation pass: subagent audits found no hard local
  path or campaign-tag leaks. Integrated fixes define the owned-row assembly
  residual/tangent symbols before use, define
  `\varepsilon_{eq}(u_e)=B_{eq}u_e` in the Plasticity3D surrogate, name the
  matched comparator as an independent reference-formula assembly path, replace
  residual run-log wording with configuration/comparison language, and narrow the
  conclusion to the tested nonlinear energy families. The rebuilt PDF is now a
  45-page A4 article.
- Earlier float-concision and notation pass: subagent audits reduced the hard
  float allowlist from 12 problem/method floats to the three method algorithms,
  shortened dense Results and Conclusion prose without changing evidence, added
  the Plasticity2D deviatoric engineering-strain invariant, clarified frozen
  design-state notation, cleaned generated figure/table wording, and narrowed
  topology rank-consistency language to observed discrepancies in the controlled
  40-iteration case. The rebuilt PDF remains a 45-page A4 article; rendered
  pages 16--19 and 25--42 were visually checked.
- Current message/math/evidence/layout pass: subagent audits tightened the
  abstract and contribution thesis, added a motivation bridge before the
  related-work role table, separated Sysala-family reference-model context from
  \v{C}ermak--Sysala--Valdman implementation context, split reduced and frozen
  objective notation, replaced visible colored-SFD wording with colored
  sparse-recovery / AD-HVP wording, standardized reference-formula and
  frozen-preconditioner PMG terminology, added a discrete-norm definition for
  validation metrics, renamed the Plasticity2D local stress vector to
  `d_\sigma`, narrowed Plasticity3D validation wording to the
  highest-successful-load value on the tested grid, and relaxed layout barriers
  that isolated benchmark floats. The rebuilt PDF is a 43-page A4 article;
  rendered pages 8--23, 26--38, and page 34 were visually checked.
- Current SIOPT-scope and self-containment pass: Bohr/Plato/Euler/Hypatia
  audits found no current A4 layout blocker and no small missing experiment for
  the present claims. Integrated fixes narrow the abstract and related-work
  novelty language to the tested suite, make the conclusion synthesize rather
  than inventory sections, define the Plasticity2D repeated-principal-stress
  direction convention, specify the topology multiplier quantile and
  correction initialization, define topology design/state-change diagnostics,
  correct the Ginzburg--Landau energy-evidence attribution, and keep the
  benchmark-specification table paper-facing. Remaining layout risks are
  target-template risks: dense Results table clusters, the Hyperelasticity
  PMG/memory block, Plasticity3D degree/scaling figures and tables, and
  FloatBarrier behavior after venue conversion.
- Current reporting-terminology and notation pass: Euclid/Huygens/Bacon/Popper
  audits found no must-fix message blocker, no small missing experiment, and no
  current A4 layout blocker. Integrated fixes clarify comparator roles in the
  abstract, include bridge architectures in the introduction positioning,
  replace row/run phrasing with evidence-facing entries/configurations,
  standardize Plasticity3D fixed-`\lambda_{\mathrm{sr}}` and strength-reduction
  terminology, define the boundary normal and quadrature-count notation, make
  the topology frozen local-compliance expression elementwise, define the
  hyperelastic centerline and Plasticity3D boundary-profile samples used in
  validation metrics, and correct p-Laplace, Ginzburg--Landau, and
  Plasticity3D evidence attribution. The current A4 layout remains clean; dense
  Results tables, the Hyperelasticity PMG/memory cluster, Plasticity3D
  degree/scaling material, appendix Table 30, and FloatBarrier behavior remain
  target-template risks only.
- Current self-containment and hygiene pass: Wegener/Galileo/Boole/Ptolemy
  audits found no message/citation must-fix and no current A4 layout blocker.
  Integrated fixes replace remaining reader-facing terminal-state and row/table
  mechanics wording with endpoint-state, entry, case, or configuration wording,
  including generated hyperelastic figure labels; make the validation
  curve/profile sample sets reproducible in prose; rewrite the owned-row
  assembly display with an element-restricted state; state the Plasticity2D
  in-plane stress pair behind the branch return and the Plasticity3D
  bottom-clamped test space; make the Plasticity3D CPU scaling table narrower
  by dropping redundant solve-time and absolute stopping-gradient columns; tune
  the Plasticity3D 3D tick labels and regenerate the hyperelastic state figure
  near its final include width; cite the
  Sysala-family and \v{C}ermak--Sysala--Valdman reference line at the
  Plasticity3D path-history limitation; narrow the conclusion's validation
  discipline to the tested nonlinear FEM energy benchmarks; and remove the
  abstract's broad "best evaluated" phrasing. Current A4 layout remains clean;
  target-template risks now additionally include benchmark Tables 4 and 6, the
  compact validation page, the topology results page, and the previously noted
  dense Results/Hyperelasticity/Plasticity3D/appendix clusters.
- Current thesis, evidence-scope, notation, and layout pass:
  Pascal/Hume/Parfit/Dewey audits found no hard A4 blocker but identified
  publish-readiness polish issues. Integrated fixes make the central thesis
  consistent: derivative construction is evaluated inside the sparse nonlinear
  solve that consumes it. The abstract now delays comparator caveats until after
  the method/evidence arc, the introduction names the three comparison roles
  (internal checks, endpoint companion comparisons, reference-model or
  implementation context), related work states the missing combined comparison
  contract, and the conclusion closes with evidence anchors rather than a
  separate validation-discipline thesis. Evidence-scope fixes remove the
  overbroad active-free-DOF claim from the direct Plasticity3D branch diagnostic,
  keep exact active-DOF matching only for the fixed-`\lambda_{\mathrm{sr}}`
  diagnostic, rename the validation "boundary profile" to an upper-slope
  coordinate profile, broaden the benchmark reported-scope table to include
  later L10/L5 diagnostics, and state that the topology rank-consistency check is
  smaller and fixed-schedule. Mathematical-polish fixes define KSP/PMG/MUMPS/Hypre
  before use in Methodology, change assembled residual notation to
  `\mathcal{R}_i` to avoid collision with element restrictions, clarify the
  2D plasticity in-plane deviatoric visualization norm, replace Ginzburg--Landau
  "branch selection" with selected stationary basin, soften topology
  connectedness to a displayed density pattern, and replace appendix "low-rank"
  with small MPI-rank-count. Layout polish shortens the JAX-FEM subcaption that
  hyphenated visibly. `paper/scripts/generate_paper_tables.py` is the source for
  generated table edits; a full rerun requires the tracked submission-bundle
  inputs, so do not delete those bundle files during table work.
- Final local cleanup pass:
  Planck/Nash/Boyle/Lorentz audits found no new experiment requirement but did
  identify remaining publish-readiness polish. Integrated fixes replace visible
  `Plasticity2D`/`Plasticity3D` manuscript labels with 2D/3D Mohr--Coulomb or
  3D plasticity wording, shorten dense Results interpretation blocks, introduce
  the constrained discrete plasticity spaces before the surrogate energies,
  define topology `\nu`, make the topology mechanics trial space homogeneous,
  define the stopping-gradient ratio before use, and align validation norm
  wording with the actual unweighted Euclidean curve comparisons. Generated
  table headers now use `Reported time [s]` consistently, the stale hard-float
  unit test now matches the current allowlist of three method algorithms, and
  `make -C paper submission-bundle` documents the bundle-refresh step. Figure
  annotations for the hyperelastic centerline and 3D plasticity `u_{\max}` curve
  now use `rel. Eucl.` to match the prose. The submission bundle was rebuilt and
  the manifest verifies; it records the pre-commit source-control anchor
  `8b8a6c9b8ce0dd06f9dd2099f18475dc88d284e8` plus current file hashes. Rendered
  checks of pages 25 and 27 and the late-results pages 31--38 found no clipping
  or unreadable updated labels.
- Final provenance/math/layout gate pass:
  Poincare found no message/comparator issue. Locke found local mathematical
  polish items now fixed: validation defines the two `u_{\max}` curve sample
  vectors, topology uses the frozen-objective notation
  `\mathcal{J}_h^m(z_h;u_h^m)`, Results avoids calling a converged comparison
  a fixed-work diagnostic, and the $p$-Laplace Results reference now points to
  the discrete problem. Meitner found a fragile `0.0` tick on the 3D plasticity
  validation colorbar; the figure generator adds left padding and the figure/PDF
  were regenerated. Ampere found the stale bundle-manifest anchor; the manifest
  checker now enforces freshness for paper bundle-refresh paths, with unit-test
  coverage for stale commits.

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

The current PDF is a 43-page A4 article. Page budget and float behavior must be
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

- `./.venv/bin/python -m py_compile paper/scripts/check_submission_bundle_manifest.py paper/scripts/generate_paper_tables.py paper/scripts/generate_paper_figures.py paper/scripts/build_submission_bundle.py`
- `make -C paper tables` after table-generator edits
- `make -C paper figures` after figure-generator edits
- commit manuscript/generated-paper/provenance-checker changes
- `make -C paper submission-bundle` after manuscript, generated-table,
  generated-figure, or provenance-source changes
- `make -C paper submission-check`
- `./.venv/bin/python paper/scripts/check_release_blockers.py --expect-blockers`
- `git check-ignore -v paper/style_guide/README.md`
- `git status --short --ignored paper/style_guide`
