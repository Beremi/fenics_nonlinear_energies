# Publish-Readiness Knowledge Graph

This note is the working map for the paper-readiness branch. It is not part of
the manuscript. It records the intended message, evidence nodes, reviewer risks,
and subagent findings so later edits stay coordinated.

## Target Message

The paper should read as a scientific methods-and-software article for nonlinear
finite-element energy problems. The central object is a JAX+PETSc solver stack
that couples local automatic differentiation, sparse distributed assembly,
nonlinear globalization, Krylov linear solvers, and preconditioner policy. The
comparison surface includes pure JAX and FEniCS reference paths where available,
external or source-family comparators where the contract is narrow, and legacy
MATLAB/Octave literature only when there is source-backed evidence.

## External Style Anchors

- Local style guide: `paper/style_guide/AGENTS.md` and
  `paper/style_guide/style_fingerprint/agent_quick_reference.md`.
  Apply the 2025 accepted KL-paper style: formal applied mathematics,
  motivation before technical construction, authorial `we`, definitions before
  dense notation, punctuated displays, and evidence-scoped claims.
- Local conflict rule: preserve this manuscript's existing `cleveref`/`\Cref`
  convention while applying the style guide's prose and claim discipline.
- Recent SIAM optimization style cue:
  Doikov, Mishchenko, and Nesterov, "Super-Universal Regularized Newton
  Method," SIAM Journal on Optimization, 2024, DOI `10.1137/22M1519444`.
  Useful structure: motivation, notation/problem classes, method, theory,
  numerical experiments, discussion. The article is open access under CC BY 4.0.
- Recent applied SIOPT style cue:
  Sysala, Béreš, Bérešová, Haslinger, Kružík, and Luber, "Convex Optimization
  Problems Inspired by Geotechnical Stability Analysis," SIAM Journal on
  Optimization, 2025, DOI `10.1137/25M1723177`. Useful structure:
  application motivation, abstract assumptions, interpretation remarks, and a
  numerical 3D slope-stability example after the mathematical framework.
- Additional SIOPT style cue:
  Grapiglia and Nesterov, "Adaptive Third-Order Methods for Composite Convex
  Optimization," SIAM Journal on Optimization, 2023, DOI `10.1137/22M1480872`.
  Useful structure: problem statement, method variants, numerical experiments,
  discussion, and conclusion, with numerical evidence interpreted after the
  algorithmic contract is clear.
- SIOPT journal scope cue: SIAM describes SIOPT as covering theory and practice
  of optimization; contributions may emphasize algorithms, software,
  computational practice, applications, or links between these subjects.

## Manuscript Nodes

- `PAPER`: Title, abstract, and conclusion.
  Status: title, abstract, and conclusion are now framed around a scientific
  \jaxpetsc{} toolset for nonlinear FEM energy solves, derivative routes,
  sparse assembly, and solver policy. The abstract now opens from the nonlinear
  FEM bottleneck, avoids internal campaign terms such as locked, promoted, and
  fairness-gated, and no longer attributes the historical lambda=1.0
  Plasticity3D speedup to the converged lambda=1.55 scaling sweep. Current
  prose chunk removes the remaining `\repo{}`-centered framing from the
  abstract, introduction, related work, discussion, and conclusion, so the
  paper-facing object is the computational toolset rather than the repository
  name. Current chunk removes remaining platform-local labels from the
  Plasticity3D evidence narrative, replacing workstation/Karolina wording with
  single-node and multi-node CPU wording, and uses auxiliary rather than
  historical for timing-only Plasticity3D evidence.
- `INTRO`: positioning and contributions.
  Status: now opens with the nonlinear FEM computational bottleneck and includes
  an explicit contribution block with solver, derivative/assembly, and
  linear-solver pillars.
- `RELATED`: literature framing.
  Status: consolidated defensive taxonomy into fewer scientific groups:
  FEM automation, differentiable FEM/AD, nonlinear solver infrastructure,
  topology/plasticity source context, and scalable software comparators.
  FEniTop is now framed as topology-optimization literature context rather than
  a direct baseline, and the Cermak--Sysala--Valdman MATLAB/Octave citation is
  implementation-lineage context only.
- `METHOD`: mathematical and algorithmic core.
  Current assets: common finite-element energy notation, derivative routes,
  globalization algorithms, colored sparse finite differences, constitutive AD.
  Status: generic algorithms now use the merit functional `\mathcal{F}` rather
  than the hyperelasticity-reserved symbol `J`; prose is more paper-facing.
- `IMPLEMENTATION`: realization of the method.
  Current assets: pure JAX, FEniCS, and JAX+PETSc strata; autodiff modes;
  Krylov/preconditioner matrix; distributed assembly.
  Status: revised as solver paths with scientific roles, not codebase strata;
  implementation prose now uses mainline/scalable/distributed terminology
  instead of maintained or production-path wording. Plasticity3D PMG wording now
  separates Hypre-coarse auxiliary/endpoint probes from the MUMPS-backed
  redundant-LU profile used by the P2 globalization/derivative rows and the
  converged P4(L2), lambda=1.55 scaling run. Current chunk also separates
  Hyperelasticity's GAMG profile from the larger PMG/MUMPS fixed-work profile
  and renames the capability-matrix feature column so solver policy is not
  mislabeled as AD only. Paper-facing Plasticity3D PMG prose now uses
  reference-formula and reference-operator labels instead of source-assembly or
  source-operator labels, while preserving the scientific comparison contract.
- `BENCHMARKS`: mathematical problem coverage.
  Families: p-Laplace, Ginzburg--Landau, Hyperelasticity, Plasticity2D,
  Plasticity3D, Topology.
  Status: repeated "repository specialization" language has been replaced with
  implemented benchmark, discrete model, endpoint surrogate, or source-family
  context as appropriate; generated tables no longer use repository/maintained
  descriptors for paper-facing claims. The Ginzburg--Landau initial state, the
  24-step hyperelastic rotating boundary map, and the topology continuation
  stall rule are now stated in the benchmark text. The Plasticity3D branch
  thresholds, return denominators, branch multipliers, and branch energies are
  now defined explicitly in the benchmark section instead of being left implicit.
  The benchmark specification matrix now labels problem-specific stopping and
  work definitions as a solve contract and uses configuration-specific rather
  than campaign-specific wording for Plasticity3D.
- `VALIDATION`: external and source-family comparison.
  Evidence: narrow hyperelastic JAX-FEM comparison; Plasticity3D validation
  ladder with endpoint-surrogate scope.
  Rule: keep validation separate from performance and never imply
  path-consistent plastic-history equivalence.
  Status: comparison metrics now state thresholds and the hyperelastic companion
  schedule. The JAX-FEM gate no longer asserts constitutive-law identity; it
  records matched mesh/schedule and terminal post-comparison under the paper
  energy. Plasticity3D Layer 1A is phrased as direct-branch source-observable
  agreement, while glued-bottom boundary-contract language is reserved for
  Layer 2. Current chunk writes strength-reduction evidence as
  `\lambda_{\mathrm{sr}}` in prose, captions, and generated tables, and
  describes the fixed-operator diagnostic as a reference-operator diagnostic.
- `RESULTS`: performance and solver behavior.
  Evidence: globalization comparison, derivative-route comparison, scaling
  studies, hyperelastic and Plasticity3D solver diagnostics, topology scaling.
  Status: a synthesis subsection now states what the result blocks establish:
  nonlinear policy is problem-dependent, multiple derivative routes are useful,
  sparse ownership/preconditioning are part of the numerical method, and the
  \jaxpetsc{} mainline remains the scalable path while serial/external
  formulations serve scoped reference roles. Plasticity3D result prose now marks
  the fixed P4(L1), lambda=1.5 derivative ablation as Hypre-backed rank-local
  timing evidence and separates it from the MUMPS-backed PMG convergence/scaling
  evidence at lambda=1.55. The converged P4(L2), lambda=1.55 scaling table now
  includes Newton iterations and the gradient-to-target ratio, so the stop
  contract is visible next to the final gradient values. Current chunk defines
  the Plasticity3D derivative-ablation observables `\omega` and `u_{\max}`,
  states the lambda=1.55 linear tolerance as the PETSc relative KSP residual
  tolerance, and marks the Hyperelasticity PMG table as fixed-work
  solver-policy evidence rather than an endpoint-accuracy ranking. Current
  chunk scopes the Plasticity3D constitutive-AD comparison to the fixed
  high-order derivative-route test and describes it as the lowest measured wall
  time under matched terminal observables, not a general preference. The
  single-node/multi-node CPU scaling evidence now has matching generated table
  labels, figure legend labels, and body-text interpretation.
- `DISCUSSION_CONCLUSION`: interpretation and scope.
  Current prose states the toolset lesson in paper-facing terms: automatic
  differentiation alone does not determine large nonlinear FEM behavior;
  globalization, sparse ownership, overlap replication, coarse-level policy,
  preconditioning, and surrogate-model interpretation are part of the numerical
  method. Current chunk reframes caveat-led language as scope of evidence and
  keeps the conclusion tied to fixed-test evidence rather than broad best-path
  claims.
- `FIGURES_TABLES`: visual and layout quality.
  Current fact: `paper/build/main.pdf` is 37 pages, A4, 10 pt article,
  text width about 7.09 in. Many floats use `[H]`, so placement is highly manual.
  Figure/layout subagent audit: assets are technically clean at current A4
  width, with embedded fonts and 600 ppi raster layers, but dense 3D
  multi-panel Plasticity3D figures and wide tables are not robust to a narrower
  journal class. High-value next step: split or simplify the worst Plasticity3D
  figures/tables before final template submission.
  Layout chunk: the Plasticity3D validation ladder now uses stacked medium-width
  panels instead of cramped side-by-side subfigures. Rendered pages 20--21 show
  readable panel titles, colorbars, axes, captions, and a clean transition to
  the validation summary table and results section.
  Layout chunk: the globalization comparison table uses wrapped benchmark and
  method columns. Rendered page 22 no longer shows the earlier benchmark/method
  label collision.
  Current chunk: the Plasticity3D lambda=1.55 scaling table uses scriptsize and
  adds Newton and gradient-target columns. Rendered pages 29--30 remain within
  the text block, with no clipping or table overlap.
  Current chunk: paper-facing generated table files and labels no longer use
  `reviewer_*` names; they were renamed to scientific names for fixed-budget
  Ginzburg--Landau globalization, Hyperelasticity distribution/PMG evidence,
  Plasticity3D derivative-degree evidence, and topology rank consistency. The
  Plasticity3D convergence figure was regenerated taller with more legend
  clearance after the layout subagent found bottom-label crowding on page 17.
  Layout chunk: the widest generated diagnostics now use stacked `tabularx`
  blocks with wrapped text columns rather than single wide `tabular*` layouts:
  globalization method comparison, derivative-route comparison,
  Hyperelasticity distribution/memory, Hyperelasticity PMG sensitivity,
  Plasticity3D derivative-degree evidence, Plasticity3D derivative ablation,
  and the appendix fixed reference-operator PMG table. Rendered pages 23--28 and
  33 show the split tables within the text block, attached to their captions,
  and free of table/figure overlap; the PDF grows from 36 to 37 pages. Current
  chunk visually checked rendered pages 9, 30, and 33 after rebuild: the solve
  contract/reference-availability tables, single-node/multi-node scaling
  figure, and reference-formula appendix tables are readable and within the
  text block.
- `REPRO`: reproducibility and submission readiness.
  Known blockers from `paper/todo.md`: target journal/template/declarations,
  repository license/archive DOI, and archive-neutral provenance for critical
  artifacts. The JAX-FEM baseline runner now writes strict JSON with `null`
  warmup timings and `allow_nan=False`; ignored local baseline metadata was
  corrected in the workspace, but archive-neutral submission bundles remain
  outstanding. Current subagent audit confirmed the same remaining blockers:
  target venue/template and declarations, repository license plus archival DOI,
  and archive-neutral provenance for paper-critical artifacts. Current chunk
  restored source/generated consistency for all edited generated tables and
  the single changed Plasticity3D scaling figure.

## Evidence Nodes

- `E1`: Derivative routes.
  Element AD, constitutive AD, and colored SFD are already described and
  compared. Strongest result: constitutive AD is fastest on the fixed
  high-order Plasticity3D derivative-route case while preserving terminal-state
  agreement.
- `E2`: Nonlinear globalization.
  Tables compare line search, trust region, and hybrid policies. The synthesis
  now scopes trust-region conclusions to the nonconvex Ginzburg--Landau probes
  and the stricter P2(L1) Plasticity3D globalization probe, while Armijo Newton
  remains the converged P4(L2), lambda=1.55 scaling path.
- `E3`: Linear solvers/preconditioners.
  PETSc Krylov, Hypre, GAMG, PMG, MUMPS/redundant coarse solves appear across
  implementation and results. Plasticity3D now distinguishes Hypre-coarse and
  MUMPS-backed PMG profiles in implementation and results; the broader
  linear-solver narrative now states that preconditioner policy, coarse solves,
  and fixed-work diagnostics are part of the numerical method rather than
  implementation detail.
- `E4`: Parallel assembly.
  PETSc ownership, ghost/overlap layouts, owned-row sparse insertion, and
  rank-local assembly are present. Current discussion and Plasticity3D
  partitioning prose connect ownership and overlap replication to the scaling
  interpretation.
- `E5`: Comparator surface.
  Pure JAX and FEniCS references exist where implemented; JAX-FEM exists for
  hyperelasticity; Plasticity3D reference-formula/reference-operator
  comparisons are narrow; topology literature includes MATLAB references but no
  current direct MATLAB runtime comparison is yet established.
- `E6`: Plasticity3D scope.
  Direct-branch source-observable agreement and fixed-load reference-operator
  diagnostics support the endpoint surrogate, but there is no true
  path-consistent incremental-history validation. This boundary must remain.

## Subagent Threads

- `Narrative/structure` (`Meitner`): completed. Key finding: restructure the
  paper around a nonlinear FEM toolset and reduce repository-internal framing.
- `Evidence/experiments/repro` (`Linnaeus`): completed. Key finding: strong
  evidence exists for scalar parity/scaling, the narrow hyperelastic JAX-FEM
  comparison, Plasticity3D endpoint-surrogate validation, derivative-route
  comparisons, and topology consistency; archive-neutral provenance and true
  path-history Plasticity3D validation remain blockers for stronger claims.
- `Figures/layout/PDF` (`Archimedes`): completed. Key finding: current PDF is
  technically clean but not robust to template narrowing; dense tables and
  Plasticity3D multi-panel figures need a dedicated layout chunk.
- `Math/notation/self-contained prose` (`Harvey`): completed. Key finding:
  avoid generic `J`, define comparison metrics, demote branch details that are
  not fully enumerated, and replace implementation spellings with scientific
  notation.
- `Legacy MATLAB/Valdman comparators` (`Halley`): completed. Key finding:
  cite Cermak--Sysala--Valdman 2019 as implementation-lineage context only, not
  as a verified numerical baseline.
- `Narrative/message` (`Mendel`): completed. Key findings: remove internal
  terms such as fairness-gated, locked, promoted, and review-level scrutiny;
  keep FEniTop as context rather than a baseline; make the conclusion less
  self-conscious. Addressed in the validation/prose chunk.
- `Math/solver contracts` (`Kant`): completed. Plasticity3D PMG profiles,
  constitutive branch formulas, and the lambda=1.55 stop contract have now been
  addressed in the manuscript. Smaller findings on Ginzburg--Landau initial
  state, hyperelastic load path, and topology stall-stop policy were addressed.
- `Plasticity3D stop/claim audit` (`Hubble`): completed. Findings addressed:
  the abstract no longer uses the historical lambda=1.0 speedup for the
  converged lambda=1.55 claim, the globalization synthesis no longer overstates
  trust-region necessity for all Plasticity3D endpoints, the MUMPS-backed
  scaling table now shows Newton iterations and gradient-target ratio, and
  "flagship" wording was removed from the P4(L1), lambda=1.5 derivative
  ablation.
- `Solver/assembly narrative` (`Godel`): completed. Findings addressed in the
  current chunk: removed remaining flagship wording, separated Hyperelasticity
  GAMG and PMG/MUMPS profiles, defined Plasticity3D derivative-ablation
  observables, scoped Hyperelasticity PMG energy as fixed-work evidence, stated
  the Plasticity3D KSP tolerance as a PETSc relative residual tolerance, and
  added a reference-formula/mainline scope sentence.
- `Reproducibility/submission` (`Epicurus`): completed. Findings partly
  addressed: source-submission `reviewer_*` table names were removed. Remaining
  blockers are process-level: archive-neutral provenance, venue/declarations,
  license/archive DOI, and a provenance validator that rejects local paths.
- `Layout/narrow-template` (`Copernicus`): completed. Findings partly
  addressed: the current Plasticity3D convergence figure legend crowding was
  fixed by increasing figure height and legend clearance. Remaining risks:
  wide generated tables and fixed-size TikZ diagrams may need a dedicated
  narrow-template robustness pass after the target venue is chosen.
- `Evidence/provenance` (`Wegener`): completed. Key findings addressed:
  JAX-FEM comparison metadata no longer asserts constitutive-law identity,
  strict JSON output is enforced for the runner, and Layer 1A wording no longer
  implies an exact glued-bottom free-mask match. Remaining blocker:
  archive-neutral provenance bundles still contain local paths outside the
  tracked manuscript.
- `Layout/PDF` (`Singer`): shutdown after timeout. Local visual inspection of
  rendered pages 1 and 19--22 found the abstract, validation pages, and
  globalization table readable after this chunk.
- `Wide table/layout` (`Avicenna`): completed. Key recommendation: split or
  wrap the largest diagnostic tables by scientific meaning rather than shrink
  the font further. Addressed by stacked outcome/work, memory/overlap, timing,
  and terminal-observable blocks in the generated table script; visual checks
  passed on pages 23--28 and 33.
- `Message/style` (`Turing`): completed. Findings partly addressed in the
  current chunk: paper-facing platform labels now use single-node and
  multi-node CPU terms, auxiliary/reference-formula/reference-operator wording
  replaces local/source labels where appropriate, and the constitutive-AD claim
  is scoped to the fixed high-order derivative-route test. Remaining high-value
  style work: compress the abstract, strengthen contribution framing across
  the SOTA transition, and add more body-text interpretation for implementation
  figures.
- `Math/self-contained` (`Aristotle`): completed. Small findings addressed:
  derivative-route notation now uses `\Phi_e`, the benchmark table says solve
  contract, and strength-reduction evidence consistently uses
  `\lambda_{\mathrm{sr}}`. Remaining high-value math work: make the plasticity
  problem data more self-contained, unify hyperelastic displacement notation,
  define Plasticity3D branch symbols without `q` conflicts, add topology
  objective data, clarify failed-row semantics, and finish display-equation
  punctuation.
- `Repro/provenance` (`Plato`): completed. Current chunk restored
  source/generated consistency for the touched generated tables and target
  scaling figure. Remaining blockers: archive-neutral provenance bundle,
  repository license plus archival DOI, target venue/template/declarations,
  strict JSON rejection of existing `NaN` provenance, stale reproducibility
  notes, and limited-access citation verification.

## First Edit Backlog

1. Rewrite title and abstract around the toolset and evidence pillars. Done in
   the first framing chunk.
2. Add an explicit contributions subsection to the introduction. Done.
3. Recast `Implementation` as "Reference and mainline solver paths." Done in
   substance.
4. Replace broad "repository" wording in benchmark definitions with
   self-contained terms such as "implemented benchmark," "discrete model," and
   "algorithmic constitutive surrogate." Done for the main manuscript sections.
5. Add a results synthesis paragraph/table using existing generated assets if
   appropriate. Done with a prose synthesis subsection.
6. Audit figures for physical size and font consistency after the narrative
   structure stabilizes. Initial audit done; Plasticity3D validation ladder
   repaired and visually checked on rendered pages 20--21.
