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
- SIOPT journal scope cue: SIAM describes SIOPT as covering theory and practice
  of optimization; contributions may emphasize algorithms, software,
  computational practice, applications, or links between these subjects.

## Manuscript Nodes

- `PAPER`: Title, abstract, and conclusion.
  Status: title, abstract, and conclusion are now framed around a scientific
  \jaxpetsc{} toolset for nonlinear FEM energy solves, derivative routes,
  sparse assembly, and solver policy.
- `INTRO`: positioning and contributions.
  Status: now opens with the nonlinear FEM computational bottleneck and includes
  an explicit contribution block with solver, derivative/assembly, and
  linear-solver pillars.
- `RELATED`: literature framing.
  Status: consolidated defensive taxonomy into fewer scientific groups:
  FEM automation, differentiable FEM/AD, nonlinear solver infrastructure,
  topology/plasticity source context, and scalable software comparators.
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
  instead of maintained or production-path wording.
- `BENCHMARKS`: mathematical problem coverage.
  Families: p-Laplace, Ginzburg--Landau, Hyperelasticity, Plasticity2D,
  Plasticity3D, Topology.
  Status: repeated "repository specialization" language has been replaced with
  implemented benchmark, discrete model, endpoint surrogate, or source-family
  context as appropriate; generated tables no longer use repository/maintained
  descriptors for paper-facing claims.
- `VALIDATION`: external and source-family comparison.
  Evidence: narrow hyperelastic JAX-FEM comparison; Plasticity3D validation
  ladder with endpoint-surrogate scope.
  Rule: keep validation separate from performance and never imply
  path-consistent plastic-history equivalence.
- `RESULTS`: performance and solver behavior.
  Evidence: globalization comparison, derivative-route comparison, scaling
  studies, hyperelastic and Plasticity3D solver diagnostics, topology scaling.
  Status: a synthesis subsection now states what the result blocks establish:
  nonlinear policy is problem-dependent, multiple derivative routes are useful,
  sparse ownership/preconditioning are part of the numerical method, and the
  \jaxpetsc{} mainline remains the scalable path while serial/external
  formulations serve scoped reference roles.
- `FIGURES_TABLES`: visual and layout quality.
  Current fact: `paper/build/main.pdf` is 35 pages, A4, 10 pt article,
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
- `REPRO`: reproducibility and submission readiness.
  Known blockers from `paper/todo.md`: target journal/template/declarations,
  repository license/archive DOI, and archive-neutral provenance for critical
  artifacts.

## Evidence Nodes

- `E1`: Derivative routes.
  Element AD, constitutive AD, and colored SFD are already described and
  compared. Strongest result: constitutive AD preferred on the locked
  Plasticity3D flagship case while preserving terminal-state agreement.
- `E2`: Nonlinear globalization.
  Tables compare line search, trust region, and hybrid policies. Needs clearer
  narrative as nonlinear-solver expertise, not isolated diagnostics.
- `E3`: Linear solvers/preconditioners.
  PETSc Krylov, Hypre, GAMG, PMG, MUMPS/redundant coarse solves appear across
  implementation and results. Needs one methodological narrative.
- `E4`: Parallel assembly.
  PETSc ownership, ghost/overlap layouts, owned-row sparse insertion, and
  rank-local assembly are present. Needs cleaner connection to final scaling
  evidence.
- `E5`: Comparator surface.
  Pure JAX and FEniCS references exist where implemented; JAX-FEM exists for
  hyperelasticity; Plasticity3D source-family comparisons are narrow; topology
  literature includes MATLAB references but no current direct MATLAB runtime
  comparison is yet established.
- `E6`: Plasticity3D scope.
  Strong source-faithfulness evidence for the endpoint surrogate, but no true
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
