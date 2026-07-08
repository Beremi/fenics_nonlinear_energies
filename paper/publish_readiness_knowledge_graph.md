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
  Current risk: "maintained repository workflow" is louder than "scientific FEM
  toolset for nonlinear problems."
- `INTRO`: positioning and contributions.
  Required edit: open with the nonlinear FEM computational bottleneck and add an
  explicit contribution block with solver, assembly/derivative, and linear-solver
  pillars.
- `RELATED`: literature framing.
  Required edit: consolidate defensive taxonomy into fewer scientific groups:
  FEM automation, differentiable FEM/AD, nonlinear solver infrastructure,
  topology/plasticity source context, and scalable software comparators.
- `METHOD`: mathematical and algorithmic core.
  Current assets: common finite-element energy notation, derivative routes,
  globalization algorithms, colored sparse finite differences, constitutive AD.
  Required edit: make nonlinear solver policy, sparse assembly, and linear
  solver/preconditioner policy equally visible methodological components.
- `IMPLEMENTATION`: realization of the method.
  Current assets: pure JAX, FEniCS, and JAX+PETSc strata; autodiff modes;
  Krylov/preconditioner matrix; distributed assembly.
  Required edit: present as implementation choices for a scientific method, not
  as codebase strata.
- `BENCHMARKS`: mathematical problem coverage.
  Families: p-Laplace, Ginzburg--Landau, Hyperelasticity, Plasticity2D,
  Plasticity3D, Topology.
  Current risk: repeated "repository specialization" wording weakens
  self-contained mathematical presentation.
- `VALIDATION`: external and source-family comparison.
  Evidence: narrow hyperelastic JAX-FEM comparison; Plasticity3D validation
  ladder with endpoint-surrogate scope.
  Rule: keep validation separate from performance and never imply
  path-consistent plastic-history equivalence.
- `RESULTS`: performance and solver behavior.
  Evidence: globalization comparison, derivative-route comparison, scaling
  studies, hyperelastic and Plasticity3D solver diagnostics, topology scaling.
  Required edit: add a compact synthesis of what each result block establishes.
- `FIGURES_TABLES`: visual and layout quality.
  Current fact: `paper/build/main.pdf` is 34 pages, A4, 10 pt article,
  text width about 7.09 in. Many floats use `[H]`, so placement is highly manual.
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
- `Evidence/experiments/repro` (`Linnaeus`): pending.
- `Figures/layout/PDF` (`Archimedes`): pending.
- `Math/notation/self-contained prose` (`Harvey`): pending.
- `Legacy MATLAB/Valdman comparators` (`Halley`): pending.

## First Edit Backlog

1. Rewrite title and abstract around the toolset and evidence pillars.
2. Add an explicit contributions subsection to the introduction.
3. Recast `Implementation` as "Reference and mainline solver paths."
4. Replace broad "repository" wording in benchmark definitions with
   self-contained terms such as "implemented benchmark," "discrete model," and
   "algorithmic constitutive surrogate."
5. Add a results synthesis paragraph/table using existing generated assets if
   appropriate.
6. Audit figures for physical size and font consistency after the narrative
   structure stabilizes.
