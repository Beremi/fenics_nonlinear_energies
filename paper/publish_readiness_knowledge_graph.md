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
external or reference-model comparators where the scope is narrow, and legacy
MATLAB literature only when there is source-backed evidence.

## External Style Anchors

- Local style guide: `paper/style_guide/AGENTS.md` and
  `paper/style_guide/style_fingerprint/agent_quick_reference.md`.
  Apply the 2025 accepted KL-paper style: formal applied mathematics,
  motivation before technical construction, authorial `we`, definitions before
  dense notation, punctuated displays, and evidence-scoped claims.
  The snapshot is intentionally local-only and ignored through
  `.git/info/exclude`; do not stage it with manuscript changes.
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
- SIOPT submission-shape cue checked on 2026-07-09: SIAM's author instructions
  ask for figures to be embedded inline and state that SIOPT has a 25-page
  policy, with longer papers published only in exceptional justified cases. The
  current 41-page A4 article is therefore a journal-template/page-budget blocker
  rather than a solved formatting issue.

## Manuscript Nodes

- `PAPER`: Title, abstract, and conclusion.
  Status: title, abstract, and conclusion are now framed around a scientific
  \jaxpetsc{} toolset for nonlinear FEM energy solves, derivative routes,
  sparse assembly, and solver policy. The abstract now opens from the nonlinear
  FEM bottleneck, avoids internal campaign terms such as locked and promoted,
  and no longer attributes the historical lambda=1.0
  Plasticity3D speedup to the converged lambda=1.55 scaling sweep. Current
  prose chunk removes the remaining `\repo{}`-centered framing from the
  abstract, introduction, related work, discussion, and conclusion, so the
  paper-facing object is the computational toolset rather than the repository
  name. Current chunk removes remaining platform-local labels from the
  Plasticity3D evidence narrative, replacing workstation/Karolina wording with
  single-node and multi-node CPU wording, and uses auxiliary rather than
  historical for timing-only Plasticity3D evidence. Current narrative chunk
  shortens the title to the JAX--PETSc toolset, local AD, sparse assembly, and
  solver-policy message; compresses the abstract from detailed audit prose into
  a two-paragraph methods-and-evidence summary; and rewrites the conclusion as
  synthesis plus three scientific extensions rather than caveat management.
  Current notation-and-claims chunk makes the abstract validation wording
  self-contained, adds PDF title/author metadata, and rewrites the conclusion
  to name PETSc sparse assembly, Newton globalization, Krylov solvers, and
  preconditioner policy explicitly. Current label/notation chunk removes the
  remaining `mainline` wording from the abstract and opening message in favor
  of primary \jaxpetsc{} realization. Current style-scope chunk splits the
  abstract comparator surface into internal reference implementations versus
  narrow external/reference-model comparisons, changes validation and
  performance verbs from broad establishment claims to reported tolerances and
  measured costs, normalizes the \jaxpetsc{} object name, and shortens the
  conclusion so it ends as synthesis rather than repeated caveat management.
  Current provenance/layout chunk removes the remaining defensive software-ranking
  phrasing in the introduction, makes the abstract's final distributed-scale
  sentence name PETSc-owned sparse assembly and solver policy, and sharpens the
  conclusion's topology statement as coupled design-and-mechanics timing with a
  separate rank-consistency check. Current front/back-matter chunk aligns the
  visible title and PDF metadata with energy minimization, removes the automatic
  date, moves code/data availability into unnumbered back matter, and replaces
  draft future-archive wording with a current-version statement. Current
  evidence-gate chunk strengthens the abstract's final sentence around
  PETSc-owned sparse assembly, Krylov and multigrid policy, and
  rank-consistency diagnostics; it also separates JAX-FEM as a matched
  hyperelastic comparison, Sysala-family slope-stability work as
  reference-model/reference-observable context, and the
  \v{C}erm{\'a}k--Sysala--Valdman MATLAB work as implementation-lineage
  context.
- `INTRO`: positioning and contributions.
  Status: now opens with the nonlinear FEM computational bottleneck and includes
  an explicit contribution block with solver, derivative/assembly, and
  linear-solver pillars. Current narrative chunk names the implemented
  \jaxpetsc{} scientific toolset before the contribution list, condenses the
  first literature tour so the SOTA table and related-work section carry the
  detailed taxonomy, and states that non-mainline comparisons are matched
  reference surfaces rather than framework rankings. Current
  notation-and-claims chunk replaces remaining comparison-surface and
  reference-family wording with scoped reference implementations, matched
  observables, and reference-model diagnostics. Current label/notation chunk
  narrows the Plasticity3D introduction claim to reference-observable agreement
  and replaces remaining organization-text `mainline` wording with primary
  \jaxpetsc{} realization. Current SOTA/math polish chunk makes the positioning
  table affirmative and source-scoped: the table records documented
  capabilities, splits JAX-FEM/Xue 2026 from AutoPDEx, removes negative absence
  claims, and replaces bridge/path language with bridge architectures and
  reference implementations. Current evidence-scope chunk compresses the SOTA
  table into a three-column role taxonomy and softens external-comparison
  language so only reported external numerical comparisons are covered by the
  matched-observable statement. Current evidence-gate chunk splits the
  \v{C}erm{\'a}k--Sysala--Valdman implementation lineage from the
  Sysala-family reference-model and reference-observable context.
- `RELATED`: literature framing.
  Status: consolidated defensive taxonomy into fewer scientific groups:
  FEM automation, differentiable FEM/AD, nonlinear solver infrastructure,
  topology/plasticity source context, and scalable software comparators.
  FEniTop is now framed as topology-optimization literature context rather than
  a direct baseline, and the Cermak--Sysala--Valdman MATLAB citation is
  implementation-lineage context only. Current notation-and-claims chunk makes
  the PETSc paragraph affirmative: ownership layout, Krylov policy,
  globalization, and preconditioner design are part of the numerical method.
  Current SOTA/math polish chunk scopes Xue 2026 as a close second-order
  comparator and describes the Sysala continuation/iterative-solver work as a
  reference line rather than a source family. Current evidence-scope chunk
  updates the PETSc web citation key to the current 2026 entry. Current
  evidence-gate chunk names the \v{C}erm{\'a}k--Sysala--Valdman MATLAB
  implementation line explicitly, avoiding a vague single-author lineage.
- `METHOD`: mathematical and algorithmic core.
  Current assets: common finite-element energy notation, derivative routes,
  globalization algorithms, colored sparse finite differences, constitutive AD.
  Status: generic algorithms now use the merit functional `\mathcal{F}` rather
  than the hyperelasticity-reserved symbol `J`; prose is more paper-facing.
  Current notation-and-claims chunk separates finite-element functions from
  coefficient vectors, defines `x_e=R_e x`, distinguishes load-step potentials
  `\Pi_h(x;\theta)` from reduced objectives `\mathcal{J}_h(z)`, and defines
  integrated element contributions versus quadrature-point densities before the
  derivative-route discussion. Current label/notation chunk makes the
  free-DOF convention self-contained for nonhomogeneous essential data by
  introducing `V_{h,0}`, the affine lift `\bar u_h(\theta)`, and the local
  element state `u_e(x;\theta)`; it also treats topology as a schematic reduced
  objective rather than a mechanics potential plus regularization. Current
  SOTA/math polish chunk defines the Armijo backtracking parameters and maximum
  count in the algorithm's given line. Current evidence-scope chunk formalizes
  colored sparse finite-difference recovery with the Hessian, sparsity pattern,
  distance-2 color groups, seed vectors, gradient-difference probes, AD
  Hessian-vector probes, and owned PETSc scatter; the reported JAX+PETSc
  colored-SFD cases are stated as using AD HVP probes, so no finite-difference
  perturbation parameter is introduced for those runs. Current style-scope
  chunk defines the quadrature stress and tangent symbols before use, labels
  the three displayed algorithms for cross-reference stability, makes the
  hybrid Newton solve line distinguish Hessian assembly/application from the
  linear solve, and rewrites colored SFD with a generic Hessian
  `H=\nabla^2\mathcal{F}(x)` rather than hyperelastic-specific notation.
  Current provenance/layout chunk defines the quadrature-point data restriction
  `\theta_q` in the generic element contribution and aligns implementation
  wording with the methods caveat: constitutive AD gives branchwise tangents,
  while colored sparse recovery may use HVP or finite-difference probes.
  Current solver-protocol chunk adds a common solver-status and timing
  vocabulary after the globalization algorithms, so completed endpoint solves,
  fixed-work diagnostics, capped runs, wall times, solver timers, and relative
  correction targets are defined before the numerical tables use them. Current
  math/evidence chunk defines the finite-difference step size `\delta` in the
  colored sparse recovery formula.
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
  Current narrative chunk changes the framework-overview caption from
  manuscript/software-structure wording to computational structure and
  distributed/reference formulation roles. Current label/notation chunk changes
  the visible implementation heading to `Primary \jaxpetsc{} path` and removes
  the remaining Figure 1 `mainline` label. Current notation-and-claims chunk
  removes remaining "hot loop" and "engineering choice" wording and aligns the
  capability matrix with the Hyperelasticity colored-SFD comparison evidence.
  Current SOTA/math polish chunk renames visible pure-JAX/FEniCS reference path
  headings and schematic labels to reference implementations, and normalizes the
  autodiff-mode schematic from `B_q^T C_q B_q` to `B_q^\top C_q B_q`. Current
  evidence-scope chunk replaces remaining framework/path wording with
  realization, construction, toolset, and formulation language, interprets the
  implementation figures in body text, and changes the constitutive-AD diagram
  label from a lowest-cost route to quadrature-point tangent assembly with exact
  local constitutive derivatives.
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
  The benchmark specification matrix now labels problem-specific stopping rules
  as stopping rules and now describes Plasticity3D endpoint, scaling, and
  diagnostic PMG policies without campaign labels. Current chunk makes the
  benchmark definitions more self-contained: Hyperelasticity now consistently
  uses displacement notation with `F(u)` and `J(u)`; Plasticity2D states the
  slope geometry, boundary conditions, material constants, gravity, zero old
  plastic strain, and Davis-B reduction scope; Plasticity3D states the material
  tuples, vertical body-force convention, elastic constants including bulk
  modulus `K`, and avoids overloading the quadrature index with the invariant
  `Q(\varepsilon)`; Topology now states the fine-grid mesh, target volume,
  density floor, elasticity/load data, fixed design pads, frozen element energy,
  and reduced-objective penalty constants. Current narrative chunk removes
  "curated" and "showcase" wording from the Plasticity2D endpoint text and
  generated summary table, distinguishing the completed endpoint case from
  fixed-iteration diagnostics. Current notation-and-claims chunk removes
  remaining visible "showcase" and "corrected glued-bottom" caption wording.
  Current SOTA/math polish chunk normalizes remaining transpose notation to
  `^\top`, defines the Plasticity2D branch-potential symbol, defines the
  topology elasticity tensor and SIMP exponent before use, punctuates the
  Davis-B and Plasticity3D potential displays, and replaces remaining raw/local
  or study-row wording with unreduced, endpoint-surrogate, and study-case
  terminology. Current style-scope chunk defines the reported discrete
  p-Laplace energy symbol, defines the hyperelastic energy density before the
  first Piola stress, records the Plasticity2D branch-discriminant
  regularization value, replaces raw Plasticity3D boundary/source labels with
  mesh and marker language, and updates generated benchmark/availability rows
  to avoid internal stack labels. Current evidence-scope chunk states that the hyperelastic
  Neo-Hookean potential is defined on admissible displacements with positive
  Jacobian determinant and that globalization rejects nonpositive-J trial
  states. It also narrows the Plasticity2D endpoint surrogate by saying the full
  two-dimensional branch-energy family is summarized rather than enumerated in
  the paper.
  Current provenance/layout chunk removes an unused hyperelastic load parameter
  from `\Pi(u)`, names the right-face path as a prescribed rotating Dirichlet
  displacement path, and replaces the Plasticity3D body-force symbol
  `\gamma^{n+1}` with material-region weight `\gamma(x)`. Current leakage and
  self-containedness chunk replaces the Plasticity2D draft branch-formula caveat
  with a scoped branch-potential definition and rewrites Plasticity3D boundary
  markers as the label sets `\Gamma_1,\ldots,\Gamma_5` with component-wise
  Dirichlet sets. Current benchmark-self-containment chunk adds a
  discretization-label table with mesh construction and representative DOF
  counts, defines the Plasticity2D plane-strain matrix and ordered principal
  stress convention before the branch formulas, names the Plasticity3D material
  regions, maps the Plasticity3D boundary labels to geometric faces, and states
  the ordered-principal-strain convention used by the 3D branch tests.
  Current math/evidence chunk removes the Plasticity2D Davis-B/branch-variable
  `\beta` collision by using `a_\lambda`, `b_\lambda`, and `\kappa_\lambda`,
  defines `e_y`, `\Gamma_{D,i}`, `\Gamma_N`, and `V_0` before the
  Plasticity3D strong form, and adds the topology plane-stress tensor and load
  functional before the mechanics solve.
  Current benchmark-readability chunk splits Plasticity2D and Plasticity3D setup
  prose into staged paragraphs for continuum model, geometry/material data,
  endpoint-surrogate functional, and claim-scope caveats without changing
  equations, numerical evidence, or validation claims.
- `VALIDATION`: external and reference-model comparison.
  Evidence: narrow hyperelastic JAX-FEM comparison; Plasticity3D validation
  ladder with endpoint-surrogate scope.
  Rule: keep validation separate from performance and never imply
  path-consistent plastic-history equivalence.
  Status: comparison metrics now state thresholds and the hyperelastic companion
  schedule. The JAX-FEM gate no longer asserts constitutive-law identity; it
  records matched mesh/schedule and terminal post-comparison under the paper
  energy. Plasticity3D Layer 1A is phrased as direct-branch observable
  agreement with the reference formulation, while glued-bottom boundary matching
  is reserved for Layer 2. Current chunk writes strength-reduction evidence as
  `\lambda_{\mathrm{sr}}` in prose, captions, and generated tables, and
  describes the fixed-operator diagnostic as a reference-operator diagnostic.
  Current narrative chunk removes remaining "boundary contract" wording from
  the validation ladder and changes the JAX-FEM generated table rows from
  "Contract/gate" to comparison-condition rows with common mesh, common
  displacement schedule, agreement threshold, and energy re-evaluation labels.
  Current notation-and-claims chunk removes source-family/source-observable
  wording from the validation section and table captions. Current
  label/notation chunk renames Plasticity3D validation from ladder to sequence,
  changes Layer 1A wording to direct-branch endpoint-observable agreement with
  the reference model, replaces remaining source-boundary wording with
  reference-model boundary conditions, and regenerates the validation figures so
  visible labels read `reference model`. Current SOTA/math polish chunk renames
  the Plasticity3D section, caption, and prose from validation
  sequence/schedule to endpoint-surrogate comparison, and defines the compared
  observable `b` in the validation relative-difference metric. Current
  style-scope chunk updates generated validation rows so endpoint observables
  are marked as comparisons, the fixed-load summary row is the only criteria
  count, and the JAX-FEM hyperelastic table reports checked differences as
  below the stated 5 percent threshold. Current provenance/layout chunk narrows
  the fixed-load Plasticity3D validation sentence to matched endpoint observables
  for the endpoint surrogate, not identity of the surrogate itself. Current
  style/evidence chunk states that the direct-branch endpoint-observable check
  uses the seven-step load branch
  `\lambda_{\mathrm{sr}}=1.0,1.1,\ldots,1.6` on `P_2(L_1)`, so the figure,
  table, and interpretation no longer rely on an undefined direct-branch label.
  Current math/evidence chunk changes direct-branch validation wording from
  agreement to comparison where no predefined threshold exists; the fixed-load
  comparison remains the thresholded part of the Plasticity3D validation table.
- `RESULTS`: performance and solver behavior.
  Evidence: globalization comparison, derivative-route comparison, scaling
  studies, hyperelastic and Plasticity3D solver diagnostics, topology scaling.
  Status: a synthesis subsection now states what the result blocks establish:
  nonlinear policy is problem-dependent, multiple derivative routes are useful,
  sparse ownership/preconditioning are part of the numerical method, and the
  primary \jaxpetsc{} realization solves the largest tested cases while
  serial/external formulations serve scoped reference roles. Plasticity3D result
  prose now marks the fixed P4(L1), lambda=1.5 derivative ablation as
  Hypre-backed rank-local timing evidence and separates it from the
  MUMPS-backed PMG convergence/scaling evidence at lambda=1.55. The converged
  P4(L2), lambda=1.55 scaling table now
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
  labels, figure legend labels, and body-text interpretation. Current narrative
  chunk removes "older timing/workflow viability" wording from the results
  opening, recasts the fixed-budget Ginzburg--Landau line as an eight-rank
  budgeted comparison, replaces "current run records" in a derivative-route
  caption with availability language, and changes the Hyperelasticity
  distribution-memory generated table labels from probe/build/result jargon to
  purpose, assembly layout, outcome, agreement check, memory comparison, and
  one-linearization terminology. Current notation-and-claims chunk removes an
  unsupported "near target" claim for failed Plasticity3D line-search Newton,
  removes a hyperelastic energy-in-table claim not supported by the visible
  table, scopes trust-region necessity to the tested Plasticity3D gate, and
  reframes topology as end-to-end timing under rank-dependent stopping with a
  separate controlled rank-consistency check. Current results-label chunk
  removes remaining body-facing `row`, `probe`, and `ablation` wording from the
  main results prose; names Hyperelasticity and Plasticity3D fixed-work cases
  as fixed-cost, fixed-nonlinear-work, or single-linearization comparisons; and
  regenerates result tables with `Outcome`, `Linearization`, `Nonlinear work`,
  and `Schedule` labels instead of generic `Result` or `fixed work` cells.
  Current label/notation chunk renames the benchmark-specification column from
  `Stopping rule` to `Solve policy`, aligns the Plasticity3D CPU scaling caption
  with `CPU setting`, and converts visible JAX-FEM/Plasticity3D figure
  annotations from raw `e` notation to rendered scientific notation. Current
  SOTA/math polish chunk retitles affected result captions as fixed-case,
  matched-case, or rank-local comparisons, removes corrected-glued-bottom
  revision language, and scopes colored SFD as a comparison and fallback route
  rather than a verification claim. Current publish-readiness chunk adds
  body-text interpretations for the main scaling, memory, and PMG-sensitivity
  figures/tables; removes stale glued-bottom, workflow, fixed-schedule, and
  rank-dependent wording from the visible manuscript; and scopes the
  Ginzburg--Landau fixed-budget result to the observed line-search timeout.
  Current evidence-scope chunk adds explicit interpretations for the
  hyperelastic multi-node fixed-work timing table and the topology scaling
  summary, separating fixed nonlinear work from adaptive-stopping timing.
  Current style-scope chunk adds Plasticity2D prose that separates the
  completed endpoint case from fixed-work diagnostics, describes topology
  rank-consistency variation as bounded rather than rank-invariant, and
  rewrites the fixed Plasticity3D derivative-route result as a measured cost
  comparison under identical reported terminal observables. Current
  provenance/layout chunk removes the ambiguous "largest distributed mechanics"
  phrase from the auxiliary Plasticity3D timing discussion and describes it as a
  high-DOF Plasticity3D timing result. Current front/back/leakage chunk removes
  setup-failure and omission phrasing from the Ginzburg--Landau and colored-SFD
  summaries, replaces remaining process-local wording with level-specific
  problem wording, and regenerates the hyperelastic PMG table with paper-facing
  `Hypre` and `MUMPS, one redundant group` labels.
  Current solver-protocol chunk adds a numerical-protocol summary at the start
  of the results section, retitles mixed timing columns as `Solve/elapsed [s]`
  and `Solve/total/wall [s]`, and clarifies the Plasticity2D generated-table
  caption as endpoint plus fixed-work diagnostic evidence. Current
  evidence-gate chunk makes the mixed timing sources auditable in the generated
  tables: the derivative-route comparison has an explicit timing-scope column,
  the Ginzburg--Landau timeout row reports elapsed wall time and the wall-time
  cap, and the Plasticity3D globalization table reports the final
  gradient-to-target ratio so the failed line-search endpoint cannot be read as
  convergence-equivalent to the trust-region rows. Current style/evidence chunk
  distinguishes the Plasticity3D `P_4(L_2)`,
  `\lambda_{\mathrm{sr}}=1.55` scaling table's stopping-gradient norm from the
  benchmark table's final-gradient norm, adds energy to the hyperelastic
  replicated/rank-local same-work rows, and changes the Ginzburg--Landau family
  highlight from same displayed energy to agreement within about `\num{1e-6}`.
  Current math/evidence chunk adds the omitted `P_2(L_2)` member of the
  `610964`-free-DOF Plasticity3D matched set, narrows fixed-work
  Hyperelasticity PMG interpretation to cost-and-terminal-state evidence rather
  than equal-accuracy timing, and makes the Plasticity3D
  `\lambda_{\mathrm{sr}}=1.55` scaling caption explicitly distinguish its
  stopping-gradient metric from the degree benchmark's final-gradient column.
- `DISCUSSION_CONCLUSION`: interpretation and scope.
  Current prose states the toolset lesson in paper-facing terms: automatic
  differentiation alone does not determine large nonlinear FEM behavior;
  globalization, sparse ownership, overlap replication, coarse-level policy,
  preconditioning, and surrogate-model interpretation are part of the numerical
  method. Current chunk reframes caveat-led language as scope of evidence and
  keeps the conclusion tied to fixed-test evidence rather than broad best-path
  claims. Current narrative chunk moves the discussion's main methodological
  lesson to the opening, replaces implementation-contract vocabulary with
  scalar-energy definition and matched-boundary language, and rewrites the
  future-work paragraph as scientific extensions. Current notation-and-claims
  chunk makes the discussion endpoint-scoped for Plasticity3D, quotes the
  hyperelastic comparison discrepancy scale, and classifies solver-policy
  diagnostics separately from symmetric external comparisons. Current SOTA/math
  polish chunk makes the Plasticity3D discussion use endpoint-surrogate
  comparison terminology consistently. Current publish-readiness chunk centers
  the abstract, introduction, discussion, and conclusion on the scientific
  \jaxpetsc{} toolset and evidence limits, replacing novelty-heavy language with
  documented realization plus benchmark evidence. Current evidence-scope chunk
  keeps broad conclusions tied to documented solver construction and numerical
  evidence rather than a software-ranking or framework-ranking claim.
- `FIGURES_TABLES`: visual and layout quality.
  Current fact: `paper/build/main.pdf` is 41 pages, A4, 10 pt article,
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
  source-submission names; they were renamed to scientific names for fixed-budget
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
  text block. Current benchmark-self-containment chunk rebuilt the 37-page PDF
  and visually checked rendered pages 13--22. The Plasticity2D handoff, dense
  Plasticity3D definitions, topology/validation transition, and page-22
  validation-to-results transition are readable and unclipped; Section 6 now
  starts on page 20 and Section 7 starts low on page 22, so future validation
  edits should recheck this boundary. Current narrative chunk rebuilt the
  37-page PDF and visually checked rendered pages 1, 2, 6, 14, 20, 25, 26, 34,
  and 35. The shorter title/abstract, condensed introduction, implementation
  caption, Plasticity2D endpoint table, validation comparison table,
  Hyperelasticity distribution-memory table, discussion, and conclusion are
  readable and unclipped. Current notation-and-claims chunk renames visible
  generated labels such as `Solve contract` and `Platform` to paper-facing
  `Stopping rule` and `CPU setting`, adds PDF metadata, and records Banach's
  residual layout warning: dense `[H]` floats, the SOTA table, and the 10-column
  Plasticity3D scaling table remain template-fragile even though the current A4
  PDF samples were clean. Current results-label chunk rebuilt the 37-page PDF
  and visually checked rendered pages 23, 24, 26, 27, 28, 29, and 32. The
  globalization/derivative-route outcome tables, Plasticity3D linearization
  table, Hyperelasticity memory/PMG tables, and topology schedule table are
  readable, unclipped, and within the text block. Current label/notation chunk
  rebuilt the 37-page PDF after targeted figure regeneration and visually
  checked rendered pages 1, 6, 7, 20, 21, and 30. The abstract/opening, primary
  \jaxpetsc{} implementation heading, Figure 1, JAX-FEM panels, Plasticity3D
  validation sequence, and CPU-scaling caption are readable, unclipped, and free
  of the flagged raw labels. Current SOTA/math polish chunk rebuilt the 38-page
  PDF and visually checked rendered pages 2, 5, 7, 9, 17, 18, 21--23, 25, and
  27--30. The expanded SOTA table, Armijo algorithm, implementation schematics,
  reference-availability table, Plasticity3D definitions, endpoint-surrogate
  comparison, derivative-route tables, Hyperelasticity rank-local tables, and
  Plasticity3D degree/scaling figures are readable, unclipped, and within the
  text block. The log scan, PDF text scan, asset validator, `qpdf --check`, and
  `git diff --check` were clean. Current publish-readiness chunk rebuilt the
  39-page PDF and visually checked pages 17, 20, 25, 27, 31, and 37--39 after
  targeted figure/table regeneration and the `xurl` bibliography break fix. The
  Plasticity3D convergence caption, JAX-FEM panels, dense derivative-degree
  table, Hyperelasticity scaling page, Plasticity3D degree/scaling page, and
  bibliography pages are readable and unclipped. Current evidence-scope chunk
  rebuilt the 40-page PDF after table/script/prose edits; visually checked pages
  1, 2, 4, 5, 13, 17, 18, 29, 34, 35, 37, 38, and 40; compressed the SOTA table
  into a readable role taxonomy; fixed `siunitx` digit grouping for signed
  decimal values; and found no unresolved refs, overfull boxes, PDF structural
  errors, or banned local/process labels in the visible text scan. Current
  provenance-gate chunk rebuilt the 40-page PDF after validation/appendix label
  edits; `qpdf --check`, the log scan, PDF text scan, and default paper asset
  validator were clean. The full figure-generation script was interrupted in
  the known mesh/TeX-helper stall, so binary figure churn from that attempted
  regeneration remains unstaged unless a later chunk intentionally refreshes
  those assets. Current submission-bundle chunk did not intentionally refresh
  figure binaries or the rendered PDF; it redirected paper-critical figure
  manifest inputs to the curated submission bundle and revalidated the existing
  generated surface. Current style/layout chunk rebuilt the 40-page PDF after
  prose and generated-table edits, moved the Plasticity3D degree/resolution
  interpretation before the forced figure to reduce the page-30 blank region,
  and visually checked rendered pages 1, 5, 9, 21, and 30--32. The PDF log
  scan, `qpdf --check`, and asset validators are clean. Remaining layout risks
  from the audit are target-template fragility rather than current A4 build
  failures: dense `[H]` float placement, Table 16 and related scriptsize result
  tables, appendix fixed-reference tables, the SOTA table, and compound
  Plasticity3D figures should be revisited after the venue class is chosen.
  Current provenance/layout chunk relaxes the worst hard-pinned floats reported
  by the PDF audit: the opening globalization tables, the hyperelastic
  validation figure, a representative early scaling figure/table, the
  Plasticity3D partitioning table, the topology figure/table block, and the
  appendix fixed-reference tables now use `!htbp` rather than `[H]`. Current
  front/back/leakage chunk rebuilt a 39-page PDF;
  `latexmk`, `qpdf --check`, the LaTeX log scan, `git diff --check`, the
  archive-neutral asset validator, and `make -C paper publish-check` passed.
  Rendered pages 1, 18--19, 23--24, 28--29, and 36--37 were checked for title,
  benchmark-definition, results-table, and back-matter readability.
  Current float/font chunk rebuilt the 40-page PDF after comparator-scope,
  p-Laplace geometry, globalization, and layout edits. `make -C paper
  submission-check` passed; `pdffonts` on `hyperelasticity_state.pdf` now
  reports Computer Modern fonts rather than the former NewTX/Termes outlier.
  Rendered pages 13, 21--22, and 30--32 were checked. The JAX-FEM comparison
  figure/table now appear before their interpretation, and the Plasticity3D
  derivative-route Table 20 now precedes the discretization/scaling Figure 26.
  Remaining current-A4 layout compromise: Section 6 leaves some blank space
  before the forced hyperelasticity figure/table block, but the prior
  figure/table inversion is gone.
  Current solver-protocol chunk rebuilt the 41-page PDF after adding the
  reporting-vocabulary and numerical-protocol tables. `make -C paper
  submission-check` passed after shortening one generated table cell that caused
  an underfull box. Rendered pages 5--6 and 24--25 were visually checked; the
  new tables are readable, unclipped, and within the text block. Current
  evidence-gate chunk rebuilt and checked the same 41-page A4 PDF after
  evidence-table edits; `make -C paper submission-check` passed, and rendered
  pages 1--2, 25--27, and 38--41 were visually inspected. The new timeout,
  timing-scope, and gradient-gate columns are readable in the current A4 build.
  Current benchmark-self-containment chunk rebuilt the 41-page PDF after adding
  the discretization-label table and Plasticity2D/3D convention prose. `make -C
  paper submission-check` passed after ragged-right table columns removed an
  underfull-box warning. Rendered pages 11 and 17--18 were checked; the new
  table and benchmark convention text are readable in the current A4 build.
  Current style/evidence/layout chunk regenerated the topology density figure
  with more bottom margin and the Plasticity3D validation surface comparison
  with separated horizontal colorbars; `make -C paper submission-check` passed,
  and rendered pages 21 and 24 were checked for visible x-axis labeling,
  separated colorbar tick labels, and unclipped captions/tables.
  Current math/evidence chunk rebuilt the 41-page PDF and visually checked
  pages 5, 16, 21, 24, 36, and 37 after math/evidence/table-label edits; the
  affected algorithms, Plasticity2D formulas, topology definitions, validation
  page, discussion opening, and appendix tables are readable and within the
  current A4 text block.
  Current benchmark-readability chunk rebuilt the 41-page PDF after Plasticity2D
  and Plasticity3D paragraph-flow edits; `make -C paper submission-check` and
  `git diff --check` passed, and rendered pages 14--21 were visually inspected
  for the Plasticity2D/3D handoff, endpoint-surrogate caveats, figure/table
  placement, and Topology transition.
- `REPRO`: reproducibility and submission readiness.
  Known blockers from `paper/todo.md`: target journal/template/declarations,
  repository license/archive DOI, and archive-neutral provenance for critical
  artifacts. The JAX-FEM baseline runner now writes strict JSON with `null`
  warmup timings and `allow_nan=False`; ignored local baseline metadata was
  corrected in the workspace, but archive-neutral submission bundles remained
  outstanding before the submission-bundle chunk. Current subagent audit
  confirmed the same remaining process blockers: target venue/template and
  declarations, repository license plus archival DOI, and final release
  provenance for paper-critical artifacts. Current chunk
  restored source/generated consistency for all edited generated tables and
  the single changed Plasticity3D scaling figure. Current evidence-scope chunk
  regenerated the ignored reproducibility note without host, platform, or local
  interpreter-path details, renamed appendix table files from source-local names
  to reference-formula/reference-operator/continuation names, and left final
  archive-neutral provenance manifests, venue declarations, DOI, and license
  decisions as submission blockers. Current provenance-gate chunk makes the
  figure manifest use structured repository-relative and explicit external
  inputs, adds a default paper-facing provenance scan to
  `validate_paper_assets.py`, and adds `--archive-neutral` plus
  `make publish-check` for submission readiness. The default validator passes
  on the manuscript-facing surface; `--archive-neutral` intentionally fails on
  the remaining raw-results/report inputs, the external reference state,
  `/home`, `.venv`, `tmp/source_compare`, `NaN`, and the ignored build
  reproducibility note until a real submission bundle exists. Current
  submission-bundle chunk creates the curated bundle at
  `artifacts/reproduction/paper_submission_2026_07_08/`, copies the
  paper-critical JSON, CSV, `.npz`, and `.mat` inputs, records source and bundle
  SHA256 hashes, rewrites local paths to archive-neutral references, and writes
  non-finite JSON numbers as `null`. Paper figure and table scripts now read
  the paper-critical validation/comparison/scaling inputs from this bundle.
  The default asset validator, `--archive-neutral`, and `make publish-check`
  now pass for the paper-facing figure/table provenance surface. Remaining
  submission blockers are target venue metadata/declarations, repository
  license, and a permanent archival release or DOI. Current style-scope chunk
  records the remaining provenance audit risk: the curated bundle supports the
  current paper-facing validation gate, but the final release should still
  provide complete per-figure and per-table provenance for every submitted
  visual artifact. Current provenance/layout chunk adds a figure-provenance
  schema: every generated figure now has a `generated_asset_sources` record with
  generator path/function and archive status. The validator now requires these
  records for TeX-included figures. The refreshed manifest records 34 generated
  figure sources, all archive-neutral. The bundle now stores derived
  Plasticity3D surface, slice, degree-energy, and convergence inputs for the
  submitted figures, with raw source hashes recorded in metadata. Current
  table-provenance chunk adds `paper/tables/generated/manifest.json` with source
  provenance for all 32 generated tables. The validator now requires table
  source records for all 30 TeX-included generated tables. Current
  archive-coverage chunk expands the curated bundle with small raw/report table
  inputs, Plasticity2D endpoint and resolution inputs, and the Plasticity3D
  recommended-scaling per-rank outputs. A table-specific fixed-reference PMG
  summary replaces internal route identifiers with paper-facing aliases, so all
  32 generated table sources are now archive-neutral. Current front/back/leakage
  chunk keeps the availability statement truthful for the current repository
  snapshot by naming the source repository and stating that no separate archival
  DOI is cited for this version; the DOI/license and target-venue declaration
  work remains release-level.
  Current float/font chunk leaves archive-neutral validation green and updates
  only paper source, generated tables, the hyperelasticity state figure, and the
  rebuilt PDF. It does not resolve release-level blockers: target template,
  declarations, license, archival release/DOI, and final bundle integration.
  Current solver-protocol chunk leaves archive-neutral validation green and
  updates only paper source, generated tables, readiness notes, and the rebuilt
  PDF. It does not resolve the release-level blockers: target template,
  declarations, license, archival release/DOI, and final bundle integration.
  Current evidence-gate chunk expands the curated bundle with the Plasticity3D
  globalization output JSONs and the Ginzburg--Landau timeout run metadata used
  by the generated tables. The bundle manifest now records 51 source files.
  Archive-neutral validation remains green; release-level blockers are still
  target template, declarations, license, archival release/DOI, and final bundle
  integration. Current style/evidence/layout chunk keeps source/generated
  consistency for the edited generated tables and only intentionally refreshes
  the topology-density and Plasticity3D-validation figure assets; unrelated
  pre-existing generated-figure drift remains outside the staged scope. Current
  math/evidence chunk aligns the PETSc audit/manifest key with the manuscript
  citation `petsc2026web`; the ignored local cached HTML filename was renamed
  to match the audit key.

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
  Direct-branch reference-observable agreement and fixed-load reference-operator
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
  campaign-review terms such as locked, promoted, and review-level scrutiny;
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
  addressed: source-submission table names were made scientific. Remaining
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
- `Math/self-contained benchmark audit` (`Schrodinger`): completed. Findings
  addressed in the current chunk: hyperelastic displacement/deformation
  notation is unified; Plasticity2D data and Davis-B scope are explicit;
  Plasticity3D material/body-force/bulk-modulus/invariant notation is explicit;
  topology objective constants and frozen element energy are explicit.
  Remaining low-risk polish: a full display-equation punctuation pass and
  failed-row semantics outside the benchmark-definition scope.
- `Narrative/style audit` (`Sagan`): completed. Findings addressed in the
  current narrative chunk: abstract compressed, title shortened, introduction
  and contribution labels made more methods-forward, SOTA transition shortened,
  and discussion/conclusion made synthesis-first.
- `Layout/provenance benchmark audit` (`Herschel`): completed. Findings
  addressed locally: pages 13--22 were rebuilt and visually checked after the
  benchmark edit; no overfull boxes, unresolved refs, or clipping were found.
  Remaining process note before submission: refresh the reproducibility note
  when preparing a final publish bundle.
- `Front-matter narrative` (`Huygens`): completed. Findings addressed in the
  current narrative chunk: title shortened, abstract compressed, comparison
  surface clarified, duplicated literature setup reduced, and organization text
  updated for mainline/reference paths.
- `Back-matter synthesis` (`Mill`): completed. Findings addressed in the
  current narrative chunk: conclusion opens synthetically, discussion leads with
  the methodological lesson, contract vocabulary is reduced, and future work is
  phrased as scientific extensions.
- `Paper-facing label sweep` (`Kuhn`): completed. Findings partly addressed:
  implementation caption, Plasticity2D endpoint labels, selected results
  opening phrases, appendix reproducibility wording, and the JAX-FEM and
  Hyperelasticity distribution generated table labels were revised. Remaining
  style work from Kuhn was addressed in the current results-label chunk for the
  main results section and generated result tables.
- `Results prose label audit` (`Raman`): completed. Findings addressed in the
  current results-label chunk: globalization and derivative-route rows are now
  cases/configurations/comparisons, Hyperelasticity memory evidence is a
  fixed-cost memory comparison, Plasticity3D derivative ablation is a
  derivative-route comparison, and topology rank consistency is described as a
  fixed-schedule test.
- `Generated results table label audit` (`Leibniz`): completed. Findings
  addressed in the current results-label chunk: generated result tables now use
  `Outcome`, `Linearization`, `Nonlinear work`, and `Schedule` labels; the
  Plasticity3D globalization failed state is rendered as `iteration cap`; and
  `fixed work` cells were removed from the affected result tables.
- `Front/back matter current audit` (`Dalton`): completed. Findings addressed
  in the notation-and-claims chunk: abstract comparison and validation wording
  made self-contained, related-work defensive phrasing removed, discussion
  validation/baseline wording made evidence-scoped, conclusion names the PETSc
  solver components, and `Additional Solver Evidence` was renamed
  `Solver-Policy Diagnostics`. Remaining structural note: the SOTA table is
  still dense and partly duplicates related-work positioning.
- `Methodology/implementation current audit` (`Kant-current`): completed.
  Findings addressed: coefficient-vector notation added, `\Pi_h` and
  `\mathcal{J}_h` roles separated, element contributions and quadrature
  densities defined before derivative routes, capability matrix aligned with
  colored-SFD evidence, and implementation prose made less codebase-like.
  Remaining mathematical work: make the trust-region and colored-SFD algorithms
  more formal if the target venue expects algorithm-level reproducibility.
- `Benchmarks/validation/results current audit` (`Bernoulli`): completed.
  Findings addressed: removed unsupported "near target" and hyperelastic
  energy-in-table claims, reframed Plasticity2D reference continuation as
  fixed-policy rank/timing evidence, corrected fixed reference-operator PMG
  convergence wording, and recast topology timing as rank-dependent stopping
  with a separate fixed-schedule consistency check. Remaining style work:
  additional body-text interpretation for some early state/scaling figures.
- `Figures/tables/layout current audit` (`Banach`): completed. Findings partly
  addressed: visible revision-history caption wording removed, generated table
  labels made paper-facing, section title improved, and PDF metadata added.
  Remaining layout risk: `[H]` float placement, the dense SOTA table, and the
  Plasticity3D scaling table should be revisited after a target journal template
  is chosen.
- `Front/back matter label audit` (`Curie-current`): completed. Findings
  addressed in the current label/notation chunk: reference-formulation wording
  was narrowed to reference-observable agreement, and Plasticity3D diagnostic
  ladder wording was replaced by validation sequence. The SOTA `SNES` note
  remains a later clarity polish rather than a blocker.
- `Method/math consistency audit` (`Lovelace-current`): completed. Findings
  addressed in the current label/notation chunk: free-DOF and affine-lift
  notation is explicit, `u_e` is defined, the topology objective is no longer
  overstated as a mechanics potential plus regularization, and the benchmark
  table now says `Solve policy`.
- `Figure/table label audit` (`Tesla-current`): completed. Findings addressed
  in the current label/notation chunk: visible `mainline` wording was removed,
  JAX-FEM and Plasticity3D validation figure annotations use LaTeX scientific
  notation rather than raw `e` notation, and only semantically changed figure
  assets should be staged; metadata-only figure churn should remain unstaged.
- `SOTA/related-work table audit` (`Feynman-current`): completed. Findings
  addressed in the current SOTA/math polish chunk: the SOTA table now uses
  documented-capability wording, splits the JAX-FEM/Xue and AutoPDEx rows,
  replaces negative absence claims with affirmative scope descriptions, removes
  visible `SNES`, and softens bridge/source-family language in the surrounding
  introduction and related work.
- `Math consistency audit` (`Dirac-current`): completed. Findings addressed in
  the current SOTA/math polish chunk: transpose notation is normalized to
  `^\top`, missing Plasticity2D and topology definitions were added, Armijo
  parameters are defined, displayed equations are punctuated, the validation
  metric defines the compared observable, and Plasticity3D validation wording is
  endpoint-surrogate scoped.
- `Figures/captions/table label audit` (`Faraday-current`): completed.
  Findings addressed in the current SOTA/math polish chunk: `study rows`,
  revision-history caption wording, local/raw labels, and `Reference paths`
  labels were replaced with paper-facing study-case, fixed-case, matched-case,
  rank-local, unreduced, and reference-implementation wording.
- `Math precision audit` (`Galileo-current`): completed. Findings addressed in
  the current validation/message chunk: hyperelastic affine/test spaces and
  Neumann boundary are defined; Plasticity2D now defines the body-force load
  vector, endpoint Mohr--Coulomb branch potential, and regularization role;
  Plasticity3D now defines component boundary/test spaces, load-vector sign,
  constitutive argument list, and the $\psi=0^\circ$ dilation scope; topology
  now separates the bilinear form from the mechanics solve and indexes the
  frozen reduced objective by outer design iteration; the hybrid Newton
  algorithm is explicitly schematic and defines actual/predicted reduction; the
  Plasticity3D degree-resolution result is scoped to observed endpoint objective
  values.
- `Narrative/self-containedness audit` (`Helmholtz-current`): completed.
  Findings addressed in the current validation/message chunk: availability prose
  is affirmative rather than draft-process wording; central claims are scoped to
  the tested benchmark classes; front/back Plasticity3D language uses
  final-load scalar/field observables and active boundary conditions rather than
  free-DOF mask language; implementation prose now says computational
  realization/framework/construction and implemented capabilities; conclusion
  names the exact fixed high-order derivative-route comparison and
  $P_4(L_2)$, $\lambda_{\mathrm{sr}}=1.55$ scaling sweep.
- `PDF layout audit` (`Euclid-current`): completed. Findings addressed in the
  current validation/message chunk: Plasticity3D validation figure text and
  colorbar ticks were enlarged and visually checked on rendered page 22;
  Plasticity3D degree/resolution subpanels were made taller with fewer repeated
  legends/labels and visually checked on rendered page 30; Figure 1 TikZ box
  hyphenation was removed and visually checked on rendered page 7. Remaining
  layout risk: the Plasticity3D derivative-degree table on page 25 is dense but
  readable; the SOTA table remains a later structural-table candidate.
- `Narrative/message audit` (`Volta-current`): completed. Findings addressed in
  the current publish-readiness chunk: the abstract and introduction now present
  a scientific toolset rather than a loose framework/workflow family; related
  work is less defensive; and the conclusion states documented realization and
  benchmark evidence instead of broader contribution claims.
- `Math/self-containedness audit` (`Franklin-current`): completed. Findings
  addressed: generic potential data now define external loads and element
  parameters; SFD, HVP, and PMG acronyms are expanded; Plasticity2D and
  Plasticity3D boundary/test-space definitions are more explicit; topology
  spaces, load patch, mesh spacings, pads, and element-restricted bilinear form
  are stated before use. Deferred risk: the full Plasticity2D branch family
  remains summarized rather than exhaustively reproduced.
- `Evidence/claims audit` (`Lorentz-current`): completed. Findings addressed:
  hyperelastic validation no longer includes a speed comparison; Plasticity3D
  validation is endpoint-surrogate scoped; generated tables use factual family
  highlights; failed globalization rows render as iteration-cap evidence; and
  appendix fixed-reference comparisons are described as wall/solve/ratio
  evidence rather than equal-iteration claims. Deferred risk: final submission
  still needs archive-neutral provenance and DOI/license decisions.
- `PDF/layout audit` (`Boyle-current`): completed. Findings addressed:
  bibliography URL breaking now uses `xurl`; JAX-FEM comparison figure fonts
  were enlarged; Plasticity3D convergence wording was regenerated; dense pages
  and bibliography pages were rendered and inspected. Remaining layout risk:
  page-25 derivative-degree and the SOTA table are acceptable in the current A4
  article but template-fragile.
- `Implementation/method wording audit` (`Arendt-current`): completed.
  Findings addressed in the current evidence-scope chunk: the implementation
  narrative avoids framework/path ranking language, the figures are interpreted
  in body text, colored SFD is mathematically defined, and the constitutive-AD
  schematic no longer claims a lowest-cost route.
- `SOTA/related-work audit` (`Nietzsche-current`): completed. Findings
  addressed: the SOTA table now reports selected computational roles rather
  than a dense capability grid, MATLAB context is scoped to implementation
  literature, and external-comparison prose no longer implies unverified
  universal coverage.
- `Repro/provenance audit` (`Mencius-current`): completed. Findings partly
  addressed: the reproducibility note no longer emits host/platform/local-path
  details, appendix generated-table names are archive-neutral, and availability
  prose asks for a final versioned archival artifact. Remaining blockers:
  archive-neutral provenance bundle, DOI/license, venue declarations, and a
  validator for local-path leakage.
- `Math/layout audit` (`Lagrange-current`): completed. Findings addressed:
  display punctuation, the positive-J hyperelastic admissible set, colored-SFD
  recovery notation, Plasticity2D formula-scope wording, `siunitx` integer
  grouping, and body-text interpretation for the Plasticity3D,
  Hyperelasticity, and Topology result tables. Remaining risk: full
  Plasticity2D branch formulas are still summarized rather than reproduced.
- `Manuscript/process leakage audit` (`Planck-current`): completed. Findings
  addressed in the current provenance-gate chunk: paper-facing figure-manifest
  inputs no longer leak absolute `/home` or `tmp/source_compare` paths;
  fixed-load/reference-operator labels were replaced by fixed-load comparison
  and fixed-reference terminology; and appendix labels no longer carry
  source/local process names. Remaining note: the ignored reproducibility note
  is still an internal run note unless folded into a final archive bundle.
- `Provenance tooling audit` (`Heisenberg-current`): completed. Findings
  addressed: the validator now parses structured manifest inputs, recursively
  scans text provenance in archive-neutral mode, blocks raw-results/report
  inputs and external references for publish checks, and exposes the current
  blocker through `--archive-neutral` and `make publish-check`.
- `Layout/provenance audit` (`Darwin-current`): completed. Findings addressed:
  no PDF build blocker was found, and the provenance gate now reports the
  concrete archive-neutral failures. Remaining layout polish: the page-30 blank
  region has been reduced by moving interpretation before the Plasticity3D
  degree/resolution figure, but dense tables and forced floats remain
  template-fragile until a target class is chosen.
- `Benchmark readability audit` (`Kepler`): completed. Findings addressed:
  Plasticity2D and Plasticity3D setup paragraphs now separate model equations,
  geometry/material data, endpoint-surrogate definitions, external-load
  interpretation, and evidence-scope caveats; the 3D lambda=1.55 endpoint study
  remains separated from the lambda=1.0 auxiliary timing evidence.
- `Archive bundle input audit` (`Erdos-current`): completed. Findings
  addressed in the current submission-bundle chunk: the minimal paper-critical
  bundle includes Plasticity3D validation JSON plus source and maintained branch
  summaries, the source `.mat` state, Plasticity3D derivative comparison,
  lambda=1.55 scaling CSVs, and JAX-FEM comparison summaries plus terminal
  `.npz` states; binary inputs are hashed and copied byte-for-byte, while
  non-finite source JSON values are normalized to `null` in the sanitized
  bundle.
- `Artifact git hygiene audit` (`Boole-current`): completed. Recommendation
  followed: keep the global `artifacts/` ignore rule intact and force-add only
  the narrow curated submission bundle rather than unignoring broad artifact
  trees or staging unrelated generated outputs.
- `Narrative/front-matter audit` (`Bohr-current`): completed. Findings addressed
  in the current style-scope chunk: abstract comparison scope is split between
  internal references and narrow external/reference-model comparisons; broad
  validation/performance verbs are scoped to reported tolerances and measured
  costs; \jaxpetsc{} naming is normalized; and the conclusion no longer repeats
  the same caveat paragraph.
- `Methods/math audit` (`Poincare-current`): completed. Findings partly
  addressed: quadrature stress/tangent symbols, algorithm labels, colored-SFD
  Hessian notation, hyperelastic density-before-stress order, Plasticity2D
  regularization value, and Plasticity3D marker language are fixed. Deferred
  item: the implementation linear-solver summary still needs a careful
  evidence-backed policy statement if expanded.
- `Evidence/results/repro audit` (`James-current`): completed. Findings
  addressed: Plasticity3D validation table status labels, Plasticity2D
  endpoint-versus-diagnostic interpretation, topology rank-variation wording,
  and JAX-FEM table threshold labels. Remaining blockers are release-level:
  target metadata, license/archive DOI, and complete final per-artifact
  provenance.
- `PDF/layout audit` (`Newton-current`): completed. Current A4 PDF has no log
  warnings, overfull boxes, or structural PDF errors. Findings partly addressed
  by the page-30 flow fix. Remaining target-template risks: forced `[H]` floats,
  dense scriptsize result tables, appendix tables, the SOTA table, and compound
  Plasticity3D figures.
- `Narrative/front-back audit` (`Descartes-current`): completed. No hard prose
  blockers found. Findings addressed in the current provenance/layout chunk:
  final abstract sentence, defensive introduction comparison wording,
  duplicated contribution interpretation, related-work closing transition,
  discussion agreement wording, and topology conclusion wording.
- `Methods/solver precision audit` (`Ohm-current`): completed. Findings partly
  addressed: branchwise constitutive tangent wording, colored recovery wording,
  hyperelastic load notation, Plasticity3D body-force symbol, validation
  sentence scope, `\theta_q` definition, and the ambiguous "largest" mechanics
  timing phrase. Deferred item: a compact exact solver-policy table remains
  useful but needs careful evidence collation.
- `Provenance audit` (`Aquinas-current`): completed. Findings partly addressed:
  validator now requires figure source provenance and the figure manifest records
  all generated figure sources with archive status. Current table-provenance
  follow-up also requires generated-table source provenance. Current archive
  bundle expansion covers the small raw/report-backed table inputs, the
  fixed-reference PMG table rows, and derived Plasticity3D visual inputs.
  Remaining provenance work is release-level: publish the bundle as part of a
  durable, licensed, citable archive and cite it in the availability statement.
- `PDF/layout audit` (`Ptolemy-current`): completed. Findings partly addressed
  by relaxing selected `[H]` floats. Remaining risk: current A4 pages must be
  rendered after rebuild to confirm the page-20, page-23, page-33, and page-35
  blank-region behavior improved rather than merely moved.
- `Generated-table provenance audit` (`Confucius-current`): completed. Key
  finding: 30 generated tables are included by TeX, while two generated table
  files are currently unused. Included-table inputs are known: static constants,
  tracked docs assets, curated bundle inputs, or raw/report paths; no included
  table had unknown file inputs. This informed the table manifest and validator
  added in the current table-provenance chunk.
- `Figure/archive gap audit` (`Einstein-current`): completed. Key finding now
  addressed for submitted figures: the bundle stores compact derived arrays for
  the Plasticity3D state/slice figures and compact histories for the convergence
  figure. Full raw-state recomputation would still require the original
  Plasticity3D state/HDF5 material, roughly 13 GiB, but that is no longer a
  submitted-figure provenance gap.
- `Current message/method/evidence/layout audit` (`Ampere`/`Rawls`/`Kierkegaard`/`Carver`):
  completed. Findings addressed in the current clarity chunk: the abstract and
  introduction now name the \v{C}erm{\'a}k--Sysala--Valdman MATLAB
  slope-stability lineage; remaining paper-facing `Plasticity3D` labels were
  softened in the abstract, discussion, and conclusion; the generic reduced
  objective and constitutive AD route now define the state, strain argument,
  branch/material data, and AD-HVP versus finite-difference colored recovery;
  distributed assembly prose now states owned/free, ghost-fill, owned-row
  insertion, and scalar-reduction order; benchmark and appendix tables now have
  body-text interpretation; fixed-reference PMG rows explicitly say
  `fixed-reference operator`; and float-page/table readability was checked on
  rendered pages 25--26 and 31. Remaining target-template risks are the same
  submission-level layout risks: dense tables and compound figures should be
  checked again after the final journal class is chosen.
- `Front/back matter audit` (`Anscombe`): completed. Findings addressed in the
  current front/back chunk: code/data availability moved out of the appendix,
  the automatic date was removed, PDF metadata now matches the full title, the
  abstract and introduction avoid audit/process phrasing, and the submission
  checklist now states the remaining target-venue, declaration, license, and DOI
  blockers without work-log wording.
- `Leakage/self-containedness audit` (`Maxwell`): completed. Findings addressed
  in the current front/back chunk: Plasticity3D marker numbers are defined as
  boundary label sets before use, the Plasticity2D branch-potential role is
  stated without draft apology, Ginzburg--Landau and colored-SFD result prose
  no longer reports setup omissions, and the hyperelastic PMG table now uses
  paper-facing coarse-solver labels.
- `Current style/math/evidence/layout audit`
  (`Goodall`/`Hooke`/`Gauss`/`Chandrasekhar`): completed. Findings addressed in
  the current submission-readiness chunk: comparator roles are split into
  internal references, matched JAX-FEM comparison, and slope-stability
  reference-observable context; the SOTA table is framed as a role map; the
  related-work opening is a taxonomy; Plasticity2D now defines the elastic,
  line-return, and apex-return branch potential rather than referring to hidden
  code; Plasticity3D bottom-coordinate notation and
  $\lambda_{\max}^{\mathrm{succ}}$ fixed-load validation semantics are defined;
  hyperelastic globalization wording now matches the table; scalar and
  hyperelastic scaling figures are interpreted in body text; generated benchmark
  tables use solver-realization labels; and the appendix table sequence is
  protected by a new aux-order check plus `make submission-check`.
- `Solver protocol/timing audit` (`Bernoulli-protocol`): completed. Findings
  addressed in the current solver-protocol chunk: timing headers now distinguish
  mixed solve, total, elapsed, and wall-time scopes; a solver-status vocabulary
  table defines completed, capped, fixed-work, and correction-target semantics;
  a results-section protocol summary maps each evidence block to its solver
  policy and stopping contract; Plasticity3D validation and performance timing
  roles remain separated; and the Plasticity2D table caption now names
  endpoint plus fixed-work diagnostic evidence.
- `Current message/math/evidence/PDF audit` (`Pauli`/`Locke`/`Parfit`/`Bacon`):
  completed. Findings addressed in the current evidence-gate chunk: the
  abstract and introduction now split comparator roles and name the
  \v{C}erm{\'a}k--Sysala--Valdman MATLAB implementation lineage; the conclusion
  avoids over-reading the PMG setup result; derivative-route timing scope,
  Ginzburg--Landau timeout cap, and Plasticity3D gradient-gate evidence are
  visible in generated tables and backed by the curated bundle. The follow-up
  benchmark-self-containment chunk addressed the mesh-hierarchy,
  `P_k(L_\ell)`, Plasticity3D marker, and Mohr--Coulomb convention findings.
  The follow-up display-equation audit checked all manuscript `equation`,
  `align`, and display-math blocks; each already carries sentence punctuation
  once labels and trailing line breaks are ignored. The solver-policy follow-up
  keeps the existing component map and numerical protocol table rather than
  adding another dense float before target-template conversion; methodology and
  implementation prose now say where run-specific policy parameters are
  reported. Deferred findings for a later pass: target-template cleanup for
  dense result tables and forced float regions.
- `Current style/evidence/layout audit` (`Ramanujan`/`McClintock`/`Carson`):
  completed. Findings addressed in the current style/evidence/layout chunk:
  front-facing prose is less provenance-like and more methods-centered; the
  Plasticity3D direct branch, stopping-gradient metric, appendix solver policy,
  Ginzburg--Landau energy agreement, and hyperelastic same-work evidence are
  explicit; and the topology-density x label plus Plasticity3D validation
  colorbars were regenerated and visually checked. Remaining blockers are
  release-level or target-template issues: venue class/declarations, license,
  archival DOI, and possible table/figure simplification after template
  conversion.
- `Current structure/math/evidence audit` (`Fermat`/`Dewey`/`Banach`):
  completed or partly integrated. Findings addressed in the current
  math/evidence chunk: Plasticity2D notation collision, missing Plasticity3D
  and topology definitions, colored-recovery `\delta`, solver acronyms,
  appendix route labels, Plasticity3D matched-degree set, validation comparison
  wording, fixed-work PMG causal scope, SOTA subset wording, and PETSc
  audit-key alignment. Larger structural recommendation remains for a dedicated
  pass: decide a main/supplement split, move the SOTA table or benchmark
  encyclopedia material out of the first narrative path, organize results more
  directly by method question, relax forced floats, and possibly merge
  discussion/conclusion after target-template constraints are known.
- `Current structural-flow/style/layout audit` (`Pascal`/`Hume`/`Zeno`):
  completed and partly integrated. Findings addressed in the current
  structural-flow chunk: the SOTA role-map table moved from the Introduction to
  Related Work, the Introduction now keeps only compact positioning prose, and
  `Solver-Policy Diagnostics` moved after the conclusion under `\appendix`.
  The roadmap now points to Discussion, Conclusion, and Appendix A explicitly.
  Claim-scope polish also replaced overbroad rank-dependent agreement wording
  with rank-consistency/MPI timing evidence under stated protocols, changed the
  conclusion's hyperelastic comparison to a narrow matched comparison, and
  qualified implementation/diagram Hessian wording by active smooth branch and
  AD-derived local second-order information. Remaining layout risk: broad
  forced `[H]` float usage in benchmarks, results, and implementation remains
  target-template fragile even though the current A4 PDF is readable.
- `Current bounded float-placement audit` (`Euler`): completed and integrated.
  Findings addressed in the current float-flow chunk: the Implementation
  schematics and capability matrix, the opening Benchmark tables, and the
  Results prelude now use flexible placement with local `\FloatBarrier`
  guards instead of broad hard placement or a hard page break. Visual checks
  confirmed that implementation diagrams/tables are interpreted after they
  appear, benchmark scope tables precede the scope prose, the long
  discretization-label table remains contained before the first model, and the
  p-Laplace results section can begin at the bottom of the prelude page without
  a forced `\clearpage`. Remaining layout risk: many later problem-specific
  `[H]` floats remain intentionally untouched until target-template conversion
  or a dedicated family-by-family float pass.

## First Edit Backlog

1. Rewrite title and abstract around the toolset and evidence pillars. Done in
   the first framing chunk.
2. Add an explicit contributions subsection to the introduction. Done.
3. Recast `Implementation` as "Reference and mainline solver paths." Done in
   substance.
4. Replace broad "repository" wording in benchmark definitions with
   self-contained terms such as "implemented benchmark," "discrete model," and
   "algorithmic constitutive surrogate." Done for the main manuscript sections;
   current chunk also added missing benchmark data and notation definitions.
5. Add a results synthesis paragraph/table using existing generated assets if
   appropriate. Done with a prose synthesis subsection.
6. Audit figures for physical size and font consistency after the narrative
   structure stabilizes. Initial audit done; Plasticity3D validation ladder
   repaired and visually checked on rendered pages 20--21.
