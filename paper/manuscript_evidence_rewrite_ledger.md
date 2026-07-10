# Manuscript Evidence Audit and Rewrite Ledger

Historical source audit: 2026-07-10. Superseded by the rewritten manuscript and
the live status in `paper/publication_revision_execution_report.md`.

## 1. Purpose and audit boundary

This ledger records the pre-rewrite text in `paper/main.tex` and
`paper/sections/*.tex` that motivated the current revision. It does not describe
the current line layout. Its line references and Section 14 hashes intentionally
identify the historical audited snapshot and must not be used as current-source
anchors. Current claims are controlled by the manuscript, protocol cards,
mathematical-status document, and execution report; a new immutable claim audit
must be generated after the clean release candidate freezes.

The task explicitly separates the novelty audit, so this ledger does not browse
or re-verify bibliographic claims. Entries in the related-work section address
manuscript scope and claim strength only; the separate literature audit must
freeze citation accuracy and closest-work novelty before release.

The audit uses the following evidence order.

1. `paper/venue_and_contribution_decision.md` fixes the SISC-style derivative-
   placement route and excludes a new-optimization-algorithm claim.
2. `paper/mathematical_status_and_claim_dictionary.md` fixes mathematical
   meanings, permitted words, and hard prohibitions.
3. `paper/protocols/*.md` fixes experiment questions and admission gates.
4. Traceable pilot reports can justify implementation decisions and provisional
   wording, but not final publication numbers because every current pilot was
   produced from a dirty worktree.
5. Historical manuscript tables and figures have no presumption of validity
   when a later quadrature, stopping, distribution, or globalization diagnostic
   contradicts their interpretation.

The local writing instructions were read in the required order:
`paper/style_guide/AGENTS.md`,
`paper/style_guide/style_fingerprint/agent_quick_reference.md`, and
`paper/style_guide/style_fingerprint/agent_cookbook.md`. The proposed narrative
therefore uses the current-journal style: motivation before machinery, explicit
definitions and assumptions, authorial `we`, equations integrated into prose,
and claims tied directly to proofs or numerical evidence.

### Evidence labels used below

| Label | Meaning for the rewrite |
| --- | --- |
| **P/S** | Proved in the manuscript or source-backed under stated assumptions; admissible now. |
| **N-pilot** | Numerically checked in a dirty-worktree pilot; useful for drafting, but the numerical claim requires a clean frozen rerun. |
| **N-historical** | Numerically observed in an older artifact whose publication contract is incomplete; descriptive only and not admissible as central evidence. |
| **U** | Unestablished or still blocked by an experiment gate. |
| **R** | Rejected for the claimed interpretation by newer evidence. |

Destinations are **Main**, **Supplement**, or **Remove**. `Remove` means remove
from the submitted manuscript and from generated publication assets; it does
not require deleting provenance artifacts.

### Principal evidence anchors

| Evidence source | Lines used in this audit | What those lines establish |
| --- | --- | --- |
| `paper/venue_and_contribution_decision.md:7-33` | Venue and central contribution | SISC derivative-placement route; no new optimization algorithm or universal route ranking. |
| `paper/venue_and_contribution_decision.md:35-80` | Research questions, main/supplement roles, exclusions | Correctness/crossover/solver interaction; topology and GL are secondary; no confounded convergence or scaling claims. |
| `paper/mathematical_status_and_claim_dictionary.md:64-223` | Conditional element/constitutive and colored-recovery arguments | Exact assumptions, proof boundary, fixed-element/assembled pilot status, and open distributed gates. |
| `paper/mathematical_status_and_claim_dictionary.md:250-737` | Family-by-family mathematical status | Permitted p-Laplace, GL, hyperelastic, Plasticity2D/3D, topology, and quadrature/stopping interpretations. |
| `paper/mathematical_status_and_claim_dictionary.md:757-895` | Claim dictionary and publication gates | Meaning of `equivalent`, `verified`, `converged`, `faster`, and hard prohibitions. |
| `artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-DERIV-001/pilot_report.md:3-38` | Smooth and P1/P2/P4 fixed-element checks | Near-roundoff route errors with branch margins; no distributed completion. |
| `artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-DISC-001/pilot_report.md:22-100` | Named-rule fixed-state sensitivity | Small energy differences do not control residual/action differences; old energy-only interpretation rejected. |
| `artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-DIST-001/pilot_report.md:3-48` | Controlled hyperelastic one-/two-rank fixed state | Canonical algebraic gate passes; timing, four ranks, factorization, and solved endpoints remain open. |
| `artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-GLOB-001/controlled_v2/report.md:3-95` | Controlled globalization pilot | Common ordinary KSP contract, one GL robustness observation, no timing or success-rate claim. |
| `artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-MC-001/pilot_report.md:8-65` | Plasticity3D material-point matrix | Five interiors, interfaces, rotations, repeated spectra, and explicit no-regularity/no-validation scope. |
| `artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-VAL-001/pilot_report.md` | Independent smooth and mechanics checks | Manufactured p-Laplace/GL rates, analytic hyperelastic affine patch, and nonaffine hyperelastic displacement/deformation/stress rates. |
| `artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-ROUTE-001/pilot_report.md:11-112` | P2 fixed-state route and P1 tight/loose solve diagnostics | All-five-label tangent-action agreement; inexact-solve sensitivity; no timing ranking. |
| `artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-ROUTE-001/local_cost_model_p1l1_v2/report.md:3-59` | P1 route/covariate pilot | Cost-model inputs and descriptive local timing only; held-out and second-architecture gates remain. |
| `artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-ROUTE-001/analysis_contract_v1/report.md` | Strict finite-map/cost-model admission | The tightened reanalysis rejects all 12 legacy local records because their timing provenance lacks raw per-rank collective-max proof. Zero training and zero holdout rows are publication-model eligible; the analysis returns `not_fit_insufficient_data` with no selector claim. |
| `artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-STOP-001/pilot_report.md:3-84` | Stopping/serialization diagnostic | Tight KSP reduces route differences; runs are capped and Riesz calibration remains open. |
| `paper/protocols/EXP-VAL-001.md:3-20` | Verification scope | Independent manufactured and analytic verification is required. The DOLFINx matched backend is removed from required scope; an optional future comparison remains blocked pending an explicitly authorized ABI repair. |
| `paper/protocols/EXP-ROUTE-001.md:117-165` | Route admission and terminal rules | Correctness before timing, held-out model gate, clean provenance, and censoring. |
| `paper/protocols/EXP-SCALE-001.md:17-70` | Scaling admission | Fixed problem/policy/work, repeated processes, state checks, and declared efficiency interpretation. |

## 2. Publication-level verdict

The current manuscript is not publication-admissible as a SISC derivative-
placement paper. Its mathematical core can be made rigorous, but its abstract,
conclusion, and performance section still depend on historical timing and
endpoint claims that fail the revised evidence contract.

The main blockers are:

1. The headline $P_4(L_1)$ Plasticity3D timing comparison uses a correction-only
   stop, `ksp_rtol=1e-2`, no Riesz-scaled residual, no saved state-equivalence
   gate, incomplete hardware provenance, and only three timings. It cannot
   support the paper's headline route ranking.
2. The nine-case Plasticity3D degree/mesh narrative is invalidated as a
   discretization result. At the same historical P2 and P4 states, changing to
   the 125-point evaluator changes the free residual to
   $7.84\times10^2$ and $2.57\times10^3$ and changes a deterministic Hessian
   action by 2.08% and 4.35%, despite small energy changes. The old energy trend
   therefore cannot establish quadrature adequacy, stationarity for a common
   functional, or continuum behavior.
3. Historical coefficient-space stopping tests are not comparable across mesh
   or degree. A Plasticity3D reference-elastic Riesz map now has a passing P1
   infrastructure smoke, but cross-degree calibration and central endpoint
   regeneration remain open.
4. The old globalization tables compare GMRES line-search bundles with STCG
   trust-region bundles. They cannot isolate globalization. The controlled
   version-2 pilot supplies one qualitative Ginzburg--Landau robustness
   observation, not a timing or success-rate result.
5. The current cost discussion gives structural counts, but the selected
   contribution requires a measured cost model and held-out route/crossover
   assessment. The P1 and P2 fixed-state route pilots establish local algebraic
   behavior only; the Karolina route campaign has been prepared but not run.
6. No central numerical result currently has a clean-commit publication rerun.
   No pilot number should be presented as final evidence until its card's
   publication preflight and provenance contract pass.

The manuscript may nevertheless be rewritten immediately around the conditional
equivalence propositions, the experiment design, and properly labeled
verification evidence. Final route, crossover, and scaling conclusions must
remain placeholders until the corresponding clean campaigns pass.

## 3. Compact SISC narrative to use

### Working title

Use a contribution title rather than a software-inventory title:

> **Derivative Placement in Distributed Nonlinear Finite-Element Newton
> Methods: Conditional Equivalence, Verification, and Route Selection**

If the held-out cost model fails, replace `Route Selection` with
`A Controlled Empirical Comparison`. Do not put `JAX+PETSc toolset` in the
title; the decision memo explicitly states that JAX--PETSc integration is not
the novelty.

### Main argument

The paper should make one argument in this order.

1. A finite-element Newton method can place second differentiation at the
   element energy, at the quadrature-point constitutive potential, or at sparse
   colored HVP recovery.
2. These routes are mathematically equivalent only for one fixed discrete
   scalar functional under explicit quadrature, affine-lift, constraint,
   strain-convention, branch, eigenvalue-ordering, sparsity, ghost, and row-
   ownership assumptions.
3. The assumptions are checked first: smooth manufactured solutions, an
   analytic hyperelastic patch, fixed-element derivatives, assembled tangent
   actions, distributed canonical objects, and branch-margin diagnostics.
4. Only admitted states enter the cost study. The cost model includes local AD,
   constitutive contraction, coloring, sparse insertion, communication, and
   memory terms; degree alone is not used as a predictor.
5. Equal-accuracy full solves and fixed-policy scaling test whether fixed-state
   route ordering survives nonlinear and distributed solver interaction.
6. Results are stated as a finite empirical route map or a conditional selector,
   never as universal superiority of one framework or derivative route.

### Target section structure

1. **Introduction**: bottleneck, closest work, three research questions,
   contributions, and reproducibility.
2. **Discrete setting and conditional route equivalence**: the fixed functional,
   affine constrained map, element/constitutive proposition, colored-recovery
   proposition, and exclusions at switches.
3. **Distributed realization and cost variables**: PETSc row ownership,
   overlap/ghost contract, measured cost decomposition, and stopping metric.
4. **Verification design**: smooth manufactured problems, hyperelastic patch,
   fixed-element/assembled route checks, distribution gate, and conditional
   Plasticity3D branch diagnostics.
5. **Controlled route experiments**: fixed-state route matrix, cost model,
   held-out assessment, and equal-accuracy full-solve confirmations.
6. **Fixed-policy distributed viability**: include only if EXP-SCALE-001 passes.
7. **Discussion and conclusions**: finite scope, branch and quadrature limits,
   empirical rather than universal selection, and reproducibility.

Ginzburg--Landau globalization, Plasticity2D, topology, external companion
implementations, complete solver-policy matrices, and noncentral field images
belong in the supplement unless they directly support one of the propositions.

### Style constraints for every rewritten section

- Use one paragraph for one role, normally 60--80 words.
- Introduce assumptions and symbols before a dense formula.
- Use `we establish` only for a proof and `we verify` only for a passed,
  independently defined numerical check.
- Use `Figure~\ref`, `Table~\ref`, `Section~\ref`, and `\eqref`; retain the
  paper's existing `cleveref` conventions where already used.
- Interpret each table or figure in the body. Captions remain factual.
- Avoid `we can`, repository labels, local paths, campaign language, and raw
  implementation identifiers in final prose.
- Use `selected-branch potential derivative` and `selected-branch tangent` for
  the plastic surrogate. Do not use unqualified `stress`, `consistent tangent`,
  `converged`, `minimizer`, or `validation` without the claim-dictionary gate.

## 4. Historical tables and figures requiring immediate quarantine

This table is the deletion/regeneration queue. Change the generation script and
manifest together; do not hand-edit only the generated TeX or PDF.

| Asset and source | Why its current interpretation fails | Action |
| --- | --- | --- |
| `fig:plasticity3d-convergence`, `paper/sections/benchmarks.tex:1147-1156` | Calls nine historical coefficient-norm runs `completed` and labels the plotted gradient as a convergence metric. EXP-DISC-001 shows different-rule residual/action changes, and EXP-STOP-001 has not calibrated the cross-degree Riesz stop. | **Remove.** Regenerate only after own-rule solves, common 125-point evaluation, and Riesz calibration pass. |
| `tab:plasticity3d-benchmark`, `paper/sections/benchmarks.tex:1161-1171` | The common absolute coefficient-gradient target does not imply equal accuracy, and P2/P4 are not stationary under the common evaluator. | **Remove.** Do not relabel the historical rows. |
| `fig:plasticity3d-highest-slice`, `paper/sections/benchmarks.tex:1189-1198` | Compares states from different quadrature-defined problems and uncalibrated stopping. A common color scale does not repair endpoint inequivalence. | **Supplement only as a historical illustration**, or remove. No discretization conclusion. |
| `fig:plasticity3d-degree-energy`, `paper/sections/results.tex:427-465` | Energy-only degree/mesh trend is explicitly rejected by EXP-DISC-001; degree, mesh, quadrature, and algebraic accuracy change together. | **Remove.** Replace only with separated enriched-rule solves or label the eventual result endpoint sensitivity. |
| `fig:plasticity3d-state`, `paper/sections/benchmarks.tex:1136-1145` | The displayed P4 endpoint has not passed the revised own-rule/common-rule and Riesz stopping gates. | **Quarantine.** It may return only as a clearly illustrative state after a clean admitted solve. |
| `fig:plasticity3d-derivative-ablation` and `tab:plasticity3d-derivative-ablation`, `paper/sections/results.tex:359-413` | Headline ranking uses `ksp_rtol=1e-2`, correction-only termination, no state/residual equivalence, no complete hardware record, and three repetitions. The tight P1 diagnostic demonstrates route sensitivity to the linear tolerance. | **Remove from main and abstract.** Replace with EXP-ROUTE-001 only after clean equal-accuracy runs. |
| `tab:derivative-route-compare`, `paper/sections/results.tex:92-120` | Same displayed energy and work counts are not derivative or endpoint equivalence. Rows lack the revised accuracy, state, repetition, and provenance gates. | **Remove.** Build a new fixed-state correctness table before a timing table. |
| `tab:plasticity3d-derivative-degree`, `paper/sections/results.tex:122-136` | Historical one-step costs do not use one canonical stored state and lack repeated randomized measurements and full covariates. | **Remove.** The new P1/P2 fixed-state screens may inform the replacement design but not a route ranking. |
| `tab:globalization-method-compare` and `tab:gl-globalization-fixed-budget`, `paper/sections/results.tex:30-90` | GMRES line-search and STCG trust bundles are confounded and predate the exact current failure contract. A separate L5 repeated-rejection pilot was explicitly superseded by terminal exhausted-line-search semantics. | **Move production-bundle results to supplement or remove.** Main text may use the controlled tier only after its endpoint/accuracy gates pass. |
| `fig:plaplace-results`, `paper/sections/results.tex:149-168`; `fig:gl-results`, `paper/sections/results.tex:179-200` | Single historical timing observations lack clean provenance, repetitions, and the revised stopping contract. The GL backend curves also use different quadrature-defined functionals. | **Remove as performance evidence.** Keep manufactured spatial verification instead. |
| `fig:hyperelasticity-results`, `paper/sections/results.tex:202-227` | Completed-path timing is not supported by replicated equal-accuracy evidence or a calibrated common stopping metric. | **Remove pending a clean campaign.** |
| `tab:hyperelasticity-distribution-memory`, `paper/sections/results.tex:241-268` | The historical pair changes several construction/distribution factors together; scalar endpoint equality is not canonical algebraic equivalence. | **Replace** with the factorized EXP-DIST design. The one-/two-rank pilot currently supports correctness only, not timing or memory advantage. |
| `tab:hyperelasticity-pmg-sensitivity`, `fig:hyperelasticity-cpu-pmg-scaling`, and `tab:hyperelasticity-cpu-pmg-scaling`, `paper/sections/results.tex:270-325` | Fixed Newton work reaches different states, and the scaling series changes coarse groups with rank. It is not equal-accuracy or fixed-policy scaling under the new protocol. | **Supplement as historical diagnostics or remove.** Replace main scaling only if EXP-SCALE-001 passes. |
| `fig:plasticity3d-scaling`, `tab:plasticity3d-scaling`, `fig:plasticity3d-cpu-scaling`, `tab:plasticity3d-cpu-scaling`, and `tab:plasticity3d-cpu-partitioning`, `paper/sections/results.tex:469-564` | Coefficient stopping is uncalibrated, quadrature policy is unresolved, and several series use different constraints/coarse policies. `Converged scaling` is prohibited. | **Remove from main.** Optional P3D scaling remains blocked by EXP-DERIV/STOP/DISC. |
| `fig:topology-density`, `fig:topology-history`, `tab:topology-benchmark`, `fig:topology-results`, `tab:topology-summary`, and `tab:topology-rank-consistency`, `paper/sections/benchmarks.tex:1448-1495` and `paper/sections/results.tex:568-631` | Historical semantics and adaptive endpoints do not supply feasibility, KKT stationarity, one fixed problem, or rank-equivalent endpoints. | **Move topology to supplement.** Use only the semantics-v2 corrected fixed-work diagnostic and label it a unit/parity check. |
| `fig:plasticity2d-energy-levels` and `tab:plasticity2d-benchmark`, `paper/sections/benchmarks.tex:704-730` | Larger rows are capped, coefficient stopping is uncalibrated, and the surrogate has no projection/envelope theorem. | **Move to supplement or remove.** Retain the apex finite-AD regression as an implementation diagnostic, not an optimization or mechanics result. |
| `fig:hyperelasticity-jaxfem-baseline` and `tab:hyperelasticity-jaxfem-baseline`, `paper/sections/validation.tex:55-128` | The companion solves a different constitutive law and only shares post-evaluation of one energy. | **Supplement.** Do not call it validation or constitutive verification. |
| `fig:plasticity3d-validation` and `tab:plasticity3d-validation`, `paper/sections/validation.tex:130-244` | The separate assembly transcribes the same formulas, raw comparator outputs/source are unavailable, and one comparison changes the active free-DOF mask. | **Supplement or remove.** EXP-MC-001 and fixed-state route checks are stronger internal verification for the selected contribution. |
| All Appendix timing tables, `paper/sections/appendix.tex:8-88` | They use legacy coefficient stops, loose KSP tolerances, incomplete equality gates, and historical timing provenance. | **Remove from submission supplement until regenerated**, or archive only as provenance. |

## 5. `paper/main.tex` ledger

| ID | Exact source | Current issue | Required replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| M-01 | `paper/main.tex:25-28` | PDF metadata retains the software-toolset title. | Use the contribution title from Section 3 and keep metadata identical to the visible title. | Decision fixed | Main |
| M-02 | `paper/main.tex:58-60` | `3D Mohr--Coulomb endpoint surrogate` is acceptable, but `constitutive-AD PMG configuration` is an implementation label embedded as a global prose macro. | Retain the surrogate macro only if it saves repeated, scientifically defined wording. Remove `\localpmg` from final prose and describe the hierarchy once in the experiment protocol. | P/S | Main |
| M-03 | `paper/main.tex:73-74` | The title presents breadth and solver policy rather than the derivative-placement contribution. | Replace with the working title; adjust only after the route/cost terminal decision determines `Route Selection` versus `Controlled Empirical Comparison`. | U pending route gate | Main |
| M-04 | `paper/main.tex:91-124` | The structure gives equal main-text weight to implementation inventory, six benchmarks, and historical performance. | Reorder to the compact SISC structure in Section 3. Put mathematical equivalence before implementation, verification before timing, and move noncentral families to the supplement. | Decision fixed | Main |
| M-05 | `paper/main.tex:126-134` | The availability statement promises a future archival DOI and describes a comparator whose source/raw output is absent. | At release, give the immutable commit/tag, DOI, run-record schema, artifact bundle, and explicit comparator limitation. Delete `will be added before publication`; a submission must contain the actual identifier or state only what exists. | U release gate | Main |

## 6. Abstract, introduction, and related work ledger

### Abstract

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| A-01 | `paper/sections/abstract.tex:2-10` | Opens as a six-family toolset paper and includes topology optimization in the core scope. | Open with the derivative-placement bottleneck in distributed nonlinear FE Newton methods. Name the three routes and the fixed-discrete-functional requirement. Mention p-Laplace and hyperelasticity as main cases and Plasticity3D only as a conditional synthetic branch benchmark. | Decision fixed | Main |
| A-02 | `paper/sections/abstract.tex:12-19` | The $222/288/372$ s headline is inadmissible and `same observables to table precision` is not an equivalence gate. | Delete the complete paragraph. After clean campaigns, report a fixed-state derivative-error result first, then a held-out route/crossover result with architecture, repetitions, interval, and equal-accuracy endpoint gate. If those gates do not pass, state that the study yields a finite empirical route map. | R/U | Main |
| A-03 | `paper/sections/abstract.tex:19-25` | Conflates old assembly/globalization/preconditioner studies with evidence and foregrounds weak external companions. | Replace with one sentence on independent manufactured/patch verification and one sentence limiting the branch surrogate to branch-interior derivative placement. Do not mention JAX-FEM or the unavailable comparator in the abstract. | N-pilot pending clean rerun | Main |
| A-04 | `paper/sections/abstract.tex:23-24` | `satisfy the stated stopping criteria` may be read as publication-level stationarity. Historical coefficient tests do not pass the new common contract. | Say only that nonconvex endpoints are interpreted through first-order tests and no second-order minimum is certified; attach no empirical completion claim until stopping calibration passes. | P for interpretation; U for runs | Main |
| A-05 | `paper/sections/abstract.tex:25-28` | The concluding claim is too generic and repository wording is inventory-like. | End with the conditional selection conclusion actually supported by the final route matrix. Add one concise reproducibility sentence naming an archived release only after it exists. | U | Main |

### Introduction

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| I-01 | `paper/sections/introduction.tex:1-8` | Sound motivation, but `toolset` broadens the paper away from the selected contribution. | Retain the bottleneck and replace `toolset` with `comparison framework` or `derivative-placement methodology`. | P/S | Main |
| I-02 | `paper/sections/introduction.tex:10-22` | The third research question concerns differing references rather than the selected cost/crossover question. | State the three decision-memo questions explicitly: conditional correctness; measurable crossover/cost variables; interaction with nonlinear accuracy, preconditioning, Krylov work, and scaling. | Decision fixed | Main |
| I-03 | `paper/sections/introduction.tex:24-30` | Presents pure JAX, FEniCS, JAX-FEM, and a separate formula assembly as contribution-level references. | Compress to one reproducibility/context paragraph. Put independent manufactured and analytic checks first. Describe JAX-FEM and the endpoint transcription as supplementary companions only. | N-pilot/P/S | Main + Supplement |
| I-04 | `paper/sections/introduction.tex:35-46` | Claims controlled derivative-route comparisons under common globalization, preconditioner, and stopping definitions. Historical central timing does not satisfy that contract. | Split contribution from achieved evidence: (1) conditional equivalence propositions; (2) PETSc-owned realization and preregistered cost variables; (3) controlled experiments, stated only after route/stop/scale gates pass. Until then use `we design` rather than `we establish experimentally`. | P for propositions; U for crossover | Main |
| I-05 | `paper/sections/introduction.tex:47-50` | A scoped protocol is useful but is framed as a contribution comparable to the mathematical and empirical result. | Keep as reproducibility methodology after the two scientific contributions. Define failure/censoring, equal accuracy, and run records concisely. | P/S | Main |
| I-06 | `paper/sections/introduction.tex:53-56` | Correctly excludes component novelty, but `integrated construction and controlled evaluation` is not yet distinguished from JetSCI and recent closest work by passed evidence. | Retain the exclusion. Replace the positive novelty sentence after the separate literature audit freezes the closest-work matrix and after the controlled route campaign passes. | U novelty/experiment | Main |
| I-07 | `paper/sections/introduction.tex:58-67` | Six-family inventory obscures the benchmark roles; topology remains too prominent. `report stationary` overstates historical endpoints. | Give roles: p-Laplace correctness, hyperelastic smooth mechanics/distribution, Plasticity3D conditional synthetic branch case, GL supplementary globalization. Move Plasticity2D and topology to supplement. Use `target stationary endpoints` unless the new residual gate passed. | Decision fixed | Main |
| I-08 | `paper/sections/introduction.tex:69-76` | Roadmap follows the old broad structure. | Rewrite after restructuring. Include an explicit reproducibility subsection per the style guide. | U until structure edits | Main |

### Related work

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| RW-01 | `paper/sections/related_work.tex:3-12` | Correct organization, but the contribution sentence presupposes completed controlled evaluation. | Keep the role-based organization. Say the paper *targets* a controlled derivative-placement comparison until the experiment gate passes. | U for empirical novelty | Main |
| RW-02 | `paper/sections/related_work.tex:51-64` | Correctly avoids architectural priority, but `principal addition` is stronger than the currently passed evidence. | Delimit the proposed addition as conditional equivalence plus PETSc ownership and a preregistered CPU route matrix. Finalize the novelty sentence only after the separate literature audit and experiment outcome. | P/U | Main |
| RW-03 | `paper/sections/related_work.tex:66-80` | Sound boundary between incremental plasticity literature and the endpoint surrogate. | Retain, but call the paper's quantity a `synthetic branch-structured endpoint surrogate` at first occurrence. | S | Main |
| RW-04 | `paper/sections/related_work.tex:82-89` | Topology paragraph describes an adaptive study as present evidence. | Move to supplement and state only the semantics-v2 fixed-work software diagnostic; no algorithm or optimization contribution. | N-pilot | Supplement |
| RW-05 | `paper/sections/related_work.tex:91-99` | Sound historical context for coloring, but the paper needs the recent sparse-AD closest-work result from the separate novelty audit. | Retain classical sources; update only from the literature audit. Do not claim coloring itself as new. | S | Main |

## 7. Methodology and implementation ledger

### Methodology

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| ME-01 | `paper/sections/methodology.tex:3-65` | Common notation is useful, but three generic objective roles devote main-text space to topology. | Retain the affine constrained state and discrete potential. Move generic reduced-design/frozen-objective equations to the topology supplement. Define explicitly $u_e=L_eR_ex+\bar u_e$ so the affine-lift condition is visible in the theorem. | P | Main/Supplement |
| ME-02 | `paper/sections/methodology.tex:67-89` | Mathematically careful, but `endpoints are stationary` can still be read as an empirical claim. | State the exact target and terminology: a computed state is an `approximate first-order stationary point` only when its named Riesz-scaled residual and correction pass. Keep the unique discrete minimizer statement only for p-Laplace. | P; empirical U | Main |
| ME-03 | `paper/sections/methodology.tex:91-100` | The fixed quadrature statement is necessary but incomplete as a comparison contract. | Add identical material/history data, free-variable map, affine lift, strain convention, and state hash. Define $P_k(L_\ell)$ once and avoid literal internal mesh names. | P | Main |
| ME-04 | `paper/sections/methodology.tex:102-151` | The branchwise chain-rule argument is informal and omits the constrained affine map from the displayed formula. | Promote it to a proposition with the six assumptions in the claim dictionary: same scalar quadrature functional; affine lift; fixed $B$; $C^2$ selected branch; fixed branch/eigen ordering; common shear convention. Prove the residual and Hessian identities in the free element variables. State exclusions at switches and unresolved repeated spectra. | P | Main |
| ME-05 | `paper/sections/methodology.tex:121-132` | `colored sparse recovery` groups AD-HVP recovery and finite-difference recovery even though only AD-HVP can be exact in exact arithmetic. | Name the reported method `colored AD-HVP recovery`. Treat finite-difference recovery as related background and retain its truncation error explicitly. | P | Main |
| ME-06 | `paper/sections/methodology.tex:153-168` | This is a structural output-count comparison, not the central finite-element/distributed cost model promised by the decision memo. | Keep the counts as motivation, then define measured terms for local kernel, $B^TCB$ contraction, coloring, HVPs, insertion, communication, preconditioner setup, and memory. State the model's train/holdout and censoring rule. Do not claim a selector until held-out validation passes. | P for counts; U for model | Main |
| ME-07 | `paper/sections/methodology.tex:170-232` | One generic algorithm conflates ordinary line search, external STCG trust, reduced-subspace trust, and hybrid variants. Its retry/failure language is underspecified. | Replace with three separately named algorithms or one exact common skeleton plus three complete policy definitions from `docs/implementation/trust_region_linesearch_algorithm.md`. Include terminal exhausted-line-search behavior, last-state preservation, nonfinite handling, radius thresholds, and counter definitions. | Implementation documented/tested; problem campaign pending | Main/Supplement |
| ME-08 | `paper/sections/methodology.tex:234-268` | Mostly descriptive, but stopping permits arbitrary coefficient gradients and corrections without defining the publication metric. | Define the Riesz dual residual and primal correction as the publication contract, separate from the coefficient-Euclidean trust norm. State that coefficient norms are diagnostics. Include independent norm-solve residual checks. | Infrastructure N-pilot; calibration U | Main |
| ME-09 | `paper/sections/methodology.tex:270-280` | Calls the bottom-clamped P3D scaling configuration `converged` and codifies loose `ksp_rtol=1e-2`/`1e-1` historical policies. | Delete. Solver parameters belong with admitted experiments; no historical P3D scaling profile is central evidence. | R | Remove |
| ME-10 | `paper/sections/methodology.tex:295-331` | The coloring argument is correct but is not stated as a proposition with all distributed conditions. | Add the complete-pattern, row-noninterference, ghost-coverage, and unique-row-ownership proposition and one-term proof. Distinguish serial assembled evidence from the still-open multi-rank ownership gate. | P; N-pilot serial; U distributed P3D | Main |
| ME-11 | `paper/sections/methodology.tex:351-383` | `supplies the element tangent at lower cost` is an unsupported performance generalization; `stress` and `tangent` are unqualified for a synthetic surrogate. | Use `selected-branch potential derivative` and `selected-branch tangent`. Say constitutive AD reduces the differentiated local dimension, while observed cost also depends on contraction, assembly, and architecture. Reserve `lower cost` for an admitted measured row. | P for identity; U for cost | Main |

### Implementation

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| IM-01 | `paper/sections/implementation.tex:3-28` | Accurate architecture, but external implementations are called references without separating independent mathematics from companions. | Describe JAX/PETSc ownership in main text. Put manufactured/analytic verification before FEniCS/JAX-FEM availability. Call unmatched implementations `companions`, not references. | P/N-pilot | Main |
| IM-02 | `paper/sections/implementation.tex:30-48` | `AD-HVP recovery adds only roundoff` lacks its pattern/ownership qualification. | Replace with: under the complete-pattern and ownership proposition, AD-HVP recovery is exact in exact arithmetic; tested floating-point equality is reported with norm and state. | P/N-pilot | Main |
| IM-03 | `paper/sections/implementation.tex:50-80` | Three framework diagrams repeat inventory and consume main-text space. | Retain at most one derivative-placement diagram that shows the three routes converging to the same PETSc-owned operator. Move or remove the framework/toolset inventory diagrams. | Editorial | Main/Supplement |
| IM-04 | `paper/sections/implementation.tex:82-128` | Long family-by-family solver inventory is not the contribution and anchors the text to invalid historical policies. | Compress to the linear algebra assumptions material to route comparison. Move full preconditioner policies to experiment cards/supplement. Do not infer SPD from a solver choice. | P/S | Main/Supplement |
| IM-05 | `paper/sections/implementation.tex:130-165` | The ownership construction is central and mathematically useful, but no current manuscript result proves all factor changes independently. | Retain the construction. Add the canonical permutation/hash contract and cite EXP-DIST as a numerical gate. After a clean rerun, report the one-/two-/four-rank fixed-state errors; until then do not claim general distributed equivalence. | P + N-pilot; U full matrix | Main |
| IM-06 | `paper/sections/implementation.tex:167-189` and `paper/sections/tikz_diagrams.tex:82-90` | The generic globalization graphic implies one hybrid fallback contract and is no longer exact enough. | Replace with a small three-method diagram matching the documented algorithms: Newton--Armijo, external STCG trust/hybrid, and reduced-subspace trust--Armijo. | Implementation documented | Main/Supplement |
| IM-07 | `paper/sections/implementation.tex:191-209` | Capability inventory gives all six families equal importance. | Regenerate for main cases only; put complete capability matrix in supplement. | Editorial | Main/Supplement |

## 8. Benchmark section ledger

### Suite framing and tables

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| B-01 | `paper/sections/benchmarks.tex:3-8` | States a breadth/toolset thesis rather than the selected derivative-placement thesis and calls the slope cases elastoplastic. | State that benchmark roles are chosen to test smoothness, constitutive contraction, branching, quadrature, and distribution. Call both plasticity cases synthetic endpoint surrogates. | Decision fixed | Main |
| B-02 | `paper/sections/benchmarks.tex:10-50` | Generated scope/reference tables encode obsolete scaling and comparison evidence. | Regenerate a compact benchmark-role table with columns: discrete functional, regularity, route availability, verification role, main/supplement status, and open gate. Remove timing from this table. | Decision fixed | Main |
| B-03 | `paper/sections/benchmarks.tex:55-99` | Mesh inventory is accurate but overlong for the new core and embeds historical non-admitted sizes. | Keep only mesh/degree cases used by final central experiments. Move the complete inventory to supplement. | P | Main/Supplement |

### p-Laplace

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| PL-01 | `paper/sections/benchmarks.tex:104-150` | The mathematical definition and strict discrete convexity are sound. | Retain. Add the short coercivity/injectivity argument that gives existence and uniqueness. Clarify that a computed vector only approximates the unique minimizer under its residual gate. | P/S | Main |
| PL-02 | `paper/sections/benchmarks.tex:151-194` | Historical energy/timing/backend figures are not the strongest current verification and use uncalibrated solver evidence. | Replace the result material with the independent manufactured study: $L^2$ rates 1.986/1.996/1.999, $H^1$ rates 0.990/0.998/0.999, and residual gate below $10^{-8}$. Publish numbers only after clean rerun; state that this does not prove L-shaped-corner regularity. | N-pilot | Main |

### Ginzburg--Landau

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| GL-01 | `paper/sections/benchmarks.tex:201-257` | The two discrete quadrature functionals and prescribed basin are correctly distinguished. | Retain in a shortened supplementary benchmark definition. Use `stationary endpoint from the prescribed initial state`, never `minimum`. | P/S | Supplement or secondary Main |
| GL-02 | `paper/sections/benchmarks.tex:259-290` | Plotted energy agreement between $\mathcal E_{h,2}$ and $\mathcal E_{h,4}$ is not fixed-functional verification. | Replace main verification with the source-extended manufactured case using the production-style three-point rule: $L^2$ rates 2.010/2.003/2.001 and $H^1$ rates 0.997/0.999/1.000 on the controlled positive branch. Keep backend energy comparison only as supplement. | N-pilot | Supplement/Verification |

### Hyperelasticity

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| HE-01 | `paper/sections/benchmarks.tex:295-385` | The density, admissible $J>0$ domain, affine lift, and P1 exact cell integration are suitable. | Retain compactly. State that the solver targets discrete equilibrium and that neither local minimality nor orientation preservation of every trial is established. | P/S | Main |
| HE-02 | `paper/sections/benchmarks.tex:387-420` | Historical 24-step endpoint-energy agreement does not establish common path, derivative correctness, or calibrated stationarity. | Replace with the affine patch verification and controlled fixed-state distribution gate. The affine patch gives production-versus-analytic relative errors $4.58\times10^{-15}$ (energy), $2.67\times10^{-15}$ (residual), and $4.38\times10^{-16}$ (Hessian), plus traction/objectivity checks. EXP-DIST gives exact topology/residual/matrix agreement and action/correction errors near $2.3\times10^{-16}$ for one versus two ranks. Use only after clean reruns. | N-pilot | Main |

### Plasticity2D

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| P2D-01 | `paper/sections/benchmarks.tex:424-669` | A 246-line formula section overwhelms the core paper; no global projection/envelope theorem establishes the potential derivative as a physical stress update. | Move the complete definition to supplement. In main text, if mentioned at all, call it a `2D synthetic Mohr--Coulomb-inspired endpoint surrogate` used for a solver diagnostic. | P for program; U physical interpretation | Supplement |
| P2D-02 | `paper/sections/benchmarks.tex:607-612` | The text says the principal angle is assigned at repeated stress and interprets derivatives away from degeneracy. The implementation now avoids `atan2(0,0)` on the selected apex branch and has finite focused AD tests. | Describe the invariant apex construction explicitly: selected hydrostatic apex stress is reconstructed without evaluating an irrelevant angle; the ordinary line branch retains the angle. State that switch regularity and generalized derivatives remain unestablished. | N-pilot | Supplement |
| P2D-03 | `paper/sections/benchmarks.tex:688-730` and `paper/sections/results.tex:327-342` | Historical endpoint and resolution/timing narrative is not KKT, physical failure, or calibrated convergence evidence. | Retain at most one field image labeled a synthetic endpoint diagnostic. Remove energy/timing trend and all `completed endpoint evidence` wording. | R/U | Supplement/Remove |

### Plasticity3D

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| P3D-01 | `paper/sections/benchmarks.tex:737-833` | Correctly distinguishes source incremental plasticity from the endpoint surrogate, but still devotes extensive continuum setup to a derivative benchmark. | Shorten to the exact discrete synthetic surrogate, constraints, material fields, and named quadrature. Keep source elastoplasticity as context only. | P/S | Conditional Main |
| P3D-02 | `paper/sections/benchmarks.tex:828-843` | Presents 1/11/24 rules and historical completed/scaling studies before the new quadrature finding. | Add the positive 125-point Duffy evaluator and state that no rule is exact for the branchwise nonpolynomial integrand. Remove all references to `completed nine-case` and auxiliary scaling evidence. | P/N-pilot | Conditional Main |
| P3D-03 | `paper/sections/benchmarks.tex:845-1066` | The program is defined carefully, but denominator assumptions, tie-break limitations, and branch regularity are not fully stated. | Add $E>0$, $-1<\nu<1/2$, $-1<s_\lambda<1$, and $s_\lambda\ne0$ for the apex; state denominator positivity. Explain that the coordinate-dependent $10^{-15}\operatorname{diag}(0,1,2)$ tie break is not an objective spectral regularization. Use ordinary Hessian only at fixed branch/simple spectrum. | P/N-pilot | Conditional Main/Supplement |
| P3D-04 | `paper/sections/benchmarks.tex:1061-1066` | `branch interfaces remain nonsmooth` is stronger than the evidence: Hessian jumps are observed, but the global regularity class is unproved. | Say `no differentiability, semismoothness, or generalized-Jacobian claim is made at branch interfaces`; report the two-sided observations descriptively in supplement. | U/N-pilot | Main/Supplement |
| P3D-05 | `paper/sections/benchmarks.tex:1068-1134` | Correct endpoint-surrogate boundary, but `continuation studies` may imply history evolution. | Use `strength-reduction sequence of reset-history endpoint problems`. Define surrogate stress only, if needed, as $\partial_\varepsilon\Phi_{\mathrm{MC},3D}$. | P | Conditional Main |
| P3D-06 | `paper/sections/benchmarks.tex:1136-1212` | All representative endpoint, convergence, degree, and timing material is historical and blocked by quadrature/stopping gates. | Delete from main. Replace with: fixed-element P1/P2/P4 branch-interior derivative errors; P2 all-five-branch fixed-state tangent-action agreement; EXP-MC interior/interface/rotation/degeneracy diagnostics; and fixed-state quadrature sensitivity. Every number remains provisional until clean rerun. | N-pilot/R historical | Conditional Main/Supplement |

### Topology

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| TO-01 | `paper/sections/benchmarks.tex:1217-1232` | The scope disclaimer is correct, but topology still occupies a full main benchmark and contributes no derivative-placement result. | Move the family to supplement. State the exact limited role: a distributed frozen-design software diagnostic with no KKT-quality endpoint. | Decision fixed | Supplement |
| TO-02 | `paper/sections/benchmarks.tex:1234-1421` | Definitions are detailed and the corrected material measure/fraction distinction is important, but this is not the paper's central optimization result. | Retain in supplement. Add the proved local first-order tangency of the reciprocal compliance model; explicitly state that it is not a global majorizer or convergence theorem. | P | Supplement |
| TO-03 | `paper/sections/benchmarks.tex:1423-1495` and `paper/sections/results.tex:568-637` | Historical adaptive results stop before final continuation, lack KKT residuals, and include rank-dependent paths. | Replace with the semantics-v2 three-iteration one-/two-rank unit/parity pilot only if a supplementary implementation diagnostic is desired: target fraction 0.2/measure 0.4, final fractions about 0.17452, all design solves capped, no feasibility or optimality claim. | N-pilot | Supplement |

## 9. Verification section ledger

The present section should be rebuilt rather than patched around the two weak
external companions.

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| V-01 | `paper/sections/validation.tex:3-6` | Opens with JAX-FEM and P3D comparator studies rather than independent verification. | Open with the verification hierarchy: analytic/manufactured checks; same-functional route checks; canonical distributed checks; only then external companions. Rename the section `Verification` unless physical validation evidence is added. | Decision fixed | Main |
| V-02 | `paper/sections/validation.tex:8-53` | Norms are historical unweighted coefficient/sample norms and gates are coarse regression thresholds. | Define each norm by scientific object. Use Riesz dual residual/primal state norms for endpoint accuracy, $L^2/H^1$ for manufactured errors, Frobenius/action norms for derivatives, and canonical weighted state norms for distribution. Keep coefficient norms as diagnostics. | P infrastructure; calibration U | Main |
| V-03 | Missing from current section | Smooth element derivative evidence is absent. | Add EXP-DERIV: five smooth local cases and P1/P2/P4 fixed-element checks, with same-functional route errors near roundoff and explicit branch margins. State that the report's fixed-element block alone does not establish MPI ownership. | N-pilot | Main |
| V-04 | Missing from current section | Independent spatial and analytic mechanics checks are absent. | Add EXP-VAL: manufactured p-Laplace and GL rates, the hyperelastic affine patch, and nonaffine hyperelastic displacement/deformation/stress rates. State exact limitations: smooth unit square, controlled GL basin, manufactured full-Dirichlet mechanics, and no production rotating-beam or matched-backend validation. | N-pilot | Main |
| V-05 | Missing from current section | Canonical distributed equality is absent. | Add EXP-DIST one-/two-rank fixed-state hyperelastic pilot: exact topology/maps/CSR, energy error $8.67\times10^{-19}$, zero stored residual/matrix difference, action $2.24\times10^{-16}$, correction $2.33\times10^{-16}$. Mark four ranks, factorized construction variants, nonlinear endpoints, and clean rerun open. | N-pilot | Main |
| V-06 | Missing from current section | The strongest current branch evidence is absent. | Add P2 all-five-label fixed-state action agreement and EXP-MC. Report five interiors, 15 two-sided interface pairs, 15 rotations, seven repeated spectra, and the explicit one-sided Hessian jumps. State that the NumPy code transcribes the same law and that no interface regularity or independent constitutive validation follows. | N-pilot | Conditional Main/Supplement |
| V-07 | `paper/sections/validation.tex:55-128` | JAX-FEM comparison uses a different volumetric law and a permissive 5% regression gate; common post-evaluated energy is not same-functional derivative verification. | Move intact evidence to supplement, retitle `External endpoint companion`, and remove it from the paper's verification chain. | N-historical | Supplement |
| V-08 | `paper/sections/validation.tex:130-197` | P3D comparator is unavailable for rerun, transcribes the same endpoint formulas, and one block changes constraints. | Replace main text with EXP-MC/fixed-state route verification. Keep only a short provenance limitation in supplement if the comparator is retained. | N-historical | Supplement/Remove |
| V-09 | `paper/sections/validation.tex:199-244` | Figures/tables visually elevate weak endpoint companions to validation evidence. | Remove from main. Do not regenerate unless the supplementary narrative needs them and the provenance gap is explicit in the caption. | N-historical | Supplement/Remove |

## 10. Results section ledger

Until clean central experiments exist, rename this section `Verification Results
and Pilot Adjudication` in the working draft or keep publication-result slots
empty. A journal submission must not present dirty-worktree pilots as final
evidence.

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| R-01 | `paper/sections/results.tex:3-28` | Announces performance/scaling evidence and normalizes one observation per configuration. | Replace with a strict admission paragraph: correctness before timing; clean run records; equal accuracy; failure/censoring; repetitions and intervals. Regenerate `tab:numerical-protocol-summary` from current cards. | Protocol fixed | Main |
| R-02 | `paper/sections/results.tex:30-90` | Old production bundles cannot isolate globalization and do not satisfy the current exact algorithm/failure-reporting contract. | Main text may report only the controlled-v2 qualitative observation: on one two-rank GL instance under common GMRES/Hypre, Newton--Armijo fails at the first iteration after 40 rejected trials and reduced-subspace trust--Armijo reaches the recorded criteria in 12 iterations. Do not report solve times, robustness probabilities, or endpoint equality. Prefer supplement unless globalization remains a research question. | N-pilot | Supplement/Secondary Main |
| R-03 | `paper/sections/results.tex:92-136` | Claims endpoint/route cost from displayed energy and iteration equality. | Replace first with a fixed-state correctness table: exact state hashes, branch counts/margins, residual/matrix/action errors. Add timing only in a separate table after clean repetitions. | R historical; N-pilot replacement | Main |
| R-04 | `paper/sections/results.tex:111-120` | `constitutive AD has the lowest reported time` and interpretation of 240--420 colors are inadmissible central conclusions. | Delete. Provisional replacement: P1 and P2 local pilots show roundoff tangent-action agreement; descriptive timing suggests a candidate route, but no ranking is admitted. Final wording depends on held-out Karolina evidence. | U | Main placeholder |
| R-05 | `paper/sections/results.tex:140-200` | p-Laplace and GL scaling rely on historical single observations and noncommon backend functionals. | Remove. Put manufactured convergence in verification; restore scaling only through a fixed-policy, repeated, equal-accuracy campaign. | R/U | Remove |
| R-06 | `paper/sections/results.tex:202-325` | Hyperelastic completed-path, memory, MG, and scaling conclusions predate the factorized distribution and fixed-policy protocols. | Main text currently admits only fixed-state one-/two-rank algebraic equivalence. Restore route/memory/scaling conclusions only after EXP-DIST and EXP-SCALE terminal decisions. | N-pilot/U | Main placeholder/Supplement |
| R-07 | `paper/sections/results.tex:327-342` | Calls one Plasticity2D row endpoint evidence and reports capped timings as solver evidence. | Remove from main. If retained in supplement, label every row fixed-work/legacy and make no stationarity, failure, or physical claim. | R/U | Supplement |
| R-08 | `paper/sections/results.tex:344-417` | Entire headline P3D route ranking fails stopping, state-equivalence, provenance, and repetition gates. | Remove. Replace only after EXP-ROUTE-001 Tier B passes with a calibrated common stop and clean repeated rows. | R | Remove/Main placeholder |
| R-09 | `paper/sections/results.tex:417-467` | Degree/energy narrative is contradicted by fixed-state quadrature derivatives. | Replace with EXP-DISC sensitivity: small energy differences coexist with order-one residual-vector differences and 2.08%/4.35% action differences; hence both enriched-rule problems must be solved. This is a diagnostic/limitation, not a discretization result. | N-pilot | Conditional Main/Discussion |
| R-10 | `paper/sections/results.tex:469-564` | P3D strong/converged scaling claims are blocked by stopping/quadrature and policy changes. | Delete. Optional clean P3D scaling can enter only after DERIV/STOP/DISC; otherwise omit it entirely. | R/U | Remove |
| R-11 | `paper/sections/results.tex:568-637` | Topology adaptive timing and rank sensitivity do not support core derivative placement or optimization. | Move to supplement and use only semantics-v2 corrected fixed-work results. | N-pilot/U | Supplement |
| R-12 | `paper/sections/results.tex:639-651` | Synthesis claims multiple completed evidence blocks that are no longer admitted. | Rewrite only after terminal experiment decisions. The final synthesis must distinguish proved equivalence, verified tested states, a finite route map, and unresolved branches/architectures. | U | Main |

### Minimum replacement result tables

Generate these in order, each from a script and strict manifest.

1. **Verification table**: p-Laplace and GL manufactured rates; hyperelastic
   analytic patch errors and nonaffine displacement/deformation/stress rates;
   exact scope limitation per row.
2. **Route correctness table**: fixed state/hash, degree/rule, branches and
   margins, element-versus-constitutive residual/action errors, colored-recovery
   error, symmetry, distribution/rank gate.
3. **Quadrature sensitivity table**: solve rule versus common evaluator for
   energy, Riesz residual, action, branch margin, and endpoint state. The current
   fixed-state pilot may be a limitation table, but not the final solve table.
4. **Cost-model table/figure**: measured covariates and held-out ordering error;
   include censored routes visibly.
5. **Equal-accuracy full-solve table**: only admitted route/state groups, with
   Riesz residual/correction, observables, work, repeated timing, interval, and
   hardware/provenance.
6. **Fixed-policy scaling figure**: only if EXP-SCALE-001 passes the endpoint and
   common-policy gates.

## 11. Discussion, conclusion, and appendix ledger

### Discussion

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| D-01 | `paper/sections/discussion.tex:3-7` | Generic methodological conclusion is plausible but not yet tied to an admitted cost matrix. | Keep as a hypothesis in the introduction; in discussion, state only the interactions actually observed in clean admitted experiments. | U | Main |
| D-02 | `paper/sections/discussion.tex:9-19` | Says reported stopping documents stationarity/equilibrium. Historical central endpoints do not pass the new metric. | Retain the mathematical distinction but replace empirical language: only rows passing the Riesz residual and correction gate are approximate stationary/equilibrium endpoints. Identify all other rows as fixed-work or historical. | P/U | Main |
| D-03 | `paper/sections/discussion.tex:21-31` | Repeats the inadmissible P4 timing headline. | Delete the first paragraph's empirical ranking. Replace with the proved fixed-branch identity, tested fixed-state errors, and explicit switch exclusions. Add route-cost conclusions only after EXP-ROUTE. | P/N-pilot/U | Main |
| D-04 | `paper/sections/discussion.tex:33-45` | Gives weak external companions equal weight with verification. | Move the companions to one limitations paragraph or supplement. Discuss independent manufactured/analytic checks and branch diagnostics first. | N-pilot/N-historical | Main/Supplement |
| D-05 | `paper/sections/discussion.tex:47-53` | Correctly warns about confounded degree/quadrature, but understates the new result. | Replace with the stronger conclusion: energy-only agreement was empirically inadequate because residual and action changed materially; old degree trends are withdrawn pending separate solves. | N-pilot | Main |
| D-06 | `paper/sections/discussion.tex:55-65` | Calls historical P3D series scaling evidence and a converged benchmark. | Delete P3D scaling conclusion. Keep topology only in supplementary limitations. | R | Remove/Supplement |
| D-07 | `paper/sections/discussion.tex:67-77` | Acknowledges incomplete timing provenance but still permits within-allocation conclusions. | State that such rows are excluded from publication inference. The final limitation paragraph should cover tested architectures, branch interiors, synthetic constitutive status, quadrature reference status, and empirical selector scope. | Protocol fixed | Main |

### Conclusion

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| C-01 | `paper/sections/conclusion.tex:3-8` | Opens with software toolset breadth and says controlled comparisons already exist. | Lead with the conditional derivative-placement result and its assumptions. Mention the implementation only as the realization used to test it. | P/U | Main |
| C-02 | `paper/sections/conclusion.tex:10-18` | Repeats the invalid $222/288/372$ s headline and same-observable claim. | Delete entirely. Insert the final fixed-state/equal-accuracy/held-out route result only after clean campaigns. If those fail, conclude with a finite empirical map and the observed censoring. | R/U | Main |
| C-03 | `paper/sections/conclusion.tex:20-27` | Gives JAX-FEM and endpoint transcription prominence despite their weak independence. | Replace with independent manufactured/analytic verification and the scoped P3D branch evidence. State no physical validation or generalized switch result. | N-pilot | Main |
| C-04 | `paper/sections/conclusion.tex:27-31` | Future work lists items that are actually present publication gates, such as repetitions and matched comparisons. | Do not submit while central gates are described as future work. Retain only genuine extensions beyond the accepted claim, e.g. additional architectures, incremental-history mechanics, and theoretically justified switch treatment. | U | Main |

### Appendix and diagrams

| ID | Exact source | Current issue | Admissible replacement | Status | Destination |
| --- | --- | --- | --- | --- | --- |
| AP-01 | `paper/sections/appendix.tex:3-61` | Legacy P3D timing/preconditioner tables use loose coefficient stops and incomplete derivative-equivalence evidence. | Remove from submission supplement. A new supplement may contain full clean route/policy matrices after the common gates pass, including censored rows. | R/U | Remove |
| AP-02 | `paper/sections/appendix.tex:65-88` | Reference continuation has different endpoints/Krylov totals and is not a verification baseline. | Remove unless a narrowly labeled provenance appendix is scientifically necessary. It contributes no central derivative-placement evidence. | N-historical | Remove |
| AP-03 | `paper/sections/tikz_diagrams.tex:27-44` | Framework diagram presents a broad toolset and topology. | Replace with the derivative-placement/equivalence diagram or remove. | Editorial | Main |
| AP-04 | `paper/sections/tikz_diagrams.tex:47-79` | Derivative diagrams are useful but omit identical-functional assumptions and PETSc ownership conditions. | Merge into one compact diagram with a central `same discrete functional/state` box and an `owned-row PETSc operator` output. | P | Main |
| AP-05 | `paper/sections/tikz_diagrams.tex:82-90` | Globalization labels are too generic for the exact algorithm contract. | Use the three exact method names and show that the stopping Riesz metric is separate from the coefficient-Euclidean trust norm. | Implementation documented | Supplement/Main if retained |

## 12. Provisional evidence that may replace historical material

These numbers are suitable for planning tables and prose, but they must not be
presented as final publication results until clean, immutable reruns reproduce
them under the relevant card.

| Evidence block | Current pilot observation | Permitted provisional interpretation | Still open |
| --- | --- | --- | --- |
| EXP-DERIV-001 | Smooth independent contractions pass; P1/P2/P4 fixed-element residual and Hessian errors are approximately $10^{-16}$, with fixed elastic branches and recorded margins. | Fixed-element identity and centered derivative consistency on tested smooth/elastic branch-interior states. | Clean rerun; full distributed route matrix; independent global residual. |
| P2 fixed-state route screen | Mixed P2 state contains all five labels; colored action is bitwise equal to element AD and constitutive action differs by $2.34\times10^{-16}$ relative; no sample is in the $10^{-8}$ switch band. | Assembled tangent-action agreement at one constructed branch-interior state. | Clean and multi-rank rerun; independently assembled residual; no switch theorem. |
| EXP-MC-001 | Five strict interiors, 15 interface pairs, 15 rotations, and seven repeated-spectrum cases pass their frozen gates; selected Hessians change by about 0.396--1 across closest interface pairs. | Internal branch-program verification and descriptive switch diagnostics. | Independent constitutive validation, generalized differentiability, clean rerun. |
| EXP-DISC-001 v2 | P2/P4 solve-rule energies differ little from 125-point evaluation, but residuals and actions change materially. | Energy-only quadrature adequacy is rejected; common-evaluator and own-rule solve gates are mandatory. | 24/125-point solved endpoints, Riesz residuals, mesh/tolerance separation, clean run. |
| EXP-DIST-001 | One-/two-rank hyperelastic fixed-state canonical algebra agrees at or near roundoff. | Rank partition did not alter the tested canonical fixed-state objects. | Four ranks, factorized construction variants, nonlinear endpoints, timing/memory, clean run. |
| EXP-VAL-001 | Expected P1 manufactured rates for p-Laplace and controlled GL; analytic hyperelastic patch agrees near roundoff; the four-level nonaffine manufactured case gives last-pair rates 1.887 (displacement $L^2$), 1.006 (deformation gradient), and 0.983 (first Piola), with minimum $J_h=0.844$. Order-4/6/8 load checks bound the maximum response contribution by $8.29\times10^{-6}$ of the FE error. | Smooth spatial, analytic patch, and independent nonaffine manufactured-formulation verification. | Clean run; no rotating-beam validation. A matched DOLFINx backend is optional and outside the required claim set. |
| EXP-GLOB-001 controlled v2 | One GL Newton--Armijo row fails on the first iteration; reduced-subspace trust--Armijo reaches its stored criteria under the same ordinary KSP contract. | One controlled robustness observation and verified failure semantics. | Canonical endpoints, independent residuals, Riesz stopping, distinct instances, repetitions; no timing claim now. |
| EXP-ROUTE P1/P2 pilots | Exact state hashes and near-roundoff actions remain useful derivative diagnostics. Under the tightened record contract, all 12 legacy rows are rejected before route-map admission because they do not prove the timing value from raw per-rank collective maxima. Zero training and zero holdout rows are publication-model eligible. | No admitted finite route map, speedup, crossover, predictive selector, or publication timing claim. | Prespecified paired train/holdout matrix, workstation and second-architecture blocks, raw-rank timing proof, factor integration, equal-accuracy full solves, and clean repetitions. |
| EXP-STOP-001 | Tightening P1 KSP tolerance from $10^{-2}$ to $10^{-8}$ reduces one-step route state disagreement to $5.28\times10^{-11}$. | Historical route discrepancies are consistent with inexact linear solves. | Cross-degree Riesz calibration, true-residual sweeps, full nonlinear endpoints. |
| EXP-TOPO-001 | Semantics-v2 one-/two-rank three-step rows agree closely but are infeasible and capped. | Material-measure/fraction unit and distributed fixed-work parity check. | KKT/feasibility/baseline campaign; outside core scope. |

### Stopping-control synchronization check

The current control documents agree on the relevant distinction.
`paper/protocols/EXP-STOP-001.md:13-20` and `:69-82` record the implemented
Plasticity3D reference-elastic Riesz map, SPD/inertia and norm-solve checks, and
true-residual path. `paper/mathematical_status_and_claim_dictionary.md:235-247`
and `:600-606` record the same P1 infrastructure result. Route-runner
integration and P1/P2/P4 tolerance/cross-mesh calibration remain open. The
manuscript must preserve that distinction: **map implemented** does not mean
**publication stopping policy calibrated**, and no historical endpoint is
retrospectively upgraded.

## 13. Ordered rewrite workflow

1. **Freeze admissible evidence.** Record terminal status for every central
   protocol. Do not write final timing language from pilots.
2. **Quarantine invalid assets.** Remove the inputs listed in Section 4 from
   `main.tex`/section sources, then update the table/figure generation scripts
   and manifests so they cannot silently return.
3. **Write the mathematical core.** Add the fixed-functional element/
   constitutive proposition and the distributed colored-recovery proposition,
   including proofs and exclusions.
4. **Rewrite title, research questions, and contribution list.** Use the route
   decision, not the old six-family toolset framing.
5. **Rebuild verification.** Insert manufactured, analytic patch, fixed-element,
   assembled-action, distribution, and branch-diagnostic blocks in the hierarchy
   in Section 9. Keep pilot labels in the working draft until clean reruns exist.
6. **Reduce benchmarks.** Keep p-Laplace and hyperelasticity central;
   Plasticity3D remains conditional. Move GL globalization, Plasticity2D,
   topology, and weak external companions to the supplement.
7. **Insert clean route results.** First correctness, then cost model and
   held-out assessment, then equal-accuracy full solves. Never combine these in
   one table without explicit gates.
8. **Insert scaling only if admitted.** EXP-SCALE-001 must use one fixed policy,
   endpoint/accuracy gates, repetitions, and uncertainty.
9. **Write discussion and conclusion from terminal outcomes.** Do not preserve a
   desired ranking if the result is an empirical map, censoring, or failed
   selector.
10. **Write the abstract last.** Include only results already present and
    admitted in the body.
11. **Run a claim-word audit.** Check every occurrence of `minimizer`,
    `equivalent`, `exact`, `stress`, `tangent`, `converged`, `validation`,
    `strong scaling`, `faster`, and `optimization solution` against the claim
    dictionary.
12. **Perform release QA.** Regenerate assets, build the PDF, inspect every page,
    verify links/citations/labels, validate strict run records and manifests, and
    replace all future-tense release placeholders with actual archival metadata.

## 14. Historical audited source snapshot

The historical line references in this ledger were checked against these
SHA-256 values. They are not hashes of the current rewritten sources.

| Source | SHA-256 |
| --- | --- |
| `paper/main.tex` | `002914586bed8ef3d031dc58736b7b9159d3da6e0f79657c1919ab34a5e6847c` |
| `paper/sections/abstract.tex` | `7cda1568b54bc318ca65ec31fd3aeb3eded00d72d044834613d9d84e847cfbaf` |
| `paper/sections/appendix.tex` | `16e977c69b9fb18cf9e104109766f8e7ce72fd543cbcb71b3db3189b81b1fc4e` |
| `paper/sections/benchmarks.tex` | `ea05f16143eb31aace3d962b68a99c439397b005ba90cc9bed74713729fabb0b` |
| `paper/sections/conclusion.tex` | `960b217674545519e8d864c3d748ff921a941ced9342f042dbb033f3aa02a945` |
| `paper/sections/discussion.tex` | `02ea9415812f79f2317979764e77b01f316e0fdc6e0920994cb8d708564184fe` |
| `paper/sections/implementation.tex` | `95506dbf95937a3c30a139df0ecff8123cebe6a9cae328260a10aea7f239db6f` |
| `paper/sections/introduction.tex` | `4adcacf76a97e8749b124a5233c8d83475bcddadb94501b64baf487b6b7fefd8` |
| `paper/sections/methodology.tex` | `4038865c8fb468065cf37a99377bb7926f8cd2acb4e47377f8b2c14b632ded39` |
| `paper/sections/related_work.tex` | `d7e61d75655a6cd4324b47435d536c118edf027d9eadef60d513eb70bedcbef5` |
| `paper/sections/results.tex` | `77b32b6f2d626beecb6bffc09e9de0f6eae210fa1f4030ff17251f6ca518230b` |
| `paper/sections/tikz_diagrams.tex` | `ce3da902c28498ba9186a9ccc454e1baac0c17e6304e9f2bf1094da9962dd18d` |
| `paper/sections/validation.tex` | `df741ef319527ea108e9da4813f4858153b63c647f7a59e03f52699f9564f98f` |
