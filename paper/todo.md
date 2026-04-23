# Paper Submission TODO

## Current Position

This project looks publishable as a strong software-and-methods paper in scientific computing / computational mechanics. It is less convincing if framed as a broad mechanics breakthrough or as a theory-first numerical analysis paper. The best current thesis is:

`A solver-centric JAX + PETSc workflow for distributed nonlinear finite-element energy minimization, with multiple derivative routes and a reproducible benchmark suite across several difficult problem families.`

The goal of the next revision should be to make that claim precise, well-supported, and easy for a reviewer to accept.

## Main Weak Points To Fix

1. The novelty boundary is still vulnerable.
   Reviewers can still ask what is fundamentally new relative to `JAX-FEM`, `AutoPDEx`, `cashocs`, `FEniTop`, `dolfin-adjoint` / `pyadjoint`, and newer `FEniCSx`-based workflows.

2. The paper is broad, but some stories are not yet deep enough.
   The benchmark spread is good, but the evidence per benchmark family is still uneven. The paper needs one or two decisive main-text results, not only broad coverage.

3. Some comparisons are not fully symmetric.
   Different load factors, historical campaigns, end-to-end timings, and not-always-identical solver settings create easy reviewer objections.

4. The plasticity story is interesting, but also the most fragile.
   The paper correctly describes the repository formulations as surrogate scalarizations, but it still needs one compact validation step that shows what these surrogates preserve relative to a standard incremental reference.

5. The results are currently more descriptive than decisive.
   The paper needs at least one flagship ablation table and one clean cross-framework comparison on a shared benchmark.

6. The venue positioning is not fixed enough yet.
   Right now the draft could be read as software, scientific computing, optimization, or mechanics. That flexibility is useful, but it weakens the submission strategy if the paper text is not tailored to one main venue family.

## Step-By-Step Improvement Plan

### Phase 1: Lock the thesis and target venue

1. Freeze the main claim in one sentence.
   Why: every section should support the same claim, and right now the draft still allows broader readings than the evidence fully supports.
   Output: one sentence inserted consistently into the abstract, end of the introduction, and conclusion.
   Files: `paper/sections/abstract.tex`, `paper/sections/introduction.tex`, `paper/sections/conclusion.tex`

2. Choose one flagship benchmark family and one secondary support story.
   Recommended default:
   `Plasticity3D` as the flagship solver-and-derivative story.
   Topology optimization as the secondary workflow / optimization story.
   Why: these are the most distinctive parts of the repository.
   Output: a short note at the top of `results.tex` and a re-ordered figure / table flow.
   Files: `paper/sections/results.tex`, `paper/sections/discussion.tex`, `paper/sections/appendix.tex`

3. Decide the primary venue track before further rewriting.
   Default venue track:
   scientific computing / computational methods.
   Alternate venue track:
   computational mechanics, only if the mechanics comparisons are strengthened.
   Why: the abstract, title, and contribution paragraph should be venue-shaped.
   Output: one chosen venue family and one backup family written at the top of this TODO when the decision is made.

4. Rewrite the title and abstract only after steps 1-3 are frozen.
   Why: doing this earlier causes churn.
   Output: title options tailored to the selected venue family.
   Files: `paper/main.tex`, `paper/sections/abstract.tex`

### Phase 2: Strengthen the core evidence

1. Add one flagship ablation table on a single benchmark with fully matched conditions.
   Recommended benchmark: `Plasticity3D`.
   Compare exactly:
   element AD, constitutive AD, colored SFD.
   Hold fixed:
   mesh family, rank count, nonlinear tolerances, line search / trust-region settings, linear solver settings, load factor, stopping rules, and hardware.
   Report at minimum:
   total wall time, assembly time, nonlinear iterations, linear iterations, final residual norm, final objective / energy, and memory if available.
   Why: this is the single highest-ROI addition for reviewer confidence.
   Output: one main-text table and one appendix table.
   Files: `paper/sections/results.tex`, `paper/sections/appendix.tex`, `paper/tables/generated/`, `paper/scripts/generate_paper_tables.py`

2. Add one external baseline on a benchmark that can be reproduced fairly.
   Good candidates:
   `JAX-FEM` on a shared small nonlinear mechanics problem.
   `AutoPDEx` on a JAX-native PDE problem.
   `FEniTop` on a topology problem.
   Rule: do not force a weak baseline. Only include a direct comparison if the benchmark, mesh, objective, and stopping rules can be matched closely enough to survive reviewer scrutiny.
   Why: one good external baseline is more valuable than many loose comparisons.
   Output: one compact comparison table and one short fairness paragraph.
   Files: `paper/sections/related_work.tex`, `paper/sections/results.tex`, `paper/sections/discussion.tex`

3. Add a plasticity surrogate validation subsection.
   Minimum version:
   take a smaller plasticity case and compare the repository surrogate objective path against a standard incremental elastoplastic reference on observable quantities such as load-displacement trend, localization pattern, and final state statistics.
   Why: this closes the most likely scientific-rigor objection.
   Output: one figure or table plus a short explanatory paragraph.
   Files: `paper/sections/benchmarks.tex`, `paper/sections/results.tex`, `paper/sections/discussion.tex`

4. Separate apples-to-apples results from non-symmetric evidence.
   Create two explicit result classes:
   `Matched comparisons` and `Context / historical comparisons`.
   Why: this prevents reviewers from reading every result as equally direct.
   Output: subsection headers and one explicit note at the start of each result family.
   Files: `paper/sections/results.tex`, `paper/sections/discussion.tex`

### Phase 3: Make the comparisons reviewer-proof

1. Add a `Fairness and limitations` subsection near the end of Results or at the start of Discussion.
   It should explicitly list:
   what is matched,
   what is only approximately comparable,
   what comes from historical runs,
   what is repository-specific.
   Why: saying this first reduces the chance that a reviewer frames it as concealment.
   Files: `paper/sections/discussion.tex`

2. Re-check every sentence that claims speed, robustness, scalability, or accuracy.
   Search terms to audit:
   `robust`, `competitive`, `credible`, `substantial`, `strong scaling`, `exact`, `argmin`.
   Why: these are classic reviewer trigger words.
   Output: either a direct supporting result / citation, or a softer wording.
   Files: all manuscript sections, especially `paper/sections/results.tex` and `paper/sections/discussion.tex`

3. Make the SOTA table maximally factual.
   Allowed columns:
   modeling layer,
   differentiation route,
   sparse / distributed solver path,
   second-order information,
   benchmark family coverage.
   Avoid:
   vague `yes / partial / limited` judgments unless each entry is defined explicitly.
   Files: `paper/sections/introduction.tex`, `paper/sections/related_work.tex`

4. Re-check the claim audit after every major rewrite.
   Rule:
   every externally sourced scientific statement in the main text must map to an exact page or section in `claim_audit.md`.
   Files: `paper/literature/claim_audit.md`

### Phase 4: Tighten the manuscript structure

1. Keep only the strongest narrative arc in the main text.
   Recommended order:
   architecture and derivative strategy,
   flagship benchmark evidence,
   one broader workflow benchmark family,
   limitations and positioning.
   Why: too many equal-weight stories make the paper feel diffuse.

2. Move supporting but non-critical detail to the appendix.
   Good appendix candidates:
   extended parameter sweeps,
   secondary mesh tables,
   extra timing breakdowns,
   sensitivity checks,
   less central benchmark visuals.
   Files: `paper/sections/appendix.tex`

3. Make repository-specific modeling choices explicit everywhere.
   This is especially important for:
   plasticity surrogate functionals,
   topology regularization and move-penalty choices,
   any benchmark-specific load path or continuation schedule.
   Why: the paper should never look like it is attributing repository choices to classical references.
   Files: `paper/sections/benchmarks.tex`, `paper/sections/methodology.tex`

4. End with a modest and precise conclusion.
   The conclusion should emphasize:
   maintained workflow,
   reproducibility,
   distributed sparse solve capability,
   multiple derivative routes,
   hard benchmark coverage.
   It should not claim universal superiority or a new general theory.
   Files: `paper/sections/conclusion.tex`

### Phase 5: Reproducibility and artifact polish

1. Make the compute environment explicit.
   Include:
   commit hash,
   Python / JAX / PETSc versions,
   machine type,
   CPU count,
   MPI layout,
   main run commands.
   Why: software / methods venues care about this a lot.
   Output: one reproducibility paragraph in the paper and one artifact note in the repository.
   Files: `paper/sections/methodology.tex`, repository root or `paper/`

2. Ensure every main-text figure and table can be regenerated from a documented command.
   Good target:
   one command per paper artifact family.
   Files: `paper/Makefile`, `paper/scripts/generate_paper_tables.py`, figure-generation scripts

3. Re-run the literature workflow before submission.
   Update:
   `paper/literature/manifest.json`
   `paper/literature/sources.md`
   `paper/literature/claim_audit.md`
   Why: the paper now depends heavily on citation rigor.

4. Prepare a short reviewer-facing repository note.
   Include:
   where the benchmarks live,
   how to regenerate paper tables,
   where the literature audit is stored,
   what is intentionally omitted because of runtime cost.
   Why: this reduces friction during review.

### Phase 6: Final pre-submission audit

1. Run a language pass focused only on overclaiming.
   Search again for:
   `novel`, `state of the art`, `exact`, `robust`, `competitive`, `strong scaling`, `substantial`.

2. Run a comparison pass focused only on fairness.
   For every baseline:
   ask whether the problem definition, mesh, solver settings, stopping rules, and hardware are really comparable.

3. Run a citation pass focused only on locator precision.
   Replace broad cites with pinpoint cites where a definition, algorithm, or benchmark detail depends on a specific place in the source.

4. Rebuild the paper from scratch.
   Target:
   `latexmk -pdf main.tex`
   Confirm:
   no undefined citations,
   no bibliography errors,
   no broken table inputs,
   no accidental figure drift.

## High-ROI Changes If Time Is Short

If there is only time for a small number of changes, do these first:

1. Add the single flagship ablation table on `Plasticity3D`.
2. Add one external baseline on a truly shared benchmark.
3. Add a short `Fairness and limitations` subsection.
4. Validate the plasticity surrogate on one smaller reference case.
5. Rewrite the abstract, introduction, and conclusion around one narrow thesis.

These five items will improve acceptance odds much more than adding more benchmark breadth.

## Journal Shortlist

Ranking snapshot used below:
`2024` JCR metrics from Clarivate category exports, dataset updated `2025-06-18`, checked on `2026-04-23`.

Filter used:
include only journals with a math-related WoS category used for the filter (`Mathematics, Applied` or `Mathematics, Interdisciplinary Applications`) and `Q1` in both `2024 JIF` and `AIS` within that category.

Notes:
- `MJL` = official Clarivate Master Journal List page.
- `JCR` = the public category export used to verify `JIF`, `AIS`, and quartiles.
- `SCImago` = secondary ranking / classification site.
- Rankings change every year, so re-check all metrics again immediately before submission.

| Journal | Fit For This Paper | WoS Category Used For Filter | 2024 JIF | JIF Q | AIS | AIS Q | Classifications | Check Links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Journal of Scientific Computing | Best current overall fit for a rigorous scientific-computing / methods paper with broad benchmark coverage | Mathematics, Applied | 3.3 | Q1 | 1.252 | Q1 | WoS: Mathematics, Applied<br>SCImago: Applied Mathematics; Computational Mathematics; Numerical Analysis; Software; Computational Theory and Mathematics | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0885-7474&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=23490&tip=sid) |
| ACM Transactions on Mathematical Software | Excellent if the revision foregrounds software architecture, reproducibility, and benchmark infrastructure | Mathematics, Applied | 3.2 | Q1 | 1.494 | Q1 | WoS: Mathematics, Applied<br>SCImago: Applied Mathematics; Software | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0098-3500&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=18120&tip=sid) |
| SIAM Journal on Scientific Computing | Strong option if the paper sharpens the nonlinear solver, sparse linear algebra, and parallel performance story | Mathematics, Applied | 2.6 | Q1 | 1.669 | Q1 | WoS: Mathematics, Applied<br>SCImago: Applied Mathematics; Computational Mathematics | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=1064-8275&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=26425&tip=sid) |
| Computational Mechanics | Best mechanics-leaning target if `Plasticity3D` and hyperelastic / mechanics validation become the main story | Mathematics, Interdisciplinary Applications | 3.8 | Q1 | 1.033 | Q1 | WoS: Mathematics, Interdisciplinary Applications<br>SCImago: Applied Mathematics; Computational Mathematics; Computational Mechanics; Mechanical Engineering | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0178-7675&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21996&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?q=28457&tip=sid) |
| Applied Mathematical Modelling | Good if the paper is positioned as an applied nonlinear modeling and optimization workflow paper rather than a software artifact paper | Mathematics, Interdisciplinary Applications | 5.1 | Q1 | 0.925 | Q1 | WoS: Mathematics, Interdisciplinary Applications<br>SCImago: Applied Mathematics; Modeling and Simulation | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0307-904X&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21996&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=28065&tip=sid) |
| Computer Methods in Applied Mechanics and Engineering | Stretch target; realistic only after stronger mechanics depth, cleaner external baselines, and a more decisive flagship result | Mathematics, Interdisciplinary Applications | 7.3 | Q1 | 1.801 | Q1 | WoS: Mathematics, Interdisciplinary Applications<br>SCImago: Computational Mechanics; Computer Science Applications; Mechanical Engineering; Mechanics of Materials | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0045-7825&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21996&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=18158&tip=sid) |
| Advances in Computational Mathematics | Plausible if the paper is tightened around computational methodology and keeps engineering benchmarking secondary | Mathematics, Applied | 2.1 | Q1 | 0.874 | Q1 | WoS: Mathematics, Applied<br>SCImago: Applied Mathematics; Computational Mathematics | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=1019-7168&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?q=28041&tip=sid) |
| Computational Optimization and Applications | Optimization-tilted fallback if the revision pushes topology, globalization, and second-order optimization much harder | Mathematics, Applied | 2.0 | Q1 | 1.145 | Q1 | WoS: Mathematics, Applied<br>SCImago: Applied Mathematics; Computational Mathematics; Control and Optimization | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0926-6003&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=28459&tip=sid) |
| Numerical Linear Algebra with Applications | Only a good match if the paper shifts more clearly toward sparse nonlinear linearization, Krylov methods, and solver methodology | Mathematics, Applied | 2.1 | Q1 | 1.013 | Q1 | WoS: Mathematics, Applied<br>SCImago: Algebra and Number Theory; Applied Mathematics | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=1070-5325&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=25712&tip=sid) |

## Recommended Submission Order

For the paper in its current natural shape, a reasonable submission order is:

1. `Journal of Scientific Computing`
2. `ACM Transactions on Mathematical Software`
3. `SIAM Journal on Scientific Computing`
4. `Computational Mechanics`
5. `Applied Mathematical Modelling`
6. `Advances in Computational Mathematics`
7. `Computational Optimization and Applications`
8. `Computer Methods in Applied Mechanics and Engineering`
9. `Numerical Linear Algebra with Applications`

## How To Reorder The Venue List After Revision

Promote `SIAM Journal on Scientific Computing` to the top if:
the revision adds a much stronger solver / scaling / sparse linear algebra story.

Promote `Computational Mechanics` higher if:
the paper becomes clearly mechanics-led and the plasticity validation is strengthened.

Promote `Computer Methods in Applied Mechanics and Engineering` only if:
the paper gains one or two very strong apples-to-apples comparisons and a more decisive flagship mechanics result.

Promote `ACM Transactions on Mathematical Software` to the top if:
the revision leans harder into software design, reproducibility, artifact quality, and reusable implementation infrastructure.

## Minimal Concrete Plan Before Submission

If the goal is to maximize acceptance probability without turning this into a new project, the best concrete path is:

1. Freeze the thesis as a software-and-methods paper.
2. Make `Plasticity3D` the flagship result.
3. Add one tightly controlled derivative-route ablation table.
4. Add one fair external baseline.
5. Add one short plasticity surrogate validation.
6. Add a fairness / limitations subsection.
7. Re-tune the title, abstract, and conclusion to the selected venue.
8. Re-check all rankings immediately before submission.
