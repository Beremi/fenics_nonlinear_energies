# Paper Submission Checklist

## Publishability Verdict

Current status: not ready for submission until the target venue declarations,
repository license, and citable archival record are finalized. Manuscript claims,
citation checks, generated assets, and archive-neutral provenance validation are
otherwise ready for final venue formatting. Validation is adequate for the
narrowed claims, but the paper must not be submitted until the external
submission metadata and citable archive are fixed.

## Review Pass Scope

This pass inspected the manuscript sources, bibliography, literature manifest,
claim audit, generated tables and figures, paper generation scripts, LaTeX
build products, and supporting artifacts needed by the paper's main claims.
An earlier audit rechecked date-sensitive SOTA and citation metadata through
live arXiv, DOI, and official project pages on 2026-04-30. No long MPI campaigns
were rerun, no checked-in meshes or raw inputs were edited, and no new
scientific claims were introduced without supporting evidence.

## Completed Readiness Work

- Updated `paper/references.bib`, `paper/literature/manifest.json`,
  `paper/literature/sources.md`, and `paper/literature/claim_audit.md` for
  JetSCI (`cattaneo2026jetsci`), Xue 2026 (`xue2026implicit`), current PETSc
  official citation metadata, richer Yashchuk arXiv metadata, and explicit
  Davis/Ginzburg/Sysala2017 evidence limits.
- Updated `paper/sections/introduction.tex` and
  `paper/sections/related_work.tex` to cover JetSCI and Xue 2026 without
  overstating novelty or source support.
- Reworded validation and performance claims in
  `paper/sections/abstract.tex`, `paper/sections/validation.tex`,
  `paper/sections/conclusion.tex`, `paper/sections/benchmarks.tex`, and
  `paper/sections/implementation.tex` so Plasticity3D gates, Plasticity2D
  fixed-work diagnostics, alternative Krylov/preconditioner diagnostics, and
  JAX-FEM comparison wording match the available evidence.
- Updated `paper/scripts/generate_paper_tables.py` and regenerated generated
  tables, including the SOTA comparison, the Plasticity3D validation summary
  with endpoint deviatoric-strain as a diagnostic row, the family highlights
  load-factor context, and the renamed fixed-source Plasticity3D table.
- Updated `paper/scripts/generate_paper_figures.py` and regenerated generated
  figures with paper notation such as `L_2` rather than implementation aliases
  such as `L1_2`.
- Updated `paper/scripts/validate_paper_assets.py` so required figures and
  generated tables are derived from the TeX sources and checked against the
  figure manifest.
- Added `paper/scripts/build_submission_bundle.py`, created
  `artifacts/reproduction/paper_submission_2026_07_08/`, redirected
  paper-critical figure/table provenance to that bundle, and made
  `make -C paper publish-check` pass locally.
- Updated `paper/scripts/generate_literature_sources.py` so the default command
  uses cached local full texts unless a required download is missing; the new
  `--refresh-downloads` flag forces a network refresh.
- Tightened the abstract, introduction, related work, discussion, and conclusion
  around the primary \jaxpetsc{} realization, scoped comparator language to
  internal reference implementations versus narrow external/reference-model
  checks, and removed repeated or broad performance/validation claims.
- Made the methods and benchmark definitions more self-contained by defining
  quadrature stress/tangent symbols, algorithm labels, the reported p-Laplace
  discrete energy, hyperelastic density-before-stress notation, Plasticity2D
  regularization, and Plasticity3D marker language.
- Regenerated generated tables after revising benchmark availability,
  Plasticity3D validation status labels, JAX-FEM threshold status labels, and
  SOTA bridge wording.
- Added body-text interpretation for Plasticity2D endpoint versus fixed-work
  diagnostics, the fixed Plasticity3D derivative-route comparison, and topology
  rank-variation evidence; reduced the page-30 Plasticity3D float gap in the
  current A4 PDF.
- Added figure source provenance for every generated figure in the figure
  manifest, tightened the asset validator so TeX-included figures must have
  source records, and classified direct figure inputs by archive status.
- Added generated-table source provenance for every generated table, tightened
  the asset validator so TeX-included generated tables must have source records,
  and classified direct table inputs by archive status.
- Expanded the curated submission bundle with small raw/report-backed table
  inputs, Plasticity2D endpoint and resolution inputs, and the Plasticity3D
  recommended-scaling summary plus per-rank result JSONs. The generated
  manifests initially moved most figure and table sources to archive-neutral.
- Added derived Plasticity3D figure inputs for the degree-energy plots,
  convergence histories, state-pair surface rendering, and highest-y-slice
  panels, and added a table-specific fixed-reference PMG summary with
  paper-facing route identifiers. The generated manifests now mark all 34
  figure sources and all 30 table sources as archive-neutral.
- Rebuilt the curated submission bundle after fixing the JAX-FEM comparison
  summary paths to point at bundle-local terminal states.
- Sharpened remaining prose and notation issues from the latest audits:
  PETSc-owned sparse assembly in the abstract, non-defensive comparison wording,
  branchwise constitutive tangents, colored HVP/finite-difference recovery,
  hyperelastic load notation, Plasticity3D body-force notation, and endpoint
  validation scope.
- Relaxed selected hard-pinned floats in the validation, results, and appendix
  sections to reduce current blank-page regions and target-template fragility.
- Addressed the latest message/method/evidence/layout audit: named the
  \v{C}erm{\'a}k--Sysala--Valdman MATLAB slope-stability lineage, made the
  reduced objective and constitutive AD notation self-contained, clarified
  colored sparse recovery as AD-HVP probes in reported runs with
  finite-difference probes as the classical variant, added owned/ghost
  distributed-assembly prose, interpreted benchmark and appendix tables in body
  text, expanded capability and fixed-reference PMG table labels, and tuned
  float-page/table readability.
- Addressed the front/back-matter and leakage audit: aligned the visible title
  and PDF metadata around energy minimization, removed the automatic date, moved
  code/data availability into unnumbered back matter, replaced draft archive
  wording with a current-version statement, made Plasticity3D boundary labels
  self-contained, and removed remaining process-local wording from selected
  results prose.
- Regenerated the hyperelastic PMG sensitivity table after polishing coarse
  solver labels, so the generated output now reports `Hypre` and `MUMPS, one
  redundant group` rather than implementation abbreviations.
- Addressed the latest narrative/math/evidence/layout audit: split comparator
  roles in the abstract and introduction, replaced the related-work rhetorical
  opening with a taxonomy, made the Plasticity2D branch-potential definition
  self-contained, defined $\lambda_{\max}^{\mathrm{succ}}$ on the fixed-load
  Plasticity3D validation grid, corrected the hyperelastic globalization timing
  interpretation, added body-text interpretations for scalar and hyperelastic
  scaling figures, changed benchmark table labels to solver-realization roles,
  and inserted an appendix float barrier so Tables 26--28 appear in order.
- Added `paper/scripts/check_pdf_aux_order.py` and the `make -C paper
  submission-check` target, which build the PDF, scan the log for warnings,
  run `qpdf`, check figure/table aux ordering, and rerun archive-neutral asset
  validation.
- Addressed the latest comparator-scope and layout audit: corrected the
  p-Laplace benchmark domain to the L-shaped mesh, documented the separate
  unit-load p-Laplace globalization stress test, softened MATLAB and SOTA
  comparator wording, renamed the colored-Hessian sparsity pattern, top-aligned
  float pages, fixed the hyperelastic state figure font family, and forced the
  Plasticity3D derivative-route table to precede the following
  discretization/scaling figure.
- Added a solver-status and timing vocabulary table, a numerical-protocol
  summary for the results section, explicit mixed timing headers for
  globalization and derivative-route comparisons, and a Plasticity2D caption
  that distinguishes endpoint values from fixed-work diagnostics.
- Addressed the latest evidence-gate audit: split comparator roles in the
  abstract and introduction, named the \v{C}erm{\'a}k--Sysala--Valdman MATLAB
  implementation lineage, added explicit timing-scope/cap/gradient-gate columns
  to the affected generated results tables, and expanded the curated bundle with
  the JSON inputs needed to audit those rows.
- Addressed the benchmark self-containedness audit: added a compact
  discretization-label table with representative DOF counts, defined the
  Plasticity2D plane-strain matrix and branch convention before the branch
  formulas, named Plasticity3D material regions, mapped Plasticity3D boundary
  labels to geometric faces, and stated the 3D ordered-principal-strain branch
  convention.
- Completed a display-equation punctuation audit across the manuscript TeX
  sources: all `equation`, `align`, and display-math blocks already treat the
  displayed formulas as sentence parts after labels and trailing line breaks are
  ignored.
- Decided against adding a compact solver-policy table before target-template
  conversion: the implementation section already gives the component map, and
  the numerical protocol table plus local result captions carry run-specific
  tolerances, caps, timing scopes, and coarse-solver variants.
- Rebuilt the paper PDF through the paper generation pipeline.

## Blocking Issues Before Submission

- Issue: Target journal, template, and submission declarations are unresolved.
  Why it blocks publishability: the paper still uses a generic `article` class
  and cannot be submitted without journal-specific formatting, reference style,
  author metadata, funding, acknowledgements, and competing-interest
  declarations.
  Evidence path or citation: `paper/main.tex` front matter and availability
  statement.
  Required action: choose the target journal, apply its template, and fill in
  ORCID, corresponding-author, funding, acknowledgements, data/software
  availability, and COI fields.
- Issue: Repository license and archival release/DOI are not decided.
  Why it blocks publishability: a software-methods submission needs an
  unambiguous license and a citable, durable version of the source/artifact
  snapshot.
  Evidence path or citation: no `LICENSE*` or `COPYING*` file is present at
  repository depth two; `paper/main.tex` does not yet cite a separate archival
  DOI.
  Required action: add the chosen repository license, include
  `artifacts/reproduction/paper_submission_2026_07_08/` in the submission
  release or artifact archive, mint or record its DOI if applicable, and update
  the manuscript availability statement.
- Issue: Durable archive integration is not yet complete.
  Why it blocks publishability: the archive-neutral gate passes and all
  generated figures and TeX-included generated tables now have archive-neutral
  source records, but the provenance bundle is still a repository artifact
  rather than a citable, licensed release artifact named in the manuscript.
  Evidence path or citation: `paper/figures/generated/manifest.json`,
  `paper/tables/generated/manifest.json`,
  `paper/scripts/validate_paper_assets.py`, and
  `artifacts/reproduction/paper_submission_2026_07_08/manifest.json`.
  Required action: include `artifacts/reproduction/paper_submission_2026_07_08/`
  in the durable release/archive, mint or record the DOI if applicable, rerun
  the archive-neutral validator from the released snapshot, and update the
  availability statement.

## Major Revisions Needed

- Complete the journal-specific front matter and declarations once the target
  venue is chosen.
- Fold the curated submission bundle into the final release/archive and make
  the manuscript availability statement cite that durable version.
- After choosing the target journal template, revisit forced `[H]` floats and
  split or simplify the dense SOTA, Plasticity3D, and appendix tables if the
  venue class narrows the text block.
- Decide whether locally cached full texts under `paper/literature/fulltext/`
  should remain an ignored private audit cache or become part of a controlled
  review artifact; do not imply public availability for restricted sources.
- Obtain accessible full text for Davis, Ginzburg, and Sysala2017 if the paper
  needs claims beyond the currently conservative metadata/context use.
- Keep Plasticity3D claims scoped to same-case reference-formula agreement,
  fixed-load reference-operator diagnostics, and endpoint-surrogate behavior
  unless a true incremental-history validation campaign is added.

## Minor Polish Items

- Recheck the SOTA table and dense result tables after the target template is
  applied; they are readable in the current A4 article but template-fragile.
- Review compound Plasticity3D figure labels in the final template for crowding
  and regenerate at the final physical size if needed.
- Revisit the availability statement after the archive/license decision so it
  reads like final submission metadata rather than a repository-local note.
- Do not add a new solver-policy table before target-template conversion;
  revisit only if reviewer feedback asks for a single lookup table for
  run-specific solver policies.

## Claim And Citation Audit Summary

All active citation keys used by the manuscript now resolve during the LaTeX
build. Local full text or generated evidence was used where available; arXiv,
DOI, official documentation, or publisher metadata was used where local full
text was unavailable. JetSCI and Xue 2026 were added as current, materially
relevant SOTA sources. PETSc and Yashchuk metadata were refreshed. Davis,
Ginzburg, and Sysala2017 remain explicitly limited because full text was not
available in this pass.

Unsupported or overbroad paper claims found during the audit were corrected in
the manuscript rather than left as open tasks: Plasticity3D now separates gated
quantities from diagnostics, Plasticity2D L6/L7 is described as fixed-work
diagnostic evidence, the former deflated-GMRES wording is softened to
alternative Krylov/preconditioner diagnostics, and source-family Plasticity3D
language stays endpoint-surrogate only. The remaining claim/evidence risk is
final archival publication: paper-critical provenance now points to a curated
bundle, but that bundle still needs the final release/DOI context and complete
per-artifact coverage for the submitted figures and tables.

## SOTA Check Outcome

Related Work and the generated SOTA table needed changes and were updated.
JetSCI (`arXiv:2604.22087`, submitted 2026-04-23) is the newest directly
overlapping hybrid JAX+PETSc source found in this pass, and Xue 2026
(`Comput. Phys. Commun. 323:110102`, DOI `10.1016/j.cpc.2026.110102`) is a
peer-reviewed finite-element differentiable-physics source covering
second-order implicit differentiation. These are now reflected in the
manuscript, bibliography, literature manifest, sources table, claim audit, and
generated SOTA table.

Additional sources such as MFEM/dFEM, newer Firedrake differentiable-programming
bridges, and FormOpt were not added because they are useful context rather than
required support for the current scoped contribution.

## Validation And Reproducibility Checks

- `./.venv/bin/python paper/scripts/generate_literature_sources.py`: initially
  failed on a transient JOSS `503 Service Unavailable` while refreshing an
  already cached full text; after the cache-first generator fix, the exact
  command passed and generated `paper/literature/sources.md` with 24 public
  entries, 9 non-public local entries, and 3 unavailable entries.
- `./.venv/bin/python paper/scripts/generate_paper_tables.py`: passed.
  The current generated table manifest records 32 generated tables; all 30
  TeX-included generated tables have source records, and all 32 generated table
  sources are archive-neutral.
- `./.venv/bin/python paper/scripts/generate_paper_figures.py --manifest-only`:
  passed and rewrote the figure manifest; the manifest records 34 generated
  figure sources, all archive-neutral.
- `./.venv/bin/python paper/scripts/build_submission_bundle.py`: passed and
  wrote `artifacts/reproduction/paper_submission_2026_07_08/manifest.json` with
  source and bundle SHA256 hashes for 51 paper-critical inputs.
- `./.venv/bin/python paper/scripts/validate_paper_assets.py`: passed on
  2026-07-09 with 29 figures, 30 generated tables, figure source records, table
  source records, and 44 paper-facing provenance-scan files checked.
- `./.venv/bin/python paper/scripts/validate_paper_assets.py --archive-neutral`:
  passed on 2026-07-09 against the curated submission bundle and
  archive-neutral table inputs.
- `make -C paper publish-check`: passed on 2026-07-09.
- `./.venv/bin/python -m py_compile paper/scripts/generate_paper_tables.py paper/scripts/check_pdf_aux_order.py`:
  passed on 2026-07-09.
- `./.venv/bin/python -m py_compile paper/scripts/generate_paper_tables.py`:
  passed on 2026-07-09 after the solver-protocol table edits.
- `(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)`:
  passed after the generated table and prose cleanup, refreshing a 41-page
  `paper/build/main.pdf`.
- `qpdf --check paper/build/main.pdf`: passed with no syntax or stream errors.
- `./.venv/bin/python paper/scripts/check_pdf_aux_order.py paper/build/main.aux`:
  passed on 2026-07-09, including the appendix table-order check.
- `make -C paper submission-check`: passed on 2026-07-09.
- `./.venv/bin/python -m py_compile paper/scripts/generate_paper_tables.py
  paper/scripts/generate_paper_figures.py paper/scripts/check_pdf_aux_order.py
  paper/scripts/validate_paper_assets.py`: passed on 2026-07-09 after the
  latest source edits.
- `pdffonts paper/figures/generated/hyperelasticity_state.pdf`: after targeted
  regeneration, the figure embeds Computer Modern fonts (`CMR`, `CMMI`,
  `CMSY`) rather than the previous NewTX/Termes outlier.
- `make -C paper submission-check`: passed again on 2026-07-09 after the
  Plasticity3D Figure 26 float-placement fix.
- `make -C paper submission-check`: passed again on 2026-07-09 after the
  solver-protocol table additions; the check covers LaTeX warning scans,
  `qpdf`, aux-order validation, and archive-neutral asset validation.
- `./.venv/bin/python paper/scripts/build_submission_bundle.py`: passed again
  on 2026-07-09 after adding Plasticity3D globalization JSONs and the
  Ginzburg--Landau timeout metadata to the curated bundle; the regenerated
  manifest records 51 source files.
- `./.venv/bin/python paper/scripts/generate_paper_tables.py`: passed again on
  2026-07-09 after adding timing-scope, wall-cap, and gradient-gate fields to
  the affected generated tables.
- `./.venv/bin/python -m py_compile paper/scripts/generate_paper_tables.py paper/scripts/build_submission_bundle.py`:
  passed on 2026-07-09.
- `make -C paper submission-check`: passed again on 2026-07-09 after the
  evidence-gate table/prose edits; the check covers LaTeX warning scans,
  `qpdf`, aux-order validation, and archive-neutral asset validation.
- Rendered and visually inspected affected pages 1--2, 25--27, and 38--41
  after the latest rebuild; the abstract/opening text, revised evidence tables,
  conclusion, availability statement, and bibliography pages are readable and
  unclipped in the current A4 PDF.
- `make -C paper submission-check`: passed again on 2026-07-09 after adding the
  benchmark discretization-label table and Plasticity2D/3D convention prose;
  the check covers LaTeX warning scans, `qpdf`, aux-order validation, and
  archive-neutral asset validation.
- Rendered and visually inspected affected pages 11 and 17--18 after the latest
  rebuild; the new discretization-label table and Plasticity2D/3D convention
  text are readable and unclipped in the current A4 PDF.
- Rendered and visually inspected representative pages 1, 5, 9, 21, and
  30--32 after the latest rebuild; the current A4 PDF is readable and unclipped,
  with page 30 improved but dense floats/tables still template-fragile.
- Rendered and visually inspected pages 13, 21--22, and 30--32 after the
  latest rebuild; the hyperelastic figure font matches the manuscript, the
  JAX-FEM comparison figure/table order is fixed, and Table 20 now precedes
  Figure 26 in the Plasticity3D results.
- Rendered and visually inspected affected pages 1, 15--18, 27--29, and
  35--36 after the latest rebuild; the new Plasticity2D equations, scaling
  interpretations, and appendix table sequence are readable and unclipped.
- Rendered and visually inspected affected pages 5--6 and 24--25 after the
  latest rebuild; the solver vocabulary table and numerical-protocol summary
  are readable, unclipped, and within the current A4 text block.
- `./.venv/bin/python -m pytest tests/test_docs_publication.py`: passed
  13 tests.
- `./.venv/bin/python -m pytest tests/test_final_report_figure_generators.py`:
  passed 3 tests.
- `rg -n "TODO|FIXME|placeholder|constitutively equivalent|validated incremental|P4\\(L1" paper/main.tex paper/sections paper/tables/generated paper/literature`:
  no matches.
- `rg -n "LaTeX Warning|Package .* Warning|Overfull|Underfull|Undefined|undefined|Citation|Reference|Fatal|Emergency|Error|Warning" paper/build/main.log paper/build/main.blg`:
  no matches in the final build logs.
- `git diff --check`: passed.

Exact remaining blockers are submission metadata and license/archive DOI. The
archive-neutral validator now passes against the local curated bundle; final
submission still must include that bundle in a durable release/archive. The only
command instability observed in the final rerun was the intermittent TeX/font
lookup or mesh-loading stall in the Makefile figure target; direct asset
validation and direct LaTeX rebuild are clean.

## Optional Future Work

- Add a true incremental-history Mohr-Coulomb Plasticity3D validation campaign.
  This would support stronger mechanics claims but is not required for the
  current endpoint-surrogate workflow paper.
- Add MFEM/dFEM, newer Firedrake differentiable-programming bridge, and FormOpt
  discussion if the paper expands from the current scoped positioning.
- Add more fairness-first external baselines on tightly matched problem
  contracts.
