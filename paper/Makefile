PYTHON := ../.venv/bin/python
LATEXMK := latexmk
MAIN := main.tex
REVISION_DIAGNOSTIC_EVIDENCE_ROOT ?= ../artifacts/reproduction/paper_revision_2026_07_10/pilots
REVISION_DIAGNOSTIC_OUT_DIR ?= build/diagnostic-revision-tables
REVISION_EVIDENCE_ROOT ?=
REVISION_EVIDENCE_MANIFEST ?=
DISTRIBUTED_COLORED_EVIDENCE_ROOT ?=
GLOBALIZATION_LOCAL_EVIDENCE_ROOT ?=
STOPPING_LOCAL_EVIDENCE_ROOT ?=
.PHONY: all layout figures tables literature literature-check validate publish-check submission-check release-check release-blockers submission-bundle pdf clean \
	revision-diagnostic-tables revision-publication-tables publication-local-check \
	distributed-colored-table distributed-colored-check distributed-colored-check-if-present \
	globalization-local-table globalization-local-check globalization-local-check-if-present \
	stopping-local-table stopping-local-check stopping-local-check-if-present \
	stopping-submission-table stopping-submission-check \
	plasticity3d-validation-run plasticity3d-validation-assets \
	plasticity3d-derivative-ablation-run plasticity3d-derivative-ablation-assets \
	jax-fem-baseline-run jax-fem-baseline-assets reproducibility

all: pdf

layout:
	$(PYTHON) scripts/measure_layout.py

figures: layout
	$(PYTHON) scripts/generate_paper_figures.py

tables: layout
	$(PYTHON) scripts/generate_paper_tables.py

# Diagnostic revision tables never write into the canonical publication directory.
revision-diagnostic-tables:
	mkdir -p "$(REVISION_DIAGNOSTIC_OUT_DIR)"
	$(PYTHON) scripts/generate_revision_evidence_tables.py \
		--out-dir "$(REVISION_DIAGNOSTIC_OUT_DIR)" \
		--evidence-root "$(REVISION_DIAGNOSTIC_EVIDENCE_ROOT)" \
		--evidence-class diagnostic

# Canonical revision tables require an explicitly supplied publication manifest.
revision-publication-tables:
	test -n "$(REVISION_EVIDENCE_ROOT)"
	test -n "$(REVISION_EVIDENCE_MANIFEST)"
	$(PYTHON) scripts/generate_revision_evidence_tables.py \
		--out-dir tables/generated \
		--evidence-root "$(REVISION_EVIDENCE_ROOT)" \
		--evidence-class publication \
		--evidence-manifest "$(REVISION_EVIDENCE_MANIFEST)"

distributed-colored-table:
	test -n "$(DISTRIBUTED_COLORED_EVIDENCE_ROOT)"
	$(PYTHON) scripts/generate_distributed_colored_table.py --evidence-root "$(DISTRIBUTED_COLORED_EVIDENCE_ROOT)"

distributed-colored-check:
	$(PYTHON) scripts/check_distributed_colored_manifest.py

distributed-colored-check-if-present:
	@if test -e tables/generated/distributed_colored_manifest.json; then \
		$(PYTHON) scripts/check_distributed_colored_manifest.py; \
	fi

globalization-local-table:
	test -n "$(GLOBALIZATION_LOCAL_EVIDENCE_ROOT)"
	$(PYTHON) scripts/generate_globalization_local_status.py --evidence-root "$(GLOBALIZATION_LOCAL_EVIDENCE_ROOT)"

globalization-local-check:
	$(PYTHON) scripts/check_globalization_local_manifest.py

globalization-local-check-if-present:
	@if test -e tables/generated/globalization_local_manifest.json; then \
		$(PYTHON) scripts/check_globalization_local_manifest.py; \
	fi

stopping-local-table:
	test -n "$(STOPPING_LOCAL_EVIDENCE_ROOT)"
	$(PYTHON) scripts/generate_stopping_local_status.py --evidence-root "$(STOPPING_LOCAL_EVIDENCE_ROOT)"

stopping-local-check:
	$(PYTHON) scripts/check_stopping_local_manifest.py

stopping-local-check-if-present:
	@if test -e tables/generated/stopping_local_manifest.json; then \
		$(PYTHON) scripts/check_stopping_local_manifest.py; \
	fi

stopping-submission-table:
	$(PYTHON) scripts/generate_stopping_submission_status.py

stopping-submission-check:
	$(PYTHON) scripts/check_stopping_submission_manifest.py

publication-local-check:
	$(PYTHON) scripts/check_distributed_colored_manifest.py
	$(PYTHON) scripts/check_globalization_local_manifest.py
	$(PYTHON) scripts/check_stopping_local_manifest.py
	$(PYTHON) scripts/check_stopping_submission_manifest.py

literature:
	$(PYTHON) scripts/generate_literature_sources.py

literature-check:
	$(PYTHON) scripts/generate_literature_sources.py --skip-download --check

plasticity3d-validation-run:
	$(PYTHON) ../experiments/runners/run_plasticity3d_validation.py

plasticity3d-validation-assets:
	$(PYTHON) ../experiments/analysis/generate_plasticity3d_validation_assets.py

plasticity3d-derivative-ablation-run:
	$(PYTHON) ../experiments/runners/run_plasticity3d_derivative_ablation.py

plasticity3d-derivative-ablation-assets:
	$(PYTHON) ../experiments/analysis/generate_plasticity3d_derivative_ablation_assets.py

jax-fem-baseline-run:
	$(PYTHON) ../experiments/runners/run_jax_fem_hyperelastic_baseline.py

jax-fem-baseline-assets:
	$(PYTHON) ../experiments/analysis/generate_jax_fem_hyperelastic_baseline_assets.py

reproducibility:
	$(PYTHON) scripts/generate_reproducibility_note.py

submission-bundle:
	$(PYTHON) scripts/build_submission_bundle.py

validate: figures tables
	$(PYTHON) scripts/validate_paper_assets.py

publish-check:
	$(MAKE) publication-local-check
	$(MAKE) literature-check
	$(PYTHON) scripts/check_submission_bundle_manifest.py
	$(PYTHON) scripts/validate_paper_assets.py --archive-neutral

submission-check:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN)
	! rg -n "(Overfull|Underfull|LaTeX Warning|Package .* Warning|undefined references|Rerun to get cross-references|multiply defined|pdfTeX warning)" build/main.log
	qpdf --check build/main.pdf
	$(PYTHON) scripts/check_pdf_aux_order.py build/main.aux
	$(PYTHON) scripts/check_float_placements.py
	$(PYTHON) scripts/check_manuscript_hygiene.py build/main.pdf
	$(PYTHON) scripts/check_revision_evidence_manifest.py
	$(MAKE) publication-local-check
	$(MAKE) literature-check
	$(PYTHON) scripts/check_submission_bundle_manifest.py
	$(PYTHON) scripts/validate_paper_assets.py --archive-neutral

release-blockers:
	$(PYTHON) scripts/check_release_blockers.py

release-check: submission-check release-blockers

pdf: validate
	$(LATEXMK) -pdf $(MAIN)

clean:
	rm -rf build
	rm -f figures/generated/*
