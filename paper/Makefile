PYTHON := ../.venv/bin/python
LATEXMK := latexmk
MAIN := main.tex

.PHONY: all layout figures tables literature validate pdf clean

all: pdf

layout:
	$(PYTHON) scripts/measure_layout.py

figures: layout
	$(PYTHON) scripts/generate_paper_figures.py

tables: layout
	$(PYTHON) scripts/generate_paper_tables.py

literature:
	$(PYTHON) scripts/generate_literature_sources.py

validate: figures tables
	$(PYTHON) scripts/validate_paper_assets.py

pdf: validate
	$(LATEXMK) -pdf $(MAIN)

clean:
	rm -rf build
	rm -f figures/generated/*
	rm -f tables/generated/*
