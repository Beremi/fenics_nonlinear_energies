from __future__ import annotations

from pathlib import Path

from experiments.analysis import (
    generate_gl_final_report_figures,
    generate_he_final_report_figures,
    generate_plaplace_final_report_figures,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_plaplace_figure_generator_uses_canonical_defaults():
    assert generate_plaplace_final_report_figures.REPO_ROOT == REPO_ROOT
    assert generate_plaplace_final_report_figures.DEFAULT_SUMMARY_JSON == (
        REPO_ROOT
        / "artifacts"
        / "reproduction"
        / "2026-03-15_refactor_stage2b_final"
        / "full"
        / "plaplace_final_suite"
        / "summary.json"
    )
    assert generate_plaplace_final_report_figures.DEFAULT_ASSET_DIR == (
        REPO_ROOT / "docs" / "assets" / "plaplace"
    )


def test_gl_figure_generator_uses_canonical_defaults():
    assert generate_gl_final_report_figures.REPO_ROOT == REPO_ROOT
    assert generate_gl_final_report_figures.DEFAULT_SUMMARY_JSON == (
        REPO_ROOT
        / "artifacts"
        / "reproduction"
        / "2026-03-15_refactor_stage2b_final"
        / "full"
        / "gl_final_suite"
        / "summary.json"
    )
    assert generate_gl_final_report_figures.DEFAULT_ASSET_DIR == (
        REPO_ROOT / "docs" / "assets" / "ginzburg_landau"
    )


def test_he_figure_generator_uses_canonical_defaults():
    assert generate_he_final_report_figures.REPO_ROOT == REPO_ROOT
    assert generate_he_final_report_figures.DEFAULT_SUMMARY_JSON == (
        REPO_ROOT
        / "artifacts"
        / "reproduction"
        / "2026-03-15_refactor_stage2b_final"
        / "full"
        / "he_final_suite_best"
        / "summary.json"
    )
    assert generate_he_final_report_figures.DEFAULT_ASSET_DIR == (
        REPO_ROOT / "docs" / "assets" / "hyperelasticity"
    )
