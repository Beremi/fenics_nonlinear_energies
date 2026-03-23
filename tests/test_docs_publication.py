from __future__ import annotations

import subprocess
from pathlib import Path

from experiments.analysis.docs_assets import common


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"


def _tracked_doc_names(subdir: str) -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", f"docs/{subdir}/*.md"],
        cwd=REPO_ROOT,
        text=True,
    )
    return sorted(Path(line).name for line in output.splitlines() if line)
ASSETS_ROOT = DOCS_ROOT / "assets"
BUILD_ROOT = REPO_ROOT / "experiments" / "analysis" / "docs_assets"
BANNED_SNIPPETS = (
    "/home/",
    "/workdir/",
    "/usr/bin/mpiexec",
    "artifacts/figures/img",
    "docs/benchmarks/",
    "docs/overview/",
    "replications/",
    "experiment_scripts/",
)


def test_current_docs_structure_exists() -> None:
    expected = [
        DOCS_ROOT / "README.md",
        DOCS_ROOT / "setup" / "quickstart.md",
        DOCS_ROOT / "setup" / "local_build.md",
        DOCS_ROOT / "problems" / "pLaplace.md",
        DOCS_ROOT / "problems" / "GinzburgLandau.md",
        DOCS_ROOT / "problems" / "HyperElasticity.md",
        DOCS_ROOT / "problems" / "Plasticity.md",
        DOCS_ROOT / "problems" / "Topology.md",
        DOCS_ROOT / "results" / "pLaplace.md",
        DOCS_ROOT / "results" / "GinzburgLandau.md",
        DOCS_ROOT / "results" / "HyperElasticity.md",
        DOCS_ROOT / "results" / "Plasticity.md",
        DOCS_ROOT / "results" / "Topology.md",
    ]
    for path in expected:
        assert path.exists(), path


def test_retired_overview_markdown_surface_is_gone() -> None:
    assert not (REPO_ROOT / "overview").exists()


def test_problem_pages_contain_required_sections() -> None:
    for name in ("pLaplace", "GinzburgLandau", "HyperElasticity", "Plasticity", "Topology"):
        text = (DOCS_ROOT / "problems" / f"{name}.md").read_text(encoding="utf-8")
        assert "## Mathematical Formulation" in text
        assert "## Commands Used" in text
        assert "![" in text
        assert ("## Maintained Implementations" in text) or ("## Implementation Status" in text)
        assert ("Energy Table Across Levels" in text) or ("Resolution / Objective Table" in text)
    assert ".gif" in (DOCS_ROOT / "problems" / "Topology.md").read_text(encoding="utf-8")


def test_results_pages_contain_required_sections() -> None:
    for name in ("pLaplace", "GinzburgLandau", "HyperElasticity", "Plasticity", "Topology"):
        text = (DOCS_ROOT / "results" / f"{name}.md").read_text(encoding="utf-8")
        assert "## Current Maintained Comparison" in text
        assert "## Reproduction Commands" in text
        assert "![" in text


def test_current_assets_exist_under_docs_assets() -> None:
    expected_pdf = [
        "plaplace/plaplace_sample_state.pdf",
        "plaplace/plaplace_energy_levels.pdf",
        "plaplace/plaplace_strong_scaling.pdf",
        "plaplace/plaplace_mesh_timing.pdf",
        "ginzburg_landau/ginzburg_landau_sample_state.pdf",
        "ginzburg_landau/ginzburg_landau_energy_levels.pdf",
        "ginzburg_landau/ginzburg_landau_strong_scaling.pdf",
        "ginzburg_landau/ginzburg_landau_mesh_timing.pdf",
        "hyperelasticity/hyperelasticity_sample_state.pdf",
        "hyperelasticity/hyperelasticity_energy_levels.pdf",
        "hyperelasticity/hyperelasticity_strong_scaling.pdf",
        "hyperelasticity/hyperelasticity_mesh_timing.pdf",
        "plasticity/mc_plasticity_p4_l5_displacement.pdf",
        "plasticity/mc_plasticity_p4_l5_deviatoric_strain_robust.pdf",
        "topology/topology_final_density.pdf",
        "topology/topology_objective_history.pdf",
        "topology/topology_strong_scaling.pdf",
        "topology/topology_mesh_timing.pdf",
    ]
    expected_png = [rel.replace(".pdf", ".png") for rel in expected_pdf]
    expected_png_only = [
        "plasticity/plasticity_p4_l7_scaling_overall_loglog.png",
        "plasticity/plasticity_p4_l7_scaling_per_linear_iteration_loglog.png",
        "plasticity/plasticity_p4_l7_setup_subparts_loglog.png",
        "plasticity/plasticity_p4_l7_callback_breakdown_loglog.png",
        "plasticity/plasticity_p4_l7_linear_breakdown_loglog.png",
        "plasticity/plasticity_p4_l7_pmg_internal_loglog.png",
    ]
    for rel in expected_pdf:
        assert (ASSETS_ROOT / rel).exists(), rel
    for rel in expected_png:
        assert (ASSETS_ROOT / rel).exists(), rel
    for rel in expected_png_only:
        assert (ASSETS_ROOT / rel).exists(), rel
    assert (ASSETS_ROOT / "topology" / "topology_parallel_final_evolution.gif").exists()


def test_docs_use_only_current_repo_relative_paths() -> None:
    md_paths = [DOCS_ROOT / "README.md", *sorted(DOCS_ROOT.glob("**/*.md")), REPO_ROOT / "README.md"]
    violations: list[str] = []
    for path in md_paths:
        text = path.read_text(encoding="utf-8")
        for snippet in BANNED_SNIPPETS:
            if snippet in text:
                violations.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")
    assert not violations, "Current docs still contain banned stale paths:\n" + "\n".join(violations)


def test_current_docs_have_one_problem_and_one_results_page_per_family() -> None:
    problems = _tracked_doc_names("problems")
    results = _tracked_doc_names("results")
    assert problems == [
        "GinzburgLandau.md",
        "HyperElasticity.md",
        "Plasticity.md",
        "Topology.md",
        "pLaplace.md",
    ]
    assert results == ["GinzburgLandau.md", "HyperElasticity.md", "Plasticity.md", "Topology.md", "pLaplace.md"]


def test_publication_style_constants_remain_locked() -> None:
    assert common.FIGURE_WIDTH_CM == 11.0
    assert common.OVERVIEW_FONT_PT == 12.0


def test_internal_figure_build_scripts_still_exist() -> None:
    expected = [
        BUILD_ROOT / "build_plaplace_figures.py",
        BUILD_ROOT / "build_ginzburg_landau_figures.py",
        BUILD_ROOT / "build_hyperelasticity_figures.py",
        BUILD_ROOT / "build_topology_figures.py",
        BUILD_ROOT / "build_all.py",
    ]
    for path in expected:
        assert path.exists(), path
