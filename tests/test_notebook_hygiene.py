from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NOTEBOOKS = [
    REPO_ROOT / "notebooks" / "demos" / "plaplace_jax_api.ipynb",
    REPO_ROOT / "notebooks" / "demos" / "ginzburg_landau_jax_api.ipynb",
    REPO_ROOT / "notebooks" / "demos" / "hyperelasticity_jax_api.ipynb",
    REPO_ROOT / "notebooks" / "demos" / "plaplace_jax_petsc_api.ipynb",
    REPO_ROOT / "notebooks" / "benchmarks" / "plaplace_fenics_benchmark.ipynb",
    REPO_ROOT / "notebooks" / "benchmarks" / "ginzburg_landau_fenics_benchmark.ipynb",
    REPO_ROOT / "notebooks" / "benchmarks" / "hyperelasticity_fenics_benchmark.ipynb",
    REPO_ROOT / "notebooks" / "benchmarks" / "topology_parallel_benchmark.ipynb",
]
BANNED_SNIPPETS = (
    "pLaplace2D_",
    "GinzburgLandau2D_",
    "HyperElasticity3D_",
    "topological_optimisation_jax",
    "experiment_scripts/",
    "replications/",
    "overview/",
)


def test_current_notebook_set_exists() -> None:
    assert sorted(path.name for path in EXPECTED_NOTEBOOKS) == [
        "ginzburg_landau_fenics_benchmark.ipynb",
        "ginzburg_landau_jax_api.ipynb",
        "hyperelasticity_fenics_benchmark.ipynb",
        "hyperelasticity_jax_api.ipynb",
        "plaplace_fenics_benchmark.ipynb",
        "plaplace_jax_api.ipynb",
        "plaplace_jax_petsc_api.ipynb",
        "topology_parallel_benchmark.ipynb",
    ]
    for path in EXPECTED_NOTEBOOKS:
        assert path.exists(), path


def test_notebooks_use_canonical_paths_and_imports() -> None:
    for path in EXPECTED_NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        for snippet in BANNED_SNIPPETS:
            assert snippet not in text, f"{path.relative_to(REPO_ROOT)} -> {snippet}"


def test_notebook_outputs_target_canonical_artifact_root() -> None:
    for path in EXPECTED_NOTEBOOKS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        joined = json.dumps(payload)
        assert "artifacts/raw_results/notebook_runs" in joined
