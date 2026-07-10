from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "paper" / "scripts" / "generate_paper_tables.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))


def _load_table_generator():
    spec = importlib.util.spec_from_file_location("generate_paper_tables", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_topology_material_measure_is_normalized_by_domain_area() -> None:
    generator = _load_table_generator()

    assert generator.normalized_topology_fraction(0.4) == 0.2
    assert generator.normalized_topology_fraction_from_row(
        {"final_volume_fraction": "0.4"}
    ) == 0.2
    assert generator.normalized_topology_fraction_from_row(
        {
            "volume_semantics_version": "2",
            "final_volume_fraction": "0.2",
            "final_normalized_fraction": "0.2",
            "final_material_measure": "0.4",
        }
    ) == 0.2


def test_reference_continuation_is_classified_as_three_dimensional() -> None:
    generator = _load_table_generator()
    run_info = json.loads(generator.P3D_REFERENCE_CONT_NP8.read_text(encoding="utf-8"))

    assert run_info["mesh"]["coord_shape"][0] == 3
    assert run_info["mesh"]["elem_shape"][0] == 35
    assert "3d_hetero_ssr" in run_info["mesh"]["mesh_file"]
    assert "plasticity3d_reference_continuation.tex" in generator.TABLE_SOURCE_INPUTS
    assert "plasticity2d_reference_continuation.tex" not in generator.TABLE_SOURCE_INPUTS
