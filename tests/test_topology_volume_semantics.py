from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.runners import run_topology_docs_suite
from src.problems.topology.jax.jax_energy import material_measure, volume_fraction
from src.problems.topology.jax.mesh import CantileverTopologyMesh
from src.problems.topology.support.volume import VolumeTarget, resolve_volume_target


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_volume_target_converts_between_fraction_and_measure() -> None:
    from_fraction = VolumeTarget.from_normalized_fraction(0.4, 2.0)
    from_measure = VolumeTarget.from_material_measure(0.8, 2.0)

    assert from_fraction == from_measure
    assert from_fraction.normalized_fraction == pytest.approx(0.4)
    assert from_fraction.material_measure == pytest.approx(0.8)


def test_volume_target_rejects_ambiguous_inconsistent_units() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        resolve_volume_target(
            2.0,
            target_normalized_fraction=0.4,
            target_material_measure=0.4,
        )


def test_serial_measure_and_fraction_have_explicit_units() -> None:
    theta = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    weights = np.array([0.5, 0.75, 0.75], dtype=np.float64)

    measure = float(material_measure(theta, weights))
    fraction = float(volume_fraction(theta, weights, 2.0))

    assert measure == pytest.approx(1.0)
    assert fraction == pytest.approx(0.5)


def test_serial_initialization_hits_normalized_target() -> None:
    mesh = CantileverTopologyMesh(
        nx=24,
        ny=12,
        length=2.0,
        height=1.0,
        traction=1.0,
        load_fraction=0.2,
        fixed_pad_cells=4,
        load_pad_cells=4,
    )
    _template, z_full, _z_free = mesh.build_design_state(
        target_normalized_fraction=0.4,
        theta_min=1e-6,
        solid_latent=10.0,
    )
    scaled = 1.0 / (1.0 + np.exp(-z_full))
    theta = 1e-6 + (1.0 - 1e-6) * scaled
    fraction = float(np.dot(mesh.nodal_volume_weights, theta) / mesh.domain_area)

    assert fraction == pytest.approx(0.4, abs=1e-12)


def test_topology_docs_suite_keeps_matched_and_paper_contracts_distinct() -> None:
    serial = run_topology_docs_suite.SERIAL_REFERENCE_ARGS
    matched_parallel = run_topology_docs_suite.DIRECT_PARALLEL_ARGS
    paper_parallel = run_topology_docs_suite.MESH_TIMING_ARGS_BASE

    for args in (serial, matched_parallel):
        assert "--volume_fraction_target" not in args
        assert args[args.index("--target-normalized-fraction") + 1] == "0.4"
        assert args[args.index("--initial-normalized-fraction") + 1] == "0.4"

    assert "--volume_fraction_target" not in paper_parallel
    assert paper_parallel[paper_parallel.index("--target-material-measure") + 1] == "0.4"
    assert paper_parallel[paper_parallel.index("--initial-normalized-fraction") + 1] == "0.4"


def test_topology_docs_suite_rejects_ambiguous_cached_results(tmp_path: Path) -> None:
    source = tmp_path / "output.json"
    legacy = {
        "parameters": {
            "length": 2.0,
            "height": 1.0,
            "volume_fraction_target": 0.4,
        }
    }
    with pytest.raises(RuntimeError, match="ambiguous volume semantics"):
        run_topology_docs_suite._require_explicit_volume_contract(legacy, source)

    current = {
        "parameters": {
            "length": 2.0,
            "height": 1.0,
            "volume_semantics_version": 2,
            "target_normalized_fraction": 0.2,
            "target_material_measure": 0.4,
            "initial_normalized_fraction": 0.4,
        }
    }
    run_topology_docs_suite._require_explicit_volume_contract(current, source)


def test_maintained_topology_commands_do_not_use_ambiguous_legacy_flag() -> None:
    command_sources = (
        "experiments/runners/run_topology_docs_suite.py",
        "experiments/runners/run_readme_docs_smoke.py",
        "experiments/runners/run_paper_reviewer_gap_experiments.py",
        "experiments/analysis/generate_report_assets.py",
        "experiments/analysis/generate_parallel_full_report.py",
        "experiments/analysis/generate_parallel_scaling_stallstop_report.py",
        "docs/setup/quickstart.md",
        "docs/problems/Topology.md",
        "docs/results/Topology.md",
    )
    for relative in command_sources:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "--volume_fraction_target" not in text, relative

