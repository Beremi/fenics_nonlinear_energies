from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_degree_energy_asset_generator_writes_plots_and_tables(tmp_path: Path) -> None:
    summary_path = tmp_path / "comparison_summary.json"
    study_dir = tmp_path / "study"
    docs_dir = tmp_path / "docs_assets"
    summary_path.write_text(
        json.dumps(
            {
                "benchmark": "Plasticity3D lambda=1.55 degree-vs-resolution energy study",
                "rows": [
                    {
                        "degree_line": "P1",
                        "mesh_alias": "L1",
                        "artifact_dir": "artifacts/raw_results/docs_showcase/example_p1",
                        "status": "completed",
                        "reused": False,
                        "free_dofs": 1000,
                        "energy": -1.0,
                        "total_time_s": 10.0,
                    },
                    {
                        "degree_line": "P1",
                        "mesh_alias": "L1_2",
                        "artifact_dir": "artifacts/raw_results/docs_showcase/example_p1_2",
                        "status": "completed",
                        "reused": True,
                        "free_dofs": 8000,
                        "energy": -1.2,
                        "total_time_s": 18.0,
                    },
                    {
                        "degree_line": "P2",
                        "mesh_alias": "L1",
                        "artifact_dir": "artifacts/raw_results/docs_showcase/example_p2",
                        "status": "completed",
                        "reused": False,
                        "free_dofs": 4000,
                        "energy": -1.1,
                        "total_time_s": 22.0,
                    },
                    {
                        "degree_line": "P4",
                        "mesh_alias": "L1",
                        "artifact_dir": "artifacts/raw_results/docs_showcase/example_p4",
                        "status": "failed",
                        "reused": False,
                        "free_dofs": 32000,
                        "energy": -1.3,
                        "total_time_s": 80.0,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "experiments/analysis/generate_plasticity3d_lambda1p55_degree_mesh_energy_assets.py",
            "--summary-json",
            str(summary_path),
            "--study-dir",
            str(study_dir),
            "--docs-out-dir",
            str(docs_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (docs_dir / "plasticity3d_lambda1p55_degree_energy_study.png").exists()
    assert (docs_dir / "plasticity3d_lambda1p55_degree_energy_study.pdf").exists()
    assert (docs_dir / "plasticity3d_lambda1p55_degree_energy_assets_summary.json").exists()
    assert (study_dir / "plasticity3d_lambda1p55_degree_energy_table.md").exists()
    assert (study_dir / "plasticity3d_lambda1p55_degree_energy_table.tex").exists()
