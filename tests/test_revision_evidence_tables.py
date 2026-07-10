from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
GENERATOR = REPO_ROOT / "paper/scripts/generate_revision_evidence_tables.py"
CHECKER = REPO_ROOT / "paper/scripts/check_revision_evidence_manifest.py"
PILOT_ROOT = REPO_ROOT / "artifacts/reproduction/paper_revision_2026_07_10/pilots"
sys.path.insert(0, str(REPO_ROOT / "paper/scripts"))
import generate_revision_evidence_tables as revision_tables  # noqa: E402


def test_revision_evidence_tables_are_generated_from_pilot_artifacts(tmp_path: Path) -> None:
    subprocess.run(
        [str(PYTHON), str(GENERATOR), "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    verification = (tmp_path / "revision_verification_summary.tex").read_text()
    derivatives = (tmp_path / "revision_derivative_checks.tex").read_text()
    quadrature = (tmp_path / "revision_quadrature_sensitivity.tex").read_text()
    evidence = (tmp_path / "revision_evidence_status.tex").read_text()
    manifest = json.loads((tmp_path / "revision_evidence_manifest.json").read_text())

    assert "1.887" in verification
    assert "1.006" in verification
    assert "Hyperelastic nonaffine" in verification
    assert "Hyperelasticity, one/two ranks & 2" in derivatives
    assert "Hyperelasticity, one/two/four ranks" not in derivatives
    assert "P_2(L_1)" in quadrature
    assert "1.84e-08" in quadrature
    assert "0/12" in evidence
    assert "0 diagnostic rows" in evidence
    assert "0 train; 0 holdout" in evidence
    assert "Descriptive paired route timing" in evidence
    assert "Post-fit crossover location" in evidence
    assert "not evaluated" in evidence
    assert manifest["schema_version"] == 2
    assert manifest["evidence_class"] == "diagnostic"
    assert manifest["publication_evidence"] is False
    assert manifest["status"] == "diagnostic_tables_not_for_submission"
    assert len(manifest["inputs"]) == 14
    assert len(manifest["outputs"]) == 4


def test_revision_evidence_root_is_configurable(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    for source in PILOT_ROOT.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(PILOT_ROOT)
        if relative.as_posix() not in {
            "EXP-VAL-001/plaplace_manufactured.json",
            "EXP-VAL-001/ginzburg_landau_manufactured.json",
            "EXP-VAL-001/hyperelastic_affine_patch.json",
            "EXP-VAL-001/hyperelastic_nonaffine_quadrature_refinement_v2/result.json",
            "EXP-DERIV-001/smooth_fixed_element_v1.json",
            "EXP-DERIV-001/p1_l1_fixed_element_v2.json",
            "EXP-DERIV-001/p2_l1_fixed_element_v2.json",
            "EXP-DERIV-001/p4_l1_fixed_element_v2.json",
            "EXP-MC-001/material_point_verification.json",
            "EXP-DIST-001/distribution_equivalence.json",
            "EXP-DISC-001/p1_l1_fixed_state_quadrature_v2.json",
            "EXP-DISC-001/p2_l1_fixed_state_quadrature_v2.json",
            "EXP-DISC-001/p4_l1_fixed_state_quadrature_v2.json",
            "EXP-ROUTE-001/analysis_contract_v1/analysis.json",
        }:
            continue
        target = evidence_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    output = tmp_path / "tables"
    subprocess.run(
        [
            str(PYTHON),
            str(GENERATOR),
            "--out-dir",
            str(output),
            "--evidence-root",
            str(evidence_root),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "revision_evidence_manifest.json").read_text())
    assert manifest["evidence_root"] == str(evidence_root.resolve())
    assert all(
        row["path_within_evidence_root"]
        for row in manifest["inputs"].values()
    )


def test_publication_generation_and_submission_checker_fail_closed(tmp_path: Path) -> None:
    publication = subprocess.run(
        [
            str(PYTHON),
            str(GENERATOR),
            "--out-dir",
            str(tmp_path / "publication"),
            "--evidence-class",
            "publication",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert publication.returncode != 0
    assert "--evidence-manifest is required" in publication.stderr

    diagnostic_dir = tmp_path / "diagnostic"
    subprocess.run(
        [str(PYTHON), str(GENERATOR), "--out-dir", str(diagnostic_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    checked = subprocess.run(
        [
            str(PYTHON),
            str(CHECKER),
            "--manifest",
            str(diagnostic_dir / "revision_evidence_manifest.json"),
            "--expect-diagnostic",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0
    assert "not submission-admissible" in checked.stdout


def test_publication_status_table_is_derived_without_inventing_crossover() -> None:
    data = {
        "route_analysis": {
            "terminal_decision": "predictive_selector_admissible",
            "empirical_map": [
                {"status": "admitted"},
                {"status": "admitted"},
                {"status": "censored"},
            ],
            "cost_model": {
                "selector_claim_admissible": True,
                "training_rows": 48,
                "holdout_rows": 20,
            },
        }
    }
    rendered = revision_tables._evidence_status_table(
        data, evidence_class="publication"
    )
    assert "2/3" in rendered
    assert "admitted finite map" in rendered
    assert "admitted descriptive paired timing" in rendered
    assert "0 confirmation rows" in rendered
    assert "not evaluated" in rendered
    assert "crossover location & 2 admitted" not in rendered


def test_publication_status_table_reports_clean_negative_without_predictive_leakage() -> None:
    data = {
        "route_analysis": {
            "terminal_decision": "finite_empirical_map_only",
            "empirical_map": [
                *[{"status": "admitted"} for _ in range(96)],
                *[{"status": "censored"} for _ in range(6)],
            ],
            "cost_model": {
                "status": "fit_gate_failed",
                "selector_claim_admissible": False,
                "training_rows": 74,
                "holdout_rows": 22,
                "preflight_failures": [],
                "failed_gates": ["median_absolute_percentage_error"],
            },
            "factorized_microbenchmark_gate": {
                "passed": False,
                "failures": ["factorized calibration holdout gates failed"],
            },
            # This uncontracted object is rejected by source admission.  The
            # renderer independently refuses to turn it into a claim.
            "post_fit_confirmation": {
                "publication_admissible": True,
                "terminal_decision": "post_fit_crossover_confirmed",
                "admitted_rows": 99,
            },
        }
    }
    rendered = revision_tables._evidence_status_table(
        data, evidence_class="publication"
    )
    assert "96/102" in rendered
    assert "admitted finite map" in rendered
    assert "admitted descriptive paired timing" in rendered
    assert "Predictive cost selector & 74 train; 22 holdout & not admitted" in rendered
    assert "negative terminal recorded; no predictive claim" in rendered
    assert "0 confirmation rows & not evaluated" in rendered
    assert "99 confirmation rows" not in rendered
    assert "descriptive diagnostic reported" in rendered
    assert "not a selector gate" in rendered


def test_diagnostic_status_table_never_uses_publication_admission_labels() -> None:
    data = {
        "route_analysis": {
            "terminal_decision": "predictive_selector_admissible",
            "empirical_map": [
                {"status": "admitted"},
                {"status": "admitted"},
                {"status": "censored"},
            ],
            "cost_model": {
                "selector_claim_admissible": True,
                "training_rows": 48,
                "holdout_rows": 20,
            },
            "factorized_microbenchmark_gate": {"passed": True, "failures": []},
        }
    }
    rendered = revision_tables._evidence_status_table(
        data, evidence_class="diagnostic"
    )
    assert "complete diagnostic map" in rendered
    assert "2 diagnostic rows" in rendered
    assert "Predictive cost selector & 48 train; 20 holdout & not admitted" in rendered
    assert "admitted finite map" not in rendered
    assert "admitted descriptive paired timing" not in rendered
