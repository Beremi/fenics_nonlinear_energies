from __future__ import annotations

import json
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "paper/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_stopping_submission_manifest as checker  # noqa: E402
import generate_stopping_submission_status as generator  # noqa: E402


def test_submission_table_is_generated_from_admitted_local_manifest(
    tmp_path: Path,
) -> None:
    manifest = generator.generate(out_dir=tmp_path)
    table = (tmp_path / generator.TABLE_NAME).read_text(encoding="utf-8")

    assert manifest["presentation_counts"] == {
        "executions": 45,
        "admitted_records": 43,
        "accepted_comparisons": 28,
        "rejected_comparisons": 15,
        "endpoint_censored_comparisons": 2,
        "reference_self_comparisons": 11,
        "accepted_nonreference_candidates": 17,
    }
    assert "Admitted records" in table
    assert "Mohr--Coulomb surrogate, fixed state & 15/15 & 13/15 & 8/15" in table
    assert "45 of the 52 computations" in table
    assert checker.validate_manifest(
        tmp_path / generator.MANIFEST_NAME,
        require_canonical=False,
        regenerate=False,
    ) == []


def test_submission_manifest_rejects_changed_counts(tmp_path: Path) -> None:
    generator.generate(out_dir=tmp_path)
    path = tmp_path / generator.MANIFEST_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["presentation_counts"]["admitted_records"] = 45
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    findings = checker.validate_manifest(
        path, require_canonical=False, regenerate=False
    )
    assert "presentation counts differ from the admitted 45-row subset" in findings
