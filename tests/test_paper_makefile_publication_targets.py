from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = REPO_ROOT / "paper" / "Makefile"


def _recipe(source: str, target: str) -> str:
    match = re.search(
        rf"^{re.escape(target)}:[^\n]*\n(?P<body>(?:\t[^\n]*\n|#.*\n|\n)*)",
        source,
        flags=re.MULTILINE,
    )
    assert match is not None, target
    return match.group("body")


def test_ordinary_table_build_cannot_generate_revision_evidence() -> None:
    source = MAKEFILE.read_text(encoding="utf-8")
    recipe = _recipe(source, "tables")

    assert "generate_paper_tables.py" in recipe
    assert "generate_revision_evidence_tables.py" not in recipe


def test_revision_targets_separate_diagnostic_and_publication_outputs() -> None:
    source = MAKEFILE.read_text(encoding="utf-8")
    diagnostic = _recipe(source, "revision-diagnostic-tables")
    publication = _recipe(source, "revision-publication-tables")

    assert "--evidence-class diagnostic" in diagnostic
    assert "REVISION_DIAGNOSTIC_OUT_DIR" in diagnostic
    assert "tables/generated" not in diagnostic
    assert "--evidence-class publication" in publication
    assert '--evidence-manifest "$(REVISION_EVIDENCE_MANIFEST)"' in publication
    assert 'test -n "$(REVISION_EVIDENCE_ROOT)"' in publication
    assert 'test -n "$(REVISION_EVIDENCE_MANIFEST)"' in publication


def test_submission_checks_all_three_admitted_local_tables() -> None:
    source = MAKEFILE.read_text(encoding="utf-8")
    local_check = _recipe(source, "publication-local-check")

    assert "scripts/check_distributed_colored_manifest.py" in local_check
    assert "scripts/check_globalization_local_manifest.py" in local_check
    assert "scripts/check_stopping_local_manifest.py" in local_check
    assert "$(MAKE) publication-local-check" in _recipe(source, "submission-check")
    assert "$(MAKE) publication-local-check" in _recipe(source, "publish-check")


def test_clean_preserves_checked_in_publication_tables() -> None:
    source = MAKEFILE.read_text(encoding="utf-8")

    assert "tables/generated" not in _recipe(source, "clean")
