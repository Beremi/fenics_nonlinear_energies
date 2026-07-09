from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_README = REPO_ROOT / "paper" / "README.md"


def test_paper_readme_documents_readiness_gates() -> None:
    text = PAPER_README.read_text(encoding="utf-8")

    for command in (
        "make -C paper publish-check",
        "make -C paper submission-check",
        "make -C paper release-blockers",
        "make -C paper release-check",
    ):
        assert command in text

    assert "release-check` is expected to fail" in text
    assert "Target venue/template" in text
    assert "Root repository license" in text
    assert "archival DOI" in text


def test_paper_readme_keeps_style_guide_local_only() -> None:
    text = PAPER_README.read_text(encoding="utf-8")

    assert "paper/style_guide/" in text
    assert ".git/info/exclude" in text
    assert "must not be staged" in text
