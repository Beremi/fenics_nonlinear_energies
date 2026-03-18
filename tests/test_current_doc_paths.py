from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [
    REPO_ROOT / "README.md",
    *sorted((REPO_ROOT / "docs").glob("**/*.md")),
]
BANNED_SNIPPETS = (
    "/home/",
    "/work/",
    "/workdir/",
    "artifacts/figures/img",
    "docs/benchmarks/",
    "docs/overview/",
    "overview/",
    "replications/",
    "experiment_scripts/",
)


def test_current_docs_do_not_reference_legacy_placeholder_paths():
    violations: list[str] = []
    for path in DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        for snippet in BANNED_SNIPPETS:
            if snippet in text:
                violations.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")
    assert not violations, "Legacy doc paths remain:\n" + "\n".join(violations)
