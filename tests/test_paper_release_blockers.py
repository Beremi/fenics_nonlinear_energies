from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "paper" / "scripts" / "check_release_blockers.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))


def _load_module():
    spec = importlib.util.spec_from_file_location("check_release_blockers", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_main(path: Path, documentclass: str, availability: str) -> None:
    paper = path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text(
        "\n".join(
            [
                documentclass,
                r"\begin{document}",
                r"\section*{Code and Data Availability}",
                availability,
                r"\bibliographystyle{unsrtnat}",
                r"\end{document}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_manifest(path: Path, limitations: list[str]) -> Path:
    manifest = path / "artifacts" / "reproduction" / "paper_submission_2026_07_08" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"known_limitations": limitations}) + "\n", encoding="utf-8")
    return manifest


def test_release_blockers_report_current_unresolved_classes(tmp_path: Path) -> None:
    checker = _load_module()
    manifest = _write_manifest(
        tmp_path,
        ["Target-journal metadata, repository license, and permanent archive DOI remain outside this bundle."],
    )
    _write_main(
        tmp_path,
        r"\documentclass{article}",
        "The code is available on GitHub. No separate archival DOI is cited for this version.",
    )

    blockers = checker.find_release_blockers(repo_root=tmp_path, bundle_manifest=manifest)

    assert {blocker.code for blocker in blockers} == {
        "target-template",
        "repository-license",
        "archival-doi",
        "durable-archive",
    }
    durable_archive = next(blocker for blocker in blockers if blocker.code == "durable-archive")
    assert durable_archive.evidence == "artifacts/reproduction/paper_submission_2026_07_08/manifest.json"


def test_release_blocker_cli_output_uses_relative_manifest_path(tmp_path: Path, capsys) -> None:
    checker = _load_module()
    manifest = _write_manifest(
        tmp_path,
        ["Target-journal metadata, repository license, and permanent archive DOI remain outside this bundle."],
    )
    _write_main(
        tmp_path,
        r"\documentclass{article}",
        "The code is available on GitHub. No separate archival DOI is cited for this version.",
    )

    exit_code = checker.main(
        [
            "--repo-root",
            str(tmp_path),
            "--bundle-manifest",
            str(manifest),
            "--expect-blockers",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "artifacts/reproduction/paper_submission_2026_07_08/manifest.json" in output
    assert str(tmp_path) not in output


def test_release_blockers_pass_when_release_metadata_is_present(tmp_path: Path) -> None:
    checker = _load_module()
    manifest = _write_manifest(tmp_path, ["No release metadata limitations remain."])
    _write_main(
        tmp_path,
        r"\documentclass{siamart220329}",
        r"The archived source and artifact bundle are available at \url{https://doi.org/10.5281/zenodo.1234567}.",
    )
    (tmp_path / "LICENSE").write_text("Chosen license text.\n", encoding="utf-8")

    blockers = checker.find_release_blockers(repo_root=tmp_path, bundle_manifest=manifest)

    assert blockers == []
