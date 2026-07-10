from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = REPO_ROOT / "paper" / "scripts" / "common.py"
MIGRATED_SCRIPTS = (
    "build_submission_bundle.py",
    "generate_paper_figures.py",
    "generate_paper_tables.py",
    "generate_reproducibility_note.py",
    "check_submission_bundle_manifest.py",
    "check_release_blockers.py",
    "validate_paper_assets.py",
)


def _load_common():
    name = "paper_common_bundle_root_test"
    spec = importlib.util.spec_from_file_location(name, COMMON_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_paper_bundle_root_defaults_to_preserved_historical_campaign(tmp_path: Path) -> None:
    common = _load_common()

    resolved = common.resolve_paper_bundle_root(repo_root=tmp_path, environ={})

    assert resolved == (
        tmp_path
        / "artifacts"
        / "reproduction"
        / "paper_submission_2026_07_08"
    ).resolve()


def test_paper_bundle_root_explicit_override_precedes_environment(tmp_path: Path) -> None:
    common = _load_common()
    environment = {
        common.PAPER_BUNDLE_ROOT_ENV: "artifacts/reproduction/from_environment"
    }

    resolved = common.resolve_paper_bundle_root(
        "artifacts/reproduction/from_cli",
        repo_root=tmp_path,
        environ=environment,
    )

    assert resolved == (
        tmp_path / "artifacts" / "reproduction" / "from_cli"
    ).resolve()


def test_process_cli_bundle_root_precedes_environment(tmp_path: Path) -> None:
    common = _load_common()

    resolved = common.configured_paper_bundle_root(
        [
            "--paper-bundle-root=artifacts/reproduction/from_process_cli",
            "--unrelated-option",
        ],
        repo_root=tmp_path,
        environ={
            common.PAPER_BUNDLE_ROOT_ENV: "artifacts/reproduction/from_environment"
        },
    )

    assert resolved == (
        tmp_path / "artifacts" / "reproduction" / "from_process_cli"
    ).resolve()


def test_paper_bundle_root_environment_override_is_repository_relative(tmp_path: Path) -> None:
    common = _load_common()

    resolved = common.resolve_paper_bundle_root(
        repo_root=tmp_path,
        environ={
            common.PAPER_BUNDLE_ROOT_ENV: "artifacts/reproduction/revision_2026_07_10"
        },
    )

    assert resolved == (
        tmp_path
        / "artifacts"
        / "reproduction"
        / "revision_2026_07_10"
    ).resolve()


@pytest.mark.parametrize(
    "invalid",
    (
        "",
        "paper",
        "artifacts/reproduction",
        "artifacts/reproduction/../raw_results/campaign",
    ),
)
def test_paper_bundle_root_rejects_empty_or_unsafe_paths(
    tmp_path: Path,
    invalid: str,
) -> None:
    common = _load_common()

    with pytest.raises(ValueError):
        common.resolve_paper_bundle_root(invalid, repo_root=tmp_path, environ={})


def test_paper_bundle_root_rejects_existing_file(tmp_path: Path) -> None:
    common = _load_common()
    candidate = tmp_path / "artifacts" / "reproduction" / "not_a_directory"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("not a campaign directory\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        common.resolve_paper_bundle_root(candidate, repo_root=tmp_path, environ={})


def test_paper_bundle_root_cli_alias_is_resolved_and_validated() -> None:
    common = _load_common()
    parser = argparse.ArgumentParser()
    common.add_paper_bundle_root_argument(parser)

    args = parser.parse_args(
        ["--bundle-root", "artifacts/reproduction/paper_revision_cli_test"]
    )

    assert args.paper_bundle_root == (
        REPO_ROOT
        / "artifacts"
        / "reproduction"
        / "paper_revision_cli_test"
    ).resolve()


def test_all_paper_bundle_consumers_use_the_shared_configuration() -> None:
    scripts_root = REPO_ROOT / "paper" / "scripts"
    for name in MIGRATED_SCRIPTS:
        source = (scripts_root / name).read_text(encoding="utf-8")
        assert "paper_submission_2026_07_08" not in source, name
        assert "add_paper_bundle_root_argument" in source, name
