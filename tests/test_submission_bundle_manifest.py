from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "paper" / "scripts" / "check_submission_bundle_manifest.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))


def _load_module():
    spec = importlib.util.spec_from_file_location("check_submission_bundle_manifest", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_submission_bundle_manifest_verifies_bundle_and_source_hashes(tmp_path: Path) -> None:
    checker = _load_module()
    source = tmp_path / "source.json"
    bundle = tmp_path / "bundle" / "source.json"
    dependency = tmp_path / "dependency.csv"
    manifest = tmp_path / "bundle" / "manifest.json"
    bundle.parent.mkdir()
    source.write_text('{"ok": true}\n', encoding="utf-8")
    bundle.write_text('{"ok": true, "sanitized": true}\n', encoding="utf-8")
    dependency.write_text("x\n1\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "source_files": [
                    {
                        "bundle_path": "bundle/source.json",
                        "bundle_sha256": _digest(bundle),
                        "source_path": "source.json",
                        "source_sha256": _digest(source),
                        "source_dependencies": [
                            {
                                "source_path": "dependency.csv",
                                "source_sha256": _digest(dependency),
                            },
                            {
                                "source_path": "external_reference/upstream/input.mat",
                                "source_sha256": "0" * 64,
                            },
                        ],
                    }
                ],
                "git_commit": "0" * 40,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = checker.check_manifest(manifest, repo_root=tmp_path, check_git_commit=False)

    assert result.bundle_files == 1
    assert result.local_sources == 2
    assert result.external_sources == 1
    assert result.git_commit is None


def test_submission_bundle_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
    checker = _load_module()
    source = tmp_path / "source.json"
    bundle = tmp_path / "bundle" / "source.json"
    manifest = tmp_path / "bundle" / "manifest.json"
    bundle.parent.mkdir()
    source.write_text('{"ok": true}\n', encoding="utf-8")
    bundle.write_text('{"ok": true}\n', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "source_files": [
                    {
                        "bundle_path": "bundle/source.json",
                        "bundle_sha256": "1" * 64,
                        "source_path": "source.json",
                        "source_sha256": _digest(source),
                    }
                ],
                "git_commit": "0" * 40,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        checker.check_manifest(manifest, repo_root=tmp_path, check_git_commit=False)
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("expected hash mismatch to fail")

    assert "SHA-256 mismatch" in message


def test_submission_bundle_manifest_rejects_stale_git_commit(tmp_path: Path) -> None:
    checker = _load_module()
    source = tmp_path / "source.json"
    bundle = tmp_path / "bundle" / "source.json"
    manifest = tmp_path / "bundle" / "manifest.json"
    tracked = tmp_path / "paper" / "main.tex"
    bundle.parent.mkdir()
    tracked.parent.mkdir()
    source.write_text('{"ok": true}\n', encoding="utf-8")
    bundle.write_text('{"ok": true}\n', encoding="utf-8")

    import subprocess

    subprocess.run(["git", "-C", str(tmp_path), "init"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test User"], check=True)
    tracked.write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "paper/main.tex"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "old"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    old_commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    tracked.write_text("new\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "paper/main.tex"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "new"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    manifest.write_text(
        json.dumps(
            {
                "git_commit": old_commit,
                "source_files": [
                    {
                        "bundle_path": "bundle/source.json",
                        "bundle_sha256": _digest(bundle),
                        "source_path": "source.json",
                        "source_sha256": _digest(source),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        checker.check_manifest(manifest, repo_root=tmp_path)
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("expected stale git commit to fail")

    assert "bundle-refresh paths changed" in message
    assert "paper/main.tex" in message
