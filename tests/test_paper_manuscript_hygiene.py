from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "paper" / "scripts" / "check_manuscript_hygiene.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_manuscript_hygiene", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manuscript_body_omits_references_by_default() -> None:
    hygiene = _load_module()
    text = "Body text is clean.\n\nReferences\nKarolina: Compute nodes.\n"

    body = hygiene.manuscript_body(text)

    assert "Karolina" not in body
    assert hygiene.find_hygiene_findings(body) == []


def test_hygiene_finds_process_and_local_labels() -> None:
    hygiene = _load_module()
    text = (
        "The mainline campaign used tmp/source_compare and P4(L1_2).\n"
        "This was a repository-local software ranking.\n"
    )

    findings = hygiene.find_hygiene_findings(text)

    assert any("internal implementation label" in finding for finding in findings)
    assert any("draft or review-process marker" in finding for finding in findings)
    assert any("local filesystem or raw-result path" in finding for finding in findings)
    assert any("implementation mesh alias" in finding for finding in findings)
    assert any("process-local or defensive comparison framing" in finding for finding in findings)
