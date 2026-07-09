from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "paper" / "scripts" / "check_float_placements.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))


def _load_module():
    spec = importlib.util.spec_from_file_location("check_float_placements", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_float_checker_accepts_allowlisted_h_float(tmp_path: Path) -> None:
    checker = _load_module()
    tex = tmp_path / "section.tex"
    tex.write_text(
        "\n".join(
            [
                r"\begin{figure}[H]",
                r"\caption{A controlled validation figure.}",
                r"\label{fig:plasticity3d-validation}",
                r"\end{figure}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    floats = checker.find_hard_floats([tex])

    assert len(floats) == 1
    assert checker.unapproved_hard_floats(floats) == []


def test_float_checker_rejects_unapproved_h_float(tmp_path: Path) -> None:
    checker = _load_module()
    tex = tmp_path / "section.tex"
    tex.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\caption{A new hard table.}",
                r"\label{tab:new-hard-table}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    floats = checker.find_hard_floats([tex])
    findings = checker.unapproved_hard_floats(floats)

    assert len(findings) == 1
    assert findings[0].label == "tab:new-hard-table"


def test_float_checker_rejects_unlabeled_h_float(tmp_path: Path) -> None:
    checker = _load_module()
    tex = tmp_path / "section.tex"
    tex.write_text(
        "\n".join(
            [
                r"\begin{algorithm}[H]",
                r"\caption{Unlabeled hard algorithm}",
                r"\end{algorithm}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    findings = checker.unapproved_hard_floats(checker.find_hard_floats([tex]))

    assert len(findings) == 1
    assert findings[0].label is None
