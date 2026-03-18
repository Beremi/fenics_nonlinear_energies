from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BANNED_PREFIXES = (
    "tools",
    "tools_petsc4py",
    "pLaplace2D_",
    "GinzburgLandau2D_",
    "HyperElasticity3D_",
    "topological_optimisation_jax",
)


def _iter_python_files() -> list[Path]:
    roots = [
        REPO_ROOT / "src",
        REPO_ROOT / "tests",
        REPO_ROOT / "experiments" / "runners",
        REPO_ROOT / "experiments" / "analysis",
    ]
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob("*.py")))
    return files


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_maintained_modules_use_canonical_imports():
    violations: list[str] = []
    for path in _iter_python_files():
        for module in sorted(_imported_modules(path)):
            if module.startswith(BANNED_PREFIXES):
                violations.append(f"{path.relative_to(REPO_ROOT)} -> {module}")
    assert not violations, "Legacy imports remain:\n" + "\n".join(violations)
