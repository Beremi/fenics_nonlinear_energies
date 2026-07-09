from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "paper" / "scripts" / "generate_literature_sources.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_literature_sources", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_literature_sources_check_rejects_stale_index(tmp_path: Path) -> None:
    literature = _load_module()
    paper = tmp_path / "paper"
    sections = paper / "sections"
    literature_dir = paper / "literature"
    sections.mkdir(parents=True)
    literature_dir.mkdir()
    (paper / "main.tex").write_text(
        "\\input{sections/intro}\n",
        encoding="utf-8",
    )
    (sections / "intro.tex").write_text(
        "\\citep{demo2026}\n",
        encoding="utf-8",
    )
    bib = paper / "references.bib"
    bib.write_text(
        """@article{demo2026,
  author = {Example Author},
  title = {A Demonstration Paper},
  journal = {Example Journal},
  year = {2026},
  doi = {10.1234/example},
  url = {https://doi.org/10.1234/example}
}
""",
        encoding="utf-8",
    )
    manifest = literature_dir / "manifest.json"
    manifest.write_text(
        """{
  "entries": {
    "demo2026": {
      "canonical_source_label": "DOI landing page",
      "canonical_source_url": "https://doi.org/10.1234/example",
      "doi": "10.1234/example",
      "fulltext_label": null,
      "fulltext_url": null,
      "isbns": [],
      "local_filename": null,
      "notes": "No public full text recorded."
    }
  }
}
""",
        encoding="utf-8",
    )
    out_md = literature_dir / "sources.md"
    out_md.write_text("stale\n", encoding="utf-8")

    old_paper_root = literature.PAPER_ROOT
    old_literature_root = literature.LITERATURE_ROOT
    old_fulltext_root = literature.FULLTEXT_ROOT
    literature.PAPER_ROOT = paper
    literature.LITERATURE_ROOT = literature_dir
    literature.FULLTEXT_ROOT = literature_dir / "fulltext"
    try:
        try:
            literature.main(
                [
                    "--bib",
                    str(bib),
                    "--manifest",
                    str(manifest),
                    "--out-md",
                    str(out_md),
                    "--skip-download",
                    "--check",
                ]
            )
        except SystemExit as exc:
            message = str(exc)
        else:
            raise AssertionError("expected stale literature index to fail")
    finally:
        literature.PAPER_ROOT = old_paper_root
        literature.LITERATURE_ROOT = old_literature_root
        literature.FULLTEXT_ROOT = old_fulltext_root

    assert "Literature source index is stale" in message

