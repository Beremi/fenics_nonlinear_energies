from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_SCRIPTS = REPO_ROOT / "paper" / "scripts"
if str(PAPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PAPER_SCRIPTS))

from common import save_pdf_and_png  # noqa: E402


def test_paper_figure_exports_use_deterministic_pdf_metadata(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])

    out = tmp_path / "figure.pdf"
    save_pdf_and_png(fig, out)
    plt.close(fig)

    pdf_bytes = out.read_bytes()
    assert b"/CreationDate (D:20000101000000Z)" in pdf_bytes
    assert b"/ModDate (D:20000101000000Z)" in pdf_bytes
    assert (tmp_path / "figure.png").is_file()
