"""Split paper/main.pdf into one PDF per section (for reviewers) and copy the full paper.

Output: review/<NN>_<section>.pdf, review/00_full_paper.pdf. Page ranges come from the
section entries of paper/main.aux, so rerun after `make paper`.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
SECTION = re.compile(r"\\contentsline \{section\}\{(?:\\numberline \{([^}]*)\})?(.*?)\}\{(\d+)\}")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main() -> None:
    aux = (ROOT / "paper" / "main.aux").read_text()
    sections = [(num or "", title, int(page)) for num, title, page in SECTION.findall(aux)]
    reader = PdfReader(ROOT / "paper" / "main.pdf")
    n_pages = len(reader.pages)
    out = ROOT / "review"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.pdf"):
        old.unlink()
    shutil.copy(ROOT / "paper" / "main.pdf", out / "00_full_paper.pdf")
    # Reviewer units: group short consecutive sections. Each unit runs from the page where its
    # first section starts to the page where the next unit starts (a shared page appears in both).
    units = [
        ("base_paper_summary_motivation_introduction", ["Summary of the base paper", "Motivation", "Introduction"]),
        ("data", ["Data"]),
        ("methods", ["Methods"]),
        ("results", ["Results"]),
        ("discussion_conclusion", ["Discussion", "Conclusion"]),
        ("references_data_and_code_availability", ["References", "Data and Code Availability"]),
        ("appendix_robustness", ["Robustness"]),
    ]
    first_page = {title: page for _, title, page in sections}
    starts = [first_page[titles[0]] for _, titles in units]
    for i, (name, _) in enumerate(units):
        stop = starts[i + 1] if i + 1 < len(units) else n_pages
        stop = max(stop, starts[i])
        writer = PdfWriter()
        for p in range(starts[i] - 1, stop):
            writer.add_page(reader.pages[p])
        fname = f"{i + 1:02d}_{name}.pdf"
        with open(out / fname, "wb") as fh:
            writer.write(fh)
        print(f"{fname}: pages {starts[i]}-{stop}")


if __name__ == "__main__":
    main()
