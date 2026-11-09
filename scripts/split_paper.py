"""Split paper/main.pdf into one PDF per section (for reviewers) and copy the full paper.

Output: review/<section number>_<section>.pdf (numbers follow the paper), review/00_full_paper.pdf. Page ranges come from the
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
        ("00a_summary_of_the_base_paper", ["Summary of the base paper"]),
        ("01_motivation", ["Motivation"]),
        ("02_introduction", ["Introduction"]),
        ("03_data", ["Data"]),
        ("04_methods", ["Methods"]),
        ("05_results", ["Results"]),
        ("06_discussion", ["Discussion"]),
        ("07_conclusion_and_ai_statement", ["Conclusion", "Statement on the use of AI tools"]),
        ("08_references", ["References"]),
        ("09_data_and_code_availability", ["Data and Code Availability"]),
        ("A_appendix_robustness", ["Robustness"]),
    ]
    # printed page numbers can differ from physical ones (the title page is unnumbered):
    # locate the first section heading in the PDF text to get the offset
    first_title = sections[0][1]
    phys = next(i + 1 for i, pg in enumerate(reader.pages)
                if first_title.lower() in (pg.extract_text() or "").lower())
    offset = phys - sections[0][2]
    first_page = {title: page + offset for _, title, page in sections}
    starts = [first_page[titles[0]] for _, titles in units]
    for i, (name, _) in enumerate(units):
        stop = starts[i + 1] if i + 1 < len(units) else n_pages
        if i + 1 < len(units):
            # if the next unit starts at the top of its page, that page is not shared
            head = (reader.pages[stop - 1].extract_text() or "").strip()[:120].lower()
            if units[i + 1][1][0].lower() in head:
                stop -= 1
        stop = max(stop, starts[i])
        writer = PdfWriter()
        for p in range(starts[i] - 1, stop):
            writer.add_page(reader.pages[p])
        fname = f"{name}.pdf"
        with open(out / fname, "wb") as fh:
            writer.write(fh)
        print(f"{fname}: pages {starts[i]}-{stop}")


if __name__ == "__main__":
    main()
