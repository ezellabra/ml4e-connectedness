"""Build the reviewer files from the compiled paper.

Output:
  review/paper_latex.pdf  copy of paper/main.pdf (the whole paper)
  review/paper_latex.zip  the LaTeX source needed to compile it: main.tex, the sections, the
                          bibliography, and the logos, figures and tables the paper actually uses

The list of files comes from paper/main.fls (written by latexmk), so rerun after `make paper`.
Unused files (for example a logo removed from the title page) are left out of the zip.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "review"
SKIP_SUFFIXES = {".aux", ".bbl", ".out", ".log", ".toc", ".fls"}


def source_files() -> list[Path]:
    """Files inside the repository that LaTeX read while compiling, plus the .bib file."""
    files = {PAPER / "references.bib"}
    for line in (PAPER / "main.fls").read_text().splitlines():
        if not line.startswith("INPUT "):
            continue
        path = (PAPER / line[len("INPUT "):]).resolve()
        if path.suffix in SKIP_SUFFIXES or not path.is_relative_to(ROOT) or not path.is_file():
            continue
        files.add(path)
    return sorted(files)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    shutil.copy(PAPER / "main.pdf", OUT / "paper_latex.pdf")
    files = source_files()
    with zipfile.ZipFile(OUT / "paper_latex.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())
    print("paper_latex.pdf: copied from paper/main.pdf")
    print(f"paper_latex.zip: {len(files)} files")


if __name__ == "__main__":
    main()
