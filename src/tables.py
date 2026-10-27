"""Minimal booktabs LaTeX table writer (tabular only, so the paper controls captions)."""
from __future__ import annotations

from pathlib import Path


def _escape(s: str) -> str:
    return s if "\\" in s else s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def write_latex(path: Path, header: list[str], rows: list[list[str]], align: str) -> None:
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(_escape(str(c)) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    Path(path).write_text("\n".join(lines))
