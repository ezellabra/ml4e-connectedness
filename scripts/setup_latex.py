"""Check for a LaTeX installation and the packages used by paper/.

Installs LaTeX only if it is missing (Linux: apt/dnf; macOS: Homebrew BasicTeX cask;
Windows: prints MiKTeX instructions). If installed, only missing packages are installed
(via tlmgr). Never reinstalls LaTeX.

On macOS the BasicTeX cask needs administrator rights. If that fails, the exact command
is printed; with --user-fallback the script instead installs TinyTeX (a TeX Live
distribution) in the user's home directory, which needs no administrator rights.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_STY = {
    "amsmath": "amsmath",
    "booktabs": "booktabs",
    "graphicx": "graphics",
    "natbib": "natbib",
    "hyperref": "hyperref",
    "geometry": "geometry",
    "caption": "caption",
    "subcaption": "caption",
}
EXTRA_TLMGR = ["latexmk"]
LINUX_APT = ["texlive-latex-extra", "texlive-fonts-recommended", "texlive-science", "latexmk", "biber"]
LINUX_DNF = ["texlive-scheme-medium", "latexmk", "biber"]
MIKTEX_URL = "https://miktex.org/download"
TINYTEX_URL = "https://yihui.org/tinytex/install-bin-unix.sh"
TEX_BIN_DIRS = [
    "/Library/TeX/texbin",
    str(Path.home() / "Library/TinyTeX/bin/universal-darwin"),
    str(Path.home() / ".TinyTeX/bin/x86_64-linux"),
]


def extend_path() -> None:
    for d in TEX_BIN_DIRS:
        if Path(d).is_dir() and d not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


def tool_ok(cmd: list[str]) -> bool:
    if shutil.which(cmd[0]) is None:
        return False
    res = subprocess.run(cmd, capture_output=True, text=True)
    first = (res.stdout or res.stderr).strip().splitlines()
    print(f"{cmd[0]}: {first[0] if first else 'no output'}")
    return res.returncode == 0


def latex_installed() -> bool:
    extend_path()
    return tool_ok(["pdflatex", "--version"])


def install_latex(user_fallback: bool) -> bool:
    system = platform.system()
    if system == "Windows":
        print(f"LaTeX is not installed. Install MiKTeX from {MIKTEX_URL} (enable on-the-fly "
              "package installation), make sure pdflatex is on PATH, install Strawberry Perl for "
              "latexmk, then rerun this script.")
        return False
    if system == "Darwin":
        cmd = ["brew", "install", "--cask", "basictex"]
    elif system == "Linux":
        if shutil.which("apt-get"):
            cmd = ["sudo", "apt-get", "install", "-y"] + LINUX_APT
        elif shutil.which("dnf"):
            cmd = ["sudo", "dnf", "install", "-y"] + LINUX_DNF
        else:
            print("No supported package manager (apt/dnf) found.")
            return False
    else:
        print(f"Unsupported OS: {system}")
        return False
    print("$ " + " ".join(cmd))
    res = subprocess.run(cmd, text=True, stdin=subprocess.DEVNULL)
    if res.returncode == 0 and latex_installed():
        return True
    print("Automatic installation failed (administrator rights are probably required). Run by hand:")
    print("    " + " ".join(cmd))
    if system == "Darwin":
        print("    then: sudo tlmgr update --self && sudo tlmgr install latexmk")
    if user_fallback and system in ("Darwin", "Linux"):
        print(f"Installing TinyTeX in the user's home directory from {TINYTEX_URL}")
        res = subprocess.run(["sh", "-c", f"curl -sL {TINYTEX_URL} | sh"], text=True)
        return res.returncode == 0 and latex_installed()
    return False


def missing_packages() -> list[str]:
    missing = []
    for sty, pkg in REQUIRED_STY.items():
        res = subprocess.run(["kpsewhich", f"{sty}.sty"], capture_output=True, text=True)
        if not res.stdout.strip():
            missing.append(pkg)
    if shutil.which("latexmk") is None:
        missing += EXTRA_TLMGR
    return sorted(set(missing))


def install_packages(pkgs: list[str]) -> None:
    if shutil.which("tlmgr") is None:
        print("tlmgr not found; install these packages with your TeX distribution: " + " ".join(pkgs))
        return
    tlmgr = shutil.which("tlmgr")
    needs_sudo = not os.access(tlmgr, os.W_OK) and str(Path.home()) not in tlmgr
    cmd = (["sudo"] if needs_sudo else []) + ["tlmgr", "install"] + pkgs
    print("$ " + " ".join(cmd))
    res = subprocess.run(cmd, text=True, stdin=subprocess.DEVNULL)
    if res.returncode != 0:
        print("Package installation failed. Run by hand:\n    " + " ".join(cmd))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-fallback", action="store_true",
                    help="install TinyTeX in the home directory if the system install fails")
    args = ap.parse_args()
    ok = latex_installed()
    if not ok:
        print("LaTeX not found; attempting installation.")
        ok = install_latex(args.user_fallback)
    if not ok:
        print("SUMMARY: LaTeX MISSING")
        return 1
    tool_ok(["latexmk", "-v"])
    missing = missing_packages()
    if missing:
        print("Installing missing LaTeX packages: " + ", ".join(missing))
        install_packages(missing)
        missing = missing_packages()
    print("SUMMARY: LaTeX OK / " + ("no missing packages" if not missing else "missing packages: " + ", ".join(missing)))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
