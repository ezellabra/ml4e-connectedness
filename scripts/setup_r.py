"""Check for an R installation and the CRAN packages `nets` and `fnets`.

Installs R only if it is missing (Linux: apt/dnf, macOS: Homebrew; Windows: prints
instructions). Installs only the CRAN packages that are missing. Never reinstalls R.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys

R_PACKAGES = ["nets", "fnets"]
CRAN_MIRROR = "https://cloud.r-project.org"
# Packages removed from CRAN are installed from the CRAN archive at a pinned version.
CRAN_ARCHIVE = {"nets": "nets/nets_0.9.1.tar.gz"}
R_WINDOWS_URL = "https://cran.r-project.org/bin/windows/base/"


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def r_installed() -> bool:
    if shutil.which("Rscript") is None:
        return False
    res = run(["Rscript", "--version"])
    print((res.stdout or res.stderr).strip())
    return res.returncode == 0


def install_r() -> bool:
    system = platform.system()
    if system == "Windows":
        print("R is not installed. Windows is not handled automatically.")
        print(f"Download and run the installer from {R_WINDOWS_URL}, then add the R 'bin' "
              "folder to PATH and rerun this script.")
        return False
    if system == "Darwin":
        if shutil.which("brew") is None:
            print("Homebrew not found. Install it from https://brew.sh and run: brew install r")
            return False
        cmd = ["brew", "install", "r"]
    elif system == "Linux":
        if shutil.which("apt-get"):
            cmd = ["sudo", "apt-get", "install", "-y", "r-base", "r-base-dev"]
        elif shutil.which("dnf"):
            cmd = ["sudo", "dnf", "install", "-y", "R"]
        else:
            print("No supported package manager (apt/dnf) found. Install R manually.")
            return False
    else:
        print(f"Unsupported OS: {system}")
        return False
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        print("Automatic installation failed (possibly missing permissions). Run by hand:")
        print("    " + " ".join(cmd))
        return False
    return r_installed()


def missing_packages() -> list[str]:
    expr = ("pk <- c(%s); cat(pk[!vapply(pk, requireNamespace, logical(1), quietly=TRUE)], sep='\\n')"
            % ", ".join(f"'{p}'" for p in R_PACKAGES))
    res = run(["Rscript", "-e", expr])
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def install_packages(pkgs: list[str]) -> None:
    vec = ", ".join(f"'{p}'" for p in pkgs)
    expr = f"install.packages(c({vec}), repos='{CRAN_MIRROR}', Ncpus=4)"
    subprocess.run(["Rscript", "-e", expr], text=True)
    still_missing = [p for p in missing_packages() if p in CRAN_ARCHIVE]
    for pkg in still_missing:
        url = f"{CRAN_MIRROR}/src/contrib/Archive/{CRAN_ARCHIVE[pkg]}"
        print(f"{pkg} is not available on current CRAN; installing archived source {url}")
        subprocess.run(["Rscript", "-e", f"install.packages('{url}', repos=NULL, type='source')"], text=True)


def main() -> int:
    r_ok = r_installed()
    if not r_ok:
        print("R not found; attempting installation.")
        r_ok = install_r()
    if not r_ok:
        print("SUMMARY: R MISSING / packages not checked")
        return 1
    missing = missing_packages()
    if missing:
        print("Installing missing R packages: " + ", ".join(missing))
        install_packages(missing)
        missing = missing_packages()
    pkg_status = "packages OK" if not missing else "packages MISSING: " + ", ".join(missing)
    print(f"SUMMARY: R OK / {pkg_status}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
