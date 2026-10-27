"""Bridge to R: runs r/run_nets.R and r/run_fnets.R through Rscript and reads back CSVs."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import ROOT
from .rolling import window_ends

R_SCRIPTS = {"nets": "run_nets.R", "fnets": "run_fnets.R"}
OUTPUTS = {
    "nets": {"A": "_A.csv", "sigma": "_sigma.csv", "pcor": "_pcor.csv", "lrpc": "_lrpc.csv"},
    "fnets": {"A": "_beta.csv", "sigma": "_gamma.csv", "pcor": "_pcor.csv", "lrpc": "_lrpc.csv",
              "var_x": "_var_x.csv", "var_common": "_var_common.csv", "var_idio": "_var_idio.csv"},
}


class RBridgeError(RuntimeError):
    pass


def rscript_available() -> bool:
    return shutil.which("Rscript") is not None


def _to_r(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_params(params: dict, seed: int, path: Path) -> None:
    rows = [("seed", seed)] + [(k, _to_r(v)) for k, v in params.items()]
    pd.DataFrame(rows, columns=["key", "value"]).to_csv(path, index=False)


def _run_chunk(script: Path, manifest: Path, params: Path, timeout: int) -> str:
    res = subprocess.run(["Rscript", str(script), str(manifest), str(params)],
                         capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RBridgeError(f"Rscript {script.name} failed (code {res.returncode}):\n{res.stderr[-2000:]}")
    return res.stdout


def _read_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path).values.astype(float)


def run_windows(method: str, windows: list[pd.DataFrame], params: dict, seed: int,
                n_jobs: int = 1, timeout: int = 3600, workdir: Path | None = None) -> list[dict | None]:
    """Run an R method on a list of return windows. Returns one dict of arrays per window
    (None if R failed on that window)."""
    if not rscript_available():
        raise RBridgeError("Rscript not found on PATH. Run scripts/setup_r.py first.")
    script = ROOT / "r" / R_SCRIPTS[method]
    tmp = Path(tempfile.mkdtemp(prefix=f"{method}_", dir=workdir))
    try:
        params_path = tmp / "params.csv"
        write_params(params, seed, params_path)
        rows = []
        for i, w in enumerate(windows):
            inp = tmp / f"w{i:04d}.csv"
            w.to_csv(inp, index_label="date")
            rows.append({"id": i, "input": str(inp), "output_prefix": str(tmp / f"o{i:04d}")})
        manifest = pd.DataFrame(rows)
        n_chunks = max(1, min(n_jobs, len(rows)))
        chunks = np.array_split(np.arange(len(rows)), n_chunks)
        paths = []
        for c, idx in enumerate(chunks):
            p = tmp / f"manifest_{c}.csv"
            manifest.iloc[idx].to_csv(p, index=False)
            paths.append(p)
        Parallel(n_jobs=n_chunks, backend="threading")(
            delayed(_run_chunk)(script, p, params_path, timeout) for p in paths)
        out = []
        for r in rows:
            prefix = Path(r["output_prefix"])
            err = Path(str(prefix) + "_error.txt")
            if err.exists():
                warnings.warn(f"{method} window {r['id']} failed: {err.read_text().strip()}")
                out.append(None)
                continue
            d = {k: _read_matrix(Path(str(prefix) + suf)) for k, suf in OUTPUTS[method].items()}
            for k in ("var_x", "var_common", "var_idio"):
                if k in d:
                    d[k] = d[k].ravel()
            d["info"] = pd.read_csv(Path(str(prefix) + "_info.csv")).iloc[0].to_dict()
            out.append(d)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def coefs_from_A(A: np.ndarray) -> np.ndarray:
    """Convert N x (N*p) [A_1 ... A_p] into a (p, N, N) coefficient array."""
    N = A.shape[0]
    p = A.shape[1] // N
    return np.stack([A[:, k * N:(k + 1) * N] for k in range(p)])


def rolling_r(method: str, returns: pd.DataFrame, window: int, step: int, params: dict, seed: int,
              n_jobs: int = 1, timeout: int = 3600, ends: np.ndarray | None = None) -> dict:
    ends = window_ends(len(returns), window, step) if ends is None else ends
    windows = [returns.iloc[e - window + 1:e + 1] for e in ends]
    res = run_windows(method, windows, params, seed, n_jobs, timeout)
    return {"dates": returns.index[ends], "labels": list(returns.columns), "fits": res}
