"""Unified estimation layer: runs every method on a common set of windows and returns,
for each window, a VAR (coefficients and innovation covariance) plus network matrices.

Every method is mapped to the same objects so that `src/compare.py` can treat them alike:
  coefs  (p, N, N)  VAR coefficients (idiosyncratic VAR for FNETS)
  sigma  (N, N)     innovation covariance (positive definite)
  granger (N, N)    directed lag network, [i, j] = sum_k |A_k[i, j]| (link j -> i)
  pcor   (N, N)     contemporaneous partial correlations of the innovations
  lrpc   (N, N)     long-run partial correlations
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import r_bridge
from .config import path
from .dy import fit_var_ols, gfevd, select_lag_bic
from .networks import clip_pcor, nearest_pd
from .rolling import window_ends
from .var_lasso import fit_var_lasso

METHODS = ("dy", "lasso", "nets", "fnets")


def partial_cor(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.abs(np.diag(K)))
    pc = -K / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pc


def var_networks(coefs: np.ndarray, sigma: np.ndarray) -> dict:
    """Granger, contemporaneous and long-run partial-correlation networks of a VAR."""
    N = sigma.shape[0]
    K = np.linalg.inv(sigma)
    A1 = np.eye(N) - coefs.sum(axis=0)
    return {
        "granger": np.abs(coefs).sum(axis=0),
        "pcor": partial_cor(K),
        "lrpc": partial_cor(A1.T @ K @ A1),
    }


def _fit_dy(y: np.ndarray, cfg: dict) -> dict:
    p = select_lag_bic(y, cfg["dy"]["max_lag"])
    fit = fit_var_ols(y, p)
    return {"coefs": fit.coefs, "sigma": fit.sigma, "lags": p, **var_networks(fit.coefs, fit.sigma)}


def _fit_lasso(y: np.ndarray, cfg: dict) -> dict:
    opts = cfg["var_lasso"]
    fit, alphas = fit_var_lasso(y, opts["max_lag"], opts)
    nz = np.abs(fit.coefs) > opts["link_threshold"]
    return {"coefs": fit.coefs, "sigma": fit.sigma, "alpha_mean": float(alphas.mean()),
            "nonzero_share": float(nz.mean()), **var_networks(fit.coefs, fit.sigma)}


def _from_r(method: str, r: dict | None, cfg: dict) -> dict | None:
    if r is None:
        return None
    floor = cfg["networks"]["pd_floor_ratio"]
    coefs = r_bridge.coefs_from_A(r["A"])
    sigma, corrected = nearest_pd(r["sigma"], floor)
    pcor, bad_pc = clip_pcor(r["pcor"])
    lrpc, bad_lr = clip_pcor(r["lrpc"])
    out = {"coefs": coefs, "sigma": sigma, "granger": np.abs(coefs).sum(axis=0), "pcor": pcor,
           "lrpc": lrpc, "sigma_corrected": corrected, "pcor_clipped": bad_pc,
           "lrpc_clipped": bad_lr, "info": r["info"]}
    if method == "fnets":
        out["common_share"] = r["var_common"] / r["var_x"]
        out["q"] = int(r["info"]["q"])
        out["cov_common"] = r["cov_common"]
        out["var_idio"] = r["var_idio"]
    return out


def fit_windows(method: str, returns: pd.DataFrame, ends: np.ndarray, window: int, cfg: dict) -> list:
    n_jobs = cfg["runtime"]["n_jobs"]
    y = returns.values
    if method in ("dy", "lasso"):
        fn = _fit_dy if method == "dy" else _fit_lasso
        return Parallel(n_jobs=n_jobs)(delayed(fn)(y[e - window + 1:e + 1], cfg) for e in ends)
    res = r_bridge.rolling_r(method, returns, window, 1, cfg[method], cfg["seed"], n_jobs,
                             cfg["runtime"]["r_timeout_seconds"], ends=ends)
    return [_from_r(method, r, cfg) for r in res["fits"]]


def _cache_key(method: str, returns: pd.DataFrame, ends: np.ndarray, window: int, cfg: dict) -> str:
    h = hashlib.sha1()
    h.update(method.encode())
    h.update(",".join(returns.columns).encode())
    h.update(str(returns.index[0]).encode() + str(returns.index[-1]).encode())
    h.update(np.asarray(ends).tobytes())
    h.update(str(window).encode())
    h.update(repr(cfg.get(method if method != "dy" else "dy", {})).encode())
    h.update(repr(cfg.get("var_lasso") if method == "lasso" else "").encode())
    h.update(repr(cfg["networks"]).encode())
    if method == "fnets":
        h.update(b"v2-cov_common")   # fits now also store the common-component covariance
    return h.hexdigest()[:16]


def estimate(method: str, returns: pd.DataFrame, window: int, step: int, horizons: list[int],
             cfg: dict, ends: np.ndarray | None = None, use_cache: bool = True) -> dict:
    """Fit `method` on rolling windows and compute GFEVD tables for each horizon."""
    ends = window_ends(len(returns), window, step) if ends is None else np.asarray(ends)
    cache = path(cfg, "processed") / f"fits_{method}_{_cache_key(method, returns, ends, window, cfg)}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as fh:
            fits = pickle.load(fh)
    else:
        fits = fit_windows(method, returns, ends, window, cfg)
        with open(cache, "wb") as fh:
            pickle.dump(fits, fh)
    N = returns.shape[1]
    thetas = {}
    for H in horizons:
        thetas[H] = np.stack([gfevd(f["coefs"], f["sigma"], H) if f is not None else np.full((N, N), np.nan)
                              for f in fits])
    return {"method": method, "dates": returns.index[ends], "labels": list(returns.columns),
            "window": window, "fits": fits, "thetas": thetas}


def estimate_static(method: str, returns: pd.DataFrame, horizons: list[int], cfg: dict) -> dict:
    """Single full-period estimate (window = whole sample)."""
    n = len(returns)
    return estimate(method, returns, n, 1, horizons, cfg, ends=np.array([n - 1]))
