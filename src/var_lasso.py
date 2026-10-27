"""Sparse VAR estimated by equation-wise lasso, with lambda chosen by time-series
cross-validation (TimeSeriesSplit) or by the extended BIC.

Connectedness is computed on the estimated VAR with the same generalized FEVD as DY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import lasso_path
from sklearn.model_selection import TimeSeriesSplit

from .config import load_config
from .dy import VARFit, gfevd, lag_matrix
from .rolling import window_ends


def _alpha_grid(X: np.ndarray, y: np.ndarray, n_alphas: int, min_ratio: float) -> np.ndarray:
    alpha_max = np.max(np.abs(X.T @ (y - y.mean()))) / len(y)
    alpha_max = max(alpha_max, 1e-12)
    return np.geomspace(alpha_max, alpha_max * min_ratio, n_alphas)


def _standardize(X: np.ndarray, y: np.ndarray, standardize: bool):
    mx, my = X.mean(axis=0), y.mean()
    sx = X.std(axis=0) if standardize else np.ones(X.shape[1])
    sx[sx == 0] = 1.0
    return (X - mx) / sx, y - my, mx, my, sx


def _path(X, y, alphas, opts):
    _, coefs, _ = lasso_path(X, y, alphas=alphas, max_iter=opts["max_iter"], tol=opts["tol"])
    return coefs  # (K, n_alphas)


def lasso_equation(X: np.ndarray, y: np.ndarray, opts: dict) -> tuple[np.ndarray, float, float]:
    """Fit one VAR equation. Returns (coefficients on original scale, intercept, chosen alpha)."""
    Xs, yc, mx, my, sx = _standardize(X, y, opts["standardize"])
    alphas = _alpha_grid(Xs, yc, opts["n_alphas"], opts["alpha_min_ratio"])
    if opts["selection"] == "cv":
        err = np.zeros(len(alphas))
        for tr, te in TimeSeriesSplit(n_splits=opts["cv_splits"]).split(X):
            Xtr, ytr, mx_tr, my_tr, sx_tr = _standardize(X[tr], y[tr], opts["standardize"])
            coefs = _path(Xtr, ytr, alphas, opts)
            pred = ((X[te] - mx_tr) / sx_tr) @ coefs + my_tr
            err += ((y[te][:, None] - pred) ** 2).mean(axis=0)
        k = int(np.argmin(err))
        coef = _path(Xs, yc, alphas, opts)[:, k]
    elif opts["selection"] == "ebic":
        coefs = _path(Xs, yc, alphas, opts)
        n, K = Xs.shape
        rss = ((yc[:, None] - Xs @ coefs) ** 2).sum(axis=0)
        df = (np.abs(coefs) > 0).sum(axis=0)
        ebic = n * np.log(rss / n) + df * np.log(n) + 2 * opts["ebic_gamma"] * df * np.log(K)
        k = int(np.argmin(ebic))
        coef = coefs[:, k]
    else:
        raise ValueError(f"Unknown lambda selection: {opts['selection']}")
    beta = coef / sx
    return beta, my - mx @ beta, alphas[k]


def fit_var_lasso(y: np.ndarray, p: int, opts: dict) -> tuple[VARFit, np.ndarray]:
    Y, X = lag_matrix(y, p)
    T_eff, N = Y.shape
    B = np.zeros((N, N * p))
    c = np.zeros(N)
    alphas = np.zeros(N)
    for i in range(N):
        B[i], c[i], alphas[i] = lasso_equation(X, Y[:, i], opts)
    resid = Y - X @ B.T - c
    sigma = resid.T @ resid / T_eff
    coefs = B.reshape(N, p, N).transpose(1, 0, 2)
    return VARFit(c, coefs, sigma, resid, p), alphas


def lasso_window(y: np.ndarray, horizon: int, opts: dict):
    fit, alphas = fit_var_lasso(y, opts["max_lag"], opts)
    return gfevd(fit.coefs, fit.sigma, horizon), fit, alphas


def rolling_lasso(returns: pd.DataFrame, window: int, step: int, horizon: int, opts: dict,
                  n_jobs: int = 1) -> dict:
    y = returns.values
    ends = window_ends(len(returns), window, step)
    res = Parallel(n_jobs=n_jobs)(
        delayed(lasso_window)(y[e - window + 1:e + 1], horizon, opts) for e in ends)
    return {
        "dates": returns.index[ends],
        "tables": np.stack([r[0] for r in res]),
        "labels": list(returns.columns),
        "coefs": [r[1].coefs for r in res],
        "sigmas": np.stack([r[1].sigma for r in res]),
        "alphas": np.stack([r[2] for r in res]),
    }


def main() -> None:
    from .data import load_returns
    from .dy import connectedness
    cfg = load_config()
    df = load_returns(cfg)
    for sel in ("cv", "ebic"):
        opts = dict(cfg["var_lasso"], selection=sel)
        theta, fit, _ = lasso_window(df.values, cfg["rolling"]["horizon"], opts)
        c = connectedness(theta)
        nz = (np.abs(fit.coefs) > 0).mean()
        print(f"Full-sample lasso VAR ({sel}): total={c['total']:.2f}%, nonzero coef share={nz:.3f}")


if __name__ == "__main__":
    main()
