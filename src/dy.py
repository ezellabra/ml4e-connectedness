"""Diebold-Yilmaz (2014) connectedness: OLS VAR + generalized FEVD (Pesaran-Shin).

The functions `ma_coefficients`, `gfevd` and `connectedness` are shared by every method
that delivers a VAR (OLS, lasso, NETS, FNETS idiosyncratic VAR), so that all methods are
measured on the same scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import load_config
from .rolling import window_ends


@dataclass
class VARFit:
    intercept: np.ndarray          # (N,)
    coefs: np.ndarray              # (p, N, N); coefs[k][i, j] = effect of y_j(t-k-1) on y_i(t)
    sigma: np.ndarray              # (N, N) residual covariance
    resid: np.ndarray              # (T - p, N)
    lags: int


def lag_matrix(y: np.ndarray, p: int, start: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y, X) with X = [y_{t-1}, ..., y_{t-p}] for t = start..T-1 (start defaults to p)."""
    start = p if start is None else start
    T = y.shape[0]
    Y = y[start:]
    X = np.hstack([y[start - k:T - k] for k in range(1, p + 1)])
    return Y, X


def fit_var_ols(y: np.ndarray, p: int, start: int | None = None) -> VARFit:
    Y, X = lag_matrix(y, p, start)
    T_eff, N = Y.shape
    Z = np.hstack([np.ones((T_eff, 1)), X])
    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    resid = Y - Z @ B
    dof = T_eff - Z.shape[1]
    sigma = resid.T @ resid / dof
    coefs = B[1:].T.reshape(N, p, N).transpose(1, 0, 2)
    return VARFit(B[0], coefs, sigma, resid, p)


def select_lag_bic(y: np.ndarray, max_lag: int) -> int:
    """BIC over p = 1..max_lag on a common estimation sample (first max_lag obs dropped)."""
    best_p, best_bic = 1, np.inf
    N = y.shape[1]
    for p in range(1, max_lag + 1):
        fit = fit_var_ols(y, p, start=max_lag)
        T_eff = fit.resid.shape[0]
        sigma_ml = fit.resid.T @ fit.resid / T_eff
        _, logdet = np.linalg.slogdet(sigma_ml)
        bic = logdet + np.log(T_eff) / T_eff * (p * N * N + N)
        if bic < best_bic:
            best_p, best_bic = p, bic
    return best_p


def ma_coefficients(coefs: np.ndarray, horizon: int) -> np.ndarray:
    """MA(inf) matrices Psi_0..Psi_{H-1} of a VAR with coefficient array (p, N, N)."""
    p, N, _ = coefs.shape
    psi = np.zeros((horizon, N, N))
    psi[0] = np.eye(N)
    for h in range(1, horizon):
        for k in range(1, min(h, p) + 1):
            psi[h] += coefs[k - 1] @ psi[h - k]
    return psi


def gfevd(coefs: np.ndarray, sigma: np.ndarray, horizon: int, normalize: bool = True) -> np.ndarray:
    """Generalized FEVD (Pesaran and Shin, 1998), H-step-ahead.

    theta[i, j] = share of the H-step forecast-error variance of variable i due to shocks
    in variable j. With normalize=True each row sums to one (Diebold and Yilmaz, 2012, 2014).
    """
    psi = ma_coefficients(coefs, horizon)
    num = np.zeros_like(sigma)
    den = np.zeros(sigma.shape[0])
    for h in range(horizon):
        ps = psi[h] @ sigma
        num += ps ** 2
        den += np.einsum("ij,ij->i", ps, psi[h])
    theta = num / np.diag(sigma)[None, :] / den[:, None]
    if normalize:
        theta = theta / theta.sum(axis=1, keepdims=True)
    return theta


def connectedness(theta: np.ndarray, labels: list[str] | None = None) -> dict:
    """Connectedness measures in percent from a row-normalized variance decomposition."""
    N = theta.shape[0]
    off = theta * 100.0
    np.fill_diagonal(off, 0.0)
    from_ = off.sum(axis=1)          # received by i from others
    to = off.sum(axis=0)             # transmitted by j to others
    out = {
        "total": off.sum() / N,
        "to": to,
        "from": from_,
        "net": to - from_,
        "pairwise_net": off.T - off,  # [i, j] = theta_ji - theta_ij: net flow from i to j
        "table": theta * 100.0,
    }
    if labels is not None:
        for k in ("to", "from", "net"):
            out[k] = pd.Series(out[k], index=labels)
        out["table"] = pd.DataFrame(out["table"], index=labels, columns=labels)
        out["pairwise_net"] = pd.DataFrame(out["pairwise_net"], index=labels, columns=labels)
    return out


def dy_window(y: np.ndarray, horizon: int, max_lag: int) -> tuple[np.ndarray, int, VARFit]:
    p = select_lag_bic(y, max_lag)
    fit = fit_var_ols(y, p)
    return gfevd(fit.coefs, fit.sigma, horizon), p, fit


def rolling_dy(returns: pd.DataFrame, window: int, step: int, horizon: int, max_lag: int,
               n_jobs: int = 1) -> dict:
    """Rolling DY. Returns dict with dates (window end), tables (n_win, N, N) and lags."""
    y = returns.values
    ends = window_ends(len(returns), window, step)
    res = Parallel(n_jobs=n_jobs)(
        delayed(dy_window)(y[e - window + 1:e + 1], horizon, max_lag) for e in ends)
    return {
        "dates": returns.index[ends],
        "tables": np.stack([r[0] for r in res]),
        "lags": np.array([r[1] for r in res]),
        "labels": list(returns.columns),
        "coefs": [r[2].coefs for r in res],
        "sigmas": np.stack([r[2].sigma for r in res]),
    }


def main() -> None:
    from .data import load_returns
    cfg = load_config()
    df = load_returns(cfg)
    theta, p, _ = dy_window(df.values, cfg["rolling"]["horizon"], cfg["dy"]["max_lag"])
    c = connectedness(theta, list(df.columns))
    print(f"Full-sample DY: p={p}, total connectedness={c['total']:.2f}%")
    print(c["net"].round(2).to_string())


if __name__ == "__main__":
    main()
