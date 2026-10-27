"""Optional out-of-sample forecasting module (Phase 7), off by default (forecast.enabled).

One-week-ahead forecasts from a rolling or expanding window for: AR(p) per series (benchmark,
p by BIC), the sparse lasso VAR, and FNETS (`predict.fnets` in R). Metrics: RMSE and MAE
relative to AR, and the Diebold-Mariano test (Harvey-Leybourne-Newbold small-sample correction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import r_bridge
from .config import load_config, path
from .data import load_returns
from .dy import fit_var_ols, lag_matrix
from .var_lasso import fit_var_lasso


def ar_forecast(y: np.ndarray, max_lag: int) -> np.ndarray:
    """One-step forecast of each column with a univariate AR(p), p in 0..max_lag by BIC."""
    out = np.empty(y.shape[1])
    for j in range(y.shape[1]):
        x = y[:, [j]]
        best, best_bic = x.mean(), np.log(x.var()) if x.var() > 0 else np.inf
        T0 = len(x) - max_lag
        for p in range(1, max_lag + 1):
            fit = fit_var_ols(x, p, start=max_lag)
            s2 = (fit.resid ** 2).mean()
            bic = np.log(s2) + np.log(T0) / T0 * (p + 1)
            if bic < best_bic:
                lags = x[::-1][:p, 0]
                best, best_bic = fit.intercept[0] + fit.coefs[:, 0, 0] @ lags, bic
        out[j] = best
    return out


def var_forecast(intercept: np.ndarray, coefs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return intercept + sum(coefs[k] @ y[-k - 1] for k in range(coefs.shape[0]))


def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray, h: int = 1, power: int = 2) -> tuple[float, float]:
    """DM test of equal accuracy; negative statistic means model a is more accurate.
    Harvey, Leybourne and Newbold (1997) correction, Student-t reference distribution."""
    d = np.abs(e_a) ** power - np.abs(e_b) ** power
    T = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=T - 1))


def run(cfg: dict) -> pd.DataFrame:
    fc = cfg["forecast"]
    returns = load_returns(cfg)
    y = returns.values
    w = fc["window"]
    targets = range(w, len(y))
    starts = [t - w if fc["scheme"] == "rolling" else 0 for t in targets]
    preds = {"ar": [], "lasso": []}
    for s, t in zip(starts, targets):
        train = y[s:t]
        preds["ar"].append(ar_forecast(train, fc["ar_max_lag"]))
        fit, _ = fit_var_lasso(train, cfg["var_lasso"]["max_lag"], cfg["var_lasso"])
        preds["lasso"].append(var_forecast(fit.intercept, fit.coefs, train))
    params = dict(cfg["fnets"], fc_restricted=fc["fnets_fc_restricted"])
    windows = [returns.iloc[s:t] for s, t in zip(starts, targets)]
    res = r_bridge.run_windows("fnets_forecast", windows, params, cfg["seed"], cfg["runtime"]["n_jobs"],
                               cfg["runtime"]["r_timeout_seconds"])
    preds["fnets"] = [r["forecast"].ravel() if r is not None else np.full(y.shape[1], np.nan) for r in res]
    actual = y[list(targets)]
    err = {m: actual - np.array(p) for m, p in preds.items()}
    rows = []
    for m in ("lasso", "fnets"):
        for j, t in enumerate(returns.columns):
            ok = ~np.isnan(err[m][:, j])
            rmse = np.sqrt(np.mean(err[m][ok, j] ** 2)) / np.sqrt(np.mean(err["ar"][ok, j] ** 2))
            mae = np.mean(np.abs(err[m][ok, j])) / np.mean(np.abs(err["ar"][ok, j]))
            dm, pv = diebold_mariano(err[m][ok, j], err["ar"][ok, j], fc["dm_horizon"])
            rows.append({"model": m, "ticker": t, "rel_rmse": rmse, "rel_mae": mae, "dm_stat": dm, "dm_pvalue": pv})
    out = pd.DataFrame(rows)
    out.to_csv(path(cfg, "tables") / "forecast_evaluation.csv", index=False, float_format="%.4f")
    return out


def main() -> None:
    cfg = load_config()
    if not cfg["forecast"]["enabled"]:
        print("Forecasting module disabled (forecast.enabled = false in config.yaml).")
        return
    print(run(cfg).groupby("model")[["rel_rmse", "rel_mae"]].mean())


if __name__ == "__main__":
    main()
