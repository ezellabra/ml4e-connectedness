"""Load and validate weekly returns, compute descriptive statistics and data checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from .config import ROOT, load_config, metadata_frame, path


class DataValidationError(ValueError):
    pass


def load_returns(cfg: dict | None = None, validate: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_csv(ROOT / cfg["paths"]["returns"], parse_dates=[cfg["data"]["date_column"]])
    df = df.set_index(cfg["data"]["date_column"])
    if validate:
        validate_returns(df, cfg)
    return df


def validate_returns(df: pd.DataFrame, cfg: dict) -> None:
    """Raise DataValidationError if the panel breaks any structural requirement."""
    idx = df.index
    if not idx.is_monotonic_increasing:
        raise DataValidationError("Dates are not sorted in increasing order.")
    if idx.has_duplicates:
        raise DataValidationError("Duplicate dates found.")
    spacing = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    expected = cfg["data"]["expected_spacing_days"]
    if not np.all(spacing == expected):
        bad = idx[1:][spacing != expected]
        raise DataValidationError(f"Non-{expected}-day spacing at: {list(bad[:5])}")
    if df.isna().any().any():
        raise DataValidationError("Missing values found: " + str(df.isna().sum()[lambda s: s > 0].to_dict()))
    if df.columns.has_duplicates:
        raise DataValidationError("Duplicate columns found.")
    meta = metadata_frame(cfg)
    missing_meta = set(df.columns) - set(meta.index)
    if missing_meta:
        raise DataValidationError(f"Tickers without metadata: {sorted(missing_meta)}")


def extreme_returns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    s = df.stack()
    s = s[s.abs() > threshold]
    out = s.rename("log_return").reset_index()
    out.columns = ["date", "ticker", "log_return"]
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def descriptive_stats(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    meta = metadata_frame(cfg)
    rows = []
    for t in df.columns:
        x = df[t].values
        adf_p = adfuller(x, regression=cfg["data"]["adf_regression"], autolag=cfg["data"]["adf_autolag"],
                        result_object=False)[1]
        rows.append({
            "ticker": t,
            "country": meta.loc[t, "country"],
            "region": meta.loc[t, "region"],
            "start": df.index[0].date().isoformat(),
            "n_obs": len(x),
            "mean": x.mean(),
            "std": x.std(ddof=1),
            "min": x.min(),
            "max": x.max(),
            "skewness": stats.skew(x),
            "kurtosis": stats.kurtosis(x, fisher=False),
            "adf_pvalue": adf_p,
        })
    return pd.DataFrame(rows).set_index("ticker")


def emerging_tickers(df: pd.DataFrame, cfg: dict) -> list[str]:
    meta = metadata_frame(cfg)
    return [t for t in df.columns if not meta.loc[t, "reference"]]


def check_covid_drops(df: pd.DataFrame, cfg: dict) -> dict:
    """Claim (a): how many of the largest weekly drops fall in the March-2020 crash.

    Three definitions are reported:
      pooled  - the n most negative country-week returns among the emerging markets;
      per_market - each market's single worst week (how many fall in the crash window);
      average - the n worst weeks of the equal-weighted emerging-market average.
    """
    em = df[emerging_tickers(df, cfg)]
    n = cfg["data"]["n_largest_drops"]
    lo, hi = pd.Timestamp(cfg["data"]["covid_window_start"]), pd.Timestamp(cfg["data"]["covid_window_end"])
    in_march = lambda d: (d.year == 2020) & (d.month == 3)
    in_crash = lambda d: (d >= lo) & (d <= hi)

    pooled = em.stack().sort_values().head(n)
    pooled_dates = pooled.index.get_level_values(0)
    worst_week = pooled_dates.value_counts().idxmax()
    per_market = em.idxmin()
    avg = em.mean(axis=1).sort_values().head(n)
    return {
        "pooled_top": pooled,
        "pooled_in_march2020": int(in_march(pooled_dates).sum()),
        "pooled_in_crash_window": int(in_crash(pooled_dates).sum()),
        "pooled_modal_week": worst_week.date().isoformat(),
        "pooled_in_modal_week": int((pooled_dates == worst_week).sum()),
        "per_market_worst": per_market,
        "per_market_in_march2020": int(in_march(pd.DatetimeIndex(per_market.values)).sum()),
        "per_market_in_crash_window": int(in_crash(pd.DatetimeIndex(per_market.values)).sum()),
        "n_markets": em.shape[1],
        "average_top": avg,
        "average_in_march2020": int(in_march(avg.index).sum()),
    }


def check_var_significance(df: pd.DataFrame, cfg: dict) -> dict:
    """Claim (b): share of significant slope coefficients in a full-sample VAR(1)."""
    p = cfg["data"]["claim_var_lags"]
    alpha = cfg["data"]["claim_alpha"]
    res = VAR(df.values).fit(p, trend="c")
    pv = res.pvalues[1:, :]  # drop intercept row
    return {
        "n_coefficients": int(pv.size),
        "n_significant": int((pv < alpha).sum()),
        "share_significant": float((pv < alpha).mean()),
        "share_significant_offdiag": float((pv[:, :][~np.eye(df.shape[1], dtype=bool)] < alpha).mean()) if p == 1 else None,
        "alpha": alpha,
        "lags": p,
    }


def main() -> None:
    cfg = load_config()
    df = load_returns(cfg)
    tables = path(cfg, "tables")
    desc = descriptive_stats(df, cfg)
    desc.to_csv(tables / "descriptive_stats.csv", float_format="%.6f")
    ext = extreme_returns(df, cfg["data"]["extreme_return_threshold"])
    ext.to_csv(tables / "extreme_returns.csv", index=False, float_format="%.6f")
    covid = check_covid_drops(df, cfg)
    var_sig = check_var_significance(df, cfg)
    print(desc.round(4).to_string())
    print("\nExtreme returns:\n", ext.to_string())
    print("\nClaim (a):", {k: v for k, v in covid.items() if not isinstance(v, (pd.Series,))})
    print(covid["pooled_top"].to_string())
    print(covid["per_market_worst"].to_string())
    print("\nClaim (b):", var_sig)


if __name__ == "__main__":
    main()
