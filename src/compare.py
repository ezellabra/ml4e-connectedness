"""Comparison of methods on three criteria: network density, agreement between methods,
and stability over time; plus the FNETS-vs-DY decomposition (common vs idiosyncratic)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .dy import connectedness
from .networks import directed_measures, jaccard, spearman_offdiag, top_links, undirected_measures

NETWORKS = ("gfevd", "granger", "pcor", "lrpc")


def pure_factor_total(cov_common: np.ndarray, var_idio: np.ndarray) -> float:
    """Total connectedness (%) of the pure-factor benchmark: Sigma^B = cov(chi) + diag(var(xi)),
    no lags, theta_ij = Sigma_ij^2 / (Sigma_ii Sigma_jj) normalized by row."""
    S = (cov_common + cov_common.T) / 2 + np.diag(var_idio)
    d = np.diag(S)
    theta = S ** 2 / np.outer(d, d)
    theta = theta / theta.sum(axis=1, keepdims=True)
    return float(connectedness(theta)["total"])


def idiosyncratic_spillover(common_share: np.ndarray, c_from: np.ndarray) -> float:
    """Mean over series of IS_i = (1 - s_i) * C_from_i (idiosyncratic spillovers, % of total variance)."""
    return float(np.mean((1.0 - np.asarray(common_share)) * np.asarray(c_from)))


def _matrix(res: dict, k: int, net: str, H: int) -> np.ndarray | None:
    f = res["fits"][k]
    if f is None:
        return None
    if net == "gfevd":
        return res["thetas"][H][k]
    return f[net]


def window_measures(res: dict, H: int, cfg: dict, mask: np.ndarray | None = None) -> pd.DataFrame:
    """One row per window: GFEVD connectedness and density/strength of each network."""
    thr = cfg["networks"]["link_threshold"]
    labels = res["labels"]
    N = len(labels)
    gthr = cfg["networks"]["gfevd_link_threshold"] or 1.0 / N
    rows = []
    for k, date in enumerate(res["dates"]):
        row = {"date": date, "method": res["method"]}
        f = res["fits"][k]
        if f is None or (mask is not None and mask[k]):
            rows.append(row)
            continue
        theta = res["thetas"][H][k]
        c = connectedness(theta)
        row["total"] = c["total"]
        row.update({f"net_{t}": v for t, v in zip(labels, c["net"])})
        row.update({f"to_{t}": v for t, v in zip(labels, c["to"])})
        row.update({f"from_{t}": v for t, v in zip(labels, c["from"])})
        row["density_gfevd"] = directed_measures(theta, gthr)["density"]
        g = directed_measures(f["granger"], thr)
        row["density_granger"] = g["density"]
        row["strength_granger"] = g["strength_total"]
        row.update({f"netdeg_granger_{t}": v for t, v in zip(labels, g["net_degree"])})
        for net in ("pcor", "lrpc"):
            u = undirected_measures(f[net], thr)
            row[f"density_{net}"] = u["density"]
            row[f"strength_{net}"] = u["strength_total"]
            row.update({f"degree_{net}_{t}": v for t, v in zip(labels, u["degree"])})
        if "common_share" in f:
            row["common_share"] = float(np.mean(f["common_share"]))
            row["q"] = f["q"]
            row["is_mean"] = idiosyncratic_spillover(f["common_share"], c["from"])
            row["cb"] = pure_factor_total(f["cov_common"], f["var_idio"])
        rows.append(row)
    return pd.DataFrame(rows)


def density_table(measures: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = ["total", "density_gfevd", "density_granger", "density_pcor", "density_lrpc",
            "strength_pcor", "strength_lrpc"]
    return pd.DataFrame({m: df[cols].mean() for m, df in measures.items()}).T


def agreement_table(results: dict[str, dict], measures: dict[str, pd.DataFrame], H: int, cfg: dict,
                    mask: np.ndarray | None = None) -> pd.DataFrame:
    """Pairwise agreement between methods."""
    share = cfg["networks"]["top_link_share"]
    labels = next(iter(results.values()))["labels"]
    rows = []
    for a, b in combinations(results, 2):
        ma, mb = measures[a], measures[b]
        row = {"method_a": a, "method_b": b,
               "corr_total": ma["total"].corr(mb["total"]),
               "spearman_total": ma["total"].corr(mb["total"], method="spearman")}
        net_corrs = [ma[f"net_{t}"].corr(mb[f"net_{t}"]) for t in labels]
        row["corr_net_mean"] = float(np.nanmean(net_corrs))
        avg_a = ma[[f"net_{t}" for t in labels]].mean().values
        avg_b = mb[[f"net_{t}" for t in labels]].mean().values
        row["spearman_avg_net_ranking"] = pd.Series(avg_a).corr(pd.Series(avg_b), method="spearman")
        for net in NETWORKS:
            sp, jc = [], []
            for k in range(len(results[a]["dates"])):
                if mask is not None and mask[k]:
                    continue
                A, B = _matrix(results[a], k, net, H), _matrix(results[b], k, net, H)
                if A is None or B is None:
                    continue
                sp.append(spearman_offdiag(np.abs(A), np.abs(B)))
                jc.append(jaccard(top_links(A, share), top_links(B, share)))
            row[f"spearman_{net}"] = float(np.nanmean(sp)) if sp else np.nan
            row[f"jaccard_top_{net}"] = float(np.nanmean(jc)) if jc else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def stability_table(results: dict[str, dict], measures: dict[str, pd.DataFrame], H: int, cfg: dict,
                    mask: np.ndarray | None = None) -> pd.DataFrame:
    """Variability of each method's network across consecutive windows."""
    share = cfg["networks"]["top_link_share"]
    rows = []
    for m, res in results.items():
        row = {"method": m}
        tot = measures[m]["total"]
        row["sd_total"] = tot.std()
        row["mean_abs_change_total"] = tot.diff().abs().mean()
        for net in NETWORKS:
            sp, jc = [], []
            for k in range(1, len(res["dates"])):
                if mask is not None and (mask[k] or mask[k - 1]):
                    continue
                A, B = _matrix(res, k - 1, net, H), _matrix(res, k, net, H)
                if A is None or B is None:
                    continue
                sp.append(spearman_offdiag(np.abs(A), np.abs(B)))
                jc.append(jaccard(top_links(A, share), top_links(B, share)))
            row[f"consecutive_spearman_{net}"] = float(np.nanmean(sp)) if sp else np.nan
            row[f"consecutive_jaccard_top_{net}"] = float(np.nanmean(jc)) if jc else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def common_factor_table(measures: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How much DY connectedness disappears once the common factor is removed (FNETS)."""
    dy, fn = measures["dy"].set_index("date"), measures["fnets"].set_index("date")
    ratio = fn["total"] / dy["total"]
    out = {
        "dy_total_mean": dy["total"].mean(),
        "fnets_idio_total_mean": fn["total"].mean(),
        "share_disappearing_mean": float(1 - ratio.mean()),
        "share_disappearing_median": float(1 - ratio.median()),
        "share_disappearing_min": float(1 - ratio.max()),
        "share_disappearing_max": float(1 - ratio.min()),
        "common_variance_share_mean": fn["common_share"].mean(),
        "is_mean": fn["is_mean"].mean(),
        "cb_mean": fn["cb"].mean(),
        "corr_common_share_dy_total": fn["common_share"].corr(dy["total"]),
        "q_mode": fn["q"].mode().iloc[0] if fn["q"].notna().any() else np.nan,
        "q_share_mode": float((fn["q"] == fn["q"].mode().iloc[0]).mean()) if fn["q"].notna().any() else np.nan,
    }
    if "nets" in measures:
        out["nets_total_mean"] = measures["nets"]["total"].mean()
    return pd.Series(out).to_frame("value")


def numerical_diagnostics(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for m, res in results.items():
        fits = res["fits"]
        valid = [f for f in fits if f is not None]
        rows.append({
            "method": m,
            "windows": len(fits),
            "failed_windows": len(fits) - len(valid),
            "sigma_projected_to_pd": sum(bool(f.get("sigma_corrected", False)) for f in valid),
            "windows_with_clipped_pcor": sum(f.get("pcor_clipped", 0) > 0 for f in valid),
            "windows_with_clipped_lrpc": sum(f.get("lrpc_clipped", 0) > 0 for f in valid),
            "lags_selected": ",".join(map(str, sorted({f["lags"] for f in valid}))) if "lags" in valid[0] else "",
        })
    return pd.DataFrame(rows).set_index("method")
