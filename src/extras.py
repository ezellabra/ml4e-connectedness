"""Additional robustness checks, per-country table and block bootstrap (paper revisions).

  * NETS with a finer penalty grid and FNETS with a fixed number of factors (baseline windows, H=4);
  * extended robustness table (idiosyncratic spillover share and pure-factor benchmark);
  * per-country table (common share, average correlation, average net connectedness);
  * moving-block bootstrap of the full-sample estimates;
  * all numbers quoted in the text, written to results/tables/placeholders.json.
Run after src/pipeline.py (it reads the pipeline's fits from the cache).
"""
from __future__ import annotations

import copy
import json
import math
import pickle
import time

import numpy as np
import pandas as pd

from . import compare, r_bridge
from .config import load_config, metadata_frame, path, set_seed
from .data import load_returns
from .dy import connectedness, dy_window
from .estimate import _from_r, estimate, estimate_static
from .rolling import window_ends
from .tables import write_latex
from .dy import gfevd

ORDINALS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
            "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
            "seventeenth", "eighteenth"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _measures(res: dict, H: int, cfg: dict) -> pd.DataFrame:
    return compare.window_measures(res, H, cfg)


def full_sample_stats(fit: dict, N: int) -> dict:
    """Full-sample quantities from one FNETS fit (dict as returned by estimate._from_r)."""
    theta = gfevd(fit["coefs"], fit["sigma"], 4)
    c = connectedness(theta)
    return {"fnets_total": c["total"], "common_share": 100 * float(np.mean(fit["common_share"])),
            "is_full": compare.idiosyncratic_spillover(fit["common_share"], c["from"]),
            "cb_full": compare.pure_factor_total(fit["cov_common"], fit["var_idio"])}


def bootstrap(cfg: dict, returns: pd.DataFrame) -> dict:
    """Moving-block bootstrap of the full-sample weekly returns; every estimation step is repeated."""
    b = cfg["extras"]["bootstrap"]
    H, max_lag = cfg["rolling"]["horizon"], cfg["dy"]["max_lag"]
    T, L = len(returns), b["block_length"]
    nb = math.ceil(T / L)
    rng = np.random.default_rng(b["seed"])
    n_total = b["n_samples"]
    # all index draws up front, so that the first B' samples do not depend on B
    idx = [np.concatenate([np.arange(s, s + L) for s in rng.integers(0, T - L + 1, size=nb)])[:T]
           for _ in range(n_total)]
    cache = path(cfg, "processed") / f"bootstrap_s{b['seed']}_L{L}_T{T}_{hash_cfg(cfg)}.pkl"
    state = pickle.load(open(cache, "rb")) if cache.exists() else {"rows": [], "seconds": 0.0}
    t_run = time.time()
    while len(state["rows"]) < n_total:
        lo = len(state["rows"])
        hi = min(lo + b["batch_size"], n_total)
        t0 = time.time()
        samples = [returns.iloc[idx[k]].set_axis(returns.index) for k in range(lo, hi)]
        out = r_bridge.run_windows("fnets", samples, cfg["fnets"], b["seed"] + lo, cfg["runtime"]["n_jobs"],
                                   cfg["runtime"]["r_timeout_seconds"])
        for smp, r in zip(samples, out):
            theta, p, _ = dy_window(smp.values, H, max_lag)
            row = {"dy_total": connectedness(theta)["total"]}
            if r is None:
                row.update({k: np.nan for k in ("fnets_total", "common_share", "is_full", "cb_full")})
            else:
                row.update(full_sample_stats(_from_r("fnets", r, cfg), returns.shape[1]))
            state["rows"].append(row)
        state["seconds"] += time.time() - t0
        pickle.dump(state, open(cache, "wb"))
        done = len(state["rows"])
        log(f"bootstrap {done}/{n_total}, {state['seconds'] / 60:.1f} min so far")
        projected_h = state["seconds"] / done * n_total / 3600
        if done == hi and lo == 0:
            log(f"projected total runtime for B={n_total}: {projected_h:.2f} h")
    df = pd.DataFrame(state["rows"])
    return {"samples": df, "seconds": state["seconds"], "B": len(df)}


def hash_cfg(cfg: dict) -> str:
    import hashlib
    return hashlib.sha1(repr((cfg["fnets"], cfg["dy"], cfg["rolling"]["horizon"], cfg["networks"])).encode()).hexdigest()[:8] + "v2"


def main() -> None:
    cfg = load_config()
    set_seed(cfg)
    returns = load_returns(cfg)
    tables = path(cfg, "tables")
    ex = cfg["extras"]
    w, H, step = cfg["rolling"]["window"], cfg["rolling"]["horizon"], cfg["rolling"]["compare_step"]
    ends = window_ends(len(returns), w, step)
    labels = list(returns.columns)
    ph: dict = {}

    # --- baseline fits (cached) and measures
    base = {m: estimate(m, returns, w, step, [H], cfg, ends=ends) for m in cfg["compare"]["methods"]}
    meas = {m: _measures(r, H, cfg) for m, r in base.items()}
    dy_tot = meas["dy"].set_index("date")["total"]
    fn = meas["fnets"].set_index("date")

    # --- A2/A3 baseline
    ph["IS"] = fn["is_mean"].mean()
    ph["IS_min"], ph["IS_max"] = fn["is_mean"].min(), fn["is_mean"].max()
    ph["CB"] = fn["cb"].mean()
    ph["rB"] = fn["cb"].corr(dy_tot)
    ph["RB"] = 100 * ph["CB"] / ex["dy_reference_total"]
    full = {m: estimate_static(m, returns, [H], cfg) for m in ("dy", "fnets")}
    fstats = full_sample_stats(full["fnets"]["fits"][0], len(labels))
    ph["IS_full"], ph["CB_full"] = fstats["is_full"], fstats["cb_full"]
    ph["full_dy_total"] = connectedness(full["dy"]["thetas"][H][0])["total"]
    ph["full_fnets_total"], ph["full_common_share"] = fstats["fnets_total"], fstats["common_share"]
    pd.DataFrame({"IS_w": fn["is_mean"], "CB_w": fn["cb"]}).to_csv(tables / "is_cb_windows_base_H4.csv",
                                                                  float_format="%.6g")

    # --- A5: NETS, 60-value grid
    cfg_n = copy.deepcopy(cfg)
    cfg_n["nets"]["lambda_grid_size"] = ex["nets_fine_grid_size"]
    log("NETS with the finer penalty grid")
    nets60 = estimate("nets", returns, w, step, [H], cfg_n, ends=ends)
    m60 = _measures(nets60, H, cfg_n)
    ph["N60"] = m60["total"].mean()
    ph["N60_jump"] = m60["total"].diff().abs().mean()
    ph["N60_corr_dy"] = m60["total"].corr(meas["dy"]["total"])
    ph["N15_jump"] = meas["nets"]["total"].diff().abs().mean()
    ph["N15"] = meas["nets"]["total"].mean()

    # --- A6: FNETS with q fixed
    cfg_q = copy.deepcopy(cfg)
    cfg_q["fnets"]["q_method"] = ex["fnets_fixed_q"]
    log("FNETS with q fixed")
    fq = estimate("fnets", returns, w, step, [H], cfg_q, ends=ends)
    mq = _measures(fq, H, cfg_q)
    mq_i = mq.set_index("date")
    ph["cs2"] = 100 * mq["common_share"].mean()
    ph["F2"] = mq["total"].mean()
    ph["lost2"] = 100 * (1 - mq_i["total"] / dy_tot).mean()
    ph["IS_q2"], ph["CB_q2"] = mq["is_mean"].mean(), mq["cb"].mean()
    ph["q_values_q2"] = sorted(set(int(x) for x in mq["q"].dropna()))
    ph["lost_base"] = 100 * (1 - fn["total"] / dy_tot).mean()
    d = ph["lost2"] - 41
    ph["q_case"] = 1 if abs(d) <= ex["q_case_tolerance"] else (2 if ph["lost2"] > ex["q_case_upper"] else
                                                                (3 if ph["lost2"] < ex["q_case_lower"] else 0))

    # --- extended Table 3
    rob = pd.read_csv(tables / "robustness_summary.csv", index_col=0)
    names = {"dy": "DY", "lasso": "Lasso VAR", "nets": "NETS", "fnets": "FNETS (idio.)"}
    rows = rob.copy()
    b0 = rob.iloc[0]
    r_n = b0.copy(); r_n["total_nets"] = ph["N60"]
    r_q = b0.copy()
    r_q["total_fnets"], r_q["fnets_share_disappearing"] = ph["F2"], ph["lost2"] / 100
    r_q["fnets_common_share"], r_q["fnets_is_mean"], r_q["fnets_cb_mean"] = ph["cs2"] / 100, ph["IS_q2"], ph["CB_q2"]
    rows.loc["NETS, 60-value grid"] = r_n
    rows.loc["FNETS, q = 2 (fixed)"] = r_q
    rows.to_csv(tables / "robustness_extended.csv", float_format="%.4f")
    tex = [[sc, str(int(r["windows"]))] + [f"{r[f'total_{m}']:.1f}" for m in names]
           + [f"{100 * r['fnets_share_disappearing']:.0f}", f"{100 * r['fnets_common_share']:.0f}",
              f"{r['fnets_is_mean']:.1f}", f"{r['fnets_cb_mean']:.1f}"] for sc, r in rows.iterrows()]
    write_latex(tables / "robustness_extended.tex",
                ["Scenario", "Windows"] + list(names.values()) + ["Lost w/o factor (\\%)", "Common var. (\\%)",
                 "Idio. spill. (\\% total var.)", "Pure-factor bench. (\\%)"], tex, "lr" + "r" * 8)

    # --- A4: per-country table
    meta = metadata_frame(cfg)
    corr = returns.corr().values
    avg_corr = (corr.sum(axis=1) - 1) / (len(labels) - 1)
    cs = np.nanmean(np.stack([f["common_share"] for f in base["fnets"]["fits"] if f is not None]), axis=0) * 100
    series = pd.read_csv(tables / "series_base_H4.csv")
    net = {m: series[series["method"] == m][[f"net_{t}" for t in labels]].mean().values for m in ("dy", "nets", "fnets")}
    ctry = pd.DataFrame({"country": meta.loc[labels, "country"].values, "common_share": cs, "avg_corr": avg_corr,
                         "net_dy": net["dy"], "net_nets": net["nets"], "net_fnets": net["fnets"]}, index=labels)
    ctry.to_csv(tables / "country_table.csv", float_format="%.4f")
    desc = pd.read_csv(tables / "descriptive_stats.csv", index_col=0)   # same row order as Table 1
    order = list(desc.index)
    write_latex(tables / "country_table.tex",
                ["Ticker", "Country", "Common share (\\%)", "Avg. corr.", "Net DY (pp)", "Net NETS (pp)", "Net FNETS (pp)"],
                [[t, ctry.loc[t, "country"], f"{ctry.loc[t, 'common_share']:.1f}", f"{ctry.loc[t, 'avg_corr']:.2f}",
                  f"{ctry.loc[t, 'net_dy']:+.1f}", f"{ctry.loc[t, 'net_nets']:+.1f}", f"{ctry.loc[t, 'net_fnets']:+.1f}"]
                 for t in order], "llrrrrr")
    foc = cfg["focus_ticker"]
    ph["cs_ARG"] = ctry.loc[foc, "common_share"]
    ph["rank_ARG"] = int((ctry["common_share"] > ctry.loc[foc, "common_share"]).sum() + 1)
    ph["ord_ARG"] = ORDINALS[ph["rank_ARG"]]
    ph["spy_dy"], ph["spy_f"] = ctry.loc["SPY", "net_dy"], ctry.loc["SPY", "net_fnets"]

    # --- A7: bootstrap
    bcfg = ex["bootstrap"]
    t0 = time.time()
    bs = bootstrap(cfg, returns)
    lo, hi = bcfg["ci"]
    est = {"dy_total": ph["full_dy_total"], "fnets_total": ph["full_fnets_total"],
           "common_share": ph["full_common_share"], "is_full": ph["IS_full"], "cb_full": ph["CB_full"]}
    rows_b = []
    for k, label in [("dy_total", "DY total connectedness"), ("fnets_total", "FNETS idiosyncratic connectedness"),
                     ("common_share", "Common share"), ("is_full", "Idiosyncratic spillover share of total variance"),
                     ("cb_full", "Pure-factor benchmark")]:
        q = np.nanpercentile(bs["samples"][k], [lo, hi])
        rows_b.append({"quantity": label, "estimate": est[k], "lo": q[0], "hi": q[1]})
    bdf = pd.DataFrame(rows_b)
    bdf.to_csv(tables / "bootstrap_full_sample.csv", index=False, float_format="%.4f")
    bs["samples"].to_csv(tables / "bootstrap_samples.csv", index=False, float_format="%.6g")
    write_latex(tables / "bootstrap_full_sample.tex", ["Quantity", "Estimate (\\%)", "2.5\\%", "97.5\\%"],
                [[r.quantity, f"{r.estimate:.1f}", f"{r.lo:.1f}", f"{r.hi:.1f}"] for r in bdf.itertuples()], "lrrr")
    ph["IS_full_lo"], ph["IS_full_hi"] = bdf.loc[3, "lo"], bdf.loc[3, "hi"]
    ph["bootstrap_B"], ph["bootstrap_seconds"] = bs["B"], bs["seconds"]
    ph["bootstrap_failed"] = int(bs["samples"]["fnets_total"].isna().sum())

    # --- placeholders
    ph["IS_round"], ph["CB_round"] = int(round(ph["IS"])), int(round(ph["CB"]))
    out = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in ph.items()}
    (tables / "placeholders.json").write_text(json.dumps(out, indent=2, default=str))
    log("extras done")


if __name__ == "__main__":
    main()
