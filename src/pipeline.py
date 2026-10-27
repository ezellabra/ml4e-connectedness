"""End-to-end pipeline: baseline comparison (Phase 5) and robustness scenarios (Phase 6)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import compare
from .config import load_config, path, set_seed
from .data import load_returns
from .dy import connectedness, rolling_dy
from .estimate import METHODS, estimate, estimate_static
from .rolling import window_ends, windows_touching_year
from .tables import write_latex


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_scenario(name: str, returns: pd.DataFrame, window: int, step: int, horizons: list[int],
                 cfg: dict, exclude_year: int | None = None) -> dict:
    tables = path(cfg, "tables")
    ends = window_ends(len(returns), window, step)
    results = {}
    for m in cfg["compare"]["methods"]:
        log(f"{name}: estimating {m} on {len(ends)} windows (w={window}, step={step})")
        results[m] = estimate(m, returns, window, step, horizons, cfg, ends=ends)
    mask = windows_touching_year(returns.index, ends, window, exclude_year) if exclude_year else None
    summaries = {}
    for H in horizons:
        tag = f"{name}_H{H}"
        measures = {m: compare.window_measures(r, H, cfg, mask) for m, r in results.items()}
        pd.concat(measures.values()).to_csv(tables / f"series_{tag}.csv", index=False, float_format="%.6g")
        dens = compare.density_table(measures)
        agree = compare.agreement_table(results, measures, H, cfg, mask)
        stab = compare.stability_table(results, measures, H, cfg, mask)
        common = compare.common_factor_table(measures)
        dens.to_csv(tables / f"density_{tag}.csv", float_format="%.4f")
        agree.to_csv(tables / f"agreement_{tag}.csv", index=False, float_format="%.4f")
        stab.to_csv(tables / f"stability_{tag}.csv", float_format="%.4f")
        common.to_csv(tables / f"common_factor_{tag}.csv", float_format="%.4f")
        summaries[H] = {"density": dens, "agreement": agree, "stability": stab, "common": common,
                        "n_windows": int(len(ends) - (mask.sum() if mask is not None else 0))}
    compare.numerical_diagnostics(results).to_csv(tables / f"numerical_diagnostics_{name}.csv")
    return summaries


def run_static(name: str, returns: pd.DataFrame, H: int, cfg: dict, save_matrices: bool = False) -> pd.DataFrame:
    """Full-period (single window) estimate for every method."""
    out_dir = path(cfg, "tables") / "static"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for m in cfg["compare"]["methods"]:
        log(f"static {name}: {m}")
        res = estimate_static(m, returns, [H], cfg)
        f = res["fits"][0]
        theta = res["thetas"][H][0]
        c = connectedness(theta, res["labels"])
        row = {"sample": name, "method": m, "total": c["total"]}
        meas = compare.window_measures(res, H, cfg).iloc[0]
        for k in ("density_gfevd", "density_granger", "density_pcor", "density_lrpc", "common_share", "q"):
            if k in meas:
                row[k] = meas[k]
        rows.append(row)
        if save_matrices:
            lab = res["labels"]
            pd.DataFrame(theta * 100, index=lab, columns=lab).to_csv(out_dir / f"{name}_{m}_gfevd.csv", float_format="%.4f")
            for net in ("granger", "pcor", "lrpc"):
                pd.DataFrame(f[net], index=lab, columns=lab).to_csv(out_dir / f"{name}_{m}_{net}.csv", float_format="%.5f")
            pd.DataFrame({"to": c["to"], "from": c["from"], "net": c["net"]}).to_csv(
                out_dir / f"{name}_{m}_directional.csv", float_format="%.4f")
            if "common_share" in f:
                pd.Series(f["common_share"], index=lab, name="common_share").to_csv(
                    out_dir / f"{name}_{m}_common_share.csv", float_format="%.4f")
    return pd.DataFrame(rows)


def robustness_row(scenario: str, s: dict) -> dict:
    dens, agree, stab, common = s["density"], s["agreement"], s["stability"], s["common"]["value"]
    row = {"scenario": scenario, "windows": s["n_windows"]}
    for m in dens.index:
        row[f"total_{m}"] = dens.loc[m, "total"]
    for m in dens.index:
        row[f"density_lrpc_{m}"] = dens.loc[m, "density_lrpc"]
    dy_pairs = agree[agree["method_a"] == "dy"].set_index("method_b")
    for m in dy_pairs.index:
        row[f"corr_total_dy_{m}"] = dy_pairs.loc[m, "corr_total"]
    for m in stab.index:
        row[f"stability_gfevd_{m}"] = stab.loc[m, "consecutive_spearman_gfevd"]
    row["fnets_share_disappearing"] = common["share_disappearing_mean"]
    row["fnets_common_share"] = common["common_variance_share_mean"]
    return row


def main() -> None:
    cfg = load_config()
    set_seed(cfg)
    returns = load_returns(cfg)
    tables = path(cfg, "tables")
    r, rob = cfg["rolling"], cfg["robustness"]
    w, H, step = r["window"], r["horizon"], r["compare_step"]

    # DY at full resolution (step 1) for the time-series figure
    log("DY rolling, step 1")
    dy1 = rolling_dy(returns, w, r["step"], H, cfg["dy"]["max_lag"], cfg["runtime"]["n_jobs"])
    rows = []
    for k, d in enumerate(dy1["dates"]):
        c = connectedness(dy1["tables"][k])
        rows.append({"date": d, "total": c["total"], "lags": dy1["lags"][k],
                     **{f"net_{t}": v for t, v in zip(dy1["labels"], c["net"])}})
    pd.DataFrame(rows).to_csv(tables / "series_dy_step1_H4.csv", index=False, float_format="%.6g")

    # Baseline comparison (horizons for robustness reuse the same fits)
    horizons = sorted({H, *rob["horizons"]})
    rob_rows = []
    base = run_scenario("base", returns, w, step, horizons, cfg)
    rob_rows.append(robustness_row("Baseline (w=104, H=4)", base[H]))
    for h in rob["horizons"]:
        rob_rows.append(robustness_row(f"H={h}", base[h]))
    for ww in rob["windows"]:
        s = run_scenario(f"w{ww}", returns, ww, step, [H], cfg)
        rob_rows.append(robustness_row(f"w={ww}", s[H]))
    no_ref = returns.drop(columns=[rob["exclude_ticker"]])
    s = run_scenario(f"no{rob['exclude_ticker']}", no_ref, w, step, [H], cfg)
    rob_rows.append(robustness_row(f"Excluding {rob['exclude_ticker']}", s[H]))
    s = run_scenario(f"ex{rob['exclude_year']}", returns, w, step, [H], cfg, exclude_year=rob["exclude_year"])
    rob_rows.append(robustness_row(f"Excluding windows touching {rob['exclude_year']}", s[H]))
    rob_df = pd.DataFrame(rob_rows).set_index("scenario")
    rob_df.to_csv(tables / "robustness_summary.csv", float_format="%.4f")

    # Static full-period estimates: full sample and the two subsamples around 2020
    static = [run_static("full", returns, H, cfg, save_matrices=True)]
    for sub in rob["subsamples"]:
        part = returns.loc[sub["start"]:sub["end"]]
        static.append(run_static(sub["name"], part, H, cfg, save_matrices=True))
    static_df = pd.concat(static, ignore_index=True)
    static_df.to_csv(tables / "static_connectedness.csv", index=False, float_format="%.4f")

    write_latex_tables(cfg, base[H], rob_df, static_df)
    meta = {"n_windows_base": base[H]["n_windows"], "first_window_end": str(window_ends(len(returns), w, step)[0])}
    (tables / "run_info.json").write_text(json.dumps(meta, indent=2))
    log("done")


def write_latex_tables(cfg: dict, base: dict, rob_df: pd.DataFrame, static_df: pd.DataFrame) -> None:
    tables = path(cfg, "tables")
    names = {"dy": "DY", "lasso": "Lasso VAR", "nets": "NETS", "fnets": "FNETS (idio.)"}
    dens, stab = base["density"], base["stability"]
    agree = base["agreement"].set_index(["method_a", "method_b"])
    dy_agree = {m: (agree.loc[("dy", m), "corr_total"], agree.loc[("dy", m), "spearman_gfevd"])
                for m in dens.index if m != "dy"}
    rows = []
    for m in dens.index:
        rows.append([names[m],
                     f"{dens.loc[m, 'total']:.1f}",
                     f"{100 * dens.loc[m, 'density_granger']:.1f}",
                     f"{100 * dens.loc[m, 'density_lrpc']:.1f}",
                     "--" if m == "dy" else f"{dy_agree[m][0] + 0.0:.2f}".replace("-0.00", "0.00"),
                     "--" if m == "dy" else f"{dy_agree[m][1] + 0.0:.2f}".replace("-0.00", "0.00"),
                     f"{stab.loc[m, 'consecutive_spearman_gfevd']:.2f}",
                     f"{stab.loc[m, 'consecutive_jaccard_top_lrpc']:.2f}"])
    write_latex(tables / "comparison_criteria.tex",
                ["Method", "Total conn. (\\%)", "Granger dens. (\\%)", "LR p.corr. dens. (\\%)",
                 "Corr. total w/ DY", "Spearman GFEVD w/ DY", "Stab. GFEVD", "Stab. LRPC top"],
                rows, "lrrrrrrr")
    rrows = []
    for sc, r in rob_df.iterrows():
        rrows.append([sc, str(int(r["windows"]))] + [f"{r[f'total_{m}']:.1f}" for m in dens.index]
                     + [f"{100 * r['fnets_share_disappearing']:.0f}", f"{100 * r['fnets_common_share']:.0f}"])
    write_latex(tables / "robustness_summary.tex",
                ["Scenario", "Windows"] + [names[m] for m in dens.index] + ["Lost w/o factor (\\%)", "Common var. (\\%)"],
                rrows, "lr" + "r" * (len(dens.index) + 2))
    srows = []
    for _, r in static_df.iterrows():
        srows.append(["Full sample" if r["sample"] == "full" else r["sample"], names[r["method"]], f"{r['total']:.1f}", f"{100 * r['density_lrpc']:.1f}"])
    write_latex(tables / "static_connectedness.tex", ["Sample", "Method", "Total conn. (\\%)", "LR p.corr. dens. (\\%)"],
                srows, "llrr")


if __name__ == "__main__":
    main()
