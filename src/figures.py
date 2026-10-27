"""Figures and descriptive table (Phase 8). Reads outputs written by src/pipeline.py.

All figures are saved as PNG and PDF. Series are distinguished by line style and marker as
well as color, so they remain readable in grayscale.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from .config import load_config, metadata_frame, path
from .tables import write_latex

METHOD_NAMES = {"dy": "Diebold-Yilmaz (OLS VAR)", "lasso": "Sparse VAR (lasso)",
                "nets": "NETS", "fnets": "FNETS (idiosyncratic)"}
METHOD_STYLE = {"dy": ("#2a78d6", "-"), "lasso": ("#eb6834", "--"),
                "nets": ("#1baf7a", "-."), "fnets": ("#008300", ":")}
REGION_COLOR = {"Latin America": "#2a78d6", "Asia": "#eb6834", "EMEA": "#1baf7a", "Global": "#6b6b6b"}
REGION_HATCH = {"Latin America": "", "Asia": "//", "EMEA": "..", "Global": "xx"}
INK = "#2b2b2b"
MUTED = "#6b6b6b"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def setup(cfg: dict) -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 1.4,
        "savefig.dpi": cfg["figures"]["dpi"], "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, cfg: dict, name: str) -> None:
    out = path(cfg, "figures")
    for fmt in cfg["figures"]["formats"]:
        fig.savefig(out / f"{name}.{fmt}")
    plt.close(fig)


def order_by_region(cfg: dict, tickers: list[str]) -> list[str]:
    meta = metadata_frame(cfg)
    regions = cfg["regions_order"]
    return sorted(tickers, key=lambda t: (regions.index(meta.loc[t, "region"]), tickers.index(t)))


def shade_events(ax, cfg: dict, label: bool) -> None:
    for i, ev in enumerate(cfg["events"]):
        a, b = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
        ax.axvspan(a, b, color="#d9d9d9", alpha=0.6, lw=0, zorder=0)
        if label:
            ax.text(a + (b - a) / 2, 1.02 + 0.09 * (i % 2), ev["label"], transform=ax.get_xaxis_transform(), fontsize=6.5,
                    color=MUTED, va="bottom", ha="center", clip_on=False)


def fig_total_connectedness(cfg: dict, scenario: str = "base_H4", name: str = "total_connectedness") -> None:
    tables = path(cfg, "tables")
    s = pd.read_csv(tables / f"series_{scenario}.csv", parse_dates=["date"])
    methods = cfg["compare"]["methods"]
    fig, axes = plt.subplots(len(methods), 1, figsize=(7, 1.9 * len(methods)), sharex=True)
    dy1 = pd.read_csv(tables / "series_dy_step1_H4.csv", parse_dates=["date"]) if scenario == "base_H4" else None
    for ax, m in zip(axes, methods):
        d = s[s["method"] == m].set_index("date")["total"]
        color, ls = METHOD_STYLE[m]
        if m == "dy" and dy1 is not None:
            ax.plot(dy1["date"], dy1["total"], color="#9ec5f4", lw=0.9, label="step 1")
            ax.plot(d.index, d.values, color=color, ls=ls, label="step 4")
            ax.legend(loc="upper right", frameon=False, ncol=2)
        else:
            ax.plot(d.index, d.values, color=color, ls=ls)
        shade_events(ax, cfg, label=False)
        ax.text(0.01, 0.04, METHOD_NAMES[m], transform=ax.transAxes, color=INK, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        ax.set_ylabel("Total (%)")
    shade_events(axes[0], cfg, label=True)
    axes[-1].set_xlabel("End of 104-week window")
    fig.tight_layout()
    save(fig, cfg, name)


def fig_overlay_total(cfg: dict) -> None:
    """All methods on one axis plus the FNETS common-variance share in a second panel."""
    s = pd.read_csv(path(cfg, "tables") / "series_base_H4.csv", parse_dates=["date"])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for m in cfg["compare"]["methods"]:
        d = s[s["method"] == m]
        c, ls = METHOD_STYLE[m]
        ax.plot(d["date"], d["total"], color=c, ls=ls, label=METHOD_NAMES[m])
    shade_events(ax, cfg, label=True)
    ax.set_ylabel("Total connectedness (%)")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    f = s[s["method"] == "fnets"]
    ax2.plot(f["date"], 100 * f["common_share"], color=INK, lw=1.2)
    shade_events(ax2, cfg, label=False)
    ax2.set_ylabel("Common share (%)")
    ax2.set_title("FNETS: share of variance explained by the common component", loc="left", color=INK)
    fig.tight_layout()
    save(fig, cfg, "total_connectedness_overlay")


def fig_heatmap(cfg: dict, method: str = "dy") -> None:
    s = pd.read_csv(path(cfg, "tables") / f"static/full_{method}_gfevd.csv", index_col=0)
    meta = metadata_frame(cfg)
    order = order_by_region(cfg, list(s.index))
    m = s.loc[order, order].values.copy()
    np.fill_diagonal(m, np.nan)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(m, cmap=SEQ, vmin=0, vmax=np.nanmax(m))
    ax.set_xticks(range(len(order)), order, rotation=90)
    ax.set_yticks(range(len(order)), order)
    ax.grid(False)
    bounds = np.cumsum([sum(meta.loc[order, "region"] == r) for r in cfg["regions_order"]])[:-1]
    for b in bounds:
        ax.axhline(b - 0.5, color=INK, lw=0.8)
        ax.axvline(b - 0.5, color=INK, lw=0.8)
    ax.set_xlabel("Shock in (from)")
    ax.set_ylabel("Variance of (to)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Share of forecast-error variance (%)")
    ax.set_title(f"{METHOD_NAMES[method]}: full-sample pairwise connectedness (H=4, diagonal omitted)",
                 loc="left", fontsize=9, color=INK)
    save(fig, cfg, f"heatmap_{method}")


def fig_network(cfg: dict) -> None:
    """Full-sample networks: DY (strongest GFEVD links), NETS and FNETS long-run partial correlations."""
    tables = path(cfg, "tables") / "static"
    meta = metadata_frame(cfg)
    specs = [("dy", "gfevd", "DY: strongest 10% of GFEVD links"),
             ("nets", "lrpc", "NETS: long-run partial correlations"),
             ("fnets", "lrpc", "FNETS: idiosyncratic long-run partial corr.")]
    mats = {k: pd.read_csv(tables / f"full_{k[0]}_{k[1]}.csv", index_col=0) for k in [(m, n) for m, n, _ in specs]}
    labels = list(next(iter(mats.values())).index)
    G0 = nx.complete_graph(labels)
    pos = nx.circular_layout(order_by_region(cfg, labels))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    share = cfg["networks"]["top_link_share"]
    for ax, (m, net, title) in zip(axes, specs):
        W = mats[(m, net)].values.copy()
        np.fill_diagonal(W, 0)
        G = nx.DiGraph() if net == "gfevd" else nx.Graph()
        G.add_nodes_from(labels)
        if net == "gfevd":
            vals = W[W > 0]
            thr = np.quantile(vals, 1 - share)
            for i, a in enumerate(labels):
                for j, b in enumerate(labels):
                    if W[i, j] >= thr:
                        G.add_edge(b, a, weight=W[i, j])
        else:
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if abs(W[i, j]) > cfg["networks"]["link_threshold"]:
                        G.add_edge(labels[i], labels[j], weight=abs(W[i, j]))
        w = np.array([d["weight"] for *_, d in G.edges(data=True)]) if G.number_of_edges() else np.array([1.0])
        widths = 0.5 + 3.0 * w / w.max()
        arrow_kw = {"arrows": True, "arrowsize": 7} if net == "gfevd" else {"arrows": False}
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=MUTED, alpha=0.7,
                               node_size=420, **arrow_kw)
        colors = [REGION_COLOR[meta.loc[n, "region"]] for n in labels]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=420, edgecolors="white", linewidths=1.5)
        focus = cfg["focus_ticker"]
        nx.draw_networkx_nodes(G, pos, nodelist=[focus], ax=ax, node_color=REGION_COLOR[meta.loc[focus, "region"]],
                               node_size=420, edgecolors=INK, linewidths=2.2)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="white", font_weight="bold")
        ax.set_title(f"{title}\n({G.number_of_edges()} links)", fontsize=9, color=INK)
        ax.axis("off")
    handles = [Patch(color=c, label=r) for r, c in REGION_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, cfg, "network_full_sample")


def fig_net_bars(cfg: dict) -> None:
    """Average net connectedness by country (rolling windows), one panel per method."""
    s = pd.read_csv(path(cfg, "tables") / "series_base_H4.csv")
    meta = metadata_frame(cfg)
    methods = cfg["compare"]["methods"]
    labels = order_by_region(cfg, [c[4:] for c in s.columns if c.startswith("net_")])
    fig, axes = plt.subplots(len(methods), 1, figsize=(7, 1.8 * len(methods)), sharex=True)
    focus = cfg["focus_ticker"]
    for ax, m in zip(axes, methods):
        vals = s[s["method"] == m][[f"net_{t}" for t in labels]].mean().values
        for i, (t, v) in enumerate(zip(labels, vals)):
            reg = meta.loc[t, "region"]
            ax.bar(i, v, color=REGION_COLOR[reg], hatch=REGION_HATCH[reg], width=0.75,
                   edgecolor=INK if t == focus else "white", linewidth=1.6 if t == focus else 0.5)
        ax.axhline(0, color=INK, lw=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_title(METHOD_NAMES[m], loc="left", color=INK)
        ax.set_ylabel("Net (pp)")
    axes[-1].set_xticks(range(len(labels)), labels, rotation=90)
    axes[-1].get_xticklabels()[labels.index(focus)].set_fontweight("bold")
    handles = [Patch(facecolor=c, hatch=REGION_HATCH[r], edgecolor="white", label=r) for r, c in REGION_COLOR.items()]
    handles.append(Patch(facecolor="white", edgecolor=INK, linewidth=1.6, label=f"{focus} (Argentina)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, cfg, "net_connectedness_by_country")


def fig_ex2020(cfg: dict) -> None:
    """Total connectedness excluding windows that touch 2020: the gap is left visible."""
    s = pd.read_csv(path(cfg, "tables") / "series_ex2020_H4.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for m in cfg["compare"]["methods"]:
        d = s[s["method"] == m]
        c, ls = METHOD_STYLE[m]
        ax.plot(d["date"], d["total"], color=c, ls=ls, label=METHOD_NAMES[m])  # NaN rows break the line
    ax.set_ylabel("Total connectedness (%)")
    ax.set_title(f"Robustness: windows touching {cfg['robustness']['exclude_year']} removed (gap not interpolated)",
                 loc="left", color=INK)
    ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, cfg, "total_connectedness_ex2020")


def descriptive_table(cfg: dict) -> None:
    d = pd.read_csv(path(cfg, "tables") / "descriptive_stats.csv", index_col=0)
    rows = [[t, r["country"], r["region"], r["start"], f"{100 * r['mean']:.2f}", f"{100 * r['std']:.2f}",
             f"{100 * r['min']:.1f}", f"{100 * r['max']:.1f}", f"{r['skewness']:.2f}", f"{r['kurtosis']:.1f}"]
            for t, r in d.iterrows()]
    write_latex(path(cfg, "tables") / "descriptive_stats.tex",
                ["Ticker", "Country", "Region", "Start", "Mean (\\%)", "Std (\\%)", "Min (\\%)", "Max (\\%)",
                 "Skew.", "Kurt."], rows, "lllrrrrrrr")


def main() -> None:
    cfg = load_config()
    setup(cfg)
    fig_total_connectedness(cfg)
    fig_overlay_total(cfg)
    fig_heatmap(cfg, "dy")
    fig_heatmap(cfg, "fnets")
    fig_network(cfg)
    fig_net_bars(cfg)
    fig_ex2020(cfg)
    descriptive_table(cfg)
    print("Figures written to", path(cfg, "figures"))


if __name__ == "__main__":
    main()
