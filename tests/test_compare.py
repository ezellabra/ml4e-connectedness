import numpy as np
import pandas as pd
import pytest

from src import compare
from src.config import load_config
from src.dy import gfevd
from src.estimate import var_networks
from src.networks import jaccard, top_links


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def fake_result(method, n_win=6, N=4, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    fits, thetas = [], []
    for _ in range(n_win):
        A = np.diag(np.full(N, 0.2)) + scale * 0.1 * rng.standard_normal((N, N))
        S = np.eye(N) + 0.3
        f = {"coefs": A[None], "sigma": S, **var_networks(A[None], S)}
        if method == "fnets":
            f["common_share"] = np.full(N, 0.6)
            f["cov_common"] = 0.6 * np.ones((N, N))
            f["var_idio"] = np.full(N, 0.4)
            f["q"] = 1
        fits.append(f)
        thetas.append(gfevd(A[None], S, 4))
    dates = pd.date_range("2020-01-03", periods=n_win, freq="28D")
    return {"method": method, "dates": dates, "labels": list("ABCD"), "fits": fits, "thetas": {4: np.stack(thetas)}}


def test_identical_methods_agree_perfectly(cfg):
    res = {"dy": fake_result("dy"), "lasso": fake_result("lasso")}
    meas = {m: compare.window_measures(r, 4, cfg) for m, r in res.items()}
    ag = compare.agreement_table(res, meas, 4, cfg)
    assert ag.loc[0, "corr_total"] == pytest.approx(1.0)
    assert ag.loc[0, "spearman_gfevd"] == pytest.approx(1.0)
    assert ag.loc[0, "jaccard_top_lrpc"] == pytest.approx(1.0)


def test_mask_drops_windows(cfg):
    r = fake_result("dy")
    mask = np.array([False, True, True, False, False, False])
    m = compare.window_measures(r, 4, cfg, mask)
    assert m["total"].isna().sum() == 2


def test_common_factor_table(cfg):
    res = {"dy": fake_result("dy"), "fnets": fake_result("fnets", seed=1, scale=0.0)}
    meas = {m: compare.window_measures(r, 4, cfg) for m, r in res.items()}
    t = compare.common_factor_table(meas)["value"]
    assert t["common_variance_share_mean"] == pytest.approx(0.6)
    ratio = (meas["fnets"]["total"] / meas["dy"]["total"]).mean()
    assert t["share_disappearing_mean"] == pytest.approx(1 - ratio)


def test_stability_constant_network(cfg):
    r = fake_result("dy", scale=0.0)
    meas = {"dy": compare.window_measures(r, 4, cfg)}
    st = compare.stability_table({"dy": r}, meas, 4, cfg)
    assert st.loc["dy", "mean_abs_change_total"] == pytest.approx(0.0)


def test_top_links_and_jaccard():
    m = np.array([[0, 5, 1], [0.5, 0, 3], [2, 0.1, 0]], dtype=float)
    assert top_links(m, 2 / 6) == {1, 5}
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
