import numpy as np
import pandas as pd
import pytest

from src import r_bridge
from src.config import load_config
from src.estimate import _from_r
from src.networks import clip_pcor, nearest_pd
from tests.test_dy import simulate_var1

pytestmark = pytest.mark.skipif(not r_bridge.rscript_available(), reason="Rscript not installed")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def frame(y):
    idx = pd.date_range("2000-01-07", periods=len(y), freq="7D")
    return pd.DataFrame(y, index=idx, columns=[f"S{i}" for i in range(y.shape[1])])


def test_nets_recovers_granger_link(cfg):
    N = 5
    A = np.diag(np.full(N, 0.2))
    A[0, 3] = 0.6
    y = simulate_var1(A, np.eye(N), 600, np.random.default_rng(5))
    out = r_bridge.run_windows("nets", [frame(y)], cfg["nets"], cfg["seed"])[0]
    assert out["A"].shape == (N, N)
    assert out["sigma"].shape == (N, N)
    assert abs(out["A"][0, 3]) > 0.3
    assert np.allclose(np.diag(out["pcor"]), 1.0)


def test_fnets_detects_factor(cfg):
    rng = np.random.default_rng(6)
    T, N = 300, 12
    f = rng.standard_normal(T)
    y = np.outer(f, rng.uniform(0.8, 1.2, N)) + 0.5 * rng.standard_normal((T, N))
    out = r_bridge.run_windows("fnets", [frame(y)], cfg["fnets"], cfg["seed"])[0]
    assert out["info"]["q"] >= 1
    share = out["var_common"] / out["var_x"]
    assert share.mean() > 0.5
    fit = _from_r("fnets", out, cfg)
    assert np.linalg.eigvalsh(fit["sigma"]).min() > 0
    assert np.abs(fit["pcor"]).max() <= 1.0


def test_failed_window_returns_none(cfg):
    y = np.random.default_rng(7).standard_normal((60, 4))
    bad = frame(y)
    bad.iloc[:, 1] = 0.0   # constant series: standardization fails inside R
    with pytest.warns(UserWarning):
        out = r_bridge.run_windows("fnets", [bad], cfg["fnets"], cfg["seed"])
    assert out == [None]


def test_missing_rscript_raises(monkeypatch, cfg):
    monkeypatch.setattr(r_bridge.shutil, "which", lambda name: None)
    with pytest.raises(r_bridge.RBridgeError):
        r_bridge.run_windows("nets", [frame(np.zeros((10, 2)))], cfg["nets"], cfg["seed"])


def test_pd_projection_and_clipping():
    S = np.array([[1.0, 1.2], [1.2, 1.0]])
    P, corrected = nearest_pd(S, 0.01)
    assert corrected and np.linalg.eigvalsh(P).min() > 0
    pc, n_bad = clip_pcor(np.array([[1.0, 3.0], [3.0, 1.0]]))
    assert n_bad == 2 and pc[0, 1] == 1.0
