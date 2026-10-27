import numpy as np
import pytest

from src.forecast import ar_forecast, diebold_mariano, var_forecast


def test_dm_detects_better_model():
    rng = np.random.default_rng(8)
    e_good = rng.standard_normal(400) * 0.5
    e_bad = rng.standard_normal(400)
    stat, p = diebold_mariano(e_good, e_bad)
    assert stat < 0 and p < 0.01


def test_dm_equal_models_not_rejected():
    rng = np.random.default_rng(9)
    stat, p = diebold_mariano(rng.standard_normal(500), rng.standard_normal(500))
    assert p > 0.01


def test_ar_forecast_recovers_ar1():
    rng = np.random.default_rng(10)
    T, phi = 3000, 0.6
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    f = ar_forecast(x[:, None], 4)
    assert f[0] == pytest.approx(phi * x[-1], abs=0.1)


def test_var_forecast():
    coefs = np.array([[[0.5, 0.0], [0.1, 0.2]]])
    y = np.array([[0.0, 0.0], [1.0, 2.0]])
    np.testing.assert_allclose(var_forecast(np.array([1.0, 0.0]), coefs, y), [1.5, 0.5])
