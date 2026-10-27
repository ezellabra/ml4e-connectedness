import numpy as np
import pytest

from src.dy import connectedness, dy_window, fit_var_ols, gfevd, select_lag_bic


def simulate_var1(A, sigma, T, rng, burn=200):
    N = A.shape[0]
    L = np.linalg.cholesky(sigma)
    y = np.zeros((T + burn, N))
    for t in range(1, T + burn):
        y[t] = A @ y[t - 1] + L @ rng.standard_normal(N)
    return y[burn:]


@pytest.fixture(scope="module")
def known_var():
    A = np.array([[0.4, 0.0, 0.3, 0.0],
                  [0.0, 0.3, 0.0, 0.0],
                  [0.0, 0.0, 0.2, 0.0],
                  [0.2, 0.0, 0.0, 0.1]])
    sigma = np.array([[1.0, 0.3, 0.0, 0.0],
                      [0.3, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
    return A, sigma


def test_rows_sum_to_one(known_var):
    A, sigma = known_var
    theta = gfevd(A[None], sigma, horizon=4)
    np.testing.assert_allclose(theta.sum(axis=1), 1.0, atol=1e-12)
    assert (theta >= 0).all()


def test_horizon_one_is_squared_correlation(known_var):
    _, sigma = known_var
    theta = gfevd(np.zeros((1, 4, 4)), sigma, horizon=1, normalize=False)
    d = np.sqrt(np.diag(sigma))
    np.testing.assert_allclose(theta, (sigma / np.outer(d, d)) ** 2, atol=1e-12)


def test_permutation_invariance():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((300, 5)) @ rng.standard_normal((5, 5))
    perm = np.array([3, 0, 4, 1, 2])
    theta, p, _ = dy_window(y, horizon=4, max_lag=2)
    theta_p, p_p, _ = dy_window(y[:, perm], horizon=4, max_lag=2)
    assert p == p_p
    np.testing.assert_allclose(theta_p, theta[np.ix_(perm, perm)], atol=1e-10)
    c, cp = connectedness(theta), connectedness(theta_p)
    assert c["total"] == pytest.approx(cp["total"], abs=1e-9)
    np.testing.assert_allclose(cp["net"], c["net"][perm], atol=1e-9)


def test_recovers_known_connectedness(known_var):
    A, sigma = known_var
    rng = np.random.default_rng(1)
    y = simulate_var1(A, sigma, 20000, rng)
    assert select_lag_bic(y, 2) == 1
    fit = fit_var_ols(y, 1)
    np.testing.assert_allclose(fit.coefs[0], A, atol=0.03)
    true = gfevd(A[None], sigma, horizon=4)
    est = gfevd(fit.coefs, fit.sigma, horizon=4)
    np.testing.assert_allclose(est, true, atol=0.02)
    c_true, c_est = connectedness(true), connectedness(est)
    assert c_est["total"] == pytest.approx(c_true["total"], abs=1.0)
    # variable 3 (index 2) drives variable 1 (index 0) and receives nothing: net transmitter
    assert c_true["net"][2] > 0 and c_est["net"][2] > 0


def test_independent_system_has_no_connectedness():
    theta = gfevd(np.diag([0.5, 0.2, -0.3])[None], np.diag([1.0, 2.0, 0.5]), horizon=8)
    np.testing.assert_allclose(theta, np.eye(3), atol=1e-12)
    assert connectedness(theta)["total"] == pytest.approx(0.0)


def test_measures_accounting():
    rng = np.random.default_rng(2)
    theta = rng.random((6, 6))
    theta /= theta.sum(axis=1, keepdims=True)
    c = connectedness(theta)
    assert c["net"].sum() == pytest.approx(0.0, abs=1e-9)
    assert c["to"].sum() == pytest.approx(c["from"].sum())
    assert c["total"] == pytest.approx(c["from"].mean())
    np.testing.assert_allclose(c["pairwise_net"], -c["pairwise_net"].T)
