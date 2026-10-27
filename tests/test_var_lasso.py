import numpy as np
import pytest

from src.config import load_config
from src.dy import connectedness, gfevd
from src.var_lasso import fit_var_lasso, lasso_window
from tests.test_dy import simulate_var1


@pytest.fixture(scope="module")
def opts():
    return dict(load_config()["var_lasso"], max_lag=1)


@pytest.fixture(scope="module")
def sparse_system():
    N = 6
    A = np.diag(np.full(N, 0.3))
    A[0, 3] = 0.4
    A[2, 5] = -0.35
    sigma = np.eye(N) + 0.2 * (np.ones((N, N)) - np.eye(N))
    y = simulate_var1(A, sigma, 3000, np.random.default_rng(3))
    return A, sigma, y


@pytest.mark.parametrize("selection", ["cv", "ebic"])
def test_support_recovery(sparse_system, opts, selection):
    A, sigma, y = sparse_system
    fit, alphas = fit_var_lasso(y, 1, dict(opts, selection=selection))
    est = fit.coefs[0]
    true_support = A != 0
    assert (np.abs(est[true_support]) > 0.1).all()
    # false positives must be small in magnitude
    assert np.abs(est[~true_support]).max() < 0.08
    assert (alphas > 0).all()


def test_connectedness_close_to_truth(sparse_system, opts):
    A, sigma, y = sparse_system
    theta, fit, _ = lasso_window(y, 4, dict(opts, selection="cv"))
    true = gfevd(A[None], sigma, 4)
    assert connectedness(theta)["total"] == pytest.approx(connectedness(true)["total"], abs=2.0)
    np.testing.assert_allclose(theta.sum(axis=1), 1.0)


def test_white_noise_is_shrunk_to_zero(opts):
    y = np.random.default_rng(4).standard_normal((400, 5))
    fit, _ = fit_var_lasso(y, 2, dict(opts, selection="ebic"))
    assert (np.abs(fit.coefs) > 0).mean() < 0.1


def test_unknown_selection_raises(sparse_system, opts):
    with pytest.raises(ValueError):
        fit_var_lasso(sparse_system[2][:200], 1, dict(opts, selection="kfold"))
