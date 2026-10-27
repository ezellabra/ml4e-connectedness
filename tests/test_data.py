import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import DataValidationError, load_returns, validate_returns


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def returns(cfg):
    return load_returns(cfg)


def test_shape_and_columns(returns, cfg):
    assert returns.shape == (816, 18)
    assert list(returns.columns) == [m["ticker"] for m in cfg["metadata"]]
    assert returns.index[0] == pd.Timestamp("2011-03-11")
    assert returns.index[-1] == pd.Timestamp("2026-10-23")


def test_validation_catches_gaps(returns, cfg):
    with pytest.raises(DataValidationError):
        validate_returns(returns.drop(returns.index[10]), cfg)


def test_validation_catches_nan(returns, cfg):
    bad = returns.copy()
    bad.iloc[3, 2] = np.nan
    with pytest.raises(DataValidationError):
        validate_returns(bad, cfg)


def test_validation_catches_unsorted(returns, cfg):
    with pytest.raises(DataValidationError):
        validate_returns(returns.iloc[::-1], cfg)
