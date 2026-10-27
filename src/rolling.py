"""Rolling-window bookkeeping shared by all methods."""
from __future__ import annotations

import numpy as np
import pandas as pd


def window_ends(n_obs: int, window: int, step: int) -> np.ndarray:
    """Row indices (inclusive) of the last observation of each rolling window."""
    if window > n_obs:
        raise ValueError(f"Window {window} longer than sample {n_obs}.")
    return np.arange(window - 1, n_obs, step)


def windows_touching_year(index: pd.DatetimeIndex, ends: np.ndarray, window: int, year: int) -> np.ndarray:
    """Boolean mask: True if the window ending at `ends[k]` contains any week of `year`."""
    years = index.year.values
    return np.array([(years[e - window + 1:e + 1] == year).any() for e in ends])
