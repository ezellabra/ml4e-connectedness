"""Configuration loading and shared helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def metadata_frame(cfg: dict) -> pd.DataFrame:
    return pd.DataFrame(cfg["metadata"]).set_index("ticker")


def set_seed(cfg: dict) -> np.random.Generator:
    np.random.seed(cfg["seed"])
    return np.random.default_rng(cfg["seed"])
