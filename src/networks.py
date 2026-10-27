"""Network measures that make the outputs of different estimators comparable."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def offdiag(m: np.ndarray) -> np.ndarray:
    return m[~np.eye(m.shape[0], dtype=bool)]


def nearest_pd(S: np.ndarray, floor_ratio: float) -> tuple[np.ndarray, bool]:
    """Symmetrize and clip eigenvalues below floor_ratio * mean eigenvalue.
    Returns (matrix, whether a correction was needed)."""
    S = (S + S.T) / 2
    w, V = np.linalg.eigh(S)
    floor = floor_ratio * np.abs(w).mean()
    if w.min() >= floor:
        return S, False
    return (V * np.maximum(w, floor)) @ V.T, True


def clip_pcor(pc: np.ndarray) -> tuple[np.ndarray, int]:
    """Clip partial correlations to [-1, 1]; returns the clipped matrix and the number of
    off-diagonal entries that were outside the range."""
    n_bad = int((np.abs(offdiag(pc)) > 1).sum())
    out = np.clip(pc, -1, 1)
    np.fill_diagonal(out, 1.0)
    return out, n_bad


def directed_measures(W: np.ndarray, threshold: float) -> dict:
    """Measures for a directed weighted network with W[i, j] = link from j to i
    (e.g. VAR coefficient of y_j(t-1) in the equation of y_i, or GFEVD share)."""
    N = W.shape[0]
    A = np.abs(W).copy()
    np.fill_diagonal(A, 0.0)
    adj = A > threshold
    out_deg = adj.sum(axis=0)
    in_deg = adj.sum(axis=1)
    return {
        "density": adj.sum() / (N * (N - 1)),
        "strength_total": A.sum() / N,
        "out_degree": out_deg,
        "in_degree": in_deg,
        "net_degree": out_deg - in_deg,
        "out_strength": A.sum(axis=0),
        "in_strength": A.sum(axis=1),
    }


def undirected_measures(P: np.ndarray, threshold: float) -> dict:
    """Measures for an undirected weighted network (partial correlations)."""
    N = P.shape[0]
    A = np.abs(P).copy()
    np.fill_diagonal(A, 0.0)
    adj = A > threshold
    return {
        "density": adj.sum() / (N * (N - 1)),
        "strength_total": A.sum() / N,
        "degree": adj.sum(axis=1),
        "strength": A.sum(axis=1),
    }


def spearman_offdiag(a: np.ndarray, b: np.ndarray) -> float:
    x, y = offdiag(a), offdiag(b)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def top_links(m: np.ndarray, share: float) -> set:
    """Indices (off-diagonal) of the strongest |m| links, top `share` of all links."""
    N = m.shape[0]
    idx = np.flatnonzero(~np.eye(N, dtype=bool))
    vals = np.abs(m.ravel()[idx])
    k = max(1, int(round(share * len(idx))))
    keep = idx[np.argsort(-vals, kind="stable")[:k]]
    return set(keep[vals[np.argsort(-vals, kind="stable")[:k]] > 0].tolist())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)
