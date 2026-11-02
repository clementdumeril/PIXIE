"""Ranking metrics for defect scores (higher score = more likely defective)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    npos, nneg = y.sum(), (~y).sum()
    if not npos or not nneg:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if not y.any():
        return float("nan")
    y = y[np.argsort(-s, kind="stable")]
    hits = np.cumsum(y)
    return float((hits[y] / (np.flatnonzero(y) + 1)).mean())
