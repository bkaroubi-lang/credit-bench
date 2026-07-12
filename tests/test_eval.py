"""Sanity checks for eval primitives.

Run with:
    python -m pytest tests/ -q
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.eval import (
    _midrank,
    delong_auc_ci,
    delong_paired_auc_diff_ci,
)


def test_midrank_ties():
    x = np.array([1.0, 2.0, 2.0, 3.0])
    # ranks: 1 -> 1, 2 (tie) -> 2.5, 3 -> 4
    np.testing.assert_allclose(_midrank(x), [1.0, 2.5, 2.5, 4.0])


def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(42)
    n = 5000
    y = rng.integers(0, 2, size=n)
    # Signal + noise so AUC is neither 0.5 nor 1.0.
    scores = y * 0.5 + rng.normal(size=n) * 0.3
    delong = delong_auc_ci(y, scores)
    sklearn_auc = roc_auc_score(y, scores)
    assert abs(delong["auc"] - sklearn_auc) < 1e-9
    assert delong["ci_lower"] < delong["auc"] < delong["ci_upper"]


def test_delong_paired_identical_scores():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, size=n)
    s = y * 0.4 + rng.normal(size=n) * 0.5
    r = delong_paired_auc_diff_ci(y, s, s)
    assert abs(r["delta"]) < 1e-12
    # SE is zero for identical scores; p defined as 1.0 by our formula.
    assert r["se"] < 1e-12
    assert r["ci_lower"] == r["ci_upper"] == 0.0


def test_delong_paired_detects_difference():
    rng = np.random.default_rng(7)
    n = 5000
    y = rng.integers(0, 2, size=n)
    # s1 is a strictly better ranker than s2.
    s1 = y * 0.5 + rng.normal(size=n) * 0.3
    s2 = y * 0.2 + rng.normal(size=n) * 0.5
    r = delong_paired_auc_diff_ci(y, s1, s2)
    assert r["delta"] > 0
    assert r["ci_lower"] > 0  # CI excludes zero
    assert r["p"] < 1e-3
