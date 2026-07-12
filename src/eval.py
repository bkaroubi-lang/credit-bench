"""Metric primitives + Sun-Xu paired DeLong + block bootstrap + calibration.

Everything here is model-agnostic: input is (y_true, y_score) with y_score being
the positive-class probability.
"""
from typing import Callable, Optional

import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ---------- basic point metrics ----------

def core_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def calibration_summary(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 20) -> dict:
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    return {
        "brier": float(brier_score_loss(y_true, y_score)),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


# ---------- DeLong (Sun & Xu 2014, O(N log N) via midrank) ----------

def _midrank(x: np.ndarray) -> np.ndarray:
    """Fractional midrank of x; ties get the mean of their positions (1-indexed)."""
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def _delong_pseudo(y_true: np.ndarray, y_score: np.ndarray):
    """Returns (AUC, V01, V10) where V01 has length m (pos) and V10 length n (neg)."""
    pos_mask = y_true == 1
    neg_mask = ~pos_mask
    m = int(pos_mask.sum())
    n = int(neg_mask.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both classes to be present.")
    pos = y_score[pos_mask]
    neg = y_score[neg_mask]
    tx = _midrank(pos)
    ty = _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    return auc, v01, v10


def delong_auc_ci(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.05) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    auc, v01, v10 = _delong_pseudo(y_true, y_score)
    var = v01.var(ddof=1) / len(v01) + v10.var(ddof=1) / len(v10)
    se = float(np.sqrt(var))
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "auc": float(auc),
        "se": se,
        "ci_lower": float(auc - z * se),
        "ci_upper": float(auc + z * se),
    }


def delong_paired_auc_diff_ci(
    y_true: np.ndarray, s1: np.ndarray, s2: np.ndarray, alpha: float = 0.05
) -> dict:
    """CI + z-test for AUC(s1) - AUC(s2) on the same y_true."""
    y_true = np.asarray(y_true).astype(int)
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    auc1, v01_1, v10_1 = _delong_pseudo(y_true, s1)
    auc2, v01_2, v10_2 = _delong_pseudo(y_true, s2)
    delta = auc1 - auc2
    m = len(v01_1)
    n = len(v10_1)
    var = (v01_1 - v01_2).var(ddof=1) / m + (v10_1 - v10_2).var(ddof=1) / n
    se = float(np.sqrt(var))
    z_stat = float(delta / se) if se > 0 else 0.0
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "auc1": float(auc1),
        "auc2": float(auc2),
        "delta": float(delta),
        "se": se,
        "ci_lower": float(delta - z_crit * se),
        "ci_upper": float(delta + z_crit * se),
        "z": z_stat,
        "p": p,
    }


# ---------- block bootstrap (assumes inputs ordered in time) ----------

def _block_bootstrap_index(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n]


def block_bootstrap_metric_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    block_size: int = 1000,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_bootstrap_index(n, block_size, rng)
        boots[b] = metric_fn(y_true[idx], y_score[idx])
    return {
        "point": float(metric_fn(y_true, y_score)),
        "ci_lower": float(np.quantile(boots, alpha / 2)),
        "ci_upper": float(np.quantile(boots, 1 - alpha / 2)),
    }


def block_bootstrap_paired_delta_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    block_size: int = 1000,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_bootstrap_index(n, block_size, rng)
        diffs[b] = metric_fn(y_true[idx], s1[idx]) - metric_fn(y_true[idx], s2[idx])
    return {
        "delta": float(metric_fn(y_true, s1) - metric_fn(y_true, s2)),
        "ci_lower": float(np.quantile(diffs, alpha / 2)),
        "ci_upper": float(np.quantile(diffs, 1 - alpha / 2)),
    }
