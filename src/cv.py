from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class TemporalFold:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def split_train_for_early_stopping(
    train_idx: np.ndarray, es_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve the tail of a temporally ordered training block off as an
    early-stopping slice. Returns (fit_idx, es_idx) with es_idx being the
    last es_frac fraction of train_idx.

    Using this ensures that eval on the fold's val_idx is not tainted by
    picking best_iteration on the same rows.
    """
    if not 0.0 < es_frac < 1.0:
        raise ValueError(f"es_frac must be in (0, 1); got {es_frac}")
    n = len(train_idx)
    n_es = max(1, int(n * es_frac))
    return train_idx[:-n_es], train_idx[-n_es:]


def holdout_split(time_col: np.ndarray, holdout_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cv_indices, holdout_indices) sorted by time_col ascending."""
    order = np.argsort(time_col, kind="mergesort")
    cv_end = int(len(order) * (1.0 - holdout_frac))
    return order[:cv_end], order[cv_end:]


def expanding_time_folds(
    time_col: np.ndarray,
    n_folds: int = 5,
    holdout_frac: float = 0.2,
    min_train_frac: float = 0.2,
) -> Iterator[TemporalFold]:
    """Expanding-window temporal CV over the pre-holdout portion.

    Split the pre-holdout portion into n_folds+1 equal time-ordered blocks:
    fold k trains on blocks [0..k] and validates on block k+1.
    """
    cv_idx, _ = holdout_split(time_col, holdout_frac)
    n_cv = len(cv_idx)
    total_blocks = n_folds + 1
    block_size = n_cv // total_blocks

    min_train_size = int(n_cv * min_train_frac)
    for k in range(n_folds):
        train_end = max(block_size * (k + 1), min_train_size)
        val_end = block_size * (k + 2) if k < n_folds - 1 else n_cv
        yield TemporalFold(
            fold=k,
            train_idx=cv_idx[:train_end],
            val_idx=cv_idx[train_end:val_end],
        )
