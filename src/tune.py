"""Light hyperparameter tuning via Optuna.

Design goals:
- Small trial budget (default 25/model): the framing is *light* tuning, not
  a leaderboard grind.
- Temporal validation to match the eval protocol: fit on fold-0's fit slice,
  early-stop on fold-0's ES slice, evaluate PR-AUC on fold-0's val slice.
- Deterministic: fixed seed on the Optuna sampler.
- Output written to results/tuned_params.json; experiment.py auto-loads it.

Run:
    python -m src.tune                    # tune all three
    python -m src.tune --models xgboost   # single model
"""
from __future__ import annotations

# Same bootstrap thread env as experiment.py — set OMP/MKL caps before
# numpy/xgboost/lightgbm initialize their pools.
import os as _os
from pathlib import Path as _Path

def _bootstrap_thread_env() -> None:
    try:
        import yaml as _yaml
        _cfg = _yaml.safe_load(
            (_Path(__file__).resolve().parents[1] / "configs" / "defaults.yaml").read_text()
        )
        _n = (_cfg.get("runtime") or {}).get("n_threads")
        if _n:
            for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                _os.environ.setdefault(_var, str(_n))
    except Exception:
        pass

_bootstrap_thread_env()

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import optuna
import yaml
from sklearn.metrics import average_precision_score

from src.cv import expanding_time_folds, split_train_for_early_stopping
from src.data import existing_categorical, load_train, to_pandas_features
from src.models import CatWrap, LGBMWrap, XGBWrap

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs" / "defaults.yaml"
RESULTS_DIR = REPO / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _lgbm_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
    }


def _xgb_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100, log=True),
    }


def _cat_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


MODEL_REGISTRY: dict[str, tuple[Callable[[optuna.Trial], dict], type]] = {
    "lightgbm": (_lgbm_space, LGBMWrap),
    "xgboost": (_xgb_space, XGBWrap),
    "catboost": (_cat_space, CatWrap),
}


def _make_objective(
    model_name: str,
    cat_cols: list[str],
    X_all,
    y_all,
    time_all,
    cv_cfg,
    n_threads: Optional[int] = None,
):
    """Optuna objective closure. Reuses the first CV fold as the tuning fold
    so the tuning protocol matches the eval protocol exactly."""
    fold = next(
        iter(
            expanding_time_folds(
                time_all,
                n_folds=cv_cfg["n_folds"],
                holdout_frac=cv_cfg["holdout_frac"],
                min_train_frac=cv_cfg["min_train_frac"],
            )
        )
    )
    fit_idx, es_idx = split_train_for_early_stopping(fold.train_idx, es_frac=cv_cfg.get("es_frac", 0.1))
    X_fit = X_all.iloc[fit_idx]
    y_fit = y_all[fit_idx]
    X_es = X_all.iloc[es_idx]
    y_es = y_all[es_idx]
    X_va = X_all.iloc[fold.val_idx]
    y_va = y_all[fold.val_idx]
    print(f"  tuning fold: fit={len(fit_idx):,}  es={len(es_idx):,}  val={len(fold.val_idx):,}")

    space_fn, cls = MODEL_REGISTRY[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        model = cls(cat_cols, params, n_threads=n_threads)
        model.fit(X_fit, y_fit, X_es, y_es)
        scores = np.asarray(model.predict_proba(X_va))
        return float(average_precision_score(y_va, scores))

    return objective


def run(models: list[str], n_trials: int, seed: int) -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    n_threads = (cfg.get("runtime") or {}).get("n_threads")
    if n_threads:
        print(f"Thread budget: pinning models to {n_threads} threads")
    print("Loading IEEE-CIS training data ...")
    df = load_train()
    X_all, y_all, time_all = to_pandas_features(df)
    cat_cols = existing_categorical(X_all)

    best: dict[str, dict] = {}
    for name in models:
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}.")
        print(f"\n=== Tuning {name} ({n_trials} trials) ===")
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"tune-{name}")
        objective = _make_objective(name, cat_cols, X_all, y_all, time_all, cfg["cv"], n_threads=n_threads)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        print(f"  best PR-AUC on tuning fold: {study.best_value:.4f}")
        print(f"  best params: {study.best_params}")
        best[name] = study.best_params

    tuned_path = RESULTS_DIR / "tuned_params.json"
    tuned_path.write_text(json.dumps(best, indent=2))
    print(f"\nWrote {tuned_path}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override configs/defaults.yaml tuning.n_trials")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    tune_cfg = cfg.get("tuning", {})
    n_trials = args.n_trials if args.n_trials is not None else tune_cfg.get("n_trials", 25)
    seed = args.seed if args.seed is not None else tune_cfg.get("seed", 0)
    run(args.models, n_trials=n_trials, seed=seed)


if __name__ == "__main__":
    main()
