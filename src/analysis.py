"""SHAP + cross-model disagreement analysis.

Consumes the holdout scores + labels written by experiment.py, refits each
model on the same pre-holdout data (with tuned params if available), computes
SHAP values on a subsample of the holdout, and reports:

  - Per-model global feature importance (mean |SHAP|, top-K features)
  - Cross-model disagreement: rows where pairwise |p_a - p_b| is largest,
    with each model's top SHAP contributors on those rows
  - Per-feature SHAP disagreement between model pairs

Run:
    python -m src.analysis                    # all pairs, default subsample
    python -m src.analysis --subsample 5000   # smaller subsample
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from src.cv import holdout_split, split_train_for_early_stopping
from src.data import existing_categorical, load_train, to_pandas_features
from src.experiment import _build_models, _load_tuned_params

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs" / "defaults.yaml"
RESULTS_DIR = REPO / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------- SHAP computation per model ----------

def _shap_values(model, model_name: str, X: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    """Return per-row SHAP values, shape (n_rows, n_features).
    Delegates to each library's fastest path. Uses _apply_category so the
    prediction-time category codes match the training-set encoding
    (independently re-inferring them can misalign codes vs. what the model
    learned)."""
    if model_name == "lightgbm":
        from src.models.gbm import _apply_category
        Xp = _apply_category(X, model._Xt_ref, cat_cols)
        contrib = model.model.predict(
            Xp, pred_contrib=True, num_iteration=model.model.best_iteration
        )
        # (n_rows, n_features + 1); last column is the expected value.
        return np.asarray(contrib)[:, :-1]

    if model_name == "xgboost":
        from src.models.gbm import _apply_category
        Xp = _apply_category(X, model._Xt_ref, cat_cols)
        dm = model.xgb.DMatrix(Xp, enable_categorical=True)
        contrib = model.model.predict(dm, pred_contribs=True)
        # (n_rows, n_features + 1); last column is the base value.
        return np.asarray(contrib)[:, :-1]

    if model_name == "catboost":
        Xp = model._prep(X)
        cats = [c for c in cat_cols if c in Xp.columns]
        pool = model.cb.Pool(Xp, cat_features=cats)
        shap_vals = model.model.get_feature_importance(pool, type="ShapValues")
        # (n_rows, n_features + 1); last column is the expected value.
        return np.asarray(shap_vals)[:, :-1]

    # DL models: SHAP would need kernel/deep explainers and is much slower.
    # Fall back to permutation importance would be another route; for now
    # skip and report an empty importance so downstream code degrades cleanly.
    print(f"  SHAP for {model_name} not implemented; skipping.")
    return np.zeros((len(X), X.shape[1]))


# ---------- Global feature importance ----------

def global_importance(shap_vals: np.ndarray, feature_names: list[str], top_k: int = 30) -> pd.DataFrame:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df.head(top_k)


# ---------- Cross-model row-level disagreement ----------

def top_disagreement_rows(
    holdout_scores: Dict[str, np.ndarray], top_k: int = 50
) -> pd.DataFrame:
    """Rank holdout rows by the maximum pairwise |p_a - p_b|.
    Returns a DataFrame with per-model scores + max_gap."""
    names = list(holdout_scores.keys())
    n = len(next(iter(holdout_scores.values())))
    max_gap = np.zeros(n)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            max_gap = np.maximum(max_gap, np.abs(holdout_scores[a] - holdout_scores[b]))
    df = pd.DataFrame({name: holdout_scores[name] for name in names})
    df["max_pairwise_gap"] = max_gap
    df = df.sort_values("max_pairwise_gap", ascending=False).head(top_k).reset_index()
    df = df.rename(columns={"index": "holdout_row"})
    return df


def per_feature_shap_delta(
    shap_by_model: Dict[str, np.ndarray], feature_names: list[str], top_k: int = 30
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """For each pair of models, per-feature mean absolute difference in SHAP
    attribution. Highlights features where the models 'explain' predictions
    with different mechanisms."""
    names = list(shap_by_model.keys())
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            delta = np.abs(shap_by_model[a] - shap_by_model[b]).mean(axis=0)
            df = pd.DataFrame({"feature": feature_names, "mean_abs_shap_delta": delta})
            df = df.sort_values("mean_abs_shap_delta", ascending=False).head(top_k).reset_index(drop=True)
            df["rank"] = df.index + 1
            out[(a, b)] = df
    return out


# ---------- Driver ----------

def run(subsample: int, seed: int) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cv_cfg = cfg["cv"]
    model_cfg = _load_tuned_params(RESULTS_DIR, cfg["models"])
    es_frac = cv_cfg.get("es_frac", 0.1)

    print("Loading IEEE-CIS training data ...")
    df = load_train()
    X_all, y_all, time_all = to_pandas_features(df)
    cat_cols = existing_categorical(X_all)
    feature_names = X_all.columns.tolist()

    cv_idx, holdout_idx = holdout_split(time_all, cv_cfg["holdout_frac"])
    fit_idx, es_idx = split_train_for_early_stopping(cv_idx, es_frac=es_frac)

    X_fit = X_all.iloc[fit_idx]
    y_fit = y_all[fit_idx]
    X_es = X_all.iloc[es_idx]
    y_es = y_all[es_idx]
    X_te = X_all.iloc[holdout_idx]
    y_te = y_all[holdout_idx]

    # Deterministic subsample of the holdout for SHAP.
    rng = np.random.default_rng(seed)
    if subsample < len(X_te):
        sub_idx = rng.choice(len(X_te), size=subsample, replace=False)
        sub_idx.sort()  # keep temporal order for interpretability
    else:
        sub_idx = np.arange(len(X_te))
    X_sub = X_te.iloc[sub_idx].reset_index(drop=True)

    shap_by_model: Dict[str, np.ndarray] = {}
    holdout_scores_on_sub: Dict[str, np.ndarray] = {}
    global_imp: Dict[str, pd.DataFrame] = {}

    models = _build_models(cat_cols, model_cfg)
    for name, model in models.items():
        print(f"\n=== {name}: refit + SHAP on {len(X_sub):,}-row subsample ===")
        model.fit(X_fit, y_fit, X_es, y_es)
        holdout_scores_on_sub[name] = np.asarray(model.predict_proba(X_sub))
        print(f"  computing SHAP ...", flush=True)
        shap_vals = _shap_values(model, name, X_sub, cat_cols)
        shap_by_model[name] = shap_vals
        gi = global_importance(shap_vals, feature_names, top_k=30)
        gi.to_csv(RESULTS_DIR / f"shap_top_features_{name}.csv", index=False)
        global_imp[name] = gi
        print(f"  top 5 features: {list(gi['feature'].head(5))}")

    # Row-level disagreement (on the subsample)
    dis = top_disagreement_rows(holdout_scores_on_sub, top_k=50)
    dis.to_csv(RESULTS_DIR / "disagreement_rows.csv", index=False)
    print(f"\nWrote results/disagreement_rows.csv (max gap seen: {dis['max_pairwise_gap'].iloc[0]:.3f})")

    # Per-feature SHAP disagreement between model pairs
    per_pair = per_feature_shap_delta(shap_by_model, feature_names, top_k=30)
    for (a, b), tbl in per_pair.items():
        tbl.to_csv(RESULTS_DIR / f"shap_delta_{a}_vs_{b}.csv", index=False)
        print(f"Wrote results/shap_delta_{a}_vs_{b}.csv  (top feature: {tbl['feature'].iloc[0]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(subsample=args.subsample, seed=args.seed)


if __name__ == "__main__":
    main()
