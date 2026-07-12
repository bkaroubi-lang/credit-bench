"""Benchmark driver: temporal CV + strict holdout, per-model metrics, pairwise
DeLong CIs on AUC deltas, block bootstrap CIs on PR-AUC deltas, calibration.

Idempotent per model: running with `--models ft_transformer` after a Phase 1
run merges FT-Transformer's holdout scores into the existing npz and
recomputes all pairwise tables over the union of models present.

Run:
    python -m src.experiment                        # all models in registry
    python -m src.experiment --models lightgbm      # single model
    python -m src.experiment --models ft_transformer --subsample 100000
    python -m src.experiment --holdout-only         # skip CV loop
"""
from __future__ import annotations

# Set OMP/MKL thread caps BEFORE importing numpy/torch/xgboost/lightgbm so their
# thread pools initialize at the intended budget. Reads runtime.n_threads from
# configs/defaults.yaml via a tiny bootstrap peek — full config loading happens
# in run() as normal.
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
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from src.cv import expanding_time_folds, holdout_split, split_train_for_early_stopping
from src.data import existing_categorical, load_train, to_pandas_features
from src.eval import (
    block_bootstrap_paired_delta_ci,
    calibration_summary,
    core_metrics,
    delong_auc_ci,
    delong_paired_auc_diff_ci,
)
from src.models import DL_AVAILABLE, registry

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs" / "defaults.yaml"
RESULTS_DIR = REPO / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _build_models(
    model_names: list[str],
    cat_cols,
    model_cfg: dict,
    n_threads: Optional[int] = None,
) -> dict:
    reg = registry()
    out = {}
    for name in model_names:
        if name not in reg:
            hint = " (install torch + pytorch-tabular)" if name in {"ft_transformer", "tabnet", "node"} else ""
            raise ValueError(f"Unknown model '{name}'{hint}. Available: {list(reg)}")
        params = model_cfg.get(name)
        out[name] = reg[name](cat_cols, params, n_threads=n_threads)
    return out


def _load_tuned_params(results_dir: Path, model_cfg: dict) -> dict:
    tuned_path = results_dir / "tuned_params.json"
    if not tuned_path.exists():
        return model_cfg
    tuned = json.loads(tuned_path.read_text())
    merged = {k: dict(v) for k, v in model_cfg.items()}
    for name, params in tuned.items():
        merged.setdefault(name, {}).update(params)
        print(f"  using tuned params for {name}: {params}")
    return merged


def _subsample_indices(cv_idx: np.ndarray, subsample: int | None) -> np.ndarray:
    if subsample is None or subsample >= len(cv_idx):
        return cv_idx
    # Tail-N of the pre-holdout portion: preserves temporal integrity vs. the
    # (unchanged) holdout window. Better than random subsample for CPU smoke
    # tests because the model still sees the freshest history.
    return cv_idx[-subsample:]


# ---------- Idempotent result merging ----------

def _merge_holdout_scores(y_te: np.ndarray, new_scores: dict[str, np.ndarray]) -> dict:
    """Update results/holdout_scores.npz: keep existing model columns, overwrite
    the ones just recomputed. Returns the merged mapping name -> scores."""
    path = RESULTS_DIR / "holdout_scores.npz"
    existing: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path) as z:
            for key in z.files:
                if key == "y_te":
                    continue
                if not key.startswith("score_"):
                    continue
                existing[key.replace("score_", "")] = z[key]
    existing.update(new_scores)
    np.savez(path, y_te=y_te, **{f"score_{n}": s for n, s in existing.items()})
    return existing


def _merge_long_csv(path: Path, new_df: pd.DataFrame, dedup_cols: list[str]) -> pd.DataFrame:
    """Load existing long-format CSV if any, drop rows whose dedup_cols match
    any row in new_df (i.e. this run supersedes them), append new_df, save.
    """
    if path.exists():
        old = pd.read_csv(path)
        merge_keys = new_df[dedup_cols].drop_duplicates()
        old_kept = old.merge(merge_keys, on=dedup_cols, how="left", indicator=True)
        old_kept = old_kept[old_kept["_merge"] == "left_only"].drop(columns="_merge")
        merged = pd.concat([old_kept, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(path, index=False)
    return merged


def _merge_wide_csv(path: Path, new_rows: dict[str, dict]) -> pd.DataFrame:
    """For CSVs indexed by model name with metric columns; last-write-wins per
    model."""
    new_df = pd.DataFrame(new_rows).T
    new_df.index.name = "model"
    if path.exists():
        old = pd.read_csv(path, index_col=0)
        old.index.name = "model"
        for name in new_df.index:
            if name in old.index:
                old = old.drop(index=name)
        merged = pd.concat([old, new_df])
    else:
        merged = new_df
    merged.to_csv(path)
    return merged


# ---------- Downstream tables (recomputed over union of models) ----------

def _recompute_pair_tables(y_te: np.ndarray, all_scores: dict[str, np.ndarray], boot_cfg: dict):
    single = {name: delong_auc_ci(y_te, s) for name, s in all_scores.items()}
    single_df = pd.DataFrame(single).T
    single_df.index.name = "model"
    single_df.to_csv(RESULTS_DIR / "holdout_auc_delong.csv")

    names = list(all_scores.keys())
    pair_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = delong_paired_auc_diff_ci(y_te, all_scores[a], all_scores[b])
            pair_rows.append({"model_a": a, "model_b": b, **r})
    pd.DataFrame(pair_rows).to_csv(RESULTS_DIR / "holdout_auc_pair_delong.csv", index=False)

    pr_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = block_bootstrap_paired_delta_ci(
                average_precision_score,
                y_te,
                all_scores[a],
                all_scores[b],
                block_size=boot_cfg["block_size"],
                n_boot=boot_cfg["n_boot"],
                alpha=boot_cfg["alpha"],
                seed=boot_cfg["seed"],
            )
            pr_rows.append({"model_a": a, "model_b": b, **r})
    pd.DataFrame(pr_rows).to_csv(RESULTS_DIR / "holdout_prauc_pair_boot.csv", index=False)

    print("Recomputed pairwise tables over models:", list(all_scores))


def _merge_calibration_json(new_calib: dict) -> dict:
    path = RESULTS_DIR / "holdout_calibration.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(new_calib)
    path.write_text(json.dumps(existing, indent=2))
    return existing


# ---------- Main ----------

def run(models_to_run: list[str], subsample: int | None, holdout_only: bool) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cv_cfg = cfg["cv"]
    boot_cfg = cfg["bootstrap"]
    cal_cfg = cfg["calibration"]
    model_cfg = _load_tuned_params(RESULTS_DIR, cfg.get("models", {}))
    es_frac = cv_cfg.get("es_frac", 0.1)
    n_threads = (cfg.get("runtime") or {}).get("n_threads")
    if n_threads:
        print(f"Thread budget: pinning all models to {n_threads} threads")
        # torch interop must be set BEFORE any torch op; do it once here.
        try:
            import torch
            torch.set_num_threads(n_threads)
            try:
                torch.set_num_interop_threads(n_threads)
            except RuntimeError:
                pass  # already initialized; intra-op still applies per-fit
        except ImportError:
            pass

    if not models_to_run:
        models_to_run = cfg.get("models_to_run") or list(registry().keys())
    print(f"Running models: {models_to_run}")
    if not DL_AVAILABLE and any(m in {"ft_transformer", "tabnet", "node"} for m in models_to_run):
        raise SystemExit(
            "DL models requested but torch/pytorch-tabular not installed.\n"
            "  conda env update -f environment.yml"
        )

    print("Loading IEEE-CIS training data ...")
    df = load_train()
    print(f"  rows={df.height:,}  cols={df.width}")

    X_all, y_all, time_all = to_pandas_features(df)
    cat_cols = existing_categorical(X_all)
    print(f"  categorical cols in X: {len(cat_cols)}")

    cv_idx_full, holdout_idx = holdout_split(time_all, cv_cfg["holdout_frac"])
    cv_idx = _subsample_indices(cv_idx_full, subsample)
    if subsample:
        print(f"  --subsample: using tail {len(cv_idx):,} of pre-holdout (of {len(cv_idx_full):,})")
    print(f"  cv rows={len(cv_idx):,}  holdout rows={len(holdout_idx):,}")

    # -------------------- Cross-validation --------------------
    if not holdout_only:
        fold_rows = []
        # Re-derive folds on the (possibly subsampled) cv_idx by mimicking the
        # cv module's logic on this index sequence directly.
        n_cv = len(cv_idx)
        total_blocks = cv_cfg["n_folds"] + 1
        block_size = n_cv // total_blocks
        min_train_size = int(n_cv * cv_cfg["min_train_frac"])
        for k in range(cv_cfg["n_folds"]):
            train_end = max(block_size * (k + 1), min_train_size)
            val_end = block_size * (k + 2) if k < cv_cfg["n_folds"] - 1 else n_cv
            train_idx = cv_idx[:train_end]
            val_idx = cv_idx[train_end:val_end]

            fit_idx, es_idx = split_train_for_early_stopping(train_idx, es_frac=es_frac)
            X_fit, y_fit = X_all.iloc[fit_idx], y_all[fit_idx]
            X_es, y_es = X_all.iloc[es_idx], y_all[es_idx]
            X_va, y_va = X_all.iloc[val_idx], y_all[val_idx]
            print(
                f"\n=== Fold {k}: fit={len(fit_idx):,}  es={len(es_idx):,}  val={len(val_idx):,} ==="
            )
            for name, model in _build_models(models_to_run, cat_cols, model_cfg, n_threads=n_threads).items():
                print(f"  fitting {name} ...", flush=True)
                t0 = time.perf_counter()
                model.fit(X_fit, y_fit, X_es, y_es)
                fit_s = time.perf_counter() - t0
                val_scores = np.asarray(model.predict_proba(X_va))
                m = core_metrics(y_va, val_scores)
                print(
                    f"    auc={m['auc']:.4f}  pr_auc={m['pr_auc']:.4f}  "
                    f"brier={m['brier']:.5f}  fit={fit_s:.1f}s"
                )
                fold_rows.append({"fold": k, "model": name, **m, "fit_s": fit_s})

        cv_df = pd.DataFrame(fold_rows)
        merged_cv = _merge_long_csv(
            RESULTS_DIR / "cv_metrics.csv", cv_df, dedup_cols=["fold", "model"]
        )
        print(f"\nWrote/merged results/cv_metrics.csv  ({len(merged_cv)} rows)")

    # -------------------- Final holdout --------------------
    print(f"\n=== Final holdout: pre-holdout train={len(cv_idx):,}  test={len(holdout_idx):,} ===")
    fit_idx, es_idx = split_train_for_early_stopping(cv_idx, es_frac=es_frac)
    X_fit, y_fit = X_all.iloc[fit_idx], y_all[fit_idx]
    X_es, y_es = X_all.iloc[es_idx], y_all[es_idx]
    X_te, y_te = X_all.iloc[holdout_idx], y_all[holdout_idx]

    new_holdout_scores: Dict[str, np.ndarray] = {}
    new_holdout_metrics: Dict[str, dict] = {}
    new_calib: Dict[str, dict] = {}
    for name, model in _build_models(models_to_run, cat_cols, model_cfg, n_threads=n_threads).items():
        print(f"  fitting {name} on all pre-holdout data ...", flush=True)
        model.fit(X_fit, y_fit, X_es, y_es)
        scores = np.asarray(model.predict_proba(X_te))
        new_holdout_scores[name] = scores
        m = core_metrics(y_te, scores)
        new_holdout_metrics[name] = m
        new_calib[name] = calibration_summary(y_te, scores, n_bins=cal_cfg["n_bins"])
        print(f"    auc={m['auc']:.4f}  pr_auc={m['pr_auc']:.4f}  brier={m['brier']:.5f}")

    all_holdout_scores = _merge_holdout_scores(y_te, new_holdout_scores)
    _merge_wide_csv(RESULTS_DIR / "holdout_metrics.csv", new_holdout_metrics)
    _recompute_pair_tables(y_te, all_holdout_scores, boot_cfg)
    _merge_calibration_json(new_calib)

    print("\nDone. Models present in results:", sorted(all_holdout_scores.keys()))
    print("Run `python -m src.report` to regenerate the markdown report.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to run this pass. Defaults to configs/defaults.yaml models_to_run or all.",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Cap pre-holdout rows to the last N (tail). Holdout untouched.",
    )
    parser.add_argument("--holdout-only", action="store_true", help="Skip the CV loop.")
    args = parser.parse_args()
    run(models_to_run=args.models or [], subsample=args.subsample, holdout_only=args.holdout_only)


if __name__ == "__main__":
    main()
