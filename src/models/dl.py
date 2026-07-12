"""Tabular deep-learning wrappers via pytorch-tabular.

Same TabularModel interface as the GBM wrappers — fit(X_train, y_train, X_val,
y_val) then predict_proba(X). This means experiment.py, tune.py, analysis.py
don't need to know anything about torch; they just build a model from the
registry and call the same three methods.

CPU-only defaults. Phase 3 (GPU) will patch the Trainer to point at a
DirectML/ROCm device.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.models.gbm import TabularModel


# ---------- Shared categorical prep ----------

def _prep_dl(X: pd.DataFrame, cat_cols: Sequence[str]) -> pd.DataFrame:
    """String-encode categoricals with NaN sentinel. pytorch-tabular's
    OrdinalEncoder assigns unseen categories at predict time a fresh code,
    so we don't need to align categories manually the way the GBMs do."""
    X = X.copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna("__NA__").astype(str)
    return X


def _split_columns(X: pd.DataFrame, cat_cols: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return (cat_cols_in_X, cont_cols_in_X). Everything not in cat_cols is
    treated as continuous."""
    cats = [c for c in cat_cols if c in X.columns]
    conts = [c for c in X.columns if c not in cats]
    return cats, conts

def _impute_continuous(X: pd.DataFrame, conts: Sequence[str], medians: dict) -> pd.DataFrame:
    X = X.copy()
    for c in conts:
        if c in X.columns:
            X[c] = X[c].fillna(medians.get(c, 0.0))
    return X


# ---------- Base wrapper ----------

class _PytorchTabularWrap(TabularModel):
    """Concrete subclasses set self._make_model_config(cats, conts)."""

    name: str = "abstract-dl"
    _target_col: str = "_target"

    def __init__(
        self,
        cat_cols: Sequence[str],
        params: Optional[dict] = None,
        trainer_params: Optional[dict] = None,
        n_threads: Optional[int] = None,
    ):
        self.cat_cols = list(cat_cols)
        self.user_params = dict(params) if params else {}
        self.n_threads = n_threads
        # Defaults: rely on pytorch-tabular's built-in checkpointing (to
        # saved_models/, which .gitignore excludes). `load_best=True` requires
        # the checkpoint, so we don't disable it.
        self.trainer_params = {
            "batch_size": 2048,
            "max_epochs": 30,
            "early_stopping": "valid_loss",
            "early_stopping_patience": 5,
            "early_stopping_mode": "min",
            "accelerator": "cpu",
            "progress_bar": "none",
            "trainer_kwargs": {"enable_model_summary": False},
        }
        if trainer_params:
            self.trainer_params.update(trainer_params)
        self._model = None
        self._cats: list[str] = []
        self._conts: list[str] = []

    def _make_model_config(self, cats: list[str], conts: list[str]):
        raise NotImplementedError

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch

        from pytorch_tabular import TabularModel as PTTabular
        from pytorch_tabular.config import (
            DataConfig,
            OptimizerConfig,
            TrainerConfig,
        )

        # Pin torch to the same thread budget the GBMs are pinned to, so
        # wall-clock is like-for-like. Intra-op only — interop is set at
        # process start (see experiment.py) since it can't change after
        # first torch op.
        if self.n_threads is not None:
            torch.set_num_threads(self.n_threads)

        Xt = _prep_dl(X_train, self.cat_cols)
        cats, conts = _split_columns(Xt, self.cat_cols)
        self._cats, self._conts = cats, conts

        # pytorch-tabular imputes NaN only for categorical cols; continuous
        # NaN passes straight into StandardScaler and then the network,
        # producing NaN outputs. Impute with training medians (reused for
        # val/holdout via self._cont_medians -> no leakage).
        self._cont_medians = {
            c: (m if pd.notna(m) else 0.0)
            for c, m in Xt[conts].median().to_dict().items()
        }
        Xt = _impute_continuous(Xt, conts, self._cont_medians)

        train_df = Xt.assign(**{self._target_col: y_train})
        val_df = None
        if X_val is not None:
            Xv = _prep_dl(X_val, self.cat_cols)
            Xv = _impute_continuous(Xv, conts, self._cont_medians)
            val_df = Xv.assign(**{self._target_col: y_val})

        data_config = DataConfig(
            target=[self._target_col],
            continuous_cols=conts,
            categorical_cols=cats,
            handle_missing_values=True,
            normalize_continuous_features=True,
        )
        trainer_config = TrainerConfig(**self.trainer_params)
        optimizer_config = OptimizerConfig()
        model_config = self._make_model_config(cats, conts)

        self._model = PTTabular(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        self._model.fit(train=train_df, validation=val_df)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xp = _prep_dl(X, self.cat_cols)
        Xp = _impute_continuous(Xp, self._conts, self._cont_medians)
        preds = self._model.predict(Xp)
        # pytorch-tabular column naming: "<target>_1_probability"
        prob_col = f"{self._target_col}_1_probability"
        if prob_col in preds.columns:
            return preds[prob_col].to_numpy()
        # Fallback: find any *_1_probability column
        candidates = [c for c in preds.columns if c.endswith("_1_probability")]
        if candidates:
            return preds[candidates[0]].to_numpy()
        raise RuntimeError(
            f"No positive-class probability column found in predictions; "
            f"got columns {list(preds.columns)}"
        )


# ---------- Concrete wrappers ----------

class FTTransformerWrap(_PytorchTabularWrap):
    name = "ft_transformer"

    def _make_model_config(self, cats, conts):
        from pytorch_tabular.models import FTTransformerConfig
        params = {
            "task": "classification",
            "input_embed_dim": 32,
            "num_heads": 4,
            "num_attn_blocks": 3,
            "attn_dropout": 0.1,
            "add_norm_dropout": 0.1,
            "ff_dropout": 0.1,
            "ff_hidden_multiplier": 4,
            "metrics": ["auroc"],
            "metrics_prob_input": [True],
            "metrics_params": [{"task": "multiclass", "num_classes": 2}],
        }
        params.update(self.user_params)
        return FTTransformerConfig(**params)


class TabNetWrap(_PytorchTabularWrap):
    name = "tabnet"

    def _make_model_config(self, cats, conts):
        from pytorch_tabular.models import TabNetModelConfig
        params = {
            "task": "classification",
            "n_d": 16,
            "n_a": 16,
            "n_steps": 3,
            "gamma": 1.3,
            "n_independent": 2,
            "n_shared": 2,
            "virtual_batch_size": 512,
            "mask_type": "sparsemax",
            "metrics": ["auroc"],
            "metrics_prob_input": [True],
            "metrics_params": [{"task": "multiclass", "num_classes": 2}],
        }
        params.update(self.user_params)
        return TabNetModelConfig(**params)


class NodeWrap(_PytorchTabularWrap):
    name = "node"

    def _make_model_config(self, cats, conts):
        from pytorch_tabular.models import NodeConfig
        # Conservative defaults: NODE is memory-hungry. 256/depth=4 fits
        # comfortably on IEEE-CIS on CPU without OOM.
        params = {
            "task": "classification",
            "num_layers": 1,
            "num_trees": 256,
            "depth": 4,
            "choice_function": "entmax15",
            "bin_function": "entmoid15",
            "metrics": ["auroc"],
            "metrics_prob_input": [True],
            "metrics_params": [{"task": "multiclass", "num_classes": 2}],
        }
        params.update(self.user_params)
        return NodeConfig(**params)
