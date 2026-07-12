"""GBM wrappers with a common interface and native categorical handling.

Each wrapper:
- accepts a pandas DataFrame X with mixed numeric + categorical columns
- knows which columns are categorical (passed in at construction)
- fits with early stopping on the validation set if provided
- returns positive-class probabilities from predict_proba(X)
"""
from typing import Optional, Sequence

import numpy as np
import pandas as pd


class TabularModel:
    name: str = "abstract"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TabularModel":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


def _as_category(X: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Build fresh categories from X. Use only on the training set -- for
    validation/test data, use _apply_category instead so categories stay
    aligned with what the model was trained on."""
    X = X.copy()
    for c in cols:
        if c in X.columns and not isinstance(X[c].dtype, pd.CategoricalDtype):
            s = X[c]
            if pd.api.types.is_float_dtype(s):
                s = s.astype("Int64")
            cat = s.astype("category")
            if pd.api.types.is_numeric_dtype(cat.cat.categories):
                cat = cat.cat.rename_categories(cat.cat.categories.astype(str))
            X[c] = cat
    return X


def _apply_category(X: pd.DataFrame, ref: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Cast `cols` in X to the *same* categories as in `ref` (the training
    set), instead of inferring fresh ones. Time-based CV splits mean the
    validation/test slice can contain card/addr/id values never seen in the
    earlier training slice; independently re-inferring categories on each
    slice misaligns codes and makes XGBoost raise "category not in the
    training set". Any value not present in ref's categories becomes NaN
    (missing) here, which both LightGBM and XGBoost already handle natively.
    """
    X = X.copy()
    for c in cols:
        if c not in X.columns or c not in ref.columns:
            continue
        dtype = ref[c].dtype
        if not isinstance(dtype, pd.CategoricalDtype):
            continue
        s = X[c]
        notna = s.notna()
        if pd.api.types.is_float_dtype(s):
            s = s.astype("Int64").astype(str)
        else:
            s = s.astype(str)
        s = s.where(notna, np.nan)
        X[c] = pd.Categorical(s, categories=dtype.categories, ordered=False)
    return X


class LGBMWrap(TabularModel):
    name = "lightgbm"

    def __init__(
        self,
        cat_cols: Sequence[str],
        params: Optional[dict] = None,
        n_threads: Optional[int] = None,
    ):
        import lightgbm as lgb

        self.lgb = lgb
        self.cat_cols = list(cat_cols)
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 100,
            "verbose": -1,
        }
        if n_threads is not None:
            self.params["num_threads"] = n_threads
        if params:
            self.params.update(params)
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        Xt = _as_category(X_train, self.cat_cols)
        self._Xt_ref = Xt  # reference categories for val/predict
        cats = [c for c in self.cat_cols if c in Xt.columns]
        dt = self.lgb.Dataset(Xt, label=y_train, categorical_feature=cats)
        valid_sets = [dt]
        callbacks = [self.lgb.log_evaluation(0)]
        if X_val is not None:
            Xv = _apply_category(X_val, Xt, self.cat_cols)
            dv = self.lgb.Dataset(Xv, label=y_val, categorical_feature=cats, reference=dt)
            valid_sets.append(dv)
            callbacks.append(self.lgb.early_stopping(100, verbose=False))
        self.model = self.lgb.train(
            self.params, dt, num_boost_round=5000, valid_sets=valid_sets, callbacks=callbacks
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xp = _apply_category(X, self._Xt_ref, self.cat_cols)
        return self.model.predict(Xp, num_iteration=self.model.best_iteration)


class XGBWrap(TabularModel):
    name = "xgboost"

    def __init__(
        self,
        cat_cols: Sequence[str],
        params: Optional[dict] = None,
        n_threads: Optional[int] = None,
    ):
        import xgboost as xgb

        self.xgb = xgb
        self.cat_cols = list(cat_cols)
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "enable_categorical": True,
        }
        if n_threads is not None:
            self.params["nthread"] = n_threads
        if params:
            self.params.update(params)
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        Xt = _as_category(X_train, self.cat_cols)
        self._Xt_ref = Xt  # reference categories for val/predict
        dtrain = self.xgb.DMatrix(Xt, label=y_train, enable_categorical=True)
        evals = [(dtrain, "train")]
        early = None
        if X_val is not None:
            Xv = _apply_category(X_val, Xt, self.cat_cols)
            dval = self.xgb.DMatrix(Xv, label=y_val, enable_categorical=True)
            evals.append((dval, "val"))
            early = 100
        self.model = self.xgb.train(
            self.params,
            dtrain,
            num_boost_round=5000,
            evals=evals,
            early_stopping_rounds=early,
            verbose_eval=False,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xp = _apply_category(X, self._Xt_ref, self.cat_cols)
        d = self.xgb.DMatrix(Xp, enable_categorical=True)
        return self.model.predict(d)


class CatWrap(TabularModel):
    name = "catboost"

    def __init__(
        self,
        cat_cols: Sequence[str],
        params: Optional[dict] = None,
        n_threads: Optional[int] = None,
    ):
        import catboost as cb

        self.cb = cb
        self.cat_cols = list(cat_cols)
        self.params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "learning_rate": 0.05,
            "depth": 8,
            "iterations": 5000,
            "l2_leaf_reg": 3.0,
            "verbose": False,
        }
        if n_threads is not None:
            self.params["thread_count"] = n_threads
        if params:
            self.params.update(params)
        self.model = None

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        # CatBoost's cat_features expects string or int, NaN not allowed.
        X = X.copy()
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].fillna("__NA__").astype(str)
        return X

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        Xt = self._prep(X_train)
        cats = [c for c in self.cat_cols if c in Xt.columns]
        train_pool = self.cb.Pool(Xt, y_train, cat_features=cats)
        clf = self.cb.CatBoostClassifier(**self.params)
        if X_val is not None:
            Xv = self._prep(X_val)
            val_pool = self.cb.Pool(Xv, y_val, cat_features=cats)
            clf.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
        else:
            clf.fit(train_pool)
        self.model = clf
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xp = self._prep(X)
        return self.model.predict_proba(Xp)[:, 1]
