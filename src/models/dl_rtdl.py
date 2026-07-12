"""FT-Transformer via rtdl-revisiting-models (Gorishniy et al.) with
torch-directml GPU support.

This replaces the pytorch-tabular-based FTTransformerWrap for two reasons:
- pytorch-tabular ties to PyTorch Lightning, which has no DirectML accelerator.
  Forcing DML tensors under a "cpu" Lightning accelerator segfaults in native
  code (0xC0000005). Not fixable without bespoke Lightning Accelerator work.
- rtdl-revisiting-models is the paper authors' modern-torch reference
  implementation. `FTTransformer.get_default_kwargs()` gives the exact
  hyperparameter set from the paper (3 blocks, d=192, 8 heads), so "default
  FT-Transformer" is well-defined and defensible in the report.

Device resolution: DirectML first (any DX12 GPU on Windows), then CPU. The
whole training + inference happens on the resolved device — no CPU/GPU
juggling within an epoch.
"""
from __future__ import annotations

import gc
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.models.gbm import TabularModel


def _resolve_device(pref: str = "auto") -> tuple[torch.device, str]:
    """Return (device, human-readable name). Prefers DirectML, falls back to
    CPU. Explicit 'cpu' or 'cuda' overrides."""
    if pref == "cpu":
        return torch.device("cpu"), "cpu"
    if pref == "cuda":
        return torch.device("cuda"), "cuda"
    # auto / dml
    try:
        import torch_directml

        if torch_directml.is_available() and torch_directml.device_count() > 0:
            dev = torch_directml.device()
            try:
                name = torch_directml.device_name(0)
            except Exception:
                name = "DirectML"
            return dev, f"DirectML: {name}"
    except ImportError:
        pass
    return torch.device("cpu"), "cpu"


class _CategoryEncoder:
    """Ordinal encoding fit on train; test-time unseen categories go to a
    dedicated bucket. Mirrors gbm._apply_category's contract: alignment to
    the training encoding, not fresh inference per split."""

    def __init__(self, cols: Sequence[str]):
        self.cols = list(cols)
        self.present: list[str] = []
        self.maps: dict[str, dict] = {}
        self.cardinalities: dict[str, int] = {}

    def fit(self, X: pd.DataFrame) -> "_CategoryEncoder":
        self.present = [c for c in self.cols if c in X.columns]
        for c in self.present:
            s = X[c].fillna("__NA__").astype(str)
            unique = sorted(s.unique())
            # Codes 0..N-1 for seen categories; code N is "unseen".
            self.maps[c] = {v: i for i, v in enumerate(unique)}
            self.cardinalities[c] = len(unique) + 1
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.present:
            return np.empty((len(X), 0), dtype=np.int64)
        out = np.zeros((len(X), len(self.present)), dtype=np.int64)
        for j, c in enumerate(self.present):
            s = X[c].fillna("__NA__").astype(str)
            m = self.maps[c]
            unseen_code = self.cardinalities[c] - 1
            out[:, j] = s.map(m).fillna(unseen_code).astype(np.int64).to_numpy()
        return out

    def get_cardinalities(self) -> list[int]:
        return [self.cardinalities[c] for c in self.present]


class _NumericScaler:
    """Standardize numerics on train stats; NaN filled with train mean."""

    def __init__(self, cols: Sequence[str]):
        self.cols = list(cols)
        self.present: list[str] = []
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "_NumericScaler":
        self.present = [c for c in self.cols if c in X.columns]
        arr = X[self.present].to_numpy(dtype=np.float32)
        self.means = np.nanmean(arr, axis=0).astype(np.float32)
        stds = np.nanstd(arr, axis=0).astype(np.float32)
        stds[stds == 0] = 1.0
        self.stds = stds
        # nanmean can be NaN if a column is entirely NaN in train; guard.
        self.means = np.where(np.isnan(self.means), 0.0, self.means).astype(np.float32)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self.present].to_numpy(dtype=np.float32)
        mask = np.isnan(arr)
        if mask.any():
            # broadcast fill with train means
            fill = np.broadcast_to(self.means, arr.shape)
            arr = np.where(mask, fill, arr)
        return (arr - self.means) / self.stds


class FTTransformerWrap(TabularModel):
    """FT-Transformer with rtdl-revisiting-models backbone.

    params override defaults; recognized keys are the training defaults
    (batch_size, max_epochs, lr, weight_decay, es_patience, device,
    eval_batch_size, seed) and any FTTransformer backbone kwarg
    (n_blocks, d_block, attention_n_heads, attention_dropout,
    ffn_d_hidden_multiplier, ffn_dropout, residual_dropout).
    """

    name = "ft_transformer"

    _TRAIN_KEYS = {
        "batch_size", "max_epochs", "lr", "weight_decay", "es_patience",
        "device", "eval_batch_size", "seed",
    }

    def __init__(
        self,
        cat_cols: Sequence[str],
        params: Optional[dict] = None,
        n_threads: Optional[int] = None,
    ):
        from rtdl_revisiting_models import FTTransformer

        self.cat_cols = list(cat_cols)
        # batch_size default is sized so that per-layer attention memory
        # (batch * n_heads * seq_len^2 * 4 bytes) plus its backward-pass
        # gradient copy fits in ~16 GB VRAM even for wide inputs like
        # IEEE-CIS (~433 tokens). At bs=64, 8 heads, seq=433: attention
        # weights are ~192 MB per layer forward + 192 MB backward, times
        # 3 layers ~1.2 GB; leaves room for FFN activations, dropout
        # masks, and the ~200 MB pre-loaded training tensors. For narrower
        # inputs (fewer than ~200 tokens), you can safely bump to 128 or
        # 256 via configs/defaults.yaml.
        # eval_batch_size must be <= batch_size on DirectML. DirectML's
        # allocator holds onto reserved training-sized chunks and does not
        # defragment; a larger eval batch needs a larger contiguous
        # allocation which the fragmented pool can't provide even when
        # aggregate free memory is plenty. Matching sizes ensures reuse
        # of the training-time chunks.
        self.train_params = {
            "batch_size": 64,
            "max_epochs": 30,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "es_patience": 5,
            "device": "auto",
            "eval_batch_size": 64,
            "seed": 0,
        }
        self.backbone_params = dict(FTTransformer.get_default_kwargs())
        if params:
            for k, v in params.items():
                if k in self._TRAIN_KEYS:
                    self.train_params[k] = v
                else:
                    self.backbone_params[k] = v
        self.n_threads = n_threads
        self.model: nn.Module | None = None
        self.device: torch.device | None = None
        self.device_name: str | None = None
        self.cat_enc: _CategoryEncoder | None = None
        self.num_scaler: _NumericScaler | None = None

    # ---- helpers ----
    def _prep_arrays(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        X_num = self.num_scaler.transform(X)
        X_cat = self.cat_enc.transform(X)
        if y is not None:
            return X_num, X_cat, np.asarray(y, dtype=np.float32)
        return X_num, X_cat, None

    def _predict_logits(self, X_num_t: torch.Tensor, X_cat_t: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        bs = self.train_params["eval_batch_size"]
        outs: list[torch.Tensor] = []
        n = X_num_t.shape[0]
        # Only pass x_cat when cardinalities are non-empty; otherwise rtdl
        # expects None.
        cat_ok = self.cat_enc.get_cardinalities() != []
        with torch.no_grad():
            for start in range(0, n, bs):
                xn = X_num_t[start:start + bs]
                xc = X_cat_t[start:start + bs] if cat_ok else None
                y = self.model(xn, xc).squeeze(-1)
                outs.append(y)
        return torch.cat(outs)

    # ---- TabularModel interface ----
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from rtdl_revisiting_models import FTTransformer

        if self.n_threads is not None:
            torch.set_num_threads(self.n_threads)

        torch.manual_seed(self.train_params["seed"])
        np.random.seed(self.train_params["seed"])

        cat_present = [c for c in self.cat_cols if c in X_train.columns]
        num_cols = [c for c in X_train.columns if c not in cat_present]
        self.cat_enc = _CategoryEncoder(cat_present).fit(X_train)
        self.num_scaler = _NumericScaler(num_cols).fit(X_train)

        device, device_name = _resolve_device(self.train_params.get("device", "auto"))
        self.device, self.device_name = device, device_name
        print(f"  FT-Transformer on {device_name}")

        # Preprocess
        X_num_tr, X_cat_tr, y_tr = self._prep_arrays(X_train, y_train)
        cards = self.cat_enc.get_cardinalities()
        cat_ok = len(cards) > 0
        print(f"  n_cont={X_num_tr.shape[1]}  n_cat={len(cards)}  n_train={len(y_tr):,}")

        # Model
        model = FTTransformer(
            n_cont_features=X_num_tr.shape[1],
            cat_cardinalities=cards,
            d_out=1,
            **self.backbone_params,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  parameters: {n_params:,}")

        # Optimizer with paper-style parameter groups (no wd on embeds/norms)
        param_groups = model.make_parameter_groups()
        opt = torch.optim.AdamW(
            param_groups,
            lr=self.train_params["lr"],
            weight_decay=self.train_params["weight_decay"],
        )
        loss_fn = nn.BCEWithLogitsLoss()

        # Move all training tensors to device up-front (100K x 434 f32 ~ 170 MB)
        X_num_tr_t = torch.from_numpy(X_num_tr).to(device)
        X_cat_tr_t = torch.from_numpy(X_cat_tr).to(device) if cat_ok else None
        y_tr_t = torch.from_numpy(y_tr).to(device)

        val_ready = X_val is not None and y_val is not None
        if val_ready:
            X_num_va, X_cat_va, y_va = self._prep_arrays(X_val, y_val)
            X_num_va_t = torch.from_numpy(X_num_va).to(device)
            X_cat_va_t = torch.from_numpy(X_cat_va).to(device) if cat_ok else None
            y_va_t = torch.from_numpy(y_va).to(device)

        n_train = len(y_tr)
        bs = self.train_params["batch_size"]
        max_epochs = self.train_params["max_epochs"]
        es_patience = self.train_params["es_patience"]
        self.model = model  # so _predict_logits works even if we early-out

        best_val_loss = float("inf")
        best_state: dict | None = None
        patience = 0

        for epoch in range(max_epochs):
            model.train()
            # randperm on CPU then move — DML may not support randperm natively
            perm = torch.randperm(n_train).to(device)
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.perf_counter()
            for start in range(0, n_train, bs):
                idx = perm[start:start + bs]
                xn = X_num_tr_t.index_select(0, idx)
                xc = X_cat_tr_t.index_select(0, idx) if cat_ok else None
                yy = y_tr_t.index_select(0, idx)
                logits = model(xn, xc).squeeze(-1)
                loss = loss_fn(logits, yy)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)
            epoch_time = time.perf_counter() - t0

            # Free training-batch scratch tensors before eval. DirectML's
            # allocator holds onto reserved blocks; without an explicit
            # collection here the very next allocation (val attention
            # scores) can OOM even at modest sizes.
            model.zero_grad(set_to_none=True)
            gc.collect()

            if val_ready:
                val_logits = self._predict_logits(X_num_va_t, X_cat_va_t)
                val_loss = float(loss_fn(val_logits, y_va_t).detach().cpu().item())
                val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
                val_auc = float(roc_auc_score(y_va, val_probs))
                print(
                    f"    epoch {epoch:2d}: train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  "
                    f"time={epoch_time:.1f}s",
                    flush=True,
                )
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= es_patience:
                        print(f"    early stopping at epoch {epoch}")
                        break
            else:
                print(f"    epoch {epoch:2d}: train_loss={train_loss:.4f}  time={epoch_time:.1f}s", flush=True)

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Stream the holdout from CPU batch-by-batch. Moving the full
        holdout to DML in one shot allocates a single large tensor that
        the fragmented DML pool can refuse even when aggregate free memory
        is plenty. Per-batch copies are tiny (~100 KB) and DML reuses the
        bs=eval_batch_size chunks already sized during training."""
        X_num, X_cat, _ = self._prep_arrays(X)
        cat_ok = self.cat_enc.get_cardinalities() != []
        bs = self.train_params["eval_batch_size"]

        self.model.eval()
        outs: list[np.ndarray] = []
        n = len(X_num)
        with torch.no_grad():
            for start in range(0, n, bs):
                end = start + bs
                xn = torch.from_numpy(X_num[start:end]).to(self.device)
                xc = torch.from_numpy(X_cat[start:end]).to(self.device) if cat_ok else None
                logits = self.model(xn, xc).squeeze(-1)
                outs.append(torch.sigmoid(logits).detach().cpu().numpy())
                del xn, xc, logits
        return np.concatenate(outs)
