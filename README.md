# credit-bench

**When does deep learning beat gradient boosting on credit/payments data?**

A rigorous benchmark of tuned XGBoost, LightGBM, and CatBoost against a modern
tabular DL baseline (FT-Transformer, `rtdl-revisiting-models` reference
implementation) on IEEE-CIS Fraud Detection (~590K transactions, temporal
holdout). All pairwise comparisons carry confidence intervals — DeLong for AUC
deltas, block bootstrap for PR-AUC — so I can report *significance*, not just
point estimates.

## Results (holdout AUC, DeLong 95% CI)

| Model | AUC | vs FT-Transformer ΔAUC (95% CI, p) |
|---|---:|---|
| CatBoost | 0.9139 [0.909, 0.919] | +0.0705 [+0.064, +0.077], p ≈ 0 |
| XGBoost | 0.9124 [0.907, 0.917] | +0.0690 [+0.063, +0.075], p ≈ 0 |
| LightGBM | 0.9072 [0.902, 0.912] | +0.0638 [+0.057, +0.070], p ≈ 0 |
| FT-Transformer | 0.8435 [0.837, 0.850] | — |

**All three tuned GBMs beat the default FT-Transformer by 6.4–7.1 AUC points**
(highly significant). CatBoost vs XGBoost is a statistical tie (Δ 0.0015,
p=0.39). Full write-up, methodology, per-fold CV, calibration curves, SHAP
disagreement analysis, and honest caveats:

**→ [results/report.md](results/report.md)**

**Companion project on LLM-agent evaluation**:
[ap-agent](https://github.com/bkaroubi-lang/ap-agent) — a Pydantic-AI accounts-payable
agent + judge-calibration harness. Headline: a finding that did *not* replicate — two Groq judges showed a kappa "collapse" in July 2026, but the key judge model was later retired by Groq and re-running with a current model pair reversed it. The durable lesson (inter-judge agreement ≠ judge quality) survives, and the repo now ships a record/replay cache so results reproduce offline.

---

## Longer form

The whole point is to answer the DL-vs-GBM question honestly: for many tabular
datasets the difference is small and dataset-dependent, and light tuning of a
GBM matters more than the choice between them. What separates this project
from a Kaggle notebook:

- Confidence intervals on the *pairwise deltas* between models (DeLong for AUC,
  block bootstrap for the rest), not just point estimates per model.
- Proper temporal cross-validation (expanding-window folds + strict held-out
  test period at the end).
- Calibration analysis: reliability curves + Brier score, not just discrimination.
- SHAP contrastive analysis: where do DL representations diverge from tree-based
  representations, and does that divergence correlate with error patterns?

## What was built

**Phase 1 — GBM stack (done).**
- IEEE-CIS Fraud Detection dataset (~590K transactions, Dec 2017 – Jun 2018).
- XGBoost, LightGBM, CatBoost with native categorical support.
- Temporal CV + DeLong CIs + block bootstrap + reliability curves + SHAP.

**Phase 2 — tabular DL baseline (done, FT-Transformer).**
- FT-Transformer (`rtdl-revisiting-models`, paper-default kwargs, no tuning),
  plugged into the same interface so `experiment.py` / `analysis.py` /
  `report.py` work identically. Its holdout scores merge into
  `holdout_scores.npz` and all pairwise CI tables recompute over the union of
  models — that's how the headline table above was produced.
- CPU proved non-viable for the DL leg (~2.1 hr/epoch), so FT-Transformer ran
  on an AMD RX 6900 XT via `torch-directml`, after a pivot off PyTorch Lightning
  (which has no DirectML backend) to a custom training loop. Full story and the
  preserved CPU checkpoint caveat are in the report.

**Not done (honest scope / next passes).**
- DL leg was trained on a 100K-tail subsample at default hyperparameters — a
  full-data, Optuna-tuned DL run is the obvious next step (the report argues it
  is unlikely to close the ~6 AUC-point gap, but it is untested).
- TabNet / NODE wrappers exist in the interface but were not run.
- A second dataset (Home Credit Default Risk) to test whether the story
  generalizes from fraud to credit scoring.

## Setup

**Recommended — [uv](https://docs.astral.sh/uv/)** (Linux/macOS; fetches Python 3.11, no conda needed):

```bash
uv sync                    # creates .venv with all deps (torch from the CPU wheel index)
uv run python run_all.py   # tune -> benchmark -> analysis -> report
```

Individual stages: `uv run python -m src.tune`, `uv run python -m src.experiment`, etc.

**Alternative — original Windows + Miniconda path:**

Prereq: Windows 11, Miniconda not required to be pre-installed.

```powershell
# From this directory:
.\scripts\install_miniconda.ps1        # one-time, silent install to %USERPROFILE%\miniconda3
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda env create -f environment.yml    # creates env "credit-bench"
conda activate credit-bench
.\scripts\download_ieee_cis.ps1        # needs %USERPROFILE%\.kaggle\kaggle.json
python run_all.py                       # tune -> benchmark -> analysis -> report
```

`run_all.py` orchestrates the four stages and caches each output — subsequent
runs skip completed steps unless `--force` is passed.

To run stages individually:

```powershell
python -m src.tune                     # Optuna, ~25 trials/model, writes tuned_params.json
python -m src.experiment               # temporal CV + holdout, writes results/*.csv
python -m src.analysis                 # per-model SHAP + cross-model disagreement
python -m src.report                   # results/report.md + plots/
```

## Adding the DL leg on top of Phase 1

Assuming Phase 1 has been run (results/ populated):

```powershell
conda env update -f environment.yml    # adds torch (CPU) + pytorch-tabular
# Optional: uncomment DL entries in configs/defaults.yaml -> models_to_run

# CPU-friendly smoke test: last 100k rows of pre-holdout
python -m src.experiment --models ft_transformer --subsample 100000 --holdout-only

# Full runs (slow on CPU — see wall-clock estimates below)
python -m src.experiment --models ft_transformer --holdout-only
python -m src.experiment --models tabnet         --holdout-only
python -m src.experiment --models node           --holdout-only

python -m src.report                   # regenerates report.md over union of models
```

Each DL run merges into `results/holdout_scores.npz` idempotently and
recomputes all pairwise CI tables. So you can add DL models one at a time
without losing GBM results.

**CPU wall-clock estimates on IEEE-CIS (~530K pre-holdout rows, 400 cols):**
- FT-Transformer, holdout-only, 30 epochs w/ early stopping: ~2-4 hr
- TabNet, holdout-only: ~1-3 hr
- NODE, holdout-only: ~3-6 hr (memory-heavy — reduce `num_trees` if OOM)

For a first pass, use `--subsample 100000 --holdout-only`: cuts wall-clock
by ~5x and gives you signal-vs-noise on whether the DL leg is competitive
before you commit to a full run.

## Layout

```
credit-bench/
  environment.yml              conda env spec (Phase 1)
  run_all.py                   orchestrator (tune -> experiment -> analysis -> report)
  scripts/
    install_miniconda.ps1      one-time Miniconda installer
    download_ieee_cis.ps1      Kaggle CLI wrapper
  src/
    data.py                    IEEE-CIS loader
    cv.py                      expanding-window temporal folds + within-fold ES split
    eval.py                    metrics, DeLong CIs, block bootstrap, calibration
    models/
      __init__.py              registry() with conditional DL exports
      gbm.py                   XGBoost/LightGBM/CatBoost wrappers
      dl.py                    FT-Transformer/TabNet/NODE via pytorch-tabular
    tune.py                    Optuna tuning on the first CV fold (GBMs)
    experiment.py              CV + holdout driver, idempotent per model
    analysis.py                SHAP + cross-model disagreement
    report.py                  markdown + PNG report
  configs/
    defaults.yaml              CV/tuning/analysis settings + per-model defaults
  data/raw/                    IEEE-CIS csvs land here
  results/                     benchmark tables, plots, CI tables, report.md
```

## Threading — like-for-like wall-clock

Every model is pinned to a shared thread budget (default 8, in
`configs/defaults.yaml` under `runtime.n_threads`). Without this pin:

- GBM libraries saturate all cores on histogram/split kernels (~linear scaling).
- PyTorch on CPU gets much less from the same cores on tabular DL (attention
  and embedding lookups don't parallelize as well).

Leaving both at library defaults would confound reported runtime with
parallelization efficiency — the GBMs would look ~2x faster than they should
in an algorithm-vs-algorithm comparison. The pin makes wall-clock a
defensible quantity in the report. Accuracy metrics are unaffected either
way (thread count doesn't move AUC).

Set `runtime.n_threads: null` in the config to disable and use each library's
defaults instead.

## Notes on the framing

The story that makes this a portfolio piece and not a demo:

1. Motivation: the DL-vs-GBM debate on tabular data is overemphasized. Cite the
   Grinsztajn et al. (2022) benchmark, Shwartz-Ziv & Armon (2022), and the
   generic finding that light tuning of a GBM often closes any gap.
2. Method: proper temporal CV, pairwise CIs, calibration, contrastive SHAP.
3. Expected finding on credit/payments: GBMs win or tie on discrimination in
   most settings, but *may* lose on calibration in imbalanced regimes, and
   *may* lose on the tails when structural features (device fingerprints, card
   embeddings) matter. Honest reporting either way.
4. Conclusion: when to reach for DL on tabular data, when not to.
