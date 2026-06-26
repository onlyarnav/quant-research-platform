## Updated `PROJECT_STATUS.md`

Replace the relevant sections with this:

```markdown
# Quantitative AI Research Platform — Project Status

## Continue From
`src/backtesting/` — Simulate trades using signals, with transaction costs + slippage.

---

## Project Structure (Current State)

```
quant-research-platform/
├── config/
│   ├── constants.py       ✅ COMPLETE
│   ├── settings.py        ✅ COMPLETE (added TRAIN_SPLIT_RATIO, VAL_SPLIT_RATIO)
│   └── __init__.py        ✅ COMPLETE
├── src/
│   ├── database/
│   │   ├── base.py              ✅ COMPLETE
│   │   ├── sessions.py          ✅ COMPLETE
│   │   ├── __init__.py          ✅ COMPLETE
│   │   ├── models/               ✅ COMPLETE (all 6 tables)
│   │   └── repository/           ✅ COMPLETE
│   │       ├── __init__.py
│   │       ├── asset_repository.py
│   │       ├── price_repository.py
│   │       ├── feature_repository.py
│   │       ├── signal_repository.py
│   │       ├── trade_repository.py
│   │       └── portfolio_metric_repository.py
│   ├── data_pipeline/      ✅ COMPLETE
│   ├── data_storage/       ✅ COMPLETE
│   ├── feature_engineering/ ✅ COMPLETE
│   ├── models/             ✅ COMPLETE
│   │   ├── dataset_builder.py
│   │   ├── base_model.py
│   │   ├── linear_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── model_evaluator.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── signals/             ✅ COMPLETE
│   │   ├── threshold_signal.py
│   │   ├── rank_signal.py
│   │   └── signal_pipeline.py
│   ├── backtesting/   ⬅️ START HERE
│   │   ├── backtesting_engine.py
│   │   ├── position_manager.py
│   │   └── portfolio_simulator.py
│   ├── portfolio/       ⬜ NOT STARTED
│   ├── analytics/       ⬜ NOT STARTED
│   ├── dashboard/       ⬜ NOT STARTED
│   ├── infra/           ⬜ NOT STARTED
│   └── utils/
│       ├── logger.py    ✅ COMPLETE
│       └── __init__.py  ✅ COMPLETE
├── tests/
│   ├── data_pipeline/    ✅ COMPLETE
│   ├── data_storage/     ✅ COMPLETE
│   ├── database/         ✅ COMPLETE (6 repos, all passing)
│   ├── models/           ✅ COMPLETE
│   │   ├── conftest.py
│   │   ├── test_dataset_builder.py
│   │   ├── test_base_model.py
│   │   ├── test_linear_model.py
│   │   ├── test_xgboost_model.py
│   │   ├── test_lightgbm_model.py
│   │   ├── test_model_evaluator.py
│   │   ├── test_training_pipeline.py
│   │   └── test_prediction_pipeline.py
│   ├── signals/          ✅ COMPLETE
│   │   ├── test_threshold_signal.py
│   │   ├── test_rank_signal.py
│   │   └── test_signal_pipeline.py
│   └── backtesting/      ⬜ NOT STARTED
├── scripts/
│   └── run_ingestion.py  ✅ COMPLETE
├── alembic/               ✅ COMPLETE (single linear migration: d74f3932a61b)
├── docker-compose.yml    ✅ COMPLETE
├── pyproject.toml        ✅ COMPLETE (added pytest-mock>=3.0)
└── .env                  ✅ COMPLETE
```

---

## Test Status
- `tests/database/` — 6 repositories, all passing
- `tests/models/` — 8 files, all passing
- `tests/signals/` — 3 files, all passing
- Total test suite: green across all completed layers

---

## Infrastructure Status
- PostgreSQL running on port 5432 (container: quant_postgres)
- MLflow running on port 5000 (container: quant_mlflow)
- Alembic migrations: single linear history, applied successfully
- All 6 platform tables created and verified

---

## Key Learnings & Principles (Session Additions)

**Models layer:**
- ABC abstract method bodies never execute — concrete guard helpers (`_check_is_fitted()`) must be called explicitly by every subclass, not assumed via inheritance
- XGBoost 2.0+: `early_stopping_rounds` is a constructor argument, not a `.fit()` kwarg
- `ModelEvaluator.information_coefficient()` returns `NaN` with a logged warning on constant input, rather than letting scipy raise a silent `ConstantInputWarning`
- `DatasetBuilder.build()`: clamp date-cutoff indices *before* validating `train_idx < val_idx`, never after — clamping after validation can silently reintroduce the bug the validation was meant to catch

**Signals layer:**
- `ThresholdSignalGenerator` evaluates each row independently (no cross-sectional dependency)
- `RankSignalGenerator` ranks within each `date` group independently — cross-sectional by design, uses `method="first"` for deterministic tie-breaking
- Both signal generators share a structural `Protocol` interface (`.generate()`), consumed generically by `SignalPipeline` — no shared base class needed

**Testing infrastructure:**
- `pytest-mock` required for `mocker` fixture — must be in `pyproject.toml`
- MLflow must always be mocked via `mocker.patch("module.mlflow")` in every test path that calls `TrainingPipeline.run()` — forgetting this causes real network calls to `MLFLOW_TRACKING_URI`, which manifests as a test "hanging" rather than failing cleanly
- `ParquetWriter`'s `LAYER_PATH_MAP` is built once at import time — monkeypatching `settings.DATA_*_PATH` alone does not update it; both `settings` attribute and `LAYER_PATH_MAP` dict entry must be patched together in tests
- Alembic: never let two independent `--autogenerate` runs both detect "no tables exist" and generate duplicate full-schema migrations — always run `alembic history` before generating a new migration to confirm a clean, single chain

---

## What Needs to Be Built Next (in order)

### 1. src/backtesting/ ⬅️ IMMEDIATE NEXT
Consumes signals from `data/signals/`, simulates realistic trade execution.

- `backtesting_engine.py` — simulate trades with transaction costs + slippage
- `position_manager.py` — open/close/rebalance position logic
- `portfolio_simulator.py` — cash, invested capital, portfolio value tracking

Then write `tests/backtesting/`.

### 2. src/portfolio/
- `equal_weight.py`
- `volatility_weighted.py`
- `mean_variance.py` — using PyPortfolioOpt

### 3. src/analytics/
- `performance_metrics.py` — Sharpe, Sortino, drawdown, Calmar
- `trade_analytics.py` — win rate, profit factor
- `benchmark_comparison.py` — vs NIFTY 50

### 4. src/dashboard/
- Streamlit app

### 5. src/infra/
- Scheduler for automated daily pipeline runs

### 6. scripts/
- `run_features.py`
- `run_training.py`
- `run_signals.py`
- `run_backtest.py`

---

## Engineering Rules (STRICT) — unchanged, still apply
[... keep the rest of the original file's Engineering Rules and Working Style Rules sections as-is ...]
```

---

Save this over your existing `PROJECT_STATUS.md`. Once you confirm, I'll give you the full spec for `backtesting_engine.py` — the first file in the new layer.