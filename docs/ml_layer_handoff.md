# ML & Signal Layer Handoff Document

This document provides quantitative strategy engineers with complete operational specifications, contracts, signal semantics, and execution procedures for the Machine Learning & Signal Layer.

---

## 1. Pipeline Execution Order and Commands

To produce production signals end-to-end from raw data to actionable signals, execute commands in the following chronological sequence:

1. **Feature Generation**:
   ```bash
   python scripts/run_features.py
   ```
   Computes 26 technical indicators and forward returns across all symbols. Output written to `data/features/equity/<asset_class>_<symbol>_<timeframe>.parquet`.

2. **Model Training & Evaluation**:
   ```bash
   python scripts/run_training.py --model-type xgboost --horizon 1 --embargo-dates 1
   ```
   Fits estimator with leak-free chronological splits and embargo. Registers the trained model in `models/<model_name>/<version>/` and updates `LATEST`.

3. **Walk-Forward Validation (Optional Diagnostics)**:
   ```bash
   python scripts/run_walk_forward.py --model-type xgboost --horizon 1 --n-splits 5 --gap-dates 1
   ```
   Runs multi-fold expanding window cross-validation and reports stability metrics across folds.

4. **Prediction Generation**:
   ```bash
   python scripts/run_prediction.py --model-name xgboost_1d --scope test
   ```
   Generates out-of-sample forward return predictions for test periods. Output written to `data/predictions/equity/equity_<symbol>_daily.parquet`.

5. **Signal Generation**:
   - **Threshold Strategy**:
     ```bash
     python scripts/run_signals.py --strategy threshold --threshold 0.01
     ```
   - **Rank Strategy (Long-Only)**:
     ```bash
     python scripts/run_signals.py --strategy rank --top-n 10
     ```
   - **Top-N / Bottom-N Strategy (Long/Short)**:
     ```bash
     python scripts/run_signals.py --strategy topn --top-n 10 --bottom-n 10
     ```
   Writes signals to `data/signals/equity/equity_<symbol>_daily.parquet`.

---

## 2. Data Contracts & Lake Architecture

### Predictions Schema (`DataLayer.PREDICTIONS`)
- **Location**: `data/predictions/<asset_class>/<asset_class>_<symbol>_<timeframe>.parquet`
- **Columns**:
  - `symbol` (`string`): Asset ticker symbol (e.g. `RELIANCE.NS`)
  - `date` (`datetime64[ns]` / `datetime64[ns, UTC]`): Date of prediction feature timestamp
  - `predicted_return` (`float64`): Expected forward return over the model horizon
  - `model_name` (`string`): Identifying name of the model family (e.g. `xgboost_1d`)
  - `model_version` (`string`): Version timestamp identifier (e.g. `20261025T143000Z`)

### Signals Schema (`DataLayer.SIGNALS`)
- **Location**: `data/signals/<asset_class>/<asset_class>_<symbol>_<timeframe>.parquet`
- **Columns**:
  - `symbol` (`string`): Asset ticker symbol
  - `date` (`datetime64[ns]` / `datetime64[ns, UTC]`): Signal effective timestamp
  - `predicted_return` (`float64`): Model prediction input
  - `signal` (`int64`): Actionable discrete signal:
    - `1`: BUY / Enter or hold long position
    - `0`: HOLD / Cash or flat position
    - `-1`: SELL / Exit long or initiate short position
  - `confidence` (`float64`): Absolute magnitude of predicted return `|predicted_return|`
  - `rank` (`int64`): Cross-sectional rank on date (1 = highest predicted return)

> [!NOTE]
> **Timezone Preservation**: Dates in feature, prediction, and signal Parquet files retain their exact original timezone and dtype as generated upstream. They are never coerced or converted.

---

## 3. Signal Semantics

The discrete `signal` column values map directly to trading execution logic:
- `+1` (BUY): Indicates positive expected alpha. Backtesters should open or maintain a long position.
- `-1` (SELL): Indicates negative expected alpha. In a long-only backtester, this triggers exiting any open long position. In a long/short backtester, this triggers opening or maintaining a short position.
- `0` (HOLD): Insufficient edge or intermediate ranking. No new position is entered; existing positions should be closed or kept neutral.
- `confidence`: `|predicted_return|`. Suitable for proportional Kelly or volatility-scaled position sizing.
- `rank`: Integer rank from 1 to N cross-sectionally per date. For rank strategies, 1 is the best candidate.

---

## 4. Backtesting Safety: Out-of-Sample Discipline

> [!IMPORTANT]
> **Backtest Only on Out-of-Sample Dates!**
> Always use `--scope test` when generating predictions for backtesting. Running predictions with `--scope all` introduces severe in-sample lookahead bias and train-set overfitting, leading to unrealistic backtest Sharpe ratios.

The scope options in `PredictionService` are:
- `test`: Restricts predictions to dates strictly after the model's `val_end_date`. **This is mandatory for backtesting**.
- `oos`: Restricts predictions to dates strictly after `train_end_date` (covers both validation and test).
- `latest`: Generates predictions for only the latest available date per symbol (for live execution).
- `all`: Full history across train, val, and test (only for retrospective diagnostic charting).

---

## 5. Model Registry & Retraining Cadence

### Directory Structure
Trained models and lineage metadata are persisted on disk under `models/`:
```text
models/
└── <model_name>/
    ├── LATEST                   # Text file containing the active champion version string
    └── <version_timestamp>/     # e.g. 20261025T143000Z
        ├── model.joblib         # Serialized model artifact
        └── metadata.json        # Lineage, metrics, feature columns, and split boundaries
```

### Weekly Retraining & Promotion
Run `scripts/run_retraining.py` periodically (e.g. via cron every weekend):
```bash
python scripts/run_retraining.py --model-type xgboost --horizon 1
```
Workflow:
1. Computes Population Stability Index (PSI) drift across features.
2. Checks model age against `settings.MODEL_RETRAIN_INTERVAL_DAYS` (default 7 days).
3. If age >= 7 days or drift is detected, trains a candidate challenger model.
4. Evaluates challenger against the existing champion out-of-sample using Information Coefficient (IC).
5. If challenger IC is not degraded beyond `settings.RETRAIN_PROMOTION_TOLERANCE` (0.01), atomically promotes challenger as the new `LATEST`.

---

## 6. Model Evaluation Metrics

Evaluation metrics computed across test folds:
- **Directional Accuracy**: Fraction of predictions with the correct sign matching actual return direction (> 0.52 is economically significant in daily Indian equities).
- **Information Coefficient (Pearson IC)**: Correlation between predicted and realized returns across the pooled universe.
- **Rank IC (Spearman IC)**: Monotonic ranking correlation, less sensitive to outliers.
- **Daily Rank IC Mean & IC IR**: Daily cross-sectional Spearman correlation averaged across dates, divided by daily IC standard deviation (IC Information Ratio > 0.5 indicates strong consistency).
- **MSE / RMSE / MAE / R²**: Classical regression tracking metrics.

---

## 7. Known Assumptions & Research Limitations

1. **Constituent Survivorship Bias**:
   The current universe is based on the active NIFTY 500 constituent list. Historical features prior to recent constituent inclusions carry mild survivorship bias.
2. **Execution Timing**:
   Features are computed using data up to date $t$ close. Predictions assume entry at close of date $t$ or open of date $t+1$. Slippage and transaction costs must be applied by the backtesting engine.
3. **Cross-Sectional Pooling**:
   Models are trained pooled across all equities rather than per-symbol, maximizing sample size and preventing overfitting on single stocks.
4. **Architectural Scope**:
   Current models encompass linear regressions, tree ensembles (Random Forest, Gradient Boosting), and gradient boosted trees (XGBoost, LightGBM). Deep learning and RL extensions can be registered via `register_model()`.

---

## 8. Downstream Dependencies

The Machine Learning & Signal Layer is completely decoupled and self-contained:
- It requires no imports or services from `src/backtesting/`, `src/portfolio/`, `src/analytics/`, or `src/dashboard/`.
- It consumes data exclusively from `data/features/` via `DatasetBuilder` or `ParquetWriter`.
- It publishes complete, verified Parquet outputs into `data/predictions/` and `data/signals/`.
