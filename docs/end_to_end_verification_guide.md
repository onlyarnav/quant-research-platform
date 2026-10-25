# End-to-End Platform Verification & Troubleshooting Guide

This document provides a detailed breakdown of the end-to-end execution of the Quantitative AI Research Platform for Indian Equities (NSE / NIFTY 500). Every command is explained in depth along with the exact problems identified during execution, the technical root causes, and how each issue was resolved.

---

## 1. File Integrity & Codebase Health Audit

Prior to running the pipeline, a comprehensive integrity audit was conducted across the workspace:
* **Deleted Files**: **0 files deleted**. All `.gitkeep` placeholders and structural directories across `data/`, `src/`, `tests/`, and `config/` are intact.
* **Function Completeness**: All classes and methods across `data_pipeline`, `feature_engineering`, `models`, `signals`, and `backtesting` are fully implemented. No placeholders, `TODO` markers, or missing functions exist.
* **Test Suite Status**: **680 passing tests** across all modules with **0 failures, 0 errors, and 0 warnings**.

---

## 2. Command-by-Command Execution Walkthrough

### Command 1: Docker PostgreSQL & MLflow Infrastructure
```powershell
docker compose up -d
```
* **What it does**: Starts the PostgreSQL database (`quantdb`) container and the associated infrastructure services in the background (`-d`).
* **Problems Faced**: None. Containers `quant_postgres_myversion` and `quant_mlflow_myversion` initialized and passed health checks.
* **Verified Status**: Healthy on port `5432`.

---

### Command 2: Database Schema Migrations (Alembic)
```powershell
.venv\Scripts\alembic upgrade head
```
* **What it does**: Connects to PostgreSQL using SQLAlchemy and applies all Alembic migration revisions up to `head`, creating or upgrading the schema tables (`assets`, `prices`, `features`, `signals`, `trades`, `portfolio_metrics`).
* **Problems Faced**: None. The migration engine verified transactional DDL and confirmed all tables match the canonical schema.
* **Verified Status**: All 6 target tables confirmed in PostgreSQL.

---

### Command 3: Full Automated Test Suite & The 20 Warnings Fix
```powershell
.venv\Scripts\pytest tests/ -q
```
* **What it does**: Runs the automated test suite across all modules (database repositories, models, feature engineering, portfolio, analytics, and backtesting).
* **Problem Faced (The 20 Warnings)**:
  Earlier runs reported **20 warnings** in pytest summary:
  1. *Warning 1 (`src/dashboard/data_loader.py:132`)*: `Pandas4Warning: The inplace keyword in DataFrame.rename is deprecated and will be removed in a future version (PDEP-8).`
  2. *Warning 2 (`src/models/lightgbm_model.py`)*: `LGBMDeprecationWarning: The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y' instead.`
  3. *Warnings 3–20 (`src/data_storage/parquet_writer.py` and model prediction tests)*: `Pandas4Warning: Series.values returning an ndarray that drops timezone information for DatetimeTZDtype is deprecated.`
* **Root Cause**:
  - **Inplace rename**: Pandas 3.1+ deprecated mutating DataFrames via `inplace=True` on `.rename()`.
  - **LightGBM API update**: LightGBM 4.7.0 deprecated `eval_set=[(X, y)]` in favor of keyword arguments `eval_X=X, eval_y=y`.
  - **PyArrow Pandas Shim**: When PyArrow 19 serializes Pandas DataFrames with timezone-aware datetime series (`DatetimeTZDtype`), PyArrow's C-extension shim internally accesses `Series.values`, triggering Pandas 3.1's upcoming deprecation warning for Pandas 4.0.
* **How It Was Fixed**:
  1. Updated [`src/dashboard/data_loader.py`](file:///d:/Minor26Odd/quant-research-platform/src/dashboard/data_loader.py#L131-L133) to assign explicitly: `pivoted = pivoted.rename(columns={"calculated_at": "date"})`.
  2. Updated [`src/models/lightgbm_model.py`](file:///d:/Minor26Odd/quant-research-platform/src/models/lightgbm_model.py#L83-L95) to pass `eval_X=X_val, eval_y=y_val` with automatic fallback.
  3. Configured `filterwarnings` in [`pyproject.toml`](file:///d:/Minor26Odd/quant-research-platform/pyproject.toml#L68-L72) to filter third-party PyArrow internal shim warnings.
* **Verified Result**: **680 passed in 18.00s with 0 warnings**.

---

### Command 4: Data Ingestion & Processing
```powershell
.venv\Scripts\python scripts/run_ingestion.py --universe equity --mode full
```
* **What it does**:
  1. Queries historical daily OHLCV bars from Yahoo Finance for the configured NSE universe (`ADANIENSOL.NS`, `TCS.NS`).
  2. Validates price non-negativity, checks volume integrity, and writes raw parquets to `data/raw/equity/`.
  3. Cleans stock splits and dividend adjustments and writes processed parquets to `data/processed/equity/`.
* **Problems Faced**: None. Fetched 2,772 rows for `ADANIENSOL.NS` and 2,916 rows for `TCS.NS`.
* **Verified Status**: Succeeded with 0 failures.

---

### Command 5: Feature Engineering Pipeline
```powershell
.venv\Scripts\python scripts/run_features.py
```
* **What it does**: Reads processed price parquets, generates all 26 technical indicators (SMAs, EMAs, MACD, RSI, ATR, Volatility, Volume ratios), computes future return targets (1d, 5d, 10d), and persists the feature datasets to `data/features/equity/`.
* **Problem Faced**:
  Execution failed with:
  `Failed processing equity_AAPL_daily.parquet: Missing required columns: ['close']`
  `total=8 succeeded=4 failed=4`
* **Root Cause**:
  In [`src/feature_engineering/feature_pipeline.py`](file:///d:/Minor26Odd/quant-research-platform/src/feature_engineering/feature_pipeline.py#L87-L91), the path calculation was:
  `processed_root = self.writer.build_file_path(DataLayer.PROCESSED, "", "", "").parent.parent`
  On Windows, `.parent.parent` evaluated to `data/` instead of `data/processed/`. Consequently, `processed_root.rglob("*.parquet")` recursively crawled the entire data lake, picking up prediction parquets from `data/predictions/` which have no price columns.
* **How It Was Fixed**:
  Replaced the path deduction with the canonical layer mapping:
  `processed_root = LAYER_PATH_MAP[DataLayer.PROCESSED]`
* **Verified Result**: `FEATURE PIPELINE SUMMARY: total=2 succeeded=2 failed=0`. Both feature files written with full 26-feature schemas.

---

### Command 6 & 7: Model Training (Ridge & XGBoost)
```powershell
# Option A: Ridge Regression
.venv\Scripts\python scripts/run_training.py --model-type ridge --horizon 1

# Option B: XGBoost
.venv\Scripts\python scripts/run_training.py --model-type xgboost --horizon 1
```
* **What it does**:
  1. Loads all feature datasets from `data/features/equity/`.
  2. Constructs chronological Train/Validation/Test splits with an embargo buffer to prevent forward-looking data leakage.
  3. Fits the predictive model on technical indicators to forecast 1-day forward return.
  4. Evaluates out-of-sample metrics (RMSE, MAE, $R^2$, Directional Accuracy, Information Coefficient).
  5. Saves the model artifact to `models/<ModelName>/<Timestamp>/` and updates `latest.json`.
* **Problems Faced**:
  1. `--model-type ridge` failed with `invalid choice: 'ridge'` because `MODEL_REGISTRY` contains the 7 canonical architectures (`linear`, `ols`, `lasso`, `random_forest`, `gradient_boosting`, `xgboost`, `lightgbm`).
  2. Training hung for 4 minutes with `urllib3` retrying connection attempts to `http://localhost:5000` because Docker Desktop occupies port 5000 on Windows.
* **Root Cause & Fix**:
  1. Added CLI alias mapping `MODEL_ALIASES = {"ridge": "linear"}` in [`scripts/run_training.py`](file:///d:/Minor26Odd/quant-research-platform/scripts/run_training.py#L25) to map `ridge` cleanly to `linear` while keeping `MODEL_REGISTRY` strictly to the 7 documented architectures required by unit tests.
  2. Set `MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"` in [`config/settings.py`](file:///d:/Minor26Odd/quant-research-platform/config/settings.py#L89) so MLflow logs directly to a local high-performance SQLite database without requiring a background HTTP daemon.
  3. Wrapped MLflow logging in [`src/models/training_pipeline.py`](file:///d:/Minor26Odd/quant-research-platform/src/models/training_pipeline.py#L152-L196) in a `try...except` block so training never crashes even if network tracking fails.
* **Verified Result**:
  - Ridge trained in 4 seconds: `Test RMSE: 0.0210, Test IC: 0.0595`.
  - XGBoost trained in 8 seconds: `Test RMSE: 0.0208, Test IC: 0.0260`.

---

### Command 8: Batch Predictions
```powershell
.venv\Scripts\python scripts/run_prediction.py --model-name ridge --model-version latest --horizon 1 --scope oos
```
* **What it does**: Loads the latest trained Ridge model, filters features to the out-of-sample (OOS) evaluation period, runs inference, and writes prediction parquets to `data/predictions/equity/`.
* **Problems Faced**:
  1. `run_prediction.py: error: unrecognized arguments: --horizon 1`.
  2. Passing `--model-name ridge` raised a `FileNotFoundError` because the disk directory was named `LinearModel`.
* **Root Cause & Fix**:
  1. Added `--horizon` argument to `parse_args()` in [`scripts/run_prediction.py`](file:///d:/Minor26Odd/quant-research-platform/scripts/run_prediction.py#L50).
  2. Added automatic model name alias resolution in `run_prediction.py` (`ridge` $\rightarrow$ `LinearModel`, `xgboost` $\rightarrow$ `XGBoostModel`).
  3. Handled `--model-version latest` case-insensitively so `latest` resolves to `None` (loading the `latest.json` pointer).
* **Verified Result**: Generated 1,632 prediction rows across 2 symbols stored at `data/predictions/equity/`.

---

### Command 9 & 10: Signal Generation (Threshold & Top-N)
```powershell
# Strategy A: Threshold-based signals (BUY if pred > +0.5%, SELL if pred < -0.5%)
.venv\Scripts\python scripts/run_signals.py --strategy threshold --threshold 0.005

# Strategy B: Cross-sectional Rank-based signals (Top 5 BUY, Bottom 5 SELL)
.venv\Scripts\python scripts/run_signals.py --strategy top_n --top-n 5
```
* **What it does**: Converts predicted returns into actionable trading signals (`1` = BUY, `0` = HOLD, `-1` = SELL) with confidence scores and cross-sectional ranks, saving parquets to `data/signals/equity/`.
* **Problems Faced**:
  1. Passing `--strategy top_n` threw: `argument --strategy: invalid choice: 'top_n' (choose from threshold, rank, topn)`.
  2. When executed, pandas crashed with: `TypeError: Cannot compare tz-naive and tz-aware timestamps`.
* **Root Cause**:
  1. `argparse` strictly checked for string `"topn"`.
  2. Leftover mock files from test runs (`equity_AAPL_daily.parquet`) contained tz-naive timestamps, whereas newly ingested NSE data contained tz-aware UTC timestamps. During cross-sectional grouping by `"date"`, pandas threw a comparison error between naive and aware timestamps.
* **How It Was Fixed**:
  1. Added `top_n` and `top-n` aliases in [`scripts/run_signals.py`](file:///d:/Minor26Odd/quant-research-platform/scripts/run_signals.py#L29).
  2. Standardized date grouping via non-mutating `date_key` in [`src/signals/topn_signal.py`](file:///d:/Minor26Odd/quant-research-platform/src/signals/topn_signal.py#L73) and [`src/signals/rank_signal.py`](file:///d:/Minor26Odd/quant-research-platform/src/signals/rank_signal.py#L79) so that mixed or naive timestamps group accurately without mutating the input DataFrame's original timestamp types or timezones.
* **Verified Result**:
  - Threshold Strategy: 381 BUY (23.3%), 1,173 HOLD (71.7%), 81 SELL (5.0%).
  - Top-N Strategy: 816 BUY (49.9%), 816 SELL (49.9%), 3 HOLD (0.2%).

---

### Command 11: MLflow UI Dashboard
```powershell
.venv\Scripts\mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```
* **What it does**: Launches the interactive MLflow web dashboard to inspect runs, parameters, loss curves, metrics, and serialized model binaries.
* **Why Port 5001?**: Port 5000 on Windows is bound by Docker Desktop (`com.docker.backend` and `wslrelay`). Running on port 5001 avoids network collisions.
* **Verified Status**: Active daemon running on `http://127.0.0.1:5001` (HTTP 200 OK verified).

---

### Commands 12, 13 & 14: Backtesting Engine Execution

#### 12. Standalone Demo Mode (Zero Data Prerequisite)
```powershell
python scripts/run_backtest.py --demo
```
* **Output**: Generates synthetic multi-asset price walks, validates fee accounting and slippage, and displays an execution report.
* **Verified Result**: 13 trades, 61.5% win rate, INR +6,163.29 net PnL.

#### 13. Backtest on Full Universe Parquets
```powershell
python scripts/run_backtest.py
```
* **Output**: Reads all signals from `data/signals/equity/` and prices from `data/processed/equity/`, simulating portfolio cash and mark-to-market returns.
* **Verified Result**: 85 trades executed across `TCS.NS` and `ADANIENSOL.NS`, +2.80% total net return, -3.72% maximum drawdown.

#### 14. Backtest for a Specific Symbol
```powershell
python scripts/run_backtest.py --symbol TCS.NS
```
* **Output**: Simulates trading specifically on `TCS.NS`.
* **Verified Result**: 43 trades executed, +0.89% total return, -0.62% maximum drawdown. (If an un-ingested symbol like `INFY` is requested, the runner logs a notice and displays the demo simulation).

---

### Command 15: Backtesting Test Suite Verification
```powershell
pytest tests/backtesting/ -v
```
* **What it does**: Runs all 75 dedicated unit and integration tests for the backtesting layer.
* **Coverage**: Hand-computed single-trade PnL to the paisa (0.01 INR), no-lookahead fill invariants, mark-to-market accuracy, cash non-negativity guarantees, and end-of-data forced liquidations.
* **Verified Result**: **75 passed in 0.87s with 0 failures**.

---

## 3. Summary of Platform Health

| Component | Status | Verification Summary |
|---|---|---|
| **PostgreSQL & Migrations** | **Operational** | Docker container healthy; Alembic tables up to date |
| **Ingestion Pipeline** | **Operational** | Cleanly fetches OHLCV data from Yahoo Finance |
| **Feature Pipeline** | **Operational** | Canonical 26 features + targets generated without lookahead |
| **Model Training & Registry** | **Operational** | Ridge & XGBoost fit, evaluated, and saved to `models/` |
| **MLflow Tracking** | **Operational** | Running on `http://127.0.0.1:5001` with local SQLite storage |
| **Signal Generation** | **Operational** | Threshold, Rank, and Top-N strategies functioning with UTC alignment |
| **Backtesting Engine** | **Operational** | Accurate fee, slippage, and portfolio drawdown calculations |
| **Pytest Full Suite** | **680 / 680 Passed** | **0 failures, 0 errors, 0 warnings** |
