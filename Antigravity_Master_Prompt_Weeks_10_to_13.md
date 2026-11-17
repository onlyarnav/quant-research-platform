# MASTER PROMPT 1: Build Weeks 10, 11, 12 and 13 in One Run

Attach the repository `quant-research-platform` as the workspace, and also attach `Data_Schema.md`, `System_Architecture.md`, `PROJECT_STATUS.md`, `claude_quant_engineer.md`, `claude_data_engineer.md`, `claude_machine_learning_engineer.md`. **Do not ask questions.** Every decision is made below. If something is ambiguous, apply the "Decision rules" in Part B, record the choice in the final report, and continue.

---

# PART A. ROLE, SCOPE, ENVIRONMENT AND RULES

## A.1 Role and goal

You are the **Lead Quantitative Strategy & Platform Engineer**. Finish the platform from Week 10 to Week 13 so that, after this run, a user can:

1. run a realistic, capital-constrained backtest from ML signals (Week 10),
2. choose portfolio construction methods and constraints (Week 11),
3. get institutional-style performance, trade and benchmark analytics (Week 12),
4. inspect everything in a working **FastAPI backend + Streamlit frontend** dashboard (Week 12),
5. run the daily pipeline, retraining, health checks and the services in Docker with scheduling, documentation, CI and end-to-end tests (Week 13).

The repository already contains a large part of this (earlier work reported Weeks 1–12 partly built and 680 passing tests). The backtesting engine is **incomplete**. Your job is to **audit, then complete**: reuse what is correct, extend what is partial, build what is missing, and never break what passes.

## A.2 Environment (Windows, PowerShell)

- Python 3.12 inside the project's virtual environment. Locate it in this order: `.\.venv\Scripts\python.exe`, `.\venv\Scripts\python.exe`, then `py -3.12`, then `python`. Always run tools as `<python> -m pytest`, `<python> -m black`, etc. If you truly cannot execute Python, stop and report that (do not guess results).
- Use `pathlib` everywhere; never hard-code `/` or `\`. Do not use shell-specific syntax inside Python code. Use `sys.executable` for subprocesses.
- Docker Desktop may be present. **Port 5000 conflicts with Windows**, so MLflow uses a local SQLite backend and UI port 5001 (follow whatever the repository already configures). API port 8000, dashboard port 8501.
- Tests that need PostgreSQL are marked `db` (see F.1). Everything you add must be testable **without Docker and without a database** except tests explicitly marked `db`.

## A.3 Files you may create or modify

`src/backtesting/`, `src/portfolio/`, `src/analytics/`, `src/dashboard/`, `src/infra/`, `scripts/` (new and existing quant/infra scripts), `config/settings.py` (strategy/portfolio/dashboard/pipeline keys only, additive), `tests/` (all folders for the modules above, plus `tests/integration/`, `tests/dashboard/`, `tests/infra/`), `docs/`, `README.md`, `.env.example`, `Dockerfile`, `.dockerignore`, `docker-compose.app.yml` (new), `.github/workflows/ci.yml` (new), `pyproject.toml` (dependencies and pytest markers only).

## A.4 Files you must NOT modify

`src/data_pipeline/`, `src/data_storage/`, `src/database/` (read and use repositories; do not change them), `src/feature_engineering/`, `src/models/`, `src/signals/`, `src/utils/`, `alembic/`, the existing `docker-compose.yml`, and any existing test of those modules. If a bug or missing hook there blocks you, **work around it inside your own module** and list it under "Cross-module issues" in the final report. Never silently edit those modules.

## A.5 Engineering rules (every file you write)

1. `from __future__ import annotations`; module docstring; full type hints; Google-style docstrings (`Args`, `Returns`, `Raises`) explaining the **why** of financial logic.
2. Logging: `from src.utils.logger import get_logger`; `logger = get_logger(__name__)`; lazy `%s` formatting; never f-strings inside `logger.*`; `print` only in `scripts/`.
3. Configuration only through `from config.settings import settings`; no `os.getenv`, no hard-coded paths, URLs, rates or thresholds (CLI defaults come from settings).
4. Validation pattern: empty input → `ValueError("<name> cannot be empty.")`; then missing columns by set difference → `ValueError(f"<name> missing required columns: {missing}")`; invalid arguments → `ValueError` with the offending value in the message.
5. Never mutate inputs. Deterministic behaviour; seeds default to 42.
6. No `TODO`, no placeholder or pseudo-code, no `pass` bodies except abstract/Protocol methods.
7. Max 300 lines per file; split into helper modules.
8. Imports: stdlib → third-party → local, alphabetical, one `from x import a, b` line per module.
9. **Dependency direction (no reverse imports):** `backtesting → portfolio → analytics → dashboard`; `portfolio` may import `backtesting`; `analytics` may import both; `dashboard` may import all three. `backtesting` must NOT import `portfolio` or `analytics`; `portfolio` must NOT import `analytics`. Wiring across layers happens in `src/analytics/backtest_pipeline.py` (E, F) or in `src/infra/`. `src/infra/` may import everything.
10. Style: copy the look of existing files (module docstring layout, blank-line rhythm, `# ====` test banners). Run black (88 cols) and isort (profile black) on everything you touch.
11. Prices: use `adj_close` when present and non-null, else `close`. Defaults: capital INR 1,000,000; cost 0.001 per leg; slippage 0.0005; max position 0.05; risk-free 0.07; 252 trading days.
12. No lookahead: any decision at date t uses only data up to and including t (or up to the decision date when an execution lag is used).

---

# PART B. STEP 0: AUDIT, BASELINE AND DECISION RULES

## B.1 Audit protocol (do this first, print a short inventory)

1. List `src/backtesting`, `src/portfolio`, `src/analytics`, `src/dashboard`, `src/infra`, `scripts`, `tests`, `docs`, `config/settings.py`, `pyproject.toml`, `.env.example`.
2. Read the existing code of every file you will touch or depend on: at minimum `backtesting_engine.py`, `position_manager.py`, `portfolio_simulator.py`, all `src/portfolio/*.py`, all `src/analytics/*.py`, all `src/dashboard/*.py`, `src/infra/*.py`, `scripts/run_backtest.py`, `scripts/run_*.py`, `src/signals/signal_pipeline.py`, `src/models/model_store.py` (if present), `src/data_storage/parquet_writer.py`, `src/database/repository/*.py`.
3. Run the baseline: `<python> -m pytest tests -q -x --ignore=tests/database` first (fast, no DB), then `<python> -m pytest tests -q` if PostgreSQL is reachable. Record counts. **Every test that passes at baseline must still pass at the end, unmodified.**
4. Produce an audit table: for each deliverable in Parts D–H mark `exists and complete`, `exists but partial` (say what is missing), or `missing`. Then build only what the table says is needed.

## B.2 Decision rules (so no questions are needed)

1. **Existing code vs this spec:** if existing code has a different but valid design and its tests pass, keep the existing design and add what is missing in a compatible way. If existing code contradicts this spec in a way that makes the spec's behaviour impossible, create a new class beside it and keep the old class as is (do not delete public classes or tests; consolidation is a later task).
2. **Frozen behaviour of the existing backtesting classes** (their tests pin it): `BacktestingEngine`, `PositionManager`, `PortfolioSimulator`, `Trade`, `PortfolioSnapshot`, `PortfolioSimulationResult` keep their public signatures and defaults; fees in the legacy simulator are charged at exit (both legs); legacy fixed notional = `initial_capital × max_position_size`; legacy execution is same-day close; legacy BUY funding order is alphabetical when signals have no `confidence` column.
   Frozen messages (substrings): `signals_df cannot be empty`, `prices_df cannot be empty`, `signals_df missing required columns`, `prices_df missing required columns`, `No overlapping`, `must be positive`, `already open`, `Insufficient cash`, `No open position`; warning texts containing `Force-closing` and `zero trades`.
3. **Frozen portfolio allocators:** `EqualWeightAllocator.allocate(signals_df)`, `VolatilityWeightedAllocator.allocate(signals_df, prices_df, as_of_date)`, `MeanVarianceAllocator.allocate(signals_df, prices_df, as_of_date)` keep signatures, return dict of symbol → weight summing to 1.0, and messages: `signals_df cannot be empty.`, `prices_df cannot be empty.`, `signals_df missing required columns`, `prices_df missing required columns`, `lookback_window must be at least 2`, warnings `No BUY signals found; returning empty allocation.`, `zero or undefined volatility`, `Excluding symbols with insufficient price history`, `Fewer than 2 eligible symbols`, `Mean-variance optimization failed`. Wrap them with adapters; do not change them.
4. **Canonical schemas are authoritative** (`Data_Schema.md`): trade log = `trade_id, symbol, entry_date, exit_date, entry_price, exit_price, position_size, fees, slippage_cost, pnl, return_pct`; portfolio history = `date, cash, invested_capital, portfolio_value, daily_return, cumulative_return, drawdown`. **Never add columns to these.** Put extra information in separate "details" DataFrames (D.8).
5. **Where results are stored:** backtest runs go to `settings.BACKTEST_RESULTS_PATH` (default `<project>/data/backtests/`), one folder per run, written with pandas/pyarrow directly (not through `ParquetWriter`, which is not yours to extend). Do not add a data-lake tier.
6. **Benchmark data:** look for the index file via `ParquetWriter` conventions (`index_NIFTY50_daily.parquet` in the processed layer). If absent, fall back to the equal-weight buy-and-hold of the backtest universe and log a warning; never fabricate index data. Do not modify the ingestion code.
7. **Signals source:** read from the signals layer via `ParquetWriter` (do not recompute signals). Prices from the processed layer. Dates are used exactly as stored (timezone preserved; if signal and price date dtypes differ in timezone raise `ValueError("Date dtype mismatch between signals and prices: ...")`).
8. **Execution timing:** default `execution_lag = 0` (same-day close, as in the existing, reported results). Implement lag ≥ 1 and a sensitivity helper (D.11); document lag 0 as optimistic.
9. **If an existing test conflicts with a requirement here,** keep the test passing and report the conflict.
10. **If you are blocked** on a sub-task by something outside your scope, implement the best workaround inside your module, add a clearly named test for the workaround, log it in the report, and continue with the rest. Do not loop on the same failure more than three attempts.

---

# PART C. CONFIGURATION, DEPENDENCIES, STRUCTURE

## C.1 New settings (all with defaults; append to `.env.example` too)

```
INITIAL_CAPITAL: float = 1_000_000.0                  # add only if missing
BACKTEST_RESULTS_PATH: Path = _PROJECT_ROOT / "data" / "backtests"
BACKTEST_EXECUTION_LAG: int = 0
BACKTEST_CASH_RESERVE: float = 0.0
BACKTEST_ALLOCATION_METHOD: str = "fixed_notional"
BACKTEST_REBALANCE_FREQUENCY: str = "on_signal"
BACKTEST_REBALANCE_TOLERANCE: float = 0.0025
BACKTEST_STOP_LOSS: float | None = None
BACKTEST_TAKE_PROFIT: float | None = None
BACKTEST_LOOKBACK_WINDOW: int = 60
BENCHMARK_SYMBOL: str = "NIFTY50"
TRADING_DAYS_PER_YEAR: int = 252
ROLLING_WINDOW: int = 63
VAR_CONFIDENCE: float = 0.95
PIPELINE_RUNS_PATH: Path = _PROJECT_ROOT / "data" / "pipeline_runs"
API_HOST: str = "127.0.0.1"
API_PORT: int = 8000
DASHBOARD_API_URL: str | None = None                  # None = dashboard uses in-process services
DASHBOARD_API_KEY: str | None = None                  # None = no auth (local use only)
DASHBOARD_CACHE_TTL_SECONDS: int = 60
```
Validators: `0 ≤ BACKTEST_CASH_RESERVE < 1`; `BACKTEST_EXECUTION_LAG ≥ 0`; allocation method and rebalance frequency in the allowed enumerations; stop-loss and take-profit `None` or in (0, 1]; `ROLLING_WINDOW ≥ 2`; `0 < VAR_CONFIDENCE < 1`; `TRADING_DAYS_PER_YEAR ≥ 1`; `API_PORT` in 1–65535; empty strings for optional values are treated as `None`.

## C.2 Dependencies (`pyproject.toml`)

Ensure present: `fastapi`, `uvicorn[standard]`, `httpx`, `streamlit>=1.35` (AppTest needs ≥ 1.28), `plotly`, `PyPortfolioOpt`, `scipy`, `apscheduler>=3.10`, `pytest-mock`, `pytest-cov` (optional). Register pytest markers `db` (needs PostgreSQL) and `integration` (end-to-end). Do not remove anything.

## C.3 Target file layout (create what is missing)

```
src/backtesting/  backtest_config.py  allocation_interface.py  price_loader.py  signal_loader.py
                  position_book.py  weighted_simulator.py  backtest_store.py  sensitivity.py
                  (existing: backtesting_engine.py position_manager.py portfolio_simulator.py)
src/portfolio/    base.py  adapters.py  signal_weighted.py  risk_parity.py  constraints.py
                  allocator_factory.py  (existing: equal_weight.py volatility_weighted.py mean_variance.py)
src/analytics/    performance_metrics.py  trade_analytics.py  benchmark_comparison.py
                  benchmark_builder.py  rolling_metrics.py  report_builder.py
                  metrics_persistence.py  backtest_pipeline.py
src/dashboard/    config.py  schemas.py  services.py  client.py  charts.py  components.py  app.py
                  api/__init__.py api/main.py api/routes_runs.py api/routes_signals.py
                  api/routes_system.py api/routes_backtests.py api/security.py
                  views/__init__.py views/overview.py views/performance.py views/trades.py
                  views/allocation.py views/benchmark.py views/signals.py views/experiments.py
                  views/pipeline.py views/run_backtest.py
src/infra/        pipeline_models.py  pipeline_steps.py  pipeline_orchestrator.py  scheduler.py
                  health.py  run_history.py
scripts/          run_backtest.py  run_analytics.py  run_daily_pipeline.py  run_scheduler.py
                  run_api.py  run_dashboard.py  health_check.py
docs/             architecture.md  runbook.md  user_guide.md  api_reference.md
                  production_readiness_checklist.md  end_to_end_verification_guide.md
root              Dockerfile  .dockerignore  docker-compose.app.yml  .github/workflows/ci.yml
```
Note: the Streamlit view modules live in `views/` (not `pages/`) so Streamlit does not auto-detect a multipage app next to `app.py`.

---

# PART D. WEEK 10: COMPLETE THE BACKTESTING ENGINE

## D.1 Goals

Keep the legacy classes (fixed-notional, same-day close, long-only). Add a second, fuller simulator that supports: weight-based allocation from any allocator, rebalancing, partial position changes with average-cost accounting, optional shorting, stop-loss/take-profit, execution lag, cash reserve, constraints via allocators, richer outputs, reproducible run storage, and a sensitivity helper. A legacy-compatible configuration must give the same closed-trade PnL in both simulators (parity test).

## D.2 `backtest_config.py`

```python
class AllocationMethod(str, Enum): FIXED_NOTIONAL, EQUAL_WEIGHT, SIGNAL_WEIGHTED, VOLATILITY_WEIGHTED, RISK_PARITY, MEAN_VARIANCE   # values: "fixed_notional", "equal_weight", ...
class RebalanceFrequency(str, Enum): ON_SIGNAL, DAILY, WEEKLY, MONTHLY                                                      # values lowercase

@dataclass(frozen=True, slots=True)
class BacktestConfig:
    strategy_name: str = "strategy"
    initial_capital: float | None = None          # None → settings.INITIAL_CAPITAL
    transaction_cost: float | None = None         # None → settings.TRANSACTION_COST
    slippage: float | None = None                 # None → settings.SLIPPAGE
    max_position_size: float | None = None        # None → settings.MAX_POSITION_SIZE
    cash_reserve: float | None = None             # None → settings.BACKTEST_CASH_RESERVE
    allocation_method: AllocationMethod | None = None      # None → settings
    rebalance_frequency: RebalanceFrequency | None = None  # None → settings
    rebalance_tolerance: float | None = None
    allow_short: bool = False
    stop_loss_pct: float | None = None            # None → settings (which may be None = disabled)
    take_profit_pct: float | None = None
    execution_lag: int | None = None
    max_open_positions: int | None = None
    min_trade_value: float = 0.0
    lookback_window: int | None = None
    force_close_at_end: bool = True
    risk_free_rate: float | None = None
    def resolved(self) -> "BacktestConfig"        # all Nones replaced by settings defaults
    def validate(self) -> None
    def to_dict(self) -> dict[str, Any]           # JSON-serialisable (enums as strings)
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BacktestConfig"
    def config_hash(self) -> str                  # sha256 of canonical JSON of resolved().to_dict(), first 8 hex chars
    def is_legacy_compatible(self) -> bool        # fixed_notional, on_signal, no shorting, no stops, lag 0, cash_reserve 0, max_open_positions None, force_close_at_end True
```
`validate()` rules (each `ValueError` with the value in the message): capital > 0; cost, slippage in [0, 0.5]; `0 < max_position_size ≤ 1`; `0 ≤ cash_reserve < 1`; tolerance ≥ 0; stops in (0, 1] or None; lag ≥ 0; `max_open_positions` ≥ 1 or None; `lookback_window ≥ 2`; if `allow_short` then `allocation_method` must be `FIXED_NOTIONAL` or `EQUAL_WEIGHT` (message: `allow_short is only supported with fixed_notional or equal_weight allocation`).

## D.3 `allocation_interface.py` (no imports from `portfolio`)

```python
@dataclass(slots=True)
class AllocationContext:
    as_of_date: pd.Timestamp
    signals: pd.DataFrame            # rows for the candidate symbols on/at decision date: symbol, signal, confidence, predicted_return, rank (missing optional columns filled with NaN)
    prices_history: pd.DataFrame     # symbol, date, close[, adj_close]; only dates <= as_of_date
    current_weights: dict[str, float]
    portfolio_value: float
    side: str = "long"               # "long" | "short"

@dataclass(slots=True)
class AllocationResult:
    weights: dict[str, float]        # fractions of TOTAL portfolio value, non-negative magnitudes, sum <= 1 - cash_reserve
    cash_weight: float
    excluded: dict[str, str]         # symbol → reason
    method: str

class Allocator(Protocol):
    def allocate(self, context: AllocationContext) -> AllocationResult: ...
```

## D.4 `price_loader.py` and `signal_loader.py`

```python
class PriceLoader:
    def __init__(self, writer: ParquetWriter | None = None) -> None
    def discover_symbols(self, asset_class: str = "equity", timeframe: str = "daily") -> list[str]
    def load(self, symbols: Sequence[str] | None = None, asset_class: str = "equity", timeframe: str = "daily",
             start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None) -> pd.DataFrame
class SignalLoader:  # same constructor; discover_symbols reads the SIGNALS layer
    def load(self, symbols=None, asset_class="equity", timeframe="daily", start=None, end=None) -> pd.DataFrame
```
- Discovery: use `ParquetWriter.build_file_path(layer, asset_class, "_", timeframe).parent`, glob `f"{asset_class}_*_{timeframe}.parquet"`, symbol = `stem.split("_", 1)[1].rsplit("_", 1)[0]`; sorted; `[]` if the folder is missing.
- `load`: read each symbol via `writer.read`; missing symbol file → warning and skip; none found → `ValueError("No <prices|signals> files found for the requested symbols.")`; concatenate; keep columns (`symbol, date, close, adj_close?, volume?` for prices; all signal columns for signals); filter `start ≤ date ≤ end` (use `align_timestamp`-style tz alignment: if the series is tz-aware and the bound is naive, localise the bound; if not aware and the bound is aware, convert to naive); sort by `symbol, date`; reset index.
- Signal validation (`SignalLoader`): required `symbol, date, predicted_return, signal`; `signal` values only in {−1, 0, 1} else `ValueError("signal must be in {-1, 0, 1}")`; duplicate `(symbol, date)` → `ValueError("Duplicate (symbol, date) rows in signals")`; NaN in symbol/date/signal → `ValueError`.
- Price validation: required `symbol, date, close`; non-positive execution prices are allowed through here (the simulator rejects them with the frozen message).

## D.5 `position_book.py`: average-cost position accounting

```python
@dataclass(slots=True)
class BookPosition:
    symbol: str; side: int                 # +1 long, -1 short
    quantity: float                        # always positive
    avg_entry_price: float                 # executed price basis
    entry_date: pd.Timestamp               # date of the first fill
    entry_fees_paid: float; entry_slippage_paid: float

class PositionBook:
    def __init__(self) -> None
    def has(self, symbol: str) -> bool
    def get(self, symbol: str) -> BookPosition | None
    def symbols(self, side: int | None = None) -> list[str]
    def add(self, symbol, side, date, quantity, executed_price, fee, slippage_cost) -> BookPosition   # new or average-up/down; side mismatch raises ValueError("Cannot add to a position of the opposite side")
    def reduce(self, symbol, quantity, ...) -> ReducedLot                                             # returns the closed lot's accounting; closes fully when remaining < 1e-9
    def market_value(self, prices: Mapping[str, float], side: int | None = None) -> float
    def unrealized_pnl(self, prices: Mapping[str, float]) -> float
    def exposures(self, prices: Mapping[str, float], equity: float) -> dict[str, float]               # long, short, gross, net as fractions of equity
```
`ReducedLot` carries `entry_date, avg_entry_price, quantity, entry_fee_share, entry_slippage_share`. Quantities are fractional (as in the legacy engine; document it).

## D.6 Fill and accounting formulas (single source, put them in a pure helper module or at the top of `weighted_simulator.py`)

- Executed price: buys (open long, cover short) `p_exec = p × (1 + slippage)`; sells (close long, open short) `p_exec = p × (1 − slippage)`.
- Fee per fill: `fee = transaction_cost × p_exec × quantity`, charged to cash at the fill.
- Long open/add: `cash −= p_exec·q + fee`; new average = `(q_old·avg_old + q·p_exec)/(q_old + q)`; accumulate `entry_fees_paid += fee`, `entry_slippage_paid += (p_exec − p)·q`.
- Long reduce/close: `cash += p_exec·q − fee`; fraction `f = q / q_before`; `gross = (p_exec − avg)·q`; trade `fees = f·entry_fees_paid + fee`; `slippage_cost = f·entry_slippage_paid + (p − p_exec)·q`; `pnl = gross − fees`; `return_pct = pnl / (avg·q)`.
- Short open/add: `cash += p_exec·q − fee`. Short reduce/cover: `cash −= p_exec·q + fee`; `gross = (avg − p_exec)·q`; fees and slippage as above (cover `p_exec = p(1+s)`, so slippage cost `(p_exec − p)·q`).
- Equity = `cash + Σ_long q·price − Σ_short q·price`. `invested_capital` in the snapshot = long market value + short market value (gross), `portfolio_value = equity`.
- Position size from weight: `target_notional = weight × equity`; `target_qty = target_notional / price` (price = current execution reference price before slippage); trade quantity = `target_qty − current_qty`.

## D.7 `weighted_simulator.py`: `WeightedPortfolioSimulator`

```python
class WeightedPortfolioSimulator:
    def __init__(self, config: BacktestConfig, allocator: Allocator | None = None) -> None   # config is validated and resolved; allocator None only allowed for FIXED_NOTIONAL
    def run(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> SimulationOutput
@dataclass(slots=True)
class SimulationOutput:
    trades: list[Trade]; trades_df: pd.DataFrame                  # canonical 11 columns
    trade_details_df: pd.DataFrame                                 # trade_id, side, exit_reason, gross_pnl, holding_days
    portfolio_history_df: pd.DataFrame                             # canonical 7 columns
    daily_details_df: pd.DataFrame                                 # date, gross_exposure, net_exposure, long_exposure, short_exposure, n_positions, turnover, realized_pnl_cum, unrealized_pnl, fees_cum, slippage_cum
    positions_history_df: pd.DataFrame                             # date, symbol, side, quantity, price, market_value, weight, unrealized_pnl
    summary: dict[str, float]                                      # n_trades, total_fees, total_slippage, avg_turnover_annualised, final_equity
```
Reuse `Trade` from `backtesting_engine.py` and the validation, execution-price and merge helpers of the legacy simulator (import or small shared helpers; do not copy-paste big blocks).

**Validation:** same frozen messages and order as the legacy simulator (empty signals, empty prices, missing columns, `No overlapping (symbol, date) rows ...`). `config.validate()` first.

**Calendar and lag:** `calendar` = sorted unique price dates. With `lag = L`, a signal on date `t` is executed on `calendar[idx(t) + L]` (signals whose execution date does not exist are dropped with an INFO log). The **decision date** for allocators is the original signal date `t` (never later). With `L = 0` decision date equals execution date.

**Symbol state machine (per symbol), updated from effective signals:**
- `+1` → state LONG; `−1` → state FLAT, or SHORT if `allow_short`; `0` → state unchanged.
- A symbol stopped out (see below) becomes FLAT and stays FLAT until a new `+1` (or `−1` when shorting) effective signal arrives after the stop date.
- Symbols with no price on a date keep `last_known_price`; a symbol never seen in prices cannot be traded.

**Daily loop (in this exact order):**
1. Update `last_known_price` for every symbol priced that day.
2. **Stops:** for each open position compute `ret = price/avg_entry − 1` (long) or `avg_entry/price − 1`… for shorts use `ret = −(price/avg_entry − 1)`; if `stop_loss_pct` is set and `ret ≤ −stop_loss_pct` → close fully at today's price (reason `stop_loss`); if `take_profit_pct` is set and `ret ≥ take_profit_pct` → close fully (`take_profit`). Stops are checked on closing prices only (no intraday data); document this.
3. Apply effective signals of the day to the state machine.
4. Determine whether to **rebalance** today: always if the set of LONG (or SHORT) symbols changed, or if today is a rebalance date for the frequency: `daily` every date; `weekly` first calendar date of each ISO week; `monthly` first calendar date of each month; `on_signal` only the set-change case.
5. If rebalancing:
   a. **Exits first:** close fully every held symbol not in its target set (`rebalance_exit` or `signal` when caused by a signal).
   b. **Targets:** `FIXED_NOTIONAL`: each target symbol's notional is `initial_capital × max_position_size` and it is **not resized** after entry (legacy behaviour; only entries/exits). Other methods: build an `AllocationContext` (candidates = target-set symbols with `confidence` from the latest signal row; `prices_history` = price rows with `date ≤ decision date`; `current_weights`; `portfolio_value = equity`; `side`) and call the allocator; weights are fractions of total equity already constrained by the allocator. For `allow_short` + `EQUAL_WEIGHT`: long and short legs each get half of the gross budget `1 − cash_reserve`, each leg split equally, each position additionally capped by `max_position_size`.
   c. Apply `max_open_positions`: keep the highest-priority symbols only (priority below).
   d. **Priority order** (deterministic): `confidence` descending (NaN last), then `rank` ascending (NaN last), then symbol ascending. Used for funding and truncation.
   e. **Reductions first** (sells of existing longs / covers of existing shorts to reach targets), then **additions/entries** in priority order. Skip a resize when `|trade notional| < max(min_trade_value, rebalance_tolerance × equity)` (entries and full exits are never skipped).
   f. **Cash rule:** buys may use at most `cash − cash_reserve × equity`; if the cash is insufficient the last affordable order is partially filled (quantity reduced), later orders are skipped (DEBUG log). Cash must never go negative; assert this in code.
6. Compute equity, exposures, realised and cumulative PnL, turnover (`traded notional / equity`), then append snapshot rows: `daily_return = equity/prev_equity − 1` (0 on the first date), `cumulative_return = equity/initial_capital − 1`, `running_max`, `drawdown = (equity − running_max)/running_max`.
7. **Final date:** if `force_close_at_end`, close all remaining positions at the last price (reason `forced_close_end`, WARNING log containing `Force-closing`) **before** the last snapshot, so the final equity includes exit costs.

Trade log: one `Trade` per closing fill (partial or full), sequential IDs; `entry_date` is the position's first entry date, `exit_date` the fill date. `trade_details_df.holding_days = (exit_date − entry_date).days`. If no trade happened log a WARNING containing `zero trades` (same text as the legacy engine) and return empty frames with the canonical columns.

## D.8 Outputs and schemas

`trades_df` and `portfolio_history_df` columns and dtypes exactly as in Data_Schema (dates as in the input). Empty trade logs still carry the 11 column names. The details frames are additional and never persisted to the database.

## D.9 `backtest_store.py`: reproducible run storage

```python
@dataclass(slots=True)
class RunRecord:
    run_id: str; strategy_name: str; created_at: str; config: dict[str, Any]
    n_trades: int; final_equity: float; start_date: str | None; end_date: str | None; extras: dict[str, Any]

class BacktestStore:
    def __init__(self, root: Path | None = None) -> None            # default settings.BACKTEST_RESULTS_PATH
    @staticmethod
    def new_run_id(strategy_name: str, config_hash: str, now: datetime | None = None) -> str   # f"{YYYYMMDDTHHMMSSZ}_{slug}_{hash}"
    def save_run(self, run_id: str, config: BacktestConfig, output: SimulationOutput, extras: dict | None = None) -> Path
    def load_run(self, run_id: str) -> tuple[RunRecord, SimulationOutput]
    def list_runs(self) -> list[RunRecord]                           # newest first
    def latest_run_id(self) -> str | None
    def delete_run(self, run_id: str) -> None
    def save_json(self, run_id: str, name: str, payload: dict[str, Any]) -> Path      # metrics.json, report.json etc.
    def load_json(self, run_id: str, name: str) -> dict[str, Any]
```
Folder layout `<root>/<run_id>/`: `config.json`, `record.json`, `trades.parquet`, `trade_details.parquet`, `portfolio_history.parquet`, `daily_details.parquet`, `positions_history.parquet`. `run_id` and names must match `^[A-Za-z0-9_.-]+$` (path-traversal protection). Writes are atomic per file (temp file then `os.replace`). Missing run → `FileNotFoundError(f"Backtest run '<id>' not found")`. NaN/inf in JSON become `null`.

## D.10 Legacy classes: small, compatible additions only

Optionally add to `BacktestingEngine` the arguments `stop_loss_pct` and `take_profit_pct` (default `None`, close-price based exit reason handled inside `_simulate_symbol`) **only if** this does not disturb existing tests; otherwise skip, since stops are fully supported by the weighted simulator.

## D.11 `sensitivity.py`

```python
def run_sensitivity(run_fn: Callable[[BacktestConfig], SimulationOutput], base: BacktestConfig,
                    cost_grid: Sequence[float] = (0.0, 0.0005, 0.001, 0.002),
                    slippage_grid: Sequence[float] = (0.0, 0.0005, 0.001),
                    lag_grid: Sequence[int] = (0, 1)) -> pd.DataFrame
```
Returns one row per combination with `transaction_cost, slippage, execution_lag, final_equity, total_return, max_drawdown, n_trades`. (Max drawdown from the history min of `drawdown`.) Purpose: show how fragile results are to costs and execution timing. Pure function; no I/O.

---

# PART E. WEEK 11: PORTFOLIO CONSTRUCTION

All allocators return weights as **fractions of total portfolio value** (magnitudes), sum ≤ `1 − cash_reserve`, through the unified `Allocator` protocol of D.3. Existing allocators are wrapped, not changed.

## E.1 `base.py`

`AllocatorBase` implementing `allocate(context)` as a template method: (1) validate context (non-empty candidates, required columns); (2) `raw = self._raw_weights(context)` (dict summing to 1.0 over eligible symbols; subclasses implement); (3) `ConstraintResult = self.constraints.apply(raw, ...)`; (4) return `AllocationResult` with `excluded` reasons and `cash_weight = 1 − Σ weights`. Constructor: `AllocatorBase(constraints: PortfolioConstraints | None = None)`; default constraints from settings (`max_position_size`, `cash_reserve`).

## E.2 `constraints.py`: `PortfolioConstraints`

```python
@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    max_position_size: float = settings.MAX_POSITION_SIZE
    min_position_weight: float = 0.0
    cash_reserve: float = 0.0
    sector_map: Mapping[str, str] = field(default_factory=dict)
    max_sector_weight: float | None = None
    asset_class_map: Mapping[str, str] = field(default_factory=dict)
    max_asset_class_weight: float | None = None
    def validate(self) -> None
    def apply(self, weights: Mapping[str, float]) -> ConstraintResult
@dataclass(slots=True)
class ConstraintResult: weights: dict[str, float]; cash_weight: float; binding: list[str]
```
`apply` algorithm (deterministic, converges): (1) investable = `1 − cash_reserve`; scale weights to sum to investable. (2) Repeat up to 100 iterations: cap each weight at `max_position_size`; scale down members of any sector whose sum exceeds `max_sector_weight` (proportionally), likewise asset classes; collect the excess; redistribute the excess pro rata among symbols with remaining room (below every applicable cap); stop when no excess remains or no symbol has room. (3) Drop weights below `min_position_weight` (set to 0) and redistribute once the same way. (4) Leftover (infeasible budget, e.g. fewer names than `investable / max_position_size`) goes to `cash_weight`; record `binding` entries such as `"max_position_size:RELIANCE.NS"`, `"max_sector_weight:Energy"`. Output weights are non-negative, each ≤ cap, sector sums ≤ cap, total + cash = 1 (within 1e-9). Unknown sector symbols are grouped under `"UNKNOWN"` and not capped unless a cap is explicitly given for `"UNKNOWN"`.

## E.3 `adapters.py`

`LegacyAllocatorAdapter(name, legacy, needs_prices: bool, constraints)`: builds the legacy call from the context (`signals_df` with `symbol`, `signal = 1`; `prices_df` = `context.prices_history`; `as_of_date`), gets the legacy dict (weights sum to 1 or `{}`), and passes it through the constraints. Legacy exceptions are not swallowed except `ValueError` for empty input, which becomes an empty `AllocationResult` with `excluded` reasons. For `side == "short"` it computes weights the same way and returns magnitudes (the simulator applies the sign).

## E.4 New allocators

- **`EqualWeightAllocator2`** is not needed; use `LegacyAllocatorAdapter` for equal weight.
- **`SignalWeightedAllocator(constraints=None, basis="confidence")`** in `signal_weighted.py`: raw weight ∝ `confidence` (basis `"confidence"`) or ∝ `max(predicted_return, 0)` (basis `"predicted_return"`); NaN or non-positive basis values get zero weight and are reported in `excluded`; if all are zero fall back to equal weight (log WARNING `All signal weights are zero; falling back to equal weighting`).
- **`RiskParityAllocator(lookback_window=60, constraints=None, max_iter=200)`** in `risk_parity.py`: equal risk contribution. Use returns over `lookback_window` trailing prices (≥ `lookback_window + 1` prices required, else symbol excluded with reason `insufficient history`). Covariance = sample covariance of daily returns (ddof=1) plus `1e-10` on the diagonal; solve minimise `Σ_i (w_i (Σw)_i − (wᵀΣw)/n)²` subject to `w ≥ 0`, `Σw = 1` with `scipy.optimize.minimize(method="SLSQP")` starting from inverse-volatility weights. If fewer than 2 eligible symbols → equal weight of the eligible one(s); if the solver fails or returns NaN → fall back to inverse-volatility weights with a WARNING containing `Risk parity optimisation failed`. Constraints applied after.
- **Volatility-weighted** and **mean-variance** come from the legacy adapters.
- Advanced methods (Black-Litterman, maximum diversification, hierarchical risk parity): **architecture only**. The registry (E.5) is the extension point; do not implement them (role file).

## E.5 `allocator_factory.py`

```python
ALLOCATOR_NAMES = ("equal_weight", "signal_weighted", "volatility_weighted", "risk_parity", "mean_variance")
def available_allocators() -> list[str]
def create_allocator(method: str | AllocationMethod, constraints: PortfolioConstraints | None = None,
                     lookback_window: int | None = None, **kwargs) -> Allocator     # "fixed_notional" is not an allocator: raise ValueError("fixed_notional does not use an allocator")
def register_allocator(name: str, factory: Callable[..., Allocator]) -> None           # duplicates raise ValueError
```
Unknown → `ValueError(f"Unknown allocation method '<name>'. Available: [...]")`.

---

# PART F. WEEK 12: PERFORMANCE ANALYTICS

Reuse what exists; add what is missing. Conventions (document them in the module docstrings): daily frequency; `periods_per_year = settings.TRADING_DAYS_PER_YEAR`; `daily_return` column taken from the history (not recomputed); returns series sorted by date; if existing implementations already follow a different, tested annualisation formula, keep it and state it in the docstring.

## F.1 `performance_metrics.py`: `PerformanceMetrics`

Constructor `(risk_free_rate=None, trading_days_per_year=None)` from settings; `ValueError` if `trading_days_per_year <= 0`. Every method takes `portfolio_history_df` with at least `date, portfolio_value, daily_return` (`drawdown` for max drawdown) and validates (empty, missing columns).

| Method | Definition |
|---|---|
| `total_return` | `V_last / V_first − 1` using the first and last `portfolio_value` |
| `annualized_return` | `(1 + total_return)^(periods_per_year / n_intervals) − 1` with `n_intervals = len(df) − 1`; if `n_intervals < 1` → log warning, return `total_return` |
| `annualized_volatility` | `std(daily_return, ddof=1) × sqrt(periods_per_year)`; fewer than 2 returns → NaN and warning |
| `sharpe_ratio` | `(annualized_return − rf) / annualized_volatility`; zero/NaN vol → NaN, warning `Cannot compute Sharpe ratio: annualized volatility is zero or undefined` |
| `sortino_ratio` | `(annualized_return − rf) / downside_deviation`, downside deviation = `sqrt(mean(min(r − rf_daily, 0)²)) × sqrt(periods)` over **all** periods (target = daily risk-free); zero downside → NaN, warning `No downside deviation observed ...` |
| `max_drawdown` | min of the history `drawdown` column (≤ 0) |
| `calmar_ratio` | `annualized_return / abs(max_drawdown)`; zero drawdown → NaN, warning |
| `value_at_risk(confidence=None)` | historical: `quantile(daily_return, 1 − confidence)`; confidence in (0, 1) else `ValueError` |
| `conditional_value_at_risk(confidence=None)` | mean of returns `≤ VaR`; if none, return VaR with a DEBUG log |
| `compute_all()` | dict with keys `total_return, annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, value_at_risk_95, conditional_value_at_risk_95, best_day, worst_day, positive_days_ratio, max_drawdown_duration_days` |

`max_drawdown_duration_days`: longest run of consecutive rows with `drawdown < 0` (calendar days from the peak date to the recovery date or end).

## F.2 `trade_analytics.py`: `TradeAnalytics`

Methods on `trades_df` (requires `pnl, return_pct, entry_date, exit_date`; empty → `ValueError`): `win_rate`, `loss_rate` (breakeven counts as neither), `average_win` and `average_loss` (0.0 with warning when none), `profit_factor` (`inf` with warning when there are no losses; `0.0` when there are no wins), `expectancy = win_rate·avg_win + loss_rate·avg_loss`, `payoff_ratio = avg_win / abs(avg_loss)` (NaN if no losses), `average_trade_duration` (`pd.Timedelta`), `total_trades`, `total_pnl`, `best_trade`, `worst_trade`, `max_consecutive_wins`, `max_consecutive_losses` (ordered by `exit_date`, then `trade_id`), `pnl_by_symbol() -> DataFrame` (symbol, n_trades, total_pnl, win_rate, avg_return_pct), `return_distribution(bins=20) -> DataFrame` (bin_left, bin_right, count), `compute_all() -> dict` (scalar entries only plus `average_trade_duration_days: float`).

## F.3 `benchmark_comparison.py` and `benchmark_builder.py`

`BenchmarkBuilder(writer=None)`:
- `from_index(symbol=None, asset_class="index", timeframe="daily", initial_capital, start, end) -> DataFrame | None`: read the processed index file; build a buy-and-hold history in the canonical 7-column format (`portfolio_value = initial_capital × price/price_first`, `invested_capital = portfolio_value`, `cash = 0`, returns, drawdown); `None` if the file is missing (log WARNING).
- `equal_weight_universe(prices_df, initial_capital, start, end) -> DataFrame`: daily rebalanced equal-weight of all symbols with a price that day (mean of daily returns), canonical format.
- `build(prices_df, initial_capital, dates, prefer_index=True)`: index first, else the equal-weight universe; returns `(history_df, label)`.
- Align by `date` using the strategy dates (inner join); benchmark history reindexed to strategy dates; missing days forward-filled (≤ 5 consecutive, else dropped).

`BenchmarkComparison(performance_metrics=None)`; `compare(strategy_history_df, benchmark_history_df) -> dict`:
- aligned inner join on `date`; empty overlap → `ValueError("No overlapping dates between strategy and benchmark.")`.
- returns: `strategy_metrics`, `benchmark_metrics` (from `compute_all`), plus:
  `excess_return = strategy_annualized − benchmark_annualized`; `beta = cov(rs, rb)/var(rb)` (ddof=1, NaN with warning if benchmark variance is 0); `alpha_jensen = (mean(rs − rf_d) − beta × mean(rb − rf_d)) × periods_per_year` (annualised, rf_d = rf/periods); `tracking_error = std(rs − rb, ddof=1) × sqrt(periods)`; `information_ratio = mean(rs − rb)/std(rs − rb, ddof=1) × sqrt(periods)` (NaN if std 0); `correlation`; `up_capture` = (mean of rs on days rb > 0)/(mean of rb on those days); `down_capture` similarly on days rb < 0; `outperformance = total_return_s − total_return_b`.
- A strategy identical to the benchmark gives `beta = 1`, `alpha_jensen = 0`, `excess_return = 0`, `tracking_error = 0`, `information_ratio = NaN`.

## F.4 `rolling_metrics.py` (pure functions on the history)

`rolling_sharpe(df, window=None)`, `rolling_volatility(df, window=None)` (annualised), `drawdown_series(df)` (date, drawdown, underwater flag), `equity_series(df)` (date, portfolio_value, cumulative_return), `monthly_returns_table(df) -> DataFrame` (rows = year, columns = `Jan`…`Dec` and `Year`; each cell compounded `(1 + r).prod() − 1`; missing months NaN), `exposure_series(daily_details_df)`. Window default `settings.ROLLING_WINDOW`; fewer rows than the window → empty series (with correct columns), never an exception.

## F.5 `report_builder.py`

```python
@dataclass(slots=True)
class StrategyReport:
    metrics: dict[str, float]; trade_stats: dict[str, float]; benchmark: dict[str, Any] | None
    benchmark_label: str | None; equity: pd.DataFrame; drawdown: pd.DataFrame
    rolling_sharpe: pd.DataFrame; rolling_volatility: pd.DataFrame; monthly_returns: pd.DataFrame
    trade_distribution: pd.DataFrame; pnl_by_symbol: pd.DataFrame; exposure: pd.DataFrame
    def to_dict(self) -> dict[str, Any]       # JSON-ready (NaN/inf → None, Timestamps → ISO strings, DataFrames → list of records)
class ReportBuilder:
    def __init__(self, metrics=None, trade_analytics=None, benchmark_comparison=None) -> None
    def build(self, portfolio_history_df, trades_df, benchmark_history_df=None, benchmark_label=None,
              daily_details_df=None) -> StrategyReport
```
Empty trades → trade stats `{}` and empty distribution (no exception). Missing benchmark → `benchmark = None`.

## F.6 `metrics_persistence.py`

`persist_metrics(session, strategy_name, metrics: Mapping[str, float], calculated_at=None) -> int` using `PortfolioMetricRepository.bulk_insert` (skip NaN/inf, log how many skipped). `persist_trades(session, trades_df) -> int` using `TradeRepository.bulk_insert` (records from the canonical columns, `Timestamp → datetime`). Pure functions taking a session; callers own the transaction. Tests marked `db`.

## F.7 `backtest_pipeline.py`: the cross-layer wiring

```python
@dataclass(slots=True)
class BacktestRunResult:
    run_id: str; config: BacktestConfig; output: SimulationOutput; report: StrategyReport; run_dir: Path | None

class BacktestPipeline:
    def __init__(self, price_loader=None, signal_loader=None, store=None, report_builder=None,
                 benchmark_builder=None, sector_map: Mapping[str, str] | None = None) -> None
    def run(self, config: BacktestConfig, symbols: Sequence[str] | None = None, asset_class: str = "equity",
            timeframe: str = "daily", start=None, end=None, persist: bool = True,
            persist_db: bool = False, signals_df: pd.DataFrame | None = None, prices_df: pd.DataFrame | None = None) -> BacktestRunResult
    def rebuild_report(self, run_id: str) -> StrategyReport      # recompute analytics from stored outputs
```
Flow: `config.resolved().validate()` → load signals and prices (or use injected frames; symbols default = intersection of signal and price symbols, intersection empty → `ValueError("No common symbols between signals and prices.")`) → choose the simulator: `PortfolioSimulator` (legacy) when `config.is_legacy_compatible()` else `WeightedPortfolioSimulator` with `create_allocator(...)` (constraints from config + sector map) → build benchmark → `ReportBuilder.build` → if `persist`: `store.save_run` plus `metrics.json` and `report.json` via `save_json` → if `persist_db`: open `get_session()` and persist metrics and trades (failures log an ERROR and do not lose the file results) → return. The legacy simulator's result is converted to a `SimulationOutput` (details frames empty or derived: `daily_details_df` with exposure from the snapshot, `trade_details_df` with `exit_reason="signal"` or `"forced_close_end"` unknown → `"signal"`).

---

# PART G. WEEK 12: DASHBOARD (FASTAPI BACKEND + STREAMLIT FRONTEND)

## G.1 Principles

Read-only except the "run backtest" action. The Streamlit app talks to a **`DashboardClient`** abstraction with two implementations: `LocalClient` (calls `DashboardService` in process; default) and `ApiClient` (HTTP via an injected `httpx.Client`; used when `settings.DASHBOARD_API_URL` is set). A single contract test suite runs against both. No business logic in views.

## G.2 `config.py`, `schemas.py`

`config.py`: `DashboardConfig` dataclass built from settings (api url, api key, cache ttl, page size limits `DEFAULT_PAGE_SIZE = 100`, `MAX_PAGE_SIZE = 1000`). `schemas.py`: Pydantic v2 models: `HealthResponse`, `RunSummary` (run_id, strategy_name, created_at, n_trades, final_equity, start_date, end_date, config), `MetricsResponse` (metrics, trade_stats, benchmark), `SeriesResponse` (columns, rows), `Page[T]` (items, total, limit, offset), `SignalRow`, `SignalCounts`, `ModelInfo`, `PipelineRunInfo`, `BacktestRequest` (strategy_name, symbols, asset_class, timeframe, start, end, config overrides subset), `BacktestJobStatus` (job_id, status in queued|running|succeeded|failed, run_id, error). A helper `sanitize(obj)` converts NaN/inf to `None`, numpy scalars to Python, Timestamps to ISO strings (used by every response).

## G.3 `services.py`: `DashboardService`

Constructor injection of `BacktestStore`, `BacktestPipeline`, `SignalLoader`, `ModelStore` (from `src.models.model_store` if the module exists; else the method returns `[]` with a log), `RunHistory`, `HealthChecker`. Methods (all return plain Python/DataFrames; empty states are valid results, not errors): `list_runs()`, `get_run(run_id)`, `get_metrics(run_id)`, `get_equity(run_id)`, `get_drawdown(run_id)`, `get_rolling(run_id, window)`, `get_monthly_returns(run_id)`, `get_trades(run_id, symbol=None, start=None, end=None, limit, offset)`, `get_trade_distribution(run_id)`, `get_allocation(run_id, on_date=None)` (latest weights, or weights over time from `positions_history_df`), `get_benchmark(run_id)`, `latest_signals(asset_class, timeframe, as_of=None)`, `signal_counts(...)`, `list_models()`, `pipeline_runs(limit)`, `health()`, `submit_backtest(request)` / `job_status(job_id)` (in-memory job registry guarded by a lock; jobs run in a background thread; statuses persisted only through the finished run on disk). Report computation is cached per `(run_id, ttl)` in memory; `StrategyReport` is rebuilt via `BacktestPipeline.rebuild_report` when `report.json` is missing. Unknown run → `FileNotFoundError` (the API maps it to 404).

## G.4 FastAPI app (`api/`)

`create_app(service: DashboardService | None = None, config: DashboardConfig | None = None) -> FastAPI` (factory; `main.py` also exposes `app = create_app()` lazily guarded so importing does not touch disk). Title, version, OpenAPI docs enabled. CORS configurable (default same-origin only). `security.py`: if `settings.DASHBOARD_API_KEY` is set, every route except `/health` requires header `X-API-Key` (constant-time compare) else 401; no key configured → open (documented as local use only).

Endpoints (JSON; every error body `{"detail": "..."}`; dates ISO; NaN → null):

| Method and path | Purpose | Errors |
|---|---|---|
| `GET /health` | DB, data freshness, model, MLflow reachability (non-fatal), overall `ok|degraded|down` | always 200 with status |
| `GET /runs` | list `RunSummary` (newest first), query `limit` | |
| `GET /runs/{run_id}` | summary and config | 404 |
| `GET /runs/{run_id}/metrics` | metrics, trade stats, benchmark | 404 |
| `GET /runs/{run_id}/equity` `/drawdown` | series | 404 |
| `GET /runs/{run_id}/rolling?window=63` | rolling Sharpe and volatility | 404, 422 (window < 2) |
| `GET /runs/{run_id}/monthly-returns` | year × month table | 404 |
| `GET /runs/{run_id}/trades?symbol=&start=&end=&limit=&offset=` | paginated trades | 404, 422 (limit > MAX) |
| `GET /runs/{run_id}/trade-distribution` | histogram bins | 404 |
| `GET /runs/{run_id}/allocation?on_date=` | weights (latest or given date), weights over time | 404 |
| `GET /runs/{run_id}/benchmark` | comparison metrics and series | 404 |
| `GET /signals/latest?asset_class=&timeframe=&limit=` | latest signal per symbol | 503 if no signal files |
| `GET /signals/counts` | counts of +1/0/−1 on the latest date | 503 |
| `GET /models` | model-store listing (name, version, created_at, metrics) | |
| `GET /pipeline/runs?limit=` | recent pipeline run reports | |
| `POST /backtests` | submit a backtest (`BacktestRequest`) → 202 with `job_id` | 422 invalid config |
| `GET /backtests/{job_id}` | job status | 404 |

Validation errors from `BacktestConfig.validate()` map to 422. Unexpected exceptions map to 500 with a generic message (log the traceback; never leak internals or secrets).

## G.5 `client.py`

`DashboardClient` Protocol with one method per service call (same names and return shapes as `DashboardService`, returning DataFrames or dicts). `LocalClient(service)`; `ApiClient(http_client: httpx.Client, base_url="", api_key=None, timeout=10.0)` which converts JSON back to DataFrames (parse date columns), raises `DashboardClientError(status_code, detail)` on non-2xx, and sets `X-API-Key` when given. `get_client(config=None) -> DashboardClient` picks the implementation from settings.

## G.6 `charts.py`: pure Plotly builders (no Streamlit import)

`equity_curve_figure(equity_df, benchmark_df=None)`, `drawdown_figure(drawdown_df)`, `rolling_figure(rolling_df, title, ylabel)`, `monthly_heatmap_figure(monthly_df)` (diverging scale centred at 0; NaN cells blank), `trade_pnl_figure(trades_df)` (bars green/red), `trade_distribution_figure(dist_df)`, `allocation_area_figure(weights_over_time_df)`, `allocation_pie_figure(latest_weights, sector_map=None)`, `signal_counts_figure(counts)`, `metric_comparison_figure(strategy_metrics, benchmark_metrics)`. Each returns a `plotly.graph_objects.Figure` with title, axis labels and `template="plotly_white"`; empty input returns a figure with a centred annotation "No data available" (never an exception).

## G.7 Streamlit frontend

`components.py` helpers (`kpi_row(metrics, specs)`, `empty_state(message)`, `error_box(exc)`, `download_csv_button(df, name)`, `format_pct`, `format_ratio`, `format_currency` (INR with Indian digit grouping), `safe_call(fn, *args)` returning `(value, error)` so a failing API call renders an error box instead of crashing).

`app.py`: `st.set_page_config(page_title="Quant Research Platform", layout="wide")`; sidebar: title, a **radio** page selector (single entry script, no multipage folder), run selector (`selectbox` of runs, newest first, with strategy name and date), API health badge. Dispatch to `views/<page>.render(client, run_id)`. Cache expensive calls with `st.cache_data(ttl=settings.DASHBOARD_CACHE_TTL_SECONDS)` wrappers that take the client as an unhashed argument (`_client`). Provide `main(client: DashboardClient | None = None)` so tests can inject a fake client.

Pages (each `render(client, run_id)`, ≤ 300 lines, every empty or error state handled):

| View | Content |
|---|---|
| Overview | KPI row (total return, annualised return, Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor, number of trades), equity curve with benchmark overlay, drawdown chart, run configuration expander |
| Performance | rolling Sharpe and volatility (window slider 21–252), monthly returns heatmap, metrics table with download |
| Trades | filters (symbol multiselect, date range, win/loss), paginated table, PnL bar chart, return distribution, per-symbol PnL table, CSV download |
| Allocation | holdings-over-time stacked area, latest weights table and pie, date picker, sector pie when a sector map exists, exposure chart (gross/net) |
| Benchmark | alpha, beta, information ratio, tracking error, capture ratios, correlation, outperformance KPIs; metric comparison chart; label of the benchmark used (index or equal-weight fallback) |
| Signals | latest signals table with filter by signal value, counts bar chart, top-N by confidence |
| Experiments | model-store listing with validation/test metrics, selection to compare two models side by side |
| Pipeline | last pipeline runs with step statuses and durations, health checks table, data freshness |
| Run Backtest | form (strategy name, symbols multiselect, dates, allocation method, rebalance frequency, cost, slippage, lag, stops, shorting, cash reserve) with validation messages, submit, job status polling (`st.rerun`-safe), link to the new run |

Format numbers consistently: percentages 2 decimals, ratios 2 decimals, `inf` → "∞", NaN → "n/a", currency `₹` with Indian grouping.

---

# PART H. WEEK 13: AUTOMATION AND PRODUCTION READINESS

## H.1 `pipeline_models.py` and `run_history.py`

```python
class StepStatus(str, Enum): SUCCESS="success", FAILED="failed", SKIPPED="skipped"
@dataclass(slots=True) class StepResult: name, status, started_at, finished_at, duration_seconds, message, details: dict
@dataclass(slots=True) class PipelineRunReport: run_id, started_at, finished_at, steps: list[StepResult], config: dict
    @property all_succeeded -> bool (True only if every step succeeded) ; @property total_duration_seconds -> float ; to_dict()/from_dict()
class RunHistory:
    def __init__(self, root: Path | None = None)       # settings.PIPELINE_RUNS_PATH
    def save(self, report) -> Path ; def list(self, limit=20) -> list[PipelineRunReport] ; def latest(self) -> PipelineRunReport | None
```
Files `<root>/<run_id>.json` (atomic writes, `run_id` = `YYYYMMDDTHHMMSSZ_pipeline`). Corrupt files are skipped with a WARNING.

## H.2 `pipeline_steps.py`: adapters around real pipelines

`PipelineContext` dataclass: `run_id, asset_class, timeframe, symbols (list | None), dry_run, models: list[str], backtest_config: BacktestConfig | None, extras: dict`. A `PipelineStep` Protocol: `name: str`, `run(context) -> StepResult details` (raises on failure). Provide concrete steps that **adapt the real, existing classes** (read their actual signatures first; never guess):

1. `IngestionStep`: calls the real ingestion entry point of `src/data_pipeline/ingestion.py` with incremental mode for the context symbols.
2. `FeatureStep`: `FeaturePipeline().run()`; fails if `report.failed > 0` and `report.succeeded == 0`; partial failures become `details["failed_files"]` and a WARNING.
3. `PredictionStep`: uses the ML layer's prediction service with the latest model for each configured model name (`scope="latest"`); if no model exists, the step **fails with a clear message** `No trained model found ...` (do not train implicitly).
4. `SignalStep`: `SignalPipeline` with the generator named in `settings`/context (threshold default; rank and top-N selectable), symbols discovered from the predictions folder.
5. `BacktestStep`: `BacktestPipeline.run(context.backtest_config or BacktestConfig(strategy_name="daily"))` persisted.
6. `AnalyticsStep`: `BacktestPipeline.rebuild_report(run_id)` and metrics persistence to the database only when a DB session can be opened (otherwise `details["db"]="skipped"`).
`dry_run=True` makes every step validate its inputs and return without writing.

## H.3 `pipeline_orchestrator.py`

```python
class PipelineOrchestrator:
    def __init__(self, steps: Sequence[PipelineStep] | None = None, history: RunHistory | None = None) -> None   # default = the six steps above
    def run(self, context: PipelineContext) -> PipelineRunReport
    def run_retraining(self, ...) -> dict      # thin wrapper around the ML RetrainingPipeline if present; re-raises on failure
```
Semantics: strict linear dependency chain. Each step is wrapped in try/except with `perf_counter` timing; on failure the step is `FAILED` with the exception message and **all later steps are `SKIPPED`** with message `Skipped: upstream step '<name>' failed`. `run` never raises (it always returns a report) and always saves the report through `RunHistory` (a failure to save is logged, not raised). INFO summary at the end. `run_retraining` is separate, not part of the daily chain, and **does** re-raise.

## H.4 `scheduler.py`

`run_once(context: PipelineContext | None = None, orchestrator=None) -> PipelineRunReport` (the primary, cron/Task-Scheduler-friendly entry; builds the default context from settings). `start_background_scheduler(run_time: str | None = None, timezone: str | None = None, job=None) -> None` using `apscheduler.schedulers.blocking.BlockingScheduler` and a `CronTrigger(hour, minute, timezone)` parsed from `settings.PIPELINE_RUN_TIME` (`"HH:MM"`, validated with a clear `ValueError`) and `settings.PIPELINE_TIMEZONE`; the job wrapper catches and logs every exception so the scheduler survives. Provide `build_scheduler(...) -> BlockingScheduler` (without starting) so tests can inspect the trigger.

## H.4a `health.py`

```python
@dataclass(slots=True) class HealthCheck: name, status (ok|warn|fail), message, details
@dataclass(slots=True) class HealthReport: checks, overall (ok|degraded|down), generated_at; to_dict()
class HealthChecker:
    def __init__(self, session_factory=None, now=None, max_data_age_days: int = 5, max_model_age_days: int | None = None) -> None
    def check_database(self) -> HealthCheck          # `SELECT 1`; fail on exception (message without credentials)
    def check_data_freshness(self) -> HealthCheck    # newest date in the features and signals layers vs today (warn if older than max_data_age_days; fail if no files)
    def check_models(self) -> HealthCheck            # latest model exists and age (warn if older than 2 × MODEL_RETRAIN_INTERVAL_DAYS)
    def check_backtests(self) -> HealthCheck         # at least one stored run; warn otherwise
    def check_mlflow(self) -> HealthCheck            # HTTP GET tracking URI with short timeout when it is an http(s) URI; local file/SQLite URI → ok with note; failure is WARN not FAIL
    def run_all(self) -> HealthReport                # overall: any fail → down; else any warn → degraded; else ok
```
Each check catches its own exceptions and returns a result (never raises). Mask the database URL password in any message.

## H.5 Scripts (all thin: `main(argv=None) -> int`, argparse, exit code 0 success / 1 failure / 2 usage, prints a concise summary)

| Script | Arguments | Action |
|---|---|---|
| `run_backtest.py` | `--strategy-name`, `--asset-class`, `--timeframe`, `--symbols`, `--start`, `--end`, `--allocation-method`, `--rebalance`, `--cost`, `--slippage`, `--lag`, `--stop-loss`, `--take-profit`, `--allow-short`, `--cash-reserve`, `--max-position-size`, `--capital`, `--persist-db`, `--sensitivity` | build `BacktestConfig`, run `BacktestPipeline`, print key metrics and run id; `--sensitivity` also prints the grid; keep any previous behaviour of the existing script reachable (default arguments reproduce it) |
| `run_analytics.py` | `--run-id` (default latest) | rebuild report and print metrics |
| `run_daily_pipeline.py` | `--asset-class`, `--timeframe`, `--symbols`, `--dry-run`, `--skip` (steps to skip), `--models` | `scheduler.run_once`; exit code 1 if any step failed |
| `run_scheduler.py` | `--run-time`, `--timezone` | `start_background_scheduler` |
| `run_api.py` | `--host`, `--port`, `--reload` | `uvicorn.run("src.dashboard.api.main:app", ...)` |
| `run_dashboard.py` | `--port` | `subprocess` with `sys.executable -m streamlit run src/dashboard/app.py --server.port ...` |
| `health_check.py` | `--json` | print `HealthChecker().run_all()`; exit 1 if overall is `down` |

## H.6 Containers, CI and configuration

- `Dockerfile`: `python:3.12-slim`, non-root user, `pip install --no-cache-dir -e .`, copies `src`, `config`, `scripts`, `alembic*`; default `CMD` runs the API; no secrets baked in; `HEALTHCHECK` calling `scripts/health_check.py`. `.dockerignore` excludes `.venv`, `data`, `models`, `*.db`, `.git`, caches, tests output.
- `docker-compose.app.yml` (**new; do not edit `docker-compose.yml`**): services `api` (port 8000), `dashboard` (8501, `DASHBOARD_API_URL=http://api:8000`), `scheduler` (runs `scripts/run_scheduler.py`); all use the same image, `env_file: .env`, bind-mount `./data` and `./models`, restart policy `unless-stopped`, healthchecks, `depends_on` the existing `postgres`/`mlflow` services by name (read their names from `docker-compose.yml`; document the command `docker compose -f docker-compose.yml -f docker-compose.app.yml up -d`).
- `.github/workflows/ci.yml`: Python 3.12, install deps, `black --check`, `isort --check`, `flake8`, `pytest -m "not db"` with coverage summary; a second job with a PostgreSQL service running Alembic and the `db` tests. Cache pip.
- `.env.example`: every setting (existing and new) with comments and safe example values; no real secrets.

## H.7 Documentation (real, accurate, short; no invented results)

`README.md` (overview, architecture diagram in text or Mermaid, quick start for Windows PowerShell and Docker, commands for every script, project structure, testing, troubleshooting); `docs/architecture.md` (layers, dependency rules, data contracts, storage, simulation semantics with formulas); `docs/runbook.md` (daily operations, retraining, failure handling, backups, resets, port conflicts); `docs/user_guide.md` (dashboard walkthrough); `docs/api_reference.md` (endpoint table, auth, examples); `docs/production_readiness_checklist.md` (checklist with each item marked done or not, with evidence paths); `docs/end_to_end_verification_guide.md` (update: the exact commands and the **actual outputs you obtained** in Part L step 6; never invent numbers).

---

# PART I. END-TO-END BEHAVIOUR TO GUARANTEE

1. `features → training → prediction → signals → backtest → analytics → API → dashboard` works on the synthetic dataset created by the integration test fixture (J.7) with no database and no network.
2. A legacy-compatible config produces the same closed-trade PnL in `PortfolioSimulator` and `WeightedPortfolioSimulator`.
3. Re-running any step is idempotent (no duplicates; results overwritten or new run ids).
4. A failing step never crashes the orchestrator; the report says what failed and what was skipped.
5. The dashboard never crashes on empty data; every view shows a clear empty state.

---

# PART J. TEST SPECIFICATION

## J.1 Global method

- Plain `pytest` functions, banners in the existing style, names `test_<unit>_<behaviour>`. Golden numbers computed by hand, shown in a comment above the assert, compared with `pytest.approx`. Every `pytest.raises` uses `match=`.
- Deterministic: `np.random.default_rng(<seed>)`. No network. Temp folders via `tmp_path`. For `ParquetWriter` tests patch **both** `settings.DATA_<LAYER>_PATH` and `LAYER_PATH_MAP[DataLayer.<LAYER>]`. Settings overrides via `monkeypatch.setattr(settings, ...)`.
- MLflow always mocked (reuse the existing autouse guard in `tests/models/conftest.py`; add an equivalent in `tests/integration/conftest.py`).
- Markers: `@pytest.mark.db` for tests needing PostgreSQL (they use the existing transaction-rollback fixture and skip with a clear reason if the database is unreachable); `@pytest.mark.integration` for end-to-end tests. Everything else runs without Docker.
- Each unit gets: validation tests, golden-value tests, edge cases, a no-mutation check where a DataFrame is accepted.
- Shared fixtures in a new `tests/conftest_quant.py` imported by the relevant `conftest.py` files (or placed in `tests/backtesting/conftest.py` and `tests/integration/conftest.py` if sharing is awkward): `price_panel(n_symbols, n_days, seed)` (geometric random walk, business days, columns `symbol, date, close, adj_close, volume`), `signal_panel(price_panel, strategy)`, `history_frame(values)` (canonical 7-column history from a list of portfolio values), `trades_frame(rows)`, `fake_dashboard_client` (implements `DashboardClient` with small deterministic frames), `populated_store(tmp_path)` (a `BacktestStore` with one saved run), `tz_variants` (naive and `Asia/Kolkata`-aware date columns).

## J.2 `tests/backtesting/` (existing files untouched; add)

- `test_backtest_config.py`: defaults resolve from settings (monkeypatched); each validation rule raises with the value in the message; `allow_short` with `risk_parity` raises; `to_dict/from_dict` round trip (enums as strings); `config_hash` stable and sensitive to every field; `is_legacy_compatible` truth table.
- `test_price_signal_loaders.py`: discovery (including a symbol with an underscore, other timeframe ignored, missing folder → `[]`); load filters dates (naive and tz-aware bounds); missing symbol skipped with warning; none found raises; duplicate `(symbol, date)` raises; invalid signal value raises; date dtype mismatch raises; output sorted; no mutation.
- `test_position_book.py`: add/average-up golden numbers (100 @ 10 then 100 @ 12 → 200 @ 11); reduce partial and full with fee/slippage shares; side mismatch raises; market value, unrealised PnL, exposures for long/short; tiny residual quantities close the position.
- `test_weighted_simulator_core.py`: fixed-notional single symbol golden trade equals the legacy engine's arithmetic (price 100→110, cost and slippage on/off); equal-weight two symbols exact quantities and cash; rebalance on set change trims an existing position; tolerance suppresses a small resize; `min_trade_value` skip; funding priority by confidence (two BUYs, cash for one → higher confidence funded; without a confidence column alphabetical); partial fill when cash is short; cash never negative (property test over a seeded random scenario); `max_open_positions` truncation; `cash_reserve` respected; weekly and monthly rebalance dates; final force-close included in the last snapshot with the `Force-closing` warning; zero-trade run warns with `zero trades`; empty-frame errors and frozen messages.
- `test_weighted_simulator_stops_lag_short.py`: stop-loss exits at the first close below the threshold with reason `stop_loss`, symbol stays flat until a new BUY; take-profit; lag = 1 executes at the next date and the allocator receives history only up to the decision date (assert with a recording fake allocator that no date > decision date is present); signals beyond the calendar end dropped; shorting open/cover PnL golden numbers and exposure; `allow_short` rejects unsupported allocation; equity identity `equity = cash + long − short` on every row.
- `test_simulator_parity.py`: for several seeded panels and a legacy-compatible config, closed-trade PnL per symbol and final equity (all positions closed) match between `PortfolioSimulator` and `WeightedPortfolioSimulator` within `1e-6`.
- `test_backtest_store.py`: save/load round trip of all frames (dtypes and tz preserved); `list_runs` newest first; `latest_run_id`; invalid run id (`../x`) raises; missing run raises; `save_json/load_json` with NaN → null; delete; atomic write leaves no partial files when serialisation fails (monkeypatch a failure).
- `test_sensitivity.py`: grid size and columns; costs monotonically reduce final equity on a deterministic scenario; lag changes results; pure (no files written).

## J.3 `tests/portfolio/` (existing files untouched; add)

- `test_constraints.py`: scaling to investable; per-position cap with redistribution; sector cap binding (golden numbers); asset-class cap; min weight dropping; infeasible budget → cash weight > 0; weights non-negative, each ≤ cap, sums consistent (property tests over seeded random inputs); `binding` messages; `UNKNOWN` sector behaviour; invalid parameters raise.
- `test_adapters.py`: each legacy allocator through the adapter returns constrained weights; empty candidates → empty result with reasons; prices truncated to the decision date; short side returns magnitudes.
- `test_signal_weighted.py`: weights proportional to confidence (golden); NaN or non-positive basis excluded; all zero → equal fallback with the warning; basis `predicted_return`.
- `test_risk_parity.py`: two assets with volatilities 1:2 → inverse-volatility-like weights verified by risk contributions equal within tolerance; identical assets → equal weights; insufficient history excluded; solver failure fallback (monkeypatch `scipy.optimize.minimize` to raise) with the warning; deterministic across calls; covariance regularisation keeps a singular matrix workable.
- `test_allocator_factory.py`: names, unknown name message, `fixed_notional` rejected, `register_allocator` duplicate rejected, created allocators satisfy the protocol and return `AllocationResult`.

## J.4 `tests/analytics/` (existing files untouched; add or extend in new files)

- `test_performance_metrics_extended.py`: golden values on a hand-built history (for example values 100, 110, 99, 108.9 → total return 0.089, daily returns, drawdown −0.10); annualisation with known periods; Sharpe/Sortino/Calmar golden and degenerate cases (flat history, no losses) with `caplog` warning texts; VaR/CVaR golden on a fixed return vector; invalid confidence raises; `compute_all` keys; `max_drawdown_duration_days` golden; single-row history.
- `test_trade_analytics_extended.py`: all metrics golden on a 6-trade frame; consecutive wins and losses ordering; `pnl_by_symbol`; distribution bins sum to the trade count; no losses → `inf` profit factor with warning; breakeven handling; empty frame raises.
- `test_benchmark.py`: identical series → beta 1, alpha 0, excess 0, tracking error 0; a series of exactly `1.5 × benchmark` returns → beta 1.5; known up/down capture; no overlap raises; zero benchmark variance → beta NaN with warning; `BenchmarkBuilder.from_index` buy-and-hold math; missing index file → `None` with warning; equal-weight universe math; forward-fill limits.
- `test_rolling_metrics.py`: window longer than data → empty frames with correct columns; rolling Sharpe golden for a constant-mean series; monthly table compounding golden (two months, partial year); drawdown series underwater flag.
- `test_report_builder.py`: full bundle on a synthetic run; `to_dict` JSON-serialisable (`json.dumps(..., allow_nan=False)`); empty trades; missing benchmark; consistent numbers between the bundle and the individual calculators.
- `test_backtest_pipeline.py`: end-to-end on injected frames for legacy and weighted configs; `persist=True` writes the run folder including `metrics.json` and `report.json`; `rebuild_report` reproduces the stored metrics; no common symbols raises; benchmark fallback label when no index data; `persist_db` failure logged without losing files (mock the session factory to raise).
- `test_metrics_persistence.py` (`db`): metrics and trades round trip through the repositories; NaN skipped; counts returned.

## J.5 `tests/dashboard/` (backend and frontend)

**Backend (`test_api_*.py`) using `fastapi.testclient.TestClient` with a `DashboardService` built on `populated_store` and fakes:** every endpoint's happy path, schema fields, and `NaN → null`; 404 for unknown run; 422 for `limit` above max, `window < 2`, invalid backtest config; pagination `total/limit/offset` correctness; `GET /signals/latest` 503 when no signal files; API-key behaviour (401 without or with a wrong key, 200 with the key, `/health` open); CORS headers when configured; `POST /backtests` returns 202 then the job reaches `succeeded` (run the job synchronously via an injected executor in tests) and the new run appears in `/runs`; unexpected exception → 500 with a generic body and no traceback text; OpenAPI schema generates without error and lists every route.

**Client contract (`test_client_contract.py`):** the same assertions run against `LocalClient(service)` and `ApiClient(TestClient(app))` (parametrised fixture): identical DataFrame columns and values for runs, metrics, equity, trades, allocation, benchmark, signals; `ApiClient` raises `DashboardClientError` with the status code on 404; API key header is sent.

**Charts (`test_charts.py`):** every builder returns a `Figure` with title and axis labels; trace counts and data lengths match the inputs; empty input → the "No data available" annotation; heatmap handles NaN; no exception for single-row input.

**Frontend (`test_app_ui.py`) using `streamlit.testing.v1.AppTest`:** load `src/dashboard/app.py` with an injected `fake_dashboard_client`; for **every page** select it in the sidebar radio, `at.run(timeout=30)`, then assert: no exception (`not at.exception`), at least one expected element (metrics, dataframe, plotly chart via `at.get("plotly_chart")` or the number of chart calls recorded by the fake), correct KPI labels, correct formatting (percentages, `∞`, `n/a`, ₹ grouping unit-tested separately in `test_components.py`). Scenarios: no runs available (empty-state message on every page); client raising `DashboardClientError` (error box shown, app still renders); changing the run selector switches data; Trades page filters reduce rows; Allocation date picker; Run Backtest page: invalid input shows validation messages without calling the client, valid submit calls `submit_backtest` once and shows the job id; Pipeline page shows step statuses and health table.

**Components (`test_components.py`):** formatting helpers golden values (`format_currency(1234567.5) == "₹12,34,567.50"`, `format_pct(0.1234) == "12.34%"`, infinity/NaN handling), `safe_call` returns `(None, error)` on exceptions.

## J.6 `tests/infra/`

- `test_pipeline_models_history.py`: `all_succeeded`, duration, JSON round trip, `RunHistory` save/list/latest ordering, corrupt file skipped with a warning, atomic write.
- `test_pipeline_orchestrator.py` with mock steps: all succeed; failure at each position skips exactly the later steps with the documented message and never calls them; an unexpected exception type is captured; the orchestrator never raises; report always saved (also when the history save fails); `dry_run` passes through; `run_retraining` re-raises.
- `test_pipeline_steps.py`: each step adapter invoked with mocked underlying pipelines (assert the real class is called with the right arguments, partial feature failures become warnings, `PredictionStep` fails with the documented message when no model exists, dry-run writes nothing).
- `test_scheduler.py`: `run_once` returns the report and respects injected orchestrator/context; `build_scheduler` trigger fields equal the parsed `PIPELINE_RUN_TIME` and timezone; invalid time string raises `ValueError`; a failing job is logged and does not propagate (call the wrapper directly).
- `test_health.py`: each check ok/warn/fail path with fakes (monkeypatched session factory, temp data folders, stub model store); `run_all` overall logic; password masking; every check swallows its own exceptions.
- `tests/scripts/test_quant_scripts.py`: load each script with `importlib`, patch collaborators, assert argument forwarding, defaults from settings, exit codes 0/1/2, `--dry-run`, JSON output of `health_check.py`.

## J.7 `tests/integration/` (`@pytest.mark.integration`; no DB, no network)

A session fixture builds a **synthetic data lake in `tmp_path`** (patch every layer path and map): 8 symbols × 400 business days of processed OHLCV with a learnable pattern (target correlated with one feature) → run `FeaturePipeline` → `TrainingPipeline` (small LightGBM or Ridge, MLflow mocked) with a `ModelStore(tmp_path)` → predictions (scope `test`) → signals (threshold and rank) → `BacktestPipeline` for three configs (legacy-compatible fixed notional; equal-weight monthly rebalance; risk-parity with stop-loss and lag 1) → analytics → `DashboardService` → `TestClient` calls → `AppTest` render of Overview and Trades. Assertions: files and schemas at every stage; signals in {−1, 0, 1}; backtest dates only in the out-of-sample window; equity identity per row; legacy parity; metrics finite or documented NaN; report JSON valid; API returns the stored run; the pages render; running the whole chain twice is idempotent. If a lower layer in this repository lacks a hook needed by the test, use the real classes where possible and otherwise a minimal fake **clearly named in the test**; list it in the report.

Also include `tests/integration/test_orchestrator_end_to_end.py`: `PipelineOrchestrator` with real adapters on the synthetic lake (ingestion step replaced by a fake that writes the synthetic processed files), asserting a green report and a saved run history.

---

# PART K. QUALITY GATES (run in order; fix everything they report)

1. `<python> -m black --check --line-length 88 src tests scripts config` (run `black` on touched files first), `<python> -m isort --check --profile black src tests scripts config`.
2. `<python> -m flake8` on touched files: zero errors (unused imports, undefined names).
3. Static scan of new and modified files: no `import logging` for loggers, no f-strings inside `logger.*`, no `TODO`, no `print(` outside `scripts/`, no `os.getenv`, no hard-coded paths/URLs/rates, each file ≤ 300 lines (`wc -l` equivalent), every public function typed and documented.
4. Dependency direction: no `src.portfolio`/`src.analytics`/`src.dashboard` import inside `src/backtesting`; no `src.analytics`/`src.dashboard` import inside `src/portfolio`; no `src.dashboard` import inside `src/analytics`; nothing in `src/` imports from `scripts/` or `tests/`. Import every new module once (`python -c "import ..."`) to catch circular imports.
5. `<python> -m pytest tests -q -m "not db and not integration"` then `-m integration`, then (if PostgreSQL is reachable) `-m db` and the whole suite. **Zero failures, zero errors; no previously passing test modified or removed.** Report: baseline vs final counts, counts per folder, runtime, any skipped test with the reason. Fix warnings that come from your own code.
6. **Real end-to-end run** (using the repository's real data if present, otherwise the synthetic generator used by the integration test, saved under a temp folder, never overwriting real data): run `scripts/run_backtest.py`, `scripts/run_analytics.py`, `scripts/health_check.py`, `scripts/run_daily_pipeline.py --dry-run`; start the API in-process (TestClient) and render the dashboard with `AppTest`; record the **actual** outputs in `docs/end_to_end_verification_guide.md`.
7. Confirm `docker compose -f docker-compose.yml -f docker-compose.app.yml config` parses (if Docker is available; otherwise validate the YAML structure with a YAML parser) and that the Dockerfile builds logically (no execution required if Docker is unavailable; say so).
8. `git status`: only files allowed in A.3 changed.

---

# PART L. FINAL REPORT (print when finished)

1. **Audit table** (before) and **inventory** (after): files created, modified (one line each), untouched.
2. **Behaviour decisions taken** from Part B and any deviation from this spec with the reason.
3. **Test results:** baseline versus final counts, per folder, runtime, skipped tests.
4. **Quality gates:** output summary of each gate.
5. **Real end-to-end results:** the commands run and the key numbers produced, clearly labelled as real or synthetic data.
6. **Cross-module issues** found in files you were not allowed to change (with file, line and suggested fix).
7. **Known limitations of what you built** (for example close-price stops, fractional shares, flat transaction cost, same-day execution at lag 0, in-memory job registry, Streamlit single-process caching) and what you recommend next.
8. **Handoff checklist:** commands to start everything (local and Docker), where results are stored, how to add a strategy, an allocator and a dashboard page.

Begin with Part B now and work through D → H in order, running tests after each part.
