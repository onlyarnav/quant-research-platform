# Quantitative Research Platform: User Guide

## 1. Introduction

The platform empowers quantitative researchers and portfolio managers to simulate, evaluate, and deploy systematic trading strategies under realistic institutional constraints.

## 2. Key Capabilities

### 2.1 Capital Allocation Schemas
Researchers can choose between multiple allocation schemes by setting `--allocation-method` or configuring `BacktestConfig`:

1. **`equal_weight`**: Allocates capital uniformly across all assets receiving an active signal on each rebalancing date, constrained by `max_position_size` and `cash_reserve`.
2. **`signal_weighted`**: Weights assets proportional to prediction model confidence or raw forecast magnitude. High-conviction signals receive larger position sizing up to the position cap.
3. **`volatility_weighted`**: Inverse-volatility allocation. Lower volatility assets receive larger weight allocations to equalize risk contribution.
4. **`risk_parity`**: Equal Risk Contribution (ERC) portfolio solver using numerical SLSQP optimization over empirical covariance matrices.
5. **`fixed_notional`**: Legacy compatible execution where positions are entered with fixed currency amounts per signal and held until exit.

### 2.2 Rebalancing Schedules
- **`daily`**: Rebalances portfolio weights on every trading day.
- **`weekly`**: Rebalances on the first trading day of each week.
- **`monthly`**: Rebalances on the first trading day of each month.
- **`on_signal`**: Rebalances only when new signals or position exit conditions occur.

### 2.3 Stop-Loss and Take-Profit Rules
Intraday risk stops protect capital against adverse market shocks:
- **Stop Loss (`stop_loss_pct`)**: Automatically closes an open lot when mark-to-market loss exceeds the threshold relative to entry price.
- **Take Profit (`take_profit_pct`)**: Locks in gains when mark-to-market gain exceeds the threshold.

### 2.4 Multi-Day Execution Lag
Setting `execution_lag = 1` enforces realistic latency: signals generated on close of day $t$ are entered at the open/close of day $t+1$, completely eliminating lookahead bias.

---

## 3. Navigating the Streamlit Research Dashboard

The dashboard provides 9 specialized views accessible via the sidebar radio selector:

1. **Overview**: Executive summary displaying headline KPIs (Total Return, Annualized Return, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Win Rate, Profit Factor, Trade Count), interactive equity curves with benchmark overlays, underwater drawdown charts, and full run configuration JSON.
2. **Performance**: In-depth risk dynamics. Includes a dynamic rolling window slider (21 to 252 days) charting rolling Sharpe ratio and volatility, alongside a compounded monthly returns heatmap matrix.
3. **Trades**: Complete trade execution logs with symbol search, date range filters, PnL by symbol breakdown bar charts, and return distribution histogram.
4. **Allocation**: Current and historical portfolio holdings weights, donut charts, and exposure breakdown.
5. **Benchmark**: Active risk and outperformance evaluation, tracking Jensen's alpha, beta, tracking error, information ratio, correlation, and up/down market capture.
6. **Signals**: Real-time signal generator monitor charting latest buy/sell/neutral signal counts and confidence scores.
7. **Experiments**: ML Model Registry browser allowing side-by-side comparison of trained model hyperparameters and validation metrics.
8. **Pipeline**: Daily automated batch run tracker displaying step statuses, durations, and system health checks.
9. **Run Backtest**: Interactive GUI form to launch new backtest runs with custom parameters, live progress tracking, and instant report generation.
