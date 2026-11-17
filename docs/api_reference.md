# Dashboard REST API Reference

The platform provides a high-performance REST API built with FastAPI. Interactive Swagger UI is available at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc`.

---

## Authentication

When `DASHBOARD_API_KEY` is configured in `.env`, all routes (except `/health`) require the `X-API-Key` HTTP header.

```http
GET /runs HTTP/1.1
Host: localhost:8000
X-API-Key: your-secret-api-key
```

If `DASHBOARD_API_KEY` is unset or empty, the API operates in open mode for local development.

---

## Endpoints

### 1. System & Health

#### `GET /health`
Returns system operational health across database, data freshness, and ML models.
- **Status Codes**: `200 OK`
- **Response**:
```json
{
  "status": "ok",
  "checks": [
    {
      "name": "database",
      "status": "ok",
      "message": "Database connection operational"
    },
    {
      "name": "data_freshness",
      "status": "ok",
      "message": "Data is fresh (0.5 days old)"
    },
    {
      "name": "models",
      "status": "ok",
      "message": "Model artifacts operational"
    }
  ],
  "generated_at": "2026-11-17T03:00:00Z"
}
```

#### `GET /models`
Returns list of trained model artifacts available in the registry.

#### `GET /pipeline/runs?limit=20`
Returns summary records of recent daily pipeline batch runs.

---

### 2. Backtest Runs & Analytics

#### `GET /runs?limit=100`
Returns list of completed backtest run summaries ordered newest first.

#### `GET /runs/{run_id}`
Returns metadata, execution summary, and configuration for a specific run.

#### `GET /runs/{run_id}/metrics`
Returns computed performance metrics, trade statistics, and benchmark comparisons.

#### `GET /runs/{run_id}/equity`
Returns chronological portfolio equity curve and cumulative returns.

#### `GET /runs/{run_id}/drawdown`
Returns underwater drawdown time series.

#### `GET /runs/{run_id}/rolling?window=63`
Returns rolling Sharpe ratio and annualized volatility.
- **Parameters**: `window` (int, default 63, minimum 2).

#### `GET /runs/{run_id}/monthly-returns`
Returns year × month compounded returns matrix.

#### `GET /runs/{run_id}/trades?symbol=&start=&end=&limit=100&offset=0`
Returns paginated trade execution records with optional symbol and date filtering.

#### `GET /runs/{run_id}/trade-distribution`
Returns histogram bin intervals and counts for trade return percentages.

#### `GET /runs/{run_id}/allocation?on_date=`
Returns asset weight allocations as of latest date or given date.

#### `GET /runs/{run_id}/benchmark`
Returns benchmark relative active risk metrics (alpha, beta, tracking error, etc.).

---

### 3. Signals

#### `GET /signals/latest?asset_class=equity&timeframe=daily`
Returns latest generated trading signal per symbol.

#### `GET /signals/counts?asset_class=equity&timeframe=daily`
Returns aggregate count of buy (+1), neutral (0), and sell (-1) signals.

---

### 4. Backtest Execution

#### `POST /backtests`
Submits a backtest job to run asynchronously in a background worker thread.
- **Status Codes**: `202 Accepted`, `422 Unprocessable Entity`
- **Request Body**:
```json
{
  "strategy_name": "trend_momentum",
  "allocation_method": "equal_weight",
  "rebalance_frequency": "daily",
  "initial_capital": 100000.0,
  "execution_lag": 1,
  "max_position_size": 0.20,
  "cash_reserve": 0.05
}
```
- **Response**: `{"job_id": "job_1234567890"}`

#### `GET /backtests/{job_id}`
Polls status of a submitted background backtest job (`queued`, `running`, `succeeded`, `failed`).
