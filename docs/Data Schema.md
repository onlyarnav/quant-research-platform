Quantitative AI Research Platform — Data Schema Specification

## 1. Purpose

This document defines all canonical data schemas used throughout the Quantitative AI Research Platform.

All engineers and AI coding assistants must strictly follow these schemas.

No module may alter schemas without updating this document and obtaining team agreement.

This file acts as the **single source of truth for all dataset and database contracts**.

---

## 2. Schema Design Principles

All schemas in this project follow these principles:

### Consistency

Schemas should be uniform across asset classes wherever possible.

### Explicitness

Every dataset should include enough metadata to be self-describing.

### Extensibility

Schemas should support future asset classes without redesign.

### Stability

Once downstream modules depend on a schema, breaking changes should be avoided.

---

## 3. Data Storage Overview

Large time-series datasets are stored as **Parquet files**.

Directory structure:

```text
data/
    raw/
    processed/
    features/
    signals/
```

Structured metadata and analytics records are stored in **PostgreSQL**.

---

## 4. File Naming Convention

All Parquet files must follow:

```text
<asset_class>_<symbol>_<timeframe>.parquet
```

Examples:

```text
equity_RELIANCE_daily.parquet
index_NIFTY50_daily.parquet
crypto_BTCUSDT_hourly.parquet
forex_USDINR_daily.parquet
commodity_GOLD_daily.parquet
```

---

## 5. Raw Market Data Schema

### Description

Represents direct market data from external providers before processing.

### Storage Location

```text
data/raw/
```

### Schema

|Column|Type|Required|Description|
|---|---|---|---|
|symbol|string|Yes|Trading symbol / ticker|
|asset_class|string|Yes|equity / index / etf / forex / crypto / commodity / macro|
|exchange|string|Yes|NSE / BINANCE / FX / etc|
|currency|string|Yes|Trading currency|
|date|datetime|Yes|Observation timestamp|
|open|float|Yes|Opening price|
|high|float|Yes|Highest price|
|low|float|Yes|Lowest price|
|close|float|Yes|Closing price|
|adj_close|float|Optional|Adjusted close if available|
|volume|float|Optional|Trading volume|
|source|string|Yes|Data source provider|

### Primary Index

```text
(symbol, date)
```

---

## 6. Processed Market Data Schema

### Description

Validated and cleaned market data used by downstream pipelines.

### Storage Location

```text
data/processed/
```

### Schema

|Column|Type|Required|Description|
|---|---|---|---|
|symbol|string|Yes|Asset symbol|
|asset_class|string|Yes|Asset type|
|exchange|string|Yes|Trading exchange|
|currency|string|Yes|Currency|
|date|datetime|Yes|Observation date|
|open|float|Yes|Open price|
|high|float|Yes|High price|
|low|float|Yes|Low price|
|close|float|Yes|Close price|
|adj_close|float|Optional|Adjusted close|
|volume|float|Optional|Volume|
|returns|float|Yes|Simple returns|
|log_returns|float|Yes|Log returns|

---

## 7. Feature Dataset Schema

### Description

Feature matrices used for machine learning models.

### Storage Location

```text
data/features/
```

### Base Required Columns

|Column|Type|Required|Description|
|---|---|---|---|
|symbol|string|Yes|Asset symbol|
|date|datetime|Yes|Observation date|

### Feature Columns (Examples)

#### Trend Features

|Column|Type|
|---|---|
|sma_5|float|
|sma_10|float|
|sma_20|float|
|sma_50|float|
|sma_200|float|
|ema_10|float|
|ema_20|float|
|ema_50|float|

#### Momentum Features

|Column|Type|
|---|---|
|rsi_14|float|
|macd|float|
|macd_signal|float|
|roc_5|float|
|roc_10|float|
|momentum_5|float|
|momentum_10|float|

#### Volatility Features

|Column|Type|
|---|---|
|volatility_10|float|
|volatility_20|float|
|volatility_50|float|
|atr_14|float|
|bollinger_width|float|

#### Mean Reversion Features

|Column|Type|
|---|---|
|zscore_20|float|
|distance_from_sma|float|
|distance_from_ema|float|

#### Volume Features

|Column|Type|
|---|---|
|volume_ma_20|float|
|volume_ratio|float|
|obv|float|

---

## 8. Target Schema

### Description

Prediction targets appended to feature datasets.

### Supported Targets

|Column|Type|Description|
|---|---|---|
|future_return_1d|float|Next day return|
|future_return_5d|float|5-day forward return|
|future_return_10d|float|10-day forward return|

---

## 9. Prediction Dataset Schema

### Description

Raw model prediction outputs before signal conversion.

### Storage Location

```text
data/predictions/
```

### Schema

|Column|Type|
|---|---|
|symbol|string|
|date|datetime|
|predicted_return|float|
|model_name|string|
|model_version|string|

---

## 10. Signal Dataset Schema

### Description

Trading signals generated from predictions.

### Storage Location

```text
data/signals/
```

### Schema

|Column|Type|Description|
|---|---|---|
|symbol|string|Asset symbol|
|date|datetime|Signal timestamp|
|predicted_return|float|Model forecast|
|signal|int|1 buy / 0 hold / -1 sell|
|confidence|float|Optional model confidence|
|rank|int|Optional rank among universe|

---

## 11. Trade Log Schema

### Description

Trade-level records produced by backtesting engine.

### Schema

|Column|Type|
|---|---|
|trade_id|int|
|symbol|string|
|entry_date|datetime|
|exit_date|datetime|
|entry_price|float|
|exit_price|float|
|position_size|float|
|fees|float|
|slippage_cost|float|
|pnl|float|
|return_pct|float|

---

## 12. Portfolio History Schema

### Description

Daily portfolio valuation and performance history.

### Schema

|Column|Type|
|---|---|
|date|datetime|
|cash|float|
|invested_capital|float|
|portfolio_value|float|
|daily_return|float|
|cumulative_return|float|
|drawdown|float|

---

## 13. Portfolio Metrics Schema

### Description

Aggregate strategy / portfolio performance metrics.

### Schema

|Column|Type|
|---|---|
|metric|string|
|value|float|

### Example Metrics

- sharpe_ratio
    
- sortino_ratio
    
- max_drawdown
    
- annual_return
    
- volatility
    
- calmar_ratio
    
- win_rate
    
- profit_factor
    

---

## 14. Database Tables

---

### 14.1 assets

|Column|Type|
|---|---|
|id|int|
|symbol|string|
|name|string|
|asset_class|string|
|exchange|string|
|currency|string|
|sector|string|
|is_active|bool|
|created_at|datetime|
|updated_at|datetime|

---

### 14.2 prices

|Column|Type|
|---|---|
|id|int|
|symbol|string|
|date|datetime|
|open|float|
|high|float|
|low|float|
|close|float|
|adj_close|float|
|volume|float|
|source|string|
|created_at|datetime|

---

### 14.3 features

|Column|Type|
|---|---|
|id|int|
|symbol|string|
|date|datetime|
|feature_set_version|string|
|created_at|datetime|

---

### 14.4 signals

|Column|Type|
|---|---|
|id|int|
|symbol|string|
|date|datetime|
|signal|int|
|predicted_return|float|
|model_version|string|

---

### 14.5 trades

|Column|Type|
|---|---|
|id|int|
|symbol|string|
|entry_date|datetime|
|exit_date|datetime|
|pnl|float|
|return_pct|float|

---

### 14.6 portfolio_metrics

|Column|Type|
|---|---|
|id|int|
|strategy_name|string|
|metric|string|
|value|float|
|calculated_at|datetime|

---

## 15. Data Validation Rules

Before data enters processed datasets:

### Required Checks

- no duplicate `(symbol, date)` rows
    
- no null required fields
    
- no negative volume
    
- `high >= low`
    
- `high >= open/close`
    
- `low <= open/close`
    
- chronological sorting enforced
    

Invalid rows must be rejected or quarantined.

---

## 16. Indexing Requirements

### Database Indexes

Required indexes:

```text
(symbol, date)
(date)
(asset_class)
```

These indexes ensure efficient historical and cross-sectional queries.

---

## 17. Data Versioning Rules

Datasets must support reproducibility.

Recommended versioning approaches:

- timestamped Parquet snapshots
    
- MLflow artifact references
    
- dataset hash metadata
    
- feature set version columns
    

---

## 18. Schema Ownership

### Owned By

Data Engineering Team

### Schema Change Policy

Any schema modification must be reviewed by:

- Data Engineer
    
- ML Engineer
    
- Quant Engineer
    

before implementation.

---

## 19. Final Rule

All engineers and AI coding assistants must treat this file as authoritative.

If generated code conflicts with this schema specification, the code is wrong.