import pytest
import pandas as pd
import numpy as np
from src.portfolio.mean_variance import MeanVarianceAllocator
from config.settings import settings

def make_price_series(symbol: str, start_date: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=len(closes))
    return pd.DataFrame({"symbol": symbol, "date": dates, "close": closes})

def test_init_rejects_lookback_below_2():
    with pytest.raises(ValueError, match="lookback_window must be at least 2"):
        MeanVarianceAllocator(lookback_window=1)

def test_init_default_risk_free_rate_from_settings():
    allocator = MeanVarianceAllocator()
    assert allocator.risk_free_rate == settings.PORTFOLIO_RISK_FREE_RATE

def test_allocate_raises_on_empty_signals_df():
    allocator = MeanVarianceAllocator()
    with pytest.raises(ValueError, match="signals_df cannot be empty"):
        allocator.allocate(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2023-01-01"))

def test_allocate_raises_on_empty_prices_df():
    allocator = MeanVarianceAllocator()
    signals = pd.DataFrame({"symbol": ["AAPL"], "signal": [1]})
    with pytest.raises(ValueError, match="prices_df cannot be empty"):
        allocator.allocate(signals, pd.DataFrame(), pd.Timestamp("2023-01-01"))

def test_allocate_raises_on_missing_signal_columns():
    allocator = MeanVarianceAllocator()
    signals = pd.DataFrame({"wrong_col": ["AAPL"], "signal": [1]})
    prices = pd.DataFrame({"symbol": ["AAPL"], "date": [pd.Timestamp("2023-01-01")], "close": [150.0]})
    with pytest.raises(ValueError, match="signals_df missing required columns"):
        allocator.allocate(signals, prices, pd.Timestamp("2023-01-01"))

def test_allocate_raises_on_missing_price_columns():
    allocator = MeanVarianceAllocator()
    signals = pd.DataFrame({"symbol": ["AAPL"], "signal": [1]})
    prices = pd.DataFrame({"symbol": ["AAPL"], "wrong_col": [pd.Timestamp("2023-01-01")], "close": [150.0]})
    with pytest.raises(ValueError, match="prices_df missing required columns"):
        allocator.allocate(signals, prices, pd.Timestamp("2023-01-01"))

def test_allocate_returns_empty_dict_when_no_buy_signals():
    allocator = MeanVarianceAllocator()
    signals = pd.DataFrame({"symbol": ["AAPL"], "signal": [0]})
    prices = pd.DataFrame({"symbol": ["AAPL"], "date": [pd.Timestamp("2023-01-01")], "close": [150.0]})
    allocation = allocator.allocate(signals, prices, pd.Timestamp("2023-01-01"))
    assert allocation == {}

def test_allocate_falls_back_to_equal_weight_with_single_symbol():
    allocator = MeanVarianceAllocator(lookback_window=5)
    symbol = "AAPL"
    signals = pd.DataFrame({"symbol": [symbol], "signal": [1]})
    prices = make_price_series(symbol, "2023-01-01", [100.0, 101.0, 102.0, 103.0, 104.0])

    allocation = allocator.allocate(signals, prices, pd.Timestamp("2023-01-05"))
    assert allocation == {symbol: 1.0}

def test_allocate_falls_back_when_insufficient_history(caplog):
    allocator = MeanVarianceAllocator(lookback_window=5)
    symbols = ["AAPL", "MSFT"]
    signals = pd.DataFrame({"symbol": symbols, "signal": [1, 1]})

    # AAPL has 5 days, MSFT only 3
    prices_aapl = make_price_series("AAPL", "2023-01-01", [100.0, 101.0, 102.0, 103.0, 104.0])
    prices_msft = make_price_series("MSFT", "2023-01-01", [200.0, 201.0, 202.0])
    prices = pd.concat([prices_aapl, prices_msft])

    allocation = allocator.allocate(signals, prices, pd.Timestamp("2023-01-05"))

    # Only AAPL is eligible, so it's < 2 eligible symbols -> equal weight fallback to {AAPL: 1.0}
    assert allocation == {"AAPL": 1.0}
    assert "Excluding symbols with insufficient price history" in caplog.text
    assert "Fewer than 2 eligible symbols" in caplog.text

def test_allocate_excludes_and_warns_for_insufficient_history_symbol(caplog):
    allocator = MeanVarianceAllocator(lookback_window=5)
    symbols = ["AAPL", "MSFT", "GOOG"]
    signals = pd.DataFrame({"symbol": symbols, "signal": [1, 1, 1]})

    prices_aapl = make_price_series("AAPL", "2023-01-01", [100.0, 101.0, 102.0, 103.0, 104.0])
    prices_msft = make_price_series("MSFT", "2023-01-01", [200.0, 201.0, 202.0, 203.0, 204.0])
    prices_goog = make_price_series("GOOG", "2023-01-01", [1000.0, 1001.0])
    prices = pd.concat([prices_aapl, prices_msft, prices_goog])

    allocation = allocator.allocate(signals, prices, pd.Timestamp("2023-01-05"))

    # AAPL and MSFT are eligible. GOOG is not.
    assert "GOOG" not in allocation
    assert "Excluding symbols with insufficient price history: ['GOOG']" in caplog.text

def test_allocate_returns_weights_summing_to_one():
    allocator = MeanVarianceAllocator(lookback_window=30)
    symbols = ["AAPL", "MSFT", "GOOG"]
    signals = pd.DataFrame({"symbol": symbols, "signal": [1, 1, 1]})

    prices_list = []
    for i, s in enumerate(symbols):
        rng = np.random.default_rng(i)
        closes = np.cumprod(1 + rng.normal(0, 0.01, 30)).tolist()
        prices_list.append(make_price_series(s, "2023-01-01", closes))
    prices = pd.concat(prices_list)

    allocation = allocator.allocate(signals, prices, pd.Timestamp("2023-02-01"))
    assert sum(allocation.values()) == pytest.approx(1.0, abs=1e-6)

def test_allocate_all_weights_non_negative():
    allocator = MeanVarianceAllocator(lookback_window=30)
    symbols = ["AAPL", "MSFT", "GOOG"]
    signals = pd.DataFrame({"symbol": symbols, "signal": [1, 1, 1]})

    prices_list = []
    for s in symbols:
        closes = np.cumprod(1 + np.random.normal(0, 0.01, 30)).tolist()
        prices_list.append(make_price_series(s, "2023-01-01", closes))
    prices = pd.concat(prices_list)

    allocation = allocator.allocate(signals, prices, pd.Timestamp("2023-02-01"))
    for weight in allocation.values():
        assert weight >= 0

def test_allocate_only_includes_eligible_symbols():
    allocator = MeanVarianceAllocator(lookback_window=30)
    buy_symbols = ["AAPL", "MSFT", "GOOG", "TSLA"]
    signals = pd.DataFrame({"symbol": buy_symbols, "signal": [1, 1, 1, 1]})

    prices_list = []
    # Only 3 symbols have enough history
    for s in ["AAPL", "MSFT", "GOOG"]:
        closes = np.cumprod(1 + np.random.normal(0, 0.01, 30)).tolist()
        prices_list.append(make_price_series(s, "2023-01-01", closes))
    # TSLA has insufficient history
    prices_list.append(make_price_series("TSLA", "2023-01-01", [100.0, 101.0]))
    prices = pd.concat(prices_list)

    allocation = allocator.allocate(signals, prices, pd.Timestamp("2023-02-01"))
    assert set(allocation.keys()).issubset(set(buy_symbols))
    assert "TSLA" not in allocation

def test_allocate_handles_optimization_failure_gracefully(caplog):
    allocator = MeanVarianceAllocator(lookback_window=5)
    symbols = ["AAPL", "MSFT"]
    signals = pd.DataFrame({"symbol": symbols, "signal": [1, 1]})

    # Constant prices cause zero volatility, which should fail in max_sharpe
    closes = [100.0, 100.0, 100.0, 100.0, 100.0]
    prices_aapl = make_price_series("AAPL", "2023-01-01", closes)
    prices_msft = make_price_series("MSFT", "2023-01-01", closes)
    prices = pd.concat([prices_aapl, prices_msft])

    # This should fail in PyPortfolioOpt and fall back to equal weighting
    allocation = allocator.allocate(signals, prices, pd.Timestamp("2023-01-05"))

    assert allocation == {"AAPL": 0.5, "MSFT": 0.5}
    assert "Mean-variance optimization failed" in caplog.text
