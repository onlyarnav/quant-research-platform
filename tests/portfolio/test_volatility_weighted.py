import pytest
import pandas as pd
import numpy as np
import logging
from src.portfolio.volatility_weighted import VolatilityWeightedAllocator

def make_price_series(symbol: str, start_date: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=len(closes))
    return pd.DataFrame({"symbol": symbol, "date": dates, "close": closes})

def test_init_rejects_lookback_below_2():
    with pytest.raises(ValueError, match="lookback_window must be at least 2"):
        VolatilityWeightedAllocator(lookback_window=1)

def test_init_default_lookback_is_20():
    allocator = VolatilityWeightedAllocator()
    assert allocator.lookback_window == 20

def test_allocate_raises_on_empty_signals_df():
    allocator = VolatilityWeightedAllocator()
    df_signals = pd.DataFrame()
    df_prices = pd.DataFrame({"symbol": ["AAPL"], "date": [pd.Timestamp("2023-01-01")], "close": [100.0]})
    with pytest.raises(ValueError, match="signals_df cannot be empty."):
        allocator.allocate(df_signals, df_prices, pd.Timestamp("2023-01-01"))

def test_allocate_raises_on_empty_prices_df():
    allocator = VolatilityWeightedAllocator()
    df_signals = pd.DataFrame({"symbol": ["AAPL"], "signal": [1]})
    df_prices = pd.DataFrame()
    with pytest.raises(ValueError, match="prices_df cannot be empty."):
        allocator.allocate(df_signals, df_prices, pd.Timestamp("2023-01-01"))

def test_allocate_raises_on_missing_signal_columns():
    allocator = VolatilityWeightedAllocator()
    df_signals = pd.DataFrame({"symbol": ["AAPL"]}) # Missing 'signal'
    df_prices = pd.DataFrame({"symbol": ["AAPL"], "date": [pd.Timestamp("2023-01-01")], "close": [100.0]})
    with pytest.raises(ValueError, match="signals_df missing required columns"):
        allocator.allocate(df_signals, df_prices, pd.Timestamp("2023-01-01"))

def test_allocate_raises_on_missing_price_columns():
    allocator = VolatilityWeightedAllocator()
    df_signals = pd.DataFrame({"symbol": ["AAPL"], "signal": [1]})
    df_prices = pd.DataFrame({"symbol": ["AAPL"], "close": [100.0]}) # Missing 'date'
    with pytest.raises(ValueError, match="prices_df missing required columns"):
        allocator.allocate(df_signals, df_prices, pd.Timestamp("2023-01-01"))

def test_allocate_lower_volatility_gets_higher_weight():
    # Low vol symbol: very slight changes
    # High vol symbol: large swings
    allocator = VolatilityWeightedAllocator(lookback_window=5)
    as_of = pd.Timestamp("2023-01-10")

    # 6 prices = 5 returns
    low_vol_closes = [100.0, 100.1, 99.9, 100.0, 100.1, 100.0]
    high_vol_closes = [100.0, 110.0, 90.0, 110.0, 90.0, 100.0]

    prices = pd.concat([
        make_price_series("LOW", "2023-01-01", low_vol_closes),
        make_price_series("HIGH", "2023-01-01", high_vol_closes)
    ])

    signals = pd.DataFrame({"symbol": ["LOW", "HIGH"], "signal": [1, 1]})

    allocation = allocator.allocate(signals, prices, as_of)
    assert allocation["LOW"] > allocation["HIGH"]
    assert sum(allocation.values()) == pytest.approx(1.0)


def test_allocate_weights_sum_to_one():
    allocator = VolatilityWeightedAllocator(lookback_window=5)
    as_of = pd.Timestamp("2023-01-10")

    symbols = ["S1", "S2", "S3"]
    price_dfs = []
    for i, s in enumerate(symbols):
        rng = np.random.default_rng(i)
        closes = (100 + rng.standard_normal(6) * 2).tolist()
        price_dfs.append(make_price_series(s, "2023-01-01", closes))

    prices = pd.concat(price_dfs)
    signals = pd.DataFrame({"symbol": symbols, "signal": [1] * 3})

    allocation = allocator.allocate(signals, prices, as_of)
    assert sum(allocation.values()) == pytest.approx(1.0)

def test_allocate_returns_empty_dict_when_no_buy_signals():
    allocator = VolatilityWeightedAllocator()
    as_of = pd.Timestamp("2023-01-01")
    signals = pd.DataFrame({"symbol": ["AAPL"], "signal": [0]})
    prices = make_price_series("AAPL", "2022-12-01", [100.0]*30)

    allocation = allocator.allocate(signals, prices, as_of)
    assert allocation == {}

def test_allocate_ignores_non_buy_signals():
    allocator = VolatilityWeightedAllocator(lookback_window=5)
    as_of = pd.Timestamp("2023-01-10")

    prices = pd.concat([
        make_price_series("BUY", "2023-01-01", [100, 101, 100, 101, 100, 101]),
        make_price_series("NOT_BUY", "2023-01-01", [100, 101, 100, 101, 100, 101])
    ])

    signals = pd.DataFrame({"symbol": ["BUY", "NOT_BUY"], "signal": [1, 0]})

    allocation = allocator.allocate(signals, prices, as_of)
    assert "BUY" in allocation
    assert "NOT_BUY" not in allocation
    assert allocation["BUY"] == 1.0

def test_allocate_excludes_symbol_with_insufficient_history():
    allocator = VolatilityWeightedAllocator(lookback_window=10)
    as_of = pd.Timestamp("2023-01-20")

    # S1: plenty of history
    # S2: only 5 days of history (need 11)
    prices = pd.concat([
        make_price_series("S1", "2023-01-01", [100.0]*20),
        make_price_series("S2", "2023-01-15", [100.0]*5)
    ])
    # Need to vary S1 prices so vol != 0
    prices.loc[prices['symbol'] == 'S1', 'close'] = np.random.randn(20) + 100

    signals = pd.DataFrame({"symbol": ["S1", "S2"], "signal": [1, 1]})

    allocation = allocator.allocate(signals, prices, as_of)
    assert "S1" in allocation
    assert "S2" not in allocation
    assert allocation["S1"] == 1.0

def test_allocate_excludes_symbol_with_zero_volatility(caplog):
    allocator = VolatilityWeightedAllocator(lookback_window=5)
    as_of = pd.Timestamp("2023-01-10")

    # S1: Volatile
    # S2: Perfectly flat (zero vol)
    prices = pd.concat([
        make_price_series("S1", "2023-01-01", [100, 110, 90, 110, 90, 100]),
        make_price_series("S2", "2023-01-01", [100, 100, 100, 100, 100, 100])
    ])

    signals = pd.DataFrame({"symbol": ["S1", "S2"], "signal": [1, 1]})

    with caplog.at_level(logging.WARNING):
        allocation = allocator.allocate(signals, prices, as_of)

    assert "S1" in allocation
    assert "S2" not in allocation
    assert "zero or undefined volatility" in caplog.text
    assert allocation["S1"] == 1.0

def test_allocate_returns_empty_when_all_symbols_excluded():
    allocator = VolatilityWeightedAllocator(lookback_window=20)
    as_of = pd.Timestamp("2023-01-01")

    # All symbols have insufficient history
    prices = pd.concat([
        make_price_series("S1", "2023-01-01", [100.0]*5),
        make_price_series("S2", "2023-01-01", [100.0]*5)
    ])

    signals = pd.DataFrame({"symbol": ["S1", "S2"], "signal": [1, 1]})

    allocation = allocator.allocate(signals, prices, as_of)
    assert allocation == {}

def test_allocate_respects_as_of_date_boundary():
    allocator = VolatilityWeightedAllocator(lookback_window=2)
    as_of = pd.Timestamp("2023-01-03")

    # Construct a price series where volatility changes drastically after as_of_date
    # Prices:
    # 01-01: 100
    # 01-02: 101
    # 01-03: 102 (Volatility at this point is low)
    # 01-04: 200 (Volatility would jump if this was included)
    # 01-05: 50

    closes = [100.0, 101.0, 102.0, 200.0, 50.0]
    prices = make_price_series("S1", "2023-01-01", closes)
    signals = pd.DataFrame({"symbol": ["S1"], "signal": [1]})

    # Case 1: as_of = 2023-01-03. Should use [100, 101, 102]
    alloc_1 = allocator.allocate(signals, prices, as_of)

    # Case 2: as_of = 2023-01-05. Should use [102, 200, 50] (tail(3))
    alloc_2 = allocator.allocate(signals, prices, pd.Timestamp("2023-01-05"))

    # We can't easily compare weights since there's only 1 symbol (both will be 1.0),
    # but we can verify that the logic uses the correct subset of data by
    # checking if the allocator runs and we can verify the volatility logic
    # separately or by using 2 symbols.

    # Let's use 2 symbols to see if the boundary affects the RELATIVE weights.
    S2_closes = [100.0, 100.0, 100.0, 100.0, 100.0] # Flat
    # To make S2 have non-zero vol, we give it slight variation
    S2_closes = [100.0, 100.1, 100.0, 100.1, 100.0]

    prices_combined = pd.concat([
        make_price_series("S1", "2023-01-01", closes),
        make_price_series("S2", "2023-01-01", S2_closes)
    ])
    signals_combined = pd.DataFrame({"symbol": ["S1", "S2"], "signal": [1, 1]})

    # At 2023-01-03:
    # S1 is [100, 101, 102] -> low vol
    # S2 is [100, 100.1, 100] -> very low vol
    # Both are relatively calm.
    alloc_early = allocator.allocate(signals_combined, prices_combined, as_of)

    # At 2023-01-05:
    # S1 is [102, 200, 50] -> massive vol
    # S2 is [100, 100.1, 100] -> very low vol
    # S2 should dominate.
    alloc_late = allocator.allocate(signals_combined, prices_combined, pd.Timestamp("2023-01-05"))

    assert alloc_late["S2"] > alloc_early["S2"]
