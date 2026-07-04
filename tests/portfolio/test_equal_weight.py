import pytest
import pandas as pd
import logging
from src.portfolio.equal_weight import EqualWeightAllocator

def test_allocate_raises_on_empty_df():
    allocator = EqualWeightAllocator()
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="signals_df cannot be empty."):
        allocator.allocate(df)

def test_allocate_raises_on_missing_columns():
    allocator = EqualWeightAllocator()
    # Missing 'signal' column
    df = pd.DataFrame({"symbol": ["AAPL", "MSFT"]})
    with pytest.raises(ValueError, match="signals_df missing required columns"):
        allocator.allocate(df)

def test_allocate_equal_weights_for_two_buy_signals():
    allocator = EqualWeightAllocator()
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "signal": [1, 1]
    })
    allocation = allocator.allocate(df)
    assert allocation == {"AAPL": 0.5, "MSFT": 0.5}

def test_allocate_weights_sum_to_one():
    allocator = EqualWeightAllocator()
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    df = pd.DataFrame({
        "symbol": symbols,
        "signal": [1] * 5
    })
    allocation = allocator.allocate(df)
    assert sum(allocation.values()) == pytest.approx(1.0)
    assert all(w == 0.2 for w in allocation.values())

def test_allocate_single_buy_signal_gets_full_weight():
    allocator = EqualWeightAllocator()
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "signal": [1]
    })
    allocation = allocator.allocate(df)
    assert allocation == {"AAPL": 1.0}

def test_allocate_ignores_hold_and_sell_signals():
    allocator = EqualWeightAllocator()
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "signal": [1, 0, -1]
    })
    allocation = allocator.allocate(df)
    assert allocation == {"AAPL": 1.0}
    assert "MSFT" not in allocation
    assert "GOOGL" not in allocation

def test_allocate_returns_empty_dict_when_no_buy_signals():
    allocator = EqualWeightAllocator()
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "signal": [0, -1]
    })
    allocation = allocator.allocate(df)
    assert allocation == {}

def test_allocate_deduplicates_repeated_symbol():
    allocator = EqualWeightAllocator()
    # Simulation of a data glitch where the same symbol appears twice with a BUY signal
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "signal": [1, 1, 1]
    })
    allocation = allocator.allocate(df)
    # Unique symbols are AAPL and MSFT, so each should get 0.5
    assert len(allocation) == 2
    assert allocation == {"AAPL": 0.5, "MSFT": 0.5}
    assert sum(allocation.values()) == pytest.approx(1.0)

def test_allocate_logs_warning_when_no_buy_signals(caplog):
    allocator = EqualWeightAllocator()
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "signal": [0]
    })
    with caplog.at_level(logging.WARNING):
        allocator.allocate(df)
    assert "No BUY signals found; returning empty allocation." in caplog.text
