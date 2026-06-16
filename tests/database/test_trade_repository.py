"""
Tests for TradeRepository.
"""

from datetime import datetime, timezone

from src.database.models.trade import Trade


def make_trade(**overrides) -> dict:
    """
    Create a valid trade payload for testing.
    """

    base = {
        "symbol": "RELIANCE.NS",
        "entry_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "exit_date": datetime(2024, 1, 5, tzinfo=timezone.utc),
        "entry_price": 2500.0,
        "exit_price": 2600.0,
        "position_size": 100.0,
        "fees": 25.0,
        "slippage_cost": 12.5,
        "pnl": 9962.5,
        "return_pct": 0.0385,
    }

    base.update(overrides)
    return base


# ============================================================
# insert()
# ============================================================

def test_insert_creates_new_trade(trade_repo):
    trade = trade_repo.insert(make_trade())

    assert isinstance(trade, Trade)
    assert trade.symbol == "RELIANCE.NS"
    assert trade.entry_price == 2500.0
    assert trade.pnl == 9962.5


def test_insert_returns_trade_with_id(trade_repo):
    trade = trade_repo.insert(make_trade())

    assert trade.id is not None


def test_insert_allows_duplicate_symbol_and_date(trade_repo):
    trade_repo.insert(make_trade())

    trade_repo.insert(make_trade())

    trades = trade_repo.get_by_symbol("RELIANCE.NS")

    assert len(trades) == 2


# ============================================================
# get_by_symbol()
# ============================================================

def test_get_by_symbol_returns_inserted_trade(trade_repo):
    trade_repo.insert(make_trade())

    trades = trade_repo.get_by_symbol("RELIANCE.NS")

    assert len(trades) == 1


def test_get_by_symbol_returns_empty_for_unknown(trade_repo):
    trades = trade_repo.get_by_symbol("UNKNOWN")

    assert trades == []


def test_get_by_symbol_ordered_by_entry_date_asc(trade_repo):
    trade_repo.insert(
        make_trade(
            entry_date=datetime(
                2024,
                1,
                3,
                tzinfo=timezone.utc,
            )
        )
    )

    trade_repo.insert(
        make_trade(
            entry_date=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
    )

    trades = trade_repo.get_by_symbol("RELIANCE.NS")

    assert trades[0].entry_date < trades[1].entry_date


# ============================================================
# get_by_date_range()
# ============================================================

def test_get_by_date_range_returns_correct_records(trade_repo):
    trade_repo.insert(
        make_trade(
            entry_date=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
    )

    trade_repo.insert(
        make_trade(
            entry_date=datetime(
                2024,
                1,
                2,
                tzinfo=timezone.utc,
            )
        )
    )

    trade_repo.insert(
        make_trade(
            entry_date=datetime(
                2024,
                1,
                3,
                tzinfo=timezone.utc,
            )
        )
    )

    trades = trade_repo.get_by_date_range(
        symbol="RELIANCE.NS",
        start_date=datetime(
            2024,
            1,
            2,
            tzinfo=timezone.utc,
        ),
        end_date=datetime(
            2024,
            1,
            2,
            tzinfo=timezone.utc,
        ),
    )

    assert len(trades) == 1
    assert trades[0].entry_date == datetime(
        2024,
        1,
        2,
        tzinfo=timezone.utc,
    )


def test_get_by_date_range_returns_empty(trade_repo):
    trades = trade_repo.get_by_date_range(
        symbol="RELIANCE.NS",
        start_date=datetime(
            2025,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        end_date=datetime(
            2025,
            1,
            31,
            tzinfo=timezone.utc,
        ),
    )

    assert trades == []


# ============================================================
# bulk_insert()
# ============================================================

def test_bulk_insert_returns_count(trade_repo):
    records = [
        make_trade(symbol="A.NS"),
        make_trade(symbol="B.NS"),
        make_trade(symbol="C.NS"),
    ]

    count = trade_repo.bulk_insert(records)

    assert count == 3


def test_bulk_insert_empty_returns_zero(trade_repo):
    count = trade_repo.bulk_insert([])

    assert count == 0


# ============================================================
# delete_by_symbol()
# ============================================================

def test_delete_by_symbol_removes_records(trade_repo):
    trade_repo.insert(make_trade())

    trade_repo.delete_by_symbol("RELIANCE.NS")

    trades = trade_repo.get_by_symbol("RELIANCE.NS")

    assert trades == []


def test_delete_by_symbol_returns_count(trade_repo):
    trade_repo.insert(make_trade())

    trade_repo.insert(
        make_trade(
            exit_date=datetime(
                2024,
                1,
                6,
                tzinfo=timezone.utc,
            )
        )
    )

    deleted = trade_repo.delete_by_symbol("RELIANCE.NS")

    assert deleted == 2


def test_delete_by_symbol_returns_zero_for_unknown(trade_repo):
    deleted = trade_repo.delete_by_symbol("UNKNOWN")

    assert deleted == 0