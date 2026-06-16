"""
Tests for AssetRepository.
"""

from src.database.models.asset import Asset


def make_asset(**overrides) -> dict:
    """
    Create a valid asset payload for testing.
    """

    base = {
        "symbol": "RELIANCE.NS",
        "name": "Reliance Industries",
        "asset_class": "equity",
        "exchange": "NSE",
        "currency": "INR",
        "sector": "Energy",
        "is_active": True,
    }
    base.update(overrides)
    return base


# ============================================================
# upsert()
# ============================================================

def test_upsert_inserts_new_asset(asset_repo):
    asset = asset_repo.upsert(make_asset())
    assert isinstance(asset, Asset)
    assert asset.symbol == "RELIANCE.NS"
    assert asset.exchange == "NSE"
    assert asset.name == "Reliance Industries"


def test_upsert_returns_asset_with_id(asset_repo):
    asset = asset_repo.upsert(make_asset())
    assert asset.id is not None


def test_upsert_updates_on_conflict(asset_repo):
    asset_repo.upsert(make_asset(name="Old Name"))
    updated = asset_repo.upsert(
        make_asset(name="New Name")
    )
    assert updated.name == "New Name"


# ============================================================
# get_by_symbol()
# ============================================================

def test_get_by_symbol_returns_inserted_asset(asset_repo):
    asset_repo.upsert(make_asset())
    assets = asset_repo.get_by_symbol("RELIANCE.NS")
    assert len(assets) == 1
    assert assets[0].symbol == "RELIANCE.NS"


def test_get_by_symbol_returns_empty_for_unknown(asset_repo):
    assets = asset_repo.get_by_symbol("UNKNOWN")
    assert assets == []


def test_get_by_symbol_returns_multiple_exchanges(asset_repo):
    asset_repo.upsert(
        make_asset(exchange="NSE")
    )
    asset_repo.upsert(
        make_asset(exchange="BSE")
    )
    assets = asset_repo.get_by_symbol("RELIANCE.NS")
    assert len(assets) == 2


# ============================================================
# get_by_asset_class()
# ============================================================

def test_get_by_asset_class_active_only_default(asset_repo):
    asset_repo.upsert(
        make_asset(symbol="ACTIVE.NS")
    )
    asset_repo.upsert(
        make_asset(
            symbol="INACTIVE.NS",
            is_active=False,
        )
    )

    assets = asset_repo.get_by_asset_class("equity")
    assert len(assets) == 1
    assert assets[0].symbol == "ACTIVE.NS"


def test_get_by_asset_class_returns_all_when_flag_false(asset_repo):
    asset_repo.upsert(
        make_asset(symbol="ACTIVE.NS")
    )

    asset_repo.upsert(
        make_asset(
            symbol="INACTIVE.NS",
            is_active=False,
        )
    )

    assets = asset_repo.get_by_asset_class(
        "equity",
        active_only=False,
    )
    assert len(assets) == 2


def test_get_by_asset_class_returns_empty(asset_repo):
    assets = asset_repo.get_by_asset_class(
        "crypto"
    )
    assert assets == []


# ============================================================
# bulk_insert()
# ============================================================

def test_bulk_insert_returns_count(asset_repo):
    records = [
        make_asset(symbol="A.NS"),
        make_asset(symbol="B.NS"),
        make_asset(symbol="C.NS"),
    ]

    count = asset_repo.bulk_insert(records)
    assert count == 3


def test_bulk_insert_empty_returns_zero(asset_repo):
    count = asset_repo.bulk_insert([])
    assert count == 0


# ============================================================
# delete_by_symbol()
# ============================================================

def test_delete_by_symbol_removes_records(asset_repo):
    asset_repo.upsert(make_asset())
    asset_repo.delete_by_symbol("RELIANCE.NS")
    assets = asset_repo.get_by_symbol("RELIANCE.NS")
    assert assets == []

def test_delete_by_symbol_returns_count(asset_repo):
    asset_repo.upsert(make_asset())
    deleted = asset_repo.delete_by_symbol("RELIANCE.NS")
    assert deleted == 1

def test_delete_by_symbol_returns_zero_for_unknown(asset_repo):
    deleted = asset_repo.delete_by_symbol(
        "UNKNOWN"
    )
    assert deleted == 0