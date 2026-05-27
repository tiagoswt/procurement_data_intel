import pytest
from app import _db_row_to_internal_dict


def _make_row(overrides=None):
    base = {
        "ean": "3433422404397",
        "current_stock": 126,
        "sales90d": 1710,
        "sales180d": 3200,
        "sales365d": 6000,
        "best_buy_price": 9.20,
        "avg_stock_price": 9.10,
        "qnt_pending": 0,
        "supplier_price": 9.50,
        "bestseller_rank": 1,
        "brand": "L'Oréal",
        "description": "Serum 30ml",
    }
    if overrides:
        base.update(overrides)
    return base


def test_db_row_to_internal_dict_maps_column_names():
    result = _db_row_to_internal_dict(_make_row())
    assert result["ean"] == "3433422404397"
    assert result["stock"] == 126
    assert result["stock_avg_price"] == 9.10
    assert result["qntPendingToDeliver"] == 0
    assert result["supplier_price"] == 9.50
    assert result["bestseller_rank"] == 1
    assert result["brand"] == "L'Oréal"
    assert result["description"] == "Serum 30ml"


def test_db_row_to_internal_dict_derives_is_bestseller_true():
    result = _db_row_to_internal_dict(_make_row({"bestseller_rank": 2}))
    assert result["is_bestseller"] is True


def test_db_row_to_internal_dict_derives_is_bestseller_false():
    result = _db_row_to_internal_dict(_make_row({"bestseller_rank": 3}))
    assert result["is_bestseller"] is False


def test_db_row_to_internal_dict_handles_null_bestseller_rank():
    result = _db_row_to_internal_dict(_make_row({"bestseller_rank": None}))
    assert result["is_bestseller"] is False
    assert result["bestseller_rank"] is None


def test_db_row_to_internal_dict_defaults_missing_fields():
    result = _db_row_to_internal_dict(_make_row())
    assert result["cnp"] == ""
    assert result["capacity"] == ""
    assert result["sales_next90d_lastyear"] == 0
    assert result["best_supplier"] == ""
    assert result["is_active"] is True
