import pytest
from pathlib import Path
from db.database import ProcurementDB
from analysis.analytics_engine import AnalyticsEngine
from models import ProductData


@pytest.fixture
def db_with_data(tmp_path):
    db = ProcurementDB(db_path=tmp_path / "test.db")
    products = [
        ProductData(ean_code="3433422404397", supplier="SupA", price=8.20, quantity=100),
        ProductData(ean_code="3433422404397", supplier="SupB", price=9.00, quantity=50),
        ProductData(ean_code="1234567890123", supplier="SupA", price=12.50, quantity=80),
    ]
    internal = [
        {"ean": "3433422404397", "description": "Serum 30ml", "brand": "L'Oréal",
         "stock": 126, "sales90d": 1710, "sales180d": 3200, "sales365d": 6000,
         "best_buy_price": 9.20, "stock_avg_price": 9.10, "qntPendingToDeliver": 0},
        {"ean": "1234567890123", "description": "Cream 50ml", "brand": "Vichy",
         "stock": 300, "sales90d": 450, "sales180d": 900, "sales365d": 1800,
         "best_buy_price": 13.00, "stock_avg_price": 12.80, "qntPendingToDeliver": 20},
    ]
    run_id = db.save_supplier_batch(["test.xlsx"], products)
    db.save_internal_data(run_id, internal)
    return db, internal


def test_compute_price_trends_returns_dataframe(db_with_data):
    db, _ = db_with_data
    engine = AnalyticsEngine(db)
    df = engine.compute_price_trends()
    assert not df.empty
    assert "ean" in df.columns
    assert "supplier" in df.columns
    assert "price_net" in df.columns


def test_compute_price_trends_filters_by_ean(db_with_data):
    db, _ = db_with_data
    engine = AnalyticsEngine(db)
    df = engine.compute_price_trends(ean="3433422404397")
    assert all(df["ean"] == "3433422404397")
    assert len(df) == 2  # two suppliers


def test_compute_stockout_risk_sorted_ascending(db_with_data):
    db, internal = db_with_data
    engine = AnalyticsEngine(db)
    df = engine.compute_stockout_risk(internal)
    assert not df.empty
    assert list(df["days_cover"]) == sorted(df["days_cover"].tolist())


def test_compute_stockout_risk_urgency_bands(db_with_data):
    db, _ = db_with_data
    internal = [
        {"ean": "AAA", "description": "X", "brand": "B",
         "stock": 5, "sales90d": 90, "qntPendingToDeliver": 0},    # 5 days -> Critical
        {"ean": "BBB", "description": "Y", "brand": "B",
         "stock": 100, "sales90d": 90, "qntPendingToDeliver": 0},  # 100 days -> OK
    ]
    engine = AnalyticsEngine(db)
    df = engine.compute_stockout_risk(internal)
    urgencies = dict(zip(df["ean"], df["urgency"]))
    assert urgencies["AAA"] == "Critical"
    assert urgencies["BBB"] == "OK"


def test_compute_supplier_win_rates_returns_dataframe(db_with_data):
    db, _ = db_with_data
    engine = AnalyticsEngine(db)
    df = engine.compute_supplier_win_rates()
    assert not df.empty
    assert "supplier" in df.columns
    assert "win_rate_pct" in df.columns
    assert "avg_price_index" in df.columns
    assert "price_stability" in df.columns


def test_compute_supplier_win_rates_cheapest_wins(db_with_data):
    db, _ = db_with_data
    engine = AnalyticsEngine(db)
    df = engine.compute_supplier_win_rates()
    # SupA has price 8.20 for EAN 3433422404397 (cheapest) and 12.50 for 1234567890123
    # SupB has price 9.00 for EAN 3433422404397 (not cheapest)
    # SupA should have higher win rate
    sup_a = df[df["supplier"] == "SupA"]["win_rate_pct"].iloc[0]
    sup_b = df[df["supplier"] == "SupB"]["win_rate_pct"].iloc[0]
    assert sup_a > sup_b


def test_compute_brand_health_returns_dataframe(db_with_data):
    db, internal = db_with_data
    engine = AnalyticsEngine(db)
    df = engine.compute_brand_health(internal)
    assert not df.empty
    assert "brand" in df.columns
    assert "sku_count" in df.columns
    assert "at_risk_skus" in df.columns
    assert "coverage_pct" in df.columns
