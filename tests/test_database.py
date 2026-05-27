import pytest
import tempfile
from datetime import date, timedelta
from pathlib import Path
from db.database import ProcurementDB
from models import ProductData


@pytest.fixture
def tmp_db(tmp_path):
    return ProcurementDB(db_path=tmp_path / "test.db")


@pytest.fixture
def sample_products():
    return [
        ProductData(ean_code="3433422404397", supplier="SupA", price=8.20, quantity=100),
        ProductData(ean_code="1234567890123", supplier="SupA", price=12.50, quantity=50),
        ProductData(ean_code="3433422404397", supplier="SupB", price=8.80, quantity=200),
    ]


@pytest.fixture
def sample_internal():
    return [
        {"ean": "3433422404397", "description": "Serum 30ml", "brand": "L'Oréal",
         "stock": 126, "sales90d": 1710, "sales180d": 3200, "sales365d": 6000,
         "best_buy_price": 9.20, "stock_avg_price": 9.10, "qntPendingToDeliver": 0},
        {"ean": "1234567890123", "description": "Cream 50ml", "brand": "Vichy",
         "stock": 300, "sales90d": 450, "sales180d": 900, "sales365d": 1800,
         "best_buy_price": 13.00, "stock_avg_price": 12.80, "qntPendingToDeliver": 20},
    ]


def test_init_creates_tables(tmp_db):
    import sqlite3
    conn = sqlite3.connect(tmp_db.db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "processing_runs" in tables
    assert "supplier_prices" in tables
    assert "internal_data" in tables
    assert "products" in tables


def test_save_supplier_batch_returns_run_id(tmp_db, sample_products):
    run_id = tmp_db.save_supplier_batch(["file_a.xlsx"], sample_products)
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_save_supplier_batch_stores_prices(tmp_db, sample_products):
    tmp_db.save_supplier_batch(["file_a.xlsx"], sample_products)
    rows = tmp_db.get_all_supplier_prices()
    assert len(rows) == 3
    eans = {r["ean"] for r in rows}
    assert "3433422404397" in eans


def test_save_internal_data(tmp_db, sample_products, sample_internal):
    run_id = tmp_db.save_supplier_batch(["file_a.xlsx"], sample_products)
    tmp_db.save_internal_data(run_id, sample_internal)
    data = tmp_db.get_latest_internal_data()
    assert len(data) == 2
    eans = {r["ean"] for r in data}
    assert "3433422404397" in eans


def test_get_price_history_filters_by_ean(tmp_db, sample_products):
    tmp_db.save_supplier_batch(["file_a.xlsx"], sample_products)
    rows = tmp_db.get_price_history(ean="3433422404397")
    assert all(r["ean"] == "3433422404397" for r in rows)
    assert len(rows) == 2  # two suppliers have this EAN


def test_get_price_history_filters_by_days(tmp_db, sample_products):
    tmp_db.save_supplier_batch(["file_a.xlsx"], sample_products)
    rows = tmp_db.get_price_history(days=1)
    assert len(rows) == 3  # all within last 1 day


def test_get_runs_returns_list(tmp_db, sample_products):
    tmp_db.save_supplier_batch(["file_a.xlsx"], sample_products)
    runs = tmp_db.get_runs()
    assert len(runs) == 1
    assert "run_at" in runs[0]
    assert "source_files" in runs[0]


def test_get_latest_internal_data_ignores_runs_without_internal_data(tmp_db, sample_products, sample_internal):
    # First run: supplier batch + internal data
    run_id_1 = tmp_db.save_supplier_batch(["file_a.xlsx"], sample_products)
    tmp_db.save_internal_data(run_id_1, sample_internal)

    # Second run: supplier batch only (no internal data attached)
    tmp_db.save_supplier_batch(["file_b.xlsx"], sample_products)

    # Should still return the internal data from run 1, not empty
    data = tmp_db.get_latest_internal_data()
    assert len(data) == 2, f"Expected 2 rows, got {len(data)}"
    eans = {r["ean"] for r in data}
    assert "3433422404397" in eans


def test_get_all_supplier_prices_active_only_excludes_expired(tmp_db):
    products_exp  = [ProductData(ean_code="1111111111111", supplier="SupExp",  price=5.0, quantity=10)]
    products_act  = [ProductData(ean_code="2222222222222", supplier="SupAct",  price=6.0, quantity=10)]
    products_none = [ProductData(ean_code="3333333333333", supplier="SupNone", price=7.0, quantity=10)]

    tmp_db.save_supplier_batch(["exp.csv"],  products_exp,  validity_days={"SupExp":  -1})
    tmp_db.save_supplier_batch(["act.csv"],  products_act,  validity_days={"SupAct":  30})
    tmp_db.save_supplier_batch(["none.csv"], products_none)

    rows = tmp_db.get_all_supplier_prices(active_only=True)
    eans = {r["ean"] for r in rows}
    assert "1111111111111" not in eans, "Expired offer should be excluded"
    assert "2222222222222" in eans,     "Active offer should be included"
    assert "3333333333333" in eans,     "No-expiry offer should be included"


def test_get_all_supplier_prices_default_includes_expired(tmp_db):
    products_exp = [ProductData(ean_code="1111111111111", supplier="SupExp", price=5.0, quantity=10)]
    tmp_db.save_supplier_batch(["exp.csv"], products_exp, validity_days={"SupExp": -1})

    rows = tmp_db.get_all_supplier_prices()
    eans = {r["ean"] for r in rows}
    assert "1111111111111" in eans, "Default (active_only=False) should include expired"
