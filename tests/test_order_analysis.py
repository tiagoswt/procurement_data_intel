import csv as csv_mod
import pytest
from scripts.order_analysis import score_opportunities, write_csvs

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _row(**kwargs):
    base = {
        "supplier": "hispalbeauty",
        "ean": "111",
        "brand": "Eucerin",
        "description": "Sun Cream",
        "current_stock": 10,
        "sales_90d": 100,
        "avg_stock_price": 12.0,
        "supplier_price_net": 10.0,
    }
    base.update(kwargs)
    return base


SAMPLE_ROWS = [
    # Type 1: net_need = 90 (sales90d=100, stock=10)
    _row(ean="001", current_stock=10, sales_90d=100, avg_stock_price=12.0, supplier_price_net=10.0),
    # Type 2: saving = 25%, stock sufficient (sales90d=50, stock=200)
    _row(ean="002", current_stock=200, sales_90d=50, avg_stock_price=12.0, supplier_price_net=9.0),
    # Outlier: price > 10x avg — should be excluded
    _row(ean="003", supplier="copedis", avg_stock_price=8.0, supplier_price_net=9_153_124_999.0),
    # Below threshold: 10% saving, stock sufficient — excluded from Type 2
    _row(ean="004", current_stock=500, sales_90d=100, avg_stock_price=5.0, supplier_price_net=4.5),
]


# ---------------------------------------------------------------------------
# Type 1 — NEED
# ---------------------------------------------------------------------------

def test_type1_classified_as_need():
    result = score_opportunities(SAMPLE_ROWS, threshold=15.0)
    need_rows = result["hispalbeauty"]["need"]
    assert any(r["ean"] == "001" for r in need_rows)
    assert all(r["opportunity_type"] == "NEED" for r in need_rows)


def test_type1_net_need_correct():
    result = score_opportunities(SAMPLE_ROWS, threshold=15.0)
    row = next(r for r in result["hispalbeauty"]["need"] if r["ean"] == "001")
    assert row["net_need"] == 90


def test_type1_sorted_by_net_need_desc():
    rows = [
        _row(ean="A", current_stock=0, sales_90d=10, supplier_price_net=10.0),   # net_need=10
        _row(ean="B", current_stock=0, sales_90d=50, supplier_price_net=10.0),   # net_need=50
        _row(ean="C", current_stock=0, sales_90d=200, supplier_price_net=10.0),  # net_need=200
    ]
    result = score_opportunities(rows, threshold=15.0)
    need_rows = result["hispalbeauty"]["need"]
    assert [r["ean"] for r in need_rows] == ["C", "B", "A"]


# ---------------------------------------------------------------------------
# Type 2 — PRICE
# ---------------------------------------------------------------------------

def test_type2_classified_as_price():
    result = score_opportunities(SAMPLE_ROWS, threshold=15.0)
    price_rows = result["hispalbeauty"]["price"]
    assert any(r["ean"] == "002" for r in price_rows)
    assert all(r["opportunity_type"] == "PRICE" for r in price_rows)


def test_type2_saving_pct_correct():
    result = score_opportunities(SAMPLE_ROWS, threshold=15.0)
    row = next(r for r in result["hispalbeauty"]["price"] if r["ean"] == "002")
    assert row["saving_pct"] == 25.0
    assert row["saving_eur_per_unit"] == 3.0


def test_type2_sorted_by_saving_pct_desc():
    rows = [
        _row(ean="X", current_stock=500, sales_90d=10, avg_stock_price=10.0, supplier_price_net=8.0),  # 20%
        _row(ean="Y", current_stock=500, sales_90d=10, avg_stock_price=10.0, supplier_price_net=6.0),  # 40%
        _row(ean="Z", current_stock=500, sales_90d=10, avg_stock_price=10.0, supplier_price_net=7.0),  # 30%
    ]
    result = score_opportunities(rows, threshold=15.0)
    price_rows = result["hispalbeauty"]["price"]
    assert [r["ean"] for r in price_rows] == ["Y", "Z", "X"]


def test_type2_below_threshold_excluded():
    result = score_opportunities(SAMPLE_ROWS, threshold=15.0)
    price_rows = result["hispalbeauty"]["price"]
    assert all(r["ean"] != "004" for r in price_rows)


def test_custom_threshold_25_excludes_25pct_item():
    # EAN 002 has exactly 25% saving — excluded when threshold=26
    result = score_opportunities(SAMPLE_ROWS, threshold=26.0)
    price_rows = result.get("hispalbeauty", {}).get("price", [])
    assert all(r["ean"] != "002" for r in price_rows)


def test_type1_item_not_duplicated_in_type2():
    # EAN 001 has net_need > 0 AND saving >= 15% — should only be in NEED
    result = score_opportunities(SAMPLE_ROWS, threshold=15.0)
    price_rows = result["hispalbeauty"]["price"]
    assert all(r["ean"] != "001" for r in price_rows)


# ---------------------------------------------------------------------------
# Data quality filters
# ---------------------------------------------------------------------------

def test_outlier_price_excluded():
    result = score_opportunities(SAMPLE_ROWS, threshold=15.0)
    copedis = result.get("copedis", {})
    assert len(copedis.get("need", [])) == 0
    assert len(copedis.get("price", [])) == 0


def test_zero_price_excluded():
    rows = [_row(ean="Z", supplier_price_net=0.0)]
    result = score_opportunities(rows, threshold=15.0)
    all_rows = result.get("hispalbeauty", {}).get("need", []) + \
               result.get("hispalbeauty", {}).get("price", [])
    assert all(r["ean"] != "Z" for r in all_rows)


def test_zero_avg_stock_price_excluded_from_type2():
    # avg_stock_price=0 means we can't compute saving — exclude from Type 2
    rows = [_row(ean="Z", current_stock=500, sales_90d=10, avg_stock_price=0.0, supplier_price_net=5.0)]
    result = score_opportunities(rows, threshold=15.0)
    price_rows = result.get("hispalbeauty", {}).get("price", [])
    assert len(price_rows) == 0


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def test_write_csvs_creates_file(tmp_path):
    suppliers = {
        "hispalbeauty": {
            "need": [{"opportunity_type": "NEED", "ean": "111", "brand": "B",
                      "description": "D", "current_stock": 10, "sales_90d": 100,
                      "net_need": 90, "avg_stock_price": 12.0,
                      "supplier_price_net": 10.0, "saving_pct": 16.7,
                      "saving_eur_per_unit": 2.0, "suggested_qty": 90}],
            "price": [],
        }
    }
    written = write_csvs(suppliers, str(tmp_path))
    assert len(written) == 1
    supplier, filepath, n_need, n_price = written[0]
    assert supplier == "hispalbeauty"
    assert n_need == 1
    assert n_price == 0
    with open(filepath, newline="") as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["ean"] == "111"
    assert rows[0]["opportunity_type"] == "NEED"


def test_write_csvs_need_before_price(tmp_path):
    suppliers = {
        "test_supplier": {
            "need": [{"opportunity_type": "NEED", "ean": "N1", "brand": "",
                      "description": "", "current_stock": 0, "sales_90d": 10,
                      "net_need": 10, "avg_stock_price": 10.0,
                      "supplier_price_net": 8.0, "saving_pct": 20.0,
                      "saving_eur_per_unit": 2.0, "suggested_qty": 10}],
            "price": [{"opportunity_type": "PRICE", "ean": "P1", "brand": "",
                       "description": "", "current_stock": 200, "sales_90d": 10,
                       "net_need": -190, "avg_stock_price": 10.0,
                       "supplier_price_net": 8.0, "saving_pct": 20.0,
                       "saving_eur_per_unit": 2.0, "suggested_qty": 10}],
        }
    }
    written = write_csvs(suppliers, str(tmp_path))
    filepath = written[0][1]
    with open(filepath, newline="") as f:
        rows = list(csv_mod.DictReader(f))
    assert rows[0]["opportunity_type"] == "NEED"
    assert rows[1]["opportunity_type"] == "PRICE"


def test_write_csvs_skips_empty_supplier(tmp_path):
    suppliers = {
        "empty_supplier": {"need": [], "price": []},
        "has_data": {
            "need": [{"opportunity_type": "NEED", "ean": "111", "brand": "",
                      "description": "", "current_stock": 0, "sales_90d": 10,
                      "net_need": 10, "avg_stock_price": 10.0,
                      "supplier_price_net": 8.0, "saving_pct": 20.0,
                      "saving_eur_per_unit": 2.0, "suggested_qty": 10}],
            "price": [],
        },
    }
    written = write_csvs(suppliers, str(tmp_path))
    assert len(written) == 1
    assert written[0][0] == "has_data"


# ---------------------------------------------------------------------------
# suggested_qty
# ---------------------------------------------------------------------------

def test_need_suggested_qty_equals_net_need():
    rows = [_row(ean="001", current_stock=10, sales_90d=100, supplier_price_net=10.0)]
    result = score_opportunities(rows, threshold=15.0)
    row = result["hispalbeauty"]["need"][0]
    assert row["suggested_qty"] == 90  # net_need = 100 - 10


def test_price_suggested_qty_equals_sales_90d():
    rows = [_row(ean="002", current_stock=200, sales_90d=50, avg_stock_price=12.0, supplier_price_net=9.0)]
    result = score_opportunities(rows, threshold=15.0)
    row = result["hispalbeauty"]["price"][0]
    assert row["suggested_qty"] == 50  # sales_90d


def test_type2_excluded_when_no_sales():
    # saving_pct=25%, stock sufficient, but zero sales — should NOT be a PRICE opportunity
    rows = [_row(ean="Z", current_stock=200, sales_90d=0, avg_stock_price=12.0, supplier_price_net=9.0)]
    result = score_opportunities(rows, threshold=15.0)
    price_rows = result.get("hispalbeauty", {}).get("price", [])
    assert len(price_rows) == 0
