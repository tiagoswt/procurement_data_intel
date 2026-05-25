# Generate Orders Skill — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a `/generate-orders` Claude Code custom slash command that queries `procurement.db`, scores purchase opportunities per supplier, and writes one ranked CSV per supplier to `output/orders/YYYY-MM-DD/`.

**Architecture:** A standalone Python analysis script (`scripts/order_analysis.py`) contains all query and scoring logic, tested independently. A thin skill file (`.claude/commands/generate-orders.md`) instructs Claude to call that script with optional arguments and interpret the results. Output lands in `output/orders/<date>/` with one CSV per supplier.

**Tech Stack:** Python 3, sqlite3 (stdlib), csv (stdlib), pytest

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `scripts/__init__.py` | Makes `scripts` importable for tests |
| Create | `scripts/order_analysis.py` | DB query, outlier filter, scoring, CSV writing |
| Create | `tests/test_order_analysis.py` | Unit tests for scoring and CSV writing logic |
| Create | `.claude/commands/generate-orders.md` | Skill file: instructs Claude to run the script |
| Create | `output/orders/.gitkeep` | Scaffolds output directory |
| Modify | `.gitignore` | Ignore generated CSV files, keep `.gitkeep` |

---

## Task 1: Scaffold output directory and gitignore

**Files:**
- Create: `output/orders/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create output directory with gitkeep**

```bash
mkdir -p output/orders
echo "" > output/orders/.gitkeep
```

- [ ] **Step 2: Add output/orders CSVs to .gitignore**

Add these lines to `.gitignore`:

```
# Generated order CSVs
output/orders/*/
```

- [ ] **Step 3: Commit**

```bash
git add output/orders/.gitkeep .gitignore
git commit -m "chore: scaffold output/orders directory"
```

---

## Task 2: Write analysis script with tests

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/order_analysis.py`
- Create: `tests/test_order_analysis.py`

### Step 1: Create scripts package

- [ ] **Create `scripts/__init__.py`** (empty file)

```bash
mkdir -p scripts
touch scripts/__init__.py
```

### Step 2: Write the analysis script

- [ ] **Create `scripts/order_analysis.py`** with this exact content:

```python
import argparse
import csv
import os
import re
import sqlite3
from collections import defaultdict
from datetime import date

DB_PATH = "procurement.db"
COLUMNS = [
    "opportunity_type", "ean", "brand", "description",
    "current_stock", "sales_90d", "net_need",
    "avg_stock_price", "supplier_price_net",
    "saving_pct", "saving_eur_per_unit",
]


def load_data(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT
            sp.supplier,
            sp.ean,
            COALESCE(p.brand, '') AS brand,
            COALESCE(p.description, '') AS description,
            COALESCE(idat.current_stock, 0) AS current_stock,
            COALESCE(idat.sales90d, 0) AS sales_90d,
            COALESCE(idat.avg_stock_price, 0) AS avg_stock_price,
            sp.price_net AS supplier_price_net
        FROM supplier_prices sp
        JOIN processing_runs pr ON sp.run_id = pr.id
        JOIN (
            SELECT sp2.ean, sp2.supplier, MAX(pr2.run_at) AS max_run_at
            FROM supplier_prices sp2
            JOIN processing_runs pr2 ON sp2.run_id = pr2.id
            GROUP BY sp2.ean, sp2.supplier
        ) latest ON sp.ean = latest.ean
                   AND sp.supplier = latest.supplier
                   AND pr.run_at = latest.max_run_at
        JOIN (
            SELECT idat2.ean, idat2.current_stock, idat2.sales90d, idat2.avg_stock_price
            FROM internal_data idat2
            JOIN processing_runs pr3 ON idat2.run_id = pr3.id
            WHERE pr3.run_at = (SELECT MAX(run_at) FROM processing_runs)
        ) idat ON sp.ean = idat.ean
        LEFT JOIN (
            SELECT ean, MAX(rowid) AS rid FROM products GROUP BY ean
        ) lp ON sp.ean = lp.ean
        LEFT JOIN products p ON p.rowid = lp.rid
        WHERE sp.price_net > 0
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def score_opportunities(rows: list, threshold: float) -> dict:
    suppliers = defaultdict(lambda: {"need": [], "price": []})

    for r in rows:
        supplier = r["supplier"]
        avg_price = r["avg_stock_price"] or 0.0
        sup_price = r["supplier_price_net"]
        sales_90d = r["sales_90d"]
        stock = r["current_stock"]
        net_need = sales_90d - stock

        if sup_price <= 0:
            continue
        if avg_price > 0 and sup_price > avg_price * 10:
            continue

        saving_pct = 0.0
        saving_eur = 0.0
        if avg_price > 0:
            saving_pct = (avg_price - sup_price) / avg_price * 100
            saving_eur = avg_price - sup_price

        row_data = {
            "ean": r["ean"],
            "brand": r["brand"],
            "description": r["description"],
            "current_stock": stock,
            "sales_90d": sales_90d,
            "net_need": net_need,
            "avg_stock_price": round(avg_price, 2),
            "supplier_price_net": round(sup_price, 2),
            "saving_pct": round(saving_pct, 1),
            "saving_eur_per_unit": round(saving_eur, 2),
        }

        if net_need > 0:
            row_data["opportunity_type"] = "NEED"
            suppliers[supplier]["need"].append(row_data)
        elif avg_price > 0 and saving_pct >= threshold:
            row_data["opportunity_type"] = "PRICE"
            suppliers[supplier]["price"].append(row_data)

    for sup_data in suppliers.values():
        sup_data["need"].sort(key=lambda x: (-x["net_need"], -x["saving_pct"]))
        sup_data["price"].sort(key=lambda x: -x["saving_pct"])

    return dict(suppliers)


def write_csvs(suppliers: dict, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for supplier, data in suppliers.items():
        rows_out = data["need"] + data["price"]
        if not rows_out:
            continue
        safe_name = re.sub(r"[^\w\-]", "_", supplier)
        filepath = os.path.join(output_dir, f"{safe_name}.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows_out)
        written.append((supplier, filepath, len(data["need"]), len(data["price"])))
    return written


def print_summary(written: list, suppliers: dict) -> None:
    print(f"\nGenerated {len(written)} order file(s):")
    for supplier, filepath, n_need, n_price in written:
        print(f"  {supplier}: {n_need} NEED + {n_price} PRICE -> {filepath}")

    all_opps = []
    for supplier, data in suppliers.items():
        for r in data["need"] + data["price"]:
            all_opps.append({**r, "supplier": supplier})

    top_need = sorted(
        [r for r in all_opps if r["opportunity_type"] == "NEED"],
        key=lambda x: -x["net_need"],
    )[:3]
    top_price = sorted(
        [r for r in all_opps if r["opportunity_type"] == "PRICE"],
        key=lambda x: -x["saving_pct"],
    )[:3]

    if top_need:
        print("\nTop 3 NEED opportunities:")
        for r in top_need:
            print(
                f"  [{r['supplier']}] {r['description'] or r['ean']}"
                f" — need {r['net_need']} units, saves {r['saving_pct']}%"
            )
    if top_price:
        print("\nTop 3 PRICE opportunities:")
        for r in top_price:
            print(
                f"  [{r['supplier']}] {r['description'] or r['ean']}"
                f" — saves {r['saving_pct']}% (EUR {r['saving_eur_per_unit']}/unit)"
            )


def main():
    parser = argparse.ArgumentParser(description="Generate procurement order opportunity CSVs")
    parser.add_argument(
        "--threshold", type=float, default=15.0,
        help="Minimum saving %% for PRICE opportunities (default: 15)"
    )
    parser.add_argument("--db", default=DB_PATH, help="Path to procurement.db")
    args = parser.parse_args()

    today = date.today().isoformat()
    output_dir = os.path.join("output", "orders", today)

    print(f"Loading data from {args.db}...")
    rows = load_data(args.db)
    print(f"  {len(rows)} matched product-supplier rows loaded")

    ventas_in_db = any(r["supplier"].lower() == "ventas" for r in rows)
    if ventas_in_db:
        print("  WARNING: ventas prices are stored in USD — compare with caution")

    print(f"Scoring opportunities (threshold={args.threshold}%)...")
    suppliers = score_opportunities(rows, args.threshold)

    print(f"Writing CSVs to {output_dir}/...")
    written = write_csvs(suppliers, output_dir)
    print_summary(written, suppliers)


if __name__ == "__main__":
    main()
```

### Step 3: Write the tests

- [ ] **Create `tests/test_order_analysis.py`** with this exact content:

```python
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
                      "saving_eur_per_unit": 2.0}],
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
                      "saving_eur_per_unit": 2.0}],
            "price": [{"opportunity_type": "PRICE", "ean": "P1", "brand": "",
                       "description": "", "current_stock": 200, "sales_90d": 10,
                       "net_need": -190, "avg_stock_price": 10.0,
                       "supplier_price_net": 8.0, "saving_pct": 20.0,
                       "saving_eur_per_unit": 2.0}],
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
                      "saving_eur_per_unit": 2.0}],
            "price": [],
        },
    }
    written = write_csvs(suppliers, str(tmp_path))
    assert len(written) == 1
    assert written[0][0] == "has_data"
```

- [ ] **Step 4: Run the tests and verify they all pass**

```bash
pytest tests/test_order_analysis.py -v
```

Expected output: all 18 tests PASS. If any fail, fix `scripts/order_analysis.py` before continuing.

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/order_analysis.py tests/test_order_analysis.py
git commit -m "feat: add order analysis script with tests"
```

---

## Task 3: Write the skill file

**Files:**
- Create: `.claude/commands/generate-orders.md`

- [ ] **Step 1: Create `.claude/commands/` directory**

```bash
mkdir -p .claude/commands
```

- [ ] **Step 2: Create `.claude/commands/generate-orders.md`** with this exact content:

````markdown
# Generate Purchase Order Opportunities

You are acting as a procurement agent. Your job is to analyse the database and identify the best purchase opportunities for each supplier, then write ranked CSV files the team can use to place orders.

## Arguments

`$ARGUMENTS` — optional flags passed by the user. Supported:
- `--threshold N` — minimum saving % for PRICE opportunities (default: 15)

## Steps

1. **Run the analysis script:**

```bash
python scripts/order_analysis.py $ARGUMENTS
```

2. **Report the results clearly:**
   - List every CSV file generated with its path
   - For each supplier, state how many NEED and PRICE opportunities were found
   - Highlight the top 3 NEED opportunities (highest urgency — most units short)
   - Highlight the top 3 PRICE opportunities (highest saving %)
   - Call out any data quality warnings printed by the script (e.g., ventas USD pricing)

3. **Provide a brief procurement interpretation:** Based on the numbers, which supplier(s) should the team prioritize this week? Which products are most at risk of stockout?

## If the script fails

Diagnose the error, fix it (edit `scripts/order_analysis.py` if needed), re-run, and then report results. Do not report failure without attempting to resolve it.
````

- [ ] **Step 3: Verify the skill file is in the right place**

```bash
ls .claude/commands/
```

Expected: `generate-orders.md` is listed.

- [ ] **Step 4: Commit**

```bash
git add .claude/commands/generate-orders.md
git commit -m "feat: add /generate-orders custom slash command"
```

---

## Task 4: Smoke test against live database

**Files:** none modified

- [ ] **Step 1: Run the script against the real DB**

```bash
python scripts/order_analysis.py --threshold 15
```

Expected:
- Prints row count (should be ~8,000–10,000 matched rows)
- Warns about ventas USD pricing
- Writes CSV files to `output/orders/<today>/`
- Prints per-supplier counts and top 3 opportunities

- [ ] **Step 2: Spot-check a generated CSV**

```bash
python -c "
import csv
with open('output/orders/$(python -c \"from datetime import date; print(date.today().isoformat())\")/hispalbeauty.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Total rows: {len(rows)}')
need = [r for r in rows if r['opportunity_type'] == 'NEED']
price = [r for r in rows if r['opportunity_type'] == 'PRICE']
print(f'NEED: {len(need)}, PRICE: {len(price)}')
print('First NEED row:', need[0] if need else 'none')
print('First PRICE row:', price[0] if price else 'none')
"
```

Expected: NEED rows appear before PRICE rows, saving_pct values are reasonable (0–60%), no corrupted prices.

- [ ] **Step 3: Test the threshold argument**

```bash
python scripts/order_analysis.py --threshold 30
```

Expected: fewer PRICE rows than with `--threshold 15`.

- [ ] **Step 4: Commit smoke test results note**

```bash
git add output/orders/.gitkeep
git commit -m "test: verify /generate-orders skill end-to-end against live DB"
```
