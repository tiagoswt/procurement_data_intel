# Analytics Depth + SQLite Persistence Bridge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLite persistence, an Analytics Dashboard tab (4 sub-tabs), dual opportunity scoring, and enhanced exports to the existing Streamlit POC without breaking any existing functionality.

**Architecture:** A new `db/` package holds `database.py` (SQLite CRUD) and `exporters.py` (Excel/HTML/JSON). `analysis/analytics_engine.py` reads from the DB to compute trends and scorecards. `analysis/enhanced_scoring.py` appends `enhanced_score` and `score_breakdown` to the existing opportunities list without touching `opportunity_engine.py`. The new `tabs/analytics_tab.py` orchestrates four sub-tabs. Auto-save hooks are added in `app.py` (supplier prices) and `tabs/opportunities_tab.py` (internal data).

**Tech Stack:** Python 3.12, Streamlit, pandas, Plotly, openpyxl, sqlite3 (stdlib), pytest — all already in `requirements.txt`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `db/__init__.py` | Create | Package marker |
| `db/database.py` | Create | SQLite CRUD — schema, save_supplier_batch, save_internal_data, query helpers |
| `db/exporters.py` | Create | Excel (5-sheet), HTML report, JSON payload |
| `analysis/analytics_engine.py` | Create | price trends, stockout risk, supplier win rates, brand health |
| `analysis/enhanced_scoring.py` | Create | Appends enhanced_score + score_breakdown to opportunities list |
| `tabs/analytics_tab.py` | Create | 4 sub-tabs: Price Trends, Stockout Risk, Supplier Scorecard, Brand Health |
| `tabs/opportunities_tab.py` | Modify | Add enhanced_score column + sort controls |
| `app.py` | Modify | Add Analytics tab, auto-save supplier batch after processing |
| `tests/test_database.py` | Create | DB CRUD tests |
| `tests/test_analytics_engine.py` | Create | Analytics computation tests |
| `tests/test_enhanced_scoring.py` | Create | Scoring formula tests |
| `tests/test_exporters.py` | Create | Export output format tests |

---

## Task 1: SQLite Database Layer

**Files:**
- Create: `db/__init__.py`
- Create: `db/database.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: Create `db/__init__.py`**

```python
# db/__init__.py
```

- [ ] **Step 2: Write the failing tests for `db/database.py`**

Create `tests/test_database.py`:

```python
import pytest
import tempfile
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
```

- [ ] **Step 3: Run tests — verify they fail**

```
pytest tests/test_database.py -v
```
Expected: `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 4: Create `db/database.py`**

```python
import sqlite3
import uuid
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "procurement.db"


class ProcurementDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS processing_runs (
                    id           TEXT PRIMARY KEY,
                    run_at       TIMESTAMP NOT NULL,
                    source_files TEXT,
                    row_count    INT,
                    notes        TEXT
                );
                CREATE TABLE IF NOT EXISTS products (
                    ean          TEXT NOT NULL,
                    description  TEXT,
                    brand        TEXT,
                    pack_size    INT,
                    run_id       TEXT REFERENCES processing_runs(id)
                );
                CREATE TABLE IF NOT EXISTS supplier_prices (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ean          TEXT NOT NULL,
                    supplier     TEXT NOT NULL,
                    price_net    REAL,
                    currency     TEXT DEFAULT 'EUR',
                    quantity     INT,
                    run_id       TEXT REFERENCES processing_runs(id),
                    ingested_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_sp_ean_ingested
                    ON supplier_prices (ean, ingested_at DESC);
                CREATE TABLE IF NOT EXISTS internal_data (
                    ean               TEXT NOT NULL,
                    run_id            TEXT REFERENCES processing_runs(id),
                    current_stock     INT,
                    sales90d          INT,
                    sales180d         INT,
                    sales365d         INT,
                    best_buy_price    REAL,
                    avg_stock_price   REAL,
                    qnt_pending       INT
                );
            """)

    def save_supplier_batch(self, source_files: List[str], products) -> str:
        """Persist a supplier processing batch. Returns the new run_id."""
        run_id = str(uuid.uuid4())
        run_at = datetime.utcnow().isoformat()
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO processing_runs (id, run_at, source_files, row_count) VALUES (?, ?, ?, ?)",
                    (run_id, run_at, json.dumps(source_files), len(products)),
                )
                for p in products:
                    ean = str(getattr(p, "ean_code", "") or "").strip()
                    if not ean:
                        ean = str(getattr(p, "supplier_code", "") or "").strip()
                    if not ean:
                        continue
                    qty = getattr(p, "quantity", None)
                    conn.execute(
                        "INSERT INTO supplier_prices (ean, supplier, price_net, currency, quantity, run_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            ean,
                            str(getattr(p, "supplier", "Unknown") or "Unknown"),
                            float(getattr(p, "price", 0) or 0),
                            "EUR",
                            int(qty) if qty is not None else None,
                            run_id,
                        ),
                    )
            logger.info(f"Batch saved: run_id={run_id}, {len(products)} prices")
            return run_id
        except Exception as e:
            logger.error(f"Failed to save supplier batch: {e}")
            raise

    def save_internal_data(self, run_id: str, internal_products: List[Dict]) -> None:
        """Persist internal product data for an existing run."""
        try:
            with self._get_conn() as conn:
                for p in internal_products:
                    ean = p.get("ean")
                    if not ean:
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO products (ean, description, brand, run_id) VALUES (?, ?, ?, ?)",
                        (ean, p.get("description"), p.get("brand"), run_id),
                    )
                    conn.execute(
                        """INSERT INTO internal_data
                           (ean, run_id, current_stock, sales90d, sales180d, sales365d,
                            best_buy_price, avg_stock_price, qnt_pending)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ean, run_id,
                            p.get("stock", 0),
                            p.get("sales90d", 0),
                            p.get("sales180d", 0),
                            p.get("sales365d", 0),
                            p.get("best_buy_price"),
                            p.get("stock_avg_price"),
                            p.get("qntPendingToDeliver", 0),
                        ),
                    )
            logger.info(f"Internal data saved: {len(internal_products)} products for run {run_id}")
        except Exception as e:
            logger.error(f"Failed to save internal data: {e}")
            raise

    def get_runs(self) -> List[Dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, run_at, source_files, row_count FROM processing_runs ORDER BY run_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_price_history(
        self,
        ean: Optional[str] = None,
        brand: Optional[str] = None,
        supplier: Optional[str] = None,
        days: int = 90,
    ) -> List[Dict]:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = """
            SELECT sp.ean, sp.supplier, sp.price_net, pr.run_at,
                   p.brand, p.description
            FROM supplier_prices sp
            JOIN processing_runs pr ON sp.run_id = pr.id
            LEFT JOIN (
                SELECT ean, MAX(rowid) AS rid FROM products GROUP BY ean
            ) latest_p ON sp.ean = latest_p.ean
            LEFT JOIN products p ON p.rowid = latest_p.rid
            WHERE pr.run_at >= ?
        """
        params: List = [since]
        if ean:
            query += " AND sp.ean = ?"
            params.append(ean)
        if brand:
            query += " AND p.brand = ?"
            params.append(brand)
        if supplier:
            query += " AND sp.supplier = ?"
            params.append(supplier)
        query += " ORDER BY pr.run_at ASC"
        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_supplier_prices(self) -> List[Dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT sp.ean, sp.supplier, sp.price_net, sp.quantity, pr.run_at,
                          p.brand, p.description
                   FROM supplier_prices sp
                   JOIN processing_runs pr ON sp.run_id = pr.id
                   LEFT JOIN (
                       SELECT ean, MAX(rowid) AS rid FROM products GROUP BY ean
                   ) latest_p ON sp.ean = latest_p.ean
                   LEFT JOIN products p ON p.rowid = latest_p.rid
                   ORDER BY pr.run_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_internal_data(self) -> List[Dict]:
        with self._get_conn() as conn:
            latest = conn.execute(
                "SELECT id FROM processing_runs ORDER BY run_at DESC LIMIT 1"
            ).fetchone()
            if not latest:
                return []
            rows = conn.execute(
                """SELECT idat.*, p.brand, p.description
                   FROM internal_data idat
                   LEFT JOIN (
                       SELECT ean, MAX(rowid) AS rid FROM products GROUP BY ean
                   ) latest_p ON idat.ean = latest_p.ean
                   LEFT JOIN products p ON p.rowid = latest_p.rid
                   WHERE idat.run_id = ?""",
                (latest["id"],),
            ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 5: Run tests — verify they pass**

```
pytest tests/test_database.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add db/__init__.py db/database.py tests/test_database.py
git commit -m "feat: add SQLite persistence layer (ProcurementDB)"
```

---

## Task 2: Auto-save Supplier Batch in app.py

**Files:**
- Modify: `app.py` (function `process_manual_supplier_files`, lines ~668–702)

- [ ] **Step 1: Add DB import at the top of `app.py`**

Find the block of imports near line 23 in `app.py` and add after the existing imports:

```python
from db.database import ProcurementDB
```

- [ ] **Step 2: Add DB auto-save after successful processing**

In `process_manual_supplier_files()`, find the block at line ~668:

```python
    if all_products:
        # Store results in session state
        st.session_state.processed_data = all_products
        st.session_state.processing_results = all_results
```

Replace with:

```python
    if all_products:
        # Store results in session state
        st.session_state.processed_data = all_products
        st.session_state.processing_results = all_results

        # Auto-save to SQLite (additive — never blocks the UI if it fails)
        try:
            db = ProcurementDB()
            source_file_names = [f.name for f in uploaded_files]
            run_id = db.save_supplier_batch(source_file_names, all_products)
            st.session_state.current_run_id = run_id
        except Exception as _db_err:
            pass  # DB save is best-effort; processing still succeeds
```

- [ ] **Step 3: Verify the app still starts**

```
streamlit run app.py
```
Expected: app loads without errors, a `procurement.db` file appears in the project root after uploading and processing any supplier file.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: auto-save supplier batch to SQLite after processing"
```

---

## Task 3: Auto-save Internal Data in Opportunities Tab

**Files:**
- Modify: `tabs/opportunities_tab.py`

- [ ] **Step 1: Find the internal data load point**

In `tabs/opportunities_tab.py`, search for `engine.load_internal_data`. It will look like:

```python
if engine.load_internal_data(internal_file):
    st.success(...)
```

- [ ] **Step 2: Add DB save after successful internal data load**

Add the DB import at the top of `tabs/opportunities_tab.py` (after the existing imports):

```python
from db.database import ProcurementDB
```

Then, immediately after the `if engine.load_internal_data(internal_file):` success branch, add:

```python
                    # Persist internal data to SQLite
                    run_id = st.session_state.get("current_run_id")
                    if run_id and engine.internal_data:
                        try:
                            ProcurementDB().save_internal_data(run_id, engine.internal_data)
                        except Exception:
                            pass
```

- [ ] **Step 3: Verify the tab still works**

Start the app, upload a supplier file, go to Opportunities, upload internal data. Confirm no errors appear and the flow works as before.

- [ ] **Step 4: Commit**

```bash
git add tabs/opportunities_tab.py
git commit -m "feat: auto-save internal data to SQLite after loading"
```

---

## Task 4: Analytics Engine

**Files:**
- Create: `analysis/analytics_engine.py`
- Create: `tests/test_analytics_engine.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analytics_engine.py`:

```python
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
         "stock": 5, "sales90d": 90, "qntPendingToDeliver": 0},    # 5 days → Critical
        {"ean": "BBB", "description": "Y", "brand": "B",
         "stock": 100, "sales90d": 90, "qntPendingToDeliver": 0},  # 100 days → OK
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
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_analytics_engine.py -v
```
Expected: `ModuleNotFoundError: No module named 'analysis.analytics_engine'`

- [ ] **Step 3: Create `analysis/analytics_engine.py`**

```python
import pandas as pd
import logging
from collections import defaultdict
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    def __init__(self, db):
        self.db = db

    def compute_price_trends(
        self,
        ean: Optional[str] = None,
        brand: Optional[str] = None,
        supplier: Optional[str] = None,
        days: int = 90,
    ) -> pd.DataFrame:
        rows = self.db.get_price_history(ean=ean, brand=brand, supplier=supplier, days=days)
        if not rows:
            return pd.DataFrame(columns=["ean", "supplier", "price_net", "run_at", "brand", "description"])
        df = pd.DataFrame(rows)
        df["run_at"] = pd.to_datetime(df["run_at"])
        return df

    def compute_price_trend_alerts(self, threshold_pct: float = 5.0) -> List[Dict]:
        """Return list of (ean, supplier) pairs where price moved >= threshold_pct between last two batches."""
        rows = self.db.get_price_history(days=180)
        if not rows:
            return []
        df = pd.DataFrame(rows)
        df["run_at"] = pd.to_datetime(df["run_at"])
        alerts = []
        for (ean, supplier), grp in df.groupby(["ean", "supplier"]):
            grp = grp.sort_values("run_at")
            if len(grp) < 2:
                continue
            last = grp.iloc[-1]["price_net"]
            prev = grp.iloc[-2]["price_net"]
            if prev and prev > 0:
                pct = (last - prev) / prev * 100
                if abs(pct) >= threshold_pct:
                    alerts.append({
                        "ean": ean,
                        "supplier": supplier,
                        "brand": grp.iloc[-1].get("brand", ""),
                        "description": grp.iloc[-1].get("description", ""),
                        "prev_price": round(prev, 4),
                        "current_price": round(last, 4),
                        "pct_change": round(pct, 2),
                        "direction": "up" if pct > 0 else "down",
                    })
        return sorted(alerts, key=lambda x: abs(x["pct_change"]), reverse=True)

    def compute_stockout_risk(self, internal_data: List[Dict]) -> pd.DataFrame:
        if not internal_data:
            return pd.DataFrame()
        rows = []
        for p in internal_data:
            sales90d = p.get("sales90d", 0) or 0
            daily = sales90d / 90 if sales90d > 0 else 0
            stock = p.get("current_stock", p.get("stock", 0)) or 0
            days_cover = (stock / daily) if daily > 0 else 9999
            if days_cover < 7:
                urgency = "Critical"
            elif days_cover < 30:
                urgency = "Warning"
            else:
                urgency = "OK"
            rows.append({
                "ean": p.get("ean", ""),
                "description": p.get("description", ""),
                "brand": p.get("brand", ""),
                "current_stock": stock,
                "daily_sales": round(daily, 2),
                "days_cover": round(min(days_cover, 9999), 1),
                "urgency": urgency,
                "best_buy_price": p.get("best_buy_price"),
            })
        df = pd.DataFrame(rows)
        df = df[df["days_cover"] < 9999].copy()
        return df.sort_values("days_cover").reset_index(drop=True)

    def compute_supplier_win_rates(self) -> pd.DataFrame:
        rows = self.db.get_all_supplier_prices()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["run_at"] = pd.to_datetime(df["run_at"])
        min_prices = (
            df.groupby(["ean", "run_at"])["price_net"].min().reset_index()
        )
        min_prices.columns = ["ean", "run_at", "min_price"]
        df = df.merge(min_prices, on=["ean", "run_at"])
        df["is_winner"] = df["price_net"] <= df["min_price"] * 1.001
        df["price_index"] = df["price_net"] / df["min_price"].replace(0, float("nan"))
        win_stats = (
            df.groupby("supplier")
            .agg(
                total_competed=("ean", "count"),
                wins=("is_winner", "sum"),
                avg_price_index=("price_index", "mean"),
                sku_count=("ean", "nunique"),
            )
            .reset_index()
        )
        win_stats["win_rate_pct"] = (
            win_stats["wins"] / win_stats["total_competed"] * 100
        ).round(1)
        win_stats["avg_price_index"] = win_stats["avg_price_index"].round(3)
        df_sorted = df.sort_values(["supplier", "ean", "run_at"])
        df_sorted["price_change_pct"] = (
            df_sorted.groupby(["supplier", "ean"])["price_net"].pct_change() * 100
        )
        stability = (
            df_sorted.groupby("supplier")["price_change_pct"].std().reset_index()
        )
        stability.columns = ["supplier", "price_std_pct"]
        win_stats = win_stats.merge(stability, on="supplier", how="left")
        win_stats["price_std_pct"] = win_stats["price_std_pct"].fillna(0)
        win_stats["price_stability"] = win_stats["price_std_pct"].apply(
            lambda x: "High" if x < 2 else ("Medium" if x < 5 else "Low")
        )
        return win_stats.sort_values("win_rate_pct", ascending=False).reset_index(drop=True)

    def compute_brand_health(self, internal_data: List[Dict]) -> pd.DataFrame:
        if not internal_data:
            return pd.DataFrame()
        history = self.db.get_price_history(days=90)
        brand_products: Dict[str, List[Dict]] = defaultdict(list)
        for p in internal_data:
            brand = (p.get("brand") or "Unknown").strip() or "Unknown"
            brand_products[brand].append(p)
        # Latest EANs for coverage
        latest_eans: set = set()
        trend_map: Dict[str, float] = {}
        if history:
            df_h = pd.DataFrame(history)
            df_h["run_at"] = pd.to_datetime(df_h["run_at"])
            latest_batch = df_h["run_at"].max()
            latest_eans = set(df_h[df_h["run_at"] == latest_batch]["ean"].unique())
            min_per_batch = (
                df_h.groupby(["ean", "run_at"])["price_net"].min().reset_index()
            )
            for ean, grp in min_per_batch.groupby("ean"):
                grp = grp.sort_values("run_at")
                if len(grp) >= 2:
                    last = grp.iloc[-1]["price_net"]
                    prev = grp.iloc[-2]["price_net"]
                    if prev and prev > 0:
                        trend_map[ean] = (last - prev) / prev * 100
        rows = []
        for brand, products in brand_products.items():
            sku_count = len(products)
            brand_eans = {p.get("ean") for p in products}
            covered = len(brand_eans & latest_eans)
            coverage_pct = (covered / sku_count * 100) if sku_count > 0 else 0
            at_risk = 0
            for p in products:
                sales90d = p.get("sales90d", 0) or 0
                if sales90d > 0:
                    daily = sales90d / 90
                    stock = p.get("current_stock", p.get("stock", 0)) or 0
                    if stock / daily < 30:
                        at_risk += 1
            trends = [
                trend_map[p.get("ean")]
                for p in products
                if p.get("ean") in trend_map
            ]
            avg_trend = sum(trends) / len(trends) if trends else 0
            top = max(products, key=lambda p: p.get("sales90d", 0))
            rows.append({
                "brand": brand,
                "sku_count": sku_count,
                "price_trend_pct": round(avg_trend, 2),
                "at_risk_skus": at_risk,
                "coverage_pct": round(coverage_pct, 1),
                "top_opportunity": top.get("description") or top.get("ean", ""),
            })
        df = pd.DataFrame(rows)
        return df.sort_values("at_risk_skus", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_analytics_engine.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/analytics_engine.py tests/test_analytics_engine.py
git commit -m "feat: add AnalyticsEngine (price trends, stockout, win rates, brand health)"
```

---

## Task 5: Enhanced Scoring

**Files:**
- Create: `analysis/enhanced_scoring.py`
- Create: `tests/test_enhanced_scoring.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_enhanced_scoring.py`:

```python
import pytest
from analysis.enhanced_scoring import compute_enhanced_scores


@pytest.fixture
def sample_opps():
    return [
        {
            "ean": "AAA",
            "product_name": "Fast mover",
            "sales90d": 1800,   # 20/day — high velocity
            "days_of_cover": 6,  # near stockout — high urgency
            "quote_price": 8.00,
            "price_breakdown": {"best_buy_price": 9.20},
            "brand_weight": 1.2,
        },
        {
            "ean": "BBB",
            "product_name": "Slow mover overstocked",
            "sales90d": 90,    # 1/day — low velocity
            "days_of_cover": 365,  # massively overstocked
            "quote_price": 5.00,
            "price_breakdown": {"best_buy_price": 5.10},
            "brand_weight": 1.0,
        },
    ]


def test_compute_enhanced_scores_adds_keys(sample_opps):
    result = compute_enhanced_scores(sample_opps)
    for opp in result:
        assert "enhanced_score" in opp
        assert "score_breakdown" in opp


def test_scores_normalized_0_to_10(sample_opps):
    result = compute_enhanced_scores(sample_opps)
    for opp in result:
        assert 0.0 <= opp["enhanced_score"] <= 10.0


def test_fast_mover_scores_higher_than_overstocked(sample_opps):
    result = compute_enhanced_scores(sample_opps)
    scores = {o["ean"]: o["enhanced_score"] for o in result}
    assert scores["AAA"] > scores["BBB"]


def test_score_breakdown_contains_components(sample_opps):
    result = compute_enhanced_scores(sample_opps)
    breakdown = result[0]["score_breakdown"]
    assert "v×" in breakdown
    assert "m×" in breakdown
    assert "u×" in breakdown
    assert "b×" in breakdown


def test_overstock_penalty_applies_above_180_days(sample_opps):
    result = compute_enhanced_scores(sample_opps)
    breakdown_b = result[1]["score_breakdown"]
    assert "÷os×" in breakdown_b  # BBB has 365 days cover → penalty applied


def test_empty_list_returns_empty(sample_opps):
    result = compute_enhanced_scores([])
    assert result == []


def test_original_dicts_not_mutated(sample_opps):
    original_ids = [id(o) for o in sample_opps]
    result = compute_enhanced_scores(sample_opps)
    # original list elements should be untouched
    for o in sample_opps:
        assert "enhanced_score" not in o
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_enhanced_scoring.py -v
```
Expected: `ModuleNotFoundError: No module named 'analysis.enhanced_scoring'`

- [ ] **Step 3: Create `analysis/enhanced_scoring.py`**

```python
from typing import List, Dict


def compute_enhanced_scores(opportunities: List[Dict]) -> List[Dict]:
    """
    Appends enhanced_score (0–10) and score_breakdown string to copies of each
    opportunity dict. Does not mutate the originals.
    """
    if not opportunities:
        return []

    max_daily = max(
        (o.get("sales90d", 0) / 90 for o in opportunities), default=1
    )
    if max_daily <= 0:
        max_daily = 1

    raw_scores = []
    for opp in opportunities:
        daily = (opp.get("sales90d", 0) or 0) / 90
        velocity = daily / max_daily

        best_buy = (opp.get("price_breakdown") or {}).get("best_buy_price") or opp.get("baseline_price", 0)
        supplier_price = opp.get("quote_price", 0) or 0
        margin = max(1.0, best_buy / supplier_price) if supplier_price > 0 and best_buy else 1.0

        days_cover = opp.get("days_of_cover", 999) or 999
        urgency = max(1.0, 30 / max(1, days_cover))

        brand_weight = opp.get("brand_weight", 1.0) or 1.0

        overstock = max(1.0, days_cover / 180) if days_cover > 180 else 1.0

        raw = velocity * margin * urgency * brand_weight / overstock
        raw_scores.append({
            "raw": raw,
            "velocity": velocity,
            "margin": margin,
            "urgency": urgency,
            "brand_weight": brand_weight,
            "overstock": overstock,
        })

    max_raw = max((s["raw"] for s in raw_scores), default=1)
    if max_raw <= 0:
        max_raw = 1

    result = []
    for opp, sd in zip(opportunities, raw_scores):
        enhanced = opp.copy()
        normalized = round((sd["raw"] / max_raw) * 10, 1)
        enhanced["enhanced_score"] = normalized

        parts = [
            f"v×{sd['velocity']:.2f}",
            f"m×{sd['margin']:.2f}",
            f"u×{sd['urgency']:.2f}",
            f"b×{sd['brand_weight']:.1f}",
        ]
        if sd["overstock"] > 1.0:
            parts.append(f"÷os×{sd['overstock']:.2f}")
        enhanced["score_breakdown"] = " · ".join(parts)
        result.append(enhanced)

    return result
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_enhanced_scoring.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/enhanced_scoring.py tests/test_enhanced_scoring.py
git commit -m "feat: add enhanced opportunity scoring with breakdown string"
```

---

## Task 6: Extend Opportunities Tab with Dual Scoring

**Files:**
- Modify: `tabs/opportunities_tab.py`

- [ ] **Step 1: Add import at the top of `tabs/opportunities_tab.py`**

After the existing imports, add:

```python
from analysis.enhanced_scoring import compute_enhanced_scores
```

- [ ] **Step 2: Add brand weight sidebar control**

In `opportunities_tab()` (around the filter controls section), add a sidebar expander for brand weights. Find the section where filters are shown and add:

```python
    with st.sidebar.expander("⚡ Enhanced Scoring Weights", expanded=False):
        st.caption("Boost tier-1 brands in the enhanced score (1.0 = no boost)")
        brand_weights = {}
        if st.session_state.get("processed_data"):
            engine = st.session_state.get("opportunity_engine")
            if engine and engine.internal_data:
                brands = sorted({p.get("brand", "") for p in engine.internal_data if p.get("brand")})
                for brand in brands[:10]:  # cap at 10 to avoid sidebar overflow
                    brand_weights[brand] = st.slider(
                        brand, min_value=1.0, max_value=1.5, value=1.0, step=0.1,
                        key=f"bw_{brand}"
                    )
```

- [ ] **Step 3: Apply enhanced scoring and add sort control**

Find the section in `opportunities_tab()` where `opportunities` are displayed (after `engine.find_opportunities(...)` is called and the list is available). Add the following block immediately after the opportunities list is built:

```python
    # Apply enhanced scoring (additive — appends enhanced_score and score_breakdown)
    if opportunities:
        for opp in opportunities:
            opp["brand_weight"] = brand_weights.get(opp.get("brand", ""), 1.0)
        opportunities = compute_enhanced_scores(opportunities)

    # Sort control
    sort_col1, sort_col2 = st.columns([1, 3])
    with sort_col1:
        sort_mode = st.radio(
            "Sort by",
            options=["Current (P1/P2)", "Enhanced Score"],
            horizontal=True,
            key="opp_sort_mode",
        )
    if sort_mode == "Enhanced Score":
        opportunities = sorted(opportunities, key=lambda x: x.get("enhanced_score", 0), reverse=True)
    else:
        opportunities = sorted(opportunities, key=lambda x: (x.get("priority", 9), -x.get("total_savings", 0)))
```

- [ ] **Step 4: Add enhanced score columns to the display table**

Find the section where the opportunity dataframe/table is built (it creates a dict per row). Add two new columns to each row dict:

```python
            "Priority": opp.get("priority_label", ""),
            "Enh. Score": opp.get("enhanced_score", "-"),
            "Score Breakdown": opp.get("score_breakdown", ""),
```

Insert these after the existing `"Priority"` column in the row dict. The exact location depends on where the display dict is built — search for `"priority_label"` in the file.

- [ ] **Step 5: Manual smoke test**

Start the app, upload a supplier file, upload internal data in the Opportunities tab. Confirm:
- Both "Current (P1/P2)" and "Enh. Score" columns appear
- "Sort by Enhanced Score" re-orders the table
- Score breakdown text is visible inline

- [ ] **Step 6: Commit**

```bash
git add tabs/opportunities_tab.py
git commit -m "feat: add dual scoring columns and sort control to opportunities tab"
```

---

## Task 7: Export Layer

**Files:**
- Create: `db/exporters.py`
- Create: `tests/test_exporters.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_exporters.py`:

```python
import json
import pytest
import pandas as pd
from db.exporters import export_excel, export_json_payload, export_html_report


@pytest.fixture
def sample_opportunities():
    return [
        {
            "ean": "3433422404397", "product_name": "Serum 30ml", "brand": "L'Oréal",
            "priority_label": "🔥 Priority 1", "enhanced_score": 8.4,
            "score_breakdown": "v×1.80 · m×1.15 · u×2.10 · b×1.2",
            "net_need": 240, "quote_price": 8.20, "supplier": "BeautyDist",
            "savings_per_unit": 1.00, "total_savings": 240.0, "days_of_cover": 6.0,
        },
    ]


@pytest.fixture
def empty_df():
    return pd.DataFrame()


def test_export_excel_returns_bytes(sample_opportunities, empty_df):
    result = export_excel(sample_opportunities, empty_df, empty_df, empty_df, empty_df)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_export_excel_has_five_sheets(sample_opportunities, empty_df):
    import io
    import openpyxl
    data = export_excel(sample_opportunities, empty_df, empty_df, empty_df, empty_df)
    wb = openpyxl.load_workbook(io.BytesIO(data))
    assert set(wb.sheetnames) == {"Opportunities", "Price Trends", "Stockout Risk", "Suppliers", "Brands"}


def test_export_json_payload_has_required_keys(sample_opportunities):
    result = export_json_payload(sample_opportunities, [], [], [], "test-batch-id")
    payload = json.loads(result)
    required = {"period", "batch_id", "headline_metrics", "top_opportunities",
                "price_trend_alerts", "stockout_risks", "supplier_movements", "overstock_warnings"}
    assert required.issubset(payload.keys())


def test_export_json_payload_headline_metrics(sample_opportunities):
    result = export_json_payload(sample_opportunities, [], [], [], "test-batch-id")
    payload = json.loads(result)
    assert payload["headline_metrics"]["opportunities_count"] == 1
    assert payload["headline_metrics"]["estimated_savings_eur"] == 240.0


def test_export_html_report_is_string(sample_opportunities):
    result = export_html_report(sample_opportunities, [], [], [], "test-batch-id")
    assert isinstance(result, str)
    assert "<!DOCTYPE html>" in result


def test_export_html_report_contains_kpi_values(sample_opportunities):
    result = export_html_report(sample_opportunities, [], [], [], "test-batch-id")
    assert "240" in result   # total savings
    assert "BeautyDist" in result
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_exporters.py -v
```
Expected: `ModuleNotFoundError: No module named 'db.exporters'`

- [ ] **Step 3: Create `db/exporters.py`**

```python
import io
import json
from datetime import datetime
from typing import List, Dict

import pandas as pd


def export_excel(
    opportunities: List[Dict],
    price_trends_df: pd.DataFrame,
    stockout_df: pd.DataFrame,
    supplier_df: pd.DataFrame,
    brand_df: pd.DataFrame,
) -> bytes:
    """Returns a 5-sheet Excel workbook as bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if opportunities:
            opp_df = pd.DataFrame([{
                "EAN": o.get("ean"),
                "Product": o.get("product_name"),
                "Brand": o.get("brand"),
                "Priority": o.get("priority_label"),
                "Enhanced Score": o.get("enhanced_score"),
                "Score Breakdown": o.get("score_breakdown"),
                "Net Need": o.get("net_need"),
                "Best Price (€)": o.get("quote_price"),
                "Supplier": o.get("supplier"),
                "Savings/unit (€)": o.get("savings_per_unit"),
                "Total Savings (€)": o.get("total_savings"),
                "Days Cover": round(float(o.get("days_of_cover") or 0), 1),
            } for o in opportunities])
        else:
            opp_df = pd.DataFrame()
        opp_df.to_excel(writer, sheet_name="Opportunities", index=False)
        price_trends_df.to_excel(writer, sheet_name="Price Trends", index=False)
        stockout_df.to_excel(writer, sheet_name="Stockout Risk", index=False)
        supplier_df.to_excel(writer, sheet_name="Suppliers", index=False)
        brand_df.to_excel(writer, sheet_name="Brands", index=False)
    return buf.getvalue()


def export_json_payload(
    opportunities: List[Dict],
    price_trend_alerts: List[Dict],
    stockout_risks: List[Dict],
    supplier_movements: List[Dict],
    batch_id: str,
) -> str:
    """Returns LLM-ready JSON payload as string."""
    total_savings = sum(o.get("total_savings", 0) for o in opportunities)
    critical = [s for s in stockout_risks if (s.get("days_cover") or 999) < 7]
    overstocked = [o for o in opportunities if (o.get("days_of_cover") or 0) > 180]
    top_opps = sorted(opportunities, key=lambda x: x.get("enhanced_score", 0), reverse=True)[:10]
    payload = {
        "period": datetime.utcnow().date().isoformat(),
        "batch_id": batch_id,
        "headline_metrics": {
            "opportunities_count": len(opportunities),
            "estimated_savings_eur": round(total_savings, 2),
            "skus_at_stockout_risk": len(critical),
            "skus_overstocked": len(overstocked),
            "new_supplier_prices_ingested": len(opportunities),
        },
        "top_opportunities": [
            {
                "ean": o.get("ean"),
                "product": o.get("product_name"),
                "brand": o.get("brand"),
                "recommended_supplier": o.get("supplier"),
                "recommended_qty": o.get("net_need"),
                "unit_price": o.get("quote_price"),
                "score": o.get("enhanced_score"),
                "score_breakdown": o.get("score_breakdown"),
                "days_of_cover": round(float(o.get("days_of_cover") or 0), 1),
                "priority": o.get("priority_label"),
            }
            for o in top_opps
        ],
        "price_trend_alerts": price_trend_alerts,
        "stockout_risks": [
            {
                "ean": s.get("ean"),
                "product": s.get("description"),
                "days_cover": round(float(s.get("days_cover") or 0), 1),
                "daily_sales": round(float(s.get("daily_sales") or 0), 1),
                "best_supplier": s.get("best_supplier"),
            }
            for s in stockout_risks[:20]
        ],
        "supplier_movements": supplier_movements[:10],
        "overstock_warnings": [
            {
                "ean": o.get("ean"),
                "product": o.get("product_name"),
                "days_cover": round(float(o.get("days_of_cover") or 0), 1),
            }
            for o in overstocked[:10]
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def export_html_report(
    opportunities: List[Dict],
    stockout_risks: List[Dict],
    supplier_win_rates: List[Dict],
    brand_alerts: List[Dict],
    batch_id: str,
) -> str:
    """Returns self-contained HTML report as string (no external dependencies)."""
    total_savings = sum(o.get("total_savings", 0) for o in opportunities)
    critical_count = sum(1 for s in stockout_risks if (s.get("days_cover") or 999) < 7)
    top_opps = sorted(opportunities, key=lambda x: x.get("enhanced_score", 0), reverse=True)[:10]

    # Inline SVG bar chart for supplier win rates
    svg_bars = ""
    if supplier_win_rates:
        max_rate = max(s.get("win_rate_pct", 0) for s in supplier_win_rates)
        for i, s in enumerate(supplier_win_rates[:6]):
            rate = s.get("win_rate_pct", 0)
            bar_w = int((rate / max(max_rate, 1)) * 200)
            y = i * 30 + 10
            svg_bars += f'<rect x="130" y="{y}" width="{bar_w}" height="20" fill="#4CAF50"/>'
            svg_bars += f'<text x="125" y="{y+15}" text-anchor="end" font-size="11">{str(s.get("supplier",""))[:16]}</text>'
            svg_bars += f'<text x="{130+bar_w+5}" y="{y+15}" font-size="11">{rate:.0f}%</text>'
    svg_h = max(200, len(supplier_win_rates[:6]) * 30 + 20)
    supplier_svg = f'<svg width="400" height="{svg_h}" style="background:#f5f5f5;border-radius:6px;padding:8px">{svg_bars}</svg>'

    opp_rows = "".join(
        f"<tr><td>{o.get('ean','')}</td><td>{o.get('product_name','')}</td>"
        f"<td>{o.get('priority_label','')}</td><td><strong>{o.get('enhanced_score','-')}</strong></td>"
        f"<td>{o.get('net_need',0)}</td><td>€{o.get('quote_price',0):.2f}</td>"
        f"<td>{o.get('supplier','')}</td><td style='color:green'>€{o.get('total_savings',0):.2f}</td></tr>"
        for o in top_opps
    )
    stockout_rows = "".join(
        f"<tr><td>{s.get('ean','')}</td><td>{s.get('description','')}</td>"
        f"<td style='color:{'#ff4444' if (s.get('days_cover',999)<7) else '#ff8c00'};font-weight:bold'>{s.get('days_cover',0):.0f}d</td>"
        f"<td>{s.get('best_supplier','')}</td></tr>"
        for s in stockout_risks[:5]
    )
    brand_rows = "".join(
        f"<tr><td>{b.get('brand','')}</td>"
        f"<td style='color:{'#ff4444' if b.get('price_trend_pct',0)>1 else ('#4CAF50' if b.get('price_trend_pct',0)<-1 else '#888')}'>"
        f"{'↑' if b.get('price_trend_pct',0)>1 else ('↓' if b.get('price_trend_pct',0)<-1 else '→')} {abs(b.get('price_trend_pct',0)):.1f}%</td>"
        f"<td>{b.get('at_risk_skus',0)}</td><td>{b.get('coverage_pct',0):.0f}%</td></tr>"
        for b in brand_alerts[:5]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Procurement Intelligence Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:0 auto;padding:20px;color:#333}}
h1{{color:#1a237e}} h2{{color:#1a237e;border-bottom:2px solid #e0e0e0;padding-bottom:8px;margin-top:32px}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}}
.kpi{{background:#f5f7ff;border-radius:8px;padding:16px;text-align:center}}
.kpi .val{{font-size:28px;font-weight:bold;color:#1a237e}} .kpi .label{{font-size:12px;color:#666}}
table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:13px}}
th{{background:#1a237e;color:white;padding:8px;text-align:left}}
td{{padding:6px 8px;border-bottom:1px solid #eee}}
.footer{{color:#999;font-size:11px;margin-top:40px}}
</style></head>
<body>
<h1>📊 Procurement Intelligence Report</h1>
<p style="color:#888">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · Batch: {str(batch_id)[:8]}...</p>
<div class="kpis">
  <div class="kpi"><div class="val">{len(opportunities)}</div><div class="label">Opportunities</div></div>
  <div class="kpi"><div class="val" style="color:#cc0000">{critical_count}</div><div class="label">Critical Stockouts</div></div>
  <div class="kpi"><div class="val" style="color:#2e7d32">€{total_savings:,.0f}</div><div class="label">Est. Savings</div></div>
</div>
<h2>Top Opportunities</h2>
<table><thead><tr><th>EAN</th><th>Product</th><th>Priority</th><th>Score</th><th>Need</th><th>Price</th><th>Supplier</th><th>Savings</th></tr></thead>
<tbody>{opp_rows}</tbody></table>
<h2>Stockout Risk</h2>
<table><thead><tr><th>EAN</th><th>Product</th><th>Days Cover</th><th>Best Supplier</th></tr></thead>
<tbody>{stockout_rows}</tbody></table>
<h2>Supplier Win Rates</h2>{supplier_svg}
<h2>Brand Trend Alerts</h2>
<table><thead><tr><th>Brand</th><th>Price Trend</th><th>At-Risk SKUs</th><th>Coverage</th></tr></thead>
<tbody>{brand_rows}</tbody></table>
<p class="footer">All numbers sourced from processed batch data. Verify before placing orders.</p>
</body></html>"""
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_exporters.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add db/exporters.py tests/test_exporters.py
git commit -m "feat: add Excel/HTML/JSON export layer"
```

---

## Task 8: Analytics Dashboard Tab

**Files:**
- Create: `tabs/analytics_tab.py`

- [ ] **Step 1: Create `tabs/analytics_tab.py`**

```python
"""
Analytics Dashboard Tab — 4 sub-tabs: Price Trends, Stockout Risk, Supplier Scorecard, Brand Health.
Reads from SQLite via AnalyticsEngine. Requires at least one saved batch.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import io
from db.database import ProcurementDB
from db.exporters import export_excel, export_html_report, export_json_payload
from analysis.analytics_engine import AnalyticsEngine


def analytics_tab():
    st.header("📊 Analytics Dashboard")

    db = ProcurementDB()
    runs = db.get_runs()

    if not runs:
        st.warning(
            "⚠️ No saved batches yet. Process supplier files in the **File Processing** tab first — "
            "data is saved automatically."
        )
        return

    engine = AnalyticsEngine(db)

    # Internal data from session (needed for stockout + brand health)
    opp_engine = st.session_state.get("opportunity_engine")
    internal_data = opp_engine.internal_data if opp_engine and hasattr(opp_engine, "internal_data") else []

    # Opportunities from session (needed for exports)
    opportunities = st.session_state.get("processed_data", [])

    st.caption(f"**{len(runs)} batch(es) stored** · Latest: {runs[0]['run_at'][:16]}")

    sub1, sub2, sub3, sub4 = st.tabs(
        ["📈 Price Trends", "⚠️ Stockout Risk", "🏆 Supplier Scorecard", "🏷️ Brand Health"]
    )

    # ── Price Trends ──────────────────────────────────────────────────────────
    with sub1:
        st.subheader("📈 Price Trends")

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            ean_filter = st.text_input("EAN search", placeholder="e.g. 3433...", key="pt_ean")
        with col_f2:
            all_brands = sorted({
                r.get("brand", "") for r in db.get_all_supplier_prices() if r.get("brand")
            })
            brand_filter = st.selectbox("Brand", ["All"] + all_brands, key="pt_brand")
        with col_f3:
            all_suppliers = sorted({
                r.get("supplier", "") for r in db.get_all_supplier_prices() if r.get("supplier")
            })
            supplier_filter = st.multiselect("Suppliers", all_suppliers, key="pt_supplier")
        with col_f4:
            days_options = {"30 days": 30, "90 days": 90, "All": 3650}
            days_label = st.selectbox("Time range", list(days_options.keys()), index=1, key="pt_days")
            days = days_options[days_label]

        threshold = st.slider("Price alert threshold (%)", 1, 20, 5, key="pt_threshold")

        df_trends = engine.compute_price_trends(
            ean=ean_filter or None,
            brand=brand_filter if brand_filter != "All" else None,
            supplier=None,
            days=days,
        )
        if supplier_filter:
            df_trends = df_trends[df_trends["supplier"].isin(supplier_filter)]

        if df_trends.empty:
            st.info("No price history found for the selected filters.")
        else:
            fig = px.line(
                df_trends,
                x="run_at",
                y="price_net",
                color="supplier",
                facet_col="ean" if not ean_filter else None,
                markers=True,
                labels={"run_at": "Batch date", "price_net": "Price (€)", "supplier": "Supplier"},
                title="Price over time by supplier",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            alerts = engine.compute_price_trend_alerts(threshold_pct=float(threshold))
            if alerts:
                st.warning(f"**{len(alerts)} price alert(s)** — movement ≥ {threshold}% between batches")
                alert_df = pd.DataFrame(alerts)[["ean", "supplier", "brand", "prev_price", "current_price", "pct_change", "direction"]]
                st.dataframe(alert_df, use_container_width=True)

    # ── Stockout Risk ─────────────────────────────────────────────────────────
    with sub2:
        st.subheader("⚠️ Stockout Risk")

        if not internal_data:
            st.info("Load internal data in the **Opportunities** tab to see stockout risk.")
        else:
            df_stock = engine.compute_stockout_risk(internal_data)

            critical = df_stock[df_stock["urgency"] == "Critical"]
            warning = df_stock[df_stock["urgency"] == "Warning"]
            ok = df_stock[df_stock["urgency"] == "OK"]

            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Critical (<7 days)", len(critical))
            c2.metric("🟠 Warning (7–30 days)", len(warning))
            c3.metric("🟢 OK (>30 days)", len(ok))

            def color_urgency(val):
                colors = {"Critical": "background-color:#3a0000;color:#ff6b6b",
                          "Warning": "background-color:#3a2000;color:#ffa040",
                          "OK": "background-color:#003a00;color:#7cf5a0"}
                return colors.get(val, "")

            styled = df_stock.style.applymap(color_urgency, subset=["urgency"])
            st.dataframe(styled, use_container_width=True, height=400)

            # Download
            stockout_bytes = export_excel([], pd.DataFrame(), df_stock, pd.DataFrame(), pd.DataFrame())
            st.download_button(
                "📥 Download Stockout Report (Excel)",
                data=stockout_bytes,
                file_name=f"stockout_risk_{runs[0]['run_at'][:10]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ── Supplier Scorecard ────────────────────────────────────────────────────
    with sub3:
        st.subheader("🏆 Supplier Scorecard")
        df_sup = engine.compute_supplier_win_rates()

        if df_sup.empty:
            st.info("Not enough batch history yet. Win rates are more meaningful after 2+ batches.")
        else:
            fig_win = px.bar(
                df_sup,
                x="win_rate_pct",
                y="supplier",
                orientation="h",
                color="win_rate_pct",
                color_continuous_scale="Greens",
                labels={"win_rate_pct": "Win rate %", "supplier": ""},
                title="Win rate — % of EANs where supplier was cheapest",
            )
            fig_win.update_layout(height=max(300, len(df_sup) * 40), showlegend=False)
            st.plotly_chart(fig_win, use_container_width=True)

            display_cols = ["supplier", "sku_count", "win_rate_pct", "avg_price_index", "price_stability"]
            st.dataframe(df_sup[display_cols], use_container_width=True)

            sup_bytes = export_excel([], pd.DataFrame(), pd.DataFrame(), df_sup, pd.DataFrame())
            st.download_button(
                "📥 Download Supplier Scorecard (Excel)",
                data=sup_bytes,
                file_name=f"supplier_scorecard_{runs[0]['run_at'][:10]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ── Brand Health ──────────────────────────────────────────────────────────
    with sub4:
        st.subheader("🏷️ Brand Health")

        if not internal_data:
            st.info("Load internal data in the **Opportunities** tab to see brand health.")
        else:
            df_brand = engine.compute_brand_health(internal_data)

            if df_brand.empty:
                st.info("No brand data available.")
            else:
                def brand_trend_arrow(val):
                    if val > 1:
                        return f"↑ {val:.1f}%"
                    elif val < -1:
                        return f"↓ {abs(val):.1f}%"
                    return f"→ {abs(val):.1f}%"

                df_display = df_brand.copy()
                df_display["price_trend"] = df_display["price_trend_pct"].apply(brand_trend_arrow)

                st.dataframe(
                    df_display[["brand", "sku_count", "price_trend", "at_risk_skus", "coverage_pct", "top_opportunity"]],
                    use_container_width=True,
                    height=400,
                )

                brand_bytes = export_excel([], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), df_brand)
                st.download_button(
                    "📥 Download Brand Health (Excel)",
                    data=brand_bytes,
                    file_name=f"brand_health_{runs[0]['run_at'][:10]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    # ── Full Export (bottom of page) ──────────────────────────────────────────
    st.divider()
    st.subheader("📤 Full Exports")
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        if st.button("📊 Generate Full Excel Workbook"):
            opp_list = list(st.session_state.get("processed_data") or [])
            df_trends_full = engine.compute_price_trends(days=90)
            df_stock_full = engine.compute_stockout_risk(internal_data) if internal_data else pd.DataFrame()
            df_sup_full = engine.compute_supplier_win_rates()
            df_brand_full = engine.compute_brand_health(internal_data) if internal_data else pd.DataFrame()
            xlsx_bytes = export_excel(opp_list, df_trends_full, df_stock_full, df_sup_full, df_brand_full)
            st.download_button(
                "⬇️ Download Full Workbook",
                data=xlsx_bytes,
                file_name=f"procurement_report_{runs[0]['run_at'][:10]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with exp_col2:
        if st.button("📄 Generate HTML Report"):
            opp_list = list(st.session_state.get("processed_data") or [])
            df_stock_full = engine.compute_stockout_risk(internal_data).to_dict("records") if internal_data else []
            df_sup_full = engine.compute_supplier_win_rates().to_dict("records")
            df_brand_full = engine.compute_brand_health(internal_data).to_dict("records") if internal_data else []
            alerts = engine.compute_price_trend_alerts()
            batch_id = st.session_state.get("current_run_id", "unknown")
            html = export_html_report(opp_list, df_stock_full, df_sup_full, df_brand_full, batch_id)
            st.download_button(
                "⬇️ Download HTML Report",
                data=html.encode("utf-8"),
                file_name=f"procurement_report_{runs[0]['run_at'][:10]}.html",
                mime="text/html",
            )

    with exp_col3:
        if st.button("🤖 Generate JSON Payload (LLM-ready)"):
            opp_list = list(st.session_state.get("processed_data") or [])
            df_stock_full = engine.compute_stockout_risk(internal_data).to_dict("records") if internal_data else []
            df_sup_full = engine.compute_supplier_win_rates().to_dict("records")
            alerts = engine.compute_price_trend_alerts()
            batch_id = st.session_state.get("current_run_id", "unknown")
            json_str = export_json_payload(opp_list, alerts, df_stock_full, df_sup_full, batch_id)
            st.download_button(
                "⬇️ Download JSON Payload",
                data=json_str.encode("utf-8"),
                file_name=f"procurement_payload_{runs[0]['run_at'][:10]}.json",
                mime="application/json",
            )
```

- [ ] **Step 2: Smoke test the tab in isolation**

Add a temporary test import at the top of a Python shell:

```python
from tabs.analytics_tab import analytics_tab
print("Import OK")
```

Expected: `Import OK` with no errors.

- [ ] **Step 3: Commit**

```bash
git add tabs/analytics_tab.py
git commit -m "feat: add Analytics Dashboard tab with 4 sub-tabs"
```

---

## Task 9: Wire Everything into app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the analytics tab import**

In `app.py`, find the existing tab imports (around line 31):

```python
from tabs.opportunities_tab import opportunities_tab
from tabs.marketing_campaign_tab import marketing_campaign_tab
```

Add:

```python
from tabs.analytics_tab import analytics_tab
```

- [ ] **Step 2: Add the Analytics tab to the manual-mode tab list**

Find the manual-mode tab definition (around line 238):

```python
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📤 File Processing", "🎯 Opportunities", "📣 Marketing Campaigns", "🛒 Order Optimization", "🔄 CSV Delimiter Converter"]
        )
```

Replace with:

```python
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📤 File Processing", "🎯 Opportunities", "📊 Analytics", "📣 Marketing Campaigns", "🛒 Order Optimization", "🔄 CSV Delimiter Converter"]
        )
```

- [ ] **Step 3: Add the tab body**

Below the existing `with tab2:` block and before `with tab3:`, insert:

```python
        with tab3:
            analytics_tab()
```

Then shift the remaining tabs:

```python
        with tab4:
            marketing_campaign_tab(groq_api_key, api_key_valid=bool(groq_api_key))
        with tab5:
            order_optimization_tab()
        with tab6:
            csv_delimiter_converter_tab()
```

- [ ] **Step 4: Full end-to-end smoke test**

```
streamlit run app.py
```

1. Upload a supplier catalog file → click "Process Supplier Files" → confirm `procurement.db` is created
2. Go to Opportunities tab → upload internal CSV → confirm "Internal data saved" (check logs)
3. Go to Analytics tab → confirm "1 batch stored" appears
4. Open Price Trends sub-tab → confirm chart renders (single batch = single point per supplier)
5. Open Stockout Risk sub-tab → confirm urgency-colored table
6. Open Supplier Scorecard → confirm win rate bars
7. Open Brand Health → confirm brand rows
8. Return to Opportunities tab → confirm "Enh. Score" and "Score Breakdown" columns appear
9. Toggle "Sort by Enhanced Score" → confirm table reorders
10. Go to Analytics tab → click "Generate JSON Payload" → download and verify JSON structure

- [ ] **Step 5: Run the full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS (20 tests across 4 files)

- [ ] **Step 6: Final commit**

```bash
git add app.py
git commit -m "feat: wire Analytics tab and enhanced scoring into app — closes analytics depth + SQLite spec"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] §3 SQLite layer → Task 1 (db/database.py)
- [x] §3.3 Auto-save trigger → Tasks 2 & 3
- [x] §4/5 Analytics Dashboard tab → Task 8
- [x] §4.2 AnalyticsEngine functions → Task 4
- [x] §5.1 Price Trends → Task 8 sub1
- [x] §5.2 Stockout Risk → Task 8 sub2
- [x] §5.3 Supplier Scorecard → Task 8 sub3
- [x] §5.4 Brand Health → Task 8 sub4
- [x] §6.2 Enhanced scoring formula → Task 5
- [x] §6.3 Dual scoring columns + sort control → Task 6
- [x] §7.1 Excel 5-sheet → Task 7 export_excel
- [x] §7.2 HTML report → Task 7 export_html_report
- [x] §7.3 JSON payload → Task 7 export_json_payload
- [x] §8 File structure → matches Tasks 1–9
- [x] §10 Success criteria → verified in Task 9 Step 4

**Type consistency:**
- `ProcurementDB.save_supplier_batch` → called in Task 2 with same signature
- `ProcurementDB.save_internal_data` → called in Task 3 with same signature
- `AnalyticsEngine(db)` → instantiated in Task 8 with same constructor
- `compute_enhanced_scores(opportunities)` → called in Task 6 with same signature
- `export_excel(opps, df, df, df, df)` → 5-arg signature consistent across Tasks 7 and 8
- `export_json_payload(opps, alerts, risks, movements, batch_id)` → consistent
- `export_html_report(opps, risks, win_rates, brand_alerts, batch_id)` → consistent
