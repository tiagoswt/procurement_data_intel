# DB Session Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-restore supplier and internal data from the DB on page refresh so tabs work without re-uploading files.

**Architecture:** A `_bootstrap_from_db()` function in `app.py` runs once per session (guarded by a session state flag), silently loading active supplier prices and the latest internal data from SQLite into session state. DB layer gets four focused fixes: corrected `get_latest_internal_data()`, `active_only` filter on `get_all_supplier_prices()`, two new columns on `internal_data`, and a `create_run()` helper. The opportunities tab gets a one-line save fix.

**Tech Stack:** Python, SQLite (via `sqlite3`), Streamlit session state, pytest

---

## File Map

| File | What changes |
|---|---|
| `db/database.py` | Fix `get_latest_internal_data()`, add `active_only` param, migrate schema, update `save_internal_data()`, add `create_run()` |
| `app.py` | Add `_db_row_to_internal_dict()`, add `_bootstrap_from_db()`, call it from `main()` |
| `tabs/opportunities_tab.py` | Fix internal data save when `current_run_id` is absent |
| `tests/test_database.py` | Tests for all DB changes |
| `tests/test_db_bootstrap.py` | Tests for bootstrap helpers |

---

## Task 1: Fix `get_latest_internal_data()` to find the latest run that has internal data

The current query picks the latest `processing_run` regardless of whether it has any `internal_data` rows. If the last run was a supplier batch, this returns empty even when internal data exists.

**Files:**
- Modify: `db/database.py` — `get_latest_internal_data()` method
- Modify: `tests/test_database.py` — add regression test

- [ ] **Step 1: Write the failing test**

Open `tests/test_database.py` and add at the bottom:

```python
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```
pytest tests/test_database.py::test_get_latest_internal_data_ignores_runs_without_internal_data -v
```

Expected: FAIL — the method returns 0 rows when the latest run has no internal data.

- [ ] **Step 3: Fix `get_latest_internal_data()` in `db/database.py`**

Find the method (around line 243) and replace it entirely:

```python
def get_latest_internal_data(self) -> List[Dict]:
    with self._get_conn() as conn:
        latest = conn.execute(
            """SELECT DISTINCT idat.run_id, pr.run_at
               FROM internal_data idat
               JOIN processing_runs pr ON idat.run_id = pr.id
               ORDER BY pr.run_at DESC LIMIT 1"""
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
            (latest["run_id"],),
        ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run the test to confirm it passes**

```
pytest tests/test_database.py::test_get_latest_internal_data_ignores_runs_without_internal_data -v
```

Expected: PASS

- [ ] **Step 5: Run full DB test suite to check for regressions**

```
pytest tests/test_database.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add db/database.py tests/test_database.py
git commit -m "fix: get_latest_internal_data queries by internal data presence not latest run"
```

---

## Task 2: Add `active_only` filter to `get_all_supplier_prices()`

Operational tabs (Opportunities, Order Optimization) must only see non-expired offers. Analytics needs full history. The fix adds an optional `active_only=False` parameter — callers that need filtering pass `active_only=True`.

**Files:**
- Modify: `db/database.py` — `get_all_supplier_prices()` method
- Modify: `tests/test_database.py` — add tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_database.py`:

```python
from datetime import date, timedelta


def test_get_all_supplier_prices_active_only_excludes_expired(tmp_db):
    expired = date.today() - timedelta(days=1)
    future  = date.today() + timedelta(days=30)
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

```
pytest tests/test_database.py::test_get_all_supplier_prices_active_only_excludes_expired tests/test_database.py::test_get_all_supplier_prices_default_includes_expired -v
```

Expected: FAIL — `get_all_supplier_prices` doesn't accept `active_only` yet.

- [ ] **Step 3: Update `get_all_supplier_prices()` in `db/database.py`**

Find the method (around line 217) and replace it:

```python
def get_all_supplier_prices(self, active_only: bool = False) -> List[Dict]:
    where = ""
    if active_only:
        where = "WHERE (sp.valid_until IS NULL OR sp.valid_until >= date('now'))"
    with self._get_conn() as conn:
        rows = conn.execute(
            f"""SELECT sp.ean, sp.supplier, sp.price_net, sp.quantity, pr.run_at,
                       p.brand, p.description
               FROM supplier_prices sp
               JOIN processing_runs pr ON sp.run_id = pr.id
               LEFT JOIN (
                   SELECT ean, MAX(rowid) AS rid FROM products GROUP BY ean
               ) latest_p ON sp.ean = latest_p.ean
               LEFT JOIN products p ON p.rowid = latest_p.rid
               {where}
               ORDER BY pr.run_at DESC"""
        ).fetchall()
    return [
        {**dict(r), "supplier": re.sub(r'_\d{8}$', '', r["supplier"] or "")}
        for r in rows
    ]
```

- [ ] **Step 4: Run the tests to confirm they pass**

```
pytest tests/test_database.py::test_get_all_supplier_prices_active_only_excludes_expired tests/test_database.py::test_get_all_supplier_prices_default_includes_expired -v
```

Expected: PASS

- [ ] **Step 5: Run full DB test suite**

```
pytest tests/test_database.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add db/database.py tests/test_database.py
git commit -m "feat: add active_only filter to get_all_supplier_prices"
```

---

## Task 3: Extend `internal_data` schema and update `save_internal_data()`

Two fields needed for accurate DB-restored sessions are missing from the schema: `supplier_price` (used in opportunity price comparisons) and `bestseller_rank` (used to derive `is_bestseller`). Add them via migration and persist them on save.

**Files:**
- Modify: `db/database.py` — `_migrate_schema()`, `save_internal_data()`
- Modify: `tests/test_database.py` — add tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_database.py`:

```python
def test_save_internal_data_persists_supplier_price_and_bestseller_rank(tmp_db, sample_products):
    run_id = tmp_db.save_supplier_batch(["file_a.xlsx"], sample_products)
    internal = [
        {"ean": "3433422404397", "stock": 100, "sales90d": 50,
         "sales180d": 100, "sales365d": 200,
         "best_buy_price": 9.20, "stock_avg_price": 9.10,
         "supplier_price": 9.50, "bestseller_rank": 1,
         "qntPendingToDeliver": 0, "brand": "L'Oréal", "description": "Serum"},
    ]
    tmp_db.save_internal_data(run_id, internal)
    data = tmp_db.get_latest_internal_data()
    assert len(data) == 1
    row = data[0]
    assert row["supplier_price"] == 9.50
    assert row["bestseller_rank"] == 1


def test_save_internal_data_handles_missing_new_fields(tmp_db, sample_products, sample_internal):
    # sample_internal fixtures don't have supplier_price/bestseller_rank — must not crash
    run_id = tmp_db.save_supplier_batch(["file_a.xlsx"], sample_products)
    tmp_db.save_internal_data(run_id, sample_internal)  # should not raise
    data = tmp_db.get_latest_internal_data()
    assert len(data) == 2
```

- [ ] **Step 2: Run the tests to confirm they fail**

```
pytest tests/test_database.py::test_save_internal_data_persists_supplier_price_and_bestseller_rank tests/test_database.py::test_save_internal_data_handles_missing_new_fields -v
```

Expected: FAIL

- [ ] **Step 3: Add migration in `_migrate_schema()` in `db/database.py`**

Find `_migrate_schema` (around line 71) and add two more migration attempts:

```python
def _migrate_schema(self):
    with self._get_conn() as conn:
        try:
            conn.execute("ALTER TABLE supplier_prices ADD COLUMN valid_until DATE")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE internal_data ADD COLUMN supplier_price REAL")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE internal_data ADD COLUMN bestseller_rank INT")
        except Exception:
            pass
```

- [ ] **Step 4: Update `save_internal_data()` to write the new fields**

Find `save_internal_data` (around line 140) and update the INSERT statement:

```python
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
                        best_buy_price, avg_stock_price, qnt_pending,
                        supplier_price, bestseller_rank)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ean, run_id,
                        p.get("stock", 0),
                        p.get("sales90d", 0),
                        p.get("sales180d", 0),
                        p.get("sales365d", 0),
                        p.get("best_buy_price"),
                        p.get("stock_avg_price"),
                        p.get("qntPendingToDeliver", 0),
                        p.get("supplier_price"),
                        p.get("bestseller_rank"),
                    ),
                )
        logger.info(f"Internal data saved: {len(internal_products)} products for run {run_id}")
    except Exception as e:
        logger.error(f"Failed to save internal data: {e}")
        raise
```

- [ ] **Step 5: Run the tests to confirm they pass**

```
pytest tests/test_database.py::test_save_internal_data_persists_supplier_price_and_bestseller_rank tests/test_database.py::test_save_internal_data_handles_missing_new_fields -v
```

Expected: PASS

- [ ] **Step 6: Run full DB test suite**

```
pytest tests/test_database.py -v
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add db/database.py tests/test_database.py
git commit -m "feat: add supplier_price and bestseller_rank to internal_data schema"
```

---

## Task 4: Add `create_run()` method to `ProcurementDB`

Needed so `opportunities_tab.py` can create an anchor run when saving internal data without a prior supplier batch in the current session.

**Files:**
- Modify: `db/database.py` — add `create_run()` method
- Modify: `tests/test_database.py` — add test

- [ ] **Step 1: Write the failing test**

Add to `tests/test_database.py`:

```python
def test_create_run_returns_run_id(tmp_db):
    run_id = tmp_db.create_run(source_files=["internal_only.csv"], row_count=42)
    assert isinstance(run_id, str) and len(run_id) > 0
    runs = tmp_db.get_runs()
    assert any(r["id"] == run_id for r in runs)
```

- [ ] **Step 2: Run the test to confirm it fails**

```
pytest tests/test_database.py::test_create_run_returns_run_id -v
```

Expected: FAIL — `ProcurementDB` has no `create_run` method.

- [ ] **Step 3: Add `create_run()` to `db/database.py`**

Add this method after `save_internal_data()`:

```python
def create_run(self, source_files: List[str], row_count: int = 0) -> str:
    """Create a bare processing run record and return its ID."""
    run_id = str(uuid.uuid4())
    run_at = datetime.utcnow().isoformat()
    with self._get_conn() as conn:
        conn.execute(
            "INSERT INTO processing_runs (id, run_at, source_files, row_count) VALUES (?, ?, ?, ?)",
            (run_id, run_at, json.dumps(source_files), row_count),
        )
    logger.info(f"Run created: run_id={run_id}")
    return run_id
```

- [ ] **Step 4: Run the test to confirm it passes**

```
pytest tests/test_database.py::test_create_run_returns_run_id -v
```

Expected: PASS

- [ ] **Step 5: Run full DB test suite**

```
pytest tests/test_database.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add db/database.py tests/test_database.py
git commit -m "feat: add create_run helper to ProcurementDB"
```

---

## Task 5: Fix internal data save path in `opportunities_tab.py`

Currently the save is silently skipped when `current_run_id` is not in session state (e.g., user uploads internal data in a fresh session without first uploading supplier files). This means internal data is never persisted to the DB in that scenario.

**Files:**
- Modify: `tabs/opportunities_tab.py` — lines 233–239

- [ ] **Step 1: Replace the broken save block in `opportunities_tab.py`**

Find lines 233–239 (the `run_id = st.session_state.get("current_run_id")` block) and replace:

```python
                # Persist internal data to SQLite
                try:
                    db = ProcurementDB()
                    run_id = st.session_state.get("current_run_id")
                    if not run_id:
                        run_id = db.create_run(
                            source_files=["internal_data"],
                            row_count=len(engine.internal_data),
                        )
                        st.session_state.current_run_id = run_id
                    db.save_internal_data(run_id, engine.internal_data)
                except Exception as _e:
                    st.warning(f"⚠️ Internal data loaded but could not be saved to database: {_e}")
```

- [ ] **Step 2: Run the existing test suite to confirm nothing broke**

```
pytest tests/ -v --ignore=tests/normalize
```

Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add tabs/opportunities_tab.py
git commit -m "fix: always persist internal data to DB even without a prior supplier batch"
```

---

## Task 6: Add bootstrap helpers and wire into `app.py`

Add two module-level functions to `app.py`: `_db_row_to_internal_dict()` maps a DB row to the dict shape `SimpleOpportunityEngine.internal_data` expects; `_bootstrap_from_db()` runs once per session and silently populates session state from the DB.

**Files:**
- Modify: `app.py` — add two functions, call bootstrap from `main()`
- Create: `tests/test_db_bootstrap.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db_bootstrap.py`:

```python
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

```
pytest tests/test_db_bootstrap.py -v
```

Expected: FAIL — `_db_row_to_internal_dict` is not yet defined in `app.py`.

- [ ] **Step 3: Add `_db_row_to_internal_dict()` to `app.py`**

Add this function just before the `main()` function definition (around line 128):

```python
def _db_row_to_internal_dict(row: dict) -> dict:
    rank = row.get("bestseller_rank")
    return {
        "ean":                     row["ean"],
        "stock":                   row.get("current_stock", 0),
        "sales90d":                row.get("sales90d", 0),
        "sales180d":               row.get("sales180d", 0),
        "sales365d":               row.get("sales365d", 0),
        "best_buy_price":          row.get("best_buy_price"),
        "stock_avg_price":         row.get("avg_stock_price"),
        "qntPendingToDeliver":     row.get("qnt_pending", 0),
        "supplier_price":          row.get("supplier_price"),
        "bestseller_rank":         rank,
        "is_bestseller":           rank in (1, 2),
        "brand":                   row.get("brand", ""),
        "description":             row.get("description", ""),
        "cnp":                     "",
        "capacity":                "",
        "sales_next90d_lastyear":  0,
        "best_supplier":           "",
        "is_active":               True,
    }
```

- [ ] **Step 4: Run the tests to confirm they pass**

```
pytest tests/test_db_bootstrap.py -v
```

Expected: PASS

- [ ] **Step 5: Add `_bootstrap_from_db()` to `app.py`**

Add this function immediately after `_db_row_to_internal_dict()`:

```python
def _bootstrap_from_db() -> None:
    if st.session_state.get("_db_bootstrapped"):
        return
    try:
        db = ProcurementDB()

        # Supplier data
        if not st.session_state.get("processed_data"):
            rows = db.get_all_supplier_prices(active_only=True)
            if rows:
                st.session_state.processed_data = [
                    ProductData(
                        ean_code=r["ean"],
                        supplier=r["supplier"],
                        price=r["price_net"],
                        quantity=r["quantity"],
                        product_name=r["description"],
                    )
                    for r in rows
                ]

        # Internal data — initialize engine if not yet created
        if "opportunity_engine" not in st.session_state:
            from analysis.opportunity_engine import SimpleOpportunityEngine
            st.session_state.opportunity_engine = SimpleOpportunityEngine()

        engine = st.session_state.opportunity_engine
        if not engine.internal_data:
            rows = db.get_latest_internal_data()
            if rows:
                engine.internal_data = [_db_row_to_internal_dict(r) for r in rows]

    except Exception:
        pass  # DB unavailable — tabs will show their normal upload prompts
    finally:
        st.session_state._db_bootstrapped = True
```

- [ ] **Step 6: Call `_bootstrap_from_db()` at the top of `main()` in `app.py`**

Find the `main()` function. After `require_auth(auth)` (around line 135), add one line:

```python
    # Require authentication before proceeding
    require_auth(auth)

    # Restore data from DB if session state is empty (page refresh / new session)
    _bootstrap_from_db()
```

- [ ] **Step 7: Run the full test suite**

```
pytest tests/ -v --ignore=tests/normalize
```

Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_db_bootstrap.py
git commit -m "feat: bootstrap session state from DB on page refresh"
```

---

## Task 7: Smoke test end-to-end

Verify the full flow works in the running app.

- [ ] **Step 1: Start the app**

```
streamlit run app.py
```

- [ ] **Step 2: Upload a supplier catalog**

In the **File Processing** tab, upload any supplier CSV. Confirm products appear and the success message shows.

- [ ] **Step 3: Upload internal data**

In the **Opportunities** tab, upload an internal data CSV. Confirm the summary shows product counts.

- [ ] **Step 4: Hard-refresh the browser**

Press `Ctrl+Shift+R` (or `Cmd+Shift+R` on Mac) to clear the browser cache and force a fresh page load.

- [ ] **Step 5: Verify tabs work without re-uploading**

Open the **Opportunities** tab — it should show opportunity analysis without asking to re-upload. Open the **Order Optimization** tab — it should show supplier data loaded without asking to process files first.

- [ ] **Step 6: Verify Analytics still shows full history including expired offers**

Open the **Analytics** tab → **Price Trends**. Confirm expired supplier offers still appear in the trend chart (they should — `active_only=False` is the default).

- [ ] **Step 7: Final commit if any fixups were needed**

```bash
git add -p
git commit -m "fix: post-smoke-test corrections"
```
