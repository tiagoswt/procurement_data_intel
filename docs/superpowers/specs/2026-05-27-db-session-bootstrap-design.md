# DB Session State Bootstrap Design

**Date:** 2026-05-27  
**Status:** Approved  
**Goal:** Replace the file-upload-gated session state pattern with DB-backed auto-loading, so supplier and internal data survive page refreshes without re-uploading.

---

## Problem

The Opportunities and Order Optimization tabs check `st.session_state.processed_data` (supplier) and `st.session_state.opportunity_engine.internal_data` (internal) before rendering. When the page refreshes or a new session starts, both are empty — the tabs refuse to work even though all data already exists in `procurement.db`. The Analytics tab does not have this problem because it reads from the DB directly.

---

## Approach

Session state bootstrap in `app.py`. A single `_bootstrap_from_db()` function runs once per session at the top of `main()`, before any tab renders. It silently loads the latest active supplier prices and latest internal data from the DB into session state. All tab code remains unchanged — they continue to read from session state as before.

---

## Files Changed

### 1. `db/database.py`

**Fix `get_latest_internal_data()`**

Current implementation finds the latest `processing_run` and looks for internal data on it. If the latest run was a supplier batch (no internal data attached), it returns empty. Fix: query for the latest run that actually has rows in `internal_data`.

```sql
-- Before (broken): finds latest processing_run regardless of internal_data presence
SELECT id FROM processing_runs ORDER BY run_at DESC LIMIT 1

-- After (fixed): finds latest run with actual internal data
SELECT DISTINCT idat.run_id, pr.run_at
FROM internal_data idat
JOIN processing_runs pr ON idat.run_id = pr.id
ORDER BY pr.run_at DESC LIMIT 1
```

**Add `active_only` parameter to `get_all_supplier_prices()`**

Default `active_only=False` preserves all existing callers (Analytics tab uses full history). When `active_only=True`, adds:

```sql
WHERE (sp.valid_until IS NULL OR sp.valid_until >= date('now'))
```

Expired offers are excluded from the bootstrap but remain visible in Analytics price trend views, which need full historical data.

**Extend `internal_data` schema (migration)**

Add two columns that are currently lost on refresh:

```sql
ALTER TABLE internal_data ADD COLUMN supplier_price REAL;
ALTER TABLE internal_data ADD COLUMN bestseller_rank INT;
```

`supplier_price` is used in opportunity price comparisons. `bestseller_rank` is used to derive `is_bestseller` (rank 1 or 2). Without these, a DB-restored session would score all products as non-bestsellers and miss some opportunity comparisons.

Migration runs in `_migrate_schema()` with `try/except` (column already exists → no-op).

**Update `save_internal_data()`**

Write the two new fields when persisting internal data:
- `p.get("supplier_price")` → `supplier_price`
- `p.get("bestseller_rank")` → `bestseller_rank`

### 2. `app.py`

**Add `_bootstrap_from_db()`**

Called once per session at the top of `main()`, before the tab layout. Guarded by `st.session_state._db_bootstrapped` to prevent re-running within the same session (e.g., on widget interactions that trigger reruns).

```
if st.session_state.get("_db_bootstrapped"):
    return

db = ProcurementDB()

# --- Supplier data ---
if not st.session_state.get("processed_data"):
    rows = db.get_all_supplier_prices(active_only=True)
    if rows:
        products = [ProductData(
            ean_code=r["ean"],
            supplier=r["supplier"],
            price=r["price_net"],
            quantity=r["quantity"],
            product_name=r["description"],
        ) for r in rows]
        st.session_state.processed_data = products

# --- Internal data ---
# opportunity_engine is initialized lazily in the Opportunities tab, so we
# must ensure it exists here before trying to populate it.
if "opportunity_engine" not in st.session_state:
    from analysis.opportunity_engine import SimpleOpportunityEngine
    st.session_state.opportunity_engine = SimpleOpportunityEngine()
engine = st.session_state.opportunity_engine
if not engine.internal_data:
    rows = db.get_latest_internal_data()
    if rows:
        engine.internal_data = [_db_row_to_internal_dict(r) for r in rows]

st.session_state._db_bootstrapped = True
```

**Add `_db_row_to_internal_dict(row)` helper**

Maps DB column names to the dict format `SimpleOpportunityEngine.internal_data` expects:

| DB column         | Dict key               | Notes                              |
|-------------------|------------------------|------------------------------------|
| `ean`             | `ean`                  |                                    |
| `current_stock`   | `stock`                |                                    |
| `sales90d`        | `sales90d`             |                                    |
| `sales180d`       | `sales180d`            |                                    |
| `sales365d`       | `sales365d`            |                                    |
| `best_buy_price`  | `best_buy_price`       |                                    |
| `avg_stock_price` | `stock_avg_price`      |                                    |
| `qnt_pending`     | `qntPendingToDeliver`  |                                    |
| `supplier_price`  | `supplier_price`       |                                    |
| `bestseller_rank` | `bestseller_rank`      |                                    |
| `brand`           | `brand`                |                                    |
| `description`     | `description`          |                                    |
| *(derived)*       | `is_bestseller`        | `bestseller_rank in [1, 2]`        |
| *(not in DB)*     | `cnp`                  | default `""`                       |
| *(not in DB)*     | `capacity`             | default `""`                       |
| *(not in DB)*     | `sales_next90d_lastyear` | default `0`                      |
| *(not in DB)*     | `best_supplier`        | default `""`                       |
| *(not in DB)*     | `is_active`            | default `True`                     |

### 3. `tabs/opportunities_tab.py`

**Fix internal data save path when `current_run_id` is missing**

Currently, internal data is only saved to the DB if `current_run_id` is set in session state. If a user uploads internal data in a fresh session (without first processing supplier files that session), the data is never persisted.

Fix: if `current_run_id` is absent, create a new `processing_run` and use its ID:

```python
run_id = st.session_state.get("current_run_id")
if not run_id:
    # No supplier batch this session — create a run to anchor internal data
    run_id = db.create_run(source_files=["internal_data"], row_count=len(engine.internal_data))
    st.session_state.current_run_id = run_id
ProcurementDB().save_internal_data(run_id, engine.internal_data)
```

This requires adding a `create_run()` method to `ProcurementDB`.

---

## What Does NOT Change

- File Processing tab — upload flow, `save_supplier_batch()`, unchanged
- Opportunities tab internal data upload section — unchanged, still saves to DB on load
- Analytics tab — already reads from DB directly, no session state involved
- Order Optimization tab — already works from session state; bootstrap populates it
- All other tabs and modules

---

## Data Validity Rules

| Data type       | Bootstrap filter              | Analytics filter |
|-----------------|-------------------------------|-----------------|
| Supplier prices | Active only (`valid_until IS NULL OR valid_until >= today`) | All (historical) |
| Internal data   | Latest run with internal data | N/A (not used)  |

---

## Error Handling

- If DB is empty (first ever run), bootstrap finds nothing and session state stays empty — tabs show their normal "please upload" messages. No change in first-run behavior.
- If DB read fails (corrupt file, etc.), bootstrap catches the exception silently and sets `_db_bootstrapped = True` to avoid retrying on every rerun.

---

## Out of Scope

- Storing `cnp`, `capacity`, `sales_next90d_lastyear`, `best_supplier` in the DB — these are display-only fields that default safely
- Per-run internal data selection (user always gets latest)
- Showing a "data restored from DB" banner (user requested silent load)
