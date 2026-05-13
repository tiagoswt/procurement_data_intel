# Analytics Depth + SQLite Persistence Bridge — Design Spec
**Date:** 2026-05-13
**Phase:** A (deepen current app) + B (add persistence)
**Status:** Approved

---

## 1. Goal

Extend the existing Streamlit POC with:
1. A SQLite persistence layer that survives page refreshes and accumulates history across upload batches
2. A new Analytics Dashboard tab with four sub-tabs: Price Trends, Stockout Risk, Supplier Scorecard, Brand Health
3. Dual opportunity scoring — current Priority (P1/P2) alongside a new formula-based Enhanced Score, both visible in the same table row so procurement can validate the new formula before committing to it
4. An enhanced export layer: Excel workbook, HTML summary report, and a structured JSON payload that is already shaped to feed Phase D (LLM narrative)

---

## 2. Architecture

### 2.1 Current flow (session-only)
```
Upload files → Field Detection (Groq) → Processor → Session State → Opportunities / Marketing / Orders
                                                                  ↳ lost on page refresh
```

### 2.2 New flow (with SQLite bridge)
```
Upload files → Field Detection (Groq) → Processor → Session State → Opportunities / Marketing / Orders
                                                          ↓ auto-save each batch
                                                       SQLite DB  ← persists across sessions
                                                          ↓
                                                    Analytics Engine ← price trends, win rates, stockout
                                                          ↓
                                          📊 Analytics Dashboard tab (new)   📤 Export Layer (enhanced)
```

Nothing is removed from the current app. All existing tabs stay unchanged.

### 2.3 Migration path
SQLite now → Postgres later (Phase B) is a one-line connection string change. The analytics engine, schema, and app code do not change. Table names and column shapes are deliberately aligned with the master plan's Postgres schema.

---

## 3. SQLite Persistence Layer

### 3.1 Module: `db/database.py`
Thin wrapper around SQLite using Python's built-in `sqlite3`. Exposes:
- `save_batch(run_id, products, supplier_prices)` — called automatically after every successful upload + process
- `get_price_history(ean, days)` — for trend charts
- `get_all_supplier_prices(run_id)` — for win rate computation
- `get_runs()` — list of all batches with timestamps

### 3.2 Schema (SQLite, Postgres-compatible column names)

```sql
processing_runs (
  id           TEXT PRIMARY KEY,  -- uuid
  run_at       TIMESTAMP NOT NULL,
  source_files TEXT,              -- JSON array of filenames
  row_count    INT,
  notes        TEXT
);

products (
  ean          TEXT NOT NULL,
  description  TEXT,
  brand        TEXT,
  pack_size    INT,
  run_id       TEXT REFERENCES processing_runs(id)
);

supplier_prices (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  ean          TEXT NOT NULL,
  supplier     TEXT NOT NULL,
  price_net    REAL,
  currency     TEXT DEFAULT 'EUR',
  quantity     INT,
  run_id       TEXT REFERENCES processing_runs(id),
  ingested_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX ON supplier_prices (ean, ingested_at DESC);

internal_data (
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
```

### 3.3 Auto-save trigger
Every time the user clicks "Process Files" and processing succeeds, `save_batch()` is called before results are written to session state. If SQLite save fails, processing results still appear — the DB is additive, never blocking.

---

## 4. Analytics Dashboard Tab

New Streamlit tab inserted between "Opportunities" and "Marketing Campaigns":
```
📤 File Processing | 🎯 Opportunities | 📊 Analytics | 📣 Marketing Campaigns | 🛒 Order Optimization | 🔄 CSV Converter
```

### 4.1 Module: `tabs/analytics_tab.py`
Orchestrates four inner sub-tabs via `st.tabs()`.

### 4.2 Module: `analysis/analytics_engine.py`
Pure Python/pandas. Reads from SQLite only. No Streamlit imports. Functions:
- `compute_price_trends(ean=None, brand=None, supplier=None, days=90)` → DataFrame
- `compute_stockout_risk(internal_data)` → DataFrame sorted by days_cover ascending
- `compute_supplier_win_rates()` → DataFrame with win_rate_pct, avg_price_index, price_stability, sku_count
- `compute_brand_health()` → DataFrame with brand, sku_count, price_trend_pct, at_risk_skus, coverage_pct, top_opportunity

---

## 5. Sub-tab Details

### 5.1 Price Trends
- **Filters (sidebar):** EAN search, brand dropdown, supplier multi-select, time range (30d / 90d / All)
- **Main chart:** Plotly line chart — one line per supplier, x-axis = batch date, y-axis = price
- **Alert banner:** auto-shown when any supplier's price moves >5% between batches (threshold configurable in sidebar)
- **Data source:** `supplier_prices` table joined on `processing_runs.run_at`

### 5.2 Stockout Risk
- **Three urgency bands:** Critical (<7 days cover, red), Warning (7–30 days, orange), OK (>30 days, green)
- **Table columns:** EAN, Product, Brand, Current Stock, Daily Sales (90d avg), Days Cover, Best Supplier, Best Price
- **Sort:** by days_cover ascending by default
- **Formula:** `days_cover = current_stock / (sales90d / 90)`
- **Data source:** `internal_data` table from latest run, joined with `supplier_prices` for best supplier

### 5.3 Supplier Scorecard
- **Win rate chart:** horizontal bar chart — % of competed EANs where supplier had cheapest price, computed across all stored runs
- **Price index table:** supplier, SKU count, avg price index vs cheapest (1.00 = always cheapest), price stability (High = price std dev <2% across batches, Medium = 2–5%, Low = >5%)
- **Data source:** `supplier_prices` table, all runs

### 5.4 Brand Health
- **Table columns:** Brand, SKU Count, Price Trend (avg % change vs previous batch), At-Risk SKUs (days_cover <30), Coverage % (SKUs with price in latest batch), Top Opportunity SKU
- **Price trend direction:** ↑ red (rising), ↓ green (falling), → grey (stable, <1% change)
- **Data source:** `supplier_prices` + `internal_data` + `products` tables

---

## 6. Opportunity Scoring — Dual Mode

### 6.1 Current scoring (unchanged)
Priority 1 (P1): net_need > 0 and supplier price < best_buy_price
Priority 2 (P2): net_need > 0 and supplier price >= best_buy_price

### 6.2 Enhanced scoring (new, additive)
Formula from master plan:
```
enhanced_score =
    velocity_score          (= sales90d / 90 / max_daily_sales_in_dataset)
  × margin_uplift           (= max(1.0, best_buy_price / supplier_price))
  × demand_coverage_gap     (= max(1.0, 30 / max(1, days_cover)))
  × strategic_brand_weight  (= configurable per brand, default 1.0, range 1.0–1.5)
  ÷ overstock_penalty       (= max(1.0, days_cover / 180) if days_cover > 180 else 1.0)
```

Score is normalised to a 0–10 scale across the current dataset.

### 6.3 UI presentation
Both scores shown as extra columns in the existing Opportunities table:

| EAN | Product | **Current** | **Enh. score** | **Score breakdown** | Net need | Best price | Supplier |
|---|---|---|---|---|---|---|---|
| ... | Revitalift Serum | P1 | 8.4 | v×1.8 · m×1.11 · u×2.1 · b×1.2 | 240 | €8.20 | BeautyDist |

- Sort buttons: "Sort by Current" / "Sort by Enhanced"
- Score breakdown shown as compact inline text (e.g. `v×1.8 · m×1.11 · u×2.1 · b×1.2 ÷os×0.4`)
- `strategic_brand_weight` configurable in sidebar per brand — allows procurement to boost tier-1 brands

### 6.4 Module: `analysis/enhanced_scoring.py`
New module. Does not modify `analysis/opportunity_engine.py`. Takes the existing opportunities list and appends `enhanced_score` and `score_breakdown` keys to each dict.

---

## 7. Export Layer

### 7.1 Excel workbook (`db/exporters.py → export_excel()`)
Five sheets generated from current session + SQLite history:
1. **Opportunities** — full opportunities list with both scoring columns
2. **Price Trends** — price history table for all EANs in this batch
3. **Stockout Risk** — urgency-ranked stockout table
4. **Suppliers** — win rate and price index scorecard
5. **Brands** — brand health summary

Download button on each tab generates the workbook pre-filtered to that tab's context (e.g. "Download" on Stockout Risk tab generates a workbook with all sheets but opens on the Stockout Risk sheet).

### 7.2 HTML summary report (`db/exporters.py → export_html_report()`)
Self-contained HTML file with:
- Headline KPI cards (opportunity count, critical stockouts, estimated savings, batch date)
- Top 10 opportunities table
- Stockout risk summary
- Supplier win rate bar chart (rendered as inline SVG)
- Brand trend alerts
- Generated timestamp and batch ID

No external dependencies — open in any browser, forward by email, no login required.

### 7.3 JSON payload (`db/exporters.py → export_json_payload()`)
Structured output matching the master plan's narrative layer input contract exactly:
```json
{
  "period": "2026-05-13",
  "batch_id": "uuid",
  "headline_metrics": {
    "opportunities_count": 12,
    "estimated_savings_eur": 47200,
    "skus_at_stockout_risk": 8,
    "skus_overstocked": 14,
    "new_supplier_prices_ingested": 4280
  },
  "top_opportunities": [...],
  "price_trend_alerts": [...],
  "stockout_risks": [...],
  "supplier_movements": [...],
  "overstock_warnings": [...]
}
```

Auto-saved to SQLite alongside the batch. Available for any future LLM call without re-exporting. This is the Phase D bridge — when the LLM narrative layer is added, it reads this exact payload with no extra plumbing.

---

## 8. New File Structure

```
procurement_data_intel/
├── db/
│   ├── __init__.py
│   ├── database.py          # SQLite connection + save_batch + query helpers
│   └── exporters.py         # Excel, HTML, JSON export functions
├── analysis/
│   ├── __init__.py
│   ├── opportunity_engine.py    # unchanged
│   ├── analytics_engine.py      # NEW — price trends, stockout, win rates, brand health
│   └── enhanced_scoring.py      # NEW — enhanced_score + breakdown appended to opps list
├── tabs/
│   ├── opportunities_tab.py     # extended — dual scoring columns + sort controls
│   ├── analytics_tab.py         # NEW — 4 sub-tabs
│   ├── marketing_campaign_tab.py    # unchanged
│   └── ...
└── app.py                       # updated tab list only
```

---

## 9. What This Sets Up for Later Phases

| Phase | What's already done here |
|---|---|
| **B — Postgres** | Schema is Postgres-compatible. Swap `sqlite3` for `psycopg2` in `db/database.py`. Everything else unchanged. |
| **D — LLM narrative** | JSON payload already produced in the right shape. LLM renderer just reads it. |
| **C — Automation** | `save_batch()` is already the right abstraction. Automated ingestion calls it instead of the manual upload handler. |

---

## 10. Success Criteria

- Every processed batch is persisted automatically; data survives page refresh
- Price Trends chart shows multi-batch history after 2+ uploads
- Stockout Risk table correctly identifies and colour-codes products by urgency
- Supplier Scorecard win rates computed across all stored batches
- Both scoring columns visible in Opportunities tab; sort by either works
- Score breakdown text legible inline without expanding
- Excel export has all 5 sheets populated
- JSON payload validates against the master plan's narrative layer contract
