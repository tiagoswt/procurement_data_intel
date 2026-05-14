# Normalization Layer — Design Spec

**Date:** 2026-05-14
**Project:** procurement_data_intel
**Status:** Approved

---

## Goal

Replace the current heuristic pipeline (`data_normalizer.py`, `field_detector.py`, `processor.py`) with a deterministic, profile-driven normalization layer that reads any supplier Excel/CSV file and returns `List[ProductData]` — the same output shape the rest of the project already consumes.

The layer is developed and tested in isolation. It connects to Streamlit only after all 7 supplier profiles are working end-to-end.

---

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Output model | Existing `ProductData` | No schema changes; feeds existing `save_supplier_batch()` unchanged |
| Pipeline | Full replacement of current pipeline | No dual-system maintenance |
| Primary interface | Python API (`normalize.ingest()`) | Streamlit calls it directly; CLI is a thin wrapper |
| Storage | No change | Existing SQLite + `save_supplier_batch()` |
| LLM bootstrap | Deferred | All 7 suppliers have known profiles; add when a new supplier arrives |

---

## Package Structure

```
normalize/
├── __init__.py        # exposes normalize.ingest()
├── cli.py             # thin CLI: normalize ingest <file>
├── detect.py          # filename → supplier_code
├── profile.py         # load + validate profiles/supplier.yaml
├── loader.py          # read xlsx/csv, find header row, iterate sheets
├── mapper.py          # apply column mapping → canonical keys
├── parsers/
│   ├── ean.py         # strip .0, preserve leading zeros, length check
│   └── price.py       # decimal separator normalization
└── validate.py        # drop bad rows, collect warnings

profiles/
├── colorwow.yaml
├── hispalbeauty.yaml
├── oferta_drogueria.yaml
├── argon_trading.yaml
└── kbeauty_eu.yaml
```

---

## Public API

```python
from normalize import ingest, NormalizeError

try:
    products, warnings = ingest("data/Garnier - Argon Trading.xlsx")
except NormalizeError as e:
    # hard failure: no profile matched, file unreadable, <50% row yield
    ...

# products: List[ProductData] — feeds existing save_supplier_batch()
# warnings: List[str]        — per-row issues to log or surface in UI
```

`ingest()` never raises on bad individual rows — only on hard file-level failures.

---

## Data Flow

```
file.xlsx
   │
   ▼ detect.py      — match filename against profile filename_patterns → supplier_code
   ▼ profile.py     — load profiles/{supplier_code}.yaml
   ▼ loader.py      — openpyxl/pandas read, skip to header_row, iterate sheets
   ▼ mapper.py      — rename source columns to {ean, product_name, price, quantity}
   ▼ parsers/       — clean EAN (strip .0, pad zeros), normalize price decimal sep
   ▼ validate.py    — drop rows missing price or product_name, collect per-row warnings
   ▼
List[ProductData]   — unchanged model, feeds existing DB layer
```

---

## Profile YAML Schema

```yaml
supplier_code: argon_trading
supplier_name: "Argon Trading"

match:
  filename_patterns:
    - "*Argon Trading*"

# brand_source options: static | filename_regex | column | sheet_name
brand_source: filename_regex
brand_regex: "^(?P<brand>.+?) - Argon Trading"

defaults:
  currency: EUR

sheets:
  - match: "*"          # sheet name glob (* = all sheets)
    header_row: 1       # 1-indexed
    decimal_sep: "."    # "." or ","
    columns:
      ean:          "EAN"
      product_name: "Description"
      price:        "PRICE"
      quantity:     ~   # null = not present in this file
```

---

## Supplier Profile Map

| File | Profile | Brand source | Header row | Decimal sep | Price column |
|---|---|---|---|---|---|
| `ColorWow offer.xlsx` | `colorwow` | static `"Color Wow"` | 1 | `.` | confirmed during profile authoring |
| `Estee Lauder - Argon Trading.xlsx` | `argon_trading` | filename regex | 1 | `.` | `PRICE` |
| `Garnier - Argon Trading.xlsx` | `argon_trading` | filename regex | 1 | `.` | `PRICE` |
| `Johnson's Baby - Argon Trading.xlsx` | `argon_trading` | filename regex | 1 | `.` | `PRICE` |
| `K-Beauty offer_EU DDP_Mar. 2026.xlsx` | `kbeauty_eu` | sheet name (trimmed) | 2 | `.` | `Supply Price (excl.vat)` |
| `OFERTA_DROGUERIA_140526.xlsx` | `oferta_drogueria` | regex on product col | 1 | `,` | confirmed during profile authoring |
| `STOCK HISPALBEAUTY ONTHEFLOOR 08052026.xlsx` | `hispalbeauty` | column `MARCA` | 1 | `.` | confirmed during profile authoring |

K-Beauty: row 1 is a banner — skip. 33 sheets, each sheet name = brand (strip trailing `_` and spaces). Only `Supply Price (excl.vat)` is mapped; other price tier columns (FOB, UAE, RRP) are ignored.

Argon Trading: one profile covers three files. Brand is parsed from the filename using `brand_regex`.

---

## Validation Rules

**Row-level** (warnings collected, row not always dropped):
- EAN cleaned: strip `.0` suffix, remove non-digits, preserve leading zeros
- EAN length must be 8, 12, 13, or 14 digits after cleaning — bad EAN → set to `None`, keep row if price + name present
- Price must be > 0 — missing or zero price → row dropped
- `product_name` must be non-empty — row dropped
- Duplicate EAN within file → keep first, warn on subsequent

**File-level** (hard errors → `NormalizeError` raised):
- No profile matched filename → `NormalizeError("No profile matched: <filename>")`
- File unreadable → `NormalizeError`
- < 50% of rows produced a valid price → `NormalizeError` (signals wrong profile or corrupt file)

---

## Build Phases

Profiles are written in difficulty order, matching the reference plan:

| Phase | Profiles | What's hard |
|---|---|---|
| 1 | ColorWow, Hispalbeauty, Droguería | Decimal sep `,` for Droguería; quantity from stock col |
| 2 | Argon Trading (covers 3 files) | Brand from filename regex; Terms cell ignored |
| 3 | K-Beauty | 33 sheets, header row 2, brand from sheet name, banner row skip |

Streamlit integration happens after Phase 3 passes all golden tests.

---

## Testing Strategy

```
tests/normalize/
├── fixtures/           # the 7 sample Excel files
├── golden/             # expected JSON snapshots per supplier
│   ├── colorwow.json
│   ├── hispalbeauty.json
│   ├── oferta_drogueria.json
│   ├── argon_trading_garnier.json
│   ├── argon_trading_estee_lauder.json
│   ├── argon_trading_johnsons.json
│   └── kbeauty_eu.json
├── test_profiles.py    # parametrized: file → ingest() → diff vs golden
├── test_parsers.py     # unit tests for ean.py and price.py
└── test_detect.py      # filename matching
```

Golden files are JSON snapshots of `List[dict]` output, committed to git. Parser unit tests cover known edge cases:
- EAN: `"887167665200.0"` → `"8871676652000"`, `"5060280378928"` → unchanged
- Price: `"25,50"` → `25.50`, `"1.234,56"` → `1234.56`, `"€29.99"` → `29.99`

Tests hit real Excel files — no mocking.

---

## Out of Scope

- LLM bootstrap for new suppliers (deferred — all current suppliers have profiles)
- Rich canonical schema (25+ fields) — existing `ProductData` is sufficient
- Currency conversion, price tiers, incoterms — not in `ProductData`
- Parquet output — existing SQLite layer unchanged
