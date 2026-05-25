# Design: `/generate-orders` Claude Code Skill

**Date:** 2026-05-25  
**Status:** Approved

## Overview

A Claude Code custom slash command (`.claude/commands/generate-orders.md`) that acts as a procurement agent. When invoked, Claude queries `procurement.db`, scores purchase opportunities per supplier, and writes one CSV per supplier to `output/orders/YYYY-MM-DD/`.

No separate Python script to maintain. Claude executes the analysis inline using Python via Bash, reasoning through edge cases (outlier prices, missing data) as it goes.

---

## Opportunity Types

### Type 1 — Need-based

Products where current stock is insufficient to cover 90-day demand.

```
net_need = sales90d − current_stock
Eligible: net_need > 0
Rank: net_need DESC, then saving_pct DESC
```

### Type 2 — Price-based overstock

Products where the supplier offers a price significantly below the current average stock price — worth buying extra even without an urgent need.

```
saving_pct = (avg_stock_price − supplier_price_net) / avg_stock_price × 100
Eligible: saving_pct ≥ 15% AND product has internal data AND NOT already in Type 1
Rank: saving_pct DESC
```

The 15% threshold is the default. It can be overridden by passing an argument: `/generate-orders --threshold 20`.

---

## Data Quality

### Outlier price filter

Supplier prices that are clearly corrupted (e.g., copedis decimal separator issues producing values in the billions) are excluded before scoring.

**Rule:** exclude rows where `supplier_price_net > avg_stock_price × 10` OR `supplier_price_net <= 0`.

### Missing internal data

Products in supplier_prices with no matching row in internal_data are excluded from both opportunity types. They cannot be scored without demand/stock context.

### avg_stock_price = 0 or NULL

Products where `avg_stock_price` is NULL or 0 are excluded from Type 2 saving calculations to avoid division-by-zero or misleading percentages. They can still appear in Type 1 if net_need > 0.

### Currency

Ventas prices are denominated in USD per their supplier profile, but are stored in the DB with the default EUR currency flag. The skill will note this in the console output and label ventas rows with `(USD)` in the supplier_price_net column header of that CSV until a proper FX conversion is in place.

---

## Output

### File structure

```
output/orders/
  2026-05-25/
    hispalbeauty.csv
    checkpoint.csv
    ventas.csv
    Icosmo.csv
    farmaciemorra.csv
    copedis.csv
```

### CSV columns

| Column | Description |
|---|---|
| `opportunity_type` | `NEED` or `PRICE` |
| `ean` | Product EAN |
| `brand` | Brand name |
| `description` | Product description |
| `current_stock` | Units currently in stock |
| `sales_90d` | Units sold in last 90 days |
| `net_need` | `sales90d − current_stock` (negative = surplus) |
| `avg_stock_price` | Current average stock price (EUR) |
| `supplier_price_net` | Supplier's offered price (EUR) |
| `saving_pct` | `(avg_stock_price − supplier_price_net) / avg_stock_price × 100` |
| `saving_eur_per_unit` | `avg_stock_price − supplier_price_net` |

Within each CSV: Type 1 rows first (ranked by net_need DESC), then Type 2 rows (ranked by saving_pct DESC).

### Console summary

After writing files, Claude prints:
- Files written and their paths
- Row counts per supplier (Type 1 / Type 2 breakdown)
- Top 3 opportunities across all suppliers

---

## Skill Invocation

```
/generate-orders
/generate-orders --threshold 20
```

The skill instructs Claude to:
1. Query `procurement.db` using Python inline
2. Apply outlier filter
3. Score and rank both opportunity types
4. Create `output/orders/YYYY-MM-DD/` directory
5. Write one CSV per supplier
6. Print summary to console
