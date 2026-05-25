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
