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
