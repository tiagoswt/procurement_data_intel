"""
CLI wrapper for normalization layer.

Usage:
    python -m normalize.cli ingest <file>
    python -m normalize.cli ingest <file> --dry-run
"""
import sys
from pathlib import Path

from normalize import NormalizeError, ingest


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "ingest":
        print("Usage: python -m normalize.cli ingest <file> [--dry-run]")
        sys.exit(1)

    file_path = sys.argv[2]
    dry_run = "--dry-run" in sys.argv

    if not Path(file_path).exists():
        print(f"Error: file not found: {file_path}")
        sys.exit(1)

    try:
        products, warnings = ingest(file_path)
    except NormalizeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Extracted {len(products)} products, {len(warnings)} warnings")

    zero_yield = [w for w in warnings if w.startswith("[ZERO YIELD]")]
    other_warnings = [w for w in warnings if not w.startswith("[ZERO YIELD]") and not w.startswith("[SKIP SUMMARY]")]
    skip_summaries = [w for w in warnings if w.startswith("[SKIP SUMMARY]")]

    if zero_yield:
        print(f"  !! {len(zero_yield)} sheet(s) produced 0 products:")
        for w in zero_yield:
            print(f"     {w}")

    if skip_summaries:
        for w in skip_summaries:
            print(f"  >> {w}")

    for w in other_warnings:
        print(f"  WARN: {w}")

    if dry_run:
        for p in products[:10]:
            print(f"  {p.ean_code}  {p.supplier}  {p.product_name[:50]}  {p.price}")
        return

    from db.database import ProcurementDB
    db = ProcurementDB()
    run_id = db.save_supplier_batch([file_path], products)
    print(f"Saved to DB: run_id={run_id}")


if __name__ == "__main__":
    main()
