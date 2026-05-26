"""
scripts/ingest_data.py — CLI helper for /ingest-data skill

Modes:
  --scan                         scan data/suppliers/ and data/internal/, output JSON
  --suggest <filepath>           AI-generate a normalize profile YAML for an Excel file
  --process '<json>'             ingest files by mapping {filepath: supplier_code}, save to DB
  --process-internal <run_id>    load all CSVs from data/internal/, save to DB under run_id
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─── helpers ─────────────────────────────────────────────────────────────────

SUPPLIERS_DIR = ROOT / "data" / "suppliers"
INTERNAL_DIR = ROOT / "data" / "internal"
SUPPORTED = {".csv", ".xlsx", ".xls"}


def _scan_folder(folder: Path):
    if not folder.exists():
        return []
    files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            stat = p.stat()
            files.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
            })
    return files


# ─── modes ───────────────────────────────────────────────────────────────────

def cmd_scan():
    from normalize.detect import detect_supplier
    from normalize.profile import list_profiles, load_profile

    supplier_files = []
    for f in _scan_folder(SUPPLIERS_DIR):
        code = detect_supplier(f["name"])
        name = None
        if code:
            try:
                name = load_profile(code).get("supplier_name", code)
            except Exception:
                name = code
        supplier_files.append({**f, "detected_code": code, "detected_name": name})

    internal_files = _scan_folder(INTERNAL_DIR)

    profiles = list_profiles()

    print(json.dumps({
        "supplier_files": supplier_files,
        "internal_files": internal_files,
        "available_profiles": profiles,
    }, indent=2))


def cmd_suggest(filepath: str):
    from normalize.wizard import suggest_profile
    try:
        yaml_text = suggest_profile(filepath)
        print(yaml_text)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def cmd_process(mapping_json: str, validity_days_json: str = None):
    """
    mapping_json: '{"<filepath>": "<supplier_code>", ...}'
    validity_days_json: '{"<supplier_name>": <days_or_null>, ...}'  (optional)
    Ingests each file, collects all ProductData, saves one supplier batch to DB.
    """
    from normalize.core import ingest
    from normalize.exceptions import NormalizeError
    from db.database import ProcurementDB

    try:
        mapping = json.loads(mapping_json)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON mapping: {e}"}), file=sys.stderr)
        sys.exit(1)

    validity_days = {}
    if validity_days_json:
        try:
            validity_days = json.loads(validity_days_json)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid validity-days JSON: {e}"}), file=sys.stderr)
            sys.exit(1)

    all_products = []
    by_supplier = {}
    failed = []
    all_warnings = []
    source_files = []

    for filepath, supplier_code in mapping.items():
        fname = Path(filepath).name
        try:
            products, warnings = ingest(filepath, supplier_code=supplier_code)
            all_products.extend(products)
            source_files.append(fname)
            by_supplier[supplier_code] = by_supplier.get(supplier_code, 0) + len(products)
            if warnings:
                all_warnings.extend([f"{fname}: {w}" for w in warnings])
            print(f"  ✓ {fname} → {len(products)} products ({supplier_code})", file=sys.stderr)
        except NormalizeError as e:
            failed.append(f"{fname}: {e}")
            print(f"  ✗ {fname}: {e}", file=sys.stderr)
        except Exception as e:
            failed.append(f"{fname}: unexpected error: {e}")
            print(f"  ✗ {fname}: unexpected error: {e}", file=sys.stderr)

    if not all_products:
        print(json.dumps({
            "run_id": None,
            "total_products": 0,
            "by_supplier": {},
            "failed": failed,
            "warnings": all_warnings,
        }))
        return

    db = ProcurementDB()
    run_id = db.save_supplier_batch(source_files, all_products, validity_days=validity_days)

    print(json.dumps({
        "run_id": run_id,
        "total_products": len(all_products),
        "by_supplier": by_supplier,
        "failed": failed,
        "warnings": all_warnings[:20],  # cap to avoid huge output
    }))


def cmd_process_internal(run_id: str):
    """
    Load all CSVs from data/internal/, parse with SimpleOpportunityEngine,
    save to DB under the given run_id.
    """
    from analysis.opportunity_engine import SimpleOpportunityEngine
    from db.database import ProcurementDB

    internal_files = _scan_folder(INTERNAL_DIR)
    if not internal_files:
        print(json.dumps({
            "run_id": run_id,
            "product_count": 0,
            "files_processed": 0,
            "message": "No files found in data/internal/",
        }))
        return

    db = ProcurementDB()
    total_products = 0
    files_processed = 0

    for f in internal_files:
        try:
            engine = SimpleOpportunityEngine()
            with open(f["path"], "r", encoding="utf-8-sig") as fh:
                ok = engine.load_internal_data(fh)
            if ok and engine.internal_data:
                db.save_internal_data(run_id, engine.internal_data)
                total_products += len(engine.internal_data)
                files_processed += 1
                print(f"  ✓ {f['name']} → {len(engine.internal_data)} products", file=sys.stderr)
            else:
                print(f"  ✗ {f['name']}: no valid rows loaded", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ {f['name']}: {e}", file=sys.stderr)

    print(json.dumps({
        "run_id": run_id,
        "product_count": total_products,
        "files_processed": files_processed,
    }))


# ─── entry point ─────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python scripts/ingest_data.py --scan | --suggest <file> | --process '<json>' | --process-internal <run_id>")
        sys.exit(1)

    mode = args[0]

    if mode == "--scan":
        cmd_scan()
    elif mode == "--suggest":
        if len(args) < 2:
            print("--suggest requires a filepath argument", file=sys.stderr)
            sys.exit(1)
        cmd_suggest(args[1])
    elif mode == "--process":
        if len(args) < 2:
            print("--process requires a JSON mapping argument", file=sys.stderr)
            sys.exit(1)
        validity_json = None
        if "--validity-days" in args:
            vi = args.index("--validity-days")
            if vi + 1 < len(args):
                validity_json = args[vi + 1]
        cmd_process(args[1], validity_days_json=validity_json)
    elif mode == "--process-internal":
        if len(args) < 2:
            print("--process-internal requires a run_id argument", file=sys.stderr)
            sys.exit(1)
        cmd_process_internal(args[1])
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
