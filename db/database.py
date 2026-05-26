import re
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
                    ingested_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    valid_until  DATE
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
        self._migrate_schema()
        self._normalize_supplier_names()

    def _migrate_schema(self):
        with self._get_conn() as conn:
            try:
                conn.execute("ALTER TABLE supplier_prices ADD COLUMN valid_until DATE")
            except Exception:
                pass  # column already exists

    def _normalize_supplier_names(self):
        """Retroactively strip _DDMMYYYY date suffix from supplier names already in the DB."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT DISTINCT supplier FROM supplier_prices").fetchall()
            for row in rows:
                old = row[0]
                new = re.sub(r'_\d{8}$', '', old)
                if new != old:
                    conn.execute(
                        "UPDATE supplier_prices SET supplier = ? WHERE supplier = ?",
                        (new, old),
                    )

    def save_supplier_batch(
        self,
        source_files: List[str],
        products,
        validity_days: Optional[Dict[str, Optional[int]]] = None,
    ) -> str:
        """Persist a supplier processing batch. Returns the new run_id.

        validity_days: maps supplier name → number of days the offer is valid.
            None value (or missing key) means no expiry.
        """
        run_id = str(uuid.uuid4())
        run_at = datetime.utcnow().isoformat()
        today = datetime.utcnow().date()
        validity_days = validity_days or {}
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
                    supplier = str(getattr(p, "supplier", "Unknown") or "Unknown")
                    qty = getattr(p, "quantity", None)
                    days = validity_days.get(supplier)
                    valid_until = (today + timedelta(days=days)).isoformat() if days is not None else None
                    conn.execute(
                        "INSERT INTO supplier_prices (ean, supplier, price_net, currency, quantity, run_id, valid_until) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            ean,
                            supplier,
                            float(getattr(p, "price", 0) or 0),
                            "EUR",
                            int(qty) if qty is not None else None,
                            run_id,
                            valid_until,
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
        return [
            {**dict(r), "supplier": re.sub(r'_\d{8}$', '', r["supplier"] or "")}
            for r in rows
        ]

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
        return [
            {**dict(r), "supplier": re.sub(r'_\d{8}$', '', r["supplier"] or "")}
            for r in rows
        ]

    def get_distinct_suppliers(self) -> List[str]:
        """Return all distinct supplier names stored in the DB."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT supplier FROM supplier_prices ORDER BY supplier"
            ).fetchall()
        return [re.sub(r'_\d{8}$', '', row[0] or "").strip() for row in rows if row[0]]

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

    def clear_all_data(self) -> None:
        """Delete all rows from every table. Irreversible."""
        with self._get_conn() as conn:
            conn.executescript("""
                DELETE FROM supplier_prices;
                DELETE FROM internal_data;
                DELETE FROM products;
                DELETE FROM processing_runs;
            """)
        logger.info("All database data cleared.")
