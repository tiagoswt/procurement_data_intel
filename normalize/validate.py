from typing import List, Optional, Tuple

from models import ProductData


def validate_rows(rows: List[dict]) -> Tuple[List[ProductData], List[str]]:
    """
    Drop rows missing required fields. Deduplicate EANs (keep first).
    Returns (products, warnings).
    """
    products: List[ProductData] = []
    warnings: List[str] = []
    seen_eans: dict = {}

    for row in rows:
        src = f"row {row.get('source_row', '?')} sheet '{row.get('source_sheet', '?')}'"

        price = row.get("price")
        try:
            if price is None or float(price) <= 0:
                continue
        except (ValueError, TypeError):
            continue

        product_name = row.get("product_name")
        if not product_name:
            continue

        ean = row.get("ean")
        if ean:
            if ean in seen_eans:
                warnings.append(
                    f"{src}: duplicate EAN {ean} (first seen at {seen_eans[ean]}) — skipped"
                )
                continue
            seen_eans[ean] = src

        products.append(
            ProductData(
                ean_code=ean,
                supplier_code=None,
                product_name=product_name,
                quantity=_to_int(row.get("quantity")),
                price=float(price),
                supplier=row.get("supplier"),
                confidence_score=1.0,
                source_file=row.get("source_file"),
            )
        )

    return products, warnings


def _to_int(val) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(float(str(val).replace(",", ".")))
    except (ValueError, TypeError):
        return None
