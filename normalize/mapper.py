import re
from typing import List, Optional, Tuple

import pandas as pd

from .parsers.ean import clean_ean
from .parsers.price import parse_price


def map_sheet(
    df: pd.DataFrame,
    sheet_cfg: dict,
    brand: str,
    supplier_name: str,
    source_file: str,
    source_sheet: str,
) -> Tuple[List[dict], List[str]]:
    columns = sheet_cfg.get("columns", {})
    decimal_sep = sheet_cfg.get("decimal_sep", ".")
    brand_source = sheet_cfg.get("brand_source", "static")
    brand_column = sheet_cfg.get("brand_column")
    brand_regex = sheet_cfg.get("brand_regex", "")
    warnings: List[str] = []
    rows: List[dict] = []

    for idx, row in df.iterrows():
        mapped: dict = {
            "supplier": supplier_name,
            "source_file": source_file,
            "source_sheet": source_sheet,
            "source_row": idx,
        }

        if brand_source == "column":
            raw_brand = _cell(row, brand_column) if brand_column else None
            mapped["brand"] = raw_brand or supplier_name
        elif brand_source == "column_regex":
            raw_col = _cell(row, brand_column) if brand_column else None
            if raw_col and brand_regex:
                m = re.match(brand_regex, raw_col)
                mapped["brand"] = m.group("brand").strip() if m else raw_col
            else:
                mapped["brand"] = supplier_name
        else:
            mapped["brand"] = brand or supplier_name

        for field, header in columns.items():
            if header is None:
                mapped[field] = None
                continue
            val = _cell(row, header)
            mapped[field] = val

        raw_ean = mapped.get("ean")
        if raw_ean:
            cleaned = clean_ean(raw_ean)
            if cleaned is None:
                warnings.append(
                    f"Row {idx} (sheet '{source_sheet}'): EAN '{raw_ean}' invalid length — set to None"
                )
            mapped["ean"] = cleaned

        raw_price = mapped.get("price")
        if raw_price is not None:
            mapped["price"] = parse_price(raw_price, decimal_sep)

        rows.append(mapped)

    return rows, warnings


def _cell(row: pd.Series, header: str) -> Optional[str]:
    """Return stripped string value of a cell, or None if missing/NaN."""
    if header not in row.index:
        return None
    val = row[header]
    if pd.isna(val) if not isinstance(val, str) else not val.strip():
        return None
    return str(val).strip()
