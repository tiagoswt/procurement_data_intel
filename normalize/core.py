import re
from pathlib import Path
from typing import List, Tuple

from models import ProductData

from .detect import detect_supplier
from .exceptions import NormalizeError
from .loader import load_sheets
from .mapper import map_sheet
from .profile import load_profile
from .validate import validate_rows


def ingest(file_path: str) -> Tuple[List[ProductData], List[str]]:
    fp = Path(file_path)
    supplier_code = detect_supplier(fp.name)
    if supplier_code is None:
        raise NormalizeError(f"No profile matched: {fp.name}")

    profile = load_profile(supplier_code)
    sheets = load_sheets(str(fp), profile["sheets"])

    if not sheets:
        raise NormalizeError(f"No matching sheets found in {fp.name}")

    all_products: List[ProductData] = []
    all_warnings: List[str] = []
    total_input_rows = 0

    for sheet_name, df, sheet_cfg in sheets:
        total_input_rows += len(df)
        brand = _resolve_brand(profile, sheet_name, fp.name)
        merged_sheet_cfg = {
            "brand_source": profile.get("brand_source", "static"),
            "brand_column": profile.get("brand_column"),
            "brand_regex": profile.get("brand_regex"),
            **sheet_cfg,
        }
        mapped_rows, map_warnings = map_sheet(
            df, merged_sheet_cfg, brand, profile["supplier_name"], str(fp), sheet_name
        )
        all_warnings.extend(map_warnings)
        products, val_warnings = validate_rows(mapped_rows)
        all_products.extend(products)
        all_warnings.extend(val_warnings)

    if not all_products:
        raise NormalizeError(f"No valid rows extracted from {fp.name}")

    if total_input_rows > 0 and len(all_products) / total_input_rows < 0.5:
        raise NormalizeError(
            f"Low yield: {len(all_products)}/{total_input_rows} rows valid in {fp.name}. "
            "Check that profile column names match the file headers."
        )

    return all_products, all_warnings


def _resolve_brand(profile: dict, sheet_name: str, filename: str) -> str:
    source = profile.get("brand_source", "static")
    if source == "static":
        return profile.get("brand_static", profile["supplier_name"])
    if source == "sheet_name":
        return sheet_name.strip().rstrip("_").strip()
    if source == "filename_regex":
        pattern = profile.get("brand_regex", "")
        m = re.match(pattern, filename)
        if m:
            return m.group("brand")
        return profile["supplier_name"]
    if source == "column":
        return ""  # mapper reads brand per-row from brand_column
    if source == "column_regex":
        return ""  # mapper reads brand per-row
    return profile["supplier_name"]
