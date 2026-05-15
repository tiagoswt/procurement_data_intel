"""
Normalize Profile Wizard

Usage:
    python -m normalize.wizard suggest <file.xlsx>
    python -m normalize.wizard suggest <file.xlsx> --save

Reads an unknown Excel file and asks Groq to suggest a normalize profile YAML.
Outputs to stdout or saves to profiles/<supplier_code>.draft.yaml.
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd


def extract_sheets_info(file_path: str, max_sheets: int = 5) -> List[dict]:
    """
    Read headers and up to 3 sample rows from each sheet.
    Auto-detects whether headers are on row 1 or row 2.
    Returns list of dicts: {sheet, header_row, columns, samples}.
    """
    xl = pd.ExcelFile(file_path, engine="openpyxl")
    results = []

    try:
        for sheet_name in xl.sheet_names[:max_sheets]:
            info = _read_sheet(xl, sheet_name)
            if info:
                results.append(info)
    finally:
        xl.close()

    return results


def _read_sheet(xl: pd.ExcelFile, sheet_name: str) -> Optional[dict]:
    """Try header on row 1, then row 2. Return info dict or None if unreadable."""
    for header_idx in [0, 1]:  # 0-indexed: row 1, then row 2
        try:
            df = pd.read_excel(xl, sheet_name=sheet_name, header=header_idx, nrows=3, dtype=str)
        except Exception:
            continue

        named_cols = [c for c in df.columns if not str(c).startswith("Unnamed")]
        if len(named_cols) < 2:
            continue  # not enough real column headers — try next header row

        df = df[named_cols]
        df = df.dropna(how="all")

        return {
            "sheet": sheet_name,
            "header_row": header_idx + 1,  # convert to 1-indexed
            "columns": named_cols,
            "samples": df.head(3).fillna("").to_dict(orient="records"),
        }

    return None
