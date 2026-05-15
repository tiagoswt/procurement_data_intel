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
from groq import Groq


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


_SYSTEM_PROMPT = """\
You are a data integration specialist. Given Excel headers and sample rows from a supplier price list, output a normalize profile YAML.

Profile schema:
```yaml
supplier_code: snake_case_id
supplier_name: "Human Readable Name"
match:
  filename_patterns:
    - "*PartialMatch*"
brand_source: static|column|column_regex|filename_regex|sheet_name
brand_static: "Brand"        # if brand_source: static
brand_column: "Col Name"     # if brand_source: column or column_regex
brand_regex: "^(?P<brand>.+?) -"  # if brand_source: column_regex or filename_regex
defaults:
  currency: EUR
sheets:
  - match: "*"
    header_row: 1
    decimal_sep: "." or ","
    columns:
      ean: "Exact EAN/Barcode column header"
      product_name: "Exact product description column header"
      price: "Exact price column header"
      quantity: null
```

Rules:
- supplier_code: snake_case, no spaces
- filename_patterns: fnmatch — e.g. "*SupplierName*"
- header_row: 1-indexed integer
- decimal_sep: "," only if prices use European format like "1.234,56", else "."
- EAN codes are 8, 12, 13, or 14 digit barcodes — look for columns named EAN, Barcode, GTIN, UPC, Code
- quantity: null if no stock/qty column
- brand_source: "static" if one brand; "sheet_name" if brand=sheet name; "column" if there's a brand/MARCA column
- Use separate sheet configs (list items) only if sheets have different structures
- Respond ONLY with valid YAML — no explanation, no markdown fences\
"""


def format_prompt(filename: str, sheets_info: List[dict]) -> str:
    """Format sheet info into a user message for Groq."""
    lines = [f"File: {filename}", ""]
    for info in sheets_info:
        lines.append(f"Sheet: {info['sheet']} (header row {info['header_row']})")
        lines.append(f"Columns: {info['columns']}")
        if info["samples"]:
            lines.append("Sample rows:")
            for row in info["samples"]:
                lines.append(f"  {dict(row)}")
        lines.append("")
    return "\n".join(lines)


def suggest_profile(file_path: str) -> str:
    """
    Read an Excel file and ask Groq to suggest a normalize profile YAML.
    Returns the raw YAML string.
    Requires GROQ_API_KEY environment variable.
    Raises: ValueError if no sheets could be read; RuntimeError if Groq returns empty response.
    """
    sheets_info = extract_sheets_info(file_path)
    if not sheets_info:
        raise ValueError(f"Could not read any sheets from {file_path}")

    prompt = format_prompt(Path(file_path).name, sheets_info)

    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    if not response.choices or not response.choices[0].message:
        raise RuntimeError("Groq returned empty response")
    return response.choices[0].message.content.strip()
