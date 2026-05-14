# Normalization Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a profile-driven `normalize/` package that reads any supplier Excel file and returns `List[ProductData]`, replacing the current heuristic pipeline.

**Architecture:** A `normalize/` Python package exposes a single `ingest(file_path) -> (List[ProductData], List[str])` function. Supplier-specific knowledge lives in YAML files under `profiles/`. The pipeline is: detect supplier via filename → load YAML profile → read Excel → map columns → parse EAN/price → validate rows → return ProductData list. No LLM involvement at runtime.

**Tech Stack:** Python 3.10+, pandas + openpyxl (already in requirements), pyyaml (to add), pytest (already in requirements).

---

## File Structure

**New files to create:**
- `normalize/__init__.py` — `NormalizeError` + re-export `ingest()`
- `normalize/core.py` — `ingest()` orchestrator
- `normalize/detect.py` — filename → supplier_code
- `normalize/profile.py` — load YAML profiles
- `normalize/loader.py` — Excel multi-sheet reader
- `normalize/mapper.py` — column mapping + brand resolution
- `normalize/validate.py` — row validation + warning collection
- `normalize/parsers/__init__.py` — empty
- `normalize/parsers/ean.py` — EAN cleaning/validation
- `normalize/parsers/price.py` — decimal separator normalization
- `normalize/cli.py` — thin CLI wrapper
- `profiles/colorwow.yaml`
- `profiles/hispalbeauty.yaml`
- `profiles/oferta_drogueria.yaml`
- `profiles/argon_trading.yaml`
- `profiles/kbeauty_eu.yaml`
- `tests/normalize/__init__.py`
- `tests/normalize/conftest.py` — golden test helpers
- `tests/normalize/golden/` — JSON snapshots (generated, then committed)
- `tests/normalize/test_parsers.py`
- `tests/normalize/test_detect.py`
- `tests/normalize/test_profiles.py`

**Files to modify:**
- `requirements.txt` — add `pyyaml>=6.0`

---

## Task 1: Scaffold `normalize/` package + add pyyaml

**Files:**
- Create: `normalize/__init__.py`
- Create: `normalize/parsers/__init__.py`
- Create: `tests/normalize/__init__.py`
- Create: `tests/normalize/conftest.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add pyyaml to requirements**

Open `requirements.txt` and add after the `# Data processing` section:

```
pyyaml>=6.0
```

- [ ] **Step 2: Install pyyaml**

```bash
pip install pyyaml>=6.0
```

Expected: `Successfully installed pyyaml-6.x`

- [ ] **Step 3: Create `normalize/__init__.py`**

```python
from .core import ingest


class NormalizeError(Exception):
    pass


__all__ = ["ingest", "NormalizeError"]
```

- [ ] **Step 4: Create `normalize/parsers/__init__.py`**

```python
```

(empty file)

- [ ] **Step 5: Create `tests/normalize/__init__.py`**

```python
```

(empty file)

- [ ] **Step 6: Create `tests/normalize/conftest.py`**

```python
import json
from pathlib import Path

import pytest

from models import ProductData


def pytest_addoption(parser):
    parser.addoption("--generate-golden", action="store_true", default=False)


@pytest.fixture
def generate_golden(request):
    return request.config.getoption("--generate-golden")


def product_to_dict(p: ProductData) -> dict:
    """Comparable dict — omits source_file (machine-specific path)."""
    return {
        "ean_code": p.ean_code,
        "product_name": p.product_name,
        "price": p.price,
        "supplier": p.supplier,
        "quantity": p.quantity,
    }


def load_golden(name: str) -> list:
    path = Path(__file__).parent / "golden" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_golden(name: str, data: list) -> None:
    path = Path(__file__).parent / "golden" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 7: Commit**

```bash
git add normalize/ tests/normalize/ requirements.txt
git commit -m "feat: scaffold normalize package skeleton"
```

---

## Task 2: EAN Parser (TDD)

**Files:**
- Create: `normalize/parsers/ean.py`
- Create: `tests/normalize/test_parsers.py` (EAN section)

- [ ] **Step 1: Write failing tests**

Create `tests/normalize/test_parsers.py`:

```python
from normalize.parsers.ean import clean_ean


def test_clean_ean_standard_13():
    assert clean_ean("5060280378928") == "5060280378928"


def test_clean_ean_strips_float_suffix():
    # Excel stores EANs as floats: 8871676652000.0
    assert clean_ean("8871676652000.0") == "8871676652000"


def test_clean_ean_12_digits():
    assert clean_ean("012345678905") == "012345678905"


def test_clean_ean_8_digits():
    assert clean_ean("12345678") == "12345678"


def test_clean_ean_14_digits():
    assert clean_ean("01234567890128") == "01234567890128"


def test_clean_ean_invalid_5_digits():
    assert clean_ean("12345") is None


def test_clean_ean_none():
    assert clean_ean(None) is None


def test_clean_ean_empty_string():
    assert clean_ean("") is None


def test_clean_ean_non_numeric_stripped():
    # Some suppliers wrap EAN in spaces or non-breaking chars
    assert clean_ean(" 5060280378928 ") == "5060280378928"


def test_clean_ean_float_input():
    # pandas may produce a float object
    assert clean_ean(5060280378928.0) == "5060280378928"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/normalize/test_parsers.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` (function doesn't exist yet)

- [ ] **Step 3: Implement `normalize/parsers/ean.py`**

```python
import re
from typing import Optional

_VALID_LENGTHS = {8, 12, 13, 14}


def clean_ean(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    if len(digits) in _VALID_LENGTHS:
        return digits
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/normalize/test_parsers.py -v
```

Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add normalize/parsers/ean.py tests/normalize/test_parsers.py
git commit -m "feat: add EAN parser with length validation"
```

---

## Task 3: Price Parser (TDD)

**Files:**
- Modify: `tests/normalize/test_parsers.py` (add price section)
- Create: `normalize/parsers/price.py`

- [ ] **Step 1: Add failing price tests to `tests/normalize/test_parsers.py`**

Append to the file:

```python
from normalize.parsers.price import parse_price


def test_parse_price_plain_float_string():
    assert parse_price("29.99") == 29.99


def test_parse_price_numeric_float():
    assert parse_price(29.99) == 29.99


def test_parse_price_numeric_int():
    assert parse_price(30) == 30.0


def test_parse_price_comma_decimal_sep():
    assert parse_price("25,50", decimal_sep=",") == 25.50


def test_parse_price_european_thousands():
    # "1.234,56" with decimal_sep="," → 1234.56
    assert parse_price("1.234,56", decimal_sep=",") == 1234.56


def test_parse_price_currency_symbol():
    assert parse_price("€29.99") == 29.99


def test_parse_price_zero_returns_none():
    assert parse_price("0") is None


def test_parse_price_zero_float_returns_none():
    assert parse_price(0.0) is None


def test_parse_price_none_returns_none():
    assert parse_price(None) is None


def test_parse_price_empty_string_returns_none():
    assert parse_price("") is None


def test_parse_price_negative_returns_none():
    assert parse_price("-5.00") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/normalize/test_parsers.py::test_parse_price_plain_float_string -v
```

Expected: `ImportError` (function doesn't exist)

- [ ] **Step 3: Implement `normalize/parsers/price.py`**

```python
import re
from typing import Optional


def parse_price(raw, decimal_sep: str = ".") -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if val > 0 else None
    s = str(raw).strip()
    s = re.sub(r"[^\d.,]", "", s)
    if not s:
        return None
    if decimal_sep == ",":
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        val = float(s)
        return val if val > 0 else None
    except ValueError:
        return None
```

- [ ] **Step 4: Run all parser tests**

```bash
pytest tests/normalize/test_parsers.py -v
```

Expected: all 21 tests PASS

- [ ] **Step 5: Commit**

```bash
git add normalize/parsers/price.py tests/normalize/test_parsers.py
git commit -m "feat: add price parser with decimal separator support"
```

---

## Task 4: `profile.py` and `detect.py`

**Files:**
- Create: `normalize/profile.py`
- Create: `normalize/detect.py`
- Create: `tests/normalize/test_detect.py`
- Create: `profiles/` directory (empty, will hold YAMLs)

- [ ] **Step 1: Create `normalize/profile.py`**

```python
from pathlib import Path
from typing import List

import yaml

PROFILES_DIR = Path(__file__).parent.parent / "profiles"


def list_profiles() -> List[str]:
    return [p.stem for p in PROFILES_DIR.glob("*.yaml")]


def load_profile(supplier_code: str) -> dict:
    path = PROFILES_DIR / f"{supplier_code}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 2: Create `normalize/detect.py`**

```python
from fnmatch import fnmatch
from typing import Optional

from .profile import list_profiles, load_profile


def detect_supplier(filename: str) -> Optional[str]:
    for code in list_profiles():
        profile = load_profile(code)
        patterns = profile.get("match", {}).get("filename_patterns", [])
        for pattern in patterns:
            if fnmatch(filename, pattern):
                return code
    return None
```

- [ ] **Step 3: Create the `profiles/` directory**

```bash
mkdir profiles
```

- [ ] **Step 4: Create a minimal test profile for detect tests**

Create `profiles/_test_supplier.yaml`:

```yaml
supplier_code: _test_supplier
supplier_name: "Test Supplier"
match:
  filename_patterns:
    - "*test_offer*"
    - "*TestCo*"
brand_source: static
brand_static: "Test Brand"
defaults:
  currency: EUR
sheets:
  - match: "*"
    header_row: 1
    decimal_sep: "."
    columns:
      ean: "EAN"
      product_name: "Name"
      price: "Price"
      quantity: ~
```

- [ ] **Step 5: Write `tests/normalize/test_detect.py`**

```python
from normalize.detect import detect_supplier


def test_detect_matches_test_supplier():
    assert detect_supplier("test_offer_2026.xlsx") == "_test_supplier"


def test_detect_matches_second_pattern():
    assert detect_supplier("TestCo_products.xlsx") == "_test_supplier"


def test_detect_no_match_returns_none():
    assert detect_supplier("completely_unknown_file.xlsx") is None


def test_detect_case_sensitive():
    # fnmatch is case-sensitive on Linux; test the pattern as written
    assert detect_supplier("TEST_OFFER_2026.xlsx") is None
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/normalize/test_detect.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add normalize/profile.py normalize/detect.py profiles/_test_supplier.yaml tests/normalize/test_detect.py
git commit -m "feat: add profile loader and filename-based supplier detection"
```

---

## Task 5: `loader.py`

**Files:**
- Create: `normalize/loader.py`

No separate unit test for loader — it's tested implicitly through the profile integration tests (Tasks 10-14). The function has one responsibility: read Excel and return DataFrames per sheet.

- [ ] **Step 1: Create `normalize/loader.py`**

```python
from fnmatch import fnmatch
from typing import List, Tuple

import pandas as pd


def load_sheets(
    file_path: str, sheets_config: List[dict]
) -> List[Tuple[str, pd.DataFrame, dict]]:
    """
    Returns (sheet_name, DataFrame, sheet_config) for every sheet
    that matches a config entry. header_row in config is 1-indexed.
    """
    xl = pd.ExcelFile(file_path, engine="openpyxl")
    results = []
    for sheet_name in xl.sheet_names:
        for sheet_cfg in sheets_config:
            pattern = sheet_cfg.get("match", "*")
            if fnmatch(sheet_name, pattern):
                header_idx = sheet_cfg.get("header_row", 1) - 1  # pandas is 0-indexed
                df = pd.read_excel(
                    xl,
                    sheet_name=sheet_name,
                    header=header_idx,
                    dtype=str,
                )
                df = df.dropna(how="all")
                results.append((sheet_name, df, sheet_cfg))
                break  # first matching config wins
    return results
```

- [ ] **Step 2: Commit**

```bash
git add normalize/loader.py
git commit -m "feat: add Excel multi-sheet loader with configurable header row"
```

---

## Task 6: `mapper.py`

**Files:**
- Create: `normalize/mapper.py`

- [ ] **Step 1: Create `normalize/mapper.py`**

```python
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
    """
    Map a DataFrame's rows to dicts with canonical keys.
    brand may be empty string when brand_source=="column" — handled per-row below.
    Returns (rows, warnings).
    """
    columns = sheet_cfg.get("columns", {})
    decimal_sep = sheet_cfg.get("decimal_sep", ".")
    brand_column = sheet_cfg.get("brand_column")
    warnings: List[str] = []
    rows: List[dict] = []

    for idx, row in df.iterrows():
        mapped: dict = {
            "supplier": supplier_name,
            "source_file": source_file,
            "source_sheet": source_sheet,
            "source_row": idx,
        }

        # Resolve brand: pre-resolved, or from a row column
        if brand:
            mapped["brand"] = brand
        elif brand_column and brand_column in df.columns:
            raw_brand = _cell(row, brand_column)
            mapped["brand"] = raw_brand or supplier_name
        else:
            mapped["brand"] = supplier_name

        # Map declared columns
        for field, header in columns.items():
            if header is None:
                mapped[field] = None
                continue
            val = _cell(row, header)
            mapped[field] = val

        # Parse EAN
        raw_ean = mapped.get("ean")
        if raw_ean:
            cleaned = clean_ean(raw_ean)
            if cleaned is None:
                warnings.append(
                    f"Row {idx} (sheet '{source_sheet}'): EAN '{raw_ean}' invalid length — set to None"
                )
            mapped["ean"] = cleaned

        # Parse price
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
```

- [ ] **Step 2: Commit**

```bash
git add normalize/mapper.py
git commit -m "feat: add column mapper with EAN/price parsing and brand resolution"
```

---

## Task 7: `validate.py`

**Files:**
- Create: `normalize/validate.py`

- [ ] **Step 1: Create `normalize/validate.py`**

```python
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
        if price is None or float(price) <= 0:
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
```

- [ ] **Step 2: Commit**

```bash
git add normalize/validate.py
git commit -m "feat: add row validator with EAN deduplication"
```

---

## Task 8: `core.py` — `ingest()` orchestrator

**Files:**
- Create: `normalize/core.py`
- Modify: `normalize/__init__.py` (NormalizeError must be importable before core imports it)

- [ ] **Step 1: Restructure `normalize/__init__.py` to avoid circular import**

The issue: `core.py` imports `NormalizeError` from `normalize/__init__.py`, but `__init__.py` imports `ingest` from `core.py`. Fix by defining `NormalizeError` in its own module.

Create `normalize/exceptions.py`:

```python
class NormalizeError(Exception):
    pass
```

Update `normalize/__init__.py`:

```python
from .exceptions import NormalizeError
from .core import ingest

__all__ = ["ingest", "NormalizeError"]
```

- [ ] **Step 2: Create `normalize/core.py`**

```python
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
        mapped_rows, map_warnings = map_sheet(
            df, sheet_cfg, brand, profile["supplier_name"], str(fp), sheet_name
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
    return profile["supplier_name"]
```

- [ ] **Step 3: Verify import works**

```bash
python -c "from normalize import ingest, NormalizeError; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add normalize/exceptions.py normalize/core.py normalize/__init__.py
git commit -m "feat: add ingest() orchestrator wiring detect→load→map→validate"
```

---

## Task 9: Phase 1 — Inspect file headers

Before writing profiles, inspect the actual column names in the Phase 1 files. This step produces no code — just information needed for Tasks 10-12.

- [ ] **Step 1: Inspect ColorWow headers**

```bash
python -c "
import pandas as pd
xl = pd.ExcelFile('data/ColorWow offer .xlsx', engine='openpyxl')
print('Sheets:', xl.sheet_names)
df = pd.read_excel(xl, sheet_name=xl.sheet_names[0], header=0, nrows=3, dtype=str)
print(df.columns.tolist())
print(df.head(2).to_string())
"
```

Record the output — you need the exact EAN, product_name, and price column header strings.

- [ ] **Step 2: Inspect Hispalbeauty headers**

```bash
python -c "
import pandas as pd
xl = pd.ExcelFile('data/STOCK HISPALBEAUTY ONTHEFLOOR  08052026.xlsx', engine='openpyxl')
print('Sheets:', xl.sheet_names)
df = pd.read_excel(xl, sheet_name=xl.sheet_names[0], header=0, nrows=3, dtype=str)
print(df.columns.tolist())
print(df.head(2).to_string())
"
```

Record the EAN column, product name column, price column, brand column (MARCA), and quantity/stock column.

- [ ] **Step 3: Inspect Droguería headers**

```bash
python -c "
import pandas as pd
xl = pd.ExcelFile('data/OFERTA_DROGUERIA_140526.xlsx', engine='openpyxl')
print('Sheets:', xl.sheet_names)
df = pd.read_excel(xl, sheet_name=xl.sheet_names[0], header=0, nrows=5, dtype=str)
print(df.columns.tolist())
print(df.head(3).to_string())
"
```

Record all column headers. Note the decimal separator used in the price column (should be comma `,`).

---

## Task 10: Phase 1 — ColorWow profile + golden test

**Files:**
- Create: `profiles/colorwow.yaml`
- Modify: `tests/normalize/test_profiles.py` (create if first profile)

Replace `<EAN_COL>`, `<NAME_COL>`, `<PRICE_COL>` with the exact headers found in Task 9.

- [ ] **Step 1: Create `profiles/colorwow.yaml`**

```yaml
supplier_code: colorwow
supplier_name: "Color Wow"

match:
  filename_patterns:
    - "*ColorWow*"
    - "*Color Wow*"

brand_source: static
brand_static: "Color Wow"

defaults:
  currency: EUR

sheets:
  - match: "*"
    header_row: 1
    decimal_sep: "."
    columns:
      ean:          "<EAN_COL>"
      product_name: "<NAME_COL>"
      price:        "<PRICE_COL>"
      quantity:     ~
```

- [ ] **Step 2: Verify profile detects file**

```bash
python -c "
from normalize.detect import detect_supplier
print(detect_supplier('ColorWow offer .xlsx'))
"
```

Expected: `colorwow`

- [ ] **Step 3: Run ingest and inspect output**

```bash
python -c "
from normalize import ingest
products, warnings = ingest('data/ColorWow offer .xlsx')
print(f'Products: {len(products)}')
print(f'Warnings: {len(warnings)}')
for p in products[:3]:
    print(p.ean_code, p.product_name, p.price)
for w in warnings[:5]:
    print('WARN:', w)
"
```

Verify: product count looks right, EANs are 13 digits, prices are positive.

- [ ] **Step 4: Generate golden file**

Create `tests/normalize/test_profiles.py`:

```python
import json
from pathlib import Path

import pytest

from normalize import ingest
from tests.normalize.conftest import load_golden, product_to_dict, save_golden

DATA_DIR = Path("data")


def test_colorwow(generate_golden):
    products, warnings = ingest(str(DATA_DIR / "ColorWow offer .xlsx"))
    actual = [product_to_dict(p) for p in products]
    if generate_golden:
        save_golden("colorwow", actual)
        return
    assert actual == load_golden("colorwow")
```

```bash
pytest tests/normalize/test_profiles.py::test_colorwow --generate-golden -v
```

Expected: test passes (golden file created at `tests/normalize/golden/colorwow.json`)

- [ ] **Step 5: Run test against golden**

```bash
pytest tests/normalize/test_profiles.py::test_colorwow -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add profiles/colorwow.yaml tests/normalize/test_profiles.py tests/normalize/golden/colorwow.json
git commit -m "feat: add ColorWow profile and golden test"
```

---

## Task 11: Phase 1 — Hispalbeauty profile + golden test

**Files:**
- Create: `profiles/hispalbeauty.yaml`
- Modify: `tests/normalize/test_profiles.py`

Replace `<EAN_COL>`, `<NAME_COL>`, `<PRICE_COL>`, `<STOCK_COL>` with headers from Task 9.

- [ ] **Step 1: Create `profiles/hispalbeauty.yaml`**

```yaml
supplier_code: hispalbeauty
supplier_name: "Hispalbeauty"

match:
  filename_patterns:
    - "*HISPALBEAUTY*"
    - "*hispalbeauty*"

brand_source: column
brand_column: "MARCA"

defaults:
  currency: EUR

sheets:
  - match: "*"
    header_row: 1
    decimal_sep: "."
    brand_column: "MARCA"
    columns:
      ean:          "<EAN_COL>"
      product_name: "<NAME_COL>"
      price:        "<PRICE_COL>"
      quantity:     "<STOCK_COL>"
```

- [ ] **Step 2: Verify detection**

```bash
python -c "
from normalize.detect import detect_supplier
print(detect_supplier('STOCK HISPALBEAUTY ONTHEFLOOR  08052026.xlsx'))
"
```

Expected: `hispalbeauty`

- [ ] **Step 3: Run ingest and inspect**

```bash
python -c "
from normalize import ingest
products, warnings = ingest('data/STOCK HISPALBEAUTY ONTHEFLOOR  08052026.xlsx')
print(f'Products: {len(products)}')
for p in products[:3]:
    print(p.ean_code, p.supplier, p.product_name[:40], p.price, p.quantity)
"
```

Verify: brands are populated from MARCA column, quantities are integers.

- [ ] **Step 4: Generate golden file**

Add test to `tests/normalize/test_profiles.py`:

```python
def test_hispalbeauty(generate_golden):
    products, warnings = ingest(str(DATA_DIR / "STOCK HISPALBEAUTY ONTHEFLOOR  08052026.xlsx"))
    actual = [product_to_dict(p) for p in products]
    if generate_golden:
        save_golden("hispalbeauty", actual)
        return
    assert actual == load_golden("hispalbeauty")
```

```bash
pytest tests/normalize/test_profiles.py::test_hispalbeauty --generate-golden -v
```

Expected: golden file created.

- [ ] **Step 5: Run test against golden**

```bash
pytest tests/normalize/test_profiles.py::test_hispalbeauty -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add profiles/hispalbeauty.yaml tests/normalize/test_profiles.py tests/normalize/golden/hispalbeauty.json
git commit -m "feat: add Hispalbeauty profile and golden test"
```

---

## Task 12: Phase 1 — Droguería profile + golden test

**Files:**
- Create: `profiles/oferta_drogueria.yaml`
- Modify: `tests/normalize/test_profiles.py`

Droguería uses comma as decimal separator and brand must be extracted from the product name column.

Replace `<EAN_COL>`, `<NAME_COL>`, `<PRICE_COL>`, `<BRAND_REGEX>` with values from Task 9. A reasonable default brand regex is `"^(?P<brand>[A-ZÁÉÍÓÚ][A-ZÁÉÍÓÚ0-9 -]+?)\s+[A-Z]"` — adjust based on actual data.

- [ ] **Step 1: Create `profiles/oferta_drogueria.yaml`**

```yaml
supplier_code: oferta_drogueria
supplier_name: "Droguería"

match:
  filename_patterns:
    - "*DROGUERIA*"
    - "*drogueria*"
    - "*OFERTA_DROGUERIA*"

brand_source: column_regex
brand_column: "<NAME_COL>"
brand_regex: "^(?P<brand>[A-ZÁÉÍÓÚ][A-ZÁÉÍÓÚ0-9 \\-]+?)\\s"

defaults:
  currency: EUR

sheets:
  - match: "*"
    header_row: 1
    decimal_sep: ","
    columns:
      ean:          "<EAN_COL>"
      product_name: "<NAME_COL>"
      price:        "<PRICE_COL>"
      quantity:     ~
```

- [ ] **Step 2: Add `column_regex` support to `normalize/core.py`**

In `_resolve_brand`, add the `column_regex` case (returns empty string — handled in mapper):

The `column_regex` source is handled in `mapper.py`. Update `normalize/mapper.py` to support it.

In `map_sheet`, after the brand resolution block, add:

```python
# column_regex: extract brand from a row column using regex
brand_source = sheet_cfg.get("brand_source") or ""
if not brand and brand_source == "column_regex":
    brand_col = sheet_cfg.get("brand_column")
    brand_rx = sheet_cfg.get("brand_regex", "")
    if brand_col and brand_col in df.columns:
        pass  # handled per-row below
```

Actually, `brand_source` and `brand_column`/`brand_regex` need to be passed down from the profile. Pass them as part of `sheet_cfg` (the profile YAML puts them at the sheet level in the profile YAML above, or at the top level). Since the YAML above puts them at top level, update `core.py` to merge top-level brand keys into each `sheet_cfg` when calling `map_sheet`.

Update `normalize/core.py`, inside the loop over sheets, before calling `map_sheet`:

```python
# Merge top-level brand config into sheet_cfg for mapper access
merged_sheet_cfg = {
    "brand_source": profile.get("brand_source", "static"),
    "brand_column": profile.get("brand_column"),
    "brand_regex": profile.get("brand_regex"),
    **sheet_cfg,
}
mapped_rows, map_warnings = map_sheet(
    df, merged_sheet_cfg, brand, profile["supplier_name"], str(fp), sheet_name
)
```

Update `normalize/mapper.py` to handle `column_regex` per-row:

```python
# After resolving brand from pre-resolved value or column:
brand_source = sheet_cfg.get("brand_source", "static")
if brand_source == "column_regex":
    brand_col = sheet_cfg.get("brand_column")
    brand_rx = sheet_cfg.get("brand_regex", "")
    raw_col = _cell(row, brand_col) if brand_col else None
    if raw_col and brand_rx:
        import re
        m = re.match(brand_rx, raw_col)
        mapped["brand"] = m.group("brand").strip() if m else raw_col
    else:
        mapped["brand"] = supplier_name
else:
    # Already set above
    pass
```

Full updated `map_sheet` function with all brand source types integrated:

```python
def map_sheet(
    df: pd.DataFrame,
    sheet_cfg: dict,
    brand: str,
    supplier_name: str,
    source_file: str,
    source_sheet: str,
) -> Tuple[List[dict], List[str]]:
    import re as _re

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
                m = _re.match(brand_regex, raw_col)
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
```

- [ ] **Step 3: Verify detection**

```bash
python -c "
from normalize.detect import detect_supplier
print(detect_supplier('OFERTA_DROGUERIA_140526.xlsx'))
"
```

Expected: `oferta_drogueria`

- [ ] **Step 4: Run ingest and inspect**

```bash
python -c "
from normalize import ingest
products, warnings = ingest('data/OFERTA_DROGUERIA_140526.xlsx')
print(f'Products: {len(products)}')
for p in products[:3]:
    print(p.ean_code, p.supplier, p.product_name[:40], p.price)
for w in warnings[:5]:
    print('WARN:', w)
"
```

Verify: prices use `.` as decimal (e.g. `2.5`, not `250000.0`), brands are populated.

- [ ] **Step 5: Generate golden**

Add to `tests/normalize/test_profiles.py`:

```python
def test_oferta_drogueria(generate_golden):
    products, warnings = ingest(str(DATA_DIR / "OFERTA_DROGUERIA_140526.xlsx"))
    actual = [product_to_dict(p) for p in products]
    if generate_golden:
        save_golden("oferta_drogueria", actual)
        return
    assert actual == load_golden("oferta_drogueria")
```

```bash
pytest tests/normalize/test_profiles.py::test_oferta_drogueria --generate-golden -v
pytest tests/normalize/test_profiles.py::test_oferta_drogueria -v
```

Expected: PASS

- [ ] **Step 6: Run all Phase 1 tests**

```bash
pytest tests/normalize/ -v
```

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add profiles/oferta_drogueria.yaml normalize/mapper.py normalize/core.py tests/normalize/test_profiles.py tests/normalize/golden/oferta_drogueria.json
git commit -m "feat: add Drogueria profile + column_regex brand support"
```

---

## Task 13: Phase 2 — Argon Trading profile + golden tests (3 files)

**Files:**
- Create: `profiles/argon_trading.yaml`
- Modify: `tests/normalize/test_profiles.py`

One profile handles all three Argon Trading files. Brand is extracted from the filename via regex.

- [ ] **Step 1: Inspect Argon Trading headers**

```bash
python -c "
import pandas as pd
xl = pd.ExcelFile('data/Garnier - Argon Trading (1).xlsx', engine='openpyxl')
print('Sheets:', xl.sheet_names)
df = pd.read_excel(xl, sheet_name=xl.sheet_names[0], header=0, nrows=3, dtype=str)
print(df.columns.tolist())
print(df.head(2).to_string())
"
```

Record exact column names for EAN, product name, price, pack_size (pcs/box), pallet_size (pcs/pal).

- [ ] **Step 2: Create `profiles/argon_trading.yaml`**

Replace `<EAN_COL>`, `<NAME_COL>`, `<PRICE_COL>` with exact headers from Step 1.

```yaml
supplier_code: argon_trading
supplier_name: "Argon Trading"

match:
  filename_patterns:
    - "*Argon Trading*"
    - "*- Argon Trading*"

brand_source: filename_regex
brand_regex: "^(?P<brand>.+?) - Argon Trading"

defaults:
  currency: EUR

sheets:
  - match: "*"
    header_row: 1
    decimal_sep: "."
    columns:
      ean:          "<EAN_COL>"
      product_name: "<NAME_COL>"
      price:        "<PRICE_COL>"
      quantity:     ~
```

- [ ] **Step 3: Verify all three files detect correctly**

```bash
python -c "
from normalize.detect import detect_supplier
files = [
    'Garnier - Argon Trading (1).xlsx',
    'Estee Lauder - Argon Trading  .xlsx',
    \"Johnson's Baby - Argon Trading (1).xlsx\",
]
for f in files:
    print(f, '->', detect_supplier(f))
"
```

Expected: all three → `argon_trading`

- [ ] **Step 4: Run ingest on Garnier and inspect**

```bash
python -c "
from normalize import ingest
products, warnings = ingest('data/Garnier - Argon Trading (1).xlsx')
print(f'Products: {len(products)}')
for p in products[:3]:
    print(p.ean_code, p.supplier, p.product_name[:50], p.price)
"
```

Verify: `p.supplier` is `"Argon Trading"`, brands extracted from filename (see `product_to_dict` — brand is on the mapped row but not on ProductData; verify the supplier field is correct).

- [ ] **Step 5: Generate golden files for all three files**

Add to `tests/normalize/test_profiles.py`:

```python
def test_argon_garnier(generate_golden):
    products, warnings = ingest(str(DATA_DIR / "Garnier - Argon Trading (1).xlsx"))
    actual = [product_to_dict(p) for p in products]
    if generate_golden:
        save_golden("argon_garnier", actual)
        return
    assert actual == load_golden("argon_garnier")


def test_argon_estee_lauder(generate_golden):
    products, warnings = ingest(str(DATA_DIR / "Estee Lauder - Argon Trading  .xlsx"))
    actual = [product_to_dict(p) for p in products]
    if generate_golden:
        save_golden("argon_estee_lauder", actual)
        return
    assert actual == load_golden("argon_estee_lauder")


def test_argon_johnsons(generate_golden):
    products, warnings = ingest(str(DATA_DIR / "Johnson's Baby - Argon Trading (1).xlsx"))
    actual = [product_to_dict(p) for p in products]
    if generate_golden:
        save_golden("argon_johnsons", actual)
        return
    assert actual == load_golden("argon_johnsons")
```

```bash
pytest tests/normalize/test_profiles.py::test_argon_garnier tests/normalize/test_profiles.py::test_argon_estee_lauder tests/normalize/test_profiles.py::test_argon_johnsons --generate-golden -v
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/normalize/ -v
```

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add profiles/argon_trading.yaml tests/normalize/test_profiles.py tests/normalize/golden/
git commit -m "feat: add Argon Trading profile covering Garnier, Estee Lauder, Johnson's Baby"
```

---

## Task 14: Phase 3 — K-Beauty profile + golden test

K-Beauty is the stress test: 33 sheets, header row 2, brand from sheet name (strip trailing `_`), price column `Supply Price (excl.vat)`.

**Files:**
- Create: `profiles/kbeauty_eu.yaml`
- Modify: `tests/normalize/test_profiles.py`

- [ ] **Step 1: Inspect K-Beauty structure**

```bash
python -c "
import pandas as pd
xl = pd.ExcelFile('data/K-Beauty offer_EU DDP_Mar. 2026.xlsx', engine='openpyxl')
print('Sheet count:', len(xl.sheet_names))
print('First 5 sheets:', xl.sheet_names[:5])
# Read first sheet with header=1 (row index 1 = 2nd row)
df = pd.read_excel(xl, sheet_name=xl.sheet_names[0], header=1, nrows=3, dtype=str)
print(df.columns.tolist())
print(df.head(2).to_string())
"
```

Verify that `header=1` (0-indexed) gives sensible column names and that `Supply Price (excl.vat)` appears. Record the exact EAN column name.

- [ ] **Step 2: Create `profiles/kbeauty_eu.yaml`**

Replace `<EAN_COL>` and `<NAME_COL>` with exact headers found in Step 1.

```yaml
supplier_code: kbeauty_eu
supplier_name: "K-Beauty EU"

match:
  filename_patterns:
    - "*K-Beauty*"
    - "*K Beauty*"

brand_source: sheet_name

defaults:
  currency: EUR

sheets:
  - match: "*"
    header_row: 2
    decimal_sep: "."
    columns:
      ean:          "<EAN_COL>"
      product_name: "<NAME_COL>"
      price:        "Supply Price (excl.vat)"
      quantity:     ~
```

- [ ] **Step 3: Verify detection**

```bash
python -c "
from normalize.detect import detect_supplier
print(detect_supplier('K-Beauty offer_EU DDP_Mar. 2026.xlsx'))
"
```

Expected: `kbeauty_eu`

- [ ] **Step 4: Run ingest and inspect**

```bash
python -c "
from normalize import ingest
products, warnings = ingest('data/K-Beauty offer_EU DDP_Mar. 2026.xlsx')
print(f'Products: {len(products)}')
print(f'Warnings: {len(warnings)}')
suppliers_seen = set(p.supplier for p in products)
print('Supplier:', suppliers_seen)
# Show 3 products from first sheet
for p in products[:3]:
    print(p.ean_code, p.supplier, p.product_name[:40], p.price)
"
```

Verify: product count is plausible (hundreds), supplier is `"K-Beauty EU"`, EANs are 13 digits.

If brand from sheet_name is needed on the product, note that `brand` is mapped internally but not stored on `ProductData` — `ProductData.supplier` is `"K-Beauty EU"` for all rows. This is correct given the existing model.

- [ ] **Step 5: Generate golden file**

Add to `tests/normalize/test_profiles.py`:

```python
def test_kbeauty_eu(generate_golden):
    products, warnings = ingest(str(DATA_DIR / "K-Beauty offer_EU DDP_Mar. 2026.xlsx"))
    actual = [product_to_dict(p) for p in products]
    if generate_golden:
        save_golden("kbeauty_eu", actual)
        return
    assert actual == load_golden("kbeauty_eu")
```

```bash
pytest tests/normalize/test_profiles.py::test_kbeauty_eu --generate-golden -v
```

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/normalize/ -v
```

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add profiles/kbeauty_eu.yaml tests/normalize/test_profiles.py tests/normalize/golden/kbeauty_eu.json
git commit -m "feat: add K-Beauty profile — 33 sheets, header row 2, brand from sheet name"
```

---

## Task 15: CLI wrapper

**Files:**
- Create: `normalize/cli.py`

- [ ] **Step 1: Create `normalize/cli.py`**

```python
"""
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
    for w in warnings:
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
```

- [ ] **Step 2: Test dry-run on a real file**

```bash
python -m normalize.cli ingest "data/ColorWow offer .xlsx" --dry-run
```

Expected: `Extracted N products, M warnings` followed by up to 10 product lines.

- [ ] **Step 3: Test hard error**

```bash
python -m normalize.cli ingest "data/nonexistent.xlsx" --dry-run
```

Expected: `Error: file not found: data/nonexistent.xlsx`

- [ ] **Step 4: Commit**

```bash
git add normalize/cli.py
git commit -m "feat: add CLI wrapper for normalize ingest"
```

---

## Self-Review

**Spec coverage check:**
- Package structure: ✓ Tasks 1–8
- EAN parser (strip .0, validate length): ✓ Task 2
- Price parser (decimal sep): ✓ Task 3
- Supplier detection via filename patterns: ✓ Task 4
- YAML profile loading: ✓ Task 4
- Multi-sheet loader with header_row: ✓ Task 5
- Column mapping + brand resolution (static/filename_regex/sheet_name/column/column_regex): ✓ Tasks 6, 12
- Row validation (drop invalid, deduplicate EAN): ✓ Task 7
- ingest() orchestrator: ✓ Task 8
- NormalizeError on hard failures (<50% yield, no match): ✓ Task 8
- Phase 1 profiles (ColorWow, Hispalbeauty, Droguería): ✓ Tasks 10–12
- Phase 2 profile (Argon Trading, 3 files): ✓ Task 13
- Phase 3 profile (K-Beauty, 33 sheets, header row 2): ✓ Task 14
- CLI wrapper: ✓ Task 15
- Golden tests for all 7 files: ✓ Tasks 10–14
- Parser unit tests: ✓ Tasks 2–3
- Output is List[ProductData] → existing save_supplier_batch(): ✓ all tasks

**No TBD placeholders remaining:** profile column headers marked `<COL>` are filled in during Task 9 (inspect step) — the plan explicitly instructs inspecting headers before writing profiles.

**Type consistency:** `ingest()` returns `Tuple[List[ProductData], List[str]]` throughout. `map_sheet()` signature consistent between Task 6 and Task 12 update. `NormalizeError` defined in `normalize/exceptions.py` and imported consistently.
