import openpyxl
import tempfile
import os
import pytest
from normalize.wizard import extract_sheets_info


def _make_excel(sheets: dict) -> str:
    """Create a temp Excel with given {sheet_name: [header_row, *data_rows]} structure."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    wb.save(path)
    return path


def test_extract_single_sheet_header_row_1():
    path = _make_excel({
        "Sheet1": [
            ["EAN", "Description", "Price"],
            ["1234567890123", "Product A", "9.99"],
            ["9780000000002", "Product B", "4.50"],
        ]
    })
    try:
        info = extract_sheets_info(path)
        assert len(info) == 1
        assert info[0]["sheet"] == "Sheet1"
        assert info[0]["header_row"] == 1
        assert "EAN" in info[0]["columns"]
        assert len(info[0]["samples"]) == 2
    finally:
        os.unlink(path)


def test_extract_detects_header_row_2():
    path = _make_excel({
        "Brands": [
            ["Supplier: ACME Corp", None, None],     # row 1 — metadata, not headers
            ["EAN", "Product Name", "Wholesale"],     # row 2 — real headers
            ["1234567890123", "Widget", "5.00"],
        ]
    })
    try:
        info = extract_sheets_info(path)
        assert info[0]["header_row"] == 2
        assert "EAN" in info[0]["columns"]
    finally:
        os.unlink(path)


def test_extract_multiple_sheets_capped_at_five():
    sheets = {f"Brand{i}": [["EAN", "Name", "Price"], ["123456789012" + str(i), f"P{i}", "1.0"]] for i in range(7)}
    path = _make_excel(sheets)
    try:
        info = extract_sheets_info(path)
        assert len(info) <= 5
    finally:
        os.unlink(path)


def test_extract_skips_unnamed_columns():
    path = _make_excel({
        "Data": [
            ["EAN", "Name", None, "Price"],
            ["1234567890123", "Product", None, "9.99"],
        ]
    })
    try:
        info = extract_sheets_info(path)
        # Unnamed columns (None headers after pandas reads) should be filtered
        assert "EAN" in info[0]["columns"]
        assert "Price" in info[0]["columns"]
        # None header becomes "Unnamed: 2" in pandas — should NOT be in columns
        assert not any(str(c).startswith("Unnamed") for c in info[0]["columns"])
    finally:
        os.unlink(path)
