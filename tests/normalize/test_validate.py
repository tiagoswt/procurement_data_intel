from normalize.validate import validate_rows


def _row(price=10.0, name="Product A", ean="1234567890123", sheet="S1"):
    return {
        "price": price,
        "product_name": name,
        "ean": ean,
        "supplier": "TestCo",
        "source_row": 0,
        "source_sheet": sheet,
        "source_file": "test.xlsx",
        "quantity": None,
    }


def test_no_summary_when_all_valid():
    rows = [_row(), _row(ean="9780000000002")]
    _, warnings = validate_rows(rows)
    assert not any("[SKIP SUMMARY]" in w for w in warnings)


def test_summary_counts_no_price():
    rows = [_row(price=None), _row(price=0.0), _row()]
    _, warnings = validate_rows(rows)
    summary = next(w for w in warnings if "[SKIP SUMMARY]" in w)
    assert "2 no_price" in summary


def test_summary_counts_no_name():
    rows = [_row(name=None), _row(name=""), _row()]
    _, warnings = validate_rows(rows)
    summary = next(w for w in warnings if "[SKIP SUMMARY]" in w)
    assert "2 no_name" in summary


def test_summary_counts_duplicate_ean():
    rows = [_row(), _row()]  # same EAN "1234567890123" twice
    _, warnings = validate_rows(rows)
    summary = next(w for w in warnings if "[SKIP SUMMARY]" in w)
    assert "1 duplicate_ean" in summary


def test_summary_shows_only_nonzero_counts():
    rows = [_row(price=None), _row()]
    _, warnings = validate_rows(rows)
    summary = next(w for w in warnings if "[SKIP SUMMARY]" in w)
    assert "no_name" not in summary
    assert "duplicate_ean" not in summary


def test_summary_total_count():
    rows = [_row(price=None), _row(name=None), _row()]
    _, warnings = validate_rows(rows)
    summary = next(w for w in warnings if "[SKIP SUMMARY]" in w)
    assert "2 skipped" in summary
