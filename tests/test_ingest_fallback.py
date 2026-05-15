# tests/test_ingest_fallback.py
import types, sys
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Stub streamlit so app.py can be imported without a running Streamlit server
if "streamlit" not in sys.modules:
    st_stub = MagicMock()
    st_stub.session_state = MagicMock()
    st_stub.session_state.__contains__ = lambda self, key: False
    sys.modules["streamlit"] = st_stub

from app import _ingest_with_fallback


class _FakeUploadedFile:
    def __init__(self, name: str, content: bytes = b"fake"):
        self.name = name
        self._content = content
        self._pos = 0

    def getbuffer(self):
        return self._content

    def seek(self, pos):
        self._pos = pos


def _make_product(supplier="Argon Trading"):
    from models import ProductData
    return ProductData(
        ean_code="1234567890123",
        supplier_code=None,
        product_name="Test Product",
        quantity=1,
        price=9.99,
        supplier=supplier,
        confidence_score=1.0,
        source_file="test.xlsx",
    )


# ── profile path ────────────────────────────────────────────────────────────

def test_uses_profile_when_matched(tmp_path):
    fake_file = _FakeUploadedFile("argon_trading_2024.xlsx")
    product = _make_product()
    processor = MagicMock()

    with patch("app.detect_supplier", return_value="argon_trading"), \
         patch("app.ingest", return_value=([product], ["warn1"])), \
         patch("app.tempfile.mkdtemp", return_value=str(tmp_path)), \
         patch("app.open", mock_open()), \
         patch("app.os.unlink"):
        products, warnings, mode = _ingest_with_fallback(fake_file, "Argon Trading", processor)

    assert products == [product]
    assert warnings == ["warn1"]
    assert "argon_trading" in mode
    processor.process_uploaded_file.assert_not_called()


# ── fallback path ────────────────────────────────────────────────────────────

def test_falls_back_on_no_profile(tmp_path):
    from normalize import NormalizeError
    fake_file = _FakeUploadedFile("unknown_supplier.xlsx")
    product = _make_product()

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.products = [product]
    mock_result.errors = []
    mock_result.processing_time = 0.1
    mock_result.total_products = 1
    processor = MagicMock()
    processor.process_uploaded_file.return_value = mock_result

    with patch("app.detect_supplier", return_value=None), \
         patch("app.ingest", side_effect=NormalizeError("No profile matched")), \
         patch("app.tempfile.mkdtemp", return_value=str(tmp_path)), \
         patch("app.open", mock_open()), \
         patch("app.os.unlink"):
        products, warnings, mode = _ingest_with_fallback(fake_file, "Unknown", processor)

    assert products == [product]
    assert warnings == []
    assert mode == "AI detection"
    processor.process_uploaded_file.assert_called_once()


# ── cleanup ───────────────────────────────────────────────────────────────────

def test_temp_file_cleaned_up_on_normalize_error(tmp_path):
    from normalize import NormalizeError
    fake_file = _FakeUploadedFile("unknown.xlsx")
    processor = MagicMock()
    processor.process_uploaded_file.return_value = MagicMock(
        success=True, products=[], errors=[], processing_time=0, total_products=0
    )

    unlink_calls = []
    with patch("app.detect_supplier", return_value=None), \
         patch("app.ingest", side_effect=NormalizeError("no profile")), \
         patch("app.tempfile.mkdtemp", return_value="/tmp/testdir"), \
         patch("app.open", mock_open()), \
         patch("app.os.unlink", side_effect=unlink_calls.append):
        _ingest_with_fallback(fake_file, "X", processor)

    assert any("unknown.xlsx" in p for p in unlink_calls)
