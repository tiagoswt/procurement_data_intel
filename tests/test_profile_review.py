# tests/test_profile_review.py
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Stub streamlit with a real session-state dict so we can test state mutations
if "streamlit" not in sys.modules:
    st_stub = MagicMock()
    sys.modules["streamlit"] = st_stub

import streamlit as st


class _SessionState(dict):
    """Dict with attribute access — mirrors Streamlit's SessionState interface."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def _reset_session(pending=None):
    st.session_state = _SessionState(
        processed_data=[],
        pending_profiles=pending if pending is not None else {},
    )


def _make_product():
    from models import ProductData
    return ProductData(
        ean_code="1234567890123",
        supplier_code=None,
        product_name="Test Product",
        quantity=1,
        price=9.99,
        supplier="Test",
        confidence_score=1.0,
        source_file="test.xlsx",
    )


def _make_entry():
    return {"file_bytes": b"fake content", "supplier_name": "Test Supplier"}


VALID_YAML = """\
supplier_code: test_supplier
supplier_name: "Test Supplier"
match:
  filename_patterns:
    - "*test*"
sheets:
  - match: "*"
    header_row: 1
    columns:
      ean: EAN
      product_name: Name
      price: Price
      quantity: null
"""

from app import _handle_profile_accept


# ── accept: valid path ────────────────────────────────────────────────────────

def test_accept_saves_profile_and_adds_products(tmp_path):
    _reset_session(pending={"test.xlsx": _make_entry()})
    product = _make_product()

    with patch("app.ingest", return_value=([product], [])), \
         patch("app.tempfile.mkdtemp", return_value=str(tmp_path)), \
         patch("app.open", mock_open()), \
         patch("app.os.unlink"), \
         patch("app.os.rmdir"):
        _handle_profile_accept("test.xlsx", VALID_YAML, _make_entry(), profiles_dir=tmp_path)

    assert (tmp_path / "test_supplier.yaml").exists()
    assert product in st.session_state.processed_data
    assert "test.xlsx" not in st.session_state.pending_profiles


# ── accept: invalid YAML ─────────────────────────────────────────────────────

def test_accept_invalid_yaml_shows_error_no_file_written(tmp_path):
    _reset_session(pending={"test.xlsx": _make_entry()})
    st.error.reset_mock()

    with patch("app.ingest") as mock_ingest:
        _handle_profile_accept("test.xlsx", "bad: yaml: :::", _make_entry(), profiles_dir=tmp_path)
        mock_ingest.assert_not_called()

    st.error.assert_called()
    assert not list(tmp_path.glob("*.yaml"))
    assert "test.xlsx" in st.session_state.pending_profiles


# ── accept: missing supplier_code ─────────────────────────────────────────────

def test_accept_missing_supplier_code_shows_error(tmp_path):
    _reset_session(pending={"test.xlsx": _make_entry()})
    st.error.reset_mock()

    yaml_no_code = "supplier_name: Test\nsheets: []\n"
    with patch("app.ingest") as mock_ingest:
        _handle_profile_accept("test.xlsx", yaml_no_code, _make_entry(), profiles_dir=tmp_path)
        mock_ingest.assert_not_called()

    st.error.assert_called()
    assert not list(tmp_path.glob("*.yaml"))
    assert "test.xlsx" in st.session_state.pending_profiles


# ── accept: ingest failure after save ─────────────────────────────────────────

def test_accept_ingest_failure_keeps_profile_no_products(tmp_path):
    from normalize import NormalizeError
    _reset_session(pending={"test.xlsx": _make_entry()})
    st.error.reset_mock()

    with patch("app.ingest", side_effect=NormalizeError("bad columns")), \
         patch("app.tempfile.mkdtemp", return_value=str(tmp_path)), \
         patch("app.open", mock_open()), \
         patch("app.os.unlink"), \
         patch("app.os.rmdir"):
        _handle_profile_accept("test.xlsx", VALID_YAML, _make_entry(), profiles_dir=tmp_path)

    assert (tmp_path / "test_supplier.yaml").exists()
    assert st.session_state.processed_data == []
    assert "test.xlsx" not in st.session_state.pending_profiles
    st.error.assert_called()
