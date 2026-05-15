# Profile Review UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a supplier file has no matching profile, auto-generate a YAML draft via Groq, display it in an editable Streamlit card, and let the user accept (saves `profiles/<code>.yaml` + re-ingests) or reject (falls back to AI detection).

**Architecture:** Three new helpers added to `app.py`: `_handle_profile_accept`, `_handle_profile_reject`, and `_show_pending_profile_reviews`. The processing loop in `process_manual_supplier_files` is split on `detect_supplier` — files with known profiles process immediately; unknown files get a draft queued in `st.session_state.pending_profiles`. The review section renders persistently on every re-run (outside the button block) until all pending profiles are resolved.

**Tech Stack:** `normalize.wizard.suggest_profile`, `normalize.ingest`, `normalize.detect.detect_supplier`, `normalize.NormalizeError`, `yaml` (PyYAML — already a dependency), Streamlit `st.text_area` / `st.button` / `st.expander` / `st.rerun`.

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `app.py` | Modify | Add `yaml` + `suggest_profile` imports; `pending_profiles` session state init; split processing loop; add `_handle_profile_accept`, `_handle_profile_reject`, `_show_pending_profile_reviews`; wire review section into `manual_supplier_processing` |
| `tests/test_profile_review.py` | Create | 6 unit tests for `_handle_profile_accept` (4) and `_handle_profile_reject` (2) |

No changes to `normalize/`, `models.py`, or `tests/test_ingest_fallback.py`.

---

## Task 1: Failing tests for `_handle_profile_accept`

**Files:**
- Create: `tests/test_profile_review.py`

- [ ] **Step 1: Create test file**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest tests/test_profile_review.py -v
```
Expected: `ImportError: cannot import name '_handle_profile_accept' from 'app'`

---

## Task 2: Implement `_handle_profile_accept` in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add imports to `app.py`**

Read `app.py`. Find the normalize imports (around line 37–38):
```python
from normalize import NormalizeError, ingest
from normalize.detect import detect_supplier
```

Add these two lines directly after them:
```python
import yaml as _yaml
from normalize.wizard import suggest_profile
```

- [ ] **Step 2: Add `_handle_profile_accept` to `app.py`**

Read `app.py`. Find `def _ingest_with_fallback(` (around line 466). Insert this function DIRECTLY ABOVE it:

```python
def _handle_profile_accept(
    filename: str,
    yaml_text: str,
    entry: dict,
    profiles_dir: Path = None,
) -> None:
    """Validate and save an accepted YAML draft, then re-ingest the file."""
    if profiles_dir is None:
        profiles_dir = Path("profiles")

    try:
        profile = _yaml.safe_load(yaml_text)
    except _yaml.YAMLError as e:
        st.error(f"Invalid YAML — fix before accepting: {e}")
        return

    supplier_code = profile.get("supplier_code")
    if not supplier_code:
        st.error("YAML must have a `supplier_code` field")
        return

    profile_path = profiles_dir / f"{supplier_code}.yaml"
    if profile_path.exists():
        st.warning(f"Profile '{supplier_code}' already exists — accepting will overwrite it")

    profile_path.write_text(yaml_text, encoding="utf-8")

    tmp_path = None
    try:
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, filename)
        with open(tmp_path, "wb") as f:
            f.write(entry["file_bytes"])

        products, warnings = ingest(tmp_path)
        st.session_state.processed_data.extend(products)
        st.success(
            f"✅ **{filename}**: {len(products)} products — Profile: {supplier_code}"
        )
        if warnings:
            with st.expander(f"⚠️ Warnings from {filename}"):
                for w in warnings:
                    st.warning(w)

    except NormalizeError as e:
        st.error(f"Profile saved but ingest failed: {e}")

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
                os.rmdir(os.path.dirname(tmp_path))
            except OSError:
                pass

    del st.session_state.pending_profiles[filename]
```

- [ ] **Step 3: Run tests**

```
python -m pytest tests/test_profile_review.py -v
```
Expected: all 4 PASS.

- [ ] **Step 4: Commit**

```bash
git add app.py tests/test_profile_review.py
git commit -m "feat: add _handle_profile_accept — validate, save, and re-ingest accepted profile"
```

---

## Task 3: Failing tests for `_handle_profile_reject`

**Files:**
- Modify: `tests/test_profile_review.py`

- [ ] **Step 1: Add import and reject tests to `tests/test_profile_review.py`**

Read `tests/test_profile_review.py`. Add this import line directly after `from app import _handle_profile_accept`:

```python
from app import _handle_profile_reject
```

Then append these two tests at the end of the file:

```python
# ── reject: success path ──────────────────────────────────────────────────────

def test_reject_processes_with_ai_detection_and_removes_pending():
    _reset_session(pending={"test.xlsx": _make_entry()})
    product = _make_product()

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.products = [product]
    mock_result.errors = []
    processor = MagicMock()
    processor.process_uploaded_file.return_value = mock_result

    _handle_profile_reject("test.xlsx", _make_entry(), processor)

    processor.process_uploaded_file.assert_called_once()
    call_kwargs = processor.process_uploaded_file.call_args.kwargs
    assert call_kwargs.get("supplier_name") == "Test Supplier"
    assert call_kwargs.get("manual_mapping") is None
    assert product in st.session_state.processed_data
    assert "test.xlsx" not in st.session_state.pending_profiles


# ── reject: processor failure ─────────────────────────────────────────────────

def test_reject_processor_failure_shows_error_and_removes_pending():
    _reset_session(pending={"test.xlsx": _make_entry()})
    st.error.reset_mock()

    mock_result = MagicMock()
    mock_result.success = False
    mock_result.products = []
    mock_result.errors = ["column not found"]
    processor = MagicMock()
    processor.process_uploaded_file.return_value = mock_result

    _handle_profile_reject("test.xlsx", _make_entry(), processor)

    st.error.assert_called()
    assert st.session_state.processed_data == []
    assert "test.xlsx" not in st.session_state.pending_profiles
```

- [ ] **Step 2: Run new tests to confirm they fail**

```
python -m pytest tests/test_profile_review.py::test_reject_processes_with_ai_detection_and_removes_pending tests/test_profile_review.py::test_reject_processor_failure_shows_error_and_removes_pending -v
```
Expected: `ImportError: cannot import name '_handle_profile_reject' from 'app'`

---

## Task 4: Implement `_handle_profile_reject` in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add `_handle_profile_reject` to `app.py`**

Read `app.py`. Find the end of `_handle_profile_accept` (the `del st.session_state.pending_profiles[filename]` line, around line 500). Insert this function DIRECTLY AFTER it (before `def _ingest_with_fallback(`):

```python
def _handle_profile_reject(filename: str, entry: dict, processor) -> None:
    """Process a pending file with AI detection instead of the generated profile."""

    class _FileObj:
        name = filename

        def getbuffer(self):
            return entry["file_bytes"]

        def seek(self, pos):
            pass

    result = processor.process_uploaded_file(
        _FileObj(),
        supplier_name=entry["supplier_name"],
        manual_mapping=None,
    )
    if result.success:
        st.session_state.processed_data.extend(result.products)
        st.success(
            f"✅ **{filename}**: {len(result.products)} products — AI detection"
        )
    else:
        st.error(f"❌ **{filename}**: Processing failed")
        for msg in result.errors[:3]:
            st.error(f"  • {msg}")

    del st.session_state.pending_profiles[filename]
```

- [ ] **Step 2: Run all 6 tests**

```
python -m pytest tests/test_profile_review.py -v
```
Expected: all 6 PASS.

- [ ] **Step 3: Run full test suite to confirm no regressions**

```
python -m pytest tests/test_ingest_fallback.py tests/test_profile_review.py -v
```
Expected: all 9 PASS.

- [ ] **Step 4: Commit**

```bash
git add app.py tests/test_profile_review.py
git commit -m "feat: add _handle_profile_reject — AI detection fallback for unaccepted profiles"
```

---

## Task 5: Modify processing loop and session state init

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add `pending_profiles` to session state init**

Read `app.py`. Find the session state initialization block (around line 67–81, where `if "processed_data" not in st.session_state:` lives). Add this entry at the end of the block:

```python
if "pending_profiles" not in st.session_state:
    st.session_state.pending_profiles = {}
```

- [ ] **Step 2: Modify the processing loop**

Read `app.py`. Find this block inside `process_manual_supplier_files()`, inside the `for i, uploaded_file in enumerate(uploaded_files):` loop (after the supplier name determination, around line 677):

```python
            # Process the file — try profile-based first, fall back to AI detection
            with st.spinner(f"Processing {uploaded_file.name}..."):
                t0 = time.time()
                products, warnings, processing_mode = _ingest_with_fallback(
                    uploaded_file, supplier_name, processor
                )
                elapsed = time.time() - t0
```

Replace it with:

```python
            # Files with a known profile process immediately.
            # Unknown files get a Groq-generated draft queued for review.
            profile_code = detect_supplier(uploaded_file.name)
            if profile_code is None:
                gen_tmp = None
                try:
                    with st.spinner(f"Generating profile for {uploaded_file.name}..."):
                        gen_dir = tempfile.mkdtemp()
                        gen_tmp = os.path.join(gen_dir, uploaded_file.name)
                        with open(gen_tmp, "wb") as _f:
                            _f.write(uploaded_file.getbuffer())
                        yaml_text = suggest_profile(gen_tmp)
                    st.session_state.pending_profiles[uploaded_file.name] = {
                        "yaml_text": yaml_text,
                        "file_bytes": bytes(uploaded_file.getbuffer()),
                        "supplier_name": supplier_name,
                    }
                    st.info(
                        f"📝 **{uploaded_file.name}**: No profile found — "
                        f"review the generated profile below"
                    )
                    continue
                except Exception:
                    st.warning(
                        f"⚠️ **{uploaded_file.name}**: Could not generate profile "
                        f"— falling back to AI detection"
                    )
                finally:
                    if gen_tmp:
                        try:
                            os.unlink(gen_tmp)
                            os.rmdir(os.path.dirname(gen_tmp))
                        except OSError:
                            pass

            with st.spinner(f"Processing {uploaded_file.name}..."):
                t0 = time.time()
                products, warnings, processing_mode = _ingest_with_fallback(
                    uploaded_file, supplier_name, processor
                )
                elapsed = time.time() - t0
```

> **Note on control flow:** The `continue` inside the `try` block skips the rest of the loop iteration (the results display block). The `finally` always runs first to clean up the temp file. When `suggest_profile` raises, execution falls through to the `with st.spinner` block below for AI detection.

- [ ] **Step 3: Run full test suite**

```
python -m pytest tests/test_ingest_fallback.py tests/test_profile_review.py -v
```
Expected: all 9 PASS.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: split processing loop — known profiles immediate, unknowns queued for review"
```

---

## Task 6: Add `_show_pending_profile_reviews` and wire into UI

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add `_show_pending_profile_reviews` to `app.py`**

Read `app.py`. Find `def manual_supplier_processing(` (around line 505). Insert this function DIRECTLY ABOVE it:

```python
def _show_pending_profile_reviews(groq_api_key: str) -> None:
    """Render editable review cards for all pending profile drafts."""
    if not st.session_state.get("pending_profiles"):
        return

    st.divider()
    st.subheader("🔍 New Supplier Profiles — Review & Accept")
    st.caption(
        "The AI generated these profiles from your files. "
        "Edit if needed, then accept to save and process."
    )

    processor = ProcurementProcessor(groq_api_key)

    for filename, entry in list(st.session_state.pending_profiles.items()):
        with st.expander(f"📄 {filename}", expanded=True):
            edited_yaml = st.text_area(
                "Profile YAML (editable)",
                value=entry["yaml_text"],
                height=300,
                key=f"profile_yaml_{filename}",
            )
            col_accept, col_reject = st.columns(2)

            with col_accept:
                if st.button("✅ Accept & Process", key=f"accept_{filename}"):
                    _handle_profile_accept(filename, edited_yaml, entry)
                    st.rerun()

            with col_reject:
                if st.button("❌ Use AI detection", key=f"reject_{filename}"):
                    _handle_profile_reject(filename, entry, processor)
                    st.rerun()
```

- [ ] **Step 2: Wire the review section into `manual_supplier_processing`**

Read `app.py`. Find this block at the end of `manual_supplier_processing()`:

```python
    # Process files button
    if st.button("🚀 Process Supplier Files", type="primary", key="manual_process_btn"):
        process_manual_supplier_files(
            uploaded_files,
            groq_api_key,
            auto_detect_fields,
            max_file_size,
            manual_supplier_name,
        )
```

Replace it with:

```python
    # Process files button
    if st.button("🚀 Process Supplier Files", type="primary", key="manual_process_btn"):
        process_manual_supplier_files(
            uploaded_files,
            groq_api_key,
            auto_detect_fields,
            max_file_size,
            manual_supplier_name,
        )

    # Always rendered — review cards persist across re-runs until resolved
    _show_pending_profile_reviews(groq_api_key)
```

- [ ] **Step 3: Run full test suite**

```
python -m pytest tests/test_ingest_fallback.py tests/test_profile_review.py -v
```
Expected: all 9 PASS.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add profile review UI — editable YAML cards with accept/reject"
```

---

## Verification Checklist

- [ ] `pytest tests/test_profile_review.py -v` — all 6 tests pass
- [ ] `pytest tests/test_ingest_fallback.py tests/test_profile_review.py -v` — all 9 pass
- [ ] Upload a file matching a known profile → processes immediately, no review card shown
- [ ] Upload an unknown supplier file → spinner shows "Generating profile...", then `📝 No profile found` info message, then review card appears below
- [ ] Review card shows editable YAML with `supplier_code` and column mappings
- [ ] Edit YAML, click **Accept & Process** → profile saved to `profiles/<code>.yaml`, file re-ingested, products in session state, card disappears
- [ ] Click **Use AI detection** → file processed with AI detection, card disappears
- [ ] Accept with invalid YAML syntax → `st.error` shown, card stays open
- [ ] Accept with YAML missing `supplier_code` → `st.error` shown, card stays open
- [ ] Groq unavailable during generation → `st.warning` shown, file falls through to AI detection immediately (no card shown)
