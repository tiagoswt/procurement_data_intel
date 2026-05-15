# Profile Review UI — Design Spec

**Date:** 2026-05-15  
**Status:** Approved  
**Goal:** When a supplier file has no profile, auto-generate a YAML draft via Groq, show it in an editable review card, and let the user accept (saves profile + re-ingests) or reject (falls back to AI detection).

---

## Problem

Currently, files with no matching profile in `profiles/` fall back to `ProcurementProcessor` (Groq AI field detection). This means every new supplier is re-detected from scratch on every upload, and no reusable mapping is built up over time. The normalize pipeline is faster and more accurate, but only works for known suppliers.

---

## Solution: Approach A — Eager processing + persistent review section

Files with known profiles process immediately. Files without profiles get a Groq-generated YAML draft queued in `st.session_state.pending_profiles`. A review section renders persistently below the Process button, showing one editable card per pending file until the user resolves it.

---

## Data Flow

```
manual_supplier_processing()
    │
    ├── [Process button]
    │       └── process_manual_supplier_files()
    │               │
    │               ├── detect_supplier(f.name) → match?
    │               │       YES → _ingest_with_fallback() → show result now
    │               │       NO  → suggest_profile(tmp_path) → queue draft
    │               │                                       → st.info placeholder
    │               └── (Groq error during suggest_profile → AI detection fallback)
    │
    └── [always rendered]
            └── _show_pending_profile_reviews()
                    ├── st.text_area per pending file (editable YAML)
                    ├── Accept → validate → save profiles/<code>.yaml
                    │                    → re-ingest → append to processed_data
                    └── Reject → AI detection → append to processed_data
                    (both paths remove entry from pending_profiles)
```

---

## Session State

```python
# Initialised at app startup alongside existing session state keys
st.session_state.pending_profiles = {}

# Shape of each entry (key = uploaded_file.name)
{
    "unknown_supplier.xlsx": {
        "yaml_text": str,       # LLM-generated YAML, editable by user
        "file_bytes": bytes,    # raw upload bytes for re-ingest after accept
        "supplier_name": str,   # used as fallback label if rejected
    }
}
```

---

## Components

### `process_manual_supplier_files()` — modified

The `for` loop splits on profile presence before calling `_ingest_with_fallback`:

```python
profile_code = detect_supplier(uploaded_file.name)
if profile_code:
    # existing path — unchanged
    products, warnings, mode = _ingest_with_fallback(uploaded_file, supplier_name, processor)
    # show result immediately
else:
    # write temp file, call suggest_profile, queue draft
    with st.spinner(f"Generating profile for {uploaded_file.name}..."):
        try:
            yaml_text = suggest_profile(tmp_path)
            st.session_state.pending_profiles[uploaded_file.name] = {
                "yaml_text": yaml_text,
                "file_bytes": bytes(uploaded_file.getbuffer()),
                "supplier_name": supplier_name,
            }
            st.info(f"📝 **{uploaded_file.name}**: No profile found — review the generated profile below")
        except Exception:
            # Groq unavailable or unreadable file — fall back immediately
            products, warnings, mode = _ingest_with_fallback(uploaded_file, supplier_name, processor)
            st.warning(f"⚠️ Could not generate profile for {uploaded_file.name} — processed with AI detection")
            # show result
```

### `_show_pending_profile_reviews()` — new

Called from `manual_supplier_processing()` outside the button block (renders on every re-run):

```python
def _show_pending_profile_reviews(groq_api_key):
    if not st.session_state.get("pending_profiles"):
        return

    st.divider()
    st.subheader("🔍 New Supplier Profiles — Review & Accept")
    st.caption("The AI generated these profiles from your files. Edit if needed, then accept to save and process.")

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

### `_handle_profile_accept(filename, yaml_text, entry)` — new

```
1. yaml.safe_load(yaml_text) — if YAMLError: st.error, return (card stays open)
2. Check profile["supplier_code"] exists — if missing: st.error, return
3. If profiles/<code>.yaml already exists: st.warning (overwrite notice) but continue
4. Write yaml_text to profiles/<supplier_code>.yaml
5. Write entry["file_bytes"] to mkdtemp()/<filename>
6. Call ingest(tmp_path)
   - Success: append products to st.session_state.processed_data, show st.success
   - NormalizeError: st.error("Profile saved but ingest failed: <reason>")
     profile file is kept; file not added to processed_data
7. Cleanup temp file + dir
8. del st.session_state.pending_profiles[filename]
```

### `_handle_profile_reject(filename, entry, processor)` — new

```
1. Create a minimal file-like object with:
     .name = filename
     .getbuffer() → entry["file_bytes"]
     .seek(pos) → no-op
   (same interface as _FakeUploadedFile in tests/test_ingest_fallback.py)
2. Call processor.process_uploaded_file(file_obj, supplier_name=entry["supplier_name"], manual_mapping=None)
3. If result.success: append result.products to st.session_state.processed_data, show st.success
4. Else: show st.error
5. del st.session_state.pending_profiles[filename]
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Groq unavailable during `suggest_profile` | Catch all exceptions → fall back to AI detection, show `st.warning` |
| LLM returns unparseable YAML | Catch `yaml.YAMLError` at accept time → `st.error`, card stays open |
| Missing `supplier_code` in YAML | `st.error`, card stays open, no file written |
| `supplier_code` would overwrite existing profile | `st.warning` shown, user can still accept |
| Ingest fails after accepting valid profile | `st.error`, profile file kept on disk, file not added to `processed_data` |
| Process clicked again while reviews pending | New entries merged into `pending_profiles`, existing entries preserved |

---

## Files Changed

| File | Change |
|---|---|
| `app.py` | Add `pending_profiles` session state init; modify processing loop; add `_show_pending_profile_reviews`, `_handle_profile_accept`, `_handle_profile_reject` |
| `tests/test_profile_review.py` | New — unit tests for accept/reject logic |

No changes to `normalize/`, `models.py`, or existing tests.

---

## Tests

`tests/test_profile_review.py` covers (all I/O mocked, no real Groq calls):

1. `suggest_profile` raises → processing loop falls back to AI detection, no pending entry created
2. Accept: valid YAML with `supplier_code` → file written to `profiles/`, ingest called, products in `processed_data`
3. Accept: invalid YAML syntax → `st.error` called, no file written, entry still in `pending_profiles`
4. Accept: YAML missing `supplier_code` → `st.error` called, no file written
5. Accept: ingest fails after save → profile file exists, nothing in `processed_data`
6. Reject: `process_uploaded_file` called with stored bytes, products in `processed_data`, entry removed
