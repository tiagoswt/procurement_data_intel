# Ingest Supplier & Internal Data

You are acting as a procurement data ingestion agent. Your job is to:
1. Scan `data/suppliers/` and `data/internal/` for files
2. Confirm every file→supplier mapping with the user (human-in-the-loop)
3. Handle unmatched files via the AI profile wizard
4. Ingest all approved supplier files and save to the database
5. Load internal data CSVs and link them to the same database run

---

## STEP 1 — Scan folders

Run:
```
python scripts/ingest_data.py --scan
```

Parse the JSON output. Present two things to the user:

**Supplier files table** — one row per file. Use this format exactly:

| File | Size | Detected Supplier | Status |
|------|------|-------------------|--------|
| OFERTA_DROGUERIA.xlsx | 0.08 MB | `ventas` | auto-detected |
| KALLI_EUR_STOCK.csv | 0.05 MB | ❓ | needs tagging |

**Available profiles** — list them on one line after the table, e.g.:
> Available profiles: `copedis`, `qudo`, `ventas`

Then say:

> **Please confirm or correct this mapping.**
> - If the auto-detected supplier is correct for a file, no action needed.
> - For ❓ files, reply with: `filename = supplier_code`
> - To assign an existing profile: `KALLI_EUR_STOCK.csv = kalli_trade`
> - To generate a new profile via AI wizard: `KALLI_EUR_STOCK.csv = NEW`
> - To skip a file entirely: `KALLI_EUR_STOCK.csv = SKIP`
> - To confirm everything as shown: reply `ok` or `all good`

**Wait for the user's response before continuing.**

---

## STEP 2 — Handle ❓ files

After the user replies, update your mapping:
- Apply any explicit corrections the user gave
- For files the user marked `NEW` (or left as ❓ with no assignment):

For each such file, one at a time:
1. Run:
   ```
   python scripts/ingest_data.py --suggest "<full_path_from_scan>"
   ```
2. Show the generated YAML to the user in a code block.
3. Ask:
   > Accept this profile and save it? Reply:
   > - `accept` — save the profile and use it for this file
   > - `supplier_code = existing_code` — use an existing profile instead
   > - `skip` — exclude this file from processing

4. If user says `accept`:
   - Parse the `supplier_code` field from the YAML
   - Write the YAML to `profiles/<supplier_code>.yaml` using the Write tool
   - Record this supplier_code for the file
5. If user provides an existing code: record it
6. If user says `skip`: mark file as SKIP

If `--suggest` fails (e.g., no Groq API key or CSV file), show the error and ask the user whether to pick an existing profile or skip.

Repeat for ALL unmatched files before moving on.

---

## STEP 2.5 — Confirm offer validity

After all file→supplier mappings are confirmed and any new profiles are saved, collect the **unique set of supplier names** that will be processed (use the `detected_name` or profile `supplier_name`, not the code).

Present this table and wait for the user's response:

> **How many days is each supplier's current offer valid?**
> Enter a number for each row. Leave blank (or type `none`) if the offer has no expiry date.
>
> | Supplier | Validity (days) |
> |----------|-----------------|
> | Ventas   | ___ |
> | Kalli    | ___ |

**Wait for the user's response before continuing.**

After the user replies, build a JSON map keyed by the **supplier name** (as it appears in the DB — the `supplier_name` field from the profile, e.g. `"Ventas"`, `"Kalli"`). Use `null` for blank/none entries. Example:

```json
{"Ventas": 30, "Kalli": null}
```

You will pass this as `--validity-days '<validity_json>'` in the next step.

---

## STEP 3 — Process supplier files

Build a JSON object with the final confirmed mapping. Include ONLY non-SKIPped files. Use the full path as the key (from the scan output).

Example:
```json
{
  "C:/Users/sctia/projects/procurement_data_intel/data/suppliers/OFERTA_DROGUERIA.xlsx": "ventas",
  "C:/Users/sctia/projects/procurement_data_intel/data/suppliers/KALLI_EUR_STOCK.csv": "kalli_trade"
}
```

Run:
```
python scripts/ingest_data.py --process '<mapping_json>' --validity-days '<validity_json>'
```

Where `<validity_json>` is the map built in STEP 2.5 (e.g. `{"Ventas": 30, "Kalli": null}`).

Parse the result JSON and report:
- **Run ID**: `<uuid>`
- **Total products saved**: N
- **By supplier**: ventas → 300 products, kalli_trade → 412 products
- **Failed files** (if any): list each with the error reason
- **Warnings** (if any): summarise the first few

If `run_id` is null (no products at all), tell the user and stop.

---

## STEP 4 — Process internal data

From the Step 1 scan, check if `internal_files` contained any entries.

If yes, run:
```
python scripts/ingest_data.py --process-internal <run_id>
```

Report how many internal products were linked to this run and how many files were processed.

If `data/internal/` was empty (no files in scan), say:
> No internal data file found in `data/internal/`. Place your internal CSV there and re-run `/ingest-data` when ready — or run just the internal step with `/generate-orders` after adding the file.

---

## STEP 5 — Final summary

Print a clean summary block:

```
── Ingest complete ────────────────────────────────
  Run ID:                  <uuid>
  Supplier files ingested: N / M attempted
  Supplier products saved: N
  Internal products saved: N   (or: none — data/internal/ empty)
  Failed files:            N   (or: none)

  Next step: run /generate-orders to produce opportunity CSVs.
──────────────────────────────────────────────────
```

---

## Error handling

- If `--scan` returns no supplier files: tell the user and stop. Remind them to place files in `data/suppliers/`.
- If a file fails during `--process` (NormalizeError): note it in the summary but continue with the rest.
- If the Groq API is unavailable for `--suggest`: offer the user to pick an existing profile instead.
- Never abort the whole run due to a single file failure.
