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


def test_hispalbeauty(generate_golden):
    products, warnings = ingest(str(DATA_DIR / "STOCK HISPALBEAUTY ONTHEFLOOR  08052026.xlsx"))
    actual = [product_to_dict(p) for p in products]
    if generate_golden:
        save_golden("hispalbeauty", actual)
        return
    assert actual == load_golden("hispalbeauty")


def test_oferta_drogueria(generate_golden):
    products, warnings = ingest(str(DATA_DIR / "OFERTA_DROGUERIA_140526.xlsx"))
    actual = [product_to_dict(p) for p in products]
    if generate_golden:
        save_golden("oferta_drogueria", actual)
        return
    assert actual == load_golden("oferta_drogueria")


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
