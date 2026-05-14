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
