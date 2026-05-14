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
