from pathlib import Path
from typing import List

import yaml

PROFILES_DIR = Path(__file__).parent.parent / "profiles"


def list_profiles() -> List[str]:
    return [p.stem for p in PROFILES_DIR.glob("*.yaml")]


def load_profile(supplier_code: str) -> dict:
    path = PROFILES_DIR / f"{supplier_code}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
