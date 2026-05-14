import re
from typing import Optional

_VALID_LENGTHS = {8, 12, 13, 14}


def clean_ean(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    if len(digits) in _VALID_LENGTHS:
        return digits
    return None
