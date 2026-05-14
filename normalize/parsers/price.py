import re
from typing import Optional


def parse_price(raw, decimal_sep: str = ".") -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if val > 0 else None
    s = str(raw).strip()
    # Check for negative sign before stripping it
    is_negative = "-" in s
    s = re.sub(r"[^\d.,]", "", s)
    if not s:
        return None
    if decimal_sep == ",":
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        val = float(s)
        if is_negative:
            val = -val
        return val if val > 0 else None
    except ValueError:
        return None
