from fnmatch import fnmatch
from typing import Optional

from .profile import list_profiles, load_profile


def detect_supplier(filename: str) -> Optional[str]:
    for code in list_profiles():
        profile = load_profile(code)
        patterns = profile.get("match", {}).get("filename_patterns", [])
        for pattern in patterns:
            if fnmatch(filename, pattern):
                return code
    return None
