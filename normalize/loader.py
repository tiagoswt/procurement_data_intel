from fnmatch import fnmatch
from typing import List, Tuple

import pandas as pd


def load_sheets(
    file_path: str, sheets_config: List[dict]
) -> List[Tuple[str, pd.DataFrame, dict]]:
    """
    Returns (sheet_name, DataFrame, sheet_config) for every sheet
    that matches a config entry. header_row in config is 1-indexed.
    """
    xl = pd.ExcelFile(file_path, engine="openpyxl")
    results = []
    for sheet_name in xl.sheet_names:
        for sheet_cfg in sheets_config:
            pattern = sheet_cfg.get("match", "*")
            if fnmatch(sheet_name, pattern):
                header_idx = sheet_cfg.get("header_row", 1) - 1  # pandas is 0-indexed
                df = pd.read_excel(
                    xl,
                    sheet_name=sheet_name,
                    header=header_idx,
                    dtype=str,
                )
                df = df.dropna(how="all")
                results.append((sheet_name, df, sheet_cfg))
                break  # first matching config wins
    return results
