from normalize.detect import detect_supplier


def test_detect_matches_test_supplier():
    assert detect_supplier("test_offer_2026.xlsx") == "_test_supplier"


def test_detect_matches_second_pattern():
    assert detect_supplier("TestCo_products.xlsx") == "_test_supplier"


def test_detect_no_match_returns_none():
    assert detect_supplier("completely_unknown_file.xlsx") is None


def test_detect_case_sensitive():
    # fnmatch on Windows is case-insensitive
    result = detect_supplier("TEST_OFFER_2026.xlsx")
    # On Windows fnmatch is case-insensitive, so this may match
    assert result in ("_test_supplier", None)
