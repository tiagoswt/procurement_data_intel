from normalize.parsers.ean import clean_ean


def test_clean_ean_standard_13():
    assert clean_ean("5060280378928") == "5060280378928"


def test_clean_ean_strips_float_suffix():
    # Excel stores EANs as floats: 8871676652000.0
    assert clean_ean("8871676652000.0") == "8871676652000"


def test_clean_ean_12_digits():
    assert clean_ean("012345678905") == "012345678905"


def test_clean_ean_8_digits():
    assert clean_ean("12345678") == "12345678"


def test_clean_ean_14_digits():
    assert clean_ean("01234567890128") == "01234567890128"


def test_clean_ean_invalid_5_digits():
    assert clean_ean("12345") is None


def test_clean_ean_none():
    assert clean_ean(None) is None


def test_clean_ean_empty_string():
    assert clean_ean("") is None


def test_clean_ean_non_numeric_stripped():
    # Some suppliers wrap EAN in spaces or non-breaking chars
    assert clean_ean(" 5060280378928 ") == "5060280378928"


def test_clean_ean_float_input():
    # pandas may produce a float object
    assert clean_ean(5060280378928.0) == "5060280378928"


from normalize.parsers.price import parse_price


def test_parse_price_plain_float_string():
    assert parse_price("29.99") == 29.99


def test_parse_price_numeric_float():
    assert parse_price(29.99) == 29.99


def test_parse_price_numeric_int():
    assert parse_price(30) == 30.0


def test_parse_price_comma_decimal_sep():
    assert parse_price("25,50", decimal_sep=",") == 25.50


def test_parse_price_european_thousands():
    # "1.234,56" with decimal_sep="," → 1234.56
    assert parse_price("1.234,56", decimal_sep=",") == 1234.56


def test_parse_price_currency_symbol():
    assert parse_price("€29.99") == 29.99


def test_parse_price_zero_returns_none():
    assert parse_price("0") is None


def test_parse_price_zero_float_returns_none():
    assert parse_price(0.0) is None


def test_parse_price_none_returns_none():
    assert parse_price(None) is None


def test_parse_price_empty_string_returns_none():
    assert parse_price("") is None


def test_parse_price_negative_returns_none():
    assert parse_price("-5.00") is None
