import pytest
from analysis.enhanced_scoring import compute_enhanced_scores


@pytest.fixture
def sample_opps():
    return [
        {
            "ean": "AAA",
            "product_name": "Fast mover",
            "sales90d": 1800,   # 20/day — high velocity
            "days_of_cover": 6,  # near stockout — high urgency
            "quote_price": 8.00,
            "price_breakdown": {"best_buy_price": 9.20},
            "brand_weight": 1.2,
        },
        {
            "ean": "BBB",
            "product_name": "Slow mover overstocked",
            "sales90d": 90,    # 1/day — low velocity
            "days_of_cover": 365,  # massively overstocked
            "quote_price": 5.00,
            "price_breakdown": {"best_buy_price": 5.10},
            "brand_weight": 1.0,
        },
    ]


def test_compute_enhanced_scores_adds_keys(sample_opps):
    result = compute_enhanced_scores(sample_opps)
    for opp in result:
        assert "enhanced_score" in opp
        assert "score_breakdown" in opp


def test_scores_normalized_0_to_10(sample_opps):
    result = compute_enhanced_scores(sample_opps)
    for opp in result:
        assert 0.0 <= opp["enhanced_score"] <= 10.0


def test_fast_mover_scores_higher_than_overstocked(sample_opps):
    result = compute_enhanced_scores(sample_opps)
    scores = {o["ean"]: o["enhanced_score"] for o in result}
    assert scores["AAA"] > scores["BBB"]


def test_score_breakdown_contains_components(sample_opps):
    result = compute_enhanced_scores(sample_opps)
    breakdown = result[0]["score_breakdown"]
    assert "v×" in breakdown
    assert "m×" in breakdown
    assert "u×" in breakdown
    assert "b×" in breakdown


def test_overstock_penalty_applies_above_180_days(sample_opps):
    result = compute_enhanced_scores(sample_opps)
    breakdown_b = result[1]["score_breakdown"]
    assert "÷os×" in breakdown_b  # BBB has 365 days cover → penalty applied


def test_empty_list_returns_empty(sample_opps):
    result = compute_enhanced_scores([])
    assert result == []


def test_original_dicts_not_mutated(sample_opps):
    original_ids = [id(o) for o in sample_opps]
    result = compute_enhanced_scores(sample_opps)
    # original list elements should be untouched
    for o in sample_opps:
        assert "enhanced_score" not in o
