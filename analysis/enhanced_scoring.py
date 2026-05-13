from typing import List, Dict


def compute_enhanced_scores(opportunities: List[Dict]) -> List[Dict]:
    """
    Appends enhanced_score (0–10) and score_breakdown string to copies of each
    opportunity dict. Does not mutate the originals.
    """
    if not opportunities:
        return []

    max_daily = max(
        (o.get("sales90d", 0) / 90 for o in opportunities), default=1
    )
    if max_daily <= 0:
        max_daily = 1

    raw_scores = []
    for opp in opportunities:
        daily = (opp.get("sales90d", 0) or 0) / 90
        velocity = daily / max_daily

        best_buy = (opp.get("price_breakdown") or {}).get("best_buy_price") or opp.get("baseline_price", 0)
        supplier_price = opp.get("quote_price", 0) or 0
        margin = max(1.0, best_buy / supplier_price) if supplier_price > 0 and best_buy else 1.0

        days_cover = opp.get("days_of_cover", 999) or 999
        urgency = max(1.0, 30 / max(1, days_cover))

        brand_weight = opp.get("brand_weight", 1.0) or 1.0

        overstock = max(1.0, days_cover / 180) if days_cover > 180 else 1.0

        raw = velocity * margin * urgency * brand_weight / overstock
        raw_scores.append({
            "raw": raw,
            "velocity": velocity,
            "margin": margin,
            "urgency": urgency,
            "brand_weight": brand_weight,
            "overstock": overstock,
        })

    max_raw = max((s["raw"] for s in raw_scores), default=1)
    if max_raw <= 0:
        max_raw = 1

    result = []
    for opp, sd in zip(opportunities, raw_scores):
        enhanced = opp.copy()
        normalized = round((sd["raw"] / max_raw) * 10, 1)
        enhanced["enhanced_score"] = normalized

        parts = [
            f"v×{sd['velocity']:.2f}",
            f"m×{sd['margin']:.2f}",
            f"u×{sd['urgency']:.2f}",
            f"b×{sd['brand_weight']:.1f}",
        ]
        if sd["overstock"] > 1.0:
            parts.append(f"÷os×{sd['overstock']:.2f}")
        enhanced["score_breakdown"] = " · ".join(parts)
        result.append(enhanced)

    return result
