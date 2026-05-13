import html
import io
import json
from datetime import datetime
from typing import List, Dict

import pandas as pd


def export_excel(
    opportunities: List[Dict],
    price_trends_df: pd.DataFrame,
    stockout_df: pd.DataFrame,
    supplier_df: pd.DataFrame,
    brand_df: pd.DataFrame,
) -> bytes:
    """Returns a 5-sheet Excel workbook as bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if opportunities:
            opp_df = pd.DataFrame([{
                "EAN": o.get("ean"),
                "Product": o.get("product_name"),
                "Brand": o.get("brand"),
                "Priority": o.get("priority_label"),
                "Enhanced Score": o.get("enhanced_score"),
                "Score Breakdown": o.get("score_breakdown"),
                "Net Need": o.get("net_need"),
                "Best Price (€)": o.get("quote_price"),
                "Supplier": o.get("supplier"),
                "Savings/unit (€)": o.get("savings_per_unit"),
                "Total Savings (€)": o.get("total_savings"),
                "Days Cover": round(float(o.get("days_of_cover") or 0), 1),
            } for o in opportunities])
        else:
            opp_df = pd.DataFrame()
        opp_df.to_excel(writer, sheet_name="Opportunities", index=False)
        price_trends_df.to_excel(writer, sheet_name="Price Trends", index=False)
        stockout_df.to_excel(writer, sheet_name="Stockout Risk", index=False)
        supplier_df.to_excel(writer, sheet_name="Suppliers", index=False)
        brand_df.to_excel(writer, sheet_name="Brands", index=False)
    return buf.getvalue()


def export_json_payload(
    opportunities: List[Dict],
    price_trend_alerts: List[Dict],
    stockout_risks: List[Dict],
    supplier_movements: List[Dict],
    batch_id: str,
) -> str:
    """Returns LLM-ready JSON payload as string."""
    total_savings = sum(o.get("total_savings", 0) for o in opportunities)
    critical = [s for s in stockout_risks if (s.get("days_cover") or 999) < 7]
    overstocked = [o for o in opportunities if (o.get("days_of_cover") or 0) > 180]
    top_opps = sorted(opportunities, key=lambda x: x.get("enhanced_score", 0), reverse=True)[:10]
    payload = {
        "period": datetime.utcnow().date().isoformat(),
        "batch_id": batch_id,
        "headline_metrics": {
            "opportunities_count": len(opportunities),
            "estimated_savings_eur": round(total_savings, 2),
            "skus_at_stockout_risk": len(critical),
            "skus_overstocked": len(overstocked),
            "new_supplier_prices_ingested": len(opportunities),
        },
        "top_opportunities": [
            {
                "ean": o.get("ean"),
                "product": o.get("product_name"),
                "brand": o.get("brand"),
                "recommended_supplier": o.get("supplier"),
                "recommended_qty": o.get("net_need"),
                "unit_price": o.get("quote_price"),
                "score": o.get("enhanced_score"),
                "score_breakdown": o.get("score_breakdown"),
                "days_of_cover": round(float(o.get("days_of_cover") or 0), 1),
                "priority": o.get("priority_label"),
            }
            for o in top_opps
        ],
        "price_trend_alerts": price_trend_alerts,
        "stockout_risks": [
            {
                "ean": s.get("ean"),
                "product": s.get("description"),
                "days_cover": round(float(s.get("days_cover") or 0), 1),
                "daily_sales": round(float(s.get("daily_sales") or 0), 1),
                "best_supplier": s.get("best_supplier"),
            }
            for s in stockout_risks[:20]
        ],
        "supplier_movements": supplier_movements[:10],
        "overstock_warnings": [
            {
                "ean": o.get("ean"),
                "product": o.get("product_name"),
                "days_cover": round(float(o.get("days_of_cover") or 0), 1),
            }
            for o in overstocked[:10]
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def export_html_report(
    opportunities: List[Dict],
    stockout_risks: List[Dict],
    supplier_win_rates: List[Dict],
    brand_alerts: List[Dict],
    batch_id: str,
) -> str:
    """Returns self-contained HTML report as string (no external dependencies)."""
    total_savings = sum(o.get("total_savings", 0) for o in opportunities)
    critical_count = sum(1 for s in stockout_risks if (s.get("days_cover") or 999) < 7)
    top_opps = sorted(opportunities, key=lambda x: x.get("enhanced_score", 0), reverse=True)[:10]

    svg_bars = ""
    if supplier_win_rates:
        max_rate = max(s.get("win_rate_pct", 0) for s in supplier_win_rates)
        for i, s in enumerate(supplier_win_rates[:6]):
            rate = s.get("win_rate_pct", 0)
            bar_w = int((rate / max(max_rate, 1)) * 200)
            y = i * 30 + 10
            svg_bars += f'<rect x="130" y="{y}" width="{bar_w}" height="20" fill="#4CAF50"/>'
            svg_bars += f'<text x="125" y="{y+15}" text-anchor="end" font-size="11">{html.escape(str(s.get("supplier",""))[:16])}</text>'
            svg_bars += f'<text x="{130+bar_w+5}" y="{y+15}" font-size="11">{rate:.0f}%</text>'
    svg_h = max(200, len(supplier_win_rates[:6]) * 30 + 20)
    supplier_svg = f'<svg width="400" height="{svg_h}" style="background:#f5f5f5;border-radius:6px;padding:8px">{svg_bars}</svg>'

    opp_rows = "".join(
        f"<tr><td>{html.escape(str(o.get('ean','')))}</td><td>{html.escape(str(o.get('product_name','')))}</td>"
        f"<td>{html.escape(str(o.get('priority_label','')))}</td><td><strong>{o.get('enhanced_score','-')}</strong></td>"
        f"<td>{o.get('net_need',0)}</td><td>€{o.get('quote_price',0):.2f}</td>"
        f"<td>{html.escape(str(o.get('supplier','')))}</td><td style='color:green'>€{o.get('total_savings',0):.2f}</td></tr>"
        for o in top_opps
    )
    stockout_rows = "".join(
        f"<tr><td>{html.escape(str(s.get('ean','')))}</td><td>{html.escape(str(s.get('description','')))}</td>"
        f"<td style='color:{'#ff4444' if (s.get('days_cover',999)<7) else '#ff8c00'};font-weight:bold'>{s.get('days_cover',0):.0f}d</td>"
        f"<td>{html.escape(str(s.get('best_supplier','')))}</td></tr>"
        for s in stockout_risks[:5]
    )
    brand_rows = "".join(
        f"<tr><td>{html.escape(str(b.get('brand','')))}</td>"
        f"<td style='color:{'#ff4444' if b.get('price_trend_pct',0)>1 else ('#4CAF50' if b.get('price_trend_pct',0)<-1 else '#888')}'>"
        f"{'↑' if b.get('price_trend_pct',0)>1 else ('↓' if b.get('price_trend_pct',0)<-1 else '→')} {abs(b.get('price_trend_pct',0)):.1f}%</td>"
        f"<td>{b.get('at_risk_skus',0)}</td><td>{b.get('coverage_pct',0):.0f}%</td></tr>"
        for b in brand_alerts[:5]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Procurement Intelligence Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:0 auto;padding:20px;color:#333}}
h1{{color:#1a237e}} h2{{color:#1a237e;border-bottom:2px solid #e0e0e0;padding-bottom:8px;margin-top:32px}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}}
.kpi{{background:#f5f7ff;border-radius:8px;padding:16px;text-align:center}}
.kpi .val{{font-size:28px;font-weight:bold;color:#1a237e}} .kpi .label{{font-size:12px;color:#666}}
table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:13px}}
th{{background:#1a237e;color:white;padding:8px;text-align:left}}
td{{padding:6px 8px;border-bottom:1px solid #eee}}
.footer{{color:#999;font-size:11px;margin-top:40px}}
</style></head>
<body>
<h1>Procurement Intelligence Report</h1>
<p style="color:#888">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · Batch: {html.escape(str(batch_id)[:8])}...</p>
<div class="kpis">
  <div class="kpi"><div class="val">{len(opportunities)}</div><div class="label">Opportunities</div></div>
  <div class="kpi"><div class="val" style="color:#cc0000">{critical_count}</div><div class="label">Critical Stockouts</div></div>
  <div class="kpi"><div class="val" style="color:#2e7d32">€{total_savings:,.0f}</div><div class="label">Est. Savings</div></div>
</div>
<h2>Top Opportunities</h2>
<table><thead><tr><th>EAN</th><th>Product</th><th>Priority</th><th>Score</th><th>Need</th><th>Price</th><th>Supplier</th><th>Savings</th></tr></thead>
<tbody>{opp_rows}</tbody></table>
<h2>Stockout Risk</h2>
<table><thead><tr><th>EAN</th><th>Product</th><th>Days Cover</th><th>Best Supplier</th></tr></thead>
<tbody>{stockout_rows}</tbody></table>
<h2>Supplier Win Rates</h2>{supplier_svg}
<h2>Brand Trend Alerts</h2>
<table><thead><tr><th>Brand</th><th>Price Trend</th><th>At-Risk SKUs</th><th>Coverage</th></tr></thead>
<tbody>{brand_rows}</tbody></table>
<p class="footer">All numbers sourced from processed batch data. Verify before placing orders.</p>
</body></html>"""
