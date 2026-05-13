import json
import pytest
import pandas as pd
from db.exporters import export_excel, export_json_payload, export_html_report


@pytest.fixture
def sample_opportunities():
    return [
        {
            "ean": "3433422404397", "product_name": "Serum 30ml", "brand": "L'Oréal",
            "priority_label": "🔥 Priority 1", "enhanced_score": 8.4,
            "score_breakdown": "v×1.80 · m×1.15 · u×2.10 · b×1.2",
            "net_need": 240, "quote_price": 8.20, "supplier": "BeautyDist",
            "savings_per_unit": 1.00, "total_savings": 240.0, "days_of_cover": 6.0,
        },
    ]


@pytest.fixture
def empty_df():
    return pd.DataFrame()


def test_export_excel_returns_bytes(sample_opportunities, empty_df):
    result = export_excel(sample_opportunities, empty_df, empty_df, empty_df, empty_df)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_export_excel_has_five_sheets(sample_opportunities, empty_df):
    import io
    import openpyxl
    data = export_excel(sample_opportunities, empty_df, empty_df, empty_df, empty_df)
    wb = openpyxl.load_workbook(io.BytesIO(data))
    assert set(wb.sheetnames) == {"Opportunities", "Price Trends", "Stockout Risk", "Suppliers", "Brands"}


def test_export_json_payload_has_required_keys(sample_opportunities):
    result = export_json_payload(sample_opportunities, [], [], [], "test-batch-id")
    payload = json.loads(result)
    required = {"period", "batch_id", "headline_metrics", "top_opportunities",
                "price_trend_alerts", "stockout_risks", "supplier_movements", "overstock_warnings"}
    assert required.issubset(payload.keys())


def test_export_json_payload_headline_metrics(sample_opportunities):
    result = export_json_payload(sample_opportunities, [], [], [], "test-batch-id")
    payload = json.loads(result)
    assert payload["headline_metrics"]["opportunities_count"] == 1
    assert payload["headline_metrics"]["estimated_savings_eur"] == 240.0


def test_export_html_report_is_string(sample_opportunities):
    result = export_html_report(sample_opportunities, [], [], [], "test-batch-id")
    assert isinstance(result, str)
    assert "<!DOCTYPE html>" in result


def test_export_html_report_contains_kpi_values(sample_opportunities):
    result = export_html_report(sample_opportunities, [], [], [], "test-batch-id")
    assert "240" in result   # total savings
    assert "BeautyDist" in result
