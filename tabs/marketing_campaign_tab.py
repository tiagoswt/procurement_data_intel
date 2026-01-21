"""
Marketing Campaign Eligibility Tab
Identifies products eligible for marketing campaigns based on price discount thresholds.
Supports configurable eligibility criteria and campaign presets.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from utils import pad_ean_code

logger = logging.getLogger(__name__)

# Constants
CAMPAIGN_PRESETS_FILE = Path("data/campaign_presets.json")
DEFAULT_DISCOUNT_THRESHOLD = 15.0


# =============================================================================
# ELIGIBILITY CALCULATION FUNCTIONS
# =============================================================================


def calculate_baseline_price(opportunity: Dict) -> Optional[float]:
    """
    Calculate baseline price as minimum of all 3 internal prices.

    Args:
        opportunity: Opportunity dictionary with price_breakdown

    Returns:
        Minimum of best_buy_price, supplier_price, stock_avg_price or None
    """
    price_breakdown = opportunity.get("price_breakdown", {})

    prices = []
    for price_key in ["best_buy_price", "supplier_price", "stock_avg_price"]:
        price = price_breakdown.get(price_key)
        if price is not None and price > 0:
            prices.append(price)

    return min(prices) if prices else None


def calculate_discount_percentage(quote_price: float, baseline_price: float) -> float:
    """
    Calculate discount percentage.

    Formula: ((baseline - quote) / baseline) * 100

    Args:
        quote_price: The supplier's quoted price
        baseline_price: The baseline price for comparison

    Returns:
        Discount percentage (positive = savings, negative = more expensive)
    """
    if baseline_price <= 0:
        return 0.0
    return ((baseline_price - quote_price) / baseline_price) * 100


def check_eligibility(
    opportunity: Dict, discount_threshold: float
) -> Tuple[bool, float, Optional[float]]:
    """
    Check if an opportunity meets campaign eligibility criteria.

    Args:
        opportunity: Opportunity dictionary
        discount_threshold: Minimum discount % required

    Returns:
        Tuple of (is_eligible, discount_percentage, baseline_price)
    """
    quote_price = opportunity.get("quote_price", 0)
    baseline_price = calculate_baseline_price(opportunity)

    if baseline_price is None or quote_price <= 0:
        return False, 0.0, None

    discount_pct = calculate_discount_percentage(quote_price, baseline_price)
    is_eligible = discount_pct >= discount_threshold

    return is_eligible, discount_pct, baseline_price


def find_eligible_products(
    opportunities: List[Dict], discount_threshold: float
) -> List[Dict]:
    """
    Filter opportunities to find campaign-eligible products.

    Args:
        opportunities: List of opportunity dictionaries
        discount_threshold: Minimum discount % required

    Returns:
        List of eligible products enriched with campaign data
    """
    eligible = []

    for opp in opportunities:
        is_elig, discount_pct, baseline_price = check_eligibility(opp, discount_threshold)

        if is_elig and baseline_price is not None:
            # Create enriched copy with campaign-specific fields
            enriched = opp.copy()
            enriched["campaign_baseline_price"] = baseline_price
            enriched["campaign_discount_pct"] = discount_pct
            enriched["campaign_savings_per_unit"] = baseline_price - opp.get("quote_price", 0)
            eligible.append(enriched)

    return eligible


# =============================================================================
# PRESET MANAGEMENT FUNCTIONS
# =============================================================================


def load_campaign_presets() -> Dict[str, Dict]:
    """Load saved campaign presets from JSON file."""
    try:
        if CAMPAIGN_PRESETS_FILE.exists():
            with open(CAMPAIGN_PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("presets", {})
    except Exception as e:
        logger.error(f"Error loading campaign presets: {e}")
    return {}


def save_campaign_preset(name: str, config: Dict) -> bool:
    """
    Save a campaign preset to JSON file.

    Args:
        name: Preset name
        config: Configuration dictionary

    Returns:
        True if saved successfully
    """
    try:
        # Ensure data directory exists
        CAMPAIGN_PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data or create new
        data = {"presets": {}, "metadata": {"version": "1.0"}}
        if CAMPAIGN_PRESETS_FILE.exists():
            with open(CAMPAIGN_PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Add timestamp
        timestamp = datetime.now().isoformat()
        config["updated_at"] = timestamp
        if name not in data.get("presets", {}):
            config["created_at"] = timestamp

        # Save preset
        if "presets" not in data:
            data["presets"] = {}
        data["presets"][name] = config
        data["metadata"] = {"version": "1.0", "last_modified": timestamp}

        with open(CAMPAIGN_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        logger.error(f"Error saving campaign preset: {e}")
        return False


def delete_campaign_preset(name: str) -> bool:
    """
    Delete a campaign preset from JSON file.

    Args:
        name: Preset name to delete

    Returns:
        True if deleted successfully
    """
    try:
        if not CAMPAIGN_PRESETS_FILE.exists():
            return False

        with open(CAMPAIGN_PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if name in data.get("presets", {}):
            del data["presets"][name]
            data["metadata"]["last_modified"] = datetime.now().isoformat()

            with open(CAMPAIGN_PRESETS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
    except Exception as e:
        logger.error(f"Error deleting campaign preset: {e}")
    return False


# =============================================================================
# UI COMPONENTS
# =============================================================================


def show_campaign_configuration() -> Dict:
    """
    Display campaign configuration UI.

    Returns:
        Current configuration dictionary
    """
    with st.expander("Campaign Configuration", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            discount_threshold = st.slider(
                "Minimum Discount Threshold (%)",
                min_value=0.0,
                max_value=50.0,
                value=st.session_state.get("campaign_discount_threshold", DEFAULT_DISCOUNT_THRESHOLD),
                step=1.0,
                help="Products with discount >= this threshold are eligible for marketing campaigns"
            )
            st.session_state.campaign_discount_threshold = discount_threshold

        with col2:
            st.metric(
                "Current Threshold",
                f"{discount_threshold:.0f}%",
                help="Products must have at least this discount to be eligible"
            )

    return {"discount_threshold": discount_threshold}


def show_preset_management(current_config: Dict) -> None:
    """Display preset save/load/delete UI."""
    presets = load_campaign_presets()

    with st.expander("Campaign Presets", expanded=False):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            preset_name = st.text_input(
                "Preset Name",
                placeholder="e.g., Summer Sale 2025",
                key="campaign_preset_name_input"
            )

        with col2:
            preset_options = ["-- Select Preset --"] + list(presets.keys())
            selected_preset = st.selectbox(
                "Load Preset",
                options=preset_options,
                key="campaign_preset_selector"
            )

        with col3:
            st.write("")  # Spacer for alignment
            if st.button("Load", key="load_preset_btn", use_container_width=True):
                if selected_preset and selected_preset != "-- Select Preset --":
                    loaded_config = presets[selected_preset]
                    st.session_state.campaign_discount_threshold = loaded_config.get(
                        "discount_threshold", DEFAULT_DISCOUNT_THRESHOLD
                    )
                    st.success(f"Loaded: {selected_preset}")
                    st.rerun()

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save Current as Preset", key="save_preset_btn", use_container_width=True):
                if preset_name and preset_name.strip():
                    if save_campaign_preset(preset_name.strip(), current_config):
                        st.success(f"Saved: {preset_name}")
                        st.rerun()
                    else:
                        st.error("Failed to save preset")
                else:
                    st.warning("Enter a preset name")

        with col2:
            if st.button("Delete Selected Preset", key="delete_preset_btn", use_container_width=True):
                if selected_preset and selected_preset != "-- Select Preset --":
                    if delete_campaign_preset(selected_preset):
                        st.success(f"Deleted: {selected_preset}")
                        st.rerun()
                    else:
                        st.error("Failed to delete preset")


def show_eligibility_summary(eligible_products: List[Dict]) -> None:
    """Display summary metrics for eligible products."""
    st.subheader("Campaign Eligibility Summary")

    if not eligible_products:
        st.info("No products meet the current eligibility criteria.")
        return

    # Calculate metrics
    total_products = len(eligible_products)
    total_savings = sum(p.get("campaign_savings_per_unit", 0) for p in eligible_products)
    avg_discount = sum(p.get("campaign_discount_pct", 0) for p in eligible_products) / total_products
    total_value = sum(p.get("quote_price", 0) for p in eligible_products)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Eligible Products",
            f"{total_products:,}",
            help="Number of products meeting the discount threshold"
        )

    with col2:
        st.metric(
            "Total Savings/Unit",
            f"{total_savings:,.2f}",
            help="Sum of savings per unit across all eligible products"
        )

    with col3:
        st.metric(
            "Avg Discount",
            f"{avg_discount:.1f}%",
            help="Average discount percentage across eligible products"
        )

    with col4:
        st.metric(
            "Total Quote Value",
            f"{total_value:,.2f}",
            help="Sum of quote prices for all eligible products"
        )


def show_brand_scatter_plot(eligible_products: List[Dict]) -> None:
    """Display scatter plot of brands: Avg Discount vs Number of Eligible SKUs."""
    if not eligible_products:
        return

    # Aggregate by brand
    brand_data = {}
    for product in eligible_products:
        brand = product.get('brand') or 'Unknown'
        if brand not in brand_data:
            brand_data[brand] = {
                'skus': set(),
                'discounts': [],
                'total_savings': 0.0
            }
        brand_data[brand]['skus'].add(product.get('ean'))
        brand_data[brand]['discounts'].append(product.get('campaign_discount_pct', 0))
        brand_data[brand]['total_savings'] += product.get('campaign_savings_per_unit', 0)

    # Build DataFrame for plotting
    plot_data = []
    for brand, data in brand_data.items():
        avg_discount = sum(data['discounts']) / len(data['discounts']) if data['discounts'] else 0
        plot_data.append({
            'Brand': brand,
            'Eligible SKUs': len(data['skus']),
            'Avg Discount (%)': round(avg_discount, 1),
            'Total Savings': round(data['total_savings'], 2)
        })

    df_brands = pd.DataFrame(plot_data)

    # Create scatter plot
    fig = px.scatter(
        df_brands,
        x='Avg Discount (%)',
        y='Eligible SKUs',
        color='Brand',
        hover_name='Brand',
        hover_data={'Total Savings': ':.2f', 'Avg Discount (%)': ':.1f', 'Eligible SKUs': True},
        title='Brand Overview: Average Discount vs Eligible SKUs'
    )

    fig.update_layout(
        xaxis_title='Average Discount (%)',
        yaxis_title='Number of Eligible SKUs',
        height=450
    )

    fig.update_traces(marker=dict(size=10))

    st.plotly_chart(fig, use_container_width=True)


def show_eligibility_filters(eligible_products: List[Dict]) -> List[Dict]:
    """
    Display filter controls and return filtered list.

    Args:
        eligible_products: List of eligible products

    Returns:
        Filtered list of products
    """
    if not eligible_products:
        return eligible_products

    with st.expander("Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        # Get unique values for filters
        brands = sorted(set(p.get("brand", "N/A") or "N/A" for p in eligible_products))
        suppliers = sorted(set(p.get("supplier", "N/A") or "N/A" for p in eligible_products))

        with col1:
            selected_brand = st.selectbox(
                "Brand",
                options=["All"] + brands,
                key="campaign_filter_brand"
            )

        with col2:
            selected_supplier = st.selectbox(
                "Supplier",
                options=["All"] + suppliers,
                key="campaign_filter_supplier"
            )

        with col3:
            min_discount_filter = st.number_input(
                "Min Discount Override (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key="campaign_filter_min_discount",
                help="Additional filter on top of threshold"
            )

        with col4:
            bestsellers_only = st.checkbox(
                "Bestsellers Only",
                value=False,
                key="campaign_filter_bestsellers"
            )

    # Apply filters
    filtered = eligible_products

    if selected_brand != "All":
        filtered = [p for p in filtered if (p.get("brand") or "N/A") == selected_brand]

    if selected_supplier != "All":
        filtered = [p for p in filtered if (p.get("supplier") or "N/A") == selected_supplier]

    if min_discount_filter > 0:
        filtered = [p for p in filtered if p.get("campaign_discount_pct", 0) >= min_discount_filter]

    if bestsellers_only:
        filtered = [p for p in filtered if p.get("is_bestseller", False)]

    return filtered


def show_eligibility_table(eligible_products: List[Dict]) -> None:
    """Display the main eligibility data table."""
    if not eligible_products:
        st.info("No products to display.")
        return

    st.subheader(f"Eligible Products ({len(eligible_products):,})")

    # Build table data
    table_data = []
    for opp in eligible_products:
        row = {
            "EAN": pad_ean_code(opp.get("ean", "")),
            "Product Name": opp.get("product_name", "N/A")[:50],
            "Brand": opp.get("brand", "N/A") or "N/A",
            "Supplier": opp.get("supplier", "N/A"),
            "Quote Price": f"{opp.get('quote_price', 0):.2f}",
            "Baseline Price": f"{opp.get('campaign_baseline_price', 0):.2f}",
            "Discount %": f"{opp.get('campaign_discount_pct', 0):.1f}%",
            "Savings/Unit": f"{opp.get('campaign_savings_per_unit', 0):.2f}",
            "Sales 90d": opp.get("sales90d", 0),
            "Sales 180d": opp.get("sales180d", 0),
            "Sales 365d": opp.get("sales365d", 0),
            "Bestseller": "Yes" if opp.get("is_bestseller", False) else "No",
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Sort by discount % descending
    df["_discount_sort"] = df["Discount %"].str.replace("%", "").astype(float)
    df = df.sort_values("_discount_sort", ascending=False).drop(columns=["_discount_sort"])

    st.dataframe(df, use_container_width=True, height=500)

    # Column guide
    with st.expander("Column Guide"):
        st.write("""
        - **EAN**: Product barcode/identifier
        - **Quote Price**: Supplier's offered price
        - **Baseline Price**: Minimum of best_buy, supplier_price, and stock_avg prices
        - **Discount %**: How much cheaper the quote is vs baseline
        - **Savings/Unit**: Absolute savings per unit (baseline - quote)
        - **Sales 90d/180d/365d**: Historical sales for context
        - **Bestseller**: High-volume product flag
        """)


def show_export_options(eligible_products: List[Dict]) -> None:
    """Display export options for eligible products."""
    if not eligible_products:
        return

    st.subheader("Export")

    # Build export data
    export_data = []
    for opp in eligible_products:
        export_data.append({
            "EAN": pad_ean_code(opp.get("ean", "")),
            "Product_Name": opp.get("product_name", ""),
            "Brand": opp.get("brand", ""),
            "Supplier": opp.get("supplier", ""),
            "Quote_Price": f"{opp.get('quote_price', 0):.2f}".replace(".", ","),
            "Baseline_Price": f"{opp.get('campaign_baseline_price', 0):.2f}".replace(".", ","),
            "Discount_Percentage": f"{opp.get('campaign_discount_pct', 0):.1f}".replace(".", ","),
            "Savings_Per_Unit": f"{opp.get('campaign_savings_per_unit', 0):.2f}".replace(".", ","),
            "Sales_90d": opp.get("sales90d", 0),
            "Sales_180d": opp.get("sales180d", 0),
            "Sales_365d": opp.get("sales365d", 0),
            "Current_Stock": opp.get("current_stock", 0),
            "Is_Bestseller": opp.get("is_bestseller", False),
        })

    df = pd.DataFrame(export_data)
    csv_content = df.to_csv(index=False, sep=";", encoding="utf-8")

    filename = f"marketing_campaign_eligible_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    st.download_button(
        label="Download Eligible Products (CSV)",
        data=csv_content,
        file_name=filename,
        mime="text/csv",
        help="CSV with semicolon delimiters for Excel compatibility"
    )


# =============================================================================
# MAIN TAB FUNCTION
# =============================================================================


def marketing_campaign_tab(groq_api_key: str, api_key_valid: bool = False) -> None:
    """
    Main marketing campaign eligibility tab.

    Args:
        groq_api_key: API key (not used directly but kept for consistency)
        api_key_valid: Whether API key is valid
    """
    st.header("Marketing Campaign Eligibility")
    st.info(
        "Identify products eligible for marketing campaigns based on price discounts. "
        "Products are eligible when their quoted price is significantly lower than baseline prices."
    )

    # Check prerequisites
    opportunity_engine = st.session_state.get("opportunity_engine")
    if not opportunity_engine:
        st.warning(
            "Run opportunity analysis first in the **Opportunities** tab to generate data for campaign eligibility."
        )
        return

    opportunities = getattr(opportunity_engine, "opportunities", [])
    if not opportunities:
        st.warning(
            "No opportunities found. Please run the opportunity analysis in the **Opportunities** tab first."
        )
        return

    # Show configuration
    config = show_campaign_configuration()

    # Show preset management
    show_preset_management(config)

    st.divider()

    # Calculate eligibility
    eligible_products = find_eligible_products(opportunities, config["discount_threshold"])

    # Show summary
    show_eligibility_summary(eligible_products)

    # Show brand scatter plot
    show_brand_scatter_plot(eligible_products)

    st.divider()

    # Apply filters
    filtered_products = show_eligibility_filters(eligible_products)

    # Show table
    show_eligibility_table(filtered_products)

    st.divider()

    # Export options
    show_export_options(filtered_products)
