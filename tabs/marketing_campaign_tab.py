"""
Marketing Campaign Eligibility Tab
Identifies products eligible for marketing campaigns based on price discount thresholds.
Supports configurable eligibility criteria.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from utils import pad_ean_code

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DISCOUNT_THRESHOLD = 15.0


# =============================================================================
# ELIGIBILITY CALCULATION FUNCTIONS
# =============================================================================


def calculate_baseline_price(opportunity: Dict) -> Optional[float]:
    """
    Calculate baseline price using supplier_price.

    Args:
        opportunity: Opportunity dictionary with price_breakdown

    Returns:
        supplier_price or None if not available
    """
    price_breakdown = opportunity.get("price_breakdown", {})
    supplier_price = price_breakdown.get("supplier_price")

    if supplier_price is not None and supplier_price > 0:
        return supplier_price
    return None


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


def show_eligibility_summary(eligible_products: List[Dict]) -> None:
    """Display summary metrics for eligible products."""
    st.subheader("Campaign Eligibility Summary")

    if not eligible_products:
        st.info("No products meet the current eligibility criteria.")
        return

    # Calculate metrics
    total_products = len(eligible_products)
    avg_discount = sum(p.get("campaign_discount_pct", 0) for p in eligible_products) / total_products

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Eligible Products",
            f"{total_products:,}",
            help="Number of products meeting the discount threshold"
        )

    with col2:
        st.metric(
            "Avg Discount",
            f"{avg_discount:.1f}%",
            help="Average discount percentage across eligible products"
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
            selected_brands = st.multiselect(
                "Brand",
                options=brands,
                default=[],
                key="campaign_filter_brand",
                placeholder="All brands"
            )

        with col2:
            selected_suppliers = st.multiselect(
                "Supplier",
                options=suppliers,
                default=[],
                key="campaign_filter_supplier",
                placeholder="All suppliers"
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

    if selected_brands:  # Empty list means "All"
        filtered = [p for p in filtered if (p.get("brand") or "N/A") in selected_brands]

    if selected_suppliers:  # Empty list means "All"
        filtered = [p for p in filtered if (p.get("supplier") or "N/A") in selected_suppliers]

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
        - **Baseline Price**: Supplier price (internal reference price)
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
# WEIGHTED ELIGIBILITY FUNCTIONS
# =============================================================================


def calculate_weighted_brand_discounts(eligible_products: List[Dict]) -> List[Dict]:
    """
    Calculate weighted discount % per brand based on sales90d.

    Formula: Weighted Brand Discount = Σ(SKU_discount_pct × SKU_sales90d) / Σ(SKU_sales90d)

    Args:
        eligible_products: List of eligible product dictionaries

    Returns:
        List of brand-level dictionaries with weighted discount metrics
    """
    brand_data = {}
    for product in eligible_products:
        brand = product.get("brand") or "Unknown"
        discount = product.get("campaign_discount_pct", 0)
        sales90d = product.get("sales90d", 0)

        if brand not in brand_data:
            brand_data[brand] = {
                "weighted_sum": 0,
                "total_sales": 0,
                "discount_sum": 0,
                "skus": [],
                "products": [],
                "total_savings": 0.0,
            }

        brand_data[brand]["weighted_sum"] += discount * sales90d
        brand_data[brand]["total_sales"] += sales90d
        brand_data[brand]["discount_sum"] += discount
        brand_data[brand]["skus"].append(product.get("ean"))
        brand_data[brand]["products"].append(product)
        brand_data[brand]["total_savings"] += product.get("campaign_savings_per_unit", 0)

    # Calculate weighted average per brand
    result = []
    for brand, data in brand_data.items():
        sku_count = len(data["skus"])
        if data["total_sales"] > 0:
            weighted_discount = data["weighted_sum"] / data["total_sales"]
        else:
            # Fallback to simple average if no sales data
            weighted_discount = data["discount_sum"] / sku_count if sku_count > 0 else 0

        result.append({
            "brand": brand,
            "weighted_discount_pct": weighted_discount,
            "total_sales90d": data["total_sales"],
            "sku_count": sku_count,
            "total_savings": data["total_savings"],
            "products": data["products"],
        })

    return result


def show_weighted_eligibility_config() -> Dict:
    """
    Display weighted campaign configuration UI.

    Returns:
        Current weighted configuration dictionary
    """
    with st.expander("Weighted Campaign Configuration", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            weighted_threshold = st.slider(
                "Minimum Weighted Discount Threshold (%)",
                min_value=0.0,
                max_value=50.0,
                value=st.session_state.get("weighted_discount_threshold", DEFAULT_DISCOUNT_THRESHOLD),
                step=1.0,
                key="weighted_threshold_slider",
                help="Brands with weighted average discount >= this threshold are eligible"
            )
            st.session_state.weighted_discount_threshold = weighted_threshold

        with col2:
            st.metric(
                "Current Threshold",
                f"{weighted_threshold:.0f}%",
                help="Brands must have at least this weighted average discount to be eligible"
            )

    return {"weighted_threshold": weighted_threshold}


def show_weighted_eligibility_filters(brand_results: List[Dict]) -> List[Dict]:
    """
    Display filter controls for weighted eligibility and return filtered list.

    Args:
        brand_results: List of brand-level weighted results

    Returns:
        Filtered list of brand results
    """
    if not brand_results:
        return brand_results

    with st.expander("Weighted Filters", expanded=False):
        col1, col2, col3 = st.columns(3)

        # Get unique brands
        brands = sorted(set(b.get("brand", "N/A") for b in brand_results))

        with col1:
            selected_brands = st.multiselect(
                "Brand",
                options=brands,
                default=[],
                key="weighted_filter_brand",
                placeholder="All brands"
            )

        with col2:
            min_weighted_discount = st.number_input(
                "Min Weighted Discount (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key="weighted_filter_min_discount",
                help="Additional filter on weighted discount"
            )

        with col3:
            min_sku_count = st.number_input(
                "Min SKU Count",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key="weighted_filter_min_skus",
                help="Minimum number of SKUs per brand"
            )

    # Apply filters
    filtered = brand_results

    if selected_brands:
        filtered = [b for b in filtered if b.get("brand") in selected_brands]

    if min_weighted_discount > 0:
        filtered = [b for b in filtered if b.get("weighted_discount_pct", 0) >= min_weighted_discount]

    if min_sku_count > 1:
        filtered = [b for b in filtered if b.get("sku_count", 0) >= min_sku_count]

    return filtered


def show_weighted_eligibility_summary(brand_results: List[Dict]) -> None:
    """Display summary metrics for weighted eligibility."""
    st.subheader("Weighted Campaign Eligibility Summary")

    if not brand_results:
        st.info("No brands meet the current weighted eligibility criteria.")
        return

    # Calculate metrics
    total_brands = len(brand_results)
    total_skus = sum(b.get("sku_count", 0) for b in brand_results)
    avg_weighted_discount = sum(b.get("weighted_discount_pct", 0) for b in brand_results) / total_brands

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Eligible Brands",
            f"{total_brands:,}",
            help="Number of brands meeting the weighted discount threshold"
        )

    with col2:
        st.metric(
            "Total SKUs",
            f"{total_skus:,}",
            help="Total number of SKUs across eligible brands"
        )

    with col3:
        st.metric(
            "Avg Weighted Discount",
            f"{avg_weighted_discount:.1f}%",
            help="Average of weighted discounts across eligible brands"
        )


def show_weighted_brand_scatter_plot(brand_results: List[Dict]) -> None:
    """Display scatter plot of brands: Weighted Discount vs Number of SKUs."""
    if not brand_results:
        return

    # Build DataFrame for plotting
    plot_data = []
    for brand_data in brand_results:
        plot_data.append({
            "Brand": brand_data.get("brand", "Unknown"),
            "SKU Count": brand_data.get("sku_count", 0),
            "Weighted Discount (%)": round(brand_data.get("weighted_discount_pct", 0), 1),
            "Total Sales 90d": brand_data.get("total_sales90d", 0),
            "Total Savings": round(brand_data.get("total_savings", 0), 2),
        })

    df_brands = pd.DataFrame(plot_data)

    # Create scatter plot
    fig = px.scatter(
        df_brands,
        x="Weighted Discount (%)",
        y="SKU Count",
        size="Total Sales 90d",
        color="Brand",
        hover_name="Brand",
        hover_data={
            "Weighted Discount (%)": ":.1f",
            "SKU Count": True,
            "Total Sales 90d": ":,",
            "Total Savings": ":.2f",
        },
        title="Brand Overview: Weighted Discount vs SKU Count (sized by Sales 90d)"
    )

    fig.update_layout(
        xaxis_title="Weighted Average Discount (%)",
        yaxis_title="Number of SKUs",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)


def show_weighted_eligibility_table(brand_results: List[Dict]) -> None:
    """Display the weighted eligibility data table with all products from eligible brands."""
    if not brand_results:
        st.info("No products to display.")
        return

    # Extract all products from filtered brands
    all_products = []
    for brand_data in brand_results:
        all_products.extend(brand_data.get("products", []))

    if not all_products:
        st.info("No products to display.")
        return

    st.subheader(f"Eligible Products ({len(all_products):,})")

    # Build table data
    table_data = []
    for opp in all_products:
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
        - **Baseline Price**: Supplier price (internal reference price)
        - **Discount %**: How much cheaper the quote is vs baseline
        - **Savings/Unit**: Absolute savings per unit (baseline - quote)
        - **Sales 90d/180d/365d**: Historical sales for context
        - **Bestseller**: High-volume product flag
        """)


def show_weighted_export_options(brand_results: List[Dict]) -> None:
    """Display export options for weighted eligibility results."""
    if not brand_results:
        return

    st.subheader("Weighted Export")

    # Build export data (brand level)
    export_data = []
    for brand_data in brand_results:
        export_data.append({
            "Brand": brand_data.get("brand", ""),
            "Weighted_Discount_Percentage": f"{brand_data.get('weighted_discount_pct', 0):.1f}".replace(".", ","),
            "SKU_Count": brand_data.get("sku_count", 0),
            "Total_Sales_90d": brand_data.get("total_sales90d", 0),
            "Total_Savings": f"{brand_data.get('total_savings', 0):.2f}".replace(".", ","),
        })

    df = pd.DataFrame(export_data)
    csv_content = df.to_csv(index=False, sep=";", encoding="utf-8")

    filename = f"weighted_campaign_eligible_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    st.download_button(
        label="Download Weighted Results (CSV)",
        data=csv_content,
        file_name=filename,
        mime="text/csv",
        key="weighted_export_button",
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

    st.divider()

    # Calculate eligibility
    eligible_products = find_eligible_products(opportunities, config["discount_threshold"])

    # Apply filters (after configuration, affects summary and scatter plot)
    filtered_products = show_eligibility_filters(eligible_products)

    # Show summary (uses filtered products)
    show_eligibility_summary(filtered_products)

    # Show brand scatter plot (uses filtered products)
    show_brand_scatter_plot(filtered_products)

    st.divider()

    # Show table
    show_eligibility_table(filtered_products)

    st.divider()

    # Export options
    show_export_options(filtered_products)

    # =========================================================================
    # WEIGHTED MARKETING CAMPAIGN ELIGIBILITY SECTION
    # =========================================================================

    st.divider()
    st.header("Weighted Marketing Campaign Eligibility")
    st.info(
        "Analyze campaign eligibility by brand with weighted discounts. "
        "Each brand's discount is weighted by the Sales 90d of its SKUs, "
        "giving more influence to high-volume products."
    )

    # Weighted configuration
    weighted_config = show_weighted_eligibility_config()

    st.divider()

    # Calculate weighted brand discounts from the same eligible products
    brand_results = calculate_weighted_brand_discounts(eligible_products)

    # Filter by weighted threshold
    eligible_brands = [
        b for b in brand_results
        if b.get("weighted_discount_pct", 0) >= weighted_config["weighted_threshold"]
    ]

    # Apply weighted filters
    filtered_brands = show_weighted_eligibility_filters(eligible_brands)

    # Show weighted summary
    show_weighted_eligibility_summary(filtered_brands)

    # Show weighted scatter plot
    show_weighted_brand_scatter_plot(filtered_brands)

    st.divider()

    # Show weighted table
    show_weighted_eligibility_table(filtered_brands)

    st.divider()

    # Weighted export options
    show_weighted_export_options(filtered_brands)
