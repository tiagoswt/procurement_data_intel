"""
Enhanced Opportunities Tab - With Multi-Supplier Allocation Support AND Sales Period Filter
Compatible with both manual upload and automatic file loading
UPDATED: Only Priority 1 and 2 (Priority 3 eliminated), Stock Average Price column added, Sales Period Filter
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any


def get_safe_allocated_quantity(opp: Dict) -> int:
    """
    Get allocated quantity with safe fallback that respects supplier constraints
    CRITICAL: This prevents allocating more than supplier has available
    """
    # If allocated_quantity exists and is valid, use it
    if "allocated_quantity" in opp and opp["allocated_quantity"] is not None:
        return opp["allocated_quantity"]

    # Calculate safe fallback that respects supplier quantity constraint
    net_need = opp.get("net_need", 0)
    supplier_qty = opp.get("supplier_quantity")

    if supplier_qty is not None and supplier_qty >= 0:
        # CRITICAL: Never allocate more than supplier has available
        return min(supplier_qty, net_need)
    else:
        # Only use net_need if supplier quantity is unknown
        return net_need


def recalculate_net_need_for_period(opp: Dict, sales_period: str) -> int:
    """
    Recalculate net need based on selected sales period
    """
    current_stock = opp.get("current_stock", 0)

    # Get sales for the selected period
    if sales_period == "sales90d":
        sales = opp.get("sales90d", 0)
    elif sales_period == "sales180d":
        sales = opp.get("sales180d", 0)
    elif sales_period == "sales365d":
        sales = opp.get("sales365d", 0)
    else:
        sales = opp.get("sales90d", 0)  # Default fallback

    # Simple formula: if we sold X in period and have Y in stock, we need max(0, X-Y)
    net_need = max(0, sales - current_stock)
    return net_need


def recalculate_opportunity_with_new_period(opp: Dict, sales_period: str) -> Dict:
    """
    Recalculate opportunity metrics based on new sales period
    """
    # Create a copy to avoid modifying original
    new_opp = opp.copy()

    # Recalculate net need based on selected period
    new_net_need = recalculate_net_need_for_period(opp, sales_period)
    new_opp["net_need"] = new_net_need
    new_opp["calculated_with_period"] = sales_period

    # Recalculate allocated quantity based on new net need
    supplier_qty = opp.get("supplier_quantity")
    if supplier_qty is not None and supplier_qty >= 0:
        new_allocated_qty = min(supplier_qty, new_net_need)
    else:
        new_allocated_qty = new_net_need

    new_opp["allocated_quantity"] = new_allocated_qty

    # Recalculate total savings based on new allocated quantity
    savings_per_unit = opp.get("savings_per_unit", 0)
    new_total_savings = savings_per_unit * new_allocated_qty
    new_opp["total_savings"] = new_total_savings

    return new_opp


def opportunities_tab(groq_api_key, api_key_valid=False):
    """Main procurement opportunities tab - UPDATED: Only Priority 1 and 2, with Sales Period Filter"""

    st.header("🎯 Procurement Opportunities with Smart Allocation")
    st.info(
        "💡 **Smart Multi-Supplier Allocation**: Automatically splits orders when suppliers have quantity constraints\n"
        "🎯 **Quality Focus**: Only Priority 1 & 2 opportunities (higher impact standards)\n"
        "📊 **Sales Period Filter**: Dynamically calculate Net Need based on different sales periods"
    )

    # Check if we have supplier data
    if not st.session_state.get("processed_data"):
        st.warning("⚠️ **Step 1**: Process supplier catalogs first")
        st.info(
            "Use either 'File Processing' or 'Auto File Loading' tab to load supplier data"
        )
        return

    # Initialize opportunity engine
    if "opportunity_engine" not in st.session_state:
        try:
            from analysis.opportunity_engine import SimpleOpportunityEngine

            st.session_state.opportunity_engine = SimpleOpportunityEngine()
        except ImportError:
            st.error(
                "❌ Opportunity engine not available. Please ensure all modules are installed."
            )
            return

    engine = st.session_state.opportunity_engine

    # Check if internal data is already loaded (from auto-loading)
    if hasattr(engine, "internal_data") and engine.internal_data:
        st.success(
            f"✅ **Internal data already loaded**: {len(engine.internal_data)} products"
        )

        # Show quick summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Internal Products", len(engine.internal_data))
        with col2:
            bestsellers = sum(
                1 for p in engine.internal_data if p.get("is_bestseller", False)
            )
            st.metric("Bestsellers", bestsellers)
        with col3:
            total_stock = sum(p.get("stock", 0) for p in engine.internal_data)
            st.metric("Total Stock", f"{total_stock:,}")

        # Skip to opportunity analysis
        show_smart_opportunity_analysis(engine)
        return

    # Show manual upload section if no auto-loaded data
    st.subheader("📊 Upload Internal Data (Manual)")
    st.info(
        "💡 **Tip**: For automatic loading, use the 'Auto File Loading' tab → 'Internal Data' section"
    )

    uploaded_file = st.file_uploader(
        "Upload Internal Product Data (CSV)",
        type=["csv"],
        help="Upload your internal product master data with EAN codes, stock, sales, and pricing information",
    )

    if uploaded_file:
        try:
            # Load and validate internal data
            with st.spinner("Loading internal product data..."):
                success = engine.load_internal_data(uploaded_file)

            if success:
                # Show data summary
                show_internal_data_summary(engine)
                show_smart_opportunity_analysis(engine)
            else:
                st.error(
                    "❌ Failed to load internal data. Please check your CSV format."
                )

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")


def show_internal_data_summary(engine):
    """Show summary of loaded internal data"""

    internal_data = engine.internal_data
    st.success(f"✅ Loaded {len(internal_data)} internal products")

    # Calculate summary stats
    total_stock = sum(p.get("stock", 0) for p in internal_data)
    total_sales90d = sum(p.get("sales90d", 0) for p in internal_data)
    total_sales180d = sum(p.get("sales180d", 0) for p in internal_data)
    total_sales365d = sum(p.get("sales365d", 0) for p in internal_data)
    bestsellers = sum(1 for p in internal_data if p.get("is_bestseller", False))
    active_products = sum(1 for p in internal_data if p.get("is_active", True))

    # Count products with different price types
    with_stock_avg = sum(
        1
        for p in internal_data
        if p.get("stock_avg_price") is not None and p.get("stock_avg_price") > 0
    )
    with_supplier_price = sum(
        1
        for p in internal_data
        if p.get("supplier_price") is not None and p.get("supplier_price") > 0
    )
    with_best_buy = sum(
        1
        for p in internal_data
        if p.get("best_buy_price") is not None and p.get("best_buy_price") > 0
    )

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Products", len(internal_data))
    with col2:
        st.metric("Active Products", active_products)
    with col3:
        st.metric("Bestsellers (1-2)", bestsellers)
    with col4:
        st.metric("Total Stock", f"{total_stock:,}")
    with col5:
        st.metric("Sales 90d", f"{total_sales90d:,}")

    # Sales period summary
    st.subheader("📊 Sales Data Availability")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Price availability moved here for space
        st.metric(
            "Stock Avg Price",
            with_stock_avg,
            f"{with_stock_avg/len(internal_data)*100:.1f}%",
        )
    with col2:
        st.metric("Sales 90d", f"{total_sales90d:,}")
    with col3:
        st.metric("Sales 180d", f"{total_sales180d:,}")
    with col4:
        st.metric("Sales 365d", f"{total_sales365d:,}")

    # Show sample data
    with st.expander("📋 Sample Data Preview"):
        sample_data = []
        for product in internal_data[:5]:
            # Get the best price for display
            prices = []
            for price_field, price_value in [
                ("Stock Avg", product.get("stock_avg_price")),
                ("Supplier", product.get("supplier_price")),
                ("Best Buy", product.get("best_buy_price")),
            ]:
                if price_value is not None and price_value > 0:
                    prices.append(f"{price_field}: €{price_value:.2f}")

            bestseller_info = ""
            if product.get("bestseller_rank"):
                rank = product.get("bestseller_rank")
                stars = "⭐" * max(0, 5 - rank)
                bestseller_info = f"Rank {rank} {stars}"

            sample_data.append(
                {
                    "EAN": product["ean"],
                    "Description": (
                        (product.get("description", "")[:50] + "...")
                        if len(product.get("description", "")) > 50
                        else product.get("description", "N/A")
                    ),
                    "Brand": product.get("brand", "N/A"),
                    "Stock": product.get("stock", 0),
                    "Sales 90d": product.get("sales90d", 0),
                    "Sales 180d": product.get("sales180d", 0),
                    "Sales 365d": product.get("sales365d", 0),
                    "Prices Available": " | ".join(prices) if prices else "None",
                    "Bestseller": bestseller_info,
                }
            )

        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)


def show_smart_opportunity_analysis(engine):
    """Show opportunity analysis interface with smart allocation and sales period filter"""

    st.subheader("💰 Smart Opportunity Analysis")
    st.info(
        "🤖 **Smart Multi-Supplier Allocation**: Automatically handles quantity constraints by splitting orders across suppliers\n"
        "🎯 **Quality Standards**: Only showing Priority 1 & 2 opportunities (higher impact focus)\n"
        "📊 **Dynamic Sales Periods**: Calculate Net Need based on your preferred sales period"
    )

    # Get supplier data
    supplier_data = st.session_state.get("processed_data")

    if not supplier_data:
        st.warning(
            "⚠️ No supplier data available. Please process supplier catalogs first."
        )
        return

    # Calculate opportunities with smart allocation (only once)
    if not hasattr(engine, "opportunities") or not engine.opportunities:
        with st.spinner(
            "🧠 Analyzing procurement opportunities with smart multi-supplier allocation (Quality Focus: Priority 1 & 2 only)..."
        ):
            try:
                opportunities = engine.find_opportunities(supplier_data)

                # Verify allocation worked
                try:
                    verification = engine.verify_allocation()
                    if not verification.get("allocation_success", False):
                        st.warning("⚠️ **Allocation Issues Detected!**")
                        st.write("**Issues found:**")
                        for issue in verification.get("allocation_issues", []):
                            st.write(f"- {issue}")
                    else:
                        st.success(
                            f"✅ **Smart allocation successful!** {verification.get('split_orders', 0)} split orders created"
                        )

                except AttributeError:
                    # Fallback verification
                    st.info("🔧 Using basic allocation verification...")

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return

        if not opportunities:
            st.info("ℹ️ No high-quality opportunities found. This could mean:")
            st.write("- No EAN matches between internal data and supplier quotes")
            st.write("- Supplier prices don't meet Priority 1 or 2 standards")
            st.write("- No net need (current stock covers sales)")
            st.write(
                "- **Quality Filter**: Lower-impact opportunities were filtered out"
            )

            # Show debug info
            with st.expander("🔍 Debug Information"):
                st.write(f"**Internal products loaded:** {len(engine.internal_data)}")
                st.write(f"**Supplier products available:** {len(supplier_data)}")

                # Show sample EANs
                internal_eans = [p["ean"] for p in engine.internal_data[:10]]
                supplier_eans = []
                for sp in supplier_data[:10]:
                    if hasattr(sp, "ean_code") and sp.ean_code:
                        supplier_eans.append(str(sp.ean_code))

                st.write(f"**Sample Internal EANs:** {internal_eans}")
                st.write(f"**Sample Supplier EANs:** {supplier_eans}")
                st.write("**Priority Standards:**")
                st.write("- Priority 1: Quote beats ALL 3 internal prices")
                st.write(
                    "- Priority 2: Quote beats supplier AND stock avg (not bestbuy)"
                )
                st.write(
                    "- **Eliminated**: Opportunities that beat only 1 price (former Priority 3)"
                )

            return

    # Show enhanced opportunity summary with allocation info
    show_enhanced_allocation_summary(engine)

    # Show allocation verification
    show_allocation_status(engine)

    # Show savings calculation verification
    show_savings_verification(engine)

    # Show priority explanation
    show_priority_and_allocation_explanation()

    # Show filtering and opportunities WITH SALES PERIOD FILTER
    filtered_opportunities = show_enhanced_allocation_filters(engine)

    # Show opportunities table with allocation details
    show_enhanced_allocation_table(filtered_opportunities)

    # Show supplier analysis with allocation insights
    show_enhanced_allocation_supplier_analysis(engine)

    # Export options with allocation data
    show_allocation_export_options(filtered_opportunities)


def show_enhanced_allocation_filters(engine):
    """Show filtering options with Sales Period Filter - UPDATED: Added Brand Filter"""

    opportunities = engine.opportunities if hasattr(engine, "opportunities") else []

    with st.expander("🔍 Filter Opportunities"):
        # First row - Sales Period Filter (prominent position)
        st.subheader("📊 Sales Period for Net Need Calculation")

        col1, col2 = st.columns([1, 3])

        with col1:
            sales_period_filter = st.selectbox(
                "Sales Period",
                options=["sales90d", "sales180d", "sales365d"],
                index=0,  # Default to 90 days
                format_func=lambda x: {
                    "sales90d": "90 Days",
                    "sales180d": "180 Days",
                    "sales365d": "365 Days",
                }.get(x, x),
                help="Select which sales period to use for Net Need calculation (Sales Period - Current Stock)",
                key="sales_period_filter",
            )

        with col2:
            # Show what this affects
            period_name = {
                "sales90d": "90 days",
                "sales180d": "180 days",
                "sales365d": "365 days",
            }.get(sales_period_filter, sales_period_filter)

            st.info(
                f"📈 **Net Need will be calculated as**: Sales in last {period_name} - Current Stock"
            )

        st.divider()

        # Second row - Standard filters (updated with brand filter)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            priority_filter = st.selectbox(
                "Priority Level",
                options=[None, 1, 2],  # REMOVED: 3
                format_func=lambda x: (
                    "All Priorities" if x is None else f"Priority {x}"
                ),
                help="Filter by opportunity priority level (only Priority 1 and 2 available)",
                key="priority_filter",
            )

        with col2:
            allocation_filter = st.selectbox(
                "Allocation Type",
                options=[None, "single_only", "split_only"],
                format_func=lambda x: {
                    None: "All Types",
                    "single_only": "Single Supplier Only",
                    "split_only": "Split Orders Only",
                }.get(x, x),
                help="Filter by allocation type",
                key="allocation_filter",
            )

        with col3:
            # NEW: Brand filter
            available_brands = sorted(
                set(
                    opp.get("brand", "").strip()
                    for opp in opportunities
                    if opp.get("brand", "").strip()
                )
            )
            if "" in [opp.get("brand", "") for opp in opportunities]:
                available_brands = ["All Brands", "Unknown/Empty"] + [
                    b for b in available_brands if b
                ]
            else:
                available_brands = ["All Brands"] + available_brands

            brand_filter = st.selectbox(
                "Brand",
                options=available_brands,
                help="Filter opportunities by product brand",
                key="brand_filter",
            )

        with col4:
            urgency_filter = st.selectbox(
                "Urgency Level",
                options=[None, "High", "Medium", "Low"],
                help="Filter by stock urgency level",
                key="urgency_filter",
            )

        # Third row - Numeric filters and checkboxes
        col1, col2, col3 = st.columns(3)

        with col1:
            min_savings = st.number_input(
                "Min Savings/Unit (€)",
                min_value=0.0,
                value=0.01,
                step=0.05,
                help="Minimum savings per unit in euros",
                key="min_savings_filter",
            )

        with col2:
            min_total = st.number_input(
                "Min Total Savings (€)",
                min_value=0.0,
                value=0.01,
                step=1.0,
                help="Minimum total savings in euros",
                key="min_total_filter",
            )

        with col3:
            bestsellers_only = st.checkbox(
                "Bestsellers Only",
                help="Show only bestselling products (rank 1-2)",
                key="bestsellers_filter",
            )

    # FIRST: Recalculate opportunities based on selected sales period
    recalculated_opportunities = []
    for opp in opportunities:
        new_opp = recalculate_opportunity_with_new_period(opp, sales_period_filter)
        recalculated_opportunities.append(new_opp)

    # THEN: Apply other filters to the recalculated opportunities
    try:
        filtered = engine.filter_opportunities(
            min_savings=min_savings,
            min_total_savings=min_total,
            urgency_filter=urgency_filter,
            bestsellers_only=bestsellers_only,
            priority_filter=priority_filter,
            allocation_type_filter=allocation_filter,
            brand_filter=brand_filter,
        )

        # Apply sales period recalculation to filtered results
        filtered_recalculated = []
        for opp in filtered:
            new_opp = recalculate_opportunity_with_new_period(opp, sales_period_filter)
            filtered_recalculated.append(new_opp)

        filtered = filtered_recalculated

    except:
        # Fallback if enhanced filter method not available
        filtered = recalculated_opportunities.copy()

        if priority_filter is not None:
            filtered = [
                opp for opp in filtered if opp.get("priority") == priority_filter
            ]
        if allocation_filter == "split_only":
            filtered = [opp for opp in filtered if opp.get("is_split_order", False)]
        elif allocation_filter == "single_only":
            filtered = [opp for opp in filtered if not opp.get("is_split_order", False)]
        if min_savings > 0:
            filtered = [
                opp for opp in filtered if opp.get("savings_per_unit", 0) >= min_savings
            ]
        if min_total > 0:
            filtered = [
                opp for opp in filtered if opp.get("total_savings", 0) >= min_total
            ]
        if urgency_filter:
            filtered = [
                opp for opp in filtered if opp.get("urgency_score") == urgency_filter
            ]
        if bestsellers_only:
            filtered = [opp for opp in filtered if opp.get("is_bestseller", False)]
        if brand_filter and brand_filter not in ["All Brands"]:
            if brand_filter == "Unknown/Empty":
                filtered = [opp for opp in filtered if not opp.get("brand", "").strip()]
            else:
                filtered = [
                    opp
                    for opp in filtered
                    if opp.get("brand", "").strip() == brand_filter
                ]

    # Show filtering results
    filter_info_parts = []

    if brand_filter and brand_filter not in ["All Brands"]:
        if brand_filter == "Unknown/Empty":
            filter_info_parts.append("products without brand info")
        else:
            filter_info_parts.append(f"'{brand_filter}' brand")

    if priority_filter is not None:
        filter_info_parts.append(f"Priority {priority_filter}")

    if allocation_filter:
        allocation_name = {
            "split_only": "split orders",
            "single_only": "single supplier orders",
        }.get(allocation_filter, allocation_filter)
        filter_info_parts.append(allocation_name)

    if urgency_filter:
        filter_info_parts.append(f"{urgency_filter.lower()} urgency")

    if bestsellers_only:
        filter_info_parts.append("bestsellers")

    if min_savings > 0:
        filter_info_parts.append(f"≥€{min_savings:.2f}/unit savings")

    if min_total > 0:
        filter_info_parts.append(f"≥€{min_total:.2f} total savings")

    if len(filtered) != len(opportunities):
        period_display = {
            "sales90d": "90 days",
            "sales180d": "180 days",
            "sales365d": "365 days",
        }.get(sales_period_filter, sales_period_filter)

        filter_description = ""
        if filter_info_parts:
            filter_description = f" (filtered by: {', '.join(filter_info_parts)})"

        st.info(
            f"ℹ️ Showing {len(filtered)} opportunities (filtered from {len(opportunities)} total){filter_description} "
            f"with Net Need calculated using {period_display} sales data"
        )
    else:
        period_display = {
            "sales90d": "90 days",
            "sales180d": "180 days",
            "sales365d": "365 days",
        }.get(sales_period_filter, sales_period_filter)

        st.info(
            f"📊 Showing all {len(filtered)} opportunities with Net Need calculated using {period_display} sales data"
        )

    # Show quality improvement message
    if len(opportunities) > 0:
        priority_1_count = sum(1 for opp in opportunities if opp.get("priority") == 1)
        priority_2_count = sum(1 for opp in opportunities if opp.get("priority") == 2)

        st.success(
            f"🎯 **Quality Focus**: {len(opportunities)} high-impact opportunities found "
            f"({priority_1_count} Priority 1, {priority_2_count} Priority 2) - "
            f"Lower-impact opportunities filtered out for strategic focus"
        )

    return filtered


def show_enhanced_allocation_table(opportunities: List[Dict]):
    """Show opportunities table with all sales columns and dynamic Net Need calculation"""

    if not opportunities:
        st.info("No opportunities match your filters.")
        return

    st.subheader(f"📋 Smart Allocation Results ({len(opportunities)} opportunities)")

    # Show which sales period is being used for Net Need
    if opportunities:
        sales_period_used = opportunities[0].get("calculated_with_period", "sales90d")
        period_display = {
            "sales90d": "90 days",
            "sales180d": "180 days",
            "sales365d": "365 days",
        }.get(sales_period_used, sales_period_used)

        st.info(f"📊 **Net Need calculated using**: {period_display} sales data")

    # Count allocation types
    split_orders = sum(1 for opp in opportunities if opp.get("is_split_order", False))
    single_orders = len(opportunities) - split_orders

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Opportunities", len(opportunities))
    with col2:
        st.metric("🔄 Split Orders", split_orders)
    with col3:
        st.metric("📦 Single Orders", single_orders)

    # Prepare data for display with all sales columns
    table_data = []
    for i, opp in enumerate(opportunities):
        # Truncate product name if too long
        product_name = opp.get("product_name", "N/A")
        if len(product_name) > 35:
            product_name = product_name[:35] + "..."

        # Format allocation info
        allocation_info = "📦 Single"
        if opp.get("is_split_order", False):
            rank = opp.get("split_order_rank", 1)
            allocated_qty = get_safe_allocated_quantity(opp)
            original_need = opp.get("original_net_need", opp.get("net_need", 0))
            allocation_info = f"🔄 Split #{rank} ({allocated_qty}/{original_need})"

        # Format supplier quantity with constraint indicators
        supplier_qty = opp.get("supplier_quantity")
        qty_display = str(int(supplier_qty)) if supplier_qty is not None else "-"

        # Add quantity constraint indicators
        net_need = opp.get("net_need", 0)  # This is now the recalculated net need
        purchasable_qty = get_safe_allocated_quantity(opp)

        if supplier_qty is not None:
            if supplier_qty == 0:
                qty_display = "0 ⚠️"
            elif supplier_qty < net_need:
                qty_display = f"{int(supplier_qty)} ⚠️ (need {net_need})"
            else:
                qty_display = f"{int(supplier_qty)} ✅"

        # Show actual purchasable quantity vs net need
        purchasable_display = f"{int(purchasable_qty)}"
        if purchasable_qty < net_need:
            purchasable_display = f"{int(purchasable_qty)} ⚠️"
        else:
            purchasable_display = f"{int(purchasable_qty)} ✅"

        # Add allocation indicators for split orders
        if opp.get("is_split_order", False):
            allocated_qty = get_safe_allocated_quantity(opp)
            qty_display = f"{qty_display} (using {allocated_qty})"
            purchasable_display = f"{allocated_qty} (split)"

        # Get price breakdown for detailed view
        price_breakdown = opp.get("price_breakdown", {})

        # Extract stock average price (NO ICONS)
        stock_avg_price = price_breakdown.get("stock_avg_price")
        stock_avg_display = "N/A"
        if stock_avg_price is not None and stock_avg_price > 0:
            stock_avg_display = f"€{stock_avg_price:.2f}"

        # Create price comparison tooltip
        price_details = []
        if price_breakdown.get("best_buy_price"):
            beat_symbol = "✅" if price_breakdown.get("beats_best_buy") else "❌"
            price_details.append(
                f"BestBuy: €{price_breakdown['best_buy_price']:.2f} {beat_symbol}"
            )
        if price_breakdown.get("supplier_price"):
            beat_symbol = "✅" if price_breakdown.get("beats_supplier") else "❌"
            price_details.append(
                f"Supplier: €{price_breakdown['supplier_price']:.2f} {beat_symbol}"
            )
        if price_breakdown.get("stock_avg_price"):
            beat_symbol = "✅" if price_breakdown.get("beats_stock_avg") else "❌"
            price_details.append(
                f"StockAvg: €{price_breakdown['stock_avg_price']:.2f} {beat_symbol}"
            )

        table_data.append(
            {
                "Row": i + 1,
                "Priority": opp.get("priority_label", "N/A"),
                "EAN": opp.get("ean", ""),
                "Product": product_name,
                "Brand": opp.get("brand", "N/A"),
                "Current Stock": opp.get("current_stock", 0),
                "Sales 90d": opp.get("sales90d", 0),
                "Sales 180d": opp.get("sales180d", 0),
                "Sales 365d": opp.get("sales365d", 0),
                "Net Need": net_need,  # This is the dynamically calculated net need
                "Stock Avg Price": stock_avg_display,
                "Baseline Price": f"€{opp.get('baseline_price', 0):.2f}",
                "Quote Price": f"€{opp.get('quote_price', 0):.2f}",
                "Total Cost": f"€{opp.get('quote_price', 0) * get_safe_allocated_quantity(opp):.2f}",
                "Savings/Unit": f"€{opp.get('savings_per_unit', 0):.2f}",
                "Total Savings": f"€{opp.get('total_savings', 0):.2f}",
                "Allocated Qty": get_safe_allocated_quantity(opp),
                "Supplier Qty": qty_display,
                "Allocation": allocation_info,
                "Supplier": opp.get("supplier", "Unknown"),
                "Urgency": opp.get("urgency_score", ""),
                "Days Cover": f"{opp.get('days_of_cover', 0):.0f}",
                "Bestseller": opp.get("bestseller_stars", ""),
                "Price Comparison": (
                    " | ".join(price_details) if price_details else "N/A"
                ),
            }
        )

    # Display table - ENSURE ALL OPPORTUNITIES ARE SHOWN
    df = pd.DataFrame(table_data)

    # Configure Streamlit to show all rows
    st.write(f"**📋 Showing ALL {len(df)} opportunities:**")

    # Display the full table without any row limits
    st.dataframe(
        df,
        use_container_width=True,
        height=600,  # Set a fixed height to allow scrolling through all rows
    )

    st.info(
        f"✅ **All {len(opportunities)} opportunities displayed** - Use scroll bar to navigate through all rows"
    )

    # Add explanation for columns
    with st.expander("📖 Column Guide"):
        st.write("**Sales Columns:**")
        st.write("• **Sales 90d/180d/365d**: Historical sales for different periods")
        st.write("• **Net Need**: Calculated as Selected Sales Period - Current Stock")
        st.write(
            "• **Current calculation**: Based on the sales period filter you selected"
        )
        st.write("")
        st.write("**Stock Average Price column shows:**")
        st.write("• **€X.XX**: The stock average price for this product")
        st.write("• **N/A**: Stock average price not available for this product")
        st.write("")
        st.write("**Total Cost column shows:**")
        st.write("• **€X.XX**: Quote Price × Allocated Quantity")
        st.write("• **Total investment** required for each product allocation")
        st.write("")
        st.write(
            "**Note**: All sales columns are shown but only the selected sales period is used for Net Need calculation"
        )

    # Show split order details if any
    if split_orders > 0:
        show_split_order_analysis(opportunities)

    # Show allocation summary
    with st.expander("📊 Allocation Summary"):
        # Group by EAN to show split order details
        ean_groups = {}
        for opp in opportunities:
            ean = opp.get("ean", "")
            if ean not in ean_groups:
                ean_groups[ean] = []
            ean_groups[ean].append(opp)

        split_ean_count = sum(1 for group in ean_groups.values() if len(group) > 1)
        single_ean_count = len(ean_groups) - split_ean_count

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Unique Products", len(ean_groups))
        with col2:
            st.metric("Products with Split Orders", split_ean_count)
        with col3:
            st.metric("Products with Single Supplier", single_ean_count)
        with col4:
            total_savings = sum(opp.get("total_savings", 0) for opp in opportunities)
            st.metric("Total Optimized Savings", f"€{total_savings:.2f}")


# [Keep all other existing functions unchanged - they don't need modification for the sales period filter]
# The rest of the functions (show_enhanced_allocation_summary, show_allocation_status, etc.)
# remain exactly the same as they were in the original file.


def show_enhanced_allocation_summary(engine):
    """Show summary metrics with allocation breakdown and stock average comparison - UPDATED: Only Priority 1 and 2"""

    try:
        summary = engine.get_summary()
    except:
        # Fallback calculation if get_summary fails
        opportunities = engine.opportunities if hasattr(engine, "opportunities") else []
        summary = {
            "total_opportunities": len(opportunities),
            "total_potential_savings": sum(
                opp.get("total_savings", 0) for opp in opportunities
            ),
            "high_urgency_count": sum(
                1 for opp in opportunities if opp.get("urgency_score") == "High"
            ),
            "bestseller_opportunities": sum(
                1 for opp in opportunities if opp.get("is_bestseller", False)
            ),
            "priority_1_count": sum(
                1 for opp in opportunities if opp.get("priority") == 1
            ),
            "priority_2_count": sum(
                1 for opp in opportunities if opp.get("priority") == 2
            ),
            # REMOVED: priority_3_count
            "split_orders_count": sum(
                1 for opp in opportunities if opp.get("is_split_order", False)
            ),
            "single_orders_count": len(opportunities)
            - sum(1 for opp in opportunities if opp.get("is_split_order", False)),
        }

    # Enhanced cost calculations including stock average comparison
    if hasattr(engine, "opportunities") and engine.opportunities:
        total_baseline_cost = 0
        total_quote_cost = 0
        total_stock_avg_cost = 0  # NEW: Stock average based cost

        for opp in engine.opportunities:
            allocated_qty = get_safe_allocated_quantity(opp)
            baseline_price = opp.get("baseline_price", 0)
            quote_price = opp.get("quote_price", 0)

            # Calculate baseline and quote costs (existing logic)
            total_baseline_cost += baseline_price * allocated_qty
            total_quote_cost += quote_price * allocated_qty

            # NEW: Calculate stock average based cost
            price_breakdown = opp.get("price_breakdown", {})
            stock_avg_price = price_breakdown.get("stock_avg_price")
            supplier_price = price_breakdown.get("supplier_price")

            # Use stock average price, fallback to supplier price if not available
            reference_price = None
            if stock_avg_price is not None and stock_avg_price > 0:
                reference_price = stock_avg_price
            elif supplier_price is not None and supplier_price > 0:
                reference_price = supplier_price

            if reference_price is not None:
                total_stock_avg_cost += reference_price * allocated_qty

        # Calculate cost savings percentage vs baseline
        cost_savings_percentage = (
            ((total_baseline_cost - total_quote_cost) / total_baseline_cost * 100)
            if total_baseline_cost > 0
            else 0
        )

        # NEW: Calculate stock average comparison metrics
        stock_avg_absolute_difference = total_stock_avg_cost - total_quote_cost
        stock_avg_savings_percentage = (
            ((total_stock_avg_cost - total_quote_cost) / total_stock_avg_cost * 100)
            if total_stock_avg_cost > 0
            else 0
        )

        # Update summary with new calculations
        summary.update(
            {
                "total_baseline_cost": total_baseline_cost,
                "total_quote_cost": total_quote_cost,
                "cost_savings_percentage": cost_savings_percentage,
                "total_stock_avg_cost": total_stock_avg_cost,  # NEW
                "stock_avg_absolute_difference": stock_avg_absolute_difference,  # NEW
                "stock_avg_savings_percentage": stock_avg_savings_percentage,  # NEW
            }
        )
    else:
        summary.update(
            {
                "total_baseline_cost": 0,
                "total_quote_cost": 0,
                "cost_savings_percentage": 0,
                "total_stock_avg_cost": 0,
                "stock_avg_absolute_difference": 0,
                "stock_avg_savings_percentage": 0,
            }
        )

    # Main metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Opportunities",
            summary["total_opportunities"],
            help="Number of products with cost-saving opportunities (Priority 1 & 2 only)",
        )

    with col2:
        st.metric(
            "Potential Savings",
            f"€{summary['total_potential_savings']:.2f}",
            help="Total potential savings across all opportunities",
        )

    with col3:
        st.metric(
            "Cost Savings %",
            f"{summary.get('cost_savings_percentage', 0):.1f}%",
            help="Percentage saved vs baseline prices for all opportunities",
        )

    with col4:
        st.metric(
            "High Urgency",
            summary["high_urgency_count"],
            help="Products with less than 14 days of stock cover",
        )

    with col5:
        st.metric(
            "Bestseller Opportunities",
            summary["bestseller_opportunities"],
            help="Opportunities for bestselling products (rank 1-2)",
        )

    # Enhanced cost breakdown with stock average comparison
    if summary.get("total_baseline_cost", 0) > 0:
        st.subheader("💰 Total Cost Comparison")

        # First row: Original baseline vs opportunity comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "📊 Baseline Total Cost",
                f"€{summary['total_baseline_cost']:.2f}",
                help="Total cost if buying at your baseline prices (best of: stock avg, supplier, or best buy price based on priority)",
            )

        with col2:
            st.metric(
                "🎯 Opportunity Total Cost",
                f"€{summary['total_quote_cost']:.2f}",
                help="Total cost if buying at the quoted opportunity prices",
            )

        with col3:
            baseline_absolute_savings = (
                summary["total_baseline_cost"] - summary["total_quote_cost"]
            )
            st.metric(
                "💵 Baseline Savings",
                f"€{baseline_absolute_savings:.2f}",
                f"{summary['cost_savings_percentage']:.1f}% vs baseline",
                help="Total money saved vs baseline prices",
            )

        # NEW: Second row - Stock Average comparison
        if summary.get("total_stock_avg_cost", 0) > 0:
            st.subheader("📈 Stock Average Price Comparison")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📊 Stock Avg Total Cost",
                    f"€{summary['total_stock_avg_cost']:.2f}",
                    help="Total cost using stock average prices (fallback to supplier price when stock avg not available)",
                )

            with col2:
                st.metric(
                    "🎯 Opportunity Total Cost",
                    f"€{summary['total_quote_cost']:.2f}",
                    help="Total cost if buying at the quoted opportunity prices",
                )

            with col3:
                st.metric(
                    "💰 Stock Avg Savings",
                    f"€{summary['stock_avg_absolute_difference']:.2f}",
                    f"{summary['stock_avg_savings_percentage']:.1f}% vs stock avg",
                    help="Total money saved vs stock average prices",
                    delta_color=(
                        "normal"
                        if summary["stock_avg_absolute_difference"] >= 0
                        else "inverse"
                    ),
                )

            # Show comparison insight
            if summary["stock_avg_absolute_difference"] > 0:
                st.success(
                    f"✅ **Stock Average Savings**: You could save €{summary['stock_avg_absolute_difference']:.2f} "
                    f"({summary['stock_avg_savings_percentage']:.1f}%) compared to your current stock average pricing!"
                )
            elif summary["stock_avg_absolute_difference"] < 0:
                st.warning(
                    f"⚠️ **Higher than Stock Average**: Opportunities cost €{abs(summary['stock_avg_absolute_difference']):.2f} "
                    f"({abs(summary['stock_avg_savings_percentage']):.1f}%) more than stock average pricing."
                )
            else:
                st.info(
                    "ℹ️ **Similar to Stock Average**: Opportunity prices are similar to stock average pricing."
                )

    # UPDATED: Priority breakdown row - Only Priority 1 and 2
    st.subheader("🎯 Priority Breakdown")
    col1, col2, col3 = st.columns(3)

    with col1:
        priority_1_pct = (
            summary["priority_1_count"] / summary["total_opportunities"] * 100
            if summary["total_opportunities"] > 0
            else 0
        )
        st.metric(
            "🔥 Priority 1",
            summary.get("priority_1_count", 0),
            f"{priority_1_pct:.1f}%",
            help="Better than ALL internal prices (bestbuy, supplier, stock avg)",
        )

    with col2:
        priority_2_pct = (
            summary["priority_2_count"] / summary["total_opportunities"] * 100
            if summary["total_opportunities"] > 0
            else 0
        )
        st.metric(
            "⭐ Priority 2",
            summary.get("priority_2_count", 0),
            f"{priority_2_pct:.1f}%",
            help="Better than operational prices (supplier & stock avg) but not bestbuy",
        )

    with col3:
        # Show total qualified opportunities instead of Priority 3
        total_qualified = summary.get("priority_1_count", 0) + summary.get(
            "priority_2_count", 0
        )
        qualification_rate = (
            (total_qualified / summary["total_opportunities"] * 100)
            if summary["total_opportunities"] > 0
            else 0
        )
        st.metric(
            "✅ Total Qualified",
            total_qualified,
            f"{qualification_rate:.1f}%",
            help="Total opportunities meeting Priority 1 or 2 criteria (100% with quality focus)",
        )

    # Allocation breakdown row
    st.subheader("🔄 Smart Allocation Breakdown")
    col1, col2, col3 = st.columns(3)

    with col1:
        split_pct = (
            summary.get("split_orders_count", 0) / summary["total_opportunities"] * 100
            if summary["total_opportunities"] > 0
            else 0
        )
        st.metric(
            "🔄 Split Orders",
            summary.get("split_orders_count", 0),
            f"{split_pct:.1f}%",
            help="Orders split across multiple suppliers due to quantity constraints",
        )

    with col2:
        single_pct = (
            summary.get("single_orders_count", 0) / summary["total_opportunities"] * 100
            if summary["total_opportunities"] > 0
            else 0
        )
        st.metric(
            "📦 Single Supplier",
            summary.get("single_orders_count", 0),
            f"{single_pct:.1f}%",
            help="Orders fulfilled by single supplier (sufficient quantity)",
        )

    with col3:
        # Calculate average suppliers per EAN for split orders
        if hasattr(engine, "opportunities") and engine.opportunities:
            split_opportunities = [
                opp for opp in engine.opportunities if opp.get("is_split_order", False)
            ]
            if split_opportunities:
                # Group by EAN to count suppliers per product
                ean_supplier_counts = {}
                for opp in split_opportunities:
                    ean = opp.get("ean", "")
                    if ean not in ean_supplier_counts:
                        ean_supplier_counts[ean] = 0
                    ean_supplier_counts[ean] += 1

                avg_suppliers = (
                    sum(ean_supplier_counts.values()) / len(ean_supplier_counts)
                    if ean_supplier_counts
                    else 0
                )
                st.metric(
                    "📊 Avg Suppliers/Split",
                    f"{avg_suppliers:.1f}",
                    help="Average number of suppliers used per split order",
                )
            else:
                st.metric("📊 Avg Suppliers/Split", "0.0")
        else:
            st.metric("📊 Avg Suppliers/Split", "0.0")


def show_savings_verification(engine):
    """Show savings calculation verification"""

    with st.expander("🧮 Verify Savings Calculations"):
        st.write("**Debug tool to verify all savings calculations are correct**")

        if st.button("🔍 Verify All Calculations", key="verify_calculations_btn"):
            try:
                verification_results = engine.verify_savings_calculations()

                if "error" in verification_results:
                    st.error(f"❌ {verification_results['error']}")
                    return

                # Show overall results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total Opportunities",
                        verification_results["total_opportunities"],
                    )
                with col2:
                    st.metric(
                        "Correct Calculations",
                        verification_results["correct_calculations"],
                    )
                with col3:
                    accuracy = verification_results["accuracy_rate"]
                    st.metric("Accuracy Rate", f"{accuracy:.1f}%")

                if verification_results["incorrect_calculations"] > 0:
                    st.error(
                        f"❌ Found {verification_results['incorrect_calculations']} calculation errors!"
                    )

                    # Show issues found
                    st.write("**🚨 Issues Found:**")
                    for i, issue in enumerate(
                        verification_results["issues_found"][:10]
                    ):  # Show first 10 issues
                        with st.expander(
                            f"Issue #{i+1}: EAN {issue['ean']} - {issue['supplier']}"
                        ):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Input Values:**")
                                st.write(f"• Net Need: {issue['net_need']}")
                                st.write(
                                    f"• Supplier Qty: {issue['supplier_quantity']}"
                                )
                                st.write(
                                    f"• Savings/Unit: €{issue['savings_per_unit']:.2f}"
                                )

                            with col2:
                                st.write("**Calculation Results:**")
                                st.write(
                                    f"• Recorded Purchasable: {issue['recorded_purchasable']}"
                                )
                                st.write(
                                    f"• Expected Purchasable: {issue['expected_purchasable']}"
                                )
                                st.write(
                                    f"• Recorded Total Savings: €{issue['recorded_total_savings']:.2f}"
                                )
                                st.write(
                                    f"• Expected Total Savings: €{issue['expected_total_savings']:.2f}"
                                )

                            if issue["purchasable_error"]:
                                st.error("❌ Purchasable quantity calculation error")
                            if issue["savings_error"]:
                                st.error("❌ Total savings calculation error")

                    if len(verification_results["issues_found"]) > 10:
                        st.error(
                            f"• ... and {len(verification_results['issues_found']) - 10} more issues"
                        )
                else:
                    st.success("✅ All savings calculations are correct!")

                # Show sample calculations
                if verification_results["calculation_details"]:
                    st.write("**📊 Sample Calculation Details:**")

                    sample_data = []
                    for detail in verification_results["calculation_details"]:
                        sample_data.append(
                            {
                                "EAN": detail["ean"],
                                "Net Need": detail["net_need"],
                                "Supplier Qty": (
                                    detail["supplier_quantity"]
                                    if detail["supplier_quantity"] is not None
                                    else "Unknown"
                                ),
                                "Purchasable": f"{detail['purchasable_qty']} (exp: {detail['expected_purchasable']})",
                                "Savings/Unit": f"€{detail['savings_per_unit']:.2f}",
                                "Total Savings": f"€{detail['total_savings']:.2f} (exp: €{detail['expected_savings']:.2f})",
                                "Status": "✅" if detail["is_correct"] else "❌",
                            }
                        )

                    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

            except AttributeError:
                st.warning(
                    "⚠️ Verification method not available in current engine version"
                )
                st.info("💡 The savings calculations should use this logic:")
                st.write("• **If supplier_qty ≥ net_need**: purchasable = net_need")
                st.write("• **If supplier_qty < net_need**: purchasable = supplier_qty")
                st.write("• **If supplier_qty unknown**: purchasable = net_need")
                st.write("• **Total Savings = savings_per_unit × purchasable**")

            except Exception as e:
                st.error(f"❌ Error during verification: {str(e)}")

        st.write("**Expected Calculation Logic:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**📦 Purchasable Quantity**")
            st.write("• If supplier_qty ≥ net_need:")
            st.write("  → purchasable = net_need")
            st.write("• If supplier_qty < net_need:")
            st.write("  → purchasable = supplier_qty")
            st.write("• If supplier_qty unknown:")
            st.write("  → purchasable = net_need")

        with col2:
            st.write("**💰 Total Savings**")
            st.write("• Formula:")
            st.write("  → total_savings = savings_per_unit × purchasable_qty")
            st.write("• Never use net_need directly")
            st.write("• Always use purchasable quantity")

        with col3:
            st.write("**🔍 Common Issues**")
            st.write("• Using net_need instead of purchasable")
            st.write("• Not handling unknown quantities")
            st.write("• Incorrect min() calculation")
            st.write("• Split order recalculation errors")


def show_allocation_status(engine):
    """Show allocation verification status"""

    try:
        verification = engine.verify_allocation()

        with st.expander("🔍 Smart Allocation Status"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Opportunities", verification.get("total_opportunities", 0)
                )
            with col2:
                st.metric("Unique EAN Codes", verification.get("unique_eans", 0))
            with col3:
                st.metric("Split Orders", verification.get("split_orders", 0))
            with col4:
                st.metric("Single Orders", verification.get("single_orders", 0))

            if not verification.get("allocation_success", False):
                st.write("**Allocation Issues:**")
                for issue in verification.get("allocation_issues", []):
                    st.write(f"• {issue}")
                st.warning(
                    "⚠️ Some products have quantity shortfalls even after smart allocation."
                )
            else:
                st.info(
                    "✅ Smart allocation working correctly - all quantity constraints handled optimally"
                )

    except AttributeError:
        # Fallback if verify_allocation method doesn't exist
        with st.expander("🔍 Allocation Status (Basic Check)"):
            st.info("⚠️ Advanced allocation verification not available")

            if hasattr(engine, "opportunities") and engine.opportunities:
                split_count = sum(
                    1
                    for opp in engine.opportunities
                    if opp.get("is_split_order", False)
                )
                single_count = len(engine.opportunities) - split_count

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Opportunities", len(engine.opportunities))
                with col2:
                    st.metric("Split Orders", split_count)
                with col3:
                    st.metric("Single Orders", single_count)

    except Exception as e:
        st.warning(f"Could not verify allocation: {str(e)}")


def show_priority_and_allocation_explanation():
    """Show explanation of priority system and smart allocation - UPDATED: Only Priority 1 and 2"""

    with st.expander("📖 Priority System & Smart Allocation Guide"):
        # Priority explanation
        st.write("### 🎯 Priority System - High Standards Only")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**🔥 Priority 1 - Maximum Impact**")
            st.write("• Quote beats ALL 3 internal prices")
            st.write("• Beats: bestbuy, supplier, AND stock average")
            st.write("• Baseline: Best of all 3 internal prices")
            st.write("• These are your most valuable opportunities")

            st.write("")
            st.write("**⭐ Priority 2 - Operational Improvement**")
            st.write("• Quote beats supplier AND stock average prices")
            st.write("• Quote does NOT beat bestbuy price")
            st.write("• Baseline: Best of (supplier, stock average)")
            st.write("• Improves your day-to-day operational costs")

            st.write("")
            st.write("**🚫 Priority 3 - ELIMINATED**")
            st.write("• Previously: Quote beats only supplier OR stock avg")
            st.write("• Now filtered out to focus on higher-impact opportunities")
            st.write("• Only accepting opportunities with clear operational benefits")

        with col2:
            st.write("### 🤖 Smart Multi-Supplier Allocation")

            st.write("**How it works:**")
            st.write(
                "1. **Single Supplier Sufficient**: Use best price supplier if they have enough quantity"
            )
            st.write(
                "2. **Quantity Constraints**: Automatically split orders across suppliers"
            )
            st.write(
                "3. **Optimal Allocation**: Always start with cheapest supplier, fill remainder with next best"
            )
            st.write("4. **Cost Transparency**: Shows blended pricing and split costs")

            st.write("")
            st.write("**💰 Savings Calculation Logic:**")
            st.write("• **Total Savings = Savings/Unit × Purchasable Quantity**")
            st.write("• **Purchasable Quantity = min(Supplier Quantity, Net Need)**")
            st.write("• **If supplier qty ≥ net need**: Use net need")
            st.write("• **If supplier qty < net need**: Use supplier qty")
            st.write("• **If supplier qty unknown**: Assume unlimited, use net need")

            st.write("")
            st.write("**📊 Sales Period Filter:**")
            st.write("• **90 days**: Short-term needs (3 months)")
            st.write("• **180 days**: Medium-term needs (6 months)")
            st.write("• **365 days**: Long-term needs (1 year)")
            st.write("• **Net Need = Selected Sales Period - Current Stock**")

            st.write("")
            st.write("**🎯 Quality Focus:**")
            st.write("• **Higher Standards**: Only Priority 1 & 2 opportunities")
            st.write("• **Better ROI**: Focus on meaningful cost improvements")
            st.write("• **Cleaner Results**: Fewer low-impact opportunities")
            st.write("• **Strategic Focus**: Operational and maximum impact only")

            st.write("")
            st.write("**Example Split Order:**")
            st.write("• Need: 100 units")
            st.write("• Supplier A: 75 units @ €10.00 (best price)")
            st.write("• Supplier B: 25 units @ €10.50 (2nd best)")
            st.write("• **Result**: Optimal cost with full fulfillment")


def show_split_order_analysis(opportunities: List[Dict]):
    """Show detailed analysis of split orders"""

    split_opportunities = [
        opp for opp in opportunities if opp.get("is_split_order", False)
    ]

    if not split_opportunities:
        return

    with st.expander(
        f"🔄 Split Order Analysis ({len(split_opportunities)} split entries)"
    ):
        # Group split orders by EAN
        split_ean_groups = {}
        for opp in split_opportunities:
            ean = opp.get("ean", "")
            if ean not in split_ean_groups:
                split_ean_groups[ean] = []
            split_ean_groups[ean].append(opp)

        st.write(
            f"**Split across {len(split_ean_groups)} products with quantity constraints**"
        )

        # Calculate split order metrics
        total_split_savings = sum(
            opp.get("total_savings", 0) for opp in split_opportunities
        )
        avg_suppliers_per_split = sum(
            len(group) for group in split_ean_groups.values()
        ) / len(split_ean_groups)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Products with Splits", len(split_ean_groups))
        with col2:
            st.metric("Avg Suppliers per Split", f"{avg_suppliers_per_split:.1f}")
        with col3:
            st.metric("Split Order Savings", f"€{total_split_savings:.2f}")

        # Show detailed split order breakdown
        split_details = []
        for ean, group in split_ean_groups.items():
            if len(group) <= 1:
                continue

            # Sort by rank
            sorted_group = sorted(group, key=lambda x: x.get("split_order_rank", 1))

            net_need = sorted_group[0].get(
                "original_net_need", sorted_group[0].get("net_need", 0)
            )
            total_allocated = sum(
                get_safe_allocated_quantity(opp) for opp in sorted_group
            )

            # Calculate blended cost
            total_cost = sum(
                opp.get("quote_price", 0) * get_safe_allocated_quantity(opp)
                for opp in sorted_group
            )
            blended_price = total_cost / total_allocated if total_allocated > 0 else 0

            # Best supplier cost (hypothetical)
            best_supplier = sorted_group[0]
            best_price = best_supplier.get("quote_price", 0)
            hypothetical_cost = best_price * net_need

            cost_difference = total_cost - hypothetical_cost
            cost_increase_pct = (
                (cost_difference / hypothetical_cost * 100)
                if hypothetical_cost > 0
                else 0
            )

            # Create supplier breakdown
            supplier_breakdown = []
            for opp in sorted_group:
                rank = opp.get("split_order_rank", 1)
                supplier = opp.get("supplier", "Unknown")
                qty = get_safe_allocated_quantity(opp)
                price = opp.get("quote_price", 0)
                percentage = (qty / net_need * 100) if net_need > 0 else 0
                supplier_breakdown.append(
                    f"#{rank}: {supplier} ({qty} units, {percentage:.1f}%, €{price:.2f})"
                )

            split_details.append(
                {
                    "EAN": ean,
                    "Product": sorted_group[0].get("product_name", "Unknown")[:40],
                    "Net Need": net_need,
                    "Total Allocated": total_allocated,
                    "Fulfillment": (
                        f"{total_allocated/net_need*100:.1f}%"
                        if net_need > 0
                        else "100%"
                    ),
                    "Suppliers Used": len(sorted_group),
                    "Blended Price": f"€{blended_price:.2f}",
                    "Best Price": f"€{best_price:.2f}",
                    "Cost vs Best": f"+€{cost_difference:.2f} (+{cost_increase_pct:.1f}%)",
                    "Supplier Breakdown": " | ".join(supplier_breakdown),
                }
            )

        if split_details:
            st.dataframe(pd.DataFrame(split_details), use_container_width=True)

        # Cost analysis summary
        total_cost_difference = sum(
            float(detail["Cost vs Best"].split("+€")[1].split(" ")[0])
            for detail in split_details
            if "+€" in detail["Cost vs Best"]
        )

        if total_cost_difference > 0:
            st.warning(
                f"💰 **Split Order Premium**: €{total_cost_difference:.2f} additional cost due to quantity constraints"
            )
            st.info(
                "💡 **Note**: This premium ensures full order fulfillment when best suppliers lack sufficient quantity"
            )


def show_enhanced_allocation_supplier_analysis(engine):
    """Show supplier performance analysis with allocation insights - UPDATED: Only Priority 1 and 2"""

    opportunities = engine.opportunities if hasattr(engine, "opportunities") else []

    if not opportunities:
        return

    with st.expander("🏢 Supplier Performance Analysis (with Smart Allocation)"):
        try:
            supplier_stats = engine.analyze_supplier_performance()
        except:
            # Fallback calculation
            supplier_stats = {}
            for opp in opportunities:
                supplier = opp.get("supplier", "Unknown")
                if supplier not in supplier_stats:
                    supplier_stats[supplier] = {
                        "opportunity_count": 0,
                        "total_potential_savings": 0,
                        "split_order_count": 0,
                        "single_order_count": 0,
                        "primary_supplier_count": 0,
                        "priority_1_count": 0,
                        "priority_2_count": 0,
                        # REMOVED: priority_3_count
                    }

                supplier_stats[supplier]["opportunity_count"] += 1
                supplier_stats[supplier]["total_potential_savings"] += opp.get(
                    "total_savings", 0
                )

                # Count by priority - only 1 and 2
                if opp.get("priority") == 1:
                    supplier_stats[supplier]["priority_1_count"] += 1
                elif opp.get("priority") == 2:
                    supplier_stats[supplier]["priority_2_count"] += 1

                if opp.get("is_split_order", False):
                    supplier_stats[supplier]["split_order_count"] += 1
                    if opp.get("split_order_rank", 1) == 1:
                        supplier_stats[supplier]["primary_supplier_count"] += 1
                else:
                    supplier_stats[supplier]["single_order_count"] += 1

        if supplier_stats:
            supplier_data = []
            for supplier, stats in supplier_stats.items():
                avg_savings = (
                    stats["total_potential_savings"] / stats["opportunity_count"]
                    if stats["opportunity_count"] > 0
                    else 0
                )

                # Calculate allocation metrics
                split_percentage = (
                    stats.get("split_order_count", 0) / stats["opportunity_count"] * 100
                    if stats["opportunity_count"] > 0
                    else 0
                )

                primary_rate = (
                    stats.get("primary_supplier_count", 0)
                    / stats.get("split_order_count", 1)
                    * 100
                    if stats.get("split_order_count", 0) > 0
                    else 0
                )

                # Calculate quality metrics - only Priority 1 and 2
                total_qualified = stats.get("priority_1_count", 0) + stats.get(
                    "priority_2_count", 0
                )
                priority_1_rate = (
                    (stats.get("priority_1_count", 0) / total_qualified * 100)
                    if total_qualified > 0
                    else 0
                )

                # NEW: Calculate cost comparison metrics for this supplier
                total_cost_stock_avg = 0
                total_cost_supplier = 0
                items_with_stock_avg = 0

                # Find all opportunities for this supplier to calculate costs
                supplier_opportunities = [
                    opp for opp in opportunities if opp.get("supplier") == supplier
                ]

                for opp in supplier_opportunities:
                    allocated_qty = get_safe_allocated_quantity(opp)
                    quote_price = opp.get("quote_price", 0)
                    total_cost_supplier += quote_price * allocated_qty

                    # Get stock average price from price breakdown
                    price_breakdown = opp.get("price_breakdown", {})
                    stock_avg_price = price_breakdown.get("stock_avg_price")

                    if stock_avg_price is not None and stock_avg_price > 0:
                        total_cost_stock_avg += stock_avg_price * allocated_qty
                        items_with_stock_avg += 1

                # Calculate savings percentage
                savings_percentage = 0
                if total_cost_stock_avg > 0:
                    savings_percentage = (
                        (total_cost_stock_avg - total_cost_supplier)
                        / total_cost_stock_avg
                    ) * 100

                supplier_data.append(
                    {
                        "Supplier": supplier,
                        "Total Opportunities": stats["opportunity_count"],
                        "🔥 Priority 1": stats.get("priority_1_count", 0),
                        "⭐ Priority 2": stats.get("priority_2_count", 0),
                        "P1 Rate": f"{priority_1_rate:.1f}%",
                        "🔄 Split Orders": stats.get("split_order_count", 0),
                        "📦 Single Orders": stats.get("single_order_count", 0),
                        "Split %": f"{split_percentage:.1f}%",
                        "Primary in Splits": f"{primary_rate:.1f}%",
                        "Total Savings": f"€{stats['total_potential_savings']:.2f}",
                        "Avg Savings": f"€{avg_savings:.2f}",
                        "Total Cost Stock Avg": (
                            f"€{total_cost_stock_avg:.2f}"
                            if items_with_stock_avg > 0
                            else "N/A"
                        ),
                        "Total Cost Supplier": f"€{total_cost_supplier:.2f}",
                        "% Savings": (
                            f"{savings_percentage:.1f}%"
                            if items_with_stock_avg > 0
                            else "N/A"
                        ),
                    }
                )

            # Sort by total potential savings
            supplier_df = pd.DataFrame(supplier_data)
            supplier_df["_sort_value"] = [
                stats["total_potential_savings"] for stats in supplier_stats.values()
            ]
            supplier_df = supplier_df.sort_values("_sort_value", ascending=False).drop(
                "_sort_value", axis=1
            )

            st.dataframe(supplier_df, use_container_width=True)

            # Show allocation insights
            st.write("**📊 Quality & Allocation Insights:**")

            total_split_opportunities = sum(
                stats.get("split_order_count", 0) for stats in supplier_stats.values()
            )
            total_single_opportunities = sum(
                stats.get("single_order_count", 0) for stats in supplier_stats.values()
            )
            total_priority_1 = sum(
                stats.get("priority_1_count", 0) for stats in supplier_stats.values()
            )
            total_priority_2 = sum(
                stats.get("priority_2_count", 0) for stats in supplier_stats.values()
            )

            if total_split_opportunities > 0:
                st.write(
                    f"• **{total_split_opportunities}** opportunities required split orders due to quantity constraints"
                )
                st.write(
                    f"• **{total_single_opportunities}** opportunities fulfilled by single suppliers"
                )

                # Find most reliable suppliers (high primary rate in splits)
                reliable_suppliers = [
                    (
                        supplier,
                        stats.get("primary_supplier_count", 0)
                        / stats.get("split_order_count", 1)
                        * 100,
                    )
                    for supplier, stats in supplier_stats.items()
                    if stats.get("split_order_count", 0) > 0
                ]
                reliable_suppliers.sort(key=lambda x: x[1], reverse=True)

                if reliable_suppliers:
                    best_supplier, best_rate = reliable_suppliers[0]
                    st.write(
                        f"• **Most reliable primary supplier**: {best_supplier} ({best_rate:.1f}% primary rate in splits)"
                    )

            # Quality insights
            st.write(
                f"• **Quality Distribution**: {total_priority_1} Priority 1 (maximum impact), {total_priority_2} Priority 2 (operational improvement)"
            )

            if supplier_stats:
                # Find highest quality supplier
                quality_suppliers = [
                    (supplier, stats.get("priority_1_count", 0))
                    for supplier, stats in supplier_stats.items()
                    if stats.get("priority_1_count", 0) > 0
                ]
                quality_suppliers.sort(key=lambda x: x[1], reverse=True)

                if quality_suppliers:
                    best_quality_supplier, p1_count = quality_suppliers[0]
                    st.write(
                        f"• **Highest impact supplier**: {best_quality_supplier} ({p1_count} Priority 1 opportunities)"
                    )


def show_allocation_export_options(opportunities: List[Dict]):
    """Show export options with allocation data and stock average prices"""

    if not opportunities:
        return

    st.subheader("💾 Export Smart Allocation Results")
    st.caption(
        "📤 **Enhanced Export**: Includes allocation details, stock average pricing, quality focus (Priority 1 & 2 only), and sales period data"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Download Smart Allocation CSV"):
            # Helper function to clean text of icons and symbols
            def clean_text(text):
                if pd.isna(text) or text == "":
                    return ""
                text_str = str(text)
                # Remove emojis and special symbols, keep only alphanumeric, spaces, and basic punctuation
                import re

                cleaned = re.sub(r"[^\w\s\-\.\(\)]", "", text_str)
                return cleaned.strip()

            # Helper function to format price with comma decimal separator
            def format_price_comma(price_value):
                if pd.isna(price_value) or price_value == "" or price_value == 0:
                    return ""
                try:
                    # Convert to float and format with comma as decimal separator
                    float_val = float(price_value)
                    return f"{float_val:.2f}".replace(".", ",")
                except (ValueError, TypeError):
                    return ""

            # Helper function to ensure numeric values
            def ensure_numeric(value):
                if pd.isna(value) or value == "":
                    return 0
                try:
                    return float(value) if isinstance(value, (int, float, str)) else 0
                except (ValueError, TypeError):
                    return 0

            # Prepare cleaned data for export
            export_data = []
            for opp in opportunities:
                # Clean and format all data
                export_row = {
                    # Basic identification
                    "Priority": ensure_numeric(opp.get("priority", 0)),
                    "Priority_Label": clean_text(opp.get("priority_label", "")),
                    "EAN": clean_text(opp.get("ean", "")),
                    "Product_Name": clean_text(opp.get("product_name", "")),
                    "Brand": clean_text(opp.get("brand", "")),
                    # Quantities (as integers)
                    "Current_Stock": int(ensure_numeric(opp.get("current_stock", 0))),
                    "Sales_90d": int(ensure_numeric(opp.get("sales90d", 0))),
                    "Sales_180d": int(ensure_numeric(opp.get("sales180d", 0))),
                    "Sales_365d": int(ensure_numeric(opp.get("sales365d", 0))),
                    "Net_Need": int(ensure_numeric(opp.get("net_need", 0))),
                    "Calculated_With_Period": clean_text(
                        opp.get("calculated_with_period", "sales90d")
                    ),
                    "Allocated_Quantity": int(
                        ensure_numeric(get_safe_allocated_quantity(opp))
                    ),
                    "Supplier_Quantity_Available": (
                        int(ensure_numeric(opp.get("supplier_quantity", 0)))
                        if opp.get("supplier_quantity") is not None
                        else ""
                    ),
                    # Prices (with comma decimal separator, no € symbol)
                    "Baseline_Price": format_price_comma(opp.get("baseline_price", 0)),
                    "Quote_Price": format_price_comma(opp.get("quote_price", 0)),
                    "Savings_Per_Unit": format_price_comma(
                        opp.get("savings_per_unit", 0)
                    ),
                    "Total_Savings": format_price_comma(opp.get("total_savings", 0)),
                    "Total_Cost": format_price_comma(
                        opp.get("quote_price", 0) * get_safe_allocated_quantity(opp)
                    ),
                    # Stock average price comparison
                    "Stock_Avg_Price": format_price_comma(
                        opp.get("price_breakdown", {}).get("stock_avg_price", "")
                    ),
                    "BestBuy_Price": format_price_comma(
                        opp.get("price_breakdown", {}).get("best_buy_price", "")
                    ),
                    "Supplier_Price": format_price_comma(
                        opp.get("price_breakdown", {}).get("supplier_price", "")
                    ),
                    # Boolean flags (True/False without icons)
                    "Beats_Stock_Avg": opp.get("price_breakdown", {}).get(
                        "beats_stock_avg", False
                    ),
                    "Beats_BestBuy": opp.get("price_breakdown", {}).get(
                        "beats_best_buy", False
                    ),
                    "Beats_Supplier": opp.get("price_breakdown", {}).get(
                        "beats_supplier", False
                    ),
                    "Is_Bestseller": opp.get("is_bestseller", False),
                    "Is_Split_Order": opp.get("is_split_order", False),
                    # Text fields (cleaned)
                    "Supplier": clean_text(opp.get("supplier", "")),
                    "Urgency_Score": clean_text(opp.get("urgency_score", "")),
                    "Allocation_Type": clean_text(
                        opp.get("allocation_type", "single_supplier")
                    ),
                    # Numeric percentages and rankings
                    "Split_Order_Rank": (
                        int(ensure_numeric(opp.get("split_order_rank", 0)))
                        if opp.get("split_order_rank")
                        else ""
                    ),
                    "Split_Percentage": ensure_numeric(
                        opp.get("split_percentage", 100.0)
                    ),
                    "Remaining_Need_After": int(
                        ensure_numeric(opp.get("remaining_need_after", 0))
                    ),
                    # Stock average savings analysis
                    "Stock_Avg_Savings_Per_Unit": format_price_comma(
                        opp.get("price_breakdown", {}).get("stock_avg_price", 0)
                        - opp.get("quote_price", 0)
                        if opp.get("price_breakdown", {}).get("stock_avg_price")
                        else ""
                    ),
                    "Stock_Avg_Total_Savings": format_price_comma(
                        (
                            opp.get("price_breakdown", {}).get("stock_avg_price", 0)
                            - opp.get("quote_price", 0)
                        )
                        * get_safe_allocated_quantity(opp)
                        if opp.get("price_breakdown", {}).get("stock_avg_price")
                        else ""
                    ),
                    # Metadata
                    "Analysis_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Quality_Focus_Mode": "Priority_1_and_2_Only",
                }

                export_data.append(export_row)

            # Create DataFrame and export with comma delimiter
            df = pd.DataFrame(export_data)

            # Ensure numeric columns are properly typed
            numeric_columns = [
                "Priority",
                "Current_Stock",
                "Sales_90d",
                "Sales_180d",
                "Sales_365d",
                "Net_Need",
                "Allocated_Quantity",
                "Split_Order_Rank",
                "Split_Percentage",
                "Remaining_Need_After",
            ]

            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Create CSV with semicolon delimiter and proper formatting
            csv_content = df.to_csv(index=False, sep=";", decimal=",", encoding="utf-8")
            filename = f"smart_allocation_with_sales_periods_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            st.download_button(
                label="💾 Download Smart Allocation CSV",
                data=csv_content,
                file_name=filename,
                mime="text/csv",
                help="Clean CSV with semicolon delimiters, no icons, prices with comma decimals, includes all sales periods",
            )

    with col2:
        if st.button("🏢 Download All Supplier Order CSVs"):
            # Group opportunities by supplier
            supplier_groups = {}
            for opp in opportunities:
                supplier = opp.get("supplier", "Unknown_Supplier")
                if supplier not in supplier_groups:
                    supplier_groups[supplier] = []
                supplier_groups[supplier].append(opp)

            if supplier_groups:
                import zipfile
                import io

                # Create a ZIP file containing all supplier CSVs
                zip_buffer = io.BytesIO()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    total_products = 0
                    total_units = 0

                    for (
                        supplier_name,
                        supplier_opportunities,
                    ) in supplier_groups.items():
                        # Prepare CSV data for this supplier
                        supplier_csv_data = []
                        supplier_quantity = 0

                        for opp in supplier_opportunities:
                            allocated_qty = get_safe_allocated_quantity(opp)
                            supplier_quantity += allocated_qty
                            total_units += allocated_qty
                            total_products += 1

                            supplier_csv_data.append(
                                {
                                    "EAN": opp.get("ean", ""),
                                    "Product_Name": opp.get("product_name", ""),
                                    "Brand": opp.get("brand", ""),
                                    "Quantity": allocated_qty,
                                    "Quote_Price": f"{opp.get('quote_price', 0):.2f}".replace(
                                        ".", ","
                                    ),
                                    "Total_Cost": f"{opp.get('quote_price', 0) * allocated_qty:.2f}".replace(
                                        ".", ","
                                    ),
                                    "Sales_Period_Used": opp.get(
                                        "calculated_with_period", "sales90d"
                                    ),
                                    "Net_Need_Calculated": opp.get("net_need", 0),
                                }
                            )

                        # Create CSV content
                        supplier_df = pd.DataFrame(supplier_csv_data)
                        csv_content = supplier_df.to_csv(
                            index=False, sep=";", decimal=","
                        )

                        # Clean supplier name for filename
                        safe_supplier_name = "".join(
                            c
                            for c in supplier_name
                            if c.isalnum() or c in (" ", "-", "_")
                        ).rstrip()
                        safe_supplier_name = safe_supplier_name.replace(" ", "_")

                        filename = (
                            f"supplier_order_{safe_supplier_name}_{timestamp}.csv"
                        )

                        # Add CSV to ZIP file
                        zip_file.writestr(filename, csv_content)

                # Prepare ZIP file for download
                zip_buffer.seek(0)
                zip_filename = f"all_supplier_orders_with_sales_data_{timestamp}.zip"

                st.success(
                    f"✅ Created ZIP file with {len(supplier_groups)} supplier CSV files"
                )
                st.info(
                    f"📦 **Package Contents**: {total_products} total products, {total_units} total units across {len(supplier_groups)} suppliers"
                )

                # Download button for the ZIP file
                st.download_button(
                    label=f"📦 Download ZIP ({len(supplier_groups)} suppliers)",
                    data=zip_buffer.getvalue(),
                    file_name=zip_filename,
                    mime="application/zip",
                    help=f"Downloads {len(supplier_groups)} CSV files with sales period data in a single ZIP archive",
                )

                # Show breakdown of what's included
                with st.expander("📋 ZIP File Contents Preview"):
                    zip_contents = []
                    for (
                        supplier_name,
                        supplier_opportunities,
                    ) in supplier_groups.items():
                        safe_supplier_name = "".join(
                            c
                            for c in supplier_name
                            if c.isalnum() or c in (" ", "-", "_")
                        ).rstrip()
                        safe_supplier_name = safe_supplier_name.replace(" ", "_")

                        supplier_quantity = sum(
                            get_safe_allocated_quantity(opp)
                            for opp in supplier_opportunities
                        )

                        zip_contents.append(
                            {
                                "File Name": f"supplier_order_{safe_supplier_name}_{timestamp}.csv",
                                "Supplier": supplier_name,
                                "Products": len(supplier_opportunities),
                                "Total Units": supplier_quantity,
                                "Columns": "EAN, Product_Name, Brand, Quantity, Quote_Price, Total_Cost, Sales_Period_Used, Net_Need_Calculated",
                            }
                        )

                    st.dataframe(pd.DataFrame(zip_contents), use_container_width=True)

                st.info(
                    f"💡 **Tip**: Extract the ZIP file to get individual CSV files for each supplier, ready for ordering"
                )
            else:
                st.info("No supplier data available for export")

    # Show export summary with quality focus info and sales period data
    split_count = sum(1 for opp in opportunities if opp.get("is_split_order", False))
    single_count = len(opportunities) - split_count
    stock_avg_available = sum(
        1
        for opp in opportunities
        if opp.get("price_breakdown", {}).get("stock_avg_price") is not None
    )

    priority_1_count = sum(1 for opp in opportunities if opp.get("priority") == 1)
    priority_2_count = sum(1 for opp in opportunities if opp.get("priority") == 2)

    # Show supplier breakdown in summary
    supplier_count = len(set(opp.get("supplier", "") for opp in opportunities))

    # Show which sales period was used
    sales_period_used = "sales90d"  # Default
    if opportunities:
        sales_period_used = opportunities[0].get("calculated_with_period", "sales90d")

    period_display = {
        "sales90d": "90 days",
        "sales180d": "180 days",
        "sales365d": "365 days",
    }.get(sales_period_used, sales_period_used)

    st.info(
        f"📊 **Enhanced Export**: {len(opportunities)} high-quality opportunities "
        f"({priority_1_count} Priority 1, {priority_2_count} Priority 2) • "
        f"{split_count} split orders, {single_count} single orders • "
        f"Stock average pricing for {stock_avg_available} products • "
        f"**{supplier_count} suppliers** with allocated products • "
        f"**Sales Period**: {period_display} data used for Net Need calculation • "
        f"**Quality Focus**: Priority 3 eliminated for strategic focus"
    )
