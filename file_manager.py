"""
file_manager.py - Fixed File Management System with Internal Data Auto-Loading
Complete solution for automatic loading of all file types
"""

import os
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict
from datetime import datetime


class AutoFileManager:
    """Enhanced file manager for automatic loading of all file types"""

    def __init__(self, base_data_folder: str = "data"):
        self.base_folder = Path(base_data_folder)
        self.suppliers_folder = self.base_folder / "suppliers"
        self.internal_data_folder = self.base_folder / "internal_data"
        self.order_files_folder = self.base_folder / "order_files"

        # Create folders if they don't exist
        for folder in [
            self.suppliers_folder,
            self.internal_data_folder,
            self.order_files_folder,
        ]:
            folder.mkdir(parents=True, exist_ok=True)

    def scan_folder(self, folder_path: Path) -> List[Dict]:
        """Scan folder for supported files"""
        if not folder_path.exists():
            return []

        files_info = []
        supported_extensions = [".csv", ".xlsx", ".xls"]

        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    file_stats = file_path.stat()
                    files_info.append(
                        {
                            "name": file_path.name,
                            "path": str(file_path),
                            "size_mb": file_stats.st_size / (1024 * 1024),
                            "modified": datetime.fromtimestamp(file_stats.st_mtime),
                        }
                    )
                except:
                    continue

        return sorted(files_info, key=lambda x: x["modified"], reverse=True)

    def get_supplier_files(self) -> List[Dict]:
        return self.scan_folder(self.suppliers_folder)

    def get_internal_data_files(self) -> List[Dict]:
        return self.scan_folder(self.internal_data_folder)

    def get_order_files(self) -> List[Dict]:
        return self.scan_folder(self.order_files_folder)

    def get_folder_status(self) -> Dict:
        return {
            "suppliers": {
                "path": str(self.suppliers_folder),
                "files_count": len(self.get_supplier_files()),
                "files": self.get_supplier_files(),
            },
            "internal_data": {
                "path": str(self.internal_data_folder),
                "files_count": len(self.get_internal_data_files()),
                "files": self.get_internal_data_files(),
            },
            "order_files": {
                "path": str(self.order_files_folder),
                "files_count": len(self.get_order_files()),
                "files": self.get_order_files(),
            },
        }


def show_folder_management_ui():
    """Show folder management in sidebar"""
    st.sidebar.header("📁 Data Folders")

    # Initialize file manager
    if "file_manager" not in st.session_state:
        st.session_state.file_manager = AutoFileManager()

    file_manager = st.session_state.file_manager

    # Refresh button
    if st.sidebar.button("🔄 Refresh Folders"):
        st.session_state.file_manager = AutoFileManager()
        st.rerun()

    # Show folder status
    status = file_manager.get_folder_status()

    # Quick status
    st.sidebar.write(f"🏢 Suppliers: {status['suppliers']['files_count']} files")
    st.sidebar.write(f"📊 Internal: {status['internal_data']['files_count']} files")
    st.sidebar.write(f"🛒 Orders: {status['order_files']['files_count']} files")

    return file_manager


def auto_file_processing_tab(groq_api_key, max_file_size, auto_detect_fields):
    """Enhanced auto file processing tab with internal data support"""

    st.header("📁 Automatic File Processing")
    st.info(
        "🤖 **Enhanced Mode**: Process supplier catalogs AND internal data automatically"
    )

    # Get file manager
    file_manager = st.session_state.get("file_manager", AutoFileManager())

    # Create tabs for different file types
    sub_tab1, sub_tab2 = st.tabs(["🏢 Supplier Catalogs", "📊 Internal Data"])

    with sub_tab1:
        process_supplier_files(file_manager, groq_api_key, auto_detect_fields)

    with sub_tab2:
        process_internal_data_files(file_manager)


def process_supplier_files(file_manager, groq_api_key, auto_detect_fields):
    """Process supplier catalog files"""

    st.subheader("🏢 Supplier Catalog Processing")

    supplier_files = file_manager.get_supplier_files()

    if not supplier_files:
        st.warning("⚠️ No supplier files found")
        st.info(f"📂 Add CSV/Excel files to: `{file_manager.suppliers_folder}`")

        if st.button("📁 Create Supplier Folder", key="create_supplier_folder"):
            file_manager.suppliers_folder.mkdir(parents=True, exist_ok=True)
            st.success("✅ Supplier folder created!")
        return

    # Display files
    st.write("**📋 Available Supplier Files:**")
    df = pd.DataFrame(supplier_files)
    df["Size"] = df["size_mb"].apply(lambda x: f"{x:.1f} MB")
    df["Modified"] = df["modified"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(df[["name", "Size", "Modified"]], use_container_width=True)

    # File selection
    selected_files = st.multiselect(
        "Select supplier files to process:",
        options=[f["name"] for f in supplier_files],
        default=[f["name"] for f in supplier_files],
        key="supplier_file_selector",
    )

    if selected_files and st.button(
        "🚀 Process Supplier Files", type="primary", key="process_supplier_btn"
    ):
        file_paths = [f["path"] for f in supplier_files if f["name"] in selected_files]

        # Process files using existing logic
        from processor import ProcurementProcessor

        processor = ProcurementProcessor(groq_api_key)

        all_results = []
        all_products = []

        progress_bar = st.progress(0)

        for i, file_path in enumerate(file_paths):
            progress_bar.progress((i + 1) / len(file_paths))

            try:
                supplier_name = Path(file_path).stem
                result = processor.process_file(file_path, supplier_name=supplier_name)
                all_results.append(result)
                all_products.extend(result.products)

                if result.success:
                    ean_count = sum(1 for p in result.products if p.ean_code)
                    st.success(
                        f"✅ {Path(file_path).name}: {result.total_products} products ({ean_count} EANs)"
                    )
                else:
                    st.error(f"❌ {Path(file_path).name}: Failed")

            except Exception as e:
                st.error(f"❌ Error processing {Path(file_path).name}: {str(e)}")

        # Store results
        st.session_state.processed_data = all_products
        if "processing_results" not in st.session_state:
            st.session_state.processing_results = []
        st.session_state.processing_results.extend(all_results)

        # Show summary
        total_products = len(all_products)
        total_eans = sum(1 for p in all_products if p.ean_code)

        st.success(
            f"🎉 **Supplier Processing Complete!** {total_products} products, {total_eans} EAN codes"
        )


def process_internal_data_files(file_manager):
    """Process internal data files for opportunity analysis"""

    st.subheader("📊 Internal Data Processing")

    internal_files = file_manager.get_internal_data_files()

    if not internal_files:
        st.warning("⚠️ No internal data files found")
        st.info(f"📂 Add CSV files to: `{file_manager.internal_data_folder}`")

        with st.expander("📖 Internal Data Requirements"):
            st.write(
                """
            **Required CSV columns:**
            - `EAN`: Product barcodes (required)
            - `stockAvgPrice`: Average stock price
            - `supplierPrice`: Current supplier price  
            - `bestbuyPrice(12M)`: Best price in last 12 months
            - `stock`: Current inventory
            - `sales90d`: Sales in last 90 days
            - `itemDescriptionEN`: Product descriptions
            - `bestSeller`: Bestseller ranking (1-4)
            """
            )

        if st.button("📁 Create Internal Data Folder", key="create_internal_folder"):
            file_manager.internal_data_folder.mkdir(parents=True, exist_ok=True)
            st.success("✅ Internal data folder created!")
        return

    # Display files
    st.write("**📋 Available Internal Data Files:**")
    df = pd.DataFrame(internal_files)
    df["Size"] = df["size_mb"].apply(lambda x: f"{x:.1f} MB")
    df["Modified"] = df["modified"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(df[["name", "Size", "Modified"]], use_container_width=True)

    # File selection
    selected_internal_file = st.selectbox(
        "Select internal data file:",
        options=[f["name"] for f in internal_files],
        key="internal_file_selector",
    )

    if selected_internal_file and st.button(
        "📊 Load Internal Data", type="primary", key="load_internal_btn"
    ):
        file_path = next(
            f["path"] for f in internal_files if f["name"] == selected_internal_file
        )

        try:
            # Load file
            df = pd.read_csv(file_path)
            st.success(
                f"✅ Loaded {len(df)} internal products from {selected_internal_file}"
            )

            # Basic validation
            required_cols = ["EAN"]
            price_cols = ["stockAvgPrice", "supplierPrice", "bestbuyPrice(12M)"]

            missing_required = [col for col in required_cols if col not in df.columns]
            available_price_cols = [col for col in price_cols if col in df.columns]

            if missing_required:
                st.error(f"❌ Missing required columns: {missing_required}")
                return

            if not available_price_cols:
                st.error(
                    f"❌ No price columns found. Need at least one of: {price_cols}"
                )
                return

            # Show summary
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Products", len(df))
            with col2:
                valid_eans = df["EAN"].notna().sum()
                st.metric("Valid EANs", valid_eans)
            with col3:
                if "stock" in df.columns:
                    total_stock = df["stock"].fillna(0).sum()
                    st.metric("Total Stock", f"{total_stock:,.0f}")
            with col4:
                if "sales90d" in df.columns:
                    total_sales = df["sales90d"].fillna(0).sum()
                    st.metric("Sales 90d", f"{total_sales:,.0f}")

            # Load into opportunity engine
            try:
                # Initialize opportunity engine
                if "opportunity_engine" not in st.session_state:
                    try:
                        from analysis.opportunity_engine import SimpleOpportunityEngine

                        st.session_state.opportunity_engine = SimpleOpportunityEngine()
                    except ImportError:
                        st.warning(
                            "⚠️ Opportunity engine not available. Basic loading only."
                        )
                        st.session_state.internal_data_loaded = df
                        return

                engine = st.session_state.opportunity_engine

                # Create CSV buffer for the engine
                from io import StringIO

                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                # Load data
                if engine.load_internal_data(csv_buffer):
                    st.success(
                        f"🎯 **Internal data loaded successfully!** Ready for opportunity analysis."
                    )

                    # Show data quality metrics
                    st.subheader("📊 Data Quality Summary")

                    # Calculate statistics
                    with_stock_avg = sum(
                        1
                        for p in engine.internal_data
                        if p.get("stock_avg_price") and p.get("stock_avg_price") > 0
                    )
                    with_supplier_price = sum(
                        1
                        for p in engine.internal_data
                        if p.get("supplier_price") and p.get("supplier_price") > 0
                    )
                    with_best_buy = sum(
                        1
                        for p in engine.internal_data
                        if p.get("best_buy_price") and p.get("best_buy_price") > 0
                    )
                    bestsellers = sum(
                        1 for p in engine.internal_data if p.get("is_bestseller", False)
                    )

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Stock Avg Price",
                            with_stock_avg,
                            f"{with_stock_avg/len(engine.internal_data)*100:.1f}%",
                        )
                    with col2:
                        st.metric(
                            "Supplier Price",
                            with_supplier_price,
                            f"{with_supplier_price/len(engine.internal_data)*100:.1f}%",
                        )
                    with col3:
                        st.metric(
                            "Best Buy Price",
                            with_best_buy,
                            f"{with_best_buy/len(engine.internal_data)*100:.1f}%",
                        )
                    with col4:
                        st.metric("Bestsellers", bestsellers)

                    # Show sample data
                    with st.expander("📋 Sample Data Preview"):
                        sample_data = []
                        for product in engine.internal_data[:5]:
                            sample_data.append(
                                {
                                    "EAN": product["ean"],
                                    "Description": product.get("description", "N/A")[
                                        :50
                                    ],
                                    "Stock": product.get("stock", 0),
                                    "Sales 90d": product.get("sales90d", 0),
                                    "Stock Avg Price": (
                                        f"€{product.get('stock_avg_price', 0):.2f}"
                                        if product.get("stock_avg_price")
                                        else "N/A"
                                    ),
                                    "Bestseller": (
                                        "⭐"
                                        * max(
                                            0, 5 - (product.get("bestseller_rank", 5))
                                        )
                                        if product.get("bestseller_rank")
                                        else ""
                                    ),
                                }
                            )
                        st.dataframe(
                            pd.DataFrame(sample_data), use_container_width=True
                        )

                    # Check if we can run opportunity analysis
                    if st.session_state.get("processed_data"):
                        st.info(
                            "💡 **Ready for opportunity analysis!** Go to the 'Opportunities' tab to find cost-saving opportunities."
                        )
                    else:
                        st.warning(
                            "⚠️ Process supplier data first to enable opportunity analysis."
                        )

                else:
                    st.error("❌ Failed to load internal data into opportunity engine")

            except Exception as e:
                st.error(f"❌ Error processing internal data: {str(e)}")
                # Store basic data anyway
                st.session_state.internal_data_loaded = df

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")


def process_order_files(file_manager):
    """Process order files"""

    st.subheader("🛒 Order File Processing")

    order_files = file_manager.get_order_files()

    if not order_files:
        st.warning("⚠️ No order files found")
        st.info(f"📂 Add CSV files to: `{file_manager.order_files_folder}`")

        with st.expander("📖 Order File Format"):
            st.write(
                """
            **Required CSV format:**
            ```
            EAN;Quantity;Description
            3433422404397;10;Product A
            1234567890123;5;Product B
            ```
            
            **Or with metadata (header row 3):**
            ```
            Order 123456;
            2024-01-15;
            EAN;Quantity;Description
            3433422404397;10;Product A
            ```
            """
            )

        if st.button("📁 Create Order Files Folder", key="create_order_folder"):
            file_manager.order_files_folder.mkdir(parents=True, exist_ok=True)
            st.success("✅ Order files folder created!")
        return

    # Display files
    st.write("**📋 Available Order Files:**")
    df = pd.DataFrame(order_files)
    df["Size"] = df["size_mb"].apply(lambda x: f"{x:.1f} MB")
    df["Modified"] = df["modified"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(df[["name", "Size", "Modified"]], use_container_width=True)

    # Basic file processing
    selected_order_files = st.multiselect(
        "Select order files:",
        options=[f["name"] for f in order_files],
        key="order_file_selector",
    )

    if selected_order_files:
        header_row = st.number_input(
            "Header row number:",
            min_value=1,
            max_value=10,
            value=3,
            help="Which row contains the column headers",
            key="order_header_row",
        )

        if st.button("📊 Preview Order Files", key="preview_order_btn"):
            for file_name in selected_order_files[:1]:  # Preview first file only
                file_path = next(
                    f["path"] for f in order_files if f["name"] == file_name
                )

                try:
                    # Simple CSV reading with specified header row
                    df = pd.read_csv(
                        file_path, sep=";", skiprows=header_row - 1, encoding="cp1252"
                    )

                    st.write(f"**Preview of {file_name}:**")
                    st.write(f"Columns found: {list(df.columns)}")
                    st.write(f"Rows: {len(df)}")
                    st.dataframe(df.head(3), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error reading {file_name}: {str(e)}")

        st.info("💡 For full order optimization, use the 'Order Optimization' tab.")


def auto_opportunities_tab():
    """Enhanced opportunities tab with better integration"""

    st.header("🎯 Procurement Opportunities")
    st.info(
        "💡 **Find cost-saving opportunities** by comparing internal data with supplier quotes"
    )

    # Check prerequisites
    if not st.session_state.get("processed_data"):
        st.warning(
            "⚠️ **Step 1**: Process supplier catalogs first using the 'Auto File Loading' tab"
        )
        return

    # Check if internal data is loaded
    if (
        "opportunity_engine" not in st.session_state
        or not hasattr(st.session_state.opportunity_engine, "internal_data")
        or not st.session_state.opportunity_engine.internal_data
    ):
        st.warning(
            "⚠️ **Step 2**: Load internal data using the 'Auto File Loading' tab → 'Internal Data' section"
        )

        # Show quick status
        file_manager = st.session_state.get("file_manager", AutoFileManager())
        internal_files = file_manager.get_internal_data_files()

        if internal_files:
            st.info(
                f"📂 Found {len(internal_files)} internal data files. Load them in the 'Auto File Loading' tab."
            )
        else:
            st.info(
                f"📂 Add internal data CSV files to: `{file_manager.internal_data_folder}`"
            )

        return

    # Run opportunity analysis
    st.success("✅ **Ready for opportunity analysis!**")

    engine = st.session_state.opportunity_engine

    # Show loaded data summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Internal Products", len(engine.internal_data))
    with col2:
        st.metric("Supplier Products", len(st.session_state.processed_data))
    with col3:
        supplier_eans = sum(1 for p in st.session_state.processed_data if p.ean_code)
        st.metric("Supplier EANs", supplier_eans)

    # Run analysis
    if st.button("🔍 Find Opportunities", type="primary"):
        with st.spinner("Analyzing opportunities..."):
            try:
                opportunities = engine.find_opportunities(
                    st.session_state.processed_data
                )

                if opportunities:
                    st.success(f"🎯 **Found {len(opportunities)} opportunities!**")

                    # Show summary
                    total_savings = sum(opp["total_savings"] for opp in opportunities)
                    high_urgency = sum(
                        1 for opp in opportunities if opp["urgency_score"] == "High"
                    )
                    bestseller_opps = sum(
                        1 for opp in opportunities if opp["is_bestseller"]
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Potential Savings", f"€{total_savings:.2f}")
                    with col2:
                        st.metric("High Urgency Items", high_urgency)
                    with col3:
                        st.metric("Bestseller Opportunities", bestseller_opps)

                    # Show top opportunities
                    st.subheader("🔝 Top Opportunities")
                    opp_data = []
                    for opp in opportunities[:10]:
                        opp_data.append(
                            {
                                "EAN": opp["ean"],
                                "Product": (
                                    opp["product_name"][:50] + "..."
                                    if len(opp["product_name"]) > 50
                                    else opp["product_name"]
                                ),
                                "Current Stock": opp["current_stock"],
                                "Net Need": opp["net_need"],
                                "Your Best Price": f"€{opp['baseline_price']:.2f}",
                                "Quote Price": f"€{opp['quote_price']:.2f}",
                                "Savings/Unit": f"€{opp['savings_per_unit']:.2f}",
                                "Total Savings": f"€{opp['total_savings']:.2f}",
                                "Supplier": opp["supplier"],
                                "Urgency": opp["urgency_score"],
                            }
                        )

                    st.dataframe(pd.DataFrame(opp_data), use_container_width=True)

                    # Export option
                    if st.button("💾 Download Opportunities CSV"):
                        csv_content = pd.DataFrame(opp_data).to_csv(index=False)
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv_content,
                            file_name=f"opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )

                else:
                    st.info("ℹ️ No opportunities found. This could mean:")
                    st.write(
                        "• Supplier prices are not better than your historical prices"
                    )
                    st.write(
                        "• No EAN matches between internal data and supplier quotes"
                    )
                    st.write("• Current stock covers sales needs")

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")


def auto_order_optimization_tab():
    """Basic auto order optimization placeholder"""
    st.header("🛒 Order Optimization (Auto)")
    st.info("🚀 Use the 'Order Optimization' tab for full functionality")

    file_manager = st.session_state.get("file_manager", AutoFileManager())
    order_files = file_manager.get_order_files()

    if order_files:
        st.write(f"📋 Found {len(order_files)} order files:")
        for f in order_files:
            st.write(f"• {f['name']}")
        st.info("💡 Process these files in the main 'Order Optimization' tab")
    else:
        st.info(f"📂 Add order files to: `{file_manager.order_files_folder}`")
