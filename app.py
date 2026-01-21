"""
Simplified Streamlit Application - Phase 1: Manual File Processing
Tab structure reorganized to use manual upload instead of automatic processing
UPDATED: Hide automatic file processing, use manual upload for all files
"""

import streamlit as st
import pandas as pd
import json
import csv
import time
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
from auth import SimpleAuth, require_auth

# Load environment variables
load_dotenv()

# Import existing modules (keep all imports for future use)
from file_manager import (
    AutoFileManager,
    show_folder_management_ui,
    auto_file_processing_tab,
    auto_opportunities_tab,
)

# Import existing modules
from tabs.opportunities_tab import opportunities_tab
from tabs.marketing_campaign_tab import marketing_campaign_tab
from models import FieldMapping, ProductData
from processor import ProcurementProcessor
from file_processor import FileProcessor
from field_detector import AIFieldDetector
from data_normalizer import DataNormalizer
from price_analyzer import PriceAnalyzer
from order_optimizer import OrderOptimizer
from utils import (
    setup_logging,
    export_to_csv,
    export_to_json,
    create_processing_report,
    validate_groq_api_key,
    format_file_size,
    get_processing_stats,
)

# Configure page
st.set_page_config(
    page_title="AI Procurement Data Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",  # Start with sidebar collapsed
)

# Setup logging
setup_logging("INFO")

# Initialize session state
if "processed_data" not in st.session_state:
    st.session_state.processed_data = []
if "processing_results" not in st.session_state:
    st.session_state.processing_results = []
if "price_analysis" not in st.session_state:
    st.session_state.price_analysis = None
if "order_optimizer" not in st.session_state:
    st.session_state.order_optimizer = OrderOptimizer()
if "buying_lists" not in st.session_state:
    st.session_state.buying_lists = []
# Add enhanced optimizer to session state
if "enhanced_order_optimizer" not in st.session_state:
    from enhanced_order_optimizer import EnhancedOrderOptimizer

    st.session_state.enhanced_order_optimizer = EnhancedOrderOptimizer()

# Add campaign configuration to session state
if "campaign_discount_threshold" not in st.session_state:
    st.session_state.campaign_discount_threshold = 15.0

# =============================================================================
# PHASE 2: FEATURE FLAGS CONFIGURATION
# =============================================================================

# Feature flag to control automatic vs manual file processing
ENABLE_AUTO_FILE_PROCESSING = False  # Set to True to re-enable automatic features
ENABLE_FOLDER_MANAGEMENT = False  # Set to True to re-enable folder management
ENABLE_AUTO_INTERNAL_DATA = False  # Set to True to re-enable auto internal data loading

# Development flag for testing automatic features
DEBUG_SHOW_AUTO_FEATURES = (
    os.getenv("DEBUG_SHOW_AUTO_FEATURES", "false").lower() == "true"
)

# =============================================================================
# MODIFIED IMPORTS SECTION - Add conditional imports
# =============================================================================

# Always import for preservation, but use conditionally
from file_manager import (
    AutoFileManager,
    show_folder_management_ui,
    auto_file_processing_tab,
    auto_opportunities_tab,
)

# =============================================================================
# MODIFIED MAIN FUNCTION - Add feature flag controls
# =============================================================================


def main():
    """Main application function with Phase 3 simplified sidebar"""

    # Initialize authentication
    auth = SimpleAuth()

    # Require authentication before proceeding
    require_auth(auth)

    # =============================================================================
    # PHASE 3: SINGLE SIMPLIFIED SIDEBAR (removes all duplicate sidebar blocks)
    # =============================================================================

    with st.sidebar:
        # Authentication status (essential)
        if auth.is_authenticated():
            st.success("🔓 Authenticated")
            if st.button("🚪 Logout", key="logout_btn"):
                auth.logout()
                st.rerun()

        # Reset all data button
        st.divider()
        st.caption("⚠️ Data Management")
        if st.button("🔄 Reset All Data", key="reset_all_data_btn", type="secondary"):
            # Clear all session state data
            keys_to_reset = [
                "processed_data",
                "processing_results",
                "price_analysis",
                "buying_lists",
                "manual_order_files",
                "opportunity_engine",
                "file_manager"
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    if key == "order_optimizer":
                        st.session_state.order_optimizer = OrderOptimizer()
                    elif key == "enhanced_order_optimizer":
                        from enhanced_order_optimizer import EnhancedOrderOptimizer
                        st.session_state.enhanced_order_optimizer = EnhancedOrderOptimizer()
                    else:
                        del st.session_state[key]

            # Reinitialize essential session state
            st.session_state.processed_data = []
            st.session_state.processing_results = []
            st.session_state.price_analysis = None
            st.session_state.order_optimizer = OrderOptimizer()
            st.session_state.buying_lists = []
            from enhanced_order_optimizer import EnhancedOrderOptimizer
            st.session_state.enhanced_order_optimizer = EnhancedOrderOptimizer()

            st.success("✅ All data has been reset!")
            time.sleep(1)
            st.rerun()

        # PHASE 3: Debug mode (minimal, only in development)
        if DEBUG_SHOW_AUTO_FEATURES:
            st.warning("🚧 **DEBUG MODE**")
            show_simplified_debug_info()

        # PHASE 3: Folder management ONLY in debug mode (hidden in production)
        if ENABLE_FOLDER_MANAGEMENT or DEBUG_SHOW_AUTO_FEATURES:
            with st.expander("📁 Advanced (Debug Only)", expanded=False):
                st.caption("Automatic folder features (debug mode)")
                file_manager = show_folder_management_ui()
                st.session_state.file_manager = file_manager
        else:
            # PHASE 3: Clean removal - no folder management in manual mode
            if "file_manager" in st.session_state:
                del st.session_state.file_manager

        # PHASE 3: Essential configuration only
        st.header("⚙️ Setup")

        # Simplified API Key configuration
        groq_api_key = get_simplified_api_key_config()

        # Manual mode guidance
        show_manual_mode_sidebar_info()

    # =============================================================================
    # PHASE 3: SIMPLIFIED HEADER & CAPTION
    # =============================================================================

    st.title("🤖 AI Procurement Data Intelligence")

    # Phase 3: Clean caption based on mode
    if ENABLE_AUTO_FILE_PROCESSING or DEBUG_SHOW_AUTO_FEATURES:
        st.caption("**Debug Mode** - Both automatic and manual features available")
    else:
        st.caption(
            "**Manual Upload Mode** - Upload files step by step for full control"
        )

    # =============================================================================
    # PHASE 3: SIMPLIFIED TAB STRUCTURE
    # =============================================================================

    if ENABLE_AUTO_FILE_PROCESSING or DEBUG_SHOW_AUTO_FEATURES:
        # Debug mode - keep all 4 tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🤖 Auto File Loading",
                "📤 Manual Upload",
                "🎯 Opportunities",
                "🛒 Order Optimization",
            ]
        )

        with tab1:
            auto_file_processing_tab(groq_api_key, 50, True)
        with tab2:
            manual_file_processing_tab(groq_api_key)
        with tab3:
            if ENABLE_AUTO_INTERNAL_DATA:
                auto_opportunities_tab(groq_api_key)
            else:
                opportunities_tab(groq_api_key, api_key_valid=bool(groq_api_key))
        with tab4:
            order_optimization_tab()
    else:
        # PHASE 3: Manual mode - clean 5-tab structure
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📤 File Processing", "🎯 Opportunities", "📣 Marketing Campaigns", "🛒 Order Optimization", "🔄 CSV Delimiter Converter"]
        )

        with tab1:
            manual_file_processing_tab(groq_api_key)
        with tab2:
            opportunities_tab(groq_api_key, api_key_valid=bool(groq_api_key))
        with tab3:
            marketing_campaign_tab(groq_api_key, api_key_valid=bool(groq_api_key))
        with tab4:
            order_optimization_tab()
        with tab5:
            csv_delimiter_converter_tab()


def create_order_file_uploader(context="default"):
    """
    Create file uploader with unique key based on context

    Args:
        context: Unique identifier for the upload context

    Returns:
        tuple: (uploaded_files, header_row, additional_info)
    """

    # Create unique keys based on context
    uploader_key = f"order_upload_{context}"
    header_key = f"header_row_{context}"
    preview_key = f"preview_files_{context}"

    st.subheader("🛒 Order File Processing")
    st.info(
        "📤 Upload your order/buying list files for optimization. These will be used in the Order Optimization tab."
    )

    # File upload section
    uploaded_order_files = st.file_uploader(
        "Upload Order Files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload CSV files containing your orders or buying lists with EAN codes and quantities.",
        key=uploader_key,
    )

    if not uploaded_order_files:
        st.warning("⚠️ Upload order files to preview and prepare them for optimization")

        # Show format guidance
        with st.expander("📖 Order File Format Guide"):
            st.write("**Expected CSV Format:**")
            st.write("• **EAN codes**: Product barcodes or references")
            st.write("• **Quantities**: Number of units needed")
            st.write("• **Descriptions**: Product names (optional)")

            st.write("\n**Simple Format Example:**")
            simple_example = {
                "EAN": ["3433422404397", "1234567890123"],
                "Quantity": [10, 5],
                "Description": ["Anti-Age Serum", "Moisturizer"],
            }
            st.dataframe(pd.DataFrame(simple_example), width='stretch')

            st.write("\n**Complex Format (with metadata):**")
            st.code(
                """Order 123456;
                    2024-01-15;
                    EAN;Quantity;Description
                    3433422404397;10;Anti-Age Serum
                    1234567890123;5;Moisturizer"""
            )

        return None, 1, {"success": False, "message": "No files uploaded"}

    # Header row configuration
    header_row = st.number_input(
        "Header Row Number",
        min_value=1,
        max_value=10,
        value=1,
        help="Which row contains the column headers (1 for simple files, 3+ for files with metadata at the top)",
        key=header_key,
    )

    # Preview files
    if st.button("📋 Preview Order Files", key=preview_key):
        preview_order_files(uploaded_order_files, header_row)

    # Store files for later use
    if uploaded_order_files:
        # Store in session state for use in Order Optimization tab
        if "manual_order_files" not in st.session_state:
            st.session_state.manual_order_files = []

        st.session_state.manual_order_files = [
            {
                "file": uploaded_file,
                "name": uploaded_file.name,
                "header_row": header_row,
            }
            for uploaded_file in uploaded_order_files
        ]

        st.success(
            f"✅ **{len(uploaded_order_files)} order files ready for optimization**"
        )
        st.info(
            "💡 **Next Step**: Go to the 'Order Optimization' tab to process these files and find the best supplier allocation!"
        )

        return (
            uploaded_order_files,
            header_row,
            {
                "success": True,
                "message": f"{len(uploaded_order_files)} files uploaded",
                "file_info": st.session_state.manual_order_files,
            },
        )

    return (
        uploaded_order_files,
        header_row,
        {"success": True, "message": "Files uploaded"},
    )


def get_simplified_api_key_config():
    """Phase 3: Simplified API key configuration for manual mode"""

    # Try to get API key from .env file first
    env_api_key = os.getenv("GROQ_API_KEY")

    # Session state for API key
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = env_api_key
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False

    # Show current status - clean and simple
    if env_api_key:
        if validate_groq_api_key(env_api_key):
            st.success("✅ AI Detection Ready")
            st.session_state.api_key_valid = True
            return env_api_key
        else:
            st.error("❌ Invalid API key in .env")

    # Manual API key input - streamlined
    with st.expander("🔧 API Key Setup", expanded=not env_api_key):
        st.write("**Optional: For AI-powered field detection**")

        manual_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enables smart field detection and mapping",
            placeholder="gsk_...",
            key="manual_api_key_input",
        )

        if manual_api_key:
            if validate_groq_api_key(manual_api_key):
                st.success("✅ API key validated")
                st.session_state.groq_api_key = manual_api_key
                st.session_state.api_key_valid = True
                return manual_api_key
            else:
                st.error("❌ Invalid API key format")

    # Fallback information
    if not env_api_key and not st.session_state.get("groq_api_key"):
        st.info("ℹ️ Using pattern-based field detection")

    return st.session_state.groq_api_key if st.session_state.api_key_valid else None


def show_manual_mode_sidebar_info():
    """Phase 3: Essential manual mode information in sidebar"""

    with st.expander("📋 Manual Workflow", expanded=False):
        st.write("**3-Step Process:**")
        st.write("1️⃣ **File Processing** - Upload supplier catalogs")
        st.write("2️⃣ **Opportunities** - Upload internal data & analyze")
        st.write("3️⃣ **Order Optimization** - Upload order files")

        st.write("**✅ Benefits:**")
        st.write("• Full control over uploads")
        st.write("• Review before processing")
        st.write("• No folder dependencies")
        st.write("• Traditional workflow")


def show_simplified_debug_info():
    """Phase 3: Minimal debug information for development"""

    with st.expander("🚧 Debug Status", expanded=False):
        st.write("**Feature Flags:**")
        st.write(f"• Auto Processing: {'ON' if ENABLE_AUTO_FILE_PROCESSING else 'OFF'}")
        st.write(f"• Folder Management: {'ON' if ENABLE_FOLDER_MANAGEMENT else 'OFF'}")
        st.write(
            f"• Auto Internal Data: {'ON' if ENABLE_AUTO_INTERNAL_DATA else 'OFF'}"
        )

        if not ENABLE_AUTO_FILE_PROCESSING:
            st.success("✅ Manual mode active")


def manual_file_processing_tab(groq_api_key):
    """
    Manual file processing tab - Supplier catalogs only
    Order files are handled in the Order Optimization tab
    """

    st.header("📤 Manual File Processing")

    # Show supplier catalog processing directly (no sub-tabs needed)
    manual_supplier_processing(groq_api_key)


def manual_supplier_processing(groq_api_key):
    """Manual supplier catalog processing"""

    st.subheader("🏢 Supplier Catalog Processing")
    st.info(
        "📤 Upload your supplier catalog files (CSV/Excel) to extract product data, EAN codes, and pricing information"
    )

    # File upload section
    uploaded_files = st.file_uploader(
        "Upload Supplier Catalog Files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload one or more supplier catalog files. Supports CSV, Excel (.xlsx), and legacy Excel (.xls) formats.",
        key="manual_supplier_upload",
    )

    if not uploaded_files:
        st.warning(
            "⚠️ Please upload at least one supplier catalog file to begin processing"
        )

        # Show format guidance
        with st.expander("📖 Supported File Formats & Data Structure"):
            st.write("**Supported File Formats:**")
            st.write(
                "• CSV files (.csv) - with various delimiters (semicolon, comma, tab)"
            )
            st.write("• Excel files (.xlsx, .xls) - modern and legacy formats")
            st.write("• Multiple encodings supported (UTF-8, CP1252, Latin-1)")

            st.write("\n**Expected Data Columns:**")
            st.write("The system will automatically detect columns containing:")
            st.write(
                "• **Product Codes**: EAN, Gencod, UPC, SKU, Reference, Code, etc."
            )
            st.write(
                "• **Product Names**: Description, Product Name, Designation, etc."
            )
            st.write("• **Prices**: Price, Unit Price, Cost, Prix, Preço, etc.")
            st.write("• **Quantities**: Quantity, Stock, Qty, Amount, etc.")

            st.write("\n**Example Data Structure:**")
            example_data = {
                "EAN": ["3433422404397", "1234567890123", "9876543210987"],
                "Product_Name": [
                    "Anti-Age Serum 30ml",
                    "Moisturizer 50ml",
                    "Vitamin C Serum",
                ],
                "Price": [29.99, 19.95, 39.50],
                "Stock": [50, 25, 30],
            }
            st.dataframe(pd.DataFrame(example_data), width='stretch')

        return

    # Processing configuration
    st.subheader("🔧 Processing Configuration")

    col1, col2 = st.columns(2)

    with col1:
        auto_detect_fields = st.checkbox(
            "🤖 Auto-detect Fields",
            value=True,
            help="Use AI to automatically detect and map data fields. Uncheck for manual field mapping.",
        )

        max_file_size = st.number_input(
            "Max File Size (MB)",
            min_value=1,
            max_value=100,
            value=50,
            help="Maximum file size allowed for upload",
        )

    with col2:
        supplier_name_option = st.radio(
            "Supplier Name",
            options=["Auto (from filename)", "Manual input"],
            help="How to determine the supplier name for each file",
        )

        if supplier_name_option == "Manual input":
            manual_supplier_name = st.text_input(
                "Supplier Name",
                placeholder="Enter supplier name (will be used for all uploaded files)",
                help="This name will be applied to all uploaded files",
            )
        else:
            manual_supplier_name = None

    # Process files button
    if st.button("🚀 Process Supplier Files", type="primary", key="manual_process_btn"):
        process_manual_supplier_files(
            uploaded_files,
            groq_api_key,
            auto_detect_fields,
            max_file_size,
            manual_supplier_name,
        )


def process_manual_supplier_files(
    uploaded_files,
    groq_api_key,
    auto_detect_fields,
    max_file_size,
    manual_supplier_name,
):
    """Process manually uploaded supplier files"""

    if not uploaded_files:
        st.error("❌ No files uploaded")
        return

    # Initialize processor
    processor = ProcurementProcessor(groq_api_key)

    all_results = []
    all_products = []

    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(
            f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})..."
        )

        try:
            # Validate file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > max_file_size:
                st.error(
                    f"❌ {uploaded_file.name}: File too large ({file_size_mb:.1f}MB > {max_file_size}MB)"
                )
                continue

            # Determine supplier name
            if manual_supplier_name:
                supplier_name = manual_supplier_name
            else:
                supplier_name = Path(uploaded_file.name).stem

            # Process the file
            with st.spinner(f"Processing {uploaded_file.name}..."):
                result = processor.process_uploaded_file(
                    uploaded_file,
                    supplier_name=supplier_name,
                    manual_mapping=(
                        None if auto_detect_fields else None
                    ),  # TODO: Add manual mapping UI
                )

            all_results.append(result)

            if result.success:
                all_products.extend(result.products)

                # Calculate stats
                ean_count = sum(1 for p in result.products if p.ean_code)
                supplier_count = sum(1 for p in result.products if p.supplier_code)

                st.success(
                    f"✅ **{uploaded_file.name}**: {result.total_products} products processed "
                    f"({ean_count} EANs, {supplier_count} supplier codes) in {result.processing_time:.1f}s"
                )

                # Show sample products
                if result.products:
                    with st.expander(f"📋 Sample Data from {uploaded_file.name}"):
                        sample_data = []
                        for product in result.products[:3]:
                            sample_data.append(
                                {
                                    "Product Code": product.ean_code
                                    or product.supplier_code
                                    or "N/A",
                                    "Product Name": product.product_name or "N/A",
                                    "Price": (
                                        f"€{product.price:.2f}"
                                        if product.price
                                        else "N/A"
                                    ),
                                    "Supplier": product.supplier or "N/A",
                                }
                            )
                        st.dataframe(
                            pd.DataFrame(sample_data), width='stretch'
                        )

            else:
                st.error(f"❌ **{uploaded_file.name}**: Processing failed")
                if result.errors:
                    for error in result.errors[:3]:  # Show first 3 errors
                        st.error(f"  • {error}")

        except Exception as e:
            st.error(f"❌ **{uploaded_file.name}**: Unexpected error - {str(e)}")
            continue

    # Final results
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")

    if all_products:
        # Store results in session state
        st.session_state.processed_data = all_products
        st.session_state.processing_results = all_results

        # Show final summary
        total_products = len(all_products)
        total_eans = sum(1 for p in all_products if p.ean_code)
        total_suppliers = len(set(p.supplier for p in all_products if p.supplier))

        st.success(f"🎉 **Processing Complete!**")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Files Processed",
                f"{len([r for r in all_results if r.success])}/{len(uploaded_files)}",
            )
        with col2:
            st.metric("Total Products", total_products)
        with col3:
            st.metric("EAN Codes", total_eans)
        with col4:
            st.metric("Suppliers", total_suppliers)

        # Next steps
        st.info(
            "✅ **Next Steps**: Go to the 'Opportunities' tab to upload internal data and find cost-saving opportunities!"
        )

    else:
        st.error(
            "❌ No products were successfully extracted from any files. Please check your file formats and try again."
        )


def manual_order_file_processing():
    """Manual order file processing - Now uses consolidated uploader"""
    uploaded_files, header_row, result = create_order_file_uploader("manual_tab")
    return result


def preview_order_files(uploaded_files, header_row):
    """Preview uploaded order files"""

    for uploaded_file in uploaded_files[:2]:  # Preview max 2 files
        try:
            # Read CSV with specified header row
            df = pd.read_csv(
                uploaded_file, sep=";", skiprows=header_row - 1, encoding="cp1252"
            )

            st.write(f"**📋 Preview: {uploaded_file.name}**")

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• **Columns found**: {list(df.columns)}")
                st.write(f"• **Total rows**: {len(df)}")
            with col2:
                st.write(f"• **Header row used**: {header_row}")
                st.write(f"• **Sample data rows**: {min(3, len(df))}")

            # Show sample data
            if len(df) > 0:
                st.dataframe(df.head(3), width='stretch')
            else:
                st.warning(f"⚠️ No data rows found in {uploaded_file.name}")

        except Exception as e:
            st.error(f"❌ **{uploaded_file.name}**: Could not preview - {str(e)}")
            st.info("💡 Try adjusting the header row number or check the file format")


def get_api_key_config_with_feature_flags():
    """Enhanced API key configuration with feature flag awareness"""

    # Try to get API key from .env file first
    env_api_key = os.getenv("GROQ_API_KEY")

    # Session state for API key
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = env_api_key
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False

    # Show current status - with mode awareness
    if env_api_key:
        if validate_groq_api_key(env_api_key):
            ai_status = "✅ AI Detection Active"
            if ENABLE_AUTO_FILE_PROCESSING:
                ai_status += " (Auto Mode)"
            else:
                ai_status += " (Manual Mode)"
            st.success(ai_status)
            st.session_state.api_key_valid = True
            return env_api_key
        else:
            st.error("❌ Invalid API key")

    # Manual API key input - simplified
    with st.expander("🔧 Manual API Key Setup"):
        manual_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key for AI field detection",
            placeholder="gsk_...",
        )

        if manual_api_key:
            if validate_groq_api_key(manual_api_key):
                success_msg = "✅ Manual API key valid"
                if ENABLE_AUTO_FILE_PROCESSING:
                    success_msg += " (Auto Mode Ready)"
                st.success(success_msg)
                st.session_state.groq_api_key = manual_api_key
                st.session_state.api_key_valid = True
                return manual_api_key
            else:
                st.error("❌ Invalid manual API key format")

    if not env_api_key and not st.session_state.get("groq_api_key"):
        fallback_msg = "ℹ️ Using pattern-based detection"
        if not ENABLE_AUTO_FILE_PROCESSING:
            fallback_msg += " (Manual Mode)"
        st.info(fallback_msg)

    return st.session_state.groq_api_key if st.session_state.api_key_valid else None


def load_internal_data_into_optimizer():
    """Load internal data from opportunities engine into order optimizer"""

    if "enhanced_order_optimizer" not in st.session_state:
        from enhanced_order_optimizer import EnhancedOrderOptimizer

        st.session_state.enhanced_order_optimizer = EnhancedOrderOptimizer()

    optimizer = st.session_state.enhanced_order_optimizer

    # Check if opportunities engine has internal data loaded
    if (
        hasattr(st.session_state, "opportunity_engine")
        and hasattr(st.session_state.opportunity_engine, "internal_data")
        and st.session_state.opportunity_engine.internal_data
    ):

        # Convert opportunities engine format to optimizer format
        internal_data = []
        for product in st.session_state.opportunity_engine.internal_data:
            internal_data.append(
                {
                    "ean": product["ean"],
                    "stock_avg_price": product.get("stock_avg_price"),
                    "supplier_price": product.get("supplier_price"),
                    "best_buy_price": product.get("best_buy_price"),
                    "brand": product.get("brand", ""),
                    "description": product.get("description", ""),
                }
            )

        optimizer.load_internal_data(internal_data)
        return True

    return False


def get_enhanced_optimization_results(optimizer) -> List[Dict]:
    """
    Create enhanced optimization results with additional pricing data
    """
    if not optimizer.optimization_results:
        return []

    enhanced_results = []

    # Get all supplier data for price comparison
    supplier_ean_lookup = {}
    if hasattr(optimizer, "supplier_data") and optimizer.supplier_data:
        for product in optimizer.supplier_data:
            if product.ean_code:
                ean = str(product.ean_code).strip()
                if ean not in supplier_ean_lookup:
                    supplier_ean_lookup[ean] = []
                supplier_ean_lookup[ean].append(product)

    # Get internal data if available (for stock average prices)
    internal_data_lookup = {}
    if (
        hasattr(st.session_state, "opportunity_engine")
        and st.session_state.opportunity_engine
    ):
        engine = st.session_state.opportunity_engine
        if hasattr(engine, "internal_data") and engine.internal_data:
            for product in engine.internal_data:
                ean = str(product.get("ean", "")).strip()
                if ean:
                    internal_data_lookup[ean] = product

    # Process each supplier's orders
    for supplier_name, supplier_orders in optimizer.optimization_results.get(
        "supplier_orders", {}
    ).items():
        for order_item in supplier_orders:
            ean = str(order_item.get("ean_code", "")).strip()

            # Get best price across all suppliers
            best_price = None
            supplier_quantity = None
            all_supplier_prices = []

            if ean in supplier_ean_lookup:
                supplier_products = supplier_ean_lookup[ean]
                prices = [p.price for p in supplier_products if p.price and p.price > 0]
                if prices:
                    best_price = min(prices)
                    all_supplier_prices = prices

                # Find quantity for current supplier
                current_supplier_product = next(
                    (p for p in supplier_products if p.supplier == supplier_name), None
                )
                if current_supplier_product and hasattr(
                    current_supplier_product, "quantity"
                ):
                    supplier_quantity = current_supplier_product.quantity

            # Get stock average price from internal data
            stock_avg_price = None
            brand = None
            if ean in internal_data_lookup:
                internal_product = internal_data_lookup[ean]
                stock_avg_price = internal_product.get("stock_avg_price")
                brand = internal_product.get("brand", "")

            # Get order details
            quantity_ordered = order_item.get("quantity", 0)
            allocated_quantity = order_item.get(
                "quantity", 0
            )  # In current implementation, this is the same
            quote_price = order_item.get("unit_price", 0)
            total_cost = order_item.get("total_price", 0)

            # Calculate savings (prefer stock avg price, fallback to best price)
            reference_price = stock_avg_price if stock_avg_price else best_price
            savings_per_unit = 0
            total_savings = 0

            if reference_price and quote_price and reference_price > quote_price:
                savings_per_unit = reference_price - quote_price
                total_savings = savings_per_unit * allocated_quantity

            # Determine allocation type
            ean_orders_count = sum(
                1
                for orders in optimizer.optimization_results.get(
                    "supplier_orders", {}
                ).values()
                for item in orders
                if str(item.get("ean_code", "")).strip() == ean
            )
            allocation_type = "🔄 Split" if ean_orders_count > 1 else "📦 Single"

            enhanced_item = {
                "ean": ean,
                "product_name": order_item.get("product_name", "Unknown Product"),
                "brand": brand or "Unknown",
                "quantity_ordered": quantity_ordered,
                "best_price": best_price,
                "stock_avg_price": stock_avg_price,
                "quote_price": quote_price,
                "total_cost": total_cost,
                "savings_per_unit": savings_per_unit,
                "total_savings": total_savings,
                "allocated_quantity": allocated_quantity,
                "supplier_quantity": supplier_quantity,
                "allocation_type": allocation_type,
                "supplier": supplier_name,
                "reference_price_source": (
                    "Stock Avg"
                    if stock_avg_price
                    else "Best Price" if best_price else "None"
                ),
                "all_supplier_prices": all_supplier_prices,
                "original_reference": order_item.get("original_reference", ""),
            }

            enhanced_results.append(enhanced_item)

    return enhanced_results


def show_enhanced_table_guide():
    """Show explanation of the enhanced table columns"""
    with st.expander("📖 Enhanced Table Column Guide"):
        st.write("**Enhanced Order Analysis Columns:**")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Product Information:**")
            st.write("• **EAN**: Product barcode")
            st.write("• **Product**: Product name/description")
            st.write("• **Brand**: Product brand (from internal data)")
            st.write("")
            st.write("**Quantities:**")
            st.write("• **Qty Ordered**: Original quantity in your order")
            st.write("• **Allocated Qty**: Quantity allocated to this supplier")
            st.write("• **Supplier Qty**: Available quantity from supplier")
            st.write("  - ✅ = Sufficient quantity")
            st.write("  - ⚠️ = Limited quantity")
            st.write("")
            st.write("**Allocation:**")
            st.write("• **📦 Single**: Fulfilled by one supplier")
            st.write("• **🔄 Split**: Split across multiple suppliers")

        with col2:
            st.write("**Pricing Analysis:**")
            st.write("• **Best Price**: Lowest price across all suppliers")
            st.write("• **Stock Avg Price**: Your average stock price")
            st.write("• **Quote Price**: Selected supplier's price")
            st.write("• **Total Cost**: Quote Price × Allocated Quantity")
            st.write("")
            st.write("**Savings Calculation:**")
            st.write("• **Savings/Unit**: Reference Price - Quote Price")
            st.write("• **Total Savings**: Savings/Unit × Allocated Quantity")
            st.write("• **Ref Price Source**: Which price used for savings:")
            st.write("  - Stock Avg (preferred)")
            st.write("  - Best Price (fallback)")
            st.write("  - None (no reference available)")
            st.write("")
            st.write("**Indicators:**")
            st.write("• ✅ = Positive/Good status")
            st.write("• ⚠️ = Warning/Limited")
            st.write("• N/A = Data not available")


def show_enhanced_order_optimization_table(optimizer):
    """
    Display the enhanced order optimization results table
    """
    if not optimizer.optimization_results:
        st.info("No optimization results available. Run optimization first.")
        return

    # Get enhanced results
    enhanced_results = get_enhanced_optimization_results(optimizer)

    if not enhanced_results:
        st.warning("No enhanced results to display.")
        return

    st.subheader("📊 Enhanced Order Analysis Table")

    # Calculate summary metrics
    total_items = len(enhanced_results)
    total_cost = sum(item["total_cost"] for item in enhanced_results)
    total_savings = sum(item["total_savings"] for item in enhanced_results)
    split_orders = sum(
        1 for item in enhanced_results if "Split" in item["allocation_type"]
    )
    single_orders = total_items - split_orders

    # Display summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Items", total_items)
    with col2:
        st.metric("Total Cost", f"€{total_cost:.2f}")
    with col3:
        st.metric("Total Savings", f"€{total_savings:.2f}")
    with col4:
        st.metric("🔄 Split Orders", split_orders)
    with col5:
        st.metric("📦 Single Orders", single_orders)

    # Prepare data for display
    table_data = []
    for i, item in enumerate(enhanced_results):
        # Format prices safely
        def format_price(price):
            if price is not None and price > 0:
                return f"€{price:.2f}"
            return "N/A"

        # Format savings with color indicators
        savings_display = (
            f"€{item['savings_per_unit']:.2f}"
            if item["savings_per_unit"] > 0
            else "€0.00"
        )
        if item["savings_per_unit"] > 0:
            savings_display += " ✅"

        # Truncate product name if too long
        product_name = item["product_name"]
        if len(product_name) > 35:
            product_name = product_name[:35] + "..."

        # Format supplier quantity with indicators
        supplier_qty_display = "Unknown"
        if item["supplier_quantity"] is not None:
            qty = int(item["supplier_quantity"])
            if qty >= item["allocated_quantity"]:
                supplier_qty_display = f"{qty} ✅"
            else:
                supplier_qty_display = f"{qty} ⚠️"

        table_data.append(
            {
                "Row": i + 1,
                "EAN": item["ean"],
                "Product": product_name,
                "Brand": item["brand"],
                "Qty Ordered": int(item["quantity_ordered"]),
                "Best Price": format_price(item["best_price"]),
                "Stock Avg Price": format_price(item["stock_avg_price"]),
                "Quote Price": format_price(item["quote_price"]),
                "Total Cost": format_price(item["total_cost"]),
                "Savings/Unit": savings_display,
                "Total Savings": f"€{item['total_savings']:.2f}",
                "Allocated Qty": int(item["allocated_quantity"]),
                "Supplier Qty": supplier_qty_display,
                "Allocation": item["allocation_type"],
                "Supplier": item["supplier"],
                "Ref Price Source": item["reference_price_source"],
            }
        )

    # Display the table
    df = pd.DataFrame(table_data)
    st.dataframe(df, width='stretch', height=600)

    # Show additional analysis
    with st.expander("📈 Pricing Analysis"):
        # Group by pricing data availability
        with_stock_avg = sum(1 for item in enhanced_results if item["stock_avg_price"])
        with_best_price = sum(1 for item in enhanced_results if item["best_price"])
        with_savings = sum(1 for item in enhanced_results if item["total_savings"] > 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Items with Stock Avg Price",
                with_stock_avg,
                f"{with_stock_avg/total_items*100:.1f}%",
            )
        with col2:
            st.metric(
                "Items with Best Price Data",
                with_best_price,
                f"{with_best_price/total_items*100:.1f}%",
            )
        with col3:
            st.metric(
                "Items with Savings",
                with_savings,
                f"{with_savings/total_items*100:.1f}%",
            )

        # Show price source breakdown
        st.write("**Reference Price Sources:**")
        source_counts = {}
        for item in enhanced_results:
            source = item["reference_price_source"]
            source_counts[source] = source_counts.get(source, 0) + 1

        for source, count in source_counts.items():
            percentage = count / total_items * 100
            st.write(f"• {source}: {count} items ({percentage:.1f}%)")

    # Export functionality
    with st.expander("💾 Export Enhanced Order Data"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Download Enhanced Order CSV"):
                # Prepare clean export data
                export_data = []
                for item in enhanced_results:
                    export_row = {
                        "EAN": item["ean"],
                        "Product_Name": item["product_name"],
                        "Brand": item["brand"],
                        "Quantity_Ordered": item["quantity_ordered"],
                        "Best_Price": item["best_price"] if item["best_price"] else "",
                        "Stock_Avg_Price": (
                            item["stock_avg_price"] if item["stock_avg_price"] else ""
                        ),
                        "Quote_Price": item["quote_price"],
                        "Total_Cost": item["total_cost"],
                        "Savings_Per_Unit": item["savings_per_unit"],
                        "Total_Savings": item["total_savings"],
                        "Allocated_Quantity": item["allocated_quantity"],
                        "Supplier_Quantity": (
                            item["supplier_quantity"]
                            if item["supplier_quantity"] is not None
                            else ""
                        ),
                        "Allocation_Type": item["allocation_type"]
                        .replace("🔄 ", "")
                        .replace("📦 ", ""),
                        "Supplier": item["supplier"],
                        "Reference_Price_Source": item["reference_price_source"],
                        "Original_Reference": item["original_reference"],
                        "Export_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    export_data.append(export_row)

                # Create CSV
                export_df = pd.DataFrame(export_data)
                csv_content = export_df.to_csv(index=False, sep=";", decimal=",")
                filename = f"enhanced_order_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                st.download_button(
                    label="💾 Download Enhanced CSV",
                    data=csv_content,
                    file_name=filename,
                    mime="text/csv",
                    help="Downloads enhanced order analysis with pricing data",
                )

        with col2:
            if st.button("📊 Download Summary Report"):
                # Create summary report
                report_lines = [
                    "ENHANCED ORDER OPTIMIZATION REPORT",
                    "=" * 50,
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "SUMMARY:",
                    f"- Total items: {total_items}",
                    f"- Total cost: €{total_cost:.2f}",
                    f"- Total savings: €{total_savings:.2f}",
                    (
                        f"- Average savings per item: €{total_savings/total_items:.2f}"
                        if total_items > 0
                        else "- Average savings per item: €0.00"
                    ),
                    f"- Split orders: {split_orders}",
                    f"- Single orders: {single_orders}",
                    "",
                    "PRICING DATA AVAILABILITY:",
                    f"- Items with stock average price: {with_stock_avg} ({with_stock_avg/total_items*100:.1f}%)",
                    f"- Items with best price data: {with_best_price} ({with_best_price/total_items*100:.1f}%)",
                    f"- Items with calculated savings: {with_savings} ({with_savings/total_items*100:.1f}%)",
                    "",
                    "SUPPLIER BREAKDOWN:",
                ]

                # Add supplier breakdown
                supplier_stats = {}
                for item in enhanced_results:
                    supplier = item["supplier"]
                    if supplier not in supplier_stats:
                        supplier_stats[supplier] = {
                            "items": 0,
                            "total_cost": 0,
                            "total_savings": 0,
                        }
                    supplier_stats[supplier]["items"] += 1
                    supplier_stats[supplier]["total_cost"] += item["total_cost"]
                    supplier_stats[supplier]["total_savings"] += item["total_savings"]

                for supplier, stats in supplier_stats.items():
                    report_lines.extend(
                        [
                            f"- {supplier}:",
                            f"  Items: {stats['items']}",
                            f"  Cost: €{stats['total_cost']:.2f}",
                            f"  Savings: €{stats['total_savings']:.2f}",
                            "",
                        ]
                    )

                report_content = "\n".join(report_lines)

                st.download_button(
                    label="📋 Download Report",
                    data=report_content,
                    file_name=f"order_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Downloads comprehensive order optimization report",
                )

        st.info(
            f"📊 **Enhanced Analysis**: {total_items} optimized items • {len(set(item['supplier'] for item in enhanced_results))} suppliers • €{total_savings:.2f} total savings potential"
        )


def csv_delimiter_converter_tab():
    """CSV Delimiter Converter - Convert comma to semicolon delimiter and format EAN codes"""

    st.header("🔄 CSV Delimiter Converter")
    st.write("Convert CSV delimiter from ',' to ';', format EAN codes to 13 digits, and convert price decimals (comma to dot)")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], key="csv_delimiter_upload")

    if uploaded_file is not None:
        st.success(f"Selected: {uploaded_file.name}")

        # Convert button
        if st.button("Convert & Download", type="primary"):
            try:
                # Read the CSV file
                try:
                    content = uploaded_file.read().decode('utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode('latin1')

                # Parse CSV - auto-detect delimiter
                from io import StringIO
                import csv

                # Detect delimiter (check first line)
                first_line = content.split('\n')[0] if content else ''
                detected_delimiter = ','  # default
                if ';' in first_line and first_line.count(';') > first_line.count(','):
                    detected_delimiter = ';'

                input_data = StringIO(content)
                rows = list(csv.reader(input_data, delimiter=detected_delimiter))

                if not rows:
                    st.error("The CSV file is empty")
                    return

                # Find EAN and price columns
                header_row = rows[0]
                ean_column_index = None
                price_column_indices = []

                for i, column_name in enumerate(header_row):
                    col_upper = column_name.upper()
                    if col_upper in ['EAN', 'EAN13', 'BARCODE', 'CODE_BARRES']:
                        ean_column_index = i
                    # Detect specific price columns: Quote_Price and Total_Cost
                    if col_upper in ['QUOTE_PRICE', 'TOTAL_COST']:
                        price_column_indices.append(i)

                # Process rows
                eans_formatted = 0
                prices_converted = 0
                output_data = StringIO()
                csv_writer = csv.writer(output_data, delimiter=';')

                for row_index, row in enumerate(rows):
                    # Format EAN if found
                    if ean_column_index is not None and row_index > 0 and len(row) > ean_column_index:
                        ean_value = row[ean_column_index]
                        if ean_value and str(ean_value).strip():
                            ean_str = str(ean_value).strip()
                            if ean_str.replace('.', '').replace(',', '').isdigit():
                                ean_digits = ''.join(filter(str.isdigit, ean_str))
                                if len(ean_digits) <= 13:
                                    formatted_ean = ean_digits.zfill(13)
                                    if ean_value != formatted_ean:
                                        eans_formatted += 1
                                    row[ean_column_index] = formatted_ean

                    # Clean and convert price values (remove quotes and convert comma to dot)
                    if row_index > 0:  # Skip header row
                        for price_idx in price_column_indices:
                            if len(row) > price_idx:
                                price_value = row[price_idx]
                                if price_value and str(price_value).strip():
                                    price_str = str(price_value).strip()
                                    original_price = price_str

                                    # Step 1: Remove quotes (both single and double)
                                    price_str = price_str.replace('"', '').replace("'", '')

                                    # Step 2: Convert comma to dot for decimal separator
                                    if ',' in price_str:
                                        price_str = price_str.replace(',', '.')

                                    # Step 3: Only replace semicolon if input delimiter is NOT semicolon
                                    if ';' in price_str and detected_delimiter != ';':
                                        price_str = price_str.replace(';', '.')

                                    # Validate it's a number
                                    if price_str != original_price:
                                        try:
                                            float(price_str)
                                            prices_converted += 1
                                            row[price_idx] = price_str
                                        except ValueError:
                                            # If not a valid number, keep original
                                            pass

                    csv_writer.writerow(row)

                # Get output
                output_content = output_data.getvalue()

                # Show success message
                success_msg = f"✅ File converted successfully! {len(rows)} rows processed"
                if prices_converted > 0:
                    success_msg += f" | {prices_converted} price(s) converted (comma to dot)"
                if eans_formatted > 0:
                    success_msg += f" | {eans_formatted} EAN(s) formatted"
                st.success(success_msg)

                # Download button
                output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_semicolon.csv"
                st.download_button(
                    label="📥 Download Converted CSV",
                    data=output_content,
                    file_name=output_filename,
                    mime="text/csv",
                    type="primary"
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to convert")


def order_optimization_tab():
    """Enhanced order optimization with conditional automatic features"""

    st.header("🛒 Enhanced Order Optimization & Smart Allocation")

    # PHASE 2: Mode-aware help text
    if ENABLE_AUTO_FILE_PROCESSING:
        st.info(
            "🧠 **Automatic Mode**: Smart multi-supplier allocation with folder-based order loading"
        )

    # Check prerequisites
    if not st.session_state.get("processed_data"):
        prereq_msg = "⚠️ **Step 1**: Load supplier data first"
        if ENABLE_AUTO_FILE_PROCESSING:
            prereq_msg += " in the 'Auto File Loading' or 'Manual Upload' tab"
        else:
            prereq_msg += " in the 'File Processing' tab"
        st.warning(prereq_msg)
        return

    # Use enhanced optimizer
    optimizer = st.session_state.enhanced_order_optimizer

    # Load supplier data if needed
    if not optimizer.supplier_data:
        optimizer.load_supplier_data(st.session_state.processed_data)
        st.success(
            f"✅ Loaded {len(st.session_state.processed_data)} supplier products"
        )

    # PHASE 2: Conditional internal data loading
    if ENABLE_AUTO_INTERNAL_DATA:
        load_internal_data_section_with_auto(optimizer)
    else:
        load_internal_data_section(optimizer)

    # PHASE 2: Conditional file loading
    if ENABLE_AUTO_FILE_PROCESSING and st.session_state.get("file_manager"):
        load_order_files_section_with_auto(optimizer)
    else:
        load_order_files_section(optimizer)

    # Show loaded data summary
    show_loaded_data_summary(optimizer)

    # Run optimization
    run_enhanced_optimization(optimizer)

    # Show results if available
    if hasattr(optimizer, "optimization_results") and optimizer.optimization_results:
        show_enhanced_optimization_results(optimizer)


def load_internal_data_section(optimizer):
    """Load internal data for stock average price comparison"""

    st.subheader("📊 Internal Data for Price Comparison")

    # Check if internal data already loaded from opportunities tab
    if (
        hasattr(st.session_state, "opportunity_engine")
        and hasattr(st.session_state.opportunity_engine, "internal_data")
        and st.session_state.opportunity_engine.internal_data
    ):

        # Use data from opportunities engine
        internal_data = st.session_state.opportunity_engine.internal_data
        optimizer.load_internal_data(internal_data)

        st.success(
            f"✅ **Auto-loaded internal data**: {len(internal_data)} products from opportunities engine"
        )

        # Show stock average price coverage
        with_stock_avg = sum(1 for p in internal_data if p.get("stock_avg_price"))
        coverage = (with_stock_avg / len(internal_data) * 100) if internal_data else 0

        st.info(
            f"📊 **Stock Average Price Coverage**: {with_stock_avg}/{len(internal_data)} products ({coverage:.1f}%)"
        )

    else:
        # Manual upload
        st.info(
            "💡 **Tip**: Load internal data in the Opportunities tab first for automatic integration"
        )

        uploaded_internal = st.file_uploader(
            "Upload Internal Product Data (CSV)",
            type=["csv"],
            help="CSV with columns: EAN, stockAvgPrice, supplierPrice, bestbuyPrice",
            key="internal_data_uploader",
        )

        if uploaded_internal:
            try:
                df = pd.read_csv(uploaded_internal)

                # Convert to expected format
                internal_data = []
                for _, row in df.iterrows():
                    if pd.notna(row.get("EAN")):
                        internal_data.append(
                            {
                                "ean": str(row["EAN"]).strip(),
                                "stock_avg_price": row.get("stockAvgPrice"),
                                "supplier_price": row.get("supplierPrice"),
                                "best_buy_price": row.get("bestbuyPrice(12M)"),
                                "brand": row.get("brand", ""),
                                "description": row.get("itemDescriptionEN", ""),
                            }
                        )

                optimizer.load_internal_data(internal_data)
                st.success(f"✅ Loaded {len(internal_data)} internal products")

            except Exception as e:
                st.error(f"❌ Error loading internal data: {str(e)}")


def load_order_files_section(optimizer):
    """Load order files section for optimization"""

    st.subheader("📋 Order Files")

    uploaded_orders = st.file_uploader(
        "Upload Order Files (CSV)",
        type=["csv"],
        accept_multiple_files=True,
        key="order_upload_optimization",
    )

    if uploaded_orders:
        header_row = st.number_input(
            "Header Row Number",
            min_value=1,
            max_value=10,
            value=3,
            help="Row containing column headers",
            key="header_row_optimization",
        )

        for uploaded_file in uploaded_orders:
            try:
                result = optimizer.add_buying_list(
                    uploaded_file, uploaded_file.name, header_row
                )
                if result["success"]:
                    st.success(
                        f"✅ {uploaded_file.name}: {result['total_items']} items"
                    )
                else:
                    st.error(f"❌ {uploaded_file.name}: {result['message']}")
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")


def load_internal_data_section_with_auto(optimizer):
    """Load internal data with automatic detection when enabled"""

    st.subheader("📊 Internal Data for Price Comparison")

    # PHASE 2: Check for auto-loaded internal data when automatic mode is enabled
    if (
        ENABLE_AUTO_INTERNAL_DATA
        and hasattr(st.session_state, "opportunity_engine")
        and hasattr(st.session_state.opportunity_engine, "internal_data")
        and st.session_state.opportunity_engine.internal_data
    ):

        # Use data from opportunities engine
        internal_data = st.session_state.opportunity_engine.internal_data
        optimizer.load_internal_data(internal_data)

        st.success(
            f"✅ **Auto-loaded internal data**: {len(internal_data)} products from opportunities engine"
        )

        # Show stock average price coverage
        with_stock_avg = sum(1 for p in internal_data if p.get("stock_avg_price"))
        coverage = (with_stock_avg / len(internal_data) * 100) if internal_data else 0

        st.info(
            f"📊 **Stock Average Price Coverage**: {with_stock_avg}/{len(internal_data)} products ({coverage:.1f}%)"
        )

    else:
        # Fall back to manual upload
        st.info(
            "💡 **Manual Upload**: Load internal data manually or enable automatic mode"
        )
        load_internal_data_section(optimizer)  # Use the manual version


def load_order_files_section_with_auto(optimizer):
    """Load order files with automatic folder detection when enabled"""

    st.subheader("📋 Order Files")

    # PHASE 2: Use automatic folder scanning when enabled
    file_manager = st.session_state.get("file_manager")

    if ENABLE_AUTO_FILE_PROCESSING and file_manager:
        # Auto-load from folder
        order_files = file_manager.get_order_files()

        if order_files:
            st.write(f"**📂 Found {len(order_files)} order files:**")
            files_df = pd.DataFrame(order_files)
            files_df["Size"] = files_df["size_mb"].apply(lambda x: f"{x:.1f} MB")
            files_df["Modified"] = files_df["modified"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                files_df[["name", "Size", "Modified"]], width='stretch'
            )

            col1, col2 = st.columns([3, 1])

            with col1:
                selected_files = st.multiselect(
                    "Select order files:",
                    options=[f["name"] for f in order_files],
                    key="order_files_selector",
                )

            with col2:
                header_row = st.number_input(
                    "Header Row",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="header_row_input",
                )

            # Process selected files
            if selected_files:
                process_selected_order_files(
                    optimizer, order_files, selected_files, header_row
                )

        else:
            st.warning("⚠️ No order files found in data folder")
            st.info(f"📂 Add CSV files to: `{file_manager.order_files_folder}`")

            # Fallback to manual upload
            st.write("**Manual Upload Fallback:**")
            load_order_files_section(optimizer)

    else:
        # Use manual upload (existing function)
        load_order_files_section(optimizer)


def show_feature_flag_status():
    """Debug function to show current feature flag status"""

    if DEBUG_SHOW_AUTO_FEATURES:
        with st.sidebar.expander("🚧 Debug: Feature Flags"):
            st.write("**Current Feature Flags:**")
            st.write(f"• Auto File Processing: {ENABLE_AUTO_FILE_PROCESSING}")
            st.write(f"• Folder Management: {ENABLE_FOLDER_MANAGEMENT}")
            st.write(f"• Auto Internal Data: {ENABLE_AUTO_INTERNAL_DATA}")
            st.write(f"• Debug Mode: {DEBUG_SHOW_AUTO_FEATURES}")


def get_current_mode_info():
    """Get information about current operational mode"""

    if ENABLE_AUTO_FILE_PROCESSING:
        return {
            "mode": "automatic",
            "features": ["folder_scanning", "auto_detection", "smart_loading"],
            "description": "Full automatic file processing with folder management",
        }
    else:
        return {
            "mode": "manual",
            "features": ["manual_upload", "file_selection", "traditional_workflow"],
            "description": "Manual file upload with traditional workflow",
        }


def process_selected_order_files(optimizer, order_files, selected_files, header_row):
    """Process selected order files from folder with duplicate prevention"""

    for file_name in selected_files:
        file_info = next((f for f in order_files if f["name"] == file_name), None)
        if not file_info:
            continue

        try:
            file_path = file_info["path"]
            with open(file_path, "rb") as f:
                file_content = f.read()

            import io

            file_like_object = io.BytesIO(file_content)
            file_like_object.name = file_name

            result = optimizer.add_buying_list(file_like_object, file_name, header_row)

            if result["success"]:
                if result.get("already_loaded", False):
                    st.info(
                        f"ℹ️ **{file_name}**: Already loaded ({result['total_items']} items)"
                    )
                else:
                    st.success(
                        f"✅ **{file_name}**: {result['total_items']} items loaded"
                    )
            else:
                st.error(f"❌ {file_name}: {result['message']}")

        except Exception as e:
            st.error(f"❌ Error processing {file_name}: {str(e)}")


def show_loaded_data_summary(optimizer):
    """Show summary of loaded data"""

    if not optimizer.buying_lists:
        return

    st.subheader("📋 Loaded Data Summary")

    # Buying lists summary
    list_data = []
    for bl in optimizer.buying_lists:
        list_data.append(
            {
                "File Name": bl["name"],
                "Items": bl["total_items"],
                "EAN Column": bl["structure"]["ean_column"] or "Not detected",
                "Quantity Column": bl["structure"]["quantity_column"] or "Default (1)",
                "Confidence": bl["structure"]["confidence"],
            }
        )

    st.dataframe(pd.DataFrame(list_data), width='stretch')

    # Data readiness check
    col1, col2, col3 = st.columns(3)

    with col1:
        total_items = sum(bl["total_items"] for bl in optimizer.buying_lists)
        st.metric("Total Order Items", total_items)

    with col2:
        st.metric("Supplier Products", len(optimizer.supplier_data))

    with col3:
        internal_products = len(optimizer.internal_data_lookup)
        st.metric("Internal Products", internal_products)

    # Readiness indicators
    if optimizer.supplier_data and optimizer.buying_lists:
        if optimizer.internal_data_lookup:
            st.success(
                "✅ **Ready for Smart Allocation**: All data loaded (supplier, orders, internal prices)"
            )
        else:
            st.warning(
                "⚠️ **Basic Allocation Only**: Missing internal data for price filtering"
            )
    else:
        st.info("ℹ️ Load supplier data and order files to begin")


def run_enhanced_optimization(optimizer):
    """Run the enhanced optimization with smart allocation"""

    if not optimizer.buying_lists:
        return

    st.subheader("🧠 Smart Multi-Supplier Optimization")

    if st.button("🚀 Run Smart Allocation", type="primary", key="run_smart_allocation"):
        with st.spinner("🧠 Running smart multi-supplier allocation..."):
            try:
                results = optimizer.process_buying_lists()

                if "error" in results:
                    st.error(f"❌ Optimization failed: {results['error']}")
                else:
                    st.success("✅ Smart allocation completed!")

                    # Show immediate summary
                    summary = optimizer.get_optimization_summary()

                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Items Processed", summary["total_items_requested"])

                    with col2:
                        st.metric(
                            "Successfully Matched",
                            summary["matched_items"],
                            f"{summary['match_rate']:.1f}%",
                        )

                    with col3:
                        st.metric("Suppliers Used", summary["suppliers_involved"])

                    with col4:
                        st.metric("Total Value", f"€{summary['total_order_value']:.2f}")

                    with col5:
                        st.metric(
                            "Smart Allocations", summary.get("total_allocations", 0)
                        )

                    # Allocation type breakdown
                    st.subheader("🔄 Allocation Breakdown")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "🔄 Split Orders", summary.get("split_orders_count", 0)
                        )

                    with col2:
                        st.metric(
                            "📦 Single Orders", summary.get("single_orders_count", 0)
                        )

                    with col3:
                        allocation_efficiency = summary.get("allocation_efficiency", 1)
                        st.metric(
                            "Allocation Efficiency", f"{allocation_efficiency:.2f}x"
                        )

                    if summary.get("split_orders_count", 0) > 0:
                        st.info(
                            f"🧠 **Smart Allocation Active**: {summary.get('split_orders_count', 0)} orders "
                            f"split across multiple suppliers for optimal fulfillment"
                        )

            except Exception as e:
                st.error(f"❌ Optimization error: {str(e)}")


def show_enhanced_optimization_results(optimizer):
    """Show enhanced optimization results with allocation details"""

    st.subheader("📊 Smart Allocation Results")

    # Get results summary
    summary = optimizer.get_optimization_summary()

    # Enhanced results table
    show_enhanced_results_table(optimizer, summary)

    # Supplier breakdown with allocation metrics
    show_enhanced_supplier_breakdown(optimizer, summary)

    # Export options
    show_enhanced_export_options(optimizer)


def show_enhanced_results_table(optimizer, summary):
    """Show enhanced results table with improved allocation details"""

    st.subheader("📋 Detailed Allocation Results")

    # Prepare enhanced table data
    table_data = []
    row_number = 1

    for supplier, items in optimizer.optimization_results["supplier_orders"].items():
        for item in items:
            # Get original quantity needed from CSV order
            original_quantity_needed = item.get(
                "original_quantity_needed", item["quantity"]
            )
            allocated_quantity = item[
                "quantity"
            ]  # This is what's allocated to this supplier

            # Format allocation type
            if item.get("is_split_order", False):
                allocation_type = f"🔄 Split #{item.get('split_rank', 1)}"
                split_info = f"({item.get('allocation_percentage', 100):.1f}% of {original_quantity_needed})"
            else:
                allocation_type = "📦 Single"
                split_info = "Full order"

            # Price comparison info
            price_comparison = item.get("price_comparison", {})
            stock_avg_price = price_comparison.get("stock_avg_price")

            # NEW: Get best buy price from internal data
            ean = item["ean_code"]
            best_buy_price = None
            if (
                hasattr(optimizer, "internal_data_lookup")
                and ean in optimizer.internal_data_lookup
            ):
                internal_product = optimizer.internal_data_lookup[ean]
                best_buy_price = internal_product.get("best_buy_price")

            # Format stock average and savings
            if stock_avg_price:
                stock_avg_display = f"€{stock_avg_price:.2f}"
                savings_vs_stock = stock_avg_price - item["unit_price"]
                savings_display = (
                    f"€{savings_vs_stock:.2f}" if savings_vs_stock > 0 else "€0.00"
                )
                # Calculate total savings = Allocated Quantity x Savings vs Stock
                total_savings = allocated_quantity * max(0, savings_vs_stock)
                total_savings_display = f"€{total_savings:.2f}"
            else:
                stock_avg_display = "N/A"
                savings_display = "N/A"
                total_savings_display = "N/A"

            # NEW: Format best buy price
            best_buy_display = f"€{best_buy_price:.2f}" if best_buy_price else "N/A"

            # ENHANCED: Improved quantity constraint info
            supplier_qty = item.get("supplier_quantity")
            if supplier_qty is not None:
                if supplier_qty >= original_quantity_needed:
                    # Supplier has enough quantity
                    qty_status = f"{supplier_qty} ✅"
                else:
                    # Supplier has insufficient quantity - show detailed breakdown
                    qty_status = f"{supplier_qty} ⚠️ (need {original_quantity_needed}) (using {allocated_quantity})"
            else:
                qty_status = "Unknown"

            table_data.append(
                {
                    "Row": row_number,
                    "EAN": item["ean_code"],
                    "Product": (
                        item["product_name"][:35] + "..."
                        if len(item["product_name"]) > 35
                        else item["product_name"]
                    ),
                    "Supplier": supplier,
                    "Quantity Needed": original_quantity_needed,  # NEW: Original quantity from CSV
                    "Allocated Qty": allocated_quantity,  # What's allocated to this supplier
                    "Quote Price": f"€{item['unit_price']:.2f}",
                    "Total Cost": f"€{item['total_price']:.2f}",
                    "Stock Avg Price": stock_avg_display,
                    "Best Buy Price": best_buy_display,  # NEW: Best buying price
                    "Savings vs Stock": savings_display,
                    "Total Savings": total_savings_display,  # NEW: Allocated Qty x Savings vs Stock
                    "Supplier Qty": qty_status,  # ENHANCED: More detailed format
                    "Allocation": allocation_type,
                    "Split Info": split_info if item.get("is_split_order") else "-",
                }
            )

            row_number += 1

    # Display table
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, width='stretch', height=500)

        # Enhanced table summary
        total_items = len(table_data)
        split_items = sum(1 for item in table_data if "Split" in item["Allocation"])
        total_savings_vs_stock = sum(
            float(item["Total Savings"].replace("€", "").replace("N/A", "0"))
            for item in table_data
        )

        # NEW: Calculate total costs at stock average price vs quote price
        total_cost_at_stock_avg = 0
        total_cost_at_quote = 0
        items_with_stock_avg = 0

        for item in table_data:
            allocated_qty = item["Allocated Qty"]
            quote_price_str = item["Quote Price"].replace("€", "")
            quote_price = float(quote_price_str)
            total_cost_at_quote += quote_price * allocated_qty

            stock_avg_str = item["Stock Avg Price"]
            if stock_avg_str != "N/A":
                stock_avg_price = float(stock_avg_str.replace("€", ""))
                total_cost_at_stock_avg += stock_avg_price * allocated_qty
                items_with_stock_avg += 1

        # Calculate savings percentage
        savings_percentage = 0
        if total_cost_at_stock_avg > 0:
            savings_percentage = (
                (total_cost_at_stock_avg - total_cost_at_quote)
                / total_cost_at_stock_avg
            ) * 100

        # NEW: Calculate quantity constraint metrics
        constrained_items = sum(1 for item in table_data if "⚠️" in item["Supplier Qty"])
        sufficient_items = sum(1 for item in table_data if "✅" in item["Supplier Qty"])

        # NEW: Calculate total quantities
        total_needed = sum(item["Quantity Needed"] for item in table_data)
        total_allocated = sum(item["Allocated Qty"] for item in table_data)

        # Display enhanced metrics - First row: Basic allocation info
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Line Items", total_items)
        with col2:
            st.metric("Split Allocations", split_items)
        with col3:
            st.metric(
                "Quantity Constrained",
                constrained_items,
                (
                    f"{constrained_items/total_items*100:.1f}%"
                    if total_items > 0
                    else "0%"
                ),
            )
        with col4:
            st.metric(
                "Sufficient Quantity",
                sufficient_items,
                f"{sufficient_items/total_items*100:.1f}%" if total_items > 0 else "0%",
            )
        with col5:
            st.metric("Total Savings vs Stock", f"€{total_savings_vs_stock:.2f}")

        # NEW: Second row - Cost comparison metrics
        st.subheader("💰 Cost Comparison Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Cost at Stock Avg Price",
                f"€{total_cost_at_stock_avg:.2f}",
                help=f"Total cost if buying {items_with_stock_avg}/{total_items} products at stock average prices",
            )
        with col2:
            st.metric(
                "Cost at Quote Price",
                f"€{total_cost_at_quote:.2f}",
                help="Total cost if buying all products at quoted prices",
            )
        with col3:
            savings_delta = (
                f"-{savings_percentage:.1f}%"
                if savings_percentage > 0
                else f"+{abs(savings_percentage):.1f}%"
            )
            st.metric(
                "Savings Percentage",
                f"{savings_percentage:.1f}%",
                delta=savings_delta,
                delta_color="inverse" if savings_percentage > 0 else "normal",
                help="Percentage saved by buying at quote prices vs stock average prices",
            )

        # Show cost comparison summary
        if items_with_stock_avg > 0:
            absolute_savings = total_cost_at_stock_avg - total_cost_at_quote
            if absolute_savings > 0:
                st.success(
                    f"💰 **Cost Savings**: You could save €{absolute_savings:.2f} ({savings_percentage:.1f}%) "
                    f"by choosing these quoted prices over your stock average prices for {items_with_stock_avg} products!"
                )
            elif absolute_savings < 0:
                st.warning(
                    f"⚠️ **Higher Cost**: These quotes would cost €{abs(absolute_savings):.2f} ({abs(savings_percentage):.1f}%) "
                    f"more than your stock average prices for {items_with_stock_avg} products."
                )
            else:
                st.info(
                    "ℹ️ **Similar Cost**: Quote prices are very close to your stock average prices."
                )
        else:
            st.info(
                "ℹ️ **No Stock Average Data**: Unable to calculate cost comparison without stock average prices."
            )

        # NEW: Quantity fulfillment analysis
        st.subheader("📦 Quantity Fulfillment Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Quantity Needed", f"{total_needed:,}")
        with col2:
            st.metric("Total Quantity Allocated", f"{total_allocated:,}")
        with col3:
            fulfillment_rate = (
                (total_allocated / total_needed * 100) if total_needed > 0 else 0
            )
            st.metric("Fulfillment Rate", f"{fulfillment_rate:.1f}%")

        if fulfillment_rate < 100:
            shortage = total_needed - total_allocated
            st.warning(
                f"⚠️ **Quantity Shortage**: {shortage:,} units cannot be fulfilled due to supplier constraints"
            )
        else:
            st.success(
                "✅ **Full Fulfillment**: All requested quantities can be allocated"
            )

    else:
        st.info("No allocation results to display")

    # Enhanced column guide
    with st.expander("📖 Enhanced Column Guide"):
        st.write("**Enhanced Table Columns:**")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Quantity Columns:**")
            st.write("• **Quantity Needed**: Original quantity from your CSV order")
            st.write(
                "• **Allocated Qty**: Quantity allocated to this specific supplier"
            )
            st.write(
                "• **Supplier Qty**: Available quantity from supplier with status:"
            )
            st.write("  - `50 ✅`: Supplier has sufficient quantity (≥ needed)")
            st.write(
                "  - `63 ⚠️ (need 371) (using 63)`: Insufficient quantity with details"
            )
            st.write("")
            st.write("**Pricing Columns:**")
            st.write("• **Quote Price**: Supplier's quoted price")
            st.write("• **Stock Avg Price**: Your average stock price")
            st.write("• **Best Buy Price**: Your best historical purchase price")
            st.write("• **Savings vs Stock**: Savings compared to stock average")
            st.write("• **Total Savings**: Allocated Qty × Savings vs Stock")

        with col2:
            st.write("**Allocation Information:**")
            st.write("• **📦 Single**: Order fulfilled by one supplier")
            st.write("• **🔄 Split #1, #2...**: Order split across multiple suppliers")
            st.write("• **Split Info**: Percentage and details of split allocation")
            st.write("")
            st.write("**Status Indicators:**")
            st.write(
                "• **✅**: Positive status (sufficient quantity, savings available)"
            )
            st.write("• **⚠️**: Warning status (quantity constraints, no savings)")
            st.write("• **N/A**: Data not available")
            st.write("• **Need/Using**: Shows shortage details in split orders")


def show_enhanced_supplier_breakdown(optimizer, summary):
    """Show enhanced supplier breakdown with allocation details - FIXED"""

    st.subheader("🏢 Supplier Order Breakdown")

    if (
        not hasattr(optimizer, "optimization_results")
        or not optimizer.optimization_results
    ):
        st.warning("⚠️ No optimization results available")
        return

    supplier_orders = optimizer.optimization_results.get("supplier_orders", {})
    if not supplier_orders:
        st.warning("⚠️ No supplier orders found")
        return

    supplier_data = []

    for supplier, items in supplier_orders.items():
        if not items:
            continue

        stats = {
            "items": len(items),
            "total_value": sum(item.get("total_price", 0) for item in items),
            "unique_products": len(set(item.get("ean_code", "") for item in items)),
            "split_order_items": sum(
                1 for item in items if item.get("is_split_order", False)
            ),
            "single_order_items": sum(
                1 for item in items if not item.get("is_split_order", False)
            ),
        }

        # Calculate cost comparisons if internal data is available
        total_cost_supplier = stats["total_value"]
        total_cost_stock_avg = 0
        items_with_stock_avg = 0

        for item in items:
            allocated_qty = item.get("quantity", 0)

            # Get stock average price if available
            if hasattr(optimizer, "internal_data_lookup"):
                ean = str(item.get("ean_code", "")).strip()
                if ean in optimizer.internal_data_lookup:
                    internal_product = optimizer.internal_data_lookup[ean]
                    stock_avg_price = internal_product.get("stock_avg_price")

                    if stock_avg_price is not None and stock_avg_price > 0:
                        total_cost_stock_avg += stock_avg_price * allocated_qty
                        items_with_stock_avg += 1

        # Calculate savings percentage and absolute savings
        savings_percentage = 0
        absolute_savings = 0
        if total_cost_stock_avg > 0:
            absolute_savings = total_cost_stock_avg - total_cost_supplier
            savings_percentage = (absolute_savings / total_cost_stock_avg) * 100

        supplier_data.append(
            {
                "Supplier": supplier,
                "Total Items": stats["items"],
                "Order Value": f"€{stats['total_value']:.2f}",
                "Unique Products": stats["unique_products"],
                "Split Orders": stats.get("split_order_items", 0),
                "Single Orders": stats.get("single_order_items", 0),
                "Total Cost Stock Avg": (
                    f"€{total_cost_stock_avg:.2f}"
                    if items_with_stock_avg > 0
                    else "N/A"
                ),
                "Total Cost Supplier": f"€{total_cost_supplier:.2f}",
                "Absolute Savings": (
                    f"€{absolute_savings:.2f}" if items_with_stock_avg > 0 else "N/A"
                ),
                "% Savings": (
                    f"{savings_percentage:.1f}%" if items_with_stock_avg > 0 else "N/A"
                ),
            }
        )

    # Create DataFrame with error handling
    if not supplier_data:
        st.warning("⚠️ No supplier data to display")
        return

    supplier_df = pd.DataFrame(supplier_data)

    # FIXED: Add error handling for Order Value column
    try:
        # Check if Order Value column exists
        if "Order Value" in supplier_df.columns:
            # Sort by order value with safe conversion
            sort_values = []
            for val in supplier_df["Order Value"]:
                try:
                    # Extract numeric value from string like "€123.45"
                    if isinstance(val, str) and "€" in val:
                        numeric_val = float(val.replace("€", "").replace(",", ""))
                        sort_values.append(numeric_val)
                    else:
                        sort_values.append(0)  # Default value for invalid entries
                except (ValueError, AttributeError):
                    sort_values.append(0)  # Default value for conversion errors

            supplier_df["_sort_value"] = sort_values
            supplier_df = supplier_df.sort_values("_sort_value", ascending=False).drop(
                "_sort_value", axis=1
            )
        else:
            st.error("❌ Error: Order Value column missing from supplier data")
            st.write("Available columns:", list(supplier_df.columns))
            return

    except Exception as e:
        st.error(f"❌ Error processing supplier data: {str(e)}")
        st.write("Debug info:")
        st.write(f"DataFrame shape: {supplier_df.shape}")
        st.write(f"Columns: {list(supplier_df.columns)}")
        if not supplier_df.empty:
            st.write("Sample data:")
            st.dataframe(supplier_df.head(2))
        return

    # Display the table
    st.dataframe(supplier_df, width='stretch')

    # NEW: Cost comparison explanation
    with st.expander("💰 Cost Comparison Columns Guide"):
        st.write("**Cost Analysis for Order Optimization:**")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Cost Columns:**")
            st.write(
                "• **Total Cost Stock Avg**: What you'd pay at stock average prices"
            )
            st.write(
                "• **Total Cost Supplier**: What you'd pay at supplier's quoted prices"
            )
            st.write(
                "• **Absolute Savings**: Total amount saved (€) by choosing supplier quotes"
            )
            st.write("• **% Savings**: Percentage saved by choosing supplier quotes")

        with col2:
            st.write("**Interpretation:**")
            st.write(
                "• **Positive Absolute Savings**: Supplier price is better than stock average"
            )
            st.write("• **N/A**: No stock average price data available for comparison")
            st.write(
                "• **Higher % Savings**: Better relative performance vs stock prices"
            )
            st.write(
                "• **Split vs Single Orders**: How orders are allocated across suppliers"
            )


def show_enhanced_export_options(optimizer):
    """Show enhanced export options with allocation details"""

    st.subheader("💾 Enhanced Export Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Download Enhanced Supplier Orders"):
            try:
                exported_files = optimizer.export_enhanced_orders()

                if exported_files:
                    st.success(
                        f"✅ Generated {len(exported_files)} enhanced order files"
                    )

                    for filename, csv_content in exported_files.items():
                        supplier_name = (
                            filename.replace("enhanced_order_", "")
                            .replace(f"_{datetime.now().strftime('%Y%m%d')}", "")
                            .replace(".csv", "")
                        )
                        lines = (
                            csv_content.count("\n") - 7
                        )  # Subtract headers and summary

                        st.download_button(
                            label=f"📁 {supplier_name} ({lines} items)",
                            data=csv_content,
                            file_name=filename,
                            mime="text/csv",
                            key=f"download_enhanced_{supplier_name}",
                        )
                else:
                    st.info("No orders to export")

            except Exception as e:
                st.error(f"❌ Export error: {str(e)}")

    with col2:
        if st.button("📊 Download Allocation Analysis"):
            try:
                # Create allocation analysis report
                analysis_lines = [
                    "SMART ALLOCATION ANALYSIS REPORT",
                    "=" * 50,
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "ALLOCATION SUMMARY:",
                ]

                summary = optimizer.get_optimization_summary()

                analysis_lines.extend(
                    [
                        f"- Total items processed: {summary['total_items_requested']}",
                        f"- Items successfully matched: {summary['matched_items']} ({summary['match_rate']:.1f}%)",
                        f"- Split orders created: {summary.get('split_orders_count', 0)}",
                        f"- Single supplier orders: {summary.get('single_orders_count', 0)}",
                        f"- Total allocations: {summary.get('total_allocations', 0)}",
                        f"- Allocation efficiency: {summary.get('allocation_efficiency', 1):.2f}x",
                        f"- Total order value: €{summary['total_order_value']:.2f}",
                        "",
                        "SUPPLIER BREAKDOWN:",
                    ]
                )

                for supplier, stats in summary["supplier_breakdown"].items():
                    analysis_lines.extend(
                        [
                            f"- {supplier}:",
                            f"  Items: {stats['items']}",
                            f"  Value: €{stats['total_value']:.2f}",
                            f"  Split items: {stats.get('split_order_items', 0)}",
                            f"  Single items: {stats.get('single_order_items', 0)}",
                            "",
                        ]
                    )

                analysis_content = "\n".join(analysis_lines)

                st.download_button(
                    label="📋 Download Analysis Report",
                    data=analysis_content,
                    file_name=f"smart_allocation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"❌ Analysis export error: {str(e)}")

    # Enhanced export info
    st.info(
        "📊 **Enhanced Exports Include**: Split order details, stock average comparisons, "
        "allocation rankings, quantity constraints, and savings analysis"
    )


# ===================================================================
# UNFULFILLED PRODUCTS FUNCTIONS - Add these to your app.py file
# ===================================================================


def get_unfulfilled_products(optimizer):
    """Get all unfulfilled products including unmatched and partially fulfilled"""

    if (
        not hasattr(optimizer, "optimization_results")
        or not optimizer.optimization_results
    ):
        return pd.DataFrame()

    unfulfilled_data = []
    results = optimizer.optimization_results

    # 1. Get completely unmatched items
    for list_result in results.get("lists_processed", []):
        for item in list_result.get("unmatched_items_list", []):
            unfulfilled_data.append(
                {
                    "Type": "Unmatched",
                    "List Name": list_result.get("list_name", "Unknown"),
                    "EAN/Reference": item.get("original_reference", "Unknown"),
                    "Product Description": item.get("description", "N/A"),
                    "Quantity Requested": item.get("quantity", 0),
                    "Quantity Fulfilled": 0,
                    "Quantity Unfulfilled": item.get("quantity", 0),
                    "Fulfillment %": "0%",
                    "Reason": item.get("reason", "Unknown reason"),
                    "Available Suppliers": item.get("available_suppliers", 0),
                    "Stock Avg Price": (
                        f"€{item.get('stock_avg_price', 0):.2f}"
                        if item.get("stock_avg_price")
                        else "N/A"
                    ),
                    "Status": "❌ Not Found",
                }
            )

    # 2. Check for partially fulfilled items by analyzing allocation vs requested quantities
    # Track original quantities requested vs allocated
    ean_requests = {}  # Track total quantities requested per EAN
    ean_allocations = {}  # Track total quantities allocated per EAN
    ean_details = {}  # Track product details per EAN

    # First, collect all requests from buying lists
    for list_result in results.get("lists_processed", []):
        list_name = list_result.get("list_name", "Unknown")

        # Get processed items to understand original requests
        for item in list_result.get("processed_items", []):
            # For enhanced optimizer, items have allocations
            if "allocations" in item:
                allocations = item.get("allocations", [])
                for allocation in allocations:
                    ean = allocation.get("ean_code", "Unknown")
                    original_ref = allocation.get("original_reference", ean)
                    product_name = allocation.get("product_name", "Unknown")

                    # Track original quantity requested (this is tricky - we need to reconstruct it)
                    original_qty = allocation.get(
                        "original_quantity_needed", allocation.get("quantity", 0)
                    )
                    allocated_qty = allocation.get("quantity", 0)

                    if ean not in ean_requests:
                        ean_requests[ean] = 0
                        ean_allocations[ean] = 0
                        ean_details[ean] = {
                            "list_name": list_name,
                            "reference": original_ref,
                            "product_name": product_name,
                        }

                    # For split orders, we need to sum up the original quantities
                    if allocation.get("is_split_order"):
                        # Add allocated quantity
                        ean_allocations[ean] += allocated_qty
                        # Original quantity should be tracked differently for split orders
                        # Use the original_quantity_needed if available
                        if "original_quantity_needed" in allocation:
                            ean_requests[ean] = max(
                                ean_requests[ean],
                                allocation["original_quantity_needed"],
                            )
                    else:
                        # Single order
                        ean_requests[ean] += original_qty
                        ean_allocations[ean] += allocated_qty
            else:
                # Basic optimizer format
                ean = item.get("ean_code", "Unknown")
                original_ref = item.get("original_reference", ean)
                product_name = item.get("product_name", "Unknown")
                quantity = item.get("quantity", 0)

                if ean not in ean_requests:
                    ean_requests[ean] = 0
                    ean_allocations[ean] = 0
                    ean_details[ean] = {
                        "list_name": list_name,
                        "reference": original_ref,
                        "product_name": product_name,
                    }

                ean_requests[ean] += quantity
                ean_allocations[ean] += quantity

    # Check for partial fulfillment
    for ean in ean_requests:
        requested = ean_requests[ean]
        allocated = ean_allocations[ean]
        unfulfilled = requested - allocated

        if unfulfilled > 0:  # Partially fulfilled
            fulfillment_pct = (allocated / requested * 100) if requested > 0 else 0
            details = ean_details[ean]

            # Get stock average price if available
            stock_avg_price = "N/A"
            if (
                hasattr(optimizer, "internal_data_lookup")
                and optimizer.internal_data_lookup
                and ean in optimizer.internal_data_lookup
            ):
                internal_product = optimizer.internal_data_lookup[ean]
                if internal_product.get("stock_avg_price"):
                    stock_avg_price = f"€{internal_product['stock_avg_price']:.2f}"

            unfulfilled_data.append(
                {
                    "Type": "Partially Fulfilled",
                    "List Name": details["list_name"],
                    "EAN/Reference": details["reference"],
                    "Product Description": details["product_name"],
                    "Quantity Requested": requested,
                    "Quantity Fulfilled": allocated,
                    "Quantity Unfulfilled": unfulfilled,
                    "Fulfillment %": f"{fulfillment_pct:.1f}%",
                    "Reason": "Insufficient supplier quantity",
                    "Available Suppliers": "Multiple",
                    "Stock Avg Price": stock_avg_price,
                    "Status": "⚠️ Partial",
                }
            )

    return pd.DataFrame(unfulfilled_data)


def show_unfulfilled_products_table(optimizer):
    """Display unfulfilled products table with export functionality"""

    st.subheader("❌ Unfulfilled Products Analysis")

    # Get unfulfilled products data
    unfulfilled_df = get_unfulfilled_products(optimizer)

    if unfulfilled_df.empty:
        st.success(
            "🎉 **Perfect Fulfillment!** All requested products have been successfully allocated."
        )
        return

    # Summary metrics
    total_unfulfilled = len(unfulfilled_df)
    unmatched_count = len(unfulfilled_df[unfulfilled_df["Type"] == "Unmatched"])
    partial_count = len(unfulfilled_df[unfulfilled_df["Type"] == "Partially Fulfilled"])

    total_qty_unfulfilled = unfulfilled_df["Quantity Unfulfilled"].sum()
    total_qty_requested = unfulfilled_df["Quantity Requested"].sum()

    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Unfulfilled Items", total_unfulfilled)
    with col2:
        st.metric("❌ Completely Unmatched", unmatched_count)
    with col3:
        st.metric("⚠️ Partially Fulfilled", partial_count)
    with col4:
        unfulfillment_rate = (
            (total_qty_unfulfilled / total_qty_requested * 100)
            if total_qty_requested > 0
            else 0
        )
        st.metric("Unfulfillment Rate", f"{unfulfillment_rate:.1f}%")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        type_filter = st.selectbox(
            "Filter by Type",
            ["All", "Unmatched", "Partially Fulfilled"],
            help="Filter products by fulfillment status",
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Quantity Unfulfilled", "Quantity Requested", "EAN/Reference"],
            help="Sort products by different criteria",
        )

    # Apply filters
    filtered_df = unfulfilled_df.copy()
    if type_filter != "All":
        filtered_df = filtered_df[filtered_df["Type"] == type_filter]

    # Sort
    if sort_by == "Quantity Unfulfilled":
        filtered_df = filtered_df.sort_values("Quantity Unfulfilled", ascending=False)
    elif sort_by == "Quantity Requested":
        filtered_df = filtered_df.sort_values("Quantity Requested", ascending=False)
    else:
        filtered_df = filtered_df.sort_values("EAN/Reference")

    # Display table
    if not filtered_df.empty:
        st.dataframe(filtered_df, width='stretch', height=400)

        # Analysis by type
        with st.expander("📊 Unfulfillment Analysis"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Unmatched Products Analysis:**")
                unmatched_items = filtered_df[filtered_df["Type"] == "Unmatched"]
                if not unmatched_items.empty:
                    st.write(f"• {len(unmatched_items)} products could not be matched")
                    st.write(
                        f"• {unmatched_items['Quantity Requested'].sum()} total units requested"
                    )

                    # Top reasons for unmatched
                    reason_counts = unmatched_items["Reason"].value_counts()
                    st.write("**Top reasons:**")
                    for reason, count in reason_counts.head(3).items():
                        st.write(f"  - {reason}: {count} items")
                else:
                    st.write("• No completely unmatched products")

            with col2:
                st.write("**Partially Fulfilled Analysis:**")
                partial_items = filtered_df[
                    filtered_df["Type"] == "Partially Fulfilled"
                ]
                if not partial_items.empty:
                    avg_fulfillment = (
                        partial_items["Fulfillment %"]
                        .str.rstrip("%")
                        .astype(float)
                        .mean()
                    )
                    st.write(f"• {len(partial_items)} products partially fulfilled")
                    st.write(f"• Average fulfillment rate: {avg_fulfillment:.1f}%")
                    st.write(
                        f"• {partial_items['Quantity Unfulfilled'].sum()} units still needed"
                    )
                else:
                    st.write("• No partially fulfilled products")

    else:
        st.info("No unfulfilled products match the current filters.")

    # Export buttons
    if not unfulfilled_df.empty:
        st.write("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Export filtered results as CSV
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📊 Export Filtered Table (CSV)",
                data=csv_data,
                file_name=f"unfulfilled_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the currently filtered unfulfilled products table",
            )

        with col2:
            # Export all unfulfilled products
            all_csv_data = unfulfilled_df.to_csv(index=False)
            st.download_button(
                label="📋 Export All Unfulfilled (CSV)",
                data=all_csv_data,
                file_name=f"all_unfulfilled_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download all unfulfilled products regardless of filters",
            )

        with col3:
            # Export summary report
            if st.button("📄 Generate Summary Report"):
                report_content = generate_unfulfilled_summary_report(
                    unfulfilled_df, optimizer
                )
                st.download_button(
                    label="📋 Download Summary Report",
                    data=report_content,
                    file_name=f"unfulfilled_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Download detailed summary report of unfulfilled products",
                )


def generate_unfulfilled_summary_report(unfulfilled_df, optimizer):
    """Generate a comprehensive summary report of unfulfilled products"""

    if unfulfilled_df.empty:
        return "UNFULFILLED PRODUCTS REPORT\n\n✅ Perfect Fulfillment: All products successfully allocated!"

    # Get optimization summary for context
    summary = (
        optimizer.get_optimization_summary()
        if hasattr(optimizer, "get_optimization_summary")
        else {}
    )

    lines = [
        "UNFULFILLED PRODUCTS SUMMARY REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OVERALL OPTIMIZATION SUMMARY:",
        f"- Total items requested: {summary.get('total_items_requested', 'N/A')}",
        f"- Successfully matched: {summary.get('matched_items', 'N/A')}",
        f"- Unfulfilled items: {len(unfulfilled_df)}",
        f"- Overall match rate: {summary.get('match_rate', 0):.1f}%",
        "",
        "UNFULFILLMENT BREAKDOWN:",
    ]

    # Analyze by type
    unmatched = unfulfilled_df[unfulfilled_df["Type"] == "Unmatched"]
    partial = unfulfilled_df[unfulfilled_df["Type"] == "Partially Fulfilled"]

    lines.extend(
        [
            f"- Completely unmatched: {len(unmatched)} items ({unmatched['Quantity Requested'].sum()} units)",
            f"- Partially fulfilled: {len(partial)} items ({partial['Quantity Unfulfilled'].sum()} units needed)",
            "",
            "QUANTITY ANALYSIS:",
            f"- Total quantity unfulfilled: {unfulfilled_df['Quantity Unfulfilled'].sum():,} units",
            f"- Total quantity requested: {unfulfilled_df['Quantity Requested'].sum():,} units",
            f"- Unfulfillment rate: {(unfulfilled_df['Quantity Unfulfilled'].sum() / unfulfilled_df['Quantity Requested'].sum() * 100):.1f}%",
            "",
        ]
    )

    # Top unmatched reasons
    if not unmatched.empty:
        lines.extend(["TOP REASONS FOR UNMATCHED PRODUCTS:", ""])
        reason_counts = unmatched["Reason"].value_counts()
        for i, (reason, count) in enumerate(reason_counts.head(5).items()):
            lines.append(f"{i+1}. {reason}: {count} items")
        lines.append("")

    # Detailed unfulfilled list
    lines.extend(["DETAILED UNFULFILLED PRODUCTS:", "-" * 30, ""])

    for _, row in unfulfilled_df.iterrows():
        lines.extend(
            [
                f"Product: {row['EAN/Reference']} - {row['Product Description']}",
                f"  Type: {row['Type']} ({row['Status']})",
                f"  Quantities: {row['Quantity Fulfilled']}/{row['Quantity Requested']} fulfilled ({row['Fulfillment %']})",
                f"  Unfulfilled: {row['Quantity Unfulfilled']} units",
                f"  Reason: {row['Reason']}",
                f"  Stock Avg Price: {row['Stock Avg Price']}",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "RECOMMENDATIONS:",
            "1. Contact suppliers for products marked as 'Unmatched'",
            "2. Consider alternative suppliers for partially fulfilled items",
            "3. Review stock average pricing for better matching",
            "4. Update supplier catalogs with missing products",
            "",
            "Report End",
        ]
    )

    return "\n".join(lines)


# Run the app
if __name__ == "__main__":
    main()
