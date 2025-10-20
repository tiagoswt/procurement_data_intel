"""
Utility functions for the procurement processing system
"""

import json
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from models import ProductData, ProcessingResult

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def export_to_json(
    products: List[ProductData], filename: str = None, supplier_name: str = None
) -> str:
    """Export products to JSON format"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        supplier_clean = (
            supplier_name.replace(" ", "_") if supplier_name else "products"
        )
        filename = f"{supplier_clean}_{timestamp}.json"

    output_data = {
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "supplier_name": supplier_name,
            "total_products": len(products),
            "export_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
        "products": [product.to_dict() for product in products],
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {len(products)} products to {filename}")
    return filename


def export_to_csv(
    products: List[ProductData], filename: str = None, supplier_name: str = None
) -> str:
    """Export products to CSV format"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        supplier_clean = (
            supplier_name.replace(" ", "_") if supplier_name else "products"
        )
        filename = f"{supplier_clean}_{timestamp}.csv"

    # Convert products to DataFrame
    for product_dict in products_data:
        if product_dict.get("ean_code"):
            product_dict["ean_code"] = pad_ean_code(product_dict["ean_code"])

    df = pd.DataFrame(products_data)

    # Reorder columns for better readability
    column_order = [
        "supplier",
        "ean_code",
        "supplier_code",
        "product_name",
        "price",
        "quantity",
        "confidence_score",
        "source_file",
    ]

    # Only use columns that exist in the data
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]

    df.to_csv(filename, index=False, encoding="utf-8")

    logger.info(f"Exported {len(products)} products to {filename}")
    return filename


def create_processing_report(
    result: ProcessingResult, supplier_name: str = None
) -> str:
    """Create a detailed processing report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = []
    report.append("PROCUREMENT DATA PROCESSING REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {timestamp}")
    report.append(f"Processing Status: {'SUCCESS' if result.success else 'FAILED'}")
    report.append(f"Processing Time: {result.processing_time:.2f} seconds")
    report.append("")

    if supplier_name:
        report.append(f"Supplier: {supplier_name}")

    report.append(f"Files Processed: {result.files_processed}")
    report.append(f"Total Products Extracted: {result.total_products}")
    report.append("")

    if result.products:
        # Product analysis
        products = result.products
        ean_count = sum(1 for p in products if p.ean_code)
        supplier_code_count = sum(1 for p in products if p.supplier_code)
        named_products = sum(1 for p in products if p.product_name)

        report.append("PRODUCT ANALYSIS")
        report.append("-" * 20)
        report.append(f"Products with EAN codes: {ean_count}")
        report.append(f"Products with supplier codes: {supplier_code_count}")
        report.append(f"Products with names: {named_products}")
        report.append(f"Data completeness: {(named_products/len(products)*100):.1f}%")
        report.append("")

        # Price analysis
        prices = [p.price for p in products if p.price is not None]
        if prices:
            report.append("PRICE ANALYSIS")
            report.append("-" * 15)
            report.append(f"Min price: {min(prices):.2f}€")
            report.append(f"Max price: {max(prices):.2f}€")
            report.append(f"Average price: {sum(prices)/len(prices):.2f}€")
            report.append(f"Total value: {sum(prices):.2f}€")
            report.append("")

        # Sample products
        report.append("SAMPLE PRODUCTS")
        report.append("-" * 16)
        for i, product in enumerate(products[:5]):
            code_info = product.ean_code or product.supplier_code or "No code"
            name_info = product.product_name or "No name"
            if len(name_info) > 50:
                name_info = name_info[:50] + "..."
            report.append(f"{i+1}. {code_info} - {name_info} - {product.price}€")

        if len(products) > 5:
            report.append(f"... and {len(products) - 5} more products")
        report.append("")

    if result.errors:
        report.append("ERRORS ENCOUNTERED")
        report.append("-" * 18)
        for i, error in enumerate(result.errors, 1):
            report.append(f"{i}. {error}")
        report.append("")

    return "\n".join(report)


def validate_groq_api_key(api_key: str) -> bool:
    """Validate Groq API key format"""
    if not api_key:
        return False

    # Basic format validation
    if not api_key.startswith("gsk_"):
        return False

    if len(api_key) < 50:  # Groq keys are typically longer
        return False

    return True


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def clean_filename(filename: str) -> str:
    """Clean filename for safe saving"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove multiple underscores
    while "__" in filename:
        filename = filename.replace("__", "_")

    return filename.strip("_")


def create_sample_data() -> Dict[str, Any]:
    """Create sample data for demonstration"""
    return {
        "headers": ["Gencod", "Designation", "Prix", "Stock"],
        "sample_row": {
            "Gencod": "1234567890123",
            "Designation": "Premium Serum Anti-Age 30ml",
            "Prix": 29.99,
            "Stock": 50,
        },
        "total_rows": 100,
    }


def get_processing_stats(products: List[ProductData]) -> Dict[str, Any]:
    """Get comprehensive processing statistics"""
    if not products:
        return {"total_products": 0}

    stats = {
        "total_products": len(products),
        "code_analysis": {
            "ean_codes": sum(1 for p in products if p.ean_code),
            "supplier_codes": sum(1 for p in products if p.supplier_code),
            "no_codes": sum(
                1 for p in products if not p.ean_code and not p.supplier_code
            ),
        },
        "data_completeness": {
            "with_names": sum(1 for p in products if p.product_name),
            "with_quantities": sum(1 for p in products if p.quantity is not None),
            "with_prices": sum(1 for p in products if p.price is not None),
        },
        "price_stats": {},
        "suppliers": {},
    }

    # Price statistics
    prices = [p.price for p in products if p.price is not None]
    if prices:
        stats["price_stats"] = {
            "min": min(prices),
            "max": max(prices),
            "average": sum(prices) / len(prices),
            "total_value": sum(prices),
        }

    # Supplier breakdown
    suppliers = {}
    for product in products:
        supplier = product.supplier or "Unknown"
        if supplier not in suppliers:
            suppliers[supplier] = 0
        suppliers[supplier] += 1

    stats["suppliers"] = suppliers

    return stats


def merge_processing_results(results: List[ProcessingResult]) -> ProcessingResult:
    """Merge multiple processing results into one"""
    if not results:
        return ProcessingResult(
            success=False,
            products=[],
            errors=["No results to merge"],
            processing_time=0,
            files_processed=0,
            total_products=0,
        )

    merged_products = []
    merged_errors = []
    total_time = 0
    total_files = 0

    success = False

    for result in results:
        merged_products.extend(result.products)
        merged_errors.extend(result.errors)
        total_time += result.processing_time
        total_files += result.files_processed

        if result.success:
            success = True

    return ProcessingResult(
        success=success,
        products=merged_products,
        errors=merged_errors,
        processing_time=total_time,
        files_processed=total_files,
        total_products=len(merged_products),
    )


def pad_ean_code(ean):
    """
    Pad EAN code with leading zeros to make it exactly 13 digits

    Args:
        ean: EAN code as string, int, float, or None

    Returns:
        str: 13-digit EAN code with leading zeros, or empty string if invalid
    """
    if not ean or pd.isna(ean):
        return ""

    # Convert to string and clean
    ean_str = str(ean).strip()

    # Handle float format (remove .0)
    if ean_str.endswith(".0"):
        ean_str = ean_str[:-2]

    # Remove any non-digit characters
    ean_digits = "".join(c for c in ean_str if c.isdigit())

    # Skip if no digits or too long
    if not ean_digits or len(ean_digits) > 13:
        return ean_digits  # Return as-is if too long

    # Pad with leading zeros to make 13 digits
    return ean_digits.zfill(13)
