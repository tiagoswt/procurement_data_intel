"""
Data normalization and validation utilities with Universal EAN Detection
"""

import pandas as pd
import logging
from typing import List, Optional, Union
from models import ProductData, FieldMapping

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalizes and validates extracted data with universal EAN detection"""

    @staticmethod
    def normalize_data(
        df: pd.DataFrame, field_mapping: FieldMapping, supplier_name: str = None
    ) -> List[ProductData]:
        """Universal normalization that detects ALL EAN variants as EAN codes"""

        products = []
        mapping_dict = field_mapping.to_dict()

        logger.info(f"🎯 Starting universal normalization with mapping: {mapping_dict}")

        # Check if we have a product code field
        product_code_field = mapping_dict.get("product_code")
        if not product_code_field:
            logger.warning("⚠️ No product_code field mapped - results may be limited")

        # Process each row
        for idx, row in df.iterrows():
            try:
                # Extract basic fields
                product_code = DataNormalizer._extract_field(row, product_code_field)
                price = DataNormalizer._extract_price(row, mapping_dict.get("price"))

                # Skip rows without essential data
                if not product_code or price is None:
                    if idx < 5:  # Log first few for debugging
                        logger.debug(
                            f"Row {idx}: Skipping - product_code='{product_code}', price='{price}'"
                        )
                    continue

                # UNIVERSAL EAN CLASSIFICATION - The key improvement
                ean_code, supplier_code = DataNormalizer._universal_classify_code(
                    product_code
                )

                # Extract optional fields
                product_name = DataNormalizer._extract_field(
                    row, mapping_dict.get("product_name")
                )
                quantity = DataNormalizer._extract_quantity(
                    row, mapping_dict.get("quantity")
                )

                # Create product object
                product = ProductData(
                    ean_code=ean_code,
                    supplier_code=supplier_code,
                    product_name=product_name,
                    quantity=quantity,
                    price=float(price),
                    supplier=supplier_name,
                    confidence_score=1.0,
                )

                products.append(product)

                # Log first few products for debugging
                if idx < 5:
                    logger.debug(
                        f"Row {idx}: '{product_code}' → EAN:'{ean_code}', Supplier:'{supplier_code}', Price:{price}"
                    )

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue

        # Log final summary
        ean_count = sum(1 for p in products if p.ean_code)
        supplier_count = sum(1 for p in products if p.supplier_code)
        logger.info(f"📊 UNIVERSAL DETECTION RESULTS:")
        logger.info(f"   Total products: {len(products)}")
        logger.info(f"   EAN codes: {ean_count} ({ean_count/len(products)*100:.1f}%)")
        logger.info(
            f"   Supplier codes: {supplier_count} ({supplier_count/len(products)*100:.1f}%)"
        )

        return products

    @staticmethod
    def _universal_classify_code(product_code) -> tuple:
        """
        Universal product code classification - LIBERAL approach to EAN detection

        Philosophy: Better to classify as EAN and be wrong than miss real EAN codes
        """

        if not product_code:
            return None, None

        # Step 1: Clean the code
        clean_code = DataNormalizer._clean_product_code(product_code)
        original_code = str(product_code).strip()

        logger.debug(f"🔍 Classifying: '{original_code}' → clean: '{clean_code}'")

        # Step 2: Numeric code classification (most EANs are numeric)
        if clean_code.isdigit():
            length = len(clean_code)

            # Perfect barcode lengths - definitely EAN
            if length in [8, 12, 13, 14]:
                logger.debug(f"✅ Perfect EAN length ({length}): '{clean_code}'")
                return clean_code, None

            # Extended barcode lengths - likely EAN
            elif 7 <= length <= 15:
                logger.debug(f"✅ Reasonable EAN length ({length}): '{clean_code}'")
                return clean_code, None

            # Very long numeric codes - probably EAN (some systems use extended formats)
            elif length > 15:
                logger.debug(f"✅ Extended EAN format ({length}): '{clean_code}'")
                return clean_code, None

            # Short numeric codes - could be either, lean towards supplier code
            elif length >= 4:
                logger.debug(
                    f"📝 Short numeric supplier code ({length}): '{clean_code}'"
                )
                return None, original_code

            # Very short codes - definitely supplier code
            else:
                logger.debug(f"📝 Very short supplier code ({length}): '{clean_code}'")
                return None, original_code

        # Step 3: Mixed alphanumeric classification
        elif clean_code:
            # Calculate digit ratio
            digit_count = sum(1 for c in clean_code if c.isdigit())
            total_length = len(clean_code)
            digit_ratio = digit_count / total_length if total_length > 0 else 0

            # Mostly numeric with reasonable length - treat as EAN
            if digit_ratio >= 0.7 and total_length >= 7:
                logger.debug(
                    f"✅ Mixed EAN ({digit_ratio:.0%} digits, len {total_length}): '{clean_code}'"
                )
                return clean_code, None

            # Some digits, reasonable length - could be EAN
            elif digit_ratio >= 0.5 and total_length >= 8:
                logger.debug(
                    f"✅ Possible EAN ({digit_ratio:.0%} digits, len {total_length}): '{clean_code}'"
                )
                return clean_code, None

            # Mostly alphabetic - supplier code
            else:
                logger.debug(
                    f"📝 Alphanumeric supplier code ({digit_ratio:.0%} digits): '{clean_code}'"
                )
                return None, original_code

        # Step 4: Empty or invalid
        else:
            logger.debug(f"❌ Empty/invalid code: '{original_code}'")
            return None, None

    @staticmethod
    def _clean_product_code(product_code) -> str:
        """Enhanced product code cleaning for universal compatibility"""

        if not product_code:
            return ""

        # Convert to string
        code_str = str(product_code).strip()

        # Remove Excel float formatting (e.g., "3337875829007.0" → "3337875829007")
        if code_str.endswith(".0"):
            code_str = code_str[:-2]

        # Remove common separators and formatting
        # Keep alphanumeric only for classification
        clean_code = ""
        for char in code_str:
            if char.isalnum():  # Keep letters and numbers
                clean_code += char
            # Skip spaces, dashes, dots, etc.

        return clean_code

    @staticmethod
    def _extract_field(row: pd.Series, field_name: Optional[str]) -> Optional[str]:
        """Extract and clean field value"""
        if not field_name or field_name not in row:
            return None

        value = row[field_name]
        if pd.isna(value):
            return None

        return str(value).strip()

    @staticmethod
    def _extract_price(row: pd.Series, field_name: Optional[str]) -> Optional[float]:
        """Extract and normalize price field with enhanced cleaning"""
        if not field_name or field_name not in row:
            return None

        value = row[field_name]
        if pd.isna(value):
            return None

        # Handle string prices (e.g., "€29.99", "25,50", "$15.00")
        if isinstance(value, str):
            # Remove currency symbols and non-numeric characters except . and ,
            cleaned = ""
            for char in value:
                if char.isdigit() or char in ".,":
                    cleaned += char

            # Handle European comma decimal separator
            if "," in cleaned and "." in cleaned:
                # If both comma and dot, assume dot is decimal separator
                cleaned = cleaned.replace(",", "")
            elif "," in cleaned:
                # If only comma, treat as decimal separator
                cleaned = cleaned.replace(",", ".")

            try:
                return float(cleaned) if cleaned else None
            except ValueError:
                logger.warning(f"Could not parse price: '{value}' → '{cleaned}'")
                return None

        # Handle numeric prices
        try:
            price = float(value)
            # Basic validation - prices should be positive and reasonable
            if 0 <= price <= 1000000:
                return price
            else:
                logger.warning(f"Price out of reasonable range: {price}")
                return price  # Return anyway, let validation handle it
        except (ValueError, TypeError):
            logger.warning(f"Could not convert price to float: '{value}'")
            return None

    @staticmethod
    def _extract_quantity(
        row: pd.Series, field_name: Optional[str]
    ) -> Optional[Union[int, float]]:
        """Extract and normalize quantity field"""
        if not field_name or field_name not in row:
            return None

        value = row[field_name]
        if pd.isna(value):
            return None

        try:
            # Convert to float first
            float_val = float(value)

            # If it's a whole number, return as int
            if float_val == int(float_val):
                return int(float_val)
            else:
                return float_val

        except (ValueError, TypeError):
            # Try to extract numbers from string
            if isinstance(value, str):
                # Extract first numeric sequence
                numeric_part = ""
                for char in value:
                    if char.isdigit() or char in ".,":
                        numeric_part += char
                    elif (
                        numeric_part
                    ):  # Stop at first non-numeric after finding numbers
                        break

                if numeric_part:
                    try:
                        numeric_part = numeric_part.replace(",", ".")
                        float_val = float(numeric_part)
                        return (
                            int(float_val) if float_val == int(float_val) else float_val
                        )
                    except ValueError:
                        pass

            logger.warning(f"Could not parse quantity: '{value}'")
            return None

    @staticmethod
    def validate_products(products: List[ProductData]) -> List[str]:
        """Enhanced validation with universal EAN considerations"""
        issues = []

        if not products:
            issues.append("No products were extracted from the file")
            return issues

        # Check for missing essential data
        products_without_code = sum(
            1 for p in products if not p.ean_code and not p.supplier_code
        )
        products_without_price = sum(
            1 for p in products if p.price is None or p.price <= 0
        )

        if products_without_code > 0:
            issues.append(f"{products_without_code} products missing product codes")

        if products_without_price > 0:
            issues.append(f"{products_without_price} products missing valid prices")

        # EAN code quality analysis
        ean_products = [p for p in products if p.ean_code]
        if ean_products:
            standard_eans = sum(
                1
                for p in ean_products
                if p.ean_code.isdigit() and len(p.ean_code) in [8, 12, 13]
            )
            non_standard_eans = len(ean_products) - standard_eans

            if non_standard_eans > 0:
                issues.append(
                    f"{non_standard_eans} EAN codes are non-standard format (but may still be valid)"
                )

        # Price validation
        prices = [p.price for p in products if p.price is not None]
        if prices:
            negative_prices = sum(1 for p in prices if p < 0)
            high_prices = sum(1 for p in prices if p > 100000)

            if negative_prices > 0:
                issues.append(f"{negative_prices} products have negative prices")
            if high_prices > 0:
                issues.append(
                    f"{high_prices} products have unusually high prices (>€100,000)"
                )

        # Check for potential duplicate EAN codes
        ean_codes = [p.ean_code for p in products if p.ean_code]
        if len(ean_codes) != len(set(ean_codes)):
            duplicates = len(ean_codes) - len(set(ean_codes))
            issues.append(f"{duplicates} duplicate EAN codes found")

        return issues

    @staticmethod
    def get_data_summary(products: List[ProductData]) -> dict:
        """Get comprehensive summary statistics of normalized data"""
        if not products:
            return {"total_products": 0}

        # Basic counts
        summary = {
            "total_products": len(products),
            "products_with_ean": sum(1 for p in products if p.ean_code),
            "products_with_supplier_code": sum(1 for p in products if p.supplier_code),
            "products_with_name": sum(1 for p in products if p.product_name),
            "products_with_quantity": sum(
                1 for p in products if p.quantity is not None
            ),
        }

        # EAN quality analysis
        ean_products = [p for p in products if p.ean_code]
        if ean_products:
            standard_eans = sum(
                1
                for p in ean_products
                if p.ean_code.isdigit() and len(p.ean_code) in [8, 12, 13]
            )
            summary["standard_ean_codes"] = standard_eans
            summary["non_standard_ean_codes"] = len(ean_products) - standard_eans

        # Price statistics
        prices = [p.price for p in products if p.price is not None]
        if prices:
            summary.update(
                {
                    "min_price": min(prices),
                    "max_price": max(prices),
                    "avg_price": sum(prices) / len(prices),
                    "total_value": sum(prices),
                }
            )

        # Data completeness percentage
        summary["data_completeness"] = (
            (summary["products_with_name"] / summary["total_products"]) * 100
            if summary["total_products"] > 0
            else 0
        )

        # EAN detection rate
        summary["ean_detection_rate"] = (
            (summary["products_with_ean"] / summary["total_products"]) * 100
            if summary["total_products"] > 0
            else 0
        )

        return summary

    @staticmethod
    def get_classification_stats(products: List[ProductData]) -> dict:
        """Get detailed classification statistics"""

        if not products:
            return {}

        ean_products = [p for p in products if p.ean_code]
        supplier_products = [p for p in products if p.supplier_code]

        stats = {
            "total_classified": len(products),
            "ean_classified": len(ean_products),
            "supplier_classified": len(supplier_products),
            "ean_rate": len(ean_products) / len(products) * 100 if products else 0,
        }

        # EAN length distribution
        if ean_products:
            length_dist = {}
            for product in ean_products:
                length = len(product.ean_code) if product.ean_code else 0
                length_dist[length] = length_dist.get(length, 0) + 1
            stats["ean_length_distribution"] = length_dist

        return stats
