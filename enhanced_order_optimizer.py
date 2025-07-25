"""
Enhanced Order Optimizer with Smart Multi-Supplier Allocation
Fixed version with proper indentation and all required functions
"""

import pandas as pd
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from models import ProductData

logger = logging.getLogger(__name__)


class EnhancedOrderOptimizer:
    """Complete order optimizer with smart multi-supplier allocation"""

    def __init__(self):
        self.supplier_data = []
        self.buying_lists = []
        self.optimization_results = {}
        self.internal_data_lookup = {}  # NEW: For stock average price comparison
        self.ean_lookup = defaultdict(list)

    # =============================================================================
    # NEW ENHANCED FUNCTIONS
    # =============================================================================

    def load_internal_data(self, internal_data: List[Dict]) -> None:
        """
        NEW: Load internal data for stock average price comparison
        Expected format: [{'ean': '123', 'stock_avg_price': 25.50, ...}, ...]
        """
        self.internal_data_lookup = {}

        for product in internal_data:
            ean = str(product.get("ean", "")).strip()
            if ean and product.get("stock_avg_price"):
                self.internal_data_lookup[ean] = {
                    "stock_avg_price": float(product["stock_avg_price"]),
                    "supplier_price": product.get("supplier_price"),
                    "best_buy_price": product.get("best_buy_price"),
                    "brand": product.get("brand", ""),
                    "description": product.get("description", ""),
                }

        logger.info(
            f"📊 Loaded {len(self.internal_data_lookup)} internal products for price comparison"
        )

    def _get_stock_average_price(self, ean: str) -> Optional[float]:
        """Get stock average price for EAN from internal data"""
        internal_product = self.internal_data_lookup.get(ean)
        if internal_product and internal_product.get("stock_avg_price"):
            return float(internal_product["stock_avg_price"])
        return None

    def _filter_suppliers_by_price(
        self, suppliers: List[ProductData], ean: str
    ) -> List[ProductData]:
        """Filter suppliers to only include those with prices better than stock average"""
        stock_avg_price = self._get_stock_average_price(ean)

        if stock_avg_price is None:
            # No stock average available, return all suppliers
            logger.debug(f"No stock average price for EAN {ean}, using all suppliers")
            return suppliers

        # Filter suppliers with prices better than stock average
        qualified_suppliers = []
        for supplier in suppliers:
            if supplier.price and supplier.price < stock_avg_price:
                qualified_suppliers.append(supplier)

        logger.debug(
            f"EAN {ean}: {len(qualified_suppliers)}/{len(suppliers)} suppliers qualify (< €{stock_avg_price:.2f})"
        )

        return qualified_suppliers

    def _smart_allocate_order(
        self, ean: str, quantity: int, suppliers: List[ProductData]
    ) -> List[Dict]:
        """Smart allocation logic with original quantity tracking"""
        if not suppliers:
            return []

        # Filter suppliers by stock average price
        qualified_suppliers = self._filter_suppliers_by_price(suppliers, ean)

        if not qualified_suppliers:
            logger.warning(
                f"No suppliers for EAN {ean} have prices better than stock average"
            )
            return []

        # Sort by price (best first)
        sorted_suppliers = sorted(qualified_suppliers, key=lambda x: x.price)

        # Check if best supplier can fulfill full order
        best_supplier = sorted_suppliers[0]
        best_supplier_qty = getattr(best_supplier, "quantity", None)

        # Single supplier allocation
        if best_supplier_qty is None or best_supplier_qty >= quantity:
            return [
                {
                    "ean_code": ean,
                    "product_name": best_supplier.product_name or "Unknown",
                    "quantity": quantity,
                    "original_quantity_needed": quantity,  # TRACK ORIGINAL QUANTITY
                    "unit_price": best_supplier.price,
                    "total_price": best_supplier.price * quantity,
                    "supplier": best_supplier.supplier,
                    "allocation_type": "single",
                    "is_split_order": False,
                    "original_reference": ean,
                    "stock_avg_price": self._get_stock_average_price(ean),
                    "supplier_quantity": best_supplier_qty,
                    "price_comparison": self._get_price_comparison_info(
                        ean, best_supplier.price
                    ),
                }
            ]

        # Multi-supplier allocation needed
        logger.info(
            f"🔄 EAN {ean}: Need {quantity}, best supplier has {best_supplier_qty} - splitting order"
        )

        allocations = []
        remaining_quantity = quantity

        for rank, supplier in enumerate(sorted_suppliers, 1):
            supplier_qty = getattr(supplier, "quantity", None)

            if supplier_qty is None or supplier_qty <= 0:
                continue

            allocated_qty = min(supplier_qty, remaining_quantity)

            if allocated_qty > 0:
                allocation = {
                    "ean_code": ean,
                    "product_name": supplier.product_name or "Unknown",
                    "quantity": allocated_qty,
                    "original_quantity_needed": quantity,  # TRACK ORIGINAL QUANTITY
                    "unit_price": supplier.price,
                    "total_price": supplier.price * allocated_qty,
                    "supplier": supplier.supplier,
                    "allocation_type": "split",
                    "is_split_order": True,
                    "split_rank": rank,
                    "remaining_after_allocation": max(
                        0, remaining_quantity - allocated_qty
                    ),
                    "original_reference": ean,
                    "stock_avg_price": self._get_stock_average_price(ean),
                    "supplier_quantity": supplier_qty,
                    "allocation_percentage": (allocated_qty / quantity * 100),
                    "price_comparison": self._get_price_comparison_info(
                        ean, supplier.price
                    ),
                }

                allocations.append(allocation)
                remaining_quantity -= allocated_qty

                if remaining_quantity <= 0:
                    break

        return allocations

    def _get_best_buy_price(self, ean: str) -> Optional[float]:
        """Get best buy price for EAN from internal data"""
        internal_product = self.internal_data_lookup.get(ean)
        if internal_product and internal_product.get("best_buy_price"):
            return float(internal_product["best_buy_price"])
        return None

    def _get_price_comparison_info(self, ean: str, quote_price: float) -> Dict:
        """Get enhanced price comparison information for transparency"""
        internal_product = self.internal_data_lookup.get(ean, {})

        stock_avg_price = internal_product.get("stock_avg_price")
        supplier_price = internal_product.get("supplier_price")
        best_buy_price = internal_product.get("best_buy_price")

        comparison = {
            "stock_avg_price": stock_avg_price,
            "supplier_price": supplier_price,
            "best_buy_price": best_buy_price,  # INCLUDE BEST BUY PRICE
            "beats_stock_avg": stock_avg_price and quote_price < stock_avg_price,
            "beats_supplier": supplier_price and quote_price < supplier_price,
            "beats_best_buy": best_buy_price and quote_price < best_buy_price,  # NEW
        }

        if stock_avg_price:
            comparison["savings_vs_stock_avg"] = stock_avg_price - quote_price

        if best_buy_price:
            comparison["savings_vs_best_buy"] = best_buy_price - quote_price  # NEW

        return comparison

    def export_enhanced_orders(self) -> Dict[str, str]:
        """Export enhanced order files with allocation details"""
        if not self.optimization_results:
            return {}

        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for supplier, items in self.optimization_results["supplier_orders"].items():
            if not items:
                continue

            # Prepare enhanced CSV data
            csv_data = []
            csv_data.append("EAN;Product_Name;Quantity;Unit_Price;Total_Price")

            total_value = 0
            split_items = 0
            single_items = 0

            for item in items:
                # Calculate savings vs stock average
                stock_avg_price = item.get("stock_avg_price", "")
                savings_vs_stock_avg = ""

                if stock_avg_price and isinstance(stock_avg_price, (int, float)):
                    stock_avg_price = f"{stock_avg_price:.2f}".replace(".", ",")

                csv_row = [
                    item["ean_code"],
                    item["product_name"].replace(";", ","),
                    str(item["quantity"]),
                    f"{item['unit_price']:.2f}".replace(".", ","),
                    f"{item['total_price']:.2f}".replace(".", ","),
                ]

                csv_data.append(";".join(csv_row))
                total_value += item["total_price"]

            # Create filename and content
            safe_supplier = "".join(
                c for c in supplier if c.isalnum() or c in (" ", "-", "_")
            ).replace(" ", "_")
            filename = f"enhanced_order_{safe_supplier}_{timestamp}.csv"
            csv_content = "\n".join(csv_data)

            exported_files[filename] = csv_content

        return exported_files

    # =============================================================================
    # ENHANCED VERSIONS OF EXISTING FUNCTIONS
    # =============================================================================

    def load_supplier_data(self, products: List[ProductData]) -> None:
        """ENHANCED: Load supplier data with better EAN normalization"""
        self.supplier_data = products
        self.ean_lookup = defaultdict(list)

        normalized_count = 0
        original_count = 0

        for product in products:
            if product.ean_code:
                # Store original EAN
                original_ean = str(product.ean_code).strip()
                self.ean_lookup[original_ean].append(product)
                original_count += 1

                # Also store normalized EAN
                normalized_ean = self._normalize_ean_code(original_ean)
                if normalized_ean and normalized_ean != original_ean:
                    self.ean_lookup[normalized_ean].append(product)
                    normalized_count += 1

        logger.info(f"📊 Enhanced EAN Lookup created:")
        logger.info(f"   Original EAN codes: {original_count}")
        logger.info(f"   Normalized EAN codes: {normalized_count}")
        logger.info(f"   Total lookup entries: {len(self.ean_lookup)}")

    def _process_buying_item(self, row: pd.Series, structure: Dict) -> Dict:
        """ENHANCED: Process buying item with smart multi-supplier allocation"""
        # Extract data (keep existing logic)
        ean_code = None
        quantity = 1
        description = None
        original_reference = None

        # Extract EAN/product code with normalization
        if structure["ean_column"]:
            raw_code = row.get(structure["ean_column"])
            if pd.notna(raw_code):
                original_reference = str(raw_code).strip()
                ean_code = self._normalize_ean_code(raw_code)

        # Extract quantity
        if structure["quantity_column"]:
            qty_value = row.get(structure["quantity_column"])
            if pd.notna(qty_value):
                try:
                    quantity = int(float(qty_value))
                except:
                    quantity = 1

        # Extract description
        if structure["description_column"]:
            desc_value = row.get(structure["description_column"])
            if pd.notna(desc_value):
                description = str(desc_value).strip()

        # Find matching suppliers
        matched_suppliers = []

        if ean_code and ean_code in self.ean_lookup:
            matched_suppliers = self.ean_lookup[ean_code]
        elif original_reference and original_reference in self.ean_lookup:
            matched_suppliers = self.ean_lookup[original_reference]
            ean_code = original_reference

        if not matched_suppliers:
            return {
                "matched": False,
                "original_reference": original_reference or description or "Unknown",
                "quantity": quantity,
                "reason": "No matching EAN code found in supplier catalogs",
            }

        # SMART ALLOCATION: Use multi-supplier logic
        allocations = self._smart_allocate_order(ean_code, quantity, matched_suppliers)

        if not allocations:
            return {
                "matched": False,
                "original_reference": original_reference,
                "quantity": quantity,
                "reason": "No suppliers with prices better than stock average",
                "available_suppliers": len(matched_suppliers),
                "stock_avg_price": self._get_stock_average_price(ean_code),
            }

        # Return the allocations
        return {
            "matched": True,
            "allocations": allocations,
            "total_suppliers": len(matched_suppliers),
            "qualified_suppliers": len(
                self._filter_suppliers_by_price(matched_suppliers, ean_code)
            ),
            "is_split_order": len(allocations) > 1,
            "original_reference": original_reference,
        }

    def _process_single_list(self, buying_list: Dict) -> Dict:
        """ENHANCED: Process buying list with smart allocation support"""
        df = buying_list["dataframe"]
        structure = buying_list["structure"]

        result = {
            "list_name": buying_list["name"],
            "total_items": len(df),
            "matched_items": 0,
            "unmatched_items": 0,
            "processed_items": [],
            "unmatched_items_list": [],
            "supplier_orders": defaultdict(list),
            "split_orders_count": 0,
            "single_orders_count": 0,
            "total_allocations": 0,
        }

        for idx, row in df.iterrows():
            item_result = self._process_buying_item(row, structure)

            if item_result["matched"]:
                result["matched_items"] += 1

                # Handle multiple allocations from smart allocation
                allocations = item_result.get("allocations", [])

                if len(allocations) > 1:
                    result["split_orders_count"] += 1
                else:
                    result["single_orders_count"] += 1

                result["total_allocations"] += len(allocations)

                # Add each allocation to appropriate supplier order
                for allocation in allocations:
                    supplier = allocation["supplier"]
                    result["supplier_orders"][supplier].append(allocation)
                    result["processed_items"].append(allocation)

            else:
                result["unmatched_items"] += 1
                result["unmatched_items_list"].append(item_result)

        return result

    def get_optimization_summary(self) -> Dict:
        """ENHANCED: Summary with smart allocation metrics"""
        if not self.optimization_results:
            return {"error": "No optimization results available"}

        results = self.optimization_results

        total_value = 0
        supplier_stats = {}
        total_split_orders = 0
        total_single_orders = 0
        total_allocations = 0

        for supplier, items in results["supplier_orders"].items():
            supplier_value = sum(item["total_price"] for item in items)
            supplier_stats[supplier] = {
                "items": len(items),
                "total_value": supplier_value,
                "unique_products": len(set(item["ean_code"] for item in items)),
                "split_order_items": sum(
                    1 for item in items if item.get("is_split_order", False)
                ),
                "single_order_items": sum(
                    1 for item in items if not item.get("is_split_order", False)
                ),
            }
            total_value += supplier_value

        # Count unique orders vs allocations
        for list_result in results.get("lists_processed", []):
            total_split_orders += list_result.get("split_orders_count", 0)
            total_single_orders += list_result.get("single_orders_count", 0)
            total_allocations += list_result.get("total_allocations", 0)

        return {
            "total_items_requested": results["total_items"],
            "matched_items": results["matched_items"],
            "unmatched_items": results["unmatched_items"],
            "match_rate": (
                (results["matched_items"] / results["total_items"] * 100)
                if results["total_items"] > 0
                else 0
            ),
            "total_order_value": total_value,
            "suppliers_involved": len(supplier_stats),
            "supplier_breakdown": supplier_stats,
            "lists_processed": len(results["lists_processed"]),
            # Enhanced allocation metrics
            "split_orders_count": total_split_orders,
            "single_orders_count": total_single_orders,
            "total_allocations": total_allocations,
            "allocation_efficiency": (
                (total_allocations / results["matched_items"])
                if results["matched_items"] > 0
                else 1
            ),
        }

    # =============================================================================
    # KEEP ALL EXISTING FUNCTIONS UNCHANGED
    # =============================================================================

    def add_buying_list(
        self, uploaded_file_or_df, list_name: str, header_row: int = 3
    ) -> Dict:
        """ENHANCED add_buying_list with robust CSV parsing and duplicate prevention"""

        # Check if buying list with this name already exists
        existing_list = next(
            (bl for bl in self.buying_lists if bl["name"] == list_name), None
        )
        if existing_list:
            logger.info(
                f"📋 Buying list '{list_name}' already loaded, skipping duplicate"
            )
            return {
                "success": True,
                "message": f"Buying list '{list_name}' already loaded (skipped duplicate)",
                "structure": existing_list["structure"],
                "total_items": existing_list["total_items"],
                "sample_data": (
                    existing_list["dataframe"].head(3).to_dict("records")
                    if len(existing_list["dataframe"]) > 0
                    else []
                ),
                "columns": list(existing_list["dataframe"].columns),
                "header_row_used": existing_list["header_row"],
                "rows_cleaned": existing_list.get("rows_cleaned", 0),
                "already_loaded": True,
            }

        # Handle both uploaded files and pre-parsed DataFrames
        if hasattr(uploaded_file_or_df, "read"):  # It's an uploaded file
            try:
                logger.info(
                    f"🎯 Processing file '{list_name}' with ROBUST parsing (header row {header_row})"
                )
                df = self.parse_csv_file_robust(uploaded_file_or_df, header_row)
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to parse CSV file: {str(e)}",
                    "error": str(e),
                }
        else:  # It's already a DataFrame
            df = uploaded_file_or_df

        # Enhanced column name cleaning
        original_columns = list(df.columns)
        df.columns = [
            col.rstrip(";").rstrip(",").strip().replace("\n", "").replace("\r", "")
            for col in df.columns
        ]

        # Remove empty rows
        initial_rows = len(df)
        df = df.dropna(how="all").reset_index(drop=True)
        final_rows = len(df)

        # Enhanced structure detection
        structure = self._analyze_buying_list_structure_enhanced(df)

        buying_list = {
            "name": list_name,
            "dataframe": df,
            "structure": structure,
            "total_items": len(df),
            "processed_items": [],
            "unmatched_items": [],
            "header_row": header_row,
            "rows_cleaned": initial_rows - final_rows,
        }

        self.buying_lists.append(buying_list)
        logger.info(f"✅ Added new buying list '{list_name}' with {len(df)} items")

        return {
            "success": True,
            "message": f"Buying list '{list_name}' processed successfully",
            "structure": structure,
            "total_items": len(df),
            "sample_data": df.head(3).to_dict("records") if len(df) > 0 else [],
            "columns": list(df.columns),
            "header_row_used": header_row,
            "rows_cleaned": initial_rows - final_rows,
            "already_loaded": False,
        }

    @staticmethod
    def parse_csv_file_robust(uploaded_file, header_row: int = 3) -> pd.DataFrame:
        """KEEP: Enhanced CSV parsing with scientific notation handling"""
        uploaded_file.seek(0)
        skiprows = header_row - 1

        parsing_strategies = [
            {"sep": ";", "encoding": "cp1252", "on_bad_lines": "skip", "dtype": str},
            {"sep": ";", "encoding": "utf-8", "on_bad_lines": "skip", "dtype": str},
            {"sep": ",", "encoding": "cp1252", "on_bad_lines": "skip", "dtype": str},
            {"sep": ";", "encoding": "cp1252", "on_bad_lines": "skip"},
            {"sep": ",", "encoding": "cp1252", "on_bad_lines": "skip"},
            {"sep": None, "encoding": "cp1252", "engine": "python"},
        ]

        for i, strategy in enumerate(parsing_strategies):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, skiprows=skiprows, **strategy)
                if len(df.columns) > 0 and len(df) > 0:
                    return df
            except Exception:
                continue

        # Final fallback
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=";", skiprows=skiprows, encoding="cp1252")

    @staticmethod
    def _manual_parse_with_field_normalization(
        uploaded_file, header_row: int
    ) -> pd.DataFrame:
        """Manual parsing that handles inconsistent field counts"""
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode("cp1252")
            lines = [line.rstrip("\r") for line in content.split("\n")]

            logger.info(
                f"📄 Manual parsing: {len(lines)} lines found, using row {header_row} as header"
            )

            if len(lines) >= header_row:
                # Use specified row as header
                header_line = lines[header_row - 1].strip()
                data_lines = [line for line in lines[header_row:] if line.strip()]

                logger.info(f"🎯 Header content: '{header_line}'")

                # Parse header
                headers = [col.strip() for col in header_line.split(";")]
                headers = [col for col in headers if col]  # Remove empty
                expected_field_count = len(headers)

                logger.info(f"📊 Expected {expected_field_count} fields: {headers}")

                # Parse and normalize data lines
                normalized_data = []
                fixed_lines = 0

                for line_num, line in enumerate(data_lines, header_row + 1):
                    try:
                        fields = [field.strip() for field in line.split(";")]

                        if len(fields) > expected_field_count:
                            # TOO MANY FIELDS - Fix this
                            logger.warning(
                                f"🔧 Line {line_num}: {len(fields)} fields, expected {expected_field_count} - FIXING"
                            )

                            # Keep first (expected_field_count - 1) fields
                            normalized_fields = fields[: expected_field_count - 1]

                            # Merge excess fields into the last column
                            merged_last_field = ";".join(
                                fields[expected_field_count - 1 :]
                            )
                            normalized_fields.append(merged_last_field)

                            logger.debug(f"   Fixed line: {normalized_fields}")
                            normalized_data.append(normalized_fields)
                            fixed_lines += 1

                        elif len(fields) < expected_field_count:
                            # TOO FEW FIELDS - Pad with empty strings
                            logger.warning(
                                f"🔧 Line {line_num}: {len(fields)} fields, expected {expected_field_count} - PADDING"
                            )

                            while len(fields) < expected_field_count:
                                fields.append("")
                            normalized_data.append(fields)
                            fixed_lines += 1

                        else:
                            # CORRECT FIELD COUNT
                            normalized_data.append(fields)

                    except Exception as e:
                        logger.warning(f"⚠️ Skipping problematic line {line_num}: {e}")
                        continue

                if normalized_data:
                    df = pd.DataFrame(normalized_data, columns=headers)

                    logger.info(f"✅ FIELD NORMALIZATION SUCCESSFUL:")
                    logger.info(f"   Headers: {headers}")
                    logger.info(f"   Data rows: {len(normalized_data)}")
                    logger.info(f"   Lines fixed: {fixed_lines}")

                    return df
                else:
                    raise Exception("No valid data rows found after normalization")

            else:
                raise Exception(
                    f"File only has {len(lines)} lines, cannot use row {header_row} as header"
                )

        except Exception as e:
            logger.error(f"Manual parsing failed: {e}")
            raise ValueError(f"Could not parse CSV with field normalization: {e}")

    def _analyze_buying_list_structure_enhanced(self, df: pd.DataFrame) -> Dict:
        """KEEP: Enhanced structure analysis"""
        columns = df.columns.tolist()
        structure = {
            "detected_columns": columns,
            "ean_column": None,
            "quantity_column": None,
            "description_column": None,
            "confidence": "low",
        }

        # Enhanced EAN detection patterns
        ean_patterns = [
            ("ean", 100),
            ("gencod", 100),
            ("gencode", 100),
            ("codigo", 100),
            ("código", 100),
            ("code", 80),
            ("ref", 70),
            ("sku", 70),
            ("barcode", 90),
            ("item", 40),
            ("produto", 50),
            ("article", 50),
        ]

        best_ean_match = None
        best_ean_score = 0

        for col in columns:
            col_clean = col.lower().strip()
            for pattern, base_score in ean_patterns:
                if pattern in col_clean:
                    score = base_score
                    if col_clean == pattern:
                        score += 20
                    if any(term in col_clean for term in ["ean", "gencod", "codigo"]):
                        score += 30
                    if score > best_ean_score:
                        best_ean_score = score
                        best_ean_match = col
                    break

        if best_ean_match:
            structure["ean_column"] = best_ean_match
            structure["confidence"] = "high" if best_ean_score >= 90 else "medium"

        # Quantity detection
        qty_patterns = [
            "qnt",
            "qty",
            "quantity",
            "quantidade",
            "qtd",
            "quant",
            "amount",
            "units",
        ]
        for col in columns:
            col_clean = col.lower().strip()
            if any(pattern in col_clean for pattern in qty_patterns):
                structure["quantity_column"] = col
                break

        # Description detection
        desc_patterns = [
            "descri",
            "description",
            "desc",
            "name",
            "nome",
            "produto",
            "product",
        ]
        for col in columns:
            col_clean = col.lower().strip()
            if any(pattern in col_clean for pattern in desc_patterns):
                structure["description_column"] = col
                break

        return structure

    def _normalize_ean_code(self, raw_ean) -> str:
        """KEEP: Enhanced EAN normalization with scientific notation handling"""
        if not raw_ean or pd.isna(raw_ean):
            return ""

        ean_str = str(raw_ean).strip()

        # Handle scientific notation
        if "E+" in ean_str.upper() or "e+" in ean_str:
            try:
                normalized_str = ean_str.replace(",", ".")
                float_val = float(normalized_str)
                result = str(int(float_val))
                return result
            except (ValueError, OverflowError):
                ean_str = "".join(c for c in ean_str if c.isdigit())

        # Handle Excel .0 suffix
        elif ean_str.endswith(".0"):
            ean_str = ean_str[:-2]

        # Remove non-digit characters
        ean_digits = "".join(c for c in ean_str if c.isdigit())

        if ean_digits and 8 <= len(ean_digits) <= 14:
            return ean_digits
        elif len(ean_digits) > 14:
            return ean_digits[:13]

        return ean_digits if ean_digits else ""

    def process_buying_lists(self) -> Dict:
        """KEEP: Main orchestration function"""
        if not self.supplier_data:
            return {
                "error": "No supplier data loaded. Process supplier catalogs first."
            }

        if not self.buying_lists:
            return {"error": "No buying lists uploaded."}

        results = {
            "total_items": 0,
            "matched_items": 0,
            "unmatched_items": 0,
            "lists_processed": [],
            "supplier_orders": defaultdict(list),
        }

        for buying_list in self.buying_lists:
            list_result = self._process_single_list(buying_list)
            results["lists_processed"].append(list_result)
            results["total_items"] += list_result["total_items"]
            results["matched_items"] += list_result["matched_items"]
            results["unmatched_items"] += list_result["unmatched_items"]

            # Aggregate supplier orders
            for supplier, items in list_result["supplier_orders"].items():
                results["supplier_orders"][supplier].extend(items)

        self.optimization_results = results
        return results

    def generate_supplier_orders(self) -> Dict[str, pd.DataFrame]:
        """KEEP: Generate order DataFrames"""
        if not self.optimization_results:
            return {}

        supplier_orders = {}
        for supplier, items in self.optimization_results["supplier_orders"].items():
            if not items:
                continue

            df_data = []
            total_value = 0

            for item in items:
                df_data.append(
                    {
                        "EAN Code": item["ean_code"],
                        "Product Name": item["product_name"],
                        "Quantity": item["quantity"],
                        "Unit Price": f"€{item['unit_price']:.2f}",
                        "Total Price": f"€{item['total_price']:.2f}",
                        "Reference": item.get("original_reference", ""),
                    }
                )
                total_value += item["total_price"]

            # Add summary row
            df_data.append(
                {
                    "EAN Code": "",
                    "Product Name": "TOTAL ORDER",
                    "Quantity": sum(item["quantity"] for item in items),
                    "Unit Price": "",
                    "Total Price": f"€{total_value:.2f}",
                    "Reference": "",
                }
            )

            supplier_orders[supplier] = pd.DataFrame(df_data)

        return supplier_orders

    def export_orders(self) -> Dict[str, str]:
        """KEEP: Basic export function"""
        supplier_orders = self.generate_supplier_orders()
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for supplier, df in supplier_orders.items():
            filename = f"order_{supplier.replace(' ', '_')}_{timestamp}.csv"
            csv_content = df.to_csv(index=False)
            exported_files[filename] = csv_content

        return exported_files

    def get_unmatched_items(self) -> pd.DataFrame:
        """KEEP: Get unmatched items"""
        if not self.optimization_results:
            return pd.DataFrame()

        unmatched_data = []
        for list_result in self.optimization_results["lists_processed"]:
            for item in list_result["unmatched_items_list"]:
                unmatched_data.append(
                    {
                        "List Name": list_result["list_name"],
                        "Reference": item["original_reference"],
                        "Quantity": item["quantity"],
                        "Reason": item["reason"],
                    }
                )

        return pd.DataFrame(unmatched_data)

    def _find_ean_by_description(self, description: str) -> Optional[str]:
        """KEEP: Find EAN by description matching"""
        if not description:
            return None

        description_lower = description.lower().strip()

        for ean_code, products in self.ean_lookup.items():
            for product in products:
                if product.product_name:
                    product_name_lower = product.product_name.lower()
                    if (
                        description_lower in product_name_lower
                        or product_name_lower in description_lower
                    ):
                        return ean_code

        return None
