"""
Order optimizer with enhanced CSV parsing for malformed files and smart allocation
UPDATED: Smart price-based allocation using stock average price filtering
"""

import pandas as pd
import logging
from collections import defaultdict
from typing import List, Dict, Optional
from datetime import datetime
from models import ProductData
import streamlit as st
from utils import pad_ean_code

logger = logging.getLogger(__name__)


class OrderOptimizer:
    """Order optimizer with enhanced CSV parsing and smart price-based allocation"""

    def __init__(self):
        self.supplier_data = []
        self.buying_lists = []
        self.optimization_results = {}

    def load_supplier_data(self, products: List[ProductData]) -> None:
        """Enhanced supplier data loading with normalized EAN codes"""
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

                # Also store normalized EAN (for scientific notation compatibility)
                normalized_ean = self._normalize_ean_code(original_ean)
                if normalized_ean and normalized_ean != original_ean:
                    self.ean_lookup[normalized_ean].append(product)
                    normalized_count += 1
                    logger.debug(
                        f"🔧 EAN normalization: '{original_ean}' → '{normalized_ean}'"
                    )

        logger.info(f"📊 EAN Lookup created:")
        logger.info(f"   Original EAN codes: {original_count}")
        logger.info(f"   Normalized EAN codes: {normalized_count}")
        logger.info(f"   Total lookup entries: {len(self.ean_lookup)}")

    def _normalize_ean_code(self, raw_ean) -> str:
        """Handle scientific notation EAN codes"""
        if not raw_ean or pd.isna(raw_ean):
            return ""

        # Convert to string and clean
        ean_str = str(raw_ean).strip()

        # CRITICAL FIX: Handle scientific notation (e.g., "3,33787E+12")
        if "E+" in ean_str.upper() or "e+" in ean_str:
            try:
                logger.info(f"🔧 Converting scientific notation: {ean_str}")
                # Replace comma with dot for proper float parsing
                normalized_str = ean_str.replace(",", ".")
                # Convert scientific notation to regular number
                float_val = float(normalized_str)
                # Convert to integer (removes decimals) then to string
                result = str(int(float_val))
                logger.info(f"✅ Converted to: {result}")
                return result
            except (ValueError, OverflowError) as e:
                logger.warning(f"❌ Scientific notation conversion failed: {e}")
                # Fallback: try to extract digits manually
                ean_str = "".join(c for c in ean_str if c.isdigit())

        # Handle Excel .0 suffix
        elif ean_str.endswith(".0"):
            ean_str = ean_str[:-2]

        # Remove any non-digit characters (spaces, dashes, commas)
        ean_digits = "".join(c for c in ean_str if c.isdigit())

        # Ensure reasonable EAN length (8-14 digits)
        if ean_digits and 8 <= len(ean_digits) <= 14:
            return ean_digits
        elif len(ean_digits) > 14:
            # Truncate if too long (can happen with scientific notation precision issues)
            return ean_digits[:13]

        return ean_digits if ean_digits else ""

    @staticmethod
    def parse_csv_file_robust(uploaded_file, header_row: int = 3) -> pd.DataFrame:
        """Enhanced CSV parsing that prevents scientific notation and handles malformed files"""
        uploaded_file.seek(0)
        skiprows = header_row - 1

        logger.info(
            f"🎯 ENHANCED PARSING: Using row {header_row} as header (preventing scientific notation)"
        )

        # Strategy 1: Force all columns as strings to prevent scientific notation
        parsing_strategies = [
            # CRITICAL: dtype=str prevents scientific notation!
            {"sep": ";", "encoding": "cp1252", "on_bad_lines": "skip", "dtype": str},
            {"sep": ";", "encoding": "utf-8", "on_bad_lines": "skip", "dtype": str},
            {"sep": ",", "encoding": "cp1252", "on_bad_lines": "skip", "dtype": str},
            # Fallback strategies without dtype
            {"sep": ";", "encoding": "cp1252", "on_bad_lines": "skip"},
            {"sep": ",", "encoding": "cp1252", "on_bad_lines": "skip"},
            {"sep": None, "encoding": "cp1252", "engine": "python"},
        ]

        for i, strategy in enumerate(parsing_strategies):
            try:
                uploaded_file.seek(0)
                logger.info(f"🔧 Trying strategy {i+1}: {strategy}")

                df = pd.read_csv(uploaded_file, skiprows=skiprows, **strategy)

                if len(df.columns) > 0 and len(df) > 0:
                    logger.info(f"✅ SUCCESS with strategy {i+1}")
                    logger.info(f"   Headers: {list(df.columns)}")
                    logger.info(f"   Rows: {len(df)}")

                    # Show sample data to verify no scientific notation
                    if len(df) > 0:
                        first_row = df.iloc[0].to_dict()
                        logger.info(f"   Sample data: {first_row}")

                    return df

            except Exception as e:
                logger.debug(f"Strategy {i+1} failed: {e}")
                continue

        # Final fallback - try the existing manual parsing method if it exists
        try:
            logger.info(f"🔧 Using manual parsing fallback...")
            return OrderOptimizer._manual_parse_with_field_normalization(
                uploaded_file, header_row
            )
        except AttributeError:
            # If the manual method doesn't exist, use basic parsing
            logger.info(f"🔧 Using basic parsing fallback...")
            uploaded_file.seek(0)
            return pd.read_csv(
                uploaded_file, sep=";", skiprows=skiprows, encoding="cp1252"
            )

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
                            # TOO MANY FIELDS - This is your exact error case!
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

    def add_buying_list(
        self, uploaded_file_or_df, list_name: str, header_row: int = 3
    ) -> Dict:
        """Enhanced add_buying_list with robust CSV parsing"""

        # Handle both uploaded files and pre-parsed DataFrames
        if hasattr(uploaded_file_or_df, "read"):  # It's an uploaded file
            try:
                logger.info(
                    f"🎯 Processing file '{list_name}' with ROBUST parsing (header row {header_row})"
                )
                # Use enhanced robust parsing instead of the old method
                df = self.parse_csv_file_robust(uploaded_file_or_df, header_row)

            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to parse CSV file with robust parser (header row {header_row}): {str(e)}",
                    "error": str(e),
                    "suggestion": "The file may have inconsistent field counts. Try adjusting the header row number or check file format.",
                    "debug_info": {
                        "attempted_header_row": header_row,
                        "parsing_method": "robust_enhanced",
                        "common_causes": [
                            "Extra commas/semicolons in product descriptions",
                            "Inconsistent column counts between rows",
                            "Wrong header row number",
                            "File encoding issues",
                        ],
                    },
                }
        else:  # It's already a DataFrame
            df = uploaded_file_or_df

        # ENHANCED: Better column name cleaning
        original_columns = list(df.columns)
        df.columns = [
            col.rstrip(";").rstrip(",").strip().replace("\n", "").replace("\r", "")
            for col in df.columns
        ]

        logger.info(
            f"🧹 Enhanced column cleaning: {original_columns} → {list(df.columns)}"
        )

        # Remove empty rows and count them
        initial_rows = len(df)
        df = df.dropna(how="all").reset_index(drop=True)
        final_rows = len(df)

        if initial_rows != final_rows:
            logger.info(f"🧹 Removed {initial_rows - final_rows} empty rows")

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
            "parsing_method": "robust_enhanced",  # Track which method was used
            "rows_cleaned": initial_rows - final_rows,
        }

        self.buying_lists.append(buying_list)

        logger.info(
            f"📋 Enhanced processing '{list_name}': {len(df)} items, structure: {structure}"
        )

        return {
            "success": True,
            "message": f"Buying list '{list_name}' processed successfully with robust parsing (header row {header_row})",
            "structure": structure,
            "total_items": len(df),
            "sample_data": df.head(3).to_dict("records") if len(df) > 0 else [],
            "columns": list(df.columns),
            "header_row_used": header_row,
            "rows_cleaned": initial_rows - final_rows,
            "parsing_method": "robust_enhanced",
            "success_details": {
                "field_normalization": "Applied automatic field count correction",
                "bad_lines_handled": "Problematic lines were automatically fixed or skipped",
                "encoding_detected": "Automatic encoding detection used",
            },
        }

    def _analyze_buying_list_structure_enhanced(self, df: pd.DataFrame) -> Dict:
        """Enhanced structure analysis with better pattern matching for EAN detection"""

        columns = df.columns.tolist()
        structure = {
            "detected_columns": columns,
            "ean_column": None,
            "quantity_column": None,
            "description_column": None,
            "confidence": "low",
        }

        logger.info(f"🔍 ENHANCED ANALYSIS: {columns}")

        # Enhanced EAN detection patterns with scoring
        ean_patterns = [
            # Exact matches (highest priority)
            ("ean", 100),
            ("gencod", 100),
            ("gencode", 100),
            ("codigo", 100),
            ("código", 100),
            # Strong partial matches
            ("code", 80),
            ("ref", 70),
            ("sku", 70),
            ("barcode", 90),
            # Weaker matches
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

                    # Bonus for exact matches
                    if col_clean == pattern:
                        score += 20

                    # Bonus for EAN-specific terms
                    if any(term in col_clean for term in ["ean", "gencod", "codigo"]):
                        score += 30

                    if score > best_ean_score:
                        best_ean_score = score
                        best_ean_match = col

                    logger.info(
                        f"✅ EAN pattern '{pattern}' found in '{col}' (score: {score})"
                    )
                    break

        if best_ean_match:
            structure["ean_column"] = best_ean_match
            structure["confidence"] = "high" if best_ean_score >= 90 else "medium"
            logger.info(
                f"🎯 Best EAN match: '{best_ean_match}' (score: {best_ean_score})"
            )

        # Enhanced quantity detection
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
                logger.info(f"✅ Quantity match: '{col}'")
                break

        # Enhanced description detection
        desc_patterns = [
            "descri",
            "description",
            "desc",
            "name",
            "nome",
            "produto",
            "product",
            "designation",
            "libelle",
            "titre",
            "item",
        ]
        best_desc_match = None
        best_desc_score = 0

        for col in columns:
            col_clean = col.lower().strip()
            for pattern in desc_patterns:
                if pattern in col_clean:
                    score = 50
                    if col_clean == pattern:
                        score += 20
                    if "descri" in col_clean:
                        score += 15

                    if score > best_desc_score:
                        best_desc_score = score
                        best_desc_match = col

        if best_desc_match:
            structure["description_column"] = best_desc_match
            logger.info(
                f"✅ Description match: '{best_desc_match}' (score: {best_desc_score})"
            )

        # Final confidence calculation
        detected_fields = sum(
            1
            for field in [
                structure["ean_column"],
                structure["quantity_column"],
                structure["description_column"],
            ]
            if field
        )

        if structure["ean_column"] and detected_fields >= 2:
            structure["confidence"] = "high"
        elif structure["ean_column"]:
            structure["confidence"] = "medium"
        else:
            structure["confidence"] = "low"

        logger.info(f"📊 ENHANCED FINAL STRUCTURE: {structure}")
        return structure

    def _get_stock_average_price(self, ean: str) -> Optional[float]:
        """Get stock average price from internal data if available"""
        if (
            hasattr(st.session_state, "opportunity_engine")
            and st.session_state.opportunity_engine
        ):
            engine = st.session_state.opportunity_engine
            if hasattr(engine, "internal_data") and engine.internal_data:
                for product in engine.internal_data:
                    if str(product.get("ean", "")).strip() == ean:
                        return product.get("stock_avg_price")
        return None

    def _smart_price_based_allocation_for_ean(
        self, ean: str, ean_opportunities: List[Dict]
    ) -> List[Dict]:
        """
        Enhanced allocation that considers stock average price for split orders
        Only uses suppliers with prices better than stock average when splitting
        """

        # Sort suppliers by price (best first)
        sorted_suppliers = sorted(
            ean_opportunities, key=lambda x: x.get("quote_price", float("inf"))
        )

        # Get the net need (same for all suppliers of this EAN)
        net_need = sorted_suppliers[0].get("net_need", 0)

        if net_need <= 0:
            # No net need - just return the best supplier
            best_opp = sorted_suppliers[0]
            best_opp["allocation_type"] = "no_need"
            best_opp["is_split_order"] = False
            return [best_opp]

        # Get stock average price for smart allocation
        stock_avg_price = self._get_stock_average_price(ean)

        # Check if best supplier can fulfill full need
        best_supplier = sorted_suppliers[0]
        best_supplier_qty = best_supplier.get("supplier_quantity")

        # Determine if we can fulfill with best supplier alone
        if best_supplier_qty is None:
            # Quantity unknown - assume we can buy all we need
            purchasable_qty = net_need
            can_fulfill_fully = True
        elif best_supplier_qty >= net_need:
            # Sufficient quantity available
            purchasable_qty = net_need
            can_fulfill_fully = True
        else:
            # Insufficient quantity - need smart allocation
            purchasable_qty = best_supplier_qty
            can_fulfill_fully = False

        if can_fulfill_fully:
            # Best supplier can fulfill full order
            best_opp = best_supplier.copy()
            best_opp["allocation_type"] = "single_best"
            best_opp["is_split_order"] = False
            best_opp["allocated_quantity"] = purchasable_qty

            # Calculate total savings based on actual purchasable quantity
            savings_per_unit = best_opp.get("savings_per_unit", 0)
            best_opp["total_savings"] = savings_per_unit * purchasable_qty
            best_opp["purchasable_quantity"] = purchasable_qty

            return [best_opp]

        # SMART ALLOCATION: Only use suppliers better than stock average price
        logger.info(
            f"🧠 EAN {ean}: Smart allocation needed (best supplier has {best_supplier_qty}, need {net_need})"
        )

        if stock_avg_price:
            logger.info(
                f"📊 Stock average price: €{stock_avg_price:.2f} - filtering suppliers"
            )

            # Filter suppliers: only those with prices better than stock average
            eligible_suppliers = []
            for supplier in sorted_suppliers:
                quote_price = supplier.get("quote_price", float("inf"))
                supplier_qty = supplier.get("supplier_quantity", 0)

                # Include if price is better than stock average AND has quantity
                if quote_price < stock_avg_price and supplier_qty and supplier_qty > 0:
                    eligible_suppliers.append(supplier)
                    logger.info(
                        f"✅ {supplier.get('supplier', 'Unknown')}: €{quote_price:.2f} < €{stock_avg_price:.2f} (eligible)"
                    )
                else:
                    if quote_price >= stock_avg_price:
                        logger.info(
                            f"❌ {supplier.get('supplier', 'Unknown')}: €{quote_price:.2f} >= €{stock_avg_price:.2f} (too expensive)"
                        )
                    elif not supplier_qty or supplier_qty <= 0:
                        logger.info(
                            f"❌ {supplier.get('supplier', 'Unknown')}: No quantity available"
                        )
        else:
            # No stock average price available - use all suppliers with quantity
            logger.info(
                "⚠️ No stock average price found - using all suppliers with quantity"
            )
            eligible_suppliers = [
                s for s in sorted_suppliers if s.get("supplier_quantity", 0) > 0
            ]

        if not eligible_suppliers:
            # No eligible suppliers for smart allocation - just use best supplier with available quantity
            logger.warning(
                f"⚠️ No eligible suppliers found for smart allocation, using best available"
            )
            best_available = next(
                (s for s in sorted_suppliers if s.get("supplier_quantity", 0) > 0),
                sorted_suppliers[0],
            )

            allocated_qty = min(
                best_available.get("supplier_quantity", net_need), net_need
            )
            best_available = best_available.copy()
            best_available["allocated_quantity"] = allocated_qty
            best_available["is_split_order"] = False
            best_available["allocation_type"] = "fallback_single"

            # Recalculate savings
            savings_per_unit = best_available.get("savings_per_unit", 0)
            best_available["total_savings"] = savings_per_unit * allocated_qty

            return [best_available]

        # Perform smart allocation across eligible suppliers
        allocated_opportunities = []
        remaining_need = net_need

        for i, supplier in enumerate(eligible_suppliers):
            supplier_qty = supplier.get("supplier_quantity", 0)

            if supplier_qty <= 0:
                continue

            # Calculate allocation for this supplier
            allocated_qty = min(supplier_qty, remaining_need)

            if allocated_qty > 0:
                # Create allocated opportunity
                allocated_opp = supplier.copy()
                allocated_opp["allocated_quantity"] = allocated_qty
                allocated_opp["original_net_need"] = net_need
                allocated_opp["remaining_need_after"] = max(
                    0, remaining_need - allocated_qty
                )
                allocated_opp["is_split_order"] = True
                allocated_opp["split_order_rank"] = i + 1
                allocated_opp["allocation_type"] = "smart_split_order"

                # Add smart allocation metadata
                allocated_opp["smart_allocation_info"] = {
                    "stock_avg_price": stock_avg_price,
                    "quote_price": supplier.get("quote_price", 0),
                    "price_advantage": (
                        stock_avg_price - supplier.get("quote_price", 0)
                        if stock_avg_price
                        else 0
                    ),
                    "eligible_suppliers_count": len(eligible_suppliers),
                    "allocation_reason": (
                        "better_than_stock_avg" if stock_avg_price else "best_available"
                    ),
                }

                # Recalculate savings based on allocated quantity
                savings_per_unit = allocated_opp.get("savings_per_unit", 0)
                allocated_opp["total_savings"] = savings_per_unit * allocated_qty
                allocated_opp["purchasable_quantity"] = allocated_qty

                allocated_opportunities.append(allocated_opp)
                remaining_need -= allocated_qty

                if remaining_need <= 0:
                    break

        # Log smart allocation results
        if allocated_opportunities:
            total_allocated = sum(
                opp.get("allocated_quantity", 0) for opp in allocated_opportunities
            )
            fulfillment_rate = (total_allocated / net_need * 100) if net_need > 0 else 0

            logger.info(f"🎯 Smart allocation completed for EAN {ean}:")
            logger.info(
                f"   Total allocated: {total_allocated}/{net_need} units ({fulfillment_rate:.1f}%)"
            )
            logger.info(f"   Suppliers used: {len(allocated_opportunities)}")
            logger.info(
                f"   All prices better than stock avg: €{stock_avg_price:.2f}"
                if stock_avg_price
                else "   No stock avg price constraint"
            )

            for i, opp in enumerate(allocated_opportunities):
                supplier_name = opp.get("supplier", "Unknown")
                qty = opp.get("allocated_quantity", 0)
                price = opp.get("quote_price", 0)
                advantage = opp.get("smart_allocation_info", {}).get(
                    "price_advantage", 0
                )
                logger.info(
                    f"   {i+1}. {supplier_name}: {qty} units @ €{price:.2f} (saves €{advantage:.2f}/unit vs stock avg)"
                )

        return allocated_opportunities

    def process_buying_lists(self) -> Dict:
        """Process all buying lists and match with supplier data"""

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

    def _process_single_list(self, buying_list: Dict) -> Dict:
        """Process a single buying list"""

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
        }

        for idx, row in df.iterrows():
            item_result = self._process_buying_item(row, structure)

            if item_result["matched"]:
                result["matched_items"] += 1
                result["processed_items"].append(item_result)

                # Add to supplier order
                supplier = item_result["best_supplier"]
                result["supplier_orders"][supplier].append(
                    {
                        "ean_code": item_result["ean_code"],
                        "product_name": item_result["product_name"],
                        "quantity": item_result["quantity"],
                        "unit_price": item_result["unit_price"],
                        "total_price": item_result["total_price"],
                        "original_reference": item_result["original_reference"],
                    }
                )
            else:
                result["unmatched_items"] += 1
                result["unmatched_items_list"].append(item_result)

        return result

    def _process_buying_item(self, row: pd.Series, structure: Dict) -> Dict:
        """Enhanced buying item processing with EAN normalization"""

        # Extract data based on detected structure
        ean_code = None
        quantity = 1
        description = None
        original_reference = None

        # Extract EAN/product code with normalization
        if structure["ean_column"]:
            raw_code = row.get(structure["ean_column"])
            if pd.notna(raw_code):
                original_reference = str(raw_code).strip()
                # Normalize the EAN for matching (handles scientific notation)
                ean_code = self._normalize_ean_code(raw_code)

                # Debug scientific notation conversion
                if "E+" in original_reference.upper() or "e+" in original_reference:
                    logger.info(
                        f"🔧 Item processing: '{original_reference}' → '{ean_code}'"
                    )

        # Extract quantity
        if structure["quantity_column"]:
            qty_value = row.get(structure["quantity_column"])
            if pd.notna(qty_value):
                try:
                    quantity = float(qty_value)
                    if quantity == int(quantity):
                        quantity = int(quantity)
                except:
                    quantity = 1

        # Extract description
        if structure["description_column"]:
            desc_value = row.get(structure["description_column"])
            if pd.notna(desc_value):
                description = str(desc_value).strip()

        # Try to find matching suppliers
        matched_suppliers = []

        # Try normalized EAN first
        if ean_code and ean_code in self.ean_lookup:
            matched_suppliers = self.ean_lookup[ean_code]
            logger.debug(f"✅ Match found for normalized EAN: {ean_code}")
        # Try original reference if normalization didn't find a match
        elif original_reference and original_reference in self.ean_lookup:
            matched_suppliers = self.ean_lookup[original_reference]
            ean_code = original_reference  # Use original if it matched
            logger.debug(f"✅ Match found for original reference: {original_reference}")

        # If still no match, try description matching
        if not matched_suppliers and description:
            found_ean = self._find_ean_by_description(description)
            if found_ean:
                ean_code = found_ean
                matched_suppliers = self.ean_lookup.get(found_ean, [])

        # Find best supplier for this EAN
        if matched_suppliers:
            best_supplier = min(matched_suppliers, key=lambda x: x.price)

            return {
                "matched": True,
                "ean_code": ean_code,
                "original_reference": original_reference,
                "normalized_ean": (
                    self._normalize_ean_code(original_reference)
                    if original_reference
                    else None
                ),
                "product_name": best_supplier.product_name or description,
                "quantity": quantity,
                "unit_price": best_supplier.price,
                "total_price": best_supplier.price * quantity,
                "best_supplier": best_supplier.supplier,
                "total_suppliers": len(matched_suppliers),
                "alternative_suppliers": [
                    {
                        "supplier": s.supplier,
                        "price": s.price,
                        "total": s.price * quantity,
                    }
                    for s in sorted(matched_suppliers, key=lambda x: x.price)
                ],
            }

        # Not matched - include debugging info
        return {
            "matched": False,
            "original_reference": original_reference or description or "Unknown",
            "normalized_ean": (
                self._normalize_ean_code(original_reference)
                if original_reference
                else None
            ),
            "quantity": quantity,
            "reason": "No matching EAN code found in supplier catalogs",
            "debug_info": {
                "extracted_ean": ean_code,
                "original_value": original_reference,
                "available_ean_count": len(self.ean_lookup),
                "structure_confidence": structure.get("confidence", "unknown"),
                "scientific_notation_detected": bool(
                    original_reference
                    and (
                        "E+" in original_reference.upper() or "e+" in original_reference
                    )
                ),
            },
        }

    def _find_ean_by_description(self, description: str) -> Optional[str]:
        """Try to find EAN by matching product description"""

        if not description:
            return None

        description_lower = description.lower().strip()

        # Simple fuzzy matching with product names
        for ean_code, products in self.ean_lookup.items():
            for product in products:
                if product.product_name:
                    product_name_lower = product.product_name.lower()
                    # Simple containment check
                    if (
                        description_lower in product_name_lower
                        or product_name_lower in description_lower
                    ):
                        return ean_code

        return None

    def get_enhanced_unmatched_items(self) -> pd.DataFrame:
        """Get detailed unmatched items with reasons and suggestions"""

        if not self.optimization_results:
            return pd.DataFrame()

        unmatched_data = []

        for list_result in self.optimization_results.get("lists_processed", []):
            for item in list_result.get("unmatched_items_list", []):
                # Get additional analysis for why item wasn't matched
                original_ref = item.get("original_reference", "")
                ean_attempted = item.get("normalized_ean", "")

                # Check if EAN exists in supplier data but had other issues
                ean_in_suppliers = False
                supplier_count = 0
                available_suppliers = []
                price_info = []

                if ean_attempted:
                    for product in self.supplier_data:
                        if (
                            hasattr(product, "ean_code")
                            and str(product.ean_code).strip() == ean_attempted
                        ):
                            ean_in_suppliers = True
                            supplier_count += 1
                            available_suppliers.append(product.supplier)
                            if hasattr(product, "price"):
                                price_info.append(
                                    f"{product.supplier}: €{product.price:.2f}"
                                )

                # Determine detailed reason
                detailed_reason = item.get("reason", "Unknown")
                if ean_attempted and not ean_in_suppliers:
                    detailed_reason = (
                        f"EAN {ean_attempted} not found in any supplier catalog"
                    )
                elif ean_attempted and ean_in_suppliers and supplier_count > 0:
                    detailed_reason = f"EAN found in {supplier_count} supplier(s) but other criteria not met"
                elif not ean_attempted:
                    detailed_reason = (
                        "Could not extract valid EAN from product reference"
                    )

                # Get stock average price if available
                stock_avg_price = None
                stock_avg_status = "Unknown"
                if ean_attempted:
                    stock_avg_price = self._get_stock_average_price(ean_attempted)
                    stock_avg_status = (
                        f"€{stock_avg_price:.2f}"
                        if stock_avg_price
                        else "Not available"
                    )

                unmatched_data.append(
                    {
                        "List_Name": list_result["list_name"],
                        "Original_Reference": original_ref,
                        "Extracted_EAN": (
                            ean_attempted if ean_attempted else "Failed to extract"
                        ),
                        "Quantity_Requested": item.get("quantity", 0),
                        "Reason": detailed_reason,
                        "EAN_Found_In_Suppliers": "Yes" if ean_in_suppliers else "No",
                        "Supplier_Count": supplier_count,
                        "Available_Suppliers": (
                            ", ".join(set(available_suppliers))
                            if available_suppliers
                            else "None"
                        ),
                        "Stock_Avg_Price": stock_avg_status,
                        "Supplier_Prices": (
                            " | ".join(price_info[:3])
                            if price_info
                            else "None available"
                        ),
                        "Suggestions": self._get_unmatched_suggestions(
                            original_ref,
                            ean_attempted,
                            ean_in_suppliers,
                            stock_avg_price,
                            price_info,
                        ),
                        "Analysis_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        return pd.DataFrame(unmatched_data)

    def _get_unmatched_suggestions(
        self,
        original_ref: str,
        ean_attempted: str,
        ean_in_suppliers: bool,
        stock_avg_price: Optional[float],
        price_info: List[str],
    ) -> str:
        """Generate helpful suggestions for unmatched items"""

        suggestions = []

        if not ean_attempted:
            suggestions.append("Check if product reference contains valid EAN/barcode")
            suggestions.append("Verify CSV structure and EAN column mapping")
        elif not ean_in_suppliers:
            suggestions.append("EAN not in supplier catalogs - check supplier coverage")
            suggestions.append("Verify EAN is correct (may be typo or old product)")
            suggestions.append("Consider requesting quote from suppliers for this EAN")
        elif ean_in_suppliers and price_info:
            if stock_avg_price:
                # Analyze if suppliers are too expensive
                expensive_suppliers = 0
                for price_str in price_info:
                    try:
                        price = float(price_str.split("€")[1].split()[0])
                        if price >= stock_avg_price:
                            expensive_suppliers += 1
                    except:
                        continue

                if expensive_suppliers == len(price_info):
                    suggestions.append("All supplier prices >= stock average price")
                    suggestions.append(
                        "Consider negotiating better prices with suppliers"
                    )
                else:
                    suggestions.append(
                        "Some suppliers have competitive prices - check quantity availability"
                    )
            else:
                suggestions.append(
                    "Stock average price not available - manual review needed"
                )
                suggestions.append("Check internal pricing data for this EAN")
        else:
            suggestions.append("EAN found but no pricing available from suppliers")
            suggestions.append("Contact suppliers to get current pricing")

        return " | ".join(suggestions)

    def export_orders(self) -> Dict[str, str]:
        """Export order files as CSV strings"""

        supplier_orders = self.generate_supplier_orders()
        exported_files = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for supplier, df in supplier_orders.items():
            filename = f"order_{supplier.replace(' ', '_')}_{timestamp}.csv"
            csv_content = df.to_csv(index=False)
            exported_files[filename] = csv_content

        return exported_files

    def generate_supplier_orders(self) -> Dict[str, pd.DataFrame]:
        """Generate order files for each supplier"""

        if not self.optimization_results:
            return {}

        supplier_orders = {}

        for supplier, items in self.optimization_results["supplier_orders"].items():
            if not items:
                continue

            # Create DataFrame for supplier
            df_data = []
            total_value = 0

            for item in items:
                df_data.append(
                    {
                        "EAN Code": pad_ean_code(item["ean_code"]),
                        "Product Name": item["product_name"],
                        "Quantity": item["quantity"],
                        "Unit Price": f"€{item['unit_price']:.2f}",
                        "Total Price": f"€{item['total_price']:.2f}",
                        "Reference": item["original_reference"],
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

    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization results"""

        if not self.optimization_results:
            return {"error": "No optimization results available"}

        results = self.optimization_results

        # Calculate savings and statistics
        total_value = 0
        supplier_stats = {}

        for supplier, items in results["supplier_orders"].items():
            supplier_value = sum(item["total_price"] for item in items)
            supplier_stats[supplier] = {
                "items": len(items),
                "total_value": supplier_value,
                "unique_products": len(set(item["ean_code"] for item in items)),
            }
            total_value += supplier_value

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
        }

    def get_unmatched_items(self) -> pd.DataFrame:
        """Get all unmatched items for review (basic version)"""

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

    # Replace the existing _allocate_quantities_for_ean method with the smart version
    def _allocate_quantities_for_ean(
        self, ean: str, ean_opportunities: List[Dict]
    ) -> List[Dict]:
        """Wrapper method that uses smart price-based allocation"""
        return self._smart_price_based_allocation_for_ean(ean, ean_opportunities)
