"""
Enhanced Procurement Opportunity Engine - With Multi-Supplier Allocation AND All Sales Periods
Core logic for detecting procurement opportunities with intelligent quantity allocation
UPDATED: Only Priority 1 and Priority 2 (Priority 3 eliminated for quality focus)
UPDATED: Store all sales periods (90d, 180d, 365d) for dynamic Net Need calculation
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SimpleOpportunityEngine:
    """Enhanced POC version with multi-supplier allocation for quantity constraints AND all sales periods"""

    def __init__(self):
        self.internal_data = []
        self.opportunities = []

    def _safe_int_conversion(self, value, default=0):
        """Safely convert value to int, handling NaN and None"""
        if pd.isna(value) or value is None:
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default

    def load_internal_data(self, uploaded_file) -> bool:
        """Load internal product data from CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)

            logger.info(
                f"📊 Loaded CSV with {len(df)} rows and columns: {list(df.columns)}"
            )

            # Clean and validate data
            df = df.dropna(subset=["EAN"])  # Remove rows without EAN

            # Convert to list of dictionaries
            internal_products = []
            for _, row in df.iterrows():
                try:
                    # Handle EAN - convert to string, handle floats
                    ean_raw = row["EAN"]
                    if pd.isna(ean_raw):
                        continue

                    # Convert float EAN to string (removes .0)
                    if isinstance(ean_raw, float):
                        ean = str(int(ean_raw))
                    else:
                        ean = str(ean_raw).strip()

                    # Parse the three price fields
                    best_buy_price = self._parse_price(row.get("bestbuyPrice(12M)"))
                    supplier_price = self._parse_price(row.get("supplierPrice"))
                    stock_avg_price = self._parse_price(row.get("stockAvgPrice"))

                    # Check if we have at least one valid price
                    valid_prices = [
                        p
                        for p in [best_buy_price, supplier_price, stock_avg_price]
                        if p is not None and p > 0
                    ]
                    if not valid_prices:
                        continue  # Skip products without any valid price

                    # Parse bestseller field - should be 1 or 2 for bestsellers
                    bestseller_raw = row.get("bestSeller", "")
                    is_bestseller = False
                    bestseller_rank = None

                    try:
                        if pd.notna(bestseller_raw):
                            bestseller_num = int(float(str(bestseller_raw)))
                            bestseller_rank = bestseller_num
                            is_bestseller = bestseller_num in [
                                1,
                                2,
                            ]  # Only 1 or 2 are bestsellers
                    except (ValueError, TypeError):
                        pass

                    product = {
                        "ean": ean,
                        "cnp": str(row.get("CNP", "")),
                        "brand": str(row.get("brand", "")),
                        "description": str(row.get("itemDescriptionEN", "")),
                        "capacity": str(row.get("itemCapacity", "")),
                        "stock": self._safe_int_conversion(row.get("stock"), 0),
                        "sales90d": self._safe_int_conversion(row.get("sales90d"), 0),
                        "sales180d": self._safe_int_conversion(row.get("sales180d"), 0),
                        "sales365d": self._safe_int_conversion(row.get("sales365d"), 0),
                        "sales_next90d_lastyear": self._safe_int_conversion(
                            row.get("salesnext90d_lastyear"), 0
                        ),
                        # FIX: Add the missing qntPendingToDeliver field
                        "qntPendingToDeliver": self._safe_int_conversion(
                            row.get("qntPendingToDeliver"), 0
                        ),
                        "best_buy_price": best_buy_price,
                        "supplier_price": supplier_price,
                        "stock_avg_price": stock_avg_price,
                        "best_supplier": str(row.get("bestbuyPrice_supplier(12M)", "")),
                        "is_bestseller": is_bestseller,
                        "bestseller_rank": bestseller_rank,
                        "is_active": bool(row.get("isActive", True)),
                    }

                    # Only add products with valid EAN and at least one valid price
                    if ean and valid_prices:
                        internal_products.append(product)

                except Exception as e:
                    logger.warning(f"Error processing row: {e}")
                    continue

            self.internal_data = internal_products
            logger.info(
                f"✅ Successfully loaded {len(internal_products)} valid internal products"
            )

            # Log bestseller statistics
            bestseller_count = sum(1 for p in internal_products if p["is_bestseller"])
            logger.info(
                f"📊 Found {bestseller_count} bestseller products (rank 1 or 2)"
            )

            # UPDATED: Log sales period data availability
            sales90d_count = sum(
                1 for p in internal_products if p.get("sales90d", 0) > 0
            )
            sales180d_count = sum(
                1 for p in internal_products if p.get("sales180d", 0) > 0
            )
            sales365d_count = sum(
                1 for p in internal_products if p.get("sales365d", 0) > 0
            )

            logger.info(f"📊 Sales data availability:")
            logger.info(f"   90 days: {sales90d_count} products")
            logger.info(f"   180 days: {sales180d_count} products")
            logger.info(f"   365 days: {sales365d_count} products")

            return True

        except Exception as e:
            logger.error(f"❌ Error loading internal data: {e}")
            return False

    def _parse_price(self, price_str) -> Optional[float]:
        """Parse price string to float, handling various formats"""
        if pd.isna(price_str):
            return None

        try:
            # Handle string prices with currency symbols
            if isinstance(price_str, str):
                # Remove currency symbols, commas, and spaces
                clean_price = (
                    price_str.replace("€", "")
                    .replace("$", "")
                    .replace(",", ".")
                    .replace(" ", "")
                    .strip()
                )
                if not clean_price:
                    return None
                return float(clean_price)
            else:
                return float(price_str)
        except (ValueError, TypeError):
            return None

    def _analyze_price_comparison(
        self, internal_product: Dict, quote_price: float
    ) -> Dict:
        """Analyze which internal prices the quote beats and determine priority"""

        # Get the three internal prices
        best_buy_price = internal_product.get("best_buy_price")
        supplier_price = internal_product.get("supplier_price")
        stock_avg_price = internal_product.get("stock_avg_price")

        # Track which prices exist and are beaten by quote
        price_analysis = {
            "best_buy_price": best_buy_price,
            "supplier_price": supplier_price,
            "stock_avg_price": stock_avg_price,
            "beats_best_buy": False,
            "beats_supplier": False,
            "beats_stock_avg": False,
            "total_prices_beaten": 0,
            "valid_prices_count": 0,
        }

        # Check which prices are valid and beaten
        if best_buy_price is not None and best_buy_price > 0:
            price_analysis["valid_prices_count"] += 1
            if quote_price < best_buy_price:
                price_analysis["beats_best_buy"] = True
                price_analysis["total_prices_beaten"] += 1

        if supplier_price is not None and supplier_price > 0:
            price_analysis["valid_prices_count"] += 1
            if quote_price < supplier_price:
                price_analysis["beats_supplier"] = True
                price_analysis["total_prices_beaten"] += 1

        if stock_avg_price is not None and stock_avg_price > 0:
            price_analysis["valid_prices_count"] += 1
            if quote_price < stock_avg_price:
                price_analysis["beats_stock_avg"] = True
                price_analysis["total_prices_beaten"] += 1

        return price_analysis

    def _determine_opportunity_priority(self, price_analysis: Dict) -> Dict:
        """Determine opportunity priority and baseline price - UPDATED: Only Priority 1 and 2"""

        best_buy_price = price_analysis["best_buy_price"]
        supplier_price = price_analysis["supplier_price"]
        stock_avg_price = price_analysis["stock_avg_price"]

        beats_best_buy = price_analysis["beats_best_buy"]
        beats_supplier = price_analysis["beats_supplier"]
        beats_stock_avg = price_analysis["beats_stock_avg"]

        # Priority 1: Quote beats ALL 3 internal prices
        if beats_best_buy and beats_supplier and beats_stock_avg:
            # Baseline is minimum of all 3 prices
            all_prices = [
                p
                for p in [best_buy_price, supplier_price, stock_avg_price]
                if p is not None and p > 0
            ]
            return {
                "priority": 1,
                "priority_label": "🔥 Priority 1",
                "baseline_price": min(all_prices) if all_prices else None,
                "description": "Better than ALL internal prices",
            }

        # Priority 2: Quote beats stockAvgPrice AND supplierPrice but NOT bestbuyPrice
        elif beats_supplier and beats_stock_avg and not beats_best_buy:
            # Baseline is minimum of stockAvgPrice and supplierPrice
            operational_prices = [
                p for p in [supplier_price, stock_avg_price] if p is not None and p > 0
            ]
            return {
                "priority": 2,
                "priority_label": "⭐ Priority 2",
                "baseline_price": (
                    min(operational_prices) if operational_prices else None
                ),
                "description": "Better than operational prices",
            }

        # REMOVED: Priority 3 - No longer accepting opportunities that only beat some prices
        # Previously this was: beats_supplier OR beats_stock_avg (but not both + bestbuy)

        # No priority - quote doesn't meet Priority 1 or 2 criteria
        else:
            return None

    def _get_bestseller_stars(self, bestseller_rank: Optional[int]) -> str:
        """Convert bestseller rank to star display (inverse relationship)"""
        if bestseller_rank is None:
            return ""

        # Inverse relationship: rank 1 = 4 stars, rank 2 = 3 stars, etc.
        star_count = max(0, 5 - bestseller_rank)
        return "⭐" * star_count

    def find_internal_product(self, ean: str) -> Optional[Dict]:
        """Find internal product by EAN code"""
        ean = str(ean).strip()
        for product in self.internal_data:
            if product["ean"] == ean:
                return product
        return None

    def calculate_simple_net_need(
        self, internal_product: Dict, sales_period: str = "sales90d"
    ) -> int:
        """
        Calculate simple net need using specified sales period INCLUDING qntPendingToDeliver
        UPDATED: New formula: sales - current_stock - qntPendingToDeliver
        """
        sales = internal_product.get(sales_period, 0)
        current_stock = internal_product.get("stock", 0)
        qnt_pending_to_deliver = internal_product.get("qntPendingToDeliver", 0)

        # Updated formula: if we sold X in period, have Y in stock, and Z pending delivery
        # we need max(0, X - Y - Z)
        net_need = max(0, sales - current_stock - qnt_pending_to_deliver)
        return net_need

    def calculate_urgency_score(
        self, internal_product: Dict, sales_period: str = "sales90d"
    ) -> str:
        """Calculate urgency score considering current stock AND pending deliveries"""
        current_stock = internal_product.get("stock", 0)
        qnt_pending_to_deliver = internal_product.get("qntPendingToDeliver", 0)
        sales = internal_product.get(sales_period, 0)

        if sales == 0:
            return "Low"

        # Calculate days in period
        period_days = {"sales90d": 90, "sales180d": 180, "sales365d": 365}.get(
            sales_period, 90
        )

        # Days of cover: how many days current stock + pending deliveries will last
        daily_sales = sales / period_days
        if daily_sales == 0:
            return "Low"

        # Total available = current stock + what we're expecting
        total_available = current_stock + qnt_pending_to_deliver
        days_of_cover = total_available / daily_sales

        if days_of_cover < 14:
            return "High"
        elif days_of_cover < 30:
            return "Medium"
        else:
            return "Low"

    def find_opportunities(self, supplier_data) -> List[Dict]:
        """Find procurement opportunities with multi-supplier allocation - UPDATED: Only Priority 1 and 2, Store All Sales Periods"""

        opportunities = []

        if not self.internal_data:
            logger.warning("No internal data loaded")
            return opportunities

        if not supplier_data:
            logger.warning("No supplier data provided")
            return opportunities

        logger.info(
            f"🔍 Analyzing {len(supplier_data)} supplier products against {len(self.internal_data)} internal products"
        )

        matched_count = 0
        all_opportunities = []  # Store all opportunities before smart allocation

        for supplier_product in supplier_data:
            # Get EAN from supplier product
            supplier_ean = None
            if hasattr(supplier_product, "ean_code") and supplier_product.ean_code:
                supplier_ean = str(supplier_product.ean_code).strip()
            elif (
                hasattr(supplier_product, "supplier_code")
                and supplier_product.supplier_code
            ):
                # Try supplier code as EAN if it looks like EAN (numeric, right length)
                code = str(supplier_product.supplier_code).strip()
                if code.isdigit() and len(code) >= 8:
                    supplier_ean = code

            if not supplier_ean:
                continue

            # Find matching internal product
            internal_product = self.find_internal_product(supplier_ean)

            if not internal_product:
                continue

            matched_count += 1

            # Calculate opportunity with priority system and quantity info
            opportunity = self._calculate_opportunity_with_priority_and_quantity(
                internal_product, supplier_product
            )

            if opportunity:
                all_opportunities.append(opportunity)

        logger.info(
            f"📊 Before allocation: {len(all_opportunities)} opportunities from {matched_count} matches"
        )

        # Smart multi-supplier allocation instead of simple deduplication
        allocated_opportunities = self._smart_multi_supplier_allocation(
            all_opportunities
        )

        # Count by priority after allocation - UPDATED: Only Priority 1 and 2
        priority_counts = {1: 0, 2: 0}  # REMOVED: 3: 0
        split_orders = 0
        for opp in allocated_opportunities:
            priority = opp.get("priority", 0)
            if priority in priority_counts:
                priority_counts[priority] += 1
            if opp.get("is_split_order", False):
                split_orders += 1

        # Sort by priority first, then by total savings
        allocated_opportunities.sort(key=lambda x: (x["priority"], -x["total_savings"]))

        logger.info(f"📊 Final Multi-Supplier Analysis Results (High Standards Only):")
        logger.info(f"   EAN Matches Found: {matched_count}")
        logger.info(f"   Before Allocation: {len(all_opportunities)}")
        logger.info(f"   After Smart Allocation: {len(allocated_opportunities)}")
        logger.info(f"   🔄 Split Orders Created: {split_orders}")
        logger.info(f"   🔥 Priority 1 (beats all 3 prices): {priority_counts[1]}")
        logger.info(
            f"   ⭐ Priority 2 (beats supplier + stock_avg): {priority_counts[2]}"
        )
        # REMOVED: Priority 3 logging
        logger.info(f"   🎯 Quality Focus: Only high-impact opportunities included")

        self.opportunities = allocated_opportunities
        return allocated_opportunities

    def _smart_multi_supplier_allocation(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Smart allocation that handles quantity constraints with multi-supplier orders
        """

        if not opportunities:
            return []

        logger.info(f"🧠 Starting smart multi-supplier allocation...")

        # Group opportunities by EAN
        ean_groups = {}
        for opp in opportunities:
            ean = str(opp.get("ean", "")).strip()
            if ean:
                if ean not in ean_groups:
                    ean_groups[ean] = []
                ean_groups[ean].append(opp)

        allocated_opportunities = []

        for ean, ean_opportunities in ean_groups.items():
            if len(ean_opportunities) == 1:
                # Single supplier - no allocation needed
                single_opp = ean_opportunities[0]
                single_opp["allocation_type"] = "single_supplier"
                single_opp["is_split_order"] = False
                allocated_opportunities.append(single_opp)
                continue

            # Multiple suppliers available - apply smart allocation
            allocated_opps = self._allocate_quantities_for_ean(ean, ean_opportunities)
            allocated_opportunities.extend(allocated_opps)

        logger.info(
            f"✅ Smart allocation complete: {len(allocated_opportunities)} final opportunities"
        )
        return allocated_opportunities

    def _allocate_quantities_for_ean(
        self, ean: str, ean_opportunities: List[Dict]
    ) -> List[Dict]:
        """
        Allocate quantities for a single EAN across multiple suppliers
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

        # Check if best supplier can fulfill full need
        best_supplier = sorted_suppliers[0]
        best_supplier_qty = best_supplier.get("supplier_quantity")

        # Determine purchasable quantity and whether we need to split
        if best_supplier_qty is None:
            # Quantity unknown - assume we can buy all we need
            purchasable_qty = net_need
            can_fulfill_fully = True
            quantity_constrained = False
        elif best_supplier_qty >= net_need:
            # Sufficient quantity available
            purchasable_qty = net_need
            can_fulfill_fully = True
            quantity_constrained = False
        else:
            # Insufficient quantity - but we'll still try single supplier first
            purchasable_qty = best_supplier_qty
            can_fulfill_fully = False
            quantity_constrained = True

        if can_fulfill_fully or best_supplier_qty is None:
            # Best supplier can fulfill full order (or quantity unknown)
            best_opp = best_supplier.copy()
            best_opp["allocation_type"] = "single_best"
            best_opp["is_split_order"] = False
            best_opp["allocated_quantity"] = purchasable_qty

            # Calculate total savings based on actual purchasable quantity
            savings_per_unit = best_opp.get("savings_per_unit", 0)
            best_opp["total_savings"] = savings_per_unit * purchasable_qty
            best_opp["purchasable_quantity"] = purchasable_qty

            # Update quantity analysis
            best_opp["quantity_analysis"] = {
                "net_need": net_need,
                "supplier_quantity": best_supplier_qty,
                "purchasable_quantity": purchasable_qty,
                "quantity_constrained": quantity_constrained,
                "quantity_shortage": max(0, net_need - purchasable_qty),
            }

            # Store alternative suppliers for reference
            alternatives = []
            for alt_supplier in sorted_suppliers[1:]:
                alt_qty = alt_supplier.get("supplier_quantity")
                alternatives.append(
                    {
                        "supplier": alt_supplier.get("supplier", "Unknown"),
                        "quote_price": alt_supplier.get("quote_price", 0),
                        "supplier_quantity": alt_qty,
                        "savings_vs_best": alt_supplier.get("quote_price", 0)
                        - best_supplier.get("quote_price", 0),
                    }
                )

            best_opp["alternative_suppliers"] = alternatives[:3]  # Top 3 alternatives
            return [best_opp]

        # Best supplier has insufficient quantity - need to split order
        logger.info(
            f"🔄 EAN {ean}: Need {net_need}, best supplier has {best_supplier_qty} - splitting order"
        )

        allocated_opportunities = []
        remaining_need = net_need

        for i, supplier in enumerate(sorted_suppliers):
            supplier_qty = supplier.get("supplier_quantity")

            # Skip suppliers with no quantity info or zero quantity
            if supplier_qty is None or supplier_qty <= 0:
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
                allocated_opp["split_order_rank"] = (
                    i + 1
                )  # 1 = primary, 2 = secondary, etc.

                # CORRECTED: Recalculate savings based on allocated quantity (what we can actually buy)
                savings_per_unit = allocated_opp.get("savings_per_unit", 0)
                allocated_opp["total_savings"] = savings_per_unit * allocated_qty
                allocated_opp["purchasable_quantity"] = (
                    allocated_qty  # Update purchasable quantity
                )

                # Update quantity analysis for allocated opportunity
                allocated_opp["quantity_analysis"] = {
                    "net_need": net_need,
                    "supplier_quantity": supplier_qty,
                    "purchasable_quantity": allocated_qty,
                    "quantity_constrained": True,  # Split orders are always quantity constrained
                    "quantity_shortage": (
                        max(0, net_need - supplier_qty)
                        if supplier_qty is not None
                        else 0
                    ),
                    "allocation_percentage": (
                        (allocated_qty / net_need * 100) if net_need > 0 else 0
                    ),
                }

                # Add split order metadata
                allocated_opp["allocation_type"] = "split_order"
                allocated_opp["split_percentage"] = (allocated_qty / net_need) * 100

                # Add information about other suppliers in the split
                other_suppliers = []
                for j, other_supplier in enumerate(sorted_suppliers):
                    if j != i:
                        other_qty = other_supplier.get("supplier_quantity", 0)
                        other_suppliers.append(
                            {
                                "supplier": other_supplier.get("supplier", "Unknown"),
                                "quote_price": other_supplier.get("quote_price", 0),
                                "supplier_quantity": other_qty,
                                "rank": j + 1,
                            }
                        )

                allocated_opp["split_order_details"] = {
                    "total_suppliers_in_split": sum(
                        1 for s in sorted_suppliers if s.get("supplier_quantity", 0) > 0
                    ),
                    "other_suppliers": other_suppliers[:2],  # Show top 2 others
                    "total_split_cost": self._calculate_split_order_cost(
                        sorted_suppliers, net_need
                    ),
                    "cost_vs_best_supplier_full": self._calculate_cost_difference(
                        sorted_suppliers, net_need
                    ),
                }

                allocated_opportunities.append(allocated_opp)
                remaining_need -= allocated_qty

                if remaining_need <= 0:
                    break

        # Log split order details
        if allocated_opportunities:
            total_allocated = sum(
                opp.get("allocated_quantity", 0) for opp in allocated_opportunities
            )
            total_cost = sum(
                opp.get("quote_price", 0) * opp.get("allocated_quantity", 0)
                for opp in allocated_opportunities
            )
            avg_price = total_cost / total_allocated if total_allocated > 0 else 0

            logger.info(f"📦 EAN {ean} split order:")
            logger.info(f"   Total allocated: {total_allocated}/{net_need} units")
            logger.info(f"   Suppliers used: {len(allocated_opportunities)}")
            logger.info(f"   Blended average price: €{avg_price:.2f}")

            for i, opp in enumerate(allocated_opportunities):
                supplier_name = opp.get("supplier", "Unknown")
                qty = opp.get("allocated_quantity", 0)
                price = opp.get("quote_price", 0)
                logger.info(f"   {i+1}. {supplier_name}: {qty} units @ €{price:.2f}")

        return allocated_opportunities

    def _calculate_split_order_cost(
        self, sorted_suppliers: List[Dict], net_need: int
    ) -> Dict:
        """Calculate total cost and savings for split order scenario"""

        total_cost = 0
        total_allocated = 0
        remaining_need = net_need

        for supplier in sorted_suppliers:
            supplier_qty = supplier.get("supplier_quantity", 0)
            if supplier_qty <= 0:
                continue

            allocated_qty = min(supplier_qty, remaining_need)
            if allocated_qty > 0:
                supplier_cost = supplier.get("quote_price", 0) * allocated_qty
                total_cost += supplier_cost
                total_allocated += allocated_qty
                remaining_need -= allocated_qty

                if remaining_need <= 0:
                    break

        avg_price = total_cost / total_allocated if total_allocated > 0 else 0

        return {
            "total_cost": total_cost,
            "total_allocated": total_allocated,
            "average_price": avg_price,
            "fulfillment_rate": (
                (total_allocated / net_need) * 100 if net_need > 0 else 0
            ),
        }

    def _calculate_cost_difference(
        self, sorted_suppliers: List[Dict], net_need: int
    ) -> Dict:
        """Calculate cost difference between split order and hypothetical full order from best supplier"""

        if not sorted_suppliers:
            return {"cost_difference": 0, "percentage_difference": 0}

        # Cost of split order
        split_cost_info = self._calculate_split_order_cost(sorted_suppliers, net_need)
        split_total_cost = split_cost_info["total_cost"]

        # Hypothetical cost if best supplier could fulfill entire order
        best_supplier = sorted_suppliers[0]
        best_price = best_supplier.get("quote_price", 0)
        hypothetical_best_cost = best_price * net_need

        cost_difference = split_total_cost - hypothetical_best_cost
        percentage_difference = (
            (cost_difference / hypothetical_best_cost * 100)
            if hypothetical_best_cost > 0
            else 0
        )

        return {
            "cost_difference": cost_difference,
            "percentage_difference": percentage_difference,
            "split_cost": split_total_cost,
            "hypothetical_best_cost": hypothetical_best_cost,
        }

    def _calculate_opportunity_with_priority_and_quantity(
        self, internal_product: Dict, supplier_product
    ) -> Optional[Dict]:
        """Calculate opportunity metrics with priority classification and supplier quantity"""

        # Get supplier quote price
        if hasattr(supplier_product, "price"):
            quote_price = supplier_product.price
        else:
            return None

        if not quote_price or quote_price <= 0:
            return None

        # NEW: Get supplier quantity information
        supplier_quantity = None
        if (
            hasattr(supplier_product, "quantity")
            and supplier_product.quantity is not None
        ):
            supplier_quantity = supplier_product.quantity

        # Analyze price comparison and determine priority
        price_analysis = self._analyze_price_comparison(internal_product, quote_price)
        priority_info = self._determine_opportunity_priority(price_analysis)

        # If no priority determined, skip this opportunity
        if not priority_info:
            return None

        baseline_price = priority_info["baseline_price"]
        if not baseline_price:
            return None

        # Calculate net need
        net_need = self.calculate_simple_net_need(internal_product)

        # Calculate savings
        savings_per_unit = baseline_price - quote_price

        # Must have positive savings
        if savings_per_unit <= 0:
            return None

        # Calculate total potential savings
        total_savings = savings_per_unit * net_need if net_need > 0 else 0

        # Get supplier name
        supplier_name = "Unknown Supplier"
        if hasattr(supplier_product, "supplier") and supplier_product.supplier:
            supplier_name = supplier_product.supplier

        # Create detailed price breakdown for transparency
        price_breakdown = {
            "best_buy_price": price_analysis["best_buy_price"],
            "supplier_price": price_analysis["supplier_price"],
            "stock_avg_price": price_analysis["stock_avg_price"],
            "beats_best_buy": price_analysis["beats_best_buy"],
            "beats_supplier": price_analysis["beats_supplier"],
            "beats_stock_avg": price_analysis["beats_stock_avg"],
            "prices_beaten": price_analysis["total_prices_beaten"],
        }

        return {
            "ean": internal_product["ean"],
            "product_name": internal_product.get("description", "Unknown Product"),
            "brand": internal_product.get("brand", ""),
            "current_stock": internal_product.get("stock", 0),
            "qntPendingToDeliver": internal_product.get(
                "qntPendingToDeliver", 0
            ),  # ← NEW LINE ADDED
            "sales90d": internal_product.get("sales90d", 0),
            "sales180d": internal_product.get("sales180d", 0),
            "sales365d": internal_product.get("sales365d", 0),
            "net_need": net_need,
            "baseline_price": baseline_price,
            "quote_price": quote_price,
            "savings_per_unit": savings_per_unit,
            "total_savings": total_savings,
            "supplier": supplier_name,
            "supplier_quantity": supplier_quantity,  # NEW: Include supplier quantity
            "priority": priority_info["priority"],
            "priority_label": priority_info["priority_label"],
            "priority_description": priority_info["description"],
            "calculation_note": priority_info.get("calculation_note", ""),
            "price_breakdown": price_breakdown,
            "is_bestseller": internal_product.get("is_bestseller", False),
            "bestseller_rank": internal_product.get("bestseller_rank"),
            "bestseller_stars": self._get_bestseller_stars(
                internal_product.get("bestseller_rank")
            ),
            "best_historical_supplier": internal_product.get("best_supplier", ""),
            "urgency_score": self.calculate_urgency_score(internal_product),
            "days_of_cover": self._calculate_days_of_cover(internal_product),
        }

    def _calculate_days_of_cover(self, internal_product: Dict) -> float:
        """Calculate how many days current stock + pending deliveries will last"""
        stock = internal_product.get("stock", 0)
        qnt_pending_to_deliver = internal_product.get("qntPendingToDeliver", 0)  # NEW
        sales90d = internal_product.get("sales90d", 0)

        if sales90d == 0:
            return 999  # Infinite cover if no sales

        daily_sales = sales90d / 90
        total_available = stock + qnt_pending_to_deliver  # UPDATED
        return total_available / daily_sales if daily_sales > 0 else 999

    def verify_savings_calculations(self) -> Dict:
        """Debug function to verify all savings calculations are correct"""
        if not self.opportunities:
            return {"status": "No opportunities to verify", "issues": []}

        verification_results = {
            "total_opportunities": len(self.opportunities),
            "correct_calculations": 0,
            "incorrect_calculations": 0,
            "issues_found": [],
            "calculation_details": [],
        }

        for i, opp in enumerate(self.opportunities):
            # Get calculation components
            net_need = opp.get("net_need", 0)
            supplier_qty = opp.get("supplier_quantity")
            purchasable_qty = opp.get("purchasable_quantity", 0)
            savings_per_unit = opp.get("savings_per_unit", 0)
            recorded_total_savings = opp.get("total_savings", 0)

            # Calculate what the total savings should be
            if supplier_qty is not None:
                expected_purchasable = (
                    min(supplier_qty, net_need) if net_need > 0 else 0
                )
            else:
                expected_purchasable = net_need if net_need > 0 else 0

            expected_total_savings = savings_per_unit * expected_purchasable

            # Check if calculations match
            calculation_correct = (
                abs(recorded_total_savings - expected_total_savings) < 0.01
            )
            purchasable_correct = abs(purchasable_qty - expected_purchasable) < 0.01

            if calculation_correct and purchasable_correct:
                verification_results["correct_calculations"] += 1
            else:
                verification_results["incorrect_calculations"] += 1

                issue_details = {
                    "opportunity_index": i,
                    "ean": opp.get("ean", "Unknown"),
                    "supplier": opp.get("supplier", "Unknown"),
                    "net_need": net_need,
                    "supplier_quantity": supplier_qty,
                    "recorded_purchasable": purchasable_qty,
                    "expected_purchasable": expected_purchasable,
                    "savings_per_unit": savings_per_unit,
                    "recorded_total_savings": recorded_total_savings,
                    "expected_total_savings": expected_total_savings,
                    "purchasable_error": not purchasable_correct,
                    "savings_error": not calculation_correct,
                }

                verification_results["issues_found"].append(issue_details)

            # Store calculation details for first 5 opportunities
            if i < 5:
                verification_results["calculation_details"].append(
                    {
                        "ean": opp.get("ean", "Unknown"),
                        "net_need": net_need,
                        "supplier_quantity": supplier_qty,
                        "purchasable_qty": purchasable_qty,
                        "expected_purchasable": expected_purchasable,
                        "savings_per_unit": savings_per_unit,
                        "total_savings": recorded_total_savings,
                        "expected_savings": expected_total_savings,
                        "is_correct": calculation_correct and purchasable_correct,
                    }
                )

        verification_results["accuracy_rate"] = (
            verification_results["correct_calculations"]
            / verification_results["total_opportunities"]
            * 100
            if verification_results["total_opportunities"] > 0
            else 0
        )

        logger.info(
            f"💰 Savings calculation verification: {verification_results['correct_calculations']}/{verification_results['total_opportunities']} correct ({verification_results['accuracy_rate']:.1f}%)"
        )

        if verification_results["incorrect_calculations"] > 0:
            logger.warning(
                f"⚠️ Found {verification_results['incorrect_calculations']} calculation errors"
            )
            for issue in verification_results["issues_found"][:3]:  # Log first 3 issues
                logger.warning(
                    f"EAN {issue['ean']}: Expected savings €{issue['expected_total_savings']:.2f}, got €{issue['recorded_total_savings']:.2f}"
                )

        return verification_results

    def verify_allocation(self) -> Dict:
        """Verify that multi-supplier allocation worked correctly"""
        if not self.opportunities:
            return {
                "status": "No opportunities to verify",
                "allocation_success": True,
                "total_opportunities": 0,
                "unique_eans": 0,
                "split_orders": 0,
            }

        # Analyze the allocation results
        ean_analysis = {}
        split_order_count = 0

        for opp in self.opportunities:
            ean = str(opp.get("ean", "")).strip()
            if not ean:
                continue

            if ean not in ean_analysis:
                ean_analysis[ean] = {
                    "total_opportunities": 0,
                    "split_order": False,
                    "suppliers": [],
                    "total_allocated": 0,
                    "net_need": 0,
                }

            ean_info = ean_analysis[ean]
            ean_info["total_opportunities"] += 1
            ean_info["suppliers"].append(opp.get("supplier", "Unknown"))
            ean_info["total_allocated"] += opp.get(
                "allocated_quantity", opp.get("net_need", 0)
            )
            ean_info["net_need"] = opp.get("original_net_need", opp.get("net_need", 0))

            if opp.get("is_split_order", False):
                ean_info["split_order"] = True
                split_order_count += 1

        # Check for any issues
        allocation_issues = []
        for ean, info in ean_analysis.items():
            if info["split_order"] and info["total_allocated"] < info["net_need"]:
                shortage = info["net_need"] - info["total_allocated"]
                allocation_issues.append(f"EAN {ean}: {shortage} units short")

        verification = {
            "total_opportunities": len(self.opportunities),
            "unique_eans": len(ean_analysis),
            "split_orders": split_order_count,
            "single_orders": len(ean_analysis) - split_order_count,
            "allocation_issues": allocation_issues,
            "allocation_success": len(allocation_issues) == 0,
            "status": (
                "✅ Multi-supplier allocation successful"
                if len(allocation_issues) == 0
                else f"⚠️ {len(allocation_issues)} allocation issues found"
            ),
        }

        if allocation_issues:
            logger.warning(f"⚠️ Allocation issues found: {allocation_issues}")
        else:
            logger.info(
                f"✅ Multi-supplier allocation verified: {len(self.opportunities)} opportunities, {split_order_count} split orders"
            )

        return verification

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for opportunities - UPDATED: Only Priority 1 and 2"""

        if not self.opportunities:
            return {
                "total_opportunities": 0,
                "total_potential_savings": 0,
                "avg_savings_per_opportunity": 0,
                "high_urgency_count": 0,
                "bestseller_opportunities": 0,
                "priority_1_count": 0,
                "priority_2_count": 0,
                # REMOVED: priority_3_count
                "split_orders_count": 0,
                "single_orders_count": 0,
                "internal_products_loaded": len(self.internal_data),
            }

        total_savings = sum(opp["total_savings"] for opp in self.opportunities)
        high_urgency = sum(
            1 for opp in self.opportunities if opp["urgency_score"] == "High"
        )
        bestseller_opps = sum(1 for opp in self.opportunities if opp["is_bestseller"])

        # Count by priority - only 1 and 2
        priority_1_count = sum(1 for opp in self.opportunities if opp["priority"] == 1)
        priority_2_count = sum(1 for opp in self.opportunities if opp["priority"] == 2)

        # Count allocation types
        split_orders = sum(
            1 for opp in self.opportunities if opp.get("is_split_order", False)
        )
        single_orders = len(self.opportunities) - split_orders

        return {
            "total_opportunities": len(self.opportunities),
            "total_potential_savings": total_savings,
            "avg_savings_per_opportunity": total_savings / len(self.opportunities),
            "high_urgency_count": high_urgency,
            "bestseller_opportunities": bestseller_opps,
            "priority_1_count": priority_1_count,
            "priority_2_count": priority_2_count,
            # REMOVED: priority_3_count
            "split_orders_count": split_orders,
            "single_orders_count": single_orders,
            "internal_products_loaded": len(self.internal_data),
            "products_with_opportunities": len(
                set(opp["ean"] for opp in self.opportunities)
            ),
            "qualification_rate": (
                len(self.opportunities) / len(self.internal_data) * 100
                if self.internal_data
                else 0
            ),
        }

    def filter_opportunities(
        self,
        min_savings: float = 0,
        min_total_savings: float = 0,
        urgency_filter: str = None,
        bestsellers_only: bool = False,
        priority_filter: int = None,
        allocation_type_filter: str = None,
        brand_filter: str = None,
    ) -> List[Dict]:
        """Filter opportunities with allocation type filter and brand filter - UPDATED: Only Priority 1 and 2"""

        filtered = self.opportunities.copy()

        # Filter by priority - only 1 and 2 available
        if priority_filter is not None:
            filtered = [opp for opp in filtered if opp["priority"] == priority_filter]

        # Filter by allocation type
        if allocation_type_filter:
            if allocation_type_filter == "split_only":
                filtered = [opp for opp in filtered if opp.get("is_split_order", False)]
            elif allocation_type_filter == "single_only":
                filtered = [
                    opp for opp in filtered if not opp.get("is_split_order", False)
                ]

        # Filter by brand
        if brand_filter and brand_filter not in ["All Brands", None]:
            if brand_filter == "Unknown/Empty":
                filtered = [opp for opp in filtered if not opp.get("brand", "").strip()]
            else:
                filtered = [
                    opp
                    for opp in filtered
                    if opp.get("brand", "").strip() == brand_filter
                ]

        # Filter by minimum savings per unit
        if min_savings > 0:
            filtered = [
                opp for opp in filtered if opp["savings_per_unit"] >= min_savings
            ]

        # Filter by minimum total savings
        if min_total_savings > 0:
            filtered = [
                opp for opp in filtered if opp["total_savings"] >= min_total_savings
            ]

        # Filter by urgency
        if urgency_filter:
            filtered = [
                opp for opp in filtered if opp["urgency_score"] == urgency_filter
            ]

        # Filter by bestsellers
        if bestsellers_only:
            filtered = [opp for opp in filtered if opp["is_bestseller"]]

        return filtered

    def analyze_supplier_performance(self) -> Dict[str, Dict]:
        """Analyze supplier performance - UPDATED: Only Priority 1 and 2"""

        supplier_stats = {}

        for opp in self.opportunities:
            supplier = opp["supplier"]

            if supplier not in supplier_stats:
                supplier_stats[supplier] = {
                    "opportunity_count": 0,
                    "total_potential_savings": 0,
                    "avg_savings_per_unit": 0,
                    "high_urgency_items": 0,
                    "bestseller_items": 0,
                    "priority_1_count": 0,
                    "priority_2_count": 0,
                    # REMOVED: priority_3_count
                    "split_order_count": 0,
                    "single_order_count": 0,
                    "primary_supplier_count": 0,  # How often they're the main supplier in splits
                    "savings_list": [],
                    "total_quantity_available": 0,
                    "products_with_quantity": 0,
                }

            stats = supplier_stats[supplier]
            stats["opportunity_count"] += 1
            stats["total_potential_savings"] += opp["total_savings"]
            stats["savings_list"].append(opp["savings_per_unit"])

            # Track quantity information
            if opp.get("supplier_quantity") is not None:
                stats["total_quantity_available"] += opp["supplier_quantity"]
                stats["products_with_quantity"] += 1

            # Count by priority - only 1 and 2
            if opp["priority"] == 1:
                stats["priority_1_count"] += 1
            elif opp["priority"] == 2:
                stats["priority_2_count"] += 1
            # REMOVED: elif opp["priority"] == 3

            # Count allocation types
            if opp.get("is_split_order", False):
                stats["split_order_count"] += 1
                # Check if they're the primary (rank 1) supplier in split
                if opp.get("split_order_rank", 1) == 1:
                    stats["primary_supplier_count"] += 1
            else:
                stats["single_order_count"] += 1

            if opp["urgency_score"] == "High":
                stats["high_urgency_items"] += 1

            if opp["is_bestseller"]:
                stats["bestseller_items"] += 1

        # Calculate averages and performance metrics
        for supplier, stats in supplier_stats.items():
            if stats["savings_list"]:
                stats["avg_savings_per_unit"] = sum(stats["savings_list"]) / len(
                    stats["savings_list"]
                )

            # Calculate split order performance
            total_orders = stats["split_order_count"] + stats["single_order_count"]
            stats["split_order_percentage"] = (
                (stats["split_order_count"] / total_orders * 100)
                if total_orders > 0
                else 0
            )
            stats["primary_supplier_rate"] = (
                (stats["primary_supplier_count"] / stats["split_order_count"] * 100)
                if stats["split_order_count"] > 0
                else 0
            )

            # Calculate quality metrics
            total_qualified = stats["priority_1_count"] + stats["priority_2_count"]
            stats["high_quality_rate"] = (
                (stats["priority_1_count"] / total_qualified * 100)
                if total_qualified > 0
                else 0
            )

        return supplier_stats


# Enhanced configuration - UPDATED: Only Priority 1 and 2
ENHANCED_CONFIG = {
    "min_savings_threshold": 0.01,  # Minimum €0.01 savings per unit
    "min_total_savings_threshold": 0.10,  # Minimum €0.10 total savings
    "urgency_days_threshold": 14,  # Days of cover for high urgency
    "priority_system_enabled": True,
    "priority_levels": [1, 2],  # UPDATED: Only Priority 1 and 2
    "multi_supplier_allocation_enabled": True,
    "show_supplier_quantities": True,
    "max_suppliers_per_ean": 3,
    "min_allocation_quantity": 1,
    "quality_focus_enabled": True,  # NEW: Quality focus mode
    "sales_periods_supported": [
        "sales90d",
        "sales180d",
        "sales365d",
    ],  # NEW: All supported sales periods
}
