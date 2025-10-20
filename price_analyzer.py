"""
Price analysis and comparison utilities with enhanced EAN support
"""

import logging
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional
from datetime import datetime
from models import ProductData, PriceComparison, ProductPrice

logger = logging.getLogger(__name__)


class PriceAnalyzer:
    """Enhanced price analysis with universal EAN support"""

    def __init__(self):
        self.products_by_supplier = {}
        self.ean_comparisons = {}
        self.all_products = []
        self.analysis_complete = False

    def load_products(self, products: List[ProductData]) -> None:
        """Load products for analysis with enhanced EAN filtering"""

        # Reset state
        self.products_by_supplier = defaultdict(list)
        self.all_products = products
        self.analysis_complete = False

        if not products:
            logger.warning("No products provided for price analysis")
            return

        # Filter and group products with EAN codes
        ean_products = []
        for product in products:
            # Only include products with EAN codes and valid prices
            if (
                product.ean_code
                and str(product.ean_code).strip()
                and product.price is not None
                and product.price > 0
            ):
                ean_products.append(product)

                # Group by supplier
                supplier = product.supplier or "Unknown Supplier"
                self.products_by_supplier[supplier].append(product)

        logger.info(f"📊 Price Analysis Setup:")
        logger.info(f"   Total input products: {len(products)}")
        logger.info(f"   Products with valid EAN+Price: {len(ean_products)}")
        logger.info(f"   Unique suppliers: {len(self.products_by_supplier)}")

        if len(ean_products) == 0:
            logger.warning(
                "⚠️ No products with valid EAN codes and prices found for analysis"
            )
        elif len(self.products_by_supplier) < 2:
            logger.warning(
                "⚠️ Less than 2 suppliers found - price comparison will be limited"
            )

    def analyze_price_comparisons(self) -> Dict[str, PriceComparison]:
        """Analyze prices across suppliers for each EAN code"""

        if not self.products_by_supplier:
            logger.warning("No products loaded for price analysis")
            return {}

        # Group all products by EAN code
        ean_groups = defaultdict(list)

        for supplier_name, products in self.products_by_supplier.items():
            for product in products:
                if (
                    product.ean_code
                    and str(product.ean_code).strip()
                    and product.price is not None
                    and product.price > 0
                ):

                    price_info = ProductPrice(
                        ean_code=str(product.ean_code).strip(),
                        supplier=supplier_name,
                        price=float(product.price),
                        product_name=product.product_name,
                        supplier_code=product.supplier_code,
                        source_file=product.source_file,
                    )
                    ean_groups[product.ean_code].append(price_info)

        logger.info(f"🔍 Analysis Input: {len(ean_groups)} unique EAN codes")

        if not ean_groups:
            logger.warning("No valid EAN groups found for analysis")
            return {}

        # Analyze each EAN group
        comparisons = {}
        competitive_products = 0
        unique_products = 0

        for ean_code, price_list in ean_groups.items():
            try:
                if len(price_list) > 1:  # Multiple suppliers offer this EAN
                    competitive_products += 1

                    prices = [float(p.price) for p in price_list]
                    best_price = min(prices)
                    worst_price = max(prices)
                    best_supplier_info = min(price_list, key=lambda x: float(x.price))

                    comparison = PriceComparison(
                        ean_code=str(ean_code),
                        product_name=best_supplier_info.product_name,
                        best_price=best_price,
                        best_supplier=best_supplier_info.supplier,
                        total_suppliers=len(price_list),
                        price_range=(best_price, worst_price),
                        savings_opportunity=worst_price - best_price,
                        all_suppliers=price_list,
                    )
                    comparisons[ean_code] = comparison

                else:  # Single supplier offers this EAN
                    unique_products += 1
                    supplier_info = price_list[0]

                    comparison = PriceComparison(
                        ean_code=str(ean_code),
                        product_name=supplier_info.product_name,
                        best_price=float(supplier_info.price),
                        best_supplier=supplier_info.supplier,
                        total_suppliers=1,
                        price_range=(
                            float(supplier_info.price),
                            float(supplier_info.price),
                        ),
                        savings_opportunity=0.0,
                        all_suppliers=price_list,
                    )
                    comparisons[ean_code] = comparison

            except Exception as e:
                logger.error(f"Error analyzing EAN {ean_code}: {e}")
                continue

        self.ean_comparisons = comparisons
        self.analysis_complete = True

        logger.info(f"✅ Price Analysis Complete:")
        logger.info(
            f"   Competitive products (multiple suppliers): {competitive_products}"
        )
        logger.info(f"   Unique products (single supplier): {unique_products}")
        logger.info(f"   Total analyzed EANs: {len(comparisons)}")

        return comparisons

    def get_supplier_analysis(self, supplier_name: str) -> Dict:
        """Get detailed analysis for a specific supplier"""

        if supplier_name not in self.products_by_supplier:
            return {"error": f"Supplier '{supplier_name}' not found in analysis"}

        supplier_products = self.products_by_supplier[supplier_name]

        # Initialize analysis structure
        analysis = {
            "supplier_name": supplier_name,
            "total_products": len(supplier_products),
            "products_with_ean": sum(1 for p in supplier_products if p.ean_code),
            "best_price_products": [],
            "competitive_products": [],
            "unique_products": [],
            "summary": {
                "unique_count": 0,
                "best_price_count": 0,
                "competitive_count": 0,
                "total_analyzed": 0,
                "potential_savings": 0.0,
            },
        }

        # Ensure analysis is complete
        if not self.analysis_complete:
            logger.info("Price analysis not complete, running analysis first")
            self.analyze_price_comparisons()

        # Analyze each product with EAN
        for product in supplier_products:
            if not product.ean_code or product.price is None:
                continue

            comparison = self.ean_comparisons.get(product.ean_code)
            if not comparison:
                continue

            try:
                product_analysis = {
                    "ean_code": str(product.ean_code),
                    "product_name": product.product_name or "N/A",
                    "price": float(product.price),
                    "best_market_price": float(comparison.best_price),
                    "best_supplier": comparison.best_supplier,
                    "total_suppliers": comparison.total_suppliers,
                    "savings_opportunity": float(comparison.savings_opportunity),
                    "is_best_price": abs(
                        float(product.price) - float(comparison.best_price)
                    )
                    < 0.01,
                    "price_difference": float(product.price)
                    - float(comparison.best_price),
                }

                if comparison.total_suppliers == 1:
                    analysis["unique_products"].append(product_analysis)
                elif product_analysis["is_best_price"]:
                    analysis["best_price_products"].append(product_analysis)
                else:
                    analysis["competitive_products"].append(product_analysis)
                    analysis["summary"]["potential_savings"] += max(
                        0, product_analysis["price_difference"]
                    )

            except Exception as e:
                logger.error(
                    f"Error analyzing product {product.ean_code} for supplier {supplier_name}: {e}"
                )
                continue

        # Update summary counts
        analysis["summary"].update(
            {
                "unique_count": len(analysis["unique_products"]),
                "best_price_count": len(analysis["best_price_products"]),
                "competitive_count": len(analysis["competitive_products"]),
                "total_analyzed": sum(
                    [
                        len(analysis["unique_products"]),
                        len(analysis["best_price_products"]),
                        len(analysis["competitive_products"]),
                    ]
                ),
            }
        )

        logger.info(
            f"📊 Supplier '{supplier_name}' analysis: {analysis['summary']['total_analyzed']} products analyzed"
        )

        return analysis

    def get_top_opportunities(self, limit: int = 10) -> List[PriceComparison]:
        """Get top savings opportunities across all suppliers"""

        if not self.analysis_complete:
            logger.info("Running price analysis to get opportunities")
            self.analyze_price_comparisons()

        # Filter competitive products with savings potential
        competitive_comparisons = [
            comp
            for comp in self.ean_comparisons.values()
            if comp.total_suppliers > 1 and comp.savings_opportunity > 0
        ]

        # Sort by savings opportunity (highest first)
        sorted_opportunities = sorted(
            competitive_comparisons, key=lambda x: x.savings_opportunity, reverse=True
        )

        top_opportunities = sorted_opportunities[:limit]

        logger.info(
            f"💡 Found {len(competitive_comparisons)} savings opportunities, returning top {len(top_opportunities)}"
        )

        return top_opportunities

    def get_market_summary(self) -> Dict:
        """Get overall market analysis summary"""

        if not self.analysis_complete:
            logger.info("Running price analysis for market summary")
            self.analyze_price_comparisons()

        if not self.ean_comparisons:
            return {
                "error": "No price comparisons available - insufficient data with EAN codes"
            }

        total_products = len(self.ean_comparisons)
        competitive_products = sum(
            1 for c in self.ean_comparisons.values() if c.total_suppliers > 1
        )
        unique_products = total_products - competitive_products

        summary = {
            "total_ean_products": total_products,
            "competitive_products": competitive_products,
            "unique_products": unique_products,
            "total_suppliers": len(self.products_by_supplier),
            "supplier_breakdown": {},
        }

        # Calculate supplier performance
        supplier_wins = defaultdict(int)
        supplier_products = defaultdict(int)
        supplier_total_savings = defaultdict(float)

        for comparison in self.ean_comparisons.values():
            # Count products per supplier
            for supplier_info in comparison.all_suppliers:
                supplier_products[supplier_info.supplier] += 1

            # Count wins (best prices) per supplier
            if comparison.total_suppliers > 1:
                supplier_wins[comparison.best_supplier] += 1

                # Calculate potential savings for each supplier
                for supplier_info in comparison.all_suppliers:
                    if supplier_info.supplier != comparison.best_supplier:
                        savings = supplier_info.price - comparison.best_price
                        supplier_total_savings[supplier_info.supplier] += max(
                            0, savings
                        )

        # Build supplier breakdown
        summary["supplier_breakdown"] = {}
        for supplier in self.products_by_supplier.keys():
            total_prods = supplier_products.get(supplier, 0)
            wins = supplier_wins.get(supplier, 0)
            win_rate = (wins / total_prods * 100) if total_prods > 0 else 0
            potential_savings = supplier_total_savings.get(supplier, 0)

            summary["supplier_breakdown"][supplier] = {
                "total_products": total_prods,
                "best_prices": wins,
                "win_rate": round(win_rate, 1),
                "potential_savings": round(potential_savings, 2),
            }

        # Calculate total market potential savings
        if competitive_products > 0:
            try:
                total_savings = sum(
                    c.savings_opportunity
                    for c in self.ean_comparisons.values()
                    if c.total_suppliers > 1 and c.savings_opportunity > 0
                )
                summary["total_potential_savings"] = round(total_savings, 2)
                summary["average_savings_per_product"] = round(
                    total_savings / competitive_products, 2
                )
            except Exception as e:
                logger.error(f"Error calculating market savings: {e}")
                summary["total_potential_savings"] = 0
                summary["average_savings_per_product"] = 0

        logger.info(
            f"📈 Market Summary: {competitive_products} competitive products, €{summary.get('total_potential_savings', 0):.2f} potential savings"
        )

        return summary

    def export_analysis_data(self) -> Dict:
        """Export all analysis data for download"""

        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "analysis_metadata": {
                    "total_input_products": len(self.all_products),
                    "products_analyzed": len(self.ean_comparisons),
                    "suppliers_analyzed": len(self.products_by_supplier),
                    "analysis_complete": self.analysis_complete,
                },
                "market_summary": self.get_market_summary(),
                "price_comparisons": [],
                "supplier_analyses": {},
            }

            # Add price comparisons
            for comp in self.ean_comparisons.values():
                export_data["price_comparisons"].append(
                    {
                        "ean_code": comp.ean_code,
                        "product_name": comp.product_name,
                        "best_price": comp.best_price,
                        "best_supplier": comp.best_supplier,
                        "total_suppliers": comp.total_suppliers,
                        "price_range": comp.price_range,
                        "savings_opportunity": comp.savings_opportunity,
                        "all_supplier_prices": [
                            {
                                "supplier": sp.supplier,
                                "price": sp.price,
                                "product_name": sp.product_name,
                            }
                            for sp in comp.all_suppliers
                        ],
                    }
                )

            # Add supplier analyses
            for supplier in self.products_by_supplier.keys():
                try:
                    supplier_analysis = self.get_supplier_analysis(supplier)
                    if "error" not in supplier_analysis:
                        export_data["supplier_analyses"][supplier] = supplier_analysis
                except Exception as e:
                    logger.error(
                        f"Error exporting supplier analysis for {supplier}: {e}"
                    )

            logger.info(
                f"📦 Export data prepared: {len(export_data['price_comparisons'])} comparisons, {len(export_data['supplier_analyses'])} supplier analyses"
            )

            return export_data

        except Exception as e:
            logger.error(f"Error preparing export data: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": f"Export failed: {str(e)}",
                "analysis_metadata": {
                    "total_input_products": (
                        len(self.all_products) if hasattr(self, "all_products") else 0
                    ),
                    "analysis_complete": False,
                },
            }

    def get_analysis_status(self) -> Dict:
        """Get current analysis status and statistics"""

        return {
            "analysis_complete": self.analysis_complete,
            "products_loaded": (
                len(self.all_products) if hasattr(self, "all_products") else 0
            ),
            "suppliers_found": len(self.products_by_supplier),
            "ean_comparisons": len(self.ean_comparisons),
            "competitive_products": sum(
                1 for c in self.ean_comparisons.values() if c.total_suppliers > 1
            ),
            "unique_products": sum(
                1 for c in self.ean_comparisons.values() if c.total_suppliers == 1
            ),
        }

    def reset_analysis(self) -> None:
        """Reset the analyzer for new data"""

        self.products_by_supplier = {}
        self.ean_comparisons = {}
        self.all_products = []
        self.analysis_complete = False

        logger.info("🔄 Price analyzer reset")
