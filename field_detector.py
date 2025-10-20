"""
AI-powered field detection and mapping using Groq with Universal EAN Detection
"""

import json
import logging
import os
from typing import Dict, List, Optional
from fuzzywuzzy import fuzz
from groq import Groq
from models import FieldMapping
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AIFieldDetector:
    """AI-powered field detection with universal EAN recognition"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ):
        # Try to get API key from parameter, env file, or environment variable
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model = model

        if self.api_key:
            logger.info("✅ Groq API key loaded - AI detection enabled")
        else:
            logger.info("ℹ️ No Groq API key found - using fallback detection")

    def detect_fields(self, headers: List[str], sample_data: Dict) -> FieldMapping:
        """
        Universal field detection that recognizes ALL EAN/barcode variants
        """

        # Pre-analyze for EAN patterns
        ean_analysis = self._analyze_ean_patterns(headers, sample_data)

        if not self.api_key:
            logger.info("No Groq API key, using enhanced fallback detection")
            return self._enhanced_fallback_detection(headers, sample_data, ean_analysis)

        # Enhanced analysis for AI
        analysis = self._create_comprehensive_analysis(
            headers, sample_data, ean_analysis
        )

        prompt = f"""
        You are a world-class expert in procurement data analysis and barcode systems. Your expertise covers:
        - European barcode standards (EAN-8, EAN-13, EAN-14)
        - North American standards (UPC-A, UPC-E)
        - International standards (GTIN)
        - Regional naming conventions (Gencod in France, Código in Spain/Portugal)

        CRITICAL MISSION: Detect ALL possible barcode/EAN fields as product_code, regardless of naming convention.

        COLUMN ANALYSIS:
        {analysis}

        UNIVERSAL BARCODE RECOGNITION RULES:
        
        1. **Column Name Patterns** (ANY of these = product_code):
           - Standard: "EAN", "UPC", "GTIN", "Barcode", "Code"
           - French: "Gencod", "Code Gencod", "Code Barre"
           - Spanish/Portuguese: "Código", "Codigo", "EAN13"
           - Generic: "SKU", "Reference", "ID", "Product_Code", "Item_Code"
           - Variations: "ProductID", "ItemID", "ArticleCode", "Ref"

        2. **Value Pattern Analysis** (ANY of these = product_code):
           - 7-14 digit numbers (covers all barcode formats)
           - Mixed alphanumeric codes with 70%+ digits
           - Values like: "3433422404397", "123456789012", "87654321"

        3. **Field Mapping Priority**:
           - product_code: ANY barcode/code field (HIGHEST PRIORITY)
           - price: Monetary values, prices, costs
           - product_name: Descriptions, names, titles
           - quantity: Stock, quantities, amounts

        4. **Examples of CORRECT Detection**:
           - "Gencod" with "3433422404397" → product_code ✅
           - "Code" with "123456789" → product_code ✅
           - "SKU" with "ABC-123-789" → product_code ✅
           - "EAN" with "1234567890123" → product_code ✅

        RETURN ONLY THIS JSON FORMAT:
        {{
            "product_code": "exact_column_name_or_null",
            "product_name": "exact_column_name_or_null", 
            "quantity": "exact_column_name_or_null",
            "price": "exact_column_name_or_null"
        }}
        
        IMPORTANT: Be LIBERAL with product_code detection. When in doubt, choose the column most likely to contain barcodes/codes.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a barcode detection expert. ALWAYS detect code-like columns as product_code. Return ONLY valid JSON without explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )

            content = response.choices[0].message.content.strip()
            logger.info(f"🤖 Groq AI response: {content}")

            # Clean and parse JSON
            content = self._clean_json_response(content)
            mapping_dict = json.loads(content)

            # Override with EAN analysis if AI missed obvious patterns
            if not mapping_dict.get("product_code") and ean_analysis["best_candidate"]:
                mapping_dict["product_code"] = ean_analysis["best_candidate"]["column"]
                logger.info(
                    f"🔧 AI Override: Forced '{ean_analysis['best_candidate']['column']}' as product_code"
                )

            field_mapping = FieldMapping(
                product_code=mapping_dict.get("product_code"),
                product_name=mapping_dict.get("product_name"),
                quantity=mapping_dict.get("quantity"),
                price=mapping_dict.get("price"),
            )

            # Validate and enhance the mapping
            validated_mapping = self._validate_and_enhance_mapping(
                field_mapping, headers, sample_data, ean_analysis
            )

            logger.info(f"✅ Final AI mapping: {validated_mapping.to_dict()}")
            return validated_mapping

        except Exception as e:
            logger.error(f"🚨 Groq AI detection failed: {e}")
            logger.info("🔧 Falling back to enhanced pattern detection...")
            return self._enhanced_fallback_detection(headers, sample_data, ean_analysis)

    def _analyze_ean_patterns(self, headers: List[str], sample_data: Dict) -> Dict:
        """Comprehensive EAN pattern analysis"""

        # All possible barcode/EAN column patterns
        barcode_patterns = [
            # Standard international
            "ean",
            "upc",
            "gtin",
            "barcode",
            "code",
            "sku",
            "id",
            "ref",
            "reference",
            # French specific
            "gencod",
            "gencode",
            "gen_cod",
            "gen cod",
            "code_gencod",
            "code_barre",
            "codebarre",
            # Spanish/Portuguese
            "codigo",
            "código",
            "ean13",
            "ean8",
            "codigobarra",
            "codigo_barra",
            # Generic variations
            "product_code",
            "productcode",
            "item_code",
            "itemcode",
            "article_code",
            "product_id",
            "productid",
            "item_id",
            "itemid",
            "article_id",
            # Other variations
            "numero",
            "number",
            "ref_produit",
            "reference_produit",
            "codprod",
        ]

        candidates = []

        for header in headers:
            score = 0
            reasons = []
            header_lower = (
                header.lower()
                .strip()
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "")
            )

            # Check column name patterns
            for pattern in barcode_patterns:
                pattern_clean = (
                    pattern.replace("_", "").replace("-", "").replace(" ", "")
                )
                if pattern_clean in header_lower or header_lower in pattern_clean:
                    score += 50
                    reasons.append(f"Column name matches '{pattern}'")
                    break

            # Check sample value patterns
            sample_value = sample_data.get(header)
            if sample_value is not None:
                clean_value = self._clean_barcode_value(sample_value)

                if clean_value.isdigit():
                    length = len(clean_value)
                    if length in [8, 12, 13, 14]:  # Perfect barcode lengths
                        score += 60
                        reasons.append(f"Perfect barcode length ({length} digits)")
                    elif 7 <= length <= 15:  # Reasonable barcode lengths
                        score += 40
                        reasons.append(f"Reasonable barcode length ({length} digits)")
                    elif length >= 5:
                        score += 20
                        reasons.append(f"Numeric code ({length} digits)")

                # Check mixed alphanumeric
                elif clean_value and len(clean_value) >= 5:
                    digit_ratio = sum(1 for c in clean_value if c.isdigit()) / len(
                        clean_value
                    )
                    if digit_ratio >= 0.7:
                        score += 30
                        reasons.append(f"Mostly numeric ({digit_ratio:.0%} digits)")
                    elif digit_ratio >= 0.3:
                        score += 15
                        reasons.append(f"Mixed alphanumeric ({digit_ratio:.0%} digits)")

            if score > 0:
                candidates.append(
                    {
                        "column": header,
                        "score": score,
                        "sample_value": sample_value,
                        "reasons": reasons,
                    }
                )

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        analysis = {
            "candidates": candidates,
            "best_candidate": candidates[0] if candidates else None,
            "total_candidates": len(candidates),
        }

        if analysis["best_candidate"]:
            logger.info(
                f"🎯 EAN Analysis: Best candidate '{analysis['best_candidate']['column']}' (score: {analysis['best_candidate']['score']})"
            )

        return analysis

    def _create_comprehensive_analysis(
        self, headers: List[str], sample_data: Dict, ean_analysis: Dict
    ) -> str:
        """Create comprehensive analysis for AI"""

        analysis_lines = []
        analysis_lines.append("=== COMPREHENSIVE COLUMN ANALYSIS ===")

        for header in headers:
            sample_value = sample_data.get(header, "")
            analysis = []

            # Check if this is a candidate from EAN analysis
            candidate_info = next(
                (c for c in ean_analysis["candidates"] if c["column"] == header), None
            )
            if candidate_info:
                analysis.append(f"🎯 EAN_CANDIDATE (score: {candidate_info['score']})")
                analysis.extend(candidate_info["reasons"])

            # Value type analysis
            if isinstance(sample_value, (int, float)):
                analysis.append("NUMERIC_VALUE")
                if str(sample_value).replace(".", "").isdigit():
                    clean_num = (
                        str(int(sample_value))
                        if isinstance(sample_value, float)
                        else str(sample_value)
                    )
                    analysis.append(f"CLEAN_DIGITS: {len(clean_num)} chars")

            elif isinstance(sample_value, str):
                analysis.append("TEXT_VALUE")
                if len(sample_value) > 30:
                    analysis.append("LONG_TEXT (likely description)")
                elif len(sample_value) < 20:
                    analysis.append("SHORT_TEXT (likely code)")

            # Additional context
            if any(
                word in header.lower() for word in ["price", "cost", "value", "amount"]
            ):
                analysis.append("💰 PRICE_INDICATOR")
            if any(
                word in header.lower()
                for word in ["name", "description", "title", "product"]
            ):
                analysis.append("📝 NAME_INDICATOR")
            if any(word in header.lower() for word in ["qty", "stock", "quantity"]):
                analysis.append("📦 QUANTITY_INDICATOR")

            analysis_lines.append(
                f"'{header}' → Sample: '{sample_value}' → {' | '.join(analysis)}"
            )

        return "\n".join(analysis_lines)

    def _clean_json_response(self, content: str) -> str:
        """Clean AI response to extract valid JSON"""

        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        # Remove any leading/trailing text
        content = content.strip()

        # Find JSON object boundaries
        start = content.find("{")
        end = content.rfind("}") + 1

        if start >= 0 and end > start:
            content = content[start:end]

        return content

    def _validate_and_enhance_mapping(
        self,
        mapping: FieldMapping,
        headers: List[str],
        sample_data: Dict,
        ean_analysis: Dict,
    ) -> FieldMapping:
        """Validate and enhance the AI mapping"""

        validated = FieldMapping()
        used_columns = set()
        mapping_dict = mapping.to_dict()

        # Priority 1: Ensure product_code is set to best EAN candidate
        if mapping_dict.get("product_code") and mapping_dict["product_code"] in headers:
            validated.product_code = mapping_dict["product_code"]
            used_columns.add(mapping_dict["product_code"])
        elif ean_analysis["best_candidate"]:
            validated.product_code = ean_analysis["best_candidate"]["column"]
            used_columns.add(ean_analysis["best_candidate"]["column"])
            logger.info(f"🔧 Enhanced: Using EAN analysis for product_code")

        # Validate other fields
        for field, column in mapping_dict.items():
            if field == "product_code":
                continue  # Already handled above

            if column and column in headers and column not in used_columns:
                sample_value = sample_data.get(column, "")

                # Basic validation
                if field == "price" and not self._looks_like_price(sample_value):
                    logger.warning(
                        f"Questionable price field: {column} = {sample_value}"
                    )

                setattr(validated, field, column)
                used_columns.add(column)

        return validated

    def _looks_like_price(self, value) -> bool:
        """Check if value looks like a price"""
        if isinstance(value, (int, float)):
            return 0 <= value <= 1000000

        if isinstance(value, str):
            try:
                cleaned = "".join(c for c in value if c.isdigit() or c in ".,")
                price = float(cleaned.replace(",", "."))
                return 0 <= price <= 1000000
            except:
                return False

        return False

    def _enhanced_fallback_detection(
        self, headers: List[str], sample_data: Dict, ean_analysis: Dict
    ) -> FieldMapping:
        """Enhanced fallback detection using pattern analysis"""

        mapping = FieldMapping()
        used_columns = set()

        # Step 1: Use EAN analysis for product_code
        if ean_analysis["best_candidate"]:
            mapping.product_code = ean_analysis["best_candidate"]["column"]
            used_columns.add(ean_analysis["best_candidate"]["column"])
            logger.info(
                f"🎯 Fallback: Using '{ean_analysis['best_candidate']['column']}' as product_code"
            )

        # Step 2: Score other field types
        field_scores = {"price": {}, "product_name": {}, "quantity": {}}

        for header in headers:
            if header in used_columns:
                continue

            header_lower = header.lower().strip()
            sample_value = sample_data.get(header, "")

            # Price scoring
            price_patterns = [
                "price",
                "preço",
                "preco",
                "valor",
                "cost",
                "custo",
                "prix",
                "ht",
                "ttc",
                "tarif",
            ]
            if any(pattern in header_lower for pattern in price_patterns):
                field_scores["price"][header] = 50

            if self._looks_like_price(sample_value):
                field_scores["price"][header] = (
                    field_scores["price"].get(header, 0) + 30
                )

            # Product name scoring
            name_patterns = [
                "name",
                "nome",
                "description",
                "descrição",
                "produto",
                "product",
                "produit",
                "designation",
                "libelle",
                "libellé",
                "titre",
                "title",
            ]
            if any(pattern in header_lower for pattern in name_patterns):
                field_scores["product_name"][header] = 50

            if isinstance(sample_value, str) and len(sample_value) > 20:
                field_scores["product_name"][header] = (
                    field_scores["product_name"].get(header, 0) + 30
                )

            # Quantity scoring
            qty_patterns = [
                "qty",
                "quantidade",
                "quant",
                "stock",
                "units",
                "quantité",
                "qte",
                "pieces",
            ]
            if any(pattern in header_lower for pattern in qty_patterns):
                field_scores["quantity"][header] = 50

            if isinstance(sample_value, (int, float)) and 0 <= sample_value <= 10000:
                field_scores["quantity"][header] = (
                    field_scores["quantity"].get(header, 0) + 20
                )

        # Step 3: Assign best scoring fields
        for field_type, scores in field_scores.items():
            if not scores:
                continue

            best_column = max(scores.items(), key=lambda x: x[1])
            if best_column[1] > 20:  # Minimum threshold
                setattr(mapping, field_type, best_column[0])
                used_columns.add(best_column[0])
                logger.info(
                    f"🔧 Fallback: {field_type} → '{best_column[0]}' (score: {best_column[1]})"
                )

        return mapping

    def _clean_barcode_value(self, value) -> str:
        """Clean barcode value for analysis"""
        if value is None:
            return ""

        value_str = str(value).strip()

        # Remove Excel .0 formatting
        if value_str.endswith(".0"):
            value_str = value_str[:-2]

        # Remove spaces, dashes, and dots
        value_str = value_str.replace(" ", "").replace("-", "").replace(".", "")

        return value_str

    def get_field_suggestions(self, headers: List[str]) -> Dict[str, List[str]]:
        """Get field suggestions for manual mapping"""

        ean_analysis = self._analyze_ean_patterns(headers, {})

        suggestions = {
            "product_code": [c["column"] for c in ean_analysis["candidates"][:3]],
            "product_name": [],
            "quantity": [],
            "price": [],
        }

        # Add other field suggestions using existing logic
        field_patterns = {
            "product_name": [
                "name",
                "description",
                "product",
                "title",
                "item",
                "designation",
            ],
            "quantity": ["quantity", "qty", "amount", "units", "pieces", "stock"],
            "price": ["price", "cost", "unit_price", "amount", "value", "prix"],
        }

        for field, patterns in field_patterns.items():
            for header in headers:
                for pattern in patterns:
                    if fuzz.ratio(header.lower(), pattern.lower()) > 60:
                        suggestions[field].append(header)

        # Remove duplicates and limit to 3 suggestions each
        for field in suggestions:
            suggestions[field] = list(dict.fromkeys(suggestions[field]))[:3]

        return suggestions
