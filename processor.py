"""
Main processing logic for procurement data with Universal EAN Detection
"""

import logging
import time
import pandas as pd
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime

from models import ProcessingResult, ProductData, FieldMapping
from field_detector import AIFieldDetector
from file_processor import FileProcessor
from data_normalizer import DataNormalizer

logger = logging.getLogger(__name__)


class ProcurementProcessor:
    """Main processor for procurement data with enhanced EAN detection"""

    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize with optional Groq API key for AI field detection"""
        self.field_detector = AIFieldDetector(groq_api_key)
        self.file_processor = FileProcessor()
        self.normalizer = DataNormalizer()

        if groq_api_key or self.field_detector.api_key:
            logger.info("✅ Processor initialized with AI field detection")
        else:
            logger.info("ℹ️ Processor initialized with fallback field detection")

    def process_file(
        self,
        file_path: str,
        supplier_name: Optional[str] = None,
        manual_mapping: Optional[FieldMapping] = None,
    ) -> ProcessingResult:
        """Process a single file with universal EAN detection"""
        start_time = time.time()
        errors = []

        try:
            # Read the file
            logger.info(f"📁 Reading file: {file_path}")
            df = self.file_processor.read_file(file_path)

            # Get file structure for analysis
            headers = df.columns.tolist()
            sample_data = df.iloc[0].to_dict() if len(df) > 0 else {}

            logger.info(f"📋 File structure - Headers: {headers}")
            logger.info(f"📊 Sample data: {sample_data}")

            # Field mapping detection
            if manual_mapping:
                field_mapping = manual_mapping
                logger.info(f"🔧 Using manual field mapping: {field_mapping.to_dict()}")
            else:
                logger.info(
                    "🤖 Starting automatic field detection with universal EAN support"
                )
                field_mapping = self.field_detector.detect_fields(headers, sample_data)
                logger.info(f"✅ Detected field mapping: {field_mapping.to_dict()}")

            # Validate mapping
            validation_result = self._validate_mapping(
                field_mapping, headers, sample_data
            )
            if not validation_result["valid"]:
                errors.extend(validation_result["errors"])
                if validation_result["critical"]:
                    return ProcessingResult(
                        success=False,
                        products=[],
                        errors=errors,
                        processing_time=time.time() - start_time,
                        files_processed=0,
                        total_products=0,
                    )

            # Extract supplier name from filename if not provided
            if not supplier_name:
                supplier_name = Path(file_path).stem

            # Normalize data with universal EAN detection
            logger.info("🎯 Starting data normalization with universal EAN detection")
            products = self.normalizer.normalize_data(df, field_mapping, supplier_name)

            # Enhanced validation
            validation_issues = self.normalizer.validate_products(products)
            if validation_issues:
                errors.extend(validation_issues)
                logger.warning(f"⚠️ Validation issues found: {validation_issues}")

            # Log classification results
            classification_stats = self.normalizer.get_classification_stats(products)
            if classification_stats:
                logger.info(
                    f"📊 Classification results: {classification_stats['ean_classified']} EANs, {classification_stats['supplier_classified']} supplier codes"
                )

            processing_time = time.time() - start_time
            logger.info(f"✅ Processing completed in {processing_time:.2f} seconds")

            return ProcessingResult(
                success=len(products) > 0,
                products=products,
                errors=errors,
                processing_time=processing_time,
                files_processed=1,
                total_products=len(products),
            )

        except Exception as e:
            error_msg = f"Failed to process file {file_path}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

            return ProcessingResult(
                success=False,
                products=[],
                errors=errors,
                processing_time=time.time() - start_time,
                files_processed=0,
                total_products=0,
            )

    def process_uploaded_file(
        self,
        uploaded_file,
        supplier_name: Optional[str] = None,
        manual_mapping: Optional[FieldMapping] = None,
    ) -> ProcessingResult:
        """Process an uploaded file from Streamlit with universal EAN detection"""
        start_time = time.time()
        errors = []

        try:
            # Read the uploaded file
            logger.info(f"📁 Reading uploaded file: {uploaded_file.name}")
            df = self.file_processor.read_uploaded_file(uploaded_file)

            # Get file structure for analysis
            headers = df.columns.tolist()
            sample_data = df.iloc[0].to_dict() if len(df) > 0 else {}

            logger.info(f"📋 File structure - Headers: {headers}")
            logger.info(f"📊 Sample row: {sample_data}")

            # Field mapping detection
            if manual_mapping:
                field_mapping = manual_mapping
                logger.info(f"🔧 Using manual field mapping: {field_mapping.to_dict()}")
            else:
                logger.info("🤖 Starting AI field detection with universal EAN support")
                field_mapping = self.field_detector.detect_fields(headers, sample_data)
                logger.info(f"✅ AI detected mapping: {field_mapping.to_dict()}")

            # Validate mapping
            validation_result = self._validate_mapping(
                field_mapping, headers, sample_data
            )
            if not validation_result["valid"]:
                errors.extend(validation_result["errors"])
                if validation_result["critical"]:
                    return ProcessingResult(
                        success=False,
                        products=[],
                        errors=errors,
                        processing_time=time.time() - start_time,
                        files_processed=0,
                        total_products=0,
                    )

            # Extract supplier name from filename if not provided
            if not supplier_name:
                supplier_name = Path(uploaded_file.name).stem

            # Normalize data with universal EAN detection
            logger.info("🎯 Starting universal data normalization")
            products = self.normalizer.normalize_data(df, field_mapping, supplier_name)

            # Add source file info
            for product in products:
                product.source_file = uploaded_file.name

            # Enhanced validation
            validation_issues = self.normalizer.validate_products(products)
            if validation_issues:
                errors.extend(validation_issues)
                logger.warning(f"⚠️ Validation issues: {validation_issues}")

            # Log detailed results
            summary = self.normalizer.get_data_summary(products)
            logger.info(f"📊 Processing summary:")
            logger.info(f"   Total products: {summary.get('total_products', 0)}")
            logger.info(
                f"   EAN codes: {summary.get('products_with_ean', 0)} ({summary.get('ean_detection_rate', 0):.1f}%)"
            )
            logger.info(
                f"   Supplier codes: {summary.get('products_with_supplier_code', 0)}"
            )

            processing_time = time.time() - start_time
            logger.info(
                f"✅ Upload processing completed in {processing_time:.2f} seconds"
            )

            return ProcessingResult(
                success=len(products) > 0,
                products=products,
                errors=errors,
                processing_time=processing_time,
                files_processed=1,
                total_products=len(products),
            )

        except Exception as e:
            error_msg = (
                f"Failed to process uploaded file {uploaded_file.name}: {str(e)}"
            )
            logger.error(error_msg)
            errors.append(error_msg)

            return ProcessingResult(
                success=False,
                products=[],
                errors=errors,
                processing_time=time.time() - start_time,
                files_processed=0,
                total_products=0,
            )

    def process_multiple_files(self, file_paths: List[str]) -> ProcessingResult:
        """Process multiple files and combine results"""
        start_time = time.time()
        all_products = []
        all_errors = []
        files_processed = 0

        logger.info(
            f"🔄 Processing {len(file_paths)} files with universal EAN detection"
        )

        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"📁 Processing file {i}/{len(file_paths)}: {file_path}")
            result = self.process_file(file_path)

            if result.success:
                all_products.extend(result.products)
                files_processed += 1
                logger.info(f"✅ File {i} success: {result.total_products} products")
            else:
                logger.error(f"❌ File {i} failed: {result.errors}")

            all_errors.extend(result.errors)

        # Combined summary
        total_ean_codes = sum(1 for p in all_products if p.ean_code)
        total_supplier_codes = sum(1 for p in all_products if p.supplier_code)

        processing_time = time.time() - start_time

        logger.info(f"🎉 Batch processing completed:")
        logger.info(f"   Files processed: {files_processed}/{len(file_paths)}")
        logger.info(f"   Total products: {len(all_products)}")
        logger.info(f"   EAN codes: {total_ean_codes}")
        logger.info(f"   Supplier codes: {total_supplier_codes}")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")

        return ProcessingResult(
            success=len(all_products) > 0,
            products=all_products,
            errors=all_errors,
            processing_time=processing_time,
            files_processed=files_processed,
            total_products=len(all_products),
        )

    def _validate_mapping(
        self, field_mapping: FieldMapping, headers: List[str], sample_data: Dict
    ) -> Dict:
        """Enhanced mapping validation with detailed feedback"""

        validation_result = {
            "valid": True,
            "critical": False,
            "errors": [],
            "warnings": [],
        }

        mapping_dict = field_mapping.to_dict()

        # Check if essential fields are mapped
        has_product_code = mapping_dict.get("product_code") is not None
        has_price = mapping_dict.get("price") is not None

        # Critical validation: Must have at least product_code OR price
        if not has_product_code and not has_price:
            validation_result["valid"] = False
            validation_result["critical"] = True
            validation_result["errors"].append(
                "No product_code or price field mapped - cannot process data"
            )
            return validation_result

        # Validate individual fields exist in headers
        for field, column in mapping_dict.items():
            if column and column not in headers:
                validation_result["errors"].append(
                    f"Mapped column '{column}' for {field} not found in file headers"
                )
                validation_result["valid"] = False

        # Check data quality in mapped fields
        if has_product_code:
            product_code_col = mapping_dict["product_code"]
            sample_code = sample_data.get(product_code_col)
            if pd.isna(sample_code) or str(sample_code).strip() == "":
                validation_result["warnings"].append(
                    f"Product code field '{product_code_col}' has empty sample value"
                )

        if has_price:
            price_col = mapping_dict["price"]
            sample_price = sample_data.get(price_col)
            try:
                if pd.isna(sample_price):
                    validation_result["warnings"].append(
                        f"Price field '{price_col}' has empty sample value"
                    )
                else:
                    # Try to convert sample price
                    test_price = float(
                        str(sample_price)
                        .replace(",", ".")
                        .replace("€", "")
                        .replace("$", "")
                    )
                    if test_price < 0 or test_price > 1000000:
                        validation_result["warnings"].append(
                            f"Price field '{price_col}' has unusual sample value: {sample_price}"
                        )
            except (ValueError, TypeError):
                validation_result["warnings"].append(
                    f"Price field '{price_col}' sample value doesn't look like a price: {sample_price}"
                )

        # Log validation results
        if validation_result["errors"]:
            logger.error(f"❌ Mapping validation errors: {validation_result['errors']}")
        if validation_result["warnings"]:
            logger.warning(
                f"⚠️ Mapping validation warnings: {validation_result['warnings']}"
            )
        if validation_result["valid"]:
            logger.info("✅ Field mapping validation passed")

        return validation_result

    def get_field_suggestions(
        self, headers: List[str], sample_data: Dict = None
    ) -> Dict:
        """Get field mapping suggestions for manual review"""
        if sample_data is None:
            sample_data = {}

        return self.field_detector.get_field_suggestions(headers)

    def preview_mapping(self, df: pd.DataFrame, field_mapping: FieldMapping) -> Dict:
        """Preview what data would be extracted with the given mapping"""

        preview = {
            "total_rows": len(df),
            "sample_extractions": [],
            "estimated_ean_codes": 0,
            "estimated_supplier_codes": 0,
            "estimated_products": 0,
        }

        mapping_dict = field_mapping.to_dict()

        # Sample first 5 rows for preview
        sample_count = min(5, len(df))
        for idx in range(sample_count):
            row = df.iloc[idx]
            extraction = {}

            # Extract fields using the mapping
            for field, column in mapping_dict.items():
                if column and column in df.columns:
                    value = row[column]
                    extraction[field] = value if not pd.isna(value) else None
                else:
                    extraction[field] = None

            # Classify the product code if present
            if extraction.get("product_code"):
                ean_code, supplier_code = DataNormalizer._universal_classify_code(
                    extraction["product_code"]
                )
                extraction["classified_as_ean"] = ean_code is not None
                extraction["classified_as_supplier"] = supplier_code is not None

            preview["sample_extractions"].append(extraction)

        # Estimate full dataset results
        ean_estimate = 0
        supplier_estimate = 0
        valid_products = 0

        for idx in range(len(df)):
            row = df.iloc[idx]

            # Check if row would produce a valid product
            product_code = None
            price = None

            if (
                mapping_dict.get("product_code")
                and mapping_dict["product_code"] in df.columns
            ):
                product_code = row[mapping_dict["product_code"]]
            if mapping_dict.get("price") and mapping_dict["price"] in df.columns:
                price = row[mapping_dict["price"]]

            # Skip invalid rows
            if (pd.isna(product_code) or str(product_code).strip() == "") or pd.isna(
                price
            ):
                continue

            valid_products += 1

            # Classify the product code
            ean_code, supplier_code = DataNormalizer._universal_classify_code(
                product_code
            )
            if ean_code:
                ean_estimate += 1
            if supplier_code:
                supplier_estimate += 1

        preview.update(
            {
                "estimated_products": valid_products,
                "estimated_ean_codes": ean_estimate,
                "estimated_supplier_codes": supplier_estimate,
                "ean_detection_rate": (
                    (ean_estimate / valid_products * 100) if valid_products > 0 else 0
                ),
            }
        )

        logger.info(
            f"📊 Mapping preview: {valid_products} products, {ean_estimate} EANs ({preview['ean_detection_rate']:.1f}%)"
        )

        return preview

    def get_processing_stats(self, products: List[ProductData]) -> Dict:
        """Get comprehensive processing statistics"""

        if not products:
            return {"total_products": 0}

        # Use the enhanced summary from data normalizer
        summary = self.normalizer.get_data_summary(products)

        # Add classification stats
        classification_stats = self.normalizer.get_classification_stats(products)
        summary.update(classification_stats)

        # Add supplier breakdown
        suppliers = {}
        for product in products:
            supplier = product.supplier or "Unknown"
            if supplier not in suppliers:
                suppliers[supplier] = {
                    "total": 0,
                    "ean_codes": 0,
                    "supplier_codes": 0,
                    "avg_price": 0,
                    "total_value": 0,
                }

            suppliers[supplier]["total"] += 1
            if product.ean_code:
                suppliers[supplier]["ean_codes"] += 1
            if product.supplier_code:
                suppliers[supplier]["supplier_codes"] += 1
            if product.price:
                suppliers[supplier]["total_value"] += product.price

        # Calculate averages
        for supplier_data in suppliers.values():
            if supplier_data["total"] > 0:
                supplier_data["avg_price"] = (
                    supplier_data["total_value"] / supplier_data["total"]
                )

        summary["suppliers"] = suppliers

        return summary
