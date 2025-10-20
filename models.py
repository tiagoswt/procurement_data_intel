"""
Data models for the AI Procurement Processing System
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, TypedDict
from datetime import datetime


@dataclass
class ProductData:
    """Standardized product data structure"""

    ean_code: Optional[str] = None  # EAN codes (exactly 13 characters)
    supplier_code: Optional[str] = None  # Supplier codes (any length != 13)
    product_name: Optional[str] = None  # Optional
    quantity: Optional[Union[int, float]] = None  # Optional
    price: float = None  # Mandatory
    supplier: Optional[str] = None
    confidence_score: Optional[float] = None
    source_file: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ProductPrice:
    """Product price information for comparison"""

    ean_code: str
    supplier: str
    price: float
    product_name: Optional[str] = None
    supplier_code: Optional[str] = None
    source_file: Optional[str] = None


@dataclass
class PriceComparison:
    """Price comparison result for an EAN"""

    ean_code: str
    product_name: Optional[str] = None
    best_price: float = None
    best_supplier: str = None
    total_suppliers: int = 0
    price_range: tuple = None
    savings_opportunity: float = 0.0
    all_suppliers: List[ProductPrice] = None


class ProcessingState(TypedDict):
    """State for the processing workflow"""

    files_to_process: List[str]
    current_file: Optional[str]
    file_headers: Optional[List[str]]
    sample_row: Optional[Dict]
    extracted_products: List[ProductData]
    errors: List[str]
    processing_complete: bool
    output_path: Optional[str]
    individual_outputs: Optional[List[str]]


@dataclass
class ProcessingResult:
    """Result of processing operation"""

    success: bool
    products: List[ProductData]
    errors: List[str]
    processing_time: float
    files_processed: int
    total_products: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "success": self.success,
            "products": [p.to_dict() for p in self.products],
            "errors": self.errors,
            "processing_time": self.processing_time,
            "files_processed": self.files_processed,
            "total_products": self.total_products,
        }


@dataclass
class FieldMapping:
    """Field mapping configuration"""

    product_code: Optional[str] = None
    product_name: Optional[str] = None
    quantity: Optional[str] = None
    price: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {
            "product_code": self.product_code,
            "product_name": self.product_name,
            "quantity": self.quantity,
            "price": self.price,
        }
