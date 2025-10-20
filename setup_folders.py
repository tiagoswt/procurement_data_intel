"""
Setup script for AI Procurement Processor - Automatic File Loading
Creates the required folder structure and provides usage instructions
"""

import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_folder_structure(base_folder="data"):
    """Create the required folder structure for automatic file loading"""

    base_path = Path(base_folder)

    folders = {
        "suppliers": "Supplier catalog files (CSV/XLSX)",
        "internal_data": "Internal product data files (CSV)",
        "order_files": "Order/buying list files (CSV)",
    }

    print("🤖 AI Procurement Processor - Setup")
    print("=" * 50)

    # Create base folder
    base_path.mkdir(exist_ok=True)
    print(f"📁 Created base folder: {base_path}")

    # Create subfolders
    for folder_name, description in folders.items():
        folder_path = base_path / folder_name
        folder_path.mkdir(exist_ok=True)
        print(f"📁 Created folder: {folder_path} - {description}")

        # Create a README file in each folder
        readme_path = folder_path / "README.md"
        create_readme_file(readme_path, folder_name, description)

    print(f"\n✅ Folder structure created successfully!")
    print(f"📂 Base location: {base_path.absolute()}")

    # Create .gitkeep files to ensure folders are tracked in git
    for folder_name in folders.keys():
        gitkeep_path = base_path / folder_name / ".gitkeep"
        gitkeep_path.touch()

    return base_path


def create_readme_file(readme_path, folder_name, description):
    """Create README files for each folder with usage instructions"""

    content = {
        "suppliers": """# Supplier Data Folder

## Purpose
This folder contains supplier catalog files with product information, EAN codes, and prices.

## Supported Formats
- CSV files (.csv)
- Excel files (.xlsx, .xls)

## Required Data Structure
Your supplier files should contain columns with:
- **Product codes** (EAN, Gencod, UPC, SKU, etc.)
- **Product names/descriptions**
- **Prices** (unit prices)
- **Quantities** (optional)

## Column Name Examples
The system will automatically detect these column patterns:
- Product codes: `EAN`, `Gencod`, `Code`, `SKU`, `Reference`, `Product_Code`
- Names: `Product_Name`, `Description`, `Designation`, `Product`
- Prices: `Price`, `Prix`, `Unit_Price`, `Cost`
- Quantities: `Quantity`, `Qty`, `Stock`

## Usage
1. Add your supplier CSV/XLSX files to this folder
2. Go to the application's "Auto File Processing" tab
3. Select files and click "Process Selected Files"
4. The system will automatically detect EAN codes and structure

## Example Files
Place files like:
- `supplier_pharmadirect.csv`
- `supplier_weleda.xlsx`
- `catalog_vichy_2024.csv`
""",
        "internal_data": """# Internal Data Folder

## Purpose
This folder contains your internal product master data for opportunity analysis.

## Required Format
CSV files with these specific columns:

### Required Columns
- `EAN`: Product barcode/EAN codes (required)
- At least one price field:
  - `stockAvgPrice`: Average stock price
  - `supplierPrice`: Current supplier price  
  - `bestbuyPrice(12M)`: Best price in last 12 months

### Optional Columns
- `CNP`: Internal product number
- `brand`: Product brand
- `itemDescriptionEN`: English product description
- `itemCapacity`: Product capacity/size
- `stock`: Current inventory level
- `sales90d`: Sales in last 90 days
- `sales180d`: Sales in last 180 days
- `sales365d`: Sales in last 365 days
- `salesnext90d_lastyear`: Sales forecast
- `bestbuyPrice_supplier(12M)`: Best price supplier name
- `bestSeller`: Bestseller ranking (1-4, where 1-2 are bestsellers)
- `isActive`: Product active status

## Usage
1. Export your product master data as CSV with above columns
2. Place the CSV file in this folder
3. Go to "Opportunities (Auto)" tab
4. Select your file and load it
5. System will analyze opportunities vs supplier quotes

## Example Files
- `internal_products_2024.csv`
- `product_master_data.csv`
""",
        "order_files": """# Order Files Folder

## Purpose
This folder contains buying lists and order files for optimization.

## Supported Formats
CSV files with product requests and quantities.

## Data Structure Options

### Simple Format (Header Row 1)
```csv
EAN;Quantity;Description
3433422404397;5;Anti-Age Serum
1234567890123;2;Moisturizer
```

### Complex Format (Header Row 3)
```csv
Encomenda 176367;
2025-07-01;
n/ ref;qnt;ean;descrição;
007772VE_01;149;3337875597197;Product Name
```

## Column Requirements
- **Product identifier**: EAN codes, product references, or SKUs
- **Quantity**: Number of units needed (optional, defaults to 1)
- **Description**: Product names (optional, helps with matching)

## Usage
1. Place your order CSV files in this folder
2. Go to "Order Optimization (Auto)" tab  
3. Select files and specify header row number
4. Load files and run optimization
5. System will distribute orders to best suppliers by price

## Header Row Examples
- **Row 1**: Simple CSV with headers at top
- **Row 3**: CSV with metadata (order numbers, dates) at top
- **Other**: Count lines in your file to find header row

## Example Files
- `order_january_2024.csv`
- `buying_list_urgent.csv`
- `restock_order_march.csv`
""",
    }

    if folder_name in content:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content[folder_name])


def create_example_files(base_folder="data"):
    """Create example files to demonstrate the expected format"""

    base_path = Path(base_folder)

    # Example supplier file
    supplier_example = """EAN,Product_Name,Price,Supplier,Stock
3433422404397,Anti-Age Serum 30ml,29.99,PharmaDirect,50
1234567890123,Moisturizing Cream 50ml,19.95,PharmaDirect,25
9876543210987,Vitamin C Serum 15ml,39.50,PharmaDirect,30"""

    supplier_file = base_path / "suppliers" / "example_supplier.csv"
    with open(supplier_file, "w", encoding="utf-8") as f:
        f.write(supplier_example)

    # Example internal data file
    internal_example = """EAN,CNP,brand,itemDescriptionEN,stockAvgPrice,supplierPrice,bestbuyPrice(12M),stock,sales90d,bestSeller
3433422404397,CNP001,BrandA,Anti-Age Serum Premium 30ml,32.50,35.00,28.00,15,45,1
1234567890123,CNP002,BrandB,Daily Moisturizer Natural 50ml,22.00,25.00,20.50,8,23,3
9876543210987,CNP003,BrandC,Vitamin C Brightening Serum 15ml,42.00,45.00,38.99,12,67,2"""

    internal_file = base_path / "internal_data" / "example_internal_data.csv"
    with open(internal_file, "w", encoding="utf-8") as f:
        f.write(internal_example)

    # Example order file
    order_example = """EAN;Quantity;Description
3433422404397;10;Anti-Age Serum
1234567890123;5;Moisturizer
9876543210987;8;Vitamin C Serum"""

    order_file = base_path / "order_files" / "example_order.csv"
    with open(order_file, "w", encoding="utf-8") as f:
        f.write(order_example)

    print(f"\n📝 Created example files:")
    print(f"   • {supplier_file}")
    print(f"   • {internal_file}")
    print(f"   • {order_file}")


def show_usage_instructions():
    """Display usage instructions"""

    print(f"\n📖 USAGE INSTRUCTIONS")
    print("=" * 50)

    instructions = """
🚀 Quick Start Guide:

1. **Add Your Data Files**
   - Copy supplier catalogs to: data/suppliers/
   - Copy internal product data to: data/internal_data/
   - Copy order files to: data/order_files/

2. **Run the Application**
   - Execute: streamlit run app.py
   - Select "🤖 Automatic (from folders)" mode
   - Files will be loaded automatically from folders

3. **Processing Workflow**
   a) Auto File Processing → Load & process supplier catalogs
   b) Opportunities (Auto) → Load internal data & find savings
   c) Order Optimization (Auto) → Load orders & optimize suppliers

4. **File Format Support**
   - CSV files (semicolon, comma, or tab separated)
   - Excel files (.xlsx, .xls)
   - Various encodings (UTF-8, CP1252, Latin-1)

5. **Automatic Detection**
   - EAN codes (all formats: EAN-13, EAN-8, UPC, Gencod)
   - Column mapping (product codes, names, prices)
   - File structure (header row detection)

📁 Folder Structure Created:
   data/
   ├── suppliers/     ← Supplier catalog files
   ├── internal_data/ ← Your product master data  
   └── order_files/   ← Buying lists & orders

🔄 Switching Modes:
   - Use "🤖 Automatic" for folder-based loading
   - Use "📤 Manual Upload" for traditional file uploads

📊 Monitoring:
   - Sidebar shows folder status and file counts
   - Click "🔄 Refresh Folders" to rescan
   - Example files provided for testing
"""

    print(instructions)


def main():
    """Main setup function"""

    try:
        # Create folder structure
        base_path = create_folder_structure()

        # Create example files
        create_example_files()

        # Show usage instructions
        show_usage_instructions()

        print(f"\n🎉 Setup completed successfully!")
        print(f"📂 Data folder: {base_path.absolute()}")
        print(f"🚀 Run: streamlit run app.py")

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        raise


if __name__ == "__main__":
    main()
