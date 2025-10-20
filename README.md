🤖 AI Procurement Data Processor - POC
A modular, web-based Proof of Concept for intelligent extraction and analysis of supplier procurement data using AI-powered field detection and price comparison.

🚀 Features
📁 Multi-file Processing: Upload and process multiple CSV/Excel files simultaneously
🤖 AI Field Detection: Automatic field mapping using Groq AI (with intelligent fallback)
🔍 Manual Field Mapping: Override automatic detection with custom field mappings
📊 Data Normalization: Smart EAN/UPC code detection and data validation
💰 Price Analysis: Cross-supplier price comparison and savings opportunities
📈 Interactive Dashboard: Real-time processing status and comprehensive results
💾 Multiple Export Formats: CSV, JSON, and detailed reports
🏗️ Architecture
The application is now fully modularized for better maintainability and scalability:

├── models.py              # Data models and structures
├── field_detector.py      # AI-powered field detection
├── file_processor.py      # File reading and validation
├── data_normalizer.py     # Data cleaning and normalization
├── processor.py           # Main processing orchestration
├── price_analyzer.py      # Price comparison and analysis
├── utils.py              # Utility functions
├── app.py                # Streamlit web application
├── requirements_streamlit.txt
└── README.md
🛠️ Installation
Prerequisites
Python 3.8 or higher
Groq API key (optional, for AI field detection)
Setup
Clone or download the project files
Install dependencies:
bash
pip install -r requirements_streamlit.txt
Set up environment variables (optional):
bash
# Create a .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
Run the application:
bash
streamlit run app.py
📖 Usage Guide
1. File Processing
Upload Files:
Navigate to the "File Processing" tab
Upload one or more CSV/Excel files containing supplier data
Supported formats: .csv, .xlsx, .xls
Configure Processing:
Set Groq API key for AI field detection (optional)
Adjust file size limits and processing options
Specify supplier name or use filename
Process Data:
Click "Process Files" to start
Monitor real-time progress
Review processing results and any errors
2. Field Mapping
Review Auto-Detection:
View automatically detected field mappings
Check field detection accuracy
Manual Override:
Upload a sample file for manual mapping
Select appropriate columns for each field type:
Product Code (EAN/SKU)
Product Name
Price
Quantity
Preview mapping results before processing
3. Results Analysis
Summary Statistics:
View total products, EAN codes, and data completeness
Review supplier breakdown and price statistics
Data Exploration:
Filter products by supplier, price range, or EAN availability
Export filtered data in multiple formats
Export Options:
Download CSV/JSON data files
Generate comprehensive processing reports
4. Price Analysis
Market Overview:
Compare prices across suppliers
Identify competitive vs. unique products
Savings Opportunities:
View top potential savings across suppliers
Analyze supplier performance and win rates
Supplier Analysis:
Deep-dive into specific supplier performance
Review best prices, competitive products, and unique offerings
🤖 AI Features
Field Detection
Groq AI Integration: Advanced language model for intelligent field mapping
Multi-language Support: Handles English, French, Portuguese, and Spanish headers
Smart Pattern Recognition: Understands domain-specific terminology like "Gencod" (French for EAN)
Fallback Detection: Robust fuzzy matching when AI is unavailable
EAN/UPC Detection
Smart Code Classification: Distinguishes between EAN codes and supplier codes
Multiple Standards: Supports EAN-13, EAN-8, and UPC-A formats
Context-Aware: Uses column names to improve classification accuracy
📊 Supported Data Formats
Input Files
CSV: Auto-detects encoding (UTF-8, Latin-1, CP1252) and delimiters
Excel: .xlsx and .xls formats with automatic sheet detection
Expected Data Structure
The system can handle various column naming conventions:

Field Type	Common Names
Product Code	Gencod, EAN, Code, SKU, Reference, Código
Product Name	Name, Description, Product, Designation
Price	Price, Prix, Preço, Valor, Cost
Quantity	Quantity, Qty, Stock, Quantidade
Output Formats
CSV: Standard comma-separated format
JSON: Structured data with metadata
Reports: Human-readable processing summaries
⚙️ Configuration
Environment Variables
bash
GROQ_API_KEY=your_groq_api_key_here    # Optional: For AI field detection
STREAMLIT_SERVER_PORT=8501             # Optional: Custom port
STREAMLIT_SERVER_ADDRESS=localhost     # Optional: Custom address
Processing Options
Max File Size: Configurable upload limit (default: 50MB)
Auto Field Detection: Toggle AI vs manual field mapping
Validation Rules: Customizable data quality checks
🔧 Development
Adding New Features
New Data Models: Add to models.py
Processing Logic: Extend processor.py or create new modules
UI Components: Add to app.py in appropriate tabs
Utilities: Add helper functions to utils.py
Testing
bash
# Run basic tests
python -m pytest

# Test individual modules
python -c "from processor import ProcurementProcessor; print('✅ Import successful')"
Code Style
bash
# Format code
black *.py

# Check style
flake8 *.py
🚨 Troubleshooting
Common Issues
File Reading Errors:
Check file encoding (try saving as UTF-8)
Verify CSV delimiter (comma, semicolon, tab)
Ensure Excel files aren't corrupted
Field Detection Problems:
Review column names for clarity
Use manual mapping for complex structures
Check sample data quality
API Key Issues:
Verify Groq API key format (gsk_...)
Check API quota and rate limits
Use fallback detection if API unavailable
Error Messages
"No products extracted": Check field mapping and data quality
"File format unsupported": Use CSV, XLSX, or XLS files
"API key invalid": Verify Groq API key format and validity
🔮 Future Enhancements
Planned Features
Batch Processing: Process entire directories
Advanced Analytics: Trend analysis and forecasting
Data Quality Scoring: Automated data quality assessment
Custom Export Templates: Configurable output formats
Multi-tenant Support: User accounts and data isolation
Scalability
Database Integration: PostgreSQL/MongoDB support
Cloud Deployment: AWS/Azure/GCP ready
API Endpoints: RESTful API for programmatic access
Real-time Processing: Webhook integrations
📄 License
This is a Proof of Concept application. Please review and adapt according to your organization's requirements and licensing needs.

🤝 Contributing
Follow the modular architecture
Add comprehensive tests for new features
Update documentation for any new functionality
Use meaningful commit messages
📞 Support
For technical support or feature requests, please review the code documentation and error messages. The modular architecture makes it easy to extend and customize for specific requirements.

Made with ❤️ for procurement professionals seeking intelligent data processing solutions.