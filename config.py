"""
Configuration file for Amazon Business Data Processor
Customize these settings to match your data structure and requirements
"""

# Input file configuration
DEFAULT_INPUT_FILE = "202502 Brands Reports v5.xlsx"

# Additional files that may be needed
ADDITIONAL_FILES = {
    'business_report': "BusinessReport-4-24-25.csv",
    'brands_asin_list': "Brands and ASINs list.xlsx", 
    'unit_financial_extra': "202502 UF.xlsx"
}

# Excel sheet names (customize if your sheets have different names)
SHEET_NAMES = {
    'asin_brand': "ASIN-Brand",
    'business': "Business Report", 
    'transaction': "Transaction Report",
    'return': "Return Report",
    'selling_econ': "Selling Econ",
    'storage': "Storage",
    'lt_storage': "LT Storage", 
    'unit_financial': "Unit Financial",
    'shipping_label': "Shipping + Label"
}

# Column names that may vary between files
COLUMN_MAPPINGS = {
    'sku_column': 'SKU',
    'asin_column': '(Child) ASIN',
    'product_sales_column': 'product sales',
    'selling_fees_column': 'selling fees',
    'fba_fees_column': 'fba fees',
    'quantity_column': 'quantity',
    'wholesale_price_column': 'Current Wholesale Price'
}

# Output file names
OUTPUT_FILES = {
    'main_report': 'full_detailed_aggregated_report.xlsx',
    'brand_report': 'full_detailed_aggregated_report_brand.xlsx',
    'gm_report': 'full_detailed_aggregated_report_gm.xlsx', 
    'underperforming_report': 'underperforming_ats_report.xlsx',
    'storage_allocation': 'monthly_fba_storage_allocation.xlsx',
    'lt_storage_summary': 'long_term_storage_summary.xlsx',
    'refund_summary': 'refund_summary_all_asins.xlsx'
}

# Processing settings
PROCESSING_CONFIG = {
    'storage_fee_threshold': 0.001,  # Minimum storage fee to include
    'recent_values_threshold': 0.0001,  # Threshold for recent values
    'currency_format': '${:,.2f}',  # Currency formatting
    'percentage_format': '{:.2f}%',  # Percentage formatting
    'number_format': '{:,}',  # Number formatting with commas
}

# Required columns for shipping and label fees
SHIPPING_LABEL_COLUMNS = [
    "FBA inbound placement service fee per unit",
    "Inbound Transportation Fee per unit", 
    "Label per unit",
    "Bagging per unit",
    "FBA disposal order fee per unit",
    "FBA removal order fee per unit",
    "Refund administration fee per unit",
    "Returns Processing Fee for Non-Apparel and Non-Shoes per unit",
    "Returns processing fee for Apparel and Shoes per unit"
]

# Business metrics columns
BUSINESS_METRICS_COLUMNS = [
    'Featured Offer (Buy Box) Percentage',
    'Sessions - Total', 
    'Unit Session Percentage'
]

# Money columns for aggregation
MONEY_COLUMNS = [
    'product sales',
    'Product Cost', 
    'selling fees',
    'fba fees',
    'other transaction fees',
    'Shipping Fee',
    'Label Fee', 
    'total storage cost',
    'Allocatable Fees',
    'Net Cost of Return'
]

