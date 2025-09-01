import pandas as pd
import json

# Path to your Excel file
excel_path = "GM by Brand.xlsx"

# Load the sheet (assumes data is in 'Sheet1')
df = pd.read_excel(excel_path, sheet_name="Sheet1")

# Extract unique non-null values from 'Brands' and 'GM' columns
unique_brands = sorted(df['Brands'].dropna().unique().tolist())
unique_gms = sorted(df['GM'].dropna().unique().tolist())

# Build metadata cache dictionary
metadata_cache = {
    "brands": unique_brands,
    "gms": unique_gms
}

# Output file path
output_path = "metadata_cache.json"

# Save to JSON file
with open(output_path, "w") as f:
    json.dump(metadata_cache, f, indent=2)

print(f"✅ metadata_cache.json created with {len(unique_brands)} brands and {len(unique_gms)} GMs.")