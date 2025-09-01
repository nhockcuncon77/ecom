#!/usr/bin/env python3
"""
Amazon Business Data Processor
A clean rewrite of Final_Data_Process_V1.ipynb logic

This script processes Amazon business data and generates comprehensive reports:
- full_detailed_aggregated_report.xlsx
- full_detailed_aggregated_report_brand.xlsx  
- full_detailed_aggregated_report_gm.xlsx
- underperforming_ats_report.xlsx
- And other supporting reports

Usage:
    python data_processor.py [input_file.xlsx]
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AmazonDataProcessor:
    def __init__(self, input_file: str = "202502 Brands Reports v5.xlsx"):
        """Initialize the data processor with input file."""
        self.input_file = input_file
        self.xlsx = None
        self.dataframes = {}
        self.reports = {}
        
        # Load all data
        self.load_data()
        
    def load_data(self):
        """Load all required data from the Excel file."""
        print(f"📁 Loading data from {self.input_file}...")
        
        try:
            self.xlsx = pd.ExcelFile(self.input_file)
            print(f"✅ Available sheets: {self.xlsx.sheet_names}")
            
            # Load all sheets
            self.dataframes['asin_brand'] = pd.read_excel(self.xlsx, sheet_name="ASIN-Brand")
            self.dataframes['business'] = pd.read_excel(self.xlsx, sheet_name="Business Report")
            self.dataframes['transaction'] = pd.read_excel(self.xlsx, sheet_name="Transaction Report", skiprows=7)
            self.dataframes['return'] = pd.read_excel(self.xlsx, sheet_name="Return Report")
            self.dataframes['selling_econ'] = pd.read_excel(self.xlsx, sheet_name="Selling Econ")
            self.dataframes['storage'] = pd.read_excel(self.xlsx, sheet_name="Storage")
            self.dataframes['lt_storage'] = pd.read_excel(self.xlsx, sheet_name="LT Storage")
            self.dataframes['unit_financial'] = pd.read_excel(self.xlsx, sheet_name="Unit Financial", skiprows=3)
            self.dataframes['shipping_label'] = pd.read_excel(self.xlsx, sheet_name="Shipping + Label")
            
            # Load additional files if they exist
            self.load_additional_files()
            
            print("✅ All data loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    def load_additional_files(self):
        """Load additional required files."""
        additional_files = {
            'business_report': "BusinessReport-4-24-25.csv",
            'brands_asin_list': "Brands and ASINs list.xlsx",
            'unit_financial_extra': "202502 UF.xlsx"
        }
        
        for name, filename in additional_files.items():
            if os.path.exists(filename):
                try:
                    if filename.endswith('.csv'):
                        self.dataframes[name] = pd.read_csv(filename)
                    else:
                        self.dataframes[name] = pd.read_excel(filename)
                    print(f"✅ Loaded {filename}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {filename}: {e}")
            else:
                print(f"⚠️ Warning: {filename} not found")
    
    def clean_numeric_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Clean numeric columns by removing $ and commas."""
        df_clean = df.copy()
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(
                    df_clean[col].astype(str).replace('[\$,]', '', regex=True),
                    errors='coerce'
                ).fillna(0)
        return df_clean
    
    def process_orders_and_aggregate(self) -> pd.DataFrame:
        """Process orders and aggregate by ASIN."""
        print("🔄 Processing orders and aggregating by ASIN...")
        
        # Filter orders
        orders_df = self.dataframes['transaction'][
            self.dataframes['transaction']['type'] == 'Order'
        ].copy()
        
        # Clean monetary columns
        to_clean_cols = ['product sales', 'selling fees', 'fba fees', 'other transaction fees', 'quantity']
        orders_df = self.clean_numeric_columns(orders_df, to_clean_cols)
        
        # Prepare SKU → ASIN mapping
        if 'business_report' in self.dataframes:
            sku_asin_map = self.dataframes['business_report'][['SKU', '(Child) ASIN']].dropna()
        else:
            sku_asin_map = self.dataframes['business'][['SKU', '(Child) ASIN']].dropna()
        
        # Handle SKU conflicts
        conflict_skus = sku_asin_map.groupby('SKU')['(Child) ASIN'].nunique()
        conflict_skus = conflict_skus[conflict_skus > 1].index.tolist()
        
        if conflict_skus:
            print(f"⚠️ Warning: {len(conflict_skus)} SKUs map to multiple ASINs. Keeping first ASIN for each SKU.")
        
        clean_sku_asin_map = sku_asin_map.drop_duplicates(subset='SKU', keep='first')
        
        # Merge orders with ASIN mapping
        orders_with_asin = clean_sku_asin_map.merge(
            orders_df, how='left', left_on='SKU', right_on='sku'
        )
        
        # Aggregate by ASIN
        agg_df = orders_with_asin.groupby('(Child) ASIN')[to_clean_cols].sum().reset_index()
        
        # Calculate subscription and premium service fees
        subscription_fee_df = self.dataframes['transaction'][
            self.dataframes['transaction']['description'] == 'Subscription'
        ].copy()
        premium_service_fee_df = self.dataframes['transaction'][
            self.dataframes['transaction']['description'] == 'Premium Services Fee'
        ].copy()
        
        subscription_fee_df['other transaction fees'] = pd.to_numeric(
            subscription_fee_df['other transaction fees'].replace('[\$,]', '', regex=True),
            errors='coerce'
        ).fillna(0)
        
        premium_service_fee_df['selling fees'] = pd.to_numeric(
            premium_service_fee_df['selling fees'].replace('[\$,]', '', regex=True),
            errors='coerce'
        ).fillna(0)
        
        total_subscription_fee = subscription_fee_df['other'].sum()
        total_premium_service_fee = premium_service_fee_df['selling fees'].sum()
        total_subscription_premium_fee = total_subscription_fee + total_premium_service_fee
        
        print(f"✅ Total Subscription Fee: ${total_subscription_fee:.2f}")
        print(f"✅ Total Premium Service Fee: ${total_premium_service_fee:.2f}")
        print(f"✅ Grand Total: ${total_subscription_premium_fee:.2f}")
        
        # Calculate allocatable fees
        total_product_sales = agg_df['product sales'].sum()
        agg_df['Allocatable Fees'] = (
            agg_df['product sales'] / total_product_sales * total_subscription_premium_fee
        ).round(2)
        
        # Calculate Gross Profit
        agg_df['Gross Profit'] = (
            agg_df['product sales'] +
            agg_df['selling fees'] +
            agg_df['fba fees'] +
            agg_df['other transaction fees']
        )
        
        return agg_df
    
    def add_unit_costs(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """Add unit costs from unit financial data."""
        print("💰 Adding unit costs...")
        
        # Clean unit financial data
        unit_financial_df = self.dataframes['unit_financial'].copy()
        unit_financial_df['Current Wholesale Price'] = pd.to_numeric(
            unit_financial_df['Current Wholesale Price'].replace('[\$,]', '', regex=True),
            errors='coerce'
        )
        
        unit_cost_map = unit_financial_df[['ASIN', 'Current Wholesale Price']].drop_duplicates(
            subset='ASIN', keep='first'
        )
        
        # Merge unit costs
        agg_df = agg_df.merge(
            unit_cost_map, how='left', left_on='(Child) ASIN', right_on='ASIN'
        )
        
        # Calculate product cost
        agg_df['Product Cost'] = -agg_df['Current Wholesale Price'] * agg_df['quantity']
        
        # Ensure all ASINs are present
        if 'business_report' in self.dataframes:
            all_asins = self.dataframes['business_report'][['(Child) ASIN']].drop_duplicates()
        else:
            all_asins = self.dataframes['business'][['(Child) ASIN']].drop_duplicates()
        
        agg_df = all_asins.merge(agg_df, on='(Child) ASIN', how='left').fillna(0)
        agg_df.drop(columns=['ASIN'], inplace=True)
        
        return agg_df
    
    def add_business_metrics(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """Add business metrics like Buy Box percentage."""
        print("📊 Adding business metrics...")
        
        # Get business metrics
        if 'business_report' in self.dataframes:
            business_metrics = self.dataframes['business_report'][
                ['(Child) ASIN', 'Featured Offer (Buy Box) Percentage', 'Sessions - Total', 'Unit Session Percentage']
            ]
        else:
            business_metrics = self.dataframes['business'][
                ['(Child) ASIN', 'Featured Offer (Buy Box) Percentage', 'Sessions - Total', 'Unit Session Percentage']
            ]
        
        business_metrics = business_metrics.drop_duplicates(subset='(Child) ASIN', keep='first')
        
        # Merge metrics
        agg_df = agg_df.merge(business_metrics, on='(Child) ASIN', how='left')
        
        return agg_df
    
    def process_storage_costs(self) -> pd.DataFrame:
        """Process storage costs and create allocation."""
        print("📦 Processing storage costs...")
        
        # Process long-term storage
        lt_storage_df = self.dataframes['lt_storage'].copy()
        lt_storage_df['amount-charged'] = pd.to_numeric(lt_storage_df['amount-charged'], errors='coerce')
        lt_storage_sum = lt_storage_df.groupby('asin')['amount-charged'].sum().reset_index()
        lt_storage_sum.rename(columns={'amount-charged': 'long-term storage amount'}, inplace=True)
        
        # Process short-term storage
        storage_df = self.dataframes['storage'].copy()
        storage_df['estimated_monthly_storage_fee'] = pd.to_numeric(
            storage_df['estimated_monthly_storage_fee'], errors='coerce'
        ).round(8)
        
        storage_df['asin'] = storage_df['asin'].astype(str).str.strip().str.upper()
        storage_df = storage_df[storage_df['estimated_monthly_storage_fee'].abs() >= 0.001]
        
        storage_df['average_quantity_on_hand'] = pd.to_numeric(
            storage_df['average_quantity_on_hand'], errors='coerce'
        ).fillna(0)
        
        avg_qty_df = storage_df.groupby('asin', as_index=False)['average_quantity_on_hand'].sum()
        
        storage_summary = storage_df.groupby('asin', as_index=False)['estimated_monthly_storage_fee'].sum()
        storage_summary = storage_summary.merge(avg_qty_df, on='asin', how='left')
        
        total_estimated_fee = round(storage_summary['estimated_monthly_storage_fee'].sum(), 5)
        storage_summary['allocation %'] = storage_summary['estimated_monthly_storage_fee'] / total_estimated_fee
        
        # Find actual FBA storage fee
        transaction_df = self.dataframes['transaction'].copy()
        transaction_df['other'] = pd.to_numeric(
            transaction_df['other'].replace('[\$,]', '', regex=True), errors='coerce'
        )
        
        actual_amount = transaction_df[
            transaction_df['description'].str.contains('FBA storage fee', case=False, na=False)
        ]['other'].sum()
        
        storage_summary['fee $'] = (storage_summary['allocation %'] * actual_amount).round(2)
        storage_summary['allocation %'] = (storage_summary['allocation %'] * 100).round(2).astype(str) + '%'
        
        # Merge short-term and long-term storage
        short_term_df = storage_summary[['asin', 'fee $']].copy()
        short_term_df.rename(columns={'fee $': 'short-term storage amount'}, inplace=True)
        
        total_storage_df = pd.merge(
            short_term_df, lt_storage_sum, how='outer', on='asin'
        ).fillna(0)
        
        total_storage_df['total storage cost'] = (
            -total_storage_df['short-term storage amount'] + total_storage_df['long-term storage amount']
        )
        total_storage_df['long-term storage amount'] = total_storage_df['long-term storage amount'].apply(
            lambda x: 0 if x == 0 else -abs(x)
        )
        total_storage_df['total storage cost'] = total_storage_df['total storage cost'].apply(
            lambda x: 0 if x == 0 else -abs(x)
        )
        
        total_storage_df['short-term storage amount'] = total_storage_df['short-term storage amount'].round(2)
        total_storage_df['total storage cost'] = total_storage_df['total storage cost'].round(2)
        
        return total_storage_df
    
    def process_shipping_and_label_fees(self) -> pd.DataFrame:
        """Process shipping and label fees from selling econ data."""
        print("🚚 Processing shipping and label fees...")
        
        # Define required columns
        required_columns = [
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
        
        # Process selling econ data
        selling_econ_df = self.dataframes['selling_econ'].copy()
        
        # Ensure required columns exist
        for col in required_columns:
            if col not in selling_econ_df.columns:
                selling_econ_df[col] = 0
        
        # Convert date columns
        selling_econ_df["Start date"] = pd.to_datetime(selling_econ_df["Start date"], errors='coerce')
        selling_econ_df["End date"] = pd.to_datetime(selling_econ_df["End date"], errors='coerce')
        
        # Sort by End date descending
        selling_econ_df.sort_values(by="End date", ascending=False, inplace=True)
        selling_econ_df.reset_index(drop=True, inplace=True)
        
        # Get most recent values per ASIN
        recent_values = {}
        threshold = 0.0001
        
        sorted_all = selling_econ_df.sort_values(by=["ASIN", "End date"], ascending=[True, False]).copy()
        
        for col in required_columns:
            asin_values = {}
            
            for asin, asin_df in sorted_all.groupby('ASIN'):
                asin_df = asin_df.sort_values(by="End date", ascending=False)
                value_series = asin_df[col]
                non_zero_values = value_series[value_series > threshold]
                
                if not non_zero_values.empty:
                    asin_values[asin] = non_zero_values.iloc[0]
                else:
                    asin_values[asin] = 0
            
            recent_values[col] = pd.Series(asin_values)
        
        # Combine into DataFrame
        recent_values_df = pd.DataFrame(recent_values)
        recent_values_df.index.name = 'ASIN'
        recent_values_df = recent_values_df.reset_index()
        recent_values_df = recent_values_df.fillna(0)
        
        # Calculate shipping and label fees
        recent_values_df["Shipping Fee"] = (
            recent_values_df["FBA inbound placement service fee per unit"] +
            recent_values_df["Inbound Transportation Fee per unit"]
        )
        
        recent_values_df["Label Fee"] = (
            recent_values_df["Label per unit"] +
            recent_values_df["Bagging per unit"]
        )
        
        return recent_values_df
    
    def process_returns(self) -> pd.DataFrame:
        """Process return data and calculate costs."""
        print("🔄 Processing returns...")
        
        # Process refunds
        refund_df = self.dataframes['transaction'][
            self.dataframes['transaction']['type'] == 'Refund'
        ].copy()
        
        refund_df['product sales'] = pd.to_numeric(
            refund_df['product sales'].replace('[\$,]', '', regex=True), errors='coerce'
        )
        refund_df['quantity'] = pd.to_numeric(refund_df['quantity'], errors='coerce')
        
        # Get SKU to ASIN mapping
        if 'business_report' in self.dataframes:
            business_unique = self.dataframes['business_report'][['SKU', '(Child) ASIN']].drop_duplicates(subset='SKU')
        else:
            business_unique = self.dataframes['business'][['SKU', '(Child) ASIN']].drop_duplicates(subset='SKU')
        
        refund_with_asin = refund_df.merge(
            business_unique, how='left', left_on='sku', right_on='SKU'
        )
        
        # Group by ASIN
        refund_grouped = (
            refund_with_asin.groupby('(Child) ASIN')[['quantity', 'product sales']]
            .sum()
            .rename(columns={
                'quantity': 'Total Returned Units',
                'product sales': 'Total Returned Sales'
            })
            .reset_index()
        )
        
        # Ensure all ASINs are included
        if 'business_report' in self.dataframes:
            all_asins = self.dataframes['business_report'][['(Child) ASIN']].drop_duplicates()
        else:
            all_asins = self.dataframes['business'][['(Child) ASIN']].drop_duplicates()
        
        refund_agg = all_asins.merge(refund_grouped, on='(Child) ASIN', how='left')
        refund_agg[['Total Returned Units', 'Total Returned Sales']] = refund_agg[
            ['Total Returned Units', 'Total Returned Sales']
        ].fillna(0)
        
        return refund_agg
    
    def create_final_aggregated_report(self) -> pd.DataFrame:
        """Create the main aggregated report."""
        print("📋 Creating final aggregated report...")
        
        # Process all components
        agg_df = self.process_orders_and_aggregate()
        agg_df = self.add_unit_costs(agg_df)
        agg_df = self.add_business_metrics(agg_df)
        
        # Add storage costs
        total_storage_df = self.process_storage_costs()
        agg_subset = agg_df.rename(columns={"(Child) ASIN": "ASIN"})
        total_storage_df = total_storage_df.rename(columns={"asin": "ASIN"})
        
        merged_df = agg_subset.merge(
            total_storage_df[['ASIN', 'total storage cost']], on='ASIN', how='left'
        )
        
        # Add shipping and label fees
        recent_values_df = self.process_shipping_and_label_fees()
        final_df = merged_df.merge(
            recent_values_df[['ASIN', 'Shipping Fee', 'Label Fee']],
            on='ASIN', how='left'
        )
        
        # Calculate fees
        final_df['Shipping Fee'] = -final_df['Shipping Fee'] * final_df['quantity']
        final_df['Label Fee'] = -final_df['Label Fee'] * final_df['quantity']
        final_df['Shipping Fee'] = final_df['Shipping Fee'].apply(lambda x: 0 if x == 0 else (-abs(x)))
        final_df['Label Fee'] = final_df['Label Fee'].apply(lambda x: 0 if x == 0 else (-abs(x)))
        
        # Add return costs
        refund_agg = self.process_returns()
        final_merged = final_df.merge(
            refund_agg[['(Child) ASIN', 'Total Returned Units', 'Total Returned Sales']],
            left_on='ASIN', right_on='(Child) ASIN', how='left'
        )
        
        # Calculate return costs
        unit_financial_df = self.dataframes['unit_financial'].copy()
        unit_financial_df['Current Wholesale Price'] = pd.to_numeric(
            unit_financial_df['Current Wholesale Price'].replace('[\$,]', '', regex=True),
            errors='coerce'
        )
        
        product_cost_df = unit_financial_df[['ASIN', 'Current Wholesale Price']].drop_duplicates(subset='ASIN')
        
        final_merged = final_merged.merge(
            product_cost_df, left_on='ASIN', right_on='ASIN', how='left'
        )
        
        final_merged['Product Cost of Return'] = (
            final_merged['Current Wholesale Price'] * final_merged['Total Returned Units']
        )
        final_merged['Return Shipping Fee'] = (
            final_merged['Shipping Fee'] / final_merged['quantity'] * final_merged['Total Returned Units']
        )
        final_merged['Return Label Fee'] = (
            final_merged['Label Fee'] / final_merged['quantity'] * final_merged['Total Returned Units']
        )
        
        # Calculate net cost of returns
        final_merged['Net Cost of Return'] = (
            final_merged['Product Cost of Return'] +
            final_merged['Return Shipping Fee'] +
            final_merged['Return Label Fee']
        )
        
        # Recalculate gross profit with all costs
        final_merged['Gross Profit'] = (
            final_merged['product sales'] +
            final_merged['Product Cost'] +
            final_merged['selling fees'] +
            final_merged['fba fees'] +
            final_merged['other transaction fees'] +
            final_merged['Shipping Fee'] +
            final_merged['Label Fee'] +
            final_merged['total storage cost'] +
            final_merged['Allocatable Fees'] +
            final_merged['Net Cost of Return']
        )
        
        # Calculate gross margin
        final_merged['Gross Margin'] = np.where(
            final_merged['product sales'] != 0,
            final_merged['Gross Profit'] * 100 / final_merged['product sales'],
            0
        )
        
        return final_merged
    
    def create_brand_report(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """Create brand-level aggregated report."""
        print("🏷️ Creating brand-level report...")
        
        # Add brand information if available
        if 'brands_asin_list' in self.dataframes:
            brand_df = self.dataframes['brands_asin_list'].copy()
            brand_df.columns = brand_df.columns.str.strip()
            
            # Ensure ASIN column exists
            asin_col = [col for col in brand_df.columns if col.strip().lower() == 'asin']
            if asin_col and asin_col[0] != 'ASIN':
                brand_df.rename(columns={asin_col[0]: 'ASIN'}, inplace=True)
            
            final_df = final_df.merge(brand_df, on='ASIN', how='left')
        
        # Aggregate by brand
        money_cols = [
            'product sales', 'Product Cost', 'selling fees', 'fba fees',
            'other transaction fees', 'Shipping Fee', 'Label Fee',
            'total storage cost', 'Allocatable Fees', 'Net Cost of Return'
        ]
        
        df_numeric = final_df.copy()
        for col in money_cols:
            if col in df_numeric.columns:
                df_numeric[col] = pd.to_numeric(
                    df_numeric[col].astype(str).str.replace('[\$,]', '', regex=True),
                    errors='coerce'
                ).fillna(0)
        
        df_numeric['quantity'] = pd.to_numeric(df_numeric['quantity'], errors='coerce')
        
        # Group by brand
        agg_dict = {col: 'sum' for col in money_cols + ['quantity']}
        brand_report = df_numeric.groupby('Brands', as_index=False).agg(agg_dict)
        
        # Recalculate gross profit and margin
        brand_report['Gross Profit'] = brand_report[money_cols].sum(axis=1)
        brand_report['Gross Margin'] = np.where(
            brand_report['product sales'] != 0,
            brand_report['Gross Profit'] * 100 / brand_report['product sales'],
            0
        )
        
        # Format columns
        for col in money_cols + ['Gross Profit']:
            brand_report[col] = brand_report[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else ""
            )
        
        brand_report['quantity'] = brand_report['quantity'].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )
        
        brand_report['Gross Margin'] = brand_report['Gross Margin'].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else ""
        )
        
        return brand_report
    
    def create_gm_report(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """Create GM-level aggregated report."""
        print("📊 Creating GM-level report...")
        
        # Add GM information if available
        if 'brands_asin_list' in self.dataframes:
            gm_df = self.dataframes['brands_asin_list'].copy()
            gm_df.columns = gm_df.columns.str.strip()
            
            # Ensure ASIN column exists
            asin_col = [col for col in gm_df.columns if col.strip().lower() == 'asin']
            if asin_col and asin_col[0] != 'ASIN':
                gm_df.rename(columns={asin_col[0]: 'ASIN'}, inplace=True)
            
            final_df = final_df.merge(gm_df, on='ASIN', how='left')
        
        # Aggregate by GM
        money_cols = [
            'product sales', 'Product Cost', 'selling fees', 'fba fees',
            'other transaction fees', 'Shipping Fee', 'Label Fee',
            'total storage cost', 'Allocatable Fees', 'Net Cost of Return'
        ]
        
        df_numeric = final_df.copy()
        for col in money_cols:
            if col in df_numeric.columns:
                df_numeric[col] = pd.to_numeric(
                    df_numeric[col].astype(str).str.replace('[\$,]', '', regex=True),
                    errors='coerce'
                ).fillna(0)
        
        df_numeric['quantity'] = pd.to_numeric(df_numeric['quantity'], errors='coerce')
        
        # Group by GM
        agg_dict = {col: 'sum' for col in money_cols + ['quantity']}
        gm_report = df_numeric.groupby('GM', as_index=False).agg(agg_dict)
        
        # Recalculate gross profit and margin
        gm_report['Gross Profit'] = gm_report[money_cols].sum(axis=1)
        gm_report['Gross Margin'] = np.where(
            gm_report['product sales'] != 0,
            gm_report['Gross Profit'] * 100 / gm_report['product sales'],
            0
        )
        
        # Format columns
        for col in money_cols + ['Gross Profit']:
            gm_report[col] = gm_report[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else ""
            )
        
        gm_report['quantity'] = gm_report['quantity'].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )
        
        gm_report['Gross Margin'] = gm_report['Gross Margin'].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else ""
        )
        
        return gm_report
    
    def create_underperforming_report(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """Create underperforming ATS report."""
        print("⚠️ Creating underperforming ATS report...")
        
        # Compare actual vs expected ATS
        unit_financial_df = self.dataframes['unit_financial'].copy()
        
        # Clean and prepare data
        final_df['quantity'] = pd.to_numeric(
            final_df['quantity'].replace(',', '', regex=True), errors='coerce'
        )
        
        # Get expected ATS from unit financial
        unit_financial_df['Amazon Top-line Sales (ATS)'] = pd.to_numeric(
            unit_financial_df['Amazon Top-line Sales (ATS)'].replace('[\$,]', '', regex=True),
            errors='coerce'
        )
        
        # Merge expected ATS
        compare_df = final_df[['ASIN', 'quantity', 'product sales']].copy()
        compare_df = compare_df.merge(
            unit_financial_df[['ASIN', 'Amazon Top-line Sales (ATS)']],
            on='ASIN', how='left'
        )
        
        # Calculate per-unit values
        compare_df['actual_per_unit'] = compare_df['product sales'] / compare_df['quantity']
        compare_df['expected_per_unit'] = compare_df['Amazon Top-line Sales (ATS)']
        compare_df['diff'] = compare_df['actual_per_unit'] - compare_df['expected_per_unit']
        
        # Flag underperforming ASINs
        compare_df['underperforming'] = compare_df['diff'] < 0
        
        # Get underperforming ASINs
        underperforming_asins = compare_df[compare_df['underperforming']]['ASIN'].unique()
        
        # Get transaction details for underperforming ASINs
        if len(underperforming_asins) > 0:
            # Get SKU to ASIN mapping
            if 'business_report' in self.dataframes:
                sku_asin_map = self.dataframes['business_report'][['SKU', '(Child) ASIN']].drop_duplicates(subset='SKU')
            else:
                sku_asin_map = self.dataframes['business'][['SKU', '(Child) ASIN']].drop_duplicates(subset='SKU')
            
            # Get order transactions
            orders_df = self.dataframes['transaction'][
                self.dataframes['transaction']['type'] == 'Order'
            ].copy()
            
            orders_df['product sales'] = pd.to_numeric(
                orders_df['product sales'].replace('[\$,]', '', regex=True), errors='coerce'
            )
            orders_df['quantity'] = pd.to_numeric(orders_df['quantity'], errors='coerce')
            
            # Merge with ASIN mapping
            orders_with_asin = sku_asin_map.merge(
                orders_df, how='right', left_on='SKU', right_on='sku'
            )
            orders_with_asin.rename(columns={'(Child) ASIN': 'ASIN'}, inplace=True)
            
            # Filter for underperforming ASINs
            underperforming_orders = orders_with_asin[
                orders_with_asin['ASIN'].isin(underperforming_asins)
            ].copy()
            
            # Calculate expected vs actual
            ats_expected_map = unit_financial_df.set_index('ASIN')['Amazon Top-line Sales (ATS)'].to_dict()
            underperforming_orders['expected_total_ats'] = (
                underperforming_orders['ASIN'].map(ats_expected_map) * 
                underperforming_orders['quantity']
            )
            
            # Filter transactions where actual < expected
            underperforming_transactions = underperforming_orders[
                underperforming_orders['product sales'] < underperforming_orders['expected_total_ats']
            ]
            
            return underperforming_transactions
        
        return pd.DataFrame()
    
    def generate_all_reports(self):
        """Generate all reports."""
        print("🚀 Starting report generation...")
        
        # Create main aggregated report
        final_df = self.create_final_aggregated_report()
        
        # Save main report
        output_path = 'full_detailed_aggregated_report.xlsx'
        final_df.to_excel(output_path, index=False)
        print(f"✅ Saved: {output_path}")
        
        # Create brand report
        brand_report = self.create_brand_report(final_df)
        output_path = 'full_detailed_aggregated_report_brand.xlsx'
        brand_report.to_excel(output_path, index=False)
        print(f"✅ Saved: {output_path}")
        
        # Create GM report
        gm_report = self.create_gm_report(final_df)
        output_path = 'full_detailed_aggregated_report_gm.xlsx'
        gm_report.to_excel(output_path, index=False)
        print(f"✅ Saved: {output_path}")
        
        # Create underperforming report
        underperforming_report = self.create_underperforming_report(final_df)
        if not underperforming_report.empty:
            output_path = 'underperforming_ats_report.xlsx'
            underperforming_report.to_excel(output_path, index=False)
            print(f"✅ Saved: {output_path}")
        else:
            print("ℹ️ No underperforming transactions found")
        
        # Save additional supporting reports
        self.save_supporting_reports()
        
        print("🎉 All reports generated successfully!")
    
    def save_supporting_reports(self):
        """Save additional supporting reports."""
        print("📄 Saving supporting reports...")
        
        # Save storage summary
        storage_summary = self.process_storage_costs()
        storage_summary.to_excel('monthly_fba_storage_allocation.xlsx', index=False)
        print("✅ Saved: monthly_fba_storage_allocation.xlsx")
        
        # Save long-term storage summary
        lt_storage_df = self.dataframes['lt_storage'].copy()
        lt_storage_df['amount-charged'] = pd.to_numeric(lt_storage_df['amount-charged'], errors='coerce')
        lt_storage_sum = lt_storage_df.groupby('asin')['amount-charged'].sum().reset_index()
        lt_storage_sum.rename(columns={'amount-charged': 'long-term storage amount'}, inplace=True)
        lt_storage_sum.to_excel('long_term_storage_summary.xlsx', index=False)
        print("✅ Saved: long_term_storage_summary.xlsx")
        
        # Save return summary
        refund_agg = self.process_returns()
        refund_agg.to_excel('refund_summary_all_asins.xlsx', index=False)
        print("✅ Saved: refund_summary_all_asins.xlsx")

def main():
    """Main function to run the data processor."""
    print("=" * 60)
    print("Amazon Business Data Processor")
    print("=" * 60)
    
    # Get input file from command line or use default
    input_file = sys.argv[1] if len(sys.argv) > 1 else "202502 Brands Reports v5.xlsx"
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        print("Usage: python data_processor.py [input_file.xlsx]")
        sys.exit(1)
    
    # Create processor and generate reports
    processor = AmazonDataProcessor(input_file)
    processor.generate_all_reports()
    
    print("\n" + "=" * 60)
    print("Report generation completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
