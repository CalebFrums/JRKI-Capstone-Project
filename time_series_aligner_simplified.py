#!/usr/bin/env python3
"""
Time Series Data Integration and Alignment System
NZ Unemployment Forecasting System - Data Integration Module

This module provides streamlined time series data integration capabilities,
designed for robust handling of multiple economic datasets with varying
completeness and temporal coverage.

Features:
- Multi-dataset temporal alignment and integration
- Data quality validation and filtering
- Feature engineering for machine learning models
- Comprehensive integration reporting
- Scalable architecture for additional data sources

Author: Data Science Team
Version: Production v2.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

class SimpleTimeSeriesAligner:
    """
    Professional time series data integration system for government economic datasets.
    
    This class provides robust data integration capabilities designed for handling
    multiple time series datasets with varying temporal coverage and data quality.
    Optimized for New Zealand economic forecasting requirements.
    """
    
    def __init__(self, data_dir="data_cleaned", output_dir="data_cleaned"):
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True)
        print(f"Time Series Aligner: {self.data_dir} -> {self.output_dir}")
    
    def _parse_quarterly_periods(self, period_column):
        """Parse quarterly period strings like '1989Q1' to datetime objects"""
        parsed_dates = []
        for period in period_column:
            if pd.isna(period):
                parsed_dates.append(pd.NaT)
                continue
                
            period_str = str(period).strip()
            if 'Q' not in period_str:
                parsed_dates.append(pd.NaT)
                continue
            
            try:
                # Parse formats like '1989Q1', '1989Q2', etc.
                year_str, quarter_str = period_str.split('Q')
                year = int(year_str)
                quarter = int(quarter_str)
                
                # Convert to first month of quarter
                month = (quarter - 1) * 3 + 1  # Q1->Jan(1), Q2->Apr(4), Q3->Jul(7), Q4->Oct(10)
                date = pd.Timestamp(year=year, month=month, day=1)
                parsed_dates.append(date)
                
            except (ValueError, IndexError):
                parsed_dates.append(pd.NaT)
                
        return pd.Series(parsed_dates)
    
    def validate_cleaned_file(self, filepath):
        """Validate that a cleaned CSV file has expected structure"""
        df_sample = pd.read_csv(filepath, nrows=10)
        filename = filepath.name
        
        errors = []
        warnings_list = []
        
        # Check 1: Must have date column
        if 'date' not in df_sample.columns:
            errors.append(f"CRITICAL: {filename} missing 'date' column - not a valid cleaned dataset")
        
        # Check 2: File-specific validation
        if 'unemployment' in filename.lower():
            unemployment_cols = [col for col in df_sample.columns if 'unemployment' in col.lower()]
            if len(unemployment_cols) == 0:
                errors.append(f"CRITICAL: {filename} claims to be unemployment data but has no unemployment columns")
            else:
                # Check unemployment rate ranges
                for col in unemployment_cols:
                    if df_sample[col].notna().sum() > 0:
                        max_val = df_sample[col].max()
                        if max_val > 1000:
                            errors.append(f"CRITICAL: {filename} column {col} has values up to {max_val} - looks like population data, not rates")
        
        elif 'cpi' in filename.lower():
            # Look for CPI patterns: 'cpi' in column name OR 'all_groups' pattern (Stats NZ CPI format)
            cpi_cols = [col for col in df_sample.columns if 
                       'cpi' in col.lower() or 'all_groups' in col.lower() or 'all groups' in col.lower()]
            if len(cpi_cols) == 0:
                errors.append(f"CRITICAL: {filename} claims to be CPI data but has no CPI columns")
        
        elif 'gdp' in filename.lower():
            # Look for GDP patterns: 'gdp' in column name OR regional industry patterns (Stats NZ GDP format)
            gdp_cols = [col for col in df_sample.columns if 
                       'gdp' in col.lower() or 'all_industries' in col.lower() or 
                       ('total' in col.lower() and any(region in col.lower() for region in 
                        ['northland', 'auckland', 'wellington', 'canterbury', 'otago']))]
            if len(gdp_cols) == 0:
                errors.append(f"CRITICAL: {filename} claims to be GDP data but has no GDP columns")
        
        # Check 3: Data completeness
        total_cells = df_sample.shape[0] * df_sample.shape[1]
        missing_cells = df_sample.isnull().sum().sum()
        if total_cells > 0:
            completeness = (total_cells - missing_cells) / total_cells
            if completeness < 0.1:
                warnings_list.append(f"WARNING: {filename} is {completeness:.1%} complete - very sparse data")
        
        return errors, warnings_list
    
    def load_datasets(self):
        """Load and validate all cleaned CSV files"""
        datasets = {}
        csv_files = list(self.data_dir.glob("cleaned_*.csv"))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No cleaned_*.csv files found in {self.data_dir}")
        
        print(f"\nLoading {len(csv_files)} cleaned datasets:")
        
        validation_errors = []
        
        for csv_file in csv_files:
            try:
                # Validate file structure first
                errors, warnings_list = self.validate_cleaned_file(csv_file)
                
                if errors:
                    print(f"ERROR {csv_file.name} VALIDATION FAILED:")
                    for error in errors:
                        print(f"   - {error}")
                    validation_errors.extend(errors)
                    continue
                
                if warnings_list:
                    for warning in warnings_list:
                        print(f"WARNING {warning}")
                
                # Load if validation passed
                df = pd.read_csv(csv_file)
                
                # Fix quarterly date parsing - check if first column contains quarterly periods
                first_col = df.columns[0]
                if 'Qrtly' in first_col or 'quarterly' in csv_file.name.lower():
                    # Check if first column has quarterly period patterns
                    sample_values = df[first_col].dropna().astype(str).head(10).tolist()
                    if any('Q' in str(val) for val in sample_values):
                        print(f"   QUARTER {csv_file.name}: Found quarterly periods in {first_col}")
                        # Convert quarterly periods to proper dates and set as date column
                        df['date'] = self._parse_quarterly_periods(df[first_col])
                        # Only drop original column if it's not already named 'date'
                        if first_col != 'date':
                            df = df.drop(columns=[first_col])
                elif 'date' in df.columns:
                    # Handle monthly formats like 2000M02 before pandas datetime parsing
                    if df['date'].dtype == 'object':
                        sample_dates = df['date'].dropna().head(5)
                        if any('M' in str(d) and len(str(d)) == 7 for d in sample_dates):
                            # Convert 2000M02 -> 2000-02-01
                            df['date'] = df['date'].str.replace('M', '-') + '-01'
                    
                    # Handle different date formats explicitly to avoid warnings
                    if df['date'].dtype == 'object':
                        # Try to infer format to avoid parsing warnings
                        sample_dates = df['date'].dropna().head(10).astype(str)
                        if any('M' in d and len(d) == 7 for d in sample_dates):
                            # Format like 2000M02
                            df['date'] = pd.to_datetime(df['date'].str.replace('M', '-') + '-01', format='%Y-%m-%d', errors='coerce')
                        elif any('Q' in d for d in sample_dates):
                            # Quarterly format - already handled above, skip extra parsing
                            pass
                        elif any('-' in d for d in sample_dates):
                            # Standard date format with explicit format specification
                            try:
                                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
                            except:
                                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        else:
                            # Fallback for other formats - pandas now infers automatically
                            df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    else:
                        # Already datetime, no conversion needed
                        pass
                    # Remove rows with invalid dates
                    df = df.dropna(subset=['date'])
                
                datasets[csv_file.name] = df
                print(f"OK {csv_file.name}: {len(df)} records")
                
            except Exception as e:
                error_msg = f"Failed to load {csv_file.name}: {e}"
                print(f"ERROR {error_msg}")
                validation_errors.append(error_msg)
        
        # Check if we have any valid datasets
        if len(datasets) == 0:
            raise ValueError("No valid cleaned datasets found - all files failed validation")
        
        # Check for critical validation errors
        if validation_errors:
            print(f"\nVALIDATION ERRORS DETECTED:")
            for error in validation_errors:
                print(f"  - {error}")
            print(f"\nThese look like wrong datasets - expecting cleaned unemployment/economic data from Stats NZ")
            
            # Only proceed if we have some valid datasets
            if len(datasets) < len(csv_files) / 2:
                raise ValueError("Too many validation failures - most files appear to be wrong datasets")
        
        return datasets
    
    def _detect_regions_from_columns(self, df):
        """Dynamically detect regions from actual column names"""
        regions = set()
        for col in df.columns:
            col_lower = col.lower()
            # Look for regional indicators in column names
            if 'unemployment' in col_lower or 'gdp' in col_lower or 'cpi' in col_lower:
                # Extract potential region names (simple approach)
                words = col_lower.replace('_', ' ').split()
                for word in words:
                    if len(word) > 4 and word not in ['unemployment', 'rate', 'gdp', 'millions', 'cpi', 'total', 'ages', 'years']:
                        regions.add(word)
        return list(regions)
    
    def _is_important_column(self, col_name):
        """Simple check for important columns to protect"""
        col_lower = col_name.lower()
        return (
            col_lower == 'date' or
            'total' in col_lower or 
            'all_ages' in col_lower or
            'unemployment_rate' in col_lower or  # Handles both cases
            'aged_' in col_lower or  # Protect age demographic columns
            '55_plus' in col_lower or  # Protect senior age columns
            'gdp_millions' in col_lower or
            'all_industries' in col_lower or  # Protect GDP columns
            'cpi_value' in col_lower or
            'all_groups' in col_lower or  # Protect CPI columns
            ('cpi_' in col_lower and 'unknown' not in col_lower) or
            # Protect ECT (Electronic Card Transaction) columns
            any(ect_pattern in col_lower for ect_pattern in [
                'actual_', 'seasonally_adjusted_', 'trend_', 
                'consumables', 'durables', 'hospitality', 'services', 
                'apparel', 'motor_vehicles', 'fuel', 'retail'
            ])
        )
    
    def clean_dataframe(self, df, dataset_name=""):
        """Simple data quality filter with Stats NZ data handling"""
        if len(df) == 0:
            return df
        
        original_cols = len(df.columns)
        
        # Handle Stats NZ missing data markers first
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            if col != 'date':
                # Replace Stats NZ missing data markers with NaN
                df_cleaned[col] = df_cleaned[col].replace(['..', 'C', 'S', 'E', 'P', 'R'], np.nan)
                # Try to convert to numeric
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        
        # Simple approach - drop obviously broken columns, keep important ones
        cols_to_drop = []
        for col in df_cleaned.columns:
            if not self._is_important_column(col):
                # Drop unknown columns
                if 'unknown' in col.lower():
                    cols_to_drop.append(col)
                # Drop completely empty columns
                elif df_cleaned[col].notna().sum() == 0:
                    cols_to_drop.append(col)
                # Drop columns with <30% data
                elif df_cleaned[col].notna().mean() < 0.3:
                    cols_to_drop.append(col)
        
        df_cleaned = df_cleaned.drop(columns=cols_to_drop)
        
        cleaned_cols = len(df_cleaned.columns)
        if cleaned_cols < original_cols:
            print(f"   CLEAN {dataset_name}: {original_cols} -> {cleaned_cols} columns")
        
        return df_cleaned
    
    def get_date_range(self, datasets):
        """Simple date range calculation"""
        all_dates = []
        for df in datasets.values():
            if 'date' in df.columns:
                # Check if already datetime type to avoid parsing warnings
                if df['date'].dtype.name.startswith('datetime'):
                    valid_dates = df['date'].dropna()
                else:
                    # Only parse if not already datetime - try common formats first
                    try:
                        # Use format='mixed' for mixed/inconsistent date formats to avoid warnings
                        valid_dates = pd.to_datetime(df['date'], format='mixed', errors='coerce').dropna()
                    except:
                        # Fallback for older pandas versions that don't support format='mixed'
                        valid_dates = pd.to_datetime(df['date'], errors='coerce').dropna()
                all_dates.extend(valid_dates)
        
        if not all_dates:
            return pd.Timestamp('2010-01-01'), pd.Timestamp('2023-12-31')
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        print(f"Date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
        
        return start_date, end_date
    
    def create_master_timeline(self, start_date, end_date, datasets):
        """Create quarterly timeline respecting natural data frequencies"""
        # Use quarterly as the master frequency to avoid artificial repetition
        # Quarterly is appropriate as unemployment is naturally reported quarterly
        timeline = pd.date_range(start=start_date, end=end_date, freq='QS')
        print(f"Master timeline: {len(timeline)} quarterly periods (respecting natural frequencies)")
        
        return timeline
    
    def add_dataset_prefixes(self, df, filename):
        """Add dataset prefixes to column names for better organization"""
        # Create dataset prefix mapping
        prefix_map = {
            'GDP': ['GDP All Industries'],
            'CPI': ['CPI All Groups', 'CPI Regional'],
            'HLF': ['HLF Labour', 'Labour force', 'Labour Force'],
            'ECT': ['ECT electronic', 'ECT means', 'ECT Number', 'ECT Total', 'ECT Totals'],
            'QEM': ['QEM average', 'QEM Average', 'QEM Filled', 'QEM filled'],
            'MEI': ['MEI Age', 'MEI high', 'MEI Industry', 'MEI Sex'],
            'LCI': ['LCI All'],
            'BUO': ['BUO ICT', 'BUO Totals', 'BUO innovation']
        }
        
        # Find matching prefix
        dataset_prefix = None
        for prefix, patterns in prefix_map.items():
            if any(pattern in filename for pattern in patterns):
                dataset_prefix = prefix
                break
        
        if dataset_prefix is None:
            # Fallback to generic naming if no match
            dataset_prefix = filename.replace('cleaned_', '').split(' ')[0].upper()[:3]
        
        # Apply prefix to all columns except 'date'
        new_df = df.copy()
        for col in new_df.columns:
            if col != 'date':
                new_df = new_df.rename(columns={col: f"{dataset_prefix}_{col}"})
        
        return new_df
    
    def align_data_to_timeline(self, df, master_timeline, filename):
        """Align data to quarterly timeline, properly handling different frequencies"""
        # Determine data frequency
        is_monthly_data = any(keyword in filename for keyword in ['MEI', 'ECT'])
        is_annual_data = 'BUO' in filename
        is_quarterly_data = any(keyword in filename for keyword in ['HLF', 'QEM']) or not (is_monthly_data or is_annual_data)
        
        # Handle annual data alignment - spread to quarters within each year
        if is_annual_data:
            print(f"   - Handling annual {filename} data (spreading to quarters)")
            aligned_df = pd.DataFrame({'date': master_timeline})
            aligned_df['date'] = pd.to_datetime(aligned_df['date'])
            
            # Get actual data date range to avoid spreading outside real data period
            if len(df) > 0:
                # Ensure date column contains valid datetime values before using .dt accessor
                valid_dates = df['date'].dropna()
                if len(valid_dates) > 0 and valid_dates.dtype.name.startswith('datetime'):
                    data_start_year = valid_dates.dt.year.min()
                    data_end_year = valid_dates.dt.year.max()
                    print(f"     Annual data available for years: {data_start_year}-{data_end_year}")
                else:
                    print(f"     WARNING: No valid datetime values in {filename}")
                    return aligned_df
            else:
                print(f"     WARNING: No valid data rows in {filename}")
                return aligned_df
            
            # Collect all columns to add at once to avoid DataFrame fragmentation
            columns_to_add = {}
            for col in df.columns:
                if col != 'date':
                    # Create annual mapping
                    annual_mapping = {}
                    for _, row in df.iterrows():
                        if pd.notna(row['date']) and pd.notna(row[col]):
                            year = row['date'].year
                            annual_mapping[year] = row[col]
                    
                    # Apply annual values ONLY to quarters within actual data range
                    quarterly_values = []
                    for date in master_timeline:
                        year = date.year
                        # Only spread data within the actual annual data range
                        if data_start_year <= year <= data_end_year and year in annual_mapping:
                            quarterly_values.append(annual_mapping[year])
                        else:
                            quarterly_values.append(np.nan)
                    
                    columns_to_add[col] = quarterly_values
            
            # Add all columns at once using pd.concat to avoid fragmentation
            if columns_to_add:
                new_cols_df = pd.DataFrame(columns_to_add, index=aligned_df.index)
                aligned_df = pd.concat([aligned_df, new_cols_df], axis=1)
        
        # Handle monthly data - aggregate to quarters (no artificial repetition)
        elif is_monthly_data:
            print(f"   - Aggregating monthly {filename} to quarterly timeline")
            df_copy = df.copy()
            
            # Ensure we have valid datetime values before using .dt accessor
            if len(df_copy) > 0 and df_copy['date'].dtype.name.startswith('datetime'):
                df_copy['quarter'] = df_copy['date'].dt.to_period('Q').dt.start_time
            else:
                print(f"     WARNING: Invalid datetime values in monthly data {filename}")
                return pd.DataFrame({'date': master_timeline})
            
            # Aggregate by taking mean of monthly values in each quarter
            aligned_df = pd.DataFrame({'date': master_timeline})
            aligned_df['date'] = pd.to_datetime(aligned_df['date'])
            for col in df_copy.columns:
                if col != 'date' and col != 'quarter':
                    quarterly_col = df_copy.groupby('quarter')[col].mean().reset_index()
                    quarterly_col.rename(columns={'quarter': 'date'}, inplace=True)
                    quarterly_col['date'] = pd.to_datetime(quarterly_col['date'])
                    aligned_df = aligned_df.merge(quarterly_col, on='date', how='left')
        
        # Handle quarterly data - direct alignment (no frequency conversion)
        elif is_quarterly_data:
            print(f"   - Direct quarterly alignment for {filename}")
            aligned_df = pd.DataFrame({'date': master_timeline})
            aligned_df['date'] = pd.to_datetime(aligned_df['date'])
            try:
                df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
            except:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            aligned_df = aligned_df.merge(df, on='date', how='left')
        
        # Default case - direct alignment
        else:
            print(f"   - Default alignment for {filename}")
            aligned_df = pd.DataFrame({'date': master_timeline})
            aligned_df['date'] = pd.to_datetime(aligned_df['date'])
            try:
                df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
            except:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            aligned_df = aligned_df.merge(df, on='date', how='left')
        
        return aligned_df
    
    def basic_missing_data_fill(self, df):
        """Realistic time series filling for government data with gaps"""
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'date':
                original_missing = df[col].isna().sum()
                
                # More aggressive filling for government data reality
                # Linear interpolation for time series continuity
                df[col] = df[col].interpolate(method='linear', limit_area='inside')
                
                # Forward fill for remaining gaps (more generous limit)
                df[col] = df[col].ffill(limit=4)
                
                # Backward fill for start-of-series gaps
                df[col] = df[col].bfill(limit=2)
                
                # Final safety net for any remaining NaN values
                remaining_missing = df[col].isna().sum()
                if remaining_missing > 0:
                    # Calculate safe fallback value
                    valid_values = df[col].dropna()
                    if len(valid_values) > 0:
                        fallback_value = valid_values.median()
                        if pd.isna(fallback_value):
                            fallback_value = valid_values.mean()
                        if pd.isna(fallback_value):
                            # Last resort defaults
                            fallback_value = 5.0 if 'unemployment_rate' in col else 0.0
                    else:
                        fallback_value = 5.0 if 'unemployment_rate' in col else 0.0
                    
                    df[col] = df[col].fillna(fallback_value)
                    print(f"   SAFETY FILL {col}: Applied fallback value {fallback_value} to {remaining_missing} values")
                
                final_missing = df[col].isna().sum()
                if original_missing > final_missing:
                    filled = original_missing - final_missing
                    print(f"   FILL {col}: Filled {filled} missing values")
        return df
    
    def create_integrated_dataset(self, datasets):
        """Simplified integration focused on client needs"""
        print(f"\nCreating integrated dataset...")
        
        # Step 1: Detect regions from actual data
        all_regions = set()
        for df in datasets.values():
            detected_regions = self._detect_regions_from_columns(df)
            all_regions.update(detected_regions)
        
        if all_regions:
            print(f"   Detected regions from data: {sorted(list(all_regions))[:5]}...")
        
        # Step 2: Get date range and create master timeline
        start_date, end_date = self.get_date_range(datasets)
        master_timeline = self.create_master_timeline(start_date, end_date, datasets)
        
        # Step 3: Create base dataframe
        integrated = pd.DataFrame({'date': master_timeline})
        # Ensure master timeline is datetime type
        integrated['date'] = pd.to_datetime(integrated['date'])
        
        # Step 4: Process each dataset type
        unemployment_added = False
        economic_indicators = 0
        
        for filename, df in datasets.items():
            # DEBUG: Track age file processing
            is_age_file = 'age group region council' in filename.lower()
            is_ect_file = 'ect' in filename.lower() and 'electronic' in filename.lower()
            
            if is_age_file:
                print(f"   DEBUG AGE FILE: Processing {filename}")
                print(f"   DEBUG AGE FILE: Initial columns={len(df.columns)}, rows={len(df)}")
            elif is_ect_file:
                print(f"   DEBUG ECT FILE: Processing {filename}")
                print(f"   DEBUG ECT FILE: Initial columns={len(df.columns)}, rows={len(df)}")
            
            if 'date' not in df.columns:
                if is_age_file:
                    print(f"   DEBUG AGE FILE: SKIPPED - No date column")
                elif is_ect_file:
                    print(f"   DEBUG ECT FILE: SKIPPED - No date column")
                continue
            
            # Clean the dataset with realistic thresholds
            clean_df = self.clean_dataframe(df, dataset_name=filename)
            
            if is_age_file:
                print(f"   DEBUG AGE FILE: After cleaning columns={len(clean_df.columns)}, rows={len(clean_df)}")
            elif is_ect_file:
                print(f"   DEBUG ECT FILE: After cleaning columns={len(clean_df.columns)}, rows={len(clean_df)}")
            
            if len(clean_df.columns) <= 1:  # Only date column left
                if is_age_file:
                    print(f"   DEBUG AGE FILE: SKIPPED - No usable data after cleaning")
                elif is_ect_file:
                    print(f"   DEBUG ECT FILE: SKIPPED - No usable data after cleaning")
                print(f"   WARNING {filename}: No usable data after cleaning")
                continue
            
            # Improved data quality gate - check if ANY column has reasonable data
            numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
            if is_age_file:
                print(f"   DEBUG AGE FILE: Numeric columns found={len(numeric_cols)}")
            elif is_ect_file:
                print(f"   DEBUG ECT FILE: Numeric columns found={len(numeric_cols)}")
            
            if len(numeric_cols) == 0:
                if is_age_file:
                    print(f"   DEBUG AGE FILE: SKIPPED - No numeric columns")
                elif is_ect_file:
                    print(f"   DEBUG ECT FILE: SKIPPED - No numeric columns")
                print(f"   WARNING {filename}: No numeric columns found")
                continue
                
            # Check best column completion rate instead of average
            best_completion = clean_df[numeric_cols].notna().mean().max()
            if is_age_file:
                print(f"   DEBUG AGE FILE: Best completion rate={best_completion:.1%}")
            elif is_ect_file:
                print(f"   DEBUG ECT FILE: Best completion rate={best_completion:.1%}")
                
            if best_completion < 0.2:  # At least one column needs 20% data
                if is_age_file:
                    print(f"   DEBUG AGE FILE: SKIPPED - Poor completion rate")
                elif is_ect_file:
                    print(f"   DEBUG ECT FILE: SKIPPED - Poor completion rate")
                print(f"   SKIP {filename}: Best column only {best_completion:.1%} complete")
                continue
            
            print(f"   OK {filename}: Best variable {best_completion:.1%} complete")
            
            # Align data to master timeline (handles monthly/quarterly conversion)
            aligned_df = self.align_data_to_timeline(clean_df, master_timeline, filename)
            
            # Add dataset prefixes for better column organization
            aligned_df = self.add_dataset_prefixes(aligned_df, filename)
            
            # Ensure consistent datetime type for merge compatibility
            if 'date' in aligned_df.columns:
                try:
                    aligned_df['date'] = pd.to_datetime(aligned_df['date'], format='mixed', errors='coerce')
                except:
                    aligned_df['date'] = pd.to_datetime(aligned_df['date'], errors='coerce')
                # Also ensure the integrated df date column is datetime
                if 'date' in integrated.columns:
                    try:
                        integrated['date'] = pd.to_datetime(integrated['date'], format='mixed', errors='coerce')
                    except:
                        integrated['date'] = pd.to_datetime(integrated['date'], errors='coerce')
            
            if is_age_file:
                print(f"   DEBUG AGE FILE: After alignment columns={len(aligned_df.columns)}, rows={len(aligned_df)}")
            
            # Apply missing data filling
            # NO DATA FILL HERE - preserve missing values to prevent data leakage
            # Data imputation will happen AFTER temporal splitting in temporal_data_splitter
            
            # Enhanced merge logic for unemployment datasets
            if any(keyword in filename.lower() for keyword in ['age group', 'sex', 'ethnic']):
                unemployment_added = True
                
                # Determine dataset type for better tracking
                if 'ethnic' in filename.lower():
                    dataset_type = "ethnic"
                    print(f"   ADDING ethnic unemployment data: {filename} ({sum('Unemployment_Rate' in col for col in aligned_df.columns)} unemployment rate columns)")
                elif 'age group' in filename.lower():
                    dataset_type = "age"
                    print(f"   ADDING age demographic data: {filename} ({sum('Unemployment_Rate' in col for col in aligned_df.columns)} unemployment rate columns)")
                elif 'sex' in filename.lower():
                    dataset_type = "sex"
                    print(f"   ADDING sex demographic data: {filename} ({sum('Unemployment_Rate' in col for col in aligned_df.columns)} unemployment rate columns)")
                else:
                    dataset_type = "demographic"
                    print(f"   ADDING demographic data: {filename}")
                
                # For unemployment datasets, merge without suffixes to preserve column names
                # Check for column conflicts and handle them explicitly
                existing_cols = set(integrated.columns)
                new_cols = set(aligned_df.columns) - {'date'}
                conflicts = existing_cols.intersection(new_cols)
                
                if conflicts:
                    print(f"      INFO: Resolving {len(conflicts)} column conflicts...")
                    # Rename conflicting columns in new dataset to avoid duplicates
                    rename_map = {}
                    for conflict_col in conflicts:
                        # Create unique name based on dataset type and original name
                        base_name = conflict_col.replace('HLF_', '').replace('_', '')
                        new_name = f"HLF_{dataset_type}_{base_name}"
                        rename_map[conflict_col] = new_name
                    
                    # Apply renames to the new dataset
                    aligned_df = aligned_df.rename(columns=rename_map)
                    print(f"      Renamed {len(rename_map)} conflicting columns with '{dataset_type}' prefix")
                
                # Merge cleanly - no conflicts now
                integrated = integrated.merge(aligned_df, on='date', how='left')
            else:
                # Economic indicators - use standard suffix approach
                economic_indicators += 1
                dataset_id = filename.replace('cleaned_', '').replace('.csv', '').replace(' ', '_')[:10]
                integrated = integrated.merge(aligned_df, on='date', how='left', suffixes=('', f'_{dataset_id}'))
                print(f"   ADDED economic indicator: {filename}")
        
        # Data quality gate for final dataset
        if not unemployment_added:
            raise ValueError("CRITICAL: No unemployment data successfully integrated")
        
        print(f"\nIntegration complete:")
        print(f"   Unemployment data: {'Yes' if unemployment_added else 'No'}")
        print(f"   Economic indicators: {economic_indicators}")
        print(f"   Time periods: {len(integrated)}")
        print(f"   Variables: {len(integrated.columns) - 1}")
        
        return integrated
    
    def _find_key_target_columns(self, df):
        """Find essential regional unemployment totals for comprehensive forecasting"""
        target_cols = []
        
        # Debug: Show available unemployment columns
        unemployment_cols = [col for col in df.columns if 'unemployment' in col.lower()]
        if len(unemployment_cols) > 0:
            print(f"   Found {len(unemployment_cols)} unemployment columns, searching for targets...")
            
        # First priority: Look for any unemployment rate columns with regional patterns
        for col in df.columns:
            col_lower = col.lower()
            if 'unemployment_rate' in col_lower or 'unemployment' in col_lower:
                # Look for regional patterns in column names
                regions = ['auckland', 'wellington', 'canterbury', 'northland', 'waikato', 'otago', 'bay', 'hawke', 'taranaki', 'west_coast', 'southland', 'nelson', 'marlborough', 'gisborne']
                if any(region in col_lower for region in regions):
                    target_cols.append(col)
                # Also include any column with 'total' or 'all_ages' indicators
                elif 'total' in col_lower or 'all_ages' in col_lower:
                    target_cols.append(col)
        
        # If still no targets found, include any unemployment rate column
        if len(target_cols) == 0:
            for col in df.columns:
                col_lower = col.lower()
                if 'unemployment_rate' in col_lower:
                    target_cols.append(col)
        
        # If still nothing, look for broader unemployment patterns
        if len(target_cols) == 0:
            for col in df.columns:
                col_lower = col.lower()
                if 'unemployment' in col_lower and 'rate' in col_lower:
                    target_cols.append(col)
        
        if len(target_cols) > 0:
            print(f"   Located {len(target_cols)} regional unemployment targets")
        else:
            print(f"   DEBUG: No unemployment targets found in columns: {[col for col in df.columns if 'unemployment' in col.lower()][:5]}")
        
        return target_cols[:8]  # Maximum 8 targets for comprehensive coverage
    
    def clean_unemployment_outliers(self, df):
        """Clean extreme unemployment outliers using statistical methods"""
        print(f"\nCleaning unemployment outliers...")
        
        outliers_cleaned = 0
        for col in df.columns:
            if 'unemployment_rate' in col:
                values = pd.to_numeric(df[col], errors='coerce')
                
                # Skip if insufficient data
                if values.dropna().empty or len(values.dropna()) < 10:
                    continue
                
                # Calculate statistical bounds (IQR method with multiplier adjustment)
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                
                # Use different multipliers based on demographic volatility
                if ('Asian' in col or 'Pacific' in col) and len(values.dropna()) < 50:
                    # Small sample minorities: More lenient bounds
                    multiplier = 3.0
                else:
                    # Larger samples: Standard bounds
                    multiplier = 2.5
                
                lower_bound = max(0.5, Q1 - multiplier * IQR)  # Never below 0.5%
                upper_bound = Q3 + multiplier * IQR
                
                # Identify outliers
                outliers = (values > upper_bound) | (values < lower_bound)
                
                if outliers.sum() > 0:
                    print(f"   {col}: Smoothing {outliers.sum()} outliers (bounds: {lower_bound:.1f}%-{upper_bound:.1f}%)")
                    
                    # Replace outliers with interpolated values
                    clean_values = values.copy()
                    clean_values[outliers] = None  # Mark outliers as missing
                    
                    # Linear interpolation for outliers
                    clean_values = clean_values.interpolate(method='linear', limit_direction='both')
                    
                    # If interpolation fails, use rolling median
                    remaining_na = clean_values.isna()
                    if remaining_na.sum() > 0:
                        rolling_median = values.rolling(window=8, center=True, min_periods=2).median()
                        clean_values[remaining_na] = rolling_median[remaining_na]
                    
                    # Final fallback to series median with safety check
                    if clean_values.isna().sum() > 0:
                        series_median = values.median()
                        if pd.isna(series_median):
                            # If median is also NaN, use reasonable unemployment default
                            fallback_value = 5.0 if 'unemployment_rate' in col else 0.0
                        else:
                            fallback_value = series_median
                        clean_values = clean_values.fillna(fallback_value)
                    
                    df[col] = clean_values
                    outliers_cleaned += outliers.sum()
        
        print(f"   SMOOTHED {outliers_cleaned} unemployment outliers")
        return df

    def add_ml_features(self, df):
        """Basic time features for regional unemployment forecasting (no lag features to avoid data leakage)"""
        print(f"\nAdding basic time features...")
        
        target_cols = self._find_key_target_columns(df)
        
        if not target_cols:
            print("   WARNING No regional unemployment targets found")
            return df
        
        print(f"   Found {len(target_cols)} regional targets:")
        for col in target_cols:
            region = col.split('_')[0]
            print(f"     - {region}")
        
        # Only add basic time features - lag features will be created after temporal split
        if df['date'].dtype.name.startswith('datetime'):
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
        else:
            print("   WARNING: Date column not datetime type - skipping time features")
            return df
        
        print("   ADDED basic time features (quarter, year)")
        print("   NOTE: Lag features will be created after temporal split to prevent data leakage")
        
        return df
    
    def calculate_data_quality_report(self, df):
        """Simple quality metrics without hardcoded complexity"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Find important columns dynamically
        important_cols = [col for col in numeric_cols if self._is_important_column(col)]
        
        quality_report = {
            'total_periods': len(df),
            'total_variables': len(numeric_cols),
            'important_variables': len(important_cols),
            'overall_completion': df[numeric_cols].notna().mean().mean(),
            'important_completion': df[important_cols].notna().mean().mean() if important_cols else 0,
            'complete_records': df.dropna().shape[0],
            'date_range': f"{df['date'].min()} to {df['date'].max()}"
        }
        
        return quality_report
    
    def run_integration(self):
        """Main integration process"""
        print("="*60)
        print("SIMPLIFIED TIME SERIES INTEGRATION")
        print("="*60)
        
        # Load datasets
        datasets = self.load_datasets()
        if not datasets:
            print("ERROR No datasets found!")
            return None
        
        # Create integrated dataset
        try:
            integrated_df = self.create_integrated_dataset(datasets)
            
            # Clean unemployment outliers BEFORE adding features
            integrated_df = self.clean_unemployment_outliers(integrated_df)
            
            # Add ML features
            integrated_df = self.add_ml_features(integrated_df)
            
            # Generate quality report
            quality_report = self.calculate_data_quality_report(integrated_df)
            
            # Final cleanup: Handle trailing NaN values in unemployment targets  
            unemployment_cols = [col for col in integrated_df.columns if 'unemployment_rate' in col]
            print(f"\nFinal cleanup: smoothing {len(unemployment_cols)} unemployment columns...")
            
            for col in unemployment_cols:
                if col in integrated_df.columns:
                    # Forward fill recent NaN values (up to 2 quarters)
                    integrated_df[col] = integrated_df[col].ffill(limit=2)
                    
            print("   Applied forward-fill smoothing for recent missing quarters")
            
            # Save outputs
            output_file = self.output_dir / "integrated_forecasting_dataset.csv"
            integrated_df.to_csv(output_file, index=False)
            
            metrics_file = self.output_dir / "integration_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            
            print(f"\nFiles saved:")
            print(f"   Dataset: {output_file}")
            print(f"   Metrics: {metrics_file}")
            
            print(f"\nDATA QUALITY SUMMARY:")
            print(f"   Overall completion: {quality_report['overall_completion']:.1%}")
            print(f"   Important vars completion: {quality_report['important_completion']:.1%}")
            print(f"   Complete records: {quality_report['complete_records']}/{quality_report['total_periods']}")
            print(f"   Variables: {quality_report['important_variables']}/{quality_report['total_variables']} important")
            
            return integrated_df, quality_report
            
        except Exception as e:
            print(f"ERROR Integration failed: {e}")
            return None

def main():
    """Execute simplified integration"""
    aligner = SimpleTimeSeriesAligner()
    result = aligner.run_integration()
    
    if result:
        print("\n" + "="*60)
        print("INTEGRATION SUCCESSFUL")
        print("="*60)
        print("Ready for unemployment forecasting models")
        print("Dataset optimized for ARIMA/LSTM/Random Forest")
        print("Dynamic regional analysis based on available data")
    else:
        print("\nERROR Integration failed - check data quality issues")

if __name__ == "__main__":
    main()