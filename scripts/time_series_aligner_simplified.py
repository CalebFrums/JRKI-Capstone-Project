#!/usr/bin/env python3
"""
Simplified Time Series Alignment Script
Unemployment Forecasting Project - Streamlined for Actual Needs

Simplified based on sub-agent feedback:
- Removed over-engineering patterns
- Single data quality filtering approach  
- Direct column selection logic
- Essential ML feature preparation
- Realistic data quality expectations

Author: Enhanced for Client Requirements
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

class SimpleTimeSeriesAligner:
    """Streamlined time series alignment focused on actual client needs"""
    
    def __init__(self, data_dir="data_cleaned", output_dir="data_cleaned"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        print(f"📊 Simple Time Series Aligner initialized")
        print(f"📁 Data: {self.data_dir} → Output: {self.output_dir}")
    
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
            cpi_cols = [col for col in df_sample.columns if 'cpi' in col.lower()]
            if len(cpi_cols) == 0:
                errors.append(f"CRITICAL: {filename} claims to be CPI data but has no CPI columns")
        
        elif 'gdp' in filename.lower():
            gdp_cols = [col for col in df_sample.columns if 'gdp' in col.lower()]
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
        
        print(f"\n📂 Loading {len(csv_files)} cleaned datasets:")
        
        validation_errors = []
        
        for csv_file in csv_files:
            try:
                # Validate file structure first
                errors, warnings_list = self.validate_cleaned_file(csv_file)
                
                if errors:
                    print(f"❌ {csv_file.name} VALIDATION FAILED:")
                    for error in errors:
                        print(f"   • {error}")
                    validation_errors.extend(errors)
                    continue
                
                if warnings_list:
                    for warning in warnings_list:
                        print(f"⚠️ {warning}")
                
                # Load if validation passed
                df = pd.read_csv(csv_file)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    # Remove rows with invalid dates
                    df = df.dropna(subset=['date'])
                
                datasets[csv_file.name] = df
                print(f"✅ {csv_file.name}: {len(df)} records")
                
            except Exception as e:
                error_msg = f"Failed to load {csv_file.name}: {e}"
                print(f"❌ {error_msg}")
                validation_errors.append(error_msg)
        
        # Check if we have any valid datasets
        if len(datasets) == 0:
            raise ValueError("No valid cleaned datasets found - all files failed validation")
        
        # Check for critical validation errors
        if validation_errors:
            print(f"\n🚨 VALIDATION ERRORS DETECTED:")
            for error in validation_errors:
                print(f"  • {error}")
            print(f"\n💡 These look like wrong datasets - expecting cleaned unemployment/economic data from Stats NZ")
            
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
            'unemployment_rate' in col_lower or
            'gdp_millions' in col_lower or
            'cpi_value' in col_lower or
            ('cpi_' in col_lower and 'unknown' not in col_lower)
        )
    
    def clean_dataframe(self, df, dataset_name=""):
        """Simple data quality filter with dynamic region detection"""
        if len(df) == 0:
            return df
        
        original_cols = len(df.columns)
        
        # Simple approach - drop obviously broken columns, keep important ones
        cols_to_drop = []
        for col in df.columns:
            if not self._is_important_column(col):
                # Drop unknown columns
                if 'unknown' in col.lower():
                    cols_to_drop.append(col)
                # Drop completely empty columns
                elif df[col].notna().sum() == 0:
                    cols_to_drop.append(col)
                # Drop columns with <30% data
                elif df[col].notna().mean() < 0.3:
                    cols_to_drop.append(col)
        
        df = df.drop(columns=cols_to_drop)
        
        cleaned_cols = len(df.columns)
        if cleaned_cols < original_cols:
            print(f"   🧹 {dataset_name}: {original_cols} → {cleaned_cols} columns")
        
        return df
    
    def get_date_range(self, datasets):
        """Simple date range calculation"""
        all_dates = []
        for df in datasets.values():
            if 'date' in df.columns:
                valid_dates = pd.to_datetime(df['date'], errors='coerce').dropna()
                all_dates.extend(valid_dates)
        
        if not all_dates:
            return pd.Timestamp('2010-01-01'), pd.Timestamp('2023-12-31')
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        print(f"📅 Date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
        
        return start_date, end_date
    
    def create_master_timeline(self, start_date, end_date):
        """Create quarterly timeline"""
        timeline = pd.date_range(start=start_date, end=end_date, freq='QS')
        print(f"⏰ Master timeline: {len(timeline)} quarterly periods")
        return timeline
    
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
                
                final_missing = df[col].isna().sum()
                if original_missing > final_missing:
                    filled = original_missing - final_missing
                    print(f"   📈 {col}: Filled {filled} missing values")
        return df
    
    def create_integrated_dataset(self, datasets):
        """Simplified integration focused on client needs"""
        print(f"\n🔄 Creating integrated dataset...")
        
        # Step 1: Detect regions from actual data
        all_regions = set()
        for df in datasets.values():
            detected_regions = self._detect_regions_from_columns(df)
            all_regions.update(detected_regions)
        
        if all_regions:
            print(f"   📍 Detected regions from data: {sorted(list(all_regions))[:5]}...")
        
        # Step 2: Get date range and create master timeline
        start_date, end_date = self.get_date_range(datasets)
        master_timeline = self.create_master_timeline(start_date, end_date)
        
        # Step 3: Create base dataframe
        integrated = pd.DataFrame({'date': master_timeline})
        
        # Step 4: Process each dataset type
        unemployment_added = False
        economic_indicators = 0
        
        for filename, df in datasets.items():
            if 'date' not in df.columns:
                continue
            
            # Clean the dataset with realistic thresholds
            clean_df = self.clean_dataframe(df, dataset_name=filename)
            
            if len(clean_df.columns) <= 1:  # Only date column left
                print(f"   ⚠️ {filename}: No usable data after cleaning")
                continue
            
            # Improved data quality gate - check if ANY column has reasonable data
            numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                print(f"   ⚠️ {filename}: No numeric columns found")
                continue
                
            # Check best column completion rate instead of average
            best_completion = clean_df[numeric_cols].notna().mean().max()
            if best_completion < 0.2:  # At least one column needs 20% data
                print(f"   ❌ {filename}: Best column only {best_completion:.1%} complete")
                continue
            
            print(f"   ✅ {filename}: Best variable {best_completion:.1%} complete")
            
            # Apply missing data filling
            clean_df = self.basic_missing_data_fill(clean_df)
            
            # Merge with integrated dataset using descriptive suffixes
            # Extract dataset identifier for suffix
            dataset_id = filename.replace('cleaned_', '').replace('.csv', '').replace(' ', '_')[:10]
            integrated = integrated.merge(clean_df, on='date', how='left', suffixes=('', f'_{dataset_id}'))
            
            # Track what we added
            if any(keyword in filename.lower() for keyword in ['age group', 'sex', 'ethnic']):
                unemployment_added = True
                print(f"   📊 Added unemployment data: {filename}")
            else:
                economic_indicators += 1
                print(f"   💰 Added economic indicator: {filename}")
        
        # Data quality gate for final dataset
        if not unemployment_added:
            raise ValueError("❌ CRITICAL: No unemployment data successfully integrated")
        
        print(f"\n✅ Integration complete:")
        print(f"   📊 Unemployment data: {'Yes' if unemployment_added else 'No'}")
        print(f"   💰 Economic indicators: {economic_indicators}")
        print(f"   📈 Time periods: {len(integrated)}")
        print(f"   📋 Variables: {len(integrated.columns) - 1}")
        
        return integrated
    
    def _find_key_target_columns(self, df):
        """Find essential regional unemployment totals for comprehensive forecasting"""
        target_cols = []
        
        # First priority: Total All Ages unemployment for main regions
        for col in df.columns:
            col_lower = col.lower()
            if 'unemployment' in col_lower and 'total_all_ages' in col_lower:
                target_cols.append(col)
        
        # If no Total All Ages found, look for regional totals from Sex Regional dataset
        if len(target_cols) < 3:
            priority_regions = ['Auckland', 'Wellington', 'Canterbury', 'Northland', 'Waikato', 'Otago']
            for col in df.columns:
                col_lower = col.lower()
                if 'unemployment' in col_lower:
                    # Check for regional totals (not demographic breakdowns)
                    if any(f'{region.lower()}_total_all_ages' in col_lower for region in priority_regions):
                        target_cols.append(col)
        
        return target_cols[:6]  # Maximum 6 regional targets for comprehensive coverage
    
    def add_ml_features(self, df):
        """Essential ML features for regional unemployment forecasting"""
        print(f"\n🤖 Adding essential ML features...")
        
        target_cols = self._find_key_target_columns(df)
        
        if not target_cols:
            print("   ⚠️ No regional unemployment targets found")
            return df
        
        print(f"   🎯 Selected {len(target_cols)} regional targets:")
        for col in target_cols:
            region = col.split('_')[0]
            print(f"     - {region}")
        
        # Essential features only - single lag per target (no moving averages)
        for target_col in target_cols:
            if target_col:
                df[f'{target_col}_lag_1'] = df[target_col].shift(1)
                region = target_col.split('_')[0]
                print(f"   📈 Added lag feature for {region}")
        
        # Basic time features
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
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
        print("🎯 SIMPLIFIED TIME SERIES INTEGRATION")
        print("="*60)
        
        # Load datasets
        datasets = self.load_datasets()
        if not datasets:
            print("❌ No datasets found!")
            return None
        
        # Create integrated dataset
        try:
            integrated_df = self.create_integrated_dataset(datasets)
            
            # Add ML features
            integrated_df = self.add_ml_features(integrated_df)
            
            # Generate quality report
            quality_report = self.calculate_data_quality_report(integrated_df)
            
            # Save outputs
            output_file = self.output_dir / "integrated_forecasting_dataset.csv"
            integrated_df.to_csv(output_file, index=False)
            
            metrics_file = self.output_dir / "integration_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            
            print(f"\n💾 Files saved:")
            print(f"   📊 Dataset: {output_file}")
            print(f"   📋 Metrics: {metrics_file}")
            
            print(f"\n📈 DATA QUALITY SUMMARY:")
            print(f"   Overall completion: {quality_report['overall_completion']:.1%}")
            print(f"   Important vars completion: {quality_report['important_completion']:.1%}")
            print(f"   Complete records: {quality_report['complete_records']}/{quality_report['total_periods']}")
            print(f"   Variables: {quality_report['important_variables']}/{quality_report['total_variables']} important")
            
            return integrated_df, quality_report
            
        except Exception as e:
            print(f"❌ Integration failed: {e}")
            return None

def main():
    """Execute simplified integration"""
    aligner = SimpleTimeSeriesAligner()
    result = aligner.run_integration()
    
    if result:
        print("\n" + "="*60)
        print("✅ INTEGRATION SUCCESSFUL")
        print("="*60)
        print("🚀 Ready for unemployment forecasting models")
        print("📊 Dataset optimized for ARIMA/LSTM/Random Forest")
        print("🌏 Dynamic regional analysis based on available data")
    else:
        print("\n❌ Integration failed - check data quality issues")

if __name__ == "__main__":
    main()