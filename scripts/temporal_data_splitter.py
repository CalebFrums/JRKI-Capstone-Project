#!/usr/bin/env python3
"""
Temporal Data Splitter with Proper Lag Feature Creation
Unemployment Forecasting Project - Data Leakage Fix

This script addresses the data leakage issue identified by proper ML methodology:
- Performs temporal train/validation/test split FIRST
- Creates lag features AFTER splitting using only available historical data
- Ensures no future information leaks into training data

Author: Enhanced for Methodological Correctness
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class TemporalDataSplitter:
    """Proper temporal splitting with lag features created after split"""
    
    def __init__(self, data_dir="data_cleaned", output_dir="model_ready_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Rolling time window configuration (dynamic based on data)
        self.train_years = 16  # Years for training data
        self.validation_years = 4  # Years for validation data
        self.test_years = 2  # Minimum years for test data
        
        # These will be calculated dynamically based on available data
        self.train_end = None
        self.validation_start = None
        self.validation_end = None
        self.test_start = None
        
        print(f"Temporal Data Splitter initialized")
        print(f"Data: {self.data_dir} -> Output: {self.output_dir}")
        print(f"Rolling window config: Train={self.train_years}y, Val={self.validation_years}y, Test>={self.test_years}y")
    
    def load_integrated_data(self):
        """Load the integrated dataset (should not have lag features)"""
        data_file = self.data_dir / "integrated_forecasting_dataset.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Integrated dataset not found: {data_file}")
        
        print(f"\nLoading integrated dataset: {data_file}")
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Check for existing lag features (should be removed)
        lag_cols = [col for col in df.columns if 'lag' in col.lower()]
        if lag_cols:
            print(f"WARNING: Found existing lag features: {len(lag_cols)} columns")
            print("   These will be ignored - creating proper lag features after split")
            df = df.drop(columns=lag_cols)
        
        print(f"Loaded dataset: {len(df)} records, {len(df.columns)} features")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def calculate_dynamic_boundaries(self, df):
        """Calculate rolling time window boundaries based on available data"""
        print(f"\nCalculating dynamic temporal boundaries...")
        
        # Get data date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        print(f"Available data range: {min_date} to {max_date}")
        
        # Calculate boundaries working backwards from the most recent data
        # Test set: most recent data (minimum test_years)
        self.test_start = max_date - pd.DateOffset(years=self.test_years)
        
        # Validation set: before test set
        self.validation_end = self.test_start - pd.DateOffset(days=1)
        self.validation_start = self.validation_end - pd.DateOffset(years=self.validation_years) + pd.DateOffset(days=1)
        
        # Training set: everything before validation (minimum train_years)
        self.train_end = self.validation_start - pd.DateOffset(days=1)
        
        # Ensure we don't go before the available data
        if self.train_end < min_date + pd.DateOffset(years=self.train_years):
            print(f"WARNING: Insufficient historical data for {self.train_years} years of training")
            print(f"Using all available data before validation period")
        
        # Convert to string format for consistency
        self.train_end = self.train_end.strftime('%Y-%m-%d')
        self.validation_start = self.validation_start.strftime('%Y-%m-%d')
        self.validation_end = self.validation_end.strftime('%Y-%m-%d')
        self.test_start = self.test_start.strftime('%Y-%m-%d')
        
        print(f"Dynamic boundaries calculated:")
        print(f"   Train: up to {self.train_end}")
        print(f"   Validation: {self.validation_start} to {self.validation_end}")
        print(f"   Test: from {self.test_start}")
        
        # Verify the splits make sense
        actual_data_end = max_date.strftime('%Y-%m-%d')
        if self.test_start > actual_data_end:
            raise ValueError(f"Test start date {self.test_start} is after available data ends {actual_data_end}")

    def perform_temporal_split(self, df):
        """Split data temporally into train/validation/test"""
        print(f"\nPerforming temporal split...")
        
        # Ensure data is sorted by date
        df = df.sort_values('date')
        
        # Create temporal masks
        train_mask = df['date'] <= self.train_end
        validation_mask = (df['date'] >= self.validation_start) & (df['date'] <= self.validation_end)
        test_mask = df['date'] >= self.test_start
        
        # Split the data
        train_data = df[train_mask].copy()
        validation_data = df[validation_mask].copy()  
        test_data = df[test_mask].copy()
        
        print(f"   Train: {len(train_data)} records ({train_data['date'].min()} to {train_data['date'].max()})")
        print(f"   Validation: {len(validation_data)} records ({validation_data['date'].min()} to {validation_data['date'].max()})")
        print(f"   Test: {len(test_data)} records ({test_data['date'].min()} to {test_data['date'].max()})")
        
        # Verify no temporal overlap
        if len(train_data) == 0 or len(validation_data) == 0 or len(test_data) == 0:
            raise ValueError("One or more splits is empty - check date boundaries")
        
        if train_data['date'].max() >= validation_data['date'].min():
            print("WARNING: Temporal overlap between train and validation")
        
        if validation_data['date'].max() >= test_data['date'].min():
            print("WARNING: Temporal overlap between validation and test")
        
        return train_data, validation_data, test_data
    
    def find_target_regions(self, df):
        """Find the main target regions for forecasting"""
        target_columns = []
        
        # Look for Male unemployment rate columns (most complete target)
        for col in df.columns:
            if 'Male_unemployment_rate' in col and 'lag' not in col.lower():
                # Extract region name
                region = col.replace('_Male_unemployment_rate', '')
                if region in ['Auckland', 'Wellington', 'Canterbury']:
                    target_columns.append(col)
        
        print(f"   Found target regions: {[col.replace('_Male_unemployment_rate', '') for col in target_columns]}")
        return target_columns
    
    def create_lag_features(self, df, target_columns, split_name):
        """Create lag features using only available historical data"""
        print(f"   Creating lag features for {split_name}...")
        
        df = df.copy()
        
        # Create lag features for target variables
        lag_features_added = 0
        for target_col in target_columns:
            if target_col in df.columns:
                # Lag 1 (previous quarter)
                df[f'{target_col}_lag1'] = df[target_col].shift(1)
                
                # Lag 4 (same quarter previous year) 
                df[f'{target_col}_lag4'] = df[target_col].shift(4)
                
                lag_features_added += 2
        
        # Create lag features for related demographic columns
        for col in df.columns:
            if ('unemployment_rate' in col and 
                'lag' not in col.lower() and 
                col not in target_columns and
                any(region in col for region in ['Auckland', 'Wellington', 'Canterbury'])):
                
                # Only lag 1 for related columns to keep feature set manageable
                df[f'{col}_lag1'] = df[col].shift(1)
                lag_features_added += 1
        
        # Create moving averages (3-quarter)
        ma_features_added = 0
        for target_col in target_columns:
            if target_col in df.columns:
                df[f'{target_col}_ma3'] = df[target_col].rolling(window=3, min_periods=1).mean()
                ma_features_added += 1
        
        # Create economic indicator changes
        econ_features_added = 0
        for col in ['cpi_value', 'lci_value_LCI_All_Se']:
            if col in df.columns:
                # Quarterly change
                df[f'{col}_change'] = df[col].pct_change(fill_method=None)
                
                # Annual change (4 quarters)
                df[f'{col}_annual_change'] = df[col].pct_change(periods=4, fill_method=None)
                
                econ_features_added += 2
        
        print(f"     Added {lag_features_added} lag features")
        print(f"     Added {ma_features_added} moving average features")
        print(f"     Added {econ_features_added} economic change features")
        
        return df
    
    def create_feature_summary(self, train_data, validation_data, test_data, target_columns):
        """Create feature summary metadata"""
        
        # Count different types of features
        all_features = train_data.columns.tolist()
        
        lag_features = [col for col in all_features if 'lag' in col.lower()]
        ma_features = [col for col in all_features if '_ma' in col]
        change_features = [col for col in all_features if 'change' in col.lower()]
        unemployment_features = [col for col in all_features if 'unemployment' in col.lower() and 'lag' not in col.lower()]
        
        summary = {
            "total_features": len(all_features) - 1,  # Exclude date
            "target_regions": [col.replace('_Male_unemployment_rate', '') for col in target_columns],
            "target_columns": target_columns,
            "data_splits": {
                "train_records": len(train_data),
                "validation_records": len(validation_data),
                "test_records": len(test_data)
            },
            "feature_types": {
                "unemployment_rates": len(unemployment_features),
                "lag_features": len(lag_features),
                "moving_averages": len(ma_features),
                "economic_indicators": len([col for col in all_features if col in ['cpi_value', 'lci_value_LCI_All_Se']]),
                "economic_changes": len(change_features)
            }
        }
        
        return summary
    
    def run_temporal_split_and_feature_creation(self):
        """Main process: temporal split followed by proper lag feature creation"""
        print("="*70)
        print("TEMPORAL DATA SPLITTING WITH PROPER LAG FEATURE CREATION")
        print("="*70)
        
        # Load integrated data
        df = self.load_integrated_data()
        
        # Calculate dynamic boundaries based on actual data range
        self.calculate_dynamic_boundaries(df)
        
        # Perform temporal split
        train_data, validation_data, test_data = self.perform_temporal_split(df)
        
        # Find target columns
        target_columns = self.find_target_regions(df)
        
        if not target_columns:
            raise ValueError("No target columns found for forecasting")
        
        print(f"\nCreating lag features for each split...")
        
        # Create lag features for each split independently
        train_data = self.create_lag_features(train_data, target_columns, "TRAIN")
        validation_data = self.create_lag_features(validation_data, target_columns, "VALIDATION") 
        test_data = self.create_lag_features(test_data, target_columns, "TEST")
        
        # Create feature summary
        feature_summary = self.create_feature_summary(train_data, validation_data, test_data, target_columns)
        
        # Save the properly split and featured datasets
        train_file = self.output_dir / "train_data.csv"
        validation_file = self.output_dir / "validation_data.csv"
        test_file = self.output_dir / "test_data.csv"
        summary_file = self.output_dir / "feature_summary.json"
        
        train_data.to_csv(train_file, index=False)
        validation_data.to_csv(validation_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        with open(summary_file, 'w') as f:
            json.dump(feature_summary, f, indent=2, default=str)
        
        print(f"\nFiles saved:")
        print(f"   Training: {train_file}")
        print(f"   Validation: {validation_file}")
        print(f"   Test: {test_file}")
        print(f"   Summary: {summary_file}")
        
        print(f"\nDATA PROCESSING SUMMARY:")
        print(f"   [OK] Dynamic temporal split calculated from data range")
        print(f"   [OK] Temporal split performed BEFORE lag feature creation")
        print(f"   [OK] Lag features use only historical data within each split")
        print(f"   [OK] No future information leaks into training data")
        print(f"   [OK] Features: {feature_summary['total_features']} total")
        print(f"   [OK] Targets: {len(target_columns)} regions")
        print(f"   [OK] System ready for quarterly updates with rolling windows")
        
        return train_data, validation_data, test_data, feature_summary

def main():
    """Execute temporal split with proper lag feature creation"""
    splitter = TemporalDataSplitter()
    result = splitter.run_temporal_split_and_feature_creation()
    
    if result:
        print("\n" + "="*70)
        print("DATA LEAKAGE FIX SUCCESSFUL")
        print("="*70)
        print("[OK] Temporal integrity preserved")
        print("[OK] Lag features created after split") 
        print("[OK] No future information in training data")
        print("[OK] Ready for model training with proper methodology")
    else:
        print("\nERROR: Temporal split failed")

if __name__ == "__main__":
    main()