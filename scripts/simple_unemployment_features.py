#!/usr/bin/env python3
"""
Simplified Unemployment Feature Engineering
NZ Unemployment Forecasting - Week 7 Model Preparation

Pragmatic approach for 27% data completion rate:
- Essential features only (lag features, basic economic indicators)
- Robust missing data handling
- Single dataset output for all model types
- Focus on Auckland, Wellington, Canterbury

Author: Team JRKI - Simplified Based on Sub-Agent Feedback
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

def validate_dataset_schema(df, filepath):
    """Validate that the dataset is actually unemployment/economic data"""
    print("🔍 Validating dataset schema and content...")
    
    errors = []
    warnings_list = []
    
    # Check 1: Must have date column
    if 'date' not in df.columns:
        errors.append("CRITICAL: No 'date' column found - this doesn't look like a time series dataset")
    
    # Check 2: Look for unemployment-related columns
    unemployment_cols = [col for col in df.columns if 'unemployment' in col.lower()]
    if len(unemployment_cols) == 0:
        errors.append("CRITICAL: No unemployment columns found - this appears to be the wrong dataset")
    
    # Check 3: Validate unemployment rate ranges (should be 0-100%)
    for col in unemployment_cols:
        if df[col].notna().sum() > 0:  # Only check if has data
            max_val = df[col].max()
            min_val = df[col].min()
            
            # Check for obviously wrong values
            if max_val > 100:
                if max_val > 1000:
                    errors.append(f"CRITICAL: {col} has values up to {max_val:.0f} - this looks like population data, not unemployment rates")
                else:
                    warnings_list.append(f"WARNING: {col} has rates above 100% (max: {max_val:.1f}) - unusual but possible")
            
            if min_val < 0:
                errors.append(f"CRITICAL: {col} has negative values ({min_val:.1f}) - unemployment rates cannot be negative")
    
    # Check 4: Look for expected economic indicators
    economic_indicators = ['cpi', 'gdp', 'lci']
    found_indicators = []
    for indicator in economic_indicators:
        indicator_cols = [col for col in df.columns if indicator in col.lower()]
        if indicator_cols:
            found_indicators.append(indicator.upper())
    
    if len(found_indicators) == 0:
        warnings_list.append("WARNING: No economic indicators (CPI, GDP, LCI) found - feature engineering will be limited")
    
    # Check 5: Date range validation
    if 'date' in df.columns:
        earliest_date = df['date'].min()
        latest_date = df['date'].max()
        
        if earliest_date.year < 1900:
            warnings_list.append(f"WARNING: Data starts from {earliest_date.year} - very old data, may not be relevant")
        
        if latest_date.year > 2030:
            warnings_list.append(f"WARNING: Data extends to {latest_date.year} - future projections detected")
        
        date_span = latest_date.year - earliest_date.year
        if date_span < 10:
            warnings_list.append(f"WARNING: Only {date_span} years of data - may be insufficient for forecasting")
    
    # Check 6: Regional coverage validation
    target_regions = ['auckland', 'wellington', 'canterbury']
    found_regions = []
    for region in target_regions:
        region_cols = [col for col in df.columns if region in col.lower() and 'unemployment' in col.lower()]
        if region_cols:
            found_regions.append(region.capitalize())
    
    if len(found_regions) == 0:
        errors.append("CRITICAL: No target regions (Auckland, Wellington, Canterbury) found in unemployment data")
    
    # Print validation results
    if errors:
        print("❌ DATASET VALIDATION FAILED:")
        for error in errors:
            print(f"  • {error}")
        print(f"\n🚨 This appears to be the WRONG DATASET for unemployment forecasting!")
        print(f"📁 File checked: {filepath}")
        print("💡 Expected: integrated_forecasting_dataset.csv with unemployment rates (0-100%) for NZ regions")
        raise ValueError("Dataset validation failed - wrong dataset provided")
    
    if warnings_list:
        print("⚠️ VALIDATION WARNINGS:")
        for warning in warnings_list:
            print(f"  • {warning}")
    
    print(f"✅ Dataset validation passed")
    print(f"  • Found {len(unemployment_cols)} unemployment columns")
    print(f"  • Found {len(found_indicators)} economic indicators: {', '.join(found_indicators)}")
    print(f"  • Found {len(found_regions)} target regions: {', '.join(found_regions)}")
    
    return True

def load_integrated_data():
    """Load and validate the integrated forecasting dataset"""
    print("📊 Loading integrated forecasting dataset...")
    
    data_path = Path("data_cleaned/integrated_forecasting_dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        # Convert date column to datetime immediately after loading
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        raise ValueError(f"Failed to load CSV file - file may be corrupted or wrong format: {e}")
    
    # Validate dataset before processing
    validate_dataset_schema(df, data_path)
    
    # Process dates
    if 'date' not in df.columns:
        raise ValueError("No date column found after validation - this should not happen")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Check for date parsing failures
    invalid_dates = df['date'].isna().sum()
    if invalid_dates > 0:
        print(f"⚠️ WARNING: {invalid_dates} invalid dates found and removed")
    
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    
    if len(df) == 0:
        raise ValueError("No valid dates found in dataset - wrong date format or corrupted data")
    
    print(f"✅ Loaded {len(df)} records from {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    return df

def handle_missing_data(df):
    """Address the 27% completion rate with practical imputation"""
    print("🔧 Handling missing data...")
    
    # Drop columns with >80% missing data (unusable for modeling)
    missing_pct = df.isnull().mean()
    usable_cols = ['date'] + [col for col in df.columns if col != 'date' and missing_pct[col] < 0.8]
    df_clean = df[usable_cols].copy()
    
    dropped_cols = len(df.columns) - len(df_clean.columns)
    print(f"📉 Dropped {dropped_cols} columns with >80% missing data")
    
    # Forward fill unemployment rates (reasonable for quarterly data)
    unemployment_cols = [col for col in df_clean.columns if 'unemployment' in col.lower()]
    for col in unemployment_cols:
        df_clean[col] = df_clean[col].ffill().bfill()
    
    # Linear interpolation for economic indicators
    economic_cols = [col for col in df_clean.columns if any(x in col.lower() for x in ['cpi', 'gdp', 'lci'])]
    for col in economic_cols:
        df_clean[col] = df_clean[col].interpolate(method='linear').ffill().bfill()
    
    # Final check - drop rows with >50% missing values
    threshold = len(df_clean.columns) * 0.5
    df_final = df_clean.dropna(thresh=threshold)
    
    print(f"✅ Cleaned data: {len(df_final)} usable records, {len(df_clean.columns)-1} features")
    return df_final

def create_essential_features(df):
    """Create only essential features for sparse data"""
    print("⚙️ Creating essential features...")
    
    df_features = df.copy()
    
    # Target regions for unemployment forecasting
    target_regions = ['auckland', 'wellington', 'canterbury']
    unemployment_cols = [col for col in df.columns if 'unemployment' in col.lower()]
    
    # Essential lag features (1-2 lags only)
    features_created = 0
    for col in unemployment_cols:
        if any(region in col.lower() for region in target_regions):
            # 1 quarter lag
            df_features[f"{col}_lag1"] = df[col].shift(1)
            # 4 quarter lag (annual pattern)
            df_features[f"{col}_lag4"] = df[col].shift(4)
            features_created += 2
    
    # Basic economic change rates
    economic_cols = [col for col in df.columns if any(x in col.lower() for x in ['cpi', 'gdp', 'lci'])]
    for col in economic_cols:
        if df[col].notna().sum() > 8:  # Need sufficient data
            # Quarterly change
            df_features[f"{col}_change"] = df[col].pct_change()
            # Annual change  
            df_features[f"{col}_annual_change"] = df[col].pct_change(periods=4)
            features_created += 2
    
    # Simple moving averages (3-quarter only)
    for col in unemployment_cols:
        if any(region in col.lower() for region in target_regions) and df[col].notna().sum() > 6:
            df_features[f"{col}_ma3"] = df[col].rolling(window=3, min_periods=1).mean()
            features_created += 1
    
    # Remove zero variance features (constant columns that don't help models)
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    zero_var_cols = []
    for col in numeric_cols:
        if col != 'date' and df_features[col].var() == 0:
            zero_var_cols.append(col)
    
    if zero_var_cols:
        df_features = df_features.drop(columns=zero_var_cols)
        print(f"🧹 Removed {len(zero_var_cols)} zero variance features")
    
    print(f"✅ Created {features_created} essential features")
    return df_features

def create_regional_targets(df):
    """Identify and validate target unemployment rates for 3 regions"""
    print("🎯 Identifying regional targets...")
    
    target_regions = ['auckland', 'wellington', 'canterbury']
    unemployment_cols = [col for col in df.columns if 'unemployment' in col.lower()]
    
    regional_targets = {}
    for region in target_regions:
        # Find unemployment columns for this region
        region_cols = [col for col in unemployment_cols if region in col.lower()]
        
        if region_cols:
            # Use the column with best data completeness
            best_col = max(region_cols, key=lambda col: df[col].notna().sum())
            completion_rate = df[best_col].notna().mean()
            
            if completion_rate > 0.3:  # At least 30% completion
                regional_targets[region] = best_col
                print(f"  • {region.capitalize()}: {best_col} ({completion_rate:.1%} complete)")
            else:
                print(f"  ⚠️ {region.capitalize()}: Insufficient data ({completion_rate:.1%} complete)")
        else:
            print(f"  ❌ {region.capitalize()}: No unemployment data found")
    
    return regional_targets

def temporal_split(df, train_ratio=0.7, val_ratio=0.15):
    """Create temporal train/validation/test splits"""
    print("📅 Creating temporal data splits...")
    
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = df.iloc[:n_train].copy()
    val_data = df.iloc[n_train:n_train + n_val].copy()
    test_data = df.iloc[n_train + n_val:].copy()
    
    print(f"  • Training: {len(train_data)} records ({train_data['date'].min().strftime('%Y-%m')} to {train_data['date'].max().strftime('%Y-%m')})")
    print(f"  • Validation: {len(val_data)} records ({val_data['date'].min().strftime('%Y-%m')} to {val_data['date'].max().strftime('%Y-%m')})")
    print(f"  • Test: {len(test_data)} records ({test_data['date'].min().strftime('%Y-%m')} to {test_data['date'].max().strftime('%Y-%m')})")
    
    return train_data, val_data, test_data

def validate_model_readiness(train_data, val_data, test_data, regional_targets):
    """Check if datasets are ready for model training"""
    print("🔍 Validating model readiness...")
    
    issues = []
    
    # Check minimum data requirements
    if len(train_data) < 20:
        issues.append(f"Training data too short: {len(train_data)} records (need 20+)")
    
    if len(test_data) < 8:
        issues.append(f"Test data too short: {len(test_data)} records (need 8+)")
    
    # Check target variable completeness in training data
    for region, target_col in regional_targets.items():
        train_completeness = train_data[target_col].notna().mean()
        if train_completeness < 0.6:
            issues.append(f"{region} training data only {train_completeness:.1%} complete")
    
    # Check for feature variance
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    zero_var_cols = []
    for col in numeric_cols:
        if train_data[col].var() == 0:
            zero_var_cols.append(col)
    
    if zero_var_cols:
        issues.append(f"Zero variance features: {len(zero_var_cols)} columns")
    
    # Report validation results
    if issues:
        print("  ⚠️ Issues found:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("  ✅ All validation checks passed")
        return True

def export_model_ready_data(train_data, val_data, test_data, regional_targets):
    """Export single dataset for all model types"""
    print("💾 Exporting model-ready datasets...")
    
    output_dir = Path("model_ready_data")
    output_dir.mkdir(exist_ok=True)
    
    # Export datasets
    train_data.to_csv(output_dir / "train_data.csv", index=False)
    val_data.to_csv(output_dir / "validation_data.csv", index=False)
    test_data.to_csv(output_dir / "test_data.csv", index=False)
    
    # Create feature summary
    feature_summary = {
        "total_features": len(train_data.columns) - 1,  # Exclude date
        "target_regions": list(regional_targets.keys()),
        "target_columns": list(regional_targets.values()),
        "data_splits": {
            "train_records": len(train_data),
            "validation_records": len(val_data),
            "test_records": len(test_data)
        },
        "feature_types": {
            "unemployment_rates": len([col for col in train_data.columns if 'unemployment' in col.lower() and 'lag' not in col]),
            "lag_features": len([col for col in train_data.columns if 'lag' in col]),
            "moving_averages": len([col for col in train_data.columns if 'ma' in col]),
            "economic_indicators": len([col for col in train_data.columns if any(x in col.lower() for x in ['cpi', 'gdp', 'lci']) and 'change' not in col]),
            "economic_changes": len([col for col in train_data.columns if 'change' in col])
        }
    }
    
    # Export summary
    import json
    with open(output_dir / "feature_summary.json", 'w') as f:
        json.dump(feature_summary, f, indent=2)
    
    print(f"✅ Exported 3 datasets and summary to {output_dir}")
    return feature_summary

def main():
    """Run simplified feature engineering pipeline"""
    print("🚀 Simple Unemployment Feature Engineering Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        df = load_integrated_data()
        
        # Step 2: Handle missing data
        df_clean = handle_missing_data(df)
        
        # Step 3: Create essential features
        df_features = create_essential_features(df_clean)
        
        # Step 4: Identify regional targets
        regional_targets = create_regional_targets(df_features)
        
        if not regional_targets:
            raise ValueError("No usable regional unemployment data found")
        
        # Step 5: Create temporal splits
        train_data, val_data, test_data = temporal_split(df_features)
        
        # Step 6: Validate model readiness
        is_ready = validate_model_readiness(train_data, val_data, test_data, regional_targets)
        
        if not is_ready:
            print("⚠️ Data quality issues detected - proceed with caution")
        
        # Step 7: Export datasets
        summary = export_model_ready_data(train_data, val_data, test_data, regional_targets)
        
        print("\n" + "=" * 60)
        print("🎯 Feature Engineering Complete!")
        print(f"✅ {summary['total_features']} features for {len(regional_targets)} regions")
        print(f"✅ Ready for ARIMA, LSTM, and ensemble model training")
        print("📁 Datasets exported to model_ready_data/")
        
        return summary
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    main()