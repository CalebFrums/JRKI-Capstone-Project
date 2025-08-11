#!/usr/bin/env python3
"""
Simple Early Approach: NZ Unemployment Forecasting Validation
Minimal implementation to prove the data can be analyzed and forecasted
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def parse_quarter_date(date_str):
    """Convert YYYY-Q# to datetime"""
    try:
        year, quarter = str(date_str).split('Q')
        month = (int(quarter) - 1) * 3 + 1
        return pd.Timestamp(year=int(year), month=month, day=1)
    except:
        return pd.NaT

def load_unemployment_data():
    """Load and clean Auckland unemployment data"""
    print("Loading unemployment data...")
    
    df = pd.read_csv('Age Group Regional Council.csv', skiprows=1)
    
    # Extract dates and Auckland total unemployment (column 10)
    dates = df.iloc[:, 0].apply(parse_quarter_date)
    unemployment = pd.to_numeric(df.iloc[:, 10], errors='coerce')
    
    # Create clean dataset
    result = pd.DataFrame({
        'date': dates,
        'unemployment_rate': unemployment
    }).dropna().sort_values('date')
    
    print(f"✅ Loaded {len(result)} records from {result['date'].min()} to {result['date'].max()}")
    return result

def load_cpi_data():
    """Load and clean national CPI data"""
    print("Loading CPI data...")
    
    df = pd.read_csv('CPI All Groups.csv', skiprows=1)
    df.columns = ['date_str', 'cpi_value']
    
    # Clean data
    df['date'] = df['date_str'].apply(parse_quarter_date)
    df['cpi_value'] = pd.to_numeric(df['cpi_value'], errors='coerce')
    
    # Remove invalid data
    result = df.dropna(subset=['date', 'cpi_value'])
    result = result[result['cpi_value'] > 0]
    
    print(f"✅ Loaded {len(result)} CPI records")
    return result[['date', 'cpi_value']]

def simple_forecast():
    """Basic ARIMA forecasting to validate approach"""
    print("\n🔮 Testing ARIMA forecasting...")
    
    # Load unemployment data
    unemployment_data = load_unemployment_data()
    
    # Prepare time series
    ts_data = unemployment_data.set_index('date')['unemployment_rate']
    
    # Simple train/test split
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]
    
    # Fit basic ARIMA model
    model = ARIMA(train_data, order=(2,1,2))
    model_fit = model.fit()
    
    # Generate forecast
    forecast = model_fit.forecast(steps=len(test_data))
    
    # Calculate basic accuracy
    mae = np.mean(np.abs(test_data - forecast))
    
    print(f"✅ ARIMA model fitted")
    print(f"📊 Mean Absolute Error: {mae:.2f}")
    print(f"🔮 Forecasting capability validated")
    
    return model_fit, forecast, test_data

def create_basic_plots():
    """Create essential visualizations to validate approach"""
    print("\n📊 Creating validation plots...")
    
    # Load data
    unemployment_data = load_unemployment_data()
    cpi_data = load_cpi_data()
    
    # Merge for correlation
    merged = pd.merge(unemployment_data, cpi_data, on='date', how='inner')
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Unemployment time series
    axes[0,0].plot(unemployment_data['date'], unemployment_data['unemployment_rate'])
    axes[0,0].set_title('Auckland Unemployment Rate')
    axes[0,0].set_ylabel('Unemployment %')
    
    # CPI time series  
    axes[0,1].plot(cpi_data['date'], cpi_data['cpi_value'])
    axes[0,1].set_title('National CPI')
    axes[0,1].set_ylabel('CPI Value')
    
    # Correlation plot
    axes[1,0].scatter(merged['cpi_value'], merged['unemployment_rate'], alpha=0.6)
    axes[1,0].set_title('CPI vs Unemployment Correlation')
    axes[1,0].set_xlabel('CPI Value')
    axes[1,0].set_ylabel('Unemployment %')
    
    # Forecast validation
    model_fit, forecast, test_data = simple_forecast()
    axes[1,1].plot(test_data.index, test_data.values, label='Actual', marker='o')
    axes[1,1].plot(test_data.index, forecast, label='Forecast', marker='s')
    axes[1,1].set_title('ARIMA Forecast Validation')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('data_cleaned/early_approach_validation.png', dpi=150)
    plt.show()
    
    # Calculate correlation
    correlation = merged['unemployment_rate'].corr(merged['cpi_value'])
    print(f"✅ CPI-Unemployment correlation: {correlation:.3f}")
    
    print("✅ Validation plots saved to: data_cleaned/early_approach_validation.png")

def main():
    """Run complete early approach validation"""
    print("=" * 50)
    print("NZ UNEMPLOYMENT FORECASTING - EARLY APPROACH")
    print("=" * 50)
    
    try:
        # Create output directory
        Path('data_cleaned').mkdir(exist_ok=True)
        
        # Validate core capabilities
        create_basic_plots()
        
        print("\n" + "=" * 50)
        print("✅ EARLY APPROACH VALIDATION COMPLETE")
        print("=" * 50)
        print("Key findings:")
        print("- ✅ Data can be successfully parsed and cleaned")
        print("- ✅ ARIMA forecasting works on unemployment data") 
        print("- ✅ CPI-unemployment correlation can be measured")
        print("- ✅ Basic visualization pipeline functional")
        print("\n🚀 Ready to proceed with expanded analysis")
        
    except Exception as e:
        print(f"❌ Error in validation: {e}")
        print("Check that CSV files are in current directory")

if __name__ == "__main__":
    main()
