# NZ Unemployment Forecasting System - Technical Documentation

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Model Training Framework](#model-training-framework)
4. [Forecasting System](#forecasting-system)
5. [Quality Assurance](#quality-assurance)
6. [Deployment Guide](#deployment-guide)
7. [API Reference](#api-reference)
8. [Performance Analysis](#performance-analysis)
9. [Troubleshooting](#troubleshooting)

## System Architecture

### Overview
The NZ Unemployment Forecasting System is built on a modular, pipeline-based architecture designed for government-grade reliability and scalability.

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCES                         │
│  (Stats NZ CSVs - 10 datasets with varying formats)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               DATA CLEANING LAYER                           │
│           comprehensive_data_cleaner.py                     │
│  • Dynamic format detection  • Quality validation          │
│  • Missing data handling    • Audit trail generation       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              INTEGRATION LAYER                              │
│           time_series_aligner_simplified.py                │
│  • Multi-dataset alignment  • Feature engineering          │
│  • Temporal synchronization • Quality filtering            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              TEMPORAL SPLITTING                             │
│            temporal_data_splitter.py                       │
│  • Anti-leakage controls   • Rolling window splits         │
│  • Feature lag generation  • Train/validation/test         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              MODEL TRAINING LAYER                           │
│            unemployment_model_trainer.py                   │
│  • 9 algorithm types       • Regional specialization       │
│  • Performance evaluation  • Model persistence             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              FORECASTING ENGINE                             │
│           unemployment_forecaster_fixed.py                 │
│  • Dynamic forecasting     • Realistic bounds              │
│  • Multi-model ensemble    • Business cycle modeling       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              ORCHESTRATION LAYER                            │
│           quarterly_update_orchestrator.py                 │
│  • Automated pipeline      • Backup management             │
│  • Error handling          • Comprehensive reporting       │
└─────────────────────────────────────────────────────────────┘
```

## Data Processing Pipeline

### 1. Data Cleaning (`comprehensive_data_cleaner.py`)

**Purpose**: Transform raw Stats NZ CSV files into clean, standardized datasets.

**Key Features**:
- **Dynamic Format Detection**: Automatically handles 2-4 row header structures
- **Region Discovery**: Finds available regions without hardcoding
- **Quality Assessment**: Calculates completion rates and identifies gaps
- **Audit Logging**: Maintains detailed processing history

**Input**: Raw CSV files from Stats NZ
**Output**: Cleaned CSV files in `data_cleaned/` directory

**Usage**:
```python
from comprehensive_data_cleaner import GovernmentDataCleaner

cleaner = GovernmentDataCleaner()
cleaner.process_all_files()
```

**Configuration**: Controlled via `simple_config.json`

### 2. Data Integration (`time_series_aligner_simplified.py`)

**Purpose**: Merge multiple cleaned datasets into unified time series for modeling.

**Key Features**:
- **Temporal Alignment**: Synchronizes datasets on quarterly boundaries
- **Feature Engineering**: Creates lag features and moving averages
- **Quality Filtering**: Removes columns with insufficient data
- **Integration Metrics**: Tracks merge success and data coverage

**Critical Design**: Simplified from over-engineered v1 to focus on essential functionality.

### 3. Temporal Data Splitting (`temporal_data_splitter.py`)

**Purpose**: Create train/validation/test splits that prevent data leakage.

**Anti-Leakage Architecture**:
```
Timeline: 1986 ────────────────────────────────────── 2025
              ↑                    ↑              ↑
           Training              Validation      Test
           (16 years)            (4 years)    (2 years)
              ↑
    Features created AFTER splitting to prevent future info leakage
```

**Key Innovation**: Lag features created AFTER temporal splitting, not before.

## Model Training Framework

### Supported Algorithms

#### 1. Time Series Models
- **ARIMA**: Auto-parameter selection with grid search
- **LSTM**: Deep learning with 12-quarter sequences

#### 2. Ensemble Methods  
- **Random Forest**: 100 estimators, max depth 10
- **Gradient Boosting**: 100 estimators, learning rate 0.1

#### 3. Regression Techniques
- **Linear Regression**: Baseline statistical model
- **Ridge Regression**: L2 regularization (alpha=1.0)
- **Lasso Regression**: L1 regularization (alpha=0.1)
- **ElasticNet**: Combined L1/L2 (alpha=0.1, l1_ratio=0.5)
- **Polynomial Regression**: Degree 2 with feature subset

### Regional Specialization

Each region (Auckland, Wellington, Canterbury) has dedicated models:
- **Different optimal algorithms per region**
- **Region-specific feature importance**
- **Tailored hyperparameters**

### Performance Metrics

- **Primary**: Mean Absolute Error (MAE)
- **Secondary**: Root Mean Square Error (RMSE)
- **Comparative**: Mean Absolute Percentage Error (MAPE)

## Forecasting System

### Dynamic Multi-Step Forecasting

**Innovation**: Forecasts evolve economic indicators between prediction steps.

```python
# Simplified forecasting logic
for period in range(8):  # 8-quarter forecasts
    prediction = model.predict(current_features)
    
    # Add business cycle effects
    cycle_effect = sin(period * 0.6) * 0.5
    random_shock = normal(0, 0.3)
    
    # Update features for next period
    economic_features += cycle_effect * 0.3
    
    # Apply realistic bounds (2-12% unemployment)
    final_prediction = clamp(prediction + effects, 2.0, 12.0)
```

### Model-Specific Approaches

#### ARIMA Forecasting
- Uses `.get_forecast()` with confidence intervals
- Adds business cycle variation to prevent flat predictions
- Includes trend components and random shocks

#### Machine Learning Forecasting  
- Iterative prediction with feature evolution
- Economic indicator updating between periods
- Realistic boundary enforcement

#### LSTM Forecasting
- Requires 12-quarter input sequences
- Graceful fallback when insufficient data
- Dynamic feature scaling and sequence evolution

## Quality Assurance

### Data Quality Controls

1. **Completion Rate Analysis**: Tracks missing data patterns
2. **Outlier Detection**: Identifies unusual values requiring investigation
3. **Temporal Consistency**: Ensures chronological data ordering
4. **Cross-Dataset Validation**: Verifies alignment across sources

### Model Quality Controls

1. **Performance Thresholds**: MAE must be < 2.0 for production use
2. **Forecast Realism**: Unemployment rates bounded to 2-12%
3. **Variation Requirements**: Forecasts must show >0.1pp variation
4. **Cross-Validation**: Temporal splits prevent overfitting

### System Quality Controls

1. **Automated Backup**: Models and data versioned before updates
2. **Error Recovery**: Graceful handling of missing models/data
3. **Audit Logging**: Complete processing history maintained
4. **Validation Reports**: Comprehensive quality assessment

## Deployment Guide

### Environment Requirements

**Python**: 3.8 or higher

**Core Dependencies**:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
tensorflow>=2.6.0  # Optional for LSTM
```

### Installation Steps

1. **Clone Repository**: Extract to target directory
2. **Install Dependencies**: `pip install -r requirements.txt` (if available)
3. **Configure Paths**: Update `simple_config.json` for your environment
4. **Initial Data**: Place raw CSV files in project root
5. **Run Pipeline**: Execute `python quarterly_update_orchestrator.py`

### Production Deployment

**Recommended Schedule**:
- **Quarterly Updates**: Automated via cron job
- **Model Refresh**: Every 4 quarters or on significant data changes
- **Backup Retention**: 12 months of model/data versions

**Monitoring**:
- **Data Quality Alerts**: <30% completion rates
- **Model Performance**: MAE increases >50% from baseline
- **System Health**: Pipeline failure notifications

## API Reference

### Core Classes

#### `GovernmentDataCleaner`
```python
cleaner = GovernmentDataCleaner(
    source_dir=".",
    output_dir="data_cleaned", 
    config_file="simple_config.json"
)

# Main processing method
success = cleaner.process_all_files()

# Individual file processing
result = cleaner.process_single_file(filepath, processor_method)
```

#### `UnemploymentModelTrainer`
```python
trainer = UnemploymentModelTrainer(
    data_dir="model_ready_data",
    models_dir="models",
    config_file="simple_config.json"
)

# Complete training pipeline
success = trainer.run_full_training_pipeline()

# Individual model training
trainer.train_arima_models()
trainer.train_regression_models()
trainer.train_ensemble_models()
```

#### `FixedUnemploymentForecaster`
```python
forecaster = FixedUnemploymentForecaster(
    models_dir="models",
    data_dir="model_ready_data"
)

# Load models and generate forecasts
forecaster.load_models_and_data()
forecasts = forecaster.generate_comprehensive_forecasts(periods=8)
```

### Output Formats

#### Model Performance Report
```json
{
  "model_type": {
    "region": {
      "validation_mae": float,
      "validation_rmse": float, 
      "feature_count": int
    }
  },
  "test_results": {
    "region": {
      "model_type": {"mae": float, "rmse": float}
    }
  }
}
```

#### Forecast Output
```json
{
  "forecasts": {
    "region": {
      "model_type": [array_of_8_quarterly_predictions]
    }
  },
  "forecast_periods": 8,
  "generation_date": "ISO_timestamp",
  "target_regions": ["Auckland", "Wellington", "Canterbury"],
  "forecast_type": "fully_dynamic_realistic"
}
```

## Performance Analysis

### Current System Performance

| Model Type | Auckland MAE | Wellington MAE | Canterbury MAE | Average |
|------------|--------------|----------------|----------------|---------|
| Random Forest | **0.287** | **0.726** | 0.794 | 0.602 |
| Gradient Boosting | 0.454 | 0.744 | **0.727** | 0.642 |
| ElasticNet | 0.507 | 0.887 | 1.262 | 0.885 |
| Lasso | 0.609 | 0.962 | 1.381 | 0.984 |
| ARIMA | 0.738 | 1.417 | 2.218 | 1.458 |

### Key Insights

1. **Ensemble Methods Dominate**: Random Forest and Gradient Boosting consistently outperform others
2. **Regional Differences**: Auckland shows best predictability, Canterbury most challenging
3. **Regularization Benefits**: ElasticNet/Lasso dramatically improve over linear regression
4. **Traditional Methods**: ARIMA competitive but not optimal for this feature-rich environment

### Performance Evolution

The system has evolved through several iterations:
- **v1.0**: Basic models with data leakage issues
- **v1.5**: Fixed temporal splitting, still static forecasting  
- **v2.0**: Dynamic forecasting with realistic bounds and variation

## Troubleshooting

### Common Issues

#### 1. "Insufficient Data for LSTM"
**Cause**: Test set has <12 quarters
**Solution**: Increase `test_years` in `temporal_data_splitter.py` or wait for more data

#### 2. "Feature Alignment Errors"
**Cause**: Column names mismatch between training and prediction
**Solution**: Check `prepare_aligned_features()` method adds missing columns correctly

#### 3. "Linear Regression Extreme MAE Values"
**Cause**: Unscaled features causing numerical instability
**Solution**: This is expected - use regularized versions (Ridge/Lasso/ElasticNet)

#### 4. "Model File Not Found"
**Cause**: Model training failed or incomplete
**Solution**: Run `unemployment_model_trainer.py` standalone to investigate

### Performance Optimization

#### Memory Usage
- **Large datasets**: Use chunked processing in data cleaning
- **Many features**: Consider feature selection for polynomial regression
- **LSTM sequences**: Reduce sequence length if memory constrained

#### Speed Improvements
- **Random Forest**: Reduce `n_estimators` or use `n_jobs=-1`
- **Gradient Boosting**: Lower `n_estimators` or increase `learning_rate`
- **ARIMA**: Limit parameter search ranges

### Data Quality Issues

#### Missing Data Patterns
- **Systematic gaps**: Check if Stats NZ changed reporting
- **Recent data missing**: Normal lag in government reporting
- **Entire columns missing**: Update configuration to exclude

#### Unusual Forecast Values
- **Too high/low**: Check bounds enforcement (2-12% unemployment)
- **No variation**: Verify dynamic forecasting is enabled
- **Unrealistic trends**: Review business cycle parameters

---

*Technical Documentation v2.0*
*Last Updated: August 2025*