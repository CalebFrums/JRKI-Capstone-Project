# New Zealand Unemployment Forecasting System

A comprehensive machine learning system for forecasting unemployment rates across New Zealand's key regions. Designed for government presentation standards and production deployment.

## Overview

This system provides multi-algorithm unemployment forecasting capabilities for Auckland, Wellington, and Canterbury regions using advanced machine learning techniques and comprehensive economic indicators.

### Key Features

- **Multi-Algorithm Approach**: ARIMA, LSTM, Random Forest, Gradient Boosting, and Regression models
- **Regional Specificity**: Dedicated models for Auckland, Wellington, and Canterbury
- **Government-Grade Data Processing**: Robust handling of Stats NZ datasets
- **Automated Pipeline**: Quarterly update orchestration with backup and validation
- **Production Ready**: Dashboard-compatible JSON outputs

## Performance Results

| Region | Best Model | Validation MAE | Test Performance |
|--------|------------|----------------|------------------|
| Auckland | Random Forest | 0.287 | 0.318 |
| Wellington | Random Forest | 0.726 | 1.074 |
| Canterbury | Gradient Boosting | 0.727 | 0.432 |

## System Architecture

```
Raw Data (Stats NZ) → Data Cleaning → Integration → Temporal Splitting → Model Training → Forecasting
                                                                               ↓
                    Quarterly Updates ← Validation ← Dashboard Outputs ← JSON Generation
```

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, tensorflow, statsmodels

### Running the Complete System

```bash
# Run complete quarterly update pipeline
python quarterly_update_orchestrator.py

# Or run individual components:
python comprehensive_data_cleaner.py      # Data cleaning
python time_series_aligner_simplified.py  # Data integration  
python temporal_data_splitter.py          # Train/test splitting
python unemployment_model_trainer.py      # Model training
python unemployment_forecaster_fixed.py   # Generate forecasts
```

### Key Outputs

- **Models**: Trained model files in `models/` directory
- **Forecasts**: `models/fixed_unemployment_forecasts.json`
- **Performance**: `models/model_evaluation_report.json`
- **Audit Trail**: `data_cleaned/audit_log.json`

## Data Sources

The system processes 9 key datasets from Stats NZ:

1. Age Group Regional Council unemployment data
2. Sex Age Group unemployment statistics  
3. Ethnic Group Regional Council data
4. Sex Regional Council unemployment rates
5. Consumer Price Index (CPI) - All Groups
6. CPI Regional data
7. GDP by industry and region
8. Labour Cost Index (LCI) - Sectors and Occupations
9. LCI by Industry Groups

## Model Types

### Time Series Models

- **ARIMA**: Classical statistical forecasting with automatic parameter selection
- **LSTM**: Deep learning neural networks for sequence prediction

### Machine Learning Models  

- **Random Forest**: Ensemble method with feature importance analysis
- **Gradient Boosting**: Advanced ensemble with regularization
- **Linear Regression**: Baseline statistical model
- **Ridge Regression**: L2 regularized linear model
- **Lasso Regression**: L1 regularized with feature selection
- **ElasticNet**: Combined L1/L2 regularization
- **Polynomial Regression**: Non-linear relationships modeling

## Configuration

System behavior is controlled through `simple_config.json`:

```json
{
  "regions": {
    "unemployment_core": ["Auckland", "Wellington", "Canterbury"]
  },
  "demographics": {
    "age_groups_basic": ["15-24 Years", "25-54 Years", "55+ Years", "Total All Ages"]
  }
}
```

## Quality Assurance

- **Data Validation**: Comprehensive quality checks and missing data analysis
- **Temporal Splitting**: Prevents data leakage with proper chronological splits
- **Model Evaluation**: Cross-validation with multiple performance metrics
- **Forecast Validation**: Realistic bounds checking (2-12% unemployment range)
- **Automated Backup**: Version control for models and data

## Production Features

### Automated Updates

- Quarterly data refresh pipeline
- Automatic model retraining
- Performance monitoring and alerts
- Comprehensive audit logging

### Error Handling

- Graceful degradation for missing data
- Fallback forecasting for model failures
- Detailed error reporting and recovery

### Integration Ready

- JSON output format for dashboards
- Power BI compatible data structures
- RESTful API integration capabilities

## Team and Governance

**Development Team**: Data Science Team
**Client**: Dr. Trang Do, Tertiary Education Commission
**Target Audience**: Ministry of Business, Innovation and Employment (MBIE)
**Version**: Production v2.0

## Security and Compliance

- Government data handling protocols
- No cloud AI dependencies
- Secure file processing
- Audit trail maintenance
- Data lineage documentation

## Support and Maintenance

For technical support or deployment assistance:

- Review `DOCUMENTATION.md` for detailed technical specifications
- Check `Summary + Milestones + Documentation/` for implementation history
- Refer to model evaluation reports for performance analysis

## License and Usage

This system is designed for New Zealand government use. All data processing
complies with Stats NZ guidelines and government security requirements.

---

*Last Updated: August 2025*
*System Status: Production Ready*
