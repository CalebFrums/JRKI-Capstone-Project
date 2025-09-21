# JRKI Unemployment Forecasting Pipeline - GUI Usage Guide

## Overview

The JRKI Unemployment Forecasting Pipeline is a comprehensive system for processing New Zealand unemployment data and generating machine learning-based forecasts. This GUI provides an intuitive interface for running the complete forecasting pipeline.

## System Requirements

### Required Software

- **Python 3.8+** with the following packages:
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - pathlib
  - tkinter (usually included with Python)
  - **Optional**: statsmodels (for ARIMA models)

### Installation

```bash
pip install pandas numpy scikit-learn joblib
pip install statsmodels  # Optional, for ARIMA support
```

## Directory Structure

### Required Setup
You only need to create the input directory manually:

```
D:\Capstone\
├── JRKI_Forecasting_Pipeline.exe        # Main GUI application
├── simple_config.json                   # Configuration file
├── comprehensive_data_cleaner.py        # Data cleaning script
├── time_series_aligner_simplified.py    # Time alignment script
├── temporal_data_splitter.py            # Data splitting script
├── unemployment_model_trainer.py        # Model training script
├── unemployment_forecaster_fixed.py     # Forecasting script
└── raw_datasets/                        # [REQUIRED] Input data directory
```

### Auto-Created Directories
The following directories are created automatically by the pipeline:

```
├── data_cleaned/                        # Created by Data Cleaning step
├── model_ready_data/                    # Created by Data Splitting step
└── models/                              # Created by Model Training step
    └── evaluation_csvs/                  # Created for CSV exports
```

## Required Input Data

### Folder: `raw_datasets/`

The system accepts CSV files from Stats NZ with specific naming patterns:

#### **Accepted File Types:**

**HLF (Household Labour Force Survey) Files:**

- `HLF Labour force status by age group region council quarterly.csv`
- `HLF Labour force status by ethnic group by regional council quarterly.csv`
- `HLF Labour Force Status by Sex by Age Group quarterly.csv`
- `HLF Labour Force status by Sex by regional council.csv`
- `HLF Labour Force status by sex by total resp ethnic group quarterly.csv`
- `HLF labour force status by sex sing or comb ethnic group.csv`

**Economic Indicator Files:**

- `GDP All Industries.csv`
- `CPI All Groups.csv`, `CPI Regional All Groups.csv`
- `LCI All sectors and Industry Group.csv`, `LCI All Sectors and Occupation Group.csv`

**Monthly Employment Indicators (MEI):**

- `MEI Age and Region by variable monthly.csv`
- `MEI high level industry by variable monthly.csv`
- `MEI Industry by variable monthly.csv`
- `MEI Sex and Age by Variable monthly.csv`
- `MEI Sex and Region by Variable Monthly.csv`

**Electronic Card Transactions (ECT):**

- `ECT electronic card transactions by industry group monthly.csv`
- `ECT means and proportion monthly.csv`
- `ECT Number of electronic card transactions A_S_T by division monthly.csv`
- `ECT Total Values electronic card transactions A_S_T by division monthly.csv`
- `ECT Totals electronic card transaction by division percentage changes monthly.csv`

**Quarterly Employment Survey (QEM):**

- `QEM average hourly earnings by industry and sex quarterly.csv`
- `QEM average hourly earnings by sector and sex percentage change quarterly.csv`
- `QEM Average Hourly Earnings by Sector and Sex quarterly.csv`
- `QEM Filled Jobs by Industry by sex and status in employment quarterly.csv`
- `QEM filled jobs by sector by sex and status in employment quarterly.csv`

**Business Use of ICT (BUO) - Annual:**

- `BUO ICT Annual.csv`
- `BUO Totals - Business Operations Annual.csv`
- `BUO Totals innovation annual.csv`

#### **Rejected File Types:**

- Non-CSV files
- Files without proper Stats NZ headers
- Files missing required date columns
- Corrupted or empty CSV files
- Files with incompatible encoding

## GUI Usage Instructions

### Starting the Application

1. Navigate to the project directory: `D:\Claude\Capstone\`
2. Run the GUI: `python jrki_gui.py`
3. The application window will open showing the pipeline control panel

### Pipeline Steps

The GUI provides 5 main pipeline steps that should be run in sequence:

#### **1. Data Cleaning**

- **Purpose**: Clean and standardize raw CSV files
- **Input**: Files in `raw_datasets/` folder
- **Output**: Cleaned files in `data_cleaned/` folder
- **What it does**:
  - Handles missing values
  - Standardizes column names
  - Removes outliers
  - Validates data quality

#### **2. Time Alignment**

- **Purpose**: Align different data frequencies (monthly, quarterly, annual)
- **Input**: Files from `data_cleaned/` folder
- **Output**: Temporally aligned dataset
- **What it does**:
  - Synchronizes data to quarterly frequency
  - Interpolates missing time periods
  - Creates consistent time series

#### **3. Data Splitting**

- **Purpose**: Split data into training, validation, and test sets
- **Input**: Aligned time series data
- **Output**: Files in `model_ready_data/` folder:
  - `train_data.csv`
  - `validation_data.csv`
  - `test_data.csv`
  - `feature_summary.json`
- **What it does**:
  - Temporal splitting (prevents data leakage)
  - Feature engineering (lag features, moving averages)
  - Quality validation

#### **4. Model Training**

- **Purpose**: Train machine learning models for unemployment forecasting
- **Input**: Split datasets from `model_ready_data/`
- **Output**: Trained models in `models/` folder:
  - `.joblib` model files
  - `model_evaluation_report.json`
  - `training_summary.json`
  - `feature_importance.json`
- **What it does**:
  - Trains Random Forest models for each region/demographic
  - Applies sparsity-aware model selection
  - Generates performance metrics

#### **5. Forecasting**

- **Purpose**: Generate unemployment rate predictions
- **Input**: Trained models from `models/` folder
- **Output**: Forecast files in `models/` folder:
  - `unemployment_forecasts.json`
  - `unemployment_forecasts_powerbi.csv`
  - `forecast_summary_powerbi.csv`
- **What it does**:
  - Generates 8-quarter forecasts
  - Creates confidence intervals
  - Formats output for Power BI integration

### Running the Pipeline

#### **Individual Steps:**

1. Click **[RUN]** next to each step in sequence
2. Monitor progress in the console output window
3. Wait for each step to complete before proceeding

#### **Full Pipeline:**

1. Click **[START] Run Full Pipeline** to execute all steps automatically
2. Use **[STOP]** to halt execution if needed
3. Use **[CLEAR] Logs** to clear the console output

### Additional Tools

#### **View Results:**

- Click **[VIEW] Results** to open the output folders
- Access forecast files, model performance reports, and logs

## Troubleshooting

### Common Issues

**"No such file or directory" errors:**

- Ensure all CSV files are in the `raw_datasets/` folder
- Check that folder names match exactly (case-sensitive)

**"Insufficient data" warnings:**

- Some demographic/regional combinations may have sparse data
- The system will automatically skip these combinations

**Memory issues:**

- Large datasets may require more RAM
- Consider processing fewer files at once

**Import errors:**

- Install missing Python packages: `pip install package_name`
- For ARIMA support: `pip install statsmodels`

### Support Files

**Configuration:**

- `simple_config.json`: Contains regional, demographic, and processing settings
- Modify thresholds and parameters as needed

**Logs:**

- All processing logs appear in the GUI console
- Use **[CLEAR] Logs** to reset the display

## Output Files

### Final Deliverables

1. **Power BI Ready Files:**
   - `unemployment_forecasts_powerbi.csv` - Detailed forecasts by region/demographic
   - `forecast_summary_powerbi.csv` - Summary statistics and KPIs

2. **Technical Reports:**
   - `model_evaluation_report.json` - Model performance metrics
   - `training_summary.json` - Training process summary
   - `feature_importance.json` - Feature importance rankings

3. **Model Files:**
   - `*.joblib` files - Trained Random Forest models for each target
   - Ready for production forecasting

## Government/MBIE Presentation

The system generates production-ready outputs suitable for:

- Ministry of Business, Innovation & Employment (MBIE) reporting
- Stats NZ integration
- Power BI dashboard creation
- Quarterly unemployment forecasting updates

---

**Version:** Production v2.2
**Team:** JRKI


