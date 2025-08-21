# NZ Unemployment Forecasting Pipeline - Execution Order

**Team JRKI Capstone Project**  
**Complete End-to-End Pipeline for Government-Quality Unemployment Forecasting**

---

## 🚀 **COMPLETE PIPELINE EXECUTION ORDER**

### **Prerequisites:**
- Python 3.8+ installed
- Required packages: `pip install pandas numpy scikit-learn statsmodels tensorflow`
- Raw Stats NZ CSV files in root directory

---

## **📋 STEP-BY-STEP EXECUTION**

### **Step 1: Data Cleaning & Integration**
```bash
cd D:\Claude\Capstone
python comprehensive_data_cleaner.py
```

**What it does:**
- Cleans 9 Stats NZ CSV files
- Removes data contamination (population data)
- Handles missing data with forward fill
- Creates audit trails and quality metrics

**Output:**
- `data_cleaned/` folder with 9 cleaned CSV files
- `data_cleaned/audit_log.json`
- `data_cleaned/data_quality_metrics.json`

**Expected runtime:** 2-3 minutes

---

### **Step 2: Time Series Integration**
```bash
python time_series_aligner_simplified.py
```

**What it does:**
- Combines 9 cleaned datasets into single time series
- Aligns quarterly data from 1914-2025
- Creates integrated forecasting dataset

**Output:**
- `data_cleaned/integrated_forecasting_dataset.csv` (445 records, 185+ variables)
- `data_cleaned/integration_metrics.json`

**Expected runtime:** 1-2 minutes

---

### **Step 3: Feature Engineering**
```bash
python simple_unemployment_features.py
```

**What it does:**
- Creates lag features, moving averages, economic indicators
- Handles missing data imputation
- Splits data into train/validation/test sets
- Validates dataset schema and content

**Output:**
- `model_ready_data/train_data.csv` (311 records, 1914-1991)
- `model_ready_data/validation_data.csv` (66 records, 1992-2008)
- `model_ready_data/test_data.csv` (68 records, 2008-2025)
- `model_ready_data/feature_summary.json`

**Expected runtime:** 1-2 minutes

---

### **Step 4: Model Training & Forecasting** ⭐ **MAIN EVENT**
```bash
python unemployment_model_trainer.py
```

**What it does:**
- Trains 4 model types (ARIMA, LSTM, Random Forest, Gradient Boosting)
- Evaluates performance across Auckland, Wellington, Canterbury
- Generates 8-period unemployment forecasts
- Creates comprehensive performance reports

**Output:**
- `models/` directory with 15 trained models (.pkl files)
- `models/model_evaluation_report.json`
- `models/feature_importance.json`
- `models/training_summary.json`
- `models/unemployment_forecasts.json` (dashboard-ready)

**Expected runtime:** 5-8 minutes (includes neural network training)

---

## 🎯 **QUICK START - FULL PIPELINE**

If all files are in place, run the complete pipeline:

```bash
cd D:\Claude\Capstone

# Complete pipeline execution
python comprehensive_data_cleaner.py && ^
python time_series_aligner_simplified.py && ^
python simple_unemployment_features.py && ^
python unemployment_model_trainer.py
```

**Total runtime:** ~10-15 minutes  
**Final output:** Ready-to-use unemployment forecasting models with dashboard integration

---

## 📊 **VERIFICATION CHECKLIST**

After each step, verify these files exist:

### **After Step 1 (Data Cleaning):**
- ✅ `data_cleaned/cleaned_*.csv` (9 files)
- ✅ `data_cleaned/audit_log.json`

### **After Step 2 (Integration):**
- ✅ `data_cleaned/integrated_forecasting_dataset.csv`

### **After Step 3 (Feature Engineering):**
- ✅ `model_ready_data/train_data.csv`
- ✅ `model_ready_data/validation_data.csv`
- ✅ `model_ready_data/test_data.csv`

### **After Step 4 (Model Training):**
- ✅ `models/*.pkl` (15 model files)
- ✅ `models/unemployment_forecasts.json`

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues:**

1. **TensorFlow not available**
   ```bash
   pip install tensorflow
   ```
   *Note: LSTM models will be skipped if TensorFlow unavailable (not critical)*

2. **Missing CSV files**
   - Ensure all raw Stats NZ files are in root directory
   - Check file names match expected patterns

3. **Memory issues**
   - Close other applications
   - Use 64-bit Python if available

4. **Permission errors**
   - Run PowerShell as Administrator
   - Check folder write permissions

### **Success Indicators:**

✅ **Data Cleaning Success:**
```
✅ Saved 9 cleaned CSV files
✅ Generated audit trails and quality metrics
```

✅ **Integration Success:**
```
✅ Created integrated dataset with 445 records
✅ All temporal alignment successful
```

✅ **Feature Engineering Success:**
```
✅ Created 153 features across 3 datasets
✅ Validation passed for all target columns
```

✅ **Model Training Success:**
```
🏆 BEST MODELS BY REGION:
• Auckland: gradient_boosting (MAE: 0.933)
• Wellington: gradient_boosting (MAE: 2.138)  
• Canterbury: gradient_boosting (MAE: 1.067)

✅ Ready for dashboard integration and MBIE presentation!
```

---

## 📈 **FINAL DELIVERABLES READY FOR USE**

### **For MBIE Presentation:**
- **`models/unemployment_forecasts.json`** - 8-quarter unemployment predictions
- **`models/training_summary.json`** - Executive summary with performance metrics
- **`models/model_evaluation_report.json`** - Detailed model comparison

### **For Dashboard Integration:**
- **JSON forecast format** - Power BI compatible
- **Performance metrics** - Model accuracy for visualization
- **Regional breakdowns** - Auckland/Wellington/Canterbury comparisons

### **For Academic Submission:**
- **Complete documentation** - SUMMARY.md, DOCUMENTATION.md
- **All model types** - ARIMA, LSTM, Random Forest, Gradient Boosting
- **Validation framework** - MAE, RMSE, MAPE metrics
- **Audit trails** - Complete process documentation

---

## ⚡ **EXPRESS EXECUTION** 

For immediate results (if you've already run the pipeline before):

```bash
cd D:\Claude\Capstone
python unemployment_model_trainer.py
```

This will use existing `model_ready_data/` and generate fresh forecasts in ~5-8 minutes.

---

**🎉 Your unemployment forecasting system is now ready for government presentation and policy decision-making!**

**Contact:** Team JRKI - NZ Unemployment Forecasting Capstone Project  
**Client:** Dr. Trang Do (TEC) → MBIE  
**Status:** ✅ Production Ready