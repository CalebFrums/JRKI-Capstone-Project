# Data Cleaning and Preprocessing Requirements
## Unemployment Forecasting Dashboard Project - Team JRKI

### Project Overview
This document outlines the data cleaning and preprocessing requirements for the unemployment forecasting dashboard project with Dr. Trang Do from the Tertiary Education Commission, intended for presentation to the Ministry of Business Innovation and Employment (MBIE).

---

## 1. Data Source and Security Requirements

### 1.1 Data Source
- **Primary Source**: Ministry of Business Innovation and Employment (MBIE)
- **Data Type**: Official New Zealand unemployment statistics
- **Data Provider**: Dr. Trang Do (Project Client)
- **Historical Scope**: Multiple years of historical unemployment data

### 1.2 Security Protocols
⚠️ **CRITICAL SECURITY REQUIREMENTS**
- **NO use of ChatGPT or cloud-based AI tools** for sensitive government data
- Use only approved tools: Power BI, Python, Snowflake
- Implement strict data handling protocols following government standards
- Create secure file sharing protocols using approved platforms only
- Establish data access controls and user permission levels
- Regular security compliance reviews and team training

---

## 2. Known Data Quality Issues

### 2.1 Expected Challenges
Based on client feedback, expect:
- **Extensive data cleaning required**
- **Missing data points** throughout the dataset
- Potential inconsistencies in data format
- Need for significant preprocessing time allocation

### 2.2 Resource Allocation
- **Allocate 40% of project timeline** specifically for data cleaning activities
- Plan for iterative cleaning process with client feedback loops

---

## 3. Data Preprocessing Requirements

### 3.1 Initial Data Assessment
```python
# Week 6 Milestone: Early data assessment with Dr. Trang
- Conduct comprehensive data exploration
- Document all data quality issues
- Identify missing data patterns
- Assess data completeness by time period and demographics
```

### 3.2 Data Validation Protocols
- Establish automated data quality checks using Python scripts
- Implement data validation during initial exploration
- Create data cleaning documentation and standard procedures
- Develop backup data sources identification plan

### 3.3 Missing Data Handling
- **Identify missing data patterns**: Systematic vs random
- **Implement appropriate imputation strategies**:
  - Time series imputation for temporal gaps
  - Demographic-specific imputation where applicable
  - Document all imputation decisions and rationale
- **Validate imputation quality** before proceeding

### 3.4 Outlier Detection and Treatment
- Implement robust outlier detection methods
- Consider economic context (e.g., economic crises, policy changes)
- Document outlier treatment decisions
- Maintain transparency in data modifications

---

## 4. Data Structure Requirements

### 4.1 Temporal Considerations
- Ensure consistent time intervals (monthly/quarterly data)
- Handle seasonal patterns appropriately
- Maintain chronological order for time series analysis
- Account for potential reporting delays in official statistics

### 4.2 Demographic Breakdown Support
**Client Requirement**: "Do you want to compare different demographics? Yes, it will be very helpful"
- Structure data to support demographic comparisons
- Ensure consistent demographic categories
- Handle demographic classification changes over time
- Enable filtering by demographic groups in dashboard

### 4.3 Economic Factors Integration
**Client Note**: "The factors are interconnected"
- Prepare data structure for multiple economic indicators
- Ensure compatibility with interconnected factor analysis
- Support for additional variables as scope evolves

---

## 5. Machine Learning Preparation

### 5.1 Multi-Algorithm Support
Prepare data for multiple forecasting approaches:
- **Time Series Models**: ARIMA/SARIMA format requirements
- **Neural Networks**: LSTM input structure and normalization
- **Ensemble Methods**: Random Forest, Gradient Boosting feature preparation
- **Regression Techniques**: Linear regression, polynomial, regularized methods

### 5.2 Feature Engineering
- Create lag features for time series models
- Generate seasonal decomposition components
- Calculate moving averages and trend indicators
- Prepare demographic interaction features

### 5.3 Data Splitting Strategy
- Maintain temporal order in train/validation/test splits
- Ensure sufficient historical data for model training
- Plan for out-of-time validation testing
- Reserve recent data for dashboard validation

---

## 6. Dashboard Integration Requirements

### 6.1 Live Update Capability
- Structure data pipeline for automated updates
- Implement data refresh mechanisms compatible with Power BI
- Design error handling for data feed interruptions
- Create manual backup update procedures

### 6.2 Performance Optimization
- Optimize data structure for dashboard loading speed
- Implement data aggregation where appropriate
- Ensure 5-second comprehension standard compliance
- Minimize data transfer overhead

---

## 7. Documentation Requirements

### 7.1 Data Lineage Documentation
- Document all data transformations
- Track data source to final output pipeline
- Maintain change log for all preprocessing decisions
- Create reproducible data cleaning scripts

### 7.2 Quality Assurance Documentation
- Record data quality metrics before and after cleaning
- Document validation procedures and results
- Maintain audit trail for compliance requirements
- Create data dictionary for all processed variables

---

## 8. Implementation Timeline

### 8.1 Week 6: Data Collection/Cleaning
- Receive data from Dr. Trang Do
- Conduct initial data assessment
- Implement basic cleaning procedures
- Document preliminary findings

### 8.2 Week 7: Model Preparation
- Complete data preprocessing for ML models
- Finalize feature engineering
- Validate data quality for forecasting
- Prepare data splits for model training

### 8.3 Ongoing: Iterative Refinement
- Fortnightly client feedback incorporation
- Continuous data quality monitoring
- Refinement based on model performance
- Dashboard integration testing

---

## 9. Risk Mitigation

### 9.1 Data Quality Contingencies
- **High Risk Identified**: Data Quality Issues (5,3 impact)
- Prepare synthetic data generation for testing
- Identify alternative data sources through client
- Plan scope reduction if data quality severely compromised

### 9.2 Communication Protocols
- Immediate notification to Dr. Trang for major data issues
- Weekly progress reports on cleaning status
- Escalation procedures for timeline impacts
- Documentation of all client consultations

---

## 10. Success Criteria

### 10.1 Data Quality Metrics
- **Missing data**: Reduced to <5% where possible
- **Outliers**: Appropriately identified and treated
- **Consistency**: Uniform format and structure
- **Completeness**: Sufficient historical coverage for forecasting

### 10.2 Functional Requirements
- **Accuracy**: Support client's accuracy requirements across demographics
- **Performance**: Enable dashboard 5-second comprehension standard
- **Reliability**: Support automated live updates
- **Security**: Maintain government data security standards

### 10.3 Client Satisfaction
- **Prototype approval**: Early validation with Dr. Trang
- **Ministry readiness**: Suitable for MBIE presentation
- **Stakeholder usability**: Policy maker friendly output

---

## Contact Information
- **Project Client**: Dr. Trang Do (dothutrang81@yahoo.com, trangdtt@gmail.com)
- **Project Manager**: Robert McDougall
- **Data Lead**: Justin Regidor
- **Project Advisor**: Anjali de Silva

---

*This document should be reviewed and updated based on actual data receipt and Dr. Trang Do's specific requirements during the Week 6 data assessment meeting.*