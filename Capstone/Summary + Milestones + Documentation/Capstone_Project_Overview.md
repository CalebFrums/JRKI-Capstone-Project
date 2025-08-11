
# Capstone Project Overview: Analysing New Zealand's Unemployment Rate

## 1. Comprehensive Summary

This project, proposed by Team JRKI, will conduct a comprehensive analysis of New Zealand's unemployment landscape. The project has been enhanced with additional datasets to move beyond simple trend reporting and explore the correlations between unemployment and key economic indicators.

The analysis will follow a strategic, layered approach:
1.  **Macro-Level Analysis:** Establish baseline relationships between national unemployment and key economic indicators like GDP, CPI (inflation), and the Labour Cost Index (wage growth).
2.  **Regional-Level Analysis:** Drill down to compare regional unemployment with regional economic performance to identify local variations and areas of concern.
3.  **Demographic & Sectoral Deep Dive:** Use the economic context from the previous layers to analyze which specific demographic groups (by age, sex, ethnicity) and which industries/occupations are most affected by the broader economic trends.
4.  **Predictive Forecasting:** Implement machine learning models to forecast future unemployment trends based on the discovered correlations.

The expected deliverables include a comprehensive report, a final presentation, and an interactive dashboard for data exploration.

## 2. Data Sources

The analysis will be based on the following 10 datasets from Statistics New Zealand:

-   **Unemployment Data (Quarterly):**
    -   `Age Group Regional Council.csv`
    -   `Ethnic Group Regional Council.csv`
    -   `Sex Age Group.csv`
    -   `Sex Regional Council.csv`
-   **Economic Indicators:**
    -   `GDP All Industries.csv`: Gross Domestic Product by industry and region (Annual).
    -   `CPI All Groups.csv`: National Consumer Price Index (Quarterly).
    -   `CPI Regional All Groups.csv`: Regional Consumer Price Index (Quarterly).
    -   `LCI All sectors and Industry Group.csv`: Labour Cost Index by industry (Quarterly).
    -   `LCI All Sectors and Occupation Group.csv`: Labour Cost Index by occupation (Quarterly).
-   **Population Data:**
    -   `DPE Estimated Resident Population by Age and Sex.csv`: National population estimates (Quarterly).

## 3. Project Milestones and Timeline

Here is a detailed 12-week project plan broken into four key milestones, now including specific tasks for visualization, dashboard development, and machine learning.

### Milestone 1: Data Foundation & Initial Exploration (Weeks 1-3)
*The goal of this phase is to clean all data, establish a baseline ML model, and create foundational charts.* 

-   **Task 1.1: Data Cleaning & Preprocessing:**
    -   Develop a robust script to process all 10 CSV files, handling complex headers, missing values, and different data frequencies.
    -   Aggregate quarterly data to annual averages where necessary to align with the annual GDP data.
-   **Task 1.2 (Charts V1): Foundational Visualizations:**
    -   Create initial time-series plots for national unemployment, CPI, and LCI.
    -   Generate bar charts comparing unemployment across regions and histograms for key variables.
-   **Task 1.3 (ML M1): Baseline Forecasting Model:**
    -   Prepare the national unemployment time-series data for machine learning (e.g., checking for stationarity).
    -   Split data into training and testing sets.
    -   Build and evaluate a simple baseline forecasting model (e.g., ARIMA) to establish a performance benchmark.

### Milestone 2: Macro, Regional & Dashboard Prototyping (Weeks 4-6)
*The goal of this phase is to analyze broad trends and build a functional dashboard prototype.* 

-   **Task 2.1: National & Regional Trend Analysis:**
    -   Correlate national/regional unemployment with GDP, CPI, and LCI data.
    -   Identify key regions of interest (outliers, high unemployment, etc.) that warrant a deeper dive.
-   **Task 2.2 (Charts V2 part 1): Correlation Visuals:**
    -   Develop scatter plots to visualize the correlation between annual GDP growth and unemployment for each region.
-   **Task 2.3 (Dashboard D1): Dashboard Scaffolding:**
    -   Set up a dashboard application (e.g., using Streamlit or Dash).
    -   Populate the dashboard with the foundational charts from Milestone 1.
    -   Implement core interactive widgets for filtering by Region and Year.

### Milestone 3: Deep Dive Analysis & Advanced ML (Weeks 7-9)
*The goal of this phase is to uncover detailed demographic insights and build advanced predictive models.* 

-   **Task 3.1: Demographic & Sectoral Deep Dive:**
    -   For the regions of interest, create detailed unemployment breakdowns by age, sex, and ethnicity.
    -   Correlate these with regional GDP by industry and LCI by occupation to understand *who* is affected by *what* economic changes.
-   **Task 3.2 (Charts V2 part 2): Deep Dive Visuals:**
    -   Create grouped bar charts and heatmaps to visualize the detailed demographic and sectoral findings.
-   **Task 3.3 (ML M2): Advanced Multivariate Forecasting:**
    -   Develop a more advanced model (e.g., VAR, XGBoost) that uses CPI, LCI, and GDP as features to improve unemployment forecast accuracy.
    -   Apply the best model to forecast unemployment for the key regions of interest.

### Milestone 4: Synthesis & Final Deliverables (Weeks 10-12)
*The goal of this phase is to consolidate all findings and produce the final, polished project outputs.* 

-   **Task 4.1: Synthesize Findings & Develop Recommendations:**
    -   Combine all insights into a cohesive narrative and formulate data-driven recommendations.
-   **Task 4.2 (Dashboard D2): Finalize Dashboard:**
    -   Integrate the advanced charts from Milestone 3 into the dashboard.
    -   Add summary statistics, key findings, and the ML forecast visualizations.
    -   Refine the layout and user experience for the final deliverable.
-   **Task 4.3: Create Final Report & Presentation:**
    -   Write the comprehensive report and create the final presentation slides, using outputs from the dashboard and analysis.

## 4. Getting Started

### Prerequisites
-   Python 3.x
-   pandas, scikit-learn, statsmodels, matplotlib, seaborn, streamlit/dash

### Installation & Usage
1.  **Place all CSV files** in the `Capstone/data_raw` directory.
2.  **Run the data cleaning script:**
    ```bash
    python scripts/clean_data.py
    ```
3.  The cleaned, analysis-ready data will be saved in the `Capstone/data_cleaned` directory.
4.  Open and run the analysis notebooks in order (e.g., `notebooks/1_EDA.ipynb`).

### Proposed Folder Structure
```
/Capstone
|-- /data_raw/ (All 10 original CSV files)
|-- /data_cleaned/ (Output of cleaning script)
|-- /scripts/
|   |-- clean_data.py
|   |-- train_model.py
|-- /notebooks/
|   |-- 1_Initial_EDA.ipynb
|   |-- 2_Macro_Regional_Analysis.ipynb
|   |-- 3_Demographic_Deep_Dive.ipynb
|   |-- 4_ML_Forecasting.ipynb
|-- /dashboard/
|   |-- app.py
|-- /Summary + Milestones + Documentation/
|   |-- Capstone_Project_Overview.md
|-- Team_JRKI_Proposal.docx
|-- README.md
```
