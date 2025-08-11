# Capstone Project Overview: Analysing New Zealand's Unemployment Rate

## 1. Comprehensive Summary

This project, proposed by Team JRKI, will conduct a comprehensive analysis of New Zealand's unemployment landscape. The project has been enhanced with additional datasets to move beyond simple trend reporting and explore the correlations between unemployment and key economic indicators.

The analysis will follow a strategic, layered approach:
1.  **Macro-Level Analysis:** Establish baseline relationships between national unemployment and key economic indicators like GDP, CPI (inflation), and the Labour Cost Index (wage growth).
2.  **Regional-Level Analysis:** Drill down to compare regional unemployment with regional economic performance to identify local variations and areas of concern.
3.  **Demographic & Sectoral Deep Dive:** Use the economic context from the previous layers to analyze which specific demographic groups (by age, sex, ethnicity) and which industries/occupations are most affected by the broader economic trends.

The expected deliverables remain a comprehensive report, a set of key findings with actionable recommendations, and a final presentation, all to be completed within the 12-week timeline.

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

Here is a detailed 12-week project plan broken into four key milestones.

### Milestone 1: Data Foundation & Initial Exploration (Weeks 1-3)
*The goal of this phase is to clean, process, and understand each dataset individually.* 

-   **Task 1.1: Data Cleaning & Preprocessing:**
    -   Develop a robust script to process all 10 CSV files. This script must handle:
        -   Multiple header rows and metadata footers.
        -   Non-numeric values (`..`, `C`, etc.) and missing data.
        -   Pivoting data from wide to a long, tidy format.
        -   Standardizing column names (e.g., `Date`, `Region`) across all datasets.
-   **Task 1.2: Data Alignment & Aggregation:**
    -   Aggregate the quarterly unemployment data into annual averages to allow for direct comparison with the annual GDP data.
    -   Create a master, cleaned data folder to store the processed files.
-   **Task 1.3: Initial Exploratory Data Analysis (EDA):**
    -   For each cleaned dataset, generate summary statistics and initial visualizations to understand its distribution, time range, and basic trends.

### Milestone 2: Macro & Regional Analysis (Weeks 4-6)
*The goal of this phase is to analyze broad national and regional trends to build context.* 

-   **Task 2.1: National Trend Analysis:**
    -   Correlate the aggregated national unemployment rate with annual GDP growth, quarterly CPI, and quarterly LCI (wage growth).
    -   Visualize these relationships to identify long-term patterns.
-   **Task 2.2: Regional Analysis:**
    -   Compare regional unemployment rates against regional CPI data.
    -   Analyze annual regional unemployment against annual regional GDP by industry to identify which industries are linked to unemployment changes in specific regions.
    -   Identify regions of interest (e.g., those with the highest unemployment, or those that deviate most from national trends).

### Milestone 3: Demographic & Sectoral Deep Dive (Weeks 7-9)
*The goal of this phase is to use the context from Milestone 2 to understand who is most affected.* 

-   **Task 3.1: Demographic Analysis:**
    -   For the regions of interest identified in Task 2.2, perform a detailed analysis of the unemployment data.
    -   Create breakdowns by age, sex, and ethnicity to pinpoint the specific demographic groups most impacted by the regional economic conditions.
-   **Task 3.2: Industry & Occupation Analysis:**
    -   Analyze the Labour Cost Index data by both Industry and Occupation group against unemployment trends.
    -   Synthesize these findings with the GDP-by-industry data to build a narrative around which sectors and job types are experiencing the most economic stress or growth.

### Milestone 4: Synthesis & Final Deliverables (Weeks 10-12)
*The goal of this phase is to consolidate all findings and produce the final project outputs.* 

-   **Task 4.1: Synthesize Findings & Develop Recommendations:**
    -   Combine the insights from all previous milestones into a single, cohesive narrative.
    -   Develop a set of clear, data-driven key findings and formulate actionable recommendations for policymakers.
-   **Task 4.2: Create Final Report & Presentation:**
    -   Write the comprehensive report detailing the methodology, analysis, findings, and recommendations.
    -   Create the final presentation slides to summarize the project for stakeholders.

## 4. Getting Started

### Prerequisites
-   Python 3.x
-   pandas library (`pip install pandas`)

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
|-- /notebooks/
|   |-- 1_Initial_EDA.ipynb
|   |-- 2_Macro_Regional_Analysis.ipynb
|   |-- 3_Demographic_Deep_Dive.ipynb
|-- /Summary + Milestones + Documentation/
|   |-- Capstone_Project_Overview.md
|-- Team_JRKI_Proposal.docx
|-- README.md
```