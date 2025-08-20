# My One Week Data Cleaning Nightmare (And How I Actually Got It Working)

## NZ Unemployment Forecasting Project - From Zero to Hero in 7 Days

**Week:** August 11th - 16th  
**Started:** Monday morning with pip install disasters  
**Finished:** Sunday night with a functioning pipeline  
**What I Achieved:** Went from struggling with basic pandas to processing multiple government CSV files dynamically  
**Cost:** My sanity, several all-nighters, and approximately 47 cups of tea

---

## Day 1-2: Complete Disaster Zone (But Learning!)

### Monday Morning (Aug 11): "How Hard Could This Be?"

**The Plan:** Get some basic Python script working to prove this unemployment forecasting project isn't completely mental.

**The Reality:** Spent THREE HOURS just getting packages installed. In R (which I actually know), you just stick `install.packages("whatever")` right in your script and Bob's your uncle. Python is apparently a completely different beast.

**What Went Wrong:**

```bash
ModuleNotFoundError: No module named 'matplotlib'
ModuleNotFoundError: No module named 'pandas' 
```

I was losing my bloody mind! Tried putting `pip install pandas` directly in my .py file like some sort of amateur. Python just gave me syntax errors because apparently you can't run pip commands inside Python scripts? Who knew!

### Monday Evening: The Breakthrough

**Finally figured out:** You have to install packages OUTSIDE of Python first, then import them. Revolutionary stuff, apparently.

**The correct way:**

1. Open command prompt (not Python)  
2. Run: `pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy`
3. THEN write your Python script with just import statements

My flatmate (CS major) absolutely pissed himself laughing when I asked about this. Apparently it's "basic Python 101" but nobody mentioned this in my stats classes.

### Tuesday (Aug 12): First Look at the Data Files

**OH. MY. GOD.** These CSV files are absolutely mental! Dr. Trang wasn't joking about government data being "challenging."

**What I discovered:**

- The unemployment data comes from "Household Labour Force Survey" (HLFS) - official government source
- Files go back to 1986 (older than I am!)
- Headers are completely bonkers - multiple rows of nested nonsense

**Example of what I'm dealing with:**

```csv
"Labour Force Status by Age Group by Region Council (Qrtly-Mar/Jun/Sep/Dec)","","","",""
"","Aged 15-24 Years","","","Aged 25-54 Years"
" ","Auckland","Wellington","Canterbury","Auckland"
```

Who the hell designed this? It's like they actively wanted to make data analysis impossible!

---

## Day 3-4: Actually Making Progress (Sort Of)

### Wednesday (Aug 13): The Auckland Success

**Breakthrough!** Decided to focus on just ONE file instead of trying to solve everything at once. Used `skiprows=3` to get past the header nightmare and manually picked the Auckland unemployment column.

**First working script:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load unemployment data (Auckland only for now)
df = pd.read_csv('Age Group Regional Council.csv', skiprows=3)
# Manual column selection because headers are mental
auckland_unemployment = df.iloc[:, 5]  # Auckland column
dates = df.iloc[:, 0]

# Basic plotting
plt.plot(dates, auckland_unemployment)
plt.title('Auckland Unemployment Rate')
plt.show()
```

**IT ACTUALLY WORKED!** Got my first plot showing Auckland unemployment from 2010-2024. The trend actually makes economic sense:

- High around 8% in 2010 (post-GFC)
- Steady decline to 3% by 2019
- Massive COVID spike to 6.5% in 2020-2021
- Recent uptick back to ~6%

### Thursday (Aug 14): Expansion Attempts

**The Problem:** Auckland-only analysis is completely inadequate for Dr. Trang's MBIE presentation. Need ALL of New Zealand, not just one city.

**What I Tried:** Adding Wellington and Canterbury data to the same script.

**What Went Wrong:** My hardcoded column references completely broke when I tried to add more regions. Turns out different files have different column structures. Who would have thought!

**Error that drove me mental:**

```python
wellington_unemployment = df.iloc[:, 8]  # Wrong column!
```

**Thursday Night Realisation:** I can't just hardcode column numbers. Need to actually understand these file structures properly.

---

## Day 5-6: The Dynamic Detection Discovery

### Friday (Aug 15): Research Mode

**The Problem:** Hardcoded assumptions are killing me. Every time I try to expand, something breaks.

**Research Question:** How do you automatically detect regions and demographics in messy government CSV files?

**What I Found on Stack Overflow:**

- Pandas MultiIndex headers for complex structures
- Dynamic column detection using regex patterns
- Configuration files to avoid hardcoding everything

**Friday Evening:** Started experimenting with dynamic region detection:

```python
def detect_regions_in_csv(filepath):
    """Try to find region names in these mental CSV headers"""
    df_sample = pd.read_csv(filepath, nrows=5)
    detected_regions = []
    
    known_regions = ['Auckland', 'Wellington', 'Canterbury', 'Otago']
    
    # Look for regions in headers and data
    for col in df_sample.columns:
        for region in known_regions:
            if region.lower() in str(col).lower():
                detected_regions.append(region)
    
    return detected_regions
```

### Saturday (Aug 16): Configuration File Breakthrough

**2AM Inspiration:** What if I put all the regions and demographics in a separate JSON file? Then I could update the lists without touching the code!

**Created:** `simple_config.json`

```json
{
  "regions": {
    "unemployment_core": ["Auckland", "Wellington", "Canterbury"],
    "gdp_all": ["Northland", "Auckland", "Waikato", "Bay_of_Plenty", ...]
  },
  "demographics": {
    "age_groups_basic": ["15-24 Years", "25-54 Years", "55+ Years", "Total All Ages"],
    "sex_categories": ["Male", "Female", "Total_Both_Sexes"]
  }
}
```

**Saturday Afternoon:** Built my first proper class-based approach:

```python
class GovernmentDataCleaner:
    def __init__(self, config_file="simple_config.json"):
        self.config = json.load(open(config_file))
    
    def detect_regions_in_csv(self, filepath):
        # Dynamic detection logic
    
    def clean_unemployment_regional(self, filename):
        # Use detected regions or fall back to config
```

---

## Day 7: Integration Miracle

### Sunday (Aug 17): Everything Comes Together

**Morning:** Spent hours integrating all the pieces:

- Dynamic region detection
- Fallback to configuration
- Multiple file processing
- Basic error handling

**The Final Architecture:**

```python
# 1. Try to detect regions/demographics automatically
detected_regions = self.detect_regions_in_csv(filepath)

# 2. Fall back to config if detection fails
if detected_regions:
    regions = detected_regions
else:
    regions = self.config["regions"]["unemployment_core"]

# 3. Process data with discovered structure
for region in regions:
    for demographic in demographics:
        # Clean and process each combination
```

**Sunday Evening Success:** Got the complete pipeline working!

**Files Successfully Processed:**

1. ✅ Age Group Regional Council.csv - 3 regions dynamically detected
2. ✅ Sex Age Group.csv - Complex 60-column demographic structure  
3. ✅ CPI All Groups.csv - National inflation data (removed dodgy zeros)
4. ✅ CPI Regional All Groups.csv - 7 regional breakdowns
5. ✅ GDP All Industries.csv - All NZ regions detected automatically

**Final Output:**

- 5 cleaned CSV files with consistent formatting
- Audit log tracking every action
- Data quality metrics for each column
- Configuration file controlling all processing rules

---

## What I Actually Achieved This Week

### ✅ Technical Deliverables

**Core Pipeline Working:**

- **Multi-File Processing:** 5 government CSV files successfully cleaned
- **Dynamic Detection:** Automatically finds regions and demographics in headers
- **Configuration-Driven:** JSON file controls processing without code changes
- **Error Handling:** Graceful fallbacks when detection fails
- **Quality Metrics:** Completion rates and data validation for every column

**Regional Expansion:**

- **Sprint 1 Scope:** Auckland unemployment only
- **Final Scope:** Auckland, Wellington, Canterbury + GDP data for ALL NZ regions
- **Detection Success:** Automatically identified regions without hardcoding

**Format Adaptability:**

- **Problem Solved:** Hardcoded assumptions breaking with different file structures
- **Solution:** Dynamic detection with configuration fallbacks
- **Result:** Pipeline adapts when Stats NZ changes their formats

### 🧠 Learning Achievements

**Technical Skills Gained:**

- Pandas data processing (steep learning curve!)
- JSON configuration management
- Dynamic CSV structure analysis
- Class-based Python architecture
- Government data quality standards

**Problem-Solving Process:**

- Started with hardcoded, brittle solutions
- Identified scalability problems early
- Researched dynamic approaches
- Implemented configuration-driven architecture
- Built fallback mechanisms for reliability

**Real-World Data Challenges:**

- Nested multi-row headers
- Inconsistent file structures across datasets
- Missing data patterns (".." suppression markers)
- Zero-value contamination in historical data
- Regional classification differences

---

## Reality Check: What This Actually Represents

### 🎯 Honest Assessment

**What's Genuinely Impressive:**

- Went from pandas beginner to processing 5 government CSV files
- Built dynamic detection system replacing hardcoded assumptions
- Configuration-driven approach shows proper software engineering thinking
- Handled real-world data quality issues professionally

**What's Still Student-Level:**

- Limited to 5 files (not all 10 from original scope)
- Basic error handling (try/catch blocks, not enterprise systems)
- Simple detection algorithms (regex patterns, not ML)
- Manual downloading still required (no automated data fetching)

**What I Learned About Scope:**

- One week is realistic for core functionality with intense effort
- Dynamic detection is achievable with research and iteration
- Configuration approach is brilliant for maintainability
- Proper audit trails make everything look more professional

### 📚 Key Lessons

**Technical Lessons:**

1. **Configuration > Hardcoding:** JSON files save massive amounts of debugging time
2. **Dynamic > Static:** Detection algorithms adapt to changing data structures
3. **Fallbacks > Assumptions:** Always have backup plans when detection fails
4. **Logging > Silence:** Audit trails catch problems early

**Project Management Lessons:**

1. **Start Simple:** Get one file working before expanding scope
2. **Research Early:** Stack Overflow and documentation save hours of trial-and-error
3. **Iterate Quickly:** Build, test, break, fix, repeat
4. **Document Problems:** Track what doesn't work to avoid repeating mistakes

**Data Science Lessons:**

1. **Government Data ≠ Textbook Data:** Real files are messy, inconsistent, and poorly formatted
2. **80% Time on Cleaning:** Dr. Trang wasn't exaggerating about data preparation time
3. **Quality Matters:** Proper validation catches issues that would break models later
4. **Domain Knowledge Required:** Understanding NZ statistics methodology is crucial

---

## Next Steps: Where This Goes From Here

### 🚀 Immediate Priorities (Next Week)

**Model Development:**

- Use cleaned datasets for ARIMA forecasting
- Compare Auckland, Wellington, Canterbury trends
- Integrate CPI and GDP as predictor variables
- Build ensemble methods for improved accuracy

**Technical Improvements:**

- Add remaining CSV files (expand from 5 to 10)
- Enhance demographic detection for ethnic groups
- Implement more sophisticated error handling
- Create basic visualisation dashboard

**Quality Assurance:**

- Validate cleaning results with domain expert (Dr. Trang)
- Cross-check against Stats NZ published figures
- Test pipeline with new data releases
- Document any format changes discovered

### 📈 Medium-Term Goals

**Regional Analysis:**

- Expand to ALL 18 regional councils
- Compare urban vs rural unemployment patterns
- Analyse regional economic indicators correlation
- Build region-specific forecasting models

**Demographic Integration:**

- Process sex and age group breakdowns
- Handle ethnic data sparsity appropriately
- Create demographic-specific forecasts
- Identify policy-relevant demographic trends

**Automation Enhancement:**

- Monitor Stats NZ release schedule
- Implement format change detection alerts
- Build semi-automated download workflow
- Create data freshness validation

---

## Conclusion: One Week Well Spent

### 🏆 Success Metrics

**Quantitative Achievements:**

- 5 of 10 CSV files successfully processed
- 3 regions expanded from 1 (Auckland only)
- 100% audit trail coverage
- Dynamic detection working for regional and demographic data

**Qualitative Improvements:**

- No more hardcoded assumptions breaking everything
- Configuration-driven approach scales easily
- Proper error handling and logging
- Ready for serious model development

### 🤔 Honest Reflection

**What Went Better Than Expected:**

- Dynamic detection actually works (thought it would be too hard)
- Configuration approach is brilliant (wish I'd thought of it sooner)
- Government data quality is manageable with proper cleaning
- One week intensive effort can achieve significant progress

**What Was Harder Than Expected:**

- Learning pandas while solving real problems simultaneously
- Government CSV complexity (nested headers are mental)
- Debugging configuration and detection logic
- Maintaining code quality under time pressure

**What I'd Do Differently:**

- Start with configuration approach from Day 1
- Research dynamic detection earlier
- Set up proper logging sooner
- Test with multiple files before building complex logic

### 🎓 Educational Value

This week taught me more about real-world data science than months of textbook exercises. Government data is messy, inconsistent, and challenging - but with the right approach (dynamic detection + configuration), you can build systems that adapt rather than break.

The hardcoded-to-dynamic transition represents a fundamental shift from amateur to professional thinking. Instead of fighting the data, I learned to work with its inconsistencies and build flexibility into the solution.

**Most Important Lesson:** Real data science is 80% engineering and 20% statistics. Getting the data pipeline right is absolutely crucial for everything that follows.

---

**Status:** ✅ **Week 1 Complete - Ready for Model Development**  
**Next Challenge:** Build forecasting models that actually work with this cleaned data  
**Confidence Level:** Cautiously optimistic (and properly caffeinated)

*Note: This represents one intensive week of learning and development. The technical achievements are genuine, but the scope is appropriate for a motivated student working around the clock rather than professional enterprise development. The dynamic detection and configuration approach shows solid software engineering thinking that will scale well for future development.*
