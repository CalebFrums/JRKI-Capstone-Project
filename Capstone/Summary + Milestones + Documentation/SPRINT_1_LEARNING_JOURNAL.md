# My Learning Journal - Sprint 1
## Unemployment Forecasting Project

**Week:** 1  
**Date:** Today  
**What I'm trying to do:** Get the basic data validation script working

---

## What I'm Learning

### My First Big Mistake (and how I fixed it)

So I was trying to get the Python script to work and kept getting this error about matplotlib not being found. I remembered that in R you can just do `install.packages("something")` right in your code, so I tried putting the pip install commands directly in the Python file.

**That was totally wrong!** 

Here's what I learned about the difference:
In R, you can install packages right in your script:
```r
install.packages("ggplot2")  # This just works
library(ggplot2)             # Then use it
```

But Python is different! You have to:
```python
# First, go to command prompt and do:
pip install pandas

# THEN in your Python script:
import pandas as pd
```

**Why is Python like this?** Basically, R was built for statisticians who want everything in one place, but Python was built to be more modular. So package installation happens outside of Python, then you import them into your scripts.

**How I fixed it:**
1. Removed all the pip install lines from my Python file (whoops!)
2. Opened command prompt and ran: `pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy`
3. Now my script should work when I import the packages

Honestly, this was pretty confusing at first because I'm used to R, but I think I get it now.

---

## What I'm hoping to accomplish in Sprint 1

I need to prove that this unemployment forecasting project can actually work. So my script should:
- Load all the CSV files without crashing
- Show me some basic correlation between inflation (CPI) and unemployment
- Make some simple plots to visualize the data
- Give me the green light that "yes, this approach makes sense"

If I can get the script to run and show me that correlation, then I know I'm on the right track for the bigger analysis.

---

## About the data I'm working with

I found out that my unemployment CSV files are actually from something called the "Household Labour Force Survey" (HLFS) which is the official NZ unemployment data. That's pretty cool - means I'm working with the real deal that the government uses.

The files I have are:
- **Sex Age Group.csv** - unemployment by gender and age (goes back to 1986!)
- **Age Group Regional Council.csv** - unemployment by age in different regions
- **Ethnic Group Regional Council.csv** - unemployment by ethnicity and region  
- **Sex Regional Council.csv** - unemployment by gender and region

Plus I have economic data like inflation (CPI), GDP, and wage data (LCI).

The tricky part is that these files have really complex headers - like multiple rows of headers that are kind of messy. So my script needs to be smart about parsing them.

---

## My thoughts so far

This is way more complicated than I thought it would be! In my econometrics classes we usually get clean datasets, but real government data is messy and has weird formatting.

I'm learning that data science is like 80% getting the data to work properly and 20% actual analysis. The Python vs R thing was just the first hurdle.

But I'm excited to see if there's actually a correlation between inflation and unemployment in NZ. My econ theory says there should be some relationship, but let's see what the data shows!

---

## Next steps

1. **Run the script** and see what happens
2. **Look at the results** - is there a correlation? How strong?
3. **Check the plots** - do they make sense economically?
4. **Document any other issues** I run into
5. **Plan what to focus on** for Sprint 2

If this works, I'll be ready to move on to the full data cleaning pipeline. If it doesn't work... well, more debugging!

**Fingers crossed!** 🤞

---

---

## SPRINT 1 RESULTS! ✅

**IT WORKED!** The script ran successfully and generated the four-panel visualization. Here's what I learned from the results:

### What the plots showed me:

**Top Left - Auckland Unemployment (2010-2026):**
- Clear downward trend from 2010-2020 (8% down to 3% - economic recovery!)
- Massive spike around 2020-2021 hitting 6.5% (definitely COVID impact)
- Recent uptick in 2024-2025 back to ~6% (current economic pressures)
- The data looks clean and follows what I'd expect economically

**Top Right - National CPI (1920-2020):**
- Holy cow, this goes back to 1920! Nearly 100 years of inflation data
- Exponential growth pattern, especially crazy acceleration after 1980s
- Shows CPI climbing from ~200 to >1200 - that's serious inflation over time
- This historical depth is actually really valuable

**Bottom Left - CPI vs Unemployment Correlation:**
- Shows negative correlation (higher inflation → lower unemployment)
- Pretty scattered though, so clearly other factors matter too
- This actually matches Phillips Curve theory from my macro econ class!
- The "interconnected factors" Dr. Trang mentioned are definitely visible

**Bottom Right - ARIMA Forecast:**
- **This is where I have problems...** 
- My ARIMA forecast is way too flat and conservative (predicting ~3.7-3.8%)
- It completely missed the recent unemployment spike
- The actual data (blue line) is much more volatile than my orange forecast line

### Key realizations:

✅ **Data pipeline works** - dates parsed correctly, missing data handled  
✅ **Economic relationships are detectable** - correlation analysis successful  
⚠️ **ARIMA is too simplistic** - need more sophisticated models for accuracy  
❌ **Only showing Auckland** - Dr. Trang needs ALL regions for MBIE!

### Major issue I need to fix:

**YOU'RE ABSOLUTELY RIGHT** - this is only Auckland data! For the Ministry presentation, Dr. Trang will need analysis across all New Zealand regions, not just Auckland. My script is too narrow in scope.

The ARIMA model is also clearly not sophisticated enough to capture the volatility and recent economic shocks. I'll need those ensemble methods and additional economic indicators you mentioned.

### What this means for Sprint 2:

1. **Expand to ALL regions** - not just Auckland focus
2. **Include more economic indicators** - GDP, LCI data integration
3. **Improve forecasting models** - ensemble methods, multivariate approaches
4. **Better handle economic shocks** - COVID, current inflation pressures

### Personal reflection:

This was actually really exciting! Seeing the correlation between inflation and unemployment play out in real NZ data was cool. But it also showed me how much more sophisticated this analysis needs to be for a government presentation.

The fact that my simple ARIMA missed the recent economic volatility is a good reminder that real-world forecasting is way harder than textbook examples.

---

**Updated status:** Sprint 1 complete - data works, but scope needs expansion!  
**Biggest insight:** Simple models miss complex economic reality  
**Next priority:** Expand from Auckland-only to all-regions analysis  
**Mood:** Accomplished but aware of work ahead! 📊

**Claude said the following:**
● 🚨 Agent Assessment Results

  The specialized agents have provided a comprehensive audit, and unfortunately there are significant gaps between what was accomplished and what's needed. Here's the summary:

  CRITICAL ISSUES IDENTIFIED:

  ❌ 1. Regional Coverage Gap

  - Reality: Only Auckland analyzed
  - Required: ALL New Zealand regions for MBIE policy-making
  - Impact: Fundamentally insufficient scope for government presentation

  ❌ 2. Missing Demographic Analysis

  - Reality: No age/ethnicity/sex breakdowns
  - Required: Dr. Trang explicitly requested demographic comparisons
  - Impact: Core client requirement completely absent

  ❌ 3. Economic Integration Deficiency

  - Reality: Only CPI correlation
  - Required: "Interconnected factors" - GDP, LCI, wages
  - Impact: Oversimplified economic model

  ⚠️ IMPORTANT GAPS:

  4. Data Quality Standards

  - Current: Basic pandas processing (dropna, to_numeric)
  - Required: Government-standard data cleaning with audit trails

  5. Forecasting Inadequacy

  - Your own assessment: "ARIMA too flat and conservative"
  - Evidence: Missed recent unemployment volatility completely

  6. Security Compliance

  - Missing: Audit trails, data lineage documentation
  - Required: Government data handling protocols

  🎯 REALITY CHECK:

  What You Actually Accomplished:
  ✅ Proved the basic data pipeline works✅ Demonstrated Auckland unemployment can be analyzed✅ Showed CPI correlation exists✅ Generated initial visualizations

  What Still Needs To Be Done:
  ❌ Expand to ALL regions (not just Auckland)❌ Add demographic breakdowns (age, ethnicity, sex)❌ Integrate all economic indicators (GDP, LCI)❌ Implement proper government-standard data cleaning❌ Improve forecasting accuracy significantly

  📋 Recommendation:

  Sprint 1 Status: FOUNDATION ESTABLISHED but significant expansion needed for Sprint 2 to meet MBIE requirements.

  You've proven the approach works, but the scope needs to grow from "Auckland proof-of-concept" to "National policy analysis tool."

  Ready to plan Sprint 2 with this fuller picture? 🚀