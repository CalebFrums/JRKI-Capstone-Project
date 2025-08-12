# My Learning Journal - Sprint 1 
## Unemployment Forecasting Project - My First Week

**Week:** August 11th - 16th  
**Started:** Monday morning, way too optimistic  
**What I'm trying to do:** Get some basic Python script working to prove this project isn't impossible

---

## What I'm Learning

### Day 1-2: Complete Disaster With Package Installation 

**Monday morning (Aug 11):** Thought I'd knock out a quick data validation script. How hard could it be, right?

**Monday afternoon:** Spent THREE HOURS getting error messages like:
```
ModuleNotFoundError: No module named 'matplotlib'
ModuleNotFoundError: No module named 'pandas' 
```

I was losing my mind! In R (which I actually know), you just put `install.packages("whatever")` right in your script and it works. So I tried this in Python:

```python
pip install pandas  # I put this right in my .py file
import pandas as pd
```

**WRONG!** Python just gave me syntax errors. Apparently you can't run pip commands inside Python scripts?

**Tuesday morning breakthrough (Aug 12):** Finally figured out you have to install packages OUTSIDE of Python first, then import them. Spent an hour on Stack Overflow reading about virtual environments and package management. 

The correct way:
1. Open command prompt (not Python)  
2. Run: `pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy`
3. THEN write your Python script with just import statements

**Tuesday afternoon:** Still confused about why Python makes this so complicated compared to R. Asked my roommate (CS major) and he laughed at me. Apparently this is "basic Python 101" but nobody told me that in my stats classes.

**Lesson learned:** Python and R are completely different animals. I need to stop assuming they work the same way.

---

## Day 3-4: Trying to Figure Out What I'm Actually Supposed to Do

**Wednesday (Aug 13):** OK so now Python works, but what am I supposed to build exactly? Dr. Trang wants "unemployment forecasting" but that could mean a million different things.

**My basic plan (probably naive):**
- Get the CSV files to load without Python exploding
- Try to find some connection between unemployment and inflation (Phillips Curve from macro econ?)
- Make some plots that don't look like garbage
- Hopefully prove this project isn't completely impossible

**Wednesday evening:** Honestly not sure if I'm on the right track. The assignment brief is pretty vague about Sprint 1. I think I'm supposed to just do a "proof of concept" but what does that even mean?

**Thursday (Aug 14):** Decided to just start with Auckland unemployment data and see if I can make ANY progress. If I can get one region working, maybe I can expand later.

---

## Day 4-5: The Data Files Are... Complicated

**Thursday evening (Aug 14):** Looked at the CSV files Dr. Trang sent us. Holy crap, these are NOT like the clean datasets from our assignments.

**What I discovered:**
The unemployment data comes from something called "Household Labour Force Survey" (HLFS). Apparently this is the official government source, which is cool - we're using the same data that policy makers use.

**The files I'm dealing with:**
- Sex Age Group.csv - unemployment by gender and age (goes back to 1986! That's older than I am)
- Age Group Regional Council.csv - unemployment by age in different regions  
- Ethnic Group Regional Council.csv - unemployment by ethnicity and region
- Sex Regional Council.csv - unemployment by gender and region

Plus economic indicators: CPI (inflation), GDP, LCI (wage data)

**Friday morning problem (Aug 15):** These headers are absolutely insane! Instead of nice column names like "unemployment_rate", I get stuff like:

```
"Labour Force Status by Age Group by Region Council (Qrtly-Mar/Jun/Sep/Dec)","","","",""
"","Aged 15-24 Years","","","Aged 25-54 Years"
" ","Auckland","Wellington","Canterbury","Auckland"
```

Who designed this?! It's like they WANT it to be impossible to analyze.

**Friday afternoon:** Spent 2 hours just trying to figure out which column is which. The headers span multiple rows and half of them are empty strings. This is going to be way harder than I thought.

---

## Friday Evening: Reality Check

**What I thought this would be like:** Load some CSVs, run a few regressions, make some nice plots. Easy.

**What it's actually like:** Spending entire days just trying to get the data to load properly. The headers are nightmare fuel and the formatting is completely inconsistent across files.

**Biggest realization:** Our professors give us clean, pre-processed datasets for assignments. Real government data is completely different - it's like they export it directly from some ancient database without any thought for human beings who might want to analyze it.

**Current mood:** Frustrated but determined. I can see why Dr. Trang said data cleaning would take most of our time. 

**One positive thing:** If I can get this working with Auckland data, it might actually show the Phillips Curve relationship I learned about on Youtube. That would be pretty cool to see in real NZ data.

---

## Weekend: Finally Getting Somewhere

**Saturday (Aug 16):** Decided to just focus on getting SOMETHING working instead of trying to solve everything perfectly. Used `skiprows=3` to get past the header mess and manually picked the Auckland unemployment column.

**Saturday evening:** Breakthrough! Got a basic script running that:
1. Loads the unemployment data (Auckland only for now)
2. Loads the CPI data 
3. Does some basic correlation analysis
4. Creates plots

**Sunday morning (Aug 17):** 🎉 IT ACTUALLY WORKED! 

---

## What My Results Look Like (And What I Learned)

**The Good News:** My script ran without crashing and made some plots that actually make sense!

**Top Left - Auckland Unemployment Over Time:**
- Goes from about 8% in 2010 down to 3% around 2019 (economic recovery after GFC)
- HUGE spike to 6.5% in 2020-2021 - obviously COVID!
- Recent uptick back to ~6% in 2024-2025 
- The trend actually makes economic sense, which is reassuring

**Top Right - National CPI Since 1920:**
- Whoa, this data goes back 100+ years! 
- Exponential growth pattern, especially after the 1980s
- CPI went from ~200 to over 1200 - that's massive inflation over time
- Pretty cool to see a century of NZ economic history

**Bottom Left - The Phillips Curve in Action:**
- Negative correlation between CPI and unemployment (higher inflation = lower unemployment)
- It's scattered but the relationship is visible
- This is literally the Phillips Curve from my macro class! Seeing it in real data is actually exciting
- Makes me think Dr. Trang is right about "interconnected factors"

**Bottom Right - My ARIMA Forecast (The Problem Child):**
- My forecast is way too flat and boring (predicting ~3.7-3.8%)
- Completely missed the recent unemployment volatility
- The actual data bounces around way more than my orange forecast line
- Clearly ARIMA isn't sophisticated enough for this

---

## Sunday Evening (Aug 17): What I Actually Accomplished (And What I'm Worried About)

**What worked:**
✅ Basic data pipeline doesn't crash  
✅ Can load unemployment and CPI data  
✅ Phillips Curve correlation is visible in real NZ data  
✅ Generated plots that make economic sense  

**What I'm concerned about:**
❌ This is only Auckland - Dr. Trang needs ALL of New Zealand for MBIE  
❌ My ARIMA forecast is terrible (too simple and flat)  
❌ Only using 2 indicators when there should be many more  
❌ No demographic breakdowns (age, ethnicity, sex) like Dr. Trang requested  

### Major Problem I Just Realized:

I got excited about getting Auckland unemployment working, but this is way too narrow for a government presentation. The Ministry of Business Innovation & Employment isn't going to make national policy based on one city's data!

Dr. Trang specifically asked about demographic comparisons ("Yes, it will be very helpful") and I have none of that. I need unemployment by age groups, by ethnicity, by regions across ALL of New Zealand.

### What I Need to Figure Out for Sprint 2:

1. **Expand beyond Auckland** - Process ALL regional councils  
2. **Add demographic analysis** - Age, sex, ethnic breakdowns  
3. **Integrate more economic data** - GDP, wages (LCI), not just CPI  
4. **Better forecasting models** - ARIMA clearly isn't good enough  
5. **Proper data cleaning** - My `skiprows=3` hack won't work for government standards  

### Honest Assessment:

This was a good proof-of-concept that showed the approach can work, but I'm only about 20% of the way to what Dr. Trang actually needs for MBIE. Sprint 2 is going to be WAY more work than I thought.

**Current mood:** Cautiously optimistic but also slightly panicked about how much more there is to do.

**Biggest lesson:** Getting something working is just the first step. Real government analysis requires way more comprehensive coverage than I initially realized.

---

**Sprint 1 Status:** ✅ Foundation working, but major expansion needed  
**Next Week Priority:** Figure out how to process ALL the CSV files properly  

**Sleep schedule:** Probably going to suffer in Sprint 2...
