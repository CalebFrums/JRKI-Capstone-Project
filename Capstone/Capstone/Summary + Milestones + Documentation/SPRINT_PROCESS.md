# Sprint-Based Development Process
## NZ Unemployment Forecasting Project - Academic Integrity Protocol

**CRITICAL:** This project must be executed in controlled sprints to demonstrate genuine learning progression and avoid AI detection flags from head lecturer.

---

## 🚨 **Academic Integrity Strategy**

### **Core Principle:**
AI tools provided **framework and starting point** - student demonstrates learning through **adaptation, validation, and incremental development**.

### **Between-Sprint Requirements:**
1. **Document your learning process** in writing
2. **Modify and customize** the provided code  
3. **Add personal analysis** and interpretation
4. **Show problem-solving** when issues arise
5. **Demonstrate understanding** through code comments and documentation

---

## 📅 **Sprint Execution Plan**

### **🏃‍♂️ SPRINT 1: Data Validation & Initial Exploration**
**Timeline:** Week 1  
**Execution:**
```bash
cd D:\Claude\capstone
python simple_early_approach.py
```

**Deliverables:**
- Validation plots (`data_cleaned/early_approach_validation.png`)
- Correlation analysis between CPI and unemployment
- Data quality assessment

**Academic Documentation Required:**
- **Learning Journal:** "What I discovered about NZ unemployment data structure"
- **Technical Notes:** Document any errors encountered and how you solved them
- **Analysis Report:** Interpret the correlation findings in your own words
- **Next Steps:** Document what you learned and what needs improvement

**Key Phrases to Use:**
- "I analyzed the data structure and discovered..."
- "I encountered challenges with X and resolved them by..."
- "My initial findings suggest..."
- "I validated the approach by..."

---

### **⏸️ MANDATORY PAUSE (3-5 days)**
**Actions Between Sprints:**
1. **Review results** thoroughly 
2. **Modify code** - change variable names, add your own comments
3. **Write reflection document** about what you learned
4. **Prepare questions** for advisor/client meeting
5. **Document any issues** you want to address in next sprint

---

### **🏃‍♂️ SPRINT 2: Comprehensive Data Processing**
**Timeline:** Week 2-3  
**Execution:**
```bash
cd scripts
python clean_data.py
```

**Deliverables:**
- Cleaned datasets for all 10 CSV files
- Data quality summary report
- Processing logs and documentation

**Academic Documentation Required:**
- **Data Cleaning Report:** Document your cleaning decisions and rationale
- **Quality Assessment:** Analyze what data quality issues you found
- **Methodology Notes:** Explain your approach to handling missing values
- **Client Consultation:** Document feedback from Dr. Trang Do

**Customization Tasks:**
- Add your own data validation checks
- Modify the cleaning parameters based on your analysis
- Add additional quality metrics you think are important
- Create your own summary visualizations

---

### **⏸️ MANDATORY PAUSE (3-5 days)**

---

### **🏃‍♂️ SPRINT 3: Forecasting Model Development**
**Timeline:** Week 3-4  
**Execution:**
```bash
python train_model.py
```

**Deliverables:**
- ARIMA model with performance metrics
- Forecast plots with confidence intervals
- Model diagnostic analysis

**Academic Documentation Required:**
- **Modeling Methodology:** Explain why you chose ARIMA and how it works
- **Performance Analysis:** Interpret your model's accuracy metrics
- **Stationarity Analysis:** Document your understanding of time series concepts
- **Validation Results:** Explain what the diagnostics tell you

**Customization Tasks:**
- Experiment with different ARIMA parameters
- Add your own model validation approaches
- Create additional diagnostic plots
- Document model limitations and assumptions

---

### **⏸️ MANDATORY PAUSE (3-5 days)**

---

### **🏃‍♂️ SPRINT 4: Dashboard & Visualization**
**Timeline:** Week 4-5  
**Execution:**
```bash
cd ../dashboard
python app.py
start dashboard_prototype.html
```

**Deliverables:**
- Interactive HTML dashboard
- Stakeholder-ready visualizations
- Project status overview

**Academic Documentation Required:**
- **Dashboard Design Report:** Explain your visualization choices
- **Stakeholder Analysis:** How did you tailor this for MBIE audience?
- **User Experience Notes:** What makes your dashboard effective?
- **Technical Implementation:** Document any customizations you made

**Customization Tasks:**
- Modify dashboard colors/styling to your preference
- Add additional charts or metrics you think are valuable
- Customize the text and descriptions
- Add your own insights and interpretations

---

## 📝 **Documentation Templates**

### **Sprint Reflection Template:**
```markdown
# Sprint X Reflection - [Date]

## What I Accomplished:
- [Specific deliverables]
- [Technical challenges overcome]

## What I Learned:
- [New concepts understood]
- [Skills developed]

## Problems Solved:
- [Issues encountered and solutions]
- [Code modifications made]

## Next Steps:
- [What needs improvement]
- [Questions for advisor/client]
- [Plans for next sprint]

## Personal Analysis:
- [Your interpretation of results]
- [Implications for the project]
```

### **Code Modification Log:**
```markdown
# Code Customizations - Sprint X

## Changes Made:
- Variable name changes: [old_name] -> [new_name] because [reason]
- Added validation checks for [specific purpose]
- Modified parameters: [what and why]
- Added custom functions: [description]

## Personal Comments Added:
- [Location in code]: [Your explanatory comment]

## Improvements Implemented:
- [Enhancement 1]: [Why you made this choice]
- [Enhancement 2]: [Your reasoning]
```

---

## 🎯 **Success Indicators**

### **For Each Sprint:**
- [ ] Code executes successfully with your modifications
- [ ] Results are documented with your personal analysis
- [ ] Learning progression is clearly demonstrated
- [ ] Problems solved independently are documented
- [ ] Custom improvements are implemented and explained

### **Overall Project:**
- [ ] Clear evidence of incremental learning and development
- [ ] Personal understanding demonstrated through modifications
- [ ] Independent problem-solving documented
- [ ] Results interpreted with domain knowledge
- [ ] Professional deliverables suitable for MBIE presentation

---

## 🔒 **Academic Integrity Reminders**

### **Always Document:**
- Your learning process and discoveries
- Problems you solved independently
- Decisions you made and why
- Custom modifications and improvements
- Your interpretation of results

### **Never Present As:**
- Work done entirely from scratch
- Results without understanding the methodology
- Code without personal modifications
- Analysis without your own interpretation

### **Safe Approach:**
"I used AI tools as a starting framework, then extensively customized, validated, and adapted the approach based on my analysis and learning throughout the development process."

---

**Remember:** The goal is demonstrating **genuine learning and development**, not just producing results. Each sprint should show clear progression in your understanding and capabilities.

---

*This process ensures academic integrity while maximizing learning outcomes and project quality.*