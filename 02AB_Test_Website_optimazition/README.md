# A/B Testing Analysis: New vs Old Landing Page

## Project Background
This project analyzes the results of an A/B test conducted by an e-commerce website to determine whether a new landing page leads to a higher user conversion rate compared to the old version.  
The goal is to provide a data-driven recommendation on whether the company should adopt the new page.

---

## Dataset Overview
- Total records: **294,478**
- Columns:
  - `user_id`
  - `timestamp`
  - `group` (control / treatment)
  - `landing_page` (old_page / new_page)
  - `converted` (0 = not converted, 1 = converted)

Each row represents a user visit and whether a conversion occurred.

---

## Methodology
- Data cleaning:
  - Removed logically inconsistent records
  - Removed duplicate users to ensure independence
- Metric of interest: **conversion rate**
- Statistical method: **Two-sample Z-test for proportions**
- Significance level: **α = 0.05**
- Test type: **Right-tailed hypothesis test**

---

## Key Result
- Z-score: **2.148**
- P-value: **0.0158**

Since the p-value is smaller than the significance level (0.05), we reject the null hypothesis.

### ✅ Conclusion
**The new landing page leads to a statistically significant increase in conversion rate compared to the old page.**

---

## Repository Structure
- `ab_test_analysis.ipynb` – Full data analysis and statistical testing
- `analysis.md` – Detailed explanation of data cleaning and hypothesis testing
- `ab_data.csv` – Original dataset

---

## Tools & Libraries
- Python
- pandas
- numpy
- scipy
- matplotlib / seaborn

