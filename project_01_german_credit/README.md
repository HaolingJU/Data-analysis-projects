# German Credit Risk Analysis

This project analyzes the German Credit dataset to identify key drivers of credit risk and
to provide interpretable, business-oriented insights for credit policy design. The analysis
focuses on understanding who is more likely to default and how banks can proactively reduce
bad debt risk using transparent analytical methods.

---

## Project Objectives

The project aims to answer the following business questions:

1. Which customers are most likely to be high-risk borrowers?
2. How can banks proactively reduce bad debt risk?
3. What loan and borrower characteristics drive credit risk?

---

## Dataset

- Source: German Credit Dataset
- Size: 1,000 loan applicants
- Target variable: Credit risk classification (good vs bad)

---

## Tools & Technologies

- **Programming Language**: Python  
- **Data Analysis**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **Modeling**: Scikit-learn  
- **Environment**: Jupyter Notebook  
- **Version Control**: Git & GitHub  

---

## Key Analytical Findings

- Approximately **30% of borrowers are classified as bad credit**, indicating moderate class
  imbalance.
- **Loan duration** is the strongest controllable risk factor, with default rates increasing
  consistently as loan terms become longer.
- **Credit amount** exhibits threshold effects, where very large loans show elevated default
  risk.
- **Loan purpose** significantly differentiates risk, with business and education loans
  displaying higher default rates.
- **Interaction effects** reveal that long-term loans combined with high-uncertainty purposes
  represent particularly high-risk segments.

---

## Business Implications

The analysis suggests several proactive risk management strategies for banks:

- Apply stricter approval criteria or shorter maximum terms for long-duration loans.
- Introduce tiered credit limits and enhanced review for very large loan amounts.
- Use purpose-based segmentation to tailor credit policies.
- Incorporate interaction-aware rules rather than relying on single-variable thresholds.

---

