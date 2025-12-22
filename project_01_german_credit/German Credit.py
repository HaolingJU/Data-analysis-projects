"""
German Credit Analysis
- Data loading & cleaning
- Feature engineering
- EDA plots exported to figures/eda/
- modeling(Logistic Regression+Decision Tree)

Dataset file:german_credit_data.csv
"""

# Load the packages used
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. import data
DATA_PATH="german_credit_data.csv"
FIG_DIR = "figures/eda"
RESULTS_DIR = "results"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
SHOW_PLOTS = False

df = pd.read_csv(DATA_PATH)
if "unnamed:0" in df.columns:
    df = df.drop(columns=["unnamed:0"])
original_df = df.copy()

# Dataset overview and structure
print("1) dataset overview")
print("shape:",original_df.shape) # 1000 rows, 11 columns
print("\nHead:\n",original_df.head())
print("\nColumns:\n",original_df.columns)

# understand the target column
print("\nRisk value counts:\n",original_df['Risk'].value_counts())

# View categorical variables and numeric variables separately.
categorical_cols = original_df.select_dtypes(include='object').columns
numerical_cols = original_df.select_dtypes(exclude='object').columns
print(categorical_cols, numerical_cols)

# What information does each category variable contain?
for col in categorical_cols:
    print(f"{col}: {original_df[col].unique()}")
    print("-" * 50)

# 2 data cleaning
print("2) Data Cleaning")
print("-"*50)
# missing value check
print("\nMissing value:\n",original_df.isnull().sum().sort_values(ascending=False))

df_clean = original_df.copy()

# standardize colum names
df_clean.columns = (
    df_clean.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
# map target to binary: good=0,bad=1
df_clean["risk"] = df_clean["risk"].map({"good": 0, "bad": 1})
print('Risk count(binary):\n',df_clean["risk"].value_counts())

# Missing values changed to unknown
missing_cols = ["saving_accounts", "checking_account"]

for col in missing_cols:
    df_clean[col] = df_clean[col].fillna("unknown")
print('missing check:\n',df_clean[missing_cols].isnull().sum())

# Grouping by age
df_clean["age_group"] = pd.cut(
    df_clean["age"],
    bins=[18, 25, 35, 45, 55, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"]
)

df_clean["duration_group"] = pd.cut(
    df_clean["duration"],
    bins=[0, 12, 24, 36, 72],
    labels=["<=12", "13-24", "25-36", "37-72"]
)
print(df_clean.info())
print('\nMissing value after cleaning:\n',df_clean.isnull().sum().sort_values(ascending=False).head(10))


# 3.1 Calculate risk distribution
print("3) EDA")

sns.set_theme()

risk_counts = df_clean["risk"].value_counts().sort_index()
risk_labels = ["Good Credit", "Bad Credit"]

# Plot
plt.figure(figsize=(6, 4))
plt.bar(risk_labels, risk_counts.values)
plt.title("Overall Credit Risk Distribution")
plt.ylabel("Number of Customers")
plt.xlabel("Credit Risk")
plt.savefig(f"{FIG_DIR}/overall_credit_risk.png",bbox_inches='tight',dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

# Calculate bad rate
bad_rate = risk_counts[1] / risk_counts.sum()
print(f"the bad rate is {bad_rate:2%}")

# 3.2.1 bad credit rate by age group
age_risk = (
    df_clean
    .groupby("age_group",observed=True)["risk"]
    .mean()
    .reset_index()
)


plt.figure(figsize=(7, 4))
sns.barplot(
    x="age_group",
    y="risk",
    data=age_risk
)
plt.title("Bad Credit Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Bad Credit Rate")
plt.ylim(0,1)
plt.savefig(f"{FIG_DIR}/bad_rate_by_age_group.png",bbox_inches='tight',dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

# 3.2.2 bad credit rate by job category
job_risk = (
    df_clean
    .groupby("job",observed=True)["risk"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(7, 4))
sns.barplot(
    x="job",
    y="risk",
    data=job_risk
)
plt.title("Bad Credit Rate by Job")
plt.xlabel("Job")
plt.ylabel("Bad Credit Rate")
plt.ylim(0,1)
plt.savefig(f"{FIG_DIR}/bad_credit_rate_by_job.png",bbox_inches='tight',dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

#3.2.3 bad credit rate by housing
housing_risk = (
    df_clean
    .groupby("housing",observed=True)["risk"]
    .mean()
    .reset_index()
)
plt.figure(figsize=(7, 4))
sns.barplot(
    x="housing",
    y="risk",
    data=housing_risk
)
plt.title("Bad Credit Rate by Housing")
plt.xlabel("Housing")
plt.ylabel("Bad Credit Rate")
plt.ylim(0,1)
plt.savefig(f"{FIG_DIR}/bad_credit_rate_by_housing.png",bbox_inches='tight',dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

#3.2.3 bad credit rate by gender
gender_risk = (
    df_clean
    .groupby("sex",observed=True)["risk"]
    .mean()
    .reset_index()
)
plt.figure(figsize=(7, 4))
sns.barplot(
    x="sex",
    y="risk",
    data=gender_risk
)
plt.title("Bad Credit Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Bad Credit Rate")
plt.ylim(0,1)
plt.savefig(f"{FIG_DIR}/bad_credit_rate_by_gender.png",bbox_inches='tight',dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

# 3.3 loan characteristics analysis
#3.3.1 credit amount distribution by credit risk
plt.figure(figsize=(7, 4))
sns.boxplot(
    x='risk',
    y='credit_amount',
    data=df_clean
)

plt.xticks([0, 1], ["Good Credit", "Bad Credit"])
plt.title("Credit Amount Distribution by Credit Risk")
plt.xlabel("Credit Risk")
plt.ylabel("Credit Amount")
plt.savefig(f"{FIG_DIR}/bad_credit_rate_by_job.png",bbox_inches='tight',dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

#3.3.2 loan duration vs credit risk
plt.figure(figsize=(7, 4))
sns.boxplot(
    x='risk',
    y='duration',
    data=df_clean
)
plt.xticks([0, 1], ["Good Credit", "Bad Credit"])
plt.title("Loan Duration Distribution by Credit Risk")
plt.xlabel("Credit Risk")
plt.ylabel("Loan Duration (Months)")
plt.savefig(f"{FIG_DIR}/duration_distribution_by_risk.png", bbox_inches="tight", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

# 3.3.3 Bad credit rate by loan duration group
duration_risk = (
    df_clean
    .groupby("duration_group",observed=True)["risk"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(7, 4))
sns.barplot(
    x="duration_group",
    y="risk",
    data=duration_risk
)
plt.title("Bad Credit Rate by Loan Duration Group")
plt.xlabel("Loan Duration Group (Months)")
plt.ylabel("Bad Credit Rate")
plt.ylim(0, 1)
plt.savefig(f"{FIG_DIR}/bad_rate_by_duration_group.png", bbox_inches="tight", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

# 3.4.1 Bad credit rate by loan purpose
purpose_risk = (
    df_clean
    .groupby("purpose",observed=True)["risk"]
    .mean()
    .reset_index()
    .sort_values("risk", ascending=False)
)

plt.figure(figsize=(9, 4))
sns.barplot(
    x="purpose",
    y="risk",
    data=purpose_risk
)
plt.title("Bad credit rate by loan purpose")
plt.xlabel("purpose")
plt.ylabel("Bad Credit Rate")
plt.ylim(0, 1)
plt.savefig(f"{FIG_DIR}/bad_rate_by_purpose.png", bbox_inches="tight", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

# 3.4.2 Interaction analysis: loan duration × loan purpose
interaction_risk = (
    df_clean
    .groupby(["purpose", "duration_group"])["risk"]
    .mean()
    .reset_index()
)

# turn it to pivot table
interaction_pivot = interaction_risk.pivot(
    index="purpose",
    columns="duration_group",
    values="risk"
)
# heat map
plt.figure(figsize=(9, 5))
sns.heatmap(
    interaction_pivot,
    annot=True,
    fmt=".2f",
    cmap="Reds"
)

plt.title("Bad Credit Rate by Loan Purpose and Duration Group")
plt.xlabel("Loan Duration Group (Months)")
plt.ylabel("Loan Purpose")
plt.savefig(f"{FIG_DIR}/interaction_duration_purpose.png", bbox_inches="tight", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()
print(f"\nEDA plots saved to: {FIG_DIR}/")

