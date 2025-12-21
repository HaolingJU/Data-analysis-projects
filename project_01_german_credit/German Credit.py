# Load the packages used
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
#import warnings
#from plotly.offline import init_notebook_mode
#init_notebook_mode(connected=True)
#warnings.filterwarnings("ignore")


# import data
df = pd.read_csv("german_credit_data")
original_df = df.copy()
original_df = original_df .drop(columns=["Unnamed: 0"])


# Dataset overview and structure
# how many columns and lines
print(original_df.shape) # 1000 rows, 11 columns
print(original_df.head())
print(original_df.describe())
print(original_df.columns)

# understand the target column
print(original_df['Risk'].value_counts())

# View categorical variables and numeric variables separately.
categorical_cols = original_df.select_dtypes(include='object').columns
numerical_cols = original_df.select_dtypes(exclude='object').columns
print(categorical_cols, numerical_cols)

# What information does each category variable contain?
for col in categorical_cols:
    print(f"{col}: {original_df[col].unique()}")
    print("-" * 50)
# 2 data cleaning
print(original_df.isnull().sum().sort_values(ascending=False))

#Uniform column naming
df_clean = original_df.copy()
df_clean.columns = (
    df_clean.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

df_clean["risk"] = df_clean["risk"].map({"good": 0, "bad": 1})
print(df_clean["risk"].value_counts())

# Missing values changed to unknown
missing_cols = ["saving_accounts", "checking_account"]

for col in missing_cols:
    df_clean[col] = df_clean[col].fillna("unknown")

print(df_clean[missing_cols].isnull().sum()) #checking missing values

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
print(df_clean.isnull().sum().sort_values(ascending=False).head(10))
print(df_clean.head())

import matplotlib.pyplot as plt

# 3.1 Calculate risk distribution
risk_counts = df_clean["risk"].value_counts().sort_index()
risk_labels = ["Good Credit", "Bad Credit"]

# Plot
plt.figure(figsize=(6, 4))
plt.bar(risk_labels, risk_counts.values)
plt.title("Overall Credit Risk Distribution")
plt.ylabel("Number of Customers")
plt.xlabel("Credit Risk")


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
plt.show()

