import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# check the dataset
DATA_PATH="ab_data.csv"
df = pd.read_csv(DATA_PATH)
if "unnamed:0" in df.columns:
    df = df.drop(columns=["unnamed:0"])

print(df.shape)
print(df.head())
print(df.info())
print(df.columns)

# data cleaning
print("\nMissing value:\n",df.isnull().sum().sort_values(ascending=False))
rows_with_any_null = df.isna().any(axis=1).sum()
print("\nNumber of rows containing any null values:", rows_with_any_null)

# check logic
valid_control = (df['group'] == 'control') & (df['landing_page'] == 'old_page')
valid_treatment = (df['group'] == 'treatment') & (df['landing_page'] == 'new_page')

df['is_valid'] = valid_control | valid_treatment # As long as one of these conditions is met, the logic is correct.
invalid_rows=df[~df['is_valid']]
print(f"\nThe amount of invalid rows: {invalid_rows.shape[0]}")


# Examine which combinations of logical errors originate (to help pinpoint the problem).
if invalid_rows.shape[0] > 0:
    print("\ncombination of logic mistake（group x landing_page）：")
    print(invalid_rows.groupby(['group', 'landing_page']).size().sort_values(ascending=False))

# delete logic mistake
before_logic=df.shape[0]
df_clean = df[df['is_valid']== True].copy()
after_logic=df_clean.shape[0]

print("\nnumber of rows before delete logic mistakes:", before_logic)
print("number of rows after delete logic mistakes:", after_logic)
print("number of rows with logic mistakes:", before_logic - after_logic)

# delete duplicates
before_dedup = df_clean.shape[0]
dup_user_rows = df_clean.duplicated(subset='user_id').sum()
print("\nNumber of duplicate rows calculated by user_id :", dup_user_rows)

df_clean = df_clean.drop_duplicates(subset='user_id', keep='first')
after_dedup = df_clean.shape[0]

print("Number of rows before remove duplicates:", before_dedup)
print("Number of rows after remove duplicates:", after_dedup)
print("Number of duplicates", before_dedup - after_dedup)

#New page traffic percentage (by row)
new_page_share=(df_clean['landing_page']=='new_page').mean()
print(f"New page traffic percentage (by row)：{new_page_share:.2%}")

# Hypothesis Testing
from scipy.stats import norm
summary = df_clean.groupby('group')['converted'].agg(['count', 'sum'])

n_control = summary.loc['control', 'count']
x_control = summary.loc['control', 'sum']

n_treatment = summary.loc['treatment', 'count']
x_treatment = summary.loc['treatment', 'sum']

p_control = x_control / n_control
p_treatment = x_treatment / n_treatment

print("\n=== Conversion Summary ===")
print(f"Control   : n={n_control}, conversions={x_control}, rate={p_control:.6f}")
print(f"Treatment : n={n_treatment}, conversions={x_treatment}, rate={p_treatment:.6f}")

# calculate pooled proportion and Standard error
p_pool = (x_control + x_treatment) / (n_control + n_treatment)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))

# calculate z-statistic
z_score = (p_treatment - p_control) / se

# p-value
p_value = 1 - norm.cdf(z_score)

alpha = 0.05
z_critical = norm.ppf(1 - alpha)

print("\n=== Z-test Result (Right-tailed, α = 0.05) ===")
print(f"Z-score      : {z_score:.6f}")
print(f"P-value      : {p_value:.6f}")
print(f"Z-critical   : {z_critical:.6f}")

# Conclusion
if p_value < alpha:
    print("\nConclusion: Reject H0.")
    print("→ The new page has a significantly higher conversion rate.")
else:
    print("\nConclusion: Fail to reject H0.")
    print("→ No sufficient evidence that the new page converts better.")

# visualization
conversion_summary = (
    df_clean.groupby('group')['converted']
            .mean()
            .reset_index()
)

plt.figure(figsize=(6, 4))
plt.bar(conversion_summary['group'],
        conversion_summary['converted'])

plt.title("Conversion Rate: Old Page vs New Page")
plt.ylabel("Conversion Rate")
plt.xlabel("Group")

# show number on the bar chart
for i, v in enumerate(conversion_summary['converted']):
    plt.text(i, v, f"{v:.2%}", ha='center', va='bottom')

plt.tight_layout()

# save the figure
plt.savefig("figures/conversion_rate_comparison.png", dpi=150)

plt.show()
