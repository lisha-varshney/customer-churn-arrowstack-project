# Customer Churn Investigation - short version
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("output", exist_ok=True)

# 1. Load data
df = pd.read_csv(r"C:\Users\PC\Desktop\Customer-Churn.csv")
print("Shape:", df.shape)

# 2. Data dictionary
dictionary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "unique": df.nunique(),
    "example": [df[c].iloc[0] for c in df.columns],
})
dictionary.to_csv(r"output\data_dictionary.csv", index=False)
print(dictionary)

# 3. Cleaning and quality checks
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())
print("Customers with tenure = 0 (new):", (df["tenure"] == 0).sum())

df = df.drop_duplicates("customerID")
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df = df.dropna(subset=["tenure"])

# New columns
df["tenure_group"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                            labels=["0-12", "13-24", "25-48", "49-72"])
df["household"] = "Single"
df.loc[df["Partner"] == "Yes", "household"] = "Partner"
df.loc[df["Dependents"] == "Yes", "household"] = "Dependents"
df.loc[(df["Partner"] == "Yes") & (df["Dependents"] == "Yes"), "household"] = "Partner+Dependents"

# 4. Churn column check
has_churn = "Churn" in df.columns
if has_churn:
    df["churn_flag"] = (df["Churn"] == "Yes").astype(int)

# 5. EDA with business questions
# Q1: How many customers are in their first year?
print("\nQ1: % customers in first 12 months:", round((df["tenure"] <= 12).mean() * 100, 1))

# Q2: Do fiber users take the security add-on less?
sec = df.groupby("InternetService")["OnlineSecurity"].apply(lambda s: round((s == "Yes").mean() * 100, 1))
print("\nQ2: % with OnlineSecurity by internet type:\n", sec)

# Q3: Which household type has the shortest tenure?
hh = df.groupby("household")["tenure"].mean().round(1)
print("\nQ3: Avg tenure by household:\n", hh)

# Q4: Churn rate by segment (only if Churn exists)
if has_churn:
    for col in ["tenure_group", "InternetService", "OnlineSecurity", "household", "SeniorCitizen"]:
        print(f"\nChurn rate by {col}:\n", (df.groupby(col, observed=True)["churn_flag"].mean() * 100).round(1))

# 6. Charts
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["tenure"], bins=30, ax=ax[0]).set_title("Tenure (months)")
sns.countplot(data=df, x="InternetService", hue="OnlineSecurity", ax=ax[1]).set_title("Internet vs Security")
sns.boxplot(data=df, x="household", y="tenure", ax=ax[2]).set_title("Tenure by household")
plt.setp(ax[2].get_xticklabels(), rotation=20)
plt.tight_layout()
plt.savefig("output/eda_charts.png", dpi=150)

# 7. Simple risk flag (assumption-based, not from real churn data)
df["high_risk"] = ((df["tenure"] <= 12) &
                   (df["InternetService"] != "No") &
                   (df["OnlineSecurity"] == "No")).astype(int)
print("\nHigh-risk customers:", df["high_risk"].sum(), f"({df['high_risk'].mean() * 100:.1f}%)")
df.to_csv("output/cleaned_data.csv", index=False)

# 8. Validation
assert df["customerID"].is_unique
assert df["tenure"].between(0, 120).all()
print("\nValidation passed.")

# 9. Limitations and next steps
print("""
LIMITATIONS:
- File has no Churn column, so real churn drivers cannot be measured.
- No billing/contract data. Risk flag is an assumption, not proof.
NEXT STEPS:
- Get dataset with Churn column and re-run (Q4 runs automatically).
- Test retention offers on high-risk customers.
""")
