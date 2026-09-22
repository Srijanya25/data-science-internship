import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("telco.csv")
print("Initial shape:", df.shape)
print(df.dtypes)

# ---------------------------------------------------------
# 2. INITIAL EXPLORATION
# ---------------------------------------------------------
summary_before = {
    "rows": df.shape[0],
    "columns": df.shape[1],
    "duplicate_rows": df.duplicated().count() - df.drop_duplicates().shape[0],
}
print(summary_before)

# ---------------------------------------------------------
# 3. IDENTIFY DATA QUALITY ISSUES
# ---------------------------------------------------------
# TotalCharges is read as object because 11 rows contain blank strings
# instead of numeric values (for customers with tenure = 0, i.e. brand-new
# customers who haven't been billed yet).
df["TotalCharges_raw"] = df["TotalCharges"]
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

missing_report = df.isna().sum()
missing_report = missing_report[missing_report > 0]
print("\nMissing values after coercion:\n", missing_report)

blank_rows = df[df["TotalCharges"].isna()][["customerID", "tenure", "MonthlyCharges", "TotalCharges_raw"]]
print("\nRows with blank TotalCharges (all have tenure = 0):\n", blank_rows)

# ---------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ---------------------------------------------------------
# These are brand-new customers (tenure = 0) who have not yet been billed,
# so TotalCharges is legitimately "not yet applicable" rather than random
# missingness. We impute with 0, which is consistent with tenure * MonthlyCharges = 0.
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df.drop(columns=["TotalCharges_raw"], inplace=True)

# ---------------------------------------------------------
# 5. DUPLICATES
# ---------------------------------------------------------
dupes = df.duplicated(subset="customerID").sum()
df = df.drop_duplicates(subset="customerID")
print(f"\nDuplicate customerIDs removed: {dupes}")

# ---------------------------------------------------------
# 6. OUTLIER DETECTION (IQR method) ON NUMERIC COLUMNS
# ---------------------------------------------------------
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
outlier_summary = {}
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary[col] = {"lower_bound": round(lower, 2), "upper_bound": round(upper, 2), "n_outliers": int(n_outliers)}
print("\nOutlier summary (IQR method):\n", outlier_summary)
# Decision: none of the three numeric columns show extreme/impossible values
# (e.g. no negative charges, no absurd tenure) -> no rows dropped as outliers.
# Values flagged by IQR are legitimate (e.g. long-tenure, high-paying customers).

# ---------------------------------------------------------
# 7. STANDARDIZE CATEGORICAL / INCONSISTENT ENTRIES
# ---------------------------------------------------------
# Several service columns use "No internet service" / "No phone service" as a
# third category duplicating "No" in meaning for downstream binary analysis.
replace_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]
for col in replace_cols:
    df[col] = df[col].replace({"No internet service": "No"})
df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

# SeniorCitizen is 0/1 -> convert to Yes/No for consistency with similar columns
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# ---------------------------------------------------------
# 8. FEATURE ENGINEERING (light preprocessing for downstream weeks)
# ---------------------------------------------------------
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 60, 100],
                             labels=["0-12", "13-24", "25-48", "49-60", "61-72"])

# ---------------------------------------------------------
# 9. FINAL CHECKS
# ---------------------------------------------------------
print("\nFinal shape:", df.shape)
print("Remaining missing values:", df.isna().sum().sum())
print(df["Churn"].value_counts(normalize=True))

df.to_csv("telco_cleaned.csv", index=False)

# ---------------------------------------------------------
# 10. VISUALS FOR THE REPORT
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(y=df["tenure"], ax=axes[0], color="#4C72B0"); axes[0].set_title("Tenure (months)")
sns.boxplot(y=df["MonthlyCharges"], ax=axes[1], color="#55A868"); axes[1].set_title("Monthly Charges")
sns.boxplot(y=df["TotalCharges"], ax=axes[2], color="#C44E52"); axes[2].set_title("Total Charges")
plt.tight_layout()
plt.savefig("boxplots_after_cleaning.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(5, 4))
df["Churn"].value_counts().plot(kind="bar", color=["#4C72B0", "#C44E52"], ax=ax)
ax.set_title("Churn Class Distribution")
ax.set_xlabel("Churn")
ax.set_ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("churn_distribution.png", dpi=150)
plt.close()

missing_before = pd.Series({"TotalCharges": 11})
fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(["TotalCharges"], [11], color="#DD8452")
ax.set_title("Missing Values Before Cleaning")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("missing_before.png", dpi=150)
plt.close()

print("\nDone. Cleaned file + 3 charts saved.")
