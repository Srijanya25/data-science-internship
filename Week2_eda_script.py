import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

df = pd.read_csv("telco_cleaned.csv")
print("Shape:", df.shape)

# ---------------------------------------------------------
# 1. BASIC STATISTICS
# ---------------------------------------------------------
desc = df[["tenure", "MonthlyCharges", "TotalCharges"]].describe()
print("\nSummary statistics:\n", desc)

churn_rate = df["Churn"].value_counts(normalize=True)
print("\nOverall churn rate:\n", churn_rate)

# ---------------------------------------------------------
# 2. NUMERIC DISTRIBUTIONS
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["tenure"], bins=30, kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Tenure Distribution")
sns.histplot(df["MonthlyCharges"], bins=30, kde=True, ax=axes[1], color="#55A868")
axes[1].set_title("Monthly Charges Distribution")
sns.histplot(df["TotalCharges"], bins=30, kde=True, ax=axes[2], color="#C44E52")
axes[2].set_title("Total Charges Distribution")
plt.tight_layout()
plt.savefig("01_numeric_distributions.png")
plt.close()

# ---------------------------------------------------------
# 3. CHURN RATE BY KEY CATEGORICAL FEATURES
# ---------------------------------------------------------
def churn_rate_by(col):
    return df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col, title in zip(
    axes,
    ["Contract", "InternetService", "PaymentMethod"],
    ["Contract Type", "Internet Service", "Payment Method"],
):
    rates = churn_rate_by(col).sort_values(ascending=False)
    sns.barplot(x=rates.index, y=rates.values, ax=ax, palette="Blues_d")
    ax.set_title(f"Churn Rate by {title}")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("02_churn_by_categorical.png")
plt.close()

# ---------------------------------------------------------
# 4. TENURE vs CHURN
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(x="Churn", y="tenure", data=df, palette=["#55A868", "#C44E52"], ax=ax)
ax.set_title("Tenure by Churn Status")
plt.tight_layout()
plt.savefig("03_tenure_vs_churn.png")
plt.close()

# ---------------------------------------------------------
# 5. MONTHLY CHARGES vs CHURN
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True,
            palette={"No": "#55A868", "Yes": "#C44E52"}, ax=ax)
ax.set_title("Monthly Charges Distribution by Churn")
plt.tight_layout()
plt.savefig("04_monthlycharges_vs_churn.png")
plt.close()

# ---------------------------------------------------------
# 6. CORRELATION HEATMAP (numeric features)
# ---------------------------------------------------------
corr = df[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, fmt=".2f")
ax.set_title("Correlation Between Numeric Features")
plt.tight_layout()
plt.savefig("05_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# 7. TENURE GROUP vs CHURN
# ---------------------------------------------------------
tg_churn = df.groupby("TenureGroup")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.barplot(x=tg_churn.index, y=tg_churn.values, palette="Oranges_d", ax=ax)
ax.set_title("Churn Rate by Tenure Group")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Tenure Group (months)")
plt.tight_layout()
plt.savefig("06_tenuregroup_vs_churn.png")
plt.close()

# ---------------------------------------------------------
# 8. SENIOR CITIZEN / PARTNER / DEPENDENTS vs CHURN
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, ["SeniorCitizen", "Partner", "Dependents"]):
    rates = churn_rate_by(col)
    sns.barplot(x=rates.index, y=rates.values, ax=ax, palette="Purples_d")
    ax.set_title(f"Churn Rate by {col}")
    ax.set_ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("07_demographics_vs_churn.png")
plt.close()

# ---------------------------------------------------------
# 9. KEY NUMBERS FOR THE REPORT (printed so we can quote them)
# ---------------------------------------------------------
print("\n--- Key figures for report ---")
print("Churn rate, Month-to-month contract:", round(churn_rate_by("Contract")["Month-to-month"], 1))
print("Churn rate, Two year contract:", round(churn_rate_by("Contract")["Two year"], 1))
print("Churn rate, Fiber optic:", round(churn_rate_by("InternetService")["Fiber optic"], 1))
print("Churn rate, DSL:", round(churn_rate_by("InternetService")["DSL"], 1))
print("Churn rate, Electronic check:", round(churn_rate_by("PaymentMethod")["Electronic check"], 1))
print("Median tenure, churned:", df[df.Churn == "Yes"]["tenure"].median())
print("Median tenure, retained:", df[df.Churn == "No"]["tenure"].median())
print("Mean MonthlyCharges, churned:", round(df[df.Churn == "Yes"]["MonthlyCharges"].mean(), 2))
print("Mean MonthlyCharges, retained:", round(df[df.Churn == "No"]["MonthlyCharges"].mean(), 2))
print("Churn rate, tenure group 0-12:", round(tg_churn["0-12"], 1))
print("Churn rate, tenure group 61-72:", round(tg_churn["61-72"], 1))
print("Corr tenure vs TotalCharges:", round(corr.loc["tenure","TotalCharges"], 2))
print("Corr MonthlyCharges vs TotalCharges:", round(corr.loc["MonthlyCharges","TotalCharges"], 2))
print("Corr tenure vs MonthlyCharges:", round(corr.loc["tenure","MonthlyCharges"], 2))

print("\nDone. All charts saved.")
