"""
Capstone Project: Breast Cancer Diagnostic Prediction & Patient Sub-grouping
Full pipeline: acquisition -> cleaning -> EDA -> supervised modeling ->
unsupervised modeling -> evaluation -> artifacts (figures + metrics json)
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

sns.set_theme(style="whitegrid", font_scale=1.0)
FIG = "/home/claude/capstone/figs"
RESULTS = {}

# ---------------------------------------------------------------
# 1. DATA ACQUISITION
# ---------------------------------------------------------------
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
df.rename(columns={"target": "diagnosis"}, inplace=True)
# diagnosis: 0 = malignant, 1 = benign  (per sklearn docs)
df["diagnosis_label"] = df["diagnosis"].map({0: "malignant", 1: "benign"})

RESULTS["dataset"] = {
    "source": "UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic) "
              "Data Set, accessed via scikit-learn's built-in loader",
    "n_rows": int(df.shape[0]),
    "n_features": int(len(raw.feature_names)),
    "class_balance": df["diagnosis_label"].value_counts().to_dict()
}

df.to_csv("/home/claude/capstone/data/raw_data.csv", index=False)

# ---------------------------------------------------------------
# 2. DATA CLEANING / PREPROCESSING
# ---------------------------------------------------------------
missing_before = int(df.isnull().sum().sum())
duplicates = int(df.duplicated().sum())
df = df.drop_duplicates()

X = df[list(raw.feature_names)]
y = df["diagnosis"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

RESULTS["cleaning"] = {
    "missing_values_found": missing_before,
    "duplicate_rows_found": duplicates,
    "scaling_method": "StandardScaler (zero mean, unit variance)"
}

# ---------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------
desc = X.describe().T[["mean", "std", "min", "max"]].round(2)
desc.to_csv("/home/claude/capstone/data/summary_stats.csv")

# 3a. Class balance plot
plt.figure(figsize=(5, 4))
sns.countplot(x="diagnosis_label", data=df, palette=["#d9534f", "#5cb85c"])
plt.title("Class Distribution: Malignant vs Benign")
plt.xlabel("")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{FIG}/01_class_balance.png", dpi=150)
plt.close()

# 3b. Correlation heatmap (subset of most relevant "mean" features)
mean_cols = [c for c in X.columns if c.startswith("mean")]
corr = X[mean_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap - 'Mean' Feature Group")
plt.tight_layout()
plt.savefig(f"{FIG}/02_correlation_heatmap.png", dpi=150)
plt.close()

# 3c. Distribution of key features by diagnosis
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, feat in zip(axes, ["mean radius", "mean texture", "mean concavity"]):
    sns.kdeplot(data=df, x=feat, hue="diagnosis_label", fill=True, alpha=0.4, ax=ax,
                palette=["#d9534f", "#5cb85c"])
    ax.set_title(feat)
plt.tight_layout()
plt.savefig(f"{FIG}/03_feature_distributions.png", dpi=150)
plt.close()

top_corr_with_target = X.corrwith(y).abs().sort_values(ascending=False).head(8)
RESULTS["eda"] = {
    "top_correlated_features_with_diagnosis": {k: round(float(v), 3) for k, v in top_corr_with_target.items()}
}

# ---------------------------------------------------------------
# 4. SUPERVISED MODELING
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42)
}

sup_results = {}
plt.figure(figsize=(6, 5))
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(model, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                 scoring="accuracy")

    sup_results[name] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
    }

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={sup_results[name]['roc_auc']:.3f})")

    if name == "Random Forest":
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["malignant", "benign"], yticklabels=["malignant", "benign"])
        plt.title("Random Forest - Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(f"{FIG}/05_confusion_matrix_rf.png", dpi=150)
        plt.close()

        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        plt.figure(figsize=(7, 5))
        sns.barplot(x=importances.values, y=importances.index, color="#5b8def")
        plt.title("Random Forest - Top 10 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{FIG}/06_feature_importance.png", dpi=150)
        plt.close()

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Supervised Models")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/04_roc_curves.png", dpi=150)
plt.close()

RESULTS["supervised"] = sup_results

# ---------------------------------------------------------------
# 5. UNSUPERVISED MODELING (PCA + KMeans)
# ---------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, cluster_labels)
ari = adjusted_rand_score(y, cluster_labels)

plt.figure(figsize=(6, 5))
scatter_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1],
                            "Cluster": cluster_labels.astype(str),
                            "Diagnosis": df["diagnosis_label"]})
sns.scatterplot(data=scatter_df, x="PC1", y="PC2", hue="Diagnosis", style="Cluster",
                 palette=["#d9534f", "#5cb85c"], alpha=0.75)
plt.title(f"PCA Projection with KMeans Clusters\n(Explained variance: PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%)")
plt.tight_layout()
plt.savefig(f"{FIG}/07_pca_clusters.png", dpi=150)
plt.close()

# Elbow method plot to justify k=2
inertias = []
k_range = range(1, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
plt.figure(figsize=(6, 4))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig(f"{FIG}/08_elbow_method.png", dpi=150)
plt.close()

RESULTS["unsupervised"] = {
    "method": "PCA (2 components) + KMeans (k=2)",
    "pca_explained_variance_pc1": round(float(explained_var[0]), 4),
    "pca_explained_variance_pc2": round(float(explained_var[1]), 4),
    "silhouette_score": round(float(sil_score), 4),
    "adjusted_rand_index_vs_true_diagnosis": round(float(ari), 4)
}

with open("/home/claude/capstone/data/results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
