"""
Week 4 Task: Supervised Learning Model Implementation
Problem: Binary classification - predicting whether a breast tumor is
         malignant or benign from diagnostic measurements.
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) dataset - a well known
         public dataset, bundled with scikit-learn (originally from the
         UCI Machine Learning Repository).
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

RANDOM_STATE = 42
OUT = "/home/claude/week4"

# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["target"] = data.target  # 0 = malignant, 1 = benign
target_names = data.target_names

print("Shape:", df.shape)
print(df["target"].value_counts())

df.to_csv(f"{OUT}/breast_cancer_data.csv", index=False)

# ---------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------
summary_stats = df.describe().T
summary_stats.to_csv(f"{OUT}/summary_stats.csv")

missing = df.isnull().sum().sum()
print("Missing values:", missing)

class_counts = df["target"].value_counts().rename(index={0: "malignant", 1: "benign"})

plt.figure(figsize=(5, 4))
sns.countplot(x=df["target"].map({0: "malignant", 1: "benign"}), palette=["#c0392b", "#2980b9"])
plt.title("Class Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT}/class_distribution.png", dpi=150)
plt.close()

# Correlation heatmap (subset of most relevant "mean" features for readability)
mean_cols = [c for c in df.columns if c.startswith("mean")] + ["target"]
plt.figure(figsize=(10, 8))
sns.heatmap(df[mean_cols].corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap (mean-value features)")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# 3. FEATURE ENGINEERING / PREPROCESSING
# ---------------------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]

# Drop highly redundant/collinear features (e.g. "area" is derivable from
# "radius"; keeping all 30 raw features is fine for tree models but hurts
# linear models via multicollinearity). We engineer a couple of ratio
# features and drop a few very highly correlated raw ones (>0.95 with
# another feature) to reduce redundancy for the linear model.
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
print("Dropping highly collinear features:", to_drop)

X_reduced = X.drop(columns=to_drop)

# Engineered feature: worst-to-mean ratio for radius (captures how much
# the "worst" (largest) measurement deviates from the average measurement
# of a tumor -- a proxy for irregularity/growth variance)
X_reduced["radius_worst_to_mean_ratio"] = X["worst radius"] / X["mean radius"]
X_reduced["texture_worst_to_mean_ratio"] = X["worst texture"] / X["mean texture"]

feature_names_final = X_reduced.columns.tolist()
print("Final feature count:", len(feature_names_final))

# ---------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------------------------------------------------------------
# 5. MODEL PIPELINES (scaling + classifier)
# ---------------------------------------------------------------------
pipelines = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=RANDOM_STATE))
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = {}
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_results[name] = scores
    print(f"{name}: CV ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

# ---------------------------------------------------------------------
# 6. HYPERPARAMETER TUNING (best-performing family: Logistic Regression)
# ---------------------------------------------------------------------
param_grid = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs"],
}
grid = GridSearchCV(
    pipelines["Logistic Regression"], param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

best_model = grid.best_estimator_

# Also fit Random Forest and SVM plainly for comparison on test set
pipelines["Logistic Regression"] = best_model
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 7. TEST-SET EVALUATION
# ---------------------------------------------------------------------
results_table = []
roc_curves = {}
for name, pipe in pipelines.items():
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results_table.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC-AUC": auc
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, auc)

    if name == "Logistic Regression":
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["malignant", "benign"])
        print(report)

results_df = pd.DataFrame(results_table).sort_values("ROC-AUC", ascending=False)
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
print(results_df)

# Confusion matrix plot (final chosen model = tuned Logistic Regression)
final_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["malignant", "benign"], yticklabels=["malignant", "benign"])
plt.title("Confusion Matrix - Logistic Regression (tuned)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()

# ROC curves comparison plot
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=150)
plt.close()

# Feature importance (Random Forest) for interpretability discussion
rf_model = pipelines["Random Forest"].named_steps["clf"]
importances = pd.Series(rf_model.feature_importances_, index=feature_names_final)
importances = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(7, 5))
importances.sort_values().plot(kind="barh", color="#2c3e50")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
plt.close()

# Logistic regression coefficients (interpretability)
lr_clf = best_model.named_steps["clf"]
coefs = pd.Series(lr_clf.coef_[0], index=feature_names_final).sort_values()
top_coefs = pd.concat([coefs.head(5), coefs.tail(5)])

plt.figure(figsize=(7, 5))
top_coefs.plot(kind="barh", color=["#c0392b" if v < 0 else "#2980b9" for v in top_coefs])
plt.title("Logistic Regression Coefficients (top/bottom 5)")
plt.xlabel("Coefficient (standardized features)")
plt.tight_layout()
plt.savefig(f"{OUT}/lr_coefficients.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# 8. SAVE SUMMARY JSON FOR REPORT GENERATION
# ---------------------------------------------------------------------
summary = {
    "dataset_shape": df.shape,
    "class_counts": class_counts.to_dict(),
    "missing_values": int(missing),
    "dropped_collinear_features": to_drop,
    "final_feature_count": len(feature_names_final),
    "train_shape": X_train.shape,
    "test_shape": X_test.shape,
    "cv_results": {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in cv_results.items()},
    "best_params": grid.best_params_,
    "best_cv_auc": float(grid.best_score_),
    "results_table": results_df.to_dict(orient="records"),
    "top_rf_features": importances.to_dict(),
    "confusion_matrix": cm.tolist(),
}

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nDONE.")
print(json.dumps(summary, indent=2, default=str))
