import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

df = pd.read_csv("telco_cleaned.csv")
print("Shape:", df.shape)

# ---------------------------------------------------------
# 1. FEATURE SELECTION
# ---------------------------------------------------------
# We segment customers using the three numeric billing/usage features
# plus a handful of categorical features that describe the type of
# customer relationship (contract commitment, service type, billing
# behaviour). CustomerID and Churn are excluded from the feature set:
# Churn is treated as a label used only afterwards to *profile* clusters,
# never as an input to clustering itself (that would leak the target).
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["Contract", "InternetService", "PaymentMethod",
                         "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"]

X_raw = df[numeric_features + categorical_features].copy()

# ---------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------
# Numeric features are standardized (mean 0, std 1) so that TotalCharges
# (scale of thousands) does not dominate distance calculations over
# tenure (scale of tens). Categorical features are one-hot encoded so
# K-Means can operate on them as binary dimensions.
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="if_binary"), categorical_features),
])
X = preprocessor.fit_transform(X_raw)
feature_names = (numeric_features +
                  list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)))
print("Encoded feature matrix shape:", X.shape)

# ---------------------------------------------------------
# 3. CHOOSING K: ELBOW METHOD + SILHOUETTE SCORE
# ---------------------------------------------------------
k_range = range(2, 9)
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil_scores[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0")
axes[0].set_title("Elbow Method: Inertia vs. k")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (Within-Cluster SS)")

axes[1].plot(list(k_range), sil_scores, marker="o", color="#C44E52")
axes[1].set_title("Silhouette Score vs. k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Average Silhouette Score")
plt.tight_layout()
plt.savefig("01_elbow_silhouette.png")
plt.close()

# Decision: k=4 gives a strong silhouette score while still being small
# enough to interpret as distinct, actionable customer segments (elbow
# also flattens noticeably from k=4 onward).
K = 4

# ---------------------------------------------------------
# 4. FINAL K-MEANS MODEL
# ---------------------------------------------------------
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)
final_sil = silhouette_score(X, df["Cluster"])
print(f"\nFinal K-Means (k={K}) silhouette score: {final_sil:.3f}")
print(df["Cluster"].value_counts().sort_index())

# ---------------------------------------------------------
# 5. PER-SAMPLE SILHOUETTE PLOT
# ---------------------------------------------------------
sample_sil = silhouette_samples(X, df["Cluster"])
fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 0
palette = sns.color_palette("Set2", K)
for i in range(K):
    vals = np.sort(sample_sil[df["Cluster"] == i])
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, color=palette[i], label=f"Cluster {i}")
    y_lower = y_upper + 20
ax.axvline(final_sil, color="red", linestyle="--", label="Average score")
ax.set_title("Silhouette Plot per Cluster")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Customer Index (grouped by cluster)")
ax.set_yticks([])
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("02_silhouette_plot.png")
plt.close()

# ---------------------------------------------------------
# 6. PCA FOR 2D VISUALIZATION
# ---------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]
explained = pca.explained_variance_ratio_
print(f"\nPCA explained variance: PC1={explained[0]*100:.1f}%, PC2={explained[1]*100:.1f}%")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette="Set2",
                 alpha=0.6, s=25, ax=ax)
ax.set_title(f"Customer Segments in PCA Space (k={K})")
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
ax.legend(title="Cluster")
plt.tight_layout()
plt.savefig("03_pca_clusters.png")
plt.close()

# ---------------------------------------------------------
# 7. CLUSTER PROFILING - NUMERIC FEATURES
# ---------------------------------------------------------
numeric_profile = df.groupby("Cluster")[numeric_features].mean().round(1)
print("\nNumeric profile by cluster:\n", numeric_profile)

fig, ax = plt.subplots(figsize=(7, 4.5))
numeric_profile_norm = (numeric_profile - numeric_profile.mean()) / numeric_profile.std()
sns.heatmap(numeric_profile_norm.T, annot=numeric_profile.T, fmt=".1f",
            cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Relative level (z-score)"})
ax.set_title("Cluster Profile: Average Numeric Feature Values")
ax.set_xlabel("Cluster")
plt.tight_layout()
plt.savefig("04_cluster_numeric_profile.png")
plt.close()

# ---------------------------------------------------------
# 8. CLUSTER PROFILING - CATEGORICAL COMPOSITION
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
    ct = pd.crosstab(df["Cluster"], df[col], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(f"{col} Composition by Cluster")
    ax.set_ylabel("% of Cluster")
    ax.set_xlabel("Cluster")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("05_cluster_categorical_composition.png")
plt.close()

# ---------------------------------------------------------
# 9. CLUSTER vs. CHURN (label used only for interpretation, not fitting)
# ---------------------------------------------------------
churn_by_cluster = df.groupby("Cluster")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
print("\nChurn rate (%) by cluster:\n", churn_by_cluster.round(1))

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.barplot(x=churn_by_cluster.index, y=churn_by_cluster.values, palette="Set2", ax=ax)
ax.set_title("Churn Rate by Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Churn Rate (%)")
for i, v in enumerate(churn_by_cluster.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("06_churn_by_cluster.png")
plt.close()

# ---------------------------------------------------------
# 10. HIERARCHICAL CLUSTERING (COMPARISON / VALIDATION)
# ---------------------------------------------------------
# Run on a random sample (dendrograms/agglomerative clustering are
# O(n^2)-O(n^3) and impractical to plot legibly for 7,043 rows).
sample_idx = df.sample(n=300, random_state=42).index
X_sample = X[df.index.get_indexer(sample_idx)] if hasattr(X, "shape") else None
X_sample = np.asarray(X)[df.index.get_indexer(sample_idx)]

Z = linkage(X_sample, method="ward")
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, leaf_rotation=90)
ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage, 300-customer sample)")
ax.set_xlabel("Cluster size (customers merged)")
ax.set_ylabel("Distance")
plt.tight_layout()
plt.savefig("07_dendrogram.png")
plt.close()

agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
agg_labels_sample = agg.fit_predict(X_sample)
agg_sil = silhouette_score(X_sample, agg_labels_sample)
kmeans_sample_labels = df.loc[sample_idx, "Cluster"].values
kmeans_sil_sample = silhouette_score(X_sample, kmeans_sample_labels)
print(f"\nOn the 300-customer sample: Agglomerative silhouette={agg_sil:.3f}, "
      f"K-Means silhouette={kmeans_sil_sample:.3f}")

# ---------------------------------------------------------
# 11. SAVE CLUSTERED DATA + KEY NUMBERS FOR THE REPORT
# ---------------------------------------------------------
df.to_csv("telco_clustered.csv", index=False)

print("\n--- Key figures for report ---")
print("Cluster sizes:\n", df["Cluster"].value_counts().sort_index())
print("Cluster % of total:\n", (df["Cluster"].value_counts(normalize=True) * 100).round(1).sort_index())
print("Silhouette score (final model):", round(final_sil, 3))
print("Numeric profile:\n", numeric_profile)
print("Churn by cluster:\n", churn_by_cluster.round(1))

print("\nDone. Clustered dataset + 7 charts saved.")
