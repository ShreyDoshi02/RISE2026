# -*- coding: utf-8 -*-
"""
Hierarchical Clustering — Simple Example
==========================================
Dataset  : Synthetic 2D data (generated with make_blobs)
Features : Feature 1, Feature 2
Goal     : Group data points into clusters using Agglomerative Clustering
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# ── 2. Create Simple Dataset ──────────────────────────────────────────────────
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.2, random_state=42)

df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ── 3. Scale the Data ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ── 4. Apply Hierarchical Clustering ─────────────────────────────────────────
model = AgglomerativeClustering(n_clusters=3)
clusters = model.fit_predict(X_scaled)
df['Cluster'] = clusters

print("\nCluster assignment counts:")
print(df['Cluster'].value_counts().sort_index())

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
score = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score: {score:.4f}")

# ── 6. Plot ───────────────────────────────────────────────────────────────────
plt.scatter(
    df['Feature 1'],
    df['Feature 2'],
    c=clusters,
    cmap='viridis',
    s=60,
    edgecolors='k',
    linewidths=0.4
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Hierarchical Clustering\nSilhouette Score: {score:.4f}")
plt.tight_layout()
plt.savefig("hierarchical_clustering_simple.png", dpi=150)
plt.show()
print("\nPlot saved → hierarchical_clustering_simple.png")

"""
Evaluation Metrics
------------------
1. Silhouette Score was used to evaluate clustering quality.
2. Data points were grouped based on their closeness in 2D space.

Key Observations
----------------
1. Three clear groups are visible in the scatter plot.
2. Higher silhouette score means better-separated clusters.
"""
