# -*- coding: utf-8 -*-
"""
K-Means Clustering — Simple Example
=====================================
Dataset  : Synthetic 2D data (generated with make_blobs)
Features : Feature 1, Feature 2
Goal     : Group data points into clusters using K-Means Clustering
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# ── 2. Create Simple Dataset ──────────────────────────────────────────────────
X, _ = make_blobs(n_samples=150, centers=4, cluster_std=1.0, random_state=42)

df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ── 3. Scale the Data ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ── 4. Elbow Method — Find Optimal Number of Clusters ────────────────────────
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.tight_layout()
plt.show()

# ── 5. Apply K-Means Clustering ───────────────────────────────────────────────
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

print("\nCluster assignment counts:")
print(df['Cluster'].value_counts().sort_index())

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
score = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score: {score:.4f}")

# ── 7. Plot ───────────────────────────────────────────────────────────────────
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
plt.title(f"K-Means Clustering\nSilhouette Score: {score:.4f}")
plt.tight_layout()
plt.savefig("kmeans_clustering_simple.png", dpi=150)
plt.show()
print("\nPlot saved → kmeans_clustering_simple.png")

"""
Evaluation Metrics
------------------
1. Elbow Method
The elbow method was used to find the optimal number of clusters.
The graph showed an elbow point around K = 4.

2. Silhouette Score
Silhouette Score was used to evaluate clustering quality.
Higher score means better-separated and well-grouped clusters.

Key Observations
----------------
1. Four clear groups are visible in the scatter plot.
2. K-Means works well when clusters are roughly equal in size and shape.
"""
