# [An example showing DBSCAN separating dense clusters and labeling sparse points as outliers]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# 1. Generate synthetic data (three blobs)
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.80, random_state=42)

# 2. Apply DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
labels = dbscan.fit_predict(X)

# 3. Identify outliers (labeled as -1)
outliers = X[labels == -1]
clusters = X[labels != -1]

# 4. Visualize the clustering result
#    We'll color points by their cluster label.
#    Outliers are given the label -1 and will appear as their own group.
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("DBSCAN Clustering of Synthetic Data (Outliers labeled as -1)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 5. Print some of the outlier points
print("Number of outliers:", len(outliers))
print("Outlier examples:\n", outliers[:5])
