## Exercise 8 (10 minutes): Visual Summary & Report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# 1. Load or assume you have a preprocessed dataset
#    Here, we load & scale the Iris dataset for demonstration
iris = load_iris()
X = iris.data
y = iris.target  # Not used for clustering, but sometimes nice for reference

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply three clustering algorithms
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Reduce to 2D with PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 4. Create a combined DataFrame for plotting & reporting
df_final = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_final['KMeans'] = kmeans.labels_
df_final['DBSCAN'] = dbscan.labels_
df_final['Agglo'] = agg.labels_

# 5. Plot side-by-side scatter plots of the clustering results in PCA space

# 6. Print a short cluster distribution report
