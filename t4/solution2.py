## Exercise 2 (10 minutes): K-Means Clustering
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    (containing numeric, imputed, and scaled features).
#    For demonstration, let's simulate df_scaled with the Iris dataset's features.
from sklearn.datasets import load_iris
import numpy as np

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# ------------------------------------------------------------------------------------

# 2. Instantiate K-Means with a chosen number of clusters, say 3

# 3. Fit the model to the data

# 4. Extract cluster labels

# 5. (Optional) Add the cluster labels to the DataFrame

# 6. Print or visualize the results

# 7. Optional quick visualization (for 2D only)
#    If you'd like a scatter plot, choose two features to plot.
