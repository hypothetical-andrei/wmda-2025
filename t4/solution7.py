## Exercise 7 (10 minutes): Anomaly Detection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 1. Generate synthetic "normal" data
#    E.g., two features representing normal operating ranges (e.g., purchase amounts, usage rates, etc.)
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=(200, 2))  # 200 points around mean=50

# 2. Generate synthetic "anomalous" data
#    Points that deviate significantly from the normal distribution
outliers = np.array([[100, 100], [10, 90], [90, 10], [120, 40], [40, 120]])

# 3. Combine the datasets
X = np.vstack((normal_data, outliers))

# 4. Apply DBSCAN
#    eps controls the neighborhood radius; min_samples is how many samples must be within eps to form a cluster

# 5. Identify outliers (DBSCAN labels them as -1)

# 6. Visualization

# 7. Reporting
