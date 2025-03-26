# [An example illustrating K-means clustering in a dataset of customer purchase history]
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Generate a synthetic customer purchase dataset
#    (In real use cases, load or query your actual data instead of random values)
np.random.seed(42)
n_customers = 100
customers = pd.DataFrame({
    'purchase_frequency': np.random.randint(1, 50, n_customers),
    'average_spent': np.random.randint(10, 500, n_customers)
})

# 2. Create and fit a KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(customers)

# 3. Add the cluster labels to the DataFrame
customers['cluster'] = kmeans.labels_

# 4. Plot the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(
    customers['purchase_frequency'],
    customers['average_spent'],
    c=customers['cluster']
)
plt.title("K-Means Clustering of Synthetic Customer Data")
plt.xlabel("Purchase Frequency")
plt.ylabel("Average Spent")
plt.show()

# 5. Print sample rows to see cluster assignments
print(customers.head(10))
