## Exercise 6 (10 minutes): Customer Segmentation Use Case
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Generate synthetic customer data
#    For example:
#    - 'purchase_frequency': how many purchases per month
#    - 'average_spent': average amount spent per purchase
#    - 'loyalty_score': a simple 1–5 rating

np.random.seed(42)
num_customers = 50

df_customers = pd.DataFrame({
    'purchase_frequency': np.random.randint(1, 15, num_customers),
    'average_spent': np.random.randint(10, 500, num_customers),
    'loyalty_score': np.random.randint(1, 6, num_customers)
})

print("=== Raw Customer Data (first 5 rows) ===")
print(df_customers.head(), "\n")

# 2. Scale the data

# 3. K-Means clustering

# 4. Add cluster labels to the DataFrame

# 5. Inspect each segment

# 6. (Optional) Quick interpretation or marketing actions
#    For example, cluster 0 may represent "frequent, high-spending customers" etc.
