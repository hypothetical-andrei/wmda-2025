# [An example illustrating how segments are used to personalize emails and offers]
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 1. Generate synthetic customer data
#    For example, each row is a customer with:
#    - Purchase frequency (times per month)
#    - Average amount spent per purchase
#    - Loyalty score (some measure of long-term engagement)
np.random.seed(42)
num_customers = 20
customer_data = pd.DataFrame({
    'purchase_frequency': np.random.randint(1, 20, num_customers),
    'avg_spent': np.random.randint(10, 300, num_customers),
    'loyalty_score': np.random.randint(1, 10, num_customers)
})

# 2. Perform K-Means clustering to segment customers
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['segment'] = kmeans.fit_predict(customer_data)

# 3. Define different marketing messages/offers for each segment
marketing_strategies = {
    0: {
        'subject': "Welcome to the Community!",
        'offer': "10% discount on your next purchase"
    },
    1: {
        'subject': "VIP Customer Appreciation",
        'offer': "Exclusive access to new products"
    },
    2: {
        'subject': "Loyalty Program Boost",
        'offer': "Double loyalty points for a limited time"
    },
}

# 4. Assign marketing messages based on segment
customer_data['email_subject'] = customer_data['segment'].apply(
    lambda seg: marketing_strategies[seg]['subject']
)
customer_data['special_offer'] = customer_data['segment'].apply(
    lambda seg: marketing_strategies[seg]['offer']
)

# 5. Print results showing personalized strategies
print(customer_data.head(10))
