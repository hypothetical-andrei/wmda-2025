# [Dealing with class imbalance in fraud detection datasets.]
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Step 1: Create a sample imbalanced dataset
np.random.seed(42)
data = {
    "TransactionID": range(1, 21),
    "Amount": np.random.randint(10, 1000, 20),
    "IsFraud": [0] * 17 + [1] * 3  # Imbalanced: 17 non-fraud, 3 fraud cases
}

df = pd.DataFrame(data)

# Step 2: Separate features and target
X = df[["Amount"]]  # Features
y = df["IsFraud"]   # Target variable

print("Original Class Distribution:", Counter(y))

# Step 3: Apply Undersampling (reduce majority class)
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)

print("Class Distribution After Undersampling:", Counter(y_under))

# Step 4: Apply Oversampling (SMOTE with reduced n_neighbors)
smote = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=1)  # Reduce k_neighbors
X_smote, y_smote = smote.fit_resample(X, y)

print("Class Distribution After Oversampling (SMOTE):", Counter(y_smote))
