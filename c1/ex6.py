# [Handling missing age values in a Titanic survival dataset.]
import pandas as pd
import numpy as np

# Step 1: Create a sample Titanic dataset with missing age values
data = {
    "PassengerID": [1, 2, 3, 4, 5],
    "Name": ["John", "Emma", "Liam", "Sophia", "Noah"],
    "Age": [22, np.nan, 24, np.nan, 30],  # Missing values in Age column
    "Survived": [1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

print("Original Dataset with Missing Age Values:\n")
print(df)

# Step 2: Handle missing age values using mean imputation
df["Age_mean"] = df["Age"].fillna(df["Age"].mean())

# Step 3: Handle missing age values using median imputation
df["Age_median"] = df["Age"].fillna(df["Age"].median())

# Step 4: Handle missing age values using mode imputation
df["Age_mode"] = df["Age"].fillna(df["Age"].mode()[0])

print("\nDataset After Handling Missing Age Values (Mean, Median, Mode):\n")
print(df[["PassengerID", "Name", "Age", "Age_mean", "Age_median", "Age_mode"]])
