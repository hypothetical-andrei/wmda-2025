# [Scaling numerical features in a housing price dataset.]
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Step 1: Create a sample housing dataset
data = {
    "HouseID": [1, 2, 3, 4, 5],
    "SquareFeet": [1500, 1800, 1200, 2000, 1600],
    "NumRooms": [3, 4, 2, 5, 3],
    "Price": [250000, 300000, 200000, 350000, 280000]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# Step 2: Initialize scalers
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Step 3: Apply MinMax Scaling (scales values between 0 and 1)
df_minmax_scaled = df.copy()
df_minmax_scaled[["SquareFeet", "NumRooms", "Price"]] = minmax_scaler.fit_transform(df[["SquareFeet", "NumRooms", "Price"]])

print("\nDataset After Min-Max Scaling:\n")
print(df_minmax_scaled)

# Step 4: Apply Standard Scaling (scales values to have mean = 0 and std deviation = 1)
df_standard_scaled = df.copy()
df_standard_scaled[["SquareFeet", "NumRooms", "Price"]] = standard_scaler.fit_transform(df[["SquareFeet", "NumRooms", "Price"]])

print("\nDataset After Standard Scaling:\n")
print(df_standard_scaled)
