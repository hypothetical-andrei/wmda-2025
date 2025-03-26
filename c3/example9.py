# [Predicting house prices based on the prices of nearest houses in the feature space (e.g., similar size, location, etc.) by taking their average price.]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# 1. Create a small synthetic dataset
data = {
    'location': ['cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB'],
    'sqft':     [1000,    2000,    1200,    1800,   900,     2200   ],
    'bedrooms': [2,       4,       3,       4,      2,       5      ],
    'price':    [200000,  400000,  260000,  350000, 180000,   450000]
}
df = pd.DataFrame(data)

# 2. Separate features (X) and target (y)
X = df[['location', 'sqft', 'bedrooms']]
y = df['price']

# 3. Encode the categorical 'location' feature
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)
# After encoding, we'll have columns: ['sqft', 'bedrooms', 'location_cityB']

# 4. (Optional) Scale the numeric features to improve distance calculations
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 6. Create and train the KNN regressor
#    Here we choose k=3 neighbors
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)

# 7. Evaluate on the test set
y_pred_test = knn_reg.predict(X_test)
print("Test Set Predictions:", y_pred_test)
print("True Values:", y_test.values)

# 8. Predict the price of a new house
#    For example: new house with 1500 sqft, 3 bedrooms in cityB
new_house = pd.DataFrame({
    'sqft': [1500],
    'bedrooms': [3],
    'location_cityB': [1]  # Because we used drop_first=True (cityA=0, cityB=1)
})

# We need to scale it the same way
new_house_scaled = scaler.transform(new_house)

predicted_price = knn_reg.predict(new_house_scaled)
print("Predicted price for the new house:", predicted_price[0])
