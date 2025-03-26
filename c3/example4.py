# [Regression tree steps for predicting house price: first split on location, then number of rooms, then square footage, etc.]
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# 1. Create a small toy dataset
data = {
    'location': ['cityA', 'cityA', 'cityB', 'cityB', 'cityA', 'cityB'],
    'rooms':    [2,      3,      2,      4,      3,      5     ],
    'sqft':     [800,   1200,    900,   1800,   1100,   2200   ],
    'price':    [100000,180000,160000,290000,200000,360000]
}
df = pd.DataFrame(data)

# 2. Separate features (X) and target (y)
X = df[['location', 'rooms', 'sqft']]
y = df['price']

# 3. Encode the categorical 'location' variable
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)

# 4. Create and train a DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_encoded, y)

# 5. Predict the price of a new house
#    For example, a new house in 'cityB', with 4 rooms and 2000 sqft
new_house = pd.DataFrame({
    'rooms': [4],
    'sqft':  [2000],
    'location_cityB': [1]  # Because we used drop_first=True, cityA = 0, cityB = 1
})

predicted_price = tree_reg.predict(new_house)
print("Predicted price for new house:", predicted_price[0])
