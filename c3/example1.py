# [Show a short scenario about predicting housing prices based on square footage, location, etc.]
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Create a toy dataset
#    In reality, you would load real data from a file or database.
data = {
    'sqft':      [1500, 2000, 1100, 2500, 1400, 2300],
    'bedrooms':  [3,    4,    2,    5,    3,    4],
    'location':  ['cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB'],
    'price':     [300000, 400000, 200000, 500000, 280000, 450000]
}
df = pd.DataFrame(data)

# 2. Separate features (X) from the target (y)
X = df[['sqft', 'bedrooms', 'location']]
y = df['price']

# 3. Convert the categorical 'location' feature into dummy (one-hot) variables
#    'drop_first=True' avoids the dummy variable trap by dropping one category.
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)

# 4. Train a Linear Regression model
model = LinearRegression()
model.fit(X_encoded, y)

# 5. Print out the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 6. Example prediction
#    Suppose we want to predict the price for a new house with:
#      - 1600 sqft
#      - 3 bedrooms
#      - location = 'cityB'
new_house = pd.DataFrame({
    'sqft': [1600],
    'bedrooms': [3],
    'location': ['cityB']
})

# One-hot encode the new data (same columns as training)
new_house_encoded = pd.get_dummies(new_house, columns=['location'], drop_first=True)

# Ensure both have matching columns by reindexing the new data
new_house_encoded = new_house_encoded.reindex(columns=X_encoded.columns, fill_value=0)

predicted_price = model.predict(new_house_encoded)
print("Predicted price for the new house:", predicted_price[0])
