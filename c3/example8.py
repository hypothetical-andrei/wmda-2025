# [Brief demonstration of using a real dataset (e.g., online advertising spend vs. sales) to build a regression model]
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. Create a toy dataset of advertising spends and product sales
#    (In a real scenario, you would load from a CSV or database)
data = {
    'TV':        [230.1, 44.5, 17.2, 151.5, 180.8, 8.7,   57.5,  120.2, 8.6,   199.8],
    'Radio':     [37.8,  39.3,  45.9, 41.3,  10.8,  48.9,  32.8,  19.6,  2.1,   2.6],
    'Newspaper': [69.2,  45.1,  69.3, 58.5,  58.4,  75.0,  23.5,  11.6,  1.0,   21.2],
    'Sales':     [22.1,  10.4,  9.3,  18.5,  12.9,  7.2,   11.8,  13.2,  4.8,   10.6]
}
df = pd.DataFrame(data)

# 2. Separate features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 3. Split into training/test sets (typical machine-learning workflow)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model using R² and MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Coefficients (TV, Radio, Newspaper):", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 on test set:", r2)
print("MSE on test set:", mse)
