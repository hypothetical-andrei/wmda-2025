## Exercise 5 (10 minutes): Hyperparameter Tuning with Grid Search
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic dataset
np.random.seed(42)
num_samples = 50
X = np.random.rand(num_samples, 2) * 10  # Two numeric features
# Define a "true" relationship
# y = 3*x0 + 2*x1 + noise
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 5, size=num_samples)

# Convert to DataFrame for convenience
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Target"] = y

# 2. Separate features and target
X = df[["Feature1", "Feature2"]]
y = df["Target"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Define the model and the parameter grid
tree = DecisionTreeRegressor(random_state=42)
param_grid = {
    "max_depth": [2, 4, 6],
    "min_samples_leaf": [1, 2, 4]
}

# 5. Set up the GridSearchCV
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=3,             # 3-fold cross-validation
    scoring="r2",     # Using R² for scoring
    n_jobs=-1         # Use all available CPU cores

# 6. Fit the grid search on the training set

# 7. Find the best parameters and evaluate the best model on the test set
