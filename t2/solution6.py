import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# 2. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Define the parameter grid for Logistic Regression
#    - 'C' is the inverse of regularization strength (smaller => stronger regularization)
#    - 'penalty' controls L1 vs. L2 regularization
#    - 'solver' must support the selected penalty; 'saga' works for both l1 and l2.

# 4. Set up the GridSearchCV with 5-fold cross-validation
#    n_jobs=-1 uses all CPU cores to speed up the search

# 5. Fit the grid search on the training data

# 6. Retrieve the best parameters and the corresponding score
print("Best Parameters found by grid search:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.3f}")

# 7. Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Best Model: {test_accuracy:.3f}")
