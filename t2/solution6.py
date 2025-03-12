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


# 3. Define the parameter grid for Logistic Regression
#    - 'C' is the inverse of regularization strength (smaller => stronger regularization)
#    - 'penalty' controls L1 vs. L2 regularization
#    - 'solver' must support the selected penalty; 'saga' works for both l1 and l2.


# 4. Set up the GridSearchCV with 5-fold cross-validation
#    n_jobs=-1 uses all CPU cores to speed up the search

# 5. Fit the grid search on the training data

# 6. Retrieve the best parameters and the corresponding score


# 7. Evaluate the best model on the test set
