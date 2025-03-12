# [Example: Fitting a logistic regression with L1 or L2 regularization on a dataset with many features and observing which features remain significant]
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Generate a synthetic classification dataset with many features
X, y = make_classification(
    n_samples=200,   # number of samples
    n_features=10,   # total number of features
    n_informative=5, # number of informative features
    n_redundant=2,   # number of redundant features
    random_state=42
)

# (Optional) Convert to a pandas DataFrame for easier viewing
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train Logistic Regression with L1 regularization
# Note: solver='saga' supports L1 for multi-class settings and is good for large datasets.
model_l1 = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)
model_l1.fit(X_train, y_train)

# 4. Train Logistic Regression with L2 regularization
model_l2 = LogisticRegression(penalty='l2', solver='saga', max_iter=1000, random_state=42)
model_l2.fit(X_train, y_train)

# 5. Compare coefficients
coeff_l1 = model_l1.coef_[0]  # Coefficients for class 1 (binary classification)
coeff_l2 = model_l2.coef_[0]

# Print coefficients side-by-side
coef_comparison = pd.DataFrame({
    "Feature": feature_names,
    "L1_Coefficient": coeff_l1,
    "L2_Coefficient": coeff_l2
})
print("Coefficient Comparison (L1 vs L2):")
print(coef_comparison)

# 6. Evaluate accuracy on the test set
acc_l1 = model_l1.score(X_test, y_test)
acc_l2 = model_l2.score(X_test, y_test)
print(f"\nTest Accuracy - L1: {acc_l1:.2f}")
print(f"Test Accuracy - L2: {acc_l2:.2f}")

# 7. Observe which features remain significant in L1
# (Non-zero coefficients in L1 typically indicate "significant" features, if heavily regularized.)
non_zero_features = coef_comparison[coef_comparison["L1_Coefficient"] != 0]["Feature"].tolist()
print("\nFeatures with non-zero coefficients under L1:")
print(non_zero_features)
