# [Example: Class-imbalanced dataset in healthcare, where false negatives might be very costly]
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 1. Generate a synthetic imbalanced dataset
#    - We'll create a binary classification problem with 90% negatives (healthy) and 10% positives (disease).
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_clusters_per_class=1,
    weights=[0.90, 0.10],  # 90% class_0 (healthy), 10% class_1 (disease)
    random_state=42
)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# 3. Train Logistic Regression with class_weight='balanced'
#    This helps the model pay more attention to the minority class.
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 4. Predict on the test set
y_pred = model.predict(X_test)

# 5. Evaluate performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (precision, recall, F1-score):")
print(classification_report(y_test, y_pred))

# Optional: Compare with no class weighting
model_no_weight = LogisticRegression(random_state=42)
model_no_weight.fit(X_train, y_train)
y_pred_no_weight = model_no_weight.predict(X_test)

print("------- Without Class Weighting -------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_no_weight))
print("\nClassification Report (precision, recall, F1-score):")
print(classification_report(y_test, y_pred_no_weight))
