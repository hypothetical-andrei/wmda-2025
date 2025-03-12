# [Example: Show confusion matrix for a binary classification problem and calculate precision, recall, and accuracy]
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# 1. Generate a synthetic binary classification dataset
X, y = make_classification(
    n_samples=100,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# 2. Split the dataset into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict on the test set
y_pred = model.predict(X_test)

# 5. Compute the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)

# 6. Calculate precision, recall, and accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPrecision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"Accuracy:  {accuracy:.2f}")
