import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# (Optional) Convert to a Pandas DataFrame for easier viewing
# df = pd.DataFrame(X, columns=wine.feature_names)
# df['target'] = y
# print(df.head())  # Uncomment to inspect

# 2. Split the data into training (80%) and testing (20%) sets

# 3. Train a Naïve Bayes classifier (from Exercise 1)


# 5. Compare metrics: accuracy, precision, and recall for each model
# Note: Because we have three classes in the Wine dataset, we set average='macro' (or 'weighted') for multi-class


# 6. Print results
for model_name, scores in metrics.items():
    print(f"=== {model_name} ===")
    print(f"Accuracy:  {scores['Accuracy']:.2f}")
    print(f"Precision: {scores['Precision']:.2f}")
    print(f"Recall:    {scores['Recall']:.2f}")
    print()

# Optional: If you’d like to see a confusion matrix for each model
# from sklearn.metrics import confusion_matrix
# print("Naive Bayes Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_nb))
# print("\nLogistic Regression Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_logreg))
