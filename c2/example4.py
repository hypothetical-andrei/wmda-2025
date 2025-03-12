# [Example: A decision tree for classifying loan approvals based on features like income, credit score, and debt ratio]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Create a small dataset
# Each record: (income, credit_score, debt_ratio, loan_approved)
data = [
    (50000, 700, 0.30, 1),
    (30000, 600, 0.40, 0),
    (80000, 750, 0.20, 1),
    (40000, 580, 0.50, 0),
    (75000, 720, 0.35, 1),
    (28000, 550, 0.45, 0),
    (90000, 780, 0.15, 1),
    (32000, 600, 0.42, 0),
    (66000, 710, 0.38, 1),
    (25000, 530, 0.50, 0)
]

df = pd.DataFrame(data, columns=["income", "credit_score", "debt_ratio", "loan_approved"])

# 2. Split into features (X) and target (y)
X = df[["income", "credit_score", "debt_ratio"]]
y = df["loan_approved"]

# 3. Create train/test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Train a Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=3, random_state=42)  # limit depth for a simpler tree
model.fit(X_train, y_train)

# 5. Predict on the test set
y_pred = model.predict(X_test)

# 6. Evaluate performance
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.2f}")

# 7. (Optional) Visualize the decision tree
plt.figure(figsize=(8, 6))
plot_tree(model, feature_names=["income", "credit_score", "debt_ratio"], filled=True)
plt.show()
