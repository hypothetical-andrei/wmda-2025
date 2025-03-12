# [Example: Show a small dataset with features like age, income, and a binary label indicating “purchased” or “not purchased.”]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Create a small dataset
data = {
    'age': [25, 40, 35, 50, 28, 60, 45],
    'income': [50000, 70000, 60000, 80000, 52000, 100000, 75000],
    'purchased': [0, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# 2. Separate features (X) and target (y)
X = df[['age', 'income']]
y = df['purchased']

# 3. Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the performance (simple accuracy)
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.2f}")

# Optional: Display the small dataset
print("\nDataset:")
print(df)
