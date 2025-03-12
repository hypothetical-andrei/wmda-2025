# [Example: Predict the likelihood of a customer making a purchase based on demographic data]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Create a small dataset (age, gender, income, purchased)
data = [
    (25, "Male", 50000, 0),
    (40, "Female", 70000, 1),
    (35, "Female", 60000, 0),
    (50, "Male", 80000, 1),
    (28, "Male", 52000, 0),
    (60, "Female", 100000, 1),
    (45, "Male", 75000, 1),
    (22, "Female", 48000, 0),
    (39, "Female", 68000, 1)
]

df = pd.DataFrame(data, columns=["age", "gender", "income", "purchased"])

# 2. Encode the categorical variable (gender) using one-hot encoding
df_encoded = pd.get_dummies(df, columns=["gender"], drop_first=True)
# Now we have a column "gender_Male" which is 1 if Male, 0 if Female

# 3. Separate features (X) and target (y)
X = df_encoded[["age", "income", "gender_Male"]]
y = df_encoded["purchased"]

# 4. Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 5. Train an SVM classifier
#    We'll use a linear kernel for interpretability, but you can try 'rbf' or others.
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# 6. Make predictions on the test set
y_pred = model.predict(X_test)

# 7. Evaluate performance (simple accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# (Optional) Display coefficients (weights) if using a linear kernel
# The 'coef_' attribute gives the weights of each feature for SVC with a linear kernel
coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
})
print("\nCoefficients (Linear SVM):")
print(coefficients)

# (Optional) Show the intercept
print("\nIntercept (bias):", model.intercept_[0])

# (Optional) Display the full DataFrame with encoded columns
print("\nEncoded Dataset:")
print(df_encoded)
