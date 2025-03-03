### **Exercise 5: Applying a Classification Model**
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Titanic dataset
df = sns.load_dataset("titanic")

# Step 2: Feature Engineering (from Exercise 4)
df["family_size"] = df["sibsp"] + df["parch"] + 1  # Create new feature
df["age"].fillna(df["age"].median(), inplace=True)  # Handle missing age values
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)  # Handle missing embarked values
df.dropna(subset=["fare"], inplace=True)  # Remove rows with missing fare values
df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)  # One-hot encoding

# Step 3: Select features and scale numerical columns
features = ["age", "fare", "family_size", "sex_male", "embarked_Q", "embarked_S"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Step 4: Define target variable and split data into training and testing sets
X = df[features]
y = df["survived"]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate model performance
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
