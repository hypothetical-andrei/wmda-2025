### **Exercise 4: Feature Engineering for Classification**
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the Titanic dataset
df = sns.load_dataset("titanic")

# Step 2: Create a new feature: family_size = sibsp + parch + 1
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Step 3: Handle missing values in age and embarked
df["age"].fillna(df["age"].median(), inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# Step 4: Encode categorical variables (one-hot encoding)
df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

# Step 5: Select numerical features for scaling
scaler = MinMaxScaler()
df[["age", "fare", "family_size"]] = scaler.fit_transform(df[["age", "fare", "family_size"]])

# Step 6: Display cleaned and transformed dataset
print("Processed Titanic Dataset (First 5 Rows):\n")
print(df[["age", "fare", "family_size", "sex_male", "embarked_Q", "embarked_S"]].head())
