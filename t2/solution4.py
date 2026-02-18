import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the Spambase dataset
#    Make sure spambase.csv is in your current working directory or provide the full path.
df = pd.read_csv("spambase.csv", header=None)

# The Spambase dataset typically has:
# - 57 numeric columns for features
# - 1 numeric column (the 58th) for the label (1 = spam, 0 = not spam)

# 2. Separate the features (X) and the target label (y)
X = df.iloc[:, :-1]   # All rows, all columns except the last
y = df.iloc[:, -1]    # All rows, only the last column

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Train a Naïve Bayes classifier (MultinomialNB is common for spam detection)

# 5. Predict on the test set

# 6. Evaluate the model
