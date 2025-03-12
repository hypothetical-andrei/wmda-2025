import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Optional: convert to DataFrame for exploration
# df = pd.DataFrame(X, columns=wine.feature_names)
# df['target'] = y
# print(df.head())

# 2. Split into training (80%) and testing (20%) sets

# 3. Train a Decision Tree Classifier
#    max_depth=3 to control overfitting a bit


# 4. Check accuracy on the test set


# (Optional) Visualize the tree structure


# (Optional) Feature importances

