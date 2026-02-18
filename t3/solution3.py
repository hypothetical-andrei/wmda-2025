## Exercise 3 (10 minutes): Regression Trees
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt

# 1. Create a synthetic dataset with multiple features
np.random.seed(42)
num_samples = 30
X = np.random.rand(num_samples, 3) * 10  # e.g., three numeric features

# Let's define a "true" relationship for the target:
# Target = 2*Feature1 + 0.5*Feature2^2 - 3*Feature3 + noise
true_y = 2 * X[:, 0] + 0.5 * (X[:, 1]**2) - 3 * X[:, 2]
noise = np.random.normal(0, 5, size=num_samples)  # Add some noise
y = true_y + noise

# Convert to a pandas DataFrame for familiarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
df["Target"] = y

# 2. Separate features and target
X = df[["Feature1", "Feature2", "Feature3"]]
y = df["Target"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Create and train the Decision Tree Regressor
#    You can tune hyperparameters like max_depth to control overfitting

# 5. Evaluate on the test set

# Optional: Inspect feature importances
print("Feature importances:", tree_reg.feature_importances_)

# Optional: You could visualize the tree with:
# from sklearn.tree import export_graphviz
# export_graphviz(tree_reg, out_file="tree.dot",
#                 feature_names=["Feature1", "Feature2", "Feature3"],
#                 filled=True, rounded=True)
# Then convert .dot to a .png with graphviz if desired.
fig = plt.figure(figsize=(25,20))
plot_tree(tree_reg)
fig.savefig("decistion_tree.png")
