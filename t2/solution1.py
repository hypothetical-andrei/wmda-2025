import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# 2. Split the dataset into training and testing sets (80/20)

# 3. Train a Naïve Bayes classifier

# 4. Predict on the test set

# 5. Print out the accuracy

# (Optional) Display the confusion matrix for deeper insight
