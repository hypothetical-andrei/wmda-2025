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
