
import nltk
from nltk.corpus import movie_reviews

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the movie_reviews dataset from NLTK
#    Each review is stored as a list of words in the corpus;
#    categories can be 'pos' or 'neg'.
documents = []
labels = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        # Join tokens to reconstruct the full review text
        review_text = " ".join(movie_reviews.words(fileid))
        documents.append(review_text)
        labels.append(category)

# Convert labels to numeric: 'pos' -> 1, 'neg' -> 0
y = [1 if label == 'pos' else 0 for label in labels]

# 2. Split the dataset into training and testing
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    documents,
    y,
    test_size=0.3,
    random_state=42
)

# 3. Convert text into numerical features (TF-IDF)

# 4. Train a Logistic Regression model

# 5. Make predictions on the test set

# 6. Evaluate performance

# (Optional) Display a few random predictions vs actual labels
random_indices = np.random.choice(len(y_test), size=5, replace=False)
for idx in random_indices:
    print("\nReview:")
    print(X_test_texts[idx][:200] + "...")
    print(f"Predicted Sentiment: {'pos' if y_pred[idx] == 1 else 'neg'} | Actual: {'pos' if y_test[idx] == 1 else 'neg'}")
