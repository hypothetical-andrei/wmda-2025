#  [A spam detection model fails when trained on incorrect email labels.]
# Incorrect Labeling in Spam Detection Example
# This script:

# Loads a small dataset of spam and ham (non-spam) emails.
# Introduces incorrect labels (simulating mislabeled training data).
# Trains a Naïve Bayes classifier on both correct and incorrect labels.
# Compares the performance of both models.
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (normally, use a larger dataset)
emails = [
    "Win a free lottery now", "Meeting at 10 AM", "You have won a cash prize",
    "Let's schedule a call", "Exclusive offer just for you", "Please review the attached document",
    "Congratulations, you are selected!", "Reminder: Project deadline tomorrow",
    "Claim your free gift today", "Discounts available only for today",
    "Schedule your doctor appointment", "Invoice for your recent transaction",
    "Limited time sale on electronics", "Join our webinar this weekend",
    "Urgent: Update your account information", "Dinner plans for Friday night"
]

labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0])  # 1 = Spam, 0 = Ham

# Introduce incorrect labels (flip 50% randomly)
np.random.seed(42)
num_flips = int(len(labels) * 0.5)  # Flip 50% of labels
flip_indices = np.random.choice(len(labels), num_flips, replace=False)
incorrect_labels = labels.copy()
incorrect_labels[flip_indices] = 1 - incorrect_labels[flip_indices]  # Flip labels

# Convert text data into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
X_train_wrong, _, y_train_wrong, _ = train_test_split(X, incorrect_labels, test_size=0.2, random_state=42)

# Train Naïve Bayes models on correct and incorrect labels
model_correct = MultinomialNB()
model_correct.fit(X_train, y_train)

model_wrong = MultinomialNB()
model_wrong.fit(X_train_wrong, y_train_wrong)

# Predictions
y_pred_correct = model_correct.predict(X_test)
y_pred_wrong = model_wrong.predict(X_test)

# Evaluation
print("=== Model Trained on Correct Labels ===")
print(classification_report(y_test, y_pred_correct))

print("\n=== Model Trained on Incorrect Labels ===")
print(classification_report(y_test, y_pred_wrong))

# Compare accuracy
accuracy_correct = accuracy_score(y_test, y_pred_correct)
accuracy_wrong = accuracy_score(y_test, y_pred_wrong)
print(f"\nAccuracy with Correct Labels: {accuracy_correct:.2f}")
print(f"Accuracy with Incorrect Labels: {accuracy_wrong:.2f}")
