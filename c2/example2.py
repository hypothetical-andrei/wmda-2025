# [Example: Classify reviews as positive or negative based on word frequency]
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Create a small dataset of movie reviews (text) and labels (1 = positive, 0 = negative)
data = [
    ("I absolutely loved this movie, it was fantastic!", 1),
    ("Horrible plot and terrible acting, wasted my time.", 0),
    ("An instant classic, superb in every aspect!", 1),
    ("I wouldn't recommend this film to anyone.", 0),
    ("It was just okay, nothing special or groundbreaking.", 0),
    ("Brilliant! I enjoyed every minute of it!", 1)
]
df = pd.DataFrame(data, columns=["text", "label"])

# 2. Convert text reviews into a bag-of-words feature matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Train a Naïve Bayes classifier (MultinomialNB is suitable for text data)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Predict on the test set
y_pred = model.predict(X_test)

# 6. Evaluate accuracy
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.2f}")

# Optional: Print predicted vs. actual labels for a quick check
comparison = pd.DataFrame({
    "Review": df["text"].iloc[y_test.index],
    "Actual Label": y_test,
    "Predicted Label": y_pred
})
print("\nPredictions vs. Actual:")
print(comparison)
