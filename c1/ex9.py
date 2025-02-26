# [Extracting keywords from product reviews to improve sentiment analysis.]
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Create a sample dataset with product reviews
reviews = [
    "This phone has a great camera and amazing battery life.",
    "The laptop performance is very fast and smooth.",
    "Terrible customer service, very disappointing experience.",
    "The product quality is top-notch and highly recommended.",
    "The delivery was late and the packaging was damaged."
]

df = pd.DataFrame({"Review": reviews})

print("Original Product Reviews:\n")
print(df)

# Step 2: Apply TF-IDF Vectorization to extract keywords
vectorizer = TfidfVectorizer(stop_words="english", max_features=5)  # Extract top 5 keywords
tfidf_matrix = vectorizer.fit_transform(df["Review"])

# Step 3: Get feature names (keywords)
keywords = vectorizer.get_feature_names_out()

print("\nExtracted Keywords from Product Reviews:\n")
print(keywords)
