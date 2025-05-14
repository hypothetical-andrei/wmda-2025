# [Example: Display a code snippet for training a simple Word2Vec model on a small text corpus]
from gensim.models import Word2Vec

# Sample corpus
sentences = [
    ["king", "queen", "man", "woman"],
    ["prince", "princess", "boy", "girl"],
    ["father", "mother", "son", "daughter"],
    ["brother", "sister", "uncle", "aunt"]
]

# Train a simple Word2Vec model
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1)

# Get vector for a word
print("Vector for 'king':", model.wv["king"])  # Example output

# Find most similar words
similar_words = model.wv.most_similar("king")
print("Most similar words to 'king':", similar_words)
