from gensim.models import Word2Vec

# 1. Prepare a toy corpus
corpus = [
    ["king", "queen", "man", "woman"],
    ["boy", "girl", "brother", "sister"],
    ["father", "mother", "son", "daughter"],
    ["prince", "princess", "uncle", "aunt"]
]

# 2. Train the Word2Vec model
model = Word2Vec(sentences=corpus, vector_size=10, window=2, min_count=1, workers=1, seed=42)

# 3. Explore learned vectors
print("Vector for 'king':\n", model.wv["king"])

# 4. Find most similar words
similar = model.wv.most_similar("king")
print("\nMost similar words to 'king':")
for word, score in similar:
    print(f"{word}: {score:.2f}")
