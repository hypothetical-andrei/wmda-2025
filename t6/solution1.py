import numpy as np
from scipy.spatial.distance import cosine

# Define simple 3D vectors for each word
word_vectors = {
    "king": np.array([0.8, 0.3, 0.7]),
    "queen": np.array([0.7, 0.4, 0.7]),
    "man": np.array([0.9, 0.2, 0.6]),
    "woman": np.array([0.7, 0.3, 0.6])
}

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

# 1. Pairwise Similarities
print("Pairwise Cosine Similarities:")
pairs = [("king", "queen"), ("king", "man"), ("queen", "woman"), ("man", "woman")]
for w1, w2 in pairs:
    sim = cosine_similarity(word_vectors[w1], word_vectors[w2])
    print(f"Similarity({w1}, {w2}) = {sim:.2f}")

# 2. Analogy: king - man + woman ≈ ?
analogy_vector = word_vectors["king"] - word_vectors["man"] + word_vectors["woman"]

# Find closest word (excluding input words)

result = closest_word(analogy_vector, word_vectors, exclude=["king", "man", "woman"])
print(f"\nAnalogy: king - man + woman ≈ {result}")
