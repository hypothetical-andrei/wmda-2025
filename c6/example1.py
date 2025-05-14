# [Example: Show a small set of vectors for words "king", "queen", "man", "woman", demonstrating distances and analogies]
import numpy as np
from scipy.spatial.distance import cosine

# Define simple 3D word embeddings for demonstration
word_vectors = {
    "king": np.array([0.8, 0.3, 0.7]),
    "queen": np.array([0.7, 0.4, 0.7]),
    "man": np.array([0.9, 0.2, 0.6]),
    "woman": np.array([0.7, 0.3, 0.6])
}

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Compute similarities
print("Cosine Similarities:")
for word1 in word_vectors:
    for word2 in word_vectors:
        if word1 != word2:
            sim = cosine_similarity(word_vectors[word1], word_vectors[word2])
            print(f"Similarity({word1}, {word2}) = {sim:.2f}")

# Word analogy: King - Man + Woman ≈ ?
def find_closest_word(target_vector, word_vectors, exclude=[]):
    closest_word = None
    max_similarity = -1
    for word, vector in word_vectors.items():
        if word not in exclude:
            sim = cosine_similarity(target_vector, vector)
            if sim > max_similarity:
                max_similarity = sim
                closest_word = word
    return closest_word

analogy_vector = word_vectors["king"] - word_vectors["man"] + word_vectors["woman"]
closest_match = find_closest_word(analogy_vector, word_vectors, exclude=["king", "man", "woman"])
print(f"Analogy: King - Man + Woman ≈ {closest_match}")
