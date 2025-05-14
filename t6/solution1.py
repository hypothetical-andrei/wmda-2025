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
# 2. Analogy: king - man + woman ≈ ?
# Find closest word (excluding input words)
