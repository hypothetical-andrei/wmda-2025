from gensim.models import Word2Vec

# 1. Prepare a toy corpus
corpus = [
    ["king", "queen", "man", "woman"],
    ["boy", "girl", "brother", "sister"],
    ["father", "mother", "son", "daughter"],
    ["prince", "princess", "uncle", "aunt"]
]

# 2. Train the Word2Vec model

# 3. Explore learned vectors
print("Vector for 'king':\n", model.wv["king"])

# 4. Find most similar words

