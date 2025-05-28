import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from gensim.models import Word2Vec

# 1. Simple text and preprocessing
text = (
    "NASA's Perseverance rover has successfully landed on Mars. "
    "It will collect rock samples and search for signs of ancient life. "
    "The mission is expected to provide valuable data. "
    "Scientists are excited about the potential discoveries."
)

# Tokenize words and prepare training data
tokens = re.findall(r'\w+', text.lower())
sentences = [tokens]  # single sentence for Word2Vec format

# Train Word2Vec model
w2v_model = Word2Vec(sentences, vector_size=16, window=2, min_count=1, sg=1, epochs=100)

# Create torch embedding layer from Word2Vec weights
word2idx = {word: i for i, word in enumerate(w2v_model.wv.index_to_key)}
vocab_size = len(word2idx)
embedding_weights = torch.tensor(w2v_model.wv.vectors)

# Convert text to input indices
input_ids = torch.tensor([word2idx[w] for w in tokens])

# Word scoring model using pretrained Word2Vec embeddings
class WordScorer(nn.Module):
    def __init__(self, embedding_weights):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(embedding_weights)
        self.linear = nn.Linear(embedding_weights.shape[1], 1)

    def forward(self, x):
        emb = self.emb(x)  # (seq_len, emb_dim)
        scores = self.linear(emb).squeeze(1)  # (seq_len,)
        return scores

# 2. Score words
model = WordScorer(embedding_weights)
with torch.no_grad():
    scores = model(input_ids)

# 3. Select top-k words (based on score)
top_k = 10
_, top_indices = torch.topk(scores, k=top_k)
top_words = [tokens[i] for i in sorted(top_indices.tolist())]
summary = " ".join(top_words)

# Output
print("Original Text:\n", text)
print("\nExtracted Summary (Top Words):\n", summary)
