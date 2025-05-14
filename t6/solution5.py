import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Example sentence tokens
tokens = ["The", "cat", "sat", "on", "mat"]
seq_len = len(tokens)

# Simulated attention scores from self-attention layer
attention_scores = torch.rand(seq_len, seq_len)
attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize over last dim

# Plot attention heatmap
