import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Simple Attention Layer
def compute_attention(query, keys):
    # query: (batch_size, hidden_dim)
    # keys: (batch_size, seq_len, hidden_dim)
    scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
    weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
    context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (batch_size, hidden_dim)
    return context, weights

# Simulated encoder outputs
batch_size = 1
seq_len = 5
hidden_dim = 8

encoder_outputs = torch.rand(batch_size, seq_len, hidden_dim)
decoder_hidden = torch.rand(batch_size, hidden_dim)

# Compute attention weights
context_vector, attn_weights = compute_attention(decoder_hidden, encoder_outputs)

# Visualize attention weights
