import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define a simple Seq2Seq model with attention
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_dim)
        # encoder_outputs: (seq_len, batch_size, hidden_dim)

        seq_len, batch_size, hidden_dim = encoder_outputs.size()

        # Repeat hidden across the sequence length
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)

        # Concatenate and compute energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)  # (batch_size, seq_len)

        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)
        return context, attention_weights

# Simulated example encoder hidden states
encoder_outputs = torch.rand(5, 1, 16)  # (sequence_length, batch_size, hidden_dim)
decoder_hidden = torch.rand(1, 16)  # (batch_size, hidden_dim)

# Initialize and compute attention
attention = Attention(hidden_dim=16)
context, attn_weights = attention(decoder_hidden, encoder_outputs)

# Visualize attention weights
plt.figure(figsize=(6,3))
plt.bar(range(len(attn_weights.squeeze())), attn_weights.squeeze().detach().numpy())
plt.xlabel("Input Sequence Index")
plt.ylabel("Attention Weight")
plt.title("Attention Weight Distribution")
plt.show()