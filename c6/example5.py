#  [Example: Show a diagram highlighting encoder self-attention layers and how they attend to different words in a sentence]
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Simulated attention scores for a sentence (5 words)
attention_scores = torch.rand(5, 5)  # (sequence_length, sequence_length)
attention_weights = F.softmax(attention_scores, dim=-1)

# Example words in a sentence
words = ["The", "cat", "sat", "on", "mat"]

# Visualizing the attention weights
plt.figure(figsize=(6,6))
plt.imshow(attention_weights.detach().numpy(), cmap="Blues")
plt.xticks(ticks=np.arange(len(words)), labels=words)
plt.yticks(ticks=np.arange(len(words)), labels=words)
plt.xlabel("Words Attended To")
plt.ylabel("Query Words")
plt.title("Self-Attention Visualization")
plt.colorbar()
plt.show()
