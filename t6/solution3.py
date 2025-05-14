from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# 1. Small sentiment dataset
texts = [
    "I love this movie",
    "This is a great film",
    "Fantastic experience",
    "I hate this movie",
    "This is a bad film",
    "Terrible experience"
]
labels = [1, 1, 1, 0, 0, 0]  # 1 = positive, 0 = negative

# 2. Tokenization

# 3. Build LSTM model

# 4. Train and evaluate
