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
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

max_len = max(len(seq) for seq in sequences)
x_data = pad_sequences(sequences, maxlen=max_len, padding='post')
y_data = np.array(labels)

# 3. Build LSTM model
model = Sequential([
    Embedding(input_dim=len(word_index)+1, output_dim=8, input_length=max_len),
    LSTM(16),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Train and evaluate
model.fit(x_data, y_data, epochs=20, verbose=1)
loss, acc = model.evaluate(x_data, y_data)
print(f"\nTraining accuracy: {acc:.2f}")
