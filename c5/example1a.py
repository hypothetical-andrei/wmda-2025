# [An example showing equivalent code for defining a neural net in both TensorFlow and PyTorch]
import tensorflow as tf
import numpy as np

# Convert to NumPy arrays
X = np.array([
    [0.1, 0.2],
    [0.4, 0.3],
    [0.6, 0.8],
    [0.9, 0.5]
])

y = np.array([1, 0, 1, 0])

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X, y, epochs=10)
