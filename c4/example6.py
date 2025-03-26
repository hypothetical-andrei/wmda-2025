# [An example showing how a photo is compressed by reducing the color palette without major visual loss]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

# 1. Load a sample image (e.g., "china.jpg") and normalize pixel values to [0, 1]
sample_image = load_sample_image("china.jpg")
data = np.array(sample_image, dtype=np.float64) / 255.0  # Shape: (height, width, channels)
h, w, c = data.shape

# 2. Reshape the image to a 2D array of pixels (height * width, channels)
data_reshaped = np.reshape(data, (h * w, c))

# 3. Apply K-Means clustering to reduce the color palette
n_colors = 16  # Number of colors in the reduced palette
kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(data_reshaped)
labels = kmeans.predict(data_reshaped)

# 4. Reconstruct the compressed image using the cluster centroids
compressed_data = kmeans.cluster_centers_[labels]
compressed_data = np.reshape(compressed_data, (h, w, c))

# 5. Plot the original and compressed images side by side
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(data)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Compressed ({n_colors} colors)")
plt.imshow(compressed_data)
plt.axis('off')

plt.tight_layout()
plt.show()
