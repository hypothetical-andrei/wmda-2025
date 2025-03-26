# [An example illustrating how to read a dendrogram for a dataset of images grouped by visual similarity]
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Generate synthetic feature vectors
#    For demonstration, imagine each row is a feature vector extracted from an image.
#    In a real project, replace this random data with actual image descriptors.
np.random.seed(42)
num_images = 10
num_features = 64  # e.g., a flattened 8x8 grayscale image
image_features = np.random.rand(num_images, num_features)

# 2. Perform hierarchical clustering (linkage)
#    "ward" attempts to minimize the variance within clusters;
#    other methods: "single", "complete", "average", etc.
Z = linkage(image_features, method='ward')

# 3. Plot the dendrogram
plt.figure(figsize=(8, 6))
dendrogram(Z, labels=[f"Img_{i}" for i in range(num_images)])
plt.title("Dendrogram of Synthetic Image Feature Vectors")
plt.xlabel("Images")
plt.ylabel("Distance (Ward linkage)")
plt.show()

# 4. How to interpret the dendrogram:
#    - Each leaf corresponds to an image.
#    - The branches show how clusters merge at increasing distances.
#    - A horizontal 'cut' at a certain distance can be used to decide the number of clusters.
