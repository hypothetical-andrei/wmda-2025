# [Demonstrate CNN filters detecting edges and corners in a simple image]
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image in grayscale
image = cv2.imread('sample_image.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if image is None:
    raise ValueError("Image not found. Please provide a valid image path.")

# Define simple edge detection filters
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])  # Sobel filter for horizontal edges

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])  # Sobel filter for vertical edges

laplacian = np.array([[ 0,  1,  0],
                      [ 1, -4,  1],
                      [ 0,  1,  0]])  # Laplacian filter for corners

# Apply the filters using cv2.filter2D
sobel_x_result = cv2.filter2D(image, -1, sobel_x)
sobel_y_result = cv2.filter2D(image, -1, sobel_y)
laplacian_result = cv2.filter2D(image, -1, laplacian)

# Display results
plt.figure(figsize=(10,5))
plt.subplot(1,4,1), plt.imshow(image, cmap='gray'), plt.title("Original")
plt.subplot(1,4,2), plt.imshow(sobel_x_result, cmap='gray'), plt.title("Sobel X")
plt.subplot(1,4,3), plt.imshow(sobel_y_result, cmap='gray'), plt.title("Sobel Y")
plt.subplot(1,4,4), plt.imshow(laplacian_result, cmap='gray'), plt.title("Laplacian")
plt.tight_layout()
plt.show()
