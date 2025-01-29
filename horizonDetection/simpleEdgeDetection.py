import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read the image in grayscale
#path = 'test_images/BlackMarble2016_00066.png'
path = 'test_images/infared.jpeg'
img = cv2.imread(path, cv2.IMREAD_COLOR)
image = cv2.imread(path, 0 )
threshold = 10


binary_mask = np.where(image < threshold, 255, 0).astype(np.uint8)
# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(binary_mask, (3,3), 0)

# Apply Laplacian of Gaussian
log = cv2.Laplacian(blurred, cv2.CV_64F)

# Perform Canny edge detection
canny = cv2.Canny(binary_mask, 100, 200)

# Find zero crossings
zero_crossings = np.zeros_like(log)
zero_crossings[:-1, :-1] = np.logical_and(log[:-1, :-1] < 0, log[1:, 1:] > 0) | \
                           np.logical_and(log[:-1, :-1] > 0, log[1:, 1:] < 0)

# Apply thresholding to get strong edges
threshold = 0.1 * np.max(np.abs(log))
edges = np.zeros_like(zero_crossings, dtype=np.uint8)
edges[np.abs(log) > threshold] = 255


plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(canny, cmap='gray'), plt.title('Canny')
plt.subplot(133), plt.imshow(edges, cmap='gray'), plt.title('Detected Edges')
plt.tight_layout()
plt.show()
