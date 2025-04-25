import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Get the current working directory
current_dir = os.getcwd()

# Specify the image file name
image_filename = 'test_images/earth_side.jpg'  # Replace with your actual image file name

# Construct the full path to the image
image_path = os.path.join(current_dir, image_filename)

# Load the image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
"""plt.imshow(img, cmap='gray'), plt.title('Detected Edges')
plt.tight_layout()
plt.show()"""

height, width = img.shape

# Initialize parameters
da = 32  # grid step [pixels]
tr0 = 4  # max difference of intensity inside vacuum homogeneous area
tr1 = 10  # min atmosphere thickness [pixels]

# Process vertical lines
for x in range(da // 2, width, da):
    col = img[:, x].copy()
    # Blur the column
    for _ in range(5):
        col[1:] = (col[1:] + col[:-1]) // 2
        col[:-1] = (col[:-1] + col[1:]) // 2

    # Scan from top to bottom
    y_top = np.argmax(np.abs(np.diff(col)) >= tr0)
    if y_top == 0:
        y_top = height
    print(x)
    y_peak = y_top + np.argmax(np.diff(col[y_top:]) <= 0)
    
    if y_peak - y_top > tr1:
        print(f'Horizon point found at x={x}, y={y_peak}')
