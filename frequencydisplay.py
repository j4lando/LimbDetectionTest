import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a sample grayscale image (replace this with your actual image)
image_path = 'test_images/earth_night.jpg'  # Replace with your actual image file name

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define vertical line positions and initialize an output image
height, width = image.shape

# Initialize parameters
da = 60  # grid step [pixels]
tr0 = 4  # max difference of intensity inside vacuum homogeneous area

# Process vertical lines
line_positions = np.arange(da // 2, width, da)  # X-coordinates of vertical lines
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored lines

# Draw vertical lines and their intensity profiles
for x in line_positions:
    # Draw the vertical line on the image
    cv2.line(output_image, (x, 0), (x, image.shape[0] - 1), (0, 0, 255), 1)  # Red vertical line

    # Compute the intensity profile along the vertical line
    intensity_profile = image[:, x]

    # Find the position of the greatest difference in intensity
    diff = np.abs(np.diff(intensity_profile))  # Compute differences between adjacent pixels
    max_diff_position = np.argmax(np.abs(np.diff(image[:, x])) >= tr0)  # Find the index of the maximum difference

    # Draw a tiny blue line at the position of the greatest difference
    cv2.line(output_image, (x - 5, max_diff_position), (x + 5, max_diff_position), (255, 0, 0), 2)  # Blue line

    # Normalize the intensity profile to fit within the image width
    normalized_profile = (intensity_profile / intensity_profile.max() * 50).astype(int)  # Scale to max width of 50 pixels

    # Draw the intensity profile as a white line to the right of the vertical line
    for y, intensity in enumerate(normalized_profile):
        output_image[y, x + 10:x + 10 + intensity] = (255, 255, 255)  # White line

# Display the result using Matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Vertical Lines with Intensity Profiles and Max Difference Points")
plt.axis("off")
plt.show()
